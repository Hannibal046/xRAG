## built-in
import argparse
import logging
import math
import os
import random
import types
import pickle,json
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_IGNORE_GLOBS"]='*.pth' ## not upload ckpt to wandb cloud

## third-party
import datasets
import torch
import torch.distributed as dist
from functools import partial
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import copy
import transformers
from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    LlamaTokenizerFast,
    SchedulerType,
    get_scheduler,
)
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
import deepspeed
from tokenizers import AddedToken
import wandb

## own
from src.model import (
    XMistralForCausalLM,
    XMistralConfig,
    XMixtralForCausalLM,
    XMixtralConfig,
    SFR,
)

from src.language_modeling.utils import (
    get_nll_loss,
    get_kl_loss,
    save_with_accelerate,
    XRAG_TOKEN,
    get_retrieval_embeds,
)

from src.language_modeling.preprocessing import (
    encode_with_chat_format_pretrain,
    encode_with_chat_format_finetune,
)

from src.utils import (
    get_yaml_file,
)

logger = get_logger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exclude_dataset_type",
        help='task type to exclude when doing finetuning',
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--distill_topk",
        type=int,
        help='topk token to distill in the self-distillation part'
    )
    parser.add_argument(
        "--base_model",
        help='base LLM load'
    )
    parser.add_argument(
        "--use_fast_tokenizer",
        type=eval,
    )
    parser.add_argument(
        "--use_rag_tuning",
        type=eval,
        help='whether to use retrieval-augmented instruction tuning'
    )
    parser.add_argument(
        "--chat_format",
        choices=['mistral','tulu','mixtral','qwen','yi','gemma']
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
    )
    parser.add_argument(
        "--update_projector_only",
        type=eval,
    )
    parser.add_argument(
        "--workdir",
        type=str,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="config file to launch the training"
    )
    parser.add_argument(
        "--task_type",
        type=str,
        help="pretrain or finetune"
    )
    parser.add_argument(
        "--retrieval_context_length",
        type=int,
        help="max token number for document encoder in dense retrieval",
    )
    parser.add_argument(
        "--alpha_nll",
        type=float,
        help="coefficient for multi-task learning",
    )
    parser.add_argument(
        "--alpha_kl",
        type=float,
        help="coefficient for multi-task learning",
    )
    parser.add_argument(
        "--kl_temperature",
        type=float,
        help="Temperature coefficient for calculation KL-Divergency loss",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--dev_file", type=str, default=None, help="A csv or a json file containing the dev data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--retriever_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--use_flash_attn",
        type=eval,
        help="If passed, will use flash attention to train the model.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        help="The maximum total sequence length (prompt+completion) of each training example.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--warmup_ratio", type=float, help="Ratio of total training steps used for warmup."
    )
    parser.add_argument("--project_name", type=str, default=None)
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--exp_note", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=eval, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=None,
        help="Log the training loss and learning rate every logging_steps steps.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=eval,
        help=(
            "Turn on gradient checkpointing. Saves memory but slows training."
        ),
    )
    parser.add_argument(
        '--clip_grad_norm',
        type=float,
        help='Clip gradient norm. Not compatible with deepspeed (use deepspeed config instead).',
    )
    
    args = parser.parse_args()
    yaml_config = get_yaml_file(args.config)

    ## priority: CLI > YAML (with all default value set to None in argument parser)
    for k,v in yaml_config.items():
        assert hasattr(args,k), f"{k} not in parsed arguments"
        if getattr(args,k) is None:
            setattr(args,k,v)

    args.train_file = os.path.join(args.workdir,args.train_file)
    if args.dev_file is not None:args.dev_file = os.path.join(args.workdir,args.dev_file)
    if args.retriever_name_or_path is not None and os.path.isdir(args.retriever_name_or_path):
        args.retriever_name_or_path = os.path.join(args.workdir,args.retriever_name_or_path)
    if os.path.isdir(os.path.join(args.workdir,args.model_name_or_path)):
        args.model_name_or_path = os.path.join(args.workdir,args.model_name_or_path)

    return args

def collator(        
        samples,
        llm_tokenizer,
        retriever_tokenizer = None,
        retrieval_context_length = 180,
    ):
    """
    collate tokenized input_ids and labels with left and right side padding supported
    
    Args:
        samples (dict): a dict contains input_ids, labels and maybe retrieval_text
        llm_tokenizer: tokenizer for llm
        retriever_tokenizer: tokenizer for retriever
        retrieval_context_length: max length for the retrieved passages
    
    Returns:
        xrag_input_ids: input_ids with xrag_token_id (xrag_labels,xrag_attention_mask)
        input_ids: input_ids for llm without xrag_token_id, vanilla rag (labels,attention_mask)
        retriever_input_ids: input_ids for retriever (retriever_attention_mask)

    """
    def padding(input_ids,labels=None,padding_side='right'):
        """
        batch padding
        """

        def _padding(ids,padding_value,padding_side='right'):
            if padding_side == 'right':
                return torch.nn.utils.rnn.pad_sequence(ids,batch_first=True,padding_value=padding_value)
            elif padding_side == 'left':
                flipped_ids = [torch.flip(x, dims=[0]) for x in ids]  
                return torch.flip(
                    torch.nn.utils.rnn.pad_sequence(flipped_ids,batch_first=True,padding_value=padding_value),
                    dims=[1],
                )
        input_ids = _padding(input_ids,padding_value=llm_tokenizer.pad_token_id,padding_side=padding_side)
        attention_mask = (input_ids != llm_tokenizer.pad_token_id).long()
        if labels is not None:
            labels = _padding(labels,padding_value=-100,padding_side=padding_side)
        return input_ids,attention_mask,labels

    xrag_input_ids,xrag_attention_mask,xrag_labels = padding(
        input_ids=[x['xrag_input_ids'] for x in samples],
        labels=[x['xrag_labels'] for x in samples] if 'xrag_labels' in samples[0].keys() else None,
        padding_side=llm_tokenizer.padding_side
    )

    ## add some noise to pretraining task TODO

    ret = {
        "xrag_input_ids":xrag_input_ids,
        "xrag_attention_mask":xrag_attention_mask,
        "xrag_labels":xrag_labels,
    }

    if 'retriever_input_text' in samples[0].keys():
        retriever_input_text = [x['retriever_input_text'] for x in samples]
        assert isinstance(retriever_input_text[0],list)
        retriever_input_text = [x for y in retriever_input_text for x in y]
        ## handling different retriever tokenization problem
        if retriever_tokenizer.name_or_path == "intfloat/e5-large-v2":
            retriever_input_text = ["passage: "+x for x in retriever_input_text]
        elif retriever_tokenizer.name_or_path == 'intfloat/e5-mistral-7b-instruct':
            retriever_input_text = [x + retriever_tokenizer.eos_token for x in retriever_input_text]

        tokenized_retrieval_text = retriever_tokenizer(
            retriever_input_text, 
            max_length=retrieval_context_length,
            padding=True, truncation=True, return_tensors="pt"
        )
        
        ret['retriever_input_ids']      = tokenized_retrieval_text['input_ids']
        ret['retriever_attention_mask'] = tokenized_retrieval_text['attention_mask']
    
    if 'input_ids' in samples[0].keys():
        input_ids = [x['input_ids'] for x in samples]
        labels =    [x['labels'] for x in samples]
     
        input_ids,attention_mask,labels = padding(input_ids,labels,padding_side=llm_tokenizer.padding_side)
        
        ret['input_ids'] = input_ids
        ret['attention_mask'] = attention_mask
        ret['labels'] = labels

    return ret


@torch.no_grad()
def validate_during_pretrain(model,dataloader,accelerator,vocab_size,retriever):
    model.eval()
    total_loss = []
    for batch in dataloader:
        retrieval_embeds = get_retrieval_embeds(
                model = retriever,
                input_ids = batch['retriever_input_ids'],
                attention_mask = batch['retriever_attention_mask'],
        )
        outputs = model(
            input_ids = batch['xrag_input_ids'],
            attention_mask = batch['xrag_attention_mask'],
            retrieval_embeds = retrieval_embeds,
        )
        nll_loss = get_nll_loss(
            labels = batch['xrag_labels'],
            logits = outputs.logits,
            vocab_size = vocab_size,
        )
        total_loss.append(nll_loss.item())
    model.train()
    if accelerator.use_distributed and accelerator.num_processes>1:
        all_ranks_objects = [None for _ in range(accelerator.num_processes)]
        dist.all_gather_object(all_ranks_objects,total_loss)
        total_loss = [x for y in all_ranks_objects for x in y]
    ppl = torch.exp(torch.tensor(sum(total_loss)/len(total_loss)))
    return ppl

def main():
    args = parse_args()
    set_seed(args.seed)
    ## we need to load retriever before accelerator init
    retriever = None
    retriever_hidden_size = -1
    retrieval_embed_length = 0 ## deprecated since ColBERT is not concluded
    retriever_tokenizer = None
    if args.retriever_name_or_path is not None:
        if args.retriever_name_or_path.lower() == 'salesforce/sfr-embedding-mistral':
            retriever = SFR.from_pretrained(args.retriever_name_or_path,torch_dtype = torch.bfloat16)
            retriever_tokenizer = AutoTokenizer.from_pretrained(args.retriever_name_or_path)
        retrieval_embed_length = retriever.get_embed_length()
        retriever_hidden_size = retriever.get_embed_dim()
        retriever.eval()

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, log_with="wandb")
    accelerator.init_trackers(
        project_name=args.project_name, 
        config=args,
        init_kwargs={
            "wandb": {
                "dir": args.workdir, 
                "name": args.exp_name if args.exp_name is not None else None,
                "notes": args.exp_note if args.exp_note is not None else None,
                "save_code": True,
            },
        }
    )
    accelerator.print(json.dumps(vars(args),indent=4))
    checkpoint_dir = [None]
    if accelerator.is_local_main_process:
        wandb_tracker = accelerator.get_tracker("wandb")
        checkpoint_dir = [os.path.join(wandb_tracker.run.dir,'checkpoint')]
    if accelerator.use_distributed:dist.broadcast_object_list(checkpoint_dir,src=0)
    args.output_dir = checkpoint_dir[0]

    if retriever is not None:
        retriever = retriever.to(accelerator.device)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=True)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    
    accelerator.wait_for_everyone()

    data_files = {}
    dataset_args = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.dev_file is not None:
        data_files['dev'] = args.dev_file
    raw_datasets = load_dataset(
        "json",
        data_files=data_files,
        **dataset_args,
    )

    ## select N samples, mainly for debug
    if args.max_train_samples is not None and len(raw_datasets['train']) > args.max_train_samples:
        selected_indices = random.sample(range(len(raw_datasets['train'])),args.max_train_samples)
        raw_datasets['train'] = raw_datasets['train'].select(selected_indices)
    
    if args.exclude_dataset_type is not None:
        for d_type in args.exclude_dataset_type:
            raw_datasets['train'] = raw_datasets['train'].filter(lambda  example:example['task_type']!=d_type)
    

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=args.use_fast_tokenizer,
    )

    if args.chat_format == 'mixtral':
        MODEL_CLASS,CONFIG_CLASS = XMixtralForCausalLM,XMixtralConfig
        tokenizer.padding_side = 'left'    
    if args.chat_format == 'mistral':
        MODEL_CLASS,CONFIG_CLASS = XMistralForCausalLM,XMistralConfig
        tokenizer.padding_side = 'left'
    config = CONFIG_CLASS.from_pretrained(args.model_name_or_path,retriever_hidden_size=retriever_hidden_size)
    model = MODEL_CLASS.from_pretrained(
        args.model_name_or_path,
        config=config,
        use_flash_attention_2=args.use_flash_attn,
        torch_dtype = torch.bfloat16 if accelerator.mixed_precision == 'bf16' else 'auto',
    )

    num_added_tokens = 0
    ## mistral tokenizer is also a LLamaTokenizer
    if isinstance(tokenizer, LlamaTokenizer) or isinstance(tokenizer, LlamaTokenizerFast):
        num_added_tokens = tokenizer.add_special_tokens({
            "pad_token": "<pad>",
        })
        assert num_added_tokens in [0, 1], "LlamaTokenizer should only add one special token - the pad_token, or no tokens if pad token present."


    ## XRAG_TOKEN simply functions as a placeholder, would not be trained
    num_added_tokens += tokenizer.add_tokens([AddedToken(XRAG_TOKEN,lstrip=False,rstrip=False)])
    xrag_token_id = tokenizer.convert_tokens_to_ids(XRAG_TOKEN)
    model.set_xrag_token_id(xrag_token_id)
    if num_added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
    vocab_size = len(tokenizer)

    # Preprocessing the datasets.
    if args.task_type == 'finetune':
        encode_function = partial(
            encode_with_chat_format_finetune, # if "messages" in raw_datasets["train"].column_names else encode_with_completion_format_finetune,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            retrieval_embed_length=retrieval_embed_length,
            use_rag_tuning = args.use_rag_tuning,
            use_retriever_embed = not (retriever is None),
            retriever_tokenizer = retriever_tokenizer,
            chat_format = args.chat_format,
        )
    elif args.task_type == 'pretrain':
        encode_function = partial(
            encode_with_chat_format_pretrain,
            tokenizer = tokenizer,
            max_seq_length = args.max_seq_length,
            retrieval_embed_length=retrieval_embed_length,
            chat_format = args.chat_format,
        )
    with accelerator.main_process_first():
        lm_datasets = raw_datasets.map(
            encode_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            remove_columns=[name for name in raw_datasets["train"].column_names if name not in ["input_ids", "labels", "attention_mask"]],
            desc=f"Tokenizing and reformatting data on rank: {accelerator.local_process_index}",
        )
        lm_datasets.set_format(type="pt")
        if args.task_type == 'finetune':
            lm_datasets['train'] = lm_datasets['train'].filter(lambda example: (example['labels'] != -100).any())
            if args.alpha_kl is not None and args.alpha_kl > 0.0:
                lm_datasets['train'] = lm_datasets['train'].filter(
                    lambda example: 
                    (example['labels']!=-100).sum() == (example['xrag_labels']!=-100).sum()
                )

    train_dataset = lm_datasets["train"]
    dev_dataset = lm_datasets['dev'] if args.dev_file is not None else None


    collate_fn = partial(
        collator,
        llm_tokenizer=tokenizer,
        retriever_tokenizer=retriever_tokenizer,
        retrieval_context_length=args.retrieval_context_length,
    )

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        collate_fn=collate_fn,
        batch_size=args.per_device_train_batch_size
    )

    dev_dataloader = None
    if dev_dataset is not None:
        dev_dataloader = DataLoader(
            dev_dataset,
            shuffle=False, 
            collate_fn=collate_fn,
            batch_size=args.per_device_train_batch_size
        )
    
    if args.update_projector_only:
        for n,p in model.named_parameters():
            if 'projector' not in n:p.requires_grad = False
            else:p.requires_grad = True
                
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=args.learning_rate)
    else:
        no_decay = ["bias", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Create the learning rate scheduler.
    # Note: the current accelerator.step() calls the .step() of the real scheduler for the `num_processes` times. This is because they assume 
    # the user initialize the scheduler with the entire training set. In the case of data parallel training, each process only
    # sees a subset (1/num_processes) of the training set. So each time the process needs to update the lr multiple times so that the total 
    # number of updates in the end matches the num_training_steps here.
    # Here we need to set the num_training_steps to either using the entire training set (when epochs is specified) or we need to multiply the 
    # num_training_steps by num_processes so that the total number of updates matches the num_training_steps.
    num_training_steps_for_scheduler = args.max_train_steps if overrode_max_train_steps else args.max_train_steps * accelerator.num_processes
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=int(num_training_steps_for_scheduler * args.warmup_ratio),
    )
    
    # # https://github.com/microsoft/DeepSpeed/pull/4966
    # if args.chat_format == 'mixtral':
    #     deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])

    # Prepare everything with `accelerator`.
    if dev_dataset is not None:
        model, optimizer, train_dataloader, lr_scheduler, dev_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler, dev_dataloader)

    else:
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler)


    

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Max Sequence Length = {args.max_seq_length}")
    logger.info(f"  Trainable Parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)/(10**6):.2f} M") ## not applicable for deepspeed

    completed_steps = 0
    starting_epoch = 0

    # logging_interval_grad_norm = 0
    logging_interval_loss = 0
    logging_interval_kl_loss = 0
    logging_interval_nll_loss = 0
    
    total_loss = 0
    total_kl_loss = 0
    total_nll_loss = 0

    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    # progress_bar = tqdm(range(args.max_train_steps), disable=True)

    # update the progress_bar if load from checkpoint
    save_one_sample = True
    
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        active_dataloader = train_dataloader

        for batch in active_dataloader:
            if save_one_sample:
                if accelerator.is_local_main_process:
                    pickle.dump(
                        batch,
                        open(os.path.join(os.path.dirname(args.output_dir),"sample_data.pkl"),'wb'),
                    )
                accelerator.print("**"*20,"show one example","**"*20)
                accelerator.print(batch.keys())
                accelerator.print(tokenizer.decode(batch['xrag_input_ids'][0]))
                accelerator.print(batch['xrag_input_ids'][0])
                if "retriever_input_text" in batch:
                    accelerator.print(batch['retriever_input_text'][0])
                if 'input_ids' in batch:
                    for input_id,label_id,attention_mask in zip(batch['input_ids'][0],batch['labels'][0],batch['attention_mask'][0]):
                        accelerator.print(f"{tokenizer.convert_ids_to_tokens([input_id])[0]}({label_id.item()})({attention_mask})",end=" ")
                accelerator.print()    
                for input_id,label_id,attention_mask in zip(batch['xrag_input_ids'][0],batch['xrag_labels'][0],batch['xrag_attention_mask'][0]):
                    accelerator.print(f"{tokenizer.convert_ids_to_tokens([input_id])[0]}({label_id.item()})({attention_mask})",end=" ")
                accelerator.print('\n'+"**"*20,"show one example","**"*20)
                save_one_sample=False

            with accelerator.accumulate(model):
                ## forward with retrieval embeds
                retrieval_kwargs = {}
                if retriever is not None:
                    retrieval_kwargs['retrieval_embeds'] = get_retrieval_embeds(
                        model = retriever,
                        input_ids = batch['retriever_input_ids'],
                        attention_mask = batch['retriever_attention_mask'],
                    )

                outputs = model(
                    input_ids = batch['xrag_input_ids'],
                    attention_mask = batch['xrag_attention_mask'],
                    **retrieval_kwargs,
                )
                loss = None
                if args.alpha_nll is not None and args.alpha_nll > 0.0:
                    
                    nll_loss = get_nll_loss(
                        labels = batch['xrag_labels'],
                        logits = outputs.logits,
                        vocab_size = vocab_size,
                    )

                    logging_interval_nll_loss += nll_loss.detach().float()

                    loss = args.alpha_nll * nll_loss

                if args.alpha_kl is not None and args.alpha_kl > 0.0:
                    
                    ## forward with retrieval tokens
                    with torch.no_grad():
                        model.eval()
                        teacher_outputs = model(
                            input_ids = batch['input_ids'],
                            attention_mask = batch['attention_mask'],
                        )
                        model.train()

                    kl_loss = get_kl_loss(
                        teacher_logits=teacher_outputs.logits,
                        teacher_labels=batch['labels'],
                        student_logits=outputs.logits,
                        student_labels=batch['xrag_labels'],
                        temperature=args.kl_temperature,
                        distill_topk=args.distill_topk,
                    )
                    logging_interval_kl_loss += kl_loss.detach().float()
                    if loss is not None:
                        loss += args.alpha_kl * kl_loss
                    else:
                        loss = args.alpha_kl * kl_loss

                logging_interval_loss += loss.detach().float()
                accelerator.backward(loss)
                if accelerator.sync_gradients and args.clip_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()       

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                if args.logging_steps and completed_steps % args.logging_steps == 0:
                    avg_loss = accelerator.gather(logging_interval_loss).mean().item() / args.gradient_accumulation_steps / args.logging_steps
                    total_loss += accelerator.gather(logging_interval_loss).mean().item() / args.gradient_accumulation_steps 

                    to_be_logged = {
                        "learning_rate": lr_scheduler.get_last_lr()[0],
                        "train_loss": avg_loss,
                        "rolling_loss":total_loss / completed_steps,
                    }
                    if args.alpha_nll is not None and args.alpha_nll > 0.0:
                        total_nll_loss += accelerator.gather(logging_interval_nll_loss).mean().item() / args.gradient_accumulation_steps
                        to_be_logged["rolling_nll_loss"] = total_nll_loss  / completed_steps

                    if args.alpha_kl is not None and args.alpha_kl > 0.0:
                        total_kl_loss  += accelerator.gather(logging_interval_kl_loss ).mean().item() / args.gradient_accumulation_steps
                        to_be_logged["rolling_kl_loss"] = total_kl_loss  / completed_steps

                    accelerator.log(to_be_logged,step=completed_steps)
                    
                    # logging_interval_grad_norm = 0
                    logging_interval_loss = 0
                    logging_interval_kl_loss = 0
                    logging_interval_nll_loss = 0
                    
                if isinstance(checkpointing_steps, int):
                    if completed_steps % checkpointing_steps == 0:
                        output_dir = os.path.join(args.output_dir, f"step_{completed_steps}")
                        save_with_accelerate(accelerator, model, tokenizer, output_dir,save_projector_only=args.update_projector_only)

                        if dev_dataloader is not None:
                            if args.task_type == 'pretrain':
                                ppl = validate_during_pretrain(model,dev_dataloader,accelerator,vocab_size,retriever)
                                accelerator.log({"dev_ppl":ppl},step=completed_steps)

                if completed_steps >= args.max_train_steps:
                    break

        if args.checkpointing_steps == "epoch":
            output_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
            save_with_accelerate(accelerator, model, tokenizer, output_dir,save_projector_only=args.update_projector_only)

    accelerator.end_training()

    ## save the last one
    output_dir = os.path.join(args.output_dir,"last")
    save_with_accelerate(accelerator, model, tokenizer, output_dir,save_projector_only=False)

if __name__ == "__main__":
    main()
