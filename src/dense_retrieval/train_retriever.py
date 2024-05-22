## built-in
import math,logging,functools,os
import types
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_IGNORE_GLOBS"]='*.bin' ## not upload ckpt to wandb cloud

## third-party
from accelerate import Accelerator
from accelerate.logging import get_logger
import transformers
transformers.logging.set_verbosity_error()
import torch
import torch.nn as nn
import torch.distributed as dist
from tqdm import tqdm
import numpy as np

## own
from src.model import (
    ColBERT,ColBERTConfig,
    PolBERT,PolBERTConfig,
    DPR,DPRConfig,
    RetrieverTokenizer,
)
from src.utils import (
    get_mrr,
    get_recall,
    set_seed,
    get_yaml_file,
)

logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    ## adding args here for more control from CLI is possible
    parser.add_argument("--config_file",default='config/colbert_msmarco.yaml')
    parser.add_argument("--torch_compile",type=eval)
    parser.add_argument("--lr",type=float)
    parser.add_argument("--poly_m",type=int)
    parser.add_argument("--mask_punctuation",type=eval)
    parser.add_argument("--poly_dropout",type=float)
    parser.add_argument("--poly_num_heads",type=int)
    parser.add_argument("--pooling_type")
    parser.add_argument("--query_pooling",type=eval)
    parser.add_argument("--use_mask_in_pooling",type=eval)
    parser.add_argument("--similarity_metric")
    parser.add_argument("--max_train_steps",type=int)
    parser.add_argument("--fp16",type=eval)
    parser.add_argument("--logging",type=eval,default=True)
    parser.add_argument("--experiment_name")
    parser.add_argument("--project_name")
    parser.add_argument("--dim",type=int)


    args = parser.parse_args()

    yaml_config = get_yaml_file(args.config_file)
    args_dict = {k:v for k,v in vars(args).items() if v is not None}
    yaml_config.update(args_dict)
    args = types.SimpleNamespace(**yaml_config)
    return args

def validate(model,dataloader,accelerator):
    model.eval()

    qid2ranking = {}
    qid2positives = {}

    for samples in dataloader:
        num_passages = samples['doc_input_ids'].shape[0]
        qid = samples['qids'][0]
        positives = samples['positives'][0]
        pids = samples['pids'].squeeze(0)
        
        assert qid not in qid2positives
        qid2positives[qid] = positives

        with torch.no_grad(), accelerator.autocast():
            query_embedding = model.get_query_embedding(
                input_ids = samples['query_input_ids'],
                attention_mask = samples['query_attention_mask'],
            )
            doc_embedding = model.get_doc_embedding(
                input_ids = samples['doc_input_ids'],
                attention_mask = samples['doc_attention_mask'],
            )
            scores = model.get_matching_score(
                query_embedding = query_embedding.expand(num_passages,-1,-1) if query_embedding.ndim==3 else query_embedding,
                doc_embedding = doc_embedding,
            )
            
        scores = scores.squeeze(0)
        _, indices = scores.sort(descending=True)
        qid2ranking[qid] = pids[indices].tolist()
    
    if accelerator.use_distributed and accelerator.num_processes>1:
        all_ranks = [None for _ in range(accelerator.num_processes)]
        dist.all_gather_object(all_ranks,qid2ranking)
        qid2ranking = {}
        for one_rank in all_ranks:
            for k,v in one_rank.items():
                assert k not in qid2ranking
                qid2ranking[k] = v
        
        all_ranks = [None for _ in range(accelerator.num_processes)]
        dist.all_gather_object(all_ranks,qid2positives)
        qid2positives = {}
        for one_rank in all_ranks:
            for k,v in one_rank.items():
                assert k not in qid2positives
                qid2positives[k] = v
    
    mrrAT10 = get_mrr(qid2ranking,qid2positives,cutoff_rank=10)['mrr@10']
    
    return mrrAT10

class ValidationDataset(torch.utils.data.Dataset):
    def __init__(self,top1000_path,qrels_path,max_test_samples):
        to_be_tested = {}
        with open(top1000_path) as f:
            for line in f:
                qid,pid,query,passage = line.split("\t")
                qid,pid = int(qid),int(pid)
                if qid not in to_be_tested:
                    sample = {"query":query,"pid":[],"passage":[],'positives':[]}
                else:
                    sample = to_be_tested[qid]
                # assert sample['query'] == query
                sample['pid'].append(pid)
                sample['passage'].append(passage)

                to_be_tested[qid] = sample
        
        with open(qrels_path) as f:
            for line in f:
                qid,_,pid,_ = [int(x) for x in line.strip().split("\t")]
                to_be_tested[qid]['positives'].append(pid)

        self.data = [{"qid":qid,**values} for qid,values in to_be_tested.items()][:max_test_samples]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self,index):
        return self.data[index]

    @staticmethod
    def collate_fn(samples,tokenizer,query_max_len,doc_max_len):
        qids      = [sample["qid"] for sample in samples]
        queries   = [sample['query'] for sample in samples]
        pids      = [sample['pid'] for sample in samples]
        passages  = [passage for sample in samples for passage in sample['passage']]
        positives = [sample['positives'] for sample in samples]

        tokenized_query = tokenizer.tokenize_query(queries,max_length=query_max_len)
        tokenized_passages = tokenizer.tokenize_document(passages,max_length=doc_max_len)

        return {
            "qids":qids,
            "pids":torch.tensor(pids),
            "positives":positives,
            "query_input_ids":tokenized_query["input_ids"],
            "query_attention_mask":tokenized_query['attention_mask'],
            "doc_input_ids":tokenized_passages['input_ids'],
            "doc_attention_mask":tokenized_passages['attention_mask'],
        }

class MSMarcoDataset(torch.utils.data.Dataset):
    def __init__(self,query_data_path,pos_doc_data_path,neg_doc_data_path,
                 query_max_len,doc_max_len,num_samples,
                 ):
        self.queries  = np.memmap(query_data_path,  dtype=np.int16, mode='r', shape=(num_samples,query_max_len))  
        self.pos_docs = np.memmap(pos_doc_data_path,dtype=np.int16, mode='r', shape=(num_samples,doc_max_len))
        self.neg_docs = np.memmap(neg_doc_data_path,dtype=np.int16, mode='r', shape=(num_samples,doc_max_len))
        self.num_samples = num_samples  
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self,idx):
        return (self.queries[idx],self.pos_docs[idx],self.neg_docs[idx])


    @staticmethod
    def collate_fn(samples,tokenizer):

        def trim_padding(input_ids,padding_id):
            ## because we padding it to make length in the preprocess script
            ## we need to trim the padded sequences in a 2-dimensional tensor to the length of the longest non-padded sequence
            non_pad_mask = input_ids != padding_id
            non_pad_lengths = non_pad_mask.sum(dim=1)
            max_length = non_pad_lengths.max().item()
            trimmed_tensor = input_ids[:,:max_length]
            return trimmed_tensor

        queries  = [x[0] for x in samples]
        pos_docs = [x[1] for x in samples]
        neg_docs = [x[2] for x in samples]

        query_input_ids = torch.from_numpy(np.stack(queries).astype(np.int32))
        query_attention_mask = (query_input_ids != tokenizer.mask_token_id).int() ## not pad token, called *query augmentation* in the paper

        doc_input_ids = torch.from_numpy(np.stack(pos_docs+neg_docs).astype(np.int32))
        doc_input_ids = trim_padding(doc_input_ids,padding_id = tokenizer.pad_token_id)
        doc_attetion_mask = (doc_input_ids != tokenizer.pad_token_id).int()


        return {
            'query_input_ids':query_input_ids,
            'query_attention_mask':query_attention_mask,

            "doc_input_ids":doc_input_ids,
            "doc_attention_mask":doc_attetion_mask,
        }

def main():
    args = parse_args()
    set_seed(args.seed)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with='wandb' if args.logging else None,
        mixed_precision='fp16' if args.fp16 else 'no',
    )

    accelerator.init_trackers(
        project_name=args.project_name, 
        config=args,
        init_kwargs={"wandb": {"dir": ".", "settings":{"console": "off"},"name":args.experiment_name}}
    )
    if accelerator.is_local_main_process:
        if args.logging:
            wandb_tracker = accelerator.get_tracker("wandb")
            LOG_DIR = wandb_tracker.run.dir


    tokenizer = RetrieverTokenizer.from_pretrained(args.base_model,additional_special_tokens=["[Q]","[D]"])
    if args.model_type == 'colbert':
        config = ColBERTConfig(
            dim = args.dim,
            similarity_metric = args.similarity_metric,
            mask_punctuation = args.mask_punctuation, 
        )
        model = ColBERT.from_pretrained(
            args.base_model,
            config = config,
            _fast_init=False,
        )
    elif args.model_type == 'polbert':
        config = PolBERTConfig(
            dim = args.dim,
            similarity_metric = args.similarity_metric,
            poly_m = args.poly_m,
            poly_dropout=args.poly_dropout,
            poly_num_heads=args.poly_num_heads,
            pooling_type = args.pooling_type,
            use_mask_in_pooling=args.use_mask_in_pooling,
            query_pooling=args.query_pooling,
            query_max_len=args.query_max_len,
            doc_max_len=args.doc_max_len,
        )
        model = PolBERT.from_pretrained(
            args.base_model,
            config = config,
            _fast_init=False
        )
    elif args.model_type == 'dpr':
        config = DPRConfig()
        model = DPR.from_pretrained(
            args.base_model,
            config = config,
            _fast_init=False,
        )

    model.resize_token_embeddings(len(tokenizer))
    model.train()
    # if torch.__version__.startswith("2") and args.torch_compile: model = torch.compile(model)

    train_dataset = MSMarcoDataset(
        args.query_data_path,
        args.pos_doc_data_path,
        args.neg_doc_data_path,
        args.query_max_len,args.doc_max_len,args.num_samples
        )
    train_collate_fn = functools.partial(MSMarcoDataset.collate_fn,tokenizer=tokenizer,)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=args.shuffle_train_set,
        collate_fn=train_collate_fn,
        num_workers=4,pin_memory=True
        )

    dev_dataset = ValidationDataset(
        top1000_path=args.top1000_path,
        qrels_path=args.qrels_path,
        max_test_samples=args.max_test_samples,
        )
    dev_collate_fn = functools.partial(
        ValidationDataset.collate_fn,
        tokenizer=tokenizer,
        query_max_len=args.query_max_len,
        doc_max_len=args.doc_max_len
        )
    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size = 1,
        shuffle=False,
        collate_fn = dev_collate_fn,
    )
    
    no_decay = ["bias", "LayerNorm.weight"]
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
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,lr=args.lr)
    
    model, optimizer, train_dataloader, dev_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, dev_dataloader,
    )

    loss_fct = nn.CrossEntropyLoss()
    
    NUM_UPDATES_PER_EPOCH = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    MAX_TRAIN_STEPS = args.max_train_steps
    MAX_TRAIN_EPOCHS = math.ceil(MAX_TRAIN_STEPS / NUM_UPDATES_PER_EPOCH)
    TOTAL_TRAIN_BATCH_SIZE = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    EVAL_STEPS = args.val_check_interval if isinstance(args.val_check_interval,int) else int(args.val_check_interval * NUM_UPDATES_PER_EPOCH)
    total_loss = 0.0
    max_mrrAT10 = 0
    progress_bar_postfix_dict = {}

    logger.info("***** Running training *****")
    logger.info(f"  Num train examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {MAX_TRAIN_EPOCHS}")
    logger.info(f"  Num Updates Per Epoch = {NUM_UPDATES_PER_EPOCH}")
    logger.info(f"  Per device train batch size = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {TOTAL_TRAIN_BATCH_SIZE}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {MAX_TRAIN_STEPS}")
    completed_steps = 0
    progress_bar = tqdm(range(MAX_TRAIN_STEPS), disable=not accelerator.is_local_main_process,ncols=100)

    for epoch in range(MAX_TRAIN_EPOCHS):
        # mrrAT10 = validate(model,dev_dataloader,accelerator)
        set_seed(args.seed+epoch)
        progress_bar.set_description(f"epoch: {epoch+1}/{MAX_TRAIN_EPOCHS}")
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                with accelerator.autocast():

                    query_embedding = model.get_query_embedding(
                        input_ids = batch["query_input_ids"],
                        attention_mask = batch["query_attention_mask"],
                    )

                    doc_embedding = model.get_doc_embedding(
                        input_ids = batch['doc_input_ids'],
                        attention_mask = batch['doc_attention_mask']
                    )
                    
                    single_device_query_num = query_embedding.shape[0]
                    single_device_doc_num   = doc_embedding.shape[0]

                    ## maybe aggregate from multiple GPU
                    if accelerator.use_distributed:
                        doc_list = [torch.zeros_like(doc_embedding) for _ in range(accelerator.num_process)]
                        dist.all_gather(tensor_list=doc_list,tensor=doc_embedding.contiguous())
                        doc_list[dist.get_rank()] = doc_embedding
                        doc_embedding = torch.cat(doc_list, dim=0)

                        query_list = [torch.zeros_like(query_embedding) for _ in range(accelerator.num_processes)]
                        dist.all_gather(tensor_list=query_list, tensor=query_embedding.contiguous())
                        query_list[dist.get_rank()] = query_embedding
                        query_embedding = torch.cat(query_list, dim=0)

                    if args.model_type in ['colbert','polbert']:
                        ## Cross-GPU in batch negatives
                        all_query_num = query_embedding.shape[0]
                        all_doc_num   = doc_embedding.shape[0]

                        matching_score = []
                        for query_idx in range(all_query_num):
                            single_matching_score = model.get_matching_score(
                                doc_embedding = doc_embedding,
                                query_embedding = query_embedding[[query_idx],:,:].expand(all_doc_num,-1,-1)
                            )
                            matching_score.append(single_matching_score)
                        matching_score = torch.stack(matching_score,dim=0)
                    
                    elif args.model_type == 'dpr':
                        ## Cross-GPU in batch negatives
                        matching_score = model.get_matching_score(
                            query_embedding = query_embedding,
                            doc_embedding = doc_embedding,
                        )
                        
                    labels = torch.cat(
                        [torch.arange(single_device_query_num) + gpu_index * single_device_doc_num 
                            for gpu_index in range(accelerator.num_processes)
                        ]
                        ,dim=0
                    ).to(matching_score.device)
                    
                    loss = loss_fct(matching_score,labels)
                    total_loss += loss.item()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    optimizer.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
                    accelerator.log({"batch_loss": loss}, step=completed_steps)
                    accelerator.log({"average_loss": total_loss/completed_steps}, step=completed_steps)
                    progress_bar_postfix_dict.update(dict(rolling_loss=f"{total_loss/completed_steps:.4f}"))
                    progress_bar.set_postfix(progress_bar_postfix_dict)
                    
                    if completed_steps % EVAL_STEPS == 0:
                        mrrAT10 = validate(model,dev_dataloader,accelerator)
                        model.train()
                        accelerator.log({"dev_mrr@10": mrrAT10}, step=completed_steps)
                        if mrrAT10 > max_mrrAT10:
                            max_mrrAT10 = mrrAT10
                            if accelerator.is_local_main_process:
                                unwrapped_model = accelerator.unwrap_model(model)
                                unwrapped_model.save_pretrained(os.path.join(LOG_DIR,f"ckpt"))
                                tokenizer.save_pretrained(os.path.join(LOG_DIR,f"ckpt"))
                        accelerator.wait_for_everyone()
                    
                    if completed_steps > MAX_TRAIN_STEPS: break
    
    accelerator.log({"best_mrr@10":max_mrrAT10},step=completed_steps)
    if accelerator.is_local_main_process:wandb_tracker.finish()
    accelerator.end_training()

if __name__ == '__main__':
    main()