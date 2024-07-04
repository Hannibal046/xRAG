## built-in
import argparse,json,os
import time
## third party
from transformers import (
    MistralForCausalLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    MixtralForCausalLM,
)
import torch
import datasets
from tqdm import tqdm
import pandas as pd

## own
from src.model import (
    XMistralForCausalLM,
    XMixtralForCausalLM,
    SFR,
)

from src.language_modeling.utils import (
    XRAG_TOKEN,
    get_retrieval_embeds,
)
from src.eval.utils import (
    stop_sequences_criteria,
    get_substring_match_score,
    eval_fact_checking,
    eval_truthfulqa,
    keyword_extraction_with_tfidf,
)
from src.utils import (
    get_jsonl,
)

def create_prompt_with_mistral_chat_format(messages,tokenizer,*args,**kwargs):
    # return tokenizer.apply_chat_template(messages,tokenize=False,add_special_tokens=False)
    formatted_text = ""
    for message in messages:
        if message['role'] == 'user':
            formatted_text += "[INST] " + message['content'] + " [/INST]"
        elif message['role'] == 'assistant':
            formatted_text += message['content'] + tokenizer.eos_token
        else:
            raise ValueError(
                "Mistral chat template only supports 'user' and 'assistant' roles. Invalid role: {}.".format(message["role"])
                )
    # formatted_text += " The answer is:"
    return formatted_text

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retrieval_prefix",
        default='colbertv2'
    )
    parser.add_argument(
        "--tf_idf_topk",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--base_model",
    )
    parser.add_argument(
        "--use_rag",
        action='store_true',
    )
    parser.add_argument(
        "--enable_progress_bar",
        type=eval,
        default=True,
    )
    parser.add_argument(
        "--data",
    )
    parser.add_argument(
        "--model_name_or_path",
    )
    parser.add_argument(
        "--eval_metrics",
    )
    parser.add_argument(
        "--n_shot",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--retriever_name_or_path",
    )
    parser.add_argument(
        "--retrieval_topk",
        type=int,
        default=[1],
        nargs='+',
    )
    parser.add_argument(
        "--retrieval_embed_length",
        type=int,default=0,
    )
    parser.add_argument(
        "--max_test_samples",
        type=int,
        help="for debug",
    )
    parser.add_argument(
        "--save_dir",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--chat_format",
        default='mistral',
    )
    args = parser.parse_args()

    ## post-process
    if args.data in ['nq_open','hotpotqa','triviaqa','webqa']:
        args.task_type = 'open_qa'
        args.eval_metrics = 'substring_match'
    elif args.data in ['truthfulqa']:
        args.task_type = 'open_qa'
        args.eval_metrics = 'truthfulqa_f1_rl'
    elif args.data in ['factkg']:
        args.task_type = 'fact_checking'
        args.eval_metrics = 'fact_checking_acc'
    
    args.retrieval_topk = [x-1 for x in args.retrieval_topk] ## rank starts from 1
    
    if args.chat_format is not None:
        args.chat_format = eval(f"create_prompt_with_{args.chat_format}_chat_format")    
    
    if args.retriever_name_or_path is not None:
        args.use_rag = True

    return args



QA_PROMPT = "Question: {question}?\n"
FECT_CHECKING_PROPMT = "Claim: {question}\n"
BACKGROUND_PROMPT_TEMPLATE = "Background: {background}\n\n"

PROMPT_TEMPLATES = {
    "open_qa":QA_PROMPT,
    'fact_checking':FECT_CHECKING_PROPMT,
}

def get_start_prompt(task_type,use_rag,sample=None):
    if task_type == 'open_qa':
        return {
            True: "Refer to the background document and answer the questions:",
            False:"Answer the questions:"
        }[use_rag]
    elif task_type == 'fact_checking':
        return {
            True: "Refer to the background document and verify the following claims with \"True\" or \"False\":",
            False:"Verify the following claims with \"True\" or \"False\":"
        }[use_rag]
        

@torch.no_grad()
def prepare_retrieval_embeds(backgrounds,retriever,tokenizer,batch_size = 16):
    backgrounds = [backgrounds[idx:idx+batch_size] for idx in range(0,len(backgrounds),batch_size)]
    device = retriever.device
    ret = []
    for background in backgrounds:
        tokenized_retrieval_text = tokenizer(
            background, 
            max_length=180,
            padding=True, truncation=True, return_tensors="pt")
        
        ## return a torch tensor of shape [batch_size,d_model]
        embeds = get_retrieval_embeds(
            model = retriever,
            input_ids = tokenized_retrieval_text['input_ids'].to(device),
            attention_mask = tokenized_retrieval_text['attention_mask'].to(device),
        ).cpu()

        embeds = [embeds[idx] for idx in range(embeds.shape[0])]
        ret.extend(embeds)
    return ret

@torch.no_grad()
def llm_for_open_generation(
    llm,llm_tokenizer,
    prompts,
    retrieval_embeds,
    batch_size = 4,
    enable_progress_bar = True,
):
    generated_answers = []
    total_test_number = len(prompts)
    device = llm.device
    batched_prompts = [prompts[idx:idx+batch_size] for idx in range(0,len(prompts),batch_size)]
    if retrieval_embeds is not None:
        batched_retrieval_embeds = [retrieval_embeds[idx:idx+batch_size] for idx in range(0,len(retrieval_embeds),batch_size)]
        assert len(batched_prompts) == len(batched_retrieval_embeds)
    
    progress_bar = tqdm(range(total_test_number),ncols=60,disable= not enable_progress_bar)
    for batch_idx in range(len(batched_prompts)):
        prompt = batched_prompts[batch_idx]
        tokenized_propmt = llm_tokenizer(prompt,padding='longest',return_tensors='pt')
        input_ids = tokenized_propmt.input_ids.to(device)
        attention_mask = tokenized_propmt.attention_mask.to(device)
        stopping_criteria = stop_sequences_criteria(llm_tokenizer, input_ids.shape[1], input_ids.shape[0])
        retrieval_kwargs = {}
        if retrieval_embeds is not None:
            embeds = batched_retrieval_embeds[batch_idx]
            embeds = [x for y in embeds for x in y]
            embeds = torch.stack(embeds).to(device)
            retrieval_kwargs['retrieval_embeds'] = embeds
            stopping_criteria = stop_sequences_criteria(llm_tokenizer, 0, input_ids.shape[0])

        ## actual computation
        generated_output = llm.generate(
            input_ids = input_ids,
            attention_mask = attention_mask,
            stopping_criteria=stopping_criteria,
            do_sample=False,
            max_new_tokens=100,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
            **retrieval_kwargs,
        )
        ## because HF generate with inputs_embeds would not return prompt
        input_length = 0 if retrieval_kwargs else input_ids.shape[1]
        results = tokenizer.batch_decode(generated_output[:,input_length:],skip_special_tokens=False)
        generated_answers.extend(results)
        progress_bar.update(batch_size)

    generated_answers = [x.strip() for x in generated_answers]
    return generated_answers

def format_one_example(
    sample,include_answer,use_rag,retrieval_embed_length,task_type,
):
    
    question   = sample['question']
    prompt_dict = dict(question=question)
    prompt = PROMPT_TEMPLATES[task_type].format_map(prompt_dict).strip()
    backgrounds = []

    if use_rag:
        backgrounds = sample['background'] ## a list
        background_prompts = ""
        
        for background in backgrounds:
            if retrieval_embed_length > 0:
                background_prompts += " ".join([XRAG_TOKEN]*retrieval_embed_length) + " "
            
            else:
                background_prompts += background + " "
        background_prompts = background_prompts.strip()
        prompt = BACKGROUND_PROMPT_TEMPLATE.format_map(dict(background=background_prompts)) + prompt


    return prompt,backgrounds

def get_n_shot_prompt(dev_data,n_shot,task_type,use_rag=False,retrieval_embed_length=0):
    assert n_shot >= 0,n_shot
    n_shot_prompt = []
    n_shot_background = []
    if dev_data is not None:
        n_shot_examples = dev_data[:n_shot]
        for example in n_shot_examples:
            prompt,background = format_one_example(example,include_answer=True,use_rag=use_rag,retrieval_embed_length=retrieval_embed_length,task_type=task_type)
            n_shot_prompt.append(prompt)
            n_shot_background.append(background)

    return n_shot_prompt,n_shot_background


def prepare_prompts(
    dev_data,test_data,task_type,tokenizer,
    n_shot = 0, use_rag = False,
    retrieval_embed_length=0,
    chat_format = None,
):
    splitter = "\n\n"
    prompts = []
    backgrounds = []
    original_n_shot = n_shot
    for idx,sample in enumerate(test_data):
        n_shot = original_n_shot
        while True:
            prompt_start  = get_start_prompt(task_type,use_rag=use_rag,sample=sample) 
            prompt_end,background    = format_one_example(
                sample,include_answer=False,use_rag=use_rag,retrieval_embed_length=retrieval_embed_length,task_type=task_type)
            if 'subject' not in sample.keys():
                n_shot_prompt,n_shot_background = get_n_shot_prompt(dev_data,n_shot=n_shot,use_rag=use_rag,retrieval_embed_length=retrieval_embed_length,task_type=task_type)
            else:
                ## select n-shot within the same subjects for MMLU
                dev_data_with_same_subjects = []
                for d in dev_data:
                    if d['subject'] == sample['subject']:
                        dev_data_with_same_subjects.append(d)
                assert len(dev_data_with_same_subjects)==5,sample['subject']
                n_shot_prompt,n_shot_background = get_n_shot_prompt(dev_data_with_same_subjects,n_shot=n_shot,use_rag=use_rag,retrieval_embed_length=retrieval_embed_length,task_type=task_type)
            
            if n_shot_prompt:  
                prompt = prompt_start + splitter + splitter.join(n_shot_prompt) + splitter + prompt_end  
            else: 
                prompt = prompt_start + splitter + prompt_end

            if chat_format is not None:
                messages = [{"role": "user", "content": prompt}]
                prompt = chat_format(messages, tokenizer) + " The answer is:"
                

            tokenized_prompt = tokenizer(prompt,truncation=False,add_special_tokens=False).input_ids

            if len(tokenized_prompt) > 2048 and n_shot >= 1:
                n_shot -= 1
            else:
                break
        
        prompts.append(prompt)
        backgrounds.append(background+n_shot_background)

    print("**"*20,"show one example","**"*20)
    print(prompts[0])
    print("**"*20,"show one example","**"*20)

    return prompts,backgrounds


def load_dataset(data,use_rag,args):
    
    dev_data = None
    test_path = f"data/eval/{data}/test.jsonl"
    test_data = None
    if os.path.isfile(test_path):
        test_data = get_jsonl(test_path)

    if use_rag:

        test_retrieval_path = os.path.join(f"data/eval/{data}/retrieval/{args.retrieval_prefix}","test.jsonl")
        test_retrieval = get_jsonl(test_retrieval_path)
        assert len(test_retrieval) == len(test_data)
        for idx in range(len(test_data)):
            test_data[idx]['background'] = [test_retrieval[idx]['topk'][rank]['text'] for rank in args.retrieval_topk]
        
        if args.tf_idf_topk > 0:
            assert args.use_rag
            documents = [x['background'][0] for x in test_data]
            keywords = keyword_extraction_with_tfidf(documents,topk=args.tf_idf_topk)
            for idx in range(len(test_data)):
                test_data[idx]['background'] = [keywords[idx]]
        
        if args.retriever_name_or_path is not None and args.retriever_name_or_path.lower() == "intfloat/e5-large-v2":
            for idx in range(len(test_data)):
                test_data[idx]['background'] = ["passage: " + x for x in test_data[idx]['background']]


    return dev_data,test_data

if __name__ == "__main__":

    args = parse_args()

    ## load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side = 'left',
        add_eos_token=False, ## import to include this!
        use_fast=False,
    )
    if tokenizer.pad_token:
        pass
    elif tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    ## load retriever and retriever_tokenizer
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    retrieval_embed_length = 0
    retriever,retriever_tokenizer = None,None
    if args.retriever_name_or_path is not None:
    
        if args.retriever_name_or_path.lower() == 'salesforce/sfr-embedding-mistral':
            retriever = SFR.from_pretrained(args.retriever_name_or_path,torch_dtype = torch.bfloat16)
            retriever_tokenizer = AutoTokenizer.from_pretrained(args.retriever_name_or_path)
        retrieval_embed_length = retriever.get_embed_length()
        retriever_hidden_size = retriever.get_embed_dim()
        retriever.eval()
        retriever = retriever.to(device)


    ## prepare prompt
    dev_data,test_data = load_dataset(
        args.data,
        args.use_rag,
        args,
    )

    if args.max_test_samples is not None:
        test_data = test_data[:args.max_test_samples]

    prompts,backgrounds = prepare_prompts(
        dev_data = dev_data,
        test_data = test_data,
        task_type = args.task_type,
        tokenizer = tokenizer,
        n_shot = args.n_shot,
        use_rag = args.use_rag,
        retrieval_embed_length = retrieval_embed_length,
        chat_format = args.chat_format, 
    )

    retrieval_embeds = None
    if retriever is not None:
        # backgrounds List[List[String]]
        num_samples = len(backgrounds)
        original_orders = []
        for idx,background in enumerate(backgrounds):
            original_orders.extend(
                [idx] * len(background)
            )
        
        backgrounds = [x for y in backgrounds for x in y]
        print(f"Preparing document embedding with {args.retriever_name_or_path}...")
        _retrieval_embeds = prepare_retrieval_embeds(
            backgrounds,
            retriever,
            retriever_tokenizer,
        )

        retrieval_embeds = [[] for _ in range(num_samples)]
        assert len(_retrieval_embeds) == len(original_orders)
        for id,embeds in zip(original_orders,_retrieval_embeds):
            retrieval_embeds[id].append(embeds)

        retriever = retriever.to("cpu")


    avg_prompt_length = tokenizer(prompts,return_length=True).length
    avg_prompt_length = sum(avg_prompt_length)/len(avg_prompt_length)
    

    ## load llm
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    MODEL_CLASS = eval(config.architectures[0])
    model = MODEL_CLASS.from_pretrained(
        args.model_name_or_path,
        torch_dtype = torch.bfloat16,
        low_cpu_mem_usage = True,
        device_map='auto',
    )
    
    model.eval()
    # model = model.to(device)
    if retriever is not None:
        assert XRAG_TOKEN in tokenizer.get_vocab() 
        model.set_xrag_token_id(tokenizer.convert_tokens_to_ids(XRAG_TOKEN))

    if args.task_type in ['open_qa','fact_checking']:
        generated_results = llm_for_open_generation(
            llm = model,
            llm_tokenizer = tokenizer,
            prompts = prompts,
            retrieval_embeds = retrieval_embeds,
            batch_size = args.eval_batch_size,
            enable_progress_bar= args.enable_progress_bar,
        )

    answers = [x['answer'] for x in test_data]
    if args.eval_metrics == 'substring_match':
        score,score_per_sample = get_substring_match_score(generated_results,answers)
    elif args.eval_metrics == 'fact_checking_acc':
        score,score_per_sample = eval_fact_checking(generated_results,answers)
    elif args.eval_metrics == 'truthfulqa_f1_rl':
        f1,rl,f1_scores,rl_scores = eval_truthfulqa(generated_results,answers)
        score = f"{f1}-{rl}"
        score_per_sample = [(f1_score,rl_score) for f1_score,rl_score in zip(f1_scores,rl_scores)]


    result_dict =   {
        "dataset":args.data,
        "batch_size":args.eval_batch_size,
        "include_retrieval":args.use_rag,
        "avg_prompt_length":avg_prompt_length,
        "model":args.model_name_or_path,
        f"{args.eval_metrics}":score,
    }

    if args.retriever_name_or_path is not None:
        result_dict['retriever'] = args.retriever_name_or_path
    print(json.dumps(result_dict,indent=4))
