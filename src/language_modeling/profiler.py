from torch.profiler import ProfilerActivity
from torch.profiler import profile as torch_profile
from torch.profiler import record_function
import json
from src.model import XMistralForCausalLM,XMistralConfig
from transformers import AutoTokenizer
from tokenizers import AddedToken
from src.language_modeling.utils import XRAG_TOKEN
import torch


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction_length",type=int)
    parser.add_argument("--num_docs",type=int, default=1)
    parser.add_argument("--generation_length",type=int)
    parser.add_argument("--use_xrag",action='store_true',default=False)
    parser.add_argument("--dataset")
    args = parser.parse_args()


    device = torch.device("cuda")
    torch_dtype = torch.bfloat16
    pretrained_model_name_or_path = "Hannibal046/xrag-7b"
    num_trails = 10
    batch_size = 12
    instruction_length = args.instruction_length
    retriever_hidden_size = 4096
    num_docs = args.num_docs 
    document_length = sum([180]*num_docs)
    generation_length = args.generation_length
    use_xrag = args.use_xrag


    config = XMistralConfig.from_pretrained(pretrained_model_name_or_path,retriever_hidden_size=retriever_hidden_size)
    model = XMistralForCausalLM.from_pretrained(pretrained_model_name_or_path,config=config,torch_dtype=torch_dtype).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
    if tokenizer.pad_token:
        pass
    elif tokenizer.unk_token:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    elif tokenizer.eos_token:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    num_added_tokens = tokenizer.add_tokens([AddedToken(XRAG_TOKEN,lstrip=False,rstrip=False)])
    xrag_token_id = tokenizer.convert_tokens_to_ids(XRAG_TOKEN)
    model.set_xrag_token_id(xrag_token_id)
    if num_added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer))
    vocab_size = len(tokenizer)

    

    retrieval_kwargs = {}
    if use_xrag:
        input_ids = torch.randint(low=0,high=vocab_size-1,size=(batch_size,instruction_length + num_docs)).to(device)
        attention_mask = torch.ones_like(input_ids)
        input_ids[:,3:3+num_docs] = xrag_token_id
        retrieval_kwargs['retrieval_embeds'] = torch.rand(num_docs*batch_size,retriever_hidden_size,dtype=torch_dtype).to(device)
    else:
        input_ids = torch.randint(low=0,high=vocab_size-1,size=(batch_size,instruction_length + document_length)).to(device)
        attention_mask = torch.ones_like(input_ids)

    model.generate(
        input_ids=input_ids,
        attention_mask = attention_mask,
        do_sample=False,
        max_new_tokens=generation_length,
        min_new_tokens=generation_length,
        pad_token_id = tokenizer.pad_token_id,
        **retrieval_kwargs,
    )


    torch.cuda.reset_peak_memory_stats(device)
    with torch_profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_flops=True,
        ) as prof:
        with record_function("model_inference"):
            for _ in range(num_trails):
                model.generate(
                    input_ids=input_ids,
                    attention_mask = attention_mask,
                    do_sample=False,
                    max_new_tokens=generation_length,
                    min_new_tokens=generation_length,
                    pad_token_id = tokenizer.pad_token_id,
                    **retrieval_kwargs,
                )

    peak_mem_usage = torch.cuda.memory_stats()["allocated_bytes.all.peak"] /2**30
    events = prof.key_averages()
    for event in events:
        if event.key == 'model_inference':
            model_inference_event = event
            break

    total_cpu_time = model_inference_event.cpu_time_total/1000**2 / num_trails
    total_cuda_time = model_inference_event.cuda_time_total/1000**2 / num_trails
    total_gflops = sum([event.flops for event in events]) / 1e9 / num_trails
    
    result_dict =  {
            "instruction_length":instruction_length,
            "document_length":document_length,
            "prompt_length":input_ids.shape[1],
            "generation_length":generation_length,
            "use_xrag":use_xrag, 
            "cpu_time":total_cpu_time,
            "cuda_time":total_cuda_time,
            "gflops":total_gflops/generation_length,
            "peak_mem":peak_mem_usage,
        }
    print(json.dumps(result_dict,indent=4))