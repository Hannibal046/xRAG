# ================== #
# This is an unoptimized version of colbert-v1 retrieval 
# ================== #
import argparse
import os
import pickle
from tqdm import tqdm
from model import ColBERT
from transformers import BertTokenizer
import torch
import faiss
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dir",default='embedding/colbert')
    parser.add_argument("--faiss_index_path")
    parser.add_argument("--pretrained_model_path")
    parser.add_argument("--query_path",default='data/queries.dev.small.tsv')
    parser.add_argument("--nprobe",type=int,default=32)
    parser.add_argument("--query_max_len",type=int,default=32)
    parser.add_argument("--doc_max_len",type=int,default=180)
    parser.add_argument("--search_k",type=int,default=1024)
    parser.add_argument("--save_k",type=int,default=1000)
    parser.add_argument("--output_path")

    args = parser.parse_args()
    
    device = torch.device("cuda:0")

    colbert = ColBERT.from_pretrained(args.pretrained_model_path)
    colbert.eval()
    colbert = colbert.to(device)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path)
    DIM = colbert.config.dim
    
    embedding_files = [os.path.join(args.embedding_dir,x) for x in os.listdir(args.embedding_dir) if x.endswith("pt")]
    embedding_files.sort(key=lambda x:os.path.basename(x).split(".")[0].split("_")[-2:])

    length_files = [os.path.join(args.embedding_dir,x) for x in os.listdir(args.embedding_dir) if x.endswith("pkl")]
    length_files.sort(key=lambda x:os.path.basename(x).split(".")[0].split("_")[-2:])

    # 1. token level retrieval
    print(f"reading faiss index from {args.faiss_index_path}")
    faiss_index = faiss.read_index(args.faiss_index_path)
    faiss_index.nprobe = args.nprobe

    # 2. sentence level reranking
    all_token_embeddings = []
    for file in embedding_files:
        print(f"loading {file}")
        all_token_embeddings.append(torch.load(file))
    dummy_embeddings = torch.zeros((args.doc_max_len,DIM)) ## since we select each doc with doc_max_len
    all_token_embeddings.append(dummy_embeddings)
    all_token_embeddings = torch.cat(all_token_embeddings,dim=0)
    print("total_embeddings.shape=",all_token_embeddings.shape)


    ## build mapping
    all_length = [pickle.load(open(x,'rb')) for x in length_files]
    all_length = [x for y in all_length for x in y]

    NUM_DOCS = len(all_length)
    NUM_EMBEDDINGS = all_token_embeddings.shape[0] - args.doc_max_len

    embedding2pid = [0 for _ in range(NUM_EMBEDDINGS)]
    pid2embedding = [0 for _ in range(NUM_DOCS)]

    start_pos = 0
    for pid,length in enumerate(all_length):
        for char_pos in range(start_pos,start_pos+length):
            embedding2pid[char_pos] = pid
        pid2embedding[pid] = start_pos
        start_pos += length

    ## load query files
    queries = []
    with open(args.query_path) as f:
        for line in f:
            qid,query = line.strip().split("\t")
            queries.append((qid,query))
    
    all_time = {
        "encoding":[],
        "total":[],
        "faiss":[],
        "topk_mapping":[],
        "get_doc_embedding":[],
        "matching":[],
    }
    ranking = []
    progress_bar = tqdm(range(len(queries)))
    for qid,query in queries:
        total_time_start = time.time()

        ## ===encoding queries=== ##
        encoding_start_time = time.time()

        query = "[Q]" + " " + query
        tokenized_query = tokenizer(query,return_tensors='pt',padding="max_length",max_length=args.query_max_len).to(device)
        input_ids = tokenized_query.input_ids
        input_ids[input_ids == tokenizer.pad_token_id] = tokenizer.mask_token_id
        attention_mask = tokenized_query.attention_mask
        with torch.no_grad():
            query_embedding = colbert.get_query_embedding(
                input_ids = tokenized_query.input_ids,
                attention_mask = tokenized_query.attention_mask,
            ).squeeze(0)

        all_time['encoding'].append(time.time()-encoding_start_time)

        ## ===faiss search=== ##
        faiss_start_time = time.time()
        embedding_to_faiss = query_embedding.cpu()
        _ , I = faiss_index.search(embedding_to_faiss, args.search_k)
        all_time['faiss'].append(time.time()-faiss_start_time)

        ## ===get top relevant docs=== ##
        topk_mapping_start_time = time.time()
        top_relevant_doc_pids = [embedding2pid[x] for y in I for x in y]
        top_relevant_doc_pids = list(set(top_relevant_doc_pids))
        all_time['topk_mapping'].append(time.time()-topk_mapping_start_time)

        ## ===get doc_embedding=== ##
        get_doc_embedding_start_time = time.time()

        lengths = torch.tensor([all_length[pid] for pid in top_relevant_doc_pids])
        
        mask = torch.arange(args.doc_max_len).unsqueeze(0)
        mask = (mask < lengths.unsqueeze(-1)).to(device)
        
        doc_start_pos_id = torch.tensor([pid2embedding[pid]  for pid in top_relevant_doc_pids])
        ## taken the doc_max_len for matrix multiplication
        ## using mask to mask out the extra token
        batch_indices = (doc_start_pos_id.unsqueeze(-1) + torch.arange(args.doc_max_len).unsqueeze(0)).view(-1)  
        doc_embeddings = all_token_embeddings[batch_indices].view(len(top_relevant_doc_pids), args.doc_max_len, -1)  
        doc_embeddings = doc_embeddings.to(device).to(query_embedding.dtype)

        all_time['get_doc_embedding'].append(time.time()-get_doc_embedding_start_time)

        ## ===matching=== ##
        matching_start_time =  time.time()
        ## using matrix multiplication would not change the relative order of L2-optimized retriever
        ## https://github.com/stanford-futuredata/ColBERT/issues/40
        scores = (doc_embeddings @ query_embedding.unsqueeze(0).permute(0,2,1)) 
        ## using mask to mask out the extra token
        scores = scores * mask.unsqueeze(-1)
        ## MaxSim operation
        scores = scores.max(1).values.sum(-1).cpu()
        scores_sorter = scores.sort(descending=True)
        pids, scores = torch.tensor(top_relevant_doc_pids)[scores_sorter.indices].tolist(), scores_sorter.values.tolist()
        pids = pids[:args.save_k]
        scores = scores[:args.save_k]
        all_time['matching'].append(time.time() - matching_start_time)

        all_time['total'].append(time.time() - total_time_start)

        total_time = sum(all_time["total"])
        progress_bar_postfix_dict = {}
        for key,value in all_time.items():
            progress_bar_postfix_dict[key] = f"{sum(value)/total_time*100:.1f}%"
        
        progress_bar_postfix_dict.pop("total")
        progress_bar.set_postfix(progress_bar_postfix_dict)

        ranking.append((qid,pids))
        progress_bar.update(1)

    with open(args.output_path,'w') as f:
        for qid,pids in ranking:
            for idx,pid in enumerate(pids):
                ## qid-pid-rank
                f.write(f"{qid}\t{pid}\t{idx+1}\n")
        
    

