import numpy as np
from tqdm import tqdm
import os

## own
from src.model import RAGTokenizerFast

if __name__ == "__main__":

    tokenizer = RAGTokenizerFast.from_pretrained("bert-base-uncased",additional_special_tokens=["[Q]","[D]"])
    query_max_len = 32
    doc_max_len = 180
    triplet_path = "data/msmarco/triples.train.small.tsv"
    batch_size = 100_000
    num_samples = 39780811

    os.makedirs("data/msmarco/processed",exist_ok=True)
    query_mmap = np.memmap('data/msmarco/processed/queries.mmap', dtype='int16',mode='w+',shape=(num_samples,query_max_len))
    pos_mmap   = np.memmap("data/msmarco/processed/pos_docs.mmap",dtype='int16',mode='w+',shape=(num_samples,doc_max_len))
    neg_mmap   = np.memmap("data/msmarco/processed/neg_docs.mmap",dtype='int16',mode='w+',shape=(num_samples,doc_max_len))

    total = 0
    progress_bar = tqdm(range(num_samples),desc='processing triplet data...')
    with open(triplet_path) as f:
        queries,poses,negs = [],[],[]
        for line in f:
            query,pos,neg = line.strip().split("\t")
            queries.append(query)
            poses.append(pos)
            negs.append(neg)

            if len(queries) == batch_size:
                query_input_ids =  tokenizer.tokenize_query(queries,max_length=query_max_len)['input_ids']
                pos_input_ids   =  tokenizer.tokenize_document(poses,max_length=doc_max_len)['input_ids']
                neg_input_ids   =  tokenizer.tokenize_document(negs,max_length=doc_max_len)['input_ids']
                
                query_mmap[total:total+batch_size] = query_input_ids.numpy().astype(np.int16)  
                pos_mmap[  total:total+batch_size] = pos_input_ids.numpy().astype(np.int16)  
                neg_mmap[  total:total+batch_size] = neg_input_ids.numpy().astype(np.int16)  

                total += batch_size
                progress_bar.update(batch_size)
                queries,poses,negs = [],[],[]

        if len(queries) > 0:
            current_size = len(queries)
            query_input_ids =  tokenizer.tokenize_query(queries,max_length=query_max_len)['input_ids']
            pos_input_ids   =  tokenizer.tokenize_document(poses,max_length=doc_max_len)['input_ids']
            neg_input_ids   =  tokenizer.tokenize_document(negs,max_length=doc_max_len)['input_ids']
                    
            query_mmap[total:total+current_size] = query_input_ids.numpy().astype(np.int16)  
            pos_mmap[  total:total+current_size] = pos_input_ids.numpy().astype(np.int16)  
            neg_mmap[  total:total+current_size] = neg_input_ids.numpy().astype(np.int16)  

            assert current_size + total == num_samples

        query_mmap.flush()
        pos_mmap.flush()
        neg_mmap.flush()