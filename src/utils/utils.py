import os,json
from transformers import AutoTokenizer,AutoModelForCausalLM

def get_jsonl(f):
    import json
    return [json.loads(x) for x in open(f).readlines()]

def write_jsonl(data,path):
    import json
    with open(path,'w') as f:
        for sample in data:
            f.write(json.dumps(sample)+'\n')



def get_bleu_score(hyps,refs,return_signature=False):
    # pip install sacrebleu
    """
    hyps:list of string
    refs:list of string
    """
    assert len(hyps) == len(refs)
    
    import sacrebleu
    scorer = sacrebleu.metrics.BLEU(force=True)
    score = scorer.corpus_score(hyps,[refs]).score
    signature = scorer.get_signature()
    if return_signature:
        return score,str(signature)
    else:
        return score

def get_rouge_score(hyps,refs):
    from compare_mt.rouge.rouge_scorer import RougeScorer
    assert len(hyps)==len(refs)
    lens = len(hyps)    
    rouge_scorer = RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=True)    
    rouge1 = rouge2 = rougel = 0.0
    for hyp,ref in zip(hyps,refs):
        score = rouge_scorer.score(ref,hyp)
        rouge1 += score['rouge1'].fmeasure
        rouge2 += score['rouge2'].fmeasure
        rougel += score['rougeLsum'].fmeasure
    rouge1 = rouge1 / lens
    rouge2 = rouge2 / lens
    rougel = rougel / lens
    return rouge1,rouge2,rougel

def load_wiki_collection(collection_path="data/wikipedia/collection.tsv",verbose=True,max_samples=None):
    wiki_collections = {}
    cnt = 0
    with open(collection_path) as f:
        for line in f:
            pid, passage, *rest = line.strip('\n\r ').split('\t')
            pid = int(pid)
            if len(rest) >= 1:
                title = rest[0]
                passage = title + ' | ' + passage
            wiki_collections[pid] = passage
            cnt += 1
            if cnt % 1000_0000 == 0 and verbose:
                print('loading wikipedia collection',cnt)
            
            if max_samples is not None and len(wiki_collections) > max_samples:
                break
    return wiki_collections

def set_seed(seed: int = 19980406):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_yaml_file(file_path):  
    import yaml
    try:  
        with open(file_path, 'r') as file:  
            return yaml.safe_load(file)  
    except FileNotFoundError:  
        print(f"YAML configuration file {file_path} not found.")  
        return {}  

def file_tqdm(file):
    import tqdm
    import os
    with tqdm.tqdm(total=os.path.getsize(file.name) / 1024.0 / 1024.0, unit="MiB") as pbar:
        for line in file:
            yield line
            pbar.update(len(line) / 1024.0 / 1024.0)

        pbar.close()

def get_mrr(qid2ranking,qid2positives,cutoff_rank=10):
    """
    qid2positives: {1:[99,13]}
    qid2ranking: {1:[99,1,32]} (sorted)
    """
    assert set(qid2positives.keys()) == set(qid2ranking.keys())

    qid2mrr = {}
    for qid in qid2positives:
        positives = qid2positives[qid]
        ranked_pids = qid2ranking[qid]

        for rank,pid in enumerate(ranked_pids,start=1):
            if pid in positives:
                if rank <= cutoff_rank:
                    qid2mrr[qid] = 1.0/rank
                break

    return {
        f"mrr@{cutoff_rank}":sum(qid2mrr.values())/len(qid2ranking.keys())
    }

def get_recall(qid2ranking,qid2positives,cutoff_ranks=[50,200,1000,5000,10000]):
    """
    qid2positives: {1:[99,13]}
    qid2ranking: {1:[99,1,32]} (sorted)
    """
    assert set(qid2positives.keys()) == set(qid2ranking.keys())
    
    qid2recall = {cutoff_rank:{} for cutoff_rank in cutoff_ranks}
    num_samples = len(qid2ranking.keys())
    
    for qid in qid2positives:
        positives = qid2positives[qid]
        ranked_pids = qid2ranking[qid]
        for rank,pid in enumerate(ranked_pids,start=1):
            if pid in positives:
                for cutoff_rank in cutoff_ranks:
                    if rank <= cutoff_rank:
                        qid2recall[cutoff_rank][qid] = qid2recall[cutoff_rank].get(qid, 0) + 1.0 / len(positives)
    
    return {
        f"recall@{cutoff_rank}":sum(qid2recall[cutoff_rank].values()) / num_samples
        for cutoff_rank in cutoff_ranks
    }