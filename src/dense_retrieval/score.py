from collections import defaultdict
import json
import argparse
from utils import get_mrr,get_recall

if __name__ == '__main__':
    parser =  argparse.ArgumentParser()
    parser.add_argument("--qrel_path",default="data/qrels.dev.small.tsv")
    parser.add_argument("--ranking_path")
    args = parser.parse_args()

    qid2positives = defaultdict(list)
    with open(args.qrel_path) as f:
        for line in f:
            qid,_,pid,label = [int(x) for x in line.strip().split()]
            assert label == 1
            qid2positives[qid].append(pid)

    qid2ranking = defaultdict(list)
    with open(args.ranking_path) as f:
        for line in f:
            qid,pid,rank = [int(x) for x in line.strip().split("\t")]
            qid2ranking[qid].append(pid)
    
    results = get_mrr(qid2ranking,qid2positives)
    results.update(get_recall(qid2ranking,qid2positives))

    print(json.dumps(results,indent=4))