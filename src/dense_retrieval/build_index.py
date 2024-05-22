import faiss
import argparse
import os
from tqdm import tqdm
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_dir",required=True)
    parser.add_argument("--dim",type=int,default=128)
    parser.add_argument("--sample_ratio",type=float,default=0.3)
    parser.add_argument("--output_path",required=True)
    parser.add_argument("--nlist",type=int,default=32768)
    parser.add_argument("--m",type=int,default=16)
    parser.add_argument("--nbits_per_idx",type=int,default=8)
    args = parser.parse_args()

    embedding_files = [os.path.join(args.embedding_dir,x) for x in os.listdir(args.embedding_dir) if x.endswith("pt")]
    embedding_files.sort(key=lambda x:os.path.basename(x).split(".")[0].split("_")[-2:])

    embeddings_for_training = []
    for file in embedding_files:
        print("loading from ",file)
        data = torch.load(file)
        sampled_data = data[torch.randint(0, high=data.size(0), size=(int(data.size(0) * args.sample_ratio),))]
        embeddings_for_training.append(sampled_data)

    embeddings_for_training = torch.cat(embeddings_for_training,dim=0)
    print(f"{embeddings_for_training.shape=}")

    ## build index
    quantizer = faiss.IndexFlatL2(args.dim)
    index = faiss.IndexIVFPQ(quantizer, args.dim, args.nlist, args.m, args.nbits_per_idx)

    ## training
    gpu_resource = faiss.StandardGpuResources()
    gpu_quantizer = faiss.index_cpu_to_gpu(gpu_resource, 0, quantizer)
    gpu_index = faiss.index_cpu_to_gpu(gpu_resource, 0, index)
    gpu_index.train(embeddings_for_training)

    ## add
    ## if OOM, try to split into small batches
    for file in tqdm(embedding_files,desc='loading from embedding files'):
        data = torch.load(file)
        gpu_index.add(data)

    cpu_index = faiss.index_gpu_to_cpu(gpu_index)

    ## save
    faiss.write_index(cpu_index, args.output_path)
