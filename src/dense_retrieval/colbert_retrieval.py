from tqdm import tqdm
import datasets
import os,json
import pandas as pd

def search(query,top_k=10):
    import requests  
    response = requests.get('http://localhost:8893/api/search', params={'query': query, 'k': top_k})  
    if response.status_code == 200:  
        return response.json()
    else:  
        print("Error:", response.status_code)
        return None

def main(queries,prefix,output_file):
    os.makedirs(prefix,exist_ok=True)
    responses = []
    for q in tqdm(queries):
        response = search(q,top_k=10)
        responses.append(response)
    with open(output_file,'w') as f:
        for response in responses:
            f.write(json.dumps(response)+'\n')

if __name__ == "__main__":
    ## sanity check
    print(search("Who won the 2022 FIFA world cup",top_k=2))

    # _prefix = "data/eval/mmlu"
    # temp_dataset = {}
    # for _split in ['dev','test']:
    #     new_data = []
    #     prefix = os.path.join(_prefix,_split)
    #     files = os.listdir(prefix)
    #     files.sort() ## because of randomness in os.listdir
    #     for file in files:
    #         file = os.path.join(prefix,file)
            
    #         if "test.csv" in file:
    #             subject = " ".join(os.path.basename(file).split("_test.csv")[0].split("_"))
    #         elif 'dev.csv' in file:
    #             subject = " ".join(os.path.basename(file).split("_dev.csv")[0].split("_"))

    #         df = pd.read_csv(file,header=None)
    #         data = [v for k,v in df.T.to_dict(orient="list").items()]
    #         for d in data:
    #             data_dict = {
    #                 "question":d[0].strip(),
    #                 "A":d[1],
    #                 "B":d[2],
    #                 "C":d[3],
    #                 "D":d[4],
    #                 "answer":d[5],
    #             }
    #             new_data.append(data_dict)
    #     temp_dataset[_split] = new_data

    # dev_data,test_data = temp_dataset['dev'],temp_dataset['test']
    # MULTIPLE_CHOICE_PROMPT = "{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer: {answer}"
    
    # dev_query = [MULTIPLE_CHOICE_PROMPT.format_map(d) for d in dev_data]
    # test_query = [MULTIPLE_CHOICE_PROMPT.format_map(d) for d in test_data]

    # prefix = "data/eval/mmlu/retrieval/colbertv2"
    # main(dev_query,prefix,os.path.join(prefix,"dev.jsonl"))
    # main(test_query,prefix,os.path.join(prefix,"test.jsonl"))



    # ##triviaqa
    # prefix = "data/eval/triviaqa"
    # dev_data = [json.loads(x) for x in open(os.path.join(prefix,"tqa-dev.jsonl")).readlines()]
    # test_data = [json.loads(x) for x in open(os.path.join(prefix,"tqa-test.jsonl")).readlines()]
    # prefix = os.path.join(prefix,"retrieval",'colbertv2')

    # queries = [x['question'] for x in dev_data]
    # output_file = os.path.join(prefix,"dev.jsonl")
    # main(queries,prefix,output_file)

    # queries = [x['question'] for x in test_data]
    # output_file = os.path.join(prefix,"test.jsonl")
    # main(queries,prefix,output_file)

    ## fm2
    # prefix = 'data/eval/fm2'
    # dev_data = [json.loads(x) for x in open(os.path.join(prefix,"fm2-dev.jsonl")).readlines()]
    # test_data = [json.loads(x) for x in open(os.path.join(prefix,"fm2-test.jsonl")).readlines()]
    # prefix = os.path.join(prefix,"retrieval",'colbertv2')

    # queries = [x['question'] for x in dev_data]
    # output_file = os.path.join(prefix,"dev.jsonl")
    # main(queries,prefix,output_file)

    # queries = [x['question'] for x in test_data]
    # output_file = os.path.join(prefix,"test.jsonl")
    # main(queries,prefix,output_file)

    # ## hotpot qa
    # dataset = datasets.load_dataset("kilt_tasks", "hotpotqa")
    # dev_data = []
    # for sample in dataset['train']:
    #     dev_data.append(
    #         {
    #             "question":sample['input'],
    #             "answer":sample['output'][0]['answer'],
    #         }
    #     )
    # test_data = []
    # for sample in dataset['validation']:
    #     test_data.append(
    #         {
    #             "question":sample['input'],
    #             "answer":sample['output'][0]['answer'],
    #         }
    #     )

    # prefix = "data/eval/hotpotqa/retrieval/colbertv2"
    # queries = [x['question'] for x in dev_data]
    # output_file = os.path.join(prefix,"dev.jsonl")
    # main(queries,prefix,output_file)

    # queries = [x['question'] for x in test_data]
    # output_file = os.path.join(prefix,"test.jsonl")
    # main(queries,prefix,output_file)

    # ## fever
    # dataset = datasets.load_dataset("kilt_tasks", "fever")
    # dev_data = []
    # for sample in dataset['train']:
    #     dev_data.append(
    #         {
    #             "question":sample['input'],
    #         }
    #     )
    # test_data = []
    # for sample in dataset['validation']:
    #     test_data.append(
    #         {
    #             "question":sample['input'],
    #         }
    #     )
    
    # prefix = "data/eval/fever/retrieval/colbertv2"
    # queries = [x['question'] for x in dev_data]
    # output_file = os.path.join(prefix,"dev.jsonl")
    # main(queries,prefix,output_file)

    # queries = [x['question'] for x in test_data]
    # output_file = os.path.join(prefix,"test.jsonl")
    # main(queries,prefix,output_file)

    # # wikitext103
    # prefix = 'data/eval/wikitext103'
    # test_data = [json.loads(x) for x in open(os.path.join(prefix,"test.jsonl")).readlines()]
    # prefix = os.path.join(prefix,"retrieval",'colbertv2')

    # queries = [x['text'] for x in test_data]
    # output_file = os.path.join(prefix,"test.jsonl")
    # main(queries,prefix,output_file)

    # ## wikitext2
    # prefix = 'data/eval/wikitext2'
    # test_data = [json.loads(x) for x in open(os.path.join(prefix,"test.jsonl")).readlines()]
    # prefix = os.path.join(prefix,"retrieval",'colbertv2')

    # queries = [x['text'] for x in test_data]
    # output_file = os.path.join(prefix,"test.jsonl")
    # main(queries,prefix,output_file)

    # ## wow
    # prefix = 'data/eval/truthfulqa'
    # test_data = [json.loads(x) for x in open(os.path.join(prefix,"test.jsonl")).readlines()]
    # prefix = os.path.join(prefix,"retrieval",'colbertv2')

    # queries = [x['question'] for x in test_data]
    # output_file = os.path.join(prefix,"test.jsonl")
    # main(queries,prefix,output_file)
    # ## wow
    prefix = 'data/eval/factkg'
    test_data = [json.loads(x) for x in open(os.path.join(prefix,"test.jsonl")).readlines()]
    prefix = os.path.join(prefix,"retrieval",'colbertv2')

    queries = [x['question'] for x in test_data]
    output_file = os.path.join(prefix,"test.jsonl")
    main(queries,prefix,output_file)

    # with open("tmp/curated_data.jsonl") as f:
    #     data = [json.loads(x) for x in f.readlines()]
    
    # for sample in tqdm(data):
    #     if 'background' not in sample.keys():
    #         query = sample['messages'][0]['content']
    #         response = search(query)
    #         sample['background'] = response['topk'][0]['text']
    
    # with open("tmp/rag_curated_data.jsonl",'w') as f:
    #     for sample in data:
    #         f.write(json.dumps(sample)+'\n')

    # with open("data/eval/webqa/test.jsonl") as f:
    #     data = [json.loads(x) for x in f.readlines()]
    
    # responses = []
    # for sample in tqdm(data):
    #     query = sample['question']
    #     response = search(query)
    #     responses.append(response)
    # os.makedirs("data/eval/webqa/retrieval/colbertv2")
    # with open("data/eval/webqa/retrieval/colbertv2/test.jsonl",'w') as f:
    #     for sample in responses:
    #         f.write(json.dumps(sample)+'\n')


    print("done")