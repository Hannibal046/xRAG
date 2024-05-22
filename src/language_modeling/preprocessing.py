import random,copy

from .utils import ParaphraseInstructions,XRAG_TOKEN

def split_background(background,tokenizer,total_max_len,single_max_len,single_min_len=20):
    """
    split a long document into multiple smaller chunks between single_max_len and single_mini_len
    
    Args:
        background: string
    
    Return:
        background: a list of string
    """
    ids = tokenizer(background,add_special_tokens=False,max_length = total_max_len,truncation=True).input_ids
    background = [ids[idx:idx+single_max_len] for idx in range(0,len(ids),single_max_len)]
    assert len(background) >= 1, background
    if len(background[-1]) <= single_min_len and len(background)>1:
        background = background[:-1]
    background = [tokenizer.decode(x) for x in background]
    return background

def _concat_messages_mixtral(messages,tokenizer):
    ## Mixtral Chat Format
    return _concat_messages_mistral(messages,tokenizer)

def _concat_messages_mistral(messages,tokenizer):
    ## Mistral Chat Format
    message_text = ""
    for message in messages:
        if message["role"] == "user":
            message_text += "[INST] " + message["content"].strip() + " [/INST]"
        elif message["role"] == "assistant":
            message_text += message["content"].strip() + tokenizer.eos_token
        else:
            raise ValueError("Invalid role: {}".format(message["role"]))
    return message_text

def _encode_chat_format(
        messages,
        tokenizer,
        max_seq_length,
        chat_format='mistral', ## tulu
    ):
    """
    encode messages to input_ids and make non-assistant part

    Args:
        messages (list): list of dict with 'role' and 'content' field
        tokenizer: llm tokenizer
        max_seq_lengh: maximun context length  
    
    Return:
        input_ids and labels
    """
    _concat_messages = eval(f"_concat_messages_{chat_format}")
    
    example_text = _concat_messages(messages,tokenizer).strip()
    tokenized_example = tokenizer(example_text, return_tensors='pt', max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()
    # assert tokenizer.eos_token_id in input_ids, (tokenizer("this is good."+tokenizer.eos_token +'\n').input_ids,input_ids)
    
    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx],tokenizer), return_tensors='pt', max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            
            if chat_format in ['mistral','mixtral']:
                messages_so_far = _concat_messages(messages[:message_idx+1],tokenizer)         

            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt', 
                max_length=max_seq_length, 
                truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100
            
            if message_end_idx >= max_seq_length:
                break
    
    # assert tokenizer.eos_token_id in input_ids, input_ids
    return {
        "input_ids":input_ids.flatten(),
        "labels":labels.flatten(),
    }

def encode_with_chat_format_pretrain(
        example,
        tokenizer,
        max_seq_length,
        retrieval_embed_length,
        chat_format='mistral',
        ):
    """
    encode messages into input_ids and labels for paraphrase pretrain

    Args:
        example: data sample with 'text' filed
        tokenizer: llm_tokenizer
        max_seq_length: maximun context length
        retrieval_embed_length: number of tokens for retrieval (typically 1 for dense retrieval model)
    
    Return:
        input_ids,labels and retriever_input_text
    """    
    # if tokenizer.eos_token_id not in tokenizer("this is good."+tokenizer.eos_token +'\n').input_ids:
    #     from transformers import AutoTokenizer
    #     new_tokenizer = AutoTokenizer.from_pretrained("allenai/tulu-2-7b")
    #     assert new_tokenizer.eos_token_id in new_tokenizer("this is good."+new_tokenizer.eos_token +'\n').input_ids, 'new_tokenizer'
    #     assert tokenizer.eos_token_id in tokenizer("this is good."+tokenizer.eos_token +'\n').input_ids, 'encode_with_chat_format_pretrain'    
    #     print(new_tokenizer)
    #     print(tokenizer)

    document = example['text'].strip()
    xrag_token = " ".join([XRAG_TOKEN]*retrieval_embed_length)
    instruction = random.choice(ParaphraseInstructions).format_map(dict(xrag_token=xrag_token))

    messages = [
        {"role":"user","content":instruction},
        {"role":"assistant","content":document},
    ]

    encoded = _encode_chat_format(messages,tokenizer,max_seq_length,chat_format)

    return {
        "xrag_input_ids":encoded['input_ids'],
        "xrag_labels":encoded['labels'],
        "retriever_input_text":[document],
    }

def encode_with_chat_format_finetune(
        example, 
        tokenizer,
        max_seq_length,
        retrieval_embed_length,
        use_rag_tuning = True,
        use_retriever_embed=False,
        retriever_tokenizer = None,
        chat_format = 'mistral'
    ):
    '''
    Here we assume each example has three fields:
        1) messages
        2) backgrounds
        3) task_type  
    '''
    messages,background = example['messages'],example['background']

    ret = {}

    if use_rag_tuning and use_retriever_embed:
        sharded_background = split_background(background,retriever_tokenizer,total_max_len=max_seq_length,single_max_len=180)
        num_split = len(sharded_background)
        ret['retriever_input_text'] = sharded_background
    
    if use_rag_tuning:

        _messages = copy.deepcopy(messages)
        xrag_tokens = " ".join([XRAG_TOKEN]*retrieval_embed_length* num_split)
            
        for idx in range(len(_messages)):
            if _messages[idx]['role'] == 'user':
                _messages[idx]['content'] = f"Refer to the background document: {xrag_tokens}\n\n" + messages[idx]['content']
                break
        encoded = _encode_chat_format(_messages,tokenizer,max_seq_length,chat_format=chat_format)
        ret['xrag_input_ids'] = encoded['input_ids']
        ret['xrag_labels'] = encoded['labels']


        ## vanilla RAG
        _messages = copy.deepcopy(messages)
        for idx in range(len(_messages)):
            if _messages[idx]['role'] == 'user':
                _messages[idx]['content'] = f"Refer to the background document: {background}\n\n" + messages[idx]['content']
                break
        
        encoded = _encode_chat_format(_messages,tokenizer,max_seq_length,chat_format=chat_format)
        ret['input_ids'] = encoded['input_ids']
        ret['labels'] = encoded['labels']
    
    return ret

def encode_with_qa_format(
        example, 
        tokenizer,
        max_seq_length,
        retrieval_embed_length,
        use_rag_tuning = True,
        use_retriever_embed=False,
        use_paraphrase_finetune = False,
        background_dropout_rate=0.0,):
    '''
    Here we assume each example has three fields:
        1) question
        2) answer
        3) background  
    '''
    def get_input_and_labels(prompt,label,background=None):
        input_ids = tokenizer(prompt,max_length=max_seq_length,truncation=True).input_ids
        labels = [-100] * len(input_ids)
        
        ## match backgrounds
        if background is not None:
            background_ids = tokenizer(background,add_special_tokens=False).input_ids 
            background_start_idx = find_matched_index(input_ids,background_ids)
            if background_start_idx != -1:
                labels[background_start_idx:background_start_idx+len(background_ids)] = input_ids[background_start_idx:background_start_idx+len(background_ids)]


        ## match labels
        label_ids = tokenizer(label,add_special_tokens=False).input_ids
        label_start_idx = find_matched_index(input_ids,label_ids)
        if label_start_idx != -1: ## extreme long propmt
            labels[label_start_idx:label_start_idx+len(label_ids)] = input_ids[label_start_idx:label_start_idx+len(label_ids)]
            labels[-1] = input_ids[-1] ## eos
        
        return torch.tensor(input_ids),torch.tensor(labels)

    question,answer,task_type = example['question'].strip(),example['answer'].strip(),example['task_type'].strip()
    start_prompt = get_start_prompt(task_type,include_retrieval=use_rag_tuning)
    ret = {}
    
    if use_rag_tuning and use_retriever_embed:
        background = example['background'].strip()
        ret['retriever_input_text'] = [background]

    if use_rag_tuning:
        
        prompt_background = " ".join([XRAG_TOKEN]*retrieval_embed_length)
        
        if use_paraphrase_finetune:
            template = PROMPT_TEMPLATES[task_type][True][True]
            prompt = start_prompt +"\n\n" + template.format_map(dict(question=question,answer=answer,background=prompt_background,real_background=background))
            input_ids,labels = get_input_and_labels(prompt,answer,background)
        else:
            template = PROMPT_TEMPLATES[task_type][True][False]
            prompt = start_prompt +"\n\n" + template.format_map(dict(question=question,answer=answer,background=prompt_background))
            input_ids,labels = get_input_and_labels(prompt,answer)
        ret["xrag_input_ids"] = input_ids.flatten()
        ret['xrag_labels'] = labels.flatten()
        
        ## for traditional-RAG, used as teacher model input
        prompt_background = background
        template = PROMPT_TEMPLATES[task_type][True][False]
        prompt = start_prompt +"\n\n" + template.format_map(dict(question=question,answer=answer,background=prompt_background))
        input_ids,labels = get_input_and_labels(prompt,answer)
        ret["input_ids"] = input_ids.flatten()
        ret['labels'] = labels.flatten()

    else:
        template = PROMPT_TEMPLATES[task_type][False]
        prompt = start_prompt + template.format_map(dict(question=question,answer=answer))
        input_ids,labels = get_input_and_labels(prompt,answer)
        ret["input_ids"] = input_ids.flatten()
        ret['labels'] = labels.flatten()
    
    return ret

def encode_with_completion_format_pretrain(example,tokenizer,max_seq_length,retrieval_embed_length,xrag_token_id):
    document = example['text'].strip()

    ## trick for only calculating loss on the document
    _document = tokenizer.eos_token + document
    xrag_token = " ".join([XRAG_TOKEN]*retrieval_embed_length)
    
    prompt = random.choice(ParaphraseInstructions).strip()
    prompt = prompt.format_map(dict(xrag_token=xrag_token,document=_document))
    
    # prompt = prompt + " " + tokenizer.eos_token

    tokenized_prompt = tokenizer(prompt,max_length=max_seq_length,truncation=True)
    input_ids = tokenized_prompt.input_ids
    # assert len([x for x in input_ids if x==tokenizer.eos_token_id])==2,input_ids
    first_eos_index = input_ids.index(tokenizer.eos_token_id)
    input_ids = input_ids[:first_eos_index] + input_ids[first_eos_index+1:] ## strip the additional eos
    input_ids = torch.tensor(input_ids)
    
    labels = input_ids.clone()
    labels[labels==xrag_token_id] = -100
    labels[:first_eos_index] = -100

    ## maybe we should add some attentino mask in the background part to make it hard for LLM to paraphrase
    return {
        "xrag_input_ids":input_ids.flatten(),
        "xrag_labels":labels.flatten(),
        "retriever_input_text":[document],
    }

def encode_with_completion_format_finetune(
        example, 
        tokenizer,
        max_seq_length,
        retrieval_embed_length,
        use_rag_tuning = True,
        use_retriever_embed=False,
        retriever_tokenizer = None,
        background_dropout_rate=0.0,
        ):
    '''
    Here we assume each example has three fields:
        1) prompt
        2) completion
        3) background  
    '''
    def get_input_and_labels(prompt,completion):
        example_text = prompt + " " + completion # + " " + tokenizer.eos_token
        tokenized_example = tokenizer(example_text,max_length=max_seq_length,truncation=True,return_tensors='pt')
        input_ids = tokenized_example.input_ids
        labels = input_ids.clone()
        tokenized_prompt_length = tokenizer(prompt,max_length=max_seq_length,truncation=True,return_length=True).length[0]
        labels[:,:tokenized_prompt_length]=-100
        return input_ids,labels

    
    # dataset = "_".join(example['id'].split("_")[:-1])
    # if dataset not in ["triviaqa","hotpotqa","nq"]:
    ####### FineTune #######
    original_prompt,completion = example['prompt'].strip(),example['completion'].strip() 
    ret = {}
    
    num_split = 1
    if use_rag_tuning and use_retriever_embed:
        background = example['background'].strip()
        sharded_background = split_background(background,retriever_tokenizer,total_max_len=max_seq_length,single_max_len=180)
        num_split = len(sharded_background)
        ret['retriever_input_text'] = sharded_background

    if use_rag_tuning:
        
        for idx,prompt_background in enumerate([
            " ".join([XRAG_TOKEN]*retrieval_embed_length* num_split),
            background,
        ]):
            prompt = original_prompt
            rag_instruction = random.choice(RAGInstructions).format_map({"background":prompt_background})
            prompt = rag_instruction + prompt
            input_ids,labels = get_input_and_labels(prompt,completion)
            prefix = ""
            if idx == 0: prefix = "xrag_"
            ret[prefix+"input_ids"] = input_ids.flatten()
            ret[prefix+'labels'] = labels.flatten()
    else:
        input_ids,labels = get_input_and_labels(original_prompt,completion)
        ret["input_ids"] = input_ids.flatten()
        ret['labels'] = labels.flatten()
    
    return ret
    
    # else:
    #     ####### Validation #######
    #     question,answer,background = example['prompt'],example['completion'],example['background']
    #     prompt_background = " ".join([XRAG_TOKEN]*retrieval_embed_length)
    #     prompt_dict = {
    #         "background":prompt_background,
    #         "question":question,
    #         "answer":"",
    #     }
    #     prompt = RAG_QA_PROMPT.format_map(prompt_dict).strip()
    #     tokenized_prompt = tokenizer(prompt,max_length=max_seq_length,truncation=True,return_tensors='pt')
        
    #     return {
    #         "xrag_input_ids":tokenized_prompt.input_ids.flatten(),
    #         "retriever_input_text":background,
    #         "answer":answer,
    #     }

QA_PROMPT = "Q: {question}?\nA: {answer}"
RAG_QA_PROMPT = "Background: {background}\n\n"+QA_PROMPT
PARAPHRASE_RAG_QA_PROMPT = "Background: {background}\nThe above background document is just a paraphrase of the following: {real_background}\n\n"+QA_PROMPT

FECT_CHECKING_PROPMT = "Claim: {question}\nAnswer: {answer}"
RAG_FECT_CHECKING_PROPMT = "Background: {background}\n\n" + FECT_CHECKING_PROPMT
PARAPHRASE_RAG_FECT_CHECKING_PROPMT = "Background: {background}\nThe above background document is just a paraphrase of the following: {real_background}\n\n" + FECT_CHECKING_PROPMT

MULTIPLE_CHOICE_PROMPT = "Question: {question}\nAnswer: {answer}"
RAG_MULTIPLE_CHOICE_PROMPT = "Background: {background}\n\n" + MULTIPLE_CHOICE_PROMPT
PARAPHRASE_RAG_MULTIPLE_CHOICE_PROMPT = "Background: {background}\nThe above background document is just a paraphrase of the following: {real_background}\n\n" + MULTIPLE_CHOICE_PROMPT


PROMPT_TEMPLATES = {
    "open_qa":{True:{True:PARAPHRASE_RAG_QA_PROMPT,False:RAG_QA_PROMPT},False:QA_PROMPT},
    'fact_checking':{True:{True:PARAPHRASE_RAG_FECT_CHECKING_PROPMT,False:RAG_FECT_CHECKING_PROPMT},False:FECT_CHECKING_PROPMT},
    'multiple_choice':{True:{True:PARAPHRASE_RAG_MULTIPLE_CHOICE_PROMPT,False:RAG_MULTIPLE_CHOICE_PROMPT},False:MULTIPLE_CHOICE_PROMPT},
}

def get_start_prompt(task_type,include_retrieval):
    if task_type == 'open_qa':
        return {
            True: "Refer to the background document and answer the questions:",
            False:"Answer the questions:"
        }[include_retrieval]
    elif task_type == 'fact_checking':
        return {
            True: "Refer to the background document and verify the following claims with \"True\" or \"False\":",
            False:"Verify the following claims with \"True\" or \"False\":"
        }[include_retrieval]
    elif task_type == 'multiple_choice':
        return {
            True:  f"The following are multiple choice questions (with answers).\nPlease refer to the background document and answer the questions:",
            False: f"The following are multiple choice questions (with answers)."
        }[include_retrieval]

