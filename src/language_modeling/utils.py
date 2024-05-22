import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os




def get_nll_loss(logits,labels,vocab_size):
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = nn.CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss

def get_kl_loss(teacher_logits,student_logits,student_labels,teacher_labels,temperature,distill_topk=None):
    
    ## make sure the teacher_logits and student_logits have the same shape
    loss_fct = nn.KLDivLoss(reduction="batchmean")
    _,_,vocab_size = student_logits.shape

    ## only compute loss in the completion part, not propmt
    
    student_mask = (student_labels!=-100).unsqueeze(-1).expand_as(student_logits) ## batch_size,num_tokens,vocab_size
    student_logits_selected = torch.masked_select(student_logits,student_mask).view(-1,vocab_size)

    teacher_mask = (teacher_labels != -100).unsqueeze(-1).expand_as(teacher_logits)
    teacher_logits_selected = torch.masked_select(teacher_logits,teacher_mask).view(-1,vocab_size)

    if distill_topk is not None:
        _, topk_teacher_indices = torch.topk(teacher_logits_selected, k=distill_topk, dim=-1)  
      
        teacher_logits_selected = torch.gather(teacher_logits_selected, 1, topk_teacher_indices)  
        student_logits_selected = torch.gather(student_logits_selected, 1, topk_teacher_indices) 

    assert teacher_logits_selected.shape == student_logits_selected.shape, (f"The shape of teacher logits is {teacher_logits_selected.shape}, while that of student is {student_logits_selected.shape}")

    kl_loss = loss_fct(
        F.log_softmax(student_logits_selected / temperature, dim=-1),
        F.softmax(    teacher_logits_selected / temperature, dim=-1),
    ) * temperature ** 2
    
    return kl_loss


def encode_with_messages_format(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    '''
    messages = example['messages']
    if len(messages) == 0:
        raise ValueError('messages field is empty.')
    
    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text
        
    example_text = _concat_messages(messages).strip()
    tokenized_example = tokenizer(example_text, max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = copy.copy(input_ids)

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]), max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx+1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = _concat_messages(messages[:message_idx+1]) + "<|assistant|>\n"
            else:
                messages_so_far = _concat_messages(messages[:message_idx+1])
            message_end_idx = tokenizer(
                messages_so_far,
                return_tensors='pt', 
                max_length=max_seq_length, 
                truncation=True
            ).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100
            
            if message_end_idx >= max_seq_length:
                break

    # attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids,
        'labels': labels,
        # 'attention_mask': attention_mask.flatten(),
    }

def encode_with_prompt_completion_format(example, tokenizer, max_seq_length):
    '''
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated 
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space
    prompt = example['prompt']
    completion = example['completion']

    background = example['background']
    background_embedding = example['background_embedding']

    prompt = f"Background: {background}\n\n{prompt}"

    prompt = prompt.strip()
    completion = completion.strip()

    if not prompt.endswith((' ', '\n', '\t')) and not completion.startswith((' ', '\n', '\t')):
        example_text = prompt + ' ' + completion
    else:
        example_text = prompt + completion

    example_text = example_text + tokenizer.eos_token
    tokenized_example = tokenizer(example_text, max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = copy.copy(input_ids)
    tokenized_prompt_length = tokenizer(prompt, max_length=max_seq_length, truncation=True,return_length=True).length
    # mask the prompt part for avoiding loss
    labels[:tokenized_prompt_length] = [-100]*tokenized_prompt_length
    # attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids,
        'labels': labels,
        "background_embedding":background_embedding,
        # 'attention_mask': attention_mask.flatten(),
    }



def save_with_accelerate(accelerator, model, tokenizer, output_dir, save_projector_only=False):
    
    unwrapped_model = accelerator.unwrap_model(model)

    if save_projector_only:    
            params_to_save = {
                n:p.float() for n,p in unwrapped_model.named_parameters() 
                if any(
                    sub_string in n 
                    for sub_string in ['embed_tokens','projector','lm_head']
                    )
                }
            if accelerator.is_main_process:
                os.makedirs(output_dir)
                torch.save(params_to_save, os.path.join(output_dir,'ckpt.pth'))
                unwrapped_model.config.save_pretrained(output_dir)

    else:    
        # When doing multi-gpu training, we need to use accelerator.get_state_dict(model) to get the state_dict.
        # Otherwise, sometimes the model will be saved with only part of the parameters.
        # Also, accelerator needs to use the wrapped model to get the state_dict.
        state_dict = accelerator.get_state_dict(model)

        unwrapped_model.save_pretrained(
            output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, state_dict=state_dict,
            safe_serialization=False, ## safetensors is buggy for now
        )

    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)

XRAG_TOKEN = "<xRAG>" 

ParaphraseInstructions = [
    'Background: {xrag_token} means the same as',
    "Background: {xrag_token} Can you put the above sentences in your own terms?",
    "Background: {xrag_token} Please provide a reinterpretation of the preceding background text.",
    "These two expressions are equivalent in essence:\n(1) {xrag_token}\n(2)",
    "Background: {xrag_token} is a paraphrase of what?",
    "Background: {xrag_token} Could you give me a different version of the background sentences above?",
    "In other words, background: {xrag_token} is just another way of saying:",
    "You're getting across the same point whether you say background: {xrag_token} or",
    "Background: {xrag_token} After uppacking the ideas in the background information above, we got:",
    "Background: {xrag_token} Please offer a restatement of the background sentences I've just read.",
    "Background: {xrag_token}, which also means:",
    "Strip away the mystery, and you'll find background: {xrag_token} is simply another rendition of:",
    "The essence of background: {xrag_token} is captured again in the following statement:",
]

# Refer to the background document and silently paraphrase its content.
RAGInstructions = [
    "Refer to the background document and answer the questions.\nBackground: {background}\n",
    "Background: {background}\n",
    "To provide accurate answers, it's essential to consider the background information presented here. Contextual Background: {background}\n",
    "Background Details: {background}\n",
    "The following background will help you understand the context for the questions. Please read it carefully before responding. Background: {background}\n",
    "Background: {background}\nYou might find the above background documents helpful.\n",
    ]



def get_retrieval_embeds(model,input_ids,attention_mask=None):
    with torch.no_grad():
        embeds = model.get_doc_embedding(
            input_ids = input_ids,
            attention_mask = attention_mask,
        )
    embeds = embeds.view(-1,embeds.shape[-1])
    return embeds 

def calculate_grad_norm(model, norm_type=2):  
    total_norm = 0  
    for p in model.parameters():  
        if p.grad is not None:  
            param_norm = p.grad.data.norm(norm_type)  
            total_norm += param_norm.item() ** norm_type  
    total_norm = total_norm ** (1. / norm_type)  
    return total_norm


def find_matched_index(main_seq, sub_seq):  
    # Lengths of the sequences  
    assert len(sub_seq)>0 and len(main_seq)>0, f"the input should not be empty, however {sub_seq=}\n {main_seq=}"
    main_len = len(main_seq)  
    sub_len = len(sub_seq)  
  
    # Early exit if sub_seq is longer than main_seq  
    if sub_len > main_len:  
        return -1  
  
    # Variable to keep track of the last index of a match  
    last_index = -1  
  
    # Iterate through main_seq to find sub_seq  
    for i in range(main_len - sub_len + 1):  
        # Check if the slice of main_seq matches sub_seq  
        if main_seq[i:i+sub_len] == sub_seq:  
            # Update the last_index to the current position  
            last_index = i  
  
    # Return the last index found or -1 if not found  
    return last_index  