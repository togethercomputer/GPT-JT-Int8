#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import json
import tqdm

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers import AutoModelForCausalLM

config = AutoConfig.from_pretrained('EleutherAI/gpt-j-6B')
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')
model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-j-6B')

model = model.half().eval().cuda()


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.pad_token_id = tokenizer.eos_token_id
tokenizer.add_bos_token = False


def infer(prompt, max_new_tokens=1):
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors='pt')
        inputs = {
            k: v.to(model.device) for k,v in inputs.items()
        }
        ret = model.generate(**inputs, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id)

        return tokenizer.decode(ret[0, inputs['input_ids'].size(1):])


def evaluate_acc(task_path, max_new_tokens=20):
    
    with open(task_path) as f:
        state_dict = json.load(f)
    
    n_c = 0
    n_t = 0
    for request in tqdm.tqdm(state_dict['request_states']):
        
        # skip perturbated ones
        if 'perturbation' in request['instance']:
            continue
        
        prompt = request['request']['prompt']
        label = request['instance']['references'][0]['output']
        pred = infer(prompt, max_new_tokens=max_new_tokens).strip().split('\n')[0]
        
        if label == pred:
            n_c += 1
        n_t += 1
        
    return n_c / n_t


tasks_list = [
    'ade_corpus_v2',
    'banking_77',
    'neurips_impact_statement_risks',
    'one_stop_english',
    'overruling',
    'semiconductor_org_types',
    'systematic_review_inclusion',
    'tai_safety_research',
    'terms_of_service',
    'tweet_eval_hate',
    'twitter_complaints',
]
for task_name in tasks_list:
    acc = evaluate_acc(f'./raft-v8/raft:subset={task_name},model=together_gpt-j-6b,data_augmentation=canonical/scenario_state.json', max_new_tokens=20)
    print(f"{task_name} accuracy: {acc}")
    

