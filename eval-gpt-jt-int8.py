# Tested multiple version of transformers seems only working version is 4.22.1
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
import json
import tqdm
import torch
from transformers import AutoConfig, AutoTokenizer
from transformers import AutoModelForCausalLM
# import bitsandbytes as bnb
import time 

model_name = "togethercomputer/GPT-JT-6B-v1"
# model_name = "EleutherAI/gpt-j-6B"
# model_name = "bigscience/bloom-3b"
mode = "int8"

print(f"Benchmark <{model_name}> mode: {mode}")

config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', load_in_8bit=True, int8_threshold=6.0)

print("Loading model starts.")
# model = model.eval().cuda()
print("Loading model is done.")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.pad_token_id = tokenizer.eos_token_id
tokenizer.add_bos_token = False
tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer('Where is Zurich?', return_tensors="pt")
outputs = model.generate(input_ids=inputs['input_ids'].cuda(), do_sample=True, max_new_tokens=16)
output_texts = tokenizer.batch_decode(outputs[0], skip_special_tokens=True)
print(f"<foo demo> output:", output_texts)


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
    start_time = time.time()
    acc = evaluate_acc(f'./raft-v8/raft:subset={task_name},model=together_gpt-j-6b,data_augmentation=canonical/scenario_state.json', max_new_tokens=20)
    end_time = time.time()
    print(f"{task_name} accuracy: {acc}, time: {end_time-start_time}")
