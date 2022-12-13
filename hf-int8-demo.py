from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

name = "EleutherAI/gpt-j-6B"
text = "Hello my name is"
max_new_tokens = 20

def generate_from_model(model, tokenizer):
    encoded_input = tokenizer(text, return_tensors='pt')
    output_sequences = model.generate(input_ids=encoded_input['input_ids'].cuda())
    return tokenizer.decode(output_sequences[0], skip_special_tokens=True)

#pipe = pipeline(model=name, model_kwargs= {"device_map": "auto", "load_in_8bit": True}, max_new_tokens=max_new_tokens)
#generated_text = pipe(text)
#print("generated_text:", generated_text)


model_8bit = AutoModelForCausalLM.from_pretrained(name, device_map="auto", load_in_8bit=True)
tokenizer = AutoTokenizer.from_pretrained(name)

generate_from_model(model_8bit, tokenizer)