"""Test DeepSeek7B 8bit loading"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

print("Starting 8bit load test...", flush=True)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

cfg_path = 'D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(cfg_path, trust_remote_code=True, local_files_only=True, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"Tokenizer loaded, vocab={len(tokenizer)}", flush=True)

print("Loading model in 8bit...", flush=True)
bnb = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
model = AutoModelForCausalLM.from_pretrained(
    cfg_path, quantization_config=bnb, device_map='auto',
    trust_remote_code=True, local_files_only=True,
)
model.eval()
device = next(model.parameters()).device
print(f"Model loaded: {type(model).__name__}, device={device}", flush=True)

# Quick test
prompt = 'Translate the word "cat" into Chinese.'
inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True, max_length=64)
input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)

print("Running inference test...", flush=True)
with torch.no_grad():
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
print(f"Inference OK, {len(out.hidden_states)} layers, last_hidden shape={out.hidden_states[-1].shape}", flush=True)

# Memory
print(f"GPU allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
print(f"GPU reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB", flush=True)

# Clean up
del model
import gc; gc.collect()
torch.cuda.empty_cache()
print("Cleanup done.", flush=True)
