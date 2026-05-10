"""Minimal test: verify DS7B 8bit loading works."""
import torch
import time

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Free GPU memory: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

path = "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
print(f"\nLoading tokenizer from {path}...")
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer loaded.")

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

print("Loading model (8bit)...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    path, quantization_config=bnb_config, device_map="auto",
    trust_remote_code=True, local_files_only=True,
)
model.eval()
print(f"Model loaded in {time.time()-t0:.1f}s")

cfg = model.config
print(f"n_layers={cfg.num_hidden_layers}, d_model={cfg.hidden_size}, n_heads={cfg.num_attention_heads}")
print(f"GPU memory used: {torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]:.1f} GB")

# Quick forward pass
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print(f"\nRunning forward pass on: '{prompt}'")
with torch.no_grad():
    outputs = model(inputs['input_ids'], attention_mask=inputs.get('attention_mask'), output_hidden_states=True)

print(f"Number of hidden states: {len(outputs.hidden_states)}")
print(f"Hidden state shape: {outputs.hidden_states[0].shape}")

# Get logits
next_token_logits = outputs.logits[0, -1, :]
top5 = torch.topk(next_token_logits, 5)
print(f"\nTop 5 predictions:")
for token_id, score in zip(top5.indices, top5.values):
    print(f"  {tokenizer.decode(token_id)}: {score:.2f}")

print("\nDS7B 8bit loading test COMPLETE!")
