"""DS7B 8bit加载 - 等待足够长时间"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import time
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

path = 'D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'

print('Loading tokenizer...', flush=True)
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print('Tokenizer OK', flush=True)

print('Loading DS7B 8bit (this may take several minutes)...', flush=True)
t0 = time.time()
bnb = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
model = AutoModelForCausalLM.from_pretrained(
    path, quantization_config=bnb, device_map='auto',
    trust_remote_code=True, local_files_only=True
)
t_load = time.time() - t0
model.eval()
device = next(model.parameters()).device
gpu = torch.cuda.memory_allocated() / 1e9
print(f'Loaded in {t_load:.1f}s! device={device}, GPU={gpu:.2f}GB', flush=True)

# 推理
print('Testing inference...', flush=True)
inputs = tokenizer('The scientist discovered a new', return_tensors='pt', truncation=True, max_length=64)
input_ids = inputs['input_ids'].to(device)
attention_mask = inputs['attention_mask'].to(device)

t0 = time.time()
with torch.no_grad():
    out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
t_fwd = time.time() - t0
hs = out.hidden_states
logits = out.logits[0, -1].float().cpu().numpy()
top5_ids = np.argsort(logits)[-5:][::-1]
top5 = [(tokenizer.decode([i]).strip(), float(logits[i])) for i in top5_ids]
print(f'Forward: {t_fwd:.2f}s, {len(hs)} layers, last norm={hs[-1].float().norm():.2f}', flush=True)
print(f'Top-5: {top5}', flush=True)

# 生成
t0 = time.time()
with torch.no_grad():
    gen_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=20, do_sample=False)
gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
t_gen = time.time() - t0
print(f'Generate ({t_gen:.1f}s): {gen_text}', flush=True)

# 释放
del model
import gc; gc.collect(); torch.cuda.empty_cache()
print(f'GPU after release: {torch.cuda.memory_allocated()/1e9:.2f}GB', flush=True)
print('Done!', flush=True)
