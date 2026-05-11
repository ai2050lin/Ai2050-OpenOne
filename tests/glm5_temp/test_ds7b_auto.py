"""测试DS7B加载 - device_map=auto方式"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import time, torch, gc
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True, use_fast=False)
print("Tokenizer OK")

print("Loading model with device_map=auto...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    local_files_only=True,
    low_cpu_mem_usage=True,
)
t1 = time.time()
print(f"Model loaded in {t1-t0:.1f}s")
if hasattr(model, "hf_device_map"):
    print(f"Device map: {model.hf_device_map}")
else:
    print("No hf_device_map attribute")

device = next(model.parameters()).device
print(f"Model device: {device}")

model.eval()
inputs = tokenizer("Hello world", return_tensors="pt", truncation=True, max_length=16).to(device)
with torch.no_grad():
    out = model(**inputs, output_hidden_states=True)
print(f"Forward OK, layers={len(out.hidden_states)}")

gpu_mem = torch.cuda.memory_allocated() / 1e9
print(f"GPU memory: {gpu_mem:.2f} GB")

del model; gc.collect(); torch.cuda.empty_cache()
print("Done!")
