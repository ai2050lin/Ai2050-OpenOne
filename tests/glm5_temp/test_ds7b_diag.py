"""DS7B加载诊断脚本"""
import sys, time, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}", flush=True)
print(f"GPU: {torch.cuda.get_device_properties(0).name}, {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB", flush=True)

sys.path.insert(0, 'tests/glm5')
from model_utils import MODEL_CONFIGS

model_path = MODEL_CONFIGS['deepseek7b']['path']
print(f"Model path: {model_path}", flush=True)
print(f"Path exists: {os.path.exists(model_path)}", flush=True)

import glob
sf_files = glob.glob(os.path.join(model_path, '*.safetensors'))
print(f"Safetensor files: {len(sf_files)}", flush=True)
for f in sf_files:
    size_gb = os.path.getsize(f) / 1e9
    print(f"  {os.path.basename(f)}: {size_gb:.2f}GB", flush=True)

print("\n--- Test 1: 8bit loading ---", flush=True)
t0 = time.time()
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True, use_fast=False)
    print(f"Tokenizer loaded in {time.time()-t0:.1f}s", flush=True)
    
    print("Configuring 8bit...", flush=True)
    bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
    
    print("Loading model (8bit)...", flush=True)
    t1 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path, quantization_config=bnb_config, device_map="auto",
        trust_remote_code=True, local_files_only=True, attn_implementation="eager"
    )
    t2 = time.time()
    print(f"Model loaded in {t2-t1:.1f}s! Class: {type(model).__name__}", flush=True)
    
    # Quick test
    print("Running forward pass...", flush=True)
    ids = tokenizer.encode("Hello world", return_tensors="pt")
    dev = next(model.parameters()).device
    ids = ids.to(dev)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    print(f"Forward OK: {len(out.hidden_states)} layers", flush=True)
    
    del model
    import gc; gc.collect(); torch.cuda.empty_cache()
    print("8bit test PASSED!", flush=True)
    
except Exception as e:
    print(f"8bit test FAILED: {e}", flush=True)
    import traceback
    traceback.print_exc()
    # 清理
    gc.collect(); torch.cuda.empty_cache()

print("\nDone!", flush=True)
