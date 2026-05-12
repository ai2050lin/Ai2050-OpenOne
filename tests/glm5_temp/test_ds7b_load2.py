"""DS7B加载测试 - 带超时和详细日志"""
import sys, time, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import torch
print(f"PyTorch: {torch.__version__}", flush=True)
print(f"CUDA: {torch.version.cuda}", flush=True)
print(f"GPU: {torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'N/A'}", flush=True)
print(f"GPU mem: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB", flush=True)

sys.path.insert(0, 'tests/glm5')
from model_utils import MODEL_CONFIGS

model_path = MODEL_CONFIGS['deepseek7b']['path']
print(f"Model path: {model_path}", flush=True)
print(f"Path exists: {os.path.exists(model_path)}", flush=True)

# 列出模型文件
import glob
files = glob.glob(os.path.join(model_path, '*.safetensors'))
print(f"Safetensor files: {len(files)}", flush=True)
for f in files:
    size_gb = os.path.getsize(f) / 1e9
    print(f"  {os.path.basename(f)}: {size_gb:.2f} GB", flush=True)

# 尝试8bit加载
print("\nAttempting 8bit load...", flush=True)
t0 = time.time()

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True, use_fast=False
    )
    print(f"Tokenizer loaded in {time.time()-t0:.1f}s", flush=True)
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    
    print("Loading model with 8bit...", flush=True)
    t1 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    t2 = time.time()
    print(f"Model loaded in {t2-t1:.1f}s!", flush=True)
    print(f"Model class: {type(model).__name__}", flush=True)
    
    # 简单forward
    input_ids = tokenizer.encode("Hello world", return_tensors="pt").to(next(model.parameters()).device)
    with torch.no_grad():
        out = model(input_ids, output_hidden_states=True)
    print(f"Forward OK: {len(out.hidden_states)} layers", flush=True)
    
    del model
    import gc; gc.collect()
    torch.cuda.empty_cache()
    print("SUCCESS!", flush=True)
    
except Exception as e:
    print(f"FAILED: {e}", flush=True)
    import traceback
    traceback.print_exc()
