"""DS7B加载测试 — 8bit方式, 带详细进度"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import time
import torch
from model_utils import MODEL_CONFIGS

print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")

try:
    import bitsandbytes as bnb
    print(f"bitsandbytes: {bnb.__version__}")
except:
    print("bitsandbytes: NOT INSTALLED!")

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

cfg = MODEL_CONFIGS["deepseek7b"]

print("\n8bit加载DS7B...")
t0 = time.time()
try:
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    print(f"Tokenizer loaded ({time.time()-t0:.1f}s)")
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    
    print("Loading model with 8bit...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    t1 = time.time()
    print(f"8bit加载耗时: {t1-t0:.1f}s")
    model.eval()
    
    print(f"device: {next(model.parameters()).device}")
    print(f"GPU mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    del model
    import gc; gc.collect(); torch.cuda.empty_cache()
    print("完成!")
    
except Exception as e:
    print(f"失败: {e}")
    import traceback; traceback.print_exc()
