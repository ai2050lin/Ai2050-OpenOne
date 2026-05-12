"""DS7B加载测试"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import torch
import gc

print(f"CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB", flush=True)

from model_utils import load_model, get_model_info, release_model

print("Loading DS7B...", flush=True)
t0 = time.time()
try:
    model, tokenizer, device = load_model('deepseek7b')
    t1 = time.time()
    print(f"DS7B loaded in {t1-t0:.1f}s, device={device}", flush=True)
    
    info = get_model_info(model, 'deepseek7b')
    print(f"Info: {info.model_class}, {info.n_layers}L, d={info.d_model}", flush=True)
    
    # 简单forward测试
    prompt = "The cat sat on the"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32)
    input_device = next(model.parameters()).device
    input_ids = inputs["input_ids"].to(input_device)
    attn_mask = inputs["attention_mask"].to(input_device)
    
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
    
    print(f"Forward OK: {len(out.hidden_states)} layers, last norm={out.hidden_states[-1].float().norm():.2f}", flush=True)
    
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    print("DS7B test PASSED!", flush=True)
    
except Exception as e:
    print(f"ERROR: {e}", flush=True)
    import traceback
    traceback.print_exc()
