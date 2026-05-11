"""
临时脚本: 测试DS7B和GLM4的8-bit加载
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_load_8bit(model_name):
    configs = {
        "deepseek7b": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "glm4": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
    }
    path = configs[model_name]
    print(f"\n=== Testing {model_name} (8-bit) ===")
    
    try:
        t0 = time.time()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, local_files_only=True, use_fast=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 尝试8-bit加载
        model = AutoModelForCausalLM.from_pretrained(
            path,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
        model.eval()
        t_load = time.time() - t0
        print(f"  Loaded in {t_load:.1f}s")
        
        # 简单前向
        prompt = "Translate the word cat into Chinese."
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs["attention_mask"].to(model.device)
        
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        n_layers = len(out.hidden_states)
        d_model = out.hidden_states[0].shape[-1]
        print(f"  Forward OK: {n_layers} layers, d_model={d_model}")
        
        gpu_mem = torch.cuda.memory_allocated() / 1e9
        print(f"  GPU memory: {gpu_mem:.2f} GB")
        
        # 释放
        del model; gc.collect(); torch.cuda.empty_cache()
        print(f"  Released. {model_name} test PASSED!")
        return True
        
    except Exception as e:
        print(f"  FAILED: {e}")
        import traceback; traceback.print_exc()
        try:
            del model; gc.collect(); torch.cuda.empty_cache()
        except:
            pass
        return False

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "deepseek7b"
    test_load_8bit(model)
