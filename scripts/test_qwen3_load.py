"""
Qwen3 加载测试脚本
解决 tokenizer 配置问题
"""

import os
import sys

# 设置 HuggingFace 环境
os.environ["HF_HOME"] = r"D:\develop\model"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_qwen3_load():
    print("=" * 60)
    print("Qwen3-4B 加载测试")
    print("=" * 60)
    
    model_name = "Qwen/Qwen2.5-0.5B"  # 先用小模型测试
    
    print(f"\n[1] 检查 GPU 显存...")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU: {gpu_name}")
        print(f"  显存: {gpu_mem:.1f} GB")
    else:
        print("  警告: CUDA 不可用，将使用 CPU")
    
    print(f"\n[2] 加载 Tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            # 关键修复：设置 pad_token
            padding_side='left'
        )
        
        # 修复 bos_token 问题
        if tokenizer.bos_token is None:
            tokenizer.bos_token = tokenizer.eos_token
            tokenizer.bos_token_id = tokenizer.eos_token_id
            print(f"  已设置 bos_token = eos_token")
        
        print(f"  [OK] Tokenizer 加载成功")
        print(f"  vocab_size: {len(tokenizer)}")
        print(f"  bos_token: {tokenizer.bos_token}")
        print(f"  eos_token: {tokenizer.eos_token}")
    except Exception as e:
        print(f"  [FAIL] Tokenizer 加载失败: {e}")
        return False
    
    print(f"\n[3] 加载模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # 方法1: 自动量化（推荐 8GB 显存）
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto",
            # 8GB 显存使用这个配置
            low_cpu_mem_usage=True,
        )
        
        print(f"  [OK] 模型加载成功")
        print(f"  设备: {device}")
        print(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
        
        # 显示显存使用
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"  显存已分配: {allocated:.2f} GB")
            print(f"  显存已保留: {reserved:.2f} GB")
        
    except Exception as e:
        print(f"  [FAIL] 模型加载失败: {e}")
        return False
    
    print(f"\n[4] 测试推理...")
    try:
        test_text = "The cat sat on the"
        inputs = tokenizer(test_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False
            )
        
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  输入: {test_text}")
        print(f"  输出: {result}")
        print(f"  [OK] 推理成功")
        
    except Exception as e:
        print(f"  [FAIL] 推理失败: {e}")
        return False
    
    print(f"\n[5] 尝试加载 Qwen2.5-3B...")
    model_name_3b = "Qwen/Qwen2.5-3B"  # 3B 版本，8GB 显存可以运行
    
    try:
        print(f"  加载 {model_name_3b}...")
        
        tokenizer_3b = AutoTokenizer.from_pretrained(
            model_name_3b,
            trust_remote_code=True,
            padding_side='left'
        )
        if tokenizer_3b.bos_token is None:
            tokenizer_3b.bos_token = tokenizer_3b.eos_token
            tokenizer_3b.bos_token_id = tokenizer_3b.eos_token_id
        
        # 释放之前的模型
        del model
        torch.cuda.empty_cache()
        
        model_3b = AutoModelForCausalLM.from_pretrained(
            model_name_3b,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        
        print(f"  [OK] {model_name_3b} 加载成功")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"  显存使用: {allocated:.2f} GB")
            
            if allocated > 7.5:
                print("  [WARN] 显存接近上限，建议使用更小的模型或量化")
        
    except Exception as e:
        print(f"  [FAIL] 加载失败: {e}")
        print("  建议：尝试 Qwen2.5-0.5B 或使用 4-bit 量化")
        return False
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Qwen 加载测试完成")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_qwen3_load()
    sys.exit(0 if success else 1)
