"""
验证 DeepSeek 7B BF16 精度 + CPU/GPU混合加载的可行性
测试: 1) 模型加载 2) Hook提取激活值 3) 基本推理
"""
import torch
import time
import json

def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_path = "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    
    print("=" * 60)
    print("DeepSeek 7B BF16 + device_map='auto' 验证")
    print("=" * 60)
    
    # 1. 加载tokenizer
    print("\n[1] 加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  vocab_size = {len(tokenizer)}")
    
    # 2. 加载模型 - BF16 + device_map='auto'
    print("\n[2] 加载模型 (BF16, device_map='auto')...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    t_load = time.time() - t0
    print(f"  加载耗时: {t_load:.1f}s")
    
    # 3. 查看device_map分配
    print("\n[3] device_map 分配情况:")
    if hasattr(model, 'hf_device_map'):
        device_map = model.hf_device_map
        gpu_layers = sum(1 for v in device_map.values() if 'cuda' in str(v))
        cpu_layers = sum(1 for v in device_map.values() if 'cpu' in str(v))
        print(f"  GPU层: {gpu_layers}, CPU层: {cpu_layers}")
        # 显示关键层的设备
        for k, v in list(device_map.items())[:5]:
            print(f"    {k}: {v}")
        print(f"    ...")
        for k, v in list(device_map.items())[-3:]:
            print(f"    {k}: {v}")
    
    # 4. GPU内存占用
    print("\n[4] GPU内存占用:")
    if torch.cuda.is_available():
        print(f"  已分配: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"  已保留: {torch.cuda.memory_reserved()/1e9:.2f} GB")
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU总VRAM: {total_vram:.1f} GB")
    
    # 5. Hook测试 - 提取residual stream
    print("\n[5] Hook提取激活值测试:")
    activations = {}
    
    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # residual stream 是 input
            if isinstance(input, tuple) and len(input) > 0:
                act = input[0].detach().cpu()
                activations[f'layer_{layer_idx}'] = {
                    'shape': list(act.shape),
                    'mean': act.float().mean().item(),
                    'std': act.float().std().item(),
                    'device_before_hook': str(module.device) if hasattr(module, 'device') else 'unknown',
                }
        return hook_fn
    
    hooks = []
    n_layers = model.config.num_hidden_layers
    for i in range(n_layers):
        h = model.model.layers[i].register_forward_hook(make_hook(i))
        hooks.append(h)
    
    # 6. 推理测试
    print("\n[6] 推理测试:")
    test_text = "The capital of France is"
    inputs = tokenizer(test_text, return_tensors="pt")
    # 移动input到模型所在设备
    input_device = next(model.parameters()).device
    inputs = {k: v.to(input_device) for k, v in inputs.items()}
    
    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    t_gen = time.time() - t0
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  输入: {test_text}")
    print(f"  输出: {generated}")
    print(f"  生成耗时: {t_gen:.2f}s")
    
    # 7. 激活值检查
    print("\n[7] 激活值统计:")
    for key in sorted(activations.keys(), key=lambda x: int(x.split('_')[1])):
        a = activations[key]
        print(f"  {key}: shape={a['shape']}, mean={a['mean']:.4f}, std={a['std']:.4f}")
    
    # 清理hooks
    for h in hooks:
        h.remove()
    
    # 8. 批量推理速度测试
    print("\n[8] 批量推理速度测试 (5条文本):")
    test_texts = [
        "The sky is",
        "Water boils at",
        "2 + 3 =",
        "The largest planet is",
        "In Chinese, hello is",
    ]
    total_tokens = 0
    t0 = time.time()
    for txt in test_texts:
        inp = tokenizer(txt, return_tensors="pt")
        inp = {k: v.to(input_device) for k, v in inp.items()}
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=15, do_sample=False)
        total_tokens += out.shape[1] - inp['input_ids'].shape[1]
    t_batch = time.time() - t0
    print(f"  总生成token数: {total_tokens}")
    print(f"  总耗时: {t_batch:.2f}s")
    print(f"  速度: {total_tokens/t_batch:.1f} tokens/s")
    
    # 9. 对比: 8bit加载
    print("\n[9] 对比: 8bit加载 (同一模型):")
    del model
    torch.cuda.empty_cache()
    
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
    t0 = time.time()
    model_8bit = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    model_8bit.eval()
    t_load_8bit = time.time() - t0
    print(f"  8bit加载耗时: {t_load_8bit:.1f}s")
    print(f"  8bit GPU内存: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # 8bit推理速度
    total_tokens = 0
    t0 = time.time()
    for txt in test_texts:
        inp = tokenizer(txt, return_tensors="pt")
        inp = {k: v.to(next(model_8bit.parameters()).device) for k, v in inp.items()}
        with torch.no_grad():
            out = model_8bit.generate(**inp, max_new_tokens=15, do_sample=False)
        total_tokens += out.shape[1] - inp['input_ids'].shape[1]
    t_batch_8bit = time.time() - t0
    print(f"  8bit速度: {total_tokens/t_batch_8bit:.1f} tokens/s")
    
    # 汇总
    print("\n" + "=" * 60)
    print("汇总对比:")
    print(f"  BF16 GPU占用: ~{torch.cuda.memory_allocated()/1e9:.1f} GB (含8bit模型)")
    print(f"  BF16速度: {total_tokens/t_batch:.1f} tokens/s (估计)")
    print(f"  8bit速度: {total_tokens/t_batch_8bit:.1f} tokens/s")
    print(f"  BF16优势: 激活值全精度，无损提取")
    print("=" * 60)


if __name__ == "__main__":
    main()
