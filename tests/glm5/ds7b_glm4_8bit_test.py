"""
DS7B/GLM4 8bit模型加载测试
==========================
8bit量化加载: GPU显存需求减半, 12GB GPU可运行DS7B/GLM4

用法:
  python tests/glm5/ds7b_glm4_8bit_test.py deepseek7b
  python tests/glm5/ds7b_glm4_8bit_test.py glm4

环境:
  PyTorch 2.10.0+cu130, transformers 4.51.0, bitsandbytes 0.49.2
  RTX 5070 (12GB), Windows 11

注意: from_pretrained的8bit加载可能需要30秒-2分钟, 请耐心等待
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import gc
import time
import torch
import numpy as np

MODEL_CONFIGS = {
    "deepseek7b": {
        "path": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "arch": "Qwen2ForCausalLM",
    },
    "glm4": {
        "path": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "arch": "GlmForCausalLM",
    },
}


def load_model_8bit(model_name):
    """8bit量化加载模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    cfg = MODEL_CONFIGS[model_name]
    
    print(f"Loading {model_name} tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Loading {model_name} model (8bit, may take 30-120s)...", flush=True)
    t0 = time.time()
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    t_load = time.time() - t0
    model.eval()
    
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9
    
    # 基本信息
    config = model.config
    n_layers = config.num_hidden_layers
    d_model = config.hidden_size
    n_heads = config.num_attention_heads
    
    print(f"Loaded in {t_load:.1f}s: device={device}, GPU={gpu_mem:.2f}GB", flush=True)
    print(f"  {n_layers} layers, d_model={d_model}, n_heads={n_heads}", flush=True)
    
    return model, tokenizer, device


def test_full(model, tokenizer, device, model_name):
    """完整测试: 前向推理 + 生成 + 权重访问 + Hook"""
    cfg = MODEL_CONFIGS[model_name]
    
    # 1. 前向推理
    print("\n[1] Forward pass...", flush=True)
    prompt = "The scientist discovered a new"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    t_fwd = time.time() - t0
    hs = out.hidden_states
    print(f"  {t_fwd:.2f}s, {len(hs)} layers, last norm={hs[-1].float().norm():.2f}", flush=True)
    
    # Top-5
    logits = out.logits[0, -1].float().cpu().numpy()
    top5_ids = np.argsort(logits)[-5:][::-1]
    top5 = [(tokenizer.decode([i]).strip(), float(logits[i])) for i in top5_ids]
    print(f"  Top-5: {top5}", flush=True)
    
    # 2. 生成
    print("[2] Generate...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        gen_ids = model.generate(input_ids, attention_mask=attention_mask,
                                  max_new_tokens=20, do_sample=False)
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    t_gen = time.time() - t0
    print(f"  {t_gen:.1f}s: '{gen_text}'", flush=True)
    
    # 3. 权重访问 (8bit dequantize)
    print("[3] Weight access (8bit dequantize)...", flush=True)
    layers = model.model.layers
    try:
        W_q = layers[0].self_attn.q_proj.weight.detach().float().cpu().numpy()
        W_o = layers[0].self_attn.o_proj.weight.detach().float().cpu().numpy()
        print(f"  W_q: {W_q.shape}, norm={np.linalg.norm(W_q):.2f}", flush=True)
        print(f"  W_o: {W_o.shape}, norm={np.linalg.norm(W_o):.2f}", flush=True)
    except Exception as e:
        print(f"  Weight access failed: {e}", flush=True)
    
    try:
        W_U = model.lm_head.weight.detach().float().cpu().numpy()
        print(f"  W_U: {W_U.shape}, norm={np.linalg.norm(W_U):.2f}", flush=True)
    except Exception as e:
        print(f"  W_U access failed: {e}", flush=True)
    
    # 4. Hook提取中间层
    print("[4] Hook extraction...", flush=True)
    n_layers = len(layers)
    captured = {}
    def make_hook(key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().float().cpu()
            else:
                captured[key] = output.detach().float().cpu()
        return hook
    
    hook_layers = [0, n_layers // 2, n_layers - 1]
    hooks = [layers[li].register_forward_hook(make_hook(f"L{li}")) for li in hook_layers]
    
    with torch.no_grad():
        model(input_ids=input_ids, attention_mask=attention_mask)
    for h in hooks:
        h.remove()
    
    for key in sorted(captured.keys()):
        t = captured[key]
        print(f"  Hook {key}: shape={t.shape}, norm={t.float().norm():.2f}", flush=True)
    
    # 5. GPU内存
    gpu_mem = torch.cuda.memory_allocated() / 1e9
    print(f"[5] GPU memory: {gpu_mem:.2f} GB", flush=True)
    
    return {
        "fwd_time": round(t_fwd, 2),
        "gen_time": round(t_gen, 1),
        "top5": top5,
        "gpu_mem_gb": round(gpu_mem, 2),
        "n_layers": len(hs),
    }


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "deepseek7b"
    assert model_name in MODEL_CONFIGS, f"Unknown model: {model_name}"
    
    t0 = time.time()
    model, tokenizer, device = load_model_8bit(model_name)
    t_load = time.time() - t0
    
    result = test_full(model, tokenizer, device, model_name)
    print(f"\n{'='*60}", flush=True)
    print(f"Result: {result}", flush=True)
    print(f"Total time: {time.time()-t0:.1f}s", flush=True)
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU after release: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
    print("Done!", flush=True)
