"""
DS7B 8bit加载测试 - 参考5月5号验证通过的方式
BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
device_map="auto"
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import gc
import time
import torch
import numpy as np

MODEL_PATH = "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


def load_ds7b_8bit():
    """8bit量化加载DS7B"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    # 1. Tokenizer
    print("[1] Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"    OK, vocab={len(tokenizer)}", flush=True)

    # 2. 8bit量化加载模型
    print("[2] Loading model in 8bit...", flush=True)
    t0 = time.time()
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,  # CPU offload兼容
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
    )
    t_load = time.time() - t0
    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9
    print(f"    Loaded in {t_load:.1f}s, device={device}, GPU={gpu_mem:.2f}GB", flush=True)
    
    # 打印device_map
    if hasattr(model, 'hf_device_map'):
        dm = model.hf_device_map
        gpu_mods = sum(1 for v in dm.values() if v == 0 or v == 'cuda:0')
        cpu_mods = sum(1 for v in dm.values() if v == 'cpu')
        print(f"    Device map: {gpu_mods} on GPU, {cpu_mods} on CPU", flush=True)
    
    return model, tokenizer, device


def test_inference(model, tokenizer, device):
    """推理测试"""
    print("\n[Test] Forward pass...", flush=True)
    prompt = "The scientist discovered a new"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    t_fwd = time.time() - t0
    hs = out.hidden_states
    print(f"    Forward: {t_fwd:.2f}s, {len(hs)} layers, last norm={hs[-1].float().norm():.2f}", flush=True)

    logits = out.logits[0, -1].float().cpu().numpy()
    top5_ids = np.argsort(logits)[-5:][::-1]
    top5 = [(tokenizer.decode([i]).strip(), float(logits[i])) for i in top5_ids]
    print(f"    Top-5: {top5}", flush=True)

    # 生成
    t0 = time.time()
    with torch.no_grad():
        gen_ids = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=20, do_sample=False)
    gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    t_gen = time.time() - t0
    print(f"    Generate ({t_gen:.1f}s): '{gen_text}'", flush=True)

    return {
        "fwd_time": round(t_fwd, 2),
        "gen_time": round(t_gen, 1),
        "n_layers": len(hs),
        "top5": top5,
        "gpu_mem_gb": round(torch.cuda.memory_allocated() / 1e9, 2),
    }


def test_weights(model):
    """测试权重访问（8bit dequantize）"""
    print("\n[Test2] Weight access (8bit dequantize)...", flush=True)
    layers = model.model.layers
    layer0 = layers[0]
    
    # 8bit模型的权重需要.float()来dequantize
    try:
        W_q = layer0.self_attn.q_proj.weight.detach().float().cpu().numpy()
        print(f"    W_q: shape={W_q.shape}, norm={np.linalg.norm(W_q):.2f}", flush=True)
    except Exception as e:
        print(f"    W_q failed: {e}", flush=True)
    
    try:
        W_o = layer0.self_attn.o_proj.weight.detach().float().cpu().numpy()
        print(f"    W_o: shape={W_o.shape}, norm={np.linalg.norm(W_o):.2f}", flush=True)
    except Exception as e:
        print(f"    W_o failed: {e}", flush=True)

    # W_U (lm_head)
    try:
        W_U = model.lm_head.weight.detach().float().cpu().numpy()
        print(f"    W_U: shape={W_U.shape}, norm={np.linalg.norm(W_U):.2f}", flush=True)
    except Exception as e:
        print(f"    W_U failed: {e}", flush=True)

    # Hook测试
    print("\n[Test3] Hook test...", flush=True)
    captured = {}
    def make_hook(key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().float().cpu()
            else:
                captured[key] = output.detach().float().cpu()
        return hook
    
    hook_layers = [0, 14, 27]
    hooks = [layers[li].register_forward_hook(make_hook(f"L{li}")) for li in hook_layers]
    
    prompt = "Hello world"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32)
    with torch.no_grad():
        model(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device))
    for h in hooks:
        h.remove()
    
    for key in sorted(captured.keys()):
        t = captured[key]
        print(f"    Hook {key}: shape={t.shape}, norm={t.float().norm():.2f}", flush=True)


if __name__ == "__main__":
    model, tokenizer, device = load_ds7b_8bit()
    
    result = test_inference(model, tokenizer, device)
    print(f"\nInference result: {result}", flush=True)
    
    test_weights(model)
    
    # 释放
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"\nGPU after release: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
    print("Done!", flush=True)
