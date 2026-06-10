"""
Phase 443: MLP vs Attention Path Decomposition
================================================
目标: 分解attention和MLP对类别自然运输的贡献

核心方法:
1. 在输入层注入类别方向扰动(alpha=1.0)
2. 记录每个last token位置的delta (基准delta)
3. 在指定层分别消融:
   a. attention output → zero (阻断attention路径)
   b. MLP output → zero (阻断MLP路径)
4. 比较消融后delta变化:
   - norm_score: 1 - ||delta_ablated|| / ||delta_orig||
   - direction_cos: cos(delta_ablated, delta_orig)
   - readout_score: 类别logit gap变化
   - cat_proj_change: 在类别方向投影的变化

注意: 消融组件输出为零时, 残差连接保持不变。
这意味着我们测量的是"移除该组件贡献"的效果。

用法:
  python tests/glm5/phase443_mlp_attn_decomposition.py qwen3 1
  python tests/glm5/phase443_mlp_attn_decomposition.py glm4 1
  python tests/glm5/phase443_mlp_attn_decomposition.py deepseek7b 1
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import os, gc, time, json
import numpy as np
import torch
from model_utils import (get_layers, get_model_info, release_model,
                          get_W_U, MODEL_CONFIGS)

# ===== 实验配置 =====
OBJECTS = {
    "apple": {
        "template": "The {obj} is a",
        "cat_words": ["fruit", "apple", "orange", "banana"],
        "opp_words": ["animal", "dog", "cat", "horse"],
    },
    "knife": {
        "template": "The {obj} is a",
        "cat_words": ["tool", "knife", "hammer", "scissors"],
        "opp_words": ["vehicle", "car", "bus", "train"],
    },
    "dog": {
        "template": "The {obj} is a",
        "cat_words": ["animal", "dog", "cat", "horse"],
        "opp_words": ["fruit", "apple", "orange", "banana"],
    },
}

ALPHA = 1.0


def load_model_auto(model_name):
    """BF16 + device_map='auto' 加载模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


def get_cat_direction(model, tokenizer, cat_words, opp_words):
    """从W_E计算类别方向"""
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        W_E = model.model.embed_tokens.weight.detach().float()
    elif hasattr(model, 'get_input_embeddings'):
        W_E = model.get_input_embeddings().weight.detach().float()
    else:
        return None
    
    cat_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in cat_words]
    opp_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in opp_words]
    
    d = W_E[cat_ids].mean(dim=0) - W_E[opp_ids].mean(dim=0)
    d = d.cpu()  # 确保在CPU上
    d = d / (d.norm() + 1e-8)
    return d


def get_cat_logit_gap(logits, tokenizer, cat_words, opp_words):
    """获取类别词与对立词的logit gap"""
    cat_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in cat_words]
    opp_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in opp_words]
    return float(logits[cat_ids].mean()) - float(logits[opp_ids].mean())


def inject_and_run(model, tokenizer, input_ids, attention_mask, last_pos, cat_dir, alpha,
                   ablation_layer=None, ablation_type=None):
    """
    运行模型: 输入层注入类别方向 + 可选的组件消融
    
    Returns: (last_hidden, logits) or None on failure
    """
    # 准备扰动方向(在正确设备上)
    input_device = next(model.parameters()).device
    perturb_vec = (alpha * cat_dir).to(input_device).to(torch.bfloat16)
    
    # Embedding hook: 注入扰动
    embed_hook = None
    
    def on_embed(module, inp, out):
        """在embedding输出上注入类别方向"""
        if isinstance(out, torch.Tensor):
            out = out.clone()
            out[0, last_pos] = out[0, last_pos] + perturb_vec
        return out
    
    # 找到embedding层
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        embed_hook = model.model.embed_tokens.register_forward_hook(on_embed)
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
        embed_hook = model.transformer.wte.register_forward_hook(on_embed)
    
    # Ablation hook
    abl_hook = None
    if ablation_layer is not None and ablation_type is not None:
        layers = get_layers(model)
        layer = layers[ablation_layer]
        
        def make_ablation_hook(atype):
            def hook(module, inp, out):
                if isinstance(out, tuple):
                    # (hidden_states, ...) 格式
                    zero_hidden = torch.zeros_like(out[0])
                    return (zero_hidden,) + out[1:]
                return torch.zeros_like(out)
            return hook
        
        if ablation_type == "attention" and hasattr(layer, 'self_attn'):
            abl_hook = layer.self_attn.register_forward_hook(make_ablation_hook("attn"))
        elif ablation_type == "mlp" and hasattr(layer, 'mlp'):
            abl_hook = layer.mlp.register_forward_hook(make_ablation_hook("mlp"))
    
    # 前向传播
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)
        last_hidden = out.hidden_states[-1][0, last_pos].detach().float().cpu()
        logits = out.logits[0, -1].float().cpu().numpy()
    except Exception as e:
        print(f"    Forward failed: {e}")
        last_hidden = None
        logits = None
    finally:
        if embed_hook is not None:
            embed_hook.remove()
        if abl_hook is not None:
            abl_hook.remove()
    
    return last_hidden, logits


def run_experiment(model_name, round_num):
    print(f"\n{'='*60}")
    print(f"Phase 443: MLP vs Attention Path Decomposition")
    print(f"Model: {model_name}, Round: {round_num}")
    print(f"{'='*60}")
    
    # ===== 1. 加载模型 =====
    print("\n[1] Loading model...")
    t0 = time.time()
    model, tokenizer = load_model_auto(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    print(f"  Loaded: {info.model_class}, {n_layers} layers, d_model={info.d_model}")
    print(f"  Load time: {time.time()-t0:.1f}s")
    
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    # ===== 2. 获取类别方向 =====
    print("\n[2] Computing category directions...")
    cat_directions = {}
    for obj_name, obj_info in OBJECTS.items():
        d = get_cat_direction(model, tokenizer, obj_info["cat_words"], obj_info["opp_words"])
        if d is not None:
            cat_directions[obj_name] = d
            print(f"  {obj_name}: direction norm={d.norm():.4f}")
    
    # ===== 3. 对每个对象做路径分解 =====
    results = {}
    
    # 测试层选择
    test_layer_candidates = [0, 1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32]
    test_layers = sorted(set([l for l in test_layer_candidates if l < n_layers] + [n_layers-1]))
    
    for obj_name, obj_info in OBJECTS.items():
        if obj_name not in cat_directions:
            continue
        
        print(f"\n{'='*50}")
        print(f"  Processing {obj_name} ({obj_info['cat_words'][0]})")
        print(f"{'='*50}")
        
        cat_dir = cat_directions[obj_name]
        
        # 构建输入
        text = obj_info["template"].format(obj=obj_name)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        last_pos = input_ids.shape[1] - 1
        
        # ===== 基准运行 =====
        print(f"  Running baseline...")
        with torch.no_grad():
            base_out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
        base_last = base_out.hidden_states[-1][0, last_pos].detach().float().cpu()
        base_logits = base_out.logits[0, -1].float().cpu().numpy()
        base_cat_gap = get_cat_logit_gap(base_logits, tokenizer, 
                                          obj_info["cat_words"], obj_info["opp_words"])
        
        # ===== 扰动运行 (无消融) =====
        print(f"  Running perturbed (alpha={ALPHA})...")
        pert_last, pert_logits = inject_and_run(
            model, tokenizer, input_ids, attention_mask, last_pos, cat_dir, ALPHA
        )
        
        if pert_last is None:
            print(f"  Perturbed run FAILED, skipping {obj_name}")
            continue
        
        pert_cat_gap = get_cat_logit_gap(pert_logits, tokenizer,
                                          obj_info["cat_words"], obj_info["opp_words"])
        
        orig_delta = pert_last - base_last
        orig_delta_norm = float(orig_delta.norm())
        orig_cat_shift = pert_cat_gap - base_cat_gap
        
        print(f"  base_cat_gap={base_cat_gap:.3f}, pert_cat_gap={pert_cat_gap:.3f}")
        print(f"  orig_delta_norm={orig_delta_norm:.4f}, cat_shift={orig_cat_shift:.3f}")
        
        if orig_delta_norm < 1e-6:
            print(f"  delta too small, skipping {obj_name}")
            continue
        
        # ===== 逐层路径分解 =====
        path_results = {}
        
        for layer_idx in test_layers:
            print(f"\n  Layer {layer_idx}/{n_layers-1}:")
            
            for ablation_type in ["attention", "mlp"]:
                # 消融运行
                abl_last, abl_logits = inject_and_run(
                    model, tokenizer, input_ids, attention_mask, last_pos, 
                    cat_dir, ALPHA,
                    ablation_layer=layer_idx, ablation_type=ablation_type
                )
                
                if abl_last is None:
                    print(f"    {ablation_type}: FAILED")
                    continue
                
                abl_cat_gap = get_cat_logit_gap(abl_logits, tokenizer,
                                                 obj_info["cat_words"], obj_info["opp_words"])
                
                # 计算指标
                abl_delta = abl_last - base_last
                abl_delta_norm = float(abl_delta.norm())
                
                # norm_score: 正=消融削弱运输, 负=消融增强运输
                norm_score = 1.0 - abl_delta_norm / orig_delta_norm if orig_delta_norm > 1e-6 else 0.0
                
                # direction_cos
                if abl_delta_norm > 1e-6 and orig_delta_norm > 1e-6:
                    direction_cos = float(torch.nn.functional.cosine_similarity(
                        abl_delta.unsqueeze(0), orig_delta.unsqueeze(0)
                    ).item())
                else:
                    direction_cos = 0.0
                
                # readout_score: 类别logit gap变化
                readout_score = abl_cat_gap - pert_cat_gap
                
                # category projection
                cat_dir_np = cat_dir.cpu().numpy() if isinstance(cat_dir, torch.Tensor) else cat_dir
                cat_dir_t = torch.tensor(cat_dir_np, dtype=torch.float32)
                
                cat_proj_orig = float((orig_delta @ cat_dir_t) / orig_delta_norm)
                cat_proj_abl = float((abl_delta @ cat_dir_t) / abl_delta_norm) if abl_delta_norm > 1e-6 else 0.0
                cat_proj_change = cat_proj_abl - cat_proj_orig
                
                key = f"L{layer_idx}_{ablation_type}"
                path_results[key] = {
                    "layer": layer_idx,
                    "ablation_type": ablation_type,
                    "norm_score": round(norm_score, 4),
                    "direction_cos": round(direction_cos, 4),
                    "readout_score": round(readout_score, 4),
                    "abl_delta_norm": round(abl_delta_norm, 4),
                    "cat_proj_change": round(cat_proj_change, 4),
                    "abl_cat_gap": round(abl_cat_gap, 4),
                }
                
                print(f"    {ablation_type}: norm_score={norm_score:.4f}, "
                      f"dir_cos={direction_cos:.4f}, readout={readout_score:.4f}")
            
            # 定期GPU检查
            if torch.cuda.is_available() and layer_idx % 8 == 0:
                gpu_mem = torch.cuda.memory_allocated() / 1e9
                print(f"    [GPU: {gpu_mem:.2f}GB]")
                gc.collect()
                torch.cuda.empty_cache()
        
        results[obj_name] = {
            "category": obj_info["cat_words"][0],
            "base_cat_gap": round(base_cat_gap, 4),
            "pert_cat_gap": round(pert_cat_gap, 4),
            "orig_delta_norm": round(orig_delta_norm, 4),
            "cat_shift": round(orig_cat_shift, 4),
            "path_decomposition": path_results,
        }
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ===== 汇总 =====
    print(f"\n{'='*60}")
    print("PHASE 443 SUMMARY")
    print(f"{'='*60}")
    
    for obj_name, obj_result in results.items():
        print(f"\n  {obj_name}:")
        
        # 分组统计
        attn_norms = []
        mlp_norms = []
        attn_readouts = []
        mlp_readouts = []
        
        for key, val in obj_result["path_decomposition"].items():
            if val["ablation_type"] == "attention":
                attn_norms.append(val["norm_score"])
                attn_readouts.append(val["readout_score"])
            else:
                mlp_norms.append(val["norm_score"])
                mlp_readouts.append(val["readout_score"])
        
        if attn_norms and mlp_norms:
            print(f"    Attention ablation: avg_norm={np.mean(attn_norms):.4f}, "
                  f"avg_readout={np.mean(attn_readouts):.4f}")
            print(f"    MLP ablation: avg_norm={np.mean(mlp_norms):.4f}, "
                  f"avg_readout={np.mean(mlp_readouts):.4f}")
            print(f"    |MLP|/|Attn| ratio: {abs(np.mean(mlp_norms))/(abs(np.mean(attn_norms))+1e-8):.2f}")
    
    # 保存
    output = {
        "model": model_name,
        "round": round_num,
        "n_layers": n_layers,
        "alpha": ALPHA,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "per_object": results,
    }
    
    os.makedirs("results/phase443_mlp_attn_decomposition", exist_ok=True)
    out_path = f"results/phase443_mlp_attn_decomposition/{model_name}_phase443_r{round_num}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {out_path}")
    
    # 释放模型
    print("\n  Releasing model...")
    release_model(model)
    model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return output


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    run_experiment(model_name, round_num)
