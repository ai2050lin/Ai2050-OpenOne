"""
Phase 498 R2: GLM4 RMSNorm Weight修复 + Gain向量深度分析
=========================================================
核心目标:
1. 从safetensors直接读取GLM4的final RMSNorm weight
2. 对Qwen3和GLM4做gain向量的深度结构分析
3. 分析gain向量g如何把"抑制性方向"变成"释放性方向"
4. 加大样本验证gain_effect的主导地位

Usage:
  python tests/glm5/phase498_gain_vector_analysis.py qwen3 2
  python tests/glm5/phase498_gain_vector_analysis.py glm4 2
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import json
import time
import numpy as np
import torch
from pathlib import Path
from model_utils import (load_model, get_layers, get_model_info,
                          release_model, get_W_U, MODEL_CONFIGS)
from datetime import datetime

CATEGORIES = {
    "fruit": {
        "objects": ["apple", "banana", "orange", "grape", "pear",
                    "peach", "mango", "plum", "cherry", "lemon",
                    "strawberry", "watermelon"],
        "relation": "is a type of fruit",
        "target_tokens": ["fruit"],
    },
    "clothing": {
        "objects": ["shirt", "dress", "jacket", "pants", "coat",
                    "skirt", "sweater", "blouse", "scarf", "vest",
                    "hoodie", "cardigan"],
        "relation": "is a type of clothing",
        "target_tokens": ["clothing"],
    },
    "emotion": {
        "objects": ["joy", "anger", "fear", "sadness", "surprise",
                    "disgust", "pride", "shame", "guilt", "envy",
                    "love", "hope"],
        "relation": "is a type of emotion",
        "target_tokens": ["emotion"],
    },
    "action": {
        "objects": ["run", "eat", "build", "throw", "buy",
                    "learn", "measure", "communicate", "swim", "write",
                    "read", "cook"],
        "relation": "is a type of action",
        "target_tokens": ["action"],
    },
    "animal": {
        "objects": ["dog", "cat", "horse", "elephant", "tiger",
                    "dolphin", "eagle", "snake", "rabbit", "whale",
                    "lion", "bear"],
        "relation": "is a type of animal",
        "target_tokens": ["animal"],
    },
}

OUTPUT_DIR = Path("results/glm5")


def load_rmsnorm_weight_from_safetensors(model_name):
    """从safetensors文件直接读取final RMSNorm weight"""
    cfg = MODEL_CONFIGS[model_name]
    model_path = Path(cfg["path"])
    
    # 查找safetensors文件
    safetensors_files = sorted(model_path.glob("*.safetensors"))
    if not safetensors_files:
        print(f"  No safetensors files found in {model_path}")
        return None
    
    print(f"  Scanning {len(safetensors_files)} safetensors files for final norm weight...")
    
    try:
        from safetensors import safe_open
        
        # 查找final norm的key
        norm_keys = [
            "model.norm.weight",
            "model.final_layernorm.weight",
            "model.decoder.final_layer_norm.weight",
        ]
        
        for sf_path in safetensors_files:
            with safe_open(str(sf_path), framework="pt", device="cpu") as f:
                keys = f.keys()
                for nk in norm_keys:
                    if nk in keys:
                        weight = f.get_tensor(nk)
                        print(f"  Found {nk} in {sf_path.name}, shape={weight.shape}")
                        return weight.float().numpy()
        
        # 如果没找到精确匹配，列出所有norm相关key
        print(f"  Norm keys not found. Listing all keys containing 'norm'...")
        for sf_path in safetensors_files:
            with safe_open(str(sf_path), framework="pt", device="cpu") as f:
                norm_keys_found = [k for k in f.keys() if 'norm' in k.lower()]
                if norm_keys_found:
                    # 只看最后几个
                    for k in norm_keys_found[-5:]:
                        t = f.get_tensor(k)
                        print(f"    {k}: shape={t.shape}")
                    # 如果只有一个1D norm weight，就是它
                    for k in norm_keys_found:
                        t = f.get_tensor(k)
                        if len(t.shape) == 1:
                            print(f"  Using {k} as final norm weight")
                            return t.float().numpy()
        
        print(f"  Could not find final norm weight in safetensors")
        return None
    except ImportError:
        print(f"  safetensors library not available")
        return None
    except Exception as e:
        print(f"  Error loading from safetensors: {e}")
        return None


def compute_D_from_hidden(hidden_np, W_U, target_ids, comp_ids):
    logits = hidden_np @ W_U.T
    target_logit = np.mean([logits[tid] for tid in target_ids if tid < len(logits)])
    comp_logits = [logits[cid] for cid in comp_ids if cid < len(logits)]
    if len(comp_logits) == 0:
        return 0.0
    return float(target_logit - np.mean(comp_logits))


def make_prompt(obj, relation):
    return f"The {obj} {relation}"


def get_target_and_comp_ids(tokenizer, target_tokens, comp_categories_tokens):
    target_ids = []
    for t in target_tokens:
        ids = tokenizer.encode(t, add_special_tokens=False)
        target_ids.extend(ids)
    comp_ids = []
    for cat_tokens in comp_categories_tokens:
        for t in cat_tokens:
            ids = tokenizer.encode(t, add_special_tokens=False)
            comp_ids.extend(ids)
    return target_ids, comp_ids


def run_gain_vector_analysis(model, tokenizer, model_name, rmsnorm_w, n_samples=12):
    """
    Gain向量深度分析:
    1. g⊙w_D vs w_D 的方向差异
    2. gain对各类别D的贡献分解
    3. gain向量在shared_semantic方向上的投影
    4. MLP输出在g⊙w_D vs w_D下的贡献差异
    """
    print("\n" + "="*60)
    print("Gain向量深度分析")
    print("="*60)
    
    W_U = get_W_U(model, model_name)
    d_model = W_U.shape[1]
    
    if rmsnorm_w is None:
        print("  [ERROR] No RMSNorm weight available, cannot do gain analysis")
        return {}
    
    print(f"  RMSNorm weight: shape={rmsnorm_w.shape}, mean={rmsnorm_w.mean():.4f}, "
          f"std={rmsnorm_w.std():.4f}, min={rmsnorm_w.min():.4f}, max={rmsnorm_w.max():.4f}")
    
    # 计算gain向量统计
    gain_above_1 = np.sum(rmsnorm_w > 1.0)
    gain_below_1 = np.sum(rmsnorm_w < 1.0)
    gain_near_0 = np.sum(np.abs(rmsnorm_w) < 0.1)
    print(f"  Gain stats: >1={gain_above_1}, <1={gain_below_1}, near0={gain_near_0}")
    
    # 找到final norm层
    final_norm = None
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        final_norm = model.model.norm
    elif hasattr(model, 'model') and hasattr(model.model, 'final_layernorm'):
        final_norm = model.model.final_layernorm
    
    all_cat_tokens = {k: v["target_tokens"] for k, v in CATEGORIES.items()}
    
    results = {}
    
    for cat_name, cat_data in CATEGORIES.items():
        comp_cats = [k for k in CATEGORIES if k != cat_name]
        comp_tokens = [CATEGORIES[k]["target_tokens"] for k in comp_cats]
        target_ids, comp_ids = get_target_and_comp_ids(
            tokenizer, cat_data["target_tokens"], comp_tokens)
        
        # 有效读出方向: g⊙w_D (对target和comp分别)
        w_D_target = np.mean([W_U[tid] for tid in target_ids if tid < W_U.shape[0]], axis=0)
        w_D_comp = np.mean([W_U[cid] for cid in comp_ids if cid < W_U.shape[0]], axis=0)
        w_D = w_D_target - w_D_comp  # DCF读出方向
        
        # Gain加权的读出方向
        gw_D_target = np.mean([W_U[tid] * rmsnorm_w for tid in target_ids if tid < W_U.shape[0]], axis=0)
        gw_D_comp = np.mean([W_U[cid] * rmsnorm_w for cid in comp_ids if cid < W_U.shape[0]], axis=0)
        gw_D = gw_D_target - gw_D_comp  # gain加权DCF读出方向
        
        # g⊙w_D vs w_D 的角度和范数
        w_D_norm = np.linalg.norm(w_D)
        gw_D_norm = np.linalg.norm(gw_D)
        if w_D_norm > 0 and gw_D_norm > 0:
            cos_angle = np.dot(w_D, gw_D) / (w_D_norm * gw_D_norm)
        else:
            cos_angle = 0.0
        
        # gain对w_D方向的逐维度贡献
        # g⊙w_D - w_D = (g-1)⊙w_D
        delta_w = gw_D - w_D
        gain_contrib_per_dim = (rmsnorm_w - 1.0) * w_D
        
        # 哪些维度被gain放大，哪些被抑制
        gain_amplified = np.sum((rmsnorm_w - 1.0) * w_D > 0)
        gain_suppressed = np.sum((rmsnorm_w - 1.0) * w_D < 0)
        
        cat_results = []
        objects = cat_data["objects"][:n_samples]
        
        for obj in objects:
            prompt = make_prompt(obj, cat_data["relation"])
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"]
            attn_mask = inputs["attention_mask"]
            
            input_device = next(model.parameters()).device
            input_ids = input_ids.to(input_device)
            attn_mask = attn_mask.to(input_device)
            
            # 获取pre-norm hidden
            captured = {}
            def hook_pre_norm(module, input, output):
                captured['pre_norm'] = input[0].detach()
            
            if final_norm is not None:
                h = final_norm.register_forward_hook(hook_pre_norm)
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attn_mask)
                h.remove()
                h_pre = captured['pre_norm'][0, -1].float().cpu().numpy()
            else:
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attn_mask,
                               output_hidden_states=True)
                h_pre = out.hidden_states[-2][0, -1].float().cpu().numpy()
            
            # D in pre-norm space (无gain, 无norm)
            D_pre = compute_D_from_hidden(h_pre, W_U, target_ids, comp_ids)
            
            # D with norm but no gain: h/rms
            rms_val = float(np.sqrt(np.mean(h_pre ** 2) + 1e-5))
            h_normed = h_pre / rms_val
            D_normed_no_gain = compute_D_from_hidden(h_normed, W_U, target_ids, comp_ids)
            
            # D with norm and gain: (h/rms)*g
            h_normed_with_gain = h_normed * rmsnorm_w
            D_normed_with_gain = compute_D_from_hidden(h_normed_with_gain, W_U, target_ids, comp_ids)
            
            # 分解: gain效应 = D_with_gain - D_no_gain
            gain_effect = D_normed_with_gain - D_normed_no_gain
            
            # 用有效读出方向分析
            proj_wD = np.dot(h_pre, w_D) / (np.linalg.norm(w_D) + 1e-10)
            proj_gwD = np.dot(h_pre, gw_D) / (np.linalg.norm(gw_D) + 1e-10)
            
            # gain如何改变各维度对D的贡献
            dim_contrib_no_gain = h_normed * w_D  # 逐维贡献 (no gain)
            dim_contrib_with_gain = h_normed_with_gain * W_U[target_ids[0]] if len(target_ids) > 0 and target_ids[0] < W_U.shape[0] else np.zeros(d_model)
            
            # 高gain维度 vs 低gain维度对D的贡献
            high_gain_dims = np.where(rmsnorm_w > np.percentile(rmsnorm_w, 75))[0]
            low_gain_dims = np.where(rmsnorm_w < np.percentile(rmsnorm_w, 25))[0]
            
            # h_normed[high_gain_dims] · W_U[tid][high_gain_dims] 对每个target token
            def dim_contrib(h_vec, dims):
                contribs = []
                for tid in target_ids:
                    if tid < W_U.shape[0]:
                        contribs.append(float(np.dot(h_vec[dims], W_U[tid][dims])))
                return float(np.mean(contribs)) if contribs else 0.0
            
            high_gain_D_no = dim_contrib(h_normed, high_gain_dims)
            high_gain_D_with = dim_contrib(h_normed_with_gain, high_gain_dims)
            low_gain_D_no = dim_contrib(h_normed, low_gain_dims)
            low_gain_D_with = dim_contrib(h_normed_with_gain, low_gain_dims)
            
            cat_results.append({
                "obj": obj,
                "D_pre": D_pre,
                "D_normed_no_gain": D_normed_no_gain,
                "D_normed_with_gain": D_normed_with_gain,
                "gain_effect": gain_effect,
                "rms": rms_val,
                "proj_wD": float(proj_wD),
                "proj_gwD": float(proj_gwD),
                "high_gain_D_no": high_gain_D_no,
                "high_gain_D_with": high_gain_D_with,
                "low_gain_D_no": low_gain_D_no,
                "low_gain_D_with": low_gain_D_with,
            })
            
            print(f"  {cat_name}/{obj}: D_pre={D_pre:.2f}, D_no_gain={D_normed_no_gain:.2f}, "
                  f"D_with_gain={D_normed_with_gain:.2f}, gain_eff={gain_effect:.3f}, "
                  f"cos(wD,gwD)={cos_angle:.3f}")
        
        means = {}
        for key in ["D_pre", "D_normed_no_gain", "D_normed_with_gain", "gain_effect",
                     "rms", "proj_wD", "proj_gwD",
                     "high_gain_D_no", "high_gain_D_with", "low_gain_D_no", "low_gain_D_with"]:
            means[f"mean_{key}"] = float(np.mean([r[key] for r in cat_results]))
        
        means["n_samples"] = len(cat_results)
        means["w_D_norm"] = float(w_D_norm)
        means["gw_D_norm"] = float(gw_D_norm)
        means["cos_angle_wD_gwD"] = float(cos_angle)
        means["gain_amplified_dims"] = int(gain_amplified)
        means["gain_suppressed_dims"] = int(gain_suppressed)
        means["sample_details"] = cat_results
        results[cat_name] = means
    
    # Gain向量全局分析
    print("\n--- Gain向量全局结构分析 ---")
    
    # 对所有类别计算g⊙w_D vs w_D的角度
    for cat_name in CATEGORIES:
        if cat_name not in results:
            continue
        r = results[cat_name]
        print(f"  {cat_name}: cos(w_D, g⊙w_D)={r['cos_angle_wD_gwD']:.3f}, "
              f"||w_D||={r['w_D_norm']:.2f}, ||g⊙w_D||={r['gw_D_norm']:.2f}, "
              f"gain_eff={r['mean_gain_effect']:.3f}")
    
    return results


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    n_samples = 12
    
    print(f"\n{'#'*60}")
    print(f"Phase 498 R{round_num}: Gain Vector Analysis")
    print(f"Model: {model_name}, n_samples={n_samples}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*60}\n")
    
    # 加载模型
    t0 = time.time()
    model, tokenizer, device = load_model(model_name)
    print(f"Model loaded in {time.time()-t0:.1f}s")
    
    # 尝试获取RMSNorm weight
    rmsnorm_w = None
    
    # 方法1: 从模型直接获取
    final_norm = None
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        final_norm = model.model.norm
    elif hasattr(model, 'model') and hasattr(model.model, 'final_layernorm'):
        final_norm = model.model.final_layernorm
    
    if final_norm is not None and hasattr(final_norm, 'weight'):
        w = final_norm.weight.detach()
        if str(w.device) != 'meta':
            rmsnorm_w = w.float().cpu().numpy()
            print(f"  Got RMSNorm weight from model: shape={rmsnorm_w.shape}")
    
    # 方法2: 从safetensors读取 (GLM4)
    if rmsnorm_w is None:
        print(f"  RMSNorm weight not available from model, trying safetensors...")
        rmsnorm_w = load_rmsnorm_weight_from_safetensors(model_name)
    
    if rmsnorm_w is not None:
        print(f"  Final RMSNorm weight: shape={rmsnorm_w.shape}, "
              f"mean={rmsnorm_w.mean():.4f}, std={rmsnorm_w.std():.4f}")
    else:
        print(f"  WARNING: No RMSNorm weight available for {model_name}")
    
    # Gain向量分析
    all_results = {
        "phase": 498,
        "round": round_num,
        "model": model_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_info": {
            "class": type(model).__name__,
            "n_layers": get_model_info(model, model_name).n_layers,
            "d_model": get_model_info(model, model_name).d_model,
        },
        "rmsnorm_weight_available": rmsnorm_w is not None,
    }
    
    if rmsnorm_w is not None:
        all_results["rmsnorm_weight_stats"] = {
            "shape": list(rmsnorm_w.shape),
            "mean": float(rmsnorm_w.mean()),
            "std": float(rmsnorm_w.std()),
            "min": float(rmsnorm_w.min()),
            "max": float(rmsnorm_w.max()),
            "pct_above_1": float(np.mean(rmsnorm_w > 1.0)),
            "pct_below_1": float(np.mean(rmsnorm_w < 1.0)),
        }
    
    try:
        gain_analysis = run_gain_vector_analysis(model, tokenizer, model_name, rmsnorm_w, n_samples)
        all_results["gain_vector_analysis"] = gain_analysis
    except Exception as e:
        print(f"Gain analysis FAILED: {e}")
        import traceback; traceback.print_exc()
        all_results["gain_analysis_error"] = str(e)
    
    # 保存结果
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUTPUT_DIR / f"phase498_{model_name}_r{round_num}.json"
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_file}")
    
    # 释放模型
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Phase 498 R{round_num} complete for {model_name}")


if __name__ == "__main__":
    main()
