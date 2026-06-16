"""
Phase 500: Gain-Support Alignment — GLM5×GPT5 路线统一验证
============================================================
核心假设: RMSNorm Gain向量(g⊙w_D) 是否正是 GPT5路线的 W·R_pre(类别支持方向)
         在学习到的读出空间中的投影？

如果 cos(v_cat, g⊙w_D) >> cos(v_cat, w_D), 说明RMSNorm Gain确实起到了
"选择性放大类别语义方向"的语义门控作用, 两条路线可以合并为统一理论:
    语言编码 = 上下文字段语义方向 × RMSNorm Gain选择性读出

测试5个类别 × 10个对象 × 3个模型(Qwen3→GLM4→DS7B)

Usage:
  python tests/glm5/phase500_gain_support_alignment.py qwen3 1
  python tests/glm5/phase500_gain_support_alignment.py glm4 1
  python tests/glm5/phase500_gain_support_alignment.py deepseek7b 1
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
from collections import defaultdict
from datetime import datetime

from model_utils import (
    load_model, get_layers, get_model_info,
    release_model, get_W_U, MODEL_CONFIGS
)

# ===== 类别配置(复刻Phase498) =====
CATEGORIES = {
    "fruit": {
        "objects": ["apple","banana","orange","grape","pear","peach","mango","plum","cherry","lemon"],
        "relation": "is a type of fruit",
        "target_tokens": ["fruit"],
    },
    "clothing": {
        "objects": ["shirt","dress","jacket","pants","coat","skirt","sweater","blouse","scarf","vest"],
        "relation": "is a type of clothing",
        "target_tokens": ["clothing"],
    },
    "emotion": {
        "objects": ["joy","anger","fear","sadness","surprise","disgust","pride","shame","guilt","envy"],
        "relation": "is a type of emotion",
        "target_tokens": ["emotion"],
    },
    "action": {
        "objects": ["run","eat","build","throw","buy","learn","measure","communicate","swim","write"],
        "relation": "is a type of action",
        "target_tokens": ["action"],
    },
    "animal": {
        "objects": ["dog","cat","horse","elephant","tiger","dolphin","eagle","snake","rabbit","whale"],
        "relation": "is a type of animal",
        "target_tokens": ["animal"],
    },
}

# 所有class token(作为通用competitor)
ALL_CLASS_TOKENS = ["fruit","animal","clothing","emotion","action"]
OUTPUT_DIR = Path("results/glm5")


def get_rmsnorm_weight(model, model_name):
    """获取final RMSNorm的weight向量(g)"""
    for attr in ['model.norm', 'model.final_layernorm', 'model.decoder.final_layer_norm', 'transformer.ln_f']:
        parts = attr.split('.')
        obj = model
        ok = True
        for p in parts:
            if hasattr(obj, p):
                obj = getattr(obj, p)
            else:
                ok = False
                break
        if ok and hasattr(obj, 'weight'):
            w = obj.weight.detach()
            if str(w.device) == 'meta':
                continue
            return w.float().cpu().numpy()
    return None


def get_lm_head_weight(model, model_name):
    """获取lm_head/unembedding权重矩阵 W_U [vocab, d_model]"""
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
        w = model.lm_head.weight.detach()
        if str(w.device) == 'meta':
            return None
        return w.float().cpu().numpy()
    return None


def make_prompt(obj_name, relation):
    return f"The {obj_name} {relation}"


def get_token_ids(tokenizer, tokens_str):
    """获取token IDs"""
    ids = []
    for t in tokens_str:
        tid = tokenizer.encode(t, add_special_tokens=False)
        if tid:
            ids.append(tid[0])
    return ids


def rms_norm_numpy(x, weight=None, eps=1e-5):
    """RMSNorm in numpy"""
    rms = np.sqrt(np.mean(x ** 2) + eps)
    normed = x / rms
    if weight is not None:
        normed = normed * weight
    return normed, rms


def run_phase500(model, tokenizer, model_name, n_objects_per_cat=None):
    """
    核心实验: 测量类别语义方向v_cat与w_D/g⊙w_D的对齐度
    
    返回:
      results[category] = {
        "cos_v_wD": cos(v_cat, w_D),
        "cos_v_gwD": cos(v_cat, g⊙w_D),
        "alignment_gain": cos_v_gwD - cos_v_wD,
        "ratio_gwD_wD": ||g⊙w_D|| / ||w_D||,
        "D_orig": 原始DCF值,
        "norm_h": ||h_pre||,
        "norm_h_normed": ||h_post||,
      }
    """
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    results = {}

    # Step 1: 获取W_U和g
    W_U = get_W_U(model, model_name)
    if W_U is None:
        print("ERROR: Cannot get W_U (lm_head weight)")
        return None
    W_U = W_U.astype(np.float64)
    print(f"W_U shape: {W_U.shape}")

    g_vec = get_rmsnorm_weight(model, model_name)
    if g_vec is None:
        print("ERROR: Cannot get RMSNorm gain weight")
        return None
    g_vec = g_vec.astype(np.float64)
    print(f"g_vec shape: {g_vec.shape}")

    # Step 2: 获取所有target/competitor token IDs
    all_target_ids = {}
    all_comp_ids = []
    for cat_name, cfg in CATEGORIES.items():
        tids = get_token_ids(tokenizer, cfg["target_tokens"])
        all_target_ids[cat_name] = tids
        # competitor = all other class tokens
        other_classes = [c for c in ALL_CLASS_TOKENS if c not in cfg["target_tokens"]]
        all_comp_ids.extend(get_token_ids(tokenizer, other_classes))
    all_comp_ids = list(set(all_comp_ids))

    # Step 3: 对每个类别,收集hidden states
    print(f"\n=== Collecting hidden states for {len(CATEGORIES)} categories ===")

    for cat_name, cfg in CATEGORIES.items():
        cat_start = time.time()
        objects = cfg["objects"]
        relation = cfg["relation"]
        
        h_pre_list = []   # pre-norm hidden states
        h_post_list = []  # post-norm hidden states
        D_orig_list = []

        for obj in objects:
            prompt = make_prompt(obj, relation)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                # 需要获取最后一层后的hidden states
                if hasattr(model, 'model'):
                    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                    hidden_states = outputs.hidden_states  # tuple of [batch, seq, d_model]
                else:
                    outputs = model(**inputs, output_hidden_states=True)
                    hidden_states = outputs.hidden_states
            
            # 最后一个token的hidden state(答案位置)
            last_hidden = hidden_states[-1][0, -1, :]  # [d_model] - 最后一层输出
            
            # 需要区分pre-norm和post-norm
            if hasattr(model, 'model') and hasattr(model.model, 'norm'):
                # 有显式final norm的情况 - last_hidden是pre-norm
                h_pre = last_hidden.float().cpu().numpy().astype(np.float64)
                h_post_np, rms_val = rms_norm_numpy(h_pre, g_vec)
                h_post = h_post_np.astype(np.float64)
            else:
                # 没有显式final norm - last_hidden已经是post-norm
                h_post = last_hidden.float().cpu().numpy().astype(np.float64)
                # 反推pre-norm(近似, 假设无gain):
                rms_approx = np.sqrt(np.mean(h_post ** 2))
                h_pre = h_post * rms_approx  # 反归一化
                h_pre = h_pre.astype(np.float64)
            
            h_pre_list.append(h_pre)
            h_post_list.append(h_post)
            
            # 计算DCF
            logits = h_post @ W_U.T
            target_ids = all_target_ids.get(cat_name, [])
            target_logit = np.mean([logits[tid] for tid in target_ids if tid < len(logits)])
            comp_logits = [logits[cid] for cid in all_comp_ids if cid < len(logits)]
            if comp_logits:
                D_orig = float(target_logit - np.mean(comp_logits))
            else:
                D_orig = 0.0
            D_orig_list.append(D_orig)
        
        # 类别语义方向: mean(category_h) - mean(all_h)
        h_pre_mean = np.mean(h_pre_list, axis=0)
        
        # w_D: unembedding方向的target-competitor差
        target_ids = all_target_ids.get(cat_name, [])
        w_D_target = np.mean([W_U[tid] for tid in target_ids if tid < len(W_U)], axis=0)
        w_D_comp = np.mean([W_U[cid] for cid in all_comp_ids if cid < len(W_U)], axis=0)
        w_D = w_D_target - w_D_comp
        
        # g⊙w_D: gain加权的unembedding方向
        gw_D = w_D * g_vec
        
        # 所有类别的mean(用于做background)
        all_h_pre = h_pre_list  # 暂时只用当前类别的
        
        # cos(v_cat, w_D) and cos(v_cat, g⊙w_D)
        v_norm = np.linalg.norm(h_pre_mean)
        wD_norm = np.linalg.norm(w_D)
        gwD_norm = np.linalg.norm(gw_D)
        
        if v_norm > 0 and wD_norm > 0:
            cos_v_wD = float(np.dot(h_pre_mean, w_D) / (v_norm * wD_norm))
        else:
            cos_v_wD = 0.0
        
        if v_norm > 0 and gwD_norm > 0:
            cos_v_gwD = float(np.dot(h_pre_mean, gw_D) / (v_norm * gwD_norm))
        else:
            cos_v_gwD = 0.0
        
        # Gain ratio
        gain_ratio = gwD_norm / wD_norm if wD_norm > 0 else 1.0
        
        results[cat_name] = {
            "cos_v_wD": round(cos_v_wD, 6),
            "cos_v_gwD": round(cos_v_gwD, 6),
            "alignment_gain": round(cos_v_gwD - cos_v_wD, 6),
            "gain_ratio": round(gain_ratio, 4),
            "D_orig_mean": round(float(np.mean(D_orig_list)), 4),
            "D_orig_std": round(float(np.std(D_orig_list)), 4),
            "norm_h_pre": round(float(v_norm), 2),
            "norm_w_D": round(float(wD_norm), 4),
            "norm_gw_D": round(float(gwD_norm), 4),
            "h_pre_norm_cat": round(float(v_norm), 2),
        }
        
        elapsed = time.time() - cat_start
        print(f"  {cat_name}: cos(v,wD)={cos_v_wD:.4f}, cos(v,gwD)={cos_v_gwD:.4f}, "
              f"Δ={cos_v_gwD-cos_v_wD:.4f}, gain_ratio={gain_ratio:.2f}, "
              f"D={np.mean(D_orig_list):.2f} ({elapsed:.1f}s)")
    
    # Step 4: 模型级汇总
    all_alignment_gains = [r["alignment_gain"] for r in results.values()]
    all_cos_gwD = [r["cos_v_gwD"] for r in results.values()]
    all_cos_wD = [r["cos_v_wD"] for r in results.values()]
    all_gain_ratios = [r["gain_ratio"] for r in results.values()]
    
    summary = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "mean_alignment_gain": round(np.mean(all_alignment_gains), 6),
        "std_alignment_gain": round(np.std(all_alignment_gains), 6),
        "mean_cos_gwD": round(np.mean(all_cos_gwD), 6),
        "mean_cos_wD": round(np.mean(all_cos_wD), 6),
        "mean_gain_ratio": round(np.mean(all_gain_ratios), 4),
        "positive_gain_count": sum(1 for g in all_alignment_gains if g > 0),
        "total_categories": len(all_alignment_gains),
        "categories": results,
    }
    
    return summary


def main():
    if len(sys.argv) < 2:
        print("Usage: python phase500_gain_support_alignment.py <model> [n_objects]")
        print("  model: qwen3, glm4, deepseek7b")
        sys.exit(1)
    
    model_name = sys.argv[1]
    n_objects = int(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)
    
    print(f"{'='*70}")
    print(f"Phase 500: Gain-Support Alignment — {model_name}")
    print(f"{'='*70}")
    print(f"Hypothesis: RMSNorm Gain(g⊙w_D) aligns with category direction in h-space")
    print(f"If cos(v_cat, g⊙w_D) >> cos(v_cat, w_D), Gain IS the semantic gate")
    print(f"")
    
    # Load model
    print(f"Loading {model_name}...")
    start = time.time()
    model, tokenizer, device = load_model(model_name)
    load_time = time.time() - start
    print(f"Loaded in {load_time:.1f}s, VRAM: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    try:
        results = run_phase500(model, tokenizer, model_name, n_objects)
        total_time = time.time() - start
        
        if results is None:
            print("ERROR: Experiment failed")
            return
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"RESULTS SUMMARY — {model_name}")
        print(f"{'='*70}")
        print(f"  mean cos(v, w_D):       {results['mean_cos_wD']:+.5f}")
        print(f"  mean cos(v, g⊙w_D):     {results['mean_cos_gwD']:+.5f}")
        print(f"  mean alignment gain:    {results['mean_alignment_gain']:+.5f}")
        print(f"  positive gain count:    {results['positive_gain_count']}/{results['total_categories']}")
        print(f"  mean gain_ratio:        {results['mean_gain_ratio']:.2f}")
        print(f"  total time:             {total_time:.1f}s")
        
        print(f"\nPer-category breakdown:")
        for cat, r in results['categories'].items():
            sig = "✅" if r['alignment_gain'] > 0.01 else ("⚠️" if r['alignment_gain'] > -0.01 else "❌")
            print(f"  {sig} {cat:10s}: cos_wD={r['cos_v_wD']:+.5f}  "
                  f"cos_gwD={r['cos_v_gwD']:+.5f}  "
                  f"Δ={r['alignment_gain']:+.5f}  "
                  f"gain_ratio={r['gain_ratio']:.2f}  "
                  f"D={r['D_orig_mean']:+.2f}")
        
        # Save
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR / f"phase500_{model_name}_r1.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {out_path}")
        
    finally:
        release_model(model)
        print("Model released.")


if __name__ == "__main__":
    main()
