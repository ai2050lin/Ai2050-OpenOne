"""
Phase 501: Gain-Support Alignment R2 — 干预差分法 + 三模板 + 扩对象
======================================================================
Phase 500 R1 用 mean(h_category) 做方向，噪声大。
R2 改用干预差分法: v_cat = h_rich - h_neutral (category-rich vs neutral prompt)
同时扩展到3类模板(short/long/neutral)，20对象/类别。

核心假设同Phase 500:
  cos(v_cat, g⊙w_D) >> cos(v_cat, w_D)
  即 Gain向量确实增强了类别语义方向的对齐。

Usage:
  python tests/glm5/phase501_gain_alignment_r2.py qwen3 1
  python tests/glm5/phase501_gain_alignment_r2.py glm4 1
  python tests/glm5/phase501_gain_alignment_r2.py deepseek7b 1
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc, json, time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from model_utils import (
    load_model, get_layers, get_model_info,
    release_model, get_W_U, MODEL_CONFIGS
)

# ===== 类别配置(扩对象到20) =====
CATEGORIES = {
    "fruit": {
        "objects": ["apple","banana","orange","grape","pear","peach","mango","plum",
                    "cherry","lemon","apricot","kiwi","pineapple","melon","coconut",
                    "lime","fig","pomegranate","papaya","avocado"],
        "relation_rich": "is a type of fruit",
        "relation_neutral": "is a thing",
        "target_tokens": ["fruit"],
    },
    "clothing": {
        "objects": ["shirt","dress","jacket","pants","coat","skirt","sweater","blouse",
                    "scarf","vest","hat","glove","sock","boot","belt",
                    "tie","jeans","shorts","hoodie","raincoat"],
        "relation_rich": "is a type of clothing",
        "relation_neutral": "is a thing",
        "target_tokens": ["clothing"],
    },
    "emotion": {
        "objects": ["joy","anger","fear","sadness","surprise","disgust","pride","shame",
                    "guilt","envy","hope","love","hate","boredom","anxiety",
                    "jealousy","gratitude","regret","curiosity","embarrassment"],
        "relation_rich": "is a type of emotion",
        "relation_neutral": "is a concept",
        "target_tokens": ["emotion"],
    },
    "action": {
        "objects": ["run","eat","build","throw","buy","learn","measure","communicate",
                    "swim","write","sing","draw","fly","climb","teach",
                    "drive","cook","dance","fight","sleep"],
        "relation_rich": "is a type of action",
        "relation_neutral": "is a concept",
        "target_tokens": ["action"],
    },
    "animal": {
        "objects": ["dog","cat","horse","elephant","tiger","dolphin","eagle","snake",
                    "rabbit","whale","lion","bear","fox","wolf","deer",
                    "monkey","shark","frog","penguin","owl"],
        "relation_rich": "is a type of animal",
        "relation_neutral": "is a thing",
        "target_tokens": ["animal"],
    },
}

# 三模板
TEMPLATES = {
    "short_rich": "The {obj} {rel}",
    "short_neutral": "The {obj} {rel}",
    "long_rich": "In classification, the item {obj} should be understood as something that {rel}",
    "long_neutral": "In classification, the item {obj} should be understood as something that {rel}",
    "neutral_rich": "Consider: {obj} {rel}",
    "neutral_neutral": "Consider: {obj} {rel}",
}

ALL_CLASS_TOKENS = ["fruit","animal","clothing","emotion","action"]
OUTPUT_DIR = Path("results/glm5")


def get_rmsnorm_weight(model, model_name):
    for attr in ['model.norm','model.final_layernorm','model.decoder.final_layer_norm','transformer.ln_f']:
        parts = attr.split('.')
        obj = model
        ok = True
        for p in parts:
            if hasattr(obj, p): obj = getattr(obj, p)
            else: ok = False; break
        if ok and hasattr(obj, 'weight'):
            w = obj.weight.detach()
            if str(w.device) != 'meta':
                return w.float().cpu().numpy()
    return None


def get_lm_head_weight(model, model_name):
    if hasattr(model, 'lm_head') and hasattr(model.lm_head, 'weight'):
        w = model.lm_head.weight.detach()
        if str(w.device) != 'meta':
            return w.float().cpu().numpy()
    return None


def get_token_ids(tokenizer, tokens_str):
    ids = []
    for t in tokens_str:
        tid = tokenizer.encode(t, add_special_tokens=False)
        if tid: ids.append(tid[0])
    return ids


def rms_norm_numpy(x, weight=None, eps=1e-5):
    rms = np.sqrt(np.mean(x**2)+eps)
    normed = x/rms
    if weight is not None: normed = normed*weight
    return normed, rms


def extract_hidden(model, tokenizer, prompt, device):
    """提取答案位置最后token的hidden state"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        if hasattr(model, 'model'):
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states
        else:
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
    # 最后一个token
    last_h = hidden_states[-1][0, -1, :].float().cpu().numpy().astype(np.float64)
    return last_h


def run_phase501(model, tokenizer, model_name):
    """
    R2: 干预差分法 — v_cat = h_rich - h_neutral
    三模板(short/long/neutral) × 20对象 × 5类别
    """
    info = get_model_info(model, model_name)
    device = model.device
    print(f"Model: {info.model_class}, layers={info.n_layers}, d_model={info.d_model}")

    # Get W_U and g
    W_U = get_W_U(model, model_name)
    if W_U is None:
        print("ERROR: Cannot get W_U")
        return None
    W_U = W_U.astype(np.float64)
    print(f"W_U: {W_U.shape}")

    g_vec = get_rmsnorm_weight(model, model_name)
    if g_vec is None:
        print("ERROR: Cannot get RMSNorm weight")
        return None
    g_vec = g_vec.astype(np.float64)
    print(f"g_vec: {g_vec.shape}")

    # Token IDs
    all_target_ids = {}
    all_comp_ids = []
    for cat_name, cfg in CATEGORIES.items():
        all_target_ids[cat_name] = get_token_ids(tokenizer, cfg["target_tokens"])
    for cat_name in CATEGORIES:
        other = [c for c in ALL_CLASS_TOKENS if c not in CATEGORIES[cat_name]["target_tokens"]]
        all_comp_ids.extend(get_token_ids(tokenizer, other))
    all_comp_ids = list(set(all_comp_ids))

    # Per-category results
    all_results = {}
    template_names = ["short", "long", "neutral"]

    for cat_name, cfg in CATEGORIES.items():
        cat_start = time.time()
        target_ids = all_target_ids.get(cat_name, [])
        
        # w_D and g⊙w_D
        w_D_target = np.mean([W_U[tid] for tid in target_ids if tid < len(W_U)], axis=0)
        w_D_comp = np.mean([W_U[cid] for cid in all_comp_ids if cid < len(W_U)], axis=0)
        w_D = w_D_target - w_D_comp
        gw_D = w_D * g_vec

        cat_result = {
            "w_D_norm": round(float(np.linalg.norm(w_D)), 4),
            "gw_D_norm": round(float(np.linalg.norm(gw_D)), 4),
            "gain_ratio": round(float(np.linalg.norm(gw_D)/np.linalg.norm(w_D)), 4),
            "templates": {},
        }

        for tmpl_name in template_names:
            tmpl_rich = TEMPLATES[f"{tmpl_name}_rich"]
            tmpl_neutral = TEMPLATES[f"{tmpl_name}_neutral"]
            
            v_cat_list = []  # h_rich - h_neutral per object
            D_rich_list = []
            D_neutral_list = []
            
            for obj in cfg["objects"]:
                # Rich prompt
                prompt_rich = tmpl_rich.format(obj=obj, rel=cfg["relation_rich"])
                h_rich = extract_hidden(model, tokenizer, prompt_rich, device)
                
                # Neutral prompt
                prompt_neutral = tmpl_neutral.format(obj=obj, rel=cfg["relation_neutral"])
                h_neutral = extract_hidden(model, tokenizer, prompt_neutral, device)
                
                # Category direction = difference
                v_cat_obj = h_rich - h_neutral
                v_cat_list.append(v_cat_obj)
                
                # DCF values
                logits_rich = h_rich @ W_U.T
                logits_neutral = h_neutral @ W_U.T
                
                tl_rich = np.mean([logits_rich[tid] for tid in target_ids if tid < len(logits_rich)])
                cl_rich = np.mean([logits_rich[cid] for cid in all_comp_ids if cid < len(logits_rich)])
                tl_neutral = np.mean([logits_neutral[tid] for tid in target_ids if tid < len(logits_neutral)])
                cl_neutral = np.mean([logits_neutral[cid] for cid in all_comp_ids if cid < len(logits_neutral)])
                
                D_rich_list.append(float(tl_rich - cl_rich))
                D_neutral_list.append(float(tl_neutral - cl_neutral))
            
            # Mean category direction for this template
            v_cat_mean = np.mean(v_cat_list, axis=0)
            v_norm = np.linalg.norm(v_cat_mean)
            wD_norm = np.linalg.norm(w_D)
            gwD_norm = np.linalg.norm(gw_D)
            
            # Cosines
            cos_v_wD = float(np.dot(v_cat_mean, w_D)/(v_norm*wD_norm)) if v_norm>0 and wD_norm>0 else 0.0
            cos_v_gwD = float(np.dot(v_cat_mean, gw_D)/(v_norm*gwD_norm)) if v_norm>0 and gwD_norm>0 else 0.0
            
            cat_result["templates"][tmpl_name] = {
                "cos_v_wD": round(cos_v_wD, 6),
                "cos_v_gwD": round(cos_v_gwD, 6),
                "alignment_gain": round(cos_v_gwD-cos_v_wD, 6),
                "norm_v": round(float(v_norm), 4),
                "D_rich_mean": round(float(np.mean(D_rich_list)), 4),
                "D_neutral_mean": round(float(np.mean(D_neutral_list)), 4),
                "D_delta": round(float(np.mean(D_rich_list)-np.mean(D_neutral_list)), 4),
                "n_objects": len(cfg["objects"]),
            }
        
        # Template-average
        all_cos_wD = [cat_result["templates"][t]["cos_v_wD"] for t in template_names]
        all_cos_gwD = [cat_result["templates"][t]["cos_v_gwD"] for t in template_names]
        all_gains = [cat_result["templates"][t]["alignment_gain"] for t in template_names]
        
        cat_result["mean_cos_v_wD"] = round(np.mean(all_cos_wD), 6)
        cat_result["mean_cos_v_gwD"] = round(np.mean(all_cos_gwD), 6)
        cat_result["mean_alignment_gain"] = round(np.mean(all_gains), 6)
        cat_result["positive_templates"] = sum(1 for g in all_gains if g > 0)
        cat_result["total_templates"] = len(all_gains)
        
        all_results[cat_name] = cat_result
        
        elapsed = time.time()-cat_start
        print(f"  {cat_name}: gain={cat_result['mean_alignment_gain']:+.5f} "
              f"({cat_result['positive_templates']}/{cat_result['total_templates']} templates +) "
              f"gain_ratio={cat_result['gain_ratio']:.2f} "
              f"[short:{cat_result['templates']['short']['alignment_gain']:+.4f} "
              f"long:{cat_result['templates']['long']['alignment_gain']:+.4f} "
              f"neutral:{cat_result['templates']['neutral']['alignment_gain']:+.4f}] "
              f"({elapsed:.1f}s)")
    
    # Summary
    all_gains = [all_results[c]["mean_alignment_gain"] for c in all_results]
    all_cos_wD = [all_results[c]["mean_cos_v_wD"] for c in all_results]
    all_cos_gwD = [all_results[c]["mean_cos_v_gwD"] for c in all_results]
    all_gr = [all_results[c]["gain_ratio"] for c in all_results]
    
    summary = {
        "model": model_name,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "mean_alignment_gain": round(np.mean(all_gains), 6),
        "std_alignment_gain": round(np.std(all_gains), 6),
        "mean_cos_v_wD": round(np.mean(all_cos_wD), 6),
        "mean_cos_v_gwD": round(np.mean(all_cos_gwD), 6),
        "mean_gain_ratio": round(np.mean(all_gr), 4),
        "positive_categories": sum(1 for g in all_gains if g > 0),
        "total_categories": len(all_gains),
        "categories": all_results,
        "method": "intervention_differential",
        "templates": template_names,
    }
    
    return summary


def main():
    if len(sys.argv)<2:
        print("Usage: python phase501_gain_alignment_r2.py <model>")
        print("  model: qwen3, glm4, deepseek7b")
        sys.exit(1)

    model_name = sys.argv[1]
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown: {model_name}, available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    print("="*70)
    print(f"Phase 501 R2: Gain-Support Alignment (Differential + 3 Templates)")
    print(f"Model: {model_name}")
    print("="*70)
    print("v_cat = h_rich - h_neutral  (category-rich minus neutral prompt)")
    print("Templates: short, long, neutral | Objects: 20/category | Categories: 5")
    print()

    # Load
    print(f"Loading {model_name}...")
    t0 = time.time()
    model, tokenizer, device = load_model(model_name)
    print(f"Loaded in {time.time()-t0:.1f}s, VRAM={torch.cuda.memory_allocated()/1e9:.2f}GB")

    try:
        results = run_phase501(model, tokenizer, model_name)
        total = time.time()-t0 if results else 0

        if results is None:
            print("ERROR: Experiment failed")
            return

        # Print
        print(f"\n{'='*70}")
        print(f"RESULTS — {model_name}")
        print(f"{'='*70}")
        print(f"  mean cos(v, w_D):     {results['mean_cos_v_wD']:+.5f}")
        print(f"  mean cos(v, g⊙w_D):   {results['mean_cos_v_gwD']:+.5f}")
        print(f"  mean alignment gain:  {results['mean_alignment_gain']:+.5f}")
        print(f"  + categories:         {results['positive_categories']}/{results['total_categories']}")
        print(f"  mean gain_ratio:      {results['mean_gain_ratio']:.2f}")
        print(f"  total time:           {total:.0f}s")

        print(f"\n  Category breakdown (3-template avg):")
        for cat,r in results['categories'].items():
            sig = "✅" if r['mean_alignment_gain']>0.005 else ("⚠️" if r['mean_alignment_gain']>-0.005 else "❌")
            print(f"  {sig} {cat:10s} Δ={r['mean_alignment_gain']:+.5f}  "
                  f"cos_wD={r['mean_cos_v_wD']:+.5f}  cos_gwD={r['mean_cos_v_gwD']:+.5f}  "
                  f"+{r['positive_templates']}/{r['total_templates']}t")

        print(f"\n  Template breakdown:")
        for tmpl in ['short','long','neutral']:
            gains = [results['categories'][c]['templates'][tmpl]['alignment_gain'] for c in results['categories']]
            print(f"    {tmpl:8s}: Δ={np.mean(gains):+.5f}  +{sum(1 for g in gains if g>0)}/5")

        # Save
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUTPUT_DIR/f"phase501_{model_name}_r1.json"
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved to {out_path}")

    finally:
        release_model(model)
        print("  Model released.")

if __name__=="__main__":
    main()
