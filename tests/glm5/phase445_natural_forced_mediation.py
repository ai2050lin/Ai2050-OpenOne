"""
Phase 445: Natural vs Forced Mediation Standardization
========================================================
目标: 把Qwen3和GLM4的中介差异变成稳定指标

定义:
- NaturalMediationScore = 在小alpha(0.5)且流形内时的mediation
- ForcedMediationScore = 在大alpha(2.0)时的mediation
- MediationDifferential = ForcedMed - NaturalMed
  (越大说明正常范围下中介越弱，需要极端推动才有效)

方法:
1. 对每个对象在alpha=0.5和alpha=2.0分别做类别推动
2. 记录: src_prop_delta, tgt_prop_delta, cat_shift, mediation
3. 过滤: 只在cat_shift合理(>-0.5)的条件下统计

用法:
  python tests/glm5/phase445_natural_forced_mediation.py qwen3 1
  python tests/glm5/phase445_natural_forced_mediation.py glm4 1
  python tests/glm5/phase445_natural_forced_mediation.py deepseek7b 1
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
    "apple": {"template": "The {obj} is a", "category": "fruit",
              "cat_words": ["fruit", "apple", "orange", "banana"],
              "opp_words": ["animal", "dog", "cat", "horse"],
              "src_props": ["red", "sweet", "juicy"],
              "tgt_props": ["fur", "alive", "legs"]},
    "knife": {"template": "The {obj} is a", "category": "tool",
              "cat_words": ["tool", "knife", "hammer", "scissors"],
              "opp_words": ["vehicle", "car", "bus", "train"],
              "src_props": ["sharp", "metal", "blade"],
              "tgt_props": ["fast", "wheels", "engine"]},
    "dog": {"template": "The {obj} is a", "category": "animal",
            "cat_words": ["animal", "dog", "cat", "horse"],
            "opp_words": ["fruit", "apple", "orange", "banana"],
            "src_props": ["fur", "alive", "legs"],
            "tgt_props": ["red", "sweet", "juicy"]},
    "car": {"template": "The {obj} is a", "category": "vehicle",
            "cat_words": ["vehicle", "car", "bus", "train"],
            "opp_words": ["tool", "knife", "hammer", "scissors"],
            "src_props": ["fast", "wheels", "engine"],
            "tgt_props": ["sharp", "metal", "blade"]},
    "orange": {"template": "The {obj} is a", "category": "fruit",
               "cat_words": ["fruit", "apple", "orange", "banana"],
               "opp_words": ["animal", "dog", "cat", "horse"],
               "src_props": ["orange", "sweet", "round"],
               "tgt_props": ["fur", "alive", "legs"]},
    "hammer": {"template": "The {obj} is a", "category": "tool",
               "cat_words": ["tool", "knife", "hammer", "scissors"],
               "opp_words": ["vehicle", "car", "bus", "train"],
               "src_props": ["heavy", "metal", "handle"],
               "tgt_props": ["fast", "wheels", "engine"]},
}

ALPHAS = [0.5, 2.0]  # Natural and Forced


def load_model_auto(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, local_files_only=True, attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


def get_cat_direction(model, tokenizer, cat_words, opp_words):
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        W_E = model.model.embed_tokens.weight.detach().float()
    elif hasattr(model, 'get_input_embeddings'):
        W_E = model.get_input_embeddings().weight.detach().float()
    else:
        return None
    cat_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in cat_words]
    opp_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in opp_words]
    d = W_E[cat_ids].mean(dim=0) - W_E[opp_ids].mean(dim=0)
    d = d.cpu()
    d = d / (d.norm() + 1e-8)
    return d


def run_with_injection(model, tokenizer, input_ids, attention_mask, last_pos,
                       cat_dir, alpha):
    """注入类别方向扰动并返回logits"""
    input_device = next(model.parameters()).device
    perturb_vec = (alpha * cat_dir).to(input_device).to(torch.bfloat16)
    
    embed_hook = None
    def on_embed(module, inp, out):
        if isinstance(out, torch.Tensor):
            out = out.clone()
            out[0, last_pos] = out[0, last_pos] + perturb_vec
        return out
    
    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
        embed_hook = model.model.embed_tokens.register_forward_hook(on_embed)
    
    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[0, -1].float().cpu().numpy()
    except Exception as e:
        print(f"    Forward failed: {e}")
        logits = None
    finally:
        if embed_hook is not None:
            embed_hook.remove()
    
    return logits


def run_experiment(model_name, round_num):
    print(f"\n{'='*60}")
    print(f"Phase 445: Natural vs Forced Mediation Standardization")
    print(f"Model: {model_name}, Round: {round_num}")
    print(f"{'='*60}")
    
    print("\n[1] Loading model...")
    t0 = time.time()
    model, tokenizer = load_model_auto(model_name)
    info = get_model_info(model, model_name)
    print(f"  Loaded: {info.model_class}, {info.n_layers} layers")
    print(f"  Load time: {time.time()-t0:.1f}s")
    
    results = {}
    
    for obj_name, obj_info in OBJECTS.items():
        print(f"\n  Processing {obj_name} ({obj_info['category']})...")
        
        cat_dir = get_cat_direction(model, tokenizer, obj_info["cat_words"], obj_info["opp_words"])
        if cat_dir is None:
            continue
        
        text = obj_info["template"].format(obj=obj_name)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)
        last_pos = input_ids.shape[1] - 1
        
        # 基准logits
        with torch.no_grad():
            base_out = model(input_ids=input_ids, attention_mask=attention_mask)
        base_logits = base_out.logits[0, -1].float().cpu().numpy()
        
        # 基准属性logits
        base_src_props = {w: float(base_logits[tokenizer.encode(w, add_special_tokens=False)[0]])
                         for w in obj_info["src_props"]}
        base_tgt_props = {w: float(base_logits[tokenizer.encode(w, add_special_tokens=False)[0]])
                         for w in obj_info["tgt_props"]}
        base_cat_gap = float(base_logits[[tokenizer.encode(w, add_special_tokens=False)[0] for w in obj_info["cat_words"]]].mean()) - \
                       float(base_logits[[tokenizer.encode(w, add_special_tokens=False)[0] for w in obj_info["opp_words"]]].mean())
        
        alpha_results = {}
        
        for alpha in ALPHAS:
            pert_logits = run_with_injection(model, tokenizer, input_ids, attention_mask,
                                            last_pos, cat_dir, alpha)
            if pert_logits is None:
                continue
            
            # 扰动后属性logits
            pert_src_props = {w: float(pert_logits[tokenizer.encode(w, add_special_tokens=False)[0]])
                             for w in obj_info["src_props"]}
            pert_tgt_props = {w: float(pert_logits[tokenizer.encode(w, add_special_tokens=False)[0]])
                             for w in obj_info["tgt_props"]}
            pert_cat_gap = float(pert_logits[[tokenizer.encode(w, add_special_tokens=False)[0] for w in obj_info["cat_words"]]].mean()) - \
                           float(pert_logits[[tokenizer.encode(w, add_special_tokens=False)[0] for w in obj_info["opp_words"]]].mean())
            
            # 计算变化
            src_prop_delta = np.mean([pert_src_props[w] - base_src_props[w] for w in obj_info["src_props"]])
            tgt_prop_delta = np.mean([pert_tgt_props[w] - base_tgt_props[w] for w in obj_info["tgt_props"]])
            cat_shift = pert_cat_gap - base_cat_gap
            mediation = tgt_prop_delta - src_prop_delta  # 正=类别推动带动属性
            
            alpha_results[str(alpha)] = {
                "alpha": alpha,
                "src_prop_delta": round(float(src_prop_delta), 4),
                "tgt_prop_delta": round(float(tgt_prop_delta), 4),
                "cat_shift": round(float(cat_shift), 4),
                "mediation": round(float(mediation), 4),
                "src_props": {w: round(v - base_src_props[w], 4) for w, v in pert_src_props.items()},
                "tgt_props": {w: round(v - base_tgt_props[w], 4) for w, v in pert_tgt_props.items()},
            }
            
            print(f"    alpha={alpha}: src_delta={src_prop_delta:.4f}, tgt_delta={tgt_prop_delta:.4f}, "
                  f"cat_shift={cat_shift:.4f}, mediation={mediation:.4f}")
        
        # 计算中介指标
        natural_med = alpha_results.get("0.5", {}).get("mediation", 0)
        forced_med = alpha_results.get("2.0", {}).get("mediation", 0)
        med_differential = forced_med - natural_med
        
        results[obj_name] = {
            "category": obj_info["category"],
            "alpha_results": alpha_results,
            "natural_mediation": round(float(natural_med), 4),
            "forced_mediation": round(float(forced_med), 4),
            "mediation_differential": round(float(med_differential), 4),
        }
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ===== 汇总 =====
    print(f"\n{'='*60}")
    print("PHASE 445 SUMMARY")
    print(f"{'='*60}")
    
    natural_meds = []
    forced_meds = []
    diff_meds = []
    
    for obj_name, r in results.items():
        print(f"\n  {obj_name} ({r['category']}):")
        print(f"    Natural mediation (alpha=0.5): {r['natural_mediation']}")
        print(f"    Forced mediation (alpha=2.0): {r['forced_mediation']}")
        print(f"    Mediation differential: {r['mediation_differential']}")
        natural_meds.append(r['natural_mediation'])
        forced_meds.append(r['forced_mediation'])
        diff_meds.append(r['mediation_differential'])
    
    if natural_meds:
        print(f"\n  === Model Level ===")
        print(f"  Avg Natural Mediation: {np.mean(natural_meds):.4f}")
        print(f"  Avg Forced Mediation: {np.mean(forced_meds):.4f}")
        print(f"  Avg Mediation Differential: {np.mean(diff_meds):.4f}")
        
        # 分类: 自然中介型 vs 强制中介型
        if np.mean(natural_meds) > 0.05:
            verdict = "NATURAL MEDIATION (category→property is natural pathway)"
        elif np.mean(forced_meds) > 0.1:
            verdict = "FORCED MEDIATION (category→property only under extreme push)"
        else:
            verdict = "WEAK/NO MEDIATION"
        print(f"  Verdict: {verdict}")
    
    # 保存
    output = {
        "model": model_name,
        "round": round_num,
        "n_layers": info.n_layers,
        "alphas": ALPHAS,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "per_object": results,
        "summary": {
            "avg_natural_mediation": round(float(np.mean(natural_meds)), 4) if natural_meds else 0,
            "avg_forced_mediation": round(float(np.mean(forced_meds)), 4) if forced_meds else 0,
            "avg_mediation_differential": round(float(np.mean(diff_meds)), 4) if diff_meds else 0,
        }
    }
    
    os.makedirs("results/phase445_natural_forced_mediation", exist_ok=True)
    out_path = f"results/phase445_natural_forced_mediation/{model_name}_phase445_r{round_num}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=float)
    print(f"\n  Saved: {out_path}")
    
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
