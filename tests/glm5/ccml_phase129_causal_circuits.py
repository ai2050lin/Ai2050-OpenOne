"""
Phase 129: 因果回路分析与条件传播拓扑
========================================

Phase 128关键发现:
  1. 维度作弊确认: 轨迹3倍优势是维度效应
  2. 语义的双重性质: 内容(欧氏) + 计算(条件)
  3. 语法绑定: 同token不同语序激活不同MLP neurons

Phase 128的核心不足:
  1. 仍然在用"激活重叠"(相关性), 不是"因果贡献"(因果性)
  2. 静态snapshot分析, 没有"时序传播流"
  3. Jaccard太粗糙 — 需要真正的条件传播图

本阶段核心转变: 从"相关性"进入"因果性"

5个实验:
- Exp 1: 激活替换(Activation Patching) — 因果回路追踪
- Exp 2: 条件传播流 — Δh差异和方向相关性
- Exp 3: 因果追踪(Causal Tracing) — 哪些层对类别判断因果重要
- Exp 4: 位置特定因果分析 — 主语/宾语角色绑定在哪层
- Exp 5: 组合传播动力学 — 属性的因果贡献
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import time
import gc
import random
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict

from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, get_W_U, release_model, MODEL_CONFIGS
)


SYNTACTIC_CAUSAL_PAIRS = [
    ("The dog bites the man", "The man bites the dog"),
    ("The cat chases the mouse", "The mouse chases the cat"),
    ("The teacher praises the student", "The student praises the teacher"),
    ("The doctor helps the patient", "The patient helps the doctor"),
    ("The police arrests the criminal", "The criminal arrests the police"),
    ("The king rules the kingdom", "The kingdom rules the king"),
    ("The hunter shoots the bear", "The bear shoots the hunter"),
    ("The mother feeds the child", "The child feeds the mother"),
    ("The boss fires the worker", "The worker fires the boss"),
    ("The cat eats the fish", "The fish eats the cat"),
    ("The dog bites the man", "The man is bitten by the dog"),
    ("The cat chases the mouse", "The mouse is chased by the cat"),
    ("The dog bites the man", "The dog does not bite the man"),
    ("The cat likes the fish", "The cat does not like the fish"),
    ("The dog bit the man", "The dog will bite the man"),
    ("The cat caught the mouse", "The cat will catch the mouse"),
]

COMPOSITIONAL_TRIPLES = [
    ("the apple", "the red apple", "the red car"),
    ("the apple", "the green apple", "the green car"),
    ("the apple", "the big apple", "the big city"),
    ("the apple", "the rotten apple", "the rotten wood"),
    ("the dog", "the big dog", "the big cat"),
    ("the dog", "the fierce dog", "the fierce lion"),
    ("the dog", "the friendly dog", "the friendly cat"),
    ("the city", "the ancient city", "the ancient temple"),
    ("the city", "the modern city", "the modern building"),
    ("the knife", "the sharp knife", "the sharp stone"),
]


def get_device_for_input(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def kl_divergence(p, q):
    return float(torch.sum(torch.exp(p) * (p - q)))


def exp1_activation_patching(model, tokenizer, device, model_info):
    """激活替换 — 因果回路追踪: 在A中替换第l层为B的, 测量输出变化"""
    print("\n" + "="*60)
    print("Exp 1: 激活替换 — 因果回路追踪")
    print("="*60)
    
    n_layers = model_info.n_layers
    layers = get_layers(model)
    sample_layers = sorted(set(list(range(0, n_layers, max(1, n_layers // 8))) + [n_layers - 1]))
    
    results = []
    for s1, s2 in SYNTACTIC_CAUSAL_PAIRS:
        print(f"\n  Patching: '{s1}' <-> '{s2}'")
        
        with torch.no_grad():
            inputs1 = tokenizer(s1, return_tensors="pt", truncation=True, max_length=128)
            inputs2 = tokenizer(s2, return_tensors="pt", truncation=True, max_length=128)
            input_ids1, attn_mask1 = inputs1["input_ids"].to(device), inputs1["attention_mask"].to(device)
            input_ids2, attn_mask2 = inputs2["input_ids"].to(device), inputs2["attention_mask"].to(device)
            
            out1 = model(input_ids=input_ids1, attention_mask=attn_mask1, output_hidden_states=True)
            out2 = model(input_ids=input_ids2, attention_mask=attn_mask2, output_hidden_states=True)
            hs2 = out2.hidden_states
            lp1 = torch.log_softmax(out1.logits[0, -1, :].float().cpu(), dim=-1)
            lp2 = torch.log_softmax(out2.logits[0, -1, :].float().cpu(), dim=-1)
        
        baseline_kl = kl_divergence(lp1, lp2)
        patch_effects = {}
        
        for li in sample_layers:
            h_b = hs2[li].clone().to(device)
            
            def make_patch_hook(patch_tensor):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        patched = output[0].clone()
                        patched[0, -1, :] = patch_tensor[0, -1, :]
                        return (patched,) + output[1:]
                    return output
                return hook
            
            hook = layers[li].register_forward_hook(make_patch_hook(h_b))
            with torch.no_grad():
                try:
                    out_patched = model(input_ids=input_ids1, attention_mask=attn_mask1)
                except Exception:
                    hook.remove()
                    continue
            hook.remove()
            
            lp_patched = torch.log_softmax(out_patched.logits[0, -1, :].float().cpu(), dim=-1)
            patched_kl = kl_divergence(lp_patched, lp2)
            causal_effect = max(0, min(1, (baseline_kl - patched_kl) / max(baseline_kl, 1e-8)))
            patch_effects[li] = {"baseline_kl": round(baseline_kl, 6), "patched_kl": round(patched_kl, 6), "causal_effect": round(causal_effect, 4)}
        
        max_layer = max(patch_effects, key=lambda k: patch_effects[k]["causal_effect"]) if patch_effects else -1
        max_effect = patch_effects[max_layer]["causal_effect"] if patch_effects else 0
        print(f"    Baseline KL: {baseline_kl:.4f}, Max causal: L{max_layer} (effect={max_effect:.4f})")
        
        effects_by_region = {
            "early": np.mean([patch_effects[li]["causal_effect"] for li in sample_layers if li < n_layers // 3]) or 0,
            "middle": np.mean([patch_effects[li]["causal_effect"] for li in sample_layers if n_layers // 3 <= li < 2 * n_layers // 3]) or 0,
            "late": np.mean([patch_effects[li]["causal_effect"] for li in sample_layers if li >= 2 * n_layers // 3]) or 0,
        }
        results.append({"pair": (s1, s2), "baseline_kl": round(baseline_kl, 4), "max_causal_layer": max_layer, "max_causal_effect": max_effect, "effects_by_region": effects_by_region, "patch_effects": patch_effects})
    
    swap_results = [r for r in results if any(k in r["pair"][0].lower() for k in ["bites", "chases", "praises", "helps", "arrests", "rules", "shoots", "feeds", "fires", "eats"])]
    passive_results = [r for r in results if " is " in r["pair"][1] and " by " in r["pair"][1]]
    negation_results = [r for r in results if " not " in r["pair"][1]]
    tense_results = [r for r in results if " will " in r["pair"][1] and " not " not in r["pair"][1]]
    
    def avg_eff(res_list, region="middle"):
        vals = [r["effects_by_region"][region] for r in res_list]
        return round(np.mean(vals), 4) if vals else 0
    
    summary = {
        "swap_pairs": {"n": len(swap_results), "avg_early": avg_eff(swap_results, "early"), "avg_middle": avg_eff(swap_results, "middle"), "avg_late": avg_eff(swap_results, "late")},
        "passive_pairs": {"n": len(passive_results), "avg_early": avg_eff(passive_results, "early"), "avg_middle": avg_eff(passive_results, "middle"), "avg_late": avg_eff(passive_results, "late")},
        "negation_pairs": {"n": len(negation_results), "avg_early": avg_eff(negation_results, "early"), "avg_middle": avg_eff(negation_results, "middle"), "avg_late": avg_eff(negation_results, "late")},
        "tense_pairs": {"n": len(tense_results), "avg_early": avg_eff(tense_results, "early"), "avg_middle": avg_eff(tense_results, "middle"), "avg_late": avg_eff(tense_results, "late")},
    }
    print(f"\n  Summary: swap(mid)={summary['swap_pairs']['avg_middle']}, passive(mid)={summary['passive_pairs']['avg_middle']}, neg(mid)={summary['negation_pairs']['avg_middle']}, tense(mid)={summary['tense_pairs']['avg_middle']}")
    
    return {"summary": summary, "pair_results": results}


def exp2_propagation_flow(model, tokenizer, device, model_info):
    """条件传播流 — Δh差异和方向相关性"""
    print("\n" + "="*60)
    print("Exp 2: 条件传播流")
    print("="*60)
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    target_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4]
    
    test_pairs = [
        ("The cat sat quietly", "The dog ran quickly"),
        ("The cat sat quietly", "The apple fell down"),
        ("The king rules", "The slave obeys"),
        ("The dog bites the man", "The man bites the dog"),
        ("The red apple", "The green apple"),
        ("The big city", "The small village"),
    ]
    
    n_random_dirs = 50
    results = {}
    
    for s1, s2 in test_pairs:
        with torch.no_grad():
            inputs1 = tokenizer(s1, return_tensors="pt", truncation=True, max_length=64)
            inputs2 = tokenizer(s2, return_tensors="pt", truncation=True, max_length=64)
            out1 = model(input_ids=inputs1["input_ids"].to(device), attention_mask=inputs1["attention_mask"].to(device), output_hidden_states=True)
            out2 = model(input_ids=inputs2["input_ids"].to(device), attention_mask=inputs2["attention_mask"].to(device), output_hidden_states=True)
        
        pair_diff = {}
        for li in target_layers:
            if li >= n_layers - 1:
                continue
            h_l_1 = out1.hidden_states[li][0, -1, :].float().cpu()
            h_lp1_1 = out1.hidden_states[li + 1][0, -1, :].float().cpu()
            h_l_2 = out2.hidden_states[li][0, -1, :].float().cpu()
            h_lp1_2 = out2.hidden_states[li + 1][0, -1, :].float().cpu()
            
            delta_1 = h_lp1_1 - h_l_1
            delta_2 = h_lp1_2 - h_l_2
            cos_delta = float(F.cosine_similarity(delta_1.unsqueeze(0), delta_2.unsqueeze(0)))
            
            torch.manual_seed(42)
            dir_corrs = []
            for _ in range(n_random_dirs):
                v = torch.randn(d_model)
                v = v / torch.norm(v)
                proj_1 = float(torch.dot(delta_1, v))
                proj_2 = float(torch.dot(delta_2, v))
                dir_corrs.append((proj_1, proj_2))
            
            p1 = np.array([d[0] for d in dir_corrs])
            p2 = np.array([d[1] for d in dir_corrs])
            dir_corr = float(np.corrcoef(p1, p2)[0, 1]) if np.std(p1) > 1e-8 and np.std(p2) > 1e-8 else 0.0
            
            pair_diff[li] = {"cos_delta": round(cos_delta, 4), "directional_correlation": round(dir_corr, 4)}
        
        results[f"{s1}|||{s2}"] = pair_diff
        print(f"  '{s1}' vs '{s2}': {pair_diff}")
    
    return {"pair_results": results}


def exp3_causal_tracing(model, tokenizer, device, model_info):
    """因果追踪 — 哪些层对类别判断因果重要"""
    print("\n" + "="*60)
    print("Exp 3: 因果追踪 — 层级因果重要性")
    print("="*60)
    
    n_layers = model_info.n_layers
    layers = get_layers(model)
    sample_layers = sorted(set(list(range(0, n_layers, max(1, n_layers // 8))) + [n_layers - 1]))
    
    TEST_ITEMS = [
        ("cat", "animal", "tool"), ("dog", "animal", "tool"),
        ("horse", "animal", "place"), ("lion", "animal", "fruit"),
        ("apple", "fruit", "animal"), ("banana", "fruit", "tool"),
        ("orange", "fruit", "place"), ("mango", "fruit", "animal"),
        ("city", "place", "animal"), ("mountain", "place", "fruit"),
        ("river", "place", "tool"), ("forest", "place", "animal"),
        ("hammer", "tool", "animal"), ("knife", "tool", "fruit"),
        ("drill", "tool", "place"), ("wrench", "tool", "animal"),
    ]
    
    results = {}
    for word, target_cat, contrast_cat in TEST_ITEMS:
        prompt_t = f"The {word} is a"
        prompt_c = f"The {contrast_cat} is a"
        target_tok_ids = tokenizer.encode(target_cat, add_special_tokens=False)
        if not target_tok_ids:
            continue
        target_id = target_tok_ids[0]
        
        inputs_t = tokenizer(prompt_t, return_tensors="pt", truncation=True, max_length=64)
        inputs_c = tokenizer(prompt_c, return_tensors="pt", truncation=True, max_length=64)
        ids_t, mask_t = inputs_t["input_ids"].to(device), inputs_t["attention_mask"].to(device)
        ids_c, mask_c = inputs_c["input_ids"].to(device), inputs_c["attention_mask"].to(device)
        
        with torch.no_grad():
            out_t = model(input_ids=ids_t, attention_mask=mask_t, output_hidden_states=True)
            out_c = model(input_ids=ids_c, attention_mask=mask_c, output_hidden_states=True)
            hs_c = out_c.hidden_states
            baseline_lp = float(torch.log_softmax(out_t.logits[0, -1, :].float().cpu(), dim=-1)[target_id])
        
        layer_effects = {}
        for li in sample_layers:
            h_c = hs_c[li].clone().to(device)
            def make_hook(patch_tensor):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        patched = output[0].clone()
                        patched[0, -1, :] = patch_tensor[0, -1, :]
                        return (patched,) + output[1:]
                    return output
                return hook
            
            hook = layers[li].register_forward_hook(make_hook(h_c))
            with torch.no_grad():
                try:
                    out_p = model(input_ids=ids_t, attention_mask=mask_t)
                except Exception:
                    hook.remove()
                    continue
            hook.remove()
            
            patched_lp = float(torch.log_softmax(out_p.logits[0, -1, :].float().cpu(), dim=-1)[target_id])
            layer_effects[li] = round(baseline_lp - patched_lp, 4)
        
        max_layer = max(layer_effects, key=layer_effects.get) if layer_effects else -1
        results[word] = {"category": target_cat, "contrast": contrast_cat, "max_causal_layer": max_layer, "max_effect": layer_effects.get(max_layer, 0), "layer_effects": layer_effects}
    
    # Summary
    cat_layers = defaultdict(list)
    for w, d in results.items():
        cat_layers[d["category"]].append(d["max_causal_layer"])
    cat_summary = {cat: {"mean_layer": round(np.mean(ll), 1), "std_layer": round(np.std(ll), 1)} for cat, ll in cat_layers.items()}
    
    print(f"  Category causal layers: {cat_summary}")
    return {"n_words": len(results), "category_summary": cat_summary, "word_results": {k: {kk: vv for kk, vv in v.items() if kk != "layer_effects"} for k, v in results.items()}}


def exp4_position_causal(model, tokenizer, device, model_info):
    """位置特定因果分析 — 主语/宾语角色绑定"""
    print("\n" + "="*60)
    print("Exp 4: 位置特定因果分析")
    print("="*60)
    
    n_layers = model_info.n_layers
    layers = get_layers(model)
    sample_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    
    test_pairs = [
        ("The dog bites the man", "The man bites the dog"),
        ("The cat chases the mouse", "The mouse chases the cat"),
        ("The teacher praises the student", "The student praises the teacher"),
    ]
    
    results = []
    for s1, s2 in test_pairs:
        inputs1 = tokenizer(s1, return_tensors="pt", truncation=True, max_length=64)
        inputs2 = tokenizer(s2, return_tensors="pt", truncation=True, max_length=64)
        ids1, mask1 = inputs1["input_ids"].to(device), inputs1["attention_mask"].to(device)
        ids2, mask2 = inputs2["input_ids"].to(device), inputs2["attention_mask"].to(device)
        toks1 = [tokenizer.decode([t]) for t in ids1[0].tolist()]
        
        with torch.no_grad():
            out1 = model(input_ids=ids1, attention_mask=mask1)
            out2 = model(input_ids=ids2, attention_mask=mask2, output_hidden_states=True)
            lp1 = torch.log_softmax(out1.logits[0, -1, :].float().cpu(), dim=-1)
            lp2 = torch.log_softmax(out2.logits[0, -1, :].float().cpu(), dim=-1)
            hs2 = out2.hidden_states
        baseline_kl = kl_divergence(lp1, lp2)
        
        min_len = min(ids1.shape[1], ids2.shape[1])
        position_effects = {}
        
        for pos in range(1, min_len - 1):
            pos_label = toks1[pos] if pos < len(toks1) else f"pos{pos}"
            pos_effects = {}
            
            for li in sample_layers:
                h_b_pos = hs2[li][0, pos, :].clone().detach().to(device)
                
                def make_pos_hook(position, patch_vec):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            patched = output[0].clone()
                            if position < patched.shape[1]:
                                patched[0, position, :] = patch_vec
                            return (patched,) + output[1:]
                        return output
                    return hook
                
                hook = layers[li].register_forward_hook(make_pos_hook(pos, h_b_pos))
                with torch.no_grad():
                    try:
                        out_p = model(input_ids=ids1, attention_mask=mask1)
                    except Exception:
                        hook.remove()
                        continue
                hook.remove()
                
                lp_p = torch.log_softmax(out_p.logits[0, -1, :].float().cpu(), dim=-1)
                patched_kl = kl_divergence(lp_p, lp2)
                effect = max(0, min(1, (baseline_kl - patched_kl) / max(baseline_kl, 1e-8)))
                pos_effects[li] = round(effect, 4)
            
            max_li = max(pos_effects, key=pos_effects.get) if pos_effects else -1
            position_effects[pos] = {"token": pos_label, "max_effect_layer": max_li, "max_effect": pos_effects.get(max_li, 0), "layer_effects": pos_effects}
            print(f"    pos={pos} ({pos_label}): max at L{max_li} = {pos_effects.get(max_li, 0):.4f}")
        
        results.append({"pair": (s1, s2), "position_effects": position_effects})
    
    return {"n_pairs": len(results), "pair_results": results}


def exp5_compositional_causal(model, tokenizer, device, model_info):
    """组合传播动力学 — 属性因果贡献"""
    print("\n" + "="*60)
    print("Exp 5: 组合传播动力学")
    print("="*60)
    
    results = {}
    for base, modified, different_noun in COMPOSITIONAL_TRIPLES:
        with torch.no_grad():
            inputs_b = tokenizer(base, return_tensors="pt", truncation=True, max_length=64)
            inputs_m = tokenizer(modified, return_tensors="pt", truncation=True, max_length=64)
            inputs_d = tokenizer(different_noun, return_tensors="pt", truncation=True, max_length=64)
            
            out_b = model(input_ids=inputs_b["input_ids"].to(device), attention_mask=inputs_b["attention_mask"].to(device))
            out_m = model(input_ids=inputs_m["input_ids"].to(device), attention_mask=inputs_m["attention_mask"].to(device))
            out_d = model(input_ids=inputs_d["input_ids"].to(device), attention_mask=inputs_d["attention_mask"].to(device))
            
            lp_b = torch.log_softmax(out_b.logits[0, -1, :].float().cpu(), dim=-1)
            lp_m = torch.log_softmax(out_m.logits[0, -1, :].float().cpu(), dim=-1)
            lp_d = torch.log_softmax(out_d.logits[0, -1, :].float().cpu(), dim=-1)
        
        base_mod_kl = kl_divergence(lp_b, lp_m)
        base_diff_kl = kl_divergence(lp_b, lp_d)
        mod_diff_kl = kl_divergence(lp_m, lp_d)
        
        results[f"{base}|||{modified}"] = {"base_mod_kl": round(base_mod_kl, 4), "base_diff_kl": round(base_diff_kl, 4), "mod_diff_kl": round(mod_diff_kl, 4)}
    
    avg_bm = np.mean([v["base_mod_kl"] for v in results.values()])
    avg_bd = np.mean([v["base_diff_kl"] for v in results.values()])
    avg_md = np.mean([v["mod_diff_kl"] for v in results.values()])
    print(f"  Avg KL(base||modified)={avg_bm:.4f}, KL(base||diff_noun)={avg_bd:.4f}, KL(mod||diff_noun)={avg_md:.4f}")
    
    return {"n_triples": len(results), "avg_base_mod_kl": round(avg_bm, 4), "avg_base_diff_kl": round(avg_bd, 4), "avg_mod_diff_kl": round(avg_md, 4), "triple_results": results}


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    print(f"\n{'#'*60}")
    print(f"Phase 129: 因果回路分析与条件传播拓扑 — {model_name}")
    print(f"{'#'*60}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    
    results = {
        "model": model_name,
        "model_info": {"class": model_info.model_class, "n_layers": model_info.n_layers, "d_model": model_info.d_model, "intermediate_size": model_info.intermediate_size},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    try:
        results["exp1_activation_patching"] = exp1_activation_patching(model, tokenizer, device, model_info)
        gc.collect(); torch.cuda.empty_cache()
        
        results["exp2_propagation_flow"] = exp2_propagation_flow(model, tokenizer, device, model_info)
        gc.collect(); torch.cuda.empty_cache()
        
        results["exp3_causal_tracing"] = exp3_causal_tracing(model, tokenizer, device, model_info)
        gc.collect(); torch.cuda.empty_cache()
        
        results["exp4_position_causal"] = exp4_position_causal(model, tokenizer, device, model_info)
        gc.collect(); torch.cuda.empty_cache()
        
        results["exp5_compositional_causal"] = exp5_compositional_causal(model, tokenizer, device, model_info)
    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()
        results["error"] = str(e)
    finally:
        release_model(model)
    
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'glm5_temp')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"phase129_{model_name}_causal_circuits.json")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
