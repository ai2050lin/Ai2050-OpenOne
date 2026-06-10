"""
Phase 437b: 扩展属性-类别中介确认测试 (R2)
===========================================
扩大对象集和属性集，确认Phase 437的核心发现：
- Qwen3: 属性由类别中介
- GLM4: 属性不由类别中介
- DS7B: 弱/混合中介

扩展：
1. 更多对象(增加orange, cat, hammer, train等)
2. 更多属性维度(shape, size)
3. 更多alpha值

用法:
  python tests/glm5/phase437b_extended_mediation.py qwen3 2
  python tests/glm5/phase437b_extended_mediation.py glm4 2
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, 'tests/glm5')

import time
import json
import os
import numpy as np
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from model_utils import get_model_info, get_W_U, release_model, MODEL_CONFIGS

# 扩展测试集
EXTENDED_TESTS = [
    # fruit → animal
    {
        "objs": ["apple", "orange", "lemon", "grape"],
        "src_cat": "fruit", "tgt_cat": "animal",
        "src_props": ["red", "sweet", "seed", "juicy", "ripe", "tree"],
        "tgt_props": ["fur", "alive", "leg", "tail", "eye", "ear"],
    },
    # tool → vehicle
    {
        "objs": ["knife", "hammer", "spoon", "axe"],
        "src_cat": "tool", "tgt_cat": "vehicle",
        "src_props": ["sharp", "metal", "blade", "handle", "cut", "steel"],
        "tgt_props": ["wheel", "engine", "road", "fast", "drive", "seat"],
    },
]

CATEGORY_WORDS = {
    "fruit":   ["fruit", "berry"],
    "animal":  ["animal", "creature"],
    "tool":    ["tool", "instrument"],
    "vehicle": ["vehicle", "transport"],
}

def get_cat_dir(W_E, tokenizer, src_cat, tgt_cat):
    src_words = CATEGORY_WORDS.get(src_cat, [src_cat])
    tgt_words = CATEGORY_WORDS.get(tgt_cat, [tgt_cat])
    src_vecs = [W_E[tokenizer.encode(w, add_special_tokens=False)[0]] 
                for w in src_words if tokenizer.encode(w, add_special_tokens=False)]
    tgt_vecs = [W_E[tokenizer.encode(w, add_special_tokens=False)[0]] 
                for w in tgt_words if tokenizer.encode(w, add_special_tokens=False)]
    if not src_vecs or not tgt_vecs:
        return None
    direction = np.mean(tgt_vecs, axis=0) - np.mean(src_vecs, axis=0)
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 0 else direction

def get_prop_logits(logits, tokenizer, props):
    result = {}
    for p in props:
        ids = tokenizer.encode(p, add_special_tokens=False)
        if ids and ids[0] < len(logits):
            result[p] = float(logits[ids[0]])
    return result

def run_experiment(model_name: str, round_num: int = 2):
    print(f"\n{'='*60}")
    print(f"Phase 437b: 扩展属性-类别中介 - {model_name} R{round_num}")
    print(f"{'='*60}")
    t_start = time.time()
    
    cfg = MODEL_CONFIGS[model_name]
    print(f"[1] Loading {model_name} (bf16+auto)...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, local_files_only=True, attn_implementation="eager")
    model.eval()
    device = next(model.parameters()).device
    
    info = get_model_info(model, model_name)
    W_E = model.get_input_embeddings().weight.detach().cpu().float().numpy()
    W_U = get_W_U(model, model_name)
    print(f"  n_layers={info.n_layers}")
    
    results = {
        "model": model_name, "round": round_num,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tests": {},
    }
    
    for test in EXTENDED_TESTS:
        objs = test["objs"]
        src_cat = test["src_cat"]
        tgt_cat = test["tgt_cat"]
        src_props = test["src_props"]
        tgt_props = test["tgt_props"]
        
        push_dir = get_cat_dir(W_E, tokenizer, src_cat, tgt_cat)
        if push_dir is None:
            continue
        
        test_key = f"{src_cat}_to_{tgt_cat}"
        test_results = {}
        
        for obj in objs:
            prompt = f"An {obj} is a kind of" if obj[0] in "aeiouAEIOU" else f"A {obj} is a kind of"
            toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = toks["input_ids"].to(device)
            seq_len = input_ids.shape[1]
            obj_pos = 1
            
            embed_layer = model.get_input_embeddings()
            inputs_embeds_clean = embed_layer(input_ids).detach().clone().to(model.dtype)
            pos_norm = inputs_embeds_clean[0, obj_pos].float().norm().item()
            
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            with torch.no_grad():
                out_bl = model(inputs_embeds=inputs_embeds_clean, position_ids=position_ids)
            logits_bl = out_bl.logits[0, -1].float().cpu().numpy()
            
            src_bl = get_prop_logits(logits_bl, tokenizer, src_props)
            tgt_bl = get_prop_logits(logits_bl, tokenizer, tgt_props)
            
            push_dir_t = torch.tensor(push_dir, dtype=inputs_embeds_clean.dtype, device=device)
            obj_result = {}
            
            for alpha in [1.0, 2.0, 4.0]:
                beta = alpha * pos_norm if pos_norm > 0 else alpha
                inputs_embeds_pushed = inputs_embeds_clean.clone()
                inputs_embeds_pushed[0, obj_pos, :] += (beta * push_dir_t).to(model.dtype)
                
                with torch.no_grad():
                    out_pushed = model(inputs_embeds=inputs_embeds_pushed, position_ids=position_ids)
                logits_pushed = out_pushed.logits[0, -1].float().cpu().numpy()
                
                if np.any(np.isnan(logits_pushed)):
                    obj_result[alpha] = {"error": "NaN"}
                    continue
                
                src_pushed = get_prop_logits(logits_pushed, tokenizer, src_props)
                tgt_pushed = get_prop_logits(logits_pushed, tokenizer, tgt_props)
                
                src_deltas = {k: round(src_pushed.get(k, 0) - src_bl.get(k, 0), 4) for k in src_props}
                tgt_deltas = {k: round(tgt_pushed.get(k, 0) - tgt_bl.get(k, 0), 4) for k in tgt_props}
                
                src_mean = np.mean(list(src_deltas.values())) if src_deltas else 0
                tgt_mean = np.mean(list(tgt_deltas.values())) if tgt_deltas else 0
                
                obj_result[alpha] = {
                    "src_mean": round(float(src_mean), 4),
                    "tgt_mean": round(float(tgt_mean), 4),
                    "mediation": round(float(tgt_mean - src_mean), 4),
                }
            
            test_results[obj] = obj_result
            print(f"  {obj}({src_cat}->{tgt_cat}): med(a2)={obj_result.get(2.0, {}).get('mediation', '?')}")
        
        # 类别平均
        med_a2 = [test_results[obj][2.0]["mediation"] for obj in objs 
                  if 2.0 in test_results[obj] and "mediation" in test_results[obj][2.0]]
        med_a4 = [test_results[obj][4.0]["mediation"] for obj in objs 
                  if 4.0 in test_results[obj] and "mediation" in test_results[obj][4.0]]
        
        avg_med_a2 = np.mean(med_a2) if med_a2 else 0
        avg_med_a4 = np.mean(med_a4) if med_a4 else 0
        
        results["tests"][test_key] = {
            "per_object": test_results,
            "avg_mediation_a2": round(float(avg_med_a2), 4),
            "avg_mediation_a4": round(float(avg_med_a4), 4),
        }
        
        print(f"  >> Average: med(a2)={avg_med_a2:.4f}, med(a4)={avg_med_a4:.4f}")
    
    os.makedirs("results/phase437_category_property_mediation", exist_ok=True)
    out_path = f"results/phase437_category_property_mediation/{model_name}_phase437b_r{round_num}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[4] Results saved to {out_path}")
    
    release_model(model)
    print(f"[5] Total time: {(time.time()-t_start)/60:.1f}min")
    return results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    run_experiment(model_name, round_num)
