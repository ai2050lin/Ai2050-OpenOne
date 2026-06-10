"""
Phase 437: 属性是否由类别中介实验
==================================
目标: 测试改变类别轨道后，属性是否跟着变

核心问题:
如果apple的类别从fruit推向animal，那么red/sweet/seeds是否下降？
fur/alive是否上升？

方法:
1. 用category方向在embedding层将apple从fruit推向animal
2. 测量属性词logit变化: red, sweet, seeds, fur, alive, bark等
3. 同理: knife从tool推向vehicle (sharp/metal→wheels/engine)

如果属性跟着类别变 → 属性由类别中介
如果属性不变 → 对象身份残差独立承载属性

用法:
  python tests/glm5/phase437_category_property_mediation.py qwen3 1
  python tests/glm5/phase437_category_property_mediation.py glm4 1
  python tests/glm5/phase437_category_property_mediation.py deepseek7b 1
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, 'tests/glm5')

import gc
import time
import json
import os
import numpy as np
import torch
from datetime import datetime

from model_utils import (load_model, get_layers, get_model_info,
                          release_model, get_W_U, MODEL_CONFIGS)


# ===== 测试配置 =====
# 每个测试: 对象 → 从源类别推向目标类别，检查属性变化
CATEGORY_PUSH_TESTS = [
    {
        "obj": "apple",
        "src_cat": "fruit", "tgt_cat": "animal",
        "push_template": "An apple is a kind of",
        "src_props": ["red", "sweet", "seed", "tree", "juicy", "ripe"],
        "tgt_props": ["fur", "alive", "bark", "leg", "tail", "eye"],
        "neutral_props": ["big", "small", "old", "new", "good", "bad"],
    },
    {
        "obj": "apple",
        "src_cat": "fruit", "tgt_cat": "tool",
        "push_template": "An apple is a kind of",
        "src_props": ["red", "sweet", "seed", "tree", "juicy", "ripe"],
        "tgt_props": ["sharp", "metal", "blade", "handle", "cut", "steel"],
        "neutral_props": ["big", "small", "old", "new", "good", "bad"],
    },
    {
        "obj": "knife",
        "src_cat": "tool", "tgt_cat": "vehicle",
        "push_template": "A knife is a kind of",
        "src_props": ["sharp", "metal", "blade", "handle", "cut", "steel"],
        "tgt_props": ["wheel", "engine", "road", "fast", "drive", "seat"],
        "neutral_props": ["big", "small", "old", "new", "good", "bad"],
    },
    {
        "obj": "dog",
        "src_cat": "animal", "tgt_cat": "fruit",
        "push_template": "A dog is a kind of",
        "src_props": ["fur", "alive", "bark", "leg", "tail", "eye"],
        "tgt_props": ["red", "sweet", "seed", "tree", "juicy", "ripe"],
        "neutral_props": ["big", "small", "old", "new", "good", "bad"],
    },
    {
        "obj": "car",
        "src_cat": "vehicle", "tgt_cat": "tool",
        "push_template": "A car is a kind of",
        "src_props": ["wheel", "engine", "road", "fast", "drive", "seat"],
        "tgt_props": ["sharp", "metal", "blade", "handle", "cut", "steel"],
        "neutral_props": ["big", "small", "old", "new", "good", "bad"],
    },
]

CATEGORY_WORDS = {
    "fruit":   ["fruit", "berry", "produce"],
    "animal":  ["animal", "creature", "beast"],
    "tool":    ["tool", "instrument", "utensil"],
    "vehicle": ["vehicle", "transport", "automobile"],
}


def get_category_direction(W_E, tokenizer, src_cat, tgt_cat):
    """计算类别推方向: W_E(tgt_center) - W_E(src_center)"""
    src_words = CATEGORY_WORDS.get(src_cat, [src_cat])
    tgt_words = CATEGORY_WORDS.get(tgt_cat, [tgt_cat])
    
    src_vecs = []
    for w in src_words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if ids:
            src_vecs.append(W_E[ids[0]])
    
    tgt_vecs = []
    for w in tgt_words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if ids:
            tgt_vecs.append(W_E[ids[0]])
    
    if not src_vecs or not tgt_vecs:
        return None
    
    src_center = np.mean(src_vecs, axis=0)
    tgt_center = np.mean(tgt_vecs, axis=0)
    direction = tgt_center - src_center
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm
    return direction


def get_logit_for_tokens(logits, tokenizer, tokens):
    """获取一组token的logit值"""
    result = {}
    for tok_str in tokens:
        ids = tokenizer.encode(tok_str, add_special_tokens=False)
        if ids and ids[0] < len(logits):
            result[tok_str] = float(logits[ids[0]])
    return result


def run_experiment(model_name: str, round_num: int = 1):
    """运行Phase 437实验"""
    print(f"\n{'='*60}")
    print(f"Phase 437: 属性是否由类别中介 - {model_name} R{round_num}")
    print(f"{'='*60}")
    t_start = time.time()
    
    # 加载模型 - 对于DS7B，使用bfloat16避免NaN
    print(f"[1] Loading {model_name}...")
    # 所有模型用bf16+auto避免8bit量化失真
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    print(f"  Loading {model_name} with bf16+auto...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, local_files_only=True, attn_implementation="eager")
    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"  [bf16+auto] {model_name} loaded, class={type(model).__name__}, GPU={gpu_mem:.2f}GB")
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    print(f"  n_layers={n_layers}, d_model={info.d_model}")
    
    # 获取W_E和W_U
    print(f"[2] Loading W_E and W_U...")
    W_E = model.get_input_embeddings().weight.detach().cpu().float().numpy()
    W_U = get_W_U(model, model_name)
    print(f"  W_E: {W_E.shape}, W_U: {W_U.shape}")
    
    results = {
        "model": model_name,
        "round": round_num,
        "n_layers": n_layers,
        "d_model": info.d_model,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "per_test": {},
    }
    
    # 注入强度扫描
    alphas = [0.5, 1.0, 2.0, 4.0]
    
    for ti, test in enumerate(CATEGORY_PUSH_TESTS):
        obj = test["obj"]
        src_cat = test["src_cat"]
        tgt_cat = test["tgt_cat"]
        
        print(f"\n[3] === Push: {obj} from {src_cat} → {tgt_cat} ===")
        
        # 获取推方向
        push_dir = get_category_direction(W_E, tokenizer, src_cat, tgt_cat)
        if push_dir is None:
            print(f"  SKIP: cannot compute push direction")
            continue
        
        prompt = test["push_template"]
        toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = toks["input_ids"].to(device)
        attention_mask = toks["attention_mask"].to(device)
        seq_len = input_ids.shape[1]
        
        # 找obj位置 (通常是第1个位置)
        obj_pos = 1
        
        embed_layer = model.get_input_embeddings()
        inputs_embeds_clean = embed_layer(input_ids).detach().clone().to(model.dtype)
        
        # 计算注入强度
        pos_norm = inputs_embeds_clean[0, obj_pos].float().norm().item()
        
        # Baseline logits
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        with torch.no_grad():
            out_baseline = model(inputs_embeds=inputs_embeds_clean, position_ids=position_ids)
        logits_baseline = out_baseline.logits[0, -1].float().cpu().numpy()
        
        # Baseline属性logits
        src_baseline = get_logit_for_tokens(logits_baseline, tokenizer, test["src_props"])
        tgt_baseline = get_logit_for_tokens(logits_baseline, tokenizer, test["tgt_props"])
        neutral_baseline = get_logit_for_tokens(logits_baseline, tokenizer, test["neutral_props"])
        
        # 类别词logits
        cat_baseline = {}
        for cat_name in [src_cat, tgt_cat]:
            for w in CATEGORY_WORDS.get(cat_name, [cat_name]):
                ids = tokenizer.encode(w, add_special_tokens=False)
                if ids and ids[0] < len(logits_baseline):
                    cat_baseline[f"{cat_name}:{w}"] = float(logits_baseline[ids[0]])
        
        print(f"  Baseline src_props: {src_baseline}")
        print(f"  Baseline tgt_props: {tgt_baseline}")
        print(f"  Baseline categories: {cat_baseline}")
        
        # 注入方向后获取logits
        push_dir_t = torch.tensor(push_dir, dtype=inputs_embeds_clean.dtype, device=device)
        
        alpha_results = {}
        for alpha in alphas:
            beta = alpha * pos_norm if pos_norm > 0 else alpha
            
            inputs_embeds_pushed = inputs_embeds_clean.clone()
            inputs_embeds_pushed[0, obj_pos, :] += (beta * push_dir_t).to(model.dtype)
            
            with torch.no_grad():
                out_pushed = model(inputs_embeds=inputs_embeds_pushed, position_ids=position_ids)
            logits_pushed = out_pushed.logits[0, -1].float().cpu().numpy()
            
            # 检查NaN
            if np.any(np.isnan(logits_pushed)):
                print(f"  alpha={alpha}: NaN in logits! Skipping.")
                alpha_results[alpha] = {"error": "NaN in logits"}
                continue
            
            # 属性logit变化
            src_pushed = get_logit_for_tokens(logits_pushed, tokenizer, test["src_props"])
            tgt_pushed = get_logit_for_tokens(logits_pushed, tokenizer, test["tgt_props"])
            neutral_pushed = get_logit_for_tokens(logits_pushed, tokenizer, test["neutral_props"])
            
            # 类别词变化
            cat_pushed = {}
            for cat_name in [src_cat, tgt_cat]:
                for w in CATEGORY_WORDS.get(cat_name, [cat_name]):
                    ids = tokenizer.encode(w, add_special_tokens=False)
                    if ids and ids[0] < len(logits_pushed):
                        cat_pushed[f"{cat_name}:{w}"] = float(logits_pushed[ids[0]])
            
            # 计算delta
            src_deltas = {k: round(src_pushed.get(k, 0) - src_baseline.get(k, 0), 4) 
                         for k in test["src_props"]}
            tgt_deltas = {k: round(tgt_pushed.get(k, 0) - tgt_baseline.get(k, 0), 4) 
                         for k in test["tgt_props"]}
            neutral_deltas = {k: round(neutral_pushed.get(k, 0) - neutral_baseline.get(k, 0), 4) 
                            for k in test["neutral_props"]}
            cat_deltas = {}
            for k in cat_baseline:
                cat_deltas[k] = round(cat_pushed.get(k, 0) - cat_baseline.get(k, 0), 4)
            
            # 统计摘要
            src_mean_delta = np.mean(list(src_deltas.values())) if src_deltas else 0
            tgt_mean_delta = np.mean(list(tgt_deltas.values())) if tgt_deltas else 0
            neutral_mean_delta = np.mean(list(neutral_deltas.values())) if neutral_deltas else 0
            
            alpha_results[alpha] = {
                "src_deltas": src_deltas,
                "tgt_deltas": tgt_deltas,
                "neutral_deltas": neutral_deltas,
                "cat_deltas": cat_deltas,
                "src_mean": round(src_mean_delta, 4),
                "tgt_mean": round(tgt_mean_delta, 4),
                "neutral_mean": round(neutral_mean_delta, 4),
                "mediation_score": round(tgt_mean_delta - src_mean_delta, 4),
            }
            
            print(f"  alpha={alpha}: src_mean={src_mean_delta:.4f}, "
                  f"tgt_mean={tgt_mean_delta:.4f}, "
                  f"mediation={tgt_mean_delta-src_mean_delta:.4f}")
        
        results["per_test"][f"{obj}_{src_cat}to{tgt_cat}"] = {
            "obj": obj,
            "src_cat": src_cat,
            "tgt_cat": tgt_cat,
            "prompt": prompt,
            "baseline_src": src_baseline,
            "baseline_tgt": tgt_baseline,
            "baseline_cat": cat_baseline,
            "push_results": alpha_results,
        }
        
        torch.cuda.empty_cache()
    
    # 保存结果
    os.makedirs("results/phase437_category_property_mediation", exist_ok=True)
    out_path = f"results/phase437_category_property_mediation/{model_name}_phase437_r{round_num}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[4] Results saved to {out_path}")
    
    # 释放模型
    release_model(model)
    
    t_total = time.time() - t_start
    print(f"[5] Total time: {t_total/60:.1f}min")
    
    return results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    run_experiment(model_name, round_num)
