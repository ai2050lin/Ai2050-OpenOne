"""
Phase 438: 运输算子跨对象迁移实验
===================================
目标: 测试类别运输方向是否可以在同类对象间迁移

方法:
1. 计算apple的fruit运输方向(注入fruit方向后的delta)
2. 把这个方向注入orange的对应层
3. 看orange是否也被推向fruit类别
4. 同时测试: fruit方向注入knife(跨类别)是否产生错误偏移

这是验证"运输方向是类别级结构"还是"对象特定"的关键实验。

用法:
  python tests/glm5/phase438_cross_object_transport.py qwen3 1
  python tests/glm5/phase438_cross_object_transport.py glm4 1
  python tests/glm5/phase438_cross_object_transport.py deepseek7b 1
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


# ===== 对象集 =====
# 同类对象对 (source -> target)
SAME_CATEGORY_PAIRS = [
    # fruit
    {"src": "apple", "src_cat": "fruit", "tgt": "orange", "tgt_cat": "fruit", "opp_cat": "animal"},
    {"src": "apple", "src_cat": "fruit", "tgt": "lemon",  "tgt_cat": "fruit", "opp_cat": "animal"},
    # animal
    {"src": "dog",   "src_cat": "animal", "tgt": "cat",   "tgt_cat": "animal", "opp_cat": "fruit"},
    {"src": "dog",   "src_cat": "animal", "tgt": "horse", "tgt_cat": "animal", "opp_cat": "fruit"},
    # tool
    {"src": "knife", "src_cat": "tool",   "tgt": "hammer","tgt_cat": "tool",   "opp_cat": "vehicle"},
    {"src": "knife", "src_cat": "tool",   "tgt": "spoon", "tgt_cat": "tool",   "opp_cat": "vehicle"},
    # vehicle
    {"src": "car",   "src_cat": "vehicle","tgt": "train", "tgt_cat": "vehicle","opp_cat": "tool"},
    {"src": "car",   "src_cat": "vehicle","tgt": "bus",   "tgt_cat": "vehicle","opp_cat": "tool"},
]

# 跨类别迁移测试
CROSS_CATEGORY_PAIRS = [
    {"src": "apple", "src_cat": "fruit",  "tgt": "knife", "tgt_cat": "tool",    "opp_cat": "animal"},
    {"src": "dog",   "src_cat": "animal", "tgt": "car",   "tgt_cat": "vehicle", "opp_cat": "fruit"},
    {"src": "knife", "src_cat": "tool",   "tgt": "apple", "tgt_cat": "fruit",   "opp_cat": "vehicle"},
]

CATEGORY_WORDS = {
    "fruit":   ["fruit", "berry", "produce"],
    "animal":  ["animal", "creature", "beast"],
    "tool":    ["tool", "instrument", "utensil"],
    "vehicle": ["vehicle", "transport", "automobile"],
}


def get_category_direction(W_E, tokenizer, src_cat, opp_cat):
    """计算类别方向: src_center - opp_center"""
    src_words = CATEGORY_WORDS.get(src_cat, [src_cat])
    opp_words = CATEGORY_WORDS.get(opp_cat, [opp_cat])
    
    src_vecs = [W_E[tokenizer.encode(w, add_special_tokens=False)[0]] 
                for w in src_words if tokenizer.encode(w, add_special_tokens=False)]
    opp_vecs = [W_E[tokenizer.encode(w, add_special_tokens=False)[0]] 
                for w in opp_words if tokenizer.encode(w, add_special_tokens=False)]
    
    if not src_vecs or not opp_vecs:
        return None
    
    direction = np.mean(src_vecs, axis=0) - np.mean(opp_vecs, axis=0)
    norm = np.linalg.norm(direction)
    return direction / norm if norm > 0 else direction


def compute_transport_direction(model, tokenizer, device, obj_name, direction,
                                 alpha=1.0, obj_pos=1, last_pos=None):
    """计算自然运输方向: delta_l = h_l(perturbed) - h_l(clean)"""
    prompt = f"An {obj_name} is a kind of" if obj_name[0] in "aeiouAEIOU" else f"A {obj_name} is a kind of"
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    if last_pos is None:
        last_pos = seq_len - 1
    
    embed_layer = model.get_input_embeddings()
    inputs_embeds_clean = embed_layer(input_ids).detach().clone().to(model.dtype)
    
    direction_t = torch.tensor(direction, dtype=inputs_embeds_clean.dtype, device=device)
    inputs_embeds_perturbed = inputs_embeds_clean.clone()
    pos_norm = inputs_embeds_clean[0, obj_pos].float().norm().item()
    beta = alpha * pos_norm if pos_norm > 0 else alpha
    inputs_embeds_perturbed[0, obj_pos, :] += (beta * direction_t).to(model.dtype)
    
    # 收集各层输出
    layers = get_layers(model)
    captured_clean = {}
    captured_pert = {}
    
    def make_hook(cap_dict, prefix):
        def hook(module, input, output):
            if isinstance(output, tuple):
                cap_dict[prefix] = output[0].detach().float().cpu()
            else:
                cap_dict[prefix] = output.detach().float().cpu()
        return hook
    
    # Clean forward
    hooks = [layers[li].register_forward_hook(make_hook(captured_clean, f"L{li}")) 
             for li in range(len(layers))]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    with torch.no_grad():
        model(inputs_embeds=inputs_embeds_clean, position_ids=position_ids)
    for h in hooks:
        h.remove()
    
    # Perturbed forward
    hooks = [layers[li].register_forward_hook(make_hook(captured_pert, f"L{li}")) 
             for li in range(len(layers))]
    with torch.no_grad():
        model(inputs_embeds=inputs_embeds_perturbed, position_ids=position_ids)
    for h in hooks:
        h.remove()
    
    # 计算delta (last_pos)
    delta_last = {}
    for key in captured_clean:
        if key in captured_pert:
            li = int(key[1:])
            delta = captured_pert[key][0, last_pos].numpy() - captured_clean[key][0, last_pos].numpy()
            delta_last[li] = delta
    
    return delta_last, prompt


def get_category_logits(model, tokenizer, device, prompt, categories_dict):
    """获取类别词logits"""
    toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = toks["input_ids"].to(device)
    attention_mask = toks["attention_mask"].to(device)
    
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[0, -1].float().cpu().numpy()
    
    result = {}
    for cat, words in categories_dict.items():
        cat_logits = {}
        for w in words:
            ids = tokenizer.encode(w, add_special_tokens=False)
            if ids and ids[0] < len(logits):
                cat_logits[w] = float(logits[ids[0]])
        if cat_logits:
            result[cat] = cat_logits
    return result


def inject_direction_at_layer(model, tokenizer, device, prompt, direction,
                               inject_layer, beta=1.0):
    """在指定层注入方向到last token"""
    toks = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = toks["input_ids"].to(device)
    attention_mask = toks["attention_mask"].to(device)
    seq_len = input_ids.shape[1]
    
    direction_t = torch.tensor(direction, dtype=torch.bfloat16, device=device)
    injected = [False]
    
    def hook(module, input, output):
        if injected[0]:
            return
        if isinstance(output, tuple):
            h = output[0]
            h[:, -1, :] += (beta * direction_t).to(h.dtype)
            injected[0] = True
            return (h,) + output[1:]
        else:
            output[:, -1, :] += (beta * direction_t).to(output.dtype)
            injected[0] = True
            return output
    
    layers = get_layers(model)
    h = layers[inject_layer].register_forward_hook(hook)
    
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
    
    h.remove()
    return out.logits[0, -1].float().cpu().numpy()


def run_experiment(model_name: str, round_num: int = 1):
    """运行Phase 438实验"""
    print(f"\n{'='*60}")
    print(f"Phase 438: 运输算子跨对象迁移 - {model_name} R{round_num}")
    print(f"{'='*60}")
    t_start = time.time()
    
    # 用bf16+auto加载所有模型
    from transformers import AutoModelForCausalLM, AutoTokenizer
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
    n_layers = info.n_layers
    print(f"  class={type(model).__name__}, n_layers={n_layers}")
    
    W_E = model.get_input_embeddings().weight.detach().cpu().float().numpy()
    W_U = get_W_U(model, model_name)
    
    # 采样层
    sample_layers = list(range(0, n_layers, max(1, n_layers // 6))) + [n_layers - 1]
    sample_layers = sorted(set(sample_layers))
    
    results = {
        "model": model_name,
        "round": round_num,
        "n_layers": n_layers,
        "sample_layers": sample_layers,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "same_category": {},
        "cross_category": {},
    }
    
    # ===== 同类别迁移 =====
    print(f"\n[2] Same-category transport transfer...")
    for pi, pair in enumerate(SAME_CATEGORY_PAIRS):
        src = pair["src"]
        tgt = pair["tgt"]
        src_cat = pair["src_cat"]
        opp_cat = pair["opp_cat"]
        
        print(f"\n  === {src}({src_cat}) -> {tgt}({src_cat}) transport transfer ===")
        
        # 1. 计算src对象的运输方向
        cat_dir = get_category_direction(W_E, tokenizer, src_cat, opp_cat)
        if cat_dir is None:
            continue
        
        src_delta, src_prompt = compute_transport_direction(
            model, tokenizer, device, src, cat_dir, alpha=1.0)
        
        # 2. 获取tgt对象的baseline logits
        tgt_prompt = f"An {tgt} is a kind of" if tgt[0] in "aeiouAEIOU" else f"A {tgt} is a kind of"
        baseline_logits = get_category_logits(
            model, tokenizer, device, tgt_prompt, CATEGORY_WORDS)
        
        # 3. 将src的运输方向注入到tgt的对应层
        transfer_results = {}
        for li in sample_layers:
            if li not in src_delta:
                continue
            
            delta = src_delta[li]
            delta_norm = np.linalg.norm(delta)
            if delta_norm < 1e-10:
                continue
            
            # 用delta_norm作为注入强度
            for beta in [1.0, 2.0]:
                try:
                    logits_inj = inject_direction_at_layer(
                        model, tokenizer, device, tgt_prompt,
                        delta, inject_layer=li, beta=beta)
                    
                    if np.any(np.isnan(logits_inj)):
                        continue
                    
                    # 计算类别logit变化
                    cat_deltas = {}
                    for cat, words in CATEGORY_WORDS.items():
                        deltas = []
                        for w in words:
                            ids = tokenizer.encode(w, add_special_tokens=False)
                            if ids and ids[0] < len(logits_inj):
                                bl = baseline_logits.get(cat, {}).get(w, 0)
                                cat_deltas[f"{cat}:{w}"] = round(float(logits_inj[ids[0]] - bl), 4)
                    
                    # 计算src_cat vs opp_cat的logit差变化
                    src_cat_delta = np.mean([cat_deltas.get(f"{src_cat}:{w}", 0) 
                                           for w in CATEGORY_WORDS.get(src_cat, [])])
                    opp_cat_delta = np.mean([cat_deltas.get(f"{opp_cat}:{w}", 0) 
                                           for w in CATEGORY_WORDS.get(opp_cat, [])])
                    
                    transfer_results[f"L{li}_b{beta}"] = {
                        "cat_deltas": cat_deltas,
                        "src_cat_delta": round(float(src_cat_delta), 4),
                        "opp_cat_delta": round(float(opp_cat_delta), 4),
                        "transfer_score": round(float(src_cat_delta - opp_cat_delta), 4),
                    }
                    
                except Exception as e:
                    transfer_results[f"L{li}_b{beta}"] = {"error": str(e)}
        
        # 打印摘要
        key_results = {k: v for k, v in transfer_results.items() 
                      if "b2.0" in k and "error" not in v}
        if key_results:
            best_layer = max(key_results.keys(), key=lambda k: key_results[k].get("transfer_score", 0))
            best_score = key_results[best_layer]["transfer_score"]
            print(f"    Best transfer: {best_layer} score={best_score:.4f}")
        
        results["same_category"][f"{src}->{tgt}"] = {
            "src": src, "tgt": tgt, "src_cat": src_cat,
            "src_delta_norms": {str(li): round(float(np.linalg.norm(src_delta[li])), 4) 
                                 if li in src_delta else 0.0 for li in sample_layers},
            "transfer": transfer_results,
        }
        
        torch.cuda.empty_cache()
    
    # ===== 跨类别迁移 =====
    print(f"\n[3] Cross-category transport transfer...")
    for pi, pair in enumerate(CROSS_CATEGORY_PAIRS):
        src = pair["src"]
        tgt = pair["tgt"]
        src_cat = pair["src_cat"]
        tgt_cat = pair["tgt_cat"]
        opp_cat = pair["opp_cat"]
        
        print(f"\n  === {src}({src_cat}) -> {tgt}({tgt_cat}) cross-category ===")
        
        # 1. 计算src的src_cat方向运输
        cat_dir = get_category_direction(W_E, tokenizer, src_cat, opp_cat)
        if cat_dir is None:
            continue
        
        src_delta, src_prompt = compute_transport_direction(
            model, tokenizer, device, src, cat_dir, alpha=1.0)
        
        # 2. 获取tgt的baseline
        tgt_prompt = f"An {tgt} is a kind of" if tgt[0] in "aeiouAEIOU" else f"A {tgt} is a kind of"
        baseline_logits = get_category_logits(
            model, tokenizer, device, tgt_prompt, CATEGORY_WORDS)
        
        # 3. 注入到tgt
        cross_results = {}
        for li in sample_layers:
            if li not in src_delta:
                continue
            delta = src_delta[li]
            delta_norm = np.linalg.norm(delta)
            if delta_norm < 1e-10:
                continue
            
            for beta in [2.0]:
                try:
                    logits_inj = inject_direction_at_layer(
                        model, tokenizer, device, tgt_prompt,
                        delta, inject_layer=li, beta=beta)
                    
                    if np.any(np.isnan(logits_inj)):
                        continue
                    
                    cat_deltas = {}
                    for cat, words in CATEGORY_WORDS.items():
                        for w in words:
                            ids = tokenizer.encode(w, add_special_tokens=False)
                            if ids and ids[0] < len(logits_inj):
                                bl = baseline_logits.get(cat, {}).get(w, 0)
                                cat_deltas[f"{cat}:{w}"] = round(float(logits_inj[ids[0]] - bl), 4)
                    
                    src_cat_delta = np.mean([cat_deltas.get(f"{src_cat}:{w}", 0) 
                                           for w in CATEGORY_WORDS.get(src_cat, [])])
                    tgt_cat_delta = np.mean([cat_deltas.get(f"{tgt_cat}:{w}", 0) 
                                           for w in CATEGORY_WORDS.get(tgt_cat, [])])
                    
                    cross_results[f"L{li}_b{beta}"] = {
                        "cat_deltas": cat_deltas,
                        "src_cat_delta": round(float(src_cat_delta), 4),
                        "tgt_cat_delta": round(float(tgt_cat_delta), 4),
                    }
                    
                except Exception as e:
                    cross_results[f"L{li}_b{beta}"] = {"error": str(e)}
        
        results["cross_category"][f"{src}({src_cat})->{tgt}({tgt_cat})"] = {
            "src": src, "tgt": tgt, "src_cat": src_cat, "tgt_cat": tgt_cat,
            "transfer": cross_results,
        }
        
        torch.cuda.empty_cache()
    
    # 保存结果
    os.makedirs("results/phase438_cross_object_transport", exist_ok=True)
    out_path = f"results/phase438_cross_object_transport/{model_name}_phase438_r{round_num}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[4] Results saved to {out_path}")
    
    release_model(model)
    
    t_total = time.time() - t_start
    print(f"[5] Total time: {t_total/60:.1f}min")
    
    return results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    run_experiment(model_name, round_num)
