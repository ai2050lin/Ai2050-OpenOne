"""
Phase 434: 注意力头因果消融实验
================================
目标: 验证Phase 431的候选routing heads是否真正搬运类别信息

方法:
1. 正常前向传播: 计算自然运输方向 delta_last = h_last(perturbed) - h_last(clean)
2. 消融候选头: 将该头输出置零
3. 消融后再次计算 delta_last_ablated
4. 因果分数 = 1 - ||delta_last_ablated|| / ||delta_last_original||
5. 同时计算头输出在自然运输方向上的投影

关键指标:
- HeadCausalScore: 消融后delta下降比例
- HeadProjectionScore: cos(head_output_delta, d_natural)
- CategoryLogitImpact: 消融后类别logit变化

用法:
  python tests/glm5/phase434_head_causal_ablation.py qwen3 1
  python tests/glm5/phase434_head_causal_ablation.py glm4 1
  python tests/glm5/phase434_head_causal_ablation.py deepseek7b 1
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


# ===== 实验配置 =====
OBJECTS = {
    "apple":  {"category": "fruit",  "opposing": "animal"},
    "dog":    {"category": "animal", "opposing": "fruit"},
    "knife":  {"category": "tool",   "opposing": "vehicle"},
    "car":    {"category": "vehicle","opposing": "tool"},
}

# Phase 431发现的候选头 (按模型)
CANDIDATE_HEADS = {
    "qwen3": [
        (0, 16), (3, 16), (6, 16), (14, 12), (20, 8),
    ],
    "glm4": [
        (1, 17), (2, 17), (3, 17), (4, 17), (6, 17),
    ],
    "deepseek7b": [
        (5, 12), (10, 12), (15, 12), (20, 12), (27, 12),
    ],
}

# 控制头 (低attn权重的头，用于对照)
CONTROL_HEADS = {
    "qwen3": [
        (0, 0), (3, 0), (6, 0),
    ],
    "glm4": [
        (1, 0), (2, 0), (3, 0),
    ],
    "deepseek7b": [
        (5, 0), (10, 0), (15, 0),
    ],
}

CATEGORY_WORDS = {
    "fruit":   ["fruit", "berry", "produce"],
    "animal":  ["animal", "creature", "beast"],
    "tool":    ["tool", "instrument", "utensil"],
    "vehicle": ["vehicle", "transport", "automobile"],
}


def get_category_direction(model, tokenizer, W_E, category, opposing):
    """计算类别入口方向: W_E(category_center) - W_E(opposing_center)"""
    cat_words = CATEGORY_WORDS.get(category, [category])
    opp_words = CATEGORY_WORDS.get(opposing, [opposing])
    
    cat_vecs = []
    for w in cat_words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if ids:
            cat_vecs.append(W_E[ids[0]])
    
    opp_vecs = []
    for w in opp_words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if ids:
            opp_vecs.append(W_E[ids[0]])
    
    if not cat_vecs or not opp_vecs:
        return None
    
    cat_center = np.mean(cat_vecs, axis=0)
    opp_center = np.mean(opp_vecs, axis=0)
    direction = cat_center - opp_center
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm
    return direction


def compute_natural_transport(model, tokenizer, device, prompt, direction,
                              alpha=1.0, obj_pos=None, last_pos=None):
    """
    计算自然运输方向: delta_l = h_l(perturbed) - h_l(clean)
    返回每层每个位置的delta
    """
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    
    if obj_pos is None:
        obj_pos = seq_len - 3  # 通常对象在倒数第3位置
    if last_pos is None:
        last_pos = seq_len - 1
    
    embed_layer = model.get_input_embeddings()
    inputs_embeds_clean = embed_layer(input_ids).detach().clone().to(model.dtype)
    
    # 在obj_pos位置注入方向
    direction_t = torch.tensor(direction, dtype=inputs_embeds_clean.dtype, device=device)
    inputs_embeds_perturbed = inputs_embeds_clean.clone()
    # 计算注入强度：匹配该位置embedding的范数
    pos_norm = inputs_embeds_clean[0, obj_pos].float().norm().item()
    beta = alpha * pos_norm if pos_norm > 0 else alpha
    inputs_embeds_perturbed[0, obj_pos, :] += (beta * direction_t).to(model.dtype)
    
    # 收集各层输出
    layers = get_layers(model)
    n_layers = len(layers)
    
    # Hook收集
    def run_with_hooks(inputs_embeds):
        captured = {}
        head_outputs = {}  # 收集attention head输出
        
        def make_layer_hook(li):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    captured[f"L{li}"] = output[0].detach()
                else:
                    captured[f"L{li}"] = output.detach()
            return hook
        
        def make_attn_hook(li):
            def hook(module, input, output):
                # output = (hidden_states, attn_weights, past_key_value)
                if isinstance(output, tuple) and len(output) >= 2:
                    captured[f"attn_L{li}"] = output[0].detach()  # attention output
                    if output[1] is not None:
                        captured[f"attn_w_L{li}"] = output[1].detach()  # attn weights
            return hook
        
        hooks = []
        for li in range(n_layers):
            layer = layers[li]
            hooks.append(layer.register_forward_hook(make_layer_hook(li)))
            if hasattr(layer, 'self_attn'):
                hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(li)))
        
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        with torch.no_grad():
            try:
                _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids)
            except Exception as e:
                print(f"  Forward error: {e}")
        
        for h in hooks:
            h.remove()
        
        return captured
    
    # Clean forward
    clean_cap = run_with_hooks(inputs_embeds_clean)
    # Perturbed forward
    perturbed_cap = run_with_hooks(inputs_embeds_perturbed)
    
    # 计算delta
    deltas = {}
    for key in clean_cap:
        if key.startswith("L"):
            li = int(key[1:])
            if key in perturbed_cap:
                delta = perturbed_cap[key].float().cpu().numpy() - clean_cap[key].float().cpu().numpy()
                deltas[key] = delta  # [1, seq_len, d_model]
    
    # 提取last_pos的delta
    delta_last = {}
    for key in deltas:
        li = int(key[1:])
        delta_last[li] = deltas[key][0, last_pos]  # [d_model]
    
    # 提取attn output delta
    attn_delta = {}
    for key in clean_cap:
        if key.startswith("attn_L") and not key.startswith("attn_w_"):
            li = int(key.split("L")[1])
            if key in perturbed_cap:
                a_delta = perturbed_cap[key].float().cpu().numpy() - clean_cap[key].float().cpu().numpy()
                attn_delta[li] = a_delta[0, last_pos]  # [d_model]
    
    return delta_last, attn_delta, deltas


def ablate_head_and_compute(model, tokenizer, device, prompt, direction,
                             alpha, obj_pos, last_pos, ablate_layer, ablate_head,
                             n_heads, head_dim):
    """
    消融特定注意力头后计算自然运输方向
    
    消融方法: 在该头的output中，将对应head的贡献置零
    """
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    
    embed_layer = model.get_input_embeddings()
    inputs_embeds_clean = embed_layer(input_ids).detach().clone().to(model.dtype)
    
    direction_t = torch.tensor(direction, dtype=inputs_embeds_clean.dtype, device=device)
    inputs_embeds_perturbed = inputs_embeds_clean.clone()
    pos_norm = inputs_embeds_clean[0, obj_pos].float().norm().item()
    beta = alpha * pos_norm if pos_norm > 0 else alpha
    inputs_embeds_perturbed[0, obj_pos, :] += (beta * direction_t).to(model.dtype)
    
    layers = get_layers(model)
    
    # Hook: 消融特定head + 收集层输出
    def run_with_ablation(inputs_embeds, do_ablate=False):
        captured = {}
        
        def make_layer_hook(li):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    captured[f"L{li}"] = output[0].detach()
                else:
                    captured[f"L{li}"] = output.detach()
            return hook
        
        def make_attn_ablate_hook(li):
            def hook(module, input, output):
                if not do_ablate or li != ablate_layer:
                    return
                # output[0] = attention output [batch, seq, d_model]
                if isinstance(output, tuple) and len(output) >= 1:
                    attn_out = output[0]
                    # 重塑为 [batch, seq, n_heads, head_dim]
                    batch, seq, d = attn_out.shape
                    attn_out_reshaped = attn_out.view(batch, seq, n_heads, head_dim)
                    # 置零目标head
                    attn_out_reshaped[:, :, ablate_head, :] = 0
                    # 重塑回来
                    attn_out_modified = attn_out_reshaped.view(batch, seq, d)
                    # 返回修改后的output
                    new_output = (attn_out_modified,) + output[1:]
                    return new_output
            return hook
        
        hooks = []
        for li in range(len(layers)):
            hooks.append(layers[li].register_forward_hook(make_layer_hook(li)))
            if hasattr(layers[li], 'self_attn'):
                hooks.append(layers[li].self_attn.register_forward_hook(make_attn_ablate_hook(li)))
        
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        with torch.no_grad():
            try:
                _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids)
            except Exception as e:
                print(f"  Forward error (ablate={do_ablate}): {e}")
        
        for h in hooks:
            h.remove()
        
        return captured
    
    # Clean + ablated
    clean_cap = run_with_ablation(inputs_embeds_clean, do_ablate=True)
    # Perturbed + ablated
    perturbed_cap = run_with_ablation(inputs_embeds_perturbed, do_ablate=True)
    
    # 计算delta
    delta_last_ablated = {}
    for key in clean_cap:
        if key.startswith("L"):
            li = int(key[1:])
            if key in perturbed_cap:
                delta = perturbed_cap[key].float().cpu().numpy() - clean_cap[key].float().cpu().numpy()
                delta_last_ablated[li] = delta[0, last_pos]
    
    return delta_last_ablated


def get_logits_for_categories(model, tokenizer, device, prompt, direction,
                               alpha, obj_pos, categories_dict):
    """获取clean和perturbed状态下类别词的logits"""
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    last_pos = seq_len - 1
    
    embed_layer = model.get_input_embeddings()
    inputs_embeds_clean = embed_layer(input_ids).detach().clone().to(model.dtype)
    
    direction_t = torch.tensor(direction, dtype=inputs_embeds_clean.dtype, device=device)
    inputs_embeds_perturbed = inputs_embeds_clean.clone()
    pos_norm = inputs_embeds_clean[0, obj_pos].float().norm().item()
    beta = alpha * pos_norm if pos_norm > 0 else alpha
    inputs_embeds_perturbed[0, obj_pos, :] += (beta * direction_t).to(model.dtype)
    
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    
    # Clean logits
    with torch.no_grad():
        out_clean = model(inputs_embeds=inputs_embeds_clean, position_ids=position_ids)
        logits_clean = out_clean.logits[0, last_pos].float().cpu().numpy()
    
    # Perturbed logits
    with torch.no_grad():
        out_pert = model(inputs_embeds=inputs_embeds_perturbed, position_ids=position_ids)
        logits_pert = out_pert.logits[0, last_pos].float().cpu().numpy()
    
    # 提取类别词logits
    cat_logits = {}
    for cat, words in categories_dict.items():
        cat_ids = []
        for w in words:
            ids = tokenizer.encode(w, add_special_tokens=False)
            if ids:
                cat_ids.append(ids[0])
        
        clean_sum = sum(logits_clean[i] for i in cat_ids if i < len(logits_clean))
        pert_sum = sum(logits_pert[i] for i in cat_ids if i < len(logits_pert))
        cat_logits[cat] = {
            "clean": float(clean_sum),
            "perturbed": float(pert_sum),
            "delta": float(pert_sum - clean_sum),
        }
    
    return cat_logits


def run_experiment(model_name: str, round_num: int = 1):
    """运行Phase 434实验"""
    print(f"\n{'='*60}")
    print(f"Phase 434: 注意力头因果消融 - {model_name} R{round_num}")
    print(f"{'='*60}")
    t_start = time.time()
    
    # 加载模型
    print(f"[1] Loading {model_name}...")
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    
    print(f"  class={info.model_class}, n_layers={info.n_layers}, "
          f"d_model={info.d_model}, n_heads unknown")
    
    # 获取head维度 - 从config获取n_heads
    n_heads = getattr(model.config, 'num_attention_heads', 
                      getattr(model.config, 'num_heads', None))
    if n_heads is None:
        # 从W_q推断
        layer0 = layers[0]
        sa = layer0.self_attn
        W_q_shape = sa.q_proj.weight.shape
        n_heads = W_q_shape[0] // info.d_model if W_q_shape[0] >= info.d_model else 32
    d_model = info.d_model
    head_dim = d_model // n_heads
    print(f"  n_heads={n_heads}, head_dim={head_dim}")
    
    # 获取W_E和W_U
    print(f"[2] Loading W_E and W_U...")
    W_E = model.get_input_embeddings().weight.detach().cpu().float().numpy()
    W_U = get_W_U(model, model_name)
    print(f"  W_E: {W_E.shape}, W_U: {W_U.shape}")
    
    # 获取候选头
    candidate_heads = CANDIDATE_HEADS.get(model_name, [])
    control_heads = CONTROL_HEADS.get(model_name, [])
    
    # 过滤掉超出范围的层
    candidate_heads = [(l, h) for l, h in candidate_heads if l < info.n_layers and h < n_heads]
    control_heads = [(l, h) for l, h in control_heads if l < info.n_layers and h < n_heads]
    
    print(f"  Candidate heads: {candidate_heads}")
    print(f"  Control heads: {control_heads}")
    
    results = {
        "model": model_name,
        "round": round_num,
        "n_layers": info.n_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "per_object": {},
    }
    
    for obj_name, obj_info in OBJECTS.items():
        print(f"\n[3] === Object: {obj_name} ({obj_info['category']}) ===")
        
        prompt = f"An {obj_name} is a kind of"
        if obj_name[0] not in "aeiouAEIOU":
            prompt = f"A {obj_name} is a kind of"
        
        # 确定位置
        toks = tokenizer(prompt, return_tensors="pt").to(device)
        seq_len = toks.input_ids.shape[1]
        obj_pos = 1  # 通常对象在第1位置 (A/An __ is...)
        last_pos = seq_len - 1
        
        print(f"  Prompt: '{prompt}', seq_len={seq_len}, obj_pos={obj_pos}, last_pos={last_pos}")
        
        # 计算类别方向
        direction = get_category_direction(model, tokenizer, W_E, 
                                            obj_info['category'], obj_info['opposing'])
        if direction is None:
            print(f"  SKIP: cannot compute category direction")
            continue
        
        # Step 1: 计算原始自然运输方向
        print(f"  [3a] Computing natural transport (original)...")
        t0 = time.time()
        delta_last_orig, attn_delta_orig, _ = compute_natural_transport(
            model, tokenizer, device, prompt, direction,
            alpha=1.0, obj_pos=obj_pos, last_pos=last_pos
        )
        t1 = time.time()
        print(f"    Done in {t1-t0:.1f}s, captured {len(delta_last_orig)} layers")
        
        # 原始delta范数 (last_pos, 各层)
        orig_norms = {li: float(np.linalg.norm(delta_last_orig[li])) 
                      for li in delta_last_orig}
        print(f"    Delta norms (selected): " + 
              ", ".join(f"L{li}={orig_norms.get(li,0):.4f}" 
                       for li in sorted(delta_last_orig.keys())[:5]) + "...")
        
        # Step 2: 获取baseline logits
        print(f"  [3b] Getting category logits...")
        cat_logits = get_logits_for_categories(
            model, tokenizer, device, prompt, direction,
            alpha=1.0, obj_pos=obj_pos, categories_dict=CATEGORY_WORDS
        )
        print(f"    Category logits: " + 
              ", ".join(f"{k}={v['delta']:.3f}" for k, v in cat_logits.items()))
        
        # Step 3: 对候选头做消融
        head_results = []
        all_heads = candidate_heads + control_heads
        
        for hi, (ablate_layer, ablate_head) in enumerate(all_heads):
            is_control = (ablate_layer, ablate_head) in control_heads
            head_type = "control" if is_control else "candidate"
            
            print(f"  [3c-{hi}] Ablating L{ablate_layer}/H{ablate_head} ({head_type})...")
            t0 = time.time()
            
            try:
                delta_last_ablated = ablate_head_and_compute(
                    model, tokenizer, device, prompt, direction,
                    alpha=1.0, obj_pos=obj_pos, last_pos=last_pos,
                    ablate_layer=ablate_layer, ablate_head=ablate_head,
                    n_heads=n_heads, head_dim=head_dim
                )
                t1 = time.time()
                
                # 计算因果分数
                ablated_norms = {li: float(np.linalg.norm(delta_last_ablated[li])) 
                                for li in delta_last_ablated}
                
                # 选择几个关键层计算因果分数
                key_layers = sorted(delta_last_orig.keys())
                causal_scores = {}
                for li in key_layers:
                    orig_n = orig_norms.get(li, 0)
                    ablated_n = ablated_norms.get(li, 0)
                    if orig_n > 1e-10:
                        causal_scores[li] = 1.0 - ablated_n / orig_n
                    else:
                        causal_scores[li] = 0.0
                
                # 头输出投影到自然运输方向
                projection_scores = {}
                for li in key_layers:
                    if li in attn_delta_orig and li in delta_last_orig:
                        attn_d = attn_delta_orig[li]
                        nat_d = delta_last_orig[li]
                        attn_norm = np.linalg.norm(attn_d)
                        nat_norm = np.linalg.norm(nat_d)
                        if attn_norm > 1e-10 and nat_norm > 1e-10:
                            projection_scores[li] = float(np.dot(attn_d, nat_d) / (attn_norm * nat_norm))
                        else:
                            projection_scores[li] = 0.0
                
                hr = {
                    "layer": ablate_layer,
                    "head": ablate_head,
                    "type": head_type,
                    "time_s": round(t1 - t0, 1),
                    "causal_scores": {str(k): round(v, 4) for k, v in causal_scores.items()},
                    "projection_scores": {str(k): round(v, 4) for k, v in projection_scores.items()},
                    "orig_delta_norm_Llast": round(orig_norms.get(max(key_layers), 0), 6),
                    "ablated_delta_norm_Llast": round(ablated_norms.get(max(key_layers), 0), 6),
                }
                
                # 摘要
                last_li = max(key_layers)
                cs_last = causal_scores.get(last_li, 0)
                ps_mid = 0
                mid_layers = [li for li in key_layers if info.n_layers//4 <= li <= 3*info.n_layers//4]
                if mid_layers and any(li in projection_scores for li in mid_layers):
                    ps_mid = np.mean([projection_scores[li] for li in mid_layers if li in projection_scores])
                
                print(f"    CausalScore(L{last_li})={cs_last:.4f}, "
                      f"ProjectionScore(mid)={ps_mid:.4f}, time={t1-t0:.1f}s")
                
                head_results.append(hr)
                
            except Exception as e:
                print(f"    FAILED: {e}")
                head_results.append({
                    "layer": ablate_layer, "head": ablate_head, "type": head_type,
                    "error": str(e)
                })
            
            # 定期清理GPU
            if hi % 3 == 0:
                torch.cuda.empty_cache()
        
        results["per_object"][obj_name] = {
            "category": obj_info["category"],
            "opposing": obj_info["opposing"],
            "prompt": prompt,
            "seq_len": seq_len,
            "obj_pos": obj_pos,
            "last_pos": last_pos,
            "category_logits": cat_logits,
            "orig_delta_norms": {str(k): round(v, 6) for k, v in orig_norms.items()},
            "head_results": head_results,
        }
    
    # 保存结果
    os.makedirs("results/phase434_head_causal_ablation", exist_ok=True)
    out_path = f"results/phase434_head_causal_ablation/{model_name}_phase434_r{round_num}.json"
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
