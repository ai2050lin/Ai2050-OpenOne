"""
Phase 102b: 改进的因果干预 — 基于Δh子空间的干预
================================================

Exp 1的核心发现:
  - 添加v_trans到hidden state几乎无效果 (ΔP ≈ 10^-5)
  - 排列检验: 孤立词翻译方向一致性不显著 (p=0.79)
  - 上下文化alignment虽高(0.96)但无因果效应

Exp 2的关键发现:
  - Δh极度低秩: 90%var只需10维 (d_model=2560)
  - 第1个奇异值占80%方差 → 大部分维度是pass-through
  - L9是翻译计算最独特的层 (Δh跨任务cosine最低)

问题诊断:
  1. 干预方向可能不在实际计算子空间中 → 改用Δh子空间内的方向
  2. 干预强度可能太小 (hidden state范数~5000, α=4太小) → 增大α
  3. 应该干预Δh而非h → 在层l-1输出上添加扰动，让层l的计算自然处理

改进方案:
  Exp 1b: Δh子空间干预
    - 从翻译vs中文的Δh差异中提取方向
    - 只在Δh的前10个主成分方向上干预
    - 用更大的α值 (基于hidden state范数校准)
    - 测量: 输出是否从中文切换到英文

  Exp 1c: Logit镜头分析
    - 不干预，而是直接分析: 每层的hidden state如果直接投影到logits
    - 看翻译方向在logit空间中的表现
    - 这告诉我们: 翻译信号在哪个层开始出现?

Run:
  python tests/glm5_temp/phase102b_improved_intervention.py --model qwen3 --exp 1b
  python tests/glm5_temp/phase102b_improved_intervention.py --model qwen3 --exp 1c
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5_temp'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import gc
import json
import time
from collections import defaultdict

from model_utils import load_model, get_layers, get_model_info, release_model, get_W_U


# 翻译对
TRANSLATION_TRAIN = [
    ("猫", "cat"), ("狗", "dog"), ("书", "book"), ("火", "fire"),
    ("花", "flower"), ("鱼", "fish"), ("树", "tree"), ("鸟", "bird"),
    ("马", "horse"), ("铁", "iron"), ("金", "gold"), ("茶", "tea"),
    ("米", "rice"), ("血", "blood"), ("眼", "eye"), ("手", "hand"),
    ("水", "water"), ("风", "wind"), ("雪", "snow"), ("星", "star"),
    ("月", "moon"), ("日", "sun"), ("山", "mountain"), ("河", "river"),
]

TRANSLATION_TEST = [
    ("龙", "dragon"), ("云", "cloud"), ("雨", "rain"), ("雷", "thunder"),
    ("石", "stone"), ("草", "grass"), ("沙", "sand"), ("冰", "ice"),
    ("光", "light"), ("影", "shadow"), ("梦", "dream"), ("歌", "song"),
]


def intervene_at_layer(model, tokenizer, prompt, layer_idx, direction_tensor, alpha, device):
    """在指定层注入方向向量到last token位置"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    layers = get_layers(model)
    intervened = [False]
    
    def hook_fn(module, input, output):
        if not intervened[0]:
            if isinstance(output, tuple):
                hidden_states = output[0].clone()
                hidden_states[:, -1, :] += alpha * direction_tensor.to(hidden_states.dtype).to(device)
                output = (hidden_states,) + output[1:]
            intervened[0] = True
        return output
    
    handle = layers[layer_idx].register_forward_hook(hook_fn)
    
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :].float().cpu()
    
    handle.remove()
    return logits


def get_token_prob(logits, tokenizer, text):
    tok_ids = tokenizer.encode(text, add_special_tokens=False)
    if not tok_ids:
        return 0.0
    probs = torch.softmax(logits, dim=-1)
    return probs[tok_ids[0]].item()


def get_top_k_tokens(logits, tokenizer, k=20):
    probs = torch.softmax(logits, dim=-1)
    topk = torch.topk(probs, k)
    results = []
    for i in range(k):
        tok_id = topk.indices[i].item()
        prob = topk.values[i].item()
        tok_str = tokenizer.decode([tok_id])
        results.append({"token": tok_str, "token_id": tok_id, "prob": prob})
    return results


# ============================================================
# Exp 1b: Δh子空间干预
# ============================================================
def exp1b_delta_h_subspace_intervention(model_name):
    """
    基于Δh子空间的改进干预
    
    核心思路:
    1. 收集翻译和中文补全的Δh
    2. 提取翻译特有的Δh方向 (Δh_trans - Δh_zh)
    3. 在Δh的主成分方向上干预
    4. 用校准后的α值 (基于hidden state范数)
    """
    print(f"\n{'='*70}")
    print(f"Exp 1b: Δh子空间干预 — {model_name}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  模型: {model_name}, 层数: {n_layers}, d_model: {d_model}")
    
    results = {}
    
    # ---- Step 1: 收集hidden states并计算范数 ----
    print(f"\n  === Step 1: 收集hidden states ===")
    
    all_hiddens = {"zh_continue": {}, "translate": {}}
    
    for zh, en in TRANSLATION_TRAIN:
        zh_prompt = f"{zh}是一种"
        trans_prompt = f'请翻译：{zh} →'
        
        for task_name, prompt in [("zh_continue", zh_prompt), ("translate", trans_prompt)]:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(inputs["input_ids"], output_hidden_states=True)
            
            hiddens = []
            for l in range(n_layers + 1):
                h = outputs.hidden_states[l][0, -1, :].float().cpu().numpy()
                hiddens.append(h)
            all_hiddens[task_name][(zh, en)] = hiddens
    
    # 打印hidden state范数
    print(f"\n  Hidden state范数 (最后token):")
    for l in [0, 3, 6, 9, 12, 18, 24, 30, 35]:
        zh_norms = [np.linalg.norm(all_hiddens["zh_continue"][pair][l]) 
                   for pair in TRANSLATION_TRAIN]
        trans_norms = [np.linalg.norm(all_hiddens["translate"][pair][l]) 
                      for pair in TRANSLATION_TRAIN]
        print(f"    L{l}: zh_norm={np.mean(zh_norms):.1f}, trans_norm={np.mean(trans_norms):.1f}")
    
    # ---- Step 2: 计算Δh并提取翻译特有方向 ----
    print(f"\n  === Step 2: 计算Δh和翻译特有方向 ===")
    
    # 逐层计算Δh
    delta_h = {"zh_continue": {}, "translate": {}}
    for task_name in ["zh_continue", "translate"]:
        for pair, hiddens in all_hiddens[task_name].items():
            deltas = []
            for l in range(n_layers):
                deltas.append(hiddens[l+1] - hiddens[l])
            delta_h[task_name][pair] = deltas
    
    # 翻译特有的Δh方向: Δh_trans - Δh_zh (对每个pair)
    trans_specific_deltas = defaultdict(list)  # layer -> [delta_diff]
    for pair in TRANSLATION_TRAIN:
        for l in range(n_layers):
            diff = delta_h["translate"][pair][l] - delta_h["zh_continue"][pair][l]
            trans_specific_deltas[l].append(diff)
    
    # 对每层的翻译特有Δh做SVD
    print(f"\n  翻译特有Δh的SVD分析:")
    trans_specific_dirs = {}
    for l in [3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33]:
        if l >= n_layers:
            continue
        diffs = np.array(trans_specific_deltas[l])  # [n_pairs, d_model]
        U, S, Vt = np.linalg.svd(diffs, full_matrices=False)
        
        # 前10个主成分
        total_var = np.sum(S**2)
        cumvar = np.cumsum(S**2) / total_var
        dim_90 = int(np.searchsorted(cumvar, 0.9)) + 1
        
        # 平均翻译特有方向
        mean_diff = np.mean(diffs, axis=0)
        mean_norm = np.linalg.norm(mean_diff)
        
        # 方向一致性
        if mean_norm > 1e-6:
            cosines = [np.dot(d, mean_diff) / (np.linalg.norm(d) * mean_norm + 1e-8) 
                      for d in diffs if np.linalg.norm(d) > 1e-6]
            alignment = float(np.mean(cosines)) if cosines else 0.0
        else:
            alignment = 0.0
        
        trans_specific_dirs[l] = {
            "mean_direction": mean_diff / mean_norm if mean_norm > 1e-6 else np.zeros(d_model),
            "mean_norm": float(mean_norm),
            "alignment": alignment,
            "dim_90": dim_90,
            "top10_sv": [float(s) for s in S[:10]],
        }
        
        print(f"    L{l}: mean_norm={mean_norm:.2f}, alignment={alignment:.3f}, "
              f"dim_90={dim_90}, S1={S[0]:.2f}")
    
    # ---- Step 3: Δh子空间干预 ----
    print(f"\n  === Step 3: Δh子空间干预 ===")
    
    # 收集baseline
    print(f"\n  收集baseline...")
    baseline_results = {}
    for zh, en in TRANSLATION_TEST:
        zh_prompt = f"{zh}是一种"
        inputs = tokenizer(zh_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            baseline_logits = model(inputs["input_ids"]).logits[0, -1, :].float().cpu()
        
        en_prob = get_token_prob(baseline_logits, tokenizer, en)
        top20 = get_top_k_tokens(baseline_logits, tokenizer, 20)
        
        # 统计top20中的英文token
        en_in_top20 = sum(1 for t in top20 if all(ord(c) < 128 for c in t['token'].strip()))
        
        baseline_results[zh] = {
            "en_translation": en,
            "en_prob": en_prob,
            "top20": top20,
            "en_tokens_in_top20": en_in_top20,
        }
    
    # 干预实验
    intervention_layers = [9, 15, 21, 27, 33]
    intervention_layers = [l for l in intervention_layers if l < n_layers]
    
    # α值: 基于hidden state范数校准
    # 目标: 干预的范数应该是hidden state范数的5-50%
    # L9 hidden norm ~100, so alpha should be 5-50
    # L18 hidden norm ~5000, so alpha should be 250-2500
    # 但我们用的是归一化方向(norm=1), 所以alpha就是添加的范数
    
    # 先用固定alpha测试, 然后用校准alpha
    alphas_fixed = [1.0, 5.0, 10.0, 50.0, 100.0]
    alphas_calibrated = {}  # 每层不同的alpha
    
    for l in intervention_layers:
        # 获取该层hidden state的平均范数
        zh_norms = [np.linalg.norm(all_hiddens["zh_continue"][pair][l]) 
                   for pair in TRANSLATION_TRAIN]
        mean_norm = float(np.mean(zh_norms))
        # 校准: alpha = 5%, 10%, 20%, 50% 的hidden state范数
        alphas_calibrated[l] = [0.05 * mean_norm, 0.1 * mean_norm, 0.2 * mean_norm, 0.5 * mean_norm]
    
    print(f"\n  校准alpha值:")
    for l in intervention_layers:
        print(f"    L{l}: alphas={[f'{a:.1f}' for a in alphas_calibrated[l]]}")
    
    intervention_results = {}
    
    for l_idx in intervention_layers:
        print(f"\n  --- 干预层: L{l_idx} ---")
        
        # 使用翻译特有Δh方向 (归一化)
        v_specific = trans_specific_dirs[l_idx]["mean_direction"]
        v_specific_tensor = torch.tensor(v_specific, dtype=torch.float32)
        
        # 也使用原始上下文化翻译方向作对比
        # 从all_hiddens中计算
        trans_hiddens_l = [all_hiddens["translate"][pair][l_idx] for pair in TRANSLATION_TRAIN]
        zh_hiddens_l = [all_hiddens["zh_continue"][pair][l_idx] for pair in TRANSLATION_TRAIN]
        contextual_deltas = [t - z for t, z in zip(trans_hiddens_l, zh_hiddens_l)]
        contextual_mean = np.mean(contextual_deltas, axis=0)
        contextual_norm = np.linalg.norm(contextual_mean)
        v_contextual = contextual_mean / contextual_norm if contextual_norm > 1e-6 else np.zeros(d_model)
        v_contextual_tensor = torch.tensor(v_contextual, dtype=torch.float32)
        
        layer_results = {}
        
        for alpha in alphas_calibrated[l_idx]:
            alpha_pct = alpha / (float(np.mean([np.linalg.norm(all_hiddens["zh_continue"][pair][l_idx]) 
                                               for pair in TRANSLATION_TRAIN])) + 1e-8) * 100
            
            # 1. Δh特有方向干预
            delta_h_results = {}
            for zh, en in TRANSLATION_TEST:
                zh_prompt = f"{zh}是一种"
                try:
                    intervened_logits = intervene_at_layer(
                        model, tokenizer, zh_prompt, l_idx, v_specific_tensor, alpha, device
                    )
                    en_prob = get_token_prob(intervened_logits, tokenizer, en)
                    top20 = get_top_k_tokens(intervened_logits, tokenizer, 20)
                    en_in_top20 = sum(1 for t in top20 if all(ord(c) < 128 for c in t['token'].strip()))
                    
                    delta_h_results[zh] = {
                        "en_prob": en_prob,
                        "en_prob_change": en_prob - baseline_results[zh]["en_prob"],
                        "en_tokens_in_top20": en_in_top20,
                        "top5": top20[:5],
                    }
                except Exception as e:
                    delta_h_results[zh] = {"error": str(e)}
            
            # 2. 上下文化翻译方向干预
            contextual_results = {}
            for zh, en in TRANSLATION_TEST:
                zh_prompt = f"{zh}是一种"
                try:
                    intervened_logits = intervene_at_layer(
                        model, tokenizer, zh_prompt, l_idx, v_contextual_tensor, alpha, device
                    )
                    en_prob = get_token_prob(intervened_logits, tokenizer, en)
                    top20 = get_top_k_tokens(intervened_logits, tokenizer, 20)
                    en_in_top20 = sum(1 for t in top20 if all(ord(c) < 128 for c in t['token'].strip()))
                    
                    contextual_results[zh] = {
                        "en_prob": en_prob,
                        "en_prob_change": en_prob - baseline_results[zh]["en_prob"],
                        "en_tokens_in_top20": en_in_top20,
                        "top5": top20[:5],
                    }
                except Exception as e:
                    contextual_results[zh] = {"error": str(e)}
            
            # 汇总
            dh_changes = [delta_h_results[zh].get("en_prob_change", 0) for zh in [z for z, e in TRANSLATION_TEST]
                         if "en_prob_change" in delta_h_results.get(zh, {})]
            ctx_changes = [contextual_results[zh].get("en_prob_change", 0) for zh in [z for z, e in TRANSLATION_TEST]
                          if "en_prob_change" in contextual_results.get(zh, {})]
            
            mean_dh = float(np.mean(dh_changes)) if dh_changes else 0
            mean_ctx = float(np.mean(ctx_changes)) if ctx_changes else 0
            
            # 英文token数量变化
            dh_en_tokens = [delta_h_results[zh].get("en_tokens_in_top20", 0) for zh in [z for z, e in TRANSLATION_TEST]]
            baseline_en_tokens = [baseline_results[zh]["en_tokens_in_top20"] for zh in [z for z, e in TRANSLATION_TEST]]
            
            print(f"    α={alpha:.1f} ({alpha_pct:.0f}%): "
                  f"Δh_dir ΔP(en)={mean_dh:.6f}, ctx_dir ΔP(en)={mean_ctx:.6f}, "
                  f"en_in_top20: {np.mean(dh_en_tokens):.1f} vs baseline {np.mean(baseline_en_tokens):.1f}")
            
            layer_results[f"alpha_{alpha:.1f}"] = {
                "alpha_pct": float(alpha_pct),
                "delta_h_intervention": delta_h_results,
                "contextual_intervention": contextual_results,
            }
        
        intervention_results[f"L{l_idx}"] = layer_results
    
    results["baseline"] = {zh: {"en_translation": v["en_translation"], "en_prob": v["en_prob"],
                                "en_tokens_in_top20": v["en_tokens_in_top20"]} 
                          for zh, v in baseline_results.items()}
    results["trans_specific_dirs"] = {str(l): {"alignment": v["alignment"], "mean_norm": v["mean_norm"],
                                                "dim_90": v["dim_90"]}
                                      for l, v in trans_specific_dirs.items()}
    results["intervention"] = intervention_results
    
    save_path = f"tests/glm5_temp/phase102b_exp1b_{model_name}_delta_h_intervention.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  保存到: {save_path}")
    
    release_model(model)
    return results


# ============================================================
# Exp 1c: Logit镜头分析
# ============================================================
def exp1c_logit_lens(model_name):
    """
    Logit镜头: 直接将每层hidden state投影到logit空间
    
    核心思路:
    - 如果h_l是层l的hidden state, 那么logits_l = h_l @ W_U^T
    - 这告诉我们: 如果模型在层l就停止, 它会输出什么?
    - 逐层跟踪翻译token的概率变化
    - 找到翻译信号首次出现的层
    
    这不干预, 只是观察。但比之前的观察更精确, 因为:
    - 我们关注的是logit空间(行为层面), 不是hidden state空间(表征层面)
    - 我们跟踪的是特定token的概率, 不是几何度量
    """
    print(f"\n{'='*70}")
    print(f"Exp 1c: Logit镜头分析 — {model_name}")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  模型: {model_name}, 层数: {n_layers}, d_model: {d_model}")
    
    # 获取W_U (lm_head权重)
    W_U = get_W_U(model)  # [vocab_size, d_model]
    W_U_tensor = torch.tensor(W_U, dtype=torch.float32)
    print(f"  W_U shape: {W_U.shape}")
    
    results = {}
    
    # 测试对
    test_pairs = TRANSLATION_TRAIN[:12] + TRANSLATION_TEST[:6]
    
    for zh, en in test_pairs:
        # 两种上下文
        zh_prompt = f"{zh}是一种"
        trans_prompt = f'请翻译：{zh} →'
        
        pair_results = {}
        
        for task_name, prompt in [("zh_continue", zh_prompt), ("translate", trans_prompt)]:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(inputs["input_ids"], output_hidden_states=True)
            
            layer_logit_probs = {}
            for l in range(n_layers + 1):
                h_l = outputs.hidden_states[l][0, -1, :].float().cpu()
                
                # LayerNorm (模拟最后的layer norm)
                # 注意: 实际模型在最终输出前还有layer norm
                # 但对于logit lens, 我们直接投影
                
                # 投影到logit空间
                logits_l = h_l @ W_U_tensor.T  # [vocab_size]
                probs_l = torch.softmax(logits_l, dim=-1)
                
                # 获取关键token的概率
                en_tok_ids = tokenizer.encode(en, add_special_tokens=False)
                zh_tok_ids = tokenizer.encode(zh, add_special_tokens=False)
                
                en_prob = probs_l[en_tok_ids[0]].item() if en_tok_ids else 0
                zh_prob = probs_l[zh_tok_ids[0]].item() if zh_tok_ids else 0
                
                # Top-1 token
                top1_id = torch.argmax(probs_l).item()
                top1_str = tokenizer.decode([top1_id])
                top1_prob = probs_l[top1_id].item()
                
                layer_logit_probs[str(l)] = {
                    "en_prob": en_prob,
                    "zh_prob": zh_prob,
                    "top1": top1_str,
                    "top1_prob": top1_prob,
                }
            
            pair_results[task_name] = layer_logit_probs
        
        results[f"{zh}_{en}"] = pair_results
        
        # 输出关键信息
        # 翻译信号首次出现的层 (en_prob > 0.01)
        first_en_layer = None
        for l in range(n_layers + 1):
            if pair_results["translate"][str(l)]["en_prob"] > 0.01:
                first_en_layer = l
                break
        
        # 中文上下文中en_prob最高的层
        max_en_in_zh = max([(l, pair_results["zh_continue"][str(l)]["en_prob"]) 
                           for l in range(n_layers + 1)], key=lambda x: x[1])
        
        print(f"  {zh}({en}): trans首次en>1%在L{first_en_layer}, "
              f"zh上下文max_en在L{max_en_in_zh[0]}(P={max_en_in_zh[1]:.4f})")
    
    # 汇总分析: 翻译信号出现的层分布
    print(f"\n  === 翻译信号出现的层分布 ===")
    first_en_layers = []
    for pair_key, pair_data in results.items():
        for l in range(n_layers + 1):
            if pair_data["translate"][str(l)]["en_prob"] > 0.01:
                first_en_layers.append(l)
                break
    
    if first_en_layers:
        print(f"    翻译prompt中, en_prob>1%首次出现在: "
              f"mean=L{np.mean(first_en_layers):.1f}, "
              f"range=L{min(first_en_layers)}-L{max(first_en_layers)}")
    
    # 对比: 在中文上下文中, en_prob的最高值
    max_en_in_zh_all = []
    for pair_key, pair_data in results.items():
        max_en = max([pair_data["zh_continue"][str(l)]["en_prob"] for l in range(n_layers + 1)])
        max_en_in_zh_all.append(max_en)
    
    print(f"    中文上下文中, max(en_prob): "
          f"mean={np.mean(max_en_in_zh_all):.6f}, "
          f"max={np.max(max_en_in_zh_all):.6f}")
    
    save_path = f"tests/glm5_temp/phase102b_exp1c_{model_name}_logit_lens.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  保存到: {save_path}")
    
    release_model(model)
    return results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, default="1b", choices=["1b", "1c"])
    args = parser.parse_args()
    
    if args.exp == "1b":
        exp1b_delta_h_subspace_intervention(args.model)
    elif args.exp == "1c":
        exp1c_logit_lens(args.model)
