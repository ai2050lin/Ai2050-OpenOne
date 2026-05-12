"""
Phase 137: 因果贡献分析 — 从相关性到因果性
==========================================

批评的核心修正:
1. Phase 136的"Spikiness>1"不能证明"分段程序" — 连续系统(如tanh(20x))也能有尖峰
2. 激活翻转≠计算图重构 — SwiGLU处处可微, flip只是高曲率区域
3. Transformer是"软条件系统": h_{l+1} = Σ α_i(x) f_i(h_l), 不是离散if-else
4. 真正重要的是"贡献流" — 哪些路径真正改变logits, 不是哪些neuron flip
5. 当前全是相关性, 没有因果性 — 需要activation patching

Phase 137策略:
  Exp 1: Activation Patching — 把否定句的某层激活patch到肯定句, 看logits如何变化
         直接测: 哪一层真正携带negation computation
  Exp 2: Logit Lens — 每层已经"知道"了什么? 对每层hidden state应用lm_head
  Exp 3: Weighted Contribution — 用实际激活幅度加权, 而非二值激活

关键方法:
  - Activation Patching: 用hook在目标层替换hidden state, 继续前向传播
  - Logit Lens: 用unembedding matrix投影各层hidden states
  - Weighted Jaccard: J_w = Σ min(|a_i|,|b_i|) / Σ max(|a_i|,|b_i|)
  - 大数据量: 20+句子对
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json
import time
import gc
import numpy as np
import torch
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, MODEL_CONFIGS, get_W_U
)


# ============================================================
# 句子设计
# ============================================================

NEGATION_PAIRS = [
    ("The dog always bites the man", "The dog never bites the man"),
    ("The cat always chases the mouse", "The cat never chases the mouse"),
    ("The sun always rises early", "The sun never rises early"),
    ("The river always flows south", "The river never flows south"),
    ("The wind always blows hard", "The wind never blows hard"),
    ("The bird always sings loud", "The bird never sings loud"),
    ("The fire always burns hot", "The fire never burns hot"),
    ("The child always plays hard", "The child never plays hard"),
    ("The doctor always helps patients", "The doctor never helps patients"),
    ("The teacher always reads books", "The teacher never reads books"),
    ("The soldier always fights hard", "The soldier never fights hard"),
    ("The farmer always grows crops", "The farmer never grows crops"),
    ("The artist always paints well", "The artist never paints well"),
    ("The writer always writes fast", "The writer never writes fast"),
    ("The driver always drives safe", "The driver never drives safe"),
    ("The bear always catches fish", "The bear never catches fish"),
    ("The wolf always hunts deer", "The wolf never hunts deer"),
    ("The fox always tricks crows", "The fox never tricks crows"),
    ("The king always rules wisely", "The king never rules wisely"),
    ("The queen always leads armies", "The queen never leads armies"),
]

TENSE_PAIRS = [
    ("The dog bites the man today", "The dog bit the man today"),
    ("The cat chases the mouse now", "The cat chased the mouse now"),
    ("The teacher helps the student now", "The teacher helped the student now"),
    ("The doctor treats the patient now", "The doctor treated the patient now"),
    ("The chef cooks the meal today", "The chef cooked the meal today"),
    ("The writer writes the book now", "The writer wrote the book now"),
    ("The child plays the game today", "The child played the game today"),
    ("The farmer grows the crop now", "The farmer grew the crop now"),
    ("The artist paints the picture now", "The artist painted the picture now"),
    ("The driver drives the car today", "The driver drove the car today"),
    ("The bird flies to the nest now", "The bird flew to the nest now"),
    ("The fish swims in the river today", "The fish swam in the river today"),
    ("The soldier guards the fort now", "The soldier guarded the fort now"),
    ("The nurse cares for the elder today", "The nurse cared for the elder today"),
    ("The scientist studies the atom now", "The scientist studied the atom now"),
    ("The bear catches the fish today", "The bear caught the fish today"),
    ("The wolf hunts the deer now", "The wolf hunted the deer now"),
    ("The king rules the kingdom today", "The king ruled the kingdom today"),
    ("The fire burns the forest now", "The fire burned the forest now"),
    ("The wind blows the leaves today", "The wind blew the leaves today"),
]

SEMANTIC_PAIRS = [
    ("The dog bites the man", "The cat bites the man"),
    ("The teacher helps the student", "The doctor helps the student"),
    ("The chef cooks the meal", "The writer cooks the meal"),
    ("The child plays the game", "The bird plays the game"),
    ("The soldier fights the enemy", "The farmer fights the enemy"),
    ("The fire burns the forest", "The water burns the forest"),
    ("The sun warms the earth", "The moon warms the earth"),
    ("The river carries the boat", "The wind carries the boat"),
    ("The king rules the kingdom", "The queen rules the kingdom"),
    ("The bear catches the fish", "The wolf catches the fish"),
]


# ============================================================
# 工具函数
# ============================================================

def get_top5(tokenizer, logits):
    """获取top-5预测"""
    top5_ids = np.argsort(logits)[-5:][::-1]
    return [(tokenizer.decode([i]).strip(), float(logits[i])) for i in top5_ids]


def get_device_for_input(model):
    """获取输入tensor应放的设备"""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Exp 1: Activation Patching — 因果追踪
# ============================================================

def exp1_activation_patching(model, tokenizer, device, model_info, model_name: str):
    """
    核心实验: 哪一层因果地实现了否定/时态?
    
    方法 (residual stream patching at diff positions):
    1. 运行 base sentence → logits_base, 所有层hidden states
    2. 运行 modified sentence → logits_mod, 所有层hidden states  
    3. 对每一层l:
       - 运行 base sentence, 但在层l处只替换差异token位置的hidden state
       - 其他位置保持base的hidden state
       - 获取 logits_patched
    4. 因果效应 = cosine_similarity(patched-base, mod-base)
    
    关键: 只替换差异token位置(如"always"→"never"的位置)
    - 整层替换没有因果信息(等于重跑modified句子, recovery永远≈1.0)
    - 只有差异位置patching才能揭示: 哪一层的hidden state携带了语法/语义信息
    
    两种patching:
    A) "Input patching": 在层l的INPUT处注入modified的差异位置hidden state
       → 测: 层l的输入是否携带了修改信息
    B) "Output patching": 在层l的OUTPUT处注入modified的差异位置hidden state
       → 测: 层l的输出是否携带了修改信息(即层l是否"加工"了修改信息)
    """

    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)

    # 采样层
    sample_step = max(1, n_layers // 12)
    sample_indices = sorted(set(
        list(range(0, n_layers, sample_step)) + [n_layers - 1]
    ))

    results = {}

    test_configs = [
        ("negation", NEGATION_PAIRS),
        ("tense", TENSE_PAIRS),
        ("semantic", SEMANTIC_PAIRS),
    ]

    for pair_type, pairs in test_configs:
        print(f"\n=== Exp 1: {pair_type} Activation Patching ===")
        type_results = []
        input_device = get_device_for_input(model)

        for pair_idx, (sent_base, sent_mod) in enumerate(pairs):
            print(f"  Pair {pair_idx+1}/{len(pairs)}: '{sent_base}' → '{sent_mod}'")

            ids_base = tokenizer.encode(sent_base, add_special_tokens=False)
            ids_mod = tokenizer.encode(sent_mod, add_special_tokens=False)

            if len(ids_base) != len(ids_mod):
                print(f"    SKIP: token数不同 ({len(ids_base)} vs {len(ids_mod)})")
                continue

            seq_len = len(ids_base)
            diff_pos = [i for i in range(seq_len) if ids_base[i] != ids_mod[i]]
            if not diff_pos:
                continue

            input_ids_base = torch.tensor([ids_base], device=input_device)
            input_ids_mod = torch.tensor([ids_mod], device=input_device)
            attention_mask = torch.ones(1, seq_len, device=input_device, dtype=torch.long)

            # === Step 1: 运行两个句子, 获取hidden states ===
            with torch.no_grad():
                out_base = model(input_ids=input_ids_base, attention_mask=attention_mask,
                                 output_hidden_states=True)
                out_mod = model(input_ids=input_ids_mod, attention_mask=attention_mask,
                                output_hidden_states=True)

            hs_base_all = [hs.detach().clone() for hs in out_base.hidden_states]
            hs_mod_all = [hs.detach().clone() for hs in out_mod.hidden_states]

            logits_base = out_base.logits[0, -1].float().cpu().numpy()
            logits_mod = out_mod.logits[0, -1].float().cpu().numpy()
            logit_diff = logits_mod - logits_base
            logit_diff_norm = float(np.linalg.norm(logit_diff))

            # === Step 2: 对每层做diff-position patching ===
            pair_data = {
                "sent_base": sent_base,
                "sent_mod": sent_mod,
                "diff_pos": diff_pos,
                "total_logit_diff": logit_diff_norm,
                "logits_base_top5": get_top5(tokenizer, logits_base),
                "logits_mod_top5": get_top5(tokenizer, logits_mod),
                "patching_diff_pos": {},
            }

            for li in sample_indices:
                lk = f"L{li}"

                # 只替换差异token位置的hidden state, 其他位置保持base
                # hs_base_all[li+1] 是层li的输出 (hs[0]=embedding)
                patched_hs = hs_base_all[li + 1].clone()
                for pos in diff_pos:
                    patched_hs[0, pos, :] = hs_mod_all[li + 1][0, pos, :]

                patched_logits = _patch_layer(
                    model, tokenizer, input_device, layers,
                    input_ids_base, attention_mask,
                    patched_hs, li
                )

                if patched_logits is None:
                    continue

                # 因果效应计算
                delta = patched_logits - logits_base

                # 1. 方向效应: patch是否把logits推向modified方向
                if logit_diff_norm > 1e-10:
                    directed_effect = float(np.dot(delta, logit_diff) / logit_diff_norm)
                else:
                    directed_effect = 0.0

                # 2. 恢复比例: cosine similarity between (patched-base) and (mod-base)
                delta_norm = float(np.linalg.norm(delta))
                if logit_diff_norm > 1e-10 and delta_norm > 1e-10:
                    cosine_recovery = float(np.dot(delta, logit_diff) /
                                            (delta_norm * logit_diff_norm))
                else:
                    cosine_recovery = 0.0

                # 3. 恢复比例(绝对): |patched-base| / |mod-base|
                relative_recovery = delta_norm / max(logit_diff_norm, 1e-10)

                pair_data["patching_diff_pos"][lk] = {
                    "directed_effect": directed_effect,
                    "cosine_recovery": cosine_recovery,
                    "relative_recovery": relative_recovery,
                    "top5_patched": get_top5(tokenizer, patched_logits),
                }

            type_results.append(pair_data)

            # 打印关键层
            mid_idx = len(sample_indices) // 2
            print_str = "    "
            for lk in [f"L{sample_indices[0]}", f"L{sample_indices[mid_idx]}", f"L{sample_indices[-1]}"]:
                pe = pair_data["patching_diff_pos"].get(lk, {})
                if pe:
                    print_str += f"{lk}: recovery={pe['cosine_recovery']:.3f}, directed={pe['directed_effect']:.2f}  "
            print(print_str)

            # 清理
            del hs_base_all, hs_mod_all
            torch.cuda.empty_cache()

        results[pair_type] = {
            "per_pair": type_results,
            "aggregated": _aggregate_patching(type_results, sample_indices),
        }

    return results


def _patch_layer(model, tokenizer, input_device, layers,
                  input_ids, attention_mask,
                  replace_tensor, target_layer):
    """
    在目标层处用hook替换hidden state, 继续前向传播获取logits
    
    replace_tensor: [1, seq_len, d_model] 替换后的tensor (只在差异位置修改)
    target_layer: 目标层索引
    
    关键: hook注册在目标层上, 替换该层的输出
    后续层会基于替换后的hidden state继续计算
    """

    captured = {"done": False}

    def replace_hook(module, input, output):
        if not captured["done"]:
            captured["done"] = True
            if isinstance(output, tuple):
                new_hidden = replace_tensor.to(output[0].device).to(output[0].dtype)
                return (new_hidden,) + output[1:]
            return replace_tensor.to(output.device).to(output.dtype)
        return output

    hook = layers[target_layer].register_forward_hook(replace_hook)

    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[0, -1].float().cpu().numpy()
    except Exception as e:
        print(f"    Patch forward failed at L{target_layer}: {e}")
        logits = None
    finally:
        hook.remove()

    return logits


def _aggregate_patching(per_pair, sample_indices):
    """聚合patching效应"""
    agg = {}

    for li in sample_indices:
        lk = f"L{li}"
        directed = []
        cosine = []
        relative = []

        for pair_data in per_pair:
            pe = pair_data.get("patching_diff_pos", {}).get(lk, {})
            if "directed_effect" in pe:
                directed.append(pe["directed_effect"])
                cosine.append(pe["cosine_recovery"])
                relative.append(pe["relative_recovery"])

        if directed:
            agg[lk] = {
                "directed_effect_mean": float(np.mean(directed)),
                "directed_effect_std": float(np.std(directed)),
                "cosine_recovery_mean": float(np.mean(cosine)),
                "cosine_recovery_std": float(np.std(cosine)),
                "relative_recovery_mean": float(np.mean(relative)),
                "relative_recovery_std": float(np.std(relative)),
                "n_pairs": len(directed),
            }

    return agg


# ============================================================
# Exp 2: Logit Lens — 每层已经"知道"了什么?
# ============================================================

def exp2_logit_lens(model, tokenizer, device, model_info, model_name: str):
    """
    Logit Lens: 对每层的hidden state应用lm_head, 看每层预测什么
    
    这比hidden state几何更直接:
    "每层已经积累了什么信息?" 而非 "hidden state长什么样"
    """

    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)

    W_U = get_W_U(model, model_name)  # [vocab_size, d_model]
    print(f"  W_U shape: {W_U.shape}")

    sample_step = max(1, n_layers // 12)
    sample_indices = sorted(set(
        list(range(0, n_layers, sample_step)) + [n_layers - 1]
    ))

    test_configs = [
        ("negation", NEGATION_PAIRS[:10]),
        ("tense", TENSE_PAIRS[:10]),
        ("semantic", SEMANTIC_PAIRS[:5]),
    ]

    results = {}
    input_device = get_device_for_input(model)

    for pair_type, pairs in test_configs:
        print(f"\n  --- {pair_type} Logit Lens ---")
        type_results = []

        for pair_idx, (sent_base, sent_mod) in enumerate(pairs):
            ids_base = tokenizer.encode(sent_base, add_special_tokens=False)
            ids_mod = tokenizer.encode(sent_mod, add_special_tokens=False)

            if len(ids_base) != len(ids_mod):
                continue

            seq_len = len(ids_base)
            diff_pos = [i for i in range(seq_len) if ids_base[i] != ids_mod[i]]
            if not diff_pos:
                continue

            input_ids_base = torch.tensor([ids_base], device=input_device)
            input_ids_mod = torch.tensor([ids_mod], device=input_device)
            attention_mask = torch.ones(1, seq_len, device=input_device, dtype=torch.long)

            with torch.no_grad():
                out_base = model(input_ids=input_ids_base, attention_mask=attention_mask,
                                 output_hidden_states=True)
                out_mod = model(input_ids=input_ids_mod, attention_mask=attention_mask,
                                output_hidden_states=True)

            hs_base = out_base.hidden_states
            hs_mod = out_mod.hidden_states

            pair_data = {
                "sent_base": sent_base,
                "sent_mod": sent_mod,
                "diff_pos": diff_pos,
                "layer_predictions": {},
            }

            for li in sample_indices:
                lk = f"L{li}"
                # hs_base[li+1] 是层li的输出 (hs[0]=embedding)

                # 只看差异token位置和last token位置
                for pos_type, pos_idx in [("diff", diff_pos[0]),
                                           ("last", seq_len - 1)]:
                    h_base = hs_base[li+1][0, pos_idx].float().cpu().numpy()
                    h_mod = hs_mod[li+1][0, pos_idx].float().cpu().numpy()

                    # Logit Lens: logits = W_U @ h + b (bias通常可忽略)
                    logits_base_l = W_U @ h_base
                    logits_mod_l = W_U @ h_mod

                    logit_diff_norm = float(np.linalg.norm(logits_mod_l - logits_base_l))

                    # KL divergence
                    from scipy.special import softmax as scipy_softmax
                    p_base = scipy_softmax(logits_base_l - np.max(logits_base_l))
                    p_mod = scipy_softmax(logits_mod_l - np.max(logits_mod_l))
                    kl_div = float(np.sum(p_mod * np.log(
                        np.maximum(p_mod / np.maximum(p_base, 1e-10), 1e-10)
                    )))

                    key = f"{lk}_{pos_type}"
                    pair_data["layer_predictions"][key] = {
                        "top5_base": get_top5(tokenizer, logits_base_l),
                        "top5_mod": get_top5(tokenizer, logits_mod_l),
                        "logit_diff_norm": logit_diff_norm,
                        "kl_divergence": kl_div,
                    }

            type_results.append(pair_data)

        results[pair_type] = type_results

    # 聚合
    agg = {}
    for pair_type, type_results in results.items():
        type_agg = {}
        for li in sample_indices:
            lk = f"L{li}"
            for pos_type in ["diff", "last"]:
                diff_norms = []
                kl_divs = []

                for pair_data in type_results:
                    key = f"{lk}_{pos_type}"
                    if key in pair_data["layer_predictions"]:
                        diff_norms.append(pair_data["layer_predictions"][key]["logit_diff_norm"])
                        kl_divs.append(pair_data["layer_predictions"][key]["kl_divergence"])

                if diff_norms:
                    if lk not in type_agg:
                        type_agg[lk] = {}
                    type_agg[lk][f"{pos_type}_logit_diff_mean"] = float(np.mean(diff_norms))
                    type_agg[lk][f"{pos_type}_kl_mean"] = float(np.mean(kl_divs))

        agg[pair_type] = type_agg

    return {"per_pair": results, "aggregated": agg}


# ============================================================
# Exp 3: Weighted Contribution — 加权贡献分析
# ============================================================

def exp3_weighted_contribution(model, tokenizer, device, model_info, model_name: str):
    """
    用实际激活幅度加权, 而非二值激活
    
    Phase 135的Jaccard问题: 把强激活和弱激活等价处理
    修正: Weighted Jaccard = Σ min(|a|,|b|) / Σ max(|a|,|b|)
    
    同时计算: 各层hidden state变化对最终logits的贡献
    即: Δlogits ≈ W_U @ Δh_last_token
    """

    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)

    W_U = get_W_U(model, model_name)

    sample_step = max(1, n_layers // 12)
    sample_indices = sorted(set(
        list(range(0, n_layers, sample_step)) + [n_layers - 1]
    ))

    test_configs = [
        ("negation", NEGATION_PAIRS[:15]),
        ("tense", TENSE_PAIRS[:10]),
        ("semantic", SEMANTIC_PAIRS[:5]),
    ]

    results = {}
    input_device = get_device_for_input(model)

    for pair_type, pairs in test_configs:
        print(f"\n  --- {pair_type} Weighted Contribution ---")
        type_results = []

        for pair_idx, (sent_base, sent_mod) in enumerate(pairs):
            ids_base = tokenizer.encode(sent_base, add_special_tokens=False)
            ids_mod = tokenizer.encode(sent_mod, add_special_tokens=False)

            if len(ids_base) != len(ids_mod):
                continue

            seq_len = len(ids_base)
            diff_pos = [i for i in range(seq_len) if ids_base[i] != ids_mod[i]]
            if not diff_pos:
                continue

            input_ids_base = torch.tensor([ids_base], device=input_device)
            input_ids_mod = torch.tensor([ids_mod], device=input_device)
            attention_mask = torch.ones(1, seq_len, device=input_device, dtype=torch.long)

            with torch.no_grad():
                out_base = model(input_ids=input_ids_base, attention_mask=attention_mask,
                                 output_hidden_states=True)
                out_mod = model(input_ids=input_ids_mod, attention_mask=attention_mask,
                                output_hidden_states=True)

            hs_base = out_base.hidden_states
            hs_mod = out_mod.hidden_states

            pair_data = {
                "sent_base": sent_base,
                "sent_mod": sent_mod,
                "diff_pos": diff_pos,
                "layer_analysis": {},
            }

            for li in sample_indices:
                lk = f"L{li}"

                h_base = hs_base[li+1][0].float().cpu().numpy()  # [seq_len, d_model]
                h_mod = hs_mod[li+1][0].float().cpu().numpy()

                # 1. 加权Jaccard (差异token位置)
                h_base_diff = h_base[diff_pos]
                h_mod_diff = h_mod[diff_pos]

                abs_base = np.abs(h_base_diff.flatten())
                abs_mod = np.abs(h_mod_diff.flatten())
                weighted_jaccard = float(
                    np.sum(np.minimum(abs_base, abs_mod)) /
                    max(np.sum(np.maximum(abs_base, abs_mod)), 1e-10)
                )

                # 二值Jaccard (for comparison, threshold=0.1)
                bin_base = (abs_base > 0.1).astype(np.float32)
                bin_mod = (abs_mod > 0.1).astype(np.float32)
                intersection = np.sum(np.minimum(bin_base, bin_mod))
                union = np.sum(np.maximum(bin_base, bin_mod))
                binary_jaccard = float(intersection / max(union, 1e-10))

                # 2. Logit贡献: W_U @ Δh (last token)
                delta_h_last = h_mod[-1] - h_base[-1]
                delta_h_diff_pos = h_mod[diff_pos[0]] - h_base[diff_pos[0]]

                logit_contribution_last = W_U @ delta_h_last
                logit_contribution_diff = W_U @ delta_h_diff_pos

                # 3. 贡献效率: |W_U @ Δh| / |Δh|
                delta_norm = float(np.linalg.norm(delta_h_last))
                logit_contrib_norm = float(np.linalg.norm(logit_contribution_last))
                efficiency = logit_contrib_norm / max(delta_norm, 1e-10)

                # 4. 方向一致性: Δh在W_U行空间中的投影比例
                # 这告诉我们Δh有多少被"传达到"logits
                proj_ratio = logit_contrib_norm**2 / max(delta_norm**2, 1e-10)
                proj_ratio = min(proj_ratio, 1.0)  # cap at 1

                # 5. 各层Δh的范数 (看哪层变化最大)
                delta_h_all = h_mod - h_base  # [seq_len, d_model]
                delta_h_norm_all = float(np.linalg.norm(delta_h_all))

                pair_data["layer_analysis"][lk] = {
                    "weighted_jaccard": weighted_jaccard,
                    "binary_jaccard": binary_jaccard,
                    "delta_h_norm_last": delta_norm,
                    "delta_h_norm_all": delta_h_norm_all,
                    "logit_contribution_norm_last": logit_contrib_norm,
                    "logit_contribution_norm_diff": float(np.linalg.norm(logit_contribution_diff)),
                    "contribution_efficiency": efficiency,
                    "projection_ratio": proj_ratio,
                }

            type_results.append(pair_data)

        results[pair_type] = type_results

    # 聚合
    agg = {}
    for pair_type, type_results in results.items():
        type_agg = {}
        for li in sample_indices:
            lk = f"L{li}"
            metrics = defaultdict(list)

            for pair_data in type_results:
                la = pair_data.get("layer_analysis", {}).get(lk, {})
                for key in ["weighted_jaccard", "binary_jaccard", "delta_h_norm_last",
                            "logit_contribution_norm_last", "contribution_efficiency",
                            "projection_ratio", "delta_h_norm_all"]:
                    if key in la:
                        metrics[key].append(la[key])

            if metrics:
                type_agg[lk] = {f"{k}_mean": float(np.mean(v)) for k, v in metrics.items()}
                type_agg[lk]["n_pairs"] = len(metrics.get("weighted_jaccard", []))

        agg[pair_type] = type_agg

    return {"per_pair": results, "aggregated": agg}


# ============================================================
# 主函数
# ============================================================

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    model_name = model_name.lower()

    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}")
        print(f"Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    print(f"=" * 60)
    print(f"Phase 137: 因果贡献分析")
    print(f"Model: {model_name}")
    print(f"=" * 60)

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"Model info: {model_info.n_layers} layers, d_model={model_info.d_model}, "
          f"intermediate={model_info.intermediate_size}, mlp_type={model_info.mlp_type}")

    all_results = {
        "model_info": {
            "name": model_name,
            "class": model_info.model_class,
            "n_layers": model_info.n_layers,
            "d_model": model_info.d_model,
            "intermediate_size": model_info.intermediate_size,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    t0 = time.time()

    # Exp 1: Activation Patching
    print(f"\n{'='*60}")
    print("Exp 1: Activation Patching — 因果追踪")
    print(f"{'='*60}")
    exp1_results = exp1_activation_patching(model, tokenizer, device, model_info, model_name)
    all_results["exp1_patching"] = exp1_results

    # Exp 2: Logit Lens
    print(f"\n{'='*60}")
    print("Exp 2: Logit Lens — 每层知道了什么")
    print(f"{'='*60}")
    exp2_results = exp2_logit_lens(model, tokenizer, device, model_info, model_name)
    all_results["exp2_logit_lens"] = exp2_results

    # Exp 3: Weighted Contribution
    print(f"\n{'='*60}")
    print("Exp 3: Weighted Contribution — 加权贡献分析")
    print(f"{'='*60}")
    exp3_results = exp3_weighted_contribution(model, tokenizer, device, model_info, model_name)
    all_results["exp3_weighted"] = exp3_results

    total_time = time.time() - t0
    all_results["total_time_seconds"] = round(total_time, 1)

    # 保存结果
    output_path = f"tests/glm5_temp/phase137_{model_name}_causal_contribution.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    simplified = _simplify_results(all_results)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(simplified, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {output_path}")

    _print_key_results(simplified, model_info)

    release_model(model)


def _simplify_results(results):
    """简化结果"""
    simplified = {
        "model_info": results["model_info"],
        "timestamp": results["timestamp"],
        "total_time_seconds": results.get("total_time_seconds"),
    }

    # Exp 1: 只保留聚合 + 简化的per_pair
    if "exp1_patching" in results:
        exp1 = results["exp1_patching"]
        simplified_exp1 = {}
        for pair_type in ["negation", "tense", "semantic"]:
            if pair_type in exp1:
                data = exp1[pair_type]
                simplified_exp1[pair_type] = {
                    "aggregated": data.get("aggregated", {}),
                    "per_pair_summary": [],
                }
                for pair_data in data.get("per_pair", []):
                    summary = {
                        "sent_base": pair_data.get("sent_base"),
                        "sent_mod": pair_data.get("sent_mod"),
                        "total_logit_diff": pair_data.get("total_logit_diff"),
                        "logits_base_top5": pair_data.get("logits_base_top5"),
                        "logits_mod_top5": pair_data.get("logits_mod_top5"),
                        "patching_summary": {},
                    }
                    for lk, pe in pair_data.get("patching_diff_pos", {}).items():
                        summary["patching_summary"][lk] = {
                            "directed_effect": pe.get("directed_effect"),
                            "cosine_recovery": pe.get("cosine_recovery"),
                            "relative_recovery": pe.get("relative_recovery"),
                        }
                    simplified_exp1[pair_type]["per_pair_summary"].append(summary)
        simplified["exp1_patching"] = simplified_exp1

    # Exp 2: 只保留聚合
    if "exp2_logit_lens" in results:
        simplified["exp2_logit_lens"] = {
            "aggregated": results["exp2_logit_lens"].get("aggregated", {}),
        }

    # Exp 3: 只保留聚合
    if "exp3_weighted" in results:
        simplified["exp3_weighted"] = {
            "aggregated": results["exp3_weighted"].get("aggregated", {}),
        }

    return simplified


def _print_key_results(results, model_info):
    """打印关键结果"""
    print(f"\n{'='*60}")
    print("Phase 137 关键结果")
    print(f"{'='*60}")

    # Exp 1
    exp1 = results.get("exp1_patching", {})
    for pair_type in ["negation", "tense", "semantic"]:
        agg = exp1.get(pair_type, {}).get("aggregated", {})
        if not agg:
            continue
        print(f"\n--- {pair_type} Patching 因果效应 (diff-pos only) ---")
        max_layer = None
        max_cosine = -2
        for lk in sorted(agg.keys()):
            data = agg[lk]
            de = data.get("directed_effect_mean", 0)
            cr = data.get("cosine_recovery_mean", 0)
            rr = data.get("relative_recovery_mean", 0)
            print(f"  {lk}: directed={de:.4f}±{data.get('directed_effect_std', 0):.4f}, "
                  f"cosine_recovery={cr:.4f}±{data.get('cosine_recovery_std', 0):.4f}, "
                  f"relative_recovery={rr:.4f}")
            if cr > max_cosine:
                max_cosine = cr
                max_layer = lk
        if max_layer:
            print(f"  → 最大因果效应层: {max_layer} (cosine_recovery={max_cosine:.4f})")

    # Exp 2
    exp2 = results.get("exp2_logit_lens", {})
    agg2 = exp2.get("aggregated", {})
    for pair_type in ["negation", "tense", "semantic"]:
        if pair_type not in agg2:
            continue
        print(f"\n--- {pair_type} Logit Lens ---")
        for lk in sorted(agg2[pair_type].keys()):
            data = agg2[pair_type][lk]
            print(f"  {lk}: diff_logit_diff={data.get('diff_logit_diff_mean', 0):.4f}, "
                  f"last_logit_diff={data.get('last_logit_diff_mean', 0):.4f}, "
                  f"diff_kl={data.get('diff_kl_mean', 0):.4f}")

    # Exp 3
    exp3 = results.get("exp3_weighted", {})
    agg3 = exp3.get("aggregated", {})
    for pair_type in ["negation", "tense", "semantic"]:
        if pair_type not in agg3:
            continue
        print(f"\n--- {pair_type} Weighted Contribution ---")
        for lk in sorted(agg3[pair_type].keys()):
            data = agg3[pair_type][lk]
            print(f"  {lk}: w_jac={data.get('weighted_jaccard_mean', 0):.4f}, "
                  f"b_jac={data.get('binary_jaccard_mean', 0):.4f}, "
                  f"delta_norm={data.get('delta_h_norm_last_mean', 0):.4f}, "
                  f"logit_contrib={data.get('logit_contribution_norm_last_mean', 0):.4f}, "
                  f"efficiency={data.get('contribution_efficiency_mean', 0):.4f}, "
                  f"proj_ratio={data.get('projection_ratio_mean', 0):.4f}")


if __name__ == "__main__":
    main()
