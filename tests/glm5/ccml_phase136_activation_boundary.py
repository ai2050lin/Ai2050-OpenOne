"""
Phase 136: 激活边界分析 — Transformer是分段条件程序还是光滑动力系统?
================================================================

核心问题:
  当输入连续变化时, 激活模式是平滑变化还是离散相变?

批评指出:
  - Transformer更像"分段条件程序"而非"光滑动力系统"
  - 中间层是"句法计算区" — 激活集合重组最剧烈
  - 语言理解 = 条件路径选择, 不是向量位置
  - 应研究激活边界, 而非Jacobian/几何

Phase 136实验:
  Exp 1: 输入插值路径 — 在两个不同语义的embedding之间插值
         测量每个α处的激活翻转率(activation flip rate)
         如果有尖锐峰 → 离散相变 → 支持分段程序假说
         如果平滑变化 → 连续变换 → 支持几何假说

  Exp 2: 边界密度 — 随机方向微扰, 统计各层边界穿越概率
         预测: 中间层边界密度最高

  Exp 3: 否定vs语义插值 — 否定插值是否比语义插值产生更多相变?

关键方法:
  - 选取token数量相同的句子对, 只插值差异token的embedding
  - 对每个α运行完整模型, hook住各层MLP和attention
  - 计算 consecutive α 之间的 activation flip rate
  - 用50+插值点获得高分辨率
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
    load_model, get_layers, get_model_info, release_model, MODEL_CONFIGS
)


# ============================================================
# 句子对设计 — token数量相同, 只有一个词不同
# ============================================================

# 语义对比: 同一句型, 只换一个语义词
SEMANTIC_PAIRS = [
    ("The weather is good", "The weather is bad"),
    ("The man is tall", "The man is short"),
    ("The cat is alive", "The cat is dead"),
    ("The door is open", "The door is closed"),
    ("The water is hot", "The water is cold"),
    ("The light is bright", "The light is dark"),
    ("The road is wide", "The road is narrow"),
    ("The food is sweet", "The food is bitter"),
    ("The bird can fly", "The bird can swim"),
    ("The ice is solid", "The ice is liquid"),
    ("The king is rich", "The king is poor"),
    ("The car is fast", "The car is slow"),
    ("The room is clean", "The room is dirty"),
    ("The fire is strong", "The fire is weak"),
    ("The tree is young", "The tree is old"),
]

# 否定对比: 使用always/never保持token数相同
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
]

# 时态对比: 动词原形 vs 过去式 (token数可能不同, 需要特殊处理)
# 改为使用更简洁的对比
TENSE_PAIRS = [
    ("He walks to school today", "He walked to school today"),
    ("She reads the book now", "She read the book now"),
    ("They play the game here", "They played the game here"),
    ("We cook the meal daily", "We cooked the meal daily"),
    ("I write the letter now", "I wrote the letter now"),
]


def get_differing_token_pos(tokenizer, sent1: str, sent2: str):
    """
    找到两个句子中不同的token位置
    假设两个句子token数量相同, 只有少数token不同
    返回: (differing_positions, tok1_ids, tok2_ids)
    """
    ids1 = tokenizer.encode(sent1, add_special_tokens=False)
    ids2 = tokenizer.encode(sent2, add_special_tokens=False)

    # 找最小公共长度
    min_len = min(len(ids1), len(ids2))
    diff_pos = []
    for i in range(min_len):
        if ids1[i] != ids2[i]:
            diff_pos.append(i)

    return diff_pos, ids1, ids2


# ============================================================
# Exp 1: 输入插值路径
# ============================================================

def exp1_interpolation_path(model, tokenizer, device, model_info, model_name: str):
    """
    核心实验: 输入embedding插值, 测量激活翻转率

    方法:
    1. 选取句子对 (token数相同)
    2. 获取两个句子的embedding
    3. 对差异token的embedding做插值: e(α) = (1-α)*e1 + α*e2
    4. 对每个α运行模型, hook住MLP gate激活和attention pattern
    5. 计算 consecutive α 之间的 activation flip rate
    """

    n_layers = model_info.n_layers
    d_model = model_info.d_model
    intermediate_size = model_info.intermediate_size
    mlp_type = model_info.mlp_type
    layers = get_layers(model)

    # 采样层
    n_sample = min(8, n_layers)
    sample_step = max(1, n_layers // n_sample)
    sample_indices = sorted(set(list(range(0, n_layers, sample_step)) + [n_layers - 1]))

    n_interp = 51  # 插值点数: 0.0, 0.02, ..., 1.0
    alphas = np.linspace(0, 1, n_interp)

    # MLP激活阈值
    thresholds = [0.0, 0.1, 0.5]

    # Top-k attention边
    top_k_attn = 5

    embed_layer = model.get_input_embeddings()

    all_results = {}

    # --- 语义插值 ---
    print("\n=== Exp 1a: 语义插值 (good→bad, tall→short, etc.) ===")
    semantic_results = _run_interpolation_batch(
        model, tokenizer, device, model_name, embed_layer,
        layers, sample_indices, n_layers, d_model, intermediate_size, mlp_type,
        SEMANTIC_PAIRS, alphas, thresholds, top_k_attn, "semantic"
    )
    all_results["semantic"] = semantic_results

    # --- 否定插值 ---
    print("\n=== Exp 1b: 否定插值 (always→never) ===")
    negation_results = _run_interpolation_batch(
        model, tokenizer, device, model_name, embed_layer,
        layers, sample_indices, n_layers, d_model, intermediate_size, mlp_type,
        NEGATION_PAIRS, alphas, thresholds, top_k_attn, "negation"
    )
    all_results["negation"] = negation_results

    return all_results


def _run_interpolation_batch(model, tokenizer, device, model_name,
                              embed_layer, layers, sample_indices, n_layers,
                              d_model, intermediate_size, mlp_type,
                              sentence_pairs, alphas, thresholds, top_k_attn,
                              pair_type):
    """
    批量运行插值实验
    """
    n_interp = len(alphas)
    n_heads = 32  # 默认, 后面会更新

    # 获取n_heads
    try:
        layer0 = layers[0]
        n_heads = layer0.self_attn.num_heads
    except:
        n_heads = 32

    results = {
        "pair_type": pair_type,
        "n_pairs": len(sentence_pairs),
        "n_interp": n_interp,
        "alphas": alphas.tolist(),
        "per_pair": [],
        "aggregated": {},
    }

    for pair_idx, (sent1, sent2) in enumerate(sentence_pairs):
        print(f"  Pair {pair_idx+1}/{len(sentence_pairs)}: '{sent1}' → '{sent2}'")

        # tokenize
        ids1 = tokenizer.encode(sent1, add_special_tokens=False)
        ids2 = tokenizer.encode(sent2, add_special_tokens=False)

        # 检查token数是否相同
        if len(ids1) != len(ids2):
            print(f"    SKIP: token数不同 ({len(ids1)} vs {len(ids2)})")
            continue

        seq_len = len(ids1)

        # 找到差异token位置
        diff_pos = []
        for i in range(seq_len):
            if ids1[i] != ids2[i]:
                diff_pos.append(i)

        if len(diff_pos) == 0:
            print(f"    SKIP: 没有差异token")
            continue

        print(f"    差异token位置: {diff_pos}")

        # 获取两个句子的embeddings
        input_ids1 = torch.tensor([ids1], device=device)
        input_ids2 = torch.tensor([ids2], device=device)

        emb1 = embed_layer(input_ids1).detach().clone()  # [1, seq_len, d_model]
        emb2 = embed_layer(input_ids2).detach().clone()

        # 对每个α运行插值
        pair_data = {
            "sent1": sent1,
            "sent2": sent2,
            "diff_pos": diff_pos,
            "mlp_masks": {},  # {layer_key: {threshold: [[alpha_idx, neuron_mask]]}}
            "attn_topk": {},  # {layer_key: [[alpha_idx, topk_edges]]}
        }

        # 存储每个α的激活状态
        alpha_mlp_masks = {li: {t: [] for t in thresholds} for li in sample_indices}
        alpha_attn_topk = {li: [] for li in sample_indices}

        for alpha_idx, alpha in enumerate(alphas):
            # 插值embedding
            interp_emb = emb1.clone()
            for pos in diff_pos:
                interp_emb[0, pos, :] = (1 - alpha) * emb1[0, pos, :] + alpha * emb2[0, pos, :]

            # Hook收集MLP gate输出和attention
            captured_mlp = {}
            captured_attn = {}

            def make_mlp_hook(key, mlp_type_local):
                def hook(module, input, output):
                    if mlp_type_local == "merged_gate_up":
                        # GLM4: gate_up_proj合并, 输出前半是gate
                        gate = output[:, :, :output.shape[-1]//2]
                        captured_mlp[key] = gate.detach().float().cpu()
                    else:
                        captured_mlp[key] = output.detach().float().cpu()
                return hook

            def make_attn_hook(key):
                def hook(module, input, output):
                    # output是tuple, 最后一个元素是attention weights (如果output_attentions=True)
                    if isinstance(output, tuple) and len(output) > 1:
                        # 有些模型output[1]是attention weights
                        attn = output[1]
                        if attn is not None:
                            captured_attn[key] = attn.detach().float().cpu()
                return hook

            hooks = []
            for li in sample_indices:
                layer = layers[li]
                mlp = layer.mlp if hasattr(layer, "mlp") else None
                if mlp is not None:
                    if mlp_type == "merged_gate_up":
                        hooks.append(mlp.gate_up_proj.register_forward_hook(
                            make_mlp_hook(f"L{li}", mlp_type)))
                    elif hasattr(mlp, 'gate_proj'):
                        hooks.append(mlp.gate_proj.register_forward_hook(
                            make_mlp_hook(f"L{li}", mlp_type)))
                    elif hasattr(mlp, 'up_proj'):
                        hooks.append(mlp.up_proj.register_forward_hook(
                            make_mlp_hook(f"L{li}", mlp_type)))

                # Attention hook
                if hasattr(layer, 'self_attn'):
                    hooks.append(layer.self_attn.register_forward_hook(
                        make_attn_hook(f"L{li}")))

            # 前向推理
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
            try:
                with torch.no_grad():
                    _ = model(inputs_embeds=interp_emb.to(model.dtype),
                             position_ids=position_ids,
                             output_attentions=True)
            except Exception as e:
                print(f"    Forward failed at alpha={alpha:.2f}: {e}")
                for h in hooks:
                    h.remove()
                continue

            for h in hooks:
                h.remove()

            # 处理MLP激活
            for li in sample_indices:
                key = f"L{li}"
                if key not in captured_mlp:
                    continue
                mlp_out = captured_mlp[key][0].numpy()  # [seq_len, intermediate]

                for t in thresholds:
                    mask = (mlp_out > t).astype(np.int8)
                    alpha_mlp_masks[li][t].append(mask)

            # 处理Attention top-k
            for li in sample_indices:
                key = f"L{li}"
                if key not in captured_attn:
                    continue
                attn = captured_attn[key]  # [1, n_heads, seq_len, seq_len]
                if attn.dim() == 4:
                    attn_np = attn[0].float().numpy()  # [n_heads, seq_len, seq_len]
                else:
                    continue

                # 每个head的top-k边
                head_topk = []
                for h in range(min(n_heads, attn_np.shape[0])):
                    attn_h = attn_np[h]  # [seq_len, seq_len]
                    flat = attn_h.flatten()
                    k = min(top_k_attn, len(flat))
                    top_indices = np.argpartition(flat, -k)[-k:]
                    top_indices = top_indices[np.argsort(flat[top_indices])[::-1]]
                    edges = set()
                    for idx in top_indices:
                        i = int(idx // seq_len)
                        j = int(idx % seq_len)
                        edges.add((i, j))
                    head_topk.append(edges)

                alpha_attn_topk[li].append(head_topk)

            # 清理
            del captured_mlp, captured_attn
            if alpha_idx % 10 == 0:
                torch.cuda.empty_cache()

        # === 计算激活翻转率 ===
        flip_rates = {}
        for li in sample_indices:
            lk = f"L{li}"
            layer_flips = {}

            for t in thresholds:
                masks = alpha_mlp_masks[li][t]
                if len(masks) < 2:
                    continue

                # 计算consecutive α之间的flip rate
                flip_series = []
                for i in range(1, len(masks)):
                    # 考虑所有token位置
                    flip = np.mean(np.abs(masks[i].astype(np.float32) - masks[i-1].astype(np.float32)))
                    flip_series.append(float(flip))

                # 也计算: 只考虑差异token位置的flip rate
                diff_flips = []
                for i in range(1, len(masks)):
                    m_cur = masks[i][diff_pos, :]  # 只取差异位置
                    m_prev = masks[i-1][diff_pos, :]
                    flip_diff = np.mean(np.abs(m_cur.astype(np.float32) - m_prev.astype(np.float32)))
                    diff_flips.append(float(flip_diff))

                layer_flips[f"t{t}_all_pos"] = flip_series
                layer_flips[f"t{t}_diff_pos"] = diff_flips

                # 统计
                flip_arr = np.array(flip_series)
                layer_flips[f"t{t}_max_flip"] = float(np.max(flip_arr))
                layer_flips[f"t{t}_mean_flip"] = float(np.mean(flip_arr))
                layer_flips[f"t{t}_flip_spikiness"] = float(np.max(flip_arr) / max(np.mean(flip_arr), 1e-10))

                # 相变检测: flip rate > mean + 3*std 的点
                if len(flip_arr) > 2:
                    mean_f = np.mean(flip_arr)
                    std_f = np.std(flip_arr)
                    spikes = [i for i, f in enumerate(flip_series) if f > mean_f + 3 * std_f]
                    layer_flips[f"t{t}_n_spikes"] = len(spikes)
                    layer_flips[f"t{t}_spike_positions"] = spikes

            # Attention top-k边的切换率
            attn_data = alpha_attn_topk[li]
            if len(attn_data) >= 2:
                edge_switch_series = []
                for i in range(1, len(attn_data)):
                    # 各head的平均Jaccard距离
                    head_jacs = []
                    for h in range(min(n_heads, len(attn_data[i]), len(attn_data[i-1]))):
                        e1 = attn_data[i-1][h]
                        e2 = attn_data[i][h]
                        union = len(e1 | e2)
                        if union > 0:
                            jac = len(e1 & e2) / union
                        else:
                            jac = 1.0
                        head_jacs.append(jac)
                    # switch rate = 1 - jaccard
                    switch_rate = 1.0 - np.mean(head_jacs) if head_jacs else 0
                    edge_switch_series.append(float(switch_rate))

                layer_flips["attn_switch"] = edge_switch_series
                switch_arr = np.array(edge_switch_series)
                layer_flips["attn_max_switch"] = float(np.max(switch_arr)) if len(switch_arr) > 0 else 0
                layer_flips["attn_mean_switch"] = float(np.mean(switch_arr)) if len(switch_arr) > 0 else 0
                layer_flips["attn_switch_spikiness"] = float(np.max(switch_arr) / max(np.mean(switch_arr), 1e-10)) if len(switch_arr) > 0 else 0

            flip_rates[lk] = layer_flips

        pair_data["flip_rates"] = flip_rates
        results["per_pair"].append(pair_data)

        print(f"    完成. MLP flip rates (L_mid, t0.0): "
              f"max={flip_rates.get(f'L{sample_indices[len(sample_indices)//2]}', {}).get('t0.0_max_flip', 'N/A')}, "
              f"spikiness={flip_rates.get(f'L{sample_indices[len(sample_indices)//2]}', {}).get('t0.0_flip_spikiness', 'N/A')}")

    # === 聚合结果 ===
    print("\n  聚合结果...")
    agg = _aggregate_flip_results(results["per_pair"], sample_indices, thresholds)
    results["aggregated"] = agg

    return results


def _aggregate_flip_results(per_pair, sample_indices, thresholds):
    """聚合所有句子对的翻转率"""
    agg = {}

    for li in sample_indices:
        lk = f"L{li}"
        layer_agg = {}

        for t in thresholds:
            # 收集所有pair的flip series
            all_flips_all = []
            all_flips_diff = []
            for pair_data in per_pair:
                fr = pair_data.get("flip_rates", {}).get(lk, {})
                if f"t{t}_all_pos" in fr:
                    all_flips_all.append(fr[f"t{t}_all_pos"])
                if f"t{t}_diff_pos" in fr:
                    all_flips_diff.append(fr[f"t{t}_diff_pos"])

            if all_flips_all:
                arr = np.array(all_flips_all)
                layer_agg[f"t{t}_mean_over_pairs"] = np.mean(arr, axis=0).tolist()
                layer_agg[f"t{t}_max_flip_mean"] = float(np.mean([np.max(a) for a in all_flips_all]))
                layer_agg[f"t{t}_mean_flip_mean"] = float(np.mean([np.mean(a) for a in all_flips_all]))
                layer_agg[f"t{t}_spikiness_mean"] = float(np.mean([
                    np.max(a) / max(np.mean(a), 1e-10) for a in all_flips_all
                ]))
                # 总相变点数
                total_spikes = sum(fr.get(f"t{t}_n_spikes", 0)
                                   for pair_data in per_pair
                                   for fr in [pair_data.get("flip_rates", {}).get(lk, {})])
                layer_agg[f"t{t}_total_spikes"] = total_spikes

            if all_flips_diff:
                arr = np.array(all_flips_diff)
                layer_agg[f"t{t}_diff_max_flip_mean"] = float(np.mean([np.max(a) for a in all_flips_diff]))
                layer_agg[f"t{t}_diff_mean_flip_mean"] = float(np.mean([np.mean(a) for a in all_flips_diff]))

        # Attention
        all_attn_switch = []
        for pair_data in per_pair:
            fr = pair_data.get("flip_rates", {}).get(lk, {})
            if "attn_switch" in fr:
                all_attn_switch.append(fr["attn_switch"])

        if all_attn_switch:
            arr = np.array(all_attn_switch)
            layer_agg["attn_mean_over_pairs"] = np.mean(arr, axis=0).tolist()
            layer_agg["attn_max_switch_mean"] = float(np.mean([np.max(a) for a in all_attn_switch]))
            layer_agg["attn_mean_switch_mean"] = float(np.mean([np.mean(a) for a in all_attn_switch]))
            layer_agg["attn_spikiness_mean"] = float(np.mean([
                np.max(a) / max(np.mean(a), 1e-10) for a in all_attn_switch
            ]))

        agg[lk] = layer_agg

    return agg


# ============================================================
# Exp 2: 边界密度
# ============================================================

def exp2_boundary_density(model, tokenizer, device, model_info, model_name: str):
    """
    测量各层的"边界粗糙度"

    方法:
    1. 对一个句子运行模型
    2. 在embedding上添加随机微扰 x+εv
    3. 统计: 多少ε会导致activation flip
    4. 定义边界密度 ρ_l = P(boundary crossing at layer l)
    """

    n_layers = model_info.n_layers
    d_model = model_info.d_model
    intermediate_size = model_info.intermediate_size
    mlp_type = model_info.mlp_type
    layers = get_layers(model)

    sample_indices = sorted(set(
        [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    ))

    n_heads = 32
    try:
        n_heads = layers[0].self_attn.num_heads
    except:
        pass

    # 测试句子
    test_sentences = [
        "The weather is good",
        "The dog always bites the man",
        "The man is tall and strong",
        "She reads the book today",
        "The fire burns the forest",
    ]

    # 微扰参数
    epsilons = [0.01, 0.05, 0.1, 0.5, 1.0]
    n_random_dirs = 20  # 每个ε的随机方向数
    threshold = 0.0

    embed_layer = model.get_input_embeddings()

    results = {
        "n_sentences": len(test_sentences),
        "epsilons": epsilons,
        "n_random_dirs": n_random_dirs,
        "per_sentence": [],
        "aggregated": {},
    }

    for sent_idx, sentence in enumerate(test_sentences):
        print(f"\n  句子 {sent_idx+1}: '{sentence}'")

        ids = tokenizer.encode(sentence, add_special_tokens=False)
        input_ids = torch.tensor([ids], device=device)
        seq_len = len(ids)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        base_emb = embed_layer(input_ids).detach().clone()  # [1, seq_len, d_model]

        # 运行base获得参考激活
        base_mlp_masks, base_attn_topk = _run_and_capture(
            model, layers, sample_indices, n_heads, base_emb, position_ids,
            mlp_type, threshold, 5, seq_len
        )

        sent_result = {
            "sentence": sentence,
            "per_epsilon": {},
        }

        for eps in epsilons:
            flip_counts = {li: 0 for li in sample_indices}
            attn_switch_counts = {li: 0 for li in sample_indices}
            n_runs = 0

            for dir_idx in range(n_random_dirs):
                # 随机方向
                v = torch.randn_like(base_emb)
                v = v / v.norm(dim=-1, keepdim=True)  # 归一化

                # 微扰
                perturbed_emb = base_emb + eps * v

                perturbed_mlp_masks, perturbed_attn_topk = _run_and_capture(
                    model, layers, sample_indices, n_heads, perturbed_emb, position_ids,
                    mlp_type, threshold, 5, seq_len
                )

                if perturbed_mlp_masks is None:
                    continue

                n_runs += 1

                # 比较MLP masks
                for li in sample_indices:
                    lk = f"L{li}"
                    if lk in base_mlp_masks and lk in perturbed_mlp_masks:
                        base_mask = base_mlp_masks[lk]
                        pert_mask = perturbed_mlp_masks[lk]
                        flip_rate = np.mean(np.abs(pert_mask.astype(np.float32) - base_mask.astype(np.float32)))
                        if flip_rate > 0.01:  # 超过1%的神经元翻转
                            flip_counts[li] += 1

                # 比较attention top-k
                for li in sample_indices:
                    lk = f"L{li}"
                    if lk in base_attn_topk and lk in perturbed_attn_topk:
                        base_topk = base_attn_topk[lk]
                        pert_topk = perturbed_attn_topk[lk]
                        head_jacs = []
                        for h in range(min(n_heads, len(base_topk), len(pert_topk))):
                            e1 = base_topk[h]
                            e2 = pert_topk[h]
                            union = len(e1 | e2)
                            if union > 0:
                                head_jacs.append(len(e1 & e2) / union)
                        if head_jacs:
                            mean_jac = np.mean(head_jacs)
                            if mean_jac < 0.9:  # 10%以上的边改变
                                attn_switch_counts[li] += 1

            # 计算边界密度
            eps_result = {}
            for li in sample_indices:
                lk = f"L{li}"
                eps_result[lk] = {
                    "mlp_boundary_density": flip_counts[li] / max(n_runs, 1),
                    "attn_boundary_density": attn_switch_counts[li] / max(n_runs, 1),
                    "n_runs": n_runs,
                }

            sent_result["per_epsilon"][str(eps)] = eps_result
            print(f"    eps={eps}: MLP boundary density = "
                  f"{[f'L{li}:{flip_counts[li]/max(n_runs,1):.3f}' for li in sample_indices]}")

        results["per_sentence"].append(sent_result)

    # 聚合
    agg = {}
    for li in sample_indices:
        lk = f"L{li}"
        layer_agg = {}
        for eps in epsilons:
            mlp_densities = []
            attn_densities = []
            for sent_res in results["per_sentence"]:
                eps_data = sent_res["per_epsilon"].get(str(eps), {})
                if lk in eps_data:
                    mlp_densities.append(eps_data[lk]["mlp_boundary_density"])
                    attn_densities.append(eps_data[lk]["attn_boundary_density"])
            if mlp_densities:
                layer_agg[f"eps{eps}_mlp_density_mean"] = float(np.mean(mlp_densities))
                layer_agg[f"eps{eps}_attn_density_mean"] = float(np.mean(attn_densities))
        agg[lk] = layer_agg

    results["aggregated"] = agg
    return results


def _run_and_capture(model, layers, sample_indices, n_heads,
                     inputs_embeds, position_ids, mlp_type, threshold,
                     top_k_attn, seq_len):
    """运行模型并捕获MLP激活和attention top-k"""
    captured_mlp = {}
    captured_attn = {}

    def make_mlp_hook(key, mlp_type_local):
        def hook(module, input, output):
            if mlp_type_local == "merged_gate_up":
                gate = output[:, :, :output.shape[-1]//2]
                captured_mlp[key] = gate.detach().float().cpu()
            else:
                captured_mlp[key] = output.detach().float().cpu()
        return hook

    def make_attn_hook(key):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) > 1:
                attn = output[1]
                if attn is not None:
                    captured_attn[key] = attn.detach().float().cpu()
        return hook

    hooks = []
    for li in sample_indices:
        layer = layers[li]
        mlp = layer.mlp if hasattr(layer, "mlp") else None
        if mlp is not None:
            if mlp_type == "merged_gate_up":
                hooks.append(mlp.gate_up_proj.register_forward_hook(
                    make_mlp_hook(f"L{li}", mlp_type)))
            elif hasattr(mlp, 'gate_proj'):
                hooks.append(mlp.gate_proj.register_forward_hook(
                    make_mlp_hook(f"L{li}", mlp_type)))
            elif hasattr(mlp, 'up_proj'):
                hooks.append(mlp.up_proj.register_forward_hook(
                    make_mlp_hook(f"L{li}", mlp_type)))
        if hasattr(layer, 'self_attn'):
            hooks.append(layer.self_attn.register_forward_hook(
                make_attn_hook(f"L{li}")))

    try:
        with torch.no_grad():
            _ = model(inputs_embeds=inputs_embeds.to(model.dtype),
                     position_ids=position_ids,
                     output_attentions=True)
    except Exception as e:
        for h in hooks:
            h.remove()
        return None, None

    for h in hooks:
        h.remove()

    # 处理MLP masks
    mlp_masks = {}
    for li in sample_indices:
        key = f"L{li}"
        if key in captured_mlp:
            mlp_out = captured_mlp[key][0].numpy()
            mlp_masks[key] = (mlp_out > threshold).astype(np.int8)

    # 处理attention top-k
    attn_topk = {}
    for li in sample_indices:
        key = f"L{li}"
        if key in captured_attn:
            attn = captured_attn[key]
            if attn.dim() == 4:
                attn_np = attn[0].float().numpy()
            else:
                continue

            head_topk = []
            for h in range(min(n_heads, attn_np.shape[0])):
                attn_h = attn_np[h]
                flat = attn_h.flatten()
                k = min(top_k_attn, len(flat))
                top_indices = np.argpartition(flat, -k)[-k:]
                top_indices = top_indices[np.argsort(flat[top_indices])[::-1]]
                edges = set()
                for idx in top_indices:
                    i = int(idx // seq_len)
                    j = int(idx % seq_len)
                    edges.add((i, j))
                head_topk.append(edges)
            attn_topk[key] = head_topk

    del captured_mlp, captured_attn
    return mlp_masks, attn_topk


# ============================================================
# Exp 3: 否定 vs 语义插值对比
# ============================================================

def exp3_comparison(model, tokenizer, device, model_info, model_name: str):
    """
    对比否定插值和语义插值的相变特征

    核心问题: 否定是否比语义变化产生更多/更尖锐的相变?
    """

    # 简化版: 只比较aggregated flip rates
    # 在exp1中已经分别运行了语义和否定插值
    # 这里返回一个对比摘要
    return {
        "note": "对比在exp1的semantic vs negation结果中直接进行",
        "hypothesis": "否定插值应该在中间层产生更尖锐的相变 (因为否定是语法操作, 中间层是句法计算区)",
    }


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
    print(f"Phase 136: 激活边界分析")
    print(f"Model: {model_name}")
    print(f"=" * 60)

    # 加载模型
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

    # Exp 1: 输入插值路径
    print(f"\n{'='*60}")
    print("Exp 1: 输入插值路径 — 激活翻转率")
    print(f"{'='*60}")
    exp1_results = exp1_interpolation_path(model, tokenizer, device, model_info, model_name)
    all_results["exp1_interpolation"] = exp1_results

    # Exp 2: 边界密度
    print(f"\n{'='*60}")
    print("Exp 2: 边界密度 — 微扰敏感性")
    print(f"{'='*60}")
    exp2_results = exp2_boundary_density(model, tokenizer, device, model_info, model_name)
    all_results["exp2_boundary_density"] = exp2_results

    # Exp 3: 对比
    exp3_results = exp3_comparison(model, tokenizer, device, model_info, model_name)
    all_results["exp3_comparison"] = exp3_results

    total_time = time.time() - t0
    all_results["total_time_seconds"] = round(total_time, 1)

    # 保存结果
    output_path = f"tests/glm5_temp/phase136_{model_name}_activation_boundary.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 简化结果以避免JSON过大
    simplified = _simplify_results(all_results)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(simplified, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {output_path}")

    # 打印关键结果
    _print_key_results(simplified, model_info)

    # 释放模型
    release_model(model)


def _simplify_results(results):
    """简化结果以避免JSON过大 — 只保留聚合和关键统计"""
    simplified = {
        "model_info": results["model_info"],
        "timestamp": results["timestamp"],
        "total_time_seconds": results.get("total_time_seconds"),
    }

    # Exp 1: 保留聚合 + 每个pair的摘要
    if "exp1_interpolation" in results:
        exp1 = results["exp1_interpolation"]
        simplified_exp1 = {}
        for pair_type in ["semantic", "negation"]:
            if pair_type in exp1:
                data = exp1[pair_type]
                simplified_exp1[pair_type] = {
                    "n_pairs": data["n_pairs"],
                    "n_interp": data["n_interp"],
                    "aggregated": data.get("aggregated", {}),
                    # 只保留每个pair的flip rate摘要, 不保存完整mask
                    "per_pair_summary": [],
                }
                for pair_data in data.get("per_pair", []):
                    summary = {
                        "sent1": pair_data.get("sent1"),
                        "sent2": pair_data.get("sent2"),
                        "diff_pos": pair_data.get("diff_pos"),
                        "flip_rates_summary": {},
                    }
                    fr = pair_data.get("flip_rates", {})
                    for lk, layer_data in fr.items():
                        layer_summary = {}
                        for key in ["t0.0_max_flip", "t0.0_mean_flip", "t0.0_flip_spikiness",
                                    "t0.0_n_spikes", "t0.0_diff_max_flip_mean",
                                    "t0.1_max_flip", "t0.1_mean_flip", "t0.1_flip_spikiness",
                                    "t0.5_max_flip", "t0.5_mean_flip", "t0.5_flip_spikiness",
                                    "attn_max_switch", "attn_mean_switch", "attn_switch_spikiness"]:
                            if key in layer_data:
                                layer_summary[key] = layer_data[key]
                        # 保留flip series用于画图 (只保留t0.0和attn)
                        for key in ["t0.0_all_pos", "t0.0_diff_pos", "attn_switch"]:
                            if key in layer_data:
                                layer_summary[key] = layer_data[key]
                        layer_summary["spike_positions"] = layer_data.get("t0.0_spike_positions", [])
                        if fr.get(lk, {}).get("t0.0_spike_positions"):
                            layer_summary["spike_positions"] = fr[lk]["t0.0_spike_positions"]
                        layer_summary["t0.0_spike_positions"] = layer_data.get("t0.0_spike_positions", [])

                        if layer_summary:
                            summary["flip_rates_summary"][lk] = layer_summary
                    simplified_exp1[pair_type]["per_pair_summary"].append(summary)
        simplified["exp1_interpolation"] = simplified_exp1

    # Exp 2: 保留聚合
    if "exp2_boundary_density" in results:
        exp2 = results["exp2_boundary_density"]
        simplified["exp2_boundary_density"] = {
            "n_sentences": exp2["n_sentences"],
            "epsilons": exp2["epsilons"],
            "aggregated": exp2.get("aggregated", {}),
        }

    # Exp 3
    if "exp3_comparison" in results:
        simplified["exp3_comparison"] = results["exp3_comparison"]

    return simplified


def _print_key_results(results, model_info):
    """打印关键结果"""
    print(f"\n{'='*60}")
    print("Phase 136 关键结果")
    print(f"{'='*60}")

    # Exp 1: 插值路径
    exp1 = results.get("exp1_interpolation", {})
    for pair_type in ["semantic", "negation"]:
        data = exp1.get(pair_type, {})
        agg = data.get("aggregated", {})
        print(f"\n--- {pair_type} 插值 ---")
        print(f"  句子对数: {data.get('n_pairs', 'N/A')}")
        for lk in sorted(agg.keys()):
            layer_data = agg[lk]
            max_flip = layer_data.get("t0.0_max_flip_mean", "N/A")
            mean_flip = layer_data.get("t0.0_mean_flip_mean", "N/A")
            spikiness = layer_data.get("t0.0_spikiness_mean", "N/A")
            total_spikes = layer_data.get("t0.0_total_spikes", "N/A")
            diff_max = layer_data.get("t0.0_diff_max_flip_mean", "N/A")
            attn_switch = layer_data.get("attn_max_switch_mean", "N/A")
            print(f"  {lk}: max_flip={max_flip:.4f}, mean_flip={mean_flip:.4f}, "
                  f"spikiness={spikiness:.2f}, spikes={total_spikes}, "
                  f"diff_max={diff_max:.4f}, attn_switch={attn_switch:.4f}")

    # Exp 2: 边界密度
    exp2 = results.get("exp2_boundary_density", {})
    agg2 = exp2.get("aggregated", {})
    print(f"\n--- 边界密度 ---")
    for eps in [0.01, 0.1, 1.0]:
        print(f"  eps={eps}:")
        for lk in sorted(agg2.keys()):
            layer_data = agg2[lk]
            mlp_key = f"eps{eps}_mlp_density_mean"
            attn_key = f"eps{eps}_attn_density_mean"
            if mlp_key in layer_data:
                print(f"    {lk}: MLP={layer_data[mlp_key]:.4f}, Attn={layer_data.get(attn_key, 'N/A')}")


if __name__ == "__main__":
    main()
