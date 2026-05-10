"""
Phase 103: 动力系统分析 — 从静态几何到局部动力学
====================================================

Phase 102的硬伤 (用户批判):
  1. "干预失败⇒不存在翻译方向" — 犯了线性因果假设错误
     翻译方向可能存在但需要gating/routing才能激活
  2. "翻译是最后阶段局部过程" — 混淆readability emergence与computation emergence
     早期层可能已完成计算，只是decoder读不出
  3. "低秩=只有10维计算" — dominant variance≠computational degrees of freedom
     小方差方向可能承载关键控制信号
  4. 仍在空间静态化 — 把Transformer当点+方向，而非条件依赖动力系统

核心方法论升级:
  从"观察几何结构"到"分析局部动力学"
  从"全局方向"到"Jacobian谱"
  从"加法干预"到"乘法/门控干预"
  从"单一测量"到"测量可信度审计"

实验设计:
  Exp 1: Jacobian谱分析 — 局部动力学的核心
    1a: 层间Jacobian ∂h_{l+1}/∂h_l 的谱估计
    1b: 翻译vs非翻译上下文的Jacobian差异
    1c: 哪些方向被放大/压缩/旋转?

  Exp 2: 低方差高影响方向 — 验证"dominant variance≠computational DOF"
    2a: Δh SVD的小奇异值方向的因果影响测试
    2b: 对比高方差方向vs低方差方向的因果影响力

  Exp 3: 乘法/门控干预 — 翻译需要gating而非加法
    3a: 乘法干预 h' = h * (1 + α*M) vs 加法干预 h' = h + α*v
    3b: LayerNorm后干预 (让方向通过归一化门控)
    3c: 注意力模式干预

  Exp 4: 上下文插值与吸引子盆地 — 相变分析
    4a: 从中文补全到翻译prompt的平滑插值
    4b: 输出分布的相变点检测
    4c: 吸引子边界映射

Run:
  python tests/glm5/ccml_phase103_dynamical_systems.py --model qwen3 --exp 1
  python tests/glm5/ccml_phase103_dynamical_systems.py --model qwen3 --exp 2
  python tests/glm5/ccml_phase103_dynamical_systems.py --model qwen3 --exp 3
  python tests/glm5/ccml_phase103_dynamical_systems.py --model qwen3 --exp 4
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
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


# ============================================================
# 测试数据
# ============================================================
TRANSLATION_PAIRS = [
    ("猫", "cat"), ("狗", "dog"), ("水", "water"), ("火", "fire"),
    ("树", "tree"), ("花", "flower"), ("鱼", "fish"), ("鸟", "bird"),
    ("铁", "iron"), ("茶", "tea"), ("马", "horse"), ("金", "gold"),
    ("雪", "snow"), ("月", "moon"), ("光", "light"), ("梦", "dream"),
    ("龙", "dragon"), ("云", "cloud"), ("冰", "ice"), ("星", "star"),
]


def get_token_prob_from_logits(logits, tokenizer, text):
    """获取特定token的概率"""
    tok_ids = tokenizer.encode(text, add_special_tokens=False)
    if not tok_ids:
        return 0.0
    probs = torch.softmax(logits, dim=-1)
    return probs[tok_ids[0]].item()


def get_top_k_from_logits(logits, tokenizer, k=10):
    """获取top-k token"""
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
# Exp 1: Jacobian谱分析
# ============================================================
def exp1_jacobian_spectrum(model_name):
    """
    Jacobian谱分析 — 局部动力学的核心

    核心思路:
    - Transformer层l做的是: h_{l+1} = F_l(h_l, context)
    - Jacobian J_l = ∂h_{l+1}/∂h_l 描述了局部动力学
    - J的谱告诉我们: 哪些方向被放大、压缩、旋转
    - 翻译vs非翻译的Jacobian差异 → 翻译计算的具体动力学

    实现方式:
    - 数值差分: J@v ≈ (F(h + ε*v) - F(h)) / ε
    - 随机探针: 用50个随机方向v估计J的谱
    - 逐层分析: 每3层计算一次Jacobian谱
    """
    print(f"\n{'='*70}")
    print(f"Exp 1: Jacobian谱分析 — {model_name}")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  模型: {model_name}, 层数: {n_layers}, d_model: {d_model}")

    results = {}

    # 测试词 (减少到8个以控制运行时间)
    test_pairs = [("猫", "cat"), ("水", "water"), ("火", "fire"), ("树", "tree"),
                  ("花", "flower"), ("铁", "iron"), ("龙", "dragon"), ("光", "light")]

    n_probes = 50  # 随机探针数量
    eps = 1.0      # 扰动幅度

    # 采样层
    sample_layers = list(range(0, n_layers, 3))  # L0, L3, L6, ...
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)

    jacobian_results = {}

    for zh, en in test_pairs:
        print(f"\n  === 处理: {zh}({en}) ===")
        word_jacobian = {}

        for task_name, prompt in [
            ("zh_continue", f"{zh}是一种"),
            ("translate", f'请把"{zh}"翻译成英文：'),
        ]:
            print(f"    任务: {task_name}")
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]

            # Baseline: 获取所有层的hidden state
            with torch.no_grad():
                base_outputs = model(input_ids, output_hidden_states=True)

            base_hiddens = {}
            for l in range(n_layers + 1):
                base_hiddens[l] = base_outputs.hidden_states[l][0, -1, :].detach().clone().float()

            task_jacobian = {}

            for l in sample_layers:
                if l >= n_layers:
                    continue

                print(f"      L{l}...", end="", flush=True)
                h_l_baseline = base_hiddens[l]  # [d_model]
                h_l_plus1_baseline = base_hiddens[l + 1]  # [d_model]
                h_l_norm = float(h_l_baseline.norm())

                # 扰动幅度与hidden state范数成比例
                eps_scaled = max(0.01 * h_l_norm, 1.0)

                # 生成随机探针
                torch.manual_seed(42)  # 可重复
                probe_vectors = torch.randn(n_probes, d_model, device=device)
                probe_vectors = F.normalize(probe_vectors, dim=1)

                # 对每个探针，计算J@v ≈ (h_{l+1}(h_l + ε*v) - h_{l+1}(h_l)) / ε
                jacobian_probes = []  # J@v_i 的结果
                logit_probes = []     # 对最终logits的影响

                layers_list = get_layers(model)

                for p_idx in range(n_probes):
                    probe = probe_vectors[p_idx]  # [d_model]

                    # Hook: 在层l的输出处添加扰动
                    perturbed_h_l_plus1 = [None]
                    intervened = [False]

                    def make_hook(eps_val, probe_vec, captured, intervened_flag):
                        def hook_fn(module, input, output):
                            if not intervened_flag[0]:
                                if isinstance(output, tuple):
                                    hidden_states = output[0].clone()
                                    # 只扰动最后token位置
                                    hidden_states[:, -1, :] += eps_val * probe_vec.to(hidden_states.dtype).to(device)
                                    captured[0] = hidden_states[0, -1, :].detach().float().cpu().numpy()
                                    output = (hidden_states,) + output[1:]
                                intervened_flag[0] = True
                            return output
                        return hook_fn

                    handle = layers_list[l].register_forward_hook(
                        make_hook(eps_scaled, probe, perturbed_h_l_plus1, intervened)
                    )

                    with torch.no_grad():
                        perturbed_outputs = model(input_ids, output_hidden_states=True)

                    handle.remove()

                    # h_{l+1}的变化
                    if perturbed_h_l_plus1[0] is not None:
                        delta_h_l_plus1 = perturbed_h_l_plus1[0] - h_l_plus1_baseline.cpu().numpy()
                        jacobian_probes.append(delta_h_l_plus1 / eps_scaled)

                    # 最终logits的变化 (用于分析层l对输出的因果影响)
                    perturbed_logits = perturbed_outputs.logits[0, -1, :].float().cpu().numpy()
                    base_logits = base_outputs.logits[0, -1, :].float().cpu().numpy()
                    delta_logits = perturbed_logits - base_logits
                    logit_probes.append(delta_logits)

                # 分析Jacobian探针结果
                if jacobian_probes:
                    J_probes = np.array(jacobian_probes)  # [n_probes, d_model]

                    # 1. 每个探针引起的h_{l+1}变化范数 → 告诉我们放大/压缩
                    change_norms = np.linalg.norm(J_probes, axis=1)
                    mean_amplification = float(np.mean(change_norms))
                    max_amplification = float(np.max(change_norms))
                    min_amplification = float(np.min(change_norms))

                    # 2. J_probes的SVD → 近似Jacobian的奇异值谱
                    # [n_probes, d_model]的SVD给出最多n_probes个奇异值
                    U_j, S_j, Vt_j = np.linalg.svd(J_probes, full_matrices=False)

                    # 3. Jacobian的有效秩
                    total_energy = np.sum(S_j**2)
                    if total_energy > 0:
                        cumvar = np.cumsum(S_j**2) / total_energy
                        effective_rank_90 = int(np.searchsorted(cumvar, 0.9)) + 1
                        effective_rank_99 = int(np.searchsorted(cumvar, 0.99)) + 1
                    else:
                        effective_rank_90 = 0
                        effective_rank_99 = 0

                    # 4. 对最终logits的影响
                    logit_changes = np.array(logit_probes)
                    logit_change_norms = np.linalg.norm(logit_changes, axis=1)
                    mean_logit_impact = float(np.mean(logit_change_norms))

                    # 5. 特定token概率变化
                    en_tok_ids = tokenizer.encode(en, add_special_tokens=False)
                    zh_tok_ids = tokenizer.encode(zh, add_special_tokens=False)

                    en_prob_changes = []
                    zh_prob_changes = []
                    for delta_logits in logit_probes:
                        if en_tok_ids:
                            en_prob_changes.append(delta_logits[en_tok_ids[0]])
                        if zh_tok_ids:
                            zh_prob_changes.append(delta_logits[zh_tok_ids[0]])

                    task_jacobian[str(l)] = {
                        "h_l_norm": float(h_l_norm),
                        "eps_scaled": float(eps_scaled),
                        "mean_amplification": mean_amplification,
                        "max_amplification": max_amplification,
                        "min_amplification": min_amplification,
                        "effective_rank_90": effective_rank_90,
                        "effective_rank_99": effective_rank_99,
                        "top10_singular_values": [float(s) for s in S_j[:10]],
                        "mean_logit_impact": mean_logit_impact,
                        "en_logit_sensitivity": float(np.mean(np.abs(en_prob_changes))) if en_prob_changes else 0,
                        "zh_logit_sensitivity": float(np.mean(np.abs(zh_prob_changes))) if zh_prob_changes else 0,
                    }

                    print(f" amp={mean_amplification:.2f}, rank90={effective_rank_90}, "
                          f"logit_impact={mean_logit_impact:.4f}, "
                          f"en_sens={task_jacobian[str(l)]['en_logit_sensitivity']:.6f}")
                else:
                    print(f" FAILED")

            word_jacobian[task_name] = task_jacobian

        jacobian_results[f"{zh}_{en}"] = word_jacobian

    # ---- 跨词聚合 ----
    print(f"\n  === 跨词聚合Jacobian分析 ===")

    aggregate = {"zh_continue": {}, "translate": {}}
    for task_name in ["zh_continue", "translate"]:
        for l in sample_layers:
            l_str = str(l)
            amps = []
            ranks = []
            logit_impacts = []
            en_sens = []
            zh_sens = []

            for word_key, word_data in jacobian_results.items():
                if task_name in word_data and l_str in word_data[task_name]:
                    d = word_data[task_name][l_str]
                    amps.append(d["mean_amplification"])
                    ranks.append(d["effective_rank_90"])
                    logit_impacts.append(d["mean_logit_impact"])
                    en_sens.append(d["en_logit_sensitivity"])
                    zh_sens.append(d["zh_logit_sensitivity"])

            if amps:
                aggregate[task_name][l_str] = {
                    "mean_amplification": float(np.mean(amps)),
                    "mean_rank_90": float(np.mean(ranks)),
                    "mean_logit_impact": float(np.mean(logit_impacts)),
                    "mean_en_sensitivity": float(np.mean(en_sens)),
                    "mean_zh_sensitivity": float(np.mean(zh_sens)),
                }

    # 打印聚合结果
    print(f"\n  层  | zh_amp | trans_amp | zh_rank | trans_rank | zh_logit | trans_logit | en_sens_zh | en_sens_trans")
    print(f"  {'─'*100}")
    for l in sample_layers:
        l_str = str(l)
        zh_d = aggregate["zh_continue"].get(l_str, {})
        tr_d = aggregate["translate"].get(l_str, {})
        if zh_d and tr_d:
            print(f"  L{l:2d} | {zh_d['mean_amplification']:7.2f} | {tr_d['mean_amplification']:9.2f} | "
                  f"{zh_d['mean_rank_90']:7.1f} | {tr_d['mean_rank_90']:10.1f} | "
                  f"{zh_d['mean_logit_impact']:8.4f} | {tr_d['mean_logit_impact']:11.4f} | "
                  f"{zh_d['mean_en_sensitivity']:10.6f} | {tr_d['mean_en_sensitivity']:14.6f}")

    # ---- 翻译vs中文的Jacobian差异 ----
    print(f"\n  === 翻译特异性: Jacobian差异最大的层 ===")
    diff_layers = []
    for l in sample_layers:
        l_str = str(l)
        zh_d = aggregate["zh_continue"].get(l_str, {})
        tr_d = aggregate["translate"].get(l_str, {})
        if zh_d and tr_d:
            # 多维度差异
            amp_diff = abs(tr_d["mean_amplification"] - zh_d["mean_amplification"])
            rank_diff = abs(tr_d["mean_rank_90"] - zh_d["mean_rank_90"])
            en_sens_ratio = tr_d["mean_en_sensitivity"] / max(zh_d["mean_en_sensitivity"], 1e-10)
            logit_diff = abs(tr_d["mean_logit_impact"] - zh_d["mean_logit_impact"])

            diff_layers.append((l, amp_diff, rank_diff, en_sens_ratio, logit_diff))

    # 按en_sensitivity_ratio排序 (翻译上下文对英文token更敏感)
    diff_layers_sorted = sorted(diff_layers, key=lambda x: x[3], reverse=True)
    print(f"  按翻译en_sensitivity_ratio排序 (翻译上下文更敏感的层):")
    for l, amp_d, rank_d, en_ratio, logit_d in diff_layers_sorted[:8]:
        print(f"    L{l}: en_sens_ratio={en_ratio:.2f}, amp_diff={amp_d:.2f}, rank_diff={rank_d:.1f}, logit_diff={logit_d:.4f}")

    results["jacobian_by_word"] = jacobian_results
    results["aggregate"] = aggregate

    # 保存
    save_path = f"tests/glm5_temp/phase103_exp1_{model_name}_jacobian_spectrum.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  保存到: {save_path}")

    release_model(model)
    return results


# ============================================================
# Exp 2: 低方差高影响方向
# ============================================================
def exp2_low_variance_high_impact(model_name):
    """
    低方差高影响方向 — 验证"dominant variance ≠ computational DOF"

    核心思路:
    - Phase 102发现Δh的SVD中，前10维占90%方差
    - 但小方差方向可能承载关键控制信号(路由比特、稀疏选择器)
    - 本实验: 对比高方差方向vs低方差方向的因果影响力

    方法:
    - 从Δh的SVD中，取top-10 (高方差) 和 bottom-100 (低方差) 方向
    - 沿每个方向添加扰动，测量最终输出的变化
    - 如果低方差方向的因果影响力 > 高方差方向，则"dominant variance ≠ DOF"
    """
    print(f"\n{'='*70}")
    print(f"Exp 2: 低方差高影响方向 — {model_name}")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  模型: {model_name}, 层数: {n_layers}, d_model: {d_model}")

    results = {}

    test_pairs = [("猫", "cat"), ("水", "water"), ("火", "fire"), ("树", "tree"),
                  ("龙", "dragon"), ("光", "light"), ("冰", "ice"), ("梦", "dream")]

    # ---- Step 1: 收集Δh并做SVD ----
    print(f"\n  === Step 1: 收集Δh并做SVD ===")

    # 用翻译prompt的Δh
    all_delta_h = []  # [n_words * n_layers, d_model]

    for zh, en in test_pairs:
        trans_prompt = f'请把"{zh}"翻译成英文：'
        inputs = tokenizer(trans_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)

        for l in range(n_layers):
            h_l = outputs.hidden_states[l][0, -1, :].float().cpu().numpy()
            h_l1 = outputs.hidden_states[l + 1][0, -1, :].float().cpu().numpy()
            all_delta_h.append(h_l1 - h_l)

    all_delta_h = np.array(all_delta_h)  # [n_samples, d_model]
    print(f"    Δh矩阵: {all_delta_h.shape}")

    # SVD
    U_dh, S_dh, Vt_dh = np.linalg.svd(all_delta_h, full_matrices=False)
    # Vt_dh: [min(n_samples, d_model), d_model] — 每行是一个奇异向量方向

    total_var = np.sum(S_dh**2)
    cumvar = np.cumsum(S_dh**2) / total_var

    print(f"    Top10 奇异值: {[f'{s:.1f}' for s in S_dh[:10]]}")
    print(f"    Top10 cumvar: {[f'{c:.3f}' for c in cumvar[:10]]}")

    # ---- Step 2: 测试不同方差方向的因果影响 ----
    print(f"\n  === Step 2: 方差方向因果影响测试 ===")

    # 选择测试层 (基于Phase 102发现的关键层)
    test_layers = [9, 21, 27, 33]
    test_layers = [l for l in test_layers if l < n_layers]

    # 扰动幅度 (与hidden state范数成比例)
    eps_base = 5.0  # 固定扰动大小

    # 方向类别:
    # top: 前5个奇异向量 (高方差)
    # mid: 第20-25个奇异向量 (中等方差)
    # low: 第80-85个奇异向量 (低方差)
    # random: 随机方向 (对照)

    direction_groups = {
        "top5": list(range(0, 5)),
        "mid20_25": list(range(20, 25)),
        "low80_85": list(range(80, min(85, len(S_dh)))),
        "random": None,  # 稍后生成
    }

    causal_impact = {}

    for zh, en in test_pairs[:4]:  # 减少到4个词
        print(f"\n  处理: {zh}({en})")
        word_impact = {}

        trans_prompt = f'请把"{zh}"翻译成英文：'
        inputs = tokenizer(trans_prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]

        # Baseline logits
        with torch.no_grad():
            base_logits = model(input_ids).logits[0, -1, :].float().cpu()

        base_en_prob = get_token_prob_from_logits(base_logits, tokenizer, en)
        base_top5 = get_top_k_from_logits(base_logits, tokenizer, 5)

        for l in test_layers:
            layer_impact = {}
            layers_list = get_layers(model)

            for group_name, indices in direction_groups.items():
                if indices is None:
                    # 随机方向
                    torch.manual_seed(42 + l)
                    dirs = torch.randn(5, d_model, device=device)
                    dirs = F.normalize(dirs, dim=1)
                    label = "random"
                else:
                    if max(indices) >= len(S_dh):
                        continue
                    dirs = torch.tensor(Vt_dh[indices], dtype=torch.float32, device=device)
                    dirs = F.normalize(dirs, dim=1)
                    label = group_name

                # 对每个方向测试因果影响
                en_prob_changes = []
                kl_divergences = []

                for d_idx in range(dirs.shape[0]):
                    direction = dirs[d_idx]

                    # Hook: 在层l添加扰动
                    intervened = [False]

                    def make_hook(eps_val, dir_vec, flag):
                        def hook_fn(module, input, output):
                            if not flag[0]:
                                if isinstance(output, tuple):
                                    hidden_states = output[0].clone()
                                    hidden_states[:, -1, :] += eps_val * dir_vec.to(hidden_states.dtype).to(device)
                                    output = (hidden_states,) + output[1:]
                                flag[0] = True
                            return output
                        return hook_fn

                    handle = layers_list[l].register_forward_hook(make_hook(eps_base, direction, intervened))

                    with torch.no_grad():
                        perturbed_logits = model(input_ids).logits[0, -1, :].float().cpu()

                    handle.remove()

                    # 因果影响度量
                    perturbed_en_prob = get_token_prob_from_logits(perturbed_logits, tokenizer, en)
                    en_prob_changes.append(perturbed_en_prob - base_en_prob)

                    # KL散度 (整体输出分布变化)
                    base_probs = torch.softmax(base_logits, dim=-1)
                    pert_probs = torch.softmax(perturbed_logits, dim=-1)
                    kl = torch.sum(base_probs * (torch.log(base_probs + 1e-10) - torch.log(pert_probs + 1e-10)))
                    kl_divergences.append(float(kl))

                layer_impact[label] = {
                    "mean_en_prob_change": float(np.mean(en_prob_changes)),
                    "mean_kl_divergence": float(np.mean(kl_divergences)),
                    "max_kl_divergence": float(np.max(kl_divergences)),
                }

                print(f"    L{l} {label:10s}: ΔP(en)={np.mean(en_prob_changes):.6f}, "
                      f"KL={np.mean(kl_divergences):.6f}")

            word_impact[str(l)] = layer_impact

        causal_impact[f"{zh}_{en}"] = word_impact

    # ---- Step 3: 关键对比 ----
    print(f"\n  === Step 3: 高方差vs低方差方向因果影响对比 ===")

    # 跨词聚合
    aggregate_impact = {}
    for l in test_layers:
        l_str = str(l)
        agg = {}
        for group in ["top5", "mid20_25", "low80_85", "random"]:
            kl_vals = []
            en_change_vals = []
            for word_key, word_data in causal_impact.items():
                if l_str in word_data and group in word_data[l_str]:
                    kl_vals.append(word_data[l_str][group]["mean_kl_divergence"])
                    en_change_vals.append(word_data[l_str][group]["mean_en_prob_change"])
            if kl_vals:
                agg[group] = {
                    "mean_kl": float(np.mean(kl_vals)),
                    "mean_en_change": float(np.mean(en_change_vals)),
                }
        aggregate_impact[l_str] = agg

    # 打印
    print(f"\n  层  | top5_KL | mid_KL  | low_KL  | random_KL | top5>low?")
    print(f"  {'─'*65}")
    for l in test_layers:
        l_str = str(l)
        agg = aggregate_impact.get(l_str, {})
        t = agg.get("top5", {}).get("mean_kl", 0)
        m = agg.get("mid20_25", {}).get("mean_kl", 0)
        lo = agg.get("low80_85", {}).get("mean_kl", 0)
        r = agg.get("random", {}).get("mean_kl", 0)
        verdict = "YES" if t > lo else "NO ← low var more impactful!"
        print(f"  L{l:2d} | {t:7.6f} | {m:7.6f} | {lo:7.6f} | {r:9.6f} | {verdict}")

    results["delta_h_svd"] = {
        "top10_sv": [float(s) for s in S_dh[:10]],
        "top10_cumvar": [float(c) for c in cumvar[:10]],
    }
    results["causal_impact"] = causal_impact
    results["aggregate_impact"] = aggregate_impact

    # 保存
    save_path = f"tests/glm5_temp/phase103_exp2_{model_name}_low_var_high_impact.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  保存到: {save_path}")

    release_model(model)
    return results


# ============================================================
# Exp 3: 乘法/门控干预
# ============================================================
def exp3_multiplicative_intervention(model_name):
    """
    乘法/门控干预 — 翻译需要gating而非加法

    Phase 102硬伤: 加法干预 h' = h + α*v 失败
    但这不意味着翻译方向不存在！可能需要gating/routing。

    本实验测试:
    3a: 加法干预 (baseline, 已知失败)
    3b: 乘法干预 h' = h * (1 + α*v_norm) — 缩放特定方向
    3c: LayerNorm后干预 — 让方向通过归一化"门控"
    3d: 注意力路由干预 — 修改特定head的注意力权重
    """
    print(f"\n{'='*70}")
    print(f"Exp 3: 乘法/门控干预 — {model_name}")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  模型: {model_name}, 层数: {n_layers}, d_model: {d_model}")

    results = {}

    # 训练集提取翻译方向
    train_pairs = [("猫", "cat"), ("狗", "dog"), ("书", "book"), ("火", "fire"),
                   ("花", "flower"), ("鱼", "fish"), ("树", "tree"), ("鸟", "bird"),
                   ("马", "horse"), ("铁", "iron"), ("金", "gold"), ("茶", "tea")]

    # 测试集
    test_pairs = [("龙", "dragon"), ("云", "cloud"), ("冰", "ice"), ("光", "light"),
                  ("梦", "dream"), ("沙", "sand"), ("影", "shadow"), ("歌", "song")]

    # ---- Step 1: 提取翻译方向 (与Phase 102相同) ----
    print(f"\n  === Step 1: 提取翻译方向 ===")

    # 上下文化翻译方向 (不同层)
    translation_dirs = {}
    for l in [9, 15, 21, 27, 33]:
        if l >= n_layers:
            continue
        deltas = []
        for zh, en in train_pairs:
            trans_prompt = f'请把"{zh}"翻译成英文：'
            zh_prompt = f"{zh}是一种"

            with torch.no_grad():
                out_trans = model(tokenizer(trans_prompt, return_tensors="pt").to(device)["input_ids"],
                                output_hidden_states=True)
                out_zh = model(tokenizer(zh_prompt, return_tensors="pt").to(device)["input_ids"],
                              output_hidden_states=True)

            h_trans = out_trans.hidden_states[l][0, -1, :].float().cpu().numpy()
            h_zh = out_zh.hidden_states[l][0, -1, :].float().cpu().numpy()
            deltas.append(h_trans - h_zh)

        mean_delta = np.mean(deltas, axis=0)
        mean_norm = np.linalg.norm(mean_delta)
        if mean_norm > 1e-6:
            translation_dirs[l] = mean_delta / mean_norm
        else:
            translation_dirs[l] = np.zeros(d_model)

        print(f"    L{l}: ||v_trans||={mean_norm:.1f}")

    # ---- Step 2: 不同干预方式对比 ----
    print(f"\n  === Step 2: 干预方式对比 ===")

    intervention_layers = [l for l in [9, 21, 27, 33] if l < n_layers]
    layers_list = get_layers(model)

    intervention_results = {}

    for l in intervention_layers:
        print(f"\n  --- 层 L{l} ---")
        v_trans = translation_dirs[l]
        v_trans_tensor = torch.tensor(v_trans, dtype=torch.float32, device=device)

        layer_results = {}

        for zh, en in test_pairs:
            zh_prompt = f"{zh}是一种"
            inputs = tokenizer(zh_prompt, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]

            # Baseline
            with torch.no_grad():
                base_logits = model(input_ids).logits[0, -1, :].float().cpu()
            base_en_prob = get_token_prob_from_logits(base_logits, tokenizer, en)
            base_top5 = get_top_k_from_logits(base_logits, tokenizer, 5)

            word_results = {"baseline_en_prob": base_en_prob}

            # --- 干预A: 加法 (已知失败，作为baseline) ---
            for alpha in [5.0, 20.0, 50.0]:
                intervened = [False]
                def make_additive_hook(alpha_val, direction, flag):
                    def hook_fn(module, input, output):
                        if not flag[0]:
                            if isinstance(output, tuple):
                                hs = output[0].clone()
                                hs[:, -1, :] += alpha_val * direction.to(hs.dtype).to(device)
                                output = (hs,) + output[1:]
                            flag[0] = True
                        return output
                    return hook_fn

                handle = layers_list[l].register_forward_hook(
                    make_additive_hook(alpha, v_trans_tensor, intervened))
                with torch.no_grad():
                    int_logits = model(input_ids).logits[0, -1, :].float().cpu()
                handle.remove()

                en_prob = get_token_prob_from_logits(int_logits, tokenizer, en)
                top5 = get_top_k_from_logits(int_logits, tokenizer, 5)
                word_results[f"additive_a{alpha:.0f}"] = {
                    "en_prob": en_prob,
                    "en_prob_change": en_prob - base_en_prob,
                    "top5": top5,
                }

            # --- 干预B: 乘法 h' = h * (1 + α * v_norm) ---
            for alpha in [0.1, 0.5, 2.0]:
                intervened = [False]
                def make_multiplicative_hook(alpha_val, direction, flag):
                    def hook_fn(module, input, output):
                        if not flag[0]:
                            if isinstance(output, tuple):
                                hs = output[0].clone()
                                # 乘法门控: 沿v_trans方向缩放
                                # h' = h * (1 + α * (v·h/||h||))
                                h_last = hs[0, -1, :].float()
                                proj = torch.dot(h_last, direction.to(h_last.dtype).to(device))
                                h_norm = h_last.norm()
                                if h_norm > 1e-6:
                                    gate = 1.0 + alpha_val * (proj / h_norm)
                                    hs[0, -1, :] = (h_last * gate).to(hs.dtype)
                                output = (hs,) + output[1:]
                            flag[0] = True
                        return output
                    return hook_fn

                handle = layers_list[l].register_forward_hook(
                    make_multiplicative_hook(alpha, v_trans_tensor, intervened))
                with torch.no_grad():
                    int_logits = model(input_ids).logits[0, -1, :].float().cpu()
                handle.remove()

                en_prob = get_token_prob_from_logits(int_logits, tokenizer, en)
                top5 = get_top_k_from_logits(int_logits, tokenizer, 5)
                word_results[f"multiplicative_a{alpha}"] = {
                    "en_prob": en_prob,
                    "en_prob_change": en_prob - base_en_prob,
                    "top5": top5,
                }

            # --- 干预C: LayerNorm后干预 ---
            # 思路: 在LayerNorm之后添加方向，让方向不受归一化压缩
            # 实现方式: 在post_attention_layernorm之后添加方向
            for alpha in [5.0, 20.0, 50.0]:
                intervened = [False]
                layer_obj = layers_list[l]

                # 找到post_attention_layernorm
                post_attn_ln = None
                for ln_name in ["post_attention_layernorm", "ln_2", "post_self_attn_layernorm"]:
                    if hasattr(layer_obj, ln_name):
                        post_attn_ln = getattr(layer_obj, ln_name)
                        break

                if post_attn_ln is not None:
                    def make_post_ln_hook(alpha_val, direction, flag):
                        def hook_fn(module, input, output):
                            if not flag[0]:
                                if isinstance(output, tuple):
                                    hs = output[0].clone()
                                    hs[:, -1, :] += alpha_val * direction.to(hs.dtype).to(device)
                                    output = (hs,) + output[1:]
                                flag[0] = True
                            return output
                        return hook_fn

                    handle = post_attn_ln.register_forward_hook(
                        make_post_ln_hook(alpha, v_trans_tensor, intervened))
                    with torch.no_grad():
                        int_logits = model(input_ids).logits[0, -1, :].float().cpu()
                    handle.remove()

                    en_prob = get_token_prob_from_logits(int_logits, tokenizer, en)
                    top5 = get_top_k_from_logits(int_logits, tokenizer, 5)
                    word_results[f"post_ln_a{alpha:.0f}"] = {
                        "en_prob": en_prob,
                        "en_prob_change": en_prob - base_en_prob,
                        "top5": top5,
                    }

            # --- 干预D: 注意力缩放 (修改特定head的输出) ---
            # 思路: 不修改residual stream，而是放大/缩小特定注意力头的输出
            for scale in [2.0, 5.0, 10.0]:
                intervened = [False]
                def make_attn_scale_hook(scale_val, flag):
                    def hook_fn(module, input, output):
                        if not flag[0]:
                            if isinstance(output, tuple):
                                hs = output[0].clone()
                                # 只缩放最后token位置的注意力输出
                                hs[:, -1, :] *= scale_val
                                output = (hs,) + output[1:]
                            flag[0] = True
                        return output
                    return hook_fn

                # 找到self_attn的o_proj
                attn = layer_obj.self_attn
                if hasattr(attn, 'o_proj'):
                    handle = attn.o_proj.register_forward_hook(
                        make_attn_scale_hook(scale, intervened))
                    with torch.no_grad():
                        int_logits = model(input_ids).logits[0, -1, :].float().cpu()
                    handle.remove()

                    en_prob = get_token_prob_from_logits(int_logits, tokenizer, en)
                    top5 = get_top_k_from_logits(int_logits, tokenizer, 5)
                    word_results[f"attn_scale_s{scale:.0f}"] = {
                        "en_prob": en_prob,
                        "en_prob_change": en_prob - base_en_prob,
                        "top5": top5,
                    }

            layer_results[zh] = word_results

            # 打印摘要
            best_additive = max([word_results.get(f"additive_a{a:.0f}", {}).get("en_prob_change", 0)
                                for a in [5.0, 20.0, 50.0]], default=0)
            best_mult = max([word_results.get(f"multiplicative_a{a}", {}).get("en_prob_change", 0)
                           for a in [0.1, 0.5, 2.0]], default=0)
            best_post_ln = max([word_results.get(f"post_ln_a{a:.0f}", {}).get("en_prob_change", 0)
                               for a in [5.0, 20.0, 50.0]], default=0)
            best_attn = max([word_results.get(f"attn_scale_s{s:.0f}", {}).get("en_prob_change", 0)
                           for s in [2.0, 5.0, 10.0]], default=0)

            print(f"    {zh}({en}): baseline={base_en_prob:.6f}, "
                  f"additive={best_additive:.6f}, mult={best_mult:.6f}, "
                  f"post_ln={best_post_ln:.6f}, attn={best_attn:.6f}")

        intervention_results[f"L{l}"] = layer_results

    # ---- Step 3: 跨词聚合 ----
    print(f"\n  === Step 3: 干预方式效果对比 ===")
    for l in intervention_layers:
        l_key = f"L{l}"
        if l_key not in intervention_results:
            continue

        for method_prefix in ["additive", "multiplicative", "post_ln", "attn_scale"]:
            changes = []
            for zh, word_data in intervention_results[l_key].items():
                for key, val in word_data.items():
                    if key.startswith(method_prefix) and "en_prob_change" in val:
                        changes.append(val["en_prob_change"])
            if changes:
                print(f"    L{l} {method_prefix}: mean_ΔP(en)={np.mean(changes):.6f}, "
                      f"max_ΔP(en)={np.max(changes):.6f}")

    results["intervention"] = intervention_results

    # 保存
    save_path = f"tests/glm5_temp/phase103_exp3_{model_name}_multiplicative_intervention.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  保存到: {save_path}")

    release_model(model)
    return results


# ============================================================
# Exp 4: 上下文插值与吸引子盆地
# ============================================================
def exp4_attractor_basins(model_name):
    """
    上下文插值与吸引子盆地 — 相变分析

    核心思路:
    - "猫是一种" → 中文吸引子
    - "猫的英文是cat" → 英文吸引子
    - 两者之间是否存在相变点？

    方法:
    - 构造上下文插值序列: 从纯中文到纯翻译
    - 在每个插值点，测量:
      (a) 输出分布中中文/英文token的概率
      (b) 各层hidden state的变化
    - 寻找相变点: P(中文)和P(英文)交叉的位置

    插值序列示例 (猫→cat):
    1. "猫是一种"
    2. "猫是一种动物"
    3. "猫在英文中"
    4. "猫的英文翻译是"
    5. "猫的英文是"
    6. "Please translate 猫 to English:"
    """
    print(f"\n{'='*70}")
    print(f"Exp 4: 上下文插值与吸引子盆地 — {model_name}")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  模型: {model_name}, 层数: {n_layers}, d_model: {d_model}")

    results = {}

    test_pairs = [("猫", "cat"), ("水", "water"), ("火", "fire"), ("树", "tree"),
                  ("龙", "dragon"), ("光", "light")]

    # 构造插值序列 (从中文到翻译)
    def make_interpolation_prompts(zh, en):
        """构造从纯中文到纯翻译的插值序列"""
        return [
            f"{zh}是一种",                          # 0: 纯中文
            f"{zh}是一种常见的",                     # 1: 中文扩展
            f"关于{zh}，我想说",                     # 2: 中文开放
            f"{zh}的另一个名字是",                    # 3: 接近翻译(中文表达)
            f"{zh}在英文中叫做",                     # 4: 翻译提示(中文)
            f"{zh}的英文是",                         # 5: 翻译提示(简洁)
            f'请把"{zh}"翻译成英文：',               # 6: 翻译指令
            f"Translate {zh} to English:",           # 7: 英文翻译指令
        ]

    attractor_results = {}

    for zh, en in test_pairs:
        print(f"\n  === 处理: {zh}({en}) ===")
        prompts = make_interpolation_prompts(zh, en)
        word_results = {}

        for p_idx, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]

            with torch.no_grad():
                outputs = model(input_ids, output_hidden_states=True)

            logits = outputs.logits[0, -1, :].float().cpu()
            probs = torch.softmax(logits, dim=-1)

            # 统计中文/英文token概率
            # 简化: 检查top-100中英文token的比例
            top100 = torch.topk(probs, 100)
            en_count = 0
            zh_count = 0
            en_total_prob = 0.0
            zh_total_prob = 0.0

            for i in range(100):
                tok_id = top100.indices[i].item()
                prob = top100.values[i].item()
                tok_str = tokenizer.decode([tok_id])

                # 简单判断: ASCII字符=英文, Unicode=中文
                is_ascii = all(ord(c) < 128 for c in tok_str.strip())
                if is_ascii and tok_str.strip():
                    en_count += 1
                    en_total_prob += prob
                elif tok_str.strip():
                    zh_count += 1
                    zh_total_prob += prob

            # 特定翻译词概率
            en_tok_ids = tokenizer.encode(en, add_special_tokens=False)
            en_translation_prob = probs[en_tok_ids[0]].item() if en_tok_ids else 0

            zh_tok_ids = tokenizer.encode(zh, add_special_tokens=False)
            zh_word_prob = probs[zh_tok_ids[0]].item() if zh_tok_ids else 0

            # 最后token的hidden state范数 (各层)
            h_norms = {}
            for l in [0, 9, 18, 27, 33]:
                if l < n_layers + 1:
                    h = outputs.hidden_states[l][0, -1, :].float()
                    h_norms[str(l)] = float(h.norm())

            word_results[str(p_idx)] = {
                "prompt": prompt,
                "en_translation_prob": en_translation_prob,
                "zh_word_prob": zh_word_prob,
                "en_total_prob_top100": en_total_prob,
                "zh_total_prob_top100": zh_total_prob,
                "en_count_top100": en_count,
                "zh_count_top100": zh_count,
                "en_zh_ratio": en_total_prob / max(zh_total_prob, 1e-10),
                "h_norms": h_norms,
                "top5": get_top_k_from_logits(logits, tokenizer, 5),
            }

            print(f"    [{p_idx}] {prompt:30s} → P({en})={en_translation_prob:.6f}, "
                  f"en_total={en_total_prob:.3f}, zh_total={zh_total_prob:.3f}, "
                  f"en/zh={en_total_prob/max(zh_total_prob,1e-10):.2f}")

        # 找相变点
        en_probs = [word_results[str(i)]["en_total_prob_top100"] for i in range(len(prompts))]
        zh_probs = [word_results[str(i)]["zh_total_prob_top100"] for i in range(len(prompts))]

        # 相变点: en_prob超过zh_prob的最早位置
        transition_point = None
        for i in range(len(prompts)):
            if en_probs[i] > zh_probs[i]:
                transition_point = i
                break

        if transition_point is not None:
            print(f"    ** 相变点: prompt[{transition_point}] '{prompts[transition_point]}' **")
        else:
            print(f"    ** 未找到相变点 (中文始终主导) **")

        word_results["transition_point"] = transition_point
        attractor_results[f"{zh}_{en}"] = word_results

    # ---- Step 2: 层间相变分析 ----
    print(f"\n  === Step 2: 层间相变分析 ===")
    print(f"  (在翻译prompt下，各层的中文/英文倾向)")

    for zh, en in test_pairs[:3]:
        print(f"\n  {zh}({en}):")
        trans_prompt = f'请把"{zh}"翻译成英文：'
        inputs = tokenizer(trans_prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)

        W_U = get_W_U(model)  # [vocab_size, d_model]

        for l in [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33]:
            if l >= n_layers + 1:
                continue
            h = outputs.hidden_states[l][0, -1, :].float().cpu().numpy()

            # Logit lens: h @ W_U^T → logits
            logits_l = h @ W_U.T
            probs_l = np.exp(logits_l - np.max(logits_l))
            probs_l = probs_l / np.sum(probs_l)

            # 特定词概率
            en_tok_ids = tokenizer.encode(en, add_special_tokens=False)
            en_prob = float(probs_l[en_tok_ids[0]]) if en_tok_ids else 0

            # top-5
            top5_idx = np.argsort(probs_l)[-5:][::-1]
            top5_tokens = [(tokenizer.decode([int(idx)]), float(probs_l[idx])) for idx in top5_idx]

            print(f"    L{l:2d}: P({en})={en_prob:.6f}, top5={[(t, f'{p:.4f}') for t, p in top5_tokens]}")

    results["attractor"] = attractor_results

    # 保存
    save_path = f"tests/glm5_temp/phase103_exp4_{model_name}_attractor_basins.json"
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
    parser.add_argument("--exp", type=int, default=1, choices=[1, 2, 3, 4])
    args = parser.parse_args()

    if args.exp == 1:
        exp1_jacobian_spectrum(args.model)
    elif args.exp == 2:
        exp2_low_variance_high_impact(args.model)
    elif args.exp == 3:
        exp3_multiplicative_intervention(args.model)
    elif args.exp == 4:
        exp4_attractor_basins(args.model)
