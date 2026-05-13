"""Phase 149: Token Coupling Jacobian + Three-Level Direction Decomposition
===========================================================================

用户理论审查的核心结论:
  1. "主动null-space路由"已被推翻 (Phase 148 Exp1: null_ratio差异仅0.0033)
  2. 扰动方向在传播中被"随机化" — 各向异性丢失
  3. 但Phase 147和148存在表面矛盾: 方向保持(cos≈0.999) vs 方向随机化
  4. 解决: "方向"不是单一对象 — 需要分三个层次测量
  5. Token Coupling是最大遗漏 — 语言本质是跨token约束传播

三个实验:
  Exp 1: Token Coupling — 跨token扰动传播
    - 在position j注入扰动, 测量position i的响应
    - C[j,i] = ||Δh_i|| / ||δ|| — token间耦合强度
    - 关键: 自耦合 vs 交叉耦合, row/null分布, 位置距离依赖

  Exp 2: 三层次方向分解 — 解决Phase 147/148矛盾
    - 层次1: 全局余弦 cos(δ_L0, δ_Li) — Phase 147的度量
    - 层次2: null_ratio逐层演化 — Phase 148的度量
    - 层次3: null-space内部余弦 cos(δ_L0_null, δ_Li_null) — 新度量!
    - 如果全局余弦高但null-内部余弦低 → 内部混合确认

  Exp 3: Rollout相图 — (epsilon, layer) → 稳定性分区
    - 变化epsilon和注入层, 测量rollout稳定性
    - 画出: 稳定区 / 亚稳态区 / 发散区

用法:
  python tests/glm5/phase149_token_coupling_and_direction.py qwen3
  python tests/glm5/phase149_token_coupling_and_direction.py glm4
  python tests/glm5/phase149_token_coupling_and_direction.py deepseek7b
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import json
import time
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from model_utils import (load_model, get_layers, get_model_info,
                          release_model, get_W_U, MODEL_CONFIGS)

# ============================================================
# 全局参数
# ============================================================
EPSILON = 2.0
N_SENTENCES_EXP1 = 8   # Exp1: token coupling需要更多数据
N_SENTENCES_EXP2 = 15  # Exp2: 方向分解是核心, 加大样本
N_SENTENCES_EXP3 = 6   # Exp3: 相图只需少量样本

TEST_PROMPTS = [
    "The scientist discovered that the",
    "In the morning, she decided to",
    "The book on the table was about",
    "After the rain stopped, the children",
    "The most important thing about science is",
    "When the sun sets over the ocean,",
    "She walked into the room and saw",
    "The professor explained that the theory",
    "Despite the challenges, the team managed",
    "The ancient city was known for its",
    "He realized that the answer was",
    "The relationship between language and thought",
    "Every morning she would read the",
    "The experiment showed that the results",
    "Music has the power to change how",
]

OUTPUT_DIR = Path("tests/glm5_temp")


# ============================================================
# 工具函数
# ============================================================
def get_device_for_input(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda")


def get_row_null_bases(W_U, n_components=200):
    """计算W_U的row space基底 (随机化SVD)"""
    d = W_U.shape[1]
    k = min(n_components, d - 2)
    
    np.random.seed(42)
    n_samples = k + 10
    
    Omega = np.random.randn(W_U.shape[0], n_samples).astype(np.float32)
    
    batch_size = 50000
    Y = np.zeros((d, n_samples), dtype=np.float32)
    for i in range(0, W_U.shape[0], batch_size):
        end = min(i + batch_size, W_U.shape[0])
        Y += W_U[i:end].T.astype(np.float32) @ Omega[i:end]
    del Omega
    
    Q, R = np.linalg.qr(Y)
    del Y, R
    
    effective_k = min(k, Q.shape[1])
    row_basis = Q[:, :effective_k].T  # [effective_k, d_model]
    
    return row_basis, effective_k


def project_to_null(vec, row_basis):
    """将向量投影到W_U的null space"""
    row_component = row_basis.T @ (row_basis @ vec)
    return vec - row_component


def project_to_row(vec, row_basis):
    """将向量投影到W_U的row space"""
    return row_basis.T @ (row_basis @ vec)


def compute_row_energy(delta, row_basis):
    """计算扰动delta在row space中的能量比例"""
    delta_norm_sq = np.sum(delta ** 2)
    if delta_norm_sq < 1e-16:
        return 0.0, 1.0
    row_coeffs = row_basis @ delta
    row_energy = np.sum(row_coeffs ** 2) / delta_norm_sq
    null_ratio = 1.0 - row_energy
    return float(row_energy), float(null_ratio)


def run_forward_with_perturbation_at_position(model, input_ids, attention_mask,
                                                inject_layer, position, delta_np, device):
    """在指定层、指定position注入扰动, 返回output_hidden_states"""
    layers = get_layers(model)

    def make_inject_hook(pos, delta_tensor):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0].clone()
                out[0, pos, :] += delta_tensor.to(out.dtype).to(out.device)
                return (out,) + output[1:]
            else:
                out = output.clone()
                out[0, pos, :] += delta_tensor.to(out.dtype).to(out.device)
                return out
        return hook

    hooks = [layers[inject_layer].register_forward_hook(
        make_inject_hook(position, torch.tensor(delta_np, dtype=torch.float32)))]

    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)
    finally:
        for h in hooks:
            h.remove()

    return out


# ============================================================
# Exp 1: Token Coupling — 跨token扰动传播
# ============================================================
def exp1_token_coupling(model, tokenizer, model_name, W_U, row_basis):
    print("\n" + "="*60)
    print("Exp 1: Token Coupling — 跨token扰动传播")
    print("核心: 在position j注入扰动, 测量position i的响应强度")
    print("="*60)

    info = get_model_info(model, model_name)
    layers = get_layers(model)
    device = get_device_for_input(model)
    d_model = info.d_model
    n_layers = info.n_layers

    # 注入层: 早层(L0), 中层(L_mid), 晚层(L_late)
    inject_layers = [0, n_layers // 2, n_layers - 5]
    # 测量层: 注入层+1, 注入层+5, 最终层
    measure_offsets = [1, 5]  # 相对于注入层的偏移

    all_results = {}

    for sent_idx in range(N_SENTENCES_EXP1):
        prompt = TEST_PROMPTS[sent_idx]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        seq_len = input_ids.shape[1]

        # Clean forward — 保存每层每个位置的hidden state
        with torch.no_grad():
            out_clean = model(input_ids=input_ids, attention_mask=attention_mask,
                              output_hidden_states=True)
        clean_hs = out_clean.hidden_states  # (n_layers+1) x [1, seq_len, d_model]

        for inject_l in inject_layers:
            if inject_l >= n_layers:
                continue

            # 生成随机扰动 (null-space方向)
            np.random.seed(sent_idx * 1000 + inject_l)
            raw_vec = np.random.randn(d_model)
            delta = project_to_null(raw_vec, row_basis)
            norm = np.linalg.norm(delta)
            if norm > 1e-8:
                delta = delta / norm * EPSILON
            else:
                continue

            # 注入在每个位置, 测量其他位置的响应
            # 为了效率, 只测试3个注入位置: first, middle, last
            inject_positions = [0, seq_len // 2, seq_len - 1]

            for inject_pos in inject_positions:
                # 扰动forward
                out_perturbed = run_forward_with_perturbation_at_position(
                    model, input_ids, attention_mask, inject_l, inject_pos, delta, device)

                # 在测量层计算耦合
                for offset in measure_offsets:
                    measure_l = inject_l + offset
                    if measure_l > n_layers:
                        continue

                    perturbed_vecs = out_perturbed.hidden_states[measure_l][0].float().cpu().numpy()  # [seq_len, d_model]
                    clean_vecs = clean_hs[measure_l][0].float().cpu().numpy()  # [seq_len, d_model]

                    # 计算每个位置的响应
                    coupling_data = []
                    for pos_i in range(seq_len):
                        delta_i = perturbed_vecs[pos_i] - clean_vecs[pos_i]
                        delta_i_norm = np.linalg.norm(delta_i)

                        if delta_i_norm < 1e-10:
                            coupling_data.append({
                                'pos': pos_i,
                                'response_norm': 0.0,
                                'row_energy': 0.0,
                                'null_ratio': 1.0,
                            })
                            continue

                        row_e, null_r = compute_row_energy(delta_i, row_basis)
                        coupling_data.append({
                            'pos': pos_i,
                            'response_norm': float(delta_i_norm),
                            'row_energy': float(row_e),
                            'null_ratio': float(null_r),
                        })

                    key = f"sent{sent_idx}_L{inject_l}_pos{inject_pos}_mL{measure_l}"
                    all_results[key] = coupling_data

                    # 打印耦合强度矩阵 (简略)
                    self_response = coupling_data[inject_pos]['response_norm']
                    cross_responses = [c['response_norm'] for i, c in enumerate(coupling_data) if i != inject_pos]
                    avg_cross = np.mean(cross_responses) if cross_responses else 0
                    max_cross_pos = max(range(len(coupling_data)), key=lambda i: coupling_data[i]['response_norm'])

                    print(f"  Sent{sent_idx} L{inject_l}→L{measure_l} inj@pos{inject_pos}: "
                          f"self={self_response:.3f}, avg_cross={avg_cross:.3f}, "
                          f"max_cross@pos{max_cross_pos}={coupling_data[max_cross_pos]['response_norm']:.3f}, "
                          f"ratio={self_response/(avg_cross+1e-10):.1f}x")

    # === 汇总分析 ===
    print("\n  === Exp 1 Summary: Token Coupling ===")

    # 按注入层分组汇总
    for inject_l in inject_layers:
        for offset in measure_offsets:
            measure_l = inject_l + offset
            if measure_l > n_layers:
                continue

            self_responses = []
            cross_responses = []
            cross_null_ratios = []
            self_null_ratios = []

            for key, coupling_data in all_results.items():
                if f"_L{inject_l}_" not in key or f"_mL{measure_l}" not in key:
                    continue

                # 解析注入位置: key格式 sent{idx}_L{layer}_pos{pos}_mL{measure}
                parts = key.split('_')
                inject_pos = None
                for p in parts:
                    if p.startswith('pos'):
                        inject_pos = int(p[3:])
                        break
                if inject_pos is None:
                    continue

                for i, cd in enumerate(coupling_data):

                    if i == inject_pos:
                        self_responses.append(cd['response_norm'])
                        self_null_ratios.append(cd['null_ratio'])
                    else:
                        cross_responses.append(cd['response_norm'])
                        cross_null_ratios.append(cd['null_ratio'])

            if self_responses and cross_responses:
                avg_self = np.mean(self_responses)
                avg_cross = np.mean(cross_responses)
                ratio = avg_self / (avg_cross + 1e-10)

                print(f"\n  L{inject_l}→L{measure_l}:")
                print(f"    Self-coupling:  {avg_self:.4f} (null_ratio={np.mean(self_null_ratios):.4f})")
                print(f"    Cross-coupling: {avg_cross:.4f} (null_ratio={np.mean(cross_null_ratios):.4f})")
                print(f"    Self/Cross ratio: {ratio:.1f}x")
                print(f"    Cross-coupling null_ratio vs self null_ratio: "
                      f"{np.mean(cross_null_ratios):.4f} vs {np.mean(self_null_ratios):.4f}")

                # 关键判据
                if ratio > 10:
                    print(f"    → 强局部性: 扰动主要留在同一token位置")
                elif ratio > 3:
                    print(f"    → 中等局部性: 扰动部分传播到其他token")
                else:
                    print(f"    → 弱局部性: 扰动广泛传播到其他token位置")

    return all_results


# ============================================================
# Exp 2: 三层次方向分解 — 解决Phase 147/148矛盾
# ============================================================
def exp2_three_level_direction(model, tokenizer, model_name, W_U, row_basis):
    print("\n" + "="*60)
    print("Exp 2: Three-Level Direction Decomposition")
    print("核心: 区分全局方向保持 vs null-space内部方向混合")
    print("="*60)

    info = get_model_info(model, model_name)
    layers = get_layers(model)
    device = get_device_for_input(model)
    d_model = info.d_model
    n_layers = info.n_layers

    # 测量层: 每隔几层采样
    sample_layers = sorted(set([0] + list(range(1, n_layers, max(1, n_layers // 8))) + [n_layers - 1, n_layers]))
    sample_layers = [l for l in sample_layers if l <= n_layers]

    results = {}

    for sent_idx in range(N_SENTENCES_EXP2):
        prompt = TEST_PROMPTS[sent_idx]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Clean forward
        with torch.no_grad():
            out_clean = model(input_ids=input_ids, attention_mask=attention_mask,
                              output_hidden_states=True)
        clean_hs = out_clean.hidden_states

        # 两种注入: null-space 和 row-space
        for dir_type in ["null", "row"]:
            np.random.seed(sent_idx * 100 + (0 if dir_type == "null" else 50))
            raw_vec = np.random.randn(d_model)

            if dir_type == "null":
                delta = project_to_null(raw_vec, row_basis)
            else:
                delta = project_to_row(raw_vec, row_basis)

            norm = np.linalg.norm(delta)
            if norm < 1e-8:
                continue
            delta = delta / norm * EPSILON

            # 注入层: L0
            inject_l = 0

            # 保存注入扰动的null分量和row分量
            delta_null_original = project_to_null(delta, row_basis)
            delta_row_original = project_to_row(delta, row_basis)
            delta_null_norm = np.linalg.norm(delta_null_original)
            delta_row_norm = np.linalg.norm(delta_row_original)

            # 扰动forward
            out_perturbed = run_forward_with_perturbation_at_position(
                model, input_ids, attention_mask, inject_l, input_ids.shape[1] - 1, delta, device)

            # 逐层分析
            layer_data = []
            for li in sample_layers:
                perturbed_vec = out_perturbed.hidden_states[li][0, -1, :].float().cpu().numpy()
                clean_vec = clean_hs[li][0, -1, :].float().cpu().numpy()
                delta_prop = perturbed_vec - clean_vec
                delta_prop_norm = np.linalg.norm(delta_prop)

                if delta_prop_norm < 1e-10:
                    layer_data.append({
                        'layer': li,
                        'global_cos': 0.0,
                        'null_ratio': 1.0,
                        'null_internal_cos': 0.0,
                        'row_internal_cos': 0.0,
                        'delta_norm': 0.0,
                    })
                    continue

                # === 层次1: 全局余弦 ===
                global_cos = float(np.dot(delta, delta_prop) / (np.linalg.norm(delta) * delta_prop_norm))

                # === 层次2: null_ratio ===
                row_e, null_r = compute_row_energy(delta_prop, row_basis)

                # === 层次3: null-space内部余弦 ===
                # 将传播后的扰动分解为null和row分量
                delta_prop_null = project_to_null(delta_prop, row_basis)
                delta_prop_row = project_to_row(delta_prop, row_basis)

                null_internal_cos = 0.0
                if delta_null_norm > 1e-8 and np.linalg.norm(delta_prop_null) > 1e-8:
                    null_internal_cos = float(np.dot(delta_null_original, delta_prop_null) /
                                              (delta_null_norm * np.linalg.norm(delta_prop_null)))

                row_internal_cos = 0.0
                if delta_row_norm > 1e-8 and np.linalg.norm(delta_prop_row) > 1e-8:
                    row_internal_cos = float(np.dot(delta_row_original, delta_prop_row) /
                                              (delta_row_norm * np.linalg.norm(delta_prop_row)))

                layer_data.append({
                    'layer': li,
                    'global_cos': global_cos,
                    'null_ratio': null_r,
                    'null_internal_cos': null_internal_cos,
                    'row_internal_cos': row_internal_cos,
                    'delta_norm': float(delta_prop_norm),
                    'delta_null_norm': float(np.linalg.norm(delta_prop_null)),
                    'delta_row_norm': float(np.linalg.norm(delta_prop_row)),
                })

            key = f"sent{sent_idx}_{dir_type}"
            results[key] = layer_data

            # 打印关键层
            print(f"  Sent{sent_idx} ({dir_type}-space input):")
            for ld in layer_data:
                li = ld['layer']
                print(f"    L{li:2d}: global_cos={ld['global_cos']:.4f}, "
                      f"null_ratio={ld['null_ratio']:.4f}, "
                      f"null_int_cos={ld['null_internal_cos']:.4f}, "
                      f"row_int_cos={ld['row_internal_cos']:.4f}, "
                      f"||δ||={ld['delta_norm']:.3f}")

    # === 汇总分析 ===
    print("\n  === Exp 2 Summary: Three-Level Direction ===")

    for dir_type in ["null", "row"]:
        # 收集所有句子在每个采样层的平均值
        layer_avgs = {}
        for key, layer_data in results.items():
            if f"_{dir_type}" not in key:
                continue
            for ld in layer_data:
                li = ld['layer']
                if li not in layer_avgs:
                    layer_avgs[li] = {
                        'global_cos': [], 'null_ratio': [],
                        'null_internal_cos': [], 'row_internal_cos': [],
                        'delta_norm': [],
                    }
                layer_avgs[li]['global_cos'].append(ld['global_cos'])
                layer_avgs[li]['null_ratio'].append(ld['null_ratio'])
                layer_avgs[li]['null_internal_cos'].append(ld['null_internal_cos'])
                layer_avgs[li]['row_internal_cos'].append(ld['row_internal_cos'])
                layer_avgs[li]['delta_norm'].append(ld['delta_norm'])

        print(f"\n  --- {dir_type.upper()}-space input (avg over {N_SENTENCES_EXP2} sentences) ---")
        print(f"  {'Layer':>6} {'global_cos':>12} {'null_ratio':>12} {'null_int_cos':>14} {'row_int_cos':>14} {'||δ||':>8}")
        for li in sorted(layer_avgs.keys()):
            avg_gc = np.mean(layer_avgs[li]['global_cos'])
            avg_nr = np.mean(layer_avgs[li]['null_ratio'])
            avg_nic = np.mean(layer_avgs[li]['null_internal_cos'])
            avg_ric = np.mean(layer_avgs[li]['row_internal_cos'])
            avg_dn = np.mean(layer_avgs[li]['delta_norm'])
            print(f"  L{li:>4d} {avg_gc:>12.4f} {avg_nr:>12.4f} {avg_nic:>14.4f} {avg_ric:>14.4f} {avg_dn:>8.3f}")

        # 关键判据
        # 找到global_cos仍然>0.9但null_internal_cos已经<0.5的层
        early_layers = [li for li in sorted(layer_avgs.keys()) if li <= n_layers // 2]
        late_layers = [li for li in sorted(layer_avgs.keys()) if li > n_layers // 2]

        if early_layers:
            early_gc = np.mean([np.mean(layer_avgs[li]['global_cos']) for li in early_layers])
            early_nic = np.mean([np.mean(layer_avgs[li]['null_internal_cos']) for li in early_layers])
            print(f"\n  Early layers (≤L{n_layers//2}): avg_global_cos={early_gc:.4f}, avg_null_int_cos={early_nic:.4f}")

        if late_layers:
            late_gc = np.mean([np.mean(layer_avgs[li]['global_cos']) for li in late_layers])
            late_nic = np.mean([np.mean(layer_avgs[li]['null_internal_cos']) for li in late_layers])
            print(f"  Late layers (>L{n_layers//2}): avg_global_cos={late_gc:.4f}, avg_null_int_cos={late_nic:.4f}")

        # 判断
        print(f"\n  *** 关键判据 ({dir_type}-space input) ***")
        if early_layers:
            if early_gc > 0.95 and early_nic < 0.5:
                print(f"  → 全局方向保持但null-内部方向混合 → 扰动混合确认!")
            elif early_gc > 0.95 and early_nic > 0.8:
                print(f"  → 全局方向和null-内部方向都保持 → 没有混合, 需要其他解释")
            elif early_gc < 0.8:
                print(f"  → 全局方向也不保持 → Phase 147的方向保持需要重新审视")
            else:
                print(f"  → 中等情况: global_cos={early_gc:.4f}, null_int_cos={early_nic:.4f}")
                print(f"  → 部分混合, 但不是完全随机化")

    return results


# ============================================================
# Exp 3: Rollout相图 — (epsilon, layer) → 稳定性分区
# ============================================================
def exp3_rollout_phase_diagram(model, tokenizer, model_name, W_U, row_basis):
    print("\n" + "="*60)
    print("Exp 3: Rollout Phase Diagram — (epsilon, layer) → stability")
    print("核心: 画出扰动传播的稳定性分区图")
    print("="*60)

    info = get_model_info(model, model_name)
    layers = get_layers(model)
    device = get_device_for_input(model)
    d_model = info.d_model
    n_layers = info.n_layers

    # 参数网格
    epsilons = [0.1, 0.5, 1.0, 2.0, 5.0]
    inject_fractions = [0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0]
    inject_layers = [int(f * (n_layers - 1)) for f in inject_fractions]
    n_rollout_tokens = 20

    results = {}

    for eps in epsilons:
        for inject_l in inject_layers:
            if inject_l >= n_layers:
                continue

            overlaps = []
            kl_divs = []

            for sent_idx in range(N_SENTENCES_EXP3):
                prompt = TEST_PROMPTS[sent_idx]
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)

                # Clean forward
                with torch.no_grad():
                    out_clean = model(input_ids=input_ids, attention_mask=attention_mask,
                                      output_hidden_states=True)
                logits_clean = out_clean.logits[0, -1].float().cpu().numpy()
                probs_clean = np.exp(logits_clean - np.max(logits_clean))
                probs_clean = probs_clean / probs_clean.sum()
                top5_clean = set(np.argsort(logits_clean)[-5:])

                # 扰动
                np.random.seed(sent_idx * 100 + int(eps * 100) + inject_l)
                raw_vec = np.random.randn(d_model)
                delta = project_to_null(raw_vec, row_basis)
                norm = np.linalg.norm(delta)
                if norm > 1e-8:
                    delta = delta / norm * eps
                else:
                    continue

                # 扰动forward
                out_perturbed = run_forward_with_perturbation_at_position(
                    model, input_ids, attention_mask, inject_l, input_ids.shape[1] - 1, delta, device)

                logits_perturbed = out_perturbed.logits[0, -1].float().cpu().numpy()
                probs_perturbed = np.exp(logits_perturbed - np.max(logits_perturbed))
                probs_perturbed = probs_perturbed / probs_perturbed.sum()
                top5_perturbed = set(np.argsort(logits_perturbed)[-5:])

                # Top-5 overlap
                overlap = len(top5_clean & top5_perturbed) / 5.0
                overlaps.append(overlap)

                # KL divergence
                kl = np.sum(probs_clean * np.log((probs_clean + 1e-10) / (probs_perturbed + 1e-10)))
                kl_divs.append(float(kl))

            avg_overlap = np.mean(overlaps) if overlaps else 0
            avg_kl = np.mean(kl_divs) if kl_divs else 0

            key = f"eps{eps}_L{inject_l}"
            results[key] = {
                'epsilon': eps,
                'inject_layer': inject_l,
                'avg_top5_overlap': float(avg_overlap),
                'avg_kl_divergence': float(avg_kl),
            }

            # 稳定性分类
            if avg_overlap > 0.8:
                stability = "STABLE"
            elif avg_overlap > 0.4:
                stability = "METASTABLE"
            else:
                stability = "DIVERGENT"

            print(f"  eps={eps:4.1f}, L{inject_l:>2d}: overlap={avg_overlap:.3f}, "
                  f"KL={avg_kl:.4f} → {stability}")

    # === 相图汇总 ===
    print("\n  === Exp 3 Summary: Rollout Phase Diagram ===")
    print(f"\n  {'eps\\L':>6}", end="")
    for inject_l in inject_layers:
        print(f"  L{inject_l:>2d}", end="")
    print()

    for eps in epsilons:
        print(f"  {eps:>5.1f}", end="")
        for inject_l in inject_layers:
            key = f"eps{eps}_L{inject_l}"
            if key in results:
                overlap = results[key]['avg_top5_overlap']
                if overlap > 0.8:
                    sym = " ■"  # 稳定
                elif overlap > 0.4:
                    sym = " ▒"  # 亚稳态
                else:
                    sym = " □"  # 发散
                print(f"  {sym}{overlap:.2f}", end="")
            else:
                print(f"    -", end="")
        print()

    print("\n  Legend: ■=STABLE(overlap>0.8), ▒=METASTABLE(0.4-0.8), □=DIVERGENT(<0.4)")

    # 找到相边界
    print("\n  Phase boundaries:")
    for eps in epsilons:
        # 找到从STABLE到METASTABLE的边界
        boundary_layer = None
        for inject_l in inject_layers:
            key = f"eps{eps}_L{inject_l}"
            if key in results and results[key]['avg_top5_overlap'] < 0.8:
                boundary_layer = inject_l
                break
        if boundary_layer is not None:
            print(f"    eps={eps:.1f}: STABLE→METASTABLE at L{boundary_layer}")
        else:
            print(f"    eps={eps:.1f}: all STABLE")

    return results


# ============================================================
# 主函数
# ============================================================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    print(f"Phase 149: Token Coupling + Three-Level Direction")
    print(f"Model: {model_name}")
    print(f"Time: {timestamp}")

    # 加载模型
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    use_8bit = model_name in ("deepseek7b", "glm4") and gpu_mem_gb < 16
    print(f"Mode: {'8bit' if use_8bit else 'bfloat16'}")

    t0 = time.time()
    model, tokenizer, device = load_model(model_name, use_8bit=use_8bit)
    info = get_model_info(model, model_name)
    print(f"Model: {info.model_class}, {info.n_layers}L, d={info.d_model}")
    print(f"Load time: {time.time()-t0:.1f}s")

    # 获取W_U和row/null space
    t0 = time.time()
    W_U = get_W_U(model, model_name)
    row_basis, k = get_row_null_bases(W_U)
    print(f"W_U: shape={W_U.shape}, row_components={k}, null_dim={W_U.shape[1]-k}")
    print(f"SVD time: {time.time()-t0:.1f}s")

    # 运行3个实验
    print("\n" + "#"*60)
    print("# Running Experiments")
    print("#"*60)

    exp1_results = exp1_token_coupling(model, tokenizer, model_name, W_U, row_basis)

    exp2_results = exp2_three_level_direction(model, tokenizer, model_name, W_U, row_basis)

    exp3_results = exp3_rollout_phase_diagram(model, tokenizer, model_name, W_U, row_basis)

    # 保存结果
    all_results = {
        "phase": 149,
        "model": model_name,
        "timestamp": timestamp,
        "model_info": {
            "class": info.model_class,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
        },
        "exp1_token_coupling": {},
        "exp2_three_level_direction": {},
        "exp3_rollout_phase_diagram": exp3_results,
    }

    # Exp1 汇总 (简化, 只保留关键数据)
    for key, coupling_data in exp1_results.items():
        # 解析注入位置
        parts = key.split('_')
        inject_pos = 0
        for p in parts:
            if p.startswith('pos'):
                inject_pos = int(p[3:])
                break

        self_resp = coupling_data[inject_pos]['response_norm'] if inject_pos < len(coupling_data) else 0
        cross_resps = [c['response_norm'] for i, c in enumerate(coupling_data) if i != inject_pos]
        all_results["exp1_token_coupling"][key] = {
            'self_response': self_resp,
            'avg_cross_response': float(np.mean(cross_resps)) if cross_resps else 0,
            'max_cross_response': max(coupling_data, key=lambda x: x['response_norm'])['response_norm']
            if coupling_data else 0,
        }

    # Exp2 汇总
    for key, layer_data in exp2_results.items():
        all_results["exp2_three_level_direction"][key] = {
            'layers': [ld['layer'] for ld in layer_data],
            'global_cos': [ld['global_cos'] for ld in layer_data],
            'null_ratio': [ld['null_ratio'] for ld in layer_data],
            'null_internal_cos': [ld['null_internal_cos'] for ld in layer_data],
            'row_internal_cos': [ld['row_internal_cos'] for ld in layer_data],
        }

    result_file = OUTPUT_DIR / f"phase149_{model_name}_{timestamp}.json"

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        raise TypeError(f"Cannot serialize {type(obj)}")

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=convert, ensure_ascii=False)

    print(f"\nResults saved to: {result_file}")

    # 释放模型
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    print("Model released.")


if __name__ == "__main__":
    main()
