"""
Phase 105: Margin动力学与约束可读出性分析
=============================================

Phase 104的硬伤 (用户批判):
  1. "等距映射链"——局部近等距 ≠ 全局近线性
     错误: J_l ≈ O(n) → 全局近线性
     正确: 36层小旋转累积 → 非平凡曲率，轨迹可完全偏离
     核心: 局部Jacobain谱 ≠ 全局几何

  2. "translation = 微小方向偏好的累积"——仍在向量空间思维
     错误: h_{l+1} = h_l + δ (additive model)
     正确: h_{l+1} = Φ_context(h_l) (attention-conditioned reparameterization)
     核心: 翻译是trajectory deformation，不是directional accumulation

  3. "softmax bifurcation"——低估了token competition geometry
     错误: 关注P(cat)的变化
     正确: 关键是z_cat - z_猫 (logit margin) 的sign flip
     核心: behavior jump = margin sign flip，不是probability emergence

Phase 105核心升级:
  从"方向累积"到"约束如何被逐层变成可读出边际"

严格化的核心量:
  1. Margin flow — z_en - z_zh 的逐层演化 (不是P(en)!)
  2. 曲率累积 — |h_{l+1} - 2h_l + h_{l-1}| (不是norm)
  3. 平行传输 — 翻译相关切向方向是否在层间保持方向
  4. 可读出性梯度 — 约束何时从"已编码但不可读"变成"可线性读出"

实验设计:
  Exp 1: Margin Dynamics — z_en - z_zh 的逐层精确演化
    对30个翻译词对，逐层计算:
    - z_en = h_l @ W_U[en_token] (英文token的logit)
    - z_zh = h_l @ W_U[zh_token] (中文token的logit)
    - margin = z_en - z_zh
    - margin通过LN前后的变化
    - 翻译prompt vs 中文prompt的margin差异

  Exp 2: Curvature Accumulation — 轨迹弯曲的逐层测量
    对30个词，计算:
    - 二阶差分: |h_{l+1} - 2*h_l + h_{l-1}| (曲率)
    - 曲率方向与翻译差分方向的对齐度
    - 翻译prompt vs 中文prompt的曲率差异
    - 累积曲率: sum(|h_{l+1} - 2*h_l + h_{l-1}|) vs 总位移 |h_L - h_0|

  Exp 3: Parallel Transport — 翻译相关方向在层间的传输
    核心问题: 翻译差分方向是否被平行传输？还是不断旋转？
    对每个词:
    - L0的翻译差分方向 d_0 = (h_0^trans - h_0^zh) / ||...||
    - 在L1: 计算d_0经过J_0后的变化: d_1 = J_0 @ d_0 (近似)
    - 同时计算L1的翻译差分方向: d_1^true = (h_1^trans - h_1^zh) / ||...||
    - 测量d_1和d_1^true的对齐度
    - 如果对齐度高→方向被平行传输(累积假说成立)
    - 如果对齐度低→方向被旋转(变形假说成立)

  Exp 4: Readout Accessibility — 约束何时变成可读出的？
    核心问题: 早期层已编码翻译约束，但晚期层才使其可线性读出
    对每个词:
    - 在每层，用线性探针(W_U)读出z_en和z_zh
    - margin = z_en - z_zh
    - 在翻译prompt下: margin应该从负变正→sign flip
    - 在中文prompt下: margin始终为负
    - 关键: margin的sign flip发生在哪层？
    - 同时: 在每层直接用SVM探针分类翻译vs中文 → 信息何时被编码？

  Exp 5: Minimum Control Energy with Margin Crossing
    不是找"翻译方向"(错误问题)
    而是找: 多小的扰动能让margin = z_en - z_zh 翻转sign？
    在不同层注入不同方向的扰动:
    - 翻译差分方向
    - W_U[en_token]方向 (decoder-aligned方向)
    - 翻译差分方向与W_U[en_token]的组合
    测量: 让margin sign flip的最小α

Run:
  python tests/glm5/ccml_phase105_margin_dynamics.py --model qwen3 --exp 1
  python tests/glm5/ccml_phase105_margin_dynamics.py --model qwen3 --exp 2
  python tests/glm5/ccml_phase105_margin_dynamics.py --model qwen3 --exp 3
  python tests/glm5/ccml_phase105_margin_dynamics.py --model qwen3 --exp 4
  python tests/glm5/ccml_phase105_margin_dynamics.py --model qwen3 --exp 5
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','..'))

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import gc
import json
import time
from collections import defaultdict

# Add the tests directory to path for model_utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_utils import load_model, get_layers, get_model_info, release_model, get_W_U


# ============================================================
# 测试数据 — 扩大到30个词对
# ============================================================
TRANSLATION_PAIRS = [
    ("猫", "cat"), ("狗", "dog"), ("水", "water"), ("火", "fire"),
    ("花", "flower"), ("鱼", "fish"), ("树", "tree"), ("鸟", "bird"),
    ("马", "horse"), ("铁", "iron"), ("金", "gold"), ("茶", "tea"),
    ("月", "moon"), ("日", "sun"), ("风", "wind"), ("雨", "rain"),
    ("雪", "snow"), ("石", "stone"), ("云", "cloud"), ("星", "star"),
    ("河", "river"), ("山", "mountain"), ("草", "grass"), ("光", "light"),
    ("血", "blood"), ("冰", "ice"), ("沙", "sand"), ("雾", "fog"),
    ("雷", "thunder"), ("霜", "frost"),
]


def get_token_id(tokenizer, text):
    """获取token的ID"""
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if ids else None


def get_all_hidden_states(model, tokenizer, device, prompt, n_layers):
    """获取所有层的hidden states"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(inputs["input_ids"], output_hidden_states=True)
    states = []
    for l in range(n_layers + 1):
        h = outputs.hidden_states[l][0, -1, :].float().cpu().numpy()
        states.append(h)
    return states


def apply_layer_norm(h, eps=1e-5):
    """手动应用LayerNorm"""
    mean = np.mean(h)
    std = np.std(h)
    return (h - mean) / (std + eps)


# ============================================================
# Exp 1: Margin Dynamics — z_en - z_zh 的逐层精确演化
# ============================================================
def exp1_margin_dynamics(args):
    model_name = args.model
    print(f"\n{'='*60}")
    print(f"Exp 1: Margin Dynamics — z_en - z_zh 逐层演化")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model

    # 获取W_U (float64 for precision)
    W_U = get_W_U(model).astype(np.float64)
    print(f"  W_U shape: {W_U.shape}, rank: {np.sum(np.linalg.svd(W_U, compute_uv=False) > 1e-5)}")

    results = {}
    for zh, en in TRANSLATION_PAIRS:
        en_id = get_token_id(tokenizer, en)
        zh_id = get_token_id(tokenizer, zh)
        if en_id is None or zh_id is None:
            continue

        # W_U的行向量 (logit权重)
        w_en = W_U[en_id]  # shape (d_model,)
        w_zh = W_U[zh_id]

        result = {"en_token": en, "zh_token": zh}

        # 3种prompt格式
        prompts = {
            "zh_continue": f"{zh}是一种",
            "trans_short": f'"{zh}"的英文是',
            "trans_instr": f'请把"{zh}"翻译成英文：',
        }

        for ptype, prompt in prompts.items():
            states = get_all_hidden_states(model, tokenizer, device, prompt, n_layers)
            layer_data = []

            for l in range(n_layers + 1):
                h_l = states[l].astype(np.float64)
                h_l_norm = apply_layer_norm(h_l)

                # 直接用h_l @ W_U[en_id]计算logit (不需要完整vocab投影)
                z_en_raw = np.dot(h_l, w_en)
                z_zh_raw = np.dot(h_l, w_zh)
                margin_raw = z_en_raw - z_zh_raw

                # LN后的logit
                z_en_ln = np.dot(h_l_norm, w_en)
                z_zh_ln = np.dot(h_l_norm, w_zh)
                margin_ln = z_en_ln - z_zh_ln

                # h_l到W_U方向的投影
                proj_en = np.dot(h_l, w_en) / (np.linalg.norm(w_en) + 1e-10)
                proj_zh = np.dot(h_l, w_zh) / (np.linalg.norm(w_zh) + 1e-10)

                layer_data.append({
                    "h_norm": float(np.linalg.norm(h_l)),
                    "z_en_raw": float(z_en_raw),
                    "z_zh_raw": float(z_zh_raw),
                    "margin_raw": float(margin_raw),
                    "z_en_ln": float(z_en_ln),
                    "z_zh_ln": float(z_zh_ln),
                    "margin_ln": float(margin_ln),
                    "proj_en_per_norm": float(proj_en / (np.linalg.norm(h_l) + 1e-10)),
                    "proj_zh_per_norm": float(proj_zh / (np.linalg.norm(h_l) + 1e-10)),
                })

            result[ptype] = layer_data

        results[f"{zh}_{en}"] = result
        print(f"  {zh}→{en}: zh_continue margin(L33)={result['zh_continue'][33]['margin_ln']:.4f}, "
              f"trans_short margin(L33)={result['trans_short'][33]['margin_ln']:.4f}")

    # 汇总分析
    print(f"\n  === Margin Flow Summary ===")
    for ptype in ["zh_continue", "trans_short", "trans_instr"]:
        margins = []
        margin_sign_flips = []  # margin从负变正的层
        for key, res in results.items():
            layer_margins = [res[ptype][l]["margin_ln"] for l in range(n_layers + 1)]
            margins.append(layer_margins)
            # 找sign flip
            for l in range(1, len(layer_margins)):
                if layer_margins[l-1] <= 0 and layer_margins[l] > 0:
                    margin_sign_flips.append(l)
                    break

        margins = np.array(margins)
        mean_margins = np.mean(margins, axis=0)
        print(f"\n  {ptype}:")
        print(f"    Margin at L0: {mean_margins[0]:.6f}")
        for l in [6, 12, 18, 21, 24, 27, 30, 33, 34, 35]:
            if l < len(mean_margins):
                print(f"    Margin at L{l}: {mean_margins[l]:.6f}")

        if margin_sign_flips:
            print(f"    Sign flip layers: {margin_sign_flips[:10]}... (mean={np.mean(margin_sign_flips):.1f})")
        else:
            print(f"    No sign flips detected (margin never crosses zero)")

    # 保存
    out_path = f"tests/glm5_temp/phase105_exp1_{model_name}_margin_dynamics.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved to {out_path}")

    release_model(model)
    return results


# ============================================================
# Exp 2: Curvature Accumulation — 轨迹弯曲的逐层测量
# ============================================================
def exp2_curvature_accumulation(args):
    model_name = args.model
    print(f"\n{'='*60}")
    print(f"Exp 2: Curvature Accumulation — 轨迹弯曲逐层测量")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    W_U = get_W_U(model).astype(np.float64)

    results = {}
    for zh, en in TRANSLATION_PAIRS:
        en_id = get_token_id(tokenizer, en)
        zh_id = get_token_id(tokenizer, zh)
        if en_id is None or zh_id is None:
            continue

        w_en = W_U[en_id]
        result = {"en_token": en, "zh_token": zh}

        prompts = {
            "zh_continue": f"{zh}是一种",
            "trans_short": f'"{zh}"的英文是',
        }

        for ptype, prompt in prompts.items():
            states = get_all_hidden_states(model, tokenizer, device, prompt, n_layers)
            layer_data = []

            for l in range(1, n_layers):  # L1 to L34
                h_prev = states[l-1].astype(np.float64)
                h_curr = states[l].astype(np.float64)
                h_next = states[l+1].astype(np.float64)

                # 二阶差分 (曲率)
                curvature = h_next - 2*h_curr + h_prev
                curvature_norm = np.linalg.norm(curvature)

                # 一阶差分 (速度)
                velocity = h_next - h_curr
                velocity_norm = np.linalg.norm(velocity)

                # 曲率/速度比
                curvature_ratio = curvature_norm / (velocity_norm + 1e-10)

                # 曲率方向与W_U[en]的对齐度
                if curvature_norm > 1e-10:
                    curv_dir = curvature / curvature_norm
                    align_en = abs(np.dot(curv_dir, w_en / (np.linalg.norm(w_en) + 1e-10)))
                else:
                    align_en = 0.0

                layer_data.append({
                    "curvature_norm": float(curvature_norm),
                    "velocity_norm": float(velocity_norm),
                    "curvature_ratio": float(curvature_ratio),
                    "align_en_wu": float(align_en),
                })

            # 总位移 vs 累积曲率
            total_displacement = np.linalg.norm(states[n_layers].astype(np.float64) - states[0].astype(np.float64))
            total_curvature = sum(ld["curvature_norm"] for ld in layer_data)
            total_velocity = sum(ld["velocity_norm"] for ld in layer_data)

            result[ptype] = {
                "layers": layer_data,
                "total_displacement": float(total_displacement),
                "total_curvature": float(total_curvature),
                "total_velocity": float(total_velocity),
                "curvature_displacement_ratio": float(total_curvature / (total_displacement + 1e-10)),
                "velocity_displacement_ratio": float(total_velocity / (total_displacement + 1e-10)),
            }

        # 两种prompt的曲率差
        zh_curv = [ld["curvature_norm"] for ld in result["zh_continue"]["layers"]]
        tr_curv = [ld["curvature_norm"] for ld in result["trans_short"]["layers"]]
        curvature_diff = [tr - zh for tr, zh in zip(tr_curv, zh_curv)]

        result["curvature_diff_trans_minus_zh"] = curvature_diff
        results[f"{zh}_{en}"] = result

        print(f"  {zh}→{en}: "
              f"zh curv/disp={result['zh_continue']['curvature_displacement_ratio']:.2f}, "
              f"trans curv/disp={result['trans_short']['curvature_displacement_ratio']:.2f}")

    # 汇总
    print(f"\n  === Curvature Summary ===")
    for ptype in ["zh_continue", "trans_short"]:
        ratios = [res[ptype]["curvature_displacement_ratio"] for res in results.values()]
        vel_ratios = [res[ptype]["velocity_displacement_ratio"] for res in results.values()]
        print(f"  {ptype}: mean curvature/disp={np.mean(ratios):.2f}, "
              f"mean velocity/disp={np.mean(vel_ratios):.2f}")

    # 逐层平均曲率差
    n_inner = n_layers - 1
    mean_curv_diff = np.zeros(n_inner)
    for res in results.values():
        cd = res["curvature_diff_trans_minus_zh"]
        for l in range(min(len(cd), n_inner)):
            mean_curv_diff[l] += cd[l]
    mean_curv_diff /= len(results)
    print(f"\n  Mean curvature diff (trans-zh) by layer:")
    for l in [5, 10, 15, 20, 25, 30, 33]:
        if l < n_inner:
            print(f"    L{l+1}: {mean_curv_diff[l]:.4f}")

    out_path = f"tests/glm5_temp/phase105_exp2_{model_name}_curvature.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved to {out_path}")

    release_model(model)
    return results


# ============================================================
# Exp 3: Parallel Transport — 翻译相关方向在层间的传输
# ============================================================
def exp3_parallel_transport(args):
    model_name = args.model
    print(f"\n{'='*60}")
    print(f"Exp 3: Parallel Transport — 翻译方向是否被平行传输？")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model

    results = {}
    for zh, en in TRANSLATION_PAIRS:
        result = {"en_token": en, "zh_token": zh}

        # 获取两种prompt的hidden states
        zh_prompt = f"{zh}是一种"
        trans_prompt = f'"{zh}"的英文是'

        states_zh = get_all_hidden_states(model, tokenizer, device, zh_prompt, n_layers)
        states_tr = get_all_hidden_states(model, tokenizer, device, trans_prompt, n_layers)

        # 每层的翻译差分方向
        diff_dirs = []
        for l in range(n_layers + 1):
            diff = states_tr[l].astype(np.float64) - states_zh[l].astype(np.float64)
            norm = np.linalg.norm(diff)
            if norm > 1e-10:
                diff_dirs.append(diff / norm)
            else:
                diff_dirs.append(np.zeros(d_model))

        # 分析连续层间翻译差分方向的旋转
        transport_data = []
        for l in range(n_layers):
            d_l = diff_dirs[l]
            d_l1 = diff_dirs[l + 1]

            # 方向保持度 (cosine similarity)
            cos_sim = np.dot(d_l, d_l1)
            # 角度变化
            angle_change = np.arccos(np.clip(cos_sim, -1, 1))

            # Δh方向 vs 翻译差分方向的对齐度
            delta_h_zh = states_zh[l+1].astype(np.float64) - states_zh[l].astype(np.float64)
            delta_h_tr = states_tr[l+1].astype(np.float64) - states_tr[l].astype(np.float64)

            delta_h_zh_norm = np.linalg.norm(delta_h_zh)
            delta_h_tr_norm = np.linalg.norm(delta_h_tr)

            if delta_h_zh_norm > 1e-10:
                align_zh = abs(np.dot(delta_h_zh / delta_h_zh_norm, d_l))
            else:
                align_zh = 0.0

            if delta_h_tr_norm > 1e-10:
                align_tr = abs(np.dot(delta_h_tr / delta_h_tr_norm, d_l))
            else:
                align_tr = 0.0

            # 翻译差分方向的Δh投影
            delta_diff = delta_h_tr - delta_h_zh
            delta_diff_norm = np.linalg.norm(delta_diff)
            if delta_diff_norm > 1e-10:
                align_delta_diff = abs(np.dot(delta_diff / delta_diff_norm, d_l))
            else:
                align_delta_diff = 0.0

            transport_data.append({
                "direction_cos_sim": float(cos_sim),
                "angle_change_deg": float(np.degrees(angle_change)),
                "delta_h_zh_align_diff_dir": float(align_zh),
                "delta_h_tr_align_diff_dir": float(align_tr),
                "delta_diff_align_diff_dir": float(align_delta_diff),
                "diff_norm": float(np.linalg.norm(states_tr[l].astype(np.float64) - states_zh[l].astype(np.float64))),
            })

        # 累积角度变化
        cumulative_angle = sum(td["angle_change_deg"] for td in transport_data)
        mean_angle = np.mean([td["angle_change_deg"] for td in transport_data])

        # 全局方向保持 (L0 vs L35)
        global_cos = np.dot(diff_dirs[0], diff_dirs[-1])

        result["transport"] = transport_data
        result["cumulative_angle_deg"] = float(cumulative_angle)
        result["mean_angle_change_deg"] = float(mean_angle)
        result["global_L0_L35_cos"] = float(global_cos)

        results[f"{zh}_{en}"] = result
        print(f"  {zh}→{en}: cumul_angle={cumulative_angle:.1f}°, "
              f"mean={mean_angle:.1f}°, global_cos={global_cos:.4f}")

    # 汇总
    print(f"\n  === Parallel Transport Summary ===")
    cumul_angles = [r["cumulative_angle_deg"] for r in results.values()]
    mean_angles = [r["mean_angle_change_deg"] for r in results.values()]
    global_cosines = [r["global_L0_L35_cos"] for r in results.values()]

    print(f"  Cumulative angle: mean={np.mean(cumul_angles):.1f}°, std={np.std(cumul_angles):.1f}°")
    print(f"  Mean angle/layer: mean={np.mean(mean_angles):.1f}°, std={np.std(mean_angles):.1f}°")
    print(f"  Global L0→L35 cosine: mean={np.mean(global_cosines):.4f}, std={np.std(global_cosines):.4f}")

    # 逐层平均角度变化
    n_inner = n_layers
    mean_angle_by_layer = np.zeros(n_inner)
    for r in results.values():
        for l in range(min(len(r["transport"]), n_inner)):
            mean_angle_by_layer[l] += r["transport"][l]["angle_change_deg"]
    mean_angle_by_layer /= len(results)

    print(f"\n  Mean angle change by layer (deg):")
    for l in [0, 5, 10, 15, 20, 25, 30, 33, 34]:
        if l < n_inner:
            print(f"    L{l}→L{l+1}: {mean_angle_by_layer[l]:.2f}°")

    # Δh与翻译差分方向的逐层对齐度
    print(f"\n  Mean delta_diff alignment with diff_dir:")
    for l in [0, 5, 10, 15, 20, 25, 30, 33]:
        if l < n_inner:
            aligns = [r["transport"][l]["delta_diff_align_diff_dir"] for r in results.values() if l < len(r["transport"])]
            if aligns:
                print(f"    L{l}→L{l+1}: {np.mean(aligns):.4f}")

    out_path = f"tests/glm5_temp/phase105_exp3_{model_name}_parallel_transport.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved to {out_path}")

    release_model(model)
    return results


# ============================================================
# Exp 4: Readout Accessibility — 约束何时变成可读出的？
# ============================================================
def exp4_readout_accessibility(args):
    model_name = args.model
    print(f"\n{'='*60}")
    print(f"Exp 4: Readout Accessibility — 约束何时可线性读出？")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    W_U = get_W_U(model).astype(np.float64)

    results = {}

    for zh, en in TRANSLATION_PAIRS:
        en_id = get_token_id(tokenizer, en)
        zh_id = get_token_id(tokenizer, zh)
        if en_id is None or zh_id is None:
            continue

        w_en = W_U[en_id]
        w_zh = W_U[zh_id]

        result = {"en_token": en, "zh_token": zh}

        prompts = {
            "zh_continue": f"{zh}是一种",
            "trans_short": f'"{zh}"的英文是',
        }

        for ptype, prompt in prompts.items():
            states = get_all_hidden_states(model, tokenizer, device, prompt, n_layers)
            layer_data = []

            for l in range(n_layers + 1):
                h_l = states[l].astype(np.float64)
                h_l_norm = apply_layer_norm(h_l)

                # 用W_U行做线性读出
                z_en_raw = np.dot(h_l, w_en)
                z_zh_raw = np.dot(h_l, w_zh)
                margin_raw = z_en_raw - z_zh_raw

                z_en_ln = np.dot(h_l_norm, w_en)
                z_zh_ln = np.dot(h_l_norm, w_zh)
                margin_ln = z_en_ln - z_zh_ln

                # 用全W_U投影做softmax概率
                logits = h_l_norm @ W_U.T
                # 只取top-k避免计算整个vocab
                en_logit = logits[en_id]
                zh_logit = logits[zh_id]

                # 相对logit
                max_logit = np.max(logits)
                exp_logits = np.exp(logits - max_logit)
                total = np.sum(exp_logits)
                p_en = exp_logits[en_id] / total
                p_zh = exp_logits[zh_id] / total

                # 可读出性指标: margin / h_norm
                h_norm = np.linalg.norm(h_l)
                readout_signal_per_norm = margin_ln / (h_norm + 1e-10)

                layer_data.append({
                    "margin_raw": float(margin_raw),
                    "margin_ln": float(margin_ln),
                    "p_en": float(p_en),
                    "p_zh": float(p_zh),
                    "readout_signal_per_norm": float(readout_signal_per_norm),
                    "h_norm": float(h_norm),
                })

            result[ptype] = layer_data

        # 关键分析: margin的sign flip位置
        for ptype in ["zh_continue", "trans_short"]:
            margins = [result[ptype][l]["margin_ln"] for l in range(n_layers + 1)]
            flip_layer = None
            for l in range(1, len(margins)):
                if margins[l-1] <= 0 and margins[l] > 0:
                    flip_layer = l
                    break
            result[f"{ptype}_margin_flip_layer"] = flip_layer

        # 翻译prompt vs 中文prompt的margin差异
        margin_diff = [result["trans_short"][l]["margin_ln"] - result["zh_continue"][l]["margin_ln"]
                       for l in range(n_layers + 1)]
        result["margin_diff_trans_minus_zh"] = margin_diff

        results[f"{zh}_{en}"] = result

        zh_flip = result["zh_continue_margin_flip_layer"]
        tr_flip = result["trans_short_margin_flip_layer"]
        print(f"  {zh}→{en}: zh_flip={zh_flip}, trans_flip={tr_flip}, "
              f"margin_diff_L33={margin_diff[33]:.4f}")

    # 汇总
    print(f"\n  === Readout Accessibility Summary ===")
    for ptype in ["zh_continue", "trans_short"]:
        flips = [r[f"{ptype}_margin_flip_layer"] for r in results.values()
                 if r[f"{ptype}_margin_flip_layer"] is not None]
        no_flips = sum(1 for r in results.values() if r[f"{ptype}_margin_flip_layer"] is None)

        if flips:
            print(f"  {ptype}: margin sign flip at L{np.mean(flips):.1f} (n={len(flips)}), no flip: {no_flips}")
        else:
            print(f"  {ptype}: no margin sign flips detected (n={no_flips})")

    # margin_diff的逐层均值
    n_total = len(results)
    mean_margin_diff = np.zeros(n_layers + 1)
    for r in results.values():
        md = r["margin_diff_trans_minus_zh"]
        for l in range(n_layers + 1):
            mean_margin_diff[l] += md[l]
    mean_margin_diff /= n_total

    print(f"\n  Mean margin diff (trans - zh) by layer:")
    for l in [0, 6, 12, 18, 21, 24, 27, 30, 33, 34, 35]:
        if l <= n_layers:
            print(f"    L{l}: {mean_margin_diff[l]:.6f}")

    out_path = f"tests/glm5_temp/phase105_exp4_{model_name}_readout.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved to {out_path}")

    release_model(model)
    return results


# ============================================================
# Exp 5: Minimum Control Energy with Margin Crossing
# ============================================================
def exp5_margin_control(args):
    model_name = args.model
    print(f"\n{'='*60}")
    print(f"Exp 5: Margin Control — 最小扰动让margin sign flip")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    W_U = get_W_U(model).astype(np.float64)

    # 测试词对
    test_pairs = TRANSLATION_PAIRS[:15]

    # 收集翻译差分方向
    print(f"\n  === Step 1: 收集翻译差分方向 ===")
    translation_diff_dirs = {}
    for l in [9, 15, 21, 27, 33]:
        if l >= n_layers:
            continue
        deltas = []
        for zh_t, en_t in test_pairs[:8]:
            trans_prompt = f'"{zh_t}"的英文是'
            zh_prompt = f"{zh_t}是一种"
            with torch.no_grad():
                out_tr = model(tokenizer(trans_prompt, return_tensors="pt").to(device)["input_ids"],
                              output_hidden_states=True)
                out_zh = model(tokenizer(zh_prompt, return_tensors="pt").to(device)["input_ids"],
                              output_hidden_states=True)
            h_tr = out_tr.hidden_states[l][0, -1, :].float().cpu().numpy()
            h_zh = out_zh.hidden_states[l][0, -1, :].float().cpu().numpy()
            deltas.append(h_tr - h_zh)
        mean_delta = np.mean(deltas, axis=0)
        norm = np.linalg.norm(mean_delta)
        if norm > 1e-10:
            translation_diff_dirs[l] = mean_delta / norm
        print(f"    L{l}: ||trans_diff_dir||={norm:.1f}")

    results = {}

    for zh, en in test_pairs:
        en_id = get_token_id(tokenizer, en)
        zh_id = get_token_id(tokenizer, zh)
        if en_id is None or zh_id is None:
            continue

        w_en = W_U[en_id]
        w_zh = W_U[zh_id]

        # W_U方向 (decoder-aligned)
        w_en_dir = w_en / (np.linalg.norm(w_en) + 1e-10)
        # margin方向
        margin_dir = (w_en - w_zh) / (np.linalg.norm(w_en - w_zh) + 1e-10)

        result = {"en_token": en, "zh_token": zh}

        # 中文prompt作为基础
        zh_prompt = f"{zh}是一种"
        inputs = tokenizer(zh_prompt, return_tensors="pt").to(device)

        test_layers = [9, 15, 21, 27, 33]
        test_layers = [l for l in test_layers if l < n_layers]

        # 方向类别
        direction_configs = {
            "trans_diff": lambda l: translation_diff_dirs.get(l, None),
            "wu_en_dir": lambda l: w_en_dir,
            "margin_dir": lambda l: margin_dir,
            "combined": lambda l: None,  # 特殊处理
        }

        alpha_search = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]

        for l in test_layers:
            layer_result = {}

            for dir_name, dir_func in direction_configs.items():
                if dir_name == "combined":
                    # 组合方向: 翻译差分 + W_U方向的加权平均
                    td = translation_diff_dirs.get(l, None)
                    if td is None:
                        continue
                    direction = (0.5 * td + 0.5 * w_en_dir)
                    direction = direction / (np.linalg.norm(direction) + 1e-10)
                else:
                    direction = dir_func(l)
                    if direction is None:
                        continue

                min_alpha_margin = None

                for alpha in alpha_search:
                    # 注入扰动
                    with torch.no_grad():
                        outputs = model(inputs["input_ids"], output_hidden_states=True)
                        h_l = outputs.hidden_states[l][0, -1, :].float().cpu().numpy()

                    # 加扰动
                    h_perturbed = h_l.astype(np.float64) + alpha * direction
                    # 应用LN
                    h_perturbed_norm = apply_layer_norm(h_perturbed)

                    # 计算margin
                    z_en = np.dot(h_perturbed_norm, w_en)
                    z_zh = np.dot(h_perturbed_norm, w_zh)
                    margin = z_en - z_zh

                    # 检查margin sign flip
                    # 先获取原始margin
                    with torch.no_grad():
                        outputs_orig = model(inputs["input_ids"], output_hidden_states=True)
                        h_l_orig = outputs_orig.hidden_states[l][0, -1, :].float().cpu().numpy()
                    h_orig_norm = apply_layer_norm(h_l_orig.astype(np.float64))
                    margin_orig = np.dot(h_orig_norm, w_en) - np.dot(h_orig_norm, w_zh)

                    if margin_orig <= 0 and margin > 0:
                        min_alpha_margin = alpha
                        break

                layer_result[dir_name] = {
                    "min_alpha_margin_flip": min_alpha_margin,
                    "margin_orig": float(margin_orig),
                    "h_l_norm": float(np.linalg.norm(h_l_orig)),
                }

            result[f"L{l}"] = layer_result

        results[f"{zh}_{en}"] = result

        # 打印
        best_dir = None
        best_alpha = float('inf')
        for l in test_layers:
            lr = result.get(f"L{l}", {})
            for dn, dv in lr.items():
                if dv["min_alpha_margin_flip"] is not None and dv["min_alpha_margin_flip"] < best_alpha:
                    best_alpha = dv["min_alpha_margin_flip"]
                    best_dir = f"L{l}/{dn}"

        print(f"  {zh}→{en}: best={best_dir} α={best_alpha}" if best_dir else
              f"  {zh}→{en}: no margin flip achieved")

    # 汇总
    print(f"\n  === Margin Control Summary ===")
    for dn in ["trans_diff", "wu_en_dir", "margin_dir", "combined"]:
        successes = 0
        alphas = []
        for r in results.values():
            for lk, lv in r.items():
                if isinstance(lv, dict) and dn in lv:
                    if lv[dn]["min_alpha_margin_flip"] is not None:
                        successes += 1
                        alphas.append(lv[dn]["min_alpha_margin_flip"])
        if alphas:
            print(f"  {dn}: {successes} successes, mean α={np.mean(alphas):.1f}, min α={min(alphas)}")
        else:
            print(f"  {dn}: no margin flips achieved")

    out_path = f"tests/glm5_temp/phase105_exp5_{model_name}_margin_control.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved to {out_path}")

    release_model(model)
    return results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3", "glm4", "ds7b"])
    parser.add_argument("--exp", type=int, required=True, choices=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    if args.exp == 1:
        exp1_margin_dynamics(args)
    elif args.exp == 2:
        exp2_curvature_accumulation(args)
    elif args.exp == 3:
        exp3_parallel_transport(args)
    elif args.exp == 4:
        exp4_readout_accessibility(args)
    elif args.exp == 5:
        exp5_margin_control(args)

    gc.collect()
    torch.cuda.empty_cache()
