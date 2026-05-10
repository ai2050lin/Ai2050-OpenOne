"""
Phase 109: 计算轨迹动力学 — 扰动传输、轨迹发散与零假设检验
==========================================================

Phase 108的硬伤 (用户批判):
  1. "Flow正交81-97°可能是高维几何伪象"
     在d=2560空间中, 随机向量夹角天然≈90° (concentration of measure)
     81-97°未必说明计算类型变化, 可能只是零假设!
     需要: 计算零分布, 检验观测值是否显著

  2. "LN反转是过度拟人化"
     LN只做centering + scaling, 不是"语义反转器"
     alignment=-0.96可能只是投影几何变化, 不等于"网络故意反转信号"
     需要: 更仔细分析LN的数学效应

  3. "应做扰动传输存活分析" (最重要)
     真正关键不是"某层方向长什么样"
     而是"某层扰动→最终行为变化多少"
     这才是真正functional dynamics

  4. "PR=44≠模型只有44维"
     低能方向可能承载控制信号、路由约束、稀有token判别
     需要验证: 低能方向是否功能上重要

  5. "应研究计算重分布而非信号传输"
     核心模式: 局部集中 → 高维分散混合 → 条件化重新投影
     不是: 稳定语义方向传播

Phase 109核心升级:
  从"几何描述"到"功能验证"
  用扰动实验检验哪些方向真正功能重要

关键实验:
  Exp 1: Flow Angle Null Hypothesis Test
    核心: 在d=2560空间中, N个随机向量的夹角期望是多少?
    对比: 观测到的flow angle vs 零分布
    如果观测值不显著 → flow正交结论无意义
    如果观测值显著 → flow正交确实反映计算结构

  Exp 2: Perturbation Transport Survival
    核心: 在L_l注入小扰动ε·v, 看L36的logit变化量
    v方向: 随机方向 vs 翻译差分方向 vs margin方向 vs W_U主方向
    如果翻译差分方向的扰动存活率显著高于随机 → 信号确实在功能上存在
    如果所有方向存活率类似 → 高维空间中扰动均匀衰减
    这才是真正验证"信息存在"的方法

  Exp 3: Trajectory Functional Divergence
    核心: 改变输入1个token, 追踪轨迹何时功能性发散
    功能性 = 对最终logit的影响, 不是几何距离
    问题: "猫是一种" vs "狗是一种" 的轨迹在哪层开始功能性分离?

  Exp 4: Reconcentration Verification
    核心: 用更严格方法验证"集中→分散→重新集中"模式
    用per-layer PR和信号熵, 加上bootstrap置信区间
    同时检查: 重新集中是否只是LN的数学效应?

Run:
  python tests/glm5/ccml_phase109_trajectory_dynamics.py --model qwen3 --exp 1
  python tests/glm5/ccml_phase109_trajectory_dynamics.py --model qwen3 --exp 2
  python tests/glm5/ccml_phase109_trajectory_dynamics.py --model qwen3 --exp 3
  python tests/glm5/ccml_phase109_trajectory_dynamics.py --model qwen3 --exp 4
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import gc
import json
import time
from collections import defaultdict
from scipy.linalg import subspace_angles

from model_utils import load_model, get_layers, get_model_info, release_model, get_W_U


# ============================================================
# 测试数据 — 按语义域分组
# ============================================================
ANIMAL_PAIRS = [
    ("猫", "cat"), ("狗", "dog"), ("鱼", "fish"), ("鸟", "bird"),
    ("马", "horse"), ("牛", "cow"), ("羊", "sheep"), ("猪", "pig"),
    ("鸡", "chicken"), ("鸭", "duck"),
]

NATURE_PAIRS = [
    ("水", "water"), ("火", "fire"), ("风", "wind"), ("雨", "rain"),
    ("雪", "snow"), ("冰", "ice"), ("雷", "thunder"), ("雾", "fog"),
    ("霜", "frost"), ("云", "cloud"),
]

OBJECT_PAIRS = [
    ("花", "flower"), ("树", "tree"), ("石", "stone"), ("铁", "iron"),
    ("金", "gold"), ("茶", "tea"), ("沙", "sand"), ("草", "grass"),
    ("血", "blood"), ("光", "light"),
]

CELESTIAL_PAIRS = [
    ("月", "moon"), ("日", "sun"), ("星", "star"), ("河", "river"),
    ("山", "mountain"), ("海", "sea"), ("天", "sky"), ("地", "earth"),
    ("夜", "night"), ("昼", "day"),
]

ALL_PAIRS = ANIMAL_PAIRS + NATURE_PAIRS + OBJECT_PAIRS + CELESTIAL_PAIRS  # 40词对

EXTRA_PAIRS = [
    ("红", "red"), ("蓝", "blue"), ("绿", "green"), ("白", "white"),
    ("黑", "black"), ("大", "big"), ("小", "small"), ("长", "long"),
    ("短", "short"), ("新", "new"), ("旧", "old"), ("快", "fast"),
    ("慢", "slow"), ("高", "tall"), ("低", "low"), ("热", "hot"),
    ("冷", "cold"), ("甜", "sweet"), ("苦", "bitter"), ("酸", "sour"),
]

ALL_PLUS_EXTRA = ALL_PAIRS + EXTRA_PAIRS  # 60词对


def get_token_id(tokenizer, text):
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if ids else None


def apply_layer_norm(h, eps=1e-5):
    mean = np.mean(h)
    std = np.std(h)
    return (h - mean) / (std + eps)


def collect_hidden_states(model, tokenizer, device, pairs, prompt_types=None):
    """收集所有词对在各层的hidden states"""
    if prompt_types is None:
        prompt_types = ["zh", "trans"]

    model_info = get_model_info(model, model.name if hasattr(model, 'name') else 'qwen3')
    n_layers = model_info.n_layers

    layer_states = defaultdict(lambda: {"zh": [], "trans": []})

    for zh, en in pairs:
        prompts = {
            "zh": f"{zh}是一种",
            "trans": f'"{zh}"的英文是',
        }
        for ptype in prompt_types:
            prompt = prompts[ptype]
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(inputs["input_ids"], output_hidden_states=True)
            for l in range(n_layers + 1):
                h = outputs.hidden_states[l][0, -1, :].float().cpu().numpy()
                layer_states[l][ptype].append(h)

    return dict(layer_states)


def collect_hidden_states_single(model, tokenizer, device, prompt):
    """收集单个prompt在各层的hidden states"""
    model_info = get_model_info(model, model.name if hasattr(model, 'name') else 'qwen3')
    n_layers = model_info.n_layers
    
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(inputs["input_ids"], output_hidden_states=True)
    
    states = {}
    for l in range(n_layers + 1):
        states[l] = outputs.hidden_states[l][0, -1, :].float().cpu().numpy()
    
    return states


# ============================================================
# Exp 1: Flow Angle Null Hypothesis Test
# ============================================================
def exp1_flow_angle_null(args):
    model_name = args.model
    print(f"\n{'='*60}")
    print("Exp 1: Flow Angle Null Hypothesis Test — flow正交是零假设吗?")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    W_U = get_W_U(model).astype(np.float64)

    # 收集hidden states
    all_pairs = ALL_PLUS_EXTRA  # 60词对
    print(f"\n  收集{len(all_pairs)}个词对的hidden states...")
    layer_states = collect_hidden_states(model, tokenizer, device, all_pairs)
    
    # 计算观测的flow angles
    print(f"\n  计算观测的层间flow angles...")
    observed_angles = {}
    
    for l in range(n_layers):
        zh_data = np.array(layer_states[l]["zh"], dtype=np.float64)
        trans_data = np.array(layer_states[l]["trans"], dtype=np.float64)
        zh_next = np.array(layer_states[l + 1]["zh"], dtype=np.float64)
        trans_next = np.array(layer_states[l + 1]["trans"], dtype=np.float64)
        
        # 平均flow方向
        flow_l = np.mean(zh_next + trans_next - zh_data - trans_data, axis=0) / 2
        flow_l_next = None  # 需要下一层
        
        if l < n_layers - 1:
            zh_next2 = np.array(layer_states[l + 2]["zh"], dtype=np.float64)
            trans_next2 = np.array(layer_states[l + 2]["trans"], dtype=np.float64)
            flow_l_next = np.mean(zh_next2 + trans_next2 - zh_next - trans_next, axis=0) / 2
        
        if flow_l_next is not None:
            # 两个flow向量之间的角度
            cos_angle = np.dot(flow_l, flow_l_next) / (np.linalg.norm(flow_l) * np.linalg.norm(flow_l_next) + 1e-10)
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            observed_angles[l] = angle
            
            if l % 6 == 0 or l >= n_layers - 3:
                print(f"    L{l}→L{l+1} vs L{l+1}→L{l+2}: angle={angle:.1f}°")
    
    # ========================================
    # A. 零假设: 随机向量的夹角分布
    # ========================================
    print(f"\n  === 零假设检验 ===")
    print(f"  d_model = {d_model}")
    
    # 理论期望: 在d维球面上, 两个随机向量的夹角期望
    # E[cos(θ)] = 0, Var[cos(θ)] = 1/d
    # 所以 θ 期望 ≈ 90°, 标准差 ≈ 180/π * 1/√d ≈ 57.3/√d 度
    std_theory = 57.3 / np.sqrt(d_model)
    print(f"\n  理论零分布 (d={d_model}):")
    print(f"    E[angle] = 90°")
    print(f"    Std[angle] ≈ {std_theory:.2f}°")
    print(f"    95% CI ≈ [90 - 1.96*{std_theory:.2f}, 90 + 1.96*{std_theory:.2f}] = "
          f"[{90 - 1.96*std_theory:.1f}°, {90 + 1.96*std_theory:.1f}°]")
    
    # 蒙特卡洛验证
    print(f"\n  蒙特卡洛验证 (10000次随机):")
    n_mc = 10000
    mc_angles = []
    for _ in range(n_mc):
        v1 = np.random.randn(d_model)
        v2 = np.random.randn(d_model)
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)
        cos_a = np.clip(np.dot(v1, v2), -1, 1)
        mc_angles.append(np.degrees(np.arccos(cos_a)))
    
    mc_angles = np.array(mc_angles)
    print(f"    Mean = {np.mean(mc_angles):.2f}°")
    print(f"    Std = {np.std(mc_angles):.2f}°")
    print(f"    95% CI = [{np.percentile(mc_angles, 2.5):.1f}°, {np.percentile(mc_angles, 97.5):.1f}°]")
    print(f"    99% CI = [{np.percentile(mc_angles, 0.5):.1f}°, {np.percentile(mc_angles, 99.5):.1f}°]")
    
    # ========================================
    # B. 更关键的零假设: flow向量不是随机向量
    # flow = h_{l+1} - h_l, 这个差分在d=2560维中也不是随机的
    # 因为h_{l+1}和h_l高度相关, 差分可能集中在特定子空间
    # ========================================
    print(f"\n  === 关键零假设: flow向量的夹角 ===")
    print(f"  如果flow集中在k维子空间, 则flow之间的夹角应该参考d=k的零分布")
    
    # 计算每层flow的有效维度
    flow_PRs = {}
    for l in range(n_layers):
        zh_data = np.array(layer_states[l]["zh"], dtype=np.float64)
        trans_data = np.array(layer_states[l]["trans"], dtype=np.float64)
        zh_next = np.array(layer_states[l + 1]["zh"], dtype=np.float64)
        trans_next = np.array(layer_states[l + 1]["trans"], dtype=np.float64)
        
        # 每个样本的flow
        flows_zh = zh_next - zh_data  # (n, d)
        flows_trans = trans_next - trans_data  # (n, d)
        all_flows = np.vstack([flows_zh, flows_trans])  # (2n, d)
        
        # flow的协方差矩阵的PR
        cov = np.cov(all_flows.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 0]
        if len(eigenvalues) > 0 and np.sum(eigenvalues**2) > 0:
            PR = (np.sum(eigenvalues)**2) / np.sum(eigenvalues**2)
        else:
            PR = 0
        flow_PRs[l] = PR
    
    print(f"\n  Flow的有效维度 (Participation Ratio):")
    for l in range(0, n_layers, 6):
        print(f"    L{l}: PR={flow_PRs[l]:.1f}")
    print(f"    L{n_layers-1}: PR={flow_PRs[n_layers-1]:.1f}")
    
    # 用flow PR作为有效维度, 计算修正后的零分布
    print(f"\n  修正零分布 (用flow PR作为有效维度):")
    for l in [0, 6, 12, 18, 24, 30, n_layers - 1]:
        if l in observed_angles:
            eff_d = max(flow_PRs[l], 2)
            std_eff = 57.3 / np.sqrt(eff_d)
            observed = observed_angles[l]
            # z-score: 偏离90°多少个标准差
            z_score = (observed - 90) / std_eff
            
            # 蒙特卡洛验证
            mc_eff = []
            for _ in range(5000):
                v1 = np.random.randn(int(eff_d))
                v2 = np.random.randn(int(eff_d))
                v1 /= np.linalg.norm(v1)
                v2 /= np.linalg.norm(v2)
                cos_a = np.clip(np.dot(v1, v2), -1, 1)
                mc_eff.append(np.degrees(np.arccos(cos_a)))
            
            mc_eff = np.array(mc_eff)
            p_value = np.mean(np.abs(mc_eff - 90) >= np.abs(observed - 90))
            
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
            print(f"    L{l}: observed={observed:.1f}°, eff_d={eff_d:.0f}, "
                  f"null_std={std_eff:.1f}°, z={z_score:.2f}, p={p_value:.4f} {sig}")
    
    # ========================================
    # C. 单样本flow角度 vs 群体平均flow角度
    # ========================================
    print(f"\n  === 单样本flow角度分析 ===")
    print(f"  之前只算了平均flow的角度, 现在看单样本flow角度的分布")
    
    for l in [0, 6, 12, 18, 24, 30, n_layers - 2]:
        if l >= n_layers - 1:
            continue
        zh_data = np.array(layer_states[l]["zh"], dtype=np.float64)
        trans_data = np.array(layer_states[l]["trans"], dtype=np.float64)
        zh_next = np.array(layer_states[l + 1]["zh"], dtype=np.float64)
        trans_next = np.array(layer_states[l + 1]["trans"], dtype=np.float64)
        
        # 每个样本的flow
        flows_zh = zh_next - zh_data  # (n, d)
        flows_trans = trans_next - trans_data  # (n, d)
        all_flows = np.vstack([flows_zh, flows_trans])
        
        # 计算所有flow对之间的角度
        # 随机采样避免O(n²)
        n_sample = min(50, len(all_flows))
        sample_idx = np.random.choice(len(all_flows), n_sample, replace=False)
        sampled_flows = all_flows[sample_idx]
        
        pair_angles = []
        for i in range(len(sampled_flows)):
            for j in range(i + 1, min(i + 10, len(sampled_flows))):  # 每个flow和10个其他flow比较
                v1 = sampled_flows[i]
                v2 = sampled_flows[j]
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 > 1e-10 and n2 > 1e-10:
                    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)
                    pair_angles.append(np.degrees(np.arccos(cos_a)))
        
        if pair_angles:
            pair_angles = np.array(pair_angles)
            print(f"    L{l}: mean_angle={np.mean(pair_angles):.1f}°, "
                  f"std={np.std(pair_angles):.1f}°, "
                  f"median={np.median(pair_angles):.1f}°, "
                  f"%<60°={100*np.mean(pair_angles < 60):.1f}%")

    # ========================================
    # D. 更直接的方法: flow的一致性(concentration)
    # 如果flow真的集中在某个方向, 那归一化后的flow之间的cos应该>0
    # 如果flow接近随机, 归一化后的flow之间的cos≈0
    # ========================================
    print(f"\n  === Flow concentration (一致性) ===")
    print(f"  如果flow方向一致 → mean pairwise cos > 0 → 角度 < 90°")
    print(f"  如果flow方向随机 → mean pairwise cos ≈ 0 → 角度 ≈ 90°")
    
    flow_concentration = {}
    for l in range(n_layers):
        zh_data = np.array(layer_states[l]["zh"], dtype=np.float64)
        trans_data = np.array(layer_states[l]["trans"], dtype=np.float64)
        zh_next = np.array(layer_states[l + 1]["zh"], dtype=np.float64)
        trans_next = np.array(layer_states[l + 1]["trans"], dtype=np.float64)
        
        flows_zh = zh_next - zh_data
        flows_trans = trans_next - trans_data
        all_flows = np.vstack([flows_zh, flows_trans])
        
        # 归一化
        norms = np.linalg.norm(all_flows, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        normalized = all_flows / norms
        
        # 平均归一化flow → 一致性
        mean_normalized = np.mean(normalized, axis=0)
        concentration = np.linalg.norm(mean_normalized)  # 0=完全随机, 1=完全一致
        
        flow_concentration[l] = float(concentration)
        
        if l % 6 == 0 or l >= n_layers - 3:
            print(f"    L{l}: concentration={concentration:.4f} "
                  f"(0=random, 1=perfect alignment)")

    # ========================================
    # E. 总结
    # ========================================
    print(f"\n  === 总结 ===")
    print(f"  零分布 (d={d_model}): mean=90°, std≈{std_theory:.1f}°")
    print(f"  观测flow angles: ", end="")
    for l in sorted(observed_angles.keys()):
        if l % 6 == 0 or l >= n_layers - 3:
            print(f"L{l}={observed_angles[l]:.0f}° ", end="")
    print()
    print(f"  Flow concentration (一致性):")
    for l in sorted(flow_concentration.keys()):
        if l % 6 == 0 or l >= n_layers - 3:
            c = flow_concentration[l]
            print(f"    L{l}: {c:.4f} ({'有结构' if c > 0.1 else '接近随机'})")

    results = {
        "null_distribution": {
            "d_model": d_model,
            "theory_mean": 90.0,
            "theory_std": float(std_theory),
            "mc_mean": float(np.mean(mc_angles)),
            "mc_std": float(np.std(mc_angles)),
            "mc_95ci": [float(np.percentile(mc_angles, 2.5)), float(np.percentile(mc_angles, 97.5))],
        },
        "observed_angles": {str(k): float(v) for k, v in observed_angles.items()},
        "flow_PRs": {str(k): float(v) for k, v in flow_PRs.items()},
        "flow_concentration": {str(k): v for k, v in flow_concentration.items()},
    }

    out_path = f"tests/glm5_temp/phase109_exp1_{model_name}_flow_null.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  保存到 {out_path}")

    release_model(model)
    return results


# ============================================================
# Exp 2: Perturbation Transport Survival
# ============================================================
def exp2_perturbation_survival(args):
    model_name = args.model
    print(f"\n{'='*60}")
    print("Exp 2: Perturbation Transport Survival — 扰动传输存活分析")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    W_U = get_W_U(model).astype(np.float64)

    # W_U的右奇异向量
    _, S_dec, Vt_dec = np.linalg.svd(W_U, full_matrices=False)
    dec_directions = Vt_dec.T  # (d_model, d_model)

    # 选几个代表性的词对
    test_pairs = [("猫", "cat"), ("水", "water"), ("红", "red"), ("月", "moon"),
                  ("火", "fire"), ("大", "big"), ("花", "flower"), ("星", "star")]

    # 收集hidden states (baseline)
    print(f"\n  收集{len(test_pairs)}个词对的baseline hidden states...")
    layer_states = collect_hidden_states(model, tokenizer, device, test_pairs)

    # ========================================
    # 核心: 在L_l注入扰动, 看L36的logit变化
    # ========================================
    print(f"\n  扰动传输分析...")
    
    epsilon = 0.1  # 扰动大小 (相对于hidden state范数)
    perturbation_layers = list(range(0, n_layers, 3)) + [n_layers - 3, n_layers - 2, n_layers - 1]
    perturbation_layers = sorted(set([l for l in perturbation_layers if l < n_layers]))

    # 定义扰动方向类型
    direction_types = ["random", "translation_diff", "W_U_top1", "W_U_top10_avg", "translation_diff_LN"]

    survival_results = {}

    for l_perturb in perturbation_layers:
        print(f"\n  === 在L{l_perturb}注入扰动 ===")
        l_survival = {}

        zh_data = np.array(layer_states[l_perturb]["zh"], dtype=np.float64)
        trans_data = np.array(layer_states[l_perturb]["trans"], dtype=np.float64)
        
        # 翻译差分方向 (各词对的平均)
        mean_diff = np.mean(trans_data - zh_data, axis=0)
        diff_norm = np.linalg.norm(mean_diff)
        if diff_norm > 0:
            trans_diff_dir = mean_diff / diff_norm
        else:
            trans_diff_dir = np.zeros(d_model)

        # LN后的翻译差分方向
        diff_ln = np.array([apply_layer_norm(trans_data[i] - zh_data[i]) for i in range(len(test_pairs))])
        mean_diff_ln = np.mean(diff_ln, axis=0)
        diff_ln_norm = np.linalg.norm(mean_diff_ln)
        if diff_ln_norm > 0:
            trans_diff_ln_dir = mean_diff_ln / diff_ln_norm
        else:
            trans_diff_ln_dir = np.zeros(d_model)

        # W_U top-1方向
        wu_top1_dir = dec_directions[:, 0]
        
        # W_U top-10平均方向
        wu_top10_dir = np.mean(dec_directions[:, :10], axis=1)
        wu_top10_norm = np.linalg.norm(wu_top10_dir)
        if wu_top10_norm > 0:
            wu_top10_dir = wu_top10_dir / wu_top10_norm

        directions = {
            "random": lambda: np.random.randn(d_model) / np.sqrt(d_model),
            "translation_diff": lambda: trans_diff_dir,
            "W_U_top1": lambda: wu_top1_dir,
            "W_U_top10_avg": lambda: wu_top10_dir,
            "translation_diff_LN": lambda: trans_diff_ln_dir,
        }

        for dir_name, dir_fn in directions.items():
            # 对每个词对做扰动
            survivals_zh = []
            survivals_trans = []
            margin_changes = []
            
            for i, (zh, en) in enumerate(test_pairs):
                for ptype in ["zh", "trans"]:
                    prompt = f"{zh}是一种" if ptype == "zh" else f'"{zh}"的英文是'
                    
                    # 获取baseline
                    inputs = tokenizer(prompt, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = model(inputs["input_ids"], output_hidden_states=True)
                    
                    # baseline logits (用L36 hidden state)
                    h_baseline = outputs.hidden_states[n_layers][0, -1, :].float().cpu().numpy()
                    logits_baseline = h_baseline @ W_U.T
                    
                    # 翻译token和中文token的margin
                    en_id = get_token_id(tokenizer, en)
                    zh_id = get_token_id(tokenizer, zh)
                    
                    if en_id is None or zh_id is None:
                        continue
                    
                    margin_baseline = logits_baseline[en_id] - logits_baseline[zh_id]
                    
                    # 注入扰动
                    perturbation_dir = dir_fn()
                    
                    # 对于random方向, 使用同样的随机种子
                    if dir_name == "random":
                        # 做3次随机取平均
                        rand_survivals = []
                        rand_margin_changes = []
                        for rand_trial in range(3):
                            rand_dir = np.random.randn(d_model)
                            rand_dir /= np.linalg.norm(rand_dir)
                            
                            # 在L_l_perturb注入扰动
                            perturbed_h = outputs.hidden_states[l_perturb][0, -1, :].float().cpu().numpy().copy()
                            h_norm = np.linalg.norm(perturbed_h)
                            perturbed_h += epsilon * h_norm * rand_dir
                            
                            # 需要重新forward从l_perturb开始
                            # 用hook方式: 在指定层替换hidden state
                            survival = _compute_perturbation_survival(
                                model, tokenizer, device, prompt, 
                                l_perturb, perturbed_h, n_layers, W_U,
                                en_id, zh_id, margin_baseline
                            )
                            rand_survivals.append(survival["survival_ratio"])
                            rand_margin_changes.append(survival["margin_change"])
                        
                        survivals_zh.append(np.mean(rand_survivals) if ptype == "zh" else 0)
                        survivals_trans.append(np.mean(rand_survivals) if ptype == "trans" else np.mean(rand_survivals))
                        margin_changes.append(np.mean(rand_margin_changes))
                    else:
                        perturbed_h = outputs.hidden_states[l_perturb][0, -1, :].float().cpu().numpy().copy()
                        h_norm = np.linalg.norm(perturbed_h)
                        perturbed_h += epsilon * h_norm * perturbation_dir
                        
                        survival = _compute_perturbation_survival(
                            model, tokenizer, device, prompt,
                            l_perturb, perturbed_h, n_layers, W_U,
                            en_id, zh_id, margin_baseline
                        )
                        survivals_zh.append(survival["survival_ratio"] if ptype == "zh" else 0)
                        survivals_trans.append(survival["survival_ratio"] if ptype == "trans" else survival["survival_ratio"])
                        margin_changes.append(survival["margin_change"])

            avg_survival = np.mean([s for s in survivals_zh + survivals_trans if s > 0]) if any(s > 0 for s in survivals_zh + survivals_trans) else 0
            avg_margin_change = np.mean(margin_changes) if margin_changes else 0
            
            l_survival[dir_name] = {
                "avg_survival_ratio": float(avg_survival),
                "avg_margin_change": float(avg_margin_change),
            }
            print(f"    {dir_name}: survival={avg_survival:.4f}, margin_change={avg_margin_change:.4f}")

        survival_results[l_perturb] = l_survival

    # ========================================
    # 对比: 翻译差分方向 vs 随机方向
    # ========================================
    print(f"\n  === 扰动存活率对比 ===")
    print(f"  {'Layer':<8} {'Random':<12} {'TransDiff':<12} {'W_U_top1':<12} {'TransDiff/Random':<16}")
    for l in perturbation_layers:
        if l in survival_results:
            r = survival_results[l]
            rand_s = r.get("random", {}).get("avg_survival_ratio", 0)
            diff_s = r.get("translation_diff", {}).get("avg_survival_ratio", 0)
            wu_s = r.get("W_U_top1", {}).get("avg_survival_ratio", 0)
            ratio = diff_s / rand_s if rand_s > 0 else float('inf')
            print(f"  L{l:<6} {rand_s:<12.4f} {diff_s:<12.4f} {wu_s:<12.4f} {ratio:<16.2f}")

    results = {
        "epsilon": epsilon,
        "perturbation_layers": perturbation_layers,
        "survival_by_layer_and_direction": {str(k): v for k, v in survival_results.items()},
    }

    out_path = f"tests/glm5_temp/phase109_exp2_{model_name}_perturbation_survival.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  保存到 {out_path}")

    release_model(model)
    return results


def _compute_perturbation_survival(model, tokenizer, device, prompt, 
                                    l_perturb, perturbed_h, n_layers, W_U,
                                    en_id, zh_id, margin_baseline):
    """在L_l_perturb注入扰动后, 计算扰动在L36的存活率和margin变化
    
    使用hook方式: 在前向传播中替换指定层的输出
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # 使用hook注入扰动
    perturbation_applied = [False]
    
    def hook_fn(module, input, output):
        if not perturbation_applied[0]:
            perturbation_applied[0] = True
            # output是tuple, 第一个是hidden_states
            if isinstance(output, tuple):
                hs = output[0].clone()
                # 替换最后一个token的hidden state
                hs[0, -1, :] = torch.tensor(perturbed_h, dtype=hs.dtype, device=hs.device)
                return (hs,) + output[1:]
            return output
        return output
    
    # 找到对应的层
    layers = get_layers(model)
    handle = layers[l_perturb].register_forward_hook(hook_fn)
    
    try:
        with torch.no_grad():
            outputs = model(inputs["input_ids"], output_hidden_states=True)
    finally:
        handle.remove()
    
    # L36的hidden state
    h_final = outputs.hidden_states[n_layers][0, -1, :].float().cpu().numpy()
    logits_perturbed = h_final @ W_U.T
    
    margin_perturbed = logits_perturbed[en_id] - logits_perturbed[zh_id]
    
    # 存活率: 扰动在最终层的大小 / 注入时的大小
    # 近似: 用margin变化作为功能存活率
    margin_change = margin_perturbed - margin_baseline
    
    # 同时用L2距离衡量几何存活率
    # (这里没有baseline的h_final, 所以用margin_change作为功能存活率代理)
    
    return {
        "survival_ratio": float(abs(margin_change) / (abs(margin_baseline) + 1e-10)),
        "margin_change": float(margin_change),
        "margin_baseline": float(margin_baseline),
        "margin_perturbed": float(margin_perturbed),
    }


# ============================================================
# Exp 3: Trajectory Functional Divergence
# ============================================================
def exp3_trajectory_divergence(args):
    model_name = args.model
    print(f"\n{'='*60}")
    print("Exp 3: Trajectory Functional Divergence — 轨迹功能发散图")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    W_U = get_W_U(model).astype(np.float64)

    # 设计词对组: 语义相近 vs 语义远 vs 翻译vs中文
    print(f"\n  定义对比组...")

    # 组1: 语义相近的中文词 (猫 vs 狗)
    similar_zh = [("猫", "狗"), ("水", "火"), ("红", "蓝"), ("大", "小"), ("春", "秋")]
    
    # 组2: 语义远的中文词 (猫 vs 山)
    distant_zh = [("猫", "山"), ("水", "星"), ("红", "铁"), ("大", "夜"), ("春", "石")]
    
    # 组3: 同一个词的中文prompt vs 翻译prompt
    trans_vs_zh = [("猫", "cat"), ("水", "water"), ("红", "red"), ("大", "big"), ("月", "moon")]

    def compute_functional_divergence(prompt1, prompt2, label1, label2):
        """计算两个prompt在每层的功能性发散"""
        states1 = collect_hidden_states_single(model, tokenizer, device, prompt1)
        states2 = collect_hidden_states_single(model, tokenizer, device, prompt2)
        
        divergence = {}
        for l in range(n_layers + 1):
            h1 = states1[l]
            h2 = states2[l]
            
            # 几何距离 (归一化)
            h1_norm = h1 / (np.linalg.norm(h1) + 1e-10)
            h2_norm = h2 / (np.linalg.norm(h2) + 1e-10)
            cos_sim = np.clip(np.dot(h1_norm, h2_norm), -1, 1)
            geo_distance = 1 - cos_sim  # 0=相同, 2=相反
            
            # 功能性发散: logits的差异
            logits1 = h1 @ W_U.T
            logits2 = h2 @ W_U.T
            
            # 1. logit L2距离 (归一化)
            l1_norm = logits1 / (np.linalg.norm(logits1) + 1e-10)
            l2_norm = logits2 / (np.linalg.norm(logits2) + 1e-10)
            logit_cos = np.clip(np.dot(l1_norm, l2_norm), -1, 1)
            logit_distance = 1 - logit_cos
            
            # 2. Top-1预测是否不同
            top1_1 = np.argmax(logits1)
            top1_2 = np.argmax(logits2)
            top1_different = int(top1_1 != top1_2)
            
            # 3. Top-10预测的overlap
            top10_1 = set(np.argsort(logits1)[-10:])
            top10_2 = set(np.argsort(logits2)[-10:])
            top10_overlap = len(top10_1 & top10_2) / 10.0
            
            # 4. 翻译token的logit差异 (如果有的话)
            divergence[l] = {
                "geo_distance": float(geo_distance),
                "logit_distance": float(logit_distance),
                "top1_different": top1_different,
                "top10_overlap": float(top10_overlap),
                "top1_token_1": int(top1_1),
                "top1_token_2": int(top1_2),
            }
        
        return divergence

    # ========================================
    # A. 语义相近 vs 语义远 vs 翻译对比
    # ========================================
    all_divergences = {}

    print(f"\n  === 组1: 语义相近的中文词 ===")
    similar_divs = []
    for zh1, zh2 in similar_zh:
        p1 = f"{zh1}是一种"
        p2 = f"{zh2}是一种"
        div = compute_functional_divergence(p1, p2, zh1, zh2)
        similar_divs.append(div)
        print(f"    {zh1} vs {zh2}: top1分歧在L", end="")
        for l in range(n_layers + 1):
            if div[l]["top1_different"]:
                print(f"{l}", end=" ")
                break
        print()

    print(f"\n  === 组2: 语义远的中文词 ===")
    distant_divs = []
    for zh1, zh2 in distant_zh:
        p1 = f"{zh1}是一种"
        p2 = f"{zh2}是一种"
        div = compute_functional_divergence(p1, p2, zh1, zh2)
        distant_divs.append(div)
        print(f"    {zh1} vs {zh2}: top1分歧在L", end="")
        for l in range(n_layers + 1):
            if div[l]["top1_different"]:
                print(f"{l}", end=" ")
                break
        print()

    print(f"\n  === 组3: 中文prompt vs 翻译prompt ===")
    trans_divs = []
    for zh, en in trans_vs_zh:
        p1 = f"{zh}是一种"
        p2 = f'"{zh}"的英文是'
        div = compute_functional_divergence(p1, p2, f"zh({zh})", f"trans({zh})")
        trans_divs.append(div)
        print(f"    zh({zh}) vs trans({zh}): top1分歧在L", end="")
        for l in range(n_layers + 1):
            if div[l]["top1_different"]:
                print(f"{l}", end=" ")
                break
        print()

    # ========================================
    # B. 汇总: 各层的平均功能发散
    # ========================================
    print(f"\n  === 各组在各层的平均功能发散 ===")
    print(f"  {'Layer':<7} {'Geo_similar':<14} {'Geo_distant':<14} {'Geo_trans':<14} "
          f"{'Logit_similar':<14} {'Logit_distant':<14} {'Logit_trans':<14} "
          f"{'Top1div_sim':<12} {'Top1div_dist':<12} {'Top1div_trans':<12}")

    for l in range(0, n_layers + 1, 3):
        geo_sim = np.mean([d[l]["geo_distance"] for d in similar_divs])
        geo_dist = np.mean([d[l]["geo_distance"] for d in distant_divs])
        geo_trans = np.mean([d[l]["geo_distance"] for d in trans_divs])
        logit_sim = np.mean([d[l]["logit_distance"] for d in similar_divs])
        logit_dist = np.mean([d[l]["logit_distance"] for d in distant_divs])
        logit_trans = np.mean([d[l]["logit_distance"] for d in trans_divs])
        top1_sim = np.mean([d[l]["top1_different"] for d in similar_divs])
        top1_dist = np.mean([d[l]["top1_different"] for d in distant_divs])
        top1_trans = np.mean([d[l]["top1_different"] for d in trans_divs])
        
        print(f"  L{l:<5} {geo_sim:<14.4f} {geo_dist:<14.4f} {geo_trans:<14.4f} "
              f"{logit_sim:<14.4f} {logit_dist:<14.4f} {logit_trans:<14.4f} "
              f"{top1_sim:<12.2f} {top1_dist:<12.2f} {top1_trans:<12.2f}")

    # ========================================
    # C. 关键问题: 功能发散是否出现在几何发散之前?
    # ========================================
    print(f"\n  === 发散时序分析 ===")
    print(f"  如果功能发散早于几何发散 → 表示先有功能变化, 再有几何变化")
    print(f"  如果几何发散早于功能发散 → 几何变化未必影响功能")
    
    for group_name, divs in [("相似词", similar_divs), ("远义词", distant_divs), ("翻译vs中文", trans_divs)]:
        # 找到geo_distance首次超过某阈值的层
        geo_threshold = 0.1  # 几何距离阈值
        logit_threshold = 0.1  # logit距离阈值
        
        geo_diverge_layer = n_layers + 1
        logit_diverge_layer = n_layers + 1
        
        for l in range(n_layers + 1):
            mean_geo = np.mean([d[l]["geo_distance"] for d in divs])
            mean_logit = np.mean([d[l]["logit_distance"] for d in divs])
            
            if mean_geo > geo_threshold and geo_diverge_layer > n_layers:
                geo_diverge_layer = l
            if mean_logit > logit_threshold and logit_diverge_layer > n_layers:
                logit_diverge_layer = l
        
        print(f"  {group_name}: geo发散层=L{geo_diverge_layer}, logit发散层=L{logit_diverge_layer}", end="")
        if geo_diverge_layer < logit_diverge_layer:
            print(f" → 几何先于功能")
        elif logit_diverge_layer < geo_diverge_layer:
            print(f" → 功能先于几何")
        else:
            print(f" → 同步")

    results = {
        "similar_divergences": {f"pair_{i}": {str(l): v for l, v in d.items()} for i, d in enumerate(similar_divs)},
        "distant_divergences": {f"pair_{i}": {str(l): v for l, v in d.items()} for i, d in enumerate(distant_divs)},
        "trans_divergences": {f"pair_{i}": {str(l): v for l, v in d.items()} for i, d in enumerate(trans_divs)},
    }

    out_path = f"tests/glm5_temp/phase109_exp3_{model_name}_trajectory_divergence.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  保存到 {out_path}")

    release_model(model)
    return results


# ============================================================
# Exp 4: Reconcentration Verification
# ============================================================
def exp4_reconcentration_verification(args):
    model_name = args.model
    print(f"\n{'='*60}")
    print("Exp 4: Reconcentration Verification — 重新集中验证")
    print(f"{'='*60}")

    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    W_U = get_W_U(model).astype(np.float64)

    # W_U的右奇异向量
    _, S_dec, Vt_dec = np.linalg.svd(W_U, full_matrices=False)
    dec_directions = Vt_dec.T

    # 收集hidden states
    all_pairs = ALL_PLUS_EXTRA  # 60词对
    print(f"\n  收集{len(all_pairs)}个词对的hidden states...")
    layer_states = collect_hidden_states(model, tokenizer, device, all_pairs)

    # ========================================
    # A. 每层翻译差分在多个基底中的PR
    # ========================================
    print(f"\n  === 翻译差分信号的PR在多个坐标系中的变化 ===")
    
    # 计算每层的self-basis
    layer_bases = {}
    for l in range(n_layers + 1):
        zh_data = np.array(layer_states[l]["zh"], dtype=np.float64)
        trans_data = np.array(layer_states[l]["trans"], dtype=np.float64)
        diff = trans_data - zh_data
        
        # SVD of差分矩阵
        U_diff, S_diff, Vt_diff = np.linalg.svd(diff, full_matrices=False)
        layer_bases[l] = {
            "U": U_diff,  # (n, n)
            "S": S_diff,
            "Vt": Vt_diff,  # (d_model, d_model) 或 (n, d_model)
            "self_directions": Vt_diff.T,  # (d_model, min(n,d)) 列是主方向
        }

    # 在4种基底中计算PR
    basis_types = {
        "decoder": dec_directions,  # W_U右奇异向量
    }
    
    # 添加每层自己的基底 (too many to store all, compute on the fly)

    results_by_basis = {}
    
    for basis_name in ["decoder", "self", "random_orthogonal"]:
        pr_curve = []
        entropy_curve = []
        
        for l in range(n_layers + 1):
            zh_data = np.array(layer_states[l]["zh"], dtype=np.float64)
            trans_data = np.array(layer_states[l]["trans"], dtype=np.float64)
            diff = trans_data - zh_data
            mean_diff = np.mean(diff, axis=0)
            
            if basis_name == "decoder":
                basis = dec_directions
            elif basis_name == "self":
                basis = layer_bases[l]["self_directions"]
            else:  # random_orthogonal
                # 生成随机正交基底
                Q, _ = np.linalg.qr(np.random.randn(d_model, d_model))
                basis = Q
            
            # 投影
            projection = mean_diff @ basis
            energy = projection ** 2
            
            # PR
            total = np.sum(energy)
            if total > 0 and np.sum(energy**2) > 0:
                PR = total**2 / np.sum(energy**2)
            else:
                PR = 0
            
            # Entropy
            if total > 0:
                p = energy / total
                p = p[p > 0]
                entropy = -np.sum(p * np.log(p))
                norm_entropy = entropy / np.log(min(d_model, len(energy)))
            else:
                norm_entropy = 0
            
            pr_curve.append(float(PR))
            entropy_curve.append(float(norm_entropy))
        
        results_by_basis[basis_name] = {
            "pr_curve": pr_curve,
            "entropy_curve": entropy_curve,
        }
        
        print(f"\n  {basis_name} basis:")
        print(f"    PR: L0={pr_curve[0]:.1f}, L6={pr_curve[6]:.1f}, L12={pr_curve[12]:.1f}, "
              f"L18={pr_curve[18]:.1f}, L24={pr_curve[24]:.1f}, L30={pr_curve[30]:.1f}, "
              f"L33={pr_curve[33]:.1f}, L36={pr_curve[36]:.1f}")
        print(f"    Entropy: L0={entropy_curve[0]:.4f}, L6={entropy_curve[6]:.4f}, "
              f"L18={entropy_curve[18]:.4f}, L36={entropy_curve[36]:.4f}")

    # ========================================
    # B. Bootstrap置信区间
    # ========================================
    print(f"\n  === Bootstrap 95% CI for PR (decoder basis) ===")
    n_bootstrap = 200
    n_pairs = len(all_pairs)
    
    pr_bootstraps = {l: [] for l in range(n_layers + 1)}
    
    for trial in range(n_bootstrap):
        idx = np.random.choice(n_pairs, n_pairs, replace=True)
        
        for l in range(n_layers + 1):
            zh_data = np.array(layer_states[l]["zh"], dtype=np.float64)
            trans_data = np.array(layer_states[l]["trans"], dtype=np.float64)
            
            diff_boot = trans_data[idx] - zh_data[idx]
            mean_diff_boot = np.mean(diff_boot, axis=0)
            
            projection = mean_diff_boot @ dec_directions
            energy = projection ** 2
            total = np.sum(energy)
            if total > 0 and np.sum(energy**2) > 0:
                PR = total**2 / np.sum(energy**2)
            else:
                PR = 0
            
            pr_bootstraps[l].append(PR)
    
    print(f"  {'Layer':<7} {'PR_mean':<10} {'PR_std':<10} {'95%CI_low':<12} {'95%CI_high':<12}")
    for l in [0, 6, 12, 18, 24, 30, 33, 34, 35, 36]:
        prs = np.array(pr_bootstraps[l])
        print(f"  L{l:<5} {np.mean(prs):<10.1f} {np.std(prs):<10.1f} "
              f"{np.percentile(prs, 2.5):<12.1f} {np.percentile(prs, 97.5):<12.1f}")

    # ========================================
    # C. LN效应的直接分析
    # ========================================
    print(f"\n  === LN对PR的数学效应 ===")
    print(f"  如果L36的PR下降只是LN的数学效应(而非模型学到的)")
    print(f"  那么对L35的hidden state直接做人工LN也应该降低PR")
    
    zh_l35 = np.array(layer_states[35]["zh"], dtype=np.float64)
    trans_l35 = np.array(layer_states[35]["trans"], dtype=np.float64)
    diff_l35 = trans_l35 - zh_l35
    mean_diff_l35 = np.mean(diff_l35, axis=0)
    
    # L35 raw PR in decoder basis
    proj_l35 = mean_diff_l35 @ dec_directions
    energy_l35 = proj_l35 ** 2
    PR_l35_raw = np.sum(energy_l35)**2 / np.sum(energy_l35**2) if np.sum(energy_l35**2) > 0 else 0
    
    # L35 with manual LN applied to each sample's diff
    diff_l35_ln = np.array([apply_layer_norm(d) for d in diff_l35])
    mean_diff_l35_ln = np.mean(diff_l35_ln, axis=0)
    proj_l35_ln = mean_diff_l35_ln @ dec_directions
    energy_l35_ln = proj_l35_ln ** 2
    PR_l35_ln = np.sum(energy_l35_ln)**2 / np.sum(energy_l35_ln**2) if np.sum(energy_l35_ln**2) > 0 else 0
    
    # L36 actual
    zh_l36 = np.array(layer_states[36]["zh"], dtype=np.float64)
    trans_l36 = np.array(layer_states[36]["trans"], dtype=np.float64)
    diff_l36 = trans_l36 - zh_l36
    mean_diff_l36 = np.mean(diff_l36, axis=0)
    proj_l36 = mean_diff_l36 @ dec_directions
    energy_l36 = proj_l36 ** 2
    PR_l36 = np.sum(energy_l36)**2 / np.sum(energy_l36**2) if np.sum(energy_l36**2) > 0 else 0
    
    # L35 hidden state → manual LN → then diff
    zh_l35_ln = np.array([apply_layer_norm(h) for h in zh_l35])
    trans_l35_ln = np.array([apply_layer_norm(h) for h in trans_l35])
    diff_l35_ln2 = trans_l35_ln - zh_l35_ln
    mean_diff_l35_ln2 = np.mean(diff_l35_ln2, axis=0)
    proj_l35_ln2 = mean_diff_l35_ln2 @ dec_directions
    energy_l35_ln2 = proj_l35_ln2 ** 2
    PR_l35_ln2 = np.sum(energy_l35_ln2)**2 / np.sum(energy_l35_ln2**2) if np.sum(energy_l35_ln2**2) > 0 else 0
    
    print(f"\n    L35 raw (decoder-basis PR): {PR_l35_raw:.1f}")
    print(f"    L35 diff after manual LN (per-sample): {PR_l35_ln:.1f}")
    print(f"    L35 hidden state after manual LN, then diff: {PR_l35_ln2:.1f}")
    print(f"    L36 actual (decoder-basis PR): {PR_l36:.1f}")
    
    if abs(PR_l35_ln2 - PR_l36) < abs(PR_l35_raw - PR_l36):
        print(f"\n    → LN的数学效应能解释L36 PR下降的大部分 (纯LN就降低了PR)")
    else:
        print(f"\n    → LN的数学效应不能解释L36 PR下降 (模型在L35→L36做了超出LN的事)")

    # ========================================
    # D. 逐样本分析: PR下降是所有样本一致还是部分样本驱动?
    # ========================================
    print(f"\n  === 逐样本差分的PR变化 ===")
    
    for l in [0, 6, 12, 18, 24, 30, 35, 36]:
        zh_data = np.array(layer_states[l]["zh"], dtype=np.float64)
        trans_data = np.array(layer_states[l]["trans"], dtype=np.float64)
        diff = trans_data - zh_data
        
        # 每个样本差分在decoder基底中的PR
        sample_prs = []
        for i in range(len(diff)):
            d = diff[i]
            proj = d @ dec_directions
            e = proj ** 2
            total = np.sum(e)
            if total > 0 and np.sum(e**2) > 0:
                pr = total**2 / np.sum(e**2)
            else:
                pr = 0
            sample_prs.append(pr)
        
        sample_prs = np.array(sample_prs)
        print(f"    L{l}: mean_PR={np.mean(sample_prs):.1f}, std={np.std(sample_prs):.1f}, "
              f"min={np.min(sample_prs):.1f}, max={np.max(sample_prs):.1f}")

    results = {
        "pr_by_basis": results_by_basis,
        "bootstrap_ci_decoder": {
            str(l): {
                "mean": float(np.mean(pr_bootstraps[l])),
                "std": float(np.std(pr_bootstraps[l])),
                "ci_95": [float(np.percentile(pr_bootstraps[l], 2.5)), 
                          float(np.percentile(pr_bootstraps[l], 97.5))],
            } for l in range(n_layers + 1)
        },
        "ln_effect": {
            "PR_l35_raw": float(PR_l35_raw),
            "PR_l35_diff_ln": float(PR_l35_ln),
            "PR_l35_state_ln_then_diff": float(PR_l35_ln2),
            "PR_l36_actual": float(PR_l36),
        },
    }

    out_path = f"tests/glm5_temp/phase109_exp4_{model_name}_reconcentration.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  保存到 {out_path}")

    release_model(model)
    return results


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3")
    parser.add_argument("--exp", type=int, default=1)
    args = parser.parse_args()

    if args.exp == 1:
        exp1_flow_angle_null(args)
    elif args.exp == 2:
        exp2_perturbation_survival(args)
    elif args.exp == 3:
        exp3_trajectory_divergence(args)
    elif args.exp == 4:
        exp4_reconcentration_verification(args)
    else:
        print(f"Unknown exp: {args.exp}")
