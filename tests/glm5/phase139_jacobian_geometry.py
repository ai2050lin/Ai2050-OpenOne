"""
Phase 139: Jacobian Geometry — 雅可比几何分析
================================================
核心目标：从"统计现象学"进入"生成机制数学"
关键实验：
  Exp A: 奇异值谱分析 — 每层雅可比矩阵的σ_i(J_l)分布
  Exp B: 语义扰动 vs 随机扰动的传播差异
  Exp C: 归一化传播比与有效秩

这是决定性实验：判断Transformer是
  1) 稳定收缩系统 (σ_i < 1 全部)
  2) 稀疏放大系统 (少数σ_i >> 1, 多数σ_i < 1) ← 最可能
  3) 临界系统 (σ_i ≈ 1 大量集中)

时间：2026-05-12 13:42
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import warnings
warnings.filterwarnings('ignore')
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

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
# 句子设计 — 大量句子保证统计可靠性
# ============================================================

BASE_SENTENCES = [
    # 简单陈述句 (10句)
    "The cat sat on the mat",
    "Dogs are playing in the park",
    "She is reading a book",
    "The sun rises in the east",
    "Birds fly south for winter",
    "The river flows to the sea",
    "He walks to work every day",
    "The children sing happily",
    "Stars shine bright at night",
    "The wind blows from the north",
    # 带否定 (10句)
    "The cat does not like water",
    "She never eats breakfast",
    "They are not coming today",
    "He cannot swim very well",
    "The dog will not bite",
    "Nobody knows the answer",
    "The door is not open",
    "She has no time to rest",
    "They never go outside",
    "He did not see the car",
    # 时态变化 (10句)
    "The cat chased the mouse",
    "She walked to the store",
    "They played soccer yesterday",
    "He was reading a novel",
    "The birds flew away",
    "She had already left",
    "The sun set behind hills",
    "He wrote a long letter",
    "They built a new house",
    "She sang a beautiful song",
    # 复杂句 (10句)
    "If it rains tomorrow we will stay home",
    "Although he was tired he kept working",
    "The scientist who discovered this won a prize",
    "She said that she would come early",
    "Because the road was closed we took a detour",
    "When the bell rings the class will end",
    "The book that I read was fascinating",
    "While she cooked he set the table",
    "Since they arrived early they got good seats",
    "Unless you study hard you will fail",
    # 情态/量词 (10句)
    "Every student must pass the exam",
    "Some people might disagree with you",
    "All cats can see in the dark",
    "Few birds can fly backwards",
    "Each child received a gift",
    "Many students enjoy reading books",
    "Both options seem reasonable",
    "Several factors contribute to this",
    "Any person can learn to code",
    "Most animals need water to survive",
]

# 否定算子句对 (用于Exp B)
NEGATION_OPERATOR_PAIRS = [
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
    ("The writer always writes clearly", "The writer never writes clearly"),
    ("The driver always drives safely", "The driver never drives safely"),
    ("The singer always sings softly", "The singer never sings softly"),
    ("The builder always builds strong", "The builder never builds strong"),
    ("The cook always makes food", "The cook never makes food"),
    ("The nurse always cares deeply", "The nurse never cares deeply"),
    ("The police always protect citizens", "The police never protect citizens"),
]

# 时态算子句对
TENSE_OPERATOR_PAIRS = [
    ("The dog bites the man", "The dog bit the man"),
    ("The cat chases the mouse", "The cat chased the mouse"),
    ("The sun rises early", "The sun rose early"),
    ("The river flows south", "The river flowed south"),
    ("The wind blows hard", "The wind blew hard"),
    ("The bird sings loud", "The bird sang loud"),
    ("The fire burns hot", "The fire burnt hot"),
    ("The child plays hard", "The child played hard"),
    ("The doctor helps patients", "The doctor helped patients"),
    ("The teacher reads books", "The teacher read books"),
    ("The soldier fights hard", "The soldier fought hard"),
    ("The farmer grows crops", "The farmer grew crops"),
    ("The artist paints well", "The artist painted well"),
    ("The writer writes clearly", "The writer wrote clearly"),
    ("The driver drives safely", "The driver drove safely"),
    ("The singer sings softly", "The singer sang softly"),
    ("The builder builds strong", "The builder built strong"),
    ("The cook makes food", "The cook made food"),
    ("The nurse cares deeply", "The nurse cared deeply"),
    ("The police protect citizens", "The police protected citizens"),
]


# ============================================================
# 工具函数
# ============================================================

def get_device_for_input(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_logit_entropy(logits: np.ndarray) -> float:
    logits_shifted = logits - np.max(logits)
    exp_l = np.exp(logits_shifted)
    probs = exp_l / np.sum(exp_l)
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(probs * np.log(probs)))


# ============================================================
# Exp A: 扰动传播的奇异值谱分析 (不需要显式计算Jacobian)
# ============================================================
# 核心思想：不直接计算 J_l = ∂h_{l+1}/∂h_l (太耗内存)
# 而是通过扰动传播来推断奇异值谱的性质
# 方法：注入多个方向的扰动，测量传播后的方向分散度
# 如果是稀疏放大系统：只有少数方向被放大，大部分被抑制

def expA_singular_spectrum_analysis(model, tokenizer, device, model_info, model_name: str):
    """
    Exp A: 扰动传播奇异值谱分析
    
    方法：
    1. 在层L注入N个随机方向的扰动 (单位球上均匀采样)
    2. 在后续层测量每个扰动方向的传播
    3. 通过传播幅度的分布推断奇异值谱形状
    
    关键指标：
    - propagation_ratio_per_direction: 每个方向的传播比
    - effective_amplification_rank: 被显著放大的方向数量
    - amplification_spectrum: 传播幅度的排序分布 (模拟奇异值谱)
    """
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    input_device = get_device_for_input(model)
    
    # 采样扰动层
    perturb_layers = []
    step = max(1, n_layers // 5)
    for i in range(0, n_layers - 2, step):
        perturb_layers.append(i)
    
    n_perturb_dirs = 50  # 扰动方向数量 (用于推断谱形状)
    eps_scale = 0.01  # 扰动强度 (相对hidden state范数的1%)
    
    # 测试句子
    test_sentences = BASE_SENTENCES[:10]
    
    all_results = {"model": model_name, "n_layers": n_layers, "d_model": d_model,
                   "n_perturb_dirs": n_perturb_dirs, "eps_scale": eps_scale,
                   "perturbation_layers": {}}
    
    for perturb_li in perturb_layers:
        print(f"\n  扰动层 L{perturb_li}...")
        layer_results = {"amplification_spectra": [], "effective_ranks": [],
                         "propagation_ratios_by_layer": defaultdict(list)}
        
        for sent_idx, sentence in enumerate(test_sentences):
            print(f"    句子 {sent_idx+1}/{len(test_sentences)}: '{sentence[:40]}...'")
            
            ids = tokenizer.encode(sentence, add_special_tokens=False)
            seq_len = len(ids)
            input_ids = torch.tensor([ids], device=input_device)
            attention_mask = torch.ones(1, seq_len, device=input_device, dtype=torch.long)
            
            # 获取原始hidden states
            with torch.no_grad():
                out_orig = model(input_ids=input_ids, attention_mask=attention_mask,
                                 output_hidden_states=True)
            
            hs_orig = [hs.detach().clone() for hs in out_orig.hidden_states]
            hs_at_perturb = hs_orig[perturb_li + 1]  # [1, seq_len, d_model]
            hs_norm = float(hs_at_perturb[0, -1, :].norm())
            eps_abs = eps_scale * hs_norm
            
            if eps_abs < 1e-8:
                continue
            
            # 采样后续观察层
            observe_layers = []
            obs_step = max(1, (n_layers - perturb_li - 1) // 4)
            for li in range(perturb_li + 1, n_layers):
                if (li - perturb_li) % obs_step == 0 or li == n_layers - 1:
                    observe_layers.append(li)
            
            # 注入N个方向的扰动
            amplification_per_dir = []  # 每个方向的最终放大倍数
            
            for dir_idx in range(n_perturb_dirs):
                torch.manual_seed(42 + dir_idx + perturb_li * 1000)
                random_dir = torch.randn(d_model, device=hs_at_perturb.device,
                                         dtype=hs_at_perturb.dtype)
                random_dir = random_dir / random_dir.norm() * eps_abs
                
                # 构建扰动后的hidden state
                hs_perturbed = hs_at_perturb.clone()
                hs_perturbed[0, -1, :] += random_dir
                
                # 从扰动层继续forward
                captured_hs = {}
                
                def make_capture_hook(key):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            captured_hs[key] = output[0].detach().clone()
                        else:
                            captured_hs[key] = output.detach().clone()
                    return hook
                
                hooks = []
                for oli in observe_layers:
                    hooks.append(layers[oli].register_forward_hook(
                        make_capture_hook(f"L{oli}")))
                
                inject_done = [False]
                def inject_hook(module, input, output):
                    if not inject_done[0]:
                        inject_done[0] = True
                        if isinstance(output, tuple):
                            return (hs_perturbed.to(output[0].device).to(output[0].dtype),) + output[1:]
                        return hs_perturbed.to(output.device).to(output.dtype)
                    return output
                
                inject_h = layers[perturb_li].register_forward_hook(inject_hook)
                
                try:
                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attention_mask)
                except Exception as e:
                    inject_h.remove()
                    for h in hooks:
                        h.remove()
                    continue
                
                inject_h.remove()
                for h in hooks:
                    h.remove()
                
                # 计算每个观察层的传播比
                dir_prop_ratios = {}
                for oli in observe_layers:
                    clk = f"L{oli}"
                    if clk not in captured_hs:
                        continue
                    # Δh at observation layer
                    hs_obs_orig = hs_orig[oli + 1]
                    delta_h = captured_hs[clk][0, -1, :] - hs_obs_orig[0, -1, :]
                    delta_h_norm = float(delta_h.norm())
                    prop_ratio = delta_h_norm / max(eps_abs, 1e-10)
                    dir_prop_ratios[oli] = prop_ratio
                
                # 用最后一层的传播比作为该方向的放大倍数
                if observe_layers:
                    last_observe = observe_layers[-1]
                    if last_observe in dir_prop_ratios:
                        amplification_per_dir.append(dir_prop_ratios[last_observe])
                
                # 收集每层的传播比
                for oli, ratio in dir_prop_ratios.items():
                    layer_results["propagation_ratios_by_layer"][f"L{oli}"].append(ratio)
            
            # 分析放大谱 (排序后的传播比 ≈ 奇异值的函数)
            if amplification_per_dir:
                spectrum = sorted(amplification_per_dir, reverse=True)
                layer_results["amplification_spectra"].append(spectrum)
                
                # 有效秩：传播比 > 1 的方向比例
                n_amplified = sum(1 for a in amplification_per_dir if a > 1.0)
                eff_rank_ratio = n_amplified / len(amplification_per_dir)
                layer_results["effective_ranks"].append(eff_rank_ratio)
        
        # 聚合该扰动层的结果
        if layer_results["amplification_spectra"]:
            # 平均谱
            max_len = max(len(s) for s in layer_results["amplification_spectra"])
            avg_spectrum = np.zeros(max_len)
            for s in layer_results["amplification_spectra"]:
                padded = np.zeros(max_len)
                padded[:len(s)] = s
                avg_spectrum += padded
            avg_spectrum /= len(layer_results["amplification_spectra"])
            
            all_results["perturbation_layers"][f"L{perturb_li}"] = {
                "avg_amplification_spectrum": avg_spectrum.tolist(),
                "mean_effective_rank_ratio": float(np.mean(layer_results["effective_ranks"])),
                "std_effective_rank_ratio": float(np.std(layer_results["effective_ranks"])),
                "propagation_ratios_summary": {},
            }
            
            # 每层的传播比统计
            for layer_key, ratios in layer_results["propagation_ratios_by_layer"].items():
                all_results["perturbation_layers"][f"L{perturb_li}"]["propagation_ratios_summary"][layer_key] = {
                    "mean": float(np.mean(ratios)),
                    "std": float(np.std(ratios)),
                    "median": float(np.median(ratios)),
                    "max": float(np.max(ratios)),
                    "min": float(np.min(ratios)),
                    "pct_above_1": float(np.mean([r > 1.0 for r in ratios])),
                }
        else:
            all_results["perturbation_layers"][f"L{perturb_li}"] = {"error": "No data"}
    
    return all_results


# ============================================================
# Exp B: 语义方向扰动 vs 随机方向扰动
# ============================================================

def expB_semantic_vs_random_perturbation(model, tokenizer, device, model_info, model_name: str):
    """
    Exp B: 语义方向扰动 vs 随机方向扰动
    
    核心假设：如果Transformer是"约束传播系统"，则：
    - 语义方向扰动应该沿"约束流形"传播，可能更稳定
    - 随机方向扰动可能横穿流形，导致更大的偏离
    
    方法：
    1. 用否定算子/时态算子计算"语义方向"
       semantic_dir = h(NOT(x)) - h(x) 或 h(PAST(x)) - h(x)
    2. 对比语义方向扰动和随机方向扰动的传播行为
    """
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    input_device = get_device_for_input(model)
    
    # 采样层
    sample_layers = []
    step = max(1, n_layers // 5)
    for i in range(0, n_layers, step):
        sample_layers.append(i)
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    
    results = {"model": model_name, "sample_layers": [f"L{l}" for l in sample_layers],
               "negation_analysis": {}, "tense_analysis": {}}
    
    # === 否定算子分析 ===
    print("\n  否定算子分析...")
    negation_results = []
    
    for base_sent, neg_sent in NEGATION_OPERATOR_PAIRS[:10]:
        # 获取两个句子的hidden states
        base_ids = tokenizer.encode(base_sent, add_special_tokens=False)
        neg_ids = tokenizer.encode(neg_sent, add_special_tokens=False)
        
        # 只比较共同长度的部分
        min_len = min(len(base_ids), len(neg_ids))
        base_ids = base_ids[:min_len]
        neg_ids = neg_ids[:min_len]
        
        base_input = torch.tensor([base_ids], device=input_device)
        neg_input = torch.tensor([neg_ids], device=input_device)
        attn_mask = torch.ones(1, min_len, device=input_device, dtype=torch.long)
        
        with torch.no_grad():
            out_base = model(input_ids=base_input, attention_mask=attn_mask,
                             output_hidden_states=True)
            out_neg = model(input_ids=neg_input, attention_mask=attn_mask,
                            output_hidden_states=True)
        
        # 计算每层的语义方向 (否定方向)
        for li in sample_layers:
            hs_base = out_base.hidden_states[li + 1][0, -1, :].float().cpu().numpy()
            hs_neg = out_neg.hidden_states[li + 1][0, -1, :].float().cpu().numpy()
            neg_direction = hs_neg - hs_base
            neg_dir_norm = np.linalg.norm(neg_direction)
            
            if neg_dir_norm < 1e-8:
                continue
            
            # 归一化语义方向
            neg_dir_unit = neg_direction / neg_dir_norm
            
            # 注入语义方向扰动 vs 随机方向扰动
            # 用base句子作为基准
            hs_at_li = out_base.hidden_states[li + 1].detach().clone()
            hs_norm = float(hs_at_li[0, -1, :].norm())
            eps_abs = 0.05 * hs_norm  # 5%扰动
            
            # 语义方向扰动
            semantic_perturbation = torch.tensor(neg_dir_unit * eps_abs,
                                                 device=hs_at_li.device, dtype=hs_at_li.dtype)
            hs_semantic = hs_at_li.clone()
            hs_semantic[0, -1, :] += semantic_perturbation
            
            # 随机方向扰动 (同幅度)
            torch.manual_seed(42 + li * 100)
            random_dir = torch.randn_like(hs_at_li[0, -1, :])
            random_dir = random_dir / random_dir.norm() * eps_abs
            hs_random = hs_at_li.clone()
            hs_random[0, -1, :] += random_dir
            
            # 从扰动层继续forward，看logit变化
            for pert_type, hs_perturbed in [("semantic", hs_semantic), ("random", hs_random)]:
                captured_hs = {}
                
                def make_capture_hook(key):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            captured_hs[key] = output[0].detach().clone()
                        else:
                            captured_hs[key] = output.detach().clone()
                    return hook
                
                hooks = []
                for oli in sample_layers:
                    hooks.append(layers[oli].register_forward_hook(
                        make_capture_hook(f"L{oli}")))
                
                inject_done = [False]
                def inject_hook(module, input, output):
                    if not inject_done[0]:
                        inject_done[0] = True
                        if isinstance(output, tuple):
                            return (hs_perturbed.to(output[0].device).to(output[0].dtype),) + output[1:]
                        return hs_perturbed.to(output.device).to(output.dtype)
                    return output
                
                inject_h = layers[li].register_forward_hook(inject_hook)
                
                try:
                    with torch.no_grad():
                        out_pert = model(input_ids=base_input, attention_mask=attn_mask)
                except:
                    inject_h.remove()
                    for h in hooks:
                        h.remove()
                    continue
                
                inject_h.remove()
                for h in hooks:
                    h.remove()
                
                logits_orig = out_base.logits[0, -1].float().cpu().numpy()
                logits_pert = out_pert.logits[0, -1].float().cpu().numpy()
                logit_shift = float(np.linalg.norm(logits_pert - logits_orig))
                
                # 每层的传播比
                layer_prop = {}
                for oli in sample_layers:
                    clk = f"L{oli}"
                    if clk in captured_hs:
                        delta_h = captured_hs[clk][0, -1, :] - out_base.hidden_states[oli + 1][0, -1, :].float()
                        layer_prop[clk] = float(delta_h.norm()) / max(eps_abs, 1e-10)
                
                negation_results.append({
                    "perturb_layer": f"L{li}",
                    "perturb_type": pert_type,
                    "logit_shift": logit_shift,
                    "layer_propagation": layer_prop,
                    "neg_dir_norm": float(neg_dir_norm),
                    "base_sentence": base_sent,
                })
    
    # 聚合否定算子结果
    for li_str in [f"L{li}" for li in sample_layers]:
        semantic_shifts = [r["logit_shift"] for r in negation_results
                          if r["perturb_layer"] == li_str and r["perturb_type"] == "semantic"]
        random_shifts = [r["logit_shift"] for r in negation_results
                        if r["perturb_layer"] == li_str and r["perturb_type"] == "random"]
        
        semantic_props = []
        random_props = []
        for r in negation_results:
            if r["perturb_layer"] == li_str and li_str in r.get("layer_propagation", {}):
                if r["perturb_type"] == "semantic":
                    semantic_props.append(r["layer_propagation"][li_str])
                else:
                    random_props.append(r["layer_propagation"][li_str])
        
        results["negation_analysis"][li_str] = {
            "semantic_logit_shift_mean": float(np.mean(semantic_shifts)) if semantic_shifts else 0,
            "semantic_logit_shift_std": float(np.std(semantic_shifts)) if semantic_shifts else 0,
            "random_logit_shift_mean": float(np.mean(random_shifts)) if random_shifts else 0,
            "random_logit_shift_std": float(np.std(random_shifts)) if random_shifts else 0,
            "semantic_prop_mean": float(np.mean(semantic_props)) if semantic_props else 0,
            "random_prop_mean": float(np.mean(random_props)) if random_props else 0,
            "shift_ratio": float(np.mean(random_shifts) / max(np.mean(semantic_shifts), 1e-10))
                        if semantic_shifts and random_shifts else 0,
        }
    
    # === 时态算子分析 (同样的流程) ===
    print("\n  时态算子分析...")
    tense_results = []
    
    for base_sent, past_sent in TENSE_OPERATOR_PAIRS[:10]:
        base_ids = tokenizer.encode(base_sent, add_special_tokens=False)
        past_ids = tokenizer.encode(past_sent, add_special_tokens=False)
        
        min_len = min(len(base_ids), len(past_ids))
        base_ids = base_ids[:min_len]
        past_ids = past_ids[:min_len]
        
        base_input = torch.tensor([base_ids], device=input_device)
        past_input = torch.tensor([past_ids], device=input_device)
        attn_mask = torch.ones(1, min_len, device=input_device, dtype=torch.long)
        
        with torch.no_grad():
            out_base = model(input_ids=base_input, attention_mask=attn_mask,
                             output_hidden_states=True)
            out_past = model(input_ids=past_input, attention_mask=attn_mask,
                            output_hidden_states=True)
        
        for li in sample_layers:
            hs_base = out_base.hidden_states[li + 1][0, -1, :].float().cpu().numpy()
            hs_past = out_past.hidden_states[li + 1][0, -1, :].float().cpu().numpy()
            tense_direction = hs_past - hs_base
            tense_dir_norm = np.linalg.norm(tense_direction)
            
            if tense_dir_norm < 1e-8:
                continue
            
            tense_dir_unit = tense_direction / tense_dir_norm
            
            hs_at_li = out_base.hidden_states[li + 1].detach().clone()
            hs_norm = float(hs_at_li[0, -1, :].norm())
            eps_abs = 0.05 * hs_norm
            
            # 语义方向扰动
            semantic_perturbation = torch.tensor(tense_dir_unit * eps_abs,
                                                 device=hs_at_li.device, dtype=hs_at_li.dtype)
            hs_semantic = hs_at_li.clone()
            hs_semantic[0, -1, :] += semantic_perturbation
            
            # 随机方向扰动
            torch.manual_seed(42 + li * 100 + 500)
            random_dir = torch.randn_like(hs_at_li[0, -1, :])
            random_dir = random_dir / random_dir.norm() * eps_abs
            hs_random = hs_at_li.clone()
            hs_random[0, -1, :] += random_dir
            
            for pert_type, hs_perturbed in [("semantic", hs_semantic), ("random", hs_random)]:
                logits_orig = out_base.logits[0, -1].float().cpu().numpy()
                
                inject_done = [False]
                def inject_hook(module, input, output):
                    if not inject_done[0]:
                        inject_done[0] = True
                        if isinstance(output, tuple):
                            return (hs_perturbed.to(output[0].device).to(output[0].dtype),) + output[1:]
                        return hs_perturbed.to(output.device).to(output.dtype)
                    return output
                
                inject_h = layers[li].register_forward_hook(inject_hook)
                
                try:
                    with torch.no_grad():
                        out_pert = model(input_ids=base_input, attention_mask=attn_mask)
                except:
                    inject_h.remove()
                    continue
                
                inject_h.remove()
                
                logits_pert = out_pert.logits[0, -1].float().cpu().numpy()
                logit_shift = float(np.linalg.norm(logits_pert - logits_orig))
                
                tense_results.append({
                    "perturb_layer": f"L{li}",
                    "perturb_type": pert_type,
                    "logit_shift": logit_shift,
                    "tense_dir_norm": float(tense_dir_norm),
                })
    
    # 聚合时态算子结果
    for li_str in [f"L{li}" for li in sample_layers]:
        semantic_shifts = [r["logit_shift"] for r in tense_results
                          if r["perturb_layer"] == li_str and r["perturb_type"] == "semantic"]
        random_shifts = [r["logit_shift"] for r in tense_results
                        if r["perturb_layer"] == li_str and r["perturb_type"] == "random"]
        
        results["tense_analysis"][li_str] = {
            "semantic_logit_shift_mean": float(np.mean(semantic_shifts)) if semantic_shifts else 0,
            "semantic_logit_shift_std": float(np.std(semantic_shifts)) if semantic_shifts else 0,
            "random_logit_shift_mean": float(np.mean(random_shifts)) if random_shifts else 0,
            "random_logit_shift_std": float(np.std(random_shifts)) if random_shifts else 0,
            "shift_ratio": float(np.mean(random_shifts) / max(np.mean(semantic_shifts), 1e-10))
                        if semantic_shifts and random_shifts else 0,
        }
    
    return results


# ============================================================
# Exp C: 归一化传播比与有效秩
# ============================================================

def expC_normalized_propagation_and_effective_rank(model, tokenizer, device, model_info, model_name: str):
    """
    Exp C: 归一化传播比和有效秩
    
    修正Phase 138的prop_ratio问题：
    - 旧: prop_ratio = ||Δh_l|| / ||ε||
    - 新: prop_ratio_norm = (||Δh_l|| / ||h_l||) / (||ε|| / ||h_perturb||)
    
    有效秩：Δh_l的SVD中，能量集中在前k个分量的比例
    """
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    input_device = get_device_for_input(model)
    
    # 采样层
    perturb_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 3]
    perturb_layers = sorted(set([l for l in perturb_layers if 0 <= l < n_layers - 2]))
    
    eps_scale = 0.05  # 5%扰动
    
    test_sentences = BASE_SENTENCES[:10]
    
    results = {"model": model_name, "eps_scale": eps_scale, "layer_results": {}}
    
    for perturb_li in perturb_layers:
        print(f"\n  扰动层 L{perturb_li}...")
        
        all_norm_ratios = defaultdict(list)
        all_effective_ranks = defaultdict(list)
        all_direction_preservations = defaultdict(list)
        
        for sent_idx, sentence in enumerate(test_sentences):
            ids = tokenizer.encode(sentence, add_special_tokens=False)
            seq_len = len(ids)
            input_ids = torch.tensor([ids], device=input_device)
            attention_mask = torch.ones(1, seq_len, device=input_device, dtype=torch.long)
            
            with torch.no_grad():
                out_orig = model(input_ids=input_ids, attention_mask=attention_mask,
                                 output_hidden_states=True)
            
            hs_orig = [hs.detach().clone() for hs in out_orig.hidden_states]
            hs_at_perturb = hs_orig[perturb_li + 1]
            hs_perturb_norm = float(hs_at_perturb[0, -1, :].norm())
            eps_abs = eps_scale * hs_perturb_norm
            
            if eps_abs < 1e-8:
                continue
            
            # 采样观察层
            observe_layers = []
            obs_step = max(1, (n_layers - perturb_li - 1) // 4)
            for li in range(perturb_li + 1, n_layers):
                if (li - perturb_li) % obs_step == 0 or li == n_layers - 1:
                    observe_layers.append(li)
            
            # 随机扰动
            torch.manual_seed(42 + perturb_li * 100 + sent_idx)
            random_dir = torch.randn(d_model, device=hs_at_perturb.device, dtype=hs_at_perturb.dtype)
            random_dir = random_dir / random_dir.norm() * eps_abs
            
            hs_perturbed = hs_at_perturb.clone()
            hs_perturbed[0, -1, :] += random_dir
            
            captured_hs = {}
            
            def make_capture_hook(key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        captured_hs[key] = output[0].detach().clone()
                    else:
                        captured_hs[key] = output.detach().clone()
                return hook
            
            hooks = []
            for oli in observe_layers:
                hooks.append(layers[oli].register_forward_hook(
                    make_capture_hook(f"L{oli}")))
            
            inject_done = [False]
            def inject_hook(module, input, output):
                if not inject_done[0]:
                    inject_done[0] = True
                    if isinstance(output, tuple):
                        return (hs_perturbed.to(output[0].device).to(output[0].dtype),) + output[1:]
                    return hs_perturbed.to(output.device).to(output.dtype)
                return output
            
            inject_h = layers[perturb_li].register_forward_hook(inject_hook)
            
            try:
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
            except:
                inject_h.remove()
                for h in hooks:
                    h.remove()
                continue
            
            inject_h.remove()
            for h in hooks:
                h.remove()
            
            # 分析传播
            for oli in observe_layers:
                clk = f"L{oli}"
                if clk not in captured_hs:
                    continue
                
                delta_h = captured_hs[clk][0, -1, :] - hs_orig[oli + 1][0, -1, :]
                delta_h_norm = float(delta_h.norm())
                hs_obs_norm = float(hs_orig[oli + 1][0, -1, :].norm())
                
                # 归一化传播比
                if hs_perturb_norm > 1e-8 and hs_obs_norm > 1e-8:
                    norm_ratio = (delta_h_norm / hs_obs_norm) / (eps_abs / hs_perturb_norm)
                else:
                    norm_ratio = delta_h_norm / max(eps_abs, 1e-10)
                
                all_norm_ratios[clk].append(norm_ratio)
                
                # 方向保持度
                cos_sim = float(torch.nn.functional.cosine_similarity(
                    delta_h.unsqueeze(0), random_dir.to(delta_h.device).unsqueeze(0)
                ).item())
                all_direction_preservations[clk].append(cos_sim)
                
                # Δh的SVD有效秩
                delta_h_np = delta_h.float().cpu().numpy()
                if delta_h_norm > 1e-8:
                    # 用能量集中度近似有效秩
                    # 简化: 不做完整SVD, 用几个统计量代替
                    squared = delta_h_np ** 2
                    total_energy = np.sum(squared)
                    if total_energy > 1e-16:
                        # 排序后的能量分布
                        sorted_energy = np.sort(squared)[::-1]
                        # 前10%维度的能量占比
                        top_k = max(1, d_model // 10)
                        top_energy_ratio = np.sum(sorted_energy[:top_k]) / total_energy
                        all_effective_ranks[clk].append({
                            "top10pct_energy_ratio": float(top_energy_ratio),
                            "max_component_ratio": float(sorted_energy[0] / total_energy),
                        })
        
        # 聚合
        layer_data = {}
        for clk in sorted(all_norm_ratios.keys()):
            norms = all_norm_ratios[clk]
            dirs = all_direction_preservations[clk]
            ranks = all_effective_ranks[clk]
            
            layer_data[clk] = {
                "norm_ratio_mean": float(np.mean(norms)),
                "norm_ratio_std": float(np.std(norms)),
                "norm_ratio_median": float(np.median(norms)),
                "direction_preserve_mean": float(np.mean(dirs)),
                "direction_preserve_std": float(np.std(dirs)),
                "top10pct_energy_ratio_mean": float(np.mean([r["top10pct_energy_ratio"] for r in ranks])) if ranks else 0,
                "max_component_ratio_mean": float(np.mean([r["max_component_ratio"] for r in ranks])) if ranks else 0,
            }
        
        results["layer_results"][f"L{perturb_li}"] = layer_data
    
    return results


# ============================================================
# 主程序
# ============================================================

def run_all_experiments(model_name: str):
    """依次运行所有实验"""
    
    print(f"\n{'='*70}")
    print(f"Phase 139: Jacobian Geometry — {model_name}")
    print(f"{'='*70}")
    
    # 加载模型
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    
    print(f"\n模型信息: class={model_info.model_class}, n_layers={model_info.n_layers}, "
          f"d_model={model_info.d_model}")
    
    all_results = {"model_name": model_name, "model_info": {
        "class": model_info.model_class, "n_layers": model_info.n_layers,
        "d_model": model_info.d_model, "vocab_size": model_info.vocab_size,
    }}
    
    try:
        # Exp A: 奇异值谱分析
        print(f"\n{'='*50}")
        print("Exp A: 奇异值谱分析 (扰动传播推断)")
        print(f"{'='*50}")
        t0 = time.time()
        expA_results = expA_singular_spectrum_analysis(model, tokenizer, device, model_info, model_name)
        expA_time = time.time() - t0
        expA_results["time_seconds"] = round(expA_time, 1)
        all_results["expA"] = expA_results
        print(f"Exp A 完成 ({expA_time:.1f}s)")
        
        # Exp B: 语义 vs 随机扰动
        print(f"\n{'='*50}")
        print("Exp B: 语义方向 vs 随机方向扰动")
        print(f"{'='*50}")
        t0 = time.time()
        expB_results = expB_semantic_vs_random_perturbation(model, tokenizer, device, model_info, model_name)
        expB_time = time.time() - t0
        expB_results["time_seconds"] = round(expB_time, 1)
        all_results["expB"] = expB_results
        print(f"Exp B 完成 ({expB_time:.1f}s)")
        
        # Exp C: 归一化传播比与有效秩
        print(f"\n{'='*50}")
        print("Exp C: 归一化传播比与有效秩")
        print(f"{'='*50}")
        t0 = time.time()
        expC_results = expC_normalized_propagation_and_effective_rank(model, tokenizer, device, model_info, model_name)
        expC_time = time.time() - t0
        expC_results["time_seconds"] = round(expC_time, 1)
        all_results["expC"] = expC_results
        print(f"Exp C 完成 ({expC_time:.1f}s)")
        
    finally:
        release_model(model)
    
    return all_results


def summarize_results(results: Dict) -> str:
    """生成结果摘要"""
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"Phase 139 结果摘要: {results['model_name']}")
    lines.append(f"{'='*60}")
    
    # Exp A 摘要
    if "expA" in results:
        lines.append("\n--- Exp A: 奇异值谱分析 ---")
        expA = results["expA"]
        for perturb_key, perturb_data in expA.get("perturbation_layers", {}).items():
            if "error" in perturb_data:
                continue
            lines.append(f"  {perturb_key}:")
            lines.append(f"    有效秩比例: {perturb_data['mean_effective_rank_ratio']:.3f} "
                        f"(±{perturb_data['std_effective_rank_ratio']:.3f})")
            # 传播比统计
            for layer_key, stats in perturb_data.get("propagation_ratios_summary", {}).items():
                lines.append(f"    {layer_key}: mean_prop={stats['mean']:.2f}, "
                            f"median={stats['median']:.2f}, "
                            f"pct>1={stats['pct_above_1']:.2f}")
    
    # Exp B 摘要
    if "expB" in results:
        lines.append("\n--- Exp B: 语义 vs 随机扰动 ---")
        expB = results["expB"]
        
        for op_name in ["negation_analysis", "tense_analysis"]:
            lines.append(f"  {op_name}:")
            for layer_key, data in expB.get(op_name, {}).items():
                shift_ratio = data.get("shift_ratio", 0)
                lines.append(f"    {layer_key}: "
                            f"语义logit偏移={data['semantic_logit_shift_mean']:.3f}, "
                            f"随机logit偏移={data['random_logit_shift_mean']:.3f}, "
                            f"比值={shift_ratio:.2f}")
    
    # Exp C 摘要
    if "expC" in results:
        lines.append("\n--- Exp C: 归一化传播比与有效秩 ---")
        expC = results["expC"]
        for perturb_key, layer_data in expC.get("layer_results", {}).items():
            lines.append(f"  扰动@{perturb_key}:")
            for obs_key, stats in layer_data.items():
                lines.append(f"    →{obs_key}: norm_ratio={stats['norm_ratio_mean']:.3f}, "
                            f"dir_preserve={stats['direction_preserve_mean']:.3f}, "
                            f"top10%_energy={stats['top10pct_energy_ratio_mean']:.3f}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    results = run_all_experiments(model_name)
    
    # 保存结果
    timestamp = time.strftime("%Y%m%d_%H%M")
    filename = f"tests/glm5_temp/phase139_{model_name}_jacobian_geometry_{timestamp}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果已保存: {filename}")
    
    # 打印摘要
    summary = summarize_results(results)
    print(summary)
