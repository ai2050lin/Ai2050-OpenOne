"""
Phase 151: 语言约束传播动力学 — Causal Token Graph + Trajectory Sensitivity + Statistical Invariants
=======================================================================================================

用户核心批评(全部正确):
  1. W_U row-space保持 ≠ 语义保护 — 可能只是"token prediction sensitive directions"
  2. 全局cos(δ,δ')→0 ≠ 语义方向丢失 — 低维结构保持+高维噪声=全局cos→0
  3. Token coupling弱(Euclidean)≠语言传播弱 — 0.05耦合可能改变top-1 token
  4. 需要测"因果灵敏度": 一个token上的微扰最终会改变多少后续token
  5. 需要"统计守恒量": rollout中什么保持不变
  6. hidden state是瞬时缓存, 真正结构在attention/routing/token coupling中

三个核心实验:
  Exp 1: Row-Space vs Semantic Space — row-space保持到底保护了什么?
    - 比较W_U row-space方向 vs 特定语义方向(如"否定"/"疑问"/"否定"token方向)
    - 测量: 语义方向在传播中的保持率 vs 随机方向的保持率
    - 若语义方向保持率 > 随机方向保持率 → row-space保持确实有语义意义
    - 若语义方向保持率 ≈ 随机方向保持率 → row-space保持只是token prediction的副产物

  Exp 2: Causal Token Sensitivity — 一个token的扰动如何改变后续生成轨迹
    - 在position j注入微扰, 测量position i的logit变化
    - 关键度量: P(top1_token改变 | 微扰在position j) — 因果灵敏度!
    - 不测欧氏范数, 测离散决策改变
    - 也要测: 微扰在position j后, 生成轨迹的分支概率

  Exp 3: Rollout Statistical Invariants — 生成过程中什么保持不变
    - 从同一起点多次生成(temperature>0), 统计哪些量守恒
    - 候选: attention graph的社区结构, token mutual information, dependency entropy
    - 与随机模型对比: 训练模型的守恒量 > 随机模型

用法:
  python tests/glm5/phase151_causal_dynamics.py qwen3
  python tests/glm5/phase151_causal_dynamics.py glm4
  python tests/glm5/phase151_causal_dynamics.py deepseek7b
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
                          release_model, get_W_U, MODEL_CONFIGS,
                          get_attr_direction, compute_cos)

# ============================================================
# 全局参数
# ============================================================
EPSILON = 1.0       # 扰动大小
N_SENTENCES = 20    # 数据量
N_ROLLOUTS = 10     # Exp3每个起点的生成次数
OUTPUT_DIR = Path("tests/glm5_temp")

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
    "The government announced that the new policy",
    "In the future, artificial intelligence will",
    "The philosopher argued that consciousness is",
    "After years of research, they found that",
    "The key difference between the two approaches is",
]

# 语义方向: 使用token embedding方向作为"语义方向"的代理
# 这些是我们认为有语义意义的方向
SEMANTIC_ATTRS = [
    # 语法/功能词方向 — 这些方向如果被保持, 说明语法结构在传播
    "not",      # 否定
    "the",      # 限定词
    "is",       # 系动词
    "and",      # 连接词
    "he",       # 代词
    "she",      # 代词
    # 内容词方向 — 这些方向如果被保持, 说明语义在传播
    "red",      # 颜色
    "big",      # 形容词
    "run",      # 动词
    "science",  # 名词
    "never",    # 否定副词
    "always",   # 肯定副词
]


# ============================================================
# 工具函数
# ============================================================
def get_device_for_input(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda")


def get_row_null_bases(W_U, n_components=None):
    """计算W_U的row space基底 — 如果n_components=None, 使用有效rank"""
    from scipy.sparse.linalg import svds
    
    W_U_T = W_U.T.astype(np.float32)
    k_max = min(500, min(W_U_T.shape) - 2)
    k_max = max(k_max, 10)
    
    print(f"  Computing SVD of W_U^T (shape={W_U_T.shape}), k_max={k_max}...")
    t0 = time.time()
    U_full, s_full, Vt_full = svds(W_U_T, k=k_max)
    idx = np.argsort(-s_full)
    U_full = U_full[:, idx]
    s_full = s_full[idx]
    print(f"  SVD done in {time.time()-t0:.1f}s")
    
    total_energy = np.sum(s_full ** 2)
    cumulative_energy = np.cumsum(s_full ** 2)
    k_90 = np.searchsorted(cumulative_energy, 0.90 * total_energy) + 1
    k_95 = np.searchsorted(cumulative_energy, 0.95 * total_energy) + 1
    k_99 = np.searchsorted(cumulative_energy, 0.99 * total_energy) + 1
    
    print(f"  Effective rank: 90%={k_90}, 95%={k_95}, 99%={k_99}")
    
    if n_components is None:
        k = k_95
    else:
        k = min(n_components, k_max)
    
    row_basis = U_full[:, :k].T  # [k, d_model]
    U_full_all = U_full  # 保存完整U用于后续分析
    
    return row_basis, k, s_full[:k_max], k_90, k_95, k_99, U_full_all


def project_to_row(vec, row_basis):
    """将向量投影到row space"""
    return row_basis.T @ (row_basis @ vec)


def project_to_null(vec, row_basis):
    """将向量投影到null space"""
    row_component = row_basis.T @ (row_basis @ vec)
    return vec - row_component


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
    """在指定层、指定position注入扰动, 返回output_hidden_states + attentions"""
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
                        output_hidden_states=True, output_attentions=True)
    except Exception as e:
        print(f"  [forward error] {e}")
        out = None
    finally:
        for h in hooks:
            h.remove()

    return out


def get_top1_token(logits_np, tokenizer):
    """获取logits的top-1 token"""
    return int(np.argmax(logits_np))


def get_topk_tokens(logits_np, k=5):
    """获取logits的top-k token ids"""
    return np.argsort(logits_np)[-k:][::-1].tolist()


# ============================================================
# Exp 1: Row-Space vs Semantic Space — row-space保持保护了什么?
# ============================================================
def exp1_semantic_vs_row(model, tokenizer, model_name, W_U, row_basis, U_full):
    """
    核心问题: W_U row-space保持 ≠ 语义保持?
    
    方法:
    1. 定义"语义方向": 特定token在W_U中的行向量(=该token的logit方向)
    2. 定义"随机方向": 随机向量
    3. 定义"row-space随机方向": 在row-space内的随机方向
    4. 在L0注入这些方向, 测量每层的保持率
    5. 比较: 语义方向保持率 vs row-space随机方向保持率 vs 全空间随机方向保持率
    
    若语义方向保持率 > row-space随机 → row-space保持有语义意义
    若语义方向保持率 ≈ row-space随机 → row-space保持只是token prediction副产物
    """
    print("\n" + "="*60)
    print("Exp 1: Row-Space vs Semantic Space")
    print("核心: row-space保持保护的是'语义'还是只是'token prediction'?")
    print("="*60)
    
    info = get_model_info(model, model_name)
    device = get_device_for_input(model)
    d_model = info.d_model
    n_layers = info.n_layers
    
    # 采样层
    sample_layers = list(range(0, n_layers + 1, max(1, n_layers // 12)))
    sample_layers = sorted(set(sample_layers + [n_layers]))
    
    # === 准备注入方向 ===
    # 类型A: 语义方向 (特定token在W_U中的方向)
    semantic_directions = {}
    for attr in SEMANTIC_ATTRS:
        direction, tok_id = get_attr_direction(model, tokenizer, attr, W_U)
        if direction is not None and np.linalg.norm(direction) > 1e-8:
            semantic_directions[attr] = {
                'direction': direction,
                'token_id': tok_id,
                'row_energy': compute_row_energy(direction, row_basis)[0],
            }
    
    print(f"  Prepared {len(semantic_directions)} semantic directions")
    for attr, data in semantic_directions.items():
        print(f"    '{attr}' (tok={data['token_id']}): row_energy={data['row_energy']:.4f}")
    
    # 类型B: row-space内随机方向 (与语义方向同维度, 但随机)
    # 类型C: 全空间随机方向
    
    results = {
        'semantic': {},   # attr -> {layer: cos_with_original}
        'row_random': {},  # trial -> {layer: cos_with_original}
        'full_random': {}, # trial -> {layer: cos_with_original}
    }
    
    for sent_idx in range(min(N_SENTENCES, 12)):
        prompt = TEST_PROMPTS[sent_idx]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        seq_len = input_ids.shape[1]
        last_pos = seq_len - 1
        
        # Clean forward
        with torch.no_grad():
            out_clean = model(input_ids=input_ids, attention_mask=attention_mask,
                              output_hidden_states=True)
        clean_hs = out_clean.hidden_states
        
        inject_l = 0
        
        # --- 类型A: 语义方向 ---
        for attr, data in semantic_directions.items():
            delta = data['direction'] * EPSILON
            
            out_perturbed = run_forward_with_perturbation_at_position(
                model, input_ids, attention_mask, inject_l, last_pos, delta, device)
            if out_perturbed is None:
                continue
            
            for li in sample_layers:
                perturbed_vec = out_perturbed.hidden_states[li][0, last_pos, :].float().cpu().numpy()
                clean_vec = clean_hs[li][0, last_pos, :].float().cpu().numpy()
                delta_prop = perturbed_vec - clean_vec
                
                cos_with_original = compute_cos(delta_prop, data['direction'])
                row_e, null_r = compute_row_energy(delta_prop, row_basis)
                
                if attr not in results['semantic']:
                    results['semantic'][attr] = {}
                if li not in results['semantic'][attr]:
                    results['semantic'][attr][li] = {'cos': [], 'row_energy': [], 'null_ratio': []}
                
                results['semantic'][attr][li]['cos'].append(cos_with_original)
                results['semantic'][attr][li]['row_energy'].append(row_e)
                results['semantic'][attr][li]['null_ratio'].append(null_r)
        
        # --- 类型B: row-space内随机方向 ---
        for trial in range(5):
            np.random.seed(sent_idx * 100 + trial)
            raw = np.random.randn(d_model)
            row_dir = project_to_row(raw, row_basis)
            norm = np.linalg.norm(row_dir)
            if norm < 1e-8:
                continue
            row_dir = row_dir / norm
            delta = row_dir * EPSILON
            
            out_perturbed = run_forward_with_perturbation_at_position(
                model, input_ids, attention_mask, inject_l, last_pos, delta, device)
            if out_perturbed is None:
                continue
            
            for li in sample_layers:
                perturbed_vec = out_perturbed.hidden_states[li][0, last_pos, :].float().cpu().numpy()
                clean_vec = clean_hs[li][0, last_pos, :].float().cpu().numpy()
                delta_prop = perturbed_vec - clean_vec
                
                cos_with_original = compute_cos(delta_prop, row_dir)
                row_e, null_r = compute_row_energy(delta_prop, row_basis)
                
                tkey = f"trial_{trial}"
                if tkey not in results['row_random']:
                    results['row_random'][tkey] = {}
                if li not in results['row_random'][tkey]:
                    results['row_random'][tkey][li] = {'cos': [], 'row_energy': [], 'null_ratio': []}
                
                results['row_random'][tkey][li]['cos'].append(cos_with_original)
                results['row_random'][tkey][li]['row_energy'].append(row_e)
                results['row_random'][tkey][li]['null_ratio'].append(null_r)
        
        # --- 类型C: 全空间随机方向 ---
        for trial in range(5):
            np.random.seed(sent_idx * 100 + 50 + trial)
            raw = np.random.randn(d_model)
            norm = np.linalg.norm(raw)
            if norm < 1e-8:
                continue
            full_dir = raw / norm
            delta = full_dir * EPSILON
            
            out_perturbed = run_forward_with_perturbation_at_position(
                model, input_ids, attention_mask, inject_l, last_pos, delta, device)
            if out_perturbed is None:
                continue
            
            for li in sample_layers:
                perturbed_vec = out_perturbed.hidden_states[li][0, last_pos, :].float().cpu().numpy()
                clean_vec = clean_hs[li][0, last_pos, :].float().cpu().numpy()
                delta_prop = perturbed_vec - clean_vec
                
                cos_with_original = compute_cos(delta_prop, full_dir)
                row_e, null_r = compute_row_energy(delta_prop, row_basis)
                
                tkey = f"trial_{trial}"
                if tkey not in results['full_random']:
                    results['full_random'][tkey] = {}
                if li not in results['full_random'][tkey]:
                    results['full_random'][tkey][li] = {'cos': [], 'row_energy': [], 'null_ratio': []}
                
                results['full_random'][tkey][li]['cos'].append(cos_with_original)
                results['full_random'][tkey][li]['row_energy'].append(row_e)
                results['full_random'][tkey][li]['null_ratio'].append(null_r)
    
    # === 汇总 ===
    print("\n  === Exp 1 Summary: Semantic vs Row-Space ===")
    
    # 计算每层各类型的平均cos(δ_prop, δ_original)
    summary = {}
    for li in sample_layers:
        # 语义方向
        sem_cos = []
        for attr in semantic_directions:
            if attr in results['semantic'] and li in results['semantic'][attr]:
                sem_cos.extend(results['semantic'][attr][li]['cos'])
        
        # row-space随机方向
        row_cos = []
        for tkey in results['row_random']:
            if li in results['row_random'][tkey]:
                row_cos.extend(results['row_random'][tkey][li]['cos'])
        
        # 全空间随机方向
        full_cos = []
        for tkey in results['full_random']:
            if li in results['full_random'][tkey]:
                full_cos.extend(results['full_random'][tkey][li]['cos'])
        
        sem_mean = np.mean(sem_cos) if sem_cos else 0
        row_mean = np.mean(row_cos) if row_cos else 0
        full_mean = np.mean(full_cos) if full_cos else 0
        
        # 语义方向 vs row-space随机的保持差异
        # 如果语义方向保持更好 → row-space保持有语义意义
        sem_vs_row = sem_mean - row_mean
        
        summary[li] = {
            'semantic_cos': float(sem_mean),
            'row_random_cos': float(row_mean),
            'full_random_cos': float(full_mean),
            'sem_vs_row_diff': float(sem_vs_row),
            'n_sem': len(sem_cos),
            'n_row': len(row_cos),
            'n_full': len(full_cos),
        }
        
        print(f"  L{li:>3d}: sem_cos={sem_mean:.4f} row_cos={row_mean:.4f} full_cos={full_mean:.4f} "
              f"sem-row={sem_vs_row:+.4f} ({'✓semantic' if sem_vs_row > 0.02 else '≈neutral' if abs(sem_vs_row) < 0.02 else '✗anti-semantic'})")
    
    # 按语义类别分析: 语法词 vs 内容词
    grammar_attrs = ["not", "the", "is", "and", "he", "she"]
    content_attrs = ["red", "big", "run", "science", "never", "always"]
    
    print("\n  --- By Category ---")
    for cat_name, cat_attrs in [("Grammar", grammar_attrs), ("Content", content_attrs)]:
        cat_cos_by_layer = {}
        for li in sample_layers:
            cat_cos = []
            for attr in cat_attrs:
                if attr in results['semantic'] and attr in semantic_directions:
                    if li in results['semantic'][attr]:
                        cat_cos.extend(results['semantic'][attr][li]['cos'])
            if cat_cos:
                cat_cos_by_layer[li] = np.mean(cat_cos)
        
        if cat_cos_by_layer:
            avg_cos = np.mean(list(cat_cos_by_layer.values()))
            print(f"  {cat_name}: avg_cos={avg_cos:.4f}")
    
    return results, summary, semantic_directions


# ============================================================
# Exp 2: Causal Token Sensitivity — 离散决策层面的因果灵敏度
# ============================================================
def exp2_causal_sensitivity(model, tokenizer, model_name, W_U, row_basis):
    """
    核心问题: 一个token的扰动如何改变后续生成轨迹?
    
    关键创新: 不测欧氏范数传播, 而测离散决策改变!
    - P(top1_token改变 | 微扰在position j) — 因果灵敏度
    - logit层面的KL散度
    - 生成轨迹的分支概率
    
    这才是"语言约束传播"的正确测量!
    """
    print("\n" + "="*60)
    print("Exp 2: Causal Token Sensitivity")
    print("核心: 不测欧氏范数, 测离散决策改变 — 真正的因果灵敏度!")
    print("="*60)
    
    info = get_model_info(model, model_name)
    device = get_device_for_input(model)
    d_model = info.d_model
    n_layers = info.n_layers
    
    results = {}
    
    for sent_idx in range(min(N_SENTENCES, 15)):
        prompt = TEST_PROMPTS[sent_idx]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        seq_len = input_ids.shape[1]
        last_pos = seq_len - 1
        
        # Clean forward — 获取每层每个位置的logits
        with torch.no_grad():
            out_clean = model(input_ids=input_ids, attention_mask=attention_mask,
                              output_hidden_states=True)
        clean_hs = out_clean.hidden_states
        
        # Clean logits at last position
        clean_logits = out_clean.logits[0, -1, :].float().cpu().numpy()
        clean_top1 = int(np.argmax(clean_logits))
        clean_top5 = get_topk_tokens(clean_logits, k=5)
        
        # === 在每个position注入微扰, 测量last token的logit变化 ===
        inject_positions = list(range(seq_len))
        
        for inject_pos in inject_positions:
            # 在每层注入随机扰动
            for inject_l in [0, n_layers // 3, n_layers // 2, 2 * n_layers // 3, n_layers - 1]:
                if inject_l >= n_layers:
                    continue
                
                np.random.seed(sent_idx * 10000 + inject_pos * 100 + inject_l)
                delta = np.random.randn(d_model)
                norm = np.linalg.norm(delta)
                if norm > 1e-8:
                    delta = delta / norm * EPSILON
                
                # 扰动forward
                out_perturbed = run_forward_with_perturbation_at_position(
                    model, input_ids, attention_mask, inject_l, inject_pos, delta, device)
                
                if out_perturbed is None:
                    continue
                
                perturbed_logits = out_perturbed.logits[0, -1, :].float().cpu().numpy()
                perturbed_top1 = int(np.argmax(perturbed_logits))
                perturbed_top5 = get_topk_tokens(perturbed_logits, k=5)
                
                # === 因果灵敏度指标 ===
                # 1. Top-1是否改变
                top1_changed = int(perturbed_top1 != clean_top1)
                
                # 2. Top-5 overlap
                top5_overlap = len(set(clean_top5) & set(perturbed_top5)) / 5.0
                
                # 3. KL散度 (logit分布)
                # 转为概率
                clean_probs = np.exp(clean_logits - np.max(clean_logits))
                clean_probs = clean_probs / np.sum(clean_probs)
                perturbed_probs = np.exp(perturbed_logits - np.max(perturbed_logits))
                perturbed_probs = perturbed_probs / np.sum(perturbed_probs)
                
                kl_div = np.sum(clean_probs * np.log((clean_probs + 1e-10) / (perturbed_probs + 1e-10)))
                
                # 4. Top-1 probability变化
                clean_top1_prob = clean_probs[clean_top1]
                perturbed_top1_prob = perturbed_probs[clean_top1]  # 原top1的新概率
                top1_prob_drop = clean_top1_prob - perturbed_top1_prob
                
                # 5. Max probability change (任何token)
                prob_diff = np.max(np.abs(perturbed_probs - clean_probs))
                
                # 6. 距离效应: inject_pos到last_pos的距离
                distance = last_pos - inject_pos
                
                key = f"sent{sent_idx}_pos{inject_pos}_L{inject_l}"
                results[key] = {
                    'inject_pos': inject_pos,
                    'inject_layer': inject_l,
                    'distance': distance,
                    'top1_changed': top1_changed,
                    'top5_overlap': float(top5_overlap),
                    'kl_divergence': float(kl_div),
                    'top1_prob_drop': float(top1_prob_drop),
                    'max_prob_change': float(prob_diff),
                    'clean_top1': int(clean_top1),
                    'perturbed_top1': int(perturbed_top1),
                }
    
    # === 汇总 ===
    print("\n  === Exp 2 Summary: Causal Token Sensitivity ===")
    
    # 按距离分组
    all_data = list(results.values())
    
    # 距离依赖性
    print("\n  --- Distance Dependence (inject at L0) ---")
    for dist_range, label in [((0, 2), "Near(d≤2)"), ((3, 6), "Mid(3-6)"), ((7, 100), "Far(d≥7)")]:
        subset = [d for d in all_data 
                  if d['inject_layer'] == 0 and dist_range[0] <= d['distance'] <= dist_range[1]]
        if subset:
            top1_rate = np.mean([d['top1_changed'] for d in subset])
            kl_mean = np.mean([d['kl_divergence'] for d in subset])
            top5_overlap = np.mean([d['top5_overlap'] for d in subset])
            prob_change = np.mean([d['max_prob_change'] for d in subset])
            print(f"  {label}: top1_change_rate={top1_rate:.3f}, KL={kl_mean:.6f}, "
                  f"top5_overlap={top5_overlap:.3f}, max_prob_change={prob_change:.6f}")
    
    # 按注入层分组
    print("\n  --- Layer Dependence (all positions) ---")
    for inject_l in [0, n_layers // 3, n_layers // 2, 2 * n_layers // 3, n_layers - 1]:
        if inject_l >= n_layers:
            continue
        subset = [d for d in all_data if d['inject_layer'] == inject_l]
        if subset:
            top1_rate = np.mean([d['top1_changed'] for d in subset])
            kl_mean = np.mean([d['kl_divergence'] for d in subset])
            prob_change = np.mean([d['max_prob_change'] for d in subset])
            print(f"  L{inject_l:>2d}: top1_change_rate={top1_rate:.3f}, KL={kl_mean:.6f}, "
                  f"max_prob_change={prob_change:.6f}")
    
    # 关键: "因果灵敏度"vs"欧氏耦合"的对比
    # Phase 150发现cross coupling几乎为0(欧氏)
    # 但因果灵敏度可能远非0!
    print("\n  --- Causal vs Euclidean Sensitivity ---")
    # 只看L0注入, 不同距离
    for dist in range(0, max(d['distance'] for d in all_data) + 1):
        subset = [d for d in all_data 
                  if d['inject_layer'] == 0 and d['distance'] == dist]
        if subset and len(subset) >= 3:
            top1_rate = np.mean([d['top1_changed'] for d in subset])
            if top1_rate > 0:
                print(f"  dist={dist}: top1_change_rate={top1_rate:.3f} ({len(subset)} trials) "
                      f"→ Euclidean coupling≈0 but CAUSAL sensitivity={top1_rate:.1%}!")
    
    return results


# ============================================================
# Exp 3: Rollout Statistical Invariants — 生成过程中什么守恒
# ============================================================
def exp3_rollout_invariants(model, tokenizer, model_name, W_U, row_basis):
    """
    核心问题: 生成过程中什么统计量保持不变?
    
    方法:
    1. 从同一起点多次生成(temperature>0), 统计哪些量守恒
    2. 候选守恒量:
       - attention graph的社区结构
       - token type distribution (语法词/内容词比例)
       - 段落结构的macro pattern
       - 每层的hidden state norm
       - 每层的attention entropy
    3. 与随机初始化模型对比
    """
    print("\n" + "="*60)
    print("Exp 3: Rollout Statistical Invariants")
    print("核心: 生成过程中什么统计量保持不变?")
    print("="*60)
    
    info = get_model_info(model, model_name)
    device = get_device_for_input(model)
    n_layers = info.n_layers
    
    # 使用3个不同的提示
    prompts = TEST_PROMPTS[:3]
    
    results = {}
    
    for prompt_idx, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        rollout_data = []
        
        for rollout_idx in range(N_ROLLOUTS):
            # 生成
            with torch.no_grad():
                gen_ids = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=30,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                )
            
            gen_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
            gen_token_ids = gen_ids[0].cpu().numpy()
            
            # 获取生成部分的hidden states + attentions
            gen_attention_mask = torch.ones_like(gen_ids)
            with torch.no_grad():
                out = model(input_ids=gen_ids, attention_mask=gen_attention_mask,
                            output_hidden_states=True, output_attentions=True)
            
            hs = out.hidden_states
            attentions = out.attentions
            
            # === 收集统计量 ===
            stats = {
                'gen_text': gen_text,
                'gen_length': len(gen_token_ids),
                'layer_norms': [],
                'layer_entropy': [],
                'token_type_ratio': 0.0,
            }
            
            # 1. 每层hidden state norm (最后一个token)
            for li in range(len(hs)):
                norm_val = float(hs[li][0, -1, :].float().norm().cpu())
                stats['layer_norms'].append(norm_val)
            
            # 2. 每层attention entropy (最后一个token)
            for li in range(min(len(attentions), n_layers)):
                if attentions[li] is not None:
                    attn = attentions[li][0].float().cpu().numpy()  # [n_heads, seq, seq]
                    last_pos = attn.shape[2] - 1
                    # 平均所有head
                    mean_attn = attn.mean(axis=0)  # [seq, seq]
                    last_attn = mean_attn[last_pos, :]  # [seq]
                    entropy = -np.sum(last_attn * np.log(last_attn + 1e-10))
                    max_entropy = np.log(last_pos + 1) if last_pos > 0 else 1
                    stats['layer_entropy'].append(float(entropy / max_entropy))
                else:
                    stats['layer_entropy'].append(0.0)
            
            # 3. Token type ratio (简单版: 有多少token是常见功能词)
            # 使用简单的启发式: 长度<=3的token更可能是功能词
            gen_tokens = [tokenizer.decode([t]) for t in gen_token_ids]
            short_tokens = sum(1 for t in gen_tokens if len(t.strip()) <= 3)
            stats['token_type_ratio'] = short_tokens / len(gen_tokens) if gen_tokens else 0
            
            rollout_data.append(stats)
        
        # === 计算守恒量 ===
        # 对每个统计量, 计算跨rollout的变异系数(CV)
        # CV越低 → 越守恒
        
        # 1. Layer norms的CV
        layer_norm_cv = []
        for li in range(len(rollout_data[0]['layer_norms'])):
            norms = [rd['layer_norms'][li] for rd in rollout_data]
            mean_norm = np.mean(norms)
            std_norm = np.std(norms)
            cv = std_norm / (mean_norm + 1e-10)
            layer_norm_cv.append(cv)
        
        # 2. Layer entropy的CV
        layer_entropy_cv = []
        for li in range(len(rollout_data[0]['layer_entropy'])):
            entropies = [rd['layer_entropy'][li] for rd in rollout_data]
            mean_ent = np.mean(entropies)
            std_ent = np.std(entropies)
            cv = std_ent / (mean_ent + 1e-10)
            layer_entropy_cv.append(cv)
        
        # 3. Token type ratio的CV
        ttr_values = [rd['token_type_ratio'] for rd in rollout_data]
        ttr_cv = np.std(ttr_values) / (np.mean(ttr_values) + 1e-10)
        
        # 4. 生成长度的CV
        len_values = [rd['gen_length'] for rd in rollout_data]
        len_cv = np.std(len_values) / (np.mean(len_values) + 1e-10)
        
        results[f"prompt{prompt_idx}"] = {
            'prompt': prompt,
            'n_rollouts': N_ROLLOUTS,
            'layer_norm_cv': layer_norm_cv,
            'layer_entropy_cv': layer_entropy_cv,
            'token_type_ratio_cv': float(ttr_cv),
            'gen_length_cv': float(len_cv),
            'mean_layer_norm_cv': float(np.mean(layer_norm_cv)),
            'mean_layer_entropy_cv': float(np.mean(layer_entropy_cv)),
            # 原始数据
            'rollout_stats': [{
                'gen_text': rd['gen_text'],
                'gen_length': rd['gen_length'],
                'token_type_ratio': rd['token_type_ratio'],
                'final_norm': rd['layer_norms'][-1] if rd['layer_norms'] else 0,
                'final_entropy': rd['layer_entropy'][-1] if rd['layer_entropy'] else 0,
            } for rd in rollout_data],
        }
    
    # === 汇总 ===
    print("\n  === Exp 3 Summary: Rollout Statistical Invariants ===")
    
    for prompt_idx in range(len(prompts)):
        key = f"prompt{prompt_idx}"
        if key not in results:
            continue
        r = results[key]
        print(f"\n  Prompt: '{prompts[prompt_idx][:40]}...'")
        print(f"    Layer norm CV (mean):    {r['mean_layer_norm_cv']:.4f}")
        print(f"    Layer entropy CV (mean): {r['mean_layer_entropy_cv']:.4f}")
        print(f"    Token type ratio CV:     {r['token_type_ratio_cv']:.4f}")
        print(f"    Gen length CV:           {r['gen_length_cv']:.4f}")
        
        # 判断: CV < 0.1 → 强守恒, CV < 0.3 → 弱守恒, CV > 0.3 → 不守恒
        for name, cv in [("Layer norm", r['mean_layer_norm_cv']),
                         ("Layer entropy", r['mean_layer_entropy_cv']),
                         ("Token type ratio", r['token_type_ratio_cv']),
                         ("Gen length", r['gen_length_cv'])]:
            if cv < 0.1:
                status = "STRONG INVARIANT ★"
            elif cv < 0.3:
                status = "weak invariant"
            else:
                status = "NOT invariant"
            print(f"    {name}: {status} (CV={cv:.4f})")
    
    # 跨prompt的平均CV
    all_norm_cv = [results[f"prompt{i}"]['mean_layer_norm_cv'] 
                   for i in range(len(prompts)) if f"prompt{i}" in results]
    all_ent_cv = [results[f"prompt{i}"]['mean_layer_entropy_cv'] 
                  for i in range(len(prompts)) if f"prompt{i}" in results]
    all_ttr_cv = [results[f"prompt{i}"]['token_type_ratio_cv'] 
                  for i in range(len(prompts)) if f"prompt{i}" in results]
    
    print(f"\n  === Cross-Prompt Averages ===")
    print(f"    Layer norm CV:    {np.mean(all_norm_cv):.4f}")
    print(f"    Layer entropy CV: {np.mean(all_ent_cv):.4f}")
    print(f"    Token type ratio CV: {np.mean(all_ttr_cv):.4f}")
    
    return results


# ============================================================
# 主函数
# ============================================================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    print(f"Phase 151: Causal Language Dynamics")
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
    row_basis, k, sv_data, k_90, k_95, k_99, U_full = get_row_null_bases(W_U, n_components=None)
    print(f"W_U: shape={W_U.shape}, effective_rank(95%)={k}, null_dim={W_U.shape[1]-k}")
    print(f"SVD time: {time.time()-t0:.1f}s")

    # 运行3个实验
    print("\n" + "#"*60)
    print("# Running Experiments")
    print("#"*60)

    exp1_results, exp1_summary, semantic_directions = exp1_semantic_vs_row(
        model, tokenizer, model_name, W_U, row_basis, U_full)

    exp2_results = exp2_causal_sensitivity(
        model, tokenizer, model_name, W_U, row_basis)

    exp3_results = exp3_rollout_invariants(
        model, tokenizer, model_name, W_U, row_basis)

    # 保存结果
    all_results = {
        "phase": 151,
        "model": model_name,
        "timestamp": timestamp,
        "model_info": {
            "class": info.model_class,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
        },
        "W_U_analysis": {
            "shape": list(W_U.shape),
            "effective_rank_95": k,
            "rank_90": k_90,
            "rank_95": k_95,
            "rank_99": k_99,
        },
        "exp1_summary": exp1_summary,
        "exp2_key_findings": {
            "avg_top1_change_rate": float(np.mean([d['top1_changed'] for d in exp2_results.values()])),
            "avg_kl_divergence": float(np.mean([d['kl_divergence'] for d in exp2_results.values()])),
            "avg_max_prob_change": float(np.mean([d['max_prob_change'] for d in exp2_results.values()])),
        },
        "exp3_results": exp3_results,
    }

    result_file = OUTPUT_DIR / f"phase151_{model_name}_{timestamp}.json"

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
