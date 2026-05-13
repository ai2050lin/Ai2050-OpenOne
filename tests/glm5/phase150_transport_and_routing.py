"""
Phase 150: Attention Transport Geometry + Conditional Transfer Matrix + Routing Topology
========================================================================================

用户核心批评(全部正确):
  1. Phase 148测的是"最终null_ratio"不是"传播动力学" — 像看墨水最终均匀分布就否定流体动力学
  2. 需要"条件转移矩阵": P(row→null) vs P(null→row) 的层间不对称性
  3. row-space定义可能错(top-200 ≠ semantic directions)
  4. "方向丢失" ≠ "没有几何结构" — 湍流中局部方向混合但统计结构稳定
  5. Token耦合1/10不等于不重要 — 弱耦合×多层累积=强约束
  6. absence of evidence ≠ evidence of absence

三个核心实验:
  Exp 1: Conditional Transfer Matrix — 逐层P(row→null) vs P(null→row)不对称性
    - 真正判别主动路由 vs 被动mixing
    - 若P(row→null) > P(null→row) → 主动路由存在
    - 若P(row→null) ≈ P(n→r) → 被动mixing

  Exp 2: Attention Transport Geometry — token-to-token信息运输的完整图谱
    - ∂h_i^(l+k)/∂h_j^(l) 的稀疏结构/谱结构/社区结构
    - 哪些token pair形成稳定耦合
    - 哪些attention head负责运输

  Exp 3: Routing Topology — 稀疏激活拓扑
    - 哪些head被激活（高注意力权重）
    - 哪些MLP neuron被激活
    - 激活模式的稳定性（不同输入是否激活相同head）

用法:
  python tests/glm5/phase150_transport_and_routing.py qwen3
  python tests/glm5/phase150_transport_and_routing.py glm4
  python tests/glm5/phase150_transport_and_routing.py deepseek7b
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
EPSILON = 1.0       # 中等扰动 — 在Phase 149b中eps=1.0给出最好的cos(δ,δ')
N_SENTENCES = 20    # 加大数据量 — 用户强调重要结果需要加大数据量
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
    d = W_U.shape[1]
    
    # 先计算W_U^T的SVD来确定有效rank
    # 使用随机化SVD来处理大矩阵
    from scipy.sparse.linalg import svds
    
    W_U_T = W_U.T.astype(np.float32)  # [d_model, vocab]
    
    # 计算较多分量以找到有效rank
    k_max = min(500, min(W_U_T.shape) - 2)  # 最多500个分量
    k_max = max(k_max, 10)
    
    print(f"  Computing SVD of W_U^T (shape={W_U_T.shape}), k_max={k_max}...")
    t0 = time.time()
    U_full, s_full, Vt_full = svds(W_U_T, k=k_max)
    # svds返回按升序排列, 需要翻转
    idx = np.argsort(-s_full)
    U_full = U_full[:, idx]
    s_full = s_full[idx]
    print(f"  SVD done in {time.time()-t0:.1f}s")
    
    # 确定有效rank: 找到singular values显著大于噪声的位置
    # 使用能量阈值: 保留90%的总能量
    total_energy = np.sum(s_full ** 2)
    cumulative_energy = np.cumsum(s_full ** 2)
    # 保留90%能量的最小k
    k_90 = np.searchsorted(cumulative_energy, 0.90 * total_energy) + 1
    # 保留95%能量的最小k
    k_95 = np.searchsorted(cumulative_energy, 0.95 * total_energy) + 1
    # 保留99%能量的最小k
    k_99 = np.searchsorted(cumulative_energy, 0.99 * total_energy) + 1
    
    print(f"  Singular values: top-5 = {s_full[:5].tolist()}")
    print(f"  Singular values: bottom-5 = {s_full[-5:].tolist()}")
    print(f"  Effective rank (90% energy): {k_90}")
    print(f"  Effective rank (95% energy): {k_95}")
    print(f"  Effective rank (99% energy): {k_99}")
    print(f"  Total energy: {total_energy:.2f}")
    
    # 使用用户指定的n_components, 或95%能量的rank
    if n_components is None:
        k = k_95
    else:
        k = min(n_components, k_max)
    
    row_basis = U_full[:, :k].T  # [k, d_model]
    
    return row_basis, k, s_full[:k_max], k_90, k_95, k_99


def project_to_null(vec, row_basis):
    """将向量投影到null space"""
    row_component = row_basis.T @ (row_basis @ vec)
    return vec - row_component


def project_to_row(vec, row_basis):
    """将向量投影到row space"""
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


# ============================================================
# Exp 1: Conditional Transfer Matrix — 逐层P(row→null) vs P(null→row)
# ============================================================
def exp1_conditional_transfer(model, tokenizer, model_name, W_U, row_basis, sv_data):
    print("\n" + "="*60)
    print("Exp 1: Conditional Transfer Matrix")
    print("核心: 逐层测量 P(row→null) vs P(null→row) — 判别主动路由")
    print("="*60)
    
    info = get_model_info(model, model_name)
    device = get_device_for_input(model)
    d_model = info.d_model
    n_layers = info.n_layers
    
    # 采样层: 每层都测 (这是最关键实验, 需要细粒度)
    sample_layers = list(range(0, n_layers + 1, max(1, n_layers // 12)))
    sample_layers = sorted(set(sample_layers + [n_layers]))
    
    # 两种注入方向: row-space 和 null-space
    # 每种方向, 在每层注入, 然后测量下一层的row/null能量分布
    
    results_row_input = {}  # row-space注入: 每层后的(row_energy, null_ratio)
    results_null_input = {}  # null-space注入: 每层后的(row_energy, null_ratio)
    results_random_input = {}  # 随机注入: 每层后的(row_energy, null_ratio)
    
    # 转移矩阵: T[l] = [[P(r→r), P(r→n)], [P(n→r), P(n→n)]]
    # P(r→n) = 从row-space注入后null_ratio的变化
    # P(n→r) = 从null-space注入后row_energy的变化
    
    for sent_idx in range(N_SENTENCES):
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
        
        for dir_type in ["row", "null", "random"]:
            # 注入层: L0 (只从第一层注入, 测量传播)
            inject_l = 0
            
            np.random.seed(sent_idx * 100 + {"row": 0, "null": 50, "random": 99}[dir_type])
            raw_vec = np.random.randn(d_model)
            
            if dir_type == "row":
                delta = project_to_row(raw_vec, row_basis)
            elif dir_type == "null":
                delta = project_to_null(raw_vec, row_basis)
            else:
                delta = raw_vec
            
            norm = np.linalg.norm(delta)
            if norm < 1e-8:
                continue
            delta = delta / norm * EPSILON
            
            # 记录初始row/null比例
            init_row_e, init_null_r = compute_row_energy(delta, row_basis)
            
            # 扰动forward
            out_perturbed = run_forward_with_perturbation_at_position(
                model, input_ids, attention_mask, inject_l, last_pos, delta, device)
            
            if out_perturbed is None:
                continue
            
            # 逐层测量
            for li in sample_layers:
                perturbed_vec = out_perturbed.hidden_states[li][0, last_pos, :].float().cpu().numpy()
                clean_vec = clean_hs[li][0, last_pos, :].float().cpu().numpy()
                delta_prop = perturbed_vec - clean_vec
                
                row_e, null_r = compute_row_energy(delta_prop, row_basis)
                
                key = f"L{li}"
                target_dict = {"row": results_row_input, 
                               "null": results_null_input,
                               "random": results_random_input}[dir_type]
                
                if key not in target_dict:
                    target_dict[key] = {"row_energy": [], "null_ratio": [], "delta_norm": []}
                
                target_dict[key]["row_energy"].append(row_e)
                target_dict[key]["null_ratio"].append(null_r)
                target_dict[key]["delta_norm"].append(np.linalg.norm(delta_prop))
    
    # === 汇总: 条件转移矩阵 ===
    print("\n  === Conditional Transfer Matrix ===")
    print(f"  (avg over {N_SENTENCES} sentences, inject at L0)")
    
    # 计算每层的转移概率
    print(f"\n  {'Layer':>6} {'P(r→r)':>8} {'P(r→n)':>8} {'P(n→r)':>8} {'P(n→n)':>8} {'Asymmetry':>12}")
    print(f"  {'-----':>6} {'------':>8} {'------':>8} {'------':>8} {'------':>8} {'----------':>12}")
    
    transfer_matrix = {}
    
    for li in sample_layers:
        key = f"L{li}"
        
        # P(r→r): row-input后row_energy保持率
        rr = np.mean(results_row_input.get(key, {}).get("row_energy", [0]))
        # P(r→n): row-input后null_ratio = 1 - rr (不, 需要更仔细)
        # 实际: row-input注入后, 传播到第li层时, 能量在row space的比例 = rr
        # 转移到null space的比例 = 1 - rr
        rn = 1.0 - rr
        
        # P(n→r): null-input后row_energy = 泄漏到row space的比例
        nr = np.mean(results_null_input.get(key, {}).get("row_energy", [0]))
        # P(n→n): null-input后null_ratio保持
        nn = 1.0 - nr
        
        # 不对称性: P(r→n) - P(n→r)
        # 如果P(r→n) > P(n→r) → 主动将row-space分量推到null-space → 主动路由
        # 如果P(r→n) ≈ P(n→r) → 对称mixing → 被动扩散
        asymmetry = rn - nr
        
        # random-input的null_ratio作为参考
        random_nr = np.mean(results_random_input.get(key, {}).get("null_ratio", [0.92]))
        
        transfer_matrix[li] = {
            "P_rr": float(rr),
            "P_rn": float(rn),
            "P_nr": float(nr),
            "P_nn": float(nn),
            "asymmetry": float(asymmetry),
            "random_null_ratio": float(random_nr),
        }
        
        print(f"  L{li:>4d} {rr:>8.4f} {rn:>8.4f} {nr:>8.4f} {nn:>8.4f} {asymmetry:>+12.6f}")
    
    # === 关键判据 ===
    print("\n  *** Key Criteria ***")
    
    # 统计各层的不对称性
    asymmetries = [transfer_matrix[li]["asymmetry"] for li in sample_layers]
    mean_asym = np.mean(asymmetries)
    std_asym = np.std(asymmetries)
    max_asym = np.max(asymmetries)
    min_asym = np.min(asymmetries)
    
    print(f"  Asymmetry (P(r→n) - P(n→r)) statistics:")
    print(f"    Mean: {mean_asym:+.6f}")
    print(f"    Std:  {std_asym:.6f}")
    print(f"    Max:  {max_asym:+.6f}")
    print(f"    Min:  {min_asym:+.6f}")
    
    # t-test: 是否显著不为0
    from scipy import stats
    t_stat, p_value = stats.ttest_1samp(asymmetries, 0)
    print(f"    t-test vs 0: t={t_stat:.4f}, p={p_value:.4f}")
    
    if p_value < 0.05 and mean_asym > 0:
        print(f"  → SIGNIFICANT ASYMMETRY (P(r→n) > P(n→r)): Active routing evidence!")
        print(f"  → Row-space扰动被主动推到null-space")
    elif p_value < 0.05 and mean_asym < 0:
        print(f"  → SIGNIFICANT ASYMMETRY (P(n→r) > P(r→n)): Reverse routing!")
        print(f"  → Null-space扰动被主动拉到row-space — 反直觉!")
    else:
        print(f"  → NO SIGNIFICANT ASYMMETRY: P(r→n) ≈ P(n→r)")
        print(f"  → Consistent with passive mixing/diffusion")
        print(f"  → But: absence of evidence ≠ evidence of absence!")
    
    # 逐层检查: 早期层是否有不对称性 (可能被后期mixing覆盖)
    early_layers = [li for li in sample_layers if li <= n_layers // 3]
    mid_layers = [li for li in sample_layers if n_layers // 3 < li <= 2 * n_layers // 3]
    late_layers = [li for li in sample_layers if li > 2 * n_layers // 3]
    
    for name, layer_set in [("Early", early_layers), ("Mid", mid_layers), ("Late", late_layers)]:
        if layer_set:
            asyms = [transfer_matrix[li]["asymmetry"] for li in layer_set]
            print(f"  {name} layers: mean_asym={np.mean(asyms):+.6f}, std={np.std(asyms):.6f}")
    
    return transfer_matrix, results_row_input, results_null_input, results_random_input


# ============================================================
# Exp 2: Attention Transport Geometry — token-to-token信息运输
# ============================================================
def exp2_attention_transport(model, tokenizer, model_name, W_U, row_basis):
    print("\n" + "="*60)
    print("Exp 2: Attention Transport Geometry")
    print("核心: token-to-token Jacobian的稀疏结构/谱结构")
    print("="*60)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    device = get_device_for_input(model)
    d_model = info.d_model
    n_layers = info.n_layers
    n_heads = info.d_model  # 会从模型获取
    
    # 获取head数量 — 从实际attention输出推断
    # 先做一次forward获取真实的n_heads
    with torch.no_grad():
        test_inputs = tokenizer("test", return_tensors="pt", truncation=True, max_length=64)
        test_out = model(input_ids=test_inputs["input_ids"].to(device),
                         attention_mask=test_inputs["attention_mask"].to(device),
                         output_attentions=True)
    
    if test_out.attentions and test_out.attentions[0] is not None:
        n_heads = test_out.attentions[0].shape[1]  # [1, n_heads, seq, seq]
    else:
        layer0 = layers[0]
        if hasattr(layer0.self_attn, 'num_heads'):
            n_heads = layer0.self_attn.num_heads
        elif hasattr(layer0.self_attn, 'n_heads'):
            n_heads = layer0.self_attn.n_heads
        else:
            n_heads = d_model // 64
    
    del test_out, test_inputs
    print(f"  n_heads={n_heads}, d_model={d_model}")
    
    # 测量几个不同距离的token coupling
    # ∂h_i^(l+k)/∂h_j^(l) — 用有限差分法
    
    results = {}
    
    for sent_idx in range(min(N_SENTENCES, 10)):  # 10个句子足够
        prompt = TEST_PROMPTS[sent_idx]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        seq_len = input_ids.shape[1]
        last_pos = seq_len - 1
        
        # Clean forward with attention weights
        with torch.no_grad():
            out_clean = model(input_ids=input_ids, attention_mask=attention_mask,
                              output_hidden_states=True, output_attentions=True)
        clean_hs = out_clean.hidden_states
        clean_attentions = out_clean.attentions  # tuple of [1, n_heads, seq, seq]
        
        # === Part A: Attention Pattern分析 ===
        # 收集每层的attention pattern
        attn_patterns = []
        for li in range(min(len(clean_attentions), n_layers)):
            if clean_attentions[li] is not None:
                # [1, n_heads, seq, seq] → 平均heads → [seq, seq]
                attn = clean_attentions[li][0].float().cpu().numpy()  # [n_heads, seq, seq]
                attn_mean = attn.mean(axis=0)  # [seq, seq]
                attn_patterns.append(attn_mean)
            else:
                attn_patterns.append(None)
        
        # 分析attention的稀疏性
        for li_idx, attn in enumerate(attn_patterns):
            if attn is None:
                continue
            li = li_idx  # 层编号
            
            # 对last token的attention pattern
            last_attn = attn[last_pos, :]  # [seq] — last token对其他token的attention
            
            # 稀疏性: 有多少token获得了>5%的attention
            threshold = 0.05
            n_active = np.sum(last_attn > threshold)
            max_attn = np.max(last_attn)
            entropy = -np.sum(last_attn * np.log(last_attn + 1e-10))
            max_entropy = np.log(seq_len)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            # 长距离attention: last token对前半部分token的attention
            mid_pos = seq_len // 2
            long_range_attn = np.sum(last_attn[:mid_pos])
            
            key = f"sent{sent_idx}_L{li}"
            results[key] = {
                'n_active_tokens': int(n_active),
                'max_attn': float(max_attn),
                'entropy': float(entropy),
                'normalized_entropy': float(normalized_entropy),
                'long_range_attn': float(long_range_attn),
                'seq_len': seq_len,
            }
        
        # === Part B: Token Coupling Jacobian (有限差分) ===
        # 在每个position注入扰动, 测量其他position的响应
        # 只测几个关键层对: L0→L5, L0→L17, L0→L35
        
        layer_pairs = [(0, min(5, n_layers-1)), 
                       (0, min(n_layers//2, n_layers-1)),
                       (0, n_layers-1)]
        
        for inject_l, measure_l in layer_pairs:
            if inject_l >= n_layers or measure_l >= n_layers:
                continue
            
            # 采样3个注入位置: first, middle, last
            inject_positions = [0, seq_len // 2, last_pos]
            
            for inject_pos in inject_positions:
                np.random.seed(sent_idx * 1000 + inject_l * 100 + inject_pos)
                delta = np.random.randn(d_model)
                norm = np.linalg.norm(delta)
                if norm > 1e-8:
                    delta = delta / norm * EPSILON
                
                # 扰动forward
                out_perturbed = run_forward_with_perturbation_at_position(
                    model, input_ids, attention_mask, inject_l, inject_pos, delta, device)
                
                if out_perturbed is None:
                    continue
                
                # 在measure_l测量所有position的响应
                perturbed_vecs = out_perturbed.hidden_states[measure_l][0].float().cpu().numpy()
                clean_vecs = clean_hs[measure_l][0].float().cpu().numpy()
                
                coupling_vec = np.zeros(seq_len)
                for pos_i in range(seq_len):
                    delta_i = perturbed_vecs[pos_i] - clean_vecs[pos_i]
                    coupling_vec[pos_i] = np.linalg.norm(delta_i)
                
                # 归一化
                total_response = np.sum(coupling_vec)
                if total_response > 1e-10:
                    coupling_normalized = coupling_vec / total_response
                else:
                    coupling_normalized = coupling_vec
                
                # 耦合稀疏性
                n_active_coupling = np.sum(coupling_normalized > 0.05)
                coupling_entropy = -np.sum(coupling_normalized * np.log(coupling_normalized + 1e-10))
                coupling_max_entropy = np.log(seq_len) if seq_len > 1 else 1
                coupling_norm_entropy = coupling_entropy / coupling_max_entropy
                
                # 自耦合 vs 交叉耦合
                self_coupling = coupling_vec[inject_pos]
                cross_coupling = np.mean([coupling_vec[i] for i in range(seq_len) if i != inject_pos])
                
                # 距离依赖: 近邻vs远距离
                distances = np.abs(np.arange(seq_len) - inject_pos)
                near_mask = distances <= 2
                far_mask = distances > 2
                near_coupling = np.mean(coupling_vec[near_mask]) if np.any(near_mask) else 0
                far_coupling = np.mean(coupling_vec[far_mask]) if np.any(far_mask) else 0
                
                key = f"sent{sent_idx}_L{inject_l}toL{measure_l}_inj{inject_pos}"
                results[f"coupling_{key}"] = {
                    'inject_layer': inject_l,
                    'measure_layer': measure_l,
                    'inject_pos': inject_pos,
                    'self_coupling': float(self_coupling),
                    'cross_coupling': float(cross_coupling),
                    'self_cross_ratio': float(self_coupling / (cross_coupling + 1e-10)),
                    'n_active_coupling': int(n_active_coupling),
                    'coupling_norm_entropy': float(coupling_norm_entropy),
                    'near_coupling': float(near_coupling),
                    'far_coupling': float(far_coupling),
                    'near_far_ratio': float(near_coupling / (far_coupling + 1e-10)),
                    'coupling_profile': coupling_normalized.tolist(),
                }
    
    # === 汇总分析 ===
    print("\n  === Exp 2 Summary: Attention Transport Geometry ===")
    
    # Part A: Attention Pattern汇总
    print("\n  --- Part A: Attention Patterns ---")
    for li in range(min(n_layers, len(attn_patterns) if 'attn_patterns' in dir() else 0)):
        pattern_data = [(k, v) for k, v in results.items() 
                        if not k.startswith('coupling_') and f'_L{li}_' in k]
        if pattern_data:
            n_actives = [v['n_active_tokens'] for _, v in pattern_data]
            entropies = [v['normalized_entropy'] for _, v in pattern_data]
            long_ranges = [v['long_range_attn'] for _, v in pattern_data]
            print(f"  L{li:>2d}: n_active={np.mean(n_actives):.1f}/{pattern_data[0][1]['seq_len']}, "
                  f"entropy={np.mean(entropies):.3f}, long_range={np.mean(long_ranges):.3f}")
    
    # Part B: Token Coupling汇总
    print("\n  --- Part B: Token Coupling ---")
    coupling_data = {k: v for k, v in results.items() if k.startswith('coupling_')}
    
    for inject_l, measure_l in layer_pairs:
        if inject_l >= n_layers or measure_l >= n_layers:
            continue
        
        group = [(k, v) for k, v in coupling_data.items()
                 if v['inject_layer'] == inject_l and v['measure_layer'] == measure_l]
        
        if group:
            self_ratios = [v['self_cross_ratio'] for _, v in group]
            near_far_ratios = [v['near_far_ratio'] for _, v in group]
            coupling_entropies = [v['coupling_norm_entropy'] for _, v in group]
            n_active = [v['n_active_coupling'] for _, v in group]
            
            print(f"  L{inject_l}→L{measure_l}: "
                  f"self/cross={np.mean(self_ratios):.1f}x, "
                  f"near/far={np.mean(near_far_ratios):.1f}x, "
                  f"entropy={np.mean(coupling_entropies):.3f}, "
                  f"n_active={np.mean(n_active):.1f}")
    
    return results


# ============================================================
# Exp 3: Routing Topology — 稀疏激活拓扑
# ============================================================
def exp3_routing_topology(model, tokenizer, model_name, W_U, row_basis):
    print("\n" + "="*60)
    print("Exp 3: Routing Topology — 稀疏激活拓扑")
    print("核心: 哪些head被激活, 激活模式是否稳定")
    print("="*60)
    
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    device = get_device_for_input(model)
    d_model = info.d_model
    n_layers = info.n_layers
    
    # 获取head数量
    # 获取head数量 — 从实际attention输出推断
    with torch.no_grad():
        test_inputs = tokenizer("test", return_tensors="pt", truncation=True, max_length=64)
        test_out = model(input_ids=test_inputs["input_ids"].to(device),
                         attention_mask=test_inputs["attention_mask"].to(device),
                         output_attentions=True)
    n_heads = test_out.attentions[0].shape[1] if test_out.attentions and test_out.attentions[0] is not None else d_model // 64
    del test_out, test_inputs
    
    # 采样层: 早/中/晚各2层
    sample_layer_indices = [0, min(4, n_layers-1), 
                            n_layers//3, n_layers//2, 
                            2*n_layers//3, min(n_layers-2, n_layers-1)]
    sample_layer_indices = sorted(set([li for li in sample_layer_indices if li < n_layers]))
    
    results = {}
    
    for sent_idx in range(min(N_SENTENCES, 12)):  # 12个句子
        prompt = TEST_PROMPTS[sent_idx]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        seq_len = input_ids.shape[1]
        last_pos = seq_len - 1
        
        # Forward with attentions
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True, output_attentions=True)
        
        attentions = out.attentions  # tuple of [1, n_heads, seq, seq]
        
        for li_idx, li in enumerate(sample_layer_indices):
            if li >= len(attentions) or attentions[li] is None:
                continue
            
            attn = attentions[li][0].float().cpu().numpy()  # [n_heads, seq, seq]
            
            # === Per-head分析 ===
            head_data = []
            for h in range(n_heads):
                head_attn = attn[h]  # [seq, seq]
                
                # Last token的attention pattern
                last_head_attn = head_attn[last_pos, :]  # [seq]
                
                # Head稀疏性
                entropy_h = -np.sum(last_head_attn * np.log(last_head_attn + 1e-10))
                max_entropy_h = np.log(seq_len) if seq_len > 1 else 1
                norm_entropy_h = entropy_h / max_entropy_h
                
                # Head的top-1 target position
                top1_pos = np.argmax(last_head_attn)
                
                # Head的long-range倾向
                mid_pos = seq_len // 2
                long_range_h = np.sum(last_head_attn[:mid_pos])
                
                # Head的diagonal倾向 (attention to self position)
                diag_attn = head_attn[last_pos, last_pos] if last_pos < seq_len else 0
                
                # Head的local倾向 (attention to nearby positions)
                local_range = max(1, seq_len // 10)
                local_start = max(0, last_pos - local_range)
                local_end = min(seq_len, last_pos + local_range + 1)
                local_attn_h = np.sum(last_head_attn[local_start:local_end])
                
                head_data.append({
                    'head': h,
                    'entropy': float(norm_entropy_h),
                    'top1_pos': int(top1_pos),
                    'long_range': float(long_range_h),
                    'diagonal': float(diag_attn),
                    'local': float(local_attn_h),
                })
            
            key = f"sent{sent_idx}_L{li}"
            results[key] = head_data
    
    # === 汇总分析 ===
    print("\n  === Exp 3 Summary: Routing Topology ===")
    
    # 对每个采样层, 分析head的激活模式
    for li in sample_layer_indices:
        layer_data = [(k, v) for k, v in results.items() if f'_L{li}' in k]
        if not layer_data:
            continue
        
        # 收集所有句子的head数据
        all_head_entropies = {h: [] for h in range(n_heads)}
        all_head_long_range = {h: [] for h in range(n_heads)}
        all_head_local = {h: [] for h in range(n_heads)}
        
        for _, heads in layer_data:
            for hd in heads:
                h = hd['head']
                all_head_entropies[h].append(hd['entropy'])
                all_head_long_range[h].append(hd['long_range'])
                all_head_local[h].append(hd['local'])
        
        # 分类head
        # 稀疏head: 低entropy (<0.3) → focus on specific tokens
        # 广播head: 高entropy (>0.7) → distribute attention broadly
        # 长距离head: long_range > 0.5
        # 局部head: local > 0.8
        
        sparse_heads = [h for h in range(n_heads) 
                        if np.mean(all_head_entropies[h]) < 0.3]
        broad_heads = [h for h in range(n_heads) 
                       if np.mean(all_head_entropies[h]) > 0.7]
        long_range_heads = [h for h in range(n_heads) 
                           if np.mean(all_head_long_range[h]) > 0.5]
        local_heads = [h for h in range(n_heads) 
                      if np.mean(all_head_local[h]) > 0.8]
        
        # Head稳定性: 跨句子entropy的标准差
        entropy_stability = {h: np.std(all_head_entropies[h]) 
                            for h in range(n_heads) if all_head_entropies[h]}
        stable_heads = [h for h, std in entropy_stability.items() if std < 0.05]
        variable_heads = [h for h, std in entropy_stability.items() if std > 0.15]
        
        print(f"\n  L{li:>2d} (n_heads={n_heads}):")
        print(f"    Sparse heads (ent<0.3): {len(sparse_heads)} — {sparse_heads[:5]}{'...' if len(sparse_heads)>5 else ''}")
        print(f"    Broad heads (ent>0.7):  {len(broad_heads)} — {broad_heads[:5]}{'...' if len(broad_heads)>5 else ''}")
        print(f"    Long-range heads:       {len(long_range_heads)} — {long_range_heads[:5]}{'...' if len(long_range_heads)>5 else ''}")
        print(f"    Local heads:            {len(local_heads)} — {local_heads[:5]}{'...' if len(local_heads)>5 else ''}")
        print(f"    Stable heads (σ<0.05):  {len(stable_heads)}")
        print(f"    Variable heads (σ>0.15):{len(variable_heads)}")
        
        # 关键: 稀疏+稳定head = 可能的路由拓扑节点
        routing_heads = list(set(sparse_heads) & set(stable_heads))
        print(f"    *** Routing heads (sparse+stable): {len(routing_heads)} — {routing_heads[:5]} ***")
        
        results[f"summary_L{li}"] = {
            'n_sparse': len(sparse_heads),
            'n_broad': len(broad_heads),
            'n_long_range': len(long_range_heads),
            'n_local': len(local_heads),
            'n_stable': len(stable_heads),
            'n_routing': len(routing_heads),
            'routing_heads': routing_heads[:10],
        }
    
    return results


# ============================================================
# 主函数
# ============================================================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    print(f"Phase 150: Transport Geometry + Conditional Transfer + Routing Topology")
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

    # 获取W_U和row/null space — 使用有效rank!
    t0 = time.time()
    W_U = get_W_U(model, model_name)
    row_basis, k, sv_data, k_90, k_95, k_99 = get_row_null_bases(W_U, n_components=None)
    print(f"W_U: shape={W_U.shape}, effective_rank(95%)={k}, null_dim={W_U.shape[1]-k}")
    print(f"Rank: 90%={k_90}, 95%={k_95}, 99%={k_99}")
    print(f"SVD time: {time.time()-t0:.1f}s")

    # 运行3个实验
    print("\n" + "#"*60)
    print("# Running Experiments")
    print("#"*60)

    exp1_results = exp1_conditional_transfer(model, tokenizer, model_name, W_U, row_basis, sv_data)

    exp2_results = exp2_attention_transport(model, tokenizer, model_name, W_U, row_basis)

    exp3_results = exp3_routing_topology(model, tokenizer, model_name, W_U, row_basis)

    # 保存结果
    transfer_matrix, _, _, _ = exp1_results
    
    all_results = {
        "phase": 150,
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
            "top_10_sv": sv_data[:10].tolist(),
            "sv_tail_10": sv_data[-10:].tolist() if len(sv_data) >= 10 else sv_data.tolist(),
        },
        "exp1_conditional_transfer": transfer_matrix,
        "exp2_attention_transport": exp2_results,
        "exp3_routing_topology": exp3_results,
    }

    result_file = OUTPUT_DIR / f"phase150_{model_name}_{timestamp}.json"

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
