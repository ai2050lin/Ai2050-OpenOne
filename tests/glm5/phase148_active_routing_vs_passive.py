"""Phase 148: 主动路由 vs 被动压缩 + 逐层泄漏追踪
================================================

核心问题 (Phase 147.5 理论审查):
  世界A: 系统主动路由扰动进null space (主动约束传播)
  世界B: 语义空间本身就低维, 扰动天然落入null space (被动压缩)

三个实验:
  Exp 1: 扰动方向依赖性 — 最关键判据
    - 注入null-space方向 vs row-space方向 vs 随机方向
    - 如果null方向传播后null_ratio更高 → 被动压缩
    - 如果所有方向传播后null_ratio相似 → 主动路由

  Exp 2: 逐层Row/Null泄漏追踪
    - 在各层注入null-space扰动
    - 逐层追踪扰动在row space和null space中的分布
    - 哪些层是"泄漏层"(null→row)?

  Exp 3: Head Ablation对最终输出的影响
    - 关闭特定attention head
    - 测量扰动后logits/top-k变化
    - 如果某些head被关闭后扰动对输出影响增大 → 该head参与"null space路由"

用法:
  python tests/glm5/phase148_active_routing_vs_passive.py qwen3
  python tests/glm5/phase148_active_routing_vs_passive.py glm4
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import json
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from model_utils import (load_model, get_layers, get_model_info,
                          release_model, get_W_U, MODEL_CONFIGS)

EPSILON = 2.0
N_SENTENCES = 10
N_ROLLOUT_TOKENS = 50

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
]

OUTPUT_DIR = Path("tests/glm5_temp")


def get_row_null_bases(W_U, n_components=200):
    """计算W_U的row space和null space基底
    
    使用随机化SVD (randomized SVD) 避免内存问题
    只需要W_U和W_U.T的矩阵-向量乘法, 不需要存储整个密集矩阵
    """
    d = W_U.shape[1]  # d_model
    k = min(n_components, d - 2)
    
    # 随机化SVD: 用随机投影获取近似row space
    # Step 1: 生成随机矩阵 Omega: [vocab, k]
    # Step 2: Y = W_U @ Omega: [d_model, k]
    # Step 3: QR分解Y = QR → Q是W_U row space的近似正交基
    # Step 4: B = Q^T @ W_U: [k, vocab]
    # Step 5: SVD(B) = U_B S Vt_B → 近似奇异值
    
    np.random.seed(42)
    n_samples = k + 10  # 多采样一些提高精度
    
    # 分批计算 Y = W_U.T @ Omega, 避免OOM
    # W_U.T: [d_model, vocab], Omega: [vocab, n_samples]
    # Y = W_U.T @ Omega: [d_model, n_samples]
    
    print(f"  Computing randomized SVD (k={k}, n_samples={n_samples})...")
    
    # 生成Omega
    Omega = np.random.randn(W_U.shape[0], n_samples).astype(np.float32)  # [vocab, n_samples]
    
    # 分批计算 Y = W_U.T @ Omega
    # 每次处理vocab的一批行
    batch_size = 50000
    Y = np.zeros((d, n_samples), dtype=np.float32)  # [d_model, n_samples]
    for i in range(0, W_U.shape[0], batch_size):
        end = min(i + batch_size, W_U.shape[0])
        Y += W_U[i:end].T.astype(np.float32) @ Omega[i:end]
    
    del Omega
    
    # QR分解
    Q, R = np.linalg.qr(Y)
    del Y, R
    
    # Q: [d_model, n_samples] — W_U row space的近似正交基
    
    # 可选: 计算B = Q^T @ W_U 来获取奇异值
    # 但为了节省内存, 只计算前几个奇异值的估计
    # B = Q.T @ W_U: [n_samples, vocab] — 太大, 跳过
    
    # 用功率迭代(power iteration)提高精度
    # Z = W_U @ (W_U.T @ Q) — 但这需要太多内存, 跳过
    
    # effective_k: Q的列数中有效的
    effective_k = min(k, Q.shape[1])
    
    row_basis = Q[:, :effective_k].T  # [effective_k, d_model]
    # 不生成大矩阵row_proj和null_proj, 而是用函数做投影
    # row_proj = row_basis.T @ row_basis  # [d, d] — 可能太大
    # null_proj = np.eye(d) - row_proj
    
    print(f"  Randomized SVD: effective_k={effective_k}, row_basis shape={row_basis.shape}")
    
    return row_basis, effective_k


def project_to_null(vec, row_basis):
    """将向量投影到W_U的null space: vec_null = vec - row_basis.T @ (row_basis @ vec)"""
    row_component = row_basis.T @ (row_basis @ vec)
    return vec - row_component


def project_to_row(vec, row_basis):
    """将向量投影到W_U的row space: vec_row = row_basis.T @ (row_basis @ vec)"""
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


def project_to_space(vec, basis):
    """将向量投影到basis张成的子空间"""
    # basis: [k, d], vec: [d]
    coeffs = basis @ vec  # [k]
    return coeffs @ basis  # [d]


def get_device_for_input(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda")


def run_forward_with_perturbation(model, input_ids, attention_mask,
                                   inject_layer, delta_np, device):
    """在指定层注入扰动, 返回所有层hidden states + logits"""
    layers = get_layers(model)
    captured = {}

    def make_inject_hook(delta_tensor):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0].clone()
                out[0, -1, :] += delta_tensor.to(out.device)
                return (out,) + output[1:]
            else:
                out = output.clone()
                out[0, -1, :] += delta_tensor.to(out.device)
                return out
        return hook

    hooks = [layers[inject_layer].register_forward_hook(make_inject_hook(
        torch.tensor(delta_np, dtype=torch.float32)))]

    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)
    finally:
        for h in hooks:
            h.remove()

    return out


def run_forward_with_ablation(model, input_ids, attention_mask,
                               inject_layer, delta_np, ablate_layer, ablate_head,
                               d_model, device):
    """在指定层注入扰动 + ablate指定head, 返回hidden states + logits
    
    Ablation策略: 在o_proj的input上置零对应head的128维
    (因为o_proj的input是concat的attn output [batch, seq, n_heads*head_dim])
    """
    layers = get_layers(model)
    layer = layers[ablate_layer]
    sa = layer.self_attn
    
    # 获取head维度
    head_dim = getattr(sa, 'head_dim', 128)
    n_heads = model.config.num_attention_heads if hasattr(model.config, 'num_attention_heads') else d_model // head_dim

    hooks = []

    def make_inject_hook(delta_tensor):
        def hook(module, input, output):
            if isinstance(output, tuple):
                out = output[0].clone()
                out[0, -1, :] += delta_tensor.to(out.device)
                return (out,) + output[1:]
            else:
                out = output.clone()
                out[0, -1, :] += delta_tensor.to(out.device)
                return out
        return hook

    def make_ablation_hook(head_idx, h_dim):
        """在o_proj的input上置零对应head的输出"""
        def hook(module, input, output):
            # input[0]是o_proj的输入: [batch, seq, n_heads*head_dim]
            # 我们需要修改这个输入, 将head_idx对应的h_dim维置零
            # 但hook只能修改output, 不能修改input
            # 替代方案: 计算该head被置零后的output差异, 从output中减去
            # o_proj: [d_model, n_heads*head_dim]
            # head_idx的输出对应的o_proj列: [head_idx*head_dim : (head_idx+1)*head_dim]
            # 置零这些列的贡献 = output - o_proj_weight[:, head_idx*h_dim:(head_idx+1)*h_dim] @ input[0][:, :, head_idx*h_dim:(head_idx+1)*h_dim]
            
            # 简化: 直接在output上减去对应贡献
            if isinstance(output, tuple):
                out = output[0].clone()
            else:
                out = output.clone()
            
            # 获取o_proj权重
            W_o = module.weight.detach().float()  # [d_model, n_heads*head_dim]
            
            # 获取input
            inp = input[0].detach().float()  # [batch, seq, n_heads*head_dim]
            
            # head_idx对应的input切片
            start = head_idx * h_dim
            end = (head_idx + 1) * h_dim
            head_input = inp[:, :, start:end]  # [batch, seq, head_dim]
            
            # head_idx对应的o_proj权重切片
            W_o_head = W_o[:, start:end]  # [d_model, head_dim]
            
            # 减去该head的贡献: out = out - W_o_head @ head_input.T
            # 对batch中的每个位置
            head_contribution = torch.matmul(head_input, W_o_head.T)  # [batch, seq, d_model]
            # 确保dtype一致
            out = out.to(head_contribution.dtype) - head_contribution.to(out.device)
            
            if isinstance(output, tuple):
                return (out,) + output[1:]
            return out
        return hook

    hooks.append(layers[inject_layer].register_forward_hook(
        make_inject_hook(torch.tensor(delta_np, dtype=torch.float32))))
    hooks.append(sa.o_proj.register_forward_hook(
        make_ablation_hook(ablate_head, head_dim)))

    try:
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)
    finally:
        for h in hooks:
            h.remove()

    return out


# ============================================================
# Exp 1: 扰动方向依赖性 — 最关键判据
# ============================================================
def exp1_direction_dependent(model, tokenizer, model_name, W_U, row_basis):
    print("\n" + "="*60)
    print("Exp 1: Direction-Dependent Propagation")
    print("核心: null方向 vs row方向 vs 随机方向 → 传播后null_ratio差异?")
    print("="*60)

    info = get_model_info(model, model_name)
    layers = get_layers(model)
    device = get_device_for_input(model)
    d_model = info.d_model
    n_layers = info.n_layers
    row_basis_t = row_basis  # numpy

    results = {}

    for sent_idx in range(N_SENTENCES):
        prompt = TEST_PROMPTS[sent_idx]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        for inject_l_frac in [0.0, 0.5]:
            inject_l = int(inject_l_frac * (n_layers - 1))

            # Clean forward
            with torch.no_grad():
                out_clean = model(input_ids=input_ids, attention_mask=attention_mask,
                                  output_hidden_states=True)
            clean_hs = out_clean.hidden_states

            # 生成三种方向的扰动
            np.random.seed(sent_idx * 100 + int(inject_l_frac * 100))
            raw_vec = np.random.randn(d_model)

            # 1. Null-space方向
            delta_null = project_to_null(raw_vec, row_basis)
            norm = np.linalg.norm(delta_null)
            if norm > 1e-8:
                delta_null = delta_null / norm * EPSILON
            else:
                continue

            # 2. Row-space方向
            delta_row = project_to_row(raw_vec, row_basis)
            norm = np.linalg.norm(delta_row)
            if norm > 1e-8:
                delta_row = delta_row / norm * EPSILON
            else:
                delta_row = None

            # 3. 随机方向
            delta_rand = raw_vec / np.linalg.norm(raw_vec) * EPSILON

            dir_results = {}
            for dir_name, delta in [("null", delta_null), ("row", delta_row), ("random", delta_rand)]:
                if delta is None:
                    continue

                # Perturbed forward
                out_perturbed = run_forward_with_perturbation(
                    model, input_ids, attention_mask, inject_l, delta, device)

                # 计算每层的null/row比例
                propagation = {}
                for li in range(n_layers + 1):
                    perturbed_vec = out_perturbed.hidden_states[li][0, -1, :].float().cpu().numpy()
                    clean_vec = clean_hs[li][0, -1, :].float().cpu().numpy()

                    delta_prop = perturbed_vec - clean_vec
                    delta_norm = np.linalg.norm(delta_prop)

                    if delta_norm < 1e-8:
                        propagation[li] = {'null_ratio': 1.0, 'row_energy': 0.0, 'delta_norm': 0.0}
                        continue

                    # row energy = ||row_basis @ delta||^2 / ||delta||^2
                    row_coeffs = row_basis_t @ delta_prop
                    row_energy, null_ratio = compute_row_energy(delta_prop, row_basis)

                    propagation[li] = {
                        'null_ratio': null_ratio,
                        'row_energy': row_energy,
                        'delta_norm': float(delta_norm),
                    }

                # Logits比较
                logits_clean = out_clean.logits[0, -1].float().cpu().numpy()
                logits_perturbed = out_perturbed.logits[0, -1].float().cpu().numpy()
                logits_corr = float(np.corrcoef(logits_clean, logits_perturbed)[0, 1])

                top1_clean = int(np.argmax(logits_clean))
                top1_perturbed = int(np.argmax(logits_perturbed))
                top1_match = int(top1_clean == top1_perturbed)

                dir_results[dir_name] = {
                    'propagation': propagation,
                    'logits_corr': logits_corr,
                    'top1_match': top1_match,
                }

            key = f"sent{sent_idx}_L{inject_l}"
            results[key] = dir_results

            # 打印关键结果
            final_l = n_layers
            print(f"  Sent {sent_idx}, L{inject_l}:")
            for dir_name, dr in dir_results.items():
                if final_l in dr['propagation']:
                    p = dr['propagation'][final_l]
                    print(f"    {dir_name:6s}: final null_ratio={p['null_ratio']:.4f}, "
                          f"row_energy={p['row_energy']:.4f}, "
                          f"logits_corr={dr['logits_corr']:.4f}, top1={dr['top1_match']}")

    # === 汇总分析 ===
    print("\n  === Exp 1 Summary ===")
    null_null_ratios = []
    row_null_ratios = []
    rand_null_ratios = []
    null_logits_corrs = []
    row_logits_corrs = []
    rand_logits_corrs = []

    for key, dir_results in results.items():
        final_l = n_layers
        for dir_name, dr in dir_results.items():
            if final_l in dr['propagation']:
                nr = dr['propagation'][final_l]['null_ratio']
                lc = dr['logits_corr']
                if dir_name == "null":
                    null_null_ratios.append(nr)
                    null_logits_corrs.append(lc)
                elif dir_name == "row":
                    row_null_ratios.append(nr)
                    row_logits_corrs.append(lc)
                elif dir_name == "random":
                    rand_null_ratios.append(nr)
                    rand_logits_corrs.append(lc)

    print(f"\n  Final Layer Null Ratio (扰动在最终层落在null space的比例):")
    if null_null_ratios:
        print(f"    Null-space input:  {np.mean(null_null_ratios):.4f} ± {np.std(null_null_ratios):.4f}")
    if row_null_ratios:
        print(f"    Row-space input:   {np.mean(row_null_ratios):.4f} ± {np.std(row_null_ratios):.4f}")
    if rand_null_ratios:
        print(f"    Random input:      {np.mean(rand_null_ratios):.4f} ± {np.std(rand_null_ratios):.4f}")

    print(f"\n  Logits Correlation (扰动对输出的影响):")
    if null_logits_corrs:
        print(f"    Null-space input:  {np.mean(null_logits_corrs):.4f} ± {np.std(null_logits_corrs):.4f}")
    if row_logits_corrs:
        print(f"    Row-space input:   {np.mean(row_logits_corrs):.4f} ± {np.std(row_logits_corrs):.4f}")
    if rand_logits_corrs:
        print(f"    Random input:      {np.mean(rand_logits_corrs):.4f} ± {np.std(rand_logits_corrs):.4f}")

    # 核心判据
    if null_null_ratios and row_null_ratios:
        diff = np.mean(null_null_ratios) - np.mean(row_null_ratios)
        logits_diff = np.mean(null_logits_corrs) - np.mean(row_logits_corrs)
        print(f"\n  *** 核心判据 ***")
        print(f"  Null Ratio差异 (null输入 - row输入): {diff:.4f}")
        print(f"  Logits Corr差异 (null输入 - row输入): {logits_diff:.4f}")

        if diff > 0.1:
            print("  → 强方向依赖 → 被动压缩 (Passive Compression)")
            print("  → null方向传播后更多留在null space, row方向更多泄漏到row space")
            print("  → 系统不主动路由, 方向本身决定了传播行为")
        elif abs(diff) < 0.05:
            print("  → 弱方向依赖 → 主动路由 (Active Routing)")
            print("  → 无论输入方向, 系统都把扰动路由到null space")
            print("  → 存在某种主动机制在重新分配扰动方向")
        else:
            print("  → 中等方向依赖 → 混合机制")

        if logits_diff > 0.01:
            print(f"  → Null-space扰动对logits影响更小 (correlation更高) → 确认输出等价类")

    return results


# ============================================================
# Exp 2: 逐层Row/Null泄漏追踪
# ============================================================
def exp2_leakage_tracking(model, tokenizer, model_name, W_U, row_basis):
    print("\n" + "="*60)
    print("Exp 2: Layer-by-Layer Row/Null Leakage Tracking")
    print("核心: null-space扰动在逐层传播中, 多少泄漏到row space?")
    print("="*60)

    info = get_model_info(model, model_name)
    layers = get_layers(model)
    device = get_device_for_input(model)
    d_model = info.d_model
    n_layers = info.n_layers
    row_basis_t = row_basis

    results = {}

    for sent_idx in range(min(6, N_SENTENCES)):
        prompt = TEST_PROMPTS[sent_idx]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Clean forward
        with torch.no_grad():
            out_clean = model(input_ids=input_ids, attention_mask=attention_mask,
                              output_hidden_states=True)
        clean_hs = out_clean.hidden_states

        for inject_l_frac in [0.0, 0.25, 0.5, 0.75]:
            inject_l = int(inject_l_frac * (n_layers - 1))

            # 生成null-space扰动
            np.random.seed(sent_idx * 200 + int(inject_l_frac * 200))
            raw_vec = np.random.randn(d_model)
            delta_null = project_to_null(raw_vec, row_basis)
            norm = np.linalg.norm(delta_null)
            if norm > 1e-8:
                delta_null = delta_null / norm * EPSILON
            else:
                continue

            # Perturbed forward
            out_perturbed = run_forward_with_perturbation(
                model, input_ids, attention_mask, inject_l, delta_null, device)

            # 逐层计算null/row泄漏
            layer_data = []
            for li in range(n_layers + 1):
                perturbed_vec = out_perturbed.hidden_states[li][0, -1, :].float().cpu().numpy()
                clean_vec = clean_hs[li][0, -1, :].float().cpu().numpy()

                delta_prop = perturbed_vec - clean_vec
                delta_norm = np.linalg.norm(delta_prop)

                if delta_norm < 1e-8:
                    layer_data.append({
                        'layer': li, 'null_ratio': 1.0, 'row_energy': 0.0,
                        'delta_norm': 0.0, 'row_abs': 0.0, 'null_abs': 0.0,
                    })
                    continue

                row_coeffs = row_basis @ delta_prop
                row_energy = np.sum(row_coeffs**2) / (delta_norm**2)
                null_ratio = 1.0 - row_energy
                row_abs = np.sqrt(np.sum(row_coeffs**2))
                null_abs = np.sqrt(max(0, delta_norm**2 - np.sum(row_coeffs**2)))

                layer_data.append({
                    'layer': li, 'null_ratio': float(null_ratio),
                    'row_energy': float(row_energy),
                    'delta_norm': float(delta_norm),
                    'row_abs': float(row_abs),
                    'null_abs': float(null_abs),
                })

            key = f"sent{sent_idx}_L{inject_l}"
            results[key] = layer_data

            # 打印关键层
            print(f"  Sent {sent_idx}, inject L{inject_l}:")
            for ld in layer_data:
                li = ld['layer']
                if li == 0 or li == inject_l + 1 or li == n_layers or \
                   li % max(1, n_layers // 5) == 0:
                    print(f"    L{li:2d}: null_ratio={ld['null_ratio']:.4f}, "
                          f"row_abs={ld['row_abs']:.3f}, null_abs={ld['null_abs']:.3f}, "
                          f"delta_norm={ld['delta_norm']:.3f}")

    # === 汇总: 按注入层分组, 平均null_ratio和row_abs随层演化 ===
    print("\n  === Exp 2 Summary: Leakage per injection layer ===")
    for inj_frac in [0.0, 0.25, 0.5, 0.75]:
        inj_l = int(inj_frac * (n_layers - 1))
        matching = {k: v for k, v in results.items() if f"_L{inj_l}" in k}
        if not matching:
            continue

        # 对每个层位取平均
        layer_avgs = {}
        for k, layer_data in matching.items():
            for ld in layer_data:
                li = ld['layer']
                if li not in layer_avgs:
                    layer_avgs[li] = {'null_ratios': [], 'row_abs': [], 'null_abs': []}
                layer_avgs[li]['null_ratios'].append(ld['null_ratio'])
                layer_avgs[li]['row_abs'].append(ld['row_abs'])
                layer_avgs[li]['null_abs'].append(ld['null_abs'])

        print(f"\n  Inject L{inj_l}:")
        for li in sorted(layer_avgs.keys()):
            if li == 0 or li == inj_l + 1 or li == n_layers or \
               li % max(1, n_layers // 5) == 0:
                avg_nr = np.mean(layer_avgs[li]['null_ratios'])
                avg_ra = np.mean(layer_avgs[li]['row_abs'])
                avg_na = np.mean(layer_avgs[li]['null_abs'])
                print(f"    L{li:2d}: avg null_ratio={avg_nr:.4f}, "
                      f"avg row_abs={avg_ra:.3f}, avg null_abs={avg_na:.3f}")

        # 找最大泄漏层
        max_leakage_layer = max(layer_avgs.keys(),
                                key=lambda l: np.mean(layer_avgs[l]['row_abs']))
        max_leak = np.mean(layer_avgs[max_leakage_layer]['row_abs'])
        print(f"    → Max row_abs at L{max_leakage_layer}: {max_leak:.3f}")

    return results


# ============================================================
# Exp 3: Head Ablation对扰动的输出影响
# ============================================================
def exp3_head_ablation(model, tokenizer, model_name, W_U, row_basis):
    print("\n" + "="*60)
    print("Exp 3: Head Ablation → Perturbation Impact on Output")
    print("核心: 关闭特定head后, null-space扰动对logits的影响是否增大?")
    print("="*60)

    info = get_model_info(model, model_name)
    layers = get_layers(model)
    device = get_device_for_input(model)
    d_model = info.d_model
    n_layers = info.n_layers

    # 获取head维度
    n_heads = getattr(layers[0].self_attn, 'num_heads', None)
    if n_heads is None:
        if hasattr(model.config, 'num_attention_heads'):
            n_heads = model.config.num_attention_heads
        else:
            n_heads = d_model // 64
    head_dim = d_model // n_heads
    print(f"  n_heads={n_heads}, head_dim={head_dim}")

    # 选择注入层和测试层
    inject_l = 0  # 早层注入, 看哪些head参与null space路由
    test_ablation_layers = [1, 2, 3, n_layers // 4, n_layers // 2]

    prompt = TEST_PROMPTS[0]
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # Clean forward
    with torch.no_grad():
        out_clean = model(input_ids=input_ids, attention_mask=attention_mask,
                          output_hidden_states=True)
    logits_clean = out_clean.logits[0, -1].float().cpu().numpy()
    top5_clean = set(np.argsort(logits_clean)[-5:])

    # 生成null-space扰动
    np.random.seed(42)
    raw_vec = np.random.randn(d_model)
    delta_null = project_to_null(raw_vec, row_basis)
    norm = np.linalg.norm(delta_null)
    delta_null = delta_null / norm * EPSILON

    # Baseline: 无ablation的扰动影响
    out_perturbed_base = run_forward_with_perturbation(
        model, input_ids, attention_mask, inject_l, delta_null, device)
    logits_base = out_perturbed_base.logits[0, -1].float().cpu().numpy()
    base_corr = float(np.corrcoef(logits_clean, logits_base)[0, 1])
    top5_base = set(np.argsort(logits_base)[-5:])
    base_top5_overlap = len(top5_clean & top5_base) / 5.0

    print(f"\n  Baseline (no ablation): logits_corr={base_corr:.6f}, top5_overlap={base_top5_overlap:.2f}")

    # Ablation测试
    results = {}
    n_heads_to_test = min(5, n_heads)  # 每层测试前5个head

    for abl_l in test_ablation_layers:
        if abl_l >= n_layers:
            continue
        for abl_h in range(n_heads_to_test):
            abl_key = f"L{abl_l}_H{abl_h}"

            try:
                out_abl = run_forward_with_ablation(
                    model, input_ids, attention_mask,
                    inject_l, delta_null, abl_l, abl_h, d_model, device)

                logits_abl = out_abl.logits[0, -1].float().cpu().numpy()
                corr = float(np.corrcoef(logits_clean, logits_abl)[0, 1])
                top5_abl = set(np.argsort(logits_abl)[-5:])
                top5_overlap = len(top5_clean & top5_abl) / 5.0

                # 关键指标: 相对于baseline的logits correlation变化
                corr_change = corr - base_corr
                overlap_change = top5_overlap - base_top5_overlap

                results[abl_key] = {
                    'logits_corr': corr,
                    'corr_change': corr_change,
                    'top5_overlap': top5_overlap,
                    'overlap_change': overlap_change,
                }

                if abs(corr_change) > 0.001 or abs(overlap_change) > 0.1:
                    print(f"    Ablate {abl_key}: corr={corr:.6f} (Δ={corr_change:+.6f}), "
                          f"top5_overlap={top5_overlap:.2f} (Δ={overlap_change:+.2f}) **")
                else:
                    print(f"    Ablate {abl_key}: corr={corr:.6f} (Δ={corr_change:+.6f}), "
                          f"top5_overlap={top5_overlap:.2f}")

            except Exception as e:
                print(f"    Ablate {abl_key}: ERROR - {e}")
                results[abl_key] = {'error': str(e)}

    # 分析
    print("\n  === Exp 3 Summary ===")
    significant = {k: v for k, v in results.items()
                   if 'error' not in v and (abs(v['corr_change']) > 0.001 or abs(v['overlap_change']) > 0.1)}

    if significant:
        print(f"  Significant ablations: {len(significant)}/{len(results)}")
        for k, v in sorted(significant.items(), key=lambda x: abs(x[1].get('corr_change', 0)), reverse=True):
            print(f"    {k}: corr_change={v['corr_change']:+.6f}, overlap_change={v['overlap_change']:+.2f}")

        # 判断: corr下降 = 该head帮助抑制扰动对输出的影响 = 参与"null space路由"
        corr_decreasers = [k for k, v in significant.items() if v['corr_change'] < -0.001]
        if corr_decreasers:
            print(f"\n  → {len(corr_decreasers)} heads whose ablation DECREASES logits_corr")
            print(f"  → These heads help suppress perturbation impact = 'null space routers'")
        else:
            print(f"\n  → No heads whose ablation significantly decreases logits_corr")
            print(f"  → Null space routing may be distributed across many heads or passive")
    else:
        print(f"  No significant ablations found")
        print(f"  → Null space routing is likely PASSIVE (geometric byproduct)")

    return results


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    print(f"Phase 148: {model_name}")
    print(f"Time: {timestamp}")

    # 加载模型
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    use_8bit = model_name in ("deepseek7b", "glm4") and gpu_mem_gb < 16
    print(f"Mode: {'8bit' if use_8bit else 'bfloat16'}")

    model, tokenizer, device = load_model(model_name, use_8bit=use_8bit)
    info = get_model_info(model, model_name)
    print(f"Model: {info.model_class}, {info.n_layers}L, d={info.d_model}")

    # 获取W_U和row/null space
    W_U = get_W_U(model, model_name)
    row_basis, k = get_row_null_bases(W_U)
    print(f"W_U: shape={W_U.shape}, row_components={k}, null_dim={W_U.shape[1]-k}")

    # 运行3个实验
    exp1_results = exp1_direction_dependent(model, tokenizer, model_name, W_U, row_basis)
    exp2_results = exp2_leakage_tracking(model, tokenizer, model_name, W_U, row_basis)
    exp3_results = exp3_head_ablation(model, tokenizer, model_name, W_U, row_basis)

    # 保存结果
    all_results = {
        "model": model_name,
        "timestamp": timestamp,
        "epsilon": EPSILON,
        "n_sentences": N_SENTENCES,
        "exp1_direction_dependent": {},
        "exp2_leakage_tracking": {},
        "exp3_head_ablation": {},
    }

    # Exp1 汇总
    for key, dir_results in exp1_results.items():
        all_results["exp1_direction_dependent"][key] = {}
        for dir_name, dr in dir_results.items():
            all_results["exp1_direction_dependent"][key][dir_name] = {
                'final_null_ratio': dr['propagation'].get(info.n_layers, {}).get('null_ratio', -1),
                'final_row_energy': dr['propagation'].get(info.n_layers, {}).get('row_energy', -1),
                'logits_corr': dr['logits_corr'],
                'top1_match': dr['top1_match'],
            }

    # Exp2 汇总
    for key, layer_data in exp2_results.items():
        all_results["exp2_leakage_tracking"][key] = {
            'first_layer': layer_data[0] if layer_data else None,
            'inject_plus1': layer_data[int(key.split('_L')[1]) + 1] if int(key.split('_L')[1]) + 1 < len(layer_data) else None,
            'final_layer': layer_data[-1] if layer_data else None,
            'max_row_abs_layer': max(layer_data, key=lambda x: x.get('row_abs', 0)) if layer_data else None,
        }

    # Exp3 汇总
    all_results["exp3_head_ablation"] = exp3_results

    result_file = OUTPUT_DIR / f"phase148_{model_name}_{timestamp}.json"

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
