"""
Phase 73: Layer Jacobian Dynamics — 从标量梯度到真正的层映射
============================================================

Phase 72的致命硬伤:
1. 我们研究的是 ∂logit_gap/∂h_l (标量输出梯度), 不是 ∂h_{l+1}/∂h_l (完整Jacobian)
2. 标量输出 f: R^768 → R 的梯度天然低秩 — 不能代表完整动力学
3. cos≈0 在768维空间是默认情况 — 不能推出"结构正交"
4. CKA=1 可疑 — 需要验证
5. effective rank 需要 Marchenko-Pastur 基线

Phase 73的核心改进:
1. ★★★ 计算真正的层Jacobian J_l = ∂h_{l+1}/∂h_l ∈ R^{768×768}
2. ★★★ 与Marchenko-Pastur随机矩阵基线比较
3. ★★★ 用CCA(不是cosine)测试正交性, 加随机基线
4. ★★★ 比较标量梯度 vs 完整Jacobian — 量化标量梯度丢失了多少信息
5. 保守解释 — 观察优先, 机制推断后置

关键问题:
- J_l的谱结构是否"特殊"? (vs 随机矩阵)
- 标量梯度g_l捕获了J_l的多少维度?
- g_spec的"旋转"是J_l的什么性质导致的?
- number和tense的正交性是否超过随机基线?
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch, numpy as np, gc, argparse, time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

GPT2_PATH = "D:/develop/model/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"

NVA_PAIRS = [
    ("cat","cats","runs","run"), ("dog","dogs","walks","walk"),
    ("bird","birds","flies","fly"), ("girl","girls","reads","read"),
    ("boy","boys","sings","sing"), ("horse","horses","jumps","jump"),
    ("bear","bears","sleeps","sleep"), ("snake","snakes","crawls","crawl"),
    ("fish","fishes","swims","swim"), ("frog","frogs","hops","hop"),
    ("star","stars","shines","shine"), ("tree","trees","grows","grow"),
    ("clock","clocks","ticks","tick"), ("bell","bells","rings","ring"),
    ("fox","foxes","hunts","hunt"),
]

TENSE_PAIRS = [
    ("walked", "walks"), ("played", "plays"), ("jumped", "jumps"),
    ("cooked", "cooks"), ("cleaned", "cleans"), ("worked", "works"),
    ("helped", "helps"), ("talked", "talks"), ("smiled", "smiles"),
    ("danced", "dances"), ("moved", "moves"), ("lived", "lives"),
]


def load_gpt2_float32():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print("[Phase73] Loading GPT-2 Small float32...")
    tokenizer = GPT2Tokenizer.from_pretrained(GPT2_PATH, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(
        GPT2_PATH, torch_dtype=torch.float32, local_files_only=True,
    )
    if torch.cuda.is_available():
        model = model.to('cuda')
    model.eval()
    device = next(model.parameters()).device
    n_layers = model.config.n_layer
    d_model = model.config.n_embd
    print(f"[Phase73] GPT-2: {n_layers} layers, d={d_model}, device={device}")
    return model, tokenizer, device


# ============================================================
# Core: Compute Full Layer Jacobian J_l = ∂h_{l+1}/∂h_l
# ============================================================
def compute_full_layer_jacobian(model, tokenizer, device, sentence, layer_idx,
                                 d_model=768, disable_param_grad=True):
    """
    计算完整的层Jacobian J_l = ∂h_{l+1}[last_pos] / ∂h_l[last_pos]
    
    返回: J ∈ R^{768×768}
    
    方法: 对h_{l+1}的每个维度做backward, 收集梯度
    - 768次backward pass (retain_graph)
    - 约4秒/层
    
    注意: 这是"同位置"Jacobian — 忽略attention的跨位置效应
    跨位置Jacobian需要 ∂h_{l+1}[pos] / ∂h_l[all_pos] ∈ R^{768 × (seq×768)}
    """
    toks = tokenizer(sentence, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    layers = model.transformer.h
    n_layers = len(layers)
    
    # 可选: 禁用参数梯度以节省内存
    if disable_param_grad:
        for p in model.parameters():
            p.requires_grad_(False)
    
    # Step 1: Forward pass, capture h_l
    h_l_captured = [None]
    def capture_hook(module, input, output):
        h_l_captured[0] = output[0].detach().clone() if isinstance(output, tuple) else output.detach().clone()
        return output
    hook1 = layers[layer_idx].register_forward_hook(capture_hook)
    with torch.no_grad():
        _ = model(input_ids=input_ids)
    hook1.remove()
    
    h_l = h_l_captured[0]  # [1, seq_len, d_model]
    
    # Step 2: Forward pass with h_l_grad, capture h_{l+1}
    h_l_grad = h_l.detach().clone().requires_grad_(True)
    
    def replace_hook(module, input, output):
        return (h_l_grad,) + output[1:] if isinstance(output, tuple) else h_l_grad
    hook2 = layers[layer_idx].register_forward_hook(replace_hook)
    
    # 下一个模块: 如果是最后一层, 则是final LayerNorm
    if layer_idx + 1 < n_layers:
        next_module = layers[layer_idx + 1]
    else:
        next_module = model.transformer.ln_f
    
    h_lp1_ref = [None]
    def capture_next(module, input, output):
        h_lp1_ref[0] = output[0] if isinstance(output, tuple) else output
        return output
    hook3 = next_module.register_forward_hook(capture_next)
    
    with torch.enable_grad():
        _ = model(input_ids=input_ids)
    
    hook2.remove()
    hook3.remove()
    
    if h_lp1_ref[0] is None:
        if disable_param_grad:
            for p in model.parameters():
                p.requires_grad_(True)
        return None
    
    h_lp1 = h_lp1_ref[0]
    # Handle different tensor shapes
    if h_lp1.dim() == 3:
        h_lp1_last = h_lp1[0, -1, :]  # [d_model]
    elif h_lp1.dim() == 2:
        h_lp1_last = h_lp1[-1, :]  # [d_model]
    else:
        h_lp1_last = h_lp1  # [d_model]
    
    # Step 3: 计算Jacobian — 逐行 (768次backward)
    J = torch.zeros(d_model, d_model, device='cpu')
    
    t0 = time.time()
    for i in range(d_model):
        if h_l_grad.grad is not None:
            h_l_grad.grad.zero_()
        h_lp1_last[i].backward(retain_graph=(i < d_model - 1))
        J[i, :] = h_l_grad.grad[0, -1, :].detach().cpu()
        if (i + 1) % 200 == 0:
            print(f"    Jacobian row {i+1}/{d_model} ({time.time()-t0:.1f}s)")
    
    # 清理
    del h_lp1_last, h_l_grad
    model.zero_grad()
    torch.cuda.empty_cache()
    
    # 恢复参数梯度
    if disable_param_grad:
        for p in model.parameters():
            p.requires_grad_(True)
    
    return J.numpy()


def compute_scalar_gradient(model, tokenizer, device, sentence, layer_idx,
                            sv_id, pv_id):
    """计算标量输出梯度 ∂logit_gap/∂h_l"""
    toks = tokenizer(sentence, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    layers = model.transformer.h
    
    h_l_captured = [None]
    def capture_hook(module, input, output):
        h_l_captured[0] = output[0].detach().clone() if isinstance(output, tuple) else output.detach().clone()
        return output
    hook = layers[layer_idx].register_forward_hook(capture_hook)
    with torch.no_grad():
        _ = model(input_ids=input_ids)
    hook.remove()
    
    if h_l_captured[0] is None:
        return None
    
    h_l = h_l_captured[0]
    h_l_grad = h_l.detach().clone().requires_grad_(True)
    
    def replace_hook(module, input, output):
        return (h_l_grad,) + output[1:] if isinstance(output, tuple) else h_l_grad
    hook2 = layers[layer_idx].register_forward_hook(replace_hook)
    with torch.enable_grad():
        out = model(input_ids=input_ids)
        logits = out.logits[0, -1, :].float()
        logit_gap = logits[pv_id] - logits[sv_id]
        logit_gap.backward()
    hook2.remove()
    
    gradient = h_l_grad.grad[0, -1, :].detach().cpu().numpy()
    return gradient


def marchenko_pastur_bounds(m, n, sigma=1.0):
    """
    Marchenko-Pastur分布的支撑区间
    对于 X ∈ R^{m×n}, i.i.d. N(0, σ²/n)
    奇异值分布的支撑: [σ_-, σ_+]
    """
    ratio = m / n
    sigma_plus = sigma * (1 + np.sqrt(ratio))
    sigma_minus = sigma * abs(1 - np.sqrt(ratio))
    return sigma_minus, sigma_plus


def participation_ratio(singular_values):
    """参与比 — 衡量有多少维度显著参与"""
    s2 = singular_values ** 2
    total = np.sum(s2)
    if total < 1e-30:
        return 0.0
    return (total ** 2) / np.sum(s2 ** 2)


def stable_rank(matrix):
    """稳定秩 = ||M||_F² / ||M||_2²"""
    frob = np.linalg.norm(matrix, 'fro')
    spec = np.linalg.norm(matrix, 2)  # = largest singular value
    if spec < 1e-30:
        return 0.0
    return (frob / spec) ** 2


def canonical_correlation(X, Y, k=None):
    """
    典型相关分析 (CCA)
    X: [n, d1], Y: [n, d2]
    返回: top-k canonical correlations
    
    CCA测量两组变量之间的最大相关性
    比cosine更可靠地衡量子空间关系
    """
    from numpy.linalg import svd as np_svd
    
    n = X.shape[0]
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    
    # 正则化
    reg = 1e-6
    Cxx = X.T @ X / n + reg * np.eye(X.shape[1])
    Cyy = Y.T @ Y / n + reg * np.eye(Y.shape[1])
    Cxy = X.T @ Y / n
    
    # 白化
    evals_x, evecs_x = np.linalg.eigh(Cxx)
    evals_y, evecs_y = np.linalg.eigh(Cyy)
    
    # 过滤小特征值
    mask_x = evals_x > 1e-10
    mask_y = evals_y > 1e-10
    
    Wx = evecs_x[:, mask_x] @ np.diag(1.0 / np.sqrt(evals_x[mask_x]))
    Wy = evecs_y[:, mask_y] @ np.diag(1.0 / np.sqrt(evals_y[mask_y]))
    
    # CCA矩阵
    M = Wx.T @ Cxy @ Wy
    
    # SVD给出canonical correlations
    s = np.linalg.svd(M, compute_uv=False)
    
    if k is not None:
        s = s[:k]
    
    return s  # canonical correlations (0到1之间)


# ============================================================
# Experiment A: Full Layer Jacobian SVD + Marchenko-Pastur Baseline
# ============================================================
def experiment_a(model, tokenizer, device, n_test=3):
    """
    ★★★ 核心实验: 计算真正的层Jacobian J_l
    
    对比:
    1. J_l的奇异值谱 vs Marchenko-Pastur随机基线
    2. J_l的有效秩/稳定秩/参与比
    3. J_l的谱是否"特殊"? (vs 同范数随机矩阵)
    
    关键问题: 
    - J_l的谱结构是"特殊"的还是随机矩阵行为?
    - 如果特殊, 在什么意义上特殊?
    """
    print("\n" + "="*70)
    print("Experiment A: Full Layer Jacobian SVD + Random Matrix Baseline")
    print("="*70)
    
    d_model = model.config.n_embd
    n_layers = model.config.n_layer
    
    # 选择测试层 (计算全部12层太慢, 选关键层)
    test_layers = [0, 3, 6, 9, 11]
    
    sentence = "The cat"  # 用一个固定输入
    
    results = {}
    
    for li in test_layers:
        print(f"\n--- Layer {li} → Layer {li+1 if li+1 < n_layers else 'LN_f'} ---")
        
        t0 = time.time()
        J = compute_full_layer_jacobian(model, tokenizer, device, sentence, li, d_model)
        dt = time.time() - t0
        
        if J is None:
            print(f"  Failed to compute Jacobian for layer {li}")
            continue
        
        print(f"  Jacobian computed in {dt:.1f}s, shape={J.shape}")
        
        # SVD
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        
        # 基本统计
        frob_norm = np.linalg.norm(J, 'fro')
        spec_norm = S[0]
        s_rank = stable_rank(J)
        p_ratio = participation_ratio(S)
        
        # 有效秩
        S_norm = S / S.sum()
        eff_rank = np.exp(-np.sum(S_norm * np.log(S_norm + 1e-30)))
        
        print(f"  ||J||_F = {frob_norm:.4f}, ||J||_2 = {spec_norm:.4f}")
        print(f"  stable_rank = {s_rank:.2f}, eff_rank = {eff_rank:.2f}, participation_ratio = {p_ratio:.2f}")
        print(f"  S[:10] = {S[:10].round(4)}")
        print(f"  S[-5:] = {S[-5:].round(6)}")
        print(f"  S/S[0][:10] = {(S/S[0])[:10].round(4)}")
        
        # ★★★ Marchenko-Pastur基线
        # 生成同范数的随机矩阵, 比较谱
        n_random = 10
        random_spectra = []
        for _ in range(n_random):
            R = np.random.randn(d_model, d_model) * (frob_norm / d_model)
            _, S_r, _ = np.linalg.svd(R, full_matrices=False)
            random_spectra.append(S_r)
        
        random_S_mean = np.mean(random_spectra, axis=0)
        random_S_std = np.std(random_spectra, axis=0)
        
        # MP理论边界
        mp_min, mp_max = marchenko_pastur_bounds(d_model, d_model, sigma=frob_norm/d_model)
        print(f"  Marchenko-Pastur bounds: [{mp_min:.4f}, {mp_max:.4f}]")
        
        # 比较实际谱 vs 随机谱
        # 关键: 实际谱的前几个奇异值是否显著大于随机基线?
        print(f"  Actual S[0:5] vs Random S[0:5] (mean±std):")
        for k in range(5):
            z_score = (S[k] - random_S_mean[k]) / max(random_S_std[k], 1e-10)
            print(f"    S[{k}]: actual={S[k]:.4f}, random={random_S_mean[k]:.4f}±{random_S_std[k]:.4f}, "
                  f"z={z_score:.1f}")
        
        # 谱衰减比: S[k]/S[k+1] — 是否有明显的gap?
        gaps = S[:-1] / np.maximum(S[1:], 1e-30)
        top_gap_idx = np.argmax(gaps[:50])  # 前50个中的最大gap
        print(f"  Largest spectral gap: S[{top_gap_idx}]/S[{top_gap_idx+1}] = {gaps[top_gap_idx]:.2f}")
        print(f"  (Random baseline largest gap: {np.max(np.mean([rs[:-1]/np.maximum(rs[1:],1e-30) for rs in random_spectra], axis=0)[:50]):.2f})")
        
        # ★★★ 条件数
        cond = S[0] / max(S[-1], 1e-30)
        print(f"  Condition number: {cond:.1f}")
        
        # 累积能量: 前k个奇异值占总能量的百分比
        cum_energy = np.cumsum(S**2) / np.sum(S**2)
        for k in [1, 5, 10, 20, 50]:
            print(f"  Cumulative energy top-{k}: {cum_energy[k-1]*100:.1f}%")
        
        results[li] = {
            'J': J, 'S': S, 'U': U, 'Vt': Vt,
            'frob_norm': frob_norm, 'spec_norm': spec_norm,
            'stable_rank': s_rank, 'eff_rank': eff_rank,
            'participation_ratio': p_ratio,
            'random_S_mean': random_S_mean,
            'cum_energy': cum_energy,
        }
        
        # 释放内存
        del J
        gc.collect()
    
    # ★★★ 跨层比较
    print("\n--- Cross-Layer Comparison ---")
    print(f"  {'Layer':>5} {'||J||_F':>8} {'||J||_2':>8} {'s_rank':>7} {'e_rank':>7} {'PR':>6} {'cond':>8} {'E@5':>6} {'E@20':>6}")
    for li in test_layers:
        if li in results:
            r = results[li]
            print(f"  L{li:4d} {r['frob_norm']:8.2f} {r['spec_norm']:8.4f} "
                  f"{r['stable_rank']:7.2f} {r['eff_rank']:7.2f} {r['participation_ratio']:6.2f} "
                  f"{r['S'][0]/max(r['S'][-1],1e-30):8.1f} "
                  f"{r['cum_energy'][4]*100:5.1f}% {r['cum_energy'][19]*100:5.1f}%")
    
    return results


# ============================================================
# Experiment B: Scalar Gradient vs Full Jacobian — 量化信息丢失
# ============================================================
def experiment_b(model, tokenizer, device, n_test=8):
    """
    ★★★ 关键比较: 标量输出梯度g_l vs 完整Jacobian J_l
    
    g_l = ∂logit_gap/∂h_l ∈ R^{768} — Phase 70-72研究的目标
    J_l = ∂h_{l+1}/∂h_l ∈ R^{768×768} — 真正的层映射
    
    关系: g_l = J_l^T · g_{l+1} (链式法则)
    
    问题:
    1. g_l在J_l的奇异向量上的能量分布如何?
    2. g_l捕获了J_l的多少个有效维度?
    3. Phase 70-72的结论有多少是标量梯度的artifact?
    """
    print("\n" + "="*70)
    print("Experiment B: Scalar Gradient vs Full Jacobian")
    print("="*70)
    
    d_model = model.config.n_embd
    sv_id = tokenizer.encode(" runs", add_special_tokens=False)[0]
    pv_id = tokenizer.encode(" run", add_special_tokens=False)[0]
    
    test_layers = [0, 3, 6, 9, 11]
    
    for li in test_layers:
        print(f"\n--- Layer {li} ---")
        
        # 计算Jacobian
        sentence = "The cat"
        J = compute_full_layer_jacobian(model, tokenizer, device, sentence, li, d_model)
        if J is None:
            continue
        
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        V = Vt.T  # 右奇异向量 — J的输入空间基
        
        # 计算标量梯度
        g_l = compute_scalar_gradient(model, tokenizer, device, sentence, li, sv_id, pv_id)
        if g_l is None:
            del J; continue
        
        g_norm = np.linalg.norm(g_l)
        
        # ★★★ g_l在J的右奇异向量上的投影
        # g_l = Σ_k (g_l · V[:,k]) V[:,k]
        projections = Vt @ g_l  # [d_model] — g_l在每个右奇异向量上的分量
        proj_energy = projections ** 2
        total_energy = np.sum(proj_energy)
        
        if total_energy < 1e-30:
            print(f"  g_l energy = 0, skipping")
            del J; continue
        
        # 累积能量
        cum_energy = np.cumsum(proj_energy) / total_energy
        
        print(f"  ||g_l|| = {g_norm:.4f}")
        print(f"  g_l energy distribution on J's right singular vectors:")
        for k in [1, 5, 10, 20, 50, 100]:
            print(f"    Top-{k} SVs capture: {cum_energy[k-1]*100:.1f}% of g_l energy")
        
        # ★★★ g_l主要在哪些奇异向量上?
        top_proj_idx = np.argsort(proj_energy)[::-1][:10]
        print(f"  Top-10 SV indices by g_l energy: {top_proj_idx}")
        print(f"  Their singular values: {S[top_proj_idx].round(4)}")
        print(f"  Their energy fractions: {(proj_energy[top_proj_idx]/total_energy*100).round(1)}%")
        
        # ★★★ 对比: 随机方向在J的奇异向量上的能量分布
        # 如果g_l比随机方向更集中 → 标量梯度确实捕获了J的主要动力学
        # 如果g_l和随机方向一样分散 → 标量梯度丢失了大量信息
        n_random = 100
        random_cum = []
        for _ in range(n_random):
            r = np.random.randn(d_model)
            r = r / np.linalg.norm(r) * g_norm
            r_proj = Vt @ r
            r_energy = r_proj ** 2
            r_total = np.sum(r_energy)
            if r_total > 1e-30:
                r_cum = np.cumsum(r_energy) / r_total
                random_cum.append(r_cum)
        
        random_cum_mean = np.mean(random_cum, axis=0)
        
        print(f"  g_l cum energy vs random baseline:")
        for k in [1, 5, 10, 20, 50]:
            print(f"    Top-{k}: g_l={cum_energy[k-1]*100:.1f}%, random={random_cum_mean[k-1]*100:.1f}%")
        
        # ★★★ 关键指标: g_l比随机方向集中多少?
        # 用"达到90%能量需要的奇异向量数"来衡量
        g_90 = np.searchsorted(cum_energy, 0.9) + 1
        r_90 = np.searchsorted(random_cum_mean, 0.9) + 1
        concentration_ratio = r_90 / max(g_90, 1)
        print(f"  SVs needed for 90% energy: g_l={g_90}, random={r_90}, ratio={concentration_ratio:.2f}x")
        
        del J
    
    # ★★★ 多输入验证: g_l的能量分布是否输入依赖?
    print("\n--- Multi-Input Verification ---")
    li = 6  # 中间层
    J = compute_full_layer_jacobian(model, tokenizer, device, "The cat", li, d_model)
    if J is None:
        return
    
    _, S_J, Vt_J = np.linalg.svd(J, full_matrices=False)
    
    for i, (sn, pn, sv, pv) in enumerate(NVA_PAIRS[:5]):
        for cond, noun in [('sing', sn), ('plur', pn)]:
            sentence = f"The {noun}"
            g = compute_scalar_gradient(model, tokenizer, device, sentence, li, sv_id, pv_id)
            if g is None:
                continue
            
            proj = Vt_J @ g
            energy = proj ** 2
            total = np.sum(energy)
            if total < 1e-30:
                continue
            cum = np.cumsum(energy) / total
            n90 = np.searchsorted(cum, 0.9) + 1
            
            # 主要奇异向量
            top3 = np.argsort(energy)[::-1][:3]
            
            print(f"  {sentence:15s} ({cond}): 90%@{n90:3d} SVs, "
                  f"top3 SVs=[{top3[0]},{top3[1]},{top3[2]}], "
                  f"S=[{S_J[top3[0]]:.2f},{S_J[top3[1]]:.2f},{S_J[top3[2]]:.2f}]")
    
    del J


# ============================================================
# Experiment C: CCA-Based Orthogonality with Random Baseline
# ============================================================
def experiment_c(model, tokenizer, device, n_test=15):
    """
    ★★★ 用CCA(不是cosine)测试正交性, 加随机基线
    
    Phase 72的问题: cos(g_spec_number, g_spec_tense) ≈ 0.05
    在768维空间, 两个随机方向的期望cos ≈ 0, std ≈ 1/√768 ≈ 0.036
    所以cos=0.05只比随机高~1.4σ — 不显著!
    
    改进:
    1. 用CCA测量子空间相关性 (比cosine更可靠)
    2. 生成随机基线: 将同类梯度随机分成两组, 计算CCA
    3. 比较: CCA(number, tense) vs CCA(random_split1, random_split2)
    4. 如果CCA(num,ten) << CCA(rand1,rand2) → 真正的结构正交性
    5. 如果CCA(num,ten) ≈ CCA(rand1,rand2) → 只是高维随机效应
    """
    print("\n" + "="*70)
    print("Experiment C: CCA-Based Orthogonality with Random Baseline")
    print("="*70)
    
    sv_id = tokenizer.encode(" runs", add_special_tokens=False)[0]
    pv_id = tokenizer.encode(" run", add_special_tokens=False)[0]
    past_id = tokenizer.encode(" walked", add_special_tokens=False)[0]
    pres_id = tokenizer.encode(" walks", add_special_tokens=False)[0]
    
    d_model = model.config.n_embd
    n_layers = model.config.n_layer
    
    test_layers = [0, 3, 6, 9, 11]
    
    for li in test_layers:
        print(f"\n--- Layer {li} ---")
        
        # 收集number梯度
        sing_grads = []
        plur_grads = []
        for i, (sn, pn, sv, pv) in enumerate(NVA_PAIRS[:n_test]):
            g_s = compute_scalar_gradient(model, tokenizer, device, f"The {sn}", li, sv_id, pv_id)
            g_p = compute_scalar_gradient(model, tokenizer, device, f"The {pn}", li, sv_id, pv_id)
            if g_s is not None:
                sing_grads.append(g_s)
            if g_p is not None:
                plur_grads.append(g_p)
        
        # 收集tense梯度
        past_grads = []
        pres_grads = []
        for i, (past_v, pres_v) in enumerate(TENSE_PAIRS[:n_test]):
            g_past = compute_scalar_gradient(model, tokenizer, device, f"He {past_v}", li, past_id, pres_id)
            g_pres = compute_scalar_gradient(model, tokenizer, device, f"He {pres_v}", li, past_id, pres_id)
            if g_past is not None:
                past_grads.append(g_past)
            if g_pres is not None:
                pres_grads.append(g_pres)
        
        if len(sing_grads) < 5 or len(plur_grads) < 5:
            print(f"  Not enough number gradients")
            continue
        if len(past_grads) < 5 or len(pres_grads) < 5:
            print(f"  Not enough tense gradients")
            continue
        
        G_sing = np.array(sing_grads)  # [n, d]
        G_plur = np.array(plur_grads)
        G_past = np.array(past_grads)
        G_pres = np.array(pres_grads)
        
        # ★★★ 1. 传统cosine测试 (Phase 72的方法)
        g_spec_num = np.mean(G_sing, axis=0) - np.mean(G_plur, axis=0)
        g_spec_tense = np.mean(G_past, axis=0) - np.mean(G_pres, axis=0)
        
        n_num = np.linalg.norm(g_spec_num)
        n_ten = np.linalg.norm(g_spec_tense)
        cos_num_ten = np.dot(g_spec_num, g_spec_tense) / max(n_num * n_ten, 1e-10)
        
        # ★★★ 2. 随机cosine基线
        n_random_cos = 1000
        random_cos_vals = []
        all_grads = np.vstack([G_sing, G_plur])  # [2n, d]
        for _ in range(n_random_cos):
            idx = np.random.permutation(len(all_grads))
            v1 = all_grads[idx[0]]
            v2 = all_grads[idx[1]]
            cos_r = np.dot(v1, v2) / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-10)
            random_cos_vals.append(cos_r)
        
        # 更好的随机基线: 两个随机方向的期望cosine
        # E[cos(v1, v2)] = 0, Var[cos] = 1/d
        expected_cos_std = 1.0 / np.sqrt(d_model)
        
        print(f"  cos(g_spec_num, g_spec_tense) = {cos_num_ten:.4f}")
        print(f"  Random baseline: E[cos]=0, std=1/√d={expected_cos_std:.4f}")
        print(f"  Actual random sample: mean={np.mean(random_cos_vals):.4f}, std={np.std(random_cos_vals):.4f}")
        print(f"  Z-score of observed cos: {abs(cos_num_ten)/expected_cos_std:.2f}σ")
        
        # ★★★ 3. CCA测试
        G_number = np.vstack([G_sing, G_plur])   # [2n, d]
        G_tense = np.vstack([G_past, G_pres])     # [2m, d]
        
        # 需要相同行数
        min_n = min(len(G_number), len(G_tense))
        G_number = G_number[:min_n]
        G_tense = G_tense[:min_n]
        
        try:
            cca_num_ten = canonical_correlation(G_number, G_tense, k=5)
            print(f"  CCA(number, tense) top-5: {cca_num_ten.round(4)}")
        except Exception as e:
            print(f"  CCA(number, tense) failed: {e}")
            cca_num_ten = None
        
        # ★★★ 4. CCA随机基线: 同类梯度随机分成两组
        # 如果number和tense的子空间真的正交, CCA应该显著低于随机基线
        n_perms = 50
        cca_random_baseline = []
        
        # 混合所有梯度, 随机分成两组
        G_all = np.vstack([G_sing, G_plur, G_past, G_pres])  # [4n, d]
        
        for _ in range(n_perms):
            idx = np.random.permutation(len(G_all))
            mid = len(G_all) // 2
            G_rand1 = G_all[idx[:mid]]
            G_rand2 = G_all[idx[mid:]]
            try:
                cca_r = canonical_correlation(G_rand1, G_rand2, k=5)
                cca_random_baseline.append(cca_r)
            except:
                pass
        
        if cca_random_baseline:
            cca_baseline_mean = np.mean(cca_random_baseline, axis=0)
            print(f"  CCA random baseline top-5: {cca_baseline_mean.round(4)}")
            
            if cca_num_ten is not None:
                # 比较: CCA(number, tense) vs CCA(random)
                ratio = cca_num_ten / np.maximum(cca_baseline_mean, 1e-10)
                print(f"  CCA ratio (num_ten / random): {ratio.round(4)}")
                if np.all(ratio < 0.5):
                    print(f"  → ★★★ CCA显著低于随机基线 — 可能存在结构正交性")
                elif np.all(ratio > 0.8):
                    print(f"  → CCA接近随机基线 — 正交性可能是高维随机效应")
                else:
                    print(f"  → CCA部分低于随机基线 — 需要更多分析")
        
        # ★★★ 5. CKA split-half reliability test
        # Phase 72的CKA=1可疑 — 验证
        print(f"\n  CKA Split-Half Reliability Test:")
        
        # Sing split-half
        if len(G_sing) >= 6:
            mid = len(G_sing) // 2
            G_sing_a = G_sing[:mid]
            G_sing_b = G_sing[mid:]
            
            def linear_cka(X, Y):
                X = X - X.mean(axis=0, keepdims=True)
                Y = Y - Y.mean(axis=0, keepdims=True)
                XY = np.trace((X.T @ X) @ (Y.T @ Y))
                XX = np.trace((X.T @ X) @ (X.T @ X))
                YY = np.trace((Y.T @ Y) @ (Y.T @ Y))
                denom = np.sqrt(max(XX * YY, 1e-30))
                return XY / denom if denom > 1e-30 else 0.0
            
            cka_sing_split = linear_cka(G_sing_a, G_sing_b)
            cka_sing_plur = linear_cka(G_sing, G_plur)
            
            # Plur split-half
            mid_p = len(G_plur) // 2
            G_plur_a = G_plur[:mid_p]
            G_plur_b = G_plur[mid_p:]
            cka_plur_split = linear_cka(G_plur_a, G_plur_b)
            
            print(f"    CKA(sing_A, sing_B) = {cka_sing_split:.4f}")
            print(f"    CKA(plur_A, plur_B) = {cka_plur_split:.4f}")
            print(f"    CKA(sing, plur)     = {cka_sing_plur:.4f}")
            
            # 诊断
            if cka_sing_split > 0.99:
                print(f"    ⚠️ CKA=1可能是sample diversity太低(所有sing句子梯度太相似)")
            elif cka_sing_split < 0.5:
                print(f"    CKA<0.5说明sing子空间的内部一致性低")
        
        print(f"  {i+1}/{n_test} inputs done" if 'i' in dir() else "")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="a", choices=["a", "b", "c", "all"])
    parser.add_argument("--n_test", type=int, default=8)
    args = parser.parse_args()
    
    model, tokenizer, device = load_gpt2_float32()
    
    try:
        if args.exp in ["a", "all"]:
            results_a = experiment_a(model, tokenizer, device, n_test=args.n_test)
        
        if args.exp in ["b", "all"]:
            experiment_b(model, tokenizer, device, n_test=args.n_test)
        
        if args.exp in ["c", "all"]:
            experiment_c(model, tokenizer, device, n_test=args.n_test)
    finally:
        del model
        torch.cuda.empty_cache()
        gc.collect()
        print("\n[Phase73] Done. GPU memory released.")
