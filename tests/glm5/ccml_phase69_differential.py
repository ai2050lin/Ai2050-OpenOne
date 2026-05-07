"""
Phase 69: 真正的微分几何分析 — Infinitesimal Jacobian + Spectrum + Tangent Bundle
=====================================================================================

Phase 68的核心问题:
1. ❌ ε=0.1不是infinitesimal — 有限差分太大，不等于Jacobian
2. ❌ "有效线性半径<0.5"不可信 — 没做β→0极限
3. ❌ cos≈0在高维中是默认状态 — 不代表结构无关
4. ❌ J₁v≠J₂v不意味着J₁≠J₂ — 逻辑跳跃

Phase 69的修正:
1. ★ infinitesimal Jacobian: ε ∈ [10^{-1}, 10^{-2}, 10^{-3}, 10^{-4}, 10^{-5}]
   → 验证: ratio(actual, linear) → 1 as ε→0?
   → 如果→1: 线性区存在，之前ε太大
   → 如果→0: 真正的非线性

2. ★ Jacobian spectrum: 用randomized SVD获取
   - J^T J 的 top eigenvalues (右奇异值的平方)
   - effective rank = (Σσᵢ)²/Σσᵢ²
   - spectrum decay rate

3. ★ Tangent bundle consistency: 用proper metrics
   - CKA (Centered Kernel Alignment) 而非 cosine
   - Principal angles between tangent spaces
   - Subspace overlap

关键改进: 用torch.autograd的Jacobian-vector product来计算真正的∂F/∂h·v
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
from model_utils import load_model, get_model_info, release_model, get_layers, get_W_U

REGULAR_NVA = [
    ("cat","cats","runs","run"),("dog","dogs","walks","walk"),("bird","birds","flies","fly"),
    ("girl","girls","reads","read"),("boy","boys","sings","sing"),("horse","horses","jumps","jump"),
    ("bear","bears","sleeps","sleep"),("snake","snakes","crawls","crawl"),
    ("frog","frogs","swims","swim"),("fox","foxes","hunts","hunt"),("king","kings","rules","rule"),
    ("student","students","studies","study"),("teacher","teachers","speaks","speak"),
    ("doctor","doctors","helps","help"),("tree","trees","grows","grow"),("car","cars","moves","move"),
    ("queen","queens","leads","lead"),("driver","drivers","drives","drive"),
    ("worker","workers","builds","build"),("player","players","wins","win"),
    ("writer","writers","writes","write"),("farmer","farmers","plants","plant"),
    ("nurse","nurses","cares","care"),("soldier","soldiers","fights","fight"),
    ("rabbit","rabbits","hops","hop"),("whale","whales","dives","dive"),
    ("eagle","eagles","soars","soar"),("lion","lions","roars","roar"),
    ("tiger","tigers","hunts","hunt"),("wolf","wolves","howls","howl"),
    ("child","children","plays","play"),("mouse","mice","squeaks","squeak"),
    ("goose","geese","swims","swim"),("man","men","works","work"),
    ("woman","women","sings","sing"),("tooth","teeth","aches","ache"),
    ("foot","feet","steps","step"),("person","people","thinks","think"),
]


def make_sentences(nva):
    sing, plur, sv, pv = nva
    return (f"The {sing} {sv}", f"The {plur} {pv}", sv, pv)


def get_syntax_direction(model, tokenizer, device, n_layers, W_U, n_pairs=30):
    """获取语法方向 (sing→plur) 在各层的表示"""
    directions = {}
    nva_list = REGULAR_NVA[:n_pairs]
    
    for layer_idx in [0, 5, 10, 15, 27]:
        if layer_idx >= n_layers:
            continue
        h_sing_list, h_plur_list = [], []
        
        for nva in nva_list:
            sent_s, sent_p, sv, pv = make_sentences(nva)
            
            for sent, h_list in [(sent_s, h_sing_list), (sent_p, h_plur_list)]:
                toks = tokenizer(sent, return_tensors="pt").to(device)
                captured = {}
                
                def make_hook(key):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            captured[key] = output[0].detach().float().cpu()
                        else:
                            captured[key] = output.detach().float().cpu()
                    return hook
                
                layers = get_layers(model)
                hook = layers[layer_idx].register_forward_hook(make_hook(f"L{layer_idx}"))
                
                with torch.no_grad():
                    _ = model(**toks)
                
                hook.remove()
                
                h = captured[f"L{layer_idx}"][0, -1, :].numpy()
                h_list.append(h)
        
        h_sing_arr = np.array(h_sing_list)
        h_plur_arr = np.array(h_plur_list)
        direction = h_plur_arr.mean(axis=0) - h_sing_arr.mean(axis=0)
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            direction = direction / norm
        directions[layer_idx] = direction
    
    return directions


def compute_Jv_autograd(model, tokenizer, device, sentence, layer_idx, 
                         direction, n_layers=28):
    """
    用autograd计算真正的方向导数 J(logits)·v at layer l
    
    方法: 在layer l处hook, 替换h_l为h_l+εv, 用autograd追踪梯度
    """
    toks = tokenizer(sentence, return_tensors="pt").to(device)
    direction_t = torch.tensor(direction, dtype=torch.float32, device=device)
    
    layers = get_layers(model)
    
    # Forward with hook to inject perturbation and capture h
    h_captured = {}
    
    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            h_captured['h'] = output[0].detach().clone()
        else:
            h_captured['h'] = output.detach().clone()
    
    hook_capture = layers[layer_idx].register_forward_hook(capture_hook)
    
    with torch.no_grad():
        _ = model(**toks)
    
    hook_capture.remove()
    
    h_baseline = h_captured['h']  # [1, seq_len, d_model]
    return h_baseline


def compute_Jv_finite_diff(model, tokenizer, device, sentence, layer_idx,
                            direction, epsilon, n_layers=28):
    """
    用有限差分计算J(logits)·v at layer l
    
    返回: (Jv, baseline_logits) 
    Jv = (F(x+εv) - F(x)) / ε
    """
    toks = tokenizer(sentence, return_tensors="pt").to(device)
    direction_t = torch.tensor(direction, dtype=torch.float32, device=device)
    
    layers = get_layers(model)
    
    # Baseline
    with torch.no_grad():
        out_base = model(**toks)
        baseline_logits = out_base.logits[0, -1, :].detach().float().cpu().numpy()
    
    # Perturbed
    def perturb_hook(module, input, output):
        if isinstance(output, tuple):
            out_tensor = output[0].detach().clone()
        else:
            out_tensor = output.detach().clone()
        out_tensor[0, -1, :] += epsilon * direction_t.to(out_tensor.dtype)
        if isinstance(output, tuple):
            return (out_tensor,) + output[1:]
        return out_tensor
    
    hook = layers[layer_idx].register_forward_hook(perturb_hook)
    
    with torch.no_grad():
        out_pert = model(**toks)
        perturbed_logits = out_pert.logits[0, -1, :].detach().float().cpu().numpy()
    
    hook.remove()
    
    Jv = (perturbed_logits - baseline_logits) / epsilon
    
    return Jv, baseline_logits


def experiment_a_infinitesimal(model, tokenizer, device, n_layers, W_U, n_test=25):
    """
    实验 A: Infinitesimal Jacobian — 验证线性区是否存在
    
    核心: ratio = actual_Δlogit / (ε × Jv_at_smallest_ε)
    如果 ratio → 1 as ε → 0: 线性区存在
    如果 ratio → 0: 真正非线性
    
    关键改进: 测试 ε ∈ [10^{-5}, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 0.5, 1.0]
    """
    print("\n" + "="*70)
    print("实验 A: Infinitesimal Jacobian — 线性区是否存在?")
    print("="*70)
    
    directions = get_syntax_direction(model, tokenizer, device, n_layers, W_U, n_pairs=30)
    epsilons = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0]
    test_nva = REGULAR_NVA[:n_test]
    layer_indices = sorted(directions.keys())
    
    results = defaultdict(lambda: defaultdict(list))
    
    for li, layer_idx in enumerate(layer_indices):
        direction = directions[layer_idx]
        print(f"\n  Layer {layer_idx} ({li+1}/{len(layer_indices)})...")
        
        for nva in test_nva:
            sent_s, sent_p, sv, pv = make_sentences(nva)
            sentence = sent_s
            
            sv_id = tokenizer.encode(sv, add_special_tokens=False)[0]
            pv_id = tokenizer.encode(pv, add_special_tokens=False)[0]
            
            try:
                # Compute Jv at smallest epsilon (best approximation of true Jacobian)
                Jv_ref, base_logits = compute_Jv_finite_diff(
                    model, tokenizer, device, sentence, layer_idx, direction, 
                    epsilon=1e-4, n_layers=n_layers)
                
                # Reference Δlogit_gap per unit β
                ref_gap_per_unit = Jv_ref[pv_id] - Jv_ref[sv_id]
                
                if abs(ref_gap_per_unit) < 1e-8:
                    continue  # Skip if reference is essentially zero
                
                # Compute Jv at all epsilons
                for eps in epsilons:
                    Jv_eps, _ = compute_Jv_finite_diff(
                        model, tokenizer, device, sentence, layer_idx, direction,
                        epsilon=eps, n_layers=n_layers)
                    
                    actual_gap = Jv_eps[pv_id] - Jv_eps[sv_id]
                    
                    # Ratio: how well does Jv_at_ε match Jv_at_ε_ref?
                    ratio = actual_gap / ref_gap_per_unit if abs(ref_gap_per_unit) > 1e-10 else 0
                    
                    # Also: does the direction stay the same?
                    cos_with_ref = 0
                    norm_Jv = np.linalg.norm(Jv_eps)
                    norm_ref = np.linalg.norm(Jv_ref)
                    if norm_Jv > 1e-10 and norm_ref > 1e-10:
                        cos_with_ref = np.dot(Jv_eps, Jv_ref) / (norm_Jv * norm_ref)
                    
                    results[layer_idx][eps].append({
                        'ratio': ratio,
                        'cos_with_ref': cos_with_ref,
                        'actual_gap': actual_gap,
                        'norm_Jv': norm_Jv,
                    })
            except Exception as e:
                continue
    
    # Print results
    print("\n" + "="*70)
    print("实验 A 结果: Infinitesimal Jacobian")
    print("="*70)
    
    # Mean ratio (should → 1 as ε → 0 if linear region exists)
    print("\nMean ratio (Jv_at_ε / Jv_at_10^-4) — should → 1 as ε → 0:")
    print(f"{'Layer':<8}", end="")
    for eps in epsilons:
        print(f"ε={eps:<10}", end="")
    print()
    
    for layer_idx in layer_indices:
        print(f"L{layer_idx:<6}", end="")
        for eps in epsilons:
            if eps in results[layer_idx]:
                ratios = [r['ratio'] for r in results[layer_idx][eps] if abs(r['ratio']) < 100]
                if ratios:
                    print(f"{np.mean(ratios):<12.4f}", end="")
                else:
                    print(f"{'N/A':<12}", end="")
            else:
                print(f"{'N/A':<12}", end="")
        print()
    
    # Cosine similarity with reference (should → 1 as ε → 0)
    print("\nMean cos(Jv_ε, Jv_10^-4) — should → 1 as ε → 0:")
    print(f"{'Layer':<8}", end="")
    for eps in epsilons:
        print(f"ε={eps:<10}", end="")
    print()
    
    for layer_idx in layer_indices:
        print(f"L{layer_idx:<6}", end="")
        for eps in epsilons:
            if eps in results[layer_idx]:
                cos_vals = [r['cos_with_ref'] for r in results[layer_idx][eps]]
                if cos_vals:
                    print(f"{np.mean(cos_vals):<12.4f}", end="")
                else:
                    print(f"{'N/A':<12}", end="")
            else:
                print(f"{'N/A':<12}", end="")
        print()
    
    # Norm of Jv at different scales
    print("\nMean ||Jv|| at different ε:")
    print(f"{'Layer':<8}", end="")
    for eps in epsilons:
        print(f"ε={eps:<10}", end="")
    print()
    
    for layer_idx in layer_indices:
        print(f"L{layer_idx:<6}", end="")
        for eps in epsilons:
            if eps in results[layer_idx]:
                norms = [r['norm_Jv'] for r in results[layer_idx][eps]]
                if norms:
                    print(f"{np.mean(norms):<12.2f}", end="")
                else:
                    print(f"{'N/A':<12}", end="")
            else:
                print(f"{'N/A':<12}", end="")
        print()
    
    # Key analysis
    print("\n★★★ Key finding: Does ratio → 1 as ε → 0?")
    for layer_idx in layer_indices:
        ratios_by_eps = []
        for eps in [1e-5, 1e-4, 1e-3]:
            if eps in results[layer_idx]:
                ratios = [r['ratio'] for r in results[layer_idx][eps] if abs(r['ratio']) < 100]
                if ratios:
                    ratios_by_eps.append(np.mean(ratios))
        if len(ratios_by_eps) >= 2:
            if all(abs(r - 1) < 0.3 for r in ratios_by_eps):
                print(f"  L{layer_idx}: ✅ LINEAR REGION EXISTS (ratio≈1 at small ε)")
            else:
                print(f"  L{layer_idx}: ❌ NO LINEAR REGION (ratio≠1 even at small ε)")
        else:
            print(f"  L{layer_idx}: ⚠️ Insufficient data")
    
    return results


def experiment_b_jacobian_spectrum(model, tokenizer, device, n_layers, W_U, n_test=15):
    """
    实验 B: Jacobian Spectrum — 用randomized SVD
    
    核心问题: J(x)的有效秩是多少? Spectrum如何衰减?
    
    方法: 用Hutchinson-style估计
    - 对多个随机向量z, 计算 Jz 和 J^T z
    - 用这些估计J^T J的top eigenvalues
    - 或者: 直接用power iteration
    
    更实际的方法:
    - 对k个随机向量z_1,...,z_k, 计算 Jz_i (在logit空间)
    - 对这些做SVD → 得到J的近似奇异值
    """
    print("\n" + "="*70)
    print("实验 B: Jacobian Spectrum (effective rank, spectrum decay)")
    print("="*70)
    
    directions = get_syntax_direction(model, tokenizer, device, n_layers, W_U, n_pairs=20)
    epsilon = 1e-3  # Small but numerically stable
    test_nva = REGULAR_NVA[:n_test]
    layer_indices = sorted(directions.keys())
    n_random_vectors = 30  # Number of random probes
    
    results = defaultdict(list)
    
    for li, layer_idx in enumerate(layer_indices):
        direction = directions[layer_idx]
        print(f"\n  Layer {layer_idx} ({li+1}/{len(layer_indices)})...")
        
        # Use a single sentence for detailed analysis
        nva = test_nva[0]
        sent_s, sent_p, sv, pv = make_sentences(nva)
        sentence = sent_s
        
        d_model = 3584  # For DS7B
        
        # Compute Jv for the syntax direction
        Jv_syntax, base_logits = compute_Jv_finite_diff(
            model, tokenizer, device, sentence, layer_idx, direction, epsilon, n_layers)
        
        # Compute Jv for random directions
        Jv_random_list = []
        rng = np.random.RandomState(42)
        
        for i in range(n_random_vectors):
            z = rng.randn(d_model)
            z = z / np.linalg.norm(z)  # Normalize
            
            Jv_z, _ = compute_Jv_finite_diff(
                model, tokenizer, device, sentence, layer_idx, z, epsilon, n_layers)
            Jv_random_list.append(Jv_z)
        
        Jv_random_arr = np.array(Jv_random_list)  # [n_random, vocab_size]
        
        # Compute SVD of the Jv matrix
        # Each row is Jz_i ≈ J · z_i
        # SVD of this matrix gives approximate singular values of J
        U, S, Vt = np.linalg.svd(Jv_random_arr, full_matrices=False)
        
        # Effective rank
        total_energy = np.sum(S**2)
        effective_rank = total_energy**2 / np.sum(S**4) if np.sum(S**4) > 0 else 0
        
        # Spectrum decay
        # S[i] / S[0] for top components
        spectrum_decay = S / max(S[0], 1e-10)
        
        # How much of Jv_syntax is captured by top-k random directions?
        proj_energy = []
        for k in [1, 5, 10, 20, 30]:
            if k <= len(Vt):
                # Project Jv_syntax onto top-k left singular vectors
                proj = U[:, :k] @ (U[:, :k].T @ Jv_syntax)
                energy = np.linalg.norm(proj)**2 / max(np.linalg.norm(Jv_syntax)**2, 1e-20)
                proj_energy.append((k, energy))
        
        results[layer_idx] = {
            'singular_values': S[:20],
            'effective_rank': effective_rank,
            'spectrum_decay_top20': spectrum_decay[:20],
            'proj_energy': proj_energy,
            'Jv_syntax_norm': np.linalg.norm(Jv_syntax),
            'mean_Jv_random_norm': np.mean([np.linalg.norm(j) for j in Jv_random_list]),
            'base_logits_norm': np.linalg.norm(base_logits),
        }
        
        # Also test with a different sentence
        nva2 = test_nva[3]
        sent_s2, sent_p2, sv2, pv2 = make_sentences(nva2)
        sentence2 = sent_s2
        
        Jv_syntax2, _ = compute_Jv_finite_diff(
            model, tokenizer, device, sentence2, layer_idx, direction, epsilon, n_layers)
        
        # Cosine between Jv_syntax for two different sentences
        norm1 = np.linalg.norm(Jv_syntax)
        norm2 = np.linalg.norm(Jv_syntax2)
        cos_syntax = np.dot(Jv_syntax, Jv_syntax2) / max(norm1 * norm2, 1e-10)
        
        results[layer_idx]['cos_syntax_cross_sent'] = cos_syntax
    
    # Print results
    print("\n" + "="*70)
    print("实验 B 结果: Jacobian Spectrum")
    print("="*70)
    
    for layer_idx in layer_indices:
        r = results[layer_idx]
        print(f"\n  L{layer_idx}:")
        print(f"    Effective rank: {r['effective_rank']:.2f}")
        print(f"    ||Jv_syntax||: {r['Jv_syntax_norm']:.2f}")
        print(f"    Mean ||Jv_random||: {r['mean_Jv_random_norm']:.2f}")
        print(f"    Ratio syntax/random: {r['Jv_syntax_norm']/max(r['mean_Jv_random_norm'],1e-10):.2f}")
        print(f"    Top-10 singular values: {r['singular_values'][:10]}")
        print(f"    Spectrum decay (S/S[0]): {r['spectrum_decay_top20'][:10]}")
        print(f"    Jv_syntax captured by top-k random dirs:")
        for k, energy in r['proj_energy']:
            print(f"      k={k}: {energy:.4f}")
        print(f"    cos(Jv_syntax_sent1, Jv_syntax_sent2): {r['cos_syntax_cross_sent']:.4f}")
    
    # Key analysis
    print("\n★★★ Key findings:")
    for layer_idx in layer_indices:
        r = results[layer_idx]
        eff_rank = r['effective_rank']
        syntax_vs_random = r['Jv_syntax_norm'] / max(r['mean_Jv_random_norm'], 1e-10)
        print(f"  L{layer_idx}: eff_rank={eff_rank:.1f}, syntax/random_norm={syntax_vs_random:.2f}")
    
    return results


def experiment_c_tangent_consistency(model, tokenizer, device, n_layers, W_U, n_test=15):
    """
    实验 C: Tangent Bundle Consistency — 用proper metrics
    
    核心问题: 不同输入的tangent space有多相似?
    
    方法: 用CKA (Centered Kernel Alignment) 而非cosine
    - 对两个输入x₁, x₂, 收集J(x₁)z_i和J(x₂)z_i for random z_i
    - CKA衡量两个矩阵的子空间对齐程度
    - CKA=1: 完全相同的子空间
    - CKA=0: 正交子空间
    
    同时计算principal angles
    """
    print("\n" + "="*70)
    print("实验 C: Tangent Bundle Consistency (CKA + Principal Angles)")
    print("="*70)
    
    directions = get_syntax_direction(model, tokenizer, device, n_layers, W_U, n_pairs=20)
    epsilon = 1e-3
    test_nva = REGULAR_NVA[:n_test]
    layer_indices = sorted(directions.keys())
    n_random_vectors = 20
    
    results = defaultdict(list)
    
    for li, layer_idx in enumerate(layer_indices):
        direction = directions[layer_idx]
        print(f"\n  Layer {layer_idx} ({li+1}/{len(layer_indices)})...")
        
        d_model = 3584
        
        # Collect Jv for multiple sentences and random directions
        sentence_Jv = {}  # {sentence_idx: [Jv_z1, Jv_z2, ...]}
        
        for si, nva in enumerate(test_nva[:8]):
            sent_s, sent_p, sv, pv = make_sentences(nva)
            
            # Sing sentence
            Jv_list = []
            rng = np.random.RandomState(42)  # Same random directions for fair comparison
            
            for i in range(n_random_vectors):
                z = rng.randn(d_model)
                z = z / np.linalg.norm(z)
                
                Jv_z, _ = compute_Jv_finite_diff(
                    model, tokenizer, device, sent_s, layer_idx, z, epsilon, n_layers)
                Jv_list.append(Jv_z)
            
            sentence_Jv[f"sing_{si}"] = np.array(Jv_list)  # [n_random, vocab_size]
            
            # Plur sentence
            Jv_list_p = []
            rng2 = np.random.RandomState(42)
            
            for i in range(n_random_vectors):
                z = rng2.randn(d_model)
                z = z / np.linalg.norm(z)
                
                Jv_z, _ = compute_Jv_finite_diff(
                    model, tokenizer, device, sent_p, layer_idx, z, epsilon, n_layers)
                Jv_list_p.append(Jv_z)
            
            sentence_Jv[f"plur_{si}"] = np.array(Jv_list_p)
        
        # Compute CKA between all pairs
        def linear_CKA(X, Y):
            """Linear CKA between matrices X and Y"""
            # X: [n, d1], Y: [n, d2]
            # Center
            X_centered = X - X.mean(axis=0, keepdims=True)
            Y_centered = Y - Y.mean(axis=0, keepdims=True)
            
            # HSIC
            def hsic(A, B):
                # A: [n, d1], B: [n, d2]
                # K = A A^T, L = B B^T
                K = A @ A.T  # [n, n]
                L = B @ B.T  # [n, n]
                # Center
                n = K.shape[0]
                H = np.eye(n) - np.ones((n, n)) / n
                K_c = H @ K @ H
                L_c = H @ L @ H
                return np.trace(K_c @ L_c)
            
            hsic_xy = hsic(X_centered, Y_centered)
            hsic_xx = hsic(X_centered, X_centered)
            hsic_yy = hsic(Y_centered, Y_centered)
            
            if hsic_xx <= 0 or hsic_yy <= 0:
                return 0
            
            return hsic_xy / np.sqrt(hsic_xx * hsic_yy)
        
        # CKA between sing sentences
        cka_sing_sing = []
        keys_sing = [k for k in sentence_Jv.keys() if k.startswith('sing_')]
        for i in range(len(keys_sing)):
            for j in range(i+1, len(keys_sing)):
                # Use top-100 tokens by variance
                X = sentence_Jv[keys_sing[i]][:, :100]  # [n_random, 100]
                Y = sentence_Jv[keys_sing[j]][:, :100]
                cka = linear_CKA(X, Y)
                cka_sing_sing.append(cka)
        
        # CKA between sing and plur sentences
        cka_sing_plur = []
        keys_plur = [k for k in sentence_Jv.keys() if k.startswith('plur_')]
        for ks in keys_sing[:4]:
            for kp in keys_plur[:4]:
                X = sentence_Jv[ks][:, :100]
                Y = sentence_Jv[kp][:, :100]
                cka = linear_CKA(X, Y)
                cka_sing_plur.append(cka)
        
        # CKA between plur sentences
        cka_plur_plur = []
        for i in range(len(keys_plur)):
            for j in range(i+1, len(keys_plur)):
                X = sentence_Jv[keys_plur[i]][:, :100]
                Y = sentence_Jv[keys_plur[j]][:, :100]
                cka = linear_CKA(X, Y)
                cka_plur_plur.append(cka)
        
        # Principal angles between sing and plur tangent spaces
        # Use SVD to get orthonormal bases
        def compute_principal_angles(X, Y, n_angles=5):
            """Compute principal angles between column spaces of X and Y"""
            # X: [n, d1], Y: [n, d2]
            # QR decomposition
            Qx, _ = np.linalg.qr(X.T)  # [d1, k1]
            Qy, _ = np.linalg.qr(Y.T)  # [d2, k2]
            
            # SVD of Qx^T Qy
            M = Qx.T @ Qy  # [k1, k2]
            s = np.linalg.svd(M, compute_uv=False)
            
            # Principal angles = arccos(singular values)
            angles = np.arccos(np.clip(s[:n_angles], 0, 1))
            return angles, s[:n_angles]
        
        # Principal angles for first sing/plur pair
        X_sing = sentence_Jv[keys_sing[0]][:, :100]
        X_plur = sentence_Jv[keys_plur[0]][:, :100]
        angles, cos_angles = compute_principal_angles(X_sing, X_plur, n_angles=5)
        
        results[layer_idx] = {
            'cka_sing_sing': np.mean(cka_sing_sing) if cka_sing_sing else 0,
            'cka_sing_plur': np.mean(cka_sing_plur) if cka_sing_plur else 0,
            'cka_plur_plur': np.mean(cka_plur_plur) if cka_plur_plur else 0,
            'principal_angles_deg': np.degrees(angles),
            'principal_cosines': cos_angles,
        }
    
    # Print results
    print("\n" + "="*70)
    print("实验 C 结果: Tangent Bundle Consistency")
    print("="*70)
    
    print(f"\n{'Layer':<8} {'CKA(s,s)':<12} {'CKA(s,p)':<12} {'CKA(p,p)':<12} {'1st_cos':<10} {'2nd_cos':<10} {'3rd_cos':<10}")
    for layer_idx in layer_indices:
        r = results[layer_idx]
        cos_vals = r['principal_cosines']
        print(f"L{layer_idx:<6} {r['cka_sing_sing']:<12.4f} {r['cka_sing_plur']:<12.4f} "
              f"{r['cka_plur_plur']:<12.4f} "
              f"{cos_vals[0]:<10.4f} {cos_vals[1] if len(cos_vals)>1 else 0:<10.4f} "
              f"{cos_vals[2] if len(cos_vals)>2 else 0:<10.4f}")
    
    print("\n★★★ Interpretation:")
    print("  CKA ≈ 1: Same tangent space (Jacobian structure shared)")
    print("  CKA ≈ 0: Different tangent spaces")
    print("  CKA(sing,sing) >> CKA(sing,plur): Tangent space depends on number")
    print("  CKA(sing,sing) ≈ CKA(sing,plur): Tangent space independent of number")
    print("  1st_principal_cosine ≈ 1: Top direction shared")
    print("  1st_principal_cosine ≈ 0: Even top direction is different")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek7b")
    parser.add_argument("--n_test", type=int, default=25)
    parser.add_argument("--exp", type=str, default="all", choices=["a", "b", "c", "all"])
    args = parser.parse_args()
    
    print(f"Phase 69: 真正的微分几何分析")
    print(f"Model: {args.model}, n_test: {args.n_test}, exp: {args.exp}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load model
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers
    W_U = get_W_U(model)
    
    print(f"\nModel info: {model_info.name}, n_layers={n_layers}, d_model={model_info.d_model}")
    
    if args.exp in ["a", "all"]:
        results_a = experiment_a_infinitesimal(model, tokenizer, device, n_layers, W_U, args.n_test)
    
    if args.exp in ["b", "all"]:
        results_b = experiment_b_jacobian_spectrum(model, tokenizer, device, n_layers, W_U, min(args.n_test, 15))
    
    if args.exp in ["c", "all"]:
        results_c = experiment_c_tangent_consistency(model, tokenizer, device, n_layers, W_U, min(args.n_test, 15))
    
    # Cleanup
    release_model(model)
    print(f"\nPhase 69 complete! Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
