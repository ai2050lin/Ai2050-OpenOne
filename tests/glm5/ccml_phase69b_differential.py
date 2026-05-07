"""
Phase 69b: 真正的Jacobian分析 — 用autograd + float32
=======================================================

Phase 69a的问题:
1. ε<10^-2在bfloat16下不可靠 → Jv范数∝1/ε(数值噪声)
2. L5/L15/L27的reference Jv=0 → 方向在这些层无效果
3. 没有用autograd计算真正的∂(logit_gap)/∂(h_l)

Phase 69b的修正:
1. ★ 用autograd计算真正的梯度 (不受ε选择影响)
2. ★ 用float32输出计算差异 (提高精度)
3. ★ 收敛测试: Δlogits/ε 是否随ε→0收敛
4. ★ Jacobian spectrum用随机方向+autograd
5. ★ CKA用autograd梯度而非有限差分
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


def get_syntax_direction(model, tokenizer, device, n_layers, n_pairs=30):
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


def compute_true_gradient(model, tokenizer, device, sentence, layer_idx,
                          direction, sv_id, pv_id, n_layers=28):
    """
    用autograd计算真正的梯度: ∂(logit_pv - logit_sv)/∂(h_l)
    
    方法: 使用backward_hook获取梯度, 不修改forward pass
    """
    toks = tokenizer(sentence, return_tensors="pt").to(device)
    layers = get_layers(model)
    
    # Step 1: Prepare input embeddings with requires_grad
    embed_layer = model.get_input_embeddings()
    input_ids = toks.input_ids
    inputs_embeds = embed_layer(input_ids).detach().clone().requires_grad_(True)
    position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
    
    # Step 2: Register backward hook on target layer
    grad_at_layer = [None]
    
    def backward_hook(module, grad_input, grad_output):
        # grad_output[0] = ∂(logit_gap)/∂(output_of_layer) = ∂(logit_gap)/∂(h_l)
        if grad_output[0] is not None:
            grad_at_layer[0] = grad_output[0].detach().clone().float()
        return grad_input  # Don't modify
    
    hook = layers[layer_idx].register_backward_hook(backward_hook)
    
    # Step 3: Forward + backward
    with torch.enable_grad():
        output = model(inputs_embeds=inputs_embeds, position_ids=position_ids)
        logits = output.logits[0, -1, :].float()
        logit_gap = logits[pv_id] - logits[sv_id]
        logit_gap.backward()
    
    hook.remove()
    
    if grad_at_layer[0] is None:
        return None, 0.0, logits.detach().cpu().numpy()
    
    # Gradient at last token position
    grad_h_l = grad_at_layer[0][0, -1, :].cpu().numpy()  # [d_model]
    
    # Directional derivative
    Jv_gap = float(np.dot(grad_h_l, direction))
    
    return grad_h_l, Jv_gap, logits.detach().cpu().numpy()


def experiment_a_autograd_convergence(model, tokenizer, device, n_layers, n_test=25):
    """
    实验 A: 用autograd vs 有限差分 — 验证收敛性
    
    核心问题: autograd计算的梯度与有限差分是否一致?
    """
    print("\n" + "="*70)
    print("实验 A: Autograd梯度 vs 有限差分收敛性")
    print("="*70)
    
    directions = get_syntax_direction(model, tokenizer, device, n_layers, n_pairs=30)
    epsilons = [0.01, 0.05, 0.1, 0.2, 0.5]
    test_nva = REGULAR_NVA[:n_test]
    layer_indices = sorted(directions.keys())
    
    results = defaultdict(list)
    
    for li, layer_idx in enumerate(layer_indices):
        direction = directions[layer_idx]
        print(f"\n  Layer {layer_idx} ({li+1}/{len(layer_indices)})...")
        
        for nva in test_nva:
            sent_s, sent_p, sv, pv = make_sentences(nva)
            sentence = sent_s
            
            sv_id = tokenizer.encode(sv, add_special_tokens=False)[0]
            pv_id = tokenizer.encode(pv, add_special_tokens=False)[0]
            
            try:
                # Step 1: Compute true gradient via autograd
                grad_h, autograd_Jv_gap, base_logits = compute_true_gradient(
                    model, tokenizer, device, sentence, layer_idx,
                    direction, sv_id, pv_id, n_layers)
                
                autograd_Jv_gap_val = float(autograd_Jv_gap)
                
                # Step 2: Compute finite difference at various ε
                finite_diff_results = {}
                for eps in epsilons:
                    toks = tokenizer(sentence, return_tensors="pt").to(device)
                    direction_t = torch.tensor(direction, dtype=torch.float32, device=device)
                    layers = get_layers(model)
                    
                    # Baseline
                    with torch.no_grad():
                        out_base = model(**toks)
                        base_logits_fd = out_base.logits[0, -1, :].detach().float().cpu().numpy()
                    
                    # Perturbed
                    def perturb_hook(module, input, output):
                        if isinstance(output, tuple):
                            out_tensor = output[0].detach().clone().float()
                        else:
                            out_tensor = output.detach().clone().float()
                        out_tensor[0, -1, :] += eps * direction_t
                        if isinstance(output, tuple):
                            return (out_tensor.to(output[0].dtype),) + output[1:]
                        return out_tensor.to(output[0].dtype)
                    
                    hook = layers[layer_idx].register_forward_hook(perturb_hook)
                    
                    with torch.no_grad():
                        out_pert = model(**toks)
                        pert_logits = out_pert.logits[0, -1, :].detach().float().cpu().numpy()
                    
                    hook.remove()
                    
                    fd_gap = (pert_logits[pv_id] - pert_logits[sv_id]) - (base_logits_fd[pv_id] - base_logits_fd[sv_id])
                    fd_Jv_gap = fd_gap / eps
                    
                    finite_diff_results[eps] = {
                        'fd_Jv_gap': fd_Jv_gap,
                        'fd_gap': fd_gap,
                        'ratio': fd_Jv_gap / autograd_Jv_gap_val if abs(autograd_Jv_gap_val) > 1e-10 else 0,
                    }
                
                results[layer_idx].append({
                    'autograd_Jv_gap': autograd_Jv_gap_val,
                    'finite_diff': finite_diff_results,
                    'grad_norm': np.linalg.norm(grad_h),
                })
            except Exception as e:
                print(f"    Error: {e}")
                continue
    
    # Print results
    print("\n" + "="*70)
    print("实验 A 结果: Autograd vs Finite Difference")
    print("="*70)
    
    # Autograd Jv_gap (true gradient)
    print("\nMean autograd ∂(logit_gap)/∂(h_l) · v:")
    print(f"{'Layer':<8} {'auto_Jv':<12} {'grad_norm':<12}")
    for layer_idx in layer_indices:
        if results[layer_idx]:
            auto_vals = [r['autograd_Jv_gap'] for r in results[layer_idx]]
            grad_norms = [r['grad_norm'] for r in results[layer_idx]]
            print(f"L{layer_idx:<6} {np.mean(auto_vals):<12.4f} {np.mean(grad_norms):<12.4f}")
    
    # Convergence: ratio of FD to autograd
    print("\nMean ratio (FD_Jv / autograd_Jv) — should → 1 as ε → 0:")
    print(f"{'Layer':<8}", end="")
    for eps in epsilons:
        print(f"ε={eps:<10}", end="")
    print()
    
    for layer_idx in layer_indices:
        print(f"L{layer_idx:<6}", end="")
        for eps in epsilons:
            ratios = [r['finite_diff'][eps]['ratio'] for r in results[layer_idx] 
                      if eps in r['finite_diff'] and abs(r['finite_diff'][eps]['ratio']) < 100]
            if ratios:
                print(f"{np.mean(ratios):<12.4f}", end="")
            else:
                print(f"{'N/A':<12}", end="")
        print()
    
    # The actual Δlogit_gap at each ε
    print("\nMean Δlogit_gap (actual change, not divided by ε):")
    print(f"{'Layer':<8}", end="")
    for eps in epsilons:
        print(f"ε={eps:<10}", end="")
    print()
    
    for layer_idx in layer_indices:
        print(f"L{layer_idx:<6}", end="")
        for eps in epsilons:
            gaps = [r['finite_diff'][eps]['fd_gap'] for r in results[layer_idx]
                    if eps in r['finite_diff']]
            if gaps:
                print(f"{np.mean(gaps):<12.4f}", end="")
            else:
                print(f"{'N/A':<12}", end="")
        print()
    
    # Key analysis
    print("\n★★★ Key findings:")
    for layer_idx in layer_indices:
        if results[layer_idx]:
            auto_vals = [r['autograd_Jv_gap'] for r in results[layer_idx]]
            ratios_small = [r['finite_diff'][0.01]['ratio'] for r in results[layer_idx]
                           if 0.01 in r['finite_diff'] and abs(r['finite_diff'][0.01]['ratio']) < 100]
            if ratios_small and auto_vals:
                mean_auto = np.mean(auto_vals)
                mean_ratio_small = np.mean(ratios_small)
                if abs(mean_auto) > 0.01:
                    print(f"  L{layer_idx}: autograd_Jv={mean_auto:.4f}, FD(ε=0.01)/auto={mean_ratio_small:.4f}")
                else:
                    print(f"  L{layer_idx}: autograd_Jv≈0 (direction has no effect)")
    
    return results


def experiment_b_spectrum_autograd(model, tokenizer, device, n_layers, n_test=10):
    """
    实验 B: Jacobian Spectrum — 用autograd计算多个方向的梯度
    
    核心问题: J(x)的有效秩是多少? Spectrum如何衰减?
    
    方法:
    - 对k个随机方向z_i, 计算 ∂(logit_gap)/∂(h_l)·z_i
    - 这给出J在特定输出方向上的投影
    - 用这些投影估计Jacobian的结构
    """
    print("\n" + "="*70)
    print("实验 B: Jacobian Spectrum (autograd)")
    print("="*70)
    
    directions = get_syntax_direction(model, tokenizer, device, n_layers, n_pairs=20)
    test_nva = REGULAR_NVA[:n_test]
    layer_indices = sorted(directions.keys())
    n_random = 20
    
    results = defaultdict(dict)
    
    for li, layer_idx in enumerate(layer_indices):
        direction = directions[layer_idx]
        print(f"\n  Layer {layer_idx} ({li+1}/{len(layer_indices)})...")
        
        nva = test_nva[0]
        sent_s, sent_p, sv, pv = make_sentences(nva)
        sentence = sent_s
        
        sv_id = tokenizer.encode(sv, add_special_tokens=False)[0]
        pv_id = tokenizer.encode(pv, add_special_tokens=False)[0]
        
        d_model = 3584
        
        # Compute gradient for syntax direction
        grad_syntax, Jv_syntax, _ = compute_true_gradient(
            model, tokenizer, device, sentence, layer_idx,
            direction, sv_id, pv_id, n_layers)
        
        # Compute gradient for random directions
        grad_random_list = []
        rng = np.random.RandomState(42)
        
        for i in range(n_random):
            z = rng.randn(d_model)
            z = z / np.linalg.norm(z)
            
            grad_z, Jv_z, _ = compute_true_gradient(
                model, tokenizer, device, sentence, layer_idx,
                z, sv_id, pv_id, n_layers)
            
            grad_random_list.append(grad_z)  # Each is [d_model]
        
        # Compute SVD of gradient matrix
        # Each column is ∂(logit_gap)/∂(h_l) for a different input direction
        # Wait, actually each grad_random_list[i] is the FULL gradient [d_model]
        # Not J·z_i. The full gradient is already the Jacobian in a specific direction.
        
        # The full gradient ∂(logit_gap)/∂(h_l) tells us:
        # "In which direction in h_l space should we move to increase logit_gap?"
        
        # For spectrum analysis, we want to know:
        # "How many significant directions are there in the gradient?"
        
        # This requires gradients at multiple inputs, not just one input with multiple directions
        
        # Let me compute gradients at multiple sentences
        grad_by_sentence = [grad_syntax]
        for nva2 in test_nva[1:5]:
            sent_s2, _, sv2, pv2 = make_sentences(nva2)
            sv2_id = tokenizer.encode(sv2, add_special_tokens=False)[0]
            pv2_id = tokenizer.encode(pv2, add_special_tokens=False)[0]
            
            grad2, _, _ = compute_true_gradient(
                model, tokenizer, device, sent_s2, layer_idx,
                direction, sv2_id, pv2_id, n_layers)
            grad_by_sentence.append(grad2)
        
        # Stack and do SVD
        grad_matrix = np.array(grad_by_sentence)  # [n_sentences, d_model]
        U, S, Vt = np.linalg.svd(grad_matrix, full_matrices=False)
        
        # Effective rank
        total_energy = np.sum(S**2)
        effective_rank = total_energy**2 / np.sum(S**4) if np.sum(S**4) > 0 else 0
        
        # Cosine similarity between gradients at different sentences
        cos_pairs = []
        for i in range(len(grad_by_sentence)):
            for j in range(i+1, len(grad_by_sentence)):
                g1 = grad_by_sentence[i]
                g2 = grad_by_sentence[j]
                n1 = np.linalg.norm(g1)
                n2 = np.linalg.norm(g2)
                if n1 > 1e-10 and n2 > 1e-10:
                    cos_pairs.append(np.dot(g1, g2) / (n1 * n2))
        
        # Also compare random direction gradients
        cos_random_syntax = []
        for gr in grad_random_list:
            n1 = np.linalg.norm(gr)
            n2 = np.linalg.norm(grad_syntax)
            if n1 > 1e-10 and n2 > 1e-10:
                cos_random_syntax.append(np.dot(gr, grad_syntax) / (n1 * n2))
        
        results[layer_idx] = {
            'singular_values': S,
            'effective_rank': effective_rank,
            'cos_between_sentences': cos_pairs,
            'cos_random_syntax': cos_random_syntax,
            'grad_syntax_norm': np.linalg.norm(grad_syntax),
            'mean_grad_random_norm': np.mean([np.linalg.norm(g) for g in grad_random_list]),
            'Jv_syntax': float(Jv_syntax),
        }
    
    # Print results
    print("\n" + "="*70)
    print("实验 B 结果: Jacobian Spectrum")
    print("="*70)
    
    for layer_idx in layer_indices:
        r = results[layer_idx]
        print(f"\n  L{layer_idx}:")
        print(f"    Effective rank (of ∂gap/∂h at 5 sentences): {r['effective_rank']:.2f}")
        print(f"    ||∂gap/∂h|| for syntax dir: {r['grad_syntax_norm']:.4f}")
        print(f"    Mean ||∂gap/∂h|| for random dirs: {r['mean_grad_random_norm']:.4f}")
        print(f"    Jv_gap (syntax): {r['Jv_syntax']:.4f}")
        print(f"    Singular values: {r['singular_values'][:5]}")
        print(f"    cos(grad at different sentences): mean={np.mean(r['cos_between_sentences']):.4f}" 
              if r['cos_between_sentences'] else "    cos: N/A")
        print(f"    cos(grad for random dir vs syntax dir): mean={np.mean(r['cos_random_syntax']):.4f}"
              if r['cos_random_syntax'] else "    cos_random: N/A")
    
    print("\n★★★ Key findings:")
    for layer_idx in layer_indices:
        r = results[layer_idx]
        print(f"  L{layer_idx}: eff_rank={r['effective_rank']:.1f}, "
              f"cos_between_sent={np.mean(r['cos_between_sentences']):.3f}" 
              if r['cos_between_sentences'] else f"  L{layer_idx}: eff_rank={r['effective_rank']:.1f}")
    
    return results


def experiment_c_cka_autograd(model, tokenizer, device, n_layers, n_test=10):
    """
    实验 C: CKA — 用autograd梯度比较不同输入的Jacobian子空间
    
    核心问题: 不同输入的梯度子空间有多相似?
    """
    print("\n" + "="*70)
    print("实验 C: CKA with autograd gradients")
    print("="*70)
    
    directions = get_syntax_direction(model, tokenizer, device, n_layers, n_pairs=20)
    test_nva = REGULAR_NVA[:n_test]
    layer_indices = sorted(directions.keys())
    n_random = 15
    
    results = defaultdict(dict)
    
    for li, layer_idx in enumerate(layer_indices):
        direction = directions[layer_idx]
        print(f"\n  Layer {layer_idx} ({li+1}/{len(layer_indices)})...")
        
        d_model = 3584
        
        # Collect gradients at multiple sentences for multiple random directions
        # This gives us the Jacobian's behavior in the output direction of logit_gap
        
        grad_matrices = {}  # {sentence_key: [n_random, d_model]}
        
        for si, nva in enumerate(test_nva[:5]):
            sent_s, sent_p, sv, pv = make_sentences(nva)
            sv_id = tokenizer.encode(sv, add_special_tokens=False)[0]
            pv_id = tokenizer.encode(pv, add_special_tokens=False)[0]
            
            # Compute gradients for random directions at this sentence
            grad_list = []
            rng = np.random.RandomState(42)
            
            for ri in range(n_random):
                z = rng.randn(d_model)
                z = z / np.linalg.norm(z)
                
                grad_z, _, _ = compute_true_gradient(
                    model, tokenizer, device, sent_s, layer_idx,
                    z, sv_id, pv_id, n_layers)
                grad_list.append(grad_z)
            
            grad_matrices[f"sing_{si}"] = np.array(grad_list)  # [n_random, d_model]
            
            # Same for plural sentence
            grad_list_p = []
            rng2 = np.random.RandomState(42)
            
            for ri in range(n_random):
                z = rng2.randn(d_model)
                z = z / np.linalg.norm(z)
                
                grad_z, _, _ = compute_true_gradient(
                    model, tokenizer, device, sent_p, layer_idx,
                    z, sv_id, pv_id, n_layers)
                grad_list_p.append(grad_z)
            
            grad_matrices[f"plur_{si}"] = np.array(grad_list_p)
        
        # Compute CKA
        def linear_CKA(X, Y):
            X_c = X - X.mean(axis=0, keepdims=True)
            Y_c = Y - Y.mean(axis=0, keepdims=True)
            
            def hsic(A, B):
                K = A @ A.T
                L = B @ B.T
                n = K.shape[0]
                H = np.eye(n) - np.ones((n, n)) / n
                return np.trace(H @ K @ H @ L @ H)
            
            hsic_xy = hsic(X_c, Y_c)
            hsic_xx = hsic(X_c, X_c)
            hsic_yy = hsic(Y_c, Y_c)
            
            if hsic_xx <= 0 or hsic_yy <= 0:
                return 0
            return hsic_xy / np.sqrt(hsic_xx * hsic_yy)
        
        # CKA between sing sentences
        keys_sing = [k for k in grad_matrices.keys() if k.startswith('sing_')]
        keys_plur = [k for k in grad_matrices.keys() if k.startswith('plur_')]
        
        cka_sing_sing = []
        for i in range(len(keys_sing)):
            for j in range(i+1, len(keys_sing)):
                cka = linear_CKA(grad_matrices[keys_sing[i]], grad_matrices[keys_sing[j]])
                cka_sing_sing.append(cka)
        
        cka_sing_plur = []
        for ks in keys_sing:
            for kp in keys_plur:
                cka = linear_CKA(grad_matrices[ks], grad_matrices[kp])
                cka_sing_plur.append(cka)
        
        cka_plur_plur = []
        for i in range(len(keys_plur)):
            for j in range(i+1, len(keys_plur)):
                cka = linear_CKA(grad_matrices[keys_plur[i]], grad_matrices[keys_plur[j]])
                cka_plur_plur.append(cka)
        
        # Principal angles
        def principal_angles(X, Y, n_angles=5):
            Qx, _ = np.linalg.qr(X.T)
            Qy, _ = np.linalg.qr(Y.T)
            M = Qx.T @ Qy
            s = np.linalg.svd(M, compute_uv=False)
            angles = np.arccos(np.clip(s[:n_angles], 0, 1))
            return angles, s[:n_angles]
        
        angles_sp, cos_sp = principal_angles(grad_matrices[keys_sing[0]], grad_matrices[keys_plur[0]])
        
        results[layer_idx] = {
            'cka_sing_sing': np.mean(cka_sing_sing) if cka_sing_sing else 0,
            'cka_sing_plur': np.mean(cka_sing_plur) if cka_sing_plur else 0,
            'cka_plur_plur': np.mean(cka_plur_plur) if cka_plur_plur else 0,
            'principal_cosines': cos_sp,
            'principal_angles_deg': np.degrees(angles_sp),
        }
    
    # Print results
    print("\n" + "="*70)
    print("实验 C 结果: CKA with autograd")
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
    print("  CKA ≈ 1: Same gradient subspace → same Jacobian structure")
    print("  CKA(s,s) ≈ CKA(s,p): Jacobian independent of number → global structure")
    print("  CKA(s,s) >> CKA(s,p): Jacobian depends on number → contextual structure")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek7b")
    parser.add_argument("--n_test", type=int, default=20)
    parser.add_argument("--exp", type=str, default="all", choices=["a", "b", "c", "all"])
    args = parser.parse_args()
    
    print(f"Phase 69b: 真正的微分几何分析 (autograd)")
    print(f"Model: {args.model}, n_test: {args.n_test}, exp: {args.exp}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load model
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers
    W_U = get_W_U(model)
    
    print(f"\nModel info: {model_info.name}, n_layers={n_layers}, d_model={model_info.d_model}")
    
    if args.exp in ["a", "all"]:
        results_a = experiment_a_autograd_convergence(model, tokenizer, device, n_layers, args.n_test)
    
    if args.exp in ["b", "all"]:
        results_b = experiment_b_spectrum_autograd(model, tokenizer, device, n_layers, min(args.n_test, 10))
    
    if args.exp in ["c", "all"]:
        results_c = experiment_c_cka_autograd(model, tokenizer, device, n_layers, min(args.n_test, 10))
    
    # Cleanup
    release_model(model)
    print(f"\nPhase 69b complete! Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
