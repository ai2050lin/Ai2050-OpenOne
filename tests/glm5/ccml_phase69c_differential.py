"""
Phase 69c: 真正的微分几何分析 — 中心差分 + 收敛验证 + CKA
==========================================================

Phase 69b OOM — autograd全模型反向传播内存不够

Phase 69c的改进:
1. ★ 中心差分: Jv ≈ (F(x+εv) - F(x-εv)) / (2ε) — O(ε²)精度
2. ★ 收敛验证: 检查Jv在ε ∈ [0.01, 0.02, 0.05, 0.1, 0.2]是否收敛
3. ★ Float32输出: 提高数值精度
4. ★ CKA: 用有限差分Jv矩阵做子空间对齐分析
5. ★ Spectrum: 用随机方向Jv估计Jacobian结构
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


def compute_Jv_centered(model, tokenizer, device, sentence, layer_idx,
                         direction, epsilon, n_layers=28, target_ids=None):
    """
    中心差分计算: Jv ≈ (F(x+εv) - F(x-εv)) / (2ε)
    精度 O(ε²), 比前向差分 O(ε) 更精确
    
    返回: (Jv, base_logits) 
    """
    toks = tokenizer(sentence, return_tensors="pt").to(device)
    direction_t = torch.tensor(direction, dtype=torch.float32, device=device)
    layers = get_layers(model)
    
    # F(x + εv)
    def plus_hook(module, input, output):
        if isinstance(output, tuple):
            out_tensor = output[0].detach().clone().float()
        else:
            out_tensor = output.detach().clone().float()
        out_tensor[0, -1, :] += epsilon * direction_t
        if isinstance(output, tuple):
            return (out_tensor.to(output[0].dtype),) + output[1:]
        return out_tensor.to(output[0].dtype) if isinstance(output, torch.Tensor) else out_tensor
    
    hook1 = layers[layer_idx].register_forward_hook(plus_hook)
    with torch.no_grad():
        out_plus = model(**toks)
        logits_plus = out_plus.logits[0, -1, :].detach().float().cpu().numpy()
    hook1.remove()
    
    # F(x - εv)
    def minus_hook(module, input, output):
        if isinstance(output, tuple):
            out_tensor = output[0].detach().clone().float()
        else:
            out_tensor = output.detach().clone().float()
        out_tensor[0, -1, :] -= epsilon * direction_t
        if isinstance(output, tuple):
            return (out_tensor.to(output[0].dtype),) + output[1:]
        return out_tensor.to(output[0].dtype) if isinstance(output, torch.Tensor) else out_tensor
    
    hook2 = layers[layer_idx].register_forward_hook(minus_hook)
    with torch.no_grad():
        out_minus = model(**toks)
        logits_minus = out_minus.logits[0, -1, :].detach().float().cpu().numpy()
    hook2.remove()
    
    # Centered difference
    Jv = (logits_plus - logits_minus) / (2 * epsilon)
    base_logits = (logits_plus + logits_minus) / 2
    
    return Jv, base_logits


def experiment_a_convergence(model, tokenizer, device, n_layers, n_test=25):
    """
    实验 A: 中心差分收敛性验证
    
    核心问题: Jv是否随ε→0收敛?
    如果Jv(ε₁) ≈ Jv(ε₂) for ε₁,ε₂ ∈ [0.01, 0.1] → Jacobian存在
    如果Jv(ε)剧烈变化 → 语法方向在非线性区域
    """
    print("\n" + "="*70)
    print("实验 A: 中心差分收敛性 (Jv是否随ε→0收敛?)")
    print("="*70)
    
    directions = get_syntax_direction(model, tokenizer, device, n_layers, n_pairs=30)
    epsilons = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
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
                Jv_by_eps = {}
                for eps in epsilons:
                    Jv, base = compute_Jv_centered(
                        model, tokenizer, device, sentence, layer_idx,
                        direction, eps, n_layers)
                    
                    gap = Jv[pv_id] - Jv[sv_id]
                    norm = np.linalg.norm(Jv)
                    
                    Jv_by_eps[eps] = {'Jv': Jv, 'gap': gap, 'norm': norm}
                
                # Compute convergence metrics
                # Ratio: Jv_gap at ε / Jv_gap at ε_ref (ε=0.01)
                ref_gap = Jv_by_eps[0.01]['gap']
                
                for eps in epsilons:
                    ratio = Jv_by_eps[eps]['gap'] / ref_gap if abs(ref_gap) > 1e-10 else 0
                    
                    # Cosine similarity with reference
                    Jv_ref = Jv_by_eps[0.01]['Jv']
                    norm_ref = Jv_by_eps[0.01]['norm']
                    norm_curr = Jv_by_eps[eps]['norm']
                    cos = np.dot(Jv_by_eps[eps]['Jv'], Jv_ref) / max(norm_ref * norm_curr, 1e-10) if norm_ref > 1e-10 and norm_curr > 1e-10 else 0
                    
                    results[layer_idx][eps].append({
                        'ratio': ratio,
                        'cos_with_ref': cos,
                        'gap': Jv_by_eps[eps]['gap'],
                        'norm': norm_curr,
                    })
            except Exception as e:
                continue
    
    # Print results
    print("\n" + "="*70)
    print("实验 A 结果: 中心差分收敛性")
    print("="*70)
    
    # Ratio: Jv_gap(ε) / Jv_gap(ε=0.01)
    print("\nMean ratio Jv_gap(ε) / Jv_gap(0.01) — should → 1 if Jacobian exists:")
    print(f"{'Layer':<8}", end="")
    for eps in epsilons:
        print(f"ε={eps:<8}", end="")
    print()
    
    for layer_idx in layer_indices:
        print(f"L{layer_idx:<6}", end="")
        for eps in epsilons:
            if eps in results[layer_idx]:
                ratios = [r['ratio'] for r in results[layer_idx][eps] if abs(r['ratio']) < 50]
                if ratios:
                    print(f"{np.mean(ratios):<10.4f}", end="")
                else:
                    print(f"{'N/A':<10}", end="")
            else:
                print(f"{'N/A':<10}", end="")
        print()
    
    # Cosine: Jv(ε) vs Jv(0.01)
    print("\nMean cos(Jv(ε), Jv(0.01)) — should → 1 if Jacobian direction stable:")
    print(f"{'Layer':<8}", end="")
    for eps in epsilons:
        print(f"ε={eps:<8}", end="")
    print()
    
    for layer_idx in layer_indices:
        print(f"L{layer_idx:<6}", end="")
        for eps in epsilons:
            if eps in results[layer_idx]:
                cos_vals = [r['cos_with_ref'] for r in results[layer_idx][eps]]
                if cos_vals:
                    print(f"{np.mean(cos_vals):<10.4f}", end="")
                else:
                    print(f"{'N/A':<10}", end="")
            else:
                print(f"{'N/A':<10}", end="")
        print()
    
    # Jv_gap values (the actual directional derivative for logit gap)
    print("\nMean Jv_gap (∂(logit_plur-logit_sing)/∂h · v):")
    print(f"{'Layer':<8}", end="")
    for eps in epsilons:
        print(f"ε={eps:<8}", end="")
    print()
    
    for layer_idx in layer_indices:
        print(f"L{layer_idx:<6}", end="")
        for eps in epsilons:
            if eps in results[layer_idx]:
                gaps = [r['gap'] for r in results[layer_idx][eps]]
                if gaps:
                    print(f"{np.mean(gaps):<10.4f}", end="")
                else:
                    print(f"{'N/A':<10}", end="")
            else:
                print(f"{'N/A':<10}", end="")
        print()
    
    # Key analysis
    print("\n★★★ Key findings:")
    for layer_idx in layer_indices:
        # Check if ratio is stable across ε
        ratios_by_eps = {}
        for eps in [0.01, 0.02, 0.05, 0.1]:
            if eps in results[layer_idx]:
                ratios = [r['ratio'] for r in results[layer_idx][eps] if abs(r['ratio']) < 50]
                if ratios:
                    ratios_by_eps[eps] = np.mean(ratios)
        
        if len(ratios_by_eps) >= 3:
            vals = list(ratios_by_eps.values())
            max_ratio = max(vals)
            min_ratio = min(vals)
            # If all ratios are close to 1, Jacobian exists
            if max(abs(v - 1) for v in vals) < 0.3:
                print(f"  L{layer_idx}: ✅ JACOBIAN EXISTS — ratio stable near 1 across ε")
            elif max_ratio / max(min_ratio, 0.01) < 3:
                print(f"  L{layer_idx}: ⚠️ PARTIALLY LINEAR — ratio varies but same order")
            else:
                print(f"  L{layer_idx}: ❌ HIGHLY NONLINEAR — ratio varies dramatically")
        else:
            # Check if Jv_gap ≈ 0 at all ε (no effect)
            gaps = []
            for eps in [0.01, 0.1]:
                if eps in results[layer_idx]:
                    gaps.extend([r['gap'] for r in results[layer_idx][eps]])
            if gaps and all(abs(g) < 0.01 for g in gaps):
                print(f"  L{layer_idx}: ~0 effect — direction has no impact on logit_gap")
            else:
                print(f"  L{layer_idx}: ⚠️ Insufficient data")
    
    return results


def experiment_b_spectrum_cka(model, tokenizer, device, n_layers, n_test=10):
    """
    实验 B: Jacobian Spectrum + CKA
    
    方法:
    1. 对每个输入, 计算 k 个随机方向的 Jv (中心差分)
    2. SVD of Jv matrix → spectrum
    3. CKA between different inputs
    """
    print("\n" + "="*70)
    print("实验 B: Jacobian Spectrum + CKA (centered difference)")
    print("="*70)
    
    directions = get_syntax_direction(model, tokenizer, device, n_layers, n_pairs=20)
    epsilon = 0.05  # Moderate ε for stability
    test_nva = REGULAR_NVA[:n_test]
    layer_indices = sorted(directions.keys())
    n_random = 15
    d_model = 3584
    
    results = defaultdict(dict)
    
    for li, layer_idx in enumerate(layer_indices):
        direction = directions[layer_idx]
        print(f"\n  Layer {layer_idx} ({li+1}/{len(layer_indices)})...")
        
        # Collect Jv matrices for multiple sentences
        Jv_by_sentence = {}
        
        for si, nva in enumerate(test_nva[:5]):
            sent_s, sent_p, sv, pv = make_sentences(nva)
            sv_id = tokenizer.encode(sv, add_special_tokens=False)[0]
            pv_id = tokenizer.encode(pv, add_special_tokens=False)[0]
            
            # Compute Jv for syntax direction + random directions
            rng = np.random.RandomState(42 + si)
            
            Jv_list = []
            # First: syntax direction
            Jv_syntax, base = compute_Jv_centered(
                model, tokenizer, device, sent_s, layer_idx,
                direction, epsilon, n_layers)
            Jv_list.append(Jv_syntax[:200])  # Use top-200 tokens for efficiency
            
            # Then random directions
            for ri in range(n_random):
                z = rng.randn(d_model)
                z = z / np.linalg.norm(z)
                
                Jv_z, _ = compute_Jv_centered(
                    model, tokenizer, device, sent_s, layer_idx,
                    z, epsilon, n_layers)
                Jv_list.append(Jv_z[:200])
            
            Jv_by_sentence[f"sing_{si}"] = np.array(Jv_list)  # [n_random+1, 200]
            
            # Same for plural
            rng2 = np.random.RandomState(42 + si)
            Jv_list_p = []
            
            Jv_syntax_p, _ = compute_Jv_centered(
                model, tokenizer, device, sent_p, layer_idx,
                direction, epsilon, n_layers)
            Jv_list_p.append(Jv_syntax_p[:200])
            
            for ri in range(n_random):
                z = rng2.randn(d_model)
                z = z / np.linalg.norm(z)
                
                Jv_z, _ = compute_Jv_centered(
                    model, tokenizer, device, sent_p, layer_idx,
                    z, epsilon, n_layers)
                Jv_list_p.append(Jv_z[:200])
            
            Jv_by_sentence[f"plur_{si}"] = np.array(Jv_list_p)
        
        # SVD of each Jv matrix
        for key in list(Jv_by_sentence.keys()):
            Jv_mat = Jv_by_sentence[key]
            U, S, Vt = np.linalg.svd(Jv_mat, full_matrices=False)
            total_energy = np.sum(S**2)
            effective_rank = total_energy**2 / np.sum(S**4) if np.sum(S**4) > 0 else 0
            Jv_by_sentence[key + '_svd'] = {'S': S, 'eff_rank': effective_rank}
        
        # CKA
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
        
        keys_sing = [k for k in Jv_by_sentence.keys() if k.startswith('sing_') and not k.endswith('_svd')]
        keys_plur = [k for k in Jv_by_sentence.keys() if k.startswith('plur_') and not k.endswith('_svd')]
        
        cka_sing_sing = [linear_CKA(Jv_by_sentence[keys_sing[i]], Jv_by_sentence[keys_sing[j]])
                        for i in range(len(keys_sing)) for j in range(i+1, len(keys_sing))]
        
        cka_sing_plur = [linear_CKA(Jv_by_sentence[ks], Jv_by_sentence[kp])
                        for ks in keys_sing for kp in keys_plur]
        
        cka_plur_plur = [linear_CKA(Jv_by_sentence[keys_plur[i]], Jv_by_sentence[keys_plur[j]])
                        for i in range(len(keys_plur)) for j in range(i+1, len(keys_plur))]
        
        # Mean effective rank
        eff_ranks = [Jv_by_sentence[k + '_svd']['eff_rank'] for k in keys_sing + keys_plur]
        
        # Mean singular values
        all_S = [Jv_by_sentence[k + '_svd']['S'] for k in keys_sing]
        mean_S = np.mean(all_S, axis=0) if all_S else np.zeros(n_random+1)
        
        results[layer_idx] = {
            'cka_sing_sing': np.mean(cka_sing_sing) if cka_sing_sing else 0,
            'cka_sing_plur': np.mean(cka_sing_plur) if cka_sing_plur else 0,
            'cka_plur_plur': np.mean(cka_plur_plur) if cka_plur_plur else 0,
            'effective_rank': np.mean(eff_ranks) if eff_ranks else 0,
            'mean_spectrum': mean_S[:10],
        }
    
    # Print results
    print("\n" + "="*70)
    print("实验 B 结果: Spectrum + CKA")
    print("="*70)
    
    print(f"\n{'Layer':<8} {'CKA(s,s)':<12} {'CKA(s,p)':<12} {'CKA(p,p)':<12} {'Eff_Rank':<12} {'Spectrum[0:5]'}")
    for layer_idx in layer_indices:
        r = results[layer_idx]
        spec = r['mean_spectrum'][:5]
        spec_str = f"{spec[0]:.1f}, {spec[1]:.1f}, {spec[2]:.1f}" if len(spec) >= 3 else "N/A"
        print(f"L{layer_idx:<6} {r['cka_sing_sing']:<12.4f} {r['cka_sing_plur']:<12.4f} "
              f"{r['cka_plur_plur']:<12.4f} {r['effective_rank']:<12.2f} [{spec_str}]")
    
    print("\n★★★ Interpretation:")
    print("  CKA(s,s) ≈ 1: Same tangent space for different singular sentences")
    print("  CKA(s,p) ≈ CKA(s,s): Number doesn't change tangent space")
    print("  CKA(s,p) << CKA(s,s): Number changes tangent space → contextual geometry")
    print("  Eff_Rank >> 1: Jacobian acts on many dimensions")
    print("  Eff_Rank ≈ 1: Jacobian concentrates on few directions")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek7b")
    parser.add_argument("--n_test", type=int, default=20)
    parser.add_argument("--exp", type=str, default="all", choices=["a", "b", "all"])
    args = parser.parse_args()
    
    print(f"Phase 69c: 真正的微分几何分析 (centered difference)")
    print(f"Model: {args.model}, n_test: {args.n_test}, exp: {args.exp}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers
    W_U = get_W_U(model)
    
    print(f"\nModel info: {model_info.name}, n_layers={n_layers}, d_model={model_info.d_model}")
    
    if args.exp in ["a", "all"]:
        results_a = experiment_a_convergence(model, tokenizer, device, n_layers, args.n_test)
    
    if args.exp in ["b", "all"]:
        results_b = experiment_b_spectrum_cka(model, tokenizer, device, n_layers, min(args.n_test, 10))
    
    release_model(model)
    print(f"\nPhase 69c complete! Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
