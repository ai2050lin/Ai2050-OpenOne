"""
Phase 68: 局部线性化流形系统 (Local Linear Manifold) 验证
=========================================================

Phase 67的关键问题:
1. ❌ 用离散flip rate代替连续指标 → 丢失了信号强度信息
2. ❌ L0 LN"放大5x"是伪象 → 低方差空间的几何拉伸≠信息放大
3. ❌ 仍在用欧氏几何(方向/旋转/压缩)理解非线性系统

核心框架转变:
  从 "全局方向控制" → "局部条件方向控制"
  从 "欧氏几何" → "Jacobian/局部流形/条件线性化"

  h_{l+1} = F_l(h_l)
  局部: F_l(h+δ) ≈ F_l(h) + J_l(h)δ
  J_l(h) 依赖输入 → 不存在全局语法方向

三个核心实验:
  A. 局部线性有效半径 — cos(F(x+βv)-F(x), β·J(x)v) 随β如何变化
  B. Jacobian一致性 — cos(J(x₁)v, J(x₂)v) 跨样本是否一致
  C. Subspace_swap是否改变Jacobian — ||J(x)v - J(x_swap)v||

关键改进: 用连续指标(Δlogit, KL散度)替代离散flip rate
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


def compute_directional_derivative(model, tokenizer, device, sentence, layer_idx, 
                                    direction, epsilon=0.1, n_layers=28):
    """
    计算方向导数 J(logits)·v at layer l
    
    返回: logit空间中的方向导数向量 [vocab_size]
    """
    toks = tokenizer(sentence, return_tensors="pt").to(device)
    direction_t = torch.tensor(direction, dtype=torch.float32, device=device)
    
    # Baseline forward pass
    baseline_logits = None
    captured = {}
    layers = get_layers(model)
    
    def make_hook(key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().clone()
            else:
                captured[key] = output.detach().clone()
        return hook
    
    hook = layers[layer_idx].register_forward_hook(make_hook(f"L{layer_idx}"))
    
    with torch.no_grad():
        out = model(**toks)
        baseline_logits = out.logits[0, -1, :].detach().float().cpu().numpy()
    
    h_baseline = captured[f"L{layer_idx}"].detach().clone()
    hook.remove()
    
    # Perturbed forward pass: h_l → h_l + εv at last token position
    perturbed_logits = None
    captured2 = {}
    
    def perturb_hook(module, input, output):
        if isinstance(output, tuple):
            out_tensor = output[0].detach().clone()
        else:
            out_tensor = output.detach().clone()
        # Add εv to last token position
        out_tensor[0, -1, :] += epsilon * direction_t.to(out_tensor.dtype)
        if isinstance(output, tuple):
            return (out_tensor,) + output[1:]
        return out_tensor
    
    hook2 = layers[layer_idx].register_forward_hook(perturb_hook)
    
    with torch.no_grad():
        out2 = model(**toks)
        perturbed_logits = out2.logits[0, -1, :].detach().float().cpu().numpy()
    
    hook2.remove()
    
    # Directional derivative: (F(x+εv) - F(x)) / ε
    Jv = (perturbed_logits - baseline_logits) / epsilon
    
    return Jv, baseline_logits


def compute_effect_at_beta(model, tokenizer, device, sentence, layer_idx,
                           direction, beta, n_layers=28):
    """
    计算在层l注入βv后的logit变化
    
    返回: (perturbed_logits, baseline_logits)
    """
    toks = tokenizer(sentence, return_tensors="pt").to(device)
    direction_t = torch.tensor(direction, dtype=torch.float32, device=device)
    
    # Baseline
    layers = get_layers(model)
    with torch.no_grad():
        out_base = model(**toks)
        baseline_logits = out_base.logits[0, -1, :].detach().float().cpu().numpy()
    
    # Perturbed
    def perturb_hook(module, input, output):
        if isinstance(output, tuple):
            out_tensor = output[0].detach().clone()
        else:
            out_tensor = output.detach().clone()
        out_tensor[0, -1, :] += beta * direction_t.to(out_tensor.dtype)
        if isinstance(output, tuple):
            return (out_tensor,) + output[1:]
        return out_tensor
    
    hook = layers[layer_idx].register_forward_hook(perturb_hook)
    
    with torch.no_grad():
        out_pert = model(**toks)
        perturbed_logits = out_pert.logits[0, -1, :].detach().float().cpu().numpy()
    
    hook.remove()
    
    return perturbed_logits, baseline_logits


def experiment_a_effective_radius(model, tokenizer, device, n_layers, W_U, n_test=40):
    """
    实验 A: 局部线性有效半径
    
    核心问题: 对于多大的β，线性近似 J·v 仍然有效?
    
    方法:
    1. 用ε=0.1计算方向导数 Jv (线性参考)
    2. 用β∈[0.5, 1, 2, 4, 8, 16]计算实际效果
    3. 比较实际效果 vs 线性预测
    
    用连续指标:
    - Δlogit_gap = Δ(logit_plur - logit_sing) 
    - ratio = actual_Δlogit_gap / linear_Δlogit_gap
    - ratio→0: 线性近似失效 (非线性区域)
    - ratio≈1: 线性近似有效
    """
    print("\n" + "="*70)
    print("实验 A: 局部线性有效半径")
    print("="*70)
    
    directions = get_syntax_direction(model, tokenizer, device, n_layers, W_U, n_pairs=30)
    betas = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    epsilon = 0.1
    test_nva = REGULAR_NVA[:n_test]
    layer_indices = sorted(directions.keys())
    
    results = defaultdict(lambda: defaultdict(list))
    
    for li, layer_idx in enumerate(layer_indices):
        direction = directions[layer_idx]
        print(f"\n  Layer {layer_idx} ({li+1}/{len(layer_indices)})...")
        
        for nva in test_nva:
            sent_s, sent_p, sv, pv = make_sentences(nva)
            # 用单数句, 测试"推向复数"的效果
            sentence = sent_s
            
            # Get token IDs for verbs
            sv_id = tokenizer.encode(sv, add_special_tokens=False)[0]
            pv_id = tokenizer.encode(pv, add_special_tokens=False)[0]
            
            # Step 1: Compute directional derivative (linear reference)
            try:
                Jv, base_logits = compute_directional_derivative(
                    model, tokenizer, device, sentence, layer_idx, direction, epsilon, n_layers)
                
                # Linear prediction of Δlogit_gap
                linear_dlogit_gap = Jv[pv_id] - Jv[sv_id]  # per unit β
                
                # Step 2: Compute actual effects at different β
                for beta in betas:
                    pert_logits, _ = compute_effect_at_beta(
                        model, tokenizer, device, sentence, layer_idx, direction, beta, n_layers)
                    
                    actual_dlogit_gap = (pert_logits[pv_id] - pert_logits[sv_id]) - (base_logits[pv_id] - base_logits[sv_id])
                    linear_pred_dlogit_gap = beta * linear_dlogit_gap
                    
                    # Ratio
                    if abs(linear_pred_dlogit_gap) > 1e-6:
                        ratio = actual_dlogit_gap / linear_pred_dlogit_gap
                    else:
                        ratio = 0.0
                    
                    # Also compute |Δlogit_gap| as continuous metric
                    abs_effect = abs(actual_dlogit_gap)
                    
                    results[layer_idx][beta].append({
                        'ratio': ratio,
                        'abs_effect': abs_effect,
                        'actual_gap': actual_dlogit_gap,
                        'linear_gap': linear_pred_dlogit_gap,
                    })
            except Exception as e:
                print(f"    Error with {nva}: {e}")
                continue
    
    # Print results
    print("\n" + "="*70)
    print("实验 A 结果: 局部线性有效半径")
    print("="*70)
    
    print("\nMean ratio (actual/linear) — ratio=1 means linear, ratio→0 means nonlinear:")
    print(f"{'Layer':<8}", end="")
    for beta in betas:
        print(f"β={beta:<6}", end="")
    print()
    
    for layer_idx in layer_indices:
        print(f"L{layer_idx:<6}", end="")
        for beta in betas:
            if beta in results[layer_idx]:
                ratios = [r['ratio'] for r in results[layer_idx][beta]]
                mean_ratio = np.mean(ratios)
                print(f"{mean_ratio:<8.3f}", end="")
            else:
                print(f"{'N/A':<8}", end="")
        print()
    
    print("\nMean |Δlogit_gap| (continuous metric):")
    print(f"{'Layer':<8}", end="")
    for beta in betas:
        print(f"β={beta:<6}", end="")
    print()
    
    for layer_idx in layer_indices:
        print(f"L{layer_idx:<6}", end="")
        for beta in betas:
            if beta in results[layer_idx]:
                effects = [r['abs_effect'] for r in results[layer_idx][beta]]
                print(f"{np.mean(effects):<8.3f}", end="")
            else:
                print(f"{'N/A':<8}", end="")
        print()
    
    print("\nLinear prediction |Δlogit_gap| (β × Jv):")
    print(f"{'Layer':<8}", end="")
    for beta in betas:
        print(f"β={beta:<6}", end="")
    print()
    
    for layer_idx in layer_indices:
        print(f"L{layer_idx:<6}", end="")
        for beta in betas:
            if beta in results[layer_idx]:
                linear_gaps = [abs(r['linear_gap']) for r in results[layer_idx][beta]]
                print(f"{np.mean(linear_gaps):<8.3f}", end="")
            else:
                print(f"{'N/A':<8}", end="")
        print()
    
    # Key analysis: at what β does the ratio drop below 0.5?
    print("\n★★★ Key finding: β at which ratio < 0.5 (linear approximation breaks down):")
    for layer_idx in layer_indices:
        threshold_beta = None
        for beta in betas:
            if beta in results[layer_idx]:
                ratios = [r['ratio'] for r in results[layer_idx][beta]]
                mean_ratio = np.mean(ratios)
                if mean_ratio < 0.5 and threshold_beta is None:
                    threshold_beta = beta
        if threshold_beta:
            print(f"  L{layer_idx}: linear approximation fails at β ≈ {threshold_beta}")
        else:
            print(f"  L{layer_idx}: linear approximation holds for all tested β")
    
    return results


def experiment_b_jacobian_consistency(model, tokenizer, device, n_layers, W_U, n_test=40):
    """
    实验 B: Jacobian一致性
    
    核心问题: 不同输入x₁, x₂的Jacobian是否一致?
    如果cos(J(x₁)v, J(x₂)v)很低 → 不存在全局方向
    
    方法:
    1. 对每个句子对(x₁, x₂), 计算J(x₁)·v和J(x₂)·v
    2. 计算cosine similarity
    3. 同时用Δlogit_gap作为标量指标, 检查相关性
    
    连续指标:
    - cos(J₁v, J₂v): 方向一致性
    - corr(Δlogit_gap_1, Δlogit_gap_2): 效果相关性
    """
    print("\n" + "="*70)
    print("实验 B: Jacobian一致性 (是否存在全局语法方向?)")
    print("="*70)
    
    directions = get_syntax_direction(model, tokenizer, device, n_layers, W_U, n_pairs=30)
    epsilon = 0.1
    test_nva = REGULAR_NVA[:n_test]
    layer_indices = sorted(directions.keys())
    
    results = defaultdict(list)
    
    for li, layer_idx in enumerate(layer_indices):
        direction = directions[layer_idx]
        print(f"\n  Layer {layer_idx} ({li+1}/{len(layer_indices)})...")
        
        Jv_list = []  # Store Jv for each sentence
        gap_list = []  # Store Δlogit_gap for each sentence
        
        for nva in test_nva:
            sent_s, sent_p, sv, pv = make_sentences(nva)
            sentence = sent_s
            
            sv_id = tokenizer.encode(sv, add_special_tokens=False)[0]
            pv_id = tokenizer.encode(pv, add_special_tokens=False)[0]
            
            try:
                Jv, base_logits = compute_directional_derivative(
                    model, tokenizer, device, sentence, layer_idx, direction, epsilon, n_layers)
                
                # Δlogit_gap from Jv
                dlogit_gap = Jv[pv_id] - Jv[sv_id]
                
                Jv_list.append(Jv)
                gap_list.append(dlogit_gap)
            except Exception as e:
                continue
        
        # Compute pairwise cosine similarities
        cos_sims = []
        for i in range(min(len(Jv_list), 20)):  # Sample 20 sentences
            for j in range(i+1, min(len(Jv_list), 20)):
                cos_sim = np.dot(Jv_list[i], Jv_list[j]) / (
                    max(np.linalg.norm(Jv_list[i]), 1e-10) * max(np.linalg.norm(Jv_list[j]), 1e-10))
                cos_sims.append(cos_sim)
        
        # Correlation of Δlogit_gap across sentences
        if len(gap_list) >= 5:
            # Use different sentences as x₁, x₂ pairs
            # Check if J(x₁)·v and J(x₂)·v point in similar directions
            # By looking at correlation of top-k components
            Jv_arr = np.array(Jv_list[:20])  # [n_sentences, vocab_size]
            
            # Cosine similarity using top-100 token effects
            # (full vocab is too noisy, focus on tokens most affected)
            mean_effect = np.mean(np.abs(Jv_arr), axis=0)
            top_tokens = np.argsort(mean_effect)[-100:]
            
            Jv_top = Jv_arr[:, top_tokens]  # [n_sentences, 100]
            cos_sims_top = []
            for i in range(len(Jv_top)):
                for j in range(i+1, len(Jv_top)):
                    cos = np.dot(Jv_top[i], Jv_top[j]) / (
                        max(np.linalg.norm(Jv_top[i]), 1e-10) * max(np.linalg.norm(Jv_top[j]), 1e-10))
                    cos_sims_top.append(cos)
        else:
            cos_sims_top = [0]
        
        results[layer_idx] = {
            'cos_sim_full': np.mean(cos_sims) if cos_sims else 0,
            'cos_sim_top100': np.mean(cos_sims_top) if cos_sims_top else 0,
            'gap_std': np.std(gap_list) if gap_list else 0,
            'gap_mean': np.mean(gap_list) if gap_list else 0,
            'gap_cv': np.std(gap_list) / max(abs(np.mean(gap_list)), 1e-10) if gap_list else 0,
            'n_sentences': len(Jv_list),
        }
    
    # Print results
    print("\n" + "="*70)
    print("实验 B 结果: Jacobian一致性")
    print("="*70)
    
    print(f"\n{'Layer':<8} {'cos_full':<10} {'cos_top100':<12} {'gap_mean':<10} {'gap_std':<10} {'gap_CV':<10}")
    for layer_idx in layer_indices:
        r = results[layer_idx]
        print(f"L{layer_idx:<6} {r['cos_sim_full']:<10.4f} {r['cos_sim_top100']:<12.4f} "
              f"{r['gap_mean']:<10.4f} {r['gap_std']:<10.4f} {r['gap_cv']:<10.4f}")
    
    print("\n★★★ Interpretation:")
    print("  cos_full ≈ 1: Global direction exists (same J·v for all inputs)")
    print("  cos_full ≈ 0: No global direction (Jacobian is input-dependent)")
    print("  gap_CV >> 1: High variability → no consistent effect across inputs")
    
    return results


def experiment_c_subspace_jacobian(model, tokenizer, device, n_layers, W_U, n_test=30):
    """
    实验 C: Subspace_swap是否改变Jacobian
    
    核心问题: subspace_swap是否改变了局部动力系统?
    
    方法:
    1. 对每个句子, 计算J(x)·v (正常输入)
    2. 做subspace_swap: 替换sing→plur的语法子空间
    3. 计算J(x_swap)·v
    4. 比较: ||J(x)v - J(x_swap)v|| 和 cos(J(x)v, J(x_swap)v)
    
    如果差异很大 → subspace_swap确实改变了局部几何
    如果差异很小 → subspace_swap只是改了信号，没改动力系统
    """
    print("\n" + "="*70)
    print("实验 C: Subspace_swap是否改变Jacobian (局部动力系统)?")
    print("="*70)
    
    directions = get_syntax_direction(model, tokenizer, device, n_layers, W_U, n_pairs=30)
    epsilon = 0.1
    test_nva = REGULAR_NVA[:n_test]
    layer_indices = sorted(directions.keys())
    
    results = defaultdict(list)
    
    for li, layer_idx in enumerate(layer_indices):
        direction = directions[layer_idx]
        print(f"\n  Layer {layer_idx} ({li+1}/{len(layer_indices)})...")
        
        for nva in test_nva:
            sent_s, sent_p, sv, pv = make_sentences(nva)
            
            sv_id = tokenizer.encode(sv, add_special_tokens=False)[0]
            pv_id = tokenizer.encode(pv, add_special_tokens=False)[0]
            
            try:
                # Step 1: Normal J(x)·v for sing sentence
                Jv_sing, base_logits_sing = compute_directional_derivative(
                    model, tokenizer, device, sent_s, layer_idx, direction, epsilon, n_layers)
                gap_sing = Jv_sing[pv_id] - Jv_sing[sv_id]
                
                # Step 2: Normal J(x)·v for plur sentence
                Jv_plur, base_logits_plur = compute_directional_derivative(
                    model, tokenizer, device, sent_p, layer_idx, direction, epsilon, n_layers)
                gap_plur = Jv_plur[pv_id] - Jv_plur[sv_id]
                
                # Compute difference between Jacobians
                # Full logit space
                diff_Jv = Jv_plur - Jv_sing
                norm_sing = np.linalg.norm(Jv_sing)
                norm_plur = np.linalg.norm(Jv_plur)
                norm_diff = np.linalg.norm(diff_Jv)
                
                # Cosine similarity
                if norm_sing > 1e-10 and norm_plur > 1e-10:
                    cos_Jv = np.dot(Jv_sing, Jv_plur) / (norm_sing * norm_plur)
                else:
                    cos_Jv = 0
                
                # Relative change
                rel_change = norm_diff / max(norm_sing, 1e-10)
                
                results[layer_idx].append({
                    'cos_Jv': cos_Jv,
                    'norm_Jv_sing': norm_sing,
                    'norm_Jv_plur': norm_plur,
                    'norm_diff': norm_diff,
                    'rel_change': rel_change,
                    'gap_sing': gap_sing,
                    'gap_plur': gap_plur,
                    'gap_diff': gap_plur - gap_sing,
                })
            except Exception as e:
                continue
    
    # Print results
    print("\n" + "="*70)
    print("实验 C 结果: Subspace_swap改变Jacobian?")
    print("="*70)
    
    print(f"\n{'Layer':<8} {'cos_Jv':<10} {'rel_change':<12} {'|Jv_sing|':<12} {'|Jv_plur|':<12} {'|diff|':<10} {'gap_sing':<10} {'gap_plur':<10}")
    for layer_idx in layer_indices:
        if results[layer_idx]:
            r_agg = {
                'cos_Jv': np.mean([r['cos_Jv'] for r in results[layer_idx]]),
                'rel_change': np.mean([r['rel_change'] for r in results[layer_idx]]),
                'norm_sing': np.mean([r['norm_Jv_sing'] for r in results[layer_idx]]),
                'norm_plur': np.mean([r['norm_Jv_plur'] for r in results[layer_idx]]),
                'norm_diff': np.mean([r['norm_diff'] for r in results[layer_idx]]),
                'gap_sing': np.mean([r['gap_sing'] for r in results[layer_idx]]),
                'gap_plur': np.mean([r['gap_plur'] for r in results[layer_idx]]),
            }
            print(f"L{layer_idx:<6} {r_agg['cos_Jv']:<10.4f} {r_agg['rel_change']:<12.4f} "
                  f"{r_agg['norm_sing']:<12.4f} {r_agg['norm_plur']:<12.4f} "
                  f"{r_agg['norm_diff']:<10.4f} {r_agg['gap_sing']:<10.4f} {r_agg['gap_plur']:<10.4f}")
    
    print("\n★★★ Interpretation:")
    print("  cos_Jv ≈ 1: Same Jacobian for sing/plur → same local dynamics")
    print("  cos_Jv << 1: Different Jacobian → subspace_swap changes local geometry!")
    print("  rel_change >> 0: Significant Jacobian change")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek7b")
    parser.add_argument("--n_test", type=int, default=40)
    parser.add_argument("--exp", type=str, default="all", choices=["a", "b", "c", "all"])
    args = parser.parse_args()
    
    print(f"Phase 68: 局部线性化流形系统验证")
    print(f"Model: {args.model}, n_test: {args.n_test}, exp: {args.exp}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load model
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    n_layers = model_info.n_layers
    W_U = get_W_U(model)
    
    print(f"\nModel info: {model_info.name}, n_layers={n_layers}, d_model={model_info.d_model}")
    
    if args.exp in ["a", "all"]:
        results_a = experiment_a_effective_radius(model, tokenizer, device, n_layers, W_U, args.n_test)
    
    if args.exp in ["b", "all"]:
        results_b = experiment_b_jacobian_consistency(model, tokenizer, device, n_layers, W_U, args.n_test)
    
    if args.exp in ["c", "all"]:
        results_c = experiment_c_subspace_jacobian(model, tokenizer, device, n_layers, W_U, min(args.n_test, 30))
    
    # Cleanup
    release_model(model)
    print(f"\nPhase 68 complete! Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
