"""
Phase 81: Operator Mechanics — From Global Fitting to Local Jacobian
====================================================================

Phase 80 critique identified FATAL flaws:
1. R^2=1.0 with n=30, d=768 is OVERFITTING (n << p, severely underdetermined)
2. Global linear fit A is NOT the operator — LOCAL JACOBIAN J(h) is
3. A matrices "orthogonal" is unreliable without train/test validation
4. MLP IS nonlinear: MLP(x) = W2 * GELU(W1*x + b1) + b2
5. R^2=1.0 only proves "too many parameters", NOT "MLP is linear"

The TRUE operator is the LOCAL JACOBIAN:
  J(h) = d(MLP_out) / d(h_input) = W_out^T @ diag(GELU'(z)) @ W_in^T

where z = W_in^T @ LN(h_mid) + b_in is the pre-GELU activation.

This Jacobian VARIES across data points because GELU'(z) depends on z,
which depends on the input. This is the JACOBIAN FIELD.

Key difference from Phase 80:
  Phase 80: Fitted global A from delta_mlp ≈ A @ h_in + b  (overfitted)
  Phase 81: Compute EXACT local J(h) at each data point  (true operator)

Four Experiments:
  A: Train/Test Split Validation ★★★★★ (MOST CRITICAL)
     - 500 train + 500 test per task
     - Fit A on train, evaluate R^2 on test
     - If R^2_test << R^2_train → "linear operator" was overfitting

  B: Analytical Local Jacobian Field ★★★★★ (THE REAL OPERATOR)
     - Compute J(h) = W_out^T @ diag(GELU'(z)) @ W_in^T for each data point
     - SVD of each J(h) → local operator spectrum
     - Analyze: variation WITHIN task, differences BETWEEN tasks

  C: Jacobian Field Topology ★★★★★
     - How much does J(h) vary across data points?
     - If Jacobians are similar within task → local linearization valid
     - If Jacobians vary → need Jacobian field theory
     - PCA of Jacobian spectrum → "Jacobian manifold"

  D: Spectrum Dynamics & Recursive Rollout ★★★★
     - Jacobian spectrum evolution across layers
     - Product of Jacobians → overall operator sensitivity
     - Conserved modes vs. unstable modes

Key insight:
  The "operator family" concept from Phase 80 may be REAL,
  but it should be detected in the JACOBIAN FIELD structure,
  not in overfitted global A matrices.

Usage:
  python ccml_phase81_operator_mechanics.py --exp a
  python ccml_phase81_operator_mechanics.py --exp b
  python ccml_phase81_operator_mechanics.py --exp c
  python ccml_phase81_operator_mechanics.py --exp d
  python ccml_phase81_operator_mechanics.py --exp all
"""

import torch
import numpy as np
import argparse
import time
from collections import defaultdict
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.decomposition import PCA
from transformer_lens import HookedTransformer


def get_model():
    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        center_unembed=False,
        center_writing_weights=False,
        fold_ln=False,
        device="cpu",
    )
    model.eval()
    return model


# ============================================================
# Task Generation — LARGE sample sizes
# ============================================================

def generate_task_samples(task_name, n=1000, seed=42):
    """Generate n samples for a given task."""
    rng = np.random.RandomState(seed)
    
    if task_name == "addition":
        # a + b = format
        samples = []
        for _ in range(n):
            a = rng.randint(1, 50)
            b = rng.randint(1, 50)
            samples.append(f"{a} + {b} =")
        return samples
    
    elif task_name == "translate_fr":
        templates = [
            "Translate to French: The {adj} {noun} {verb}",
            "Translate to French: A {adj} {noun} {verb}",
            "Translate to French: The {noun} is {adj}",
            "Translate to French: My {noun} is {adj}",
        ]
        adjectives = ["big", "small", "old", "new", "red", "blue", "fast", "slow",
                      "hot", "cold", "dark", "bright", "long", "short", "tall", "wide",
                      "soft", "hard", "loud", "quiet", "sweet", "bitter", "rough", "smooth",
                      "thick", "thin", "heavy", "light", "strong", "weak", "clean", "dirty",
                      "rich", "poor", "happy", "sad", "young", "beautiful", "simple", "complex"]
        nouns = ["cat", "dog", "bird", "fish", "tree", "house", "car", "book",
                "child", "woman", "man", "river", "mountain", "star", "sun", "moon",
                "flower", "stone", "door", "wind", "rain", "fire", "sky", "sea",
                "road", "city", "village", "forest", "field", "lake", "cloud", "night"]
        verbs = ["runs", "walks", "sits", "stands", "jumps", "flies", "swims",
                "sleeps", "eats", "drinks", "reads", "writes", "sings", "dances",
                "falls", "rises", "grows", "shines", "moves", "stops"]
        
        samples = []
        for i in range(n):
            template = templates[i % len(templates)]
            adj = adjectives[rng.randint(0, len(adjectives))]
            noun = nouns[rng.randint(0, len(nouns))]
            verb = verbs[rng.randint(0, len(verbs))]
            samples.append(template.format(adj=adj, noun=noun, verb=verb))
        return samples
    
    elif task_name == "antonym":
        words = [
            "hot", "big", "fast", "happy", "light", "strong", "loud", "rough",
            "wide", "tall", "cold", "small", "slow", "sad", "dark", "weak",
            "quiet", "smooth", "narrow", "short", "bright", "heavy", "soft",
            "hard", "old", "young", "rich", "poor", "thick", "thin",
            "open", "closed", "full", "empty", "dry", "wet", "clean", "dirty",
            "safe", "dangerous", "easy", "difficult", "near", "far", "high", "low",
            "early", "late", "sweet", "bitter", "deep", "shallow", "long", "brief",
            "ugly", "beautiful", "simple", "complex", "cheap", "expensive", "few", "many",
            "love", "hate", "win", "lose", "buy", "sell", "push", "pull",
            "rise", "fall", "create", "destroy", "build", "break", "teach", "learn",
            "lead", "follow", "attack", "defend", "give", "take", "laugh", "cry",
            "awake", "asleep", "alive", "dead", "true", "false", "right", "wrong",
            "good", "bad", "new", "ancient", "smart", "foolish", "brave", "afraid",
            "polite", "rude", "honest", "deceitful", "generous", "greedy", "patient", "hasty",
            "calm", "anxious", "gentle", "fierce", "humble", "proud", "kind", "cruel",
            "joyful", "sorrowful", "peaceful", "violent", "loyal", "treacherous", "wise", "ignorant",
        ]
        # Remove duplicates while preserving order
        seen = set()
        unique_words = []
        for w in words:
            if w not in seen:
                seen.add(w)
                unique_words.append(w)
        
        samples = []
        for i in range(n):
            word = unique_words[i % len(unique_words)]
            # Add slight variation
            templates = [
                f"The opposite of {word} is",
                f"The antonym of {word} is",
                f"{word} is the opposite of",
            ]
            samples.append(templates[i % len(templates)])
        return samples
    
    elif task_name == "capital":
        countries = [
            "France", "Germany", "Japan", "Italy", "Spain", "China", "Brazil", "India",
            "Russia", "Egypt", "UK", "Canada", "Mexico", "Korea", "Turkey", "Norway",
            "Sweden", "Poland", "Greece", "Portugal", "Australia", "Argentina", "Chile",
            "Peru", "Colombia", "Thailand", "Vietnam", "Finland", "Denmark", "Austria",
            "Netherlands", "Belgium", "Switzerland", "Ireland", "Czech Republic", "Romania",
            "Hungary", "Ukraine", "Indonesia", "Philippines", "Malaysia", "Singapore",
            "New Zealand", "South Africa", "Nigeria", "Kenya", "Morocco", "Israel",
            "Saudi Arabia", "Iran", "Iraq", "Pakistan", "Bangladesh", "Sri Lanka",
            "Nepal", "Mongolia", "Cambodia", "Laos", "Myanmar", "Jordan",
            "Lebanon", "Syria", "Cuba", "Jamaica", "Haiti", "Dominican Republic",
            "Venezuela", "Ecuador", "Bolivia", "Paraguay", "Uruguay", "Panama",
            "Costa Rica", "Guatemala", "Honduras", "El Salvador", "Nicaragua",
            "Iceland", "Lithuania", "Latvia", "Estonia", "Slovakia", "Slovenia",
            "Croatia", "Serbia", "Bulgaria", "Albania", "Georgia", "Armenia",
            "Azerbaijan", "Kazakhstan", "Uzbekistan", "Afghanistan", "Tanzania",
            "Uganda", "Ethiopia", "Ghana", "Cameroon", "Senegal", "Tunisia",
            "Algeria", "Libya", "Sudan", "Zimbabwe", "Mozambique", "Madagascar",
        ]
        # Remove duplicates
        seen = set()
        unique_countries = []
        for c in countries:
            if c not in seen:
                seen.add(c)
                unique_countries.append(c)
        
        samples = []
        for i in range(n):
            country = unique_countries[i % len(unique_countries)]
            templates = [
                f"The capital of {country} is",
                f"{country}'s capital is",
                f"What is the capital of {country}? It is",
            ]
            samples.append(templates[i % len(templates)])
        return samples
    
    elif task_name == "continue":
        starts = [
            "The cat sat on", "The dog ran to", "The bird flew up", "The fish swam down",
            "The tree grew very", "The sun was very", "The wind blew the", "The rain fell on",
            "Once upon a time", "In the beginning", "Long ago there", "The story begins",
            "It was a dark", "The morning came", "The evening fell", "The night was",
            "She walked into", "He looked at the", "They went to the", "We found a",
            "The door opened and", "The light turned", "The sound came from", "The water was",
            "In the garden", "On the mountain", "By the river", "Under the tree",
            "After the storm", "Before the dawn", "During the night", "Through the forest",
        ]
        samples = []
        for i in range(n):
            s = starts[i % len(starts)]
            # Add some variation
            if i > len(starts):
                s = "Continue: " + s
            samples.append(s)
        return samples
    
    return []


# ============================================================
# Experiment A: Train/Test Split Validation ★★★★★
# ============================================================

def exp_a_train_test_split(model):
    """
    THE MOST CRITICAL EXPERIMENT.
    
    Phase 80 claimed R^2=1.0 for delta_mlp ≈ A @ h_in + b
    But with n=30, d=768, this is SEVERE OVERFITTING.
    
    This experiment:
    1. Generate 1000 samples per task
    2. Split: 500 train, 500 test
    3. Fit A on train (both OLS and Ridge)
    4. Evaluate R^2 on test
    5. If R^2_test << 1.0 → "linear operator" was overfitting
    """
    print("=" * 70)
    print("Experiment A: Train/Test Split Validation")
    print("THE CRITICAL TEST: Was R^2=1.0 real or overfitting?")
    print("=" * 70)
    
    task_names = ["addition", "translate_fr", "antonym", "capital"]
    n_total = 1000
    n_train = 500
    n_test = 500
    layers = [3, 6, 9]
    
    for layer in layers:
        print(f"\n{'='*60}")
        print(f"  Layer {layer}")
        print(f"{'='*60}")
        
        for task_name in task_names:
            samples = generate_task_samples(task_name, n=n_total)
            train_samples = samples[:n_train]
            test_samples = samples[n_train:]
            
            # Collect (h_in, delta_mlp) pairs
            h_ins_train, delta_mlps_train = [], []
            h_ins_test, delta_mlps_test = [], []
            
            print(f"\n  {task_name}: collecting train data ({n_train} samples)...")
            for text in train_samples:
                tokens = model.to_tokens(text)
                _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
                
                h_in = cache[f'blocks.{layer}.hook_resid_pre'][-1].detach().cpu().numpy()
                h_mid = cache[f'blocks.{layer}.hook_resid_mid'][-1].detach().cpu().numpy()
                h_out = cache[f'blocks.{layer}.hook_resid_post'][-1].detach().cpu().numpy()
                
                delta_mlp = h_out - h_mid
                h_ins_train.append(h_in)
                delta_mlps_train.append(delta_mlp)
            
            print(f"  {task_name}: collecting test data ({n_test} samples)...")
            for text in test_samples:
                tokens = model.to_tokens(text)
                _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
                
                h_in = cache[f'blocks.{layer}.hook_resid_pre'][-1].detach().cpu().numpy()
                h_mid = cache[f'blocks.{layer}.hook_resid_mid'][-1].detach().cpu().numpy()
                h_out = cache[f'blocks.{layer}.hook_resid_post'][-1].detach().cpu().numpy()
                
                delta_mlp = h_out - h_mid
                h_ins_test.append(h_in)
                delta_mlps_test.append(delta_mlp)
            
            h_ins_train = np.array(h_ins_train)
            delta_mlps_train = np.array(delta_mlps_train)
            h_ins_test = np.array(h_ins_test)
            delta_mlps_test = np.array(delta_mlps_test)
            
            # ---- Method 1: OLS (same as Phase 80) ----
            reg_ols = LinearRegression()
            reg_ols.fit(h_ins_train, delta_mlps_train)
            r2_train_ols = reg_ols.score(h_ins_train, delta_mlps_train)
            r2_test_ols = reg_ols.score(h_ins_test, delta_mlps_test)
            
            # ---- Method 2: Ridge with different alphas ----
            ridge_results = {}
            for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
                reg_ridge = Ridge(alpha=alpha)
                reg_ridge.fit(h_ins_train, delta_mlps_train)
                r2_train_r = reg_ridge.score(h_ins_train, delta_mlps_train)
                r2_test_r = reg_ridge.score(h_ins_test, delta_mlps_test)
                ridge_results[alpha] = (r2_train_r, r2_test_r)
            
            # ---- Method 3: Random baseline ----
            # What R^2 do we get with random A?
            # Shuffle the delta_mlps_test to get baseline
            rng = np.random.RandomState(42)
            delta_shuffled = rng.permutation(delta_mlps_test)
            ss_res_shuffled = np.sum((delta_mlps_test - delta_shuffled) ** 2)
            ss_tot = np.sum((delta_mlps_test - delta_mlps_test.mean(axis=0)) ** 2)
            r2_random = 1 - ss_res_shuffled / (ss_tot + 1e-12)
            
            # ---- Report ----
            print(f"\n    {task_name} (n_train={n_train}, n_test={n_test}, d={h_ins_train.shape[1]}):")
            print(f"    OLS:   R^2_train = {r2_train_ols:.6f},  R^2_test = {r2_test_ols:.6f}")
            print(f"    Ridge results:")
            for alpha, (r2_tr, r2_te) in ridge_results.items():
                gap = r2_tr - r2_te
                print(f"      alpha={alpha:>6.2f}: R^2_train={r2_tr:.6f}, R^2_test={r2_te:.6f}, gap={gap:.6f}")
            print(f"    Random baseline: R^2 = {r2_random:.6f}")
            
            # ---- Verdict ----
            if r2_test_ols < 0.5:
                verdict = "OVERFITTING: Global linear model does NOT generalize!"
            elif r2_test_ols < 0.8:
                verdict = "PARTIAL: Some linear structure, but nonlinear components significant"
            elif r2_test_ols < 0.95:
                verdict = "STRONG: Linear model generalizes well, but not exact"
            else:
                verdict = "EXACT: MLP is effectively linear for this task"
            
            print(f"    VERDICT: {verdict}")
            
            # ---- Key analysis: what fraction of test variance is explained? ----
            # Also compute per-output-dimension R^2
            predictions_test = reg_ols.predict(h_ins_test)
            per_dim_r2 = 1 - np.sum((delta_mlps_test - predictions_test) ** 2, axis=0) / \
                            (np.sum((delta_mlps_test - delta_mlps_test.mean(axis=0)) ** 2, axis=0) + 1e-12)
            
            n_dims_above_90 = np.sum(per_dim_r2 > 0.9)
            n_dims_above_50 = np.sum(per_dim_r2 > 0.5)
            n_dims_positive = np.sum(per_dim_r2 > 0)
            
            print(f"    Per-dimension R^2: >0.9: {n_dims_above_90}/{len(per_dim_r2)}, "
                  f">0.5: {n_dims_above_50}/{len(per_dim_r2)}, "
                  f">0: {n_dims_positive}/{len(per_dim_r2)}")
    
    # ---- Cross-task generalization test ----
    print(f"\n{'='*60}")
    print(f"  Cross-Task Generalization Test")
    print(f"  If A_task is the 'real operator', can it predict OTHER tasks?")
    print(f"{'='*60}")
    
    for layer in layers:
        print(f"\n  Layer {layer}:")
        
        # Collect data for all tasks
        task_data = {}
        for task_name in task_names:
            samples = generate_task_samples(task_name, n=200, seed=123)
            h_ins, delta_mlps = [], []
            for text in samples:
                tokens = model.to_tokens(text)
                _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
                h_in = cache[f'blocks.{layer}.hook_resid_pre'][-1].detach().cpu().numpy()
                h_mid = cache[f'blocks.{layer}.hook_resid_mid'][-1].detach().cpu().numpy()
                h_out = cache[f'blocks.{layer}.hook_resid_post'][-1].detach().cpu().numpy()
                delta_mlp = h_out - h_mid
                h_ins.append(h_in)
                delta_mlps.append(delta_mlp)
            task_data[task_name] = (np.array(h_ins), np.array(delta_mlps))
        
        # Fit A on each task, test on others
        for source_task in task_names:
            h_train, delta_train = task_data[source_task]
            reg = Ridge(alpha=1.0)
            reg.fit(h_train, delta_train)
            
            r2_within = reg.score(h_train, delta_train)
            
            cross_r2 = {}
            for target_task in task_names:
                if target_task == source_task:
                    continue
                h_test, delta_test = task_data[target_task]
                r2_cross = reg.score(h_test, delta_test)
                cross_r2[target_task] = r2_cross
            
            cross_str = ", ".join([f"{t}={r2:.4f}" for t, r2 in cross_r2.items()])
            print(f"    A({source_task}) within={r2_within:.4f}, cross: {cross_str}")


# ============================================================
# Experiment B: Analytical Local Jacobian Field ★★★★★
# ============================================================

def compute_gelu_derivative(pre_act):
    """Compute GELU'(x) for each element in pre_act using autograd."""
    x = pre_act.clone().detach().requires_grad_(True)
    y = torch.nn.functional.gelu(x)
    grad = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))[0]
    return grad.detach()


def compute_mlp_jacobian(model, cache, layer, position=-1):
    """
    Compute the EXACT local Jacobian of the MLP output w.r.t. its input.
    
    MLP: out = GELU(LN(h_mid) @ W_in + b_in) @ W_out + b_out
    
    The Jacobian w.r.t. the post-LN input x = LN(h_mid):
      J = W_out^T @ diag(GELU'(z)) @ W_in^T
    
    where z = x @ W_in + b_in is the pre-GELU activation.
    
    This is the TRUE local operator — it varies with each data point
    because GELU'(z) depends on the activation pattern.
    """
    # Get pre-GELU activation from cache
    pre = cache[f'blocks.{layer}.mlp.hook_pre'][position]  # [d_mlp]
    
    # Compute GELU derivative at each neuron
    gelu_deriv = compute_gelu_derivative(pre)  # [d_mlp]
    
    # Get weight matrices
    W_in = model.blocks[layer].mlp.W_in.detach()   # [d_model, d_mlp]
    W_out = model.blocks[layer].mlp.W_out.detach()  # [d_mlp, d_model]
    
    # J = W_out^T @ diag(gelu_deriv) @ W_in^T
    # Efficient computation: 
    # diag(gelu_deriv) @ W_in^T = gelu_deriv.unsqueeze(1) * W_in.T  →  [d_mlp, d_model]
    # Then W_out^T @ result → [d_model, d_model]
    
    scaled_W_in_T = gelu_deriv.unsqueeze(1) * W_in.T  # [d_mlp, d_model]
    J = W_out.T @ scaled_W_in_T  # [d_model, d_model]
    
    return J


def exp_b_local_jacobian(model):
    """
    THE REAL OPERATOR: Compute exact local Jacobian at each data point.
    
    Key questions:
    1. How much does J(h) vary within a task? (tests linearization validity)
    2. How much does J(h) differ between tasks? (tests operator family hypothesis)
    3. What is the spectral structure of J(h)? (reveals computation type)
    """
    print("=" * 70)
    print("Experiment B: Analytical Local Jacobian Field")
    print("THE REAL OPERATOR: J(h) = d(MLP_out)/d(h_input)")
    print("=" * 70)
    
    task_names = ["addition", "translate_fr", "antonym", "capital"]
    n_samples = 200  # Enough for statistical analysis
    layers = [0, 3, 6, 9, 11]
    
    # Collect Jacobians for all tasks and layers
    all_jacobian_data = {}  # {layer: {task: [spectral_data]}}
    
    for layer in layers:
        print(f"\n{'='*60}")
        print(f"  Layer {layer}")
        print(f"{'='*60}")
        
        layer_data = {}
        
        for task_name in task_names:
            samples = generate_task_samples(task_name, n=n_samples, seed=42)
            
            singular_values_list = []
            gelu_active_ratios = []
            top3_energies = []
            effective_ranks = []
            operator_norms = []
            traces = []
            
            print(f"  {task_name}: computing {n_samples} Jacobians...")
            
            for idx, text in enumerate(samples):
                tokens = model.to_tokens(text)
                _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
                
                # Compute exact Jacobian
                J = compute_mlp_jacobian(model, cache, layer, position=-1)
                
                # SVD of Jacobian
                U, S, Vt = torch.linalg.svd(J, full_matrices=False)
                S_np = S.numpy()
                
                # Spectral metrics
                total_energy = np.sum(S_np ** 2)
                top3_energy = np.sum(S_np[:3] ** 2) / total_energy
                cum_energy = np.cumsum(S_np ** 2) / total_energy
                eff_rank = np.searchsorted(cum_energy, 0.95) + 1
                op_norm = np.sqrt(total_energy)
                trace_val = np.trace(J.numpy())
                
                # GELU activation ratio
                pre = cache[f'blocks.{layer}.mlp.hook_pre'][-1].detach()
                gelu_deriv = compute_gelu_derivative(pre)
                active_ratio = (gelu_deriv > 0.1).float().mean().item()
                
                singular_values_list.append(S_np)
                top3_energies.append(top3_energy)
                effective_ranks.append(eff_rank)
                operator_norms.append(op_norm)
                traces.append(trace_val)
                gelu_active_ratios.append(active_ratio)
            
            # Stack singular values for analysis
            sv_matrix = np.array(singular_values_list)  # [n_samples, d_model]
            
            # Compute statistics
            mean_sv = sv_matrix.mean(axis=0)
            std_sv = sv_matrix.std(axis=0)
            cv_sv = std_sv / (mean_sv + 1e-12)  # Coefficient of variation
            
            mean_top3 = np.mean(top3_energies)
            std_top3 = np.std(top3_energies)
            mean_eff_rank = np.mean(effective_ranks)
            std_eff_rank = np.std(effective_ranks)
            mean_op_norm = np.mean(operator_norms)
            std_op_norm = np.std(operator_norms)
            mean_active = np.mean(gelu_active_ratios)
            
            layer_data[task_name] = {
                'sv_matrix': sv_matrix,
                'mean_sv': mean_sv,
                'std_sv': std_sv,
                'cv_sv': cv_sv,
                'top3_energies': top3_energies,
                'effective_ranks': effective_ranks,
                'operator_norms': operator_norms,
                'traces': traces,
                'gelu_active_ratios': gelu_active_ratios,
            }
            
            print(f"    {task_name}:")
            print(f"      OpNorm: {mean_op_norm:.4f} +/- {std_op_norm:.4f}")
            print(f"      Top3 energy: {mean_top3:.4f} +/- {std_top3:.4f}")
            print(f"      Eff rank(95%): {mean_eff_rank:.1f} +/- {std_eff_rank:.1f}")
            print(f"      GELU active ratio: {mean_active:.4f}")
            print(f"      SV coefficient of variation (top-10 avg): {cv_sv[:10].mean():.4f}")
            print(f"      Top-5 SVs: {mean_sv[:5].tolist()}")
            print(f"      Top-5 SV std: {std_sv[:5].tolist()}")
        
        all_jacobian_data[layer] = layer_data
        
        # ---- Cross-task Jacobian comparison ----
        print(f"\n    Cross-task Jacobian comparison:")
        task_names_list = list(layer_data.keys())
        for i, t1 in enumerate(task_names_list):
            for j, t2 in enumerate(task_names_list):
                if i >= j:
                    continue
                
                # Compare mean singular value spectra
                sv1 = layer_data[t1]['mean_sv']
                sv2 = layer_data[t2]['mean_sv']
                
                # Cosine similarity of SV spectra
                cos_sv = np.dot(sv1, sv2) / (np.linalg.norm(sv1) * np.linalg.norm(sv2) + 1e-12)
                
                # Compare operator norms
                norm1 = np.mean(layer_data[t1]['operator_norms'])
                norm2 = np.mean(layer_data[t2]['operator_norms'])
                
                # Overlap of top-k singular vectors (via subspace alignment)
                # Use PCA of Jacobian spectra as proxy
                sv1_all = layer_data[t1]['sv_matrix']  # [n, d_model]
                sv2_all = layer_data[t2]['sv_matrix']
                
                # Correlation of mean SV spectra
                corr = np.corrcoef(sv1, sv2)[0, 1]
                
                print(f"      {t1} vs {t2}: SV_cosine={cos_sv:.4f}, SV_corr={corr:.4f}, "
                      f"OpNorm_ratio={norm1/(norm2+1e-12):.4f}")
    
    # ---- Summary: Jacobian Variation Analysis ----
    print(f"\n{'='*60}")
    print(f"  Jacobian Variation Analysis (KEY RESULT)")
    print(f"  If CV(singular_values) is LOW → linearization is valid locally")
    print(f"  If CV(singular_values) is HIGH → need Jacobian field theory")
    print(f"{'='*60}")
    
    for layer in layers:
        print(f"\n  Layer {layer}:")
        for task_name in task_names:
            data = all_jacobian_data[layer][task_name]
            cv = data['cv_sv']
            
            # Mean CV for top-20 singular values
            cv_top20 = cv[:20].mean()
            cv_top5 = cv[:5].mean()
            cv_50_100 = cv[50:100].mean()
            
            # Within-task Jacobian variation
            sv_matrix = data['sv_matrix']
            # Pairwise cosine similarity of SV spectra within task
            n = min(50, sv_matrix.shape[0])
            from itertools import combinations
            pair_sims = []
            for i, j in combinations(range(n), 2):
                sim = np.dot(sv_matrix[i], sv_matrix[j]) / (
                    np.linalg.norm(sv_matrix[i]) * np.linalg.norm(sv_matrix[j]) + 1e-12)
                pair_sims.append(sim)
            mean_pair_sim = np.mean(pair_sims)
            
            print(f"    {task_name}: CV(top5)={cv_top5:.4f}, CV(top20)={cv_top20:.4f}, "
                  f"CV(mid)={cv_50_100:.4f}, within_sim={mean_pair_sim:.4f}")
    
    return all_jacobian_data


# ============================================================
# Experiment C: Jacobian Field Topology ★★★★★
# ============================================================

def exp_c_jacobian_topology(model):
    """
    Study the TOPOLOGY of the Jacobian field.
    
    Key questions:
    1. Do different tasks occupy different regions of "Jacobian space"?
    2. Is there a smooth manifold of Jacobians, or sharp transitions?
    3. What is the effective dimensionality of the Jacobian variation?
    4. Can we identify "Jacobian flow regions" corresponding to operator families?
    """
    print("=" * 70)
    print("Experiment C: Jacobian Field Topology")
    print("How does J(h) vary across the data manifold?")
    print("=" * 70)
    
    task_names = ["addition", "translate_fr", "antonym", "capital", "continue"]
    n_samples = 150
    layers = [3, 6, 9]
    
    for layer in layers:
        print(f"\n{'='*60}")
        print(f"  Layer {layer}")
        print(f"{'='*60}")
        
        # Collect spectral features for all tasks
        all_features = []
        all_labels = []
        task_features = {}
        
        for task_name in task_names:
            samples = generate_task_samples(task_name, n=n_samples, seed=42)
            
            features = []
            for text in samples:
                tokens = model.to_tokens(text)
                _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
                
                J = compute_mlp_jacobian(model, cache, layer, position=-1)
                _, S, _ = torch.linalg.svd(J, full_matrices=False)
                S_np = S.numpy()
                
                # Use top-50 singular values as features
                features.append(S_np[:50])
            
            features = np.array(features)  # [n_samples, 50]
            task_features[task_name] = features
            all_features.append(features)
            all_labels.extend([task_name] * len(features))
        
        all_features = np.vstack(all_features)  # [n_total, 50]
        
        # ---- PCA of Jacobian spectra ----
        print(f"\n  PCA of Jacobian Spectra (top-50 SVs as features):")
        pca = PCA(n_components=10)
        projected = pca.fit_transform(all_features)
        
        print(f"    Variance explained: {pca.explained_variance_ratio_[:5]}")
        print(f"    Cumulative: {np.cumsum(pca.explained_variance_ratio_[:5])}")
        
        # Project each task's features
        print(f"\n    Task centroids on PC1-PC2:")
        for task_name in task_names:
            mask = np.array(all_labels) == task_name
            centroid = projected[mask, :2].mean(axis=0)
            spread = projected[mask, :2].std(axis=0)
            print(f"      {task_name}: PC1={centroid[0]:.4f}+/-{spread[0]:.4f}, "
                  f"PC2={centroid[1]:.4f}+/-{spread[1]:.4f}")
        
        # ---- Task separability in Jacobian space ----
        print(f"\n  Task Separability (can Jacobians distinguish tasks?):")
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.model_selection import cross_val_score
        
        lda = LinearDiscriminantAnalysis()
        scores = cross_val_score(lda, all_features, all_labels, cv=5)
        print(f"    LDA 5-fold accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")
        
        # ---- Jacobian field continuity ----
        print(f"\n  Jacobian Field Continuity:")
        print(f"  (Are Jacobians smoothly varying or sharply transitioning?)")
        
        for task_name in task_names:
            features = task_features[task_name]
            # Pairwise distances in spectral space
            from scipy.spatial.distance import pdist
            dists = pdist(features[:50], metric='cosine')
            
            # Histogram of distances
            q25, q50, q75 = np.percentile(dists, [25, 50, 75])
            max_dist = np.max(dists)
            
            print(f"    {task_name}: cosine dist quartiles [{q25:.4f}, {q50:.4f}, {q75:.4f}], max={max_dist:.4f}")
        
        # ---- Cross-task Jacobian overlap ----
        print(f"\n  Cross-task Jacobian Overlap:")
        print(f"  (Do tasks share the same Jacobian manifold?)")
        
        task_names_list = list(task_features.keys())
        for i, t1 in enumerate(task_names_list):
            for j, t2 in enumerate(task_names_list):
                if i >= j:
                    continue
                
                f1 = task_features[t1]
                f2 = task_features[t2]
                
                # Subspace overlap: how well does one task's PCA subspace cover the other?
                pca1 = PCA(n_components=10)
                pca1.fit(f1)
                pca2 = PCA(n_components=10)
                pca2.fit(f2)
                
                Q1 = pca1.components_.T  # [50, 10]
                Q2 = pca2.components_.T  # [50, 10]
                
                # Subspace alignment
                alignment = np.linalg.norm(Q1.T @ Q2, 'fro') / np.sqrt(10)
                
                # Can one task's PCA reconstruct the other?
                f2_proj = pca1.transform(f2)
                f2_recon = pca1.inverse_transform(f2_proj)
                recon_error = np.mean((f2 - f2_recon) ** 2) / (np.mean(f2 ** 2) + 1e-12)
                
                print(f"    {t1} -> {t2}: alignment={alignment:.4f}, recon_error={recon_error:.4f}")
    
    # ---- Key question: How many distinct "operator types" are there? ----
    print(f"\n{'='*60}")
    print(f"  Operator Type Clustering")
    print(f"  (Are there discrete operator families or a continuous spectrum?)")
    print(f"{'='*60}")
    
    for layer in layers:
        print(f"\n  Layer {layer}:")
        
        all_features = []
        all_labels = []
        
        for task_name in task_names:
            samples = generate_task_samples(task_name, n=100, seed=42)
            for text in samples:
                tokens = model.to_tokens(text)
                _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
                J = compute_mlp_jacobian(model, cache, layer, position=-1)
                _, S, _ = torch.linalg.svd(J, full_matrices=False)
                all_features.append(S.numpy()[:30])
                all_labels.append(task_name)
        
        all_features = np.array(all_features)
        
        # K-means clustering in spectral space
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
        
        for k in [2, 3, 4, 5]:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            pred = km.fit_predict(all_features)
            
            ari = adjusted_rand_score(all_labels, pred)
            nmi = normalized_mutual_info_score(all_labels, pred)
            
            print(f"    K={k}: ARI={ari:.4f}, NMI={nmi:.4f}")
            
            # Show cluster composition
            from collections import Counter
            for c in range(k):
                mask = pred == c
                task_dist = Counter([all_labels[i] for i in range(len(mask)) if mask[i]])
                total_in_cluster = sum(task_dist.values())
                top_tasks = task_dist.most_common(3)
                top_str = ", ".join([f"{t}({c/total_in_cluster:.2f})" for t, c in top_tasks])
                print(f"      Cluster {c} (n={total_in_cluster}): {top_str}")


# ============================================================
# Experiment D: Spectrum Dynamics & Recursive Rollout ★★★★
# ============================================================

def exp_d_spectrum_dynamics(model):
    """
    Study how the Jacobian spectrum evolves across layers.
    
    Key questions:
    1. How does the operator change from shallow to deep layers?
    2. Are there "phase transitions" in the spectrum?
    3. What happens when we compose operators across layers?
    4. Which modes are conserved, which are expanded/contracted?
    """
    print("=" * 70)
    print("Experiment D: Spectrum Dynamics & Recursive Rollout")
    print("How does the operator evolve across layers?")
    print("=" * 70)
    
    task_names = ["addition", "translate_fr", "antonym", "capital"]
    n_samples = 50
    all_layers = list(range(12))
    
    # ---- Part 1: Layer-by-layer spectrum evolution ----
    print(f"\n{'='*60}")
    print(f"  Part 1: Jacobian Spectrum Evolution Across Layers")
    print(f"{'='*60}")
    
    for task_name in task_names:
        print(f"\n  {task_name}:")
        samples = generate_task_samples(task_name, n=n_samples, seed=42)
        
        # For each sample, compute Jacobians at all layers
        sample_spectra = []  # [n_samples, n_layers, d_model]
        
        for text in samples:
            tokens = model.to_tokens(text)
            _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            
            layer_spectra = []
            for layer in all_layers:
                J = compute_mlp_jacobian(model, cache, layer, position=-1)
                _, S, _ = torch.linalg.svd(J, full_matrices=False)
                layer_spectra.append(S.numpy())
            
            sample_spectra.append(np.array(layer_spectra))  # [12, d_model]
        
        sample_spectra = np.array(sample_spectra)  # [n_samples, 12, d_model]
        mean_spectra = sample_spectra.mean(axis=0)  # [12, d_model]
        std_spectra = sample_spectra.std(axis=0)    # [12, d_model]
        
        # Report key metrics per layer
        print(f"    {'Layer':>5} {'OpNorm':>8} {'Top1':>8} {'Top3%':>8} {'Rank95':>8} {'Trace':>10}")
        for layer in all_layers:
            sv = mean_spectra[layer]
            total_energy = np.sum(sv ** 2)
            top3_energy = np.sum(sv[:3] ** 2) / total_energy
            cum_energy = np.cumsum(sv ** 2) / total_energy
            eff_rank = np.searchsorted(cum_energy, 0.95) + 1
            
            # Trace = sum of diagonal of J = sum of eigenvalues (not singular values!)
            # For non-symmetric J, trace ≠ sum of SVs. But sum of SVs gives a proxy.
            trace_proxy = np.sum(sv)
            
            print(f"    {layer:>5} {np.sqrt(total_energy):>8.4f} {sv[0]:>8.4f} "
                  f"{top3_energy:>8.4f} {eff_rank:>8d} {trace_proxy:>10.4f}")
        
        # Spectral transitions: how much does the spectrum change between adjacent layers?
        print(f"\n    Spectral transition (cosine similarity between adjacent layer spectra):")
        for layer in range(11):
            sv1 = mean_spectra[layer]
            sv2 = mean_spectra[layer + 1]
            cos = np.dot(sv1, sv2) / (np.linalg.norm(sv1) * np.linalg.norm(sv2) + 1e-12)
            print(f"      L{layer} -> L{layer+1}: {cos:.4f}")
    
    # ---- Part 2: Recursive Rollout — Composing Jacobians ----
    print(f"\n{'='*60}")
    print(f"  Part 2: Recursive Rollout — Operator Composition")
    print(f"  J_total(L0->Lk) ≈ J_Lk @ ... @ J_L0")
    print(f"  This tells us how perturbations propagate through the network")
    print(f"{'='*60}")
    
    # Use a single sample from each task for the rollout analysis
    for task_name in task_names[:2]:  # Just addition and translate_fr for speed
        print(f"\n  {task_name}:")
        samples = generate_task_samples(task_name, n=5, seed=42)
        
        for sample_idx, text in enumerate(samples):
            tokens = model.to_tokens(text)
            _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            
            # Compute Jacobians at all layers
            jacobians = []
            for layer in all_layers:
                J = compute_mlp_jacobian(model, cache, layer, position=-1)
                jacobians.append(J.numpy())
            
            # Compose Jacobians: J_total(k) = J_k @ ... @ J_0
            # This is the sensitivity of the output at layer k w.r.t. the input at layer 0
            print(f"    Sample {sample_idx}: Composed operator spectrum")
            
            J_composed = np.eye(768)  # Start with identity
            for k in range(12):
                J_composed = jacobians[k] @ J_composed
                
                # SVD of composed operator
                S_composed = np.linalg.svd(J_composed, compute_uv=False)
                
                total_energy = np.sum(S_composed ** 2)
                top1 = S_composed[0]
                top3_energy = np.sum(S_composed[:3] ** 2) / total_energy if total_energy > 0 else 0
                cum_energy = np.cumsum(S_composed ** 2) / (total_energy + 1e-12)
                eff_rank = np.searchsorted(cum_energy, 0.95) + 1 if total_energy > 0 else 0
                
                # Operator norm growth
                op_norm = np.sqrt(total_energy)
                
                # Largest singular value (dominant direction)
                # Smallest non-zero singular value (weakest direction)
                s_min_nonzero = S_composed[S_composed > 1e-10]
                s_min = s_min_nonzero[-1] if len(s_min_nonzero) > 0 else 0
                
                # Condition number
                cond = top1 / (s_min + 1e-12)
                
                print(f"      After L{k}: OpNorm={op_norm:.4f}, top1={top1:.4f}, "
                      f"s_min={s_min:.6f}, cond={cond:.2f}, rank95={eff_rank}")
            
            if sample_idx >= 1:  # Just show 2 samples per task
                break
    
    # ---- Part 3: Conserved vs. Unstable Modes ----
    print(f"\n{'='*60}")
    print(f"  Part 3: Conserved vs. Unstable Modes")
    print(f"  Which directions are amplified/attenuated through the network?")
    print(f"{'='*60}")
    
    for task_name in task_names[:2]:
        print(f"\n  {task_name}:")
        samples = generate_task_samples(task_name, n=20, seed=42)
        
        # Collect composed Jacobians for multiple samples
        all_composed_SV = []
        
        for text in samples:
            tokens = model.to_tokens(text)
            _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            
            jacobians = []
            for layer in all_layers:
                J = compute_mlp_jacobian(model, cache, layer, position=-1)
                jacobians.append(J.numpy())
            
            # Full composition
            J_composed = np.eye(768)
            for k in range(12):
                J_composed = jacobians[k] @ J_composed
            
            S = np.linalg.svd(J_composed, compute_uv=False)
            all_composed_SV.append(S)
        
        all_composed_SV = np.array(all_composed_SV)  # [n_samples, 768]
        mean_SV = all_composed_SV.mean(axis=0)
        
        # Classify modes
        amplified = np.sum(mean_SV > 1.1)  # Grew by >10%
        conserved = np.sum((mean_SV > 0.9) & (mean_SV <= 1.1))  # Within 10% of 1
        attenuated = np.sum(mean_SV <= 0.9)  # Shrank by >10%
        near_zero = np.sum(mean_SV < 0.01)  # Essentially killed
        
        print(f"    Full 12-layer composed operator:")
        print(f"      Top-5 SVs: {mean_SV[:5].tolist()}")
        print(f"      Amplified (>1.1): {amplified}")
        print(f"      Conserved (0.9-1.1): {conserved}")
        print(f"      Attenuated (<0.9): {attenuated}")
        print(f"      Near zero (<0.01): {near_zero}")
        print(f"      Dynamic range: {mean_SV[0]:.4f} to {mean_SV[mean_SV > 0.01][-1]:.6f}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="all", 
                       choices=["a", "b", "c", "d", "all"])
    args = parser.parse_args()
    
    print("Phase 81: Operator Mechanics — From Global Fitting to Local Jacobian")
    print("=" * 70)
    print("KEY FIX: R^2=1.0 was overfitting. Use train/test split + exact Jacobian.")
    print("TRUE OPERATOR: J(h) = W_out^T @ diag(GELU'(z)) @ W_in^T")
    print("This VARIES with each data point — it's a FIELD, not a matrix.")
    print("=" * 70)
    
    start_time = time.time()
    model = get_model()
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.1f}s")
    
    if args.exp in ["a", "all"]:
        t0 = time.time()
        exp_a_train_test_split(model)
        print(f"\n  Experiment A time: {time.time()-t0:.1f}s")
    
    if args.exp in ["b", "all"]:
        t0 = time.time()
        jacobian_data = exp_b_local_jacobian(model)
        print(f"\n  Experiment B time: {time.time()-t0:.1f}s")
    
    if args.exp in ["c", "all"]:
        t0 = time.time()
        exp_c_jacobian_topology(model)
        print(f"\n  Experiment C time: {time.time()-t0:.1f}s")
    
    if args.exp in ["d", "all"]:
        t0 = time.time()
        exp_d_spectrum_dynamics(model)
        print(f"\n  Experiment D time: {time.time()-t0:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\n\nPhase 81 completed. Total time: {total_time:.1f}s")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M')}")
