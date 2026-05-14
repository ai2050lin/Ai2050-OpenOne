"""
Phase 162: Logits Manifold & Fisher Geometry — The Correct State Space
======================================================================

Based on user's Phase 160-161 critique — THE most critical theoretical pivot:

User's key insight: "The real state of a language model is not h_t (hidden state),
but P_t = P(next token | h_t) — the probability distribution over the vocabulary."

This means:
  - Hidden state is a WORKSPACE, not the STATE
  - The true state lives in the probability simplex, not Euclidean space
  - The correct geometry is Fisher geometry (information geometry), not Euclidean

Core experiments:

  Exp 1: Logits Geometry vs Hidden State Geometry
    - Extract logits at each token position
    - Compute pairwise distances in:
      a) L2 in hidden state space (our old metric)
      b) L2 in logit space (new: projection of h into W_U)
      c) KL divergence between output distributions
      d) Fisher-Rao distance (approximated via KL)
    - Key question: Is logit-space geometry more predictive than hidden-state geometry?

  Exp 2: Fisher Information Structure
    - Compute Fisher Information Matrix at each point
    - Study eigenvalue structure: what is the intrinsic Fisher dimension?
    - Compute geodesic distances (approximated) between logit points
    - Key question: What is the true intrinsic dimension of the probability manifold?

  Exp 3: Local vs Global Dimensionality
    - Compute local Fisher dimension in different semantic regions
    - Compare dimensions across: narrative, technical, dialogue, poetry
    - Key question: Is the "low-dimensional" structure local or global?

  Exp 4: Predictive State Compression in Logit Space
    - Compress logits (not hidden states) using information bottleneck
    - Compare: PCA in logit space vs PCA in hidden state space
    - Key question: How many logit dimensions suffice for prediction?

CRITICAL DESIGN DECISIONS (based on user feedback):
  - We study LOGITS (ℓ_t = W_out h_t), not hidden states
  - We use KL/Fisher distances, not L2 distances
  - We test LOCAL structure across semantic domains
  - We do NOT assume vector space structure — the state is a distribution

Usage: python tests/glm5/phase162_logits_manifold.py <model_name>
  model_name: qwen3, glm4, deepseek7b
"""

import sys
import os
import time
import json
import gc
import numpy as np
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'tests/glm5')

import torch
from model_utils import load_model, get_model_info, get_layers, release_model, get_W_U


# ===== 1. DIVERSE PROMPT GENERATION (longer, cross-domain) =====

def generate_domain_prompts(n_per_domain=200):
    """Generate prompts from 4 distinct semantic domains to test local vs global structure"""
    
    # Domain 1: NARRATIVE (stories, events, characters)
    narrative_templates = [
        "The {adj} {noun} {verb} through the {place} while the {adj2} {noun2} watched carefully",
        "After the {noun} {verb} the {adj} {noun2}, everyone in the {place} was {adj2}",
        "She {verb} the {adj} {noun} into the {place} and {verb2} at the {noun2}",
        "The {noun} had been {adj} for so long that when it finally {verb}, the {noun2} {verb2}",
        "In the {adj} {place}, the {noun} {verb} while the {noun2} {verb2} nearby",
        "When the {adj} {noun} arrived at the {place}, the {noun2} was already {verb2}",
        "The {noun} and the {noun2} {verb} together across the {adj} {place}",
        "After {verb2} the {adj} {noun}, the {noun2} decided to {verb} the {place}",
        "The {adj} {noun} {verb} every {noun2} in the {place} until nothing {adj2} remained",
        "While the {noun} {verb}, the {noun2} {verb2} a {adj} {place} nearby",
    ]
    
    # Domain 2: TECHNICAL (science, logic, systems)
    technical_templates = [
        "The {noun} algorithm {verb} the {adj} data by computing the {noun2} of each {adj2} variable",
        "In the {adj} system, the {noun} {verb} when the {noun2} reaches the {adj2} threshold",
        "The {noun} function {verb} a {adj} {noun2} that {verb2} the input into the output space",
        "By {verb2} the {adj} {noun}, the {noun2} can be {adj2} with high accuracy",
        "The {noun} model {verb} the {adj2} relationship between {noun2} and the {adj} parameter",
        "When the {adj} {noun} {verb}, the {noun2} converges to a {adj2} solution",
        "The {adj} {noun} {verb} the {noun2} by applying a {adj2} transformation to the input",
        "The {noun} architecture {verb} {adj} {noun2} layers that {verb2} the representation",
        "A {adj} {noun} can {verb} the {noun2} without {adj2} computational overhead",
        "The {adj} {noun2} {verb} when the {noun} {verb2} the {adj2} state",
    ]
    
    # Domain 3: DIALOGUE (conversation, questions, interactions)
    dialogue_templates = [
        '"Do you think the {adj} {noun} will {verb}?" she asked, looking at the {noun2}',
        '"I cannot {verb} the {adj} {noun}," he replied, {verb2} at the {noun2}',
        '"The {noun} has been {adj} since the {noun2} {verb}," they agreed',
        '"Why did the {adj} {noun} {verb}?" she wondered about the {noun2}',
        '"If the {noun} {verb}, then the {noun2} must be {adj}," he {verb2}',
        '"Can you {verb} the {adj} {noun2}?" she asked the {noun}',
        '"The {noun} {verb} because the {adj} {noun2} was {adj2}," they explained',
        '"I {verb} that the {noun} is {adj}," she said about the {noun2}',
        '"When will the {noun2} {verb}?" the {adj} {noun} wanted to know',
        '"The {adj} {noun} {verb} the {noun2}," he confirmed with a {adj2} expression',
    ]
    
    # Domain 4: DESCRIPTIVE (qualities, states, observations)
    descriptive_templates = [
        "The {adj} {noun} was so {adj2} that it made the {noun2} look {adj} by comparison",
        "There is something {adj} about the way the {noun} {verb} near the {noun2}",
        "The {noun} appeared {adj} as it {verb} across the {adj2} {place}",
        "What makes the {adj} {noun} {verb} is its {adj2} {noun2}",
        "The {adj} quality of the {noun} {verb} whenever the {noun2} is {adj2}",
        "Looking at the {adj} {noun}, one can {verb} how {adj2} the {noun2} has become",
        "The {noun} is {adj} in the way it {verb} the {adj2} {noun2}",
        "Despite being {adj}, the {noun} {verb} with {adj2} intensity near the {noun2}",
        "The {adj2} {noun2} {verb} a {adj} quality that the {noun} lacks",
        "Nothing is more {adj} than a {noun} that {verb} the {adj2} {noun2}",
    ]
    
    # Common word pools
    nouns = [
        "cat", "dog", "bird", "fish", "child", "woman", "man", "teacher", "doctor", "student",
        "tree", "flower", "river", "mountain", "car", "book", "house", "city", "king", "queen",
        "soldier", "artist", "scientist", "writer", "musician", "engineer", "farmer", "driver",
        "chair", "table", "door", "window", "road", "bridge", "tower", "castle", "garden", "forest",
        "ocean", "island", "valley", "desert", "planet", "star", "moon", "sun", "cloud", "rain",
        "apple", "bread", "water", "fire", "stone", "glass", "paper", "silk", "gold", "iron",
        "village", "library", "kitchen", "market", "harbor", "cathedral", "museum", "theater", "palace", "temple",
        "algorithm", "system", "model", "function", "network", "process", "parameter", "variable", "vector", "matrix",
        "component", "structure", "layer", "method", "framework", "data", "signal", "output", "input", "feature",
        "machine", "computer", "circuit", "program", "code", "logic", "pattern", "rule", "theory", "equation",
    ]
    verbs = [
        "runs", "sits", "walks", "jumps", "flies", "swims", "reads", "writes", "eats", "drinks",
        "sings", "dances", "sleeps", "wakes", "falls", "rises", "moves", "stops", "starts", "ends",
        "opens", "closes", "grows", "shrinks", "turns", "stands", "lies", "hangs", "breaks", "builds",
        "discovers", "analyzes", "creates", "observes", "remembers", "forgets", "believes", "doubts", "loves", "hates",
        "computes", "transforms", "converges", "generates", "processes", "measures", "estimates", "approximates", "optimizes", "updates",
        "reached", "found", "knew", "felt", "thought", "said", "believed", "wondered", "agreed", "decided",
    ]
    adjs = [
        "red", "blue", "green", "big", "small", "old", "new", "fast", "slow", "tall",
        "short", "dark", "bright", "warm", "cold", "soft", "hard", "rich", "poor", "young",
        "heavy", "light", "thick", "thin", "wide", "narrow", "deep", "shallow", "clean", "dirty",
        "ancient", "modern", "beautiful", "strange", "powerful", "gentle", "fierce", "calm", "wild", "quiet",
        "complex", "simple", "efficient", "robust", "precise", "stable", "dynamic", "static", "linear", "nonlinear",
    ]
    places = [
        "garden", "forest", "castle", "village", "river", "mountain", "ocean", "desert", "city", "valley",
        "library", "temple", "market", "harbor", "cathedral", "museum", "theater", "palace", "kitchen", "tower",
    ]
    
    rng = np.random.RandomState(42)
    all_prompts = {}
    
    for domain_name, templates in [
        ("narrative", narrative_templates),
        ("technical", technical_templates),
        ("dialogue", dialogue_templates),
        ("descriptive", descriptive_templates),
    ]:
        prompts = []
        for _ in range(n_per_domain):
            template = templates[rng.randint(len(templates))]
            prompt = template.format(
                noun=nouns[rng.randint(len(nouns))],
                noun2=nouns[rng.randint(len(nouns))],
                verb=verbs[rng.randint(len(verbs))],
                verb2=verbs[rng.randint(len(verbs))],
                adj=adjs[rng.randint(len(adjs))],
                adj2=adjs[rng.randint(len(adjs))],
                place=places[rng.randint(len(places))],
            )
            prompts.append(prompt)
        all_prompts[domain_name] = prompts
    
    return all_prompts


# ===== 2. DATA COLLECTION =====

def collect_logits_and_hidden_states(model, tokenizer, device, prompts, model_info, max_seq_len=48):
    """Collect logits AND hidden states for each token position"""
    model_dtype = next(model.parameters()).dtype
    
    all_logits = []      # logits vectors (before softmax)
    all_probs = []       # probability distributions
    all_hidden = []      # hidden state vectors
    all_domains = []     # domain label
    all_positions = []   # position in sequence
    
    final_norm = model.model.norm
    lm_head = model.lm_head
    
    for prompt in prompts:
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_seq_len)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            if input_ids.shape[1] < 4:
                continue
            
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            
            # Get hidden states from the last transformer layer
            hs = out.hidden_states[model_info.n_layers]  # [1, seq_len, d_model]
            
            # Compute logits via final_norm + lm_head
            for pos in range(1, min(hs.shape[1], max_seq_len)):
                h = hs[0, pos, :].detach()  # [d_model]
                
                try:
                    # Compute logits
                    h_unsqueezed = h.unsqueeze(0).unsqueeze(0).to(device=device, dtype=model_dtype)
                    with torch.no_grad():
                        h_normed = final_norm(h_unsqueezed)
                        logits = lm_head(h_normed)  # [1, 1, vocab]
                        logits_vec = logits[0, 0, :].float().cpu().numpy()  # [vocab]
                    
                    # Skip if NaN/Inf (common with 8bit quantization)
                    if np.any(np.isnan(logits_vec)) or np.any(np.isinf(logits_vec)):
                        continue
                    
                    # Softmax to get probabilities
                    logits_max = np.max(logits_vec)
                    exp_logits = np.exp(logits_vec - logits_max)
                    probs = exp_logits / np.sum(exp_logits)
                    
                    all_logits.append(logits_vec)
                    all_probs.append(probs)
                    all_hidden.append(h.float().cpu().numpy())
                    all_positions.append(pos)
                except Exception:
                    continue
        
        except Exception:
            continue
    
    # Filter out NaN/Inf entries (common with 8bit quantization)
    if all_logits:
        logits_arr = np.array(all_logits, dtype=np.float32)
        probs_arr = np.array(all_probs, dtype=np.float32)
        hidden_arr = np.array(all_hidden, dtype=np.float32)
        pos_arr = np.array(all_positions, dtype=np.int32)
        
        # Find rows without NaN/Inf
        valid_mask = (
            ~np.any(np.isnan(logits_arr), axis=1) & 
            ~np.any(np.isinf(logits_arr), axis=1) &
            ~np.any(np.isnan(probs_arr), axis=1) &
            ~np.any(np.isnan(hidden_arr), axis=1)
        )
        
        n_invalid = int(np.sum(~valid_mask))
        if n_invalid > 0:
            print(f"  WARNING: Filtered {n_invalid}/{len(valid_mask)} entries with NaN/Inf")
        
        logits_arr = logits_arr[valid_mask]
        probs_arr = probs_arr[valid_mask]
        hidden_arr = hidden_arr[valid_mask]
        pos_arr = pos_arr[valid_mask]
    else:
        logits_arr = np.array([])
        probs_arr = np.array([])
        hidden_arr = np.array([])
        pos_arr = np.array([])
    
    return {
        "logits": logits_arr,
        "probs": probs_arr,
        "hidden": hidden_arr,
        "positions": pos_arr,
    }


# ===== 3. EXP 1: LOGITS GEOMETRY VS HIDDEN STATE GEOMETRY =====

def exp1_logits_vs_hidden_geometry(data, n_sample=800):
    """Compare geometry in logit space vs hidden state space vs probability space"""
    logits = np.nan_to_num(data["logits"], nan=0.0, posinf=0.0, neginf=0.0)
    probs = np.nan_to_num(data["probs"], nan=1e-10, posinf=1.0, neginf=1e-10)
    hidden = np.nan_to_num(data["hidden"], nan=0.0, posinf=0.0, neginf=0.0)
    
    N = len(logits)
    if N < 100:
        return {"error": "insufficient_data"}
    
    n_sample = min(n_sample, N)
    rng = np.random.RandomState(42)
    idx = rng.choice(N, n_sample, replace=False)
    
    L = logits[idx]  # [n_sample, vocab]
    P = probs[idx]   # [n_sample, vocab]
    H = hidden[idx]  # [n_sample, d_model]
    
    # Subsample pairs for efficiency
    n_pairs = min(3000, n_sample * (n_sample - 1) // 2)
    pair_idx = []
    while len(pair_idx) < n_pairs:
        i, j = rng.randint(0, n_sample, 2)
        if i != j:
            pair_idx.append((i, j))
    
    print(f"  Exp1: Computing {n_pairs} pairwise distances across 3 spaces...")
    
    l2_hidden = []
    l2_logits = []
    kl_probs = []
    fisher_approx = []
    
    for i, j in pair_idx:
        # L2 in hidden state space
        d_h = np.linalg.norm(H[i] - H[j])
        l2_hidden.append(float(d_h))
        
        # L2 in logit space
        d_l = np.linalg.norm(L[i] - L[j])
        l2_logits.append(float(d_l))
        
        # KL divergence between probability distributions (symmetrized)
        p, q = P[i], P[j]
        # Clip for numerical stability
        p = np.clip(p, 1e-10, 1.0)
        q = np.clip(q, 1e-10, 1.0)
        p = p / p.sum()
        q = q / q.sum()
        kl_pq = np.sum(p * np.log(p / q))
        kl_qp = np.sum(q * np.log(q / p))
        kl_sym = 0.5 * (kl_pq + kl_qp)
        kl_probs.append(float(kl_sym))
        
        # Fisher-Rao approximation: 2 * arccos(sum(sqrt(p*q)))
        # This is the exact Fisher-Rao distance on the probability simplex
        sqrt_pq = np.sqrt(p * q)
        inner = np.sum(sqrt_pq)
        inner = np.clip(inner, -1.0, 1.0)
        fr_dist = 2.0 * np.arccos(inner)
        fisher_approx.append(float(fr_dist))
    
    l2_hidden = np.array(l2_hidden)
    l2_logits = np.array(l2_logits)
    kl_probs = np.array(kl_probs)
    fisher_approx = np.array(fisher_approx)
    
    # Compute correlations between different distance measures
    from scipy.stats import pearsonr, spearmanr
    
    # Key correlations
    corr_l2h_kl, p_l2h_kl = pearsonr(l2_hidden, kl_probs)
    corr_l2l_kl, p_l2l_kl = pearsonr(l2_logits, kl_probs)
    corr_l2h_fr, p_l2h_fr = pearsonr(l2_hidden, fisher_approx)
    corr_l2l_fr, p_l2l_fr = pearsonr(l2_logits, fisher_approx)
    corr_kl_fr, p_kl_fr = pearsonr(kl_probs, fisher_approx)
    corr_l2h_l2l, p_l2h_l2l = pearsonr(l2_hidden, l2_logits)
    
    # Spearman correlations
    sp_l2h_kl, _ = spearmanr(l2_hidden, kl_probs)
    sp_l2l_kl, _ = spearmanr(l2_logits, kl_probs)
    sp_l2h_fr, _ = spearmanr(l2_hidden, fisher_approx)
    sp_l2l_fr, _ = spearmanr(l2_logits, fisher_approx)
    
    # Bin analysis: binned by L2_hidden, compute mean KL
    n_bins = 20
    l2h_sorted_idx = np.argsort(l2_hidden)
    bin_size = len(l2h_sorted_idx) // n_bins
    
    bin_l2h_means = []
    bin_l2l_means = []
    bin_kl_means = []
    bin_fr_means = []
    
    for b in range(n_bins):
        start = b * bin_size
        end = (b + 1) * bin_size if b < n_bins - 1 else len(l2h_sorted_idx)
        idx_in_bin = l2h_sorted_idx[start:end]
        bin_l2h_means.append(float(np.mean(l2_hidden[idx_in_bin])))
        bin_l2l_means.append(float(np.mean(l2_logits[idx_in_bin])))
        bin_kl_means.append(float(np.mean(kl_probs[idx_in_bin])))
        bin_fr_means.append(float(np.mean(fisher_approx[idx_in_bin])))
    
    # Quantile analysis: "close in one space, far in another"
    l2h_med = np.median(l2_hidden)
    kl_med = np.median(kl_probs)
    
    close_h_far_kl = np.mean((l2_hidden < l2h_med) & (kl_probs > kl_med))
    far_h_close_kl = np.mean((l2_hidden > l2h_med) & (kl_probs < kl_med))
    close_l_far_kl = np.mean((l2_logits < np.median(l2_logits)) & (kl_probs > kl_med))
    far_l_close_kl = np.mean((l2_logits > np.median(l2_logits)) & (kl_probs < kl_med))
    
    print(f"  r(L2_hidden, KL) = {corr_l2h_kl:.3f}")
    print(f"  r(L2_logits, KL) = {corr_l2l_kl:.3f}")
    print(f"  r(L2_hidden, Fisher-Rao) = {corr_l2h_fr:.3f}")
    print(f"  r(L2_logits, Fisher-Rao) = {corr_l2l_fr:.3f}")
    print(f"  r(KL, Fisher-Rao) = {corr_kl_fr:.3f}")
    
    return {
        "n_pairs": n_pairs,
        "n_samples": n_sample,
        "distances_summary": {
            "l2_hidden_mean": float(np.mean(l2_hidden)),
            "l2_hidden_std": float(np.std(l2_hidden)),
            "l2_logits_mean": float(np.mean(l2_logits)),
            "l2_logits_std": float(np.std(l2_logits)),
            "kl_mean": float(np.mean(kl_probs)),
            "kl_std": float(np.std(kl_probs)),
            "fisher_rao_mean": float(np.mean(fisher_approx)),
            "fisher_rao_std": float(np.std(fisher_approx)),
        },
        "pearson_correlations": {
            "l2_hidden_vs_kl": float(corr_l2h_kl),
            "l2_logits_vs_kl": float(corr_l2l_kl),
            "l2_hidden_vs_fisher_rao": float(corr_l2h_fr),
            "l2_logits_vs_fisher_rao": float(corr_l2l_fr),
            "kl_vs_fisher_rao": float(corr_kl_fr),
            "l2_hidden_vs_l2_logits": float(corr_l2h_l2l),
        },
        "spearman_correlations": {
            "l2_hidden_vs_kl": float(sp_l2h_kl),
            "l2_logits_vs_kl": float(sp_l2l_kl),
            "l2_hidden_vs_fisher_rao": float(sp_l2h_fr),
            "l2_logits_vs_fisher_rao": float(sp_l2l_fr),
        },
        "mismatch_analysis": {
            "close_hidden_far_kl_pct": float(close_h_far_kl * 100),
            "far_hidden_close_kl_pct": float(far_h_close_kl * 100),
            "close_logits_far_kl_pct": float(close_l_far_kl * 100),
            "far_logits_close_kl_pct": float(far_l_close_kl * 100),
        },
        "binned_by_l2_hidden": {
            "l2_hidden_means": bin_l2h_means,
            "l2_logits_means": bin_l2l_means,
            "kl_means": bin_kl_means,
            "fisher_rao_means": bin_fr_means,
        },
    }


# ===== 4. EXP 2: FISHER INFORMATION STRUCTURE =====

def exp2_fisher_information(data, n_sample=500, top_k_dims=100):
    """Compute Fisher Information Matrix structure at sampled points"""
    logits = np.nan_to_num(data["logits"], nan=0.0, posinf=0.0, neginf=0.0)
    probs = np.nan_to_num(data["probs"], nan=1e-10, posinf=1.0, neginf=1e-10)
    
    N = len(logits)
    if N < 100:
        return {"error": "insufficient_data"}
    
    n_sample = min(n_sample, N)
    rng = np.random.RandomState(42)
    idx = rng.choice(N, n_sample, replace=False)
    
    L = logits[idx]  # [n_sample, vocab]
    P = probs[idx]   # [n_sample, vocab]
    
    vocab_size = L.shape[1]
    
    print(f"  Exp2: Computing Fisher Information Matrix for {n_sample} points (vocab={vocab_size})...")
    
    # Fisher Information Matrix for categorical distribution parameterized by logits:
    # G_{ij} = E[d log p / dθ_i * d log p / dθ_j] = sum_k p_k * (δ_{ik} - p_i)(δ_{jk} - p_j)
    # This equals: G = diag(p) - p*p^T
    # 
    # For computational efficiency, we use the approximation:
    # G ≈ diag(p) - p*p^T (exact for softmax)
    
    # However, vocab is ~150K, so full G is too large. We use top-K logit dimensions.
    # Strategy: reduce to top-K logits (by variance across samples)
    
    # Step 1: Find top-K logit dimensions by variance
    # Filter NaN in variance computation
    L_clean = np.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0)
    logit_var = np.var(L_clean, axis=0)
    top_dims = np.argsort(logit_var)[-top_k_dims:]
    L_reduced = L_clean[:, top_dims]  # [n_sample, K]
    
    # Step 2: For each point, compute the Fisher metric on the reduced space
    # G_reduced = J^T G J, where J is the Jacobian of reduced parameterization
    # Approximation: G_reduced ≈ diag(p_reduced) - p_reduced * p_reduced^T
    # But this is not exact because the reduced space is a sub-manifold.
    
    # Instead, we compute the Jacobian of the softmax w.r.t. the K selected logits:
    # ∂p_k / ∂θ_i = p_k(δ_{ki} - p_i) for i in top_dims
    
    # For efficiency, we compute Fisher eigenvalues using the structure:
    # G has rank K-1 (since probabilities sum to 1)
    # Eigenvalues of G = eigenvalues of diag(p) - pp^T
    
    fisher_eigenvalue_stats = []
    fisher_ranks = []
    fisher_condition_numbers = []
    
    batch_size = 50
    for batch_start in range(0, n_sample, batch_size):
        batch_end = min(batch_start + batch_size, n_sample)
        
        for b in range(batch_start, batch_end):
            p = P[b]
            # Fisher on full vocab: G = diag(p) - pp^T
            # Project onto top-K sub-space
            p_top = p[top_dims]  # [K]
            p_top = np.clip(p_top, 1e-10, 1.0)
            # Renormalize after clipping
            p_top = p_top / p_top.sum()
            
            # Check for NaN/Inf — skip this point
            if np.any(np.isnan(p_top)) or np.any(np.isinf(p_top)):
                continue
            
            # G_reduced = diag(p_top) - p_top * p_top^T
            G = np.diag(p_top) - np.outer(p_top, p_top)
            
            # Make symmetric and add regularization for numerical stability
            G = 0.5 * (G + G.T)
            reg = max(np.trace(np.abs(G)) * 1e-10, 1e-14)
            G += np.eye(len(G)) * reg
            
            # Eigenvalues with robust fallback
            try:
                eigvals = np.linalg.eigvalsh(G)
                if np.any(np.isnan(eigvals)) or np.any(np.isinf(eigvals)):
                    raise np.linalg.LinAlgError("NaN/Inf in eigenvalues")
            except (np.linalg.LinAlgError, ValueError):
                # Fallback: use scipy or simple diagonal approximation
                try:
                    from scipy.linalg import eigh
                    eigvals = eigh(G, eigvals_only=True)
                    if np.any(np.isnan(eigvals)):
                        raise np.linalg.LinAlgError("NaN in scipy eigenvalues")
                except:
                    # Last resort: just use diagonal entries
                    eigvals = np.diag(G).copy()
            eigvals = np.sort(np.real(eigvals))[::-1]  # descending, real part
            
            # Effective rank (number of eigenvalues > threshold)
            total_var = np.sum(eigvals)
            if total_var > 0:
                cumvar = np.cumsum(eigvals) / total_var
                eff_rank = int(np.searchsorted(cumvar, 0.95)) + 1
            else:
                eff_rank = 0
            
            # Condition number (max/min non-zero eigenvalue)
            pos_eigvals = eigvals[eigvals > 1e-12]
            if len(pos_eigvals) >= 2:
                cond = float(pos_eigvals[0] / pos_eigvals[-1])
            else:
                cond = float('inf')
            
            fisher_eigenvalue_stats.append({
                "top1": float(eigvals[0]) if len(eigvals) > 0 else 0,
                "top5_sum": float(np.sum(eigvals[:5])),
                "top10_sum": float(np.sum(eigvals[:10])),
                "top20_sum": float(np.sum(eigvals[:20])),
                "total_sum": float(total_var),
                "top1_pct": float(eigvals[0] / total_var * 100) if total_var > 0 else 0,
                "top5_pct": float(np.sum(eigvals[:5]) / total_var * 100) if total_var > 0 else 0,
                "top10_pct": float(np.sum(eigvals[:10]) / total_var * 100) if total_var > 0 else 0,
            })
            fisher_ranks.append(eff_rank)
            fisher_condition_numbers.append(cond)
    
    # Aggregate statistics
    eigenvalue_summary = {}
    for key in fisher_eigenvalue_stats[0].keys():
        vals = [s[key] for s in fisher_eigenvalue_stats]
        eigenvalue_summary[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
        }
    
    # Fisher effective rank distribution
    rank_counts = defaultdict(int)
    for r in fisher_ranks:
        rank_counts[int(r)] += 1
    
    print(f"  Fisher effective rank: mean={np.mean(fisher_ranks):.1f}, median={np.median(fisher_ranks):.1f}")
    print(f"  Top-1 eigenvalue %: {eigenvalue_summary['top1_pct']['mean']:.1f}%")
    print(f"  Top-5 eigenvalue %: {eigenvalue_summary['top5_pct']['mean']:.1f}%")
    print(f"  Top-10 eigenvalue %: {eigenvalue_summary['top10_pct']['mean']:.1f}%")
    
    # PCA of logit space (for comparison with Fisher structure)
    from sklearn.decomposition import PCA
    pca_logit = PCA(n_components=min(top_k_dims, 50))
    L_centered = L_reduced - L_reduced.mean(axis=0)
    pca_logit.fit(L_centered)
    
    pca_var_explained = pca_logit.explained_variance_ratio_[:20].tolist()
    
    # Compare: PCA variance concentration vs Fisher eigenvalue concentration
    # If PCA concentrates more → logit space is more "Euclidean low-dim"
    # If Fisher concentrates more → probability manifold is more "intrinsically low-dim"
    
    return {
        "n_samples": n_sample,
        "top_k_dims": top_k_dims,
        "fisher_eigenvalue_summary": eigenvalue_summary,
        "fisher_effective_rank": {
            "mean": float(np.mean(fisher_ranks)),
            "std": float(np.std(fisher_ranks)),
            "median": float(np.median(fisher_ranks)),
            "distribution": {str(k): v for k, v in sorted(rank_counts.items())},
        },
        "condition_number": {
            "mean": float(np.mean([c for c in fisher_condition_numbers if c != float('inf')])),
            "median": float(np.median([c for c in fisher_condition_numbers if c != float('inf')])),
        },
        "logit_pca_variance_explained": pca_var_explained,
        "comparison_pca_vs_fisher": {
            "pca_top5_pct": float(sum(pca_var_explained[:5]) * 100),
            "pca_top10_pct": float(sum(pca_var_explained[:10]) * 100),
            "fisher_top5_pct_mean": eigenvalue_summary["top5_pct"]["mean"],
            "fisher_top10_pct_mean": eigenvalue_summary["top10_pct"]["mean"],
        },
    }


# ===== 5. EXP 3: LOCAL VS GLOBAL DIMENSIONALITY =====

def exp3_local_vs_global_dimension(data, domain_labels, n_per_domain=200):
    """Compute local intrinsic dimension for each semantic domain"""
    logits = np.nan_to_num(data["logits"], nan=0.0, posinf=0.0, neginf=0.0)
    probs = np.nan_to_num(data["probs"], nan=1e-10, posinf=1.0, neginf=1e-10)
    hidden = np.nan_to_num(data["hidden"], nan=0.0, posinf=0.0, neginf=0.0)
    
    results = {}
    
    for domain in ["narrative", "technical", "dialogue", "descriptive"]:
        domain_idx = [i for i, d in enumerate(domain_labels) if d == domain]
        if len(domain_idx) < 50:
            results[domain] = {"error": "insufficient_samples", "n": len(domain_idx)}
            continue
        
        n_use = min(n_per_domain, len(domain_idx))
        rng = np.random.RandomState(42)
        sel_idx = rng.choice(domain_idx, n_use, replace=False)
        
        L = logits[sel_idx]
        P = probs[sel_idx]
        H = hidden[sel_idx]
        
        print(f"  Exp3: {domain} domain, n={n_use}")
        
        # Method 1: Local PCA dimension in logit space
        # Use top-500 logit dims by variance
        logit_var = np.var(L, axis=0)
        top_k = min(500, L.shape[1])
        top_dims = np.argsort(logit_var)[-top_k:]
        L_red = L[:, top_dims]
        
        from sklearn.decomposition import PCA
        pca_logit = PCA(n_components=min(50, top_k))
        pca_logit.fit(L_red - L_red.mean(axis=0))
        var_explained = pca_logit.explained_variance_ratio_
        
        # Intrinsic dimension at 90% and 95% variance
        cumvar = np.cumsum(var_explained)
        dim_90 = int(np.searchsorted(cumvar, 0.90)) + 1
        dim_95 = int(np.searchsorted(cumvar, 0.95)) + 1
        dim_99 = int(np.searchsorted(cumvar, 0.99)) + 1 if cumvar[-1] >= 0.99 else len(cumvar)
        
        # Method 2: Local PCA dimension in hidden state space
        pca_hidden = PCA(n_components=min(50, H.shape[1]))
        pca_hidden.fit(H - H.mean(axis=0))
        var_explained_h = pca_hidden.explained_variance_ratio_
        cumvar_h = np.cumsum(var_explained_h)
        dim_90_h = int(np.searchsorted(cumvar_h, 0.90)) + 1
        dim_95_h = int(np.searchsorted(cumvar_h, 0.95)) + 1
        dim_99_h = int(np.searchsorted(cumvar_h, 0.99)) + 1 if cumvar_h[-1] >= 0.99 else len(cumvar_h)
        
        # Method 3: Two-NN estimator (intrinsic dimension estimator)
        # This is a non-parametric method that estimates intrinsic dimension
        # from the ratio of distances to 1st and 2nd nearest neighbors
        from scipy.spatial.distance import pdist, squareform
        
        # Use logit space for Two-NN
        # Subsample for efficiency
        n_2nn = min(200, len(L_red))
        idx_2nn = rng.choice(len(L_red), n_2nn, replace=False)
        L_2nn = L_red[idx_2nn]
        
        D = squareform(pdist(L_2nn, 'euclidean'))
        np.fill_diagonal(D, np.inf)
        
        # Ratio of 2nd-NN to 1st-NN distances
        mu_ratios = []
        for i in range(len(D)):
            sorted_dists = np.sort(D[i])
            if sorted_dists[0] > 1e-10 and sorted_dists[1] > 1e-10:
                mu_ratios.append(sorted_dists[1] / sorted_dists[0])
        
        mu_ratios = np.array(mu_ratios)
        # Intrinsic dimension d = -1 / <log(mu)> 
        # (Maximum Likelihood estimator from Two-NN)
        if len(mu_ratios) > 10 and np.all(mu_ratios > 0):
            d_intrinsic = -1.0 / np.mean(np.log(mu_ratios))
        else:
            d_intrinsic = -1
        
        # Method 4: Local KL neighborhood structure
        # For each point, compute KL to its k nearest neighbors in hidden space
        # Then compute KL to k random points
        # If local KL is much smaller than random KL → local manifold structure
        n_local = min(100, len(P))
        idx_local = rng.choice(len(P), n_local, replace=False)
        P_local = P[idx_local]
        H_local = H[idx_local]
        
        D_hidden = squareform(pdist(H_local, 'euclidean'))
        np.fill_diagonal(D_hidden, np.inf)
        
        k = 5
        local_kl_means = []
        random_kl_means = []
        
        for i in range(n_local):
            # k nearest neighbors in hidden space
            nn_idx = np.argsort(D_hidden[i])[:k]
            
            # KL to nearest neighbors
            kl_nn = []
            for j in nn_idx:
                p, q = P_local[i], P_local[j]
                p = np.clip(p, 1e-10, 1.0); p /= p.sum()
                q = np.clip(q, 1e-10, 1.0); q /= q.sum()
                kl = 0.5 * (np.sum(p * np.log(p/q)) + np.sum(q * np.log(q/p)))
                kl_nn.append(kl)
            local_kl_means.append(np.mean(kl_nn))
            
            # KL to k random points
            rand_idx = rng.choice(n_local, k, replace=False)
            kl_rand = []
            for j in rand_idx:
                p, q = P_local[i], P_local[j]
                p = np.clip(p, 1e-10, 1.0); p /= p.sum()
                q = np.clip(q, 1e-10, 1.0); q /= q.sum()
                kl = 0.5 * (np.sum(p * np.log(p/q)) + np.sum(q * np.log(q/p)))
                kl_rand.append(kl)
            random_kl_means.append(np.mean(kl_rand))
        
        local_kl_means = np.array(local_kl_means)
        random_kl_means = np.array(random_kl_means)
        kl_ratio = np.mean(random_kl_means) / max(np.mean(local_kl_means), 1e-10)
        
        results[domain] = {
            "n_samples": n_use,
            "logit_pca_dim": {
                "dim_90": dim_90,
                "dim_95": dim_95,
                "dim_99": dim_99,
                "top5_var_pct": float(np.sum(var_explained[:5]) * 100),
                "top10_var_pct": float(np.sum(var_explained[:10]) * 100),
            },
            "hidden_pca_dim": {
                "dim_90": dim_90_h,
                "dim_95": dim_95_h,
                "dim_99": dim_99_h,
                "top5_var_pct": float(np.sum(var_explained_h[:5]) * 100),
                "top10_var_pct": float(np.sum(var_explained_h[:10]) * 100),
            },
            "two_nn_intrinsic_dim": float(d_intrinsic) if d_intrinsic > 0 else None,
            "local_kl_structure": {
                "local_kl_mean": float(np.mean(local_kl_means)),
                "local_kl_std": float(np.std(local_kl_means)),
                "random_kl_mean": float(np.mean(random_kl_means)),
                "random_kl_std": float(np.std(random_kl_means)),
                "ratio_random_to_local": float(kl_ratio),
            },
        }
        
        print(f"    Logit PCA: dim90={dim_90}, dim95={dim_95}")
        print(f"    Hidden PCA: dim90={dim_90_h}, dim95={dim_95_h}")
        print(f"    Two-NN intrinsic dim: {d_intrinsic:.1f}" if d_intrinsic > 0 else "    Two-NN: N/A")
        print(f"    Local KL structure: ratio={kl_ratio:.2f}")
    
    # Cross-domain comparison: compute pairwise distances between domain centroids
    domain_centroids_logit = {}
    domain_centroids_hidden = {}
    for domain in results:
        if "error" in results[domain]:
            continue
        domain_idx = [i for i, d in enumerate(domain_labels) if d == domain]
        domain_centroids_logit[domain] = np.mean(logits[domain_idx], axis=0)
        domain_centroids_hidden[domain] = np.mean(hidden[domain_idx], axis=0)
    
    cross_domain = {}
    domain_names = list(domain_centroids_logit.keys())
    for i, d1 in enumerate(domain_names):
        for j, d2 in enumerate(domain_names):
            if i < j:
                # L2 distance between centroids (hidden space)
                l2_h = float(np.linalg.norm(domain_centroids_hidden[d1] - domain_centroids_hidden[d2]))
                
                # KL between average probability distributions
                p1 = np.mean(probs[[i for i, d in enumerate(domain_labels) if d == d1]], axis=0)
                p2 = np.mean(probs[[i for i, d in enumerate(domain_labels) if d == d2]], axis=0)
                p1 = np.clip(p1, 1e-10, 1.0); p1 /= p1.sum()
                p2 = np.clip(p2, 1e-10, 1.0); p2 /= p2.sum()
                kl_cross = 0.5 * (np.sum(p1 * np.log(p1/p2)) + np.sum(p2 * np.log(q/p1))) if False else \
                           float(np.sum(p1 * np.log(p1/p2)) + np.sum(p2 * np.log(p2/p1))) / 2
                
                # Fisher-Rao
                sqrt_p1p2 = np.sqrt(p1 * p2)
                inner = np.clip(np.sum(sqrt_p1p2), -1.0, 1.0)
                fr_cross = 2.0 * np.arccos(inner)
                
                cross_domain[f"{d1}_vs_{d2}"] = {
                    "l2_hidden": l2_h,
                    "kl_symmetric": float(kl_cross),
                    "fisher_rao": float(fr_cross),
                }
    
    results["cross_domain"] = cross_domain
    
    return results


# ===== 6. EXP 4: PREDICTIVE STATE COMPRESSION IN LOGIT SPACE =====

def exp4_logit_predictive_compression(data, dims_to_test=None):
    """Compress logits (not hidden states) to find true predictive dimension"""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    
    logits = np.nan_to_num(data["logits"], nan=0.0, posinf=0.0, neginf=0.0)
    hidden = np.nan_to_num(data["hidden"], nan=0.0, posinf=0.0, neginf=0.0)
    
    N = len(logits)
    if N < 200:
        return {"error": "insufficient_data"}
    
    # We need transitions: (logits_t, hidden_t) → logits_{t+1}
    # For this, we need to pair consecutive positions
    # Since we collected all positions sequentially, adjacent entries are transitions
    # But they might be from different prompts, so we need position info
    
    # Alternative: use logits to predict next-token logits
    # This is a more direct test of predictive compression
    
    # For logit-space analysis, reduce dimensionality first
    # Use top-K logit dims by variance
    top_k = min(500, logits.shape[1])
    logit_var = np.var(logits, axis=0)
    top_dims = np.argsort(logit_var)[-top_k:]
    L = logits[:, top_dims]  # [N, K]
    
    # Also do PCA of hidden state space
    H = hidden
    
    if dims_to_test is None:
        dims_to_test = [2, 5, 10, 15, 20, 30, 50, 75, 100, 150]
    
    print(f"  Exp4: Logit predictive compression, N={N}, top_k_logit_dims={top_k}")
    
    # Full PCA in both spaces
    pca_logit = PCA(n_components=min(150, top_k))
    L_centered = L - L.mean(axis=0)
    pca_logit.fit(L_centered)
    Z_logit = pca_logit.transform(L_centered)  # [N, 150]
    
    pca_hidden = PCA(n_components=min(200, H.shape[1]))
    H_centered = H - H.mean(axis=0)
    pca_hidden.fit(H_centered)
    Z_hidden = pca_hidden.transform(H_centered)  # [N, 200]
    
    # For next-token prediction, we need logits_t → logits_{t+1} transitions
    # Since we process prompts sequentially, consecutive positions from same prompt
    # form natural transitions. But we don't have prompt boundaries here.
    # 
    # Alternative approach: use the logit space itself as the target.
    # Measure how well d-dimensional compression preserves the logit information.
    # This is a "self-prediction" task: how many logit dimensions are needed
    # to reconstruct the full logit vector?
    
    # Better: use the information bottleneck approach from Phase 161
    # But now in LOGIT space, not hidden state space
    
    # Task: predict next-token logits from current logit PCA projection
    # We need to construct transitions
    
    # Since we can't reliably pair consecutive tokens across prompts,
    # we use a different measure: reconstruction quality
    
    # Measure 1: Logit reconstruction R² at dimension d
    # This measures how well d logit-PCA dimensions reconstruct the full logit vector
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}
    
    for d in dims_to_test:
        pca_logit_r2_folds = []
        pca_hidden_r2_folds = []
        
        for train_idx, test_idx in kf.split(range(N)):
            # Logit PCA bottleneck: project to d PCA dims, reconstruct
            Z_l_train = Z_logit[train_idx, :d]
            Z_l_test = Z_logit[test_idx, :d]
            L_test = L_centered[test_idx]
            
            # Reconstruction: L_hat = Z_d @ V_d^T
            V_d = pca_logit.components_[:d]  # [d, K]
            L_recon = Z_l_test @ V_d  # [n_test, K]
            
            # R² for reconstruction
            ss_res = np.sum((L_test - L_recon) ** 2)
            ss_tot = np.sum(L_test ** 2)
            r2_recon = 1.0 - ss_res / max(ss_tot, 1e-10)
            pca_logit_r2_folds.append(float(r2_recon))
            
            # Hidden PCA bottleneck: project to d PCA dims, then predict logits
            Z_h_train = Z_hidden[train_idx, :d]
            Z_h_test = Z_hidden[test_idx, :d]
            L_train = L_centered[train_idx]
            
            reg = Ridge(alpha=1.0)
            reg.fit(Z_h_train, L_train)
            r2_pred = reg.score(Z_h_test, L_test)
            pca_hidden_r2_folds.append(float(r2_pred))
        
        results[d] = {
            "logit_pca_reconstruction_r2": {
                "mean": float(np.mean(pca_logit_r2_folds)),
                "std": float(np.std(pca_logit_r2_folds)),
            },
            "hidden_pca_to_logit_r2": {
                "mean": float(np.mean(pca_hidden_r2_folds)),
                "std": float(np.std(pca_hidden_r2_folds)),
            },
        }
        
        print(f"    d={d}: logit_recon_R²={np.mean(pca_logit_r2_folds):.3f}, "
              f"hidden_to_logit_R²={np.mean(pca_hidden_r2_folds):.3f}")
    
    # Find compression cliffs
    logit_cliff_d = None
    hidden_cliff_d = None
    prev_logit_r2 = 0
    prev_hidden_r2 = 0
    for d in sorted(results.keys()):
        lr2 = results[d]["logit_pca_reconstruction_r2"]["mean"]
        hr2 = results[d]["hidden_pca_to_logit_r2"]["mean"]
        if logit_cliff_d is None and lr2 > 0.8:
            logit_cliff_d = d
        if hidden_cliff_d is None and hr2 > 0.5:
            hidden_cliff_d = d
        prev_logit_r2 = lr2
        prev_hidden_r2 = hr2
    
    results["summary"] = {
        "logit_reconstruction_cliff_d": logit_cliff_d,
        "hidden_to_logit_cliff_d": hidden_cliff_d,
    }
    
    return results


# ===== 7. MAIN =====

def run_experiment(model_name):
    """Run all Phase 162 experiments for a single model"""
    print(f"\n{'='*60}")
    print(f"Phase 162: Logits Manifold & Fisher Geometry — {model_name}")
    print(f"{'='*60}")
    
    # Load model
    print(f"\n[1] Loading {model_name}...")
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    print(f"  Model: {info.model_class}, {info.n_layers}L, d={info.d_model}, vocab={info.vocab_size}")
    
    # Generate domain-specific prompts
    print(f"\n[2] Generating domain-specific prompts...")
    domain_prompts = generate_domain_prompts(n_per_domain=150)
    all_prompts = []
    domain_labels = []
    for domain, prompts in domain_prompts.items():
        all_prompts.extend(prompts)
        domain_labels.extend([domain] * len(prompts))
    print(f"  Total: {len(all_prompts)} prompts across {len(domain_prompts)} domains")
    
    # Collect data
    print(f"\n[3] Collecting logits and hidden states...")
    t0 = time.time()
    data = collect_logits_and_hidden_states(model, tokenizer, device, all_prompts, info)
    t_collect = time.time() - t0
    print(f"  Collected {len(data['logits'])} positions in {t_collect:.1f}s")
    
    if len(data['logits']) < 100:
        print(f"  ERROR: Insufficient data ({len(data['logits'])} positions)")
        release_model(model)
        return {"model_name": model_name, "error": "insufficient_data"}
    
    # Run experiments
    print(f"\n[4] Exp 1: Logits Geometry vs Hidden State Geometry...")
    t0 = time.time()
    exp1_results = exp1_logits_vs_hidden_geometry(data)
    t1 = time.time() - t0
    print(f"  Done in {t1:.1f}s")
    
    print(f"\n[5] Exp 2: Fisher Information Structure...")
    t0 = time.time()
    exp2_results = exp2_fisher_information(data)
    t2 = time.time() - t0
    print(f"  Done in {t2:.1f}s")
    
    print(f"\n[6] Exp 3: Local vs Global Dimensionality...")
    t0 = time.time()
    exp3_results = exp3_local_vs_global_dimension(data, domain_labels)
    t3 = time.time() - t0
    print(f"  Done in {t3:.1f}s")
    
    print(f"\n[7] Exp 4: Logit Predictive Compression...")
    t0 = time.time()
    exp4_results = exp4_logit_predictive_compression(data)
    t4 = time.time() - t0
    print(f"  Done in {t4:.1f}s")
    
    # Release model
    release_model(model)
    
    # Compile results
    results = {
        "model_name": model_name,
        "model_info": {
            "class": info.model_class,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
            "vocab_size": info.vocab_size,
        },
        "n_total_positions": len(data["logits"]),
        "exp1_logits_vs_hidden_geometry": exp1_results,
        "exp2_fisher_information": exp2_results,
        "exp3_local_vs_global_dimension": exp3_results,
        "exp4_logit_predictive_compression": exp4_results,
        "timing": {
            "collection_s": round(t_collect, 1),
            "exp1_s": round(t1, 1),
            "exp2_s": round(t2, 1),
            "exp3_s": round(t3, 1),
            "exp4_s": round(t4, 1),
        },
    }
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_path = f"tests/glm5_temp/phase162_{model_name}_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  Results saved to: {out_path}")
    
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python phase162_logits_manifold.py <model_name>")
        print("  model_name: qwen3, glm4, deepseek7b")
        sys.exit(1)
    
    model_name = sys.argv[1]
    if model_name not in ("qwen3", "glm4", "deepseek7b"):
        print(f"Unknown model: {model_name}")
        sys.exit(1)
    
    run_experiment(model_name)
