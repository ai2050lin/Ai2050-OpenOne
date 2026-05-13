"""
Phase 161: Predictive Information Closure & Causal Dynamics
============================================================

Based on user's Phase 160 critique — the MOST CRITICAL theoretical pivot:

Core insight from user:
  "What information must be preserved for the future to remain predictable?"
  This is fundamentally different from "how does hidden state change?"

User's 5 key corrections:
  1. Hidden state ≠ State — hidden state is a "dynamic workspace/cache"
  2. True state = Predictive equivalence class — P(·|h_a) ≈ P(·|h_b) → same state
  3. Local low-dim ≠ Global low-dim — we sample from a tiny region of language space
  4. Token is self-conditioning — system generates its own input (autoregressive)
  5. Need information-theoretic measures — I(z_t; future), not Var(z)

User's 4 Routes to the mathematical core:
  Route 1: Predictive state learning (z_t = φ(h_t), optimize z_{t+1} ≈ f(z_t, u_t))
  Route 2: Future equivalence classes (P(·|h_a) ≈ P(·|h_b) → same state)
  Route 3: Causal perturbation (along Pred-direction vs PCA-direction)
  Route 4: State compression rate (bits needed to predict future)

CRITICAL: The user warns against:
  - Assuming hidden state IS the state
  - Assuming low-dim manifold is GLOBAL (not just local)
  - Treating token as external input (it's self-conditioned)
  - Using geometric measures instead of predictive/information measures

Experiments:
  Exp 1: Predictive Equivalence Classes (Route 2)
    - For pairs of hidden states, measure KL(P(·|h_a) || P(·|h_b))
    - Compare with Euclidean distance ||h_a - h_b||
    - Key question: do geometrically close states predict similar futures?
    - Estimate number of effective predictive states

  Exp 2: Causal Perturbation (Route 3) — THE DEFINITIVE TEST
    - Perturb h along Pred-directions vs PCA-directions at final layer
    - Measure KL divergence between original and perturbed output distributions
    - If Pred-perturbation causes more output change → Pred-directions are causally relevant
    - If PCA-perturbation causes little change despite large geometric shift → PCA is "workspace noise"

  Exp 3: Information Bottleneck State Learning (Route 1 & 4)
    - Train z = φ(h) with bottleneck dimension d
    - Optimize: predict next-token logits from z
    - Find the "compression cliff" — where adding more dimensions stops helping prediction
    - This directly estimates the true state dimension

  Exp 4: Self-Conditioning Verification
    - Compare: model's own predicted token vs random token injection
    - Measure how much h(t+1) depends on WHICH specific token is injected
    - This tests whether the system is truly "self-conditioning"

Usage: python tests/glm5/phase161_predictive_closure.py <model_name>
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


# ===== 1. DIVERSE PROMPT GENERATION (longer, more diverse) =====

def generate_diverse_prompts(n=600):
    """Generate diverse text prompts (15-40 tokens each) for trajectory analysis"""
    nouns = [
        "cat", "dog", "bird", "fish", "child", "woman", "man", "boy", "girl", "teacher",
        "doctor", "student", "tree", "flower", "river", "mountain", "car", "book", "house", "city",
        "king", "queen", "soldier", "artist", "scientist", "writer", "musician", "engineer", "farmer", "driver",
        "chair", "table", "door", "window", "road", "bridge", "tower", "castle", "garden", "forest",
        "ocean", "island", "valley", "desert", "planet", "star", "moon", "sun", "cloud", "rain",
        "apple", "bread", "water", "fire", "stone", "glass", "paper", "silk", "gold", "iron",
        "village", "library", "kitchen", "market", "harbor", "cathedral", "museum", "theater", "palace", "temple",
    ]
    verbs = [
        "runs", "sits", "walks", "jumps", "flies", "swims", "reads", "writes", "eats", "drinks",
        "sings", "dances", "sleeps", "wakes", "falls", "rises", "moves", "stops", "starts", "ends",
        "opens", "closes", "grows", "shrinks", "turns", "stands", "lies", "hangs", "breaks", "builds",
        "discovers", "analyzes", "creates", "observes", "remembers", "forgets", "believes", "doubts", "loves", "hates",
    ]
    adjs = [
        "red", "blue", "green", "big", "small", "old", "new", "fast", "slow", "tall",
        "short", "dark", "bright", "warm", "cold", "soft", "hard", "rich", "poor", "young",
        "heavy", "light", "thick", "thin", "wide", "narrow", "deep", "shallow", "clean", "dirty",
        "ancient", "modern", "beautiful", "strange", "powerful", "gentle", "fierce", "calm", "wild", "quiet",
    ]
    places = [
        "park", "house", "school", "office", "garden", "street", "market", "church", "field", "lake",
        "beach", "harbor", "airport", "station", "hospital", "museum", "theater", "prison", "palace", "temple",
    ]
    advs = ["quickly", "slowly", "carefully", "quietly", "loudly", "gently", "suddenly",
            "happily", "sadly", "gracefully", "boldly", "silently", "proudly", "bravely"]
    connectors = ["and", "but", "or", "so", "yet", "while", "although", "because", "since", "when"]
    prep = ["in", "on", "at", "by", "with", "from", "to", "near", "beside", "under"]

    prompts = []
    seen = set()

    def add(p):
        if p not in seen and len(p.split()) >= 6:
            seen.add(p)
            prompts.append(p)

    # Pattern 1: Multi-clause sentences (longer)
    for nn in nouns[:25]:
        for v in verbs[:8]:
            for c in connectors[:6]:
                add(f"The {nn} {v} {c} the people in the")
                add(f"A {nn} {v} {c} the old {nn} was")

    # Pattern 2: Adjective-noun-verb-prep
    for a in adjs[:12]:
        for nn in nouns[:12]:
            for v in verbs[:6]:
                add(f"The {a} {nn} {v} toward the edge of the")
                add(f"A {a} {nn} {v} away from the old")

    # Pattern 3: Narrative with conjunctions
    for nn in nouns[:15]:
        for v in verbs[:6]:
            add(f"After the {nn} {v} the other people started to")
            add(f"Before the {nn} {v} the group had already")
            add(f"When the {nn} {v} everyone in the room")

    # Pattern 4: Questions with context
    for nn in nouns[:12]:
        for v in verbs[:5]:
            add(f"Does the {nn} {v} when the weather gets cold in the")
            add(f"Will the {nn} {v} if the conditions change in the")

    # Pattern 5: Negation and contrast
    for nn in nouns[:12]:
        for v in verbs[:5]:
            add(f"The {nn} does not {v} but the other one does in the")
            add(f"Although the {nn} {v} the result was not what they")

    # Pattern 6: Complex conditionals
    for nn in nouns[:10]:
        for v in verbs[:4]:
            add(f"If the {nn} {v} then the system will have to")
            add(f"Unless the {nn} {v} the plan will not succeed in the")

    # Pattern 7: Location + description
    for nn in nouns[:10]:
        for p in places[:8]:
            add(f"The {nn} in the {p} was very different from the one in the")
            add(f"In the {p} the {nn} was known for its ability to")

    # Pattern 8: Scientific/expository
    for nn in nouns[:12]:
        add(f"Research shows that the {nn} can be found in many different parts of the")
        add(f"Scientists discovered that {nn}s are essential for the function of the")
        add(f"The study of {nn} reveals that it has a significant impact on the")

    # Pattern 9: Longer narrative
    for a in adjs[:8]:
        for nn in nouns[:8]:
            add(f"Once there was a {a} {nn} who lived in a small village near the")
            add(f"In the land of the {a} {nn} there was a great river that flowed")

    # Pattern 10: Descriptive with multiple adjectives
    for a1 in adjs[:6]:
        for a2 in adjs[:6]:
            if a1 != a2:
                for nn in nouns[:6]:
                    add(f"The {a1} and {a2} {nn} stood near the old building by the")

    # Pattern 11: Temporal sequences
    for nn in nouns[:10]:
        for v in verbs[:5]:
            add(f"First the {nn} {v} and then it started to move toward the")
            add(f"After the {nn} {v} the next thing that happened was that the")

    # Pattern 12: Comparative
    for a1 in adjs[:6]:
        for a2 in adjs[:6]:
            if a1 != a2:
                add(f"The {a1} one was better than the {a2} one because it had more")

    # Pattern 13: Passive voice
    for nn in nouns[:10]:
        for v in verbs[:5]:
            vb = v.rstrip("s")
            add(f"The {nn} was {vb}ed by the group of people who were in the")

    # Pattern 14: Existential
    for a in adjs[:8]:
        for nn in nouns[:8]:
            add(f"There was a {a} {nn} that everyone in the village knew about and")

    # Pattern 15: Longer compound sentences
    for nn1 in nouns[:8]:
        for nn2 in nouns[:8]:
            if nn1 != nn2:
                add(f"The {nn1} and the {nn2} were both trying to reach the other side of the")

    print(f"[prompts] Generated {len(prompts)} unique prompts")
    return prompts[:n]


# ===== 2. DATA COLLECTION =====

def collect_full_data(model, tokenizer, device, model_info, prompts):
    """
    Collect hidden states at ALL token positions, ALL sampled layers.
    Also collect output logits for predictive equivalence analysis.
    
    Returns:
        H_all: dict of {layer_idx: (N_total, d_model)} — hidden states per layer
        logits_all: (N_total, vocab_top_k) — output logits for top-k tokens
        token_ids: list of arrays
        embeddings: list of arrays
        H_final: (N_total, d_model) — final layer hidden states
        normed_final: (N_total, d_model) — final hidden states after LayerNorm
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    # Sample layers for analysis
    if n_layers <= 10:
        sample_layers = list(range(n_layers))
    else:
        # Uniform sampling + key layers
        step = n_layers // 8
        sample_layers = sorted(set(list(range(0, n_layers, step)) + [n_layers // 2, n_layers - 1]))
    
    print(f"  Sampling {len(sample_layers)} layers: {sample_layers}")
    
    try:
        input_device = next(model.parameters()).device
    except StopIteration:
        input_device = device
    
    embed_layer = model.get_input_embeddings()
    
    # Collect data
    H_per_layer = defaultdict(list)  # layer_idx -> list of (T, d_model)
    logits_list = []  # (N_total, vocab_size) — but we'll only store top-k
    token_ids_list = []
    embeddings_list = []
    H_final_list = []  # Final layer hidden states (before LN)
    normed_final_list = []  # After LN
    
    TOP_K = 100  # Only store top-100 logits for memory efficiency
    failed = 0
    n_prompts = len(prompts)
    
    # Get lm_head and final norm for computing logits from hidden states
    W_U = None
    
    for i, prompt in enumerate(prompts):
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            seq_len = input_ids.shape[1]
            
            if seq_len < 5:
                failed += 1
                continue
            
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
            
            # Extract hidden states at sampled layers
            for li in sample_layers:
                h_layer = out.hidden_states[li][0, :, :].detach().float().cpu().numpy()
                if not (np.any(np.isnan(h_layer)) or np.any(np.isinf(h_layer))):
                    H_per_layer[li].append(h_layer)
            
            # Final layer hidden states (for predictive equivalence)
            h_final = out.hidden_states[n_layers][0, :, :].detach().float().cpu().numpy()
            if np.any(np.isnan(h_final)) or np.any(np.isinf(h_final)):
                failed += 1
                del out
                continue
            
            # Compute logits from final hidden state
            # logits = lm_head(norm(h_final))
            # We'll compute this later using the model's forward
            # For now, just use the model's logits output
            if hasattr(out, 'logits') and out.logits is not None:
                logits_all_pos = out.logits[0, :, :].detach().float().cpu().numpy()  # (T, vocab)
                # Only store top-k for each position
                top_k_indices = np.argsort(logits_all_pos, axis=-1)[:, -TOP_K:]
                top_k_values = np.take_along_axis(logits_all_pos, top_k_indices, axis=-1)
                logits_list.append((top_k_indices, top_k_values))  # list of (top_k_idx, top_k_val)
            else:
                logits_list.append(None)
            
            H_final_list.append(h_final)
            
            # Token IDs and embeddings
            tok_ids = input_ids[0].detach().cpu().numpy()
            with torch.no_grad():
                emb = embed_layer(input_ids)[0, :, :].detach().float().cpu().numpy()
            
            token_ids_list.append(tok_ids)
            embeddings_list.append(emb)
            
            del out
            
            if (i + 1) % 100 == 0:
                total_toks = sum(h.shape[0] for h in H_final_list)
                print(f"  [{i+1}/{n_prompts}] collected, total tokens: {total_toks}")
                
        except Exception as e:
            failed += 1
            if i < 5:
                print(f"  Warning: prompt {i} failed: {e}")
    
    # Stack final hidden states
    H_final = np.vstack(H_final_list) if H_final_list else np.array([])
    
    # Stack per-layer hidden states
    H_all = {}
    for li in sample_layers:
        if H_per_layer[li]:
            H_all[li] = np.vstack(H_per_layer[li])
    
    n_successful = len(H_final_list)
    total_toks = H_final.shape[0] if len(H_final) > 0 else 0
    print(f"[collect] {n_successful} successful, {failed} failed, total tokens: {total_toks}")
    
    return {
        'H_final': H_final,
        'H_all': H_all,
        'logits_list': logits_list,
        'token_ids': token_ids_list,
        'embeddings': embeddings_list,
        'H_final_list': H_final_list,
        'sample_layers': sample_layers,
        'n_successful': n_successful,
        'total_toks': total_toks,
    }


# ===== 3. EXPERIMENTS =====

def exp1_predictive_equivalence(model, tokenizer, device, model_info, data):
    """
    Exp 1: Predictive Equivalence Classes (Route 2)
    
    CORE QUESTION: Are geometrically close states also predictively similar?
    
    Method:
      1. For each hidden state h, compute P(next_token | h) using lm_head
      2. For pairs of states, measure:
         - L2 distance ||h_a - h_b||
         - KL divergence KL(P(·|h_a) || P(·|h_b))
      3. Plot the relationship
      4. Estimate number of effective predictive states
    
    If geometric proximity ≈ predictive similarity → hidden state IS the state
    If they're independent → hidden state is a "workspace" not a "state"
    """
    print("\n" + "=" * 60)
    print("Exp 1: Predictive Equivalence Classes")
    print("=" * 60)
    
    H_final = data['H_final']
    if len(H_final) < 100:
        print("  WARNING: Not enough data for reliable analysis")
        return {"error": "insufficient_data"}
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    # Compute logits from final hidden states using the model's lm_head and norm
    # We need: logits = lm_head(norm(h_final))
    try:
        final_norm = model.model.norm  # Final LayerNorm
        lm_head = model.lm_head
    except AttributeError:
        print("  WARNING: Cannot access model.model.norm or lm_head directly")
        return {"error": "model_architecture_issue"}
    
    # Process in batches to avoid OOM
    # Determine model dtype from lm_head weight
    try:
        model_dtype = next(lm_head.parameters()).dtype
    except StopIteration:
        model_dtype = torch.bfloat16
    
    batch_size = 200
    all_logits = []
    
    for start in range(0, len(H_final), batch_size):
        end = min(start + batch_size, len(H_final))
        h_batch = torch.tensor(H_final[start:end], dtype=model_dtype)
        
        with torch.no_grad():
            # Apply final LayerNorm
            h_normed = final_norm(h_batch.to(device))
            # Compute logits
            logits = lm_head(h_normed)  # (batch, vocab)
            all_logits.append(logits.detach().float().cpu().numpy())
    
    all_logits = np.vstack(all_logits)  # (N, vocab)
    print(f"  Computed logits shape: {all_logits.shape}")
    
    # Compute log-probabilities (numerically stable)
    # Subtract max for stability
    logits_max = np.max(all_logits, axis=-1, keepdims=True)
    log_sum_exp = logits_max + np.log(np.sum(np.exp(all_logits - logits_max), axis=-1, keepdims=True))
    log_probs = all_logits - log_sum_exp  # (N, vocab)
    
    # Sample pairs and compute KL + L2
    N = len(H_final)
    n_pairs = min(5000, N * (N - 1) // 2)
    
    # Random sampling of pairs
    np.random.seed(42)
    idx_a = np.random.randint(0, N, size=n_pairs)
    idx_b = np.random.randint(0, N, size=n_pairs)
    # Ensure different states
    mask = idx_a != idx_b
    idx_a = idx_a[mask]
    idx_b = idx_b[mask]
    n_pairs = len(idx_a)
    
    print(f"  Computing {n_pairs} pair comparisons...")
    
    # Compute L2 distances
    l2_dists = np.linalg.norm(H_final[idx_a] - H_final[idx_b], axis=-1)
    
    # Compute KL divergences (on top-500 vocab for efficiency)
    # KL(P_a || P_b) = sum P_a * log(P_a / P_b)
    # Use top-500 tokens from each distribution for efficiency
    TOP_KL = 500
    kl_divs = []
    
    for k in range(0, n_pairs, 100):
        batch_end = min(k + 100, n_pairs)
        batch_idx_a = idx_a[k:batch_end]
        batch_idx_b = idx_b[k:batch_end]
        
        # Get log probabilities for these pairs
        log_p_a = log_probs[batch_idx_a]  # (batch, vocab)
        log_p_b = log_probs[batch_idx_b]  # (batch, vocab)
        
        # KL(P_a || P_b) = sum exp(log_p_a) * (log_p_a - log_p_b)
        # For numerical stability, only compute over top tokens
        top_indices_a = np.argsort(log_p_a, axis=-1)[:, -TOP_KL:]
        
        batch_kl = []
        for j in range(len(batch_idx_a)):
            # Get top-KL tokens from distribution a
            top_idx = top_indices_a[j]
            p_a = np.exp(log_p_a[j, top_idx])
            p_b = np.exp(log_p_b[j, top_idx])
            # Add small epsilon for stability
            p_a = np.maximum(p_a, 1e-10)
            p_b = np.maximum(p_b, 1e-10)
            # Renormalize
            p_a = p_a / p_a.sum()
            p_b = p_b / p_b.sum()
            kl = np.sum(p_a * (np.log(p_a) - np.log(p_b)))
            batch_kl.append(float(kl))
        
        kl_divs.extend(batch_kl)
    
    kl_divs = np.array(kl_divs)
    
    # Analyze relationship
    # Bin L2 distances and compute mean KL in each bin
    n_bins = 20
    l2_percentiles = np.percentile(l2_dists, np.linspace(0, 100, n_bins + 1))
    
    bin_kl_means = []
    bin_kl_stds = []
    bin_l2_means = []
    bin_counts = []
    
    for b in range(n_bins):
        mask = (l2_dists >= l2_percentiles[b]) & (l2_dists < l2_percentiles[b + 1])
        if mask.sum() > 0:
            bin_kl_means.append(float(np.mean(kl_divs[mask])))
            bin_kl_stds.append(float(np.std(kl_divs[mask])))
            bin_l2_means.append(float(np.mean(l2_dists[mask])))
            bin_counts.append(int(mask.sum()))
        else:
            bin_kl_means.append(0)
            bin_kl_stds.append(0)
            bin_l2_means.append(0)
            bin_counts.append(0)
    
    # Correlation between L2 and KL
    from scipy.stats import spearmanr, pearsonr
    valid = (kl_divs > 0) & np.isfinite(kl_divs) & np.isfinite(l2_dists)
    if valid.sum() > 100:
        pearson_r, pearson_p = pearsonr(l2_dists[valid], kl_divs[valid])
        spearman_r, spearman_p = spearmanr(l2_dists[valid], kl_divs[valid])
    else:
        pearson_r, pearson_p = 0, 1
        spearman_r, spearman_p = 0, 1
    
    print(f"  Pearson r(L2, KL): {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"  Spearman r(L2, KL): {spearman_r:.4f} (p={spearman_p:.2e})")
    
    # Estimate number of predictive equivalence classes
    # Method: cluster states by predictive similarity
    # If KL < threshold → same class
    kl_thresholds = [0.1, 0.5, 1.0, 2.0]
    n_classes_per_threshold = {}
    
    for thresh in kl_thresholds:
        # Use a greedy clustering approach
        # Assign each state to the first existing class where all members have KL < thresh
        # This is approximate but gives a sense of scale
        n_sample = min(500, N)
        sample_idx = np.random.choice(N, size=n_sample, replace=False)
        sample_logits = all_logits[sample_idx]
        
        # Compute pairwise KL for sample
        classes = []
        assigned = np.zeros(n_sample, dtype=bool)
        
        for i in range(n_sample):
            if assigned[i]:
                continue
            # Create new class
            classes.append([i])
            assigned[i] = True
            
            # Find all unassigned states with KL < thresh to this class
            log_p_i = log_probs[sample_idx[i]]
            top_idx_i = np.argsort(log_p_i)[-TOP_KL:]
            p_i = np.maximum(np.exp(log_p_i[top_idx_i]), 1e-10)
            p_i = p_i / p_i.sum()
            
            for j in range(i + 1, n_sample):
                if assigned[j]:
                    continue
                log_p_j = log_probs[sample_idx[j]]
                p_j = np.maximum(np.exp(log_p_j[top_idx_i]), 1e-10)
                p_j = p_j / p_j.sum()
                kl_ij = np.sum(p_i * (np.log(p_i) - np.log(p_j)))
                
                if kl_ij < thresh:
                    classes[-1].append(j)
                    assigned[j] = True
        
        n_classes = len(classes)
        n_classes_per_threshold[str(thresh)] = n_classes
        print(f"  KL threshold {thresh}: ~{n_classes} predictive equivalence classes (from {n_sample} samples)")
    
    # Find "predictive divergence" cases: small L2 but large KL
    l2_median = np.median(l2_dists)
    kl_median = np.median(kl_divs)
    
    # Cases where L2 < median but KL > median (close in geometry, far in prediction)
    close_l2_far_kl = np.sum((l2_dists < l2_median) & (kl_divs > kl_median))
    # Cases where L2 > median but KL < median (far in geometry, close in prediction)
    far_l2_close_kl = np.sum((l2_dists > l2_median) & (kl_divs < kl_median))
    
    print(f"  Close-L2/Far-KL pairs: {close_l2_far_kl}/{n_pairs} ({100*close_l2_far_kl/n_pairs:.1f}%)")
    print(f"  Far-L2/Close-KL pairs: {far_l2_close_kl}/{n_pairs} ({100*far_l2_close_kl/n_pairs:.1f}%)")
    
    results = {
        "n_pairs": n_pairs,
        "l2_dist_mean": float(np.mean(l2_dists)),
        "l2_dist_median": float(np.median(l2_dists)),
        "kl_div_mean": float(np.mean(kl_divs)),
        "kl_div_median": float(np.median(kl_divs)),
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "bin_l2_means": bin_l2_means,
        "bin_kl_means": bin_kl_means,
        "bin_kl_stds": bin_kl_stds,
        "bin_counts": bin_counts,
        "n_classes_per_threshold": n_classes_per_threshold,
        "close_l2_far_kl_pct": float(100 * close_l2_far_kl / n_pairs),
        "far_l2_close_kl_pct": float(100 * far_l2_close_kl / n_pairs),
    }
    
    del all_logits, log_probs
    gc.collect()
    
    return results


def exp2_causal_perturbation(model, tokenizer, device, model_info, data):
    """
    Exp 2: Causal Perturbation — THE DEFINITIVE TEST (Route 3)
    
    CORE QUESTION: Which directions in hidden space are causally relevant?
    
    Method:
      1. Compute PCA directions and Pred-directions from token trajectories
      2. Perturb h_final along each direction: h' = h + ε·v
      3. Compute KL(P(·|h) || P(·|h'))
      4. Compare: Pred-directions vs PCA-directions
    
    EXPECTATION (if user's theory is correct):
      - Pred-perturbation causes large output change (high KL)
      - PCA-perturbation causes small output change (low KL)
      - This would prove: "statistical importance ≠ causal importance"
    """
    print("\n" + "=" * 60)
    print("Exp 2: Causal Perturbation (Pred vs PCA directions)")
    print("=" * 60)
    
    H_final = data['H_final']
    H_final_list = data['H_final_list']
    token_ids = data['token_ids']
    embeddings = data['embeddings']
    
    if len(H_final) < 200:
        print("  WARNING: Not enough data")
        return {"error": "insufficient_data"}
    
    d_model = model_info.d_model
    
    # Step 1: Build token transitions at the last token position only
    # (For computing Pred-directions)
    # Actually, we need ALL token positions at the FINAL LAYER for Pred-direction computation
    # But H_final contains all token positions already (from collect_full_data)
    
    # Build h(t) -> h(t+1) transitions from all prompts
    H_t_list = []
    H_t1_list = []
    
    for i, h in enumerate(H_final_list):
        T = h.shape[0]
        if T < 3:
            continue
        for t in range(T - 1):
            H_t_list.append(h[t])
            H_t1_list.append(h[t + 1])
    
    H_t = np.array(H_t_list, dtype=np.float32)
    H_t1 = np.array(H_t1_list, dtype=np.float32)
    N_trans = H_t.shape[0]
    print(f"  Token transitions for Pred-direction: {N_trans}")
    
    if N_trans < 100:
        print("  WARNING: Too few transitions")
        return {"error": "insufficient_transitions"}
    
    # Step 2: Compute PCA directions
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=min(100, N_trans, d_model))
    pca.fit(H_final)
    
    pca_dirs = pca.components_  # (100, d_model)
    pca_vars = pca.explained_variance_ratio_
    
    print(f"  PCA: top-5 variance = {pca_vars[:5].tolist()}")
    
    # Step 3: Compute Pred-directions
    # Fit: h(t+1) ≈ W · h(t), then SVD(W) gives Pred-directions
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    
    # Project to PCA space first (for numerical stability)
    Z_t = pca.transform(H_t)    # (N, 100)
    Z_t1 = pca.transform(H_t1)  # (N, 100)
    
    train_idx, test_idx = train_test_split(np.arange(N_trans), test_size=0.2, random_state=42)
    reg = Ridge(alpha=1.0)
    reg.fit(Z_t[train_idx], Z_t1[train_idx])
    W = reg.coef_  # (100, 100)
    
    r2_test = reg.score(Z_t[test_idx], Z_t1[test_idx])
    print(f"  Linear prediction R² (test): {r2_test:.4f}")
    
    # SVD of W
    U_w, S_w, Vt_w = np.linalg.svd(W)
    
    # Pred-directions in original space
    pred_dirs_pca = Vt_w  # (100, 100) — predictive directions in PCA space
    pred_dirs = Vt_w @ pca.components_  # (100, d_model) — in original space
    
    # Normalize both sets of directions
    for i in range(min(50, pred_dirs.shape[0])):
        norm = np.linalg.norm(pred_dirs[i])
        if norm > 0:
            pred_dirs[i] /= norm
    for i in range(min(50, pca_dirs.shape[0])):
        norm = np.linalg.norm(pca_dirs[i])
        if norm > 0:
            pca_dirs[i] /= norm
    
    # Step 4: Compute alignment between Pred and PCA directions
    alignment_same_idx = []
    for i in range(min(10, pred_dirs.shape[0])):
        cos_sim = abs(np.dot(pred_dirs[i], pca_dirs[i]))
        alignment_same_idx.append(float(cos_sim))
    
    print(f"  Alignment |cos(Pred_i, PCA_i)| for i=1..10: {[f'{a:.3f}' for a in alignment_same_idx]}")
    
    # Step 5: Causal perturbation experiment
    # Select test states (last token of each prompt)
    test_states = []
    for i, h in enumerate(H_final_list):
        if h.shape[0] >= 3:
            test_states.append(h[-1])  # Last token position
    
    n_test = min(200, len(test_states))
    test_states = np.array(test_states[:n_test])
    
    print(f"  Test states for perturbation: {n_test}")
    
    # Get model's final norm and lm_head
    try:
        final_norm = model.model.norm
        lm_head = model.lm_head
    except AttributeError:
        print("  WARNING: Cannot access norm/lm_head")
        return {"error": "model_architecture_issue"}
    
    # Determine model dtype
    try:
        model_dtype = next(lm_head.parameters()).dtype
    except StopIteration:
        model_dtype = torch.bfloat16
    
    # Perturbation strengths (in terms of fraction of state norm)
    epsilons = [0.01, 0.05, 0.1, 0.5, 1.0]
    
    # Number of directions to test
    n_dirs = min(10, pred_dirs.shape[0], pca_dirs.shape[0])
    
    results_per_epsilon = {}
    
    for eps in epsilons:
        print(f"\n  --- ε = {eps} (fraction of state norm) ---")
        
        kl_pred_dirs = []  # KL for Pred-direction perturbations
        kl_pca_dirs = []   # KL for PCA-direction perturbations
        kl_random_dirs = []  # KL for random direction perturbations
        
        for state_idx in range(min(50, n_test)):  # Test on 50 states
            h_orig = test_states[state_idx]
            h_norm = np.linalg.norm(h_orig)
            if h_norm < 1e-8:
                continue
            
            # Compute original output distribution
            h_tensor = torch.tensor(h_orig, dtype=model_dtype).unsqueeze(0).to(device)
            with torch.no_grad():
                h_normed = final_norm(h_tensor)
                logits_orig = lm_head(h_normed)[0].detach().float().cpu().numpy()
            
            # Original log-probabilities
            logits_orig_max = np.max(logits_orig)
            log_probs_orig = logits_orig - logits_orig_max - np.log(np.sum(np.exp(logits_orig - logits_orig_max)))
            
            for dir_idx in range(n_dirs):
                perturbation_scale = eps * h_norm
                
                # --- Pred-direction perturbation ---
                delta_pred = perturbation_scale * pred_dirs[dir_idx]
                h_pert_pred = h_orig + delta_pred
                h_tensor = torch.tensor(h_pert_pred, dtype=model_dtype).unsqueeze(0).to(device)
                with torch.no_grad():
                    try:
                        h_normed = final_norm(h_tensor)
                        logits_pert = lm_head(h_normed)[0].detach().float().cpu().numpy()
                    except:
                        continue
                
                logits_pert_max = np.max(logits_pert)
                log_probs_pert = logits_pert - logits_pert_max - np.log(np.sum(np.exp(logits_pert - logits_pert_max)))
                
                # KL divergence (on top-500 tokens)
                top_idx = np.argsort(logits_orig)[-500:]
                p_orig = np.maximum(np.exp(log_probs_orig[top_idx]), 1e-10)
                p_pert = np.maximum(np.exp(log_probs_pert[top_idx]), 1e-10)
                p_orig /= p_orig.sum()
                p_pert /= p_pert.sum()
                kl_pred = np.sum(p_orig * (np.log(p_orig) - np.log(p_pert)))
                kl_pred_dirs.append(float(kl_pred))
                
                # --- PCA-direction perturbation ---
                delta_pca = perturbation_scale * pca_dirs[dir_idx]
                h_pert_pca = h_orig + delta_pca
                h_tensor = torch.tensor(h_pert_pca, dtype=model_dtype).unsqueeze(0).to(device)
                with torch.no_grad():
                    try:
                        h_normed = final_norm(h_tensor)
                        logits_pert = lm_head(h_normed)[0].detach().float().cpu().numpy()
                    except:
                        continue
                
                logits_pert_max = np.max(logits_pert)
                log_probs_pert = logits_pert - logits_pert_max - np.log(np.sum(np.exp(logits_pert - logits_pert_max)))
                
                p_pert = np.maximum(np.exp(log_probs_pert[top_idx]), 1e-10)
                p_pert /= p_pert.sum()
                kl_pca = np.sum(p_orig * (np.log(p_orig) - np.log(p_pert)))
                kl_pca_dirs.append(float(kl_pca))
                
                # --- Random direction perturbation (for baseline) ---
                random_dir = np.random.randn(d_model)
                random_dir /= np.linalg.norm(random_dir)
                delta_random = perturbation_scale * random_dir
                h_pert_random = h_orig + delta_random
                h_tensor = torch.tensor(h_pert_random, dtype=model_dtype).unsqueeze(0).to(device)
                with torch.no_grad():
                    try:
                        h_normed = final_norm(h_tensor)
                        logits_pert = lm_head(h_normed)[0].detach().float().cpu().numpy()
                    except:
                        continue
                
                logits_pert_max = np.max(logits_pert)
                log_probs_pert = logits_pert - logits_pert_max - np.log(np.sum(np.exp(logits_pert - logits_pert_max)))
                
                p_pert = np.maximum(np.exp(log_probs_pert[top_idx]), 1e-10)
                p_pert /= p_pert.sum()
                kl_random = np.sum(p_orig * (np.log(p_orig) - np.log(p_pert)))
                kl_random_dirs.append(float(kl_random))
        
        # Compute statistics
        pred_mean = float(np.mean(kl_pred_dirs)) if kl_pred_dirs else 0
        pca_mean = float(np.mean(kl_pca_dirs)) if kl_pca_dirs else 0
        random_mean = float(np.mean(kl_random_dirs)) if kl_random_dirs else 0
        
        pred_std = float(np.std(kl_pred_dirs)) if kl_pred_dirs else 0
        pca_std = float(np.std(kl_pca_dirs)) if kl_pca_dirs else 0
        random_std = float(np.std(kl_random_dirs)) if kl_random_dirs else 0
        
        # Per-direction breakdown
        pred_per_dir = []
        pca_per_dir = []
        n_samples_per_dir = len(kl_pred_dirs) // n_dirs if n_dirs > 0 else 0
        
        for d in range(n_dirs):
            start_idx = d * n_samples_per_dir
            end_idx = (d + 1) * n_samples_per_dir
            if end_idx <= len(kl_pred_dirs):
                pred_per_dir.append(float(np.mean(kl_pred_dirs[start_idx:end_idx])))
                pca_per_dir.append(float(np.mean(kl_pca_dirs[start_idx:end_idx])))
        
        print(f"    Pred KL:  {pred_mean:.6f} ± {pred_std:.6f}")
        print(f"    PCA KL:   {pca_mean:.6f} ± {pca_std:.6f}")
        print(f"    Random KL: {random_mean:.6f} ± {random_std:.6f}")
        print(f"    Pred/PCA ratio: {pred_mean/max(pca_mean, 1e-10):.3f}")
        print(f"    Pred/Random ratio: {pred_mean/max(random_mean, 1e-10):.3f}")
        
        results_per_epsilon[str(eps)] = {
            "pred_kl_mean": pred_mean, "pred_kl_std": pred_std,
            "pca_kl_mean": pca_mean, "pca_kl_std": pca_std,
            "random_kl_mean": random_mean, "random_kl_std": random_std,
            "pred_pca_ratio": float(pred_mean / max(pca_mean, 1e-10)),
            "pred_random_ratio": float(pred_mean / max(random_mean, 1e-10)),
            "pred_per_dir": pred_per_dir,
            "pca_per_dir": pca_per_dir,
            "n_dirs": n_dirs,
        }
    
    results = {
        "alignment_pred_pca": alignment_same_idx,
        "prediction_r2_test": float(r2_test),
        "n_transitions": N_trans,
        "per_epsilon": results_per_epsilon,
    }
    
    return results


def exp3_information_bottleneck(data):
    """
    Exp 3: Information Bottleneck State Learning (Route 1 & 4)
    
    CORE QUESTION: What is the minimal dimension z needed to predict next-token logits?
    
    Method:
      1. Train z = PCA_d(h) for different bottleneck dimensions d
      2. Predict next-token logits from z using linear regression
      3. Find the "compression cliff" — where adding more dimensions stops helping
      4. Compare with Pred-direction-based bottleneck
    
    This directly estimates the true state dimension.
    """
    print("\n" + "=" * 60)
    print("Exp 3: Information Bottleneck State Learning")
    print("=" * 60)
    
    H_final = data['H_final']
    H_final_list = data['H_final_list']
    logits_list = data['logits_list']
    
    if len(H_final) < 200 or logits_list is None:
        print("  WARNING: Not enough data or missing logits")
        return {"error": "insufficient_data"}
    
    d_model = H_final.shape[1]
    
    # Build transition pairs: (h(t), logits(t)) → predict logits(t+1)
    # Actually, we want: from h(t) at position t, predict the logits at position t+1
    # But logits_list contains top-k logits per prompt, not per position
    
    # Simpler approach: from h at each position, predict the logits at that same position
    # This tests: "how much of the output information is captured by d dimensions of h?"
    
    # Even simpler and more directly useful:
    # From h(t) at last layer, predict the output logits
    # Then compress h(t) to d dimensions and see how well we can still predict logits
    
    # Build dataset: (h, logits) pairs
    # We need to reconstruct full logits from the top-k stored data
    # This is tricky... let me instead recompute logits for a subset
    
    # Alternative approach: use the logit lens technique
    # At each position, the logits are: lm_head(norm(h))
    # So we can compute "how well does PCA_d(h) predict the logits?"
    
    # Let's do this differently:
    # For each d, project h to PCA_d space, reconstruct, and measure how well the 
    # reconstructed h predicts the same logits as the original h
    
    from sklearn.decomposition import PCA
    from sklearn.model_selection import KFold
    from sklearn.linear_model import Ridge
    
    # Use ALL hidden states (not just last token of each prompt)
    N = H_final.shape[0]
    print(f"  Total hidden states: {N}")
    
    # Full PCA
    pca_full = PCA(n_components=min(200, N, d_model))
    pca_full.fit(H_final)
    
    # For each bottleneck dimension d, measure:
    # 1. PCA reconstruction R²: how well does PCA_d(h) reconstruct h?
    # 2. Predictive R²: how well does PCA_d(h(t)) predict h(t+1)?
    
    # Build h(t) -> h(t+1) transitions
    H_t_list = []
    H_t1_list = []
    for h in H_final_list:
        T = h.shape[0]
        if T < 3:
            continue
        for t in range(T - 1):
            H_t_list.append(h[t])
            H_t1_list.append(h[t + 1])
    
    H_t = np.array(H_t_list, dtype=np.float32)
    H_t1 = np.array(H_t1_list, dtype=np.float32)
    N_trans = H_t.shape[0]
    
    print(f"  Token transitions: {N_trans}")
    
    # Project to PCA space
    Z_t = pca_full.transform(H_t)    # (N_trans, 200)
    Z_t1 = pca_full.transform(H_t1)  # (N_trans, 200)
    
    # Also compute Pred-directions for bottleneck
    reg_full = Ridge(alpha=1.0)
    reg_full.fit(Z_t, Z_t1)
    W = reg_full.coef_  # (200, 200)
    U_w, S_w, Vt_w = np.linalg.svd(W)
    
    # Bottleneck dimensions to test
    dims = [2, 5, 10, 15, 20, 30, 50, 75, 100, 150]
    dims = [d for d in dims if d <= 200]
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results_per_d = {}
    
    for d in dims:
        # === PCA bottleneck ===
        # Project h(t) to top-d PCA, predict h(t+1) in top-d PCA space
        pca_r2_folds = []
        pred_r2_folds = []
        recon_r2_folds = []
        
        for train_idx, test_idx in kf.split(range(N_trans)):
            Z_t_train_d, Z_t_test_d = Z_t[train_idx, :d], Z_t[test_idx, :d]
            Z_t1_train_d, Z_t1_test_d = Z_t1[train_idx, :d], Z_t1[test_idx, :d]
            
            # Full PCA projections (for Pred-direction computation)
            Z_t_train_full = Z_t[train_idx]   # (N_train, pca_full_dim)
            Z_t_test_full = Z_t[test_idx]
            Z_t1_train_full = Z_t1[train_idx]
            Z_t1_test_full = Z_t1[test_idx]
            
            # PCA bottleneck: predict in PCA-d space
            reg_pca = Ridge(alpha=0.1)
            reg_pca.fit(Z_t_train_d, Z_t1_train_d)
            r2_pca = reg_pca.score(Z_t_test_d, Z_t1_test_d)
            pca_r2_folds.append(float(r2_pca))
            
            # Pred bottleneck: project to top-d Pred-directions
            P_pred = Vt_w[:d, :]  # (d, pca_full_dim)
            Z_t_pred_train = Z_t_train_full @ P_pred.T  # (N_train, d)
            Z_t_pred_test = Z_t_test_full @ P_pred.T
            Z_t1_pred_train = Z_t1_train_full @ P_pred.T
            Z_t1_pred_test = Z_t1_test_full @ P_pred.T
            
            reg_pred = Ridge(alpha=0.1)
            reg_pred.fit(Z_t_pred_train, Z_t1_pred_train)
            r2_pred = reg_pred.score(Z_t_pred_test, Z_t1_pred_test)
            pred_r2_folds.append(float(r2_pred))
            
            # Reconstruction R² (how well does PCA-d reconstruct h?)
            # This measures the "variance bottleneck"
            H_t_test_orig = H_t[test_idx]
            Z_t_test_full_pca = pca_full.transform(H_t_test_orig)
            pca_dim = pca_full.n_components_
            H_t_recon = pca_full.inverse_transform(
                np.hstack([Z_t_test_d, np.zeros((len(test_idx), pca_dim - d))])
            )
            ss_res = np.sum((H_t_test_orig - H_t_recon) ** 2)
            ss_tot = np.sum((H_t_test_orig - H_t_test_orig.mean(axis=0)) ** 2)
            r2_recon = 1 - ss_res / max(ss_tot, 1e-10)
            recon_r2_folds.append(float(r2_recon))
        
        pca_mean = float(np.mean(pca_r2_folds))
        pred_mean = float(np.mean(pred_r2_folds))
        recon_mean = float(np.mean(recon_r2_folds))
        
        print(f"  d={d:3d}: PCA pred R²={pca_mean:.4f}, Pred pred R²={pred_mean:.4f}, "
              f"Recon R²={recon_mean:.4f}, Δ(Pred-PCA)={pred_mean-pca_mean:+.4f}")
        
        results_per_d[str(d)] = {
            "pca_pred_r2": pca_mean,
            "pca_pred_r2_std": float(np.std(pca_r2_folds)),
            "pred_pred_r2": pred_mean,
            "pred_pred_r2_std": float(np.std(pred_r2_folds)),
            "recon_r2": recon_mean,
            "delta_pred_pca": float(pred_mean - pca_mean),
        }
    
    # Find the "compression cliff"
    # Look for where adding more dimensions gives diminishing returns
    pca_r2_list = [results_per_d[str(d)]["pca_pred_r2"] for d in dims]
    pred_r2_list = [results_per_d[str(d)]["pred_pred_r2"] for d in dims]
    
    # Find elbow: where the derivative drops below threshold
    pca_derivs = [pca_r2_list[i+1] - pca_r2_list[i] for i in range(len(pca_r2_list)-1)]
    pred_derivs = [pred_r2_list[i+1] - pred_r2_list[i] for i in range(len(pred_r2_list)-1)]
    
    # Compression cliff: first d where derivative < 0.01
    pca_cliff_d = dims[0]
    for i, deriv in enumerate(pca_derivs):
        if deriv < 0.01:
            pca_cliff_d = dims[i + 1]
            break
    
    pred_cliff_d = dims[0]
    for i, deriv in enumerate(pred_derivs):
        if deriv < 0.01:
            pred_cliff_d = dims[i + 1]
            break
    
    print(f"\n  PCA compression cliff: d={pca_cliff_d}")
    print(f"  Pred compression cliff: d={pred_cliff_d}")
    
    results = {
        "dims_tested": dims,
        "per_dim": results_per_d,
        "pca_cliff_d": pca_cliff_d,
        "pred_cliff_d": pred_cliff_d,
        "n_transitions": N_trans,
    }
    
    return results


def exp4_self_conditioning(model, tokenizer, device, model_info, data):
    """
    Exp 4: Self-Conditioning Dynamics Verification
    
    CORE QUESTION: Is the system truly "self-conditioning"?
    
    The user's insight: the token is NOT an external input.
    The model generates the token, then re-injects it.
    This makes it a "self-conditioning dynamical system".
    
    Method:
      1. For each token position, get:
         - The model's predicted next token (argmax of logits)
         - The actual next token
      2. Compare: when predicted = actual (correct prediction), 
         is the transition simpler?
      3. Measure: how much of h(t+1) is explained by h(t) alone,
         vs h(t) + embed(token[t+1])
      4. Compare: model's own predicted token vs random token injection
    
    This tests whether the system is Markov, self-conditioning, or externally driven.
    """
    print("\n" + "=" * 60)
    print("Exp 4: Self-Conditioning Dynamics")
    print("=" * 60)
    
    H_final_list = data['H_final_list']
    token_ids = data['token_ids']
    embeddings = data['embeddings']
    
    d_model = model_info.d_model
    
    # Step 1: For each prompt, compute predicted vs actual next token
    try:
        input_device = next(model.parameters()).device
    except StopIteration:
        input_device = device
    
    embed_layer = model.get_input_embeddings()
    
    # Statistics
    total_positions = 0
    correct_predictions = 0  # Where predicted = actual
    h_delta_correct = []  # ||h(t+1) - h(t)|| when prediction is correct
    h_delta_incorrect = []  # ||h(t+1) - h(t)|| when prediction is wrong
    pred_confidence_correct = []  # Max softmax prob when correct
    pred_confidence_incorrect = []  # Max softmax prob when wrong
    
    # For the Markov analysis
    h_t_correct = []
    h_t1_correct = []
    emb_correct = []
    h_t_incorrect = []
    h_t1_incorrect = []
    emb_incorrect = []
    
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    from sklearn.decomposition import PCA
    
    n_prompts = min(400, len(H_final_list))
    print(f"  Processing {n_prompts} prompts...")
    
    for i in range(n_prompts):
        h = H_final_list[i]
        tok_ids = token_ids[i]
        T = h.shape[0]
        
        if T < 4:
            continue
        
        # Get logits for each position
        prompt_text = tokenizer.decode(tok_ids)
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=64)
        input_ids_tensor = inputs["input_ids"].to(input_device)
        
        with torch.no_grad():
            out = model(input_ids=input_ids_tensor, output_hidden_states=True)
        
        if not hasattr(out, 'logits') or out.logits is None:
            del out
            continue
        
        logits = out.logits[0].detach().float().cpu().numpy()  # (T, vocab)
        h_all = out.hidden_states[model_info.n_layers][0].detach().float().cpu().numpy()  # (T, d_model)
        
        del out
        
        for t in range(min(T - 1, logits.shape[0] - 1)):
            total_positions += 1
            
            # Predicted next token at position t
            predicted_token = int(np.argmax(logits[t]))
            actual_token = int(tok_ids[t + 1]) if t + 1 < len(tok_ids) else -1
            
            if actual_token < 0:
                continue
            
            # Softmax confidence
            logit_t = logits[t]
            logit_max = np.max(logit_t)
            probs = np.exp(logit_t - logit_max)
            probs = probs / probs.sum()
            max_prob = float(np.max(probs))
            
            # Delta h
            delta_h = np.linalg.norm(h_all[t + 1] - h_all[t])
            
            is_correct = (predicted_token == actual_token)
            
            if is_correct:
                correct_predictions += 1
                h_delta_correct.append(delta_h)
                pred_confidence_correct.append(max_prob)
                h_t_correct.append(h_all[t])
                h_t1_correct.append(h_all[t + 1])
                emb_correct.append(embeddings[i][t + 1] if i < len(embeddings) else np.zeros(d_model))
            else:
                h_delta_incorrect.append(delta_h)
                pred_confidence_incorrect.append(max_prob)
                h_t_incorrect.append(h_all[t])
                h_t1_incorrect.append(h_all[t + 1])
                emb_incorrect.append(embeddings[i][t + 1] if i < len(embeddings) else np.zeros(d_model))
    
    print(f"  Total positions: {total_positions}")
    print(f"  Correct predictions: {correct_predictions} ({100*correct_predictions/max(total_positions,1):.1f}%)")
    
    # Analyze: is the transition simpler when prediction is correct?
    if h_delta_correct and h_delta_incorrect:
        mean_delta_correct = float(np.mean(h_delta_correct))
        mean_delta_incorrect = float(np.mean(h_delta_incorrect))
        print(f"  Mean ||Δh|| when correct: {mean_delta_correct:.4f}")
        print(f"  Mean ||Δh|| when incorrect: {mean_delta_incorrect:.4f}")
        print(f"  Ratio (incorrect/correct): {mean_delta_incorrect/max(mean_delta_correct, 1e-10):.3f}")
    
    if pred_confidence_correct and pred_confidence_incorrect:
        print(f"  Mean confidence when correct: {np.mean(pred_confidence_correct):.4f}")
        print(f"  Mean confidence when incorrect: {np.mean(pred_confidence_incorrect):.4f}")
    
    # Markov analysis for correct vs incorrect predictions
    # For correct predictions: h(t) should better predict h(t+1) because the token was "expected"
    # For incorrect predictions: h(t) alone should be worse because the token was "surprising"
    
    markov_results = {}
    
    for label, h_t_data, h_t1_data, emb_data in [
        ("correct", h_t_correct, h_t1_correct, emb_correct),
        ("incorrect", h_t_incorrect, h_t1_incorrect, emb_incorrect),
    ]:
        if len(h_t_data) < 50:
            continue
        
        H_t_arr = np.array(h_t_data, dtype=np.float32)
        H_t1_arr = np.array(h_t1_data, dtype=np.float32)
        E_arr = np.array(emb_data, dtype=np.float32)
        N = H_t_arr.shape[0]
        
        # PCA
        d_pca = min(50, N, d_model)
        pca = PCA(n_components=d_pca)
        pca.fit(np.vstack([H_t_arr, H_t1_arr]))
        
        Z_t = pca.transform(H_t_arr)
        Z_t1 = pca.transform(H_t1_arr)
        Z_e = pca.transform(E_arr)
        
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        
        r2_alone_folds = []
        r2_with_emb_folds = []
        
        for train_idx, test_idx in kf.split(range(N)):
            Z_t_train, Z_t_test = Z_t[train_idx], Z_t[test_idx]
            Z_t1_train, Z_t1_test = Z_t1[train_idx], Z_t1[test_idx]
            Z_e_train, Z_e_test = Z_e[train_idx], Z_e[test_idx]
            
            # h(t) alone
            reg = Ridge(alpha=0.1)
            reg.fit(Z_t_train, Z_t1_train)
            r2_alone = reg.score(Z_t_test, Z_t1_test)
            r2_alone_folds.append(float(r2_alone))
            
            # h(t) + embedding
            combined_train = np.hstack([Z_t_train, Z_e_train])
            combined_test = np.hstack([Z_t_test, Z_e_test])
            reg2 = Ridge(alpha=0.1)
            reg2.fit(combined_train, Z_t1_train)
            r2_with = reg2.score(combined_test, Z_t1_test)
            r2_with_emb_folds.append(float(r2_with))
        
        r2_alone_mean = float(np.mean(r2_alone_folds))
        r2_with_mean = float(np.mean(r2_with_emb_folds))
        delta_r2 = r2_with_mean - r2_alone_mean
        
        markov_results[label] = {
            "n_samples": N,
            "r2_h_alone": r2_alone_mean,
            "r2_h_with_emb": r2_with_mean,
            "delta_r2": delta_r2,
        }
        
        print(f"  [{label}] R²(h(t) alone): {r2_alone_mean:.4f}, "
              f"R²(h(t)+emb): {r2_with_mean:.4f}, ΔR²: {delta_r2:.4f}")
    
    # Key comparison: is ΔR² larger for incorrect predictions?
    # If yes → when the model is "surprised", the new token matters more
    # If no → the token always matters equally regardless of prediction
    
    if "correct" in markov_results and "incorrect" in markov_results:
        delta_r2_correct = markov_results["correct"]["delta_r2"]
        delta_r2_incorrect = markov_results["incorrect"]["delta_r2"]
        print(f"\n  KEY COMPARISON:")
        print(f"    ΔR² (correct): {delta_r2_correct:.4f}")
        print(f"    ΔR² (incorrect): {delta_r2_incorrect:.4f}")
        print(f"    Ratio: {delta_r2_incorrect/max(delta_r2_correct, 1e-10):.3f}")
        
        if delta_r2_incorrect > delta_r2_correct:
            print(f"    → When prediction is WRONG, new token matters MORE (self-conditioning confirmed)")
        else:
            print(f"    → New token matters equally regardless of prediction")
    
    results = {
        "total_positions": total_positions,
        "correct_predictions": correct_predictions,
        "correct_pct": float(100 * correct_predictions / max(total_positions, 1)),
        "mean_delta_h_correct": float(np.mean(h_delta_correct)) if h_delta_correct else 0,
        "mean_delta_h_incorrect": float(np.mean(h_delta_incorrect)) if h_delta_incorrect else 0,
        "markov_analysis": markov_results,
    }
    
    return results


# ===== 4. MAIN =====

def run_all_experiments(model_name: str):
    """Run all experiments for a given model"""
    print(f"\n{'='*70}")
    print(f"Phase 161: Predictive Information Closure — {model_name}")
    print(f"{'='*70}")
    
    # Load model
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"  Model: {model_info.model_class}, {model_info.n_layers} layers, d_model={model_info.d_model}")
    
    # Generate prompts
    prompts = generate_diverse_prompts(n=600)
    
    # Collect data
    print(f"\n--- Data Collection ---")
    data = collect_full_data(model, tokenizer, device, model_info, prompts)
    
    if data['n_successful'] < 50:
        print(f"ERROR: Only {data['n_successful']} successful prompts, cannot proceed")
        release_model(model)
        return None
    
    # Run experiments
    results = {
        "model_name": model_name,
        "model_info": {
            "class": model_info.model_class,
            "n_layers": model_info.n_layers,
            "d_model": model_info.d_model,
            "vocab_size": model_info.vocab_size,
        },
        "n_successful_prompts": data['n_successful'],
        "total_tokens": data['total_toks'],
    }
    
    # Exp 1: Predictive Equivalence
    print(f"\n{'='*60}")
    print("Running Exp 1: Predictive Equivalence Classes")
    print(f"{'='*60}")
    try:
        exp1_results = exp1_predictive_equivalence(model, tokenizer, device, model_info, data)
        results["exp1_predictive_equivalence"] = exp1_results
    except Exception as e:
        print(f"  Exp 1 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results["exp1_predictive_equivalence"] = {"error": str(e)}
    
    # Exp 2: Causal Perturbation
    print(f"\n{'='*60}")
    print("Running Exp 2: Causal Perturbation")
    print(f"{'='*60}")
    try:
        exp2_results = exp2_causal_perturbation(model, tokenizer, device, model_info, data)
        results["exp2_causal_perturbation"] = exp2_results
    except Exception as e:
        print(f"  Exp 2 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results["exp2_causal_perturbation"] = {"error": str(e)}
    
    # Exp 3: Information Bottleneck
    print(f"\n{'='*60}")
    print("Running Exp 3: Information Bottleneck")
    print(f"{'='*60}")
    try:
        exp3_results = exp3_information_bottleneck(data)
        results["exp3_information_bottleneck"] = exp3_results
    except Exception as e:
        print(f"  Exp 3 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results["exp3_information_bottleneck"] = {"error": str(e)}
    
    # Exp 4: Self-Conditioning
    print(f"\n{'='*60}")
    print("Running Exp 4: Self-Conditioning Dynamics")
    print(f"{'='*60}")
    try:
        exp4_results = exp4_self_conditioning(model, tokenizer, device, model_info, data)
        results["exp4_self_conditioning"] = exp4_results
    except Exception as e:
        print(f"  Exp 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results["exp4_self_conditioning"] = {"error": str(e)}
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    result_path = f"tests/glm5_temp/phase161_{model_name}_{timestamp}.json"
    
    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    results = convert(results)
    
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Results saved to {result_path}]")
    
    # Release model
    release_model(model)
    
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python phase161_predictive_closure.py <model_name>")
        print("  model_name: qwen3, glm4, deepseek7b")
        sys.exit(1)
    
    model_name = sys.argv[1].lower()
    if model_name not in ("qwen3", "glm4", "deepseek7b"):
        print(f"ERROR: Unknown model '{model_name}'. Use: qwen3, glm4, deepseek7b")
        sys.exit(1)
    
    results = run_all_experiments(model_name)
    
    if results:
        print(f"\n{'='*70}")
        print(f"Phase 161 Complete: {model_name}")
        print(f"{'='*70}")
        
        # Print key findings
        for exp_name in ["exp1_predictive_equivalence", "exp2_causal_perturbation",
                        "exp3_information_bottleneck", "exp4_self_conditioning"]:
            if exp_name in results and "error" not in results[exp_name]:
                print(f"\n  {exp_name}: OK")
            elif exp_name in results:
                print(f"\n  {exp_name}: ERROR — {results[exp_name].get('error', 'unknown')}")
