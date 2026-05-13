"""
Phase 160: Token Trajectory Dynamics & Predictive State Variables
==================================================================

Core pivot from user's Phase 159 critique:
  "Layer ≠ Time" — layers are iterative refinement, NOT temporal evolution.
  Token-to-token evolution IS the real dynamics.

  Key questions shift from:
    "How does h change across layers?" (iterative refinement)
  TO:
    "How does h change across token positions?" (true temporal dynamics)

User's 5 key directives:
  1. Stop using PCA — use predictive latent learning (optimize z for prediction)
  2. Stop studying layers — study token trajectories (h(t)→h(t+1) at fixed layer)
  3. Stop finding geometric conservation — find informational closure
  4. Distinguish representation vs computation — hidden state = "working memory", not "state"
  5. Language = "compressed predictive dynamics": z_{t+1} = f(z_t, u_t)

Critical insight:
  The "state variable" of language is NOT "what geometric structure does h have?",
  but "what is the MINIMAL SUFFICIENT STATISTIC for predicting the next state?"

Experiments:
  Exp 1: Token Trajectory PCA Dimensionality (basic characterization)
  Exp 2: Predictive Dimensionality (SVD of W: h(t+1) ≈ W·h(t))
  Exp 3: Predictive vs PCA Directions (cross-validated R² comparison)
  Exp 4: Markov Property Test (predict with/without new token embedding)
  Exp 5: Token vs Layer Dynamics Comparison (which is more low-dimensional?)
"""

import sys
import os
import time
import json
import gc
import numpy as np
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, 'tests/glm5')

import torch
from model_utils import load_model, get_model_info, get_layers, release_model, get_W_U


# ===== 1. DIVERSE PROMPT GENERATION =====

def generate_diverse_prompts(n=500):
    """Generate diverse text prompts (15-25 tokens each)"""
    nouns = [
        "cat", "dog", "bird", "fish", "child", "woman", "man", "boy", "girl", "teacher",
        "doctor", "student", "tree", "flower", "river", "mountain", "car", "book", "house", "city",
        "king", "queen", "soldier", "artist", "scientist", "writer", "musician", "engineer", "farmer", "driver",
        "chair", "table", "door", "window", "road", "bridge", "tower", "castle", "garden", "forest",
        "ocean", "island", "valley", "desert", "planet", "star", "moon", "sun", "cloud", "rain",
        "apple", "bread", "water", "fire", "stone", "glass", "paper", "silk", "gold", "iron",
        "teacher", "village", "library", "kitchen", "market", "harbor", "cathedral", "museum", "theater", "palace",
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
        "forest", "mountain", "valley", "river", "island", "desert", "cave", "bridge", "tower", "castle",
    ]
    advs = ["quickly", "slowly", "carefully", "quietly", "loudly", "gently", "suddenly", "carefully",
            "happily", "sadly", "gracefully", "boldly", "silently", "proudly", "bravely"]

    prompts = []
    seen = set()

    def add(p):
        if p not in seen and len(p.split()) >= 5:
            seen.add(p)
            prompts.append(p)

    # Pattern 1: Simple subject-verb-object
    for nn in nouns[:30]:
        for v in verbs[:12]:
            add(f"The {nn} {v} toward the")
            add(f"A {nn} {v} away from the")

    # Pattern 2: Adjective-noun-verb
    for a in adjs[:15]:
        for nn in nouns[:15]:
            for v in verbs[:6]:
                add(f"The {a} {nn} {v} in the")

    # Pattern 3: Plural subjects
    for nn in nouns[:20]:
        for v in verbs[:8]:
            ns = nn + "s" if not nn.endswith("s") else nn + "es"
            add(f"The {ns} {v} toward the")

    # Pattern 4: Questions
    for nn in nouns[:15]:
        for v in verbs[:8]:
            add(f"Does the {nn} {v} in the")
            add(f"Will the {nn} {v} when the")

    # Pattern 5: Negation
    for nn in nouns[:12]:
        for v in verbs[:8]:
            add(f"The {nn} does not {v} in the")

    # Pattern 6: Conditional
    for nn in nouns[:10]:
        for v in verbs[:6]:
            add(f"If the {nn} {v} then the")
            add(f"When the {nn} {v} the other")

    # Pattern 7: Causal
    for nn in nouns[:10]:
        for v in verbs[:6]:
            add(f"Because the {nn} {v} the")
            add(f"Although the {nn} {v} the")

    # Pattern 8: Location-based
    for nn in nouns[:12]:
        for p in places[:10]:
            add(f"The {nn} in the {p} was")

    # Pattern 9: Adverbial
    for adv in advs[:8]:
        for nn in nouns[:10]:
            for v in verbs[:5]:
                add(f"The {nn} {adv} {v} toward the")

    # Pattern 10: Complex structures
    for nn in nouns[:10]:
        for v in verbs[:5]:
            add(f"After the {nn} {v} the people")
            add(f"Before the {nn} {v} the group")
            add(f"While the {nn} {v} the rest")

    # Pattern 11: Narrative openings
    for a in adjs[:10]:
        for nn in nouns[:10]:
            add(f"Once there was a {a} {nn} who lived in")
            add(f"In the land of the {a} {nn} there")

    # Pattern 12: Scientific/expository
    for nn in nouns[:10]:
        add(f"Research shows that the {nn} can")
        add(f"Scientists discovered that {nn}s are")
        add(f"The study of {nn} reveals that")

    # Pattern 13: Longer descriptive
    for a in adjs[:8]:
        for nn in nouns[:8]:
            add(f"The very {a} {nn} stood near the old")
            add(f"A truly {a} {nn} appeared beside the")

    print(f"[prompts] Generated {len(prompts)} unique prompts")
    return prompts[:n]


# ===== 2. DATA COLLECTION =====

def collect_token_trajectory_data(model, tokenizer, device, model_info, prompts):
    """
    Collect hidden states at ALL token positions for the last transformer layer.
    
    Returns:
        H_tokens: list of arrays, H_tokens[i] = (T_i, d_model) for prompt i
        token_ids: list of arrays, token_ids[i] = (T_i,) token IDs for prompt i
        embeddings: list of arrays, embeddings[i] = (T_i, d_model) token embeddings
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    try:
        input_device = next(model.parameters()).device
    except StopIteration:
        input_device = device
    
    embed_layer = model.get_input_embeddings()
    
    H_tokens = []
    token_ids_list = []
    embeddings_list = []
    failed = 0
    
    for i, prompt in enumerate(prompts):
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            seq_len = input_ids.shape[1]
            
            # Skip very short prompts (< 5 tokens)
            if seq_len < 5:
                failed += 1
                continue
            
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
            
            # Last transformer layer output (before final LN)
            # hidden_states[n_layers] = after all transformer layers
            h_all_positions = out.hidden_states[n_layers][0, :, :].detach().float().cpu().numpy()
            
            # Check for NaN
            if np.any(np.isnan(h_all_positions)) or np.any(np.isinf(h_all_positions)):
                failed += 1
                del out
                continue
            
            # Token IDs
            tok_ids = input_ids[0].detach().cpu().numpy()
            
            # Token embeddings
            with torch.no_grad():
                emb = embed_layer(input_ids)[0, :, :].detach().float().cpu().numpy()
            
            H_tokens.append(h_all_positions)  # (T, d_model)
            token_ids_list.append(tok_ids)     # (T,)
            embeddings_list.append(emb)        # (T, d_model)
            
            del out
            
            if (i + 1) % 100 == 0:
                total_toks = sum(h.shape[0] for h in H_tokens)
                print(f"  [{i+1}/{len(prompts)}] collected, total tokens so far: {total_toks}")
                
        except Exception as e:
            failed += 1
            if i < 5:
                print(f"  Warning: prompt {i} failed: {e}")
    
    total_toks = sum(h.shape[0] for h in H_tokens)
    n_successful = len(H_tokens)
    print(f"[collect] {n_successful} successful, {failed} failed, total tokens: {total_toks}")
    
    return H_tokens, token_ids_list, embeddings_list


# ===== 3. EXPERIMENTS =====

def exp1_token_trajectory_pca(H_tokens, d_model):
    """
    Exp 1: Token Trajectory PCA Dimensionality
    
    Question: Is h(t) at a fixed layer low-dimensional across token positions?
    Compare with: layer trajectory PCA from Phase 158/159
    """
    print("\n" + "=" * 60)
    print("Exp 1: Token Trajectory PCA Dimensionality")
    print("=" * 60)
    
    # Concatenate all token positions
    H_all = np.vstack(H_tokens)  # (N_total, d_model)
    N = H_all.shape[0]
    print(f"  Total token positions: {N}")
    
    from sklearn.decomposition import PCA
    
    # Full PCA
    pca = PCA(n_components=min(200, N, d_model))
    pca.fit(H_all)
    
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    
    # Key thresholds
    d_50 = int(np.searchsorted(cumvar, 0.50)) + 1
    d_90 = int(np.searchsorted(cumvar, 0.90)) + 1
    d_95 = int(np.searchsorted(cumvar, 0.95)) + 1
    d_99 = int(np.searchsorted(cumvar, 0.99)) + 1
    
    print(f"  d for 50% var: {d_50}")
    print(f"  d for 90% var: {d_90}")
    print(f"  d for 95% var: {d_95}")
    print(f"  d for 99% var: {d_99}")
    print(f"  Top 10 explained var ratio: {pca.explained_variance_ratio_[:10].tolist()}")
    print(f"  Cumulative var (d=2,5,10,20,30,50,100): "
          f"{[f'{cumvar[d-1]:.4f}' for d in [2,5,10,20,30,50,100] if d-1 < len(cumvar)]}")
    
    # Also compute PCA for EACH prompt separately (within-prompt dimensionality)
    within_prompt_dims = []
    for h in H_tokens:
        if h.shape[0] >= 10:  # Need enough tokens
            pca_local = PCA(n_components=min(30, h.shape[0], h.shape[1]))
            pca_local.fit(h)
            cumvar_local = np.cumsum(pca_local.explained_variance_ratio_)
            d90_local = int(np.searchsorted(cumvar_local, 0.90)) + 1
            within_prompt_dims.append(d90_local)
    
    if within_prompt_dims:
        print(f"  Within-prompt d90: mean={np.mean(within_prompt_dims):.1f}, "
              f"median={np.median(within_prompt_dims):.1f}, std={np.std(within_prompt_dims):.1f}")
    
    results = {
        "N_total": N,
        "d_50": d_50, "d_90": d_90, "d_95": d_95, "d_99": d_99,
        "top10_var_ratio": pca.explained_variance_ratio_[:10].tolist(),
        "cumvar_at_dims": {str(d): float(cumvar[d-1]) for d in [2,5,10,20,30,50,100] if d-1 < len(cumvar)},
        "within_prompt_d90_mean": float(np.mean(within_prompt_dims)) if within_prompt_dims else None,
        "within_prompt_d90_median": float(np.median(within_prompt_dims)) if within_prompt_dims else None,
    }
    
    return results, pca, H_all


def exp2_predictive_dimensionality(H_tokens, d_pca=100):
    """
    Exp 2: Predictive Dimensionality
    
    Question: What directions in h(t) are most useful for predicting h(t+1)?
    Method: Fit W: h(t+1) ≈ W·h(t), then SVD(W) gives predictive directions.
    
    This is the CORE experiment: compare predictive directions with PCA directions.
    """
    print("\n" + "=" * 60)
    print("Exp 2: Predictive Dimensionality (Linear)")
    print("=" * 60)
    
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import train_test_split
    
    # Step 1: Build pairs (h(t), h(t+1)) from all token trajectories
    H_t_list = []   # h(t)
    H_t1_list = []  # h(t+1)
    
    for h in H_tokens:
        T = h.shape[0]
        if T < 3:
            continue
        for t in range(T - 1):
            H_t_list.append(h[t])
            H_t1_list.append(h[t + 1])
    
    H_t = np.array(H_t_list, dtype=np.float32)    # (N, d_model)
    H_t1 = np.array(H_t1_list, dtype=np.float32)  # (N, d_model)
    N = H_t.shape[0]
    
    print(f"  Token transition pairs: {N}")
    
    # Step 2: PCA projection to d_pca dimensions
    pca = PCA(n_components=min(d_pca, N, H_t.shape[1]))
    pca.fit(np.vstack([H_t, H_t1]))
    
    Z_t = pca.transform(H_t)   # (N, d_pca)
    Z_t1 = pca.transform(H_t1) # (N, d_pca)
    
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    print(f"  PCA var explained (top {d_pca}): {cumvar[-1]:.4f}")
    
    # Step 3: Fit linear prediction in PCA space
    # Z_t1 ≈ W · Z_t
    # Use train/test split
    train_idx, test_idx = train_test_split(np.arange(N), test_size=0.2, random_state=42)
    
    Z_t_train, Z_t_test = Z_t[train_idx], Z_t[test_idx]
    Z_t1_train, Z_t1_test = Z_t1[train_idx], Z_t1[test_idx]
    
    # Fit ridge regression
    reg = Ridge(alpha=1.0)
    reg.fit(Z_t_train, Z_t1_train)
    W = reg.coef_  # (d_pca, d_pca)
    
    # SVD of W
    U_w, S_w, Vt_w = np.linalg.svd(W)
    
    print(f"  Prediction R² (all {d_pca} PCA dims): "
          f"train={reg.score(Z_t_train, Z_t1_train):.4f}, "
          f"test={reg.score(Z_t_test, Z_t1_test):.4f}")
    
    # Predictive dimensionality: effective rank of W
    S_w_norm = S_w / S_w.sum()
    cum_s = np.cumsum(S_w_norm)
    d_pred_50 = int(np.searchsorted(cum_s, 0.50)) + 1
    d_pred_90 = int(np.searchsorted(cum_s, 0.90)) + 1
    d_pred_95 = int(np.searchsorted(cum_s, 0.95)) + 1
    
    print(f"  Predictive dimensionality (SVD of W):")
    print(f"    d_pred 50%: {d_pred_50}")
    print(f"    d_pred 90%: {d_pred_90}")
    print(f"    d_pred 95%: {d_pred_95}")
    print(f"    Top 10 singular values: {S_w[:10].tolist()}")
    
    # Step 4: Compare predictive directions with PCA directions
    # Right singular vectors of W = predictive directions in h(t) space
    # Project back to original space: predictive_dir_i = Vt_w[i] @ pca.components_
    pred_dirs = Vt_w @ pca.components_  # (d_pca, d_model) — predictive directions in original space
    pca_dirs = pca.components_           # (d_pca, d_model) — PCA directions
    
    # Compute alignment between top predictive and PCA directions
    alignments = []
    for i in range(min(10, d_pca)):
        # Cosine similarity between predictive dir i and PCA dir i
        cos_sim = np.abs(np.dot(pred_dirs[i], pca_dirs[i])) / (
            np.linalg.norm(pred_dirs[i]) * np.linalg.norm(pca_dirs[i]) + 1e-10)
        alignments.append(float(cos_sim))
    
    print(f"  Alignment (|cos|) between top-10 predictive & PCA dirs: "
          f"{[f'{a:.3f}' for a in alignments]}")
    
    # Also check if top predictive directions align with ANY PCA direction
    # (Not just the same index, but the best match)
    best_alignments = []
    for i in range(min(10, d_pca)):
        cos_all = np.abs(pred_dirs[i] @ pca_dirs.T) / (
            np.linalg.norm(pred_dirs[i]) * np.linalg.norm(pca_dirs, axis=1) + 1e-10)
        best_j = np.argmax(cos_all)
        best_alignments.append(float(cos_all[best_j]))
    
    print(f"  Best alignment of predictive dirs with any PCA dir: "
          f"{[f'{a:.3f}' for a in best_alignments]}")
    
    results = {
        "N_transitions": N,
        "d_pca": d_pca,
        "pca_r2_train": float(reg.score(Z_t_train, Z_t1_train)),
        "pca_r2_test": float(reg.score(Z_t_test, Z_t1_test)),
        "d_pred_50": d_pred_50,
        "d_pred_90": d_pred_90,
        "d_pred_95": d_pred_95,
        "top10_svd_W": S_w[:10].tolist(),
        "alignment_pred_pca_same_idx": alignments,
        "alignment_pred_pca_best_match": best_alignments,
    }
    
    return results, pca, W, pred_dirs, Z_t, Z_t1


def exp3_predictive_vs_pca(Z_t, Z_t1, pca, W, pred_dirs, d_model):
    """
    Exp 3: Predictive vs PCA Directions (Cross-Validated)
    
    Key experiment: Does the predictive basis outperform PCA for next-state prediction?
    """
    print("\n" + "=" * 60)
    print("Exp 3: Predictive vs PCA Directions (Cross-Validated)")
    print("=" * 60)
    
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    
    d_pca = Z_t.shape[1]
    N = Z_t.shape[0]
    
    dims_to_test = [2, 5, 10, 15, 20, 30, 50, 75, 100]
    dims_to_test = [d for d in dims_to_test if d <= d_pca]
    
    # Predictive directions: project h(t) onto top-d right singular vectors of W
    # Then predict h(t+1) projected onto top-d PCA directions
    U_w, S_w, Vt_w = np.linalg.svd(W)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results_per_d = {}
    
    for d in dims_to_test:
        pca_r2_folds = []
        pred_r2_folds = []
        
        for train_idx, test_idx in kf.split(range(N)):
            Z_t_train, Z_t_test = Z_t[train_idx], Z_t[test_idx]
            Z_t1_train, Z_t1_test = Z_t1[train_idx], Z_t1[test_idx]
            
            # === PCA basis: use top-d PCA components ===
            # Project onto top-d PCA, predict, then measure in that subspace
            reg_pca = Ridge(alpha=0.1)
            reg_pca.fit(Z_t_train[:, :d], Z_t1_train[:, :d])
            Z_t1_pred_pca = reg_pca.predict(Z_t_test[:, :d])
            
            ss_res = np.sum((Z_t1_test[:, :d] - Z_t1_pred_pca) ** 2)
            ss_tot = np.sum((Z_t1_test[:, :d] - Z_t1_test[:, :d].mean(axis=0)) ** 2)
            r2_pca = 1 - ss_res / max(ss_tot, 1e-10)
            pca_r2_folds.append(float(r2_pca))
            
            # === Predictive basis: use top-d predictive directions ===
            # Project Z_t onto top-d right singular vectors of W
            P_pred = Vt_w[:d, :]  # (d, d_pca) — predictive projection
            Z_t_pred_train = Z_t_train @ P_pred.T  # (N_train, d)
            Z_t_pred_test = Z_t_test @ P_pred.T     # (N_test, d)
            Z_t1_pred_train = Z_t1_train @ P_pred.T  # (N_train, d)
            Z_t1_pred_test = Z_t1_test @ P_pred.T    # (N_test, d)
            
            reg_pred = Ridge(alpha=0.1)
            reg_pred.fit(Z_t_pred_train, Z_t1_pred_train)
            Z_t1_pred_pred = reg_pred.predict(Z_t_pred_test)
            
            ss_res = np.sum((Z_t1_pred_test - Z_t1_pred_pred) ** 2)
            ss_tot = np.sum((Z_t1_pred_test - Z_t1_pred_test.mean(axis=0)) ** 2)
            r2_pred = 1 - ss_res / max(ss_tot, 1e-10)
            pred_r2_folds.append(float(r2_pred))
        
        pca_mean = float(np.mean(pca_r2_folds))
        pca_std = float(np.std(pca_r2_folds))
        pred_mean = float(np.mean(pred_r2_folds))
        pred_std = float(np.std(pred_r2_folds))
        
        print(f"  d={d:3d}: PCA R²={pca_mean:.4f}±{pca_std:.4f}, "
              f"Pred R²={pred_mean:.4f}±{pred_std:.4f}, "
              f"Δ={pred_mean - pca_mean:+.4f}")
        
        results_per_d[str(d)] = {
            "pca_r2_mean": pca_mean, "pca_r2_std": pca_std,
            "pred_r2_mean": pred_mean, "pred_r2_std": pred_std,
            "delta": float(pred_mean - pca_mean),
        }
    
    results = {"dims_tested": dims_to_test, "per_dim": results_per_d}
    return results


def exp4_markov_property(H_tokens, embeddings_list, d_pca=100):
    """
    Exp 4: Markov Property Test
    
    Question: Can h(t+1) be predicted from h(t) alone, or does the new token matter?
    
    Method:
      A) Predict h(t+1) from h(t) alone → R²_A
      B) Predict h(t+1) from h(t) + embed(token[t+1]) → R²_B
      ΔR² = R²_B - R²_A tells us how much the new token matters.
    
    If ΔR² ≈ 0: h(t) is approximately Markov (history is fully compressed)
    If ΔR² >> 0: new token injection is crucial (state is NOT Markov)
    """
    print("\n" + "=" * 60)
    print("Exp 4: Markov Property Test")
    print("=" * 60)
    
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    
    # Build pairs
    H_t_list = []
    H_t1_list = []
    E_new_list = []  # new token embedding at position t+1
    
    for i, h in enumerate(H_tokens):
        T = h.shape[0]
        if T < 3:
            continue
        emb = embeddings_list[i]
        for t in range(T - 1):
            H_t_list.append(h[t])
            H_t1_list.append(h[t + 1])
            E_new_list.append(emb[t + 1])  # embedding of the NEW token
    
    H_t = np.array(H_t_list, dtype=np.float32)     # (N, d_model)
    H_t1 = np.array(H_t1_list, dtype=np.float32)   # (N, d_model)
    E_new = np.array(E_new_list, dtype=np.float32) # (N, d_model)
    N = H_t.shape[0]
    
    print(f"  Token transition pairs: {N}")
    
    # PCA on combined h(t) and h(t+1)
    pca = PCA(n_components=min(d_pca, N, H_t.shape[1]))
    pca.fit(np.vstack([H_t, H_t1]))
    
    Z_t = pca.transform(H_t)    # (N, d_pca)
    Z_t1 = pca.transform(H_t1)  # (N, d_pca)
    
    # PCA for new token embeddings (same d_pca for fair comparison)
    pca_e = PCA(n_components=min(d_pca, N, E_new.shape[1]))
    pca_e.fit(E_new)
    Z_new = pca_e.transform(E_new)  # (N, d_pca_e)
    
    print(f"  PCA: h(t) → {d_pca} dims, embed → {Z_new.shape[1]} dims")
    
    # Test at multiple dimensions
    dims_to_test = [2, 5, 10, 20, 30, 50]
    dims_to_test = [d for d in dims_to_test if d <= d_pca]
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    results_per_d = {}
    
    for d in dims_to_test:
        r2_alone_folds = []    # R² using h(t) alone
        r2_with_new_folds = [] # R² using h(t) + new token embedding
        
        for train_idx, test_idx in kf.split(range(N)):
            Z_t_train, Z_t_test = Z_t[train_idx, :d], Z_t[test_idx, :d]
            Z_t1_train, Z_t1_test = Z_t1[train_idx, :d], Z_t1[test_idx, :d]
            Z_new_train, Z_new_test = Z_new[train_idx], Z_new[test_idx]
            
            # A) h(t) alone → predict h(t+1)
            reg_alone = Ridge(alpha=0.1)
            reg_alone.fit(Z_t_train, Z_t1_train)
            r2_alone = reg_alone.score(Z_t_test, Z_t1_test)
            r2_alone_folds.append(float(r2_alone))
            
            # B) h(t) + new token embedding → predict h(t+1)
            # Concatenate h(t) and embed(token[t+1]) in PCA space
            d_e = min(d, Z_new.shape[1])
            combined_train = np.hstack([Z_t_train, Z_new_train[:, :d_e]])
            combined_test = np.hstack([Z_t_test, Z_new_test[:, :d_e]])
            
            reg_with = Ridge(alpha=0.1)
            reg_with.fit(combined_train, Z_t1_train)
            r2_with = reg_with.score(combined_test, Z_t1_test)
            r2_with_new_folds.append(float(r2_with))
        
        alone_mean = float(np.mean(r2_alone_folds))
        alone_std = float(np.std(r2_alone_folds))
        with_mean = float(np.mean(r2_with_new_folds))
        with_std = float(np.std(r2_with_new_folds))
        delta = with_mean - alone_mean
        
        print(f"  d={d:3d}: h(t) alone R²={alone_mean:.4f}±{alone_std:.4f}, "
              f"h(t)+embed R²={with_mean:.4f}±{with_std:.4f}, "
              f"ΔR²={delta:+.4f}")
        
        results_per_d[str(d)] = {
            "alone_r2_mean": alone_mean, "alone_r2_std": alone_std,
            "with_new_r2_mean": with_mean, "with_new_r2_std": with_std,
            "delta_r2": float(delta),
        }
    
    results = {"dims_tested": dims_to_test, "per_dim": results_per_d}
    return results


def exp5_token_vs_layer_dynamics(model, tokenizer, device, model_info, H_tokens, d_pca=100):
    """
    Exp 5: Token vs Layer Dynamics Comparison
    
    Question: Is token-to-token dynamics more or less low-dimensional than layer-to-layer?
    
    Method:
      - Token dynamics: predict h_L(t+1) from h_L(t) at the last layer
      - Layer dynamics: predict h_{ℓ+1}(0) from h_ℓ(0) at the first token position
      - Compare effective dimensionality
    """
    print("\n" + "=" * 60)
    print("Exp 5: Token vs Layer Dynamics Comparison")
    print("=" * 60)
    
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    # === Token dynamics ===
    # Already have H_tokens from Exp 1
    
    H_t_list = []
    H_t1_list = []
    for h in H_tokens:
        T = h.shape[0]
        if T < 3:
            continue
        for t in range(T - 1):
            H_t_list.append(h[t])
            H_t1_list.append(h[t + 1])
    
    H_t = np.array(H_t_list, dtype=np.float32)
    H_t1_tok = np.array(H_t1_list, dtype=np.float32)
    N_token = H_t.shape[0]
    
    # === Layer dynamics ===
    # Need to collect hidden states across layers for the first token position
    # Use a subset of prompts for efficiency
    n_layer_prompts = min(100, len(H_tokens) * 2)
    
    # Re-generate prompts for layer analysis
    layer_prompts = generate_diverse_prompts(n_layer_prompts)
    
    try:
        input_device = next(model.parameters()).device
    except StopIteration:
        input_device = device
    
    H_layer = np.zeros((n_layer_prompts, n_layers + 1, d_model), dtype=np.float32)
    n_successful_layer = 0
    
    print(f"  Collecting layer data for {n_layer_prompts} prompts...")
    for i, prompt in enumerate(layer_prompts[:n_layer_prompts]):
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(input_device)
            attention_mask = inputs["attention_mask"].to(input_device)
            
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
            
            has_nan = False
            for l in range(n_layers + 1):
                h = out.hidden_states[l][0, -1, :].detach().float().cpu().numpy()
                if np.any(np.isnan(h)) or np.any(np.isinf(h)):
                    has_nan = True
                    break
                H_layer[n_successful_layer, l] = h
            
            if not has_nan:
                n_successful_layer += 1
            
            del out
        except Exception:
            pass
    
    H_layer = H_layer[:n_successful_layer]
    print(f"  Layer data: {n_successful_layer} successful prompts")
    
    # === Compare dimensionality ===
    
    # Token dynamics PCA
    pca_tok = PCA(n_components=min(d_pca, N_token, d_model))
    pca_tok.fit(np.vstack([H_t, H_t1_tok]))
    cumvar_tok = np.cumsum(pca_tok.explained_variance_ratio_)
    
    # Layer dynamics PCA
    H_layer_flat = H_layer.reshape(-1, d_model)
    pca_lay = PCA(n_components=min(d_pca, H_layer_flat.shape[0], d_model))
    pca_lay.fit(H_layer_flat)
    cumvar_lay = np.cumsum(pca_lay.explained_variance_ratio_)
    
    # Effective dimensionality comparison
    for threshold in [0.90, 0.95, 0.99]:
        d_tok = int(np.searchsorted(cumvar_tok, threshold)) + 1
        d_lay = int(np.searchsorted(cumvar_lay, threshold)) + 1
        print(f"  d for {threshold*100:.0f}% var: token={d_tok}, layer={d_lay}")
    
    # Prediction R² comparison
    # Token dynamics: predict h_L(t+1) from h_L(t) in PCA space
    Z_t_tok = pca_tok.transform(H_t)
    Z_t1_tok = pca_tok.transform(H_t1_tok)
    
    # Layer dynamics: predict h_{ℓ+1}(0) from h_ℓ(0) for each layer pair
    # Average across layer pairs
    layer_r2_values = []
    sample_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2]
    
    for l in sample_layers:
        if l >= n_layers:
            continue
        X_lay = H_layer[:, l, :]
        Y_lay = H_layer[:, l+1, :]
        Z_X = pca_lay.transform(X_lay)
        Z_Y = pca_lay.transform(Y_lay)
        
        # Use d_pca dimensions
        d_use = min(d_pca, Z_X.shape[1])
        reg = Ridge(alpha=0.1)
        reg.fit(Z_X[:, :d_use], Z_Y[:, :d_use])
        r2 = reg.score(Z_X[:, :d_use], Z_Y[:, :d_use])
        layer_r2_values.append(float(r2))
    
    # Token dynamics R²
    tok_r2_values = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    d_use = min(d_pca, Z_t_tok.shape[1])
    
    for train_idx, test_idx in kf.split(range(N_token)):
        reg = Ridge(alpha=0.1)
        reg.fit(Z_t_tok[train_idx, :d_use], Z_t1_tok[train_idx, :d_use])
        r2 = reg.score(Z_t_tok[test_idx, :d_use], Z_t1_tok[test_idx, :d_use])
        tok_r2_values.append(float(r2))
    
    print(f"  Token dynamics R² (d={d_use}): {np.mean(tok_r2_values):.4f}±{np.std(tok_r2_values):.4f}")
    print(f"  Layer dynamics R² (d={d_use}): {np.mean(layer_r2_values):.4f}±{np.std(layer_r2_values):.4f}")
    
    # Key comparison: how well does d=2, d=5, d=10 predict?
    for d in [2, 5, 10, 20, 30]:
        if d > d_use:
            continue
        
        # Token
        tok_r2_d = []
        for train_idx, test_idx in kf.split(range(N_token)):
            reg = Ridge(alpha=0.1)
            reg.fit(Z_t_tok[train_idx, :d], Z_t1_tok[train_idx, :d])
            r2 = reg.score(Z_t_tok[test_idx, :d], Z_t1_tok[test_idx, :d])
            tok_r2_d.append(float(r2))
        
        # Layer (at L0→L1 for comparison)
        if H_layer.shape[0] > 10:
            Z_X_lay = pca_lay.transform(H_layer[:, 0, :])
            Z_Y_lay = pca_lay.transform(H_layer[:, 1, :])
            reg_lay = Ridge(alpha=0.1)
            reg_lay.fit(Z_X_lay[:, :d], Z_Y_lay[:, :d])
            r2_lay = reg_lay.score(Z_X_lay[:, :d], Z_Y_lay[:, :d])
        else:
            r2_lay = 0.0
        
        print(f"  d={d:3d}: Token R²={np.mean(tok_r2_d):.4f}, Layer(0→1) R²={r2_lay:.4f}")
    
    results = {
        "N_token_transitions": N_token,
        "N_layer_prompts": n_successful_layer,
        "cumvar_token_at_dims": {str(d): float(cumvar_tok[d-1]) for d in [2,5,10,20,30,50,100] if d-1 < len(cumvar_tok)},
        "cumvar_layer_at_dims": {str(d): float(cumvar_lay[d-1]) for d in [2,5,10,20,30,50,100] if d-1 < len(cumvar_lay)},
        "token_r2_full": {"mean": float(np.mean(tok_r2_values)), "std": float(np.std(tok_r2_values))},
        "layer_r2_full": {"mean": float(np.mean(layer_r2_values)), "std": float(np.std(layer_r2_values))},
    }
    
    return results


# ===== 4. MAIN =====

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    print(f"\n{'='*60}")
    print(f"Phase 160: Token Trajectory Dynamics — {model_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*60}")
    
    # Load model
    t0 = time.time()
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    print(f"  Model: {model_info.model_class}, {model_info.n_layers} layers, "
          f"d_model={model_info.d_model}")
    
    # Generate diverse prompts
    prompts = generate_diverse_prompts(500)
    
    # Collect token trajectory data
    print(f"\n--- Collecting token trajectory data ---")
    H_tokens, token_ids_list, embeddings_list = collect_token_trajectory_data(
        model, tokenizer, device, model_info, prompts)
    
    if len(H_tokens) < 10:
        print("ERROR: Too few successful prompts. Aborting.")
        return
    
    # Run experiments
    all_results = {
        "model_name": model_name,
        "model_class": model_info.model_class,
        "n_layers": model_info.n_layers,
        "d_model": model_info.d_model,
        "n_prompts": len(H_tokens),
        "total_tokens": sum(h.shape[0] for h in H_tokens),
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M'),
    }
    
    # Exp 1: PCA dimensionality
    r1, pca, H_all = exp1_token_trajectory_pca(H_tokens, model_info.d_model)
    all_results["exp1_pca"] = r1
    
    # Exp 2: Predictive dimensionality
    r2, pca_2, W, pred_dirs, Z_t, Z_t1 = exp2_predictive_dimensionality(H_tokens, d_pca=100)
    all_results["exp2_predictive"] = r2
    
    # Exp 3: Predictive vs PCA comparison
    r3 = exp3_predictive_vs_pca(Z_t, Z_t1, pca_2, W, pred_dirs, model_info.d_model)
    all_results["exp3_pred_vs_pca"] = r3
    
    # Exp 4: Markov property
    r4 = exp4_markov_property(H_tokens, embeddings_list, d_pca=100)
    all_results["exp4_markov"] = r4
    
    # Exp 5: Token vs Layer comparison
    r5 = exp5_token_vs_layer_dynamics(model, tokenizer, device, model_info, H_tokens, d_pca=100)
    all_results["exp5_token_vs_layer"] = r5
    
    # Release model
    release_model(model)
    
    # Save results
    d_pca_actual = min(100, H_all.shape[0], H_all.shape[1])
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(x) for x in obj]
        return obj
    
    all_results = convert_numpy(all_results)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Phase 160 Summary — {model_name}")
    print(f"{'='*60}")
    print(f"  Total token transitions: {all_results['total_tokens']}")
    print(f"  PCA dim for 90% var: {r1.get('d_90', 'N/A')}")
    print(f"  Predictive dim (SVD of W, 90%): {r2.get('d_pred_90', 'N/A')}")
    print(f"  Markov ΔR² (d=30): {r4.get('per_dim', {}).get('30', {}).get('delta_r2', 'N/A')}")
    
    # Save to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    out_path = f"tests/glm5_temp/phase160_{model_name}_{timestamp}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to: {out_path}")
    
    elapsed = time.time() - t0
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == "__main__":
    main()
