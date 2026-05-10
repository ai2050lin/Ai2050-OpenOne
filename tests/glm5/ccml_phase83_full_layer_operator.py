"""
Phase 83: Full Layer Operator Decomposition — From Components to Primitives
=============================================================================

Phase 82 critique identified 4 CRITICAL gaps:

1. LayerNorm's dominant role IGNORED
   - LN is nonlinear, global coupling, changes geometry
   - Many observations (low-rank, directional amplification) may come from LN, not MLP

2. Attention Jacobian is the REAL task-conditioned operator
   - Attention direction cosine ~0.01 (vs MLP eigenspace overlap ~0.2-0.5)
   - J_attn is where computation policy lives

3. Jacobian is single-step local, not rollout computation
   - Need to study F^k(h) for reasoning/CoT/planning

4. Operator Basis Decomposition
   - The "conditional linear computation" hypothesis:
     F(h,x) = Σ_i g_i(x) A_i h
   - Shared basis operators + task-conditioned gating coefficients

Four Experiments:
  A: LayerNorm Jacobian Analysis ★★★★★
     - Analytical computation of J_ln
     - Compare J_ln spectrum/structure with J_mlp
     - How much of "low-rank" and "directional amplification" comes from LN?

  B: Attention Jacobian ★★★★★ (MOST CRITICAL)
     - Analytical computation from cached attention patterns
     - Cross-task comparison of J_attn
     - Is J_attn more task-specific than J_mlp?

  C: Full Layer Jacobian Decomposition ★★★★★
     - J_full = (I + J_mlp @ J_ln2) @ (I + J_attn @ J_ln1)
     - Quantify each component's contribution to task-specificity
     - Which component carries the most task-specific information?

  D: Operator Basis Decomposition ★★★★★ (THEORETICAL CORE)
     - Collect all per-task average Jacobians
     - SVD of stacked Jacobians → basis operators A_1...A_k
     - Test: J_task ≈ Σ_i c_i(task) A_i?
     - This directly tests the "conditional linear computation" hypothesis

Key insight from critique:
  Transformer computation = shared universal basis × task-conditioned subspace routing
                           × recursive rollout amplification

  The "language computation algebra" may not be in individual operators,
  but in the COMBINATION COEFFICIENTS (gating).

Usage:
  python ccml_phase83_full_layer_operator.py --exp a
  python ccml_phase83_full_layer_operator.py --exp b
  python ccml_phase83_full_layer_operator.py --exp c
  python ccml_phase83_full_layer_operator.py --exp d
  python ccml_phase83_full_layer_operator.py --exp all
"""

import torch
import numpy as np
import argparse
import time
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


def generate_prompts(task, n):
    prompts = []
    np.random.seed(42)
    if task == "addition":
        for i in range(n):
            a, b = np.random.randint(1, 50), np.random.randint(1, 50)
            prompts.append(f"{a} + {b} =")
    elif task == "antonym":
        words = ["hot", "big", "fast", "happy", "strong", "light", "good", "old",
                 "rich", "tall", "love", "warm", "clean", "safe", "hard", "loud",
                 "bright", "sharp", "sweet", "brave"]
        for i in range(n):
            w = words[i % len(words)]
            prompts.append(f"The opposite of {w} is")
    elif task == "capital":
        countries = ["France", "Germany", "Japan", "Brazil", "Italy",
                     "Spain", "China", "India", "Egypt", "Australia",
                     "Canada", "Mexico", "Korea", "Russia", "Turkey",
                     "Norway", "Sweden", "Poland", "Greece", "Thailand"]
        for i in range(n):
            c = countries[i % len(countries)]
            prompts.append(f"The capital of {c} is")
    elif task == "translate_fr":
        words = ["cat", "dog", "house", "water", "book", "tree", "sun", "moon",
                 "fire", "earth", "heart", "hand", "night", "day", "star",
                 "flower", "bird", "fish", "rain", "snow"]
        for i in range(n):
            w = words[i % len(words)]
            prompts.append(f"The French word for {w} is")
    return prompts


# ============================================================
# LayerNorm Jacobian (Analytical)
# ============================================================
def compute_ln_jacobian(h_input, ln_module):
    """
    Compute the analytical Jacobian of LayerNorm.

    LN(x) = gamma * (x - mu) / sigma + beta

    where mu = mean(x), sigma = std(x)

    J_ln[i,j] = gamma[i] / sigma * (delta_{ij} - 1/n - (x_i - mu)(x_j - mu) / (n * sigma^2))

    This can be written as:
    J_ln = (1/sigma) * diag(gamma) @ (I - (1/n) 11^T - (1/(n*sigma^2)) (x-mu)(x-mu)^T)
    """
    x = h_input.detach().clone()
    n = x.shape[-1]
    gamma = ln_module.w.detach()  # [d_model]
    # beta doesn't affect Jacobian (additive constant)

    mu = x.mean()
    sigma = x.std(correction=0)  # population std

    x_centered = x - mu  # [d_model]

    # J_ln = (1/sigma) * diag(gamma) @ (I - (1/n) 11^T - (1/(n*sigma^2)) x_c x_c^T)
    # This is a rank-2 modification of a diagonal matrix

    # Compute efficiently:
    # J @ v = (1/sigma) * gamma * (v - (1/n)*sum(v) - (1/(n*sigma^2)) * <x_c, v> * x_c)

    # For the full matrix:
    J = torch.zeros(n, n, device=x.device)

    # Diagonal part: gamma_i / sigma * (1 - 1/n - x_c_i^2 / (n * sigma^2))
    diag_vals = gamma / sigma * (1 - 1/n - x_centered**2 / (n * sigma**2))
    J = torch.diag(diag_vals)

    # Off-diagonal part: gamma_i / sigma * (-1/n - x_c_i * x_c_j / (n * sigma^2))
    # This can be written as outer product
    # -gamma/sigma * (1/n * 11^T + (1/(n*sigma^2)) x_c x_c^T)
    off_diag = -gamma.unsqueeze(1) / sigma * (
        torch.ones(n, 1, device=x.device) / n +
        x_centered.unsqueeze(1) * x_centered.unsqueeze(0) / (n * sigma**2)
    )
    # Only set off-diagonal
    J = J + off_diag * (1 - torch.eye(n, device=x.device))

    # Actually, let me just compute it directly
    # J[i,j] = gamma[i] / sigma * (delta_{ij} - 1/n - x_c[i]*x_c[j] / (n*sigma^2))
    rank1 = torch.outer(torch.ones(n, device=x.device), torch.ones(n, device=x.device)) / n
    rank2 = torch.outer(x_centered, x_centered) / (n * sigma**2)

    J = torch.diag(gamma) / sigma @ (torch.eye(n, device=x.device) - rank1 - rank2)

    return J


def compute_mlp_jacobian(model, cache, layer, pos=-1):
    """Compute exact MLP Jacobian (from Phase 81)."""
    pre_gelu = cache[f'blocks.{layer}.mlp.hook_pre'][pos].detach()
    x = pre_gelu.clone().requires_grad_(True)
    y = torch.nn.functional.gelu(x)
    gelu_deriv = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))[0].detach()
    W_in = model.blocks[layer].mlp.W_in.detach()
    W_out = model.blocks[layer].mlp.W_out.detach()
    scaled_W_in_T = gelu_deriv.unsqueeze(1) * W_in.T
    J = W_out.T @ scaled_W_in_T
    return J, gelu_deriv


def compute_attn_jacobian(model, cache, layer, pos=-1):
    """
    Compute the analytical Jacobian of attention output w.r.t. LN input at position pos.

    attn_out = Σ_h W_O_h^T z_h
    where z_h = Σ_s α_{h,s} v_{h,s}

    Three contributions to ∂attn_out/∂x_pos:
    1. Query contribution (through attention weights)
    2. Value contribution (at position pos)
    3. Key contribution (at position pos)
    """
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    d_model = model.cfg.d_model

    # Get cached values
    pattern = cache[f'blocks.{layer}.attn.hook_pattern'][pos].detach()  # [n_heads, seq, seq]
    # Actually, pattern is [n_heads, seq_q, seq_k]
    # For position pos (last), we want pattern[:, pos, :]
    # But pos=-1 means last position
    z = cache[f'blocks.{layer}.attn.hook_z'][pos].detach()  # [n_heads, d_head]

    # We need the sequence length
    h_pre = cache[f'blocks.{layer}.hook_resid_pre']  # [seq, d_model]
    seq_len = h_pre.shape[0]

    # Get the LN input for the last position
    h_in = h_pre[pos].detach()  # [d_model]

    # Compute LN output (input to attention)
    ln1 = model.blocks[layer].ln1
    x_attn = ln1(h_in.unsqueeze(0)).squeeze(0).detach()  # [d_model]

    # Get weight matrices
    W_Q = model.blocks[layer].attn.W_Q.detach()  # [n_heads, d_model, d_head]
    W_K = model.blocks[layer].attn.W_K.detach()  # [n_heads, d_model, d_head]
    W_V = model.blocks[layer].attn.W_V.detach()  # [n_heads, d_model, d_head]
    W_O = model.blocks[layer].attn.W_O.detach()  # [n_heads, d_head, d_model]

    # Compute Q, K, V for all positions (after LN)
    # We need the LN output for ALL positions to compute K and V
    h_pre_all = h_pre.detach()  # [seq, d_model]
    x_attn_all = ln1(h_pre_all).detach()  # [seq, d_model]

    # Q for last position: [n_heads, d_head]
    q = torch.einsum('hmd,m->hd', W_Q, x_attn)  # [n_heads, d_head]

    # K, V for all positions: [seq, n_heads, d_head]
    k = torch.einsum('hmd,sm->shd', W_K, x_attn_all)  # [seq, n_heads, d_head]
    v = torch.einsum('hmd,sm->shd', W_V, x_attn_all)  # [seq, n_heads, d_head]

    # Attention pattern for last position: [n_heads, seq]
    alpha = cache[f'blocks.{layer}.attn.hook_pattern'][:, pos, :].detach()  # [n_heads, seq]

    # Weighted average of keys: k_bar_h = Σ_s α_{h,s} k_{h,s} [n_heads, d_head]
    k_bar = torch.einsum('hs,shd->hd', alpha, k)  # [n_heads, d_head]

    # z_h = Σ_s α_{h,s} v_{h,s} [n_heads, d_head]
    z_h = torch.einsum('hs,shd->hd', alpha, v)  # [n_heads, d_head]

    # Now compute the Jacobian ∂attn_out/∂x_attn (w.r.t. LN output at last position)
    # attn_out = Σ_h W_O_h^T z_h

    J_attn = torch.zeros(d_model, d_model, device=h_in.device)

    for h in range(n_heads):
        alpha_h = alpha[h]  # [seq]
        q_h = q[h]  # [d_head]
        k_h = k[:, h, :]  # [seq, d_head]
        v_h = v[:, h, :]  # [seq, d_head]
        k_bar_h = k_bar[h]  # [d_head]
        z_h_val = z_h[h]  # [d_head]
        W_O_h = W_O[h]  # [d_head, d_model]
        W_Q_h = W_Q[h]  # [d_model, d_head]
        W_K_h = W_K[h]  # [d_model, d_head]
        W_V_h = W_V[h]  # [d_model, d_head]

        # Contribution 1: Query (through attention weights)
        # ∂z_h/∂q_h = (1/√d_k) Σ_s α_{h,s} v_{h,s} (k_{h,s} - k̄_h)^T
        # Then ∂z_h/∂x_attn = (∂z_h/∂q_h) W_Q_h^T
        query_contrib_z = torch.zeros(d_head, d_head, device=h_in.device)
        for s in range(seq_len):
            diff = k_h[s] - k_bar_h  # [d_head]
            query_contrib_z += alpha_h[s] * torch.outer(v_h[s], diff)  # [d_head, d_head]
        query_contrib_z /= np.sqrt(d_head)

        # ∂z_h/∂x_attn (through q) = query_contrib_z @ W_Q_h^T  [d_head, d_model]
        dz_dx_query = query_contrib_z @ W_Q_h.T  # [d_head, d_model]

        # Contribution to attn_out: W_O_h^T @ dz_dx_query  [d_model, d_model]
        J_attn += W_O_h.T @ dz_dx_query  # [d_model, d_model]

        # Contribution 2: Value (at last position)
        # ∂z_h/∂x_attn (through v_{h,last}) = α_{h,last} W_V_h^T  [d_head, d_model]
        dz_dx_value = alpha_h[-1] * W_V_h.T  # [d_head, d_model]

        # Contribution to attn_out
        J_attn += W_O_h.T @ dz_dx_value  # [d_model, d_model]

        # Contribution 3: Key (at last position)
        # ∂z_h/∂x_attn (through k_{h,last}) = α_{h,last} (v_{h,last} - z_h) (q_h^T/√d_k) W_K_h^T
        # This is a rank-1 update
        key_vec = alpha_h[-1] * (v_h[-1] - z_h_val)  # [d_head]
        scale = 1.0 / np.sqrt(d_head)
        # rank-1: key_vec @ (q_h^T @ W_K_h^T / √d_k)  [d_head, d_model]
        dz_dx_key = scale * torch.outer(key_vec, q_h) @ W_K_h.T  # [d_head, d_model]

        # Contribution to attn_out
        J_attn += W_O_h.T @ dz_dx_key  # [d_model, d_model]

    return J_attn


# ============================================================
# Experiment A: LayerNorm Jacobian Analysis
# ============================================================
def exp_a_layernorm():
    print("=" * 70)
    print("EXPERIMENT A: LayerNorm Jacobian Analysis")
    print("=" * 70)

    model = get_model()
    tasks = ["addition", "antonym", "capital", "translate_fr"]
    layers = [3, 6, 9]
    n_samples = 50

    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer}")
        print(f"{'='*60}")

        for task in tasks:
            prompts = generate_prompts(task, n_samples)
            ln1_jacobians = []
            ln2_jacobians = []
            mlp_jacobians = []

            for prompt in prompts[:20]:  # 20 samples for speed
                tokens = model.to_tokens(prompt)
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

                h_in = cache[f'blocks.{layer}.hook_resid_pre'][-1].detach()
                h_mid = cache[f'blocks.{layer}.hook_resid_mid'][-1].detach()

                # LN1 Jacobian (before attention)
                ln1 = model.blocks[layer].ln1
                J_ln1 = compute_ln_jacobian(h_in, ln1)

                # LN2 Jacobian (before MLP)
                ln2 = model.blocks[layer].ln2
                J_ln2 = compute_ln_jacobian(h_mid, ln2)

                # MLP Jacobian
                J_mlp, _ = compute_mlp_jacobian(model, cache, layer)

                ln1_jacobians.append(J_ln1)
                ln2_jacobians.append(J_ln2)
                mlp_jacobians.append(J_mlp)

            # Average Jacobians
            J_ln1_avg = torch.stack(ln1_jacobians).mean(0)
            J_ln2_avg = torch.stack(ln2_jacobians).mean(0)
            J_mlp_avg = torch.stack(mlp_jacobians).mean(0)

            # Spectral comparison
            S_ln1 = torch.linalg.svdvals(J_ln1_avg)
            S_ln2 = torch.linalg.svdvals(J_ln2_avg)
            S_mlp = torch.linalg.svdvals(J_mlp_avg)

            print(f"\n  {task:15s}:")
            print(f"    LN1: op_norm={S_ln1[0]:.4f}, top-5=[{', '.join(f'{s:.3f}' for s in S_ln1[:5].tolist())}], "
                  f"rank90={np.searchsorted(np.cumsum(S_ln1.numpy())/S_ln1.sum(), 0.9)}")
            print(f"    LN2: op_norm={S_ln2[0]:.4f}, top-5=[{', '.join(f'{s:.3f}' for s in S_ln2[:5].tolist())}], "
                  f"rank90={np.searchsorted(np.cumsum(S_ln2.numpy())/S_ln2.sum(), 0.9)}")
            print(f"    MLP: op_norm={S_mlp[0]:.4f}, top-5=[{', '.join(f'{s:.3f}' for s in S_mlp[:5].tolist())}], "
                  f"rank90={np.searchsorted(np.cumsum(S_mlp.numpy())/S_mlp.sum(), 0.9)}")

            # Cross-task LN Jacobian similarity
            # (compute later when all tasks are available)

        # Cross-task LN Jacobian comparison
        print(f"\n--- Cross-Task LN Jacobian Similarity ---")
        task_ln_jacobians = {}
        for task in tasks:
            prompts = generate_prompts(task, 20)
            ln1_js = []
            for prompt in prompts:
                tokens = model.to_tokens(prompt)
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
                h_in = cache[f'blocks.{layer}.hook_resid_pre'][-1].detach()
                ln1 = model.blocks[layer].ln1
                J_ln1 = compute_ln_jacobian(h_in, ln1)
                ln1_js.append(J_ln1)
            task_ln_jacobians[task] = torch.stack(ln1_js).mean(0)

        print(f"  LN1 Flattened cosine:")
        header = f"  {'':15s}" + "".join(f"{t[:8]:>9s}" for t in tasks)
        print(header)
        for i, t1 in enumerate(tasks):
            row = f"  {t1:15s}"
            for j, t2 in enumerate(tasks):
                cos = torch.nn.functional.cosine_similarity(
                    task_ln_jacobians[t1].flatten().unsqueeze(0),
                    task_ln_jacobians[t2].flatten().unsqueeze(0)
                ).item()
                row += f"{cos:>9.4f}"
            print(row)

    print("\n" + "=" * 70)
    print("EXPERIMENT A COMPLETE")
    print("=" * 70)


# ============================================================
# Experiment B: Attention Jacobian
# ============================================================
def exp_b_attention_jacobian():
    print("=" * 70)
    print("EXPERIMENT B: Attention Jacobian")
    print("=" * 70)

    model = get_model()
    tasks = ["addition", "antonym", "capital", "translate_fr"]
    layers = [3, 6, 9]
    n_samples = 30

    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer}")
        print(f"{'='*60}")

        task_attn_jacobians = {}
        task_mlp_jacobians = {}

        for task in tasks:
            prompts = generate_prompts(task, n_samples)
            attn_js = []
            mlp_js = []

            for prompt in prompts:
                tokens = model.to_tokens(prompt)
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

                # Attention Jacobian
                J_attn = compute_attn_jacobian(model, cache, layer)
                attn_js.append(J_attn)

                # MLP Jacobian
                J_mlp, _ = compute_mlp_jacobian(model, cache, layer)
                mlp_js.append(J_mlp)

            task_attn_jacobians[task] = torch.stack(attn_js)
            task_mlp_jacobians[task] = torch.stack(mlp_js)

        # ---- Part 1: Attention Jacobian statistics ----
        print(f"\n--- Attention Jacobian Statistics ---")
        for task in tasks:
            J_avg = task_attn_jacobians[task].mean(0)
            S = torch.linalg.svdvals(J_avg)
            print(f"  {task:15s}: op_norm={S[0]:.4f}, top-5=[{', '.join(f'{s:.3f}' for s in S[:5].tolist())}], "
                  f"rank90={np.searchsorted(np.cumsum(S.numpy())/S.sum(), 0.9)}")

        # ---- Part 2: Cross-task Attention Jacobian similarity ----
        print(f"\n--- Cross-Task Attention Jacobian Similarity (Flattened cosine) ---")
        task_attn_avg = {t: js.mean(0) for t, js in task_attn_jacobians.items()}

        header = f"  {'':15s}" + "".join(f"{t[:8]:>9s}" for t in tasks)
        print(header)
        for i, t1 in enumerate(tasks):
            row = f"  {t1:15s}"
            for j, t2 in enumerate(tasks):
                cos = torch.nn.functional.cosine_similarity(
                    task_attn_avg[t1].flatten().unsqueeze(0),
                    task_attn_avg[t2].flatten().unsqueeze(0)
                ).item()
                row += f"{cos:>9.4f}"
            print(row)

        # ---- Part 3: Compare with MLP Jacobian similarity ----
        print(f"\n--- Cross-Task MLP Jacobian Similarity (Flattened cosine) ---")
        task_mlp_avg = {t: js.mean(0) for t, js in task_mlp_jacobians.items()}

        header = f"  {'':15s}" + "".join(f"{t[:8]:>9s}" for t in tasks)
        print(header)
        for i, t1 in enumerate(tasks):
            row = f"  {t1:15s}"
            for j, t2 in enumerate(tasks):
                cos = torch.nn.functional.cosine_similarity(
                    task_mlp_avg[t1].flatten().unsqueeze(0),
                    task_mlp_avg[t2].flatten().unsqueeze(0)
                ).item()
                row += f"{cos:>9.4f}"
            print(row)

        # ---- Part 4: Attention vs MLP eigenspace overlap ----
        print(f"\n--- Eigenspace Overlap: J_attn vs J_mlp (top-k) ---")
        for k in [5, 10, 20]:
            print(f"  Top-{k}:")
            for task in tasks:
                U_attn, S_attn, _ = torch.linalg.svd(task_attn_avg[task], full_matrices=False)
                U_mlp, S_mlp, _ = torch.linalg.svd(task_mlp_avg[task], full_matrices=False)

                # Subspace overlap
                P_attn = U_attn[:, :k] @ U_attn[:, :k].T
                proj = U_mlp[:, :k].T @ P_attn @ U_mlp[:, :k]
                overlap = torch.trace(proj) / k

                print(f"    {task:15s}: overlap={overlap:.4f}")

        # ---- Part 5: Task-specificity index ----
        print(f"\n--- Task-Specificity Index: ||Delta J|| / ||J|| ---")
        # Compare cross-task variation for attn vs mlp
        for task1, task2 in [("addition", "antonym"), ("addition", "capital"), ("addition", "translate_fr")]:
            J_attn_1 = task_attn_avg[task1]
            J_attn_2 = task_attn_avg[task2]
            J_mlp_1 = task_mlp_avg[task1]
            J_mlp_2 = task_mlp_avg[task2]

            delta_attn = (J_attn_1 - J_attn_2).norm() / ((J_attn_1.norm() + J_attn_2.norm()) / 2)
            delta_mlp = (J_mlp_1 - J_mlp_2).norm() / ((J_mlp_1.norm() + J_mlp_2.norm()) / 2)

            print(f"  {task1} vs {task2}: "
                  f"attn_delta={delta_attn:.4f}, mlp_delta={delta_mlp:.4f}, "
                  f"ratio(attn/mlp)={delta_attn/(delta_mlp+1e-10):.2f}")

    print("\n" + "=" * 70)
    print("EXPERIMENT B COMPLETE")
    print("=" * 70)


# ============================================================
# Experiment C: Full Layer Jacobian Decomposition
# ============================================================
def exp_c_full_decomposition():
    print("=" * 70)
    print("EXPERIMENT C: Full Layer Jacobian Decomposition")
    print("=" * 70)

    model = get_model()
    tasks = ["addition", "antonym", "capital", "translate_fr"]
    layers = [3, 6, 9]
    n_samples = 30

    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer}")
        print(f"{'='*60}")

        task_components = {}
        for task in tasks:
            prompts = generate_prompts(task, n_samples)
            J_ln1_list = []
            J_attn_list = []
            J_ln2_list = []
            J_mlp_list = []
            J_full_list = []

            for prompt in prompts:
                tokens = model.to_tokens(prompt)
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

                h_in = cache[f'blocks.{layer}.hook_resid_pre'][-1].detach()
                h_mid = cache[f'blocks.{layer}.hook_resid_mid'][-1].detach()
                h_out = cache[f'blocks.{layer}.hook_resid_post'][-1].detach()

                # Component Jacobians
                ln1 = model.blocks[layer].ln1
                ln2 = model.blocks[layer].ln2
                J_ln1 = compute_ln_jacobian(h_in, ln1)
                J_ln2 = compute_ln_jacobian(h_mid, ln2)
                J_attn = compute_attn_jacobian(model, cache, layer)
                J_mlp, _ = compute_mlp_jacobian(model, cache, layer)

                # Full layer Jacobian (chain rule):
                # h_mid = h_in + Attn(LN1(h_in))
                # h_out = h_mid + MLP(LN2(h_mid))
                # J_full = (I + J_mlp @ J_ln2) @ (I + J_attn @ J_ln1)
                I = torch.eye(768, device=h_in.device)
                J_full = (I + J_mlp @ J_ln2) @ (I + J_attn @ J_ln1)

                J_ln1_list.append(J_ln1)
                J_attn_list.append(J_attn)
                J_ln2_list.append(J_ln2)
                J_mlp_list.append(J_mlp)
                J_full_list.append(J_full)

            task_components[task] = {
                'J_ln1': torch.stack(J_ln1_list),
                'J_attn': torch.stack(J_attn_list),
                'J_ln2': torch.stack(J_ln2_list),
                'J_mlp': torch.stack(J_mlp_list),
                'J_full': torch.stack(J_full_list),
            }

        # ---- Part 1: Component norms ----
        print(f"\n--- Component Operator Norms ---")
        for task in tasks:
            for comp in ['J_ln1', 'J_attn', 'J_ln2', 'J_mlp', 'J_full']:
                J_avg = task_components[task][comp].mean(0)
                S = torch.linalg.svdvals(J_avg)
                norm = S[0].item()
                rank95 = np.searchsorted(np.cumsum(S.numpy()) / S.sum(), 0.95)
                if comp in ['J_ln1', 'J_mlp', 'J_full']:
                    print(f"  {task:15s} {comp:8s}: ||J||={norm:.4f}, rank95={rank95}")

        # ---- Part 2: Task-specificity by component ----
        print(f"\n--- Task-Specificity by Component (cross-task cosine) ---")
        for comp in ['J_ln1', 'J_attn', 'J_ln2', 'J_mlp', 'J_full']:
            task_avgs = {t: task_components[t][comp].mean(0) for t in tasks}
            # Average cross-task cosine
            cross_cosines = []
            for i, t1 in enumerate(tasks):
                for j, t2 in enumerate(tasks):
                    if j > i:
                        cos = torch.nn.functional.cosine_similarity(
                            task_avgs[t1].flatten().unsqueeze(0),
                            task_avgs[t2].flatten().unsqueeze(0)
                        ).item()
                        cross_cosines.append(cos)
            avg_cross = np.mean(cross_cosines)
            min_cross = np.min(cross_cosines)

            # Also compute spectral cosine
            task_svs = {t: torch.linalg.svdvals(task_avgs[t]) for t in tasks}
            spec_cosines = []
            for i, t1 in enumerate(tasks):
                for j, t2 in enumerate(tasks):
                    if j > i:
                        cos = torch.nn.functional.cosine_similarity(
                            task_svs[t1].unsqueeze(0),
                            task_svs[t2].unsqueeze(0)
                        ).item()
                        spec_cosines.append(cos)

            print(f"  {comp:8s}: matrix_cos={avg_cross:.4f} (min={min_cross:.4f}), "
                  f"spec_cos={np.mean(spec_cosines):.4f}")

        # ---- Part 3: Full layer eigenspace overlap ----
        print(f"\n--- Full Layer Eigenspace Overlap (top-k) ---")
        task_full_avg = {t: task_components[t]['J_full'].mean(0) for t in tasks}

        for k in [5, 10, 20]:
            overlap_matrix = np.zeros((len(tasks), len(tasks)))
            for i, t1 in enumerate(tasks):
                U1, _, _ = torch.linalg.svd(task_full_avg[t1], full_matrices=False)
                for j, t2 in enumerate(tasks):
                    U2, _, _ = torch.linalg.svd(task_full_avg[t2], full_matrices=False)
                    P1 = U1[:, :k] @ U1[:, :k].T
                    proj = U2[:, :k].T @ P1 @ U2[:, :k]
                    overlap_matrix[i, j] = torch.trace(proj).item() / k

            if k in [5, 20]:
                print(f"  Top-{k}:")
                for i, t1 in enumerate(tasks):
                    row = f"    {t1:15s}" + "".join(f"{overlap_matrix[i,j]:>8.4f}" for j in range(len(tasks)))
                    print(row)

    print("\n" + "=" * 70)
    print("EXPERIMENT C COMPLETE")
    print("=" * 70)


# ============================================================
# Experiment D: Operator Basis Decomposition
# ============================================================
def exp_d_operator_basis():
    """
    Core question: Can per-task Jacobians be decomposed as
    J_task = Σ_i c_i(task) A_i?

    This tests the "conditional linear computation" hypothesis:
    F(h, x) = Σ_i g_i(x) A_i h

    Method:
    1. Collect average Jacobians for all (task, layer) combinations
    2. Flatten each to a vector
    3. SVD of the stacked matrix → basis operators
    4. Reconstruction test: how many basis operators needed?
    """
    print("=" * 70)
    print("EXPERIMENT D: Operator Basis Decomposition")
    print("=" * 70)

    model = get_model()
    tasks = ["addition", "antonym", "capital", "translate_fr"]
    layers = [3, 6, 9]
    n_samples = 50

    # ---- Step 1: Collect all Jacobians ----
    print("\n--- Collecting Jacobians ---")
    all_jacobians = {}  # (task, layer) -> average Jacobian

    for layer in layers:
        for task in tasks:
            prompts = generate_prompts(task, n_samples)
            Js = []
            for prompt in prompts[:30]:
                tokens = model.to_tokens(prompt)
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
                J, _ = compute_mlp_jacobian(model, cache, layer)
                Js.append(J)
            all_jacobians[(task, layer)] = torch.stack(Js).mean(0)
            print(f"  ({task}, L{layer}): done")

    # ---- Step 2: Operator basis decomposition per layer ----
    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer}: Operator Basis Decomposition")
        print(f"{'='*60}")

        # Stack all task Jacobians as flattened vectors
        task_names = []
        J_flat = []
        for task in tasks:
            J = all_jacobians[(task, layer)]
            J_flat.append(J.flatten())
            task_names.append(task)

        J_matrix = torch.stack(J_flat)  # [n_tasks, d^2]
        n_tasks = len(tasks)
        d_sq = J_matrix.shape[1]

        print(f"  Jacobian matrix shape: {J_matrix.shape}")

        # SVD to find basis operators
        U, S, Vt = torch.linalg.svd(J_matrix, full_matrices=False)

        print(f"\n  Basis operator singular values: {S.tolist()}")
        print(f"  Fraction explained by k basis operators:")
        cumsum = torch.cumsum(S**2, dim=0) / (S**2).sum()
        for k in [1, 2, 3, 4]:
            print(f"    k={k}: {cumsum[k-1].item():.6f}")

        # ---- Step 3: Reconstruction quality ----
        print(f"\n  Reconstruction quality (Frobenius norm ratio):")
        for k in [1, 2, 3, 4]:
            # Reconstruct using top-k basis operators
            U_k = U[:, :k]
            S_k = S[:k]
            Vt_k = Vt[:k, :]

            # Reconstructed Jacobians
            J_recon = U_k @ torch.diag(S_k) @ Vt_k

            # Per-task reconstruction quality
            for i, task in enumerate(task_names):
                J_orig = all_jacobians[(task, layer)]
                J_rec = J_recon[i].reshape(768, 768)
                quality = torch.norm(J_rec) / torch.norm(J_orig)
                if k in [1, 2, 4]:
                    print(f"    k={k}, {task:15s}: ||J_recon||/||J_orig|| = {quality:.4f}")

        # ---- Step 4: Basis operators analysis ----
        print(f"\n  Basis Operator Properties:")
        for k in range(min(4, len(S))):
            # Reshape basis operator from Vt
            A_k = Vt[k].reshape(768, 768)
            S_A = torch.linalg.svdvals(A_k)
            print(f"    A_{k+1}: op_norm={S_A[0]:.2f}, "
                  f"rank95={np.searchsorted(np.cumsum(S_A.numpy())/S_A.sum(), 0.95)}, "
                  f"sv={S[k].item():.4f}")

        # ---- Step 5: Task coefficients in basis ----
        print(f"\n  Task coefficients (g_i(task)) in basis:")
        # J_task = Σ_i c_i A_i where c_i = U[i] * S[i]
        coeffs = U * S.unsqueeze(0)  # [n_tasks, n_basis]
        for i, task in enumerate(task_names):
            print(f"    {task:15s}: " + ", ".join(f"g_{k+1}={coeffs[i,k]:.2f}" for k in range(min(4, len(S)))))

    # ---- Step 6: Cross-layer operator basis ----
    print(f"\n{'='*60}")
    print(f"Cross-Layer Operator Basis")
    print(f"{'='*60}")

    # Stack all Jacobians (task × layer) as flattened vectors
    all_J_flat = []
    all_labels = []
    for layer in layers:
        for task in tasks:
            J = all_jacobians[(task, layer)]
            all_J_flat.append(J.flatten())
            all_labels.append(f"L{layer}_{task}")

    J_all = torch.stack(all_J_flat)  # [12, d^2]
    U_all, S_all, Vt_all = torch.linalg.svd(J_all, full_matrices=False)

    print(f"\n  Cross-layer basis SVs: {S_all.tolist()}")
    cumsum = torch.cumsum(S_all**2, dim=0) / (S_all**2).sum()
    print(f"  Cumulative fraction explained:")
    for k in [1, 2, 3, 4, 6, 8, 12]:
        if k <= len(cumsum):
            print(f"    k={k}: {cumsum[k-1].item():.6f}")

    # Task coefficients
    print(f"\n  Task coefficients in cross-layer basis:")
    coeffs_all = U_all * S_all.unsqueeze(0)
    for i, label in enumerate(all_labels):
        top3 = coeffs_all[i, :3].tolist()
        print(f"    {label:20s}: g1={top3[0]:.2f}, g2={top3[1]:.2f}, g3={top3[2]:.2f}")

    # ---- Step 7: Full layer Jacobian basis ----
    print(f"\n{'='*60}")
    print(f"Full Layer Jacobian (Attn + MLP + LN) Basis")
    print(f"{'='*60}")

    for layer in [6]:  # Just layer 6 for full analysis
        print(f"\n  Layer {layer}:")
        task_full_jacobians = {}
        for task in tasks:
            prompts = generate_prompts(task, 20)
            J_fulls = []
            for prompt in prompts:
                tokens = model.to_tokens(prompt)
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

                h_in = cache[f'blocks.{layer}.hook_resid_pre'][-1].detach()
                h_mid = cache[f'blocks.{layer}.hook_resid_mid'][-1].detach()

                ln1 = model.blocks[layer].ln1
                ln2 = model.blocks[layer].ln2
                J_ln1 = compute_ln_jacobian(h_in, ln1)
                J_ln2 = compute_ln_jacobian(h_mid, ln2)
                J_attn = compute_attn_jacobian(model, cache, layer)
                J_mlp, _ = compute_mlp_jacobian(model, cache, layer)

                I = torch.eye(768, device=h_in.device)
                J_full = (I + J_mlp @ J_ln2) @ (I + J_attn @ J_ln1)
                J_fulls.append(J_full)

            task_full_jacobians[task] = torch.stack(J_fulls).mean(0)

        # Basis decomposition
        J_flat_full = torch.stack([task_full_jacobians[t].flatten() for t in tasks])
        U_f, S_f, Vt_f = torch.linalg.svd(J_flat_full, full_matrices=False)

        print(f"  Full layer basis SVs: {S_f.tolist()}")
        cumsum_f = torch.cumsum(S_f**2, dim=0) / (S_f**2).sum()
        for k in [1, 2, 3, 4]:
            print(f"    k={k}: {cumsum_f[k-1].item():.6f}")

        # Compare MLP-only vs Full-layer basis
        J_mlp_only = torch.stack([all_jacobians[(t, layer)].flatten() for t in tasks])
        U_m, S_m, Vt_m = torch.linalg.svd(J_mlp_only, full_matrices=False)

        print(f"\n  Comparison: MLP-only vs Full-layer basis")
        print(f"  MLP-only SVs: {S_m.tolist()}")
        print(f"  Full-layer SVs: {S_f.tolist()}")

    print("\n" + "=" * 70)
    print("EXPERIMENT D COMPLETE")
    print("=" * 70)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True,
                       choices=["a", "b", "c", "d", "all"])
    args = parser.parse_args()

    start = time.time()
    if args.exp in ["a", "all"]:
        exp_a_layernorm()
    if args.exp in ["b", "all"]:
        exp_b_attention_jacobian()
    if args.exp in ["c", "all"]:
        exp_c_full_decomposition()
    if args.exp in ["d", "all"]:
        exp_d_operator_basis()

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")
