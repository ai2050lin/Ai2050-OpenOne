"""
Phase 82 Experiments C+D: Full Layer Jacobian + Subspace Intervention (Fast)
============================================================================

Combined and optimized version.
"""

import torch
import numpy as np
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


def compute_mlp_jacobian(model, cache, layer, pos=-1):
    """Compute exact MLP Jacobian."""
    pre_gelu = cache[f'blocks.{layer}.mlp.hook_pre'][pos].detach()
    x = pre_gelu.clone().requires_grad_(True)
    y = torch.nn.functional.gelu(x)
    gelu_deriv = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))[0].detach()

    W_in = model.blocks[layer].mlp.W_in.detach()
    W_out = model.blocks[layer].mlp.W_out.detach()

    scaled_W_in_T = gelu_deriv.unsqueeze(1) * W_in.T
    J = W_out.T @ scaled_W_in_T
    return J, gelu_deriv


def main():
    print("=" * 70)
    print("EXPERIMENT C+D: Full Layer Analysis + Subspace Intervention")
    print("=" * 70)

    model = get_model()
    tasks = ["addition", "antonym", "capital", "translate_fr"]
    layers = [3, 6, 9]
    n_samples = 100

    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer}")
        print(f"{'='*60}")

        # ---- Collect data ----
        task_data = {}
        for task in tasks:
            prompts = generate_prompts(task, n_samples)
            Js_mlp = []
            attn_outs = []
            h_ins = []
            h_mids = []
            h_outs = []
            delta_mlps = []
            gelu_derivs_list = []

            for prompt in prompts:
                tokens = model.to_tokens(prompt)
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

                h_in = cache[f'blocks.{layer}.hook_resid_pre'][-1].detach()
                h_mid = cache[f'blocks.{layer}.hook_resid_mid'][-1].detach()
                h_out = cache[f'blocks.{layer}.hook_resid_post'][-1].detach()
                attn_out = cache[f'blocks.{layer}.hook_attn_out'][-1].detach()
                delta_mlp = h_out - h_mid

                J, gd = compute_mlp_jacobian(model, cache, layer)

                Js_mlp.append(J)
                attn_outs.append(attn_out)
                h_ins.append(h_in)
                h_mids.append(h_mid)
                h_outs.append(h_out)
                delta_mlps.append(delta_mlp)
                gelu_derivs_list.append(gd)

            task_data[task] = {
                'Js_mlp': torch.stack(Js_mlp),
                'attn_outs': torch.stack(attn_outs),
                'h_ins': torch.stack(h_ins),
                'h_mids': torch.stack(h_mids),
                'h_outs': torch.stack(h_outs),
                'delta_mlps': torch.stack(delta_mlps),
                'gelu_derivs': torch.stack(gelu_derivs_list),
            }

        # ============================================================
        # Part C1: MLP Jacobian cross-task similarity
        # ============================================================
        print(f"\n--- C1: MLP Jacobian Cross-Task Similarity ---")
        task_J_avg = {}
        for task in tasks:
            J_avg = task_data[task]['Js_mlp'].mean(dim=0)
            task_J_avg[task] = J_avg

        # Flattened cosine
        print("  Flattened matrix cosine:")
        header = f"  {'':15s}" + "".join(f"{t[:8]:>9s}" for t in tasks)
        print(header)
        for i, t1 in enumerate(tasks):
            row = f"  {t1:15s}"
            for j, t2 in enumerate(tasks):
                cos = torch.nn.functional.cosine_similarity(
                    task_J_avg[t1].flatten().unsqueeze(0),
                    task_J_avg[t2].flatten().unsqueeze(0)
                ).item()
                row += f"{cos:>9.4f}"
            print(row)

        # ============================================================
        # Part C2: Attention output direction comparison
        # ============================================================
        print(f"\n--- C2: Attention Output Direction Comparison ---")
        task_attn_mean = {}
        for task in tasks:
            task_attn_mean[task] = task_data[task]['attn_outs'].mean(dim=0)

        header = f"  {'':15s}" + "".join(f"{t[:8]:>9s}" for t in tasks)
        print(header)
        for i, t1 in enumerate(tasks):
            row = f"  {t1:15s}"
            for j, t2 in enumerate(tasks):
                cos = torch.nn.functional.cosine_similarity(
                    task_attn_mean[t1].unsqueeze(0),
                    task_attn_mean[t2].unsqueeze(0)
                ).item()
                row += f"{cos:>9.4f}"
            print(row)

        # ============================================================
        # Part C3: Attention vs MLP magnitude
        # ============================================================
        print(f"\n--- C3: Attention vs MLP Output Magnitude ---")
        for task in tasks:
            ao = task_data[task]['attn_outs']
            mo = task_data[task]['delta_mlps']
            hi = task_data[task]['h_ins']
            ho = task_data[task]['h_outs']
            total_delta = ho - hi

            attn_frac = (ao.norm(dim=1) / (total_delta.norm(dim=1) + 1e-10)).mean()
            mlp_frac = (mo.norm(dim=1) / (total_delta.norm(dim=1) + 1e-10)).mean()

            print(f"  {task:15s}: attn_norm={ao.norm(dim=1).mean():.3f}, "
                  f"mlp_norm={mo.norm(dim=1).mean():.3f}, "
                  f"attn_frac={attn_frac:.3f}, mlp_frac={mlp_frac:.3f}")

        # ============================================================
        # Part C4: Attention pattern comparison
        # ============================================================
        print(f"\n--- C4: Attention Pattern Comparison ---")
        task_patterns = {}
        for task in tasks:
            prompts = generate_prompts(task, 30)
            patterns = []
            for prompt in prompts:
                tokens = model.to_tokens(prompt)
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
                pattern = cache[f'blocks.{layer}.attn.hook_pattern'].detach()  # [n_heads, seq_q, seq_k]
                # Last position attending to all
                patterns.append(pattern[:, -1, :])
            # Average across samples (need to handle different sequence lengths)
            # Just compute norm of average pattern
            avg_norm = torch.stack([p.norm() for p in patterns]).mean()
            task_patterns[task] = patterns

        # Compare: average pairwise L2 distance between attention patterns
        print("  Attention pattern L2 distance (last position, avg across heads):")
        for i, t1 in enumerate(tasks):
            for j, t2 in enumerate(tasks):
                if j <= i:
                    continue
                # Compare using first 10 samples with same sequence length
                dists = []
                for k in range(min(10, len(task_patterns[t1]), len(task_patterns[t2]))):
                    p1 = task_patterns[t1][k]
                    p2 = task_patterns[t2][k]
                    if p1.shape == p2.shape:
                        dists.append((p1 - p2).norm().item())
                if dists:
                    print(f"    {t1} vs {t2}: {np.mean(dists):.4f}")

        # ============================================================
        # Part D1: Universal vs Task-Specific Jacobian prediction
        # ============================================================
        print(f"\n--- D1: Universal vs Task-Specific Jacobian Prediction ---")
        J_universal = torch.stack([task_data[t]['Js_mlp'].mean(dim=0) for t in tasks]).mean(dim=0)
        U_univ, S_univ, Vt_univ = torch.linalg.svd(J_universal, full_matrices=False)

        for task in tasks:
            Js = task_data[task]['Js_mlp'][:50]
            h_normeds = []
            ln = model.blocks[layer].ln2
            for i in range(50):
                h_mid = task_data[task]['h_mids'][i]
                h_normed = ln(h_mid.unsqueeze(0)).squeeze(0).detach()
                h_normeds.append(h_normed)
            h_normeds = torch.stack(h_normeds)
            delta_mlps = task_data[task]['delta_mlps'][:50]

            # Per-sample R²
            r2_task_list = []
            r2_univ_list = []
            for i in range(len(Js)):
                pred_t = Js[i] @ h_normeds[i]
                pred_u = J_universal @ h_normeds[i]
                r2_t = 1 - (delta_mlps[i] - pred_t).norm() / delta_mlps[i].norm()
                r2_u = 1 - (delta_mlps[i] - pred_u).norm() / delta_mlps[i].norm()
                r2_task_list.append(r2_t.item())
                r2_univ_list.append(r2_u.item())

            print(f"  {task:15s}: R²(task_J)={np.mean(r2_task_list):.4f}±{np.std(r2_task_list):.4f}, "
                  f"R²(univ_J)={np.mean(r2_univ_list):.4f}±{np.std(r2_univ_list):.4f}, "
                  f"gap={np.mean(r2_task_list)-np.mean(r2_univ_list):.4f}")

        # ============================================================
        # Part D2: Eigenspace decomposition of ΔJ
        # ============================================================
        print(f"\n--- D2: Eigenspace Decomposition of ΔJ ---")
        for task in tasks:
            J_task = task_data[task]['Js_mlp'].mean(dim=0)
            delta_J = J_task - J_universal

            U_d, S_d, Vt_d = torch.linalg.svd(delta_J, full_matrices=False)

            total_power = S_d.sum()
            cumsum = torch.cumsum(S_d, dim=0) / total_power
            rank_90 = (cumsum < 0.9).sum().item() + 1
            rank_99 = (cumsum < 0.99).sum().item() + 1

            # Project ΔJ onto universal eigenspace
            for k in [10, 50, 100]:
                P = U_univ[:, :k] @ U_univ[:, :k].T
                delta_J_proj = P @ delta_J @ P.T
                frac_in_univ = torch.norm(delta_J_proj) / (torch.norm(delta_J) + 1e-10)
                if k in [10, 100]:
                    print(f"  {task:15s}: ΔJ rank90={rank_90}, rank99={rank_99}, "
                          f"top-5 ΔSV=[{', '.join(f'{s:.2f}' for s in S_d[:5].tolist())}], "
                          f"ΔJ in universal(k={k})={frac_in_univ:.4f}")

        # ============================================================
        # Part D3: Subspace-projected Jacobian prediction
        # ============================================================
        print(f"\n--- D3: Task-Specific Subspace Predicts delta_mlp ---")
        # Key test: project J_task through universal vs task-specific subspaces
        for k_sub in [5, 10, 20]:
            print(f"\n  Subspace dimension k={k_sub}:")
            for task in tasks:
                J_task = task_data[task]['Js_mlp'].mean(dim=0)
                U_task, S_task, Vt_task = torch.linalg.svd(J_task, full_matrices=False)

                # Test: use only top-k singular vectors of task Jacobian
                # to predict delta_mlp
                ln = model.blocks[layer].ln2
                r2_univ_sub_list = []
                r2_task_sub_list = []

                for i in range(50):
                    h_mid = task_data[task]['h_mids'][i]
                    h_normed = ln(h_mid.unsqueeze(0)).squeeze(0).detach()
                    delta_mlp = task_data[task]['delta_mlps'][i]

                    # Task-specific subspace projection
                    P_task = U_task[:, :k_sub]  # [768, k]
                    pred_task_sub = P_task @ (P_task.T @ (J_task @ h_normed))
                    r2_task_sub = 1 - (delta_mlp - pred_task_sub).norm() / delta_mlp.norm()

                    # Universal subspace projection
                    P_univ = U_univ[:, :k_sub]  # [768, k]
                    pred_univ_sub = P_univ @ (P_univ.T @ (J_task @ h_normed))
                    r2_univ_sub = 1 - (delta_mlp - pred_univ_sub).norm() / delta_mlp.norm()

                    r2_task_sub_list.append(r2_task_sub.item())
                    r2_univ_sub_list.append(r2_univ_sub.item())

                print(f"    {task:15s}: R²(task_sub)={np.mean(r2_task_sub_list):.4f}, "
                      f"R²(univ_sub)={np.mean(r2_univ_sub_list):.4f}, "
                      f"gap={np.mean(r2_task_sub_list)-np.mean(r2_univ_sub_list):.4f}")

    print("\n" + "=" * 70)
    print("EXPERIMENT C+D COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nTotal time: {time.time()-start:.1f}s")
