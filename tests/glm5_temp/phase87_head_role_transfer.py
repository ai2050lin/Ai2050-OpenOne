"""
Phase 87+ Supplement: Head-Level Role Transfer Test
====================================================
Experiment D showed individual heads have filler-independent role signals
(consistency 0.6-0.87 at head level vs 0.43 at full representation level).

This test: Does a head-level role transfer operator generalize to unseen fillers?

If YES at head level but NOT at full representation level:
  → Binding IS happening at individual heads
  → But gets diluted when heads are combined
  → This would be a MAJOR finding about the binding mechanism
"""
import torch
import torch.nn.functional as F
import numpy as np

def main():
    print("=" * 70)
    print("Phase 87+ Supplement: Head-Level Role Transfer")
    print("=" * 70)

    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        center_unembed=False,
        center_writing_weights=False,
        fold_ln=False,
        device="cuda",
    )
    model.eval()
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    print(f"GPT-2-small: {n_heads} heads, d_head={d_head}")

    anchor = "Rome"
    train_fillers = ["Paris", "Berlin", "London", "Tokyo", "Madrid", "Oslo", "Seoul"]
    test_fillers = ["Alice", "Bob", "Carol", "Dave"]
    verbs = ["loves", "hates"]
    layers = [2, 4, 6, 8]

    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer}")
        print(f"{'='*60}")

        # Collect per-head z vectors for subject and object roles
        head_z_train = {h: {'subj': [], 'obj': []} for h in range(n_heads)}
        head_z_test = {h: {'subj': [], 'obj': []} for h in range(n_heads)}

        for filler in train_fillers + test_fillers:
            is_test = filler in test_fillers
            for verb in verbs:
                prompt_subj = f"{filler} {verb} {anchor}"
                prompt_obj = f"{anchor} {verb} {filler}"
                t_subj = model.to_tokens(prompt_subj)
                t_obj = model.to_tokens(prompt_obj)
                with torch.no_grad():
                    _, c_subj = model.run_with_cache(t_subj, remove_batch_dim=True)
                    _, c_obj = model.run_with_cache(t_obj, remove_batch_dim=True)

                for h in range(n_heads):
                    z_subj = c_subj[f'blocks.{layer}.attn.hook_z'][-1, h, :].detach().cpu()
                    z_obj = c_obj[f'blocks.{layer}.attn.hook_z'][-1, h, :].detach().cpu()

                    if is_test:
                        head_z_test[h]['subj'].append(z_subj)
                        head_z_test[h]['obj'].append(z_obj)
                    else:
                        head_z_train[h]['subj'].append(z_subj)
                        head_z_train[h]['obj'].append(z_obj)

        # For each head, test role transfer operator
        print(f"\n  Head-level role transfer analysis:")
        print(f"  {'Head':>4} | {'Train_cos':>9} | {'Test_cos':>8} | {'Gap':>6} | {'Consist':>7} | {'Verdict':>15}")
        print(f"  {'-'*4}-+-{'-'*9}-+-{'-'*8}-+-{'-'*6}-+-{'-'*7}-+-{'-'*15}")

        binding_heads = []
        
        for h in range(n_heads):
            # Average across verbs for each filler
            train_s = []
            train_o = []
            for i in range(0, len(head_z_train[h]['subj']), 2):
                train_s.append((head_z_train[h]['subj'][i] + head_z_train[h]['subj'][i+1]) / 2)
                train_o.append((head_z_train[h]['obj'][i] + head_z_train[h]['obj'][i+1]) / 2)

            test_s = []
            test_o = []
            for i in range(0, len(head_z_test[h]['subj']), 2):
                test_s.append((head_z_test[h]['subj'][i] + head_z_test[h]['subj'][i+1]) / 2)
                test_o.append((head_z_test[h]['obj'][i] + head_z_test[h]['obj'][i+1]) / 2)

            if len(train_s) < 3:
                continue

            train_s = torch.stack(train_s)  # [N, d_head]
            train_o = torch.stack(train_o)

            # Ridge regression: T(z_subj) ≈ z_obj
            H_s = train_s.numpy()
            H_o = train_o.numpy()
            lam = 0.1
            I_d = np.eye(H_s.shape[1])
            T = np.linalg.solve(H_s.T @ H_s + lam * I_d, H_s.T @ H_o)

            # Train eval
            pred_train = H_s @ T
            train_cos = [F.cosine_similarity(torch.tensor(pred_train[i]), torch.tensor(H_o[i]), dim=0).item()
                        for i in range(len(H_s))]
            train_mean = np.mean(train_cos)

            # Test eval
            if test_s:
                test_s_t = torch.stack(test_s).numpy()
                test_o_t = torch.stack(test_o).numpy()
                pred_test = test_s_t @ T
                test_cos = [F.cosine_similarity(torch.tensor(pred_test[i]), torch.tensor(test_o_t[i]), dim=0).item()
                           for i in range(len(test_s_t))]
                test_mean = np.mean(test_cos)
            else:
                test_mean = 0.0

            gap = train_mean - test_mean

            # Cross-filler consistency
            role_shifts = train_o - train_s
            cross_cos = []
            for i in range(len(role_shifts)):
                for j in range(i+1, len(role_shifts)):
                    cross_cos.append(F.cosine_similarity(role_shifts[i], role_shifts[j], dim=0).item())
            consistency = np.mean(cross_cos) if cross_cos else 0.0

            # Also: mean role shift direction method
            d_role = role_shifts.mean(0)
            d_role_norm = d_role / d_role.norm()
            # How well does this mean direction work for test fillers?
            if test_s:
                test_alphas = [((test_o[i] - test_s[i]) * d_role_norm).sum().item() 
                              for i in range(len(test_s))]
                test_shifted = [test_s[i] + test_alphas[i] * d_role_norm for i in range(len(test_s))]
                test_shift_cos = [F.cosine_similarity(test_shifted[i], test_o[i], dim=0).item()
                                 for i in range(len(test_s))]
                test_shift_mean = np.mean(test_shift_cos)
            else:
                test_shift_mean = 0.0

            if gap < 0.10:
                verdict = "BINDING!"
                binding_heads.append((h, gap, consistency, test_shift_mean))
            elif gap < 0.20:
                verdict = "Partial"
                binding_heads.append((h, gap, consistency, test_shift_mean))
            else:
                verdict = "Artifact"

            print(f"  {h:4d} | {train_mean:9.4f} | {test_mean:8.4f} | {gap:6.4f} | {consistency:7.4f} | {verdict:>15}")

        # Summary for this layer
        if binding_heads:
            print(f"\n  ★ Binding candidate heads at Layer {layer}:")
            for h, gap, cons, shift in sorted(binding_heads, key=lambda x: x[1]):
                print(f"    Head {h}: gap={gap:.4f}, consistency={cons:.4f}, shift_test={shift:.4f}")
        else:
            print(f"\n  No binding heads found at Layer {layer}.")

        # Also compute: what fraction of role signal is shared vs filler-specific?
        # Compare head-level vs full-representation level
        all_role_shifts_full = []
        for filler in train_fillers:
            for verb in verbs:
                prompt_subj = f"{filler} {verb} {anchor}"
                prompt_obj = f"{anchor} {verb} {filler}"
                t_subj = model.to_tokens(prompt_subj)
                t_obj = model.to_tokens(prompt_obj)
                with torch.no_grad():
                    _, c_subj = model.run_with_cache(t_subj, remove_batch_dim=True)
                    _, c_obj = model.run_with_cache(t_obj, remove_batch_dim=True)
                h_subj = c_subj[f'blocks.{layer}.hook_resid_post'][-1].detach().cpu()
                h_obj = c_obj[f'blocks.{layer}.hook_resid_post'][-1].detach().cpu()
                all_role_shifts_full.append(h_obj - h_subj)

        full_cross = []
        for i in range(len(all_role_shifts_full)):
            for j in range(i+1, len(all_role_shifts_full)):
                full_cross.append(F.cosine_similarity(
                    all_role_shifts_full[i], all_role_shifts_full[j], dim=0).item())
        
        head_best_consistency = max([cons for _, _, cons, _ in binding_heads]) if binding_heads else 0
        full_consistency = np.mean(full_cross) if full_cross else 0
        
        print(f"\n  Comparison: Best head consistency = {head_best_consistency:.4f} vs Full representation = {full_consistency:.4f}")
        if head_best_consistency > full_consistency + 0.2:
            print(f"  → Head-level binding signal is MUCH STRONGER than full representation!")
            print(f"  → Binding is DISTRIBUTED across heads, gets diluted when combined.")

    print("\n" + "=" * 70)
    print("Head-level role transfer analysis complete.")

if __name__ == "__main__":
    main()
