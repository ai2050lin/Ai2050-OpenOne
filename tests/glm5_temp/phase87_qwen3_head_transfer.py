"""
Phase 87+ Final: Head-Level Role Transfer on Qwen3
===================================================
Verify that head-level binding also exists on Qwen3.
If YES → head-level binding is a general phenomenon, not GPT-2-specific.
"""
import torch
import torch.nn.functional as F
import numpy as np

def main():
    print("=" * 70)
    print("Phase 87+ Final: Head-Level Role Transfer on Qwen3")
    print("=" * 70)

    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        center_unembed=False,
        center_writing_weights=False,
        fold_ln=False,
        device="cuda",
    )
    model.eval()
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    print(f"Qwen3-1.5B: {n_heads} heads, d_head={d_head}")

    anchor = "Rome"
    train_fillers = ["Paris", "Berlin", "London", "Tokyo", "Madrid"]
    test_fillers = ["Alice", "Bob", "Carol"]
    verbs = ["loves", "hates"]
    layers = [8, 12, 16, 20]

    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Layer {layer}")
        print(f"{'='*60}")

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

        binding_count = 0
        partial_count = 0
        best_gap = 1.0
        best_head = -1
        best_consistency = 0.0

        for h in range(n_heads):
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

            if len(train_s) < 3 or len(test_s) < 1:
                continue

            train_s = torch.stack(train_s)
            train_o = torch.stack(train_o)

            H_s = train_s.numpy()
            H_o = train_o.numpy()
            lam = 0.1
            I_d = np.eye(H_s.shape[1])
            T = np.linalg.solve(H_s.T @ H_s + lam * I_d, H_s.T @ H_o)

            pred_train = H_s @ T
            train_cos = [F.cosine_similarity(torch.tensor(pred_train[i]), torch.tensor(H_o[i]), dim=0).item()
                        for i in range(len(H_s))]
            train_mean = np.mean(train_cos)

            test_s_t = torch.stack(test_s).numpy()
            test_o_t = torch.stack(test_o).numpy()
            pred_test = test_s_t @ T
            test_cos = [F.cosine_similarity(torch.tensor(pred_test[i]), torch.tensor(test_o_t[i]), dim=0).item()
                       for i in range(len(test_s_t))]
            test_mean = np.mean(test_cos)

            gap = train_mean - test_mean

            role_shifts = train_o - train_s
            cross_cos = []
            for i in range(len(role_shifts)):
                for j in range(i+1, len(role_shifts)):
                    cross_cos.append(F.cosine_similarity(role_shifts[i], role_shifts[j], dim=0).item())
            consistency = np.mean(cross_cos) if cross_cos else 0.0

            if gap < best_gap:
                best_gap = gap
                best_head = h
                best_consistency = consistency

            if gap < 0.10:
                binding_count += 1
                print(f"  Head {h:2d}: BINDING! gap={gap:.4f}, consist={consistency:.4f}, test_cos={test_mean:.4f}")
            elif gap < 0.20:
                partial_count += 1
                if gap < 0.15:
                    print(f"  Head {h:2d}: Partial   gap={gap:.4f}, consist={consistency:.4f}, test_cos={test_mean:.4f}")

        # Full representation level
        full_role_shifts = []
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
                full_role_shifts.append(h_obj - h_subj)

        full_cross = []
        for i in range(len(full_role_shifts)):
            for j in range(i+1, len(full_role_shifts)):
                full_cross.append(F.cosine_similarity(full_role_shifts[i], full_role_shifts[j], dim=0).item())
        full_consistency = np.mean(full_cross) if full_cross else 0

        print(f"\n  Summary Layer {layer}:")
        print(f"    Binding heads: {binding_count}/{n_heads}")
        print(f"    Partial heads: {partial_count}/{n_heads}")
        print(f"    Best head: H{best_head} (gap={best_gap:.4f}, consist={best_consistency:.4f})")
        print(f"    Best head consistency vs Full representation: {best_consistency:.4f} vs {full_consistency:.4f}")
        if best_consistency > full_consistency + 0.15:
            print(f"    -> Head-level binding signal STRONGER than full representation! (Same as GPT-2-small)")
        else:
            print(f"    -> Head and full representation similar (different from GPT-2-small)")

    print("\n" + "=" * 70)
    print("Qwen3 head-level role transfer analysis complete.")

if __name__ == "__main__":
    main()
