"""
Phase 86+ Minimal: Causal Intervention on Qwen3 (Simplified)
=============================================================
Focus on the CRITICAL test: entity swap intervention
"""

import torch
import numpy as np
import time

def main():
    print("=" * 70)
    print("Phase 86+ Minimal: Qwen3 Entity Swap Intervention")
    print("=" * 70)

    # Load model
    print("\nLoading Qwen3-1.5B...")
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        center_unembed=False,
        center_writing_weights=False,
        fold_ln=False,
        device="cuda",
    )
    model.eval()
    print("Model loaded.")

    # Helper
    def safe_token_id(text):
        try:
            return model.to_single_token(text)
        except AssertionError:
            tokens = model.to_tokens(text)
            return tokens[0, -1].item()

    # Step 1: Verify model predictions
    print("\n--- Step 1: Verify Predictions ---")
    prompts_to_check = [
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Japan is",
        "The capital of Italy is",
    ]
    for p in prompts_to_check:
        tokens = model.to_tokens(p)
        with torch.no_grad():
            logits = model(tokens)[0, -1]
            top_3 = torch.topk(logits, 3)
            top_strs = [model.to_string(t.item()).strip() for t in top_3.indices]
        print(f"  '{p}' -> {top_strs}")

    # Step 2: Compute entity direction
    print("\n--- Step 2: Compute Entity Direction ---")
    prompt_fr = "The capital of France is"
    prompt_de = "The capital of Germany is"
    tokens_fr = model.to_tokens(prompt_fr)
    tokens_de = model.to_tokens(prompt_de)

    # Test at layers 8, 12, 16
    test_layers = [8, 12, 16]
    entity_directions = {}
    for layer in test_layers:
        with torch.no_grad():
            _, cache_fr = model.run_with_cache(tokens_fr, remove_batch_dim=True)
            _, cache_de = model.run_with_cache(tokens_de, remove_batch_dim=True)
        h_fr = cache_fr[f'blocks.{layer}.hook_resid_post'][-1].detach()
        h_de = cache_de[f'blocks.{layer}.hook_resid_post'][-1].detach()
        d = h_fr - h_de
        entity_directions[layer] = (d / d.norm()).cuda()
        print(f"  Layer {layer}: direction norm = {d.norm().item():.2f}")

    # Free cache memory
    del cache_fr, cache_de
    torch.cuda.empty_cache()

    # Step 3: Entity Swap Intervention
    print("\n--- Step 3: Entity Swap Intervention ---")
    print("  d = h('France') - h('Germany')")
    print("  Add alpha*d to 'The capital of Germany is'")

    paris_id = safe_token_id("Paris")
    berlin_id = safe_token_id("Berlin")

    with torch.no_grad():
        baseline_logits = model(tokens_de)[0, -1]
    p_paris_base = torch.softmax(baseline_logits, dim=-1)[paris_id].item()
    p_berlin_base = torch.softmax(baseline_logits, dim=-1)[berlin_id].item()
    baseline_pred = model.to_string(baseline_logits.argmax().item()).strip()
    baseline_ratio = p_paris_base / (p_berlin_base + 1e-10)

    print(f"\n  Baseline: pred='{baseline_pred}', P(Paris)={p_paris_base:.6f}, P(Berlin)={p_berlin_base:.6f}")
    print(f"  Baseline Paris/Berlin ratio: {baseline_ratio:.6f}")
    print(f"\n  {'Layer':>6s}  {'alpha':>8s}  {'P(Paris)':>10s}  {'P(Berlin)':>10s}  {'Ratio':>10s}  {'Pred':>12s}  {'Ratio_vs_base':>14s}")

    for layer in test_layers:
        d_entity = entity_directions[layer]
        for alpha in [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]:
            def hook_fn(h, hook, direction=d_entity, scale=alpha):
                h[:, -1, :] = h[:, -1, :] + scale * direction
                return h

            with torch.no_grad():
                model.reset_hooks()
                model.add_hook(f'blocks.{layer}.hook_resid_post', hook_fn)
                int_logits = model(tokens_de)[0, -1]
                model.reset_hooks()

            p_paris = torch.softmax(int_logits, dim=-1)[paris_id].item()
            p_berlin = torch.softmax(int_logits, dim=-1)[berlin_id].item()
            ratio = p_paris / (p_berlin + 1e-10)
            new_pred = model.to_string(int_logits.argmax().item()).strip()
            ratio_vs_base = ratio / (baseline_ratio + 1e-10)

            print(f"  {layer:6d}  {alpha:8.1f}  {p_paris:10.6f}  {p_berlin:10.6f}  {ratio:10.4f}  {new_pred:>12s}  {ratio_vs_base:14.2f}x")

    # Step 4: Cross-entity generalization
    print("\n--- Step 4: Cross-Entity Generalization ---")
    print("  Apply d_France-Germany to 'The capital of Japan is'")

    prompt_jp = "The capital of Japan is"
    tokens_jp = model.to_tokens(prompt_jp)
    tokyo_id = safe_token_id("Tokyo")

    with torch.no_grad():
        jp_logits = model(tokens_jp)[0, -1]
    p_tokyo_base = torch.softmax(jp_logits, dim=-1)[tokyo_id].item()
    p_paris_jp_base = torch.softmax(jp_logits, dim=-1)[paris_id].item()
    jp_pred = model.to_string(jp_logits.argmax().item()).strip()

    print(f"  Baseline: pred='{jp_pred}', P(Tokyo)={p_tokyo_base:.6f}, P(Paris)={p_paris_jp_base:.6f}")

    for layer in test_layers:
        d = entity_directions[layer]
        for alpha in [0.0, 5.0, 10.0, 20.0]:
            def hook_fn(h, hook, direction=d, scale=alpha):
                h[:, -1, :] = h[:, -1, :] + scale * direction
                return h

            with torch.no_grad():
                model.reset_hooks()
                model.add_hook(f'blocks.{layer}.hook_resid_post', hook_fn)
                int_logits = model(tokens_jp)[0, -1]
                model.reset_hooks()

            p_tokyo = torch.softmax(int_logits, dim=-1)[tokyo_id].item()
            p_paris = torch.softmax(int_logits, dim=-1)[paris_id].item()
            new_pred = model.to_string(int_logits.argmax().item()).strip()

            print(f"  L{layer} a={alpha:.0f}: P(Tokyo)={p_tokyo:.6f}, P(Paris)={p_paris:.6f}, pred='{new_pred}'")

    # Step 5: Entity-Relation Subspace Separation (quick test)
    print("\n--- Step 5: Entity-Relation Subspace Separation ---")
    relations = ["capital", "currency", "language"]
    entities = ["France", "Germany", "Japan", "Italy", "Spain",
                "China", "India", "Brazil", "Egypt", "Australia"]
    templates = {
        "capital": "The capital of {} is",
        "currency": "The currency of {} is",
        "language": "The language of {} is",
    }

    for layer in [8, 12, 16]:
        grid = {}
        for relation in relations:
            for entity in entities:
                prompt = templates[relation].format(entity)
                tokens = model.to_tokens(prompt)
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
                h = cache[f'blocks.{layer}.hook_resid_post'][-1].detach().cpu()
                grid[(relation, entity)] = h

        # Entity variation
        e_vars = []
        for r in relations:
            for i, e1 in enumerate(entities):
                for j, e2 in enumerate(entities):
                    if j > i:
                        e_vars.append(grid[(r, e1)] - grid[(r, e2)])

        # Relation variation
        r_vars = []
        for e in entities:
            for i, r1 in enumerate(relations):
                for j, r2 in enumerate(relations):
                    if j > i:
                        r_vars.append(grid[(r1, e)] - grid[(r2, e)])

        e_var = torch.stack(e_vars)
        r_var = torch.stack(r_vars)
        e_var_c = e_var - e_var.mean(0)
        r_var_c = r_var - r_var.mean(0)

        _, Se, Vhe = torch.linalg.svd(e_var_c, full_matrices=False)
        _, Sr, Vhr = torch.linalg.svd(r_var_c, full_matrices=False)

        k = 5
        Ve = Vhe[:k, :].T
        Vr = Vhr[:k, :].T

        proj_e_on_r = e_var_c @ Vr @ Vr.T
        e_explained = (proj_e_on_r ** 2).sum() / (e_var_c ** 2).sum()

        proj_r_on_e = r_var_c @ Ve @ Ve.T
        r_explained = (proj_r_on_e ** 2).sum() / (r_var_c ** 2).sum()

        # Decode accuracy
        X = torch.stack([grid[(r, e)] for r in relations for e in entities])
        e_labels = [e for r in relations for e in entities]
        r_labels = [r for r in relations for e in entities]
        e_means = {e: X[[i for i, l in enumerate(e_labels) if l == e]].mean(0) for e in entities}
        r_means = {r: X[[i for i, l in enumerate(r_labels) if l == r]].mean(0) for r in relations}

        e_acc = sum(1 for i, h in enumerate(X) if min(e_means, key=lambda e: (h - e_means[e]).norm()) == e_labels[i]) / len(X)
        r_acc = sum(1 for i, h in enumerate(X) if min(r_means, key=lambda r: (h - r_means[r]).norm()) == r_labels[i]) / len(X)

        print(f"\n  Layer {layer}:")
        print(f"    Entity var by relation subspace: {e_explained:.4f}")
        print(f"    Relation var by entity subspace: {r_explained:.4f}")
        print(f"    Entity decode: {e_acc:.4f}, Relation decode: {r_acc:.4f}")
        if e_explained < 0.3 and r_explained < 0.3:
            print(f"    *** SEPARABLE! ***")

        del grid
        torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")
