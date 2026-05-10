"""
Phase 86 Experiment C (Revised): Causal Subspace Intervention
===============================================================

REVISED APPROACH: GPT-2-small's factual knowledge is too weak for
reliable intervention tests. Instead, we use:

1. **Logit Lens Intervention**: At each layer, project residual stream
   through unembedding to see what the model "thinks". Then intervene
   on the residual stream and check how the logit lens changes.

2. **Model-Confirmed Prompts**: Use prompts where the model DOES produce
   reliable, strong predictions. Then test if subspace intervention
   changes those predictions.

3. **Activation Patching**: Instead of adding raw direction vectors,
   patch specific subspace components from one prompt into another.

Key Principle:
  The goal is NOT to prove the model knows facts.
  The goal IS to prove that subspace intervention CAUSALLY changes computation.
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


def exp_c_revised():
    print("=" * 70)
    print("EXPERIMENT C (REVISED): Causal Subspace Intervention")
    print("=" * 70)

    model = get_model()

    # ---- Part 1: Find model-confirmed prompts ----
    print("\n--- Part 1: Finding Model-Confirmed Prompts ---")
    print("  Test which prompts the model reliably predicts correctly")

    # Test a range of prompts and check model's top predictions
    test_prompts = {
        # Syntactic patterns (high confidence expected)
        "The cat sat on the": "mat",
        "Once upon a": "time",
        "It was a dark and stormy": "night",
        "In the beginning": "was",
        "To be or not to": "be",

        # Relational patterns with entity-relation structure
        "The capital of France is Paris. The capital of Germany is": "Berlin",
        "The opposite of hot is": "cold",
        "The opposite of big is": "small",
        "The plural of cat is": "cats",
        "The past tense of walk is": "walked",

        # Simple entity substitution (where model should know)
        "The color of grass is": "green",
        "The color of the sky is": "blue",
        "The sound a dog makes is": "bark",
        "Water freezes at": "zero",
    }

    confirmed_prompts = {}

    for prompt, expected in test_prompts.items():
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            logits = model(tokens)[0, -1]
            top_5 = torch.topk(logits, 5)
            top_tokens = [model.to_string(t.item()).strip() for t in top_5.indices]
            top_probs = torch.softmax(top_5.values, dim=-1).tolist()

        # Check if expected token is in top-5
        expected_found = any(expected.lower() in t.lower() for t in top_tokens)
        top_pred = top_tokens[0]
        top_prob = top_probs[0]

        status = "CONFIRMED" if expected_found else f"top={top_pred}"
        print(f"  '{prompt}' -> top: {top_pred} ({top_prob:.3f}) [{status}]")

        if expected_found or top_prob > 0.1:
            confirmed_prompts[prompt] = {
                'top_pred': top_pred,
                'top_prob': top_prob,
                'top_tokens': top_tokens,
            }

    print(f"\n  Confirmed {len(confirmed_prompts)}/{len(test_prompts)} prompts")

    # ---- Part 2: Entity-Relation Subspace Intervention on Confirmed Prompts ----
    print("\n--- Part 2: Subspace Intervention on Confirmed Prompts ---")

    # Use "opposite of X" pairs - model likely knows these
    entity_pairs = [
        ("hot", "cold"),
        ("big", "small"),
        ("fast", "slow"),
        ("happy", "sad"),
        ("strong", "weak"),
        ("light", "dark"),
        ("good", "bad"),
        ("old", "young"),
        ("rich", "poor"),
        ("tall", "short"),
    ]

    # Step 1: Find the entity subspace direction
    # Compare "The opposite of hot is" vs "The opposite of big is"
    # The difference should be the entity encoding

    prompt_template = "The opposite of {} is"
    layers_to_test = [2, 4, 6, 8, 10]

    # Collect representations for each entity
    entity_reps = {}  # (layer, entity) -> h
    for entity, _ in entity_pairs:
        prompt = prompt_template.format(entity)
        tokens = model.to_tokens(prompt)
        for layer in layers_to_test:
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            h = cache[f'blocks.{layer}.hook_resid_post'][-1].detach()
            entity_reps[(layer, entity)] = h

    # Compute entity direction: d_hot_big = h("hot") - h("big")
    print("\n  Entity direction test: d = h('hot') - h('big')")
    print("  Intervene on 'The opposite of big is' -> should shift toward 'cold'")

    # First verify what the model predicts
    for entity, antonym in entity_pairs[:5]:
        prompt = prompt_template.format(entity)
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            logits = model(tokens)[0, -1]
            top_3 = torch.topk(logits, 3)
            top_strs = [model.to_string(t.item()).strip() for t in top_3.indices]
        print(f"  '{prompt}' -> {top_strs}")

    # Step 2: Intervention - add entity direction to shift prediction
    # Use hot-big direction on "The opposite of big is"
    # If entity subspace is causal, should shift prediction toward hot's antonym (cold)

    target_prompt = "The opposite of big is"
    target_tokens = model.to_tokens(target_prompt)

    # Get baseline prediction
    with torch.no_grad():
        baseline_logits = model(target_tokens)[0, -1]

    cold_id = model.to_single_token(" cold")
    small_id = model.to_single_token(" small")

    print(f"\n  Baseline 'The opposite of big is':")
    p_cold_base = torch.softmax(baseline_logits, dim=-1)[cold_id].item()
    p_small_base = torch.softmax(baseline_logits, dim=-1)[small_id].item()
    print(f"    P(cold) = {p_cold_base:.6f}, P(small) = {p_small_base:.6f}")

    # Test "The opposite of hot is"
    hot_prompt = "The opposite of hot is"
    hot_tokens = model.to_tokens(hot_prompt)
    with torch.no_grad():
        hot_logits = model(hot_tokens)[0, -1]
    p_cold_hot = torch.softmax(hot_logits, dim=-1)[cold_id].item()
    print(f"  Reference 'The opposite of hot is': P(cold) = {p_cold_hot:.6f}")

    # Compute entity direction at each layer
    print(f"\n  Entity Swap Intervention: d_entity = h('hot') - h('big')")
    print(f"  Add d_entity to 'The opposite of big is'")
    print(f"  {'Layer':>6s}  {'alpha':>8s}  {'P(cold)':>10s}  {'P(small)':>10s}  {'cold/small':>12s}  {'Effect':>20s}")

    for layer in layers_to_test:
        d_entity = entity_reps[(layer, "hot")] - entity_reps[(layer, "big")]
        d_entity_norm = d_entity / d_entity.norm()

        for alpha in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
            def hook_intervene(h, hook, direction=d_entity_norm, scale=alpha):
                h[:, -1, :] = h[:, -1, :] + scale * direction.to(h.device)
                return h

            with torch.no_grad():
                model.reset_hooks()
                model.add_hook(f'blocks.{layer}.hook_resid_post', hook_intervene)
                intervened_logits = model(target_tokens)[0, -1]
                model.reset_hooks()

            p_cold = torch.softmax(intervened_logits, dim=-1)[cold_id].item()
            p_small = torch.softmax(intervened_logits, dim=-1)[small_id].item()
            ratio = p_cold / (p_small + 1e-10)

            if alpha == 0.0:
                effect = "baseline"
            elif ratio > p_cold_base / (p_small_base + 1e-10) * 2:
                effect = "STRONG SHIFT"
            elif ratio > p_cold_base / (p_small_base + 1e-10) * 1.2:
                effect = "shift"
            else:
                effect = "no effect"

            print(f"  {layer:6d}  {alpha:8.1f}  {p_cold:10.6f}  {p_small:10.6f}  {ratio:12.4f}  {effect:>20s}")

    # ---- Part 3: Logit Lens Intervention ----
    print("\n--- Part 3: Logit Lens Intervention ---")
    print("  At each layer, check what the model 'thinks' before and after intervention")

    # Use the best-performing layer from Part 2
    # Check logit lens at each layer with and without intervention
    d_entity = entity_reps[(6, "hot")] - entity_reps[(6, "big")]
    d_entity_norm = d_entity / d_entity.norm()
    alpha_test = 10.0

    print(f"\n  Intervening at Layer 6 with alpha={alpha_test}")
    print(f"  Logit lens at each layer (without/with intervention):")

    for check_layer in [4, 6, 8, 10]:
        # Without intervention
        with torch.no_grad():
            _, cache = model.run_with_cache(target_tokens, remove_batch_dim=True)
        h_clean = cache[f'blocks.{check_layer}.hook_resid_post'][-1].detach()

        # Apply logit lens (project through unembedding)
        W_U = model.W_U.detach()  # [d_model, d_vocab]
        logits_clean = h_clean @ W_U  # [d_vocab]
        top_clean = torch.topk(logits_clean, 5)
        tokens_clean = [model.to_string(t.item()).strip() for t in top_clean.indices]

        # With intervention at layer 6
        def hook_intervene(h, hook, direction=d_entity_norm, scale=alpha_test):
            h[:, -1, :] = h[:, -1, :] + scale * direction.to(h.device)
            return h

        with torch.no_grad():
            model.reset_hooks()
            model.add_hook(f'blocks.6.hook_resid_post', hook_intervene)
            _, cache_int = model.run_with_cache(target_tokens, remove_batch_dim=True)
            model.reset_hooks()

        h_int = cache_int[f'blocks.{check_layer}.hook_resid_post'][-1].detach()
        logits_int = h_int @ W_U
        top_int = torch.topk(logits_int, 5)
        tokens_int = [model.to_string(t.item()).strip() for t in top_int.indices]

        # Check cold and small probabilities
        p_cold_clean = torch.softmax(logits_clean, dim=-1)[cold_id].item()
        p_cold_int = torch.softmax(logits_int, dim=-1)[cold_id].item()
        p_small_clean = torch.softmax(logits_clean, dim=-1)[small_id].item()
        p_small_int = torch.softmax(logits_int, dim=-1)[small_id].item()

        print(f"\n  Check Layer {check_layer}:")
        print(f"    Clean top-5: {tokens_clean}")
        print(f"    Intervened top-5: {tokens_int}")
        print(f"    P(cold): {p_cold_clean:.6f} -> {p_cold_int:.6f} (delta={p_cold_int - p_cold_clean:+.6f})")
        print(f"    P(small): {p_small_clean:.6f} -> {p_small_int:.6f} (delta={p_small_int - p_small_clean:+.6f})")

    # ---- Part 4: Activation Patching (Resampling Ablation) ----
    print("\n--- Part 4: Activation Patching ---")
    print("  Patch: take entity subspace from 'hot' prompt, inject into 'big' prompt")
    print("  Method: compute mean direction per entity, project onto entity subspace, swap")

    # Collect multiple samples per entity
    n_samples = 20
    entity_avg_reps = {}

    for entity, _ in entity_pairs[:5]:
        reps_per_layer = {l: [] for l in layers_to_test}
        # Use different sentence frames for the same entity
        frames = [
            f"The opposite of {entity} is",
            f"The word opposite to {entity} is",
            f"{entity} is the opposite of",
        ]
        for frame in frames:
            tokens = model.to_tokens(frame)
            for layer in layers_to_test:
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
                h = cache[f'blocks.{layer}.hook_resid_post'][-1].detach()
                reps_per_layer[layer].append(h)

        entity_avg_reps[entity] = {l: torch.stack(reps).mean(0) for l, reps in reps_per_layer.items()}

    # Compute entity subspace directions
    # Entity subspace = PCA of representations across entities
    for layer in [6]:
        # Stack all entity representations
        entity_list = list(entity_avg_reps.keys())
        H = torch.stack([entity_avg_reps[e][layer] for e in entity_list])  # [n_entities, d_model]
        H_centered = H - H.mean(0)

        # SVD to get entity subspace
        _, S, Vh = torch.linalg.svd(H_centered, full_matrices=False)
        k = min(3, S.shape[0])
        entity_subspace = Vh[:k, :].T  # [d_model, k]

        print(f"\n  Layer {layer}: Entity subspace dimensionality = {k}")
        print(f"    Top singular values: {[round(s,2) for s in S[:5].tolist()]}")

        # For each entity pair, compute the projection difference
        print(f"\n  Entity subspace projection test:")
        for entity, antonym in entity_pairs[:5]:
            h_entity = entity_avg_reps[entity][layer]

            # Project onto entity subspace
            proj = h_entity @ entity_subspace  # [k]
            print(f"    '{entity}' projection: {proj.tolist()[:3]}")

    # ---- Part 5: Cross-Entity Generalization Test ----
    print("\n--- Part 5: Cross-Entity Generalization ---")
    print("  Compute entity direction from one pair, apply to another")
    print("  If universal -> direction should generalize")

    # Use "hot" - "big" direction, apply to "fast" prompt
    # "fast" -> "slow", "hot" -> "cold"
    # If d_hot_big shifts "fast" toward "hot"'s antonym space,
    # that means entity direction is universal

    fast_prompt = "The opposite of fast is"
    fast_tokens = model.to_tokens(fast_prompt)

    slow_id = model.to_single_token(" slow")
    cold_id = model.to_single_token(" cold")

    with torch.no_grad():
        fast_logits = model(fast_tokens)[0, -1]
    p_slow_base = torch.softmax(fast_logits, dim=-1)[slow_id].item()
    p_cold_base_fast = torch.softmax(fast_logits, dim=-1)[cold_id].item()

    print(f"\n  Baseline 'The opposite of fast is': P(slow)={p_slow_base:.6f}, P(cold)={p_cold_base_fast:.6f}")

    # d_hot_big direction applied to "fast" prompt
    print(f"\n  Applying d_hot_big to 'The opposite of fast is':")
    print(f"  {'Layer':>6s}  {'alpha':>8s}  {'P(slow)':>10s}  {'P(cold)':>10s}  {'cold/slow':>12s}  {'Effect':>20s}")

    for layer in layers_to_test:
        d = entity_reps[(layer, "hot")] - entity_reps[(layer, "big")]
        d_norm = d / d.norm()

        for alpha in [0.0, 5.0, 10.0, 20.0]:
            def hook_intervene(h, hook, direction=d_norm, scale=alpha):
                h[:, -1, :] = h[:, -1, :] + scale * direction.to(h.device)
                return h

            with torch.no_grad():
                model.reset_hooks()
                model.add_hook(f'blocks.{layer}.hook_resid_post', hook_intervene)
                int_logits = model(fast_tokens)[0, -1]
                model.reset_hooks()

            p_slow = torch.softmax(int_logits, dim=-1)[slow_id].item()
            p_cold = torch.softmax(int_logits, dim=-1)[cold_id].item()
            ratio = p_cold / (p_slow + 1e-10)

            if alpha == 0.0:
                effect = "baseline"
            elif p_cold > p_cold_base_fast * 2:
                effect = "shifted to cold"
            elif p_cold > p_cold_base_fast * 1.2:
                effect = "slight shift"
            else:
                effect = "no effect"

            print(f"  {layer:6d}  {alpha:8.1f}  {p_slow:10.6f}  {p_cold:10.6f}  {ratio:12.4f}  {effect:>20s}")

    # ---- Part 6: Subspace-Specific Intervention ----
    print("\n--- Part 6: Subspace-Specific Intervention ---")
    print("  Instead of adding raw direction, project onto ENTITY subspace first")
    print("  This ensures we only modify entity-relevant dimensions")

    # Compute entity subspace from multiple entities
    layer = 6
    entity_list_sub = ["hot", "big", "fast", "happy", "strong", "light"]
    H_sub = torch.stack([entity_reps[(layer, e)] for e in entity_list_sub])
    H_sub_centered = H_sub - H_sub.mean(0)

    _, S_sub, Vh_sub = torch.linalg.svd(H_sub_centered, full_matrices=False)
    k_sub = min(5, S_sub.shape[0])
    entity_basis = Vh_sub[:k_sub, :].T  # [d_model, k]

    print(f"\n  Entity subspace (k={k_sub}) from {len(entity_list_sub)} entities")
    print(f"  Top singular values: {[round(s,2) for s in S_sub[:5].tolist()]}")

    # Now: project d_hot_big onto entity subspace, then intervene
    d_raw = entity_reps[(layer, "hot")] - entity_reps[(layer, "big")]
    # Project onto entity subspace
    d_projected = entity_basis @ (entity_basis.T @ d_raw)  # project and reconstruct

    # How much of d_raw is in the entity subspace?
    projection_ratio = d_projected.norm() / d_raw.norm()
    print(f"  Projection ratio: {projection_ratio:.4f} ({projection_ratio**2*100:.1f}% of variance in entity subspace)")

    # Test projected vs unprojected intervention
    target_prompt = "The opposite of big is"
    target_tokens = model.to_tokens(target_prompt)

    with torch.no_grad():
        baseline_logits = model(target_tokens)[0, -1]
    p_cold_base = torch.softmax(baseline_logits, dim=-1)[cold_id].item()
    p_small_base = torch.softmax(baseline_logits, dim=-1)[small_id].item()

    print(f"\n  Baseline: P(cold)={p_cold_base:.6f}, P(small)={p_small_base:.6f}")

    for use_projected in [False, True]:
        d = d_projected if use_projected else d_raw
        d_norm = d / d.norm()
        label = "projected" if use_projected else "raw"

        print(f"\n  Using {label} direction:")
        for alpha in [5.0, 10.0, 20.0]:
            def hook_intervene(h, hook, direction=d_norm, scale=alpha):
                h[:, -1, :] = h[:, -1, :] + scale * direction.to(h.device)
                return h

            with torch.no_grad():
                model.reset_hooks()
                model.add_hook(f'blocks.{layer}.hook_resid_post', hook_intervene)
                int_logits = model(target_tokens)[0, -1]
                model.reset_hooks()

            p_cold = torch.softmax(int_logits, dim=-1)[cold_id].item()
            p_small = torch.softmax(int_logits, dim=-1)[small_id].item()
            ratio = p_cold / (p_small + 1e-10)

            print(f"    alpha={alpha:.1f}: P(cold)={p_cold:.6f}, P(small)={p_small:.6f}, ratio={ratio:.4f}")

    print("\n" + "=" * 70)
    print("EXPERIMENT C (REVISED) COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    start = time.time()
    exp_c_revised()
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")
