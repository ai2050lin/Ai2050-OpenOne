"""
Phase 86+ : Causal Subspace Intervention on Qwen3
===================================================

GPT-2-small's factual knowledge is too weak for clean causal intervention.
This script tests the same intervention on Qwen3, which has much stronger
factual knowledge.

Key test:
  d_entity = h("France") - h("Germany")
  Add alpha * d_entity to "The capital of Germany is"
  If output shifts from "Berlin" toward "Paris" -> Entity subspace is CAUSAL
"""

import torch
import numpy as np
import time
from transformer_lens import HookedTransformer


def main():
    print("=" * 70)
    print("Phase 86+: Causal Intervention on Qwen3")
    print("=" * 70)

    # Load Qwen3
    print("\nLoading Qwen3...")
    model = HookedTransformer.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        center_unembed=False,
        center_writing_weights=False,
        fold_ln=False,
        device="cuda",
    )
    model.eval()
    print(f"Model loaded: {model.cfg.model_name}")

    # ---- Part 1: Verify model's factual knowledge ----
    print("\n--- Part 1: Verify Qwen3 Factual Knowledge ---")

    test_prompts = [
        ("The capital of France is", "Paris"),
        ("The capital of Germany is", "Berlin"),
        ("The capital of Japan is", "Tokyo"),
        ("The capital of Italy is", "Rome"),
        ("The capital of Spain is", "Madrid"),
        ("The currency of France is", "euro"),
        ("The language of France is", "French"),
        ("The opposite of hot is", "cold"),
        ("The opposite of big is", "small"),
    ]

    for prompt, expected in test_prompts:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            logits = model(tokens)[0, -1]
            top_5 = torch.topk(logits, 5)
            top_tokens = [model.to_string(t.item()).strip() for t in top_5.indices]
            top_probs = torch.softmax(top_5.values, dim=-1).tolist()

        found = any(expected.lower() in t.lower() for t in top_tokens)
        status = "OK" if found else "MISS"
        print(f"  [{status}] '{prompt}' -> {top_tokens[:3]} (prob: {[f'{p:.3f}' for p in top_probs[:3]]})")

    # ---- Part 2: Entity-Relation Binding Decomposition ----
    print("\n--- Part 2: Entity-Relation Subspace Separation on Qwen3 ---")

    relations = ["capital", "currency", "language"]
    entities = ["France", "Germany", "Japan", "Italy", "Spain",
                "China", "India", "Brazil", "Egypt", "Australia"]

    templates = {
        "capital": "The capital of {} is",
        "currency": "The currency of {} is",
        "language": "The language of {} is",
    }

    layers_to_check = [4, 8, 12, 16, 20]  # Qwen3 has more layers

    # Build representation grid
    grid_reps = {}  # (layer, relation, entity) -> h

    for layer in layers_to_check:
        for relation in relations:
            for entity in entities:
                prompt = templates[relation].format(entity)
                tokens = model.to_tokens(prompt)
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
                h = cache[f'blocks.{layer}.hook_resid_post'][-1].detach().cpu()
                grid_reps[(layer, relation, entity)] = h

    print(f"  Collected {len(grid_reps)} representation vectors")

    # Compute subspace separation
    for layer in layers_to_check:
        # Entity variation: fix relation, vary entity
        entity_var_vecs = []
        for relation in relations:
            for i, e1 in enumerate(entities):
                for j, e2 in enumerate(entities):
                    if j > i:
                        h1 = grid_reps[(layer, relation, e1)]
                        h2 = grid_reps[(layer, relation, e2)]
                        entity_var_vecs.append(h1 - h2)

        # Relation variation: fix entity, vary relation
        relation_var_vecs = []
        for entity in entities:
            for i, r1 in enumerate(relations):
                for j, r2 in enumerate(relations):
                    if j > i:
                        h1 = grid_reps[(layer, r1, entity)]
                        h2 = grid_reps[(layer, r2, entity)]
                        relation_var_vecs.append(h1 - h2)

        entity_var = torch.stack(entity_var_vecs)
        relation_var = torch.stack(relation_var_vecs)

        entity_var_c = entity_var - entity_var.mean(0)
        relation_var_c = relation_var - relation_var.mean(0)

        _, S_e, Vh_e = torch.linalg.svd(entity_var_c, full_matrices=False)
        _, S_r, Vh_r = torch.linalg.svd(relation_var_c, full_matrices=False)

        pr_entity = (S_e.sum()**2) / (S_e**2).sum()
        pr_relation = (S_r.sum()**2) / (S_r**2).sum()

        k = min(5, S_e.shape[0], S_r.shape[0])
        V_e = Vh_e[:k, :].T  # [d_model, k]
        V_r = Vh_r[:k, :].T

        # Entity var explained by relation subspace
        proj_e = entity_var_c @ V_r @ V_r.T
        entity_explained = (proj_e ** 2).sum() / (entity_var_c ** 2).sum()

        # Relation var explained by entity subspace
        proj_r = relation_var_c @ V_e @ V_e.T
        relation_explained = (proj_r ** 2).sum() / (relation_var_c ** 2).sum()

        # Decoding accuracy
        X = []
        entity_labels = []
        relation_labels = []
        for relation in relations:
            for entity in entities:
                X.append(grid_reps[(layer, relation, entity)])
                entity_labels.append(entity)
                relation_labels.append(relation)
        X = torch.stack(X)

        entity_means = {e: X[[i for i, l in enumerate(entity_labels) if l == e]].mean(0) for e in entities}
        relation_means = {r: X[[i for i, l in enumerate(relation_labels) if l == r]].mean(0) for r in relations}

        entity_correct = sum(1 for i, h in enumerate(X)
                           if min(entity_means, key=lambda e: (h - entity_means[e]).norm()) == entity_labels[i])
        relation_correct = sum(1 for i, h in enumerate(X)
                             if min(relation_means, key=lambda r: (h - relation_means[r]).norm()) == relation_labels[i])

        entity_acc = entity_correct / len(X)
        relation_acc = relation_correct / len(X)

        print(f"\n  Layer {layer}:")
        print(f"    Entity PR={pr_entity:.1f}, Relation PR={pr_relation:.1f}")
        print(f"    Entity var explained by relation: {entity_explained:.4f}")
        print(f"    Relation var explained by entity: {relation_explained:.4f}")
        print(f"    Entity decode acc: {entity_acc:.4f}, Relation decode acc: {relation_acc:.4f}")

        if entity_explained < 0.3 and relation_explained < 0.3:
            print(f"    *** ENTITY AND RELATION SUBSPACES ARE SEPARABLE! ***")

    # ---- Part 3: Causal Subspace Intervention ----
    print("\n--- Part 3: Causal Entity Swap Intervention ---")
    print("  d_entity = h('France') - h('Germany')")
    print("  Add alpha * d_entity to 'The capital of Germany is'")

    # First verify predictions
    for entity in ["France", "Germany", "Japan"]:
        prompt = f"The capital of {entity} is"
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            logits = model(tokens)[0, -1]
            top_3 = torch.topk(logits, 3)
            top_strs = [model.to_string(t.item()).strip() for t in top_3.indices]
        print(f"  '{prompt}' -> {top_strs}")

    # Helper: safely get token ID
    def safe_token_id(text):
        try:
            return model.to_single_token(text)
        except AssertionError:
            tokens = model.to_tokens(text)
            return tokens[0, -1].item()

    # Compute entity direction at each layer
    prompt_fr = "The capital of France is"
    prompt_de = "The capital of Germany is"
    tokens_fr = model.to_tokens(prompt_fr)
    tokens_de = model.to_tokens(prompt_de)

    entity_directions = {}
    for layer in [8, 12, 16, 20]:
        with torch.no_grad():
            _, cache_fr = model.run_with_cache(tokens_fr, remove_batch_dim=True)
            _, cache_de = model.run_with_cache(tokens_de, remove_batch_dim=True)
        h_fr = cache_fr[f'blocks.{layer}.hook_resid_post'][-1].detach()
        h_de = cache_de[f'blocks.{layer}.hook_resid_post'][-1].detach()
        d = h_fr - h_de
        entity_directions[layer] = (d / d.norm()).cuda()

    # Get token IDs
    paris_id = safe_token_id("Paris")
    berlin_id = safe_token_id("Berlin")

    # Baseline
    with torch.no_grad():
        baseline_logits = model(tokens_de)[0, -1]
    p_paris_base = torch.softmax(baseline_logits, dim=-1)[paris_id].item()
    p_berlin_base = torch.softmax(baseline_logits, dim=-1)[berlin_id].item()
    baseline_pred = model.to_string(baseline_logits.argmax().item()).strip()

    print(f"\n  Baseline 'The capital of Germany is': pred='{baseline_pred}'")
    print(f"  P(Paris)={p_paris_base:.6f}, P(Berlin)={p_berlin_base:.6f}")

    # Intervention
    print(f"\n  Entity Swap Intervention Results:")
    print(f"  {'Layer':>6s}  {'alpha':>8s}  {'P(Paris)':>10s}  {'P(Berlin)':>10s}  {'Paris/Berlin':>14s}  {'Pred':>15s}  {'Effect':>20s}")

    for layer in [8, 12, 16, 20]:
        d_entity = entity_directions[layer]

        for alpha in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
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

            if alpha == 0.0:
                effect = "baseline"
            elif "Paris" in new_pred:
                effect = "*** SWAPPED! ***"
            elif ratio > p_paris_base / (p_berlin_base + 1e-10) * 5:
                effect = "STRONG SHIFT"
            elif ratio > p_paris_base / (p_berlin_base + 1e-10) * 1.5:
                effect = "shift"
            else:
                effect = "no effect"

            print(f"  {layer:6d}  {alpha:8.1f}  {p_paris:10.6f}  {p_berlin:10.6f}  {ratio:14.4f}  {new_pred:>15s}  {effect:>20s}")

    # ---- Part 4: Cross-Entity Generalization ----
    print("\n--- Part 4: Cross-Entity Generalization ---")
    print("  Apply d_France-Germany to 'The capital of Japan is'")

    prompt_jp = "The capital of Japan is"
    tokens_jp = model.to_tokens(prompt_jp)

    tokyo_id = safe_token_id("Tokyo")

    with torch.no_grad():
        jp_logits = model(tokens_jp)[0, -1]
    p_tokyo_base = torch.softmax(jp_logits, dim=-1)[tokyo_id].item()
    p_paris_jp = torch.softmax(jp_logits, dim=-1)[paris_id].item()
    jp_pred = model.to_string(jp_logits.argmax().item()).strip()

    print(f"\n  Baseline: pred='{jp_pred}', P(Tokyo)={p_tokyo_base:.6f}, P(Paris)={p_paris_jp:.6f}")

    for layer in [12, 16, 20]:
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
            ratio = p_paris / (p_tokyo + 1e-10)

            print(f"  L{layer} a={alpha:.0f}: P(Tokyo)={p_tokyo:.6f}, P(Paris)={p_paris:.6f}, ratio={ratio:.4f}, pred='{new_pred}'")

    # ---- Part 5: Relation Swap Intervention ----
    print("\n--- Part 5: Relation Swap Intervention ---")
    print("  d_relation = h('capital') - h('currency')")
    print("  Add alpha * d_relation to 'The currency of France is'")

    prompt_cap = "The capital of France is"
    prompt_cur = "The currency of France is"
    tokens_cap = model.to_tokens(prompt_cap)
    tokens_cur = model.to_tokens(prompt_cur)

    # Verify
    for prompt in [prompt_cap, prompt_cur]:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            logits = model(tokens)[0, -1]
            top_3 = torch.topk(logits, 3)
            top_strs = [model.to_string(t.item()).strip() for t in top_3.indices]
        print(f"  '{prompt}' -> {top_strs}")

    euro_id = safe_token_id("euro")

    # Compute relation directions
    relation_directions = {}
    for layer in [12, 16, 20]:
        with torch.no_grad():
            _, cache_cap = model.run_with_cache(tokens_cap, remove_batch_dim=True)
            _, cache_cur = model.run_with_cache(tokens_cur, remove_batch_dim=True)
        h_cap = cache_cap[f'blocks.{layer}.hook_resid_post'][-1].detach()
        h_cur = cache_cur[f'blocks.{layer}.hook_resid_post'][-1].detach()
        d = h_cap - h_cur
        relation_directions[layer] = (d / d.norm()).cuda()

    with torch.no_grad():
        cur_logits = model(tokens_cur)[0, -1]
    p_paris_cur = torch.softmax(cur_logits, dim=-1)[paris_id].item()
    p_euro_cur = torch.softmax(cur_logits, dim=-1)[euro_id].item()
    cur_pred = model.to_string(cur_logits.argmax().item()).strip()

    print(f"\n  Baseline: pred='{cur_pred}', P(Paris)={p_paris_cur:.6f}, P(euro)={p_euro_cur:.6f}")

    for layer in [12, 16, 20]:
        d_rel = relation_directions[layer]
        for alpha in [0.0, 2.0, 5.0, 10.0, 20.0]:
            def hook_fn(h, hook, direction=d_rel, scale=alpha):
                h[:, -1, :] = h[:, -1, :] + scale * direction
                return h

            with torch.no_grad():
                model.reset_hooks()
                model.add_hook(f'blocks.{layer}.hook_resid_post', hook_fn)
                int_logits = model(tokens_cur)[0, -1]
                model.reset_hooks()

            p_paris = torch.softmax(int_logits, dim=-1)[paris_id].item()
            p_euro = torch.softmax(int_logits, dim=-1)[euro_id].item()
            new_pred = model.to_string(int_logits.argmax().item()).strip()

            print(f"  L{layer} a={alpha:.0f}: P(Paris)={p_paris:.6f}, P(euro)={p_euro:.6f}, pred='{new_pred}'")

    # ---- Part 6: Compositional Intervention ----
    print("\n--- Part 6: Compositional Intervention ---")
    print("  Combine entity + relation directions")
    print("  Target: 'The currency of Germany is'")
    print("  Add: d_France-Germany (entity) + d_capital-currency (relation)")
    print("  Expected: shift toward 'The capital of France' = Paris")

    prompt_target = "The currency of Germany is"
    tokens_target = model.to_tokens(prompt_target)

    with torch.no_grad():
        target_logits = model(tokens_target)[0, -1]
    target_pred = model.to_string(target_logits.argmax().item()).strip()
    p_paris_target = torch.softmax(target_logits, dim=-1)[paris_id].item()
    print(f"\n  Baseline: '{prompt_target}' -> '{target_pred}', P(Paris)={p_paris_target:.6f}")

    for layer in [16]:
        d_ent = entity_directions[layer]
        d_rel = relation_directions[layer]

        for alpha_ent, alpha_rel in [(0,0), (5,5), (10,10), (20,20), (5,0), (0,5)]:
            def hook_fn(h, hook, de=d_ent, dr=d_rel, ae=alpha_ent, ar=alpha_rel):
                h[:, -1, :] = h[:, -1, :] + ae * de + ar * dr
                return h

            with torch.no_grad():
                model.reset_hooks()
                model.add_hook(f'blocks.{layer}.hook_resid_post', hook_fn)
                int_logits = model(tokens_target)[0, -1]
                model.reset_hooks()

            p_paris = torch.softmax(int_logits, dim=-1)[paris_id].item()
            new_pred = model.to_string(int_logits.argmax().item()).strip()

            print(f"  a_ent={alpha_ent:.0f}, a_rel={alpha_rel:.0f}: P(Paris)={p_paris:.6f}, pred='{new_pred}'")

    print("\n" + "=" * 70)
    print("QWEN3 INTERVENTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")
