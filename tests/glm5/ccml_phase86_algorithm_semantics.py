"""
Phase 86: Algorithm Semantics — From Representation Geometry to Computation Structure
=====================================================================================

THE CRITICAL GAP IDENTIFIED BY PHASE 85 CRITIQUE:

  We have been studying REPRESENTATION GEOMETRY:
    - Δh, cosine, subspace, participation ratio, rewriting direction

  We have NOT been studying ALGORITHM SEMANTICS:
    - What is the model COMPUTING?
    - Which dimensions correspond to: variables, relations, memory slots, 
      temporary state, control flow?
    - representation change ≠ algorithm identification

THE KEY INSIGHT:

  Transformer computation = latent state rewriting
    h_{t+1} = h_t + Δ(h_t, context)

  This makes it a:
    - recursive state machine
    - latent workspace
    - iterative rewrite system

  NOT a "collection of matrix operators."

THE FOUR EXPERIMENTS:

A. Entity-Relation Binding Decomposition ★★★★★ (MOST CRITICAL)
   - Create a 2D grid: {entity} × {relation}
   - Example: "The capital of France" vs "The capital of Germany"
             "The currency of France" vs "The currency of Germany"
   - By fixing entity and varying relation → find RELATION subspace
   - By fixing relation and varying entity → find ENTITY subspace
   - Test: are entity and relation subspaces linearly separable?
   - This identifies the "variable binding" structure of computation

B. Role-Filler Decomposition ★★★★★
   - "Paris loves Rome" vs "Rome loves Paris" — same words, different roles
   - Subject and object are ROLES; Paris and Rome are FILLERS
   - Test: can we find separate subspaces for role vs filler?
   - This identifies how the model binds variables to positions

C. Causal Subspace Intervention ★★★★★ (THE DECISIVE TEST)
   - Find entity direction: d_entity = h("France") - h("Germany")
   - Add α·d_entity to "The capital of Germany" prompt
   - If output shifts from "Berlin" toward "Paris" → entity subspace is CAUSAL
   - This is NOT ablation — this is DIRECTIONAL INTERVENTION
   - Only this can prove: representation carries computable structure

D. Algorithm State Tracing ★★★★
   - Multi-step reasoning: "A is B. B is C. A is?" → "C"
   - Track how "answer state" evolves across reasoning steps
   - Does the representation show phase transitions?
   - Does the reasoning path form a specific orbit structure?

WHY THIS MATTERS:

  If we can show:
  1. Entity subspace exists (separable from relation subspace)
  2. Relation subspace exists (separable from entity subspace)
  3. Causal intervention on entity subspace changes computation
  4. Causal intervention on relation subspace changes computation

  Then we've identified the COMPUTATIONAL STRUCTURE, not just geometry.
  We've shown that representations carry VARIABLES and RELATIONS as
  separable components — exactly what an algorithm needs.

Usage:
  python ccml_phase86_algorithm_semantics.py --exp a
  python ccml_phase86_algorithm_semantics.py --exp b
  python ccml_phase86_algorithm_semantics.py --exp c
  python ccml_phase86_algorithm_semantics.py --exp d
  python ccml_phase86_algorithm_semantics.py --exp all
"""

import torch
import numpy as np
import argparse
import time
from collections import defaultdict
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


# ============================================================
# Task Generators for Algorithm Semantics
# ============================================================

# Relation templates: {relation_name: template_with_placeholder}
RELATION_TEMPLATES = {
    "capital": "The capital of {} is",
    "currency": "The currency of {} is",
    "language": "The language of {} is",
    "continent": "The continent of {} is",
}

# Entity lists for each relation
ENTITIES = {
    "capital": ["France", "Germany", "Japan", "Brazil", "Italy",
                "Spain", "China", "India", "Egypt", "Australia",
                "Canada", "Mexico", "Korea", "Russia", "Turkey"],
    "currency": ["France", "Germany", "Japan", "Brazil", "Italy",
                 "Spain", "China", "India", "Egypt", "Australia",
                 "Canada", "Mexico", "Korea", "Russia", "Turkey"],
    "language": ["France", "Germany", "Japan", "Brazil", "Italy",
                 "Spain", "China", "India", "Egypt", "Australia",
                 "Canada", "Mexico", "Korea", "Russia", "Turkey"],
    "continent": ["France", "Germany", "Japan", "Brazil", "Italy",
                  "Spain", "China", "India", "Egypt", "Australia",
                  "Canada", "Mexico", "Korea", "Russia", "Turkey"],
}

# For role-filler decomposition
ROLE_FILLER_TEMPLATES = [
    "{subj} loves {obj}",
    "{subj} hates {obj}",
    "{subj} visits {obj}",
    "{subj} helps {obj}",
    "{subj} knows {obj}",
]

ROLE_FILLER_ENTITIES = ["Paris", "Rome", "Berlin", "London", "Tokyo",
                         "Madrid", "Oslo", "Seoul", "Delhi", "Cairo"]


# ============================================================
# Experiment A: Entity-Relation Binding Decomposition
# ============================================================
def exp_a_entity_relation_binding():
    """
    THE MOST CRITICAL EXPERIMENT: Variable Binding Structure

    We create a 2D grid: {entity} × {relation}
    - "The capital of France is" vs "The capital of Germany is" (vary entity)
    - "The capital of France is" vs "The currency of France is" (vary relation)

    By comparing these, we can decompose the representation into:
    - Entity subspace: dimensions that change when entity changes (relation fixed)
    - Relation subspace: dimensions that change when relation changes (entity fixed)

    If these subspaces are SEPARABLE → the model has variable binding structure.
    """
    print("=" * 70)
    print("EXPERIMENT A: Entity-Relation Binding Decomposition")
    print("=" * 70)

    model = get_model()
    relations = ["capital", "currency", "language"]
    entities = ["France", "Germany", "Japan", "Brazil", "Italy",
                "Spain", "China", "India", "Egypt", "Australia",
                "Canada", "Mexico", "Korea", "Russia", "Turkey"]

    # ---- Part 1: Collect representations for the 2D grid ----
    print("\n--- Part 1: Building Entity × Relation Representation Grid ---")

    # For each (relation, entity) pair, get the last-token residual stream
    # at each layer
    layers_to_check = [2, 4, 6, 8, 10]
    grid_reps = {}  # (layer, relation, entity) -> h vector

    for layer in layers_to_check:
        for relation in relations:
            for entity in entities:
                prompt = RELATION_TEMPLATES[relation].format(entity)
                tokens = model.to_tokens(prompt)
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

                h = cache[f'blocks.{layer}.hook_resid_post'][-1].detach()
                grid_reps[(layer, relation, entity)] = h

    print(f"  Collected {len(grid_reps)} representation vectors")

    # ---- Part 2: Entity subspace vs Relation subspace ----
    print("\n--- Part 2: Entity Subspace vs Relation Subspace ---")
    print("  Key test: Are entity changes and relation changes in SEPARATE subspaces?")

    for layer in layers_to_check:
        # Entity variation: fix relation, vary entity
        # For each relation, compute the set of representations across entities
        entity_variation_vectors = []  # Δh when entity changes (relation fixed)
        relation_variation_vectors = []  # Δh when relation changes (entity fixed)

        # Entity variation: for each relation, compute pairwise entity diffs
        for relation in relations:
            for i, e1 in enumerate(entities):
                for j, e2 in enumerate(entities):
                    if j > i:
                        h1 = grid_reps[(layer, relation, e1)]
                        h2 = grid_reps[(layer, relation, e2)]
                        entity_variation_vectors.append(h1 - h2)

        # Relation variation: for each entity, compute pairwise relation diffs
        for entity in entities:
            for i, r1 in enumerate(relations):
                for j, r2 in enumerate(relations):
                    if j > i:
                        h1 = grid_reps[(layer, r1, entity)]
                        h2 = grid_reps[(layer, r2, entity)]
                        relation_variation_vectors.append(h1 - h2)

        entity_var = torch.stack(entity_variation_vectors)  # [N_e, d_model]
        relation_var = torch.stack(relation_variation_vectors)  # [N_r, d_model]

        # SVD of each variation set
        entity_var_centered = entity_var - entity_var.mean(0)
        relation_var_centered = relation_var - relation_var.mean(0)

        # SVD: A = U @ diag(S) @ Vh
        # For subspace analysis we need RIGHT singular vectors (Vh.T)
        # which span the COLUMN space (d_model dimensions)
        _, S_e, Vh_e = torch.linalg.svd(entity_var_centered, full_matrices=False)
        _, S_r, Vh_r = torch.linalg.svd(relation_var_centered, full_matrices=False)

        # Vh_e: [k_e, d_model], Vh_r: [k_r, d_model]
        # Vh_e.T: [d_model, k_e] — right singular vectors as columns

        # Participation ratios
        pr_entity = (S_e.sum()**2) / (S_e**2).sum()
        pr_relation = (S_r.sum()**2) / (S_r**2).sum()

        # Subspace overlap using principal angles
        k = min(10, S_e.shape[0], S_r.shape[0])

        V_e_top = Vh_e[:k, :].T  # [d_model, k]
        V_r_top = Vh_r[:k, :].T  # [d_model, k]

        # Subspace overlap = ||V_e^T V_r||_F / sqrt(k)
        # If subspaces are orthogonal, this is 0
        # If subspaces are identical, this is 1
        overlap_matrix = V_e_top.T @ V_r_top  # [k, k]
        subspace_overlap = torch.linalg.norm(overlap_matrix, 'fro').item() / np.sqrt(k)

        # Variance of entity variation explained by relation subspace
        # Project each entity_var row onto relation subspace
        proj_e_on_r = entity_var_centered @ V_r_top @ V_r_top.T  # [N_e, d_model]
        entity_var_explained = (proj_e_on_r ** 2).sum() / (entity_var_centered ** 2).sum()

        # Variance of relation variation explained by entity subspace
        proj_r_on_e = relation_var_centered @ V_e_top @ V_e_top.T  # [N_r, d_model]
        relation_var_explained = (proj_r_on_e ** 2).sum() / (relation_var_centered ** 2).sum()

        print(f"\n  Layer {layer}:")
        print(f"    Entity variation: PR={pr_entity:.1f}, top-5 SVs={[round(s,2) for s in S_e[:5].tolist()]}")
        print(f"    Relation variation: PR={pr_relation:.1f}, top-5 SVs={[round(s,2) for s in S_r[:5].tolist()]}")
        print(f"    Subspace overlap (normalized): {subspace_overlap:.4f}")
        print(f"    Entity var explained by relation subspace: {entity_var_explained:.4f}")
        print(f"    Relation var explained by entity subspace: {relation_var_explained:.4f}")

        if entity_var_explained < 0.3 and relation_var_explained < 0.3:
            print(f"    *** ENTITY AND RELATION SUBSPACES ARE SEPARABLE! ***")
            print(f"    *** Model has VARIABLE BINDING structure ***")

    # ---- Part 3: Linear Separability Test ----
    print("\n--- Part 3: Linear Separability of Entity and Relation ---")
    print("  Can a linear probe distinguish entity from relation in the representation?")

    for layer in layers_to_check:
        # Build dataset: (representation, entity_label, relation_label)
        X = []
        entity_labels = []
        relation_labels = []

        for relation in relations:
            for entity in entities:
                h = grid_reps[(layer, relation, entity)]
                X.append(h)
                entity_labels.append(entity)
                relation_labels.append(relation)

        X = torch.stack(X)  # [N, d_model]
        X_centered = X - X.mean(0)

        # Entity probe: predict which entity from h
        # Use leave-one-out approach: for each entity, compute mean h
        entity_means = {}
        for entity in entities:
            idxs = [i for i, e in enumerate(entity_labels) if e == entity]
            entity_means[entity] = X[idxs].mean(0)

        # Relation probe: predict which relation from h
        relation_means = {}
        for relation in relations:
            idxs = [i for i, r in enumerate(relation_labels) if r == relation]
            relation_means[relation] = X[idxs].mean(0)

        # Test: can we decode entity by nearest-mean?
        entity_correct = 0
        for i, h in enumerate(X):
            dists = {e: (h - m).norm().item() for e, m in entity_means.items()}
            pred = min(dists, key=dists.get)
            if pred == entity_labels[i]:
                entity_correct += 1

        # Test: can we decode relation by nearest-mean?
        relation_correct = 0
        for i, h in enumerate(X):
            dists = {r: (h - m).norm().item() for r, m in relation_means.items()}
            pred = min(dists, key=dists.get)
            if pred == relation_labels[i]:
                relation_correct += 1

        entity_acc = entity_correct / len(X)
        relation_acc = relation_correct / len(X)
        chance_entity = 1.0 / len(entities)
        chance_relation = 1.0 / len(relations)

        print(f"\n  Layer {layer}:")
        print(f"    Entity decoding accuracy: {entity_acc:.4f} (chance: {chance_entity:.4f})")
        print(f"    Relation decoding accuracy: {relation_acc:.4f} (chance: {chance_relation:.4f})")

    # ---- Part 4: Cross-Task Cosine Decomposition ----
    print("\n--- Part 4: Cosine Decomposition ---")
    print("  How much of representation change is entity-driven vs relation-driven?")

    for layer in layers_to_check:
        # Same entity, different relation: pure relation effect
        same_entity_diff_relation = []
        for entity in entities[:10]:
            for i, r1 in enumerate(relations):
                for j, r2 in enumerate(relations):
                    if j > i:
                        h1 = grid_reps[(layer, r1, entity)]
                        h2 = grid_reps[(layer, r2, entity)]
                        cos = torch.nn.functional.cosine_similarity(
                            h1.unsqueeze(0), h2.unsqueeze(0)
                        ).item()
                        same_entity_diff_relation.append(cos)

        # Same relation, different entity: pure entity effect
        same_relation_diff_entity = []
        for relation in relations:
            for i, e1 in enumerate(entities[:10]):
                for j, e2 in enumerate(entities[:10]):
                    if j > i:
                        h1 = grid_reps[(layer, relation, e1)]
                        h2 = grid_reps[(layer, relation, e2)]
                        cos = torch.nn.functional.cosine_similarity(
                            h1.unsqueeze(0), h2.unsqueeze(0)
                        ).item()
                        same_relation_diff_entity.append(cos)

        # Different entity, different relation: combined effect
        diff_entity_diff_relation = []
        for i, (r1, e1) in enumerate([(r, e) for r in relations for e in entities[:5]]):
            for j, (r2, e2) in enumerate([(r, e) for r in relations for e in entities[:5]]):
                if j > i and r1 != r2 and e1 != e2:
                    h1 = grid_reps[(layer, r1, e1)]
                    h2 = grid_reps[(layer, r2, e2)]
                    cos = torch.nn.functional.cosine_similarity(
                        h1.unsqueeze(0), h2.unsqueeze(0)
                    ).item()
                    diff_entity_diff_relation.append(cos)

        print(f"\n  Layer {layer}:")
        print(f"    Same entity, diff relation: cosine = {np.mean(same_entity_diff_relation):.4f} ± {np.std(same_entity_diff_relation):.4f}")
        print(f"    Same relation, diff entity: cosine = {np.mean(same_relation_diff_entity):.4f} ± {np.std(same_relation_diff_entity):.4f}")
        print(f"    Diff entity, diff relation: cosine = {np.mean(diff_entity_diff_relation):.4f} ± {np.std(diff_entity_diff_relation):.4f}")

        # Key diagnostic: which factor dominates?
        entity_effect = 1 - np.mean(same_relation_diff_entity)  # larger = more entity variation
        relation_effect = 1 - np.mean(same_entity_diff_relation)  # larger = more relation variation

        print(f"    Entity effect strength: {entity_effect:.4f}")
        print(f"    Relation effect strength: {relation_effect:.4f}")
        if entity_effect > 0 and relation_effect > 0:
            ratio = entity_effect / relation_effect
            print(f"    Entity/Relation effect ratio: {ratio:.2f}")
            if 0.5 < ratio < 2.0:
                print(f"    *** ENTITY AND RELATION HAVE COMPARABLE MAGNITUDE — BINDING IS BALANCED ***")

    print("\n" + "=" * 70)
    print("EXPERIMENT A COMPLETE")
    print("=" * 70)


# ============================================================
# Experiment B: Role-Filler Decomposition
# ============================================================
def exp_b_role_filler():
    """
    Test whether the model separates ROLES from FILLERS.

    "Paris loves Rome" vs "Rome loves Paris"
    Same words, different ROLES (subject vs object).

    If the model has variable binding:
    - There should be a "role subspace" that encodes subject/object
    - There should be a "filler subspace" that encodes Paris/Rome
    - These should be separable
    """
    print("=" * 70)
    print("EXPERIMENT B: Role-Filler Decomposition")
    print("=" * 70)

    model = get_model()
    layers_to_check = [2, 4, 6, 8, 10]

    # ---- Part 1: Collect representations for role-filler grid ----
    print("\n--- Part 1: Building Role × Filler Representation Grid ---")

    # Use pairs of cities in subject-object positions
    city_pairs = [
        ("Paris", "Rome"), ("Berlin", "London"), ("Tokyo", "Seoul"),
        ("Madrid", "Oslo"), ("Delhi", "Cairo"), ("Paris", "Berlin"),
        ("Rome", "London"), ("Tokyo", "Delhi"), ("Seoul", "Cairo"),
        ("Madrid", "Paris"), ("London", "Berlin"), ("Oslo", "Seoul"),
    ]

    verbs = ["loves", "hates", "visits", "helps", "knows"]

    # For efficiency, use a subset
    # Use symmetric pairs so both orderings exist for swap analysis
    test_cities = ["Paris", "Rome", "Berlin", "London", "Tokyo", "Madrid", "Oslo", "Seoul"]
    test_verbs = verbs[:3]

    # Generate all ordered pairs (both directions)
    test_pairs_ordered = []
    for i, c1 in enumerate(test_cities[:5]):
        for j, c2 in enumerate(test_cities[:5]):
            if i != j:
                test_pairs_ordered.append((c1, c2))

    # Collect representations at the LAST token position
    # Template: "{subj} {verb} {obj}"
    grid_reps = {}  # (layer, verb, subj, obj) -> h at last position

    for layer in layers_to_check:
        for verb in test_verbs:
            for subj, obj in test_pairs_ordered:
                prompt = f"{subj} {verb} {obj}"
                tokens = model.to_tokens(prompt)
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

                # Get representation at last token position
                h = cache[f'blocks.{layer}.hook_resid_post'][-1].detach()
                grid_reps[(layer, verb, subj, obj)] = h

    print(f"  Collected {len(grid_reps)} representation vectors")

    # ---- Part 2: Role subspace analysis ----
    print("\n--- Part 2: Role Subspace — Subject vs Object ---")

    for layer in layers_to_check:
        # Same cities, swapped roles: "Paris loves Rome" vs "Rome loves Paris"
        # The difference should be the "role encoding"

        role_diff_vectors = []  # Δh from swapping subject and object
        for verb in test_verbs:
            for subj, obj in test_pairs_ordered:
                # Only use pairs where swapped version also exists
                if (layer, verb, obj, subj) in grid_reps:
                    h_normal = grid_reps[(layer, verb, subj, obj)]
                    h_swapped = grid_reps[(layer, verb, obj, subj)]
                    role_diff_vectors.append(h_normal - h_swapped)

        role_diffs = torch.stack(role_diff_vectors)  # [N, d_model]
        role_diffs_centered = role_diffs - role_diffs.mean(0)

        U_role, S_role, Vh_role = torch.linalg.svd(role_diffs_centered, full_matrices=False)
        pr_role = (S_role.sum()**2) / (S_role**2).sum()

        # Also: same role, different filler
        filler_diff_vectors = []  # Δh from changing which city is in a role
        for verb in test_verbs:
            pairs_subset = test_pairs_ordered[:6]
            for i, (s1, o1) in enumerate(pairs_subset):
                for j, (s2, o2) in enumerate(pairs_subset):
                    if j > i:
                        # Same structure (subj-verb-obj), different fillers
                        h1 = grid_reps[(layer, verb, s1, o1)]
                        h2 = grid_reps[(layer, verb, s2, o2)]
                        filler_diff_vectors.append(h1 - h2)

        filler_diffs = torch.stack(filler_diff_vectors)
        filler_diffs_centered = filler_diffs - filler_diffs.mean(0)

        _, S_filler, Vh_filler = torch.linalg.svd(filler_diffs_centered, full_matrices=False)
        pr_filler = (S_filler.sum()**2) / (S_filler**2).sum()

        # Subspace overlap using RIGHT singular vectors
        k = min(5, S_role.shape[0], S_filler.shape[0])
        V_role_top = Vh_role[:k, :].T  # [d_model, k]
        V_filler_top = Vh_filler[:k, :].T  # [d_model, k]

        overlap_matrix = V_role_top.T @ V_filler_top
        subspace_overlap = torch.linalg.norm(overlap_matrix, 'fro').item() / np.sqrt(k)

        # Variance explained
        proj_role_on_filler = role_diffs_centered @ V_filler_top @ V_filler_top.T
        role_var_in_filler = (proj_role_on_filler ** 2).sum() / (role_diffs_centered ** 2).sum()
        proj_filler_on_role = filler_diffs_centered @ V_role_top @ V_role_top.T
        filler_var_in_role = (proj_filler_on_role ** 2).sum() / (filler_diffs_centered ** 2).sum()

        # Cosine between role and filler diffs
        avg_role_diff = role_diffs.mean(0)
        avg_filler_diff = filler_diffs.mean(0)
        role_filler_cos = torch.nn.functional.cosine_similarity(
            avg_role_diff.unsqueeze(0), avg_filler_diff.unsqueeze(0)
        ).item()

        print(f"\n  Layer {layer}:")
        print(f"    Role (subj<->obj swap) variation: PR={pr_role:.1f}, top-5 SVs={[round(s,2) for s in S_role[:5].tolist()]}")
        print(f"    Filler (city identity) variation: PR={pr_filler:.1f}, top-5 SVs={[round(s,2) for s in S_filler[:5].tolist()]}")
        print(f"    Subspace overlap: {subspace_overlap:.4f}")
        print(f"    Role-filler cosine: {role_filler_cos:.4f}")

        if abs(role_filler_cos) < 0.3:
            print(f"    *** ROLE AND FILLER ENCODINGS ARE NEARLY ORTHOGONAL! ***")

    # ---- Part 3: Position-specific encoding ----
    print("\n--- Part 3: Position-Specific Token Representations ---")
    print("  How does the same word differ when it's in subject vs object position?")

    for layer in layers_to_check:
        # For city "Paris": compare "Paris loves Rome" (subj) vs "Rome loves Paris" (obj)
        city = "Paris"
        subj_reps = []
        obj_reps = []

        for verb in test_verbs:
            for other in ["Rome", "Berlin", "London", "Madrid"]:
                # Paris as subject
                prompt_subj = f"{city} {verb} {other}"
                tokens = model.to_tokens(prompt_subj)
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
                # Get representation at the position of "Paris" (first content token)
                h_subj = cache[f'blocks.{layer}.hook_resid_post'][1].detach()  # position 1 ≈ "Paris"
                subj_reps.append(h_subj)

                # Paris as object
                prompt_obj = f"{other} {verb} {city}"
                tokens = model.to_tokens(prompt_obj)
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
                # Get representation at the position of "Paris" (last content token before last)
                h_obj = cache[f'blocks.{layer}.hook_resid_post'][-2].detach()
                obj_reps.append(h_obj)

        subj_reps = torch.stack(subj_reps)
        obj_reps = torch.stack(obj_reps)

        # Same word, different role → should be different if role is encoded
        cross_role_cos = []
        for i in range(len(subj_reps)):
            cos = torch.nn.functional.cosine_similarity(
                subj_reps[i].unsqueeze(0), obj_reps[i].unsqueeze(0)
            ).item()
            cross_role_cos.append(cos)

        # Same role, different context → should be similar if role is stable
        within_subj_cos = []
        for i in range(len(subj_reps)):
            for j in range(i+1, len(subj_reps)):
                cos = torch.nn.functional.cosine_similarity(
                    subj_reps[i].unsqueeze(0), subj_reps[j].unsqueeze(0)
                ).item()
                within_subj_cos.append(cos)

        print(f"\n  Layer {layer}: '{city}' as subject vs object:")
        print(f"    Cross-role cosine (subj vs obj): {np.mean(cross_role_cos):.4f} ± {np.std(cross_role_cos):.4f}")
        print(f"    Within-subj cosine: {np.mean(within_subj_cos):.4f} ± {np.std(within_subj_cos):.4f}")

        if np.mean(cross_role_cos) < 0.9:
            print(f"    *** SAME WORD HAS DIFFERENT REPRESENTATIONS IN DIFFERENT ROLES! ***")
            print(f"    *** Role information is superimposed on entity representation ***")

    print("\n" + "=" * 70)
    print("EXPERIMENT B COMPLETE")
    print("=" * 70)


# ============================================================
# Experiment C: Causal Subspace Intervention
# ============================================================
def exp_c_causal_intervention():
    """
    THE DECISIVE TEST: Does subspace intervention causally change computation?

    This is NOT ablation. This is DIRECTIONAL INTERVENTION:
      h ← h + α · v

    Where v is a direction extracted from the representation structure.

    Key tests:
    1. Entity swap: Add d_entity = h("France") - h("Germany") to "capital of Germany"
       → If output shifts from "Berlin" toward "Paris", entity subspace is CAUSAL

    2. Relation swap: Add d_relation = h("capital") - h("currency") to "currency of France"
       → If output shifts from "euro" toward "Paris", relation subspace is CAUSAL

    This proves: representation carries computable structure, not just geometric patterns.
    """
    print("=" * 70)
    print("EXPERIMENT C: Causal Subspace Intervention")
    print("=" * 70)

    model = get_model()
    layers_to_check = [4, 6, 8]

    # ---- Part 1: Entity Swap Intervention ----
    print("\n--- Part 1: Entity Swap Intervention ---")
    print("  d_entity = h('France') - h('Germany')")
    print("  Add α·d_entity to 'The capital of Germany is'")
    print("  Expected: output shifts from 'Berlin' toward 'Paris'")

    # Collect entity direction at each layer
    prompt_france = "The capital of France is"
    prompt_germany = "The capital of Germany is"

    entity_directions = {}  # layer -> direction vector

    for layer in layers_to_check:
        tokens_fr = model.to_tokens(prompt_france)
        tokens_de = model.to_tokens(prompt_germany)

        with torch.no_grad():
            _, cache_fr = model.run_with_cache(tokens_fr, remove_batch_dim=True)
            _, cache_de = model.run_with_cache(tokens_de, remove_batch_dim=True)

        h_fr = cache_fr[f'blocks.{layer}.hook_resid_post'][-1].detach()
        h_de = cache_de[f'blocks.{layer}.hook_resid_post'][-1].detach()

        # Entity direction: points from "Germany" representation toward "France"
        d_entity = h_fr - h_de
        entity_directions[layer] = d_entity / d_entity.norm()  # normalize

    # Now intervene: add α·d_entity to "capital of Germany" at specific layer
    print(f"\n  Baseline: 'The capital of Germany is' → ", end="")
    tokens_de = model.to_tokens(prompt_germany)
    with torch.no_grad():
        baseline_logits = model(tokens_de)[0, -1]
    baseline_pred = model.to_string(baseline_logits.argmax().item())
    print(f"'{baseline_pred}'")

    print(f"\n  Baseline: 'The capital of France is' → ", end="")
    tokens_fr = model.to_tokens(prompt_france)
    with torch.no_grad():
        fr_logits = model(tokens_fr)[0, -1]
    fr_pred = model.to_string(fr_logits.argmax().item())
    print(f"'{fr_pred}'")

    # Get token IDs for measuring probability shift
    paris_id = model.to_single_token(" Paris")
    berlin_id = model.to_single_token(" Berlin")

    print(f"\n  Entity Swap Intervention Results:")
    print(f"  {'Layer':>6s}  {'alpha':>8s}  {'P(Paris)':>10s}  {'P(Berlin)':>10s}  {'P(Paris)/P(Berlin)':>18s}  {'Effect':>20s}")

    for layer in layers_to_check:
        d_entity = entity_directions[layer]

        for alpha in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
            # Intervene: add α·d_entity to residual stream at this layer
            def hook_intervene(h, hook, direction=d_entity, scale=alpha):
                # Add direction to the last token position
                h[:, -1, :] = h[:, -1, :] + scale * direction.to(h.device)
                return h

            with torch.no_grad():
                model.reset_hooks()
                model.add_hook(f'blocks.{layer}.hook_resid_post', hook_intervene)
                intervened_logits = model(tokens_de)[0, -1]
                model.reset_hooks()

            p_paris = torch.softmax(intervened_logits, dim=-1)[paris_id].item()
            p_berlin = torch.softmax(intervened_logits, dim=-1)[berlin_id].item()
            ratio = p_paris / (p_berlin + 1e-10)

            # Check if prediction changed
            new_pred = model.to_string(intervened_logits.argmax().item())
            if alpha == 0.0:
                effect = "baseline"
            elif new_pred == "Paris" or new_pred == " Paris":
                effect = "SWAPPED→Paris"
            elif ratio > 2.0:
                effect = "shifted→Paris"
            elif ratio > 1.0:
                effect = "slight shift"
            else:
                effect = "no effect"

            print(f"  {layer:6d}  {alpha:8.1f}  {p_paris:10.6f}  {p_berlin:10.6f}  {ratio:18.4f}  {effect:>20s}")

    # ---- Part 2: Relation Swap Intervention ----
    print("\n--- Part 2: Relation Swap Intervention ---")
    print("  d_relation = h('capital') - h('currency')")
    print("  Add α·d_relation to 'The currency of France is'")
    print("  Expected: output shifts from 'euro' toward 'Paris'")

    # Collect relation direction
    prompt_capital = "The capital of France is"
    prompt_currency = "The currency of France is"

    relation_directions = {}

    for layer in layers_to_check:
        tokens_cap = model.to_tokens(prompt_capital)
        tokens_cur = model.to_tokens(prompt_currency)

        with torch.no_grad():
            _, cache_cap = model.run_with_cache(tokens_cap, remove_batch_dim=True)
            _, cache_cur = model.run_with_cache(tokens_cur, remove_batch_dim=True)

        h_cap = cache_cap[f'blocks.{layer}.hook_resid_post'][-1].detach()
        h_cur = cache_cur[f'blocks.{layer}.hook_resid_post'][-1].detach()

        d_relation = h_cap - h_cur
        relation_directions[layer] = d_relation / d_relation.norm()

    print(f"\n  Baseline: 'The currency of France is' → ", end="")
    tokens_cur = model.to_tokens(prompt_currency)
    with torch.no_grad():
        cur_logits = model(tokens_cur)[0, -1]
    cur_pred = model.to_string(cur_logits.argmax().item())
    print(f"'{cur_pred}'")

    # Get relevant token IDs
    paris_id = model.to_single_token(" Paris")
    euro_id = model.to_single_token(" euro")

    print(f"\n  Relation Swap Intervention Results:")
    print(f"  {'Layer':>6s}  {'alpha':>8s}  {'P(Paris)':>10s}  {'P(euro)':>10s}  {'P(Paris)/P(euro)':>18s}  {'Effect':>20s}")

    for layer in layers_to_check:
        d_relation = relation_directions[layer]

        for alpha in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
            def hook_intervene(h, hook, direction=d_relation, scale=alpha):
                h[:, -1, :] = h[:, -1, :] + scale * direction.to(h.device)
                return h

            with torch.no_grad():
                model.reset_hooks()
                model.add_hook(f'blocks.{layer}.hook_resid_post', hook_intervene)
                intervened_logits = model(tokens_cur)[0, -1]
                model.reset_hooks()

            p_paris = torch.softmax(intervened_logits, dim=-1)[paris_id].item()
            p_euro = torch.softmax(intervened_logits, dim=-1)[euro_id].item()
            ratio = p_paris / (p_euro + 1e-10)

            new_pred = model.to_string(intervened_logits.argmax().item())
            if alpha == 0.0:
                effect = "baseline"
            elif "Paris" in new_pred:
                effect = "SWAPPED→Paris"
            elif ratio > 2.0:
                effect = "shifted→Paris"
            elif ratio > 1.0:
                effect = "slight shift"
            else:
                effect = "no effect"

            print(f"  {layer:6d}  {alpha:8.1f}  {p_paris:10.6f}  {p_euro:10.6f}  {ratio:18.4f}  {effect:>20s}")

    # ---- Part 3: Cross-Entity Intervention Generalization ----
    print("\n--- Part 3: Cross-Entity Intervention Generalization ---")
    print("  d_entity computed from France-Germany pair")
    print("  Apply to Italy-Japan pair: does the SAME direction work?")
    print("  If yes → entity direction is UNIVERSAL, not pair-specific")

    # Entity direction from France-Germany (already computed)
    # Test on: "The capital of Japan is" — should shift toward "Rome" (Italy)
    prompt_japan = "The capital of Japan is"
    tokens_jp = model.to_tokens(prompt_japan)

    rome_id = model.to_single_token(" Rome")
    tokyo_id = model.to_single_token(" Tokyo")

    print(f"\n  Baseline: 'The capital of Japan is' → ", end="")
    with torch.no_grad():
        jp_logits = model(tokens_jp)[0, -1]
    jp_pred = model.to_string(jp_logits.argmax().item())
    print(f"'{jp_pred}'")

    print(f"\n  Cross-Entity Generalization:")
    print(f"  {'Layer':>6s}  {'alpha':>8s}  {'P(Rome)':>10s}  {'P(Tokyo)':>10s}  {'Ratio':>10s}  {'Effect':>20s}")

    for layer in layers_to_check:
        d_entity = entity_directions[layer]

        for alpha in [0.0, 2.0, 5.0, 10.0]:
            def hook_intervene(h, hook, direction=d_entity, scale=alpha):
                h[:, -1, :] = h[:, -1, :] + scale * direction.to(h.device)
                return h

            with torch.no_grad():
                model.reset_hooks()
                model.add_hook(f'blocks.{layer}.hook_resid_post', hook_intervene)
                intervened_logits = model(tokens_jp)[0, -1]
                model.reset_hooks()

            p_rome = torch.softmax(intervened_logits, dim=-1)[rome_id].item()
            p_tokyo = torch.softmax(intervened_logits, dim=-1)[tokyo_id].item()
            ratio = p_rome / (p_tokyo + 1e-10)

            new_pred = model.to_string(intervened_logits.argmax().item())
            if alpha == 0.0:
                effect = "baseline"
            elif ratio > 2.0:
                effect = "generalized!"
            elif ratio > 1.0:
                effect = "slight generalization"
            else:
                effect = "no generalization"

            print(f"  {layer:6d}  {alpha:8.1f}  {p_rome:10.6f}  {p_tokyo:10.6f}  {ratio:10.4f}  {effect:>20s}")

    # ---- Part 4: Multi-Layer Compositional Intervention ----
    print("\n--- Part 4: Multi-Layer Compositional Intervention ---")
    print("  Intervene at MULTIPLE layers simultaneously")
    print("  This tests whether entity information is distributed across layers")

    target_layers = [4, 6, 8]

    # Use smaller alpha when intervening at multiple layers
    for alpha_per_layer in [0.0, 1.0, 2.0, 5.0]:
        def hook_multi(h, hook, directions=entity_directions, scale=alpha_per_layer):
            layer_idx = int(hook.name.split('.')[1])
            if layer_idx in directions:
                d = directions[layer_idx].to(h.device)
                h[:, -1, :] = h[:, -1, :] + scale * d
            return h

        with torch.no_grad():
            model.reset_hooks()
            for l in target_layers:
                model.add_hook(f'blocks.{l}.hook_resid_post', hook_multi)
            intervened_logits = model(tokens_de)[0, -1]
            model.reset_hooks()

        p_paris = torch.softmax(intervened_logits, dim=-1)[paris_id].item()
        p_berlin = torch.softmax(intervened_logits, dim=-1)[berlin_id].item()
        ratio = p_paris / (p_berlin + 1e-10)
        new_pred = model.to_string(intervened_logits.argmax().item())

        print(f"  Multi-layer α={alpha_per_layer:.1f}: P(Paris)={p_paris:.6f}, "
              f"P(Berlin)={p_berlin:.6f}, ratio={ratio:.4f}, pred='{new_pred}'")

    print("\n" + "=" * 70)
    print("EXPERIMENT C COMPLETE")
    print("=" * 70)


# ============================================================
# Experiment D: Algorithm State Tracing
# ============================================================
def exp_d_algorithm_tracing():
    """
    Track algorithm execution in multi-step reasoning.

    Use Chain-of-Thought prompts where the model must:
    1. Parse the premises
    2. Apply transitive reasoning
    3. Generate the answer

    Track how the "algorithm state" evolves across reasoning steps.
    Look for phase transitions in the representation.
    """
    print("=" * 70)
    print("EXPERIMENT D: Algorithm State Tracing")
    print("=" * 70)

    model = get_model()
    layers_to_check = [2, 4, 6, 8, 10]

    # ---- Part 1: Simple transitive reasoning ----
    print("\n--- Part 1: Simple Transitive Reasoning ---")
    print("  'A is B. B is C. Therefore A is' → 'C'")
    print("  Track: how does the answer state evolve across the sequence?")

    # Create reasoning chains
    chains = [
        ("Paris is the capital of France. France is in Europe. Paris is in", " Europe"),
        ("Tokyo is the capital of Japan. Japan is in Asia. Tokyo is in", " Asia"),
        ("Berlin is the capital of Germany. Germany is in Europe. Berlin is in", " Europe"),
        ("Cairo is the capital of Egypt. Egypt is in Africa. Cairo is in", " Africa"),
        ("Seoul is the capital of Korea. Korea is in Asia. Seoul is in", " Asia"),
        ("Madrid is the capital of Spain. Spain is in Europe. Madrid is in", " Europe"),
        ("Rome is the capital of Italy. Italy is in Europe. Rome is in", " Europe"),
        ("Delhi is the capital of India. India is in Asia. Delhi is in", " Asia"),
        ("Oslo is the capital of Norway. Norway is in Europe. Oslo is in", " Europe"),
        ("London is the capital of England. England is in Europe. London is in", " Europe"),
    ]

    # Also create single-premise controls (no reasoning needed)
    single_premise = [
        "Paris is in",
        "Tokyo is in",
        "Berlin is in",
        "Cairo is in",
        "Seoul is in",
        "Madrid is in",
        "Rome is in",
        "Delhi is in",
        "Oslo is in",
        "London is in",
    ]

    # Track token-level representations for each position
    for layer in layers_to_check:
        chain_reps = []  # [chain_idx, seq_pos, d_model]
        single_reps = []  # [prompt_idx, d_model]

        for chain_prompt, _ in chains:
            tokens = model.to_tokens(chain_prompt)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

            # Get representation at last token position for each position
            h = cache[f'blocks.{layer}.hook_resid_post'][-1].detach()
            chain_reps.append(h)

        for prompt in single_premise:
            tokens = model.to_tokens(prompt)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

            h = cache[f'blocks.{layer}.hook_resid_post'][-1].detach()
            single_reps.append(h)

        chain_reps = torch.stack(chain_reps)  # [n_chains, d_model]
        single_reps = torch.stack(single_reps)  # [n_singles, d_model]

        # Compare: chain vs single premise at last position
        # Do reasoning chains produce different representations?
        chain_mean = chain_reps.mean(0)
        single_mean = single_reps.mean(0)

        cross_cos = torch.nn.functional.cosine_similarity(
            chain_mean.unsqueeze(0), single_mean.unsqueeze(0)
        ).item()

        # Also compare pairwise within each group
        chain_pairwise = []
        for i in range(len(chain_reps)):
            for j in range(i+1, len(chain_reps)):
                cos = torch.nn.functional.cosine_similarity(
                    chain_reps[i].unsqueeze(0), chain_reps[j].unsqueeze(0)
                ).item()
                chain_pairwise.append(cos)

        single_pairwise = []
        for i in range(len(single_reps)):
            for j in range(i+1, len(single_reps)):
                cos = torch.nn.functional.cosine_similarity(
                    single_reps[i].unsqueeze(0), single_reps[j].unsqueeze(0)
                ).item()
                single_pairwise.append(cos)

        print(f"\n  Layer {layer}:")
        print(f"    Chain vs Single mean cosine: {cross_cos:.4f}")
        print(f"    Chain internal cosine: {np.mean(chain_pairwise):.4f} ± {np.std(chain_pairwise):.4f}")
        print(f"    Single internal cosine: {np.mean(single_pairwise):.4f} ± {np.std(single_pairwise):.4f}")

    # ---- Part 2: Position-by-position representation tracking ----
    print("\n--- Part 2: Position-by-Position Representation Evolution ---")
    print("  Track h at each token position through the reasoning chain")

    # Use one detailed example
    test_chain = "Paris is the capital of France. France is in Europe. Paris is in"
    tokens = model.to_tokens(test_chain)
    n_tokens = tokens.shape[1]

    print(f"\n  Prompt: '{test_chain}'")
    print(f"  Token count: {n_tokens}")

    # Decode each token
    token_strs = [model.to_string(tokens[0, t].item()) for t in range(n_tokens)]
    print(f"  Tokens: {token_strs}")

    for layer in [4, 6, 8]:
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, remove_batch_dim=True)

        h_all = cache[f'blocks.{layer}.hook_resid_post'].detach()  # [seq_len, d_model]

        # Distance from the first token representation
        h_ref = h_all[0]
        dists_from_start = [(h_all[t] - h_ref).norm().item() for t in range(n_tokens)]

        # Step-by-step distances
        step_dists = [(h_all[t+1] - h_all[t]).norm().item() for t in range(n_tokens - 1)]

        print(f"\n  Layer {layer}:")
        print(f"    Distances from position 0: {[round(d, 2) for d in dists_from_start]}")
        print(f"    Step-by-step distances: {[round(d, 2) for d in step_dists]}")

        # Key question: is there a "reasoning phase transition" at "Therefore"?
        # Look for a spike in step distance at the reasoning step

    # ---- Part 3: Comparing reasoning with different difficulty ----
    print("\n--- Part 3: Reasoning Difficulty Effect on Representation ---")
    print("  1-hop: 'The capital of France is' (direct retrieval)")
    print("  2-hop: 'Paris is capital of France. France is in Europe. Paris is in' (transitive)")
    print("  Does 2-hop reasoning create different representation dynamics?")

    one_hop_prompts = [
        "The capital of France is",
        "The capital of Japan is",
        "The capital of Germany is",
        "The capital of Italy is",
        "The capital of Spain is",
        "The continent of France is",
        "The continent of Japan is",
        "The continent of Germany is",
        "The continent of Italy is",
        "The continent of Spain is",
    ]

    two_hop_prompts = [
        "Paris is the capital of France. France is in Europe. Paris is in",
        "Tokyo is the capital of Japan. Japan is in Asia. Tokyo is in",
        "Berlin is the capital of Germany. Germany is in Europe. Berlin is in",
        "Rome is the capital of Italy. Italy is in Europe. Rome is in",
        "Madrid is the capital of Spain. Spain is in Europe. Madrid is in",
    ]

    for layer in layers_to_check:
        one_hop_reps = []
        for prompt in one_hop_prompts:
            tokens = model.to_tokens(prompt)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            h = cache[f'blocks.{layer}.hook_resid_post'][-1].detach()
            one_hop_reps.append(h)

        two_hop_reps = []
        for prompt in two_hop_prompts:
            tokens = model.to_tokens(prompt)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            h = cache[f'blocks.{layer}.hook_resid_post'][-1].detach()
            two_hop_reps.append(h)

        one_hop_reps = torch.stack(one_hop_reps)
        two_hop_reps = torch.stack(two_hop_reps)

        # Compare means
        cos = torch.nn.functional.cosine_similarity(
            one_hop_reps.mean(0).unsqueeze(0),
            two_hop_reps.mean(0).unsqueeze(0)
        ).item()

        # Compare variance (effective dimensionality)
        U1, S1, _ = torch.linalg.svd(one_hop_reps - one_hop_reps.mean(0), full_matrices=False)
        U2, S2, _ = torch.linalg.svd(two_hop_reps - two_hop_reps.mean(0), full_matrices=False)

        pr1 = (S1.sum()**2) / (S1**2).sum()
        pr2 = (S2.sum()**2) / (S2**2).sum()

        print(f"\n  Layer {layer}:")
        print(f"    1-hop vs 2-hop mean cosine: {cos:.4f}")
        print(f"    1-hop PR: {pr1:.1f}, 2-hop PR: {pr2:.1f}")

        if pr2 > pr1 * 1.2:
            print(f"    *** 2-hop reasoning uses HIGHER-DIMENSIONAL representation space ***")
            print(f"    *** Transitive reasoning requires more computational dimensions ***")

    # ---- Part 4: Subspace separation by reasoning type ----
    print("\n--- Part 4: Answer Subspace vs Context Subspace ---")
    print("  In reasoning chains, can we separate:")
    print("  - 'Answer subspace': dimensions encoding the target answer")
    print("  - 'Context subspace': dimensions encoding the reasoning structure")

    # Collect representations for chains with same answer, different paths
    same_answer_chains = [
        ("Paris is in Europe.", " Europe"),
        ("France is in Europe. Paris is in France. Paris is in", " Europe"),
        ("Germany is in Europe. France borders Germany. France is in", " Europe"),
        ("Spain is in Europe. France borders Spain. France is in", " Europe"),
    ]

    # Collect representations for chains with different answers, same structure
    same_structure_chains = [
        "Paris is in Europe. France is in",
        "Tokyo is in Asia. Japan is in",
        "Berlin is in Europe. Germany is in",
        "Cairo is in Africa. Egypt is in",
    ]

    for layer in [6, 8]:
        answer_reps = []
        for prompt, _ in same_answer_chains:
            tokens = model.to_tokens(prompt)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            h = cache[f'blocks.{layer}.hook_resid_post'][-1].detach()
            answer_reps.append(h)

        structure_reps = []
        for prompt in same_structure_chains:
            tokens = model.to_tokens(prompt)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            h = cache[f'blocks.{layer}.hook_resid_post'][-1].detach()
            structure_reps.append(h)

        answer_reps = torch.stack(answer_reps)
        structure_reps = torch.stack(structure_reps)

        # Compare: same answer should be more similar than same structure
        # (if answer is encoded in a separable subspace)
        answer_pairwise = []
        for i in range(len(answer_reps)):
            for j in range(i+1, len(answer_reps)):
                cos = torch.nn.functional.cosine_similarity(
                    answer_reps[i].unsqueeze(0), answer_reps[j].unsqueeze(0)
                ).item()
                answer_pairwise.append(cos)

        structure_pairwise = []
        for i in range(len(structure_reps)):
            for j in range(i+1, len(structure_reps)):
                cos = torch.nn.functional.cosine_similarity(
                    structure_reps[i].unsqueeze(0), structure_reps[j].unsqueeze(0)
                ).item()
                structure_pairwise.append(cos)

        print(f"\n  Layer {layer}:")
        print(f"    Same answer (different paths): cosine = {np.mean(answer_pairwise):.4f} ± {np.std(answer_pairwise):.4f}")
        print(f"    Same structure (different answers): cosine = {np.mean(structure_pairwise):.4f} ± {np.std(structure_pairwise):.4f}")

        if np.mean(answer_pairwise) > np.mean(structure_pairwise):
            print(f"    *** ANSWER IS MORE STRONGLY ENCODED THAN REASONING STRUCTURE ***")
        else:
            print(f"    *** REASONING STRUCTURE IS MORE STRONGLY ENCODED THAN ANSWER ***")

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
        exp_a_entity_relation_binding()
    if args.exp in ["b", "all"]:
        exp_b_role_filler()
    if args.exp in ["c", "all"]:
        exp_c_causal_intervention()
    if args.exp in ["d", "all"]:
        exp_d_algorithm_tracing()

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")
