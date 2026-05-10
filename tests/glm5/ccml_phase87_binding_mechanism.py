"""
Phase 87: Binding Mechanism — Separating Feature Separation from Symbolic Binding
==================================================================================

THE CRITICAL DISTINCTION (from Phase 86 critique):

  What we proved in Phase 86:
    ✓ Entity and Relation subspaces are partially separable
    ✓ Role and Filler directions are nearly orthogonal (cosine ≈ 0)
    ✓ Entity direction has causal effect on output (16.3x shift on Qwen3)
    → This proves: STRUCTURED DISTRIBUTED REPRESENTATIONS

  What we did NOT prove:
    ✗ Symbolic variable binding (bind(role, filler) with compositional semantics)
    ✗ Stable binding operators (T: h_{X,subj} → h_{X,obj})
    ✗ Compositional generalization (entity_A + relation_B → relation_B(entity_A))
    ✗ Cross-template invariance (same relation subspace across different phrasings)

THE EVIDENCE HIERARCHY:
  Level 1: Separable representations (we're here)
  Level 2: Stable binding operator
  Level 3: Composable variable system
  Level 4: True algorithm execution

THE THREE CRITICAL EXPERIMENTS:

A. Role Transfer Operator ★★★★★ (MOST CRITICAL)
   - Test if there exists a STABLE LINEAR OPERATOR T such that:
     T(h_{X, subject}) ≈ h_{X, object}
   - If T exists and generalizes across fillers X → TRUE BINDING
   - If T is filler-dependent → just position-conditioned contextualization
   - This is the key experiment that distinguishes:
     * Symbolic binding: T works for any X
     * Feature separation: T only works for training X

B. Cross-Template Invariance ★★★★★
   - Test if "capital" relation subspace is the SAME across:
     * "The capital of France is"
     * "France's capital is"
     * "What is the capital of France"
     * "France has capital"
   - If relation subspace is template-invariant → relation is ABSTRACT
   - If relation subspace is template-specific → relation is just template statistics

C. Compositional Generalization ★★★★
   - Test if entity_A direction + relation_B direction → relation_B(entity_A) output
   - Specifically: if we add d_entity(France) to "The currency of Germany is"
     → Does output shift toward "euro" (currency of France)?
   - This tests if entity and relation are truly COMPOSABLE

D. Mechanism Localization (Head-Level) ★★★★
   - Which attention heads carry role information?
   - Which MLP layers perform the binding?
   - Head-level intervention to find the BINDING CIRCUIT
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import sys
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_score

def get_model(model_name="gpt2-small"):
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(
        model_name,
        center_unembed=False,
        center_writing_weights=False,
        fold_ln=False,
        device="cuda",
    )
    model.eval()
    return model

def safe_token_id(model, text):
    try:
        return model.to_single_token(text)
    except AssertionError:
        tokens = model.to_tokens(text)
        return tokens[0, -1].item()

# ============================================================
# EXPERIMENT A: Role Transfer Operator
# ============================================================
def experiment_a(model):
    """
    THE CRITICAL TEST: Is there a stable linear operator T such that
    T(h_{X,subject}) ≈ h_{X,object}?
    
    If binding is truly symbolic:
      T should work for ANY filler X (including unseen fillers)
    
    If it's just position-conditioned contextualization:
      T will only work for the fillers it was trained on
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT A: Role Transfer Operator")
    print("=" * 70)
    print("\nKey question: Is there a STABLE LINEAR OPERATOR T such that")
    print("  T(h_{X,subject}) ≈ h_{X,object} for ANY filler X?")
    print("If YES → true variable binding")
    print("If NO  → just position-conditioned contextualization")
    
    # Use symmetric templates so both orderings are natural
    # "X loves Y" vs "Y loves X"
    # "X hates Y" vs "Y hates X"
    # "X follows Y" vs "Y follows X"
    
    subjects = ["Paris", "Rome", "Berlin", "London", "Tokyo", "Madrid", "Oslo", "Seoul",
                "Alice", "Bob", "Carol", "Dave"]
    objects = subjects.copy()
    verbs = ["loves", "hates", "follows", "helps", "knows"]
    
    # Split into TRAIN and TEST fillers
    # Train: first 8 cities + first 2 names
    train_fillers = subjects[:8] + ["Alice", "Bob"]
    # Test: remaining names (UNSEEN during operator training)
    test_fillers = ["Carol", "Dave", "Eve", "Frank"]
    
    layers_to_check = [2, 4, 6, 8, 10]
    
    # Collect representations for ALL filler pairs
    print("\nCollecting representations...")
    # For each verb, for each pair (X, Y) where X≠Y:
    #   h_{X,subj} = repr of "X {verb} Y" at position of X
    #   h_{X,obj}  = repr of "Y {verb} X" at position of X
    
    # Actually, to control for position, we need the SAME filler at
    # subject position vs object position in the SAME structural context.
    # 
    # Better approach: 
    #   "X {verb} Z" → h_{X as subject} (at position of X)
    #   "Z {verb} X" → h_{X as object} (at position of X)
    # where Z is a fixed anchor word.
    
    # Even better: use PAIRED templates that keep the target word
    # at the SAME absolute position but with different roles.
    # 
    # Template 1: "{target} {verb} {anchor}"  → target is subject
    # Template 2: "{anchor} {verb} {target}"  → target is object
    #
    # Problem: position changes. Let's instead use:
    # Both at the LAST token position (where computation is done).
    
    anchor = "Rome"  # Fixed anchor word
    
    all_fillers = train_fillers + test_fillers
    filler_reps = {}  # (layer, filler, role) -> h at last position
    
    for layer in layers_to_check:
        for filler in all_fillers:
            if filler == anchor:
                continue
            for verb in verbs[:2]:  # Use 2 verbs for speed
                # Subject: "Filler {verb} Rome"
                prompt_subj = f"{filler} {verb} {anchor}"
                tokens_subj = model.to_tokens(prompt_subj)
                with torch.no_grad():
                    _, cache_subj = model.run_with_cache(tokens_subj, remove_batch_dim=True)
                h_subj = cache_subj[f'blocks.{layer}.hook_resid_post'][-1].detach().cpu()
                
                # Object: "Rome {verb} Filler"
                prompt_obj = f"{anchor} {verb} {filler}"
                tokens_obj = model.to_tokens(prompt_obj)
                with torch.no_grad():
                    _, cache_obj = model.run_with_cache(tokens_obj, remove_batch_dim=True)
                h_obj = cache_obj[f'blocks.{layer}.hook_resid_post'][-1].detach().cpu()
                
                key = (layer, filler, verb)
                if key not in filler_reps:
                    filler_reps[(layer, filler, 'subject')] = []
                    filler_reps[(layer, filler, 'object')] = []
                filler_reps[(layer, filler, 'subject')].append(h_subj)
                filler_reps[(layer, filler, 'object')].append(h_obj)
    
    # Average across verbs
    for layer in layers_to_check:
        for filler in all_fillers:
            if filler == anchor:
                continue
            for role in ['subject', 'object']:
                key = (layer, filler, role)
                if key in filler_reps and isinstance(filler_reps[key], list):
                    filler_reps[key] = torch.stack(filler_reps[key]).mean(0)
    
    print("\nAnalyzing role transfer operator at each layer...")
    
    for layer in layers_to_check:
        print(f"\n--- Layer {layer} ---")
        
        # Build training pairs
        train_h_subj = []
        train_h_obj = []
        for filler in train_fillers:
            if filler == anchor:
                continue
            key_s = (layer, filler, 'subject')
            key_o = (layer, filler, 'object')
            if key_s in filler_reps and key_o in filler_reps:
                if isinstance(filler_reps[key_s], torch.Tensor):
                    train_h_subj.append(filler_reps[key_s])
                    train_h_obj.append(filler_reps[key_o])
        
        if len(train_h_subj) < 3:
            print("  Not enough training fillers, skipping")
            continue
        
        train_h_subj = torch.stack(train_h_subj)  # [N_train, d]
        train_h_obj = torch.stack(train_h_obj)     # [N_train, d]
        
        # Build test pairs
        test_h_subj = []
        test_h_obj = []
        for filler in test_fillers:
            key_s = (layer, filler, 'subject')
            key_o = (layer, filler, 'object')
            if key_s in filler_reps and key_o in filler_reps:
                if isinstance(filler_reps[key_s], torch.Tensor):
                    test_h_subj.append(filler_reps[key_s])
                    test_h_obj.append(filler_reps[key_o])
        
        test_h_subj = torch.stack(test_h_subj) if test_h_subj else None
        test_h_obj = torch.stack(test_h_obj) if test_h_obj else None
        
        # ---- Method 1: Ridge Regression T(h_subj) ≈ h_obj ----
        # Find T: [d, d] matrix such that h_subj @ T^T ≈ h_obj
        # Use Ridge: T = (H_subj^T H_subj + λI)^{-1} H_subj^T H_obj
        
        lam = 1.0  # Regularization
        H_s = train_h_subj.numpy()
        H_o = train_h_obj.numpy()
        
        # Ridge regression: T = (H_s^T H_s + λI)^{-1} H_s^T H_o
        I_d = np.eye(H_s.shape[1])
        T_matrix = np.linalg.solve(H_s.T @ H_s + lam * I_d, H_s.T @ H_o)  # [d, d]
        
        # Evaluate on TRAIN set
        pred_train = H_s @ T_matrix
        train_cosines = []
        for i in range(len(H_s)):
            cos = F.cosine_similarity(
                torch.tensor(pred_train[i]), 
                torch.tensor(H_o[i]), 
                dim=0
            ).item()
            train_cosines.append(cos)
        
        # Evaluate on TEST set (UNSEEN fillers)
        test_cosines = []
        if test_h_subj is not None:
            H_s_test = test_h_subj.numpy()
            H_o_test = test_h_obj.numpy()
            pred_test = H_s_test @ T_matrix
            for i in range(len(H_s_test)):
                cos = F.cosine_similarity(
                    torch.tensor(pred_test[i]),
                    torch.tensor(H_o_test[i]),
                    dim=0
                ).item()
                test_cosines.append(cos)
        
        # ---- Method 2: Mean difference direction ----
        # Simpler: d_role = mean(h_obj - h_subj) over training fillers
        d_role = (train_h_obj - train_h_subj).mean(0)  # [d]
        d_role_norm = d_role / d_role.norm()
        
        # Test: h_subj + α * d_role ≈ h_obj for the right α?
        # Use optimal α per sample: α = (h_obj - h_subj) · d_role
        train_alphas = []
        for i in range(len(train_h_subj)):
            alpha_i = ((train_h_obj[i] - train_h_subj[i]) * d_role_norm).sum().item()
            train_alphas.append(alpha_i)
        
        # Apply to test
        test_alphas = []
        test_shift_cosines = []
        if test_h_subj is not None:
            for i in range(len(test_h_subj)):
                alpha_i = ((test_h_obj[i] - test_h_subj[i]) * d_role_norm).sum().item()
                test_alphas.append(alpha_i)
                h_shifted = test_h_subj[i] + alpha_i * d_role_norm
                cos = F.cosine_similarity(h_shifted, test_h_obj[i], dim=0).item()
                test_shift_cosines.append(cos)
        
        train_shift_cosines = []
        for i in range(len(train_h_subj)):
            h_shifted = train_h_subj[i] + train_alphas[i] * d_role_norm
            cos = F.cosine_similarity(h_shifted, train_h_obj[i], dim=0).item()
            train_shift_cosines.append(cos)
        
        # ---- Method 3: Cross-filler consistency ----
        # If T is truly a stable operator, then for fillers A and B:
        #   h_{A,obj} - h_{A,subj} should be parallel to h_{B,obj} - h_{B,subj}
        # This is the STRONGEST test: the ROLE SHIFT should be the SAME for all fillers
        
        role_shifts = train_h_obj - train_h_subj  # [N_train, d]
        cross_cosines = []
        for i in range(len(role_shifts)):
            for j in range(i+1, len(role_shifts)):
                cos = F.cosine_similarity(role_shifts[i], role_shifts[j], dim=0).item()
                cross_cosines.append(cos)
        
        # Also test with test fillers
        if test_h_subj is not None:
            test_role_shifts = test_h_obj - test_h_subj
            test_train_cross = []
            for i in range(len(test_role_shifts)):
                for j in range(len(role_shifts)):
                    cos = F.cosine_similarity(test_role_shifts[i], role_shifts[j], dim=0).item()
                    test_train_cross.append(cos)
        else:
            test_train_cross = []
        
        # Print results
        print(f"  Train fillers: {len(train_h_subj)}, Test fillers: {len(test_h_subj) if test_h_subj is not None else 0}")
        
        print(f"\n  Method 1: Ridge Regression Operator T(h_subj) → h_obj")
        print(f"    Train cosine: {np.mean(train_cosines):.4f} ± {np.std(train_cosines):.4f}")
        if test_cosines:
            print(f"    TEST cosine (UNSEEN fillers): {np.mean(test_cosines):.4f} ± {np.std(test_cosines):.4f}")
            # KEY: if test ≈ train → operator generalizes → binding!
            # if test << train → operator is filler-specific → no binding
        else:
            print(f"    TEST cosine: N/A")
        
        print(f"\n  Method 2: Mean Role Shift Direction")
        print(f"    Role shift magnitude: {d_role.norm().item():.2f}")
        print(f"    Train shift cosine: {np.mean(train_shift_cosines):.4f}")
        if test_shift_cosines:
            print(f"    TEST shift cosine: {np.mean(test_shift_cosines):.4f}")
        
        print(f"\n  Method 3: Cross-Filler Consistency (STRONGEST TEST)")
        print(f"    Role shift cross-cosine (train): {np.mean(cross_cosines):.4f} ± {np.std(cross_cosines):.4f}")
        if test_train_cross:
            print(f"    Role shift cross-cosine (test<->train): {np.mean(test_train_cross):.4f}")
        
        # ---- KEY DIAGNOSTIC ----
        print(f"\n  ★ DIAGNOSTIC: Is this TRUE BINDING or POSITION ARTIFACT?")
        if test_cosines:
            train_mean = np.mean(train_cosines)
            test_mean = np.mean(test_cosines)
            gap = train_mean - test_mean
            print(f"    Train-Test gap: {gap:.4f}")
            if gap < 0.05:
                print(f"    → T GENERALIZES to unseen fillers! STRONG binding evidence!")
            elif gap < 0.15:
                print(f"    → T partially generalizes. Moderate binding evidence.")
            else:
                print(f"    → T does NOT generalize. Likely position artifact, not binding.")
        
        if cross_cosines:
            cc_mean = np.mean(cross_cosines)
            if cc_mean > 0.7:
                print(f"    Cross-filler consistency: {cc_mean:.4f} → ROLE SHIFT is FILLER-INDEPENDENT!")
            elif cc_mean > 0.3:
                print(f"    Cross-filler consistency: {cc_mean:.4f} → Partial filler-independence")
            else:
                print(f"    Cross-filler consistency: {cc_mean:.4f} → Role shift is FILLER-DEPENDENT. Not binding.")


# ============================================================
# EXPERIMENT B: Cross-Template Invariance
# ============================================================
def experiment_b(model):
    """
    Test if relation subspace is the SAME across different sentence templates.
    
    If "capital" relation subspace is the same in:
      - "The capital of France is"
      - "France's capital is"
      - "France has capital"
    → relation is ABSTRACT, not template-bound
    
    If relation subspace differs across templates
    → relation is just template statistics
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT B: Cross-Template Invariance")
    print("=" * 70)
    print("\nKey question: Is the 'capital' relation subspace the SAME")
    print("across different sentence templates?")
    print("If YES → relation is ABSTRACT")
    print("If NO  → relation is just template statistics")
    
    entities = ["France", "Germany", "Japan", "Italy", "Spain", "Brazil", "India", "Egypt"]
    
    # Multiple templates for the SAME relation (capital)
    templates = [
        "The capital of {entity} is",
        "{entity}'s capital is",
        "What is the capital of {entity}",
        "{entity} has capital",
    ]
    
    # Also use a DIFFERENT relation for comparison
    currency_templates = [
        "The currency of {entity} is",
        "{entity}'s currency is",
        "What is the currency of {entity}",
        "{entity} uses currency",
    ]
    
    layers_to_check = [4, 6, 8, 10]
    
    print("\nCollecting representations across templates...")
    
    # Collect: for each template, for each entity, get h at last position
    capital_reps = {}  # (layer, template_idx, entity) -> h
    currency_reps = {}  # (layer, template_idx, entity) -> h
    
    for layer in layers_to_check:
        for t_idx, template in enumerate(templates):
            for entity in entities:
                prompt = template.format(entity=entity)
                tokens = model.to_tokens(prompt)
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
                h = cache[f'blocks.{layer}.hook_resid_post'][-1].detach().cpu()
                capital_reps[(layer, t_idx, entity)] = h
        
        for t_idx, template in enumerate(currency_templates):
            for entity in entities[:4]:  # Fewer for speed
                prompt = template.format(entity=entity)
                tokens = model.to_tokens(prompt)
                with torch.no_grad():
                    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
                h = cache[f'blocks.{layer}.hook_resid_post'][-1].detach().cpu()
                currency_reps[(layer, t_idx, entity)] = h
    
    print("\nAnalyzing cross-template invariance...")
    
    for layer in layers_to_check:
        print(f"\n--- Layer {layer} ---")
        
        # For each template pair, compute:
        # 1. Entity variation direction (vary entity, fix template)
        # 2. Cross-template consistency: same entity, different templates
        
        # Step 1: Entity variation per template
        entity_dirs_per_template = {}
        for t_idx in range(len(templates)):
            diffs = []
            for i, e1 in enumerate(entities):
                for j, e2 in enumerate(entities):
                    if j > i:
                        h1 = capital_reps[(layer, t_idx, e1)]
                        h2 = capital_reps[(layer, t_idx, e2)]
                        diffs.append(h1 - h2)
            entity_dirs_per_template[t_idx] = torch.stack(diffs)
        
        # Step 2: Cross-template similarity of entity variation
        # If relation subspace is template-invariant, then entity variation
        # should be similar across templates
        print("\n  Cross-template similarity of entity variation:")
        for i in range(len(templates)):
            for j in range(i+1, len(templates)):
                # Compute subspace overlap between entity variations in template i vs j
                diffs_i = entity_dirs_per_template[i]
                diffs_j = entity_dirs_per_template[j]
                
                # Average cosine between all pairs
                cosines = []
                min_len = min(len(diffs_i), len(diffs_j))
                for k in range(min_len):
                    cos = F.cosine_similarity(diffs_i[k], diffs_j[k], dim=0).item()
                    cosines.append(cos)
                
                # Also: project template_j entity diffs onto template_i subspace
                # SVD of template_i diffs
                _, S_i, Vh_i = torch.linalg.svd(diffs_i - diffs_i.mean(0), full_matrices=False)
                k = min(5, S_i.shape[0])
                V_i_top = Vh_i[:k, :].T  # [d, k]
                
                diffs_j_centered = diffs_j - diffs_j.mean(0)
                proj_j_on_i = diffs_j_centered @ V_i_top @ V_i_top.T
                var_explained = (proj_j_on_i ** 2).sum() / (diffs_j_centered ** 2).sum()
                
                print(f"    Template {i}<->{j}: mean cosine={np.mean(cosines):.4f}, "
                      f"subspace var explained={var_explained:.4f}")
        
        # Step 3: Same entity across different templates
        # If relation is abstract, same entity should have SIMILAR representation
        # across templates (modulo template-specific variation)
        print("\n  Same-entity cross-template consistency:")
        for e_idx, entity in enumerate(entities[:4]):
            reps_across_templates = []
            for t_idx in range(len(templates)):
                reps_across_templates.append(capital_reps[(layer, t_idx, entity)])
            
            # Pairwise cosine
            cross_template_cosines = []
            for i in range(len(reps_across_templates)):
                for j in range(i+1, len(reps_across_templates)):
                    cos = F.cosine_similarity(
                        reps_across_templates[i], 
                        reps_across_templates[j], 
                        dim=0
                    ).item()
                    cross_template_cosines.append(cos)
            
            print(f"    {entity}: mean cross-template cosine = {np.mean(cross_template_cosines):.4f}")
        
        # Step 4: CRITICAL - Capital vs Currency template comparison
        # Are capital templates more similar to each other than to currency templates?
        print("\n  ★ Capital vs Currency template comparison:")
        
        # Get capital entity directions (averaged across templates)
        capital_diffs_all = []
        for t_idx in range(len(templates)):
            for i, e1 in enumerate(entities[:4]):
                for j, e2 in enumerate(entities[:4]):
                    if j > i:
                        h1 = capital_reps[(layer, t_idx, e1)]
                        h2 = capital_reps[(layer, t_idx, e2)]
                        capital_diffs_all.append(h1 - h2)
        capital_diffs = torch.stack(capital_diffs_all)
        
        currency_diffs_all = []
        for t_idx in range(len(currency_templates)):
            for i, e1 in enumerate(entities[:4]):
                for j, e2 in enumerate(entities[:4]):
                    if j > i:
                        h1 = currency_reps[(layer, t_idx, e1)]
                        h2 = currency_reps[(layer, t_idx, e2)]
                        currency_diffs_all.append(h1 - h2)
        currency_diffs = torch.stack(currency_diffs_all)
        
        # Subspace analysis
        _, S_cap, Vh_cap = torch.linalg.svd(capital_diffs - capital_diffs.mean(0), full_matrices=False)
        _, S_cur, Vh_cur = torch.linalg.svd(currency_diffs - currency_diffs.mean(0), full_matrices=False)
        
        k = min(5, S_cap.shape[0], S_cur.shape[0])
        V_cap_top = Vh_cap[:k, :].T
        V_cur_top = Vh_cur[:k, :].T
        
        # Cross-relation subspace overlap
        overlap = V_cap_top.T @ V_cur_top
        overlap_norm = torch.linalg.norm(overlap, 'fro').item() / np.sqrt(k)
        
        # Variance explained
        cur_in_cap = currency_diffs - currency_diffs.mean(0)
        proj_cur_on_cap = cur_in_cap @ V_cap_top @ V_cap_top.T
        var_cur_in_cap = (proj_cur_on_cap ** 2).sum() / (cur_in_cap ** 2).sum()
        
        cap_in_cur = capital_diffs - capital_diffs.mean(0)
        proj_cap_on_cur = cap_in_cur @ V_cur_top @ V_cur_top.T
        var_cap_in_cur = (proj_cap_on_cur ** 2).sum() / (cap_in_cur ** 2).sum()
        
        print(f"    Capital-Currency subspace overlap: {overlap_norm:.4f}")
        print(f"    Currency var explained by Capital subspace: {var_cur_in_cap:.4f}")
        print(f"    Capital var explained by Currency subspace: {var_cap_in_cur:.4f}")
        
        if overlap_norm < 0.3:
            print(f"    → Capital and Currency use DIFFERENT subspaces → relation-specific!")
        else:
            print(f"    → Capital and Currency share subspaces → entity-driven, not relation-driven")


# ============================================================
# EXPERIMENT C: Compositional Generalization
# ============================================================
def experiment_c(model):
    """
    Test if entity_A direction + relation_B direction → relation_B(entity_A) output.
    
    Specific test:
      d_entity = h("France") - h("Germany") (from capital context)
      Add d_entity to "The currency of Germany is"
      → Does output shift toward "euro" (currency of France)?
    
    If YES → entity and relation are truly composable
    If NO  → entity direction is relation-specific, not abstract
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT C: Compositional Generalization")
    print("=" * 70)
    print("\nKey question: Can entity direction from one relation context")
    print("transfer to a DIFFERENT relation context?")
    print("If YES → entity is an abstract variable")
    print("If NO  → entity direction is relation-specific")
    
    # Relations and their expected answers
    relations = {
        "capital": {
            "France": "Paris", "Germany": "Berlin", "Japan": "Tokyo", "Italy": "Rome",
        },
        "currency": {
            "France": "euro", "Germany": "euro", "Japan": "yen", "Italy": "euro",
        },
    }
    
    layers_to_check = [4, 6, 8, 10]
    
    # Step 1: Get entity directions from each relation context
    print("\nComputing entity directions from different relation contexts...")
    
    entity_directions = {}  # (layer, source_relation) -> d_entity (France - Germany)
    
    for rel_name, entity_answers in relations.items():
        for layer in layers_to_check:
            # "The {relation} of France is"
            prompt_fr = f"The {rel_name} of France is"
            prompt_de = f"The {rel_name} of Germany is"
            
            tokens_fr = model.to_tokens(prompt_fr)
            tokens_de = model.to_tokens(prompt_de)
            
            with torch.no_grad():
                _, cache_fr = model.run_with_cache(tokens_fr, remove_batch_dim=True)
                _, cache_de = model.run_with_cache(tokens_de, remove_batch_dim=True)
            
            h_fr = cache_fr[f'blocks.{layer}.hook_resid_post'][-1].detach().cpu()
            h_de = cache_de[f'blocks.{layer}.hook_resid_post'][-1].detach().cpu()
            
            d = h_fr - h_de
            entity_directions[(layer, rel_name)] = d / d.norm()
    
    # Step 2: Cross-relation comparison of entity directions
    print("\nCross-relation comparison of entity directions (France - Germany):")
    for layer in layers_to_check:
        d_cap = entity_directions[(layer, "capital")]
        d_cur = entity_directions[(layer, "currency")]
        cos = F.cosine_similarity(d_cap, d_cur, dim=0).item()
        print(f"  Layer {layer}: capital_entity_dir <-> currency_entity_dir cosine = {cos:.4f}")
    
    # Step 3: Cross-relation intervention
    print("\nCross-relation intervention:")
    print("  Adding capital-context entity direction to currency prompt...")
    
    # Get baseline probabilities for currency prompts
    for layer in [6, 8, 10]:
        prompt_de_cur = f"The currency of Germany is"
        tokens_de_cur = model.to_tokens(prompt_de_cur)
        
        with torch.no_grad():
            logits_baseline = model(tokens_de_cur)[0, -1]
            probs_baseline = torch.softmax(logits_baseline, dim=-1)
        
        # Try to find euro and yen tokens
        euro_id = safe_token_id(model, "euro")
        yen_id = safe_token_id(model, "yen")
        
        p_euro_base = probs_baseline[euro_id].item()
        p_yen_base = probs_baseline[yen_id].item()
        
        print(f"\n  Layer {layer}: 'The currency of Germany is'")
        print(f"    Baseline: P(euro)={p_euro_base:.4f}, P(yen)={p_yen_base:.4f}")
        
        # Intervention: add capital-context entity direction
        d_cap_entity = entity_directions[(layer, "capital")].cuda()
        d_cur_entity = entity_directions[(layer, "currency")].cuda()
        
        for alpha in [5.0, 10.0, 20.0]:
            # Using CAPITAL entity direction on CURRENCY prompt
            def hook_capital_entity(resid, hook):
                resid[0, -1, :] += alpha * d_cap_entity
                return resid
            
            with torch.no_grad():
                logits_cap = model.run_with_hooks(
                    tokens_de_cur,
                    fwd_hooks=[(f'blocks.{layer}.hook_resid_post', hook_capital_entity)],
                )[0, -1]
                probs_cap = torch.softmax(logits_cap, dim=-1)
            
            p_euro_cap = probs_cap[euro_id].item()
            p_yen_cap = probs_cap[yen_id].item()
            
            ratio_euro_cap = p_euro_cap / p_euro_base if p_euro_base > 1e-8 else float('inf')
            
            # Using CURRENCY entity direction on CURRENCY prompt (within-relation control)
            def hook_currency_entity(resid, hook):
                resid[0, -1, :] += alpha * d_cur_entity
                return resid
            
            with torch.no_grad():
                logits_cur = model.run_with_hooks(
                    tokens_de_cur,
                    fwd_hooks=[(f'blocks.{layer}.hook_resid_post', hook_currency_entity)],
                )[0, -1]
                probs_cur = torch.softmax(logits_cur, dim=-1)
            
            p_euro_cur = probs_cur[euro_id].item()
            ratio_euro_cur = p_euro_cur / p_euro_base if p_euro_base > 1e-8 else float('inf')
            
            print(f"    alpha={alpha:.0f}: "
                  f"Cross-relation: P(euro)={p_euro_cap:.4f} ({ratio_euro_cap:.2f}x), "
                  f"Within-relation: P(euro)={p_euro_cur:.4f} ({ratio_euro_cur:.2f}x)")
        
        # KEY DIAGNOSTIC
        if ratio_euro_cap > 1.5 and ratio_euro_cap / max(ratio_euro_cur, 0.01) > 0.3:
            print(f"    → Entity direction PARTIALLY transfers across relations!")
        elif ratio_euro_cap > 1.5:
            print(f"    → Entity direction transfers, but within-relation is stronger.")
        else:
            print(f"    → Entity direction does NOT transfer → entity is relation-specific!")


# ============================================================
# EXPERIMENT D: Head-Level Binding Circuit
# ============================================================
def experiment_d(model):
    """
    Which attention heads carry role information?
    Which MLP layers perform the binding?
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT D: Head-Level Binding Circuit")
    print("=" * 70)
    print("\nKey question: Which specific heads/MLPs encode role vs filler?")
    
    # Use a simple template and look at individual head contributions
    verbs = ["loves", "hates"]
    fillers = ["Paris", "Rome", "Berlin", "London", "Tokyo"]
    
    anchor = "Rome"
    layers_to_check = [2, 4, 6, 8]
    n_heads = model.cfg.n_heads
    
    print(f"\nModel has {n_heads} attention heads per layer")
    
    for layer in layers_to_check:
        print(f"\n--- Layer {layer} ---")
        
        # Collect per-head output for subject vs object role
        head_outputs_subj = {h: [] for h in range(n_heads)}
        head_outputs_obj = {h: [] for h in range(n_heads)}
        
        for verb in verbs:
            for filler in fillers:
                if filler == anchor:
                    continue
                
                # Subject: "Filler {verb} Rome"
                prompt_subj = f"{filler} {verb} {anchor}"
                tokens_subj = model.to_tokens(prompt_subj)
                
                # Object: "Rome {verb} Filler"
                prompt_obj = f"{anchor} {verb} {filler}"
                tokens_obj = model.to_tokens(prompt_obj)
                
                with torch.no_grad():
                    _, cache_subj = model.run_with_cache(tokens_subj, remove_batch_dim=True)
                    _, cache_obj = model.run_with_cache(tokens_obj, remove_batch_dim=True)
                
                for h in range(n_heads):
                    # Get head output: z @ W_O
                    z_subj = cache_subj[f'blocks.{layer}.attn.hook_z'][-1, h, :].detach().cpu()
                    z_obj = cache_obj[f'blocks.{layer}.attn.hook_z'][-1, h, :].detach().cpu()
                    
                    head_outputs_subj[h].append(z_subj)
                    head_outputs_obj[h].append(z_obj)
        
        # Analyze per-head: which heads differ most between subject vs object?
        head_role_sensitivity = {}
        for h in range(n_heads):
            z_subj = torch.stack(head_outputs_subj[h])
            z_obj = torch.stack(head_outputs_obj[h])
            
            # Role sensitivity: how much does this head's output change with role?
            diffs = z_obj - z_subj  # [N, d_head]
            avg_diff = diffs.norm(dim=1).mean().item()
            
            # Also: cross-filler consistency of role signal
            # For TRUE role binding, the role shift should be similar across fillers
            if len(diffs) > 1:
                cross_cosines = []
                for i in range(len(diffs)):
                    for j in range(i+1, len(diffs)):
                        cos = F.cosine_similarity(diffs[i], diffs[j], dim=0).item()
                        cross_cosines.append(cos)
                consistency = np.mean(cross_cosines)
            else:
                consistency = 0.0
            
            head_role_sensitivity[h] = (avg_diff, consistency)
        
        # Sort by role sensitivity
        sorted_heads = sorted(head_role_sensitivity.items(), 
                            key=lambda x: x[1][0], reverse=True)
        
        print(f"  Top-5 heads by role sensitivity:")
        for h, (sens, cons) in sorted_heads[:5]:
            print(f"    Head {h}: role_sensitivity={sens:.2f}, cross_filler_consistency={cons:.4f}")
            if cons > 0.5:
                print(f"      → FILLER-INDEPENDENT role signal! Binding candidate!")
            elif cons > 0.2:
                print(f"      → Partial filler-independence")
            else:
                print(f"      → Filler-dependent. Not pure role signal.")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, 
                       choices=["a", "b", "c", "d"], help="Which experiment to run")
    parser.add_argument("--model", type=str, default="gpt2-small",
                       help="Model name (gpt2-small, Qwen/Qwen2.5-1.5B)")
    args = parser.parse_args()
    
    model = get_model(args.model)
    
    if args.exp == "a":
        experiment_a(model)
    elif args.exp == "b":
        experiment_b(model)
    elif args.exp == "c":
        experiment_c(model)
    elif args.exp == "d":
        experiment_d(model)
