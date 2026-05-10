"""
Phase 87+ Qwen3: Cross-Relation Compositional Generalization Test
==================================================================
The CRITICAL test on a model with reliable factual knowledge.

Key questions:
1. Does entity direction transfer across relations? (compositional generalization)
2. Is Capital-Currency subspace separation real on a better model?
3. Does the role transfer operator generalize on Qwen3?
"""
import torch
import torch.nn.functional as F
import numpy as np

def main():
    print("=" * 70)
    print("Phase 87+ Qwen3: Cross-Relation & Binding Tests")
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
    print("Qwen3-1.5B loaded.")

    def safe_token_id(text):
        try:
            return model.to_single_token(text)
        except AssertionError:
            tokens = model.to_tokens(text)
            return tokens[0, -1].item()

    # Step 1: Verify model knows the facts
    print("\n--- Step 1: Verify Factual Knowledge ---")
    test_prompts = [
        ("The capital of France is", "Paris"),
        ("The capital of Germany is", "Berlin"),
        ("The capital of Japan is", "Tokyo"),
        ("The currency of France is", "euro"),
        ("The currency of Japan is", "yen"),
        ("The language of France is", "French"),
        ("The language of Germany is", "German"),
    ]
    for prompt, expected in test_prompts:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            logits = model(tokens)[0, -1]
            top_5 = torch.topk(logits, 5)
            top_strs = [model.to_string(t.item()).strip() for t in top_5.indices]
            top_probs = torch.softmax(logits, dim=-1)[top_5.indices]
        expected_id = safe_token_id(expected)
        p_expected = torch.softmax(logits, dim=-1)[expected_id].item()
        print(f"  '{prompt}' -> top5={top_strs}, P({expected})={p_expected:.4f}")

    # Step 2: Cross-relation entity direction comparison
    print("\n--- Step 2: Cross-Relation Entity Direction ---")
    entities = ["France", "Germany", "Japan", "Italy", "Spain", "Brazil", "India", "Egypt"]
    relations = ["capital", "currency", "language"]
    layers = [8, 12, 16, 20]

    entity_dirs = {}  # (layer, relation) -> d_entity (mean across entity pairs)
    
    for layer in layers:
        for rel in relations:
            # Compute entity variation direction: France - Germany
            prompt_fr = f"The {rel} of France is"
            prompt_de = f"The {rel} of Germany is"
            tokens_fr = model.to_tokens(prompt_fr)
            tokens_de = model.to_tokens(prompt_de)
            with torch.no_grad():
                _, cache_fr = model.run_with_cache(tokens_fr, remove_batch_dim=True)
                _, cache_de = model.run_with_cache(tokens_de, remove_batch_dim=True)
            h_fr = cache_fr[f'blocks.{layer}.hook_resid_post'][-1].detach()
            h_de = cache_de[f'blocks.{layer}.hook_resid_post'][-1].detach()
            d = h_fr - h_de
            entity_dirs[(layer, rel)] = d / d.norm()

    print("\nCross-relation entity direction similarity (France-Germany):")
    for layer in layers:
        cos_cap_cur = F.cosine_similarity(entity_dirs[(layer, 'capital')], 
                                          entity_dirs[(layer, 'currency')], dim=0).item()
        cos_cap_lang = F.cosine_similarity(entity_dirs[(layer, 'capital')], 
                                           entity_dirs[(layer, 'language')], dim=0).item()
        cos_cur_lang = F.cosine_similarity(entity_dirs[(layer, 'currency')], 
                                           entity_dirs[(layer, 'language')], dim=0).item()
        print(f"  Layer {layer}: cap<->cur={cos_cap_cur:.4f}, cap<->lang={cos_cap_lang:.4f}, cur<->lang={cos_cur_lang:.4f}")

    # Step 3: Capital vs Currency subspace overlap (the critical test)
    print("\n--- Step 3: Capital vs Currency Subspace Overlap ---")
    
    for layer in layers:
        # Collect entity variation for capital
        capital_diffs = []
        for i, e1 in enumerate(entities):
            for j, e2 in enumerate(entities):
                if j > i:
                    h1_key = (layer, 'capital', e1)
                    h2_key = (layer, 'capital', e2)
                    # Need to compute these
                    prompt1 = f"The capital of {e1} is"
                    prompt2 = f"The capital of {e2} is"
                    t1 = model.to_tokens(prompt1)
                    t2 = model.to_tokens(prompt2)
                    with torch.no_grad():
                        _, c1 = model.run_with_cache(t1, remove_batch_dim=True)
                        _, c2 = model.run_with_cache(t2, remove_batch_dim=True)
                    h1 = c1[f'blocks.{layer}.hook_resid_post'][-1].detach()
                    h2 = c2[f'blocks.{layer}.hook_resid_post'][-1].detach()
                    capital_diffs.append(h1 - h2)
        
        currency_diffs = []
        for i, e1 in enumerate(entities[:4]):  # Fewer for speed
            for j, e2 in enumerate(entities[:4]):
                if j > i:
                    prompt1 = f"The currency of {e1} is"
                    prompt2 = f"The currency of {e2} is"
                    t1 = model.to_tokens(prompt1)
                    t2 = model.to_tokens(prompt2)
                    with torch.no_grad():
                        _, c1 = model.run_with_cache(t1, remove_batch_dim=True)
                        _, c2 = model.run_with_cache(t2, remove_batch_dim=True)
                    h1 = c1[f'blocks.{layer}.hook_resid_post'][-1].detach()
                    h2 = c2[f'blocks.{layer}.hook_resid_post'][-1].detach()
                    currency_diffs.append(h1 - h2)
        
        cap_d = torch.stack(capital_diffs)
        cur_d = torch.stack(currency_diffs)
        
        _, S_cap, Vh_cap = torch.linalg.svd(cap_d - cap_d.mean(0), full_matrices=False)
        _, S_cur, Vh_cur = torch.linalg.svd(cur_d - cur_d.mean(0), full_matrices=False)
        
        k = min(5, S_cap.shape[0], S_cur.shape[0])
        V_cap = Vh_cap[:k, :].T
        V_cur = Vh_cur[:k, :].T
        
        overlap = V_cap.T @ V_cur
        overlap_norm = torch.linalg.norm(overlap, 'fro').item() / np.sqrt(k)
        
        # Variance explained
        cur_centered = cur_d - cur_d.mean(0)
        proj = cur_centered @ V_cap @ V_cap.T
        var_cur_in_cap = (proj ** 2).sum() / (cur_centered ** 2).sum()
        
        cap_centered = cap_d - cap_d.mean(0)
        proj2 = cap_centered @ V_cur @ V_cur.T
        var_cap_in_cur = (proj2 ** 2).sum() / (cap_centered ** 2).sum()
        
        print(f"  Layer {layer}: overlap={overlap_norm:.4f}, cur_in_cap={var_cur_in_cap:.4f}, cap_in_cur={var_cap_in_cur:.4f}")

    # Step 4: Cross-relation intervention
    print("\n--- Step 4: Cross-Relation Entity Intervention ---")
    
    euro_id = safe_token_id("euro")
    yen_id = safe_token_id("yen")
    
    for layer in [12, 16, 20]:
        prompt_de_cur = f"The currency of Germany is"
        tokens_de_cur = model.to_tokens(prompt_de_cur)
        
        with torch.no_grad():
            logits_base = model(tokens_de_cur)[0, -1]
            probs_base = torch.softmax(logits_base, dim=-1)
        
        p_euro_base = probs_base[euro_id].item()
        p_yen_base = probs_base[yen_id].item()
        
        print(f"\n  Layer {layer}: 'The currency of Germany is'")
        print(f"    Baseline: P(euro)={p_euro_base:.4f}, P(yen)={p_yen_base:.4f}")
        
        # Get entity directions from different relation contexts
        d_cap_entity = entity_dirs[(layer, 'capital')].cuda()
        d_cur_entity = entity_dirs[(layer, 'currency')].cuda()
        
        # Also get from language context
        d_lang_entity = entity_dirs[(layer, 'language')].cuda()
        
        for alpha in [5.0, 10.0, 20.0]:
            # Using CAPITAL entity direction on CURRENCY prompt
            def hook_cap(resid, hook):
                resid[0, -1, :] += alpha * d_cap_entity
                return resid
            
            with torch.no_grad():
                logits_cap = model.run_with_hooks(
                    tokens_de_cur,
                    fwd_hooks=[(f'blocks.{layer}.hook_resid_post', hook_cap)],
                )[0, -1]
                probs_cap = torch.softmax(logits_cap, dim=-1)
            
            p_euro_cap = probs_cap[euro_id].item()
            ratio_cap = p_euro_cap / p_euro_base if p_euro_base > 1e-8 else 0
            
            # Using CURRENCY entity direction (within-relation control)
            def hook_cur(resid, hook):
                resid[0, -1, :] += alpha * d_cur_entity
                return resid
            
            with torch.no_grad():
                logits_cur = model.run_with_hooks(
                    tokens_de_cur,
                    fwd_hooks=[(f'blocks.{layer}.hook_resid_post', hook_cur)],
                )[0, -1]
                probs_cur = torch.softmax(logits_cur, dim=-1)
            
            p_euro_cur = probs_cur[euro_id].item()
            ratio_cur = p_euro_cur / p_euro_base if p_euro_base > 1e-8 else 0
            
            # Using LANGUAGE entity direction
            def hook_lang(resid, hook):
                resid[0, -1, :] += alpha * d_lang_entity
                return resid
            
            with torch.no_grad():
                logits_lang = model.run_with_hooks(
                    tokens_de_cur,
                    fwd_hooks=[(f'blocks.{layer}.hook_resid_post', hook_lang)],
                )[0, -1]
                probs_lang = torch.softmax(logits_lang, dim=-1)
            
            p_euro_lang = probs_lang[euro_id].item()
            ratio_lang = p_euro_lang / p_euro_base if p_euro_base > 1e-8 else 0
            
            print(f"    alpha={alpha:.0f}: Cross(cap->cur): P(euro)={p_euro_cap:.4f} ({ratio_cap:.2f}x), "
                  f"Within(cur): P(euro)={p_euro_cur:.4f} ({ratio_cur:.2f}x), "
                  f"Cross(lang->cur): P(euro)={p_euro_lang:.4f} ({ratio_lang:.2f}x)")
        
        # KEY DIAGNOSTIC
        print(f"\n    ★ If cross-relation ratio > 0.5 * within-relation ratio:")
        print(f"      Entity direction is PARTIALLY abstract (composable)")
        print(f"    ★ If cross-relation ratio << within-relation ratio:")
        print(f"      Entity direction is relation-specific (NOT composable)")

    # Step 5: Role Transfer Operator on Qwen3
    print("\n--- Step 5: Role Transfer Operator (Qwen3) ---")
    
    anchor = "Rome"
    train_fillers = ["Paris", "Berlin", "London", "Tokyo", "Madrid", "Oslo", "Seoul"]
    test_fillers = ["Alice", "Bob", "Carol", "Dave"]
    verbs = ["loves", "hates"]
    
    for layer in [8, 12, 16, 20]:
        train_h_subj = []
        train_h_obj = []
        test_h_subj = []
        test_h_obj = []
        
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
                h_subj = c_subj[f'blocks.{layer}.hook_resid_post'][-1].detach().cpu()
                h_obj = c_obj[f'blocks.{layer}.hook_resid_post'][-1].detach().cpu()
                
                if is_test:
                    test_h_subj.append(h_subj)
                    test_h_obj.append(h_obj)
                else:
                    train_h_subj.append(h_subj)
                    train_h_obj.append(h_obj)
        
        # Average across verbs
        train_h_subj_avg = torch.stack([train_h_subj[i] for i in range(0, len(train_h_subj), 2)]).mean(0) if len(train_h_subj) > 2 else torch.stack(train_h_subj).mean(0)
        train_h_obj_avg = torch.stack([train_h_obj[i] for i in range(0, len(train_h_obj), 2)]).mean(0) if len(train_h_obj) > 2 else torch.stack(train_h_obj).mean(0)
        
        # Actually, keep individual pairs for better statistics
        # Pair subj/obj for same filler-verb
        train_pairs_s = []
        train_pairs_o = []
        for i in range(0, len(train_h_subj), 2):
            train_pairs_s.append((train_h_subj[i] + train_h_subj[i+1]) / 2)
            train_pairs_o.append((train_h_obj[i] + train_h_obj[i+1]) / 2)
        
        train_h_s = torch.stack(train_pairs_s)
        train_h_o = torch.stack(train_pairs_o)
        
        test_pairs_s = []
        test_pairs_o = []
        for i in range(0, len(test_h_subj), 2):
            test_pairs_s.append((test_h_subj[i] + test_h_subj[i+1]) / 2)
            test_pairs_o.append((test_h_obj[i] + test_h_obj[i+1]) / 2)
        
        test_h_s = torch.stack(test_pairs_s) if test_pairs_s else None
        test_h_o = torch.stack(test_pairs_o) if test_pairs_o else None
        
        # Ridge regression T
        H_s = train_h_s.numpy()
        H_o = train_h_o.numpy()
        lam = 1.0
        I_d = np.eye(H_s.shape[1])
        T = np.linalg.solve(H_s.T @ H_s + lam * I_d, H_s.T @ H_o)
        
        # Train evaluation
        pred_train = H_s @ T
        train_cos = [F.cosine_similarity(torch.tensor(pred_train[i]), torch.tensor(H_o[i]), dim=0).item()
                     for i in range(len(H_s))]
        
        # Test evaluation
        if test_h_s is not None:
            H_s_test = test_h_s.numpy()
            H_o_test = test_h_o.numpy()
            pred_test = H_s_test @ T
            test_cos = [F.cosine_similarity(torch.tensor(pred_test[i]), torch.tensor(H_o_test[i]), dim=0).item()
                       for i in range(len(H_s_test))]
        else:
            test_cos = []
        
        # Cross-filler consistency
        role_shifts = train_h_o - train_h_s
        cross_cos = []
        for i in range(len(role_shifts)):
            for j in range(i+1, len(role_shifts)):
                cross_cos.append(F.cosine_similarity(role_shifts[i], role_shifts[j], dim=0).item())
        
        gap = np.mean(train_cos) - (np.mean(test_cos) if test_cos else 0)
        print(f"\n  Layer {layer}:")
        print(f"    Train cosine: {np.mean(train_cos):.4f}")
        if test_cos:
            print(f"    Test cosine (UNSEEN): {np.mean(test_cos):.4f}")
        print(f"    Train-Test gap: {gap:.4f}")
        print(f"    Cross-filler consistency: {np.mean(cross_cos):.4f}")
        
        if gap < 0.05:
            print(f"    -> T GENERALIZES! Strong binding evidence on Qwen3!")
        elif gap < 0.15:
            print(f"    -> T partially generalizes. Moderate binding evidence.")
        else:
            print(f"    -> T does NOT generalize. Position artifact.")

    print("\n" + "=" * 70)
    print("Phase 87+ Qwen3 experiment complete.")

if __name__ == "__main__":
    main()
