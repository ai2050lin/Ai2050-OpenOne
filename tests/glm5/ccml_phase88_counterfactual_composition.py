"""
Phase 88: Counterfactual Compositional Generalization
=====================================================

THE CORE QUESTION (from Phase 87 critique):
  "Linear generalization ≠ Binding. A linear T that transfers subject→object 
   representations could just be capturing local manifold structure."

THE DECISIVE TEST:
  Can we predict how an entity will be represented in a NEW relation,
  based on how it's represented in KNOWN relations + how OTHER entities 
  are represented in that new relation?

  If Transformer has TRUE compositional binding:
    h(entity_X | relation_R) = compose(h_entity(X), h_relation(R))
    → Can predict unseen entity-relation combinations

  If Transformer has only CONDITIONAL GEOMETRY:
    h(entity_X | relation_R) = f(X, R) where f is entangled
    → Cannot predict unseen combinations from separate entity/relation info

THREE EXPERIMENTS:

A. Cross-Relation Entity Transport ★★★★★ (MOST CRITICAL)
   - Learn entity direction in relation_A (e.g., capital)
   - Apply to predict entity representation in relation_B (e.g., currency)
   - If entity is an ABSTRACT VARIABLE → should work
   - If entity is CONTEXT-CONDITIONED → should fail
   
   KEY CONTROL: Compare against baseline of just using the same-relation 
   entity direction. If cross-relation transport is NO BETTER than same-relation
   baseline minus the advantage of same-relation, then entity is not abstract.

B. Relation-Entity Composition Test ★★★★★
   - For entity X seen in relations {R1, R2, R3} but NOT R4:
     Can we predict h(X | R4) from h(X | R1..R3) + h(Y | R4) for other Y?
   - Train: predict h(X | R_new) from h(X | R_known) + h(others | R_new)
   - Test: on held-out entity-relation pairs
   - If composition works → systematic compositional representation
   - If composition fails → contextual geometry, not symbolic variables

C. Surface Form Control ★★★★★ (CRITICAL CONTROL)
   - Test whether "binding" effects survive after controlling for:
     * Token position (subject at pos 2 vs pos 4)
     * Syntax structure (SVO vs OVS order)
     * Template frequency (common vs rare phrasings)
   - Use scrambled templates: "Paris capital France the of is"
   - If binding survives scrambling → genuine semantic role
   - If binding depends on canonical order → statistical template effect

D. Geometric Structure of Context-Conditioning ★★★★
   - Map how entity representations change across relations
   - Is there STRUCTURE in the context-conditioning?
   - Specifically: does h(X|R1) - h(X|R2) correlate with h(Y|R1) - h(Y|R2)?
   - If yes → systematic context-conditioning (not random warping)
   - This would reveal the GEOMETRIC STRUCTURE of "conditional geometry"
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import sys
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine

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
    except:
        return None

# ============================================================
# DATA: Multi-relation entities with known answers
# ============================================================

# Relations and their templates
RELATIONS = {
    "capital": [
        "The capital of {entity} is",
        "{entity}'s capital is", 
    ],
    "currency": [
        "The currency of {entity} is",
        "{entity}'s currency is",
    ],
    "language": [
        "The language of {entity} is",
        "{entity}'s language is",
    ],
}

# Entities with known answers across relations
# We need entities where the model likely knows answers for multiple relations
ENTITIES = {
    "France":   {"capital": "Paris",    "currency": "euro",    "language": "French"},
    "Germany":  {"capital": "Berlin",   "currency": "euro",    "language": "German"},
    "Japan":    {"capital": "Tokyo",    "currency": "yen",     "language": "Japanese"},
    "Italy":    {"capital": "Rome",     "currency": "euro",    "language": "Italian"},
    "Spain":    {"capital": "Madrid",   "currency": "euro",    "language": "Spanish"},
    "China":    {"capital": "Beijing",  "currency": "yuan",    "language": "Chinese"},
    "Brazil":   {"capital": "Brasilia", "currency": "real",    "language": "Portuguese"},
    "Russia":   {"capital": "Moscow",   "currency": "ruble",   "language": "Russian"},
    "India":    {"capital": "Delhi",    "currency": "rupee",   "language": "Hindi"},
    "England":  {"capital": "London",   "currency": "pound",   "language": "English"},
    "Korea":    {"capital": "Seoul",    "currency": "won",     "language": "Korean"},
    "Mexico":   {"capital": "Mexico City", "currency": "peso", "language": "Spanish"},
    "Egypt":    {"capital": "Cairo",    "currency": "pound",   "language": "Arabic"},
    "Turkey":   {"capital": "Ankara",   "currency": "lira",    "language": "Turkish"},
    "Sweden":   {"capital": "Stockholm", "currency": "krona", "language": "Swedish"},
}

def get_entity_repr(model, entity, relation, template_idx=0, layer=None):
    """Get entity token representation in a given relation context."""
    if layer is None:
        layer = model.cfg.n_layers // 2
    
    template = RELATIONS[relation][template_idx]
    prompt = template.format(entity=entity)
    
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    
    # Find entity token position
    entity_tokens = model.to_tokens(entity, prepend_bos=False)
    # For simplicity, use the last token before " is" 
    # The entity representation at the last token of the entity
    # Actually use the last token of the prompt (position before prediction)
    last_pos = tokens.shape[1] - 1
    
    # Get representation at last position (where prediction happens)
    repr_vec = cache[f'blocks.{layer}.hook_resid_post'][0, last_pos, :].detach().cpu()
    
    return repr_vec

def get_last_token_repr(model, prompt, layer):
    """Get representation at last token position."""
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    last_pos = tokens.shape[1] - 1
    return cache[f'blocks.{layer}.hook_resid_post'][0, last_pos, :].detach().cpu()

def get_logit_for_answer(model, prompt, answer):
    """Get logit for a specific answer token."""
    tokens = model.to_tokens(prompt)
    ans_id = safe_token_id(model, answer)
    if ans_id is None:
        return None
    with torch.no_grad():
        logits = model(tokens)[0, -1, :]
    return logits[ans_id].item()

def get_prob_for_answer(model, prompt, answer):
    """Get probability for a specific answer token."""
    tokens = model.to_tokens(prompt)
    ans_id = safe_token_id(model, answer)
    if ans_id is None:
        return None
    with torch.no_grad():
        logits = model(tokens)[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    return probs[ans_id].item()


# ============================================================
# EXPERIMENT A: Cross-Relation Entity Transport
# ============================================================

def experiment_a(model):
    """
    Test if entity representation in one relation can predict 
    entity representation in another relation.
    
    If entity is ABSTRACT VARIABLE:
      h(France|capital) - h(Germany|capital) ≈ h(France|currency) - h(Germany|currency)
      
    If entity is CONTEXT-CONDITIONED:
      The directions are DIFFERENT across relations.
    """
    print("\n" + "="*70)
    print("EXPERIMENT A: Cross-Relation Entity Transport")
    print("="*70)
    print("\nQuestion: Is entity an abstract variable or context-conditioned?")
    print("  If abstract: same entity direction across relations")
    print("  If conditioned: different entity direction per relation")
    
    entities_list = list(ENTITIES.keys())
    relations_list = list(RELATIONS.keys())
    layers = [2, 4, 6, 8, 10]
    
    results = {}
    
    for layer in layers:
        print(f"\n--- Layer {layer} ---")
        
        # Collect representations
        reprs = {}  # reprs[entity][relation] = vector
        for entity in entities_list:
            reprs[entity] = {}
            for relation in relations_list:
                try:
                    reprs[entity][relation] = get_entity_repr(model, entity, relation, 0, layer)
                except Exception as e:
                    print(f"  Error for {entity}/{relation}: {e}")
                    reprs[entity][relation] = None
        
        # Compute entity directions within each relation
        # entity_dir(relation) = mean(h(X, relation) - h(Y, relation)) for pairs X,Y
        
        # Test 1: Cross-relation entity direction consistency
        print("\n  Test 1: Entity direction consistency across relations")
        print("  (Is h(France|cap) - h(Germany|cap) parallel to h(France|cur) - h(Germany|cur)?)")
        
        cross_relation_cosines = []
        for i, rel1 in enumerate(relations_list):
            for j, rel2 in enumerate(relations_list):
                if i >= j:
                    continue
                # Compute entity direction pairs
                cosines = []
                for ei in range(len(entities_list)):
                    for ej in range(ei+1, len(entities_list)):
                        ent_i, ent_j = entities_list[ei], entities_list[ej]
                        if reprs[ent_i][rel1] is None or reprs[ent_i][rel2] is None:
                            continue
                        if reprs[ent_j][rel1] is None or reprs[ent_j][rel2] is None:
                            continue
                        
                        dir1 = reprs[ent_i][rel1] - reprs[ent_j][rel1]
                        dir2 = reprs[ent_i][rel2] - reprs[ent_j][rel2]
                        
                        if dir1.norm() < 1e-6 or dir2.norm() < 1e-6:
                            continue
                        
                        cos = F.cosine_similarity(dir1.unsqueeze(0), dir2.unsqueeze(0)).item()
                        cosines.append(cos)
                
                if cosines:
                    mean_cos = np.mean(cosines)
                    cross_relation_cosines.extend(cosines)
                    print(f"    {rel1} <-> {rel2}: mean cosine = {mean_cos:.4f} (n={len(cosines)})")
        
        # Test 2: Can we TRANSPORT entity from one relation to another?
        print("\n  Test 2: Entity transport across relations")
        print("  (Learn W: h(X|rel_A) -> h(X|rel_B), test on unseen X)")
        
        for rel_source in relations_list:
            for rel_target in relations_list:
                if rel_source == rel_target:
                    continue
                
                # Use leave-one-out cross-validation
                transport_cosines = []
                
                for test_idx in range(len(entities_list)):
                    train_entities = [e for i, e in enumerate(entities_list) if i != test_idx]
                    test_entity = entities_list[test_idx]
                    
                    # Get train representations
                    X_train = []
                    Y_train = []
                    for ent in train_entities:
                        if reprs[ent][rel_source] is None or reprs[ent][rel_target] is None:
                            continue
                        X_train.append(reprs[ent][rel_source].numpy())
                        Y_train.append(reprs[ent][rel_target].numpy())
                    
                    if len(X_train) < 3:
                        continue
                    
                    X_train = np.array(X_train)
                    Y_train = np.array(Y_train)
                    
                    # Learn transport matrix
                    ridge = Ridge(alpha=1.0)
                    ridge.fit(X_train, Y_train)
                    
                    # Test on held-out entity
                    if reprs[test_entity][rel_source] is None or reprs[test_entity][rel_target] is None:
                        continue
                    
                    x_test = reprs[test_entity][rel_source].numpy().reshape(1, -1)
                    y_true = reprs[test_entity][rel_target].numpy()
                    y_pred = ridge.predict(x_test).flatten()
                    
                    # Cosine similarity between predicted and true
                    cos = F.cosine_similarity(
                        torch.tensor(y_pred).unsqueeze(0),
                        torch.tensor(y_true).unsqueeze(0)
                    ).item()
                    transport_cosines.append(cos)
                
                if transport_cosines:
                    print(f"    Transport {rel_source} -> {rel_target}: "
                          f"mean cosine = {np.mean(transport_cosines):.4f} "
                          f"(n={len(transport_cosines)})")
        
        # Test 3: BASELINE - predict h(X|rel_B) from h(X|rel_B) of neighbors
        print("\n  Test 3: Baseline - predict from same-relation neighbors")
        print("  (If transport is no better than this, entity is not abstract)")
        
        for rel_target in relations_list:
            baseline_cosines = []
            
            for test_idx in range(len(entities_list)):
                train_entities = [e for i, e in enumerate(entities_list) if i != test_idx]
                test_entity = entities_list[test_idx]
                
                # Use same-relation representations of other entities to predict
                X_train = []
                Y_train = []
                for ent in train_entities:
                    if reprs[ent][rel_target] is None:
                        continue
                    # Use entity identity as features (one-hot or mean repr)
                    # Actually use the representation itself as both input and output
                    X_train.append(reprs[ent][rel_target].numpy())
                    Y_train.append(reprs[ent][rel_target].numpy())
                
                if len(X_train) < 3:
                    continue
                
                # This baseline always predicts the mean of train representations
                # which is trivially correlated. Better baseline: predict from entity mean.
                
                # Actually, the proper baseline is: how well can we predict h(X|rel_B)
                # from the MEAN representation of X across known relations?
                
                # Let me do a different baseline: 
                # predict h(X|rel_target) from h(X|rel_source) using IDENTITY mapping
                # (just cosine between same-entity different-relation representations)
                
            # Instead, let's compute same-entity cross-relation cosine
            same_entity_cross_cos = []
            for entity in entities_list:
                for i, rel1 in enumerate(relations_list):
                    for j, rel2 in enumerate(relations_list):
                        if i >= j:
                            continue
                        if reprs[entity][rel1] is None or reprs[entity][rel2] is None:
                            continue
                        cos = F.cosine_similarity(
                            reprs[entity][rel1].unsqueeze(0),
                            reprs[entity][rel2].unsqueeze(0)
                        ).item()
                        same_entity_cross_cos.append(cos)
            
            if same_entity_cross_cos:
                print(f"    Same-entity cross-relation cosine: "
                      f"{np.mean(same_entity_cross_cos):.4f} "
                      f"(this is the 'is same entity even recognizable?' baseline)")
        
        results[layer] = {
            'cross_relation_cosines': cross_relation_cosines,
        }
    
    return results


# ============================================================
# EXPERIMENT B: Relation-Entity Composition
# ============================================================

def experiment_b(model):
    """
    Test compositional generalization:
    Can we predict h(X|R_new) from h(X|R_known) + h(Y|R_new)?
    
    Train: predict h(X|R4) from [h(X|R1), h(X|R2), h(X|R3)] + relation_info(R4)
    Test: on held-out entity-relation pairs
    """
    print("\n" + "="*70)
    print("EXPERIMENT B: Relation-Entity Composition Test")
    print("="*70)
    print("\nQuestion: Can we predict entity representation in a NEW relation?")
    print("  If compositional: h(X|R_new) = compose(h_entity(X), h_relation(R_new))")
    print("  If entangled: need full context, cannot compose from parts")
    
    entities_list = list(ENTITIES.keys())
    relations_list = list(RELATIONS.keys())
    layers = [4, 6, 8, 10]
    
    for layer in layers:
        print(f"\n--- Layer {layer} ---")
        
        # Collect all representations
        reprs = {}
        for entity in entities_list:
            reprs[entity] = {}
            for relation in relations_list:
                try:
                    reprs[entity][relation] = get_entity_repr(model, entity, relation, 0, layer)
                except:
                    reprs[entity][relation] = None
        
        # For each target relation, try to predict from known relations
        for target_rel_idx, target_rel in enumerate(relations_list):
            known_rels = [r for i, r in enumerate(relations_list) if i != target_rel_idx]
            
            print(f"\n  Target relation: {target_rel}")
            print(f"  Known relations: {known_rels}")
            
            # Strategy: For each entity X, predict h(X|target_rel)
            # Features: concatenation of h(X|known_rel_1), h(X|known_rel_2), ...
            # Plus: mean of h(Y|target_rel) for Y != X (relation context)
            
            loo_cosines = []  # leave-one-out
            
            for test_idx in range(len(entities_list)):
                test_entity = entities_list[test_idx]
                train_entities = [e for i, e in enumerate(entities_list) if i != test_idx]
                
                # Build features
                X_train = []
                Y_train = []
                for ent in train_entities:
                    features = []
                    valid = True
                    for kr in known_rels:
                        if reprs[ent][kr] is None:
                            valid = False
                            break
                        features.append(reprs[ent][kr].numpy())
                    
                    if not valid or reprs[ent][target_rel] is None:
                        continue
                    
                    # Add relation context: mean representation of other entities in target relation
                    rel_context = np.zeros_like(reprs[ent][target_rel].numpy())
                    count = 0
                    for other_ent in train_entities:
                        if other_ent != ent and reprs[other_ent][target_rel] is not None:
                            rel_context += reprs[other_ent][target_rel].numpy()
                            count += 1
                    if count > 0:
                        rel_context /= count
                    features.append(rel_context)
                    
                    X_train.append(np.concatenate(features))
                    Y_train.append(reprs[ent][target_rel].numpy())
                
                if len(X_train) < 3:
                    continue
                
                # Build test features
                test_features = []
                valid = True
                for kr in known_rels:
                    if reprs[test_entity][kr] is None:
                        valid = False
                        break
                    test_features.append(reprs[test_entity][kr].numpy())
                
                if not valid or reprs[test_entity][target_rel] is None:
                    continue
                
                # Relation context for test entity
                rel_context = np.zeros_like(reprs[test_entity][target_rel].numpy())
                count = 0
                for other_ent in train_entities:
                    if reprs[other_ent][target_rel] is not None:
                        rel_context += reprs[other_ent][target_rel].numpy()
                        count += 1
                if count > 0:
                    rel_context /= count
                test_features.append(rel_context)
                
                X_test = np.concatenate(test_features).reshape(1, -1)
                Y_true = reprs[test_entity][target_rel].numpy()
                
                # Learn and predict
                ridge = Ridge(alpha=1.0)
                ridge.fit(np.array(X_train), np.array(Y_train))
                Y_pred = ridge.predict(X_test).flatten()
                
                cos = F.cosine_similarity(
                    torch.tensor(Y_pred).unsqueeze(0),
                    torch.tensor(Y_true).unsqueeze(0)
                ).item()
                loo_cosines.append(cos)
            
            if loo_cosines:
                print(f"    LOO prediction cosine: {np.mean(loo_cosines):.4f} (n={len(loo_cosines)})")
            
            # BASELINE: predict from same-relation nearest neighbor
            baseline_cosines = []
            for test_idx in range(len(entities_list)):
                test_entity = entities_list[test_idx]
                if reprs[test_entity][target_rel] is None:
                    continue
                
                # Find nearest neighbor in known relations and use its target_rel repr
                best_cos = -2
                for other_ent in entities_list:
                    if other_ent == test_entity:
                        continue
                    if reprs[other_ent][target_rel] is None:
                        continue
                    
                    # Similarity in known relations
                    sim = 0
                    count = 0
                    for kr in known_rels:
                        if reprs[test_entity][kr] is None or reprs[other_ent][kr] is None:
                            continue
                        s = F.cosine_similarity(
                            reprs[test_entity][kr].unsqueeze(0),
                            reprs[other_ent][kr].unsqueeze(0)
                        ).item()
                        sim += s
                        count += 1
                    if count > 0:
                        sim /= count
                    
                    if sim > best_cos:
                        best_cos = sim
                        # Baseline: use nearest neighbor's target representation
                        baseline_pred = reprs[other_ent][target_rel]
                
                if 'baseline_pred' in dir():
                    cos = F.cosine_similarity(
                        baseline_pred.unsqueeze(0),
                        reprs[test_entity][target_rel].unsqueeze(0)
                    ).item()
                    baseline_cosines.append(cos)
            
            if baseline_cosines:
                print(f"    Baseline (nearest-neighbor in known rels): "
                      f"{np.mean(baseline_cosines):.4f} (n={len(baseline_cosines)})")
            
            # RANDOM BASELINE: random vector
            if loo_cosines:
                dim = len(reprs[entities_list[0]][target_rel].numpy())
                random_cosines = []
                for _ in range(100):
                    rand_vec = torch.randn(dim)
                    rand_vec = rand_vec / rand_vec.norm()
                    true_vec = reprs[entities_list[0]][target_rel]
                    true_vec = true_vec / true_vec.norm()
                    cos = F.cosine_similarity(rand_vec.unsqueeze(0), true_vec.unsqueeze(0)).item()
                    random_cosines.append(cos)
                print(f"    Random baseline: {np.mean(np.abs(random_cosines)):.4f}")


# ============================================================
# EXPERIMENT C: Surface Form Control
# ============================================================

def experiment_c(model):
    """
    Control for surface form statistics.
    
    Key test: Does "binding" survive when we scramble the template?
    If yes → genuine semantic role
    If no → statistical template effect
    """
    print("\n" + "="*70)
    print("EXPERIMENT C: Surface Form Control")
    print("="*70)
    print("\nQuestion: Is the 'binding' effect genuine or a statistical artifact?")
    
    layers = [4, 6, 8, 10]
    
    # Canonical vs scrambled templates
    canonical_templates = {
        "capital": "The capital of {entity} is",
        "currency": "The currency of {entity} is",
    }
    
    # Scrambled versions (same tokens, different order)
    # We can't truly scramble without changing meaning, but we can:
    # 1. Use different phrasing with SAME semantic role
    # 2. Use passive vs active voice
    # 3. Change token positions while keeping meaning
    
    alternative_templates = {
        "capital": [
            "{entity}'s capital is",           # Different syntax
            "What is the capital of {entity}", # Question form
            "The capital city of {entity} is",  # Extra word
        ],
        "currency": [
            "{entity}'s currency is",
            "What currency does {entity} use",
            "The official currency of {entity} is",
        ],
    }
    
    # Also test: same template but with NON-semantic role filler
    # e.g., "The capital of France is" vs "The weather of France is"
    # This controls for: is it just "entity after preposition"?
    
    non_semantic_templates = {
        "capital": [
            "The size of {entity} is",       # No standard answer
            "The weather of {entity} is",     # No standard answer  
            "The population of {entity} is",  # Different relation
        ],
    }
    
    entities_for_test = ["France", "Germany", "Japan", "Italy", "Spain", "China"]
    
    for layer in layers:
        print(f"\n--- Layer {layer} ---")
        
        # Test 1: Cross-template entity direction consistency
        print("\n  Test 1: Cross-template entity direction consistency")
        print("  (Is entity direction the SAME across different phrasings?)")
        
        for relation in ["capital", "currency"]:
            templates = [canonical_templates[relation]] + alternative_templates.get(relation, [])
            
            # Collect representations for each template
            template_reprs = {}  # template_idx -> {entity: vector}
            for t_idx, template in enumerate(templates):
                template_reprs[t_idx] = {}
                for entity in entities_for_test:
                    prompt = template.format(entity=entity)
                    try:
                        template_reprs[t_idx][entity] = get_last_token_repr(model, prompt, layer)
                    except:
                        pass
            
            # Compute entity directions for each template
            for i in range(len(templates)):
                for j in range(i+1, len(templates)):
                    cosines = []
                    for ei in range(len(entities_for_test)):
                        for ej in range(ei+1, len(entities_for_test)):
                            ent_i, ent_j = entities_for_test[ei], entities_for_test[ej]
                            if ent_i not in template_reprs[i] or ent_j not in template_reprs[i]:
                                continue
                            if ent_i not in template_reprs[j] or ent_j not in template_reprs[j]:
                                continue
                            
                            dir_i = template_reprs[i][ent_i] - template_reprs[i][ent_j]
                            dir_j = template_reprs[j][ent_i] - template_reprs[j][ent_j]
                            
                            if dir_i.norm() < 1e-6 or dir_j.norm() < 1e-6:
                                continue
                            
                            cos = F.cosine_similarity(dir_i.unsqueeze(0), dir_j.unsqueeze(0)).item()
                            cosines.append(cos)
                    
                    if cosines:
                        print(f"    {relation}: Template {i} <-> {j}: "
                              f"entity dir cosine = {np.mean(cosines):.4f} (n={len(cosines)})")
        
        # Test 2: Semantic vs non-semantic template comparison
        print("\n  Test 2: Semantic vs non-semantic template")
        print("  (Is 'capital' entity direction different from 'size' entity direction?)")
        
        sem_reprs = {}   # entity -> repr for "The capital of {entity} is"
        nonsem_reprs = {}  # entity -> repr for "The size of {entity} is"
        
        for entity in entities_for_test:
            try:
                sem_reprs[entity] = get_last_token_repr(
                    model, f"The capital of {entity} is", layer)
                nonsem_reprs[entity] = get_last_token_repr(
                    model, f"The size of {entity} is", layer)
            except:
                pass
        
        # Entity directions in semantic vs non-semantic context
        sem_vs_nonsem_cosines = []
        for ei in range(len(entities_for_test)):
            for ej in range(ei+1, len(entities_for_test)):
                ent_i, ent_j = entities_for_test[ei], entities_for_test[ej]
                if ent_i not in sem_reprs or ent_j not in sem_reprs:
                    continue
                if ent_i not in nonsem_reprs or ent_j not in nonsem_reprs:
                    continue
                
                dir_sem = sem_reprs[ent_i] - sem_reprs[ent_j]
                dir_nonsem = nonsem_reprs[ent_i] - nonsem_reprs[ent_j]
                
                if dir_sem.norm() < 1e-6 or dir_nonsem.norm() < 1e-6:
                    continue
                
                cos = F.cosine_similarity(dir_sem.unsqueeze(0), dir_nonsem.unsqueeze(0)).item()
                sem_vs_nonsem_cosines.append(cos)
        
        if sem_vs_nonsem_cosines:
            print(f"    Capital vs Size entity direction cosine: "
                  f"{np.mean(sem_vs_nonsem_cosines):.4f}")
            print(f"    (If high → entity direction is just 'which country', not semantic role)")
            print(f"    (If low → entity direction encodes relation-specific information)")
        
        # Test 3: Position control
        print("\n  Test 3: Position control")
        print("  (Same entity at different positions → same representation?)")
        
        # "The capital of France is" vs "France's capital is"
        # In first: entity at position 3 (0-indexed)
        # In second: entity at position 0
        
        pos_reprs = {}
        for entity in entities_for_test[:4]:
            try:
                # Get entity token position and its representation
                prompt1 = f"The capital of {entity} is"
                prompt2 = f"{entity}'s capital is"
                
                tokens1 = model.to_tokens(prompt1)
                tokens2 = model.to_tokens(prompt2)
                
                with torch.no_grad():
                    _, cache1 = model.run_with_cache(tokens1)
                    _, cache2 = model.run_with_cache(tokens2)
                
                # Last token representation (prediction point)
                last1 = cache1[f'blocks.{layer}.hook_resid_post'][0, -1, :].detach().cpu()
                last2 = cache2[f'blocks.{layer}.hook_resid_post'][0, -1, :].detach().cpu()
                
                pos_reprs[entity] = (last1, last2)
            except:
                pass
        
        # Compare representations at prediction point
        pos_cosines = []
        for entity, (r1, r2) in pos_reprs.items():
            cos = F.cosine_similarity(r1.unsqueeze(0), r2.unsqueeze(0)).item()
            pos_cosines.append(cos)
        
        if pos_cosines:
            print(f"    Same meaning, different position: cosine = {np.mean(pos_cosines):.4f}")
            print(f"    (If high → representation is meaning-driven)")
            print(f"    (If low → representation is position-driven)")


# ============================================================
# EXPERIMENT D: Geometric Structure of Context-Conditioning
# ============================================================

def experiment_d(model):
    """
    Map the STRUCTURE of context-conditioning.
    
    Key question: Is there systematic structure in how entity 
    representations change across relations?
    
    Specifically: does h(X|R1) - h(X|R2) correlate with h(Y|R1) - h(Y|R2)?
    If yes → systematic context-conditioning (geometric structure)
    If no → random warping per entity
    """
    print("\n" + "="*70)
    print("EXPERIMENT D: Geometric Structure of Context-Conditioning")
    print("="*70)
    print("\nQuestion: Is there STRUCTURE in how context conditions representations?")
    
    entities_list = list(ENTITIES.keys())[:10]
    relations_list = list(RELATIONS.keys())
    layers = [4, 6, 8, 10]
    
    for layer in layers:
        print(f"\n--- Layer {layer} ---")
        
        # Collect representations
        reprs = {}
        for entity in entities_list:
            reprs[entity] = {}
            for relation in relations_list:
                try:
                    reprs[entity][relation] = get_entity_repr(model, entity, relation, 0, layer)
                except:
                    reprs[entity][relation] = None
        
        # For each pair of relations, compute context-shift vectors
        # shift(X, R1->R2) = h(X|R2) - h(X|R1)
        
        print("\n  Test 1: Cross-entity consistency of context shifts")
        print("  (Is shift(X, cap->cur) parallel to shift(Y, cap->cur)?)")
        
        for i, rel1 in enumerate(relations_list):
            for j, rel2 in enumerate(relations_list):
                if i >= j:
                    continue
                
                # Compute shift vectors for each entity
                shifts = {}
                for entity in entities_list:
                    if reprs[entity][rel1] is None or reprs[entity][rel2] is None:
                        continue
                    shifts[entity] = reprs[entity][rel2] - reprs[entity][rel1]
                
                # Compare shift vectors across entities
                shift_cosines = []
                shift_entities = list(shifts.keys())
                for ei in range(len(shift_entities)):
                    for ej in range(ei+1, len(shift_entities)):
                        s1 = shifts[shift_entities[ei]]
                        s2 = shifts[shift_entities[ej]]
                        if s1.norm() < 1e-6 or s2.norm() < 1e-6:
                            continue
                        cos = F.cosine_similarity(s1.unsqueeze(0), s2.unsqueeze(0)).item()
                        shift_cosines.append(cos)
                
                if shift_cosines:
                    print(f"    Shift {rel1} -> {rel2}: "
                          f"cross-entity cosine = {np.mean(shift_cosines):.4f} "
                          f"(n={len(shift_cosines)})")
        
        # Test 2: Can we predict context-shift from relation difference?
        print("\n  Test 2: Is context-shift predictable from relation pair?")
        
        # If shifts are consistent across entities, we can learn a 
        # "relation transition operator" that's entity-independent
        
        for i, rel1 in enumerate(relations_list):
            for j, rel2 in enumerate(relations_list):
                if i >= j:
                    continue
                
                # Learn shift operator: h(X|rel2) ≈ h(X|rel1) + T(h(X|rel1))
                # or simply: h(X|rel2) ≈ W @ h(X|rel1) + b
                
                loo_cosines = []
                shift_entities = [e for e in entities_list 
                                  if reprs[e][rel1] is not None and reprs[e][rel2] is not None]
                
                for test_idx in range(len(shift_entities)):
                    train_ents = [e for i, e in enumerate(shift_entities) if i != test_idx]
                    test_ent = shift_entities[test_idx]
                    
                    X_train = np.array([reprs[e][rel1].numpy() for e in train_ents])
                    Y_train = np.array([reprs[e][rel2].numpy() for e in train_ents])
                    
                    ridge = Ridge(alpha=1.0)
                    ridge.fit(X_train, Y_train)
                    
                    X_test = reprs[test_ent][rel1].numpy().reshape(1, -1)
                    Y_true = reprs[test_ent][rel2].numpy()
                    Y_pred = ridge.predict(X_test).flatten()
                    
                    cos = F.cosine_similarity(
                        torch.tensor(Y_pred).unsqueeze(0),
                        torch.tensor(Y_true).unsqueeze(0)
                    ).item()
                    loo_cosines.append(cos)
                
                if loo_cosines:
                    print(f"    Predict {rel2} from {rel1}: "
                          f"LOO cosine = {np.mean(loo_cosines):.4f}")
        
        # Test 3: PCA of context-conditioning structure
        print("\n  Test 3: Dimensionality of context-conditioning")
        
        # Stack all entity representations across all relations
        all_reprs = []
        labels = []
        for entity in entities_list:
            for relation in relations_list:
                if reprs[entity][relation] is not None:
                    all_reprs.append(reprs[entity][relation].numpy())
                    labels.append((entity, relation))
        
        if len(all_reprs) > 3:
            all_reprs = np.array(all_reprs)
            pca = PCA()
            pca.fit(all_reprs)
            
            # How many PCs to explain 90% variance?
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            n_90 = np.searchsorted(cumvar, 0.9) + 1
            n_95 = np.searchsorted(cumvar, 0.95) + 1
            n_99 = np.searchsorted(cumvar, 0.99) + 1
            
            print(f"    PCs for 90% variance: {n_90}")
            print(f"    PCs for 95% variance: {n_95}")
            print(f"    PCs for 99% variance: {n_99}")
            print(f"    Total representations: {len(all_reprs)}, dimension: {all_reprs.shape[1]}")
            
            # Check if first few PCs separate by entity vs relation
            if len(labels) > 0:
                pc1_vals = all_reprs[:, 0]
                pc2_vals = all_reprs[:, 1]
                
                # Correlation of PC1 with entity identity
                entity_ids = {e: i for i, e in enumerate(entities_list)}
                relation_ids = {r: i for i, r in enumerate(relations_list)}
                
                entity_labels = [entity_ids[l[0]] for l in labels]
                relation_labels = [relation_ids[l[1]] for l in labels]
                
                # Use logistic regression to check if PCs predict entity/relation
                if len(set(entity_labels)) > 1:
                    try:
                        lr_ent = LogisticRegression(max_iter=1000, multi_class='ovr')
                        ent_scores = cross_val_score(lr_ent, all_reprs[:, :10], entity_labels, cv=min(3, len(set(entity_labels))))
                        print(f"    PC1-10 predict entity: accuracy = {ent_scores.mean():.4f}")
                    except:
                        pass
                
                if len(set(relation_labels)) > 1:
                    try:
                        lr_rel = LogisticRegression(max_iter=1000, multi_class='ovr')
                        rel_scores = cross_val_score(lr_rel, all_reprs[:, :10], relation_labels, cv=min(3, len(set(relation_labels))))
                        print(f"    PC1-10 predict relation: accuracy = {rel_scores.mean():.4f}")
                    except:
                        pass


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, 
                       choices=["a", "b", "c", "d", "all"])
    parser.add_argument("--model", type=str, default="gpt2-small")
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    model = get_model(args.model)
    print(f"Model loaded: {args.model}, n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model}")
    
    if args.exp in ["a", "all"]:
        results_a = experiment_a(model)
    
    if args.exp in ["b", "all"]:
        results_b = experiment_b(model)
    
    if args.exp in ["c", "all"]:
        results_c = experiment_c(model)
    
    if args.exp in ["d", "all"]:
        results_d = experiment_d(model)
    
    print("\n" + "="*70)
    print("PHASE 88 COMPLETE")
    print("="*70)
