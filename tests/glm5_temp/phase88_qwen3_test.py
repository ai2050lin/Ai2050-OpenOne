"""
Phase 88 Qwen3: Counterfactual Compositional Generalization on Qwen3
=====================================================================
Key question: Does Qwen3 (which has real knowledge) show different
patterns from GPT-2-small (which doesn't)?
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def get_model():
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(
        "Qwen/Qwen2.5-1.5B",
        center_unembed=False,
        center_writing_weights=False,
        fold_ln=False,
        device="cuda",
    )
    model.eval()
    return model

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

def get_last_token_repr(model, prompt, layer):
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    last_pos = tokens.shape[1] - 1
    return cache[f'blocks.{layer}.hook_resid_post'][0, last_pos, :].detach().cpu()

def get_entity_repr(model, entity, relation, template_idx=0, layer=None):
    if layer is None:
        layer = model.cfg.n_layers // 2
    template = RELATIONS[relation][template_idx]
    prompt = template.format(entity=entity)
    return get_last_token_repr(model, prompt, layer)

def safe_token_id(model, text):
    try:
        return model.to_single_token(text)
    except:
        return None

def get_prob_for_answer(model, prompt, answer):
    tokens = model.to_tokens(prompt)
    ans_id = safe_token_id(model, answer)
    if ans_id is None:
        return None
    with torch.no_grad():
        logits = model(tokens)[0, -1, :]
    probs = torch.softmax(logits, dim=-1)
    return probs[ans_id].item()

# ============================================================
# Experiment A: Cross-Relation Entity Transport
# ============================================================

def experiment_a(model):
    print("\n" + "="*70)
    print("EXPERIMENT A: Cross-Relation Entity Transport (Qwen3)")
    print("="*70)
    
    entities_list = list(ENTITIES.keys())
    relations_list = list(RELATIONS.keys())
    # Qwen3 has ~28 layers
    layers = [8, 12, 16, 20, 24]
    
    for layer in layers:
        print(f"\n--- Layer {layer} ---")
        
        # First: check if model actually knows the facts
        print("\n  Fact check: Does Qwen3 know these answers?")
        for entity in ["France", "Germany", "Japan"]:
            for relation in relations_list:
                template = RELATIONS[relation][0]
                prompt = template.format(entity=entity)
                answer = ENTITIES[entity][relation]
                prob = get_prob_for_answer(model, prompt, answer)
                if prob is not None:
                    print(f"    {prompt} -> {answer}: P={prob:.4f}")
        
        # Collect representations
        reprs = {}
        for entity in entities_list:
            reprs[entity] = {}
            for relation in relations_list:
                try:
                    reprs[entity][relation] = get_entity_repr(model, entity, relation, 0, layer)
                except Exception as e:
                    print(f"  Error for {entity}/{relation}: {e}")
                    reprs[entity][relation] = None
        
        # Test 1: Entity direction consistency across relations
        print("\n  Test 1: Entity direction consistency across relations")
        
        for i, rel1 in enumerate(relations_list):
            for j, rel2 in enumerate(relations_list):
                if i >= j:
                    continue
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
                    print(f"    {rel1} <-> {rel2}: mean cosine = {np.mean(cosines):.4f} (n={len(cosines)})")
        
        # Test 2: Cross-relation entity transport
        print("\n  Test 2: Entity transport across relations")
        
        for rel_source in relations_list:
            for rel_target in relations_list:
                if rel_source == rel_target:
                    continue
                
                transport_cosines = []
                
                for test_idx in range(len(entities_list)):
                    train_entities = [e for i, e in enumerate(entities_list) if i != test_idx]
                    test_entity = entities_list[test_idx]
                    
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
                    
                    ridge = Ridge(alpha=1.0)
                    ridge.fit(X_train, Y_train)
                    
                    if reprs[test_entity][rel_source] is None or reprs[test_entity][rel_target] is None:
                        continue
                    
                    x_test = reprs[test_entity][rel_source].numpy().reshape(1, -1)
                    y_true = reprs[test_entity][rel_target].numpy()
                    y_pred = ridge.predict(x_test).flatten()
                    
                    cos = F.cosine_similarity(
                        torch.tensor(y_pred).unsqueeze(0),
                        torch.tensor(y_true).unsqueeze(0)
                    ).item()
                    transport_cosines.append(cos)
                
                if transport_cosines:
                    print(f"    Transport {rel_source} -> {rel_target}: "
                          f"mean cosine = {np.mean(transport_cosines):.4f}")
        
        # Test 3: Same-entity cross-relation cosine
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
            print(f"\n  Same-entity cross-relation cosine: {np.mean(same_entity_cross_cos):.4f}")
            print(f"  (GPT-2-small was 0.90-0.97; if lower here, Qwen3 has more relation-specific encoding)")


# ============================================================
# Experiment C: Surface Form Control
# ============================================================

def experiment_c(model):
    print("\n" + "="*70)
    print("EXPERIMENT C: Surface Form Control (Qwen3)")
    print("="*70)
    
    layers = [8, 12, 16, 20, 24]
    
    canonical_templates = {
        "capital": "The capital of {entity} is",
        "currency": "The currency of {entity} is",
    }
    
    alternative_templates = {
        "capital": [
            "{entity}'s capital is",
            "What is the capital of {entity}",
            "The capital city of {entity} is",
        ],
        "currency": [
            "{entity}'s currency is",
            "What currency does {entity} use",
            "The official currency of {entity} is",
        ],
    }
    
    entities_for_test = ["France", "Germany", "Japan", "Italy", "Spain", "China"]
    
    for layer in layers:
        print(f"\n--- Layer {layer} ---")
        
        # Test 1: Cross-template entity direction consistency
        print("\n  Test 1: Cross-template entity direction consistency")
        
        for relation in ["capital", "currency"]:
            templates = [canonical_templates[relation]] + alternative_templates.get(relation, [])
            
            template_reprs = {}
            for t_idx, template in enumerate(templates):
                template_reprs[t_idx] = {}
                for entity in entities_for_test:
                    prompt = template.format(entity=entity)
                    try:
                        template_reprs[t_idx][entity] = get_last_token_repr(model, prompt, layer)
                    except:
                        pass
            
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
        
        # Test 2: Semantic vs non-semantic template
        print("\n  Test 2: Semantic vs non-semantic template")
        
        sem_reprs = {}
        nonsem_reprs = {}
        
        for entity in entities_for_test:
            try:
                sem_reprs[entity] = get_last_token_repr(
                    model, f"The capital of {entity} is", layer)
                nonsem_reprs[entity] = get_last_token_repr(
                    model, f"The size of {entity} is", layer)
            except:
                pass
        
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
            print(f"    Capital vs Size entity direction cosine: {np.mean(sem_vs_nonsem_cosines):.4f}")
            print(f"    (GPT-2-small was 0.56-0.88; if lower here, Qwen3 encodes more relation-specific info)")
        
        # Test 3: Position control
        print("\n  Test 3: Position control")
        
        pos_reprs = {}
        for entity in entities_for_test[:4]:
            try:
                prompt1 = f"The capital of {entity} is"
                prompt2 = f"{entity}'s capital is"
                
                r1 = get_last_token_repr(model, prompt1, layer)
                r2 = get_last_token_repr(model, prompt2, layer)
                
                pos_reprs[entity] = (r1, r2)
            except:
                pass
        
        pos_cosines = []
        for entity, (r1, r2) in pos_reprs.items():
            cos = F.cosine_similarity(r1.unsqueeze(0), r2.unsqueeze(0)).item()
            pos_cosines.append(cos)
        
        if pos_cosines:
            print(f"    Same meaning, different position: cosine = {np.mean(pos_cosines):.4f}")


# ============================================================
# Experiment D: Geometric Structure of Context-Conditioning
# ============================================================

def experiment_d(model):
    print("\n" + "="*70)
    print("EXPERIMENT D: Geometric Structure of Context-Conditioning (Qwen3)")
    print("="*70)
    
    entities_list = list(ENTITIES.keys())[:10]
    relations_list = list(RELATIONS.keys())
    layers = [8, 12, 16, 20, 24]
    
    for layer in layers:
        print(f"\n--- Layer {layer} ---")
        
        reprs = {}
        for entity in entities_list:
            reprs[entity] = {}
            for relation in relations_list:
                try:
                    reprs[entity][relation] = get_entity_repr(model, entity, relation, 0, layer)
                except:
                    reprs[entity][relation] = None
        
        # Test 1: Cross-entity consistency of context shifts
        print("\n  Test 1: Cross-entity consistency of context shifts")
        
        for i, rel1 in enumerate(relations_list):
            for j, rel2 in enumerate(relations_list):
                if i >= j:
                    continue
                
                shifts = {}
                for entity in entities_list:
                    if reprs[entity][rel1] is None or reprs[entity][rel2] is None:
                        continue
                    shifts[entity] = reprs[entity][rel2] - reprs[entity][rel1]
                
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
                          f"cross-entity cosine = {np.mean(shift_cosines):.4f}")
        
        # Test 2: Predict context-shift from relation pair
        print("\n  Test 2: Predict h(X|rel2) from h(X|rel1)")
        
        for i, rel1 in enumerate(relations_list):
            for j, rel2 in enumerate(relations_list):
                if i >= j:
                    continue
                
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
                    print(f"    Predict {rel2} from {rel1}: LOO cosine = {np.mean(loo_cosines):.4f}")
        
        # Test 3: PCA dimensionality
        print("\n  Test 3: Dimensionality of context-conditioning")
        
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
            
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            n_90 = np.searchsorted(cumvar, 0.9) + 1
            n_95 = np.searchsorted(cumvar, 0.95) + 1
            n_99 = np.searchsorted(cumvar, 0.99) + 1
            
            print(f"    PCs for 90% variance: {n_90}")
            print(f"    PCs for 95% variance: {n_95}")
            print(f"    PCs for 99% variance: {n_99}")
            
            # Check if PCs separate by entity vs relation
            entity_ids = {e: i for i, e in enumerate(entities_list)}
            relation_ids = {r: i for i, r in enumerate(relations_list)}
            
            entity_labels = [entity_ids[l[0]] for l in labels if l[0] in entity_ids]
            relation_labels = [relation_ids[l[1]] for l in labels if l[1] in relation_ids]
            
            valid_reprs = all_reprs[:len(entity_labels)]
            
            if len(set(entity_labels)) > 1 and len(valid_reprs) > 10:
                try:
                    lr_ent = LogisticRegression(max_iter=1000, multi_class='ovr')
                    ent_scores = cross_val_score(lr_ent, valid_reprs[:, :10], entity_labels, 
                                                cv=min(3, len(set(entity_labels))))
                    print(f"    PC1-10 predict entity: accuracy = {ent_scores.mean():.4f}")
                except:
                    pass
            
            if len(set(relation_labels)) > 1 and len(valid_reprs) > 10:
                try:
                    lr_rel = LogisticRegression(max_iter=1000, multi_class='ovr')
                    rel_scores = cross_val_score(lr_rel, valid_reprs[:, :10], relation_labels,
                                                cv=min(3, len(set(relation_labels))))
                    print(f"    PC1-10 predict relation: accuracy = {rel_scores.mean():.4f}")
                except:
                    pass


if __name__ == "__main__":
    print("Loading Qwen3 model...")
    model = get_model()
    print(f"Model loaded: n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model}")
    
    experiment_a(model)
    experiment_c(model)
    experiment_d(model)
    
    print("\n" + "="*70)
    print("PHASE 88 QWEN3 COMPLETE")
    print("="*70)
