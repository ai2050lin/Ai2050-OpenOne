"""
Phase 89A: Semantic Velocity Field
===================================

THE CORE QUESTION:
  Is the Transformer a dynamical system where layers = time?

  If YES: velocity v_l = h_{l+1} - h_l should be predictable from h_l alone
          (autonomous system: v = f(h), not v = f(h, l))

  If NO:  velocity depends on layer index, layers are not "time"
          but feature extraction stages

KEY PREDICTIONS:
  1. Velocity field v_l is low-dimensional (few PCs explain most variance)
  2. v_l can be predicted from h_l (not from l) -> autonomous dynamics
  3. Different relations produce different velocity fields
  4. Velocity fields of different relations share a common basis

EXPERIMENTS:
  A. Velocity Field Structure - dimensionality, relation-dependence
  B. Autonomy Test - can v be predicted from h alone?
  C. Cross-Relation Velocity Decomposition - shared vs task-specific basis
  D. Trajectory Divergence - do different entities diverge then converge?
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import sys
from sklearn.linear_model import Ridge
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
# DATA
# ============================================================

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


def get_all_layer_reprs(model, prompt):
    """Get representations at ALL layers for a single prompt."""
    tokens = model.to_tokens(prompt)
    n_layers = model.cfg.n_layers
    
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    
    last_pos = tokens.shape[1] - 1
    
    reprs = []
    # Layer 0 = embedding (hook_resid_pre at block 0)
    reprs.append(cache['blocks.0.hook_resid_pre'][0, last_pos, :].detach().cpu())
    
    # Layers 1..N-1 = after each transformer block
    for l in range(n_layers):
        reprs.append(cache[f'blocks.{l}.hook_resid_post'][0, last_pos, :].detach().cpu())
    
    return reprs  # len = n_layers + 1


# ============================================================
# EXPERIMENT A: Velocity Field Structure
# ============================================================

def experiment_a(model):
    """
    Study the structure of velocity fields v_l = h_{l+1} - h_l.
    
    Key questions:
    1. Are velocity fields low-dimensional?
    2. Do different relations produce different velocity fields?
    3. Is there a relation-specific velocity + shared velocity decomposition?
    """
    print("\n" + "="*70)
    print("EXPERIMENT A: Velocity Field Structure")
    print("="*70)
    
    entities_list = list(ENTITIES.keys())
    relations_list = list(RELATIONS.keys())
    n_layers = model.cfg.n_layers
    
    # Collect all-layer representations for all entity-relation combos
    print("\nCollecting all-layer representations...")
    all_reprs = {}  # (entity, relation) -> [h_0, h_1, ..., h_N]
    
    for entity in entities_list:
        for relation in relations_list:
            template = RELATIONS[relation][0]
            prompt = template.format(entity=entity)
            try:
                reprs = get_all_layer_reprs(model, prompt)
                all_reprs[(entity, relation)] = reprs
            except Exception as e:
                print(f"  Error for {entity}/{relation}: {e}")
    
    print(f"  Collected {len(all_reprs)} trajectories")
    
    # Compute velocity fields
    all_velocities = {}  # (entity, relation) -> [v_0, v_1, ..., v_{N-1}]
    for key, reprs in all_reprs.items():
        vels = []
        for l in range(len(reprs) - 1):
            vels.append(reprs[l+1] - reprs[l])
        all_velocities[key] = vels
    
    # === Test 1: Velocity dimensionality per layer ===
    print("\n--- Test 1: Velocity dimensionality per layer ---")
    print("  (How many PCs to explain 90% of velocity variance?)")
    
    for l in range(min(n_layers, len(next(iter(all_velocities.values()))))):
        # Collect all velocities at this layer
        vels_at_l = []
        for key, vels in all_velocities.items():
            if l < len(vels):
                vels_at_l.append(vels[l].numpy())
        
        if len(vels_at_l) < 3:
            continue
        
        vels_at_l = np.array(vels_at_l)
        pca = PCA()
        pca.fit(vels_at_l)
        
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_90 = np.searchsorted(cumvar, 0.9) + 1
        n_95 = np.searchsorted(cumvar, 0.95) + 1
        
        # Velocity norm
        mean_vel_norm = np.mean([np.linalg.norm(v) for v in vels_at_l])
        
        print(f"  Layer {l:2d} -> {l+1:2d}: PCs(90%)={n_90:3d}, PCs(95%)={n_95:3d}, "
              f"mean|v|={mean_vel_norm:.4f}, n={len(vels_at_l)}")
    
    # === Test 2: Relation-specific velocity structure ===
    print("\n--- Test 2: Relation-specific velocity structure ---")
    print("  (Do different relations produce different velocity directions?)")
    
    for l in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
        if l >= len(next(iter(all_velocities.values()))):
            continue
        
        # Compute mean velocity per relation
        mean_vels = {}
        for relation in relations_list:
            vels = []
            for entity in entities_list:
                key = (entity, relation)
                if key in all_velocities and l < len(all_velocities[key]):
                    vels.append(all_velocities[key][l].numpy())
            if vels:
                mean_vels[relation] = np.mean(vels, axis=0)
        
        # Cross-relation velocity cosine
        print(f"\n  Layer {l} -> {l+1}:")
        for i, r1 in enumerate(relations_list):
            for j, r2 in enumerate(relations_list):
                if i >= j:
                    continue
                if r1 in mean_vels and r2 in mean_vels:
                    v1 = mean_vels[r1]
                    v2 = mean_vels[r2]
                    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    if n1 > 1e-8 and n2 > 1e-8:
                        cos = np.dot(v1, v2) / (n1 * n2)
                        print(f"    {r1} <-> {r2}: mean velocity cosine = {cos:.4f}")
    
    # === Test 3: Shared vs relation-specific velocity decomposition ===
    print("\n--- Test 3: Shared vs relation-specific velocity decomposition ---")
    print("  (Can velocity be decomposed into shared + relation-specific?)")
    
    for l in [0, n_layers//2, n_layers-1]:
        if l >= len(next(iter(all_velocities.values()))):
            continue
        
        # Stack all velocities at this layer
        all_v = []
        labels = []  # (relation_idx, entity_idx)
        for r_idx, relation in enumerate(relations_list):
            for e_idx, entity in enumerate(entities_list):
                key = (entity, relation)
                if key in all_velocities and l < len(all_velocities[key]):
                    all_v.append(all_velocities[key][l].numpy())
                    labels.append(r_idx)
        
        if len(all_v) < 10:
            continue
        
        all_v = np.array(all_v)
        labels = np.array(labels)
        
        # PCA on all velocities
        pca = PCA(n_components=min(20, all_v.shape[0]-1))
        pca.fit(all_v)
        
        # Project velocities of each relation onto PCs
        print(f"\n  Layer {l} -> {l+1}:")
        print(f"    Top 5 PC variances: {pca.explained_variance_ratio_[:5]}")
        
        # Check if PCs separate by relation
        projected = pca.transform(all_v)
        for pc_idx in range(min(3, projected.shape[1])):
            # ANOVA-like: variance between vs within relations
            grand_mean = np.mean(projected[:, pc_idx])
            ss_between = 0
            ss_within = 0
            for r_idx in range(len(relations_list)):
                mask = labels == r_idx
                if np.sum(mask) > 0:
                    group_mean = np.mean(projected[mask, pc_idx])
                    ss_between += np.sum(mask) * (group_mean - grand_mean)**2
                    ss_within += np.sum((projected[mask, pc_idx] - group_mean)**2)
            
            if ss_within > 0:
                f_ratio = (ss_between / (len(relations_list)-1)) / (ss_within / (len(all_v) - len(relations_list)))
                print(f"    PC{pc_idx+1}: F-ratio(relation) = {f_ratio:.2f} "
                      f"(variance_ratio={pca.explained_variance_ratio_[pc_idx]:.4f})")
    
    # === Test 4: Velocity norm profile across layers ===
    print("\n--- Test 4: Velocity norm profile across layers ---")
    print("  (Where does the most 'computation' happen?)")
    
    for relation in relations_list:
        norms = []
        for l in range(min(n_layers, len(next(iter(all_velocities.values()))))):
            vels = []
            for entity in entities_list:
                key = (entity, relation)
                if key in all_velocities and l < len(all_velocities[key]):
                    vels.append(all_velocities[key][l].numpy())
            if vels:
                mean_norm = np.mean([np.linalg.norm(v) for v in vels])
                norms.append(mean_norm)
        
        # Find peak
        if norms:
            peak_l = np.argmax(norms)
            print(f"  {relation}: peak velocity at layer {peak_l} "
                  f"(|v|={norms[peak_l]:.4f}), "
                  f"min at layer {np.argmin(norms)} (|v|={min(norms):.4f})")
    
    return all_reprs, all_velocities


# ============================================================
# EXPERIMENT B: Autonomy Test
# ============================================================

def experiment_b(model, all_reprs=None, all_velocities=None):
    """
    Test if the velocity field is autonomous: v = f(h), not v = f(h, l).
    
    Key test: Can we predict v_l from h_l better than from h_l + l?
    If v is autonomous, adding layer info should NOT improve prediction.
    """
    print("\n" + "="*70)
    print("EXPERIMENT B: Autonomy Test")
    print("="*70)
    print("\nQuestion: Is velocity v_l = f(h_l) or v_l = f(h_l, l)?")
    print("  If autonomous: v depends only on current state h")
    print("  If non-autonomous: v depends on both h and layer index")
    
    entities_list = list(ENTITIES.keys())
    relations_list = list(RELATIONS.keys())
    n_layers = model.cfg.n_layers
    
    # Collect data if not provided
    if all_reprs is None:
        print("\nCollecting all-layer representations...")
        all_reprs = {}
        for entity in entities_list:
            for relation in relations_list:
                template = RELATIONS[relation][0]
                prompt = template.format(entity=entity)
                try:
                    reprs = get_all_layer_reprs(model, prompt)
                    all_reprs[(entity, relation)] = reprs
                except:
                    pass
    
    if all_velocities is None:
        all_velocities = {}
        for key, reprs in all_reprs.items():
            vels = []
            for l in range(len(reprs) - 1):
                vels.append(reprs[l+1] - reprs[l])
            all_velocities[key] = vels
    
    # === Test 1: Cross-layer velocity prediction ===
    print("\n--- Test 1: Cross-layer velocity prediction ---")
    print("  (Can velocity at layer l predict velocity at layer l+1?)")
    
    for l in range(1, n_layers - 1):
        # Collect v_l and v_{l+1}
        v_current = []
        v_next = []
        for key, vels in all_velocities.items():
            if l < len(vels) and l+1 < len(vels):
                v_current.append(vels[l].numpy())
                v_next.append(vels[l+1].numpy())
        
        if len(v_current) < 5:
            continue
        
        v_current = np.array(v_current)
        v_next = np.array(v_next)
        
        # Predict v_{l+1} from v_l
        ridge = Ridge(alpha=1.0)
        ridge.fit(v_current, v_next)
        pred = ridge.predict(v_current)
        
        # Cosine between predicted and actual
        cosines = []
        for i in range(len(v_next)):
            c = F.cosine_similarity(
                torch.tensor(pred[i]).unsqueeze(0),
                torch.tensor(v_next[i]).unsqueeze(0)
            ).item()
            cosines.append(c)
        
        print(f"  Predict v_{l+1} from v_l: mean cosine = {np.mean(cosines):.4f}")
    
    # === Test 2: State-dependent velocity prediction (AUTONOMY) ===
    print("\n--- Test 2: State-dependent velocity (AUTONOMY TEST) ---")
    print("  (Can we predict v from h alone, without knowing the layer?)")
    
    # Collect ALL (h_l, v_l) pairs across all layers
    all_h_list = []
    all_v_list = []
    all_l_list = []  # layer indices
    
    for key, reprs in all_reprs.items():
        vels = all_velocities.get(key, [])
        for l in range(len(reprs) - 1):
            all_h_list.append(reprs[l].numpy())
            if l < len(vels):
                all_v_list.append(vels[l].numpy())
            else:
                all_v_list.append((reprs[l+1] - reprs[l]).numpy())
            all_l_list.append(l)
    
    all_h = np.array(all_h_list)
    all_v = np.array(all_v_list)
    all_l = np.array(all_l_list)
    
    all_h = np.array(all_h)
    all_v = np.array(all_v)
    all_l = np.array(all_l)
    
    # Model 1: v = f(h) only
    ridge_h = Ridge(alpha=1.0)
    ridge_h.fit(all_h, all_v)
    pred_h = ridge_h.predict(all_h)
    
    # Model 2: v = f(h, l) with layer as extra feature
    # Normalize layer index
    l_normalized = all_l.astype(float) / max(all_l.max(), 1.0)
    h_with_l = np.column_stack([all_h, l_normalized])
    ridge_hl = Ridge(alpha=1.0)
    ridge_hl.fit(h_with_l, all_v)
    pred_hl = ridge_hl.predict(h_with_l)
    
    # Compare
    cos_h = []
    cos_hl = []
    for i in range(len(all_v)):
        c_h = F.cosine_similarity(
            torch.tensor(pred_h[i]).unsqueeze(0),
            torch.tensor(all_v[i]).unsqueeze(0)
        ).item()
        c_hl = F.cosine_similarity(
            torch.tensor(pred_hl[i]).unsqueeze(0),
            torch.tensor(all_v[i]).unsqueeze(0)
        ).item()
        cos_h.append(c_h)
        cos_hl.append(c_hl)
    
    print(f"\n  Model v = f(h):     mean cosine = {np.mean(cos_h):.4f}")
    print(f"  Model v = f(h, l):  mean cosine = {np.mean(cos_hl):.4f}")
    print(f"  Improvement from adding layer: {np.mean(cos_hl) - np.mean(cos_h):.4f}")
    
    if np.mean(cos_hl) - np.mean(cos_h) < 0.02:
        print("  -> Layer info adds <2% improvement: SYSTEM IS NEARLY AUTONOMOUS")
        print("  -> Supports 'layer as time' interpretation")
    else:
        print("  -> Layer info adds significant improvement: SYSTEM IS NON-AUTONOMOUS")
        print("  -> Layers are feature extraction stages, not temporal evolution")
    
    # === Test 3: Within-layer vs cross-layer velocity prediction ===
    print("\n--- Test 3: Within-layer vs cross-layer velocity prediction ---")
    
    # For each layer, test: can a model trained on layer l predict layer m?
    test_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4]
    
    for train_l in test_layers:
        # Collect training data at this layer
        h_train = []
        v_train = []
        for key, reprs in all_reprs.items():
            if train_l < len(reprs) - 1:
                h_train.append(reprs[train_l].numpy())
                v_train.append((reprs[train_l+1] - reprs[train_l]).numpy())
        
        if len(h_train) < 5:
            continue
        
        h_train = np.array(h_train)
        v_train = np.array(v_train)
        
        ridge_layer = Ridge(alpha=1.0)
        ridge_layer.fit(h_train, v_train)
        
        # Test on each other layer
        for test_l in test_layers:
            h_test = []
            v_test = []
            for key, reprs in all_reprs.items():
                if test_l < len(reprs) - 1:
                    h_test.append(reprs[test_l].numpy())
                    v_test.append((reprs[test_l+1] - reprs[test_l]).numpy())
            
            if len(h_test) < 5:
                continue
            
            pred = ridge_layer.predict(np.array(h_test))
            
            cosines = []
            for i in range(len(v_test)):
                c = F.cosine_similarity(
                    torch.tensor(pred[i]).unsqueeze(0),
                    torch.tensor(v_test[i]).unsqueeze(0)
                ).item()
                cosines.append(c)
            
            same = "(SAME)" if train_l == test_l else ""
            print(f"  Train L{train_l}, Test L{test_l}: mean cosine = {np.mean(cosines):.4f} {same}")


# ============================================================
# EXPERIMENT C: Cross-Relation Velocity Decomposition
# ============================================================

def experiment_c(model, all_reprs=None, all_velocities=None):
    """
    Decompose velocity fields into shared vs relation-specific components.
    
    If unified dynamics: there should be a shared velocity basis
    with small relation-specific corrections.
    """
    print("\n" + "="*70)
    print("EXPERIMENT C: Cross-Relation Velocity Decomposition")
    print("="*70)
    
    entities_list = list(ENTITIES.keys())
    relations_list = list(RELATIONS.keys())
    n_layers = model.cfg.n_layers
    
    if all_velocities is None:
        if all_reprs is None:
            print("\nCollecting all-layer representations...")
            all_reprs = {}
            for entity in entities_list:
                for relation in relations_list:
                    template = RELATIONS[relation][0]
                    prompt = template.format(entity=entity)
                    try:
                        reprs = get_all_layer_reprs(model, prompt)
                        all_reprs[(entity, relation)] = reprs
                    except:
                        pass
        
        all_velocities = {}
        for key, reprs in all_reprs.items():
            vels = []
            for l in range(len(reprs) - 1):
                vels.append(reprs[l+1] - reprs[l])
            all_velocities[key] = vels
    
    # === Test 1: Shared velocity PCA ===
    print("\n--- Test 1: Shared velocity basis ---")
    print("  (Is there a low-dimensional basis shared across all relations?)")
    
    for l in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
        if l >= len(next(iter(all_velocities.values()))):
            continue
        
        # Per-relation velocity PCA
        print(f"\n  Layer {l} -> {l+1}:")
        
        relation_pcs = {}
        for relation in relations_list:
            vels = []
            for entity in entities_list:
                key = (entity, relation)
                if key in all_velocities and l < len(all_velocities[key]):
                    vels.append(all_velocities[key][l].numpy())
            
            if len(vels) < 3:
                continue
            
            vels = np.array(vels)
            pca = PCA()
            pca.fit(vels)
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            n_90 = np.searchsorted(cumvar, 0.9) + 1
            relation_pcs[relation] = (pca, n_90)
            print(f"    {relation}: PCs(90%) = {n_90}, "
                  f"top-3 var = {pca.explained_variance_ratio_[:3]}")
        
        # Cross-relation: project one relation's velocities onto another's PCs
        print(f"\n    Cross-relation projection quality:")
        rel_list = [r for r in relations_list if r in relation_pcs]
        for i, r1 in enumerate(rel_list):
            for j, r2 in enumerate(rel_list):
                if i >= j:
                    continue
                
                # Get r1's velocities
                vels_r1 = []
                for entity in entities_list:
                    key = (entity, r1)
                    if key in all_velocities and l < len(all_velocities[key]):
                        vels_r1.append(all_velocities[key][l].numpy())
                
                # Get r2's PCA basis
                pca_r2, n_90_r2 = relation_pcs[r2]
                
                # Project r1's velocities onto r2's top PCs
                vels_r1 = np.array(vels_r1)
                projected = pca_r2.transform(vels_r1)
                reconstructed = pca_r2.inverse_transform(
                    np.column_stack([projected[:, :n_90_r2], 
                                    np.zeros((len(vels_r1), len(pca_r2.components_) - n_90_r2))])
                )
                
                # Reconstruction quality
                cosines = []
                for k in range(len(vels_r1)):
                    orig = vels_r1[k]
                    recon = reconstructed[k]
                    n_orig = np.linalg.norm(orig)
                    n_recon = np.linalg.norm(recon)
                    if n_orig > 1e-8 and n_recon > 1e-8:
                        cos = np.dot(orig, recon) / (n_orig * n_recon)
                        cosines.append(cos)
                
                if cosines:
                    print(f"      {r1} vel projected onto {r2} top-{n_90_r2} PCs: "
                          f"recon cosine = {np.mean(cosines):.4f}")
    
    # === Test 2: Velocity decomposition: v = v_shared + v_relation ===
    print("\n--- Test 2: Velocity decomposition: v = v_shared + v_relation ---")
    
    for l in [n_layers//2, n_layers-1]:
        if l >= len(next(iter(all_velocities.values()))):
            continue
        
        # Compute global mean velocity
        all_v = []
        for key, vels in all_velocities.items():
            if l < len(vels):
                all_v.append(vels[l].numpy())
        
        if len(all_v) < 5:
            continue
        
        all_v = np.array(all_v)
        global_mean_v = np.mean(all_v, axis=0)
        
        # Compute relation-specific mean velocities
        relation_mean_v = {}
        for relation in relations_list:
            vels = []
            for entity in entities_list:
                key = (entity, relation)
                if key in all_velocities and l < len(all_velocities[key]):
                    vels.append(all_velocities[key][l].numpy())
            if vels:
                relation_mean_v[relation] = np.mean(vels, axis=0)
        
        # Decompose: v = v_shared + v_relation_specific
        # v_shared = global mean (shared across all tasks)
        # v_relation_specific = relation_mean - global_mean
        
        print(f"\n  Layer {l} -> {l+1}:")
        print(f"    Global mean velocity norm: {np.linalg.norm(global_mean_v):.4f}")
        
        for relation, mean_v in relation_mean_v.items():
            v_specific = mean_v - global_mean_v
            shared_frac = np.linalg.norm(global_mean_v) / (np.linalg.norm(global_mean_v) + np.linalg.norm(v_specific) + 1e-10)
            print(f"    {relation}: |v_shared|={np.linalg.norm(global_mean_v):.4f}, "
                  f"|v_specific|={np.linalg.norm(v_specific):.4f}, "
                  f"shared_fraction={shared_frac:.4f}")
        
        # Are relation-specific velocities orthogonal?
        print(f"\n    Relation-specific velocity orthogonality:")
        rel_names = list(relation_mean_v.keys())
        for i, r1 in enumerate(rel_names):
            for j, r2 in enumerate(rel_names):
                if i >= j:
                    continue
                vs1 = relation_mean_v[r1] - global_mean_v
                vs2 = relation_mean_v[r2] - global_mean_v
                n1, n2 = np.linalg.norm(vs1), np.linalg.norm(vs2)
                if n1 > 1e-8 and n2 > 1e-8:
                    cos = np.dot(vs1, vs2) / (n1 * n2)
                    print(f"      {r1} <-> {r2}: cosine of specific components = {cos:.4f}")


# ============================================================
# EXPERIMENT D: Trajectory Analysis
# ============================================================

def experiment_d(model, all_reprs=None):
    """
    Analyze full trajectories across layers.
    
    Key questions:
    1. Do different entities converge then diverge (or vice versa)?
    2. Is there a "bottleneck" layer where trajectories are most compressed?
    3. Do trajectories of the same entity across relations share structure?
    """
    print("\n" + "="*70)
    print("EXPERIMENT D: Trajectory Analysis")
    print("="*70)
    
    entities_list = list(ENTITIES.keys())
    relations_list = list(RELATIONS.keys())
    n_layers = model.cfg.n_layers
    
    if all_reprs is None:
        print("\nCollecting all-layer representations...")
        all_reprs = {}
        for entity in entities_list:
            for relation in relations_list:
                template = RELATIONS[relation][0]
                prompt = template.format(entity=entity)
                try:
                    reprs = get_all_layer_reprs(model, prompt)
                    all_reprs[(entity, relation)] = reprs
                except:
                    pass
    
    # === Test 1: Trajectory divergence profile ===
    print("\n--- Test 1: Trajectory divergence across layers ---")
    print("  (Where are representations most different? Most similar?)")
    
    # Same entity, different relations: how much does relation separate?
    # Same relation, different entities: how much does entity separate?
    
    for l in range(min(n_layers + 1, len(next(iter(all_reprs.values()))))):
        # Same entity, different relations
        same_ent_diff_rel = []
        for entity in entities_list:
            reprs = []
            for relation in relations_list:
                key = (entity, relation)
                if key in all_reprs and l < len(all_reprs[key]):
                    reprs.append(all_reprs[key][l])
            for i in range(len(reprs)):
                for j in range(i+1, len(reprs)):
                    c = F.cosine_similarity(reprs[i].unsqueeze(0), reprs[j].unsqueeze(0)).item()
                    same_ent_diff_rel.append(c)
        
        # Same relation, different entities
        same_rel_diff_ent = []
        for relation in relations_list:
            reprs = []
            for entity in entities_list:
                key = (entity, relation)
                if key in all_reprs and l < len(all_reprs[key]):
                    reprs.append(all_reprs[key][l])
            for i in range(len(reprs)):
                for j in range(i+1, len(reprs)):
                    c = F.cosine_similarity(reprs[i].unsqueeze(0), reprs[j].unsqueeze(0)).item()
                    same_rel_diff_ent.append(c)
        
        # All cross-pair
        all_cross = []
        all_reprs_at_l = []
        for key, reprs in all_reprs.items():
            if l < len(reprs):
                all_reprs_at_l.append(reprs[l])
        for i in range(len(all_reprs_at_l)):
            for j in range(i+1, len(all_reprs_at_l)):
                c = F.cosine_similarity(all_reprs_at_l[i].unsqueeze(0), all_reprs_at_l[j].unsqueeze(0)).item()
                all_cross.append(c)
        
        if same_ent_diff_rel and same_rel_diff_ent:
            if l % 2 == 0 or l in [0, n_layers//2, n_layers]:  # Print selectively
                print(f"  L{l:2d}: same_ent/diff_rel={np.mean(same_ent_diff_rel):.4f}, "
                      f"same_rel/diff_ent={np.mean(same_rel_diff_ent):.4f}, "
                      f"all_cross={np.mean(all_cross):.4f}")
    
    # === Test 2: Dimensionality profile ===
    print("\n--- Test 2: Representation dimensionality across layers ---")
    
    for l in range(min(n_layers + 1, len(next(iter(all_reprs.values()))))):
        reprs_at_l = []
        for key, reprs in all_reprs.items():
            if l < len(reprs):
                reprs_at_l.append(reprs[l].numpy())
        
        if len(reprs_at_l) < 5:
            continue
        
        reprs_at_l = np.array(reprs_at_l)
        pca = PCA()
        pca.fit(reprs_at_l)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_90 = np.searchsorted(cumvar, 0.9) + 1
        
        # Participation ratio
        ev = pca.explained_variance_ratio_
        pr = (np.sum(ev))**2 / np.sum(ev**2)
        
        if l % 2 == 0 or l in [0, n_layers//2, n_layers]:
            print(f"  L{l:2d}: PCs(90%)={n_90:3d}, PR={pr:.2f}")
    
    # === Test 3: Trajectory curvature ===
    print("\n--- Test 3: Trajectory curvature ---")
    print("  (Are trajectories smooth or sharp-turning?)")
    
    for key, reprs in all_reprs.items():
        entity, relation = key
        curvatures = []
        for l in range(1, len(reprs) - 1):
            v1 = reprs[l] - reprs[l-1]
            v2 = reprs[l+1] - reprs[l]
            n1, n2 = v1.norm().item(), v2.norm().item()
            if n1 > 1e-8 and n2 > 1e-8:
                cos = F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()
                # Curvature ~ angle between consecutive velocity vectors
                curvatures.append(1 - cos)  # 0 = straight, 2 = U-turn
        
        if curvatures and entity in ["France", "Japan", "China"]:
            peak_curv_l = np.argmax(curvatures)
            print(f"  {entity}/{relation}: mean curvature={np.mean(curvatures):.4f}, "
                  f"peak at L{peak_curv_l} (curv={curvatures[peak_curv_l]:.4f})")


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
    
    # Shared data collection for efficiency
    all_reprs = None
    all_velocities = None
    
    if args.exp in ["a", "all"]:
        all_reprs, all_velocities = experiment_a(model)
    
    if args.exp in ["b", "all"]:
        experiment_b(model, all_reprs, all_velocities)
    
    if args.exp in ["c", "all"]:
        experiment_c(model, all_reprs, all_velocities)
    
    if args.exp in ["d", "all"]:
        experiment_d(model, all_reprs)
    
    print("\n" + "="*70)
    print("PHASE 89A COMPLETE")
    print("="*70)
