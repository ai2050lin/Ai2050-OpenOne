"""
Phase 89A: Layer Velocity Field — Multi-Model (Qwen3, GLM4, DS7B)
====================================================================
Uses model_utils.load_model() for all models (bfloat16, CPU->CUDA).
No 8bit quantization - uses the proven loading approach from model_utils.

KEY TESTS:
  A. Velocity dimensionality per layer
  B. Autonomy test (v = f(h) vs v = f(h, l))
  C. Relation-specific velocity structure
  D. Trajectory divergence profile

Run:
  python tests/glm5/ccml_phase89a_multimodel_velocity.py --model qwen3
  python tests/glm5/ccml_phase89a_multimodel_velocity.py --model glm4
  python tests/glm5/ccml_phase89a_multimodel_velocity.py --model deepseek7b
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import gc
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA

from model_utils import load_model, get_layers, get_model_info, release_model, MODEL_CONFIGS

# ============================================================
# DATA
# ============================================================

RELATIONS = {
    "capital": "The capital of {entity} is",
    "currency": "The currency of {entity} is",
    "language": "The official language of {entity} is",
}

ENTITIES = [
    "France", "Germany", "Japan", "Brazil", "India",
    "Australia", "Canada", "Mexico", "Egypt", "Thailand",
]

# ============================================================
# Representation extraction via hooks
# ============================================================

def get_all_layer_reprs(model, tokenizer, device, prompt):
    """Get representations at all layers using forward hook."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    layers = get_layers(model)
    n_layers = len(layers)
    
    captured = {}
    def make_hook(key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0][0, -1, :].detach().cpu().float()
            else:
                captured[key] = output[0, -1, :].detach().cpu().float()
        return hook
    
    hooks = []
    # Hook each transformer layer
    for li in range(n_layers):
        hooks.append(layers[li].register_forward_hook(make_hook(f"L{li}")))
    
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    
    for h in hooks:
        h.remove()
    
    # Build reprs: [embedding, L0_out, L1_out, ..., L{N-1}_out]
    reprs = []
    # Embedding = hidden_states[0]
    reprs.append(out.hidden_states[0][0, -1, :].detach().cpu().float())
    # Layer outputs from hooks
    for li in range(n_layers):
        key = f"L{li}"
        if key in captured:
            reprs.append(captured[key])
        else:
            # Fallback to hidden_states
            reprs.append(out.hidden_states[li+1][0, -1, :].detach().cpu().float())
    
    return reprs


# ============================================================
# EXPERIMENT A: Velocity Dimensionality
# ============================================================

def experiment_a(model, tokenizer, device, model_name):
    """Full velocity field analysis."""
    
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    print("\n" + "="*70)
    print(f"EXPERIMENT A: Velocity Dimensionality ({model_name})")
    print(f"  n_layers={n_layers}, d_model={d_model}")
    print("="*70)
    
    # Collect all-layer representations
    print("Collecting representations...")
    all_reprs = {}  # (entity, relation) -> [h_0, h_1, ..., h_N]
    
    for entity in ENTITIES:
        for relation, template in RELATIONS.items():
            prompt = template.format(entity=entity)
            try:
                reprs = get_all_layer_reprs(model, tokenizer, device, prompt)
                all_reprs[(entity, relation)] = reprs
            except Exception as e:
                print(f"  Error: {entity}/{relation}: {e}")
    
    print(f"  Collected {len(all_reprs)} trajectories")
    
    if len(all_reprs) == 0:
        print("  ERROR: No trajectories collected!")
        return
    
    # Compute velocities
    all_velocities = {}
    for key, reprs in all_reprs.items():
        vels = []
        for l in range(len(reprs) - 1):
            vels.append(reprs[l+1] - reprs[l])
        all_velocities[key] = vels
    
    relations_list = list(RELATIONS.keys())
    
    # === Test 1: Velocity dimensionality per layer ===
    print("\n--- Velocity dimensionality per layer ---")
    
    max_vel_len = len(next(iter(all_velocities.values())))
    for l in range(min(n_layers, max_vel_len)):
        vels_at_l = []
        for key, vels in all_velocities.items():
            if l < len(vels):
                vels_at_l.append(vels[l].numpy())
        
        if len(vels_at_l) < 5:
            continue
        
        vels_at_l = np.array(vels_at_l)
        pca = PCA()
        pca.fit(vels_at_l)
        
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_90 = np.searchsorted(cumvar, 0.9) + 1
        n_95 = np.searchsorted(cumvar, 0.95) + 1
        mean_vel_norm = np.mean([np.linalg.norm(v) for v in vels_at_l])
        
        if l in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1] or l % max(1, n_layers//8) == 0:
            print(f"  L{l:2d}->{l+1:2d}: PCs(90%)={n_90:3d}, PCs(95%)={n_95:3d}, "
                  f"|v|={mean_vel_norm:.2f}, top-3 var={pca.explained_variance_ratio_[:3].round(4)}")
    
    # === Test 2: Relation-specific velocity cosine ===
    print("\n--- Relation-specific velocity cosine ---")
    
    test_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    
    for l in test_layers:
        if l >= max_vel_len:
            continue
        
        mean_vels = {}
        for relation in relations_list:
            vels = []
            for entity in ENTITIES:
                key = (entity, relation)
                if key in all_velocities and l < len(all_velocities[key]):
                    vels.append(all_velocities[key][l].numpy())
            if vels:
                mean_vels[relation] = np.mean(vels, axis=0)
        
        print(f"\n  L{l}->{l+1}:")
        for i, r1 in enumerate(relations_list):
            for j, r2 in enumerate(relations_list):
                if i >= j:
                    continue
                if r1 in mean_vels and r2 in mean_vels:
                    v1, v2 = mean_vels[r1], mean_vels[r2]
                    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    if n1 > 1e-8 and n2 > 1e-8:
                        cos = np.dot(v1, v2) / (n1 * n2)
                        print(f"    {r1} <-> {r2}: cos={cos:.4f}")
    
    # === Test 3: Autonomy test ===
    print("\n--- Autonomy Test: v = f(h) vs v = f(h, l) ---")
    
    all_h_list = []
    all_v_list = []
    all_l_list = []
    
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
    all_l_arr = np.array(all_l_list)
    
    # Model 1: v = f(h)
    ridge_h = Ridge(alpha=1.0)
    ridge_h.fit(all_h, all_v)
    pred_h = ridge_h.predict(all_h)
    
    # Model 2: v = f(h, l)
    l_norm = all_l_arr.astype(float) / max(all_l_arr.max(), 1.0)
    h_with_l = np.column_stack([all_h, l_norm])
    ridge_hl = Ridge(alpha=1.0)
    ridge_hl.fit(h_with_l, all_v)
    pred_hl = ridge_hl.predict(h_with_l)
    
    cos_h, cos_hl = [], []
    for i in range(len(all_v)):
        c_h = F.cosine_similarity(torch.tensor(pred_h[i]).unsqueeze(0), torch.tensor(all_v[i]).unsqueeze(0)).item()
        c_hl = F.cosine_similarity(torch.tensor(pred_hl[i]).unsqueeze(0), torch.tensor(all_v[i]).unsqueeze(0)).item()
        cos_h.append(c_h)
        cos_hl.append(c_hl)
    
    improvement = np.mean(cos_hl) - np.mean(cos_h)
    print(f"  v = f(h):    mean cosine = {np.mean(cos_h):.4f}")
    print(f"  v = f(h, l): mean cosine = {np.mean(cos_hl):.4f}")
    print(f"  Layer improvement: {improvement:.4f}")
    if improvement < 0.02:
        print(f"  -> NEARLY AUTONOMOUS (layer adds <2%)")
    else:
        print(f"  -> NON-AUTONOMOUS (layer adds significant info)")
    
    # === Test 4: Cross-layer velocity prediction ===
    print("\n--- Cross-layer velocity prediction ---")
    
    test_layers_pred = [0, n_layers//4, n_layers//2, 3*n_layers//4]
    
    for train_l in test_layers_pred:
        h_train, v_train = [], []
        for key, reprs in all_reprs.items():
            if train_l < len(reprs) - 1:
                h_train.append(reprs[train_l].numpy())
                v_train.append((reprs[train_l+1] - reprs[train_l]).numpy())
        
        if len(h_train) < 5:
            continue
        
        ridge = Ridge(alpha=1.0)
        ridge.fit(np.array(h_train), np.array(v_train))
        
        for test_l in test_layers_pred:
            h_test, v_test = [], []
            for key, reprs in all_reprs.items():
                if test_l < len(reprs) - 1:
                    h_test.append(reprs[test_l].numpy())
                    v_test.append((reprs[test_l+1] - reprs[test_l]).numpy())
            
            if len(h_test) < 5:
                continue
            
            pred = ridge.predict(np.array(h_test))
            cosines = []
            for i in range(len(v_test)):
                c = F.cosine_similarity(torch.tensor(pred[i]).unsqueeze(0), torch.tensor(v_test[i]).unsqueeze(0)).item()
                cosines.append(c)
            
            same = " (SAME)" if train_l == test_l else ""
            print(f"  Train L{train_l}, Test L{test_l}: cos={np.mean(cosines):.4f}{same}")
    
    # === Test 5: Trajectory divergence profile ===
    print("\n--- Trajectory divergence across layers ---")
    
    for l in range(min(n_layers + 1, len(next(iter(all_reprs.values()))))):
        same_ent_diff_rel = []
        for entity in ENTITIES:
            reprs = [all_reprs.get((entity, r), [None]*30) for r in relations_list]
            valid_reprs = [r[l] for r in reprs if l < len(r) and r[l] is not None]
            for i in range(len(valid_reprs)):
                for j in range(i+1, len(valid_reprs)):
                    c = F.cosine_similarity(valid_reprs[i].unsqueeze(0), valid_reprs[j].unsqueeze(0)).item()
                    same_ent_diff_rel.append(c)
        
        same_rel_diff_ent = []
        for relation in relations_list:
            reprs = [all_reprs.get((e, relation), [None]*30) for e in ENTITIES]
            valid_reprs = [r[l] for r in reprs if l < len(r) and r[l] is not None]
            for i in range(len(valid_reprs)):
                for j in range(i+1, len(valid_reprs)):
                    c = F.cosine_similarity(valid_reprs[i].unsqueeze(0), valid_reprs[j].unsqueeze(0)).item()
                    same_rel_diff_ent.append(c)
        
        if same_ent_diff_rel and same_rel_diff_ent:
            if l in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers] or l % max(1, n_layers//8) == 0:
                print(f"  L{l:2d}: same_ent/diff_rel={np.mean(same_ent_diff_rel):.4f}, "
                      f"same_rel/diff_ent={np.mean(same_rel_diff_ent):.4f}")
    
    # === Test 6: Velocity decomposition ===
    print("\n--- Velocity decomposition: v_shared + v_specific ---")
    
    for l in [n_layers//2, n_layers-1]:
        if l >= max_vel_len:
            continue
        
        all_v = []
        for key, vels in all_velocities.items():
            if l < len(vels):
                all_v.append(vels[l].numpy())
        
        if len(all_v) < 5:
            continue
        
        global_mean = np.mean(all_v, axis=0)
        
        print(f"\n  L{l}->{l+1}: |v_shared|={np.linalg.norm(global_mean):.4f}")
        for relation in relations_list:
            vels = []
            for entity in ENTITIES:
                key = (entity, relation)
                if key in all_velocities and l < len(all_velocities[key]):
                    vels.append(all_velocities[key][l].numpy())
            if vels:
                rel_mean = np.mean(vels, axis=0)
                v_specific = rel_mean - global_mean
                shared_frac = np.linalg.norm(global_mean) / (np.linalg.norm(global_mean) + np.linalg.norm(v_specific) + 1e-10)
                print(f"    {relation}: |v_specific|={np.linalg.norm(v_specific):.4f}, shared_frac={shared_frac:.4f}")
    
    return all_reprs, all_velocities


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       choices=list(MODEL_CONFIGS.keys()))
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    print(f"Model loaded: {args.model}, class={info.model_class}, "
          f"n_layers={info.n_layers}, d_model={info.d_model}")
    
    experiment_a(model, tokenizer, device, args.model)
    
    # Cleanup
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n" + "="*70)
    print(f"PHASE 89A COMPLETE ({args.model})")
    print("="*70)
