"""Phase 279 Summary — cross-model comparison"""
import sys, json, numpy as np
from pathlib import Path

RESULT_DIR = Path("results/phase279_compositional_topology")

models = ["qwen3", "glm4", "deepseek7b"]
model_nlayers = {"qwen3": 36, "glm4": 40, "deepseek7b": 28}

def load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

print("=" * 70)
print("Phase 279 Cross-Model Summary")
print("=" * 70)

# Block A: Relation
print("\n=== Block A: Relation Dynamics ===")
for m in models:
    data = load_json(RESULT_DIR / f"{m}_block_a_relation.json")
    if not data:
        continue
    print(f"\n{m} (n_layers={model_nlayers[m]}):")
    
    # Swap analysis
    swap = data.get("swap_analysis", {})
    for key, val in swap.items():
        cos_vals = list(val["per_layer_cosine"].values())
        if cos_vals:
            print(f"  Swap '{key}': min={min(cos_vals):.4f}, mid={cos_vals[len(cos_vals)//2]:.4f}, "
                  f"last={cos_vals[-1]:.4f}")
    
    # SVO vs entity
    svo_ent = data.get("svo_vs_entity", {})
    if svo_ent:
        for key in list(svo_ent.keys())[:3]:
            pl_cos = svo_ent[key].get("per_layer_cosine", {})
            cos_vals = list(pl_cos.values())
            if cos_vals:
                print(f"  SVO vs entity '{key}': mid={cos_vals[len(cos_vals)//2]:.4f}, last={cos_vals[-1]:.4f}")
    
    # Subspace overlap
    so = data.get("subspace_overlap", {})
    if so:
        layers_sorted = sorted(so.keys(), key=int)
        print(f"  Subspace overlap: L0={so.get('0','N/A')}, "
              f"mid={so.get(str(model_nlayers[m]//2), 'N/A')}, "
              f"last={so.get(str(model_nlayers[m]-1), 'N/A')}")
    
    # Increment ranks
    svo_r = data.get("svo_increment_ranks", {})
    ent_r = data.get("entity_increment_ranks", {})
    mid_l = str(model_nlayers[m] // 2)
    svo_vt1 = svo_r.get(mid_l, {}).get("var_top1", "N/A")
    ent_vt1 = ent_r.get(mid_l, {}).get("var_top1", "N/A")
    svo_ns = svo_r.get(mid_l, {}).get("n_sig", "N/A")
    ent_ns = ent_r.get(mid_l, {}).get("n_sig", "N/A")
    print(f"  Mid-layer var_top1: SVO={svo_vt1}, entity={ent_vt1}")
    print(f"  Mid-layer n_sig: SVO={svo_ns}, entity={ent_ns}")

# Block B: Composition
print("\n=== Block B: Composition Test ===")
for m in models:
    data = load_json(RESULT_DIR / f"{m}_block_b_composition.json")
    if not data:
        continue
    nl = model_nlayers[m]
    print(f"\n{m}:")
    
    nl_profile = data.get("nonlinearity_profile", {})
    key_layers = [0, nl//4, nl//2, 3*nl//4, nl]
    for l in key_layers:
        if str(l) in nl_profile:
            print(f"  L{l}: rel_delta={nl_profile[str(l)]['mean_rel_delta']:.4f}")
    
    # Nonlinearity direction structure
    nl_dirs = data.get("nonlinearity_directions", {})
    if nl_dirs:
        mid_l = str(nl // 2)
        if mid_l in nl_dirs:
            print(f"  Mid-layer nonlin dirs: var_top1={nl_dirs[mid_l].get('var_top1','N/A'):.4f}, "
                  f"n_sig={nl_dirs[mid_l].get('n_sig','N/A')}")

# Block C: Operator
print("\n=== Block C: Operator Dynamics ===")
for m in models:
    data = load_json(RESULT_DIR / f"{m}_block_c_operator.json")
    if not data:
        continue
    nl = model_nlayers[m]
    print(f"\n{m}:")
    
    sigs = data.get("operator_signatures", {})
    mid_l = str(nl // 2)
    for op in sorted(sigs.keys()):
        if mid_l in sigs[op]:
            print(f"  {op}: mid rel_diff={sigs[op][mid_l]['mean_rel_diff']:.4f}")
    
    # Negation test
    neg = data.get("negation_test", {})
    if neg:
        for key, pl in neg.items():
            cos_vals = [v["cosine_not_vs_antonym"] for v in pl.values()]
            if cos_vals:
                print(f"  Negation '{key}': mean cos(not_X, antonym)={np.mean(cos_vals):.4f}")
    
    # Operator subspace overlap
    ov = data.get("operator_subspace_overlap", {})
    if ov:
        for key in sorted(ov.keys())[:5]:
            print(f"  Overlap '{key}': {ov[key]:.4f}")

# Block D: Recursion
print("\n=== Block D: Recursive Closure ===")
for m in models:
    data = load_json(RESULT_DIR / f"{m}_block_d_recursion.json")
    if not data:
        continue
    print(f"\n{m}:")
    
    curv = data.get("trajectory_curvature", {})
    for key in sorted(curv.keys()):
        print(f"  {key}: curvature={curv[key]['mean_curvature']:.4f}")
    
    # Divergence from base
    div = data.get("recursion_divergence", {})
    nl = model_nlayers[m]
    mid_l = str(nl // 2)
    last_l = str(nl)
    for key in sorted(div.keys()):
        pl = div[key].get("per_layer", {})
        if mid_l in pl:
            print(f"  {key}: mid_cos={pl[mid_l]['cosine']:.4f}, "
                  f"last_cos={pl.get(last_l, {}).get('cosine', 'N/A')}")

print("\n" + "=" * 70)
print("Phase 279 Summary Complete")
