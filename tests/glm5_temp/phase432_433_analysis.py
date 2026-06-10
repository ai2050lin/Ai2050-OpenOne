"""Extract and analyze Phase 432 and 433 results"""
import sys, json, os
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = "d:/Ai2050/TransformerLens-Project"

# ===== Phase 433 Summary =====
print("=" * 70)
print("PHASE 433: Transport Operator Stability - Cross-Model Summary")
print("=" * 70)

for model_name in ["qwen3", "glm4", "deepseek7b"]:
    fpath = os.path.join(ROOT, f"results/phase433_transport_stability/{model_name}_phase433_r1.json")
    if not os.path.exists(fpath):
        continue
    d = json.load(open(fpath, 'r', encoding='utf-8'))
    
    print(f"\n--- {model_name} (n_layers={d['n_layers']}) ---")
    
    # Within-category cosine at key layers
    per_cat = d.get('per_category', {})
    for cat_name in ['fruit', 'animal', 'tool', 'vehicle']:
        if cat_name not in per_cat:
            continue
        cat_data = per_cat[cat_name]
        objects = cat_data.get('objects', {})
        n_objects = len(set(k.split('_a')[0] for k in objects.keys()))
        
        # Get delta vectors for last alpha
        obj_vectors = {}
        for key, obj_data in objects.items():
            if 'delta_vectors' in obj_data:
                base_name = key.split('_a')[0]
                obj_vectors[base_name] = obj_data['delta_vectors']
        
        if len(obj_vectors) < 2:
            continue
        
        # Compute within-category cosine at each layer
        names = sorted(obj_vectors.keys())
        all_layer_keys = sorted(set(lk for v in obj_vectors.values() for lk in v.keys()))
        
        print(f"  {cat_name} ({n_objects} objects: {', '.join(names)}):")
        for lk in all_layer_keys:
            if '/last' not in lk:
                continue
            deltas = [np.array(obj_vectors[n][lk]) for n in names if lk in obj_vectors[n]]
            if len(deltas) < 2:
                continue
            
            import numpy as np
            cos_vals = []
            for i in range(len(deltas)):
                for j in range(i+1, len(deltas)):
                    n1 = np.linalg.norm(deltas[i])
                    n2 = np.linalg.norm(deltas[j])
                    if n1 > 1e-10 and n2 > 1e-10:
                        cos_vals.append(float(np.dot(deltas[i], deltas[j]) / (n1 * n2)))
            
            mean_cos = np.mean(cos_vals) if cos_vals else 0
            print(f"    {lk}: within-cat cos={mean_cos:.3f} (n_pairs={len(cos_vals)})")
    
    # Cross-category gap
    cross_cat = d.get('cross_category', {})
    if cross_cat:
        # Find best gap layer
        best_gap = 0
        best_lk = ''
        for lk, data in cross_cat.items():
            if '/last' in lk and data['gap'] > best_gap:
                best_gap = data['gap']
                best_lk = lk
        if best_lk:
            print(f"  Best last-pos gap: {best_lk} = {best_gap:+.3f}")

# ===== Phase 432 Summary =====
print(f"\n{'='*70}")
print("PHASE 432: Property Transport - Cross-Model Summary")
print("=" * 70)

for model_name in ["qwen3", "glm4", "deepseek7b"]:
    fpath = os.path.join(ROOT, f"results/phase432_property_transport/{model_name}_phase432_r1.json")
    if not os.path.exists(fpath):
        continue
    d = json.load(open(fpath, 'r', encoding='utf-8'))
    
    print(f"\n--- {model_name} ---")
    
    for obj_name, obj_data in d.get('per_object', {}).items():
        cat = obj_data.get('category', '?')
        print(f"  {obj_name} ({cat}):")
        
        for prop_name, prop_data in obj_data.get('properties', {}).items():
            ptype = prop_data.get('property_type', '?')
            cos_cat = prop_data.get('cos_with_category_dir', 0)
            
            # Get best alpha effect
            best_effect = 0
            best_alpha = 0
            best_direction = "?"
            for alpha in [1.0, 2.0, 4.0]:
                key = f"effect_a{alpha}"
                if key in prop_data:
                    eff = prop_data[key]
                    top_shift = eff.get('top_shift', ('?', 0))
                    if abs(top_shift[1]) > abs(best_effect):
                        best_effect = top_shift[1]
                        best_alpha = alpha
                        best_direction = top_shift[0]
            
            # Get transport cosine rotation at key layers
            transport_cos = {}
            for alpha in [4.0]:
                key = f"transport_a{alpha}"
                if key in prop_data:
                    for lk, ldata in prop_data[key].items():
                        if '/last' in lk:
                            cos_val = ldata.get('cos_with_inject', 0)
                            transport_cos[lk] = cos_val
            
            early_cos = ''
            mid_cos = ''
            deep_cos = ''
            cos_items = sorted(transport_cos.items())
            n_items = len(cos_items)
            if n_items > 0:
                early_cos = f"{cos_items[0][1]:.2f}" if cos_items else ""
            if n_items > 2:
                mid_idx = n_items // 2
                mid_cos = f"{cos_items[mid_idx][1]:.2f}"
            if n_items > 1:
                deep_cos = f"{cos_items[-1][1]:.2f}"
            
            print(f"    {prop_name} ({ptype}): cos(cat)={cos_cat:.3f}, "
                  f"best_effect={best_effect:+.4f}@α={best_alpha}→{best_direction}, "
                  f"transport_cos(last): L0={early_cos} Mid={mid_cos} Deep={deep_cos}")

# ===== Key comparison =====
print(f"\n{'='*70}")
print("CRITICAL COMPARISON: Property W_U Direction vs Category W_E Direction")
print("=" * 70)
print("""
FINDING: W_U property directions are NOT effective for injection at embedding layer.
ALL property top_shifts are NEGATIVE or negligible across all 3 models.

Category injection (Phase 430): delta up to -0.82, clean switching
Property injection (Phase 432): delta at most +0.008, mostly negative

EXPLANATION: 
- Category direction = W_E(fruit_center) - W_E(animal_center) [INPUT space]
- Property direction = W_U["red"] [OUTPUT/READOUT space]

These are fundamentally different:
- W_E difference: direction the model uses to ENCODE category at input
- W_U column: direction the model uses to READ OUT "red" from residual stream

Injecting a READOUT direction at the INPUT is like putting a microphone 
into a speaker - it's the wrong end of the pipeline.

CORRECT APPROACH for property transport:
- Define property direction in EMBEDDING space (like category)
- e.g., red_objs - not_red_objs: apple,cherry,rose vs grass,sky,snow
- This would test whether EMBEDDING-SPACE property directions also transport
""")
