"""Extract key transported vs probe comparison from Phase 430"""
import sys, json, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

results_dir = "d:/Ai2050/TransformerLens-Project/results/phase430_natural_transport"

# Key question: transported direction vs probe direction effectiveness
for model_name in ["qwen3", "glm4", "deepseek7b"]:
    for rnd in [1, 2]:
        fname = f"{model_name}_phase430_r{rnd}.json"
        fpath = os.path.join(results_dir, fname)
        if not os.path.exists(fpath):
            continue
        d = json.load(open(fpath, 'r', encoding='utf-8'))
        per_obj = d.get('per_object', {})
        print(f"\n{'='*70}")
        print(f"{model_name} R{rnd}")
        print(f"{'='*70}")
        
        for obj_name, obj_data in per_obj.items():
            cat = obj_data.get('category', '?')
            opp = obj_data.get('opposing', '?')
            bl = obj_data.get('baseline', {})
            bl_cand = bl.get('cand_probs', {})
            print(f"\n--- {obj_name} ({cat} vs {opp}) baseline: {cat}={bl_cand.get(cat,0):.3f} ---")
            
            # 1. Cosine rotation: embedding direction vs transported direction at each layer
            td_norms = obj_data.get('transported_directions_norms', {})
            if td_norms:
                for alpha_str in sorted(td_norms.keys(), key=float):
                    alpha_data = td_norms[alpha_str]
                    cos_vals = []
                    for layer_key in sorted(alpha_data.keys(), key=lambda x: int(x[1:])):
                        ld = alpha_data[layer_key]
                        cos_obj = ld.get('cos_obj', 0)
                        cos_last = ld.get('cos_last', 0)
                        cos_vals.append(f"L{layer_key[1:]}o={cos_obj:.2f}l={cos_last:.2f}")
                    print(f"  Cos(embed,transported) alpha={alpha_str}: {' '.join(cos_vals[:5])}")
            
            # 2. Best injection result: transported vs probe at each layer
            inj = obj_data.get('inject_results', {})
            probe_inj = obj_data.get('probe_inject_results', {})
            
            if inj:
                print(f"  Transported injection (best per layer/position):")
                for alpha_str in sorted(inj.keys(), key=float):
                    alpha_data = inj[alpha_str]
                    for layer_key in sorted(alpha_data.keys(), key=lambda x: 0 if x=='embed' else int(x[1:])):
                        layer_data = alpha_data[layer_key]
                        for pos_prefix in ['obj', 'last']:
                            best_delta = None
                            best_H = None
                            best_alpha_frac = None
                            for frac_key, frac_data in layer_data.items():
                                if not frac_key.startswith(pos_prefix):
                                    continue
                                delta = frac_data.get('delta', 0)
                                if best_delta is None or abs(delta) > abs(best_delta):
                                    best_delta = delta
                                    best_H = frac_data.get('full_entropy', 0)
                                    best_alpha_frac = frac_key
                            if best_delta is not None and abs(best_delta) > 0.05:
                                print(f"    alpha={alpha_str} {layer_key}/{pos_prefix}: delta={best_delta:.3f}, H={best_H:.1f}, {best_alpha_frac}")
            
            if probe_inj:
                print(f"  Probe injection (best per layer/position):")
                for layer_key in sorted(probe_inj.keys(), key=lambda x: 0 if x=='embed' else int(x[1:])):
                    layer_data = probe_inj[layer_key]
                    for pos_prefix in ['obj', 'last']:
                        best_delta = None
                        best_H = None
                        best_alpha_frac = None
                        for frac_key, frac_data in layer_data.items():
                            if not frac_key.startswith(pos_prefix):
                                continue
                            delta = frac_data.get('delta', 0)
                            if best_delta is None or abs(delta) > abs(best_delta):
                                best_delta = delta
                                best_H = frac_data.get('full_entropy', 0)
                                best_alpha_frac = frac_key
                        if best_delta is not None and abs(best_delta) > 0.05:
                            print(f"    {layer_key}/{pos_prefix}: delta={best_delta:.3f}, H={best_H:.1f}, {best_alpha_frac}")
            
            # 3. Causal trace
            ct = obj_data.get('causal_trace', {})
            if ct:
                print(f"  Causal trace:")
                for loc_key in sorted(ct.keys()):
                    loc_data = ct[loc_key]
                    recovery = loc_data.get('recovery', 0)
                    clean_level = loc_data.get('clean_level', 0)
                    corrupt_level = loc_data.get('corrupt_level', 0)
                    restored_level = loc_data.get('restored_level', 0)
                    print(f"    {loc_key}: recovery={recovery:.3f}, clean={clean_level:.3f}, corrupt={corrupt_level:.3f}, restored={restored_level:.3f}")
