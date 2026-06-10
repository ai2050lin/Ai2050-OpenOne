"""Verify Phase 430 results and extract key data for analysis validation"""
import sys, json, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

results_dir = "d:/Ai2050/TransformerLens-Project/results/phase430_natural_transport"

for model_name in ["qwen3", "glm4", "deepseek7b"]:
    for rnd in [1, 2]:
        fname = f"{model_name}_phase430_r{rnd}.json"
        fpath = os.path.join(results_dir, fname)
        if not os.path.exists(fpath):
            continue
        d = json.load(open(fpath, 'r', encoding='utf-8'))
        per_obj = d.get('per_object', {})
        print(f"\n{'='*60}")
        print(f"{model_name} R{rnd}")
        print(f"{'='*60}")
        
        for obj_name, obj_data in per_obj.items():
            print(f"\n--- {obj_name} ---")
            # Baseline
            bl = obj_data.get('baseline', {})
            if bl:
                cand_probs = bl.get('cand_probs', {})
                cat = obj_data.get('category', '?')
                opp = obj_data.get('opposing', '?')
                print(f"  Baseline: {cat}={cand_probs.get(cat,0):.3f}, {opp}={cand_probs.get(opp,0):.3f}")
            
            # Injection results - transported vs probe
            inj = obj_data.get('injection_results', {})
            if inj:
                for layer_key in sorted(inj.keys(), key=lambda x: int(x) if x.isdigit() else 0):
                    layer_data = inj[layer_key]
                    # Check for transported and probe results
                    for pos in ['obj', 'last']:
                        pos_data = layer_data.get(pos, {})
                        transported = pos_data.get('transported', {})
                        probe = pos_data.get('probe', {})
                        if transported and probe:
                            t_delta = transported.get('best_delta', '?')
                            t_H = transported.get('best_entropy', '?')
                            p_delta = probe.get('best_delta', '?')
                            p_H = probe.get('best_entropy', '?')
                            t_alpha = transported.get('best_alpha_frac', '?')
                            p_alpha = probe.get('best_alpha_frac', '?')
                            if t_delta != '?' or p_delta != '?':
                                print(f"  L{layer_key}/{pos}: Transported delta={t_delta}, H={t_H}, alpha={t_alpha} | Probe delta={p_delta}, H={p_H}, alpha={p_alpha}")
            
            # Cosine rotation data
            cos_data = obj_data.get('cosine_with_embed_direction', {})
            if cos_data:
                layers_sorted = sorted(cos_data.keys(), key=lambda x: int(x) if x.isdigit() else 0)
                cos_vals = []
                for lk in layers_sorted[:8]:
                    for pos in ['obj', 'last']:
                        cv = cos_data[lk].get(pos, None)
                        if cv is not None:
                            cos_vals.append(f"L{lk}/{pos[:1]}={cv:.3f}")
                if cos_vals:
                    print(f"  Cos(embed,transported): {' '.join(cos_vals[:8])}")
