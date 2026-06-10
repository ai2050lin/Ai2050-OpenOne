import json
from pathlib import Path

ROOT = Path(r"D:\Ai2050\TransformerLens-Project")

models = ["qwen3", "glm4", "deepseek7b"]
objects = ["apple", "knife", "car"]

print("=" * 80)
print("PHASE 429B R1 CROSS-MODEL COMPARISON")
print("=" * 80)

for model_name in models:
    path = ROOT / "results" / "phase429b_norm_scaled" / f"{model_name}_phase429b_r1.json"
    if not path.exists():
        print(f"\n{model_name}: FILE NOT FOUND")
        continue
    
    with open(path) as f:
        d = json.load(f)
    
    print(f"\n{'='*80}")
    print(f"  {model_name} (n_layers={d['n_layers']}, d_model={d['d_model']})")
    print(f"{'='*80}")
    
    # Residual norms
    print("\n--- Residual Norms ---")
    for obj_name in objects:
        if obj_name in d.get("residual_norms_summary", {}):
            norms = d["residual_norms_summary"][obj_name]
            print(f"  {obj_name}:")
            for layer, vals in norms.items():
                print(f"    {layer}: obj={vals['obj']:.1f}, last={vals['last']:.1f}")
    
    # Key perturbation results
    print("\n--- Category Switch at LAST TOKEN (layer_probe, negative alpha) ---")
    for obj_name in objects:
        if obj_name not in d["per_object"]:
            continue
        od = d["per_object"][obj_name]
        base = od["baselines"]["category"]["level"]
        base_top = od["baselines"]["category"]["top"]
        base_H = od["baselines"]["category"]["full_entropy"]
        base_c = od["baselines"]["category"]["confidence"]
        
        print(f"\n  {obj_name} (base: level={base:.3f}, top={base_top}, H={base_H:.2f}, c={base_c:.3f}):")
        
        key = "category_layer_probe"
        if key not in od["perturbations"]:
            continue
        pd = od["perturbations"][key]
        
        # Find best negative alpha at each layer/position
        for pos in ["obj", "last"]:
            if pos not in pd:
                continue
            for layer in sorted(pd[pos].keys()):
                curve = pd[pos][layer]
                # Show all negative alphas
                for af in ["-2.0", "-1.0", "-0.5"]:
                    if af in curve:
                        c = curve[af]
                        switch = "SWITCH" if c["top"] != base_top else ""
                        clean = "CLEAN" if c["full_entropy"] < 6 and c["confidence"] > 0.3 else ""
                        confused = "CONFUSED" if c["full_entropy"] > 8 or c["confidence"] < 0.1 else ""
                        label = switch or clean or confused or "partial"
                        print(f"    layer_probe@{layer}/{pos} a={af}: D={c['delta']:+.3f} top={c['top'][:5]} "
                              f"H={c['full_entropy']:.1f} c={c['confidence']:.3f} mag={c['actual_magnitude']:.1f} [{label}]")
        
        # Also show embed_dir for comparison
        key2 = "category_embed_dir"
        if key2 in od["perturbations"]:
            pd2 = od["perturbations"][key2]
            for pos in ["obj", "last"]:
                if pos not in pd2:
                    continue
                for af in ["-2.0", "-1.0"]:
                    if af in pd2[pos].get("embed", {}):
                        c = pd2[pos]["embed"][af]
                        switch = "SWITCH" if c["top"] != base_top else ""
                        clean = "CLEAN" if c["full_entropy"] < 6 and c["confidence"] > 0.3 else ""
                        label = switch or clean or "partial"
                        print(f"    embed_dir@embed/{pos} a={af}: D={c['delta']:+.3f} top={c['top'][:5]} "
                              f"H={c['full_entropy']:.1f} c={c['confidence']:.3f} [{label}]")
    
    # Positive alpha results for comparison
    print("\n--- Category Enhancement at LAST TOKEN (layer_probe, positive alpha) ---")
    for obj_name in objects:
        if obj_name not in d["per_object"]:
            continue
        od = d["per_object"][obj_name]
        base_top = od["baselines"]["category"]["top"]
        
        key = "category_layer_probe"
        if key not in od["perturbations"]:
            continue
        pd = od["perturbations"][key]
        
        for pos in ["last"]:
            if pos not in pd:
                continue
            best_layer = ""
            best_delta = 0
            for layer in sorted(pd[pos].keys()):
                curve = pd[pos][layer]
                for af in ["0.5", "1.0", "2.0"]:
                    if af in curve:
                        c = curve[af]
                        if abs(c["delta"]) > abs(best_delta):
                            best_delta = c["delta"]
                            best_layer = f"{layer}/a={af}"
            
            if best_layer:
                print(f"  {obj_name}/{pos}: best positive = D={best_delta:+.3f} at {best_layer}")

print("\n" + "=" * 80)
print("KEY COMPARISON TABLE: Layer-Probe @ Last Token, a_frac=-1.0")
print("=" * 80)
print(f"{'Object':<8} {'Model':<12} {'Delta':>7} {'Top':>8} {'H':>6} {'Conf':>6} {'Switch':>8}")
print("-" * 60)

for obj_name in objects:
    for model_name in models:
        path = ROOT / "results" / "phase429b_norm_scaled" / f"{model_name}_phase429b_r1.json"
        if not path.exists():
            continue
        with open(path) as f:
            d = json.load(f)
        
        if obj_name not in d["per_object"]:
            continue
        od = d["per_object"][obj_name]
        base_top = od["baselines"]["category"]["top"]
        
        key = "category_layer_probe"
        if key not in od["perturbations"]:
            continue
        pd = od["perturbations"][key]
        
        # Find deepest layer with a_frac=-1.0 at last position
        if "last" not in pd:
            continue
        
        best_delta = 0
        best_data = None
        for layer, curve in pd["last"].items():
            if "-1.0" in curve:
                c = curve["-1.0"]
                if abs(c["delta"]) > abs(best_delta):
                    best_delta = c["delta"]
                    best_data = (layer, c)
        
        if best_data:
            layer, c = best_data
            switch = "YES" if c["top"] != base_top else "no"
            print(f"{obj_name:<8} {model_name:<12} {c['delta']:>+7.3f} {c['top'][:5]:>8} "
                  f"{c['full_entropy']:>6.1f} {c['confidence']:>6.3f} {switch:>8}")
