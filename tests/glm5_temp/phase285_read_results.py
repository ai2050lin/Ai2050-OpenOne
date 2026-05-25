"""Read and display Phase 285 results across models."""
import json, sys

models = ["qwen3", "glm4", "deepseek7b"]
for m in models:
    path = f"results/phase285_real_patching/{m}_real_patching.json"
    with open(path) as f:
        d = json.load(f)
    
    pcl = d["per_component_layer"]
    print(f"\n{'='*60}")
    print(f"{m.upper()}: {d['n_pairs_tested']} pairs, {d['total_time_s']:.0f}s")
    print(f"  Hooks: attn={d['self_attn_hook_worked']}, mlp={d['mlp_hook_worked']}")
    print(f"  {'L':>4} {'attn':>8} {'mlp':>8} {'resid':>8} {'attn/resid':>10} {'mlp/resid':>10}")
    print(f"  {'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10}")
    
    for lk in sorted(pcl["attn"].keys(), key=lambda x: int(x)):
        a = pcl["attn"][lk]["mean"]
        ml = pcl["mlp"][lk]["mean"]
        r = pcl["resid"][lk]["mean"]
        ar = a / max(r, 1e-10)
        mr = ml / max(r, 1e-10)
        print(f"  L{int(lk):>3} {a:8.3f} {ml:8.3f} {r:8.3f} {ar:10.3f} {mr:10.3f}")
    
    # Compute averages
    attn_vals = [pcl["attn"][lk]["mean"] for lk in pcl["attn"]]
    mlp_vals = [pcl["mlp"][lk]["mean"] for lk in pcl["mlp"]]
    resid_vals = [pcl["resid"][lk]["mean"] for lk in pcl["resid"]]
    
    avg_attn = sum(attn_vals) / len(attn_vals)
    avg_mlp = sum(mlp_vals) / len(mlp_vals)
    avg_resid = sum(resid_vals) / len(resid_vals)
    
    print(f"  {'AVG':>4} {avg_attn:8.3f} {avg_mlp:8.3f} {avg_resid:8.3f} {avg_attn/avg_resid:10.3f} {avg_mlp/avg_resid:10.3f}")
