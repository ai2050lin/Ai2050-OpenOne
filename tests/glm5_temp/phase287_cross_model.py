"""Cross-model analysis for Phase 287: Route-Content Separation."""

import json, numpy as np
from pathlib import Path
from collections import defaultdict

RESULT_DIR = Path("results/phase287_route_content")

print("=" * 70)
print("Phase 287: Cross-Model Route-Content Separation Analysis")
print("=" * 70)

all_data = {}
for model_name in ["qwen3", "glm4", "deepseek7b"]:
    path = RESULT_DIR / f"{model_name}_route_content.json"
    if path.exists():
        all_data[model_name] = json.load(open(path))
        print(f"\nLoaded {model_name}: {path}")

# ========== Head-level comparison ==========
print(f"\n{'='*70}")
print("HEAD-LEVEL ROUTE vs CONTENT COMPARISON")
print(f"{'='*70}")
print(f"{'Model':>12} {'Head':>12} {'Full_A':>8} {'Routing':>8} {'Content':>8} {'R/C':>6} {'Interpretation':>20}")
for model_name in ["qwen3", "glm4", "deepseek7b"]:
    if model_name not in all_data:
        continue
    data = all_data[model_name]
    for hlabel, pa in sorted(data["per_head_analysis"].items()):
        rr = pa["mean_routing_ratio"]
        cr = pa["mean_content_ratio"]
        fa = pa["mean_full_A_ratio"]
        rc = rr / cr if cr > 0.001 else float('inf')
        print(f"{model_name:>12} {hlabel:>12} {fa:8.3f} {rr:8.3f} {cr:8.3f} {rc:6.2f} {pa['interpretation']:>20}")

# ========== Category-level breakthrough ==========
print(f"\n{'='*70}")
print("CATEGORY-LEVEL ROUTE vs CONTENT BREAKTHROUGH FINDINGS")
print(f"{'='*70}")

# Key finding: NEGATION is content-dominant in DS7B
for model_name in ["qwen3", "glm4", "deepseek7b"]:
    if model_name not in all_data:
        continue
    data = all_data[model_name]
    print(f"\n--- {model_name.upper()} ---")
    if "per_category_analysis" in data:
        for cat in sorted(data["per_category_analysis"].keys()):
            ca = data["per_category_analysis"][cat]
            rr = ca["routing"]
            cr = ca["content"]
            rc = rr / cr if cr > 0.001 else float('inf')
            bias = "ROUTING-biased" if rc > 1.5 else ("CONTENT-biased" if rc < 0.67 else "BALANCED")
            print(f"  {cat:<18} routing={rr:.3f} content={cr:.3f} R/C={rc:.2f} → {bias}")

# ========== Cross-model routing vs content trend ==========
print(f"\n{'='*70}")
print("CROSS-MODEL SUMMARY")
print(f"{'='*70}")

for model_name in ["qwen3", "glm4", "deepseek7b"]:
    if model_name not in all_data:
        continue
    data = all_data[model_name]
    routing_vals = []
    content_vals = []
    for hlabel, pa in data["per_head_analysis"].items():
        routing_vals.append(pa["mean_routing_ratio"])
        content_vals.append(pa["mean_content_ratio"])
    
    mean_r = np.mean(routing_vals)
    mean_c = np.mean(content_vals)
    std_r = np.std(routing_vals)
    std_c = np.std(content_vals)
    
    print(f"\n{model_name}:")
    print(f"  Routing  mean={mean_r:.3f} ± {std_r:.3f}")
    print(f"  Content  mean={mean_c:.3f} ± {std_c:.3f}")
    print(f"  R/C ratio mean={mean_r/mean_c:.2f}")
    
    # Count routing-dominant vs content-dominant categories
    if "per_category_analysis" in data:
        routing_dom = 0
        content_dom = 0
        balanced = 0
        for cat, ca in data["per_category_analysis"].items():
            rr = ca["routing"]
            cr = ca["content"]
            rc = rr / cr if cr > 0.001 else float('inf')
            if rc > 1.5:
                routing_dom += 1
            elif rc < 0.67:
                content_dom += 1
            else:
                balanced += 1
        print(f"  Categories: {routing_dom} routing-dominant, {content_dom} content-dominant, {balanced} balanced")

# ========== KEY INSIGHT: NEGATION DIFFERENCE ==========
print(f"\n{'='*70}")
print("KEY INSIGHT: NEGATION IS CONTENT-DOMINANT (DS7B)")
print(f"{'='*70}")
for model_name in ["qwen3", "glm4", "deepseek7b"]:
    if model_name not in all_data:
        continue
    data = all_data[model_name]
    if "per_category_analysis" in data and "negation" in data["per_category_analysis"]:
        neg = data["per_category_analysis"]["negation"]
        rr = neg["routing"]
        cr = neg["content"]
        print(f"  {model_name}: negation routing={rr:.3f} content={cr:.3f} content/routing={cr/rr:.2f}x")
    else:
        print(f"  {model_name}: no negation data")

# ========== TRANSLATION comparison ==========
print(f"\n{'='*70}")
print("TRANSLATION ROUTING vs CONTENT")
print(f"{'='*70}")
for model_name in ["qwen3", "glm4", "deepseek7b"]:
    if model_name not in all_data:
        continue
    data = all_data[model_name]
    if "per_category_analysis" in data and "translation" in data["per_category_analysis"]:
        trans = data["per_category_analysis"]["translation"]
        rr = trans["routing"]
        cr = trans["content"]
        print(f"  {model_name}: routing={rr:.3f} content={cr:.3f} R/C={rr/cr:.2f}")

print(f"\n{'='*70}")
print("Analysis complete!")
