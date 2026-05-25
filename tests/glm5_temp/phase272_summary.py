"""Phase 272 Cross-Model Summary"""
import sys, json, numpy as np
sys.stdout.reconfigure(encoding='utf-8')

model = sys.argv[1] if len(sys.argv) > 1 else "qwen3"

# Load all results
with open(f"results/phase272_path_reuse/{model}_path_overlap.json") as f:
    overlap = json.load(f)
with open(f"results/phase272_path_reuse/{model}_divergence.json") as f:
    divergence = json.load(f)
with open(f"results/phase272_path_reuse/{model}_context_routing.json") as f:
    context = json.load(f)

print(f"\n{'='*70}")
print(f"Phase 272 Summary: {model}")
print(f"{'='*70}")

# === Experiment A: Path Overlap ===
print(f"\n--- Experiment A: Path Overlap (Within vs Between) ---")
print(f"{'Layer':<8} {'Metric':<20} {'Within':>10} {'Between':>10} {'Delta':>10}")
print("-" * 60)
for li_str in sorted(overlap["within_category"].keys(), key=int):
    w = overlap["within_category"][li_str]
    b = overlap["between_category"][li_str]
    for metric in ["head_importance_corr", "delta_cosine", "top_dim_jaccard", "attn_pattern_corr"]:
        wv = w.get(metric, {}).get("mean", 0)
        bv = b.get(metric, {}).get("mean", 0)
        delta = wv - bv
        print(f"L{li_str:<7} {metric:<20} {wv:10.4f} {bv:10.4f} {delta:+10.4f}")

# === Experiment B: Divergence ===
print(f"\n--- Experiment B: Divergence Trajectory ---")
print(f"{'Layer':<8} {'Within cos_dist':>15} {'Between cos_dist':>17} {'Delta':>10}")
print("-" * 55)
for li_str in sorted(divergence["within_cos_trajectory"].keys(), key=int):
    w = divergence["within_cos_trajectory"].get(li_str, {})
    b = divergence["between_cos_trajectory"].get(li_str, {})
    wm = w.get("mean", 0)
    bm = b.get("mean", 0)
    print(f"L{li_str:<7} {wm:15.4f} {bm:17.4f} {bm-wm:+10.4f}")

# Head correlation trajectory
print(f"\n--- Head Importance Correlation (Within vs Between) ---")
print(f"{'Layer':<8} {'Within corr':>12} {'Between corr':>13} {'Delta':>10}")
print("-" * 48)
for li_str in sorted(divergence["within_head_corr_trajectory"].keys(), key=int):
    w = divergence["within_head_corr_trajectory"].get(li_str, {})
    b = divergence["between_head_corr_trajectory"].get(li_str, {})
    wm = w.get("mean", 0)
    bm = b.get("mean", 0)
    print(f"L{li_str:<7} {wm:12.4f} {bm:13.4f} {wm-bm:+10.4f}")

# Divergence layer stats
w_stats = divergence["within_divergence_layer_stats"]
b_stats = divergence["between_divergence_layer_stats"]
print(f"\n--- Divergence Layer ---")
print(f"  Within:  mean={w_stats.get('mean', 'N/A')}, median={w_stats.get('median', 'N/A')}, "
      f"found={w_stats['n_found']}/{w_stats['n_total']}")
print(f"  Between: mean={b_stats.get('mean', 'N/A')}, median={b_stats.get('median', 'N/A')}, "
      f"found={b_stats['n_found']}/{b_stats['n_total']}")

# === Experiment C: Context Routing ===
print(f"\n--- Experiment C: Context-Conditional Routing ---")
print(f"{'Layer':<8} {'cos_dist':>10} {'head_corr':>11} {'delta_cos':>11} {'top_jaccard':>12}")
print("-" * 55)
for li_str in sorted(context["avg_cos_distance"].keys(), key=int):
    cd = context["avg_cos_distance"].get(li_str, {}).get("mean", 0)
    hc = context["avg_head_corr"].get(li_str, {}).get("mean", 0)
    dc = context["avg_delta_cosine"].get(li_str, {}).get("mean", 0)
    tj = context["avg_top_dim_jaccard"].get(li_str, {}).get("mean", 0)
    print(f"L{li_str:<7} {cd:10.4f} {hc:11.4f} {dc:11.4f} {tj:12.4f}")
