"""Phase 274 CRTM Summary: Cross-model comparison."""
import json, numpy as np
from pathlib import Path

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

print("="*80)
print("  PHASE 274 CRTM — CROSS-MODEL COMPARISON")
print("="*80)

# ===== Exp A: Routing Activation Maps =====
print("\n### Exp A: Routing Activation Map — Within vs Between Overlap ###")
print(f"{'Model':<12} {'Within Cos':>11} {'Between Cos':>12} {'Δ Cos':>7} {'Within Jac':>11} {'Between Jac':>12} {'Δ Jac':>7}")
print("-"*75)

for model in ["qwen3", "glm4", "deepseek7b"]:
    path = Path(f"results/phase274_crtm/{model}_exp_a.json")
    if not path.exists():
        continue
    data = load_json(path)
    ws = data.get("within_summary", {})
    bs = data.get("between_summary", {})
    wc = ws.get("cosine_mean", 0)
    bc = bs.get("cosine_mean", 0)
    wj = ws.get("jaccard_mean", 0)
    bj = bs.get("jaccard_mean", 0)
    print(f"{model:<12} {wc:>11.4f} {bc:>12.4f} {wc-bc:>+7.4f} {wj:>11.4f} {bj:>12.4f} {wj-bj:>+7.4f}")

# Per-layer analysis
print("\n### Exp A: Per-Layer Delta (Within-Between) Jaccard ###")
for model in ["qwen3", "glm4", "deepseek7b"]:
    path = Path(f"results/phase274_crtm/{model}_exp_a.json")
    if not path.exists():
        continue
    data = load_json(path)
    pl = data.get("per_layer", {})
    layers = sorted([int(l) for l in pl.keys()])
    deltas = [pl[str(l)]["within_jaccard"] - pl[str(l)]["between_jaccard"] for l in layers]
    # Find peak delta layer
    peak_idx = int(np.argmax(deltas))
    print(f"  {model}: peak delta at L{layers[peak_idx]} ({deltas[peak_idx]:+.4f}), "
          f"mean delta={np.mean(deltas):+.4f}")

# ===== Exp B: Activation Resampling =====
print("\n### Exp B: Activation Resampling (Patching) — Concept Shift ###")
print(f"{'Model':<12} {'Within Peak Shift':>17} {'Between Peak Shift':>19} {'Within Peak L':>14} {'Between Peak L':>15}")
print("-"*80)

for model in ["qwen3", "glm4", "deepseek7b"]:
    path = Path(f"results/phase274_crtm/{model}_exp_b.json")
    if not path.exists():
        continue
    data = load_json(path)
    ws = data.get("within_peak_shift", 0)
    bs = data.get("between_peak_shift", 0)
    wl = data.get("within_peak_layer", 0)
    bl = data.get("between_peak_layer", 0)
    print(f"{model:<12} {ws:>17.4f} {bs:>19.4f} {wl:>14.1f} {bl:>15.1f}")

# Detailed: per-pair concept shift curves
print("\n### Exp B: Detailed Pair Analysis ###")
for model in ["qwen3", "glm4", "deepseek7b"]:
    path = Path(f"results/phase274_crtm/{model}_exp_b.json")
    if not path.exists():
        continue
    data = load_json(path)
    
    print(f"\n  {model}:")
    for pair_type in ["within_pairs", "between_pairs"]:
        pairs = data.get(pair_type, [])
        for p in pairs[:3]:  # Show first 3 pairs
            pair_name = p.get("pair", "?")
            shifts = {l: p[l]["concept_shift"] for l in p if l.isdigit()}
            if not shifts:
                continue
            peak_l = max(shifts, key=lambda l: abs(shifts[l]))
            baseline = p.get("_clean_baseline", {})
            clean_a = baseline.get("clean_logit_A", 0)
            clean_b = baseline.get("clean_logit_B_in_A", 0)
            print(f"    [{pair_type[:6]}] {pair_name}: peak_shift=L{peak_l}({shifts[peak_l]:+.2f}), "
                  f"clean_A={clean_a:.1f}, clean_B_in_A={clean_b:.1f}")

# ===== Exp C: Path Reuse Ratio =====
print("\n### Exp C: Path Reuse Ratio ###")
print(f"{'Model':<12} {'Within Jac':>11} {'Between Jac':>12} {'Δ':>7} {'Within Wt':>10} {'Between Wt':>11} {'Δ':>7}")
print("-"*70)

for model in ["qwen3", "glm4", "deepseek7b"]:
    path = Path(f"results/phase274_crtm/{model}_exp_c.json")
    if not path.exists():
        continue
    data = load_json(path)
    ws = data.get("within_summary", {})
    bs = data.get("between_summary", {})
    wj = ws.get("jaccard_mean", 0)
    bj = bs.get("jaccard_mean", 0)
    ww = ws.get("weighted_reuse_mean", 0)
    bw = bs.get("weighted_reuse_mean", 0)
    print(f"{model:<12} {wj:>11.4f} {bj:>12.4f} {wj-bj:>+7.4f} {ww:>10.4f} {bw:>11.4f} {ww-bw:>+7.4f}")

# Per-layer reuse
print("\n### Exp C: Per-Layer Path Reuse Delta (Within-Between Jaccard) ###")
for model in ["qwen3", "glm4", "deepseek7b"]:
    path = Path(f"results/phase274_crtm/{model}_exp_c.json")
    if not path.exists():
        continue
    data = load_json(path)
    pl = data.get("per_layer", {})
    layers = sorted([int(l) for l in pl.keys()])
    deltas = [pl[str(l)]["delta"] for l in layers]
    peak_idx = int(np.argmax(deltas))
    min_idx = int(np.argmin(deltas))
    print(f"  {model}: max delta at L{layers[peak_idx]} ({deltas[peak_idx]:+.4f}), "
          f"min delta at L{layers[min_idx]} ({deltas[min_idx]:+.4f}), "
          f"mean={np.mean(deltas):+.4f}")

# ===== Exp D: Conditional Routing for Ambiguous Words =====
print("\n### Exp D: Conditional Routing for Ambiguous Words ###")
for model in ["qwen3", "glm4", "deepseek7b"]:
    path = Path(f"results/phase274_crtm/{model}_exp_d.json")
    if not path.exists():
        continue
    data = load_json(path)
    
    print(f"\n  {model}:")
    for r in data:
        c1 = r.get("context1", "")[:30]
        c2 = r.get("context2", "")[:30]
        lm = r.get("layer_metrics", {})
        layers = sorted([int(l) for l in lm.keys()])
        
        # Find peak divergence layer
        cos_dists = [lm[str(l)]["cosine_distance"] for l in layers]
        peak_idx = int(np.argmax(cos_dists))
        peak_layer = layers[peak_idx]
        peak_dist = cos_dists[peak_idx]
        
        # Find min divergence layer (most shared)
        min_idx = int(np.argmin(cos_dists))
        min_layer = layers[min_idx]
        min_dist = cos_dists[min_idx]
        
        print(f"    '{c1}...' vs '{c2}...'")
        print(f"      peak divergence: L{peak_layer} (cos_dist={peak_dist:.4f}), "
              f"most shared: L{min_layer} (cos_dist={min_dist:.4f})")

# ===== Key comparison with Phase 272/273 =====
print("\n" + "="*80)
print("  COMPARISON: Phase 272 (delta_cosine) vs 273 (noise) vs 274 (CRTM)")
print("="*80)
print("""
Phase 272: delta_cosine (layer delta direction consistency)
  - Within > Between: Δ=+0.08 to +0.25 (ALL models consistent)

Phase 273: noise causal map (logit change from noise)
  - Within vs Between: INCONSISTENT across models (Δ=-0.17 to +0.10)

Phase 274: CRTM routing activation (routing fingerprint overlap)
  - Within > Between cosine: Δ=+0.09 to +0.17 (ALL models consistent)
  - Within > Between jaccard: Δ=+0.09 to +0.18 (ALL models consistent)
  - Within > Between weighted_reuse: Δ=+0.10 to +0.16 (ALL models consistent)
""")
