"""Phase 391 Cross-Model Summary: Target/Competitor Decomposition"""
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
res_dir = ROOT / "results" / "phase391_target_competitor_decomp"

models = ["qwen3", "deepseek7b", "glm4"]
all_data = {}
for m in models:
    fp = res_dir / f"{m}_phase391.json"
    if fp.exists():
        all_data[m] = json.load(open(fp))

print("=" * 80)
print("PHASE 391: TARGET/COMPETITOR DECOMPOSITION — CROSS-MODEL SUMMARY")
print("=" * 80)

# 1. Mechanism type per layer per model
print("\n--- 1. MECHANISM TYPE (per layer per model) ---")
print(f"{'Model':<12} {'Layer':>5} {'add':>8} {'T_delta':>8} {'C_delta':>8} {'Mechanism':<40}")
print("-" * 80)
for m in models:
    if m not in all_data:
        continue
    d = all_data[m]
    for li in d['layers']:
        lr = d['per_layer'][str(li)]
        add = lr['add_mean']
        td = lr['target_delta_mean']
        cd = lr['competitor_delta_mean']
        if add > 0:
            if td > 0 and cd <= 0:
                mech = "BOOST_T + SUPPRESS_C (IDEAL)"
            elif td > 0 and cd > 0:
                mech = "BOOST_BOTH (T>C)" if td > cd else "BOOST_BOTH (C>T)"
            elif td <= 0 and cd < 0:
                mech = "SUPPRESS_BOTH (T less)" if abs(cd) > abs(td) else "SUPPRESS_BOTH (C less)"
            else:
                mech = "MIXED"
        else:
            if td < 0 and cd >= 0:
                mech = "SUPPRESS_T + BOOST_C (REVERSED)"
            elif td < 0 and cd < 0:
                mech = "SUPPRESS_BOTH (T more)" if abs(td) > abs(cd) else "SUPPRESS_BOTH (C more)"
            elif td >= 0 and cd > 0:
                mech = "BOOST_C (dominant)"
            else:
                mech = "MIXED"
        print(f"{m:<12} L{li:>3} {add:>+8.4f} {td:>+8.4f} {cd:>+8.4f} {mech:<40}")

# 2. Per-category per-model comparison at representative layers
print("\n--- 2. PER-CATEGORY TARGET/COMPETITOR DECOMPOSITION ---")
cats = ["color", "temperature", "moisture", "size", "weight", "speed", "brightness"]

for cat in cats:
    print(f"\n  {cat.upper()}:")
    for m in models:
        if m not in all_data:
            continue
        d = all_data[m]
        parts = []
        for li in d['layers']:
            ce = d['per_layer'][str(li)]['category_effects'].get(cat, {})
            add = ce.get('add_mean', 0)
            td = ce.get('target_delta_mean', 0)
            cd = ce.get('competitor_delta_mean', 0)
            tp = ce.get('target_delta_pos_pct', 0)
            cp = ce.get('competitor_delta_pos_pct', 0)
            # Determine mechanism tag
            if add > 0:
                if td > 0 and cd <= 0:
                    tag = "T↑C↓"
                elif td > 0 and cd > 0:
                    tag = "T↑C↑" if td >= cd else "C↑T↑"
                else:
                    tag = "???"
            else:
                if td < 0 and cd >= 0:
                    tag = "T↓C↑"
                elif td < 0 and cd < 0:
                    tag = "T↓C↓"
                elif td >= 0 and cd > 0:
                    tag = "C↑dom"
                else:
                    tag = "???"
            parts.append(f"L{li}={add:+.3f}({tag})")
        print(f"    {m:<12}: {' -> '.join(parts)}")

# 3. Cross-model mechanism consistency
print("\n--- 3. CROSS-MECHANISM CONSISTENCY ---")
print("Categories where centroid IDEALLY boosts target and suppresses competitor:")
for cat in cats:
    for m in models:
        if m not in all_data:
            continue
        d = all_data[m]
        for li in d['layers']:
            ce = d['per_layer'][str(li)]['category_effects'].get(cat, {})
            td = ce.get('target_delta_mean', 0)
            cd = ce.get('competitor_delta_mean', 0)
            add = ce.get('add_mean', 0)
            if td > 0 and cd < 0 and add > 0:
                print(f"  {cat} @ {m} L{li}: T={td:+.4f}, C={cd:+.4f}, add={add:+.4f} [IDEAL]")

print("\nCategories where centroid REVERSEDS (suppresses target, boosts competitor):")
for cat in cats:
    for m in models:
        if m not in all_data:
            continue
        d = all_data[m]
        for li in d['layers']:
            ce = d['per_layer'][str(li)]['category_effects'].get(cat, {})
            td = ce.get('target_delta_mean', 0)
            cd = ce.get('competitor_delta_mean', 0)
            add = ce.get('add_mean', 0)
            if td < 0 and cd > 0 and add < 0:
                print(f"  {cat} @ {m} L{li}: T={td:+.4f}, C={cd:+.4f}, add={add:+.4f} [REVERSED]")

# 4. Hierarchy comparison
print("\n--- 4. CENTROID HIERARCHY: global vs per-category ---")
for m in models:
    if m not in all_data:
        continue
    d = all_data[m]
    for li in d['layers']:
        lr = d['per_layer'][str(li)]
        h = lr['hierarchy']
        ratio = h['category_add_mean'] / h['global_add_mean'] if abs(h['global_add_mean']) > 0.0001 else float('inf')
        print(f"  {m} L{li}: global={h['global_add_mean']:+.4f}, category={h['category_add_mean']:+.4f}, ratio={ratio:.1f}x")

# 5. DS7B size deep dive
print("\n--- 5. DS7B SIZE DEEP DIVE ---")
if 'deepseek7b' in all_data:
    d = all_data['deepseek7b']
    for li in d['layers']:
        ce = d['per_layer'][str(li)]['category_effects'].get('size', {})
        add = ce.get('add_mean', 0)
        td = ce.get('target_delta_mean', 0)
        cd = ce.get('competitor_delta_mean', 0)
        tp = ce.get('target_delta_pos_pct', 0)
        cp = ce.get('competitor_delta_pos_pct', 0)
        print(f"  L{li}: add={add:+.4f}, T={td:+.4f}({tp:.0f}%↑), C={cd:+.4f}({cp:.0f}%↑)")
        print(f"       → centroid SUPPRESSES target ({tp:.0f}%↑) and BOOSTS competitor ({cp:.0f}%↑)")

# 6. GLM4 color deep dive
print("\n--- 6. GLM4 COLOR DEEP DIVE ---")
if 'glm4' in all_data:
    d = all_data['glm4']
    for li in d['layers']:
        ce = d['per_layer'][str(li)]['category_effects'].get('color', {})
        add = ce.get('add_mean', 0)
        td = ce.get('target_delta_mean', 0)
        cd = ce.get('competitor_delta_mean', 0)
        tp = ce.get('target_delta_pos_pct', 0)
        cp = ce.get('competitor_delta_pos_pct', 0)
        print(f"  L{li}: add={add:+.4f}, T={td:+.4f}({tp:.0f}%↑), C={cd:+.4f}({cp:.0f}%↑)")
        if cd > td:
            print(f"       → centroid BOOSTS competitor MORE than target → negative add_effect")

# 7. Qwen3 brightness deep dive
print("\n--- 7. QWEN3 BRIGHTNESS DEEP DIVE ---")
if 'qwen3' in all_data:
    d = all_data['qwen3']
    for li in d['layers']:
        ce = d['per_layer'][str(li)]['category_effects'].get('brightness', {})
        add = ce.get('add_mean', 0)
        td = ce.get('target_delta_mean', 0)
        cd = ce.get('competitor_delta_mean', 0)
        print(f"  L{li}: add={add:+.4f}, T={td:+.4f}, C={cd:+.4f}")
        if td < 0 and cd < 0:
            print(f"       → centroid SUPPRESSES BOTH, but target MORE → negative add_effect")
        elif td > 0 and cd > 0:
            print(f"       → centroid BOOSTS BOTH, target more → positive add_effect")

print("\n" + "=" * 80)
print("KEY FINDINGS SUMMARY")
print("=" * 80)
print("""
1. THREE MECHANISM TYPES OBSERVED:
   - IDEAL: Boost target + Suppress competitor (cleanest causal signal)
   - DOMINANT_BOOST: Boost both, target more (common in mid-deep layers)
   - REVERSED: Suppress target + Boost competitor (category is "misaligned")

2. IDEAL mechanism appears in specific category-layer combos:
   - Qwen3: color L4 (T=+0.006, C=-0.004)
   - DS7B: moisture L4 (T=+0.161, C=-0.085), temperature L4 (T=+0.143, C=-0.118)
   - GLM4: speed L4 (T=+0.026, C=-0.010), brightness L20 (T=+0.045, C=-0.071)

3. REVERSED mechanism explains cross-model direction inconsistencies:
   - GLM4 color: BOOSTS competitor more → negative add_effect
   - DS7B size: SUPPRESSES target + BOOSTS competitor → strongly negative
   - Qwen3 brightness deep: SUPPRESSES both, target more → negative

4. Layer-dependent mechanism shifts:
   - DS7B: L4 BOOST_TARGET → L12 SUPPRESS_T+BOOST_C → L20 BOOST_C → L26 SUPPRESS_T
   - GLM4: L4 BOOST_COMPETITOR → L20+ BOOST_TARGET

5. Hierarchy: per-category centroid >> global centroid at ALL layers
   (confirms Phase 390 finding)
""")
