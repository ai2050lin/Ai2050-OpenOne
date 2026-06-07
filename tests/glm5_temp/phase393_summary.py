"""Phase 393 Summary: Cross-model hierarchy comparison"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
results_dir = ROOT / "results" / "phase393_centroid_hierarchy"

models = ["qwen3", "deepseek7b", "glm4"]
levels = ["L0_global", "L1_category", "L2_obj_cat", "L3_pair"]
categories = ["brightness", "color", "moisture", "size", "speed", "temperature", "weight"]

def np_sign(x):
    if x > 0: return 1
    elif x < 0: return -1
    return 0

all_data = {}
for m in models:
    path = results_dir / f"{m}_phase393.json"
    if path.exists():
        with open(path) as f:
            all_data[m] = json.load(f)

print("=" * 80)
print("Phase 393: Conditional Centroid Hierarchy + T/C Decomposition — Cross-Model Summary")
print("=" * 80)

# 1. Overall hierarchy comparison
print("\n### 1. Overall add_effect across hierarchy levels ###")
for m in models:
    if m not in all_data:
        continue
    data = all_data[m]
    print(f"\n  {m}:")
    for li_str, lr in data['per_layer'].items():
        print(f"    Layer {li_str}:")
        for level in levels:
            hr = lr['hierarchy_correct'].get(level, {})
            add = hr.get('add_mean', 0)
            td = hr.get('target_delta_mean', 0)
            cd = hr.get('competitor_delta_mean', 0)
            mech = hr.get('mechanism', '?')
            ideal_pct = 0
            cat_data = hr.get('category_breakdown', {})
            if cat_data:
                ideal_count = sum(1 for c in cat_data.values() if c.get('mechanism') == 'IDEAL')
                ideal_pct = ideal_count / len(cat_data) * 100
            print(f"      {level:12s}: add={add:+.4f}, T={td:+.4f}, C={cd:+.4f}, "
                  f"IDEAL={ideal_pct:.0f}%, mech={mech}")

# 2. IDEAL ratio across hierarchy levels (KEY PREDICTION TEST)
print("\n### 2. IDEAL Ratio Across Hierarchy Levels (Key Prediction) ###")
print("  Prediction: L0 < L1 < L2 < L3 in IDEAL ratio")
for m in models:
    if m not in all_data:
        continue
    data = all_data[m]
    for li_str, lr in data['per_layer'].items():
        ideal_pcts = []
        for level in levels:
            hr = lr['hierarchy_correct'].get(level, {})
            cat_data = hr.get('category_breakdown', {})
            if cat_data:
                ideal_count = sum(1 for c in cat_data.values() if c.get('mechanism') == 'IDEAL')
                ideal_pcts.append(f"{ideal_count}/{len(cat_data)}")
            else:
                ideal_pcts.append("N/A")
        print(f"  {m} L{li_str}: {' -> '.join(ideal_pcts)}")

# 3. Selectivity across hierarchy (target selectivity = T-C difference)
print("\n### 3. Selectivity (|T_delta - C_delta|) Across Hierarchy ###")
for m in models:
    if m not in all_data:
        continue
    data = all_data[m]
    for li_str, lr in data['per_layer'].items():
        sels = []
        for level in levels:
            hr = lr['hierarchy_correct'].get(level, {})
            td = hr.get('target_delta_mean', 0)
            cd = hr.get('competitor_delta_mean', 0)
            sel = abs(td - cd)
            sels.append(f"{sel:.4f}")
        print(f"  {m} L{li_str}: {' -> '.join(sels)}")

# 4. SYMMETRIC check
print("\n### 4. SYMMETRIC Check (L1 category, correct vs incorrect) ###")
for m in models:
    if m not in all_data:
        continue
    data = all_data[m]
    for li_str, lr in data['per_layer'].items():
        sym_count = lr.get('symmetric_count', 0)
        sym_total = lr.get('symmetric_total', 0)
        pct = sym_count / sym_total * 100 if sym_total > 0 else 0
        print(f"  {m} L{li_str}: SYMMETRIC {sym_count}/{sym_total} ({pct:.0f}%)")

        # Show category-level detail
        for cat in categories:
            hr_cor = lr['hierarchy_correct']['L1_category']['category_breakdown'].get(cat, {})
            hr_inc = lr['hierarchy_incorrect']['L1_category']['category_breakdown'].get(cat, {})
            add_cor = hr_cor.get('add_mean', 0)
            add_inc = hr_inc.get('add_mean', 0)
            mech_cor = hr_cor.get('mechanism', '?')
            mech_inc = hr_inc.get('mechanism', '?')
            is_sym = "SYM" if (np_sign(add_cor) != np_sign(add_inc) and add_cor != 0 and add_inc != 0) else "ASYM"
            print(f"    {cat:12s}: cor={add_cor:+.4f}({mech_cor}) inc={add_inc:+.4f}({mech_inc}) {is_sym}")

# 5. Key categories showing hierarchy improvement
print("\n### 5. Categories Showing IDEAL at Deeper Hierarchy Levels ###")
for m in models:
    if m not in all_data:
        continue
    data = all_data[m]
    for li_str, lr in data['per_layer'].items():
        print(f"\n  {m} L{li_str}:")
        for cat in categories:
            mechs = []
            for level in levels:
                hr = lr['hierarchy_correct'].get(level, {})
                cat_data = hr.get('category_breakdown', {}).get(cat, {})
                mech = cat_data.get('mechanism', '?')
                add = cat_data.get('add_mean', 0)
                td = cat_data.get('target_delta_mean', 0)
                cd = cat_data.get('competitor_delta_mean', 0)
                mechs.append(f"{mech}")
            # Highlight if later levels show IDEAL but earlier don't
            has_late_ideal = any(mechs[i] == 'IDEAL' for i in range(2, 4))
            has_early_ideal = any(mechs[i] == 'IDEAL' for i in range(0, 2))
            marker = ""
            if has_late_ideal and not has_early_ideal:
                marker = " [HIERARCHY IMPROVEMENT]"
            elif has_late_ideal and has_early_ideal:
                marker = " [CONSISTENT IDEAL]"
            print(f"    {cat:12s}: {' -> '.join(mechs)}{marker}")

# 6. GLM4 special case: L2_obj_cat has most IDEAL
print("\n### 6. GLM4 L2_obj_cat Analysis (Unexpected Leader) ###")
if 'glm4' in all_data:
    data = all_data['glm4']
    for li_str, lr in data['per_layer'].items():
        hr = lr['hierarchy_correct'].get('L2_obj_cat', {})
        cat_data = hr.get('category_breakdown', {})
        for cat in categories:
            ce = cat_data.get(cat, {})
            if ce.get('mechanism') == 'IDEAL':
                print(f"  GLM4 L{li_str} L2_obj_cat {cat}: "
                      f"add={ce['add_mean']:+.4f}, T={ce['target_delta_mean']:+.4f}, "
                      f"C={ce['competitor_delta_mean']:+.4f}")

print("\n" + "=" * 80)
print("Phase 393 Summary Complete")
