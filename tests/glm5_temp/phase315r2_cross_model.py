"""
Phase 315-R2 Cross-Model Analysis
==================================
Analyze Phase 315-R2 confirmation results across Qwen3, GLM4, DS7B.
"""
import json, sys
import numpy as np
from pathlib import Path

RESULT_DIR = Path("results/phase315r2_confirm")
models = ["qwen3", "glm4", "deepseek7b"]

all_data = {}
for m in models:
    fp = RESULT_DIR / f"{m}_confirm.json"
    if fp.exists():
        with open(fp, "r", encoding="utf-8") as f:
            all_data[m] = json.load(f)

print("=" * 80)
print("PHASE 315-R2 CROSS-MODEL ANALYSIS")
print("=" * 80)

# =====================================================================
# TEST 1: Attribute Confirmation - Cross Model
# =====================================================================
print("\n" + "=" * 80)
print("TEST 1: Attribute Context Activation (50+ pairs)")
print("=" * 80)

print("\n--- Attribute Fill Ratio vs Random (lower = attribute pairs FURTHER apart) ---")
print(f"{'Layer':<8}", end="")
for m in models:
    print(f"{m:<16}", end="")
print()
print("-" * 56)

# Get common layer indices
all_layers = set()
for m in models:
    all_layers.update(all_data[m]["test1_attribute_confirmation"].keys())
for li in sorted(all_layers, key=int):
    print(f"L{li:<6}", end="")
    for m in models:
        d = all_data[m]["test1_attribute_confirmation"].get(li, {})
        r = d.get("attribute_fill_ratio_vs_random", "N/A")
        if r != "N/A":
            print(f"{r:<16.3f}", end="")
        else:
            print(f"{'N/A':<16}", end="")
    print()

print("\n--- Static Ratio vs Random (should be ~1.0) ---")
print(f"{'Layer':<8}", end="")
for m in models:
    print(f"{m:<16}", end="")
print()
for li in sorted(all_layers, key=int):
    print(f"L{li:<6}", end="")
    for m in models:
        d = all_data[m]["test1_attribute_confirmation"].get(li, {})
        r = d.get("static_ratio_vs_random", "N/A")
        if r != "N/A":
            print(f"{r:<16.3f}", end="")
        else:
            print(f"{'N/A':<16}", end="")
    print()

print("\n--- Raw Distance Comparison (Qwen3) ---")
q3 = all_data["qwen3"]["test1_attribute_confirmation"]
for li in sorted(q3.keys(), key=int):
    d = q3[li]
    print(f"  L{li}: static={d['static_mean_dist']:.4f}, attr_fill={d['attribute_fill_mean_dist']:.4f}, "
          f"random={d['random_mean_dist']:.4f}")

# =====================================================================
# TEST 2: Function Template Comparison
# =====================================================================
print("\n" + "=" * 80)
print("TEST 2: Function Template Comparison")
print("=" * 80)

for m in models:
    print(f"\n--- {m} ---")
    func_data = all_data[m]["test2_function_templates"]
    for li in sorted(func_data.keys(), key=int):
        best_tmpl = ""
        best_ratio = 0
        for tmpl_name, td in func_data[li].items():
            r = td["ratio_vs_random"]
            if r > best_ratio:
                best_ratio = r
                best_tmpl = tmpl_name
        # Show all templates for this layer
        tmpl_strs = []
        for tmpl_name in ["static", "function_designed", "function_purpose", "function_using", "function_tool"]:
            if tmpl_name in func_data[li]:
                r = func_data[li][tmpl_name]["ratio_vs_random"]
                tmpl_strs.append(f"{tmpl_name.replace('function_', '')}={r:.2f}")
        print(f"  L{li}: {', '.join(tmpl_strs)}  | best={best_tmpl}({best_ratio:.2f})")

# =====================================================================
# TEST 3: Negation Direction Causal Test
# =====================================================================
print("\n" + "=" * 80)
print("TEST 3: Negation Direction Causal Test")
print("=" * 80)

for m in models:
    print(f"\n--- {m} ---")
    neg_data = all_data[m]["test3_negation_causal"]
    neg_dir_norms = []
    for pair_key, pd in neg_data.items():
        neg_dir_norms.append(pd["neg_dir_norm"])
        print(f"  '{pair_key}': neg_dir_norm={pd['neg_dir_norm']:.2f}")
        
        # Show hook causal: max neg_vs_random across read layers
        for li_key, li_data in pd.get("hook_causal", {}).items():
            neg_effects = li_data.get("neg_vs_random", {})
            if neg_effects:
                best_neg = max(neg_effects.items(), key=lambda x: x[1])
                print(f"    {li_key}: best_neg_effect={best_neg[0]}({best_neg[1]:.2f}x random)")
    
    avg_norm = np.mean(neg_dir_norms)
    print(f"  >> Average neg_dir_norm: {avg_norm:.2f}")

# =====================================================================
# KEY CROSS-MODEL COMPARISONS
# =====================================================================
print("\n" + "=" * 80)
print("KEY CROSS-MODEL FINDINGS")
print("=" * 80)

# 1. Negation direction norm comparison
print("\n1. Negation Direction Norm (model scale):")
for m in models:
    neg_data = all_data[m]["test3_negation_causal"]
    norms = [pd["neg_dir_norm"] for pd in neg_data.values()]
    print(f"   {m}: avg={np.mean(norms):.2f}, range=[{min(norms):.2f}, {max(norms):.2f}]")

# 2. Best negation causal effect at mid-layers
print("\n2. Best Negation Causal Effect (mid-layer):")
for m in models:
    neg_data = all_data[m]["test3_negation_causal"]
    all_neg_effects = []
    for pd in neg_data.values():
        for li_key, li_data in pd.get("hook_causal", {}).items():
            neg_effects = li_data.get("neg_vs_random", {})
            for tok, val in neg_effects.items():
                if tok != "not":  # exclude "not" token itself
                    all_neg_effects.append(val)
    print(f"   {m}: max={max(all_neg_effects):.2f}x, mean={np.mean(all_neg_effects):.2f}x")

# 3. Function template: best template per model
print("\n3. Best Function Template (L6-L12 avg):")
for m in models:
    func_data = all_data[m]["test2_function_templates"]
    tmpl_avg = {}
    for li in ["6", "12"]:
        if li in func_data:
            for tmpl_name, td in func_data[li].items():
                if tmpl_name.startswith("function_"):
                    tmpl_avg[tmpl_name] = tmpl_avg.get(tmpl_name, []) + [td["ratio_vs_random"]]
    for t, vals in tmpl_avg.items():
        tmpl_avg[t] = np.mean(vals)
    best = max(tmpl_avg.items(), key=lambda x: x[1])
    print(f"   {m}: {best[0]} (avg ratio={best[1]:.2f})")

# 4. Attribute fill ratio - comparison across models
print("\n4. Attribute Fill Ratio (attr_fill_ratio_vs_random):")
print("   NOTE: R2 test design differs from Phase 315 - uses non-parallel sentences")
print("   ratio < 1 means attr_fill pairs FURTHER than random (not closer)")
for m in models:
    attr_data = all_data[m]["test1_attribute_confirmation"]
    ratios = [d["attribute_fill_ratio_vs_random"] for d in attr_data.values()]
    print(f"   {m}: L2={ratios[0]:.3f}, peak_L12={min(ratios):.3f}")

# 5. Negation effect by read layer
print("\n5. Negation Causal Effect by Read Layer (avg across pairs, excl. 'not'):")
for m in models:
    neg_data = all_data[m]["test3_negation_causal"]
    layer_effects = {}
    for pd in neg_data.values():
        for li_key, li_data in pd.get("hook_causal", {}).items():
            neg_effects = li_data.get("neg_vs_random", {})
            semantic_neg = [v for k, v in neg_effects.items() if k != "not"]
            if semantic_neg:
                layer_effects.setdefault(li_key, []).append(np.mean(semantic_neg))
    print(f"   {m}:", end="")
    for li in sorted(layer_effects.keys()):
        print(f" {li}={np.mean(layer_effects[li]):.2f}x", end="")
    print()

print("\n" + "=" * 80)
print("DESIGN ISSUE NOTE")
print("=" * 80)
print("""
Phase 315-R2 Test 1 (Attribute Confirmation) has a design flaw:
- attr_fill sentences: "The apple has the quality of being red" vs "The red is a quality"
  These are NON-PARALLEL sentences (different structures), causing artificially large distance.
- static sentences: "the apple was there" vs "the red was there"
  These are PARALLEL sentences (same structure, different words).
- The ratio comparison is invalid because the baseline structures differ.

Phase 315 original used parallel templates:
- "The apple is usually ___" vs "The red is usually ___" (same template, different fill)
- This properly measures whether attribute activation makes the pair closer.

The R2 attr_fill_ratio < 1 reflects the non-parallel sentence design, NOT actual
attribute deactivation. This must be redesigned in future tests.
""")

print("Done.")
