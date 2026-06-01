"""
Phase 317 Cross-Model Analysis
================================
"""
import json
import numpy as np
from pathlib import Path

RESULT_DIR = Path("results/phase317_comprehensive")

models = ["qwen3", "glm4", "deepseek7b"]
all_data = {}
for m in models:
    with open(RESULT_DIR / f"{m}_phase317.json", "r", encoding="utf-8") as f:
        all_data[m] = json.load(f)

print("=" * 70)
print("PHASE 317 CROSS-MODEL ANALYSIS")
print("=" * 70)

# ========== TEST 1: Attribute Activation ==========
print("\n" + "=" * 70)
print("TEST 1: Attribute Activation with PARALLEL Templates")
print("=" * 70)

for tmpl_name in ["static", "attribute_probe", "attribute_fill"]:
    print(f"\n--- Template: {tmpl_name} ---")
    print(f"{'Layer':<8}", end="")
    for m in models:
        print(f"{m:<15}", end="")
    print()
    
    # Collect all unique layer keys across models
    all_layer_keys = set()
    for m in models:
        all_layer_keys.update(all_data[m]["test1_attribute_parallel"][tmpl_name].keys())
    
    for lk in sorted(all_layer_keys, key=lambda x: int(x)):
        print(f"L{lk:<6}", end="")
        for m in models:
            data = all_data[m]["test1_attribute_parallel"][tmpl_name].get(lk, {})
            ratio = data.get("ratio_vs_random", 0)
            print(f"{ratio:<15.2f}", end="")
        print()

# Compare with Phase 315 original results
print("\n--- Phase 315 vs Phase 317 Comparison (Qwen3) ---")
print("Phase 315 (15 pairs, cross-template random baseline):")
print("  attribute_fill L2: ratio=5.45 (random=static template!)")
print("  attribute_fill L6: ratio=3.27")
print("Phase 317 (50 pairs, within-template random baseline):")
for tmpl_name in ["static", "attribute_probe", "attribute_fill"]:
    data = all_data["qwen3"]["test1_attribute_parallel"][tmpl_name]
    l2 = data.get("2", {}).get("ratio_vs_random", 0)
    l6 = data.get("6", {}).get("ratio_vs_random", 0)
    print(f"  {tmpl_name} L2: ratio={l2:.2f}, L6: ratio={l6:.2f}")

print("\nCRITICAL: Phase 315's attribute_fill ratio=5.45 was INFLATED by cross-template")
print("random baseline. Within-template baseline shows only 1.1-1.8x.")

# ========== TEST 2: Context × Causal Interaction ==========
print("\n" + "=" * 70)
print("TEST 2: Context × Causal Interaction")
print("=" * 70)

for m in models:
    data = all_data[m]["test2_context_causal"]
    ratios = []
    random_ratios = []
    for pair_key, pair_data in data.items():
        for rl_key, rl_data in pair_data.get("interaction_ratios", {}).items():
            ratios.append(rl_data.get("attr_dir_ratio", 0))
            random_ratios.append(rl_data.get("random_dir_ratio", 0))
    
    mean_ratio = np.mean(ratios) if ratios else 0
    mean_random = np.mean(random_ratios) if random_ratios else 0
    
    print(f"\n{m}:")
    print(f"  Mean attr_dir interaction ratio: {mean_ratio:.3f}")
    print(f"  Mean random_dir interaction ratio: {mean_random:.3f}")
    print(f"  Net context gating (attr - random): {mean_ratio - mean_random:.3f}")
    print(f"  → {'CONFIRMED' if mean_ratio > 1.2 else 'NOT confirmed'} (threshold: 1.2x)")

# Layer-wise analysis
print("\nLayer-wise context gating (Qwen3):")
for rl_key in ["L12", "L18", "L24", "L34"]:
    ratios = []
    for pair_key, pair_data in all_data["qwen3"]["test2_context_causal"].items():
        ir = pair_data.get("interaction_ratios", {}).get(rl_key, {})
        ratios.append(ir.get("attr_dir_ratio", 0))
    if ratios:
        print(f"  {rl_key}: mean={np.mean(ratios):.2f}")

# ========== TEST 3: Negation ==========
print("\n" + "=" * 70)
print("TEST 3: Expanded Negation")
print("=" * 70)

for neg_type in ["regular", "double_negation", "weak_negation"]:
    print(f"\n--- Negation type: {neg_type} ---")
    print(f"{'Metric':<30}", end="")
    for m in models:
        print(f"{m:<15}", end="")
    print()
    
    for m in models:
        type_data = all_data[m]["test3_negation_expanded"][neg_type]
        
        # Collect neg_dir vs random ratios from all pairs and layers
        neg_vs_random = []
        antonym_vs_random = []
        neg_selectivity = []
        
        for pair_key, pair_data in type_data.items():
            for rl_key, rl_data in pair_data.get("neg_dir_causal", {}).items():
                max_neg = max(rl_data.get("target_vs_random", {}).values(), default=0)
                neg_vs_random.append(max_neg)
            for rl_key, rl_data in pair_data.get("random_dir_causal", {}).items():
                max_rand = max(rl_data.get("target_vs_random", {}).values(), default=0)
            # Antonym comparison
            for rl_key, rl_data in pair_data.get("antonym_dir_causal", {}).items():
                max_ant = max(rl_data.get("target_vs_random", {}).values(), default=0)
                antonym_vs_random.append(max_ant)
    
    # Print summary
    for m in models:
        type_data = all_data[m]["test3_negation_expanded"][neg_type]
        neg_dir_norms = []
        ant_dir_norms = []
        neg_selectivities = []
        antonym_selectivities = []
        
        for pair_key, pair_data in type_data.items():
            neg_dir_norms.append(pair_data.get("neg_dir_norm", 0))
            ant_dir_norms.append(pair_data.get("antonym_dir_norm", 0))
            
            # Selectivity from last layer
            last_layer_key = None
            last_layer_num = -1
            for rl_key in pair_data.get("neg_dir_causal", {}):
                rl_num = int(rl_key.replace("L", ""))
                if rl_num > last_layer_num:
                    last_layer_num = rl_num
                    last_layer_key = rl_key
            
            if last_layer_key and last_layer_key in pair_data.get("neg_dir_causal", {}):
                neg_causal = pair_data["neg_dir_causal"][last_layer_key]
                rand_causal = pair_data.get("random_dir_causal", {}).get(last_layer_key, {})
                neg_top = max(neg_causal.get("target_vs_random", {}).values(), default=0)
                rand_top = max(rand_causal.get("target_vs_random", {}).values(), default=1)
                neg_selectivities.append(neg_top / rand_top if rand_top > 0.01 else 0)
            
            if last_layer_key and last_layer_key in pair_data.get("antonym_dir_causal", {}):
                ant_causal = pair_data["antonym_dir_causal"][last_layer_key]
                ant_top = max(ant_causal.get("target_vs_random", {}).values(), default=0)
                antonym_selectivities.append(ant_top)
    
    # Print in a readable format
    for metric_name, get_values in [
        ("neg_dir_norm (mean)", lambda m: [p.get("neg_dir_norm", 0) for p in all_data[m]["test3_negation_expanded"][neg_type].values()]),
        ("antonym_dir_norm (mean)", lambda m: [p.get("antonym_dir_norm", 0) for p in all_data[m]["test3_negation_expanded"][neg_type].values()]),
    ]:
        print(f"{metric_name:<30}", end="")
        for m in models:
            vals = get_values(m)
            print(f"{np.mean(vals):<15.2f}", end="")
        print()

# Double negation special analysis
print("\n--- Double Negation: Does neg_dir activate positive words? ---")
for m in models:
    type_data = all_data[m]["test3_negation_expanded"]["double_negation"]
    print(f"\n{m}:")
    for pair_key, pair_data in type_data.items():
        pos_sent = pair_data["pos"]
        # Get last layer
        last_layer_key = None
        last_layer_num = -1
        for rl_key in pair_data.get("neg_dir_causal", {}):
            rl_num = int(rl_key.replace("L", ""))
            if rl_num > last_layer_num:
                last_layer_num = rl_num
                last_layer_key = rl_key
        
        if last_layer_key and last_layer_key in pair_data.get("neg_dir_causal", {}):
            neg_effects = pair_data["neg_dir_causal"][last_layer_key].get("target_effects", {})
            # For "not bad", check if "okay"/"acceptable" is higher than "bad"
            pos_words = [k for k in neg_effects if k in ["okay", "acceptable", "good", "correct", "right", "possible", "feasible", "happy"]]
            neg_words = [k for k in neg_effects if k in ["bad", "terrible", "poor", "wrong", "incorrect", "impossible", "unhappy", "sad"]]
            
            pos_max = max([neg_effects[k] for k in pos_words], default=0)
            neg_max = max([neg_effects[k] for k in neg_words], default=0)
            
            print(f"  '{pos_sent[:40]}' → pos_words_max={pos_max:.2f}, neg_words_max={neg_max:.2f}, "
                  f"{'MITIGATING' if pos_max > neg_max else 'REINFORCING'}")

print("\n" + "=" * 70)
print("CROSS-MODEL SUMMARY")
print("=" * 70)

print("""
Test 1 - Attribute Activation (PARALLEL templates):
  Phase 315's attribute_fill ratio=5.45 was INFLATED by cross-template comparison.
  Within-template baseline shows:
    Qwen3: 1.1-1.8x (modest, increasing with depth)
    GLM4:   1.0-1.5x (very weak, early layers < 1)
    DS7B:   1.1-1.5x (modest, increasing with depth)
  CONCLUSION: Attribute context makes ALL representations closer,
    not specifically attribute-related pairs. The real effect is modest.

Test 2 - Context × Causal Interaction:
  Qwen3: mean=1.20 (borderline, NOT confirmed)
  GLM4:   mean=1.14 (NOT confirmed)
  DS7B:   mean=1.26 (CONFIRMED at 1.2x threshold)
  CONCLUSION: Context gating is MODEL-SPECIFIC and PAIR-SPECIFIC.
    Not a universal mechanism. Some pairs show strong amplification,
    others show suppression. Average effect is weak.

Test 3 - Negation:
  1. Antonym direction is consistently STRONGER than negation direction.
  2. neg_dir selectivity vs random is modest (1.3-2.9x).
  3. Double negation ("not bad") primarily REINFORCES the negated word,
     not mitigating it. Only Qwen3 shows occasional mitigation.
  4. Neg_dir_norm scales dramatically:
     GLM4: 2-4, Qwen3: 17-38, DS7B: 47-116
  CONCLUSION: Negation direction is "semantic polarity shift", not
    "logical negation operator". It pushes toward negative/neutral
    territory but doesn't reliably perform logical negation.
""")
