"""Phase 315 Cross-Model Analysis: Context Activation + Causal Relation"""
import sys, json, numpy as np
from pathlib import Path

RESULT_DIR = Path("results/phase315_context_causal")

models = ["qwen3", "glm4", "deepseek7b"]
all_data = {}
for m in models:
    fp = RESULT_DIR / f"{m}_context_causal.json"
    if fp.exists():
        with open(fp, "r", encoding="utf-8") as f:
            all_data[m] = json.load(f)

print("=" * 80)
print("PHASE 315 CROSS-MODEL ANALYSIS")
print("=" * 80)

# ============================================================
# Part A: Context Activation - Key Findings
# ============================================================
print("\n" + "=" * 80)
print("PART A: CONTEXT ACTIVATION ANALYSIS")
print("=" * 80)

# 1. Attribute relation: static vs context
print("\n--- 1. Attribute Relation: Context Activation ---")
print(f"{'Model':<12} {'L2 Static':>10} {'L2 AttrFill':>12} {'L2 Ratio':>10} {'L6 Static':>10} {'L6 AttrFill':>12} {'L6 Ratio':>10}")
for m in models:
    if m not in all_data:
        continue
    part_a = all_data[m].get("part_a_context_activation", {})
    attr = part_a.get("attribute", {})
    layers = attr.get("layers", {})
    for li in ["2", "6"]:
        li_data = layers.get(li, {})
        static = li_data.get("contexts", {}).get("static", {}).get("mean_dist", 0)
        fill = li_data.get("contexts", {}).get("attribute_fill", {}).get("mean_dist", 0)
        ratio = li_data.get("contexts", {}).get("attribute_fill", {}).get("ratio_vs_random", 0)
        print(f"{m:<12} {static:>10.3f} {fill:>12.3f} {ratio:>10.2f}", end="")
    print()

# 2. Function relation: static vs context
print("\n--- 2. Function Relation: Context Activation ---")
print(f"{'Model':<12} {'L2 Static':>10} {'L2 FuncFill':>12} {'L2 Ratio':>10} {'L6 Static':>10} {'L6 FuncFill':>12} {'L6 Ratio':>10}")
for m in models:
    if m not in all_data:
        continue
    part_a = all_data[m].get("part_a_context_activation", {})
    func = part_a.get("function", {})
    layers = func.get("layers", {})
    for li in ["2", "6"]:
        li_data = layers.get(li, {})
        static = li_data.get("contexts", {}).get("static", {}).get("mean_dist", 0)
        fill = li_data.get("contexts", {}).get("function_fill", {}).get("mean_dist", 0)
        ratio = li_data.get("contexts", {}).get("function_fill", {}).get("ratio_vs_random", 0)
        print(f"{m:<12} {static:>10.3f} {fill:>12.3f} {ratio:>10.2f}", end="")
    print()

# 3. Same_class context activation
print("\n--- 3. Same_Class: Category Probe vs Static ---")
print(f"{'Model':<12} {'L2 Static':>10} {'L2 CatProbe':>12} {'L2 Ratio':>10} {'L6 Static':>10} {'L6 CatProbe':>12} {'L6 Ratio':>10}")
for m in models:
    if m not in all_data:
        continue
    part_a = all_data[m].get("part_a_context_activation", {})
    sc = part_a.get("same_class", {})
    layers = sc.get("layers", {})
    for li in ["2", "6"]:
        li_data = layers.get(li, {})
        static = li_data.get("contexts", {}).get("static", {}).get("mean_dist", 0)
        probe = li_data.get("contexts", {}).get("category_probe", {}).get("mean_dist", 0)
        ratio = li_data.get("contexts", {}).get("category_probe", {}).get("ratio_vs_random", 0)
        print(f"{m:<12} {static:>10.3f} {probe:>12.3f} {ratio:>10.2f}", end="")
    print()

# 4. Negation: all contexts
print("\n--- 4. Negation: Static vs Probe Context ---")
print(f"{'Model':<12} {'L2 StaticPos':>12} {'L2 StaticNeg':>12} {'L2 NegProbe':>12}")
for m in models:
    if m not in all_data:
        continue
    part_a = all_data[m].get("part_a_context_activation", {})
    neg = part_a.get("negation", {})
    layers = neg.get("layers", {})
    li = "2"
    li_data = layers.get(li, {})
    sp = li_data.get("contexts", {}).get("static_pos", {}).get("mean_dist", 0)
    sn = li_data.get("contexts", {}).get("static_neg", {}).get("mean_dist", 0)
    np_ctx = li_data.get("contexts", {}).get("negation_probe", {}).get("mean_dist", 0)
    print(f"{m:<12} {sp:>12.3f} {sn:>12.3f} {np_ctx:>12.3f}")

# 5. Attribute context activation across layers (key finding)
print("\n--- 5. Attribute Fill Context: Ratio vs Random Across Layers ---")
print(f"{'Model':<12}", end="")
# Get layer keys from first model
ref_layers = []
for m in models:
    if m in all_data:
        attr = all_data[m]["part_a_context_activation"]["attribute"]
        ref_layers = sorted(attr.get("layers", {}).keys(), key=int)
        break
for li in ref_layers:
    print(f"{'L'+li:>8}", end="")
print()

for m in models:
    if m not in all_data:
        continue
    part_a = all_data[m].get("part_a_context_activation", {})
    attr = part_a.get("attribute", {})
    layers = attr.get("layers", {})
    print(f"{m:<12}", end="")
    for li in ref_layers:
        li_data = layers.get(li, {})
        ratio = li_data.get("contexts", {}).get("attribute_fill", {}).get("ratio_vs_random", 0)
        print(f"{ratio:>8.2f}", end="")
    print()

# 6. Function_probe ratio across layers (should be <1 everywhere)
print("\n--- 6. Function Probe Context: Ratio vs Random Across Layers ---")
print(f"{'Model':<12}", end="")
for li in ref_layers:
    print(f"{'L'+li:>8}", end="")
print()

for m in models:
    if m not in all_data:
        continue
    part_a = all_data[m].get("part_a_context_activation", {})
    func = part_a.get("function", {})
    layers = func.get("layers", {})
    print(f"{m:<12}", end="")
    for li in ref_layers:
        li_data = layers.get(li, {})
        ratio = li_data.get("contexts", {}).get("function_probe", {}).get("ratio_vs_random", 0)
        print(f"{ratio:>8.2f}", end="")
    print()

# ============================================================
# Part B: Causal Relation - Key Findings
# ============================================================
print("\n" + "=" * 80)
print("PART B: CAUSAL RELATION ANALYSIS")
print("=" * 80)

# 1. Best causal efficacy per relation type
print("\n--- 1. Best Causal Efficacy (target_vs_random) per Relation Type ---")
print(f"{'Relation':<15}", end="")
for m in models:
    print(f" {m:>15}", end="")
print()

for rel_type in ["same_class", "hypernym", "negation", "antonym", "attribute", "function"]:
    print(f"{rel_type:<15}", end="")
    for m in models:
        if m not in all_data:
            print(f" {'N/A':>15}", end="")
            continue
        part_b = all_data[m].get("part_b_causal_relation", {})
        rel = part_b.get(rel_type, {})
        hook_results = rel.get("hook_causal_results", {})
        best = 0
        for key, data in hook_results.items():
            if key.startswith("relation_dir"):
                for tok, ratio in data.get("target_vs_random", {}).items():
                    best = max(best, ratio)
        print(f" {best:>12.2f}x", end="")
    print()

# 2. Relation direction norms
print("\n--- 2. Relation Direction Norms ---")
print(f"{'Relation':<15}", end="")
for m in models:
    print(f" {m:>12}", end="")
print()

for rel_type in ["same_class", "hypernym", "negation", "antonym", "attribute", "function"]:
    print(f"{rel_type:<15}", end="")
    for m in models:
        if m not in all_data:
            print(f" {'N/A':>12}", end="")
            continue
        part_b = all_data[m].get("part_b_causal_relation", {})
        rel = part_b.get(rel_type, {})
        norm = rel.get("relation_dir_norm", 0)
        print(f" {norm:>12.2f}", end="")
    print()

# 3. Causal efficacy at each read layer for same_class
print("\n--- 3. Same_Class Causal Efficacy Across Read Layers ---")
for m in models:
    if m not in all_data:
        continue
    part_b = all_data[m].get("part_b_causal_relation", {})
    rel = part_b.get("same_class", {})
    hook_results = rel.get("hook_causal_results", {})
    print(f"\n  {m}:")
    for key, data in sorted(hook_results.items()):
        if key.startswith("relation_dir"):
            top = max(data.get("target_vs_random", {}).items(), key=lambda x: x[1], default=("N/A", 0))
            print(f"    {key}: delta_norm={data['delta_h_norm']:.3f}, best_target={top[0]}({top[1]:.2f}x)")

# 4. Attribute causal at each read layer
print("\n--- 4. Attribute Causal Efficacy Across Read Layers ---")
for m in models:
    if m not in all_data:
        continue
    part_b = all_data[m].get("part_b_causal_relation", {})
    rel = part_b.get("attribute", {})
    hook_results = rel.get("hook_causal_results", {})
    print(f"\n  {m}:")
    for key, data in sorted(hook_results.items()):
        if key.startswith("relation_dir"):
            top = max(data.get("target_vs_random", {}).items(), key=lambda x: x[1], default=("N/A", 0))
            print(f"    {key}: delta_norm={data['delta_h_norm']:.3f}, best_target={top[0]}({top[1]:.2f}x)")

# 5. Key comparison: Qwen3 vs DS7B norm differences
print("\n--- 5. Norm Comparison: Qwen3 vs DS7B (ratio) ---")
print(f"{'Relation':<15} {'Qwen3':>12} {'DS7B':>12} {'DS7B/Qwen3':>12}")
for rel_type in ["same_class", "hypernym", "negation", "antonym", "attribute", "function"]:
    qwen_norm = all_data.get("qwen3", {}).get("part_b_causal_relation", {}).get(rel_type, {}).get("relation_dir_norm", 0)
    ds_norm = all_data.get("deepseek7b", {}).get("part_b_causal_relation", {}).get(rel_type, {}).get("relation_dir_norm", 0)
    ratio = ds_norm / qwen_norm if qwen_norm > 0 else 0
    print(f"{rel_type:<15} {qwen_norm:>12.2f} {ds_norm:>12.2f} {ratio:>12.1f}x")

# ============================================================
# KEY OBJECTIVE FINDINGS SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("KEY OBJECTIVE FINDINGS")
print("=" * 80)

print("""
Finding 91: Attribute relations are conditionally activated by context
  - Static context: attribute ratio ≈ 1.0-1.3x random (NOT preserved)
  - Attribute fill context: L2 ratio = 5.0-5.7x random (STRONGLY preserved)
  - This is consistent across all three models
  → Attribute relations EXIST in the model but need appropriate context to activate

Finding 92: Function_probe context DESTROYS function relation (ratio < 1)
  - function_probe ("You use a knife to") ratio = 0.20-0.47x random
  - function_fill ("A knife is designed for") ratio = 0.49-2.0x (mixed)
  - function_probe makes related pairs FURTHER than random
  → "You use X to" template creates divergent predictions, not shared representations
  → This is NOT the same as "function relations don't exist"

Finding 93: Same_class relations are dramatically enhanced by category context
  - Static: L2 ratio ≈ 1.2-1.5x
  - Category probe ("The apple and the banana are both"): L2 ratio = 11-15x
  → Category context strongly activates shared representation paths

Finding 94: Negation relation has ratio < 1 in ALL contexts
  - negation pairs (happy/not_happy) are FURTHER apart than random pairs
  - This is expected: negation creates OPPOSITION, not similarity
  → The Mantel test in Phase 314 found "negation preserved" because it measured
     distance ordering, not direction. Not_happy IS far from happy, as expected.

Finding 95: All relation types have causal efficacy (1.4-7.5x random)
  - same_class: best 1.7-4.6x (Qwen3 strongest)
  - hypernym: best 1.6-7.5x (Qwen3 strongest)
  - negation: best 1.4-3.6x
  - antonym: best 2.9-4.5x
  - attribute: best 1.9-3.1x
  - function: best 1.9-3.8x
  → Even attribute and function have causal efficacy despite weak static preservation

Finding 96: DS7B relation direction norms are 3-34x larger than Qwen3
  - antonym: DS7B=104 vs Qwen3=32 (3.3x)
  - negation: DS7B=53 vs Qwen3=16 (3.3x)
  - same_class: DS7B=43 vs Qwen3=13 (3.3x)
  - attribute: DS7B=25 vs Qwen3=8 (3.1x)
  → Consistent with Phase 310-311 finding that DS7B has much larger norm scales
""")
