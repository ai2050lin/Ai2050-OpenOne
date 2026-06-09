"""
Phase 425 R2 跨模型对比汇总
===========================
"""
import sys, os
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
results_dir = ROOT / "results" / "phase425_embedding_perturbation"

models = ["qwen3", "glm4", "deepseek7b"]
perturb_types = ["add_category", "remove_category", "add_opposing", "add_random"]
categories = ["fruit", "animal", "tool", "vehicle", "place"]
tasks = ["category", "property", "part"]

print("=" * 80)
print("Phase 425 R2 Cross-Model Comparison")
print("=" * 80)

# Load all R2 results
all_data = {}
for model in models:
    path = results_dir / f"{model}_phase425_r2.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            all_data[model] = json.load(f)

# ===== Table 1: Category Task - Basin Transition Targets =====
print("\n" + "=" * 80)
print("Table 1: Category Task - Where does object go when category direction is removed?")
print("=" * 80)

for model in models:
    if model not in all_data:
        continue
    data = all_data[model]
    print(f"\n--- {model} ---")
    for obj_name, obj_data in data["per_object"].items():
        cat_task = obj_data["tasks"].get("category", {})
        baseline_top = cat_task.get("baseline", {}).get("top", "N/A")
        for ptype in ["remove_category", "add_opposing"]:
            for alpha in [1.0, 2.0]:
                key = f"{ptype}_a{alpha}"
                perturb = cat_task.get("perturbations", {}).get(key, {})
                if perturb:
                    new_top = perturb.get("top", "N/A")
                    delta = perturb.get("delta", 0)
                    print(f"  {obj_name}({obj_data['category']}): {ptype} a{alpha}: "
                          f"{baseline_top} -> {new_top} (delta={delta:+.3f})")

# ===== Table 2: Perturbation Sensitivity by Category×Task =====
print("\n" + "=" * 80)
print("Table 2: Mean |delta| by category×task×perturbation (alpha=1.0)")
print("=" * 80)

header = f"{'category_task':<25}" + "".join(f"{m:<20}" for m in models)
print(header)

for cat in categories:
    for task in tasks:
        row = f"{cat}_{task:<18}"
        for model in models:
            if model not in all_data:
                row += f"{'N/A':<20}"
                continue
            summary = all_data[model].get("summary", {})
            for ptype in ["add_category", "remove_category", "add_random"]:
                effect_key = f"{cat}_{task}"
                ptype_data = summary.get(ptype, {})
                if effect_key in ptype_data:
                    mean_d = ptype_data[effect_key].get("mean", 0)
                    row += f"{ptype[:3]}:{mean_d:+.3f} "
                else:
                    row += f"{ptype[:3]}:N/A   "
        print(row)

# ===== Table 3: Key Metric - Category Task Sensitivity =====
print("\n" + "=" * 80)
print("Table 3: Category Task - remove_category delta (轨道跃迁幅度)")
print("=" * 80)

for model in models:
    if model not in all_data:
        continue
    data = all_data[model]
    print(f"\n--- {model} ---")
    for obj_name, obj_data in data["per_object"].items():
        cat_task = obj_data["tasks"].get("category", {})
        baseline_level = cat_task.get("baseline", {}).get("level", 0)
        for alpha in [0.5, 1.0, 2.0]:
            key = f"remove_category_a{alpha}"
            perturb = cat_task.get("perturbations", {}).get(key, {})
            if perturb:
                new_level = perturb.get("level", 0)
                new_top = perturb.get("top", "N/A")
                print(f"  {obj_name}: level {baseline_level:.2f} -> {new_level:.2f} "
                      f"(-> {new_top}) [a={alpha}]")

# ===== Table 4: Random vs Category perturbation comparison =====
print("\n" + "=" * 80)
print("Table 4: Semantic vs Random perturbation (alpha=2.0) - Category Task")
print("Key question: Is category direction effect > random direction effect?")
print("=" * 80)

for model in models:
    if model not in all_data:
        continue
    data = all_data[model]
    print(f"\n--- {model} ---")
    for obj_name, obj_data in data["per_object"].items():
        cat_task = obj_data["tasks"].get("category", {})
        for alpha in [2.0]:
            remove_key = f"remove_category_a{alpha}"
            random_key = f"add_random_a{alpha}"
            remove_p = cat_task.get("perturbations", {}).get(remove_key, {})
            random_p = cat_task.get("perturbations", {}).get(random_key, {})
            if remove_p and random_p:
                r_delta = abs(remove_p.get("delta", 0))
                rn_delta = abs(random_p.get("delta", 0))
                ratio = r_delta / max(rn_delta, 0.001)
                semantic = "YES" if ratio > 2.0 else "NO"
                print(f"  {obj_name}: |remove_cat|={r_delta:.3f} |random|={rn_delta:.3f} "
                      f"ratio={ratio:.1f} semantic_specific={semantic}")

# ===== Table 5: Property task sensitivity =====
print("\n" + "=" * 80)
print("Table 5: Property Task - Is property knowledge stored in embedding?")
print("If remove_category doesn't change property -> property not in category direction")
print("=" * 80)

for model in models:
    if model not in all_data:
        continue
    data = all_data[model]
    print(f"\n--- {model} ---")
    for obj_name, obj_data in data["per_object"].items():
        prop_task = obj_data["tasks"].get("property", {})
        baseline_top = prop_task.get("baseline", {}).get("top", "N/A")
        for ptype in ["remove_category", "add_opposing", "add_random"]:
            for alpha in [2.0]:
                key = f"{ptype}_a{alpha}"
                perturb = prop_task.get("perturbations", {}).get(key, {})
                if perturb:
                    new_top = perturb.get("top", "N/A")
                    delta = perturb.get("delta", 0)
                    changed = "CHANGED" if new_top != baseline_top else "same"
                    print(f"  {obj_name}: {ptype} a{alpha}: {baseline_top}->{new_top} "
                          f"delta={delta:+.3f} [{changed}]")

# ===== Summary =====
print("\n" + "=" * 80)
print("KEY FINDINGS SUMMARY")
print("=" * 80)

for model in models:
    if model not in all_data:
        continue
    data = all_data[model]
    print(f"\n--- {model} ---")
    
    # Count basin transitions per perturbation type
    transitions = {}
    for obj_name, obj_data in data["per_object"].items():
        for task_name, task_data in obj_data["tasks"].items():
            baseline_top = task_data.get("baseline", {}).get("top", "")
            for ptype in perturb_types:
                for alpha in [1.0, 2.0]:
                    key = f"{ptype}_a{alpha}"
                    perturb = task_data.get("perturbations", {}).get(key, {})
                    if perturb and perturb.get("top", "") != baseline_top:
                        pkey = f"{ptype}_a{alpha}"
                        if pkey not in transitions:
                            transitions[pkey] = 0
                        transitions[pkey] += 1
    
    print("  Basin transitions by perturbation:")
    for pkey in sorted(transitions.keys()):
        print(f"    {pkey}: {transitions[pkey]} transitions")
    
    # Average |delta| for category task
    cat_deltas = {pt: [] for pt in perturb_types}
    for obj_name, obj_data in data["per_object"].items():
        cat_task = obj_data["tasks"].get("category", {})
        for pt in perturb_types:
            for alpha in [1.0]:
                key = f"{pt}_a{alpha}"
                perturb = cat_task.get("perturbations", {}).get(key, {})
                if perturb:
                    cat_deltas[pt].append(abs(perturb.get("delta", 0)))
    
    print("  Category task |delta| (alpha=1.0):")
    for pt in perturb_types:
        if cat_deltas[pt]:
            print(f"    {pt}: mean={np.mean(cat_deltas[pt]):.3f} "
                  f"std={np.std(cat_deltas[pt]):.3f}")

print("\nDone!")
