import json
import numpy as np

models = ["qwen3", "glm4", "deepseek7b"]
model_names = {"qwen3": "Qwen3-4B", "glm4": "GLM4-9B", "deepseek7b": "DS7B"}

all_data = {}
for m in models:
    f = f"results/subspace_topology/exp3_reuse_diff_{m}.json"
    all_data[m] = json.load(open(f, encoding='utf-8'))

# === Part 1: 概念层级对 - 关键层对比 ===
print("=" * 80)
print("Part 1: 概念层级对 - 跨模型对比")
print("=" * 80)

for pair_key in ["apple_fruit", "dog_animal", "red_color"]:
    print(f"\n--- {pair_key} ---")
    for m in models:
        if pair_key not in all_data[m]["concept_pairs"]:
            continue
        layers = all_data[m]["concept_pairs"][pair_key]["layers"]
        name = model_names[m]
        # 选取浅/中/深层
        sorted_li = sorted(layers.keys(), key=lambda x: int(x))
        early = sorted_li[0]
        mid_idx = len(sorted_li) // 2
        mid = sorted_li[mid_idx]
        late = sorted_li[-1]
        
        e = layers[early]
        mi = layers[mid]
        l = layers[late]
        
        print(f"  {name}: L{early} cos={e['cos_mean']:.3f} shared_A={e['shared_ratio_A']:.3f} | "
              f"L{mid} cos={mi['cos_mean']:.3f} shared_A={mi['shared_ratio_A']:.3f} | "
              f"L{late} cos={l['cos_mean']:.3f} shared_A={l['shared_ratio_A']:.3f} | "
              f"delta_unique: L{early}={e['unique_delta_ratio']:.3f} L{mid}={mi['unique_delta_ratio']:.3f} L{late}={l['unique_delta_ratio']:.3f}")

# === Part 2: 任务复用 ===
print("\n" + "=" * 80)
print("Part 2: 任务复用 (translate_en/fr)")
print("=" * 80)
for m in models:
    name = model_names[m]
    tp = all_data[m]["task_pair"]["translate_en_fr"]["layers"]
    sorted_li = sorted(tp.keys(), key=lambda x: int(x))
    # 找cos最高和最低的层
    max_cos_li = max(sorted_li, key=lambda x: tp[x]["cos_mean"])
    min_cos_li = min(sorted_li, key=lambda x: tp[x]["cos_mean"])
    mid_li = sorted_li[len(sorted_li)//2]
    
    print(f"  {name}: min_cos=L{min_cos_li}({tp[min_cos_li]['cos_mean']:.3f}) "
          f"mid=L{mid_li}({tp[mid_li]['cos_mean']:.3f}) "
          f"max_cos=L{max_cos_li}({tp[max_cos_li]['cos_mean']:.3f}) "
          f"shared@mid={tp[mid_li]['avg_shared_ratio']:.3f} "
          f"delta_unique@mid={tp[mid_li]['unique_delta_ratio']:.3f}")

# === Part 3: 逻辑功能对 ===
print("\n" + "=" * 80)
print("Part 3: 逻辑功能对")
print("=" * 80)

for pair_key in ["and_or", "not_but"]:
    print(f"\n--- {pair_key} ---")
    for m in models:
        name = model_names[m]
        lp = all_data[m]["logic_pairs"][pair_key]["layers"]
        sorted_li = sorted(lp.keys(), key=lambda x: int(x))
        
        # 找cos最低的层(最分化)
        min_cos_li = min(sorted_li, key=lambda x: lp[x]["cos_mean"])
        max_cos_li = max(sorted_li, key=lambda x: lp[x]["cos_mean"])
        
        print(f"  {name}: min_cos=L{min_cos_li}({lp[min_cos_li]['cos_mean']:.3f} shared={lp[min_cos_li]['avg_shared_ratio']:.3f} delta_unique={lp[min_cos_li]['unique_delta_ratio']:.4f}) "
              f"max_cos=L{max_cos_li}({lp[max_cos_li]['cos_mean']:.3f})")

# === Part 4: 跨概念骨干 ===
print("\n" + "=" * 80)
print("Part 4: 跨概念复用骨干")
print("=" * 80)
for m in models:
    name = model_names[m]
    bb = all_data[m].get("cross_concept_backbone", {})
    overlaps = bb.get("backbone_overlaps", {})
    strengths = bb.get("pair_strengths", {})
    
    if overlaps:
        avg_overlap = np.mean(list(overlaps.values()))
        print(f"  {name}: 平均骨干对齐={avg_overlap:.4f}")
        for k, v in overlaps.items():
            print(f"    {k}: cos={v:.4f}")
    if strengths:
        avg_strength = np.mean(list(strengths.values()))
        print(f"  {name}: 平均共享强度={avg_strength:.4f}")
        for k, v in strengths.items():
            print(f"    {k}: {v:.4f}")

# === 关键发现总结 ===
print("\n" + "=" * 80)
print("关键发现总结")
print("=" * 80)

print("\n1. 概念层级复用模式:")
print("   - 所有模型: cos随深度增加 (0.4-0.6 → 0.8-0.9)")
print("   - shared_A/B: 0.35-0.55, 约40-50%的方差在子空间间共享")
print("   - delta_unique: 0.7-0.95, 均值差异主要在独特子空间")

print("\n2. 任务复用 (translate_en/fr):")
for m in models:
    name = model_names[m]
    tp = all_data[m]["task_pair"]["translate_en_fr"]["layers"]
    mid_li = sorted(tp.keys(), key=lambda x: int(x))[len(tp)//2]
    cos_mid = tp[mid_li]["cos_mean"]
    shared_mid = tp[mid_li]["avg_shared_ratio"]
    print(f"   - {name}: 中间层cos={cos_mid:.3f}, shared={shared_mid:.3f}")

print("\n3. AND/OR分化:")
for m in models:
    name = model_names[m]
    lp = all_data[m]["logic_pairs"]["and_or"]["layers"]
    sorted_li = sorted(lp.keys(), key=lambda x: int(x))
    min_cos_li = min(sorted_li, key=lambda x: lp[x]["cos_mean"])
    print(f"   - {name}: 最低cos=L{min_cos_li}({lp[min_cos_li]['cos_mean']:.3f}), delta_unique={lp[min_cos_li]['unique_delta_ratio']:.4f}")

print("\n4. 跨概念骨干:")
for m in models:
    name = model_names[m]
    bb = all_data[m].get("cross_concept_backbone", {})
    overlaps = bb.get("backbone_overlaps", {})
    if overlaps:
        avg = np.mean(list(overlaps.values()))
        print(f"   - {name}: 骨干对齐={avg:.4f}")
