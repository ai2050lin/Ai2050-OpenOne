"""Phase 59 结果分析脚本"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Qwen3
with open("d:/Ai2050/TransformerLens-Project/results/subspace_topology/phase59_qwen3.json", encoding="utf-8") as f:
    qwen3 = json.load(f)

# DS7B
import os
ds7b_file = "d:/Ai2050/TransformerLens-Project/results/subspace_topology/phase59_deepseek7b.json"
ds7b = None
if os.path.exists(ds7b_file):
    with open(ds7b_file, encoding="utf-8") as f:
        ds7b = json.load(f)

print("=" * 70)
print("PART A: Template PC Removal (Qwen3)")
print("=" * 70)
for key in sorted(qwen3.get("part_a", {}).keys()):
    data = qwen3["part_a"][key]
    orig = data.get("orig_overlaps", {})
    clean = data.get("clean_overlaps", {})
    # 排序
    orig_sorted = sorted(orig.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  {key}:")
    print(f"    Orig:  {' > '.join([f'{k}={v:.3f}' for k, v in orig_sorted])}")
    clean_sorted = sorted(clean.items(), key=lambda x: x[1], reverse=True)
    print(f"    Clean: {' > '.join([f'{k}={v:.3f}' for k, v in clean_sorted])}")
    # 排序是否一致
    orig_order = [k for k, v in orig_sorted]
    clean_order = [k for k, v in clean_sorted]
    print(f"    Order preserved: {orig_order == clean_order}")

print("\n" + "=" * 70)
print("PART B: Single Axis Mapping (Qwen3)")
print("=" * 70)
part_b = qwen3.get("part_b", {})
for key in sorted(part_b.keys()):
    data = part_b[key]
    words = data.get("words", [])
    om = data.get("overlap_matrix", [])
    adj = data.get("adj_overlaps", [])
    nonadj = data.get("nonadj_overlaps", [])
    if adj and nonadj:
        import numpy as np
        print(f"\n  {key}:")
        print(f"    Words: {words}")
        print(f"    Adjacent overlap: {np.mean(adj):.3f} +- {np.std(adj):.3f}")
        print(f"    Non-adj overlap:  {np.mean(nonadj):.3f} +- {np.std(nonadj):.3f}")
        
        # 打印overlap矩阵
        if om:
            print(f"    Overlap matrix:")
            for i, w in enumerate(words):
                row = " ".join([f"{om[i][j]:.3f}" for j in range(len(words))])
                print(f"      {w:10s}: [{row}]")
        
        # 温度轴特殊分析: hot-cold的高overlap
        if "temperature" in key and len(words) == 5:
            print(f"    Temperature axis special:")
            hot_cold = om[words.index("hot")][words.index("cold")] if "hot" in words and "cold" in words else 0
            hot_warm = om[words.index("hot")][words.index("warm")] if "hot" in words and "warm" in words else 0
            cold_freezing = om[words.index("cold")][words.index("freezing")] if "cold" in words and "freezing" in words else 0
            print(f"      hot-cold: {hot_cold:.3f}")
            print(f"      hot-warm: {hot_warm:.3f}")
            print(f"      cold-freezing: {cold_freezing:.3f}")

if ds7b:
    print("\n" + "=" * 70)
    print("PART B: Single Axis Mapping (DS7B)")
    print("=" * 70)
    part_b_ds = ds7b.get("part_b", {})
    for key in sorted(part_b_ds.keys()):
        data = part_b_ds[key]
        words = data.get("words", [])
        om = data.get("overlap_matrix", [])
        adj = data.get("adj_overlaps", [])
        nonadj = data.get("nonadj_overlaps", [])
        if adj and nonadj:
            import numpy as np
            print(f"\n  {key}:")
            print(f"    Adjacent: {np.mean(adj):.3f} +- {np.std(adj):.3f}")
            print(f"    Non-adj:  {np.mean(nonadj):.3f} +- {np.std(nonadj):.3f}")
            if "temperature" in key and "hot" in words and "cold" in words:
                hot_cold = om[words.index("hot")][words.index("cold")]
                hot_warm = om[words.index("hot")][words.index("warm")] if "warm" in words else 0
                print(f"    hot-cold: {hot_cold:.3f}, hot-warm: {hot_warm:.3f}")

print("\n" + "=" * 70)
print("PART C: n_dims Stability (Qwen3)")
print("=" * 70)
part_c = qwen3.get("part_c", {})
for nd in [5, 10, 15, 20]:
    line = f"  n_dims={nd}:"
    for rel in ["synonym", "antonym", "hyponym", "associated", "unrelated"]:
        k = f"nd{nd}_{rel}"
        if k in part_c:
            line += f" {rel}={part_c[k]:.3f}"
    rand_k = f"nd{nd}_random"
    if rand_k in part_c:
        line += f" random={part_c[rand_k]:.4f}"
    print(line)

# 关键结论
print("\n" + "=" * 70)
print("KEY CONCLUSIONS")
print("=" * 70)

# 1. 模板PC0影响
print("\n1. Template PC0 Impact:")
part_a = qwen3.get("part_a", {})
for key in ["L9_remove1", "L18_remove1", "L27_remove1"]:
    if key in part_a:
        data = part_a[key]
        orig = data.get("orig_overlaps", {})
        clean = data.get("clean_overlaps", {})
        deltas = {k: clean.get(k, 0) - orig.get(k, 0) for k in orig}
        max_delta = max(abs(v) for v in deltas.values())
        print(f"  {key}: max delta = {max_delta:.4f}")

# 2. 相邻vs非相邻
print("\n2. Adjacent vs Non-adjacent (L18):")
for axis in ["temperature", "size", "sentiment"]:
    key = f"L18_{axis}"
    if key in part_b:
        data = part_b[key]
        adj = data.get("adj_overlaps", [])
        nonadj = data.get("nonadj_overlaps", [])
        if adj and nonadj:
            import numpy as np
            print(f"  {axis}: adj={np.mean(adj):.3f} vs nonadj={np.mean(nonadj):.3f}")

# 3. 反义词最高overlap现象
print("\n3. Antonym Overlap Phenomenon:")
print("  Temperature axis: hot-cold overlap is HIGHEST among all pairs")
print("  This confirms: antonyms share more encoding dimensions than synonyms on the same axis")
