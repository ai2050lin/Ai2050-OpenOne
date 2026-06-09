"""Phase 426 R2 跨模型对比汇总"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
results_dir = ROOT / "results" / "phase426_alpha_basin_boundary"

models = ["qwen3", "glm4", "deepseek7b"]

# 加载R2数据
data = {}
for m in models:
    path = results_dir / f"{m}_phase426_r2.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data[m] = json.load(f)

print("=" * 80)
print("Phase 426 R2: 跨模型Alpha轨道边界扫描对比")
print("=" * 80)

# ===== 1. 临界Alpha对比 =====
print("\n### 1. 临界Alpha (Top-1 Switch Point) 对比 ###")
print(f"{'Object_Task_Perturb':<45} {'Qwen3':>8} {'GLM4':>8} {'DS7B':>8}")
print("-" * 75)

# 收集所有switch points
all_switches = {}
for m in models:
    if m in data and "top1_switches" in data[m]:
        for key, alpha in data[m]["top1_switches"].items():
            if key not in all_switches:
                all_switches[key] = {}
            all_switches[key][m] = alpha

# 只展示remove_category和add_opposing的category任务
category_switches = {k: v for k, v in all_switches.items() 
                     if "category" in k and ("remove_category" in k or "add_opposing" in k)}
for key in sorted(category_switches.keys()):
    vals = category_switches[key]
    q = vals.get("qwen3", "-")
    g = vals.get("glm4", "-")
    d = vals.get("deepseek7b", "-")
    print(f"  {key:<43} {str(q):>8} {str(g):>8} {str(d):>8}")

# ===== 2. 跃迁目标对比 =====
print("\n### 2. 跃迁目标对比 (remove_category, alpha=1.0, category任务) ###")
print(f"{'Object':<12} {'Qwen3_target':>14} {'GLM4_target':>14} {'DS7B_target':>14}")
print("-" * 58)

common_objects = ["apple", "orange", "dog", "horse", "knife", "hammer", "car", "bus"]
for obj in common_objects:
    targets = {}
    for m in models:
        if m in data and obj in data[m].get("per_object", {}):
            obj_data = data[m]["per_object"][obj]
            task = obj_data.get("tasks", {}).get("category", {})
            curve = task.get("perturbation_curves", {}).get("remove_category", {})
            if "1.0" in curve:
                targets[m] = curve["1.0"].get("top", "?")
            else:
                targets[m] = "?"
        else:
            targets[m] = "?"
    print(f"  {obj:<12} {targets.get('qwen3','?'):>14} {targets.get('glm4','?'):>14} {targets.get('deepseek7b','?'):>14}")

# ===== 3. Property任务是否受影响 =====
print("\n### 3. Property任务受影响对比 (remove_category, alpha=1.0) ###")
print(f"{'Object':<12} {'Qwen3_Δprop':>14} {'GLM4_Δprop':>14} {'DS7B_Δprop':>14}")
print("-" * 58)

for obj in common_objects:
    deltas = {}
    for m in models:
        if m in data and obj in data[m].get("per_object", {}):
            obj_data = data[m]["per_object"][obj]
            task = obj_data.get("tasks", {}).get("property", {})
            curve = task.get("perturbation_curves", {}).get("remove_category", {})
            if "1.0" in curve:
                deltas[m] = curve["1.0"].get("delta", 0)
            else:
                deltas[m] = 0
        else:
            deltas[m] = 0
    q = deltas.get("qwen3", 0)
    g = deltas.get("glm4", 0)
    d = deltas.get("deepseek7b", 0)
    print(f"  {obj:<12} {q:>+14.3f} {g:>+14.3f} {d:>+14.3f}")

# ===== 4. 语义特异性比 (alpha=1.0, category任务) =====
print("\n### 4. 语义特异性比 (|category Δ| / |random Δ|, alpha=1.0, category任务) ###")
print(f"{'Object':<12} {'Qwen3_ratio':>14} {'GLM4_ratio':>14} {'DS7B_ratio':>14} {'Q_catΔ':>8} {'G_catΔ':>8} {'D_catΔ':>8}")
print("-" * 88)

for obj in common_objects:
    ratios = {}
    cat_deltas = {}
    for m in models:
        if m in data and obj in data[m].get("per_object", {}):
            obj_data = data[m]["per_object"][obj]
            task = obj_data.get("tasks", {}).get("category", {})
            cat_curve = task.get("perturbation_curves", {}).get("remove_category", {})
            rand_curve = task.get("perturbation_curves", {}).get("add_random", {})
            if "1.0" in cat_curve and "1.0" in rand_curve:
                cat_d = abs(cat_curve["1.0"].get("delta", 0))
                rand_d = abs(rand_curve["1.0"].get("delta", 0))
                cat_deltas[m] = cat_d
                ratios[m] = cat_d / rand_d if rand_d > 0.01 else float('inf')
            else:
                cat_deltas[m] = 0
                ratios[m] = 0
        else:
            cat_deltas[m] = 0
            ratios[m] = 0
    
    q_r = ratios.get("qwen3", 0)
    g_r = ratios.get("glm4", 0)
    d_r = ratios.get("deepseek7b", 0)
    q_c = cat_deltas.get("qwen3", 0)
    g_c = cat_deltas.get("glm4", 0)
    d_c = cat_deltas.get("deepseek7b", 0)
    
    def fmt_ratio(r):
        if r == float('inf'):
            return "∞"
        return f"{r:.1f}"
    
    print(f"  {obj:<12} {fmt_ratio(q_r):>14} {fmt_ratio(g_r):>14} {fmt_ratio(d_r):>14} "
          f"{q_c:>8.3f} {g_c:>8.3f} {d_c:>8.3f}")

# ===== 5. remove_identity效果对比 =====
print("\n### 5. remove_identity效果对比 (alpha=1.0, category任务) ###")
print(f"{'Object':<12} {'Qwen3_Δid':>14} {'GLM4_Δid':>14} {'DS7B_Δid':>14}")
print("-" * 58)

for obj in common_objects:
    deltas = {}
    for m in models:
        if m in data and obj in data[m].get("per_object", {}):
            obj_data = data[m]["per_object"][obj]
            task = obj_data.get("tasks", {}).get("category", {})
            curve = task.get("perturbation_curves", {}).get("remove_identity", {})
            if "1.0" in curve:
                deltas[m] = curve["1.0"].get("delta", 0)
            else:
                deltas[m] = 0
        else:
            deltas[m] = 0
    q = deltas.get("qwen3", 0)
    g = deltas.get("glm4", 0)
    d = deltas.get("deepseek7b", 0)
    print(f"  {obj:<12} {q:>+14.3f} {g:>+14.3f} {d:>+14.3f}")

# ===== 6. 精细Alpha曲线 (apple/category/remove_category) =====
print("\n### 6. 精细Alpha曲线: apple → category → remove_category ###")
print(f"{'Alpha':>8} {'Qwen3_lvl':>12} {'GLM4_lvl':>12} {'DS7B_lvl':>12} {'Q_top':>8} {'G_top':>8} {'D_top':>8}")
print("-" * 72)

for m in models:
    if m not in data:
        continue
    # 获取alpha grid
    alpha_grid = data[m].get("alpha_grid", [])

# 使用qwen3的alpha grid作为标准(应该相同)
alpha_grid = data.get("qwen3", {}).get("alpha_grid", [])
for alpha in alpha_grid:
    alpha_str = str(alpha)
    row = {"qwen3": {}, "glm4": {}, "deepseek7b": {}}
    for m in models:
        if m in data and "apple" in data[m].get("per_object", {}):
            obj_data = data[m]["per_object"]["apple"]
            task = obj_data.get("tasks", {}).get("category", {})
            curve = task.get("perturbation_curves", {}).get("remove_category", {})
            if alpha_str in curve:
                row[m] = curve[alpha_str]
    
    q_lvl = row["qwen3"].get("level", "-") if row["qwen3"] else "-"
    g_lvl = row["glm4"].get("level", "-") if row["glm4"] else "-"
    d_lvl = row["deepseek7b"].get("level", "-") if row["deepseek7b"] else "-"
    q_top = row["qwen3"].get("top", "-")[:3] if row["qwen3"] else "-"
    g_top = row["glm4"].get("top", "-")[:3] if row["glm4"] else "-"
    d_top = row["deepseek7b"].get("top", "-")[:3] if row["deepseek7b"] else "-"
    
    def fl(v):
        return f"{v:.2f}" if isinstance(v, (int, float)) else str(v)
    
    print(f"  {alpha:>6.2f} {fl(q_lvl):>12} {fl(g_lvl):>12} {fl(d_lvl):>12} {q_top:>8} {g_top:>8} {d_top:>8}")

# ===== 7. 核心发现总结 =====
print("\n" + "=" * 80)
print("### 核心发现 ###")
print("=" * 80)

# 计算平均临界alpha
avg_critical = {}
for m in models:
    if m in data and "top1_switches" in data[m]:
        cat_switches = [v for k, v in data[m]["top1_switches"].items() 
                       if "category" in k and "remove_category" in k]
        if cat_switches:
            avg_critical[m] = np.mean(cat_switches)

print(f"\n1. 平均临界Alpha (category/remove_category):")
for m in models:
    v = avg_critical.get(m, "N/A")
    print(f"   {m}: {v:.3f}" if isinstance(v, float) else f"   {m}: {v}")

# property是否受影响
print(f"\n2. Property任务受类别扰动影响程度 (alpha=1.0, |delta|均值):")
for m in models:
    deltas = []
    if m in data:
        for obj_name, obj_data in data[m].get("per_object", {}).items():
            task = obj_data.get("tasks", {}).get("property", {})
            curve = task.get("perturbation_curves", {}).get("remove_category", {})
            if "1.0" in curve:
                deltas.append(abs(curve["1.0"].get("delta", 0)))
    if deltas:
        print(f"   {m}: mean|Δ|={np.mean(deltas):.3f}, max|Δ|={np.max(deltas):.3f}")
    else:
        print(f"   {m}: no data")

print(f"\n3. 对象身份残差影响 (alpha=1.0, category任务|delta|均值):")
for m in models:
    deltas = []
    if m in data:
        for obj_name, obj_data in data[m].get("per_object", {}).items():
            task = obj_data.get("tasks", {}).get("category", {})
            curve = task.get("perturbation_curves", {}).get("remove_identity", {})
            if "1.0" in curve:
                deltas.append(abs(curve["1.0"].get("delta", 0)))
    if deltas:
        print(f"   {m}: mean|Δ|={np.mean(deltas):.3f}, max|Δ|={np.max(deltas):.3f}")
    else:
        print(f"   {m}: no data")

print("\nDone.")
