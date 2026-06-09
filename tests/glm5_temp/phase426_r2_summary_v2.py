"""Phase 426 R2 跨模型对比汇总 - 修复版"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # 项目根目录
results_dir = ROOT / "results" / "phase426_alpha_basin_boundary"

models = ["qwen3", "glm4", "deepseek7b"]

# 加载R2数据
data = {}
for m in models:
    path = results_dir / f"{m}_phase426_r2.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data[m] = json.load(f)
        print(f"  Loaded {m}: {len(data[m].get('per_object', {}))} objects")
    else:
        print(f"  MISSING: {path}")

# 取所有模型的公共对象
all_obj_sets = [set(data[m].get("per_object", {}).keys()) for m in models if m in data]
common_objects = sorted(set.intersection(*all_obj_sets)) if all_obj_sets else []
print(f"\n  Common objects ({len(common_objects)}): {common_objects}")

print("\n" + "=" * 80)
print("Phase 426 R2: 跨模型Alpha轨道边界扫描对比")
print("=" * 80)

# ===== 1. 临界Alpha对比 =====
print("\n### 1. 临界Alpha (Top-1 Switch Point) 对比 ###")
print(f"{'Object_Task_Perturb':<50} {'Qwen3':>8} {'GLM4':>8} {'DS7B':>8}")
print("-" * 78)

all_switches = {}
for m in models:
    if m in data and "top1_switches" in data[m]:
        for key, alpha in data[m]["top1_switches"].items():
            if key not in all_switches:
                all_switches[key] = {}
            all_switches[key][m] = alpha

# 展示category任务的switch
cat_switches = {k: v for k, v in all_switches.items() 
                if "category" in k and ("remove_category" in k or "add_opposing" in k)}
for key in sorted(cat_switches.keys()):
    vals = cat_switches[key]
    q = f"{vals['qwen3']:.2f}" if 'qwen3' in vals else "-"
    g = f"{vals['glm4']:.2f}" if 'glm4' in vals else "-"
    d = f"{vals['deepseek7b']:.2f}" if 'deepseek7b' in vals else "-"
    print(f"  {key:<48} {q:>8} {g:>8} {d:>8}")

# ===== 2. 跃迁目标对比 =====
print("\n### 2. 跃迁目标对比 (remove_category, alpha=1.0, category任务) ###")
print(f"{'Object':<12} {'Q_target':>12} {'G_target':>12} {'D_target':>12} {'Q_Δ':>8} {'G_Δ':>8} {'D_Δ':>8}")
print("-" * 76)

for obj in common_objects:
    row = {}
    for m in models:
        if m in data and obj in data[m].get("per_object", {}):
            obj_data = data[m]["per_object"][obj]
            task = obj_data.get("tasks", {}).get("category", {})
            curve = task.get("perturbation_curves", {}).get("remove_category", {})
            if "1.0" in curve:
                row[m] = curve["1.0"]
            else:
                row[m] = {}
        else:
            row[m] = {}
    
    q_t = row.get("qwen3", {}).get("top", "?")[:5]
    g_t = row.get("glm4", {}).get("top", "?")[:5]
    d_t = row.get("deepseek7b", {}).get("top", "?")[:5]
    q_d = row.get("qwen3", {}).get("delta", 0)
    g_d = row.get("glm4", {}).get("delta", 0)
    d_d = row.get("deepseek7b", {}).get("delta", 0)
    print(f"  {obj:<12} {q_t:>12} {g_t:>12} {d_t:>12} {q_d:>+8.2f} {g_d:>+8.2f} {d_d:>+8.2f}")

# ===== 3. Property任务受影响 =====
print("\n### 3. Property任务受影响 (remove_category, alpha=1.0) ###")
print(f"{'Object':<12} {'Q_Δprop':>10} {'G_Δprop':>10} {'D_Δprop':>10} {'Q_top':>8} {'G_top':>8} {'D_top':>8}")
print("-" * 70)

for obj in common_objects:
    row = {}
    for m in models:
        if m in data and obj in data[m].get("per_object", {}):
            obj_data = data[m]["per_object"][obj]
            task = obj_data.get("tasks", {}).get("property", {})
            curve = task.get("perturbation_curves", {}).get("remove_category", {})
            if "1.0" in curve:
                row[m] = curve["1.0"]
            else:
                row[m] = {}
        else:
            row[m] = {}
    
    q_d = row.get("qwen3", {}).get("delta", 0)
    g_d = row.get("glm4", {}).get("delta", 0)
    d_d = row.get("deepseek7b", {}).get("delta", 0)
    q_t = row.get("qwen3", {}).get("top", "?")[:5]
    g_t = row.get("glm4", {}).get("top", "?")[:5]
    d_t = row.get("deepseek7b", {}).get("top", "?")[:5]
    print(f"  {obj:<12} {q_d:>+10.3f} {g_d:>+10.3f} {d_d:>+10.3f} {q_t:>8} {g_t:>8} {d_t:>8}")

# ===== 4. 语义特异性比 =====
print("\n### 4. 语义特异性比 (|cat Δ|/|rand Δ|, alpha=1.0, category任务) ###")
print(f"{'Object':<12} {'Q_ratio':>10} {'G_ratio':>10} {'D_ratio':>10} {'Q_catΔ':>8} {'G_catΔ':>8} {'D_catΔ':>8}")
print("-" * 80)

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
    
    def fr(r):
        return "∞" if r == float('inf') else f"{r:.1f}" if r > 0 else "0"
    
    print(f"  {obj:<12} {fr(ratios.get('qwen3',0)):>10} {fr(ratios.get('glm4',0)):>10} "
          f"{fr(ratios.get('deepseek7b',0)):>10} {cat_deltas.get('qwen3',0):>8.3f} "
          f"{cat_deltas.get('glm4',0):>8.3f} {cat_deltas.get('deepseek7b',0):>8.3f}")

# ===== 5. remove_identity效果 =====
print("\n### 5. remove_identity效果 (alpha=1.0, category任务) ###")
print(f"{'Object':<12} {'Q_Δid':>10} {'G_Δid':>10} {'D_Δid':>10}")
print("-" * 46)

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
    print(f"  {obj:<12} {deltas.get('qwen3',0):>+10.3f} {deltas.get('glm4',0):>+10.3f} {deltas.get('deepseek7b',0):>+10.3f}")

# ===== 6. 精细Alpha曲线 =====
print("\n### 6. 精细Alpha曲线: apple → category → remove_category ###")
alpha_grid = data.get("qwen3", {}).get("alpha_grid", [])

print(f"{'Alpha':>8} {'Q_lvl':>8} {'G_lvl':>8} {'D_lvl':>8} {'Q_top':>6} {'G_top':>6} {'D_top':>6} {'Q_ent':>8} {'G_ent':>8} {'D_ent':>8}")
print("-" * 80)

for alpha in alpha_grid:
    alpha_str = str(alpha)
    row = {}
    for m in models:
        if m in data and "apple" in data[m].get("per_object", {}):
            obj_data = data[m]["per_object"]["apple"]
            task = obj_data.get("tasks", {}).get("category", {})
            curve = task.get("perturbation_curves", {}).get("remove_category", {})
            if alpha_str in curve:
                row[m] = curve[alpha_str]
    
    def fl(v, w=8):
        return f"{v:.3f}" if isinstance(v, (int, float)) else str(v).rjust(w)
    
    q_lvl = row.get("qwen3", {}).get("level", "-")
    g_lvl = row.get("glm4", {}).get("level", "-")
    d_lvl = row.get("deepseek7b", {}).get("level", "-")
    q_top = row.get("qwen3", {}).get("top", "-")[:3]
    g_top = row.get("glm4", {}).get("top", "-")[:3]
    d_top = row.get("deepseek7b", {}).get("top", "-")[:3]
    q_ent = row.get("qwen3", {}).get("entropy", "-")
    g_ent = row.get("glm4", {}).get("entropy", "-")
    d_ent = row.get("deepseek7b", {}).get("entropy", "-")
    
    print(f"  {alpha:>6.2f} {fl(q_lvl):>8} {fl(g_lvl):>8} {fl(d_lvl):>8} "
          f"{q_top:>6} {g_top:>6} {d_top:>6} {fl(q_ent):>8} {fl(g_ent):>8} {fl(d_ent):>8}")

# ===== 7. 核心数据汇总 =====
print("\n" + "=" * 80)
print("### 核心数据汇总 ###")
print("=" * 80)

# 平均临界Alpha
print(f"\n1. 平均临界Alpha (category, remove_category):")
for m in models:
    if m in data and "top1_switches" in data[m]:
        cat_sw = [v for k, v in data[m]["top1_switches"].items() 
                 if "category" in k and "remove_category" in k]
        if cat_sw:
            print(f"   {m}: mean={np.mean(cat_sw):.3f}, values={sorted(cat_sw)}")
        else:
            print(f"   {m}: no category switches")

# Property受影响程度
print(f"\n2. Property任务受类别扰动影响 (alpha=1.0, |delta|):")
for m in models:
    deltas = []
    if m in data:
        for obj_name, obj_data in data[m].get("per_object", {}).items():
            task = obj_data.get("tasks", {}).get("property", {})
            curve = task.get("perturbation_curves", {}).get("remove_category", {})
            if "1.0" in curve:
                deltas.append(abs(curve["1.0"].get("delta", 0)))
    if deltas:
        print(f"   {m}: mean|Δ|={np.mean(deltas):.3f}, max={np.max(deltas):.3f}, n={len(deltas)}")

# Identity残差影响
print(f"\n3. remove_identity效果 (alpha=1.0, category, |delta|):")
for m in models:
    deltas = []
    if m in data:
        for obj_name, obj_data in data[m].get("per_object", {}).items():
            task = obj_data.get("tasks", {}).get("category", {})
            curve = task.get("perturbation_curves", {}).get("remove_identity", {})
            if "1.0" in curve:
                deltas.append(abs(curve["1.0"].get("delta", 0)))
    if deltas:
        print(f"   {m}: mean|Δ|={np.mean(deltas):.3f}, max={np.max(deltas):.3f}")

print("\nDone.")
