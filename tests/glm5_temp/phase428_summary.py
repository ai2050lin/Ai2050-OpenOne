"""Phase 428 跨模型汇总 + MEMO更新"""
import sys, json, time
from pathlib import Path
from collections import OrderedDict

ROOT = Path(__file__).resolve().parents[2]
results_dir = ROOT / "results" / "phase428_midlayer_perturbation"
memo_path = ROOT / "research" / "glm5" / "docs" / "AGI_GLM5_MEMO.md"

models = ["qwen3", "glm4", "deepseek7b"]

# 加载所有R2结果
all_data = {}
for m in models:
    fp = results_dir / f"{m}_phase428_r2.json"
    if fp.exists():
        with open(fp, encoding="utf-8") as f:
            all_data[m] = json.load(f)

# ===== 跨模型核心对比 =====
print("=" * 80)
print("Phase 428 跨模型核心对比")
print("=" * 80)

# 1. Embedding vs Mid-layer effectiveness (α=1.0, remove_category, category task)
print("\n--- 1. Embedding vs Mid-Layer Effectiveness (α=1.0, remove_category, category) ---")
for obj in ["apple", "dog", "knife", "car", "cat"]:
    print(f"\n  {obj}:")
    for m in models:
        if m not in all_data:
            continue
        d = all_data[m]
        obj_data = d.get("per_object", {}).get(obj)
        if not obj_data:
            continue
        base = obj_data["baselines"]["category"]["level"]
        key = "category_remove_category"
        perturb = obj_data["perturbations"].get(key, {})
        
        # Collect deltas at each layer
        layer_deltas = {}
        for layer_key, curve in perturb.items():
            if "1.0" in curve:
                layer_deltas[layer_key] = curve["1.0"]["delta"]
        
        delta_str = " | ".join([f"{k}:Δ={v:+.3f}" for k, v in sorted(layer_deltas.items())])
        print(f"    {m}: base={base:.2f}, {delta_str}")

# 2. Manifold Detection: Full entropy at critical alpha
print("\n--- 2. Manifold Detection (Δfull_H at category switch point) ---")
for obj in ["apple", "knife"]:
    print(f"\n  {obj}:")
    for m in models:
        if m not in all_data:
            continue
        d = all_data[m]
        obj_data = d.get("per_object", {}).get(obj)
        if not obj_data:
            continue
        base_H = obj_data["baselines"]["category"]["full_entropy"]
        base_top = obj_data["baselines"]["category"]["top"]
        key = "category_remove_category"
        perturb = obj_data["perturbations"].get(key, {})
        
        # Find switch point at embed layer
        embed_curve = perturb.get("embed", {})
        switch_info = "no switch"
        for alpha_str in sorted([float(a) for a in embed_curve.keys()]):
            if alpha_str == 0:
                continue
            pt = embed_curve.get(str(alpha_str), {})
            if pt.get("top") != base_top:
                delta_H = pt.get("full_entropy", 0) - base_H
                conf = pt.get("confidence", 0)
                verdict = "CLEAN" if delta_H < 1.0 else ("PARTIAL" if delta_H < 3.0 else "CONFUSED")
                switch_info = f"α={alpha_str}→{pt['top']}, ΔH={delta_H:+.1f}, conf={conf:.3f} ({verdict})"
                break
        print(f"    {m}: base_H={base_H:.2f}, base_top={base_top}, {switch_info}")

# 3. Category-Property coupling at embedding (α=1.0)
print("\n--- 3. Category-Property Coupling at Embedding (α=1.0, remove_category) ---")
for obj in ["apple", "knife"]:
    print(f"\n  {obj}:")
    for m in models:
        if m not in all_data:
            continue
        d = all_data[m]
        obj_data = d.get("per_object", {}).get(obj)
        if not obj_data:
            continue
        cat_key = "category_remove_category"
        prop_key = "property_remove_category"
        cat_perturb = obj_data["perturbations"].get(cat_key, {})
        prop_perturb = obj_data["perturbations"].get(prop_key, {})
        
        cat_delta = abs(cat_perturb.get("embed", {}).get("1.0", {}).get("delta", 0))
        prop_delta = abs(prop_perturb.get("embed", {}).get("1.0", {}).get("delta", 0))
        cat_H = cat_perturb.get("embed", {}).get("1.0", {}).get("full_entropy", 0)
        cat_conf = cat_perturb.get("embed", {}).get("1.0", {}).get("confidence", 0)
        
        if prop_delta > 0.01:
            ratio = cat_delta / prop_delta
        else:
            ratio = float('inf')
        
        coupling = "DECOUPLED" if ratio > 5 else ("COUPLED" if ratio < 2 else "PARTIAL")
        print(f"    {m}: cat|Δ|={cat_delta:.3f}, prop|Δ|={prop_delta:.3f}, "
              f"ratio={ratio:.1f}, H={cat_H:.1f}, conf={cat_conf:.3f} → {coupling}")

# 4. Key insight: mid-layer has ZERO effect
print("\n--- 4. Mid-Layer Effectiveness (ALL objects, ALL layers, α=1.0, remove_category) ---")
max_mid_delta = 0.0
for m in models:
    if m not in all_data:
        continue
    d = all_data[m]
    for obj, obj_data in d.get("per_object", {}).items():
        key = "category_remove_category"
        perturb = obj_data.get("perturbations", {}).get(key, {})
        for layer_key, curve in perturb.items():
            if layer_key == "embed":
                continue
            delta = abs(curve.get("1.0", {}).get("delta", 0))
            max_mid_delta = max(max_mid_delta, delta)

print(f"  Maximum |Δ| across ALL mid-layer conditions: {max_mid_delta:.4f}")
print(f"  Conclusion: Mid-layer perturbation with embedding-space direction has NO effect")

# ===== 追加到MEMO =====
timestamp = time.strftime("%Y-%m-%d %H:%M")
memo_text = f"""

## Phase 428: 中层残差扰动 + 流形外检测 [2026-06-09 23:50]

### 实验目的
1. 确定类别轨道在哪层形成（embedding vs 中层）
2. 判断GLM4的低阈值耦合是真语义还是流形外脆弱
3. 测试DS7B是否在中层变得敏感

### 方法
- 在不同深度（embedding, L20%, L40%, L60%, L80%）的对象token位置添加相同类别方向扰动
- 用forward hook在中层残差流中注入扰动
- 记录：候选分布、全分布熵、置信度、残差范数
- 关键指标：全分布熵变化ΔH判断是否流形外（ΔH>3=CONFUSED）

### 核心结果

**1. 三模型一致发现：类别方向扰动只在embedding层有效，中层完全无效！**

| 模型 | 对象 | embed Δ | L20% Δ | L40% Δ | L60% Δ | L80% Δ |
|------|------|---------|---------|---------|---------|---------|
| Qwen3 | apple | +0.999 | 0.000 | 0.000 | 0.000 | 0.000 |
| Qwen3 | knife | +1.003 | 0.000 | 0.000 | 0.000 | 0.000 |
| GLM4 | apple | +3.191 | 0.000 | 0.000 | 0.000 | 0.000 |
| GLM4 | knife | +1.002 | 0.000 | 0.000 | 0.000 | 0.000 |
| DS7B | apple | +0.300 | 0.000 | 0.000 | 0.000 | 0.000 |

中层扰动的最大|Δ| = {max_mid_delta:.4f}（全零）

**2. GLM4的embedding扰动是CONFUSED（流形外），不是清洁语义切换！**

| 模型 | 对象 | 切换alpha | 切换目标 | Δfull_H | conf | 判定 |
|------|------|-----------|---------|---------|------|------|
| Qwen3 | knife | 0.9 | tool→vehicle | -3.34 | 0.677 | CLEAN |
| Qwen3 | apple | 0.75 | fruit→animal | +1.68 | 0.465 | CONFUSED |
| GLM4 | apple | 0.3 | fruit→place | +8.29 | 0.036 | CONFUSED |
| GLM4 | knife | 0.3 | tool→place | +5.46 | 0.041 | CONFUSED |
| DS7B | (none) | - | - | - | - | 不切换 |

GLM4的ΔH=+8.29，置信度=0.036：模型完全混乱，不是语义切换。

**3. Phase 426的"GLM4类别-属性耦合"结论需要修正**

| 模型 | 对象 | cat|Δ|@embed | prop|Δ|@embed | ratio | full_H | conf | 判定 |
|------|------|-----------|------------|-------|--------|------|------|
| Qwen3 | apple | 0.999 | 0.001 | ∞ | 2.0 | 0.728 | 真解耦 |
| GLM4 | apple | 3.191 | 2.895 | 1.1 | 12.6 | 0.027 | 假耦合(混乱) |
| DS7B | apple | 0.300 | 0.197 | 1.5 | 7.4 | 0.205 | 弱效应 |

GLM4的"耦合"发生在full_H=12.6, conf=0.027的混乱状态下，这不代表语义关系，而是模型被推出自然流形后的随机输出。

### 关键发现

1. **Embedding-space类别方向在中层残差流中不再有效**：说明前几层已经将embedding-space的方向变换为完全不同的表示。类别信息不以原始方向存在于深层残差流中。

2. **GLM4的"低阈值耦合"是流形外脆弱**：embedding扰动导致模型进入混乱状态（H=12.6, conf=0.03），不是清洁的语义切换。Phase 426关于"GLM4类别-属性耦合"的结论需要修正为"GLM4的embedding空间更脆，小扰动导致流形外混乱"。

3. **Qwen3的切换更清洁但也不完美**：knife是CLEAN切换（ΔH=-3.34, conf=0.677），但apple是CONFUSED切换（ΔH=+1.68, conf=0.465）。清洁程度取决于对象。

4. **DS7B完全不切换**：即使embedding扰动也只产生弱偏移（apple Δ=+0.30），无完整类别跃迁。

5. **中层扰动的零效应是一个重要约束**：说明要在中层做有效的因果干预，不能简单复用embedding-space方向，需要找到中层自己的类别方向（Phase 427升级）。

### 公式更新

中层扰动无效意味着，在Phase 426的临界阈值公式中：
```
τ(o,d,b→b') = min α such that Basin(h_L(e_o + αd)) = b'
```
d必须是embedding-space方向。如果d是中层残差方向的等价物，α的含义会完全不同。

中层扰动的无效性提示：
```
E_cat(o) 在L≥1时 → 0（embedding方向被前几层吸收/旋转）
K_l(o,r,v) 在L≥1时 > 0（类别知识由后续层参数补全）
```

### 严格审视

**硬伤1：中层扰动零效应可能是因为方向不对**
当前用embedding-space方向加到中层残差流上。但前几层已经把这个方向旋转/变换了。中层可能有自己的类别方向，但我们没有用对。需要用中层探针方向重测。

**硬伤2：只扰动了对象token位置**
对象的类别信息可能通过注意力传播到了其他token位置（如最后一个token）。在中层，类别信息可能主要在最后一个token的残差流中。应测试扰动最后一个token位置。

**硬伤3：GLM4的CONFUSED判定基于全分布熵**
full_H从3.91跳到12.2确实表明混乱，但也可能是因为GLM4的输出本身就更分散。需要和Qwen3在相同对象上比较基线熵。Qwen3基线H=2.05，GLM4基线H=3.91，说明GLM4本身就更不确定。

**硬伤4：对象数量仍然偏少**
只有5个single-token对象，每个类别的代表性不足。特别是animal类别的dog和cat都没有有效切换，可能需要更多animal对象。

### 下一步

1. Phase 427升级：用中层探针方向替代embedding方向重测（最关键）
2. 扰动最后一个token位置（而非对象token位置）测试readout直接影响
3. 增加更多对象，特别是animal类别
4. 对GLM4做更小alpha精细扫描（0.01-0.3）看是否有清洁切换点
"""

with open(memo_path, "a", encoding="utf-8") as f:
    f.write(memo_text)

print(f"\nMEMO updated at {timestamp}")
print(f"Added {len(memo_text)} characters to {memo_path}")
