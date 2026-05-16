"""Phase 191 跨模型综合对比"""
import sys, json, os, glob
sys.stdout.reconfigure(encoding='utf-8')

# 加载所有模型数据
data_dir = "tests/glm5_temp"
models = ["qwen3", "glm4", "deepseek7b"]
all_data = {}

for model in models:
    pattern = os.path.join(data_dir, f"phase191_{model}_*.json")
    files = glob.glob(pattern)
    if files:
        with open(files[-1], 'r', encoding='utf-8') as f:
            all_data[model] = json.load(f)
            print(f"Loaded {model}: {files[-1]}")
    else:
        print(f"No data for {model}")

print(f"\n{'='*80}")
print("Phase 191 综合对比: 回路代数与角色绑定")
print(f"{'='*80}")

# ===== Exp1: 算子代数 =====
print(f"\n{'='*60}")
print("Exp1: 算子代数 — 三模型对比")
print(f"{'='*60}")

# 1. 算子间正交性
print(f"\n--- 算子间正交性: cos(d_A, d_B) ---")
print(f"{'算子对':<20} {'Qwen3':>10} {'GLM4':>10} {'DS7B':>10} {'趋势':>10}")
for pair_key in ["NOT_vs_PAST", "NOT_vs_PLURAL", "PAST_vs_PLURAL"]:
    vals = []
    for model in models:
        if model in all_data:
            val = all_data[model].get("exp1", {}).get("orthogonality", {}).get(pair_key, {}).get("mean_cos", "N/A")
            vals.append(val)
        else:
            vals.append("N/A")
    vals_str = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in vals]
    trend = ""
    if all(isinstance(v, float) for v in vals):
        if all(v < 0.3 for v in vals):
            trend = "弱相关"
        elif all(v < 0.5 for v in vals):
            trend = "中等相关"
        else:
            trend = "强相关"
    print(f"{pair_key:<20} {vals_str[0]:>10} {vals_str[1]:>10} {vals_str[2]:>10} {trend:>10}")

# 2. 算子加法性
print(f"\n--- 算子加法性: cos(d_combined, d1+d2) ---")
print(f"{'组合':<20} {'Qwen3':>10} {'GLM4':>10} {'DS7B':>10}")
for pair_name in ["NOT+PAST", "NOT+PLURAL", "PAST+PLURAL"]:
    vals = []
    for model in models:
        if model in all_data:
            val = all_data[model].get("exp1", {}).get("additivity", {}).get(pair_name, {}).get("mean_cos_d_combined_vs_d1_plus_d2", "N/A")
            vals.append(val)
        else:
            vals.append("N/A")
    vals_str = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in vals]
    print(f"{pair_name:<20} {vals_str[0]:>10} {vals_str[1]:>10} {vals_str[2]:>10}")

# 3. 三算子组合
print(f"\n--- 三算子组合: cos(d_all, d1+d2+d3) ---")
for model in models:
    if model in all_data:
        val = all_data[model].get("exp1", {}).get("triple_cos_mean", "N/A")
        res = all_data[model].get("exp1", {}).get("triple_residual_mean", "N/A")
        print(f"  {model}: cos={val:.4f}, residual={res:.4f}")

# 4. 交换律 (间接)
print(f"\n--- 算子交换律 (间接): d_NOT(base) vs d_NOT(PAST(base)) ---")
for model in models:
    if model in all_data:
        comm = all_data[model].get("exp1", {}).get("commutativity", {})
        for key, val in comm.items():
            cos1 = val.get("cos_d_op1_base_vs_d_op1_op2base", "N/A")
            cos2 = val.get("cos_d_op2_base_vs_d_op2_op1base", "N/A")
            print(f"  {model} {key}: cos1={cos1:.4f}, cos2={cos2:.4f}")

# ===== Exp2: 角色绑定 =====
print(f"\n{'='*60}")
print("Exp2: 角色绑定 — 三模型对比")
print(f"{'='*60}")

# 1. 主动 vs 被动 / 角色反转
print(f"\n--- 语态与角色对比 ---")
print(f"{'指标':<35} {'Qwen3':>10} {'GLM4':>10} {'DS7B':>10}")
for metric_key, metric_label in [
    ("active_passive_cos", "cos(主动, 被动[等价])"),
    ("active_reversed_cos", "cos(主动, 角色反转[不同])"),
    ("content_role_orthogonality", "cos(content, role_fiber)"),
    ("role_direction_consistency", "角色方向跨句对一致性"),
]:
    vals = []
    for model in models:
        if model in all_data:
            val = all_data[model].get("exp2", {}).get(metric_key, "N/A")
            vals.append(val)
        else:
            vals.append("N/A")
    vals_str = [f"{v:.4f}" if isinstance(v, float) else str(v) for v in vals]
    print(f"{metric_label:<35} {vals_str[0]:>10} {vals_str[1]:>10} {vals_str[2]:>10}")

# 2. Fiber structure支持
print(f"\n--- Fiber structure判断 ---")
for model in models:
    if model in all_data:
        cos_cr = all_data[model].get("exp2", {}).get("content_role_orthogonality", "N/A")
        if isinstance(cos_cr, float):
            judgment = "强支持" if abs(cos_cr) < 0.1 else ("支持" if abs(cos_cr) < 0.2 else ("弱支持" if abs(cos_cr) < 0.3 else "不支持"))
            print(f"  {model}: |cos(content, role)|={cos_cr:.4f} → {judgment}")

# 3. 角色反转差异的层间演化
print(f"\n--- 角色反转差异的关键层 ---")
for model in models:
    if model in all_data:
        diff_data = all_data[model].get("exp2", {}).get("role_diff_by_layer_key", {})
        print(f"  {model}: {diff_data}")

# ===== Exp4: 跨层状态精化 =====
print(f"\n{'='*60}")
print("Exp4: 跨层状态精化 — 三模型对比")
print(f"{'='*60}")

# 1. Peak layers
print(f"\n--- 各功能peak layer ---")
print(f"{'功能':<15} {'Qwen3':>15} {'GLM4':>15} {'DS7B':>15}")
for func in ["negation", "tense", "role_binding"]:
    vals = []
    for model in models:
        if model in all_data:
            peak = all_data[model].get("exp4", {}).get("peak_layers", {}).get(func, {})
            layer = peak.get("peak_layer", "N/A")
            diff = peak.get("peak_diff", "N/A")
            if isinstance(layer, int) and isinstance(diff, float):
                vals.append(f"L{layer}({diff:.1f})")
            else:
                vals.append(str(layer))
        else:
            vals.append("N/A")
    print(f"{func:<15} {vals[0]:>15} {vals[1]:>15} {vals[2]:>15}")

# 2. Curve shapes
print(f"\n--- 区分曲线形状 ---")
for model in models:
    if model in all_data:
        shapes = all_data[model].get("exp4", {}).get("curve_shapes", {})
        shape_str = ", ".join([f"{k}={v.get('shape', '?')}" for k, v in shapes.items()])
        print(f"  {model}: {shape_str}")

# 3. 各功能在最后层的区分能力
print(f"\n--- 最后层区分能力 ---")
for model in models:
    if model in all_data:
        n_layers = all_data[model].get("model_info", {}).get("n_layers", "?")
        disc = all_data[model].get("exp4", {}).get("discrimination_summary", {})
        for func in ["negation", "tense", "role_binding"]:
            last_key = f"L{n_layers}"
            if func in disc and last_key in disc[func]:
                d = disc[func][last_key]
                print(f"  {model} {func}: ||Δh||={d.get('mean_diff_norm', '?'):.4f}, cos={d.get('mean_cosine', '?'):.4f}")

# ===== 关键发现总结 =====
print(f"\n{'='*80}")
print("★★★ Phase 191 关键发现总结 ★★★")
print(f"{'='*80}")

print("""
1. 算子代数性质:
   - 算子间弱相关(cos≈0.16-0.32): NOT/PAST/PLURAL方向不是完全正交
     但也不是同方向 → 支持独立算子假说
   - 加法性部分成立(cos≈0.85): 组合≈求和, 但有显著非线性残差
   - 交换律不成立(cos≈0.74-0.80): 算子效果依赖于上下文
     → 语义算子形成非交换代数结构

2. 角色绑定的Fiber structure:
   - 三模型一致: cos(content, role_fiber)≈0.07-0.16
   - 角色差异方向与内容方向近似正交
   - 支持Fiber Bundle模型: base=内容, fiber=关系算子

3. 跨层状态精化:
   - 所有功能: 单调递增(monotone_increasing)
   - role_binding区分度最大(三模型一致)
   - 语义区分在深层最强 → 深层在做"符号精化"

4. 路由拓扑:
   - 简单陈述/否定/疑问: 路由相似
   - 条件/因果: 路由显著不同(entropy_diff > 0.17)
   → 条件和因果句有特殊的信息路由模式
""")
