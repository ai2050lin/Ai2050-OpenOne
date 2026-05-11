"""Phase 130 汇总: Jacobian传播流与约束传播分析"""
import sys, json, os
sys.stdout.reconfigure(encoding='utf-8')

base = "d:/Ai2050/TransformerLens-Project/tests/glm5_temp"
models = {"qwen3": "qwen3", "deepseek7b": "deepseek7b", "glm4": "glm4"}

all_data = {}
for m, short in models.items():
    path = os.path.join(base, f"phase130_{short}_jacobian_flow.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            all_data[m] = json.load(f)

print("="*70)
print("Phase 130 汇总: Jacobian传播流与约束传播分析")
print("="*70)

# === Exp 2: 扰动传播追踪 ===
print("\n## Exp 2: 扰动传播追踪")

print("\n### 扰动放大倍数 (amplification) — 越深层注入, 放大越大")
print(f"{'注入层':<12} {'Qwen3':<12} {'DS7B':<12} {'GLM4':<12}")
for model_name, data in all_data.items():
    exp2 = data.get("exp2_perturbation_propagation", {})
    inject_summary = exp2.get("inject_summary", {})
    break  # 获取层列表

inject_layers = all_data.get("qwen3", {}).get("exp2_perturbation_propagation", {}).get("inject_layers", [])
for inject_l in inject_layers:
    ik = f"inject_L{inject_l}"
    vals = []
    for m in ["qwen3", "deepseek7b", "glm4"]:
        if m in all_data:
            v = all_data[m].get("exp2_perturbation_propagation", {}).get("inject_summary", {}).get(ik, {}).get("mean_amp", "N/A")
            if isinstance(v, float):
                vals.append(f"{v:.1f}")
            else:
                vals.append(str(v))
        else:
            vals.append("N/A")
    print(f"L{inject_l:<10} {vals[0]:<12} {vals[1]:<12} {vals[2]:<12}")

print("\n### 方向保持度 (cos_with_inject) — 越深层注入, 方向保持越低")
for inject_l in inject_layers:
    ik = f"inject_L{inject_l}"
    vals = []
    for m in ["qwen3", "deepseek7b", "glm4"]:
        if m in all_data:
            v = all_data[m].get("exp2_perturbation_propagation", {}).get("inject_summary", {}).get(ik, {}).get("mean_cos", "N/A")
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        else:
            vals.append("N/A")
    print(f"L{inject_l:<10} {vals[0]:<12} {vals[1]:<12} {vals[2]:<12}")

# === Exp 4: 组合传播 ===
print("\n\n## Exp 4: 组合传播 — 多约束叠加")

print("\n### 否定+时态 (Neg+Past) 组合")
print(f"{'层对':<12} {'Qwen3 lin_acc':<16} {'Qwen3 nl_ratio':<16} {'DS7B lin_acc':<16} {'DS7B nl_ratio':<16} {'GLM4 lin_acc':<16} {'GLM4 nl_ratio':<16}")
for pair_key in ["L0->L1", "L9->L10", "L18->L19", "L27->L28", "L7->L8", "L14->L15", "L21->L22", "L10->L11", "L20->L21", "L30->L31"]:
    vals = []
    for m in ["qwen3", "deepseek7b", "glm4"]:
        if m in all_data:
            d = all_data[m].get("exp4_compositional_flow", {}).get("neg_past_composition", {}).get(pair_key, {})
            la = d.get("linear_accuracy", "N/A")
            nr = d.get("nonlinearity_ratio", "N/A")
            if isinstance(la, float):
                vals.append(f"{la:.4f}")
            else:
                vals.append(str(la))
            if isinstance(nr, float):
                vals.append(f"{nr:.4f}")
            else:
                vals.append(str(nr))
        else:
            vals.extend(["N/A", "N/A"])
    if any(v != "N/A" for v in vals):
        print(f"{pair_key:<12} {vals[0]:<16} {vals[1]:<16} {vals[2]:<16} {vals[3]:<16} {vals[4]:<16} {vals[5]:<16}")

print("\n### 否定+被动 (Neg+Passive) 组合")
for pair_key in ["L0->L1", "L9->L10", "L18->L19", "L27->L28", "L7->L8", "L14->L15", "L21->L22", "L10->L11", "L20->L21", "L30->L31"]:
    vals = []
    for m in ["qwen3", "deepseek7b", "glm4"]:
        if m in all_data:
            d = all_data[m].get("exp4_compositional_flow", {}).get("neg_passive_composition", {}).get(pair_key, {})
            la = d.get("linear_accuracy", "N/A")
            nr = d.get("nonlinearity_ratio", "N/A")
            if isinstance(la, float):
                vals.append(f"{la:.4f}")
            else:
                vals.append(str(la))
            if isinstance(nr, float):
                vals.append(f"{nr:.4f}")
            else:
                vals.append(str(nr))
        else:
            vals.extend(["N/A", "N/A"])
    if any(v != "N/A" for v in vals):
        print(f"{pair_key:<12} {vals[0]:<16} {vals[1]:<16} {vals[2]:<16} {vals[3]:<16} {vals[4]:<16} {vals[5]:<16}")

print("\n### 三重组合 (Neg+Past+Passive) 非线性比")
for pair_key in ["L0->L1", "L9->L10", "L18->L19", "L27->L28", "L7->L8", "L14->L15", "L21->L22", "L10->L11", "L20->L21", "L30->L31"]:
    vals = []
    for m in ["qwen3", "deepseek7b", "glm4"]:
        if m in all_data:
            d = all_data[m].get("exp4_compositional_flow", {}).get("triple_composition", {}).get(pair_key, {})
            nr = d.get("nonlinearity_ratio", "N/A")
            if isinstance(nr, float):
                vals.append(f"{nr:.4f}")
            else:
                vals.append(str(nr))
        else:
            vals.append("N/A")
    if any(v != "N/A" for v in vals):
        print(f"{pair_key:<12} {vals[0]:<12} {vals[1]:<12} {vals[2]:<12}")

# === Exp 5: 轨迹吸引子 ===
print("\n\n## Exp 5: 轨迹吸引子 — 类内vs跨类距离")

print("\n### Discrimination (cross-intra) by layer")
for m in ["qwen3", "deepseek7b", "glm4"]:
    if m in all_data:
        disc = all_data[m].get("exp5_trajectory_attractor", {}).get("discrimination", {})
        if disc:
            # 按层排序
            sorted_layers = sorted(disc.keys(), key=lambda x: int(x[1:]))
            vals = [f"{disc[lk]:.4f}" for lk in sorted_layers]
            print(f"  {m}: {', '.join(f'{lk}={v}' for lk, v in zip(sorted_layers, vals))}")
            
            # 找最大discrimination层
            max_lk = max(disc, key=disc.get)
            print(f"    → Peak discrimination at {max_lk} = {disc[max_lk]:.4f}")

print("\n" + "="*70)
print("关键发现总结:")
print("="*70)

print("""
1. 扰动传播: 扰动被指数放大 (L0: 13-205x → L27-30: 110-605x)
   - 放大倍数随层深递增, 说明系统在深层对扰动极敏感
   - 方向保持度(cos)随层深递减(0.13→0.02), 说明扰动被旋转到新方向
   - 这就是"条件传播"的物理体现: 扰动被算子J_l不断变换

2. 组合传播: 非线性交互随层深增加
   - 早期层(L0): 否定+被动 nearly linear (0.95-0.99), nl_ratio≈0.11-0.18
   - 中间层(L9-10): 部分非线性 (0.83-0.91), nl_ratio≈0.50-0.65
   - 深层(L27-30): 强非线性 (0.68-0.86), nl_ratio≈0.79-0.92
   - 这说明: 语法约束的组合在深层是高度非线性的!

3. 三重组合: 非线性效应更强
   - L0: nl_ratio≈0.15-0.42
   - L10: nl_ratio≈0.50-0.73
   - L20-30: nl_ratio≈0.78-0.89
   - 随约束增多, 非线性累积!

4. 轨迹吸引子: 类别区分在中间层最强
   - Qwen3: Peak at L9 (disc=0.1107) and L30 (disc=0.1255)
   - DS7B: Peak at L8-12 (disc=0.0895-0.1027)
   - GLM4: Peak at L8-12 (disc=0.1059-0.1361)
   - 中间层有最强的语义区分度

5. 否定+时态 vs 否定+被动:
   - 否定+被动在早期层几乎是线性的(0.95-0.99), nl_ratio≈0.11
   - 否定+时态从一开始就是非线性的(0.65-0.89), nl_ratio≈1.13-1.29
   - 说明: 不同约束类型的组合机制不同
""")
