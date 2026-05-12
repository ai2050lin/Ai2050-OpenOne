"""
Phase 136 汇总分析: 激活边界分析
================================
"""
import json
import numpy as np

def load_results(model_name):
    path = f"tests/glm5_temp/phase136_{model_name}_activation_boundary.json"
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

models = ["qwen3", "glm4", "deepseek7b"]
model_labels = {"qwen3": "Qwen3", "glm4": "GLM4", "deepseek7b": "DS7B"}
results = {}
for m in models:
    try:
        results[m] = load_results(m)
    except:
        print(f"WARNING: Cannot load {m}")

print("=" * 70)
print("Phase 136 汇总: 激活边界分析")
print("=" * 70)

# ============================================================
# 1. Spikiness 跨模型对比
# ============================================================
print("\n## 1. MLP Flip Spikiness (分段程序假说的核心验证) ##")
print("   Spikiness = max_flip / mean_flip, >1表示存在相变")

for pair_type in ["semantic", "negation"]:
    print(f"\n   --- {pair_type} 插值 ---")
    header = f"   {'Layer':<8}"
    for m in models:
        if m in results:
            header += f" {model_labels[m]:<12}"
    print(header)
    
    # 收集所有层的spikiness
    all_layers = set()
    for m in models:
        if m in results:
            agg = results[m].get("exp1_interpolation", {}).get(pair_type, {}).get("aggregated", {})
            all_layers.update(agg.keys())
    
    for lk in sorted(all_layers, key=lambda x: int(x[1:])):
        row = f"   {lk:<8}"
        for m in models:
            if m in results:
                agg = results[m].get("exp1_interpolation", {}).get(pair_type, {}).get("aggregated", {})
                spike = agg.get(lk, {}).get("t0.0_spikiness_mean", None)
                if spike is not None:
                    row += f" {spike:<12.2f}"
                else:
                    row += f" {'N/A':<12}"
        print(row)

# ============================================================
# 2. Max Flip Rate 跨模型对比
# ============================================================
print("\n## 2. MLP Max Flip Rate (最大激活翻转率) ##")

for pair_type in ["semantic", "negation"]:
    print(f"\n   --- {pair_type} 插值 ---")
    header = f"   {'Layer':<8}"
    for m in models:
        if m in results:
            header += f" {model_labels[m]:<12}"
    print(header)
    
    all_layers = set()
    for m in models:
        if m in results:
            agg = results[m].get("exp1_interpolation", {}).get(pair_type, {}).get("aggregated", {})
            all_layers.update(agg.keys())
    
    for lk in sorted(all_layers, key=lambda x: int(x[1:])):
        row = f"   {lk:<8}"
        for m in models:
            if m in results:
                agg = results[m].get("exp1_interpolation", {}).get(pair_type, {}).get("aggregated", {})
                max_flip = agg.get(lk, {}).get("t0.0_max_flip_mean", None)
                if max_flip is not None:
                    row += f" {max_flip:<12.4f}"
                else:
                    row += f" {'N/A':<12}"
        print(row)

# ============================================================
# 3. Total Spikes (相变点数)
# ============================================================
print("\n## 3. Total Phase Transition Spikes (t0.0, 15句对x50插值点) ##")

for pair_type in ["semantic", "negation"]:
    print(f"\n   --- {pair_type} 插值 ---")
    header = f"   {'Layer':<8}"
    for m in models:
        if m in results:
            header += f" {model_labels[m]:<12}"
    print(header)
    
    all_layers = set()
    for m in models:
        if m in results:
            agg = results[m].get("exp1_interpolation", {}).get(pair_type, {}).get("aggregated", {})
            all_layers.update(agg.keys())
    
    for lk in sorted(all_layers, key=lambda x: int(x[1:])):
        row = f"   {lk:<8}"
        for m in models:
            if m in results:
                agg = results[m].get("exp1_interpolation", {}).get(pair_type, {}).get("aggregated", {})
                spikes = agg.get(lk, {}).get("t0.0_total_spikes", None)
                if spikes is not None:
                    row += f" {spikes:<12}"
                else:
                    row += f" {'N/A':<12}"
        print(row)

# ============================================================
# 4. Attention Switch Rate
# ============================================================
print("\n## 4. Attention Max Switch Rate (最大attention边切换率) ##")

for pair_type in ["semantic", "negation"]:
    print(f"\n   --- {pair_type} 插值 ---")
    header = f"   {'Layer':<8}"
    for m in models:
        if m in results:
            header += f" {model_labels[m]:<12}"
    print(header)
    
    all_layers = set()
    for m in models:
        if m in results:
            agg = results[m].get("exp1_interpolation", {}).get(pair_type, {}).get("aggregated", {})
            all_layers.update(agg.keys())
    
    for lk in sorted(all_layers, key=lambda x: int(x[1:])):
        row = f"   {lk:<8}"
        for m in models:
            if m in results:
                agg = results[m].get("exp1_interpolation", {}).get(pair_type, {}).get("aggregated", {})
                switch = agg.get(lk, {}).get("attn_max_switch_mean", None)
                if switch is not None:
                    row += f" {switch:<12.4f}"
                else:
                    row += f" {'N/A':<12}"
        print(row)

# ============================================================
# 5. 边界密度 (仅Qwen3可靠)
# ============================================================
print("\n## 5. 边界密度 (Boundary Density) ##")
print("   注意: GLM4/DS7B为8bit量化, 结果可能受量化噪声影响")

for m in models:
    if m not in results:
        continue
    print(f"\n   --- {model_labels[m]} ---")
    agg2 = results[m].get("exp2_boundary_density", {}).get("aggregated", {})
    
    header = f"   {'eps':<8}"
    for lk in sorted(agg2.keys(), key=lambda x: int(x[1:])):
        header += f" {lk:<12}"
    print(header)
    
    for eps in [0.01, 0.05, 0.1, 0.5, 1.0]:
        row = f"   {eps:<8}"
        for lk in sorted(agg2.keys(), key=lambda x: int(x[1:])):
            mlp_key = f"eps{eps}_mlp_density_mean"
            val = agg2.get(lk, {}).get(mlp_key, None)
            if val is not None:
                row += f" {val:<12.3f}"
            else:
                row += f" {'N/A':<12}"
        print(row)

# ============================================================
# 6. 关键发现总结
# ============================================================
print("\n" + "=" * 70)
print("Phase 136 关键发现")
print("=" * 70)

print("""
1. Transformer确实是"分段条件程序":
   - 所有模型所有层的Spikiness > 1.0
   - 存在明确的激活翻转尖峰, 而非平滑变化
   - 证明: 输入插值路径穿过离散的激活边界

2. 深层有更尖锐的相变:
   - Qwen3: L0 spikiness=1.3 → L32 spikiness=5.1 (语义), L32=3.95 (否定)
   - DS7B: L27 spikiness=3.24 (语义), 4.41 (否定)
   - 深层的计算更"分段化"

3. 否定vs语义的spikiness差异跨模型不一致:
   - Qwen3: 语义spikiness > 否定 (5.1 vs 3.95)
   - GLM4: 否定spikiness > 语义 (2.05 vs 1.79 在L35)
   - DS7B: 否定spikiness > 语义 (4.41 vs 3.24 在L27)
   - 可能是模型规模/架构差异

4. 8bit量化严重影响边界密度测量:
   - Qwen3 (bfloat16): eps=0.01时边界密度很低 (0-0.03)
   - GLM4 (8bit): eps=0.01时所有层边界密度=1.0 (噪声)
   - DS7B (8bit): 中间层边界密度=1.0 (噪声)
   - 结论: 边界密度实验只能用bfloat16模型

5. 最后层的"异常低边界密度" (DS7B):
   - L27在eps=0.01时密度=0.24, eps=0.1时=0.17
   - 这可能意味着最后层的计算更加"鲁棒/确定性"
   - 与Phase 135的发现一致: 最后层Jaccard回升

6. Attention比MLP更稳定:
   - Attention max switch rate < MLP max flip rate
   - 即使在eps=1.0, attention边界密度仍低于MLP
   - 支持Phase 134-135的"注意力更稳定"结论
""")
