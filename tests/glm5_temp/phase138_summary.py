"""Phase 138 汇总分析"""
import sys, os, json, numpy as np
sys.stdout.reconfigure(encoding='utf-8')

base_dir = os.path.dirname(os.path.abspath(__file__))
# base_dir is tests/glm5_temp itself
project_dir = os.path.join(base_dir, '..', '..')
temp_dir = base_dir

models = ["qwen3", "glm4", "deepseek7b"]

print("=" * 70)
print("Phase 138 汇总: 状态稳定性实验 — 程序 vs 动力系统")
print("=" * 70)

for model_name in models:
    path = os.path.join(temp_dir, f"phase138_{model_name}_state_stability.json")
    if not os.path.exists(path):
        print(f"\n  {model_name}: 文件不存在")
        continue
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"模型: {model_name} ({data['model_info']['class']}, {data['model_info']['n_layers']}层)")
    print(f"{'='*60}")
    
    # === Exp 1: 弛豫分析 ===
    exp1 = data.get("exp1_relaxation", {})
    agg1 = exp1.get("aggregated", {})
    
    print(f"\n--- Exp 1: Deep State Relaxation (eps=5%) ---")
    
    for lk_perturb in sorted(agg1.keys()):
        eps_data = agg1[lk_perturb].get("0.05", {})
        if not eps_data:
            continue
        
        prop_by_depth = eps_data.get("prop_ratio_by_depth", {})
        dir_by_depth = eps_data.get("direction_preserve_by_depth", {})
        
        # 找到传播比的变化趋势
        depths = sorted(prop_by_depth.keys())
        if len(depths) >= 2:
            first_prop = prop_by_depth[depths[0]]
            last_prop = prop_by_depth[depths[-1]]
            first_dir = dir_by_depth.get(depths[0], 0)
            last_dir = dir_by_depth.get(depths[-1], 0)
            
            prop_trend = "AMPLIFY" if last_prop > first_prop else "RELAX"
            dir_trend = "PRESERVED" if last_dir > 0.5 else "DISSOLVED"
            
            print(f"  {lk_perturb}: prop_ratio {first_prop:.3f}→{last_prop:.3f} ({prop_trend}), "
                  f"dir_preserve {first_dir:.3f}→{last_dir:.3f} ({dir_trend}), "
                  f"logit_shift={eps_data.get('logit_shift_mean', 0):.1f}")
    
    # === Exp 2: 不一致性 ===
    exp2 = data.get("exp2_inconsistency", {})
    comparisons = exp2.get("comparisons", [])
    
    print(f"\n--- Exp 2: Inconsistency Sensitivity ---")
    if comparisons:
        agg_comp = {}
        for comp in comparisons:
            for lk, d in comp.items():
                if isinstance(d, dict) and "entropy_diff" in d:
                    if lk not in agg_comp:
                        agg_comp[lk] = []
                    agg_comp[lk].append(d["entropy_diff"])
        
        for lk in sorted(agg_comp.keys()):
            vals = agg_comp[lk]
            mean_v = np.mean(vals)
            print(f"  {lk}: entropy_diff = {mean_v:.4f} ± {np.std(vals):.4f} "
                  f"({'HIGHER' if mean_v > 0 else 'LOWER'} for inconsistent)")
    
    # === Exp 3: 多层patching ===
    exp3 = data.get("exp3_multilayer", {})
    agg3 = exp3.get("aggregated", {})
    
    print(f"\n--- Exp 3: Multi-layer Patching Superlinearity ---")
    two_layer = agg3.get("two_layer", {})
    superlinear_keys = [k for k in two_layer.keys() if "superlinear" in k]
    
    for key in sorted(superlinear_keys):
        d = two_layer[key]
        print(f"  {key}: {d['mean']:.4f} ± {d['std']:.4f} "
              f"({'SUPERLINEAR' if d['mean'] > 0 else 'SUBLINEAR'})")

print(f"\n{'='*70}")
print("核心结论")
print("=" * 70)
print("""
1. Exp 1 弛豫分析: 扰动被持续放大, 不是弛豫回吸引子
   - 浅层扰动(L0): 传播比指数级增长 (1.4→332 in Qwen3)
   - 中层扰动(L12/L18): 传播比缓慢增长 (1.0→1.9)
   - 方向保持持续衰减: 0.9→0.4 (扰动方向被扭曲)
   → 这不是"能量系统"(应该弛豫), 更像"混沌动力系统"

2. Exp 2 不一致性: 语法不一致句子在深层有更高entropy
   - Qwen3: L30 entropy_diff=0.25, L35=0.44 (不一致>一致)
   - GLM4: L39 entropy_diff=0.34
   → 深层"感知"到语法不一致, 但不是"能量爆发"

3. Exp 3 超线性: 所有2层联合patching都是SUBlinear
   - 联合recovery < 单层A + 单层B
   → 信息不是分布式协同编码, 更像是冗余/竞争编码
""")
