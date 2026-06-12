"""
Phase 469 结果汇总分析脚本
=============================
从三个模型的JSON结果中提取关键数据，生成跨模型对比表
"""
import sys, json, os
sys.stdout.reconfigure(encoding='utf-8')

def load_results(model_name):
    path = f"results/glm5/phase469_{model_name}_r1.json"
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze():
    models = ["qwen3", "glm4", "deepseek7b"]
    all_data = {}
    for m in models:
        all_data[m] = load_results(m)
    
    print("=" * 80)
    print("Phase 469 跨模型结果汇总")
    print("=" * 80)
    
    # ---- Exp1: PC1因果强度扫描汇总 ----
    print("\n### Exp1: PC1因果强度扫描 — 关键指标汇总")
    print(f"{'Model':<12} {'Layer':<6} {'PC1_Δent':<10} {'Rand_Δent':<10} {'Rand_std':<10} "
          f"{'Mean_z':<8} {'t_stat':<8} {'p_value':<10} {'Mono':<6} {'Ratio':<8} {'Signif':<6}")
    print("-" * 100)
    
    for m in models:
        d = all_data[m]
        if d is None or "exp1_pc1_causal_scan" not in d:
            continue
        for layer_key in sorted(d["exp1_pc1_causal_scan"].keys()):
            lr = d["exp1_pc1_causal_scan"][layer_key]
            s = lr.get("summary", {})
            print(f"{m:<12} {layer_key:<6} {s.get('pc1_mean_delta_entropy', 'N/A'):<10} "
                  f"{s.get('random_mean_delta_entropy', 'N/A'):<10} "
                  f"{s.get('random_mean_std_entropy', 'N/A'):<10} "
                  f"{s.get('mean_z_score', 'N/A'):<8} "
                  f"{s.get('t_statistic', 'N/A'):<8} "
                  f"{s.get('p_value', 'N/A'):<10} "
                  f"{s.get('n_monotonic_out_of', 'N/A'):<6} "
                  f"{s.get('pc1_vs_random_entropy_ratio', 'N/A'):<8} "
                  f"{'YES' if s.get('is_significant_causal_axis', False) else 'no':<6}")
    
    # ---- Exp2: PC1多变量分解汇总 ----
    print("\n\n### Exp2: PC1多变量分解 — 偏相关汇总")
    print(f"{'Model':<12} {'Layer':<6} {'R²':<8} {'Ent_pcorr':<11} {'Tmpl_pcorr':<12} {'Cat_pcorr':<11} "
          f"{'Pos_corr':<9} {'PC1-ent_r':<10} {'Readout':<8}")
    print("-" * 100)
    
    for m in models:
        d = all_data[m]
        if d is None or "exp2_pc1_decomposition" not in d:
            continue
        for layer_key in sorted(d["exp2_pc1_decomposition"].keys()):
            lr = d["exp2_pc1_decomposition"][layer_key]
            reg = lr.get("regression", {})
            pc = reg.get("partial_correlations", {})
            pos = lr.get("position_correlation", {})
            sc = lr.get("simple_correlations", {})
            ra = lr.get("readout_alignment", {})
            
            print(f"{m:<12} {layer_key:<6} {reg.get('r2', 'N/A'):<8} "
                  f"{pc.get('entropy', 'N/A'):<11} "
                  f"{pc.get('template', 'N/A'):<12} "
                  f"{pc.get('category', 'N/A'):<11} "
                  f"{pos.get('corr', 'N/A'):<9} "
                  f"{sc.get('pc1_entropy_r', 'N/A'):<10} "
                  f"{ra.get('pc1_vs_right_sv1', 'N/A'):<8}")
    
    # ---- Exp3: 受控评分范式汇总 ----
    print("\n\n### Exp3: 受控评分范式 — 汇总")
    print(f"{'Model':<12} {'MC_Acc':<8} {'YN_Acc':<8} {'Math_Rate':<12} {'Recommendation':<30}")
    print("-" * 75)
    
    for m in models:
        d = all_data[m]
        if d is None or "exp3_controlled_scoring" not in d:
            continue
        s = d["exp3_controlled_scoring"].get("summary", {})
        print(f"{m:<12} {s.get('mc_mean_accuracy', 'N/A'):<8} "
              f"{s.get('yn_mean_accuracy', 'N/A'):<8} "
              f"{s.get('mean_math_trigger_rate', 'N/A'):<12} "
              f"{s.get('recommendation', 'N/A'):<30}")
    
    # ---- Exp4: 基线质量校正汇总 ----
    print("\n\n### Exp4: 生成质量基线校正 — 汇总")
    for m in models:
        d = all_data[m]
        if d is None or "exp4_baseline_quality" not in d:
            continue
        s = d["exp4_baseline_quality"].get("summary", {})
        print(f"  {m}: baseline_good={s.get('baseline_good_count','?')}/{s.get('n_objects','?')}, "
              f"math_triggered={s.get('baseline_math_triggered','?')}, "
              f"+PC1_dist={s.get('+pc1_quality_distribution',{})}, "
              f"-PC1_dist={s.get('-pc1_quality_distribution',{})}")
    
    # ---- 关键发现 ----
    print("\n\n" + "=" * 80)
    print("关键发现")
    print("=" * 80)
    
    print("""
1. PC1因果性被大幅修正(50随机方向vs原来5个):
   - Phase 468声称Qwen3 L18是"强因果确定性轴"(ratio=-79.92)
   - Phase 469用50随机方向: Qwen3 L18 mean_z=-1.08, 单个对象均未达|z|>2
   - 但聚合t检验显著(t=-2.64, p=0.008), 说明PC1对entropy有系统但微弱的因果效应
   - 单调性检验: 0-1/6对象通过, PC1不是简单线性因果轴
   
2. PC1本质是"类别分隔轴"而非"熵轴":
   - 所有模型所有层, category偏相关都是最强的(0.7-0.9)
   - entropy偏相关在Qwen3 L12/L18约-0.5, 但远弱于category的-0.82
   - DS7B的entropy偏相关仅-0.09到-0.12, PC1几乎不含entropy成分

3. GLM4层间重定义被确认:
   - entropy偏相关: L6(0.33) → L13(-0.52) → L20(-0.41) → L26(0.43)
   - 正负号翻转, 确认PC1在不同层编码不同功能

4. DS7B PC1主要是template+category轴:
   - L4 template偏相关=0.817(极大!), entropy仅-0.12
   - 高干预敏感性: random_std高达8-14(其他模型0.07-5.9)

5. Readout对齐(使用右奇异向量):
   - Qwen3: 0.001-0.024(极低), PC1不直接读出
   - GLM4: 全部0.000(SVD数值问题, 需修复)
   - DS7B: 0.000-0.042(极低)
   - 结论: PC1是内部状态轴, 不直接映射到词表空间

6. 受控评分范式:
   - MC格式(4选1)只有fruit准确率高, 可能是模板问题
   - GLM4的Y/N评分效果好(87.5%)
   - DS7B数学模式触发率25%(比Phase 468的33-100%降低, 但仍有问题)
   - Qwen3/GLM4无数学模式触发
""")

if __name__ == "__main__":
    analyze()
