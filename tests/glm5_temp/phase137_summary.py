"""
Phase 137 汇总分析 — 三模型因果贡献分析
=========================================
"""
import sys, os, json, numpy as np

def load_results(model_name):
    path = f"tests/glm5_temp/phase137_{model_name}_causal_contribution.json"
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_patching():
    """分析patching结果"""
    print("=" * 70)
    print("Phase 137 汇总: Activation Patching")
    print("=" * 70)
    
    for model_name in ["qwen3", "glm4", "deepseek7b"]:
        data = load_results(model_name)
        if data is None:
            print(f"\n{model_name}: 数据缺失")
            continue
        
        print(f"\n--- {model_name} ---")
        exp1 = data.get("exp1_patching", {})
        
        for pair_type in ["negation", "tense", "semantic"]:
            agg = exp1.get(pair_type, {}).get("aggregated", {})
            if not agg:
                continue
            
            print(f"\n  {pair_type} (diff-pos patching):")
            # 按层排序
            layers_sorted = sorted(agg.keys(), key=lambda x: int(x[1:]))
            for lk in layers_sorted:
                d = agg[lk]
                cr = d.get("cosine_recovery_mean", 0)
                cr_std = d.get("cosine_recovery_std", 0)
                de = d.get("directed_effect_mean", 0)
                rr = d.get("relative_recovery_mean", 0)
                n = d.get("n_pairs", 0)
                print(f"    {lk}: cosine_recovery={cr:.4f}±{cr_std:.4f}, "
                      f"relative_recovery={rr:.4f}, directed={de:.2f}, n={n}")

def analyze_logit_lens():
    """分析logit lens结果"""
    print("\n" + "=" * 70)
    print("Phase 137 汇总: Logit Lens")
    print("=" * 70)
    
    for model_name in ["qwen3", "glm4", "deepseek7b"]:
        data = load_results(model_name)
        if data is None:
            continue
        
        print(f"\n--- {model_name} ---")
        exp2 = data.get("exp2_logit_lens", {})
        agg = exp2.get("aggregated", {})
        
        for pair_type in ["negation", "tense", "semantic"]:
            if pair_type not in agg:
                continue
            print(f"\n  {pair_type}:")
            layers_sorted = sorted(agg[pair_type].keys(), key=lambda x: int(x[1:]))
            for lk in layers_sorted:
                d = agg[pair_type][lk]
                diff_ld = d.get("diff_logit_diff_mean", 0)
                last_ld = d.get("last_logit_diff_mean", 0)
                diff_kl = d.get("diff_kl_mean", 0)
                print(f"    {lk}: diff_pos_logit_diff={diff_ld:.2f}, "
                      f"last_pos_logit_diff={last_ld:.2f}, diff_kl={diff_kl:.4f}")

def analyze_weighted():
    """分析加权贡献结果"""
    print("\n" + "=" * 70)
    print("Phase 137 汇总: Weighted Contribution")
    print("=" * 70)
    
    for model_name in ["qwen3", "glm4", "deepseek7b"]:
        data = load_results(model_name)
        if data is None:
            continue
        
        print(f"\n--- {model_name} ---")
        exp3 = data.get("exp3_weighted", {})
        agg = exp3.get("aggregated", {})
        
        for pair_type in ["negation", "tense", "semantic"]:
            if pair_type not in agg:
                continue
            print(f"\n  {pair_type}:")
            layers_sorted = sorted(agg[pair_type].keys(), key=lambda x: int(x[1:]))
            for lk in layers_sorted:
                d = agg[pair_type][lk]
                wj = d.get("weighted_jaccard_mean", 0)
                bj = d.get("binary_jaccard_mean", 0)
                dn = d.get("delta_h_norm_last_mean", 0)
                lc = d.get("logit_contribution_norm_last_mean", 0)
                eff = d.get("contribution_efficiency_mean", 0)
                pr = d.get("projection_ratio_mean", 0)
                print(f"    {lk}: w_jac={wj:.4f}, b_jac={bj:.4f}, "
                      f"delta_norm={dn:.2f}, logit_contrib={lc:.2f}, "
                      f"efficiency={eff:.4f}, proj_ratio={pr:.4f}")

def key_findings():
    """打印关键发现"""
    print("\n" + "=" * 70)
    print("Phase 137 关键发现总结")
    print("=" * 70)
    
    print("""
1. PATCHING RECOVERY随深度单调递减 (最关键发现)
   Qwen3 Negation: L0=0.993, L12=0.777, L18=0.706, L24=0.316, L35=0.102
   GLM4 Negation:  L0=0.970, L21=0.310, L39=-0.237
   DS7B (可用对): L0≈0.95, L14≈0.5, L27≈-0.5

   解释: 不是"深层信息少", 而是"深层hidden state高度上下文依赖"
   - 在base上下文中注入modified的深层hidden state → 上下文冲突
   - 在base上下文中注入modified的浅层hidden state → 后续层可以"调和"
   - 这证明: Transformer需要从头到尾一致地处理修改信息

2. LOGIT LENS: 修改信息随深度累积
   Negation diff_pos logit_diff: L0=103.8 → L33=2221.0 (21倍增长)
   Tense diff_pos logit_diff:    L0=55.1  → L33=1584.4 (29倍增长)
   KL divergence: L0=0.035 → L30=15.3 (437倍增长!)
   
   → 每层都在累积修改信息, 不是某几层"突然"获得

3. WEIGHTED vs BINARY JACCARD: 核心发现!
   Weighted Jaccard随深度下降: L0=0.61 → L33=0.50
   Binary Jaccard随深度上升:   L0=0.69 → L33=0.97
   
   含义: 
   - 深层激活几乎相同的neuron集合(b_jac→1.0)
   - 但这些neuron的激活幅度显著不同(w_jac下降)
   - 直接支持"软条件系统"假说: 不是离散开关, 是连续幅度调制

4. CONTRIBUTION EFFICIENCY跨层恒定 ≈ 8.5-10.7
   - 各层的Δh对logits的贡献效率相同
   - 深层logit贡献大是因为Δh幅度大, 不是效率高
   - Δh幅度: L0=1.5 → L33=166.4 (110倍增长, 指数级)

5. 否定 vs 时态 vs 语义的对比
   - Patching: 时态的recovery下降最快, 否定最慢
     → 时态变化(bites→bit)需要更多层来"调和"
     → 否定变化(always→never)在embedding层就充分编码
   - Logit Lens: 语义的logit_diff最小, 否定和时态相当
     → 语义变化对logits的影响最小

6. 深层patching产生NEGATIVE recovery (最反直觉的发现)
   - L35 (Qwen3): 多个pair的cosine_recovery < 0
   - L39 (GLM4): 多个pair的cosine_recovery < 0
   - L27 (DS7B): recovery常为-0.5到-0.95
   
   含义: 把modified的深层hidden state放进base上下文
   → 不只是"无效", 而是"反向"!
   → 深层hidden state编码了全句的约束关系
   → 单独替换差异位置的深层表示会破坏这种约束关系
   → 直接支持"约束传播"假说: 语言是约束系统, 不是向量系统
""")

if __name__ == "__main__":
    analyze_patching()
    analyze_logit_lens()
    analyze_weighted()
    key_findings()
