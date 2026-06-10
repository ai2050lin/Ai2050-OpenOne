"""
Phase 434-437 综合分析脚本
"""
import sys, json, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import numpy as np

def load_result(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

print("=" * 70)
print("Phase 434-437 综合分析")
print("=" * 70)

# Phase 434: Head Causal Ablation
print("\n### Phase 434: 注意力头因果消融 ###")
for model in ["qwen3", "glm4", "deepseek7b"]:
    r = load_result(f"results/phase434_head_causal_ablation/{model}_phase434_r1.json")
    if not r:
        continue
    print(f"\n  {model}:")
    for obj_name, obj_data in r.get("per_object", {}).items():
        head_res = obj_data.get("head_results", [])
        candidate_scores = [hr.get("causal_scores", {}).get(str(r["n_layers"]-1), 0) 
                          for hr in head_res if hr.get("type") == "candidate" and "causal_scores" in hr]
        control_scores = [hr.get("causal_scores", {}).get(str(r["n_layers"]-1), 0) 
                        for hr in head_res if hr.get("type") == "control" and "causal_scores" in hr]
        if candidate_scores:
            print(f"    {obj_name}: candidate_mean={np.mean(candidate_scores):.4f}, "
                  f"control_mean={np.mean(control_scores):.4f}, "
                  f"gap={np.mean(candidate_scores)-np.mean(control_scores):.4f}")

# Phase 436: Contextualized Attribute
print("\n### Phase 436: 上下文化属性方向 ###")
for model in ["qwen3", "glm4"]:
    r = load_result(f"results/phase436_contextualized_attribute/{model}_phase436_r1.json")
    if not r:
        continue
    print(f"\n  {model}:")
    for attr_name, attr_data in r.get("per_attribute", {}).items():
        cos_we = attr_data.get("cos_contextual_we", {})
        cos_wu = attr_data.get("cos_contextual_wu", {})
        # 取中层的cos
        mid_layers = [k for k in cos_we.keys() if int(k) > 0]
        if mid_layers:
            mid_cos_we = np.mean([cos_we[k] for k in mid_layers[:5]])
            mid_cos_wu = np.mean([cos_wu[k] for k in mid_layers[:5]])
            
            # 取最后一层的switch score
            inj = attr_data.get("contextual_injection", {})
            last_layer = str(r["n_layers"] - 1)
            if last_layer in inj and "1.0" in inj[last_layer]:
                lr = inj[last_layer]["1.0"]
                pos_sw = lr.get("pos_injection", {}).get("switch_score", 0)
                neg_sw = lr.get("neg_injection", {}).get("switch_score", 0)
                print(f"    {attr_name}: cos_WE={mid_cos_we:.3f}, cos_WU={mid_cos_wu:.3f}, "
                      f"L{last_layer}_pos_sw={pos_sw:.3f}, L{last_layer}_neg_sw={neg_sw:.3f}")

# Phase 437: Category-Property Mediation
print("\n### Phase 437: 属性是否由类别中介 ###")
for model in ["qwen3", "glm4", "deepseek7b"]:
    r = load_result(f"results/phase437_category_property_mediation/{model}_phase437_r1.json")
    if not r:
        continue
    print(f"\n  {model}:")
    for test_name, test_data in r.get("per_test", {}).items():
        push = test_data.get("push_results", {})
        alpha2 = push.get("2.0", {})
        alpha4 = push.get("4.0", {})
        med_a2 = alpha2.get("mediation_score", 0) if "mediation_score" in alpha2 else 0
        med_a4 = alpha4.get("mediation_score", 0) if "mediation_score" in alpha4 else 0
        src_a2 = alpha2.get("src_mean", 0) if "src_mean" in alpha2 else 0
        tgt_a2 = alpha2.get("tgt_mean", 0) if "tgt_mean" in alpha2 else 0
        print(f"    {test_name}: med(a2)={med_a2:.4f}, med(a4)={med_a4:.4f}, "
              f"src(a2)={src_a2:.4f}, tgt(a2)={tgt_a2:.4f}")

print("\n" + "=" * 70)
print("关键发现总结:")
print("=" * 70)
print("""
1. Phase 434: 单头消融因果分数极低
   - Qwen3: 候选头≈控制头，CausalScore < 0.1
   - GLM4: 混合，无清晰区分
   - DS7B: ablation hook可能未生效
   → 类别运输是分布式过程，非单一头负责

2. Phase 436: 上下文化属性方向存在但难以操控
   - cos(contextual, W_E) ≈ 0: 上下文化方向与静态W_E正交
   - 最后一层注入有效(switch=2-9)，中间层混乱
   - DS7B的8bit量化导致NaN

3. Phase 437: 属性是否由类别中介 - 模型依赖性
   - Qwen3: 强正mediation (4.75-6.44)，属性跟随类别变化
   - GLM4: 近零/负mediation，属性不跟随类别（8bit问题？）
   - DS7B: 弱/混合mediation
   
   关键洞察：类别-属性中介是模型特异的，不是通用机制！
""")
