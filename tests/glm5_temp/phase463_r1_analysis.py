"""Phase 463 R1 结果分析"""
import json, os
import numpy as np

def load_result(model, r):
    path = f"results/glm5/phase463_{model}_r{r}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)

models = ["qwen3", "deepseek7b", "glm4"]

print("=" * 80)
print("Phase 463 R1 核心结果分析")
print("=" * 80)

# ============ Exp1: 语义/语言正交分解 ============
print("\n### Exp1: 语义/语言正交分解patch ###")
for model in models:
    data = load_result(model, 1)
    exp1 = data.get("exp1_semantic_language_orthogonal", {})
    print(f"\n--- {model.upper()} ---")
    for layer_key in sorted(exp1.keys()):
        layer_data = exp1[layer_key]
        # 取第一个对象的结构信息
        first_key = list(layer_data.keys())[0]
        d = layer_data[first_key]
        cos_sem_lang = d.get("cos_sem_lang", "N/A")
        sem_only_ratio = d.get("sem_only_ratio", "N/A")
        lang_only_ratio = d.get("lang_only_ratio", "N/A")
        
        # 汇总所有对象的delta
        sem_only_en_deltas = [v["sem_only_en_delta"] for v in layer_data.values()]
        lang_only_en_deltas = [v["lang_only_en_delta"] for v in layer_data.values()]
        sem_only_zh_deltas = [v["sem_only_zh_delta"] for v in layer_data.values()]
        lang_only_zh_deltas = [v["lang_only_zh_delta"] for v in layer_data.values()]
        
        print(f"  {layer_key}: cos(sem,lang)={cos_sem_lang:.4f}, "
              f"sem_only_ratio={sem_only_ratio:.4f}, lang_only_ratio={lang_only_ratio:.4f}")
        print(f"    sem_only_enΔ: mean={np.mean(sem_only_en_deltas):.2f}, "
              f"lang_only_enΔ: mean={np.mean(lang_only_en_deltas):.2f}")
        print(f"    sem_only_zhΔ: mean={np.mean(sem_only_zh_deltas):.2f}, "
              f"lang_only_zhΔ: mean={np.mean(lang_only_zh_deltas):.2f}")

# ============ Exp2: 大样本跨语言Patch ============
print("\n### Exp2: 大样本跨语言Patch (按类别汇总) ###")
for model in models:
    data = load_result(model, 1)
    exp2 = data.get("exp2_large_sample_patch", {})
    summary = exp2.get("summary", {})
    print(f"\n--- {model.upper()} ---")
    for cat in sorted(summary.keys()):
        for li in sorted(summary[li] for li in summary[cat] if isinstance(li, int) or li.startswith("L")):
            pass
        # 简化输出
        for li_key in sorted(summary[cat].keys()):
            d = summary[cat][li_key]
            print(f"  {cat} {li_key}: enΔ={d['avg_en_delta']:.2f}, zhΔ={d['avg_zh_delta']:.2f}, n={d['n']}")

# ============ Exp3: Additive vs Mean-code vs Random ============
print("\n### Exp3: Additive vs Mean-code vs Random ###")
for model in models:
    data = load_result(model, 1)
    exp3 = data.get("exp3_patch_methods_comparison", {})
    print(f"\n--- {model.upper()} ---")
    add_deltas = [v["additive_en_delta"] for v in exp3.values()]
    mean_deltas = [v["mean_code_en_delta"] for v in exp3.values()]
    rand_deltas = [v["random_en_delta"] for v in exp3.values()]
    print(f"  Additive: mean={np.mean(add_deltas):.2f}, std={np.std(add_deltas):.2f}")
    print(f"  Mean-code: mean={np.mean(mean_deltas):.2f}, std={np.std(mean_deltas):.2f}")
    print(f"  Random:    mean={np.mean(rand_deltas):.2f}, std={np.std(rand_deltas):.2f}")
    print(f"  Additive vs Random selectivity: {np.mean(add_deltas) - np.mean(rand_deltas):.2f}")
    print(f"  Mean-code vs Random selectivity: {np.mean(mean_deltas) - np.mean(rand_deltas):.2f}")

# ============ Exp4: Holdout可写性 ============
print("\n### Exp4: Holdout可写性验证 ###")
for model in models:
    data = load_result(model, 1)
    exp4 = data.get("exp4_holdout_writability", {})
    avg_sel = exp4.get("avg_holdout_selectivity", 0)
    per_obj = exp4.get("per_object", {})
    # 按beta分组
    for beta in [5.0, 10.0]:
        sels = [v["selectivity"] for v in per_obj.values() if v.get("beta") == beta]
        if sels:
            print(f"  {model.upper()} beta={beta}: avg_sel={np.mean(sels):.2f}, std={np.std(sels):.2f}")

# ============ Exp5: 翻译方向精细分解 ============
print("\n### Exp5: 翻译方向精细分解 ###")
for model in models:
    data = load_result(model, 1)
    exp5 = data.get("exp5_translate_fine_decomposition", {})
    print(f"\n--- {model.upper()} ---")
    for layer_key in sorted(exp5.keys()):
        d = exp5[layer_key]
        cos_ts = d.get("cos_target_vs_source", "N/A")
        cos_tc = d.get("cos_target_vs_content", "N/A")
        cos_cc = d.get("cos_cmd_vs_content", "N/A")
        eff_rank = d.get("effective_rank", "N/A")
        
        # 格式化
        cos_ts_str = f"{cos_ts:.3f}" if isinstance(cos_ts, float) else str(cos_ts)
        cos_tc_str = f"{cos_tc:.3f}" if isinstance(cos_tc, float) else str(cos_tc)
        cos_cc_str = f"{cos_cc:.3f}" if isinstance(cos_cc, float) else str(cos_cc)
        rank_str = f"{eff_rank:.2f}" if isinstance(eff_rank, float) else str(eff_rank)
        
        print(f"  {layer_key}: cos(tgt,src)={cos_ts_str}, "
              f"cos(tgt,content)={cos_tc_str}, "
              f"cos(cmd,content)={cos_cc_str}, "
              f"eff_rank={rank_str}")

print("\n" + "=" * 80)
print("关键发现汇总:")
print("=" * 80)

# 检查DS7B的一维性
ds7b_exp5 = load_result("deepseek7b", 1).get("exp5_translate_fine_decomposition", {})
for layer_key in sorted(ds7b_exp5.keys()):
    d = ds7b_exp5[layer_key]
    cos_ts = abs(d.get("cos_target_vs_source", 0))
    cos_tc = abs(d.get("cos_target_vs_content", 0))
    cos_cc = abs(d.get("cos_cmd_vs_content", 0))
    if cos_ts > 0.99:
        print(f"  DS7B {layer_key}: 一维语言轴确认! |cos(tgt,src)|={cos_ts:.4f}, |cos(tgt,content)|={cos_tc:.4f}, |cos(cmd,content)|={cos_cc:.4f}")

# 检查GLM4的holdout
glm4_exp4 = load_result("glm4", 1).get("exp4_holdout_writability", {})
print(f"  GLM4 holdout selectivity: {glm4_exp4.get('avg_holdout_selectivity', 0):.2f}")

# 检查Qwen3的Exp1结果
qwen3_exp1 = load_result("qwen3", 1).get("exp1_semantic_language_orthogonal", {})
for layer_key in sorted(qwen3_exp1.keys()):
    layer_data = qwen3_exp1[layer_key]
    first_key = list(layer_data.keys())[0]
    d = layer_data[first_key]
    sem_only_en = [v["sem_only_en_delta"] for v in layer_data.values()]
    lang_only_en = [v["lang_only_en_delta"] for v in layer_data.values()]
    print(f"  Qwen3 {layer_key}: sem_only_enΔ={np.mean(sem_only_en):.2f}, lang_only_enΔ={np.mean(lang_only_en):.2f}")
