"""Phase 461 R1 结果摘要分析"""
import json, os, sys
import numpy as np

results_dir = "results/glm5"
models = ["qwen3", "glm4", "deepseek7b"]

all_results = {}
for m in models:
    path = os.path.join(results_dir, f"phase461_{m}_r1.json")
    if os.path.exists(path):
        with open(path) as f:
            all_results[m] = json.load(f)

print("="*70)
print("Phase 461 R1 结果摘要")
print("="*70)

# ========== Exp1: W_down行级贡献分解 ==========
print("\n### Exp1: W_down行级贡献分解 ###")
for m in models:
    if m not in all_results or "exp1_wdown_row" not in all_results[m]:
        continue
    if "error" in all_results[m]["exp1_wdown_row"]:
        print(f"  {m}: ERROR - {all_results[m]['exp1_wdown_row']['error']}")
        continue
    print(f"\n  {m}:")
    exp1 = all_results[m]["exp1_wdown_row"]
    for cat in ["fruit", "animal", "tool", "vehicle"]:
        if cat not in exp1:
            continue
        for layer_key in sorted(exp1[cat].keys(), key=lambda x: int(x[1:])):
            d = exp1[cat][layer_key]
            if "error" in d:
                print(f"    {cat} {layer_key}: {d['error']}")
                continue
            shared_ratio = d.get("shared_private_ratio", 0)
            overlap = d.get("overlap_top_k", 0)
            corr = d.get("shared_private_corr", 0)
            print(f"    {cat} {layer_key}: shared/private={shared_ratio:.1f}, overlap_top20={overlap}/20, corr={corr:.3f}")

# ========== Exp2: 跨对象差分结构 ==========
print("\n### Exp2: 跨对象差分结构对比 ###")
for m in models:
    if m not in all_results or "exp2_cross_object_diff" not in all_results[m]:
        continue
    if "error" in all_results[m]["exp2_cross_object_diff"]:
        print(f"  {m}: ERROR")
        continue
    print(f"\n  {m}:")
    exp2 = all_results[m]["exp2_cross_object_diff"]
    for cat in ["fruit", "animal", "tool", "vehicle"]:
        if cat not in exp2:
            continue
        for layer_key in sorted(exp2[cat].keys(), key=lambda x: int(x[1:])):
            d = exp2[cat][layer_key]
            eff_rank = d.get("effective_rank_90pct", 0)
            avg_cos = d.get("avg_private_cosine_offdiag", 0)
            sv = d.get("singular_values", [])
            ve = d.get("variance_explained", [])
            ve_str = ", ".join([f"{v:.2f}" for v in ve[:4]]) if ve else "N/A"
            print(f"    {cat} {layer_key}: eff_rank={eff_rank}, avg_private_cos={avg_cos:.3f}, var_expl=[{ve_str}]")
    
    # 跨类别
    if "_cross_category" in exp2:
        print(f"  跨类别对比:")
        cross = exp2["_cross_category"]
        for layer_key in sorted(cross.keys(), key=lambda x: int(x[1:])):
            d = cross[layer_key]
            shared_cos = d.get("shared_cosine", {})
            priv_cos = d.get("private_cross_cosine", {})
            avg_shared = np.mean(list(shared_cos.values())) if shared_cos else 0
            avg_priv = np.mean(list(priv_cos.values())) if priv_cos else 0
            print(f"    {layer_key}: avg_shared_center_cos={avg_shared:.3f}, avg_cross_priv_cos={avg_priv:.3f}")

# ========== Exp3: 翻译命令编码 ==========
print("\n### Exp3: 翻译命令编码 ###")
for m in models:
    if m not in all_results or "exp3_translate" not in all_results[m]:
        continue
    if "error" in all_results[m]["exp3_translate"]:
        print(f"  {m}: ERROR")
        continue
    print(f"\n  {m}:")
    exp3 = all_results[m]["exp3_translate"]
    for word in ["apple", "dog", "hammer"]:
        if word not in exp3:
            continue
        word_data = exp3[word]
        for layer_key in sorted(word_data.keys(), key=lambda x: int(x[1:])):
            d = word_data[layer_key]
            en2zh_norm = d.get("en2zh_diff_norm", 0)
            zh2en_norm = d.get("zh2en_diff_norm", 0)
            cross_cos = d.get("en2zh_vs_zh2en_diff_cos", 0)
            if en2zh_norm is not None:
                print(f"    {word} {layer_key}: en2zh_diff={en2zh_norm:.1f}, zh2en_diff={zh2en_norm:.1f}, "
                      f"cross_cos={cross_cos:.3f}")

# ========== Exp4: 跨语言中间层探针 ==========
print("\n### Exp4: 跨语言中间层探针 ###")
for m in models:
    if m not in all_results or "exp4_cross_lang_probe" not in all_results[m]:
        continue
    if "error" in all_results[m]["exp4_cross_lang_probe"]:
        print(f"  {m}: ERROR")
        continue
    print(f"\n  {m}:")
    exp4 = all_results[m]["exp4_cross_lang_probe"]
    for layer_key in sorted(exp4.keys(), key=lambda x: int(x[1:])):
        d = exp4[layer_key]
        en_acc = d.get("en_probe_acc", 0)
        zh_acc = d.get("zh_probe_acc_cross_lang", 0)
        avg_cos = d.get("avg_cosine_en_zh", 0)
        print(f"    {layer_key}: en_acc={en_acc:.2f}, zh_cross_acc={zh_acc:.2f} (random=0.25), "
              f"avg_cos={avg_cos:.3f}")

# ========== Exp5: 大beta合成测试 ==========
print("\n### Exp5: 大beta合成测试 ###")
for m in models:
    if m not in all_results or "exp5_large_beta" not in all_results[m]:
        continue
    if "error" in all_results[m]["exp5_large_beta"]:
        print(f"  {m}: ERROR")
        continue
    print(f"\n  {m}:")
    exp5 = all_results[m]["exp5_large_beta"]
    for case in exp5:
        case_data = exp5[case]
        for layer_key in sorted(case_data.keys(), key=lambda x: int(x[1:])):
            d = case_data[layer_key]
            base_margin = d.get("base_margin", 0)
            print(f"    {case} {layer_key}: base_margin={base_margin:.2f}", end="")
            for beta in [5, 10, 20, 50]:
                sel_key = f"beta{beta}_selectivity"
                delta_t = f"beta{beta}_delta_target"
                delta_c = f"beta{beta}_delta_comp"
                if sel_key in d:
                    print(f", β{beta}: sel={d[sel_key]:.2f}(Δt={d.get(delta_t,0):.2f},Δc={d.get(delta_c,0):.2f})", end="")
            print()
