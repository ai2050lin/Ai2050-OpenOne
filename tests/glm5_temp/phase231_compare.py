"""Phase 231 三模型详细对比分析"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import numpy as np

models = {
    "Qwen3": "tests/glm5_temp/phase231_qwen3_results.json",
    "GLM4": "tests/glm5_temp/phase231_glm4_results.json",
    "DS7B": "tests/glm5_temp/phase231_deepseek7b_results.json",
}

all_data = {}
for m, path in models.items():
    with open(path, 'r', encoding='utf-8') as f:
        all_data[m] = json.load(f)

print("=" * 80)
print("Phase 231 三模型对比分析")
print("=" * 80)

# ===== ExpA: 算子拟合 =====
print("\n### ExpA: 线性算子拟合 - 方向模型 vs 算子模型 ###\n")

print("Layer-by-layer comparison (op_advantage = op_R² - dir_R²):")
print(f"{'Model':8s} {'Layer':6s} {'dir_R²':8s} {'op_R²':8s} {'advantage':10s} {'dir_cos':8s} {'op_cos':8s}")
print("-" * 60)

for m in models:
    expA = all_data[m].get("expA", {})
    if "error" in expA:
        print(f"{m:8s} ERROR: {expA['error']}")
        continue
    
    # 找最佳层 (按op_advantage)
    best_lk = None
    best_adv = -999
    for lk in sorted(expA.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0):
        lv = expA[lk]
        adv = lv.get("op_advantage", -999)
        if adv > best_adv:
            best_adv = adv
            best_lk = lk
    
    # 打印所有层
    for lk in sorted(expA.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0):
        lv = expA[lk]
        if "mean_dir_r2_test" not in lv:
            continue
        marker = " ★" if lk == best_lk else ""
        print(f"{m:8s} {lk:6s} {lv['mean_dir_r2_test']:8.4f} {lv['mean_op_r2_test']:8.4f} "
              f"{lv['op_advantage']:10.4f} {lv['mean_dir_cos_test']:8.4f} {lv['mean_op_cos_test']:8.4f}{marker}")

# Per-category comparison at best layer
print("\n### Per-category at best layer ###")
print(f"{'Model':8s} {'Category':12s} {'dir_R²':8s} {'op_R²':8s} {'advantage':10s}")
print("-" * 55)

for m in models:
    expA = all_data[m].get("expA", {})
    if "error" in expA:
        continue
    best_lk = max(expA.keys(), key=lambda k: expA[k].get("op_advantage", -999))
    best = expA[best_lk]
    if "per_category" in best:
        for cat, cd in best["per_category"].items():
            print(f"{m:8s} {cat:12s} {cd['dir_r2']:8.4f} {cd['op_r2']:8.4f} {cd['op_advantage']:10.4f}")
    print()

# PCA-space R² (more informative than full-space R²)
print("\n### PCA-space R² (more reliable with small sample) ###")
print(f"{'Model':8s} {'Layer':6s} {'dir_R²_pca':10s} {'op_R²_pca':10s} {'adv_pca':10s}")
print("-" * 50)

for m in models:
    expA = all_data[m].get("expA", {})
    if "error" in expA:
        continue
    best_lk = max(expA.keys(), key=lambda k: expA[k].get("op_advantage_pca", -999) if "op_advantage_pca" in expA[k] else expA[k].get("op_advantage", -999))
    for lk in sorted(expA.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0):
        lv = expA[lk]
        if "mean_dir_r2_pca_test" not in lv:
            continue
        if abs(lv.get("mean_dir_r2_pca_test", 0)) > 0.01 or abs(lv.get("mean_op_r2_pca_test", 0)) > 0.01:
            print(f"{m:8s} {lk:6s} {lv['mean_dir_r2_pca_test']:10.4f} {lv['mean_op_r2_pca_test']:10.4f} "
                  f"{lv.get('op_advantage_pca', 0):10.4f}")

# Per-adjective detail for DS7B best layer
print("\n### DS7B per-adjective at best layer ###")
ds7b_expA = all_data["DS7B"].get("expA", {})
if ds7b_expA and "error" not in ds7b_expA:
    best_lk = max(ds7b_expA.keys(), key=lambda k: ds7b_expA[k].get("op_advantage", -999))
    best = ds7b_expA[best_lk]
    print(f"Best layer: {best_lk}")
    per_adj = best.get("per_adj", {})
    print(f"{'Adj':12s} {'dir_R²':8s} {'op_R²':8s} {'adv':8s} {'dir_cos':8s} {'op_cos':8s} {'Δ_norm':8s}")
    print("-" * 65)
    for adj in sorted(per_adj.keys(), key=lambda a: per_adj[a].get("op_r2_test", -999), reverse=True):
        d = per_adj[adj]
        print(f"{adj:12s} {d.get('dir_r2_test',0):8.4f} {d.get('op_r2_test',0):8.4f} "
              f"{d.get('op_r2_test',0)-d.get('dir_r2_test',0):8.4f} "
              f"{d.get('dir_cos_test',0):8.4f} {d.get('op_cos_test',0):8.4f} "
              f"{d.get('delta_norm',0):8.4f}")

# ===== ExpB: 因果注入 =====
print("\n\n### ExpB: Operation Causal Injection ###\n")

for m in models:
    expB = all_data[m].get("expB", {})
    if "error" in expB:
        print(f"{m}: ERROR")
        continue
    print(f"\n{m}:")
    for op_name in ["translate", "explain", "negate"]:
        op_data = expB.get(op_name, {})
        if not op_data:
            continue
        for layer_key in sorted(op_data.keys()):
            layer_data = op_data[layer_key]
            for prompt_key in sorted(layer_data.keys()):
                prompt_data = layer_data[prompt_key]
                # 找最大beta的效果
                max_beta_key = max(prompt_data.keys(), key=lambda k: int(k.split("_")[1]) if k.startswith("beta_") else 0)
                max_data = prompt_data[max_beta_key]
                print(f"  {op_name:10s} {layer_key:5s} {prompt_key[:25]:25s} "
                      f"KL={max_data.get('kl_divergence',0):.4f} "
                      f"top1_change={max_data.get('top1_changed',False)} "
                      f"op_related={max_data.get('op_related_token',False)} "
                      f"logit_diff={max_data.get('logit_diff_norm',0):.2f}")

# ===== ExpC: 预测回路 =====
print("\n\n### ExpC: Prediction Circuits ###\n")

for m in models:
    expC = all_data[m].get("expC", {})
    if "error" in expC:
        print(f"{m}: ERROR")
        continue
    print(f"\n{m}:")
    for circuit_name, circuit_data in expC.items():
        na = circuit_data.get("negation_analysis")
        if na:
            print(f"  {circuit_name}: flip_ratio={na['flip_ratios_mean']:.4f}, "
                  f"KL={na['kl_divergence']:.4f}, "
                  f"prob_suppression={na.get('prob_suppression',0):.4f}")
        
        ta = circuit_data.get("temporal_analysis")
        if ta:
            print(f"  {circuit_name} temporal: past-pres_KL={ta['past_pres_kl']:.4f}, "
                  f"past-fut_KL={ta['past_fut_kl']:.4f}, pres-fut_KL={ta['pres_fut_kl']:.4f}")
        
        # Show top-3 tokens for each variant
        for var_name, var_data in circuit_data.items():
            if isinstance(var_data, dict) and "top10_tokens" in var_data:
                tokens = var_data["top10_tokens"][:3]
                probs = var_data.get("top10_probs", [0]*3)[:3]
                print(f"    {var_name:15s}: {[(t,f'{p:.3f}') for t,p in zip(tokens, probs)]}")

# ===== ExpD: 非交换性 =====
print("\n\n### ExpD: Operator Non-commutativity ###\n")

print(f"{'Model':8s} {'Layer':6s} {'noncomm':10s} {'cos_ctx':10s} {'ctx_ratio':10s}")
print("-" * 50)
for m in models:
    expD = all_data[m].get("expD", {})
    if "error" in expD:
        print(f"{m}: ERROR")
        continue
    for lk in sorted(expD.keys(), key=lambda x: int(x[1:]) if x[1:].isdigit() else 0):
        lv = expD[lk]
        if "mean_noncomm_dist" not in lv:
            continue
        print(f"{m:8s} {lk:6s} {lv['mean_noncomm_dist']:10.4f} {lv['mean_cos_context_dep']:10.4f} "
              f"{lv['context_dep_ratio']:10.4f}")

# ===== 关键判决 =====
print("\n\n" + "=" * 80)
print("核心判决")
print("=" * 80)

print("""
1. 算子模型 vs 方向模型:
   - Qwen3: 算子优势微小(0.07), R²都极负 → 两模型都不好
   - GLM4:  算子有正优势(0.18), R²也极负 → 算子稍好但仍差
   - DS7B:  ★★★ 算子R²=+0.56(正!), 优势=0.77 → 算子模型有效!

2. 为什么DS7B成功而Qwen3/GLM4失败?
   - DS7B的residual stream norm在深层极大(~100x增长)
   - 这使得Δ向量的信噪比更高
   - PCA+Ridge在DS7B上更有效

3. 否定翻转 (最可靠的发现):
   - Qwen3: flip_ratio=0.10, KL=3.15
   - GLM4:  flip_ratio=0.07, KL=2.10
   - DS7B:  flip_ratio=0.08, KL=3.08
   → 三模型一致: "not"将top-10概率压制到原来的7-10%! 这是最强的"预测修正器"

4. 非交换性:
   - 所有模型: cos_context_dep随层深下降(0.66→0.19-0.36)
   → 形容词效果确实依赖于上下文, 且深层更强
   - context_dep_ratio ≈ 0.8-1.2 → 上下文依赖和独立Δ同量级
""")
