"""Phase 134 汇总分析"""
import json
import numpy as np
from pathlib import Path

temp_dir = Path("d:/Ai2050/TransformerLens-Project/tests/glm5_temp")
models = ["qwen3", "glm4", "deepseek7b"]

print("=" * 70)
print("Phase 134: 条件激活边界分析 — 三模型汇总")
print("=" * 70)

for model_name in models:
    fname = f"phase134_{model_name}_activation_boundary.json"
    fpath = temp_dir / fname
    if not fpath.exists():
        print(f"\n{model_name}: file not found")
        continue

    with open(fpath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    mi = data["model_info"]
    print(f"\n{'='*60}")
    print(f"  {model_name}: {mi['class']}, layers={mi['n_layers']}, d={mi['d_model']}")
    print(f"{'='*60}")

    # === Exp 1: MLP激活稀疏性 ===
    print(f"\n--- Exp 1: MLP激活稀疏性 ---")
    r1 = data.get("exp1_activation_sparsity", {})
    summary1 = r1.get("summary", {})
    for lk in sorted(summary1.keys()):
        ld = summary1[lk]
        base_ratio = ld.get("base", {}).get("mean_ratio", 0)
        neg_ratio = ld.get("negation", {}).get("mean_ratio", 0)
        past_ratio = ld.get("past", {}).get("mean_ratio", 0)
        base_gini = ld.get("base", {}).get("mean_gini", 0)
        print(f"  {lk}: ratio(base={base_ratio:.4f}, neg={neg_ratio:.4f}, past={past_ratio:.4f}), "
              f"gini={base_gini:.4f}")

    # === Exp 2: Attention Head ===
    print(f"\n--- Exp 2: Attention Head熵 ---")
    r2 = data.get("exp2_attention_patterns", {})
    vd = r2.get("variant_diff", {})
    for lk in sorted(vd.keys()):
        ld = vd[lk]
        base_e = ld.get("base_mean_entropy", 0)
        neg_e = ld.get("neg_mean_entropy", 0)
        past_e = ld.get("past_mean_entropy", 0)
        delta_neg = neg_e - base_e
        delta_past = past_e - base_e
        print(f"  {lk}: entropy(base={base_e:.4f}, neg={neg_e:.4f}(+{delta_neg:.4f}), "
              f"past={past_e:.4f}(+{delta_past:.4f}))")

    # === Exp 3: 约束激活边界 ===
    print(f"\n--- Exp 3: 约束效应几何 ---")
    r3 = data.get("exp3_constraint_boundary", {})
    summary3 = r3.get("summary", {})
    sa = r3.get("subspace_analysis", {})

    for lk in sorted(summary3.keys()):
        ld = summary3[lk]
        print(f"  {lk}: cos(neg,past)={ld['mean_cos_neg_past']:.4f}, "
              f"rel_neg={ld['mean_rel_neg']:.4f}, rel_past={ld['mean_rel_past']:.4f}, "
              f"dim_overlap={ld['mean_dim_overlap']:.4f}")

    for lk in sorted(sa.keys()):
        ld = sa[lk]
        print(f"  {lk} subspace: neg_rank={ld['neg_rank']}, past_rank={ld['past_rank']}, "
              f"subspace_cos={ld['subspace_cosine']:.4f}")

    # === Exp 4: 约束投影稳定性 ===
    print(f"\n--- Exp 4: 约束投影质量 ---")
    r4 = data.get("exp4_constraint_projection", {})
    summary4 = r4.get("summary", {})

    for key in sorted(summary4.keys()):
        ld = summary4[key]
        print(f"  {key}: quality={ld['mean_quality']:.4f}±{ld['std_quality']:.4f}, "
              f"alpha_rank={ld['mean_alpha_rank']:.1f}")

# === 跨模型对比 ===
print(f"\n{'='*70}")
print("跨模型核心对比")
print(f"{'='*70}")

print("\n--- 约束效应相对大小 (Exp 3) ---")
print(f"{'Layer':<10} {'Model':<12} {'cos(n,p)':<12} {'rel_neg':<12} {'rel_past':<12} {'dim_overlap':<12}")
for model_name in models:
    fname = f"phase134_{model_name}_activation_boundary.json"
    fpath = temp_dir / fname
    if not fpath.exists():
        continue
    with open(fpath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    r3 = data.get("exp3_constraint_boundary", {})
    summary3 = r3.get("summary", {})
    for lk in sorted(summary3.keys()):
        ld = summary3[lk]
        print(f"{lk:<10} {model_name:<12} {ld['mean_cos_neg_past']:.4f}       "
              f"{ld['mean_rel_neg']:.4f}       {ld['mean_rel_past']:.4f}       "
              f"{ld['mean_dim_overlap']:.4f}")

print("\n--- 约束投影质量 (Exp 4, eps=0.01) ---")
print(f"{'Layer_Constraint':<20} {'Qwen3':<15} {'GLM4':<15} {'DS7B':<15}")
for constraint in ["neg", "past"]:
    for layer_prefix in ["L1", "L9", "L7", "L10"]:
        key_pattern = f"{layer_prefix}_{constraint}_eps_0.010"
        vals = {}
        for model_name in models:
            fname = f"phase134_{model_name}_activation_boundary.json"
            fpath = temp_dir / fname
            if not fpath.exists():
                continue
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            r4 = data.get("exp4_constraint_projection", {})
            summary4 = r4.get("summary", {})
            if key_pattern in summary4:
                vals[model_name] = summary4[key_pattern]["mean_quality"]

        if vals:
            q = vals.get("qwen3", -1)
            g = vals.get("glm4", -1)
            d = vals.get("deepseek7b", -1)
            print(f"{key_pattern:<20} {q:.4f}          {g:.4f}          {d:.4f}")

print("\n--- Attention: 否定句entropy增量 ---")
print(f"{'Layer':<10} {'Qwen3':<12} {'GLM4':<12} {'DS7B':<12}")
for model_name in models:
    fname = f"phase134_{model_name}_activation_boundary.json"
    fpath = temp_dir / fname
    if not fpath.exists():
        continue
    with open(fpath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    r2 = data.get("exp2_attention_patterns", {})
    vd = r2.get("variant_diff", {})
    for lk in sorted(vd.keys()):
        base_e = vd[lk].get("base_mean_entropy", 0)
        neg_e = vd[lk].get("neg_mean_entropy", 0)
        delta = neg_e - base_e
        if model_name == list(models)[0] and lk == list(sorted(vd.keys()))[0]:
            print(f"{'Layer':<10} {'Qwen3':<12} {'GLM4':<12} {'DS7B':<12}")

# 更精确的跨模型对比
print("\n--- Attention entropy增量 (neg - base), 所有层 ---")
entropy_deltas = {}
for model_name in models:
    fname = f"phase134_{model_name}_activation_boundary.json"
    fpath = temp_dir / fname
    if not fpath.exists():
        continue
    with open(fpath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    r2 = data.get("exp2_attention_patterns", {})
    vd = r2.get("variant_diff", {})
    for lk, ld in vd.items():
        if lk not in entropy_deltas:
            entropy_deltas[lk] = {}
        base_e = ld.get("base_mean_entropy", 0)
        neg_e = ld.get("neg_mean_entropy", 0)
        entropy_deltas[lk][model_name] = neg_e - base_e

for lk in sorted(entropy_deltas.keys()):
    vals = entropy_deltas[lk]
    q = vals.get("qwen3", 0)
    g = vals.get("glm4", 0)
    d = vals.get("deepseek7b", 0)
    print(f"  {lk}: Qwen3={q:+.4f}, GLM4={g:+.4f}, DS7B={d:+.4f}")
