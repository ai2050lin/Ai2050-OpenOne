"""Phase 135 汇总分析"""
import sys, os, json, numpy as np
sys.stdout.reconfigure(encoding='utf-8')

base_dir = os.path.join(os.path.dirname(__file__), '..')
temp_dir = os.path.join(base_dir, 'glm5_temp')

models = ["qwen3", "glm4", "deepseek7b"]
model_labels = {"qwen3": "Qwen3", "glm4": "GLM4", "deepseek7b": "DS7B"}

all_data = {}
for m in models:
    path = os.path.join(temp_dir, f"phase135_{m}_activation_overlap.json")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            all_data[m] = json.load(f)

print("=" * 70)
print("Phase 135: 激活重叠分析 — 跨模型汇总")
print("=" * 70)

# === Exp 1: MLP激活重叠 ===
print("\n## Exp 1: MLP激活重叠 (threshold=0.0)")
print("-" * 70)

print(f"{'Layer':<10} | {'Qwen3 J(b,n)':>14} {'J(b,p)':>10} | {'GLM4 J(b,n)':>14} {'J(b,p)':>10} | {'DS7B J(b,n)':>14} {'J(b,p)':>10} | {'Random':>8}")
print("-" * 110)

for m in models:
    if m not in all_data:
        continue
    r1 = all_data[m].get("exp1_mlp_overlap", {})
    summary = r1.get("summary", {})
    random_jac = r1.get("random_baseline", {}).get("expected_random_jaccard", 0)

    layers_data = {}
    for key, ld in summary.items():
        if "_t0.0" in key:
            layer = key.replace("_t0.0", "")
            layers_data[layer] = ld

    # Sort by layer number
    def layer_num(l):
        try:
            return int(l[1:])
        except:
            return 0

    sorted_layers = sorted(layers_data.keys(), key=layer_num)

    if m == models[0]:
        for lk in sorted_layers:
            ld = layers_data[lk]
            # Collect data from all models for this layer
            vals = []
            for m2 in models:
                if m2 not in all_data:
                    vals.append(("N/A", "N/A"))
                    continue
                r12 = all_data[m2].get("exp1_mlp_overlap", {}).get("summary", {})
                key2 = f"{lk}_t0.0"
                if key2 in r12:
                    vals.append((f"{r12[key2]['mean_jaccard_base_neg']:.4f}",
                                 f"{r12[key2]['mean_jaccard_base_past']:.4f}"))
                else:
                    vals.append(("N/A", "N/A"))

            rand_vals = []
            for m2 in models:
                rj = all_data.get(m2, {}).get("exp1_mlp_overlap", {}).get("random_baseline", {}).get("expected_random_jaccard", 0)
                rand_vals.append(f"{rj:.4f}")

            print(f"{lk:<10} | {vals[0][0]:>14} {vals[0][1]:>10} | {vals[1][0]:>14} {vals[1][1]:>10} | {vals[2][0]:>14} {vals[2][1]:>10} | {rand_vals[0]:>8}")
        break

# === Exp 4: 语义vs语法 ===
print("\n## Exp 4: 语义vs语法对比 (MLP Jaccard)")
print("-" * 70)
print(f"{'Layer':<10} | {'Qwen3 语义J':>12} {'语法J':>10} {'比':>6} | {'GLM4 语义J':>12} {'语法J':>10} {'比':>6} | {'DS7B 语义J':>12} {'语法J':>10} {'比':>6}")
print("-" * 110)

for m in models:
    if m not in all_data:
        continue
    r4 = all_data[m].get("exp4_semantic_vs_syntax", {})
    sem_summary = r4.get("summary", {})
    r1 = all_data[m].get("exp1_mlp_overlap", {}).get("summary", {})

    for lk, ld in sorted(sem_summary.items()):
        sem_j = ld["mean_semantic_jaccard"]
        # Find corresponding syntax Jaccard
        syn_key = f"{lk}_t0.0"
        syn_j = r1.get(syn_key, {}).get("mean_jaccard_base_neg", 0)
        ratio = syn_j / max(sem_j, 0.001)

        print(f"{lk:<10} | {sem_j:>12.4f} {syn_j:>10.4f} {ratio:>6.2f}x", end="")
        if m != models[-1]:
            print(" | ", end="")
    print()

# === 关键统计 ===
print("\n## 关键统计")
print("-" * 70)

# 语法Jaccard范围
syn_jacs = []
sem_jacs = []
for m in models:
    if m not in all_data:
        continue
    r1 = all_data[m].get("exp1_mlp_overlap", {}).get("summary", {})
    r4 = all_data[m].get("exp4_semantic_vs_syntax", {}).get("summary", {})

    for key, ld in r1.items():
        if "_t0.0" in key and "L0" not in key:  # 排除L0
            syn_jacs.append(ld["mean_jaccard_base_neg"])

    for lk, ld in r4.items():
        sem_jacs.append(ld["mean_semantic_jaccard"])

if syn_jacs and sem_jacs:
    print(f"语法变化 MLP Jaccard: {np.mean(syn_jacs):.4f} ± {np.std(syn_jacs):.4f} (range: {np.min(syn_jacs):.4f} - {np.max(syn_jacs):.4f})")
    print(f"语义变化 MLP Jaccard: {np.mean(sem_jacs):.4f} ± {np.std(sem_jacs):.4f} (range: {np.min(sem_jacs):.4f} - {np.max(sem_jacs):.4f})")
    print(f"语法/语义 Jaccard比: {np.mean(syn_jacs)/max(np.mean(sem_jacs), 0.001):.2f}x")

# Exp 2: Head路由
print("\n## Exp 2: Head路由重叠 (cosine similarity)")
print("-" * 70)
for m in models:
    if m not in all_data:
        continue
    r2 = all_data[m].get("exp2_head_routing", {}).get("summary", {})
    print(f"\n{model_labels[m]}:")
    for lk, ld in sorted(r2.items()):
        print(f"  {lk}: cos(neg)={ld['mean_head_cos_neg']:.4f}, cos(past)={ld['mean_head_cos_past']:.4f}, "
              f"min_cos(neg)={ld['mean_min_cos_neg']:.4f}, neg_pos_change={ld['mean_neg_pos_change_rate']:.4f}")

# 关键对比
print("\n## 核心对比: 语法vs语义vs随机 MLP Jaccard")
print("-" * 70)
for m in models:
    if m not in all_data:
        continue
    r1 = all_data[m].get("exp1_mlp_overlap", {})
    random_jac = r1.get("random_baseline", {}).get("expected_random_jaccard", 0)

    # 中间层的语法Jaccard
    r1s = r1.get("summary", {})
    mid_jacs_neg = []
    for key, ld in r1s.items():
        if "_t0.0" in key and "L0" not in key:
            mid_jacs_neg.append(ld["mean_jaccard_base_neg"])

    r4 = all_data[m].get("exp4_semantic_vs_syntax", {}).get("summary", {})
    sem_jacs_m = [ld["mean_semantic_jaccard"] for ld in r4.values()]

    if mid_jacs_neg and sem_jacs_m:
        print(f"  {model_labels[m]}: 语法={np.mean(mid_jacs_neg):.4f}, "
              f"语义={np.mean(sem_jacs_m):.4f}, 随机={random_jac:.4f}, "
              f"语法/随机={np.mean(mid_jacs_neg)/max(random_jac, 0.001):.1f}x, "
              f"语义/随机={np.mean(sem_jacs_m)/max(random_jac, 0.001):.1f}x")
