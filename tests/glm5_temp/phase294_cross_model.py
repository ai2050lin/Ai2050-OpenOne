"""Phase 294 Cross-Model Comparison"""
import json, sys
from collections import defaultdict
import numpy as np

models = {}
for model_name in ["qwen3", "glm4", "deepseek7b"]:
    fpath = f"results/phase294_token_alignment/{model_name}_alignment.json"
    try:
        models[model_name] = json.load(open(fpath, "r", encoding="utf-8"))
    except:
        print(f"WARNING: {model_name} data not found")

print("=" * 80)
print("PHASE 294 CROSS-MODEL COMPARISON: TOKEN ALIGNMENT FIX")
print("=" * 80)

# 1. Aligned vs Misaligned Crossover (resid_post)
print("\n=== 1. ALIGNED vs MISALIGNED CROSSOVER (resid_post) ===")
for model_name, d in models.items():
    nl = d["n_layers"]
    all_layers = sorted(set(int(k.split(",")[0]) for k in d["exp_A_summary"]))
    
    # Find crossover for operand
    op_cross = None
    for layer in all_layers:
        k_al = f"{layer},resid_post,operand_aligned"
        k_mi = f"{layer},resid_post,operand_misaligned"
        if k_al in d["exp_A_summary"] and k_mi in d["exp_A_summary"]:
            al = d["exp_A_summary"][k_al]["mean_np"]
            mi = d["exp_A_summary"][k_mi]["mean_np"]
            if al > mi and layer > 0:
                prev_al = d["exp_A_summary"].get(f"{layer-1},resid_post,operand_aligned", {}).get("mean_np", 0)
                prev_mi = d["exp_A_summary"].get(f"{layer-1},resid_post,operand_misaligned", {}).get("mean_np", 0)
                if prev_al <= prev_mi:
                    op_cross = layer
                break
    
    # Find crossover for last
    la_cross = None
    for layer in all_layers:
        k_al = f"{layer},resid_post,last_aligned"
        k_mi = f"{layer},resid_post,last_misaligned"
        if k_al in d["exp_A_summary"] and k_mi in d["exp_A_summary"]:
            al = d["exp_A_summary"][k_al]["mean_np"]
            mi = d["exp_A_summary"][k_mi]["mean_np"]
            if al >= mi and layer > 0:
                prev_al = d["exp_A_summary"].get(f"{layer-1},resid_post,last_aligned", {}).get("mean_np", 0)
                prev_mi = d["exp_A_summary"].get(f"{layer-1},resid_post,last_misaligned", {}).get("mean_np", 0)
                if prev_al < prev_mi:
                    la_cross = layer
                break
    
    print(f"  {model_name:12s}: operand crossover L{op_cross}, last crossover L{la_cross}")

# 2. Last Layer Anomaly
print("\n=== 2. LAST LAYER ANOMALY (resid_post) ===")
for model_name, d in models.items():
    nl = d["n_layers"]
    last_layer = nl - 1
    prev_layer = nl - 2
    for role in ["operand_aligned", "last_aligned"]:
        k_last = f"{last_layer},resid_post,{role}"
        k_prev = f"{prev_layer},resid_post,{role}"
        if k_last in d["exp_A_summary"] and k_prev in d["exp_A_summary"]:
            v_last = d["exp_A_summary"][k_last]["mean_np"]
            v_prev = d["exp_A_summary"][k_prev]["mean_np"]
            drop_pct = (1 - v_last / v_prev) * 100 if v_prev > 0 else 0
            print(f"  {model_name:12s} L{last_layer} {role:18s}: {v_last:.4f} (L{prev_layer}={v_prev:.4f}, drop={drop_pct:.0f}%)")

# 3. Key Layer Comparison Table
print("\n=== 3. KEY LAYER COMPARISON (resid_post, operand_aligned) ===")
print(f"{'Layer%':>8}", end="")
for m in models: print(f" | {m:>10}_al {m:>10}_mi", end="")
print()
for pct in [0, 14, 28, 42, 57, 71, 85, 95]:
    print(f"  {pct:4d}%", end="")
    for model_name, d in models.items():
        nl = d["n_layers"]
        layer = min(int(pct / 100 * nl), nl - 1)
        k_al = f"{layer},resid_post,operand_aligned"
        k_mi = f"{layer},resid_post,operand_misaligned"
        al = d["exp_A_summary"].get(k_al, {}).get("mean_np", 0)
        mi = d["exp_A_summary"].get(k_mi, {}).get("mean_np", 0)
        print(f" | {al:10.4f} {mi:10.4f}", end="")
    print()

# 4. Last Token Aligned vs Misaligned
print("\n=== 4. LAST TOKEN COMPARISON (resid_post, last_aligned vs last_misaligned) ===")
print(f"{'Layer%':>8}", end="")
for m in models: print(f" | {m:>10}_al {m:>10}_mi", end="")
print()
for pct in [0, 14, 28, 42, 57, 71, 85, 95]:
    print(f"  {pct:4d}%", end="")
    for model_name, d in models.items():
        nl = d["n_layers"]
        layer = min(int(pct / 100 * nl), nl - 1)
        k_al = f"{layer},resid_post,last_aligned"
        k_mi = f"{layer},resid_post,last_misaligned"
        al = d["exp_A_summary"].get(k_al, {}).get("mean_np", 0)
        mi = d["exp_A_summary"].get(k_mi, {}).get("mean_np", 0)
        print(f" | {al:10.4f} {mi:10.4f}", end="")
    print()

# 5. Component Synergy
print("\n=== 5. COMPONENT SYNERGY (best layer per model) ===")
for model_name, d in models.items():
    best_syn_op = (0, 0, 0, 0, 0)
    best_syn_la = (0, 0, 0, 0, 0)
    for k, v in d["exp_C_summary"].items():
        li = int(k.split(",")[0])
        comp = k.split(",")[1]
        role = k.split(",")[2]
        if comp == "attn_mlp":
            a_k = f"{li},attn_only,{role}"
            m_k = f"{li},mlp_only,{role}"
            a_v = d["exp_C_summary"].get(a_k, {}).get("mean_np", 0)
            m_v = d["exp_C_summary"].get(m_k, {}).get("mean_np", 0)
            b_v = v["mean_np"]
            syn = b_v - a_v - m_v
            if role == "operand" and abs(syn) > abs(best_syn_op[4]):
                best_syn_op = (li, a_v, m_v, b_v, syn)
            if role == "last" and abs(syn) > abs(best_syn_la[4]):
                best_syn_la = (li, a_v, m_v, b_v, syn)
    print(f"  {model_name:12s}:")
    li, a, m, b, s = best_syn_op
    print(f"    operand: L{li} attn={a:.4f} mlp={m:.4f} both={b:.4f} syn={s:+.4f}")
    li, a, m, b, s = best_syn_la
    print(f"    last:    L{li} attn={a:.4f} mlp={m:.4f} both={b:.4f} syn={s:+.4f}")

# 6. A->B Operator Effect
print("\n=== 6. A->B OPERATOR EFFECT (best layer) ===")
for model_name, d in models.items():
    best = (0, "", 0)
    for k, v in d["exp_B_summary"].items():
        if "operator" in k and v["mean_np"] > best[2]:
            best = (int(k.split(",")[0]), k, v["mean_np"])
    print(f"  {model_name:12s}: L{best[0]} NP={best[2]:.4f}")

# 7. Summary: Does alignment fix the last-layer anomaly?
print("\n=== 7. SUMMARY: ALIGNMENT EFFECT ===")
for model_name, d in models.items():
    nl = d["n_layers"]
    last = nl - 1
    prev = nl - 2
    
    # Last layer drops for operand_aligned and last_aligned
    k_la = f"{last},resid_post,last_aligned"
    k_la_prev = f"{prev},resid_post,last_aligned"
    if k_la in d["exp_A_summary"] and k_la_prev in d["exp_A_summary"]:
        la_drop = d["exp_A_summary"][k_la_prev]["mean_np"] - d["exp_A_summary"][k_la]["mean_np"]
    else:
        la_drop = -1
    
    # Does aligned catch up at deep layers?
    k_la_deep = f"{prev},resid_post,last_aligned"
    k_mi_deep = f"{prev},resid_post,last_misaligned"
    if k_la_deep in d["exp_A_summary"] and k_mi_deep in d["exp_A_summary"]:
        ratio = d["exp_A_summary"][k_la_deep]["mean_np"] / d["exp_A_summary"][k_mi_deep]["mean_np"]
    else:
        ratio = 0
    
    print(f"  {model_name:12s}: last_layer_drop={la_drop:.4f}, deep_ratio(L{prev})={ratio:.3f}")
