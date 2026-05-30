"""Phase 294 Complete Analysis - All components, all layers"""
import json, sys
from collections import defaultdict

model = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
fpath = f"results/phase294_token_alignment/{model}_alignment.json"
d = json.load(open(fpath, "r", encoding="utf-8"))

print(f"=== {model.upper()} Phase 294 Complete Analysis ===")

# Exp A: Full table for all components
for comp in ["resid_post", "attn", "mlp"]:
    print(f"\n=== Exp A: B->A {comp} ===")
    print(f"{'Layer':>5} | {'opnd_al':>7} | {'opnd_mi':>7} | {'last_al':>7} | {'last_mi':>7} | {'op_a/m':>6} | {'la_a/m':>6}")
    all_layers = sorted(set(int(k.split(",")[0]) for k in d["exp_A_summary"]))
    for layer in all_layers:
        vals = {}
        for role in ["operand_aligned", "operand_misaligned", "last_aligned", "last_misaligned"]:
            k = f"{layer},{comp},{role}"
            vals[role] = d["exp_A_summary"][k]["mean_np"] if k in d["exp_A_summary"] else 0
        oa = vals.get("operand_aligned", 0)
        om = vals.get("operand_misaligned", 0)
        la = vals.get("last_aligned", 0)
        lm = vals.get("last_misaligned", 0)
        ratio_o = oa / om if om > 0.001 else 0
        ratio_l = la / lm if lm > 0.001 else 0
        if oa > 0.001 or om > 0.001 or la > 0.001 or lm > 0.001:
            print(f"L{layer:4d} | {oa:7.4f} | {om:7.4f} | {la:7.4f} | {lm:7.4f} | {ratio_o:6.3f} | {ratio_l:6.3f}")

# Exp B: A->B (remove negation) - operator position
print(f"\n=== Exp B: A->B operator position ===")
for comp in ["resid_post", "attn", "mlp"]:
    all_layers = sorted(set(int(k.split(",")[0]) for k in d["exp_B_summary"]))
    vals_list = []
    for layer in all_layers:
        k = f"{layer},{comp},operator"
        if k in d["exp_B_summary"]:
            v = d["exp_B_summary"][k]["mean_np"]
            if abs(v) > 0.001:
                vals_list.append((layer, v))
    if vals_list:
        print(f"  {comp}: L0={vals_list[0][1]:.4f}", end="")
        if len(vals_list) > 1:
            best = max(vals_list, key=lambda x: x[1])
            print(f" best=L{best[0]}({best[1]:.4f})", end="")
        print()

# Exp C: Synergy
print(f"\n=== Exp C: Component Synergy ===")
for role in ["operand", "last"]:
    print(f"  {role}:")
    all_layers = sorted(set(int(k.split(",")[0]) for k in d["exp_C_summary"]))
    for layer in all_layers:
        a_k = f"{layer},attn_only,{role}"
        m_k = f"{layer},mlp_only,{role}"
        b_k = f"{layer},attn_mlp,{role}"
        a_np = d["exp_C_summary"][a_k]["mean_np"] if a_k in d["exp_C_summary"] else 0
        m_np = d["exp_C_summary"][m_k]["mean_np"] if m_k in d["exp_C_summary"] else 0
        b_np = d["exp_C_summary"][b_k]["mean_np"] if b_k in d["exp_C_summary"] else 0
        syn = b_np - a_np - m_np
        if abs(a_np) > 0.005 or abs(m_np) > 0.005 or abs(b_np) > 0.005:
            print(f"    L{layer:2d}: attn={a_np:.4f} mlp={m_np:.4f} both={b_np:.4f} syn={syn:+.4f}")

# Subtype from raw data (limited but useful)
print(f"\n=== Subtype Analysis (from raw data, limited) ===")
subtype_data = defaultdict(lambda: defaultdict(list))
for r in d["exp_A_raw"]:
    if r["component"] == "resid_post":
        subtype_data[(r["layer"], r["role"])][r["subtype"]].append(r["natural_prog"])

for key in sorted(subtype_data.keys()):
    layer, role = key
    print(f"  L{layer} {role}:")
    for st in sorted(subtype_data[key]):
        vals = subtype_data[key][st]
        avg = sum(vals) / len(vals)
        print(f"    {st}: NP={avg:.4f} (n={len(vals)})")

# Key comparisons
print(f"\n=== KEY COMPARISONS ===")
# 1. Last layer anomaly
print("1. Last Layer Anomaly:")
for layer_key in [max(set(int(k.split(",")[0]) for k in d["exp_A_summary"]))]:
    for comp in ["resid_post"]:
        for role in ["operand_aligned", "last_aligned"]:
            k = f"{layer_key},{comp},{role}"
            k_prev = f"{layer_key-1},{comp},{role}"
            v = d["exp_A_summary"][k]["mean_np"] if k in d["exp_A_summary"] else 0
            v_prev = d["exp_A_summary"][k_prev]["mean_np"] if k_prev in d["exp_A_summary"] else 0
            drop = v_prev - v
            print(f"  L{layer_key} {comp} {role}: NP={v:.4f} (L{layer_key-1}={v_prev:.4f}, drop={drop:.4f})")

# 2. Aligned vs misaligned crossover
print("\n2. Aligned > Misaligned Crossover (resid_post):")
for pos in ["operand", "last"]:
    all_layers = sorted(set(int(k.split(",")[0]) for k in d["exp_A_summary"]))
    for layer in all_layers:
        k_al = f"{layer},resid_post,{pos}_aligned"
        k_mi = f"{layer},resid_post,{pos}_misaligned"
        if k_al in d["exp_A_summary"] and k_mi in d["exp_A_summary"]:
            al = d["exp_A_summary"][k_al]["mean_np"]
            mi = d["exp_A_summary"][k_mi]["mean_np"]
            if al > mi and layer > 0:
                # Check previous layer
                k_al_prev = f"{layer-1},resid_post,{pos}_aligned"
                k_mi_prev = f"{layer-1},resid_post,{pos}_misaligned"
                if k_al_prev in d["exp_A_summary"] and k_mi_prev in d["exp_A_summary"]:
                    al_prev = d["exp_A_summary"][k_al_prev]["mean_np"]
                    mi_prev = d["exp_A_summary"][k_mi_prev]["mean_np"]
                    if al_prev <= mi_prev:
                        print(f"  {pos}: crossover at L{layer} (L{layer-1}: al={al_prev:.4f} mi={mi_prev:.4f} | L{layer}: al={al:.4f} mi={mi:.4f})")
                break
