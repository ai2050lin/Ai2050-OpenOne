"""Analyze Phase 294 JSON data structure"""
import json, sys
from collections import defaultdict

model = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
fpath = f"results/phase294_token_alignment/{model}_alignment.json"
d = json.load(open(fpath, "r", encoding="utf-8"))

print(f"=== {model.upper()} Phase 294 Data Structure ===")

# Exp A summary
keys_a = sorted(d["exp_A_summary"].keys())
layers_a = sorted(set(int(k.split(",")[0]) for k in keys_a))
print(f"\nExp A summary: {len(keys_a)} keys, layers={layers_a}")
for k in keys_a[:8]:
    v = d["exp_A_summary"][k]
    print(f"  {k}: mean_np={v['mean_np']:.5f} n={v['n']}")
deep_a = [k for k in keys_a if int(k.split(",")[0]) > 10]
print(f"Deep layer keys: {len(deep_a)}")
for k in deep_a[:8]:
    v = d["exp_A_summary"][k]
    print(f"  {k}: mean_np={v['mean_np']:.5f} n={v['n']}")

# Exp B summary
keys_b = sorted(d["exp_B_summary"].keys())
layers_b = sorted(set(int(k.split(",")[0]) for k in keys_b))
print(f"\nExp B summary: {len(keys_b)} keys, layers={layers_b}")

# Exp C summary  
keys_c = sorted(d["exp_C_summary"].keys())
layers_c = sorted(set(int(k.split(",")[0]) for k in keys_c))
print(f"\nExp C summary: {len(keys_c)} keys, layers={layers_c}")

# Full table: Exp A resid_post, operand/last, aligned vs misaligned
print(f"\n=== Exp A: Full Layer Table (resid_post) ===")
print(f"{'Layer':>5} | {'opnd_al':>8} | {'opnd_mi':>8} | {'last_al':>8} | {'last_mi':>8} | {'op_al/mi':>8} | {'la_al/mi':>8}")
for layer in layers_a:
    vals = {}
    for role in ["operand_aligned", "operand_misaligned", "last_aligned", "last_misaligned"]:
        k = f"{layer},resid_post,{role}"
        if k in d["exp_A_summary"]:
            vals[role] = d["exp_A_summary"][k]["mean_np"]
        else:
            vals[role] = 0
    oa = vals.get("operand_aligned", 0)
    om = vals.get("operand_misaligned", 0)
    la = vals.get("last_aligned", 0)
    lm = vals.get("last_misaligned", 0)
    ratio_o = oa / om if om > 0.001 else 0
    ratio_l = la / lm if lm > 0.001 else 0
    if oa > 0.001 or om > 0.001 or la > 0.001 or lm > 0.001:
        print(f"L{layer:4d} | {oa:8.4f} | {om:8.4f} | {la:8.4f} | {lm:8.4f} | {ratio_o:8.3f} | {ratio_l:8.3f}")

# Also show attn and mlp
print(f"\n=== Exp A: attn vs mlp (operand_aligned) ===")
for layer in layers_a:
    a_k = f"{layer},attn,operand_aligned"
    m_k = f"{layer},mlp,operand_aligned"
    a_np = d["exp_A_summary"][a_k]["mean_np"] if a_k in d["exp_A_summary"] else 0
    m_np = d["exp_A_summary"][m_k]["mean_np"] if m_k in d["exp_B_summary"] else 0
    if a_np > 0.001 or m_np > 0.001:
        print(f"  L{layer:4d}: attn={a_np:.4f} mlp={m_np:.4f}")
