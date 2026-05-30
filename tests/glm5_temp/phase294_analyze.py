"""Phase 294 Deep Analysis"""
import json, sys
from collections import defaultdict

model = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
fpath = f"results/phase294_token_alignment/{model}_alignment.json"
d = json.load(open(fpath, "r", encoding="utf-8"))

print(f"=== {model.upper()} Phase 294 Deep Analysis ===")
print(f"Exp A: {d['exp_A_count']} | Exp B: {d['exp_B_count']} | Exp C: {d['exp_C_count']}")

# Parse role into (position, alignment)
def parse_role(role):
    if "_aligned" in role:
        return role.replace("_aligned", ""), "aligned"
    elif "_misaligned" in role:
        return role.replace("_misaligned", ""), "misaligned"
    else:
        return role, "none"

# Exp A: B->A aligned vs misaligned
print("\n=== Exp A: B->A resid_post by layer x position x alignment ===")
layer_role_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for r in d["exp_A_raw"]:
    if r["component"] != "resid_post":
        continue
    pos, align = parse_role(r["role"])
    layer_role_data[r["layer"]][pos][align].append(r["natural_prog"])

for layer in sorted(layer_role_data.keys()):
    parts = []
    for pos in ["operator", "operand", "last"]:
        if pos not in layer_role_data[layer]:
            continue
        a = layer_role_data[layer][pos]
        avg_al = sum(a["aligned"]) / len(a["aligned"]) if a["aligned"] else 0
        avg_mi = sum(a["misaligned"]) / len(a["misaligned"]) if a["misaligned"] else 0
        avg_no = sum(a["none"]) / len(a["none"]) if a["none"] else 0
        if avg_al > 0.005 or avg_mi > 0.005 or avg_no > 0.005:
            parts.append(f"{pos}:al={avg_al:.3f} mi={avg_mi:.3f} no={avg_no:.3f}")
    if parts:
        print(f"  L{layer:2d}: {' | '.join(parts)}")

# Exp B: A->B
print("\n=== Exp B: A->B best layer per component (operator position) ===")
comp_best = defaultdict(lambda: {"np": 0, "layer": -1})
for r in d["exp_B_raw"]:
    pos, _ = parse_role(r["role"])
    if pos != "operator":
        continue
    if r["natural_prog"] > comp_best[r["component"]]["np"]:
        comp_best[r["component"]] = {"np": r["natural_prog"], "layer": r["layer"]}
for comp in ["attn", "mlp", "resid_post"]:
    if comp in comp_best:
        b = comp_best[comp]
        print(f"  {comp}: best L{b['layer']} NP={b['np']:.4f}")

# Exp C: Component synergy
print("\n=== Exp C: Component Synergy by layer x position ===")
syn_data = defaultdict(lambda: defaultdict(dict))
for r in d["exp_C_raw"]:
    pos, _ = parse_role(r["role"])
    comp = r["component"]  # attn_only, mlp_only, both
    syn_data[(r["layer"], pos)][comp] = r["natural_prog"]

for layer in sorted(set(k[0] for k in syn_data)):
    for pos in ["operator", "operand", "last"]:
        key = (layer, pos)
        if key not in syn_data:
            continue
        s = syn_data[key]
        a = s.get("attn_only", 0)
        m = s.get("mlp_only", 0)
        b = s.get("both", 0)
        syn = b - a - m
        if abs(a) > 0.01 or abs(m) > 0.01 or abs(b) > 0.01:
            print(f"  L{layer:2d} {pos:10s}: attn={a:.4f} mlp={m:.4f} both={b:.4f} synergy={syn:+.4f}")

# Subtype analysis
print("\n=== Subtype x Position (B->A, resid_post, key layers) ===")
subtype_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
for r in d["exp_A_raw"]:
    if r["component"] != "resid_post":
        continue
    pos, align = parse_role(r["role"])
    subtype_data[r["layer"]][pos][r["subtype"]].append(r["natural_prog"])

for layer in [0, 4, 16, 30, 34, 35]:
    if layer not in subtype_data:
        continue
    print(f"\n  Layer {layer}:")
    for pos in ["operator", "operand", "last"]:
        if pos not in subtype_data[layer]:
            continue
        parts = []
        for st in sorted(subtype_data[layer][pos]):
            vals = subtype_data[layer][pos][st]
            avg = sum(vals) / len(vals)
            parts.append(f"{st}={avg:.3f}({len(vals)})")
        print(f"    {pos}: {' | '.join(parts)}")

# Last layer anomaly
print("\n=== Last Layer Anomaly (resid_post, operand+last, aligned vs misaligned) ===")
for layer in sorted(layer_role_data.keys()):
    if layer < max(layer_role_data.keys()) - 5:
        continue
    for pos in ["operand", "last"]:
        if pos not in layer_role_data[layer]:
            continue
        a = layer_role_data[layer][pos]
        avg_al = sum(a["aligned"]) / len(a["aligned"]) if a["aligned"] else 0
        avg_mi = sum(a["misaligned"]) / len(a["misaligned"]) if a["misaligned"] else 0
        print(f"  L{layer:2d} {pos}: aligned={avg_al:.4f}({len(a['aligned'])}) mis={avg_mi:.4f}({len(a['misaligned'])})")
