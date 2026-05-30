"""Phase 293 Cross-Model Comparison"""
import json
from pathlib import Path
import numpy as np

R = Path("results/phase293_component_contract")

def load(m):
    with open(R / f"{m}_component.json", encoding="utf-8") as f:
        return json.load(f)

models = {"qwen3": "Qwen3", "glm4": "GLM4", "deepseek7b": "DS7B"}
data = {m: load(m) for m in models}

print("=" * 80)
print("Phase 293: Component Contract — Cross-Model Comparison")
print("=" * 80)

# ====== EXP A: Component x Layer Best ======
print("\n## Exp A: Best Layer per Component (by Natural Progress)")
print(f"{'Model':<8} {'Best Attn':>10} {'NP':>8} {'Best MLP':>10} {'NP':>8} {'Best Resid':>10} {'NP':>8}")
print("-" * 70)
for m, name in models.items():
    d = data[m]["exp_A_layer_component"]
    best_a = max((v for k, v in d.items() if v["component"] == "attn"), key=lambda x: x["mean_np"])
    best_m = max((v for k, v in d.items() if v["component"] == "mlp"), key=lambda x: x["mean_np"])
    best_r = max((v for k, v in d.items() if v["component"] == "resid_post"), key=lambda x: x["mean_np"])
    print(f"{name:<8} L{best_a['layer']:>8} {best_a['mean_np']:>8.5f} L{best_m['layer']:>8} {best_m['mean_np']:>8.5f} L{best_r['layer']:>8} {best_r['mean_np']:>8.5f}")

# ====== MLP vs Attn Advantage ======
print("\n## MLP Advantage at L0 (mlp_NP - attn_NP)")
print(f"{'Model':<8} {'Attn NP':>10} {'MLP NP':>10} {'Resid NP':>10} {'MLP_adv':>10} {'Attn/MLP ratio':>15}")
print("-" * 70)
for m, name in models.items():
    d = data[m]["exp_A_layer_component"]
    a0 = d.get("L0_attn", {})
    m0 = d.get("L0_mlp", {})
    r0 = d.get("L0_resid_post", {})
    if a0 and m0:
        adv = m0["mean_np"] - a0["mean_np"]
        ratio = a0["mean_np"] / m0["mean_np"] if m0["mean_np"] > 0 else 0
        print(f"{name:<8} {a0['mean_np']:>10.5f} {m0['mean_np']:>10.5f} {r0.get('mean_np',0):>10.5f} {adv:>+10.4f} {ratio:>15.3f}")

# ====== Position Effect: Early vs Deep ======
print("\n## Position Effect: Early Layers (resid_post)")
print(f"{'Model':<8} {'Layer':>6} {'Operator':>10} {'Operand':>10} {'Last':>10} {'Op-Last':>10} {'Op/Last':>10}")
print("-" * 70)
for m, name in models.items():
    d = data[m]["exp_B_position"]
    for li in [0, 1, 2]:
        op = d.get(f"L{li}_resid_post_operator", {})
        od = d.get(f"L{li}_resid_post_operand", {})
        lt = d.get(f"L{li}_resid_post_last", {})
        if op and od and lt:
            gap = op["mean_np"] - lt["mean_np"]
            ratio = op["mean_np"] / lt["mean_np"] if lt["mean_np"] > 0 else 0
            print(f"{name:<8} L{li:>4} {op['mean_np']:>10.5f} {od['mean_np']:>10.5f} {lt['mean_np']:>10.5f} {gap:>+10.4f} {ratio:>10.2f}x")

print("\n## Position Effect: Deep Layers (resid_post)")
print(f"{'Model':<8} {'Layer':>6} {'Operator':>10} {'Operand':>10} {'Last':>10} {'Last-Op':>10} {'Last/Op':>10}")
print("-" * 70)
deep_layers = {"qwen3": [22, 23, 24], "glm4": [22, 38], "deepseek7b": [22, 25, 26]}
for m, name in models.items():
    d = data[m]["exp_B_position"]
    for li in deep_layers.get(m, []):
        op = d.get(f"L{li}_resid_post_operator", {})
        od = d.get(f"L{li}_resid_post_operand", {})
        lt = d.get(f"L{li}_resid_post_last", {})
        if op and od and lt:
            gap = lt["mean_np"] - op["mean_np"]
            ratio = lt["mean_np"] / op["mean_np"] if op["mean_np"] > 0 else 0
            print(f"{name:<8} L{li:>4} {op['mean_np']:>10.5f} {od['mean_np']:>10.5f} {lt['mean_np']:>10.5f} {gap:>+10.4f} {ratio:>10.2f}x")

# ====== Last Layer Anomaly ======
print("\n## Last Layer Resid_Post Anomaly")
print(f"{'Model':<8} {'L0 NP':>10} {'Last NP':>10} {'L0 KR':>8} {'Last KR':>8} {'Drop':>10}")
print("-" * 60)
last_layers = {"qwen3": 35, "glm4": 38, "deepseek7b": 27}
for m, name in models.items():
    d = data[m]["exp_B_position"]
    l0 = d.get("L0_resid_post_all", {})
    ll = d.get(f"L{last_layers[m]}_resid_post_all", {})
    if l0 and ll:
        drop = l0["mean_np"] - ll["mean_np"]
        print(f"{name:<8} {l0['mean_np']:>10.5f} {ll['mean_np']:>10.5f} {l0['mean_kr']:>8.2f} {ll['mean_kr']:>8.2f} {drop:>+10.4f}")

# ====== Exp C: Alpha Curves at L0 ======
print("\n## Exp C: Alpha Curves at L0 (resid_post component)")
print(f"{'Model':<8} {'a=0 NP':>8} {'a=0.5 NP':>10} {'a=1 NP':>8} {'Slope':>8} {'Linear dev':>12}")
print("-" * 60)
for m, name in models.items():
    d = data[m]["exp_C_alpha"]
    alphas_nps = {}
    for alpha in [0, 0.25, 0.5, 0.75, 1.0]:
        key = f"L0_resid_post_a{alpha:.2f}"
        v = d.get(key)
        if v:
            alphas_nps[alpha] = v["mean_np"]
    if len(alphas_nps) >= 3:
        a0 = alphas_nps.get(0, 0)
        a1 = alphas_nps.get(1.0, 0)
        a05 = alphas_nps.get(0.5, 0)
        slope = a1 - a0
        # Linear interpolation at alpha=0.5
        linear_05 = (a0 + a1) / 2
        lin_dev = abs(a05 - linear_05)
        print(f"{name:<8} {a0:>8.5f} {a05:>10.5f} {a1:>8.5f} {slope:>+8.4f} {lin_dev:>12.5f}")

# ====== Component x Position Interaction ======
print("\n## Component x Position at L0")
print(f"{'Model':<8} {'Attn+Op':>10} {'Attn+Last':>10} {'MLP+Op':>10} {'MLP+Last':>10} {'Attn Op/Lst':>12} {'MLP Op/Lst':>12}")
print("-" * 80)
for m, name in models.items():
    d = data[m]["exp_B_position"]
    ao = d.get("L0_attn_operator", {})
    al = d.get("L0_attn_last", {})
    mo = d.get("L0_mlp_operator", {})
    ml = d.get("L0_mlp_last", {})
    if ao and al and mo and ml:
        a_ratio = ao["mean_np"] / al["mean_np"] if al["mean_np"] > 0 else 0
        m_ratio = mo["mean_np"] / ml["mean_np"] if ml["mean_np"] > 0 else 0
        print(f"{name:<8} {ao['mean_np']:>10.5f} {al['mean_np']:>10.5f} {mo['mean_np']:>10.5f} {ml['mean_np']:>10.5f} {a_ratio:>12.2f}x {m_ratio:>12.2f}x")

# ====== Resid_post L0-L4 Identical Check ======
print("\n## Resid_post 'all' Position: L0-L4 Consistency")
for m, name in models.items():
    d = data[m]["exp_B_position"]
    nps = []
    for li in range(5):
        v = d.get(f"L{li}_resid_post_all")
        if v:
            nps.append(v["mean_np"])
    if nps:
        is_same = all(abs(n - nps[0]) < 0.001 for n in nps)
        print(f"  {name}: L0-L4 = {[f'{n:.5f}' for n in nps]}, Identical={is_same}")

# ====== KR Comparison ======
print("\n## KR at Best Component Layer")
print(f"{'Model':<8} {'Attn best KR':>12} {'MLP best KR':>12} {'Resid best KR':>14}")
print("-" * 50)
for m, name in models.items():
    d = data[m]["exp_A_layer_component"]
    best_a = min((v for k, v in d.items() if v["component"] == "attn" and v["mean_np"] > 0.01), key=lambda x: x["mean_kr"])
    best_m = min((v for k, v in d.items() if v["component"] == "mlp" and v["mean_np"] > 0.01), key=lambda x: x["mean_kr"])
    best_r = min((v for k, v in d.items() if v["component"] == "resid_post" and v["mean_np"] > 0.01), key=lambda x: x["mean_kr"])
    print(f"{name:<8} L{best_a['layer']} KR={best_a['mean_kr']:>6.2f} L{best_m['layer']} KR={best_m['mean_kr']:>6.2f} L{best_r['layer']} KR={best_r['mean_kr']:>6.2f}")

print("\n" + "=" * 80)
print("Phase 293 Comparison Complete")
