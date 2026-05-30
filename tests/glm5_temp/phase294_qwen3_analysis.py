import json
import numpy as np
from collections import defaultdict

d = json.load(open('results/phase294_token_alignment/qwen3_alignment.json', 'r', encoding='utf-8'))
print('=== Qwen3 Phase 294 Summary ===')
print(f'Exp A: {d["exp_A_count"]} results')
print(f'Exp B: {d["exp_B_count"]} results')
print(f'Exp C: {d["exp_C_count"]} results')

# Group Exp A results by (layer, component, role)
a_data = d.get("exp_A_raw", [])
b_data = d.get("exp_B_raw", [])
c_data = d.get("exp_C_raw", [])

# Aligned vs Misaligned comparison (resid_post only)
print('\n=== Resid_post: Aligned vs Misaligned ===')
for role_type in ['operand', 'last']:
    print(f'\n  {role_type}:')
    a_agg = defaultdict(list)
    for r in a_data:
        if r["component"] == "resid_post" and r["role"] == f"{role_type}_aligned":
            a_agg[r["layer"]].append(r["natural_prog"])
    m_agg = defaultdict(list)
    for r in a_data:
        if r["component"] == "resid_post" and r["role"] == f"{role_type}_misaligned":
            m_agg[r["layer"]].append(r["natural_prog"])

    for li in sorted(set(list(a_agg.keys()) + list(m_agg.keys()))):
        a_val = np.mean(a_agg.get(li, [0]))
        m_val = np.mean(m_agg.get(li, [0]))
        diff = a_val - m_val
        if li < 6 or li >= 33 or abs(diff) > 0.01:
            print(f'    L{li}: aligned={a_val:.5f} mis={m_val:.5f} diff={diff:+.5f}')

# A->B operator effect
print('\n=== A->B Operator Effect (best layer per component) ===')
for comp in ['attn', 'mlp', 'resid_post']:
    b_agg = defaultdict(list)
    for r in b_data:
        if r["component"] == comp and r["role"] == "operator":
            b_agg[r["layer"]].append(r["natural_prog"])
    if b_agg:
        best_li = max(b_agg, key=lambda l: np.mean(b_agg[l]))
        print(f'  {comp}: best L{best_li} NP={np.mean(b_agg[best_li]):.5f}')

# Component synergy
print('\n=== Component Synergy (key layers) ===')
for role in ['operand', 'last']:
    print(f'\n  {role}:')
    c_attn = defaultdict(list)
    c_mlp = defaultdict(list)
    c_both = defaultdict(list)
    for r in c_data:
        if r["role"] == role:
            if r["component"] == "attn_only":
                c_attn[r["layer"]].append(r["natural_prog"])
            elif r["component"] == "mlp_only":
                c_mlp[r["layer"]].append(r["natural_prog"])
            elif r["component"] == "attn_mlp":
                c_both[r["layer"]].append(r["natural_prog"])

    for li in sorted(set(list(c_attn.keys()) + list(c_mlp.keys()) + list(c_both.keys()))):
        a_v = np.mean(c_attn.get(li, [0]))
        m_v = np.mean(c_mlp.get(li, [0]))
        b_v = np.mean(c_both.get(li, [0]))
        syn = b_v - a_v - m_v
        if li < 5 or li >= 33 or abs(syn) > 0.005:
            print(f'    L{li}: attn={a_v:.5f} mlp={m_v:.5f} both={b_v:.5f} synergy={syn:+.5f}')

# Last layer focus
print('\n=== Last Layer Focus (resid_post) ===')
for li in [0, 1, 33, 34, 35]:
    for role in ['operand_aligned', 'last_aligned']:
        vals = [r["natural_prog"] for r in a_data
                if r["component"] == "resid_post" and r["layer"] == li and r["role"] == role]
        if vals:
            print(f'  L{li} {role}: NP={np.mean(vals):.5f} (n={len(vals)})')

# Subtype breakdown
print('\n=== Subtype Breakdown (B->A, resid_post, L0, operand_aligned) ===')
st_agg = defaultdict(list)
for r in a_data:
    if r["component"] == "resid_post" and r["layer"] == 0 and r["role"] == "operand_aligned":
        st_agg[r["subtype"]].append(r["natural_prog"])
for st, vals in sorted(st_agg.items()):
    print(f'  {st}: NP={np.mean(vals):.5f} (n={len(vals)})')

# Subtype breakdown for last position at deep layers
print('\n=== Subtype Breakdown (B->A, resid_post, L34, last_aligned) ===')
st_agg2 = defaultdict(list)
for r in a_data:
    if r["component"] == "resid_post" and r["layer"] == 34 and r["role"] == "last_aligned":
        st_agg2[r["subtype"]].append(r["natural_prog"])
for st, vals in sorted(st_agg2.items()):
    print(f'  {st}: NP={np.mean(vals):.5f} (n={len(vals)})')
