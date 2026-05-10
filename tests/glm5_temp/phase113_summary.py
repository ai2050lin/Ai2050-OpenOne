import json
d = json.load(open('phase113_qwen3_all_results.json'))

# Exp 1
s = d['exp1_null_rotation']['summary']
print('=== Exp 1: Null Hypothesis Test ===')
for k, v in sorted(s.items()):
    ra = v['real_angle']
    nb = v['null_b_mean']
    ns = v['null_b_std']
    z = v['null_b_z']
    vd = v['verdict']
    print(f"  {k}: real={ra:.1f}, null_B={nb:.1f}+-{ns:.2f}, z={z:+.2f}, {vd}")

# How many REJECT_NULL?
reject = sum(1 for v in s.values() if v['is_significant'])
print(f"\nREJECT_NULL: {reject}/{len(s)} layer pairs")

# Individual angles
ia = d['exp1_null_rotation']['individual_angles']
print('\n=== Individual Sample Angles ===')
for k in sorted(ia.keys()):
    v = ia[k]
    print(f"  {k}: mean={v['mean']:.1f}, std={v['std']:.1f}")

# Exp 2 Principal Angles
print('\n=== Exp 2: Principal Angles ===')
pa = d['exp2_subspace_dynamics']['principal_angles']
for k in sorted(pa.keys()):
    v = pa[k]
    print(f"  {k}: min={v['min_angle']:.1f}, max={v['max_angle']:.1f}, mean={v['mean_angle']:.1f}")

# Null for subspace
null_min = d['exp2_subspace_dynamics']['null_min_angle_mean']
null_min_std = d['exp2_subspace_dynamics']['null_min_angle_std']
print(f"  Null random subspace: min_angle={null_min:.1f}+-{null_min_std:.2f}")

# Subspace info - PR
si = d['exp2_subspace_dynamics']['subspace_info']
print('\n=== Subspace Participation Ratio ===')
for l in sorted(si.keys(), key=int):
    v = si[l]
    print(f"  L{l}: PR={v['participation_ratio']:.1f}, dim_50={v['cumvar_50']}, dim_90={v['cumvar_90']}, dim_95={v['cumvar_95']}")

# Exp 3
print('\n=== Exp 3: Operator Decomposition ===')
od = d['exp3_operator_decomposition']
for k in sorted(od.keys()):
    v = od[k]
    print(f"  {k}: attn_frac={v['attn_energy_fraction']:.1%}, mlp_frac={v['mlp_energy_fraction']:.1%}, cos_align={v['cos_alignment']:.3f}, attn_PR={v['attn_diff_pr']:.1f}, mlp_PR={v['mlp_diff_pr']:.1f}")

# Exp 4
print('\n=== Exp 4: Route Topology ===')
rt = d['exp4_route_topology']
rs = rt['rank_stability']
print('Rank stability (Spearman):')
for k in sorted(rs.keys()):
    v = rs[k]
    print(f"  {k}: rho={v['spearman_rho']:.4f}, p={v['p_value']:.2e}")

pp = rt['path_persistence']
print('Path persistence (individual top-10 overlap):')
for k in sorted(pp.keys()):
    v = pp[k]
    if v['mean'] > 0.001:
        print(f"  {k}: mean={v['mean']:.2%}, std={v['std']:.2%}")
