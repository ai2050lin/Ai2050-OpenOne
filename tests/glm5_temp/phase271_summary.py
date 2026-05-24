import json, sys
sys.stdout.reconfigure(encoding='utf-8')

model = sys.argv[1] if len(sys.argv) > 1 else "qwen3"

# Transport R² Sanity
with open(f'results/phase271_topology_preservation/{model}_transport_sanity.json') as f:
    sanity = json.load(f)

print(f'=== Transport R2 Sanity Check ({model}) ===')
for l, r in sanity.items():
    print(f'  L{l}:')
    print(f'    Transport R2 = {r["r2_transport_original"]:.4f} (p_perm={r["p_value_transport"]:.4f})')
    print(f'    Persistence R2 = {r["r2_persist_original"]:.4f}')
    print(f'    Combined R2 = {r["r2_combined_original"]:.4f}')
    print(f'    Shuffled R2 = {r["r2_shuffled_mean"]:.4f} +/- {r["r2_shuffled_std"]:.4f} (max={r["r2_shuffled_max"]:.4f})')
    print(f'    Random Subspace R2 = {r["r2_random_subspace"]:.4f} (p_perm={r["p_value_random"]:.4f})')
    print(f'    Random Shuffled R2 = {r["r2_random_shuffled_mean"]:.4f}')
    print()

# Topology Preservation
with open(f'results/phase271_topology_preservation/{model}_topology_preservation.json') as f:
    topo = json.load(f)

print(f'=== Topology Preservation ({model}) ===')
mantel = topo["experiment_b_topology"]["mantel_correlation"]
print('Mantel correlation vs final layer:')
for space in ["full", "vis", "inv"]:
    print(f'  {space}:', end='')
    for l in sorted(mantel[space].keys(), key=int):
        print(f' L{l}={mantel[space][l]["spearman_r"]:.4f}', end='')
    print()

wb = topo["experiment_b_topology"]["within_between"]
print('\nWithin vs Between category:')
for l in sorted(wb.keys(), key=int):
    w = wb[l]
    print(f'  L{l}: W_full={w["within_full"]:.4f}, B_full={w["between_full"]:.4f}, '
          f'Diff={w["within_full"]-w["between_full"]:.4f} | '
          f'W_vis={w["within_vis"]:.4f}, W_inv={w["within_inv"]:.4f} | '
          f'B_vis={w["between_vis"]:.4f}, B_inv={w["between_inv"]:.4f}')

# Cross-space agreement
cs = topo["experiment_c_cross_space"]
print('\nV_vis vs V_inv Cross-space Agreement:')
for l in sorted(cs.keys(), key=int):
    c = cs[l]
    print(f'  L{l}: Full-Vvis={c["full_vis_spearman"]:.4f}, Full-Vinv={c["full_inv_spearman"]:.4f}, '
          f'Vvis-Vinv={c["vis_inv_spearman"]:.4f} | var: vis={c["vis_variance_frac"]:.3f}, inv={c["inv_variance_frac"]:.3f}')

# Nearest neighbor
nn = topo["experiment_b_topology"]["nearest_neighbor"]
print('\nNearest Neighbor Preservation (k=5):')
for l in sorted(nn.keys(), key=int):
    print(f'  L{l}: {nn[l]:.4f}')

# Adjacent stability
adj = topo["experiment_b_topology"]["adjacent_stability"]
print('\nAdjacent-layer topology stability (selected):')
for l in sorted(adj.keys(), key=int):
    if int(l) % 9 == 0 or int(l) == 35:
        a = adj[l]
        print(f'  L{l}->L{int(l)+1}: Full={a["full"]:.4f}, V_vis={a["vis"]:.4f}, V_inv={a["inv"]:.4f}')
