import json

for m in ['qwen3', 'glm4', 'deepseek7b']:
    with open(f'results/phase270_subspace_transport/{m}_subspace_transport.json') as f:
        data = json.load(f)

    cum = data['cumulative_transport']
    adj = data['adjacent_transport']
    spec = data['spectrum']
    var_d = data['variance_decomposition']

    n_total = max(int(k) for k in cum.keys()) + 1

    print(f"=== {m} ===")

    # Cumulative transport: key layers
    print("  Cumulative transport (L{l} -> final):")
    key_layers = [0, n_total//4, n_total//2, 3*n_total//4, n_total-1]
    for l in key_layers:
        c = cum.get(str(l))
        if c:
            print(f"    L{l}: persist_R2={c['r2_persistence']:.4f}, transport_R2={c['r2_transport']:.4f}, "
                  f"combined_R2={c['r2_combined']:.4f}, transport_ratio={c['transport_ratio']:.4f}")

    # Adjacent transport: first, mid, last
    print("  Adjacent transport:")
    for l in [0, n_total//2, n_total-2]:
        a = adj.get(str(l))
        if a:
            print(f"    L{l}->L{l+1}: persist_R2={a['r2_persistence']:.4f}, transport_R2={a['r2_transport']:.4f}, "
                  f"delta_vis={a['delta_vis_frac']:.3f}, delta_inv={a['delta_inv_frac']:.3f}")

    # Variance decomposition: first, mid, last
    print("  Variance decomposition:")
    for l in [0, n_total//2, n_total-1]:
        v = var_d.get(str(l))
        if v:
            print(f"    L{l}: vis_frac={v['vis_frac']:.4f}, inv_frac={v['inv_frac']:.4f}")

    # Spectrum: first, mid, last
    print("  Spectrum:")
    for l in [0, n_total//2, n_total-1]:
        s = spec.get(str(l))
        if s:
            print(f"    L{l}: PR={s['pr']:.1f}, n50={s['n_50var']}, n90={s['n_90var']}, n99={s['n_99var']}, "
                  f"tail={s['tail_energy_pct']:.2f}%")

    print()
