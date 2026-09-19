"""Actual training/radius figures and full-coordinate changes; no axis selection."""
from rdc_formation_common import *


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = OUT/'figures/training_v2'; folder.mkdir(parents=True, exist_ok=True)
    finish = folder/'training_index.json'
    if finish.exists():
        assert all(sha(BASE/r['path']) == r['sha256'] for r in read(finish)['figures'])
        return
    start = time.monotonic(); records = []
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    labels = ['True target', 'Class-matched shuffle', 'Surface-class mass']
    conditions = read(OUT/'training/protocol.json')['conditions']
    colors = ['#216485', '#b46727', '#648049']
    def savefig(fig, name, caption, sources):
        path = folder/(name+'.png'); fig.savefig(path, dpi=160, facecolor='white', bbox_inches='tight'); plt.close(fig)
        records.append({'name': name, 'path': path.relative_to(BASE).as_posix(), 'sha256': sha(path),
            'caption': caption, 'sources': [{'path': str(p.relative_to(BASE)), 'sha256': sha(p)} for p in sources],
            'coordinate_policy': 'All original axes kept in source arrays; overview raster pixels may merge neighboring coordinates. No Top-K, compression basis or amplitude-based index reorder.'})
    summaries = read(OUT/'training_analysis/result.json')['summary']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), layout='constrained')
    for ci, condition in enumerate(conditions):
        for si, seed in enumerate([2747, 2748]):
            entry = next(r for r in summaries if r['condition'] == condition and r['seed'] == seed and r['step'] == 'deployed_BF16')
            rr = next(r for r in entry['endpoint_reports'] if r['family'] == 'all' and r['split'] == 'old_exposed_new_wording_diagnostic')
            point = rr['NLL_minus_baseline']; x = ci+(si-.5)*.18
            axes[0].vlines(x, *point['interval95'], color=colors[ci]); axes[0].plot(x, point['mean'], 'o' if si == 0 else '^', color=colors[ci])
            count = rr['both_worlds_first_token_correct']; axes[1].plot(x, count, 'o' if si == 0 else '^', color=colors[ci], markersize=8)
            axes[1].annotate(str(count), (x, count), xytext=(0, 8 if si == 0 else -15), textcoords='offset points', ha='center', color=colors[ci])
    for ax in axes:
        ax.set_xticks(range(3), labels); ax.grid(axis='y', alpha=.2)
    axes[0].axhline(0, color='gray', linestyle='--'); axes[0].set(title='Correct literal token NLL change', ylabel='Final BF16 minus original BF16 NLL')
    axes[1].axhline(99, color='gray', linestyle='--', label='Original: 99 / 160')
    axes[1].set(title='Both opposite worlds win full-vocabulary argmax', ylabel='Correct pairs out of 160', ylim=(0, 160)); axes[1].legend(loc='lower right')
    fig.suptitle('Probability improvement and relation-pair separation are different measurements\nActual final learned displacements differ; circle=seed2747, triangle=seed2748', fontsize=14)
    savefig(fig, 'training_probability_pair_boundary', 'All320diagnostic expressions; NLL intervals resample case/source groups and equal-weight present families. Pair counts are firsttoken, not complete ownhistory answers. Actual unmatched final learning endpoints, not equal-radius controls.', [OUT/'training_analysis/result.json'])
    radius = read(OUT/'radius_analysis/result.json')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), layout='constrained')
    directions = [(conditions[0], 'identity', 'True target'), (conditions[1], 'identity', 'Class-matched shuffle'),
        (conditions[2], 'identity', 'Surface-class mass'), (conditions[0], 'reverse', 'Reversed true'), (conditions[0], 'coordinate_shuffle', 'Index-shuffled true')]
    palette = colors+['#8b4c6d', '#75828b']
    for ax, precision in zip(axes, ['FP32_bridge', 'native_BF16']):
        for di, (condition, transform, label) in enumerate(directions):
            for si, seed in enumerate([2747, 2748]):
                rr = sorted([r for r in radius['records'] if r['variant']['condition'] == condition and r['variant']['transform'] == transform and r['variant']['seed'] == seed and r['variant']['precision'] == precision], key=lambda r: r['variant']['radius_factor'])
                x = [0.]+[r['CPU_full_parameter_reconstructed_radius'] for r in rr]
                y = [0.]+[next(p for p in r['endpoint_reports'] if p['family'] == 'all' and p['split'] == 'prospective_source_confirmation')['NLL_minus_baseline']['mean'] for r in rr]
                ax.plot(x, y, '-' if si == 0 else '--', marker='o' if si == 0 else '^', color=palette[di], label=label if si == 0 else None, alpha=.9)
        ax.set(title=precision+' (its own precision baseline)', xlabel='Actual complete-parameter displacement L2', ylabel='New 192 Chinese-source NLL minus baseline'); ax.grid(alpha=.18); ax.axhline(0, color='black', linewidth=.6)
    axes[1].legend(fontsize=8, loc='upper left')
    fig.suptitle('Direction and radius jointly condition actual heldout response\n0, half and full matched radius only; joining lines are visual guides, not inferred continuous laws', fontsize=14)
    savefig(fig, 'matched_radius_response', 'All40predeclared controls; all74,711,040parameters. Two fixed seeds are shown separately, not treated as a seed-population sample. FP32 bridge and nativeBF16 baselines differ and are not mixed.', [OUT/'radius_analysis/result.json'])
    gradient = read(OUT/'gradient/result.json')
    fig, ax = plt.subplots(figsize=(7.5, 5.8), layout='constrained')
    values = np.array(gradient['full_parameter_gradient_cosine'])
    im = ax.imshow(values, vmin=-1, vmax=1, cmap='RdBu_r')
    for i in range(3):
        for j in range(3): ax.text(j, i, f'{values[i,j]:.4f}', ha='center', va='center', color='white' if abs(values[i,j])>.7 else 'black', fontsize=12)
    ax.set_xticks(range(3), labels, rotation=15, ha='right'); ax.set_yticks(range(3), labels)
    fig.colorbar(im, ax=ax, label='Complete-parameter gradient cosine', shrink=.8)
    fig.suptitle('Initial gradient geometry on the fixed, proportional 256-example panel\n216 natural + 40 controlled; all 74,711,040 scalar parameters', fontsize=12)
    savefig(fig, 'complete_gradient_compatibility', 'Every scalar in gate/up/down contributes. This256example composition differs from the8example one-per-family engineering pilot; differences cannot be attributed to sample size alone.', [OUT/'gradient/result.json'])
    base_receipt = read(OUT/'training/baseline/commits/native_fields.json')
    with np.load(ROOT/base_receipt['field_path']) as z: original = z['hidden'].astype(float)
    native_norm = original/np.sqrt(np.mean(original**2, -1, keepdims=True)).clip(1e-12)
    matrices = []; names = []; sources = [ROOT/base_receipt['field_path']]
    for run in read(OUT/'training/result.json')['runs']:
        receipt = run['deployment_fields']; path = ROOT/receipt['field_path']
        assert sha(path) == receipt['field_sha256']; sources.append(path)
        with np.load(path) as z: current = z['hidden'].astype(float)
        norm = current/np.sqrt(np.mean(current**2, -1, keepdims=True)).clip(1e-12)
        matrices.append(np.stack([np.mean((current-original)**2, axis=0), np.mean((norm-native_norm)**2, axis=0)]))
        names.append(labels[conditions.index(run['condition'])]+' / '+str(run['seed']))
    arrays = np.stack(matrices)
    prior = OUT/'figures/commits/training_full_coordinate_changes.json'
    if prior.exists():
        receipt = read(prior)
        assert sha(ROOT/receipt['field_path']) == receipt['field_sha256']
        with np.load(ROOT/receipt['field_path']) as z: assert np.array_equal(z['mean_squared_change'], arrays)
    else:
        receipt = commit_array('figures', 'training_full_coordinate_changes', mean_squared_change=arrays)
    for vi, title in enumerate(['Raw coordinate change', 'Whole-vector RMS-normalized change']):
        display = np.arcsinh(arrays[:, vi]/1e-10)
        fig, axes = plt.subplots(6, 1, figsize=(16, 12), layout='constrained', sharex=True, sharey=True)
        for i, ax in enumerate(axes):
            im = ax.imshow(display[i], aspect='auto', origin='upper', interpolation='nearest', cmap='magma', vmin=0, vmax=display.max())
            ax.set_title(names[i], fontsize=10); ax.set_yticks([0,16,24,36]); ax.set_ylabel('H boundary')
        axes[-1].set_xlabel('Every original residual coordinate (0..2559)')
        fig.colorbar(im, ax=axes, shrink=.7, label='asinh(mean squared change / 1e-10); no threshold or clipping')
        fig.suptitle(title+' after actual BF16 parameter learning\nSame 44 declared expressions; all 37 boundaries and 2,560 coordinates. H0..H16 unchanged; effects begin at H17.', fontsize=14)
        savefig(fig, 'complete_hidden_change_'+str(vi), 'All coordinates, including exact-zero early boundaries and low-amplitude later background. Shared color scale within this view. Expression mean is descriptive;44expressions are not44independent sources. Same coordinate index at different layers is not assumed functionally identical.', sources)
    value = {'timestamp': stamp(), 'source': snapshot(__file__), 'all_passed': True, 'figures': records,
        'full_coordinate_figure_field': receipt, 'visual_review': 'Pending actual image inspection; renderer success is not visual verification.',
        'phase_complete': False, 'seconds': time.monotonic()-start}
    save(finish, value)
    save(OUT/'figures/current_training_figures.json', {'timestamp': stamp(), 'index': finish.relative_to(BASE).as_posix(),
        'sha256': sha(finish), 'revision': 'Separated overlapping H16/H17 tick labels and improved title spacing. Original images and exact numeric arrays preserved.'})
    ledger('phase2747_training_figures', value['seconds']); print('FORMATION_TRAINING_FIGURES', len(records), value['seconds'], flush=True)


if __name__ == '__main__': main()
