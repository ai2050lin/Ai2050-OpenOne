"""Scientific plots of complete prefix paths, precision limits and every coordinate."""
from rdc_formation_common import *
from rdc_formation_readout import checked_arrays


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = OUT/'figures/propagation_v1'; folder.mkdir(parents=True, exist_ok=True)
    finish = folder/'index.json'
    if finish.exists(): return
    source = OUT/'parameter_propagation/analysis/result.json'
    result = read(source); assert result['all_passed']
    names = result['run_order']; reports = result['endpoint_summary']; images = []
    labels = ['True / 2747', 'Class shuffle / 2747', 'Class mass / 2747',
              'True / 2748', 'Class shuffle / 2748', 'Class mass / 2748']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.spines.top': False, 'axes.spines.right': False})
    def savefig(fig, name, caption):
        path = folder/(name+'.png'); fig.savefig(path, dpi=160, bbox_inches='tight', facecolor='white'); plt.close(fig)
        images.append({'name': name, 'path': path.relative_to(BASE).as_posix(), 'sha256': sha(path), 'caption': caption,
            'source': {'path': source.relative_to(BASE).as_posix(), 'sha256': sha(source)}})
    fig, axes = plt.subplots(3, 4, figsize=(17, 11), layout='constrained')
    for ni, name in enumerate(names):
        rr = sorted([r for r in reports if r['run'] == name and r['family'] == 'all'], key=lambda r: r['scale'])
        for ci, reference in enumerate(['smooth', 'native']):
            ax = axes[ni%3, (ni//3)*2+ci]; x = [r['scale'] for r in rr]
            for control, color, label in [('full_prefix', '#226986', 'Full prefix'), ('last_position_only', '#ba762c', 'Last only')]:
                metric = 'smooth_KL' if reference == 'smooth' else 'native_centered_KL'
                ax.plot(x, [r[control][metric]['mean'] for r in rr], 'o-', color=color, label=label)
            metric = reference+'_no_change_KL'
            ax.plot(x, [r[metric]['mean'] for r in rr], 's--', color='#697c67', label='No change')
            if reference == 'native':
                ax.plot(x, [r['finite_precision_KL']['mean'] for r in rr], ':', color='#975376', label='Raw FP32 finite')
            ax.set(xscale='log', yscale='log', title=labels[ni]+' | '+reference, xlabel='Direction scale', ylabel='Full-vocabulary KL')
            ax.set_xticks([.1,.3,1.], ['0.1','0.3','1']); ax.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8); axes[0, 1].legend(fontsize=8)
    fig.suptitle('Linear parameter propagation: smooth reference and native BF16 are separate targets\n64 declared expressions; equal family / source weighting. Log axes; each panel has its own vertical range.', fontsize=14)
    savefig(fig, 'propagation_precision_and_zero_baseline', 'Every run/scale retained. Native linear prediction observes original native endpoint; this is a disclosed reference, not early-only extraction. Curves connect three tested scales, not a continuous fitted law.')
    families = sorted({r['family'] for r in reports if r['family'] != 'all'})
    for reference in ['smooth', 'native']:
        arr = np.array([[next(r for r in reports if r['run'] == name and r['scale'] == scale and r['family'] == family)['last_minus_full_'+reference+'_KL']['mean']
                         for scale in [.1,.3,1.] for family in families] for name in names])
        fig, ax = plt.subplots(figsize=(16, 5.3), layout='constrained')
        transformed = np.arcsinh(arr/1e-4); lim = max(abs(transformed).max(), 1e-12)
        im = ax.imshow(transformed, cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
        ax.set_yticks(range(6), labels)
        ax.set_xticks(range(18), [f.replace('relation_', '')+'\ns='+str(s) for s in [.1,.3,1.] for f in families], rotation=40, ha='right', fontsize=8)
        for at in [5.5,11.5]: ax.axvline(at, color='white')
        ax.set_title(reference+' reference: last-only KL minus full-prefix KL\nPositive = full-prefix better; negative = full-prefix worse. All six families and scales retained.')
        fig.colorbar(im, ax=ax, shrink=.8, label='asinh(KL difference / 1e-4); signed, no threshold')
        savefig(fig, 'propagation_family_'+reference, 'Display of source-balanced means; all bootstrap intervals are in the analysis JSON. Each controlled family has only two source groups, limiting inference; Chinese natural has24sources. Colors are not significance tests.')
    arrays = checked_arrays(result['coordinate_field'])
    for field in ['hidden', 'gate']:
        for vi, view in enumerate(['raw', 'RMS_normalized']):
            data = arrays[field+'_history_MSE'][vi]; display = np.arcsinh(data/1e-10)
            fig, axes = plt.subplots(6, 1, figsize=(16, 11.5), sharex=True, sharey=True, layout='constrained')
            for i, ax in enumerate(axes):
                im = ax.imshow(display[i], aspect='auto', interpolation='nearest', cmap='magma', vmin=0, vmax=display.max())
                ax.set_title(labels[i]); ax.set_yticks([0,9,19], [16,25,35]); ax.set_ylabel('Block')
            axes[-1].set_xlabel('All original '+('residual coordinates' if field == 'hidden' else 'gate MLP unit indices')+'; 0..'+str(data.shape[-1]-1))
            fig.colorbar(im, ax=axes, shrink=.7, label='asinh(mean squared full-minus-last tangent / 1e-10)')
            fig.suptitle(field+' | '+view+': effect of the omitted earlier-prefix tangent on every current-position coordinate\nAll64expressions averaged descriptively; all20blocks retained; common scale across six runs, no ranking or Top-K.', fontsize=13)
            savefig(fig, 'complete_history_'+field+'_'+view, 'Raw and baseline-whole-vector-RMS normalized views are separate. Same-index cross-layer function is not assumed. Raster pixels may merge neighboring coordinates; exact native-index numeric arrays remain queryable.')
    save(finish, {'timestamp': stamp(), 'source': snapshot(__file__), 'all_passed': True, 'figures': images,
        'numeric_fields': result['coordinate_field'], 'visual_review': 'Pending actual main-agent image inspection.'})
    print('FORMATION_PROPAGATION_FIGURES', len(images), flush=True)


if __name__ == '__main__': main()
