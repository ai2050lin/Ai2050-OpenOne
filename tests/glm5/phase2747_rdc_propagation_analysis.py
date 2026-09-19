"""Precision-separated endpoint inference and unselected coordinate history fields."""
from rdc_formation_common import *
from rdc_formation_propagation import material, PROP, SCALES
from rdc_formation_readout import checked_arrays
from phase2747_rdc_training_analysis import source_stat

FIELDS_NAMES = ['hidden', 'Q', 'gate', 'up', 'product', 'MLP_input', 'MLP_write']


def main():
    folder = PROP/'analysis'; finish = folder/'result.json'
    if finish.exists(): return
    start = time.monotonic()
    actual = read(PROP/'smooth/result.json'); assert actual['all_passed']
    rows = material(); names = actual['direction_names']
    records = gzread(PROP/'smooth/endpoint_records.json.gz')
    assert len(records) == len(rows)*len(names)*len(SCALES)
    summary = []
    for name in names:
        for scale in SCALES:
            for family in ['all']+sorted({r['family'] for r in rows}):
                rr = [r for r in records if r['run'] == name and r['scale'] == scale and (family == 'all' or r['family'] == family)]
                def measure(fn): return source_stat([fn(r) for r in rr], rr)
                value = {'run': name, 'scale': scale, 'family': family}
                value['baseline_precision_KL'] = measure(lambda r: r['smooth_baseline_to_native_baseline_KL'])
                value['finite_precision_KL'] = measure(lambda r: r['native_finite_to_raw_smooth_finite_KL'])
                value['native_no_change_KL'] = measure(lambda r: r['native_finite_to_native_baseline_KL'])
                value['smooth_no_change_KL'] = measure(lambda r: r['smooth_finite_to_smooth_baseline_KL'])
                for control in ['full_prefix', 'last_position_only']:
                    value[control] = {
                        'smooth_KL': measure(lambda r: r[control]['smooth_finite_to_linear_KL']),
                        'native_centered_KL': measure(lambda r: r[control]['native_finite_to_baseline_centered_linear_KL']),
                        'native_centered_minus_no_change_KL': measure(lambda r: r[control]['native_finite_to_baseline_centered_linear_KL']-r['native_finite_to_native_baseline_KL']),
                        'smooth_minus_no_change_KL': measure(lambda r: r[control]['smooth_finite_to_linear_KL']-r['smooth_finite_to_smooth_baseline_KL']),
                        'native_argmax_agreement': measure(lambda r: r[control]['native_centered_argmax'] == r[control]['native_finite_argmax'])}
                value['last_minus_full_smooth_KL'] = measure(lambda r: r['last_position_only']['smooth_finite_to_linear_KL']-r['full_prefix']['smooth_finite_to_linear_KL'])
                value['last_minus_full_native_KL'] = measure(lambda r: r['last_position_only']['native_finite_to_baseline_centered_linear_KL']-r['full_prefix']['native_finite_to_baseline_centered_linear_KL'])
                summary.append(value)
    layer_results = []; collected = {name: [] for name in FIELDS_NAMES}; source_receipts = []
    for block, receipt in zip(range(16, 36), actual['complete_field_receipts']):
        a = checked_arrays(receipt); source_receipts.append(receipt)
        metrics = a['full_coordinate_finite_error']
        assert metrics.shape == (64, 6, 3, 2, 7, 3)
        for j, name in enumerate(FIELDS_NAMES):
            tangent = a['tangent_'+name].astype(float)
            full, last = tangent[:, ::2], tangent[:, 1::2]
            baseline = a['base_'+name].astype(float)
            # RMS normalization uses the same observed smooth baseline vector
            # for both routes. It does not recenter or fit the derivative.
            rms = np.sqrt(np.mean(baseline**2, -1)).clip(1e-12)
            squared = (full-last)**2
            maps = np.stack([squared.mean(0), (squared/rms[:, None, None]**2).mean(0)])
            collected[name].append(maps)
            precision = np.mean((baseline-a['native_base_'+name].astype(float))**2, -1)
            layer_results.append({'block': block, 'field': name, 'coordinates': baseline.shape[-1],
                'precision_baseline_MSE': source_stat(precision, rows),
                'full_minus_last_tangent_MSE_per_run': [source_stat(squared[:, ri].mean(-1), rows) for ri in range(6)],
                'source_balanced_finite_error': [
                    {'run': run, 'scale': scale, 'control': control,
                     'observed_delta_MSE': source_stat(metrics[:, ri, si, ci, j, 0], rows),
                     'linear_error_MSE': source_stat(metrics[:, ri, si, ci, j, 1], rows)}
                    for ri, run in enumerate(names) for si, scale in enumerate(SCALES)
                    for ci, control in enumerate(['full_prefix', 'last_position_only'])]})
        del a, metrics
        print('FORMATION_PROPAGATION_ANALYSIS_LAYER', block, flush=True)
    fields = {name+'_history_MSE': np.stack(values, axis=2) for name, values in collected.items()}
    # [view, run, block, native coordinate], no coordinate is discarded.
    for array in fields.values(): assert array.shape[:3] == (2, 6, 20) and np.isfinite(array).all()
    receipt = commit_array('figures', 'parameter_history_complete_coordinates', **fields)
    result = {'timestamp': stamp(), 'source': snapshot(__file__), 'all_passed': True,
        'endpoint_summary': summary, 'layer_summary': layer_results, 'coordinate_field': receipt,
        'coordinate_axes': ['view(raw MSE, baseline-RMS normalized MSE)', 'run', 'block16..35', 'original coordinate'],
        'run_order': names, 'source_receipts': source_receipts,
        'sample_weighting': 'Endpoint and layer intervals equal present families then sources, conditional on frozen seeds. Coordinate overview means average all64expressions descriptively, not independent-coordinate inference.',
        'limits': ['Original endpoint is explicitly observed in baseline-centered prediction.',
                   'NativeBF16 is discontinuously rounded and is not the differentiable FP32 reference.',
                   'Positive last-minus-full means fullprefix is better; negative means worse.',
                   'Per-family panels have only two controlled source groups each; their intervals do not establish broad family generalization.',
                   'No-change is an explicit baseline; outperforming an intentionally incomplete tangent alone is insufficient.',
                   'Full-vocabulary KL, target NLL and own-history success remain separate.'],
        'seconds': time.monotonic()-start}
    save(finish, result); ledger('phase2747_propagation_analysis', result['seconds'])
    print('FORMATION_PROPAGATION_ANALYSIS_DONE', result['seconds'], flush=True)


if __name__ == '__main__': main()
