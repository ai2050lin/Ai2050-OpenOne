"""Calibrated paired residuals with tie/rank and numerical invariance auditing."""
from rdc_formation_common import *
from rdc_formation_readout import checked_arrays
from phase2747_rdc_training_analysis import source_stat


def main():
    folder = OUT/'calibration_analysis'; finish = folder/'result.json'
    if finish.exists(): return
    start = time.monotonic(); result = read(OUT/'calibration/result.json'); assert result['all_passed']
    data = gzread(OUT/'material/rows.json.gz'); rows = data['validation']+data['diagnostic']+data['fresh']
    control = np.array([r['kind'] == 'controlled_relation' for r in rows]); loaded = {}; checks = []
    for record in result['records']:
        a, b = checked_arrays(record['field']), checked_arrays(record['variant']['receipt'])
        choice = record['selected']; name = record['variant']['name']
        unchanged = np.array_equal(a['argmax'], b['argmax'])
        if choice['alpha'] == 0:
            assert unchanged
            error = np.max(abs(a['binary_gold_margin'][control]-a['uncalibrated_binary_gold_margin'][control]/choice['temperature']))
            assert error < 1e-10
        else: error = None
        checks.append({'variant': name, 'selected': choice, 'argmax_unchanged': unchanged,
            'temperature_margin_identity_error': error,
            'uncalibrated_controlled_exact_margin_ties': int((a['uncalibrated_binary_gold_margin'][control] == 0).sum()),
            'calibrated_controlled_exact_margin_ties': int((a['binary_gold_margin'][control] == 0).sum())})
        loaded[name] = a
    paired = []
    radius = read(OUT/'radius_analysis/result.json')['records']
    for precision in ['native_BF16','FP32_bridge']:
        baseline = loaded['native' if precision == 'native_BF16' else 'bridge']
        for seed in [2747,2748]:
            for factor in [.5,1.]:
                current = [r['variant'] for r in radius if r['variant']['precision'] == precision and r['variant']['seed'] == seed and r['variant']['radius_factor'] == factor]
                true_name = next(r['name'] for r in current if r['condition'] == 'true_token' and r['transform'] == 'identity')
                for variant in current:
                    a = loaded[variant['name']]
                    for split in sorted({r['split'] for r in rows}):
                        ix = [i for i,r in enumerate(rows) if r['split'] == split]; rr = [rows[i] for i in ix]
                        paired.append({'variant': variant, 'matched_true': true_name, 'split': split,
                            'calibrated_minus_own_precision_baseline_NLL': source_stat(a['NLL'][ix]-baseline['NLL'][ix],rr),
                            'calibrated_control_minus_matched_true_NLL': source_stat(a['NLL'][ix]-loaded[true_name]['NLL'][ix],rr)})
    save(finish, {'timestamp': stamp(), 'source': snapshot(__file__), 'all_passed': True,
        'checks': checks, 'paired': paired, 'seconds': time.monotonic()-start,
        'tie_definition': 'Binary rank correct means strictly positive target-minus-other margin, ties fail. Full-vocabulary argmax uses the lowest token index on ties. Thus full argmax pair count can exceed strict binary-rank pair count without inconsistency.',
        'limits': ['All54selectedalpha=0 is the outcome of this frozen finite validation grid, not proof that every possible prior mixture is useless.',
            'Positive scalar temperature does not change token order; NLL gains from it are not new relations.',
            'Residual differences after independently validation-calibrating variants are not a unique semantic component.',
            'Each bootstrap is conditional on fixed seeds and validation choice, not post-selection or seed-population uncertainty.']})
    print('FORMATION_CALIBRATION_ANALYSIS', len(checks),len(paired), flush=True)


if __name__ == '__main__': main()
