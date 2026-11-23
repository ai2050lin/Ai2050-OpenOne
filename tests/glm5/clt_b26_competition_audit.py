"""Verify frozen populations, predictor artifacts and the conditional causal handoff."""
import numpy as np

import clt_b26_competition as campaign
import clt_b26_three_cues_close as audit
from source_binding_design import read, sha

B = campaign.B


def verify():
    pre = B.common.checked()
    previous_names = B.previous_names()
    names = set(previous_names)
    natural = {}
    for split in ('pilot', 'discovery', 'confirm', 'intervention'):
        rows = B.materials(split, pre)
        audit.check_material(rows)
        current = {name for row in rows for name in row['names']}
        assert len(current) == 5 * B.SIZES[split] and names.isdisjoint(current)
        names.update(current)
        if split != 'intervention':
            report, _, dataset = audit.audit_natural(pre, split)
            assert report['passed']
            natural[split] = report
        if split == 'confirm':
            confirm_rows, confirm_data = rows, dataset
            confirm_names = set(names)

    out = B.OUT
    result = read(out/'evaluation.json')
    sealed = read(out/'candidate.json')
    assert result['provenance'] == pre['provenance'] == sealed['provenance']
    assert result['candidate_sha'] == sha(out/'candidate.json')
    assert result['collection_sha'] == sha(out/'confirm-collection.json')
    assert sealed['ranking_sha'] == sha(out/'competition-discovery.npz')
    assert sealed['predictor_sha'] == sha(out/'predictor.npz')
    assert sealed['prereg_sha'] == sha(out/'prereg.json')
    assert sealed['discovery_sha'] == sha(out/'discovery-collection.json')
    assert pre['reference_predictor_sha'] == sha(campaign.REFERENCE/'predictor.npz')
    with (np.load(out/'competition-discovery.npz') as ranking,
          np.load(out/'predictor.npz') as current,
          np.load(campaign.REFERENCE/'predictor.npz') as prior):
        assert ranking['count'][0] == len(B.materials('discovery', pre))
        for j, branch in enumerate(('q', 'k')):
            scores = ranking['scores'][j]
            assert np.isfinite(scores).all() and np.array_equal(scores, current[f'rank_{branch}'])
            columns = np.argsort(-scores, kind='stable')
            for budget in B.BUDGETS:
                assert np.array_equal(current[f'{branch}_selected_{budget}'], columns[:budget])
            assert np.array_equal(current[f'{branch}_prior_512'], prior[f'{branch}_selected_512'])

    reconstructed = np.concatenate([chunk['mass_actual'] for chunk in confirm_data])
    raw_top = reconstructed.argmax(1)
    cue_indices = np.array([row['cue_targets'] for row in confirm_rows])

    def centered(mass):
        logged = np.log(np.maximum(np.take_along_axis(mass, cue_indices, axis=1), 1e-20))
        return logged - logged.mean(1, keepdims=True)

    def error(label):
        mass = np.concatenate([chunk[f'mass_{label}'] for chunk in confirm_data])
        assert np.mean(mass.argmax(1) == raw_top) == result['prediction'][label]['five_way_agreement']
        values = np.square(centered(mass) - centered(reconstructed)).mean(1)
        assert np.isclose(values.mean(), result['prediction'][label]['margin_mse'])
        return values

    admitted = []
    b = np.random.default_rng(B.SEED+61).integers(0, B.SIZES['confirm'], (10000, B.SIZES['confirm']))
    for branch in ('q', 'k'):
        label = f'{branch}_selected_512'
        selected_error = error(label)
        error(f'{branch}_prior_512')
        comparisons = []
        for baseline in (f'{branch}_random_512', f'{branch}_mean_0'):
            baseline_error = error(baseline)
            improvement = (baseline_error - selected_error).reshape(-1, 12).mean(1)
            lower = float(np.quantile(improvement[b].mean(1), .0125))
            gain = 1 - selected_error.mean() / max(baseline_error.mean(), 1e-20)
            comparison = result['gates'][branch]['against'][baseline]
            assert np.isclose(comparison['gain'], gain) and np.isclose(comparison['paired'][1], lower)
            comparisons.append(gain >= .20 and lower > 0)
        passes = result['prediction'][label]['five_way_agreement'] >= .90 and all(comparisons)
        assert passes == result['gates'][branch]['passed']
        if passes:
            admitted.append(branch)
    assert admitted == result['admitted']
    cal_path = out/'calibration-causal.json'
    cause = audit.check_causal(pre, confirm_names)
    if admitted:
        lock = read(out/'causal-freeze.json')
        cal = read(cal_path)
        assert cal['freeze_sha'] == sha(out/'causal-freeze.json')
        assert lock['evaluation_sha'] == sha(out/'evaluation.json')
        assert cal['traces_sha'] == sha(out/'calibration-causal.npz')
        assert list(cal['branches']) == admitted
        calibrated = [key for key in admitted if cal['branches'][key]['technical_pass']]
        assert bool(calibrated) == (cause is not None)
        if cause:
            assert list(cause['branches']) == calibrated
            for branch in calibrated:
                tested = cause['branches'][branch]
                assert tested['passes'] == (tested['technical_pass'] and tested['primary'][1] > 0)
    else:
        assert cause is None and not cal_path.exists() and not (out/'causal-freeze.json').exists()

    facts = audit.native_and_swaps(confirm_rows, confirm_data, result)
    receipt = {
        'provenance': pre['provenance'],
        'name_disjoint': True,
        'natural': natural,
        'native_audit': facts,
        'causal_audit': cause is not None,
        'files': {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != 'verified.json'},
    }
    B.common.commit('verified.json', receipt)
    print(f"Verified {len(confirm_rows)} new confirmation rows, {facts['scalars']} native scalars, admission={admitted}", flush=True)


if __name__ == '__main__':
    verify()
