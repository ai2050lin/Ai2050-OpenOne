"""Cross-check frozen pair custody and served study from the completed run."""
from collections import Counter
import json
from urllib.request import urlopen

from scipy.stats import beta

from mlp_b34_confirm_v3 import configure
import mlp_b34_confirm_design as D
from shared.dist_protocol.mechanism_trace import digest


def get(path):
    with urlopen('http://127.0.0.1:5032'+path, timeout=45) as response:
        return json.load(response)


def main():
    previous = configure()
    from mlp_b34_confirm_analyze import chunks

    plan = D.checked_plan()
    result = D.READ(D.OUT/'report.json')
    reg = D.READ(D.OUT/'registration.json')
    holdout = D.READ(D.OUT/'holdout-materials.json')['items']
    assert holdout == D.READ(previous/'holdout-materials.json')['items'][1:]
    discovery = D.READ(previous/'discovery-materials.json')['items']
    assert not {name for row in discovery for name in row['entities']} & {name for row in holdout for name in row['entities']}
    assert not {row['frame'] for row in discovery} & {row['frame'] for row in holdout}
    assert len({row['id'] for row in holdout}) == len(holdout) == 767
    cases = Counter()
    max_error = 0.
    for samples, rows, raw in chunks(plan, 'holdout'):
        for sample, row in zip(samples, rows):
            ids_a, ids_b = sample['tokens']
            assert len(ids_a) == len(ids_b) and ids_a[-1] == ids_b[-1] and ids_a != ids_b
            max_error = max(row['noop_error'], max_error)
            assert row['restore_bitwise'] and row['noop_error'] <= .001
            for dose in row['doses']:
                within = dose['norm'] > 0 and abs(dose['norm']-dose['control_norm']) <= .01*dose['norm']
                assert dose['norm_valid'] == within
                cases[dose['dose']] += within
    assert all(cases[row['dose']] == row['switch']['n'] for row in result['dose_curve'])
    for row in result['capability'].values():
        loss, n = row['control_only'], row['n']
        upper = beta.ppf(.975, loss+1, n-loss) if loss < n else 1.
        assert row['loss_upper'] == upper and row['passed'] == (upper <= .02)
    index = get('/api/traces/research-space')
    study = next(s for s in index['studies'] if s['run_id'] == D.RUN)
    assert study['report_sha256'] == reg['report_sha256'] and study['model_snapshot'] == D.SNAPSHOT
    objects = [o for o in index['objects'] if o['body']['run_id'] == D.RUN]
    assert Counter(x['type'] for x in objects) == {'feature':66, 'edge':1}
    served = get('/api/research-library/artifacts/'+reg['report_sha256'])
    assert digest(served) == reg['report_sha256'] and served['provenance'] == plan['provenance']
    assert served['mediation_passed'] == result['mediation_passed'] and served['dose_curve'] == result['dose_curve']
    report = {'provenance': plan['provenance'], 'study_key': study['study_key'], 'report_sha256': reg['report_sha256'],
        'validated_pairs': len(holdout), 'matched_by_dose': dict(cases), 'max_noop_error': max_error,
        'feature_cards': 66, 'relation_records': 1, 'served_report_verified': True}
    D.SAVE(D.OUT/'verification.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
