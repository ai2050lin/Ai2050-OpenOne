"""Independent completed-protocol arithmetic audit, no model import or new Phase."""
import argparse, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import RESULT, read, save, sha
OUT = RESULT/'phase2683_crossmodel_function_atlas'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('protocol', choices=('qwen14', 'glm4', 'ds7', 'ds7_answer'))
    key = parser.parse_args().protocol
    folder = OUT/key; complete = read(folder/'analysis/completion.json')
    frozen = read(OUT/'protocol/frozen.json')['models'][key]
    material = read(folder/'material/cases.json'); records = read(folder/'analysis/records.json')
    checks = {'512_complete':complete['cases'] == len(records) == len(material) == 512,
              'frozen_material':sha(folder/'material/cases.json') == frozen['material_sha256'],
              'frozen_calibration':sha(folder/'material/calibration.json') == frozen['calibration_sha256'],
              'records_bound_to_actual_material':all(
                  r['case_index'] == i and r['source_case_index'] == material[i]['source_case_index']
                  and r['case_id'] == material[i]['case_id'] for i, r in enumerate(records))}
    expected_groups = {r['family']+'_'+r['language'] for r in material}
    paths = sorted((folder/'maps').glob('counts_*.npz'))
    checks['all16_families_languages'] = {p.stem.removeprefix('counts_') for p in paths} == expected_groups and len(paths) == 16
    sums = {}; family_task = {}
    for path in paths:
        with np.load(path) as z:
            totals = {}
            for metric, shape in (('h', (frozen['layers']+1, 2, frozen['hidden'])),
                                  ('a', (frozen['layers'], 2, frozen['mlp_units']))):
                positive = z[metric+'__all4_positive']; negative = z[metric+'__all4_negative']
                same = z[metric+'__all4_same_nonzero']
                assert np.array_equal(positive+negative, same)
                assert np.all(same.astype(np.int32)+z[metric+'__any_zero']+z[metric+'__opposed'] >= 4)
                totals[metric] = int((same[:,1] == 4).sum())
                for name in ('all4_positive', 'all4_negative', 'all4_same_nonzero', 'any_zero', 'opposed'):
                    label = metric+'__'+name; value = z[label]
                    assert value.shape == shape and np.issubdtype(value.dtype, np.integer)
                    assert (value >= 0).all() and (value <= 4).all()
                    if label not in sums: sums[label] = np.zeros(shape, dtype=np.uint16)
                    sums[label] += value
            family_task[path.stem.removeprefix('counts_')] = totals
    with np.load(folder/'maps/global_counts.npz') as z:
        checks['every_global_coordinate_equals16_family_sum'] = set(z.files) == set(sums) and all(np.array_equal(z[k], v) for k, v in sums.items())
    with np.load(folder/'maps/global_sums.npz') as z:
        chunks = z['completed_chunks'].tolist()
        checks['all16_chunks_committed_once'] = set(chunks) == expected_groups and len(chunks) == 16
        for metric in ('h', 'a'):
            lo = z[metric+'__min_abs_delta_sum']; hi = z[metric+'__max_abs_delta_sum']
            assert lo.shape == hi.shape == sums[metric+'__all4_same_nonzero'].shape
            assert np.isfinite(hi).all() and (lo >= 0).all() and (lo <= hi).all()
        for name in z.files:
            if name.startswith('moment__'): assert np.isfinite(z[name]).all()
    checks['full_coordinate_counts_amplitudes_finite'] = True
    behavior = {}
    for record in records:
        group = record['language']+'/'+record['output_function']
        row = behavior.setdefault(group, dict(n=0, content_correct=0, strict_correct=0, eos=0, final_available=0))
        row['n'] += 1
        for name in ('content_correct', 'strict_correct', 'eos'): row[name] += int(record[name])
        row['final_available'] += int(record['final_answer_available'])
    checks['eight_behavior_cells64_each'] = len(behavior) == 8 and all(r['n'] == 64 for r in behavior.values())
    checks['final_available_aggregation'] = sum(r['final_available'] for r in behavior.values()) == complete['final_answer_available']
    checks['shape_difference_aggregation'] = sum(r['unpadded_prefill_state_max_difference'] > 0 for r in records) == complete['unpadded_prefill_changed']
    full_gate_addresses = {}
    for metric in ('h', 'a'):
        addresses = np.argwhere(sums[metric+'__all4_same_nonzero'][:,1] == 64)
        full_gate_addresses[metric] = [{'layer_or_checkpoint':int(l), 'coordinate':int(j),
            'positive_base_groups':int(sums[metric+'__all4_positive'][l,1,j]),
            'negative_base_groups':int(sums[metric+'__all4_negative'][l,1,j])} for l, j in addresses]
    report = {'protocol':key, 'all_checks_passed':all(checks.values()), 'checks':checks,
              'global64_task_same_nonzero_coordinates':{m:int((sums[m+'__all4_same_nonzero'][:,1] == 64).sum()) for m in ('h','a')},
              'family4_task_same_nonzero_coordinates':family_task, 'behavior':behavior,
              'all_global64_task_gate_addresses':full_gate_addresses,
              'execution_shape_difference':{'changed':complete['unpadded_prefill_changed'],
                                           'maximum':max(r['unpadded_prefill_state_max_difference'] for r in records)},
              'boundary':'All native coordinates audited without TopK, dimensional projection or new model forwards. Direction counts are finite conditional patterns, not semantic specificity; body-prefix controls are not semantic abstraction. Padded field and natural generation are distinct numerical protocols. This is one completed protocol, not completion of Phase2683 or the whole campaign.'}
    save(folder/'analysis/independent_protocol_audit.json', report)
    print(report, flush=True)
    assert report['all_checks_passed']


if __name__ == '__main__': main()
