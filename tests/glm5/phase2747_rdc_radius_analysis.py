"""Source-paired radius results and exact full-gradient/direction compatibility."""
from rdc_formation_common import *
from rdc_formation_direction import NAMES, exact_direction
from rdc_formation_readout import checked_arrays
from rdc_relation_native_parameters import parameter, decode
from phase2747_rdc_training_analysis import endpoint_reports, source_stat


def main():
    import torch
    folder = OUT/'radius_analysis'; finish = folder/'result.json'
    if finish.exists(): return
    start = time.monotonic()
    radius = read(OUT/'radius/result.json'); gradient = read(OUT/'gradient/result.json')
    assert radius['all_passed'] and gradient['all_passed']
    data = gzread(OUT/'material/rows.json.gz'); rows = data['validation']+data['diagnostic']+data['fresh']
    bases = {name: checked_arrays(read(OUT/'training/baseline/commits'/(name+'.json'))) for name in ['native', 'bridge']}
    original = {n: decode(parameter(ROOT, 'model.layers.16.mlp.'+n, MODELS['qwen4'])) for n in NAMES}
    gradients = [checked_arrays(record['field']) for record in gradient['records']]
    records = []; directions = {}; last_key = None
    for rr in radius['records']:
        v = rr['variant']; name = v['name']; path = folder/'records'/(name+'.json')
        if path.exists(): records.append(read(path)); continue
        key = v['run'], v['transform']
        if key != last_key:
            directions, receipt = exact_direction(v['run'], original)
            rng = np.random.default_rng(v.get('permutation_seed', 0))
            for n in NAMES:
                if v['transform'] == 'reverse': directions[n] = -directions[n]
                elif v['transform'] == 'coordinate_shuffle':
                    order = rng.permutation(directions[n].size)
                    directions[n] = directions[n].ravel()[order].reshape(directions[n].shape)
            last_key = key
        scale = rr['matching']['scale']; dots = np.zeros(3); squared_norm = 0.; hashes = {}
        for n in NAMES:
            actual = (original[n].astype(np.float64)+scale*directions[n]).astype(np.float32)
            if v['precision'] == 'native_BF16': actual = torch.from_numpy(actual).bfloat16().float().numpy()
            change = actual.astype(np.float64)-original[n].astype(np.float64)
            hashes[n] = identity(change)
            squared_norm += np.sum(change*change)
            for i, gg in enumerate(gradients): dots[i] += np.sum(change*gg[n].astype(np.float64))
        norm = float(np.sqrt(squared_norm))
        assert abs(norm-rr['matching']['actual_radius']) <= 1e-11, (name, norm, rr['matching']['actual_radius'])
        packet = checked_arrays(rr['evaluation']); baseline_name = 'native' if v['precision'] == 'native_BF16' else 'bridge'
        comparison = endpoint_reports(rows, packet, bases[baseline_name])
        value = {'timestamp': stamp(), 'all_passed': True, 'variant': v, 'source_radius_record': rr,
            'CPU_full_parameter_reconstructed_radius': norm, 'direction_identity': hashes,
            'complete_gradient_dot_actual_delta': {record['condition']: float(dots[i]) for i, record in enumerate(gradient['records'])},
            'complete_gradient_cos_actual_delta': {record['condition']: float(dots[i]/norm/record['all_parameter_gradient_norm']) for i, record in enumerate(gradient['records'])},
            'gradient_scope': 'Initial256training subset at FP32bridge; BF16endpoint pairing is vector geometry, not an exact derivative through rounding.',
            'baseline': baseline_name, 'endpoint_reports': comparison}
        save(path, value); records.append(value)
        print('FORMATION_RADIUS_ANALYZED', len(records), name, flush=True)
    paired = []; lookup = {r['variant']['name']: r for r in records}
    for record in records:
        v = record['variant']
        if v['condition'] == 'true_token' and v['transform'] == 'identity': continue
        reference_name = 'true_token_'+str(v['seed'])+'_'+v['precision']+'_r'+str(v['radius_factor']).replace('.', 'p')
        control = checked_arrays(record['source_radius_record']['evaluation'])
        true = checked_arrays(lookup[reference_name]['source_radius_record']['evaluation'])
        for split in sorted({r['split'] for r in rows}):
            ix = [i for i, r in enumerate(rows) if r['split'] == split]
            paired.append({'variant': v['name'], 'true_reference': reference_name, 'split': split,
                'control_minus_true_NLL': source_stat(control['NLL'][ix]-true['NLL'][ix], [rows[i] for i in ix])})
    result = {'timestamp': stamp(), 'source': snapshot(__file__), 'all_passed': True, 'variants': len(records),
        'records': records, 'same_radius_paired_comparisons': paired, 'seconds': time.monotonic()-start,
        'scope': 'All complete-coordinate directions and actual precision/radii, no outcome-selected displacement. Angle and NLL are not semantic-module identities.'}
    save(finish, result); ledger('phase2747_radius_analysis', result['seconds'])


if __name__ == '__main__': main()
