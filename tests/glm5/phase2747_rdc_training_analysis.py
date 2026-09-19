"""Actual formation summaries and complete local parameter-to-unit calculations."""
from collections import defaultdict
from rdc_formation_common import *
from rdc_relation_native_parameters import parameter, decode

TRAIN = OUT / 'training'
ANALYSIS = OUT / 'training_analysis'


def source_stat(values, rows):
    # Stratify by actual language family, resample source groups; coordinates
    # and two worlds are never treated as separate independent observations.
    by_family = defaultdict(lambda: defaultdict(list))
    for v, r in zip(values, rows):
        by_family[r['family']][r['source_group']].append(float(v))
    groups = {f: np.array([np.mean(v) for g, v in sorted(gg.items())]) for f, gg in sorted(by_family.items())}
    rng = np.random.default_rng(274703)
    samples = np.zeros(2000)
    for g in groups.values():
        samples += g[rng.integers(len(g), size=(2000, len(g)))].mean(1)/len(groups)
    return {'mean': float(np.mean([g.mean() for g in groups.values()])),
        'interval95': np.quantile(samples, [.025, .975]).tolist(), 'expressions': len(rows),
        'source_groups': sum(map(len, groups.values())), 'families': list(groups),
        'weighting': 'Equal present families, then equal source_group, then equal expressions in group.',
        'scope': 'Document/case-bootstrap conditional on these two fixed training seeds, not a seed-population interval.'}


def endpoint_reports(rows, packet, baseline):
    reports = []
    target = np.array([r['target'] for r in rows])
    for split in sorted({r['split'] for r in rows}):
        for family in ['all']+sorted({r['family'] for r in rows if r['split'] == split}):
            ix = [i for i, r in enumerate(rows) if r['split'] == split and (family == 'all' or r['family'] == family)]
            rr = [rows[i] for i in ix]
            value = {'split': split, 'family': family,
                'NLL': source_stat(packet['NLL'][ix], rr),
                'NLL_minus_baseline': source_stat(packet['NLL'][ix]-baseline['NLL'][ix], rr),
                'argmax_correct': source_stat((packet['argmax'][ix] == target[ix]).astype(float), rr),
                'argmax_agreement_with_baseline': source_stat((packet['argmax'][ix] == baseline['argmax'][ix]).astype(float), rr),
                'entropy_minus_baseline': source_stat(packet['entropy'][ix]-baseline['entropy'][ix], rr)}
            if rr[0]['kind'] == 'controlled_relation':
                pairs = defaultdict(list)
                for i in ix:
                    pairs[rows[i]['pair_id']].append(i)
                assert all(len(v) == 2 for v in pairs.values())
                value['both_worlds_first_token_correct'] = sum(all(packet['argmax'][i] == target[i] for i in ii) for ii in pairs.values())
                value['pairs'] = len(pairs)
            reports.append(value)
    return reports


def read_arrays(receipt):
    path = ROOT / receipt['field_path']
    assert sha(path) == receipt['field_sha256']
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def local_parameter_prediction(checkpoint, bridge_fields, actual_fields, originals):
    delta = read_arrays(checkpoint['delta'])
    x = bridge_fields['MLP_input'][:, 16].astype(np.float64)
    assert np.array_equal(actual_fields['hidden'][:, :17].view(np.uint32), bridge_fields['hidden'][:, :17].view(np.uint32))
    assert np.array_equal(actual_fields['MLP_input'][:, :17].view(np.uint32), bridge_fields['MLP_input'][:, :17].view(np.uint32))
    assert np.array_equal(actual_fields['Q'][:, :17].view(np.uint32), bridge_fields['Q'][:, :17].view(np.uint32))
    def weight(name):
        base = originals[name]
        current = (base+delta[name])+delta['reconstruction_residual__'+name]
        return base.astype(np.float64), current.astype(np.float64)
    wg, wg1 = weight('gate_proj.weight')
    g = x @ wg.T
    dg = x @ (wg1-wg).T
    del wg, wg1
    wu, wu1 = weight('up_proj.weight')
    u = x @ wu.T
    du = x @ (wu1-wu).T
    del wu, wu1
    sig = np.exp(-np.logaddexp(0, -g))
    phi = g*sig
    dphi = (sig+g*sig*(1-sig))*dg
    phi1 = (g+dg)*np.exp(-np.logaddexp(0, -(g+dg)))
    a0 = phi*u
    a1 = phi1*(u+du)
    da_linear = u*dphi+phi*du
    wd, wd1 = weight('down_proj.weight')
    m0 = a0 @ wd.T
    dm_finite = a1 @ wd1.T-m0
    dm_linear = da_linear @ wd.T+a0 @ (wd1-wd).T
    arrays = {'predicted_delta_gate': dg.astype(np.float32), 'predicted_delta_up': du.astype(np.float32),
        'predicted_delta_product_finite': (a1-a0).astype(np.float32),
        'predicted_delta_product_linear': da_linear.astype(np.float32),
        'predicted_delta_MLP_write_finite': dm_finite.astype(np.float32),
        'predicted_delta_MLP_write_linear': dm_linear.astype(np.float32)}
    report = {}
    for field, base, pred in [('gate', g, dg), ('up', u, du), ('product', a0, a1-a0), ('MLP_write', m0, dm_finite)]:
        observed = actual_fields[field][:, 16].astype(np.float64)-bridge_fields[field][:, 16].astype(np.float64)
        arrays['observed_delta_'+field] = observed.astype(np.float32)
        mse = ((pred-observed)**2).mean(-1)
        zero = (observed**2).mean(-1)
        report[field] = {'zero_change_MSE_mean': float(zero.mean()), 'full_finite_MSE_mean': float(mse.mean()),
            'same_value_FP64_vs_FP32_bridge_baseline_MSE': float(((base-bridge_fields[field][:, 16])**2).mean()),
            'complete_coordinates_per_expression': observed.shape[-1]}
    report['MLP_write']['linear_parameter_MSE_mean'] = float(((dm_linear-arrays['observed_delta_MLP_write'])**2).mean())
    report['product']['linear_parameter_MSE_mean'] = float(((da_linear-arrays['observed_delta_product'])**2).mean())
    del delta, wd, wd1
    return arrays, report


def main():
    start = time.monotonic()
    data = gzread(OUT / 'material/rows.json.gz')
    rows = data['validation']+data['diagnostic']+data['fresh']
    protocol = read(TRAIN / 'protocol.json')
    if not (TRAIN / 'baseline/commits/bridge_fields.json').exists():
        print('FORMATION_ANALYSIS_WAITING_FOR_BASELINES', flush=True)
        return
    native = read_arrays(read(TRAIN / 'baseline/commits/native.json'))
    bridge = read_arrays(read(TRAIN / 'baseline/commits/bridge.json'))
    field_ids = read(TRAIN / 'evaluation_rows.json')['fields']
    lookup = {r['sample_id']: r for r in rows}
    field_rows = [lookup[i] for i in field_ids]
    initial = read_arrays(read(TRAIN / 'baseline/commits/bridge_fields.json'))
    originals = {n: decode(parameter(ROOT, 'model.layers.16.mlp.'+n, MODELS['qwen4'])).astype(np.float32)
                 for n in ['gate_proj.weight', 'up_proj.weight', 'down_proj.weight']}
    summaries = []
    for seed in protocol['seeds']:
        for condition in protocol['conditions']:
            name = condition+'_'+str(seed)
            progress = read(TRAIN / name / 'progress.json') if (TRAIN / name / 'progress.json').exists() else {}
            for checkpoint in progress.get('checkpoints', []):
                step = checkpoint['step']
                receipt_path = ANALYSIS / name / f'step_{step:03d}.json'
                if receipt_path.exists():
                    result = read(receipt_path)
                    assert result['source']['sha256'] == sha(__file__)
                    summaries.append(result)
                    continue
                packet = read_arrays(checkpoint['evaluation'])
                field = read_arrays(checkpoint['fields'])
                pred, local = local_parameter_prediction(checkpoint, initial, field, originals)
                local_receipt = commit_array('training_analysis/'+name, f'local_{step:03d}', **pred)
                layer_changes = []
                for kind in ['hidden', 'Q', 'gate', 'up', 'product', 'MLP_input', 'MLP_write']:
                    delta = field[kind].astype(np.float64)-initial[kind].astype(np.float64)
                    axes = tuple(range(2, delta.ndim))
                    mse = (delta*delta).mean(axes)
                    for layer in range(mse.shape[1]):
                        layer_changes.append({'field': kind, 'layer': layer, 'raw_complete_coordinate_MSE': source_stat(mse[:, layer], field_rows)})
                    del delta
                result = {'timestamp': stamp(), 'source': snapshot(__file__), 'all_passed': True,
                    'run': name, 'condition': condition, 'seed': seed, 'step': step,
                    'parameter_delta_FP32_L2': checkpoint['delta_FP32_L2'],
                    'endpoint_reports': endpoint_reports(rows, packet, bridge),
                    'complete_layer_changes': layer_changes, 'local_native_parameter_prediction': local,
                    'local_prediction_fields': local_receipt, 'source_checkpoint': checkpoint,
                    'checks': 'AllH0..H16, allQ0..Q16 and allMLPinputs0..16 bit invariant against same bridge; fullparameter reconstruction preserves every bit.',
                    'scope': 'Local full-coordinate parameter arithmetic is a known architecture calculation; not independently learned language extraction. Post-checkpoint naturalNLL is authentic fixed-prefix scoring, not ownhistory competence.'}
                save(receipt_path, result)
                summaries.append(result)
                print('FORMATION_ANALYZED', name, step, local['MLP_write']['full_finite_MSE_mean'], flush=True)
                del packet, field, pred
            deployed_path = TRAIN / name / 'result.json'
            if deployed_path.exists():
                run = read(deployed_path)
                receipt_path = ANALYSIS / name / 'deployed_BF16.json'
                if receipt_path.exists():
                    summaries.append(read(receipt_path))
                else:
                    packet = read_arrays(run['deployment'])
                    result = {'timestamp': stamp(), 'source': snapshot(__file__), 'all_passed': True,
                        'run': name, 'condition': condition, 'seed': seed, 'step': 'deployed_BF16',
                        'actual_parameter_delta_BF16_L2': run['deployed_BF16_delta_L2'],
                        'endpoint_reports': endpoint_reports(rows, packet, native),
                        'scope': 'NativeBF16deployment versus originalnativeBF16, not the FP32bridge baseline.'}
                    save(receipt_path, result)
                    summaries.append(result)
    final = len(summaries) == 30 and (TRAIN / 'result.json').exists()
    result = {'timestamp': stamp(), 'source': snapshot(__file__), 'all_passed': final,
        'execution_state': 'complete' if final else 'partial_committed_checkpoints_only',
        'analyzed_checkpoints_or_deployments': len(summaries), 'planned_checkpoints_or_deployments': 30,
        'baseline_bridge_minus_native': endpoint_reports(rows, bridge, native),
        'summary': [{k: r[k] for k in ['run', 'condition', 'seed', 'step', 'endpoint_reports']} for r in summaries],
        'source_files': [str(p.relative_to(BASE)) for p in sorted(ANALYSIS.glob('*/*.json'))],
        'seconds': time.monotonic()-start,
        'boundary': 'No inference of unique semantic basis from raw NLL or coordinate change; matched radii, calibration and ownhistory remain distinct tasks.'}
    save(ANALYSIS / ('result.json' if final else 'progress.json'), result)
    print('FORMATION_ANALYSIS_STATE', result['execution_state'], len(summaries), result['seconds'], flush=True)


if __name__ == '__main__':
    main()
