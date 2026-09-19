"""Full prior/temperature grid, selected only on source-equal natural validation."""
import argparse
from collections import Counter
from rdc_formation_readout import *
from phase2747_rdc_training_analysis import source_stat


def variants():
    result = []
    for name in ['native', 'bridge']:
        result.append({'name': name, 'precision': 'native_BF16' if name == 'native' else 'FP32_bridge',
            'receipt': read(OUT/'training/baseline/commits'/(name+'.json'))})
    protocol = read(OUT/'training/protocol.json')
    for seed in protocol['seeds']:
        for condition in protocol['conditions']:
            name = condition+'_'+str(seed)
            r = read(OUT/'training'/name/'result.json')
            for precision, receipt in [('FP32_bridge', r['checkpoints'][-1]['evaluation']), ('native_BF16', r['deployment'])]:
                result.append({'name': name+'_'+precision, 'precision': precision, 'receipt': receipt})
    for r in read(OUT/'radius/result.json')['records']:
        result.append({'name': r['variant']['name'], 'precision': r['variant']['precision'], 'receipt': r['evaluation']})
    assert len(result) == 54 and len({r['name'] for r in result}) == 54
    return result


def validation_weights(rows):
    eligible = [i for i, r in enumerate(rows) if r['split'] == 'validation']
    assert len(eligible) == 192 and all(rows[i]['kind'] == 'natural_content' for i in eligible)
    groups = Counter(rows[i]['source_group'] for i in eligible)
    weights = np.zeros(len(rows))
    for i in eligible:
        weights[i] = 1/len(groups)/groups[rows[i]['source_group']]
    assert abs(weights.sum()-1) < 1e-12
    return weights


def reports(rows, original, calibrated):
    target = np.array([r['target'] for r in rows])
    result = []
    for split in sorted({r['split'] for r in rows}):
        for family in ['all']+sorted({r['family'] for r in rows if r['split'] == split}):
            ix = [i for i, r in enumerate(rows) if r['split'] == split and (family == 'all' or r['family'] == family)]
            rr = [rows[i] for i in ix]
            record = {'split': split, 'family': family,
                'NLL_uncalibrated': source_stat(original['NLL'][ix], rr),
                'NLL_calibrated': source_stat(calibrated['NLL'][ix], rr),
                'NLL_calibrated_minus_uncalibrated': source_stat(calibrated['NLL'][ix]-original['NLL'][ix], rr),
                'calibrated_argmax_correct': source_stat((calibrated['argmax'][ix] == target[ix]).astype(float), rr),
                'uncalibrated_argmax_correct': source_stat((original['argmax'][ix] == target[ix]).astype(float), rr)}
            if all(r['kind'] == 'controlled_relation' for r in rr):
                from collections import defaultdict
                pairs = defaultdict(list)
                for i in ix: pairs[rows[i]['pair_id']].append(i)
                assert all(len(v) == 2 for v in pairs.values())
                record['pairs'] = len(pairs)
                for prefix, packet in [('calibrated', calibrated), ('uncalibrated', original)]:
                    record[prefix+'_both_worlds_full_argmax_correct'] = sum(all(packet['argmax'][i] == target[i] for i in ii) for ii in pairs.values())
                    record[prefix+'_both_worlds_binary_rank_correct'] = sum(all(packet['binary_gold_margin'][i] > 0 for i in ii) for ii in pairs.values())
            result.append(record)
    return result


def run(pilot=False):
    import torch
    folder = OUT/'calibration'
    result_path = folder/('pilot.json' if pilot else 'result.json')
    if result_path.exists(): return
    assert read(OUT/'training/result.json')['all_passed']
    if not pilot: assert read(folder/'pilot.json')['all_passed']
    start = time.monotonic()
    p = read(OUT/'followup/protocol.json')['calibration']
    data = gzread(OUT/'material/rows.json.gz')
    rows = data['validation']+data['diagnostic']+data['fresh']
    with np.load(FIELDS/'material/vocabulary.npz') as z:
        counts = [z['prior_count'].astype(float), z['prior_source_balanced_count']]
    choices = variants() if not pilot else [{'name': 'native', 'precision': 'native_BF16', 'receipt': read(OUT/'training/baseline/commits/native.json')}]
    engine = None
    try:
        engine = Readout()
        pairs = {'en': [engine.token('Yes'), engine.token('No')], 'zh': [engine.token('是'), engine.token('否')]}
        alternatives = []
        for row in rows:
            if row['kind'] == 'controlled_relation':
                pair = pairs[row['language']]
                assert row['target'] in pair
                alternatives.append(next(x for x in pair if x != row['target']))
            else: alternatives.append(row['target'])
        priors = np.stack([(count+beta/engine.V)/(count.sum()+beta) for count in counts for beta in p['total_concentration_beta']])
        prior_parameters = [(kind, beta) for kind in p['priors'] for beta in p['total_concentration_beta']]
        assert np.max(abs(priors.sum(-1)-1)) < 1e-12 and np.min(priors) > 0
        target = np.array([r['target'] for r in rows])
        prior_gold = priors[:, target]
        weights = validation_weights(rows)
        records = []
        for choice in choices:
            out = folder/choice['name']/'result.json'
            if out.exists() and not pilot:
                records.append(read(out)); continue
            tick = time.monotonic()
            source = checked_arrays(choice['receipt'])
            n = 8 if pilot else len(rows)
            # Only target/paired-token probabilities for the entire grid are
            # needed for selection. Every probability uses the full V denominator.
            log_gold = np.zeros((len(p['temperatures']), n))
            log_other = np.zeros_like(log_gold)
            native_margin = np.zeros(n)
            numeric = []
            with torch.no_grad():
                for i in range(n):
                    zz = engine.logits(source['postnorm_BF16'][i])
                    lp = zz.log_softmax(-1)
                    diff = abs(float(-lp[target[i]])-source['NLL'][i])
                    assert diff < 1e-10 and int(zz.argmax()) == source['argmax'][i], (choice['name'], i, diff)
                    numeric.append(diff)
                    native_margin[i] = float(lp[target[i]]-lp[alternatives[i]])
                    for j, temperature in enumerate(p['temperatures']):
                        lt = (zz/temperature).log_softmax(-1)
                        log_gold[j, i] = float(lt[target[i]])
                        log_other[j, i] = float(lt[alternatives[i]])
            if pilot:
                result = {'timestamp': stamp(), 'all_passed': True, 'source': snapshot(__file__),
                    'positions': n, 'temperatures': p['temperatures'], 'full_vocabulary': engine.V,
                    'maximum_B1_NLL_reconstruction_error': max(numeric), 'seconds': time.monotonic()-start,
                    'scope': 'Readout/numerical engineering admission only, no model-effect selection.'}
                save(result_path, result); print('FORMATION_CALIBRATION_PILOT', result, flush=True); return
            grid = []
            for ti, temperature in enumerate(p['temperatures']):
                for pi, (prior_kind, beta) in enumerate(prior_parameters):
                    for alpha in p['mixture_alpha']:
                        ll = -log_gold[ti] if alpha == 0 else -np.logaddexp(np.log1p(-alpha)+log_gold[ti], np.log(alpha)+np.log(prior_gold[pi]))
                        grid.append({'temperature_index': ti, 'prior_index': pi, 'temperature': temperature,
                            'prior': prior_kind, 'beta': beta, 'alpha': alpha, 'validation_NLL': float(weights@ll)})
            selected = min(enumerate(grid), key=lambda item: (item[1]['validation_NLL'], item[0]))[1]
            pi, ti, alpha = selected['prior_index'], selected['temperature_index'], selected['alpha']
            selected_prior = torch.tensor(priors[pi], device='cuda')
            calibrated = {'NLL': [], 'argmax': [], 'entropy': [], 'binary_gold_margin': [],
                'binary_gold_conditional_probability': [], 'KL_uncalibrated_to_calibrated': []}
            with torch.no_grad():
                for i in range(n):
                    zz = engine.logits(source['postnorm_BF16'][i])
                    lp = zz.log_softmax(-1)
                    lt = (zz/selected['temperature']).log_softmax(-1)
                    cp = (1-alpha)*lt.exp()+alpha*selected_prior
                    clp = cp.log()
                    calibrated['NLL'].append(float(-clp[target[i]]))
                    calibrated['argmax'].append(int(cp.argmax()))
                    calibrated['entropy'].append(float(-(cp*clp).sum()))
                    margin = float(clp[target[i]]-clp[alternatives[i]])
                    calibrated['binary_gold_margin'].append(margin)
                    calibrated['binary_gold_conditional_probability'].append(float(torch.sigmoid(clp[target[i]]-clp[alternatives[i]])))
                    calibrated['KL_uncalibrated_to_calibrated'].append(float((lp.exp()*(lp-clp)).sum()))
            calibrated = {k: np.asarray(v) for k, v in calibrated.items()}
            expected = -log_gold[ti] if alpha == 0 else -np.logaddexp(np.log1p(-alpha)+log_gold[ti], np.log(alpha)+np.log(prior_gold[pi]))
            assert np.max(abs(calibrated['NLL']-expected)) < 1e-10
            source['binary_gold_margin'] = native_margin
            receipt = commit_array('calibration', choice['name'], **calibrated,
                all_temperature_log_gold=log_gold, all_temperature_log_other=log_other,
                uncalibrated_binary_gold_margin=native_margin)
            record = {'timestamp': stamp(), 'all_passed': True, 'variant': choice, 'selected': selected,
                'grid': grid, 'grid_candidates': len(grid), 'complete_vocabulary': engine.V,
                'validation_weighting': 'Equal source groups, then equal positions in each group; no test outcomes enter choice.',
                'field': receipt, 'maximum_B1_NLL_reconstruction_error': max(numeric),
                'reports': reports(rows, source, calibrated), 'seconds': time.monotonic()-tick}
            save(out, record); records.append(record)
            save(folder/'progress.json', {'timestamp': stamp(), 'complete': len(records), 'total': 54, 'latest': choice['name']})
            print('FORMATION_CALIBRATED', choice['name'], len(records), selected, round(record['seconds'], 2), flush=True)
        paired = []
        lookup = {r['variant']['name']: r for r in records}
        baselines = {name: checked_arrays(lookup[name]['field']) for name in ['native', 'bridge']}
        for record in records:
            if record['variant']['name'] in baselines: continue
            baseline_name = 'native' if record['variant']['precision'] == 'native_BF16' else 'bridge'
            current = checked_arrays(record['field'])
            baseline = baselines[baseline_name]
            for split in sorted({r['split'] for r in rows}):
                ix = [i for i, r in enumerate(rows) if r['split'] == split]
                paired.append({'variant': record['variant']['name'], 'baseline': baseline_name, 'split': split,
                    'calibrated_NLL_delta': source_stat(current['NLL'][ix]-baseline['NLL'][ix], [rows[i] for i in ix])})
        result = {'timestamp': stamp(), 'source': snapshot(__file__), 'all_passed': True, 'variants': 54,
            'positions_each': len(rows), 'records': records, 'calibrated_paired_comparisons': paired,
            'seconds': time.monotonic()-start,
            'scope': 'Validation-selected scalar prior controls; any retained difference is not a uniquely semantic residual. Gold affects scoring/validation selection only, never full-vocabulary token-choice control.'}
        save(result_path, result); ledger('phase2747_calibration', result['seconds'])
    except Exception as exc:
        failure(folder, start, exc); raise
    finally:
        if engine is not None: engine.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--pilot', action='store_true')
    run(parser.parse_args().pilot)
