"""Frozen full-coordinate mappings read through all V tokens; no gold bias choice."""
import argparse
from collections import defaultdict
from rdc_formation_readout import *
from phase2747_rdc_transfer_prepare import NAMES

METRICS = ['KL_target_to_prediction', 'KL_prediction_to_target', 'entropy', 'argmax',
    'target_argmax_agreement', 'gold_digit_NLL', 'digit_mass', 'letter_mass',
    'gold_conditional_digit_probability', 'gold_digit_margin', 'digit_rank_argmax',
    'complete_vocabulary_argmax_gold_correct']
BIASES = ['none', 'uniform_digit_plus8', 'uniform_letter_plus8']


def bias_metrics(lp, lq, digit, letter, gold, bias_ids=None):
    import torch
    p, q = lp.exp(), lq.exp()
    kl = (p*(lp-lq)).sum()
    reverse = (q*(lq-lp)).sum()
    entropy = -(q*lq).sum()
    dmass, lmass = q[digit].sum(), q[letter].sum()
    predicted = int(lq.argmax())
    gold_nll = -lq[gold]
    digit_probability = (lq[gold]-torch.logsumexp(lq[digit],dim=0)).exp()
    other_digits = [v for v in digit if v != gold]
    margin = lq[gold]-lq[other_digits].max()
    if bias_ids is not None:
        m = q[bias_ids].sum()
        pm = p[bias_ids].sum()
        multiplier = float(np.exp(8.))
        z = 1+(multiplier-1)*m
        logz = z.log()
        kl = kl-8*pm+logz
        reverse = (reverse+(multiplier-1)*(q[bias_ids]*(lq[bias_ids]-lp[bias_ids])).sum()+8*multiplier*m)/z-logz
        entropy = (entropy-(multiplier-1)*(q[bias_ids]*lq[bias_ids]).sum()-8*multiplier*m)/z+logz
        best_bias_position = int(lq[bias_ids].argmax())
        best_bias_token = bias_ids[best_bias_position]
        if float(lq[best_bias_token]+8) > float(lq[predicted]): predicted = best_bias_token
        # In a rare exact tie, argmax uses the first vocabulary index.
        elif float(lq[best_bias_token]+8) == float(lq[predicted]): predicted = min(predicted, best_bias_token)
        gold_nll = gold_nll-(8 if gold in bias_ids else 0)+logz
        dmass = dmass*(multiplier if bias_ids == digit else 1)/z
        lmass = lmass*(multiplier if bias_ids == letter else 1)/z
    return np.array([float(kl), float(reverse), float(entropy), predicted,
        int(predicted == int(lp.argmax())), float(gold_nll), float(dmass), float(lmass),
        float(digit_probability), float(margin), digit[int(lq[digit].argmax())], int(predicted == gold)])


def direct_check(lp, lq, bias, digit, letter, gold, analytic):
    import torch
    changed = lq.clone(); changed[bias] += 8
    changed = changed.log_softmax(-1)
    direct = bias_metrics(lp, changed, digit, letter, gold)
    error = np.max(abs(direct-analytic))
    assert error < 1e-9, error
    assert abs(float(changed[gold]-changed[[v for v in digit if v != gold]].max())-analytic[9]) < 1e-10
    return float(error)


def summarize(records, probes, answer_index):
    values = []
    for record in records:
        arrays = checked_arrays(record['readout'])
        for split in sorted({p['split'] for p in probes})+['fixed_Answer_query']:
            ix = [answer_index] if split == 'fixed_Answer_query' else [i for i, p in enumerate(probes) if p['split'] == split]
            for ci, candidate in enumerate(NAMES):
                for bi, bias in enumerate(BIASES):
                    metrics = arrays['metrics'][ci, bi, ix].mean(0)
                    values.append({'direction': record['direction'], 'split': record['split'],
                        'source_group': record['source_group'], 'query_split': split, 'candidate': candidate, 'bias': bias,
                        **{name: float(metrics[i]) for i, name in enumerate(METRICS) if name not in ['argmax', 'digit_rank_argmax']}})
    compressed(OUT/'transfer/full_vocabulary_group_metrics.json.gz', values)
    buckets = defaultdict(list)
    for v in values: buckets[tuple(v[k] for k in ['direction', 'split', 'query_split', 'candidate', 'bias'])].append(v)
    summary = []
    for key, vv in sorted(buckets.items()):
        entry = dict(zip(['direction', 'split', 'query_split', 'candidate', 'bias'], key))
        entry['semantic_groups'] = len(vv)
        entry['metrics'] = {m: clustered([v[m] for v in vv], [v['source_group'] for v in vv])
            for m in ['KL_target_to_prediction', 'KL_prediction_to_target', 'target_argmax_agreement',
                'gold_digit_NLL', 'digit_mass', 'gold_conditional_digit_probability',
                'gold_digit_margin', 'complete_vocabulary_argmax_gold_correct']}
        summary.append(entry)
    # Paired comparisons keep one semantic group as the uncertainty unit.
    lookup = {tuple(v[k] for k in ['direction', 'split', 'query_split', 'source_group', 'candidate', 'bias']): v for v in values}
    paired = defaultdict(list)
    for v in values:
        if v['candidate'] == 'query_conditioned' and v['bias'] == 'none':
            key = tuple(v[k] for k in ['direction', 'split', 'query_split', 'source_group'])
            for candidate in ['identity', 'query_only', 'affine', 'shuffled_pair']:
                control = lookup[key+(candidate, 'none')]
                paired[key[:3]+(candidate,)].append((v['source_group'], control['KL_target_to_prediction']-v['KL_target_to_prediction']))
    return summary, [{'direction': key[0], 'split': key[1], 'query_split': key[2], 'control': key[3],
        'control_minus_mapped_KL': clustered([v[1] for v in vv], [v[0] for v in vv])} for key, vv in sorted(paired.items())]


def run(pilot=False):
    import torch
    folder = OUT/'transfer'
    result_path = folder/('readout_pilot.json' if pilot else 'readout_result.json')
    if result_path.exists(): return
    if not pilot: assert read(folder/'readout_pilot.json')['all_passed']
    assert read(folder/'preparation.json')['all_passed']
    start = time.monotonic(); engine = None
    try:
        engine = Readout()
        digit = [engine.token(str(i)) for i in range(1, 9)]
        letter = [engine.token(c) for c in 'abcdefgh']
        assert len(set(digit+letter)) == 16
        probes = read(OLD/'probes/protocol.json')['probes']
        answer = next(i for i, p in enumerate(probes) if p['text'] == '\nAnswer:')
        preparation = gzread(folder/'records.json.gz')
        selected = preparation[:2] if pilot else preparation
        if not pilot:
            immutable(folder/'readout_protocol.json', {'timestamp': stamp(), 'source': snapshot(__file__),
                'readout_source': snapshot(Path(__file__).with_name('rdc_formation_readout.py')),
                'digit_tokens': digit, 'letter_tokens': letter, 'uniform_bias': 8., 'answer_query': answer,
                'reference': 'Same stored original target BF16postnorm, new B1 originalBF16 head readout.',
                'old_shape_comparison': 'Old QueryEngine used groups up to16; report resulting B1 vs old endpoint floor explicitly.',
                'no_gold': 'Bias sets are all digits1..8 or all lettersa..h, never chosen from correct answer. Gold used only for scores.',
                'scope': 'All100 query KLs compare actual query response. Correct program digit NLL is a diagnostic target; only fixed Answer query is reported as answer-oriented accuracy.'})
        records = []; numeric = []
        with torch.no_grad():
            for record in selected:
                name = rank(record['direction']+'/'+record['source_group'])[:24]
                out = folder/'readout_records'/record['direction']/(name+'.json')
                if out.exists() and not pilot:
                    records.append(read(out)); continue
                tick = time.monotonic()
                arrays = checked_arrays(record['field'])
                query_indices = sorted({0, answer}) if pilot else list(range(100))
                mm = np.zeros((5, 3, len(query_indices), len(METRICS)))
                baseline = []; floor = []
                gold = engine.token(record['target_material']['target'])
                assert gold in digit
                for qi, query in enumerate(query_indices):
                    lp = engine.logits(arrays['target_postnorm_BF16'][query]).log_softmax(-1)
                    base_entropy = float(-(lp.exp()*lp).sum())
                    baseline.append([base_entropy, int(lp.argmax()), float(-lp[gold]), float(lp[digit].exp().sum())])
                    old = arrays['original_readout_statistics'][query]
                    floor.append([base_entropy-float(old[0]), int(lp.argmax()) != int(old[3])])
                    for ci, candidate in enumerate(NAMES):
                        lq = engine.logits(arrays['predictions'][ci, query]).log_softmax(-1)
                        for bi, bias in enumerate([None, digit, letter]):
                            mm[ci, bi, qi] = bias_metrics(lp, lq, digit, letter, gold, bias)
                            if pilot and bias is not None:
                                numeric.append(direct_check(lp, lq, bias, digit, letter, gold, mm[ci, bi, qi]))
                        assert np.array_equal(mm[ci, :, qi, 8], np.repeat(mm[ci, 0, qi, 8], 3))
                        assert np.array_equal(mm[ci, :, qi, 9], np.repeat(mm[ci, 0, qi, 9], 3))
                if pilot: continue
                receipt = commit_array('transfer_readout/'+record['direction'], name,
                    metrics=mm, baseline=np.array(baseline), old_batch_vs_new_B1=np.array(floor))
                result = {'timestamp': stamp(), 'all_passed': True, 'direction': record['direction'],
                    'source_group': record['source_group'], 'split': record['split'], 'readout': receipt,
                    'source_prediction': record['field'], 'metrics': METRICS, 'candidates': NAMES, 'biases': BIASES,
                    'old_B1_argmax_mismatch_queries': int(np.array(floor)[:, 1].sum()),
                    'old_B1_entropy_difference_max_abs': float(np.max(abs(np.array(floor)[:, 0]))),
                    'seconds': time.monotonic()-tick}
                save(out, result); records.append(result)
                save(folder/'readout_progress.json', {'timestamp': stamp(), 'completed': len(records), 'total': 256, 'latest': name})
                if len(records) % 8 == 0: print('FORMATION_TRANSFER_READOUT', len(records), 256, round(time.monotonic()-start, 2), flush=True)
        if pilot:
            result = {'timestamp': stamp(), 'all_passed': True, 'source': snapshot(__file__), 'semantic_direction_records': 2,
                'query_indices': query_indices, 'complete_vocabulary': engine.V, 'analytic_vs_direct_checks': len(numeric),
                'maximum_full_distribution_bias_identity_error': max(numeric), 'digit_tokens': digit, 'letter_tokens': letter,
                'seconds': time.monotonic()-start, 'scope': 'Numerical admission, not an outcome selection.'}
        else:
            summary, paired = summarize(records, probes, answer)
            result = {'timestamp': stamp(), 'all_passed': True, 'source': snapshot(__file__), 'records': len(records),
                'readouts': 256*100*5, 'bias_conditions': 3, 'complete_vocabulary': engine.V,
                'summary': summary, 'paired_mapping_comparisons': paired, 'metrics': METRICS,
                'old_B1_argmax_mismatches': sum(r['old_B1_argmax_mismatch_queries'] for r in records),
                'seconds': time.monotonic()-start, 'scope': 'FullV directed mapping comparison, no invertibility/isomorphism or unique semantic basis established. Uniform class bias preserves conditional digit ranking by construction.'}
        save(result_path, result); ledger('phase2747_transfer_readout_'+('pilot' if pilot else 'main'), result['seconds'])
        print('FORMATION_TRANSFER_READOUT_DONE', pilot, result['seconds'], flush=True)
    except Exception as exc:
        failure(folder, start, exc); raise
    finally:
        if engine is not None: engine.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--pilot', action='store_true')
    run(parser.parse_args().pilot)
