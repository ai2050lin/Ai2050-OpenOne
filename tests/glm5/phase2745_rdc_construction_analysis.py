"""Full-coordinate conditional ANOVA and paired native construction audit.

ANOVA is a descriptive balanced-grid decomposition, not a semantic factorization.
Its interaction is recoverable exactly from raw states and persisted margins.
"""
import argparse
from phase2745_rdc_construction_contract import freeze
from rdc_construction_common import *


def occurrence_alignment(a, b):
    from collections import defaultdict
    places, seen = defaultdict(list), defaultdict(int)
    for i, token in enumerate(a):
        places[token].append(i)
    mapping = []
    for token in b:
        mapping.append(places[token][seen[token]])
        seen[token] += 1
    assert sorted(mapping) == list(range(len(a)))
    # B coordinates mapped into A occurrence positions.
    return np.argsort(mapping)


def preflight():
    rng = np.random.default_rng(2745)
    y = rng.normal(size=(7, 6, 11))
    mu, ap, bq = y.mean((0, 1)), y.mean(1), y.mean(0)
    interaction = y-ap[:, None]-bq[None]+mu
    total = np.mean(y*y, (0, 1))-mu**2
    prefix = np.mean(ap*ap, 0)-mu**2
    query = np.mean((bq-mu)**2, 0)
    error = float(abs(total-prefix-query-(interaction**2).mean((0, 1))).max())
    assert error < 1e-12
    tokens_a, tokens_b = [1, 3, 1, 2], [2, 1, 3, 1]
    alignment = occurrence_alignment(tokens_a, tokens_b)
    assert np.array_equal(np.array(tokens_b)[alignment], tokens_a)
    return {'synthetic': True, 'energy_identity_max_error': error,
            'repeated_token_occurrence_alignment_checked': True, 'all_passed': True}


def analyse_pairs(key, rows, probes, native, out):
    records, trajectories = [], []
    for begin in range(0, len(rows), 2):
        a, b = rows[begin:begin+2]
        cp = [read(BASE / 'capture' / key / 'commits' / (r['sample_id']+'.json')) for r in [a, b]]
        pf = BASE / 'capture' / key / 'pairs' / ('pair_'+rank(a['pair_id'])[:20]+'.npz')
        with np.load(pf) as z:
            trajectories.append(z['all_query_coordinate_mean_squared_difference'])
        r = {n: a[n] for n in ['pair_id', 'source_group', 'family', 'language', 'case', 'split']}
        assert a['truth'] != b['truth']
        raw_separation = cp[0]['actual_current']['conditional_yes_probability']-cp[1]['actual_current']['conditional_yes_probability']
        r['unoriented_world_A_minus_B_yes_probability'] = raw_separation
        r['world_A_truth'], r['world_B_truth'] = a['truth'], b['truth']
        r['answer_direction_separation'] = raw_separation*(1 if a['truth'] else -1)
        r['B1_both_argmax_correct'] = all(c['actual_current']['argmax_correct'] for c in cp)
        r['world_scores'] = [c['actual_current'] for c in cp]
        r['query_construction'] = []
        alignment = occurrence_alignment(a['prompt_ids'], b['prompt_ids'])
        with np.load(BASE / 'capture' / key / 'fields' / (a['sample_id']+'.npz')) as za, np.load(
             BASE / 'capture' / key / 'fields' / (b['sample_id']+'.npz')) as zb:
            for layer in [0, native['early']]:
                per_query = []
                for q, probe in enumerate(probes):
                    prefix = f'p{q}_L{layer}_'
                    qa, qb = unbits(za[prefix+'q_before_rope']).astype(float), unbits(zb[prefix+'q_before_rope']).astype(float)
                    aa, ab = unbits(za[prefix+'attention']).astype(float), unbits(zb[prefix+'attention']).astype(float)
                    norm_a = np.sqrt(np.mean(qa*qa, -1, keepdims=True)).clip(1e-12)
                    norm_b = np.sqrt(np.mean(qb*qb, -1, keepdims=True)).clip(1e-12)
                    perm = np.r_[alignment, np.arange(len(alignment), ab.shape[-1])]
                    per_query.append({'query': q, 'query_language': probe['language'],
                        'Q_MSE': float(np.mean((qb-qa)**2)), 'Q_head_RMS_normalized_MSE': float(np.mean((qb/norm_b-qa/norm_a)**2)),
                        'Q_bit_equal': bool(np.array_equal(za[prefix+'q_before_rope'], zb[prefix+'q_before_rope'])),
                        'attention_position_aligned_MSE': float(np.mean((ab-aa)**2)),
                        'attention_token_occurrence_aligned_MSE': float(np.mean((ab[:, perm]-aa)**2)),
                        'native_attention_sum_max_error': float(max(abs(aa.sum(-1)-1).max(), abs(ab.sum(-1)-1).max()))})
                r['query_construction'].append({'block': layer, 'queries': per_query})
        records.append(r)
        if (begin+2)%32 == 0:
            print('CONSTRUCTION_PAIR_ANALYSIS', key, begin+2, len(rows), flush=True)
    path = out / 'all_pair_layer_query_MSE.npz'
    if path.exists():
        with np.load(path) as old:
            assert np.array_equal(old['pair_layer_query_MSE'], np.stack(trajectories))
    else:
        npz(path, pair_layer_query_MSE=np.stack(trajectories))
    compressed(out / 'paired_observations_truth_oriented.json.gz', records)
    return summarize_pairs(records)


def summarize_pairs(records):
    summaries = []
    for family in ['all']+sorted({r['family'] for r in records}):
        rr = [r for r in records if family == 'all' or r['family'] == family]
        groups = [r['source_group'] for r in rr]
        first = [q for r in rr for q in r['query_construction'][0]['queries']]
        early = [q for r in rr for q in r['query_construction'][1]['queries']]
        summaries.append({'family': family, 'pairs': len(rr), 'semantic_groups': len(set(groups)),
            'answer_direction_separation': clustered([r['answer_direction_separation'] for r in rr], groups),
            'B1_both_argmax_correct': sum(r['B1_both_argmax_correct'] for r in rr),
            'first_block_Q_bit_equal': sum(q['Q_bit_equal'] for q in first),
            'first_block_pair_query_count': len(first),
            'early_Q_MSE_mean': float(np.mean([q['Q_MSE'] for q in early])),
            'early_Q_normalized_MSE_mean': float(np.mean([q['Q_head_RMS_normalized_MSE'] for q in early])),
            'scope': 'B1 first position and fixed-query diagnostics; not free-generation correctness, semantic layer localization, or causal attribution.'})
    return summaries


def main(key, refresh_pairs=False):
    out = BASE / 'analysis' / key
    if (out / 'result.json').exists():
        prior = read(out / 'result.json')
        assert prior['all_passed']
        if not refresh_pairs:
            return
        history = out/'history'/('pair_orientation_'+str(time.time_ns()))
        history.mkdir(parents=True, exist_ok=True)
        for name in ['result.json', 'paired_observations.json.gz']:
            shutil.copyfile(out/name, history/name)
        _, material = freeze()
        native = read(BASE/'capture'/key/'result.json')
        scope = material['models'][key]
        start = time.monotonic()
        records = gzread(out/'paired_observations.json.gz')
        pair_rows = {r['pair_id']: r for r in scope['rows'] if r['world'] == 0}
        for record in records:
            a = pair_rows[record['pair_id']]
            raw = record['world_scores'][0]['conditional_yes_probability']-record['world_scores'][1]['conditional_yes_probability']
            record.update(unoriented_world_A_minus_B_yes_probability=raw,
                world_A_truth=a['truth'], world_B_truth=not a['truth'],
                answer_direction_separation=raw*(1 if a['truth'] else -1))
        compressed(out/'paired_observations_truth_oriented.json.gz', records)
        summary = summarize_pairs(records)
        prior.update(timestamp=stamp(), native_relation_summary=summary,
            pair_analysis_source=snapshot(__file__),
            paired_observations_file='paired_observations_truth_oriented.json.gz',
            pair_orientation_correction={'reason': 'World A is not always the affirmative world. Orient the probability difference by actual external truth, rather than assuming A=Yes. Original native fields, pair trajectories and ANOVA are unchanged.',
                'prior_reports_preserved': str(history.relative_to(BASE)),
                'formula': '(2*truth_A-1)*(Pyes_A-Pyes_B)',
                'scope': 'Corrects reported answer-direction summaries; both-argmax-correct counts and all paired state prediction inputs/targets are unchanged.'})
        save(out/'result.json', prior)
        ledger('construction_pair_orientation_correction_'+key, time.monotonic()-start)
        print('CONSTRUCTION_PAIR_ORIENTATION_CORRECTED', key, flush=True)
        return
    start = time.monotonic()
    numerical_preflight = preflight()
    _, material = freeze()
    native = read(BASE / 'capture' / key / 'result.json')
    assert native['all_passed']
    rows, probes = material['models'][key]['rows'], material['models'][key]['probes']
    boundaries = native['selected_query_boundaries']+['postnorm']
    width, P, Q, L = native['width'], len(rows), len(probes), len(boundaries)
    languages = ['all', 'en', 'zh']
    qi = [np.arange(Q)]+[np.array([q for q, p in enumerate(probes) if p['language'] == lang]) for lang in languages[1:]]
    # Bounded margins: no PQD full grid nor giant interaction tensor is materialized.
    query_sums = np.zeros((2, L, Q, width))
    row_means = np.zeros((2, P, L, width))
    squares = np.zeros((2, 3, L, width))
    row_mean_squares = np.zeros_like(squares)
    for p, row in enumerate(rows):
        file = BASE / 'capture' / key / 'fields' / (row['sample_id']+'.npz')
        with np.load(file) as z:
            ix = z['query_layer_indices'].tolist()
            state = z['query_selected_states']
            for ell, boundary in enumerate(boundaries):
                raw = unbits(z['postnorm'] if boundary == 'postnorm' else state[ix.index(boundary)]).astype(float)
                for v, y in enumerate([raw, raw/np.sqrt(np.mean(raw*raw, -1, keepdims=True)).clip(1e-12)]):
                    query_sums[v, ell] += y
                    row_means[v, p, ell] = y.mean(0)
                    for g, qq in enumerate(qi):
                        squares[v, g, ell] += (y[qq]**2).sum(0)
                        row_mean_squares[v, g, ell] += y[qq].mean(0)**2
        if (p+1)%32 == 0:
            print('CONSTRUCTION_ANOVA', key, p+1, P, flush=True)
    query_means = query_sums/P
    del query_sums
    summaries, checks = [], []
    for ell, boundary in enumerate(boundaries):
        mu = query_means[:, ell].mean(1)
        pa = row_means[:, :, ell]-mu[:, None]
        qb = query_means[:, ell]-mu[:, None]
        energy = np.empty((2, 3, 4, width))
        for v, view in enumerate(['raw', 'per_vector_RMS']):
          for g, qq in enumerate(qi):
            mug = query_means[v, ell, qq].mean(0)
            total = squares[v, g, ell]/(P*len(qq))-mug**2
            prefix = row_mean_squares[v, g, ell]/P-mug**2
            query = np.mean((query_means[v, ell, qq]-mug)**2, 0)
            interaction = total-prefix-query
            error = float(min(total.min(), prefix.min(), interaction.min()))
            tolerance = max(1e-10, float(abs(total).max())*1e-9)
            assert error > -tolerance, ('ANOVA energy accounting failed', key, boundary, view, error, tolerance)
            energy[v, g] = np.stack([total, prefix, query, interaction])
            means = energy[v, g].mean(-1)
            summaries.append({'boundary': boundary, 'view': view, 'query_language': languages[g],
                'total_coordinate_variance_mean': float(means[0]),
                'prefix_main_effect_energy_mean': float(means[1]),
                'query_main_effect_energy_mean': float(means[2]),
                'interaction_energy_mean': float(means[3]),
                'interaction_fraction': float(means[3]/means[0]) if means[0] > 1e-15 else None,
                'minimum_raw_energy_roundoff': error})
        file = out / 'anova' / ('boundary_'+str(boundary)+'.npz')
        npz(file, grand_mean=mu, prefix_effect=pa, query_effect=qb,
            coordinate_energies_by_query_language=energy)
        checks.append({'boundary': boundary, 'prefix_effect_centering_max': float(abs(pa.mean(1)).max()),
            'query_effect_centering_max': float(abs(qb.mean(1)).max()), 'sha256': sha(file)})
    del row_means, query_means, squares, row_mean_squares
    pair_summary = analyse_pairs(key, rows, probes, native, out)
    result = {'timestamp': stamp(), 'source': snapshot(__file__), 'all_passed': True, 'model': key,
        'prefixes': P, 'queries': Q, 'boundaries': boundaries, 'width': width,
        'views': ['raw', 'per_vector_RMS'], 'query_language_groups': languages,
        'anova_summaries': summaries, 'centering_checks': checks, 'native_relation_summary': pair_summary,
        'paired_observations_file': 'paired_observations_truth_oriented.json.gz',
        'interaction_reconstruction': 'For any saved row and boundary: C[p,q,d]=Y[p,q,d]-mu[d]-A[p,d]-B[q,d]. Each view independently normalizes the full native vector before decomposition.',
        'formula_status': 'Balanced-grid two-way descriptive ANOVA identity, not a new mathematical law or unique causal/semantic decomposition.',
        'attention_alignment': 'Position alignment and token-ID occurrence-order alignment both reported; occurrence alignment is not a semantic role oracle.',
        'first_block_scope': 'Before first attention, Q is formed from current-token embedding/normalization, so same-query first-block equality is an architecture check, not a newly found semantic invariant.',
        'numerical_preflight': numerical_preflight,
        'seconds': time.monotonic()-start, 'global_encoding_mechanism_closed': False}
    save(out / 'sample_order.json', [{'sample_id': r['sample_id'], 'source_group': r['source_group'], 'split': r['split']} for r in rows])
    save(out / 'result.json', result)
    ledger('construction_full_coordinate_anova_'+key, result['seconds'])
    print('CONSTRUCTION_ANALYSIS_DONE', key, result['seconds'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['qwen4', 'qwen14', 'glm4'])
    parser.add_argument('--refresh-pairs', action='store_true')
    args = parser.parse_args()
    main(args.model, args.refresh_pairs)
