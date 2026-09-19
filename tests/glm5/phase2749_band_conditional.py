"""Phase2749: generation-stage-conditional conditional-output-field rules.

Phase2748 fitted its primary rules on first natural prefixes only; those rules
degrade on later own-history steps (worse than the fixed training mean on
qwen4/qwen14, slightly better on glm4).  This phase asks the direct follow-up
with the same registered estimator family: fit one conditional rule per
generation-stage band on real native free histories, select hyperparameters by
five-fold context-group cross-validation inside the validation split only,
freeze the selection, then read diagnostic targets once and report the full
cross-band transfer matrix against band-mean, first-step-trained-mean and
within-context target-shuffle controls.

CPU-only float64. No model loads, no quantization, no delta transplant, no
Top-K/PCA truncation. All coordinates of every declared boundary are used.
"""
import argparse
from collections import defaultdict
import numpy as np
from rdc_question_common import *
import rdc_question_fit
import rdc_question_selectivity
import rdc_question_common
import rdc_question_data as data
from rdc_question_fit import FeatureKernel, Spectrum, denominators
from rdc_question_selectivity import group_center

OUT9 = BASE / 'phase2749'
PHYSICAL9 = Path('C:/AI2050-RDC-Archive/rdc_query_construction_20260913/phase2749')
BANDS = ['B0', 'B1', 'B2', 'B3', 'B4', 'L']
ALPHAS = [0.0, 4.0]
RHOS = [0.0, 1.0, 3.0]
DFS = [32, 64, 128, 256, 512]
FOLD_SEED = 2749001
SHUFFLE_SEED_BASE = 2749012
BOOT_SEEDS = {'drop': 2749010, 'quoref': 2749011}
BOOT_DRAWS = 2000
START = time.monotonic()


def elapsed():
    return round(time.monotonic() - START, 1)


def in_band(step, band):
    if band == 'B0':
        return step == 0
    if band == 'B1':
        return step == 1
    if band == 'B2':
        return 2 <= step < 4
    if band == 'B3':
        return 4 <= step < 8
    if band == 'B4':
        return step >= 8
    if band == 'L':
        return step >= 1
    raise KeyError(band)


def guard9(expected=0):
    guard(expected)
    assert OUT9.resolve() == PHYSICAL9.resolve()
    import shutil
    assert shutil.disk_usage('C:/').free - expected > 4 * 1024 ** 3


def freeze(key):
    path = OUT9 / key / 'execution.json'
    execution = {'source': snapshot(__file__),
        'solver': snapshot(Path(rdc_question_fit.__file__)),
        'selectivity': snapshot(Path(rdc_question_selectivity.__file__)),
        'data': snapshot(Path(data.__file__)),
        'common': snapshot(Path(rdc_question_common.__file__)),
        'phase2748_selection_sha256': sha(BASE / 'phase2748/fit' / key / 'validation_selection.json'),
        'phase2748_native_history_prediction_sha256': sha(BASE / 'phase2748/native_history_prediction' / key / 'result.json')}
    if path.exists():
        old = read(path)
        assert old['execution'] == execution
        return old
    spec = {'timestamp': stamp(), 'execution': execution,
        'status': 'frozen_before_any_Phase2749_band_observation',
        'question': 'Does conditional-output-field rule structure exist per generation stage, and how '
                    'does it transfer across stages? Phase2748 showed first-prefix-fitted rules fail on '
                    'later steps; here every band gets its own rule from the same estimator family and '
                    'the full source->eval transfer matrix is measured.',
        'bands': {'B0': 'step 0 (first prefix)', 'B1': 'step 1', 'B2': 'steps 2-3', 'B3': 'steps 4-7',
                  'B4': 'steps 8+', 'L': 'all steps >=1 (pooled; overlaps B1..B4, kept for direct '
                  'comparison with the Phase2748 later_all bin)'},
        'inputs': 'Per natural generation step t of each validation/diagnostic question: query '
                  'x_t=[H12_t; block12 native attention read_t] (the Phase2748 native_source_read '
                  'family), context = separately consumed context_H12_mean, target = native postnorm_t. '
                  'Only the actual model past is used; no later target enters any fit.',
        'estimator': 'Exact Phase2748 family: FeatureKernel train-only standardization, centered kernel '
                     'k=kq+kc+rho*kq*kc on the full sample space, contrast transform B with '
                     'B^T B=W+alpha*P^T W P (P subtracts within-context token means), full eigenspectrum, '
                     'DF-target ridge by monotone positive-lambda bisection, solution '
                     'O=B^T U diag(1/(e+lambda)) U^T B, prediction = centered cross-kernel @ O @ (Y-mean) + mean. '
                     'No spectral truncation, every eigencomponent retained.',
        'grid': {'contrast_strength_alpha': ALPHAS, 'interaction_rho': RHOS, 'effective_df_targets': DFS,
                 'note': 'DF 512 added beyond the Phase2748 grid because Phase2748 selection hit the DF 256 '
                         'boundary on all three models; alpha 16 dropped to bound CV cost after Phase2748 '
                         'selected alpha 4 everywhere. Declared before any Phase2749 fit.'},
        'weights': 'Uniform over token samples within each band training set. Context groups at token level.',
        'split_protocol': 'Fit and select ONLY on validation free histories (48 contexts, 192 questions). '
                          'Five-fold context-group CV, fold seed 2749001; fold objective = mean over cohorts '
                          'of 0.5*(absolute_MSE/fold-train total variance + within-context_MSE/fold-train '
                          'within variance), denominators from each fold training part per cohort. Selection '
                          'minimizes the mean fold objective; ties broken lexicographically (alpha, rho, DF). '
                          'Diagnostic (96 contexts, 384 questions) targets are read only after cv.json is frozen.',
        'control': 'Within-context token-level target permutation per band (seed 2749012+band_index): '
                   'preserves each context target multiset, destroys query-target alignment. The control '
                   'receives its own full CV selection, so any primary advantage is conservative.',
        'baselines': {'band_mean': 'constant = weighted mean of validation band targets (the Phase2748 '
                      'training-mean analogue, per band)', 'phase2748_first_step_mean': 'constant = '
                      'target_mean of the deployed Phase2748 primary rule (train-split first-prefix '
                      'postnorm mean), identical vector on every band'},
        'estimands': '6x6 transfer matrix (source band rule -> eval band): token-weighted MSE, per-question '
                     'means aggregated per cohort then equal-cohort, A/B/C decomposition against the eval '
                     'band validation mean (MSE = A + B - 2C exactly), within-context (group-centered) MSE, '
                     'query/context standardized RMS matrices (distribution-shift lens), whole-context '
                     'paired bootstrap, 2000 draws, seeds 2749010/2749011, for rule-minus-band-mean per '
                     '(source, eval) and diagonal-minus-shuffle per eval band.',
        'descriptive': 'Per diagnostic band: target energy per token, top-8 generated token ids with counts. '
                       'Per-question score arrays retained in question_scores.npz for later phases.',
        'retention': 'Fitted rule state kept minimal (standardization, rho, coefficients, mean, ridge); '
                     'operators not materialized. All results immutable JSON under phase2749/<model>/.',
        'limits': 'Band rules are regression fits on observed native histories, not generators; no vocabulary '
                  'readout or answer claim here. B4 conditions on surviving long histories, not the original '
                  'population. Cross-band transfer is descriptive, not causal identification.'}
    immutable(path, spec)
    return spec


def load_split(key, split):
    rows, groups, questions = data.index(key, {split})
    assert len(groups) == (48 if split == 'validation' else 96)
    means = {gid: data.field(g['context_field'], ['context_H12_mean'])['context_H12_mean'] for gid, g in groups.items()}
    items = []
    for r in rows:
        rec = questions[r['question_id']]['history']
        v = data.field(rec['field'], ['H12_last_BF16', 'native_source_read_BF16', 'postnorm_BF16', 'generated_ids'])
        h, rd, y = v['H12_last_BF16'], v['native_source_read_BF16'], v['postnorm_BF16']
        assert h.shape == rd.shape == y.shape and len(v['generated_ids']) == len(h) and len(h) >= 2
        items.append({'question_id': r['question_id'], 'group_id': r['group_id'], 'cohort': r['cohort'],
            'wci': r['within_context_index'], 'h': h, 'rd': rd, 'y': y, 'ids': v['generated_ids']})
    assert len(items) == (192 if split == 'validation' else 384)
    return items, means


def band_arrays(items, means, band):
    X, C, Y, meta, ids = [], [], [], [], []
    for qi, it in enumerate(items):
        steps = np.asarray([s for s in range(len(it['h'])) if in_band(s, band)])
        if not len(steps):
            continue
        X.append(np.concatenate([it['h'][steps], it['rd'][steps]], axis=1))
        C.append(np.repeat(means[it['group_id']][None], len(steps), axis=0))
        Y.append(it['y'][steps])
        ids.append(it['ids'][steps])
        meta.extend((qi, int(s)) for s in steps)
    return (np.concatenate(X), np.concatenate(C), np.concatenate(Y), meta, np.concatenate(ids))


def shuffle_targets(Y, groups, seed):
    rng = np.random.default_rng(seed)
    out = Y.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        out[idx] = Y[rng.permutation(idx)]
    return out


def fold_objective(pred, yte, groups_te, cohorts_te, den):
    pc = group_center(pred, groups_te, np.ones(len(pred)))
    ac = group_center(yte, groups_te, np.ones(len(yte)))
    absq = np.mean((pred - yte) ** 2, axis=1)
    withinq = np.mean((pc - ac) ** 2, axis=1)
    total, count = 0.0, 0
    for c in sorted(set(cohorts_te)):
        m = cohorts_te == c
        total += .5 * (float(absq[m].mean()) / den[c]['total'] + float(withinq[m].mean()) / den[c]['within'])
        count += 1
    return total / max(count, 1)


def cv_select(X, C, Y, groups, cohorts, seed, label):
    n = len(X)
    ctx = np.array(sorted(set(groups)))
    rng = np.random.default_rng(seed)
    rng.shuffle(ctx)
    fold_ids = np.full(n, -1)
    for f, block in enumerate(np.array_split(ctx, 5)):
        fold_ids[np.isin(groups, block)] = f
    assert (fold_ids >= 0).all()
    scores = {(a, r, d): [] for a in ALPHAS for r in RHOS for d in DFS}
    folds_report = []
    for f in range(5):
        tr = fold_ids != f
        te = ~tr
        if int(te.sum()) == 0 or int(tr.sum()) < 32:
            folds_report.append({'fold': f, 'skipped': True, 'train': int(tr.sum()), 'test': int(te.sum())})
            continue
        w = np.ones(int(tr.sum())) / int(tr.sum())
        feature = FeatureKernel(X[tr], C[tr], w)
        gtr, ytr, ctr = groups[tr], Y[tr], cohorts[tr]
        den = denominators(ytr, gtr, ctr)
        tr_parts = feature.train_parts
        te_parts = feature.parts(X[te], C[te])
        entry = {'fold': f, 'train': int(tr.sum()), 'test': int(te.sum()), 'candidates': {}}
        for rho in RHOS:
            gram, _, _ = feature.kernel(tr_parts, rho)
            cross, _, _ = feature.kernel(te_parts, rho)
            for alpha in ALPHAS:
                spec = Spectrum(gram, gtr, w, alpha)
                for df in DFS:
                    ridge, numerical = spec.ridge(df)
                    pred = spec.predictions(cross, ytr, w, ridge)
                    val = fold_objective(pred, Y[te], groups[te], cohorts[te], den)
                    scores[(alpha, rho, df)].append(val)
                    entry['candidates']['%g/%g/%d' % (alpha, rho, df)] = val
                del spec
            del gram, cross
        del feature, te_parts
        folds_report.append(entry)
        print('CV_FOLD', label, 'fold', f, 'train', int(tr.sum()), 'test', int(te.sum()), 't', elapsed(), flush=True)
    averaged = {k: (float(np.mean(v)) if v else float('inf')) for k, v in scores.items()}
    best = min(averaged, key=lambda k: (averaged[k], k[0], k[1], k[2]))
    choice = {'alpha': float(best[0]), 'rho': float(best[1]), 'DF': int(best[2]), 'objective': averaged[best]}
    print('CV_SELECTED', label, choice, 't', elapsed(), flush=True)
    return {'choice': choice, 'grid_averaged': {'%g/%g/%d' % k: v for k, v in averaged.items()},
            'fold_report': folds_report, 'seed': seed}


def final_fit(X, C, Y, groups, choice, label):
    n = len(X)
    w = np.ones(n) / n
    feature = FeatureKernel(X, C, w)
    gram, _, _ = feature.kernel(feature.train_parts, choice['rho'])
    spec = Spectrum(gram, groups, w, choice['alpha'])
    ridge, numerical = spec.ridge(choice['DF'])
    mean = w @ Y
    coef = spec.solution(ridge) @ (Y - mean)
    fitted = gram @ coef + mean
    direct = spec.predictions(gram, Y, w, ridge)
    err = float(np.abs(direct - fitted).max())
    assert np.allclose(direct, fitted, atol=1e-7, rtol=1e-7), (label, err)
    train_mse = float(((fitted - Y) ** 2).mean())
    del gram, spec
    return {'feature': feature, 'coef': coef, 'mean': mean, 'ridge': ridge, 'numerical': numerical,
        'choice': dict(choice), 'train_MSE': train_mse, 'audit_max_error': err}


def aggregate(tok_values, meta, items):
    per_q = defaultdict(list)
    for (qi, _), v in zip(meta, tok_values):
        per_q[qi].append(float(v))
    qmap = {qi: float(np.mean(v)) for qi, v in per_q.items()}
    by_ctx = defaultdict(list)
    cohort = {}
    for qi, v in qmap.items():
        it = items[qi]
        by_ctx[it['group_id']].append(v)
        cohort[it['group_id']] = it['cohort']
    out = {'token_weighted': float(np.mean(tok_values))}
    for c in sorted(set(cohort.values())):
        values = {g: float(np.mean(by_ctx[g])) for g in sorted(by_ctx) if cohort[g] == c}
        out[c] = {'contexts': len(values), 'question_mean': float(np.mean(list(values.values()))),
                  'context_values': values}
    out['equal_cohort'] = float(np.mean([out[c]['question_mean'] for c in sorted(set(cohort.values()))]))
    assert np.isfinite(out['token_weighted'])
    return out


def per_question_values(tok_values, meta, items, qindex, nq):
    acc = defaultdict(list)
    for (qi, _), v in zip(meta, tok_values):
        acc[qi].append(float(v))
    out = np.full(nq, np.nan)
    for qi, v in acc.items():
        out[qindex[items[qi]['question_id']]] = float(np.mean(v))
    return out


def boot_draws(delta, seed):
    rng = np.random.default_rng(seed)
    return delta[rng.integers(0, len(delta), (BOOT_DRAWS, len(delta)))].mean(1)


def paired_summary(a_vals, b_vals):
    a_vals = {c: a_vals[c] for c in ['drop', 'quoref']}
    b_vals = {c: b_vals[c] for c in ['drop', 'quoref']}
    out, draws = {}, []
    for c in ['drop', 'quoref']:
        a, b = a_vals[c], b_vals[c]
        keys = sorted(a)
        assert keys == sorted(b), 'context sets must match for paired bootstrap'
        delta = np.array([a[k] - b[k] for k in keys])
        s = boot_draws(delta, BOOT_SEEDS[c])
        draws.append(s)
        out[c] = {'contexts': int(len(delta)), 'mean': float(delta.mean()),
                  'interval95': [float(x) for x in np.quantile(s, [.025, .975])]}
    combined = (draws[0] + draws[1]) / 2
    out['equal'] = {'mean': float(combined.mean()),
                    'interval95': [float(x) for x in np.quantile(combined, [.025, .975])]}
    return out


def predictions_of(state, X, C):
    parts = state['feature'].parts(X, C)
    cross, _, _ = state['feature'].kernel(parts, state['choice']['rho'])
    pred = cross @ state['coef'] + state['mean']
    del parts, cross
    assert np.isfinite(pred).all()
    return pred


def rule_scores(pred, Y, groups, meta, items, qindex, nq, reference):
    diff2 = np.mean((pred - Y) ** 2, axis=1)
    pc = pred - reference
    ac = Y - reference
    B_tok = np.mean(pc ** 2, axis=1)
    C_tok = np.mean(pc * ac, axis=1)
    pwc = group_center(pred, groups, np.ones(len(pred)))
    ywc = group_center(Y, groups, np.ones(len(Y)))
    W_tok = np.mean((pwc - ywc) ** 2, axis=1)
    del pred, pc, ac, pwc, ywc
    scores = {'absolute': aggregate(diff2, meta, items), 'B': aggregate(B_tok, meta, items),
        'C': aggregate(C_tok, meta, items), 'within': aggregate(W_tok, meta, items)}
    arrays = {'absolute': per_question_values(diff2, meta, items, qindex, nq),
        'within': per_question_values(W_tok, meta, items, qindex, nq)}
    return scores, arrays


def rms_under(state, X, C):
    q = (X - state['feature'].state['query_mean']) / state['feature'].state['query_scale']
    c = (C - state['feature'].state['context_mean']) / state['feature'].state['context_scale']
    return float(np.sqrt((q * q).mean())), float(np.sqrt((c * c).mean()))


def main(key):
    guard9()
    spec = freeze(key)
    folder = OUT9 / key
    finalpath = folder / 'result.json'
    if finalpath.exists():
        old = read(finalpath)
        assert old['execution_sha256'] == sha(folder / 'execution.json')
        print('BAND_CONDITIONAL_ALREADY_COMPLETE', key, flush=True)
        return
    print('PHASE2749_START', key, flush=True)
    # ---------------- validation side: fit and select ----------------
    items_v, means_v = load_split(key, 'validation')
    bands_v = {b: band_arrays(items_v, means_v, b) for b in BANDS}
    counts_v = {b: int(len(bands_v[b][0])) for b in BANDS}
    assert counts_v['B0'] == 192 and counts_v['B1'] == 192
    assert counts_v['L'] == counts_v['B1'] + counts_v['B2'] + counts_v['B3'] + counts_v['B4']
    assert sum(counts_v[b] for b in ['B0', 'B1', 'B2', 'B3', 'B4']) == sum(len(it['h']) for it in items_v)
    selections, rules, controls = {}, {}, {}
    for bi, b in enumerate(BANDS):
        X, C, Y, meta, ids = bands_v[b]
        groups = np.array([items_v[qi]['group_id'] for qi, _ in meta])
        cohorts = np.array([items_v[qi]['cohort'] for qi, _ in meta])
        sel_p = cv_select(X, C, Y, groups, cohorts, FOLD_SEED, key + ':' + b + ':primary')
        seed_s = SHUFFLE_SEED_BASE + bi
        Ys = shuffle_targets(Y, groups, seed_s)
        sel_c = cv_select(X, C, Ys, groups, cohorts, FOLD_SEED, key + ':' + b + ':shuffle')
        selections[b] = {'primary': sel_p, 'shuffle_control': sel_c, 'shuffle_seed': seed_s}
        rules[b] = final_fit(X, C, Y, groups, sel_p['choice'], key + ':' + b + ':primary')
        controls[b] = final_fit(X, C, Ys, groups, sel_c['choice'], key + ':' + b + ':shuffle')
        print('BAND_FIT', b, 'train_MSE %.6f' % rules[b]['train_MSE'],
              'shuffle %.6f' % controls[b]['train_MSE'],
              'choice a%g/r%g/df%d' % (rules[b]['choice']['alpha'], rules[b]['choice']['rho'], rules[b]['choice']['DF']),
              't', elapsed(), flush=True)
    cv = {'timestamp': stamp(), 'model': key, 'bands': BANDS,
        'grid': {'alphas': ALPHAS, 'rhos': RHOS, 'df_targets': DFS},
        'fold_seed': FOLD_SEED, 'shuffle_seed_base': SHUFFLE_SEED_BASE,
        'validation_band_tokens': counts_v, 'selections': selections}
    immutable(folder / 'cv.json', cv)
    print('CV_FROZEN', key, 't', elapsed(), flush=True)
    del items_v, bands_v
    # ---------------- Phase2748 frozen reference ----------------
    ref_sel = read(BASE / 'phase2748/fit' / key / 'validation_selection.json')
    prim28 = ref_sel['primary_rule']
    ref_res = read(BASE / 'phase2748/native_history_prediction' / key / 'result.json')
    anchors = {}
    for rec in ref_res['records']:
        if rec['variant'] == prim28 and rec['split'] == 'diagnostic':
            for x in rec['summaries']:
                ec = x['prediction_minus_training_mean']['equal_cohort']
                anchors[x['bin']] = {'prediction_MSE_equal_cohort': ec['left_mean'],
                    'delta_vs_training_mean': ec['mean_left_minus_right']}
    deployed = read(BASE / 'phase2748/fit' / key / 'deployed' / (prim28 + '.json'))
    assert sha(ROOT / deployed['field']['path']) == deployed['field']['sha256']
    with np.load(ROOT / deployed['field']['path']) as z:
        mean28 = z['target_mean'].astype(np.float64)
    # ---------------- diagnostic side: read once, evaluate ----------------
    guard9()
    items_d, means_d = load_split(key, 'diagnostic')
    qorder = np.asarray(sorted(it['question_id'] for it in items_d))
    qindex = {q: i for i, q in enumerate(qorder.tolist())}
    nq = len(qorder)
    npz_arrays = {'question_order': qorder}
    totals_d = {}
    result_bands = {}
    rms_out = {}
    for j in BANDS:
        Xd, Cd, Yd, meta_d, ids_d = band_arrays(items_d, means_d, j)
        groups_d = np.array([items_d[qi]['group_id'] for qi, _ in meta_d])
        totals_d[j] = int(len(meta_d))
        assert np.isfinite(Yd).all()
        reference = rules[j]['mean']
        unique_ids, id_counts = np.unique(ids_d, return_counts=True)
        top = sorted(zip(unique_ids.tolist(), id_counts.tolist()), key=lambda t: (-t[1], t[0]))[:8]
        A_tok = np.mean((Yd - reference) ** 2, axis=1)
        aggA = aggregate(A_tok, meta_d, items_d)
        entry = {'tokens': int(len(meta_d)), 'questions': int(len({qi for qi, _ in meta_d})),
            'contexts': int(len(set(groups_d))), 'target_energy_per_token': float((Yd ** 2).mean()),
            'around_validation_band_mean_MSE': {k: aggA[k] for k in ['token_weighted', 'equal_cohort', 'drop', 'quoref']},
            'top_generated_ids': [[int(i), int(c)] for i, c in top], 'rules': {}, 'bootstrap': {}}
        npz_arrays['A__%s' % j] = per_question_values(A_tok, meta_d, items_d, qindex, nq)
        del A_tok, aggA
        scores = {}
        rule_items = [('primary:' + i, rules[i]) for i in BANDS] + [('shuffle:' + i, controls[i]) for i in BANDS] + \
            [('band_mean', None), ('phase2748_first_step_mean', None)]
        for name, state in rule_items:
            if state is None:
                value = reference if name == 'band_mean' else mean28
                pred = np.repeat(value[None], len(Yd), axis=0)
            else:
                pred = predictions_of(state, Xd, Cd)
            sc, arrays = rule_scores(pred, Yd, groups_d, meta_d, items_d, qindex, nq, reference)
            del pred
            scores[name] = sc
            family, _, src = name.partition(':')
            if src:
                npz_arrays['%s__%s__%s__absolute' % (family, src, j)] = arrays['absolute']
                npz_arrays['%s__%s__%s__within' % (family, src, j)] = arrays['within']
            else:
                npz_arrays['%s__%s__absolute' % (family, j)] = arrays['absolute']
                npz_arrays['%s__%s__within' % (family, j)] = arrays['within']
            del arrays
        for name, sc in scores.items():
            entry['rules'][name] = {metric: {k: agg[k] for k in ['token_weighted', 'equal_cohort', 'drop', 'quoref']}
                                    for metric, agg in sc.items()}
        entry['bootstrap']['diagonal_minus_shuffle'] = paired_summary(
            {c: scores['primary:' + j]['absolute'][c]['context_values'] for c in ['drop', 'quoref']},
            {c: scores['shuffle:' + j]['absolute'][c]['context_values'] for c in ['drop', 'quoref']})
        entry['bootstrap']['primary_minus_band_mean'] = {
            i: paired_summary({c: scores['primary:' + i]['absolute'][c]['context_values'] for c in ['drop', 'quoref']},
                              {c: scores['band_mean']['absolute'][c]['context_values'] for c in ['drop', 'quoref']})
            for i in BANDS}
        rms_out[j] = {i: dict(zip(['query_RMS', 'context_RMS'], rms_under(rules[i], Xd, Cd))) for i in BANDS}
        result_bands[j] = entry
        print('EVAL_BAND', j, 'tokens', entry['tokens'],
              'diag %.4f' % scores['primary:' + j]['absolute']['equal_cohort'],
              'bandmean %.4f' % scores['band_mean']['absolute']['equal_cohort'],
              'mean2748 %.4f' % scores['phase2748_first_step_mean']['absolute']['equal_cohort'],
              'shuffle %.4f' % scores['shuffle:' + j]['absolute']['equal_cohort'], 't', elapsed(), flush=True)
        del scores, Xd, Cd, Yd, ids_d, meta_d, groups_d
    npzpath = folder / 'question_scores.npz'
    assert not npzpath.exists()
    npz(npzpath, **npz_arrays)
    result = {'timestamp': stamp(), 'all_passed': True, 'model': key,
        'execution_sha256': sha(folder / 'execution.json'), 'cv_sha256': sha(folder / 'cv.json'),
        'bands': BANDS, 'grid': cv['grid'],
        'validation_band_tokens': counts_v, 'diagnostic_band_tokens': totals_d,
        'selected': {b: {'primary': {k: selections[b]['primary']['choice'][k] for k in ['alpha', 'rho', 'DF', 'objective']},
                         'shuffle_control': {k: selections[b]['shuffle_control']['choice'][k] for k in ['alpha', 'rho', 'DF', 'objective']}} for b in BANDS},
        'training_fit_MSE': {b: {'primary': rules[b]['train_MSE'], 'shuffle_control': controls[b]['train_MSE'],
                                 'operator_reconstruction_max_error': rules[b]['audit_max_error'],
                                 'lambda': rules[b]['ridge']} for b in BANDS},
        'transfer': result_bands, 'query_RMS_shift': rms_out,
        'phase2748_reference': {'primary_rule': prim28, 'diagnostic_bin_summaries': anchors,
            'native_history_prediction_result_sha256': sha(BASE / 'phase2748/native_history_prediction' / key / 'result.json'),
            'first_step_trained_mean_source': 'phase2748/fit/%s/deployed/%s.json target_mean' % (key, prim28)},
        'bootstrap': {'seeds': BOOT_SEEDS, 'draws': BOOT_DRAWS, 'unit': 'whole context, paired',
                      'equal_cohort': 'mean of the two cohort bootstrap draw vectors, then quantiles'},
        'question_scores_file': {'path': npzpath.relative_to(ROOT).as_posix(), 'sha256': sha(npzpath)},
        'seconds': time.monotonic() - START, 'limits': spec['limits']}
    immutable(finalpath, result)
    print('BAND_CONDITIONAL_COMPLETE', key, 'tokens', sum(totals_d.values()), round(result['seconds'], 1), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['qwen4', 'qwen14', 'glm4'], required=True)
    main(parser.parse_args().model)
