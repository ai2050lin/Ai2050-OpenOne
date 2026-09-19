"""Phase2756: P2 four-factor decomposition of generation-band rule gains.

Phase2749 (V2-P1) established per-band conditional rules on real native free
histories: every band's own rule beats its band mean, cross-band transfer
partitions into {B0}, {B1}, {B2+}, and each band carries real within-context
signal.  V2-P2 asks WHICH factor carries each band's gain: the same question
at multiple consumed-history positions is subject to a four-factor
decomposition -- lexical (token identity), frequency (unigram-count bucket),
format (top-8高频格式 token 类), length/position (consumed-prefix step proxy)
-- with the residual attributed to content.

Design (frozen before any Phase2756 observation):
  * Table predictors fitted on VALIDATION band targets only, shrunk toward the
    band mean with weight n/(n+5); evaluated on DIAGNOSTIC band targets with
    the exact Phase2749 aggregation (token MSE -> per-question -> per-context
    -> equal-cohort), so share ratios are same-scale with the archived
    Phase2749 gains.
  * Factors: lex = saturated token-id table; freq = 5 buckets (unseen + seen
    count quartiles from ALL validation band tokens of that model); fmt = top-8
    validation-count token ids each its own class, rest = 'other'; len = step
    bucket inside each band (B0/B1 single-valued -> len gain is 0 by
    construction, reported as such); additive = lex + len - band_mean (lex
    subsumes fmt and freq since both are functions of the id).
  * G_full = archived Phase2749 diagonal primary-rule gain
    (band_mean_ec - primary:j:ec, equal_cohort, from phase2749 result.json);
    shares = (band_mean_ec - predictor_ec) / G_full; content share = 1 -
    additive share.
  * Consistency gate: recomputed band-mean predictor must reproduce the
    archived Phase2749 band-mean equal-cohort MSE to 1e-8 relative.
  * Bootstrap: whole-context paired full-minus-additive, 2000 draws, per
    cohort, seeds 2756001/2756002; full per-question values from the archived
    phase2749 question_scores.npz key primary__{band}__{band}__absolute on the
    same question_order.
  * Preregistered hypothesis: B0/B1 additive share >= 0.5 (surface-factor
    dominated, per the Phase2749 band token descriptions); B2/B3/B4 content
    share > 0.5.

CPU-only float64. No model loads, no quantization, no truncation.
"""
from collections import Counter, defaultdict
import shutil
import numpy as np
from rdc_question_common import *
import rdc_question_common
import rdc_question_data as data
import phase2749_band_conditional as p9

OUT56 = BASE / 'phase2756'
B4BANDS = ['B0', 'B1', 'B2', 'B3', 'B4']
SHRINK_K = 5
FMT_TOPK = 8
BOOT_SEEDS = {'drop': 2756001, 'quoref': 2756002}
BOOT_DRAWS = 2000
LEN_EDGES = {'B0': None, 'B1': None,          # single-valued bands
             'B2': [4],                       # steps {2,3} -> bucket = step//2? no: explicit
             'B3': [6],                       # steps {4..7} -> {4,5} vs {6,7}
             'B4': [12, 16, 32]}              # steps 8+: {8-11,12-15,16-31,32+}
START = time.monotonic()


def elapsed():
    return round(time.monotonic() - START, 1)


def len_bucket(band, step):
    if band in ('B0', 'B1'):
        return 0
    edges = LEN_EDGES[band]
    return int(np.searchsorted(edges, step, side='right'))


def guard56(expected=0):
    guard(expected)
    import shutil
    assert shutil.disk_usage('D:/').free - expected > 4 * 1024 ** 3


def freeze(key):
    path = OUT56 / key / 'execution.json'
    execution = {'source': snapshot(__file__),
        'common': snapshot(Path(rdc_question_common.__file__)),
        'data': snapshot(Path(data.__file__)),
        'phase2749_module': snapshot(Path(p9.__file__)),
        'phase2749_result_sha256': sha(BASE / 'phase2749' / key / 'result.json'),
        'phase2749_scores_sha256': sha(BASE / 'phase2749' / key / 'question_scores.npz'),
        'phase2749_execution_sha256': sha(BASE / 'phase2749' / key / 'execution.json')}
    if path.exists():
        old = read(path)
        assert old['execution'] == execution
        return old
    spec = {'timestamp': stamp(), 'execution': execution,
        'status': 'frozen_before_any_Phase2756_observation',
        'question': 'Which factor carries each generation band rule gain: lexical identity, '
                    'unigram frequency, format tokens, consumed length, or residual content? '
                    'V2-P2 four-factor decomposition on the Phase2749 band framework.',
        'bands_decomposed': B4BANDS,
        'note_L_excluded': 'L overlaps B1..B4 and mixes len buckets; decomposition declared for B0..B4 only.',
        'predictors': {'lex': 'saturated token-id table, shrink n/(n+5) to band mean',
            'freq': '5 buckets: unseen + seen-count quartiles over ALL validation band tokens of the model',
            'fmt': 'top-8 validation-count ids each its own class, rest other',
            'len': 'step bucket inside band (absolute step; B0/B1 single-valued -> zero gain by construction)',
            'additive': 'lex + len - band_mean (lex subsumes fmt/freq as functions of id)'},
        'shrink_k': SHRINK_K, 'fmt_topk': FMT_TOPK, 'len_edges': {k: v for k, v in LEN_EDGES.items()},
        'gain_reference': 'G_full from archived phase2749 result.json diagonal primary rule, equal_cohort',
        'consistency_gate': 'recomputed band-mean equal-cohort MSE must match archived value (rel 1e-8)',
        'bootstrap': {'unit': 'whole context, paired full-minus-additive', 'draws': BOOT_DRAWS,
                      'seeds': BOOT_SEEDS, 'full_source': 'phase2749 question_scores.npz primary__{b}__{b}__absolute'},
        'hypothesis': {'B0_B1': 'additive share >= 0.5', 'B2_B3_B4': 'content share > 0.5'},
        'limits': 'len uses step number as consumed-length proxy (per-question prefix length monotone in step; '
                  'cross-question length differences not modelled). Tables fitted on validation only; sparse '
                  'units shrink to the band mean, so shares are honest generalization estimates. Cohort '
                  '(drop/quoref) is the only relation-family axis available in this framework.'}
    immutable(path, spec)
    return spec


class TablePredictor:
    def __init__(self, band_mean):
        self.band_mean = band_mean
        self.tab = {}

    def fit(self, keys, Y):
        for k in np.unique(keys):
            idx = keys == k
            n = int(idx.sum())
            w = n / (n + SHRINK_K)
            self.tab[int(k)] = (w * Y[idx].mean(0), n)

    def pred(self, keys):
        out = np.empty((len(keys), self.band_mean.shape[0]))
        for i, k in enumerate(keys.tolist()):
            m = self.tab.get(int(k))
            out[i] = self.band_mean if m is None else m[0]
        return out


def freq_bucket_fn(cnt, seen_counts, quantiles):
    def f(tok_id):
        c = cnt.get(int(tok_id), 0)
        if c == 0:
            return 0
        return 1 + int(np.searchsorted(quantiles, c, side='right'))
    return f


def fmt_class_fn(fmt_ids):
    rank = {int(t): i + 1 for i, t in enumerate(fmt_ids)}
    def f(tok_id):
        return rank.get(int(tok_id), 0)
    return f


def main(key):
    guard56()
    spec = freeze(key)
    folder = OUT56 / key
    finalpath = folder / 'result.json'
    if finalpath.exists():
        old = read(finalpath)
        assert old['execution_sha256'] == sha(folder / 'execution.json')
        print('FACTOR_DECOMP_ALREADY_COMPLETE', key, flush=True)
        return
    print('PHASE2756_START', key, flush=True)
    r49 = read(BASE / 'phase2749' / key / 'result.json')
    z49_path = BASE / 'phase2749' / key / 'question_scores.npz'

    # ---------- validation side: frequency/format stats + table fits ----------
    items_v, means_v = p9.load_split(key, 'validation')
    bands_v = {b: p9.band_arrays(items_v, means_v, b) for b in B4BANDS}
    counts_v = {b: int(len(bands_v[b][0])) for b in B4BANDS}
    assert counts_v['B0'] == 192 and counts_v['B1'] == 192
    cnt = Counter()
    for b in B4BANDS:
        cnt.update(bands_v[b][4].tolist())
    seen_ids = sorted(cnt)
    seen_counts = np.array([cnt[i] for i in seen_ids], dtype=np.float64)
    quantiles = np.quantile(seen_counts, [0.25, 0.5, 0.75])
    fmt_ids = [i for i, _ in sorted(cnt.items(), key=lambda t: (-t[1], t[0]))[:FMT_TOPK]]
    f_freq = freq_bucket_fn(cnt, seen_counts, quantiles)
    f_fmt = fmt_class_fn(fmt_ids)

    tables = {}
    band_means = {}
    for b in B4BANDS:
        X, C, Y, meta, ids = bands_v[b]
        bm = Y.mean(0)
        band_means[b] = bm
        steps = np.array([s for _, s in meta], dtype=np.int64)
        tabs = {
            'lex': TablePredictor(bm), 'freq': TablePredictor(bm),
            'fmt': TablePredictor(bm), 'len': TablePredictor(bm)}
        tabs['lex'].fit(ids, Y)
        tabs['freq'].fit(np.array([f_freq(t) for t in ids]), Y)
        tabs['fmt'].fit(np.array([f_fmt(t) for t in ids]), Y)
        tabs['len'].fit(np.array([len_bucket(b, s) for s in steps]), Y)
        tables[b] = tabs
        print('VAL_FIT', key, b, 'tokens', counts_v[b], 't', elapsed(), flush=True)
    del items_v, bands_v

    # ---------- diagnostic side: evaluate all factor predictors ----------
    guard56()
    items_d, means_d = p9.load_split(key, 'diagnostic')
    qorder = np.asarray(sorted(it['question_id'] for it in items_d))
    qindex = {q: i for i, q in enumerate(qorder.tolist())}
    nq = len(qorder)
    npz_arrays = {'question_order': qorder}
    bands_out = {}
    hypothesis_hits = {}
    for b in B4BANDS:
        Xd, Cd, Yd, meta_d, ids_d = p9.band_arrays(items_d, means_d, b)
        groups_d = np.array([items_d[qi]['group_id'] for qi, _ in meta_d])
        steps_d = np.array([s for _, s in meta_d], dtype=np.int64)
        tabs = tables[b]
        bm = band_means[b]
        # consistency gate: recomputed band-mean == archived band-mean MSE
        arch_bm = r49['transfer'][b]['rules']['band_mean']['absolute']['equal_cohort']
        ours = float(np.mean((np.repeat(bm[None], len(Yd), 0) - Yd) ** 2))
        # equal-cohort aggregation of the constant predictor:
        agg = p9.aggregate(np.mean((np.repeat(bm[None], len(Yd), 0) - Yd) ** 2, axis=1), meta_d, items_d)
        rel = abs(agg['equal_cohort'] - arch_bm) / max(abs(arch_bm), 1e-12)
        assert rel < 1e-8, ('band_mean consistency gate', b, agg['equal_cohort'], arch_bm)
        # factor predictions
        preds = {'lex': tabs['lex'].pred(ids_d),
                 'freq': tabs['freq'].pred(np.array([f_freq(t) for t in ids_d])),
                 'fmt': tabs['fmt'].pred(np.array([f_fmt(t) for t in ids_d])),
                 'len': tabs['len'].pred(np.array([len_bucket(b, s) for s in steps_d]))}
        preds['additive'] = preds['lex'] + preds['len'] - bm[None]
        entry = {'tokens': int(len(meta_d)), 'band_mean_consistency_rel_err': rel,
                 'archived': {'band_mean_ec': arch_bm,
                              'full_ec': r49['transfer'][b]['rules']['primary:' + b]['absolute']['equal_cohort'],
                              'shuffle_ec': r49['transfer'][b]['rules']['shuffle:' + b]['absolute']['equal_cohort']},
                 'predictors': {}}
        G_full = entry['archived']['band_mean_ec'] - entry['archived']['full_ec']
        entry['G_full_equal_cohort'] = G_full
        per_q_full = None
        with np.load(z49_path) as z:
            per_q_full = z['primary__%s__%s__absolute' % (b, b)]
            assert len(per_q_full) == nq
        shares_boot = {}
        for name, pred in preds.items():
            diff2 = np.mean((pred - Yd) ** 2, axis=1)
            agg = p9.aggregate(diff2, meta_d, items_d)
            ec = agg['equal_cohort']
            share = (arch_bm - ec) / G_full if G_full > 0 else float('nan')
            entry['predictors'][name] = {'ec': ec, 'token_weighted': agg['token_weighted'],
                                         'drop': agg['drop']['question_mean'],
                                         'quoref': agg['quoref']['question_mean'], 'share': share}
            if name == 'additive':
                per_q_add = p9.per_question_values(diff2, meta_d, items_d, qindex, nq)
                npz_arrays['additive__%s__absolute' % b] = per_q_add
                # band-mean per-question for completeness
                npz_arrays['bandmean__%s__absolute' % b] = p9.per_question_values(
                    np.mean((np.repeat(bm[None], len(Yd), 0) - Yd) ** 2, axis=1), meta_d, items_d, qindex, nq)
                # paired bootstrap full-minus-additive at context level
                q2ctx = {}
                q2coh = {}
                for it in items_d:
                    q2ctx[it['question_id']] = it['group_id']
                    q2coh[it['question_id']] = it['cohort']
                cv_f, cv_a = defaultdict(list), defaultdict(list)
                for q, i in qindex.items():
                    f, a = per_q_full[i], per_q_add[i]
                    if not (np.isfinite(f) and np.isfinite(a)):
                        continue
                    cv_f[q2ctx[q]].append(f)
                    cv_a[q2ctx[q]].append(a)
                coh_ctx = {}
                for q, c in q2coh.items():
                    coh_ctx[q2ctx[q]] = c
                summ = {}
                draws_all = []
                for c in ['drop', 'quoref']:
                    keys = sorted(k for k in cv_f if coh_ctx[k] == c)
                    delta = np.array([np.mean(cv_f[k]) - np.mean(cv_a[k]) for k in keys])
                    s = p9.boot_draws(delta, BOOT_SEEDS[c])
                    draws_all.append(s)
                    summ[c] = {'contexts': len(delta), 'mean': float(delta.mean()),
                               'interval95': [float(x) for x in np.quantile(s, [.025, .975])]}
                comb = (draws_all[0] + draws_all[1]) / 2
                summ['equal'] = {'mean': float(comb.mean()),
                                 'interval95': [float(x) for x in np.quantile(comb, [.025, .975])]}
                shares_boot = summ
        entry['bootstrap_full_minus_additive'] = shares_boot
        add_share = entry['predictors']['additive']['share']
        content_share = 1.0 - add_share
        entry['content_share'] = content_share
        if b in ('B0', 'B1'):
            hypothesis_hits[b] = bool(add_share >= 0.5)
        else:
            hypothesis_hits[b] = bool(content_share > 0.5)
        bands_out[b] = entry
        print('EVAL_BAND', key, b, 'G_full %.4f' % G_full,
              ' '.join('%s %.4f' % (n, entry['predictors'][n]['share']) for n in ['lex', 'freq', 'fmt', 'len', 'additive']),
              'content %.4f' % content_share, 't', elapsed(), flush=True)
        del preds, Xd, Cd, Yd, ids_d, meta_d, groups_d
    all_passed = all(hypothesis_hits.values())
    result = {'timestamp': stamp(), 'all_passed': all_passed, 'model': key,
        'execution_sha256': sha(folder / 'execution.json'),
        'phase2749_result_sha256': spec['execution']['phase2749_result_sha256'],
        'bands': B4BANDS, 'counts_validation': counts_v,
        'fmt_top_ids': [int(i) for i in fmt_ids],
        'freq_quantiles': [float(x) for x in quantiles],
        'decomposition': bands_out, 'hypothesis_hits': hypothesis_hits,
        'bootstrap': {'seeds': BOOT_SEEDS, 'draws': BOOT_DRAWS},
        'seconds': time.monotonic() - START, 'limits': spec['limits']}
    immutable(finalpath, result)
    npz(folder / 'factor_scores.npz', **npz_arrays)
    print('FACTOR_DECOMP_COMPLETE', key, 'all_passed', all_passed, round(result['seconds'], 1), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['qwen4', 'qwen14', 'glm4'], required=True)
    main(parser.parse_args().model)
