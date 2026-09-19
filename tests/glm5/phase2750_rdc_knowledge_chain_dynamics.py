"""Phase2750: P4 knowledge-chain carrier test via paired-world stepwise postnorm divergence.

The V2 plan P4 hypothesis: chain-type relations may not live in the final-vector
direction but in the multi-token generation trajectory.  Test: compare the
stepwise postnorm divergence between the paired own-history worlds
(true_token = real answer-token history vs within_surface_class_permuted_token
= value-pair-shuffled history) across relation families, primarily
knowledge_chain vs attribute_binding.  If the internal postnorm divergence of
knowledge_chain questions is systematically delayed/diffuse relative to
attribute_binding, the carrier-in-trajectory claim is supported and the
extraction target becomes a trajectory-level operator.

Data: Phase2747 own-history qwen4 fields, 512 questions x 7 worlds, per-step
postnorm_BF16 (n,2560) plus all_hidden_BF16 (n,37,2560) for the 44 frozen
expressions.  No model loads; CPU-only float64; full coordinates, no
truncation, no Top-K/PCA.  Same-prefix constraint: divergence statistics are
computed only on generation steps whose generated token id is identical across
the two worlds, so every reported difference is same-position causal history
content, not token-sequence mismatch.

Preregistration is frozen in execution.json before any divergence is measured.
"""
import argparse
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import rdc_construction_common
from rdc_construction_common import (ROOT, RESULT, BASE, MEMO, stamp, sha, read, save,
                                     snapshot, guard, ledger, failure, unbits, npz)

OUT50 = BASE / 'phase2750'
PHYSICAL50 = Path('C:/AI2050-RDC-Archive/rdc_query_construction_20260913/phase2750')
OH47 = BASE / 'phase2747/own_history/qwen4'
VARIANTS = ['native', 'true_token_2747', 'within_surface_class_permuted_token_2747',
            'true_token_2748', 'within_surface_class_permuted_token_2748',
            'surface_class_mass_2747', 'surface_class_mass_2748']
WORLD_PAIRS = [('true_token_2747', 'within_surface_class_permuted_token_2747'),
               ('true_token_2748', 'within_surface_class_permuted_token_2748')]
STABILITY_PAIR = ('true_token_2747', 'true_token_2748')
NATIVE_REFS = [('true_token_2747', 'native'),
               ('within_surface_class_permuted_token_2747', 'native')]
CONTROLLED_FAMILIES = ['attribute_binding', 'knowledge_chain', 'long_distance_role',
                       'negation_scope', 'word_sense']
NATURAL_FAMILY = 'natural_cmrc'
CURVE_T = 16
SLOPE_WINDOW = 9
NULL_DRAWS = 200
NULL_SEED = 2750012
BOOT_DRAWS = 2000
BOOT_SEEDS = {'first': 2750010}
TAU_FACTOR = 4.0
LAYER_REPORT_T = 8
START = time.monotonic()


def elapsed():
    return round(time.monotonic() - START, 1)


def guard50(expected=0):
    guard(expected)
    assert OUT50.resolve() == PHYSICAL50.resolve()
    import shutil
    assert shutil.disk_usage('C:/').free - expected > 4 * 1024 ** 3


def load_records(variant):
    folder = OH47 / variant / 'records'
    paths = sorted(folder.glob('*.json'))
    assert len(paths) == 512, (variant, len(paths))
    records = {}
    for p in paths:
        r = read(p)
        assert r['variant'] == variant
        records[r['sample_id']] = r
    return records


class VectorCache:
    """Lazy float64 postnorm cache over the immutable Phase2747 fields."""

    def __init__(self):
        self.cache = {}

    def get(self, variant, sid, record):
        key = (variant, sid)
        if key not in self.cache:
            z = np.load(record['field']['physical_path'], allow_pickle=False)
            stored = z['postnorm_BF16']
            n = len(record['generated_ids'])
            assert stored.shape == (n, 2560), (sid, stored.shape, n)
            self.cache[key] = unbits(stored).astype(np.float64)
        return self.cache[key]


def paired_divergence(a_vec, a_ids, b_vec, b_ids):
    """Same-prefix stepwise divergence between two worlds of one question."""
    n = min(len(a_ids), len(b_ids))
    div = None
    for i in range(n):
        if a_ids[i] != b_ids[i]:
            div = i
            break
    if div is None and len(a_ids) != len(b_ids):
        div = n
    comparable = n if div is None else div
    a, b = a_vec[:comparable], b_vec[:comparable]
    m = ((a - b) ** 2).mean(-1)
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    dot = (a * b).sum(-1)
    ok = (na > 0) & (nb > 0)
    cosd = np.ones((comparable,), dtype=np.float64)
    cosd[ok] = 1.0 - dot[ok] / (na[ok] * nb[ok])
    relrms = np.sqrt(m) / np.maximum(np.sqrt((a ** 2).mean(-1)), 1e-12)
    return {'m': m, 'cosd': cosd, 'relrms': relrms, 'token_div': div,
            'comparable': comparable, 'n_a': len(a_ids), 'n_b': len(b_ids)}


def cross_divergence(a_vec, b_vec):
    """Unrelated-position divergence used for the reshuffle null."""
    n = min(a_vec.shape[0], b_vec.shape[0])
    return ((a_vec[:n] - b_vec[:n]) ** 2).mean(-1)


def main(partial=False):
    folder = OUT50 / 'qwen4'
    finish = folder / ('partial_' + str(time.time_ns()) + '.json' if partial else 'result.json')
    if finish.exists():
        return
    guard50()
    folder.mkdir(parents=True, exist_ok=True)

    # ---- freeze preregistration before any divergence is measured ----
    execution = {'source': snapshot(__file__),
                 'construction_common': snapshot(Path(rdc_construction_common.__file__)),
                 'phase2747_own_history_analysis_sha256': sha(OH47.parent / 'analysis/result.json'),
                 'phase2747_run_results_sha256': {v: sha(OH47 / v / 'result.json') for v in VARIANTS}}
    exec_path = folder / 'execution.json'
    if exec_path.exists():
        old = read(exec_path)
        assert old['execution'] == execution, 'Phase2750 execution drift'
    else:
        save(exec_path, {'timestamp': stamp(), 'execution': execution,
            'status': 'frozen_before_any_Phase2750_divergence_measurement',
            'question': 'P4 carrier test: does the chain-type relation live in the multi-token '
                        'generation trajectory rather than the final-vector direction? Compare the '
                        'stepwise postnorm divergence between paired own-history worlds '
                        '(true_token vs within_surface_class_permuted_token) across relation '
                        'families; knowledge_chain vs attribute_binding is the preregistered contrast.',
            'model': 'qwen4',
            'world_pairs': [str(p) for p in WORLD_PAIRS],
            'same_prefix_rule': 'All divergence statistics use only generation steps whose generated '
                                'token id is identical across the two worlds (same-position causal '
                                'history content). Steps after the behavioural token divergence are '
                                'excluded from the causal readout and reported separately.',
            'measurements': 'Stepwise full-coordinate MSE m(t), cosine distance, relative RMS; '
                            'postnorm divergence onset tau = first step with m(t) > 4 x family '
                            'question-reshuffle null median (right-censored at the comparable-prefix '
                            'length when never exceeded); log m slope over steps 0..8.',
            'null': 'Within-family question reshuffling of the (true, permuted) pairing, '
                    + str(NULL_DRAWS) + ' draws, seed ' + str(NULL_SEED)
                    + '; per-step null median/q90 over cross-question pairs.',
            'bootstrap': 'Whole-question cluster bootstrap stratified by source_group, '
                         + str(BOOT_DRAWS) + ' draws, seed ' + str(BOOT_SEEDS['first']) + '.',
            'preregistered_criteria': {
                'delayed': 'median(tau_kc) - median(tau_ab) > 0 with bootstrap CI95 excluding 0',
                'diffuse': 'slope_kc - slope_ab > 0 (log m grows more slowly for knowledge_chain) '
                           'with bootstrap CI95 excluding 0',
                'carrier_in_trajectory_supported': 'delayed OR diffuse; both => strong',
                'rejected_if': 'knowledge_chain divergence profile statistically indistinguishable '
                               'from attribute_binding or earlier/sharper'}})

    # ---- load all seven worlds ----
    worlds = {v: load_records(v) for v in VARIANTS}
    ids = set(worlds['native'])
    for v in VARIANTS:
        assert set(worlds[v]) == ids
    families = {sid: worlds['native'][sid]['family'] for sid in ids}
    fam_count = defaultdict(int)
    for f in families.values():
        fam_count[f] += 1
    assert all(fam_count[f] == 64 for f in CONTROLLED_FAMILIES) and fam_count[NATURAL_FAMILY] == 192
    controlled = sorted(sid for sid in ids if families[sid] in CONTROLLED_FAMILIES)
    natural = sorted(sid for sid in ids if families[sid] == NATURAL_FAMILY)
    everyone = controlled + natural
    vec = VectorCache()

    def vget(variant, sid):
        return vec.get(variant, sid, worlds[variant][sid])

    # ---- per-question paired divergence (preregistered pairs, stability pair, native refs) ----
    per_q = {}
    pair_list = WORLD_PAIRS + [STABILITY_PAIR] + NATIVE_REFS
    for pair in pair_list:
        wA, wB = pair
        for sid in everyone:
            ra, rb = worlds[wA][sid], worlds[wB][sid]
            per_q[(pair, sid)] = paired_divergence(
                vget(wA, sid), ra['generated_ids'], vget(wB, sid), rb['generated_ids'])

    # ---- family reshuffle null (per preregistered world pair) ----
    rng = np.random.default_rng(NULL_SEED)
    null_med, null_q90 = {}, {}
    for pair in WORLD_PAIRS:
        wA, wB = pair
        for fam in CONTROLLED_FAMILIES + [NATURAL_FAMILY]:
            members = [s for s in everyone if families[s] == fam]
            samples = defaultdict(list)
            for _ in range(NULL_DRAWS):
                perm = rng.permutation(members)
                for i, sid in enumerate(members):
                    other = perm[i]
                    if other == sid:
                        continue
                    dvals = cross_divergence(vget(wA, sid), vget(wB, other))
                    for t in range(dvals.shape[0]):
                        samples[t].append(float(dvals[t]))
            for t, vals in sorted(samples.items()):
                null_med[(pair, fam, t)] = float(np.median(vals))
                null_q90[(pair, fam, t)] = float(np.quantile(vals, 0.9))

    # ---- per-question tau (right-censored) and slopes ----
    qrows = []
    for pair in pair_list:
        has_null = pair in WORLD_PAIRS
        for sid in everyone:
            fam = families[sid]
            d = per_q[(pair, sid)]
            comp = d['comparable']
            tau = None
            if has_null:
                for t in range(comp):
                    if (pair, fam, t) not in null_med:
                        break
                    if d['m'][t] > TAU_FACTOR * null_med[(pair, fam, t)]:
                        tau = t
                        break
            tau_censored = comp if (has_null and tau is None) else tau
            window = d['m'][:min(comp, SLOPE_WINDOW)]
            slope = None
            if window.shape[0] >= 3 and float(window.max()) > 0:
                logm = np.log(np.maximum(window, 1e-300))
                slope = float(np.polyfit(np.arange(logm.shape[0], dtype=np.float64), logm, 1)[0])
            qrows.append({'pair': str(pair), 'sample_id': sid, 'family': fam,
                          'kind': worlds['native'][sid]['kind'],
                          'comparable': comp, 'token_div': d['token_div'],
                          'n_a': d['n_a'], 'n_b': d['n_b'],
                          'm0': float(d['m'][0]) if comp > 0 else None,
                          'm1': float(d['m'][1]) if comp > 1 else None,
                          'm2': float(d['m'][2]) if comp > 2 else None,
                          'cosd0': float(d['cosd'][0]) if comp > 0 else None,
                          'tau_postnorm': tau, 'tau_censored': tau_censored, 'slope': slope,
                          'internal_precedes_behaviour': bool(
                              tau is not None and (d['token_div'] is None or tau < d['token_div']))})

    # ---- per-family curves (median over questions with that step) ----
    def family_curve(pair, sids, field):
        med, cnt = [], []
        for t in range(CURVE_T):
            vals = [float(per_q[(pair, s)][field][t]) for s in sids
                    if per_q[(pair, s)][field].shape[0] > t]
            med.append(float(np.median(vals)) if vals else None)
            cnt.append(len(vals))
        return {'median': med, 'valid_counts': cnt}

    curve_targets = CONTROLLED_FAMILIES + [NATURAL_FAMILY, 'all_controlled']
    curves = {}
    for pair in pair_list:
        blob = {}
        for fam in curve_targets:
            sids = everyone if fam == 'all_controlled' else \
                [s for s in everyone if families[s] == fam]
            blob[fam] = {f: family_curve(pair, sids, f) for f in ('m', 'cosd', 'relrms')}
        curves[str(pair)] = blob

    # ---- preregistered contrast: knowledge_chain vs attribute_binding ----
    primary = WORLD_PAIRS[0]

    def stratified_draw(rng):
        groups = defaultdict(list)
        for sid in controlled:
            groups[worlds['native'][sid]['source_group']].append(sid)
        out = []
        for members in groups.values():
            out.extend(members[rng.integers(len(members))] for _ in members)
        return out

    def stats_from(sids):
        kc = [r for r in qrows if r['pair'] == str(primary)
              and r['sample_id'] in set(sids) and r['family'] == 'knowledge_chain']
        ab = [r for r in qrows if r['pair'] == str(primary)
              and r['sample_id'] in set(sids) and r['family'] == 'attribute_binding']

        def med(rows, key):
            vals = [r[key] for r in rows if r[key] is not None]
            return float(np.median(vals)) if vals else None
        kc_tau, ab_tau = med(kc, 'tau_censored'), med(ab, 'tau_censored')
        kc_slope, ab_slope = med(kc, 'slope'), med(ab, 'slope')
        kc_m0, ab_m0 = med(kc, 'm0'), med(ab, 'm0')
        kc_m1, ab_m1 = med(kc, 'm1'), med(ab, 'm1')
        kc_m2, ab_m2 = med(kc, 'm2'), med(ab, 'm2')
        return {'d_tau_censored_median': None if kc_tau is None or ab_tau is None else kc_tau - ab_tau,
                'd_slope_median': None if kc_slope is None or ab_slope is None else kc_slope - ab_slope,
                'd_m0': None if kc_m0 is None or ab_m0 is None else kc_m0 - ab_m0,
                'd_m1': None if kc_m1 is None or ab_m1 is None else kc_m1 - ab_m1,
                'd_m2': None if kc_m2 is None or ab_m2 is None else kc_m2 - ab_m2,
                'kc_internal_rate': float(np.mean([r['internal_precedes_behaviour'] for r in kc])),
                'ab_internal_rate': float(np.mean([r['internal_precedes_behaviour'] for r in ab])),
                'kc_no_token_divergence_rate': float(np.mean([r['token_div'] is None for r in kc])),
                'ab_no_token_divergence_rate': float(np.mean([r['token_div'] is None for r in ab]))}

    point = stats_from(controlled)
    boots = {k: [] for k in point}
    rng1 = np.random.default_rng(BOOT_SEEDS['first'])
    for _ in range(BOOT_DRAWS):
        s = stats_from(stratified_draw(rng1))
        for k in boots:
            boots[k].append(s[k])
    ci = {}
    for k, vals in boots.items():
        vals = [v for v in vals if v is not None]
        ci[k] = [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]

    def ci_excludes(v, interval):
        if v is None:
            return False
        return not (interval[0] <= v <= interval[1])

    delayed = (point['d_tau_censored_median'] is not None
               and point['d_tau_censored_median'] > 0
               and ci_excludes(point['d_tau_censored_median'], ci['d_tau_censored_median']))
    diffuse = (point['d_slope_median'] is not None
               and point['d_slope_median'] > 0
               and ci_excludes(point['d_slope_median'], ci['d_slope_median']))
    criteria = {'point': point, 'bootstrap_ci95': ci,
                'delayed': delayed, 'diffuse': diffuse,
                'carrier_in_trajectory_supported': delayed or diffuse,
                'strength': 'strong' if (delayed and diffuse) else
                            ('supported' if (delayed or diffuse)
                             else 'not_supported_by_preregistered_criteria')}

    # ---- band decomposition (B0/B1/B2+ energy shares on the comparable prefix) ----
    bands = {}
    for pair in WORLD_PAIRS:
        for fam in CONTROLLED_FAMILIES:
            acc = {'B0': [], 'B1': [], 'B2p': []}
            for sid in controlled:
                if families[sid] != fam:
                    continue
                d = per_q[(pair, sid)]
                tot = float(d['m'].sum())
                if tot <= 0:
                    continue
                acc['B0'].append(float(d['m'][:1].sum() / tot))
                acc['B1'].append(float(d['m'][1:2].sum() / tot))
                acc['B2p'].append(float(d['m'][2:].sum() / tot))
            bands[str(pair) + '/' + fam] = {k: {'median': float(np.median(v)) if v else None,
                                                'n': len(v)} for k, v in acc.items()}

    # ---- layer x step divergence for the 44 frozen expressions ----
    layer_data = {}
    for pair in WORLD_PAIRS:
        wA, wB = pair
        keep = [sid for sid in everyone
                if worlds[wA][sid].get('full_hidden_collected')
                and worlds[wB][sid].get('full_hidden_collected')]
        per_fam = {'all': [], 'knowledge_chain': [], 'attribute_binding': []}
        for sid in keep:
            comp = per_q[(pair, sid)]['comparable']
            if comp == 0:
                continue
            za = np.load(worlds[wA][sid]['field']['physical_path'], allow_pickle=False)
            zb = np.load(worlds[wB][sid]['field']['physical_path'], allow_pickle=False)
            ha = unbits(za['all_hidden_BF16'][:comp]).astype(np.float64)
            hb = unbits(zb['all_hidden_BF16'][:comp]).astype(np.float64)
            energy = 0.5 * ((ha ** 2).mean(-1) + (hb ** 2).mean(-1))
            nd = ((ha - hb) ** 2).mean(-1) / np.maximum(energy, 1e-12)
            fam = families[sid]
            per_fam['all'].append(nd[:LAYER_REPORT_T])
            if fam in per_fam:
                per_fam[fam].append(nd[:LAYER_REPORT_T])
            del ha, hb, za, zb
        blob = {}
        for key, arrs in per_fam.items():
            if not arrs:
                blob[key] = None
                continue
            tmin = min(a.shape[0] for a in arrs)
            stack = np.stack([a[:tmin] for a in arrs])
            med = np.median(stack, axis=0)
            first = []
            for L in range(med.shape[1]):
                col = med[:, L]
                base = float(np.median(col))
                first.append(next((int(t) for t in range(col.shape[0])
                                   if col[t] > TAU_FACTOR * base), None))
            blob[key] = {'n_questions': len(arrs), 'steps_reported': int(tmin), 'layers': 37,
                         'layer_step_rel_energy_median': med.T.tolist(),
                         'layer_first_threshold_step': first}
        blob['frozen_subset_ids'] = keep
        layer_data[str(pair)] = blob

    # ---- summary of the token-divergence boundary ----
    token_boundary = {}
    for pair in WORLD_PAIRS:
        for fam in CONTROLLED_FAMILIES:
            rows = [r for r in qrows if r['pair'] == str(pair) and r['family'] == fam]
            divs = [r['token_div'] for r in rows if r['token_div'] is not None]
            token_boundary[str(pair) + '/' + fam] = {
                'no_token_divergence_rate': float(np.mean([r['token_div'] is None for r in rows])),
                'token_div_median': float(np.median(divs)) if divs else None,
                'n': len(rows)}

    # ---- stability pair scale (run/material-generation noise reference) ----
    stability = {}
    for fam in CONTROLLED_FAMILIES:
        sids = [s for s in controlled if families[s] == fam]
        vals = []
        for sid in sids:
            d = per_q[(STABILITY_PAIR, sid)]
            if d['comparable'] > 0:
                vals.append(float(d['m'][0]))
        stability[fam] = {'m0_median': float(np.median(vals)) if vals else None, 'n': len(vals)}

    value = {'timestamp': stamp(), 'source': snapshot(__file__),
             'all_passed': True, 'partial': partial,
             'status': 'P4 carrier test complete on preregistered criteria',
             'model': 'qwen4',
             'questions': {'controlled': len(controlled), 'natural': len(natural),
                           'family_counts': dict(fam_count)},
             'world_pairs': [str(p) for p in WORLD_PAIRS],
             'stability_pair': str(STABILITY_PAIR),
             'null': {'draws': NULL_DRAWS, 'seed': NULL_SEED,
                      'median_keys': len(null_med), 'q90_keys': len(null_q90)},
             'curves': curves, 'criteria': criteria, 'bands': bands,
             'token_boundary': token_boundary, 'stability_pair_m0_by_family': stability,
             'layers': layer_data,
             'question_scores_npz': 'phase2750/qwen4/question_scores.npz',
             'question_scores_meta': 'phase2750/qwen4/question_scores_meta.json',
             'seconds': round(time.monotonic() - START, 1),
             'limits': ['qwen4 only: glm4/qwen14 own-history runs exist for the native world only, '
                        'so no paired-world divergence is defined for them in Phase2747 data.',
                        'Divergence statistics are same-prefix (identical generated token ids across '
                        'worlds); steps after behavioural divergence are excluded from the causal '
                        'readout. Post-divergence behaviour is reported only through token_div.',
                        'tau is right-censored at the comparable-prefix length when the threshold is '
                        'never crossed; medians use the censored values.',
                        'The layer x step panel covers the 44 frozen expressions of Phase2747 '
                        '(4 per controlled family per world pair); it is descriptive, not an '
                        'independently powered test.',
                        'natural_cmrc has no controlled answer structure and stays descriptive.',
                        'No model loads; all quantities derive from immutable Phase2747 fields.',
                        'The 2747/2748 material generations are two deployment generations; their '
                        'stability-pair difference is measured and reported as a run-noise scale, '
                        'not averaged into the primary contrast.']}

    # ---- per-question scores (fixed schema, immutable npz) ----
    pair_code = {str(p): i for i, p in enumerate(pair_list)}
    fam_code = {f: i for i, f in enumerate(CONTROLLED_FAMILIES + [NATURAL_FAMILY])}
    sid_index = {s: i for i, s in enumerate(sorted(ids))}
    cols = defaultdict(list)
    for r in qrows:
        cols['pair_code'].append(pair_code[r['pair']])
        cols['sample_id_idx'].append(sid_index[r['sample_id']])
        cols['family_code'].append(fam_code[r['family']])
        cols['comparable'].append(r['comparable'])
        cols['token_div'].append(-1 if r['token_div'] is None else r['token_div'])
        for k in ('m0', 'm1', 'm2', 'cosd0', 'slope'):
            cols[k].append(np.nan if r[k] is None else r[k])
        cols['tau_censored'].append(-1 if r['tau_censored'] is None else r['tau_censored'])
        cols['internal_precedes_behaviour'].append(int(r['internal_precedes_behaviour']))
    npz(folder / 'question_scores.npz',
        **{k: np.asarray(v) for k, v in cols.items()})
    save(folder / 'question_scores_meta.json',
         {'pair_code': {str(v): k for k, v in pair_code.items()},
          'family_code': {str(v): k for k, v in fam_code.items()},
          'sample_ids': sorted(ids),
          'sentinel': {'token_div -1': 'sequences identical on the whole overlap',
                       'tau_censored -1': 'no null defined for this pair (stability/native pairs)',
                       'NaN': 'step beyond comparable prefix or short slope window'}})
    save(finish, value)
    ledger('phase2750_knowledge_chain_dynamics', value['seconds'])
    print('PHASE2750_KNOWLEDGE_CHAIN_DYNAMICS', criteria['strength'],
          'delayed', delayed, 'diffuse', diffuse, elapsed(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true')
    main(parser.parse_args().partial)
