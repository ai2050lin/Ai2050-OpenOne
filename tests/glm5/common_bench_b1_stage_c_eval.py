"""Stage C one-shot evaluation on confirm (COMMON-BENCH-B1-STAGE-C-20261110-v1).

Applies the estimators frozen in stagec-frozen.json to the confirm split and judges P1-P3 by the criteria in
stagec-prereg.json.  Written and committed before confirm was captured.  Refuses to run twice.
"""
import json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_bench_b1 as B
import common_bench_b1_stage_c as C

OUT = B.OUT
REPS, SEED = 2000, 202611100389


def cluster_ci(hit, e1, e2, n_ent, rng):
    """Accuracy with a 95 % interval from resampling entities (an item counts with the product of its two entity draws)."""
    hit = np.asarray(hit, float)
    if len(hit) == 0:
        return {'n': 0, 'accuracy': None, 'ci95': None}
    vals = []
    for _ in range(REPS):
        m = rng.multinomial(n_ent, np.ones(n_ent) / n_ent)
        w = m[e1] * m[e2]
        if w.sum():
            vals.append((hit * w).sum() / w.sum())
    return {'n': int(len(hit)), 'accuracy': round(float(hit.mean()), 4),
            'ci95': [round(float(np.quantile(vals, .025)), 4), round(float(np.quantile(vals, .975)), 4)]}


def main():
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from tools import provenance
    pre, frozen = C.checked(), B.read(OUT / 'stagec-frozen.json')
    assert not (OUT / 'stagec-confirm-report.json').exists(), 'confirm was already evaluated'
    assert B.sha(OUT / 'stagec-estimators.npz') == frozen['estimators_sha256']
    man = B.read(OUT / 'stagec-native-confirm.json')
    assert all(B.sha(C.store('confirm') / k) == v['sha256'] for k, v in man['files'].items())
    est = np.load(OUT / 'stagec-estimators.npz')
    lab = C.labels('confirm')
    S, L, Cc, e1, e2 = lab['S'], lab['L'], lab['C'], lab['e1'], lab['e2']
    n_ent, rng, start = len(lab['entity_names']), np.random.default_rng(SEED), time.monotonic()
    primary = lab['screened'] & lab['correct']
    sets = {'primary': primary, 'all_items': np.ones(len(S), bool)}
    ci = lambda hit, mask: cluster_ci(hit, e1[mask], e2[mask], n_ent, rng)

    # behaviour gate (stage B rule) on confirm
    cells = {}
    for s in (1, -1):
        for l in (1, -1):
            m = lab['screened'] & (S == s) & (L == l)
            cells[f"{'document' if s == 1 else 'parametric'}_{'en' if l == 1 else 'zh'}"] = round(float(lab['correct'][m].mean()), 4)
    frames = {int(f): float(lab['correct'][lab['screened'] & (lab['frame'] == f)].mean()) for f in np.unique(lab['frame'])}
    per_entity = {lab['entity_names'][e]: round(float(lab['correct'][lab['target_entity'] == e].mean()), 3)
                  for e in range(n_ent) if (lab['target_entity'] == e).any()}
    gate = len(lab['dropped_entities']) <= 0.2 * n_ent and min(cells.values()) >= 0.9 and min(frames.values()) >= 0.9
    behaviour = {'items': int(len(S)), 'forced_all': round(float(lab['correct'].mean()), 4), 'dropped_entities': lab['dropped_entities'],
                 'entities': n_ent, 'items_after_screen': int(lab['screened'].sum()), 'primary_items': int(primary.sum()),
                 'cells_after_screen': cells, 'min_frame_after_screen': round(min(frames.values()), 4),
                 'weak_entities': {k: v for k, v in per_entity.items() if v < 1}, 'gate_passed': bool(gate)}

    # P1 / P2: frozen readouts at the selected layers
    transfers = {}
    for name, layer in frozen['selected_layers'].items():
        h = C.layer('confirm', layer).astype(np.float64)
        var = est[f'{name}__var'].astype(np.float64)
        cond = S if name in ('P1_language_across_source', 'P2b_colour_across_source') else L
        res = {}
        for c in (1, -1):
            for sname, mask in sets.items():
                sel = mask & (cond == -c)
                if name.startswith('P1'):
                    def score(cen_c, dir_c):
                        mp, mn = est[f'{name}__mean_pos_{dir_c}'].astype(np.float64), est[f'{name}__mean_neg_{dir_c}'].astype(np.float64)
                        centre = (mp + mn) / 2 if cen_c == 'own' else est[f'{name}__centre_{cen_c}'].astype(np.float64)
                        return ((h[sel] - centre) @ ((mp - mn) / np.maximum(var, 1e-6)) >= 0) == (L[sel] == 1)
                    hits = {'M1': score('own', c), 'M2': score(-c, c), 'M3': score('own', -c)}
                else:
                    tr, tg = est[f'{name}__centroids_{c}'].astype(np.float64), est[f'{name}__centroids_{-c}'].astype(np.float64)
                    shift = est[f'{name}__centre_{-c}'].astype(np.float64) - tr.mean(0)
                    hits = {'M1': C.nearest(h[sel], tr, var) == Cc[sel], 'M2': C.nearest(h[sel], tr + shift, var) == Cc[sel],
                            'M3': C.nearest(h[sel], tg, var) == Cc[sel]}
                res[f"{'plus' if c == 1 else 'minus'}__{sname}"] = {k: ci(v, sel) for k, v in hits.items()}
        transfers[name] = {'layer': layer, **res}
    B.log('evaluate', 1, 4, start)

    # P3: unseen combination at hs[34]; descriptive curve over layers
    truth = C.cell_index(S, L, Cc)
    p3, curve = {}, {'A1': [], 'A2': []}
    res_cells = [(1, -1, B.COLORS.index('green')), (-1, 1, B.COLORS.index('white'))]
    for layer in range(C.NL):
        h = C.layer('confirm', layer).astype(np.float64)
        rv = est['residual_variance'][layer].astype(np.float64)
        row = {}
        for mi, pw in (('A1', False), ('A2', True)):
            cen = C.CELL_X[pw] @ est['coef_A2' if pw else 'coef_A1'][layer].astype(np.float64)
            pred = C.nearest(h, cen, rv)
            row[mi] = [round(float((pred == truth)[primary & (truth == C.cell_index(np.array(s), np.array(l), np.array(c)))].mean()), 4) for s, l, c in res_cells]
            if layer == frozen['p3_layer']:
                p3[mi] = {}
                for j, (s, l, c) in enumerate(res_cells):
                    cell = truth == C.cell_index(np.array(s), np.array(l), np.array(c))
                    for sname, mask in sets.items():
                        p3[mi][f'reserved_{j}__{sname}'] = ci((pred == truth)[mask & cell], mask & cell)
                    wrong = pred[primary & cell & (pred != truth)]
                    p3[mi][f'reserved_{j}__confusions'] = {str(C.CELLS[int(k)]): int(v) for k, v in zip(*np.unique(wrong, return_counts=True))}
                p3[mi]['seen_cells__primary'] = ci((pred == truth)[primary & ~lab['reserved']], primary & ~lab['reserved'])
            curve[mi].append(row[mi])
    B.log('evaluate', 2, 4, start)

    # P4: one-step update, descriptive
    p4, prev = [], C.layer('confirm', 0).astype(np.float64)[primary]
    Sp, Lp, Cp = S[primary], L[primary], Cc[primary]
    for l in range(35):
        nxt = C.layer('confirm', l + 1).astype(np.float64)[primary]
        inc, tv = est['increment_A2'][l].astype(np.float64), np.maximum(est['total_variance'][l + 1].astype(np.float64), 1e-6)
        cell_inc = C.CELL_X[True] @ inc
        decoded = C.nearest(prev, C.CELL_X[True] @ est['coef_A2'][l].astype(np.float64), est['residual_variance'][l].astype(np.float64))
        preds = {'identity': prev, 'mean_increment': prev + est['mean_increment'][l],
                 'oracle_cell_increment': prev + cell_inc[C.cell_index(Sp, Lp, Cp)], 'decoded_cell_increment': prev + cell_inc[decoded]}
        raw = {k: float(((nxt - v) ** 2).mean()) for k, v in preds.items()}
        std = {k: float((((nxt - v) ** 2) / tv).mean()) for k, v in preds.items()}
        p4.append({'from': l, 'to': l + 1, 'decoded_cell_accuracy': round(float((decoded == C.cell_index(Sp, Lp, Cp)).mean()), 4),
                   'raw_ratio_to_mean_increment': {k: round(raw[k] / raw['mean_increment'], 4) for k in raw},
                   'standardised_ratio_to_mean_increment': {k: round(std[k] / std['mean_increment'], 4) for k in std}})
        prev = nxt
    B.log('evaluate', 3, 4, start)

    # answer prediction by predictor group (10-way)
    dlab = C.labels('discovery')
    drows, crows = B.read(OUT / 'discovery-materials.json'), B.read(OUT / 'confirm-materials.json')
    dfit = dlab['screened'] & dlab['correct'] & ~dlab['reserved']
    docs = lambda rows, mask: [' '.join(map(str, r['tokens'])) for r, m in zip(rows, mask) if m]
    vec = CountVectorizer(token_pattern=r'\S+', ngram_range=(1, 2))
    clf = LogisticRegression(max_iter=2000).fit(vec.fit_transform(docs(drows, dfit)), np.array([B.CLASSES.index(r['gold']) for r in drows])[dfit])
    gold = np.array([B.CLASSES.index(r['gold']) for r in crows])
    text_pred = clf.predict(vec.transform(docs(crows, np.ones(len(crows), bool))))
    groups = {'text_only__bigram_logreg': ci(text_pred == gold, sets['all_items']),
              'model_behaviour__forced_choice': ci(lab['correct'], sets['all_items'])}
    for layer in (27, 32, 34, 35):
        h = C.layer('confirm', layer).astype(np.float64)
        dec = C.nearest(h, C.CELL_X[True] @ est['coef_A2'][layer].astype(np.float64), est['residual_variance'][layer].astype(np.float64))
        groups[f'model_state__decoded_cell_hs{layer}'] = ci((dec % 10) == gold, sets['all_items'])

    # verdicts by the pre-registered criteria, primary set
    def both(name, key):
        return [transfers[name][f'{d}__primary'][key]['accuracy'] for d in ('plus', 'minus')]
    m2, m3 = both('P1_language_across_source', 'M2'), both('P1_language_across_source', 'M3')
    verdict = {'P1_language_across_source': 'supported' if min(m2) >= .95 and max(a - b for a, b in zip(m3, m2)) <= .02
               else 'direction_matters' if min(m2) < .95 and min(m3) >= .95 else 'unresolved'}
    for name in ('P2a_colour_across_language', 'P2b_colour_across_source'):
        v = both(name, 'M2')
        verdict[name] = 'supported' if min(v) >= .80 else 'not_supported' if min(v) < .50 else 'inconclusive'
    a2 = [p3['A2'][f'reserved_{j}__primary']['accuracy'] for j in (0, 1)]
    verdict['P3_unseen_combination'] = 'supported' if min(a2) >= .80 else 'not_supported' if min(a2) < .50 else 'inconclusive'
    report = {'schema': 'common_bench_stagec_confirm.v1', 'run_id': C.RUN, 'prereg_sha256': B.sha(OUT / 'stagec-prereg.json'),
              'frozen_sha256': B.sha(OUT / 'stagec-frozen.json'), 'eval_code_sha256': B.sha(__file__),
              'native_manifest_sha256': B.sha(OUT / 'stagec-native-confirm.json'), 'criteria': pre['criteria'],
              'behaviour': behaviour, 'eligible': bool(gate), 'verdicts': verdict, 'transfers': transfers,
              'P3': {'layer': frozen['p3_layer'], 'reserved_cells': ['document x zh x green', 'parametric x en x white'],
                     'weighting_note': 'both models use the residual variance of the A2 discovery fit', **p3,
                     'descriptive_reserved_accuracy_by_layer': curve},
              'P4_descriptive': p4, 'answer_prediction_10way': groups,
              'interval': f'entity-resampling bootstrap, {REPS} draws, seed {SEED}',
              'custody': 'confirm opened once for this report; it cannot serve as a confirmation after any retuning',
              'formal_evidence_promotion': 0, 'elapsed_seconds': round(time.monotonic() - start, 1),
              'provenance': provenance.provenance(model_path=B.MODEL, seed=SEED, tool=C.RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    B.save('stagec-confirm-report.json', report)
    path = provenance.write_result('COMMON-BENCH-B1-stageC-confirm', {'verdicts': verdict, 'behaviour': behaviour,
                                   'report_sha256': B.sha(OUT / 'stagec-confirm-report.json')}, model_path=B.MODEL, seed=SEED, tool=C.RUN)
    print(json.dumps({'verdicts': verdict, 'eligible': bool(gate), 'provenance_result': str(path)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
