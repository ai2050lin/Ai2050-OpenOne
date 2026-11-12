"""Per-edit prediction inside the block-24 transition (Claude Phase 417).  Qwen3-14B BF16.

Phase 416: a class rule predicts about three quarters of the block-24 exchanges because the model splits inside a class.
Here every edit gets its own predictor, read from the graph edge N1 (name tokens -> answer position through heads
25.18, 25.24, 25.15):

  x = score(P) - score(B) at the answer position after a NARROW edit: in block 25, for these three heads and the
      attention row of the answer position only, keys and values of the name tokens and of the holder positions are
      taken from the donor (Phase 415, arm nat|NH:C).  P = record named by the donor pointers, B = base answer.
  prediction for the BROAD edit (both name tokens exchanged as residual states at the input of block 24, seen by every
      position and every later block): the answer moves to P iff x > T.

T is one number, fixed on the opened panel holdout_h25 (Phase 415), where both edits were run on the same donors.
Stages: smoke -> prereg -> gate -> run.
"""
import argparse, hashlib, json, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_binding_design as D
import source_binding_pointer as PT
import source_binding_nameloc as NL
import source_binding_fidelity3 as F3
import source_binding_h2518 as H25

SRC, MODEL = D.OUT, 'qwen14'
H25_OUT = H25.OUT
OUT = D.ROOT / 'tests/glm5/result/source_binding_peredit_v1'
RUN, SEED = 'SOURCE-BINDING-PEREDIT-20261112-v1', 202611120417
SPLIT, FAMILIES, REPS = 'holdout_per', 48, 4
C = H25.C
FEATS = {'xNH': 'x|NH:C', 'xN': 'x|N:C', 'xsole': 'x|soleNH:C'}
CLASSES = ['P=I', 'I=B', '3way']
OTHER_PANELS = F3.OTHER_PANELS + ['source_binding_fidelity_v3/holdout_fid3']
CRITERIA = {
    'model': 'Qwen3-14B original BF16, CPU byte reads, CUDA layer streaming. GLM-4-9B is not run (failed the behaviour gate of this task).',
    'material': 'layout-factorial families as in Phase 405. holdout_per: 48 families (768 prompts). Entity names occur in no earlier '
                'panel of this line; the 16 discovery code words, 4 seen frames.',
    'broad_edit': 'as in Phases 412-416: at the input of block 24 the residual states of both question name tokens are replaced by '
                  'those of a donor prompt of the same family; 4 seeded donor relations per item.',
    'narrow_edit': 'as in Phase 415 (arm nat|NH:C), same donors: in block 25, for heads 25.18, 25.24, 25.15 and the attention row of '
                   'the answer position only, keys and values of the name tokens and of the holder positions are those of the donor '
                   'in its unedited run. Nothing else is edited.',
    'predictor': 'x = score(P) - score(B) after the narrow edit; the broad edit is predicted to move the answer to P iff x > T. T is '
                 'chosen on the opened panel holdout_h25 as the candidate (97 quantiles of x from 0.02 to 0.98) with the highest '
                 'accuracy and is frozen in prereg.json.',
    'comparisons': {'class rule': 'Phase 416: P iff P = I', 'none': 'the answer never moves',
                    'natural score': 'score(P) - score(B) of the unedited base run (no edit at all); AUC only',
                    'xN, xsole': 'the narrow edit with the name tokens only / in the sole background; AUC only, descriptive'},
    'outcome': 'on conflict edits (P != B): follow = the forced choice of the broad edit is P. AUC of x for follow, overall and inside '
               'each class (P = I, I = B, all different); accuracy of the frozen rule; family-bootstrap 95 % intervals (300 draws for '
               'AUC, 1000 for accuracies).',
    'controls': 'tail recomputation: the no-edit arm equals the base forward (same choice, score difference <= 1.0); exchanging with '
                'the prompt itself changes nothing; the sole background alone keeps accuracy >= 0.90.',
    'two_pass': 'gate stage: unmodified forward and the v1 behaviour gate. Edits run only if the gate passed.',
    'predictions_on_holdout': {
        'P1_ranking': 'AUC of x >= 0.80 over all conflict edits and >= 0.75 inside each class. not_supported if the overall AUC < 0.70.',
        'P2_frozen_rule': 'accuracy of the frozen rule >= 0.75; minus the class rule >= 0.08 with a lower bound > 0; minus none with a '
                          'lower bound > 0. not_supported if either lower bound <= 0.',
        'P3_beyond_unedited_scores': 'AUC of x minus AUC of the natural score has a family-bootstrap lower bound > 0. not_supported '
                                     'otherwise.'},
    'not_covered': 'the single-name sites and blocks 23 and 25 (no narrow edit was run for them on an opened panel); a predictor that '
                   'needs no edit at all (the unedited scores rank poorly); other tasks and models. The predictor is itself an '
                   'intervention on three heads: it shows that the route through these heads carries most of the per-edit '
                   'variation of the broad edit, not that it is the only route.',
    'disclosure': 'predictor and threshold were chosen on the opened panel holdout_h25 after looking at several candidates offline '
                  '(unedited scores, base and donor margins, the narrow edit in three variants); on that panel the rule is fitted. '
                  'The engine is the unchanged Phase 415 engine, smoke-tested on 2 exposed families of pilot_name.',
}


def save(name, body):
    from deploy.storage_policy import check_write
    p = OUT / name
    check_write(p, 1 << 20, operation='来源绑定逐条预测产物')
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(body, ensure_ascii=False, indent=1, default=lambda x: x.item() if isinstance(x, np.generic) else x), encoding='utf-8')
    t.replace(p)


def build_materials():
    taken = {n for rows in D.read(SRC / f'{MODEL}-materials.json')['splits'].values() for r in rows for n in r['names']}
    for p in PT.PANELS + [PT.OUT / 'pilot_layout-materials.json', PT.OUT / 'holdout_layout-materials.json',
                          NL.OUT / 'pilot_name-materials.json',
                          NL.OUT / 'holdout_name-materials.json'] + [D.ROOT / 'tests/glm5/result' / f'{x}-materials.json' for x in OTHER_PANELS]:
        taken |= {n for r in D.read(p) for n in r['names']}
    PT.SPLITS[SPLIT] = FAMILIES
    return PT.build_materials(SPLIT, taken)


def final_arms(rows):
    H25.RUN, H25.SEED = RUN, SEED                 # donor draws of this run (the stream of the Phase 415 module contains its run id)
    dons, rels = H25.draws(rows)
    both = [['names', C], ['holders', C]]
    arms = {'sole': {'hide': H25.SOLE}, 'self:NH:C': {'hide': [], 'patch': both, 'donor': list(range(len(rows)))}}
    for k, don in enumerate(dons):
        arms[f'swap#{k}'] = {'hide': [], 'swap': True, 'donor': don, 'rel': rels[k]}
        arms[f'x|NH:C#{k}'] = {'hide': [], 'patch': both, 'donor': don, 'rel': rels[k]}
        arms[f'x|N:C#{k}'] = {'hide': [], 'patch': [['names', C]], 'donor': don, 'rel': rels[k]}
        arms[f'x|soleNH:C#{k}'] = {'hide': H25.SOLE, 'patch': both, 'donor': don, 'rel': rels[k]}
    return arms


def engine(rows, arms=None, stage='run'):
    H25.OUT = OUT
    return H25.engine(rows, arms, stage)


def auc(x, y):
    x, y = np.asarray(x, float), np.asarray(y, bool)
    if y.all() or not y.any():
        return float('nan')
    order = np.argsort(x, kind='stable')
    r = np.empty(len(x))
    r[order] = np.arange(1, len(x) + 1)
    d = defaultdict(list)
    for i, v in enumerate(x):
        d[v].append(i)
    for idx in d.values():
        if len(idx) > 1:
            r[idx] = r[idx].mean()
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def table(rows, arms, items, swap='swap', feats=FEATS):
    """One row per conflict edit of the broad arm: family, class, follow, the features."""
    fam_id = {f: i for i, f in enumerate(sorted({r['family'] for r in rows}))}
    T = defaultdict(list)
    for k in range(REPS):
        a = arms[f'{swap}#{k}']
        for i, (r, j) in enumerate(zip(rows, a['donor'])):
            B, P, I = F3.records(r, rows[j], 'names', 24)
            c = F3.klass(B, P, I)
            if not c:
                continue
            T['fam'].append(fam_id[r['family']]); T['cls'].append(c); T['follow'].append(items[f'{swap}#{k}'][i]['pred_record'] == P)
            b = items['base'][i]['scores']
            T['nat'].append(b[P] - b[B])
            for f, arm in feats.items():
                s = items[f'{arm}#{k}'][i]['scores']
                T[f].append(s[P] - s[B])
    return {k: np.array(v) for k, v in T.items()}, len(fam_id)


def threshold(x, y):
    cand = np.quantile(x, np.linspace(.02, .98, 97))
    acc = [float(np.mean((x > t) == y)) for t in cand]
    i = int(np.argmax(acc))
    return float(cand[i]), acc[i]


def frozen():
    rows, arms, items = (D.read(H25_OUT / x) for x in (f'{H25.SPLIT}-materials.json', 'design.json', 'items.json'))
    T, _ = table(rows, arms, items, feats={'xNH': 'nat|NH:C'})
    t, acc = threshold(T['xNH'], T['follow'])
    return {'T': round(t, 4), 'accuracy_on_holdout_h25': round(acc, 4), 'auc_on_holdout_h25': round(auc(T['xNH'], T['follow']), 4),
            'edits': int(len(T['follow'])), 'items_sha256': D.sha(H25_OUT / 'items.json')}


def summarise(results, arms, rows, T0):
    T, nf = table(rows, arms, results)
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, nf, (1000, nf))
    y, fam = T['follow'], T['fam']
    members = [np.flatnonzero(fam == f) for f in range(nf)]

    def rate(v, keep=None):
        v = np.asarray(v, float)
        keep = np.ones(len(v), bool) if keep is None else keep
        per, cnt = np.bincount(fam[keep], weights=v[keep], minlength=nf), np.bincount(fam[keep], minlength=nf)
        bs = per[draws].sum(1) / np.maximum(cnt[draws].sum(1), 1)
        return {'rate': round(float(v[keep].mean()), 4), 'ci95': [round(float(np.quantile(bs, .025)), 4), round(float(np.quantile(bs, .975)), 4)]}

    def auc_ci(fn):
        vals = []
        for d in draws[:300]:
            idx = np.concatenate([members[f] for f in d])
            vals.append(fn(idx))
        return [round(float(np.nanquantile(vals, .025)), 4), round(float(np.nanquantile(vals, .975)), 4)]
    allidx = np.arange(len(y))
    out = {'conflict_edits': int(len(y)), 'follow': rate(y), 'auc': {}, 'auc_by_class': {}, 'class_follow': {}}
    for f in ['xNH', 'xN', 'xsole', 'nat']:
        out['auc'][f] = {'auc': round(auc(T[f], y), 4), 'ci95': auc_ci(lambda idx, f=f: auc(T[f][idx], y[idx]))}
    out['auc_xNH_minus_nat'] = {'rate': round(auc(T['xNH'], y) - auc(T['nat'], y), 4),
                                'ci95': auc_ci(lambda idx: auc(T['xNH'][idx], y[idx]) - auc(T['nat'][idx], y[idx]))}
    for c in CLASSES:
        k = T['cls'] == c
        out['auc_by_class'][c] = {'n': int(k.sum()), **{f: round(auc(T[f][k], y[k]), 4) for f in ['xNH', 'xN', 'xsole', 'nat']}}
        out['class_follow'][c] = round(float(y[k].mean()), 4)
    rule, cls, none = (T['xNH'] > T0) == y, (T['cls'] == 'P=I') == y, ~y
    best_t, best_acc = threshold(T['xNH'], y)
    out['accuracy'] = {'frozen_rule': rate(rule), 'class_rule': rate(cls), 'none': rate(none),
                       'frozen_minus_class': rate(rule.astype(float) - cls.astype(float)),
                       'frozen_minus_none': rate(rule.astype(float) - none.astype(float)),
                       'frozen_rule_by_class': {c: round(float(rule[T['cls'] == c].mean()), 4) for c in CLASSES},
                       'predicted_follow_share': round(float((T['xNH'] > T0).mean()), 4),
                       'best_threshold_on_this_panel': {'T': round(best_t, 4), 'accuracy': round(best_acc, 4), 'note': 'descriptive; fitted here'}}
    q = np.quantile(T['xNH'], np.linspace(0, 1, 11))
    out['follow_by_decile_of_x'] = [{'x_from': round(float(q[i]), 2), 'x_to': round(float(q[i + 1]), 2),
                                     'follow': round(float(y[(T['xNH'] >= q[i]) & ((T['xNH'] < q[i + 1]) | (i == 9))].mean()), 4)} for i in range(10)]
    return out


def verdicts(s, meta):
    tri = lambda ok, bad: 'supported' if ok else 'not_supported' if bad else 'inconclusive'
    v = {'controls_valid': bool(meta['controls_ok'])}
    names = list(CRITERIA['predictions_on_holdout'])
    if not v['controls_valid']:
        return {**v, **{k: 'controls_invalid' for k in names}}
    a = s['auc']['xNH']['auc']
    v['P1_ranking'] = tri(a >= .80 and min(c['xNH'] for c in s['auc_by_class'].values()) >= .75, a < .70)
    A = s['accuracy']
    dc, dn = A['frozen_minus_class'], A['frozen_minus_none']
    v['P2_frozen_rule'] = tri(A['frozen_rule']['rate'] >= .75 and dc['rate'] >= .08 and dc['ci95'][0] > 0 and dn['ci95'][0] > 0,
                              dc['ci95'][0] <= 0 or dn['ci95'][0] <= 0)
    v['P3_beyond_unedited_scores'] = tri(s['auc_xNH_minus_nat']['ci95'][0] > 0, True)
    return v


def smoke():
    from deploy.storage_policy import checked_dir
    checked_dir(OUT, 1 << 30, operation='来源绑定逐条预测产物')
    rows = [r for r in D.read(NL.OUT / 'pilot_name-materials.json') if int(r['family'][-4:]) < 2]
    arms = final_arms(rows)
    results, meta = engine(rows, arms, stage='smoke')
    s = summarise(results, arms, rows, 0.0)
    print(json.dumps({'meta': {k: v for k, v in meta.items() if k != 'attention_mass'}, 'arms': len(arms), 'conflict_edits': s['conflict_edits']}), flush=True)


DEPS = (PT, NL, F3, H25)


def prereg():
    from deploy.storage_policy import checked_dir
    from tools import provenance
    from tools.power_analysis import check_phase
    checked_dir(OUT, 1 << 30, operation='来源绑定逐条预测产物')
    assert not (OUT / 'prereg.json').exists()
    plan = D.checked(MODEL)
    fz = frozen()
    rows = build_materials()
    assert len(rows) == FAMILIES * 16 and len({tuple(r['tokens']) for r in rows}) == len(rows)
    arms = final_arms(rows)
    save(f'{SPLIT}-materials.json', rows)
    save('design.json', arms)
    body = {'schema': 'source_binding_peredit_prereg.v1', 'run_id': RUN, 'seed': SEED, 'created_at': datetime.now().astimezone().isoformat(),
            'base_plan_sha256': D.sha(SRC / 'plan.json'), 'model_snapshot': plan['models'][MODEL], 'code_sha256': D.sha(__file__),
            'dependency_sha256': {m.__name__: D.sha(m.__file__) for m in DEPS},
            'criteria': CRITERIA, 'frozen': fz,
            'materials': {'sha256': D.sha(OUT / f'{SPLIT}-materials.json'), 'families': FAMILIES, 'rows': len(rows)},
            'design': {'sha256': D.sha(OUT / 'design.json'), 'arms': len(arms)},
            'power': {'unit': 'paired binary outcome per item (frozen rule right / class rule right); alpha = 0.05',
                      'P2_as_if_one_edit_per_item': {'expected_delta': 0.10, 'p_d': 0.40, **check_phase(0.10, len(rows), p_d=0.40)},
                      'caveat': 'about 2,500 conflict edits clustered in 768 items and 48 families; intervals are family bootstraps'},
            'custody': 'holdout_per is read first by the gate stage (behaviour only) and then once by the run stage.',
            'resources': {'gpu_minutes_estimate': 6, 'disk_GiB_estimate': 0.1, 'models_concurrent': 1,
                          'shape': 'chunks of 192 prompts in one batch (the Phase 415 shape)'},
            'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    body['provenance']['model_revision'] = plan['models'][MODEL]
    save('prereg.json', body)
    print(json.dumps({'prereg_sha256': D.sha(OUT / 'prereg.json'), 'design': body['design'], 'frozen': fz, 'power': body['power']}), flush=True)


def checked():
    p = D.read(OUT / 'prereg.json')
    assert p['code_sha256'] == D.sha(__file__) and all(p['dependency_sha256'][m.__name__] == D.sha(m.__file__) for m in DEPS), \
        'source changed after pre-registration'
    assert p['materials']['sha256'] == D.sha(OUT / f'{SPLIT}-materials.json') and p['design']['sha256'] == D.sha(OUT / 'design.json')
    D.checked(MODEL)
    return p


def gate():
    checked()
    assert not (OUT / 'gate.json').exists(), 'holdout behaviour already read'
    rows = D.read(OUT / f'{SPLIT}-materials.json')
    results, meta = engine(rows, stage='gate')
    save('gate-items.json', results['base'])
    g = D.behavior(rows, results['base'])
    save('gate.json', {'schema': 'source_binding_peredit_gate.v1', 'run_id': RUN, 'items_sha256': D.sha(OUT / 'gate-items.json'), **g, **meta})
    print(json.dumps({k: g[k] for k in ('passed', 'forced_accuracy', 'top1_accuracy')}), flush=True)


def run():
    from tools import provenance
    pre = checked()
    assert not (OUT / 'report.json').exists(), 'holdout already used'
    g = D.read(OUT / 'gate.json')
    rows, arms = D.read(OUT / f'{SPLIT}-materials.json'), D.read(OUT / 'design.json')
    names = list(CRITERIA['predictions_on_holdout'])
    if g['passed']:
        results, meta = engine(rows, arms, stage='run')
        meta.pop('attention_mass', None)
        save('items.json', {k: [{'id': x['id'], 'pred_record': x['pred_record'], 'scores': x['scores']} for x in v] for k, v in results.items()})
        base = D.read(OUT / 'gate-items.json')
        meta['base_replay_max_score_difference'] = max(abs(a - b) for x, y in zip(results['base'], base) for a, b in zip(x['scores'], y['scores']))
        s = summarise(results, arms, rows, pre['frozen']['T'])
        v = verdicts(s, meta)
        print(json.dumps({'meta': meta, 'verdicts': v, 'summary': s}), flush=True)
    else:
        s, meta, v = {'note': 'behaviour gate failed; no edit was run'}, {}, {k: 'behavior_ineligible' for k in names}
    report = {'schema': 'source_binding_peredit_report.v1', 'run_id': RUN, 'split': SPLIT, 'gate_passed': g['passed'],
              'prereg_sha256': D.sha(OUT / 'prereg.json'), 'gate_sha256': D.sha(OUT / 'gate.json'),
              'items_sha256': D.sha(OUT / 'items.json') if g['passed'] else None, 'criteria': pre['criteria'], 'frozen': pre['frozen'],
              'verdicts': v, 'summary': s, **meta, 'formal_evidence_promotion': 0,
              'custody': 'holdout_per read once by the gate stage and once by this run',
              'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    save('report.json', report)
    path = provenance.write_result('SOURCE-BINDING-peredit-holdout', {'verdicts': v, 'gate_passed': g['passed'],
                                                                    'report_sha256': D.sha(OUT / 'report.json')},
                                   model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)
    print(json.dumps({'verdicts': v, 'result': str(path)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['smoke', 'prereg', 'gate', 'run'])
    {'smoke': smoke, 'prereg': prereg, 'gate': gate, 'run': run}[ap.parse_args().stage]()
