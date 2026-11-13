"""Fidelity of graph revision 4 inside the block-24 transition (Claude Phase 422).  Qwen3-14B BF16.

Revision 3 (Phase 416) predicted the exchange of both name tokens at block 24 by a class rule (agreement 0.766 over the
three sites).  Revision 4 predicts it edit by edit from two graph edges:

  N1  heads 25.18, 25.24, 25.15 carry the pointer from the name tokens to the answer position,
  N7  heads of block 27 check that pointer against the content of the name tokens.

  NARROW edit: the three block-25 heads read donor keys / values at name tokens and holders, and block 27 reads donor
        keys / values at the name tokens (answer row of these two blocks only; nothing else is edited).
  rule: the broad edit (both name tokens exchanged as residual states at the input of block 24, seen by every position
        and every later block) gives the answer that the narrow edit gives.  No fitted number.
  Secondary: x27 = score(P) - score(B) after the narrow edit; P iff P != B and x27 > T, T fixed on the opened panel
        holdout_a27 (Phase 421), where both edits were run on the same donors.
Stages: smoke -> prereg -> gate -> run.
"""
import argparse, json, sys
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
import source_binding_peredit as PE
import source_binding_downstream as DN
import source_binding_attn27 as A27

SRC, MODEL = D.OUT, 'qwen14'
OUT = D.ROOT / 'tests/glm5/result/source_binding_fidelity_v4'
RUN, SEED = 'SOURCE-BINDING-FIDELITY-20261112-v4', 202611120422
SPLIT, FAMILIES, REPS = 'holdout_fid4', 64, 4
T417 = -7.75
A27_OUT = A27.OUT                 # the engine wrapper below points A27.OUT at this run
OTHER_PANELS = A27.OTHER_PANELS + ['source_binding_attn27_v1/holdout_a27']
FEATS = {'x27': 'don:N', 'x25': 'full'}
CLASSES = PE.CLASSES
CRITERIA = {
    'model': 'Qwen3-14B original BF16, CPU byte reads, CUDA layer streaming. GLM-4-9B is not run (failed the behaviour gate of this task).',
    'material': 'layout-factorial families as in Phase 405. holdout_fid4: 64 families (1,024 prompts). Entity names occur in no earlier '
                'panel of this line; the 16 discovery code words, 4 seen frames.',
    'broad_edit': 'as in Phases 412-421: at the input of block 24 the residual states of both question name tokens are replaced by '
                  'those of a donor prompt of the same family; 4 seeded donor relations per item (4,096 edits).',
    'predictors': {'graph_v4': 'the forced choice after the narrow edit through edges N1 and N7 (block-25 heads and donor names in '
                               'block 27); no fitted number',
                   'graph_v4_threshold': 'P iff P != B and x27 > T (T frozen in prereg.json, fitted on holdout_a27); else B',
                   'narrow25_answer': 'the forced choice after the narrow edit in the block-25 heads alone (edge N1 only)',
                   'graph_v3': 'Phase 416 class rule: P iff P != B and P = I; else B',
                   'rule_417': 'P iff P != B and x25 > -7.75, x25 = the narrow exchange in the block-25 heads alone (Phase 417)',
                   'none': 'B', 'position': 'P'},
    'narrow_edits': 'x25: block 25, heads 25.18, 25.24, 25.15, attention row of the answer position, keys and values of name tokens '
                    'and holders from the donor in its unedited run. x27: the same, and in block 27 the answer row reads donor keys / '
                    'values at the name tokens (every head). Nothing else is edited.',
    'outcome': 'agreement = the predicted record equals the forced choice of the broad edit, over all 4,096 edits and over the '
               'conflict edits (P != B); AUC of x27 and x25 for follow on conflict edits, overall and by class; family-bootstrap '
               '95 % intervals (1000 draws; 300 for AUC).',
    'controls': 'tail recomputation equals the base forward (same choice, score difference <= 1.0); the narrow edit with the prompt '
                'itself, alone and with its own names as donor in block 27, changes nothing (same bound).',
    'two_pass': 'gate stage: unmodified forward and the v1 behaviour gate. Edits run only if the gate passed.',
    'predictions_on_holdout': {
        'F1_agreement': 'graph_v4 agreement over all edits >= 0.80 and over conflict edits >= 0.77. not_supported if the agreement over '
                        'all edits < 0.75.',
        'F2_beyond_revision_3': 'agreement over all edits: graph_v4 minus graph_v3 >= 0.08 with a lower bound > 0; minus none with a '
                                'lower bound > 0. not_supported if either lower bound <= 0.',
        'F3_block_27_edge_adds': 'over all edits graph_v4 minus narrow25_answer and graph_v4 minus rule_417 have lower bounds > 0, and '
                                 'AUC(x27) - AUC(x25) has a lower bound > 0. not_supported if any lower bound <= 0.',
        'F4_ranking': 'AUC of x27 >= 0.90 over conflict edits and >= 0.88 inside each class. not_supported if the overall AUC < 0.85.'},
    'not_covered': 'the single-name sites and blocks 23 and 25 (revision-3 class rule and follow-rate table, not retested here); name '
                   'tokens at other blocks (0.981 in Phase 416); a predictor without any edit; other tasks and models. x27 is itself '
                   'an intervention on three heads of block 25 and on the name keys / values of block 27: the test shows that these '
                   'two edges carry the per-edit variation of the broad edit, not that they are the only route.',
    'disclosure': 'the rule was chosen on the opened panel holdout_a27 (Phase 421) among the arms run there (narrow edit alone, base '
                  'attention weights, names hidden, donor names, donor names and holders, head sets frozen), each as a thresholded '
                  'score and as the answer of the narrow edit. The threshold rule separates follow from not-follow in 0.876 of the '
                  'conflict edits there, but as a prediction of the record it agrees in 0.780 only, because 0.09 of the broad edits '
                  'answer a third record; the answer of the narrow edit agrees in 0.811 (all edits 0.839) and was made the primary '
                  'rule. Thresholds of F1 were set from these numbers. The engine is the unchanged Phase 421 engine, smoke-tested on '
                  '2 exposed families of pilot_name.',
}


def save(name, body):
    from deploy.storage_policy import check_write
    p = OUT / name
    check_write(p, 1 << 20, operation='来源绑定保真度 v4 产物')
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


def final_arms(rows, run=RUN, seed=SEED):
    H25.RUN, H25.SEED = run, seed                 # donor draws: the stream of the Phase 415 module contains run id and seed
    dons, rels = H25.draws(rows)
    me = list(range(len(rows)))
    arms = {'self': {'donor': me, 'resp': None, 'rep': 'self'}, 'self:don': {'donor': me, 'resp': None, 'rep': 'self', 'm27': {'donor': ['names']}}}
    for k, don in enumerate(dons):
        base = {'donor': don, 'rel': rels[k], 'rep': k}
        arms[f'swap#{k}'] = {'swap': True, 'donor': don, 'rel': rels[k]}
        arms[f'full#{k}'] = {**base, 'resp': None, 'mass': True}
        arms[f'don:N#{k}'] = {**base, 'resp': None, 'm27': {'donor': ['names']}}
    return arms


def engine(rows, arms=None, stage='run'):
    A27.OUT = OUT
    return A27.engine(rows, arms, stage)


def table(rows, arms, items):
    """One row per broad edit: family, B, P, I, class (None when P = B), the model's choice, the features."""
    fam_id = {f: i for i, f in enumerate(sorted({r['family'] for r in rows}))}
    T = defaultdict(list)
    for k in range(REPS):
        for i, (r, j) in enumerate(zip(rows, arms[f'swap#{k}']['donor'])):
            B, P, I = F3.records(r, rows[j], 'names', 24)
            T['fam'].append(fam_id[r['family']]); T['B'].append(B); T['P'].append(P); T['I'].append(I)
            T['cls'].append(F3.klass(B, P, I) or '-'); T['out'].append(items[f'swap#{k}'][i]['pred_record'])
            for f, arm in FEATS.items():
                s = items[f'{arm}#{k}'][i]['scores']
                T[f].append(s[P] - s[B])
                T['ans' + f[1:]].append(items[f'{arm}#{k}'][i]['pred_record'])
    return {k: np.array(v) for k, v in T.items()}, len(fam_id)


def frozen():
    rows, arms, items = (D.read(A27_OUT / x) for x in (f'{A27.SPLIT}-materials.json', 'design.json', 'items.json'))
    T, _ = table(rows, arms, items)
    c = T['P'] != T['B']
    y = (T['out'] == T['P'])[c]
    t, acc = PE.threshold(T['x27'][c], y)
    return {'T': round(t, 4), 'accuracy_on_holdout_a27_conflict': round(acc, 4), 'auc_on_holdout_a27': round(PE.auc(T['x27'][c], y), 4),
            'edits': int(c.sum()), 'items_sha256': D.sha(A27_OUT / 'items.json')}


def summarise(results, arms, rows, T0):
    T, nf = table(rows, arms, results)
    draws = np.random.default_rng(SEED).integers(0, nf, (1000, nf))
    fam, B, P, out = T['fam'], T['B'], T['P'], T['out']
    c = P != B
    pred = {'graph_v4': T['ans27'], 'graph_v4_threshold': np.where(c & (T['x27'] > T0), P, B), 'narrow25_answer': T['ans25'],
            'graph_v3': np.where(c & (P == T['I']), P, B),
            'rule_417': np.where(c & (T['x25'] > T417), P, B), 'none': B, 'position': P}
    r4 = lambda v: round(float(v), 4)

    def rate(v, keep):
        v = np.asarray(v, float)
        per, cnt = np.bincount(fam[keep], weights=v[keep], minlength=nf), np.bincount(fam[keep], minlength=nf)
        bs = per[draws].sum(1) / np.maximum(cnt[draws].sum(1), 1)
        return {'rate': r4(v[keep].mean()), 'ci95': [r4(np.quantile(bs, .025)), r4(np.quantile(bs, .975))]}
    hit = {k: (v == out).astype(float) for k, v in pred.items()}
    every = np.ones(len(out), bool)
    s = {'edits': int(len(out)), 'conflict_edits': int(c.sum()), 'model_choice_changed': r4((out != B).mean())}
    for name, keep in (('all_edits', every), ('conflict_edits_only', c)):
        s[name] = {'agreement': {k: rate(v, keep) for k, v in hit.items()},
                   **{f'graph_v4_minus_{k}': rate(hit['graph_v4'] - hit[k], keep) for k in ('graph_v3', 'rule_417', 'narrow25_answer', 'graph_v4_threshold', 'none')}}
    y = (out == P)[c]
    cf, x27, x25, cls = fam[c], T['x27'][c], T['x25'][c], T['cls'][c]
    members = [np.flatnonzero(cf == f) for f in range(nf)]

    def auc_of(fn):
        vals = [fn(np.concatenate([members[f] for f in d])) for d in draws[:300]]
        return {'auc': r4(fn(np.arange(len(y)))), 'ci95': [r4(np.nanquantile(vals, .025)), r4(np.nanquantile(vals, .975))]}
    s['follow'] = r4(y.mean())
    s['auc'] = {'x27': auc_of(lambda i: PE.auc(x27[i], y[i])), 'x25': auc_of(lambda i: PE.auc(x25[i], y[i])),
                'x27 - x25': auc_of(lambda i: PE.auc(x27[i], y[i]) - PE.auc(x25[i], y[i]))}
    s['auc_by_class'] = {k: {'n': int((cls == k).sum()), 'x27': r4(PE.auc(x27[cls == k], y[cls == k])), 'x25': r4(PE.auc(x25[cls == k], y[cls == k])),
                             'follow': r4(y[cls == k].mean()), 'graph_v4_agreement': r4(hit['graph_v4'][c][cls == k].mean())} for k in CLASSES}
    bt, ba = PE.threshold(x27, y)
    s['best_threshold_on_this_panel'] = {'T': r4(bt), 'accuracy': r4(ba), 'note': 'descriptive; fitted here'}
    q = np.quantile(x27, np.linspace(0, 1, 11))
    s['follow_by_decile_of_x27'] = [r4(y[(x27 >= q[i]) & ((x27 < q[i + 1]) | (i == 9))].mean()) for i in range(10)]
    s['other_answers_share'] = r4(((out != B) & (out != P)).mean())
    s['broad_answer'] = {'B': r4((out == B).mean()), 'P': r4((out == P)[c].mean()), 'I_when_I_differs_from_B_and_P': r4(
        (out == T['I'])[(T['I'] != B) & (T['I'] != P)].mean())}
    return s


def verdicts(s, meta):
    tri = lambda ok, bad: 'supported' if ok else 'not_supported' if bad else 'inconclusive'
    v = {'controls_valid': bool(meta['controls_ok'])}
    names = list(CRITERIA['predictions_on_holdout'])
    if not v['controls_valid']:
        return {**v, **{k: 'controls_invalid' for k in names}}
    a, cf = s['all_edits'], s['conflict_edits_only']
    g = a['agreement']['graph_v4']['rate']
    v['F1_agreement'] = tri(g >= .80 and cf['agreement']['graph_v4']['rate'] >= .77, g < .75)
    d3, dn = a['graph_v4_minus_graph_v3'], a['graph_v4_minus_none']
    v['F2_beyond_revision_3'] = tri(d3['rate'] >= .08 and d3['ci95'][0] > 0 and dn['ci95'][0] > 0, d3['ci95'][0] <= 0 or dn['ci95'][0] <= 0)
    los = [a['graph_v4_minus_rule_417']['ci95'][0], a['graph_v4_minus_narrow25_answer']['ci95'][0], s['auc']['x27 - x25']['ci95'][0]]
    v['F3_block_27_edge_adds'] = tri(min(los) > 0, True)
    x = s['auc']['x27']['auc']
    v['F4_ranking'] = tri(x >= .90 and min(k['x27'] for k in s['auc_by_class'].values()) >= .88, x < .85)
    return v


def smoke():
    from deploy.storage_policy import checked_dir
    checked_dir(OUT, 1 << 30, operation='来源绑定保真度 v4 产物')
    rows = [r for r in D.read(NL.OUT / 'pilot_name-materials.json') if int(r['family'][-4:]) < 2]
    arms = final_arms(rows)
    results, meta = engine(rows, arms, stage='smoke')
    meta.pop('attention_mass_block27', None)
    s = summarise(results, arms, rows, 0.0)
    print(json.dumps({'meta': meta, 'arms': len(arms), 'verdicts': verdicts(s, meta),
                      'agreement': {k: v['rate'] for k, v in s['all_edits']['agreement'].items()}}), flush=True)
    a = D.read(A27_OUT / 'report.json')['summary']      # the table of this script reproduces the Phase 421 numbers on the opened panel
    T, _ = table(*(D.read(A27_OUT / x) for x in (f'{A27.SPLIT}-materials.json', 'design.json', 'items.json')))
    c = T['P'] != T['B']
    assert abs(PE.auc(T['x27'][c], (T['out'] == T['P'])[c]) - a['auc']['don:N']['auc']) < 1e-4 and int(c.sum()) == a['conflict_edits']
    print(json.dumps({'frozen': frozen()}), flush=True)


DEPS = (PT, NL, F3, H25, PE, DN, A27, A27.FE, A27.AR)


def prereg():
    from deploy.storage_policy import checked_dir
    from tools import provenance
    from tools.power_analysis import check_phase
    checked_dir(OUT, 1 << 30, operation='来源绑定保真度 v4 产物')
    assert not (OUT / 'prereg.json').exists()
    plan = D.checked(MODEL)
    fz = frozen()
    rows = build_materials()
    assert len(rows) == FAMILIES * 16 and len({tuple(r['tokens']) for r in rows}) == len(rows)
    arms = final_arms(rows)
    save(f'{SPLIT}-materials.json', rows)
    save('design.json', arms)
    body = {'schema': 'source_binding_fidelity4_prereg.v1', 'run_id': RUN, 'seed': SEED, 'created_at': datetime.now().astimezone().isoformat(),
            'base_plan_sha256': D.sha(SRC / 'plan.json'), 'model_snapshot': plan['models'][MODEL], 'code_sha256': D.sha(__file__),
            'dependency_sha256': {m.__name__: D.sha(m.__file__) for m in DEPS},
            'graph_summary': 'results/2026-11-12/222216_SOURCE-BINDING-chain-summary-v4.json (edges N1, N7)',
            'criteria': CRITERIA, 'frozen': fz,
            'materials': {'sha256': D.sha(OUT / f'{SPLIT}-materials.json'), 'families': FAMILIES, 'rows': len(rows)},
            'design': {'sha256': D.sha(OUT / 'design.json'), 'arms': len(arms)},
            'power': {'unit': 'paired binary outcome per item (prediction right / wrong under two rules); alpha = 0.05',
                      'F2_v4_against_v3_as_if_one_edit_per_item': {'expected_delta': 0.10, 'p_d': 0.30, **check_phase(0.10, len(rows), p_d=0.30)},
                      'F3_v4_against_rule_417_as_if_one_edit_per_item': {'expected_delta': 0.05, 'p_d': 0.20, **check_phase(0.05, len(rows), p_d=0.20)},
                      'caveat': '4,096 edits clustered in 1,024 items and 64 families; intervals are family bootstraps'},
            'custody': 'holdout_fid4 is read first by the gate stage (behaviour only) and then once by the run stage.',
            'resources': {'gpu_minutes_estimate': 6, 'disk_GiB_estimate': 0.05, 'models_concurrent': 1,
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
    save('gate.json', {'schema': 'source_binding_fidelity4_gate.v1', 'run_id': RUN, 'items_sha256': D.sha(OUT / 'gate-items.json'), **g, **meta})
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
        meta.pop('attention_mass_block27', None)
        save('items.json', DN.slim(results))
        base = D.read(OUT / 'gate-items.json')
        meta['base_replay_max_score_difference'] = max(abs(a - b) for x, y in zip(results['base'], base) for a, b in zip(x['scores'], y['scores']))
        s = summarise(results, arms, rows, pre['frozen']['T'])
        v = verdicts(s, meta)
        print(json.dumps({'meta': meta, 'verdicts': v, 'summary': s}), flush=True)
    else:
        s, meta, v = {'note': 'behaviour gate failed; no edit was run'}, {}, {k: 'behavior_ineligible' for k in names}
    report = {'schema': 'source_binding_fidelity4_report.v1', 'run_id': RUN, 'split': SPLIT, 'gate_passed': g['passed'],
              'prereg_sha256': D.sha(OUT / 'prereg.json'), 'gate_sha256': D.sha(OUT / 'gate.json'),
              'items_sha256': D.sha(OUT / 'items.json') if g['passed'] else None, 'criteria': pre['criteria'], 'frozen': pre['frozen'],
              'verdicts': v, 'summary': s, **meta, 'formal_evidence_promotion': 0,
              'custody': 'holdout_fid4 read once by the gate stage and once by this run',
              'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    save('report.json', report)
    path = provenance.write_result('SOURCE-BINDING-fidelity-v4-holdout', {'verdicts': v, 'gate_passed': g['passed'],
                                                                        'report_sha256': D.sha(OUT / 'report.json')},
                                   model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)
    print(json.dumps({'verdicts': v, 'result': str(path)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['smoke', 'prereg', 'gate', 'run'])
    {'smoke': smoke, 'prereg': prereg, 'gate': gate, 'run': run}[ap.parse_args().stage]()
