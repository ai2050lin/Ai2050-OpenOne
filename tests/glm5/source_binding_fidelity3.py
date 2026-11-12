"""Fidelity of the source-binding graph, revision 3 (Claude Phase 416) on a new panel.  Qwen3-14B BF16.

Revision 2 failed at block 24 (Phase 412): it treated the transition as a switch.  Phase 413 showed a graded merge.
Revision 3 predicts the transition in two ways, both fixed from panels that are already open (holdout_fid2 of Phase
412 and holdout_arb of Phase 413) and both without a parameter fitted on the new panel:

  rule    one record per edit.  Blocks <= 22 and >= 26 as before.  Block 23: still the position regime.  Block 24: the
          answer moves to the donor pointer P only if the entity-name tokens are exchanged and the pointer agrees with
          the name the donor asks for (P = I); otherwise it stays.  Block 25: it stays.
  table   a follow rate for every (site, block, class) cell of the transition, class = how P relates to the base
          answer B and to I (P = I / I = B / all different); the rates are read from the opened panels at
          pre-registration and compared with the new panel.

Edits, donors, outcome and engine are those of Phases 409 / 412.  Stages: smoke -> prereg -> gate -> run.
"""
import argparse, hashlib, json, sys
from datetime import datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_binding_design as D
import source_binding_pointer as PT
import source_binding_nameloc as NL
import source_binding_fidelity2 as F2
import source_binding_arbitration as AR
import source_binding_backup as BK
import source_binding_h2518 as H25

SRC, MODEL = D.OUT, 'qwen14'
FID2_OUT = F2.OUT
OUT = D.ROOT / 'tests/glm5/result/source_binding_fidelity_v3'
RUN, SEED = 'SOURCE-BINDING-FIDELITY-20261112-v3', 202611120416
SPLIT, FAMILIES, REPS = 'holdout_fid3', 64, 4
SITES = ('qsrc', 'qent', 'names')
LAYERS = [8, 14, 18, 22, 23, 24, 25, 26]
TIER = {8: 'repl', 14: 'repl', 18: 'repl', 22: 'repl', 26: 'repl', 23: 't23', 24: 't24', 25: 't25'}
TIERS = ['repl', 't23', 't24', 't25']
MODES = ['graph_v3', 'graph_v2', 'none', 'identity', 'position']
CLASSES = ['P=I', 'I=B', '3way']
OTHER_PANELS = H25.OTHER_PANELS + ['source_binding_h2518_v1/holdout_h25']
CRITERIA = {
    'model': 'Qwen3-14B original BF16, CPU byte reads, CUDA layer streaming. GLM-4-9B is not run (failed the behaviour gate of this task).',
    'material': 'layout-factorial families as in Phase 405. holdout_fid3: 64 families (1,024 prompts; 64 instead of 48 so that the block-24 contrast has power >= 0.8 even with one edit counted per item). Entity names occur in no earlier '
                'panel of this line; the 16 discovery code words, 4 seen frames. There is no pilot panel.',
    'edit': 'as in Phases 409 / 412: at the input of block l the residual state of the archive-name token, of the entity-name tokens '
            'or of both is replaced by the state of the same tokens in a donor prompt of the same family; 4 of the 15 donor relations '
            'per (item, site, layer), seeded. Blocks 8, 14, 18, 22, 23, 24, 25, 26.',
    'records': 'B = base answer; P = record named by the donor pointers (position rule); I = record the donor asks for by name '
               '(identity rule).',
    'predictor': {
        'graph_v3': 'blocks <= 22 and 26: revision 2. Block 23: position rule (P). Block 24: P if the site contains the entity-name '
                    'tokens and P = I, otherwise B. Block 25: B.',
        'graph_v2': 'revision 2 (Phase 412): position rule up to block 24, no effect from block 26 on; at block 25 no effect is assumed '
                    '(revision 2 made no statement there).',
        'none / identity / position': 'as in Phase 412',
        'table': 'follow rate (choice is P) per cell (site, block, class) on edits with P != B; names at blocks 23, 24, 25 and qsrc, '
                 'qent at block 24; classes P = I, I = B, all different (single sites have no third class). Rates pooled over the '
                 'opened panels that hold the cell and frozen in prereg.json.',
        'why': 'opened panels: at block 24 edits of class P = I that exchange the entity name follow P on 0.65-0.74, all other conflict '
               'classes on 0.25-0.36; at block 23 the names site follows on 0.75-1.00, at block 25 on 0.19-0.39.'},
    'tiers': {'repl': 'blocks 8, 14, 18, 22, 26 (third new panel for these cells)', 't23': 'block 23: names seen in Phase 413, qsrc and '
              'qent never run', 't24': 'block 24', 't25': 'block 25: names seen in Phase 413, qsrc and qent never run'},
    'outcome': 'agreement = the forced choice of the edited run is the predicted record; family-bootstrap 95 % intervals. Table: '
               'absolute difference between the frozen rate and the rate on the new panel per cell.',
    'tail_recomputation': 'the no-edit arm recomputed from block 0 gives the same forced choice as the base forward on every item, '
                          'maximal score difference <= 1.0',
    'two_pass': 'gate stage: unmodified forward and the v1 behaviour gate. Edits run only if the gate passed.',
    'predictions_on_holdout': {
        'G1_replication': 'repl: agreement of graph_v3 >= 0.95. not_supported if < 0.85.',
        'G2_block_24': 't24: graph_v3 minus none >= 0.02 with a family-bootstrap lower bound > 0. not_supported if the lower bound <= 0.',
        'G3_block_23': 't23: graph_v3 >= 0.75 and graph_v3 minus none has a lower bound > 0. not_supported if < 0.65 or the lower '
                       'bound <= 0.',
        'G4_block_25': 't25: graph_v3 >= 0.70 and graph_v3 minus position has a lower bound > 0. not_supported if < 0.60 or the lower '
                       'bound <= 0.',
        'G5_table': 'mean absolute error of the frozen follow rates over the cells <= 0.05 and the largest error <= 0.10. '
                    'not_supported if the mean > 0.08 or the largest > 0.15.'},
    'not_covered': 'a per-edit prediction inside the transition beyond the class rule (the rule can be right on about three quarters '
                   'of the block-24 edits at most, because the model itself splits within a class); attention edits; backup sites '
                   '(scored in Phases 409 and 412); next-token distribution beyond the four candidate words; other tasks and models. '
                   'Agreement measures this graph on this family of edits, not the share of the model that is understood.',
    'disclosure': 'the block-24 rule and the table were derived from the opened panels holdout_fid2 and holdout_arb after Phase 415; '
                  'on those panels the rule is therefore fitted, not tested. Blocks 23 and 25 for the single-name sites have never '
                  'been run: there the rule is an extrapolation from the names site. The engine is the unchanged Phase 412 engine, '
                  'smoke-tested on 2 exposed families of pilot_name.',
}


def stream(purpose, key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(f'{RUN}|{SEED}|{purpose}|{key}'.encode()).digest()[:16], 'little'))


def save(name, body):
    from deploy.storage_policy import check_write
    p = OUT / name
    check_write(p, 1 << 20, operation='来源绑定保真度 v3 产物')
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


def records(r, d, site, l):
    return r['target_record'], F2.predict(r, d, site, l, 'position'), F2.predict(r, d, site, l, 'identity')


def klass(B, P, I):
    return None if P == B else 'P=I' if P == I else 'I=B' if I == B else '3way'


def predict(r, d, site, l, mode):
    if mode == 'graph_v2':
        return r['target_record'] if l == 25 else F2.predict(r, d, site, l, 'graph')
    if mode != 'graph_v3':
        return F2.predict(r, d, site, l, mode)
    B, P, I = records(r, d, site, l)
    if l == 23:
        return P
    if l == 24:
        return P if site != 'qsrc' and P == I else B
    if l == 25:
        return B
    return F2.predict(r, d, site, l, 'graph')


def design(rows):
    index = {r['id']: i for i, r in enumerate(rows)}
    arms = {}
    for site in SITES:
        donors = []
        for r in rows:
            row = {}
            for rel in F2.RELATIONS:
                j = index[f"{r['family']}-S{r['source'] ^ rel[0]}E{r['entity'] ^ rel[1]}O{r['order'] ^ rel[2]}X{r['x'] ^ rel[3]}"]
                if len(F2.tokens_of(r, F2.SITES[site])) == len(F2.tokens_of(rows[j], F2.SITES[site])):
                    row[rel] = j
            assert len(row) >= REPS
            donors.append(row)
        for l in LAYERS:
            picks = []
            for r, row in zip(rows, donors):
                rels = sorted(row)
                picks.append([rels[i] for i in stream(f'draw|{site}|{l}', r['id']).permutation(len(rels))[:REPS]])
            for k in range(REPS):
                arms[f'{site}@{l}#{k}'] = {'site': site, 'layer': l, 'rel': [list(p[k]) for p in picks],
                                           'donor': [row[p[k]] for row, p in zip(donors, picks)]}
    return arms


def follow_counts(rows, arms, items, keep):
    """(site, layer, class) -> [edits, choices equal to P], on conflict edits."""
    T = {}
    for name, a in arms.items():
        if not keep(a):
            continue
        for i, (r, j) in enumerate(zip(rows, a['donor'])):
            B, P, I = records(r, rows[j], a['site'], a['layer'])
            c = klass(B, P, I)
            if c:
                t = T.setdefault(f"{a['site']}@{a['layer']}|{c}", [0, 0])
                t[0] += 1
                t[1] += items[name][i]['pred_record'] == P
    return T


def frozen_table():
    T = {}
    for out, split, keep in ((FID2_OUT, F2.SPLIT, lambda a: a['site'] in SITES and a['layer'] == 24),
                             (AR.OUT, AR.SPLIT, lambda a: a['cond'] == 'swap' and a['site'] in SITES and a['layer'] in (23, 24, 25))):
        rows, arms, items = (D.read(out / x) for x in (f'{split}-materials.json', 'design.json', 'items.json'))
        for k, (n, f) in follow_counts(rows, arms, items, keep).items():
            t = T.setdefault(k, [0, 0])
            t[0] += n
            t[1] += f
    return {k: {'n': n, 'rate': round(f / n, 4)} for k, (n, f) in sorted(T.items())}


def engine(rows, arms=None, stage='run'):
    F2.OUT = OUT          # progress log of the unchanged Phase 412 engine goes to this run's folder
    return F2.engine(rows, arms, stage)


def summarise(results, arms, rows, table):
    fam_id = {f: i for i, f in enumerate(sorted({r['family'] for r in rows}))}
    nf = len(fam_id)
    draws = np.random.default_rng(SEED).integers(0, nf, (1000, nf))
    rec = {k: [] for k in ('fam', 'tier', 'site', 'layer', 'changed') + tuple(MODES)}
    for name, a in arms.items():
        for i, (r, j) in enumerate(zip(rows, a['donor'])):
            out = results[name][i]['pred_record']
            rec['fam'].append(fam_id[r['family']]); rec['tier'].append(TIER[a['layer']]); rec['site'].append(a['site']); rec['layer'].append(a['layer'])
            rec['changed'].append(out != r['target_record'])
            for m in MODES:
                rec[m].append(predict(r, rows[j], a['site'], a['layer'], m) == out)
    A = {k: np.array(v) for k, v in rec.items()}

    def rate(v, keep):
        v = np.asarray(v, float)
        per, cnt = np.bincount(A['fam'][keep], weights=v[keep], minlength=nf), np.bincount(A['fam'][keep], minlength=nf)
        bs = per[draws].sum(1) / np.maximum(cnt[draws].sum(1), 1)
        return {'rate': round(float(v[keep].mean()), 4), 'ci95': [round(float(np.quantile(bs, .025)), 4), round(float(np.quantile(bs, .975)), 4)]}

    def block(keep):
        return {'n': int(keep.sum()), 'model_choice_changed': round(float(A['changed'][keep].mean()), 4),
                'agreement': {m: rate(A[m], keep) for m in MODES},
                **{f'graph_v3_minus_{m}': rate(A['graph_v3'].astype(float) - A[m].astype(float), keep) for m in MODES[1:]}}
    out = {'edits': len(A['fam']), 'base_correct': round(float(np.mean([x['forced_correct'] for x in results['base']])), 4),
           'tiers': {t: block(A['tier'] == t) for t in TIERS}, 'all_edits': block(A['tier'] != ''), 'cells': {}}
    for site in SITES:
        for l in LAYERS:
            keep = (A['site'] == site) & (A['layer'] == l)
            out['cells'][f'{site}@{l}'] = {'n': int(keep.sum()), 'changed': round(float(A['changed'][keep].mean()), 4),
                                           **{m: round(float(A[m][keep].mean()), 4) for m in MODES}}
    obs = follow_counts(rows, arms, results, lambda a: a['layer'] in (23, 24, 25))
    out['follow_rates'] = {k: {'n': n, 'rate': round(f / n, 4)} for k, (n, f) in sorted(obs.items())}
    err = {k: round(abs(out['follow_rates'][k]['rate'] - v['rate']), 4) for k, v in table.items()}
    out['table'] = {'cells': {k: {'frozen': table[k]['rate'], 'observed': out['follow_rates'][k]['rate'], 'abs_error': e} for k, e in err.items()},
                    'mean_abs_error': round(float(np.mean(list(err.values()) or [0])), 4), 'max_abs_error': max(err.values(), default=0)}
    return out


def verdicts(s, meta):
    tri = lambda ok, bad: 'supported' if ok else 'not_supported' if bad else 'inconclusive'
    v = {'controls_valid': bool(meta['tail_control_ok'])}
    names = list(CRITERIA['predictions_on_holdout'])
    if not v['controls_valid']:
        return {**v, **{k: 'controls_invalid' for k in names}}
    T = s['tiers']
    ag = lambda t: T[t]['agreement']['graph_v3']['rate']
    v['G1_replication'] = tri(ag('repl') >= .95, ag('repl') < .85)
    d = T['t24']['graph_v3_minus_none']
    v['G2_block_24'] = tri(d['rate'] >= .02 and d['ci95'][0] > 0, d['ci95'][0] <= 0)
    d = T['t23']['graph_v3_minus_none']
    v['G3_block_23'] = tri(ag('t23') >= .75 and d['ci95'][0] > 0, ag('t23') < .65 or d['ci95'][0] <= 0)
    d = T['t25']['graph_v3_minus_position']
    v['G4_block_25'] = tri(ag('t25') >= .70 and d['ci95'][0] > 0, ag('t25') < .60 or d['ci95'][0] <= 0)
    t = s['table']
    v['G5_table'] = tri(t['mean_abs_error'] <= .05 and t['max_abs_error'] <= .10, t['mean_abs_error'] > .08 or t['max_abs_error'] > .15)
    return v


def smoke():
    from deploy.storage_policy import checked_dir
    checked_dir(OUT, 1 << 30, operation='来源绑定保真度 v3 产物')
    rows = [r for r in D.read(NL.OUT / 'pilot_name-materials.json') if int(r['family'][-4:]) < 2]
    arms = design(rows)
    results, meta = engine(rows, arms, stage='smoke')
    s = summarise(results, arms, rows, {})
    print(json.dumps({'meta': meta, 'arms': len(arms), 'edits': s['edits'], 'tier_n': {t: v['n'] for t, v in s['tiers'].items()}}), flush=True)


DEPS = (PT, NL, F2, AR, BK, H25)


def prereg():
    from deploy.storage_policy import checked_dir
    from tools import provenance
    from tools.power_analysis import check_phase
    checked_dir(OUT, 1 << 30, operation='来源绑定保真度 v3 产物')
    assert not (OUT / 'prereg.json').exists()
    plan = D.checked(MODEL)
    table = frozen_table()
    assert len(table) == 13, sorted(table)
    rows = build_materials()
    assert len(rows) == FAMILIES * 16 and len({tuple(r['tokens']) for r in rows}) == len(rows)
    for r in rows[:64]:
        assert all(predict(r, r, s, l, m) == r['target_record'] for s in SITES for l in LAYERS for m in MODES)
    arms = design(rows)
    save(f'{SPLIT}-materials.json', rows)
    save('design.json', arms)
    count, differ = {t: 0 for t in TIERS}, 0
    for a in arms.values():
        for r, j in zip(rows, a['donor']):
            count[TIER[a['layer']]] += 1
            differ += a['layer'] == 24 and predict(r, rows[j], a['site'], 24, 'graph_v3') != r['target_record']
    pd = round(differ / count['t24'], 4)
    body = {'schema': 'source_binding_fidelity_prereg.v3', 'run_id': RUN, 'seed': SEED, 'created_at': datetime.now().astimezone().isoformat(),
            'base_plan_sha256': D.sha(SRC / 'plan.json'), 'chain_summary': 'results/2026-11-12/063401_SOURCE-BINDING-chain-summary-v3.json',
            'model_snapshot': plan['models'][MODEL], 'code_sha256': D.sha(__file__),
            'dependency_sha256': {m.__name__: D.sha(m.__file__) for m in DEPS},
            'criteria': CRITERIA, 'table': table,
            'table_sources': {'holdout_fid2': D.sha(FID2_OUT / 'items.json'), 'holdout_arb': D.sha(AR.OUT / 'items.json')},
            'materials': {'sha256': D.sha(OUT / f'{SPLIT}-materials.json'), 'families': FAMILIES, 'rows': len(rows)},
            'design': {'sha256': D.sha(OUT / 'design.json'), 'arms': len(arms), 'edits_per_tier': count,
                       'block_24_edits_where_graph_v3_differs_from_none': int(differ)},
            'power': {'unit': 'paired binary outcome per edit (graph_v3 right / none right); alpha = 0.05',
                      'G2': {'expected_delta': 0.03, 'p_d': pd, **check_phase(0.03, count['t24'], p_d=pd)},
                      'G2_as_if_one_edit_per_item': {'expected_delta': 0.03, 'p_d': pd, **check_phase(0.03, len(rows), p_d=pd)},
                      'caveat': 'edits are clustered in 1,024 items and 64 families; intervals are family bootstraps'},
            'custody': 'holdout_fid3 is read first by the gate stage (behaviour only) and then once by the run stage.',
            'resources': {'gpu_minutes_estimate': 15, 'disk_GiB_estimate': 0.3, 'models_concurrent': 1,
                          'shape': 'chunks of 256 prompts, inner batch 128, 96 arms; the Phase 412 shape'},
            'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    body['provenance']['model_revision'] = plan['models'][MODEL]
    save('prereg.json', body)
    print(json.dumps({'prereg_sha256': D.sha(OUT / 'prereg.json'), 'design': body['design'], 'power': body['power'], 'table': table}), flush=True)


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
    save('gate.json', {'schema': 'source_binding_fidelity_gate.v3', 'run_id': RUN, 'items_sha256': D.sha(OUT / 'gate-items.json'), **g, **meta})
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
        save('items.json', {k: [{'id': x['id'], 'pred_record': x['pred_record'], 'scores': x['scores']} for x in v] for k, v in results.items()})
        base = D.read(OUT / 'gate-items.json')
        meta['base_replay_max_score_difference'] = max(abs(a - b) for x, y in zip(results['base'], base) for a, b in zip(x['scores'], y['scores']))
        s = summarise(results, arms, rows, pre['table'])
        v = verdicts(s, meta)
        print(json.dumps({'meta': meta, 'verdicts': v, 'tiers': s['tiers'], 'cells': s['cells'], 'table': s['table']}), flush=True)
    else:
        s, meta, v = {'note': 'behaviour gate failed; no edit was run'}, {}, {k: 'behavior_ineligible' for k in names}
    report = {'schema': 'source_binding_fidelity_report.v3', 'run_id': RUN, 'split': SPLIT, 'gate_passed': g['passed'],
              'prereg_sha256': D.sha(OUT / 'prereg.json'), 'gate_sha256': D.sha(OUT / 'gate.json'),
              'items_sha256': D.sha(OUT / 'items.json') if g['passed'] else None, 'criteria': pre['criteria'], 'verdicts': v, 'summary': s,
              **meta, 'formal_evidence_promotion': 0, 'custody': 'holdout_fid3 read once by the gate stage and once by this run',
              'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    save('report.json', report)
    path = provenance.write_result('SOURCE-BINDING-fidelity-v3-holdout', {'verdicts': v, 'gate_passed': g['passed'],
                                                                       'report_sha256': D.sha(OUT / 'report.json')},
                                   model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)
    print(json.dumps({'verdicts': v, 'result': str(path)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['smoke', 'prereg', 'gate', 'run'])
    {'smoke': smoke, 'prereg': prereg, 'gate': gate, 'run': run}[ap.parse_args().stage]()
