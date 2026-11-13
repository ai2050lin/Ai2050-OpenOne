"""Which heads of blocks 23-27 read the name tokens at the answer position (Claude Phase 424).  Qwen3-14B BF16.

Phase 423: the answer of a narrow edit in which the answer position reads donor keys / values at the site tokens in
blocks L..27 through EVERY head agrees with the broad edit in 0.931; through the heads already on the graph
(25.18, 25.24, 25.15 and block 27) only in 0.585 when the exchange is made at block 23.  Here the 200 heads of blocks
23-27 are screened one by one (leave one out of the all-heads rule) and a sparse head set is frozen and tested.

Stages: smoke, pilot1 (leave-one-out, opened panel holdout_sites), pilot2 (sparse sets, same panel) -> prereg -> gate -> run.
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
import source_binding_sites as S

SRC, MODEL = D.OUT, 'qwen14'
OUT = D.ROOT / 'tests/glm5/result/source_binding_readheads_v1'
S_OUT = S.OUT
RUN, SEED = 'SOURCE-BINDING-READHEADS-20261112-v1', 202611120424
SPLIT, FAMILIES = 'holdout_rh', 64
NH, REPS = 40, 4
BLOCKS = (23, 24, 25, 26, 27)
SITES, LAYERS, CELLS = S.SITES, S.LAYERS, S.CELLS
OTHER_PANELS = S.OTHER_PANELS + ['source_binding_sites_v1/holdout_sites']
SCREEN = ('qsrc@23', 'qent@23')       # cells of the leave-one-out screen
SCREEN_FAMILIES, SCREEN_REPS = 32, 2


def save(name, body):
    from deploy.storage_policy import check_write
    p = OUT / name
    check_write(p, 1 << 20, operation='来源绑定读取头产物')
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


def engine(rows, arms=None, stage='run'):
    S.OUT = OUT
    return S.engine(rows, arms, stage)


def spec_all(site, l):
    return {b: [[[site], None]] for b in range(l, 28)}


def spec_heads(site, l, heads):
    """heads: {block: [head, ...]} - only these heads read the donor site tokens, from block l on."""
    return {b: [[[site], sorted(hs)]] for b, hs in heads.items() if int(b) >= l and hs}


def donors(rows, run, seed):
    H25.RUN, H25.SEED = run, seed                 # donor draws: the stream of the Phase 415 module contains run id and seed
    return H25.draws(rows)


def screen_arms(rows, run, seed, cells=SCREEN):
    dons, rels = donors(rows, run, seed)
    arms = {'self': {'donor': list(range(len(rows))), 'don': {b: [[['names', 'holders'], None]] for b in BLOCKS}}}
    for k in range(SCREEN_REPS):
        base = {'donor': dons[k], 'rel': rels[k]}
        for cell in cells:
            site, l = cell.split('@')[0], int(cell.split('@')[1])
            arms[f'swap|{cell}#{k}'] = {'swap': [site, l], **base}
            arms[f'all|{cell}#{k}'] = {'don': spec_all(site, l), **base}
            for b in [x for x in BLOCKS if x >= l]:
                arms[f'only{b}|{cell}#{k}'] = {'don': {b: [[[site], None]]}, **base}
                arms[f'drop{b}|{cell}#{k}'] = {'don': {x: [[[site], None]] for x in BLOCKS if x != b and x >= l}, **base}
                for h in range(NH):
                    arms[f'loo{b}.{h}|{cell}#{k}'] = {'don': {**spec_all(site, l), b: [[[site], [x for x in range(NH) if x != h]]]}, **base}
    return arms


def xtable(rows, arms, items, cells, reps):
    """Per cell: conflict edits with x = score(P) - score(B) of every narrow arm and the answers."""
    out = {}
    for cell in cells:
        site, l = cell.split('@')[0], int(cell.split('@')[1])
        names = sorted({n.split('|')[0] for n in arms if n.split('|')[-1].split('#')[0] == cell and not n.startswith('swap')})
        T = defaultdict(list)
        for k in range(reps):
            a = arms[f'swap|{cell}#{k}']
            for i, (r, j) in enumerate(zip(rows, a['donor'])):
                B, P, I = F3.records(r, rows[j], site, l)
                if P == B:
                    continue
                T['out'].append(items[f'swap|{cell}#{k}'][i]['pred_record']); T['P'].append(P); T['B'].append(B)
                bs = items['base'][i]['scores']
                T['nat'].append(bs[P] - bs[B])
                for n in names:
                    it = items[f'{n}|{cell}#{k}'][i]
                    T['x:' + n].append(it['scores'][P] - it['scores'][B]); T['a:' + n].append(it['pred_record'])
        out[cell] = {k: np.array(v) for k, v in T.items()}
    return out


def smoke():
    from deploy.storage_policy import checked_dir
    checked_dir(OUT, 1 << 30, operation='来源绑定读取头产物')
    rows = [r for r in D.read(NL.OUT / 'pilot_name-materials.json') if int(r['family'][-4:]) < 2]
    dons, rels = donors(rows, RUN, SEED)
    arms = {'self': {'donor': list(range(len(rows))), 'don': {b: [[['names', 'holders'], None]] for b in BLOCKS}},
            'swap|names@23#0': {'swap': ['names', 23], 'donor': dons[0]}, 'all|names@23#0': {'don': spec_all('names', 23), 'donor': dons[0]},
            'sp|names@23#0': {'don': spec_heads('names', 23, {23: [1, 2], 25: [18, 24, 15], 27: list(range(NH))}), 'donor': dons[0]}}
    results, meta = engine(rows, arms, stage='smoke')
    T = xtable(rows, arms, results, ['names@23'], 1)['names@23']
    print(json.dumps({'meta': meta, 'agree_all': float((T['a:all'] == T['out']).mean()), 'agree_sparse': float((T['a:sp'] == T['out']).mean())}), flush=True)


def pilot1(cells=SCREEN, tag='pilot1'):
    """Leave-one-out screen of the heads of blocks L-27 on 32 opened families of holdout_sites (new donor draws)."""
    from deploy.storage_policy import checked_dir
    checked_dir(OUT, 1 << 30, operation='来源绑定读取头产物')
    rows = [r for r in D.read(S_OUT / f'{S.SPLIT}-materials.json')]
    fams = sorted({r['family'] for r in rows})[:SCREEN_FAMILIES]
    rows = [r for r in rows if r['family'] in fams]
    arms = screen_arms(rows, RUN + '|' + tag, SEED, cells)
    results, meta = engine(rows, arms, stage=tag)
    T = xtable(rows, arms, results, cells, SCREEN_REPS)
    res = {'meta': meta, 'families': len(fams), 'cells': {}}
    for cell, t in T.items():
        full = (t['x:all'] - t['nat']).mean()
        bl = [b for b in BLOCKS if b >= int(cell.split('@')[1])]
        c = {'conflict_edits': int(len(t['out'])), 'full_shift': round(float(full), 3),
             'agreement': {n: round(float((t['a:' + n] == t['out']).mean()), 4) for n in ['all'] + [f'only{b}' for b in bl] + [f'drop{b}' for b in bl]},
             'share': {n: round(float((t['x:' + n] - t['nat']).mean() / full), 4) for n in [f'only{b}' for b in bl] + [f'drop{b}' for b in bl]},
             'loo_loss': {f'{b}.{h}': round(float((t['x:all'] - t[f'x:loo{b}.{h}']).mean() / full), 4) for b in bl for h in range(NH)}}
        res['cells'][cell] = c
    save(f'{tag}.json', res)
    print(json.dumps({'meta': meta, **{c: {'agreement': v['agreement'], 'share': v['share'],
                                           'top': sorted(v['loo_loss'].items(), key=lambda kv: -kv[1])[:14],
                                           'bottom': sorted(v['loo_loss'].items(), key=lambda kv: kv[1])[:4]} for c, v in res['cells'].items()}}), flush=True)



# ---- after pilot 1 (leave-one-out on 32 opened families): heads with the largest loss, by block ----
H12 = {23: [36, 37, 38], 24: [17, 19], 25: [15, 16, 18, 24], 27: [0, 11, 13]}
H9 = {23: [36, 37, 38], 25: [15, 18, 24], 27: [0, 11, 13]}
H16 = {23: [36, 37, 38], 24: [17, 19, 33, 36], 25: [15, 16, 18, 24, 38], 26: [18], 27: [0, 11, 13]}   # + pilot 1b (screen at block 24)
EDGES2 = {25: [15, 18, 24], 27: list(range(NH))}        # the two edges of revision 4 (rule a of Phase 423)
RANDOM_SETS = 5


def head_sets():
    rng = np.random.default_rng(SEED)
    sets = {'H16': H16, 'H12': H12, 'H9': H9, 'two_edges': EDGES2}
    for j in range(RANDOM_SETS):     # as many heads per block as H16, drawn from the other heads of the block
        sets[f'R{j}'] = {b: sorted(int(h) for h in rng.choice([x for x in range(NH) if x not in hs], len(hs), replace=False)) for b, hs in H16.items()}
    return sets


def final_arms(rows, run=RUN, seed=SEED):
    dons, rels = donors(rows, run, seed)
    arms = {'self': {'donor': list(range(len(rows))), 'don': {b: [[['names', 'holders'], None]] for b in BLOCKS}}}
    sets = head_sets()
    for k in range(REPS):
        base = {'donor': dons[k], 'rel': rels[k]}
        for site in SITES:
            for l in LAYERS:
                cell = f'{site}@{l}'
                arms[f'swap|{cell}#{k}'] = {'swap': [site, l], **base}
                arms[f'all|{cell}#{k}'] = {'don': spec_all(site, l), **base}
                for name, hs in sets.items():
                    spec = spec_heads(site, l, hs)
                    if spec:
                        arms[f'{name}|{cell}#{k}'] = {'don': spec, **base}
    return arms


def summarise(results, arms, rows):
    T, nf, names = S.table(rows, arms, results)
    draws = np.random.default_rng(SEED).integers(0, nf, (1000, nf))
    fam, out = T['fam'], T['out']
    r4 = lambda v: round(float(v), 4)

    def rate(v, keep):
        v = np.asarray(v, float)
        per, cnt = np.bincount(fam[keep], weights=v[keep], minlength=nf), np.bincount(fam[keep], minlength=nf)
        bs = per[draws].sum(1) / np.maximum(cnt[draws].sum(1), 1)
        return {'rate': r4(v[keep].mean()), 'ci95': [r4(np.quantile(bs, .025)), r4(np.quantile(bs, .975))]}
    hit = {**{v: (T[v] == out).astype(float) for v in names}, 'graph_v3': (T['v3'] == out).astype(float), 'none': (T['B'] == out).astype(float)}
    rnd = [f'R{j}' for j in range(RANDOM_SETS)]
    hit['random_mean'] = np.mean([hit[k] for k in rnd], 0)
    layer = np.array([int(c.split('@')[1]) for c in T['cell']])
    scopes = {'all': np.ones(len(out), bool), **{f'block_{l}': layer == l for l in LAYERS}, **{c: T['cell'] == c for c in CELLS}}
    s = {'edits': int(len(out)), 'scopes': {}}
    for name, keep in scopes.items():
        s['scopes'][name] = {'n': int(keep.sum()), 'agreement': {k: rate(v, keep) for k, v in hit.items()},
                             **{f'{a}_minus_{b}': rate(hit[a] - hit[b], keep) for a, b in (
                                 ('all', 'H16'), ('all', 'H12'), ('all', 'H9'), ('H16', 'random_mean'), ('H16', 'two_edges'),
                                 ('H16', 'graph_v3'), ('H16', 'H12'), ('H12', 'H9'), ('H16', 'none'))}}
    s['random_max_overall'] = max(s['scopes']['all']['agreement'][k]['rate'] for k in rnd)
    return s


def pilot2():
    """The frozen arms on the 32 opened families of holdout_sites that the screen did not use."""
    from deploy.storage_policy import checked_dir
    checked_dir(OUT, 1 << 30, operation='来源绑定读取头产物')
    rows = [r for r in D.read(S_OUT / f'{S.SPLIT}-materials.json')]
    fams = sorted({r['family'] for r in rows})[SCREEN_FAMILIES:]
    rows = [r for r in rows if r['family'] in fams]
    arms = final_arms(rows, RUN + '|pilot2', SEED)
    results, meta = engine(rows, arms, stage='pilot2')
    s = summarise(results, arms, rows)
    save('pilot2-summary.json', {'meta': meta, 'families': len(fams), **s})
    print(json.dumps({'meta': meta, **{n: {k: v['rate'] for k, v in sc['agreement'].items()} for n, sc in s['scopes'].items()}}), flush=True)



# ---- frozen after pilot 2 ----
PRIMARY = 'H16'
THRESHOLDS = {
    'R1_sixteen_heads_nearly_suffice': 'agreement of H16 over the nine cells >= 0.89 and the all-heads rule exceeds it by <= 0.04. '
                                       'not_supported if the difference > 0.08.',
    'R2_the_heads_are_specific': 'H16 minus the mean of 5 random sets with the same number of heads per block >= 0.15 with a lower '
                                 'bound > 0, and no random set exceeds no effect by more than 0.02. not_supported if the lower '
                                 'bound <= 0.',
    'R3_block_23_and_24_heads_are_needed': 'at block 23: H16 minus the two graph edges of revision 4 (25.15, 25.18, 25.24 and block '
                                           '27) >= 0.20 with a lower bound > 0. not_supported if the lower bound <= 0.',
    'R4_beyond_revision_3': 'H16 minus the revision-3 rule over the nine cells >= 0.04 with a lower bound > 0. not_supported if the '
                            'lower bound <= 0.',
}
CRITERIA = {
    'model': 'Qwen3-14B original BF16, CPU byte reads, CUDA layer streaming. GLM-4-9B is not run (failed the behaviour gate of this task).',
    'material': 'layout-factorial families as in Phase 405. holdout_rh: 64 families (1,024 prompts). Entity names occur in no earlier '
                'panel of this line; the 16 discovery code words, 4 seen frames.',
    'broad_edit': 'as in Phase 423: at the input of block L (23, 24, 25) the residual states of the site tokens (qsrc, qent or both) '
                  'are replaced by those of a donor of the same family. 4 seeded donor relations per item; 9 cells x 4,096 edits.',
    'narrow_rules': {'all': 'the answer position reads donor keys / values at the site tokens in blocks L..27, every head (Phase 423 rule c)',
                     'H16 (primary)': 'only heads 23.36, 23.37, 23.38, 24.17, 24.19, 24.33, 24.36, 25.15, 25.16, 25.18, 25.24, 25.38, '
                                      '26.18, 27.0, 27.11, 27.13 do (those of blocks >= L)',
                     'H12': 'H16 without 24.33, 24.36, 25.38, 26.18', 'H9': '23.36-38, 25.15, 25.18, 25.24, 27.0, 27.11, 27.13',
                     'two_edges': '25.15, 25.18, 25.24 and every head of block 27 (Phase 423 rule a)',
                     'R0-R4': 'as many heads per block as H16, drawn from the other heads of the block (seeded)'},
    'prediction': 'the broad edit gives the answer of the narrow edit. Comparisons: revision-3 rule, no effect.',
    'outcome': 'agreement = the predicted record equals the forced choice of the broad edit; per cell, per block, overall; '
               'family-bootstrap 95 % intervals (1000 draws).',
    'controls': 'tail recomputation equals the base forward (same choice, score difference <= 1.0); the narrow edit with the prompt '
                'itself as donor (names and holders, blocks 23-27, every head) changes nothing (same bound).',
    'two_pass': 'gate stage: unmodified forward and the v1 behaviour gate. Edits run only if the gate passed.',
    'predictions_on_holdout': THRESHOLDS,
    'not_covered': 'what each head reads and writes (QK and OV are exchanged together); keys and values are shared inside a group of 5 '
                   'heads, so neighbours (23.36-38, 24.33 / 24.36, 25.15-18) may differ only by their queries; heads were screened by '
                   'leaving one out of the all-heads rule, which misses redundant heads; the routes through other positions; other '
                   'tasks and models.',
    'disclosure': 'screens and pilots used the opened panel holdout_sites with new donor draws. Pilot 1: leave-one-out of the 200 '
                  'heads of blocks 23-27 for qsrc@23 and qent@23 on 32 families (H12, H9). Pilot 2 on the other 32 families showed '
                  'the largest gap to the all-heads rule for exchanges at block 24 (H12 0.877 against 0.925); pilot 1b screened '
                  'names@24 on the first 32 families and added 24.33, 24.36, 25.38, 26.18 (H16); pilot 2 was run again (H16 0.914, '
                  'all heads 0.931, random sets 0.676 = no effect). Thresholds were set on that output. Smoke test on 2 exposed '
                  'families of pilot_name.',
}


def verdicts(s, meta):
    tri = lambda ok, bad: 'supported' if ok else 'not_supported' if bad else 'inconclusive'
    v = {'controls_valid': bool(meta['controls_ok'])}
    if not v['controls_valid']:
        return {**v, **{k: 'controls_invalid' for k in THRESHOLDS}}
    A = s['scopes']['all']
    g, d = A['agreement'][PRIMARY]['rate'], A['all_minus_H16']['rate']
    v['R1_sixteen_heads_nearly_suffice'] = tri(g >= .89 and d <= .04, d > .08)
    r = A['H16_minus_random_mean']
    v['R2_the_heads_are_specific'] = tri(r['rate'] >= .15 and r['ci95'][0] > 0 and s['random_max_overall'] - A['agreement']['none']['rate'] <= .02,
                                         r['ci95'][0] <= 0)
    e = s['scopes']['block_23']['H16_minus_two_edges']
    v['R3_block_23_and_24_heads_are_needed'] = tri(e['rate'] >= .20 and e['ci95'][0] > 0, e['ci95'][0] <= 0)
    x = A['H16_minus_graph_v3']
    v['R4_beyond_revision_3'] = tri(x['rate'] >= .04 and x['ci95'][0] > 0, x['ci95'][0] <= 0)
    return v


DEPS = (PT, NL, F3, H25, S, S.AR, S.F4)


def prereg():
    from deploy.storage_policy import checked_dir
    from tools import provenance
    from tools.power_analysis import check_phase
    checked_dir(OUT, 1 << 30, operation='来源绑定读取头产物')
    assert not (OUT / 'prereg.json').exists()
    plan = D.checked(MODEL)
    assert verdicts(D.read(OUT / 'pilot2-summary.json'), {'controls_ok': True})
    rows = build_materials()
    assert len(rows) == FAMILIES * 16 and len({tuple(r['tokens']) for r in rows}) == len(rows)
    arms = final_arms(rows)
    save(f'{SPLIT}-materials.json', rows)
    save('design.json', arms)
    body = {'schema': 'source_binding_readheads_prereg.v1', 'run_id': RUN, 'seed': SEED, 'created_at': datetime.now().astimezone().isoformat(),
            'base_plan_sha256': D.sha(SRC / 'plan.json'), 'model_snapshot': plan['models'][MODEL], 'code_sha256': D.sha(__file__),
            'dependency_sha256': {m.__name__: D.sha(m.__file__) for m in DEPS},
            'criteria': CRITERIA,
            'frozen': {'primary': PRIMARY, 'sets': head_sets(), 'pilot_sha256': {f: D.sha(OUT / f) for f in ('pilot1.json', 'pilot1b.json', 'pilot2-summary.json')}},
            'materials': {'sha256': D.sha(OUT / f'{SPLIT}-materials.json'), 'families': FAMILIES, 'rows': len(rows)},
            'design': {'sha256': D.sha(OUT / 'design.json'), 'arms': len(arms)},
            'power': {'unit': 'paired binary outcome per item (prediction right / wrong under H16 and under revision 3); alpha = 0.05',
                      'R4_as_if_one_edit_per_item': {'expected_delta': 0.06, 'p_d': 0.20, **check_phase(0.06, len(rows), p_d=0.20)},
                      'caveat': '36,864 edits clustered in 1,024 items and 64 families; intervals are family bootstraps'},
            'custody': 'holdout_rh is read first by the gate stage (behaviour only) and then once by the run stage.',
            'resources': {'gpu_minutes_estimate': 13, 'disk_GiB_estimate': 0.1, 'models_concurrent': 1,
                          'shape': 'chunks of 192 prompts in one batch; pilot peak 6.1 GiB (32 families: 6.2 minutes)'},
            'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    body['provenance']['model_revision'] = plan['models'][MODEL]
    save('prereg.json', body)
    print(json.dumps({'prereg_sha256': D.sha(OUT / 'prereg.json'), 'design': body['design'], 'power': body['power'],
                      'pilot_verdicts': verdicts(D.read(OUT / 'pilot2-summary.json'), {'controls_ok': True})}), flush=True)


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
    save('gate.json', {'schema': 'source_binding_readheads_gate.v1', 'run_id': RUN, 'items_sha256': D.sha(OUT / 'gate-items.json'), **g, **meta})
    print(json.dumps({k: g[k] for k in ('passed', 'forced_accuracy', 'top1_accuracy')}), flush=True)


def run():
    from tools import provenance
    pre = checked()
    assert not (OUT / 'report.json').exists(), 'holdout already used'
    g = D.read(OUT / 'gate.json')
    rows, arms = D.read(OUT / f'{SPLIT}-materials.json'), D.read(OUT / 'design.json')
    if g['passed']:
        results, meta = engine(rows, arms, stage='run')
        save('items.json', {k: [{'id': x['id'], 'pred_record': x['pred_record']} for x in v] for k, v in results.items()})
        base = D.read(OUT / 'gate-items.json')
        meta['base_replay_max_score_difference'] = max(abs(a - b) for x, y in zip(results['base'], base) for a, b in zip(x['scores'], y['scores']))
        s = summarise(results, arms, rows)
        v = verdicts(s, meta)
        print(json.dumps({'meta': meta, 'verdicts': v}), flush=True)
    else:
        s, meta, v = {'note': 'behaviour gate failed; no edit was run'}, {}, {k: 'behavior_ineligible' for k in THRESHOLDS}
    report = {'schema': 'source_binding_readheads_report.v1', 'run_id': RUN, 'split': SPLIT, 'gate_passed': g['passed'],
              'prereg_sha256': D.sha(OUT / 'prereg.json'), 'gate_sha256': D.sha(OUT / 'gate.json'),
              'items_sha256': D.sha(OUT / 'items.json') if g['passed'] else None, 'criteria': pre['criteria'], 'frozen': pre['frozen'],
              'verdicts': v, 'summary': s, **meta, 'formal_evidence_promotion': 0,
              'custody': 'holdout_rh read once by the gate stage and once by this run',
              'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    save('report.json', report)
    path = provenance.write_result('SOURCE-BINDING-readheads-holdout', {'verdicts': v, 'gate_passed': g['passed'],
                                                                      'report_sha256': D.sha(OUT / 'report.json')},
                                   model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)
    print(json.dumps({'verdicts': v, 'result': str(path)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['smoke', 'pilot1', 'pilot1b', 'pilot2', 'prereg', 'gate', 'run'])
    {'smoke': smoke, 'pilot1': pilot1, 'pilot1b': lambda: pilot1(('names@24',), 'pilot1b'), 'pilot2': pilot2, 'prereg': prereg, 'gate': gate, 'run': run}[ap.parse_args().stage]()
