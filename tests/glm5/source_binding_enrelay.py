"""Is the end-of-turn relay of the Chinese task also present in English prompts? (Claude Phase 436).  Qwen3-14B BF16.

Phase 434: in the Chinese task the newline after <|im_end|> (tok1) collects the asked names, written by the block 23-25
members of SB.H.read16 (W13), and the answer position reads it; the direct reading at the answer position alone is weak.
In English prompts the direct reading suffices (Phase 433, colours 0.934), but the end-of-turn tokens still carry nearly
all of the score shift (0.997).  Here, in the colours task and in the original source-binding task, the two routes are
separated: the answer row is barred from the new names in every block (relay only), and the relay token's write is
restricted to W13 or to random heads.  Engine, restrictions and head sets are those frozen in Phase 434.
Stages: smoke, pilot (opened panels) -> prereg -> gate -> run (new panels).
"""
import argparse, json, sys
from datetime import datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_binding_design as D
import source_binding_pointer as PT
import source_binding_nameloc as NL
import source_binding_h2518 as H25
import source_binding_readheads as RH
import source_binding_transfer as TR
import source_binding_zhroute as Z
import source_binding_eotheads as E

SRC, MODEL = D.OUT, 'qwen14'
OUT = D.ROOT / 'tests/glm5/result/source_binding_enrelay_v1'
RUN, SEED = 'SOURCE-BINDING-ENRELAY-20261114-v1', 202611140436
TASKS = ('colours', 'orig')
TR.TEXT['orig'] = {'tail': '\nReturn only the code word, with no explanation.'}
NEW_COLOURS = {'sources': ['Monica', 'Nathan', 'Oscar', 'Paula', 'Rachel', 'Simon'],
               'items': ['shovel', 'teapot', 'mitten', 'pocket', 'ribbon', 'saddle', 'tablet', 'vessel', 'basin', 'crate']}
ORIG_SPLIT, ORIG_OTHER = 'holdout_en', RH.OTHER_PANELS + ['source_binding_readheads_v1/holdout_rh']
NOFINAL = {b: [['final', 'names', None]] for b in E.ALLB}


def merge(*hs):
    out = {}
    for h in hs:
        for b, v in h.items():
            out[b] = out.get(b, []) + v
    return out


def arms_set(site, l, base):
    out = {'swap': E.arm(site, l, base, E.PART), 'direct': E.arm(site, l, base, []), 'tok1': E.arm(site, l, base, ['end1']),
           'via:end': E.arm(site, l, base, ['end']),
           'relay': E.arm(site, l, base, ['end1'], NOFINAL), 'relay:end': E.arm(site, l, base, ['end'], NOFINAL),
           'relay:W13': E.arm(site, l, base, ['end1'], merge(NOFINAL, E.write_only(E.W13))),
           'noroute': E.arm(site, l, base, [], NOFINAL), 'tok1:W13': E.arm(site, l, base, ['end1'], E.write_only(E.W13))}
    for j, rs_ in enumerate(E.random_sets(E.W13)):
        out[f'relay:R{j}'] = E.arm(site, l, base, ['end1'], merge(NOFINAL, E.write_only(rs_)))
    return out


PAIRS = [('relay', 'noroute'), ('relay:W13', 'noroute'), ('relay', 'relay:W13'), ('tok1', 'direct'), ('direct', 'noroute'),
         ('direct', 'none'), ('relay', 'none'), ('noroute', 'none'), ('tok1:W13', 'direct'), ('via:end', 'tok1'), ('relay:end', 'relay')]


def tag(rows, task):
    for r in rows:
        r.setdefault('task', task)
    return E.annotate(rows)


def engine(rows, arms=None, stage='run'):
    E.OUT, E.CHUNK = OUT, 48
    return E.engine(rows, arms, stage)


def run_set(rows, key):
    arms = E.make_arms(rows, f'{RUN}|{key}', arms_set, E.FCELLS, E.REPS)
    return arms


def opened(task):
    if task == 'colours':
        rows = D.read(Z.OUT / 'colours-confirm-materials.json')
    else:
        rows = D.read(RH.OUT / f'{RH.SPLIT}-materials.json')
    fams = sorted({r['family'] for r in rows})[:24]
    return tag([r for r in rows if r['family'] in fams], task)


def save(name, body):
    from deploy.storage_policy import check_write
    p = OUT / name
    check_write(p, 1 << 20, operation='来源绑定英文中转产物')
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(body, ensure_ascii=False, indent=1, default=lambda x: x.item() if isinstance(x, np.generic) else x), encoding='utf-8')
    t.replace(p)


KS = ['direct', 'tok1', 'via:end', 'relay', 'relay:end', 'relay:W13', 'relay:R0', 'relay:R1', 'noroute', 'tok1:W13', 'none']


def smoke():
    from deploy.storage_policy import checked_dir
    checked_dir(OUT, 1 << 30, operation='来源绑定英文中转产物')
    rows = tag([r for r in D.read(TR.OUT / 'colours-pilot-materials.json') if int(r['family'][-4:]) < 2], 'colours')
    arms = E.make_arms(rows, f'{RUN}|smoke', arms_set, ['names@23'], 1)
    results, meta = engine(rows, arms, stage='smoke')
    print(json.dumps(meta), flush=True)
    print(E.show(E.summarise(results, arms, rows), KS), flush=True)


def pilot():
    from deploy.storage_policy import checked_dir
    checked_dir(OUT, 1 << 30, operation='来源绑定英文中转产物')
    for task in TASKS:
        rows = opened(task)
        arms = run_set(rows, f'{task}|pilot')
        results, meta = engine(rows, arms, stage=f'pilot-{task}')
        s = E.summarise(results, arms, rows, PAIRS)
        save(f'{task}-pilot-summary.json', {'meta': meta, **s})
        print(task, json.dumps(meta), flush=True)
        print(E.show(s, KS), flush=True)
        print(json.dumps({k: v['rate'] for k, v in s['scopes']['all']['diff'].items()}), flush=True)


def build_new(task):
    if task == 'colours':
        used = set(Z.CONFIRM['colours']['sources'] + Z.CONFIRM['colours']['items'])
        for split in ('pilot', 'holdout'):
            used |= set(TR.POOLS['colours']['sources'][split] + TR.POOLS['colours']['items'][split])
        assert not set(NEW_COLOURS['sources'] + NEW_COLOURS['items']) & used
        TR.POOLS['colours']['sources']['en2'] = NEW_COLOURS['sources']
        TR.POOLS['colours']['items']['en2'] = NEW_COLOURS['items']
        TR.FAMILIES['en2'] = 48
        rows = TR.build('colours', 'en2')
    else:
        taken = {nm for rows in D.read(SRC / f'{MODEL}-materials.json')['splits'].values() for r in rows for nm in r['names']}
        for p in PT.PANELS + [PT.OUT / 'pilot_layout-materials.json', PT.OUT / 'holdout_layout-materials.json',
                              NL.OUT / 'pilot_name-materials.json', NL.OUT / 'holdout_name-materials.json'] + \
                [D.ROOT / 'tests/glm5/result' / f'{x}-materials.json' for x in ORIG_OTHER]:
            taken |= {nm for r in D.read(p) for nm in r['names']}
        PT.SPLITS[ORIG_SPLIT] = 48
        rows = PT.build_materials(ORIG_SPLIT, taken)
    rows = tag(rows, task)
    H25.RUN, H25.SEED = RUN, SEED
    H25.draws(rows)
    assert len(rows) == 768 and len({tuple(r['tokens']) for r in rows}) == 768
    return rows


def materials():
    for task in TASKS:
        rows = build_new(task)
        print(task, len(rows), min(len(r['tokens']) for r in rows), max(len(r['tokens']) for r in rows), flush=True)



# ---- frozen after the pilot on the opened panels (colours: Phase 433 new-pool panel; orig: Phase 424 panel holdout_rh) ----
THRESHOLDS = {
    'E1_relay_alone_moves_no_answer': 'in both English tasks, with the answer row barred from the new names in every block and only the '
                                      'relay token free, agreement differs from no effect by <= 0.01. not_supported if > 0.03 in either.',
    'E2_relay_alone_carries_a_small_score_shift': 'in both tasks the share of the broad edit\'s score shift reached by the relay alone '
                                                  'lies in [0.02, 0.30] with a lower bound > 0. not_supported if the lower bound <= 0 '
                                                  'in either or the share > 0.45.',
    'E3_written_by_the_same_heads': 'in both tasks the relay written only through W13 keeps >= 0.6 of the relay-alone share, and every '
                                    'random 13-head set keeps <= 0.02 in share. not_supported if W13 keeps < 0.3 in either.',
    'E4_no_answer_gain_over_direct': 'in both tasks the direct reading alone agrees >= 0.90 and the relay token added on top changes '
                                     'agreement by <= 0.02 in size. not_supported if the gain > 0.05 in either.',
}
CRITERIA = {
    'model': 'Qwen3-14B original BF16, CPU byte reads, CUDA layer streaming. GLM-4-9B is not run.',
    'material': 'colours: the English task of Phase 426, new panel of 48 families (768 prompts) with source and item pools disjoint from '
                'all earlier colours pools; orig: the source-binding task (layout-factorial families as in Phase 405), new panel '
                'holdout_en of 48 families with entity names that occur in no earlier panel of this line.',
    'broad_edit': 'at the input of block L (23, 24) the residual states of the source name in the question, the item name or both are '
                  'replaced by those of a donor of the same family. 2 seeded donor relations per item; 6 cells x 1,536 edits per task.',
    'restriction': 'the engine of Phase 434, unchanged: positions outside the free set are reset to the base run after every block; '
                   'tok1 = the newline after <|im_end|> free; direct = nothing free; relay = tok1 free and the answer row sees base keys '
                   '/ values at the name tokens in every block (no direct reading); noroute = nothing free and no direct reading; '
                   'relay:W13 = relay with the relay row reading the names only through the 13 heads of Phase 434; relay:Rj = the same '
                   'with the 3 random 13-head sets of Phase 434.',
    'prediction': 'the broad edit gives the answer of the restricted edit; agreement and the share of the score shift on conflict '
                  'edits; family-bootstrap 95 % intervals (1000 draws).',
    'controls': 'as in Phase 434 (the name exchange with the prompt itself, all free and with hides in blocks 25 and 30).',
    'two_pass': 'gate stage, per task: unmodified forward and the v1 behaviour gate. Edits run only for a task whose gate passed.',
    'predictions_on_holdout': THRESHOLDS,
    'not_covered': 'what the relay token holds; why the Chinese prompt depends on it and the English prompts do not (language, name '
                   'form and question structure differ together); exchanges at block 25; other models.',
    'disclosure': 'one pilot on opened panels (24 families per task, new donor draws). The direction of E1 was already visible in the '
                  'smoke test on 2 exposed families. Thresholds were set after the pilot (relay alone = no effect in both tasks; relay '
                  'share 0.058 / 0.156; W13 0.053 / 0.111; random 0; tok1 - direct +0.002 / +0.003).',
}


def verdicts(S5, metas):
    tri = lambda ok, bad: 'supported' if ok else 'not_supported' if bad else 'inconclusive'
    v = {f'{t}_controls_valid': bool(metas.get(t, {}).get('controls_ok')) for t in TASKS}
    if not all(t in S5 and v[f'{t}_controls_valid'] for t in TASKS):
        return {**v, **{k: 'behavior_ineligible_or_controls_invalid' for k in THRESHOLDS}}
    A = {t: S5[t]['scopes']['all'] for t in TASKS}
    ag = lambda t, n: A[t]['agreement'][n]['rate']
    sh = lambda t, n: A[t]['shift_share'][n]
    d1 = [abs(ag(t, 'relay') - ag(t, 'none')) for t in TASKS]
    v['E1_relay_alone_moves_no_answer'] = tri(max(d1) <= .01, max(d1) > .03)
    r = [sh(t, 'relay') for t in TASKS]
    v['E2_relay_alone_carries_a_small_score_shift'] = tri(all(.02 <= x['rate'] <= .30 and x['ci95'][0] > 0 for x in r),
                                                          any(x['ci95'][0] <= 0 or x['rate'] > .45 for x in r))
    keep = [sh(t, 'relay:W13')['rate'] / sh(t, 'relay')['rate'] for t in TASKS]
    rnd = max(sh(t, f'relay:R{j}')['rate'] for t in TASKS for j in range(E.RANDOM_SETS))
    v['E3_written_by_the_same_heads'] = tri(min(keep) >= .6 and rnd <= .02, min(keep) < .3)
    g = [abs(ag(t, 'tok1') - ag(t, 'direct')) for t in TASKS]
    v['E4_no_answer_gain_over_direct'] = tri(min(ag(t, 'direct') for t in TASKS) >= .90 and max(g) <= .02,
                                             max(ag(t, 'tok1') - ag(t, 'direct') for t in TASKS) > .05)
    v['relay_W13_keep'] = [round(x, 4) for x in keep]
    return v


DEPS = (PT, NL, H25, RH, TR, Z, E, E.AR, E.F3, E.RL, RH.S)


def prereg():
    from deploy.storage_policy import checked_dir
    from tools import provenance
    from tools.power_analysis import check_phase
    checked_dir(OUT, 1 << 30, operation='来源绑定英文中转产物')
    assert not (OUT / 'prereg.json').exists()
    plan = D.checked(MODEL)
    mats, designs = {}, {}
    for task in TASKS:
        rows = build_new(task)
        arms = run_set(rows, f'{task}|new')
        save(f'{task}-new-materials.json', rows)
        save(f'{task}-design.json', arms)
        mats[task] = {'sha256': D.sha(OUT / f'{task}-new-materials.json'), 'families': 48, 'rows': len(rows)}
        designs[task] = {'sha256': D.sha(OUT / f'{task}-design.json'), 'arms': len(arms)}
    pilot = {t: D.read(OUT / f'{t}-pilot-summary.json') for t in TASKS}
    body = {'schema': 'source_binding_enrelay_prereg.v1', 'run_id': RUN, 'seed': SEED, 'created_at': datetime.now().astimezone().isoformat(),
            'base_plan_sha256': D.sha(SRC / 'plan.json'), 'model_snapshot': plan['models'][MODEL], 'code_sha256': D.sha(__file__),
            'dependency_sha256': {m.__name__: D.sha(m.__file__) for m in DEPS},
            'criteria': CRITERIA, 'frozen': {'W13': E.W13, 'random_write': E.random_sets(E.W13),
                                             'pilot_sha256': {t: D.sha(OUT / f'{t}-pilot-summary.json') for t in TASKS}},
            'materials': mats, 'design': designs,
            'power': {'unit': 'paired binary outcome per item (prediction right / wrong under direct and under no effect, the contrast that '
                              'makes E1 informative); alpha = 0.05',
                      'as_if_one_edit_per_item': {'expected_delta': 0.25, 'p_d': 0.30, **check_phase(0.25, 768, p_d=0.30)},
                      'caveat': 'E1 and E4 are bounds on small differences; with 9,216 edits per task the family-bootstrap intervals in the '
                                'pilot were +-0.01 or narrower'},
            'custody': 'each new panel is read first by the gate stage (behaviour only) and then once by the run stage.',
            'resources': {'gpu_minutes_estimate': 40, 'disk_GiB_estimate': 0.1, 'models_concurrent': 1,
                          'shape': 'chunks of 48 prompts; pilot peak 3.9 GiB, 9.2 minutes per task for 24 families'},
            'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    body['provenance']['model_revision'] = plan['models'][MODEL]
    save('prereg.json', body)
    print(json.dumps({'prereg_sha256': D.sha(OUT / 'prereg.json'), 'design': designs, 'power': body['power'],
                      'pilot_verdicts': verdicts(pilot, {t: pilot[t]['meta'] for t in TASKS})}), flush=True)


def checked():
    p = D.read(OUT / 'prereg.json')
    assert p['code_sha256'] == D.sha(__file__) and all(p['dependency_sha256'][m.__name__] == D.sha(m.__file__) for m in DEPS), \
        'source changed after pre-registration'
    assert all(p['materials'][t]['sha256'] == D.sha(OUT / f'{t}-new-materials.json') and p['design'][t]['sha256'] == D.sha(OUT / f'{t}-design.json')
               for t in TASKS)
    D.checked(MODEL)
    return p


def gate():
    checked()
    assert not (OUT / 'gate.json').exists(), 'new panels already read'
    out = {}
    for task in TASKS:
        rows = D.read(OUT / f'{task}-new-materials.json')
        results, meta = engine(rows, stage=f'gate-{task}')
        save(f'{task}-gate-items.json', results['base'])
        g = D.behavior(rows, results['base'])
        out[task] = {'items_sha256': D.sha(OUT / f'{task}-gate-items.json'), **g, **meta}
    save('gate.json', {'schema': 'source_binding_enrelay_gate.v1', 'run_id': RUN, 'tasks': out})
    print(json.dumps({t: {k: g[k] for k in ('passed', 'forced_accuracy', 'top1_accuracy')} for t, g in out.items()}), flush=True)


def run():
    from tools import provenance
    pre = checked()
    assert not (OUT / 'report.json').exists(), 'new panels already used'
    g = D.read(OUT / 'gate.json')['tasks']
    S5, metas = {}, {}
    for task in TASKS:
        if not g[task]['passed']:
            continue
        rows, arms = D.read(OUT / f'{task}-new-materials.json'), D.read(OUT / f'{task}-design.json')
        for a in arms.values():
            a['hide'] = {int(b): v for b, v in a.get('hide', {}).items()}
        results, meta = engine(rows, arms, stage=f'run-{task}')
        save(f'{task}-items.json', {k: [{'id': x['id'], 'pred_record': x['pred_record'], 'scores': x['scores']} for x in v] for k, v in results.items()})
        base = D.read(OUT / f'{task}-gate-items.json')
        meta['base_replay_max_score_difference'] = max(abs(a - b) for x, y in zip(results['base'], base) for a, b in zip(x['scores'], y['scores']))
        S5[task], metas[task] = E.summarise(results, arms, rows, PAIRS), meta
        print(task, json.dumps(meta), flush=True)
        print(E.show(S5[task], KS), flush=True)
    v = verdicts(S5, metas)
    report = {'schema': 'source_binding_enrelay_report.v1', 'run_id': RUN, 'gate_passed': {t: g[t]['passed'] for t in TASKS},
              'prereg_sha256': D.sha(OUT / 'prereg.json'), 'gate_sha256': D.sha(OUT / 'gate.json'),
              'items_sha256': {t: D.sha(OUT / f'{t}-items.json') for t in S5}, 'criteria': pre['criteria'], 'frozen': pre['frozen'],
              'verdicts': v, 'summary': S5, 'meta': metas, 'formal_evidence_promotion': 0,
              'custody': 'each new panel read once by the gate stage and once by this run',
              'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    save('report.json', report)
    path = provenance.write_result('SOURCE-BINDING-enrelay-holdout', {'verdicts': v, 'gate_passed': all(g[t]['passed'] for t in TASKS),
                                                                    'gate_passed_by_task': {t: g[t]['passed'] for t in TASKS},
                                                                    'report_sha256': D.sha(OUT / 'report.json')},
                                   model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)
    print(json.dumps({'verdicts': v, 'result': str(path)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['smoke', 'pilot', 'materials', 'prereg', 'gate', 'run'])
    {'smoke': smoke, 'pilot': pilot, 'materials': materials, 'prereg': prereg, 'gate': gate, 'run': run}[ap.parse_args().stage]()
