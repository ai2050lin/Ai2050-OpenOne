"""Which part of the Chinese prompt weakens the direct reading (Claude Phase 438).  Qwen3-14B BF16.

Phase 437: with the same content and name-final questions, the direct-reading share is 0.878 for an English prompt (T2)
and 0.483 for a Chinese one (T3).  Here the Chinese prompt T3 is changed one part at a time to English, and the English
prompt T2 gets Chinese-character names:
  M1  T3 with the English answer cue (Answer:)          M2  T3 with the English instruction line
  M3  T3 with the English question line (Chinese names)  M4  T3 with English intro, list headers and records
  M5  English prompt (as T2) with Chinese-character team and personal names, English animals
  T3  re-measured on the same new name pools (baseline)
Engine, arms and measures are those of Phase 437 (source_binding_zhcause.py), unchanged.
Stages: materials, pilotgate, pilot (pilot pools) -> prereg -> gate -> run (holdout pools).
"""
import argparse, json, sys
from datetime import datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_binding_design as D
import source_binding_transfer as TR
import source_binding_zhcause as C

SRC, MODEL = D.OUT, 'qwen14'
OUT = D.ROOT / 'tests/glm5/result/source_binding_langparts_v1'
RUN, SEED = 'SOURCE-BINDING-LANGPARTS-20261114-v1', 202611140438
TASKS = ('T3n', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7')
ZH, EN = C.TEXT['T3'], C.TEXT['T2']
EN_ZHVAL = {'record': [lambda it, v: f'{it} got the {v}', lambda it, v: f'{it}: {v}', lambda it, v: f'the animal for {it} is the {v}',
                       lambda it, v: f'the {v} went to {it}']}
NEW = {'sources': {'pilot': TR.POOLS['zh']['sources']['pilot'] + TR.POOLS['zh']['sources']['holdout'],
                   'holdout': ['山队', '河队', '林队', '海队', '风队', '雨队', '雷队', '石队']},
       'items': {'pilot': TR.POOLS['zh']['items']['pilot'] + TR.POOLS['zh']['items']['holdout'],
                 'holdout': ['于洋', '白杨', '任静', '左岩', '易航', '华晨', '龙飞', '米兰', '祁峰', '邹琴', '乔木', '段雯', '岳山', '庞博']}}
TEXT = {
    'T3n': dict(ZH),
    'M1': {**ZH, 'answer': EN['answer']},
    'M2': {**ZH, 'tail': EN['tail']},
    'M3': {**ZH, 'qprefix': EN['qprefix'], 'question': EN['question']},
    'M4': {**ZH, 'intro': EN['intro'], 'header': EN['header'], 'record': EN_ZHVAL['record']},
    'M5': dict(EN),
    'M6': {**ZH, 'tail': EN['tail'], 'answer': EN['answer']},       # added after the first pilot: both late parts English
    'M7': {**EN, 'record': EN_ZHVAL['record'], 'tail': ZH['tail'], 'answer': ZH['answer']},       # added after the first pilot: English prompt, both late parts Chinese
}
for t in TASKS:
    C.TEXT[t] = TEXT[t]
    TR.TEXT[t] = TEXT[t]
    C.POOLS[t] = {**NEW, 'values': C.EN_POOLS['values'] if t == 'M5' else TR.POOLS['zh']['values']}   # M7 v2: Chinese animals (with English animals after the Chinese cue the pilot top-1 was 0.48)
C.TASKS = TASKS
C.RUN, C.SEED = RUN, SEED          # name draws and donor draws of this run


def save(name, body):
    from deploy.storage_policy import check_write
    p = OUT / name
    check_write(p, 1 << 20, operation='来源绑定语言成分产物')
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(body, ensure_ascii=False, indent=1, default=lambda x: x.item() if isinstance(x, np.generic) else x), encoding='utf-8')
    t.replace(p)


def engine(rows, arms=None, stage='run'):
    C.OUT = OUT
    return C.engine(rows, arms, stage)


def build(task, split):
    """C.build, but a family is kept only if its 16 layouts have equal token counts overall and for both question names, judged
    on the actual tokenization (mixed-script prompts tokenize names differently next to different characters)."""
    import source_binding_relay as RL
    import source_binding_arbitration as AR
    tok = RL.tokenizer()
    P, rows, seen = C.POOLS[task], [], set()
    for family in range(C.FAMILIES[split]):
        rnd = C.stream(f'{task}|{split}', family)
        for _ in range(2000):
            sources = [str(v) for v in rnd.choice(P['sources'][split], 2, replace=False)]
            items = [str(v) for v in rnd.choice(P['items'][split], 2, replace=False)]
            key = (tuple(sorted(sources)), tuple(sorted(items)))
            if key in seen:
                continue
            seen.add(key)
            values = [str(v) for v in rnd.choice(P['values'], 4, replace=False)]
            fam = []
            for s in (0, 1):
                for e in (0, 1):
                    for o in (0, 1):
                        for x in (0, 1):
                            r = C.material(task, family, split, s, e, o, x, family % 4, sources, items, values)
                            prompt = tok.apply_chat_template([{'role': 'user', 'content': r['content']}], tokenize=False,
                                                             add_generation_prompt=True, enable_thinking=False) + C.TEXT[task]['answer']
                            enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
                            ids, off = enc['input_ids'], prompt.index(r['content'])
                            span = lambda a, b: [i for i, (lo, hi) in enumerate(enc['offset_mapping']) if hi > off + a and lo < off + b]
                            choice = []
                            for val in values:
                                suffix = tok.encode(prompt + val, add_special_tokens=False)
                                assert suffix[:-1] == ids, (task, val)
                                choice.append(suffix[-1])
                            assert len(set(choice)) == 4
                            up = {'qsrc': span(*r['query_source_span']), 'qent': span(*r['query_name_span'])}
                            assert up['qsrc'] and up['qent'] and not set(up['qsrc']) & set(up['qent'])
                            fam.append({**r, 'records': {str(k): {kk: span(a, b) for kk, (a, b) in v.items()} for k, v in r['records'].items()},
                                        'headers': {str(k): span(a, b) for k, (a, b) in r['headers'].items()}, 'prompt': prompt,
                                        'tokens': ids, 'choice_ids': choice, 'gold_id': choice[r['target_record']], 'up': up})
            sig = {(len(r['tokens']), len(r['up']['qsrc']), len(r['up']['qent']), len(AR.holders_of(r))) for r in fam}
            if len(sig) == 1:
                rows += fam
                break
        else:
            raise RuntimeError(f'{task} {split}: no usable name combination for family {family}')
    assert len({tuple(r['tokens']) for r in rows}) == len(rows)
    rows = C.Z.annotate(rows)
    C.H25.RUN, C.H25.SEED = RUN, SEED
    C.H25.draws(rows)
    return rows


def materials(tasks=TASKS):
    from deploy.storage_policy import checked_dir
    checked_dir(OUT, 1 << 30, operation='来源绑定语言成分产物')
    for task in tasks:
        rows = build(task, 'pilot')
        save(f'{task}-pilot-materials.json', rows)
        r = rows[5]
        print(task, len(rows), 'qsrc', sorted({len(x['up']['qsrc']) for x in rows}), 'qent', sorted({len(x['up']['qent']) for x in rows}), flush=True)
        print('  ' + r['prompt'].replace('\n', ' | ')[-260:], '->', r['gold'], flush=True)


def pilotgate(tasks=TASKS):
    out = {}
    for task in tasks:
        rows = D.read(OUT / f'{task}-pilot-materials.json')
        results, meta = engine(rows, stage=f'pilotgate-{task}')
        g = D.behavior(rows, results['base'])
        out[task] = {k: g[k] for k in ('passed', 'forced_accuracy', 'top1_accuracy')}
        out[task]['worst'] = min((v['forced_accuracy'], v['top1_accuracy'], k) for k, v in g['cells'].items())
        save(f'{task}-pilot-gate.json', {**g, **meta})
    print(json.dumps(out, ensure_ascii=False), flush=True)


def pilot(tasks=TASKS):
    for task in tasks:
        if not D.read(OUT / f'{task}-pilot-gate.json')['passed']:
            print(task, 'pilot behaviour gate failed; skipped', flush=True)
            continue
        rows = D.read(OUT / f'{task}-pilot-materials.json')
        arms = C.arms_for(rows, f'{task}|pilot')
        results, meta = engine(rows, arms, stage=f'pilot-{task}')
        s = C.summarise(results, arms, rows)
        save(f'{task}-pilot-summary.json', {'meta': meta, **s})
        print(json.dumps(meta), flush=True)
        print(C.show(task, s), flush=True)



# ---- frozen after the pilots (pilot pools; M6, M7 added after the first pilot, M7 v2 with Chinese animals) ----
THRESHOLDS = {
    'P1_late_parts_in_english_restore_it': 'M6 (Chinese prompt, English instruction line and answer cue) direct share >= 0.75 and its '
                                           'lower bound above the upper bound of T3n. not_supported if the intervals overlap.',
    'P2_late_parts_in_chinese_remove_it': 'M7 (English prompt, Chinese instruction line and answer cue) direct share <= 0.45 and its '
                                          'upper bound below the lower bound of M5 (English prompt, same names). not_supported if they '
                                          'overlap.',
    'P3_each_late_part_counts': 'M1 (only the cue English) and M2 (only the instruction English) each have a lower bound above the upper '
                                'bound of T3n and an upper bound below the lower bound of M6. not_supported if M1 or M2 overlaps T3n.',
    'P4_question_and_name_script_do_not': 'the direct shares of M3 (English question) and T3n differ by <= 0.08, and M5 (English prompt '
                                          'with Chinese-character names) >= 0.80. not_supported if |M3 - T3n| > 0.15 or M5 < 0.70.',
    'P5_english_context_does_not_help': 'M4 (English intro, headers and records; Chinese question, instruction and cue) direct share <= '
                                        'that of T3n. not_supported if M4 exceeds T3n by more than 0.05.',
}
CRITERIA = {
    'model': 'Qwen3-14B original BF16, CPU byte reads, CUDA layer streaming. GLM-4-9B is not run.',
    'material': 'eight tasks with the structure of Phase 426 and the zh content (team names, personal names, animals), names ending the '
                'question: T3n Chinese prompt (T3 of Phase 437); M1 T3n with the English answer cue; M2 T3n with the English instruction '
                'line; M3 T3n with the English question line; M4 T3n with English intro, list headers and records; M5 English prompt (T2 '
                'of Phase 437) with Chinese-character names and English animals; M6 T3n with English instruction line and cue; M7 M5 with '
                'Chinese instruction line and cue and Chinese animals. Holdout: 48 families per task (768 prompts), Chinese names from '
                'pools disjoint from the pilot pools and from every earlier zh pool; inside a family the 16 layouts have equal token '
                'counts overall and for both question names (actual tokenization).',
    'broad_edit_and_arms': 'as in Phase 437 (source_binding_zhcause.py, unchanged): names exchanged at block 23 or 24; direct = positions '
                           'between the first question name and the answer reset to the base run after every block; via:end = the five '
                           'end-of-turn tokens also free; 2 donor relations per item; 6 cells x 1,536 edits per task.',
    'measures': 'direct share = share of the broad edit\'s score shift reached by direct, family-bootstrap 95 % intervals; tasks are '
                'compared by interval overlap (families differ between tasks).',
    'controls': 'as in Phase 437, with the near-tie rule.',
    'two_pass': 'gate stage, per task: unmodified forward and the v1 behaviour gate. Edits run only for a task whose gate passed.',
    'predictions_on_holdout': THRESHOLDS,
    'not_covered': 'whether it is the instruction text, its script, or only its last tokens before the template that matters; other '
                   'models.',
    'disclosure': 'first pilot with T3n and M1-M5 (16 families each): direct shares 0.486, 0.662, 0.593, 0.452, 0.315, 0.880. M6 and M7 '
                  'were added afterwards to test the reading; M7 failed its pilot gate on top-1 (0.48: after the Chinese cue the model '
                  'answers in Chinese) and was changed to Chinese animals (v2, pilot gate 1.000). Pilot M6 0.845, M7 0.298. Thresholds '
                  'were set after these pilots.',
}


def verdicts(S5, metas):
    tri = lambda ok, bad: 'supported' if ok else 'not_supported' if bad else 'inconclusive'
    v = {f'{t}_controls_valid': bool(metas.get(t, {}).get('controls_ok')) for t in TASKS}
    have = {t for t in TASKS if t in S5 and v[f'{t}_controls_valid']}
    sh = lambda t: S5[t]['shift_share']['direct']
    lo = lambda t: sh(t)['ci95'][0]
    hi = lambda t: sh(t)['ci95'][1]
    r = lambda t: sh(t)['rate']
    need = lambda *ts: set(ts) <= have
    na = 'behavior_ineligible_or_controls_invalid'
    v['P1_late_parts_in_english_restore_it'] = tri(r('M6') >= .75 and lo('M6') > hi('T3n'), lo('M6') <= hi('T3n')) if need('M6', 'T3n') else na
    v['P2_late_parts_in_chinese_remove_it'] = tri(r('M7') <= .45 and hi('M7') < lo('M5'), hi('M7') >= lo('M5')) if need('M7', 'M5') else na
    v['P3_each_late_part_counts'] = tri(all(lo(t) > hi('T3n') and hi(t) < lo('M6') for t in ('M1', 'M2')),
                                        any(lo(t) <= hi('T3n') for t in ('M1', 'M2'))) if need('M1', 'M2', 'M6', 'T3n') else na
    v['P4_question_and_name_script_do_not'] = tri(abs(r('M3') - r('T3n')) <= .08 and r('M5') >= .80,
                                                  abs(r('M3') - r('T3n')) > .15 or r('M5') < .70) if need('M3', 'T3n', 'M5') else na
    v['P5_english_context_does_not_help'] = tri(r('M4') <= r('T3n'), r('M4') > r('T3n') + .05) if need('M4', 'T3n') else na
    return v


DEPS = (C, C.RL, C.F3, C.H25, C.RH, C.TR, C.Z, C.RH.S, C.Z.AR)


def prereg():
    from tools import provenance
    from tools.power_analysis import check_phase
    assert not (OUT / 'prereg.json').exists()
    plan = D.checked(MODEL)
    mats, designs = {}, {}
    for task in TASKS:
        rows = build(task, 'holdout')
        pw = set(NEW['sources']['pilot'] + NEW['items']['pilot'])
        assert len(rows) == 768 and not any(w in pw for r in rows for w in r['sources'] + r['names'])
        arms = C.arms_for(rows, f'{task}|holdout')
        save(f'{task}-holdout-materials.json', rows)
        save(f'{task}-design.json', arms)
        mats[task] = {'sha256': D.sha(OUT / f'{task}-holdout-materials.json'), 'families': 48, 'rows': len(rows)}
        designs[task] = {'sha256': D.sha(OUT / f'{task}-design.json'), 'arms': len(arms)}
    pilot = {t: D.read(OUT / f'{t}-pilot-summary.json') for t in TASKS}
    body = {'schema': 'source_binding_langparts_prereg.v1', 'run_id': RUN, 'seed': SEED, 'created_at': datetime.now().astimezone().isoformat(),
            'base_plan_sha256': D.sha(SRC / 'plan.json'), 'model_snapshot': plan['models'][MODEL], 'code_sha256': D.sha(__file__),
            'dependency_sha256': {m.__name__: D.sha(m.__file__) for m in DEPS},
            'criteria': CRITERIA, 'frozen': {'pilot_sha256': {t: D.sha(OUT / f'{t}-pilot-summary.json') for t in TASKS}},
            'materials': mats, 'design': designs,
            'power': {'unit': 'paired binary outcome per item within a task (prediction right / wrong under direct and under no effect); '
                              'alpha = 0.05',
                      'as_if_one_edit_per_item': {'expected_delta': 0.10, 'p_d': 0.30, **check_phase(0.10, 768, p_d=0.30)},
                      'caveat': 'the predictions compare direct shares between tasks by interval overlap; pilot intervals were about '
                                '+-0.02 to +-0.05 with 16 families'},
            'custody': 'each holdout panel is read first by the gate stage (behaviour only) and then once by the run stage.',
            'resources': {'gpu_minutes_estimate': 60, 'disk_GiB_estimate': 0.15, 'models_concurrent': 1,
                          'shape': 'chunks of 96 prompts; pilot peak 3.6 GiB, about 2.7 minutes per task for 16 families'},
            'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    body['provenance']['model_revision'] = plan['models'][MODEL]
    save('prereg.json', body)
    print(json.dumps({'prereg_sha256': D.sha(OUT / 'prereg.json'), 'design': designs, 'power': body['power'],
                      'pilot_verdicts': verdicts(pilot, {t: pilot[t]['meta'] for t in TASKS})}), flush=True)


def checked():
    p = D.read(OUT / 'prereg.json')
    assert p['code_sha256'] == D.sha(__file__) and all(p['dependency_sha256'][m.__name__] == D.sha(m.__file__) for m in DEPS), \
        'source changed after pre-registration'
    assert all(p['materials'][t]['sha256'] == D.sha(OUT / f'{t}-holdout-materials.json') and p['design'][t]['sha256'] == D.sha(OUT / f'{t}-design.json')
               for t in TASKS)
    D.checked(MODEL)
    return p


def gate():
    checked()
    out = D.read(OUT / 'gate.json')['tasks'] if (OUT / 'gate.json').exists() else {}
    for task in TASKS:                    # resumable per task (GPU lease), each task read once
        if task in out:
            continue
        rows = D.read(OUT / f'{task}-holdout-materials.json')
        results, meta = engine(rows, stage=f'gate-{task}')
        save(f'{task}-gate-items.json', results['base'])
        g = D.behavior(rows, results['base'])
        out[task] = {'items_sha256': D.sha(OUT / f'{task}-gate-items.json'), **g, **meta}
        save('gate.json', {'schema': 'source_binding_langparts_gate.v1', 'run_id': RUN, 'tasks': out})
    print(json.dumps({t: {k: g[k] for k in ('passed', 'forced_accuracy', 'top1_accuracy')} for t, g in out.items()}), flush=True)


def run():
    from tools import provenance
    pre = checked()
    assert not (OUT / 'report.json').exists(), 'holdout panels already used'
    g = D.read(OUT / 'gate.json')['tasks']
    assert set(g) == set(TASKS)
    for task in TASKS:                    # resumable per task: a finished task keeps its saved items
        if not g[task]['passed'] or (OUT / f'{task}-summary.json').exists():
            continue
        rows, arms = D.read(OUT / f'{task}-holdout-materials.json'), D.read(OUT / f'{task}-design.json')
        results, meta = engine(rows, arms, stage=f'run-{task}')
        save(f'{task}-items.json', {k: [{'id': x['id'], 'pred_record': x['pred_record'], 'scores': x['scores']} for x in v] for k, v in results.items()})
        base = D.read(OUT / f'{task}-gate-items.json')
        meta['base_replay_max_score_difference'] = max(abs(a - b) for x, y in zip(results['base'], base) for a, b in zip(x['scores'], y['scores']))
        save(f'{task}-summary.json', {'meta': meta, **C.summarise(results, arms, rows)})
        print(json.dumps(meta), flush=True)
    S5, metas = {}, {}
    for task in TASKS:
        if g[task]['passed']:
            x = D.read(OUT / f'{task}-summary.json')
            metas[task] = x.pop('meta')
            S5[task] = x
            print(C.show(task, x), flush=True)
    v = verdicts(S5, metas)
    report = {'schema': 'source_binding_langparts_report.v1', 'run_id': RUN, 'gate_passed': {t: g[t]['passed'] for t in TASKS},
              'prereg_sha256': D.sha(OUT / 'prereg.json'), 'gate_sha256': D.sha(OUT / 'gate.json'),
              'items_sha256': {t: D.sha(OUT / f'{t}-items.json') for t in S5}, 'criteria': pre['criteria'],
              'verdicts': v, 'summary': S5, 'meta': metas, 'formal_evidence_promotion': 0,
              'custody': 'each holdout panel read once by the gate stage and once by this run',
              'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    save('report.json', report)
    path = provenance.write_result('SOURCE-BINDING-langparts-holdout', {'verdicts': v, 'gate_passed': all(g[t]['passed'] for t in TASKS),
                                                                      'gate_passed_by_task': {t: g[t]['passed'] for t in TASKS},
                                                                      'report_sha256': D.sha(OUT / 'report.json')},
                                   model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)
    print(json.dumps({'verdicts': v, 'result': str(path)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['materials', 'pilotgate', 'pilot', 'pilot_M67', 'pilot_M7', 'prereg', 'gate', 'run'])
    {'materials': materials, 'pilotgate': pilotgate, 'pilot': pilot,
     'pilot_M67': lambda: (materials(('M6', 'M7')), pilotgate(('M6', 'M7')), pilot(('M6', 'M7'))),
     'pilot_M7': lambda: (materials(('M7',)), pilotgate(('M7',)), pilot(('M7',))), 'prereg': prereg, 'gate': gate, 'run': run}[ap.parse_args().stage]()
