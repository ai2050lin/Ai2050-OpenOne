"""Why the direct reading is weak in the Chinese task (Claude Phase 437).  Qwen3-14B BF16.

Phases 433 / 436: the share of the broad edit's score shift reached by the answer position's direct reading of the
names is 0.48 in the Chinese task (zh) and 0.89 in English (colours, original task).  zh and colours differ in three
things at once: language, content (team names, personal names, animals vs first names, objects, colours) and question
structure (in zh the item name is followed by more question text; in English the name ends the question).
Three intermediate tasks change one factor at a time:
  T1  English, zh content (team names, pinyin personal names, animals), zh question order (more text after the name)
  T2  as T1, but the name ends the question
  T3  Chinese, zh content, but the name ends the question
Measure: per task, the position-restricted broad edit of Phase 433 with nothing free (direct), only the five end-of-turn
tokens free (via:end) and everything free (swap); agreement and score-shift share.
Stages: materials, pilotgate, pilot (pilot pools) -> prereg -> gate -> run (holdout pools).
"""
import argparse, hashlib, json, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_binding_design as D
import source_binding_relay as RL
import source_binding_fidelity3 as F3
import source_binding_h2518 as H25
import source_binding_readheads as RH
import source_binding_transfer as TR
import source_binding_zhroute as Z

SRC, MODEL = D.OUT, 'qwen14'
OUT = D.ROOT / 'tests/glm5/result/source_binding_zhcause_v1'
RUN, SEED = 'SOURCE-BINDING-ZHCAUSE-20261114-v1', 202611140437
TASKS = ('T1', 'T2', 'T3')
FAMILIES = {'pilot': 16, 'holdout': 48}
REPS, CELLS = 2, Z.CELLS
EN_POOLS = {'sources': {'pilot': ['Red Team', 'Blue Team', 'Green Team', 'Gold Team'],
                        'holdout': ['White Team', 'Black Team', 'Silver Team', 'Purple Team', 'Orange Team', 'Grey Team']},
            'items': {'pilot': ['Wang Qiang', 'Li Na', 'Zhang Wei', 'Liu Yang', 'Chen Jing'],
                      'holdout': ['Yang Fan', 'Zhao Lei', 'Zhou Min', 'Wu Gang', 'Sun Li', 'Zheng Hao', 'Feng Xue', 'Xu Ting', 'He Jun', 'Gao Jie']},
            'values': ['cat', 'dog', 'horse', 'cow', 'lion', 'bird', 'fish', 'fox', 'rabbit', 'snake', 'pig', 'duck', 'bear', 'wolf', 'deer', 'mouse']}
EN = {'intro': 'Two teams each assigned an animal to the same people. Use only the list of the team named in the question.\n',
      'header': lambda src: f"{src}'s list:\n",
      'record': [lambda it, v: f'{it} got the {v}', lambda it, v: f'{it}: {v}', lambda it, v: f'the animal for {it} is the {v}',
                 lambda it, v: f'the {v} went to {it}'],
      'qprefix': 'Question: ', 'tail': '\nReply with the animal only.', 'answer': 'Answer:\n'}
TEXT = {
    'T1': {**EN, 'question': [lambda s, i: f"According to {s}'s list, {i} got which animal in the end?",
                              lambda s, i: f"{i} corresponds to which animal on the list of {s}?",
                              lambda s, i: f"Please look at {s}'s list: {i} was given what animal?",
                              lambda s, i: f"Looking at {s}'s list, {i} has which animal next to the name?"]},   # v2: replaces "First find {i}, then look at {s}'s list: which animal is it?" (pilot gate 0.953 in that wording)
    'T2': {**EN, 'question': [lambda s, i: f"According to {s}'s list, which animal was given to {i}?",
                              lambda s, i: f"On the list of {s}, which animal belongs to {i}?",
                              lambda s, i: f"Please look at {s}'s list: what animal went to {i}?",
                              lambda s, i: f"Look at {s}'s list. Which animal is assigned to {i}?"]},
    'T3': {**TR.TEXT['zh'], 'qprefix': '问题：',
           'question': [lambda s, i: f'按照{s}的名单，哪种动物分给了{i}？', lambda s, i: f'在{s}的名单里，哪种动物对应{i}？',
                        lambda s, i: f'请看{s}的名单，什么动物分给了{i}？', lambda s, i: f'看{s}的名单，哪种动物属于{i}？']},
}
T3_POOLS = {'sources': {'pilot': TR.POOLS['zh']['sources']['pilot'], 'holdout': ['雪队', '云队', '星队', '月队', '光队']},
            'items': {'pilot': TR.POOLS['zh']['items']['pilot'],
                      'holdout': ['丁洋', '卢斌', '孔雪', '田甜', '金鑫', '夏雨', '钱程', '韦华', '柳青', '余晖']},
            'values': TR.POOLS['zh']['values']}     # holdout pools disjoint from every earlier zh pool
POOLS = {'T1': EN_POOLS, 'T2': EN_POOLS, 'T3': T3_POOLS}
for t in TASKS:
    TR.TEXT[t] = TEXT[t]           # Z.annotate reads the instruction tail from TR.TEXT


def stream(purpose, key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(f'{RUN}|{SEED}|{purpose}|{key}'.encode()).digest()[:16], 'little'))


def save(name, body):
    from deploy.storage_policy import check_write
    p = OUT / name
    check_write(p, 1 << 20, operation='来源绑定中文原因产物')
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(body, ensure_ascii=False, indent=1, default=lambda x: x.item() if isinstance(x, np.generic) else x), encoding='utf-8')
    t.replace(p)


def material(task, family, split, s, e, o, x, frame, sources, items, values):
    T = TEXT[task]
    pieces, records, heads, offset = [], {}, {}, 0

    def add(text):
        nonlocal offset
        span = (offset, offset + len(text))
        pieces.append(text)
        offset += len(text)
        return span
    add(T['intro'])
    for src in ([0, 1] if o == 0 else [1, 0]):
        heads[src] = add(T['header'](sources[src]))
        for it in ([0, 1] if (family + src + x) % 2 == 0 else [1, 0]):
            text = T['record'][frame](items[it], values[src * 2 + it])
            a, b = add(text + '\n')
            vs = a + text.index(values[src * 2 + it])
            records[src * 2 + it] = {'span': [a, b - 1], 'value_span': [vs, vs + len(values[src * 2 + it])]}
    q = T['question'][frame](sources[s], items[e])
    add(T['qprefix'])
    qa, _ = add(q)
    add(T['tail'])
    return {'id': f'{task}_{split}-{family:04d}-S{s}E{e}O{o}X{x}', 'family': f'{task}_{split}-{family:04d}', 'task': task, 'frame': frame,
            'source': s, 'entity': e, 'order': o, 'x': x, 'names': items, 'sources': sources, 'values': values, 'gold': values[2 * s + e],
            'target_record': 2 * s + e, 'content': ''.join(pieces), 'records': records, 'headers': heads,
            'query_source_span': [qa + q.index(sources[s]), qa + q.index(sources[s]) + len(sources[s])],
            'query_name_span': [qa + q.index(items[e]), qa + q.index(items[e]) + len(items[e])]}


def build(task, split):
    tok = RL.tokenizer()
    P, rows, seen = POOLS[task], [], set()
    width = lambda w: len(tok.encode(('' if task == 'T3' else ' ') + w, add_special_tokens=False))
    for family in range(FAMILIES[split]):
        rnd = stream(f'{task[:2] if task != "T3" else "T3"}|{split}', family)
        while True:                       # equal token counts inside a family, so that every donor relation is usable
            sources = [str(v) for v in rnd.choice(P['sources'][split], 2, replace=False)]
            items = [str(v) for v in rnd.choice(P['items'][split], 2, replace=False)]
            key = (tuple(sorted(sources)), tuple(sorted(items)))
            if key not in seen and width(sources[0]) == width(sources[1]) and width(items[0]) == width(items[1]):
                seen.add(key)
                break
        values = [str(v) for v in rnd.choice(P['values'], 4, replace=False)]
        for s in (0, 1):
            for e in (0, 1):
                for o in (0, 1):
                    for x in (0, 1):
                        r = material(task, family, split, s, e, o, x, family % 4, sources, items, values)
                        prompt = tok.apply_chat_template([{'role': 'user', 'content': r['content']}], tokenize=False,
                                                         add_generation_prompt=True, enable_thinking=False) + TEXT[task]['answer']
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
                        rows.append({**r, 'records': {str(k): {kk: span(a, b) for kk, (a, b) in v.items()} for k, v in r['records'].items()},
                                     'headers': {str(k): span(a, b) for k, (a, b) in r['headers'].items()}, 'prompt': prompt,
                                     'tokens': ids, 'choice_ids': choice, 'gold_id': choice[r['target_record']], 'up': up})
    assert len({tuple(r['tokens']) for r in rows}) == len(rows)
    rows = Z.annotate(rows)
    H25.RUN, H25.SEED = RUN, SEED
    H25.draws(rows)
    return rows


def arms_for(rows, key):
    dons, rels = RH.donors(rows, f'{RUN}|{key}', SEED)
    me = list(range(len(rows)))
    arms = {'self': {'swap': ['names', 23], 'donor': me, 'free': list(Z.GROUPS)}, 'self:direct': {'swap': ['names', 23], 'donor': me, 'free': []}}
    for k in range(REPS):
        base = {'donor': dons[k], 'rel': rels[k]}
        for cell in CELLS:
            site, l = cell.split('@')[0], int(cell.split('@')[1])
            sw = {'swap': [site, l], **base}
            arms[f'swap|{cell}#{k}'] = {**sw, 'free': list(Z.GROUPS)}
            arms[f'direct|{cell}#{k}'] = {**sw, 'free': []}
            arms[f'via:end|{cell}#{k}'] = {**sw, 'free': ['end']}
    return arms


ARMS = ['direct', 'via:end']


def engine(rows, arms=None, stage='run'):
    """Phase 433 engine; the control rule allows a changed choice only on near ties of the base run (top-two gap <= 1.0)."""
    Z.OUT = OUT
    results, meta = Z.engine(rows, arms, stage)
    if arms:
        gap = [sorted(x['scores'])[-1] - sorted(x['scores'])[-2] for x in results['base']]
        ok, ties = True, 0
        for k in ('self', 'self:direct'):
            for x, y, g in zip(results[k], results['base'], gap):
                if x['pred_record'] != y['pred_record']:
                    ties += 1
                    ok &= g <= Z.TOL
            ok &= meta[f'{k}_max_score_difference'] <= Z.TOL
        meta.update({'controls_ok_exact': meta['controls_ok'], 'control_changes_on_near_ties': ties, 'controls_ok': bool(ok)})
    return results, meta


def summarise(results, arms, rows):
    fam_id = {f: i for i, f in enumerate(sorted({r['family'] for r in rows}))}
    nf = len(fam_id)
    T = defaultdict(list)
    for cell in CELLS:
        site, l = cell.split('@')[0], int(cell.split('@')[1])
        for k in range(REPS):
            for i, (r, j) in enumerate(zip(rows, arms[f'swap|{cell}#{k}']['donor'])):
                B, P, I = F3.records(r, rows[j], site, l)
                o = results[f'swap|{cell}#{k}'][i]
                bs = results['base'][i]['scores']
                T['fam'].append(fam_id[r['family']]); T['B'].append(B); T['P'].append(P); T['out'].append(o['pred_record'])
                T['nat'].append(bs[P] - bs[B] if P != B else 0.0); T['xs'].append(o['scores'][P] - o['scores'][B] if P != B else 0.0)
                for n in ARMS:
                    it = results[f'{n}|{cell}#{k}'][i]
                    T[n].append(it['pred_record']); T['x:' + n].append(it['scores'][P] - it['scores'][B] if P != B else 0.0)
    T = {k: np.array(v) for k, v in T.items()}
    draws = np.random.default_rng(SEED).integers(0, nf, (1000, nf))
    fam, out, B, c = T['fam'], T['out'], T['B'], T['P'] != T['B']
    r4 = lambda v: round(float(v), 4)

    def rate(v):
        v = np.asarray(v, float)
        per, cnt = np.bincount(fam, weights=v, minlength=nf), np.bincount(fam, minlength=nf)
        bs = per[draws].sum(1) / cnt[draws].sum(1)
        return {'rate': r4(v.mean()), 'ci95': [r4(np.quantile(bs, .025)), r4(np.quantile(bs, .975))]}

    def share(n):
        num = np.bincount(fam[c], weights=(T['x:' + n] - T['nat'])[c], minlength=nf)
        den = np.bincount(fam[c], weights=(T['xs'] - T['nat'])[c], minlength=nf)
        bs = num[draws].sum(1) / den[draws].sum(1)
        return {'rate': r4(num.sum() / den.sum()), 'ci95': [r4(np.quantile(bs, .025)), r4(np.quantile(bs, .975))], 'per_family': [r4(a / b) for a, b in zip(num, den)]}
    hit = {n: (T[n] == out).astype(float) for n in ARMS}
    hit['none'] = (B == out).astype(float)
    return {'edits': int(len(out)), 'broad_changed': r4((out != B).mean()), 'agreement': {n: rate(v) for n, v in hit.items()},
            'shift_share': {n: share(n) for n in ARMS}, 'direct_minus_none': rate(hit['direct'] - hit['none']),
            'via_end_minus_direct': rate(hit['via:end'] - hit['direct'])}


def show(t, s):
    return (f"{t}: changed {s['broad_changed']:.3f}  agreement direct {s['agreement']['direct']['rate']:.3f}  via:end {s['agreement']['via:end']['rate']:.3f}"
            f"  none {s['agreement']['none']['rate']:.3f}  |  share direct {s['shift_share']['direct']['rate']:.3f} {s['shift_share']['direct']['ci95']}"
            f"  via:end {s['shift_share']['via:end']['rate']:.3f}")


def materials(tasks=TASKS):
    from deploy.storage_policy import checked_dir
    checked_dir(OUT, 1 << 30, operation='来源绑定中文原因产物')
    tok = RL.tokenizer()
    for task in tasks:
        rows = build(task, 'pilot')
        save(f'{task}-pilot-materials.json', rows)
        r = rows[5]
        print(task, len(rows), 'qsrc', sorted({len(x['up']['qsrc']) for x in rows}), 'qent', sorted({len(x['up']['qent']) for x in rows}),
              'qtail', sorted({len(x['grp']['qtail']) for x in rows}), flush=True)
        print(r['content'].split('\n')[-2], '->', r['gold'], '|', [tok.decode([r['tokens'][i]]) for i in r['grp']['qtail']], flush=True)


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
        arms = arms_for(rows, f'{task}|pilot')
        results, meta = engine(rows, arms, stage=f'pilot-{task}')
        s = summarise(results, arms, rows)
        save(f'{task}-pilot-summary.json', {'meta': meta, **s})
        print(json.dumps(meta), flush=True)
        print(show(task, s), flush=True)



# ---- frozen after the pilot (pilot pools; T1 wording v2) ----
REFERENCE = {'zh_direct_share_phase433': 0.4826, 'colours_direct_share_phase433': 0.8915}
THRESHOLDS = {
    'L1_language_is_the_factor': 'direct share of T2 (English) minus T3 (Chinese), same content and question structure, >= 0.25 and the '
                                 'intervals do not overlap (T3 upper bound < T2 lower bound). not_supported if they overlap.',
    'L2_structure_is_not_the_factor_in_chinese': 'T3 (Chinese, name ending the question) direct share <= 0.60 (the zh task: 0.48). '
                                                 'not_supported if > 0.70.',
    'L3_content_is_not_the_factor_in_english': 'T1 and T2 (English, zh content) direct shares >= 0.70 (colours: 0.89). not_supported '
                                               'if either < 0.60.',
    'L4_order_effect_small_in_english': 'the direct shares of T1 and T2 differ by <= 0.10. not_supported if > 0.20.',
    'L5_relay_returns_in_chinese_only': 'agreement: T3 via:end minus direct >= 0.15 with a lower bound > 0; T1 and T2 via:end minus '
                                        'direct <= 0.05. not_supported if the T3 lower bound <= 0.',
}
CRITERIA = {
    'model': 'Qwen3-14B original BF16, CPU byte reads, CUDA layer streaming. GLM-4-9B is not run.',
    'material': 'three new tasks with the structure of Phase 426 (two sources x two items, four records; 16 layouts per family, 4 '
                'wordings): T1 English with the zh content (team names, pinyin personal names, one-token English animals), the name '
                'followed by more question text as in zh; T2 as T1 with the name ending the question; T3 Chinese with the zh content '
                'and the name ending the question. Holdout: 48 families per task (768 prompts), pools disjoint from the pilot pools '
                '(and, for T3, from every earlier zh pool); answer words shared. Inside a family the two sources and the two items have '
                'equal token counts.',
    'broad_edit': 'as in Phase 433: at the input of block L (23, 24) the residual states of the source name in the question, the item '
                  'name or both are replaced by those of a donor of the same family; 2 seeded donor relations per item; 6 cells x '
                  '1,536 edits per task.',
    'arms': 'the Phase 433 engine, unchanged: direct (positions between the first question name and the answer reset to the base run '
            'after every block), via:end (the five end-of-turn tokens free), swap (nothing reset).',
    'measures': 'direct share = share of the broad edit\'s score shift (score(P) - score(B) on conflict edits) reached by direct; '
                'agreement with the broad edit\'s choice; family-bootstrap 95 % intervals (1000 draws). Across tasks the families '
                'differ, so tasks are compared by interval overlap.',
    'controls': 'the name exchange with the prompt itself, nothing reset and all reset, keeps the score difference <= 1.0 and the '
                'choice, except on near ties of the base run (top-two gap <= 1.0), which are counted and reported. This rule was '
                'added after the pilot, where 2 of 512 self-exchange answers of T2 flipped on such ties.',
    'two_pass': 'gate stage, per task: unmodified forward and the v1 behaviour gate. Edits run only for a task whose gate passed.',
    'predictions_on_holdout': THRESHOLDS,
    'not_covered': 'which part of the Chinese prompt (characters, tokenization of names and answers, the Chinese instruction or answer '
                   'cue) makes the direct reading weak; GLM-4-9B and other models.',
    'disclosure': 'pilot on 16 families per task (pilot pools). T1 failed the pilot behaviour gate (frame 3 "First find {i}, then look '
                  'at {s}\'s list: which animal is it?" at 0.953); that wording was replaced (v2) and the T1 pilot repeated (1.000). '
                  'Pilot direct shares: T1 0.791, T2 0.838, T3 0.469. Thresholds were set after the pilot. The zh and colours values '
                  'are references from Phase 433 (other panels), not re-measured.',
}


def verdicts(S5, metas):
    tri = lambda ok, bad: 'supported' if ok else 'not_supported' if bad else 'inconclusive'
    v = {f'{t}_controls_valid': bool(metas.get(t, {}).get('controls_ok')) for t in TASKS}
    have = {t for t in TASKS if t in S5 and v[f'{t}_controls_valid']}
    sh = lambda t: S5[t]['shift_share']['direct']
    if {'T2', 'T3'} <= have:
        d = sh('T2')['rate'] - sh('T3')['rate']
        v['L1_language_is_the_factor'] = tri(d >= .25 and sh('T3')['ci95'][1] < sh('T2')['ci95'][0], sh('T3')['ci95'][1] >= sh('T2')['ci95'][0])
    else:
        v['L1_language_is_the_factor'] = 'behavior_ineligible_or_controls_invalid'
    v['L2_structure_is_not_the_factor_in_chinese'] = tri(sh('T3')['rate'] <= .60, sh('T3')['rate'] > .70) if 'T3' in have else 'behavior_ineligible_or_controls_invalid'
    if {'T1', 'T2'} <= have:
        x = [sh('T1')['rate'], sh('T2')['rate']]
        v['L3_content_is_not_the_factor_in_english'] = tri(min(x) >= .70, min(x) < .60)
        v['L4_order_effect_small_in_english'] = tri(abs(x[0] - x[1]) <= .10, abs(x[0] - x[1]) > .20)
    else:
        v['L3_content_is_not_the_factor_in_english'] = v['L4_order_effect_small_in_english'] = 'behavior_ineligible_or_controls_invalid'
    if have == set(TASKS):
        g = S5['T3']['via_end_minus_direct']
        v['L5_relay_returns_in_chinese_only'] = tri(g['rate'] >= .15 and g['ci95'][0] > 0 and all(S5[t]['via_end_minus_direct']['rate'] <= .05 for t in ('T1', 'T2')),
                                                    g['ci95'][0] <= 0)
    else:
        v['L5_relay_returns_in_chinese_only'] = 'behavior_ineligible_or_controls_invalid'
    return v


DEPS = (RL, F3, H25, RH, TR, Z, RH.S, Z.AR)


def prereg():
    from tools import provenance
    from tools.power_analysis import check_phase
    assert not (OUT / 'prereg.json').exists()
    plan = D.checked(MODEL)
    mats, designs = {}, {}
    for task in TASKS:
        rows = build(task, 'holdout')
        pw = set(POOLS[task]['sources']['pilot'] + POOLS[task]['items']['pilot'])
        assert len(rows) == 768 and not any(w in pw for r in rows for w in r['sources'] + r['names'])
        arms = arms_for(rows, f'{task}|holdout')
        save(f'{task}-holdout-materials.json', rows)
        save(f'{task}-design.json', arms)
        mats[task] = {'sha256': D.sha(OUT / f'{task}-holdout-materials.json'), 'families': 48, 'rows': len(rows)}
        designs[task] = {'sha256': D.sha(OUT / f'{task}-design.json'), 'arms': len(arms)}
    pilot = {t: D.read(OUT / f'{t}-pilot-summary.json') for t in TASKS}
    body = {'schema': 'source_binding_zhcause_prereg.v1', 'run_id': RUN, 'seed': SEED, 'created_at': datetime.now().astimezone().isoformat(),
            'base_plan_sha256': D.sha(SRC / 'plan.json'), 'model_snapshot': plan['models'][MODEL], 'code_sha256': D.sha(__file__),
            'dependency_sha256': {m.__name__: D.sha(m.__file__) for m in DEPS},
            'criteria': CRITERIA, 'reference': REFERENCE,
            'frozen': {'pilot_sha256': {t: D.sha(OUT / f'{t}-pilot-summary.json') for t in TASKS}},
            'materials': mats, 'design': designs,
            'power': {'unit': 'paired binary outcome per item within T3 (prediction right / wrong under via:end and under direct); alpha = 0.05',
                      'L5_as_if_one_edit_per_item': {'expected_delta': 0.15, 'p_d': 0.25, **check_phase(0.15, 768, p_d=0.25)},
                      'caveat': 'L1-L4 compare direct shares between tasks by family-bootstrap intervals; pilot intervals were about '
                                '+-0.02 to +-0.04 with 16 families'},
            'custody': 'each holdout panel is read first by the gate stage (behaviour only) and then once by the run stage.',
            'resources': {'gpu_minutes_estimate': 25, 'disk_GiB_estimate': 0.1, 'models_concurrent': 1,
                          'shape': 'chunks of 96 prompts; pilot peak 3.2 GiB, 2.7 minutes per task for 16 families'},
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
    assert not (OUT / 'gate.json').exists(), 'holdout panels already read'
    out = {}
    for task in TASKS:
        rows = D.read(OUT / f'{task}-holdout-materials.json')
        results, meta = engine(rows, stage=f'gate-{task}')
        save(f'{task}-gate-items.json', results['base'])
        g = D.behavior(rows, results['base'])
        out[task] = {'items_sha256': D.sha(OUT / f'{task}-gate-items.json'), **g, **meta}
    save('gate.json', {'schema': 'source_binding_zhcause_gate.v1', 'run_id': RUN, 'tasks': out})
    print(json.dumps({t: {k: g[k] for k in ('passed', 'forced_accuracy', 'top1_accuracy')} for t, g in out.items()}), flush=True)


def run():
    from tools import provenance
    pre = checked()
    assert not (OUT / 'report.json').exists(), 'holdout panels already used'
    g = D.read(OUT / 'gate.json')['tasks']
    S5, metas = {}, {}
    for task in TASKS:
        if not g[task]['passed']:
            continue
        rows, arms = D.read(OUT / f'{task}-holdout-materials.json'), D.read(OUT / f'{task}-design.json')
        results, meta = engine(rows, arms, stage=f'run-{task}')
        save(f'{task}-items.json', {k: [{'id': x['id'], 'pred_record': x['pred_record'], 'scores': x['scores']} for x in v] for k, v in results.items()})
        base = D.read(OUT / f'{task}-gate-items.json')
        meta['base_replay_max_score_difference'] = max(abs(a - b) for x, y in zip(results['base'], base) for a, b in zip(x['scores'], y['scores']))
        S5[task], metas[task] = summarise(results, arms, rows), meta
        print(json.dumps(meta), flush=True)
        print(show(task, S5[task]), flush=True)
    v = verdicts(S5, metas)
    report = {'schema': 'source_binding_zhcause_report.v1', 'run_id': RUN, 'gate_passed': {t: g[t]['passed'] for t in TASKS},
              'prereg_sha256': D.sha(OUT / 'prereg.json'), 'gate_sha256': D.sha(OUT / 'gate.json'),
              'items_sha256': {t: D.sha(OUT / f'{t}-items.json') for t in S5}, 'criteria': pre['criteria'], 'reference': pre['reference'],
              'verdicts': v, 'summary': S5, 'meta': metas, 'formal_evidence_promotion': 0,
              'custody': 'each holdout panel read once by the gate stage and once by this run',
              'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    save('report.json', report)
    path = provenance.write_result('SOURCE-BINDING-zhcause-holdout', {'verdicts': v, 'gate_passed': all(g[t]['passed'] for t in TASKS),
                                                                    'gate_passed_by_task': {t: g[t]['passed'] for t in TASKS},
                                                                    'report_sha256': D.sha(OUT / 'report.json')},
                                   model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)
    print(json.dumps({'verdicts': v, 'result': str(path)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['materials', 'pilotgate', 'pilot', 'pilot_T1v2', 'pilot_T2', 'prereg', 'gate', 'run'])
    {'materials': materials, 'pilotgate': pilotgate, 'pilot': pilot,
     'pilot_T1v2': lambda: (materials(('T1',)), pilotgate(('T1',)), pilot(('T1',))), 'pilot_T2': lambda: pilot(('T2',)), 'prereg': prereg, 'gate': gate, 'run': run}[ap.parse_args().stage]()
