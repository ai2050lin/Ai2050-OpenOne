"""Do the read heads of the source-binding graph carry over to other tasks? (Claude Phase 426).  Qwen3-14B BF16.

Graph revision 5 was built on one task (two archives, two invented entity names, English code words).  Here the frozen
rule of Phases 423 / 424 is applied, without any new selection, to two other tasks with the same abstract structure
(two sources x two items -> four records; the question names one source and one item):

  colours  English, another domain and wording: two people each wrote a colour for the same objects.  Sources are first
           names, items are common nouns (single tokens), answers are colour words (none of the original code words).
  zh       Chinese: two teams' lists assign an animal to the same people; names, sources, answers and the whole prompt
           are Chinese.

Broad edit: residual states of the source name, the item name or both in the question replaced by a donor's at the
input of block 23, 24 or 25.  Rule: the broad edit gives the answer of a narrow edit in which the answer position reads
donor keys / values at these tokens through (a) every head of blocks L..27, (b) the 16 heads of SB.H.read16,
(c) random heads.  Heads and blocks are those frozen in Phase 424.
Stages: materials, pilotgate, pilot (exposed pilot panels) -> prereg -> gate -> run.
"""
import argparse, hashlib, json, sys
from datetime import datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_binding_design as D
import source_binding_relay as RL
import source_binding_fidelity3 as F3
import source_binding_h2518 as H25
import source_binding_sites as S
import source_binding_readheads as RH

SRC, MODEL = D.OUT, 'qwen14'
OUT = D.ROOT / 'tests/glm5/result/source_binding_transfer_v1'
RUN, SEED = 'SOURCE-BINDING-TRANSFER-20261113-v1', 202611130426
TASKS = ('colours', 'zh')
FAMILIES = {'pilot': 16, 'holdout': 48}
REPS, LAYERS, SITES, CELLS = 4, S.LAYERS, S.SITES, S.CELLS
POOLS = {
    'colours': {'sources': {'pilot': ['Alice', 'Henry', 'Maria', 'David', 'Sarah'], 'holdout': ['Peter', 'Laura', 'James', 'Emma', 'Oliver', 'Nina', 'Victor']},
                'items': {'pilot': ['lamp', 'chair', 'table', 'clock', 'mirror', 'basket'],
                          'holdout': ['bottle', 'blanket', 'ladder', 'candle', 'pillow', 'hammer', 'bucket', 'carpet', 'kettle', 'jacket']},
                'values': ['red', 'blue', 'green', 'black', 'white', 'pink', 'gold', 'grey', 'brown', 'purple', 'orange', 'yellow']},
    'zh': {'sources': {'pilot': ['红队', '蓝队', '绿队', '黄队'], 'holdout': ['白队', '黑队', '紫队', '金队', '银队', '青队']},
           'items': {'pilot': ['王强', '李娜', '张伟', '刘洋', '陈静'], 'holdout': ['杨帆', '赵磊', '周敏', '吴刚', '孙丽', '郑浩', '冯雪', '许婷', '何军', '高洁']},
           'values': ['猫', '狗', '马', '牛', '羊', '鸟', '鱼', '虎', '龙', '兔', '蛇', '猪', '鸡', '鸭', '熊', '狼']},
}
TEXT = {
    'colours': {'intro': 'Two people each wrote down a colour for the same objects. Trust only the person named in the question.\n',
                'header': lambda src: f"{src}'s list:\n",
                'record': [lambda it, v: f'the {it} is {v}', lambda it, v: f'{it}: {v}', lambda it, v: f'for the {it}, {v}', lambda it, v: f'{v} goes with the {it}'],
                'question': [lambda src, it: f'Which colour did {src} write for the {it}?', lambda src, it: f'Look at the {it}. What colour is it on the list of {src}?',
                             lambda src, it: f'In the list written by {src}, what colour does the {it} have?',
                             lambda src, it: f'Take the {it} and the list of {src}. Which colour is given?'],
                'tail': '\nReply with the colour only.', 'answer': 'Answer:\n'},
    'zh': {'intro': '两支队伍各自给同样的几个人分配了一种动物。只看问题里指定的那支队伍的名单。\n',
           'header': lambda src: f'{src}的名单：\n',
           'record': [lambda it, v: f'{it}分到的是{v}', lambda it, v: f'{it}：{v}', lambda it, v: f'给{it}的动物是{v}', lambda it, v: f'{v}分给了{it}'],
           'question': [lambda src, it: f'按照{src}的名单，{it}分到的动物是什么？', lambda src, it: f'{it}在{src}的名单里对应哪种动物？',
                        lambda src, it: f'请看{src}的名单，{it}分到了什么动物？', lambda src, it: f'先找到{it}，再看{src}的名单，是哪种动物？'],
           'tail': '\n只回答动物名称，不要解释。', 'answer': '答案：\n'},
}


def stream(purpose, key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(f'{RUN}|{SEED}|{purpose}|{key}'.encode()).digest()[:16], 'little'))


def save(name, body):
    from deploy.storage_policy import check_write
    p = OUT / name
    check_write(p, 1 << 20, operation='来源绑定换任务产物')
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(body, ensure_ascii=False, indent=1, default=lambda x: x.item() if isinstance(x, np.generic) else x), encoding='utf-8')
    t.replace(p)


def material(task, family, split, s, e, o, x, frame, sources, items, values):
    """Same schema as the v1 material: records 2*source+item, headers, question with one source and one item."""
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
            records[src * 2 + it] = {'span': [a, b - 1], 'value_span': [a + text.index(values[src * 2 + it]), a + text.index(values[src * 2 + it]) + len(values[src * 2 + it])]}
    q = T['question'][frame](sources[s], items[e])
    add('Question: ' if task == 'colours' else '问题：')
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
    for family in range(FAMILIES[split]):
        rnd = stream(f'{task}|{split}', family)
        while True:
            sources = [str(v) for v in rnd.choice(P['sources'][split], 2, replace=False)]
            items = [str(v) for v in rnd.choice(P['items'][split], 2, replace=False)]
            key = (tuple(sorted(sources)), tuple(sorted(items)))
            if key not in seen:
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
                            assert suffix[:-1] == ids, (task, val, 'answer word must be one token with a stable prefix')
                            choice.append(suffix[-1])
                        assert len(set(choice)) == 4
                        recs = {str(k): {key: span(a, b) for key, (a, b) in info.items()} for k, info in r['records'].items()}
                        up = {'qsrc': span(*r['query_source_span']), 'qent': span(*r['query_name_span'])}
                        assert up['qsrc'] and up['qent'] and not set(up['qsrc']) & set(up['qent']) and max(up['qsrc'] + up['qent']) < len(ids) - 1
                        rows.append({**r, 'records': recs, 'headers': {str(k): span(a, b) for k, (a, b) in r['headers'].items()}, 'prompt': prompt,
                                     'tokens': ids, 'choice_ids': choice, 'gold_id': choice[r['target_record']], 'up': up})
    assert len({tuple(r['tokens']) for r in rows}) == len(rows)
    return rows


def engine(rows, arms=None, stage='run'):
    S.OUT = OUT
    return S.engine(rows, arms, stage)


def final_arms(rows, key):
    dons, rels = RH.donors(rows, f'{RUN}|{key}', SEED)
    arms = {'self': {'donor': list(range(len(rows))), 'don': {b: [[['names', 'holders'], None]] for b in RH.BLOCKS}}}
    sets = RH.head_sets()                      # the frozen sets of Phase 424 (H16, H12, H9, two_edges, R0-R4), same seed
    for k in range(REPS):
        base = {'donor': dons[k], 'rel': rels[k]}
        for site in SITES:
            for l in LAYERS:
                cell = f'{site}@{l}'
                arms[f'swap|{cell}#{k}'] = {'swap': [site, l], **base}
                arms[f'all|{cell}#{k}'] = {'don': RH.spec_all(site, l), **base}
                arms[f'late|{cell}#{k}'] = {'don': {b: [[[site], None]] for b in range(28, 40)}, **base}
                arms[f'allH|{cell}#{k}'] = {'don': {b: [[[site, 'holders'], None]] for b in range(l, 28)}, **base}
                for name, hs in sets.items():
                    spec = RH.spec_heads(site, l, hs)
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
    hit = {**{v: (T[v] == out).astype(float) for v in names}, 'graph_v3': (T['v3'] == out).astype(float),
           'none': (T['B'] == out).astype(float), 'position': (T['P'] == out).astype(float)}
    rnd = [f'R{j}' for j in range(RH.RANDOM_SETS)]
    hit['random_mean'] = np.mean([hit[k] for k in rnd], 0)
    layer = np.array([int(c.split('@')[1]) for c in T['cell']])
    scopes = {'all': np.ones(len(out), bool), **{f'block_{l}': layer == l for l in LAYERS}, **{c: T['cell'] == c for c in CELLS}}
    s = {'edits': int(len(out)), 'model_choice_changed': r4((out != T['B']).mean()), 'scopes': {}}
    for name, keep in scopes.items():
        s['scopes'][name] = {'n': int(keep.sum()), 'model_choice_changed': r4((out != T['B'])[keep].mean()),
                             'agreement': {k: rate(v, keep) for k, v in hit.items()},
                             **{f'{a}_minus_{b}': rate(hit[a] - hit[b], keep) for a, b in (
                                 ('all', 'none'), ('all', 'late'), ('all', 'H16'), ('H16', 'random_mean'), ('H16', 'none'), ('H16', 'two_edges'),
                                 ('H16', 'graph_v3'), ('all', 'graph_v3'), ('allH', 'all'), ('allH', 'none'))}}
    s['random_max_overall'] = max(s['scopes']['all']['agreement'][k]['rate'] for k in rnd)
    return s


def show(s):
    ks = ['all', 'allH', 'H16', 'H12', 'two_edges', 'late', 'random_mean', 'graph_v3', 'position', 'none']
    lines = ['scope      chg    ' + ' '.join(f'{k[:6]:>6s}' for k in ks)]
    for n, v in s['scopes'].items():
        lines.append(f"{n:10s} {v['model_choice_changed']:.3f}  " + ' '.join(f"{v['agreement'][k]['rate']:6.3f}" for k in ks))
    return '\n'.join(lines)


def materials():
    from deploy.storage_policy import checked_dir
    checked_dir(OUT, 1 << 30, operation='来源绑定换任务产物')
    for task in TASKS:
        rows = build(task, 'pilot')
        save(f'{task}-pilot-materials.json', rows)
        r = rows[5]
        print(task, len(rows), 'tokens', min(len(x['tokens']) for x in rows), max(len(x['tokens']) for x in rows),
              'qsrc', sorted({len(x['up']['qsrc']) for x in rows}), 'qent', sorted({len(x['up']['qent']) for x in rows}), flush=True)
        print(r['prompt'], '->', r['gold'], flush=True)
        H25.RUN, H25.SEED = RUN, SEED
        H25.draws(rows)                         # every item has >= 4 donors with equal token counts


def pilotgate():
    out = {}
    for task in TASKS:
        rows = D.read(OUT / f'{task}-pilot-materials.json')
        results, meta = engine(rows, stage=f'pilotgate-{task}')
        g = D.behavior(rows, results['base'])
        out[task] = {k: g[k] for k in ('passed', 'forced_accuracy', 'top1_accuracy')}
        out[task]['worst_cell'] = min((v['forced_accuracy'], v['top1_accuracy'], k) for k, v in g['cells'].items())
        save(f'{task}-pilot-gate.json', {**g, **meta})
    print(json.dumps(out, ensure_ascii=False), flush=True)


def pilot():
    for task in TASKS:
        if not D.read(OUT / f'{task}-pilot-gate.json')['passed']:
            print(task, 'pilot behaviour gate failed; skipped', flush=True)
            continue
        rows = D.read(OUT / f'{task}-pilot-materials.json')
        arms = final_arms(rows, f'{task}|pilot')
        results, meta = engine(rows, arms, stage=f'pilot-{task}')
        s = summarise(results, arms, rows)
        save(f'{task}-pilot-summary.json', {'meta': meta, **s})
        print(task, json.dumps(meta), flush=True)
        print(show(s), flush=True)



# ---- frozen after the pilot on the exposed pilot panels (16 families per task) ----
THRESHOLDS = {
    'C1_colours_direct_reading_carries_over': 'colours: agreement of the all-heads rule >= 0.90 and minus no effect >= 0.12 with a lower '
                                              'bound > 0. not_supported if the lower bound <= 0.',
    'C2_colours_same_sixteen_heads': 'colours: H16 >= 0.88; all heads exceed H16 by <= 0.04; H16 minus the mean of the 5 random sets '
                                     '>= 0.12 with a lower bound > 0; no random set exceeds no effect by more than 0.02. '
                                     'not_supported if the lower bound <= 0.',
    'C3_colours_block_23_heads': 'colours, exchange at block 23: H16 minus the two edges of revision 4 >= 0.15 with a lower bound > 0. '
                                 'not_supported if the lower bound <= 0.',
    'Z1_zh_small_but_real': 'zh: H16 minus no effect has a lower bound > 0 and is <= 0.10; the all-heads rule exceeds no effect by '
                            '<= 0.08. not_supported if H16 minus no effect > 0.15 (the rule carries over) or its lower bound <= 0 '
                            '(nothing carries over).',
    'Z2_zh_far_below_colours': 'the gain of H16 over no effect in zh is at most half of the gain in colours. not_supported otherwise.',
}
CRITERIA = {
    'model': 'Qwen3-14B original BF16, CPU byte reads, CUDA layer streaming. GLM-4-9B is not run.',
    'material': 'two new tasks with the abstract structure of the source-binding task (two sources x two items, four records, the '
                'question names one source and one item; 16 layouts per family; 4 wordings). colours: English, sources are first '
                'names, items common nouns, answers colour words. zh: Chinese prompt, sources are team names, items personal names, '
                'answers one-character animals. Holdout: 48 families per task (768 prompts); sources and items come from word pools '
                'disjoint from the pilot pools, answer words are shared with the pilot.',
    'broad_edit': 'as in Phases 423 / 424: at the input of block L (23, 24, 25) the residual states of the source name in the '
                  'question (qsrc), the item name (qent) or both are replaced by those of a donor of the same family. 4 seeded donor '
                  'relations per item; 9 cells x 3,072 edits per task.',
    'narrow_rules': 'frozen in Phases 423 / 424, nothing selected on the new tasks. all: the answer position reads donor keys / values '
                    'at the site tokens in blocks L..27 through every head. H16: only through the 16 heads of SB.H.read16. H12, H9, '
                    'two_edges, R0-R4 (random heads, the Phase 424 draws). late: every head of blocks 28-39 instead. allH: as all, '
                    'also at the holder positions (descriptive).',
    'prediction': 'the broad edit gives the answer of the narrow edit. Comparisons: no effect, the position rule, the revision-3 '
                  'rule of the original task.',
    'outcome': 'agreement = the predicted record equals the forced choice of the broad edit; per cell, per block, overall; '
               'family-bootstrap 95 % intervals (1000 draws).',
    'controls': 'tail recomputation equals the base forward (same choice, score difference <= 1.0); the narrow edit with the prompt '
                'itself as donor changes nothing (same bound).',
    'two_pass': 'gate stage, per task: unmodified forward and the v1 behaviour gate (every layout and wording cell: forced >= 0.95, '
                'top-1 >= 0.90). Edits run only for a task whose gate passed.',
    'predictions_on_holdout': THRESHOLDS,
    'not_covered': 'why the rule fails to carry over to the Chinese task (the pilot rules out blocks 28-39 and the holder positions '
                   'as simple substitutes); tasks with another structure (one source, more items, free-form answers); other models.',
    'disclosure': 'one pilot per task on 16 exposed families (pilot word pools), run twice (the second time with the descriptive arm '
                  'allH). Pilot: colours all 0.935, H16 0.925, random = no effect 0.734; zh all 0.809, H16 0.832, no effect 0.781. '
                  'The predictions for zh were written after seeing that the rule largely fails there. The pilot behaviour gate '
                  'passed for both tasks (1.000).',
}


def verdicts(S5, metas):
    tri = lambda ok, bad: 'supported' if ok else 'not_supported' if bad else 'inconclusive'
    v = {}
    c, z = S5.get('colours'), S5.get('zh')
    for task, keys in (('colours', ['C1_colours_direct_reading_carries_over', 'C2_colours_same_sixteen_heads', 'C3_colours_block_23_heads']),
                       ('zh', ['Z1_zh_small_but_real', 'Z2_zh_far_below_colours'])):
        v[f'{task}_controls_valid'] = bool(metas.get(task, {}).get('controls_ok'))
        if task not in S5:
            v.update({k: 'behavior_ineligible' for k in keys})
        elif not v[f'{task}_controls_valid']:
            v.update({k: 'controls_invalid' for k in keys})
    if c and v['colours_controls_valid']:
        A = c['scopes']['all']
        d = A['all_minus_none']
        v['C1_colours_direct_reading_carries_over'] = tri(A['agreement']['all']['rate'] >= .90 and d['rate'] >= .12 and d['ci95'][0] > 0, d['ci95'][0] <= 0)
        r = A['H16_minus_random_mean']
        v['C2_colours_same_sixteen_heads'] = tri(A['agreement']['H16']['rate'] >= .88 and A['all_minus_H16']['rate'] <= .04 and r['rate'] >= .12
                                                 and r['ci95'][0] > 0 and c['random_max_overall'] - A['agreement']['none']['rate'] <= .02, r['ci95'][0] <= 0)
        e = c['scopes']['block_23']['H16_minus_two_edges']
        v['C3_colours_block_23_heads'] = tri(e['rate'] >= .15 and e['ci95'][0] > 0, e['ci95'][0] <= 0)
    if z and v['zh_controls_valid']:
        A = z['scopes']['all']
        h = A['H16_minus_none']
        v['Z1_zh_small_but_real'] = tri(h['ci95'][0] > 0 and h['rate'] <= .10 and A['all_minus_none']['rate'] <= .08, h['rate'] > .15 or h['ci95'][0] <= 0)
        if c and v['colours_controls_valid']:
            v['Z2_zh_far_below_colours'] = tri(h['rate'] <= .5 * c['scopes']['all']['H16_minus_none']['rate'], True)
        else:
            v['Z2_zh_far_below_colours'] = 'behavior_ineligible'
    return v


DEPS = (RL, F3, H25, S, RH, S.AR, S.F4)


def prereg():
    from tools import provenance
    from tools.power_analysis import check_phase
    assert not (OUT / 'prereg.json').exists()
    plan = D.checked(MODEL)
    mats, designs = {}, {}
    for task in TASKS:
        rows = build(task, 'holdout')
        pilot_words = set(POOLS[task]['sources']['pilot'] + POOLS[task]['items']['pilot'])
        assert len(rows) == FAMILIES['holdout'] * 16 and not any(w in pilot_words for r in rows for w in r['sources'] + r['names'])
        arms = final_arms(rows, f'{task}|holdout')
        save(f'{task}-holdout-materials.json', rows)
        save(f'{task}-design.json', arms)
        mats[task] = {'sha256': D.sha(OUT / f'{task}-holdout-materials.json'), 'families': FAMILIES['holdout'], 'rows': len(rows)}
        designs[task] = {'sha256': D.sha(OUT / f'{task}-design.json'), 'arms': len(arms)}
    n = FAMILIES['holdout'] * 16
    body = {'schema': 'source_binding_transfer_prereg.v1', 'run_id': RUN, 'seed': SEED, 'created_at': datetime.now().astimezone().isoformat(),
            'base_plan_sha256': D.sha(SRC / 'plan.json'), 'model_snapshot': plan['models'][MODEL], 'code_sha256': D.sha(__file__),
            'dependency_sha256': {m.__name__: D.sha(m.__file__) for m in DEPS},
            'criteria': CRITERIA, 'frozen': {'sets': RH.head_sets(), 'pilot_sha256': {t: D.sha(OUT / f'{t}-pilot-summary.json') for t in TASKS}},
            'materials': mats, 'design': designs,
            'power': {'unit': 'paired binary outcome per item (prediction right / wrong under H16 and under no effect); alpha = 0.05',
                      'colours_as_if_one_edit_per_item': {'expected_delta': 0.15, 'p_d': 0.20, **check_phase(0.15, n, p_d=0.20)},
                      'zh_as_if_one_edit_per_item': {'expected_delta': 0.04, 'p_d': 0.06, **check_phase(0.04, n, p_d=0.06)},
                      'caveat': '27,648 edits per task clustered in 768 items and 48 families; intervals are family bootstraps'},
            'custody': 'each holdout panel is read first by the gate stage (behaviour only) and then once by the run stage.',
            'resources': {'gpu_minutes_estimate': 22, 'disk_GiB_estimate': 0.1, 'models_concurrent': 1,
                          'shape': 'chunks of 192 prompts in one batch; pilot peak 6.4 GiB (16 families: 3.6 minutes per task)'},
            'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    body['provenance']['model_revision'] = plan['models'][MODEL]
    save('prereg.json', body)
    pilot = {t: D.read(OUT / f'{t}-pilot-summary.json') for t in TASKS}
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
    assert not (OUT / 'gate.json').exists(), 'holdout behaviour already read'
    out = {}
    for task in TASKS:
        rows = D.read(OUT / f'{task}-holdout-materials.json')
        results, meta = engine(rows, stage=f'gate-{task}')
        save(f'{task}-gate-items.json', results['base'])
        g = D.behavior(rows, results['base'])
        out[task] = {'items_sha256': D.sha(OUT / f'{task}-gate-items.json'), **g, **meta}
    save('gate.json', {'schema': 'source_binding_transfer_gate.v1', 'run_id': RUN, 'tasks': out})
    print(json.dumps({t: {k: g[k] for k in ('passed', 'forced_accuracy', 'top1_accuracy')} for t, g in out.items()}), flush=True)


def run():
    from tools import provenance
    pre = checked()
    assert not (OUT / 'report.json').exists(), 'holdout already used'
    g = D.read(OUT / 'gate.json')['tasks']
    S5, metas = {}, {}
    for task in TASKS:
        if not g[task]['passed']:
            continue
        rows, arms = D.read(OUT / f'{task}-holdout-materials.json'), D.read(OUT / f'{task}-design.json')
        results, meta = engine(rows, arms, stage=f'run-{task}')
        save(f'{task}-items.json', {k: [{'id': x['id'], 'pred_record': x['pred_record']} for x in v] for k, v in results.items()})
        base = D.read(OUT / f'{task}-gate-items.json')
        meta['base_replay_max_score_difference'] = max(abs(a - b) for x, y in zip(results['base'], base) for a, b in zip(x['scores'], y['scores']))
        S5[task], metas[task] = summarise(results, arms, rows), meta
        print(task, json.dumps(meta), flush=True)
        print(show(S5[task]), flush=True)
    v = verdicts(S5, metas)
    report = {'schema': 'source_binding_transfer_report.v1', 'run_id': RUN, 'gate_passed': {t: g[t]['passed'] for t in TASKS},
              'prereg_sha256': D.sha(OUT / 'prereg.json'), 'gate_sha256': D.sha(OUT / 'gate.json'),
              'items_sha256': {t: D.sha(OUT / f'{t}-items.json') for t in S5}, 'criteria': pre['criteria'], 'frozen': pre['frozen'],
              'verdicts': v, 'summary': S5, 'meta': metas, 'formal_evidence_promotion': 0,
              'custody': 'each holdout panel read once by the gate stage and once by this run',
              'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    save('report.json', report)
    path = provenance.write_result('SOURCE-BINDING-transfer-holdout', {'verdicts': v, 'gate_passed': all(g[t]['passed'] for t in TASKS),
                                                                     'gate_passed_by_task': {t: g[t]['passed'] for t in TASKS},
                                                                     'report_sha256': D.sha(OUT / 'report.json')},
                                   model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)
    print(json.dumps({'verdicts': v, 'result': str(path)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['materials', 'pilotgate', 'pilot', 'prereg', 'gate', 'run'])
    {'materials': materials, 'pilotgate': pilotgate, 'pilot': pilot, 'prereg': prereg, 'gate': gate, 'run': run}[ap.parse_args().stage]()
