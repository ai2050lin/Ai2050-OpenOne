"""Relay positions: from which earlier positions does the final position read the answer?  Qwen3-14B BF16, source-binding task.

Follows Claude Phase 396: blocking attention from the final position to the record tokens never hurts, while blocking
every earlier position from block 24 on destroys the answer.  So the answer is read from other positions.  Here attention
from the final position is blocked by position group, from block 24 on, everything downstream recomputed:
  only-G   every earlier position outside G is blocked (is G sufficient?)
  block-G  the positions of G are blocked (is G necessary?)
Stages: prereg -> pilot_a (discovery, single groups) -> pilot_b (discovery, cumulative sets) -> freeze
        -> gate (new panel, behaviour only) -> run (new panel, once, only if the gate passed).
Uses the frozen runtime, material builder and behaviour gate of SOURCE-BINDING-20261110-v1 without modifying them.
"""
import argparse, gc, hashlib, json, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_binding_design as D
from source_binding_runtime import Runtime

ROOT, SRC = D.ROOT, D.OUT
ABL = ROOT / 'tests/glm5/result/source_binding_ablation_v1'
OUT = ROOT / 'tests/glm5/result/source_binding_relay_v1'
RUN, SEED, MODEL, SPLIT = 'SOURCE-BINDING-RELAY-20261110-v1', 202611100397, 'qwen14', 'holdout_relay'
LAYER, EXTRA, FAMILIES, CHUNK, NB = 24, [8, 16], 96, 256, 40
INSTR = 'Return only the code word, with no explanation.'
TAIL = ['<|im_end|>', '\n', '<|im_start|>', 'assistant', '\n', '<think>', '\n\n', '</think>', '\n\n', 'Answer']
GROUPS = ['pre_other', 'records', 'qrest', 'qsrc', 'qent', 'instr'] + [f'tail{i}' for i in range(len(TAIL))]
CRITERIA = {
    'design_note': 'a smoke test on 16 exposed discovery items showed redundancy (several single groups are each sufficient, no single '
                   'group is necessary). Before pre-registration the design was changed accordingly: the sufficient set and the '
                   'necessary set are selected separately, and the record tokens form their own group.',
    'model': 'Qwen3-14B original BF16, CPU byte reads, CUDA layer streaming. GLM-4-9B is not run (failed the behaviour gate of this task).',
    'groups': {'records': 'the tokens of the four records', 'pre_other': 'everything else before the question: template head, introduction, archive headers',
               'qsrc': 'the archive name in the question', 'qent': 'the entity name in the question',
               'qrest': 'the other tokens of the question sentence', 'instr': 'the instruction sentence "' + INSTR + '"',
               'tail0..tail9': 'the ten template tokens after the user turn, one group each: ' + ' '.join(repr(t) for t in TAIL),
               'note': 'the groups are disjoint and cover every position before the final token'},
    'edit': 'attention mask of the final position only, all heads, from block 24 on; all earlier computation is untouched; '
            'downstream recomputed by the original blocks',
    'outcome': '4-way forced choice among the four code words of the item (chance 0.25); all items analysed; '
               'family-bootstrap 95 % intervals (2000 resamples)',
    'selection_on_discovery': 'pilot_a (families 32-63): only-g and block-g for every single group. Groups are ordered by only-g '
                              'accuracy (descending; ties by block-g ascending). pilot_b (families 0-31): only-S_k and block-S_k for '
                              'the cumulative sets S_k of the first k groups, k = 1..8. The sufficient set S is the smallest S_k with '
                              'only-S_k >= 0.95 (if none, S_8, flagged). The necessary set N is the smallest S_k with block-S_k <= 0.50 '
                              '(if none, S_8, flagged).',
    'material': 'holdout_relay: 96 new families x 8 items; new entity names that occur neither in SOURCE-BINDING-20261110-v1 nor in '
                'holdout_abl; the 16 discovery code words and the 4 seen frames',
    'two_pass': 'gate stage: unmodified forward and the v1 behaviour gate (every S x E x O cell and frame: n >= 16, forced >= 0.95, '
                'top-1 >= 0.90). Edits run only if the gate passed; otherwise every verdict is behavior_ineligible.',
    'predictions_on_holdout': {
        'controls_valid': 'noop reproduces the base logits bit for bit; blocking every earlier position gives <= 0.40',
        'R1_sufficient': {'supported': 'only-S >= 0.90', 'not_supported': 'only-S < 0.70'},
        'R2_necessary': {'supported': 'block-N <= 0.50', 'not_supported': 'block-N >= 0.80'},
        'R3_redundant_routes': {'supported': 'for every group g of N (if N has more than one group), block-(N minus g), i.e. leaving g '
                                             'readable, is at least 0.10 above block-N with family-bootstrap lower bound > 0',
                                'not_supported': 'for some g the gain is below 0.02', 'not_applicable': 'N has one group'}},
    'descriptive': 'only-g and block-g for every single group; only-S and block-N from blocks 8 and 16 on; top-1, margins, KL',
    'not_covered': 'which heads move the answer; how the relay positions obtain it; other models, unseen answer words, unseen frames',
    'disclosure': 'the code was smoke-tested on discovery items (exposed) before pre-registration; thresholds were written first',
}


def stream(purpose, key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(f'{RUN}|{SEED}|{purpose}|{key}'.encode()).digest()[:16], 'little'))


def save(name, body):
    from deploy.storage_policy import check_write
    p = OUT / name
    check_write(p, 1 << 20, operation='来源绑定接力位置产物')
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(body, ensure_ascii=False, indent=1, default=lambda x: x.item() if isinstance(x, np.generic) else x), encoding='utf-8')
    t.replace(p)


def log(stage, done, total, start, **kw):
    from source_binding_resources import resources
    sec = time.monotonic() - start
    row = {'time': datetime.now().astimezone().isoformat(), 'stage': stage, 'done': done, 'total': total, 'seconds': round(sec, 1),
           'eta_seconds': round(sec / done * (total - done), 1) if done else None, **resources(), **kw}
    if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
        t = sys.modules['torch']
        row.update(cuda_allocated_MiB=round(t.cuda.memory_allocated() / 2 ** 20), cuda_peak_MiB=round(t.cuda.max_memory_allocated() / 2 ** 20))
    print(json.dumps(row, ensure_ascii=False), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'progress.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
    assert row['ram_available_bytes'] > 2 * 2 ** 30 and row['commit_available_bytes'] > 4 * 2 ** 30, 'RAM / commit resource stop'
    if row.get('shared_gpu_bytes') is not None:
        assert row['shared_gpu_bytes'] < 2 ** 30, 'GPU shared residency above 1 GiB'


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(D.MODELS[MODEL], local_files_only=True)


def group_positions(tok, r):
    """Token indices (unpadded) of every group; disjoint cover of all positions before the final token."""
    prompt, n = r['prompt'], len(r['tokens'])
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    assert enc['input_ids'] == r['tokens']
    off = prompt.index(r['content'])
    qa, qb = r['query_span']
    assert r['content'][qb - len(INSTR):qb] == INSTR and qb == len(r['content'])
    inside = lambda lo, hi, span: hi > off + span[0] and lo < off + span[1]
    g = {k: [] for k in GROUPS}
    tail = [i for i, (lo, hi) in enumerate(enc['offset_mapping']) if lo >= off + qb][:-1]
    assert [tok.decode([r['tokens'][i]]) for i in tail] == TAIL and tail[-1] == n - 2
    rec = {i for k in range(4) for i in r['records'][str(k)]['span']}
    for j, i in enumerate(tail):
        g[f'tail{j}'] = [i]
    for i, (lo, hi) in enumerate(enc['offset_mapping'][:tail[0]]):
        if lo < off + qa:
            g['records' if i in rec else 'pre_other'].append(i)
        elif inside(lo, hi, r['query_source_span']):
            g['qsrc'].append(i)
        elif inside(lo, hi, r['query_name_span']):
            g['qent'].append(i)
        elif lo >= off + qb - len(INSTR):
            g['instr'].append(i)
        else:
            g['qrest'].append(i)
    flat = sorted(i for v in g.values() for i in v)
    assert flat == list(range(n - 1)) and all(g.values())
    return g


def old_rows(split):
    return D.read(SRC / f'{MODEL}-materials.json')['splits'][split]


def build_materials(tok):
    used = {n for rows in D.read(SRC / f'{MODEL}-materials.json')['splits'].values() for r in rows for n in r['names']}
    used |= {n for r in D.read(ABL / 'holdout_abl-materials.json') for n in r['names']}
    rows = []
    for family in range(FAMILIES):
        rnd, names = stream('names', family), []
        while len(names) < 2:
            name = ''.join(rnd.choice(list('bdfgklmnprstvz')) + rnd.choice(list('aeiou')) for _ in range(3)).capitalize()
            if name not in used:
                used.add(name)
                names.append(name)
        values = rnd.choice(D.WORDS[:16], 4, replace=False).tolist()
        for s in (0, 1):
            for e in (0, 1):
                for o in (0, 1):
                    r = D.material(family, SPLIT, s, e, o, family % 4, names, values)
                    prompt = tok.apply_chat_template([{'role': 'user', 'content': r['content']}], tokenize=False,
                                                     add_generation_prompt=True, enable_thinking=False) + 'Answer:\n'
                    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
                    ids, off = enc['input_ids'], prompt.index(r['content'])
                    recs = {str(k): {key: [i for i, (lo, hi) in enumerate(enc['offset_mapping']) if hi > off + a and lo < off + b]
                                     for key, (a, b) in info.items()} for k, info in r['records'].items()}
                    choice = []
                    for val in r['values']:
                        suffix = tok.encode(prompt + val, add_special_tokens=False)
                        assert suffix[:-1] == ids, 'answer word must be one token with a stable prefix'
                        choice.append(suffix[-1])
                    r = {**r, 'records': recs, 'headers': {str(k): v for k, v in r['headers'].items()},
                         'prompt': prompt, 'tokens': ids, 'choice_ids': choice, 'gold_id': choice[r['target_record']]}
                    r['groups'] = group_positions(tok, r)
                    rows.append(r)
    return rows


def with_groups(rows):
    tok = tokenizer()
    return [{**r, 'groups': group_positions(tok, r)} for r in rows]


def prereg():
    from deploy.storage_policy import checked_dir
    from tools import provenance
    from tools.power_analysis import check_phase
    checked_dir(OUT, 1 << 30, operation='来源绑定接力位置产物')
    assert not (OUT / 'prereg.json').exists()
    plan = D.checked(MODEL)
    rows = build_materials(tokenizer())
    assert len(rows) == FAMILIES * 8 and len({tuple(r['tokens']) for r in rows}) == len(rows)
    save(f'{SPLIT}-materials.json', rows)
    n = len(rows)
    body = {'schema': 'source_binding_relay_prereg.v1', 'run_id': RUN, 'seed': SEED, 'created_at': datetime.now().astimezone().isoformat(),
            'base_plan_sha256': D.sha(SRC / 'plan.json'), 'model_snapshot': plan['models'][MODEL], 'code_sha256': D.sha(__file__),
            'criteria': CRITERIA, 'layer': LAYER, 'extra_layers': EXTRA,
            'materials': {'split': SPLIT, 'sha256': D.sha(OUT / f'{SPLIT}-materials.json'), 'families': FAMILIES, 'rows': n},
            'power': {'unit': 'paired binary outcome per item (McNemar); alpha = 0.05',
                      'R1_R2': {'expected_delta': 0.40, 'p_d': 0.50, **check_phase(0.40, n, p_d=0.50)},
                      'R3': {'expected_delta': 0.10, 'p_d': 0.20, **check_phase(0.10, n, p_d=0.20)},
                      'caveat': 'items are clustered in 96 families and 4 frames; R3 is stated on family-bootstrap lower bounds'},
            'custody': 'the relay set comes from discovery (exposed). holdout_relay is generated here, read first by the gate stage '
                       '(behaviour only) and then once by the run stage.',
            'resources': {'gpu_minutes_estimate': 30, 'disk_GiB_estimate': 0.2, 'models_concurrent': 1},
            'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    body['provenance']['model_revision'] = plan['models'][MODEL]
    save('prereg.json', body)
    print(json.dumps({'prereg_sha256': D.sha(OUT / 'prereg.json'), 'power': body['power']}, ensure_ascii=False)[:600], flush=True)


def checked():
    p = D.read(OUT / 'prereg.json')
    assert p['code_sha256'] == D.sha(__file__), 'source changed after pre-registration'
    assert p['materials']['sha256'] == D.sha(OUT / f'{SPLIT}-materials.json')
    D.checked(MODEL)
    return p


def engine(rows, arms=(), stage='run'):
    """arms: list of (name, layer, kind, groups); kind in noop / only / block / all."""
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    t = torch
    assert snapshot(D.MODELS[MODEL]) == D.read(SRC / f'{MODEL}-identity.json')['model_snapshot']
    results = {a[0]: [] for a in arms}
    results['base'] = []
    start, noop_equal = time.monotonic(), []
    with GpuLease(), t.inference_mode():
        rt = Runtime(MODEL)
        total = len(rows) * NB
        log(stage, 0, total, start)
        for chunk in range(0, len(rows), CHUNK):
            rs = rows[chunk:chunk + CHUNK]
            ids, pos, pe, mask = rt.context(rs)
            h, n, length = rt.embed(ids), len(rs), ids.shape[1]
            masks = {}
            for name, _, kind, groups in arms:
                m = mask.clone()
                for i, r in enumerate(rs):
                    off = length - len(r['tokens'])
                    sel = sorted(off + j for g in groups for j in r['groups'][g])
                    if kind == 'block':
                        m[i, 0, -1, sel] = False
                    elif kind in ('only', 'all'):
                        keep = m[i, 0, -1].clone()
                        m[i, 0, -1, :-1] = False
                        if kind == 'only':
                            m[i, 0, -1, sel] = keep[sel]
                masks[name] = m
            state = {}
            for b in range(NB):
                for name, l, _, _ in arms:
                    if l == b:
                        state[name] = h[:, -1].clone()
                layer = rt.block(b)
                for lo in range(0, n, 8):
                    hi = min(lo + 8, n)
                    kw = {'position_ids': pos[lo:hi], 'position_embeddings': tuple(p[lo:hi] for p in pe), 'use_cache': False}
                    y = layer(h[lo:hi], attention_mask=mask[lo:hi], **kw)
                    for key, s in state.items():
                        x = h[lo:hi].clone()
                        x[:, -1] = s[lo:hi]
                        s[lo:hi] = layer(x, attention_mask=masks[key][lo:hi], **kw)[:, -1]
                    h[lo:hi] = y
                del layer
                gc.collect()
                t.cuda.empty_cache()
                if (b + 1) % 5 == 0:
                    log(stage, chunk * NB + (b + 1) * n, total, start, block=b, arms=len(state))
            norm = rt.Norm(rt.dim, eps=rt.cfg.rms_norm_eps).to(device='cuda', dtype=t.bfloat16)
            norm.weight = t.nn.Parameter(rt.weight('model.norm.weight'), requires_grad=False)
            W = rt.weight('lm_head.weight' if 'lm_head.weight' in rt.index else 'model.embed_tokens.weight')
            lg0 = (norm(h[:, -1]) @ W.T).float()
            lp0 = lg0.log_softmax(-1)
            results['base'] += rt.score(lg0, rs)
            for key, s in state.items():
                lg = (norm(s) @ W.T).float()
                if key.startswith('noop'):
                    noop_equal.append(bool(t.equal(lg, lg0)))
                kl = (lp0.exp() * (lp0 - lg.log_softmax(-1))).sum(-1).cpu().tolist()
                scored = rt.score(lg, rs)
                for x, v in zip(scored, kl):
                    x['kl'] = v
                results[key] += scored
            del W, norm, h, state, lg0, lp0, ids, pos, pe, mask, masks
            gc.collect()
            t.cuda.empty_cache()
        peak = round(t.cuda.max_memory_allocated() / 2 ** 20)
    return results, {'noop_bitwise_equal': noop_equal, 'peak_MiB': peak, 'seconds': round(time.monotonic() - start, 1)}


def summarise(results, rows, pairs=()):
    fam = np.array([r['family'] for r in rows])
    members = [np.flatnonzero(fam == f) for f in sorted(set(fam))]
    draws = np.random.default_rng(SEED).integers(0, len(members), (2000, len(members)))
    cnt = np.array([len(m) for m in members])

    def ci(v):
        v = np.asarray(v, float)
        per = np.array([v[m].sum() for m in members])
        bs = per[draws].sum(1) / cnt[draws].sum(1)
        return {'rate': round(float(v.mean()), 4), 'ci95': [round(float(np.quantile(bs, .025)), 4), round(float(np.quantile(bs, .975)), 4)]}
    col = lambda k, f: np.array([x[f] for x in results[k]], float)
    base = col('base', 'gold_margin')
    out = {'n': len(rows), 'arms': {}, 'differences': {}}
    for k in results:
        out['arms'][k] = {'forced': ci(col(k, 'forced_correct')), 'top1': ci(col(k, 'top1_correct')),
                          'gold_margin_change': round(float((col(k, 'gold_margin') - base).mean()), 3),
                          'mean_kl': round(float(col(k, 'kl').mean()), 4) if k != 'base' else 0.0}
    for a, b in pairs:
        out['differences'][f'{a}-{b}'] = ci(col(a, 'forced_correct') - col(b, 'forced_correct'))
    return out


def discovery_half(second):
    rows = old_rows('discovery')
    return with_groups([r for r in rows if (int(r['family'][-4:]) >= 32) == second])


def base_arms(l=LAYER):
    return [(f'noop@{l}', l, 'noop', []), (f'all@{l}', l, 'all', [])]


def pilot_a(smoke=False):
    rows = discovery_half(True)
    if smoke:
        rows = rows[:16]
    else:
        checked()
        assert not (OUT / 'pilot-a.json').exists()
    arms = base_arms() + [(f'{k}:{g}', LAYER, k, [g]) for g in GROUPS for k in ('only', 'block')]
    results, meta = engine(rows, arms, stage='pilot_a')
    s = summarise(results, rows)
    acc = lambda k: s['arms'][k]['forced']['rate']
    order = sorted(GROUPS, key=lambda g: (-acc(f'only:{g}'), acc(f'block:{g}'), GROUPS.index(g)))
    table = {g: [acc(f'only:{g}'), acc(f'block:{g}')] for g in order}
    if smoke:
        print(json.dumps({'smoke': True, 'meta': meta, 'all': acc(f'all@{LAYER}'), 'only_block': table}), flush=True)
        return order
    assert all(meta['noop_bitwise_equal'])
    save('pilot-a.json', {'schema': 'source_binding_relay_pilot_a.v1', 'run_id': RUN, 'scope': 'discovery families 32-63 (exposed)',
                          'summary': s, 'order': order, 'group_sizes_first_item': {g: len(rows[0]['groups'][g]) for g in GROUPS}, **meta})
    print(json.dumps({'all': acc(f'all@{LAYER}'), 'only_block': table}), flush=True)


def pilot_b(smoke=False):
    rows = discovery_half(False)
    if smoke:
        rows, order = rows[:16], pilot_a(smoke=True)
    else:
        checked()
        assert not (OUT / 'pilot-b.json').exists()
        order = D.read(OUT / 'pilot-a.json')['order']
    arms = base_arms() + [(f'{k}:S{n}', LAYER, k, order[:n]) for n in range(1, 9) for k in ('only', 'block')]
    results, meta = engine(rows, arms, stage='pilot_b')
    s = summarise(results, rows)
    acc = lambda k: s['arms'][k]['forced']['rate']
    k = next((n for n in range(1, 9) if acc(f'only:S{n}') >= .95), None)
    kn = next((n for n in range(1, 9) if acc(f'block:S{n}') <= .50), None)
    table = {f'S{n}': [acc(f'only:S{n}'), acc(f'block:S{n}')] for n in range(1, 9)}
    if smoke:
        print(json.dumps({'smoke': True, 'k': k, 'kn': kn, 'order': order, 'only_block': table}), flush=True)
        return
    assert all(meta['noop_bitwise_equal'])
    save('pilot-b.json', {'schema': 'source_binding_relay_pilot_b.v1', 'run_id': RUN, 'scope': 'discovery families 0-31 (exposed)',
                          'summary': s, 'order': order, 'k': k if k else 8, 'rule_met': k is not None, 'relay_set': order[:k or 8],
                          'necessary_k': kn if kn else 8, 'necessary_rule_met': kn is not None, 'necessary_set': order[:kn or 8], **meta})
    print(json.dumps({'sufficient_set': order[:k or 8], 'necessary_set': order[:kn or 8], 'only_block': table}), flush=True)


def freeze():
    checked()
    assert not (OUT / 'frozen.json').exists()
    b = D.read(OUT / 'pilot-b.json')
    save('frozen.json', {'schema': 'source_binding_relay_frozen.v1', 'run_id': RUN, 'layer': LAYER, 'relay_set': b['relay_set'], 'necessary_set': b['necessary_set'],
                         'rule_met': b['rule_met'], 'necessary_rule_met': b['necessary_rule_met'], 'order': b['order'], 'prereg_sha256': D.sha(OUT / 'prereg.json'),
                         'pilot_a_sha256': D.sha(OUT / 'pilot-a.json'), 'pilot_b_sha256': D.sha(OUT / 'pilot-b.json'),
                         'created_at': datetime.now().astimezone().isoformat()})
    print(json.dumps(D.read(OUT / 'frozen.json'), ensure_ascii=False), flush=True)


def frozen():
    pre, fz = checked(), D.read(OUT / 'frozen.json')
    assert fz['prereg_sha256'] == D.sha(OUT / 'prereg.json') and fz['pilot_b_sha256'] == D.sha(OUT / 'pilot-b.json')
    return pre, fz


def gate():
    frozen()
    assert not (OUT / 'gate.json').exists(), 'holdout behaviour already read'
    rows = D.read(OUT / f'{SPLIT}-materials.json')
    results, meta = engine(rows, stage='gate')
    save('gate-items.json', results['base'])
    g = D.behavior(rows, results['base'])
    save('gate.json', {'schema': 'source_binding_relay_gate.v1', 'run_id': RUN, 'items_sha256': D.sha(OUT / 'gate-items.json'), **g, **meta})
    print(json.dumps({k: g[k] for k in ('passed', 'forced_accuracy', 'top1_accuracy')}), flush=True)


def run():
    from tools import provenance
    pre, fz = frozen()
    assert not (OUT / 'report.json').exists(), 'holdout already used'
    g = D.read(OUT / 'gate.json')
    rows = D.read(OUT / f'{SPLIT}-materials.json')
    S, N = fz['relay_set'], fz['necessary_set']
    names = ('R1_sufficient', 'R2_necessary', 'R3_redundant_routes')
    if g['passed']:
        arms = base_arms() + [('only:S', LAYER, 'only', S), ('block:N', LAYER, 'block', N)]
        arms += [(f'{k}:{x}', LAYER, k, [x]) for x in GROUPS for k in ('only', 'block')]
        if len(N) > 1:
            arms += [(f'block:N-{x}', LAYER, 'block', [y for y in N if y != x]) for x in N]
        arms += [(f'only:S@{l}', l, 'only', S) for l in EXTRA] + [(f'block:N@{l}', l, 'block', N) for l in EXTRA]
        results, meta = engine(rows, arms, stage='run')
        save('items.json', results)
        base = D.read(OUT / 'gate-items.json')
        meta['base_replay_max_score_difference'] = max(abs(a - b) for x, y in zip(results['base'], base) for a, b in zip(x['scores'], y['scores']))
        s = summarise(results, rows, [(f'block:N-{x}', 'block:N') for x in N] if len(N) > 1 else [])
        acc = lambda k: s['arms'][k]['forced']['rate']
        tri = lambda ok, bad: 'supported' if ok else 'not_supported' if bad else 'inconclusive'
        controls = all(meta['noop_bitwise_equal']) and acc(f'all@{LAYER}') <= .40
        v = {'controls_valid': bool(controls)}
        if controls:
            v['R1_sufficient'] = tri(acc('only:S') >= .90, acc('only:S') < .70)
            v['R2_necessary'] = tri(acc('block:N') <= .50, acc('block:N') >= .80)
            ds = list(s['differences'].values())
            v['R3_redundant_routes'] = 'not_applicable' if len(N) == 1 else tri(
                all(d['rate'] >= .10 and d['ci95'][0] > 0 for d in ds), any(d['rate'] < .02 for d in ds))
        else:
            v.update({k: 'controls_invalid' for k in names})
    else:
        s, meta, v = {'note': 'behaviour gate failed; no edit was run'}, {}, {k: 'behavior_ineligible' for k in names}
    report = {'schema': 'source_binding_relay_report.v1', 'run_id': RUN, 'split': SPLIT, 'layer': LAYER, 'sufficient_set': S, 'necessary_set': N, 'gate_passed': g['passed'],
              'prereg_sha256': D.sha(OUT / 'prereg.json'), 'frozen_sha256': D.sha(OUT / 'frozen.json'), 'gate_sha256': D.sha(OUT / 'gate.json'),
              'items_sha256': D.sha(OUT / 'items.json') if g['passed'] else None, 'criteria': pre['criteria'], 'verdicts': v, 'summary': s,
              **meta, 'formal_evidence_promotion': 0, 'custody': 'holdout_relay read once by the gate stage and once by this run',
              'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    save('report.json', report)
    path = provenance.write_result('SOURCE-BINDING-relay-holdout', {'verdicts': v, 'gate_passed': g['passed'], 'sufficient_set': S, 'necessary_set': N,
                                                                    'report_sha256': D.sha(OUT / 'report.json')},
                                   model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)
    print(json.dumps({'verdicts': v, 'sufficient_set': S, 'necessary_set': N, 'result': str(path)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['prereg', 'pilot_a', 'pilot_b', 'freeze', 'gate', 'run'])
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    {'prereg': prereg, 'pilot_a': lambda: pilot_a(a.smoke), 'pilot_b': lambda: pilot_b(a.smoke), 'freeze': freeze, 'gate': gate, 'run': run}[a.stage]()
