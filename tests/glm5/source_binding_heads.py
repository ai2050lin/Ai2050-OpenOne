"""Which attention heads let the relay position read the answer from the correct record?  Qwen3-14B BF16, source binding.

Follows Claude Phase 399: with the final position restricted to read only the relay position R (the line break after the
user turn) from block 24 on, R obtains the answer from the correct record in blocks 28-31 (single route).
Here only one kind of edge is edited: R -> tokens of the correct record, per attention head.
  only:K     in blocks 28-39, every head outside K cannot attend from R to the correct record
  blockw:K   heads of K cannot attend from R to the correct record, and R attends to nothing from block 32 on (window 28-31)
  block:K    heads of K cannot attend from R to the correct record; nothing else is edited
Everything else at R and all later positions are recomputed by the original blocks.
Stages: prereg -> pilot_a (single heads) -> pilot_b (cumulative sets) -> freeze -> gate -> run.
"""
import argparse, gc, hashlib, json, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_binding_design as D
import source_binding_relay as RL
import source_binding_upstream as UP
from source_binding_runtime import Runtime

SRC, MODEL = D.OUT, 'qwen14'
RELAY_OUT, ABL_OUT, UP_OUT = RL.OUT, RL.ABL, UP.OUT
OUT = D.ROOT / 'tests/glm5/result/source_binding_heads_v1'
RUN, SEED, SPLIT = 'SOURCE-BINDING-HEADS-20261111-v1', 202611110400, 'holdout_heads'
FINAL_LAYER, FAMILIES, CHUNK, NB, NH, R_FROM_END = 24, 96, 64, 40, 40, UP.R_FROM_END
WINDOW, LATE, CLOSE = [28, 29, 30, 31], list(range(28, 40)), 32
KS, NCTRL, TOP_SINGLE = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32], 3, 8
CRITERIA = {
    'model': 'Qwen3-14B original BF16, CPU byte reads, CUDA layer streaming. GLM-4-9B is not run (failed the behaviour gate of this task).',
    'readout': 'as in Phase 399: in every arm, from block 24 on the final position may attend only to R (and itself). '
               'The reference arm has this restriction and no edit at R.',
    'edge': 'attention from R (line break after the user turn, 10 tokens before the end) to the tokens of the record of the requested '
            'archive and entity. Candidate heads: 40 heads x blocks 28-31 (the window fixed by Phase 399: L0 = 28, all@32 = 0.999).',
    'arms': {'only:K': 'in blocks 28-39 every head outside K cannot use the edge (sufficiency of K)',
             'blockw:K': 'heads of K cannot use the edge and R attends to nothing but itself from block 32 on (necessity inside the window)',
             'block:K': 'heads of K cannot use the edge, nothing else edited (necessity when later blocks may compensate)',
             'block:rec': 'no head in blocks 28-39 can use the edge (floor; Phase 399 block rec_correct)'},
    'outcome': '4-way forced choice among the four code words (chance 0.25); all items analysed; family-bootstrap 95 % intervals',
    'selection_on_discovery': {
        'order': 'pilot_a (families 32-47, 128 items): the 160 candidate heads ordered by the mean gold margin of only:{h} '
                 '(descending; ties by block, head). blockw:{h} is recorded for description, not used for the order.',
        'sets': 'pilot_b (families 0-31, 256 items): cumulative sets K_k of the first k heads, k in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]. '
                'S = smallest K_k with only:K_k >= 0.80; N = smallest K_k with blockw:K_k <= 0.50 (if none, K_32, flagged)',
        'controls': 'for S and for N, 3 random head sets with the same number of heads per block, drawn from candidate heads outside the set '
                    '(seeded by SHA-256 of run id, seed, purpose, index)'},
    'material': 'holdout_heads: 96 new families x 8 items; entity names that occur in no earlier panel of this line; the 16 discovery code '
                'words, 4 seen frames',
    'two_pass': 'gate stage: unmodified forward and the v1 behaviour gate. Edits run only if the gate passed; otherwise behavior_ineligible.',
    'predictions_on_holdout': {
        'controls_valid': 'noop reproduces the base logits bit for bit; reference >= 0.90; block:rec <= 0.40',
        'H1_sufficient': {'supported': 'only:S >= 0.70, and against each of the 3 controls only:S - only:ctrl >= 0.20 with lower bound > 0',
                          'not_supported': 'only:S < 0.50'},
        'H2_necessary_in_window': {'supported': 'blockw:N <= 0.50, and for each of the 3 controls blockw:ctrl - blockw:N >= 0.20 with lower '
                                                'bound > 0', 'not_supported': 'blockw:N >= 0.80'},
        'H3_no_later_compensation': {'supported': 'block:N <= 0.50', 'not_supported': 'block:N >= 0.80 (other heads or later blocks compensate)'}},
    'descriptive': 'per block b in 28-31: only heads of b / all heads of b blocked; only blocks 32-39; the first 8 heads of the order singly; '
                   'top-1, margins, KL',
    'not_covered': 'what the heads compute (QK / OV), how the query reaches R before block 28, routes that do not pass through R, other '
                   'models, unseen answer words, unseen frames. A head set is a set of edges at one position, not a full circuit.',
    'disclosure': 'the code was smoke-tested on 16 discovery items (exposed) before pre-registration; thresholds were written first',
}


def stream(purpose, key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(f'{RUN}|{SEED}|{purpose}|{key}'.encode()).digest()[:16], 'little'))


def save(name, body):
    from deploy.storage_policy import check_write
    p = OUT / name
    check_write(p, 1 << 20, operation='来源绑定读取头产物')
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(body, ensure_ascii=False, indent=1, default=lambda x: x.item() if isinstance(x, np.generic) else x), encoding='utf-8')
    t.replace(p)


def log(*a, **kw):
    RL.OUT = OUT
    RL.log(*a, **kw)


def build_materials():
    tok = RL.tokenizer()
    used = {n for rows in D.read(SRC / f'{MODEL}-materials.json')['splits'].values() for r in rows for n in r['names']}
    for p in (ABL_OUT / 'holdout_abl-materials.json', RELAY_OUT / 'holdout_relay-materials.json', UP_OUT / 'holdout_up-materials.json'):
        used |= {n for r in D.read(p) for n in r['names']}
    rows = []
    for family in range(FAMILIES):
        rnd = stream('names', family)
        names = []
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
                        assert suffix[:-1] == ids
                        choice.append(suffix[-1])
                    r = {**r, 'records': recs, 'headers': {str(k): v for k, v in r['headers'].items()}, 'prompt': prompt, 'tokens': ids,
                         'choice_ids': choice, 'gold_id': choice[r['target_record']]}
                    rows.append(UP.add_groups(tok, r))
    return rows


def prereg():
    from deploy.storage_policy import checked_dir
    from tools import provenance
    from tools.power_analysis import check_phase
    checked_dir(OUT, 1 << 30, operation='来源绑定读取头产物')
    assert not (OUT / 'prereg.json').exists()
    plan = D.checked(MODEL)
    rows = build_materials()
    assert len(rows) == FAMILIES * 8 and len({tuple(r['tokens']) for r in rows}) == len(rows)
    save(f'{SPLIT}-materials.json', rows)
    n = len(rows)
    body = {'schema': 'source_binding_heads_prereg.v1', 'run_id': RUN, 'seed': SEED, 'created_at': datetime.now().astimezone().isoformat(),
            'base_plan_sha256': D.sha(SRC / 'plan.json'), 'upstream_report_sha256': D.sha(UP_OUT / 'report.json'),
            'model_snapshot': plan['models'][MODEL], 'code_sha256': D.sha(__file__), 'relay_code_sha256': D.sha(RL.__file__),
            'upstream_code_sha256': D.sha(UP.__file__),
            'criteria': CRITERIA, 'materials': {'split': SPLIT, 'sha256': D.sha(OUT / f'{SPLIT}-materials.json'), 'families': FAMILIES, 'rows': n},
            'power': {'unit': 'paired binary outcome per item (McNemar); alpha = 0.05',
                      'H1_H2_H3_level': {'expected_delta': 0.40, 'p_d': 0.50, **check_phase(0.40, n, p_d=0.50)},
                      'H1_H2_vs_control': {'expected_delta': 0.20, 'p_d': 0.40, **check_phase(0.20, n, p_d=0.40)},
                      'caveat': 'items are clustered in 96 families and 4 frames; control contrasts are stated on family-bootstrap lower bounds'},
            'custody': 'the head order and the sets come from discovery (exposed). holdout_heads is generated here, read first by the gate '
                       'stage (behaviour only) and then once by the run stage.',
            'resources': {'gpu_minutes_estimate': 90, 'disk_GiB_estimate': 0.2, 'models_concurrent': 1},
            'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    body['provenance']['model_revision'] = plan['models'][MODEL]
    save('prereg.json', body)
    print(json.dumps({'prereg_sha256': D.sha(OUT / 'prereg.json'), 'power': body['power']}), flush=True)


def checked():
    p = D.read(OUT / 'prereg.json')
    assert p['code_sha256'] == D.sha(__file__) and p['relay_code_sha256'] == D.sha(RL.__file__) and p['upstream_code_sha256'] == D.sha(UP.__file__), \
        'source changed after pre-registration'
    assert p['materials']['sha256'] == D.sha(OUT / f'{SPLIT}-materials.json')
    D.checked(MODEL)
    return p


def only(K):
    """Edits for only:K - every head outside K loses the edge in blocks 28-39."""
    keep = {b: set() for b in LATE}
    for b, h in K:
        keep[b].add(h)
    return {b: [h for h in range(NH) if h not in keep[b]] for b in LATE}


def block(K):
    out = {}
    for b, h in K:
        out.setdefault(b, []).append(h)
    return out


def engine(rows, arms=(), stage='run'):
    """arms: (name, edits {block: [heads]}, close) - heads listed lose the edge R -> correct record in that block;
    close = block from which R attends to nothing but itself (or None).  Per arm only the state of positions >= R is kept."""
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    t = torch
    assert snapshot(D.MODELS[MODEL]) == D.read(SRC / f'{MODEL}-identity.json')['model_snapshot']
    arms = [(name, {int(b): list(h) for b, h in edits.items() if h}, close) for name, edits, close in arms]
    fork = {name: min(list(edits) + ([close] if close is not None else [])) for name, edits, close in arms}
    assert all(f > FINAL_LAYER for f in fork.values())
    results = {k: [] for k in ['base', 'noop', 'ref'] + [a[0] for a in arms]}
    if not arms:
        results = {'base': []}
    start, noop_equal = time.monotonic(), []
    with GpuLease(), t.inference_mode():
        rt = Runtime(MODEL)
        assert rt.heads == NH and rt.cfg.num_hidden_layers == NB
        total = len(rows) * NB
        log(stage, 0, total, start)
        for chunk in range(0, len(rows), CHUNK):
            rs = rows[chunk:chunk + CHUNK]
            ids, pos, pe, mask = rt.context(rs)
            h, n, length = rt.embed(ids), len(rs), ids.shape[1]
            R = length - R_FROM_END
            frow = t.zeros((n, length), dtype=t.bool, device='cuda')
            frow[:, R] = True
            frow[:, -1] = True
            selfrow = t.zeros((length,), dtype=t.bool, device='cuda')
            selfrow[R] = True
            rec = t.zeros((n, length), dtype=t.bool, device='cuda')
            for i, r in enumerate(rs):
                assert len(r['tokens']) - R_FROM_END == r['up']['tail0'][-1] + 1
                rec[i, [length - len(r['tokens']) + j for j in r['up']['rec_correct']]] = True
            state = {}
            for b in range(NB):
                if arms and b == FINAL_LAYER:
                    state['noop'], state['ref'] = h[:, R:].clone(), h[:, R:].clone()
                for name in fork:
                    if fork[name] == b:
                        state[name] = state['ref'].clone()
                layer = rt.block(b)
                for lo in range(0, n, 8):
                    hi = min(lo + 8, n)
                    kw = {'position_ids': pos[lo:hi], 'position_embeddings': tuple(p[lo:hi] for p in pe), 'use_cache': False}
                    y = layer(h[lo:hi], attention_mask=mask[lo:hi], **kw)
                    if state:
                        step = lambda key, m: layer(t.cat([h[lo:hi, :R], state[key][lo:hi]], 1), attention_mask=m, **kw)[:, R:]
                        new = {'noop': step('noop', mask[lo:hi])}
                        mref = mask[lo:hi].clone()
                        mref[:, 0, -1] = frow[lo:hi]
                        new['ref'] = step('ref', mref)
                        for name, edits, close in arms:
                            if name not in state:
                                continue
                            closed = close is not None and b >= close
                            if b not in edits and not closed:
                                new[name] = step(name, mref)
                                continue
                            m = mref.expand(-1, NH, -1, -1).clone()
                            if closed:
                                m[:, :, R] = selfrow
                            elif b in edits:
                                m[:, edits[b], R] = m[:, edits[b], R] & ~rec[lo:hi, None]
                            new[name] = step(name, m)
                        for key, v in new.items():
                            state[key][lo:hi] = v
                    h[lo:hi] = y
                del layer
                gc.collect()
                t.cuda.empty_cache()
                if (b + 1) % 4 == 0:
                    log(stage, chunk * NB + (b + 1) * n, total, start, block=b, arms=len(state))
            norm = rt.Norm(rt.dim, eps=rt.cfg.rms_norm_eps).to(device='cuda', dtype=t.bfloat16)
            norm.weight = t.nn.Parameter(rt.weight('model.norm.weight'), requires_grad=False)
            W = rt.weight('lm_head.weight' if 'lm_head.weight' in rt.index else 'model.embed_tokens.weight')
            lg0 = (norm(h[:, -1]) @ W.T).float()
            lp0 = lg0.log_softmax(-1)
            results['base'] += rt.score(lg0, rs)
            for key, s in state.items():
                lg = (norm(s[:, -1]) @ W.T).float()
                if key == 'noop':
                    noop_equal.append(bool(t.equal(lg, lg0)))
                kl = (lp0.exp() * (lp0 - lg.log_softmax(-1))).sum(-1).cpu().tolist()
                scored = rt.score(lg, rs)
                for x, v in zip(scored, kl):
                    x['kl'] = v
                results[key] += scored
            del W, norm, h, state, lg0, lp0, ids, pos, pe, mask, rec, frow
            gc.collect()
            t.cuda.empty_cache()
        peak = round(t.cuda.max_memory_allocated() / 2 ** 20)
    return results, {'noop_bitwise_equal': noop_equal, 'peak_MiB': peak, 'seconds': round(time.monotonic() - start, 1)}


def finish(name, rows, arms, smoke, extra):
    results, meta = engine(rows, arms, stage=name)
    s = RL.summarise(results, rows)
    acc = lambda k: s['arms'][k]['forced']['rate']
    margin = lambda k: round(float(np.mean([x['gold_margin'] for x in results[k]])), 4)
    body = extra(acc, margin)
    if smoke:
        print(json.dumps({'smoke': name, 'meta': meta, 'ref': acc('ref'), 'block:rec': acc('block:rec'), **body})[:6000], flush=True)
        return body
    assert all(meta['noop_bitwise_equal'])
    save(f'{name.replace("_", "-")}.json', {'schema': f'source_binding_heads_{name}.v1', 'run_id': RUN, 'summary': s, **body, **meta})
    print(json.dumps({'ref': acc('ref'), 'block:rec': acc('block:rec'), **{k: v for k, v in body.items() if k != 'heads'}}), flush=True)
    return body


FLOOR = ('block:rec', {b: list(range(NH)) for b in LATE}, None)


def pilot_a(smoke=False):
    rows = [r for r in UP.discovery_half(True) if int(r['family'][-4:]) < 48][:16 if smoke else None]
    cand = [(b, h) for b in WINDOW for h in range(NH)]
    if smoke:
        cand = [(b, h) for b in WINDOW for h in range(0, NH, 4)]
    else:
        checked()
        assert not (OUT / 'pilot-a.json').exists()

    def extra(acc, margin):
        table = {f'{b}.{h}': {'only_margin': margin(f'only:{b}.{h}'), 'only_acc': acc(f'only:{b}.{h}'),
                              'blockw_margin': margin(f'blockw:{b}.{h}'), 'blockw_acc': acc(f'blockw:{b}.{h}')} for b, h in cand}
        order = sorted(cand, key=lambda x: (-table[f'{x[0]}.{x[1]}']['only_margin'], x))
        return {'order': [list(x) for x in order], 'ref_margin': margin('ref'), 'floor_margin': margin('block:rec'),
                'closed_window_margin': margin('blockw:none'), 'closed_window_acc': acc('blockw:none'), 'heads': table}
    arms = [FLOOR, ('blockw:none', {}, CLOSE)]
    arms += [(f'only:{b}.{h}', only([(b, h)]), None) for b, h in cand] + [(f'blockw:{b}.{h}', block([(b, h)]), CLOSE) for b, h in cand]
    return finish('pilot_a', rows, arms, smoke, extra)


def pilot_b(smoke=False):
    rows = UP.discovery_half(False)[:16 if smoke else None]
    if smoke:
        order = pilot_a(True)['order']
    else:
        checked()
        assert not (OUT / 'pilot-b.json').exists()
        order = D.read(OUT / 'pilot-a.json')['order']
    order = [tuple(x) for x in order]

    def extra(acc, margin):
        ks = next((k for k in KS if acc(f'only:K{k}') >= .80), None)
        kn = next((k for k in KS if acc(f'blockw:K{k}') <= .50), None)
        return {'sufficient_set': [list(x) for x in order[:ks or KS[-1]]], 'sufficient_rule_met': ks is not None,
                'necessary_set': [list(x) for x in order[:kn or KS[-1]]], 'necessary_rule_met': kn is not None,
                'sets': {f'K{k}': {'only': acc(f'only:K{k}'), 'blockw': acc(f'blockw:K{k}'), 'block': acc(f'block:K{k}')} for k in KS}}
    arms = [FLOOR]
    for k in KS:
        arms += [(f'only:K{k}', only(order[:k]), None), (f'blockw:K{k}', block(order[:k]), CLOSE), (f'block:K{k}', block(order[:k]), None)]
    return finish('pilot_b', rows, arms, smoke, extra)


def controls(K, purpose):
    K = [tuple(x) for x in K]
    out = []
    for i in range(NCTRL):
        rnd, taken, pick = stream(purpose, i), set(K), []
        for b in sorted({b for b, _ in K}):
            need = sum(1 for x in K if x[0] == b)
            pool = [(b, h) for h in range(NH) if (b, h) not in taken]
            if len(pool) < need:
                pool += [(c, h) for c in WINDOW for h in range(NH) if (c, h) not in taken and c != b]
            for j in rnd.choice(len(pool), need, replace=False):
                pick.append(pool[int(j)])
                taken.add(pool[int(j)])
        out.append(sorted(pick))
    return out


def freeze():
    checked()
    assert not (OUT / 'frozen.json').exists()
    a, b = D.read(OUT / 'pilot-a.json'), D.read(OUT / 'pilot-b.json')
    save('frozen.json', {'schema': 'source_binding_heads_frozen.v1', 'run_id': RUN, 'order': a['order'],
                         **{k: b[k] for k in ('sufficient_set', 'sufficient_rule_met', 'necessary_set', 'necessary_rule_met')},
                         'controls_S': controls(b['sufficient_set'], 'ctrlS'), 'controls_N': controls(b['necessary_set'], 'ctrlN'),
                         'prereg_sha256': D.sha(OUT / 'prereg.json'), 'pilot_a_sha256': D.sha(OUT / 'pilot-a.json'),
                         'pilot_b_sha256': D.sha(OUT / 'pilot-b.json'), 'created_at': datetime.now().astimezone().isoformat()})
    fz = D.read(OUT / 'frozen.json')
    print(json.dumps({k: fz[k] for k in ('sufficient_set', 'sufficient_rule_met', 'necessary_set', 'necessary_rule_met')}), flush=True)


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
    save('gate.json', {'schema': 'source_binding_heads_gate.v1', 'run_id': RUN, 'items_sha256': D.sha(OUT / 'gate-items.json'), **g, **meta})
    print(json.dumps({k: g[k] for k in ('passed', 'forced_accuracy', 'top1_accuracy')}), flush=True)


def run():
    from tools import provenance
    pre, fz = frozen()
    assert not (OUT / 'report.json').exists(), 'holdout already used'
    g = D.read(OUT / 'gate.json')
    rows = D.read(OUT / f'{SPLIT}-materials.json')
    S, N = [tuple(x) for x in fz['sufficient_set']], [tuple(x) for x in fz['necessary_set']]
    cs, cn = fz['controls_S'], fz['controls_N']
    names = ('H1_sufficient', 'H2_necessary_in_window', 'H3_no_later_compensation')
    if g['passed']:
        arms = [FLOOR, ('blockw:none', {}, CLOSE), ('only:S', only(S), None), ('blockw:N', block(N), CLOSE), ('block:N', block(N), None)]
        arms += [(f'only:ctrl{i}', only(c), None) for i, c in enumerate(cs)] + [(f'blockw:ctrl{i}', block(c), CLOSE) for i, c in enumerate(cn)]
        for b in WINDOW:
            hs = [(b, h) for h in range(NH)]
            arms += [(f'only:B{b}', only(hs), None), (f'blockw:B{b}', block(hs), CLOSE)]
        arms += [('only:B32-39', only([(b, h) for b in LATE if b >= CLOSE for h in range(NH)]), None)]
        for b, h in [tuple(x) for x in fz['order'][:TOP_SINGLE]]:
            arms += [(f'only:{b}.{h}', only([(b, h)]), None), (f'blockw:{b}.{h}', block([(b, h)]), CLOSE)]
        results, meta = engine(rows, arms, stage='run')
        save('items.json', results)
        base = D.read(OUT / 'gate-items.json')
        meta['base_replay_max_score_difference'] = max(abs(a - b) for x, y in zip(results['base'], base) for a, b in zip(x['scores'], y['scores']))
        pairs = [('only:S', f'only:ctrl{i}') for i in range(NCTRL)] + [(f'blockw:ctrl{i}', 'blockw:N') for i in range(NCTRL)]
        s = RL.summarise(results, rows, pairs)
        acc = lambda k: s['arms'][k]['forced']['rate']
        tri = lambda ok, bad: 'supported' if ok else 'not_supported' if bad else 'inconclusive'
        beat = lambda keys: all(s['differences'][k]['rate'] >= .20 and s['differences'][k]['ci95'][0] > 0 for k in keys)
        ok = all(meta['noop_bitwise_equal']) and acc('ref') >= .90 and acc('block:rec') <= .40
        v = {'controls_valid': bool(ok)}
        if ok:
            v['H1_sufficient'] = tri(acc('only:S') >= .70 and beat([f'only:S-only:ctrl{i}' for i in range(NCTRL)]), acc('only:S') < .50)
            v['H2_necessary_in_window'] = tri(acc('blockw:N') <= .50 and beat([f'blockw:ctrl{i}-blockw:N' for i in range(NCTRL)]),
                                              acc('blockw:N') >= .80)
            v['H3_no_later_compensation'] = tri(acc('block:N') <= .50, acc('block:N') >= .80)
        else:
            v.update({k: 'controls_invalid' for k in names})
    else:
        s, meta, v = {'note': 'behaviour gate failed; no edit was run'}, {}, {k: 'behavior_ineligible' for k in names}
    report = {'schema': 'source_binding_heads_report.v1', 'run_id': RUN, 'split': SPLIT, 'sufficient_set': fz['sufficient_set'],
              'necessary_set': fz['necessary_set'], 'controls_S': cs, 'controls_N': cn,
              'gate_passed': g['passed'], 'prereg_sha256': D.sha(OUT / 'prereg.json'), 'frozen_sha256': D.sha(OUT / 'frozen.json'),
              'gate_sha256': D.sha(OUT / 'gate.json'), 'items_sha256': D.sha(OUT / 'items.json') if g['passed'] else None,
              'criteria': pre['criteria'], 'verdicts': v, 'summary': s, **meta, 'formal_evidence_promotion': 0,
              'custody': 'holdout_heads read once by the gate stage and once by this run',
              'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    save('report.json', report)
    path = provenance.write_result('SOURCE-BINDING-heads-holdout', {'verdicts': v, 'gate_passed': g['passed'], 'sufficient_set': fz['sufficient_set'],
                                                                    'necessary_set': fz['necessary_set'], 'report_sha256': D.sha(OUT / 'report.json')},
                                   model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)
    print(json.dumps({'verdicts': v, 'S': fz['sufficient_set'], 'N': fz['necessary_set'], 'result': str(path)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['prereg', 'pilot_a', 'pilot_b', 'freeze', 'gate', 'run'])
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    f = {'prereg': prereg, 'pilot_a': pilot_a, 'pilot_b': pilot_b, 'freeze': freeze, 'gate': gate, 'run': run}[a.stage]
    f(a.smoke) if a.stage.startswith('pilot') else f()
