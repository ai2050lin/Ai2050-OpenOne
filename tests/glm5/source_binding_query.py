"""Where does the relay position get the query that head 30.4 uses?  Qwen3-14B BF16, source binding.

Follows Claude Phase 401: at the relay position R head 30.4 reads the value token of the record that its query selects;
the query (archive x entity, partly record layout) is already in the state of R before block 28.
Here the attention of R (all heads) is edited only in blocks [l, 28); from block 28 on R is untouched, so the value
read itself is not edited.  Readout as before: from block 24 on the final position may attend only to R.
  all@l     in blocks [l, 28) R attends only to itself and the chat-template head (the default attention target)
  only-G    in blocks [L0, 28) R attends only to the positions of G (plus itself and the template head)
  block-G   in blocks [L0, 28) R cannot attend to the positions of G
Two outcomes: keep (4-way forced choice) and the attention mass of head 30.4 from R on the correct record, recomputed
from the q / k of the edited forward.  One chunk per panel, inner batch 128.
Stages: prereg -> pilot_a (timing) -> pilot_b (single groups) -> pilot_c (cumulative sets) -> freeze -> gate -> run.
"""
import argparse, gc, hashlib, json, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_binding_design as D
import source_binding_relay as RL
import source_binding_upstream as UP
import source_binding_heads as HD
import source_binding_qkov as QK
from source_binding_runtime import Runtime

SRC, MODEL = D.OUT, 'qwen14'
PANELS = QK.PANELS + [QK.OUT / 'holdout_qkov-materials.json']
OUT = D.ROOT / 'tests/glm5/result/source_binding_query_v1'
RUN, SEED, SPLIT = 'SOURCE-BINDING-QUERY-20261111-v1', 202611110402, 'holdout_query'
FINAL_LAYER, END, FAMILIES, BATCH, NB, R_FROM_END = 24, 28, 96, 128, 40, UP.R_FROM_END
TIMING, HEAD, KMAX = [0, 4, 8, 12, 16, 18, 20, 22, 24, 26], (30, 4), 8
GROUPS = ['tpl_head', 'intro', 'hdr_correct', 'hdr_other', 'rec_correct', 'rec_other_source', 'rec_other_entity', 'rec_neither',
          'qsrc', 'qent', 'qrest', 'instr', 'tail0']
CAND = [g for g in GROUPS if g != 'tpl_head']
UNIONS = {'records4': ['rec_correct', 'rec_other_source', 'rec_other_entity', 'rec_neither'], 'question': ['qsrc', 'qent', 'qrest']}
CRITERIA = {
    'model': 'Qwen3-14B original BF16, CPU byte reads, CUDA layer streaming. GLM-4-9B is not run (failed the behaviour gate of this task).',
    'readout': 'in every edited arm and in the reference, from block 24 on the final position may attend only to R (and itself)',
    'edit': 'attention mask of R only, all heads, in blocks [l, 28). In all@l and only-G arms R can always attend to itself and to the '
            'chat-template head (tpl_head), which is the default attention target; tpl_head is not a candidate group; from block 28 on R is untouched (heads 28.21 and 30.4 read freely); '
            'every later position is recomputed by the original blocks',
    'groups_before_R': {'hdr_correct': 'header line of the requested archive', 'hdr_other': 'header line of the other archive',
                        'tpl_head': 'chat-template tokens before the user text', 'intro': 'the introduction sentence', 'rec_correct': 'record of the requested archive and entity',
                        'rec_other_source': 'same entity, other archive', 'rec_other_entity': 'same archive, other entity',
                        'rec_neither': 'other archive, other entity', 'qsrc': 'archive name in the question',
                        'qent': 'entity name in the question', 'qrest': 'other tokens of the question', 'instr': 'the instruction sentence',
                        'tail0': '<|im_end|>', 'note': 'disjoint; they cover every position before R'},
    'outcomes': {'keep': '4-way forced choice among the four code words (chance 0.25)',
                 'mass': 'attention of block 30 head 4 from R on the tokens of the correct record, recomputed from the q and k of the edited '
                         'forward (the query at work)', 'intervals': 'all items; family-bootstrap 95 %'},
    'selection_on_discovery': {
        'L0': 'pilot_a (families 32-63): all@l for l in [0, 4, 8, 12, 16, 18, 20, 22, 24, 26]. L0 = the largest l with keep <= 0.40; '
              'if none, 0 (flagged)',
        'order': 'pilot_b (families 32-63), window [L0, 28): the 12 candidate groups ordered by only-g keep (descending; ties by block-g keep ascending)',
        'sets': 'pilot_c (families 0-31): cumulative sets S_k of the first k groups, k = 1..8. Sufficient set S = smallest S_k with '
                'only-S_k keep >= 0.90; necessary set N = smallest S_k with block-S_k keep <= 0.50 (if none, S_8, flagged)'},
    'material': 'holdout_query: 96 new families x 8 items; entity names that occur in no earlier panel of this line; the 16 discovery '
                'code words, 4 seen frames',
    'two_pass': 'gate stage: unmodified forward and the v1 behaviour gate. Edits run only if the gate passed; otherwise behavior_ineligible.',
    'predictions_on_holdout': {
        'controls_valid': 'noop reproduces the base logits bit for bit; reference keep >= 0.90; all@L0 keep <= 0.40',
        'V1_sufficient': {'supported': 'only-S keep >= 0.80', 'not_supported': 'only-S keep < 0.60'},
        'V2_necessary': {'supported': 'block-N keep <= 0.50', 'not_supported': 'block-N keep >= 0.80'},
        'V3_redundant_routes': {'supported': 'for every group g of N (if N has more than one group), block-(N minus g) keep is at least '
                                             '0.10 above block-N with family-bootstrap lower bound > 0',
                                'not_supported': 'for some g the gain is below 0.02', 'not_applicable': 'N has one group'},
        'A1_query_formed_from_S': {'supported': 'only-S mass >= 0.70', 'not_supported': 'only-S mass < 0.40'},
        'A2_query_lost_without_N': {'supported': 'block-N mass <= 0.50', 'not_supported': 'block-N mass >= 0.80'}},
    'descriptive': 'all@l for every timing block; only-g and block-g for every single group; only / block for the union of the four '
                   'records and for the whole question (an only-arm on the correct record alone hands R the selection for free, the '
                   'unions do not); top-1, margins',
    'not_covered': 'which heads write the query into R; how the positions R reads from obtained their content; routes that do not pass '
                   'through R; other models, unseen answer words, unseen frames',
    'disclosure': 'the code was smoke-tested on 32 discovery items (exposed) before pre-registration; thresholds were written first. '
                  'The smoke test showed that blocking the template head alone lowers the attention mass of head 30.4 (0.96 -> 0.43) while '
                  'the answer is kept (0.94), so tpl_head was made always readable in all@l and only-G arms and removed from the candidates.',
}


def stream(purpose, key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(f'{RUN}|{SEED}|{purpose}|{key}'.encode()).digest()[:16], 'little'))


def save(name, body):
    from deploy.storage_policy import check_write
    p = OUT / name
    check_write(p, 1 << 20, operation='来源绑定查询上溯产物')
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(body, ensure_ascii=False, indent=1, default=lambda x: x.item() if isinstance(x, np.generic) else x), encoding='utf-8')
    t.replace(p)


def log(*a, **kw):
    RL.OUT = OUT
    RL.log(*a, **kw)


def add_groups(tok, r):
    r = UP.add_groups(tok, r)
    s = r['source']
    hc, ho = r['headers'][str(s)], r['headers'][str(1 - s)]
    up = dict(r['up'])
    rest = [i for i in up.pop('pre_other') if i not in hc and i not in ho]
    first = min(i for i, (lo, hi) in enumerate(tok(r['prompt'], add_special_tokens=False, return_offsets_mapping=True)['offset_mapping'])
                if hi > r['prompt'].index(r['content']))
    up.update(hdr_correct=hc, hdr_other=ho, tpl_head=[i for i in rest if i < first], intro=[i for i in rest if i >= first])
    assert hc and ho and sorted(i for v in up.values() for i in v) == list(range(len(r['tokens']) - R_FROM_END)) and set(up) == set(GROUPS)
    return {**r, 'up': up}


def discovery_half(second):
    tok = RL.tokenizer()
    rows = D.read(SRC / f'{MODEL}-materials.json')['splits']['discovery']
    return [add_groups(tok, r) for r in rows if (int(r['family'][-4:]) >= 32) == second]


def build_materials():
    tok = RL.tokenizer()
    used = {n for rows in D.read(SRC / f'{MODEL}-materials.json')['splits'].values() for r in rows for n in r['names']}
    for p in PANELS:
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
                    r = {**r, 'records': recs, 'headers': {str(k): [i for i, (lo, hi) in enumerate(enc['offset_mapping']) if hi > off + a and lo < off + b]
                                     for k, (a, b) in r['headers'].items()}, 'prompt': prompt, 'tokens': ids,
                         'choice_ids': choice, 'gold_id': choice[r['target_record']]}
                    rows.append(add_groups(tok, r))
    return rows


def prereg():
    from deploy.storage_policy import checked_dir
    from tools import provenance
    from tools.power_analysis import check_phase
    checked_dir(OUT, 1 << 30, operation='来源绑定查询上溯产物')
    assert not (OUT / 'prereg.json').exists()
    plan = D.checked(MODEL)
    rows = build_materials()
    assert len(rows) == FAMILIES * 8 and len({tuple(r['tokens']) for r in rows}) == len(rows)
    save(f'{SPLIT}-materials.json', rows)
    n = len(rows)
    body = {'schema': 'source_binding_query_prereg.v1', 'run_id': RUN, 'seed': SEED, 'created_at': datetime.now().astimezone().isoformat(),
            'base_plan_sha256': D.sha(SRC / 'plan.json'), 'qkov_report_sha256': D.sha(QK.OUT / 'report.json'),
            'model_snapshot': plan['models'][MODEL], 'code_sha256': D.sha(__file__),
            'dependency_sha256': {m.__name__: D.sha(m.__file__) for m in (RL, UP, HD, QK)},
            'criteria': CRITERIA, 'materials': {'split': SPLIT, 'sha256': D.sha(OUT / f'{SPLIT}-materials.json'), 'families': FAMILIES, 'rows': n},
            'power': {'unit': 'paired binary outcome per item (McNemar); alpha = 0.05',
                      'V1_V2': {'expected_delta': 0.40, 'p_d': 0.50, **check_phase(0.40, n, p_d=0.50)},
                      'V3': {'expected_delta': 0.10, 'p_d': 0.20, **check_phase(0.10, n, p_d=0.20)},
                      'A1_A2': 'continuous attention mass; stated on means with family-bootstrap intervals, n = 768',
                      'caveat': 'items are clustered in 96 families and 4 frames'},
            'custody': 'L0 and the sets come from discovery (exposed). holdout_query is generated here, read first by the gate stage '
                       '(behaviour only) and then once by the run stage.',
            'resources': {'gpu_minutes_estimate': 30, 'disk_GiB_estimate': 0.2, 'models_concurrent': 1,
                          'shape': 'one chunk per panel, inner batch 128, fixed before pre-registration'},
            'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    body['provenance']['model_revision'] = plan['models'][MODEL]
    save('prereg.json', body)
    print(json.dumps({'prereg_sha256': D.sha(OUT / 'prereg.json'), 'V1_V2': body['power']['V1_V2'], 'V3': body['power']['V3']}), flush=True)


def checked():
    p = D.read(OUT / 'prereg.json')
    assert p['code_sha256'] == D.sha(__file__) and all(p['dependency_sha256'][m.__name__] == D.sha(m.__file__) for m in (RL, UP, HD, QK)), \
        'source changed after pre-registration'
    assert p['materials']['sha256'] == D.sha(OUT / f'{SPLIT}-materials.json')
    D.checked(MODEL)
    return p


def engine(rows, arms=(), stage='run'):
    """arms: (name, start, kind, groups); kind in all / only / block, applied to R in blocks [start, 28)."""
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    t = torch
    assert snapshot(D.MODELS[MODEL]) == D.read(SRC / f'{MODEL}-identity.json')['model_snapshot']
    assert all(0 <= a[1] < END for a in arms)
    start, N = time.monotonic(), len(rows)
    results, mass = {}, {}
    with GpuLease(), t.inference_mode():
        rt = Runtime(MODEL)
        hd, grp = rt.hd, rt.heads // rt.kv
        ids, pos, pe, mask = rt.context(rows)
        h, length = rt.embed(ids), ids.shape[1]
        R = length - R_FROM_END
        log(stage, 0, NB, start, rows=N, length=length, arms=len(arms))
        frow = t.zeros((N, length), dtype=t.bool, device='cuda')
        frow[:, R] = True
        frow[:, -1] = True
        cor = t.zeros((N, length), dtype=t.bool, device='cuda')
        rrow = {}
        for i, r in enumerate(rows):
            off = length - len(r['tokens'])
            assert len(r['tokens']) - R_FROM_END == r['up']['tail0'][-1] + 1
            cor[i, [off + j for j in r['up']['rec_correct']]] = True
        for name, _, kind, groups in arms:
            row = mask[:, 0, R].clone()
            for i, r in enumerate(rows):
                off = length - len(r['tokens'])
                sel = [off + j for g in groups for j in r['up'][g]]
                if kind == 'block':
                    row[i, sel] = False
                else:
                    keep = row[i].clone()
                    row[i] = False
                    row[i, R] = True
                    sink = [off + j for j in r['up']['tpl_head']]
                    row[i, sink] = keep[sink]
                    if kind == 'only':
                        row[i, sel] = keep[sel]
            rrow[name] = row
        state, hook = {}, {}
        qs, ks = slice(HEAD[1] * hd, (HEAD[1] + 1) * hd), slice((HEAD[1] // grp) * hd, (HEAD[1] // grp + 1) * hd)
        for b in range(NB):
            if arms and b == FINAL_LAYER:
                state['noop'], state['ref'] = h[:, R:].clone(), h[:, R:].clone()
            for name, l, _, _ in arms:
                if l == b:
                    state[name] = (state['ref'] if b > FINAL_LAYER else h[:, R:]).clone()
            layer = rt.block(b)
            handles = []
            if arms and b == HEAD[0]:
                a = layer.self_attn
                handles = [a.q_proj.register_forward_hook(lambda m, i, o: hook.__setitem__('q', o[:, R, qs].clone())),
                           a.k_proj.register_forward_hook(lambda m, i, o: hook.__setitem__('k', o[..., ks].clone()))]
                rot = lambda x, co, si: x * co + rt.mod.rotate_half(x) * si
            y = t.empty_like(h)
            for lo in range(0, N, BATCH):
                hi = min(lo + BATCH, N)
                kw = {'position_ids': pos[lo:hi], 'position_embeddings': tuple(p[lo:hi] for p in pe), 'use_cache': False}
                y[lo:hi] = layer(h[lo:hi], attention_mask=mask[lo:hi], **kw)

                def measure(key):
                    if not handles:
                        return
                    cos, sin = pe[0][lo:hi], pe[1][lo:hi]
                    qr = rot(a.q_norm(hook['q']), cos[:, R], sin[:, R])
                    kr = rot(a.k_norm(hook['k']), cos, sin)
                    w = ((qr[:, None, :] * kr).sum(-1) * a.scaling).float().masked_fill(~mask[lo:hi, 0, R], float('-inf')).softmax(-1)
                    mass.setdefault(key, []).extend((w * cor[lo:hi]).sum(-1).cpu().tolist())
                measure('base')
                if not state:
                    continue
                new = {}
                mref = mask[lo:hi].clone()
                mref[:, 0, -1] = frow[lo:hi]
                for key in state:
                    if key == 'noop':
                        m = mask[lo:hi]
                    else:
                        m = mref if b >= FINAL_LAYER else mask[lo:hi]
                        if key != 'ref' and b < END:
                            m = m.clone()
                            m[:, 0, R] = rrow[key][lo:hi]
                    new[key] = layer(t.cat([h[lo:hi, :R], state[key][lo:hi]], 1), attention_mask=m, **kw)[:, R:]
                    measure(key)
                for key, v in new.items():
                    state[key][lo:hi] = v
            h = y
            for x in handles:
                x.remove()
            del layer, y
            gc.collect()
            t.cuda.empty_cache()
            if (b + 1) % 4 == 0:
                log(stage, b + 1, NB, start, block=b, arms=len(state))
        norm = rt.Norm(rt.dim, eps=rt.cfg.rms_norm_eps).to(device='cuda', dtype=t.bfloat16)
        norm.weight = t.nn.Parameter(rt.weight('model.norm.weight'), requires_grad=False)
        W = rt.weight('lm_head.weight' if 'lm_head.weight' in rt.index else 'model.embed_tokens.weight')
        logits = lambda x: t.cat([(norm(x[lo:lo + BATCH]) @ W.T).float() for lo in range(0, N, BATCH)])
        lg0 = logits(h[:, -1])
        results['base'] = rt.score(lg0, rows)
        equal = None
        for key, s in state.items():
            lg = logits(s[:, -1])
            if key == 'noop':
                equal = bool(t.equal(lg, lg0))
            results[key] = rt.score(lg, rows)
            del lg
        peak = round(t.cuda.max_memory_allocated() / 2 ** 20)
    return results, mass, {'noop_bitwise_equal': equal, 'peak_MiB': peak, 'seconds': round(time.monotonic() - start, 1), 'padded_length': length}


def summarise(results, mass, rows, pairs=()):
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
    out = {'n': len(rows), 'arms': {}, 'differences': {}}
    for k in results:
        out['arms'][k] = {'keep': ci(col(k, 'forced_correct')), 'top1': ci(col(k, 'top1_correct')),
                          'gold_margin': round(float(col(k, 'gold_margin').mean()), 3), 'mass': ci(mass[k]) if k in mass else None}
    for a, b in pairs:
        out['differences'][f'{a}-{b}'] = ci(col(a, 'forced_correct') - col(b, 'forced_correct'))
    return out


def finish(name, rows, arms, smoke, extra):
    results, mass, meta = engine(rows, arms, stage=name)
    s = summarise(results, mass, rows)
    keep = lambda k: s['arms'][k]['keep']['rate']
    ms = lambda k: s['arms'][k]['mass']['rate']
    body = extra(keep, ms)
    if smoke:
        print(json.dumps({'smoke': name, 'meta': meta, 'ref': [keep('ref'), ms('ref')], **body}), flush=True)
        return body
    assert meta['noop_bitwise_equal']
    save(f'{name.replace("_", "-")}.json', {'schema': f'source_binding_query_{name}.v1', 'run_id': RUN, 'summary': s, **body, **meta})
    print(json.dumps({'ref': [keep('ref'), ms('ref')], **body}), flush=True)
    return body


def pilot_a(smoke=False):
    rows = discovery_half(True)[:32 if smoke else None]
    if not smoke:
        checked()
        assert not (OUT / 'pilot-a.json').exists()

    def extra(keep, ms):
        table = {str(l): [keep(f'all@{l}'), ms(f'all@{l}')] for l in TIMING}
        l0 = max([l for l in TIMING if table[str(l)][0] <= .40], default=None)
        return {'timing_keep_mass': table, 'L0': l0 if l0 is not None else 0, 'L0_rule_met': l0 is not None}
    return finish('pilot_a', rows, [(f'all@{l}', l, 'all', []) for l in TIMING], smoke, extra)


def pilot_b(smoke=False):
    rows = discovery_half(True)[:32 if smoke else None]
    if smoke:
        l0 = pilot_a(True)['L0']
    else:
        checked()
        assert not (OUT / 'pilot-b.json').exists()
        l0 = D.read(OUT / 'pilot-a.json')['L0']

    def extra(keep, ms):
        order = sorted(CAND, key=lambda g: (-keep(f'only:{g}'), keep(f'block:{g}'), CAND.index(g)))
        return {'L0': l0, 'order': order, 'only_block_keep_mass': {g: [keep(f'only:{g}'), keep(f'block:{g}'), ms(f'only:{g}'), ms(f'block:{g}')]
                                                                 for g in order},
                'block_tpl_head_keep_mass': [keep('block:tpl_head'), ms('block:tpl_head')]}
    return finish('pilot_b', rows, [(f'{k}:{g}', l0, k, [g]) for g in CAND for k in ('only', 'block')] + [('block:tpl_head', l0, 'block', ['tpl_head'])],
                  smoke, extra)


def pilot_c(smoke=False):
    rows = discovery_half(False)[:32 if smoke else None]
    if smoke:
        b = pilot_b(True)
    else:
        checked()
        assert not (OUT / 'pilot-c.json').exists()
        b = D.read(OUT / 'pilot-b.json')
    l0, order = b['L0'], b['order']
    ks = range(1, KMAX + 1)

    def extra(keep, ms):
        k = next((n for n in ks if keep(f'only:S{n}') >= .90), None)
        kn = next((n for n in ks if keep(f'block:S{n}') <= .50), None)
        return {'L0': l0, 'order': order, 'sufficient_set': order[:k or KMAX], 'sufficient_rule_met': k is not None,
                'necessary_set': order[:kn or KMAX], 'necessary_rule_met': kn is not None,
                'only_block_keep_mass': {f'S{n}': [keep(f'only:S{n}'), keep(f'block:S{n}'), ms(f'only:S{n}'), ms(f'block:S{n}')] for n in ks}}
    return finish('pilot_c', rows, [(f'{k}:S{n}', l0, k, order[:n]) for n in ks for k in ('only', 'block')], smoke, extra)


def freeze():
    checked()
    assert not (OUT / 'frozen.json').exists()
    c = D.read(OUT / 'pilot-c.json')
    save('frozen.json', {'schema': 'source_binding_query_frozen.v1', 'run_id': RUN, **{k: c[k] for k in (
        'L0', 'order', 'sufficient_set', 'sufficient_rule_met', 'necessary_set', 'necessary_rule_met')},
        'L0_rule_met': D.read(OUT / 'pilot-a.json')['L0_rule_met'], 'prereg_sha256': D.sha(OUT / 'prereg.json'),
        **{f'pilot_{x}_sha256': D.sha(OUT / f'pilot-{x}.json') for x in 'abc'}, 'created_at': datetime.now().astimezone().isoformat()})
    print(json.dumps(D.read(OUT / 'frozen.json'), ensure_ascii=False), flush=True)


def frozen():
    pre, fz = checked(), D.read(OUT / 'frozen.json')
    assert fz['prereg_sha256'] == D.sha(OUT / 'prereg.json') and fz['pilot_c_sha256'] == D.sha(OUT / 'pilot-c.json')
    return pre, fz


def gate():
    frozen()
    assert not (OUT / 'gate.json').exists(), 'holdout behaviour already read'
    rows = D.read(OUT / f'{SPLIT}-materials.json')
    results, _, meta = engine(rows, stage='gate')
    save('gate-items.json', results['base'])
    g = D.behavior(rows, results['base'])
    save('gate.json', {'schema': 'source_binding_query_gate.v1', 'run_id': RUN, 'items_sha256': D.sha(OUT / 'gate-items.json'), **g, **meta})
    print(json.dumps({k: g[k] for k in ('passed', 'forced_accuracy', 'top1_accuracy')}), flush=True)


def run():
    from tools import provenance
    pre, fz = frozen()
    assert not (OUT / 'report.json').exists(), 'holdout already used'
    g = D.read(OUT / 'gate.json')
    rows = D.read(OUT / f'{SPLIT}-materials.json')
    S, N, l0 = fz['sufficient_set'], fz['necessary_set'], fz['L0']
    names = ('V1_sufficient', 'V2_necessary', 'V3_redundant_routes', 'A1_query_formed_from_S', 'A2_query_lost_without_N')
    if g['passed']:
        arms = [('only:S', l0, 'only', S), ('block:N', l0, 'block', N)]
        if len(N) > 1:
            arms += [(f'block:N-{x}', l0, 'block', [y for y in N if y != x]) for x in N]
        arms += [(f'all@{l}', l, 'all', []) for l in TIMING]
        arms += [(f'{k}:{x}', l0, k, [x]) for x in CAND for k in ('only', 'block')] + [('block:tpl_head', l0, 'block', ['tpl_head'])]
        arms += [(f'{k}:{u}', l0, k, gs) for u, gs in UNIONS.items() for k in ('only', 'block')]
        results, mass, meta = engine(rows, arms, stage='run')
        save('items.json', {'results': results, 'mass': mass})
        base = D.read(OUT / 'gate-items.json')
        meta['base_replay_max_score_difference'] = max(abs(a - b) for x, y in zip(results['base'], base) for a, b in zip(x['scores'], y['scores']))
        s = summarise(results, mass, rows, [(f'block:N-{x}', 'block:N') for x in N] if len(N) > 1 else [])
        keep = lambda k: s['arms'][k]['keep']['rate']
        ms = lambda k: s['arms'][k]['mass']['rate']
        tri = lambda ok, bad: 'supported' if ok else 'not_supported' if bad else 'inconclusive'
        ok = meta['noop_bitwise_equal'] and keep('ref') >= .90 and keep(f'all@{l0}') <= .40
        v = {'controls_valid': bool(ok)}
        if ok:
            v['V1_sufficient'] = tri(keep('only:S') >= .80, keep('only:S') < .60)
            v['V2_necessary'] = tri(keep('block:N') <= .50, keep('block:N') >= .80)
            ds = list(s['differences'].values())
            v['V3_redundant_routes'] = 'not_applicable' if len(N) == 1 else tri(
                all(d['rate'] >= .10 and d['ci95'][0] > 0 for d in ds), any(d['rate'] < .02 for d in ds))
            v['A1_query_formed_from_S'] = tri(ms('only:S') >= .70, ms('only:S') < .40)
            v['A2_query_lost_without_N'] = tri(ms('block:N') <= .50, ms('block:N') >= .80)
        else:
            v.update({k: 'controls_invalid' for k in names})
    else:
        s, meta, v = {'note': 'behaviour gate failed; no edit was run'}, {}, {k: 'behavior_ineligible' for k in names}
    report = {'schema': 'source_binding_query_report.v1', 'run_id': RUN, 'split': SPLIT, 'L0': l0, 'sufficient_set': S, 'necessary_set': N,
              'gate_passed': g['passed'], 'prereg_sha256': D.sha(OUT / 'prereg.json'), 'frozen_sha256': D.sha(OUT / 'frozen.json'),
              'gate_sha256': D.sha(OUT / 'gate.json'), 'items_sha256': D.sha(OUT / 'items.json') if g['passed'] else None,
              'criteria': pre['criteria'], 'verdicts': v, 'summary': s, **meta, 'formal_evidence_promotion': 0,
              'custody': 'holdout_query read once by the gate stage and once by this run',
              'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    save('report.json', report)
    path = provenance.write_result('SOURCE-BINDING-query-holdout', {'verdicts': v, 'gate_passed': g['passed'], 'L0': l0, 'sufficient_set': S,
                                                                    'necessary_set': N, 'report_sha256': D.sha(OUT / 'report.json')},
                                   model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)
    print(json.dumps({'verdicts': v, 'L0': l0, 'sufficient_set': S, 'necessary_set': N, 'result': str(path)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['prereg', 'pilot_a', 'pilot_b', 'pilot_c', 'freeze', 'gate', 'run'])
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    f = {'prereg': prereg, 'pilot_a': pilot_a, 'pilot_b': pilot_b, 'pilot_c': pilot_c, 'freeze': freeze, 'gate': gate, 'run': run}[a.stage]
    f(a.smoke) if a.stage.startswith('pilot') else f()
