"""Before block 24: how does 'which archive, which entity' enter the relay position?  Qwen3-14B BF16, source binding.

Follows Claude Phase 402: the relay position R forms the read query in blocks 24-25; if R sees only the four records in
blocks 24-27 (nothing of the question) the query still forms, so the request is already in the state of R before
block 24.  Here the attention of R (all heads) is edited only in blocks [l, 24).
  iso arms   in blocks 24-27 R sees only the four records (plus itself and the chat-template head), so the request can
             reach the query only through the state R had at block 24 (primary)
  nat arms   blocks 24-27 untouched: R may also read the question directly there (descriptive)
  all@l / only-G / block-G as before, applied in [l, 24) or [L0, 24)
Speed: only positions >= R differ from the base forward, so every arm recomputes just these 11 positions per block,
with the keys and values of the earlier positions taken from the base forward.  This recomputation is not bit-identical
to the library layer; the no-edit arm is therefore checked against the base logits with a tolerance.
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
import source_binding_query as QY
from source_binding_runtime import Runtime

SRC, MODEL = D.OUT, 'qwen14'
PANELS = QY.PANELS + [QY.OUT / 'holdout_query-materials.json']
OUT = D.ROOT / 'tests/glm5/result/source_binding_request_v1'
RUN, SEED, SPLIT = 'SOURCE-BINDING-REQUEST-20261111-v1', 202611110403, 'holdout_request'
FINAL_LAYER, END, ISO_END, FAMILIES, BATCH, NB, NH, R_FROM_END = 24, 24, 28, 96, 128, 40, 40, UP.R_FROM_END
TIMING, HEAD, KMAX, TOL = [0, 4, 8, 12, 14, 16, 18, 20, 22], (30, 4), 8, 1.0
GROUPS, CAND, UNIONS = QY.GROUPS, QY.CAND, QY.UNIONS
CRITERIA = {
    'model': 'Qwen3-14B original BF16, CPU byte reads, CUDA layer streaming. GLM-4-9B is not run (failed the behaviour gate of this task).',
    'readout': 'in every edited arm and in the references, from block 24 on the final position may attend only to R (and itself)',
    'edit': 'attention mask of R only, all heads, in blocks [l, 24). In all@l and only-G arms R can always attend to itself and to the '
            'chat-template head (default attention target; not a candidate group). Every later position is recomputed.',
    'iso': 'primary arms: in blocks 24-27 R can attend only to the four records, itself and the template head, so in the window where the '
           'query forms nothing of the question is visible to R (Phase 402: keep 0.919, mass 0.830). ref_iso has this restriction only.',
    'nat': 'descriptive arms: blocks 24-27 untouched, so R may read the question directly there. ref_nat has no edit at R.',
    'groups_before_R': QY.CRITERIA['groups_before_R'],
    'outcomes': {'keep': '4-way forced choice among the four code words (chance 0.25)',
                 'mass': 'attention of block 30 head 4 from R on the tokens of the correct record in the edited forward',
                 'intervals': 'all items; family-bootstrap 95 %'},
    'tail_recomputation': 'arms recompute positions >= R with their own code path (same weights, same formulas). Control: the no-edit arm, '
                          'recomputed this way from block 0, must give the same forced choice as the base forward on every item and a '
                          'maximal absolute difference of the four choice scores <= 1.0',
    'selection_on_discovery': {
        'L0': 'pilot_a (families 32-63): iso all@l for l in [0, 4, 8, 12, 14, 16, 18, 20, 22]. L0 = the largest l with keep <= 0.40; '
              'if none, 0 (flagged)',
        'order': 'pilot_b (families 32-63), window [L0, 24), iso: the 12 candidate groups ordered by only-g keep (descending; ties by '
                 'block-g keep ascending)',
        'sets': 'pilot_c (families 0-31), iso: cumulative sets S_k of the first k groups, k = 1..8. Sufficient set S = smallest S_k with '
                'only-S_k keep >= 0.80; necessary set N = smallest S_k with block-S_k keep <= 0.50 (if none, S_8, flagged)'},
    'material': 'holdout_request: 96 new families x 8 items; entity names that occur in no earlier panel of this line; the 16 discovery '
                'code words, 4 seen frames',
    'two_pass': 'gate stage: unmodified forward and the v1 behaviour gate. Edits run only if the gate passed; otherwise behavior_ineligible.',
    'predictions_on_holdout': {
        'controls_valid': 'tail recomputation control holds; ref_nat keep >= 0.90; ref_iso keep >= 0.80; iso all@L0 keep <= 0.40',
        'W1_sufficient': {'supported': 'iso only-S keep >= 0.70', 'not_supported': 'keep < 0.50'},
        'W2_necessary': {'supported': 'iso block-N keep <= 0.50', 'not_supported': 'keep >= 0.75'},
        'W3_redundant_routes': {'supported': 'for every group g of N (if N has more than one group), iso block-(N minus g) keep is at '
                                             'least 0.10 above iso block-N with family-bootstrap lower bound > 0',
                                'not_supported': 'for some g the gain is below 0.02', 'not_applicable': 'N has one group'},
        'W4_direct_route_compensates': {'supported': 'nat block-N keep >= 0.80 (with blocks 24-27 free, R recovers the request there)',
                                        'not_supported': 'nat block-N keep <= 0.50'}},
    'descriptive': 'mass for every arm; iso and nat all@l; iso only-g and block-g for every single group and the two unions; nat only-S',
    'not_covered': 'which heads move the request into R; how the question tokens were contextualised; routes that do not pass through R; '
                   'other models, unseen answer words, unseen frames',
    'disclosure': 'the code was smoke-tested on 32 discovery items (exposed) before pre-registration; thresholds were written first; the '
                  'tolerance of the tail-recomputation control was fixed after seeing its size in the smoke test (max score difference '
                  'reported in the record)',
}


def stream(purpose, key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(f'{RUN}|{SEED}|{purpose}|{key}'.encode()).digest()[:16], 'little'))


def save(name, body):
    from deploy.storage_policy import check_write
    p = OUT / name
    check_write(p, 1 << 20, operation='来源绑定请求上溯产物')
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(body, ensure_ascii=False, indent=1, default=lambda x: x.item() if isinstance(x, np.generic) else x), encoding='utf-8')
    t.replace(p)


def log(*a, **kw):
    RL.OUT = OUT
    RL.log(*a, **kw)


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
                    span = lambda a, b: [i for i, (lo, hi) in enumerate(enc['offset_mapping']) if hi > off + a and lo < off + b]
                    recs = {str(k): {key: span(a, b) for key, (a, b) in info.items()} for k, info in r['records'].items()}
                    choice = []
                    for val in r['values']:
                        suffix = tok.encode(prompt + val, add_special_tokens=False)
                        assert suffix[:-1] == ids
                        choice.append(suffix[-1])
                    r = {**r, 'records': recs, 'headers': {str(k): span(a, b) for k, (a, b) in r['headers'].items()}, 'prompt': prompt,
                         'tokens': ids, 'choice_ids': choice, 'gold_id': choice[r['target_record']]}
                    rows.append(QY.add_groups(tok, r))
    return rows


DEPS = (RL, UP, HD, QK, QY)


def prereg():
    from deploy.storage_policy import checked_dir
    from tools import provenance
    from tools.power_analysis import check_phase
    checked_dir(OUT, 1 << 30, operation='来源绑定请求上溯产物')
    assert not (OUT / 'prereg.json').exists()
    plan = D.checked(MODEL)
    rows = build_materials()
    assert len(rows) == FAMILIES * 8 and len({tuple(r['tokens']) for r in rows}) == len(rows)
    save(f'{SPLIT}-materials.json', rows)
    n = len(rows)
    body = {'schema': 'source_binding_request_prereg.v1', 'run_id': RUN, 'seed': SEED, 'created_at': datetime.now().astimezone().isoformat(),
            'base_plan_sha256': D.sha(SRC / 'plan.json'), 'query_report_sha256': D.sha(QY.OUT / 'report.json'),
            'model_snapshot': plan['models'][MODEL], 'code_sha256': D.sha(__file__),
            'dependency_sha256': {m.__name__: D.sha(m.__file__) for m in DEPS},
            'criteria': CRITERIA, 'materials': {'split': SPLIT, 'sha256': D.sha(OUT / f'{SPLIT}-materials.json'), 'families': FAMILIES, 'rows': n},
            'power': {'unit': 'paired binary outcome per item (McNemar); alpha = 0.05',
                      'W1_W2_W4': {'expected_delta': 0.30, 'p_d': 0.50, **check_phase(0.30, n, p_d=0.50)},
                      'W3': {'expected_delta': 0.10, 'p_d': 0.20, **check_phase(0.10, n, p_d=0.20)},
                      'caveat': 'items are clustered in 96 families and 4 frames'},
            'custody': 'L0 and the sets come from discovery (exposed). holdout_request is generated here, read first by the gate stage '
                       '(behaviour only) and then once by the run stage.',
            'resources': {'gpu_minutes_estimate': 20, 'disk_GiB_estimate': 0.2, 'models_concurrent': 1,
                          'shape': 'one chunk per panel, inner batch 128, arms recompute 11 positions; fixed before pre-registration'},
            'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    body['provenance']['model_revision'] = plan['models'][MODEL]
    save('prereg.json', body)
    print(json.dumps({'prereg_sha256': D.sha(OUT / 'prereg.json'), 'W': body['power']['W1_W2_W4'], 'W3': body['power']['W3']}), flush=True)


def checked():
    p = D.read(OUT / 'prereg.json')
    assert p['code_sha256'] == D.sha(__file__) and all(p['dependency_sha256'][m.__name__] == D.sha(m.__file__) for m in DEPS), \
        'source changed after pre-registration'
    assert p['materials']['sha256'] == D.sha(OUT / f'{SPLIT}-materials.json')
    D.checked(MODEL)
    return p


def engine(rows, arms=(), stage='run'):
    """arms: (name, start, kind, groups, iso); kind in none / all / only / block, applied to R in blocks [start, 24)."""
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    t, F = torch, torch.nn.functional
    assert snapshot(D.MODELS[MODEL]) == D.read(SRC / f'{MODEL}-identity.json')['model_snapshot']
    arms = list(arms)
    if arms:
        arms = [('noop', 0, 'none', [], False), ('ref_nat', FINAL_LAYER, 'none', [], False), ('ref_iso', FINAL_LAYER, 'none', [], True)] + arms
    assert all(0 <= a[1] <= END for a in arms)
    start, N = time.monotonic(), len(rows)
    results, mass = {}, {}
    with GpuLease(), t.inference_mode():
        rt = Runtime(MODEL)
        hd, kv, grp = rt.hd, rt.kv, rt.heads // rt.kv
        ids, pos, pe, mask = rt.context(rows)
        h, length = rt.embed(ids), ids.shape[1]
        R = length - R_FROM_END
        T = length - R
        log(stage, 0, NB, start, rows=N, length=length, arms=len(arms))
        frow = t.zeros((N, length), dtype=t.bool, device='cuda')
        frow[:, R] = True
        frow[:, -1] = True
        cor = t.zeros((N, length), dtype=t.bool, device='cuda')
        base_row = mask[:, 0, R]

        def row_for(kind, groups):
            row = base_row.clone()
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
            return row
        for i, r in enumerate(rows):
            assert len(r['tokens']) - R_FROM_END == r['up']['tail0'][-1] + 1
            cor[i, [length - len(r['tokens']) + j for j in r['up']['rec_correct']]] = True
        isorow = row_for('only', UNIONS['records4'])
        rrow = {name: row_for(kind, groups) for name, _, kind, groups, _ in arms if kind != 'none'}
        rot = lambda x, co, si: x * co + rt.mod.rotate_half(x) * si
        state, cap = {}, {}
        for b in range(NB):
            for name, l, _, _, _ in arms:
                if l == b:
                    state[name] = h[:, R:].clone()
            layer = rt.block(b)
            a = layer.self_attn
            handles = []
            if state:
                handles = [a.k_proj.register_forward_hook(lambda m, i, o: cap.__setitem__('k', o[:, :R])),
                           a.v_proj.register_forward_hook(lambda m, i, o: cap.__setitem__('v', o[:, :R]))]
            y = t.empty_like(h)
            for lo in range(0, N, BATCH):
                hi = min(lo + BATCH, N)
                B = hi - lo
                cos, sin = pe[0][lo:hi], pe[1][lo:hi]
                y[lo:hi] = layer(h[lo:hi], attention_mask=mask[lo:hi], position_ids=pos[lo:hi], position_embeddings=(cos, sin), use_cache=False)
                if not state:
                    continue
                kpre, vpre = cap['k'], cap['v']
                tm0 = mask[lo:hi][:, :, R:, :]
                for name, l, kind, _, iso in arms:
                    if name not in state:
                        continue
                    tm = tm0.clone()
                    if name != 'noop' and b >= FINAL_LAYER:
                        tm[:, 0, -1] = frow[lo:hi]
                    if kind != 'none' and l <= b < END:
                        tm[:, 0, 0] = rrow[name][lo:hi]
                    if iso and FINAL_LAYER <= b < ISO_END:
                        tm[:, 0, 0] = isorow[lo:hi]
                    x = state[name][lo:hi]
                    hn = layer.input_layernorm(x)
                    q = a.q_norm(a.q_proj(hn).view(B, T, NH, hd)).transpose(1, 2)
                    k = a.k_norm(t.cat([kpre, a.k_proj(hn)], 1).view(B, length, kv, hd)).transpose(1, 2)
                    v = t.cat([vpre, a.v_proj(hn)], 1).view(B, length, kv, hd).transpose(1, 2)
                    q, k = rot(q, cos[:, None, R:], sin[:, None, R:]), rot(k, cos[:, None], sin[:, None])
                    k, v = rt.mod.repeat_kv(k, grp), rt.mod.repeat_kv(v, grp)
                    if b == HEAD[0]:
                        w = ((q[:, HEAD[1], 0, None, :] * k[:, HEAD[1]]).sum(-1) * a.scaling).float().masked_fill(~tm[:, 0, 0], float('-inf'))
                        mass.setdefault(name, []).extend((w.softmax(-1) * cor[lo:hi]).sum(-1).cpu().tolist())
                    o = F.scaled_dot_product_attention(q, k, v, attn_mask=tm)
                    x = x + a.o_proj(o.transpose(1, 2).reshape(B, T, -1))
                    state[name][lo:hi] = x + layer.mlp(layer.post_attention_layernorm(x))
            h = y
            for x in handles:
                x.remove()
            cap.clear()
            del layer, y, a
            gc.collect()
            t.cuda.empty_cache()
            if (b + 1) % 4 == 0:
                log(stage, b + 1, NB, start, block=b, arms=len(state))
        norm = rt.Norm(rt.dim, eps=rt.cfg.rms_norm_eps).to(device='cuda', dtype=t.bfloat16)
        norm.weight = t.nn.Parameter(rt.weight('model.norm.weight'), requires_grad=False)
        W = rt.weight('lm_head.weight' if 'lm_head.weight' in rt.index else 'model.embed_tokens.weight')
        logits = lambda x: t.cat([(norm(x[lo:lo + BATCH]) @ W.T).float() for lo in range(0, N, BATCH)])
        results['base'] = rt.score(logits(h[:, -1]), rows)
        for key, s in state.items():
            results[key] = rt.score(logits(s[:, -1]), rows)
        peak = round(t.cuda.max_memory_allocated() / 2 ** 20)
    meta = {'peak_MiB': peak, 'seconds': round(time.monotonic() - start, 1), 'padded_length': length}
    if arms:
        meta['noop_same_choice'] = all(x['pred_record'] == y['pred_record'] for x, y in zip(results['noop'], results['base']))
        meta['noop_max_score_difference'] = max(abs(p - q) for x, y in zip(results['noop'], results['base']) for p, q in zip(x['scores'], y['scores']))
        meta['tail_control_ok'] = bool(meta['noop_same_choice'] and meta['noop_max_score_difference'] <= TOL)
    return results, mass, meta


def finish(name, rows, arms, smoke, extra):
    results, mass, meta = engine(rows, arms, stage=name)
    s = QY.summarise(results, mass, rows)
    keep = lambda k: s['arms'][k]['keep']['rate']
    ms = lambda k: s['arms'][k]['mass']['rate']
    body = extra(keep, ms)
    refs = {k: [keep(k), ms(k)] for k in ('ref_nat', 'ref_iso')}
    if smoke:
        print(json.dumps({'smoke': name, 'meta': meta, **refs, **body}), flush=True)
        return body
    assert meta['tail_control_ok']
    save(f'{name.replace("_", "-")}.json', {'schema': f'source_binding_request_{name}.v1', 'run_id': RUN, 'summary': s, **body, **meta})
    print(json.dumps({**refs, **body, 'noop_max_score_difference': meta['noop_max_score_difference']}), flush=True)
    return body


def pilot_a(smoke=False):
    rows = QY.discovery_half(True)[:32 if smoke else None]
    if not smoke:
        checked()
        assert not (OUT / 'pilot-a.json').exists()

    def extra(keep, ms):
        table = {str(l): [keep(f'iso:all@{l}'), ms(f'iso:all@{l}'), keep(f'nat:all@{l}'), ms(f'nat:all@{l}')] for l in TIMING}
        l0 = max([l for l in TIMING if table[str(l)][0] <= .40], default=None)
        return {'timing_iso_keep_mass_nat_keep_mass': table, 'L0': l0 if l0 is not None else 0, 'L0_rule_met': l0 is not None}
    arms = [(f'{c}:all@{l}', l, 'all', [], c == 'iso') for l in TIMING for c in ('iso', 'nat')]
    return finish('pilot_a', rows, arms, smoke, extra)


def pilot_b(smoke=False):
    rows = QY.discovery_half(True)[:32 if smoke else None]
    if smoke:
        l0 = pilot_a(True)['L0']
    else:
        checked()
        assert not (OUT / 'pilot-b.json').exists()
        l0 = D.read(OUT / 'pilot-a.json')['L0']

    def extra(keep, ms):
        order = sorted(CAND, key=lambda g: (-keep(f'only:{g}'), keep(f'block:{g}'), CAND.index(g)))
        return {'L0': l0, 'order': order, 'only_block_keep_mass': {g: [keep(f'only:{g}'), keep(f'block:{g}'), ms(f'only:{g}'), ms(f'block:{g}')]
                                                                 for g in order}}
    return finish('pilot_b', rows, [(f'{k}:{g}', l0, k, [g], True) for g in CAND for k in ('only', 'block')], smoke, extra)


def pilot_c(smoke=False):
    rows = QY.discovery_half(False)[:32 if smoke else None]
    if smoke:
        b = pilot_b(True)
    else:
        checked()
        assert not (OUT / 'pilot-c.json').exists()
        b = D.read(OUT / 'pilot-b.json')
    l0, order = b['L0'], b['order']
    ks = range(1, KMAX + 1)

    def extra(keep, ms):
        k = next((n for n in ks if keep(f'only:S{n}') >= .80), None)
        kn = next((n for n in ks if keep(f'block:S{n}') <= .50), None)
        return {'L0': l0, 'order': order, 'sufficient_set': order[:k or KMAX], 'sufficient_rule_met': k is not None,
                'necessary_set': order[:kn or KMAX], 'necessary_rule_met': kn is not None,
                'only_block_keep_mass': {f'S{n}': [keep(f'only:S{n}'), keep(f'block:S{n}'), ms(f'only:S{n}'), ms(f'block:S{n}')] for n in ks}}
    return finish('pilot_c', rows, [(f'{k}:S{n}', l0, k, order[:n], True) for n in ks for k in ('only', 'block')], smoke, extra)


def freeze():
    checked()
    assert not (OUT / 'frozen.json').exists()
    c = D.read(OUT / 'pilot-c.json')
    save('frozen.json', {'schema': 'source_binding_request_frozen.v1', 'run_id': RUN, **{k: c[k] for k in (
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
    save('gate.json', {'schema': 'source_binding_request_gate.v1', 'run_id': RUN, 'items_sha256': D.sha(OUT / 'gate-items.json'), **g, **meta})
    print(json.dumps({k: g[k] for k in ('passed', 'forced_accuracy', 'top1_accuracy')}), flush=True)


def run():
    from tools import provenance
    pre, fz = frozen()
    assert not (OUT / 'report.json').exists(), 'holdout already used'
    g = D.read(OUT / 'gate.json')
    rows = D.read(OUT / f'{SPLIT}-materials.json')
    S, N, l0 = fz['sufficient_set'], fz['necessary_set'], fz['L0']
    names = ('W1_sufficient', 'W2_necessary', 'W3_redundant_routes', 'W4_direct_route_compensates')
    if g['passed']:
        arms = [('iso:only:S', l0, 'only', S, True), ('iso:block:N', l0, 'block', N, True),
                ('nat:only:S', l0, 'only', S, False), ('nat:block:N', l0, 'block', N, False)]
        if len(N) > 1:
            arms += [(f'iso:block:N-{x}', l0, 'block', [y for y in N if y != x], True) for x in N]
        arms += [(f'{c}:all@{l}', l, 'all', [], c == 'iso') for l in TIMING for c in ('iso', 'nat')]
        arms += [(f'iso:{k}:{x}', l0, k, [x], True) for x in CAND for k in ('only', 'block')]
        arms += [(f'iso:{k}:{u}', l0, k, gs, True) for u, gs in UNIONS.items() for k in ('only', 'block')]
        results, mass, meta = engine(rows, arms, stage='run')
        save('items.json', {'results': results, 'mass': mass})
        base = D.read(OUT / 'gate-items.json')
        meta['base_replay_max_score_difference'] = max(abs(a - b) for x, y in zip(results['base'], base) for a, b in zip(x['scores'], y['scores']))
        s = QY.summarise(results, mass, rows, [(f'iso:block:N-{x}', 'iso:block:N') for x in N] if len(N) > 1 else [])
        keep = lambda k: s['arms'][k]['keep']['rate']
        tri = lambda ok, bad: 'supported' if ok else 'not_supported' if bad else 'inconclusive'
        ok = meta['tail_control_ok'] and keep('ref_nat') >= .90 and keep('ref_iso') >= .80 and keep(f'iso:all@{l0}') <= .40
        v = {'controls_valid': bool(ok)}
        if ok:
            v['W1_sufficient'] = tri(keep('iso:only:S') >= .70, keep('iso:only:S') < .50)
            v['W2_necessary'] = tri(keep('iso:block:N') <= .50, keep('iso:block:N') >= .75)
            ds = list(s['differences'].values())
            v['W3_redundant_routes'] = 'not_applicable' if len(N) == 1 else tri(
                all(d['rate'] >= .10 and d['ci95'][0] > 0 for d in ds), any(d['rate'] < .02 for d in ds))
            v['W4_direct_route_compensates'] = tri(keep('nat:block:N') >= .80, keep('nat:block:N') <= .50)
        else:
            v.update({k: 'controls_invalid' for k in names})
    else:
        s, meta, v = {'note': 'behaviour gate failed; no edit was run'}, {}, {k: 'behavior_ineligible' for k in names}
    report = {'schema': 'source_binding_request_report.v1', 'run_id': RUN, 'split': SPLIT, 'L0': l0, 'sufficient_set': S, 'necessary_set': N,
              'gate_passed': g['passed'], 'prereg_sha256': D.sha(OUT / 'prereg.json'), 'frozen_sha256': D.sha(OUT / 'frozen.json'),
              'gate_sha256': D.sha(OUT / 'gate.json'), 'items_sha256': D.sha(OUT / 'items.json') if g['passed'] else None,
              'criteria': pre['criteria'], 'verdicts': v, 'summary': s, **meta, 'formal_evidence_promotion': 0,
              'custody': 'holdout_request read once by the gate stage and once by this run',
              'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    save('report.json', report)
    path = provenance.write_result('SOURCE-BINDING-request-holdout', {'verdicts': v, 'gate_passed': g['passed'], 'L0': l0, 'sufficient_set': S,
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
