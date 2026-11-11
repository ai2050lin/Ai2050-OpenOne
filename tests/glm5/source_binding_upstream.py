"""One hop upstream: how does the answer reach the relay position?  Qwen3-14B BF16, source-binding task.

Follows Claude Phase 397: from block 24 on the final position can read the answer from a single token, the line break
after the user turn (relay position R).  Here the attention of R itself is edited, and the answer is read out through
the R route only: in every arm the final position may attend only to R from block 24 on (reference accuracy 1.000).
  all@l     from block l on, R attends to nothing but itself (when has the answer arrived at R?)
  only-G    from block L0 on, R attends only to the positions of G (and itself)
  block-G   from block L0 on, R cannot attend to the positions of G
All later positions are recomputed by the original blocks.
Stages: prereg -> pilot_a (timing) -> pilot_b (single groups) -> pilot_c (cumulative sets) -> freeze -> gate -> run.
"""
import argparse, gc, json, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_binding_design as D
import source_binding_relay as RL
from source_binding_runtime import Runtime

SRC, MODEL = D.OUT, 'qwen14'
OUT = D.ROOT / 'tests/glm5/result/source_binding_upstream_v1'
RUN, SEED, SPLIT = 'SOURCE-BINDING-UPSTREAM-20261110-v1', 202611100398, 'holdout_up'
FINAL_LAYER, TIMING, FAMILIES, CHUNK, NB, R_FROM_END = 24, [0, 4, 8, 12, 16, 20, 24, 28, 32, 36], 96, 64, 40, 10
GROUPS = ['rec_correct', 'rec_other_source', 'rec_other_entity', 'rec_neither', 'pre_other', 'qsrc', 'qent', 'qrest', 'instr', 'tail0']
CRITERIA = {
    'model': 'Qwen3-14B original BF16, CPU byte reads, CUDA layer streaming. GLM-4-9B is not run (failed the behaviour gate of this task).',
    'relay_position': 'R = the line-break token after <|im_end|> of the user turn (group tail1 of Phase 397), 10 tokens before the end',
    'readout': 'in every arm, from block 24 on the final position may attend only to R (and itself), so the answer has to pass through R. '
               'The reference arm has this restriction and no edit at R.',
    'groups_before_R': {'rec_correct': 'the record of the requested archive and entity', 'rec_other_source': 'same entity, other archive',
                        'rec_other_entity': 'same archive, other entity', 'rec_neither': 'other archive, other entity',
                        'pre_other': 'template head, introduction, archive headers', 'qsrc': 'archive name in the question',
                        'qent': 'entity name in the question', 'qrest': 'other tokens of the question', 'instr': 'the instruction sentence',
                        'tail0': '<|im_end|>', 'note': 'disjoint; they cover every position before R'},
    'edit': 'attention mask of R only, all heads, from the stated block on; every later position is recomputed by the original blocks',
    'outcome': '4-way forced choice among the four code words (chance 0.25); all items analysed; family-bootstrap 95 % intervals',
    'selection_on_discovery': {
        'L0': 'pilot_a (families 32-63): all@l for l in [0, 4, 8, 12, 16, 20, 24, 28, 32, 36]. L0 = the largest l with accuracy <= 0.40 '
              '(the answer has not yet arrived at R at block L0); if none, 0 (flagged)',
        'order': 'pilot_b (families 32-63), from block L0: groups ordered by only-g accuracy (descending; ties by block-g ascending)',
        'sets': 'pilot_c (families 0-31): cumulative sets S_k of the first k groups, k = 1..6. Sufficient set S = smallest S_k with '
                'only-S_k >= 0.90; necessary set N = smallest S_k with block-S_k <= 0.50 (if none, S_6, flagged)'},
    'material': 'holdout_up: 96 new families x 8 items; entity names that occur in no earlier panel; the 16 discovery code words, 4 seen frames',
    'two_pass': 'gate stage: unmodified forward and the v1 behaviour gate. Edits run only if the gate passed; otherwise behavior_ineligible.',
    'predictions_on_holdout': {
        'controls_valid': 'noop reproduces the base logits bit for bit; reference >= 0.90; all@L0 <= 0.40',
        'U1_sufficient': {'supported': 'only-S >= 0.80', 'not_supported': 'only-S < 0.60'},
        'U2_necessary': {'supported': 'block-N <= 0.50', 'not_supported': 'block-N >= 0.80'},
        'U3_redundant_routes': {'supported': 'for every group g of N (if N has more than one group), block-(N minus g) is at least 0.10 '
                                             'above block-N with family-bootstrap lower bound > 0',
                                'not_supported': 'for some g the gain is below 0.02', 'not_applicable': 'N has one group'}},
    'descriptive': 'all@l for every timing layer; only-g and block-g for every single group; top-1, margins, KL',
    'not_covered': 'which heads move the answer; routes that do not pass through R; other models, unseen answer words, unseen frames',
    'disclosure': 'the code was smoke-tested on 16 discovery items (exposed) before pre-registration; thresholds were written first. '
                  'The smoke test showed that the answer had not reached R at block 24, so the timing grid was extended to 28, 32, 36.',
}


def save(name, body):
    from deploy.storage_policy import check_write
    p = OUT / name
    check_write(p, 1 << 20, operation='来源绑定上溯产物')
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(body, ensure_ascii=False, indent=1, default=lambda x: x.item() if isinstance(x, np.generic) else x), encoding='utf-8')
    t.replace(p)


def log(*a, **kw):
    RL.OUT = OUT
    RL.log(*a, **kw)


def add_groups(tok, r):
    g = RL.group_positions(tok, r)
    s, e = r['source'], r['entity']
    role = {2 * s + e: 'rec_correct', 2 * (1 - s) + e: 'rec_other_source', 2 * s + (1 - e): 'rec_other_entity', 2 * (1 - s) + (1 - e): 'rec_neither'}
    up = {v: sorted(r['records'][str(k)]['span']) for k, v in role.items()}
    assert sorted(i for v in up.values() for i in v) == sorted(g['records'])
    for k in ('pre_other', 'qsrc', 'qent', 'qrest', 'instr', 'tail0'):
        up[k] = g[k]
    R = g['tail1'][0]
    assert R == len(r['tokens']) - R_FROM_END and sorted(i for v in up.values() for i in v) == list(range(R))
    return {**r, 'up': up}


def discovery_half(second):
    tok = RL.tokenizer()
    rows = D.read(SRC / f'{MODEL}-materials.json')['splits']['discovery']
    return [add_groups(tok, r) for r in rows if (int(r['family'][-4:]) >= 32) == second]


def build_materials():
    tok = RL.tokenizer()
    used = {n for rows in D.read(SRC / f'{MODEL}-materials.json')['splits'].values() for r in rows for n in r['names']}
    for p in (RL.ABL / 'holdout_abl-materials.json', RL.OUT / 'holdout_relay-materials.json'):
        used |= {n for r in D.read(p) for n in r['names']}
    rows = []
    for family in range(FAMILIES):
        rnd = np.random.default_rng(int.from_bytes(__import__('hashlib').sha256(f'{RUN}|{SEED}|names|{family}'.encode()).digest()[:16], 'little'))
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
                    rows.append(add_groups(tok, r))
    return rows


def prereg():
    from deploy.storage_policy import checked_dir
    from tools import provenance
    from tools.power_analysis import check_phase
    checked_dir(OUT, 1 << 30, operation='来源绑定上溯产物')
    assert not (OUT / 'prereg.json').exists()
    plan = D.checked(MODEL)
    rows = build_materials()
    assert len(rows) == FAMILIES * 8 and len({tuple(r['tokens']) for r in rows}) == len(rows)
    save(f'{SPLIT}-materials.json', rows)
    n = len(rows)
    body = {'schema': 'source_binding_upstream_prereg.v1', 'run_id': RUN, 'seed': SEED, 'created_at': datetime.now().astimezone().isoformat(),
            'base_plan_sha256': D.sha(SRC / 'plan.json'), 'relay_report_sha256': D.sha(RL.OUT / 'report.json') if (RL.OUT / 'report.json').exists() else None,
            'model_snapshot': plan['models'][MODEL], 'code_sha256': D.sha(__file__), 'relay_code_sha256': D.sha(RL.__file__),
            'criteria': CRITERIA, 'materials': {'split': SPLIT, 'sha256': D.sha(OUT / f'{SPLIT}-materials.json'), 'families': FAMILIES, 'rows': n},
            'power': {'unit': 'paired binary outcome per item (McNemar); alpha = 0.05',
                      'U1_U2': {'expected_delta': 0.40, 'p_d': 0.50, **check_phase(0.40, n, p_d=0.50)},
                      'U3': {'expected_delta': 0.10, 'p_d': 0.20, **check_phase(0.10, n, p_d=0.20)},
                      'caveat': 'items are clustered in 96 families and 4 frames; U3 is stated on family-bootstrap lower bounds'},
            'custody': 'L0 and the sets come from discovery (exposed). holdout_up is generated here, read first by the gate stage '
                       '(behaviour only) and then once by the run stage.',
            'resources': {'gpu_minutes_estimate': 60, 'disk_GiB_estimate': 0.2, 'models_concurrent': 1},
            'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    body['provenance']['model_revision'] = plan['models'][MODEL]
    save('prereg.json', body)
    print(json.dumps({'prereg_sha256': D.sha(OUT / 'prereg.json')}), flush=True)


def checked():
    p = D.read(OUT / 'prereg.json')
    assert p['code_sha256'] == D.sha(__file__) and p['relay_code_sha256'] == D.sha(RL.__file__), 'source changed after pre-registration'
    assert p['materials']['sha256'] == D.sha(OUT / f'{SPLIT}-materials.json')
    D.checked(MODEL)
    return p


def engine(rows, arms=(), stage='run'):
    """arms: (name, layer, kind, groups, restrict_final); kind in none / all / only / block.  Full-sequence state per arm."""
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
            R = length - R_FROM_END
            frow = t.zeros((n, length), dtype=t.bool, device='cuda')
            frow[:, R] = True
            frow[:, -1] = True
            rrow = {}
            for name, _, kind, groups, _ in arms:
                if kind == 'none':
                    continue
                row = mask[:, 0, R].clone()
                for i, r in enumerate(rs):
                    off = length - len(r['tokens'])
                    sel = [off + j for g in groups for j in r['up'][g]]
                    if kind == 'block':
                        row[i, sel] = False
                    else:
                        keep = row[i].clone()
                        row[i] = False
                        row[i, R] = True
                        if kind == 'only':
                            row[i, sel] = keep[sel]
                rrow[name] = row
            fork = {name: min(l, FINAL_LAYER) if final else l for name, l, kind, _, final in arms}
            state = {}
            for b in range(NB):
                for name in fork:
                    if fork[name] == b:
                        state[name] = h.clone()
                layer = rt.block(b)
                for lo in range(0, n, 8):
                    hi = min(lo + 8, n)
                    kw = {'position_ids': pos[lo:hi], 'position_embeddings': tuple(p[lo:hi] for p in pe), 'use_cache': False}
                    y = layer(h[lo:hi], attention_mask=mask[lo:hi], **kw)
                    for name, l, kind, _, final in arms:
                        if name not in state:
                            continue
                        m = mask[lo:hi].clone()
                        if kind != 'none' and b >= l:
                            m[:, 0, R] = rrow[name][lo:hi]
                        if final and b >= FINAL_LAYER:
                            m[:, 0, -1] = frow[lo:hi]
                        state[name][lo:hi] = layer(state[name][lo:hi], attention_mask=m, **kw)
                    h[lo:hi] = y
                del layer
                gc.collect()
                t.cuda.empty_cache()
                if (b + 1) % 10 == 0:
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
            del W, norm, h, state, lg0, lp0, ids, pos, pe, mask, rrow, frow
            gc.collect()
            t.cuda.empty_cache()
        peak = round(t.cuda.max_memory_allocated() / 2 ** 20)
    return results, {'noop_bitwise_equal': noop_equal, 'peak_MiB': peak, 'seconds': round(time.monotonic() - start, 1)}


def common():
    return [('noop', FINAL_LAYER, 'none', [], False), ('ref', FINAL_LAYER, 'none', [], True)]


def finish(name, rows, arms, smoke, extra):
    results, meta = engine(rows, arms, stage=name)
    s = RL.summarise(results, rows)
    acc = lambda k: s['arms'][k]['forced']['rate']
    body = extra(acc)
    if smoke:
        print(json.dumps({'smoke': name, 'meta': meta, 'ref': acc('ref'), **body}), flush=True)
        return body
    assert all(meta['noop_bitwise_equal'])
    save(f'{name.replace("_", "-")}.json', {'schema': f'source_binding_upstream_{name}.v1', 'run_id': RUN, 'summary': s, **body, **meta})
    print(json.dumps({'ref': acc('ref'), **body}), flush=True)
    return body


def pilot_a(smoke=False):
    rows = discovery_half(True)[:16 if smoke else None]
    if not smoke:
        checked()
        assert not (OUT / 'pilot-a.json').exists()

    def extra(acc):
        table = {str(l): acc(f'all@{l}') for l in TIMING}
        l0 = max([l for l in TIMING if table[str(l)] <= .40], default=None)
        return {'timing': table, 'L0': l0 if l0 is not None else 0, 'L0_rule_met': l0 is not None}
    return finish('pilot_a', rows, common() + [(f'all@{l}', l, 'all', [], True) for l in TIMING], smoke, extra)


def pilot_b(smoke=False):
    rows = discovery_half(True)[:16 if smoke else None]
    if smoke:
        l0 = pilot_a(True)['L0']
    else:
        checked()
        assert not (OUT / 'pilot-b.json').exists()
        l0 = D.read(OUT / 'pilot-a.json')['L0']

    def extra(acc):
        order = sorted(GROUPS, key=lambda g: (-acc(f'only:{g}'), acc(f'block:{g}'), GROUPS.index(g)))
        return {'L0': l0, 'order': order, 'only_block': {g: [acc(f'only:{g}'), acc(f'block:{g}')] for g in order}}
    return finish('pilot_b', rows, common() + [(f'{k}:{g}', l0, k, [g], True) for g in GROUPS for k in ('only', 'block')], smoke, extra)


def pilot_c(smoke=False):
    rows = discovery_half(False)[:16 if smoke else None]
    if smoke:
        b = pilot_b(True)
    else:
        checked()
        assert not (OUT / 'pilot-c.json').exists()
        b = D.read(OUT / 'pilot-b.json')
    l0, order = b['L0'], b['order']

    def extra(acc):
        k = next((n for n in range(1, 7) if acc(f'only:S{n}') >= .90), None)
        kn = next((n for n in range(1, 7) if acc(f'block:S{n}') <= .50), None)
        return {'L0': l0, 'order': order, 'sufficient_set': order[:k or 6], 'sufficient_rule_met': k is not None,
                'necessary_set': order[:kn or 6], 'necessary_rule_met': kn is not None,
                'only_block': {f'S{n}': [acc(f'only:S{n}'), acc(f'block:S{n}')] for n in range(1, 7)}}
    return finish('pilot_c', rows, common() + [(f'{k}:S{n}', l0, k, order[:n], True) for n in range(1, 7) for k in ('only', 'block')], smoke, extra)


def freeze():
    checked()
    assert not (OUT / 'frozen.json').exists()
    c = D.read(OUT / 'pilot-c.json')
    save('frozen.json', {'schema': 'source_binding_upstream_frozen.v1', 'run_id': RUN, **{k: c[k] for k in (
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
    results, meta = engine(rows, stage='gate')
    save('gate-items.json', results['base'])
    g = D.behavior(rows, results['base'])
    save('gate.json', {'schema': 'source_binding_upstream_gate.v1', 'run_id': RUN, 'items_sha256': D.sha(OUT / 'gate-items.json'), **g, **meta})
    print(json.dumps({k: g[k] for k in ('passed', 'forced_accuracy', 'top1_accuracy')}), flush=True)


def run():
    from tools import provenance
    pre, fz = frozen()
    assert not (OUT / 'report.json').exists(), 'holdout already used'
    g = D.read(OUT / 'gate.json')
    rows = D.read(OUT / f'{SPLIT}-materials.json')
    S, N, l0 = fz['sufficient_set'], fz['necessary_set'], fz['L0']
    names = ('U1_sufficient', 'U2_necessary', 'U3_redundant_routes')
    if g['passed']:
        arms = common() + [('only:S', l0, 'only', S, True), ('block:N', l0, 'block', N, True)]
        if len(N) > 1:
            arms += [(f'block:N-{x}', l0, 'block', [y for y in N if y != x], True) for x in N]
        arms += [(f'all@{l}', l, 'all', [], True) for l in TIMING]
        arms += [(f'{k}:{x}', l0, k, [x], True) for x in GROUPS for k in ('only', 'block')]
        results, meta = engine(rows, arms, stage='run')
        save('items.json', results)
        base = D.read(OUT / 'gate-items.json')
        meta['base_replay_max_score_difference'] = max(abs(a - b) for x, y in zip(results['base'], base) for a, b in zip(x['scores'], y['scores']))
        s = RL.summarise(results, rows, [(f'block:N-{x}', 'block:N') for x in N] if len(N) > 1 else [])
        acc = lambda k: s['arms'][k]['forced']['rate']
        tri = lambda ok, bad: 'supported' if ok else 'not_supported' if bad else 'inconclusive'
        controls = all(meta['noop_bitwise_equal']) and acc('ref') >= .90 and acc(f'all@{l0}') <= .40
        v = {'controls_valid': bool(controls)}
        if controls:
            v['U1_sufficient'] = tri(acc('only:S') >= .80, acc('only:S') < .60)
            v['U2_necessary'] = tri(acc('block:N') <= .50, acc('block:N') >= .80)
            ds = list(s['differences'].values())
            v['U3_redundant_routes'] = 'not_applicable' if len(N) == 1 else tri(
                all(d['rate'] >= .10 and d['ci95'][0] > 0 for d in ds), any(d['rate'] < .02 for d in ds))
        else:
            v.update({k: 'controls_invalid' for k in names})
    else:
        s, meta, v = {'note': 'behaviour gate failed; no edit was run'}, {}, {k: 'behavior_ineligible' for k in names}
    report = {'schema': 'source_binding_upstream_report.v1', 'run_id': RUN, 'split': SPLIT, 'L0': l0, 'sufficient_set': S, 'necessary_set': N,
              'gate_passed': g['passed'], 'prereg_sha256': D.sha(OUT / 'prereg.json'), 'frozen_sha256': D.sha(OUT / 'frozen.json'),
              'gate_sha256': D.sha(OUT / 'gate.json'), 'items_sha256': D.sha(OUT / 'items.json') if g['passed'] else None,
              'criteria': pre['criteria'], 'verdicts': v, 'summary': s, **meta, 'formal_evidence_promotion': 0,
              'custody': 'holdout_up read once by the gate stage and once by this run',
              'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    save('report.json', report)
    path = provenance.write_result('SOURCE-BINDING-upstream-holdout', {'verdicts': v, 'gate_passed': g['passed'], 'L0': l0, 'sufficient_set': S,
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
