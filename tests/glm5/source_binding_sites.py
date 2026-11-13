"""The narrow-edit rule of graph revision 4 at the other sites of the transition (Claude Phase 423).  Qwen3-14B BF16.

Phase 422: for both name tokens exchanged at block 24, the answer of a narrow edit through edges N1 and N7 agrees with
the broad edit in 0.831.  Here the same kind of rule is tested for the archive name alone (qsrc), the entity name alone
(qent) and both (names), exchanged at the input of block 23, 24 or 25 - the nine cells for which revision 3 has a
class rule and a follow-rate table only.

Broad edit: residual states of the site tokens replaced by those of a donor at the input of block L.
Narrow edit: the answer position only; in chosen blocks its attention row reads donor keys / values (unedited donor
run) at the site tokens (and, for the block-25 heads, at the holders).  The rule predicts that the broad edit gives the
answer of the narrow edit.
Stages: smoke, pilot (opened panel holdout_fid4) -> prereg -> gate -> run.
"""
import argparse, gc, json, sys, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_binding_design as D
import source_binding_relay as RL
import source_binding_pointer as PT
import source_binding_nameloc as NL
import source_binding_arbitration as AR
import source_binding_fidelity3 as F3
import source_binding_h2518 as H25
import source_binding_fidelity4 as F4
from source_binding_runtime import Runtime

SRC, MODEL = D.OUT, 'qwen14'
OUT = D.ROOT / 'tests/glm5/result/source_binding_sites_v1'
F4_OUT = F4.OUT
RUN, SEED = 'SOURCE-BINDING-SITES-20261112-v1', 202611120423
SPLIT, FAMILIES = 'holdout_sites', 64
CHUNK, NB, NH, TOL, REPS = 192, 40, 40, 1.0, 4
C = H25.C
SITES, LAYERS = ('qsrc', 'qent', 'names'), (23, 24, 25)
CELLS = [f'{s}@{l}' for s in SITES for l in LAYERS]
OTHER_PANELS = F4.OTHER_PANELS + ['source_binding_fidelity_v4/holdout_fid4']
GROUPS = {'qsrc': lambda r: sorted(r['up']['qsrc']), 'qent': lambda r: sorted(r['up']['qent']), 'names': AR.names_of, 'holders': AR.holders_of}


def save(name, body):
    from deploy.storage_policy import check_write
    p = OUT / name
    check_write(p, 1 << 20, operation='来源绑定其余位点产物')
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(body, ensure_ascii=False, indent=1, default=lambda x: x.item() if isinstance(x, np.generic) else x), encoding='utf-8')
    t.replace(p)


def log(*a, **kw):
    RL.OUT = OUT
    RL.log(*a, **kw)


def build_materials():
    taken = {n for rows in D.read(SRC / f'{MODEL}-materials.json')['splits'].values() for r in rows for n in r['names']}
    for p in PT.PANELS + [PT.OUT / 'pilot_layout-materials.json', PT.OUT / 'holdout_layout-materials.json',
                          NL.OUT / 'pilot_name-materials.json',
                          NL.OUT / 'holdout_name-materials.json'] + [D.ROOT / 'tests/glm5/result' / f'{x}-materials.json' for x in OTHER_PANELS]:
        taken |= {n for r in D.read(p) for n in r['names']}
    PT.SPLITS[SPLIT] = FAMILIES
    return PT.build_materials(SPLIT, taken)


def variants(site, l):
    """name -> {block: [[groups, heads or None], ...]} : what the answer row reads from the donor."""
    v = {'a': {25: [[[site], C]], 27: [[[site], None]]},
         'b': {25: [[[site, 'holders'], C]], 27: [[[site], None]]},
         'c': {b: [[[site], None]] for b in range(l, 28)},
         'e': {b: [[[site], None]] for b in range(l, NB)}}
    v['d'] = {**v['c'], 25: [[[site], None], [['holders'], C]]}
    return v


def make_arms(rows, rule, run=RUN, seed=SEED):
    """rule(site, l) -> {variant name: spec}."""
    H25.RUN, H25.SEED = run, seed                 # donor draws: the stream of the Phase 415 module contains run id and seed
    dons, rels = H25.draws(rows)
    me = list(range(len(rows)))
    arms = {'self': {'donor': me, 'don': {b: [[['names', 'holders'], None]] for b in range(23, 28)}}}
    for k, don in enumerate(dons):
        for site in SITES:
            for l in LAYERS:
                arms[f'swap|{site}@{l}#{k}'] = {'swap': [site, l], 'donor': don, 'rel': rels[k]}
                for name, spec in rule(site, l).items():
                    arms[f'{name}|{site}@{l}#{k}'] = {'don': spec, 'donor': don, 'rel': rels[k]}
    return arms


def engine(rows, arms=None, stage='run'):
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    t, F = torch, torch.nn.functional
    assert snapshot(D.MODELS[MODEL]) == D.read(SRC / f'{MODEL}-identity.json')['model_snapshot']
    start, total = time.monotonic(), len(rows)
    results = {}
    with GpuLease(), t.inference_mode():
        rt = Runtime(MODEL)
        hd, kv, grp = rt.hd, rt.kv, rt.heads // rt.kv
        rot = lambda x, co, si: x * co + rt.mod.rotate_half(x) * si
        log(stage, 0, total * NB, start, rows=total)
        for c0 in range(0, total, CHUNK):
            rs = rows[c0:c0 + CHUNK]
            N = len(rs)
            ids, pos, pe, mask = rt.context(rs)
            h, length = rt.embed(ids), ids.shape[1]
            off = [length - len(r['tokens']) for r in rs]
            cos, sin = pe
            plan = {}
            if arms:
                P0 = min(off[i] + AR.names_of(r)[0] for i, r in enumerate(rs))
                plan['noop'] = {'first': 0, 'P': length - 1, 'sw': None, 'don': {}}
                cache = {}
                for name, a in arms.items():
                    local = [j - c0 for j in a['donor'][c0:c0 + N]]
                    assert all(0 <= j < N and rs[j]['family'] == r['family'] for j, r in zip(local, rs))

                    def pairs(g):
                        key = (g, tuple(local))
                        if key not in cache:
                            fn = GROUPS[g]
                            cache[key] = tuple(t.tensor(v, device='cuda') for v in (
                                [i for i, r in enumerate(rs) for _ in fn(r)], [off[i] + x for i, r in enumerate(rs) for x in fn(r)],
                                [j for j in local for _ in fn(rs[j])], [off[j] + y for j in local for y in fn(rs[j])]))
                        return cache[key]
                    if a.get('swap'):
                        site, l = a['swap']
                        plan[name] = {'first': l, 'P': P0, 'sw': l, 'idx': pairs(site), 'don': {}}
                    else:
                        don = {int(b): [(pairs(g), None if hs is None else t.tensor(hs, device='cuda')) for gs, hs in v for g in gs]
                               for b, v in a['don'].items()}
                        plan[name] = {'first': min(don), 'P': length - 1, 'sw': None, 'don': don}
            state, cap = {}, {}
            for b in range(NB):
                for name, p in plan.items():
                    if p['first'] == b:
                        state[name] = h[:, p['P']:].clone()
                    if p['sw'] == b:
                        i0, i1, i2, i3 = p['idx']
                        state[name][i0, i1 - p['P']] = h[i2, i3]
                layer = rt.block(b)
                a = layer.self_attn
                handles = []
                if state:
                    handles = [a.k_proj.register_forward_hook(lambda m, i, o: cap.__setitem__('k', o)),
                               a.v_proj.register_forward_hook(lambda m, i, o: cap.__setitem__('v', o))]
                y = layer(h, attention_mask=mask, position_ids=pos, position_embeddings=(cos, sin), use_cache=False)
                kb, vb = cap.get('k'), cap.get('v')
                kd = vd = None
                for name in state:
                    p = plan[name]
                    P = p['P']
                    T = length - P
                    x = state[name]
                    hn = layer.input_layernorm(x)
                    q = a.q_norm(a.q_proj(hn).view(N, T, NH, hd)).transpose(1, 2)
                    k = a.k_norm(t.cat([kb[:, :P], a.k_proj(hn)], 1).view(N, length, kv, hd)).transpose(1, 2)
                    v = t.cat([vb[:, :P], a.v_proj(hn)], 1).view(N, length, kv, hd).transpose(1, 2)
                    k, v = rt.mod.repeat_kv(k, grp), rt.mod.repeat_kv(v, grp)
                    q = rot(q, cos[:, None, P:], sin[:, None, P:])
                    if b in p['don']:               # answer row (T = 1): donor keys / values of its unedited run
                        if kd is None:
                            kd = rt.mod.repeat_kv(a.k_norm(kb.view(N, length, kv, hd)).transpose(1, 2), grp)
                            vd = rt.mod.repeat_kv(vb.view(N, length, kv, hd).transpose(1, 2), grp)
                        for (bi, bp, di, dp), hs in p['don'][b]:
                            if hs is None:
                                k[bi, :, bp] = kd[di, :, dp]
                                v[bi, :, bp] = vd[di, :, dp]
                            else:
                                k[bi[:, None], hs[None], bp[:, None]] = kd[di[:, None], hs[None], dp[:, None]]
                                v[bi[:, None], hs[None], bp[:, None]] = vd[di[:, None], hs[None], dp[:, None]]
                    o = F.scaled_dot_product_attention(q, rot(k, cos[:, None], sin[:, None]), v, attn_mask=mask[:, :, P:, :])
                    x = x + a.o_proj(o.transpose(1, 2).reshape(N, T, -1))
                    state[name] = x + layer.mlp(layer.post_attention_layernorm(x))
                    del q, k, v, o
                h = y
                for x in handles:
                    x.remove()
                cap.clear()
                del layer, y, a, kb, vb, kd, vd
                gc.collect()
                t.cuda.empty_cache()
                if (b + 1) % 4 == 0:
                    free, tot = t.cuda.mem_get_info()
                    log(stage, c0 * NB + (b + 1) * N, total * NB, start, block=b, arms=len(state), device_used_MiB=round((tot - free) / 2 ** 20))
            norm = rt.Norm(rt.dim, eps=rt.cfg.rms_norm_eps).to(device='cuda', dtype=t.bfloat16)
            norm.weight = t.nn.Parameter(rt.weight('model.norm.weight'), requires_grad=False)
            W = rt.weight('lm_head.weight' if 'lm_head.weight' in rt.index else 'model.embed_tokens.weight')
            logits = lambda x: (norm(x) @ W.T).float()
            results.setdefault('base', []).extend(rt.score(logits(h[:, -1]), rs))
            for key in plan:
                results.setdefault(key, []).extend(rt.score(logits(state[key][:, -1]), rs))
            del W, norm, h, state, ids, pos, pe, mask, plan
            gc.collect()
            t.cuda.empty_cache()
        peak = round(t.cuda.max_memory_allocated() / 2 ** 20)
    meta = {'peak_MiB': peak, 'seconds': round(time.monotonic() - start, 1)}
    if arms:
        diff = lambda k: max(abs(p - q) for x, y in zip(results[k], results['base']) for p, q in zip(x['scores'], y['scores']))
        same = lambda k: all(x['pred_record'] == y['pred_record'] for x, y in zip(results[k], results['base']))
        ctl = ('noop', 'self')
        meta.update({f'{k}_max_score_difference': diff(k) for k in ctl})
        meta['controls_ok'] = bool(all(same(k) and diff(k) <= TOL for k in ctl))
    return results, meta


def slim(results):
    return {k: [{'id': x['id'], 'pred_record': x['pred_record'], 'scores': x['scores']} for x in v] for k, v in results.items()}


def table(rows, arms, items):
    """One row per broad edit: family, cell, B, P, I, the model's choice, the answer of every narrow variant of that cell."""
    fam_id = {f: i for i, f in enumerate(sorted({r['family'] for r in rows}))}
    names = sorted({n.split('|')[0] for n in arms if '|' in n and not n.startswith('swap')})
    T = defaultdict(list)
    for cell in CELLS:
        site, l = cell.split('@')
        for k in range(REPS):
            a = arms[f'swap|{cell}#{k}']
            for i, (r, j) in enumerate(zip(rows, a['donor'])):
                B, P, I = F3.records(r, rows[j], site, int(l))
                T['fam'].append(fam_id[r['family']]); T['cell'].append(cell); T['B'].append(B); T['P'].append(P); T['I'].append(I)
                T['out'].append(items[f'swap|{cell}#{k}'][i]['pred_record'])
                T['v3'].append(F3.predict(r, rows[j], site, int(l), 'graph_v3'))
                for v in names:
                    key = f'{v}|{cell}#{k}'
                    T[v].append(items[key][i]['pred_record'] if key in items else -1)
    return {k: np.array(v) for k, v in T.items()}, len(fam_id), names


def smoke():
    from deploy.storage_policy import checked_dir
    checked_dir(OUT, 1 << 30, operation='来源绑定其余位点产物')
    rows = [r for r in D.read(NL.OUT / 'pilot_name-materials.json') if int(r['family'][-4:]) < 2]
    arms = make_arms(rows, variants)
    results, meta = engine(rows, arms, stage='smoke')
    T, nf, names = table(rows, arms, results)
    print(json.dumps({'meta': meta, 'arms': len(arms), 'agreement': {v: round(float((T[v] == T['out']).mean()), 3) for v in names + ['v3', 'B']}}), flush=True)


def pilot():
    """Opened panel holdout_fid4 with the donor draws of Phase 422: five narrow variants in each of the nine cells."""
    from deploy.storage_policy import checked_dir
    checked_dir(OUT, 1 << 30, operation='来源绑定其余位点产物')
    rows = D.read(F4_OUT / f'{F4.SPLIT}-materials.json')
    arms = make_arms(rows, variants, F4.RUN, F4.SEED)
    results, meta = engine(rows, arms, stage='pilot')
    old = D.read(F4_OUT / 'items.json')
    meta['replay_of_phase_422_swap_same_choice'] = float(np.mean([x['pred_record'] == y['pred_record'] for k in range(REPS)
                                                                  for x, y in zip(results[f'swap|names@24#{k}'], old[f'swap#{k}'])]))
    save('pilot-design.json', arms)
    save('pilot-items.json', slim(results))
    save('pilot-meta.json', meta)
    print(json.dumps(meta), flush=True)



# ---- frozen after the pilot on the opened panel holdout_fid4 ----
PRIMARY = 'c'
RULES = {'c': 'graph_v4_sites', 'a': 'two_edges', 'b': 'rule_422', 'd': 'c_plus_holders', 'e': 'to_last_block'}
THRESHOLDS = {
    'X1_agreement': 'rule c over all nine cells >= 0.90 and in every cell >= 0.84. not_supported if the overall agreement < 0.85.',
    'X2_beyond_revision_3': 'rule c minus the revision-3 rule: overall >= 0.06 with a lower bound > 0, and a lower bound > 0 at each of '
                            'the blocks 23, 24, 25. not_supported if the overall lower bound <= 0.',
    'X3_reading_ends_by_block_27': 'rule e (donor keys / values up to the last block) differs from rule c by <= 0.01 overall. '
                                   'not_supported if the difference > 0.03.',
    'X4_blocks_25_and_27_alone_do_not_suffice_at_23': 'at block 23 rule c minus rule a >= 0.25 with a lower bound > 0. not_supported '
                                                      'if the lower bound <= 0.',
}
CRITERIA = {
    'model': 'Qwen3-14B original BF16, CPU byte reads, CUDA layer streaming. GLM-4-9B is not run (failed the behaviour gate of this task).',
    'material': 'layout-factorial families as in Phase 405. holdout_sites: 64 families (1,024 prompts). Entity names occur in no '
                'earlier panel of this line; the 16 discovery code words, 4 seen frames.',
    'broad_edit': 'at the input of block L (23, 24, 25) the residual states of the site tokens (archive name qsrc, entity name qent, '
                  'or both) are replaced by those of a donor of the same family; every position and every later block sees it. '
                  '4 seeded donor relations per item, the same in every cell: 9 cells x 4,096 edits.',
    'narrow_rules': {'c (primary)': 'the answer position only: in blocks L..27 its attention row reads donor keys / values (unedited '
                                    'donor run) at the site tokens, every head. No fitted number',
                     'a': 'block 25, heads 25.18, 25.24, 25.15 and block 27, every head (the two edges N1, N7 of revision 4)',
                     'b': 'as a, the block-25 heads also read the donor holders (the Phase 422 rule)',
                     'd': 'as c, the block-25 heads 25.18, 25.24, 25.15 also read the donor holders',
                     'e': 'as c up to the last block (L..39)'},
    'prediction': 'the broad edit gives the answer of the narrow edit. Comparisons: revision-3 rule (block 23: the pointer record; '
                  'block 24: the pointer record iff the site is not qsrc and P = I; block 25: no effect), no effect (B), position (P).',
    'outcome': 'agreement = the predicted record equals the forced choice of the broad edit; per cell, per block, overall, and on '
               'conflict edits (P != B); family-bootstrap 95 % intervals (1000 draws).',
    'controls': 'tail recomputation equals the base forward (same choice, score difference <= 1.0); the narrow edit with the prompt '
                'itself as donor (names and holders, blocks 23-27, every head) changes nothing (same bound).',
    'two_pass': 'gate stage: unmodified forward and the v1 behaviour gate. Edits run only if the gate passed.',
    'predictions_on_holdout': THRESHOLDS,
    'not_covered': 'which heads of blocks 23-27 do the reading besides those already on the graph (rule c lets every head read); the '
                   'routes through other positions that make up the remaining disagreement; name tokens at other blocks (0.981 in '
                   'Phase 416); other tasks and models. The rule is an intervention followed by a forward pass, not a computation '
                   'on the graph alone.',
    'disclosure': 'one pilot on the opened panel holdout_fid4 (donor draws of Phase 422) with rules a-e in all nine cells; rule c was '
                  'chosen there (overall 0.927; a 0.788, b 0.809, d 0.925, e 0.928; revision 3 0.822). In one cell (qsrc@23) rule c '
                  'was below revision 3 (0.900 against 0.921). Thresholds were set on these numbers. Smoke test on 2 exposed '
                  'families of pilot_name.',
}


def summarise(results, arms, rows):
    T, nf, names = table(rows, arms, results)
    draws = np.random.default_rng(SEED).integers(0, nf, (1000, nf))
    fam, out = T['fam'], T['out']
    r4 = lambda v: round(float(v), 4)

    def rate(v, keep):
        v = np.asarray(v, float)
        per, cnt = np.bincount(fam[keep], weights=v[keep], minlength=nf), np.bincount(fam[keep], minlength=nf)
        bs = per[draws].sum(1) / np.maximum(cnt[draws].sum(1), 1)
        return {'rate': r4(v[keep].mean()), 'ci95': [r4(np.quantile(bs, .025)), r4(np.quantile(bs, .975))]}
    hit = {**{RULES[v]: (T[v] == out).astype(float) for v in names}, 'graph_v3': (T['v3'] == out).astype(float),
           'none': (T['B'] == out).astype(float), 'position': (T['P'] == out).astype(float)}
    g = hit[RULES[PRIMARY]]
    layer = np.array([int(c.split('@')[1]) for c in T['cell']])
    conflict = T['P'] != T['B']
    scopes = {'all': np.ones(len(out), bool), **{f'block_{l}': layer == l for l in LAYERS}, **{c: T['cell'] == c for c in CELLS}}
    s = {'edits': int(len(out)), 'scopes': {}}
    for name, keep in scopes.items():
        s['scopes'][name] = {'n': int(keep.sum()), 'conflict_edits': int((keep & conflict).sum()), 'model_choice_changed': r4((out != T['B'])[keep].mean()),
                             'agreement': {k: rate(v, keep) for k, v in hit.items()},
                             'agreement_on_conflict_edits': {k: r4(v[keep & conflict].mean()) for k, v in hit.items()},
                             **{f'primary_minus_{k}': rate(g - hit[k], keep) for k in ('graph_v3', 'none', 'two_edges', 'to_last_block', 'c_plus_holders')}}
    s['min_cell_agreement'] = min(s['scopes'][c]['agreement'][RULES[PRIMARY]]['rate'] for c in CELLS)
    s['third_record_share'] = r4(((out != T['B']) & (out != T['P'])).mean())
    return s


def verdicts(s, meta):
    tri = lambda ok, bad: 'supported' if ok else 'not_supported' if bad else 'inconclusive'
    v = {'controls_valid': bool(meta['controls_ok'])}
    if not v['controls_valid']:
        return {**v, **{k: 'controls_invalid' for k in THRESHOLDS}}
    S, P = s['scopes'], RULES[PRIMARY]
    a = S['all']['agreement'][P]['rate']
    v['X1_agreement'] = tri(a >= .90 and s['min_cell_agreement'] >= .84, a < .85)
    d = S['all']['primary_minus_graph_v3']
    v['X2_beyond_revision_3'] = tri(d['rate'] >= .06 and d['ci95'][0] > 0 and all(S[f'block_{l}']['primary_minus_graph_v3']['ci95'][0] > 0 for l in LAYERS),
                                    d['ci95'][0] <= 0)
    e = abs(S['all']['primary_minus_to_last_block']['rate'])
    v['X3_reading_ends_by_block_27'] = tri(e <= .01, e > .03)
    d = S['block_23']['primary_minus_two_edges']
    v['X4_blocks_25_and_27_alone_do_not_suffice_at_23'] = tri(d['rate'] >= .25 and d['ci95'][0] > 0, d['ci95'][0] <= 0)
    return v


DEPS = (PT, NL, AR, F3, H25, F4)


def prereg():
    from deploy.storage_policy import checked_dir
    from tools import provenance
    from tools.power_analysis import check_phase
    checked_dir(OUT, 1 << 30, operation='来源绑定其余位点产物')
    assert not (OUT / 'prereg.json').exists()
    plan = D.checked(MODEL)
    rows = build_materials()
    assert len(rows) == FAMILIES * 16 and len({tuple(r['tokens']) for r in rows}) == len(rows)
    arms = make_arms(rows, variants)
    save(f'{SPLIT}-materials.json', rows)
    save('design.json', arms)
    body = {'schema': 'source_binding_sites_prereg.v1', 'run_id': RUN, 'seed': SEED, 'created_at': datetime.now().astimezone().isoformat(),
            'base_plan_sha256': D.sha(SRC / 'plan.json'), 'model_snapshot': plan['models'][MODEL], 'code_sha256': D.sha(__file__),
            'dependency_sha256': {m.__name__: D.sha(m.__file__) for m in DEPS},
            'criteria': CRITERIA, 'frozen': {'primary': PRIMARY, 'pilot_summary_sha256': D.sha(OUT / 'pilot-summary.json')},
            'materials': {'sha256': D.sha(OUT / f'{SPLIT}-materials.json'), 'families': FAMILIES, 'rows': len(rows)},
            'design': {'sha256': D.sha(OUT / 'design.json'), 'arms': len(arms)},
            'power': {'unit': 'paired binary outcome per item (prediction right / wrong under rule c and under revision 3); alpha = 0.05',
                      'X2_as_if_one_edit_per_item': {'expected_delta': 0.08, 'p_d': 0.20, **check_phase(0.08, len(rows), p_d=0.20)},
                      'caveat': '36,864 edits clustered in 1,024 items and 64 families; intervals are family bootstraps'},
            'custody': 'holdout_sites is read first by the gate stage (behaviour only) and then once by the run stage.',
            'resources': {'gpu_minutes_estimate': 11, 'disk_GiB_estimate': 0.2, 'models_concurrent': 1,
                          'shape': 'chunks of 192 prompts in one batch; pilot peak 5.9 GiB, 9.6 minutes'},
            'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    body['provenance']['model_revision'] = plan['models'][MODEL]
    save('prereg.json', body)
    print(json.dumps({'prereg_sha256': D.sha(OUT / 'prereg.json'), 'design': body['design'], 'power': body['power']}), flush=True)


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
    save('gate.json', {'schema': 'source_binding_sites_gate.v1', 'run_id': RUN, 'items_sha256': D.sha(OUT / 'gate-items.json'), **g, **meta})
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
    report = {'schema': 'source_binding_sites_report.v1', 'run_id': RUN, 'split': SPLIT, 'gate_passed': g['passed'],
              'prereg_sha256': D.sha(OUT / 'prereg.json'), 'gate_sha256': D.sha(OUT / 'gate.json'),
              'items_sha256': D.sha(OUT / 'items.json') if g['passed'] else None, 'criteria': pre['criteria'], 'frozen': pre['frozen'],
              'verdicts': v, 'summary': s, **meta, 'formal_evidence_promotion': 0,
              'custody': 'holdout_sites read once by the gate stage and once by this run',
              'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    save('report.json', report)
    path = provenance.write_result('SOURCE-BINDING-sites-holdout', {'verdicts': v, 'gate_passed': g['passed'],
                                                                  'report_sha256': D.sha(OUT / 'report.json')},
                                   model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)
    print(json.dumps({'verdicts': v, 'result': str(path)}, ensure_ascii=False), flush=True)


def pilot_summary():
    rows = D.read(F4_OUT / f'{F4.SPLIT}-materials.json')
    s = summarise(D.read(OUT / 'pilot-items.json'), D.read(OUT / 'pilot-design.json'), rows)
    save('pilot-summary.json', s)
    print(json.dumps({'verdicts_on_pilot': verdicts(s, {'controls_ok': True}), 'min_cell': s['min_cell_agreement'],
                      'all': {k: v['rate'] for k, v in s['scopes']['all']['agreement'].items()},
                      'minus_v3_by_block': {l: s['scopes'][f'block_{l}']['primary_minus_graph_v3'] for l in LAYERS},
                      'c_minus_a_23': s['scopes']['block_23']['primary_minus_two_edges']}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['smoke', 'pilot', 'pilot_summary', 'prereg', 'gate', 'run'])
    {'smoke': smoke, 'pilot': pilot, 'pilot_summary': pilot_summary, 'prereg': prereg, 'gate': gate, 'run': run}[ap.parse_args().stage]()
