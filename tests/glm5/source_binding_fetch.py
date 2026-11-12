"""How does the answer position fetch the record after it has read the names?  Qwen3-14B BF16, source binding.

Follows Claude Phase 410: in the natural run the answer position (final) reads the name tokens of the question directly in
blocks 24-25.  Here only the attention of the answer position in blocks >= 26 is edited (natural configuration otherwise;
only the answer position is recomputed), over 13 disjoint label-free groups of earlier positions (the chat-template head
and the position itself stay visible; the four code words form ONE group, so no mask tells the model which record is right):
  groups     in blocks >= 26 final cannot attend to group g (block) / can attend to nothing but g (only); combinations of
             the carriers vals / R / think (descriptive: which sources are redundant)
  edge       fixed in advance: vals, the four code-word tokens (the direct fetch).  Regime E: in blocks >= 26 final sees only vals.
  window     in regime E vals is visible only in [26, l)   -> M (earliest sufficient end);
             then only in [s, M)                           -> L0 (latest sufficient start).  Regime W: vals only in [L0, M).
  heads      in regime W, the 40 heads of the window with the largest attention mass from final on the edge are ranked by
             'only this head may use the edge'; cumulative sets give a sufficient set S and, inside regime W, a necessary
             set N; N is also cut in the otherwise unrestricted model (is the route compensated?)
Stages: prereg -> pilot_a .. pilot_f (own exposed panel) -> freeze -> gate -> run.
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
import source_binding_request as RQ
import source_binding_twodim as TD
import source_binding_pointer as PT
import source_binding_nameloc as NL
from source_binding_runtime import Runtime

SRC, MODEL = D.OUT, 'qwen14'
OUT = D.ROOT / 'tests/glm5/result/source_binding_fetch_v1'
RUN, SEED = 'SOURCE-BINDING-FETCH-20261112-v1', 202611120411
SPLITS = {'pilot_fetch': 16, 'holdout_fetch': 48}
CHUNK, BATCH, NB, NH, TOL, START = 256, 128, 40, 40, 1.0, 26
TOKENS = ['final']
ENDS, EDGE, CARRIERS = [27, 28, 29, 30, 31, 32, 34, 36, 38, 40], ['vals'], ['vals', 'R', 'think']
COMBOS = [['vals', 'R'], ['vals', 'think'], ['R', 'think'], ['vals', 'R', 'think']]
CLASSES = ['correct', 'other_source', 'other_entity', 'neither']
GROUPS = ['vals', 'recs', 'hdrs', 'intro', 'qsrc', 'qent', 'qrest', 'instr', 'end', 'R', 'asst', 'think', 'ans']
SUB = [f'val_{c}' for c in CLASSES] + [f'rec_{c}' for c in CLASSES] + ['hdr_correct', 'hdr_other']
KG, SCREEN, KS, NCTRL, CUT, KEEP = len(GROUPS), 40, [1, 2, 3, 4, 6, 8, 12, 16, 24, 32], 3, .65, .85
OTHER_PANELS = ['source_binding_writers_v1/pilot_head', 'source_binding_writers_v1/holdout_head', 'source_binding_fidelity_v1/holdout_fid',
                'source_binding_readers_v1/pilot_read', 'source_binding_readers_v1/holdout_read']
CRITERIA = {
    'model': 'Qwen3-14B original BF16, CPU byte reads, CUDA layer streaming. GLM-4-9B is not run (failed the behaviour gate of this task).',
    'material': 'layout-factorial families as in Phase 405. pilot_fetch: 16 families (256 prompts, exposed by the pilot stages); '
                'holdout_fetch: 48 families (768 prompts). Entity names occur in no earlier panel of this line.',
    'edit': 'attention mask of the answer position (last token) in blocks >= 26 only; natural configuration otherwise. The position '
            'itself and the chat-template head always stay visible. Block 26 is the first block after the window in which the answer '
            'position reads the names (Phase 410, opened result; fixed here before any data of this run).',
    'key_groups': {'vals': 'the code-word tokens of all four records', 'recs': 'the remaining tokens of the four records',
                   'hdrs': 'the two archive header lines',
                   'labelled sub-groups (descriptive only, never used for selection)': 'val_* / rec_* per record class (correct / other '
                   'archive / other entity / neither), hdr_correct / hdr_other',
                   'intro': 'introduction sentence', 'qsrc / qent / qrest': 'archive name, entity name, other tokens of the question',
                   'instr': 'instruction sentence', 'end': '<|im_end|>', 'R': 'the line break after it (relay position)',
                   'asst': '<|im_start|> assistant line-break', 'think': 'empty think block', 'ans': "'Answer'"},
    'outcome': 'the correct record is chosen (4-way forced choice, chance 0.25); all items; family-bootstrap 95 % intervals',
    'regimes': {'natural': 'nothing is cut except what the arm names',
                'E': 'in blocks >= 26 the answer position can attend only to the edge groups',
                'W': 'regime E, and the edge is visible only in blocks [L0, M)'},
    'selection_on_pilot': {
        'edge': 'fixed before pre-registration: vals (see disclosure). pilot_a: block / only for every group and block for the '
                'labelled sub-groups; pilot_b: block and only for the combinations of vals, R, think (both descriptive)',
        'M': 'pilot_c: regime E and vals is visible only in [26, l), l in [27, 28, 29, 30, 31, 32, 34, 36, 38, 40]; M = the smallest l '
             'with keep >= 0.85 (l = 40 is regime E itself; if regime E is below 0.85, M = 40, flagged)',
        'L0': 'pilot_d: regime E and vals is visible only in [s, M), s = 26 .. M - 1; L0 = the largest s with keep >= 0.85 (if none, '
              '26, flagged); the head window is [L0, min(M, L0 + 8))',
        'heads': 'pilot_e: among the heads of the head window, the 40 with the largest mean attention mass from final on the edge are '
                 'ranked by the mean gold margin of only:{h} (regime W and no other head can use the edge). pilot_f: cumulative '
                 'sets K_k, k in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]: S = smallest K_k with only keep >= 0.85 (if none, K_32, flagged); '
                 'N = smallest K_k with blockW keep <= 0.65 (if none, K_32, flagged)',
        'controls': '3 random head sets with the same number of heads per block from the head window, outside the set (SHA-256 seeded)'},
    'arms_on_holdout': {'edge_only': 'regime E', 'floor': 'in blocks >= 26 final can attend to none of the 13 groups', 'win': 'regime W',
                        'only:S': 'regime W and only the heads of S can use the edge',
                        'blockW:N': 'regime W and the heads of N cannot use the edge',
                        'nat:N': 'natural regime; the heads of N cannot use the edge in their blocks',
                        'nat:edge': 'natural regime; final cannot attend to the edge in blocks >= 26',
                        'groups': 'block / only for each of the 13 groups, block for each of the 10 labelled sub-groups, block / only for the 4 '
                                  'combinations of vals, R, think; blocks >= 26 (descriptive)'},
    'two_pass': 'gate stage: unmodified forward and the v1 behaviour gate. Edits run only if the gate passed; otherwise behavior_ineligible.',
    'predictions_on_holdout': {
        'controls_valid': 'tail recomputation control holds (same forced choice as the base forward on every item, score difference <= 1.0); '
                          'floor keep <= 0.65',
        'H0_edge_sufficient': {'supported': 'edge_only keep >= 0.85', 'not_supported': 'edge_only keep < 0.70'},
        'H0w_window_sufficient': {'supported': 'win keep >= 0.85', 'not_supported': 'win keep < 0.70'},
        'H1_heads_sufficient': {'supported': 'only:S keep >= 0.75 and at least 0.15 above each of the 3 controls with lower bound > 0',
                                'not_supported': 'only:S keep < 0.60'},
        'H2_heads_necessary_in_route': {'supported': 'blockW:N keep <= 0.70 and at least 0.15 below each of the 3 controls with lower '
                                                     'bound > 0', 'not_supported': 'blockW:N keep >= 0.90'},
        'H3_not_compensated_in_natural_run': {'supported': 'nat:N keep <= 0.70', 'not_supported': 'nat:N keep >= 0.90'}},
    'descriptive': 'chosen-record classes; the first 4 heads singly; attention masses; the group table on the holdout',
    'not_covered': 'what the heads compute (QK / OV); routes that do not pass through the attention of the answer position in blocks >= '
                   '26 are not edited, only measured as what remains; heads outside the 40 screened; other models. A head set is a set '
                   'of edges, not a full circuit.',
    'disclosure': 'the code was smoke-tested on 2 pilot families (exposed) before pre-registration; thresholds are those of Phase 410 '
                  'and were written first. The first smoke version used the code word of the correct record as a group of its own; '
                  'letting the answer position see only that token hands it the answer (keep 1.00 at any block), so the groups were '
                  'made label-free and the labelled sub-groups are kept as descriptive block arms only. In the second smoke '
                  'version three groups were each sufficient alone (vals, R, think: 1.00) and the data-driven rule picked think, a '
                  'position that already holds the answer; the edge was therefore fixed to vals (the direct fetch, the question of '
                  'this run), the window rule was changed to the earliest sufficient window, and the redundancy between vals, R '
                  'and think is reported descriptively. Thresholds unchanged.',
}


def stream(purpose, key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(f'{RUN}|{SEED}|{purpose}|{key}'.encode()).digest()[:16], 'little'))


def save(name, body):
    from deploy.storage_policy import check_write
    p = OUT / name
    check_write(p, 1 << 20, operation='来源绑定取值产物')
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(body, ensure_ascii=False, indent=1, default=lambda x: x.item() if isinstance(x, np.generic) else x), encoding='utf-8')
    t.replace(p)


def log(*a, **kw):
    RL.OUT = OUT
    RL.log(*a, **kw)


def all_materials():
    taken = {n for rows in D.read(SRC / f'{MODEL}-materials.json')['splits'].values() for r in rows for n in r['names']}
    for p in PT.PANELS + [PT.OUT / 'pilot_layout-materials.json', PT.OUT / 'holdout_layout-materials.json',
                          NL.OUT / 'pilot_name-materials.json', NL.OUT / 'holdout_name-materials.json'] + \
            [D.ROOT / 'tests/glm5/result' / f'{x}-materials.json' for x in OTHER_PANELS]:
        taken |= {n for r in D.read(p) for n in r['names']}
    PT.SPLITS.update(SPLITS)
    return {split: PT.build_materials(split, taken) for split in SPLITS}


def keys(r):
    """Disjoint label-free groups of all positions before the answer position (without the chat-template head), and labelled sub-groups."""
    up, n = r['up'], len(r['tokens'])
    sub = {}
    for c in CLASSES:
        rec = next(v for v in r['records'].values() if v['span'] == up[f'rec_{c}'])
        sub[f'val_{c}'] = list(rec['value_span'])
        sub[f'rec_{c}'] = [j for j in rec['span'] if j not in rec['value_span']]
    sub.update({'hdr_correct': list(up['hdr_correct']), 'hdr_other': list(up['hdr_other'])})
    g = {'vals': sorted(j for c in CLASSES for j in sub[f'val_{c}']), 'recs': sorted(j for c in CLASSES for j in sub[f'rec_{c}']),
         'hdrs': sorted(sub['hdr_correct'] + sub['hdr_other']), **{k: list(up[k]) for k in ('intro', 'qsrc', 'qent', 'qrest', 'instr')},
         'end': [n - 11], 'R': [n - UP.R_FROM_END], 'asst': [n - 9, n - 8, n - 7], 'think': [n - 6, n - 5, n - 4, n - 3], 'ans': [n - 2]}
    assert UP.R_FROM_END == 10 and up['tail0'] == [n - 11] and list(g) == GROUPS and sorted(sub) == sorted(SUB)
    assert sorted(list(up['tpl_head']) + [j for v in g.values() for j in v]) == list(range(n - 1))
    return g, sub, n - 1


DEPS = (RL, UP, HD, QK, QY, RQ, TD, PT, NL)


def prereg():
    from deploy.storage_policy import checked_dir
    from tools import provenance
    from tools.power_analysis import check_phase
    checked_dir(OUT, 1 << 30, operation='来源绑定取值产物')
    assert not (OUT / 'prereg.json').exists()
    plan = D.checked(MODEL)
    mats = all_materials()
    for split, rows in mats.items():
        assert len(rows) == SPLITS[split] * 16 and len({tuple(r['tokens']) for r in rows}) == len(rows)
        for r in rows:
            keys(r)
        save(f'{split}-materials.json', rows)
    n = len(mats['holdout_fetch'])
    body = {'schema': 'source_binding_fetch_prereg.v1', 'run_id': RUN, 'seed': SEED, 'created_at': datetime.now().astimezone().isoformat(),
            'base_plan_sha256': D.sha(SRC / 'plan.json'), 'readers_report_sha256': D.sha(D.ROOT / 'tests/glm5/result/source_binding_readers_v1/report.json'),
            'model_snapshot': plan['models'][MODEL], 'code_sha256': D.sha(__file__),
            'dependency_sha256': {m.__name__: D.sha(m.__file__) for m in DEPS},
            'criteria': CRITERIA, 'materials': {split: {'sha256': D.sha(OUT / f'{split}-materials.json'), 'families': SPLITS[split],
                                                        'rows': len(rows)} for split, rows in mats.items()},
            'power': {'unit': 'paired binary outcome per item (McNemar); alpha = 0.05',
                      'H_level': {'expected_delta': 0.25, 'p_d': 0.50, **check_phase(0.25, n, p_d=0.50)},
                      'H_vs_control': {'expected_delta': 0.15, 'p_d': 0.40, **check_phase(0.15, n, p_d=0.40)},
                      'caveat': 'items are clustered in 48 families and 4 frames'},
            'custody': 'pilot_fetch is exposed by the pilot stages, which fix windows, edges and heads. holdout_fetch is read first by the '
                       'gate stage (behaviour only) and then once by the run stage.',
            'resources': {'gpu_minutes_estimate': 30, 'disk_GiB_estimate': 0.3, 'models_concurrent': 1,
                          'shape': 'chunks of 256 prompts, inner batch 128, per-arm tails; fixed before pre-registration'},
            'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    body['provenance']['model_revision'] = plan['models'][MODEL]
    save('prereg.json', body)
    print(json.dumps({'prereg_sha256': D.sha(OUT / 'prereg.json'), 'power': [body['power']['H_level'], body['power']['H_vs_control']]}), flush=True)


def checked():
    p = D.read(OUT / 'prereg.json')
    assert p['code_sha256'] == D.sha(__file__) and all(p['dependency_sha256'][m.__name__] == D.sha(m.__file__) for m in DEPS), \
        'source changed after pre-registration'
    assert all(v['sha256'] == D.sha(OUT / f'{k}-materials.json') for k, v in p['materials'].items())
    D.checked(MODEL)
    return p


def engine(rows, arms=(), stage='run', screen=None):
    """arms: (name, 'final', parts), parts = [(kind, groups, blocks)] with kind 'block' (final cannot attend to the groups) or 'only'
    (final can attend to no group except these) and blocks = {block: None (all heads) | [heads]}.
    screen: {'final': (edge groups, [blocks])} -> mean attention mass of every head of these blocks from final on the edge."""
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    t, F = torch, torch.nn.functional
    assert snapshot(D.MODELS[MODEL]) == D.read(SRC / f'{MODEL}-identity.json')['model_snapshot']
    arms = [('noop', 'final', [('block', [], {0: []})])] + list(arms) if arms else []
    start, total = time.monotonic(), len(rows)
    results, masses = {}, {}
    with GpuLease(), t.inference_mode():
        rt = Runtime(MODEL)
        hd, kv, grp = rt.hd, rt.kv, rt.heads // rt.kv
        rot = lambda x, co, si: x * co + rt.mod.rotate_half(x) * si
        log(stage, 0, total * NB, start, rows=total, arms=len(arms))
        for c0 in range(0, total, CHUNK):
            rs = rows[c0:c0 + CHUNK]
            N = len(rs)
            ids, pos, pe, mask = rt.context(rs)
            h, length = rt.embed(ids), ids.shape[1]
            off = [length - len(r['tokens']) for r in rs]
            info = {}
            for tok in TOKENS:
                rw = t.zeros((N, length), dtype=t.bool, device='cuda')
                gm = {g: t.zeros((N, length), dtype=t.bool, device='cuda') for g in GROUPS + SUB + ['all']}
                firsts = []
                for i, r in enumerate(rs):
                    g, sub, first = keys(r)
                    rw[i, length - 1] = True
                    for k, v in g.items():
                        gm[k][i, [off[i] + j for j in v]] = True
                        gm['all'][i, [off[i] + j for j in v]] = True
                    for k, v in sub.items():
                        gm[k][i, [off[i] + j for j in v]] = True
                    firsts.append(off[i] + first)
                info[tok] = {'rw': rw, 'gm': gm, 'P': min(firsts)}
            spec = {}
            for name, tok, parts in arms:
                gm, ms = info[tok]['gm'], []
                for kind, groups, blocks in parts:
                    sel = t.zeros((N, length), dtype=t.bool, device='cuda')
                    for g in groups:
                        sel |= gm[g]
                    ms.append((sel if kind == 'block' else gm['all'] & ~sel, {int(b): v for b, v in blocks.items()}))   # key masks
                spec[name] = (tok, ms, min(b for _, bl in ms for b in bl))
            sc = {tok: (sum((info[tok]['gm'][g] for g in gs[1:]), info[tok]['gm'][gs[0]].clone()), set(bl)) for tok, (gs, bl) in (screen or {}).items()}
            state, cap = {}, {}
            for b in range(NB):
                for name, (tok, _, fork) in spec.items():
                    if fork == b:
                        state[name] = h[:, info[tok]['P']:].clone()
                layer = rt.block(b)
                a = layer.self_attn
                need_q = any(b in bl for _, bl in sc.values())
                handles = []
                if state or need_q:
                    handles = [a.k_proj.register_forward_hook(lambda m, i, o: cap.__setitem__('k', o)),
                               a.v_proj.register_forward_hook(lambda m, i, o: cap.__setitem__('v', o)),
                               a.q_proj.register_forward_hook(lambda m, i, o: cap.__setitem__('q', o))]
                y = t.empty_like(h)
                for lo in range(0, N, BATCH):
                    hi = min(lo + BATCH, N)
                    B = hi - lo
                    cos, sin = pe[0][lo:hi], pe[1][lo:hi]
                    y[lo:hi] = layer(h[lo:hi], attention_mask=mask[lo:hi], position_ids=pos[lo:hi], position_embeddings=(cos, sin), use_cache=False)
                    kb, vb, qb = cap.get('k'), cap.get('v'), cap.get('q')
                    for tok, (edge, bl) in sc.items():
                        if b not in bl:
                            continue
                        P = info[tok]['P']
                        q = rot(a.q_norm(qb[:, P:].reshape(B, length - P, NH, hd)).transpose(1, 2), cos[:, None, P:], sin[:, None, P:])
                        k = rt.mod.repeat_kv(rot(a.k_norm(kb.reshape(B, length, kv, hd)).transpose(1, 2), cos[:, None], sin[:, None]), grp)
                        w = (q @ k.transpose(2, 3) * a.scaling).float().masked_fill(~mask[lo:hi][:, :, P:, :], float('-inf')).softmax(-1)
                        m = (w * edge[lo:hi, None, None, :]).sum(-1)
                        rwt = info[tok]['rw'][lo:hi, None, P:].float()
                        acc = masses.setdefault(tok, {}).setdefault(b, [t.zeros(NH, device='cuda'), 0])
                        acc[0] += ((m * rwt).sum(-1) / rwt.sum(-1).clamp_min(1)).sum(0)
                        acc[1] += B
                    for name in state:
                        tok, ms, _ = spec[name]
                        P = info[tok]['P']
                        T = length - P
                        tm = mask[lo:hi][:, :, P:, :]
                        for cm, blocks in ms:
                            if b not in blocks or name == 'noop':
                                continue
                            c = (info[tok]['rw'][lo:hi, P:, None] & cm[lo:hi, None, :])[:, None]
                            if blocks[b] is None:
                                tm = tm & ~c
                            elif blocks[b]:
                                tm = tm.expand(-1, NH, -1, -1).clone()
                                tm[:, blocks[b]] = tm[:, blocks[b]] & ~c
                        x = state[name][lo:hi]
                        hn = layer.input_layernorm(x)
                        q = a.q_norm(a.q_proj(hn).view(B, T, NH, hd)).transpose(1, 2)
                        k = a.k_norm(t.cat([kb[:, :P], a.k_proj(hn)], 1).view(B, length, kv, hd)).transpose(1, 2)
                        v = t.cat([vb[:, :P], a.v_proj(hn)], 1).view(B, length, kv, hd).transpose(1, 2)
                        q, k = rot(q, cos[:, None, P:], sin[:, None, P:]), rot(k, cos[:, None], sin[:, None])
                        o = F.scaled_dot_product_attention(q, rt.mod.repeat_kv(k, grp), rt.mod.repeat_kv(v, grp), attn_mask=tm)
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
                    log(stage, c0 * NB + (b + 1) * N, total * NB, start, block=b, arms=len(state))
            norm = rt.Norm(rt.dim, eps=rt.cfg.rms_norm_eps).to(device='cuda', dtype=t.bfloat16)
            norm.weight = t.nn.Parameter(rt.weight('model.norm.weight'), requires_grad=False)
            W = rt.weight('lm_head.weight' if 'lm_head.weight' in rt.index else 'model.embed_tokens.weight')
            logits = lambda x: t.cat([(norm(x[lo:lo + BATCH]) @ W.T).float() for lo in range(0, N, BATCH)])
            results.setdefault('base', []).extend(rt.score(logits(h[:, -1]), rs))
            for key, s in state.items():
                results.setdefault(key, []).extend(rt.score(logits(s[:, -1]), rs))
            del W, norm, h, state, ids, pos, pe, mask, info, spec
            gc.collect()
            t.cuda.empty_cache()
        peak = round(t.cuda.max_memory_allocated() / 2 ** 20)
        masses = {tok: {b: (acc[0] / acc[1]).cpu().tolist() for b, acc in d.items()} for tok, d in masses.items()}
    meta = {'peak_MiB': peak, 'seconds': round(time.monotonic() - start, 1)}
    if arms:
        meta['noop_same_choice'] = all(x['pred_record'] == y['pred_record'] for x, y in zip(results['noop'], results['base']))
        meta['noop_max_score_difference'] = max(abs(p - q) for x, y in zip(results['noop'], results['base']) for p, q in zip(x['scores'], y['scores']))
        meta['tail_control_ok'] = bool(meta['noop_same_choice'] and meta['noop_max_score_difference'] <= TOL)
    return results, masses, meta


def summarise(results, rows, pairs=()):
    fam = np.array([r['family'] for r in rows])
    members = [np.flatnonzero(fam == f) for f in sorted(set(fam))]
    draws = np.random.default_rng(SEED).integers(0, len(members), (1000, len(members)))
    cnt = np.array([len(m) for m in members])
    s0, e0 = np.array([r['source'] for r in rows]), np.array([r['entity'] for r in rows])

    def ci(v):
        v = np.asarray(v, float)
        per = np.array([v[m].sum() for m in members])
        bs = per[draws].sum(1) / cnt[draws].sum(1)
        return {'rate': round(float(v.mean()), 4), 'ci95': [round(float(np.quantile(bs, .025)), 4), round(float(np.quantile(bs, .975)), 4)]}
    col = lambda k: np.array([x['forced_correct'] for x in results[k]], float)
    out = {'n': len(rows), 'arms': {}, 'differences': {}}
    for k, res in results.items():
        pred = np.array([x['pred_record'] for x in res])
        ps, pe = pred // 2, pred % 2
        out['arms'][k] = {'keep': ci(col(k)), 'margin': round(float(np.mean([x['gold_margin'] for x in res])), 3),
                          'classes': [round(float(v.mean()), 4) for v in ((ps == s0) & (pe == e0), (ps == s0) & (pe != e0),
                                                                          (ps != s0) & (pe == e0), (ps != s0) & (pe != e0))]}
    for a, b in pairs:
        out['differences'][f'{a}-{b}'] = ci(col(a) - col(b))
    return out


def pilot_rows(smoke):
    if smoke:
        return [r for r in all_materials()['pilot_fetch'] if int(r['family'][-4:]) < 2]
    checked()
    return D.read(OUT / 'pilot_fetch-materials.json')


def finish(name, rows, arms, smoke, extra, screen=None):
    results, masses, meta = engine(rows, arms, stage=name, screen=screen)
    s = summarise(results, rows)
    keep = lambda k: s['arms'][k]['keep']['rate']
    margin = lambda k: s['arms'][k]['margin']
    body = extra(keep, margin, masses)
    if smoke:
        print(json.dumps({'smoke': name, 'meta': meta, **body})[:5000], flush=True)
        return body
    assert meta['tail_control_ok'] and not (OUT / f'{name.replace("_", "-")}.json').exists()
    save(f'{name.replace("_", "-")}.json', {'schema': f'source_binding_fetch_{name}.v1', 'run_id': RUN, 'summary': s, **body, **meta})
    print(json.dumps({'noop_max_score_difference': meta['noop_max_score_difference'], 'seconds': meta['seconds'], **body})[:6000], flush=True)
    return body


def prev(name, smoke, fn):
    return fn(True) if smoke else D.read(OUT / f'{name}.json')


ALL = {b: None for b in range(START, NB)}


def outside(lo, hi):
    return {b: None for b in range(START, NB) if not lo <= b < hi}


def regime_e(c):
    return [('only', c['edge'], ALL)]


def only(K, c):
    """Regime W and only the heads of K can use the edge (heads of K lie in the window)."""
    keep = {}
    for b, h in K:
        keep.setdefault(b, set()).add(h)
    return regime_e(c) + [('block', c['edge'], {b: ([h for h in range(NH) if h not in keep[b]] if b in keep else None) for b in range(START, NB)})]


def by_block(K):
    out = {}
    for b, h in K:
        out.setdefault(b, []).append(h)
    return out


def blockw(K, c):
    """Regime W and the heads of K cannot use the edge."""
    return regime_e(c) + [('block', c['edge'], {**outside(c['L0'], c['M']), **by_block(K)})]


def nat(K, c):
    return [('block', c['edge'], by_block(K))] if K else [('block', [], {0: []})]


def pilot_a(smoke=False):
    rows = pilot_rows(smoke)

    def extra(keep, margin, _):
        out = {}
        for tok in TOKENS:
            order = sorted(GROUPS, key=lambda g: (-keep(f'{tok}:only:{g}'), keep(f'{tok}:block:{g}'), -GROUPS.index(g)))
            out[tok] = {'order': order, 'block_only': {g: [keep(f'{tok}:block:{g}'), keep(f'{tok}:only:{g}')] for g in order},
                        'sub_block': {g: keep(f'{tok}:block:{g}') for g in SUB}}
        return out
    arms = [(f'{tok}:{k}:{g}', tok, [(k, [g], ALL)]) for tok in TOKENS for g in GROUPS for k in ('block', 'only')]
    arms += [(f'{tok}:block:{g}', tok, [('block', [g], ALL)]) for tok in TOKENS for g in SUB]
    return finish('pilot_a', rows, arms, smoke, extra)


def pilot_b(smoke=False):
    rows = pilot_rows(smoke)
    prev('pilot-a', smoke, pilot_a)
    name = lambda k, c: f'final:{k}:' + '+'.join(c)

    def extra(keep, margin, _):
        return {'final': {'edge': EDGE, 'combinations_block_only': {'+'.join(c): [keep(name('block', c)), keep(name('only', c))] for c in COMBOS}}}
    return finish('pilot_b', rows, [(name(k, c), 'final', [(k, c, ALL)]) for c in COMBOS for k in ('block', 'only')], smoke, extra)


def pilot_c(smoke=False):
    rows = pilot_rows(smoke)
    b = prev('pilot-b', smoke, pilot_b)

    def extra(keep, margin, _):
        table = {str(l): keep(f'final:until@{l}') for l in ENDS}
        m = next((l for l in ENDS if table[str(l)] >= KEEP), None)
        return {'final': {'edge': b['final']['edge'], 'until': table, 'M': m if m is not None else NB, 'M_rule_met': m is not None}}
    arms = [(f'final:until@{l}', 'final', regime_e(b['final']) + [('block', b['final']['edge'], outside(START, l))]) for l in ENDS]
    return finish('pilot_c', rows, arms, smoke, extra)


def pilot_d(smoke=False):
    rows = pilot_rows(smoke)
    c = prev('pilot-c', smoke, pilot_c)['final']
    starts = list(range(START, c['M']))

    def extra(keep, margin, _):
        table = {str(s): keep(f'final:start@{s}') for s in starts}
        l0 = max([s for s in starts if table[str(s)] >= KEEP], default=None)
        l0v = l0 if l0 is not None else START
        return {'final': {**c, 'L0': l0v, 'L0_rule_met': l0 is not None, 'window': list(range(l0v, min(c['M'], l0v + 8))), 'start': table}}
    arms = [(f'final:start@{s}', 'final', regime_e(c) + [('block', c['edge'], outside(s, c['M']))]) for s in starts]
    return finish('pilot_d', rows, arms, smoke, extra)


def pilot_e(smoke=False):
    rows = pilot_rows(smoke)
    c = prev('pilot-d', smoke, pilot_d)
    if not smoke:
        assert not (OUT / 'pilot-e.json').exists()
    _, masses, _ = engine(rows, stage='pilot_e_screen', screen={tok: (c[tok]['edge'], c[tok]['window']) for tok in TOKENS})
    cand = {}
    for tok in TOKENS:
        flat = sorted(((masses[tok][b][h], b, h) for b in c[tok]['window'] for h in range(NH)), reverse=True)[:8 if smoke else SCREEN]
        cand[tok] = [(b, h, round(m, 4)) for m, b, h in flat]

    def extra(keep, margin, _):
        out = {}
        for tok in TOKENS:
            table = {f'{b}.{h}': {'mass': m, 'only_margin': margin(f'{tok}:only:{b}.{h}'), 'only_keep': keep(f'{tok}:only:{b}.{h}')}
                     for b, h, m in cand[tok]}
            order = sorted([(b, h) for b, h, _ in cand[tok]], key=lambda x: (-table[f'{x[0]}.{x[1]}']['only_margin'], x))
            out[tok] = {**c[tok], 'order': [list(x) for x in order], 'floor_keep': keep(f'{tok}:floor'), 'win_keep': keep(f'{tok}:win'),
                        'heads': table}
        return out
    arms = []
    for tok in TOKENS:
        arms += [(f'{tok}:floor', tok, [('block', GROUPS, ALL)]), (f'{tok}:win', tok, blockw([], c[tok]))]
        arms += [(f'{tok}:only:{b}.{h}', tok, only([(b, h)], c[tok])) for b, h, _ in cand[tok]]
    return finish('pilot_e', rows, arms, smoke, extra)


def pilot_f(smoke=False):
    rows = pilot_rows(smoke)
    d = prev('pilot-e', smoke, pilot_e)

    def extra(keep, margin, _):
        out = {}
        for tok in TOKENS:
            order = d[tok]['order']
            ks = [k for k in KS if k <= len(order)]
            a = next((k for k in ks if keep(f'{tok}:only:K{k}') >= KEEP), None)
            n = next((k for k in ks if keep(f'{tok}:blockW:K{k}') <= CUT), None)
            out[tok] = {**{x: d[tok][x] for x in ('L0', 'M', 'window', 'edge', 'order')},
                        'sufficient_set': order[:a or ks[-1]], 'sufficient_rule_met': a is not None,
                        'necessary_set': order[:n or ks[-1]], 'necessary_rule_met': n is not None,
                        'sets_only_blockW_nat': {f'K{k}': [keep(f'{tok}:only:K{k}'), keep(f'{tok}:blockW:K{k}'), keep(f'{tok}:nat:K{k}')] for k in ks}}
        return out
    arms = []
    for tok in TOKENS:
        c, order = d[tok], [tuple(x) for x in d[tok]['order']]
        for k in [k for k in KS if k <= len(order)]:
            arms += [(f'{tok}:only:K{k}', tok, only(order[:k], c)), (f'{tok}:blockW:K{k}', tok, blockw(order[:k], c)),
                     (f'{tok}:nat:K{k}', tok, nat(order[:k], c))]
    return finish('pilot_f', rows, arms, smoke, extra)


def controls(K, window, purpose):
    K, out = [tuple(x) for x in K], []
    for i in range(NCTRL):
        rnd, taken, pick = stream(purpose, i), set(K), []
        for b in sorted({b for b, _ in K}):
            need = sum(1 for x in K if x[0] == b)
            pool = [(b, h) for h in range(NH) if (b, h) not in taken]
            if len(pool) < need:
                pool += [(c, h) for c in window for h in range(NH) if (c, h) not in taken and c != b]
            for j in rnd.choice(len(pool), need, replace=False):
                pick.append(pool[int(j)])
                taken.add(pool[int(j)])
        out.append(sorted(pick))
    return out


def freeze():
    checked()
    assert not (OUT / 'frozen.json').exists()
    e = D.read(OUT / 'pilot-f.json')
    body = {'schema': 'source_binding_fetch_frozen.v1', 'run_id': RUN, 'prereg_sha256': D.sha(OUT / 'prereg.json'),
            **{f'pilot_{x}_sha256': D.sha(OUT / f'pilot-{x}.json') for x in 'abcdef'}, 'created_at': datetime.now().astimezone().isoformat()}
    for tok in TOKENS:
        c = {k: e[tok][k] for k in ('L0', 'M', 'window', 'edge', 'order', 'sufficient_set', 'sufficient_rule_met', 'necessary_set', 'necessary_rule_met')}
        c['controls_S'] = controls(c['sufficient_set'], c['window'], f'{tok}|ctrlS')
        c['controls_N'] = controls(c['necessary_set'], c['window'], f'{tok}|ctrlN')
        body[tok] = c
    save('frozen.json', body)
    print(json.dumps({tok: {k: body[tok][k] for k in ('L0', 'M', 'edge', 'sufficient_set', 'necessary_set')} for tok in TOKENS}), flush=True)


def gate():
    checked()
    assert (OUT / 'frozen.json').exists() and not (OUT / 'gate.json').exists(), 'holdout behaviour already read'
    rows = D.read(OUT / 'holdout_fetch-materials.json')
    results, _, meta = engine(rows, stage='gate')
    save('gate-items.json', results['base'])
    g = D.behavior(rows, results['base'])
    save('gate.json', {'schema': 'source_binding_fetch_gate.v1', 'run_id': RUN, 'items_sha256': D.sha(OUT / 'gate-items.json'), **g, **meta})
    print(json.dumps({k: g[k] for k in ('passed', 'forced_accuracy', 'top1_accuracy')}), flush=True)


def run():
    from tools import provenance
    pre, fz = checked(), D.read(OUT / 'frozen.json')
    assert fz['prereg_sha256'] == D.sha(OUT / 'prereg.json') and fz['pilot_f_sha256'] == D.sha(OUT / 'pilot-f.json')
    assert not (OUT / 'report.json').exists(), 'holdout already used'
    g = D.read(OUT / 'gate.json')
    rows = D.read(OUT / 'holdout_fetch-materials.json')
    names = ('H0_edge_sufficient', 'H0w_window_sufficient', 'H1_heads_sufficient', 'H2_heads_necessary_in_route',
             'H3_not_compensated_in_natural_run')
    v = {}
    if g['passed']:
        arms, pairs = [], []
        for tok in TOKENS:
            c = fz[tok]
            S, N = [tuple(x) for x in c['sufficient_set']], [tuple(x) for x in c['necessary_set']]
            mk = lambda name, parts: (f'{tok}:{name}', tok, parts)
            arms += [mk('edge_only', regime_e(c)), mk('floor', [('block', GROUPS, ALL)]), mk('win', blockw([], c)),
                     mk('nat:edge', [('block', c['edge'], ALL)]), mk('only:S', only(S, c)), mk('blockW:N', blockw(N, c)), mk('nat:N', nat(N, c))]
            arms += [mk(f'{k}:{g}', [(k, [g], ALL)]) for g in GROUPS for k in ('block', 'only')]
            arms += [mk(f'block:{g}', [('block', [g], ALL)]) for g in SUB]
            arms += [mk(f'{k}:' + '+'.join(cc), [(k, cc, ALL)]) for cc in COMBOS for k in ('block', 'only')]
            arms += [mk(f'only:ctrl{i}', only([tuple(x) for x in k], c)) for i, k in enumerate(c['controls_S'])]
            arms += [mk(f'blockW:ctrl{i}', blockw([tuple(x) for x in k], c)) for i, k in enumerate(c['controls_N'])]
            for b, h in [tuple(x) for x in c['order'][:4]]:
                arms += [mk(f'only:{b}.{h}', only([(b, h)], c)), mk(f'blockW:{b}.{h}', blockw([(b, h)], c))]
            pairs += [(f'{tok}:only:S', f'{tok}:only:ctrl{i}') for i in range(NCTRL)] + [(f'{tok}:blockW:ctrl{i}', f'{tok}:blockW:N') for i in range(NCTRL)]
        results, _, meta = engine(rows, arms, stage='run')
        save('items.json', {k: [{'id': x['id'], 'pred_record': x['pred_record'], 'gold_margin': x['gold_margin']} for x in r] for k, r in results.items()})
        base = D.read(OUT / 'gate-items.json')
        meta['base_replay_max_score_difference'] = max(abs(a - b) for x, y in zip(results['base'], base) for a, b in zip(x['scores'], y['scores']))
        s = summarise(results, rows, pairs)
        keep = lambda k: s['arms'][k]['keep']['rate']
        tri = lambda ok, bad: 'supported' if ok else 'not_supported' if bad else 'inconclusive'
        for tok in TOKENS:
            beat = lambda keys: all(s['differences'][k]['rate'] >= .15 and s['differences'][k]['ci95'][0] > 0 for k in keys)
            ok = meta['tail_control_ok'] and keep(f'{tok}:floor') <= CUT
            v[tok] = {'controls_valid': bool(ok)}
            if ok:
                v[tok]['H0_edge_sufficient'] = tri(keep(f'{tok}:edge_only') >= .85, keep(f'{tok}:edge_only') < .70)
                v[tok]['H0w_window_sufficient'] = tri(keep(f'{tok}:win') >= .85, keep(f'{tok}:win') < .70)
                v[tok]['H1_heads_sufficient'] = tri(keep(f'{tok}:only:S') >= .75 and beat([f'{tok}:only:S-{tok}:only:ctrl{i}' for i in range(NCTRL)]),
                                                    keep(f'{tok}:only:S') < .60)
                v[tok]['H2_heads_necessary_in_route'] = tri(keep(f'{tok}:blockW:N') <= .70 and beat(
                    [f'{tok}:blockW:ctrl{i}-{tok}:blockW:N' for i in range(NCTRL)]), keep(f'{tok}:blockW:N') >= .90)
                v[tok]['H3_not_compensated_in_natural_run'] = tri(keep(f'{tok}:nat:N') <= .70, keep(f'{tok}:nat:N') >= .90)
            else:
                v[tok].update({k: 'controls_invalid' for k in names})
    else:
        s, meta = {'note': 'behaviour gate failed; no edit was run'}, {}
        v = {tok: {k: 'behavior_ineligible' for k in names} for tok in TOKENS}
    report = {'schema': 'source_binding_fetch_report.v1', 'run_id': RUN, 'split': 'holdout_fetch', 'gate_passed': g['passed'],
              'frozen': {tok: fz[tok] for tok in TOKENS}, 'prereg_sha256': D.sha(OUT / 'prereg.json'), 'frozen_sha256': D.sha(OUT / 'frozen.json'),
              'gate_sha256': D.sha(OUT / 'gate.json'), 'items_sha256': D.sha(OUT / 'items.json') if g['passed'] else None,
              'criteria': pre['criteria'], 'verdicts': v, 'summary': s, **meta, 'formal_evidence_promotion': 0,
              'custody': 'holdout_fetch read once by the gate stage and once by this run',
              'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    save('report.json', report)
    path = provenance.write_result('SOURCE-BINDING-fetch-holdout', {'verdicts': v, 'gate_passed': g['passed'],
                                                                      'sets': {tok: {k: fz[tok][k] for k in ('window', 'edge', 'sufficient_set', 'necessary_set')}
                                                                               for tok in TOKENS}, 'report_sha256': D.sha(OUT / 'report.json')},
                                   model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)
    print(json.dumps({'verdicts': v, 'result': str(path)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    stages = {'prereg': prereg, 'pilot_a': pilot_a, 'pilot_b': pilot_b, 'pilot_c': pilot_c, 'pilot_d': pilot_d, 'pilot_e': pilot_e, 'pilot_f': pilot_f,
              'freeze': freeze, 'gate': gate, 'run': run}
    ap.add_argument('--stage', required=True, choices=list(stages))
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    stages[a.stage](a.smoke) if a.stage.startswith('pilot') else stages[a.stage]()
