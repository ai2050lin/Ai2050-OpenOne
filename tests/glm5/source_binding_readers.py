"""Who reads the name tokens of the question in the natural run?  Qwen3-14B BF16, source binding.

Follows Claude Phases 406-409: the archive-name token of the question carries 'which block' (blocks 16-22) and the entity-name
tokens carry a block -> line map (block 22); exchanging the state of the relay position R in the natural configuration never
changes the answer (Phase 409), so the natural reader of these tokens is unknown.  Here only the attention of the positions
AFTER a name token T onto T is edited (natural configuration otherwise; all later positions recomputed), separately for T:
  groups     in all blocks reader group g cannot attend to T (block) / no other reader can (only)
             -> edge = the smallest set of reader groups that is sufficient alone.  Regime E: every other reader is cut.
  timing     in regime E the edge may be used only from block l on            -> L0 (latest sufficient start)
  window     in regime E the edge may be used only in [L0, L0 + d)            -> M.  Regime W: edge only in [L0, M).
  heads      in regime W, the 40 heads of the window with the largest attention mass from the edge readers on T are ranked
             by 'only this head may use the edge'; cumulative sets give a sufficient set S and, inside regime W, a
             necessary set N; N is also cut in the otherwise unrestricted model (is the route compensated?)
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
OUT = D.ROOT / 'tests/glm5/result/source_binding_readers_v1'
RUN, SEED = 'SOURCE-BINDING-READERS-20261111-v1', 202611110410
SPLITS = {'pilot_read': 16, 'holdout_read': 48}
CHUNK, BATCH, NB, NH, TOL = 256, 128, 40, 40, 1.0
TOKENS = ['qsrc', 'qent']
TIMING, ENDS = [0, 8, 12, 16, 18, 20, 22, 24, 26, 28], [1, 2, 4, 6, 8, 12]
GROUPS = ['other_name', 'q_mid', 'q_end', 'instr', 'end', 'R', 'asst', 'think', 'ans', 'final']
KG, SCREEN, KS, NCTRL, CUT, KEEP = len(GROUPS), 40, [1, 2, 3, 4, 6, 8, 12, 16, 24, 32], 3, .65, .85
OTHER_PANELS = ['source_binding_writers_v1/pilot_head', 'source_binding_writers_v1/holdout_head', 'source_binding_fidelity_v1/holdout_fid']
CRITERIA = {
    'model': 'Qwen3-14B original BF16, CPU byte reads, CUDA layer streaming. GLM-4-9B is not run (failed the behaviour gate of this task).',
    'material': 'layout-factorial families as in Phase 405. pilot_read: 16 families (256 prompts, exposed by the pilot stages); '
                'holdout_read: 48 families (768 prompts). Entity names occur in no earlier panel of this line.',
    'edit': 'attention mask from positions after a name token onto that name token only (archive-name token qsrc; entity-name tokens '
            'qent), natural configuration otherwise. Nothing else is masked; the name tokens themselves are computed as usual.',
    'reader_groups_after_the_name': {
        'other_name': 'the other name of the question when it follows', 'q_mid': 'question words after the name',
        'q_end': 'the token that closes the question (punctuation + line break)', 'instr': 'the instruction sentence (10 tokens)',
        'end': '<|im_end|> of the user turn', 'R': 'the line break after it (relay position of Phases 397-405)',
        'asst': '<|im_start|> assistant line-break', 'think': 'the 4 tokens of the empty think block', 'ans': "'Answer'",
        'final': "':' + line break, the position that outputs the answer"},
    'outcome': 'the correct record is chosen (4-way forced choice, chance 0.25; a lost binary dimension gives about 0.50); all items; '
               'family-bootstrap 95 % intervals',
    'regimes': {'natural': 'nothing is cut except what the arm names',
                'E': 'in all blocks every reader group outside the edge cannot attend to T',
                'W': 'regime E, and the edge readers can attend to T only in blocks [L0, M)'},
    'selection_on_pilot': {
        'edge': 'pilot_a: for every reader group g, g cannot attend to T in any block (block) / no other reader can (only); groups '
                'ordered by only keep (descending), ties by block keep (ascending), then by position. pilot_b: cumulative sets of the '
                'first k groups, k = 1..10; edge = the smallest set with only keep >= 0.85 (k = 10 is the unedited model)',
        'L0': 'pilot_c: regime E and the edge is cut in blocks < l, l in [0, 8, 12, 16, 18, 20, 22, 24, 26, 28]; L0 = the largest l '
              'with keep >= 0.85 (l = 0 is regime E itself)',
        'M': 'pilot_d: regime E and the edge is cut outside [L0, L0 + d), d in [1, 2, 4, 6, 8, 12]; M = L0 + the smallest d with keep '
             '>= 0.85 (if none, 40, flagged); the head window is [L0, min(M, L0 + 8))',
        'heads': 'pilot_e: among the heads of the head window, the 40 with the largest mean attention mass from the edge readers on T '
                 'are ranked by the mean gold margin of only:{h} (regime W and no other head can use the edge). pilot_f: cumulative '
                 'sets K_k, k in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]: S = smallest K_k with only keep >= 0.85 (if none, K_32, flagged); '
                 'N = smallest K_k with blockW keep <= 0.65 (if none, K_32, flagged)',
        'controls': '3 random head sets with the same number of heads per block from the head window, outside the set (SHA-256 seeded)'},
    'arms_on_holdout': {'edge_only': 'regime E', 'floor': 'no position after T can attend to T in any block', 'win': 'regime W',
                        'only:S': 'regime W and only the heads of S can use the edge',
                        'blockW:N': 'regime W and the heads of N cannot use the edge',
                        'nat:N': 'natural regime; the heads of N cannot use the edge in their blocks',
                        'nat:edge': 'natural regime; the edge readers cannot attend to T in any block',
                        'groups': 'block / only for each of the 10 reader groups, all blocks (descriptive)'},
    'two_pass': 'gate stage: unmodified forward and the v1 behaviour gate. Edits run only if the gate passed; otherwise behavior_ineligible.',
    'predictions_on_holdout_per_token': {
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
    'not_covered': 'what the readers do with what they read (how the record is fetched afterwards); heads outside the 40 screened per '
                   'token; other models. An attention edge removes name identity and pointer together; the timing stage separates them '
                   'only by block. A head set is a set of edges, not a full circuit.',
    'disclosure': 'the code was smoke-tested on 2 pilot families (exposed) before pre-registration; the thresholds 0.85 / 0.75 / 0.70 / '
                  '0.65 / 0.60 / 0.15 are those of Phase 407 and were written first. In the smoke test several reader groups were each '
                  'sufficient alone and none was necessary for the archive name (cutting the answer position alone kept 0.94), so the '
                  'necessity-first selection of Phase 407 gave no usable window; selection was changed to sufficiency-first (edge, '
                  'start and window are the smallest that suffice alone) and necessity is tested inside the route and in the natural '
                  'run. Thresholds unchanged.',
}


def stream(purpose, key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(f'{RUN}|{SEED}|{purpose}|{key}'.encode()).digest()[:16], 'little'))


def save(name, body):
    from deploy.storage_policy import check_write
    p = OUT / name
    check_write(p, 1 << 20, operation='来源绑定读取者产物')
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


def after(r, tok):
    """Disjoint reader groups: the positions after the last token of the name."""
    up, n, last = r['up'], len(r['tokens']), max(r['up'][tok])
    other = 'qent' if tok == 'qsrc' else 'qsrc'
    qr = [j for j in up['qrest'] if j > last]
    assert qr and qr[-1] == max(up['qrest']) and up['instr'][0] == qr[-1] + 1 and up['tail0'] == [n - 11]
    g = {'other_name': [j for j in up[other] if j > last], 'q_mid': qr[:-1], 'q_end': qr[-1:], 'instr': list(up['instr']), 'end': [n - 11],
         'R': [n - UP.R_FROM_END], 'asst': [n - 9, n - 8, n - 7], 'think': [n - 6, n - 5, n - 4, n - 3], 'ans': [n - 2], 'final': [n - 1]}
    assert UP.R_FROM_END == 10 and sorted(j for v in g.values() for j in v) == list(range(last + 1, n))
    return g, last + 1


DEPS = (RL, UP, HD, QK, QY, RQ, TD, PT, NL)


def prereg():
    from deploy.storage_policy import checked_dir
    from tools import provenance
    from tools.power_analysis import check_phase
    checked_dir(OUT, 1 << 30, operation='来源绑定读取者产物')
    assert not (OUT / 'prereg.json').exists()
    plan = D.checked(MODEL)
    mats = all_materials()
    for split, rows in mats.items():
        assert len(rows) == SPLITS[split] * 16 and len({tuple(r['tokens']) for r in rows}) == len(rows)
        for r in rows:
            for tok in TOKENS:
                after(r, tok)
        save(f'{split}-materials.json', rows)
    n = len(mats['holdout_read'])
    body = {'schema': 'source_binding_readers_prereg.v1', 'run_id': RUN, 'seed': SEED, 'created_at': datetime.now().astimezone().isoformat(),
            'base_plan_sha256': D.sha(SRC / 'plan.json'), 'fidelity_report_sha256': D.sha(D.ROOT / 'tests/glm5/result/source_binding_fidelity_v1/report.json'),
            'model_snapshot': plan['models'][MODEL], 'code_sha256': D.sha(__file__),
            'dependency_sha256': {m.__name__: D.sha(m.__file__) for m in DEPS},
            'criteria': CRITERIA, 'materials': {split: {'sha256': D.sha(OUT / f'{split}-materials.json'), 'families': SPLITS[split],
                                                        'rows': len(rows)} for split, rows in mats.items()},
            'power': {'unit': 'paired binary outcome per item (McNemar); alpha = 0.05',
                      'H_level': {'expected_delta': 0.25, 'p_d': 0.50, **check_phase(0.25, n, p_d=0.50)},
                      'H_vs_control': {'expected_delta': 0.15, 'p_d': 0.40, **check_phase(0.15, n, p_d=0.40)},
                      'caveat': 'items are clustered in 48 families and 4 frames'},
            'custody': 'pilot_read is exposed by the pilot stages, which fix windows, edges and heads. holdout_read is read first by the '
                       'gate stage (behaviour only) and then once by the run stage.',
            'resources': {'gpu_minutes_estimate': 40, 'disk_GiB_estimate': 0.3, 'models_concurrent': 1,
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
    """arms: (name, tok, parts), parts = [(kind, groups, blocks)] with kind 'block' (the readers of groups cannot attend to the
    name tok) or 'only' (no reader after the name except groups can) and blocks = {block: None (all heads) | [heads]}.
    screen: {tok: (edge groups, [blocks])} -> mean attention mass of every head of these blocks from the edge readers on the name."""
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    t, F = torch, torch.nn.functional
    assert snapshot(D.MODELS[MODEL]) == D.read(SRC / f'{MODEL}-identity.json')['model_snapshot']
    arms = [('noop', 'qsrc', [('block', [], {0: []})])] + list(arms) if arms else []
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
                gm = {g: t.zeros((N, length), dtype=t.bool, device='cuda') for g in GROUPS + ['all']}
                firsts = []
                for i, r in enumerate(rs):
                    g, first = after(r, tok)
                    rw[i, [off[i] + j for j in r['up'][tok]]] = True
                    for k, v in g.items():
                        gm[k][i, [off[i] + j for j in v]] = True
                        gm['all'][i, [off[i] + j for j in v]] = True
                    firsts.append(off[i] + first)
                info[tok] = {'rw': rw, 'gm': gm, 'P': min(firsts)}
            spec = {}
            for name, tok, parts in arms:
                gm, ms = info[tok]['gm'], []
                for kind, groups, blocks in parts:
                    sel = t.zeros((N, length), dtype=t.bool, device='cuda')
                    for g in groups:
                        sel |= gm[g]
                    ms.append((sel if kind == 'block' else gm['all'] & ~sel, {int(b): v for b, v in blocks.items()}))   # reader masks
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
                        m = (w * info[tok]['rw'][lo:hi, None, None, :]).sum(-1)
                        rwt = edge[lo:hi, None, P:].float()
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
                            c = (cm[lo:hi, P:, None] & info[tok]['rw'][lo:hi, None, :])[:, None]
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
        return [r for r in all_materials()['pilot_read'] if int(r['family'][-4:]) < 2]
    checked()
    return D.read(OUT / 'pilot_read-materials.json')


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
    save(f'{name.replace("_", "-")}.json', {'schema': f'source_binding_readers_{name}.v1', 'run_id': RUN, 'summary': s, **body, **meta})
    print(json.dumps({'noop_max_score_difference': meta['noop_max_score_difference'], 'seconds': meta['seconds'], **body})[:6000], flush=True)
    return body


def prev(name, smoke, fn):
    return fn(True) if smoke else D.read(OUT / f'{name}.json')


ALL = {b: None for b in range(NB)}


def outside(lo, hi):
    return {b: None for b in range(NB) if not lo <= b < hi}


def regime_e(c):
    return [('only', c['edge'], ALL)]


def only(K, c):
    """Regime W and only the heads of K can use the edge."""
    keep = {}
    for b, h in K:
        keep.setdefault(b, set()).add(h)
    return regime_e(c) + [('block', c['edge'], {b: ([h for h in range(NH) if h not in keep[b]] if b in keep else None) for b in range(NB)})]


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
            out[tok] = {'order': order, 'block_only': {g: [keep(f'{tok}:block:{g}'), keep(f'{tok}:only:{g}')] for g in order}}
        return out
    return finish('pilot_a', rows, [(f'{tok}:{k}:{g}', tok, [(k, [g], ALL)]) for tok in TOKENS for g in GROUPS for k in ('block', 'only')],
                  smoke, extra)


def pilot_b(smoke=False):
    rows = pilot_rows(smoke)
    a = prev('pilot-a', smoke, pilot_a)

    def extra(keep, margin, _):
        out = {}
        for tok in TOKENS:
            k = next(n for n in range(1, KG + 1) if keep(f'{tok}:only:G{n}') >= KEEP or n == KG)
            out[tok] = {'edge': a[tok]['order'][:k], 'cumulative': {f'G{n}': [keep(f'{tok}:block:G{n}'), keep(f'{tok}:only:G{n}')]
                                                                   for n in range(1, KG + 1)}}
        return out
    arms = [(f'{tok}:{kind}:G{n}', tok, [(kind, a[tok]['order'][:n], ALL)]) for tok in TOKENS for n in range(1, KG + 1) for kind in ('block', 'only')]
    return finish('pilot_b', rows, arms, smoke, extra)


def pilot_c(smoke=False):
    rows = pilot_rows(smoke)
    b = prev('pilot-b', smoke, pilot_b)

    def extra(keep, margin, _):
        out = {}
        for tok in TOKENS:
            table = {str(l): [keep(f'{tok}:from@{l}'), keep(f'{tok}:until@{l}')] for l in TIMING}
            l0 = max([l for l in TIMING if table[str(l)][0] >= KEEP], default=None)
            out[tok] = {'edge': b[tok]['edge'], 'timing_from_until': table, 'L0': l0 if l0 is not None else 0, 'L0_rule_met': l0 is not None}
        return out
    arms = []
    for tok in TOKENS:
        for l in TIMING:
            arms += [(f'{tok}:from@{l}', tok, regime_e(b[tok]) + [('block', b[tok]['edge'], outside(l, NB))]),
                     (f'{tok}:until@{l}', tok, regime_e(b[tok]) + [('block', b[tok]['edge'], outside(0, l))])]
    return finish('pilot_c', rows, arms, smoke, extra)


def pilot_d(smoke=False):
    rows = pilot_rows(smoke)
    c = prev('pilot-c', smoke, pilot_c)

    def extra(keep, margin, _):
        out = {}
        for tok in TOKENS:
            l0 = c[tok]['L0']
            d = next((d for d in ENDS if keep(f'{tok}:win{d}') >= KEEP), None)
            m = min(l0 + d, NB) if d is not None else NB
            out[tok] = {'edge': c[tok]['edge'], 'L0': l0, 'M': m, 'M_rule_met': d is not None, 'window': list(range(l0, min(m, l0 + 8, NB))),
                        'window_keep': {str(d): keep(f'{tok}:win{d}') for d in ENDS}}
        return out
    arms = [(f'{tok}:win{d}', tok, regime_e(c[tok]) + [('block', c[tok]['edge'], outside(c[tok]['L0'], c[tok]['L0'] + d))])
            for tok in TOKENS for d in ENDS]
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
    body = {'schema': 'source_binding_readers_frozen.v1', 'run_id': RUN, 'prereg_sha256': D.sha(OUT / 'prereg.json'),
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
    rows = D.read(OUT / 'holdout_read-materials.json')
    results, _, meta = engine(rows, stage='gate')
    save('gate-items.json', results['base'])
    g = D.behavior(rows, results['base'])
    save('gate.json', {'schema': 'source_binding_readers_gate.v1', 'run_id': RUN, 'items_sha256': D.sha(OUT / 'gate-items.json'), **g, **meta})
    print(json.dumps({k: g[k] for k in ('passed', 'forced_accuracy', 'top1_accuracy')}), flush=True)


def run():
    from tools import provenance
    pre, fz = checked(), D.read(OUT / 'frozen.json')
    assert fz['prereg_sha256'] == D.sha(OUT / 'prereg.json') and fz['pilot_f_sha256'] == D.sha(OUT / 'pilot-f.json')
    assert not (OUT / 'report.json').exists(), 'holdout already used'
    g = D.read(OUT / 'gate.json')
    rows = D.read(OUT / 'holdout_read-materials.json')
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
    report = {'schema': 'source_binding_readers_report.v1', 'run_id': RUN, 'split': 'holdout_read', 'gate_passed': g['passed'],
              'frozen': {tok: fz[tok] for tok in TOKENS}, 'prereg_sha256': D.sha(OUT / 'prereg.json'), 'frozen_sha256': D.sha(OUT / 'frozen.json'),
              'gate_sha256': D.sha(OUT / 'gate.json'), 'items_sha256': D.sha(OUT / 'items.json') if g['passed'] else None,
              'criteria': pre['criteria'], 'verdicts': v, 'summary': s, **meta, 'formal_evidence_promotion': 0,
              'custody': 'holdout_read read once by the gate stage and once by this run',
              'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    save('report.json', report)
    path = provenance.write_result('SOURCE-BINDING-readers-holdout', {'verdicts': v, 'gate_passed': g['passed'],
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
