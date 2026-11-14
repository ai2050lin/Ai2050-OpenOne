"""Heads of the end-of-turn relay in the Chinese source-binding task (Claude Phase 434).  Qwen3-14B BF16.

Phase 433: in the zh task the exchanged question names reach the answer mainly through the five end-of-turn template
tokens E = <|im_end|> nl <|im_start|> assistant nl (only E free: agreement 0.954; direct reading only: 0.792).  Here the
relay is split into a WRITE step (E positions read the names) and a READ step (the answer position reads E), located by
token, by block window and by head.

All arms restrict the broad edit as in Phase 433 (positions outside the free set are reset to the base run after every
block).  On top, 'hide' specs make chosen query rows see BASE keys / values at chosen key positions, in chosen blocks and
heads:  write hide = E rows do not see the new names; read hide = the answer row does not see the new E states;
direct hide = the answer row does not see the new names.
Stages: smoke, pilot1 (tokens and block windows), pilot2 (heads), pilot3 (sparse sets) on the opened zh panel of
Phase 433 -> prereg -> gate -> run on a new zh panel.
"""
import argparse, gc, json, sys, time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_binding_design as D
import source_binding_relay as RL
import source_binding_arbitration as AR
import source_binding_fidelity3 as F3
import source_binding_h2518 as H25
import source_binding_readheads as RH
import source_binding_transfer as TR
import source_binding_zhroute as Z
from source_binding_runtime import Runtime

SRC, MODEL = D.OUT, 'qwen14'
OUT = D.ROOT / 'tests/glm5/result/source_binding_eotheads_v1'
RUN, SEED = 'SOURCE-BINDING-EOTHEADS-20261113-v1', 202611130434
CHUNK, NB, NH, TOL, REPS = 96, 40, 40, 1.0, 2
CELLS = ['qsrc@23', 'qent@23', 'names@23']
PART = ('mid', 'qtail', 'instr', 'end0', 'end1', 'end2', 'end3', 'end4', 'think', 'cue')
ALIAS = {'end': ('end0', 'end1', 'end2', 'end3', 'end4')}
E_FROM_END = 11            # E = the 5 positions length-11 .. length-7 (all rows end at the last position: left padding)
WINDOWS = {'w23-25': range(23, 26), 'w26-28': range(26, 29), 'w29-31': range(29, 32), 'w32-35': range(32, 36), 'w36-39': range(36, 40)}


def save(name, body):
    from deploy.storage_policy import check_write
    p = OUT / name
    check_write(p, 1 << 20, operation='来源绑定结束标记段读取头产物')
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(body, ensure_ascii=False, indent=1, default=lambda x: x.item() if isinstance(x, np.generic) else x), encoding='utf-8')
    t.replace(p)


def log(*a, **kw):
    RL.OUT = OUT
    RL.log(*a, **kw)


def annotate(rows):
    rows = Z.annotate(rows)
    for r in rows:
        e = r['grp'].pop('end')
        assert e == list(range(len(r['tokens']) - E_FROM_END, len(r['tokens']) - E_FROM_END + 5))
        r['grp'].update({f'end{j}': [x] for j, x in enumerate(e)})
        assert set(r['grp']) == set(PART)
    return rows


def expand(gs):
    return [x for g in gs for x in ALIAS.get(g, (g,))]


def arm(site, l, base, free, hide=None):
    """hide: {block: [[query 'E' | 'final', key 'names' | 'E', heads or None], ...]}"""
    return {'swap': [site, l], **base, 'free': expand(free), 'hide': {int(b): v for b, v in (hide or {}).items()}}


def engine(rows, arms=None, stage='run'):
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    t, F = torch, torch.nn.functional
    assert snapshot(D.MODELS[MODEL]) == D.read(SRC / f'{MODEL}-identity.json')['model_snapshot']
    start, total = time.monotonic(), len(rows)
    results = {}
    site_of = {'qsrc': lambda r: sorted(r['up']['qsrc']), 'qent': lambda r: sorted(r['up']['qent']), 'names': AR.names_of}
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
            E0 = length - E_FROM_END
            plan, cache = {}, {}
            if arms:
                P0 = min(off[i] + AR.names_of(r)[0] for i, r in enumerate(rs))
                nm = tuple(t.tensor(v, device='cuda') for v in ([i for i, r in enumerate(rs) for _ in AR.names_of(r)],
                                                               [off[i] + x for i, r in enumerate(rs) for x in AR.names_of(r)]))
                for name, a in arms.items():
                    local = [j - c0 for j in a['donor'][c0:c0 + N]]
                    assert all(0 <= j < N and rs[j]['family'] == r['family'] for j, r in zip(local, rs))
                    site, l = a['swap']
                    key = (site, tuple(local))
                    if key not in cache:
                        fn = site_of[site]
                        cache[key] = tuple(t.tensor(v, device='cuda') for v in (
                            [i for i, r in enumerate(rs) for _ in fn(r)], [off[i] + x for i, r in enumerate(rs) for x in fn(r)],
                            [j for j in local for _ in fn(rs[j])], [off[j] + y for j in local for y in fn(rs[j])]))
                    frozen = tuple(g for g in PART if g not in a['free'])
                    fz = None
                    if frozen:
                        if ('fz', frozen) not in cache:
                            cache[('fz', frozen)] = tuple(t.tensor(v, device='cuda') for v in (
                                [i for i, r in enumerate(rs) for g in frozen for _ in r['grp'][g]],
                                [off[i] + x for i, r in enumerate(rs) for g in frozen for x in r['grp'][g]]))
                        fz = cache[('fz', frozen)]
                    hide = {b: [(qg, kg, None if hs is None else t.tensor(hs, device='cuda')) for qg, kg, hs in v] for b, v in a.get('hide', {}).items()}
                    plan[name] = {'first': l, 'P': P0, 'idx': cache[key], 'fz': fz, 'hide': hide}
            state, cap = {}, {}
            for b in range(NB):
                for name, p in plan.items():
                    if p['first'] == b:
                        state[name] = h[:, p['P']:].clone()
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
                kB = vB = None
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
                    kr = rot(k, cos[:, None], sin[:, None])
                    o = F.scaled_dot_product_attention(q, kr, v, attn_mask=mask[:, :, P:, :])
                    if b in p['hide']:
                        if kB is None:          # base keys / values of this block (rotated keys)
                            kB = rot(rt.mod.repeat_kv(a.k_norm(kb.view(N, length, kv, hd)).transpose(1, 2), grp), cos[:, None], sin[:, None])
                            vB = rt.mod.repeat_kv(vb.view(N, length, kv, hd).transpose(1, 2), grp)
                        for qg, kg, hs in p['hide'][b]:
                            k2, v2 = kr.clone(), v.clone()
                            if kg == 'names':
                                bi, bp = nm
                            else:
                                bi = t.arange(N, device='cuda').repeat_interleave(5)
                                bp = t.arange(E0, E0 + 5, device='cuda').repeat(N)
                            if hs is None:
                                k2[bi, :, bp] = kB[bi, :, bp]
                                v2[bi, :, bp] = vB[bi, :, bp]
                            else:
                                k2[bi[:, None], hs[None], bp[:, None]] = kB[bi[:, None], hs[None], bp[:, None]]
                                v2[bi[:, None], hs[None], bp[:, None]] = vB[bi[:, None], hs[None], bp[:, None]]
                            rows_q = slice(E0 - P, E0 - P + 5) if qg == 'E' else slice(T - 1, T)
                            o[:, :, rows_q] = F.scaled_dot_product_attention(q[:, :, rows_q], k2, v2, attn_mask=mask[:, :, P:, :][:, :, rows_q])
                            del k2, v2
                    x = x + a.o_proj(o.transpose(1, 2).reshape(N, T, -1))
                    x = x + layer.mlp(layer.post_attention_layernorm(x))
                    if p['fz'] is not None:
                        fi, fp = p['fz']
                        x[fi, fp - P] = y[fi, fp]
                    state[name] = x
                    del q, k, v, o, kr
                h = y
                for x in handles:
                    x.remove()
                cap.clear()
                del layer, y, a, kb, vb, kB, vB
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
        ctl = [k for k in ('self', 'self:hide') if k in results]
        meta.update({f'{k}_max_score_difference': diff(k) for k in ctl})
        meta['controls_ok'] = bool(all(same(k) and diff(k) <= TOL for k in ctl))
    return results, meta


ALLB = range(23, NB)


def pilot1_arms(site, l, base):
    out = {'swap': arm(site, l, base, PART), 'direct': arm(site, l, base, []), 'via:end': arm(site, l, base, ['end']),
           'nodirect': arm(site, l, base, ['end'], {b: [['final', 'names', None]] for b in ALLB})}
    for j in range(5):
        out[f'tok{j}'] = arm(site, l, base, [f'end{j}'])
    for w, bs in WINDOWS.items():
        out[f'wonly:{w}'] = arm(site, l, base, ['end'], {b: [['E', 'names', None]] for b in ALLB if b not in bs})
        out[f'ronly:{w}'] = arm(site, l, base, ['end'], {b: [['final', 'E', None]] for b in ALLB if b not in bs})
    return out


def make_arms(rows, key, builder, cells=CELLS, reps=REPS):
    dons, rels = RH.donors(rows, f'{RUN}|{key}', SEED)
    me = list(range(len(rows)))
    arms = {'self': arm('names', 23, {'donor': me}, PART),
            'self:hide': arm('names', 23, {'donor': me}, ['end'], {b: [['E', 'names', None], ['final', 'E', None]] for b in (25, 30)})}
    for k in range(reps):
        base = {'donor': dons[k], 'rel': rels[k]}
        for cell in cells:
            site, l = cell.split('@')[0], int(cell.split('@')[1])
            for n, a in builder(site, l, base).items():
                arms[f'{n}|{cell}#{k}'] = a
    return arms


def table(rows, arms, items):
    fam_id = {f: i for i, f in enumerate(sorted({r['family'] for r in rows}))}
    names = sorted({n.split('|')[0] for n in arms if '|' in n and not n.startswith('swap')})
    reps = len({n.split('#')[1] for n in arms if n.startswith('swap|')})
    T = defaultdict(list)
    for cell in sorted({n.split('|')[1].split('#')[0] for n in arms if n.startswith('swap|')}):
        site, l = cell.split('@')[0], int(cell.split('@')[1])
        for k in range(reps):
            a = arms[f'swap|{cell}#{k}']
            for i, (r, j) in enumerate(zip(rows, a['donor'])):
                B, P, I = F3.records(r, rows[j], site, l)
                o = items[f'swap|{cell}#{k}'][i]
                T['fam'].append(fam_id[r['family']]); T['cell'].append(cell); T['B'].append(B); T['P'].append(P); T['out'].append(o['pred_record'])
                bs = items['base'][i]['scores']
                T['nat'].append(bs[P] - bs[B] if P != B else 0.0)
                T['xs'].append(o['scores'][P] - o['scores'][B] if P != B else 0.0)
                for n in names:
                    it = items[f'{n}|{cell}#{k}'][i]
                    T[n].append(it['pred_record'])
                    T['x:' + n].append(it['scores'][P] - it['scores'][B] if P != B else 0.0)
    return {k: np.array(v) for k, v in T.items()}, len(fam_id), names


def summarise(results, arms, rows, pairs=()):
    T, nf, names = table(rows, arms, results)
    draws = np.random.default_rng(SEED).integers(0, nf, (1000, nf))
    fam, out, B = T['fam'], T['out'], T['B']
    r4 = lambda v: round(float(v), 4)

    def rate(v, keep):
        v = np.asarray(v, float)
        per, cnt = np.bincount(fam[keep], weights=v[keep], minlength=nf), np.bincount(fam[keep], minlength=nf)
        bs = per[draws].sum(1) / np.maximum(cnt[draws].sum(1), 1)
        return {'rate': r4(v[keep].mean()), 'ci95': [r4(np.quantile(bs, .025)), r4(np.quantile(bs, .975))]}

    def share(num, keep):
        n_ = np.bincount(fam[keep], weights=num[keep], minlength=nf)
        d_ = np.bincount(fam[keep], weights=(T['xs'] - T['nat'])[keep], minlength=nf)
        bs = n_[draws].sum(1) / d_[draws].sum(1)
        return {'rate': r4(n_.sum() / d_.sum()), 'ci95': [r4(np.quantile(bs, .025)), r4(np.quantile(bs, .975))]}
    hit = {n: (T[n] == out).astype(float) for n in names}
    hit['none'] = (B == out).astype(float)
    conflict = T['P'] != B
    scopes = {'all': np.ones(len(out), bool), **{c: T['cell'] == c for c in sorted(set(T['cell']))}}
    s = {'edits': int(len(out)), 'scopes': {}}
    for name, keep in scopes.items():
        c = keep & conflict
        s['scopes'][name] = {'n': int(keep.sum()), 'broad_changed': r4((out != B)[keep].mean()),
                             'agreement': {n: rate(v, keep) for n, v in hit.items()},
                             'shift_share': {n: share(T['x:' + n] - T['nat'], c) for n in names},
                             'diff': {f'{a}-{b}': rate(hit[a] - hit[b], keep) for a, b in pairs if a in hit and b in hit}}
    return s


def show(s, ks):
    lines = ['scope      chg    ' + ' '.join(f'{k[:10]:>10s}' for k in ks)]
    for n, v in s['scopes'].items():
        lines.append(f"{n:10s} {v['broad_changed']:.3f}  " + ' '.join(f"{v['agreement'][k]['rate']:10.3f}" for k in ks))
        lines.append(f"{'  share':10s}        " + ' '.join(f"{v['shift_share'][k]['rate']:10.3f}" if k in v['shift_share'] else ' ' * 10 for k in ks))
    return '\n'.join(lines)


def opened_rows(first=None):
    rows = D.read(Z.OUT / 'zh-confirm-materials.json')
    fams = sorted({r['family'] for r in rows})
    keep = set(fams[:first] if first and first > 0 else fams[first:] if first else fams)
    return annotate([r for r in rows if r['family'] in keep])


def smoke():
    from deploy.storage_policy import checked_dir
    checked_dir(OUT, 1 << 30, operation='来源绑定结束标记段读取头产物')
    rows = annotate([r for r in D.read(TR.OUT / 'zh-pilot-materials.json') if int(r['family'][-4:]) < 2])
    arms = make_arms(rows, 'smoke', pilot1_arms, ['names@23'], 1)
    results, meta = engine(rows, arms, stage='smoke')
    s = summarise(results, arms, rows)
    print(json.dumps(meta), flush=True)
    print(show(s, ['direct', 'via:end', 'nodirect', 'tok0', 'tok3', 'tok4', 'wonly:w23-25', 'ronly:w36-39', 'none']), flush=True)


def pilot1():
    """Opened zh panel of Phase 433 (first 24 families): tokens, write windows, read windows."""
    from deploy.storage_policy import checked_dir
    checked_dir(OUT, 1 << 30, operation='来源绑定结束标记段读取头产物')
    rows = opened_rows(24)
    arms = make_arms(rows, 'pilot1', pilot1_arms)
    results, meta = engine(rows, arms, stage='pilot1')
    s = summarise(results, arms, rows)
    save('pilot1-summary.json', {'meta': meta, **s})
    print(json.dumps(meta), flush=True)
    print(show(s, ['direct', 'via:end', 'nodirect'] + [f'tok{j}' for j in range(5)] + ['none']), flush=True)
    print(show(s, [f'wonly:{w}' for w in WINDOWS] + [f'ronly:{w}' for w in WINDOWS]), flush=True)


# ---- after pilot 1: the relay token is end1 (the newline after <|im_end|>); writing happens in blocks 23-25 ----
WB = (23, 24, 25)
RB = range(23, NB)


def pilot2_arms(site, l, base):
    out = {'swap': arm(site, l, base, PART), 'direct': arm(site, l, base, []), 'tok1': arm(site, l, base, ['end1'])}
    for b in WB:
        for hh in range(NH):
            out[f'wloo:{b}.{hh}'] = arm(site, l, base, ['end1'], {b: [['E', 'names', [hh]]]})
        out[f'wdrop:{b}'] = arm(site, l, base, ['end1'], {b: [['E', 'names', None]]})
    for b in RB:
        out[f'rdrop:{b}'] = arm(site, l, base, ['end1'], {b: [['final', 'E', None]]})
    for name, bs in {'r26-31': range(26, 32), 'r26-35': range(26, 36), 'r32-39': range(32, 40), 'r26-39': range(26, 40), 'r23-39': RB}.items():
        out[f'ronly:{name}'] = arm(site, l, base, ['end1'], {b: [['final', 'E', None]] for b in RB if b not in bs})
    return out


def pilot2():
    """Opened zh panel of Phase 433 (the other 24 families): leave-one-out of the 120 heads of blocks 23-25 at the relay
    token (write) and single-block removal of the answer row's reading of it (read)."""
    global CHUNK
    CHUNK = 48
    rows = opened_rows(-24)
    arms = make_arms(rows, 'pilot2', pilot2_arms, ['names@23', 'qent@23'], 1)
    results, meta = engine(rows, arms, stage='pilot2')
    T, nf, names = table(rows, arms, results)
    out, conflict = T['out'], T['P'] != T['B']
    full = (T['x:tok1'] - T['nat'])[conflict].mean()
    loss = lambda n: round(float((T['x:tok1'] - T['x:' + n])[conflict].mean() / full), 4)
    agree = lambda n: round(float((T[n] == out).mean()), 4)
    res = {'meta': meta, 'edits': int(len(out)), 'agreement': {n: agree(n) for n in names}, 'none': round(float((T['B'] == out).mean()), 4),
           'loss_share': {n: loss(n) for n in names if n not in ('swap',)}}
    save('pilot2.json', res)
    L = res['loss_share']
    print(json.dumps(meta), flush=True)
    print('agreement', {n: res['agreement'][n] for n in ['direct', 'tok1'] + [f'wdrop:{b}' for b in WB] + [n for n in names if n.startswith('ronly')]}, 'none', res['none'], flush=True)
    print('write loo top', sorted(((L[n], n) for n in L if n.startswith('wloo')), reverse=True)[:20], flush=True)
    print('write loo bottom', sorted(((L[n], n) for n in L if n.startswith('wloo')))[:5], flush=True)
    print('write block', {b: L[f'wdrop:{b}'] for b in WB}, flush=True)
    print('read block', {b: L[f'rdrop:{b}'] for b in RB}, flush=True)


# ---- after pilot 2: the heads that write at the relay token are the block 23-25 members of SB.H.read16 ----
W13 = {23: [36, 37, 38], 24: [17, 19, 33, 36], 25: [15, 16, 18, 24, 38]}
RANDOM_SETS = 3


def random_sets(spec, n=RANDOM_SETS, tag='w'):
    rng = np.random.default_rng(SEED + (0 if tag == 'w' else 1))
    return [{b: sorted(int(x) for x in rng.choice([x for x in range(NH) if x not in hs], len(hs), replace=False)) for b, hs in spec.items()} for _ in range(n)]


def write_only(heads):
    """E rows see the new names only through these heads (blocks 23-25); nowhere else."""
    return {b: [['E', 'names', [x for x in range(NH) if x not in heads.get(b, [])] or None]] for b in RB if heads.get(b) != list(range(NH))}


def read_only(heads):
    """The answer row sees the new relay state only through these heads (by block); all heads where the value is 'all'."""
    out = {}
    for b in RB:
        hs = heads.get(b)
        if hs == 'all':
            continue
        out[b] = [['final', 'E', [x for x in range(NH) if x not in (hs or [])] or None]]
    return out


def pilot3_arms(site, l, base):
    out = {'swap': arm(site, l, base, PART), 'direct': arm(site, l, base, []), 'tok1': arm(site, l, base, ['end1']),
           'w:W13': arm(site, l, base, ['end1'], write_only(W13))}
    for j, rs_ in enumerate(random_sets(W13)):
        out[f'w:R{j}'] = arm(site, l, base, ['end1'], write_only(rs_))
    for b in (27, 30):
        for hh in range(NH):
            out[f'rloo:{b}.{hh}'] = arm(site, l, base, ['end1'], {b: [['final', 'E', [hh]]]})
    out['rdrop:27+30'] = arm(site, l, base, ['end1'], {b: [['final', 'E', None]] for b in (27, 30)})
    out['ronly:27+30+34-39'] = arm(site, l, base, ['end1'], read_only({27: 'all', 30: 'all', **{b: 'all' for b in range(34, 40)}}))
    return out


def pilot3():
    """Opened zh panel of Phase 433 (first 24 families, new donor draws): sparse write set; read leave-one-out in blocks 27, 30."""
    global CHUNK
    CHUNK = 48
    rows = opened_rows(24)
    arms = make_arms(rows, 'pilot3', pilot3_arms, ['names@23', 'qent@23'], 1)
    results, meta = engine(rows, arms, stage='pilot3')
    T, nf, names = table(rows, arms, results)
    out, conflict = T['out'], T['P'] != T['B']
    full = (T['x:tok1'] - T['nat'])[conflict].mean()
    loss = lambda n: round(float((T['x:tok1'] - T['x:' + n])[conflict].mean() / full), 4)
    res = {'meta': meta, 'edits': int(len(out)), 'agreement': {n: round(float((T[n] == out).mean()), 4) for n in names},
           'none': round(float((T['B'] == out).mean()), 4), 'loss_share': {n: loss(n) for n in names if n != 'swap'}}
    save('pilot3.json', res)
    L, A = res['loss_share'], res['agreement']
    print(json.dumps(meta), flush=True)
    print('agreement', {n: A[n] for n in names if not n.startswith('rloo')}, 'none', res['none'], flush=True)
    print('loss', {n: L[n] for n in names if not n.startswith('rloo') and n != 'swap'}, flush=True)
    print('read loo top', sorted(((L[n], n) for n in L if n.startswith('rloo')), reverse=True)[:16], flush=True)
    print('read loo bottom', sorted(((L[n], n) for n in L if n.startswith('rloo')))[:5], flush=True)


# ---- after pilot 3: the answer row reads the relay token mainly through 27.11 and 30.35 / 30.38 / 30.39 ----
R4 = {27: [11], 30: [35, 38, 39]}
LATE = {b: 'all' for b in range(34, 40)}
FCELLS = [f'{s}@{l}' for s in ('qsrc', 'qent', 'names') for l in (23, 24)]


def final_arm_set(site, l, base):
    rr = random_sets(R4, tag='r')
    out = {'swap': arm(site, l, base, PART), 'direct': arm(site, l, base, []), 'via:end': arm(site, l, base, ['end']),
           'tok1': arm(site, l, base, ['end1']), 'w:W13': arm(site, l, base, ['end1'], write_only(W13)),
           'r:R4+late': arm(site, l, base, ['end1'], read_only({**R4, **LATE})), 'r:R4': arm(site, l, base, ['end1'], read_only(R4)),
           'r:late': arm(site, l, base, ['end1'], read_only(LATE))}
    for j, rs_ in enumerate(random_sets(W13)):
        out[f'w:R{j}'] = arm(site, l, base, ['end1'], write_only(rs_))
    for j, rs_ in enumerate(rr):
        out[f'r:RR{j}+late'] = arm(site, l, base, ['end1'], read_only({**rs_, **LATE}))
    both = {**write_only(W13)}
    for b, v in read_only({**R4, **LATE}).items():
        both[b] = both.get(b, []) + v
    out['wr:W13,R4+late'] = arm(site, l, base, ['end1'], both)
    return out


PAIRS = [('tok1', 'direct'), ('via:end', 'direct'), ('w:W13', 'direct'), ('tok1', 'w:W13'), ('r:R4+late', 'direct'), ('tok1', 'r:R4+late'),
         ('r:R4+late', 'r:late'), ('wr:W13,R4+late', 'direct'), ('tok1', 'wr:W13,R4+late'), ('direct', 'none')]


def pilot4():
    """The final arms on the opened zh panel of Phase 433 (other 24 families, new donor draws); thresholds are set on this."""
    global CHUNK
    CHUNK = 48
    rows = opened_rows(-24)
    arms = make_arms(rows, 'pilot4', final_arm_set, FCELLS, REPS)
    results, meta = engine(rows, arms, stage='pilot4')
    s = summarise(results, arms, rows, PAIRS)
    save('pilot4-summary.json', {'meta': meta, **s})
    print(json.dumps(meta), flush=True)
    ks = ['direct', 'via:end', 'tok1', 'w:W13', 'w:R0', 'r:R4+late', 'r:R4', 'r:late', 'r:RR0+late', 'r:RR1+late', 'wr:W13,R4+late', 'none']
    print(show(s, ks), flush=True)
    print(json.dumps({k: v for k, v in s['scopes']['all']['diff'].items()}), flush=True)


NEW_POOL = {'sources': ['靛队', '米队', '墨队', '碧队', '朱队'],
            'items': ['方圆', '石磊', '江涛', '白雪', '黎明', '秦岚', '魏东', '薛凯', '程曦', '苏晴']}


def build_new():
    used = set()
    for split in ('pilot', 'holdout'):
        used |= set(TR.POOLS['zh']['sources'][split] + TR.POOLS['zh']['items'][split])
    used |= set(Z.CONFIRM['zh']['sources'] + Z.CONFIRM['zh']['items'])
    assert not set(NEW_POOL['sources'] + NEW_POOL['items']) & used
    TR.POOLS['zh']['sources']['new'] = NEW_POOL['sources']
    TR.POOLS['zh']['items']['new'] = NEW_POOL['items']
    TR.FAMILIES['new'] = 48
    rows = annotate(TR.build('zh', 'new'))
    H25.RUN, H25.SEED = RUN, SEED
    H25.draws(rows)
    return rows


def materials():
    rows = build_new()
    print(len(rows), 'tokens', min(len(r['tokens']) for r in rows), max(len(r['tokens']) for r in rows),
          'qsrc', sorted({len(r['up']['qsrc']) for r in rows}), 'qent', sorted({len(r['up']['qent']) for r in rows}), flush=True)



# ---- frozen after pilot 4 ----
THRESHOLDS = {
    'K1_one_relay_token': 'only the newline after <|im_end|> free (tok1): agreement >= 0.93 and minus direct >= 0.12 with a lower bound > 0; '
                          'all five end-of-turn tokens free differ from tok1 by <= 0.01. not_supported if the tok1 lower bound <= 0.',
    'K2_write_heads': 'the relay token reads the new names only through the 13 heads 23.36-38, 24.17/19/33/36, 25.15/16/18/24/38 (blocks '
                      '23-25): agreement >= 0.89 and minus direct >= 0.10 with a lower bound > 0; tok1 exceeds it by <= 0.06; no random '
                      'set of as many heads per block exceeds direct by more than 0.01. not_supported if the lower bound <= 0.',
    'K3_read_heads': 'the answer row reads the relay token only through 27.11, 30.35, 30.38, 30.39 and every head of blocks 34-39: agreement '
                     '>= 0.93; minus blocks 34-39 alone >= 0.07 with a lower bound > 0; random replacements of the four heads exceed '
                     'blocks 34-39 alone by <= 0.02. not_supported if the lower bound <= 0.',
    'K4_sparse_relay': 'both restrictions together: agreement >= 0.89 and minus direct >= 0.10 with a lower bound > 0. not_supported if '
                       'the lower bound <= 0.',
}
CRITERIA = {
    'model': 'Qwen3-14B original BF16, CPU byte reads, CUDA layer streaming. GLM-4-9B is not run.',
    'material': 'the zh task of Phase 426 (Chinese prompt; team names as sources, personal names as items, one-character animals). New '
                'panel: 48 families (768 prompts); source and item pools disjoint from all earlier zh pools (Phases 426, 433); answer '
                'words shared.',
    'broad_edit': 'at the input of block L (23, 24) the residual states of the source name in the question, the item name or both are '
                  'replaced by those of a donor of the same family. 2 seeded donor relations per item; 6 cells x 1,536 edits.',
    'restriction': 'as in Phase 433: from block L on, after every block, positions outside the free set are reset to the base run; the '
                   'names and the answer position are always free. tok1 = only the newline after <|im_end|> free; via:end = the five '
                   'end-of-turn tokens free; direct = nothing free. Head restrictions (blocks >= L, attention rows only): write = the '
                   'end-of-turn rows see base keys / values at the name tokens except through the named heads; read = the answer row sees '
                   'base keys / values at the end-of-turn tokens except through the named heads.',
    'head_sets': {'W13': W13, 'R4': R4, 'late': 'every head of blocks 34-39',
                  'random': '3 seeded sets per side with as many heads per block, drawn from the other heads of the block'},
    'prediction': 'the broad edit gives the answer of the restricted edit; agreement and the share of the score shift on conflict edits; '
                  'family-bootstrap 95 % intervals (1000 draws).',
    'controls': 'the name exchange with the prompt itself, all free and with write / read hides in blocks 25 and 30, changes nothing '
                '(same choice, score difference <= 1.0).',
    'two_pass': 'gate stage: unmodified forward and the v1 behaviour gate. Edits run only if the gate passed.',
    'predictions_on_holdout': THRESHOLDS,
    'not_covered': 'what the write heads put into the relay token and what the read heads take from it (QK / OV are hidden together); '
                   'why the relay matters in zh and little in English; the checking heads of block 27 (27.12, 27.14 oppose the read, '
                   'pilot 3); exchanges at block 25; other models.',
    'disclosure': 'four pilots on the opened zh panel of Phase 433 with new donor draws: pilot 1 (tokens, block windows), pilot 2 '
                  '(leave-one-out of the 120 heads of blocks 23-25 at the relay token; single-block removal of the read), pilot 3 '
                  '(the write set W13, which is the block 23-25 part of SB.H.read16, and leave-one-out of the read in blocks 27 and 30), '
                  'pilot 4 (the final arms). The read set R4 and the thresholds were set on pilots 3 and 4.',
}


def verdicts(s, meta):
    tri = lambda ok, bad: 'supported' if ok else 'not_supported' if bad else 'inconclusive'
    v = {'controls_valid': bool(meta['controls_ok'])}
    if not v['controls_valid']:
        return {**v, **{k: 'controls_invalid' for k in THRESHOLDS}}
    A = s['scopes']['all']
    ag = lambda n: A['agreement'][n]['rate']
    d = A['diff']
    x = d['tok1-direct']
    v['K1_one_relay_token'] = tri(ag('tok1') >= .93 and x['rate'] >= .12 and x['ci95'][0] > 0 and abs(ag('via:end') - ag('tok1')) <= .01, x['ci95'][0] <= 0)
    x = d['w:W13-direct']
    v['K2_write_heads'] = tri(ag('w:W13') >= .89 and x['rate'] >= .10 and x['ci95'][0] > 0 and ag('tok1') - ag('w:W13') <= .06
                              and max(ag(f'w:R{j}') for j in range(RANDOM_SETS)) - ag('direct') <= .01, x['ci95'][0] <= 0)
    x = d['r:R4+late-r:late']
    v['K3_read_heads'] = tri(ag('r:R4+late') >= .93 and x['rate'] >= .07 and x['ci95'][0] > 0
                             and max(ag(f'r:RR{j}+late') for j in range(RANDOM_SETS)) - ag('r:late') <= .02, x['ci95'][0] <= 0)
    x = d['wr:W13,R4+late-direct']
    v['K4_sparse_relay'] = tri(ag('wr:W13,R4+late') >= .89 and x['rate'] >= .10 and x['ci95'][0] > 0, x['ci95'][0] <= 0)
    return v


DEPS = (RL, AR, F3, H25, RH, TR, Z, RH.S)


def prereg():
    from deploy.storage_policy import checked_dir
    from tools import provenance
    from tools.power_analysis import check_phase
    checked_dir(OUT, 1 << 30, operation='来源绑定结束标记段读取头产物')
    assert not (OUT / 'prereg.json').exists()
    plan = D.checked(MODEL)
    rows = build_new()
    arms = make_arms(rows, 'new', final_arm_set, FCELLS, REPS)
    save('zh-new-materials.json', rows)
    save('design.json', arms)
    body = {'schema': 'source_binding_eotheads_prereg.v1', 'run_id': RUN, 'seed': SEED, 'created_at': datetime.now().astimezone().isoformat(),
            'base_plan_sha256': D.sha(SRC / 'plan.json'), 'model_snapshot': plan['models'][MODEL], 'code_sha256': D.sha(__file__),
            'dependency_sha256': {m.__name__: D.sha(m.__file__) for m in DEPS},
            'criteria': CRITERIA, 'frozen': {'W13': W13, 'R4': R4, 'random_write': random_sets(W13), 'random_read': random_sets(R4, tag='r'),
                                             'pilot_sha256': {f: D.sha(OUT / f) for f in ('pilot1-summary.json', 'pilot2.json', 'pilot3.json', 'pilot4-summary.json')}},
            'materials': {'sha256': D.sha(OUT / 'zh-new-materials.json'), 'families': 48, 'rows': len(rows)},
            'design': {'sha256': D.sha(OUT / 'design.json'), 'arms': len(arms)},
            'power': {'unit': 'paired binary outcome per item (prediction right / wrong under the sparse relay and under direct); alpha = 0.05',
                      'K4_as_if_one_edit_per_item': {'expected_delta': 0.12, 'p_d': 0.20, **check_phase(0.12, len(rows), p_d=0.20)},
                      'caveat': '9,216 edits clustered in 768 items and 48 families; intervals are family bootstraps'},
            'custody': 'the new panel is read first by the gate stage (behaviour only) and then once by the run stage.',
            'resources': {'gpu_minutes_estimate': 22, 'disk_GiB_estimate': 0.1, 'models_concurrent': 1,
                          'shape': 'chunks of 48 prompts in one batch; pilot 4 peak 4.5 GiB, 10.5 minutes for 24 families'},
            'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    body['provenance']['model_revision'] = plan['models'][MODEL]
    save('prereg.json', body)
    print(json.dumps({'prereg_sha256': D.sha(OUT / 'prereg.json'), 'design': body['design'], 'power': body['power'],
                      'pilot4_verdicts': verdicts(D.read(OUT / 'pilot4-summary.json'), {'controls_ok': True})}), flush=True)


def checked():
    p = D.read(OUT / 'prereg.json')
    assert p['code_sha256'] == D.sha(__file__) and all(p['dependency_sha256'][m.__name__] == D.sha(m.__file__) for m in DEPS), \
        'source changed after pre-registration'
    assert p['materials']['sha256'] == D.sha(OUT / 'zh-new-materials.json') and p['design']['sha256'] == D.sha(OUT / 'design.json')
    D.checked(MODEL)
    return p


def gate():
    checked()
    assert not (OUT / 'gate.json').exists(), 'new panel already read'
    rows = D.read(OUT / 'zh-new-materials.json')
    results, meta = engine(rows, stage='gate')
    save('gate-items.json', results['base'])
    g = D.behavior(rows, results['base'])
    save('gate.json', {'schema': 'source_binding_eotheads_gate.v1', 'run_id': RUN, 'items_sha256': D.sha(OUT / 'gate-items.json'), **g, **meta})
    print(json.dumps({k: g[k] for k in ('passed', 'forced_accuracy', 'top1_accuracy')}), flush=True)


def run():
    global CHUNK
    from tools import provenance
    CHUNK = 48
    pre = checked()
    assert not (OUT / 'report.json').exists(), 'new panel already used'
    g = D.read(OUT / 'gate.json')
    rows, arms = D.read(OUT / 'zh-new-materials.json'), D.read(OUT / 'design.json')
    for a in arms.values():
        a['hide'] = {int(b): v for b, v in a.get('hide', {}).items()}
    if g['passed']:
        results, meta = engine(rows, arms, stage='run')
        save('items.json', {k: [{'id': x['id'], 'pred_record': x['pred_record'], 'scores': x['scores']} for x in v] for k, v in results.items()})
        base = D.read(OUT / 'gate-items.json')
        meta['base_replay_max_score_difference'] = max(abs(a - b) for x, y in zip(results['base'], base) for a, b in zip(x['scores'], y['scores']))
        s = summarise(results, arms, rows, PAIRS)
        v = verdicts(s, meta)
        print(json.dumps(meta), flush=True)
        print(show(s, ['direct', 'via:end', 'tok1', 'w:W13', 'w:R0', 'r:R4+late', 'r:R4', 'r:late', 'r:RR0+late', 'wr:W13,R4+late', 'none']), flush=True)
    else:
        s, meta, v = {'note': 'behaviour gate failed; no edit was run'}, {}, {k: 'behavior_ineligible' for k in THRESHOLDS}
    report = {'schema': 'source_binding_eotheads_report.v1', 'run_id': RUN, 'gate_passed': g['passed'],
              'prereg_sha256': D.sha(OUT / 'prereg.json'), 'gate_sha256': D.sha(OUT / 'gate.json'),
              'items_sha256': D.sha(OUT / 'items.json') if g['passed'] else None, 'criteria': pre['criteria'], 'frozen': pre['frozen'],
              'verdicts': v, 'summary': s, **meta, 'formal_evidence_promotion': 0,
              'custody': 'the new panel read once by the gate stage and once by this run',
              'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    save('report.json', report)
    path = provenance.write_result('SOURCE-BINDING-eotheads-holdout', {'verdicts': v, 'gate_passed': g['passed'],
                                                                     'report_sha256': D.sha(OUT / 'report.json')},
                                   model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)
    print(json.dumps({'verdicts': v, 'result': str(path)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['smoke', 'pilot1', 'pilot2', 'pilot3', 'pilot4', 'materials', 'prereg', 'gate', 'run'])
    {'smoke': smoke, 'pilot1': pilot1, 'pilot2': pilot2, 'pilot3': pilot3, 'pilot4': pilot4, 'materials': materials, 'prereg': prereg, 'gate': gate, 'run': run}[ap.parse_args().stage]()
