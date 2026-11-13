"""Why attention 27 opposes the block-25 write (Claude Phase 421).  Qwen3-14B BF16.

Phase 420: after the narrow edit (heads 25.18, 25.24, 25.15 read donor name tokens and holders at the answer position)
freezing attention 27 to its output of the unedited run nearly doubles the shift of the pointer score.  A frozen
attention 27 still writes what it wrote in the base run, so the opposition is its RESPONSE to the edited stream.
At the answer position the keys and values of all other positions are those of the base run; the response can come
from (a) the attention pattern (the edited query looks elsewhere), (b) the position's own key / value.

Arms, all on top of the narrow edit, all at block 27 and the attention row of the answer position:
  ko          whole attention output frozen (Phase 420)
  pattern     attention weights of the base run, values as computed
  hide:G      a token group is masked (N names, H holders, R records and headers)
  don:G       keys and values of a group are those of the donor (unedited donor run), all heads
  fz:<set>    the outputs of a head set are frozen, the rest responds;  keep:<set>  only the set responds
Broad edit (names exchanged at block 24): swap, swap|hide:N, swap|ko-heads - does attention 27 then help?
Also the attention mass of block 27 by token group, base run against the narrow edit.
Stages: smoke, pilot (opened panel holdout_dn) -> prereg -> gate -> run.
"""
import argparse, gc, json, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_binding_design as D
import source_binding_relay as RL
import source_binding_pointer as PT
import source_binding_nameloc as NL
import source_binding_fetch as FE
import source_binding_arbitration as AR
import source_binding_fidelity3 as F3
import source_binding_h2518 as H25
import source_binding_peredit as PE
import source_binding_downstream as DN
from source_binding_runtime import Runtime

SRC, MODEL = D.OUT, 'qwen14'
OUT = D.ROOT / 'tests/glm5/result/source_binding_attn27_v1'
RUN, SEED = 'SOURCE-BINDING-ATTN27-20261112-v1', 202611120421
SPLIT, FAMILIES = 'holdout_a27', 64
CHUNK, NB, NH, TOL, REPS, L, PB, B27 = 192, 40, 40, 1.0, 4, 24, 25, 27
C = H25.C
COMPS = DN.COMPS
NOT27 = [c for c in COMPS if c != 'a27']
OTHER_PANELS = DN.OTHER_PANELS + ['source_binding_downstream_v1/holdout_dn']
HIDE = {'N': ['names'], 'H': ['holders'], 'NH': ['names', 'holders'], 'R': ['records']}
MASS = ['names', 'holders', 'vals', 'recs']


def save(name, body):
    from deploy.storage_policy import check_write
    p = OUT / name
    check_write(p, 1 << 20, operation='来源绑定注意力 27 产物')
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


GROUPS = {'names': AR.names_of, 'holders': AR.holders_of,
          'records': lambda r: sorted(j for g in ('vals', 'recs', 'hdrs') for j in FE.keys(r)[0][g]),
          'vals': lambda r: FE.keys(r)[0]['vals'], 'recs': lambda r: sorted(FE.keys(r)[0]['recs'] + FE.keys(r)[0]['hdrs'])}


def make_arms(rows, head_sets, run=RUN, seed=SEED):
    """head_sets: name -> heads of block 27."""
    H25.RUN, H25.SEED = run, seed                 # donor draws: the stream of the Phase 415 module contains run id and seed
    dons, rels = H25.draws(rows)
    me = list(range(len(rows)))
    arms = {'self': {'donor': me, 'resp': None, 'rep': 'self'}, 'self:don': {'donor': me, 'resp': None, 'rep': 'self', 'm27': {'donor': ['names', 'holders']}}}
    for k, don in enumerate(dons):
        base = {'donor': don, 'rel': rels[k], 'rep': k}
        arms[f'swap#{k}'] = {'swap': True, 'donor': don, 'rel': rels[k]}
        arms[f'swap|hide:N#{k}'] = {'swap': True, 'donor': don, 'rel': rels[k], 'hide27': ['names']}
        arms[f'swap|hide:NH#{k}'] = {'swap': True, 'donor': don, 'rel': rels[k], 'hide27': ['names', 'holders']}
        arms[f'full#{k}'] = {**base, 'resp': None, 'mass': True}
        arms[f'ko#{k}'] = {**base, 'resp': NOT27}
        arms[f'pattern#{k}'] = {**base, 'resp': None, 'm27': {'pattern': True}}
        for n, g in HIDE.items():
            arms[f'hide:{n}#{k}'] = {**base, 'resp': None, 'm27': {'hide': g}}
        for n in ('N', 'NH'):
            arms[f'don:{n}#{k}'] = {**base, 'resp': None, 'm27': {'donor': HIDE[n]}}
        for n, hs in head_sets.items():
            arms[f'fz:{n}#{k}'] = {**base, 'resp': None, 'm27': {'freeze': list(hs)}}
            if len(hs) > 1:
                arms[f'keep:{n}#{k}'] = {**base, 'resp': None, 'm27': {'freeze': [h for h in range(NH) if h not in hs]}}
    return arms


def engine(rows, arms=None, stage='run'):
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    t, F = torch, torch.nn.functional
    assert snapshot(D.MODELS[MODEL]) == D.read(SRC / f'{MODEL}-identity.json')['model_snapshot']
    start, total = time.monotonic(), len(rows)
    results, mass, mass_n = {}, {}, 0
    with GpuLease(), t.inference_mode():
        rt = Runtime(MODEL)
        hd, kv, grp = rt.hd, rt.kv, rt.heads // rt.kv
        rot = lambda x, co, si: x * co + rt.mod.rotate_half(x) * si
        hs = t.tensor(C, device='cuda')
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
                def flags(fn):
                    m = t.zeros((N, length), dtype=t.bool, device='cuda')
                    for i, r in enumerate(rs):
                        m[i, [off[i] + j for j in fn(r)]] = True
                    return m
                grp_mask = {g: flags(fn) for g, fn in GROUPS.items()}
                P0 = min(off[i] + AR.names_of(r)[0] for i, r in enumerate(rs))
                plan['noop'] = {'first': 0, 'P': length - 1, 'sw': None, 'resp': None, 'pt': None, 'm27': {}}
                for name, a in arms.items():
                    local = [j - c0 for j in a['donor'][c0:c0 + N]]
                    assert all(0 <= j < N and rs[j]['family'] == r['family'] for j, r in zip(local, rs))
                    pairs = lambda fn: tuple(t.tensor(v, device='cuda') for v in (
                        [i for i, r in enumerate(rs) for _ in fn(r)], [off[i] + x for i, r in enumerate(rs) for x in fn(r)],
                        [j for j in local for _ in fn(rs[j])], [off[j] + y for j in local for y in fn(rs[j])]))
                    if a.get('swap'):
                        plan[name] = {'first': L, 'P': P0, 'sw': L, 'idx': pairs(AR.names_of), 'resp': None, 'pt': None, 'hide27': a.get('hide27', [])}
                    else:
                        m27 = dict(a.get('m27') or {})
                        if m27.get('donor'):
                            m27['pairs'] = [pairs(GROUPS[g]) for g in m27['donor']]
                        if m27.get('freeze') is not None:
                            m27['freeze'] = t.tensor(m27['freeze'], device='cuda', dtype=t.long)
                        plan[name] = {'first': PB, 'P': length - 1, 'sw': None, 'pt': [pairs(H25.GROUP[g]) for g in ('names', 'holders')],
                                      'resp': None if a['resp'] is None else set(a['resp']), 'rep': a['rep'], 'm27': m27, 'mass': a.get('mass')}
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
                wrote, base_a, base_m, base_w, base_o = {}, None, None, None, None
                for name in state:                      # 'noop' comes first: it supplies the frozen outputs of this block
                    p = plan[name]
                    P, resp, pt = p['P'], p['resp'], p['pt']
                    T = length - P
                    x = state[name]
                    edit = pt is not None and b == PB
                    if edit and p['rep'] in wrote:      # the write of block 25 is the same for every arm of one donor draw
                        x = wrote[p['rep']]
                    elif resp is None or edit or f'a{b}' in resp:
                        tm = mask[:, :, P:, :]
                        hn = layer.input_layernorm(x)
                        q = a.q_norm(a.q_proj(hn).view(N, T, NH, hd)).transpose(1, 2)
                        k = a.k_norm(t.cat([kb[:, :P], a.k_proj(hn)], 1).view(N, length, kv, hd)).transpose(1, 2)
                        v = t.cat([vb[:, :P], a.v_proj(hn)], 1).view(N, length, kv, hd).transpose(1, 2)
                        k, v = rt.mod.repeat_kv(k, grp), rt.mod.repeat_kv(v, grp)
                        q = rot(q, cos[:, None, P:], sin[:, None, P:])
                        if edit or (b == B27 and T == 1 and p['m27'].get('donor')):
                            kd = rt.mod.repeat_kv(a.k_norm(kb.view(N, length, kv, hd)).transpose(1, 2), grp)
                            vd = rt.mod.repeat_kv(vb.view(N, length, kv, hd).transpose(1, 2), grp)
                            if edit:                    # heads C, answer row: donor keys / values of its unedited run (Phase 415)
                                for bi, bp, di, dp in pt:
                                    k[bi[:, None], hs[None], bp[:, None]] = kd[di[:, None], hs[None], dp[:, None]]
                                    v[bi[:, None], hs[None], bp[:, None]] = vd[di[:, None], hs[None], dp[:, None]]
                            else:                       # block 27, every head
                                for bi, bp, di, dp in p['m27']['pairs']:
                                    k[bi, :, bp] = kd[di, :, dp]
                                    v[bi, :, bp] = vd[di, :, dp]
                            del kd, vd
                        k = rot(k, cos[:, None], sin[:, None])
                        if b == B27 and T == 1:         # explicit attention of the answer row, the same code for every arm
                            m27 = p['m27']
                            lg = (q.float() @ k.float().transpose(2, 3)) / hd ** .5
                            lg = lg.masked_fill(~tm, float('-inf'))
                            for g in m27.get('hide', []):
                                lg = lg.masked_fill(grp_mask[g][:, None, None, :], float('-inf'))
                            w = lg.softmax(-1)
                            if name == 'noop':
                                base_w = w
                            if name == 'noop' or p.get('mass'):
                                key = 'base' if name == 'noop' else 'edited'
                                for g in MASS:
                                    mass[(key, g)] = mass.get((key, g), 0) + (w[:, :, 0] * grp_mask[g][:, None]).sum(-1).sum(0).cpu().numpy()
                                mass[(key, 'n')] = mass.get((key, 'n'), 0) + N
                            if m27.get('pattern'):
                                w = base_w
                            o = (w @ v.float()).to(v.dtype)
                            if name == 'noop':
                                base_o = o
                            if m27.get('freeze') is not None:
                                o[:, m27['freeze']] = base_o[:, m27['freeze']]
                            del lg, w
                        else:
                            if b == B27 and p.get('hide27'):
                                tm = tm.expand(N, 1, T, length).clone()
                                for g in p['hide27']:
                                    tm[:, :, -1, :] &= ~grp_mask[g][:, None]
                            o = F.scaled_dot_product_attention(q, k, v, attn_mask=tm)
                        ao = a.o_proj(o.transpose(1, 2).reshape(N, T, -1))
                        if name == 'noop':
                            base_a = ao
                        x = x + ao
                        if edit:
                            wrote[p['rep']] = x
                        del q, k, v, o
                    else:
                        x = x + base_a
                    if resp is None or f'm{b}' in resp:
                        mo = layer.mlp(layer.post_attention_layernorm(x))
                        if name == 'noop':
                            base_m = mo
                    else:
                        mo = base_m
                    state[name] = x + mo
                h = y
                for x in handles:
                    x.remove()
                cap.clear()
                del layer, y, a, kb, vb, wrote, base_a, base_m, base_w, base_o
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
        ctl = ('noop', 'self', 'self:don')
        meta.update({f'{k}_max_score_difference': diff(k) for k in ctl})
        meta['controls_ok'] = bool(all(same(k) and diff(k) <= TOL for k in ctl))
        meta['attention_mass_block27'] = {key: {g: [round(float(x), 4) for x in mass[(key, g)] / mass[(key, 'n')]] for g in MASS}
                                          for key in ('base', 'edited')}
    return results, meta


slim = DN.slim


def smoke():
    from deploy.storage_policy import checked_dir
    checked_dir(OUT, 1 << 30, operation='来源绑定注意力 27 产物')
    rows = [r for r in D.read(NL.OUT / 'pilot_name-materials.json') if int(r['family'][-4:]) < 2]
    arms = make_arms(rows, {'h0': [0], 'a': [1, 2, 3]})
    results, meta = engine(rows, arms, stage='smoke')
    T, nf, stems = DN.table(rows, arms, results)
    am = meta.pop('attention_mass_block27')
    print(json.dumps({'meta': meta, 'arms': len(arms), 'conflict_edits': int(len(T['follow'])),
                      'mean_x': {s: round(float(T[s].mean()), 2) for s in stems}, 'mean_nat': round(float(T['nat'].mean()), 2),
                      'mass_names_base_sum': round(sum(am['base']['names']), 3), 'mass_names_edited_sum': round(sum(am['edited']['names']), 3)}), flush=True)


def pilot(head_sets=None, tag='pilot'):
    """Opened panel holdout_dn with the donor draws of Phase 420."""
    from deploy.storage_policy import checked_dir
    checked_dir(OUT, 1 << 30, operation='来源绑定注意力 27 产物')
    rows = D.read(DN.OUT / f'{DN.SPLIT}-materials.json')
    arms = make_arms(rows, head_sets or {str(h): [h] for h in range(NH)}, DN.RUN, DN.SEED)
    results, meta = engine(rows, arms, stage='pilot')
    old = D.read(DN.OUT / 'items.json')
    mx = lambda a, b: max(abs(p - q) for k in range(REPS) for x, y in zip(results[f'{a}#{k}'], old[f'{b}#{k}']) for p, q in zip(x['scores'], y['scores']))
    meta['replay_of_phase_420'] = {
        'swap_same_choice': float(np.mean([x['pred_record'] == y['pred_record'] for k in range(REPS)
                                           for x, y in zip(results[f'swap#{k}'], old[f'swap#{k}'])])),
        'full_max_score_difference': mx('full', 'full'), 'ko_max_score_difference': mx('ko', 'ko:a27')}
    save(f'{tag}-design.json', arms)
    save(f'{tag}-items.json', slim(results))
    save(f'{tag}-meta.json', meta)
    print(json.dumps({k: v for k, v in meta.items() if k != 'attention_mass_block27'}), flush=True)



# ---- frozen after pilot 1 on the opened panel holdout_dn ----
K = [0, 3, 12, 14]          # heads of block 27 whose single freeze raises the share most (pilot 1: +0.37, +0.23, +0.24, +0.24)
RANDOM_SETS = 5
BROAD = ['swap', 'swap|hide:N', 'swap|hide:NH']


def final_sets():
    rng = np.random.default_rng(SEED)
    rest = [h for h in range(NH) if h not in K]
    return {'K': K, **{str(h): [h] for h in K}, **{f'R{j}': sorted(int(h) for h in rng.choice(rest, len(K), replace=False)) for j in range(RANDOM_SETS)}}


def summarise(results, arms, rows, mass):
    T, nf, stems = DN.table(rows, arms, results)
    draws = np.random.default_rng(SEED).integers(0, nf, (1000, nf))
    y, fam, nat, full = T['follow'], T['fam'], T['nat'], T['full']
    members = [np.flatnonzero(fam == f) for f in range(nf)]
    fsum = lambda v: np.bincount(fam, weights=np.asarray(v, float), minlength=nf)
    den, cnt = fsum(full - nat), np.bincount(fam, minlength=nf)
    r4 = lambda v: round(float(v), 4)
    ci = lambda b: [r4(np.quantile(b, .025)), r4(np.quantile(b, .975))]

    def frac(v):        # in units of the full shift of score(P) - score(B)
        num = fsum(v)
        return {'rate': r4(num.sum() / den.sum()), 'ci95': ci(num[draws].sum(1) / den[draws].sum(1))}

    def rate(v):
        num = fsum(v)
        return {'rate': r4(np.mean(v)), 'ci95': ci(num[draws].sum(1) / cnt[draws].sum(1))}

    def auc_of(fn):
        vals = [fn(np.concatenate([members[f] for f in d])) for d in draws[:300]]
        return {'auc': r4(fn(np.arange(len(y)))), 'ci95': [r4(np.nanquantile(vals, .025)), r4(np.nanquantile(vals, .975))]}
    sh = {s: T[s] - nat for s in stems}
    out = {'conflict_edits': int(len(y)), 'follow': rate(y), 'full_shift': rate(sh['full']),
           'share': {s: frac(sh[s]) for s in stems}, 'gain_over_full': {s: frac(sh[s] - sh['full']) for s in stems if s != 'full'}}
    g = out['gain_over_full']
    rnd = [f'fz:R{j}' for j in range(RANDOM_SETS)]
    out['random_head_sets'] = {'max_gain': max(g[k]['rate'] for k in rnd), 'mean_gain': r4(np.mean([g[k]['rate'] for k in rnd])),
                               'K_minus_mean_random': frac(sh['fz:K'] - np.mean([sh[k] for k in rnd], 0))}
    out['pattern_minus_ko'] = frac(sh['pattern'] - sh['ko'])
    out['auc'] = {s: auc_of(lambda i, s=s: PE.auc(T[s][i], y[i])) for s in ('nat', 'full', 'ko', 'pattern', 'don:N', 'don:NH', 'hide:N', 'fz:K')}
    for a in ('don:N', 'pattern', 'fz:K'):
        out['auc'][f'{a} - full'] = auc_of(lambda i, a=a: PE.auc(T[a][i], y[i]) - PE.auc(T['full'][i], y[i]))
    fol = {s: [] for s in BROAD}
    for k in range(REPS):
        for i, (r, j) in enumerate(zip(rows, arms[f'swap#{k}']['donor'])):
            B, P, I = F3.records(r, rows[j], 'names', 24)
            if F3.klass(B, P, I):
                for s in BROAD:
                    fol[s].append(results[f'{s}#{k}'][i]['pred_record'] == P)
    fol = {s: np.array(v, float) for s, v in fol.items()}
    out['broad_edit_follow'] = {**{s: rate(v) for s, v in fol.items()},
                                **{f'{s} - swap': rate(fol[s] - fol['swap']) for s in BROAD[1:]}}
    out['attention_mass_block27'] = {
        key: {**{g2: {'all_heads_sum': r4(sum(mass[key][g2])), 'K_sum': r4(sum(mass[key][g2][h] for h in K))} for g2 in MASS},
              'names_by_K_head': {str(h): mass[key]['names'][h] for h in K}} for key in ('base', 'edited')}
    return out


THRESHOLDS = {   # fixed after pilot 2 on the opened panel holdout_dn, before pre-registration. Gains are in units of the full shift.
    'Q1_opposition_is_the_attention_pattern': 'with the base-run attention weights in block 27 the gain over full is >= 0.45 with a '
                                              'lower bound > 0 and >= 0.7 x the gain of freezing the whole output. not_supported if '
                                              'the lower bound <= 0.',
    'Q2_it_needs_the_unedited_names': 'with donor keys / values of the name tokens in block 27 the gain is >= 0.45 with a lower bound '
                                      '> 0; hiding records and headers changes the share by <= 0.10 in size. not_supported if the '
                                      'lower bound <= 0.',
    'Q3_four_heads_carry_it': 'freezing heads 27.0, 27.3, 27.12, 27.14: gain >= 0.45 and >= 0.7 x the gain of freezing the whole '
                              'output; no random set of 4 other heads gains more than 0.30; gain(K) minus the mean random gain has a '
                              'lower bound > 0. not_supported if that lower bound <= 0.',
    'Q4_in_the_broad_edit_attention_27_helps': 'follow(swap with the names hidden from the answer row of block 27) - follow(swap) '
                                               '<= -0.05 with an upper bound < 0. not_supported if the upper bound >= 0.',
    'Q5_repairing_the_re-read_sharpens_the_prediction': 'AUC(x with donor names in block 27) - AUC(x of the narrow edit) has a lower '
                                                        'bound > 0. not_supported otherwise.',
}
CRITERIA = {
    'model': 'Qwen3-14B original BF16, CPU byte reads, CUDA layer streaming. GLM-4-9B is not run (failed the behaviour gate of this task).',
    'material': 'layout-factorial families as in Phase 405. holdout_a27: 64 families (1,024 prompts). Entity names occur in no earlier '
                'panel of this line; the 16 discovery code words, 4 seen frames.',
    'edits': 'narrow edit (Phases 415 / 417, under every arm but the swap arms): block 25, heads 25.18, 25.24, 25.15, attention row of '
             'the answer position, keys and values of name tokens and holders are those of the donor. On top, at block 27 and the '
             'attention row of the answer position only: ko (attention output of the base run); pattern (attention weights of the '
             'base run, values as computed); hide:G (group masked: N names, H holders, R records and headers); don:G (keys and '
             'values of the group from the donor in its unedited run, every head); fz:<set> (head outputs before the output '
             'projection frozen to the base run); keep:<set> (all other heads frozen). Broad edit: both name tokens exchanged as '
             'residual states at the input of block 24; swap|hide:G masks a group for the answer row of block 27. 4 seeded donor '
             'relations per item, the same in every arm.',
    'head_sets': 'K = 27.0, 27.3, 27.12, 27.14 (pilot 1: the four largest single-head gains); 5 seeded random sets of 4 other heads.',
    'measures': 'x = score(P) - score(B) at the answer position. Share = mean(x_arm - x_unedited) / mean(x_full - x_unedited) over '
                'conflict edits (P != B); gain = share - 1. AUC of x for follow (the broad edit answers P). Attention mass of block 27 '
                'by token group, base run and narrow edit (descriptive). Family-bootstrap 95 % intervals (1000 draws; 300 for AUC).',
    'controls': 'tail recomputation equals the base forward (same choice, score difference <= 1.0); the narrow edit with the prompt '
                'itself, alone and with its own names as donor in block 27, changes nothing (same bound).',
    'two_pass': 'gate stage: unmodified forward and the v1 behaviour gate. Edits run only if the gate passed.',
    'predictions_on_holdout': THRESHOLDS,
    'not_covered': 'which tokens the four heads move their attention to and what they then write (the mass table is descriptive; '
                   'hiding the names alone gives a smaller gain than donor names); the other negative components (attention 31, 33); '
                   'the single-name sites; other tasks and models. In block 27 the answer row is computed in float32 for every arm.',
    'disclosure': 'two pilots on the opened panel holdout_dn (donor draws of Phase 420). Pilot 1: the arms above and every head of '
                  'block 27 frozen alone; K was chosen there. Pilot 2: the final arms; thresholds were set on it. Smoke test on 2 '
                  'exposed families of pilot_name.',
}


def verdicts(s, meta):
    tri = lambda ok, bad: 'supported' if ok else 'not_supported' if bad else 'inconclusive'
    v = {'controls_valid': bool(meta['controls_ok'])}
    if not v['controls_valid']:
        return {**v, **{k: 'controls_invalid' for k in THRESHOLDS}}
    g, A, R = s['gain_over_full'], s['auc'], s['random_head_sets']
    ko = g['ko']['rate']
    x = g['pattern']
    v['Q1_opposition_is_the_attention_pattern'] = tri(x['rate'] >= .45 and x['ci95'][0] > 0 and x['rate'] >= .7 * ko, x['ci95'][0] <= 0)
    x = g['don:N']
    v['Q2_it_needs_the_unedited_names'] = tri(x['rate'] >= .45 and x['ci95'][0] > 0 and abs(g['hide:R']['rate']) <= .10, x['ci95'][0] <= 0)
    x, lo = g['fz:K'], R['K_minus_mean_random']['ci95'][0]
    v['Q3_four_heads_carry_it'] = tri(x['rate'] >= .45 and x['rate'] >= .7 * ko and R['max_gain'] <= .30 and lo > 0, lo <= 0)
    x = s['broad_edit_follow']['swap|hide:N - swap']
    v['Q4_in_the_broad_edit_attention_27_helps'] = tri(x['rate'] <= -.05 and x['ci95'][1] < 0, x['ci95'][1] >= 0)
    v['Q5_repairing_the_re-read_sharpens_the_prediction'] = tri(A['don:N - full']['ci95'][0] > 0, True)
    return v


def pilot2():
    """The frozen arms on the opened panel holdout_dn (donor draws of Phase 420); thresholds are set on this output."""
    pilot(final_sets(), 'pilot2')
    rows = D.read(DN.OUT / f'{DN.SPLIT}-materials.json')
    meta = D.read(OUT / 'pilot2-meta.json')
    save('pilot2-summary.json', summarise(D.read(OUT / 'pilot2-items.json'), D.read(OUT / 'pilot2-design.json'), rows, meta['attention_mass_block27']))



DEPS = (PT, NL, FE, AR, F3, H25, PE, DN)


def prereg():
    from deploy.storage_policy import checked_dir
    from tools import provenance
    from tools.power_analysis import check_phase
    checked_dir(OUT, 1 << 30, operation='来源绑定注意力 27 产物')
    assert not (OUT / 'prereg.json').exists()
    plan = D.checked(MODEL)
    fz = {'K': K, 'sets': final_sets(), 'pilot2_summary_sha256': D.sha(OUT / 'pilot2-summary.json')}
    rows = build_materials()
    assert len(rows) == FAMILIES * 16 and len({tuple(r['tokens']) for r in rows}) == len(rows)
    arms = make_arms(rows, final_sets())
    save(f'{SPLIT}-materials.json', rows)
    save('design.json', arms)
    body = {'schema': 'source_binding_attn27_prereg.v1', 'run_id': RUN, 'seed': SEED, 'created_at': datetime.now().astimezone().isoformat(),
            'base_plan_sha256': D.sha(SRC / 'plan.json'), 'model_snapshot': plan['models'][MODEL], 'code_sha256': D.sha(__file__),
            'dependency_sha256': {m.__name__: D.sha(m.__file__) for m in DEPS},
            'criteria': CRITERIA, 'frozen': fz,
            'materials': {'sha256': D.sha(OUT / f'{SPLIT}-materials.json'), 'families': FAMILIES, 'rows': len(rows)},
            'design': {'sha256': D.sha(OUT / 'design.json'), 'arms': len(arms)},
            'power': {'unit': 'paired binary outcome per item (follows P under swap / under swap with the names hidden in block 27); alpha = 0.05',
                      'Q4_as_if_one_edit_per_item': {'expected_delta': 0.08, 'p_d': 0.20, **check_phase(0.08, len(rows), p_d=0.20)},
                      'caveat': 'about 3,400 conflict edits clustered in 1,024 items and 64 families; shares, gains and AUCs are judged '
                                'by family-bootstrap intervals (pilot intervals of the gains +-0.06 or narrower)'},
            'custody': 'holdout_a27 is read first by the gate stage (behaviour only) and then once by the run stage.',
            'resources': {'gpu_minutes_estimate': 8, 'disk_GiB_estimate': 0.1, 'models_concurrent': 1,
                          'shape': 'chunks of 192 prompts in one batch (the Phase 415 shape); pilot peak 4.6 GiB'},
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
    save('gate.json', {'schema': 'source_binding_attn27_gate.v1', 'run_id': RUN, 'items_sha256': D.sha(OUT / 'gate-items.json'), **g, **meta})
    print(json.dumps({k: g[k] for k in ('passed', 'forced_accuracy', 'top1_accuracy')}), flush=True)


def run():
    from tools import provenance
    pre = checked()
    assert not (OUT / 'report.json').exists(), 'holdout already used'
    g = D.read(OUT / 'gate.json')
    rows, arms = D.read(OUT / f'{SPLIT}-materials.json'), D.read(OUT / 'design.json')
    if g['passed']:
        results, meta = engine(rows, arms, stage='run')
        save('items.json', slim(results))
        base = D.read(OUT / 'gate-items.json')
        meta['base_replay_max_score_difference'] = max(abs(a - b) for x, y in zip(results['base'], base) for a, b in zip(x['scores'], y['scores']))
        s = summarise(results, arms, rows, meta['attention_mass_block27'])
        v = verdicts(s, meta)
        print(json.dumps({'verdicts': v}), flush=True)
    else:
        s, meta, v = {'note': 'behaviour gate failed; no edit was run'}, {}, {k: 'behavior_ineligible' for k in THRESHOLDS}
    report = {'schema': 'source_binding_attn27_report.v1', 'run_id': RUN, 'split': SPLIT, 'gate_passed': g['passed'],
              'prereg_sha256': D.sha(OUT / 'prereg.json'), 'gate_sha256': D.sha(OUT / 'gate.json'),
              'items_sha256': D.sha(OUT / 'items.json') if g['passed'] else None, 'criteria': pre['criteria'], 'frozen': pre['frozen'],
              'verdicts': v, 'summary': s, **meta, 'formal_evidence_promotion': 0,
              'custody': 'holdout_a27 read once by the gate stage and once by this run',
              'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    save('report.json', report)
    path = provenance.write_result('SOURCE-BINDING-attn27-holdout', {'verdicts': v, 'gate_passed': g['passed'],
                                                                   'report_sha256': D.sha(OUT / 'report.json')},
                                   model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)
    print(json.dumps({'verdicts': v, 'result': str(path)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['smoke', 'pilot', 'pilot2', 'prereg', 'gate', 'run'])
    {'smoke': smoke, 'pilot': pilot, 'pilot2': pilot2, 'prereg': prereg, 'gate': gate, 'run': run}[ap.parse_args().stage]()
