"""Downstream of the block-25 write (Claude Phase 420).  Qwen3-14B BF16.

Phases 417 / 419: after a narrow edit (heads 25.18, 25.24, 25.15 read donor name tokens and holders at the answer
position) the score of the pointer record predicts the broad block-24 exchange (AUC 0.89-0.91); a linear read-out of
what the heads write reaches 0.78.  The rest is made by what follows the write.  The write lands on the answer
position only, so everything downstream is the computation of that one position: MLP 25, then attention and MLP of
blocks 26-39 (keys and values of all other positions stay those of the base run).

Path patching by component: a component is RESPONSIVE (it sees the edited stream and writes what it computes) or
FROZEN (it writes what it wrote in the unedited base run).  Arms: full (all responsive, the Phase 417 predictor),
direct (all frozen: the write reaches the unembedding only through the final norm), cum<b> (responsive up to block b),
ko:<c> (all but c), only:<c>, and the same for component sets.
Stages: smoke, pilot (opened panel holdout_per) -> prereg -> gate -> run.
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
import source_binding_peredit as PE
import source_binding_readout as RO
from source_binding_runtime import Runtime

SRC, MODEL = D.OUT, 'qwen14'
OUT = D.ROOT / 'tests/glm5/result/source_binding_downstream_v1'
RUN, SEED = 'SOURCE-BINDING-DOWNSTREAM-20261112-v1', 202611120420
SPLIT, FAMILIES = 'holdout_dn', 64
CHUNK, NB, NH, TOL, REPS, L, PB = 192, 40, 40, 1.0, 4, 24, 25
C = H25.C
COMPS = ['m25'] + [f'{k}{b}' for b in range(PB + 1, NB) for k in ('a', 'm')]
OTHER_PANELS = RO.OTHER_PANELS + ['source_binding_readout_v1/holdout_ro']
CLASSES = PE.CLASSES
RANDOM_SETS = 5


def save(name, body):
    from deploy.storage_policy import check_write
    p = OUT / name
    check_write(p, 1 << 20, operation='来源绑定下游追踪产物')
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


def upto(b):
    return [c for c in COMPS if int(c[1:]) <= b]


def arm_sets(sets):
    """name -> responsive components (None = all)."""
    out = {'full': None, 'direct': [], 'attn': [c for c in COMPS if c[0] == 'a'], 'mlp': [c for c in COMPS if c[0] == 'm']}
    for b in range(PB, NB - 1):
        out[f'cum{b}'] = upto(b)
    for name, s in sets.items():
        out[f'only:{name}'] = list(s)
        out[f'ko:{name}'] = [c for c in COMPS if c not in s]
    return out


def make_arms(rows, sets, run=RUN, seed=SEED):
    H25.RUN, H25.SEED = run, seed                 # donor draws: the stream of the Phase 415 module contains run id and seed
    dons, rels = H25.draws(rows)
    me = list(range(len(rows)))
    arms = {'self': {'donor': me, 'resp': None, 'rep': 'self'}, 'self:direct': {'donor': me, 'resp': [], 'rep': 'self'}}
    for k, don in enumerate(dons):
        arms[f'swap#{k}'] = {'swap': True, 'donor': don, 'rel': rels[k]}
        for name, resp in arm_sets(sets).items():
            arms[f'{name}#{k}'] = {'donor': don, 'rel': rels[k], 'resp': resp, 'rep': k}
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
                P0 = min(off[i] + AR.names_of(r)[0] for i, r in enumerate(rs))
                plan['noop'] = {'first': 0, 'P': length - 1, 'sw': None, 'resp': None, 'pt': None}
                for name, a in arms.items():
                    local = [j - c0 for j in a['donor'][c0:c0 + N]]
                    assert all(0 <= j < N and rs[j]['family'] == r['family'] for j, r in zip(local, rs))
                    pairs = lambda fn: tuple(t.tensor(v, device='cuda') for v in (
                        [i for i, r in enumerate(rs) for _ in fn(r)], [off[i] + x for i, r in enumerate(rs) for x in fn(r)],
                        [j for j in local for _ in fn(rs[j])], [off[j] + y for j in local for y in fn(rs[j])]))
                    if a.get('swap'):
                        plan[name] = {'first': L, 'P': P0, 'sw': L, 'idx': pairs(AR.names_of), 'resp': None, 'pt': None}
                    else:
                        plan[name] = {'first': PB, 'P': length - 1, 'sw': None, 'pt': [pairs(H25.GROUP[g]) for g in ('names', 'holders')],
                                      'resp': None if a['resp'] is None else set(a['resp']), 'rep': a['rep']}
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
                wrote, base_a, base_m = {}, None, None
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
                        if edit:                        # heads C, answer row: donor keys / values of its unedited run (Phase 415)
                            kd = rt.mod.repeat_kv(a.k_norm(kb.view(N, length, kv, hd)).transpose(1, 2), grp)
                            vd = rt.mod.repeat_kv(vb.view(N, length, kv, hd).transpose(1, 2), grp)
                            for bi, bp, di, dp in pt:
                                k[bi[:, None], hs[None], bp[:, None]] = kd[di[:, None], hs[None], dp[:, None]]
                                v[bi[:, None], hs[None], bp[:, None]] = vd[di[:, None], hs[None], dp[:, None]]
                            del kd, vd
                        o = F.scaled_dot_product_attention(q, rot(k, cos[:, None], sin[:, None]), v, attn_mask=tm)
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
                del layer, y, a, kb, vb, wrote, base_a, base_m
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
        ctl = ('noop', 'self', 'self:direct')
        meta.update({f'{k}_max_score_difference': diff(k) for k in ctl})
        meta['controls_ok'] = bool(all(same(k) and diff(k) <= TOL for k in ctl))
    return results, meta


def table(rows, arms, items):
    """One row per conflict edit of the broad arm: family, class, follow, x of every arm, the unedited score."""
    fam_id = {f: i for i, f in enumerate(sorted({r['family'] for r in rows}))}
    stems = sorted({n.split('#')[0] for n in arms if '#' in n and not n.startswith('swap')})
    T = defaultdict(list)
    for k in range(REPS):
        a = arms[f'swap#{k}']
        for i, (r, j) in enumerate(zip(rows, a['donor'])):
            B, P, I = F3.records(r, rows[j], 'names', 24)
            c = F3.klass(B, P, I)
            if not c:
                continue
            T['fam'].append(fam_id[r['family']]); T['cls'].append(c); T['follow'].append(items[f'swap#{k}'][i]['pred_record'] == P)
            b = items['base'][i]['scores']
            T['nat'].append(b[P] - b[B])
            for s in stems:
                x = items[f'{s}#{k}'][i]['scores']
                T[s].append(x[P] - x[B])
    return {k: np.array(v) for k, v in T.items()}, len(fam_id), stems


def slim(results):
    return {k: [{'id': x['id'], 'pred_record': x['pred_record'], 'scores': x['scores']} for x in v] for k, v in results.items()}


def smoke():
    from deploy.storage_policy import checked_dir
    checked_dir(OUT, 1 << 30, operation='来源绑定下游追踪产物')
    rows = [r for r in D.read(NL.OUT / 'pilot_name-materials.json') if int(r['family'][-4:]) < 2]
    arms = make_arms(rows, {c: [c] for c in COMPS})
    results, meta = engine(rows, arms, stage='smoke')
    T, nf, stems = table(rows, arms, results)
    print(json.dumps({'meta': meta, 'arms': len(arms), 'conflict_edits': int(len(T['follow'])),
                      'mean_x': {s: round(float(T[s].mean()), 2) for s in ('full', 'direct', 'attn', 'mlp', 'cum30')},
                      'mean_nat': round(float(T['nat'].mean()), 2)}), flush=True)


blocks = lambda b0, b1, kinds='am': [c for c in COMPS if b0 <= int(c[1:]) <= b1 and c[0] in kinds]
KO_RANK = ['a28', 'm26', 'm31', 'm27', 'a35', 'a26', 'm25', 'a29', 'a37', 'a36', 'm29', 'm30', 'm32', 'm37']   # pilot 1, lost share
W1 = blocks(25, 28)
PILOT2 = {**{f'top{k}': KO_RANK[:k] for k in (4, 6, 8, 10, 12, 14)},
          'W1': W1, 'W1+a35-37': W1 + blocks(35, 37, 'a'), 'W1+a34-37': W1 + blocks(34, 37, 'a'), 'W1+a29-37': W1 + blocks(29, 37, 'a'),
          'W1+a29-39': W1 + blocks(29, 39, 'a'), 'W1+a29-37+m31': W1 + blocks(29, 37, 'a') + ['m31'],
          'W1+b29-33+a34-37': blocks(25, 33) + blocks(34, 37, 'a'), 'b25-33': blocks(25, 33), 'a34-37': blocks(34, 37, 'a'),
          'a29-37': blocks(29, 37, 'a'), 'b34-37': blocks(34, 37), 'W1+b34-37': W1 + blocks(34, 37),
          'W1-a27+a34-37': [c for c in W1 if c != 'a27'] + blocks(34, 37, 'a'), 'a27': ['a27'],
          'm25-28+a': blocks(25, 28, 'm') + blocks(26, 39, 'a'), 'm25-31+a': blocks(25, 31, 'm') + blocks(26, 39, 'a'),
          'm25-28': blocks(25, 28, 'm'), 'a26-28': blocks(26, 28, 'a')}


def pilot(sets=None, tag='pilot'):
    """Opened panel holdout_per with the donor draws of Phase 417: every component alone and knocked out; then sets."""
    from deploy.storage_policy import checked_dir
    checked_dir(OUT, 1 << 30, operation='来源绑定下游追踪产物')
    rows = D.read(PE.OUT / f'{PE.SPLIT}-materials.json')
    arms = make_arms(rows, sets or {c: [c] for c in COMPS}, PE.RUN, PE.SEED)
    results, meta = engine(rows, arms, stage='pilot')
    old = D.read(PE.OUT / 'items.json')
    meta['replay_of_phase_417'] = {
        'swap_same_choice': float(np.mean([x['pred_record'] == y['pred_record'] for k in range(REPS)
                                           for x, y in zip(results[f'swap#{k}'], old[f'swap#{k}'])])),
        'full_max_score_difference': max(abs(p - q) for k in range(REPS) for x, y in zip(results[f'full#{k}'], old[f'x|NH:C#{k}'])
                                         for p, q in zip(x['scores'], y['scores']))}
    save(f'{tag}-design.json', arms)
    save(f'{tag}-items.json', slim(results))
    save(f'{tag}-meta.json', meta)
    print(json.dumps(meta), flush=True)



# ---- frozen after the two pilots on the opened panel holdout_per ----
E, LT, MID = blocks(25, 28), blocks(34, 37, 'a'), blocks(29, 33)


def final_sets():
    rng = np.random.default_rng(SEED)
    ms, at = [c for c in COMPS if c[0] == 'm'], [c for c in COMPS if c[0] == 'a']
    sets = {'E': E, 'LT': LT, 'S': E + LT, 'SM': E + MID + LT, **{c: [c] for c in COMPS}}
    for j in range(RANDOM_SETS):    # same size and type mix as S (4 MLPs, 7 attention layers), drawn from all 29 components
        sets[f'R{j}'] = sorted(rng.choice(ms, 4, replace=False).tolist() + rng.choice(at, 7, replace=False).tolist(), key=COMPS.index)
    return sets


def summarise(results, arms, rows, T0):
    T, nf, stems = table(rows, arms, results)
    draws = np.random.default_rng(SEED).integers(0, nf, (1000, nf))
    y, fam, nat, full = T['follow'], T['fam'], T['nat'], T['full']
    members = [np.flatnonzero(fam == f) for f in range(nf)]
    fsum = lambda v: np.bincount(fam, weights=np.asarray(v, float), minlength=nf)
    den, cnt = fsum(full - nat), np.bincount(fam, minlength=nf)
    r4 = lambda v: round(float(v), 4)
    ci = lambda b: [r4(np.quantile(b, .025)), r4(np.quantile(b, .975))]

    def frac(v):        # share of the full shift of score(P) - score(B); v = shift per edit
        num = fsum(v)
        return {'rate': r4(num.sum() / den.sum()), 'ci95': ci(num[draws].sum(1) / den[draws].sum(1))}

    def rate(v):
        num = fsum(v)
        return {'rate': r4(np.mean(v)), 'ci95': ci(num[draws].sum(1) / cnt[draws].sum(1))}

    def auc_of(fn):
        vals = [fn(np.concatenate([members[f] for f in d])) for d in draws[:300]]
        return {'auc': r4(fn(np.arange(len(y)))), 'ci95': [r4(np.nanquantile(vals, .025)), r4(np.nanquantile(vals, .975))]}
    sh = {s: T[s] - nat for s in stems}
    out = {'conflict_edits': int(len(y)), 'follow': rate(y), 'mean_unedited_gap': r4(nat.mean()), 'full_shift': rate(sh['full']),
           'share_of_full_shift': {s: frac(sh[s]) for s in stems if not s.startswith('ko:')},
           'share_lost_when_frozen': {s[3:]: frac(sh['full'] - sh[s]) for s in stems if s.startswith('ko:')}}
    only = out['share_of_full_shift']
    out['single_component_max_share'] = max(only[f'only:{c}']['rate'] for c in COMPS)
    out['synergy_S_minus_E_minus_LT'] = frac(sh['only:S'] - sh['only:E'] - sh['only:LT'])
    rnd = [f'only:R{j}' for j in range(RANDOM_SETS)]
    out['random_sets'] = {'mean_share': r4(np.mean([only[k]['rate'] for k in rnd])), 'max_share': max(only[k]['rate'] for k in rnd),
                          'S_minus_mean_random': frac(sh['only:S'] - np.mean([sh[k] for k in rnd], 0)),
                          'mean_lost': r4(np.mean([out['share_lost_when_frozen'][f'R{j}']['rate'] for j in range(RANDOM_SETS)]))}
    A = ['nat', 'direct', 'attn', 'mlp', 'only:E', 'only:LT', 'only:S', 'only:SM', 'ko:S', 'ko:a27', 'full'] + rnd
    out['auc'] = {s: auc_of(lambda i, s=s: PE.auc(T[s][i], y[i])) for s in A}
    for a, b in (('only:S', 'direct'), ('only:SM', 'direct'), ('full', 'only:SM'), ('direct', 'nat')):
        out['auc'][f'{a} - {b}'] = auc_of(lambda i, a=a, b=b: PE.auc(T[a][i], y[i]) - PE.auc(T[b][i], y[i]))
    out['auc_by_class'] = {c: {s: r4(PE.auc(T[s][T['cls'] == c], y[T['cls'] == c])) for s in ('nat', 'direct', 'only:S', 'only:SM', 'full')}
                           for c in CLASSES}
    rule = lambda s: ((T[s] > T0) == y).astype(float)
    cls = ((T['cls'] == 'P=I') == y).astype(float)
    out['accuracy'] = {'full': rate(rule('full')), 'only:SM': rate(rule('only:SM')), 'only:S': rate(rule('only:S')), 'direct': rate(rule('direct')),
                       'class_rule': rate(cls), 'none': rate(~y), 'only:SM - class_rule': rate(rule('only:SM') - cls),
                       'full - only:SM': rate(rule('full') - rule('only:SM'))}
    out['correlation_with_full_shift'] = {s: r4(np.corrcoef(sh[s], sh['full'])[0, 1]) for s in ('direct', 'only:E', 'only:LT', 'only:S', 'only:SM')}
    return out


THRESHOLDS = {   # fixed after pilot 3 on the opened panel holdout_per, before pre-registration
    'N1_no_direct_path': 'direct: share of the full shift <= 0.05 and AUC(direct) - AUC(unedited score) has an upper bound < 0.02. '
                         'not_supported if the share > 0.15.',
    'N2_no_single_component': 'the largest share of any of the 29 components alone <= 0.15. not_supported if > 0.30.',
    'N3_attention_and_mlp_both_needed': 'share with only the attention layers responsive <= 0.30, with only the MLPs <= 0.10. '
                                        'not_supported if their sum > 0.60.',
    'N4_two_windows_interact': 'share(only S) - share(only E) - share(only LT) >= 0.25 with a lower bound > 0; share lost when E is '
                               'frozen >= 0.35 and when LT is frozen >= 0.35, lower bounds > 0. not_supported if the synergy lower '
                               'bound <= 0.',
    'N5_prediction_is_made_downstream': 'AUC(only SM) >= 0.82; AUC(only SM) - AUC(direct) has a lower bound > 0; accuracy of the frozen '
                                        'rule on only SM minus the class rule has a lower bound > 0. not_supported if either lower '
                                        'bound <= 0.',
    'N6_attention_27_opposes': 'share lost when attention 27 is frozen <= -0.40 with an upper bound < 0 (the shift grows). '
                               'not_supported if the upper bound >= 0.',
    'N7_the_set_is_not_privileged': 'the best of the 5 random sets of the same size and type mix reaches >= 0.8 x share(only S). '
                                    'not_supported if it stays below 0.5 x share(only S).',
}
CRITERIA = {
    'model': 'Qwen3-14B original BF16, CPU byte reads, CUDA layer streaming. GLM-4-9B is not run (failed the behaviour gate of this task).',
    'material': 'layout-factorial families as in Phase 405. holdout_dn: 64 families (1,024 prompts). Entity names occur in no earlier '
                'panel of this line; the 16 discovery code words, 4 seen frames.',
    'edits': 'broad edit (target): at the input of block 24 the residual states of both question name tokens are those of a donor of '
             'the same family; 4 seeded donor relations per item. Narrow edit (Phase 415 / 417): in block 25, heads 25.18, 25.24, 25.15, '
             'attention row of the answer position, keys and values of name tokens and holders are those of the donor.',
    'components': 'after the narrow edit only the answer position differs from the base run. Components: MLP 25 and attention / MLP of '
                  'blocks 26-39 at that position (29). A responsive component reads the edited stream; a frozen one writes its output '
                  'of the unedited base run. E = blocks 25-28 (7), LT = attention 34-37 (4), MID = blocks 29-33 (10), S = E + LT, '
                  'SM = E + MID + LT (frozen: MLP 34-37 and blocks 38-39). Random sets: 4 MLPs and 7 attention layers from all 29.',
    'measures': 'x = score(P) - score(B) at the answer position (P record named by the donor pointers, B base answer). Share of the '
                'full shift = mean(x_arm - x_unedited) / mean(x_full - x_unedited) over conflict edits (P != B). AUC of x for follow '
                '(the broad edit answers P). Frozen rule of Phase 417: follow iff x > -7.75. Family-bootstrap 95 % intervals '
                '(1000 draws; 300 for AUC).',
    'controls': 'tail recomputation equals the base forward (same choice, score difference <= 1.0); the narrow edit with the prompt '
                'itself, all responsive and all frozen, changes nothing (same bound).',
    'two_pass': 'gate stage: unmodified forward and the v1 behaviour gate. Edits run only if the gate passed.',
    'predictions_on_holdout': THRESHOLDS,
    'not_covered': 'single heads and MLP units inside the components; why attention 27 opposes (a re-read of the unedited name tokens '
                   'is a guess, not tested); the single-name sites; other tasks and models. Freezing to the base run is an '
                   'intervention; shares of interacting components do not add up.',
    'disclosure': 'three pilots on the opened panel holdout_per (donor draws of Phase 417). Pilot 1: every component alone and frozen. '
                  'Pilot 2: 24 hand-made sets; the windows E, LT, MID were chosen there. The aim was a small sufficient set; pilot 3 '
                  'showed random sets of the same size reaching the same share, so the claim was restated as N7 (no privileged set) '
                  'and the windows are kept as a description of early and late stages. Thresholds were set after pilot 3. Smoke test '
                  'on 2 exposed families of pilot_name.',
}


def verdicts(s, meta):
    tri = lambda ok, bad: 'supported' if ok else 'not_supported' if bad else 'inconclusive'
    v = {'controls_valid': bool(meta['controls_ok'])}
    if not v['controls_valid']:
        return {**v, **{k: 'controls_invalid' for k in THRESHOLDS}}
    o, l, A, acc = s['share_of_full_shift'], s['share_lost_when_frozen'], s['auc'], s['accuracy']
    d = o['direct']['rate']
    v['N1_no_direct_path'] = tri(d <= .05 and A['direct - nat']['ci95'][1] < .02, d > .15)
    m = s['single_component_max_share']
    v['N2_no_single_component'] = tri(m <= .15, m > .30)
    at, ml = o['attn']['rate'], o['mlp']['rate']
    v['N3_attention_and_mlp_both_needed'] = tri(at <= .30 and ml <= .10, at + ml > .60)
    sy = s['synergy_S_minus_E_minus_LT']
    v['N4_two_windows_interact'] = tri(sy['rate'] >= .25 and sy['ci95'][0] > 0 and all(l[k]['rate'] >= .35 and l[k]['ci95'][0] > 0 for k in ('E', 'LT')),
                                       sy['ci95'][0] <= 0)
    lo1, lo2 = A['only:SM - direct']['ci95'][0], acc['only:SM - class_rule']['ci95'][0]
    v['N5_prediction_is_made_downstream'] = tri(A['only:SM']['auc'] >= .82 and lo1 > 0 and lo2 > 0, lo1 <= 0 or lo2 <= 0)
    v['N6_attention_27_opposes'] = tri(l['a27']['rate'] <= -.40 and l['a27']['ci95'][1] < 0, l['a27']['ci95'][1] >= 0)
    r, S = s['random_sets']['max_share'], o['only:S']['rate']
    v['N7_the_set_is_not_privileged'] = tri(r >= .8 * S, r < .5 * S)
    return v


def pilot3():
    """The frozen arms on the opened panel holdout_per (donor draws of Phase 417); thresholds are set on this output."""
    pilot(final_sets(), 'pilot3')
    rows = D.read(PE.OUT / f'{PE.SPLIT}-materials.json')
    s = summarise(D.read(OUT / 'pilot3-items.json'), D.read(OUT / 'pilot3-design.json'), rows, PE.frozen()['T'])
    save('pilot3-summary.json', s)



DEPS = (PT, NL, AR, F3, H25, PE, RO)


def prereg():
    from deploy.storage_policy import checked_dir
    from tools import provenance
    from tools.power_analysis import check_phase
    checked_dir(OUT, 1 << 30, operation='来源绑定下游追踪产物')
    assert not (OUT / 'prereg.json').exists()
    plan = D.checked(MODEL)
    fz = {'T': PE.frozen()['T'], 'sets': final_sets(), 'pilot3_summary_sha256': D.sha(OUT / 'pilot3-summary.json')}
    rows = build_materials()
    assert len(rows) == FAMILIES * 16 and len({tuple(r['tokens']) for r in rows}) == len(rows)
    arms = make_arms(rows, final_sets())
    save(f'{SPLIT}-materials.json', rows)
    save('design.json', arms)
    body = {'schema': 'source_binding_downstream_prereg.v1', 'run_id': RUN, 'seed': SEED, 'created_at': datetime.now().astimezone().isoformat(),
            'base_plan_sha256': D.sha(SRC / 'plan.json'), 'model_snapshot': plan['models'][MODEL], 'code_sha256': D.sha(__file__),
            'dependency_sha256': {m.__name__: D.sha(m.__file__) for m in DEPS},
            'criteria': CRITERIA, 'frozen': fz,
            'materials': {'sha256': D.sha(OUT / f'{SPLIT}-materials.json'), 'families': FAMILIES, 'rows': len(rows)},
            'design': {'sha256': D.sha(OUT / 'design.json'), 'arms': len(arms)},
            'power': {'unit': 'paired binary outcome per item (frozen rule on only SM right / class rule right); alpha = 0.05',
                      'N5_as_if_one_edit_per_item': {'expected_delta': 0.10, 'p_d': 0.40, **check_phase(0.10, len(rows), p_d=0.40)},
                      'caveat': 'about 3,400 conflict edits clustered in 1,024 items and 64 families; shares and AUCs are judged by '
                                'family-bootstrap intervals, pilot intervals of the shares were +-0.03 or narrower on 48 families'},
            'custody': 'holdout_dn is read first by the gate stage (behaviour only) and then once by the run stage.',
            'resources': {'gpu_minutes_estimate': 8, 'disk_GiB_estimate': 0.3, 'models_concurrent': 1,
                          'shape': 'chunks of 192 prompts in one batch (the Phase 415 shape); pilot peak 4.5 GiB'},
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
    save('gate.json', {'schema': 'source_binding_downstream_gate.v1', 'run_id': RUN, 'items_sha256': D.sha(OUT / 'gate-items.json'), **g, **meta})
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
        s = summarise(results, arms, rows, pre['frozen']['T'])
        v = verdicts(s, meta)
        print(json.dumps({'meta': meta, 'verdicts': v}), flush=True)
    else:
        s, meta, v = {'note': 'behaviour gate failed; no edit was run'}, {}, {k: 'behavior_ineligible' for k in THRESHOLDS}
    report = {'schema': 'source_binding_downstream_report.v1', 'run_id': RUN, 'split': SPLIT, 'gate_passed': g['passed'],
              'prereg_sha256': D.sha(OUT / 'prereg.json'), 'gate_sha256': D.sha(OUT / 'gate.json'),
              'items_sha256': D.sha(OUT / 'items.json') if g['passed'] else None, 'criteria': pre['criteria'], 'frozen': pre['frozen'],
              'verdicts': v, 'summary': s, **meta, 'formal_evidence_promotion': 0,
              'custody': 'holdout_dn read once by the gate stage and once by this run',
              'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    save('report.json', report)
    path = provenance.write_result('SOURCE-BINDING-downstream-holdout', {'verdicts': v, 'gate_passed': g['passed'],
                                                                       'report_sha256': D.sha(OUT / 'report.json')},
                                   model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)
    print(json.dumps({'verdicts': v, 'result': str(path)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['smoke', 'pilot', 'pilot2', 'pilot3', 'prereg', 'gate', 'run'])
    {'smoke': smoke, 'pilot': pilot, 'pilot2': lambda: pilot(PILOT2, 'pilot2'), 'pilot3': pilot3, 'prereg': prereg, 'gate': gate, 'run': run}[ap.parse_args().stage]()
