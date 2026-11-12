"""What the block-25 heads (25.18, 25.24, 25.15) carry into the answer position, by source (Claude Phase 415).  Qwen3-14B BF16.

Phases 410 / 414: at the answer position these heads read the question names (the pointers) and the positions after
the names (holders).  Here the content of each source is exchanged for these heads only: in block 25, at the attention
row of the answer position, the keys and values of the name tokens (N), of the holders (H) or of both (NH) are taken
from a donor prompt of the same family.  Nothing else is edited, so every position still holds the base content.
The choice is then classified: B the base answer, P the record at the ordinal position of the donor target (a pointer
was carried), I the record the donor asks for by name (an identity was carried).
Backgrounds: nat (nothing else edited); sole (in blocks 24-25 the answer position reads names and holders through
these heads only).  Also: attention mass of the heads by token group (natural run), and in the block-24 conflict of
Phase 413 the heads lose the names, the holders or both.
Stages: smoke, pilot (opened rows of holdout_bak) -> prereg -> gate -> run.
"""
import argparse, gc, hashlib, json, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_binding_design as D
import source_binding_relay as RL
import source_binding_pointer as PT
import source_binding_nameloc as NL
import source_binding_fidelity2 as F2
import source_binding_fetch as FE
import source_binding_arbitration as AR
import source_binding_backup as BK
from source_binding_runtime import Runtime

SRC, MODEL = D.OUT, 'qwen14'
OUT = D.ROOT / 'tests/glm5/result/source_binding_h2518_v1'
RUN, SEED = 'SOURCE-BINDING-H2518-20261112-v1', 202611120415
SPLIT, FAMILIES = 'holdout_h25', 48
CHUNK, NB, NH, TOL, REPS, L, PB = 192, 40, 40, 1.0, 4, 24, 25
C = [18, 24, 15]
SETS = {'C': C, '18': [18]}
OTHER_PANELS = BK.OTHER_PANELS + ['source_binding_backup_v1/holdout_bak']
NONC = [h for h in range(NH) if h not in C]
SOLE = [[g, 24, 25, None] for g in ('names', 'holders')] + [[g, 25, 26, NONC] for g in ('names', 'holders')]
THRESHOLDS = {   # fixed after the pilot on opened rows, before pre-registration
    'D1_names_carry_a_pointer': 'nat|N:C on three-way edits: mean shift of score(P) - score(B) >= 2.0 with a lower bound > 0; shift to '
                                'P minus shift to I has a lower bound > 0; the random head set (nat|NH:rand) shifts by less than 0.5 '
                                'in size. not_supported if the P shift < 1.0 or the P-minus-I lower bound <= 0.',
    'D2_holders_carry_little_alone': 'nat|H:C: choice changed on <= 0.02 of all edits and the P shift is below a quarter of the P '
                                     'shift of nat|N:C. not_supported if changed > 0.05 or the P shift exceeds half of it.',
    'D3_names_feed_the_new_pointer_in_conflict': 'follow(swap-N) - follow(swap) <= -0.20 with an upper bound < 0. not_supported if '
                                                 '> -0.10.',
    'D4_pull_back_is_partly_carried_content': 'follow(swap-Hv0) - follow(swap) >= 0.03 with a lower bound > 0. not_supported if the '
                                              'lower bound <= 0.',
    'D5_pull_back_is_partly_attention_share': 'follow(swap-H) - follow(swap-Hv0) has a lower bound > 0. not_supported otherwise.',
    'D6_donor_holders_add_no_push': '|follow(swap+Hdon) - follow(swap-H)| <= 0.03. not_supported if > 0.06.',
}
CRITERIA = {
    'model': 'Qwen3-14B original BF16, CPU byte reads, CUDA layer streaming. GLM-4-9B is not run (failed the behaviour gate of this task).',
    'material': 'layout-factorial families as in Phase 405. holdout_h25: 48 families (768 prompts). Entity names occur in no earlier '
                'panel of this line; the 16 discovery code words, 4 seen frames.',
    'heads': 'C = 25.18, 25.24, 25.15 (frozen in Phase 414 from opened rows); 25.18 alone; one random set of 3 other heads of block 25.',
    'exchange': 'in block 25, for the heads of the set and the attention row of the answer position only, the keys and values (before '
                'the rotary step, which keeps the base positions) of the name tokens (N), of the holders (H) or of both (NH) are '
                'replaced by those of the same tokens of a donor prompt in its unedited run. Zeroing (v0): the values of a source are '
                'set to 0 for these heads while keys and attention weights stay as they are. Donor: same family, differs in a '
                'non-empty subset of (archive asked, entity asked, archive order, entity order), equal token counts; 4 seeded '
                'relations per item, the same draws in every arm.',
    'backgrounds': {'nat': 'nothing else edited', 'sole': 'in block 24 the answer position cannot read names or holders; in block 25 '
                    'only the heads of C can'},
    'outcome': 'B = base answer, P = record of the base prompt at the ordinal position of the donor target (position rule of the '
               'graph), I = record the donor asks for by name. Rates of the forced choice on three-way edits (B, P, I all '
               'different) and on all edits where the class differs from B; mean shift of score(P) - score(B) and score(I) - '
               'score(B) against the unedited run; family-bootstrap 95 % intervals.',
    'controls': 'tail recomputation: the no-edit arm equals the base forward (same choice, score difference <= 1.0); exchanging with '
                'the prompt itself changes nothing (same bound); the sole background alone keeps accuracy >= 0.90.',
    'disclosure': 'the pilot (16 opened families of holdout_bak) was run twice. The first run showed that exchanging the holder '
                  'content for these heads moves nothing although hiding the holders does (Phase 414); the arms that zero the '
                  'values with the attention weights kept (v0) and that give the heads the donor holders inside the conflict were '
                  'added to separate carried content from attention share. Thresholds were set after the second run. The engine was '
                  'smoke-tested on 2 exposed families of pilot_name.',
    'two_pass': 'gate stage: unmodified forward and the v1 behaviour gate. Edits run only if the gate passed.',
    'not_covered': 'the QK side separately from the OV side (keys and values are exchanged together); the MLP of block 25 and later '
                   'blocks that turn what is written into the fetch; other tasks, answer words, frames and models. An exchange is '
                   'an intervention, not a natural run.',
}


def stream(purpose, key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(f'{RUN}|{SEED}|{purpose}|{key}'.encode()).digest()[:16], 'little'))


def save(name, body):
    from deploy.storage_policy import check_write
    p = OUT / name
    check_write(p, 1 << 20, operation='来源绑定 25.18 分解产物')
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


GROUP = {'holders': AR.holders_of, 'names': AR.names_of}
SRC_GROUPS = {'N': ['names'], 'H': ['holders'], 'NH': ['names', 'holders']}


def draws(rows):
    index = {r['id']: i for i, r in enumerate(rows)}
    out, rels_out = [[] for _ in range(REPS)], [[] for _ in range(REPS)]
    for r in rows:
        row = {}
        for rel in F2.RELATIONS:
            j = index[f"{r['family']}-S{r['source'] ^ rel[0]}E{r['entity'] ^ rel[1]}O{r['order'] ^ rel[2]}X{r['x'] ^ rel[3]}"]
            d = rows[j]
            if len(r['tokens']) == len(d['tokens']) and all(len(f(r)) == len(f(d)) for f in GROUP.values()):
                row[rel] = j
        assert len(row) >= REPS
        rels = sorted(row)
        for k, i in enumerate(stream('draw', r['id']).permutation(len(rels))[:REPS]):
            out[k].append(row[rels[i]])
            rels_out[k].append(list(rels[i]))
    return out, rels_out


def final_arms(rows):
    dons, rels = draws(rows)
    rnd = [int(h) for h in stream('random', 0).choice(NONC, len(C), replace=False)]
    arms = {'sole': {'hide': SOLE}, 'self:NH:C': {'hide': [], 'patch': [['names', C], ['holders', C]], 'donor': list(range(len(rows)))}}
    for k, don in enumerate(dons):
        for bg, hide in (('nat', []), ('sole', SOLE)):
            for src, groups in SRC_GROUPS.items():
                for sn, hs in SETS.items():
                    arms[f'{bg}|{src}:{sn}#{k}'] = {'hide': hide, 'patch': [[g, hs] for g in groups], 'donor': don, 'rel': rels[k]}
        arms[f'nat|NH:rand#{k}'] = {'hide': [], 'patch': [[g, rnd] for g in ('names', 'holders')], 'donor': don, 'rel': rels[k]}
        for name, groups in (('swap', []), ('swap-N', ['names']), ('swap-H', ['holders']), ('swap-NH', ['names', 'holders'])):
            arms[f'{name}#{k}'] = {'hide': [[g, PB, PB + 1, C] for g in groups], 'swap': True, 'donor': don, 'rel': rels[k]}
        for name, pt in (('swap-Hv0', [['holders', C, 'zero']]), ('swap-Nv0', [['names', C, 'zero']]), ('swap+Hdon', [['holders', C, 'donor']])):
            arms[f'{name}#{k}'] = {'hide': [], 'patch': pt, 'swap': True, 'donor': don, 'rel': rels[k]}
        arms[f'nat|Hv0:C#{k}'] = {'hide': [], 'patch': [['holders', C, 'zero']], 'donor': don, 'rel': rels[k]}
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
                hid = {g: flags(fn) for g, fn in GROUP.items()}
                mg = {g: flags(lambda r, g=g: FE.keys(r)[0][g]) for g in FE.GROUPS}
                P0 = min(off[i] + AR.names_of(r)[0] for i, r in enumerate(rs))
                for name, a in {**arms, 'noop': {'hide': []}}.items():
                    ms = [(hid[g], b0, b1, hs) for g, b0, b1, hs in a['hide']]
                    pt, idx, sw, P = [], None, None, length - 1
                    local = [j - c0 for j in a['donor'][c0:c0 + N]] if 'donor' in a else None
                    if local is not None:
                        assert all(0 <= j < N and rs[j]['family'] == r['family'] for j, r in zip(local, rs))
                    pairs = lambda fn: tuple(t.tensor(v, device='cuda') for v in (
                        [i for i, r in enumerate(rs) for _ in fn(r)], [off[i] + x for i, r in enumerate(rs) for x in fn(r)],
                        [j for j in local for _ in fn(rs[j])], [off[j] + y for j in local for y in fn(rs[j])]))
                    for g, hs, *mode in a.get('patch', []):
                        pt.append((pairs(GROUP[g]), t.tensor(hs, device='cuda'), (mode or ['donor'])[0]))
                    if a.get('swap'):
                        idx, sw, P = pairs(AR.names_of), L, P0
                    first = 0 if name == 'noop' else min([PB if pt else NB] + ([sw] if sw is not None else []) + [m[1] for m in ms])
                    plan[name] = (first, P, sw, idx, ms, pt)
            state, cap = {}, {}
            for b in range(NB):
                for name, (first, P, sw, idx, ms, pt) in plan.items():
                    if first == b:
                        state[name] = h[:, P:].clone()
                    if sw == b:
                        state[name][idx[0], idx[1] - P] = h[idx[2], idx[3]]
                layer = rt.block(b)
                a = layer.self_attn
                handles = []
                if state:
                    handles = [a.k_proj.register_forward_hook(lambda m, i, o: cap.__setitem__('k', o)),
                               a.v_proj.register_forward_hook(lambda m, i, o: cap.__setitem__('v', o))]
                y = layer(h, attention_mask=mask, position_ids=pos, position_embeddings=(cos, sin), use_cache=False)
                kb, vb = cap.get('k'), cap.get('v')
                for name in state:
                    first, P, sw, idx, ms, pt = plan[name]
                    T = length - P
                    x = state[name]
                    tm = mask[:, :, P:, :]
                    act = [(m, hs) for m, b0, b1, hs in ms if b0 <= b < b1]
                    if act:
                        tm = tm.expand(N, NH, T, length).clone()
                        for m, hs in act:
                            if hs is None:
                                tm[:, :, -1, :] &= ~m[:, None]
                            else:
                                tm[:, hs, -1, :] &= ~m[:, None]
                    hn = layer.input_layernorm(x)
                    q = a.q_norm(a.q_proj(hn).view(N, T, NH, hd)).transpose(1, 2)
                    k = a.k_norm(t.cat([kb[:, :P], a.k_proj(hn)], 1).view(N, length, kv, hd)).transpose(1, 2)
                    v = t.cat([vb[:, :P], a.v_proj(hn)], 1).view(N, length, kv, hd).transpose(1, 2)
                    k, v = rt.mod.repeat_kv(k, grp), rt.mod.repeat_kv(v, grp)
                    q = rot(q, cos[:, None, P:], sin[:, None, P:])
                    last = None
                    if b == PB and pt:      # the exchanged keys / values are seen by the attention row of the answer position only
                        k2, v2 = k.clone(), v.clone()
                        for (bi, bp, di, dp), hs, mode in pt:
                            if mode == 'zero':
                                v2[bi[:, None], hs[None], bp[:, None]] = 0
                            else:           # donor keys / values of its unedited run (kb, vb)
                                kd = rt.mod.repeat_kv(a.k_norm(kb.view(N, length, kv, hd)).transpose(1, 2), grp)
                                vd = rt.mod.repeat_kv(vb.view(N, length, kv, hd).transpose(1, 2), grp)
                                k2[bi[:, None], hs[None], bp[:, None]] = kd[di[:, None], hs[None], dp[:, None]]
                                v2[bi[:, None], hs[None], bp[:, None]] = vd[di[:, None], hs[None], dp[:, None]]
                                del kd, vd
                        last = F.scaled_dot_product_attention(q[:, :, -1:], rot(k2, cos[:, None], sin[:, None]), v2, attn_mask=tm[:, :, -1:])
                        del k2, v2
                    k = rot(k, cos[:, None], sin[:, None])
                    if b == PB and name == 'noop':
                        w = (q[:, C, -1:, :].float() @ k[:, C].float().transpose(2, 3) / hd ** .5)[:, :, 0]
                        w = w.masked_fill(~mask[:, 0, -1, :][:, None], float('-inf')).softmax(-1)
                        for g, m in mg.items():
                            mass[g] = mass.get(g, 0) + (w * m[:, None]).sum(-1).sum(0).cpu().numpy()
                        mass_n += N
                    o = F.scaled_dot_product_attention(q, k, v, attn_mask=tm)
                    if last is not None:
                        o[:, :, -1:] = last
                    x = x + a.o_proj(o.transpose(1, 2).reshape(N, T, -1))
                    state[name] = x + layer.mlp(layer.post_attention_layernorm(x))
                    del q, k, v, o
                h = y
                for x in handles:
                    x.remove()
                cap.clear()
                del layer, y, a, kb, vb
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
        meta.update({'noop_max_score_difference': diff('noop'), 'self_exchange_max_score_difference': diff('self:NH:C'),
                     'sole_alone_accuracy': BK.acc(results, 'sole')})
        meta['controls_ok'] = bool(same('noop') and same('self:NH:C') and diff('noop') <= TOL and diff('self:NH:C') <= TOL
                                   and meta['sole_alone_accuracy'] >= .90)
        meta['attention_mass'] = {g: {f'25.{hh}': round(float(v[i] / mass_n), 4) for i, hh in enumerate(C)} for g, v in mass.items()}
    return results, meta


def summarise(results, arms, rows):
    fam_id = {f: i for i, f in enumerate(sorted({r['family'] for r in rows}))}
    nf = len(fam_id)
    bs = np.random.default_rng(SEED).integers(0, nf, (1000, nf))
    fam = np.array([fam_id[r['family']] for r in rows])
    cells = {}
    for name, a in arms.items():
        if '#' not in name:
            continue
        c = cells.setdefault(name.split('#')[0], {k: [] for k in ('fam', 'B', 'P', 'I', 'out', 'dP', 'dI', 'rel')})
        ref = results['sole' if name.startswith('sole|') else 'base']
        for i, (r, j, rel) in enumerate(zip(rows, a['donor'], a['rel'])):
            x, d = results[name][i], rows[j]
            Bq, Pq, Iq = r['target_record'], AR.pointer(r, d, 'names'), d['target_record']
            c['fam'].append(fam[i]); c['B'].append(Bq); c['P'].append(Pq); c['I'].append(Iq); c['out'].append(x['pred_record'])
            c['rel'].append(''.join(map(str, rel)))
            sh = lambda q: (x['scores'][q] - x['scores'][Bq]) - (ref[i]['scores'][q] - ref[i]['scores'][Bq])
            c['dP'].append(sh(Pq)); c['dI'].append(sh(Iq))
    A = {k: {x: np.array(v) for x, v in c.items()} for k, c in cells.items()}

    def rate(v, f, keep):
        v = np.asarray(v, float)
        per, cnt = np.bincount(f[keep], weights=v[keep], minlength=nf), np.bincount(f[keep], minlength=nf)
        b = per[bs].sum(1) / np.maximum(cnt[bs].sum(1), 1)
        return {'rate': round(float(v[keep].mean()), 4), 'ci95': [round(float(np.quantile(b, .025)), 4), round(float(np.quantile(b, .975)), 4)]}

    out = {'base_correct': BK.acc(results, 'base'), 'cells': {}, 'paired': {}, 'relations': {}}
    for k, c in A.items():
        three = (c['P'] != c['B']) & (c['I'] != c['B']) & (c['P'] != c['I'])
        pc, ic = c['P'] != c['B'], c['I'] != c['B']
        out['cells'][k] = {
            'n': len(three), 'changed': rate(c['out'] != c['B'], c['fam'], three | ~three),
            'three_way': {'n': int(three.sum()), 'P': rate(c['out'] == c['P'], c['fam'], three), 'I': rate(c['out'] == c['I'], c['fam'], three),
                          'B': round(float((c['out'] == c['B'])[three].mean()), 4)},
            'follow_P_when_P_differs': rate(c['out'] == c['P'], c['fam'], pc), 'follow_I_when_I_differs': rate(c['out'] == c['I'], c['fam'], ic),
            'score_shift_to_P': rate(c['dP'], c['fam'], three), 'score_shift_to_I': rate(c['dI'], c['fam'], three),
            'score_shift_P_minus_I': rate(c['dP'] - c['dI'], c['fam'], three)}
    ref = A['swap']
    pc = ref['P'] != ref['B']
    for k in ('swap-N', 'swap-H', 'swap-NH', 'swap-Hv0', 'swap-Nv0', 'swap+Hdon'):
        out['paired'][f'{k}-swap'] = rate((A[k]['out'] == ref['P']).astype(float) - (ref['out'] == ref['P']).astype(float), ref['fam'], pc)
    fol = lambda k: (A[k]['out'] == ref['P']).astype(float)
    out['paired']['swap-H-swap-Hv0'] = rate(fol('swap-H') - fol('swap-Hv0'), ref['fam'], pc)
    out['paired']['swap+Hdon-swap-H'] = rate(fol('swap+Hdon') - fol('swap-H'), ref['fam'], pc)
    for k in ('nat|N:C', 'nat|H:C', 'nat|NH:C', 'sole|N:C', 'sole|H:C', 'sole|NH:C'):
        c = A[k]
        three = (c['P'] != c['B']) & (c['I'] != c['B']) & (c['P'] != c['I'])
        for rel in sorted(set(c['rel'][three])):
            m = three & (c['rel'] == rel)
            out['relations'][f'{k}<{rel}'] = {'n': int(m.sum()), 'P': round(float((c['out'] == c['P'])[m].mean()), 4),
                                              'I': round(float((c['out'] == c['I'])[m].mean()), 4)}
    return out


def verdicts(s, meta):
    tri = lambda ok, bad: 'supported' if ok else 'not_supported' if bad else 'inconclusive'
    v = {'controls_valid': bool(meta['controls_ok'])}
    if not v['controls_valid']:
        return {**v, **{k: 'controls_invalid' for k in THRESHOLDS}}
    c, P = s['cells'], s['paired']
    n, h, r = c['nat|N:C'], c['nat|H:C'], c['nat|NH:rand']
    sp, pi = n['score_shift_to_P'], n['score_shift_P_minus_I']
    v['D1_names_carry_a_pointer'] = tri(sp['rate'] >= 2 and sp['ci95'][0] > 0 and pi['ci95'][0] > 0 and abs(r['score_shift_to_P']['rate']) < .5,
                                        sp['rate'] < 1 or pi['ci95'][0] <= 0)
    hp = h['score_shift_to_P']['rate']
    v['D2_holders_carry_little_alone'] = tri(h['changed']['rate'] <= .02 and hp < sp['rate'] / 4, h['changed']['rate'] > .05 or hp > sp['rate'] / 2)
    d = P['swap-N-swap']
    v['D3_names_feed_the_new_pointer_in_conflict'] = tri(d['rate'] <= -.20 and d['ci95'][1] < 0, d['rate'] > -.10)
    d = P['swap-Hv0-swap']
    v['D4_pull_back_is_partly_carried_content'] = tri(d['rate'] >= .03 and d['ci95'][0] > 0, d['ci95'][0] <= 0)
    v['D5_pull_back_is_partly_attention_share'] = tri(P['swap-H-swap-Hv0']['ci95'][0] > 0, True)
    d = abs(P['swap+Hdon-swap-H']['rate'])
    v['D6_donor_holders_add_no_push'] = tri(d <= .03, d > .06)
    return v


def show(s, meta):
    print(json.dumps({'meta': meta}), flush=True)
    for k, c in s['cells'].items():
        print(k, c['three_way']['n'], 'chg', c['changed']['rate'], '3way P/I/B', c['three_way']['P']['rate'], c['three_way']['I']['rate'], c['three_way']['B'],
              'folP', c['follow_P_when_P_differs']['rate'], 'folI', c['follow_I_when_I_differs']['rate'],
              'dP', c['score_shift_to_P']['rate'], 'dI', c['score_shift_to_I']['rate'], 'dP-dI', c['score_shift_P_minus_I'], flush=True)
    print(json.dumps(s['paired']), flush=True)


def smoke():
    from deploy.storage_policy import checked_dir
    checked_dir(OUT, 1 << 30, operation='来源绑定 25.18 分解产物')
    rows = [r for r in D.read(NL.OUT / 'pilot_name-materials.json') if int(r['family'][-4:]) < 2]
    arms = final_arms(rows)
    results, meta = engine(rows, arms, stage='smoke')
    s = summarise(results, arms, rows)
    print(json.dumps({'meta': {k: v for k, v in meta.items() if k != 'attention_mass'}, 'arms': len(arms), 'cells': len(s['cells'])}), flush=True)


def pilot():
    """Opened rows: the first 16 families of holdout_bak (Phase 414)."""
    from deploy.storage_policy import checked_dir
    checked_dir(OUT, 1 << 30, operation='来源绑定 25.18 分解产物')
    rows = [r for r in D.read(BK.OUT / f'{BK.SPLIT}-materials.json') if int(r['family'][-4:]) < 16]
    arms = final_arms(rows)
    results, meta = engine(rows, arms, stage='pilot')
    s = summarise(results, arms, rows)
    save('pilot.json', {'code_sha256': D.sha(__file__), 'panel': 'holdout_bak families 0-15 (opened in Phase 414)', 'summary': s, **meta})
    show(s, meta)


DEPS = (RL, PT, NL, F2, FE, AR, BK)


def prereg():
    from tools import provenance
    from tools.power_analysis import check_phase
    assert THRESHOLDS and not (OUT / 'prereg.json').exists()
    plan = D.checked(MODEL)
    rows = build_materials()
    assert len(rows) == FAMILIES * 16 and len({tuple(r['tokens']) for r in rows}) == len(rows)
    arms = final_arms(rows)
    save(f'{SPLIT}-materials.json', rows)
    save('design.json', arms)
    three = int(sum(len({r['target_record'], AR.pointer(r, rows[j], 'names'), rows[j]['target_record']}) == 3
                    for k in range(REPS) for r, j in zip(rows, arms[f'nat|N:C#{k}']['donor'])))
    body = {'schema': 'source_binding_h2518_prereg.v1', 'run_id': RUN, 'seed': SEED, 'created_at': datetime.now().astimezone().isoformat(),
            'base_plan_sha256': D.sha(SRC / 'plan.json'), 'model_snapshot': plan['models'][MODEL], 'code_sha256': D.sha(__file__),
            'dependency_sha256': {m.__name__: D.sha(m.__file__) for m in DEPS},
            'criteria': CRITERIA, 'thresholds': THRESHOLDS, 'pilot_sha256': D.sha(OUT / 'pilot.json'),
            'materials': {'sha256': D.sha(OUT / f'{SPLIT}-materials.json'), 'families': FAMILIES, 'rows': len(rows)},
            'design': {'sha256': D.sha(OUT / 'design.json'), 'arms': len(arms), 'three_way_edits': three},
            'power': {'unit': 'paired binary outcome per edit (choice is P / choice is I); alpha = 0.05',
                      'P_against_I': {'expected_delta': 0.15, 'p_d': 0.40, **check_phase(0.15, three, p_d=0.40)},
                      'as_if_one_edit_per_item': {'expected_delta': 0.15, 'p_d': 0.40, **check_phase(0.15, len(rows) // 2, p_d=0.40)},
                      'caveat': 'edits are clustered in 768 items and 48 families; intervals are family bootstraps'},
            'custody': 'holdout_h25 is read first by the gate stage (behaviour only) and then once by the run stage.',
            'resources': {'gpu_minutes_estimate': 10, 'disk_GiB_estimate': 0.1, 'models_concurrent': 1,
                          'shape': 'chunks of 192 prompts in one batch; fixed before pre-registration'},
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
    save('gate.json', {'schema': 'source_binding_h2518_gate.v1', 'run_id': RUN, 'items_sha256': D.sha(OUT / 'gate-items.json'), **g, **meta})
    print(json.dumps({k: g[k] for k in ('passed', 'forced_accuracy', 'top1_accuracy')}), flush=True)


def run():
    from tools import provenance
    pre = checked()
    assert not (OUT / 'report.json').exists(), 'holdout already used'
    g = D.read(OUT / 'gate.json')
    rows, arms = D.read(OUT / f'{SPLIT}-materials.json'), D.read(OUT / 'design.json')
    if g['passed']:
        results, meta = engine(rows, arms, stage='run')
        save('items.json', {k: [{'id': x['id'], 'pred_record': x['pred_record'], 'scores': x['scores']} for x in v] for k, v in results.items()})
        base = D.read(OUT / 'gate-items.json')
        meta['base_replay_max_score_difference'] = max(abs(a - b) for x, y in zip(results['base'], base) for a, b in zip(x['scores'], y['scores']))
        s = summarise(results, arms, rows)
        v = verdicts(s, meta)
        show(s, meta)
    else:
        s, meta, v = {'note': 'behaviour gate failed; no edit was run'}, {}, {k: 'behavior_ineligible' for k in THRESHOLDS}
    report = {'schema': 'source_binding_h2518_report.v1', 'run_id': RUN, 'split': SPLIT, 'gate_passed': g['passed'],
              'prereg_sha256': D.sha(OUT / 'prereg.json'), 'gate_sha256': D.sha(OUT / 'gate.json'),
              'items_sha256': D.sha(OUT / 'items.json') if g['passed'] else None, 'criteria': pre['criteria'], 'thresholds': pre['thresholds'],
              'verdicts': v, 'summary': s, **meta, 'formal_evidence_promotion': 0,
              'custody': 'holdout_h25 read once by the gate stage and once by this run',
              'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    save('report.json', report)
    path = provenance.write_result('SOURCE-BINDING-h2518-holdout', {'verdicts': v, 'gate_passed': g['passed'],
                                                                  'report_sha256': D.sha(OUT / 'report.json')},
                                   model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)
    print(json.dumps({'verdicts': v, 'result': str(path)}, ensure_ascii=False), flush=True)


STAGES = {f.__name__: f for f in (smoke, pilot, prereg, gate, run)}
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=list(STAGES))
    STAGES[ap.parse_args().stage]()
