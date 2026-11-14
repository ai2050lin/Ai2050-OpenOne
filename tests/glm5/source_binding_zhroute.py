"""Which positions carry the exchanged names to the answer in the Chinese task (Claude Phase 432).  Qwen3-14B BF16.

Phase 426: in the Chinese task (zh) the answer position reading the new names directly in blocks L..27 explains only a
small part of the broad edit (0.812 against no effect 0.775); in the English colours task it explains nearly all
(0.927).  Here the broad edit is restricted by position: after the name tokens are exchanged at the input of block L,
every position outside a chosen set is reset to its base-run state after each block, so the exchanged content can reach
the answer position only through the name tokens themselves and the chosen positions.

Position groups (between the first question name and the answer position, names excluded):
  mid    between the two question names          qtail  after the last name, up to the end of the question
  instr  the instruction line after the question tpl    chat-template tokens after the content (and the answer cue)
Arms per cell (site x block L): swap (nothing reset = the broad edit), direct (all groups reset), via:G (only G free),
drop:G (all free but G).  The prediction is the broad edit's answer.
Stages: smoke, pilot (opened holdout panels of Phase 426) -> prereg -> gate -> run.
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
from source_binding_runtime import Runtime

SRC, MODEL = D.OUT, 'qwen14'
OUT = D.ROOT / 'tests/glm5/result/source_binding_zhroute_v1'
RUN, SEED = 'SOURCE-BINDING-ZHROUTE-20261113-v1', 202611130432
TASKS = ('zh', 'colours')
CHUNK, NB, NH, TOL, REPS = 96, 40, 40, 1.0, 2
SITES, LAYERS = ('qsrc', 'qent', 'names'), (23, 24)
CELLS = [f'{s}@{l}' for s in SITES for l in LAYERS]
GROUPS = ('mid', 'qtail', 'instr', 'end', 'think', 'cue')     # a partition of the holder positions
ALIAS = {'tpl': ('end', 'think', 'cue')}
ARMS = ['direct', 'via:tpl', 'drop:tpl', 'via:end', 'via:think', 'via:cue', 'drop:end', 'drop:think', 'drop:cue', 'via:mid', 'via:qtail', 'via:instr']
PILOT_FAMILIES = 24
CONFIRM = {'zh': {'sources': ['橙队', '灰队', '粉队', '棕队', '赤队'],
                  'items': ['林峰', '罗兰', '梁超', '宋佳', '唐宁', '韩冰', '曹阳', '邓丽', '彭涛', '萧然']},
           'colours': {'sources': ['George', 'Helen', 'Ivan', 'Julia', 'Kevin', 'Linda'],
                       'items': ['towel', 'wallet', 'helmet', 'pencil', 'saucer', 'drawer', 'barrel', 'button', 'spoon', 'rope']}}


def save(name, body):
    from deploy.storage_policy import check_write
    p = OUT / name
    check_write(p, 1 << 20, operation='来源绑定中文路线产物')
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(body, ensure_ascii=False, indent=1, default=lambda x: x.item() if isinstance(x, np.generic) else x), encoding='utf-8')
    t.replace(p)


def log(*a, **kw):
    RL.OUT = OUT
    RL.log(*a, **kw)


def annotate(rows):
    """Adds r['grp']: token positions of the groups mid / qtail / instr / tpl (a partition of the holder positions)."""
    tok = RL.tokenizer()
    for r in rows:
        offs = tok(r['prompt'], add_special_tokens=False, return_offsets_mapping=True)['offset_mapping']
        assert len(offs) == len(r['tokens'])
        cs = r['prompt'].index(r['content'])
        qend = cs + r['content'].index(TR.TEXT[r['task']]['tail'])
        cend = cs + len(r['content'])
        nm = AR.names_of(r)
        lo, hi = min(nm), max(nm)
        g = {k: [] for k in ('mid', 'qtail', 'instr', 'tpl')}
        for i in AR.holders_of(r):
            st = offs[i][0]
            g['mid' if i < hi else 'qtail' if st < qend else 'instr' if st < cend else 'tpl'].append(i)
        tp = g.pop('tpl')
        assert len(tp) == 10 and tp == list(range(len(r['tokens']) - 11, len(r['tokens']) - 1))
        g.update(end=tp[:5], think=tp[5:9], cue=tp[9:])     # <|im_end|> nl <|im_start|> assistant nl | <think> .. </think> nl nl | answer cue word
        assert all(g.values()) and sorted(sum(g.values(), [])) == AR.holders_of(r) and set(g) == set(GROUPS)
        r['grp'] = g
    return rows


def arms_for(rows, key, cells=CELLS):
    dons, rels = RH.donors(rows, f'{RUN}|{key}', SEED)
    me = list(range(len(rows)))
    arms = {'self': {'swap': ['names', 23], 'donor': me, 'free': list(GROUPS)}, 'self:direct': {'swap': ['names', 23], 'donor': me, 'free': []}}
    for k in range(REPS):
        base = {'donor': dons[k], 'rel': rels[k]}
        for cell in cells:
            site, l = cell.split('@')[0], int(cell.split('@')[1])
            sw = {'swap': [site, l], **base}
            arms[f'swap|{cell}#{k}'] = {**sw, 'free': list(GROUPS)}
            for n in ARMS:
                kind, _, g = n.partition(':')
                ex = list(ALIAS.get(g, (g,)))
                free = [] if kind == 'direct' else ex if kind == 'via' else [x for x in GROUPS if x not in ex]
                arms[f'{n}|{cell}#{k}'] = {**sw, 'free': free}
    return arms


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
            plan = {}
            if arms:
                P0 = min(off[i] + AR.names_of(r)[0] for i, r in enumerate(rs))
                cache = {}
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
                    frozen = [g for g in GROUPS if g not in a['free']]
                    fz = None
                    if frozen:
                        fk = ('fz', tuple(frozen))
                        if fk not in cache:
                            cache[fk] = tuple(t.tensor(v, device='cuda') for v in (
                                [i for i, r in enumerate(rs) for g in frozen for _ in r['grp'][g]],
                                [off[i] + x for i, r in enumerate(rs) for g in frozen for x in r['grp'][g]]))
                        fz = cache[fk]
                    plan[name] = {'first': l, 'P': P0, 'idx': cache[key], 'fz': fz}
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
                    o = F.scaled_dot_product_attention(q, rot(k, cos[:, None], sin[:, None]), v, attn_mask=mask[:, :, P:, :])
                    x = x + a.o_proj(o.transpose(1, 2).reshape(N, T, -1))
                    x = x + layer.mlp(layer.post_attention_layernorm(x))
                    if p['fz'] is not None:          # positions outside the free groups go back to the base run
                        fi, fp = p['fz']
                        x[fi, fp - P] = y[fi, fp]
                    state[name] = x
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
        meta.update({f'{k}_max_score_difference': diff(k) for k in ('self', 'self:direct')})
        meta['controls_ok'] = bool(all(same(k) and diff(k) <= TOL for k in ('self', 'self:direct')))
    return results, meta


def table(rows, arms, items):
    fam_id = {f: i for i, f in enumerate(sorted({r['family'] for r in rows}))}
    T = defaultdict(list)
    for cell in sorted({n.split('|')[1].split('#')[0] for n in arms if n.startswith('swap|')}):
        site, l = cell.split('@')[0], int(cell.split('@')[1])
        for k in range(REPS):
            a = arms[f'swap|{cell}#{k}']
            for i, (r, j) in enumerate(zip(rows, a['donor'])):
                B, P, I = F3.records(r, rows[j], site, l)
                o = items[f'swap|{cell}#{k}'][i]
                T['fam'].append(fam_id[r['family']]); T['cell'].append(cell); T['frame'].append(r['frame'])
                T['B'].append(B); T['P'].append(P); T['out'].append(o['pred_record'])
                bs = items['base'][i]['scores']
                T['nat'].append(bs[P] - bs[B] if P != B else 0.0)
                T['xs'].append(o['scores'][P] - o['scores'][B] if P != B else 0.0)
                for n in ARMS:
                    it = items[f'{n}|{cell}#{k}'][i]
                    T[n].append(it['pred_record'])
                    T['x:' + n].append(it['scores'][P] - it['scores'][B] if P != B else 0.0)
    return {k: np.array(v) for k, v in T.items()}, len(fam_id)


def summarise(results, arms, rows):
    T, nf = table(rows, arms, results)
    draws = np.random.default_rng(SEED).integers(0, nf, (1000, nf))
    fam, out, B = T['fam'], T['out'], T['B']
    r4 = lambda v: round(float(v), 4)

    def rate(v, keep):
        v = np.asarray(v, float)
        per, cnt = np.bincount(fam[keep], weights=v[keep], minlength=nf), np.bincount(fam[keep], minlength=nf)
        bs = per[draws].sum(1) / np.maximum(cnt[draws].sum(1), 1)
        return {'rate': r4(v[keep].mean()), 'ci95': [r4(np.quantile(bs, .025)), r4(np.quantile(bs, .975))]}

    def share(num, keep):         # share of the broad edit's score shift (conflict edits)
        n_ = np.bincount(fam[keep], weights=num[keep], minlength=nf)
        d_ = np.bincount(fam[keep], weights=(T['xs'] - T['nat'])[keep], minlength=nf)
        bs = n_[draws].sum(1) / d_[draws].sum(1)
        return {'rate': r4(n_.sum() / d_.sum()), 'ci95': [r4(np.quantile(bs, .025)), r4(np.quantile(bs, .975))]}
    hit = {n: (T[n] == out).astype(float) for n in ARMS}
    hit['none'] = (B == out).astype(float)
    changed, conflict = out != B, T['P'] != B
    layer = np.array([int(c.split('@')[1]) for c in T['cell']])
    scopes = {'all': np.ones(len(out), bool), **{f'block_{l}': layer == l for l in LAYERS},
              **{c: T['cell'] == c for c in CELLS}, **{f'frame_{f}': T['frame'] == f for f in range(4)}}
    scopes = {k: v for k, v in scopes.items() if v.any()}
    s = {'edits': int(len(out)), 'scopes': {}}
    for name, keep in scopes.items():
        c = keep & conflict
        s['scopes'][name] = {'n': int(keep.sum()), 'broad_changed': r4(changed[keep].mean()),
                             'agreement': {n: rate(v, keep) for n, v in hit.items()},
                             'recovered_of_changed': {n: r4(hit[n][keep & changed].mean()) if (keep & changed).any() else None for n in ARMS},
                             'shift_share': {n: share(T['x:' + n] - T['nat'], c) for n in ARMS} if c.any() else None,
                             **{f'{a}_minus_{b}': rate(hit[a] - hit[b], keep) for a, b in
                                [('direct', 'none'), ('via:tpl', 'direct'), ('via:tpl', 'none'), ('via:cue', 'direct'), ('via:end', 'direct'),
                                 ('via:think', 'direct'), ('via:qtail', 'direct'), ('via:instr', 'direct'), ('drop:tpl', 'none'), ('drop:cue', 'none')]}}
        s['scopes'][name]['drop_loss'] = {n[5:]: rate(1 - hit[n], keep) for n in ARMS if n.startswith('drop:')}
    return s


def show(s, scopes=('all', 'block_23', 'block_24') + tuple(CELLS)):
    ks = ARMS + ['none']
    lines = ['scope      chg    ' + ' '.join(f'{k[:9]:>9s}' for k in ks) + '   | shift share direct / via:tpl / end / think / cue']
    for n in scopes:
        v = s['scopes'][n]
        sh = v['shift_share']
        lines.append(f"{n:10s} {v['broad_changed']:.3f}  " + ' '.join(f"{v['agreement'][k]['rate']:9.3f}" for k in ks)
                     + ('   | ' + ' '.join(f"{sh[k]['rate']:6.3f}" for k in ['direct', 'via:tpl', 'via:end', 'via:think', 'via:cue']) if sh else ''))
    return '\n'.join(lines)


def smoke():
    from deploy.storage_policy import checked_dir
    checked_dir(OUT, 1 << 30, operation='来源绑定中文路线产物')
    rows = annotate([r for r in D.read(TR.OUT / 'zh-pilot-materials.json') if int(r['family'][-4:]) < 2])
    arms = arms_for(rows, 'smoke', ['names@23'])
    results, meta = engine(rows, arms, stage='smoke')
    T, _ = table(rows, arms, results)
    print(json.dumps({'meta': meta, 'arms': len(arms), 'agreement': {n: round(float((T[n] == T['out']).mean()), 3) for n in ARMS},
                      'groups_example': {g: rows[3]['grp'][g] for g in GROUPS}}), flush=True)


def pilot():
    """Opened holdout panels of Phase 426, first 24 families per task, new donor draws."""
    from deploy.storage_policy import checked_dir
    checked_dir(OUT, 1 << 30, operation='来源绑定中文路线产物')
    for task in TASKS:
        rows = D.read(TR.OUT / f'{task}-holdout-materials.json')
        fams = sorted({r['family'] for r in rows})[:PILOT_FAMILIES]
        rows = annotate([r for r in rows if r['family'] in fams])
        arms = arms_for(rows, f'{task}|pilot')
        results, meta = engine(rows, arms, stage=f'pilot-{task}')
        s = summarise(results, arms, rows)
        save(f'{task}-pilot-summary.json', {'meta': meta, 'families': len(fams), **s})
        print(task, json.dumps(meta), flush=True)
        print(show(s), flush=True)


def pilot2():
    """Template positions split (end / think / cue): zh, the other 24 opened families of the Phase 426 holdout, new draws."""
    rows = D.read(TR.OUT / 'zh-holdout-materials.json')
    fams = sorted({r['family'] for r in rows})[PILOT_FAMILIES:]
    rows = annotate([r for r in rows if r['family'] in fams])
    arms = arms_for(rows, 'zh|pilot2')
    results, meta = engine(rows, arms, stage='pilot2-zh')
    s = summarise(results, arms, rows)
    save('zh-pilot2-summary.json', {'meta': meta, 'families': len(fams), **s})
    print('zh', json.dumps(meta), flush=True)
    print(show(s), flush=True)


def build_confirm(task):
    """New panel: source and item pools disjoint from the pilot and holdout pools of Phase 426; answer words shared."""
    for split in ('pilot', 'holdout'):
        assert not set(CONFIRM[task]['sources'] + CONFIRM[task]['items']) & set(TR.POOLS[task]['sources'][split] + TR.POOLS[task]['items'][split])
    TR.POOLS[task]['sources']['confirm'] = CONFIRM[task]['sources']
    TR.POOLS[task]['items']['confirm'] = CONFIRM[task]['items']
    TR.FAMILIES['confirm'] = 48
    return annotate(TR.build(task, 'confirm'))


def materials():
    from deploy.storage_policy import checked_dir
    checked_dir(OUT, 1 << 30, operation='来源绑定中文路线产物')
    for task in TASKS:
        rows = build_confirm(task)
        H25.RUN, H25.SEED = RUN, SEED
        H25.draws(rows)
        print(task, len(rows), 'tokens', min(len(r['tokens']) for r in rows), max(len(r['tokens']) for r in rows),
              'qsrc', sorted({len(r['up']['qsrc']) for r in rows}), 'qent', sorted({len(r['up']['qent']) for r in rows}),
              {g: sorted({len(r['grp'][g]) for r in rows}) for g in GROUPS}, flush=True)



# ---- frozen after pilots 1 and 2 (opened holdout panels of Phase 426) ----
THRESHOLDS = {
    'Z1_template_carries_it_in_zh': 'zh: via:tpl >= 0.94 and via:tpl minus direct >= 0.12 with a lower bound > 0. not_supported if the '
                                    'lower bound <= 0.',
    'Z2_the_end_of_turn_tokens': 'zh: via:end >= 0.92 and via:end minus direct >= 0.10 with a lower bound > 0; via:cue and via:think '
                                 'exceed direct by <= 0.03; drop:end exceeds direct by <= 0.05. not_supported if the via:end lower '
                                 'bound <= 0.',
    'Z3_not_the_question_positions': 'zh: via:qtail, via:instr and via:mid each exceed direct by <= 0.03. not_supported if any > 0.08.',
    'C1_colours_direct_reading': 'colours: direct >= 0.90 and minus no effect >= 0.20 with a lower bound > 0; via:tpl minus direct '
                                 '<= 0.06. not_supported if direct < 0.85.',
    'C2_contrast_in_score_shift': 'share of the broad edit\'s score shift reached by direct: zh <= 0.60 and colours >= 0.80 (upper bound '
                                  'of zh below lower bound of colours). not_supported if the intervals overlap.',
}
CRITERIA = {
    'model': 'Qwen3-14B original BF16, CPU byte reads, CUDA layer streaming. GLM-4-9B is not run.',
    'material': 'the two tasks of Phase 426 (zh: Chinese prompt, team names as sources, personal names as items, one-character '
                'animals; colours: English, first names, common nouns, colour words). New panels: 48 families per task (768 prompts); '
                'source and item pools disjoint from the pilot and holdout pools of Phase 426; answer words shared.',
    'broad_edit': 'at the input of block L (23, 24) the residual states of the source name in the question (qsrc), the item name (qent) '
                  'or both are replaced by those of a donor of the same family. 2 seeded donor relations per item; 6 cells x 1,536 edits '
                  'per task.',
    'restriction': 'from block L on, after every block, every position between the first question name and the answer position that '
                   'is not in a free group is reset to its base-run state; the name tokens and the answer position are always free. '
                   'Groups (a partition): mid (between the two names), qtail (rest of the question), instr (instruction line), end '
                   '(<|im_end|> nl <|im_start|> assistant nl), think (<think> nl nl </think> nl nl), cue (the answer cue word); tpl = '
                   'end + think + cue. Arms: direct (no group free), via:G (only G), drop:G (all but G).',
    'prediction': 'the broad edit gives the answer of the restricted edit. Agreement and the share of the broad edit\'s score shift '
                  '(score(P) - score(B) on conflict edits) are reported; family-bootstrap 95 % intervals (1000 draws).',
    'controls': 'the name exchange with the prompt itself, all groups free and no group free, changes nothing (same choice, score '
                'difference <= 1.0).',
    'two_pass': 'gate stage, per task: unmodified forward and the v1 behaviour gate. Edits run only for a task whose gate passed.',
    'predictions_on_holdout': THRESHOLDS,
    'not_covered': 'which of the five end-of-turn tokens and which heads there read the names, and which heads of the answer position '
                   'read them back; why the English task does not need this relay; exchanges at block 25; other models.',
    'disclosure': 'two pilots on the opened holdout panels of Phase 426 with new donor draws: pilot 1 (24 families per task) with the '
                  'template as one group; pilot 2 (the other 24 zh families) with the template split into end, think and cue. '
                  'Thresholds were set on these numbers. Smoke test on 2 exposed families of the zh pilot panel.',
}


def verdicts(S5, metas):
    tri = lambda ok, bad: 'supported' if ok else 'not_supported' if bad else 'inconclusive'
    v = {f'{t}_controls_valid': bool(metas.get(t, {}).get('controls_ok')) for t in TASKS}
    z, c = S5.get('zh'), S5.get('colours')
    keys = {'zh': ['Z1_template_carries_it_in_zh', 'Z2_the_end_of_turn_tokens', 'Z3_not_the_question_positions'],
            'colours': ['C1_colours_direct_reading']}
    for t, ks in keys.items():
        if t not in S5:
            v.update({k: 'behavior_ineligible' for k in ks})
        elif not v[f'{t}_controls_valid']:
            v.update({k: 'controls_invalid' for k in ks})
    ag = lambda S, n: S['scopes']['all']['agreement'][n]['rate']
    if z and v['zh_controls_valid']:
        A = z['scopes']['all']
        d = A['via:tpl_minus_direct']
        v['Z1_template_carries_it_in_zh'] = tri(ag(z, 'via:tpl') >= .94 and d['rate'] >= .12 and d['ci95'][0] > 0, d['ci95'][0] <= 0)
        e = A['via:end_minus_direct']
        v['Z2_the_end_of_turn_tokens'] = tri(ag(z, 'via:end') >= .92 and e['rate'] >= .10 and e['ci95'][0] > 0
                                             and ag(z, 'via:cue') - ag(z, 'direct') <= .03 and ag(z, 'via:think') - ag(z, 'direct') <= .03
                                             and ag(z, 'drop:end') - ag(z, 'direct') <= .05, e['ci95'][0] <= 0)
        q = [ag(z, f'via:{g}') - ag(z, 'direct') for g in ('qtail', 'instr', 'mid')]
        v['Z3_not_the_question_positions'] = tri(max(q) <= .03, max(q) > .08)
    if c and v['colours_controls_valid']:
        A = c['scopes']['all']
        d = A['direct_minus_none']
        v['C1_colours_direct_reading'] = tri(ag(c, 'direct') >= .90 and d['rate'] >= .20 and d['ci95'][0] > 0
                                             and ag(c, 'via:tpl') - ag(c, 'direct') <= .06, ag(c, 'direct') < .85)
    if z and c and v['zh_controls_valid'] and v['colours_controls_valid']:
        sz, sc = z['scopes']['all']['shift_share']['direct'], c['scopes']['all']['shift_share']['direct']
        v['C2_contrast_in_score_shift'] = tri(sz['rate'] <= .60 and sc['rate'] >= .80 and sz['ci95'][1] < sc['ci95'][0], sz['ci95'][1] >= sc['ci95'][0])
    else:
        v['C2_contrast_in_score_shift'] = 'behavior_ineligible'
    return v


DEPS = (RL, AR, F3, H25, RH, TR, RH.S)


def prereg():
    from deploy.storage_policy import checked_dir
    from tools import provenance
    from tools.power_analysis import check_phase
    checked_dir(OUT, 1 << 30, operation='来源绑定中文路线产物')
    assert not (OUT / 'prereg.json').exists()
    plan = D.checked(MODEL)
    mats, designs = {}, {}
    for task in TASKS:
        rows = build_confirm(task)
        arms = arms_for(rows, f'{task}|confirm')
        save(f'{task}-confirm-materials.json', rows)
        save(f'{task}-design.json', arms)
        mats[task] = {'sha256': D.sha(OUT / f'{task}-confirm-materials.json'), 'families': 48, 'rows': len(rows)}
        designs[task] = {'sha256': D.sha(OUT / f'{task}-design.json'), 'arms': len(arms)}
    n = 48 * 16
    body = {'schema': 'source_binding_zhroute_prereg.v1', 'run_id': RUN, 'seed': SEED, 'created_at': datetime.now().astimezone().isoformat(),
            'base_plan_sha256': D.sha(SRC / 'plan.json'), 'model_snapshot': plan['models'][MODEL], 'code_sha256': D.sha(__file__),
            'dependency_sha256': {m.__name__: D.sha(m.__file__) for m in DEPS},
            'criteria': CRITERIA, 'frozen': {'pilot_sha256': {f: D.sha(OUT / f) for f in ('zh-pilot-summary.json', 'colours-pilot-summary.json', 'zh-pilot2-summary.json')}},
            'materials': mats, 'design': designs,
            'power': {'unit': 'paired binary outcome per item (prediction right / wrong under via:end and under direct); alpha = 0.05',
                      'Z2_as_if_one_edit_per_item': {'expected_delta': 0.12, 'p_d': 0.20, **check_phase(0.12, n, p_d=0.20)},
                      'caveat': '9,216 edits per task clustered in 768 items and 48 families; intervals are family bootstraps'},
            'custody': 'each new panel is read first by the gate stage (behaviour only) and then once by the run stage.',
            'resources': {'gpu_minutes_estimate': 16, 'disk_GiB_estimate': 0.1, 'models_concurrent': 1,
                          'shape': 'chunks of 96 prompts in one batch; pilot 2 peak 6.9 GiB, 7.3 minutes for 24 families'},
            'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    body['provenance']['model_revision'] = plan['models'][MODEL]
    save('prereg.json', body)
    print(json.dumps({'prereg_sha256': D.sha(OUT / 'prereg.json'), 'design': designs, 'power': body['power']}), flush=True)


def checked():
    p = D.read(OUT / 'prereg.json')
    assert p['code_sha256'] == D.sha(__file__) and all(p['dependency_sha256'][m.__name__] == D.sha(m.__file__) for m in DEPS), \
        'source changed after pre-registration'
    assert all(p['materials'][t]['sha256'] == D.sha(OUT / f'{t}-confirm-materials.json') and p['design'][t]['sha256'] == D.sha(OUT / f'{t}-design.json')
               for t in TASKS)
    D.checked(MODEL)
    return p


def gate():
    checked()
    assert not (OUT / 'gate.json').exists(), 'new panels already read'
    out = {}
    for task in TASKS:
        rows = D.read(OUT / f'{task}-confirm-materials.json')
        results, meta = engine(rows, stage=f'gate-{task}')
        save(f'{task}-gate-items.json', results['base'])
        g = D.behavior(rows, results['base'])
        out[task] = {'items_sha256': D.sha(OUT / f'{task}-gate-items.json'), **g, **meta}
    save('gate.json', {'schema': 'source_binding_zhroute_gate.v1', 'run_id': RUN, 'tasks': out})
    print(json.dumps({t: {k: g[k] for k in ('passed', 'forced_accuracy', 'top1_accuracy')} for t, g in out.items()}), flush=True)


def run():
    from tools import provenance
    pre = checked()
    assert not (OUT / 'report.json').exists(), 'new panels already used'
    g = D.read(OUT / 'gate.json')['tasks']
    S5, metas = {}, {}
    for task in TASKS:
        if not g[task]['passed']:
            continue
        rows, arms = D.read(OUT / f'{task}-confirm-materials.json'), D.read(OUT / f'{task}-design.json')
        results, meta = engine(rows, arms, stage=f'run-{task}')
        save(f'{task}-items.json', {k: [{'id': x['id'], 'pred_record': x['pred_record'], 'scores': x['scores']} for x in v] for k, v in results.items()})
        base = D.read(OUT / f'{task}-gate-items.json')
        meta['base_replay_max_score_difference'] = max(abs(a - b) for x, y in zip(results['base'], base) for a, b in zip(x['scores'], y['scores']))
        S5[task], metas[task] = summarise(results, arms, rows), meta
        print(task, json.dumps(meta), flush=True)
        print(show(S5[task]), flush=True)
    v = verdicts(S5, metas)
    report = {'schema': 'source_binding_zhroute_report.v1', 'run_id': RUN, 'gate_passed': {t: g[t]['passed'] for t in TASKS},
              'prereg_sha256': D.sha(OUT / 'prereg.json'), 'gate_sha256': D.sha(OUT / 'gate.json'),
              'items_sha256': {t: D.sha(OUT / f'{t}-items.json') for t in S5}, 'criteria': pre['criteria'], 'frozen': pre['frozen'],
              'verdicts': v, 'summary': S5, 'meta': metas, 'formal_evidence_promotion': 0,
              'custody': 'each new panel read once by the gate stage and once by this run',
              'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    save('report.json', report)
    path = provenance.write_result('SOURCE-BINDING-zhroute-holdout', {'verdicts': v, 'gate_passed': all(g[t]['passed'] for t in TASKS),
                                                                    'gate_passed_by_task': {t: g[t]['passed'] for t in TASKS},
                                                                    'report_sha256': D.sha(OUT / 'report.json')},
                                   model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)
    print(json.dumps({'verdicts': v, 'result': str(path)}, ensure_ascii=False), flush=True)


def pilot_verdicts():
    """The verdict code applied to the pilot summaries (colours from pilot 1, which has no end / think / cue split)."""
    z, c = D.read(OUT / 'zh-pilot2-summary.json'), D.read(OUT / 'colours-pilot-summary.json')
    print(json.dumps(verdicts({'zh': z}, {'zh': z['meta']})), flush=True)
    print('colours direct', c['scopes']['all']['agreement']['direct'], 'share', c['scopes']['all']['shift_share']['direct'],
          'zh share', z['scopes']['all']['shift_share']['direct'], flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['smoke', 'pilot', 'pilot2', 'materials', 'pilot_verdicts', 'prereg', 'gate', 'run'])
    {'smoke': smoke, 'pilot': pilot, 'pilot2': pilot2, 'materials': materials, 'pilot_verdicts': pilot_verdicts, 'prereg': prereg, 'gate': gate, 'run': run}[ap.parse_args().stage]()
