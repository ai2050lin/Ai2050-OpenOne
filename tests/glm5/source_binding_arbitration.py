"""Who decides when the direct route and the backups disagree at block 24 (Claude Phase 413).  Qwen3-14B BF16.

Phase 412: a name token exchanged at the input of block 24 changes the answer on about a quarter of the edits only,
although the answer position reads the name tokens in blocks 24-25 (Phase 410).  After such an exchange three kinds
of source disagree:
  direct   the answer position reads the exchanged name tokens from block 24 on      -> the donor pointer
  backup   the positions between the name tokens and the answer position read the names before block 24 and are read
           by the answer position later                                              -> the base answer
  own      whatever the answer position itself read from the name tokens before block 24 -> the base answer
Each condition removes one kind of source with an attention edit on the answer position only, on the same donor draws:
  swap                  the exchange alone (the conflict)
  cut_backup            + the answer position cannot read the holder positions in blocks 24-39
  cut_direct            + the answer position cannot read the exchanged tokens in blocks 24-39
  late_only             + the answer position cannot read the name tokens in blocks 0-23
  late_only_cut_backup  both
  blind                 + the answer position cannot read the name tokens or the holder positions in blocks 0-23
  swap_all              the name tokens and every holder position are exchanged together (only `own` keeps the base)
Outcome: on edits whose donor pointer names another record than the base answer, the share of forced choices that
follow the donor pointer.  Stages: smoke / pilot (opened panel holdout_fid2) -> prereg -> gate -> run.
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
from source_binding_runtime import Runtime

SRC, MODEL = D.OUT, 'qwen14'
OUT = D.ROOT / 'tests/glm5/result/source_binding_arbitration_v1'
RUN, SEED = 'SOURCE-BINDING-ARBITRATION-20261112-v1', 202611120413
SPLIT, FAMILIES = 'holdout_arb', 48
CHUNK, BATCH, NB, NH, TOL, REPS, L = 192, 96, 40, 40, 1.0, 4, 24
SITES = {'qsrc': ['qsrc'], 'qent': ['qent'], 'names': ['qsrc', 'qent']}
CONDS = {'swap': [], 'cut_backup': [('holders', L, NB)], 'cut_direct': [('site', L, NB)], 'late_only': [('names', 0, L)],
         'late_only_cut_backup': [('names', 0, L), ('holders', L, NB)],
         'blind': [('names', 0, L), ('holders', 0, L)]}
CURVE = [22, 23, 25, 26]
CONTROLS = {f'm:{c}': m for c, m in CONDS.items() if m}
OTHER_PANELS = F2.OTHER_PANELS + ['source_binding_fidelity_v2/holdout_fid2']
THRESHOLDS = {   # fixed after the pilot on the opened panel, before pre-registration; all on conflict edits of the new panel
    'A0_conflict_is_graded': 'names|swap: follow between 0.20 and 0.65 (neither source wins outright). not_supported outside.',
    'A1_direct_route_causes_the_changes': 'cut_direct: follow <= 0.10 on each of qsrc, qent, names. not_supported if any > 0.20.',
    'A2_backups_read_late_hold_back': 'names: follow(cut_backup) - follow(swap) >= 0.05 with a family-bootstrap lower bound > 0. '
                                      'not_supported if the lower bound <= 0.',
    'A3_own_early_reading_holds_back': 'names: follow(late_only) - follow(swap) >= 0.15 with a lower bound > 0. not_supported if the '
                                       'difference < 0.05 or the lower bound <= 0.',
    'A4_both_removed': 'names|late_only_cut_backup: follow >= 0.65 and above both single conditions. not_supported if < 0.50.',
}
CRITERIA = {
    'model': 'Qwen3-14B original BF16, CPU byte reads, CUDA layer streaming. GLM-4-9B is not run (failed the behaviour gate of this task).',
    'material': 'layout-factorial families as in Phase 405. holdout_arb: 48 families (768 prompts). Entity names occur in no earlier '
                'panel of this line; the 16 discovery code words, 4 seen frames.',
    'exchange': 'at the input of block 24 the residual state of the tokens of a site (archive name, entity name, both) is replaced by '
                'the state of the same tokens in a donor prompt of the same family; 4 of the admissible donor relations per (item, '
                'site), seeded; the same draws in every condition of a site. swap_all: all positions from the first name token to '
                'the position before the answer position. Curve: both names exchanged at blocks 22, 23, 25, 26, same draws.',
    'attention_edits': 'only the attention row of the answer position (last token) is edited. holders = every position after the first '
                       'name token except the name tokens and the answer position. cut_backup: holders hidden in blocks 24-39. '
                       'cut_direct: the exchanged tokens hidden in blocks 24-39. late_only: both name tokens hidden in blocks 0-23. blind: name tokens and holders hidden in blocks 0-23.',
    'outcome': 'P = the record of the base prompt named by the donor pointers (position regime of the graph, Phase 409 / 412), B = the '
               'base answer. On conflict edits (P != B): follow = forced choice is P, stay = forced choice is B. Family-bootstrap 95 % '
               'intervals; paired differences between conditions on the same edits.',
    'controls': 'tail recomputation: the no-edit arm gives the same forced choice as the base forward on every item, maximal score '
                'difference <= 1.0. Every attention edit alone (no exchange) is run and its forced accuracy reported; a condition '
                'whose attention edit alone keeps < 0.90 is not interpreted.',
    'two_pass': 'gate stage: unmodified forward and the v1 behaviour gate. Edits run only if the gate passed.',
    'descriptive': 'qsrc and qent cells of A0, A2-A4; blind; swap_all; the curve over blocks 22-26; mean score of the pointer record '
                   'minus the base record and whether the two removals add up in that score; relations; halves of the base margin',
    'disclosure': 'conditions and thresholds were chosen on a pilot of 12 families of holdout_fid2 (opened in Phase 412; pilot.json). '
                  'The pilot was run twice: first with five conditions, then with blind and blind_cut_backup (name tokens hidden in '
                  'blocks 0-23, holders hidden in all blocks) added. blind_cut_backup was dropped because its attention edit alone '
                  'destroys the task (accuracy 0.27), so pilot.json holds one condition that this script no longer runs. The engine was '
                  'smoke-tested on 2 exposed families of pilot_name. The donor pointer P is the position rule of the graph, already '
                  'scored in Phases 409 and 412.',
    'not_covered': 'block 25 as an exchange layer with attention edits; which heads carry the backups into the answer position; other '
                   'tasks, answer words, frames and models. Removing a source by hiding it is an intervention on the attention '
                   'pattern, not a natural run.',
}


def stream(purpose, key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(f'{RUN}|{SEED}|{purpose}|{key}'.encode()).digest()[:16], 'little'))


def save(name, body):
    from deploy.storage_policy import check_write
    p = OUT / name
    check_write(p, 1 << 20, operation='来源绑定冲突仲裁产物')
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


def names_of(r):
    return sorted(r['up']['qsrc'] + r['up']['qent'])


def holders_of(r):
    nm = set(names_of(r))
    return [j for j in range(min(nm), len(r['tokens']) - 1) if j not in nm]


def toks(r, site):
    if site == 'all':
        return list(range(names_of(r)[0], len(r['tokens']) - 1))
    return sorted(j for g in SITES[site] for j in r['up'][g])


def pointer(r, d, site):
    return F2.predict(r, d, 'names' if site == 'all' else site, L, 'position')


def design(rows):
    index = {r['id']: i for i, r in enumerate(rows)}
    arms = {}
    for site in list(SITES) + ['all']:
        donors = []
        for r in rows:
            row = {}
            for rel in F2.RELATIONS:
                j = index[f"{r['family']}-S{r['source'] ^ rel[0]}E{r['entity'] ^ rel[1]}O{r['order'] ^ rel[2]}X{r['x'] ^ rel[3]}"]
                if len(toks(r, site)) == len(toks(rows[j], site)):
                    row[rel] = j
            assert len(row) >= REPS
            donors.append(row)
        picks = []
        for r, row in zip(rows, donors):
            rels = sorted(row)
            picks.append([rels[i] for i in stream(f'draw|{site}', r['id']).permutation(len(rels))[:REPS]])
        cells = [('swap', L)] if site == 'all' else [(c, L) for c in CONDS] + ([('swap', l) for l in CURVE] if site == 'names' else [])
        for cond, l in cells:
            for k in range(REPS):
                arms[f'{site}|{cond}@{l}#{k}'] = {'site': site, 'cond': cond, 'layer': l, 'rel': [list(p[k]) for p in picks],
                                                  'donor': [row[p[k]] for row, p in zip(donors, picks)]}
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
            plan = {}
            if arms:
                P = min(off[i] + names_of(r)[0] for i, r in enumerate(rs))

                def hidden(fn):
                    m = t.zeros((N, length), dtype=t.bool, device='cuda')
                    for i, r in enumerate(rs):
                        m[i, [off[i] + j for j in fn(r)]] = True
                    return m
                hid = {'names': hidden(names_of), 'holders': hidden(holders_of), **{s: hidden(lambda r, s=s: toks(r, s)) for s in SITES}}
                spec = lambda cond, site: [(hid[site if g == 'site' else g], b0, b1) for g, b0, b1 in CONDS[cond]]
                for name, a in arms.items():
                    bi, bp, di, dp = [], [], [], []
                    for i, r in enumerate(rs):
                        j = a['donor'][c0 + i] - c0
                        assert 0 <= j < N and rs[j]['family'] == r['family']
                        x, y = toks(r, a['site']), toks(rs[j], a['site'])
                        assert len(x) == len(y)
                        bi += [i] * len(x)
                        bp += [off[i] + v for v in x]
                        di += [j] * len(y)
                        dp += [off[j] + v for v in y]
                    ms = spec(a['cond'], a['site'])
                    plan[name] = (min([a['layer']] + [m[1] for m in ms]), a['layer'], tuple(t.tensor(v, device='cuda') for v in (bi, bp, di, dp)), ms)
                for name, c in CONTROLS.items():
                    ms = spec(name[2:], 'names')
                    plan[name] = (min(m[1] for m in ms), None, None, ms)
                plan['noop'] = (0, None, None, [])
            state, cap = {}, {}
            for b in range(NB):
                for name, (first, sw, idx, ms) in plan.items():
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
                y = t.empty_like(h)
                for lo in range(0, N, BATCH):
                    hi = min(lo + BATCH, N)
                    B = hi - lo
                    cos, sin = pe[0][lo:hi], pe[1][lo:hi]
                    y[lo:hi] = layer(h[lo:hi], attention_mask=mask[lo:hi], position_ids=pos[lo:hi], position_embeddings=(cos, sin), use_cache=False)
                    kb, vb = cap.get('k'), cap.get('v')
                    for name in state:
                        T = length - P
                        x = state[name][lo:hi]
                        tm = mask[lo:hi][:, :, P:, :]
                        act = [m for m, b0, b1 in plan[name][3] if b0 <= b < b1]
                        if act:
                            tm = tm.clone()
                            for m in act:
                                tm[:, 0, -1, :] &= ~m[lo:hi]
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
                    free, tot = t.cuda.mem_get_info()
                    log(stage, c0 * NB + (b + 1) * N, total * NB, start, block=b, arms=len(state), device_used_MiB=round((tot - free) / 2 ** 20))
            norm = rt.Norm(rt.dim, eps=rt.cfg.rms_norm_eps).to(device='cuda', dtype=t.bfloat16)
            norm.weight = t.nn.Parameter(rt.weight('model.norm.weight'), requires_grad=False)
            W = rt.weight('lm_head.weight' if 'lm_head.weight' in rt.index else 'model.embed_tokens.weight')
            logits = lambda x: t.cat([(norm(x[lo:lo + BATCH]) @ W.T).float() for lo in range(0, N, BATCH)])
            results.setdefault('base', []).extend(rt.score(logits(h[:, -1]), rs))
            for key, s in state.items():
                results.setdefault(key, []).extend(rt.score(logits(s[:, -1]), rs))
            del W, norm, h, state, ids, pos, pe, mask, plan
            gc.collect()
            t.cuda.empty_cache()
        peak = round(t.cuda.max_memory_allocated() / 2 ** 20)
    meta = {'peak_MiB': peak, 'seconds': round(time.monotonic() - start, 1)}
    if arms:
        meta['noop_same_choice'] = all(x['pred_record'] == y['pred_record'] for x, y in zip(results['noop'], results['base']))
        meta['noop_max_score_difference'] = max(abs(p - q) for x, y in zip(results['noop'], results['base']) for p, q in zip(x['scores'], y['scores']))
        meta['tail_control_ok'] = bool(meta['noop_same_choice'] and meta['noop_max_score_difference'] <= TOL)
        meta['attention_edit_alone_accuracy'] = {k[2:]: round(float(np.mean([x['forced_correct'] for x in results[k]])), 4) for k in CONTROLS}
    return results, meta


def summarise(results, arms, rows):
    fam_id = {f: i for i, f in enumerate(sorted({r['family'] for r in rows}))}
    nf = len(fam_id)
    draws = np.random.default_rng(SEED).integers(0, nf, (1000, nf))
    fam = np.array([fam_id[r['family']] for r in rows])
    margin = np.array([x['gold_margin'] for x in results['base']])
    cells = {}
    for name, a in arms.items():
        key = f"{a['site']}|{a['cond']}@{a['layer']}"
        c = cells.setdefault(key, {k: [] for k in ('fam', 'conf', 'follow', 'stay', 'changed', 'rel', 'margin', 'pb')})
        for i, (r, j, rel) in enumerate(zip(rows, a['donor'], a['rel'])):
            out, P = results[name][i]['pred_record'], pointer(r, rows[j], a['site'])
            c['fam'].append(fam[i])
            c['conf'].append(P != r['target_record'])
            c['follow'].append(out == P)
            c['stay'].append(out == r['target_record'])
            c['changed'].append(out != r['target_record'])
            c['rel'].append(''.join(map(str, rel)))
            c['margin'].append(margin[i])
            c['pb'].append(results[name][i]['scores'][P] - results[name][i]['scores'][r['target_record']])
    C = {k: {x: np.array(v) for x, v in c.items()} for k, c in cells.items()}

    def rate(v, f, keep):
        v = np.asarray(v, float)
        per, cnt = np.bincount(f[keep], weights=v[keep], minlength=nf), np.bincount(f[keep], minlength=nf)
        bs = per[draws].sum(1) / np.maximum(cnt[draws].sum(1), 1)
        return {'rate': round(float(v[keep].mean()), 4), 'ci95': [round(float(np.quantile(bs, .025)), 4), round(float(np.quantile(bs, .975)), 4)]}

    out = {'base_correct': round(float(np.mean([x['forced_correct'] for x in results['base']])), 4), 'cells': {}, 'paired': {}, 'relations': {},
           'by_base_margin': {}, 'additivity': {}}
    for key, c in C.items():
        k = c['conf']
        out['cells'][key] = {'n': len(k), 'n_conflict': int(k.sum()), 'follow': rate(c['follow'], c['fam'], k), 'stay': rate(c['stay'], c['fam'], k),
                             'score_pointer_minus_base': round(float(c['pb'][k].mean()), 3),
                             'other': round(float(1 - c['follow'][k].mean() - c['stay'][k].mean()), 4),
                             'changed_when_no_conflict': round(float(c['changed'][~k].mean()), 4) if (~k).any() else None}
    for site in SITES:
        ref = C[f'{site}|swap@{L}']
        for cond in CONDS:
            if cond != 'swap':
                c = C[f'{site}|{cond}@{L}']
                assert (c['conf'] == ref['conf']).all()
                out['paired'][f'{site}|{cond}-swap'] = rate(c['follow'].astype(float) - ref['follow'].astype(float), ref['fam'], ref['conf'])
        pb = lambda cond: C[f'{site}|{cond}@{L}']['pb']
        for a, b, ab in (('cut_backup', 'late_only', 'late_only_cut_backup'),):
            out['additivity'][f'{site}|{ab}'] = {
                'sum_of_single_effects': round(float((pb(a) + pb(b) - 2 * pb('swap'))[ref['conf']].mean()), 3),
                'joint_effect': round(float((pb(ab) - pb('swap'))[ref['conf']].mean()), 3),
                'joint_minus_sum': rate(pb(ab) - pb(a) - pb(b) + pb('swap'), ref['fam'], ref['conf'])}
        for rel in sorted(set(ref['rel'][ref['conf']])):
            k = ref['conf'] & (ref['rel'] == rel)
            out['relations'][f'{site}<{rel}'] = {'n': int(k.sum()), **{cond: round(float(C[f'{site}|{cond}@{L}']['follow'][k].mean()), 4) for cond in CONDS}}
        med = np.median(margin)
        out['by_base_margin'][site] = {h: {'n': int(k.sum()), 'follow': round(float(ref['follow'][k].mean()), 4)}
                                       for h, k in (('low_half', ref['conf'] & (ref['margin'] <= med)), ('high_half', ref['conf'] & (ref['margin'] > med)))}
    return out


def verdicts(s, meta):
    f = lambda key: s['cells'][f'{key}@{L}']['follow']['rate']
    tri = lambda ok, bad: 'supported' if ok else 'not_supported' if bad else 'inconclusive'
    v = {'controls_valid': bool(meta['tail_control_ok'] and min(meta['attention_edit_alone_accuracy'].values()) >= .90)}
    if not v['controls_valid']:
        return {**v, **{k: 'controls_invalid' for k in THRESHOLDS}}
    d = lambda cond: s['paired'][f'names|{cond}-swap']
    cd = [f(f'{site}|cut_direct') for site in SITES]
    v['A0_conflict_is_graded'] = tri(.20 <= f('names|swap') <= .65, True)
    v['A1_direct_route_causes_the_changes'] = tri(max(cd) <= .10, max(cd) > .20)
    v['A2_backups_read_late_hold_back'] = tri(d('cut_backup')['rate'] >= .05 and d('cut_backup')['ci95'][0] > 0, d('cut_backup')['ci95'][0] <= 0)
    v['A3_own_early_reading_holds_back'] = tri(d('late_only')['rate'] >= .15 and d('late_only')['ci95'][0] > 0,
                                               d('late_only')['rate'] < .05 or d('late_only')['ci95'][0] <= 0)
    j = f('names|late_only_cut_backup')
    v['A4_both_removed'] = tri(j >= .65 and j > max(f('names|cut_backup'), f('names|late_only')), j < .50)
    return v


def explore(stage, rows):
    arms = design(rows)
    results, meta = engine(rows, arms, stage=stage)
    return summarise(results, arms, rows), meta, len(arms)


def smoke():
    from deploy.storage_policy import checked_dir
    checked_dir(OUT, 1 << 30, operation='来源绑定冲突仲裁产物')
    rows = [r for r in D.read(NL.OUT / 'pilot_name-materials.json') if int(r['family'][-4:]) < 2]
    s, meta, n = explore('smoke', rows)
    print(json.dumps({'meta': meta, 'arms': n, 'cells': len(s['cells'])}), flush=True)


def pilot():
    """Opened panel holdout_fid2 (Phase 412), first 12 families: every number may be looked at."""
    from deploy.storage_policy import checked_dir
    checked_dir(OUT, 1 << 30, operation='来源绑定冲突仲裁产物')
    rows = [r for r in D.read(F2.OUT / f'{F2.SPLIT}-materials.json') if int(r['family'][-4:]) < 12]
    assert len(rows) == 192
    s, meta, n = explore('pilot', rows)
    save('pilot.json', {'run_id': RUN, 'panel': 'holdout_fid2 families 0-11 (opened in Phase 412)', 'code_sha256': D.sha(__file__), 'arms': n,
                        'summary': s, **meta})
    print(json.dumps({'meta': meta, 'cells': {k: [v['n_conflict'], v['follow']['rate'], v['stay']['rate'], v['changed_when_no_conflict']]
                                              for k, v in s['cells'].items()}, 'paired': s['paired'], 'margin': s['by_base_margin'],
                      'pb': {k: v['score_pointer_minus_base'] for k, v in s['cells'].items()}, 'additivity': s['additivity']}), flush=True)


DEPS = (RL, PT, NL, F2)


def prereg():
    from tools import provenance
    from tools.power_analysis import check_phase
    assert THRESHOLDS and not (OUT / 'prereg.json').exists()
    plan = D.checked(MODEL)
    rows = build_materials()
    assert len(rows) == FAMILIES * 16 and len({tuple(r['tokens']) for r in rows}) == len(rows)
    arms = design(rows)
    save(f'{SPLIT}-materials.json', rows)
    save('design.json', arms)
    conf = {k: int(sum(pointer(r, rows[j], a['site']) != r['target_record'] for r, j in zip(rows, a['donor']))) for k, a in arms.items()}
    n_conf = {s: sum(v for k, v in conf.items() if k.startswith(f'{s}|swap@{L}#')) for s in list(SITES) + ['all']}
    body = {'schema': 'source_binding_arbitration_prereg.v1', 'run_id': RUN, 'seed': SEED, 'created_at': datetime.now().astimezone().isoformat(),
            'base_plan_sha256': D.sha(SRC / 'plan.json'), 'model_snapshot': plan['models'][MODEL], 'code_sha256': D.sha(__file__),
            'dependency_sha256': {m.__name__: D.sha(m.__file__) for m in DEPS},
            'criteria': CRITERIA, 'thresholds': THRESHOLDS, 'pilot_sha256': D.sha(OUT / 'pilot.json'),
            'materials': {'sha256': D.sha(OUT / f'{SPLIT}-materials.json'), 'families': FAMILIES, 'rows': len(rows)},
            'design': {'sha256': D.sha(OUT / 'design.json'), 'arms': len(arms), 'conflict_edits_per_site': n_conf},
            'power': {'unit': 'paired binary outcome per conflict edit (follow under the condition / under swap); alpha = 0.05',
                      'paired_difference': {'expected_delta': 0.20, 'p_d': 0.40, **check_phase(0.20, n_conf['names'], p_d=0.40)},
                      'A2_smallest_effect_item_level': {'expected_delta': 0.05, 'p_d': 0.15, **check_phase(0.05, len(rows), p_d=0.15)},
                      'as_if_one_edit_per_item': {'expected_delta': 0.20, 'p_d': 0.40, **check_phase(0.20, len(rows), p_d=0.40)},
                      'caveat': 'edits are clustered in 768 items and 48 families; intervals are family bootstraps'},
            'custody': 'holdout_arb is read first by the gate stage (behaviour only) and then once by the run stage.',
            'resources': {'gpu_minutes_estimate': 20, 'disk_GiB_estimate': 0.3, 'models_concurrent': 1,
                          'shape': 'chunks of 192 prompts, inner batch 96; fixed before pre-registration'},
            'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    body['provenance']['model_revision'] = plan['models'][MODEL]
    save('prereg.json', body)
    print(json.dumps({'prereg_sha256': D.sha(OUT / 'prereg.json'), 'design': body['design'], 'power': body['power']['paired_difference'],
                      'power_item_level': body['power']['as_if_one_edit_per_item']}), flush=True)


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
    save('gate.json', {'schema': 'source_binding_arbitration_gate.v1', 'run_id': RUN, 'items_sha256': D.sha(OUT / 'gate-items.json'), **g, **meta})
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
        print(json.dumps({'meta': meta, 'verdicts': v, 'cells': {k: [c['n_conflict'], c['follow'], c['stay']['rate']] for k, c in s['cells'].items()},
                          'paired': s['paired']}), flush=True)
    else:
        s, meta, v = {'note': 'behaviour gate failed; no edit was run'}, {}, {k: 'behavior_ineligible' for k in THRESHOLDS}
    report = {'schema': 'source_binding_arbitration_report.v1', 'run_id': RUN, 'split': SPLIT, 'gate_passed': g['passed'],
              'prereg_sha256': D.sha(OUT / 'prereg.json'), 'gate_sha256': D.sha(OUT / 'gate.json'),
              'items_sha256': D.sha(OUT / 'items.json') if g['passed'] else None, 'criteria': pre['criteria'], 'thresholds': pre['thresholds'],
              'verdicts': v, 'summary': s, **meta, 'formal_evidence_promotion': 0,
              'custody': 'holdout_arb read once by the gate stage and once by this run',
              'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    save('report.json', report)
    path = provenance.write_result('SOURCE-BINDING-arbitration-holdout', {'verdicts': v, 'gate_passed': g['passed'],
                                                                        'report_sha256': D.sha(OUT / 'report.json')},
                                   model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)
    print(json.dumps({'verdicts': v, 'result': str(path)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['smoke', 'pilot', 'prereg', 'gate', 'run'])
    {'smoke': smoke, 'pilot': pilot, 'prereg': prereg, 'gate': gate, 'run': run}[ap.parse_args().stage]()
