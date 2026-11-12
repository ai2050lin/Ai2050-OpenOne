"""Which heads carry the backups into the answer position (Claude Phase 414).  Qwen3-14B BF16.

Phase 411: with the four code words hidden from the answer position the answer is still right; the positions between
the question names and the answer position (holders) carry a ready answer.  Phase 413: the same holders pull the answer
back when a name token is exchanged at block 24.  Here the route holders -> answer position is located.

Regime V: the answer position cannot read the code words from block 26 on (the direct fetch is off).  Only the
attention row of the answer position is edited.
  floor     V + holders hidden in blocks 26-39
  window    V + holders visible only in blocks [L0, M)                      (earliest sufficient window)
  only S    window + in the window only the heads of S read the holders     (sufficient head set)
  block N   window + the heads of N cannot read the holders                 (necessary inside the window)
  V-nat N   V + the heads of N cannot read the holders, everything else free
  conflict  both names exchanged at block 24 (Phase 413) + the heads of N cannot read the holders
Stages: smoke, pilot_a (windows), pilot_b (head ranking), pilot_c (sets), freeze, pilot_d (all holdout arms on the
opened rows) -> prereg -> gate -> run.
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
from source_binding_runtime import Runtime

SRC, MODEL = D.OUT, 'qwen14'
OUT = D.ROOT / 'tests/glm5/result/source_binding_backup_v1'
RUN, SEED = 'SOURCE-BINDING-BACKUP-20261112-v1', 202611120414
SPLIT, FAMILIES = 'holdout_bak', 48
CHUNK, BATCH, NB, NH, TOL, REPS, L, START = 192, 96, 40, 40, 1.0, 4, 24, 26
KS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
V = [['vals', START, NB, None]]
OTHER_PANELS = AR.OTHER_PANELS + ['source_binding_arbitration_v1/holdout_arb']
THRESHOLDS = {   # fixed after pilot_d on the opened rows, before pre-registration
    'B0_backup_route_exists': 'V >= 0.95 and floor <= 0.60. not_supported if V < 0.90 or floor > 0.70.',
    'B0w_window_suffices': 'window (holders visible only in blocks 30-36) >= 0.90. not_supported if < 0.80.',
    'B1_S_suffices': 'only S >= 0.80 and every random set <= floor + 0.15. not_supported if only S < 0.65 or a random set comes '
                     'within 0.10 of it.',
    'B2_N_necessary_in_window': 'block N <= 0.70 and every random set >= 0.90. not_supported if block N > 0.80 or a random set is '
                                'within 0.10 of it.',
    'B3_N_is_compensated_outside_the_window': 'V-nat N >= 0.90 (the holders can still be read by other heads and in blocks 37-39). '
                                              'not_supported if < 0.80.',
    'B4_pull_back_is_read_in_blocks_24_25': 'conflict: follow(holders hidden in 24-25) - follow(swap) >= 0.05 with a lower bound > 0, '
                                            'and |follow(holders hidden from 26 on) - follow(swap)| <= 0.03. not_supported if the '
                                            'lower bound <= 0 or the late difference exceeds 0.06 in size.',
    'B5_C_carries_the_pull_back': 'conflict: follow(C loses the holders) - follow(swap) >= 0.05 with a lower bound > 0, and above the '
                                  'random set with a lower bound > 0. not_supported if either lower bound <= 0.',
    'B6_answer_backup_heads_do_not': 'conflict: |follow(N loses the holders) - follow(swap)| <= 0.03. not_supported if > 0.06.',
}
CRITERIA = {
    'model': 'Qwen3-14B original BF16, CPU byte reads, CUDA layer streaming. GLM-4-9B is not run (failed the behaviour gate of this task).',
    'material': 'layout-factorial families as in Phase 405. holdout_bak: 48 families (768 prompts). Entity names occur in no earlier '
                'panel of this line; the 16 discovery code words, 4 seen frames.',
    'edits': 'only the attention row of the answer position (last token) is edited, per block and per head. vals = the tokens of the '
             'four code words; holders = every position after the first question name except the name tokens and the answer position.',
    'regime_V': 'vals hidden from the answer position in blocks 26-39 (Phase 411: the direct fetch happens in block 28).',
    'selection': 'on 16 opened families of holdout_arb. Window: M = the earliest end with accuracy >= 0.95 when holders are visible only '
                 'in [26, M); L0 = the latest start with accuracy >= 0.95 for [L0, M). Head ranking: in the window regime one head at '
                 'a time loses the holders; heads ordered by the drop of the mean gold margin. S = the smallest top-K whose members '
                 'alone (all other heads of the window blocks lose the holders) reach accuracy >= 0.85; N = the smallest top-K whose '
                 'removal inside the window regime gives accuracy <= 0.65. K in 1, 2, 3, 4, 6, 8, 12, 16, 24, 32.',
    'selection_conflict': 'on the same opened rows, both names exchanged at block 24. Holders hidden in block 24 only, block 25 only, '
                          'both. Head ranking (donor draw 0): one head of blocks 24-25 at a time loses the holders; heads ordered by '
                          'the gain of the mean score of the pointer record minus the base record. C = the smallest top-K whose '
                          'removal gives >= 70 % of the gain in follow obtained by hiding the holders from all heads of blocks 24-25.',
    'disclosure': 'the first version of the final arm set tested whether N carries the pull-back of Phase 413; on the opened rows it '
                  'does not (-0.02) and hiding the holders from block 26 on has no effect either, so the conflict selection '
                  '(pilot_e, pilot_f) and verdicts B4-B6 were added and B3 was restated as compensation after pilot_d showed V-nat N '
                  'at 0.99. All of this happened on opened rows before pre-registration. The engine was smoke-tested on 2 exposed '
                  'families of pilot_name.',
    'controls': 'tail recomputation: the no-edit arm gives the same forced choice as the base forward, maximal score difference <= 1.0. '
                '3 random head sets with the same number of heads per block, from the same blocks, outside S and N (SHA-256 seeded); '
                'for C one such set outside C.',
    'conflict': 'as in Phase 413: both names exchanged at the input of block 24 with a donor of the same family, 4 seeded donor '
                'relations per item; follow = the forced choice is the record named by the donor pointers, on edits where that record '
                'is not the base answer; paired differences on the same edits, family-bootstrap 95 % intervals.',
    'two_pass': 'gate stage: unmodified forward and the v1 behaviour gate. Edits run only if the gate passed.',
    'not_covered': 'what the holders hold and how they got it (Phases 404-409 cover R); the OV content of the heads; holders read '
                   'before block 26; other tasks, answer words, frames and models. Hiding is an intervention on the attention '
                   'pattern, not a natural run.',
}


def stream(purpose, key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(f'{RUN}|{SEED}|{purpose}|{key}'.encode()).digest()[:16], 'little'))


def save(name, body):
    from deploy.storage_policy import check_write
    p = OUT / name
    check_write(p, 1 << 20, operation='来源绑定备份头产物')
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


GROUP = {'vals': lambda r: FE.keys(r)[0]['vals'], 'holders': AR.holders_of, 'names': AR.names_of}


# ---------------------------------------------------------------- arm builders (hide = [group, b0, b1, hidden heads | None])
def outside(l0, m):
    return [['holders', START, l0, None], ['holders', m, NB, None]]


def per_block(heads):
    d = {}
    for b, h in heads:
        d.setdefault(b, []).append(h)
    return d


def only(heads, l0, m):
    d = per_block(heads)
    return [['holders', b, b + 1, [h for h in range(NH) if h not in d.get(b, [])]] for b in range(l0, m)]


def block(heads):
    return [['holders', b, b + 1, hs] for b, hs in per_block(heads).items()]


def randoms(heads, avoid, tag):
    out = []
    for k in range(3):
        rnd, pick = stream(f'random|{tag}', k), []
        for b, hs in sorted(per_block(heads).items()):
            pool = [h for h in range(NH) if [b, h] not in avoid and (b, h) not in avoid]
            pick += [[b, int(h)] for h in rnd.choice(pool, len(hs), replace=False)]
        out.append(pick)
    return out


def draws(rows):
    """Donor draws of the conflict arms (site names), as in Phase 413."""
    index = {r['id']: i for i, r in enumerate(rows)}
    donors, picks = [], []
    for r in rows:
        row = {}
        for rel in F2.RELATIONS:
            j = index[f"{r['family']}-S{r['source'] ^ rel[0]}E{r['entity'] ^ rel[1]}O{r['order'] ^ rel[2]}X{r['x'] ^ rel[3]}"]
            if len(AR.toks(r, 'names')) == len(AR.toks(rows[j], 'names')):
                row[rel] = j
        assert len(row) >= REPS
        rels = sorted(row)
        donors.append(row)
        picks.append([rels[i] for i in stream('draw|names', r['id']).permutation(len(rels))[:REPS]])
    return [[row[p[k]] for row, p in zip(donors, picks)] for k in range(REPS)]


def final_arms(rows, fz):
    l0, m, S, N = fz['L0'], fz['M'], fz['S'], fz['N']
    w = outside(l0, m)
    avoid = S + N
    arms = {'V': {'hide': V}, 'floor': {'hide': V + [['holders', START, NB, None]]}, 'window': {'hide': V + w},
            'only:S': {'hide': V + w + only(S, l0, m)}, 'block:N': {'hide': V + w + block(N)}, 'Vnat:N': {'hide': V + block(N)},
            'nat:N': {'hide': block(N)}, 'nat:holders': {'hide': [['holders', START, NB, None]]}}
    for k, c in enumerate(randoms(S, avoid, 'S')):
        arms[f'only:randS{k}'] = {'hide': V + w + only(c, l0, m)}
    for k, c in enumerate(randoms(N, avoid, 'N')):
        arms[f'block:randN{k}'] = {'hide': V + w + block(c)}
        arms[f'Vnat:randN{k}'] = {'hide': V + block(c)}
    C = fz['C']
    rc = randoms(C, avoid + C, 'C')[0]
    arms['nat:C'] = {'hide': block(C)}
    for k, don in enumerate(draws(rows)):
        for name, hide in (('swap', []), ('swap+C', block(C)), ('swap+randC', block(rc)), ('swap+N', block(N)),
                           ('swap+holders24-25', [['holders', L, START, None]]), ('swap+holders26', [['holders', START, NB, None]]),
                           ('swap+holders', [['holders', L, NB, None]])):
            arms[f'{name}#{k}'] = {'hide': hide, 'donor': don}
    return arms


# ---------------------------------------------------------------- engine
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
                hid = {}
                for g, fn in GROUP.items():
                    hid[g] = t.zeros((N, length), dtype=t.bool, device='cuda')
                    for i, r in enumerate(rs):
                        hid[g][i, [off[i] + j for j in fn(r)]] = True
                P0 = min(off[i] + AR.names_of(r)[0] for i, r in enumerate(rs))
                for name, a in {**arms, 'noop': {'hide': []}}.items():
                    ms = [(hid[g], b0, b1, hs) for g, b0, b1, hs in a['hide'] if b0 < b1]
                    idx, sw, P = None, None, length - 1
                    if 'donor' in a:
                        bi, bp, di, dp = [], [], [], []
                        for i, r in enumerate(rs):
                            j = a['donor'][c0 + i] - c0
                            assert 0 <= j < N and rs[j]['family'] == r['family']
                            x, y = AR.toks(r, 'names'), AR.toks(rs[j], 'names')
                            assert len(x) == len(y)
                            bi += [i] * len(x)
                            bp += [off[i] + v for v in x]
                            di += [j] * len(y)
                            dp += [off[j] + v for v in y]
                        idx, sw, P = tuple(t.tensor(v, device='cuda') for v in (bi, bp, di, dp)), L, P0
                    plan[name] = (min([NB if sw is None else sw] + [m[1] for m in ms]) if name != 'noop' else 0, P, sw, idx, ms)
            state, cap = {}, {}
            for b in range(NB):
                for name, (first, P, sw, idx, ms) in plan.items():
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
                    kfull, vfull = {}, {}
                    for name in state:
                        P = plan[name][1]
                        T = length - P
                        x = state[name][lo:hi]
                        tm = mask[lo:hi][:, :, P:, :]
                        act = [(m, hs) for m, b0, b1, hs in plan[name][4] if b0 <= b < b1]
                        if act:
                            tm = tm.expand(B, NH if any(hs is not None for _, hs in act) else 1, T, length).clone()
                            for m, hs in act:
                                if hs is None:
                                    tm[:, :, -1, :] &= ~m[lo:hi, None]
                                else:
                                    tm[:, hs, -1, :] &= ~m[lo:hi, None]
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
            for key in plan:
                results.setdefault(key, []).extend(rt.score(logits(state[key][:, -1]) if key in state else logits(h[:, -1]), rs))
            del W, norm, h, state, ids, pos, pe, mask, plan
            gc.collect()
            t.cuda.empty_cache()
        peak = round(t.cuda.max_memory_allocated() / 2 ** 20)
    meta = {'peak_MiB': peak, 'seconds': round(time.monotonic() - start, 1)}
    if arms:
        meta['noop_same_choice'] = all(x['pred_record'] == y['pred_record'] for x, y in zip(results['noop'], results['base']))
        meta['noop_max_score_difference'] = max(abs(p - q) for x, y in zip(results['noop'], results['base']) for p, q in zip(x['scores'], y['scores']))
        meta['tail_control_ok'] = bool(meta['noop_same_choice'] and meta['noop_max_score_difference'] <= TOL)
    return results, meta


def acc(results, name):
    return round(float(np.mean([x['forced_correct'] for x in results[name]])), 4)


def margin(results, name):
    return round(float(np.mean([x['gold_margin'] for x in results[name]])), 3)


def summarise(results, arms, rows):
    fam_id = {f: i for i, f in enumerate(sorted({r['family'] for r in rows}))}
    nf = len(fam_id)
    bs = np.random.default_rng(SEED).integers(0, nf, (1000, nf))
    fam = np.array([fam_id[r['family']] for r in rows])

    def rate(v, f, keep=None):
        v = np.asarray(v, float)
        keep = np.ones(len(v), bool) if keep is None else keep
        per, cnt = np.bincount(f[keep], weights=v[keep], minlength=nf), np.bincount(f[keep], minlength=nf)
        b = per[bs].sum(1) / np.maximum(cnt[bs].sum(1), 1)
        return {'rate': round(float(v[keep].mean()), 4), 'ci95': [round(float(np.quantile(b, .025)), 4), round(float(np.quantile(b, .975)), 4)]}

    out = {'base_correct': acc(results, 'base'), 'accuracy': {}, 'conflict': {}, 'paired': {}}
    conf = {}
    for name, a in arms.items():
        if 'donor' not in a:
            out['accuracy'][name] = {**rate([x['forced_correct'] for x in results[name]], fam), 'gold_margin': margin(results, name)}
            continue
        c = conf.setdefault(name.split('#')[0], {'fam': [], 'conf': [], 'follow': [], 'stay': []})
        for i, (r, j) in enumerate(zip(rows, a['donor'])):
            o, P = results[name][i]['pred_record'], AR.pointer(r, rows[j], 'names')
            c['fam'].append(fam[i])
            c['conf'].append(P != r['target_record'])
            c['follow'].append(o == P)
            c['stay'].append(o == r['target_record'])
    C = {k: {x: np.array(v) for x, v in c.items()} for k, c in conf.items()}
    for k, c in C.items():
        out['conflict'][k] = {'n_conflict': int(c['conf'].sum()), 'follow': rate(c['follow'], c['fam'], c['conf']),
                              'stay': round(float(c['stay'][c['conf']].mean()), 4)}
        if k != 'swap':
            ref = C['swap']
            assert (ref['conf'] == c['conf']).all()
            out['paired'][f'{k}-swap'] = rate(c['follow'].astype(float) - ref['follow'].astype(float), ref['fam'], ref['conf'])
    if 'swap+C' in C:
        out['paired']['swap+C-swap+randC'] = rate(C['swap+C']['follow'].astype(float) - C['swap+randC']['follow'].astype(float),
                                                  C['swap']['fam'], C['swap']['conf'])
    return out


def verdicts(s, meta):
    a = lambda k: s['accuracy'][k]['rate']
    tri = lambda ok, bad: 'supported' if ok else 'not_supported' if bad else 'inconclusive'
    v = {'controls_valid': bool(meta['tail_control_ok'])}
    if not v['controls_valid']:
        return {**v, **{k: 'controls_invalid' for k in THRESHOLDS}}
    rs, rn, rv = [a(f'only:randS{k}') for k in range(3)], [a(f'block:randN{k}') for k in range(3)], [a(f'Vnat:randN{k}') for k in range(3)]
    v['B0_backup_route_exists'] = tri(a('V') >= .95 and a('floor') <= .60, a('V') < .90 or a('floor') > .70)
    v['B0w_window_suffices'] = tri(a('window') >= .90, a('window') < .80)
    v['B1_S_suffices'] = tri(a('only:S') >= .80 and max(rs) <= a('floor') + .15, a('only:S') < .65 or max(rs) > a('only:S') - .10)
    v['B2_N_necessary_in_window'] = tri(a('block:N') <= .70 and min(rn) >= .90, a('block:N') > .80 or min(rn) < a('block:N') + .10)
    v['B3_N_is_compensated_outside_the_window'] = tri(a('Vnat:N') >= .90, a('Vnat:N') < .80)
    P = lambda k: s['paired'][k]
    early, late, c, cr, n = P('swap+holders24-25-swap'), P('swap+holders26-swap'), P('swap+C-swap'), P('swap+C-swap+randC'), P('swap+N-swap')
    v['B4_pull_back_is_read_in_blocks_24_25'] = tri(early['rate'] >= .05 and early['ci95'][0] > 0 and abs(late['rate']) <= .03,
                                                   early['ci95'][0] <= 0 or abs(late['rate']) > .06)
    v['B5_C_carries_the_pull_back'] = tri(c['rate'] >= .05 and c['ci95'][0] > 0 and cr['ci95'][0] > 0, c['ci95'][0] <= 0 or cr['ci95'][0] <= 0)
    v['B6_answer_backup_heads_do_not'] = tri(abs(n['rate']) <= .03, abs(n['rate']) > .06)
    return v


# ---------------------------------------------------------------- pilot (opened rows of holdout_arb)
def pilot_rows(n=16):
    from deploy.storage_policy import checked_dir
    checked_dir(OUT, 1 << 30, operation='来源绑定备份头产物')
    return [r for r in D.read(AR.OUT / f'{AR.SPLIT}-materials.json') if int(r['family'][-4:]) < n]


def smoke():
    pilot_rows(0)
    rows = [r for r in D.read(NL.OUT / 'pilot_name-materials.json') if int(r['family'][-4:]) < 2]
    fz = {'L0': 28, 'M': 30, 'S': [[28, 1], [29, 2]], 'N': [[28, 1], [28, 3], [29, 2]], 'C': [[24, 5], [25, 6]]}
    arms = final_arms(rows, fz)
    results, meta = engine(rows, arms, stage='smoke')
    s = summarise(results, arms, rows)
    print(json.dumps({'meta': meta, 'arms': len(arms), 'cells': len(s['accuracy']) + len(s['conflict'])}), flush=True)


def pilot_a():
    rows = pilot_rows()
    arms = {'V': {'hide': V}, 'floor': {'hide': V + [['holders', START, NB, None]]}}
    for s in range(START, NB):
        for e in range(s + 1, NB + 1):
            arms[f'w{s}-{e}'] = {'hide': V + outside(s, e)}
    results, meta = engine(rows, arms, stage='pilot_a')
    A = {k: acc(results, k) for k in arms}
    M = next(e for e in range(START + 1, NB + 1) if A[f'w{START}-{e}'] >= .95)
    L0 = max(s for s in range(START, M) if A[f'w{s}-{M}'] >= .95)
    save('pilot-a.json', {'code_sha256': D.sha(__file__), 'rows': len(rows), 'accuracy': A, 'M': M, 'L0': L0, **meta})
    print(json.dumps({'meta': meta, 'V': A['V'], 'floor': A['floor'], 'M': M, 'L0': L0, 'ends': {e: A[f'w{START}-{e}'] for e in range(START + 1, NB + 1)},
                      'starts': {s: A[f'w{s}-{M}'] for s in range(START, M)}, 'single': {s: A[f'w{s}-{s + 1}'] for s in range(START, NB)}}), flush=True)


def pilot_b():
    rows, pa = pilot_rows(), D.read(OUT / 'pilot-a.json')
    l0, m = pa['L0'], pa['M']
    w = outside(l0, m)
    arms = {'window': {'hide': V + w}}
    for b in range(l0, m):
        for hh in range(NH):
            arms[f'{b}.{hh}'] = {'hide': V + w + block([[b, hh]])}
    results, meta = engine(rows, arms, stage='pilot_b')
    ref = margin(results, 'window')
    drop = {k: round(ref - margin(results, k), 3) for k in arms if k != 'window'}
    order = [[int(x) for x in k.split('.')] for k in sorted(drop, key=lambda k: -drop[k])]
    save('pilot-b.json', {'code_sha256': D.sha(__file__), 'L0': l0, 'M': m, 'window_margin': ref, 'margin_drop': drop, 'order': order, **meta})
    print(json.dumps({'meta': meta, 'window': [acc(results, 'window'), ref], 'top': [[k, drop[f'{k[0]}.{k[1]}'], acc(results, f'{k[0]}.{k[1]}')] for k in order[:16]]}), flush=True)


def pilot_c():
    rows, pb = pilot_rows(), D.read(OUT / 'pilot-b.json')
    l0, m, order = pb['L0'], pb['M'], pb['order']
    w = outside(l0, m)
    arms = {}
    for k in KS:
        arms[f'only{k}'] = {'hide': V + w + only(order[:k], l0, m)}
        arms[f'block{k}'] = {'hide': V + w + block(order[:k])}
        arms[f'Vnat{k}'] = {'hide': V + block(order[:k])}
    results, meta = engine(rows, arms, stage='pilot_c')
    A = {k: acc(results, k) for k in arms}
    ks = next((k for k in KS if A[f'only{k}'] >= .85), None)
    kn = next((k for k in KS if A[f'block{k}'] <= .65), None)
    save('pilot-c.json', {'code_sha256': D.sha(__file__), 'accuracy': A, 'K_S': ks, 'K_N': kn, **meta})
    print(json.dumps({'meta': meta, 'K_S': ks, 'K_N': kn, 'sets': {k: [A[f'only{k}'], A[f'block{k}'], A[f'Vnat{k}']] for k in KS}}), flush=True)


def conflict(results, name, rows, donor):
    P = [AR.pointer(r, rows[j], 'names') for r, j in zip(rows, donor)]
    keep = [i for i, r in enumerate(rows) if P[i] != r['target_record']]
    return (float(np.mean([results[name][i]['pred_record'] == P[i] for i in keep])),
            float(np.mean([results[name][i]['scores'][P[i]] - results[name][i]['scores'][rows[i]['target_record']] for i in keep])))


def pilot_e():
    """Conflict regime: where the holders are read (blocks 24-25) and which heads read them (ranking on donor draw 0)."""
    rows = pilot_rows()
    dons = draws(rows)
    arms = {}
    for k, don in enumerate(dons):
        for name, hide in (('swap', []), ('w24-26', [['holders', 24, 26, None]]), ('w24-25', [['holders', 24, 25, None]]),
                           ('w25-26', [['holders', 25, 26, None]])):
            arms[f'{name}#{k}'] = {'hide': hide, 'donor': don}
    for b in (24, 25):
        for hh in range(NH):
            arms[f'{b}.{hh}'] = {'hide': block([[b, hh]]), 'donor': dons[0]}
    results, meta = engine(rows, arms, stage='pilot_e')
    ref = conflict(results, 'swap#0', rows, dons[0])[1]
    gain = {k: round(conflict(results, k, rows, dons[0])[1] - ref, 3) for k in arms if '#' not in k}
    order = [[int(x) for x in k.split('.')] for k in sorted(gain, key=lambda k: -gain[k])]
    win = {n: round(float(np.mean([conflict(results, f'{n}#{k}', rows, dons[k])[0] for k in range(REPS)])), 4) for n in ('swap', 'w24-26', 'w24-25', 'w25-26')}
    save('pilot-e.json', {'code_sha256': D.sha(__file__), 'follow': win, 'score_gain': gain, 'order': order, **meta})
    print(json.dumps({'meta': meta, 'follow': win, 'top': [[k, gain[f'{k[0]}.{k[1]}']] for k in order[:16]], 'sum_all': round(sum(gain.values()), 2)}), flush=True)


def pilot_f():
    rows, order = pilot_rows(), D.read(OUT / 'pilot-e.json')['order']
    dons = draws(rows)
    arms = {}
    for k, don in enumerate(dons):
        arms[f'swap#{k}'] = {'hide': [], 'donor': don}
        arms[f'full#{k}'] = {'hide': [['holders', 24, 26, None]], 'donor': don}
        for K in KS:
            arms[f'K{K}#{k}'] = {'hide': block(order[:K]), 'donor': don}
    results, meta = engine(rows, arms, stage='pilot_f')
    f = {n: round(float(np.mean([conflict(results, f'{n}#{k}', rows, dons[k])[0] for k in range(REPS)])), 4) for n in ['swap', 'full'] + [f'K{K}' for K in KS]}
    kc = next((K for K in KS if f[f'K{K}'] - f['swap'] >= .7 * (f['full'] - f['swap'])), None)
    save('pilot-f.json', {'code_sha256': D.sha(__file__), 'follow': f, 'K_C': kc, 'rule': 'smallest K whose removal gives >= 70 % of the '
                          'gain in follow obtained by hiding the holders from all heads in blocks 24-25', **meta})
    print(json.dumps({'meta': meta, 'K_C': kc, 'follow': f}), flush=True)


def freeze():
    assert not (OUT / 'frozen.json').exists()
    pb, pc, pe, pf = (D.read(OUT / f'pilot-{k}.json') for k in 'bcef')
    assert pc['K_S'] and pc['K_N'] and pf['K_C']
    save('frozen.json', {'L0': pb['L0'], 'M': pb['M'], 'S': pb['order'][:pc['K_S']], 'N': pb['order'][:pc['K_N']], 'C': pe['order'][:pf['K_C']],
                         'pilot_sha256': {k: D.sha(OUT / f'pilot-{k}.json') for k in 'abcef'}})
    print(json.dumps(D.read(OUT / 'frozen.json')), flush=True)


def pilot_d():
    rows, fz = pilot_rows(), D.read(OUT / 'frozen.json')
    arms = final_arms(rows, fz)
    results, meta = engine(rows, arms, stage='pilot_d')
    s = summarise(results, arms, rows)
    save('pilot-d.json', {'code_sha256': D.sha(__file__), 'frozen_sha256': D.sha(OUT / 'frozen.json'), 'summary': s, **meta})
    print(json.dumps({'meta': meta, 'accuracy': {k: [v['rate'], v['gold_margin']] for k, v in s['accuracy'].items()},
                      'conflict': {k: [v['follow']['rate'], v['stay']] for k, v in s['conflict'].items()}, 'paired': s['paired']}), flush=True)


DEPS = (RL, PT, NL, F2, FE, AR)


def prereg():
    from tools import provenance
    from tools.power_analysis import check_phase
    assert THRESHOLDS and not (OUT / 'prereg.json').exists()
    plan = D.checked(MODEL)
    rows = build_materials()
    assert len(rows) == FAMILIES * 16 and len({tuple(r['tokens']) for r in rows}) == len(rows)
    fz = D.read(OUT / 'frozen.json')
    arms = final_arms(rows, fz)
    save(f'{SPLIT}-materials.json', rows)
    save('design.json', arms)
    n_conf = int(sum(AR.pointer(r, rows[j], 'names') != r['target_record'] for k in range(REPS) for r, j in zip(rows, arms[f'swap#{k}']['donor'])))
    body = {'schema': 'source_binding_backup_prereg.v1', 'run_id': RUN, 'seed': SEED, 'created_at': datetime.now().astimezone().isoformat(),
            'base_plan_sha256': D.sha(SRC / 'plan.json'), 'model_snapshot': plan['models'][MODEL], 'code_sha256': D.sha(__file__),
            'dependency_sha256': {m.__name__: D.sha(m.__file__) for m in DEPS},
            'criteria': CRITERIA, 'thresholds': THRESHOLDS, 'frozen': fz, 'frozen_sha256': D.sha(OUT / 'frozen.json'),
            'pilot_d_sha256': D.sha(OUT / 'pilot-d.json'),
            'materials': {'sha256': D.sha(OUT / f'{SPLIT}-materials.json'), 'families': FAMILIES, 'rows': len(rows)},
            'design': {'sha256': D.sha(OUT / 'design.json'), 'arms': len(arms), 'conflict_edits': n_conf},
            'power': {'unit': 'paired binary outcome per item (arm right / comparison arm right); alpha = 0.05',
                      'accuracy_contrasts': {'expected_delta': 0.25, 'p_d': 0.40, **check_phase(0.25, len(rows), p_d=0.40)},
                      'B4_item_level': {'expected_delta': 0.05, 'p_d': 0.15, **check_phase(0.05, len(rows), p_d=0.15)},
                      'caveat': 'items are clustered in 48 families; intervals are family bootstraps'},
            'custody': 'holdout_bak is read first by the gate stage (behaviour only) and then once by the run stage.',
            'resources': {'gpu_minutes_estimate': 10, 'disk_GiB_estimate': 0.1, 'models_concurrent': 1,
                          'shape': 'chunks of 192 prompts, inner batch 96; fixed before pre-registration'},
            'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    body['provenance']['model_revision'] = plan['models'][MODEL]
    save('prereg.json', body)
    print(json.dumps({'prereg_sha256': D.sha(OUT / 'prereg.json'), 'design': body['design'], 'power': body['power']}), flush=True)


def checked():
    p = D.read(OUT / 'prereg.json')
    assert p['code_sha256'] == D.sha(__file__) and all(p['dependency_sha256'][m.__name__] == D.sha(m.__file__) for m in DEPS), \
        'source changed after pre-registration'
    assert p['materials']['sha256'] == D.sha(OUT / f'{SPLIT}-materials.json') and p['design']['sha256'] == D.sha(OUT / 'design.json')
    assert p['frozen_sha256'] == D.sha(OUT / 'frozen.json')
    D.checked(MODEL)
    return p


def gate():
    checked()
    assert not (OUT / 'gate.json').exists(), 'holdout behaviour already read'
    rows = D.read(OUT / f'{SPLIT}-materials.json')
    results, meta = engine(rows, stage='gate')
    save('gate-items.json', results['base'])
    g = D.behavior(rows, results['base'])
    save('gate.json', {'schema': 'source_binding_backup_gate.v1', 'run_id': RUN, 'items_sha256': D.sha(OUT / 'gate-items.json'), **g, **meta})
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
        print(json.dumps({'meta': meta, 'verdicts': v, 'accuracy': {k: c['rate'] for k, c in s['accuracy'].items()},
                          'conflict': {k: c['follow'] for k, c in s['conflict'].items()}, 'paired': s['paired']}), flush=True)
    else:
        s, meta, v = {'note': 'behaviour gate failed; no edit was run'}, {}, {k: 'behavior_ineligible' for k in THRESHOLDS}
    report = {'schema': 'source_binding_backup_report.v1', 'run_id': RUN, 'split': SPLIT, 'gate_passed': g['passed'],
              'prereg_sha256': D.sha(OUT / 'prereg.json'), 'gate_sha256': D.sha(OUT / 'gate.json'),
              'items_sha256': D.sha(OUT / 'items.json') if g['passed'] else None, 'criteria': pre['criteria'], 'thresholds': pre['thresholds'],
              'frozen': pre['frozen'], 'verdicts': v, 'summary': s, **meta, 'formal_evidence_promotion': 0,
              'custody': 'holdout_bak read once by the gate stage and once by this run',
              'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    save('report.json', report)
    path = provenance.write_result('SOURCE-BINDING-backup-holdout', {'verdicts': v, 'gate_passed': g['passed'],
                                                                   'report_sha256': D.sha(OUT / 'report.json')},
                                   model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)
    print(json.dumps({'verdicts': v, 'result': str(path)}, ensure_ascii=False), flush=True)


STAGES = {f.__name__: f for f in (smoke, pilot_a, pilot_b, pilot_c, pilot_e, pilot_f, freeze, pilot_d, prereg, gate, run)}
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=list(STAGES))
    STAGES[ap.parse_args().stage]()
