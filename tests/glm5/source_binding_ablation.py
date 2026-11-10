"""Answer-subspace removal with equal-removal controls and blocked re-fetch, Qwen3-14B BF16 (source-binding task).

Follows Claude Phase 395 (deletion asymmetry on Qwen3-4B) and GLM5 Phase 2906 (SOURCE-BINDING-20261110-v1).
The task answers come from records in the prompt, so two readings of "removing the answer subspace does not hurt
document answers" can be separated here:
  Q1  equal removal: does removing the answer-word centre subspace at the final position hurt more than removing
      an unrelated subspace of at least the same removed norm, and more than the output-weight subspace?
  Q2  re-fetch: after the removal, do later layers restore the answer by attending to the records again?
      (removal alone vs removal plus blocking attention from the final position to all record tokens)
  Q3  sustained removal at every later boundary.
Uses the frozen runtime, material builder and behaviour gate of SOURCE-BINDING-20261110-v1 without modifying them.
Stages: prereg -> fit (GPU, discovery families 0-31) -> pilot (GPU, discovery families 32-63) -> freeze
        -> gate (GPU, new panel, behaviour only) -> run (GPU, new panel, only if the gate passed).
"""
import argparse, gc, hashlib, json, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import source_binding_design as D
from source_binding_runtime import Runtime

ROOT, SRC = D.ROOT, D.OUT
OUT = ROOT / 'tests/glm5/result/source_binding_ablation_v1'
RUN, SEED, MODEL, SPLIT = 'SOURCE-BINDING-ABLATION-20261110-v1', 202611100396, 'qwen14', 'holdout_abl'
GRID, FAMILIES, CHUNK, NB = [24, 28, 32, 36], 96, 256, 40
KINDS = {  # name: (basis, block attention to records, sustained)
    'noop': (None, False, False), 'mask': (None, True, False),
    'centre': ('centre', False, False), 'out': ('out', False, False), 'rand': ('rand', False, False),
    'nuis15': ('nuis15', False, False), 'nuisM': ('nuisM', False, False),
    'centre+mask': ('centre', True, False), 'out+mask': ('out', True, False), 'nuisM+mask': ('nuisM', True, False),
    'sustain_centre': ('centre', False, True), 'sustain_out': ('out', False, True), 'sustain_nuisM': ('nuisM', False, True)}
CRITERIA = {
    'model': 'Qwen3-14B original BF16, CPU byte reads, CUDA layer streaming (runtime of SOURCE-BINDING-20261110-v1). '
             'GLM-4-9B is not run: it failed the behaviour gate of this task in GLM5 Phase 2906.',
    'material': 'holdout_abl: 96 new families x 8 items; 192 new entity names that occur in no split of '
                'SOURCE-BINDING-20261110-v1; the 16 discovery code words and the 4 seen frames. Scope of every claim: '
                'new entities, seen answer words, seen frames.  confirm and lockbox_frame of v1 are not used.',
    'fit': 'discovery families 0-31 (256 items, exposed): final-position state at every boundary hs[24..40]. '
           'centre: 15-dim span of the 16 answer-word class means; nuis15: top 15 principal directions of the residuals '
           'after removing the answer-word means; nuisM: top k such directions, k the smallest of [15, 30, 60, 120, 200] '
           'whose mean removed norm on the fit items is at least that of centre (equal-removal control); '
           'rand: fixed random 15-dim subspace; out: 15-dim span of the output-weight rows of the 16 answer words '
           '(same at every layer); mu: fit mean.',
    'edit': 'final position only.  removal at hs[l] (input of block l): h <- h - Q Q^T (h - mu). '
            'mask: from block l on, attention from the final position to every token of the four records is blocked '
            '(all heads). sustained: the removal is repeated at every later boundary with that boundary\'s own basis. '
            'Everything downstream is recomputed by the original blocks.',
    'outcome': '4-way forced choice among the four code words of the item (chance 0.25); all items are analysed, '
               'none is filtered by correctness; rates carry family-bootstrap 95 % intervals (2000 resamples)',
    'two_pass': 'gate stage: unmodified forward only, behaviour gate of v1 (every S x E x O cell and every frame: '
                'n >= 16, forced >= 0.95, top-1 >= 0.90, no exclusions). run stage edits only if the gate passed; '
                'otherwise every verdict is behavior_ineligible.',
    'layer_rule': 'on the discovery pilot (families 32-63): the earliest layer in [24, 28, 32, 36] whose mask-only arm '
                  'keeps forced accuracy >= 0.90 (the answer has reached the final position); if none, 36 (flagged)',
    'predictions_on_holdout': {
        'controls_valid': 'noop reproduces the base logits bit for bit; rand >= 0.95; mask >= 0.90',
        'Q1_equal_removal': {'supported': 'acc(nuisM) - acc(centre) >= 0.20 and acc(out) - acc(centre) >= 0.20, both with '
                                          'family-bootstrap lower bound > 0.05, and nuisM removes at least as much norm as centre',
                             'not_supported': 'either difference < 0.05'},
        'Q2_refetch': {'supported': 'acc(centre) - acc(centre+mask) >= 0.20 with lower bound > 0.05, and acc(nuisM+mask) >= 0.90',
                       'not_supported': 'that difference < 0.05'},
        'Q3_sustained': {'supported': 'acc(sustain_centre) <= 0.50 and acc(sustain_nuisM) >= 0.90',
                         'not_supported': 'acc(sustain_centre) >= 0.80'}},
    'descriptive': 'top-1, gold margin change, full-vocabulary KL, removed norms, out and sustain_out arms, all pilot layers',
    'not_covered': 'free generation, other positions, unseen answer words, unseen frames, head- or unit-level attribution, other models',
    'disclosure': 'before pre-registration the code was smoke-tested on discovery items (exposed); thresholds were written first',
}


def stream(purpose, key):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(f'{RUN}|{SEED}|{purpose}|{key}'.encode()).digest()[:16], 'little'))


def save(name, body):
    from deploy.storage_policy import check_write
    p = OUT / name
    check_write(p, 1 << 20, operation='来源绑定删除对照产物')
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(body, ensure_ascii=False, indent=1, default=lambda x: x.item() if isinstance(x, np.generic) else x), encoding='utf-8')
    t.replace(p)


def log(stage, done, total, start, **kw):
    from source_binding_resources import resources
    sec = time.monotonic() - start
    row = {'time': datetime.now().astimezone().isoformat(), 'stage': stage, 'done': done, 'total': total, 'seconds': round(sec, 1),
           'eta_seconds': round(sec / done * (total - done), 1) if done else None, **resources(), **kw}
    if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
        t = sys.modules['torch']
        row.update(cuda_allocated_MiB=round(t.cuda.memory_allocated() / 2 ** 20), cuda_peak_MiB=round(t.cuda.max_memory_allocated() / 2 ** 20))
    print(json.dumps(row, ensure_ascii=False), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'progress.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')
    assert row['ram_available_bytes'] > 2 * 2 ** 30 and row['commit_available_bytes'] > 4 * 2 ** 30, 'RAM / commit resource stop'
    if row.get('shared_gpu_bytes') is not None:
        assert row['shared_gpu_bytes'] < 2 ** 30, 'GPU shared residency above 1 GiB'


def old_rows(split=None):
    m = D.read(SRC / f'{MODEL}-materials.json')['splits']
    return m[split] if split else m


def build_materials():
    from transformers import AutoTokenizer
    used = {n for rows in old_rows().values() for r in rows for n in r['names']}
    tok = AutoTokenizer.from_pretrained(D.MODELS[MODEL], local_files_only=True)
    rows = []
    for family in range(FAMILIES):
        rnd, names = stream('names', family), []
        while len(names) < 2:
            name = ''.join(rnd.choice(list('bdfgklmnprstvz')) + rnd.choice(list('aeiou')) for _ in range(3)).capitalize()
            if name not in used:
                used.add(name)
                names.append(name)
        values = rnd.choice(D.WORDS[:16], 4, replace=False).tolist()
        for s in (0, 1):
            for e in (0, 1):
                for o in (0, 1):
                    r = D.material(family, SPLIT, s, e, o, family % 4, names, values)
                    prompt = tok.apply_chat_template([{'role': 'user', 'content': r['content']}], tokenize=False,
                                                     add_generation_prompt=True, enable_thinking=False) + 'Answer:\n'
                    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
                    off, offsets = prompt.index(r['content']), enc['offset_mapping']

                    def indices(span):
                        a, b = span
                        out = [i for i, (lo, hi) in enumerate(offsets) if hi > off + a and lo < off + b]
                        assert out
                        return out
                    recs = {str(k): {key: indices(v) for key, v in info.items()} for k, info in r['records'].items()}
                    choice = []
                    for val in r['values']:
                        suffix = tok.encode(prompt + val, add_special_tokens=False)
                        assert suffix[:-1] == enc['input_ids'], 'answer word must be one token with a stable prefix'
                        choice.append(suffix[-1])
                    rows.append({**r, 'records': recs, 'prompt': prompt, 'tokens': enc['input_ids'], 'choice_ids': choice,
                                 'gold_id': choice[r['target_record']]})
    return rows


def prereg():
    from deploy.storage_policy import checked_dir
    from tools import provenance
    from tools.power_analysis import check_phase
    checked_dir(OUT, 1 << 30, operation='来源绑定删除对照产物')
    assert not (OUT / 'prereg.json').exists()
    plan = D.checked(MODEL)
    rows = build_materials()
    assert len(rows) == FAMILIES * 8 and len({tuple(r['tokens']) for r in rows}) == len(rows)
    save(f'{SPLIT}-materials.json', rows)
    n = len(rows)
    body = {'schema': 'source_binding_ablation_prereg.v1', 'run_id': RUN, 'seed': SEED, 'created_at': datetime.now().astimezone().isoformat(),
            'base_plan_sha256': D.sha(SRC / 'plan.json'), 'base_materials_sha256': D.sha(SRC / f'{MODEL}-materials.json'),
            'model_snapshot': plan['models'][MODEL], 'code_sha256': D.sha(__file__), 'criteria': CRITERIA, 'grid': GRID,
            'materials': {'split': SPLIT, 'sha256': D.sha(OUT / f'{SPLIT}-materials.json'), 'families': FAMILIES, 'rows': n,
                          'max_tokens': max(len(r['tokens']) for r in rows)},
            'power': {'unit': 'paired binary outcome per item (McNemar); alpha = 0.05; expected flip rate 0.30',
                      'Q1': {'expected_delta': 0.20, **check_phase(0.20, n, p_d=0.30)},
                      'Q2': {'expected_delta': 0.20, **check_phase(0.20, n, p_d=0.30)},
                      'Q3': {'expected_delta': 0.40, **check_phase(0.40, n, p_d=0.50)},
                      'caveat': 'items are clustered in 96 families and 4 frames; thresholds are stated on family-bootstrap lower bounds'},
            'custody': 'subspaces and the layer come from discovery (exposed). holdout_abl is generated here, read first by '
                       'the gate stage (behaviour only) and then once by the run stage.',
            'resources': {'gpu_minutes_estimate': 40, 'disk_GiB_estimate': 0.5, 'models_concurrent': 1},
            'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    body['provenance']['model_revision'] = plan['models'][MODEL]
    save('prereg.json', body)
    print(json.dumps({'prereg_sha256': D.sha(OUT / 'prereg.json'), 'power': body['power']}, ensure_ascii=False), flush=True)


def checked():
    p = D.read(OUT / 'prereg.json')
    assert p['code_sha256'] == D.sha(__file__), 'source changed after pre-registration'
    assert p['materials']['sha256'] == D.sha(OUT / f'{SPLIT}-materials.json')
    D.checked(MODEL)
    return p


def engine(rows, arms=(), z=None, capture=False, stage='run'):
    """arms: list of (name, layer).  Returns scored rows per arm (plus 'base'), captured states, peak memory."""
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    t = torch
    assert snapshot(D.MODELS[MODEL]) == D.read(SRC / f'{MODEL}-identity.json')['model_snapshot']
    results = {f'{k}@{l}': [] for k, l in arms}
    results['base'] = []
    caps, removed, start, noop_equal = [], {}, time.monotonic(), []
    gpu = lambda a: t.tensor(np.asarray(a), device='cuda', dtype=t.float32)
    with GpuLease(), t.inference_mode():
        rt = Runtime(MODEL)
        Z = {k: gpu(v) for k, v in z.items()} if z is not None else {}

        def remove(x, basis, layer):
            Q = Z['out'] if basis == 'out' else Z[f'{layer}__{basis}']
            old = x.float()
            new = (old - ((old - Z[f'{layer}__mu']) @ Q.T) @ Q).to(x.dtype)
            return new, (new.float() - old).norm(dim=1)

        total = len(rows) * NB
        log(stage, 0, total, start)
        for chunk in range(0, len(rows), CHUNK):
            rs = rows[chunk:chunk + CHUNK]
            ids, pos, pe, mask = rt.context(rs)
            h, n, length = rt.embed(ids), len(rs), ids.shape[1]
            blocked = mask.clone()
            for i, r in enumerate(rs):
                idx = [length - len(r['tokens']) + j for k in range(4) for j in r['records'][str(k)]['span']]
                blocked[i, 0, -1, idx] = False
            state, cap = {}, t.zeros((n, NB + 1, rt.dim), dtype=t.float16) if capture else None
            for b in range(NB):
                if capture:
                    cap[:, b] = h[:, -1].float().cpu().half()
                for name, l in arms:
                    if l == b:
                        s = h[:, -1].clone()
                        if KINDS[name][0]:
                            s, nr = remove(s, KINDS[name][0], b)
                            removed.setdefault(f'{name}@{l}', []).extend(nr.cpu().tolist())
                        state[f'{name}@{l}'] = s
                layer = rt.block(b)
                for lo in range(0, n, 8):
                    hi = min(lo + 8, n)
                    kw = {'position_ids': pos[lo:hi], 'position_embeddings': tuple(p[lo:hi] for p in pe), 'use_cache': False}
                    y = layer(h[lo:hi], attention_mask=mask[lo:hi], **kw)
                    for key, s in state.items():
                        basis, use_mask, sustain = KINDS[key.split('@')[0]]
                        x = h[lo:hi].clone()
                        x[:, -1] = s[lo:hi]
                        ya = layer(x, attention_mask=(blocked if use_mask else mask)[lo:hi], **kw)[:, -1]
                        if sustain:
                            ya, _ = remove(ya, basis, b + 1)
                        s[lo:hi] = ya
                    h[lo:hi] = y
                del layer
                gc.collect()
                t.cuda.empty_cache()
                if (b + 1) % 5 == 0:
                    log(stage, chunk * NB + (b + 1) * n, total, start, block=b, arms=len(state))
            if capture:
                cap[:, NB] = h[:, -1].float().cpu().half()
                caps.append(cap.numpy())
            norm = rt.Norm(rt.dim, eps=rt.cfg.rms_norm_eps).to(device='cuda', dtype=t.bfloat16)
            norm.weight = t.nn.Parameter(rt.weight('model.norm.weight'), requires_grad=False)
            W = rt.weight('lm_head.weight' if 'lm_head.weight' in rt.index else 'model.embed_tokens.weight')
            lg0 = (norm(h[:, -1]) @ W.T).float()
            lp0 = lg0.log_softmax(-1)
            results['base'] += rt.score(lg0, rs)
            for key, s in state.items():
                lg = (norm(s) @ W.T).float()
                if key.startswith('noop'):
                    noop_equal.append(bool(t.equal(lg, lg0)))
                kl = (lp0.exp() * (lp0 - lg.log_softmax(-1))).sum(-1).cpu().tolist()
                scored = rt.score(lg, rs)
                for x, v in zip(scored, kl):
                    x['kl'] = v
                results[key] += scored
            del W, norm, h, state, lg0, lp0, ids, pos, pe, mask, blocked
            gc.collect()
            t.cuda.empty_cache()
        peak = round(t.cuda.max_memory_allocated() / 2 ** 20)
    return results, (np.concatenate(caps) if capture else None), {k: float(np.mean(v)) for k, v in removed.items()}, \
        {'noop_bitwise_equal': noop_equal, 'peak_MiB': peak, 'seconds': round(time.monotonic() - start, 1)}


def subspaces(states, rows, outrows):
    """states: [n, 41, d] final-position states of the fit items."""
    words = sorted({r['gold'] for r in rows})
    assert len(words) == 16
    y = np.array([words.index(r['gold']) for r in rows])
    counts = np.bincount(y, minlength=16)
    assert counts.min() >= 4, counts
    z = {'out': np.linalg.svd(outrows - outrows.mean(0), full_matrices=False)[2][:15].astype(np.float32)}
    info = {'fit_items': len(rows), 'word_counts_min_max': [int(counts.min()), int(counts.max())], 'layers': {}}
    for l in range(min(GRID), NB + 1):
        h = states[:, l].astype(np.float64)
        mu = h.mean(0)
        cen = np.stack([h[y == k].mean(0) for k in range(16)])
        Q = np.linalg.svd(cen - cen.mean(0), full_matrices=False)[2][:15]
        res = h - cen[y]
        V = np.linalg.svd(res, full_matrices=False)[2]
        rm = lambda B: float(np.linalg.norm((h - mu) @ B.T, axis=1).mean())
        k = next((k for k in (15, 30, 60, 120, 200) if rm(V[:k]) >= rm(Q)), 200)
        g = stream('rand', l).standard_normal((15, h.shape[1]))
        z[f'{l}__mu'], z[f'{l}__centre'] = mu.astype(np.float32), Q.astype(np.float32)
        z[f'{l}__nuis15'], z[f'{l}__nuisM'] = V[:15].astype(np.float32), V[:k].astype(np.float32)
        z[f'{l}__rand'] = np.linalg.qr(g.T)[0].T.astype(np.float32)
        info['layers'][l] = {'nuisM_k': k, 'removed_centre': round(rm(Q), 2), 'removed_nuisM': round(rm(V[:k]), 2),
                             'removed_nuis15': round(rm(V[:15]), 2), 'removed_out': round(rm(z['out'].astype(np.float64)), 2),
                             'matched': bool(rm(V[:k]) >= rm(Q)), 'mean_state_norm': round(float(np.linalg.norm(h, axis=1).mean()), 2)}
    return z, info


def fit_rows():
    rows = old_rows('discovery')
    return [r for r in rows if int(r['family'][-4:]) < 32], [r for r in rows if int(r['family'][-4:]) >= 32]


def out_rows(rows):
    import torch
    rt = Runtime(MODEL)
    ids = {}
    for r in rows:
        ids.update(dict(zip(r['values'], r['choice_ids'])))
    words = sorted(ids)
    assert words == sorted(D.WORDS[:16])
    W = rt.weight('lm_head.weight' if 'lm_head.weight' in rt.index else 'model.embed_tokens.weight', device='cpu')
    return W[[ids[w] for w in words]].float().numpy()


def fit(smoke=False):
    if not smoke:
        checked()
        assert not (OUT / 'subspaces.npz').exists()
    rows, _ = fit_rows()
    _, states, _, meta = engine(rows, capture=True, stage='fit')
    z, info = subspaces(states, rows, out_rows(old_rows('discovery')))
    if smoke:
        return z, info
    np.savez(OUT / 'subspaces.npz', **z)
    save('fit.json', {'schema': 'source_binding_ablation_fit.v1', 'run_id': RUN, 'subspaces_sha256': D.sha(OUT / 'subspaces.npz'), **info, **meta})
    print(json.dumps({'subspaces_sha256': D.sha(OUT / 'subspaces.npz'), **{str(l): info['layers'][l] for l in GRID}}), flush=True)


def summarise(results, rows, removed):
    fam = np.array([r['family'] for r in rows])
    names = sorted(set(fam))
    members = [np.flatnonzero(fam == f) for f in names]
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, len(names), (2000, len(names)))

    def ci(v):
        v = np.asarray(v, float)
        per = np.array([v[m].sum() for m in members])
        cnt = np.array([len(m) for m in members])
        bs = per[draws].sum(1) / cnt[draws].sum(1)
        return {'rate': round(float(v.mean()), 4), 'ci95': [round(float(np.quantile(bs, .025)), 4), round(float(np.quantile(bs, .975)), 4)]}
    col = lambda k, f: np.array([x[f] for x in results[k]], float)
    base_margin = col('base', 'gold_margin')
    out = {'n': len(rows), 'arms': {}, 'differences': {}}
    for k in results:
        out['arms'][k] = {'forced': ci(col(k, 'forced_correct')), 'top1': ci(col(k, 'top1_correct')),
                          'gold_margin_change': round(float((col(k, 'gold_margin') - base_margin).mean()), 3),
                          'mean_kl': round(float(col(k, 'kl').mean()), 4) if k != 'base' else 0.0,
                          'mean_removed_norm': round(removed[k], 2) if k in removed else None}
    for l in sorted({int(k.split('@')[1]) for k in results if '@' in k}):
        f = lambda name: col(f'{name}@{l}', 'forced_correct')
        for a, b in (('nuisM', 'centre'), ('out', 'centre'), ('centre', 'centre+mask'), ('nuisM', 'nuisM+mask'), ('out', 'out+mask')):
            out['differences'][f'{a}-{b}@{l}'] = ci(f(a) - f(b))
    return out


def arms_for(layers):
    return [(name, l) for l in layers for name in KINDS]


def pilot(smoke=False):
    _, rows = fit_rows()
    if smoke:
        z, info = fit(smoke=True)
        print(json.dumps({'fit': {str(l): info['layers'][l] for l in GRID}}), flush=True)
        rows = rows[:16]
    else:
        checked()
        assert not (OUT / 'pilot.json').exists()
        z = dict(np.load(OUT / 'subspaces.npz'))
    results, _, removed, meta = engine(rows, arms_for(GRID), z, stage='pilot')
    s = summarise(results, rows, removed)
    acc = lambda k: s['arms'][k]['forced']['rate']
    layer = next((l for l in GRID if acc(f'mask@{l}') >= .90), None)
    body = {'schema': 'source_binding_ablation_pilot.v1', 'run_id': RUN, 'scope': 'discovery families 32-63 (exposed)', 'summary': s,
            'layer': layer if layer is not None else GRID[-1], 'layer_rule_met': layer is not None, **meta}
    table = {k: acc(k) for k in s['arms']}
    if smoke:
        print(json.dumps({'smoke': True, 'layer': body['layer'], 'meta': meta, 'forced': table}), flush=True)
        return
    assert all(meta['noop_bitwise_equal'])
    save('pilot-items.json', results)
    body['items_sha256'] = D.sha(OUT / 'pilot-items.json')
    save('pilot.json', body)
    print(json.dumps({'layer': body['layer'], 'rule_met': body['layer_rule_met'], 'forced': table}), flush=True)


def freeze():
    checked()
    assert not (OUT / 'frozen.json').exists()
    p, f = D.read(OUT / 'pilot.json'), D.read(OUT / 'fit.json')
    save('frozen.json', {'schema': 'source_binding_ablation_frozen.v1', 'run_id': RUN, 'layer': p['layer'], 'layer_rule_met': p['layer_rule_met'],
                         'pilot_expectation': verdicts(p['summary'], p['layer'], True, f['layers'][str(p['layer'])], [True]),
                         'prereg_sha256': D.sha(OUT / 'prereg.json'), 'subspaces_sha256': D.sha(OUT / 'subspaces.npz'),
                         'fit_sha256': D.sha(OUT / 'fit.json'), 'pilot_sha256': D.sha(OUT / 'pilot.json'),
                         'created_at': datetime.now().astimezone().isoformat()})
    print(json.dumps(D.read(OUT / 'frozen.json'), ensure_ascii=False), flush=True)


def frozen():
    pre, fz = checked(), D.read(OUT / 'frozen.json')
    assert fz['prereg_sha256'] == D.sha(OUT / 'prereg.json') and fz['subspaces_sha256'] == D.sha(OUT / 'subspaces.npz')
    return pre, fz


def gate():
    frozen()
    assert not (OUT / 'gate.json').exists(), 'holdout behaviour already read'
    rows = D.read(OUT / f'{SPLIT}-materials.json')
    results, _, _, meta = engine(rows, stage='gate')
    save('gate-items.json', results['base'])
    g = D.behavior(rows, results['base'])
    save('gate.json', {'schema': 'source_binding_ablation_gate.v1', 'run_id': RUN, 'items_sha256': D.sha(OUT / 'gate-items.json'), **g, **meta})
    print(json.dumps({k: g[k] for k in ('passed', 'forced_accuracy', 'top1_accuracy')}), flush=True)


def verdicts(s, l, gate_passed, fitinfo, noop):
    names = ('Q1_equal_removal', 'Q2_refetch', 'Q3_sustained')
    if not gate_passed:
        return {k: 'behavior_ineligible' for k in names}
    acc = lambda k: s['arms'][f'{k}@{l}']['forced']['rate']
    d = lambda k: s['differences'][f'{k}@{l}']
    tri = lambda ok, bad: 'supported' if ok else 'not_supported' if bad else 'inconclusive'
    controls = all(noop) and acc('rand') >= .95 and acc('mask') >= .90
    v = {'controls_valid': bool(controls), 'equal_removal_matched': bool(fitinfo['matched'])}
    if not controls:
        return {**v, **{k: 'controls_invalid' for k in names}}
    a, b = d('nuisM-centre'), d('out-centre')
    v['Q1_equal_removal'] = tri(min(a['rate'], b['rate']) >= .20 and min(a['ci95'][0], b['ci95'][0]) > .05 and fitinfo['matched'],
                                min(a['rate'], b['rate']) < .05)
    c = d('centre-centre+mask')
    v['Q2_refetch'] = tri(c['rate'] >= .20 and c['ci95'][0] > .05 and acc('nuisM+mask') >= .90, c['rate'] < .05)
    v['Q3_sustained'] = tri(acc('sustain_centre') <= .50 and acc('sustain_nuisM') >= .90, acc('sustain_centre') >= .80)
    return v


def run():
    from tools import provenance
    pre, fz = frozen()
    assert not (OUT / 'report.json').exists(), 'holdout already used'
    g, f = D.read(OUT / 'gate.json'), D.read(OUT / 'fit.json')
    rows = D.read(OUT / f'{SPLIT}-materials.json')
    l = fz['layer']
    if g['passed']:
        results, _, removed, meta = engine(rows, arms_for([l]), dict(np.load(OUT / 'subspaces.npz')), stage='run')
        save('items.json', results)
        base = D.read(OUT / 'gate-items.json')
        meta['base_replay_max_score_difference'] = max(abs(a - b) for x, y in zip(results['base'], base) for a, b in zip(x['scores'], y['scores']))
        s = summarise(results, rows, removed)
    else:
        s, meta = {'note': 'behaviour gate failed; no edit was run'}, {'noop_bitwise_equal': []}
    v = verdicts(s, l, g['passed'], f['layers'][str(l)], meta['noop_bitwise_equal'])
    report = {'schema': 'source_binding_ablation_report.v1', 'run_id': RUN, 'split': SPLIT, 'layer': l, 'gate_passed': g['passed'],
              'prereg_sha256': D.sha(OUT / 'prereg.json'), 'frozen_sha256': D.sha(OUT / 'frozen.json'), 'gate_sha256': D.sha(OUT / 'gate.json'),
              'items_sha256': D.sha(OUT / 'items.json') if g['passed'] else None, 'criteria': pre['criteria'], 'fit_layer': f['layers'][str(l)],
              'verdicts': v, 'summary': s, **meta, 'formal_evidence_promotion': 0,
              'custody': 'holdout_abl read once by the gate stage and once by this run',
              'provenance': provenance.provenance(model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    save('report.json', report)
    path = provenance.write_result('SOURCE-BINDING-ablation-holdout', {'verdicts': v, 'gate_passed': g['passed'], 'layer': l,
                                                                       'report_sha256': D.sha(OUT / 'report.json')},
                                   model_path=D.MODELS[MODEL], seed=SEED, tool=RUN)
    print(json.dumps({'verdicts': v, 'result': str(path)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['prereg', 'fit', 'pilot', 'freeze', 'gate', 'run'])
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    {'prereg': prereg, 'fit': fit, 'pilot': lambda: pilot(a.smoke), 'freeze': freeze, 'gate': gate, 'run': run}[a.stage]()
