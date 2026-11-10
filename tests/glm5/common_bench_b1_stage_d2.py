"""Stage D2: redo of the stage D intervention test on new, unexposed entities (benchmark COMMON-BENCH-B1-20261110-v3).

Why a redo (Phases 391, 392, 394): stage D v1 derived its random streams from a few constant bytes of the item id,
judged verdicts without the behaviour gate, and its lockbox had been opened before.  A repair run also showed that an
output-weight direction of the same norm rewrites the answer at hs[32], so a high rewrite rate does not show that the
model uses the class centres.  D2 therefore asks three separate questions on a new panel:
  T1  replication of the hs[32] rewrite with correct seeds and balanced targets (steerability only);
  T2  at an earlier layer, does the class-centre shift work through downstream computation where the equal-norm
      output-weight direction and the direct write path do not?
  T3  does removing the colour-centre subspace damage the natural answer, more than removing the output-weight subspace?
Stages: prereg -> centres (CPU, discovery) -> pilot (GPU, discovery) -> freeze -> gate (GPU, holdout2 behaviour only)
        -> run (GPU, holdout2 interventions, only if the gate passed).
"""
import os
os.environ['HF_DEACTIVATE_ASYNC_LOAD'] = '1'
import argparse, hashlib, json, random, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_bench_b1 as B
import common_bench_b1_stage_c as C

SRC = B.OUT                                    # frozen v3 benchmark and stage C / D v1 products (read only)
OUT = B.ROOT / 'tests/glm5/result/common_bench_b1_d2'
B.OUT = OUT                                    # B.save / B.log write here; C keeps its own reference to SRC
RUN, SEED, SPLIT = 'COMMON-BENCH-B1-STAGE-D2-20261110-v1', 202611100395, 'holdout2'
B.BASE[SPLIT] = 70000
COLORS = B.COLORS
NEW = {
    'red': ['garnets', 'holly berries', 'ketchup', 'paprika powder', 'goji berries', 'tomato paste'],
    'yellow': ['turmeric powder', 'sulfur powder', 'rapeseed flowers', 'forsythia flowers', 'ripe starfruit', 'custard'],
    'green': ['moss', 'fern leaves', 'kale', 'celery stalks', 'asparagus', 'clover leaves'],
    'white': ['chalk', 'tofu', 'plain yogurt', 'baking soda', 'marshmallows', 'jasmine flowers'],
    'black': ['asphalt', 'anthracite', 'gunpowder', 'graphite', 'gorillas', 'licorice candy'],
}
STEER_LAYERS, ABL_LAYERS, EARLY, PILOT_FAMILIES = [16, 20, 24, 27, 29, 32], [24, 27, 29, 32, 35], [16, 20, 24, 27, 29], 64
T1_ARMS = ['centre', 'cross_language', 'cross_source', 'random', 'wrong_donor', 'out', 'direct']
EARLY_ARMS = ['centre', 'out', 'direct']
ABL_ARMS = ['centre4', 'out4', 'out8', 'random4', 'nuisance4']
CRITERIA = {
    'material': 'holdout2: 30 new parametric entities (6 per colour) that occur in no earlier B1 version, split or pilot; '
                '240 families x 8 items; the 16 seen sentence frames; new container numbers.  Scope of every claim: '
                'new entities in seen frames.  No pre-screening of the entities by the model.',
    'two_pass': 'gate stage: unmodified forward only, full stage B behaviour gate (entity screen < 0.75 with at most 20 % '
                'entities excluded; then every source x language cell >= 0.90, every frame >= 0.90, overall top-1 >= 0.80). '
                'run stage starts interventions only if the gate passed; otherwise every verdict is behavior_ineligible.',
    'primary_set': 'items of the screened panel that the unmodified model answers correctly (10-way forced choice)',
    'seeds': 'every random stream is derived from SHA-256 of run id, seed, purpose and the full item id; '
             'target colours are balanced within source x language x original colour',
    'steering': 'answer position only; hs[l] (output of block l-1) += delta, blocks l..35 recomputed. '
                'centre: pooled discovery class-centre difference at that layer (alpha = 1, frozen in stage D v1 for l = 32); '
                'out: difference of the output-weight rows (mean over the first tokens of the class, requested language) '
                'of target and original colour, scaled to the norm of the centre delta; '
                'direct: the centre delta of layer l added after the last block (direct write path only); '
                'random: Gaussian direction of the same norm; wrong_donor: centre difference of two unrelated colours; '
                'cross_language / cross_source: centres from the other output language / other answer source',
    'norm_match': 'actual BF16 perturbation norm of out, random and wrong_donor is recorded against centre; '
                  'out and random must be within 2 % in at least 99 % of items, else controls are invalid',
    'ablation': 'answer position only; h <- h - Q Q^T (h - mu) at hs[l], mu = discovery grand mean. '
                'centre4: span of the five pooled class centres; out4: span of the colour output-weight rows averaged over '
                'both languages; out8: per-language spans stacked; random4: fixed random 4-dim subspace; '
                'nuisance4: top four principal directions of discovery residuals after removing the 20 cell means',
    'selection_on_discovery_pilot': {
        'T2_layer': 'layer in [16, 20, 24, 27, 29] maximising min(centre - out, centre - direct) success; ties: earlier layer',
        'T3_layer': 'layer in [24, 27, 29, 32, 35] maximising min(colour accuracy after out4, out8, nuisance4) minus '
                    'colour accuracy after centre4; ties: earlier layer',
        'note': 'if the discovery values at the selected layer already miss the thresholds below, the holdout run is '
                'still made once and the pre-run expectation is recorded as expected_fail'},
    'predictions_on_holdout2': {
        'T1_replication': {'controls_valid': 'random: success <= 0.05 and unchanged >= 0.90; wrong_donor: success <= 0.10; norm match ok',
                           'D1_centre': 'success >= 0.80 (not supported < 0.50)',
                           'D2_cross_language': 'success >= 0.60 and other_language_target <= 0.10 (not supported < 0.30)',
                           'D3_cross_source': 'success >= 0.60 (not supported < 0.30)',
                           'meaning': 'steerability on new entities; says nothing about natural use because the out arm is expected to succeed too'},
        'T2_downstream_use': {'supported': 'at the frozen layer: centre success >= 0.60, and both (centre - out) and '
                                           '(centre - direct) >= 0.30 with entity-bootstrap 95 % lower bound > 0.10',
                              'not_supported': '(centre - out) < 0.10 or centre success < 0.30'},
        'T3_necessity': {'supported': 'at the frozen layer: colour accuracy after centre4 <= 0.50, after random4 and '
                                      'nuisance4 >= 0.90, language kept after centre4 >= 0.95',
                         'not_supported': 'colour accuracy after centre4 >= 0.80'},
        'T3b_beats_output_subspace': {'supported': 'min(colour accuracy after out4, out8) - colour accuracy after centre4 >= 0.20 '
                                                   'with entity-bootstrap 95 % lower bound > 0.05',
                                      'not_supported': 'that difference < 0.05'}},
    'descriptive': 'full-vocabulary KL(base || edited), margins, results by source and language, all pilot layers',
    'not_covered': 'free generation, capability on unrelated tasks, unseen frames, other positions, other models; '
                   'no attention-head or MLP-unit attribution',
    'disclosure': 'before pre-registration the code was smoke-tested on two discovery families (16 items, exposed data); '
                  'the thresholds were written before that smoke test and not changed',
}


def stream(purpose, key):
    h = hashlib.sha256(f'{RUN}|{SEED}|{purpose}|{key}'.encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:16], 'little'))


def new_families():
    old = {e for v in B.ENTITIES.values() for e in v}
    new = [e for v in NEW.values() for e in v]
    assert len(new) == len(set(new)) == 30 and not old & set(new)
    for p in (SRC.parent / 'common_bench_b1/common_bench_b1.v1.py', SRC.parent / 'common_bench_b1_v2/common_bench_b1.v2.py'):
        text = p.read_text(encoding='utf-8')
        assert not [e for e in new if f"'{e}'" in text], 'entity already used in an earlier version'
    assert not any(w in e.lower() for e in new for w in COLORS)
    pairs = []
    for a, c1 in enumerate(COLORS):
        for b, c2 in enumerate(COLORS):
            if c1 != c2:
                for s in ((a + b) % 6, (a + b + 3) % 6):
                    pairs += [(NEW[c1][i], c1, NEW[c2][(i + s) % 6], c2) for i in range(6)]
    random.Random(SEED).shuffle(pairs)
    return pairs


def targets(rows):
    """Balanced target colours within source x language x original colour; order from a hash of the full id."""
    strata, out = {}, {}
    for r in rows:
        strata.setdefault((r['source'], r['language'], r['gold_color']), []).append(r)
    for key, rs in sorted(strata.items()):
        rs = sorted(rs, key=lambda r: hashlib.sha256(f"{RUN}|target|{r['id']}".encode()).hexdigest())
        others = [c for c in range(5) if c != COLORS.index(key[2])]
        off = int(hashlib.sha256(f'{RUN}|offset|{key}'.encode()).hexdigest(), 16) % 4
        for i, r in enumerate(rs):
            out[r['id']] = others[(i + off) % 4]
    return out


def prereg():
    from transformers import AutoTokenizer
    from deploy.storage_policy import checked_dir
    from tools import provenance
    from tools.power_analysis import check_phase
    checked_dir(OUT, 256 << 20, operation='阶段 D2 干预产物')
    assert not (OUT / 'd2-prereg.json').exists()
    plan = B.read(SRC / 'plan.json')
    tok = AutoTokenizer.from_pretrained(B.MODEL, local_files_only=True)
    pairs = new_families()
    rows = [r for f, pair in enumerate(pairs) for r in B.material(tok, SPLIT, f, pair)]
    assert len({tuple(r['tokens']) for r in rows}) == len(rows) == 1920
    seeds = {int.from_bytes(hashlib.sha256(f"{RUN}|{SEED}|random|{r['id']}".encode()).digest()[:16], 'little') for r in rows}
    assert len(seeds) == len(rows), 'random streams must differ for every item'
    tg = targets(rows)
    pairs_count = {}
    for r in rows:
        k = (r['gold_color'], COLORS[tg[r['id']]])
        pairs_count[k] = pairs_count.get(k, 0) + 1
    assert len(pairs_count) == 20 and max(pairs_count.values()) <= 1.5 * min(pairs_count.values()), pairs_count
    B.save(f'{SPLIT}-materials.json', rows)
    n = len(rows)
    body = {'schema': 'common_bench_d2_prereg.v1', 'run_id': RUN, 'seed': SEED, 'created_at': datetime.now().astimezone().isoformat(),
            'benchmark_plan_sha256': B.sha(SRC / 'plan.json'), 'stagec_frozen_sha256': B.sha(SRC / 'stagec-frozen.json'),
            'staged_v1_centres_sha256': B.sha(SRC / 'staged-centres.npz'), 'model_snapshot': plan['model_snapshot'],
            'code_sha256': B.sha(__file__), 'benchmark_code_sha256': B.sha(B.__file__), 'criteria': CRITERIA,
            'steer_layers': STEER_LAYERS, 'ablation_layers': ABL_LAYERS, 'pilot_families': PILOT_FAMILIES,
            'materials': {'split': SPLIT, 'sha256': B.sha(OUT / f'{SPLIT}-materials.json'), 'families': len(pairs), 'rows': n,
                          'entities': NEW, 'frames': sorted({r['frame'] for r in rows}),
                          'target_pair_counts_min_max': [min(pairs_count.values()), max(pairs_count.values())]},
            'power': {'unit': 'paired binary outcome per item (McNemar); alpha = 0.05',
                      'T1': {'expected_delta': 0.10, **check_phase(0.10, n)},
                      'T2': {'expected_delta': 0.30, 'p_d': 0.40, **check_phase(0.30, n, p_d=0.40)},
                      'T3b': {'expected_delta': 0.20, 'p_d': 0.30, **check_phase(0.20, n, p_d=0.30)},
                      'caveat': 'items are clustered in 30 entities and 16 frames; every rate carries an entity-resampling '
                                'interval and the T2 / T3b thresholds are stated on its lower bound'},
            'custody': 'centres, subspaces and layer choices come from discovery (exposed). holdout2 is generated here, '
                       'first read by the gate stage (behaviour only) and then once by the run stage. '
                       'confirm, lockbox and lockbox_frame are not used.',
            'provenance': provenance.provenance(model_path=B.MODEL, seed=SEED, tool=RUN)}
    body['provenance']['model_revision'] = plan['model_snapshot']
    B.save('d2-prereg.json', body)
    print(json.dumps({'prereg_sha256': B.sha(OUT / 'd2-prereg.json'), 'power': body['power']}, ensure_ascii=False), flush=True)


def checked():
    p = B.read(OUT / 'd2-prereg.json')
    assert p['code_sha256'] == B.sha(__file__), 'stage D2 source changed after pre-registration'
    assert p['materials']['sha256'] == B.sha(OUT / f'{SPLIT}-materials.json')
    assert p['staged_v1_centres_sha256'] == B.sha(SRC / 'staged-centres.npz')
    return p


def basis(m, k):
    """Orthonormal rows spanning the top-k directions of the centred rows of m."""
    return np.linalg.svd(m - m.mean(0), full_matrices=False)[2][:k]


def centres(save=True):
    lab = C.labels('discovery')
    fit = lab['screened'] & lab['correct'] & ~lab['reserved']
    cell = C.cell_index(lab['S'], lab['L'], lab['C'])
    old = np.load(SRC / 'staged-centres.npz')
    out, start = {}, time.monotonic()
    layers = sorted(set(STEER_LAYERS + ABL_LAYERS))
    for n, l in enumerate(layers):
        h = C.layer('discovery', l).astype(np.float64)
        cen = np.stack([h[fit & (lab['C'] == k)].mean(0) for k in range(5)])
        res = h[fit].copy()
        for c in np.unique(cell[fit]):
            res[cell[fit] == c] -= res[cell[fit] == c].mean(0)
        g = stream('random4', l).standard_normal((4, h.shape[1]))
        out[f'{l}__centre'], out[f'{l}__mu'] = cen.astype(np.float32), h[fit].mean(0).astype(np.float32)
        out[f'{l}__centre4'] = basis(cen, 4).astype(np.float32)
        out[f'{l}__nuisance4'] = np.linalg.svd(res, full_matrices=False)[2][:4].astype(np.float32)
        out[f'{l}__random4'] = np.linalg.qr(g.T)[0].T.astype(np.float32)
        B.log('centres', n + 1, len(layers), start)
    assert np.allclose(out['32__centre'], old['32__pooled'], atol=1e-4), 'layer 32 centres must equal the frozen stage D v1 centres'
    for name in ('L1', 'L-1', 'S1', 'S-1'):
        out[f'32__{name}'] = old[f'32__{name}']
    if save:
        checked()
        assert not (OUT / 'd2-centres.npz').exists()
        np.savez(OUT / 'd2-centres.npz', **out)
        print(json.dumps({'centres_sha256': B.sha(OUT / 'd2-centres.npz'), 'fit_items': int(fit.sum()),
                          'centre_delta_norm_mean': {l: round(float(np.mean([np.linalg.norm(out[f'{l}__centre'][a] - out[f'{l}__centre'][b])
                                                                              for a in range(5) for b in range(a)])), 2) for l in layers}}), flush=True)
    return out


def execute(split, rows, steer, ablate, z, base_only=False, save_rows=False):
    """steer: {layer: [arms]}, ablate: {layer: [arms]}.  Returns per-item records."""
    import torch
    from transformers import AutoTokenizer
    from shared.dist_protocol.mechanism_trace import GpuLease
    import mlp_condition_design as MD, mlp_condition_run as MR, mlp_condition_stream_load as SL
    plan = B.read(SRC / 'plan.json')
    ids = [plan['choice_ids'][k] for k in B.CLASSES]
    tok = AutoTokenizer.from_pretrained(B.MODEL, local_files_only=True)
    tok.padding_side = 'left'
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tg = targets(B.read((OUT if split == SPLIT else SRC) / f'{split}-materials.json'))
    fams = sorted({r['family'] for r in rows})
    start, items = time.monotonic(), []
    with GpuLease(), torch.inference_mode():
        MD.checked = lambda: plan
        MR.save = lambda name, body, _: None
        m = SL.load(B.MODEL, dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa').eval().requires_grad_(False)
        W = m.lm_head.weight
        outrows = np.stack([W[j].float().mean(0).cpu().numpy() for j in ids]).astype(np.float32)        # 10 x d
        ref = OUT / 'd2-outrows.npy'
        if ref.exists():
            assert np.array_equal(np.load(ref), outrows), 'output-weight rows differ from the saved ones'
        elif save_rows:
            np.save(ref, outrows)
        ob = {'out4': basis((outrows[:5] + outrows[5:]) / 2, 4),
              'out8': np.linalg.svd(np.concatenate([outrows[:5] - outrows[:5].mean(0), outrows[5:] - outrows[5:].mean(0)]),
                                    full_matrices=False)[2][:8]}
        state = {'mode': None, 'block': None, 'delta': None, 'Q': None, 'mu': None, 'norm': None}

        def hook(block):
            def fn(mod, args, out):
                if state['block'] != block or state['mode'] is None:
                    return None
                h = out[0] if isinstance(out, tuple) else out
                old = h[:, -1].float()
                if state['mode'] == 'add':
                    new = (old + state['delta']).to(h.dtype)
                else:
                    new = (old - ((old - state['mu']) @ state['Q'].T) @ state['Q']).to(h.dtype)
                state['norm'] = (new.float() - old).norm(dim=1).cpu().numpy()
                h = h.clone()
                h[:, -1] = new
                return (h, *out[1:]) if isinstance(out, tuple) else h
            return fn
        handles = [layer.register_forward_hook(hook(b)) for b, layer in enumerate(m.model.layers)]

        def run(batch, mode=None, block=None, **kw):
            state.update(mode=mode, block=block, norm=None, **kw)
            lg = m(**batch, use_cache=False, logits_to_keep=1).logits[:, 0].float()
            state['mode'] = None
            v = torch.stack([lg[:, j].max(1).values for j in ids], 1)
            return v.argmax(1).cpu().numpy(), lg.argmax(1).cpu().numpy(), v.cpu().numpy(), torch.log_softmax(lg, 1), state['norm']

        gpu = lambda a: torch.tensor(np.asarray(a), device='cuda', dtype=torch.float32)
        B.log(f'{"gate" if base_only else "intervene"}-{split}', 0, len(fams), start)
        for fi, f in enumerate(fams):
            rs = [r for r in rows if r['family'] == f]
            batch = tok.pad({'input_ids': [r['tokens'] for r in rs]}, return_tensors='pt').to('cuda')
            pred0, top0, sc0, lp0, _ = run(batch)
            rec = [{'id': r['id'], 'family': f, 'frame': r['frame'], 'source': r['source'], 'language': r['language'],
                    'query': r['query'], 'entities': r['entities'], 'gold': B.CLASSES.index(r['gold']), 'base': int(p),
                    'base_top1': int(t), 'base_top1_correct': bool(int(t) in ids[B.CLASSES.index(r['gold'])]),
                    'target_color': tg[r['id']], 'arms': {}} for r, p, t in zip(rs, pred0, top0)]
            if not base_only:
                K = [COLORS.index(r['gold_color']) for r in rs]
                T = [tg[r['id']] for r in rs]
                LG = [0 if r['language'] == 1 else 5 for r in rs]
                for l, arms in steer.items():
                    cen = z[f'{l}__centre'].astype(np.float64)
                    d = {'centre': np.stack([cen[t] - cen[k] for k, t in zip(K, T)])}
                    nrm = np.linalg.norm(d['centre'], axis=1, keepdims=True)
                    o = np.stack([outrows[g + t].astype(np.float64) - outrows[g + k] for k, t, g in zip(K, T, LG)])
                    d['out'] = o / np.linalg.norm(o, axis=1, keepdims=True) * nrm
                    d['direct'] = d['centre']
                    if 'random' in arms:
                        g = np.stack([stream('random', r['id']).standard_normal(cen.shape[1]) for r in rs])
                        d['random'] = g / np.linalg.norm(g, axis=1, keepdims=True) * nrm
                        ab = [stream('wrong_donor', r['id']).permutation([c for c in range(5) if c not in (k, t)])[:2] for r, k, t in zip(rs, K, T)]
                        d['wrong_donor'] = np.stack([cen[b] - cen[a] for a, b in ab])
                        for name, field in (('cross_language', 'language'), ('cross_source', 'source')):
                            cs = {v: z[f"{l}__{field[0].upper()}{v}"].astype(np.float64) for v in (1, -1)}
                            d[name] = np.stack([cs[-r[field]][t] - cs[-r[field]][k] for r, k, t in zip(rs, K, T)])
                    ref_norm = None
                    for arm in arms:
                        pred, top, sc, lp, norm = run(batch, 'add', 35 if arm == 'direct' else l - 1, delta=gpu(d[arm]))
                        ref_norm = norm if arm == 'centre' else ref_norm
                        kl = (lp0.exp() * (lp0 - lp)).sum(1).cpu().numpy()
                        for i, x in enumerate(rec):
                            want = LG[i] + T[i]
                            x['arms'][f'{arm}@L{l}'] = {
                                'pred': int(pred[i]), 'success': bool(pred[i] == want), 'other_language_target': bool(pred[i] == (5 - LG[i]) + T[i]),
                                'unchanged': bool(pred[i] == x['gold']), 'language_kept': bool(pred[i] // 5 == LG[i] // 5),
                                'top1_success': bool(int(top[i]) in ids[want]), 'kl': round(float(kl[i]), 4),
                                'margin_shift': round(float((sc[i, want] - sc[i, LG[i] + K[i]]) - (sc0[i, want] - sc0[i, LG[i] + K[i]])), 3),
                                'norm_ratio': round(float(norm[i] / ref_norm[i]), 4)}
                for l, arms in ablate.items():
                    mu = gpu(z[f'{l}__mu'])
                    for arm in arms:
                        Q = gpu(ob[arm] if arm in ob else z[f'{l}__{arm}'])
                        pred, top, sc, lp, norm = run(batch, 'ablate', l - 1, Q=Q, mu=mu)
                        kl = (lp0.exp() * (lp0 - lp)).sum(1).cpu().numpy()
                        for i, x in enumerate(rec):
                            x['arms'][f'{arm}@A{l}'] = {'pred': int(pred[i]), 'correct': bool(pred[i] == x['gold']),
                                                        'colour_correct': bool(pred[i] % 5 == x['gold'] % 5),
                                                        'language_kept': bool(pred[i] // 5 == x['gold'] // 5),
                                                        'kl': round(float(kl[i]), 4), 'removed_norm': round(float(norm[i]), 3)}
            items += rec
            if (fi + 1) % 8 == 0 or fi + 1 == len(fams):
                B.log(f'{"gate" if base_only else "intervene"}-{split}', fi + 1, len(fams), start)
        for h in handles:
            h.remove()
        peak = round(torch.cuda.max_memory_allocated() / 2 ** 20)
    return items, peak, round(time.monotonic() - start, 1)


def gate_of(items):
    """Full stage B behaviour gate on unmodified outputs."""
    g = B.PREREG['behavior_gate']
    acc = lambda sel, key=None: float(np.mean([(x['base'] == x['gold']) if key is None else x[key] for x in sel])) if sel else None
    ent = {}
    for x in items:
        if x['source'] == -1:
            ent.setdefault(x['entities'][0 if x['query'] == 1 else 1], []).append(x['base'] == x['gold'])
    ent = {k: {'n': len(v), 'forced': round(float(np.mean(v)), 4)} for k, v in sorted(ent.items())}
    dropped = [k for k, v in ent.items() if v['forced'] < 0.75]
    kept = [x for x in items if not set(x['entities']) & set(dropped)]
    cells = {f"{'document' if s == 1 else 'parametric'}_{'en' if l == 1 else 'zh'}":
             acc([x for x in kept if x['source'] == s and x['language'] == l]) for s in (1, -1) for l in (1, -1)}
    frames = {str(fr): acc([x for x in kept if x['frame'] == fr]) for fr in sorted({x['frame'] for x in kept})}
    top1 = acc(kept, 'base_top1_correct')
    checks = {'excluded_entities_ok': len(dropped) <= g['max_excluded_entity_fraction'] * len(ent),
              'cells_ok': all(v is not None and v >= g['min_cell_accuracy'] for v in cells.values()),
              'frames_ok': bool(frames) and min(frames.values()) >= g['min_frame_accuracy'],
              'top1_ok': top1 is not None and top1 >= g['min_overall_top1']}
    return {'items': len(items), 'forced_all': acc(items), 'top1_all': acc(items, 'base_top1_correct'), 'entities': ent,
            'dropped_entities': dropped, 'items_after_screen': len(kept), 'forced_after_screen': acc(kept), 'top1_after_screen': top1,
            'cells': cells, 'frames': frames, 'checks': checks, 'passed': bool(all(checks.values())),
            'primary_ids': [x['id'] for x in kept if x['base'] == x['gold']]}


def cluster(vals, e1, e2, n_ent, rng, reps=2000):
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return {'n': 0, 'rate': None, 'ci95': None}
    bs = []
    for _ in range(reps):
        m = rng.multinomial(n_ent, np.ones(n_ent) / n_ent)
        w = m[e1] * m[e2]
        if w.sum():
            bs.append((vals * w).sum() / w.sum())
    return {'n': int(len(vals)), 'rate': round(float(vals.mean()), 4),
            'ci95': [round(float(np.quantile(bs, .025)), 4), round(float(np.quantile(bs, .975)), 4)]}


def summarise(items, primary_ids, names):
    prim = [x for x in items if x['id'] in set(primary_ids)]
    e1 = np.array([names.index(x['entities'][0]) for x in prim])
    e2 = np.array([names.index(x['entities'][1]) for x in prim])
    rng = np.random.default_rng(SEED)
    ci = lambda v, mask=None: cluster(np.asarray(v, float)[mask] if mask is not None else v, e1[mask] if mask is not None else e1,
                                      e2[mask] if mask is not None else e2, len(names), rng)
    col = lambda key, f: np.array([x['arms'][key][f] for x in prim], float)
    out = {'primary_items': len(prim), 'arms': {}, 'differences': {}}
    for key in sorted(prim[0]['arms']) if prim else []:
        fields = ('success', 'other_language_target', 'unchanged', 'language_kept', 'top1_success') if '@L' in key else ('correct', 'colour_correct', 'language_kept')
        main = 'success' if '@L' in key else 'colour_correct'
        a = {f: ci(col(key, f)) for f in fields}
        a['mean_kl'] = round(float(col(key, 'kl').mean()), 3)
        if '@L' in key:
            a['mean_margin_shift'] = round(float(col(key, 'margin_shift').mean()), 3)
            r = col(key, 'norm_ratio')
            a['norm_ratio'] = {'median': round(float(np.median(r)), 4), 'within_2pct': round(float((np.abs(r - 1) <= .02).mean()), 4)}
        else:
            a['mean_removed_norm'] = round(float(col(key, 'removed_norm').mean()), 2)
        for cond, field in (('by_language', 'language'), ('by_source', 'source')):
            a[cond] = {str(v): ci(col(key, main), np.array([x[field] == v for x in prim])) for v in (1, -1)}
        out['arms'][key] = a
    keys = set(out['arms'])
    for l in STEER_LAYERS:
        if f'centre@L{l}' in keys:
            for other in ('out', 'direct'):
                out['differences'][f'centre-{other}@L{l}'] = ci(col(f'centre@L{l}', 'success') - col(f'{other}@L{l}', 'success'))
    for l in ABL_LAYERS:
        if f'centre4@A{l}' in keys:
            c4 = col(f'centre4@A{l}', 'colour_correct')
            out['differences'][f'min_out-centre4@A{l}'] = ci(np.minimum(col(f'out4@A{l}', 'colour_correct'), col(f'out8@A{l}', 'colour_correct')) - c4)
            out['differences'][f'min_other-centre4@A{l}'] = ci(np.minimum.reduce([col(f'{a}@A{l}', 'colour_correct') for a in ('out4', 'out8', 'nuisance4')]) - c4)
    return out


def designs(t2_layers, t3_layers):
    steer = {32: T1_ARMS, **{l: EARLY_ARMS for l in t2_layers if l != 32}}
    return steer, {l: ABL_ARMS for l in t3_layers}


def pilot(smoke=False):
    lab = C.labels('discovery')
    rows = [r for r in B.read(SRC / 'discovery-materials.json') if r['family'] < (2 if smoke else PILOT_FAMILIES)]
    if smoke:
        from deploy.storage_policy import checked_dir
        checked_dir(OUT, 256 << 20, operation='阶段 D2 干预产物')
        z = centres(save=False)
    else:
        checked()
        assert not (OUT / 'd2-pilot.json').exists()
        z = dict(np.load(OUT / 'd2-centres.npz'))
    steer, ablate = designs(STEER_LAYERS, ABL_LAYERS)
    items, peak, sec = execute('discovery', rows, steer, ablate, z, save_rows=not smoke)
    names = lab['entity_names']
    prim = [x['id'] for x in items if x['base'] == x['gold'] and not set(x['entities']) & set(lab['dropped_entities'])]
    s = summarise(items, prim, names)
    rate = lambda k, f: s['arms'][k][f]['rate']
    t2 = {l: round(min(rate(f'centre@L{l}', 'success') - rate(f'out@L{l}', 'success'),
                       rate(f'centre@L{l}', 'success') - rate(f'direct@L{l}', 'success')), 4) for l in EARLY}
    t3 = {l: round(min(rate(f'{a}@A{l}', 'colour_correct') for a in ('out4', 'out8', 'nuisance4')) - rate(f'centre4@A{l}', 'colour_correct'), 4)
          for l in ABL_LAYERS}
    pick = lambda d: min(d, key=lambda l: (-d[l], l))
    body = {'schema': 'common_bench_d2_pilot.v1', 'run_id': RUN, 'scope': 'discovery only (exposed; also the source of the centres)',
            'families': len(rows) // 8, 'summary': s, 'T2_score_by_layer': t2, 'T3_score_by_layer': t3,
            'T2_layer': pick(t2), 'T3_layer': pick(t3), 'peak_MiB': peak, 'elapsed_seconds': sec}
    if smoke:
        print(json.dumps({'smoke': True, 'items': len(items), 'T2': t2, 'T3': t3, 'peak_MiB': peak, 'seconds': sec,
                          'arms': {k: (v.get('success') or v.get('colour_correct'))['rate'] for k, v in s['arms'].items()}}), flush=True)
        return
    B.save('d2-pilot-items.json', items)
    body['items_sha256'] = B.sha(OUT / 'd2-pilot-items.json')
    B.save('d2-pilot.json', body)
    print(json.dumps({'T2': t2, 'T3': t3, 'T2_layer': body['T2_layer'], 'T3_layer': body['T3_layer']}), flush=True)


def freeze():
    checked()
    assert not (OUT / 'd2-frozen.json').exists()
    p = B.read(OUT / 'd2-pilot.json')
    r = lambda k, f: p['summary']['arms'][k][f]['rate']
    l2, l3 = p['T2_layer'], p['T3_layer']
    expect = {'T2': 'expected_pass' if p['T2_score_by_layer'][str(l2)] >= .30 and r(f'centre@L{l2}', 'success') >= .60 else 'expected_fail',
              'T3': 'expected_pass' if r(f'centre4@A{l3}', 'colour_correct') <= .50 and r(f'random4@A{l3}', 'colour_correct') >= .90
                    and r(f'nuisance4@A{l3}', 'colour_correct') >= .90 else 'expected_fail',
              'T3b': 'expected_pass' if p['T3_score_by_layer'][str(l3)] >= .20 else 'expected_fail'}
    B.save('d2-frozen.json', {'schema': 'common_bench_d2_frozen.v1', 'run_id': RUN, 'T1_layer': 32, 'alpha': 1.0, 'T2_layer': l2, 'T3_layer': l3,
                              'pre_run_expectation': expect, 'prereg_sha256': B.sha(OUT / 'd2-prereg.json'),
                              'centres_sha256': B.sha(OUT / 'd2-centres.npz'), 'outrows_sha256': B.sha(OUT / 'd2-outrows.npy'),
                              'pilot_sha256': B.sha(OUT / 'd2-pilot.json'), 'created_at': datetime.now().astimezone().isoformat()})
    print(json.dumps({'frozen_sha256': B.sha(OUT / 'd2-frozen.json'), 'T2_layer': l2, 'T3_layer': l3, 'expectation': expect}), flush=True)


def frozen():
    pre, fz = checked(), B.read(OUT / 'd2-frozen.json')
    assert fz['prereg_sha256'] == B.sha(OUT / 'd2-prereg.json') and fz['centres_sha256'] == B.sha(OUT / 'd2-centres.npz')
    assert fz['outrows_sha256'] == B.sha(OUT / 'd2-outrows.npy')
    return pre, fz


def gate():
    frozen()
    assert not (OUT / 'd2-gate.json').exists(), 'holdout2 behaviour already read'
    rows = B.read(OUT / f'{SPLIT}-materials.json')
    items, peak, sec = execute(SPLIT, rows, {}, {}, None, base_only=True)
    B.save('d2-gate-items.json', items)
    g = gate_of(items)
    B.save('d2-gate.json', {'schema': 'common_bench_d2_gate.v1', 'run_id': RUN, 'split': SPLIT, 'items_sha256': B.sha(OUT / 'd2-gate-items.json'),
                            **g, 'peak_MiB': peak, 'elapsed_seconds': sec, 'created_at': datetime.now().astimezone().isoformat()})
    print(json.dumps({k: g[k] for k in ('passed', 'checks', 'forced_all', 'dropped_entities', 'cells')}, ensure_ascii=False), flush=True)


def verdicts(s, fz, gate_passed):
    if not gate_passed:
        return {k: 'behavior_ineligible' for k in ('T1_D1_centre', 'T1_D2_cross_language', 'T1_D3_cross_source', 'T2_downstream_use',
                                                   'T3_necessity', 'T3b_beats_output_subspace')}
    r = lambda k, f: s['arms'][k][f]['rate']
    tri = lambda ok, bad: 'supported' if ok else 'not_supported' if bad else 'inconclusive'
    l2, l3 = fz['T2_layer'], fz['T3_layer']
    nm = all(s['arms'][f'{a}@L32']['norm_ratio']['within_2pct'] >= .99 for a in ('out', 'random'))
    controls = r('random@L32', 'success') <= .05 and r('random@L32', 'unchanged') >= .90 and r('wrong_donor@L32', 'success') <= .10 and nm
    v = {'T1_controls_valid': bool(controls), 'norm_match_ok': bool(nm)}
    t1 = {'T1_D1_centre': tri(r('centre@L32', 'success') >= .80, r('centre@L32', 'success') < .50),
          'T1_D2_cross_language': tri(r('cross_language@L32', 'success') >= .60 and r('cross_language@L32', 'other_language_target') <= .10,
                                      r('cross_language@L32', 'success') < .30),
          'T1_D3_cross_source': tri(r('cross_source@L32', 'success') >= .60, r('cross_source@L32', 'success') < .30)}
    v.update(t1 if controls else {k: 'controls_invalid' for k in t1})
    do, dd = s['differences'][f'centre-out@L{l2}'], s['differences'][f'centre-direct@L{l2}']
    cs = r(f'centre@L{l2}', 'success')
    v['T2_downstream_use'] = tri(cs >= .60 and min(do['rate'], dd['rate']) >= .30 and min(do['ci95'][0], dd['ci95'][0]) > .10,
                                 do['rate'] < .10 or cs < .30)
    c4 = r(f'centre4@A{l3}', 'colour_correct')
    v['T3_necessity'] = tri(c4 <= .50 and r(f'random4@A{l3}', 'colour_correct') >= .90 and r(f'nuisance4@A{l3}', 'colour_correct') >= .90
                            and r(f'centre4@A{l3}', 'language_kept') >= .95, c4 >= .80)
    db = s['differences'][f'min_out-centre4@A{l3}']
    v['T3b_beats_output_subspace'] = tri(db['rate'] >= .20 and db['ci95'][0] > .05, db['rate'] < .05)
    return v


def run_split():
    from tools import provenance
    pre, fz = frozen()
    assert not (OUT / 'd2-report.json').exists(), 'holdout2 already used'
    g = B.read(OUT / 'd2-gate.json')
    rows = B.read(OUT / f'{SPLIT}-materials.json')
    names = sorted({e for r in rows for e in r['entities']})
    if g['passed']:
        steer, ablate = designs([fz['T2_layer']], [fz['T3_layer']])
        items, peak, sec = execute(SPLIT, rows, steer, ablate, dict(np.load(OUT / 'd2-centres.npz')))
        B.save('d2-items.json', items)
        base = {x['id']: x['base'] for x in B.read(OUT / 'd2-gate-items.json')}
        mismatch = sum(base[x['id']] != x['base'] for x in items)
        s = summarise(items, g['primary_ids'], names)
        s['base_mismatch_between_passes'] = mismatch
    else:
        s, peak, sec = {'note': 'behaviour gate failed; no intervention was run'}, None, 0
    v = verdicts(s, fz, g['passed'])
    report = {'schema': 'common_bench_d2_report.v1', 'run_id': RUN, 'split': SPLIT, 'prereg_sha256': B.sha(OUT / 'd2-prereg.json'),
              'frozen_sha256': B.sha(OUT / 'd2-frozen.json'), 'gate_sha256': B.sha(OUT / 'd2-gate.json'),
              'items_sha256': B.sha(OUT / 'd2-items.json') if g['passed'] else None, 'frozen': fz, 'gate_passed': g['passed'],
              'criteria': pre['criteria'], 'verdicts': v, 'summary': s, 'peak_MiB': peak, 'elapsed_seconds': sec,
              'formal_evidence_promotion': 0, 'custody': 'holdout2 read once by the gate stage and once by this run',
              'provenance': provenance.provenance(model_path=B.MODEL, seed=SEED, tool=RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    B.save('d2-report.json', report)
    path = provenance.write_result('COMMON-BENCH-B1-stageD2-holdout2', {'verdicts': v, 'gate_passed': g['passed'],
                                                                        'report_sha256': B.sha(OUT / 'd2-report.json')},
                                   model_path=B.MODEL, seed=SEED, tool=RUN)
    print(json.dumps({'verdicts': v, 'result': str(path)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['prereg', 'centres', 'pilot', 'freeze', 'gate', 'run'])
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    {'prereg': prereg, 'centres': centres, 'pilot': lambda: pilot(a.smoke), 'freeze': freeze, 'gate': gate, 'run': run_split}[a.stage]()
