"""Stage C (Phase 384) on benchmark COMMON-BENCH-B1-20261110-v3.

Stages:  prereg -> capture --split discovery -> fit -> [capture --split confirm -> evaluate, later, once].
Fits use discovery only, never the reserved combination cells.  Criteria below were written before capture.
"""
import os
os.environ['HF_DEACTIVATE_ASYNC_LOAD'] = '1'
import argparse, json, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_bench_b1 as B

ROOT, OUT = B.ROOT, B.OUT
RUN = 'COMMON-BENCH-B1-STAGE-C-20261110-v1'
POSITIONS = ['doc_end', 'question_end', 'answer_prefix']
NL, D = 37, 2560
CRITERIA = {
    'scope': 'answer_prefix position; raw residual states hs[1..35]; hs[36] is reported but never selected (final norm)',
    'fit_set': 'discovery items that pass the stage B behaviour rule and are outside the reserved combination cells',
    'evaluation_set': 'confirm, once, after the estimator file is frozen; behaviour-correct items after the entity screen '
                      'are primary, all items are reported beside them',
    'layer_selection': 'on discovery only, leave-entities-out folds: earliest raw layer from which the M2 accuracy '
                       '(minimum of both transfer directions) stays >= 0.95 for three consecutive layers; '
                       'if none, the raw layer with the highest value',
    'readouts': {'weights': 'mean difference or class centroids, divided by per-coordinate variance of the fit items; no tuning',
                 'M1': 'centre and direction (or centroids) from the training condition',
                 'M2': 'direction (or centroid offsets) from the training condition, centre = mean of the target condition '
                       '(uses the condition label of the target items, not their class labels)',
                 'M3': 'everything from the target condition (reference)'},
    'P1_language_across_source': {'supported': 'M2 >= 0.95 in both directions and M3 - M2 <= 0.02',
                                  'direction_matters': 'M2 < 0.95 in some direction while M3 >= 0.95',
                                  'otherwise': 'unresolved'},
    'P2a_colour_across_output_language': {'supported': 'M2 >= 0.80 in both directions (chance 0.20)',
                                          'not_supported': 'M2 < 0.50 in some direction', 'otherwise': 'inconclusive'},
    'P2b_colour_across_source': {'supported': 'M2 >= 0.80 in both directions (chance 0.20)',
                                 'not_supported': 'M2 < 0.50 in some direction', 'otherwise': 'inconclusive'},
    'P3_unseen_combination': {'model': 'A2 = constant + source + language + colour + all pairwise interactions, '
                                       'least squares on the fit set; A1 = main effects only',
                              'test': '20-way nearest predicted centroid (source x language x colour) at hs[34], '
                                      'per-coordinate residual variance weighting, on confirm items of the two reserved cells',
                              'supported': 'A2 accuracy >= 0.80 in both reserved cells (chance 0.05)',
                              'not_supported': 'A2 accuracy < 0.50 in some reserved cell', 'otherwise': 'inconclusive'},
    'P4_update': 'descriptive, no pass line: one-step prediction of hs[l+1] from hs[l] by identity, mean increment, '
                 'labelled-cell increment (oracle) and decoded-cell increment (cell decoded from hs[l], no labels); '
                 'raw and per-coordinate standardised error relative to the mean-increment baseline',
    'baseline_groups': {'text_only': 'bigram token logistic regression refit on the fit set',
                        'model_state': 'decoded-cell predictors', 'oracle': 'predictors given S, L or colour labels'},
    'power': 'see stagec-prereg.json; three columns computed with tools/power_analysis.py',
}


def bits_to_f32(a):
    return (np.asarray(a, dtype=np.uint32) << 16).view(np.float32)


def store(split):
    from deploy.storage_policy import storage_dir
    return storage_dir('reusable', 'common_bench_b1_v3', split)


def prereg():
    from tools import provenance
    from tools.power_analysis import check_phase
    plan = B.read(OUT / 'plan.json')
    assert not (OUT / 'stagec-prereg.json').exists()
    rows = B.read(OUT / 'confirm-materials.json')     # labels only; nothing is run through the model here
    cells = [{'source': 1, 'language': -1, 'answer': 'green'}, {'source': -1, 'language': 1, 'answer': 'white'}]
    n_res = [sum(r['source'] == c['source'] and r['language'] == c['language'] and r['gold_color'] == c['answer'] for r in rows) for c in cells]
    half = len(rows) // 2
    body = {'schema': 'common_bench_stagec_prereg.v1', 'run_id': RUN, 'benchmark_plan_sha256': B.sha(OUT / 'plan.json'),
            'model_snapshot': plan['model_snapshot'], 'code_sha256': B.sha(__file__), 'criteria': CRITERIA,
            'created_at': datetime.now().astimezone().isoformat(),
            'power': {'transfer_tests': {'expected_delta': 0.03, 'n_per_direction': half, **check_phase(0.03, half)},
                      'reserved_cells': {'expected_delta': 0.10, 'n_per_cell': n_res,
                                         'checks': [check_phase(0.10, n) for n in n_res]},
                      'caveat': 'items cluster in 25 confirm entities; entity-clustered bootstrap intervals accompany every accuracy'},
            'storage': {'bytes_per_item': NL * 3 * D * 2 + 36 * D * 2, 'target': str(store('discovery').parent),
                        'purpose': 'reusable native fields for stages C, D and E; rebuildable from the frozen materials'},
            'custody': 'confirm is captured only after stagec-frozen.json exists; lockbox stays sealed in stage C',
            'provenance': provenance.provenance(model_path=B.MODEL, seed=B.SEED, tool=RUN)}
    body['provenance']['model_revision'] = plan['model_snapshot']
    B.save('stagec-prereg.json', body)
    print(json.dumps({'prereg_sha256': B.sha(OUT / 'stagec-prereg.json'), 'power': body['power']}, ensure_ascii=False), flush=True)


def checked():
    p = B.read(OUT / 'stagec-prereg.json')
    assert p['code_sha256'] == B.sha(__file__), 'stage C source changed after pre-registration'
    assert p['benchmark_plan_sha256'] == B.sha(OUT / 'plan.json')
    return p


def capture(split):
    import torch
    from transformers import AutoTokenizer
    from deploy.storage_policy import checked_dir
    from shared.dist_protocol.mechanism_trace import GpuLease
    import mlp_condition_design as MD, mlp_condition_run as MR, mlp_condition_stream_load as SL
    checked()
    plan = B.read(OUT / 'plan.json')
    assert split == 'discovery' or (split == 'confirm' and (OUT / 'stagec-frozen.json').exists()), 'confirm is sealed until the fit is frozen'
    assert B.sha(OUT / f'{split}-materials.json') == plan['materials'][split]['sha256']
    rows = B.read(OUT / f'{split}-materials.json')
    n = len(rows)
    need = n * (NL * 3 * D * 2 + 36 * D * 2)
    folder = checked_dir(store(split), need, operation=f'阶段 C 原生场（{split}）')
    assert not (folder / 'manifest.json').exists(), 'already captured'
    hidden = np.lib.format.open_memmap(folder / 'hidden.npy', mode='w+', dtype=np.uint16, shape=(n, NL, 3, D))
    post = np.lib.format.open_memmap(folder / 'postattn.npy', mode='w+', dtype=np.uint16, shape=(n, 36, D))
    ids = [plan['choice_ids'][k] for k in B.CLASSES]
    tok = AutoTokenizer.from_pretrained(B.MODEL, local_files_only=True)
    tok.padding_side = 'left'
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    start, behaviour, norm_check = time.monotonic(), [], None
    with GpuLease(), torch.inference_mode():
        MD.checked = lambda: plan
        MR.save = lambda name, body, _: None
        m = SL.load(B.MODEL, dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa').eval().requires_grad_(False)
        grabbed, last = {}, {}
        hooks = [layer.post_attention_layernorm.register_forward_pre_hook(
            lambda mod, args, b=b: grabbed.__setitem__(b, args[0][:, -1])) for b, layer in enumerate(m.model.layers)]
        hooks.append(m.model.layers[35].register_forward_hook(lambda mod, args, out: last.__setitem__('h', (out[0] if isinstance(out, tuple) else out)[:, -1])))
        fams = n // 8
        B.log(f'capture-{split}', 0, fams, start)
        for f in range(fams):
            rs = rows[f * 8:f * 8 + 8]
            batch = tok.pad({'input_ids': [r['tokens'] for r in rs]}, return_tensors='pt').to('cuda')
            width = batch['input_ids'].shape[1]
            out = m(**batch, use_cache=False, output_hidden_states=True)
            hs = torch.stack(out.hidden_states, 1)                      # [8, 37, T, D]
            for i, r in enumerate(rs):
                off = width - len(r['tokens'])
                assert int(batch['input_ids'][i, off]) == r['tokens'][0]
                idx = [off + r['positions'][p] for p in POSITIONS]
                hidden[f * 8 + i] = hs[i][:, idx].contiguous().view(torch.uint16).cpu().numpy()
                post[f * 8 + i] = torch.stack([grabbed[b][i] for b in range(36)]).contiguous().view(torch.uint16).cpu().numpy()
                lg = out.logits[i, -1].float()
                v = [float(lg[j].max()) for j in ids]
                behaviour.append({'id': r['id'], 'scores': v, 'pred': int(np.argmax(v)), 'gold': B.CLASSES.index(r['gold']),
                                  'top1': int(lg.argmax())})
            if norm_check is None:
                raw, final = last['h'].float(), hs[:, 36, -1].float()
                norm_check = {'hs36_equals_last_block_output': bool(torch.equal(raw, final)),
                              'hs36_equals_final_norm_of_last_block': bool(torch.allclose(m.model.norm(last['h']).float(), final, atol=1e-2, rtol=1e-2))}
            if (f + 1) % 20 == 0 or f + 1 == fams:
                hidden.flush(); post.flush()
                B.log(f'capture-{split}', f + 1, fams, start)
        for h in hooks:
            h.remove()
        peak = round(torch.cuda.max_memory_allocated() / 2 ** 20)
    del hidden, post
    B.save(f'stagec-behaviour-{split}.json', behaviour)
    manifest = {'schema': 'common_bench_native.v1', 'run_id': RUN, 'split': split, 'items': n, 'positions': POSITIONS,
                'encoding': 'BF16 bit patterns stored as uint16; hidden [item, 37 states, 3 positions, 2560]; '
                            'postattn [item, 36 blocks, 2560] = input of post_attention_layernorm at answer_prefix',
                'files': {k: {'sha256': B.sha(folder / k), 'bytes': (folder / k).stat().st_size} for k in ('hidden.npy', 'postattn.npy')},
                'materials_sha256': plan['materials'][split]['sha256'], 'model_snapshot': plan['model_snapshot'],
                'behaviour_sha256': B.sha(OUT / f'stagec-behaviour-{split}.json'), 'boundary_check': norm_check,
                'peak_MiB': peak, 'elapsed_seconds': round(time.monotonic() - start, 1),
                'purpose': 'reusable for stages C-E', 'review_by': '2026-12-10'}
    (folder / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    B.save(f'stagec-native-{split}.json', manifest)
    acc = float(np.mean([b['pred'] == b['gold'] for b in behaviour]))
    print(json.dumps({'captured': split, 'items': n, 'forced_accuracy': acc, 'boundary_check': norm_check}), flush=True)


# ---------- analysis helpers (CPU) ----------

def labels(split):
    rows = B.read(OUT / f'{split}-materials.json')
    beh = B.read(OUT / f'stagec-behaviour-{split}.json')
    assert [r['id'] for r in rows] == [b['id'] for b in beh]
    names = sorted({e for r in rows for e in r['entities']})
    lab = {'S': np.array([r['source'] for r in rows]), 'L': np.array([r['language'] for r in rows]),
           'q': np.array([r['query'] for r in rows]), 'C': np.array([B.COLORS.index(r['gold_color']) for r in rows]),
           'family': np.array([r['family'] for r in rows]), 'frame': np.array([r['frame'] for r in rows]),
           'e1': np.array([names.index(r['entities'][0]) for r in rows]), 'e2': np.array([names.index(r['entities'][1]) for r in rows]),
           'target_entity': np.array([names.index(r['entities'][0 if r['query'] == 1 else 1]) if r['source'] == -1 else -1 for r in rows]),
           'correct': np.array([b['pred'] == b['gold'] for b in beh]), 'entity_names': names,
           'entity_color': {n: c for r in rows for n, c in zip(r['entities'], r['parametric_colors'])}}
    reserved = np.zeros(len(rows), bool)
    for c in B.PREREG['combo_holdout']['cells']:
        reserved |= (lab['S'] == (1 if c['source'] == 'document' else -1)) & (lab['L'] == (1 if c['language'] == 'en' else -1)) \
            & (lab['C'] == B.COLORS.index(c['answer']))
    lab['reserved'] = reserved
    # stage B behaviour rule, step 1: entity screen on parametric items
    bad = [e for e in range(len(names)) if (lab['target_entity'] == e).any() and lab['correct'][lab['target_entity'] == e].mean() < 0.75]
    lab['dropped_entities'] = [names[e] for e in bad]
    lab['screened'] = ~(np.isin(lab['e1'], bad) | np.isin(lab['e2'], bad))
    return lab


def layer(split, l, pos=2):
    h = np.load(store(split) / 'hidden.npy', mmap_mode='r')
    return bits_to_f32(np.ascontiguousarray(h[:, l, pos]))


def design(S, L, C, pairwise=True):
    e = np.zeros((len(S), 4))
    for k in range(4):
        e[C == k, k] = 1
    e[C == 4] = -1
    cols = [np.ones(len(S)), S, L, *e.T]
    if pairwise:
        cols += [S * L, *(S[:, None] * e).T, *(L[:, None] * e).T]
    return np.stack(cols, 1).astype(np.float64)


CELLS = [(s, l, c) for s in (1, -1) for l in (1, -1) for c in range(5)]
CELL_X = {pw: design(np.array([c[0] for c in CELLS]), np.array([c[1] for c in CELLS]), np.array([c[2] for c in CELLS]), pw) for pw in (True, False)}


def cell_index(S, L, C):
    return (S == -1) * 10 + (L == -1) * 5 + C


def nearest(h, centroids, var):
    w = 1 / np.maximum(var, 1e-6)
    d = (h ** 2 * w).sum(1)[:, None] - 2 * (h * w) @ centroids.T + (centroids ** 2 * w).sum(1)[None]
    return d.argmin(1)


def binary_transfer(h, y, cond, train, test, var):
    """Language (y = +-1) readout fitted in condition c, tested in the other condition; returns {direction: (M1, M2, M3)}."""
    out = {}
    for c in (1, -1):
        def fit(mask):
            mp, mn = h[mask & (y == 1)].mean(0), h[mask & (y == -1)].mean(0)
            return (mp + mn) / 2, (mp - mn) / np.maximum(var, 1e-6)
        cen_tr, w_tr = fit(train & (cond == c))
        cen_tg, w_tg = fit(train & (cond == -c))
        cen_tg_plain = h[train & (cond == -c)].mean(0)
        sel = test & (cond == -c)
        acc = lambda cen, w: float((((h[sel] - cen) @ w >= 0) == (y[sel] == 1)).mean())
        out[c] = (acc(cen_tr, w_tr), acc(cen_tg_plain, w_tr), acc(cen_tg, w_tg))
    return out


def colour_transfer(h, C, cond, train, test, var):
    out = {}
    for c in (1, -1):
        cent = lambda mask: np.stack([h[mask & (C == k)].mean(0) for k in range(5)])
        tr, tg = cent(train & (cond == c)), cent(train & (cond == -c))
        shift = h[train & (cond == -c)].mean(0) - tr.mean(0)
        sel = test & (cond == -c)
        acc = lambda cen: float((nearest(h[sel], cen, var) == C[sel]).mean())
        out[c] = (acc(tr), acc(tr + shift), acc(tg))
    return out


def onset(values):
    for l in range(1, 34):
        if all(v >= 0.95 for v in values[l:l + 3]):
            return l
    return int(1 + np.argmax(values[1:36]))


def fit():
    checked()
    assert not (OUT / 'stagec-frozen.json').exists()
    lab = labels('discovery')
    cells_ok = {}
    for s in (1, -1):
        for l in (1, -1):
            m = lab['screened'] & (lab['S'] == s) & (lab['L'] == l)
            cells_ok[f"{'document' if s == 1 else 'parametric'}_{'en' if l == 1 else 'zh'}"] = float(lab['correct'][m].mean())
    frames_ok = {int(f): float(lab['correct'][lab['screened'] & (lab['frame'] == f)].mean()) for f in np.unique(lab['frame'])}
    n_ent = len(lab['entity_names'])
    gate = len(lab['dropped_entities']) <= 0.2 * n_ent and min(cells_ok.values()) >= 0.9 and min(frames_ok.values()) >= 0.9
    behaviour = {'items': int(len(lab['S'])), 'forced_all': float(lab['correct'].mean()), 'dropped_entities': lab['dropped_entities'],
                 'cells_after_screen': cells_ok, 'min_frame_after_screen': min(frames_ok.values()), 'gate_passed': bool(gate)}
    print(json.dumps(behaviour, ensure_ascii=False), flush=True)
    assert gate, 'discovery behaviour gate failed on the full split; stop'
    fitset = lab['screened'] & lab['correct'] & ~lab['reserved']
    S, L, C = lab['S'], lab['L'], lab['C']
    names, colour_of = lab['entity_names'], lab['entity_color']
    by_colour = {c: [i for i, n in enumerate(names) if colour_of[n] == c] for c in B.COLORS}
    folds = []
    for k in range(4):
        held = [by_colour[c][k] for c in B.COLORS]
        test = np.isin(lab['e1'], held) & np.isin(lab['e2'], held) & fitset
        train = ~np.isin(lab['e1'], held) & ~np.isin(lab['e2'], held) & fitset
        folds.append((train, test))
    # pseudo-reserved cells for a discovery-internal rehearsal of P3 (never the real reserved cells)
    pseudo = ((S == 1) & (L == 1) & (C == 0)) | ((S == -1) & (L == -1) & (C == 4))
    start = time.monotonic()
    curves = {k: np.zeros((NL, 2, 3)) for k in ('P1_language_across_source', 'P2a_colour_across_language', 'P2b_colour_across_source')}
    controls = {p: {f: np.zeros(NL) for f in ('S', 'L', 'C')} for p in ('doc_end', 'question_end')}
    rehearsal = {m: np.zeros((NL, 2)) for m in ('A1', 'A2')}
    rehearsal_n = np.zeros(2)
    coef = {pw: np.zeros((NL, CELL_X[pw].shape[1], D), np.float32) for pw in (True, False)}
    resvar, totvar = np.zeros((NL, D), np.float32), np.zeros((NL, D), np.float32)
    inc = np.zeros((NL - 1, CELL_X[True].shape[1], D), np.float32)
    mean_inc = np.zeros((NL - 1, D), np.float32)
    prev = None
    for l in range(NL):
        h = layer('discovery', l).astype(np.float64)
        for train, test in folds:
            var = h[train].var(0)
            for name, res in (('P1_language_across_source', binary_transfer(h, L, S, train, test, var)),
                              ('P2a_colour_across_language', colour_transfer(h, C, L, train, test, var)),
                              ('P2b_colour_across_source', colour_transfer(h, C, S, train, test, var))):
                curves[name][l] += np.array([res[1], res[-1]]) / len(folds)
            for mi, pw in (('A1', False), ('A2', True)):
                tr = train & ~pseudo
                beta = np.linalg.lstsq(design(S[tr], L[tr], C[tr], pw), h[tr], rcond=None)[0]
                rv = (h[tr] - design(S[tr], L[tr], C[tr], pw) @ beta).var(0)
                for j, cellmask in enumerate(((S == 1) & (L == 1) & (C == 0), (S == -1) & (L == -1) & (C == 4))):
                    sel = test & cellmask
                    rehearsal[mi][l, j] += float((nearest(h[sel], CELL_X[pw] @ beta, rv) == cell_index(S, L, C)[sel]).sum())
                    if l == 0 and mi == 'A1':
                        rehearsal_n[j] += sel.sum()
        for pos_i, pos in enumerate(('doc_end', 'question_end')):
            hp = layer('discovery', l, pos_i).astype(np.float64)
            for fac, y in (('S', S), ('L', L)):
                a = []
                for train, test in folds:
                    var = hp[train].var(0)
                    mp, mn = hp[train & (y == 1)].mean(0), hp[train & (y == -1)].mean(0)
                    a.append((((hp[test] - (mp + mn) / 2) @ ((mp - mn) / np.maximum(var, 1e-6)) >= 0) == (y[test] == 1)).mean())
                controls[pos][fac][l] = float(np.mean(a))
            a = []
            for train, test in folds:
                cen = np.stack([hp[train & (C == k)].mean(0) for k in range(5)])
                a.append((nearest(hp[test], cen, hp[train].var(0)) == C[test]).mean())
            controls[pos]['C'][l] = float(np.mean(a))
        for pw in (True, False):
            X = design(S[fitset], L[fitset], C[fitset], pw)
            beta = np.linalg.lstsq(X, h[fitset], rcond=None)[0]
            coef[pw][l] = beta
            if pw:
                resvar[l] = (h[fitset] - X @ beta).var(0)
        totvar[l] = h[fitset].var(0)
        if prev is not None and l <= 35:
            delta = h[fitset] - prev[fitset]
            inc[l - 1] = np.linalg.lstsq(design(S[fitset], L[fitset], C[fitset], True), delta, rcond=None)[0]
            mean_inc[l - 1] = delta.mean(0)
        prev = h
        B.log('fit', l + 1, NL, start)
    for mi in rehearsal:
        rehearsal[mi] /= rehearsal_n
    sel = {k: onset(v[:, :, 1].min(1)) for k, v in curves.items()}
    readouts = {}
    for name, l in sel.items():
        h = layer('discovery', l).astype(np.float64)
        y, cond = (L, S) if name.startswith('P1') else (C, L if name.startswith('P2a') else S)
        entry = {'layer': l, 'var': h[fitset].var(0).astype(np.float32)}
        for c in (1, -1):
            m = fitset & (cond == c)
            if name.startswith('P1'):
                entry[f'mean_pos_{c}'], entry[f'mean_neg_{c}'] = h[m & (y == 1)].mean(0).astype(np.float32), h[m & (y == -1)].mean(0).astype(np.float32)
            else:
                entry[f'centroids_{c}'] = np.stack([h[m & (y == k)].mean(0) for k in range(5)]).astype(np.float32)
            entry[f'centre_{c}'] = h[m].mean(0).astype(np.float32)
        readouts[name] = entry
    flat = {f'{n}__{k}': v for n, e in readouts.items() for k, v in e.items() if k != 'layer'}
    np.savez(OUT / 'stagec-estimators.npz', coef_A2=coef[True], coef_A1=coef[False], residual_variance=resvar,
             total_variance=totvar, increment_A2=inc, mean_increment=mean_inc, **flat)
    r3 = lambda a: [round(float(x), 4) for x in a]
    frozen = {'schema': 'common_bench_stagec_frozen.v1', 'run_id': RUN, 'prereg_sha256': B.sha(OUT / 'stagec-prereg.json'),
              'estimators_sha256': B.sha(OUT / 'stagec-estimators.npz'), 'created_at': datetime.now().astimezone().isoformat(),
              'native_manifest_sha256': B.sha(OUT / 'stagec-native-discovery.json'), 'behaviour': behaviour,
              'fit_items': int(fitset.sum()), 'reserved_items_excluded': int((lab['reserved'] & lab['screened']).sum()),
              'folds': [{'train': int(a.sum()), 'test': int(b.sum())} for a, b in folds],
              'selected_layers': sel, 'p3_layer': 34,
              'discovery_cv': {k: {'direction_plus_M1_M2_M3_at_selected': r3(v[sel[k], 0]), 'direction_minus_M1_M2_M3_at_selected': r3(v[sel[k], 1]),
                                   'M1_min_by_layer': r3(v[:, :, 0].min(1)), 'M2_min_by_layer': r3(v[:, :, 1].min(1)),
                                   'M3_min_by_layer': r3(v[:, :, 2].min(1))} for k, v in curves.items()},
              'direction_key': {'P1': 'plus = fitted on document, tested on parametric', 'P2a': 'plus = fitted on English output, tested on Chinese output',
                                'P2b': 'plus = fitted on document, tested on parametric'},
              'p3_rehearsal_pseudo_cells': {'cells': ['document x en x red', 'parametric x zh x black'], 'test_items': [int(x) for x in rehearsal_n],
                                            'A1_by_layer': [r3(x) for x in rehearsal['A1']], 'A2_by_layer': [r3(x) for x in rehearsal['A2']],
                                            'at_layer_34': {'A1': r3(rehearsal['A1'][34]), 'A2': r3(rehearsal['A2'][34])}},
              'position_controls': {p: {f: r3(v) for f, v in d.items()} for p, d in controls.items()},
              'note': 'all numbers here come from discovery cross-validation; none is a confirmation'}
    B.save('stagec-frozen.json', frozen)
    print(json.dumps({'frozen': B.sha(OUT / 'stagec-frozen.json'), 'selected_layers': sel,
                      'cv_at_selected': {k: frozen['discovery_cv'][k]['direction_plus_M1_M2_M3_at_selected'] for k in curves},
                      'p3_rehearsal_34': frozen['p3_rehearsal_pseudo_cells']['at_layer_34']}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['prereg', 'capture', 'fit'])
    ap.add_argument('--split', choices=['discovery', 'confirm'])
    a = ap.parse_args()
    capture(a.split) if a.stage == 'capture' else globals()[a.stage]()
