"""Stage D (Phase 384): predict first, then intervene in the original model.  Benchmark COMMON-BENCH-B1-20261110-v3.

Claim under test: the colour class centres read out in stage C are used by the model.  At the answer position the
residual hs[32] is shifted by alpha * (centre[k'] - centre[k]); the outcome categories were predicted before any run.
Stages: prereg -> vectors (CPU, discovery) -> pilot (GPU, discovery) -> freeze -> run --split lockbox|lockbox_frame --unseal -> report.
"""
import os
os.environ['HF_DEACTIVATE_ASYNC_LOAD'] = '1'
import argparse, json, sys, time
from datetime import datetime
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_bench_b1 as B
import common_bench_b1_stage_c as C

OUT, RUN, SEED = B.OUT, 'COMMON-BENCH-B1-STAGE-D-20261110-v1', 202611100390
LAYER, GRID, PILOT_FAMILIES = 32, [0.5, 1.0, 1.5, 2.0], 64
STEER = ['pooled', 'cross_language', 'cross_source', 'random', 'wrong_donor', 'direct_pooled', 'direct_cross_language']
DIRECT = {'direct_pooled', 'direct_cross_language'}
CRITERIA = {
    'intervention': 'answer position only; add delta to hs[32] (output of block 31) and recompute blocks 32-35; '
                    'delta = alpha * (centre[target colour] - centre[original colour]); centres are discovery class means '
                    '(fit set of stage C, reserved cells excluded)',
    'arms': {'pooled': 'centres from all fit items',
             'cross_language': 'centres only from items whose requested output language differs from the item',
             'cross_source': 'centres only from items whose answer source differs from the item',
             'random': 'Gaussian direction with the norm of the pooled delta',
             'wrong_donor': 'pooled centre difference between two colours that are neither the original nor the target',
             'direct_pooled': 'pooled delta added after the last block (before the final norm): direct write path only',
             'direct_cross_language': 'cross_language delta added after the last block: direct write path only (descriptive)',
             'donor_q': 'replace hs[32] by the state of the same family, same source and language, other queried slot',
             'donor_L': 'replace hs[32] by the state of the same family, same source and slot, other output language'},
    'target_colour': 'drawn per item from the four other colours with a fixed seed',
    'outcome': '10-way first-token forced choice; success = target colour in the requested language; '
               'other_language_target = target colour in the other language; unchanged = original answer',
    'primary_set': 'items the unmodified model answers correctly, after the stage B entity screen',
    'alpha_rule': 'on the discovery pilot: smallest alpha in [0.5, 1, 1.5, 2] with pooled success >= 0.80; else 2.0 (flagged)',
    'predictions_on_lockbox': {
        'D1_pooled': {'supported': 'success >= 0.80', 'not_supported': 'success < 0.50'},
        'D2_cross_language': {'supported': 'success >= 0.60 and other_language_target <= 0.10', 'not_supported': 'success < 0.30'},
        'D3_cross_source': {'supported': 'success >= 0.60', 'not_supported': 'success < 0.30'},
        'controls_valid': 'random: success <= 0.05 and unchanged >= 0.90; wrong_donor: success <= 0.10',
        'side_effect': 'pooled arm keeps the requested language in >= 0.95 of items',
        'donor_q': 'predicted outcome = donor colour in the own language (descriptive upper bound)',
        'donor_L': 'predicted outcome = own colour in the donor language (descriptive)',
        'direct_vs_total': 'descriptive: direct_pooled and direct_cross_language success beside the recomputed arms'},
    'disclosure': 'before this file was frozen, the code was smoke-tested on two discovery families (16 items); '
                  'the thresholds above were already written and were not changed afterwards',
    'verdict_rule': 'D1-D3 are judged only if the controls are valid; otherwise specificity is not established',
    'not_covered': 'capability on unrelated tasks, free generation after intervention, other positions and layers',
}


def cluster_ci(hit, e1, e2, n_ent, rng, reps=2000):
    hit = np.asarray(hit, float)
    if len(hit) == 0:
        return {'n': 0, 'rate': None, 'ci95': None}
    vals = []
    for _ in range(reps):
        m = rng.multinomial(n_ent, np.ones(n_ent) / n_ent)
        w = m[e1] * m[e2]
        if w.sum():
            vals.append((hit * w).sum() / w.sum())
    return {'n': int(len(hit)), 'rate': round(float(hit.mean()), 4),
            'ci95': [round(float(np.quantile(vals, .025)), 4), round(float(np.quantile(vals, .975)), 4)]}


def prereg():
    from tools import provenance
    from tools.power_analysis import check_phase
    assert not (OUT / 'staged-prereg.json').exists()
    plan = B.read(OUT / 'plan.json')
    n = plan['materials']['lockbox']['rows']
    body = {'schema': 'common_bench_staged_prereg.v1', 'run_id': RUN, 'seed': SEED, 'layer': LAYER, 'alpha_grid': GRID,
            'benchmark_plan_sha256': B.sha(OUT / 'plan.json'), 'stagec_frozen_sha256': B.sha(OUT / 'stagec-frozen.json'),
            'model_snapshot': plan['model_snapshot'], 'code_sha256': B.sha(__file__), 'criteria': CRITERIA,
            'created_at': datetime.now().astimezone().isoformat(),
            'power': {'unit': 'paired binary outcome per item against the unmodified run', 'expected_delta': 0.10,
                      'n_lockbox': n, **check_phase(0.10, n), 'per_condition_half': check_phase(0.10, n // 2),
                      'caveat': 'lockbox has 15 entities and 90 families per split; entity-resampling intervals accompany every rate'},
            'custody': 'discovery is used for the centres and the alpha choice; lockbox and lockbox_frame are opened once, '
                       'only with --unseal after staged-frozen.json exists; confirm is not used (opened in stage C)',
            'provenance': provenance.provenance(model_path=B.MODEL, seed=SEED, tool=RUN)}
    body['provenance']['model_revision'] = plan['model_snapshot']
    B.save('staged-prereg.json', body)
    print(json.dumps({'prereg_sha256': B.sha(OUT / 'staged-prereg.json'), 'power': body['power']}, ensure_ascii=False), flush=True)


def checked():
    p = B.read(OUT / 'staged-prereg.json')
    assert p['code_sha256'] == B.sha(__file__), 'stage D source changed after pre-registration'
    assert p['stagec_frozen_sha256'] == B.sha(OUT / 'stagec-frozen.json')
    return p


def vectors():
    checked()
    assert not (OUT / 'staged-centres.npz').exists()
    lab = C.labels('discovery')
    fit = lab['screened'] & lab['correct'] & ~lab['reserved']
    out = {}
    for layer in (27, LAYER, 35):
        h = C.layer('discovery', layer).astype(np.float64)
        conds = {'pooled': fit, 'L1': fit & (lab['L'] == 1), 'L-1': fit & (lab['L'] == -1),
                 'S1': fit & (lab['S'] == 1), 'S-1': fit & (lab['S'] == -1)}
        for name, m in conds.items():
            out[f'{layer}__{name}'] = np.stack([h[m & (lab['C'] == k)].mean(0) for k in range(5)]).astype(np.float32)
    np.savez(OUT / 'staged-centres.npz', **out)
    pooled = out[f'{LAYER}__pooled'].astype(np.float64)
    norms = [float(np.linalg.norm(pooled[a] - pooled[b])) for a in range(5) for b in range(a)]
    h = C.layer('discovery', LAYER).astype(np.float64)[fit]
    print(json.dumps({'centres_sha256': B.sha(OUT / 'staged-centres.npz'), 'fit_items': int(fit.sum()),
                      'pooled_difference_norm_min_max': [round(min(norms), 2), round(max(norms), 2)],
                      'mean_state_norm': round(float(np.linalg.norm(h, axis=1).mean()), 2)}), flush=True)


def plan_item(r, cent, alpha):
    """Deterministic targets and deltas for one item."""
    rng = np.random.default_rng([SEED, int.from_bytes(r['id'].encode()[-12:], 'little') % (2 ** 32)])
    k = B.COLORS.index(r['gold_color'])
    others = [c for c in range(5) if c != k]
    kt = int(rng.choice(others))
    a, b = [c for c in others if c != kt][:2]
    pooled = cent['pooled']
    d = {'pooled': pooled[kt] - pooled[k], 'cross_language': cent[f"L{-r['language']}"][kt] - cent[f"L{-r['language']}"][k],
         'cross_source': cent[f"S{-r['source']}"][kt] - cent[f"S{-r['source']}"][k], 'wrong_donor': pooled[b] - pooled[a]}
    g = rng.standard_normal(pooled.shape[1])
    d['random'] = g / np.linalg.norm(g) * np.linalg.norm(d['pooled'])
    d['direct_pooled'], d['direct_cross_language'] = d['pooled'], d['cross_language']
    return k, kt, {name: alpha * v for name, v in d.items()}


def execute(split, families, alphas, extra_layers=()):
    import torch
    from transformers import AutoTokenizer
    from shared.dist_protocol.mechanism_trace import GpuLease
    import mlp_condition_design as MD, mlp_condition_run as MR, mlp_condition_stream_load as SL
    plan = B.read(OUT / 'plan.json')
    assert B.sha(OUT / f'{split}-materials.json') == plan['materials'][split]['sha256']
    rows = [r for r in B.read(OUT / f'{split}-materials.json') if r['family'] in families]
    z = np.load(OUT / 'staged-centres.npz')
    cents = {layer: {k.split('__')[1]: z[k].astype(np.float64) for k in z.files if k.startswith(f'{layer}__')} for layer in (27, LAYER, 35)}
    ids = [plan['choice_ids'][k] for k in B.CLASSES]
    tok = AutoTokenizer.from_pretrained(B.MODEL, local_files_only=True)
    tok.padding_side = 'left'
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    start, items = time.monotonic(), []
    with GpuLease(), torch.inference_mode():
        MD.checked = lambda: plan
        MR.save = lambda name, body, _: None
        m = SL.load(B.MODEL, dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa').eval().requires_grad_(False)
        state = {'mode': None, 'block': None, 'delta': None, 'saved': None}

        def hook(block):
            def fn(mod, args, out):
                if state['block'] != block or state['mode'] is None:
                    return None
                h = out[0] if isinstance(out, tuple) else out
                if state['mode'] == 'save':
                    state['saved'] = h[:, -1].clone()
                    return None
                h = h.clone()
                if state['mode'] == 'add':
                    h[:, -1] = (h[:, -1].float() + state['delta']).to(h.dtype)
                else:
                    h[:, -1] = state['delta'].to(h.dtype)
                return (h, *out[1:]) if isinstance(out, tuple) else h
            return fn
        handles = [layer.register_forward_hook(hook(b)) for b, layer in enumerate(m.model.layers)]

        def run(batch, mode=None, block=None, delta=None):
            state.update(mode=mode, block=block, delta=delta)
            lg = m(**batch, use_cache=False, logits_to_keep=1).logits[:, 0].float()
            state['mode'] = None
            v = torch.stack([lg[:, j].max(1).values for j in ids], 1)
            return v.argmax(1).cpu().numpy(), lg.argmax(1).cpu().numpy(), v.cpu().numpy()

        fams = sorted(families)
        B.log(f'intervene-{split}', 0, len(fams), start)
        for fi, f in enumerate(fams):
            rs = [r for r in rows if r['family'] == f]
            batch = tok.pad({'input_ids': [r['tokens'] for r in rs]}, return_tensors='pt').to('cuda')
            pred0, top0, sc0 = run(batch, 'save', LAYER - 1)
            saved = state['saved']
            rec = [{'id': r['id'], 'family': f, 'frame': r['frame'], 'source': r['source'], 'language': r['language'],
                    'query': r['query'], 'entities': r['entities'], 'gold': B.CLASSES.index(r['gold']), 'base': int(p), 'arms': {}}
                   for r, p in zip(rs, pred0)]
            for name, perm in (('donor_q', [i ^ 2 for i in range(8)]), ('donor_L', [i ^ 1 for i in range(8)])):
                assert all(rs[i]['source'] == rs[j]['source'] for i, j in enumerate(perm))
                pred, top, _ = run(batch, 'set', LAYER - 1, saved[perm])
                for i, x in enumerate(rec):
                    x['arms'][name] = {'pred': int(pred[i]), 'donor_gold': B.CLASSES.index(rs[perm[i]]['gold']),
                                       'top1_in_pred_class': bool(int(top[i]) in ids[int(pred[i])])}
            for layer, alist in [(LAYER, alphas)] + [(l, [1.0]) for l in extra_layers]:
                for alpha in alist:
                    plans = [plan_item(r, cents[layer], alpha) for r in rs]
                    for arm in (STEER if layer == LAYER else ['pooled']):
                        delta = torch.tensor(np.stack([p[2][arm] for p in plans]), device='cuda', dtype=torch.float32)
                        pred, top, sc = run(batch, 'add', 35 if arm in DIRECT else layer - 1, delta)
                        for i, x in enumerate(rec):
                            k, kt, _ = plans[i]
                            lang = 0 if rs[i]['language'] == 1 else 5
                            key = f'{arm}@{alpha}' if layer == LAYER else f'{arm}@L{layer}'
                            x['target_color'] = kt
                            x['arms'][key] = {'pred': int(pred[i]), 'success': bool(pred[i] == lang + kt),
                                              'other_language_target': bool(pred[i] == (5 - lang) + kt),
                                              'unchanged': bool(pred[i] == x['gold']), 'language_kept': bool(pred[i] // 5 == lang // 5),
                                              'top1_success': bool(int(top[i]) in ids[lang + kt]),
                                              'margin_target_minus_original': round(float(sc[i, lang + kt] - sc[i, lang + k]), 3),
                                              'base_margin_target_minus_original': round(float(sc0[i, lang + kt] - sc0[i, lang + k]), 3)}
            items += rec
            if (fi + 1) % 8 == 0 or fi + 1 == len(fams):
                B.log(f'intervene-{split}', fi + 1, len(fams), start)
        for h in handles:
            h.remove()
        peak = round(torch.cuda.max_memory_allocated() / 2 ** 20)
    return items, peak, round(time.monotonic() - start, 1)


def summarise(items, split, keys):
    """Rates on the primary set with entity-resampling intervals."""
    rows = B.read(OUT / f'{split}-materials.json')
    names = sorted({e for r in rows for e in r['entities']})
    rng = np.random.default_rng(SEED)
    # entity screen: parametric base accuracy per target entity
    acc = {}
    for x in items:
        if x['source'] == -1:
            acc.setdefault(x['entities'][0 if x['query'] == 1 else 1], []).append(x['base'] == x['gold'])
    dropped = [e for e, v in acc.items() if np.mean(v) < 0.75]
    prim = [x for x in items if x['base'] == x['gold'] and not set(x['entities']) & set(dropped)]
    e1 = np.array([names.index(x['entities'][0]) for x in prim])
    e2 = np.array([names.index(x['entities'][1]) for x in prim])
    ci = lambda v, mask=None: cluster_ci(np.array(v)[mask] if mask is not None else v, e1[mask] if mask is not None else e1,
                                         e2[mask] if mask is not None else e2, len(names), rng)
    out = {'items': len(items), 'base_forced_accuracy': round(float(np.mean([x['base'] == x['gold'] for x in items])), 4),
           'dropped_entities': dropped, 'primary_items': len(prim), 'arms': {}}
    for key in keys:
        a = [x['arms'][key] for x in prim]
        out['arms'][key] = {f: ci([y[f] for y in a]) for f in ('success', 'other_language_target', 'unchanged', 'language_kept', 'top1_success')}
        out['arms'][key]['mean_margin_shift'] = round(float(np.mean([y['margin_target_minus_original'] - y['base_margin_target_minus_original'] for y in a])), 3)
        for cond, field in (('by_language', 'language'), ('by_source', 'source')):
            out['arms'][key][cond] = {str(v): ci([y['success'] for y in a], np.array([x[field] == v for x in prim])) for v in (1, -1)}
    dq = [x['arms']['donor_q'] for x in prim]
    dl = [x['arms']['donor_L'] for x in prim]
    out['donor_q'] = {'takes_donor_colour_own_language': ci([y['pred'] == (g // 5) * 5 + y['donor_gold'] % 5 for y, g in zip(dq, [x['gold'] for x in prim])]),
                      'unchanged': ci([y['pred'] == x['gold'] for y, x in zip(dq, prim)])}
    out['donor_L'] = {'own_colour_donor_language': ci([y['pred'] == y['donor_gold'] for y in dl]),
                      'unchanged': ci([y['pred'] == x['gold'] for y, x in zip(dl, prim)])}
    return out


def pilot():
    checked()
    assert not (OUT / 'staged-pilot.json').exists()
    fams = set(range(PILOT_FAMILIES))
    items, peak, sec = execute('discovery', fams, GRID, extra_layers=(27, 35))
    B.save('staged-pilot-items.json', items)
    keys = [f'{a}@{al}' for al in GRID for a in STEER] + ['pooled@L27', 'pooled@L35']
    s = summarise(items, 'discovery', keys)
    alpha = next((al for al in GRID if s['arms'][f'pooled@{al}']['success']['rate'] >= 0.80), None)
    B.save('staged-pilot.json', {'schema': 'common_bench_staged_pilot.v1', 'run_id': RUN, 'scope': 'discovery only (also the source of the centres)',
                                 'families': len(fams), 'summary': s, 'alpha_selected': alpha if alpha is not None else GRID[-1],
                                 'alpha_rule_met': alpha is not None, 'peak_MiB': peak, 'elapsed_seconds': sec,
                                 'items_sha256': B.sha(OUT / 'staged-pilot-items.json')})
    print(json.dumps({'alpha_selected': alpha, 'pooled_success': {al: s['arms'][f'pooled@{al}']['success']['rate'] for al in GRID}}), flush=True)


def freeze():
    checked()
    assert not (OUT / 'staged-frozen.json').exists()
    p = B.read(OUT / 'staged-pilot.json')
    B.save('staged-frozen.json', {'schema': 'common_bench_staged_frozen.v1', 'run_id': RUN, 'layer': LAYER, 'alpha': p['alpha_selected'],
                                  'alpha_rule_met': p['alpha_rule_met'], 'prereg_sha256': B.sha(OUT / 'staged-prereg.json'),
                                  'centres_sha256': B.sha(OUT / 'staged-centres.npz'), 'pilot_sha256': B.sha(OUT / 'staged-pilot.json'),
                                  'created_at': datetime.now().astimezone().isoformat()})
    print(json.dumps({'frozen_sha256': B.sha(OUT / 'staged-frozen.json'), 'alpha': p['alpha_selected']}), flush=True)


def run_split(split, unseal):
    from tools import provenance
    pre, fz = checked(), B.read(OUT / 'staged-frozen.json')
    assert unseal and split in ('lockbox', 'lockbox_frame'), 'lockbox splits need --unseal'
    assert fz['centres_sha256'] == B.sha(OUT / 'staged-centres.npz')
    assert not (OUT / f'staged-{split}-report.json').exists(), 'already opened'
    rows = B.read(OUT / f'{split}-materials.json')
    items, peak, sec = execute(split, {r['family'] for r in rows}, [fz['alpha']])
    B.save(f'staged-{split}-items.json', items)
    al = fz['alpha']
    s = summarise(items, split, [f'{a}@{al}' for a in STEER])
    r = lambda arm, f: s['arms'][f'{arm}@{al}'][f]['rate']
    controls = r('random', 'success') <= 0.05 and r('random', 'unchanged') >= 0.90 and r('wrong_donor', 'success') <= 0.10
    tri = lambda ok, bad: 'supported' if ok else 'not_supported' if bad else 'inconclusive'
    verdict = {'controls_valid': bool(controls),
               'D1_pooled': tri(r('pooled', 'success') >= .80, r('pooled', 'success') < .50),
               'D2_cross_language': tri(r('cross_language', 'success') >= .60 and r('cross_language', 'other_language_target') <= .10,
                                        r('cross_language', 'success') < .30),
               'D3_cross_source': tri(r('cross_source', 'success') >= .60, r('cross_source', 'success') < .30),
               'side_effect_language_kept': bool(r('pooled', 'language_kept') >= .95),
               'note': None if controls else 'controls not valid: specificity is not established, D1-D3 are descriptive only'}
    report = {'schema': 'common_bench_staged_report.v1', 'run_id': RUN, 'split': split, 'layer': LAYER, 'alpha': al,
              'prereg_sha256': B.sha(OUT / 'staged-prereg.json'), 'frozen_sha256': B.sha(OUT / 'staged-frozen.json'),
              'items_sha256': B.sha(OUT / f'staged-{split}-items.json'), 'criteria': pre['criteria'], 'verdicts': verdict,
              'summary': s, 'peak_MiB': peak, 'elapsed_seconds': sec, 'formal_evidence_promotion': 0,
              'custody': f'{split} opened once for this report',
              'provenance': provenance.provenance(model_path=B.MODEL, seed=SEED, tool=RUN)}
    report['provenance']['model_revision'] = pre['model_snapshot']
    B.save(f'staged-{split}-report.json', report)
    path = provenance.write_result(f'COMMON-BENCH-B1-stageD-{split}', {'verdicts': verdict, 'report_sha256': B.sha(OUT / f'staged-{split}-report.json')},
                                   model_path=B.MODEL, seed=SEED, tool=RUN)
    print(json.dumps({'split': split, 'verdicts': verdict, 'result': str(path)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['prereg', 'vectors', 'pilot', 'freeze', 'run'])
    ap.add_argument('--split', choices=['lockbox', 'lockbox_frame'])
    ap.add_argument('--unseal', action='store_true')
    a = ap.parse_args()
    run_split(a.split, a.unseal) if a.stage == 'run' else globals()[a.stage]()
