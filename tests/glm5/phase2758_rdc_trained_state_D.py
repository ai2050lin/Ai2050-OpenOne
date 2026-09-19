"""Phase 2758: trained-state retest of the 2757 dual-world discrimination panel.

Question (2757 接续): 2757 showed on native qwen3-4b that (a) dual-world
discrimination D is saturated negative (median -22.31, 98% of pairs D<=0) and
(b) block12 targeted source deletion is NOT behaviourally specific (J2 reversed).
Both conclusions were declared non-transferable to the trained state. This phase
deploys all six 2747 trained BF16 checkpoints (3 conditions x 2 seeds) and
repeats the 2757 panel verbatim: 160 pairs x 2 worlds, 13 variants (base,
sema/randn/rand x 25/50/100% block12 v-row zeroing), single-pass hook scoring.

Deployment: block16 mlp weights rebuilt exactly as 2747 did
(original_native_FP32 + delta + reconstruction_residual, FP32) and copied into
the BF16 parameters (identical to the 2747 FP32->BF16 deployment cast).

Preregistered criteria (frozen in execution.json before any behavioural forward):
  C1 (primary, per run): mechanism_localised iff J1 and J2, same definitions as
      2757 (J1: bands with dD_sema > dD_randn >= 2 of 3; J2: k=100% pair-level
      bootstrap, 1000 resamples, seed 2757010, 95% CI lower bound of
      median_i[D_randn - D_sema] > 0).
      C1_summary = trained_localised iff >= 4 of 6 runs pass.
  C2 (direction, per seed): D_base_median(surface_class_mass) >
      D_base_median(true_token) within each seed (2747: surface-class
      supervision preserved discrimination, true-token harmed it).
  C3 (bias-correction origin hypothesis): D_base_median > native D_base_median
      (-22.3125, Phase 2757 result.json) for all 6 runs.
Integrity gates:
  G0 deployment identity: delta npz SHA256 equals its receipt; recomputed
      deployed BF16 delta L2 (float64) equals the recorded
      deployed_BF16_delta_L2 within relative 1e-9.
  G1a pass-through: inactive hook == plain forward, bit-exact (first 8 pairs,
      every run). G1b: sema_100 shifts base logits on every pair, every run.
Exploratory (non-preregistered): per-pair D_base correlation with native,
family-level gap structure.
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import rdc_construction_common as cc
import rdc_feature_common as fc
import phase2747_rdc_material as mat2747

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2758' / 'qwen4_trained_D'
BANDS = [25, 50, 100]
RAND_SEEDS = [2757001, 2757002]
BOOT_SEED = 2757010
BOOT_N = 1000
NATIVE_D_BASE_MEDIAN = -22.3125  # Phase 2757 result.json, frozen reference

RUNS = [('true_2747', 'true_token', 2747),
        ('perm_2747', 'within_surface_class_permuted_token', 2747),
        ('mass_2747', 'surface_class_mass', 2747),
        ('true_2748', 'true_token', 2748),
        ('perm_2748', 'within_surface_class_permuted_token', 2748),
        ('mass_2748', 'surface_class_mass', 2748)]
TRAIN_ROOT = BASE / 'phase2747' / 'training'

PREREG = {
    'phase': 2758,
    'question': 'Does 2747 trained state change the 2757 findings: saturated '
                'negative dual-world discrimination and absent block12 '
                'targeted-vs-random behavioural specificity?',
    'model': 'qwen3-4b with block16 mlp replaced by each of the six 2747 '
             'deployed BF16 checkpoints (3 conditions x 2 seeds)',
    'panel': '2747 diagnostic controlled_relation: 160 pairs x 2 worlds (same as 2757)',
    'variants': 'identical to 2757: base + sema/randn/rand x 25/50/100% block12 '
                'v-row zeroing, single-pass hook scoring, same control seeds',
    'native_reference_frozen': {'D_base_median': NATIVE_D_BASE_MEDIAN,
                                'source': 'phase2757/qwen4_block12/result.json'},
    'criteria': {'C1': 'per run: mechanism_localised iff J1 and J2 (2757 '
                       'definitions, bootstrap seed 2757010); '
                       'C1_summary=trained_localised iff >=4/6 runs pass',
                 'C2': 'D_base_median(mass) > D_base_median(true) within each seed',
                 'C3': 'D_base_median > -22.3125 for all 6 runs'},
    'integrity_gates': {'G0': 'delta npz sha256 == receipt; recomputed deployed '
                              'BF16 delta L2 == recorded within rel 1e-9',
                        'G1a': 'inactive hook == plain forward, bit-exact, '
                               'first 8 pairs every run',
                        'G1b': 'sema_100 shifts base logits on every pair, every run'},
    'exploratory': ['per-pair D_base Pearson vs native (2757 npz)',
                    'family-level gap breakdown per run'],
    'frozen_before_any_behavioural_forward': True,
}


def span_positions(row):
    text = row['text']
    offs = row['token_offsets']
    assert len(offs) == len(row['prompt_ids']), (len(offs), len(row['prompt_ids']))
    c0 = text.find(row['body'])
    assert c0 >= 0, row['sample_id']
    c1 = c0 + len(row['body'])
    return np.array([i for i, (a, b) in enumerate(offs) if a < c1 and b > c0], dtype=np.int64)


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def deploy(model, named, original, run_key, condition, seed):
    """Rebuild trained block16 mlp exactly as 2747 deployment; G0 identity gate."""
    folder = TRAIN_ROOT / ('%s_%d' % (condition, seed))
    receipt = fc.read(folder / 'commits' / 'delta_128.json')
    npz_path = ROOT / receipt['field_path']
    if not npz_path.exists():
        npz_path = Path(receipt['physical_path'])
    assert sha256_file(npz_path) == receipt['field_sha256'], ('G0 sha mismatch', run_key)
    with np.load(npz_path) as z:
        with __import__('torch').no_grad():
            for n, p in named.items():
                rebuilt = original[n].numpy() + z[n]
                rebuilt = rebuilt + z['reconstruction_residual__' + n]
                p.copy_(__import__('torch').tensor(rebuilt, device=p.device))
    import torch
    deployed_norm = float(torch.stack(
        [(p.detach().float().cpu() - original[n]).double().square().sum()
         for n, p in named.items()]).sum().sqrt())
    recorded = fc.read(folder / 'result.json')['deployed_BF16_delta_L2']
    rel = abs(deployed_norm - recorded) / max(abs(recorded), 1e-30)
    assert rel < 1e-9, ('G0 L2 mismatch', run_key, deployed_norm, recorded)
    return deployed_norm


def restore(model, named, original):
    import torch
    with torch.no_grad():
        for n, p in named.items():
            p.copy_(original[n].to(p.device).to(p.dtype))


def run_panel(model, rows, pair_ids, pairs, run_key):
    import torch
    device = next(model.parameters()).device
    hook_state = {'positions': None, 'last_out': None}

    def vhook(module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        p = hook_state['positions']
        if p is None:
            hook_state['last_out'] = out
        else:
            out[0, list(p), :] = 0
        return None

    handle = model.model.layers[12].self_attn.v_proj.register_forward_hook(vhook)
    g1a_diffs = []
    per_variant = defaultdict(lambda: defaultdict(list))

    rng_a = np.random.default_rng(RAND_SEEDS[0])
    rng_b = np.random.default_rng(RAND_SEEDS[1])

    with torch.inference_mode():
        for pi, pid in enumerate(pair_ids):
            w0, w1 = pairs[pid]
            ctxs = []
            for widx in (w0, w1):
                row = rows[widx]
                pids_l = row['prompt_ids']
                body_pos = span_positions(row)
                n = len(pids_l)
                nonbody = np.array([i for i in range(n)
                                    if i not in set(body_pos.tolist())], dtype=np.int64)
                ids = torch.tensor([pids_l], device=device)
                hook_state['positions'] = None
                base_logits = model(ids).logits[0, -1]
                vout = hook_state['last_out'][0]
                plain = model(ids).logits[0, -1]
                if pi < 8:
                    g1a_diffs.append(float((plain - base_logits).abs().max()))
                lp = torch.log_softmax(base_logits.float(), dim=-1)
                m_base = float(lp[rows[w1 if widx == w0 else w0]['target']] - lp[row['target']])
                pos_norm = vout.float().norm(dim=-1).cpu().numpy()
                ctxs.append(dict(widx=widx, row=row, ids=ids, body=body_pos,
                                 nonbody=nonbody, m_base=m_base,
                                 pos_norm=pos_norm, base_logits=base_logits))
            assert max(g1a_diffs) == 0.0, ('G1a violated', run_key, g1a_diffs)

            m0, m1 = ctxs[0]['m_base'], ctxs[1]['m_base']
            per_variant['base']['D'].append(min(m0, m1))
            per_variant['base']['m0'].append(m0)
            per_variant['base']['m1'].append(m1)

            nbody = len(ctxs[0]['body'])
            n_nonbody = len(ctxs[0]['nonbody'])
            assert len(ctxs[1]['body']) == nbody and len(ctxs[1]['nonbody']) == n_nonbody
            plan = []
            for k in BANDS:
                nsel = int(np.ceil(k / 100 * nbody))
                plan.append(('sema_%d' % k, 'sema', nsel, None))
                nsel_ctl = min(nsel, n_nonbody)
                plan.append(('randn_%d' % k, 'randn', nsel_ctl, None))
                plan.append(('rand_%d_a' % k, 'rand', nsel_ctl, rng_a))
                plan.append(('rand_%d_b' % k, 'rand', nsel_ctl, rng_b))

            for name, kind, nsel, rgen in plan:
                ms = {}
                shifted_all = True
                for ctx in ctxs:
                    if kind == 'sema':
                        pool = ctx['body']
                        sel = pool[np.argsort(-ctx['pos_norm'][pool])][:nsel]
                    elif kind == 'randn':
                        pool = ctx['nonbody']
                        sel = pool[np.argsort(-ctx['pos_norm'][pool])][:nsel]
                    else:
                        pool = ctx['nonbody']
                        sel = np.sort(rgen.choice(pool, size=nsel, replace=False))
                    hook_state['positions'] = sel.tolist()
                    logits = model(ctx['ids']).logits[0, -1]
                    hook_state['positions'] = None
                    if not bool((logits - ctx['base_logits']).abs().max() > 0):
                        shifted_all = False
                    lp = torch.log_softmax(logits.float(), dim=-1)
                    other_tgt = rows[w1 if ctx['widx'] == w0 else w0]['target']
                    ms[ctx['widx']] = float(lp[other_tgt] - lp[ctx['row']['target']])
                if name == 'sema_100':
                    assert shifted_all, ('G1b violated', run_key, pid)
                per_variant[name]['D'].append(min(ms[w0], ms[w1]))
                per_variant[name]['m0'].append(ms[w0])
                per_variant[name]['m1'].append(ms[w1])
            if pi % 40 == 0:
                print('P2758 %s PAIR %d/160' % (run_key, pi), flush=True)
    handle.remove()
    return per_variant, g1a_diffs


def stats_for(per_variant, fam_of_pair):
    D_base = np.array(per_variant['base']['D'])
    res = {'D_base_median': float(np.median(D_base)),
           'D_base_mean': float(D_base.mean()),
           'D_base_negative_fraction': float((D_base <= 0).mean())}
    band_stats = {}
    for k in BANDS:
        d_sema = D_base - np.array(per_variant['sema_%d' % k]['D'])
        d_randn = D_base - np.array(per_variant['randn_%d' % k]['D'])
        a = np.array(per_variant['rand_%d_a' % k]['D'])
        b = np.array(per_variant['rand_%d_b' % k]['D'])
        band_stats[k] = {'dD_sema_median': float(np.median(d_sema)),
                         'dD_randn_median': float(np.median(d_randn)),
                         'dD_rand_median': float(np.median(D_base - (a + b) / 2))}
    res['band_stats'] = band_stats
    j1_bands = [k for k in BANDS
                if band_stats[k]['dD_sema_median'] > band_stats[k]['dD_randn_median']]
    res['J1_bands'] = j1_bands
    res['J1'] = len(j1_bands) >= 2
    k = 100
    d_sema = D_base - np.array(per_variant['sema_%d' % k]['D'])
    d_randn = D_base - np.array(per_variant['randn_%d' % k]['D'])
    per_pair_gap = d_randn - d_sema
    rngb = np.random.default_rng(BOOT_SEED)
    stats = np.empty(BOOT_N)
    for b in range(BOOT_N):
        idx = rngb.integers(0, len(per_pair_gap), len(per_pair_gap))
        stats[b] = np.median(per_pair_gap[idx])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    res['J2'] = {'gap_median': float(np.median(per_pair_gap)),
                 'boot_ci95': [float(lo), float(hi)], 'pass': bool(lo > 0)}
    res['J2_pass'] = bool(lo > 0)
    res['verdict'] = ('mechanism_localised'
                      if res['J1'] and res['J2_pass'] else 'not_localised')
    fam_break = {}
    for fam in sorted(set(fam_of_pair.tolist())):
        sel = fam_of_pair == fam
        fam_break[fam] = {'gap_median': float(np.median(per_pair_gap[sel])),
                          'n': int(sel.sum())}
    res['family_breakdown_k100'] = fam_break
    return res, D_base, per_pair_gap


def main():
    t0 = time.time()
    cc.guard(0)
    assert not (OUT / 'result.json').exists(), 'result.json immutable; delete before rerun'
    OUT.mkdir(parents=True, exist_ok=True)
    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__), 'prereg': PREREG}
    fc.save(OUT / 'execution.json', execution)

    material, data = mat2747.freeze()
    rows = [r for r in data['diagnostic'] if r['kind'] == 'controlled_relation']
    assert len(rows) == 320, len(rows)
    pairs = defaultdict(list)
    for idx, r in enumerate(rows):
        pairs[r['pair_id']].append(idx)
    assert all(len(v) == 2 for v in pairs.values()) and len(pairs) == 160
    pair_ids = sorted(pairs)
    fam_of_pair = np.array([rows[pairs[pid][0]]['family'] for pid in pair_ids])

    from phase2662_symmetric_mapping_contract import load_native
    model, tok = load_native('qwen4')
    model.eval()
    assert model.config.num_hidden_layers == 36

    target = model.model.layers[16].mlp
    named = dict(target.named_parameters())
    original = {n: p.detach().float().cpu().clone() for n, p in named.items()}
    assert sum(p.numel() for p in named.values()) == 74711040

    with np.load(BASE / 'phase2757' / 'qwen4_block12' / 'pair_scores.npz') as z:
        native_D_base = z['D_base'].astype(np.float64)

    run_results = {}
    arrays = {'pair_ids': np.array(pair_ids, dtype=np.str_),
              'families': fam_of_pair.astype(np.str_),
              'native_D_base': native_D_base}
    for run_key, condition, seed in RUNS:
        deployed_norm = deploy(model, named, original, run_key, condition, seed)
        per_variant, g1a = run_panel(model, rows, pair_ids, pairs, run_key)
        res, D_base, per_pair_gap = stats_for(per_variant, fam_of_pair)
        res['deployed_BF16_delta_L2_recomputed'] = deployed_norm
        res['G1a_max_abs_diff'] = g1a
        res['pearson_D_base_vs_native'] = float(np.corrcoef(D_base.astype(np.float64),
                                                            native_D_base)[0, 1])
        run_results[run_key] = res
        for name, fields in per_variant.items():
            for f, vals in fields.items():
                arrays['%s__%s__%s' % (run_key, name, f)] = np.array(vals)
        arrays['%s__per_pair_gap_k100' % run_key] = per_pair_gap
        print('P2758_RUN_DONE %s %s Dmed=%.4f J2=%s' %
              (run_key, res['verdict'], res['D_base_median'], res['J2_pass']), flush=True)
    restore(model, named, original)

    results = {'runs': run_results}
    c1_passes = [k for k, r in run_results.items() if r['verdict'] == 'mechanism_localised']
    results['C1_pass_runs'] = c1_passes
    results['C1_summary'] = 'trained_localised' if len(c1_passes) >= 4 else 'not_localised_majority'
    c2 = {}
    for seed in (2747, 2748):
        c2['seed_%d' % seed] = bool(run_results['mass_%d' % seed]['D_base_median'] >
                                    run_results['true_%d' % seed]['D_base_median'])
    results['C2_mass_gt_true'] = c2
    results['C2_pass'] = all(c2.values())
    c3 = {k: bool(r['D_base_median'] > NATIVE_D_BASE_MEDIAN) for k, r in run_results.items()}
    results['C3_above_native'] = c3
    results['C3_pass'] = all(c3.values())
    results['native_reference'] = {'D_base_median': NATIVE_D_BASE_MEDIAN,
                                   'verdict': 'not_localised',
                                   'J2_ci95': [-1.0, -0.25]}
    results['seconds'] = time.time() - t0
    fc.npz(OUT / 'pair_scores.npz', **arrays)
    fc.save(OUT / 'pair_scores_meta.json',
            {'schema': 'per run (6) x per variant (13) x fields D/m0/m1; '
                       'D = min over worlds of m = nll(other.target) - nll(this.target)',
             'runs': [k for k, _, _ in RUNS]})
    fc.save(OUT / 'result.json', results)
    print('PHASE2758_DONE ' + json.dumps(
        {'C1_summary': results['C1_summary'], 'C1_pass_runs': c1_passes,
         'C2_pass': results['C2_pass'], 'C3_pass': results['C3_pass'],
         'D_base_medians': {k: r['D_base_median'] for k, r in run_results.items()},
         'seconds': results['seconds']}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:  # crash report to disk (stdout capture unreliable)
        import traceback
        (OUT / 'crash.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
