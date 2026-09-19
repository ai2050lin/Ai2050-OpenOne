"""Phase2753: P5 step 2 refinement — single-pair subset isolation and complement net effect.

Questions (frozen before any Phase2753 observation), continuing Phase2752:
  Q1  Which of the three Phase2751 pair signatures (true_vs_permuted / true_vs_mass /
      permuted_vs_mass) ALONE carries each supervision condition's own-objective
      improvement?  Phase2752 measured necessity (pz_<pair>: remove one pair from full)
      and union sufficiency (only_union); it never isolated a single pair.
  Q2  Phase2752 found the true_token runs' complement (everything outside the union
      subset) is net harmful.  only_complement measures the complement's standalone
      effect directly, per condition, and its family profile.
  Q3  Additivity: red(only_union) + red(only_complement) vs red(full_noR) per run
      (descriptive; nonlinear interaction + bit-repair residual excluded).

Variants per run (all within-session, FP32 block behind the float-in/BF16-out bridge):
  base               original (no delta)                 [cached per condition]
  full               original + D + R                    (reference)
  full_noR           original + D
  only_<pair>        original + D ONLY on that pair's top-1% subset   (3 pairs)
  only_union         original + D ONLY on the union subset
  only_complement    original + D ONLY outside the union subset
R (bit-repair residual, |R|~1e-7) is excluded from all only_/full_noR variants,
mirroring the Phase2752 residual policy; full keeps it.

Panel: the frozen 896-position Phase2747 panel (validation 192 + diagnostic 512 +
fresh 192), with the Phase2752 material extension (surface_class = classes[target];
permuted_target = within-(kind, cohort, language, surface_class)-group permutation,
seed 2752001 — byte-identical procedure, so permuted targets match Phase2752).

Preregistered criteria:
  C1_carrier_consistency  For each supervision condition, argmax_p red(only_p) is the
                          same pair across its two material generations; support = 3/3
                          conditions.  red(x) = mean own_obj(base) - mean own_obj(x).
  C2_complement_asymmetry red(only_complement) < 0 in 2/2 true_token runs AND
                          red(only_complement) is larger (less negative) in the
                          permuted run than in the true run of the same generation,
                          2/2 generations.
  C3_additivity           Descriptive only: |red(only_union) + red(only_complement)
                          - red(full_noR)| recorded per run; no pass/fail.
  C4_pair_effect_matrix   Descriptive only: full 3 pairs x 3 conditions x 2 generations
                          red matrix with family decomposition (own_mean_by_family).
Storage: direct on D: inside BASE (post-migration layout, 2026-09-15); no junction.
"""
import gc
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import rdc_formation_common as fcom
from rdc_construction_common import (BASE, ROOT, stamp, sha, read, save, snapshot,
                                     guard, ledger, npz, load as cc_load)
from phase2747_rdc_training import bridge, evaluate, packet
import phase2747_rdc_material as mat2747

OUT53 = BASE / 'phase2753'
OUT51_NPZ = BASE / 'phase2751/qwen4_block16/subset_indices.npz'
TRAIN_PHYS = BASE / 'phase2747_fields/training'
RESULT52 = BASE / 'phase2752/qwen4_block16/result.json'
MAT_KEYS = ['gate_proj.weight', 'up_proj.weight', 'down_proj.weight']
GATE_N = 24_903_680
UP_N = 24_903_680
RUNS = ['true_token_2747', 'within_surface_class_permuted_token_2747', 'surface_class_mass_2747',
        'true_token_2748', 'within_surface_class_permuted_token_2748', 'surface_class_mass_2748']
PAIRS = ['true_vs_permuted', 'true_vs_mass', 'permuted_vs_mass']
CONDITIONS = ['true_token', 'within_surface_class_permuted_token', 'surface_class_mass']
VARIANTS = ['full', 'full_noR', 'only_true_vs_permuted', 'only_true_vs_mass',
            'only_permuted_vs_mass', 'only_union', 'only_complement']
SEED_PANEL_PERM = 2752001
START = time.monotonic()


def elapsed():
    return round(time.monotonic() - START, 1)


def condition_of(run):
    return run.rsplit('_', 1)[0]


def split_indices(idx):
    """Pooled flat index over [gate flat | up flat | down flat] -> per-matrix flat indices."""
    g = idx[idx < GATE_N].astype(np.int64)
    u = idx[(idx >= GATE_N) & (idx < GATE_N + UP_N)].astype(np.int64) - GATE_N
    d = idx[idx >= GATE_N + UP_N].astype(np.int64) - GATE_N - UP_N
    return {'gate_proj.weight': np.sort(g), 'up_proj.weight': np.sort(u),
            'down_proj.weight': np.sort(d)}


def main():
    import shutil
    folder = OUT53 / 'qwen4_block16'
    finish = folder / 'result.json'
    if finish.exists():
        return
    guard(0)
    assert shutil.disk_usage('D:/').free > 4 * 1024 ** 3
    folder.mkdir(parents=True, exist_ok=True)

    # ---- frozen design (before any Phase2753 behavioral observation) ----
    z_sub = np.load(OUT51_NPZ, allow_pickle=False)
    pair_idx = {p: z_sub['pooled__' + p + '__top1pct'].astype(np.int64) for p in PAIRS}
    union_flat = np.unique(np.concatenate([pair_idx[p] for p in PAIRS]))
    union = split_indices(union_flat)
    pair_split = {p: split_indices(pair_idx[p]) for p in PAIRS}
    u_sizes = {k: int(len(union[k])) for k in MAT_KEYS}

    execution = {'source': snapshot(__file__),
                 'phase2751_subset_npz_sha256': sha(OUT51_NPZ),
                 'phase2747_training_result_sha256': sha(BASE / 'phase2747/training/result.json'),
                 'phase2752_result_sha256': sha(RESULT52),
                 'storage': 'direct on D: under BASE (post-migration layout 2026-09-15); no junction',
                 'subset_sizes': {p: int(len(pair_idx[p])) for p in PAIRS},
                 'union_sizes_per_matrix': u_sizes,
                 'union_size_total': int(sum(u_sizes.values())),
                 'panel': 'phase2747 material freeze: validation + diagnostic + fresh, same order',
                 'panel_material_extension': {
                     'surface_class': 'classes[target] (deterministic)',
                     'permuted_target': 'within (kind, cohort, language, surface_class) group '
                                        'permutation, seed 2752001 — identical procedure and '
                                        'seed to Phase2752, so permuted targets match',
                     'reason': 'permuted/mass own objectives on the panel require the '
                               'Phase2752 definitions'},
                 'runs': RUNS, 'variants': VARIANTS,
                 'residual_policy': 'all only_/full_noR variants exclude R (bit-repair term, '
                                    '|R|~1e-7); full keeps R — mirrors Phase2752',
                 'criteria': {
                     'C1_carrier_consistency':
                         'argmax_p red(only_p) identical across the two generations of each '
                         'condition; support = 3/3 conditions',
                     'C2_complement_asymmetry':
                         'red(only_complement) < 0 in 2/2 true_token runs AND red(only_complement) '
                         'greater in the permuted run than the true run of the same generation, '
                         '2/2 generations',
                     'C3_additivity': 'descriptive: |red(only_union)+red(only_complement)-'
                                      'red(full_noR)| per run',
                     'C4_pair_effect_matrix': 'descriptive: 3 pairs x 3 conditions x 2 generations '
                                              'red matrix + family decomposition'},
                 'question': 'Which single pair signature alone carries each condition\'s '
                             'behavioral effect, and is the true-token complement harm a '
                             'standalone, condition-specific effect?',
                 'status': 'frozen_before_any_Phase2753_intervention_observation'}
    exec_path = folder / 'execution.json'
    if exec_path.exists():
        old = read(exec_path)
        assert old['execution'] == execution, 'Phase2753 execution drift'
    else:
        save(exec_path, {'timestamp': stamp(), 'execution': execution})

    # ---- load model, material, classes ----
    model, tok = cc_load('qwen4', folder)
    assert all(p.device.type == 'cuda' for p in model.parameters())
    gpu_name = torch.cuda.get_device_name(0)
    material, data = mat2747.freeze()
    with np.load(ROOT / material['vocabulary_receipt']['field_path']) as z:
        classes = torch.tensor(z['classes'].astype(np.int64), device='cuda')
    panel = data['validation'] + data['diagnostic'] + data['fresh']
    assert len(panel) == 896

    # ---- material extension (identical to Phase2752) ----
    classes_np = classes.cpu().numpy()
    for r in panel:
        r['surface_class'] = int(classes_np[int(r['target'])])
    groups_panel = defaultdict(list)
    for i, r in enumerate(panel):
        groups_panel[(r['kind'], r['cohort'], r['language'], r['surface_class'])].append(i)
    rng_mat = np.random.default_rng(SEED_PANEL_PERM)
    permutation = np.arange(len(panel))
    for key, ix in groups_panel.items():
        permutation[ix] = rng_mat.permutation(ix)
    n_changed = 0
    for i, r in enumerate(panel):
        r['permuted_target'] = panel[int(permutation[i])]['target']
        assert classes_np[r['target']] == classes_np[r['permuted_target']]
        n_changed += int(r['target'] != r['permuted_target'])
    panel_ext = {'panel_rows': len(panel), 'groups': len(groups_panel),
                 'permuted_changed_fraction': n_changed / len(panel)}
    print('PHASE2753_PROGRESS panel_extension', json.dumps(panel_ext), elapsed(), flush=True)

    target = model.model.layers[16].mlp
    with torch.no_grad():
        target.float()
    named = dict(target.named_parameters())
    assert sum(p.numel() for p in named.values()) == 74_711_040
    original = {k: named[k].detach().cpu().clone() for k in MAT_KEYS}
    handles = bridge(target)

    def set_state(state):
        with torch.no_grad():
            for k in MAT_KEYS:
                named[k].copy_(torch.from_numpy(np.ascontiguousarray(state[k])).to(named[k].device))

    def eval_variant(run_condition, rows):
        acc = defaultdict(list)
        with torch.no_grad():
            for row in rows:
                post, z = packet(model, row)
                lp = z.double().log_softmax(-1)
                p = lp.exp()
                acc['nll_target'].append(float(-lp[int(row['target'])]))
                acc['nll_permuted'].append(float(-lp[int(row['permuted_target'])]))
                am = int(z.argmax())
                acc['argmax_is_target'].append(am == int(row['target']))
                acc['argmax_is_permuted'].append(am == int(row['permuted_target']))
                acc['entropy'].append(float(-(lp * p).sum()))
                if run_condition == 'surface_class_mass':
                    own = float(-torch.logsumexp(lp[classes == int(row['surface_class'])], dim=0))
                elif run_condition == 'true_token':
                    own = float(-lp[int(row['target'])])
                else:
                    own = float(-lp[int(row['permuted_target'])])
                acc['own_objective'].append(own)
        return {k: np.asarray(v) for k, v in acc.items()}

    # ---- base state + integrity gate ----
    set_state({k: original[k].numpy() for k in MAT_KEYS})
    base_check = evaluate(model, panel, classes)
    baseline = np.load(fcom.FIELDS / 'training/baseline/bridge.npz', allow_pickle=False)
    exact = {k: bool(np.array_equal(base_check[k], baseline[k]))
             for k in ['postnorm_BF16', 'NLL', 'argmax', 'entropy', 'surface_mass']}
    dnll = float(np.abs(base_check['NLL'].astype(np.float64) - baseline['NLL'].astype(np.float64)).mean())
    agree = float(np.mean(base_check['argmax'] == baseline['argmax']))
    dent = float(np.abs(base_check['entropy'].astype(np.float64) - baseline['entropy'].astype(np.float64)).mean())
    dsm = float(np.abs(base_check['surface_mass'].astype(np.float64) - baseline['surface_mass'].astype(np.float64)).mean())
    integrity = {'exact_array_equality': exact, 'mean_abs_dNLL': dnll, 'argmax_agreement': agree,
                 'mean_abs_dEntropy': dent, 'mean_abs_dSurfaceMass': dsm,
                 'gpu_name': gpu_name,
                 'strict_pass': all(exact.values()),
                 'soft_pass': bool(dnll < 0.05 and agree >= 0.99 and dent < 0.05 and dsm < 0.001)}
    assert integrity['strict_pass'] or integrity['soft_pass'], ('bridge baseline integrity failed', integrity)

    families = [r['family'] for r in panel]
    arrays = {}

    def summarize(ev):
        fam = defaultdict(list)
        for f, v in zip(families, ev['own_objective']):
            fam[f].append(v)
        return {'own_mean': float(ev['own_objective'].mean()),
                'nll_target_mean': float(ev['nll_target'].mean()),
                'nll_permuted_mean': float(ev['nll_permuted'].mean()),
                'argmax_target_rate': float(ev['argmax_is_target'].mean()),
                'argmax_permuted_rate': float(ev['argmax_is_permuted'].mean()),
                'own_mean_by_family': {f: float(np.mean(v)) for f, v in sorted(fam.items())}}

    metrics = {}

    # ---- per-run interventions ----
    for run in RUNS:
        cond = condition_of(run)
        if cond not in metrics:
            set_state({k: original[k].numpy() for k in MAT_KEYS})
            base_ev = eval_variant(cond, panel)
            metrics['base__' + cond] = summarize(base_ev)
            arrays['base__' + cond + '__own'] = base_ev['own_objective'].astype(np.float64)
            print('PHASE2753_PROGRESS base', cond, round(metrics['base__' + cond]['own_mean'], 6),
                  elapsed(), flush=True)

        zd = np.load(TRAIN_PHYS / run / 'delta_128.npz', allow_pickle=False)
        D = {k: zd[k].astype(np.float32) for k in MAT_KEYS}
        R = {k: zd['reconstruction_residual__' + k].astype(np.float32) for k in MAT_KEYS}
        del zd
        o_plus_d = {k: original[k].numpy() + D[k] for k in MAT_KEYS}
        full_state = {k: o_plus_d[k] + R[k] for k in MAT_KEYS}

        states = {'full': full_state, 'full_noR': o_plus_d}
        for p in PAIRS:
            st = {k: original[k].numpy().copy() for k in MAT_KEYS}
            for k in MAT_KEYS:
                ix = pair_split[p][k]
                st[k].flat[ix] = o_plus_d[k].flat[ix]
            states['only_' + p] = st
        st = {k: original[k].numpy().copy() for k in MAT_KEYS}
        for k in MAT_KEYS:
            ix = union[k]
            st[k].flat[ix] = o_plus_d[k].flat[ix]
        states['only_union'] = st
        st = {k: o_plus_d[k].copy() for k in MAT_KEYS}
        for k in MAT_KEYS:
            ix = union[k]
            st[k].flat[ix] = original[k].numpy().flat[ix]
        states['only_complement'] = st

        for variant in VARIANTS:
            set_state(states[variant])
            ev = eval_variant(cond, panel)
            metrics.setdefault(run, {})[variant] = summarize(ev)
            arrays[run + '__' + variant + '__own'] = ev['own_objective'].astype(np.float64)
            arrays[run + '__' + variant + '__nll_target'] = ev['nll_target'].astype(np.float64)
            arrays[run + '__' + variant + '__nll_permuted'] = ev['nll_permuted'].astype(np.float64)
            print('PHASE2753_PROGRESS', run, variant,
                  round(metrics[run][variant]['own_mean'], 6), elapsed(), flush=True)
        del states, D, R, full_state, o_plus_d
        gc.collect()

    # ---- restore original exactly ----
    set_state({k: original[k].numpy() for k in MAT_KEYS})
    restored = evaluate(model, panel[:4], classes)
    restore_exact = {k: bool(np.array_equal(restored[k], base_check[k][:4])) for k in restored}
    assert all(restore_exact.values()), ('original state not restored bit-exactly', restore_exact)

    # ---- criteria ----
    results = {}
    red = {}   # red[run][variant]
    for run in RUNS:
        cond = condition_of(run)
        m = metrics[run]
        base_own = metrics['base__' + cond]['own_mean']
        red[run] = {v: base_own - m[v]['own_mean'] for v in VARIANTS}
        fam_delta = {v: {f: m[v]['own_mean_by_family'][f] - metrics['base__' + cond]['own_mean_by_family'][f]
                         for f in m[v]['own_mean_by_family']} for v in VARIANTS}
        add_gap = abs(red[run]['only_union'] + red[run]['only_complement'] - red[run]['full_noR'])
        results[run] = {'condition': cond,
                        'reduction_vs_base': {k: float(x) for k, x in red[run].items()},
                        'additivity_gap_union_plus_complement_vs_full_noR': float(add_gap),
                        'family_delta_own_by_variant': fam_delta}

    c1_details, c1_ok = {}, []
    for cond in CONDITIONS:
        argmaxes = {}
        for gen in ['2747', '2748']:
            run = cond + '_' + gen
            argmaxes[gen] = max(PAIRS, key=lambda p: red[run]['only_' + p])
        same = argmaxes['2747'] == argmaxes['2748']
        c1_ok.append(same)
        c1_details[cond] = {'argmax_2747': argmaxes['2747'], 'argmax_2748': argmaxes['2748'],
                            'consistent': same}

    c2_parts, c2_ok = {}, []
    for gen in ['2747', '2748']:
        t = red['true_token_' + gen]['only_complement']
        p_ = red['within_surface_class_permuted_token_' + gen]['only_complement']
        ok = bool(t < 0 and p_ > t)
        c2_ok.append(ok)
        c2_parts[gen] = {'red_comp_true': float(t), 'red_comp_permuted': float(p_), 'asymmetric_harm': ok}

    verdict = {'C1_carrier_consistency_conditions': int(sum(c1_ok)),
               'C1_details': c1_details,
               'C2_complement_asymmetry_generations': int(sum(c2_ok)),
               'C2_details': c2_parts,
               'preregistered': execution['criteria']}

    # ---- descriptive cross-check with Phase2752 necessity decomposition ----
    try:
        r52 = read(RESULT52)
        xcheck = {run: {'pz_pairs': r52['criteria_results'][run]['per_pair_zero_effect'],
                        'red_only_union_2752': r52['criteria_results'][run]['reduction_vs_base']['only_union']}
                  for run in RUNS}
    except Exception as ex:  # noqa
        xcheck = {'error': str(ex)}

    value = {'timestamp': stamp(), 'source': snapshot(__file__),
             'all_passed': True,
             'status': 'P5 step 2 refinement complete',
             'model': 'qwen4', 'gpu_name': gpu_name,
             'trainable_block': 'model.layers[16].mlp', 'runs': RUNS, 'variants': VARIANTS,
             'union_size_total': int(sum(u_sizes.values())),
             'panel_extension': panel_ext,
             'integrity_gate': integrity, 'restore_exact': restore_exact,
             'metrics': metrics, 'criteria_results': results, 'verdict': verdict,
             'phase2752_crosscheck': xcheck,
             'subset_npz': str(OUT51_NPZ),
             'seconds': elapsed(),
             'limits': ['Interventions cover only the trained layers[16].mlp block; the rest of '
                        'the model is original BF16 throughout.',
                        'only_*/full_noR exclude the bit-repair residual (|R|~1e-7); full keeps it.',
                        'Effects decompose one trained checkpoint at a time; interactions between '
                        'pairs are captured only by the additivity gap, not by factorial design.',
                        'Forward numerics may differ from the 2747 run if the GPU architecture '
                        'changed; the integrity gate handles this via the preregistered soft path, '
                        'and all variant comparisons are within-session.']}
    npz(folder / 'isolation_scores.npz', **arrays)
    save(folder / 'isolation_scores_meta.json',
         {'timestamp': stamp(), 'panel_sample_ids': [r['sample_id'] for r in panel],
          'families': families, 'runs': RUNS, 'variants': VARIANTS,
          'array_keys': sorted(arrays.keys())})
    save(finish, value)
    ledger('phase2753_pair_isolation', value['seconds'])
    print('PHASE2753_PAIR_ISOLATION', json.dumps(verdict), elapsed(), flush=True)


if __name__ == '__main__':
    main()
