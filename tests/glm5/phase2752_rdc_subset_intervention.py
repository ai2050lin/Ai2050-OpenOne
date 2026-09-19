"""Phase2752: P5 step 2 — targeted freeze/shuffle interventions on the Phase2751 subsets.

Question (frozen before any intervention observation): do the Phase2751 localized
top-1% parameter subsets of the layers[16].mlp block causally carry the
supervision-specific behavioral effect formed by the Phase2747 training runs?
For each of the six audited checkpoints (3 supervision conditions x 2 material
generations) the exact final FP32 state is reconstructed bit-exactly
(original + delta_128 + reconstruction_residual, the Phase2747 receipt chain),
the Phase2751 pooled top-1% subsets are decoded to block coordinates, and the
delta is intervened on:

  full          original + D + R                     (reference)
  pz_<pair>     full minus D on that pair subset     (3 pairs)
  pz_union      full minus D on the union subset
  rzero_union   full minus D on a random matched set (seed 2752000, from complement)
  only_union    original + D on the union subset only (residual excluded: it
                belongs to the full-state receipt; magnitude ~1e-7)
  ronly_union   original + D on the random set only
  shuffle_union full with union-delta values permuted within each matrix
                (seed 2752100 + run ordinal)

All variants share the Phase2747 in-training evaluation regime: FP32 trainable
block behind the float-in/BF16-out bridge, all other parameters original BF16,
single-row forwards over the frozen 896-position panel (validation 192 +
diagnostic 512 + fresh 192, identical order to Phase2747).  The base state
(original, no delta) is asserted against the committed Phase2747 bridge
baseline: bit-exact if the GPU architecture matches the 2747 run, otherwise a
preregistered soft gate (mean |dNLL| < 0.05, argmax agreement >= 0.99).

Preregistered criteria (per run, own objective = the run's training objective
on the panel):
  P1 necessity   d_obj(pz_union)  > d_obj(rzero_union)   (d_obj = mean obj - mean obj(full); positive = worse)
  P2 sufficiency red(only_union)  > red(ronly_union)     (red = mean obj(base) - mean obj(variant))
  P3 placement   d_obj(shuffle_union) > 0
  localisation_functional = P1 in 6/6 runs AND P2 in 6/6 runs (binomial null p = 1/64 each).
P4 (descriptive specificity): the 3 pair subsets x 6 runs effect matrix.
"""
import argparse
import gc
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import rdc_formation_common as fcom
from rdc_construction_common import (BASE, ROOT, stamp, sha, read, save, snapshot,
                                     guard, ledger, npz, failure, load as cc_load)
from phase2747_rdc_training import bridge, evaluate, packet
import phase2747_rdc_material as mat2747

OUT52 = BASE / 'phase2752'
PHYSICAL52 = BASE / 'phase2752'
OUT51_NPZ = BASE / 'phase2751/qwen4_block16/subset_indices.npz'
TRAIN_PHYS = BASE / 'phase2747_fields/training'
MAT_KEYS = ['gate_proj.weight', 'up_proj.weight', 'down_proj.weight']
GATE_N = 24_903_680
UP_N = 24_903_680
RUNS = ['true_token_2747', 'within_surface_class_permuted_token_2747', 'surface_class_mass_2747',
        'true_token_2748', 'within_surface_class_permuted_token_2748', 'surface_class_mass_2748']
PAIRS = ['true_vs_permuted', 'true_vs_mass', 'permuted_vs_mass']
VARIANTS = ['full', 'pz_true_vs_permuted', 'pz_true_vs_mass', 'pz_permuted_vs_mass',
            'pz_union', 'rzero_union', 'only_union', 'ronly_union', 'shuffle_union']
SEED_RANDOM = 2752000
SEED_SHUFFLE_BASE = 2752100
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


def main(partial=False):
    import shutil
    folder = OUT52 / 'qwen4_block16'
    finish = folder / ('partial_' + str(time.time_ns()) + '.json' if partial else 'result.json')
    if finish.exists():
        return
    guard(0)
    assert OUT52.resolve() == PHYSICAL52.resolve(), 'phase2752 junction missing'
    assert shutil.disk_usage('D:/').free > 4 * 1024 ** 3
    folder.mkdir(parents=True, exist_ok=True)

    # ---- frozen design (before any behavioral observation) ----
    z_sub = np.load(OUT51_NPZ, allow_pickle=False)
    pair_idx = {p: z_sub['pooled__' + p + '__top1pct'].astype(np.int64) for p in PAIRS}
    union_flat = np.unique(np.concatenate([pair_idx[p] for p in PAIRS]))
    union = split_indices(union_flat)
    pair_split = {p: split_indices(pair_idx[p]) for p in PAIRS}
    pop_per_matrix = {'gate_proj.weight': GATE_N, 'up_proj.weight': UP_N,
                      'down_proj.weight': GATE_N}
    u_sizes = {k: int(len(union[k])) for k in MAT_KEYS}

    def build_random(rng):
        """Per-matrix random sets, same sizes as the union, disjoint from it."""
        out = {}
        for k in MAT_KEYS:
            mask = np.ones(pop_per_matrix[k], dtype=bool)
            mask[union[k]] = False
            cand = mask.nonzero()[0]
            out[k] = np.sort(rng.choice(cand, size=u_sizes[k], replace=False).astype(np.int64))
        return out

    random_set = build_random(np.random.default_rng(SEED_RANDOM))

    execution = {'source': snapshot(__file__),
                 'phase2751_subset_npz_sha256': sha(OUT51_NPZ),
                 'phase2747_training_result_sha256': sha(BASE / 'phase2747/training/result.json'),
                 'subset_sizes': {p: int(len(pair_idx[p])) for p in PAIRS},
                 'union_sizes_per_matrix': u_sizes,
                 'union_size_total': int(sum(u_sizes.values())),
                 'pair_union_coverage': {p: int(len(np.intersect1d(pair_idx[p], union_flat)))
                                         for p in PAIRS},
                 'random_set_sizes_per_matrix': {k: int(len(v)) for k, v in random_set.items()},
                 'random_shares_any_union_coord': {k: int(len(np.intersect1d(random_set[k], union[k])))
                                                   for k in MAT_KEYS},
                 'panel': 'phase2747 material freeze: validation + diagnostic + fresh, same order',
                 'panel_material_extension': {
                     'surface_class': 'classes[target] (deterministic; the exact rule used by '
                                      'the 2747 freeze for train rows)',
                     'permuted_target': 'within (kind, cohort, language, surface_class) group '
                                        'permutation over the 896-row panel, seed 2752001, '
                                        'same-label coincidences retained — mirrors the 2747 '
                                        'train-side permutation protocol',
                     'reason': 'the 2747 freeze assigned permuted_target/surface_class to train '
                               'rows only; the permuted/mass own objectives on the panel '
                               'require the same definitions there'},
                 'runs': RUNS, 'variants': VARIANTS,
                 'seeds': {'random_control': SEED_RANDOM,
                           'shuffle': str(SEED_SHUFFLE_BASE) + '+run_ordinal'},
                 'residual_policy': 'full-derived variants keep R; only_/ronly_ exclude R '
                                    '(bit-repair term of the full-state receipt, |R|~1e-7)',
                 'integrity_gate': 'base vs committed bridge baseline: exact array equality, '
                                   'else soft gate mean|dNLL|<0.05 AND argmax agreement>=0.99 '
                                   'AND mean |dEntropy|<0.05 AND mean |dSurfaceMass|<0.001 '
                                   '(GPU architecture may differ from the 2747 run)',
                 'criteria': {'P1_necessity': 'd_obj(pz_union) > d_obj(rzero_union) per run; '
                                              'support = 6/6',
                              'P2_sufficiency': 'red(only_union) > red(ronly_union) per run; '
                                                'support = 6/6',
                              'P3_placement': 'd_obj(shuffle_union) > 0; support >= 5/6',
                              'localisation_functional': 'P1 and P2 both 6/6'},
                 'question': 'Do the Phase2751 top-1% parameter subsets causally carry the '
                             'supervision-specific behavioral effect of the Phase2747 '
                             'checkpoints on their own training objective over the frozen '
                             '896-position panel?',
                 'status': 'frozen_before_any_Phase2752_intervention_observation'}
    exec_path = folder / 'execution.json'
    if exec_path.exists():
        old = read(exec_path)
        assert old['execution'] == execution, 'Phase2752 execution drift'
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

    # ---- frozen material extension: surface_class + permuted_target on panel rows ----
    classes_np = classes.cpu().numpy()
    for r in panel:
        r['surface_class'] = int(classes_np[int(r['target'])])
    groups_panel = defaultdict(list)
    for i, r in enumerate(panel):
        groups_panel[(r['kind'], r['cohort'], r['language'], r['surface_class'])].append(i)
    rng_mat = np.random.default_rng(2752001)
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
    print('PHASE2752_PROGRESS panel_extension', json.dumps(panel_ext), elapsed(), flush=True)

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
    first_base = eval_variant('true_token', panel)
    arrays = {'base__nll_target': first_base['nll_target'].astype(np.float64),
              'base__nll_permuted': first_base['nll_permuted'].astype(np.float64),
              'base__argmax_is_target': first_base['argmax_is_target'].astype(np.int8),
              'base__argmax_is_permuted': first_base['argmax_is_permuted'].astype(np.int8)}

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
    for run_ordinal, run in enumerate(RUNS):
        cond = condition_of(run)
        if cond not in metrics:
            set_state({k: original[k].numpy() for k in MAT_KEYS})
            base_ev = eval_variant(cond, panel)
            metrics['base__' + cond] = summarize(base_ev)
            arrays['base__' + cond + '__own'] = base_ev['own_objective'].astype(np.float64)
            print('PHASE2752_PROGRESS base', cond, round(metrics['base__' + cond]['own_mean'], 6),
                  elapsed(), flush=True)

        zd = np.load(TRAIN_PHYS / run / 'delta_128.npz', allow_pickle=False)
        D = {k: zd[k].astype(np.float32) for k in MAT_KEYS}
        R = {k: zd['reconstruction_residual__' + k].astype(np.float32) for k in MAT_KEYS}
        del zd
        full_state = {k: original[k].numpy() + D[k] + R[k] for k in MAT_KEYS}
        o_and_r = {k: original[k].numpy() + R[k] for k in MAT_KEYS}
        o_plus_d = {k: original[k].numpy() + D[k] for k in MAT_KEYS}

        states = {'full': full_state}
        for p in PAIRS:
            states['pz_' + p] = {k: full_state[k].copy() for k in MAT_KEYS}
            for k in MAT_KEYS:
                ix = pair_split[p][k]
                states['pz_' + p][k].flat[ix] = o_and_r[k].flat[ix]
        states['pz_union'] = {k: full_state[k].copy() for k in MAT_KEYS}
        for k in MAT_KEYS:
            ix = union[k]
            states['pz_union'][k].flat[ix] = o_and_r[k].flat[ix]
        states['rzero_union'] = {k: full_state[k].copy() for k in MAT_KEYS}
        for k in MAT_KEYS:
            ix = random_set[k]
            states['rzero_union'][k].flat[ix] = o_and_r[k].flat[ix]
        # only_* : keep D ONLY on the set; everything else back to original
        states['only_union'] = {k: original[k].numpy().copy() for k in MAT_KEYS}
        for k in MAT_KEYS:
            ix = union[k]
            states['only_union'][k].flat[ix] = o_plus_d[k].flat[ix]
        states['ronly_union'] = {k: original[k].numpy().copy() for k in MAT_KEYS}
        for k in MAT_KEYS:
            ix = random_set[k]
            states['ronly_union'][k].flat[ix] = o_plus_d[k].flat[ix]
        rng_sh = np.random.default_rng(SEED_SHUFFLE_BASE + run_ordinal)
        states['shuffle_union'] = {k: full_state[k].copy() for k in MAT_KEYS}
        for k in MAT_KEYS:
            ix = union[k]
            v = states['shuffle_union'][k]
            vals = v.flat[ix].copy()
            rng_sh.shuffle(vals)
            v.flat[ix] = vals

        for variant in VARIANTS:
            set_state(states[variant])
            ev = eval_variant(cond, panel)
            metrics.setdefault(run, {})[variant] = summarize(ev)
            arrays[run + '__' + variant + '__own'] = ev['own_objective'].astype(np.float64)
            arrays[run + '__' + variant + '__nll_target'] = ev['nll_target'].astype(np.float64)
            arrays[run + '__' + variant + '__nll_permuted'] = ev['nll_permuted'].astype(np.float64)
            print('PHASE2752_PROGRESS', run, variant,
                  round(metrics[run][variant]['own_mean'], 6), elapsed(), flush=True)
        del states, D, R, full_state, o_and_r, o_plus_d
        gc.collect()

    # ---- restore original exactly ----
    set_state({k: original[k].numpy() for k in MAT_KEYS})
    restored = evaluate(model, panel[:4], classes)
    restore_exact = {k: bool(np.array_equal(restored[k], base_check[k][:4])) for k in restored}
    assert all(restore_exact.values()), ('original state not restored bit-exactly', restore_exact)

    # ---- criteria ----
    results = {}
    p1_list, p2_list, p3_list = [], [], []
    for run in RUNS:
        cond = condition_of(run)
        m = metrics[run]
        base_own = metrics['base__' + cond]['own_mean']
        d_obj = {v: m[v]['own_mean'] - m['full']['own_mean'] for v in VARIANTS if v != 'full'}
        red = {v: base_own - m[v]['own_mean'] for v in VARIANTS}
        p1 = bool(d_obj['pz_union'] > d_obj['rzero_union'])
        p2 = bool(red['only_union'] > red['ronly_union'])
        p3 = bool(d_obj['shuffle_union'] > 0)
        p1_list.append(p1)
        p2_list.append(p2)
        p3_list.append(p3)
        results[run] = {'condition': cond,
                        'd_obj_vs_full': {k: float(x) for k, x in d_obj.items()},
                        'reduction_vs_base': {k: float(x) for k, x in red.items()},
                        'P1_necessity': p1, 'P2_sufficiency': p2, 'P3_placement': p3,
                        'per_pair_zero_effect': {p: float(d_obj['pz_' + p]) for p in PAIRS}}

    verdict = {'P1_necessity_runs': int(sum(p1_list)), 'P2_sufficiency_runs': int(sum(p2_list)),
               'P3_placement_runs': int(sum(p3_list)),
               'localisation_functional': bool(sum(p1_list) == 6 and sum(p2_list) == 6),
               'preregistered': execution['criteria']}

    value = {'timestamp': stamp(), 'source': snapshot(__file__),
             'all_passed': True, 'partial': partial,
             'status': 'P5 step 2 complete on preregistered criteria',
             'model': 'qwen4', 'gpu_name': gpu_name,
             'trainable_block': 'model.layers[16].mlp', 'runs': RUNS, 'variants': VARIANTS,
             'union_size_total': int(sum(u_sizes.values())),
             'panel_extension': panel_ext,
             'integrity_gate': integrity, 'restore_exact': restore_exact,
             'metrics': metrics, 'criteria_results': results, 'verdict': verdict,
             'subset_npz': str(OUT51_NPZ),
             'seconds': elapsed(),
             'limits': ['Interventions cover only the trained layers[16].mlp block; the rest of '
                        'the model is original BF16 throughout.',
                        'only_union/ronly_union exclude the bit-repair residual (|R|~1e-7); '
                        'full-derived variants keep it.',
                        'The random control is a single global set (seed 2752000), not '
                        're-drawn per run; shuffle seeds vary per run.',
                        'Objective effects are measured on the training-objective alignment of '
                        'each run; cross-condition transfer of subset effects is descriptive '
                        'only (P4 matrix), not a preregistered test.',
                        'Forward numerics may differ from the 2747 run if the GPU architecture '
                        'changed; the integrity gate handles this via the preregistered soft '
                        'path, and all variant comparisons are within-session.']}
    npz(folder / 'intervention_scores.npz', **arrays)
    save(folder / 'intervention_scores_meta.json',
         {'timestamp': stamp(), 'panel_sample_ids': [r['sample_id'] for r in panel],
          'families': families, 'runs': RUNS, 'variants': VARIANTS,
          'array_keys': sorted(arrays.keys())})
    save(finish, value)
    ledger('phase2752_subset_intervention', value['seconds'])
    print('PHASE2752_SUBSET_INTERVENTION', json.dumps(verdict), elapsed(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true')
    main(parser.parse_args().partial)
