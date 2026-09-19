"""Phase2751: P5 step 1 — parameter-direction pair subset localisation at gate/up/down scalars.

V2 plan P5 (three-step): (1) localise the parameter-subset differences between the
direction pairs (correct-supervised true_token vs shuffle-supervised
within_surface_class_permuted_token vs class-supervised surface_class_mass),
resolved to gate/up/down scalars; (2) directional freeze/shuffle interventions;
(3) cross-model transfer.  This phase executes step 1 on the six audited
Phase2747 training checkpoints (3 supervision conditions x 2 material
generations, single trainable block = layers[16].mlp, 74,711,040 FP32 scalars).

All coordinates are used; no truncation, no Top-K readout of activations, no
model loads.  A coordinate subset is called directional only if the pair
difference field reproduces across the two independent material generations
(pre-registered thresholds: cross-generation Pearson >= 0.5 AND top-1%
absolute-difference Jaccard >= 0.3).
"""
import argparse
import itertools
import json
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import rdc_construction_common
from rdc_construction_common import (BASE, stamp, sha, read, save, snapshot, guard,
                                     ledger, npz)

OUT51 = BASE / 'phase2751'
PHYSICAL51 = BASE / 'phase2751'
TRAIN = BASE / 'phase2747_fields/training'
TRAIN_RESULT = BASE / 'phase2747/training'
RUNS = ['true_token_2747', 'within_surface_class_permuted_token_2747', 'surface_class_mass_2747',
        'true_token_2748', 'within_surface_class_permuted_token_2748', 'surface_class_mass_2748']
MATRICES = [('gate', 'gate_proj.weight', (9728, 2560)),
            ('up', 'up_proj.weight', (9728, 2560)),
            ('down', 'down_proj.weight', (2560, 9728))]
PAIRS = [('true_vs_permuted', 'true_token', 'within_surface_class_permuted_token'),
         ('true_vs_mass', 'true_token', 'surface_class_mass'),
         ('permuted_vs_mass', 'within_surface_class_permuted_token', 'surface_class_mass')]
KS = [0.001, 0.01, 0.05, 0.10]
PREREG = {'pearson_min': 0.5, 'jaccard_top1pct_min': 0.3}
FIELDS_BLOCK = 16
START = time.monotonic()


def elapsed():
    return round(time.monotonic() - START, 1)


def guard51(expected=0):
    guard(expected)
    assert OUT51.resolve() == PHYSICAL51.resolve()
    import shutil
    assert shutil.disk_usage('D:/').free - expected > 4 * 1024 ** 3


def load_delta(run, key):
    """Exact FP32 update = stored delta value (actual - original); the
    reconstruction_residual array is a bit-level repair term, not the delta."""
    z = np.load(TRAIN / run / 'delta_128.npz', allow_pickle=False)
    a = z[key]
    assert a.dtype == np.float32, (run, key, a.dtype)
    out = a.astype(np.float64)
    del z
    return out


def cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def top_k_jaccard(d1, d2, k):
    n1 = set(np.argsort(np.abs(d1))[-k:].tolist())
    n2 = set(np.argsort(np.abs(d2))[-k:].tolist())
    return len(n1 & n2) / max(1, len(n1 | n2))


def main(partial=False):
    folder = OUT51 / 'qwen4_block16'
    finish = folder / ('partial_' + str(time.time_ns()) + '.json' if partial else 'result.json')
    if finish.exists():
        return
    guard51()
    folder.mkdir(parents=True, exist_ok=True)

    execution = {'source': snapshot(__file__),
                 'construction_common': snapshot(Path(rdc_construction_common.__file__)),
                 'phase2747_training_result_sha256': sha(TRAIN_RESULT / 'result.json'),
                 'phase2748_parameter_formation_sha256': sha(BASE / 'phase2748/parameter_formation/result.json'),
                 'phase2747_training_source_sha256': sha(TRAIN_RESULT.parent.parent.parent
                                                         / 'tests/glm5/phase2747_rdc_training.py')
                 if False else sha(Path(r'D:\AI2050\Ai2050-OpenOne\tests\glm5\phase2747_rdc_training.py'))}
    exec_path = folder / 'execution.json'
    if exec_path.exists():
        old = read(exec_path)
        assert old['execution'] == execution, 'Phase2751 execution drift'
    else:
        save(exec_path, {'timestamp': stamp(), 'execution': execution,
            'status': 'frozen_before_any_Phase2751_delta_observation',
            'question': 'P5 step 1: is there a parameter-coordinate subset of the trainable '
                        'layers[16].mlp gate/up/down block whose training-update difference between '
                        'supervision directions (correct true_token vs shuffle '
                        'within_surface_class_permuted_token vs class surface_class_mass) reproduces '
                        'across the two independent material generations 2747 and 2748?',
            'trainable_block': 'model.layers[16].mlp (74,711,040 FP32 scalars; all other parameters '
                               'bit-frozen per the Phase2747 unchanged-parameter receipts)',
            'runs': RUNS,
            'exact_delta': 'delta_128.npz values are exact FP32 (actual - original); the '
                           'reconstruction_residual arrays are bit-level repair terms excluded from '
                           'the mathematical delta',
            'analysis': 'Module-level direction cosine matrix (6x6 per matrix and pooled; '
                        'pooled dot products accumulated per matrix, equivalent to the '
                        'concatenation dot product); '
                        'direction-pair difference fields D_true-permuted, D_true-mass, '
                        'D_permuted-mass per generation with cross-generation Pearson and cosine; '
                        'pooled difference fields are the CONCATENATION of the gate/up/down '
                        'flattened fields (74,711,040 scalars in matrix order gate, up, down) '
                        '— the three matrices have equal parameter counts, so any elementwise '
                        'accumulation across matrices would silently mix distinct coordinates; '
                        'top-k (0.1/1/5/10%) absolute-difference Jaccard across generations; '
                        'neuron-level tri-matrix consistency (gate row, up row, down column); '
                        'difference magnitude vs same-condition cross-generation noise quantiles; '
                        'descriptive link of subset neurons to the 44-frozen-expression block16 '
                        'gate/product median activations.',
            'preregistered_criteria': {
                'subset_stable': 'cross-generation Pearson >= 0.5 AND top-1% Jaccard >= 0.3 '
                                 'for a direction pair (pooled 74,711,040 scalars)',
                'localisation_supported': 'at least one direction pair is subset_stable (pooled)',
                'rejected_if': 'all direction pairs fail both thresholds'}})

    matrices_report = {}
    pooled_stats = {'dot': defaultdict(float),
                    'parts': {'d2747': defaultdict(list), 'd2748': defaultdict(list),
                              'noise': defaultdict(list)}}
    neuron_scores = {p[0]: {'gate': None, 'up': None, 'down': None} for p in PAIRS}

    for mat_name, key, shape in MATRICES:
        deltas = {r: load_delta(r, key) for r in RUNS}
        flat = {r: deltas[r].reshape(-1) for r in RUNS}
        n = flat[RUNS[0]].shape[0]
        cmat = {f'{a}|{b}': cos(flat[a], flat[b]) for a, b in itertools.combinations(RUNS, 2)}
        pair_blob = {}
        for r in RUNS:
            pooled_stats['dot'][r] += float(np.dot(flat[r], flat[r]))
        for pname, ca, cb in PAIRS:
            d47 = flat[f'{ca}_2747'] - flat[f'{cb}_2747']
            d48 = flat[f'{ca}_2748'] - flat[f'{cb}_2748']
            noise = flat[f'{ca}_2747'] - flat[f'{ca}_2748']
            for field, vec in (('d2747', d47), ('d2748', d48), ('noise', noise)):
                # store per-matrix parts; pooled = concatenation over [gate|up|down]
                # (the matrices have equal parameter counts, so elementwise addition
                # would silently mix three different parameter coordinates)
                pooled_stats['parts'][field][pname].append(vec.astype(np.float32))
            jac = {f'{k:g}': top_k_jaccard(d47, d48, int(round(k * n))) for k in KS}
            q = [10, 25, 50, 75, 90, 99]
            pair_blob[pname] = {
                'cross_generation_pearson': float(np.corrcoef(d47, d48)[0, 1]),
                'cross_generation_cosine': cos(d47, d48),
                'topk_jaccard': jac,
                'magnitude': {
                    'difference_field_abs_quantiles': {str(x): float(np.quantile(np.abs(d47), x / 100))
                                                       for x in q},
                    'crossgen_noise_abs_quantiles': {str(x): float(np.quantile(np.abs(noise), x / 100))
                                                     for x in q},
                    'energy_ratio_D_to_noise': float(np.sum(d47 ** 2) / max(np.sum(noise ** 2), 1e-30)),
                    'energy_ratio_D_to_updates': float(
                        np.sum(d47 ** 2) / max(np.sum(flat[f'{ca}_2747'] ** 2)
                                               + np.sum(flat[f'{cb}_2747'] ** 2), 1e-30))}}
            if mat_name == 'gate':
                neuron_scores[pname]['gate'] = np.sqrt(
                    (d47.reshape(shape) ** 2).sum(axis=1))
            if mat_name == 'up':
                neuron_scores[pname]['up'] = np.sqrt(
                    (d47.reshape(shape) ** 2).sum(axis=1))
            if mat_name == 'down':
                neuron_scores[pname]['down'] = np.sqrt(
                    (d47.reshape(shape) ** 2).sum(axis=0))
        matrices_report[mat_name] = {'shape': list(shape), 'direction_cosines': cmat,
                                     'pairs': pair_blob}
        for a, b in itertools.combinations(RUNS, 2):
            pooled_stats['dot'][f'{a}|{b}'] = pooled_stats['dot'].get(f'{a}|{b}', 0.0) \
                + float(np.dot(flat[a], flat[b]))
        del deltas, flat

    pooled_cos = {}
    for a, b in itertools.combinations(RUNS, 2):
        num = pooled_stats['dot'][f'{a}|{b}']
        den = np.sqrt(pooled_stats['dot'][a] * pooled_stats['dot'][b])
        pooled_cos[f'{a}|{b}'] = float(num / den) if den else 0.0
    pooled_report = {'scalars': 74_711_040, 'direction_cosines': pooled_cos, 'pairs': {}}
    pooled_pairs = {}
    pooled_top = {}

    def chunked_pair_stats(u32, v32, chunk=8_000_000):
        sx = sy = sxx = syy = sxy = 0.0
        for i in range(0, u32.shape[0], chunk):
            x = u32[i:i + chunk].astype(np.float64)
            y = v32[i:i + chunk].astype(np.float64)
            sx += x.sum(); sy += y.sum()
            sxx += np.dot(x, x); syy += np.dot(y, y); sxy += np.dot(x, y)
        n = float(u32.shape[0])
        cov = sxy - sx * sy / n
        vx = sxx - sx * sx / n
        vy = syy - sy * sy / n
        pear = float(cov / np.sqrt(vx * vy)) if vx > 0 and vy > 0 else 0.0
        cs = float(sxy / np.sqrt(sxx * syy)) if sxx > 0 and syy > 0 else 0.0
        return pear, cs

    def chunked_sq(v32, chunk=8_000_000):
        s = 0.0
        for i in range(0, v32.shape[0], chunk):
            x = v32[i:i + chunk].astype(np.float64)
            s += float(np.dot(x, x))
        return s

    for pname, ca, cb in PAIRS:
        d47 = np.concatenate(pooled_stats['parts']['d2747'][pname])
        d48 = np.concatenate(pooled_stats['parts']['d2748'][pname])
        assert d47.shape[0] == 74_711_040 and d48.shape[0] == 74_711_040
        pear, cs = chunked_pair_stats(d47, d48)
        n = d47.shape[0]
        order47 = np.argsort(np.abs(d47))
        order48 = np.argsort(np.abs(d48))
        jac = {}
        for k in KS:
            kk = int(round(k * n))
            s1 = set(order47[-kk:].tolist())
            s2 = set(order48[-kk:].tolist())
            jac[f'{k:g}'] = len(s1 & s2) / max(1, len(s1 | s2))
        stable = (pear >= PREREG['pearson_min'] and jac['0.01'] >= PREREG['jaccard_top1pct_min'])
        noise = np.concatenate(pooled_stats['parts']['noise'][pname])
        noise_sq = chunked_sq(noise)
        d47_sq = chunked_sq(d47)
        pooled_pairs[pname] = {'cross_generation_pearson': pear,
                               'cross_generation_cosine': cs,
                               'topk_jaccard': jac, 'subset_stable': stable,
                               'energy_ratio_D_to_noise': float(d47_sq / max(noise_sq, 1e-30))}
        pooled_top[pname] = order47[-int(round(0.01 * n)):].copy()
        del d47, d48, noise, order47, order48
    pooled_report['pairs'] = pooled_pairs
    del pooled_stats['parts']

    # neuron-level tri-matrix consistency (2747 fields)
    neuron = {}
    for pname, _, _ in PAIRS:
        gs = neuron_scores[pname]['gate']
        us = neuron_scores[pname]['up']
        ds = neuron_scores[pname]['down']
        kg = set(np.argsort(gs)[-int(round(0.01 * gs.shape[0])):].tolist())
        ku = set(np.argsort(us)[-int(round(0.01 * us.shape[0])):].tolist())
        kd = set(np.argsort(ds)[-int(round(0.01 * ds.shape[0])):].tolist())
        neuron[pname] = {
            'gate_top1pct_rows': len(kg), 'up_top1pct_rows': len(ku),
            'down_top1pct_cols': len(kd),
            'tri_overlap_neurons': len(kg & ku & kd),
            'any2_overlap_neurons': len((kg & ku) | (kg & kd) | (ku & kd)),
            'expected_if_independent': 9728 * 0.01 ** 3,
            'gate_row_top_neurons': sorted(kg)[:64], 'up_row_top_neurons': sorted(ku)[:64],
            'down_col_top_neurons': sorted(kd)[:64]}

    # descriptive: block16 activations over the 44 frozen expressions
    fields_link = {}
    try:
        zf = np.load(TRAIN / 'true_token_2747' / 'deployed_BF16_fields.npz', allow_pickle=False)
        gate_f = np.abs(zf['gate'][:, FIELDS_BLOCK, :].astype(np.float64))
        prod_f = np.abs(zf['product'][:, FIELDS_BLOCK, :].astype(np.float64))
        act_gate = np.median(gate_f, axis=0)
        act_prod = np.median(prod_f, axis=0)
        kg = set(neuron['true_vs_permuted']['gate_row_top_neurons'])
        allg = set(range(9728))
        top_rows = sorted(kg)
        rest = sorted(allg - kg)
        fields_link = {'scope': 'descriptive; block16 gate/product median |activation| over the 44 '
                                'frozen expressions, gate top-1% true_vs_permuted difference rows vs '
                                'the remaining rows',
                       'n_top_rows': len(top_rows),
                       'gate_act_top_median': float(np.median(act_gate[top_rows])),
                       'gate_act_rest_median': float(np.median(act_gate[rest])),
                       'product_act_top_median': float(np.median(act_prod[top_rows])),
                       'product_act_rest_median': float(np.median(act_prod[rest]))}
        del zf, gate_f, prod_f
    except Exception as exc:
        fields_link = {'error': str(exc)}

    stable_pairs = {k: v['subset_stable'] for k, v in pooled_pairs.items()}
    verdict = {'subset_stable_by_pair_pooled': stable_pairs,
               'localisation_supported': any(stable_pairs.values()),
               'preregistered_thresholds': PREREG}

    value = {'timestamp': stamp(), 'source': snapshot(__file__),
             'all_passed': True, 'partial': partial,
             'status': 'P5 step 1 complete on preregistered criteria',
             'model': 'qwen4', 'trainable_block': 'model.layers[16].mlp',
             'runs': RUNS, 'delta_source': 'delta_128.npz (exact FP32 actual-original)',
             'verdict': verdict,
             'matrices': matrices_report, 'pooled': pooled_report,
             'neuron_level_2747': neuron,
             'fields_link_descriptive': fields_link,
             'subset_npz': 'phase2751/qwen4_block16/subset_indices.npz',
             'seconds': round(time.monotonic() - START, 1),
             'limits': ['Step 1 only: no intervention (step 2) and no cross-model transfer '
                        '(step 3) is attempted here.',
                        'One trainable block by design (layers[16].mlp); conclusions do not speak '
                        'to frozen parameters.',
                        'Two material generations give two independent delta realisations per '
                        'condition; no third generation exists for a holdout replication.',
                        'The fields link is descriptive (44 frozen expressions, block16 medians), '
                        'not a functional test.',
                        '2748 parameter_formation remains the authoritative checkpoint-integrity '
                        'receipt chain; this phase adds the pair-difference localisation layer.']}

    arrays = {}
    for pname, idx in pooled_top.items():
        arrays['pooled__' + pname + '__top1pct'] = idx.astype(np.int64)
    npz(folder / 'subset_indices.npz', **arrays)
    save(finish, value)
    ledger('phase2751_parameter_direction_subset', value['seconds'])
    print('PHASE2751_PARAMETER_DIRECTION_SUBSET', json.dumps(verdict), elapsed(), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true')
    main(parser.parse_args().partial)
