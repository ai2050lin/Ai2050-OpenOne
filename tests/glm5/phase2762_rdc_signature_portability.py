"""Phase 2762: cross-model functional portability of the signature geometry -
2752-style sufficiency/necessity intervention run on the Qwen3-1.7B trained
checkpoints of Phase 2755.

Question (user request 3, feasibility-limited): 2751/2752 established on
qwen3-4b that the top-1% signature parameter subsets causally carry the
supervision-specific behavioural effect; 2755 reproduced the signature
GEOMETRY on qwen3-1.7b. Direct parameter transplant across models is not
defensible (different widths: 2560 vs 2048); portability is therefore tested
as FUNCTIONAL portability: on the second model, do its OWN 2755-derived
top-1% subsets carry the behavioural effect (sufficiency P2, necessity P1,
placement P3) under the identical 2752 protocol? Success = the 2752 causal
structure reproduces 6/6 runs on a different-width model, upgrading
"geometry reproduces" to "function transfers".

Protocol = Phase 2752 verbatim, adapted:
  model qwen3-1.7b (28 layers, block16 mlp, 37,748,736 FP32 scalars),
  checkpoints BASE/phase2755/qwen25_3b/<run>/delta_128.npz,
  subsets BASE/phase2755/qwen25_3b/pooled_top1pct_indices.npz
  (keys pooled__<pair>__top1pct, 1% each),
  panel = 2747 freeze validation+diagnostic+fresh (896 rows) with the same
  surface_class/permuted_target extension (seed 2752001),
  variants: full, pz_union, rzero_union, only_union, ronly_union, shuffle_union
  (pair-level pz_* omitted; recorded as a protocol reduction),
  criteria per run:
    P1 necessity   d_obj(pz_union) > d_obj(rzero_union)
    P2 sufficiency red(only_union) > red(ronly_union)
    P3 placement   d_obj(shuffle_union) > 0
    portable_functional = P1 6/6 AND P2 6/6.
Integrity gates: base eval determinism (bit-exact repeat), restore bit-exact
at end, soft cross-check vs 2755 recorded base own-objectives (|d|<0.05).
"""
import gc
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM

import rdc_construction_common as cc
import rdc_feature_common as fc
import phase2747_rdc_material as mat2747
from phase2747_rdc_training import bridge, packet, CONDITIONS

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2762' / 'qwen17_block16'
MODEL_DIR = ROOT / 'models/hf/qwen3-1.7b'
CKPT_DIR = BASE / 'phase2755' / 'qwen25_3b'
SUBSET_NPZ = CKPT_DIR / 'pooled_top1pct_indices.npz'
MAT_KEYS = ['gate_proj.weight', 'up_proj.weight', 'down_proj.weight']
GATE_N = 12_582_912
UP_N = 12_582_912
RUNS = ['true_token_2747', 'within_surface_class_permuted_token_2747',
        'surface_class_mass_2747',
        'true_token_2748', 'within_surface_class_permuted_token_2748',
        'surface_class_mass_2748']
PAIRS = ['true_vs_permuted', 'true_vs_mass', 'permuted_vs_mass']
VARIANTS = ['full', 'pz_union', 'rzero_union', 'only_union', 'ronly_union',
            'shuffle_union']
SEED_RANDOM = 2762000
SEED_SHUFFLE_BASE = 2762100
SEED_PANEL_PERM = 2752001
START = time.monotonic()


def elapsed():
    return round(time.monotonic() - START, 1)


def condition_of(run):
    return run.rsplit('_', 1)[0]


def split_indices(idx):
    g = idx[idx < GATE_N].astype(np.int64)
    u = idx[(idx >= GATE_N) & (idx < GATE_N + UP_N)].astype(np.int64) - GATE_N
    d = idx[idx >= GATE_N + UP_N].astype(np.int64) - GATE_N - UP_N
    return {'gate_proj.weight': np.sort(g), 'up_proj.weight': np.sort(u),
            'down_proj.weight': np.sort(d)}


def main():
    import shutil
    assert not (OUT / 'result.json').exists(), 'result.json immutable; delete before rerun'
    assert shutil.disk_usage('D:/').free > 4 * 1024 ** 3
    OUT.mkdir(parents=True, exist_ok=True)

    execution = {'source': cc.snapshot(__file__),
                 'subset_npz_sha256': cc.sha(SUBSET_NPZ),
                 'phase2755_result_sha256': cc.sha(CKPT_DIR / 'result.json'),
                 'criteria': {'P1_necessity': 'd_obj(pz_union) > d_obj(rzero_union) '
                                              'per run; support = 6/6',
                              'P2_sufficiency': 'red(only_union) > red(ronly_union) '
                                                'per run; support = 6/6',
                              'P3_placement': 'd_obj(shuffle_union) > 0; >= 5/6',
                              'portable_functional': 'P1 and P2 both 6/6'},
                 'protocol_reduction': 'pair-level pz_* variants omitted vs 2752 '
                                       '(union-level tests retained); seeds renamed '
                                       '(random 2762000, shuffle 2762100+ordinal)',
                 'status': 'frozen_before_any_intervention_observation'}
    fc.save(OUT / 'execution.json', execution)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map={'': 'cuda:0'},
        attn_implementation='eager', local_files_only=True).eval()
    assert model.dtype == torch.bfloat16
    assert model.config.num_hidden_layers == 28 and model.config.hidden_size == 2048 \
        and model.config.intermediate_size == 6144

    material, data = mat2747.freeze()
    with np.load(ROOT / material['vocabulary_receipt']['field_path']) as z:
        classes = torch.tensor(z['classes'].astype(np.int64), device='cuda')
    panel = data['validation'] + data['diagnostic'] + data['fresh']
    assert len(panel) == 896

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
    for i, r in enumerate(panel):
        r['permuted_target'] = panel[int(permutation[i])]['target']
        assert classes_np[r['target']] == classes_np[r['permuted_target']]

    target = model.model.layers[16].mlp
    with torch.no_grad():
        target.float()
    named = dict(target.named_parameters())
    assert sum(p.numel() for p in named.values()) == 37_748_736
    original = {k: named[k].detach().cpu().clone() for k in MAT_KEYS}
    handles = bridge(target)

    # ---- frozen subset design ----
    z_sub = np.load(SUBSET_NPZ, allow_pickle=False)
    pair_idx = {p: z_sub['pooled__' + p + '__top1pct'].astype(np.int64) for p in PAIRS}
    union_flat = np.unique(np.concatenate([pair_idx[p] for p in PAIRS]))
    union = split_indices(union_flat)
    pop_per_matrix = {'gate_proj.weight': GATE_N, 'up_proj.weight': UP_N,
                      'down_proj.weight': GATE_N}
    u_sizes = {k: int(len(union[k])) for k in MAT_KEYS}
    random_set = {}
    for k in MAT_KEYS:
        mask = np.ones(pop_per_matrix[k], dtype=bool)
        mask[union[k]] = False
        cand = mask.nonzero()[0]
        random_set[k] = np.sort(np.random.default_rng(SEED_RANDOM).choice(
            cand, size=u_sizes[k], replace=False).astype(np.int64))
    assert all(len(np.intersect1d(random_set[k], union[k])) == 0 for k in MAT_KEYS)

    def set_state(state):
        with torch.no_grad():
            for k in MAT_KEYS:
                named[k].copy_(torch.from_numpy(np.ascontiguousarray(state[k]))
                               .to(named[k].device))

    def eval_variant(run_condition, rows):
        acc = defaultdict(list)
        with torch.no_grad():
            for row in rows:
                post, z = packet(model, row)
                lp = z.double().log_softmax(-1)
                acc['nll_target'].append(float(-lp[int(row['target'])]))
                acc['nll_permuted'].append(float(-lp[int(row['permuted_target'])]))
                am = int(z.argmax())
                acc['argmax_is_target'].append(am == int(row['target']))
                if run_condition == 'surface_class_mass':
                    own = float(-torch.logsumexp(lp[classes == int(row['surface_class'])], dim=0))
                elif run_condition == 'true_token':
                    own = float(-lp[int(row['target'])])
                else:
                    own = float(-lp[int(row['permuted_target'])])
                acc['own_objective'].append(own)
        return {k: np.asarray(v) for k, v in acc.items()}

    # ---- integrity: determinism + soft cross-check vs 2755 base ----
    set_state({k: original[k].numpy() for k in MAT_KEYS})
    base_a = eval_variant('true_token', panel)
    base_b = eval_variant('true_token', panel)
    det = all(bool(np.array_equal(base_a[k], base_b[k])) for k in base_a)
    assert det, 'base eval not deterministic'
    arrays = {'base__true_token__own': base_a['own_objective'].astype(np.float64)}
    soft = {}
    r2755 = fc.read(CKPT_DIR / 'result.json')
    base_own_2755 = {c: r2755['runs']['%s_2747' % c]['base_own_mean']
                     for c in CONDITIONS}
    for c in CONDITIONS:
        ev = base_a if c == 'true_token' else eval_variant(c, panel)
        if c != 'true_token':
            arrays['base__%s__own' % c] = ev['own_objective'].astype(np.float64)
        if c in base_own_2755:
            soft[c] = float(abs(ev['own_objective'].mean() - base_own_2755[c]))
            assert soft[c] < 0.05, ('2755 base own-objective drift', c, soft[c])
    print('P2762 base determinism OK; soft drift vs 2755: %s' % soft, elapsed(), flush=True)

    families = [r['family'] for r in panel]

    def summarize(ev):
        return {'own_mean': float(ev['own_objective'].mean()),
                'nll_target_mean': float(ev['nll_target'].mean()),
                'argmax_target_rate': float(ev['argmax_is_target'].mean())}

    metrics = {}
    for run in RUNS:
        cond = condition_of(run)
        metrics['base__' + cond] = {'own_mean': float(
            arrays['base__%s__own' % cond].mean())}

        zd = np.load(CKPT_DIR / run / 'delta_128.npz', allow_pickle=False)
        D = {k: zd[k].astype(np.float32) for k in MAT_KEYS}
        R = {k: zd['reconstruction_residual__' + k].astype(np.float32) for k in MAT_KEYS}
        del zd
        full_state = {k: original[k].numpy() + D[k] + R[k] for k in MAT_KEYS}
        o_and_r = {k: original[k].numpy() + R[k] for k in MAT_KEYS}
        o_plus_d = {k: original[k].numpy() + D[k] for k in MAT_KEYS}

        states = {'full': full_state}
        states['pz_union'] = {k: full_state[k].copy() for k in MAT_KEYS}
        for k in MAT_KEYS:
            ix = union[k]
            states['pz_union'][k].flat[ix] = o_and_r[k].flat[ix]
        states['rzero_union'] = {k: full_state[k].copy() for k in MAT_KEYS}
        for k in MAT_KEYS:
            ix = random_set[k]
            states['rzero_union'][k].flat[ix] = o_and_r[k].flat[ix]
        states['only_union'] = {k: original[k].numpy().copy() for k in MAT_KEYS}
        for k in MAT_KEYS:
            ix = union[k]
            states['only_union'][k].flat[ix] = o_plus_d[k].flat[ix]
        states['ronly_union'] = {k: original[k].numpy().copy() for k in MAT_KEYS}
        for k in MAT_KEYS:
            ix = random_set[k]
            states['ronly_union'][k].flat[ix] = o_plus_d[k].flat[ix]
        rng_sh = np.random.default_rng(SEED_SHUFFLE_BASE + RUNS.index(run))
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
            arrays['%s__%s__own' % (run, variant)] = ev['own_objective'].astype(np.float64)
            print('P2762_PROGRESS', run, variant,
                  round(metrics[run][variant]['own_mean'], 6), elapsed(), flush=True)
        del states, D, R, full_state, o_and_r, o_plus_d
        gc.collect()

    set_state({k: original[k].numpy() for k in MAT_KEYS})
    restored = eval_variant('true_token', panel[:8])
    restore_exact = bool(np.array_equal(restored['own_objective'],
                                        base_a['own_objective'][:8]))
    assert restore_exact, 'original state not restored bit-exactly'

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
                        'P1_necessity': p1, 'P2_sufficiency': p2, 'P3_placement': p3}

    verdict = {'P1_necessity_runs': int(sum(p1_list)),
               'P2_sufficiency_runs': int(sum(p2_list)),
               'P3_placement_runs': int(sum(p3_list)),
               'portable_functional': bool(sum(p1_list) == 6 and sum(p2_list) == 6)}
    value = {'timestamp': fc.stamp(), 'model': 'qwen3-1.7b', 'gpu': torch.cuda.get_device_name(0),
             'runs': RUNS, 'variants': VARIANTS,
             'union_size_total': int(sum(u_sizes.values())),
             'base_soft_drift_vs_2755': soft,
             'metrics': metrics, 'criteria_results': results, 'verdict': verdict,
             'subset_npz': str(SUBSET_NPZ),
             'seconds': elapsed(),
             'limits': ['only_union/ronly_union exclude the bit-repair residual '
                        '(|R|~1e-7); full-derived variants keep it (2752 policy).',
                        'Single random control set (seed 2762000), not re-drawn per run.',
                        'Second model shares the qwen3 tokenizer family; cross-tokenizer '
                        'transfer remains untested (Qwen2.5-3B was rejected at 2755).',
                        'Subsets are the model-own 2755 top-1% unions - this tests '
                        'functional portability of the causal structure, not literal '
                        'coordinate transfer across widths.']}
    fc.npz(OUT / 'intervention_scores.npz', **arrays)
    fc.save(OUT / 'intervention_scores_meta.json',
            {'panel_sample_ids': [r['sample_id'] for r in panel],
             'families': families, 'runs': RUNS, 'variants': VARIANTS,
             'array_keys': sorted(arrays.keys())})
    fc.save(OUT / 'result.json', value)
    print('PHASE2762_DONE ' + json.dumps(verdict), elapsed(), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        import traceback
        (OUT / 'crash.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
