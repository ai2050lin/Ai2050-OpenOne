"""Phase2755: P5 step 3a — cross-model signature replication on Qwen2.5-3B-Instruct.

Question (frozen before any observation): does the Phase2751 signature geometry —
supervision-condition delta directions that are cross-generation stable, occupy
near-orthogonal per-condition subspaces, and concentrate on a top-1% subset —
replicate when the identical Phase2747 training protocol (same frozen material,
same seeds, same 128x16 schedule, same normalized FP32 steps, same layer-16 MLP
block) is run on a second, differently pretrained model?

Model: Qwen/Qwen3-1.7B, unquantized BF16 safetensors (hf-mirror direct download
2026-09-15); Qwen3ForCausalLM, hidden 2048, intermediate 6144, 28 layers, vocab
151936, tie_word_embeddings=true.  Trainable block = model.layers[16].mlp,
3*2048*6144 = 37,748,736 scalars (gate=up=down=12,582,912).
Selection history (preregistered): the first candidate Qwen2.5-3B-Instruct was
rejected by the tokenizer gate itself — material ids 151667/151668 decode as
Qwen3 <think>/</think> but differ under the Qwen2.5 tokenizer, so the frozen
pre-tokenized material does not transfer across those conventions.  Qwen3-1.7B
shares the qwen3-4b tokenizer family exactly; it is a genuinely different
network (1.7B vs 4B parameters, 28 vs 36 layers, different weights).

Tokenizer gate (preregistered): every unique material token id must decode to
the identical token string under the qwen3-4b and Qwen2.5-3B tokenizers.
Otherwise the material's pre-tokenized ids/classes do not transfer; abort.

Protocol (verbatim Phase2747): seeds [2747, 2748] over the same frozen 2048
training rows (identical draws), 3 conditions, 128 steps x batch 16 with
per-example autograd.grad mean accumulation, FP32 normalized step (-0.02/||g||),
checkpoints [1, 8, 32, 128] archived as bit-exact delta + reconstruction
residual float32 npz.  Training data, seeds, draws, step rule are identical to
Phase2747; only the host model differs.

Preregistered criteria:
  G1_replication   all 3 direction pairs: cross-generation pooled (concatenated
                   gate|up|down) Pearson >= 0.5 AND top-1% Jaccard >= 0.3
                   (Phase2751 thresholds); support = 3/3.
  G2_tripartite    per generation, pooled same-generation cross-condition
                   cosines (true|permuted, true|mass, permuted|mass) all < 0.5;
                   support = 2/2 generations.
  G3_concentration per pair, energy ratio E(||D||^2)/E(||noise||^2) >= 2 with
                   D = pair difference field of one generation and noise =
                   cross-generation difference of one condition; support = 3/3.
  B1_behavior      own-objective improvement (bridge base - final state) > 0 in
                   >= 5/6 runs on the 896-question panel.
Descriptive: per-matrix 6x6 cosine matrices, pooled 6x6, neuron-level pair
intersections of top-1% gate rows / up rows / down columns, side-by-side
comparison with the archived Phase2751 qwen4 numbers.

Storage: direct on D: under BASE/phase2755; no junction.
"""
import gc
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rdc_construction_common import (BASE, ROOT, stamp, sha, read, save, snapshot,
                                     guard, ledger, npz)
from phase2747_rdc_training import (bridge, evaluate, packet, mean_gradient, CONDITIONS)
import phase2747_rdc_material as mat2747

OUT55 = BASE / 'phase2755'
MODEL_DIR = ROOT / 'models/hf/qwen3-1.7b'
SEEDS = [2747, 2748]
STEPS = 128
BATCH = 16
CHECKPOINTS = [1, 8, 32, 128]
STEP_NORM = 0.02
BLOCK = 'model.layers[16].mlp'
BLOCK_PARAMS = 37_748_736
PER_MATRIX = 12_582_912
SEED_PANEL_PERM = 2752001
PREREG = {'pearson_min': 0.5, 'jaccard_top1pct_min': 0.3,
          'cross_condition_cosine_max': 0.5, 'energy_ratio_min': 2.0,
          'behavior_runs_min': 5}
RESULT51 = BASE / 'phase2751/qwen4_block16/result.json'
MAT_KEYS = ['gate_proj.weight', 'up_proj.weight', 'down_proj.weight']
PAIR_DEFS = [('true_vs_permuted', 'true_token', 'within_surface_class_permuted_token'),
             ('true_vs_mass', 'true_token', 'surface_class_mass'),
             ('permuted_vs_mass', 'within_surface_class_permuted_token', 'surface_class_mass')]
KS = [0.001, 0.005, 0.01, 0.05]
EXPECTED_RUNS = [c + '_' + str(s) for s in SEEDS for c in CONDITIONS]
START = time.monotonic()


def elapsed():
    return round(time.monotonic() - START, 1)


def cos64(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.dot(a, b) / (np.dot(a, a) * np.dot(b, b)) ** 0.5)


def main():
    folder = OUT55 / 'qwen25_3b'
    finish = folder / 'result.json'
    if finish.exists():
        return
    guard(0)
    assert shutil.disk_usage('D:/').free > 15 * 1024 ** 3
    folder.mkdir(parents=True, exist_ok=True)

    # ---- frozen design (before any model forward or training observation) ----
    execution = {'source': snapshot(__file__),
                 'model_dir': str(MODEL_DIR),
                 'model_config_sha256': sha(MODEL_DIR / 'config.json'),
                 'phase2747_training_result_sha256': sha(BASE / 'phase2747/training/result.json'),
                 'phase2751_result_sha256': sha(RESULT51),
                 'storage': 'direct on D: under BASE; no junction',
                 'block': BLOCK, 'block_params': BLOCK_PARAMS,
                 'per_matrix_params': PER_MATRIX,
                 'protocol': {'seeds': SEEDS, 'conditions': CONDITIONS, 'steps': STEPS,
                              'batch_examples': BATCH, 'checkpoints': CHECKPOINTS,
                              'step_FP32_norm': STEP_NORM,
                              'draws': 'np.random.default_rng(seed).permutation(2048).reshape(128,16) '
                                       'over the frozen Phase2747 train rows (identical to Phase2747)'},
                 'tokenizer_gate': 'every unique material token id decodes to the identical '
                                   'token string under qwen3-4b and Qwen3-1.7B tokenizers; '
                                   'abort otherwise (first candidate Qwen2.5-3B-Instruct was '
                                   'rejected by this gate: ids 151667/151668 = Qwen3 '
                                   '<think>/</think> differ under Qwen2.5)',
                 'criteria': {
                     'G1_replication': 'pooled cross-generation Pearson >= 0.5 and top-1% '
                                       'Jaccard >= 0.3 for all 3 pairs (Phase2751 thresholds)',
                     'G2_tripartite': 'per generation all 3 pooled cross-condition cosines < 0.5; '
                                      'support = 2/2',
                     'G3_concentration': 'energy ratio D/noise >= 2 for all 3 pairs',
                     'B1_behavior': 'own-objective improvement > 0 in >= 5/6 runs'},
                 'status': 'frozen_before_any_model_forward'}
    exec_path = folder / 'execution.json'
    if exec_path.exists():
        old = read(exec_path)
        assert old['execution'] == execution, 'Phase2755 execution drift'
    else:
        save(exec_path, {'timestamp': stamp(), 'execution': execution})

    # ---- tokenizer gate ----
    tok_new = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
    tok_old = AutoTokenizer.from_pretrained(ROOT / 'models/hf/qwen3-4b', local_files_only=True, use_fast=True)
    material, data = mat2747.freeze()
    ids_all = set()
    for part in ('train', 'validation', 'diagnostic', 'fresh'):
        for r in data[part]:
            ids_all.update(r['ids'])
    mism = [int(i) for i in sorted(ids_all) if tok_new.decode([int(i)]) != tok_old.decode([int(i)])]
    gate = {'n_ids': len(ids_all), 'mismatches': len(mism), 'mismatch_examples': mism[:10]}
    assert not mism, ('tokenizer gate failed', gate)
    print('PHASE2755_PROGRESS tokenizer_gate_ok n_ids=%d' % len(ids_all), elapsed(), flush=True)

    # ---- model (BF16 native baseline first, then FP32-block bridge) ----
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map={'': 'cuda:0'},
        attn_implementation='eager', local_files_only=True).eval()
    assert model.dtype == torch.bfloat16 and not getattr(model, 'is_quantized', False)
    assert all(p.device.type == 'cuda' for p in model.parameters())
    assert model.config.num_hidden_layers == 28 and model.config.hidden_size == 2048 \
        and model.config.intermediate_size == 6144
    with np.load(ROOT / material['vocabulary_receipt']['field_path']) as z:
        classes = torch.tensor(z['classes'].astype(np.int64), device='cuda')
    panel = data['validation'] + data['diagnostic'] + data['fresh']
    assert len(panel) == 896

    native_eval = evaluate(model, panel, classes)   # pure BF16, block untouched

    # panel material extension (identical to Phase2752/2753)
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

    # FP32-block bridge; capture parameter handles only AFTER the upcast
    target = model.model.layers[16].mlp
    with torch.no_grad():
        target.float()
    named = dict(target.named_parameters())
    assert sum(p.numel() for p in named.values()) == BLOCK_PARAMS
    original = {k: named[k].detach().cpu().clone() for k in MAT_KEYS}
    handles = bridge(target)
    for p in target.parameters():
        p.requires_grad_(True)
    for layer in model.model.layers[17:]:
        original_forward = layer.forward

        def forward(*args, _forward=original_forward, **kwargs):
            return torch.utils.checkpoint.checkpoint(_forward, *args, use_reentrant=False, **kwargs) \
                if torch.is_grad_enabled() else _forward(*args, **kwargs)

        layer.forward = forward

    def set_state(state):
        with torch.no_grad():
            for k in MAT_KEYS:
                named[k].copy_(torch.from_numpy(np.ascontiguousarray(state[k])).to(named[k].device))

    def eval_panel(cond):
        acc = defaultdict(list)
        with torch.no_grad():
            for row in panel:
                post, z = packet(model, row)
                lp = z.double().log_softmax(-1)
                if cond == 'surface_class_mass':
                    own = float(-torch.logsumexp(lp[classes == int(row['surface_class'])], dim=0))
                elif cond == 'true_token':
                    own = float(-lp[int(row['target'])])
                else:
                    own = float(-lp[int(row['permuted_target'])])
                acc['own'].append(own)
                acc['nll_target'].append(float(-lp[int(row['target'])]))
                acc['nll_permuted'].append(float(-lp[int(row['permuted_target'])]))
        return {k: np.asarray(v) for k, v in acc.items()}

    set_state({k: original[k].numpy() for k in MAT_KEYS})
    base_evs = {c: eval_panel(c) for c in CONDITIONS}
    arrays = {'base__' + c + '__own': base_evs[c]['own'].astype(np.float64) for c in CONDITIONS}
    base_own = {c: float(base_evs[c]['own'].mean()) for c in CONDITIONS}
    dnll = float(np.abs(base_evs['true_token']['nll_target'] -
                        native_eval['NLL'].astype(np.float64)).mean())
    integrity = {'mean_abs_bridge_minus_native_dNLL': dnll,
                 'gpu': torch.cuda.get_device_name(0)}
    assert dnll < 0.05, ('bridge baseline drift', integrity)
    print('PHASE2755_PROGRESS baselines %s' % json.dumps(
        {'base_own': base_own, 'bridge_minus_native_dNLL': dnll}), elapsed(), flush=True)

    # ---- training: 6 runs (resumable per run) ----
    runs_meta = {}
    for seed in SEEDS:
        draws = np.random.default_rng(seed).permutation(len(data['train'])).reshape(STEPS, BATCH)
        for condition in CONDITIONS:
            name = condition + '_' + str(seed)
            runfolder = folder / name
            done_128 = (runfolder / ('delta_%03d.npz' % STEPS)).exists()
            if not done_128:
                set_state({k: original[k].numpy() for k in MAT_KEYS})
                trace = []
                runstart = time.monotonic()
                for step, ix in enumerate(draws, 1):
                    gradients, loss = mean_gradient(model, [data['train'][i] for i in ix],
                                                    condition, classes, list(named.values()))
                    norm = float(torch.stack([g.double().square().sum() for g in gradients]).sum().sqrt())
                    assert norm > 0 and np.isfinite(norm)
                    with torch.no_grad():
                        for p, g in zip(named.values(), gradients):
                            p.add_(g, alpha=-STEP_NORM / norm)
                    trace.append({'step': step, 'objective': loss, 'full_gradient_norm': norm,
                                  'examples': ix.tolist()})
                    del gradients
                    if step % 16 == 0:
                        print('PHASE2755_PROGRESS', name, step,
                              round(time.monotonic() - runstart, 1), flush=True)
                    if step in CHECKPOINTS:
                        delta = {}
                        norm_square = 0.0
                        for n, p in named.items():
                            actual = p.detach().cpu().numpy()
                            value = actual - original[n].numpy()
                            approximate = original[n].numpy() + value
                            residual = actual - approximate
                            rebuilt = approximate + residual
                            assert np.array_equal(rebuilt.view(np.uint32), actual.view(np.uint32))
                            delta[n] = value
                            delta['reconstruction_residual__' + n] = residual
                            exact = actual.astype(np.float64) - original[n].numpy().astype(np.float64)
                            norm_square += float(np.sum(exact ** 2))
                        npz(runfolder / ('delta_%03d.npz' % step), **delta)
                        trace[-1]['checkpoint_delta_L2'] = float(np.sqrt(norm_square))
                del trace
            # deploy this run's final state bit-exactly and record behavioral meta
            z = np.load(runfolder / ('delta_%03d.npz' % STEPS), allow_pickle=False)
            state = {}
            for k in MAT_KEYS:
                v = original[k].numpy() + z[k]
                state[k] = v + z['reconstruction_residual__' + k]
            set_state(state)
            del z, state
            final_eval = eval_panel(condition)
            arrays[name + '__own'] = final_eval['own'].astype(np.float64)
            arrays[name + '__nll_target'] = final_eval['nll_target'].astype(np.float64)
            arrays[name + '__nll_permuted'] = final_eval['nll_permuted'].astype(np.float64)
            improvement = base_own[condition] - float(final_eval['own'].mean())
            runs_meta[name] = {'condition': condition, 'seed': seed,
                               'resumed': done_128,
                               'final_own_mean': float(final_eval['own'].mean()),
                               'base_own_mean': base_own[condition],
                               'improvement': improvement}
            print('PHASE2755_RUN_DONE', name, 'improvement', round(improvement, 4),
                  elapsed(), flush=True)
            gc.collect()

    # ---- restore original exactly ----
    set_state({k: original[k].numpy() for k in MAT_KEYS})
    recheck = eval_panel('true_token')
    restore_exact = bool(np.array_equal(recheck['own'], base_evs['true_token']['own']))
    assert set(EXPECTED_RUNS) == set(runs_meta)

    # ---- geometry extraction (CPU; flat float32 per run = 6 x 258 MiB) ----
    flat = {}
    for run in EXPECTED_RUNS:
        z = np.load(folder / run / 'delta_128.npz', allow_pickle=False)
        flat[run] = np.concatenate([z[k].reshape(-1) for k in MAT_KEYS]).astype(np.float32)
        del z
    conds = CONDITIONS
    G = PER_MATRIX
    # per-matrix 6x6 cosine
    per_matrix = {}
    for mi, mk in enumerate(MAT_KEYS):
        sl = slice(mi * G, (mi + 1) * G)
        M = np.stack([flat[run][sl] for run in EXPECTED_RUNS]).astype(np.float64)
        nrm = np.linalg.norm(M, axis=1, keepdims=True)
        C = (M @ M.T) / np.maximum(nrm * nrm.T, 1e-30)
        per_matrix[mk] = {'keys': EXPECTED_RUNS,
                          'cosine': [[float(x) for x in row] for row in C]}
        del M, C
    pooled_pairs = {}
    pooled_top = {}
    for pair_name, ca, cb in PAIR_DEFS:
        d47 = np.concatenate([flat[ca + '_2747'][mi * G:(mi + 1) * G] -
                              flat[cb + '_2747'][mi * G:(mi + 1) * G] for mi in range(3)])
        d48 = np.concatenate([flat[ca + '_2748'][mi * G:(mi + 1) * G] -
                              flat[cb + '_2748'][mi * G:(mi + 1) * G] for mi in range(3)])
        noise = np.concatenate([flat[ca + '_2747'][mi * G:(mi + 1) * G] -
                                flat[ca + '_2748'][mi * G:(mi + 1) * G] for mi in range(3)])
        pear = float(np.corrcoef(d47.astype(np.float64), d48.astype(np.float64))[0, 1])
        cs = cos64(d47, d48)
        n = d47.shape[0]
        o47 = np.argsort(np.abs(d47.astype(np.float64)))
        o48 = np.argsort(np.abs(d48.astype(np.float64)))
        jac = {}
        for k in KS:
            kk = int(round(k * n))
            a = set(o47[-kk:].tolist())
            b = set(o48[-kk:].tolist())
            jac['%g' % k] = len(a & b) / len(a | b)
        stable = bool(pear >= PREREG['pearson_min'] and jac['0.01'] >= PREREG['jaccard_top1pct_min'])
        e_ratio = float(np.dot(d47.astype(np.float64), d47.astype(np.float64)) /
                        max(np.dot(noise.astype(np.float64), noise.astype(np.float64)), 1e-30))
        pooled_pairs[pair_name] = {'cross_generation_pearson': pear, 'cross_generation_cosine': cs,
                                   'topk_jaccard': jac, 'subset_stable': stable,
                                   'energy_ratio_D_to_noise': e_ratio,
                                   'norm_2747': float(np.linalg.norm(d47.astype(np.float64))),
                                   'norm_2748': float(np.linalg.norm(d48.astype(np.float64)))}
        pooled_top[pair_name] = o47[-int(round(0.01 * n)):].copy()
        del d47, d48, noise, o47, o48
    # pooled same-generation condition vectors and cross-condition cosines
    pooled_cond = {(c, s): np.concatenate([flat[c + '_' + str(s)][mi * G:(mi + 1) * G]
                                           for mi in range(3)])
                   for c in conds for s in SEEDS}
    cross_condition = {str(s): {'true|permuted': cos64(pooled_cond[('true_token', s)],
                                                      pooled_cond[('within_surface_class_permuted_token', s)]),
                                'true|mass': cos64(pooled_cond[('true_token', s)],
                                                   pooled_cond[('surface_class_mass', s)]),
                                'permuted|mass': cos64(pooled_cond[('within_surface_class_permuted_token', s)],
                                                       pooled_cond[('surface_class_mass', s)])}
                       for s in SEEDS}
    # neuron-level top-1% pair intersections (per generation)
    neuron = {}
    for s in SEEDS:
        idxs = {}
        for mk in MAT_KEYS:
            offset = mi_of(mk) * G
            for cond in conds:
                seg = flat[cond + '_' + str(s)][offset:offset + G]
                if mk == 'down_proj.weight':
                    score = np.linalg.norm(seg.reshape(2_048, 6_144).astype(np.float64), axis=0)
                else:
                    score = np.linalg.norm(seg.reshape(6_144, 2_048).astype(np.float64), axis=1)
                k1 = max(1, int(round(0.01 * len(score))))
                idxs[(mk, cond)] = set(np.argsort(score)[-k1:].tolist())
        counts = {}
        for mk in MAT_KEYS:
            short = {'gate_proj.weight': 'gate', 'up_proj.weight': 'up',
                     'down_proj.weight': 'down'}[mk]
            counts[short + '_tp'] = len(idxs[(mk, 'true_token')] &
                                        idxs[(mk, 'within_surface_class_permuted_token')])
            counts[short + '_tm'] = len(idxs[(mk, 'true_token')] &
                                        idxs[(mk, 'surface_class_mass')])
            counts[short + '_pm'] = len(idxs[(mk, 'within_surface_class_permuted_token')] &
                                        idxs[(mk, 'surface_class_mass')])
        neuron[str(s)] = counts

    # ---- criteria verdicts ----
    g1 = {p: bool(d['subset_stable']) for p, d in pooled_pairs.items()}
    g2 = {str(s): bool(all(v < PREREG['cross_condition_cosine_max']
                           for v in cross_condition[str(s)].values())) for s in SEEDS}
    g3 = {p: bool(d['energy_ratio_D_to_noise'] >= PREREG['energy_ratio_min'])
          for p, d in pooled_pairs.items()}
    b1_runs = {n: bool(m['improvement'] > 0) for n, m in runs_meta.items()}
    verdict = {'G1_replication_pairs': int(sum(g1.values())), 'G1_details': g1,
               'G2_tripartite_generations': int(sum(g2.values())), 'G2_details': cross_condition,
               'G3_concentration_pairs': int(sum(g3.values())), 'G3_details': g3,
               'B1_behavior_runs': int(sum(b1_runs.values())), 'B1_details': b1_runs,
               'preregistered': PREREG}
    try:
        r51 = read(RESULT51)
        qwen4_ref = {p: {'pearson': r51['pooled_pairs'][p]['cross_generation_pearson'],
                         'jaccard_top1': r51['pooled_pairs'][p]['topk_jaccard']['0.01'],
                         'energy_ratio': r51['pooled_pairs'][p]['energy_ratio_D_to_noise']}
                     for p in pooled_pairs}
    except Exception as ex:  # noqa
        qwen4_ref = {'error': str(ex)}

    value = {'timestamp': stamp(), 'source': snapshot(__file__),
             'status': 'P5 step 3a complete',
             'model': 'qwen3-1.7b (Qwen3ForCausalLM)', 'block': BLOCK,
             'block_params': BLOCK_PARAMS, 'runs': runs_meta,
             'tokenizer_gate': gate, 'integrity': integrity,
             'restore_exact': restore_exact,
             'pooled_pairs': pooled_pairs, 'per_matrix_cosine': per_matrix,
             'cross_condition_pooled_cosine': cross_condition,
             'neuron_pair_intersections_top1pct': neuron,
             'verdict': verdict, 'qwen4_phase2751_reference': qwen4_ref,
             'seconds': elapsed(),
             'limits': ['Single new model, one block (layers[16].mlp); geometry claims are '
                        'two-model comparisons, not a universal law.',
                        'Training data, draws, seeds, steps and step rule are identical to '
                        'Phase2747; the optimizer (normalized gradient) has no adaptive state.',
                        'Behavioral evaluation is own-objective only.']}
    npz(folder / 'behavior_scores.npz', **arrays)
    npz(folder / 'pooled_top1pct_indices.npz',
        **{'pooled__' + p + '__top1pct': pooled_top[p].astype(np.int64) for p in pooled_top})
    save(folder / 'behavior_scores_meta.json',
         {'timestamp': stamp(), 'panel_sample_ids': [r['sample_id'] for r in panel],
          'families': [r['family'] for r in panel], 'conditions': CONDITIONS, 'seeds': SEEDS})
    save(finish, value)
    ledger('phase2755_cross_model_training', value['seconds'])
    print('PHASE2755_VERDICT', json.dumps(verdict), elapsed(), flush=True)


def mi_of(mk):
    return {'gate_proj.weight': 0, 'up_proj.weight': 1, 'down_proj.weight': 2}[mk]


if __name__ == '__main__':
    main()
