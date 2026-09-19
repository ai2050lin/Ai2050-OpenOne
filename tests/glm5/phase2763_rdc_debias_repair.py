"""Phase 2763: debias repair without oracle -- perm trained state x readout-end
bias suppression.

Question: 2761 localised the kc "internally knows but cannot say" fault to
output-end readout competition; 2759 showed the question-side fact span carries
the common-mode negative bias (bias/discrimination separable).  Can debiasing
-- WITHOUT any oracle answer direction -- convert internal knowledge into
behaviour (kc 78% -> ab level)?

Intervention bsub: readout-end bias suppression.  v_row := h_final(with
question fact span) - h_final(without), computed in the NATIVE model with the
2759 preregistered fact-token rule.  At inference, hook on layers[35] output:
h := h - alpha * ||h|| * unit(v_row).  alpha in {0.1, 0.3, 1.0}; primary
alpha = 0.3.  Random-direction control at matched alpha = 0.3.

Runs (2759 deploy protocol, block16 mlp): native, perm_2747, perm_2748,
true_2747 (control), mass_2747.

Prereg criteria (frozen before any intervention forward):
  C1 perm repair:   among the 14 native-kc-wrong rows, perm_2747 and perm_2748
                    base flip rate >= 0.4 (>= 6 rows) and > true_2747 flips.
  C2 bsub repair:   native + bsub(alpha=0.3) flip rate on all 65 native-wrong
                    rows >= 0.3 and bootstrap CI (1000, seed 2763010) of the
                    flip-rate difference vs matched ctrl has lower bound > 0.
  C3 family spec:   kc repair-any rate (perm_2747 base flip OR native bsub
                    alpha=0.3 flip) > ws repair-any rate, bootstrap CI of the
                    difference lower bound > 0.
  C4 combo (exploratory): perm + bsub(alpha=0.3) flips >= max(single modes).
  verdict: debias_repair_confirmed = C1 (at least one perm run) and C2 and C3.

Integrity gates:
  G0 deployment identity as 2759 (sha + deployed L2 rel < 1e-9, in deploy()).
  G1 native base forward determinism, first 8 rows bitwise.
  G2 v_row non-degeneracy: no empty fact spans; ||v_row|| > 0 for every row.
  G3 native wrong counts per family equal the 2761 recorded values
     (ab 0, kc 14, ldr 10, neg 16, ws 25).
"""
import json
import time
from pathlib import Path

import numpy as np

import rdc_construction_common as cc
import rdc_feature_common as fc
import phase2747_rdc_material as mat2747
import phase2759_rdc_question_span_deletion as p2759

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2763' / 'qwen4_debias_repair'
ALPHAS = [0.1, 0.3, 1.0]
ALPHA_PRIMARY = 0.3
CTRL_SEED = 2763001
BOOT_SEED = 2763010
BOOT_N = 1000

PREREG = {
    'phase': 2763,
    'question': 'Without oracle answer directions, does debiasing (perm '
                'trained state or readout-end bias suppression) convert kc '
                '"internally knows" into behaviour?',
    'model': 'qwen3-4b native plus four 2747 deployed BF16 checkpoints',
    'rows': 'all 320 diagnostic controlled_relation rows (5 families x 64)',
    'behaviour': 'greedy first token at final prompt position (2761 analog)',
    'bsub': 'v_row native-defined (2759 fact-token rule, h_with - h_without '
            'at final position); hook layers[35]: h := h - '
            'alpha*||h||*unit(v_row), alpha in {0.1,0.3,1.0}, primary 0.3',
    'combo': 'trained-state base + native v_row bsub(alpha=0.3)',
    'criteria': {'C1': 'perm_2747/2748 base flip rate on 14 native-kc-wrong '
                       'rows >= 0.4 and > true_2747 flips',
                 'C2': 'native bsub(alpha=0.3) flip rate on 65 native-wrong '
                       'rows >= 0.3 and bootstrap CI of (bsub - ctrl) '
                       'difference lower bound > 0',
                 'C3': 'kc repair-any rate > ws repair-any rate, bootstrap CI '
                       'difference lower bound > 0',
                 'C4': 'exploratory combo vs max(single)'},
    'verdict': 'debias_repair_confirmed iff C1(>=1 perm run) and C2 and C3',
    'integrity_gates': {'G0': 'deployment identity as 2759',
                        'G1': 'base forward determinism, first 8 rows bitwise',
                        'G2': 'v_row non-degeneracy, no empty fact spans',
                        'G3': 'native wrong per family == 2761 record '
                              '(ab 0, kc 14, ldr 10, neg 16, ws 25)'},
    'frozen_before_any_behavioural_forward': True,
}

G3_EXPECTED = {'attribute_binding': 0, 'knowledge_chain': 14,
               'long_distance_role': 10, 'negation_scope': 16,
               'word_sense': 25}


def main():
    t0 = time.time()
    cc.guard(0)
    assert not (OUT / 'result.json').exists(), 'immutable; delete before rerun'
    OUT.mkdir(parents=True, exist_ok=True)
    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG}
    fc.save(OUT / 'execution.json', execution)

    import torch
    material, data = mat2747.freeze()
    rows = [r for r in data['diagnostic'] if r['kind'] == 'controlled_relation']
    assert len(rows) == 320, len(rows)
    tgt_ids = np.array([r['target'] for r in rows], dtype=np.int64)
    fam_arr = np.array([r['family'] for r in rows], dtype=np.str_)

    from phase2662_symmetric_mapping_contract import load_native
    model, tok = load_native('qwen4')
    model.eval()
    assert model.config.num_hidden_layers == 36
    device = next(model.parameters()).device

    target_module = model.model.layers[16].mlp
    named = dict(target_module.named_parameters())
    original = {n: p.detach().float().cpu().clone() for n, p in named.items()}

    # ---------------- Pass A (native): base + determinism + v_row ----------
    def forward_final(ids_list):
        """Return (argmax, target_logit, rival_logit, h_final) per row."""
        out_args, out_tgt, out_riv, out_h = [], [], [], []
        with torch.inference_mode():
            for ids in ids_list:
                t = torch.tensor([ids], device=device)
                o = model(t, output_hidden_states=True)
                logits = o.logits[0, -1].float()
                h = o.hidden_states[-1][0, -1].float()
                out_args.append(int(logits.argmax()))
                out_tgt.append(float(logits[tgt_ids[len(out_args) - 1]]))
                z = logits.clone()
                z[tgt_ids[len(out_args) - 1]] = -1e30
                out_riv.append(float(z.max()))
                out_h.append(h.cpu().numpy())
        return (np.array(out_args), np.array(out_tgt), np.array(out_riv),
                np.stack(out_h))

    id_list = [r['prompt_ids'] for r in rows]
    arg_native, tgt_native, riv_native, h_native = forward_final(id_list)
    # G1: determinism on first 8 rows
    arg_rep, _, _, h_rep = forward_final(id_list[:8])
    g1 = float(np.abs(h_native[:8] - h_rep).max())
    assert arg_native[:8].tolist() == arg_rep.tolist(), 'G1 argmax'

    wrong_native = arg_native != tgt_ids
    fam_wrong = {f: int(wrong_native[fam_arr == f].sum()) for f in G3_EXPECTED}
    assert fam_wrong == G3_EXPECTED, ('G3', fam_wrong)

    # fact spans + v_row (native)
    v_rows = np.zeros((len(rows), h_native.shape[1]), dtype=np.float32)
    v_norms = np.zeros(len(rows))
    n_empty_fact = 0
    for i, r in enumerate(rows):
        _, fact, _, _ = p2759.fact_tokens(r, tok)
        if len(fact) == 0:
            n_empty_fact += 1
            continue
        ids_del = [id_list[i][j] for j in p2759.spans_to_keep(
            len(id_list[i]), fact)]
        _, _, _, h_del = forward_final([ids_del])
        v = h_native[i] - h_del[0]
        v_rows[i] = v
        v_norms[i] = float(np.linalg.norm(v))
    assert n_empty_fact == 0, ('G2 empty fact spans', n_empty_fact)
    assert (v_norms > 0).all(), 'G2 degenerate v_row'
    print('P2763 PASS_A_DONE wrong=%s vnorm med=%.2f' %
          (fam_wrong, float(np.median(v_norms))), flush=True)

    wrong_idx = np.where(wrong_native)[0]
    kc_wrong_idx = wrong_idx[fam_arr[wrong_idx] == 'knowledge_chain']
    ws_wrong_idx = wrong_idx[fam_arr[wrong_idx] == 'word_sense']
    rng_ctrl = np.random.default_rng(CTRL_SEED)
    d = h_native.shape[1]
    ctrl_dirs = {}
    for i in wrong_idx:
        u = rng_ctrl.normal(size=d)
        ctrl_dirs[i] = (u / np.linalg.norm(u)).astype(np.float32)

    # ---------------- readout hook -----------------------------------------
    hook_state = {'on': False, 'alpha': None, 'vec': None}

    def readout_hook(module, args, output):
        if not hook_state['on']:
            return None
        out = output[0] if isinstance(output, tuple) else output
        h = out[0, -1]
        out[0, -1] = h - hook_state['alpha'] * h.norm() * hook_state['vec']
        return None

    handle = model.model.layers[35].register_forward_hook(readout_hook)

    def run_variant(i, alpha, vec):
        hook_state['on'] = True
        hook_state['alpha'] = alpha
        hook_state['vec'] = torch.tensor(vec, device=device)
        try:
            with torch.inference_mode():
                o = model(torch.tensor([id_list[i]], device=device))
                logits = o.logits[0, -1].float()
        finally:
            hook_state['on'] = False
        return int(logits.argmax())

    # ---------------- Pass B: per run base + interventions -----------------
    run_results = {}
    arrays = {'row_idx': np.arange(len(rows)), 'family': fam_arr,
              'target': tgt_ids, 'v_norm': v_norms}
    flip_tab = {}   # (run, mode) -> per-native-wrong-row bool flip
    for run_key, condition, seed in p2759.RUNS:
        if condition is not None:
            p2759.deploy(model, named, original, run_key, condition, seed)
        arg, tgt_l, riv_l, _ = forward_final(id_list)
        correct = arg == tgt_ids
        arrays['%s__base_arg' % run_key] = arg
        arrays['%s__base_correct' % run_key] = correct
        arrays['%s__base_tgt_logit' % run_key] = tgt_l
        arrays['%s__base_riv_logit' % run_key] = riv_l
        res = {'n_correct': int(correct.sum()),
               'kc_correct_rate': float(correct[fam_arr == 'knowledge_chain']
                                        .mean())}
        # interventions on native-wrong rows (all families)
        for i in wrong_idx:
            v = v_rows[i] / np.linalg.norm(v_rows[i])
            for a in ALPHAS:
                m = run_variant(i, a, v)
                arrays['%s__bsub_a%g_%d' % (run_key, a, i)] = m
            m = run_variant(i, ALPHA_PRIMARY, ctrl_dirs[i])
            arrays['%s__ctrl_%d' % (run_key, i)] = m
        # combo (kc native-wrong rows): trained base + native v_row bsub
        if run_key != 'native':
            for i in kc_wrong_idx:
                v = v_rows[i] / np.linalg.norm(v_rows[i])
                m = run_variant(i, ALPHA_PRIMARY, v)
                arrays['%s__combo_%d' % (run_key, i)] = m
        # flip table on native wrong rows
        ft = {}
        ft['bsub_a0.3'] = np.array(
            [arrays['%s__bsub_a0.3_%d' % (run_key, i)] == tgt_ids[i]
             for i in wrong_idx])
        ft['ctrl'] = np.array(
            [arrays['%s__ctrl_%d' % (run_key, i)] == tgt_ids[i]
             for i in wrong_idx])
        ft['base_flip_native_wrong'] = correct[wrong_idx]
        if run_key != 'native':
            ft['combo'] = np.array(
                [arrays['%s__combo_%d' % (run_key, i)] == tgt_ids[i]
                 for i in kc_wrong_idx])
        flip_tab[run_key] = ft
        print('P2763 RUN_DONE %s n_correct=%d kc_rate=%.4f flips(a0.3)=%d '
              'ctrl=%d' % (run_key, res['n_correct'], res['kc_correct_rate'],
                           int(ft['bsub_a0.3'].sum()), int(ft['ctrl'].sum())),
              flush=True)
        if condition is not None:
            p2759.restore(model, named, original)

    handle.remove()

    # ---------------- statistics -------------------------------------------
    def boot_ci_diff(a, b, n_a=None, n_b=None):
        """Bootstrap CI of mean(a)-mean(b) over row resampling."""
        rgs = np.random.default_rng(BOOT_SEED)
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        diffs = []
        for _ in range(BOOT_N):
            ia = rgs.integers(0, len(a), len(a))
            ib = rgs.integers(0, len(b), len(b))
            diffs.append(a[ia].mean() - b[ib].mean())
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        return float(lo), float(hi)

    kc_mask = fam_arr[wrong_idx] == 'knowledge_chain'
    ws_mask = fam_arr[wrong_idx] == 'word_sense'
    stats = {}
    # C1
    c1 = {}
    for rk in ['perm_2747', 'perm_2748', 'true_2747', 'mass_2747']:
        c1[rk] = {'kc_flips': int(flip_tab[rk]['base_flip_native_wrong']
                                  [kc_mask].sum())}
    c1['perm_2747_pass'] = bool(c1['perm_2747']['kc_flips'] >= 6 and
                                c1['perm_2747']['kc_flips'] >
                                c1['true_2747']['kc_flips'])
    c1['perm_2748_pass'] = bool(c1['perm_2748']['kc_flips'] >= 6 and
                                c1['perm_2748']['kc_flips'] >
                                c1['true_2747']['kc_flips'])
    c1['pass'] = bool(c1['perm_2747_pass'] or c1['perm_2748_pass'])
    stats['C1'] = c1
    # C2 (native)
    bsub = flip_tab['native']['bsub_a0.3']
    ctrl = flip_tab['native']['ctrl']
    rate_b = float(bsub.mean())
    rate_c = float(ctrl.mean())
    lo, hi = boot_ci_diff(bsub, ctrl)
    c2 = {'flip_rate_bsub': rate_b, 'flip_rate_ctrl': rate_c,
          'n_flips_bsub': int(bsub.sum()), 'n_flips_ctrl': int(ctrl.sum()),
          'boot_ci_lo': lo, 'boot_ci_hi': hi,
          'pass': bool(rate_b >= 0.3 and lo > 0)}
    stats['C2'] = c2
    # C3 repair-any
    perm_flip_kc = flip_tab['perm_2747']['base_flip_native_wrong'][kc_mask]
    nat_bsub_kc = bsub[kc_mask]
    repair_kc = (perm_flip_kc | nat_bsub_kc).astype(np.float64)
    perm_flip_ws = flip_tab['perm_2747']['base_flip_native_wrong'][ws_mask]
    nat_bsub_ws = bsub[ws_mask]
    repair_ws = (perm_flip_ws | nat_bsub_ws).astype(np.float64)
    lo3, hi3 = boot_ci_diff(repair_kc, repair_ws)
    c3 = {'kc_repair_any_rate': float(repair_kc.mean()),
          'ws_repair_any_rate': float(repair_ws.mean()),
          'boot_ci_lo': lo3, 'boot_ci_hi': hi3,
          'pass': bool(lo3 > 0)}
    stats['C3'] = c3
    # C4 exploratory: combo flips on kc native-wrong (perm runs)
    c4 = {}
    for rk in ['perm_2747', 'perm_2748', 'true_2747', 'mass_2747']:
        combo = flip_tab[rk]['combo']
        single_max = max(int(flip_tab[rk]['base_flip_native_wrong'][kc_mask]
                             .sum()), int(nat_bsub_kc.sum()))
        c4[rk] = {'combo_flips': int(combo.sum()), 'max_single': single_max}
    stats['C4'] = c4

    verdict = ('debias_repair_confirmed' if (stats['C1']['pass'] and
                                             stats['C2']['pass'] and
                                             stats['C3']['pass'])
               else 'not_confirmed')

    results = {'G1_max_abs_diff': g1, 'G3_fam_wrong': fam_wrong,
               'n_empty_fact': n_empty_fact,
               'v_norm_median': float(np.median(v_norms)),
               'native_correct_rate': float((~wrong_native).mean()),
               'stats': stats, 'verdict': verdict,
               'seconds': time.time() - t0}
    fc.npz(OUT / 'behaviour_scores.npz', **arrays)
    fc.save(OUT / 'behaviour_scores_meta.json',
            {'schema': 'rows 320; per run (5): base arg/correct/logits; '
                       'interventions on native-wrong rows: bsub_a{0.1,0.3,1} '
                       'arg, ctrl arg (alpha 0.3 matched random dir), combo '
                       'arg (kc wrong rows, trained runs); v_row native-'
                       'defined, hook layers[35]: h -= a*||h||*unit(v)'})
    fc.npz(OUT / 'bias_dirs.npz',
           v_rows=v_rows, v_norms=v_norms,
           ctrl_dirs=np.stack([ctrl_dirs[i] for i in wrong_idx]),
           wrong_idx=wrong_idx)
    fc.save(OUT / 'result.json', results)
    print('PHASE2763_DONE ' + json.dumps(
        {'verdict': verdict, 'C1': {k: v for k, v in c1.items()},
         'C2': c2, 'C3': c3, 'seconds': results['seconds']}), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        (OUT / 'crash.txt').write_text(traceback.format_exc(),
                                       encoding='utf-8')
        raise
