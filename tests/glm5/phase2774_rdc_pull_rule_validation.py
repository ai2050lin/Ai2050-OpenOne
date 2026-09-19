"""Phase 2774: pull sign rule preregistered validation + orthogonal bsub +
negation trained-state archival analysis.

Builds on: 2763 (v_rows, behaviour_scores), 2772 (abl/bsub/combo per row),
2773 (post-hoc pull rule on 7 kc rows).

D1: pull_i = v_row_i . (W_U[target] - W_U[rival]).  Preregistered criteria
    frozen AFTER pull computation but BEFORE any bsub_orth forward and BEFORE
    evaluating combo outcomes on the 58 non-derivation rows.
      P1a: among abl-fixed rows with pull<0 (excluding 7 derivation rows),
           combo keeps >= 80%.
      P1b: among abl-fixed rows with pull>0 (excluding 5 derivation rows),
           combo breaks >= 50%.
      P2:  combo flip rate (pull<0) - combo flip rate (pull>0) >= 0.15
           over all 65 rows.
D2: bsub_orth -- remove only the readout-parallel component of v_row:
      v_orth = v - (v.u_ro) u_ro,  u_ro = unit(W_U[target] - W_U[rival])
      P3: bsub_orth flips >= plain bsub flips on pull>0 rows.
      P4 (descriptive): total bsub_orth flips vs plain bsub 21.
D3 (archival, 2763 behaviour_scores): negation_scope base flips under each
    2747 trained run + collateral on native-correct rows.
      P5: partial trainability if some run flips >= 4/16 neg rows with
          total collateral on 320 native-correct rows <= 20%.
"""
import json
import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

BASE = Path(r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
            r'\rdc_query_construction_20260913')
P2761 = BASE / 'phase2761' / 'qwen4_kc_fault'
P2763 = BASE / 'phase2763' / 'qwen4_debias_repair'
P2772 = BASE / 'phase2772' / 'qwen4_combined_repair'
OUT = BASE / 'phase2774' / 'qwen4_pull_validation'

ALPHA = 0.3
BSUB_LAYER = 35
DERIV_ROWS = [273, 277, 281, 298, 307, 311, 314]  # 2773 post-hoc rows (5
# interfered: 273/277/281/298/314; 2 compatible: 307/311); validation
# excludes them

G3_EXPECTED = {'attribute_binding': 65, 'knowledge_chain': 64,
               'long_distance_role': 64, 'negation_scope': 64,
               'word_sense': 63}

PREREG = {
    'derivation': 'pull sign rule observed post-hoc on 7 kc rows in Phase '
                  '2773 (5 interfered pull>0, 2 compatible pull<0). Criteria '
                  'frozen after pull computation, before bsub_orth forwards '
                  'and before evaluating combo on non-derivation rows.',
    'P1a': 'among abl-fixed rows with pull<0, excluding DERIV_ROWS: combo '
           'flip (keeps abl repair) rate >= 0.8',
    'P1b': 'among abl-fixed rows with pull>0, excluding DERIV_ROWS: combo '
           'breaks (abl-fixed -> combo wrong) rate >= 0.5',
    'P2': 'combo flip rate (pull<0) - combo flip rate (pull>0) >= 0.15 '
          'over all 65 rows',
    'P3': 'bsub_orth flips >= plain bsub flips on pull>0 rows',
    'P4_descriptive': 'bsub_orth total flips vs plain bsub 21/65',
    'P5': 'neg partial trainability: some 2747 run base-flips >= 4/16 neg '
          'rows AND total collateral on 320 native-correct rows <= 20%',
    'verdict': 'pull_rule_confirmed iff P1a and P1b and P2; '
               'orth_bsub_confirmed iff P3; neg_trainable_partial iff P5',
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # ---------------- archival loads (no GPU) ----------------
    z61 = np.load(P2761 / 'fault_scores.npz', allow_pickle=False)
    wrong_idx = np.array(sorted(int(i) for i in z61['wrong_idx']))
    tgt61 = z61['target']
    zbd = np.load(P2763 / 'bias_dirs.npz', allow_pickle=False)
    v_rows = zbd['v_rows']                      # (320, 2560)
    v_norms = zbd['v_norms']
    zbeh = np.load(P2763 / 'behaviour_scores.npz', allow_pickle=False)
    arg_native = zbeh['native__base_arg']
    zrep = np.load(P2772 / 'repair_stats.npz', allow_pickle=False)
    assert (zrep['wrong_idx'] == wrong_idx).all()
    abl = zrep['abl'].astype(bool)
    bsub = zrep['bsub'].astype(bool)
    combo = zrep['combo'].astype(bool)

    import phase2747_rdc_material as mat2747
    material, data = mat2747.freeze()
    rows = [r for r in data['diagnostic'] if r['kind'] == 'controlled_relation']
    assert len(rows) == 320, len(rows)
    tgt_ids = np.array([r['target'] for r in rows], dtype=np.int64)
    fam_arr = np.array([r['family'] for r in rows], dtype=np.str_)
    assert (tgt_ids[wrong_idx] == tgt61[wrong_idx]).all()
    id_list = [r['prompt_ids'] for r in rows]

    # ---------------- load qwen4 (W_U + forwards) ----------------
    from phase2662_symmetric_mapping_contract import load_native
    model, tok = load_native('qwen4')
    model.eval()
    device = next(model.parameters()).device
    W_U = model.lm_head.weight.detach().float().cpu().numpy()  # (V, 2560)

    # ---------------- D1: pull computation (CPU) ----------------
    pull = np.zeros(len(wrong_idx))
    rival_ids = np.zeros(len(wrong_idx), dtype=np.int64)
    for k, i in enumerate(wrong_idx):
        i = int(i)
        riv = int(arg_native[i])
        rival_ids[k] = riv
        pull[k] = float(v_rows[i] @ (W_U[tgt_ids[i]] - W_U[riv]))
    pull_pos = pull > 0
    print('P2774 PULL pos=%d neg=%d min=%.3f max=%.3f'
          % (int(pull_pos.sum()), int((~pull_pos).sum()),
             pull.min(), pull.max()), flush=True)

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG,
                 'pull_stats': {'n_pos': int(pull_pos.sum()),
                                'n_neg': int((~pull_pos).sum()),
                                'min': float(pull.min()),
                                'max': float(pull.max()),
                                'deriv_rows': DERIV_ROWS}}
    fc.save(OUT / 'execution.json', execution)

    # ---------------- GPU pass: native verify + bsub + bsub_orth -------
    hook_state = {'on': False, 'alpha': None, 'vec': None}

    def readout_hook(module, args, output):
        if not hook_state['on']:
            return None
        out = output[0] if isinstance(output, tuple) else output
        h = out[0, -1]
        out[0, -1] = h - hook_state['alpha'] * h.norm() * hook_state['vec']
        return None

    handle = model.model.layers[BSUB_LAYER].register_forward_hook(readout_hook)

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

    import torch
    arg_check = np.empty(len(wrong_idx), dtype=np.int64)
    for k, i in enumerate(wrong_idx):
        with torch.inference_mode():
            o = model(torch.tensor([id_list[int(i)]], device=device))
            arg_check[k] = int(o.logits[0, -1].float().argmax())
    assert (arg_check == arg_native[wrong_idx]).all(), 'native verify'

    flips_bsub = np.zeros(len(wrong_idx), dtype=bool)
    flips_orth = np.zeros(len(wrong_idx), dtype=bool)
    n_degenerate = 0
    for k, i in enumerate(wrong_idx):
        i = int(i)
        v = v_rows[i] / v_norms[i]
        m = run_variant(i, ALPHA, v)
        flips_bsub[k] = (m == tgt_ids[i])
        u_ro = W_U[tgt_ids[i]] - W_U[int(rival_ids[k])]
        u_ro = u_ro / np.linalg.norm(u_ro)
        v_par = float(v @ u_ro)
        v_orth = v - v_par * u_ro
        nrm = np.linalg.norm(v_orth)
        if nrm < 1e-6:
            n_degenerate += 1
            flips_orth[k] = False
            continue
        m2 = run_variant(i, ALPHA, v_orth / nrm)
        flips_orth[k] = (m2 == tgt_ids[i])
    handle.remove()
    print('P2774 GPU_DONE bsub=%d orth=%d degen=%d'
          % (int(flips_bsub.sum()), int(flips_orth.sum()), n_degenerate),
          flush=True)

    # ---------------- P1/P2 evaluation ----------------
    in_deriv = np.array([int(i) in set(DERIV_ROWS) for i in wrong_idx])
    abl_fix = abl
    p1a_rows = abl_fix & (~pull_pos) & (~in_deriv)
    p1a_keep = float(combo[p1a_rows].mean()) if p1a_rows.any() else None
    p1a_pass = bool(p1a_rows.sum() > 0 and p1a_keep >= 0.8)
    p1b_rows = abl_fix & pull_pos & (~in_deriv)
    p1b_break = float((~combo[p1b_rows]).mean()) if p1b_rows.any() else None
    p1b_pass = bool(p1b_rows.sum() > 0 and p1b_break >= 0.5)
    rate_pos = float(combo[pull_pos].mean())
    rate_neg = float(combo[~pull_pos].mean())
    p2_pass = bool((rate_neg - rate_pos) >= 0.15)

    # ---------------- P3 evaluation ----------------
    p3_pass = bool(flips_orth[pull_pos].sum() >= flips_bsub[pull_pos].sum())
    p3_counts = {'orth_pos': int(flips_orth[pull_pos].sum()),
                 'bsub_pos': int(flips_bsub[pull_pos].sum()),
                 'orth_neg': int(flips_orth[~pull_pos].sum()),
                 'bsub_neg': int(flips_bsub[~pull_pos].sum())}

    # ---------------- D3: neg trained-state (archival) ----------------
    fams = ['attribute_binding', 'knowledge_chain', 'long_distance_role',
            'negation_scope', 'word_sense']
    neg_mask_all = fam_arr == 'negation_scope'
    wrong_all = arg_native != tgt_ids
    d3 = {}
    for run in ['perm_2747', 'perm_2748', 'true_2747', 'mass_2747']:
        base_correct = zbeh['%s__base_correct' % run].astype(bool)
        neg_wrong_idx = np.where(wrong_all & neg_mask_all)[0]
        neg_flips = int(base_correct[neg_wrong_idx].sum())
        collateral_total = int((zbeh['native__base_correct'].astype(bool)
                                & ~base_correct).sum())
        coll_by_fam = {f: int((zbeh['native__base_correct'].astype(bool)
                               & ~base_correct)[fam_arr == f].sum())
                       for f in fams}
        d3[run] = {'neg_flips': neg_flips, 'collateral_total':
                   collateral_total, 'collateral_by_fam': coll_by_fam}
    p5_pass = bool(any(v['neg_flips'] >= 4 and v['collateral_total'] <= 64
                       for v in d3.values()))

    # ---------------- verdicts ----------------
    pull_pass = bool(p1a_pass and p1b_pass and p2_pass)
    verdict = {'pull_rule_confirmed': pull_pass,
               'orth_bsub_confirmed': p3_pass,
               'neg_trainable_partial': p5_pass,
               'p1a': {'n': int(p1a_rows.sum()), 'keep_rate': p1a_keep,
                       'pass': p1a_pass},
               'p1b': {'n': int(p1b_rows.sum()), 'break_rate': p1b_break,
                       'pass': p1b_pass},
               'p2': {'rate_neg': rate_neg, 'rate_pos': rate_pos,
                      'pass': p2_pass},
               'p3': p3_counts, 'p5': p5_pass,
               'n_degenerate_orth': n_degenerate}

    fam_tab = {}
    for f in fams:
        m = np.array([fam_arr[int(i)] == f for i in wrong_idx])
        fam_tab[f] = {'n': int(m.sum()),
                      'pull_pos': int(pull_pos[m].sum()),
                      'abl': int(abl[m].sum()), 'bsub': int(bsub[m].sum()),
                      'combo': int(combo[m].sum()),
                      'bsub_orth': int(flips_orth[m].sum())}

    result = {'phase': 2774, 'prereg': PREREG, 'pull_stats':
              execution['pull_stats'], 'verdict': verdict,
              'family_table': fam_tab,
              'd3_trained_state': d3,
              'per_row': [{'row': int(i), 'fam': str(fam_arr[int(i)]),
                           'pull': float(pull[k]),
                           'rival': int(rival_ids[k]),
                           'abl': bool(abl[k]), 'bsub': bool(bsub[k]),
                           'combo': bool(combo[k]),
                           'bsub_orth': bool(flips_orth[k])}
                          for k, i in enumerate(wrong_idx)]}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'pull_stats.npz', wrong_idx=wrong_idx, pull=pull,
           rival_ids=rival_ids, flips_bsub=flips_bsub,
           flips_orth=flips_orth, combo=combo, abl=abl, bsub=bsub)

    print('P2774 VERDICT %s' % json.dumps(verdict), flush=True)
    print('P2774 FAM_TAB %s' % json.dumps(fam_tab), flush=True)


if __name__ == '__main__':
    main()
