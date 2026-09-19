"""Phase 2782 C-R1 (POST-HOC exploratory control, origin acknowledged:
motivated by the 2782 finding that pure projection (lam=0) collapses
repair to 0/35 at every k while collateral stays 7/13 -- damage and
repair appear DECOUPLED).  Question: is natural-row collateral a GENERIC
perturbation effect (any unit-norm bsub at alpha breaks ~half of the 13
native-correct rows) or direction-specific?

Design (descriptive, no preregistered criterion):
  R1: random unit directions (10 draws/row, seed 27820) bsub at alpha in
      {0.1, 0.2, 0.3} at L35 on the 13 native-correct natural rows ->
      break rate per alpha (mean over draws).
  R2: the 65 controlled-subspace projected direction (k=65, lam=0) on
      the same 13 rows at alpha=0.3 -> already measured (7/13) inside
      2782 main; re-measured here for one-to-one comparability.
Interpretation: if random break rate ~= 7/13, collateral is generic
(threshold/decision-boundary effect); if ~0, the controlled-subspace
component itself carries the damage.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc
import phase2780_rdc_natural_pilot as p2780

BASE = cc.BASE
OUT = BASE / 'phase2782' / 'qwen4_direction_regularization'
ITEMS = p2780.ITEMS
ALPHAS = [0.1, 0.2, 0.3]
N_DRAWS = 10
SEED = 27820


def main():
    import torch
    from transformers import AutoTokenizer
    OUT.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(
        str(cc.ROOT / 'models' / 'hf' / 'qwen3-4b'), local_files_only=True,
        trust_remote_code=True, use_fast=True)
    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'status': 'POST-HOC exploratory control C-R1'}
    fc.save(OUT / 'execution_ctrl.json', execution)

    from phase2662_symmetric_mapping_contract import load_native
    model, tok2 = load_native('qwen4')
    model.eval()
    device = next(model.parameters()).device

    state = {'bsub': None}
    cap_hfin = {'h': None}
    handles = []

    def make_bsub_hook():
        def hook(module, args, output):
            if state['bsub'] is None:
                return None
            out = output[0] if isinstance(output, tuple) else output
            h = out[0, -1]
            out[0, -1] = h - state['bsub'][0] * h.norm() * state['bsub'][1]
            return None
        return hook

    def make_fin_hook():
        def hook(module, args, output):
            cap_hfin['h'] = output.detach()
            return None
        return hook

    handles.append(model.model.layers[35].register_forward_hook(
        make_bsub_hook()))
    handles.append(model.model.layers[35].register_forward_hook(
        make_fin_hook()))

    def fwd(ids):
        with torch.inference_mode():
            return model(torch.tensor([ids], device=device))

    def arg_of(o):
        return int(o.logits[0, -1].float().argmax())

    def h_final(ids):
        cap_hfin['h'] = None
        with torch.inference_mode():
            model(torch.tensor([ids], device=device))
        return cap_hfin['h'][0, -1].float().cpu().numpy().copy()

    ids_full, ids_wo, tgt_n, multi = [], [], [], []
    for (pre, span, suf, ans) in ITEMS:
        ids_f = tok(pre, add_special_tokens=False)['input_ids'] + \
            tok(span, add_special_tokens=False)['input_ids'] + \
            tok(suf, add_special_tokens=False)['input_ids']
        ids_w = tok(pre, add_special_tokens=False)['input_ids'] + \
            tok(suf, add_special_tokens=False)['input_ids']
        ids_full.append(ids_f)
        ids_wo.append(ids_w)
        tt = tok(ans, add_special_tokens=False)['input_ids']
        multi.append(len(tt) > 1)
        tgt_n.append(tt[0])
    tgt_arr = np.array(tgt_n, dtype=np.int64)

    arg_nat = np.array([arg_of(fwd(ids_full[i]))
                        for i in range(len(ITEMS))])
    wrong_mask = (arg_nat != tgt_arr) & ~np.array(multi)
    correct_idx = [int(i) for i in np.where(~wrong_mask)[0]]
    assert len(correct_idx) == 13, len(correct_idx)

    v_rows = {}
    for i in correct_idx:
        v = h_final(ids_full[i]) - h_final(ids_wo[i])
        v_rows[i] = v / np.linalg.norm(v)

    # k=65 lam=0 projection basis from controlled 2763 wrong rows (same as main)
    z61 = np.load(BASE / 'phase2761' / 'qwen4_kc_fault' / 'fault_scores.npz',
                  allow_pickle=False)
    cw = np.array(sorted(int(i) for i in z61['wrong_idx']))
    zbd = np.load(BASE / 'phase2763' / 'qwen4_debias_repair' / 'bias_dirs.npz',
                  allow_pickle=False)
    v_c = zbd['v_rows'][cw]
    v_c = v_c / np.linalg.norm(v_c, axis=1, keepdims=True)
    _, _, Vh = np.linalg.svd(v_c.astype(np.float64), full_matrices=False)
    B65 = Vh.T

    rng = np.random.default_rng(SEED)
    res = {'alphas': {}}
    for alpha in ALPHAS:
        rand_breaks, proj_breaks = [], []
        for i in correct_idx:
            b = 0
            for _ in range(N_DRAWS):
                r = rng.normal(size=v_rows[i].shape[0])
                r = r / np.linalg.norm(r)
                state['bsub'] = (alpha, torch.tensor(
                    r.astype(np.float32), device=device))
                try:
                    m = arg_of(fwd(ids_full[i]))
                finally:
                    state['bsub'] = None
                b += int(m != arg_nat[i])
            rand_breaks.append(b / N_DRAWS)
            pv = B65 @ (B65.T @ v_rows[i])
            pv = pv / np.linalg.norm(pv)
            state['bsub'] = (alpha, torch.tensor(
                pv.astype(np.float32), device=device))
            try:
                m = arg_of(fwd(ids_full[i]))
            finally:
                state['bsub'] = None
            proj_breaks.append(int(m != arg_nat[i]))
        res['alphas'][str(alpha)] = {
            'random_break_mean': round(float(np.mean(rand_breaks)), 4),
            'random_break_per_row': [round(float(x), 2)
                                     for x in rand_breaks],
            'proj65_break': int(sum(proj_breaks)),
            'n': len(correct_idx)}
        print('P2782C alpha=%.1f random_break=%.2f/13 proj65_break=%d/13'
              % (alpha, float(np.mean(rand_breaks)) * 13,
                 int(sum(proj_breaks))), flush=True)

    fc.save(OUT / 'result_ctrl.json', res)
    for h in handles:
        h.remove()
    print('P2782C DONE', flush=True)


if __name__ == '__main__':
    main()
