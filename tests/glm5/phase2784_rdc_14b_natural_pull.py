"""Phase 2784: natural-language pull replication on Qwen3-14B +
capacity boundary check.

Motivation: 2776 restored the 14B scale point on the controlled panel
(carrier_replicated_14b, neg native 0 wrong = capacity boundary); 2780
established natural-language pull triage + repair on 4B (9/35 vs ctrl 0)
with a repair-damage Pareto frontier (2781-2783).  This phase tests both
threads at 14B on the SAME 48 frozen natural items (2780 ITEMS).

Preregistered (frozen before any 14B forward):
  P1 (capacity): native wrong count n_wrong at the first-token criterion
     (multi-token answers excluded).  If n_wrong == 0 ->
     'natural_errors_absent_14b' (capacity boundary; interventions
     skipped, verdict recorded, descriptive only).
  P2 (conditional on n_wrong >= 8): span-deletion v_row at L39 decoder
     output (pre-final-norm; 14B has 40 layers), pull = v . (W_U[tgt] -
     W_U[rival]); arms: pull<0 -> bsub alpha=0.3 at L39; pull>=0 ->
     sign-flip bsub (ORACLE-GUIDED, flagged).  Natural pull rule
     confirmed iff pipeline flips >= 3 AND random-direction ctrl flips
     = 0 (10 draws/row, seed 27840).
  P3 (descriptive, conditional): collateral on up to 20 sampled
     native-correct natural rows with the dictated arm at alpha=0.3.
Gates: G0 decode roundtrip on first 8 items; G1 determinism (first 4);
  G2 all v_norms > 0.
Placement/loader: identical to 2776 (CPU embed+L0-19 / GPU L20-39 +
lm_head, manual meta+shard assign).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc
import phase2780_rdc_natural_pilot as p2780

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2784' / 'qwen14b_natural_pull'
ITEMS = p2780.ITEMS
ALPHA = 0.3
L_LAST = 39          # last decoder layer (0-based) of 40
N_CTRL_DRAWS = 10
CTRL_SEED = 27840
N_COLL_ROWS = 20

PREREG = {
    'P1': 'native n_wrong at first-token criterion; if 0 -> '
          'natural_errors_absent_14b, interventions skipped',
    'P2': 'conditional n_wrong>=8: pull<0 bsub / pull>=0 flip bsub '
          'alpha=0.3 at L39; natural pull confirmed iff flips >= 3 AND '
          'ctrl = 0',
    'P3': 'descriptive collateral on <= 20 native-correct rows',
    'verdict': 'recorded per P1/P2',
}


def main():
    import torch
    from transformers import AutoTokenizer
    OUT.mkdir(parents=True, exist_ok=True)

    tok14 = AutoTokenizer.from_pretrained(
        str(ROOT / 'models' / 'hf' / 'Qwen3-14B'), local_files_only=True,
        trust_remote_code=True, use_fast=True)

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG}
    fc.save(OUT / 'execution.json', execution)

    from phase2776_rdc_14b_replication import build_model_manual
    model, cfg, plan, n_assigned, n_buf_fix, W_O_unused = \
        build_model_manual()
    assert cfg.num_hidden_layers == 40
    device = torch.device('cuda')

    # ---- tokenisation, identical recipe to 2780 (14B tokenizer)
    ids_full, ids_wo, tgt_n, multi = [], [], [], []
    for (pre, span, suf, ans) in ITEMS:
        ids_f = tok14(pre, add_special_tokens=False)['input_ids'] + \
            tok14(span, add_special_tokens=False)['input_ids'] + \
            tok14(suf, add_special_tokens=False)['input_ids']
        ids_w = tok14(pre, add_special_tokens=False)['input_ids'] + \
            tok14(suf, add_special_tokens=False)['input_ids']
        ids_full.append(ids_f)
        ids_wo.append(ids_w)
        tt = tok14(ans, add_special_tokens=False)['input_ids']
        multi.append(len(tt) > 1)
        tgt_n.append(tt[0])
    tgt_arr = np.array(tgt_n, dtype=np.int64)
    g0 = all(tok14.decode(ids_full[i]) ==
             ITEMS[i][0] + ITEMS[i][1] + ITEMS[i][2] for i in range(8))
    assert g0, 'G0 decode roundtrip'
    print('P2784 G0_OK', flush=True)

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

    handles.append(model.model.layers[L_LAST].register_forward_hook(
        make_bsub_hook()))
    handles.append(model.model.layers[L_LAST].register_forward_hook(
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

    a0 = [arg_of(fwd(ids_full[i])) for i in range(4)]
    a1 = [arg_of(fwd(ids_full[i])) for i in range(4)]
    assert a0 == a1, 'G1 determinism'
    print('P2784 G1_OK', flush=True)

    arg_nat = np.empty(len(ITEMS), dtype=np.int64)
    for i in range(len(ITEMS)):
        arg_nat[i] = arg_of(fwd(ids_full[i]))
    wrong_mask = (arg_nat != tgt_arr) & ~np.array(multi)
    wrong_idx = [int(i) for i in np.where(wrong_mask)[0]]
    correct_idx = [int(i) for i in np.where(~wrong_mask)[0]]
    n_wrong = len(wrong_idx)
    print('P2784 P1 n_wrong=%d n_correct=%d n_multi=%d'
          % (n_wrong, len(correct_idx), int(sum(multi))), flush=True)

    result = {'phase': 2784, 'prereg': PREREG,
              'p1': {'n_wrong': n_wrong, 'n_correct': len(correct_idx),
                     'n_multi_answer': int(sum(multi))}}
    if n_wrong == 0:
        result['verdict'] = 'natural_errors_absent_14b'
        fc.save(OUT / 'result.json', result)
        for h in handles:
            h.remove()
        print('P2784 VERDICT natural_errors_absent_14b', flush=True)
        return

    # ---- conditional branch: v_rows, pull, arms
    v_rows, v_norms = {}, {}
    for i in wrong_idx + correct_idx:
        v = h_final(ids_full[i]) - h_final(ids_wo[i])
        nrm = float(np.linalg.norm(v))
        assert nrm > 0, ('G2 degenerate', i)
        v_rows[i] = v / nrm
        v_norms[i] = nrm
    print('P2784 G2_OK v_norms>0 on %d rows' % len(v_rows), flush=True)

    W_U = model.get_submodule('lm_head').weight

    prep = {}
    for i in wrong_idx + correct_idx:
        zi = fwd(ids_full[i]).logits[0, -1].float()
        tmp = zi.clone()
        tmp[tgt_arr[i]] = -1e30
        riv = int(tmp.argmax())
        d_uv = (W_U[tgt_arr[i]] - W_U[riv]).float()
        pl = float(v_rows[i] @ d_uv.cpu().numpy())
        if pl < 0:
            prep[i] = {'rival': riv, 'pull': pl, 'arm': 'combo'}
        else:
            u_ro = d_uv / d_uv.norm()
            vf = v_rows[i] - 2.0 * pl * u_ro.cpu().numpy()
            vf = vf / np.linalg.norm(vf)
            prep[i] = {'rival': riv, 'pull': pl, 'arm': 'flip',
                       'v_flip': vf}
    n_flip = sum(1 for p in prep.values() if p['arm'] == 'flip')
    print('P2784 ARMS combo=%d flip=%d' % (len(prep) - n_flip, n_flip),
          flush=True)

    flips = 0
    per_row = []
    rng = np.random.default_rng(CTRL_SEED)
    for i in wrong_idx:
        if prep[i]['arm'] == 'combo':
            state['bsub'] = (ALPHA, torch.tensor(
                v_rows[i].astype(np.float32), device=device))
        else:
            state['bsub'] = (ALPHA, torch.tensor(
                prep[i]['v_flip'].astype(np.float32), device=device))
        try:
            m = arg_of(fwd(ids_full[i]))
        finally:
            state['bsub'] = None
        fl = bool(m == tgt_arr[i])
        flips += int(fl)
        cf = 0
        for _ in range(N_CTRL_DRAWS):
            r = rng.normal(size=v_rows[i].shape[0])
            r = r / np.linalg.norm(r)
            state['bsub'] = (ALPHA, torch.tensor(
                r.astype(np.float32), device=device))
            try:
                mc = arg_of(fwd(ids_full[i]))
            finally:
                state['bsub'] = None
            cf += int(mc == tgt_arr[i])
        per_row.append({'row': i, 'arm': prep[i]['arm'],
                        'pull': round(prep[i]['pull'], 4),
                        'flipped': fl, 'ctrl_flips': int(cf)})
        print('P2784 ROW %d arm=%s pull=%.3f flip=%s ctrl=%d'
              % (i, prep[i]['arm'], prep[i]['pull'], fl, cf), flush=True)

    coll_breaks = None
    coll_detail = []
    if correct_idx:
        samp = list(rng.choice(correct_idx,
                               size=min(N_COLL_ROWS, len(correct_idx)),
                               replace=False))
        brk = 0
        for i0 in samp:
            i = int(i0)
            if prep[i]['arm'] == 'combo':
                state['bsub'] = (ALPHA, torch.tensor(
                    v_rows[i].astype(np.float32), device=device))
            else:
                state['bsub'] = (ALPHA, torch.tensor(
                    prep[i]['v_flip'].astype(np.float32), device=device))
            try:
                m = arg_of(fwd(ids_full[i]))
            finally:
                state['bsub'] = None
            b = bool(m != arg_nat[i])
            brk += int(b)
            coll_detail.append({'row': i, 'broke': b})
        coll_breaks = int(brk)

    p2_pass = bool(flips >= 3 and
                   all(r['ctrl_flips'] == 0 for r in per_row))
    result['p2'] = {'pipeline_flips': int(flips),
                    'ctrl_total_flips': int(sum(r['ctrl_flips']
                                                for r in per_row)),
                    'pass': p2_pass}
    result['p3'] = {'coll_breaks': coll_breaks,
                    'n': len(coll_detail), 'detail': coll_detail}
    result['per_row'] = per_row
    if n_wrong < 8:
        result['verdict'] = 'insufficient_errors_14b'
    else:
        result['verdict'] = ('natural_pull_replicated_14b' if p2_pass
                             else 'natural_pull_failed_14b')
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'natural_pull_stats_14b.npz',
           arg_native=arg_nat, tgt=tgt_arr,
           wrong=np.array(wrong_idx, dtype=np.int64),
           pulls=np.array([prep[i]['pull'] for i in wrong_idx],
                          dtype=np.float64))
    for h in handles:
        h.remove()
    print('P2784 VERDICT %s flips=%d coll=%s'
          % (result['verdict'], flips, coll_breaks), flush=True)


if __name__ == '__main__':
    main()
