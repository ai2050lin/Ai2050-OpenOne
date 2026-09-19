"""Phase 2781: collateral governance for the natural-language pipeline.

Phase 2780 established natural-language repair (9/35, ctrl 0) but failed
P-G3 (collateral 7/13) at the fixed alpha=0.3.  This phase sweeps the
intervention strength on the SAME 48 frozen items with the SAME v_row
recipe and the SAME combo arm (pull<0 -> top-4 carrier-head ablation +
bsub), preregistering a tradeoff criterion.

Preregistered (frozen before any forward):
  T1: alpha grid {0.1, 0.2, 0.3} (frozen); combo arm on all natural wrong
      rows (repair) and all natural native-correct rows (breakage) per
      alpha.
  T2: collateral_governed iff there EXISTS alpha in the grid with
      repair_rate >= 0.2 (>= 7/35) AND collateral <= 2/13; else
      'governance_failed' and the tradeoff curve is the result.
Gates: G1 native args reproduce 2780 (35 wrong / 13 correct of 37
  single-token rows); G2 v_norms > 0.
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
OUT = BASE / 'phase2781' / 'qwen4_collateral_governance'

M_LAYERS = [31, 32, 33, 34, 35]
TOP_HEADS = 4
ALPHA_GRID = [0.1, 0.2, 0.3]
ITEMS = p2780.ITEMS

PREREG = {
    'T1': 'alpha grid {0.1,0.2,0.3} frozen; combo arm on natural wrong and '
          'native-correct rows per alpha',
    'T2': 'collateral_governed iff exists alpha with repair >= 7/35 AND '
          'collateral <= 2/13; else governance_failed + tradeoff curve',
    'verdict': 'recorded per T2',
}


def main():
    import torch
    from transformers import AutoTokenizer
    OUT.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(
        str(cc.ROOT / 'models' / 'hf' / 'qwen3-4b'), local_files_only=True,
        trust_remote_code=True, use_fast=True)

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'alpha_grid': ALPHA_GRID}
    fc.save(OUT / 'execution.json', execution)

    from phase2662_symmetric_mapping_contract import load_native
    model, tok2 = load_native('qwen4')
    model.eval()
    device = next(model.parameters()).device
    n_heads = model.config.num_attention_heads
    qdim = model.model.layers[0].self_attn.q_proj.out_features
    hd = qdim // n_heads
    W_U = model.lm_head.weight.detach().float().cpu().numpy()
    W_O = {l: model.model.layers[l].self_attn.o_proj.weight.detach().
           float().cpu().numpy() for l in M_LAYERS}

    state = {'bsub': None, 'abl': frozenset()}
    cap = {'on': False, 'oproj': {}}
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

    def make_abl_pre(l):
        def pre(module, args):
            hs_ = [h for (ll, h) in state['abl'] if ll == l]
            if not hs_:
                return None
            x = args[0].clone()
            for h in hs_:
                x[0, -1, h * hd:(h + 1) * hd] = 0
            return (x,) + tuple(args[1:])
        return pre

    def make_cap_pre(l):
        def pre(module, args):
            if cap['on']:
                cap['oproj'][l] = args[0].detach()
            return None
        return pre

    handles.append(model.model.layers[35].register_forward_hook(
        make_bsub_hook()))
    # v_row definition MUST match 2780/archival 2763: h at L35 DECODER
    # OUTPUT (pre-final-norm), NOT hidden_states[-1] (post-norm).  A
    # post-norm variant was run by accident and collapsed repair to 1/35
    # at alpha=0.3 (vs 9/35) -- recorded as an incidental finding.
    cap_hfin = {'h': None}

    def make_fin_hook():
        def hook(module, args, output):
            cap_hfin['h'] = output.detach()
            return None
        return hook

    handles.append(model.model.layers[35].register_forward_hook(
        make_fin_hook()))
    for l in M_LAYERS:
        handles.append(model.model.layers[l].self_attn.o_proj.
                       register_forward_pre_hook(make_abl_pre(l)))
        handles.append(model.model.layers[l].self_attn.o_proj.
                       register_forward_pre_hook(make_cap_pre(l)))

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
    n_wrong = int(wrong_mask.sum())
    print('P2781 G1_OK wrong=%d' % n_wrong, flush=True)

    v_rows = []
    for i in range(len(ITEMS)):
        v = h_final(ids_full[i]) - h_final(ids_wo[i])
        nrm = float(np.linalg.norm(v))
        assert nrm > 0, ('G2 degenerate', i)
        v_rows.append(v / nrm)

    def top4_heads(ids, u):
        cap['on'] = True
        cap['oproj'].clear()
        try:
            fwd(ids)
        finally:
            cap['on'] = False
        gains = {}
        for l in M_LAYERS:
            X = cap['oproj'][l][0, -1].float().cpu().numpy()
            Wl = W_O[l]
            for h in range(n_heads):
                sl = slice(h * hd, (h + 1) * hd)
                gains[(l, h)] = float(u @ (Wl[:, sl] @ X[sl]))
        return [(l, h) for (l, h), g in
                sorted(gains.items(), key=lambda x: -x[1])
                if g > 0][:TOP_HEADS]

    wrong_idx = [int(i) for i in np.where(wrong_mask)[0]]
    # correct pool = all rows natively correct at the first-token criterion
    # (2 single-token-correct + 11 multi-token answers whose first token the
    # model reproduces) -- identical composition to the 2780 P-G3 pool.
    correct_idx = [int(i) for i in np.where(~wrong_mask)[0]]
    n_correct = len(correct_idx)
    n_single_correct = int(sum(1 for i in correct_idx if not multi[i]))
    assert n_wrong == 35 and n_correct == 13, (n_wrong, n_correct)
    print('P2781 POOL correct=%d (single %d, multi-first-token %d)'
          % (n_correct, n_single_correct, n_correct - n_single_correct),
          flush=True)

    # per-row rival + heads (alpha-independent)
    prep = {}
    for i in wrong_idx + correct_idx:
        zi = fwd(ids_full[i]).logits[0, -1].float()
        tmp = zi.clone()
        tmp[tgt_arr[i]] = -1e30
        riv = int(tmp.argmax())
        u = W_U[riv] - W_U[tgt_arr[i]]
        u = u / np.linalg.norm(u)
        prep[i] = {'rival': riv, 'heads': top4_heads(ids_full[i], u)}

    curve = {}
    for alpha in ALPHA_GRID:
        rep = 0
        for i in wrong_idx:
            state['abl'] = frozenset(prep[i]['heads'])
            state['bsub'] = (alpha, torch.tensor(v_rows[i], device=device))
            try:
                m = arg_of(fwd(ids_full[i]))
            finally:
                state['abl'] = frozenset()
                state['bsub'] = None
            rep += int(m == tgt_arr[i])
        brk = 0
        for i in correct_idx:
            state['abl'] = frozenset(prep[i]['heads'])
            state['bsub'] = (alpha, torch.tensor(v_rows[i], device=device))
            try:
                m = arg_of(fwd(ids_full[i]))
            finally:
                state['abl'] = frozenset()
                state['bsub'] = None
            brk += int(m != arg_nat[i])
        curve[str(alpha)] = {'repair': int(rep), 'n_wrong': n_wrong,
                             'collateral': int(brk), 'n_correct': n_correct}
        print('P2781 A=%.2f repair=%d/%d coll=%d/%d'
              % (alpha, rep, n_wrong, brk, n_correct), flush=True)

    governed = any(c['repair'] >= 7 and c['collateral'] <= 2
                   for c in curve.values())
    verdict = {'collateral_governed': bool(governed),
               'governing_alpha': [a for a in ALPHA_GRID
                                   if curve[str(a)]['repair'] >= 7 and
                                   curve[str(a)]['collateral'] <= 2],
               'curve': curve}
    result = {'phase': 2781, 'prereg': PREREG, 'verdict': verdict}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'governance_stats.npz',
           alphas=np.array(ALPHA_GRID),
           repair=np.array([curve[str(a)]['repair'] for a in ALPHA_GRID]),
           collateral=np.array([curve[str(a)]['collateral']
                                for a in ALPHA_GRID]))
    for h in handles:
        h.remove()
    print('P2781 VERDICT %s' % json.dumps(verdict), flush=True)


if __name__ == '__main__':
    main()
