"""Phase 2782: direction regularization for the natural-language pipeline.

Phase 2781 localized the natural-language collateral bottleneck to
DIRECTION QUALITY (collateral insensitive to alpha: 4/13 even at 0.1
with ~0 repair).  Hypothesis: the natural span-deletion v_row carries
noise components unrelated to the readout-competition bias structure
that the controlled-panel v_rows encode.  Regularize by projecting the
natural v_row onto the principal subspace of the 65 controlled
wrong-row v_rows (archival 2763 bias_dirs.npz) with residual shrinkage.

Preregistered (frozen before any forward):
  T1: subspace S_k = top-k left singular vectors (k in {4,8,16,32,65},
      frozen) of the 65 unit controlled v_rows; regularization
      v_reg = P_k v + lam * (v - P_k v), lam in {0.0, 0.5} (frozen),
      then unit-normalized.  alpha fixed 0.3.  Arms identical to 2780
      dictated arms (pull<0 -> combo top-4 heads + bsub; pull>=0 ->
      sign-flip bsub, ORACLE-GUIDED as flagged in 2778) with the arm
      chosen by the ORIGINAL (unregularized) pull sign -- frozen.
  T2: direction_governed iff there EXISTS (k, lam) with repair >= 7/35
      AND collateral <= 2/13; else 'regularization_failed' + grid table.
  B0 (sanity gate): the identity config (k=65 full rank IS the whole
      span of the 65 v_rows, lam=0 projection-only is a real config;
      unregularized baseline lam=1 equivalent) must reproduce the 2780
      alpha=0.3 dictated-arm numbers: repair 9/35, collateral 7/13
      within +/-1 -- implemented as explicit no-projection run (k=0
      excluded from grid; baseline = plain v, recorded as 'plain').
Gates: G1 native args reproduce 2780/2781 (35 wrong / 13 correct of 48
  at the first-token criterion); G2 v_norms > 0; G3 subspace built ONLY
  from controlled 2763 v_rows (no natural item used).
Descriptive: per-config energy fraction ||P_k v||/||v|| on wrong rows.
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
P2761 = BASE / 'phase2761' / 'qwen4_kc_fault'
P2763 = BASE / 'phase2763' / 'qwen4_debias_repair'
P2774 = BASE / 'phase2774' / 'qwen4_pull_validation'

M_LAYERS = [31, 32, 33, 34, 35]
TOP_HEADS = 4
ALPHA = 0.3
K_GRID = [4, 8, 16, 32, 65]
LAM_GRID = [0.0, 0.5]
ITEMS = p2780.ITEMS

PREREG = {
    'T1': 'subspace top-k of 65 controlled v_rows, k in {4,8,16,32,65}; '
          'v_reg = P_k v + lam (v - P_k v), lam in {0.0,0.5}; alpha=0.3; '
          'arms per ORIGINAL pull sign',
    'T2': 'direction_governed iff exists (k,lam) with repair >= 7/35 AND '
          'collateral <= 2/13; else regularization_failed + grid table',
    'B0': 'plain-v baseline must reproduce 2780 dictated-arm numbers '
          'repair 9/35 collateral 7/13 within +/-1',
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
                 'prereg': PREREG, 'k_grid': K_GRID, 'lam_grid': LAM_GRID}
    fc.save(OUT / 'execution.json', execution)

    # ---- G3: build the subspace FIRST, from archival controlled rows only
    z61 = np.load(P2761 / 'fault_scores.npz', allow_pickle=False)
    c_wrong = np.array(sorted(int(i) for i in z61['wrong_idx']))
    zbd = np.load(P2763 / 'bias_dirs.npz', allow_pickle=False)
    # v_rows in 2763 are UNNORMALIZED (norms 26-90 stored in v_norms);
    # unit-normalize per row before SVD so large-norm rows do not dominate.
    v_c = zbd['v_rows'][c_wrong]   # raw rows of the 65 controlled wrong rows
    vn_c = zbd['v_norms'][c_wrong]
    v_c = v_c / np.linalg.norm(v_c, axis=1, keepdims=True)
    assert abs(zbd['v_norms'][c_wrong].min() - vn_c.min()) < 1e-3
    assert np.allclose(np.linalg.norm(v_c, axis=1), 1.0, atol=1e-5)
    assert v_c.shape[0] == len(c_wrong), (v_c.shape, len(c_wrong))
    assert (np.array(sorted(int(i) for i in zbd['wrong_idx'])) ==
            c_wrong).all(), '2763/2761 wrong_idx mismatch'
    U, S, Vh = np.linalg.svd(v_c.astype(np.float64), full_matrices=False)
    # feature-space direction subspace = ROW space of v_c = rows of Vh;
    # Basis65 = Vh.T (2560, 65) columns are orthonormal directions.
    print('P2782 G3_OK subspace from %d controlled v_rows, top-5 sing val '
          'frac %s' % (v_c.shape[0],
                       np.round((S[:5] ** 2 / (S ** 2).sum()), 4).tolist()),
          flush=True)

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

    def make_fin_hook():
        def hook(module, args, output):
            cap_hfin['h'] = output.detach()
            return None
        return hook

    handles.append(model.model.layers[35].register_forward_hook(
        make_bsub_hook()))
    cap_hfin = {'h': None}
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

    # ---- natural items, identical recipe to 2780/2781 (L35 pre-norm)
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
    wrong_idx = [int(i) for i in np.where(wrong_mask)[0]]
    correct_idx = [int(i) for i in np.where(~wrong_mask)[0]]
    n_wrong, n_correct = len(wrong_idx), len(correct_idx)
    assert n_wrong == 35 and n_correct == 13, (n_wrong, n_correct)
    print('P2782 G1_OK wrong=%d correct=%d' % (n_wrong, n_correct),
          flush=True)

    v_rows, v_norms = {}, {}
    for i in wrong_idx + correct_idx:
        v = h_final(ids_full[i]) - h_final(ids_wo[i])
        nrm = float(np.linalg.norm(v))
        assert nrm > 0, ('G2 degenerate', i)
        v_rows[i] = v / nrm
        v_norms[i] = nrm
    print('P2782 G2_OK v_norms>0 on %d rows' % len(v_rows), flush=True)

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

    # ---- per-row prep: rival, original pull, arm, heads (alpha-independent)
    prep = {}
    for i in wrong_idx + correct_idx:
        zi = fwd(ids_full[i]).logits[0, -1].float()
        tmp = zi.clone()
        tmp[tgt_arr[i]] = -1e30
        riv = int(tmp.argmax())
        d_uv = W_U[tgt_arr[i]] - W_U[riv]
        pl = float(v_rows[i] @ d_uv)
        if pl < 0:
            u_gain = (W_U[riv] - W_U[tgt_arr[i]]) / \
                np.linalg.norm(W_U[riv] - W_U[tgt_arr[i]])
            prep[i] = {'rival': riv, 'pull': pl, 'arm': 'combo',
                       'heads': top4_heads(ids_full[i], u_gain)}
        else:
            u_ro = d_uv / np.linalg.norm(d_uv)
            v_flip = v_rows[i] - 2.0 * pl * u_ro  # v.u_ro == pl
            v_flip = v_flip / np.linalg.norm(v_flip)
            prep[i] = {'rival': riv, 'pull': pl, 'arm': 'flip',
                       'v_flip': v_flip}
    n_flip_arm = sum(1 for p in prep.values() if p['arm'] == 'flip')
    print('P2782 ARMS combo=%d flip=%d (incl correct rows)'
          % (len(prep) - n_flip_arm, n_flip_arm), flush=True)

    energy = {i: float(np.linalg.norm(v_rows[i] @ Vh.T))
              for i in wrong_idx}

    def run_cfg(tag, vreg):
        rep = 0
        detail = []
        for i in wrong_idx:
            if prep[i]['arm'] == 'combo':
                state['abl'] = frozenset(prep[i]['heads'])
                state['bsub'] = (ALPHA, torch.tensor(
                    vreg[i].astype(np.float32), device=device))
            else:
                state['abl'] = frozenset()
                state['bsub'] = (ALPHA, torch.tensor(
                    prep[i]['v_flip'].astype(np.float32), device=device))
            try:
                m = arg_of(fwd(ids_full[i]))
            finally:
                state['abl'] = frozenset()
                state['bsub'] = None
            fl = bool(m == tgt_arr[i])
            rep += int(fl)
            detail.append({'row': i, 'flipped': fl})
        brk = 0
        for i in correct_idx:
            if prep[i]['arm'] == 'combo':
                state['abl'] = frozenset(prep[i]['heads'])
                state['bsub'] = (ALPHA, torch.tensor(
                    vreg[i].astype(np.float32), device=device))
            else:
                state['abl'] = frozenset()
                state['bsub'] = (ALPHA, torch.tensor(
                    prep[i]['v_flip'].astype(np.float32), device=device))
            try:
                m = arg_of(fwd(ids_full[i]))
            finally:
                state['abl'] = frozenset()
                state['bsub'] = None
            brk += int(m != arg_nat[i])
        print('P2782 CFG %s repair=%d/%d coll=%d/%d'
              % (tag, rep, n_wrong, brk, n_correct), flush=True)
        return {'tag': tag, 'repair': int(rep), 'collateral': int(brk),
                'detail': detail}

    # ---- B0 sanity gate: plain v (unregularized)
    b0 = run_cfg('plain', v_rows)
    b0_ok = abs(b0['repair'] - 9) <= 1 and abs(b0['collateral'] - 7) <= 1
    assert b0_ok, ('B0 baseline mismatch with 2780',
                   b0['repair'], b0['collateral'])
    print('P2782 B0_OK plain reproduces 2780 dictated-arm baseline',
          flush=True)

    grid = {}
    for k in K_GRID:
        for lam in LAM_GRID:
            Bk = Vh.T[:, :k]                   # (2560, k) orthonormal basis
            vreg = {}
            for i in wrong_idx + correct_idx:
                v = v_rows[i]
                pv = Bk @ (Bk.T @ v)
                r = pv + lam * (v - pv)
                nrm = float(np.linalg.norm(r))
                vreg[i] = r / nrm if nrm > 0 else v
            grid['k%d_lam%.1f' % (k, lam)] = run_cfg(
                'k%d_lam%.1f' % (k, lam), vreg)

    governed = [(g['tag'], g['repair'], g['collateral'])
                for g in grid.values()
                if g['repair'] >= 7 and g['collateral'] <= 2]
    verdict = {'direction_governed': bool(governed),
               'governing_cfgs': governed,
               'baseline': {'repair': b0['repair'],
                            'collateral': b0['collateral']},
               'grid': {t: {'repair': g['repair'], 'collateral':
                            g['collateral']}
                        for t, g in grid.items()},
               'energy_frac_wrong_mean': float(np.mean(list(
                   energy.values())))}
    result = {'phase': 2782, 'prereg': PREREG, 'verdict': verdict,
              'energy_frac': {str(i): round(energy[i], 4)
                              for i in wrong_idx}}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'regularization_stats.npz',
           S=S, K=np.array(K_GRID), LAM=np.array(LAM_GRID),
           repair=np.array([grid[t]['repair'] for t in grid]),
           collateral=np.array([grid[t]['collateral'] for t in grid]),
           tags=np.array(list(grid.keys())))
    for h in handles:
        h.remove()
    print('P2782 VERDICT %s' % json.dumps(
        {k2: v2 for k2, v2 in verdict.items() if k2 != 'grid'}),
        flush=True)


if __name__ == '__main__':
    main()
