"""Phase 2786: deconstruct WHICH orthogonal component of v_row carries
the repair (close the 2782 geometry gap from the controlled side).

Background.  2780 C-X1 showed repair needs v_row's orthogonal structure
(pure oracle-axis subtraction 0/65); 2782 showed the repair content lies
OUTSIDE the span of the 65 controlled v_rows when applied to natural
text.  Before attacking the natural side we must know, on the controlled
panel itself, WHICH subcomponent of v_row does the repairing.

Setup.  Rows: the 21 pull<0 bsub-fixed rows (2772/2774 archival:
plain bsub alpha=0.3 @L35 repairs 21/21).  Basis: SVD of the 21 unit
v_rows -> left singular vectors e_1..e_8 (feature-space directions in
Vh rows).  Arms (all bsub alpha=0.3 @L35, v renormalised after edit):
  G1  baseline: bsub(v) must reproduce 21/21.
  A_nec(j):  v' = v - (v.e_j)e_j   (drop component j)  j=1..8
      necessary if flips <= 10 (less than half of 21).
  B_suf(j):  v' = (v.e_j)e_j       (component j alone) j=1..8
      sufficient if flips >= 8.
  C_null:    v' = v - (v.u)u with random unit u (10 draws/row, seed
      27860); and component-only null (v.u)u (10 draws/row).
Prereg (frozen before any forward):
  repair_carrier_localized iff any B_suf(j) >= 8/21
  repair_carrier_distributed iff all B_suf(j) <= 3/21 AND all
      A_nec(j) >= 15/21 (no single drop kills)
  otherwise: mixed (report counts).
Gates: G1; G2 unit-norm check on e_j.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2786' / 'qwen4_repair_carrier'
P2761 = BASE / 'phase2761' / 'qwen4_kc_fault'
P2763 = BASE / 'phase2763' / 'qwen4_debias_repair'
P2774 = BASE / 'phase2774' / 'qwen4_pull_validation'

ALPHA = 0.3
BSUB_LAYER = 35
N_PC = 8
N_NULL = 10
NULL_SEED = 27860

PREREG = {
    'G1': 'baseline bsub(v) reproduces 21/21 on the pull<0 bsub-fixed rows',
    'P1': 'necessary components: A_nec(j) <= 10/21',
    'P2': 'sufficient components: B_suf(j) >= 8/21',
    'verdict': 'repair_carrier_localized iff any B_suf >= 8; '
               'repair_carrier_distributed iff all B_suf <= 3 and all '
               'A_nec >= 15; else mixed',
}


def main():
    import torch
    OUT.mkdir(parents=True, exist_ok=True)

    z61 = np.load(P2761 / 'fault_scores.npz', allow_pickle=False)
    wrong_idx = np.array(sorted(int(i) for i in z61['wrong_idx']))
    zbd = np.load(P2763 / 'bias_dirs.npz', allow_pickle=False)
    v_rows = zbd['v_rows']
    v_norms = zbd['v_norms']
    zp = np.load(P2774 / 'pull_stats.npz', allow_pickle=False)
    pull = zp['pull']
    bsub_ok = zp['bsub'].astype(bool)
    assert (zp['wrong_idx'] == wrong_idx).all()

    import phase2747_rdc_material as mat2747
    material, data = mat2747.freeze()
    rows = [r for r in data['diagnostic']
            if r['kind'] == 'controlled_relation']
    tgt_ids = np.array([r['target'] for r in rows], dtype=np.int64)
    fam_arr = np.array([r['family'] for r in rows], dtype=np.str_)
    id_list = [r['prompt_ids'] for r in rows]

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG}
    fc.save(OUT / 'execution.json', execution)

    from phase2662_symmetric_mapping_contract import load_native
    model, tok = load_native('qwen4')
    model.eval()
    device = next(model.parameters()).device

    hook_state = {'on': False, 'alpha': None, 'vec': None}

    def readout_hook(module, args, output):
        if not hook_state['on']:
            return None
        out = output[0] if isinstance(output, tuple) else output
        h = out[0, -1]
        out[0, -1] = h - hook_state['alpha'] * h.norm() * hook_state['vec']
        return None

    handle = model.model.layers[BSUB_LAYER].register_forward_hook(
        readout_hook)

    def run_bsub(i, vec):
        hook_state['on'] = True
        hook_state['alpha'] = ALPHA
        hook_state['vec'] = torch.tensor(vec, device=device)
        try:
            with torch.inference_mode():
                o = model(torch.tensor([id_list[i]], device=device))
                m = int(o.logits[0, -1].float().argmax())
        finally:
            hook_state['on'] = False
        return m == tgt_ids[i]

    # rows: pull<0 & bsub-fixed
    sel = [k for k in range(len(wrong_idx))
           if pull[k] < 0 and bool(bsub_ok[k])]
    assert len(sel) == 21, len(sel)
    row_ids = [int(wrong_idx[k]) for k in sel]
    V = np.stack([v_rows[i] / v_norms[i] for i in row_ids])  # (21, d)

    # basis: feature-space directions = rows of Vh
    U, S, Vh = np.linalg.svd(V.astype(np.float64), full_matrices=False)
    E = Vh[:N_PC]                                   # (8, d)
    energy = (S[:N_PC] ** 2 / (S ** 2).sum()).tolist()

    # G1 baseline
    base = [int(run_bsub(i, V[ki])) for ki, i in enumerate(row_ids)]
    assert sum(base) == 21, ('G1 failed', sum(base))
    print('P2786 G1_OK baseline 21/21', flush=True)

    # A_nec: drop component j
    nec = {}
    for j in range(N_PC):
        e = E[j] / np.linalg.norm(E[j])
        flips = 0
        rows_j = {}
        for ki, i in enumerate(row_ids):
            v = V[ki]
            vf = v - float(v @ e) * e
            vf = vf / np.linalg.norm(vf)
            ok = run_bsub(i, vf)
            rows_j[str(i)] = bool(ok)
            flips += int(ok)
        nec[j] = {'flips': flips, 'rows': rows_j}
        print('P2786 A_nec j=%d flips=%d/21' % (j, flips), flush=True)

    # B_suf: component j alone
    suf = {}
    for j in range(N_PC):
        e = E[j] / np.linalg.norm(E[j])
        flips = 0
        rows_j = {}
        for ki, i in enumerate(row_ids):
            v = V[ki]
            vf = float(v @ e) * e
            n = np.linalg.norm(vf)
            if n < 1e-8:
                rows_j[str(i)] = False
                continue
            vf = vf / n
            ok = run_bsub(i, vf)
            rows_j[str(i)] = bool(ok)
            flips += int(ok)
        suf[j] = {'flips': flips, 'rows': rows_j}
        print('P2786 B_suf j=%d flips=%d/21' % (j, flips), flush=True)

    # nulls
    rng = np.random.default_rng(NULL_SEED)
    null_drop = np.zeros((len(row_ids), N_NULL), dtype=np.int64)
    null_alone = np.zeros((len(row_ids), N_NULL), dtype=np.int64)
    for ki, i in enumerate(row_ids):
        v = V[ki]
        for d in range(N_NULL):
            u = rng.standard_normal(V.shape[1])
            u /= np.linalg.norm(u)
            vd = v - float(v @ u) * u
            vd /= np.linalg.norm(vd)
            null_drop[ki, d] = int(run_bsub(i, vd))
            va = float(v @ u) * u
            na = np.linalg.norm(va)
            if na > 1e-8:
                va /= na
                null_alone[ki, d] = int(run_bsub(i, va))
    print('P2786 NULL drop=%d alone=%d (of %d each)'
          % (null_drop.sum(), null_alone.sum(),
             len(row_ids) * N_NULL), flush=True)

    suf_flips = [suf[j]['flips'] for j in range(N_PC)]
    nec_flips = [nec[j]['flips'] for j in range(N_PC)]
    localized = any(f >= 8 for f in suf_flips)
    distributed = all(f <= 3 for f in suf_flips) and \
        all(f >= 15 for f in nec_flips)
    verdict = {
        'verdict': 'repair_carrier_localized' if localized else
                   ('repair_carrier_distributed' if distributed
                    else 'mixed'),
        'energy_top8': energy,
        'sufficient_flips': suf_flips,
        'necessary_flips': nec_flips,
        'null_drop_total': int(null_drop.sum()),
        'null_alone_total': int(null_alone.sum()),
    }
    result = {'phase': 2786, 'prereg': PREREG, 'verdict': verdict,
              'row_ids': row_ids,
              'sufficient_rows': {str(j): suf[j]['rows']
                                  for j in range(N_PC)},
              'necessary_rows': {str(j): nec[j]['rows']
                                 for j in range(N_PC)},
              'null_drop': null_drop.tolist(),
              'null_alone': null_alone.tolist()}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'carrier_stats.npz',
           row_ids=np.array(row_ids, dtype=np.int64),
           S=S, energy=np.array(energy),
           nec=np.array(nec_flips), suf=np.array(suf_flips),
           null_drop=null_drop, null_alone=null_alone)
    print('P2786 VERDICT %s' % json.dumps(verdict), flush=True)


if __name__ == '__main__':
    main()
