"""Phase 2786b: leave-one-out validation of the single-component
repair carrier (out-of-sample gate for 2786).

2786 found B_suf(0)=21/21 (component e0 of the in-sample SVD alone
fully repairs).  But e0 was computed FROM the 21 rows -- in-sample.
Here: for each row ki, e0_loo = top right-singular vector of the OTHER
20 unit v_rows; intervention vf = (v_ki.e0_loo)e0_loo (renormalised),
bsub alpha=0.3 @L35.
Prereg (frozen before any forward):
  carrier_generalizes iff LOO repairs >= 15/21;
  in_sample_artifact iff LOO <= 7/21; else partial.
Control: cos(e0_loo, e0_full) recorded; also negative control vf =
  (v_ki.e1_loo)e1_loo (second component, expect low repair).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2786' / 'qwen4_repair_carrier_loo'
P2761 = BASE / 'phase2761' / 'qwen4_kc_fault'
P2763 = BASE / 'phase2763' / 'qwen4_debias_repair'
P2774 = BASE / 'phase2774' / 'qwen4_pull_validation'

ALPHA = 0.3
BSUB_LAYER = 35

PREREG = {
    'loo': 'carrier_generalizes iff LOO e0-component repairs >= 15/21; '
           'in_sample_artifact iff <= 7/21; else partial',
    'ctrl': 'e1_loo component-only repair recorded as negative control',
}


def main():
    import torch
    OUT.mkdir(parents=True, exist_ok=True)

    z61 = np.load(P2761 / 'fault_scores.npz', allow_pickle=False)
    wrong_idx = np.array(sorted(int(i) for i in z61['wrong_idx']))
    zbd = np.load(P2763 / 'bias_dirs.npz', allow_pickle=False)
    v_rows, v_norms = zbd['v_rows'], zbd['v_norms']
    zp = np.load(P2774 / 'pull_stats.npz', allow_pickle=False)
    pull, bsub_ok = zp['pull'], zp['bsub'].astype(bool)
    assert (zp['wrong_idx'] == wrong_idx).all()

    import phase2747_rdc_material as mat2747
    material, data = mat2747.freeze()
    rows = [r for r in data['diagnostic']
            if r['kind'] == 'controlled_relation']
    tgt_ids = np.array([r['target'] for r in rows], dtype=np.int64)
    id_list = [r['prompt_ids'] for r in rows]

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG}
    fc.save(OUT / 'execution.json', execution)

    sel = [k for k in range(len(wrong_idx))
           if pull[k] < 0 and bool(bsub_ok[k])]
    assert len(sel) == 21
    row_ids = [int(wrong_idx[k]) for k in sel]
    V = np.stack([v_rows[i] / v_norms[i] for i in row_ids])

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

    loo_rows = {}
    loo_flips = 0
    e1_flips = 0
    cos_l = []
    for ki in range(len(row_ids)):
        others = np.delete(V, ki, axis=0)
        _, _, Vh = np.linalg.svd(others.astype(np.float64),
                                 full_matrices=False)
        e0 = Vh[0] / np.linalg.norm(Vh[0])
        e1 = Vh[1] / np.linalg.norm(Vh[1])
        _, _, Vhf = np.linalg.svd(V.astype(np.float64),
                                  full_matrices=False)
        cos_l.append(float(e0 @ (Vhf[0] / np.linalg.norm(Vhf[0]))))
        i = row_ids[ki]
        v = V[ki]
        vf = float(v @ e0) * e0
        vf /= np.linalg.norm(vf)
        ok = run_bsub(i, vf)
        loo_rows[str(i)] = bool(ok)
        loo_flips += int(ok)
        vf1 = float(v @ e1) * e1
        n1 = np.linalg.norm(vf1)
        if n1 > 1e-8:
            vf1 /= n1
            e1_flips += int(run_bsub(i, vf1))
        print('P2786B LOO ki=%d row=%d flip=%s' % (ki, i, ok), flush=True)

    verdict = {
        'verdict': ('carrier_generalizes' if loo_flips >= 15 else
                    ('in_sample_artifact' if loo_flips <= 7
                     else 'partial')),
        'loo_flips': loo_flips, 'n': 21,
        'e1_loo_flips': e1_flips,
        'cos_e0loo_e0full_mean': float(np.mean(cos_l)),
    }
    result = {'phase': '2786b', 'prereg': PREREG, 'verdict': verdict,
              'row_ids': row_ids, 'loo_rows': loo_rows}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'loo_stats.npz',
           row_ids=np.array(row_ids, dtype=np.int64),
           loo=np.array([int(loo_rows[str(i)]) for i in row_ids]))
    print('P2786B VERDICT %s' % json.dumps(verdict), flush=True)


if __name__ == '__main__':
    main()
