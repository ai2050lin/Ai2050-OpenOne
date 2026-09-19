"""Phase 2786c: e0-component-only repair recipe on ALL 65 wrong rows.

2786/2786b established: the single shared anchor direction e0 (top PC
of the 21 pull<0 bsub-fixed v_rows) alone fully repairs those 21 rows,
out-of-sample (LOO 21/21).  Descriptive extension: apply the SAME
frozen recipe (e0 from the 21 rows, vf = (v.e0)e0 renormalised, bsub
alpha=0.3 @L35) to all 65 controlled wrong rows.  Questions:
  Q1 does it stay safe on the 21 (no regression)?
  Q2 does it repair rows where FULL v_row bsub failed (44 rows)?
  Q3 family decomposition of any extra flips.
No prereg gate beyond descriptive; null = e1-component on the 44
bsub-failed rows (10 draws of random unit vector on a 5-row subset for
scale).  Frozen before any forward: e0 computed from the archival 21
rows only.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2786' / 'qwen4_e0_recipe_all65'
P2761 = BASE / 'phase2761' / 'qwen4_kc_fault'
P2763 = BASE / 'phase2763' / 'qwen4_debias_repair'
P2774 = BASE / 'phase2774' / 'qwen4_pull_validation'

ALPHA = 0.3
BSUB_LAYER = 35

PREREG = {
    'design': 'e0 frozen from the 21 pull<0 bsub-fixed rows; recipe '
              'vf=(v.e0)e0 unit; bsub alpha=0.3 @L35; descriptive on 65',
    'q1': 'the 21 archival bsub-fixed rows must remain repaired (>=19)',
    'q2': 'extra flips on the 44 failed rows recorded (any >= 8 would '
          'beat the archival full-v recipe on this panel)',
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
    fam_arr = np.array([r['family'] for r in rows], dtype=np.str_)
    id_list = [r['prompt_ids'] for r in rows]

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG}
    fc.save(OUT / 'execution.json', execution)

    # freeze e0 from the 21
    sel21 = [k for k in range(len(wrong_idx))
             if pull[k] < 0 and bool(bsub_ok[k])]
    assert len(sel21) == 21
    ids21 = [int(wrong_idx[k]) for k in sel21]
    V21 = np.stack([v_rows[i] / v_norms[i] for i in ids21])
    _, _, Vh = np.linalg.svd(V21.astype(np.float64), full_matrices=False)
    e0 = Vh[0] / np.linalg.norm(Vh[0])
    e1 = Vh[1] / np.linalg.norm(Vh[1])

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

    res = {}
    flips_by_fam = {}
    extra_rows = []
    for k, i in enumerate(wrong_idx):
        i = int(i)
        v = v_rows[i] / v_norms[i]
        vf = float(v @ e0) * e0
        n = np.linalg.norm(vf)
        ok = False
        if n > 1e-8:
            vf /= n
            ok = run_bsub(i, vf)
        res[str(i)] = bool(ok)
        f = str(fam_arr[i])
        flips_by_fam.setdefault(f, [0, 0])
        flips_by_fam[f][1] += 1
        flips_by_fam[f][0] += int(ok)
        if ok and not bool(bsub_ok[k]):
            extra_rows.append(i)
    total = sum(int(x) for x in res.values())
    kept21 = sum(int(res[str(i)]) for i in ids21)
    print('P2786C total=%d/65 kept21=%d/21 extra=%d'
          % (total, kept21, len(extra_rows)), flush=True)
    print('P2786C fam %s' % json.dumps(flips_by_fam), flush=True)

    # negative control: e1-component on the 44 failed rows
    ctrl_flips = 0
    for k, i in enumerate(wrong_idx):
        if bool(bsub_ok[k]):
            continue
        i = int(i)
        v = v_rows[i] / v_norms[i]
        vf = float(v @ e1) * e1
        n = np.linalg.norm(vf)
        if n > 1e-8:
            vf /= n
            ctrl_flips += int(run_bsub(i, vf))
    print('P2786C e1_ctrl_flips=%d/44' % ctrl_flips, flush=True)

    verdict = {'total_flips': total, 'kept21': kept21,
               'extra_rows': extra_rows,
               'extra_by_fam': {str(fam_arr[i]): None for i in []},
               'flips_by_fam': flips_by_fam,
               'e1_ctrl_flips': ctrl_flips,
               'q1_pass': bool(kept21 >= 19)}
    result = {'phase': '2786c', 'prereg': PREREG, 'verdict': verdict,
              'flip_rows': res,
              'wrong_idx': [int(i) for i in wrong_idx]}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'recipe_stats.npz',
           wrong_idx=wrong_idx.astype(np.int64),
           flips=np.array([int(res[str(int(i))])
                           for i in wrong_idx], dtype=np.int64),
           pull=pull, bsub_arch=bsub_ok.astype(np.int64))
    print('P2786C VERDICT %s' % json.dumps(verdict), flush=True)


if __name__ == '__main__':
    main()
