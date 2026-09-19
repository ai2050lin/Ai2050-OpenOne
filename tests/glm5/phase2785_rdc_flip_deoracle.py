"""Phase 2785: de-oracle the flip arm (remove W_U[target] dependency).

Background.  Phase 2775 proved the sign-flip transform
  v' = v - 2*(v.u_ro)u_ro,  u_ro = unit(W_U[target]-W_U[rival])
causally gates repair (flip-up 12/24, flip-down 21->0, null 0/65), but
u_ro uses the TARGET embedding -- an oracle.  A mechanism is only
'understood' if intervention works from model-side information alone.

Arms (all use the SAME transform, only the axis u differs):
  O  oracle reference: u = unit(W_U[target]-W_U[rival])
     Gate O: must reproduce exactly 12/24 flips (harness check).
  D1 rival axis:      u = unit(W_U[rival])
  D2 argmax axis:     u = unit(W_U[argmax_native])   (ground-truth-free;
        assert argmax_native == rival on all 24 rows)
  D3 centered axis:   u = unit(W_U[rival] - mean_vocab(W_U))
Null: per-row random unit axis (10 draws, seed 27850), same transform,
      v_flip renormalised identically.

Rows: the same 24 pull>0 controlled rows as 2775 C1a (includes 8 neg).
Prereg (frozen before any forward):
  G1  native argmax on the 24 rows equals 2763 archival arg_native.
  GO  oracle arm flips == 12/24.
  N1  deoracle_ok iff any D-arm flips >= 4 AND > null q95 of its rows.
  Descriptive: family decomposition of the best D-arm; per-row overlap
  with the oracle flip set.
Verdict: flip_deoracled iff GO and N1.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2785' / 'qwen4_flip_deoracle'
P2761 = BASE / 'phase2761' / 'qwen4_kc_fault'
P2763 = BASE / 'phase2763' / 'qwen4_debias_repair'
P2774 = BASE / 'phase2774' / 'qwen4_pull_validation'

ALPHA = 0.3
BSUB_LAYER = 35
N_NULL = 10
NULL_SEED = 27850

PREREG = {
    'G1': 'native argmax on the 24 pull>0 rows equals archival arg_native',
    'GO': 'oracle arm reproduces exactly 12/24 flips (2775 C1a)',
    'N1': 'deoracle_ok iff any D-arm (D1/D2/D3) flips >= 4/24 AND > '
          'null q95 (per-row random-axis null, 10 draws)',
    'verdict': 'flip_deoracled iff GO and N1',
}


def main():
    import torch
    OUT.mkdir(parents=True, exist_ok=True)

    z61 = np.load(P2761 / 'fault_scores.npz', allow_pickle=False)
    wrong_idx = np.array(sorted(int(i) for i in z61['wrong_idx']))
    zbd = np.load(P2763 / 'bias_dirs.npz', allow_pickle=False)
    v_rows = zbd['v_rows']
    v_norms = zbd['v_norms']
    zbeh = np.load(P2763 / 'behaviour_scores.npz', allow_pickle=False)
    arg_native = zbeh['native__base_arg']
    zp = np.load(P2774 / 'pull_stats.npz', allow_pickle=False)
    pull = zp['pull']
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
    W_U = model.lm_head.weight.detach().float().cpu().numpy()
    wu_mean = W_U.mean(axis=0)

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

    # rows
    pos_k = list(np.where(pull > 0)[0])
    assert len(pos_k) == 24
    row_ids = [int(wrong_idx[k]) for k in pos_k]
    rivals = [int(zp['rival_ids'][k]) for k in pos_k]

    # G1 native verify
    arg_check = {}
    for i in row_ids:
        with torch.inference_mode():
            o = model(torch.tensor([id_list[i]], device=device))
            arg_check[i] = int(o.logits[0, -1].float().argmax())
        assert arg_check[i] == int(arg_native[i]), ('native drift', i)
    print('P2785 G1_OK native verified on 24 rows', flush=True)

    # D2 sanity: argmax_native == rival
    assert all(arg_native[i] == r for i, r in zip(row_ids, rivals)), \
        'argmax != rival on some rows'

    rng = np.random.default_rng(NULL_SEED)

    def arm_flips(axis_fn, tag, rows_dict):
        flips = 0
        for k, i in enumerate(row_ids):
            v = v_rows[i] / v_norms[i]
            u = axis_fn(k, i)
            u = u / np.linalg.norm(u)
            vf = v - 2.0 * float(v @ u) * u
            vf = vf / np.linalg.norm(vf)
            ok = run_bsub(i, vf)
            rows_dict[str(i)] = bool(ok)
            flips += int(ok)
        print('P2785 ARM %s flips=%d/24' % (tag, flips), flush=True)
        return flips

    res_rows = {}
    # O oracle reference (gate)
    res_rows['O'] = {}
    f_O = arm_flips(
        lambda k, i: W_U[tgt_ids[i]] - W_U[rivals[k]],
        'O_oracle', res_rows['O'])
    assert f_O == 12, ('oracle gate failed', f_O)
    print('P2785 GO_OK oracle reproduces 12/24', flush=True)

    # D1 rival axis
    res_rows['D1'] = {}
    f_D1 = arm_flips(lambda k, i: W_U[rivals[k]], 'D1_rival', res_rows['D1'])

    # D2 argmax axis (ground-truth-free)
    res_rows['D2'] = {}
    f_D2 = arm_flips(
        lambda k, i: W_U[int(arg_native[i])], 'D2_argmax', res_rows['D2'])

    # D3 centered rival axis
    res_rows['D3'] = {}
    f_D3 = arm_flips(
        lambda k, i: W_U[rivals[k]] - wu_mean, 'D3_centered', res_rows['D3'])

    # per-row random-axis null
    rng_state = rng
    null_flips = np.zeros((len(row_ids), N_NULL), dtype=np.int64)
    for ki, i in enumerate(row_ids):
        v = v_rows[i] / v_norms[i]
        for d in range(N_NULL):
            u = rng_state.standard_normal(W_U.shape[1])
            u = u / np.linalg.norm(u)
            vf = v - 2.0 * float(v @ u) * u
            vf = vf / np.linalg.norm(vf)
            null_flips[ki, d] = int(run_bsub(i, vf))
    null_total = int(null_flips.sum())
    null_per_row_max = int(null_flips.sum(axis=1).max())
    print('P2785 NULL total=%d/%d per_row_max=%d'
          % (null_total, len(row_ids) * N_NULL, null_per_row_max),
          flush=True)

    # N1 gate
    d_arms = {'D1': f_D1, 'D2': f_D2, 'D3': f_D3}
    best = max(d_arms, key=d_arms.get)
    n1_pass = bool(d_arms[best] >= 4 and
                   d_arms[best] > np.percentile(null_flips.sum(axis=0),
                                                95))
    # family decomposition of best arm
    best_rows = res_rows[best]
    fam_dec = {}
    for k, i in enumerate(row_ids):
        f = str(fam_arr[i])
        fam_dec.setdefault(f, [0, 0])
        fam_dec[f][1] += 1
        fam_dec[f][0] += int(best_rows[str(i)])
    overlap_O = len(set(i for i in row_ids if res_rows['O'][str(i)]) &
                    set(i for i in row_ids if best_rows[str(i)]))

    verdict = {
        'flip_deoracled': bool(n1_pass),
        'GO_pass': True,
        'oracle_flips': f_O,
        'D1_flips': f_D1, 'D2_flips': f_D2, 'D3_flips': f_D3,
        'best_arm': best,
        'null_total': null_total,
        'null_per_row_max': null_per_row_max,
        'best_fam_decomp': fam_dec,
        'best_overlap_with_oracle': overlap_O,
    }
    result = {'phase': 2785, 'prereg': PREREG, 'verdict': verdict,
              'flip_rows': res_rows,
              'row_ids': [int(i) for i in row_ids],
              'rivals': [int(r) for r in rivals],
              'null_flips': null_flips.tolist()}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'deoracle_stats.npz',
           row_ids=np.array(row_ids, dtype=np.int64),
           pull_pos=pull[pos_k],
           null_flips=null_flips)
    print('P2785 VERDICT %s' % json.dumps(verdict), flush=True)


if __name__ == '__main__':
    main()
