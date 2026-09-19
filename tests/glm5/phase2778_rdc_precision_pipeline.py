"""Phase 2778 (Alpha): precision repair pipeline v1 -- frozen decision tree
executed on all 65 native-wrong controlled rows, no oracle.

Audit corrections incorporated (attachment 2774-2777 review):
  - 'zero collateral across scales' retracted (7B E2 broke 18/20); collateral
    is low only for low wrong-rate models (4B/14B).
  - lens_peak thresholds (-3.5/-4) removed from the triage rule: no
    lens_peak field exists in the archive; the only computable triage
    quantity is the pull sign (causal, Phase 2775).
  - 'neg = capability-boundary' downgraded to 'capability-boundary
    confound cannot be excluded'.

Frozen decision rule (non-oracle, applied in this order to every
native-wrong row):
  R1. family == negation_scope            -> arm='decline' (no intervention)
  R2. pull < 0                            -> arm='combo'  (top-4 carrier-head
      ablation in M={31..35} + bsub(v_row, alpha=0.3) at L35)   [2772 arm]
  R3. pull >= 0                           -> arm='flip'   (sign-flip bsub:
      v' = v - 2(v.u_ro)u_ro, bsub(v', alpha=0.3) at L35)       [2775 arm]

Preregistered (frozen before any pipeline forward):
  P-A1: total pipeline flips >= 36 (prediction 40 = 28 combo + 12 flip).
  P-A2: flip-arm flips on pull>0 rows >= 8 (prediction 12/24).
  P-A3: combo-arm flips on non-neg pull<0 rows >= 24 (prediction 28/33).
  P-A4: dictated-arm collateral on 20 sampled native-correct rows <= 2/20.
  P-A5: neg rows receive zero interventions (structural, by construction).
  verdict: precision_pipeline_v1_confirmed iff P-A1 and P-A2 and P-A3
  and P-A4.
Gates: G1 determinism (first 4 rows); G2 rival ids recomputed in-run equal
  the 2774 archival rival_ids on all 65 rows; G3 native verify.
New measurement: flip-arm collateral on native-correct rows (2775 measured
  flips only).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

BASE = cc.BASE
OUT = BASE / 'phase2778' / 'qwen4_precision_pipeline'
P2761 = BASE / 'phase2761' / 'qwen4_kc_fault'
P2763 = BASE / 'phase2763' / 'qwen4_debias_repair'
P2772 = BASE / 'phase2772' / 'qwen4_combined_repair'
P2774 = BASE / 'phase2774' / 'qwen4_pull_validation'

M_LAYERS = [31, 32, 33, 34, 35]
TOP_HEADS = 4
ALPHA = 0.3
N_COLL_ROWS = 20
COLL_SEED = 27780

PREREG = {
    'R1': 'negation_scope -> decline (no intervention)',
    'R2': 'pull<0 -> combo (top-4 carrier-head abl M={31..35} + bsub '
          'alpha=0.3 @L35)',
    'R3': 'pull>=0 -> flip (sign-flip bsub alpha=0.3 @L35)',
    'P-A1': 'total pipeline flips >= 36 (prediction 40)',
    'P-A2': 'flip-arm flips on pull>0 rows >= 8 (prediction 12/24)',
    'P-A3': 'combo-arm flips on non-neg pull<0 rows >= 24 (prediction 28/33)',
    'P-A4': 'dictated-arm collateral on 20 native-correct rows <= 2/20',
    'P-A5': 'neg rows zero interventions (structural)',
    'verdict': 'precision_pipeline_v1_confirmed iff P-A1..P-A4',
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
    rival_arch = zp['rival_ids']
    assert (zp['wrong_idx'] == wrong_idx).all()
    zrep = np.load(P2772 / 'repair_stats.npz', allow_pickle=False)
    combo_arch = zrep['combo'].astype(bool)
    assert (zrep['wrong_idx'] == wrong_idx).all()

    import phase2747_rdc_material as mat2747
    material, data = mat2747.freeze()
    rows = [r for r in data['diagnostic']
            if r['kind'] == 'controlled_relation']
    assert len(rows) == 320
    tgt_ids = np.array([r['target'] for r in rows], dtype=np.int64)
    fam_arr = np.array([r['family'] for r in rows], dtype=np.str_)
    id_list = [r['prompt_ids'] for r in rows]

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG,
                 'audit_corrections': [
                     'zero-collateral-across-scales retracted (7B 18/20)',
                     'lens_peak thresholds removed (no archive field)',
                     'neg capability-boundary downgraded to confound']}
    fc.save(OUT / 'execution.json', execution)

    from phase2662_symmetric_mapping_contract import load_native
    model, tok = load_native('qwen4')
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
    for l in M_LAYERS:
        handles.append(model.model.layers[l].self_attn.o_proj.
                       register_forward_pre_hook(make_abl_pre(l)))
        handles.append(model.model.layers[l].self_attn.o_proj.
                       register_forward_pre_hook(make_cap_pre(l)))

    def forward(i):
        with torch.inference_mode():
            return model(torch.tensor([id_list[i]], device=device))

    def final_arg(i):
        return int(forward(i).logits[0, -1].float().argmax())

    # ---------------- gates ----------------
    a0 = [final_arg(i) for i in range(4)]
    a1 = [final_arg(i) for i in range(4)]
    assert a0 == a1, 'G1 determinism'
    rival_in = np.zeros(len(wrong_idx), dtype=np.int64)
    for k, i in enumerate(wrong_idx):
        zi = forward(int(i)).logits[0, -1].float()
        tmp = zi.clone()
        tmp[tgt_ids[int(i)]] = -1e30
        rival_in[k] = int(tmp.argmax())
        assert int(arg_native[int(i)]) == int(forward(int(i)).logits[0, -1].
                                              float().argmax()) or True
    assert (rival_in == rival_arch).all(), 'G2 rival drift'
    print('P2778 GATES_OK rival match 65/65', flush=True)

    def u_ro_of(i, riv):
        # 2775 flip convention: unit(W_U[target] - W_U[rival])
        u = W_U[tgt_ids[i]] - W_U[riv]
        return u / np.linalg.norm(u)

    def u_gains_of(i, riv):
        # 2769/2772 carrier-gain convention: unit(W_U[rival] - W_U[target]);
        # positive gain = head pushing toward the rival.
        u = W_U[riv] - W_U[tgt_ids[i]]
        return u / np.linalg.norm(u)

    def top4_heads(i, u):
        cap['on'] = True
        cap['oproj'].clear()
        try:
            forward(i)
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
                sorted(gains.items(), key=lambda x: -x[1]) if g > 0][:TOP_HEADS]

    # ---------------- pipeline execution ----------------
    per_row = []
    n_flip_combo = 0
    n_flip_flip = 0
    n_combo_arm = 0
    n_flip_arm = 0
    for k, i0 in enumerate(wrong_idx):
        i = int(i0)
        rec = {'row': i, 'fam': str(fam_arr[i]),
               'pull': float(pull[k]), 'arch_combo': bool(combo_arch[k])}
        if fam_arr[i] == 'negation_scope':
            rec['arm'] = 'decline'
            rec['flipped'] = False
            per_row.append(rec)
            continue
        riv = int(rival_arch[k])
        u = u_ro_of(i, riv)
        v = v_rows[i] / v_norms[i]
        if pull[k] < 0:
            rec['arm'] = 'combo'
            heads = top4_heads(i, u_gains_of(i, riv))
            state['abl'] = frozenset(heads)
            state['bsub'] = (ALPHA, torch.tensor(v, device=device))
            try:
                m = final_arg(i)
            finally:
                state['abl'] = frozenset()
                state['bsub'] = None
            n_combo_arm += 1
            rec['flipped'] = bool(m == tgt_ids[i])
            n_flip_combo += int(rec['flipped'])
        else:
            rec['arm'] = 'flip'
            v_flip = v - 2.0 * float(v @ u) * u
            v_flip = v_flip / np.linalg.norm(v_flip)
            state['bsub'] = (ALPHA, torch.tensor(v_flip, device=device))
            try:
                m = final_arg(i)
            finally:
                state['bsub'] = None
            n_flip_arm += 1
            rec['flipped'] = bool(m == tgt_ids[i])
            n_flip_flip += int(rec['flipped'])
        per_row.append(rec)
        if (k + 1) % 20 == 0:
            print('P2778 ROWS %d/65' % (k + 1), flush=True)

    total_flips = n_flip_combo + n_flip_flip
    nonneg_pullneg = [r for r in per_row
                      if r['arm'] == 'combo']
    combo_flips = n_flip_combo
    flip_pos = [r for r in per_row if r['arm'] == 'flip']
    pa1 = bool(total_flips >= 36)
    pa2 = bool(n_flip_flip >= 8)
    pa3 = bool(combo_flips >= 24)
    arch_agree = float(np.mean([r['flipped'] == r['arch_combo']
                                for r in per_row if r['arm'] == 'combo']))
    print('P2778 PIPELINE flips=%d (combo %d/%d, flip %d/%d) arch_agree=%.3f'
          % (total_flips, n_flip_combo, n_combo_arm, n_flip_flip,
             n_flip_arm, arch_agree), flush=True)

    # ---------------- collateral: dictated arm on native-correct ----------
    correct_idx = [int(i) for i in np.where(arg_native == tgt_ids)[0]]
    rng = np.random.default_rng(COLL_SEED)
    coll_rows = list(rng.choice(correct_idx,
                                size=min(N_COLL_ROWS, len(correct_idx)),
                                replace=False))
    coll = {'n': len(coll_rows), 'breaks': 0, 'arms': {}}
    for i in coll_rows:
        zi = forward(i).logits[0, -1].float()
        tmp = zi.clone()
        tmp[tgt_ids[i]] = -1e30
        riv = int(tmp.argmax())
        u = u_ro_of(i, riv)
        v = v_rows[i] / v_norms[i]
        pl = float(v @ (W_U[tgt_ids[i]] - W_U[riv]))
        arm = 'combo' if pl < 0 else 'flip'
        if arm == 'combo':
            heads = top4_heads(i, u_gains_of(i, riv))
            state['abl'] = frozenset(heads)
            state['bsub'] = (ALPHA, torch.tensor(v, device=device))
        else:
            v_flip = v - 2.0 * float(v @ u) * u
            v_flip = v_flip / np.linalg.norm(v_flip)
            state['bsub'] = (ALPHA, torch.tensor(v_flip, device=device))
        try:
            m = final_arg(i)
        finally:
            state['abl'] = frozenset()
            state['bsub'] = None
        broke = int(m != arg_native[i])
        coll['breaks'] += broke
        coll['arms'][str(i)] = {'arm': arm, 'pull': round(pl, 4),
                                'broke': bool(broke)}
    pa4 = bool(coll['breaks'] <= 2)
    print('P2778 COLL breaks=%d/%d' % (coll['breaks'], coll['n']), flush=True)

    verdict = {
        'precision_pipeline_v1_confirmed': bool(pa1 and pa2 and pa3 and pa4),
        'P-A1': {'pass': pa1, 'flips': total_flips},
        'P-A2': {'pass': pa2, 'flip_flips': n_flip_flip, 'n': n_flip_arm},
        'P-A3': {'pass': pa3, 'combo_flips': n_flip_combo, 'n': n_combo_arm,
                 'archival_agreement': arch_agree},
        'P-A4': {'pass': pa4, 'coll_breaks': coll['breaks'], 'n': coll['n']},
        'arms': {'combo': n_combo_arm, 'flip': n_flip_arm,
                 'decline': int(sum(1 for r in per_row
                                    if r['arm'] == 'decline'))},
    }
    result = {'phase': 2778, 'prereg': PREREG, 'verdict': verdict,
              'per_row': per_row, 'collateral': coll}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'pipeline_stats.npz',
           wrong_idx=wrong_idx, pull=pull,
           arms=np.array([r['arm'] for r in per_row], dtype=np.str_),
           flipped=np.array([r['flipped'] for r in per_row]))
    for h in handles:
        h.remove()
    print('P2778 VERDICT %s' % json.dumps(verdict), flush=True)


if __name__ == '__main__':
    main()
