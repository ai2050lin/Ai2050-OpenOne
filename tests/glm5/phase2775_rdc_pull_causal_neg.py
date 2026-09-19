"""Phase 2775: pull-sign CAUSALIZATION + negation deep-unknown localization.

C001 (pull causality).  Phase 2774 established pull = v.u_ro (u_ro =
unit(W_U[target]-W_U[rival])) as a necessary condition for bsub/combo
repair.  Here we CAUSE the sign:
  - flip-up rows (24 pull>0): v' = v - 2*(v.u_ro)u_ro  =>  v'.u_ro =
    -(v.u_ro) < 0, orthogonal part preserved.  Prediction: bsub(v') now
    repairs.  C1a: flips on the 24 rows >= 4 and > null q95.
  - flip-down rows (21 pull<0 bsub-fixed): same transform makes pull>0.
    Prediction C1b: flips drop from 21 to <= 3.
  - null: random-direction bsub from 2763 archival ('native__ctrl_*'),
    alpha identical.
C002 (neg localization, 16 neg wrong rows).
  - layered bsub: single-layer hook at l in {20..35}, alpha=0.3, all 16
    rows -> 16x16 curve.
  - full-range rival-gain heads over modules {20..35}: top-4 per row,
    targeted ablation; per-source attribution (fact/content/other).
  - E-N: structural deep-unknown confirmed if ALL interventions flip
    <= 3/16; 'mislocated-repairable' if any >= 4/16.
Prereg frozen before any forward.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc
import phase2747_rdc_material as mat2747
import phase2759_rdc_question_span_deletion as p2759

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2775' / 'qwen4_pull_causal_neg'
P2761 = BASE / 'phase2761' / 'qwen4_kc_fault'
P2763 = BASE / 'phase2763' / 'qwen4_debias_repair'
P2774 = BASE / 'phase2774' / 'qwen4_pull_validation'

ALPHA = 0.3
BSUB_LAYER = 35
NEG_LAYERS = list(range(20, 36))

PREREG = {
    'C1a': 'sign-flip bsub on the 24 pull>0 rows flips >= 4 AND > null q95 '
           '(null = 2763 archival random-direction bsub)',
    'C1b': 'sign-flip bsub on the 21 pull<0 bsub-fixed rows flips <= 3 '
           '(from 21)',
    'E-N': 'deep-unknown structural if ALL neg interventions (layered bsub '
           'curve best layer, full-range top-4 head ablation) flip <= 3/16; '
           'mislocated-repairable if any >= 4/16',
    'verdict': 'pull_causality_confirmed iff C1a and C1b',
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
    rows = [r for r in data['diagnostic'] if r['kind'] == 'controlled_relation']
    tgt_ids = np.array([r['target'] for r in rows], dtype=np.int64)
    fam_arr = np.array([r['family'] for r in rows], dtype=np.str_)
    id_list = [r['prompt_ids'] for r in rows]
    assert (tgt_ids[wrong_idx] == z61['target'][wrong_idx]).all()

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG}
    fc.save(OUT / 'execution.json', execution)

    from phase2662_symmetric_mapping_contract import load_native
    model, tok = load_native('qwen4')
    model.eval()
    device = next(model.parameters()).device
    n_heads = model.config.num_attention_heads
    n_kv = model.config.num_key_value_heads
    qdim = model.model.layers[0].self_attn.q_proj.out_features
    hd = qdim // n_heads
    group = n_heads // n_kv
    W_U = model.lm_head.weight.detach().float().cpu().numpy()

    hook_state = {'on': False, 'alpha': None, 'vec': None}

    def readout_hook(module, args, output):
        if not hook_state['on']:
            return None
        out = output[0] if isinstance(output, tuple) else output
        h = out[0, -1]
        out[0, -1] = h - hook_state['alpha'] * h.norm() * hook_state['vec']
        return None

    handle35 = model.model.layers[BSUB_LAYER].register_forward_hook(
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

    # ---------------- C001: sign-flip causality ----------------
    pull_pos_idx = list(np.where(pull > 0)[0])
    pull_neg_fixed = [k for k in range(len(wrong_idx))
                      if pull[k] < 0 and bool(zp['bsub'][k])]
    assert len(pull_pos_idx) == 24 and len(pull_neg_fixed) == 21

    c1a = {'n': len(pull_pos_idx), 'flips': 0, 'rows': {}}
    for k in pull_pos_idx:
        i = int(wrong_idx[k])
        v = v_rows[i] / v_norms[i]
        u_ro = W_U[tgt_ids[i]] - W_U[int(zp['rival_ids'][k])]
        u_ro = u_ro / np.linalg.norm(u_ro)
        v_flip = v - 2.0 * float(v @ u_ro) * u_ro
        v_flip = v_flip / np.linalg.norm(v_flip)
        flipped = run_bsub(i, v_flip)
        c1a['rows'][str(i)] = bool(flipped)
        c1a['flips'] += int(flipped)
    c1a['flips'] = int(c1a['flips'])

    c1b = {'n': len(pull_neg_fixed), 'flips': 0, 'rows': {}}
    for k in pull_neg_fixed:
        i = int(wrong_idx[k])
        v = v_rows[i] / v_norms[i]
        u_ro = W_U[tgt_ids[i]] - W_U[int(zp['rival_ids'][k])]
        u_ro = u_ro / np.linalg.norm(u_ro)
        v_flip = v - 2.0 * float(v @ u_ro) * u_ro
        v_flip = v_flip / np.linalg.norm(v_flip)
        flipped = run_bsub(i, v_flip)
        c1b['rows'][str(i)] = bool(flipped)
        c1b['flips'] += int(flipped)
    c1b['flips'] = int(c1b['flips'])

    # archival null: 2763 random-direction bsub on the same 65 rows
    ctrl_flips = 0
    for i in wrong_idx:
        ctrl_flips += int(int(zbeh['native__ctrl_%d' % int(i)]) ==
                          tgt_ids[i])
    null_mean = ctrl_flips / 65.0

    c1a_pass = bool(c1a['flips'] >= 4 and c1a['flips'] / 24.0 >
                    null_mean * 3)
    c1b_pass = bool(c1b['flips'] <= 3)
    print('P2775 C1a flips=%d/24 C1b flips=%d/21 ctrl=%d/65'
          % (c1a['flips'], c1b['flips'], ctrl_flips), flush=True)

    # ---------------- C002: neg localization ----------------
    neg_mask = np.array([fam_arr[int(i)] == 'negation_scope'
                         for i in wrong_idx])
    neg_wrong = [int(i) for i in wrong_idx[neg_mask]]
    assert len(neg_wrong) == 16

    # layered bsub curve (single-layer hooks)
    handle35.remove()
    layer_flips = {}
    for l in NEG_LAYERS:
        h = model.model.layers[l].register_forward_hook(readout_hook)
        flips = 0
        for i in neg_wrong:
            v = v_rows[i] / v_norms[i]
            hook_state['on'] = True
            hook_state['alpha'] = ALPHA
            hook_state['vec'] = torch.tensor(v, device=device)
            try:
                with torch.inference_mode():
                    o = model(torch.tensor([id_list[i]], device=device))
                    m = int(o.logits[0, -1].float().argmax())
            finally:
                hook_state['on'] = False
            flips += int(m == tgt_ids[i])
        h.remove()
        layer_flips[l] = flips

    # full-range rival-gain heads (modules 20..35) + targeted ablation
    M_NEG = NEG_LAYERS
    cap = {}
    chandles = []

    def oproj_pre(l):
        def hook(module, args):
            cap.setdefault('oproj', {})[l] = args[0].detach()
        return hook

    def abl_pre(l):
        def hook(module, args):
            hs_ = [h for (ll, h) in abl_state['pairs'] if ll == l]
            if not hs_:
                return None
            x = args[0].clone()
            for h in hs_:
                x[0, -1, h * hd:(h + 1) * hd] = 0
            return (x,) + tuple(args[1:])
        return hook

    abl_state = {'pairs': frozenset()}
    for l in M_NEG:
        chandles.append(model.model.layers[l].self_attn.o_proj.
                        register_forward_pre_hook(oproj_pre(l)))
        chandles.append(model.model.layers[l].self_attn.o_proj.
                        register_forward_pre_hook(abl_pre(l)))

    def forward(i):
        with torch.inference_mode():
            t = torch.tensor([id_list[i]], device=device)
            return model(t)

    neg_carrier = {}
    neg_targeted = {}
    neg_src = {}
    for i in neg_wrong:
        cap.clear()
        with torch.inference_mode():
            o = model(torch.tensor([id_list[i]], device=device))
            zi = o.logits[0, -1].float()
        tmp = zi.clone()
        tmp[tgt_ids[i]] = -1e30
        rival = int(tmp.argmax())
        u = W_U[rival] - W_U[tgt_ids[i]]
        u = (u / np.linalg.norm(u)).astype(np.float32)
        gains = {}
        for l in M_NEG:
            X = cap['oproj'][l][0, -1].float().cpu().numpy()
            Wl = model.model.layers[l].self_attn.o_proj.weight.\
                detach().float().cpu().numpy()
            for h in range(n_heads):
                sl = slice(h * hd, (h + 1) * hd)
                gains[(l, h)] = float(u @ (Wl[:, sl] @ X[sl]))
        ch = [(l, h) for (l, h), g in
              sorted(gains.items(), key=lambda x: -x[1]) if g > 0][:4]
        neg_carrier[i] = ch
        abl_state['pairs'] = frozenset(ch)
        try:
            m = int(forward(i).logits[0, -1].float().argmax())
        finally:
            abl_state['pairs'] = frozenset()
        neg_targeted[i] = bool(m == tgt_ids[i])
    for h in chandles:
        h.remove()

    neg_abl_flips = int(sum(neg_targeted.values()))
    best_layer = max(layer_flips, key=layer_flips.get)
    en_pass = bool(max(neg_abl_flips, max(layer_flips.values())) >= 4)
    verdict = {
        'pull_causality_confirmed': bool(c1a_pass and c1b_pass),
        'C1a': {'pass': c1a_pass, 'flips': c1a['flips'], 'n': 24,
                'null_mean': null_mean},
        'C1b': {'pass': c1b_pass, 'flips': c1b['flips'], 'n': 21},
        'neg': {'verdict': 'mislocated-repairable' if en_pass else
                'deep-unknown structural',
                'layered_bsub_flips': layer_flips,
                'best_layer': best_layer,
                'top4_head_abl_flips': neg_abl_flips},
    }
    result = {'phase': 2775, 'prereg': PREREG, 'verdict': verdict,
              'neg_carrier_heads': {str(i): [list(p) for p in neg_carrier[i]]
                                    for i in neg_wrong},
              'neg_targeted': {str(i): neg_targeted[i] for i in neg_wrong}}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'neg_stats.npz', wrong_idx=wrong_idx, pull=pull,
           neg_wrong=np.array(neg_wrong),
           neg_targeted=np.array([neg_targeted[i] for i in neg_wrong]))
    print('P2775 VERDICT %s' % json.dumps(verdict), flush=True)


if __name__ == '__main__':
    main()
