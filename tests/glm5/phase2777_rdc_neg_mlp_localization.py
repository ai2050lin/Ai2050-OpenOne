"""Phase 2777: neg deep-unknown localization -- early layers + MLP path.

Phase 2775 established neg (16 negation_scope wrong rows) as structurally
deep-unknown at the readout: single-layer bsub over L={20..35} best 1/16,
full-range top-4 rival-gain head ablation 0/16, plus 2773 (deep-layer
attention writeback ablation 1/16) and 2774 (trained states 0/16).  The
remaining hypotheses: (i) rival signal written into the residual stream
BEFORE L20; (ii) rival signal carried by the MLP pathway, not attention
writeback; (iii) both.

Preregistered (frozen before any forward):
  E1 (early-layer bsub sweep): single-layer bsub hook at l in {0..19},
      alpha=0.3, v_row per row, all 16 rows.  early_reachable if any layer
      flips >= 4/16; else early_immune.
  E2 (MLP-path lesion): zero the MLP block output at the final position,
      one module layer at a time over M={20..35} (16x16 forwards), plus one
      union run (all M layers simultaneously).  mlp_relevant if any single
      layer or the union flips >= 4/16.
  E3 (MLP sparse-neuron attribution): per row, rival-gain per down_proj
      input neuron g_j = (W_out^T u_ro)_j * a_j  (a = down_proj input at
      final position, captured across all M layers); ablate the row's top-4
      positive-gain neurons (union over M).  neg_mlp_localized if flips
      >= 4/16; else mlp_sparse_immune.
  Overall: deep_unknown_beyond_readout iff E1 immune AND E2 not relevant
      AND E3 immune.
Gates: native arg verify on all 16 neg rows before any intervention.
  PC1 (bsub machinery positive control): state-based bsub at L35 on 2
      kc bsub-fixed rows (2772 archival) must flip >= 1/2, else the run
      aborts (hook machinery dead -> all-zero verdicts void).
  PC2 (MLP-zero liveness): union MLP-zero on neg rows must move the
      target-rival logit gap (mean |delta| recorded; asserted > 0.1).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

BASE = cc.BASE
OUT = BASE / 'phase2777' / 'qwen4_neg_mlp_localization'
P2761 = BASE / 'phase2761' / 'qwen4_kc_fault'
P2763 = BASE / 'phase2763' / 'qwen4_debias_repair'
P2772 = BASE / 'phase2772' / 'qwen4_combined_repair'

ALPHA = 0.3
EARLY_LAYERS = list(range(0, 20))
M_NEG = list(range(20, 36))
TOP_NEURONS = 4

PREREG = {
    'E1': 'early_reachable if any layer in {0..19} bsub flips >= 4/16, '
          'else early_immune',
    'E2': 'mlp_relevant if any single-layer final-position MLP-output zero '
          '(M={20..35}) or the all-M union flips >= 4/16',
    'E3': 'neg_mlp_localized if per-row top-4 positive-gain down_proj '
          'neuron ablation flips >= 4/16, else mlp_sparse_immune',
    'overall': 'deep_unknown_beyond_readout iff E1 early_immune AND E2 '
               'not relevant AND E3 mlp_sparse_immune',
    'PC1': 'bsub machinery positive control: state-based bsub at L35 on '
           '2 kc bsub-fixed rows flips >= 1/2, else abort',
    'PC2': 'MLP-zero liveness: union MLP-zero moves target-rival logit '
           'gap, mean |delta| > 0.1, else abort',
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

    import phase2747_rdc_material as mat2747
    material, data = mat2747.freeze()
    rows = [r for r in data['diagnostic']
            if r['kind'] == 'controlled_relation']
    assert len(rows) == 320
    tgt_ids = np.array([r['target'] for r in rows], dtype=np.int64)
    fam_arr = np.array([r['family'] for r in rows], dtype=np.str_)
    id_list = [r['prompt_ids'] for r in rows]

    neg_mask = np.array([fam_arr[int(i)] == 'negation_scope'
                         for i in wrong_idx])
    neg_wrong = [int(i) for i in wrong_idx[neg_mask]]
    assert len(neg_wrong) == 16, len(neg_wrong)

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG,
                 'neg_rows': neg_wrong}
    fc.save(OUT / 'execution.json', execution)

    from phase2662_symmetric_mapping_contract import load_native
    model, tok = load_native('qwen4')
    model.eval()
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    assert n_layers == 36
    W_U = model.lm_head.weight.detach().float().cpu().numpy()
    W_OUT = {l: model.model.layers[l].mlp.down_proj.weight.detach().
             float().cpu().numpy() for l in M_NEG}  # (5120, 13824)
    n_neurons = W_OUT[M_NEG[0]].shape[1]

    state = {'bsub': None,        # (layer, alpha, vec_tensor)
             'zero_mlp': frozenset(),
             'zero_neurons': frozenset()}   # {(l, j)}

    handles = []

    def make_bsub_hook(l):
        def hook(module, args, output):
            if state['bsub'] is None or state['bsub'][0] != l:
                return None
            out = output[0] if isinstance(output, tuple) else output
            _, alpha, vec = state['bsub']
            h = out[0, -1]
            out[0, -1] = h - alpha * h.norm() * vec
            return None
        return hook

    def make_mlp_hook(l):
        def hook(module, args, output):
            if l not in state['zero_mlp']:
                return None
            out = output[0] if isinstance(output, tuple) else output
            out[0, -1, :] = 0
            return None
        return hook

    def make_neuron_pre(l):
        def pre(module, args):
            js = [j for (ll, j) in state['zero_neurons'] if ll == l]
            if not js:
                return None
            x = args[0].clone()
            for j in js:
                x[0, -1, j] = 0
            return (x,)
        return pre

    cap = {'on': False, 'dpin': {}}

    def make_dpin_pre(l):
        def pre(module, args):
            if cap['on']:
                cap['dpin'][l] = args[0].detach()
            return None
        return pre

    for l in range(n_layers):
        handles.append(model.model.layers[l].
                       register_forward_hook(make_bsub_hook(l)))
        handles.append(model.model.layers[l].mlp.
                       register_forward_hook(make_mlp_hook(l)))
        handles.append(model.model.layers[l].mlp.down_proj.
                       register_forward_pre_hook(make_neuron_pre(l)))
        handles.append(model.model.layers[l].mlp.down_proj.
                       register_forward_pre_hook(make_dpin_pre(l)))

    def forward(i):
        with torch.inference_mode():
            t = torch.tensor([id_list[i]], device=device)
            return model(t)

    # ---------------- gate: native verify ----------------
    arg_check = {}
    for i in neg_wrong:
        arg_check[i] = int(forward(i).logits[0, -1].float().argmax())
        assert arg_check[i] == int(arg_native[i]), ('native drift', i)
    print('P2777 GATE_OK native verified on 16 neg rows', flush=True)

    # ---------------- PC1/PC2: machinery positive controls ----------
    zrep = np.load(P2772 / 'repair_stats.npz', allow_pickle=False)
    bsub_arr = zrep['bsub'].astype(bool)
    assert (zrep['wrong_idx'] == wrong_idx).all()
    pc_candidates = [int(i) for k, i in enumerate(wrong_idx)
                     if bsub_arr[k]
                     and fam_arr[int(i)] != 'negation_scope'][:2]
    pc1 = {'rows': [], 'flips': 0}
    for i in pc_candidates:
        v = v_rows[i] / v_norms[i]
        state['bsub'] = (35, ALPHA, torch.tensor(v, device=device))
        try:
            m = int(forward(i).logits[0, -1].float().argmax())
        finally:
            state['bsub'] = None
        flipped = bool(m == tgt_ids[i])
        pc1['rows'].append({'row': i, 'flipped': flipped})
        pc1['flips'] += int(flipped)
    assert pc1['flips'] >= 1, ('PC1 bsub machinery dead', pc1)
    print('P2777 PC1_OK flips=%d/2 rows=%s'
          % (pc1['flips'], [r['row'] for r in pc1['rows']]), flush=True)

    pc2 = {'deltas': []}
    for i in neg_wrong:
        with torch.inference_mode():
            z0 = forward(i).logits[0, -1].float()
        base_gap = float(z0[tgt_ids[i]] - z0[arg_check[i]])
        state['zero_mlp'] = frozenset(M_NEG)
        try:
            with torch.inference_mode():
                z1 = forward(i).logits[0, -1].float()
        finally:
            state['zero_mlp'] = frozenset()
        new_gap = float(z1[tgt_ids[i]] - z1[arg_check[i]])
        pc2['deltas'].append(round(new_gap - base_gap, 4))
    pc2['mean_abs_delta'] = float(np.mean(np.abs(pc2['deltas'])))
    assert pc2['mean_abs_delta'] > 0.1, ('PC2 mlp-zero no-op', pc2)
    print('P2777 PC2_OK mean_abs_delta=%.4f' % pc2['mean_abs_delta'],
          flush=True)

    # ---------------- E1: early-layer bsub sweep ----------------
    e1 = {}
    for l in EARLY_LAYERS:
        flips = 0
        for i in neg_wrong:
            v = v_rows[i] / v_norms[i]
            state['bsub'] = (l, ALPHA, torch.tensor(v, device=device))
            try:
                m = int(forward(i).logits[0, -1].float().argmax())
            finally:
                state['bsub'] = None
            flips += int(m == tgt_ids[i])
        e1[l] = flips
    e1_best = max(e1, key=e1.get)
    e1_reachable = bool(e1[e1_best] >= 4)
    print('P2777 E1_DONE best L%d=%d' % (e1_best, e1[e1_best]), flush=True)

    # ---------------- E2: MLP-output zero per layer + union ----------
    e2 = {}
    for l in M_NEG:
        flips = 0
        for i in neg_wrong:
            state['zero_mlp'] = frozenset([l])
            try:
                m = int(forward(i).logits[0, -1].float().argmax())
            finally:
                state['zero_mlp'] = frozenset()
            flips += int(m == tgt_ids[i])
        e2[l] = flips
    e2_union = 0
    for i in neg_wrong:
        state['zero_mlp'] = frozenset(M_NEG)
        try:
            m = int(forward(i).logits[0, -1].float().argmax())
        finally:
            state['zero_mlp'] = frozenset()
        e2_union += int(m == tgt_ids[i])
    e2_best = max(e2, key=e2.get)
    e2_relevant = bool(e2[e2_best] >= 4 or e2_union >= 4)
    print('P2777 E2_DONE best L%d=%d union=%d'
          % (e2_best, e2[e2_best], e2_union), flush=True)

    # ---------------- E3: sparse-neuron attribution ----------------
    e3_rows = {}
    e3_flips = 0
    for i in neg_wrong:
        cap['on'] = True
        cap['dpin'].clear()
        try:
            o = forward(i)
            zi = o.logits[0, -1].float()
        finally:
            cap['on'] = False
        tmp = zi.clone()
        tmp[tgt_ids[i]] = -1e30
        rival = int(tmp.argmax())
        u = W_U[rival] - W_U[tgt_ids[i]]
        u = (u / np.linalg.norm(u)).astype(np.float32)
        gains = []
        for l in M_NEG:
            a = cap['dpin'][l][0, -1].float().cpu().numpy()
            wu = W_OUT[l].T @ u  # (13824,)
            g = wu * a
            for j in np.where(g > 0)[0]:
                gains.append((float(g[j]), l, int(j)))
        gains.sort(reverse=True)
        top = [(l, j) for (_, l, j) in gains[:TOP_NEURONS]]
        e3_rows[i] = {'rival': rival,
                      'top_neurons': [[l, j] for (l, j) in top],
                      'top_gains': [round(g, 4) for (g, _, _) in
                                    gains[:TOP_NEURONS]]}
        state['zero_neurons'] = frozenset(top)
        try:
            m = int(forward(i).logits[0, -1].float().argmax())
        finally:
            state['zero_neurons'] = frozenset()
        e3_rows[i]['flipped'] = bool(m == tgt_ids[i])
        e3_flips += int(m == tgt_ids[i])
    e3_localized = bool(e3_flips >= 4)

    verdict = {
        'E1': {'verdict': 'early_reachable' if e1_reachable else
               'early_immune', 'curve': e1, 'best_layer': int(e1_best),
               'best_flips': int(e1[e1_best])},
        'E2': {'verdict': 'mlp_relevant' if e2_relevant else 'mlp_immune',
               'curve': e2, 'best_layer': int(e2_best),
               'best_flips': int(e2[e2_best]), 'union_flips': int(e2_union)},
        'E3': {'verdict': 'neg_mlp_localized' if e3_localized else
               'mlp_sparse_immune', 'flips': int(e3_flips), 'n': 16},
        'overall': ('deep_unknown_beyond_readout'
                    if (not e1_reachable and not e2_relevant
                        and not e3_localized) else
                    'neg_localized'),
    }

    result = {'phase': 2777, 'prereg': PREREG, 'verdict': verdict,
              'pc1': pc1, 'pc2': pc2,
              'e3_rows': {str(i): e3_rows[i] for i in neg_wrong}}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'neg_loc_stats.npz',
           neg_wrong=np.array(neg_wrong),
           e1_curve=np.array([e1[l] for l in EARLY_LAYERS]),
           e2_curve=np.array([e2[l] for l in M_NEG]),
           e3_flips=np.array([int(e3_rows[i]['flipped'])
                              for i in neg_wrong]))
    for h in handles:
        h.remove()
    print('P2777 VERDICT %s' % json.dumps(verdict), flush=True)


if __name__ == '__main__':
    main()
