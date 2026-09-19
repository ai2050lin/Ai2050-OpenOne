"""Phase 2779 (Beta): neg margin radius -- top-k neuron ablation scan.

Phase 2777 established neg as deep-unknown beyond readout with a shared
sparse MLP readout carrier ((35,2708) top-1 gain in 16/16 rows) that is NOT
causally sufficient at k=4.  This phase quantifies the margin radius: how
many simultaneous neuron ablations are needed to flip a neg row?

Per row: rank ALL positive-gain down_proj neurons over M={20..35} by
g_j = (W_out^T u_ro)_j * a_j (2777 attribution protocol).  For k in
{4, 8, 16, 32, 64}: ablate the row's top-k neurons simultaneously.

Preregistered (frozen before any intervention forward):
  M-R1: margin_radius = min k with flips >= 4/16; if none up to k=64,
        margin_radius > 64 (censored) -> 'distributed confirmed'.
  M-R2: descriptive: mean |Delta(target-rival logit)| per k (must grow
        with k; liveness check asserted > 0.1 at k=64).
  M-R3: shared-carrier arm: the 6 most frequent 2777 top-4 neurons
        {(35,2708),(31,715),(26,6940),(29,7457),(34,5319),(35,716)}
        ablated as a FIXED set on all 16 rows (descriptive).
  M-R4 (archival gap recovery): Phase 2775 saved no per-row C1a data, so
        the family decomposition of its 12/24 flip-up repairs is unknown.
        Re-run flip-up bsub on ALL 24 pull>0 rows with per-family
        recording.  If neg rows flip >= 4/8, 'neg deep-unknown' is
        OVERTURNED for the sign-flip readout path; if 0-3/8, consistent.
        Total flips expected ~12 (2775 archival).
  verdict: neg_margin_quantified (M-R1 recorded with the scan curve) +
        M-R4 decomposition adjudicates the neg-flip-path question.
Gates: G1 native verify on 16 neg rows; G2 PC (bsub machinery) reuse of
  2777 result accepted -- here liveness is asserted via M-R2 at k=64.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

BASE = cc.BASE
OUT = BASE / 'phase2779' / 'qwen4_neg_margin_radius'
P2761 = BASE / 'phase2761' / 'qwen4_kc_fault'
P2763 = BASE / 'phase2763' / 'qwen4_debias_repair'
P2774 = BASE / 'phase2774' / 'qwen4_pull_validation'
ALPHA_BSUB = 0.3

M_NEG = list(range(20, 36))
K_LIST = [4, 8, 16, 32, 64]
SHARED = [(35, 2708), (31, 715), (26, 6940), (29, 7457), (34, 5319),
          (35, 716)]

PREREG = {
    'M-R1': 'margin_radius = min k in {4,8,16,32,64} with flips >= 4/16; '
            'none -> >64 censored (distributed confirmed)',
    'M-R2': 'mean |Delta(target-rival logit)| grows with k; >0.1 at k=64 '
            '(liveness)',
    'M-R3': 'shared 6-neuron fixed set ablation, descriptive',
    'verdict': 'neg_margin_quantified',
}


def main():
    import torch
    OUT.mkdir(parents=True, exist_ok=True)

    z61 = np.load(P2761 / 'fault_scores.npz', allow_pickle=False)
    wrong_idx = np.array(sorted(int(i) for i in z61['wrong_idx']))
    zbeh = np.load(P2763 / 'behaviour_scores.npz', allow_pickle=False)
    arg_native = zbeh['native__base_arg']

    import phase2747_rdc_material as mat2747
    material, data = mat2747.freeze()
    rows = [r for r in data['diagnostic']
            if r['kind'] == 'controlled_relation']
    tgt_ids = np.array([r['target'] for r in rows], dtype=np.int64)
    fam_arr = np.array([r['family'] for r in rows], dtype=np.str_)
    id_list = [r['prompt_ids'] for r in rows]

    neg_mask = np.array([fam_arr[int(i)] == 'negation_scope'
                         for i in wrong_idx])
    neg_wrong = [int(i) for i in wrong_idx[neg_mask]]
    assert len(neg_wrong) == 16

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'neg_rows': neg_wrong}
    fc.save(OUT / 'execution.json', execution)

    from phase2662_symmetric_mapping_contract import load_native
    model, tok = load_native('qwen4')
    model.eval()
    device = next(model.parameters()).device
    assert model.config.num_hidden_layers == 36
    W_U = model.lm_head.weight.detach().float().cpu().numpy()
    W_OUT = {l: model.model.layers[l].mlp.down_proj.weight.detach().
             float().cpu().numpy() for l in M_NEG}

    state = {'zero_neurons': frozenset()}
    cap = {'on': False, 'dpin': {}}
    handles = []

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

    def make_dpin_pre(l):
        def pre(module, args):
            if cap['on']:
                cap['dpin'][l] = args[0].detach()
            return None
        return pre

    for l in M_NEG:
        handles.append(model.model.layers[l].mlp.down_proj.
                       register_forward_pre_hook(make_neuron_pre(l)))
        handles.append(model.model.layers[l].mlp.down_proj.
                       register_forward_pre_hook(make_dpin_pre(l)))

    def forward(i):
        with torch.inference_mode():
            return model(torch.tensor([id_list[i]], device=device))

    def logits_final(i):
        return forward(i).logits[0, -1].float()

    # ---------------- G1: native verify + ranking ----------------
    rank = {}
    for i in neg_wrong:
        assert int(logits_final(i).argmax()) == int(arg_native[i]), i
    print('P2779 G1_OK native verified 16/16', flush=True)

    for i in neg_wrong:
        zi = logits_final(i)
        tmp = zi.clone()
        tmp[tgt_ids[i]] = -1e30
        rival = int(tmp.argmax())
        base_gap = float(zi[tgt_ids[i]] - zi[rival])
        u = W_U[tgt_ids[i]] - W_U[rival]
        u = (u / np.linalg.norm(u)).astype(np.float32)
        cap['on'] = True
        cap['dpin'].clear()
        try:
            logits_final(i)
        finally:
            cap['on'] = False
        gains = []
        for l in M_NEG:
            a = cap['dpin'][l][0, -1].float().cpu().numpy()
            wu = W_OUT[l].T @ u
            g = wu * a
            for j in np.where(g > 0)[0]:
                gains.append((float(g[j]), l, int(j)))
        gains.sort(reverse=True)
        rank[i] = {'rival': rival, 'base_gap': base_gap,
                   'order': [(l, j) for (_, l, j) in gains]}
    print('P2779 RANK_DONE', flush=True)

    # ---------------- top-k scan ----------------
    scan = {}
    for k in K_LIST:
        flips = 0
        deltas = []
        for i in neg_wrong:
            zi = logits_final(i)
            gap0 = float(zi[tgt_ids[i]] - zi[rank[i]['rival']])
            top = rank[i]['order'][:k]
            state['zero_neurons'] = frozenset(top)
            try:
                z1 = logits_final(i)
            finally:
                state['zero_neurons'] = frozenset()
            gap1 = float(z1[tgt_ids[i]] - z1[rank[i]['rival']])
            deltas.append(gap1 - gap0)
            flips += int(z1.argmax() == tgt_ids[i])
        scan[k] = {'flips': int(flips),
                   'mean_abs_delta': float(np.mean(np.abs(deltas)))}
        print('P2779 K=%d flips=%d/16 |d|=%.3f'
              % (k, flips, scan[k]['mean_abs_delta']), flush=True)

    assert scan[64]['mean_abs_delta'] > 0.1, 'M-R2 liveness failed'

    # ---------------- shared-carrier fixed set ----------------
    sh_flips = 0
    for i in neg_wrong:
        state['zero_neurons'] = frozenset(SHARED)
        try:
            z1 = logits_final(i)
        finally:
            state['zero_neurons'] = frozenset()
        sh_flips += int(z1.argmax() == tgt_ids[i])

    radius = None
    for k in K_LIST:
        if scan[k]['flips'] >= 4:
            radius = k
            break
    # ---------------- M-R4: C1a decomposition (archival gap) ----------
    zp = np.load(P2774 / 'pull_stats.npz', allow_pickle=False)
    pull_all = zp['pull']
    rival_all = zp['rival_ids']
    assert (zp['wrong_idx'] == wrong_idx).all()
    zbd = np.load(P2763 / 'bias_dirs.npz', allow_pickle=False)
    v_rows = zbd['v_rows']
    v_norms = zbd['v_norms']
    W_U_cpu = W_U

    bsub_state = {'vec': None}

    def make_bsub_hook():
        def hook(module, args, output):
            if bsub_state['vec'] is None:
                return None
            out = output[0] if isinstance(output, tuple) else output
            h = out[0, -1]
            out[0, -1] = h - ALPHA_BSUB * h.norm() * bsub_state['vec']
            return None
        return hook

    h35 = model.model.layers[35].register_forward_hook(make_bsub_hook())
    m_r4 = []
    for k, i0 in enumerate(wrong_idx):
        if pull_all[k] <= 0:
            continue
        i = int(i0)
        riv = int(rival_all[k])
        u = W_U_cpu[tgt_ids[i]] - W_U_cpu[riv]
        u = u / np.linalg.norm(u)
        v = v_rows[i] / v_norms[i]
        v_flip = v - 2.0 * float(v @ u) * u
        v_flip = v_flip / np.linalg.norm(v_flip)
        bsub_state['vec'] = torch.tensor(v_flip, device=device)
        try:
            m = int(logits_final(i).argmax())
        finally:
            bsub_state['vec'] = None
        m_r4.append({'row': i, 'fam': str(fam_arr[i]),
                     'flipped': bool(m == tgt_ids[i])})
    h35.remove()
    m_r4_neg = [r for r in m_r4 if r['fam'] == 'negation_scope']
    m_r4_neg_flips = int(sum(r['flipped'] for r in m_r4_neg))
    m_r4_total_flips = int(sum(r['flipped'] for r in m_r4))
    print('P2779 M-R4 total=%d/24 neg=%d/%d'
          % (m_r4_total_flips, m_r4_neg_flips, len(m_r4_neg)), flush=True)

    verdict = {
        'margin_radius': radius if radius is not None else '>64 (censored)',
        'distributed_confirmed': radius is None,
        'scan': scan,
        'shared_arm_flips': int(sh_flips),
        'm_r2_pass': bool(scan[64]['mean_abs_delta'] > 0.1),
        'm_r4': {'total_flips': m_r4_total_flips, 'n': len(m_r4),
                 'neg_flips': m_r4_neg_flips, 'neg_n': len(m_r4_neg),
                 'neg_flip_path_overturned': bool(m_r4_neg_flips >= 4),
                 'per_row': m_r4},
    }
    result = {'phase': 2779, 'prereg': PREREG, 'verdict': verdict,
              'rank_first8': {str(i): [[l, j] for (l, j) in
                                       rank[i]['order'][:8]]
                              for i in neg_wrong},
              'base_gaps': {str(i): round(rank[i]['base_gap'], 3)
                            for i in neg_wrong}}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'margin_stats.npz',
           neg_wrong=np.array(neg_wrong),
           k_flips=np.array([scan[k]['flips'] for k in K_LIST]),
           shared_flips=np.array([sh_flips]))
    for h in handles:
        h.remove()
    print('P2779 VERDICT %s' % json.dumps(verdict), flush=True)


if __name__ == '__main__':
    main()
