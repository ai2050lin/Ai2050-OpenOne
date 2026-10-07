# -*- coding: utf-8 -*-
"""Phase 2989: MLP neuron-level registry (plan v4 P1, OMEGA-G).

First neuron-granularity audit of the chain (only prior card:
M2906, weight-space only). Registry = MLP intermediate neurons
(inter=9728) x readout axes {lang, cls} (unit dirs from 2979),
layers L6-12 load-bearing band + L17/L34 contrast (9 layers).

Three pre-registered lenses (dual-caliber discipline):
  T1 snapshot: per-neuron class contrast of the axis-projected
      neuron contribution proj_i(w)=act_i*<W_down[:,i],u>;
      label-permutation null (2000/axis) pooled per neuron;
      family maxT over 9 layers (true permuted family max);
      top-128 |Delta| energy share vs permuted null.
  T2 geometry: |cos(W_down[:,i], u)| vs random-rotation null
      (2931 caliber).
  T3 causal: real ablation of top-128 neurons (|Delta_lang|
      rank) per layer (down_proj input slice zero) vs 20
      random same-k controls; stat = change of L2-condition
      language separation (median fr - median en of P=fin.u);
      direct/rebalance share (2950 caliber).

Verdict branches (frozen):
  anchor_fail_all_void
  T1 maxT p<.05 & T1b share p<.05 & T3 p<.05
      -> neuron_registry_concentrated_causal
  T1 or T1b significant, else
      -> snapshot_only_no_causal
  none significant -> distributed_no_neuron_registry

Anchors: a1 Vt8<1e-6; a2 determinism<1e-4; a3 L2 norms/coss
vs 2973; a4 word-order + single-token 74/74; a5 MLP projection
decomposition identity (act@Wd)@u == act@c < 1e-8 rel;
a6 n17 vs 2979 rel<1e-9; a7 B_rec vs 2973 rel<1e-4;
a8 L2 prof34+BL vs 2987 npz bit<1e-12; a9 act hook vs
recompute rel<1e-6; a10 ablated slices exactly zero (<1e-30)
and rest untouched.
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2939 = os.path.join(BASE, 'phase2939', 'rotation_target',
                        'rotation_target.npz')
SRC_2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                        'probe_relativity.npz')
SRC_2973 = os.path.join(BASE, 'phase2973', 'fr_scale_audit',
                        'fr_scale_audit.npz')
EXEC_2977 = os.path.join(BASE, 'phase2977',
                         'two_axis_fusion_injection',
                         'execution.json')
SRC_2979 = os.path.join(BASE, 'phase2979', 'reversal_anatomy',
                        'reversal_anatomy.npz')
SRC_2987 = os.path.join(BASE, 'phase2987', 'context_minimal_audit',
                        'context_minimal_audit.npz')
OUT = os.path.join(BASE, 'phase2989', 'mlp_neuron_registry')

MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
L34 = 34
REG_LAYERS = [6, 7, 8, 9, 10, 11, 12, 17, 34]
N_PERM = 2000
K_TOP = 128
N_RAND = 20
RNG_MAIN = 2989
P_GATE = 0.05
BIT_TOL = 1e-12


def sha8(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]


def log(msg, lines):
    lines.append('[%s] %s' % (time.strftime('%H:%M:%S'), msg))
    with open(os.path.join(OUT, 'run_log.txt'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)

    # ---------- cells (2977 exec verbatim, 2987 proven) ----------
    e77 = json.load(open(EXEC_2977, encoding='utf-8'))
    F_EN = e77['cells']['F_en']
    F_FR = e77['cells']['F_fr']
    C_EN = e77['cells']['C_en']
    C_FR = e77['cells']['C_fr']
    assert (len(F_EN), len(F_FR), len(C_EN), len(C_FR)) \
        == (15, 15, 22, 22), 'cell size drift'
    cells = [('F', 'en', w) for w in F_EN] \
        + [('F', 'fr', w) for w in F_FR] \
        + [('C', 'en', w) for w in C_EN] \
        + [('C', 'fr', w) for w in C_FR]
    n_test = len(cells)
    lang = np.array([0 if c[1] == 'en' else 1 for c in cells])
    cls = np.array([0 if c[0] == 'F' else 1 for c in cells])

    prereg = {
        'design': '74 cells (2977 verbatim), L2 condition '
                  '[the, w] single-sample forwards; registry '
                  'layers %s; neuron = MLP intermediate '
                  '(9728/layer); capture = down_proj input '
                  '(pos 1); axes u_lang/u_cls from 2979 npz'
                  % REG_LAYERS,
        'T1_snapshot': 'per-neuron class contrast of '
                       'axis-projected contribution; null = '
                       '%d label permutations per axis (pooled '
                       'per-neuron |Delta| null + TRUE family '
                       'maxT over 9 layers); top-%d |Delta| '
                       'energy share vs permuted null; gate '
                       'p<%.2f (lang primary, cls secondary)'
                       % (N_PERM, K_TOP, P_GATE),
        'T2_geometry': '|cos(W_down[:,i],u)| distribution vs '
                       'random unit directions (rotation null '
                       '2931); descriptive exceed count vs '
                       'chance 486',
        'T3_causal': 'per layer top-%d neurons by |Delta_lang| '
                     'ablated (down_proj input zero) vs %d '
                     'random same-k controls (rng 2989 stream '
                     '3); stat = change of L2 lang separation '
                     '(median fr - median en of P=fin.u_lang); '
                     'gate p<%.2f one-sided rank in null; '
                     'direct share = predicted/observed '
                     '(2950 caliber)' % (K_TOP, N_RAND,
                                         P_GATE),
        'verdict': 'anchor fail => anchor_fail_all_void; '
                   'T1 maxT p<%.2f and T1b share p<%.2f and '
                   'T3 p<%.2f => '
                   'neuron_registry_concentrated_causal; '
                   '(T1 or T1b) significant else => '
                   'snapshot_only_no_causal; '
                   'else => distributed_no_neuron_registry'
                   % (P_GATE, P_GATE, P_GATE),
        'anchors': {
            'a1': 'Vt8 rebuild < 1e-6',
            'a2': 'determinism L2 rel < 1e-4',
            'a3': 'L2 norms/coss vs 2973: rel<1e-4, cos<1e-6',
            'a4': 'word-order identity + single-token 74/74',
            'a5': 'MLP projection decomposition identity '
                  '(act@Wd)@u == act@c rel < 1e-8',
            'a6': 'n17 vs 2979 rel < 1e-9',
            'a7': 'B_rec vs 2973 rel < 1e-4',
            'a8': 'L2 prof34+BL vs 2987 npz bit < 1e-12',
            'a9': 'act hook vs recompute rel < 1e-6',
            'a10': 'ablated down_proj input slices zero '
                   '(< 1e-30) and rest untouched (< 1e-9)'},
        'nondegeneracy': 'per-layer act std > 0; P baseline '
                         'nonzero; checked before stats',
        'stats': 'T1 primary lang axis with true family maxT; '
                 'cls axis secondary; T3 lang only; raw p '
                 'reported per layer',
    }

    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2989,
                   'name': 'mlp_neuron_registry',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2939': sha8(SRC_2939),
                               's2927': sha8(SRC_2927),
                               's2973': sha8(SRC_2973),
                               's2977exec': sha8(EXEC_2977),
                               's2979npz': sha8(SRC_2979),
                               's2987npz': sha8(SRC_2987)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'l34': L34, 'reg_layers': REG_LAYERS,
                   'n_perm': N_PERM, 'k_top': K_TOP,
                   'n_rand': N_RAND, 'rng': RNG_MAIN,
                   'p_gate': P_GATE,
                   'cells': {'F_en': F_EN, 'F_fr': F_FR,
                             'C_en': C_EN, 'C_fr': C_FR},
                   'prereg': prereg},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- sources ----------
    z39 = np.load(SRC_2939, allow_pickle=True)
    z27 = np.load(SRC_2927, allow_pickle=True)
    dirs27 = z27['dirs_word'].astype(np.float64)
    _, _, Vt_loc = np.linalg.svd(dirs27, full_matrices=False)
    a1_diff = float(np.abs(Vt_loc[:8] - z39['Vt8']).max())
    a1_ok = bool(a1_diff < 1e-6)
    log('a1 Vt8 rebuild diff %.2e ok=%s' % (a1_diff, a1_ok),
        lines)
    u35 = dirs27[NL - 1]
    z73 = np.load(SRC_2973, allow_pickle=True)
    norms73 = z73['norms'].astype(np.float64)
    coss73 = z73['coss'].astype(np.float64)
    B73 = z73['B_rec'].astype(np.float64)
    words73 = [str(w) for w in z73['words']]
    z79 = np.load(SRC_2979, allow_pickle=True)
    d79_l = z79['d_lang_u'].astype(np.float64)
    d79_c = z79['d_cls_u'].astype(np.float64)
    n17_79 = z79['n17'].astype(np.float64).reshape(-1)
    words79 = [str(w) for w in z79['words']]
    z87 = np.load(SRC_2987, allow_pickle=True)
    prof34_87 = z87['prof_all'].astype(np.float64)
    BL_87 = z87['BL'].astype(np.float64)
    words87 = [str(w) for w in z87['words']]
    cells77 = ['%s:%s:%s' % c for c in cells]
    a4_ok = bool(cells77 == words73 and words79 == words73
                 and words87 == words73)
    log('a4 word-order identity ok=%s' % a4_ok, lines)

    # ---------- model ----------
    import sys
    sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
    from phase2662_symmetric_mapping_contract import \
        load_native
    from transformers import AutoTokenizer
    import torch as _t

    tok = AutoTokenizer.from_pretrained(
        MD, local_files_only=True, trust_remote_code=True,
        use_fast=True)
    tid_map = {}
    n_single = 0
    for _, _, w in cells:
        ids = tok(' ' + w, add_special_tokens=False)[
            'input_ids']
        if len(ids) != 1:
            ids = tok(w, add_special_tokens=False)[
                'input_ids']
        if len(ids) == 1:
            n_single += 1
        tid_map[w] = int(ids[0]) if len(ids) == 1 else -1
    a4b_ok = bool(n_single == n_test)
    log('a4b single-token %d/%d ok=%s'
        % (n_single, n_test, a4b_ok), lines)
    ids_the = tok(' the', add_special_tokens=False)[
        'input_ids']
    assert len(ids_the) == 1
    func_tid = int(ids_the[0])

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    log('model loaded', lines)

    # hooks
    cap_act = {li: [] for li in REG_LAYERS}
    cap_op = {li: [] for li in range(NL)}
    cap_x17 = []
    fin_cap = {}
    abl = {'li': None, 'idx': None}
    handles = []

    def hs_of(args, kwargs):
        if args and args[0] is not None:
            return args[0]
        return kwargs.get('hidden_states')

    def hook_x17(module, args, kwargs):
        x = hs_of(args, kwargs)
        if x is None or x.dim() < 2:
            return None
        cap_x17.append(
            x[:, -1, :].detach().float().cpu().numpy())
        return None

    def hook_op(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('input')
            if x is None or x.dim() < 2:
                return None
            cap_op[li].append(
                x[:, -1, :].detach().float().cpu().numpy())
            return None
        return h

    def hook_down(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('input')
            if x is None or x.dim() < 2:
                return None
            modified = False
            if abl['li'] == li and abl['idx'] is not None:
                x = x.clone()
                x[:, 1, abl['idx']] = 0.0
                modified = True
            cap_act[li].append(
                x[:, 1, :].detach().float().cpu().numpy())
            if modified:
                if args:
                    args = (x,) + tuple(args[1:])
                else:
                    kwargs = dict(kwargs)
                    kwargs['input'] = x
                return args, kwargs
            return None
        return h

    def pre_norm(module, args, kwargs):
        fin_cap['x'] = args[0][:, -1, :].detach() \
            .float().cpu().numpy()

    handles.append(
        layers[17].self_attn
        .register_forward_pre_hook(
            hook_x17, with_kwargs=True))
    for li in range(NL):
        handles.append(
            layers[li].self_attn.o_proj
            .register_forward_pre_hook(
                hook_op(li), with_kwargs=True))
        if li in REG_LAYERS:
            handles.append(
                layers[li].mlp.down_proj
                .register_forward_pre_hook(
                    hook_down(li), with_kwargs=True))
    handles.append(model.model.norm.register_forward_pre_hook(
        pre_norm, with_kwargs=True))

    def clear_cap():
        for li in cap_act:
            del cap_act[li][:]
        for li in cap_op:
            del cap_op[li][:]
        del cap_x17[:]
        fin_cap.pop('x', None)

    def forward1(toks, abl_li=None, abl_idx=None):
        clear_cap()
        abl['li'] = abl_li
        abl['idx'] = None if abl_idx is None \
            else np.ascontiguousarray(
                abl_idx, dtype=np.int64)
        with _t.no_grad():
            model(_t.tensor([toks], device='cuda'))
        abl['li'] = None
        abl['idx'] = None
        return (fin_cap['x'].astype(np.float64),
                {li: cap_act[li][0].astype(np.float64)
                 for li in REG_LAYERS},
                {li: cap_op[li][0].astype(np.float64)
                 for li in range(NL)})

    # weight-space registry constants
    # Wd[li]: (9728, 2560), row i = down_proj column i
    Wd = {}
    c_lang = {}
    c_cls = {}
    for li in REG_LAYERS:
        Wdm = layers[li].mlp.down_proj.weight.detach() \
            .float().cpu().numpy().astype(np.float64)
        Wd[li] = Wdm.T
        c_lang[li] = Wd[li] @ d79_l
        c_cls[li] = Wd[li] @ d79_c
    Wo_all = np.zeros((NL, NH * HD))
    Mnorm = np.zeros(NL)
    for li in range(NL):
        Wo = layers[li].self_attn.o_proj.weight.detach() \
            .float().cpu().numpy()
        Wo_all[li] = u35 @ Wo
        Mnorm[li] = float(np.linalg.norm(Wo_all[li]))
    Wo34 = layers[L34].self_attn.o_proj.weight.detach() \
        .float().cpu().numpy().astype(np.float64)
    C34 = u35 @ Wo34

    # ---------- main sweep: 74 cells L2 ----------
    act_reg = {li: np.zeros((n_test, 9728))
               for li in REG_LAYERS}
    prof_all = np.zeros((n_test, NL))
    normsL = np.zeros((n_test, NL))
    cossL = np.zeros((n_test, NL))
    BL = np.zeros(n_test)
    n17L = np.zeros(n_test)
    headC = np.zeros((n_test, NH))
    finP = np.zeros((n_test, 2560))
    for i, (_, _, w) in enumerate(cells):
        seq = [func_tid, tid_map[w]]
        fin, acts, ops = forward1(seq)
        finP[i] = fin @ d79_l
        for li in range(NL):
            x = ops[li].reshape(-1)
            normsL[i, li] = float(np.linalg.norm(x))
            pr = float(np.dot(x, Wo_all[li]))
            prof_all[i, li] = pr
            cossL[i, li] = pr / max(
                normsL[i, li] * Mnorm[li], 1e-30)
        BL[i] = (float(prof_all[i, 6:13].mean())
                 - float(prof_all[i, 28:36].mean()))
        n17L[i] = float(np.linalg.norm(
            cap_x17[0].astype(np.float64).reshape(-1)))
        x34 = ops[L34].reshape(-1)
        for h in range(NH):
            headC[i, h] = float(np.dot(
                C34[h * HD:(h + 1) * HD],
                x34[h * HD:(h + 1) * HD]))
        for li in REG_LAYERS:
            act_reg[li][i] = acts[li].reshape(-1)
        if (i + 1) % 20 == 0:
            log('cells [%d/%d]' % (i + 1, n_test), lines)

    # a9: act hook vs recompute (word 'man', layer 8)
    li_chk = 8
    seq_chk = [func_tid, tid_map['man']]

    def _recompute_act(li):
        # capture the MLP module input (post-attention
        # layernorm output = true down_proj input source),
        # then recompute the intermediate activation by hand
        got = {}

        def mlp_pre(module, args, kwargs):
            x = args[0] if args else kwargs.get(
                'hidden_states')
            got['x'] = x.detach()

        hd_h = layers[li].mlp.register_forward_pre_hook(
            mlp_pre, with_kwargs=True)
        with _t.no_grad():
            model(_t.tensor([seq_chk], device='cuda'))
        hd_h.remove()
        mlp = layers[li].mlp
        x = got['x']
        rec = mlp.act_fn(mlp.gate_proj(x)) \
            * mlp.up_proj(x)
        return rec[:, 1, :].detach().float() \
            .cpu().numpy().reshape(-1)

    _, acts_chk, _ = forward1(seq_chk)
    rec9 = _recompute_act(li_chk)
    got9 = acts_chk[li_chk].reshape(-1)
    a9_rel = float(np.abs(rec9 - got9).max()
                   / max(float(np.abs(got9).max()), 1e-30))
    a9_ok = bool(a9_rel < 1e-6)
    log('a9 act hook vs recompute rel %.2e ok=%s'
        % (a9_rel, a9_ok), lines)

    # ---------- anchors ----------
    a2_seq = seq_chk
    _, _, opa = forward1(a2_seq)
    _, _, opb = forward1(a2_seq)
    a2_rel = float(np.abs(opa[30] - opb[30]).max()
                   / max(float(np.abs(opa[30]).max()),
                         1e-30))
    a2_ok = bool(a2_rel < 1e-4)
    log('a2 determinism L2 rel %.2e ok=%s'
        % (a2_rel, a2_ok), lines)

    a3_rel = float(np.abs(normsL - norms73).max()
                   / max(float(np.abs(norms73).max()), 1e-30))
    a3_cos = float(np.abs(cossL - coss73).max())
    a3_ok = bool(a3_rel < 1e-4 and a3_cos < 1e-6)
    log('a3 L2 norms/coss vs 2973 rel %.2e cos %.2e ok=%s'
        % (a3_rel, a3_cos, a3_ok), lines)

    # a5: projection decomposition identity
    a5_rel = 0.0
    for li in REG_LAYERS:
        lhs = act_reg[li] @ c_lang[li]
        rhs = (act_reg[li] @ Wd[li]) @ d79_l
        a5_rel = max(a5_rel, float(
            np.abs(lhs - rhs).max()
            / max(float(np.abs(rhs).max()), 1e-30)))
    a5_ok = bool(a5_rel < 1e-8)
    log('a5 projection decomposition rel %.2e ok=%s'
        % (a5_rel, a5_ok), lines)

    a6_rel = float(np.abs(n17L - n17_79).max()
                   / max(float(np.abs(n17_79).mean()),
                         1e-30))
    a6_ok = bool(a6_rel < 1e-9)
    log('a6 n17 vs 2979 rel %.2e ok=%s' % (a6_rel, a6_ok),
        lines)

    a7_rel = float(np.abs(BL - B73).max()
                   / max(float(np.abs(B73).max()), 1e-30))
    a7_ok = bool(a7_rel < 1e-4)
    log('a7 B_rec vs 2973 rel %.2e ok=%s' % (a7_rel, a7_ok),
        lines)

    a8_diff = float(np.abs(prof_all - prof34_87[0]).max())
    a8_diff = max(a8_diff,
                  float(np.abs(BL - BL_87[0]).max()))
    a8_ok = bool(a8_diff < BIT_TOL)
    log('a8 L2 prof34+BL vs 2987 bit %.2e ok=%s'
        % (a8_diff, a8_ok), lines)

    # a10: ablation slice-zero check
    rng = np.random.default_rng([RNG_MAIN, 3])
    idx_rand = rng.choice(9728, K_TOP, replace=False)
    _, acts_abl, _ = forward1(seq_chk, abl_li=li_chk,
                              abl_idx=idx_rand)
    ab9 = acts_abl[li_chk].reshape(-1)
    ck9 = acts_chk[li_chk].reshape(-1)
    a10_zero = float(np.abs(ab9[idx_rand]).max())
    keep = np.setdiff1d(np.arange(9728), idx_rand)
    rest_diff = float(np.abs(ab9[keep] - ck9[keep]).max())
    a10_ok = bool(a10_zero < 1e-30 and rest_diff < 1e-9)
    log('a10 ablated slices zero %.2e rest-diff %.2e ok=%s'
        % (a10_zero, rest_diff, a10_ok), lines)

    anchor_ok = all([a1_ok, a2_ok, a3_ok, a4_ok, a4b_ok,
                     a5_ok, a6_ok, a7_ok, a8_ok, a9_ok,
                     a10_ok])

    # non-degeneracy gates
    act_std = {li: float(act_reg[li].std())
               for li in REG_LAYERS}
    P_base = float(np.abs(finP).mean())
    gates_ok = bool(all(v > 0 for v in act_std.values())
                    and P_base > 0)

    # ---------- pre-init all result containers ----------
    verdict = None
    T1 = T2 = T3 = None
    p_lang = p_lang_share = p_t3 = None
    Delta = {ax: {li: np.zeros(9728) for li in REG_LAYERS}
             for ax in ('lang', 'cls')}
    maxT_p = {ax: {li: float('nan') for li in REG_LAYERS}
              for ax in ('lang', 'cls')}
    share_top = {ax: {li: (float('nan'), float('nan'))
                      for li in REG_LAYERS}
                 for ax in ('lang', 'cls')}
    sig_count = {ax: {li: -1 for li in REG_LAYERS}
                 for ax in ('lang', 'cls')}
    cos_stat = {ax: {li: None for li in REG_LAYERS}
                for ax in ('lang', 'cls')}

    if anchor_ok and gates_ok:
        axes = {'lang': d79_l, 'cls': d79_c}
        labels = {'lang': lang, 'cls': cls}
        perm_rng = np.random.default_rng([RNG_MAIN, 1])
        for ax in ('lang', 'cls'):
            lab = labels[ax]
            n0 = int((lab == 0).sum())
            n1 = int((lab == 1).sum())
            perm_max = {}
            for li in REG_LAYERS:
                proj = act_reg[li] \
                    * (c_lang[li] if ax == 'lang'
                       else c_cls[li])[None, :]
                D = (proj[lab == 0].mean(0)
                     - proj[lab == 1].mean(0))
                Delta[ax][li] = D
                obs_max = float(np.abs(D).max())
                s_sorted = np.sort(np.abs(D))[::-1]
                share_obs = float(s_sorted[:K_TOP].sum()
                                  / max(float(s_sorted.sum()),
                                        1e-30))
                pool = np.zeros((N_PERM, 9728),
                                dtype=np.float32)
                PT = np.ascontiguousarray(proj.T)
                for pi in range(N_PERM):
                    pl = perm_rng.permutation(lab)
                    w = ((pl == 0).astype(np.float64) / n0
                         - (pl == 1).astype(np.float64) / n1)
                    dn = PT @ w
                    pool[pi] = dn.astype(np.float32)
                abs_pool = np.abs(pool)
                perm_max[li] = abs_pool.max(axis=1)
                p95_pool = np.quantile(abs_pool, 0.95,
                                       axis=0)
                sig_count[ax][li] = int(
                    (np.abs(D) > p95_pool).sum())
                share_null = np.zeros(N_PERM)
                for pi in range(N_PERM):
                    sn = np.sort(abs_pool[pi])[::-1]
                    share_null[pi] = float(
                        sn[:K_TOP].sum()
                        / max(float(sn.sum()), 1e-30))
                share_top[ax][li] = (
                    share_obs,
                    float((share_null
                           >= share_obs - 1e-30).sum())
                    / N_PERM)
                # per-layer raw p (also true maxT input)
                maxT_p[ax][li] = float(
                    (perm_max[li]
                     >= obs_max - 1e-30).sum()) / N_PERM
                del pool
            # true family maxT over 9 layers
            fam = np.stack([perm_max[li]
                            for li in REG_LAYERS])
            fam_max = fam.max(axis=0)
            for j, li in enumerate(REG_LAYERS):
                obs_max = float(
                    np.abs(Delta[ax][li]).max())
                maxT_p[ax][li] = float(
                    (fam_max
                     >= obs_max - 1e-30).sum()) / N_PERM
            log('T1 %s done (maxT min p %.4f)'
                % (ax, min(maxT_p[ax].values())), lines)

        # ---------- T2 geometry ----------
        rng_rot = np.random.default_rng([RNG_MAIN, 2])
        for ax in ('lang', 'cls'):
            for li in REG_LAYERS:
                cvec = c_lang[li] if ax == 'lang' \
                    else c_cls[li]
                R = rng_rot.normal(size=(256, 2560))
                R /= np.linalg.norm(R, axis=1,
                                    keepdims=True)
                cosnull = np.abs(Wd[li] @ R.T).ravel()
                p95 = float(np.quantile(cosnull, 0.95))
                cos_stat[ax][li] = {
                    'obs_absmax': float(np.abs(cvec).max()),
                    'null_p95': p95,
                    'n_exceed': int(
                        (np.abs(cvec) > p95).sum()),
                    'chance': int(0.05 * 9728)}

        # ---------- T3 causal ablation ----------
        def sep_of(P):
            return float(np.median(P[lang == 1])
                         - np.median(P[lang == 0]))

        sep0 = sep_of(finP)
        T3 = {}
        for li in REG_LAYERS:
            order = np.argsort(
                np.abs(Delta['lang'][li]))[::-1]
            idx_top = order[:K_TOP]
            dP_fr = -(act_reg[li][lang == 1]
                      * c_lang[li][None, :])[:, idx_top] \
                .sum(1)
            dP_en = -(act_reg[li][lang == 0]
                      * c_lang[li][None, :])[:, idx_top] \
                .sum(1)
            dP_pred = float(np.median(dP_fr)
                            - np.median(dP_en))
            P_abl = np.zeros((n_test, 2560))
            for i, (_, _, w) in enumerate(cells):
                seq = [func_tid, tid_map[w]]
                fin_i, _, _ = forward1(
                    seq, abl_li=li, abl_idx=idx_top)
                P_abl[i] = fin_i @ d79_l
            dsep_real = sep_of(P_abl) - sep0
            dsep_rand = np.zeros(N_RAND)
            for r in range(N_RAND):
                idx_r = rng.choice(9728, K_TOP,
                                   replace=False)
                P_r = np.zeros((n_test, 2560))
                for i, (_, _, w) in enumerate(cells):
                    seq = [func_tid, tid_map[w]]
                    fin_i, _, _ = forward1(
                        seq, abl_li=li, abl_idx=idx_r)
                    P_r[i] = fin_i @ d79_l
                dsep_rand[r] = sep_of(P_r) - sep0
            p_rank = float(
                (np.abs(dsep_rand)
                 >= abs(dsep_real) - 1e-30).sum() + 1) \
                / (N_RAND + 1)
            T3[li] = {
                'dsep_real': dsep_real,
                'dsep_pred_direct': dP_pred,
                'dsep_rand_median':
                    float(np.median(dsep_rand)),
                'dsep_rand_p95': float(np.quantile(
                    np.abs(dsep_rand), 0.95)),
                'p_rank': p_rank,
                'direct_share':
                    float(dP_pred / dsep_real)
                    if abs(dsep_real) > 1e-30 else None}
            log('T3 L%d dsep_real %.3f p=%.3f'
                % (li, dsep_real, p_rank), lines)
        p_t3 = min(T3[li]['p_rank'] for li in REG_LAYERS)

        p_lang = min(maxT_p['lang'].values())
        p_lang_share = min(
            share_top['lang'][li][1]
            for li in REG_LAYERS)
        # ---------- verdict ----------
        if p_lang < P_GATE and p_lang_share < P_GATE \
                and p_t3 < P_GATE:
            verdict = 'neuron_registry_concentrated_causal'
        elif p_lang < P_GATE or p_lang_share < P_GATE:
            verdict = 'snapshot_only_no_causal'
        else:
            verdict = 'distributed_no_neuron_registry'
        log('T1 lang maxT p=%.4f share p=%.4f; T3 p=%.4f'
            % (p_lang, p_lang_share, p_t3), lines)
    else:
        log('ANCHOR/GATE FAIL: anchor_ok=%s gates_ok=%s'
            % (anchor_ok, gates_ok), lines)
        for li in REG_LAYERS:
            Delta['lang'][li] = np.zeros(9728)
            Delta['cls'][li] = np.zeros(9728)

    elapsed = time.monotonic() - t0
    if verdict is None:
        verdict = 'anchor_fail_all_void' \
            if not anchor_ok else 'gate_fail_all_void'

    # ---------- persist ----------
    res = {
        'phase': 2989,
        'final_verdict': verdict,
        'anchor_all_ok': bool(anchor_ok and gates_ok),
        'anchors': {
            'a1_diff': a1_diff, 'a2_rel': a2_rel,
            'a3_rel': a3_rel, 'a3_cos': a3_cos,
            'a4_word_identity': a4_ok,
            'a4b_single_token': a4b_ok,
            'a5_rel': a5_rel, 'a6_rel': a6_rel,
            'a7_rel': a7_rel, 'a8_diff': a8_diff,
            'a9_rel': a9_rel, 'a10_zero': a10_zero},
        'nondegeneracy': {'act_std': act_std,
                          'P_base_mean_abs': P_base},
        'T1': {'maxT_p': {ax: {str(li): maxT_p[ax][li]
                               for li in REG_LAYERS}
                          for ax in ('lang', 'cls')},
               'sig_count': {ax: {str(li): sig_count[ax][li]
                                  for li in REG_LAYERS}
                             for ax in ('lang', 'cls')},
               'share_top': {ax: {str(li): [
                   share_top[ax][li][0],
                   share_top[ax][li][1]]
                   for li in REG_LAYERS}
                   for ax in ('lang', 'cls')}},
        'T2': {ax: {str(li): cos_stat[ax][li]
                    for li in REG_LAYERS}
                for ax in ('lang', 'cls')},
        'T3': {str(li): {k: (float(v) if v is not None
                             else None)
                         for k, v in T3[li].items()}
                for li in REG_LAYERS} if T3 else None,
        'p_lang_maxT': p_lang,
        'p_lang_share': p_lang_share,
        'p_t3_min': p_t3,
        'elapsed_s': round(elapsed, 1),
        'correction_note': 'none',
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    np.savez_compressed(
        os.path.join(OUT, 'mlp_neuron_registry.npz'),
        words=np.array(cells77), lang=lang, cls=cls,
        finP=finP,
        **{('act_%d' % li): act_reg[li]
           for li in REG_LAYERS},
        **{('delta_lang_%d' % li): Delta['lang'][li]
           for li in REG_LAYERS},
        **{('delta_cls_%d' % li): Delta['cls'][li]
           for li in REG_LAYERS},
        **{('c_lang_%d' % li): c_lang[li]
           for li in REG_LAYERS},
        **{('c_cls_%d' % li): c_cls[li]
           for li in REG_LAYERS},
        prof34=prof_all, BL=BL, headC=headC)
    log('PHASE2989 DONE verdict=%s elapsed=%.1fs'
        % (verdict, elapsed), lines)
    print('PHASE2989 DONE verdict=%s elapsed=%.1fs'
          % (verdict, elapsed), flush=True)


if __name__ == '__main__':
    main()
