# -*- coding: utf-8 -*-
"""Phase 2990: L12 word-class neuron deep-dive (plan v4 P1.2).

2989 found a single neuron standing out on the cls axis:
L12 neuron #2 (c_cls 0.588, maxT p<1e-4, top-128 energy share
0.762, down column near-collinear). This phase asks: what IS
it, is it causally load-bearing alone, and how does it couple
to the 2964 head-level carrier (L34/h15)?

Design:
  A) 74-cell registry replication (2989 verbatim, L12 focus):
     act_12 (74, 9728) bit-level anchor vs 2989 npz.
  B) 2964 30-word protocol replication (stored tids verbatim):
     headC30 (30, 32) at L34 bit-level anchor vs stored C[34];
     h15 = argmax|gap| identity.
  C) T2 causal: real ablation of L12 neuron #2 alone (k=1,
     down_proj input zero) vs 99 distinct random single-neuron
     controls; stat = change of L2-condition cls separation
     (median C - median F of P=fin.u_cls); N_RAND=99 -> exact
     rank p resolution 1/100. Dose curve f in {.25,.5,.75,1}:
     effect(f) = sep_base - sep(f); monotone gate = exact
     permutation spearman(f, effect) one-sided p<.05 (24 perms).
  D) T3 cross-granularity (descriptive): same ablations run on
     the 2964 30-word set; Delta gap per head at L34; h15 shift
     vs the 99-run null.
  E) T1/T4 descriptive: neuron #2 activation profile (74+30
     words, point-biserial r_cls with permutation p, top/tail
     words); down-column geometry (cos vs u_cls percentile).

Verdict branches (frozen):
  anchor_fail_all_void
  gates fail -> degenerate_void
  T2 p<.05 & dose p<.05 -> neuron2_single_causal_monotone
  T2 p<.05 & dose p>=.05 -> neuron2_single_causal_nonmonotone
  T2 p>=.05 -> neuron2_not_single_causal

Anchors:
  a1 Vt8 rebuild < 1e-6
  a2 determinism < 1e-4
  a3 c_cls_12 recompute vs 2989 bit < 1e-12
  a4 act_12 vs 2989 npz bit < 1e-12
  a5 2964 gap recompute (stored C[34]) < 1e-5 rel + argmax==15
  a5b headC30 vs 2964 stored C[34] rel < 1e-5
  a6 n17 vs 2979 rel < 1e-9
  a7 L2 prof34+BL vs 2987 bit < 1e-12
  a8 single-token 74/74 + 30/30 (stored tids used for 30)
  a9 act hook vs recompute rel < 1e-6
  a10 k=1 ablation slice exactly zero + rest untouched
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
SRC_2989 = os.path.join(BASE, 'phase2989', 'mlp_neuron_registry',
                        'mlp_neuron_registry.npz')
SRC_2964 = os.path.join(BASE, 'phase2964', 'carrier_anatomy',
                        'carrier_anatomy.npz')
OUT = os.path.join(BASE, 'phase2990', 'neuron2_deep_dive')

MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
L12 = 12
L34 = 34
N2 = 2            # the L12 neuron under audit
N_PERM = 2000
N_RAND = 99
DOSE = [0.25, 0.5, 0.75]
RNG_MAIN = 2990
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


def rankdata(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    if den < 1e-30:
        return 0.0
    return float((ra * rb).sum() / den)


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)

    # ---------- cells (2977 exec verbatim, 2989 proven) ----------
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
    n74 = len(cells)
    lang = np.array([0 if c[1] == 'en' else 1 for c in cells])
    cls = np.array([0 if c[0] == 'F' else 1 for c in cells])

    prereg = {
        'design': 'two word sets: 74 cells (2977 verbatim, L2 '
                  '[the, w]) + 2964 30-word set (stored tids '
                  'verbatim); neuron under audit = L12 #%d '
                  '(2989 cls top-1); single-sample forwards'
                  % N2,
        'T2_causal': 'ablate L12 neuron %d alone (down_proj '
                     'input zero) vs %d distinct random '
                     'single-neuron controls (rng 2990 stream '
                     '3); stat = change of cls separation '
                     '(median C - median F of P=fin.u_cls); '
                     'exact rank p; dose f in %s, effect(f)='
                     'sep_base - sep(f), monotone gate = '
                     'exact perm spearman one-sided p<%.2f '
                     '(24 perms, expect rho>0)'
                     % (N2, N_RAND, DOSE, P_GATE),
        'T3_cross': 'descriptive: same ablations on the 2964 '
                    '30-word set; per-head gap shift at L34; '
                    'h15 shift vs %d-run null (no gate)'
                    % N_RAND,
        'T1_scan': 'descriptive: neuron %d activation profile '
                   '(74+30 words), point-biserial r_cls with '
                   '%d-label permutations, top/tail words'
                   % (N2, N_PERM),
        'T4_geom': 'descriptive: cos(W_down[:,%d], u_cls) '
                   'percentile among 9728; c_lang contrast'
                   % N2,
        'verdict': 'anchor fail => anchor_fail_all_void; '
                   'gates fail => degenerate_void; '
                   'T2 p<%.2f & dose p<%.2f => '
                   'neuron2_single_causal_monotone; '
                   'T2 p<%.2f & dose p>=.05 => '
                   'neuron2_single_causal_nonmonotone; '
                   'T2 p>=.05 => neuron2_not_single_causal'
                   % (P_GATE, P_GATE, P_GATE),
        'anchors': {
            'a1': 'Vt8 rebuild < 1e-6',
            'a2': 'determinism < 1e-4',
            'a3': 'c_cls_12 recompute vs 2989 bit < 1e-12',
            'a4': 'act_12 vs 2989 npz bit < 1e-12',
            'a5': '2964 gap recompute < 1e-5 rel + argmax 15',
            'a5b': 'headC30 vs 2964 C[34] rel < 1e-5',
            'a6': 'n17 vs 2979 rel < 1e-9',
            'a7': 'L2 prof34+BL vs 2987 bit < 1e-12',
            'a8': 'single-token 74/74; 30/30 stored tids',
            'a9': 'act hook vs recompute rel < 1e-6',
            'a10': 'k=1 ablation slice zero + rest untouched'},
        'nondegeneracy': 'act12 std>0; |sep_base|>0; '
                         'dsep_rand spread>0',
        'stats': 'T2 primary; dose secondary; T1/T3/T4 '
                 'descriptive (registered)',
    }

    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2990,
                   'name': 'neuron2_deep_dive',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2939': sha8(SRC_2939),
                               's2927': sha8(SRC_2927),
                               's2973': sha8(SRC_2973),
                               's2977exec': sha8(EXEC_2977),
                               's2979npz': sha8(SRC_2979),
                               's2987npz': sha8(SRC_2987),
                               's2989npz': sha8(SRC_2989),
                               's2964npz': sha8(SRC_2964)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'l12': L12, 'l34': L34, 'neuron': N2,
                   'n_perm': N_PERM, 'n_rand': N_RAND,
                   'dose': DOSE, 'rng': RNG_MAIN,
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
    z87 = np.load(SRC_2987, allow_pickle=True)
    prof34_87 = z87['prof_all'].astype(np.float64)
    BL_87 = z87['BL'].astype(np.float64)
    z89 = np.load(SRC_2989, allow_pickle=True)
    z64 = np.load(SRC_2964, allow_pickle=True)
    words30 = [str(w) for w in z64['words']]
    tids30 = z64['tids'].astype(np.int64)
    lab30 = z64['labels'].astype(np.int64)
    C34_st = z64['C'][34].astype(np.float64)
    gap_st = z64['gap_heads_L34'].astype(np.float64)

    cells77 = ['%s:%s:%s' % c for c in cells]
    a8a_ok = bool(cells77 == words73)
    log('a8a word-order identity 74 ok=%s' % a8a_ok, lines)

    # a5: gap recompute from stored C[34]
    gap_rec = np.array([
        float(np.median(C34_st[lab30 == 0, h])
              - np.median(C34_st[lab30 == 1, h]))
        for h in range(NH)])
    a5_rel = float(np.abs(gap_rec - gap_st).max()
                   / max(float(np.abs(gap_st).max()), 1e-30))
    a5_arg = int(np.argmax(np.abs(gap_st)))
    a5_ok = bool(a5_rel < 1e-5 and a5_arg == 15)
    log('a5 gap recompute rel %.2e argmax=%d ok=%s'
        % (a5_rel, a5_arg, a5_ok), lines)

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
    a8b_ok = bool(n_single == n74)
    log('a8b single-token 74: %d/%d ok=%s'
        % (n_single, n74, a8b_ok), lines)
    ids_the = tok(' the', add_special_tokens=False)[
        'input_ids']
    assert len(ids_the) == 1
    func_tid = int(ids_the[0])
    a8c_ok = True
    for w, t in zip(words30, tids30):
        ids = tok(' ' + w.split(':')[1],
                  add_special_tokens=False)['input_ids']
        if len(ids) != 1 or int(ids[0]) != int(t):
            a8c_ok = False
            break
    log('a8c 2964 stored tids reproduce 30/30 ok=%s'
        % a8c_ok, lines)

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    log('model loaded', lines)

    # hooks
    cap_act = []
    cap_op = {li: [] for li in range(NL)}
    cap_x17 = []
    fin_cap = {}
    abl = {'idx': None, 'factor': 1.0}
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

    def hook_down12(module, args, kwargs):
        x = args[0] if args else kwargs.get('input')
        if x is None or x.dim() < 2:
            return None
        modified = False
        if abl['idx'] is not None:
            x = x.clone()
            idx_t = _t.as_tensor(
                np.ascontiguousarray(abl['idx']),
                dtype=_t.long, device=x.device)
            x[:, 1, idx_t] = x[:, 1, idx_t] \
                * (1.0 - abl['factor'])
            modified = True
        cap_act.append(
            x[:, 1, :].detach().float().cpu().numpy())
        if modified:
            if args:
                args = (x,) + tuple(args[1:])
            else:
                kwargs = dict(kwargs)
                kwargs['input'] = x
            return args, kwargs
        return None

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
    handles.append(
        layers[L12].mlp.down_proj
        .register_forward_pre_hook(
            hook_down12, with_kwargs=True))
    handles.append(model.model.norm.register_forward_pre_hook(
        pre_norm, with_kwargs=True))

    def clear_cap():
        del cap_act[:]
        for li in cap_op:
            del cap_op[li][:]
        del cap_x17[:]
        fin_cap.pop('x', None)

    def forward1(toks, abl_idx=None, factor=1.0):
        clear_cap()
        abl['idx'] = abl_idx
        abl['factor'] = factor
        with _t.no_grad():
            model(_t.tensor([toks], device='cuda'))
        abl['idx'] = None
        abl['factor'] = 1.0
        return (fin_cap['x'].astype(np.float64),
                cap_act[0].astype(np.float64).reshape(-1),
                {li: cap_op[li][0].astype(np.float64)
                 for li in range(NL)})

    # weight-space constants (L12 focus)
    Wdm = layers[L12].mlp.down_proj.weight.detach() \
        .float().cpu().numpy().astype(np.float64)
    Wd12 = Wdm.T                       # (9728, 2560)
    c_lang12 = Wd12 @ d79_l
    c_cls12 = Wd12 @ d79_c
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

    # ---------- sweep A: 74 cells ----------
    act12_74 = np.zeros((n74, 9728))
    prof_all = np.zeros((n74, NL))
    normsL = np.zeros((n74, NL))
    cossL = np.zeros((n74, NL))
    BL = np.zeros(n74)
    n17L = np.zeros(n74)
    finP_cls = np.zeros(n74)
    for i, (_, _, w) in enumerate(cells):
        seq = [func_tid, tid_map[w]]
        fin, act12_i, ops = forward1(seq)
        finP_cls[i] = float(
            (fin.reshape(-1)) @ d79_c)
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
        act12_74[i] = act12_i
        if (i + 1) % 20 == 0:
            log('A cells [%d/%d]' % (i + 1, n74), lines)

    # ---------- sweep B: 2964 30-word set ----------
    headC30 = np.zeros((30, NH))
    for i, w in enumerate(words30):
        seq = [func_tid, int(tids30[i])]
        _, _, ops = forward1(seq)
        x34 = ops[L34].reshape(-1)
        for h in range(NH):
            headC30[i, h] = float(np.dot(
                C34[h * HD:(h + 1) * HD],
                x34[h * HD:(h + 1) * HD]))
    log('B 30-word sweep done', lines)

    # a5b: headC30 vs stored C[34]
    a5b_rel = float(np.abs(headC30 - C34_st).max()
                    / max(float(np.abs(C34_st).max()),
                          1e-30))
    a5b_ok = bool(a5b_rel < 1e-5)
    log('a5b headC30 vs 2964 C[34] rel %.2e ok=%s'
        % (a5b_rel, a5b_ok), lines)

    # ---------- anchors ----------
    a2_seq = [func_tid, tid_map['man']]
    _, acta, opa = forward1(a2_seq)
    _, actb, opb = forward1(a2_seq)
    a2_rel = float(np.abs(opa[30] - opb[30]).max()
                   / max(float(np.abs(opa[30]).max()),
                         1e-30))
    a2_ok = bool(a2_rel < 1e-4)
    log('a2 determinism rel %.2e ok=%s' % (a2_rel, a2_ok),
        lines)

    a3_diff = float(np.abs(c_cls12 - z89['c_cls_12']
                           .astype(np.float64)).max())
    a3_ok = bool(a3_diff < BIT_TOL)
    log('a3 c_cls_12 recompute vs 2989 bit %.2e ok=%s'
        % (a3_diff, a3_ok), lines)

    a4_diff = float(np.abs(act12_74
                           - z89['act_12'].astype(
                               np.float64)).max())
    a4_ok = bool(a4_diff < BIT_TOL)
    log('a4 act_12 vs 2989 npz bit %.2e ok=%s'
        % (a4_diff, a4_ok), lines)

    a6_rel = float(np.abs(n17L - n17_79).max()
                   / max(float(np.abs(n17_79).mean()),
                         1e-30))
    a6_ok = bool(a6_rel < 1e-9)
    log('a6 n17 vs 2979 rel %.2e ok=%s' % (a6_rel, a6_ok),
        lines)

    a7_diff = max(
        float(np.abs(prof_all - prof34_87[0]).max()),
        float(np.abs(BL - BL_87[0]).max()))
    a7_ok = bool(a7_diff < BIT_TOL)
    log('a7 L2 prof34+BL vs 2987 bit %.2e ok=%s'
        % (a7_diff, a7_ok), lines)

    # a9: act hook vs recompute (word 'man', L12)
    got = {}

    def mlp_pre(module, args, kwargs):
        x = args[0] if args else kwargs.get(
            'hidden_states')
        got['x'] = x.detach()

    hd_h = layers[L12].mlp.register_forward_pre_hook(
        mlp_pre, with_kwargs=True)
    with _t.no_grad():
        model(_t.tensor([a2_seq], device='cuda'))
    hd_h.remove()
    mlp = layers[L12].mlp
    xg = got['x']
    rec9 = mlp.act_fn(mlp.gate_proj(xg)) \
        * mlp.up_proj(xg)
    rec9 = rec9[:, 1, :].detach().float() \
        .cpu().numpy().reshape(-1)
    a9_rel = float(np.abs(rec9 - acta).max()
                   / max(float(np.abs(acta).max()), 1e-30))
    a9_ok = bool(a9_rel < 1e-6)
    log('a9 act hook vs recompute rel %.2e ok=%s'
        % (a9_rel, a9_ok), lines)

    # a10: k=1 ablation slice exactly zero
    _, act_abl, _ = forward1(a2_seq, abl_idx=np.array([N2]),
                             factor=1.0)
    a10_zero = float(abs(act_abl[N2]))
    keep = np.setdiff1d(np.arange(9728),
                        np.array([N2]))
    a10_rest = float(np.abs(act_abl[keep] - acta[keep]).max())
    a10_ok = bool(a10_zero < 1e-30 and a10_rest < 1e-9)
    log('a10 k=1 slice zero %.2e rest %.2e ok=%s'
        % (a10_zero, a10_rest, a10_ok), lines)

    anchor_ok = all([a1_ok, a2_ok, a3_ok, a4_ok, a5_ok,
                     a5b_ok, a6_ok, a7_ok, a8a_ok, a8b_ok,
                     a8c_ok, a9_ok, a10_ok])

    # non-degeneracy gates
    act12_std = float(act12_74.std())
    sep_base = float(np.median(finP_cls[cls == 1])
                     - np.median(finP_cls[cls == 0]))
    gates_ok = bool(act12_std > 0 and abs(sep_base) > 0)

    # ---------- pre-init all result containers ----------
    verdict = None
    T1 = T2 = T3 = T4 = None
    p_t2 = float('nan')
    p_dose = float('nan')
    dsep_real = float('nan')
    rho_dose = float('nan')

    if anchor_ok and gates_ok:
        # ---------- T1 scan (descriptive) ----------
        a2_74 = act12_74[:, N2]
        lab01 = cls.astype(np.float64)
        ac = a2_74 - a2_74.mean()
        den = float(np.sqrt((ac ** 2).sum()
                            * ((lab01
                                - lab01.mean()) ** 2).sum()))
        r_cls = float((ac * (lab01
                             - lab01.mean())).sum() / den) \
            if den > 1e-30 else 0.0
        prng = np.random.default_rng([RNG_MAIN, 1])
        perm_r = np.zeros(N_PERM)
        for pi in range(N_PERM):
            pl = prng.permutation(cls)
            l01 = pl.astype(np.float64)
            dd = float(np.sqrt((ac ** 2).sum()
                               * ((l01
                                   - l01.mean()) ** 2).sum()))
            perm_r[pi] = float((ac * (l01
                                      - l01.mean())).sum()
                               / dd) if dd > 1e-30 else 0.0
        p_rcls = float((np.abs(perm_r)
                        >= abs(r_cls) - 1e-30).sum()) \
            / N_PERM
        order74 = np.argsort(a2_74)
        top_words = [(cells77[i], round(float(a2_74[i]), 3))
                     for i in order74[::-1][:6]]
        tail_words = [(cells77[i], round(float(a2_74[i]), 3))
                      for i in order74[:6]]
        # 30-word extension
        act30 = np.zeros(30)
        for i, w in enumerate(words30):
            seq = [func_tid, int(tids30[i])]
            _, act_i, _ = forward1(seq)
            act30[i] = float(act_i[N2])
        r30 = spearman(act30, (lab30 == 1)
                       .astype(np.float64))
        T1 = {'r_cls_74': r_cls, 'p_rcls_perm': p_rcls,
              'top_words': top_words,
              'tail_words': tail_words,
              'act30_mean_F': float(act30[lab30 == 0].mean()),
              'act30_mean_C': float(act30[lab30 == 1].mean()),
              'spearman_act30_vs_cls': r30,
              'act30_min': float(act30.min()),
              'act30_max': float(act30.max())}
        log('T1 r_cls=%.3f p=%.4f; 30w F=%.3f C=%.3f'
            % (r_cls, p_rcls, T1['act30_mean_F'],
               T1['act30_mean_C']), lines)

        # ---------- T2 causal ----------
        rng3 = np.random.default_rng([RNG_MAIN, 3])
        rand_idx = rng3.choice(9728, N_RAND, replace=False)

        def sep74(factor, idx):
            P = np.zeros(n74)
            for i, (_, _, w) in enumerate(cells):
                seq = [func_tid, tid_map[w]]
                fin, _, _ = forward1(seq, abl_idx=idx,
                                     factor=factor)
                P[i] = float(
                    (fin.reshape(-1)) @ d79_c)
            return float(np.median(P[cls == 1])
                         - np.median(P[cls == 0])), P

        sep_abl, P_abl74 = sep74(1.0, np.array([N2]))
        dsep_real = sep_abl - sep_base
        dsep_rand = np.zeros(N_RAND)
        headC_rand = np.zeros((N_RAND, 30, NH))
        for r in range(N_RAND):
            idx_r = np.array([int(rand_idx[r])])
            s_r, _ = sep74(1.0, idx_r)
            dsep_rand[r] = s_r - sep_base
            for i, w in enumerate(words30):
                seq = [func_tid, int(tids30[i])]
                _, _, ops = forward1(seq, abl_idx=idx_r,
                                     factor=1.0)
                x34 = ops[L34].reshape(-1)
                for h in range(NH):
                    headC_rand[r, i, h] = float(np.dot(
                        C34[h * HD:(h + 1) * HD],
                        x34[h * HD:(h + 1) * HD]))
            if (r + 1) % 20 == 0:
                log('T2/T3 rand [%d/%d]' % (r + 1, N_RAND),
                    lines)
        spread_rand = float(dsep_rand.max()
                            - dsep_rand.min())
        p_t2 = float((np.abs(dsep_rand)
                      >= abs(dsep_real) - 1e-30).sum() + 1) \
            / (N_RAND + 1)

        # dose curve: f grid [0, .25, .5, .75, 1]
        # effect(0) = 0 by definition; effect(1) = |dsep_real|
        eff = [0.0]
        for f in DOSE:
            s_f, _ = sep74(f, np.array([N2]))
            eff.append(abs(sep_base - s_f))
        eff.append(abs(dsep_real))
        fs = np.array([0.0] + DOSE + [1.0])
        assert len(fs) == 5 and len(eff) == 5, \
            'dose grid mismatch'
        ev = np.array(eff)
        rho_dose = spearman(fs, ev)
        nperms = 0
        ge = 0
        import itertools
        for perm in itertools.permutations(range(5)):
            nperms += 1
            rp = spearman(fs, ev[np.array(perm)])
            if rp >= rho_dose - 1e-30:
                ge += 1
        p_dose = float(ge) / nperms
        T2 = {'dsep_real': dsep_real,
              'dsep_rand_median':
                  float(np.median(dsep_rand)),
              'dsep_rand_spread': spread_rand,
              'p_rank': p_t2,
              'dose_f': fs.tolist(),
              'dose_effect_abs': ev.tolist(),
              'rho_dose': rho_dose,
              'p_dose': p_dose}
        log('T2 dsep_real %.4f p=%.4f; dose rho=%.3f '
            'p=%.4f' % (dsep_real, p_t2, rho_dose,
                        p_dose), lines)

        # ---------- T3 cross-granularity ----------
        def gap30(headC):
            return np.array([
                float(np.median(headC[lab30 == 0, h])
                      - np.median(headC[lab30 == 1, h]))
                for h in range(NH)])

        headC_abl30 = np.zeros((30, NH))
        for i, w in enumerate(words30):
            seq = [func_tid, int(tids30[i])]
            _, _, ops = forward1(seq, abl_idx=np.array([N2]),
                                 factor=1.0)
            x34 = ops[L34].reshape(-1)
            for h in range(NH):
                headC_abl30[i, h] = float(np.dot(
                    C34[h * HD:(h + 1) * HD],
                    x34[h * HD:(h + 1) * HD]))
        gap_base30 = gap30(headC30)
        gap_abl30 = gap30(headC_abl30)
        dgap = gap_abl30 - gap_base30
        dgap_null = np.zeros((N_RAND, NH))
        for r in range(N_RAND):
            dgap_null[r] = gap30(headC_rand[r]) \
                - gap_base30
        p_h15 = float((np.abs(dgap_null[:, 15])
                       >= abs(dgap[15]) - 1e-30).sum()
                      + 1) / (N_RAND + 1)
        T3 = {'gap_base_h15': float(gap_base30[15]),
              'gap_abl_h15': float(gap_abl30[15]),
              'dgap_h15': float(dgap[15]),
              'p_h15': p_h15,
              'dgap_absmax_head':
                  int(np.argmax(np.abs(dgap))),
              'dgap_absmax': float(np.abs(dgap).max()),
              'null_h15_p95': float(np.quantile(
                  np.abs(dgap_null[:, 15]), 0.95))}
        log('T3 dgap_h15 %.4f p=%.4f absmax head %d'
            % (dgap[15], p_h15, T3['dgap_absmax_head']),
            lines)

        # ---------- T4 geometry ----------
        coln = np.linalg.norm(Wd12, axis=1)
        cos_all = np.abs(c_cls12) / np.maximum(coln,
                                               1e-30)
        cos_n2 = float(cos_all[N2])
        pct = float((cos_all < cos_n2).sum()) / 9728.0
        T4 = {'cos_down_ucls': cos_n2,
              'percentile': pct,
              'c_cls_n2': float(c_cls12[N2]),
              'c_lang_n2': float(c_lang12[N2]),
              'col_norm': float(coln[N2]),
              'cos_lang_n2': float(
                  abs(c_lang12[N2]) / max(coln[N2],
                                          1e-30))}
        log('T4 cos=%.4f pct=%.4f' % (cos_n2, pct), lines)

        # ---------- verdict ----------
        if p_t2 < P_GATE and p_dose < P_GATE:
            verdict = 'neuron2_single_causal_monotone'
        elif p_t2 < P_GATE:
            verdict = 'neuron2_single_causal_nonmonotone'
        else:
            verdict = 'neuron2_not_single_causal'
    elif not gates_ok:
        verdict = 'degenerate_void'
        log('gates failed: act_std=%.4g sep_base=%.4g'
            % (act12_std, sep_base), lines)

    if verdict is None:
        verdict = 'anchor_fail_all_void'
        log('anchor_fail: a1=%s a2=%s a3=%s a4=%s a5=%s '
            'a5b=%s a6=%s a7=%s a8=%s/%s/%s a9=%s a10=%s'
            % (a1_ok, a2_ok, a3_ok, a4_ok, a5_ok, a5b_ok,
               a6_ok, a7_ok, a8a_ok, a8b_ok, a8c_ok,
               a9_ok, a10_ok), lines)

    elapsed = round(time.monotonic() - t0, 1)
    log('verdict=%s elapsed=%ss' % (verdict, elapsed), lines)

    # ---------- persist ----------
    np.savez_compressed(
        os.path.join(OUT, 'neuron2_deep_dive.npz'),
        act12_74=act12_74, act2_74=act12_74[:, N2],
        P_cls_base=finP_cls, P_cls_abl=P_abl74
        if anchor_ok and gates_ok else np.zeros(n74),
        headC30=headC30, act30=act30
        if anchor_ok and gates_ok else np.zeros(30),
        dsep_rand=dsep_rand if anchor_ok
        and gates_ok else np.zeros(N_RAND),
        headC_rand=headC_rand if anchor_ok
        and gates_ok else np.zeros((N_RAND, 30, NH)),
        gap_base30=gap_base30 if anchor_ok
        and gates_ok else np.zeros(NH),
        dgap=dgap if anchor_ok and gates_ok
        else np.zeros(NH),
        words=np.array(cells77), words30=np.array(words30),
        lab30=lab30, cls=cls, lang=lang)

    result = {
        'phase': 2990, 'name': 'neuron2_deep_dive',
        'final_verdict': verdict, 'elapsed_s': elapsed,
        'anchor_all_ok': bool(anchor_ok and gates_ok),
        'anchors': {
            'a1_diff': a1_diff, 'a2_rel': a2_rel,
            'a3_bit': a3_diff, 'a4_bit': a4_diff,
            'a5_rel': a5_rel, 'a5_argmax': a5_arg,
            'a5b_rel': a5b_rel, 'a6_rel': a6_rel,
            'a7_bit': a7_diff, 'a8a': a8a_ok,
            'a8b': a8b_ok, 'a8c': a8c_ok,
            'a9_rel': a9_rel, 'a10_zero': a10_zero,
            'a10_rest': a10_rest},
        'gates': {'act12_std': act12_std,
                  'sep_base': sep_base,
                  'spread_rand': spread_rand
                  if anchor_ok and gates_ok else None},
        'T1': T1, 'T2': T2, 'T3': T3, 'T4': T4,
        'p_t2': p_t2, 'p_dose': p_dose,
        'neuron': {'layer': L12, 'index': N2},
        'correction_note':
            'run2 a7 anchor fail (bit 15.9): missing '
            'condition-dim index on 2987 npz comparison '
            '(prof_all/BL are (5,74,36)/(5,74), L2 '
            'condition = [0] as in 2989 a8); fixed and '
            'rerun -> run3 authoritative; run3 also '
            'crashed: same fin-reshape bug in sep74 '
            'closure (second float(fin@d) site), fixed '
            '-> run4 authoritative',
        'sources_sha8': {'s2989': sha8(SRC_2989),
                         's2964': sha8(SRC_2964),
                         's2979': sha8(SRC_2979),
                         's2987': sha8(SRC_2987)},
        'npz_sha256_8': sha8(os.path.join(
            OUT, 'neuron2_deep_dive.npz')),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    log('PHASE2990 DONE', lines)


if __name__ == '__main__':
    main()
