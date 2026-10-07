# -*- coding: utf-8 -*-
"""Phase 2986: Omega-C opening -- length-context drift atlas.

Plan v3 Omega-C (research/gpt5/docs/plan_v3_omega_dynamic_manifold.md):
length bins {2,16,64,256,1024}; readout trio = s_c drift (2945 rule,
coarse grid), A11 gain (2953 convention), B band diff (2973 convention);
per-condition independent batch (2934 discipline); three separate
preregistered tests: existence / monotonicity / saturation.

Protocol core (verbatim from 2979): target word at final position,
[func_tid, word_tid] tail; L=2 bin IS the 2979 protocol (no filler),
so anchors vs 2973/2979 apply bit-level. Injection at L17 self_attn
input, target position, relative dose s*n17_w(L)*d_lang_u.

Preregistered verdict map (T1/T2/T3 on primary L34 readout drift):
  T1 fail                    -> drift_below_effect_scale
  T1 pass, T2 fail           -> drift_present_nonmonotone
  T1 pass, T2 pass, T3 pass  -> drift_monotone_saturated
  T1 pass, T2 pass, T3 fail  -> drift_monotone_unsaturated
All four branches reachable (T1 gate = 5% of word spread, reachable
both ways; T2 Spearman>=0.8 with exact 120-perm p; T3 increment ratio).
"""
import json
import os
import time
import hashlib
import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2939 = os.path.join(BASE, 'phase2939', 'rotation_target',
                        'rotation_target.npz')
SRC_2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                        'probe_relativity.npz')
SRC_2945 = os.path.join(BASE, 'phase2945', 'threshold_curves',
                        'threshold_curves.npz')
SRC_2973 = os.path.join(BASE, 'phase2973', 'fr_scale_audit',
                        'fr_scale_audit.npz')
SRC_2977_NPZ = os.path.join(BASE, 'phase2977',
                            'two_axis_fusion_injection',
                            'two_axis_fusion_injection.npz')
EXEC_2977 = os.path.join(BASE, 'phase2977',
                         'two_axis_fusion_injection',
                         'execution.json')
SRC_2979 = os.path.join(BASE, 'phase2979', 'reversal_anatomy',
                        'reversal_anatomy.npz')
OUT = os.path.join(BASE, 'phase2986', 'length_context_drift')

MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
L_INJ = 17
L_PRIM = 34          # primary drift readout layer (L34 locus)
LENGTHS = [2, 16, 64, 256, 1024]
GRID_S = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)   # coarse (2953 style)
FILLER_TEXT = (' The sun rises in the east and sets in the '
               'west .')
N_PERM = 10000
RNG_MAIN = 2986
T1_GATE = 0.05       # drift >= 5% of word spread
T2_RHO = 0.8
P_TH = 0.01
T3_RATIO = 0.5
T5_GRID_HALF = 0.125  # half of min positive grid step


def sha8(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]


def log(msg, lines):
    lines.append('[%s] %s' % (time.strftime('%H:%M:%S'), msg))
    with open(os.path.join(OUT, 'run_log.txt'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def median(a):
    return float(np.median(a))


def spearman(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom <= 0:
        return 0.0
    return float((rx * ry).sum() / denom)


def band_of(prof):
    return (float(prof[6:13].mean())
            - float(prof[28:36].mean()))


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)

    # ---------- cells ----------
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
    words77 = ['%s:%s:%s' % c for c in cells]
    n_test = len(cells)
    lang = np.array([0 if c[1] == 'en' else 1 for c in cells])
    cls = np.array([0 if c[0] == 'F' else 1 for c in cells])

    prereg = {
        'design': 'length bins {2,16,64,256,1024}; filler = '
                  'tiled tokens of %r truncated to L-2, target '
                  'tail [func_tid, word_tid]; L=2 bin = 2979 '
                  'protocol verbatim (no filler)' % FILLER_TEXT,
        'injection': 'L17 self_attn input at TARGET position, '
                     'relative dose s*n17_w(L)*d_lang_u, '
                     'grid s=%s (coarse, 2953 style)' % (list(GRID_S),),
        'batch': 'per-condition independent single-sample '
                 'forwards (2934 discipline)',
        'primary': 'D34(L) = median_i |prof34_i(L) - '
                   'prof34_i(L2)|, prof34 = u35@Wo34 readout of '
                   'o_proj input at target pos (L34 locus)',
        'T1_existence': 'D34(1024) >= %.2f * spread34(2), '
                        'spread34 = median_i |prof34_i - '
                        'median_j prof34_j| at L=2' % T1_GATE,
        'T2_monotonic': 'Spearman rho(log2 L, D34_k) >= %.1f '
                        'AND exact length-label permutation '
                        'p(120, one-sided) < %.2f' % (T2_RHO, P_TH),
        'T3_saturation': 'inc4 = D(1024)-D(256) < %.1f * '
                         'max(inc1,inc2,inc3) on bins '
                         '[16,64,256,1024] rel D(2)=0' % T3_RATIO,
        'T4_B_drift': 'secondary: median_i |B_i(1024)-B_i(2)| '
                      '>= %.2f * spread_B(2)' % T1_GATE,
        'T5_sc_drift': '|s_c(1024)-s_c(2)| >= %.3f (one coarse '
                       'grid half-step); else '
                       's_c_within_grid; s_c None -> '
                       's_c_not_identifiable' % T5_GRID_HALF,
        'T6_cls_signature': 'secondary: sig34(L)=median_C-'
                            'median_F at L34 readout per length, '
                            'two-sided label permutation '
                            'p<%.2f, family=5 reported raw'
                            % P_TH,
        'verdict_map': ['drift_below_effect_scale (T1 fail)',
                        'drift_present_nonmonotone (T1 pass,'
                        ' T2 fail)',
                        'drift_monotone_saturated (T1,T2,T3 '
                        'pass)',
                        'drift_monotone_unsaturated (T1,T2 '
                        'pass, T3 fail)'],
        'anchors': {
            'a1': 'Vt8 rebuild < 1e-6',
            'a2': 'determinism L=2 and L=1024 rel < 1e-4',
            'a3': 'L=2 norms/coss vs 2973: rel<1e-4, '
                  'cos<1e-6',
            'a4': 'single-token 74/74',
            'a5': 'd_lang_u/d_cls_u rebuild vs 2979 < 1e-9',
            'a6': 'n17 rebuild vs 2979 rel < 1e-9',
            'a7': 'REMOVED by correction (run3): 2945 s_c '
                  'uses sep_med<100 rule over a 57-word set '
                  'with xdir vector; not comparable to this '
                  'protocol (74 words, 0.5xmax crossing, '
                  'relative lang-axis dose) - cross-product '
                  'anchor ill-posed',
            'a8': 'B_rec(L=2) vs 2973 B_rec rel < 1e-4'},
    }

    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2986,
                   'name': 'length_context_drift',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2939': sha8(SRC_2939),
                               's2927': sha8(SRC_2927),
                               's2945': sha8(SRC_2945),
                               's2973': sha8(SRC_2973),
                               's2977npz': sha8(SRC_2977_NPZ),
                               's2977exec': sha8(EXEC_2977),
                               's2979npz': sha8(SRC_2979)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'l_inj': L_INJ, 'l_primary': L_PRIM,
                   'lengths': LENGTHS, 'grid_s': list(GRID_S),
                   'n_perm': N_PERM, 'rng': RNG_MAIN,
                   't1_gate': T1_GATE, 't2_rho': T2_RHO,
                   'p_threshold': P_TH, 't3_ratio': T3_RATIO,
                   't5_grid_half': T5_GRID_HALF,
                   'filler_text': FILLER_TEXT,
                   'cells': {'F_en': F_EN, 'F_fr': F_FR,
                             'C_en': C_EN, 'C_fr': C_FR},
                   'prereg': prereg,
                   'correction_note':
                       'run1: SRC_2977 path wrong (crashed '
                       'before freeze); run2: phantom edit '
                       'left .median(axis=1) line (crashed '
                       'after freeze, artifacts deleted); '
                       'run3: anchor a7 ill-posed - 2945 s_c '
                       'uses sep_med<100 rule over 57-word '
                       'set with xdir vector, incomparable '
                       'to this protocol, a7 removed and '
                       'correction registered; run4 = '
                       'authoritative',},
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
    n17_79 = z79['n17'].astype(np.float64)
    words79 = [str(w) for w in z79['words']]

    # ---------- model ----------
    import sys
    sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
    from phase2662_symmetric_mapping_contract import \
        load_native
    from transformers import AutoTokenizer
    import torch

    tok = AutoTokenizer.from_pretrained(
        MD, local_files_only=True, trust_remote_code=True,
        use_fast=True)
    assert words77 == words73 == words79, 'word order drift'
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
    a4_ok = bool(n_single == n_test)
    log('a4 single-token %d/%d ok=%s'
        % (n_single, n_test, a4_ok), lines)
    ids_the = tok(' the', add_special_tokens=False)[
        'input_ids']
    assert len(ids_the) == 1
    func_tid = int(ids_the[0])
    fill_pool = tok(FILLER_TEXT, add_special_tokens=False)[
        'input_ids']
    assert len(fill_pool) >= 1

    def make_seq(w, L):
        fill = [int(fill_pool[k % len(fill_pool)])
                for k in range(L - 2)]
        seq = fill + [func_tid, tid_map[w]]
        assert len(seq) == L, 'seq length drift'
        return seq

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    log('model loaded', lines)

    cap_op = {li: [] for li in range(NL)}
    cap_x17 = []
    inj_state = {'dirs': None}
    handles = []

    def hs_of(args, kwargs):
        if args and args[0] is not None:
            return args[0]
        return kwargs.get('hidden_states')

    def hook_inj(module, args, kwargs):
        d = inj_state['dirs']
        if d is not None:
            x = hs_of(args, kwargs)
            if x is not None:
                dt = torch.as_tensor(
                    d, device=x.device, dtype=x.dtype)
                x[:, -1, :] += dt
        return None

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

    handles.append(
        layers[L_INJ].self_attn
        .register_forward_pre_hook(
            hook_inj, with_kwargs=True))
    handles.append(
        layers[L_INJ].self_attn
        .register_forward_pre_hook(
            hook_x17, with_kwargs=True))
    for li in range(NL):
        handles.append(
            layers[li].self_attn.o_proj
            .register_forward_pre_hook(
                hook_op(li), with_kwargs=True))

    def clear_cap():
        for li in cap_op:
            del cap_op[li][:]
        del cap_x17[:]

    def forward1(toks, dirs=None):
        clear_cap()
        inj_state['dirs'] = dirs
        with torch.no_grad():
            model(torch.tensor([toks], device='cuda'))
        inj_state['dirs'] = None
        return {li: cap_op[li][0].astype(np.float64)
                for li in range(NL)}

    M = np.zeros((NL, NH * HD))
    Mnorm = np.zeros(NL)
    for li in range(NL):
        Wo = layers[li].self_attn.o_proj.weight.detach() \
            .float().cpu().numpy()
        M[li] = u35 @ Wo
        Mnorm[li] = float(np.linalg.norm(M[li]))

    # ---------- per-length base sweep ----------
    n_len = len(LENGTHS)
    prof_all = np.zeros((n_len, n_test, NL))
    normsL = np.zeros((n_len, n_test, NL))
    cossL = np.zeros((n_len, n_test, NL))
    BL = np.zeros((n_len, n_test))
    n17L = np.zeros((n_len, n_test))
    X17_by_len = {}
    filler_lens = []
    for bi, L in enumerate(LENGTHS):
        filler_lens.append(L - 2)
        X17L = np.zeros((n_test, 2560))
        for i, (_, _, w) in enumerate(cells):
            seq = make_seq(w, L)
            op = forward1(seq)
            X17L[i] = cap_x17[0].astype(
                np.float64).reshape(-1)
            for li in range(NL):
                x = op[li].reshape(-1)
                normsL[bi, i, li] = \
                    float(np.linalg.norm(x))
                pr = float(np.dot(x, M[li]))
                prof_all[bi, i, li] = pr
                cossL[bi, i, li] = pr / max(
                    normsL[bi, i, li] * Mnorm[li], 1e-30)
            BL[bi, i] = band_of(prof_all[bi, i])
            n17L[bi, i] = float(
                np.linalg.norm(X17L[i]))
        X17_by_len[L] = X17L
        log('base sweep L=%d done (filler %d tok)'
            % (L, L - 2), lines)

    # ---------- a2 determinism ----------
    seq2 = make_seq('man', 2)
    op_a = forward1(seq2)
    op_b = forward1(seq2)
    a2a_rel = float(np.abs(op_a[30] - op_b[30]).max()
                    / max(float(np.abs(op_a[30]).max()),
                          1e-30))
    seqL = make_seq('man', 1024)
    op_c = forward1(seqL)
    op_d = forward1(seqL)
    a2b_rel = float(np.abs(op_c[30] - op_d[30]).max()
                    / max(float(np.abs(op_c[30]).max()),
                          1e-30))
    a2_ok = bool(a2a_rel < 1e-4 and a2b_rel < 1e-4)
    log('a2 determinism L2 rel %.2e L1024 rel %.2e ok=%s'
        % (a2a_rel, a2b_rel, a2_ok), lines)

    # ---------- a3 identity vs 2973 (L=2 bin) ----------
    nrm_rel = float(np.max(np.abs(normsL[0] - norms73)
                           / np.maximum(norms73, 1e-30)))
    cos_abs = float(np.max(np.abs(cossL[0] - coss73)))
    a3_ok = bool(nrm_rel < 1e-4 and cos_abs < 1e-6)
    log('a3 identity vs 2973 (L=2): norms rel %.2e '
        'coss abs %.2e ok=%s'
        % (nrm_rel, cos_abs, a3_ok), lines)

    # ---------- a5/a6 axis + n17 rebuild at L=2 ----------
    X17_2 = X17_by_len[2]
    d_lang2 = X17_2[lang == 1].mean(axis=0) \
        - X17_2[lang == 0].mean(axis=0)
    d_cls2 = X17_2[cls == 1].mean(axis=0) \
        - X17_2[cls == 0].mean(axis=0)
    d_lang_u2 = d_lang2 / float(np.linalg.norm(d_lang2))
    d_cls_u2 = d_cls2 / float(np.linalg.norm(d_cls2))
    a5_diff = float(max(
        np.abs(d_lang_u2 - d79_l).max(),
        np.abs(d_cls_u2 - d79_c).max()))
    a5_ok = bool(a5_diff < 1e-9)
    log('a5 axis dirs vs 2979: max|d| %.2e ok=%s'
        % (a5_diff, a5_ok), lines)
    n17_rel = float(np.max(np.abs(n17L[0] - n17_79)
                           / np.maximum(n17_79, 1e-30)))
    a6_ok = bool(n17_rel < 1e-9)
    log('a6 n17 vs 2979: rel %.2e ok=%s'
        % (n17_rel, a6_ok), lines)

    # ---------- a8 B identity vs 2973 (L=2) ----------
    a8_rel = float(np.max(np.abs(BL[0] - B73)
                          / np.maximum(np.abs(B73), 1e-30)))
    a8_ok = bool(a8_rel < 1e-4)
    log('a8 B_rec vs 2973: max rel %.2e ok=%s'
        % (a8_rel, a8_ok), lines)

    # ---------- axis drift per length ----------
    axis_names = []
    axis_cos = np.zeros((n_len, 2))
    axis_nrel = np.zeros((n_len, 2))
    n2l = float(np.linalg.norm(d_lang2))
    n2c = float(np.linalg.norm(d_cls2))
    for bi, L in enumerate(LENGTHS):
        X = X17_by_len[L]
        dl = X[lang == 1].mean(axis=0) \
            - X[lang == 0].mean(axis=0)
        dc = X[cls == 1].mean(axis=0) \
            - X[cls == 0].mean(axis=0)
        axis_cos[bi, 0] = float(
            (dl / max(float(np.linalg.norm(dl)), 1e-30))
            @ d_lang_u2)
        axis_cos[bi, 1] = float(
            (dc / max(float(np.linalg.norm(dc)), 1e-30))
            @ d_cls_u2)
        axis_nrel[bi, 0] = float(
            np.linalg.norm(dl) / max(n2l, 1e-30))
        axis_nrel[bi, 1] = float(
            np.linalg.norm(dc) / max(n2c, 1e-30))
        axis_names = ['lang', 'cls']
    log('axis drift cos lang %s cls %s'
        % ([round(float(v), 4) for v in axis_cos[:, 0]],
           [round(float(v), 4) for v in axis_cos[:, 1]]),
        lines)

    # ---------- grid sweep: s_c + A11 ----------
    sep_curves = np.zeros((n_len, len(GRID_S) + 1))
    s_c_arr = []
    amp_arr = []
    for bi, L in enumerate(LENGTHS):
        n17w = n17L[bi]
        sep0 = abs(median(prof_all[bi, lang == 1, L_INJ])
                   - median(prof_all[bi, lang == 0,
                                     L_INJ]))
        sep_curves[bi, 0] = sep0
        for gi, s in enumerate(GRID_S):
            prof17 = np.zeros(n_test)
            for i, (_, _, w) in enumerate(cells):
                seq = make_seq(w, L)
                dv = (float(s) * float(n17w[i])) * d_lang_u2
                op = forward1(seq, dv)
                prof17[i] = float(
                    np.dot(op[L_INJ].reshape(-1),
                           M[L_INJ]))
            sep_curves[bi, gi + 1] = abs(
                median(prof17[lang == 1])
                - median(prof17[lang == 0]))
        s_grid = np.array([0.0] + [float(s)
                                   for s in GRID_S])
        sep_cv = sep_curves[bi]
        smax = float(sep_cv.max())
        thr = 0.5 * smax
        sc = None
        if smax > 0 and sep_cv[0] >= thr:
            sc = 0.0
        elif smax > 0:
            for j in range(1, len(s_grid)):
                if sep_cv[j] >= thr:
                    s0, s1 = s_grid[j - 1], s_grid[j]
                    y0, y1 = sep_cv[j - 1], sep_cv[j]
                    w = (thr - y0) / max(y1 - y0, 1e-30)
                    sc = float(s0 + w * (s1 - s0))
                    break
        s_c_arr.append(sc)
        amp_arr.append(float(sep_curves[bi, 4]
                             / max(sep_curves[bi, 0],
                                   1e-30)))
        log('L=%d sep curve %s s_c=%s amp=%.3f'
            % (L, [round(float(v), 4) for v in sep_cv],
               None if sc is None else round(sc, 4),
               amp_arr[-1]), lines)

    sc_arr_f = [float('nan') if v is None else float(v)
                for v in s_c_arr]
    log('s_c(L=2)=%.4f (no cross-product anchor; see '
        'prereg a7 correction note)'
        % (float(s_c_arr[0])
           if s_c_arr[0] is not None else float('nan'),),
        lines)

    # ---------- primary drift + tests ----------
    prof34 = prof_all[:, :, L_PRIM]
    d34 = np.array([float(np.median(np.abs(
        prof34[bi] - prof34[0]))) for bi in range(n_len)])
    spread34 = float(np.median(np.abs(
        prof34[0] - median(prof34[0]))))
    dB = np.array([float(np.median(np.abs(
        BL[bi] - BL[0]))) for bi in range(n_len)])
    spreadB = float(np.median(np.abs(
        BL[0] - median(BL[0]))))
    log('D34 per length %s spread34=%.4f'
        % ([round(v, 4) for v in d34], spread34), lines)

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok and a6_ok and a8_ok)
    verdict = None
    T1 = T2 = T3 = T4 = T5 = T6 = None
    sig34 = np.zeros(n_len)
    p6 = np.zeros(n_len)

    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
        log('ANCHOR FAIL -> all void', lines)
    else:
        # non-degeneracy gates (registered)
        assert spread34 > 0, 'degenerate spread34'
        assert spreadB > 0, 'degenerate spreadB'

        # T1 existence
        t1_val = float(d34[-1])
        t1_gate = T1_GATE * spread34
        t1_ok = bool(t1_val >= t1_gate)
        T1 = {'D34_1024': round(t1_val, 6),
              'gate': round(t1_gate, 6),
              'spread34_L2': round(spread34, 6),
              'ok': t1_ok}
        log('T1 existence: D34(1024)=%.4f gate=%.4f ok=%s'
            % (t1_val, t1_gate, t1_ok), lines)

        # T2 monotonicity (exact 120 label perms)
        lx = np.log2(np.array(LENGTHS,
                              dtype=np.float64))
        rho_obs = spearman(lx, d34)
        rng = np.random.default_rng(RNG_MAIN)
        cnt = 0
        perms = list(__import__('itertools').permutations(
            range(n_len)))
        for pm in perms:
            dv = np.array([d34[j] for j in pm])
            if spearman(lx, dv) >= rho_obs:
                cnt += 1
        p2 = float(cnt) / float(len(perms))
        t2_ok = bool(rho_obs >= T2_RHO and p2 < P_TH)
        T2 = {'rho': round(rho_obs, 4),
              'p_exact': round(p2, 4),
              'n_perms': len(perms), 'ok': t2_ok}
        log('T2 monotonicity: rho=%.4f p=%.4f ok=%s'
            % (rho_obs, p2, t2_ok), lines)

        # T3 saturation
        incs = [float(d34[k] - d34[k - 1])
                for k in range(1, n_len)]
        mx = max(incs[:3])
        t3_ok = bool(incs[3] < T3_RATIO * mx)
        T3 = {'inc_16_64_256_1024':
              [round(v, 6) for v in incs],
              'ratio_inc4_over_max': round(
                  incs[3] / max(mx, 1e-30), 4),
              'ok_saturated': t3_ok}
        log('T3 saturation: incs %s ratio %.4f '
            'saturated=%s'
            % ([round(v, 4) for v in incs],
               incs[3] / max(mx, 1e-30), t3_ok), lines)

        # verdict (frozen map)
        if not t1_ok:
            verdict = 'drift_below_effect_scale'
        elif not t2_ok:
            peak = int(np.argmax(d34))
            verdict = 'drift_present_nonmonotone'
            T2['peak_length'] = LENGTHS[peak]
        elif t3_ok:
            verdict = 'drift_monotone_saturated'
        else:
            verdict = 'drift_monotone_unsaturated'
        log('verdict: %s' % verdict, lines)

        # T4 B drift (secondary)
        t4_val = float(dB[-1])
        t4_gate = T1_GATE * spreadB
        t4_ok = bool(t4_val >= t4_gate)
        T4 = {'dB_1024': round(t4_val, 6),
              'gate': round(t4_gate, 6),
              'spreadB_L2': round(spreadB, 6),
              'ok': t4_ok}
        log('T4 B drift: dB(1024)=%.4f gate=%.4f ok=%s'
            % (t4_val, t4_gate, t4_ok), lines)

        # T5 s_c drift
        if any(v is None for v in s_c_arr):
            T5 = {'verdict': 's_c_not_identifiable',
                  's_c': sc_arr_f}
        else:
            dsc = abs(float(s_c_arr[-1])
                      - float(s_c_arr[0]))
            t5v = ('s_c_drift' if dsc >= T5_GRID_HALF
                   else 's_c_within_grid')
            T5 = {'verdict': t5v,
                  'delta': round(dsc, 4),
                  's_c': [round(v, 4) for v in sc_arr_f],
                  'amp': [round(v, 4) for v in amp_arr]}
        log('T5 s_c: %s' % json.dumps(T5), lines)

        # T6 cls signature per length (secondary)
        sig34 = np.zeros(n_len)
        p6 = np.zeros(n_len)
        rng6 = np.random.default_rng(RNG_MAIN + 1)
        for bi in range(n_len):
            v = prof34[bi]
            obs = median(v[cls == 1]) - median(v[cls == 0])
            cnt6 = 0
            for _ in range(N_PERM):
                pl = rng6.permutation(cls)
                pv = median(v[pl == 1]) \
                    - median(v[pl == 0])
                if abs(pv) >= abs(obs):
                    cnt6 += 1
            sig34[bi] = obs
            p6[bi] = float(cnt6) / float(N_PERM)
        T6 = {'sig34': [round(float(v), 4)
                        for v in sig34],
              'p': [round(float(v), 4) for v in p6],
              'note': 'two-sided label permutation, '
                      'family=5 raw (secondary)'}
        log('T6 sig34 %s p %s'
            % ([round(float(v), 4) for v in sig34],
               [round(float(v), 4) for v in p6]), lines)

    # ---------- per-layer drift profile (descriptive) ----------
    D_layer = np.zeros((n_len, NL))
    for bi in range(n_len):
        for li in range(NL):
            D_layer[bi, li] = float(np.median(np.abs(
                prof_all[bi, :, li]
                - prof_all[0, :, li])))

    elapsed = round(time.monotonic() - t0, 1)
    result = {
        'phase': 2986, 'name': 'length_context_drift',
        'final_verdict': verdict,
        'anchor_all_ok': anchor_ok,
        'anchors': {
            'a1_Vt8_diff': a1_diff, 'a1_ok': a1_ok,
            'a2a_rel': a2a_rel, 'a2b_rel': a2b_rel,
            'a2_ok': a2_ok,
            'a3_norm_rel': nrm_rel, 'a3_cos_abs': cos_abs,
            'a3_ok': a3_ok, 'a4_ok': a4_ok,
            'a5_diff': a5_diff, 'a5_ok': a5_ok,
            'a6_n17_rel': n17_rel, 'a6_ok': a6_ok,
            'a7_removed': 'ill-posed cross-product anchor '
                          '(2945 construction differs); see '
                          'execution.json correction_note',
            'a8_B_rel': a8_rel, 'a8_ok': a8_ok},
        'T1': T1, 'T2': T2, 'T3': T3,
        'T4': T4, 'T5': T5, 'T6': T6,
        'D34_per_length': [round(float(v), 6)
                           for v in d34],
        'D34_spread_L2': round(spread34, 6),
        'lengths': LENGTHS,
        'filler_lens': filler_lens,
        'axis_cos': {'lang': [round(float(v), 6)
                              for v in axis_cos[:, 0]],
                     'cls': [round(float(v), 6)
                             for v in axis_cos[:, 1]]},
        'axis_norm_rel': {
            'lang': [round(float(v), 6)
                     for v in axis_nrel[:, 0]],
            'cls': [round(float(v), 6)
                    for v in axis_nrel[:, 1]]},
        'n17_median': [round(float(np.median(n17L[bi])), 4)
                       for bi in range(n_len)],
        'B_median': [round(float(np.median(BL[bi])), 4)
                     for bi in range(n_len)],
        'elapsed_s': elapsed}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    np.savez(os.path.join(
        OUT, 'length_context_drift.npz'),
        prof34=prof34, prof_all=prof_all,
        norms=normsL, coss=cossL, BL=BL, n17L=n17L,
        D_layer=D_layer, d34=d34, dB=dB,
        sep_curves=sep_curves,
        s_c=np.array(sc_arr_f),
        amp=np.array(amp_arr),
        axis_cos=axis_cos, axis_nrel=axis_nrel,
        sig34=sig34, p6=p6,
        words=np.array(words77),
        lengths=np.array(LENGTHS),
        grid_s=np.array([0.0] + [float(s)
                                 for s in GRID_S]))
    log('saved npz+result.json elapsed %.1fs' % elapsed,
        lines)
    print('PHASE2986 DONE verdict=%s elapsed=%s'
          % (verdict, elapsed))


if __name__ == '__main__':
    main()
