"""Phase 2992: sparse dictionary registration (plan v4 P2).

Question.  The chain has neuron-level (2989) and head-level
(2987/2991) registries, but no feature-level registration.
Full SAE training is deferred; plan v4 P2 asks the GATE
question first: does a lightweight overcomplete top-k sparse
dictionary on the L6-12 load-bearing band align with the
readout axes (c_lang / c_cls) BEYOND a random-rotation null
(2931 caliber), and do aligned atoms carry causal weight
consistent with their snapshot contribution (2950 dual-
caliber discipline)?

Design.
  Base      rerun 74 cells x {L2, L16N} forwards with the
            2991 op/x17/down_proj hook set; bit-level anchors
            vs 2987 headC and 2989 MLP acts.
  T1 dict   per layer li in 6..12: stack 148 samples (74x2),
            row-normalize, seeded k-means (k=96, n_init=3),
            atoms unit-normalized; OMP sparse coding (k=16);
            reconstruction residual (nondegeneracy gate).
  T2 align  per layer x {lang, cls}: max_j |atom_j . chat|
            vs 200 Haar random-direction nulls (2931
            caliber); p_raw -> Bonferroni x14 family (7
            layers x 2 axes).  PRIMARY gate.
  T3 dual   12 stratified cells (L2 cond, lang axis, m=8):
            snapshot direct-path prediction pred_axis =
            dh . c_lang[li] (residual identity path,
            analytic) vs REALIZED downstream effect:
            (res34_mod - res34_base) . d_lang_u measured
            at the L34 decoder-layer input residual
            (2560 space, same caliber as d_lang_u);
            random-atom control (m=8); sign agreement +
            magnitude ratio (rebalancing accounting).

Verdict branches (frozen, primary = T2 family):
  anchor_fail_all_void
  degenerate_void        (median OMP recon rel > 0.6)
  p_fam_min >= 0.05      -> dictionary_alignment_within_null
  p_fam_min <  0.05 and sign_agree >= 0.7
                         -> dictionary_aligned_causal_supported
  else                   -> dictionary_aligned_snapshot_only
  GATE (plan v4): if null not passed, "monosemantic feature"
  language is banned and full SAE training is NOT chartered.

Anchors:
  a1  Vt8 rebuild vs 2939 < 1e-6
  a1w words identity 2987/2989/here (exact, composite fmt)
  a2  rerun L2 headC vs 2987 rel < 1e-9
  a3  rerun L16N headC vs 2987 rel < 1e-9
  a4  rerun L2 act vs 2989 rel < 1e-9 (9 layers)
  a7  determinism rel < 1e-4 (L2, L16N)
  a8  c_lang/c_cls rebuild vs 2989 exact < 1e-12
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
SRC_2979 = os.path.join(BASE, 'phase2979', 'reversal_anatomy',
                        'reversal_anatomy.npz')
SRC_2987 = os.path.join(BASE, 'phase2987',
                        'context_minimal_audit',
                        'context_minimal_audit.npz')
SRC_2989 = os.path.join(BASE, 'phase2989',
                        'mlp_neuron_registry',
                        'mlp_neuron_registry.npz')
EXEC_2977 = os.path.join(BASE, 'phase2977',
                         'two_axis_fusion_injection',
                         'execution.json')
OUT = os.path.join(BASE, 'phase2992',
                   'sparse_dictionary_registration')

MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
L34 = 34
REG_LAYERS = [6, 7, 8, 9, 10, 11, 12, 17, 34]
DICT_LAYERS = [6, 7, 8, 9, 10, 11, 12]
K_ATOMS = 96          # dictionary size per layer
K_SPARSE = 16        # OMP sparsity
N_NULL = 1000        # Haar random-direction nulls (2931);
                     # floor p_fam = 14/1001 = 0.014 < gate
                     # (run1 N_NULL=200 made the family gate
                     # unreachable: 14/201 = 0.0697 > 0.05)
N_FAM = 14           # Bonferroni family: 7 layers x 2 axes
RECON_GATE = 0.6     # nondegeneracy: median recon rel
M_ABL = 8            # ablated atoms per cell
N_SAMPLE_CELLS = 12  # 3 per (cls, lang) stratum
SIGN_GATE = 0.7      # realized-vs-predicted sign agreement
P_GATE = 0.05
RNG_MAIN = 2992
BIT_TOL = 1e-9
FILLER_NEUTRAL = (' The sun rises in the east and sets in '
                  'the west .')


def sha8(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]


def log(msg, lines):
    lines.append('[%s] %s' % (time.strftime('%H:%M:%S'), msg))
    with open(os.path.join(OUT, 'run_log.txt'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def _dist2(X, C):
    """Pairwise sq distances via Gram trick (no big
    broadcast): (n,k) = |X|^2 - 2 X C^T + |C|^2."""
    x2 = np.sum(X ** 2, axis=1)[:, None]
    c2 = np.sum(C ** 2, axis=1)[None, :]
    d = x2 - 2.0 * (X @ C.T) + c2
    return np.maximum(d, 0.0)


def kmeans(X, k, rng, n_init=3, iters=60):
    """Seeded k-means with k-means++ init.  Unit-norm atoms."""
    n = X.shape[0]
    best = None
    best_inertia = None
    for _ in range(n_init):
        idx = [int(rng.integers(n))]
        d2 = _dist2(X, X[idx[0]][None, :]).ravel()
        for _ in range(k - 1):
            tot = float(d2.sum())
            if tot <= 0:
                j = int(rng.integers(n))
            else:
                j = int(rng.choice(n, p=d2 / tot))
            idx.append(j)
            d2 = np.minimum(d2, _dist2(
                X, X[j][None, :]).ravel())
        C = X[np.array(idx)].copy()
        lab = np.zeros(n, dtype=int)
        for _ in range(iters):
            dist = _dist2(X, C)
            lab = np.argmin(dist, axis=1)
            newC = C.copy()
            for j in range(k):
                m = lab == j
                if m.any():
                    newC[j] = X[m].mean(0)
                else:  # empty -> farthest sample
                    far = int(np.argmax(dist.min(1)))
                    newC[j] = X[far]
            shift = float(np.abs(newC - C).max())
            C = newC
            if shift < 1e-10:
                break
        dist = _dist2(X, C)
        inertia = float(dist[np.arange(n), lab].sum())
        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best = C
    nrm = np.linalg.norm(best, axis=1, keepdims=True)
    return best / np.maximum(nrm, 1e-30)


def omp(xn, A, k):
    """Orthogonal matching pursuit of unit vector xn in A
    (atoms = rows).  Returns coeff vector and recon rel."""
    r = xn.copy()
    support = []
    c = np.zeros(A.shape[0])
    for _ in range(k):
        proj = A @ r
        j = int(np.argmax(np.abs(proj)))
        if abs(proj[j]) < 1e-12:
            break
        if j in support:
            break
        support.append(j)
        As = A[support].T
        cs, *_ = np.linalg.lstsq(As, xn, rcond=None)
        r = xn - As @ cs
    if support:
        c[support] = cs
    return c, float(np.linalg.norm(r))


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)

    # ---------- execution freeze (BEFORE any compute) ----------
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2992,
                   'name': 'sparse_dictionary_registration',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'prereg': {
                       'design': 'plan v4 P2 gate: rerun 74 '
                                 'cells x {L2,L16N}; T1 '
                                 'k-means dictionary (k=%d, '
                                 'n_init=3) per layer 6-12 '
                                 'on 148 row-normalized MLP '
                                 'hidden samples + OMP '
                                 '(k=%d); T2 max-atom '
                                 'axis alignment vs %d '
                                 'Haar random-direction '
                                 'nulls (2931 caliber), '
                                 'Bonferroni x%d family; '
                                 'T3 12 stratified cells, '
                                 'lang axis, m=%d atoms: '
                                 'snapshot direct-path '
                                 'pred_axis=dh.c_lang '
                                 '(residual identity path) '
                                 'vs realized downstream '
                                 '(res34_mod-res34_base)'
                                 '.d_lang_u at L34 decoder '
                                 'input + random-atom '
                                 'control; sign agreement'
                                 % (K_ATOMS, K_SPARSE,
                                    N_NULL, N_FAM, M_ABL),
                       'T2_primary': 'p_fam_min >= 0.05 -> '
                                     'dictionary_alignment_'
                                     'within_null (SAE not '
                                     'chartered, '
                                     'monosemantic language '
                                     'banned); p_fam_min < '
                                     '0.05 and sign_agree '
                                     '>= 0.7 -> '
                                     'dictionary_aligned_'
                                     'causal_supported; '
                                     'else '
                                     'dictionary_aligned_'
                                     'snapshot_only; '
                                     'median recon rel > '
                                     '0.6 -> '
                                     'degenerate_void',
                       'anchors': {
                           'a1': 'Vt8 rebuild < 1e-6',
                           'a1w': 'words identity '
                                  '2987/2989/here exact',
                           'a2': 'rerun L2 headC vs 2987 '
                                 'rel < 1e-9',
                           'a3': 'rerun L16N headC vs '
                                 '2987 rel < 1e-9',
                           'a4': 'rerun L2 act vs 2989 '
                                 'rel < 1e-9 (9 layers)',
                           'a7': 'determinism rel < 1e-4',
                           'a8': 'c_lang/c_cls rebuild '
                                 'exact < 1e-12'},
                       'rng': RNG_MAIN,
                       'p_gate': P_GATE,
                       'k_atoms': K_ATOMS,
                       'k_sparse': K_SPARSE,
                       'n_null': N_NULL,
                       'n_fam': N_FAM,
                       'recon_gate': RECON_GATE,
                       'm_abl': M_ABL,
                       'n_sample_cells': N_SAMPLE_CELLS,
                       'sign_gate': SIGN_GATE}},
                  f, indent=1)
    log('execution.json frozen', lines)

    # ---------- cells (2977 exec verbatim) ----------
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

    # ---------- upstream artifacts ----------
    z39 = np.load(SRC_2939, allow_pickle=True)
    z27 = np.load(SRC_2927, allow_pickle=True)
    dirs27 = z27['dirs_word'].astype(np.float64)
    _, _, Vt_loc = np.linalg.svd(dirs27,
                                 full_matrices=False)
    a1_diff = float(np.abs(Vt_loc[:8]
                           - z39['Vt8']).max())
    a1_ok = bool(a1_diff < 1e-6)
    log('a1 Vt8 rebuild diff %.2e ok=%s'
        % (a1_diff, a1_ok), lines)
    u35 = dirs27[NL - 1]
    z79 = np.load(SRC_2979, allow_pickle=True)
    d79_l = z79['d_lang_u'].astype(np.float64)
    d79_c = z79['d_cls_u'].astype(np.float64)
    z87 = np.load(SRC_2987, allow_pickle=True)
    z89 = np.load(SRC_2989, allow_pickle=True)
    words87 = [str(w) for w in z87['words']]
    words89 = [str(w) for w in z89['words']]
    words_here = ['%s:%s:%s' % c for c in cells]

    # a1w words identity three-way (2989 'F:en:he' caliber)
    a1w_ok = bool(words87 == words89 == words_here)
    log('a1w words identity 2987/2989/here: %s'
        % a1w_ok, lines)

    headC87 = z87['headC'].astype(np.float64)
    act_l89 = {li: z89['act_%d' % li].astype(np.float64)
               for li in REG_LAYERS}
    c_lang89 = {li: z89['c_lang_%d' % li].astype(np.float64)
                for li in REG_LAYERS}
    c_cls89 = {li: z89['c_cls_%d' % li].astype(np.float64)
               for li in REG_LAYERS}

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
    assert n_single == n_test, 'single-token check'
    ids_the = tok(' the', add_special_tokens=False)[
        'input_ids']
    assert len(ids_the) == 1
    func_tid = int(ids_the[0])
    neutral_pool = tok(FILLER_NEUTRAL,
                       add_special_tokens=False)['input_ids']
    n14 = [int(neutral_pool[k % len(neutral_pool)])
           for k in range(14)]

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    log('model loaded', lines)

    cap_act = {li: [] for li in REG_LAYERS}
    cap_op = {li: [] for li in range(NL)}
    cap_x17 = []
    cap_res34 = []
    cap_hmod = {}
    handles = []
    mod_pending = {'li': None, 'delta': None}

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

    def hook_res34(module, args, kwargs):
        x = hs_of(args, kwargs)
        if x is None or x.dim() < 2:
            return None
        cap_res34.append(
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
            # word sits at last position in both conds
            # (L2 [func,w]: -1 == pos 1, 2989-identical)
            cap_act[li].append(
                x[:, -1, :].detach().float().cpu().numpy())
            return None
        return h

    def hook_mod(li):
        def h(module, args, kwargs):
            if mod_pending['li'] != li \
                    or mod_pending['delta'] is None:
                return None
            x = args[0] if args else kwargs.get('input')
            if x is None or x.dim() < 2:
                return None
            x2 = x.clone()
            x2[:, -1, :] = x2[:, -1, :] \
                - mod_pending['delta']
            cap_hmod[li] = x2[:, -1, :].detach() \
                .float().cpu().numpy()
            return (x2,), kwargs
        return h

    handles.append(
        layers[17].self_attn
        .register_forward_pre_hook(
            hook_x17, with_kwargs=True))
    # L34 decoder-layer input (residual, 2560 space)
    handles.append(
        layers[L34].register_forward_pre_hook(
            hook_res34, with_kwargs=True))
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
        if li in DICT_LAYERS:
            handles.append(
                layers[li].mlp.down_proj
                .register_forward_pre_hook(
                    hook_mod(li), with_kwargs=True))

    def clear_cap():
        for li in cap_act:
            del cap_act[li][:]
        for li in cap_op:
            del cap_op[li][:]
        del cap_x17[:]
        del cap_res34[:]
        cap_hmod.clear()

    def forward1(toks):
        clear_cap()
        with _t.no_grad():
            model(_t.tensor([toks], device='cuda'))
        return ({li: cap_act[li][0].astype(np.float64)
                 for li in REG_LAYERS},
                {li: cap_op[li][0].astype(np.float64)
                 for li in range(NL)},
                cap_res34[0].astype(np.float64)
                .reshape(-1))

    def forward_mod(toks, li_mod, delta_t):
        mod_pending['li'] = li_mod
        mod_pending['delta'] = delta_t
        try:
            out = forward1(toks)
        finally:
            mod_pending['li'] = None
            mod_pending['delta'] = None
        return out

    # weight-space readout rows (2989/2987 verbatim)
    M = np.zeros((NL, NH * HD))
    for li in range(NL):
        Wo = layers[li].self_attn.o_proj.weight.detach() \
            .float().cpu().numpy()
        M[li] = u35 @ Wo
    M34v = M[L34]

    # registry axes rebuild (a8 identity gate)
    a8_diff = 0.0
    Wd = {}
    c_lang = {}
    c_cls = {}
    for li in REG_LAYERS:
        Wdm = layers[li].mlp.down_proj.weight.detach() \
            .float().cpu().numpy().astype(np.float64)
        Wd[li] = Wdm.T
        c_lang[li] = Wd[li] @ d79_l
        c_cls[li] = Wd[li] @ d79_c
        a8_diff = max(a8_diff,
                      float(np.abs(c_lang[li]
                                   - c_lang89[li]).max()),
                      float(np.abs(c_cls[li]
                                   - c_cls89[li]).max()))
    a8_ok = bool(a8_diff < 1e-12)
    log('a8 c_lang/c_cls rebuild vs 2989: %.2e ok=%s'
        % (a8_diff, a8_ok), lines)

    # ---------- base sweep: 2 conds ----------
    # C34 per-head slicing (2991 verbatim headC caliber)
    Wo34 = layers[L34].self_attn.o_proj.weight.detach() \
        .float().cpu().numpy().astype(np.float64)
    C34 = u35 @ Wo34
    M34v = M[L34]
    CONDS = [('L2', []), ('L16N', n14)]
    n_cond = 2
    headC = np.zeros((n_cond, n_test, NH))
    prof34 = np.zeros((n_cond, n_test))
    res34_base = np.zeros((n_cond, n_test, 2560))
    act = {li: {ci: np.zeros((n_test, 9728))
                for ci in range(n_cond)}
           for li in REG_LAYERS}
    for ci, (cname, fill) in enumerate(CONDS):
        for i, (_, _, w) in enumerate(cells):
            seq = list(fill) + [func_tid, tid_map[w]]
            assert len(seq) == len(fill) + 2
            acts, ops, res = forward1(seq)
            for li in REG_LAYERS:
                act[li][ci][i] = acts[li].reshape(-1)
            res34_base[ci, i] = res
            x34 = ops[L34].reshape(-1)
            for h in range(NH):
                headC[ci, i, h] = float(np.dot(
                    C34[h * HD:(h + 1) * HD],
                    x34[h * HD:(h + 1) * HD]))
            prof34[ci, i] = float(np.dot(x34, M34v))
        log('cond %s done' % cname, lines)

    # ---------- anchors ----------
    op_a = forward1([func_tid, tid_map['man']])[1]
    op_b = forward1([func_tid, tid_map['man']])[1]
    a7a = float(np.abs(op_a[30] - op_b[30]).max()
                / max(float(np.abs(op_a[30]).max()), 1e-30))
    op_c = forward1(list(n14)
                    + [func_tid, tid_map['man']])[1]
    op_d = forward1(list(n14)
                    + [func_tid, tid_map['man']])[1]
    a7b = float(np.abs(op_c[30] - op_d[30]).max()
                / max(float(np.abs(op_c[30]).max()), 1e-30))
    a7_ok = bool(a7a < 1e-4 and a7b < 1e-4)
    log('a7 determinism %.2e / %.2e ok=%s'
        % (a7a, a7b, a7_ok), lines)

    # a2: L2 headC vs 2987 (bit level)
    a2_rel = float(np.max(np.abs(headC[0] - headC87[0])
                          / np.maximum(np.abs(headC87[0])
                                       .max(), 1e-30)))
    a2_ok = bool(a2_rel < BIT_TOL)
    log('a2 L2 headC vs 2987: %.2e ok=%s'
        % (a2_rel, a2_ok), lines)

    a3_rel = float(np.max(np.abs(headC[1] - headC87[2])
                          / np.maximum(np.abs(headC87[2])
                                       .max(), 1e-30)))
    a3_ok = bool(a3_rel < BIT_TOL)
    log('a3 L16N headC vs 2987: %.2e ok=%s'
        % (a3_rel, a3_ok), lines)

    a4_rel = 0.0
    for li in REG_LAYERS:
        a4_rel = max(a4_rel, float(np.max(
            np.abs(act[li][0] - act_l89[li])
            / max(float(np.abs(act_l89[li]).max()), 1e-30))))
    a4_ok = bool(a4_rel < BIT_TOL)
    log('a4 L2 act vs 2989 (9 layers): %.2e ok=%s'
        % (a4_rel, a4_ok), lines)

    anchor_ok = bool(a1_ok and a1w_ok and a2_ok and a3_ok
                     and a4_ok and a7_ok and a8_ok)

    # preinit (anchor_fail path must stay construct-safe)
    verdict = None
    T1 = T2 = T3 = None
    atoms = {}
    sample_idx = []
    t3_rows = []
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
        log('ANCHOR FAIL -> all void', lines)
    else:
        # ---------- T1 dictionary per layer ----------
        rng_null = np.random.default_rng(RNG_MAIN + 50)
        G = rng_null.standard_normal((N_NULL, 9728))
        Gn = G / np.linalg.norm(G, axis=1, keepdims=True)
        atoms = {}
        recon_med = {}
        T1 = {}
        for li in DICT_LAYERS:
            X = np.vstack([act[li][0], act[li][1]])
            nrm = np.linalg.norm(X, axis=1, keepdims=True)
            Xn = X / np.maximum(nrm, 1e-30)
            rng_km = np.random.default_rng(
                RNG_MAIN + 10 + li)
            A = kmeans(Xn, K_ATOMS, rng_km)
            atoms[li] = A
            coeffs = np.zeros((Xn.shape[0], K_ATOMS))
            rels = np.zeros(Xn.shape[0])
            for i in range(Xn.shape[0]):
                coeffs[i], rels[i] = omp(Xn[i], A,
                                         K_SPARSE)
            recon_med[li] = float(np.median(rels))
            T1[li] = {'recon_rel_median':
                      round(recon_med[li], 4),
                      'k_atoms': K_ATOMS,
                      'k_sparse': K_SPARSE}
            log('T1 layer %d: recon rel median %.4f'
                % (li, recon_med[li]), lines)
        recon_all_ok = bool(np.median(
            [recon_med[li] for li in DICT_LAYERS])
            <= RECON_GATE)
        log('T1 recon gate (median %.4f <= %.2f): %s'
            % (float(np.median([recon_med[li]
                                for li in DICT_LAYERS])),
               RECON_GATE, recon_all_ok), lines)

        # ---------- T2 alignment vs random null (primary) ----------
        T2 = {'family': {}, 'p_fam_min': None,
              'best': None}
        p_fam_min = 1.0
        best_key = None
        for li in DICT_LAYERS:
            A = atoms[li]
            An = A @ Gn.T   # (K_ATOMS, N_NULL) cos to nulls
            null_max = np.abs(An).max(axis=0)
            for ax, cc in (('lang', c_lang[li]),
                           ('cls', c_cls[li])):
                chu = cc / max(float(np.linalg.norm(cc)),
                               1e-30)
                s = A @ chu
                obs = float(np.abs(s).max())
                j = int(np.argmax(np.abs(s)))
                p_raw = float(
                    (np.sum(null_max >= obs) + 1)
                    / (N_NULL + 1))
                p_fam = min(1.0, p_raw * N_FAM)
                T2['family']['%d_%s' % (li, ax)] = {
                    'max_abs_cos': round(obs, 4),
                    'top_atom': j,
                    'top_atom_cos': round(float(s[j]),
                                          4),
                    'null_max_mean': round(
                        float(null_max.mean()), 4),
                    'null_max_p95': round(
                        float(np.percentile(null_max,
                                            95)), 4),
                    'p_raw': round(p_raw, 4),
                    'p_fam': round(p_fam, 4)}
                if p_fam < p_fam_min:
                    p_fam_min = p_fam
                    best_key = '%d_%s' % (li, ax)
        T2['p_fam_min'] = round(p_fam_min, 4)
        T2['best'] = best_key
        log('T2 alignment: p_fam_min=%.4f best=%s'
            % (p_fam_min, best_key), lines)
        log('T2 family: %s'
            % json.dumps(T2['family']), lines)

        # ---------- T3 dual-caliber ablation ----------
        # 12 stratified cells (3 per cls x lang stratum)
        rng_cell = np.random.default_rng(RNG_MAIN + 60)
        sample_idx = []
        for c0 in (0, 1):
            for l0 in (0, 1):
                pool = [i for i in range(n_test)
                        if cls[i] == c0 and lang[i] == l0]
                pick = rng_cell.choice(
                    pool, size=3, replace=False)
                sample_idx += [int(p) for p in pick]
        t3_rows = []
        n_skip = 0
        for li in DICT_LAYERS:
            A = atoms[li]
            chu = c_lang[li] / max(
                float(np.linalg.norm(c_lang[li])), 1e-30)
            atom_ax = A @ chu            # (K_ATOMS,)
            rng_arm = np.random.default_rng(
                RNG_MAIN + 70 + li)
            for i in sample_idx:
                h = act[li][0][i]
                proj = A @ h             # (K_ATOMS,)
                contrib = proj * atom_ax
                top = np.argsort(
                    -np.abs(contrib))[:M_ABL]
                dh = -(proj[top] @ A[top])
                # direct-path snapshot prediction
                # (residual identity path, analytic):
                # equals dh @ c_lang[li] exactly
                pred_axis = float(dh @ c_lang[li])
                if abs(pred_axis) < 1e-9:
                    n_skip += 1
                    continue
                dt = _t.tensor(
                    dh.astype(np.float32),
                    device='cuda')
                _, _, res_mod = forward_mod(
                    [func_tid, tid_map[cells[i][2]]],
                    li, dt)
                h_mod = cap_hmod[li].reshape(-1) \
                    .astype(np.float64)
                # guard normalized by the rounded quantity
                # scale (h-dh), not |h|max: SwiGLU outputs
                # are sparse so |h|max understates the bf16
                # ULP of (h-dh); expected ~2^-9 if applied
                hdh = h - dh
                mod_rel = float(np.abs(
                    h_mod - hdh).max()
                    / max(float(np.abs(hdh).max()),
                          1e-30))
                real_axis = float(
                    (res_mod - res34_base[0, i])
                    @ d79_l)
                # random-atom control
                ridx = rng_arm.choice(
                    K_ATOMS, size=M_ABL,
                    replace=False)
                dhr = -(A[ridx].T
                        @ (A[ridx] @ h))
                dtr = _t.tensor(
                    dhr.astype(np.float32),
                    device='cuda')
                _, _, res_mod_r = forward_mod(
                    [func_tid, tid_map[cells[i][2]]],
                    li, dtr)
                rand_axis = float(
                    (res_mod_r - res34_base[0, i])
                    @ d79_l)
                t3_rows.append({
                    'layer': li, 'cell': i,
                    'pred_axis': round(pred_axis, 6),
                    'real_axis': round(real_axis, 6),
                    'rand_axis': round(rand_axis, 6),
                    'mod_guard_rel': round(mod_rel, 8),
                    'sign_agree':
                        bool(np.sign(real_axis)
                             == np.sign(pred_axis))})
        n_rows = len(t3_rows)
        n_agree = sum(1 for r in t3_rows
                      if r['sign_agree'])
        sign_agree = (n_agree / n_rows) if n_rows else 0.0
        mag_ratio = float(np.median(
            [abs(r['real_axis'])
             / max(abs(r['pred_axis']), 1e-30)
             for r in t3_rows])) if n_rows else 0.0
        spec = float(np.median(
            [abs(r['real_axis'])
             / max(abs(r['rand_axis']), 1e-30)
             for r in t3_rows if abs(r['rand_axis'])
             > 1e-30])) if n_rows else 0.0
        guard_max = max((r['mod_guard_rel']
                         for r in t3_rows), default=0.0)
        T3 = {'n_rows': n_rows, 'n_skip_pred_zero': n_skip,
              'sign_agree': round(sign_agree, 4),
              'mag_ratio_median': round(mag_ratio, 4),
              'specificity_vs_rand_median':
                  round(spec, 4),
              'mod_guard_rel_max': round(guard_max, 8),
              'rows': t3_rows}
        log('T3 dual caliber: n=%d sign_agree=%.4f '
            'mag_ratio=%.4f spec=%.4f guard=%.1e'
            % (n_rows, sign_agree, mag_ratio, spec,
               guard_max), lines)

        # ---------- verdict (primary T2 family) ----------
        if not recon_all_ok:
            verdict = 'degenerate_void'
            log('DEGENERATE dictionary -> void', lines)
        elif p_fam_min >= P_GATE:
            verdict = 'dictionary_alignment_within_null'
        elif sign_agree >= SIGN_GATE:
            verdict = 'dictionary_aligned_causal_supported'
        else:
            verdict = 'dictionary_aligned_snapshot_only'
        log('VERDICT: %s' % verdict, lines)

    elapsed = round(time.monotonic() - t0, 1)
    result = {
        'phase': 2992,
        'name': 'sparse_dictionary_registration',
        'final_verdict': verdict,
        'anchor_all_ok': anchor_ok,
        'anchors': {
            'a1_Vt8': a1_ok,
            'a1w_words': a1w_ok,
            'a2_headC_L2_rel': a2_rel,
            'a3_headC_L16N_rel': a3_rel,
            'a4_act_vs2989_rel': a4_rel,
            'a7_determinism': a7_ok,
            'a8_axis_rebuild': a8_ok},
        'T1': T1,
        'T2': T2,
        'T3': T3,
        'elapsed_s': elapsed,
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    npz_path = os.path.join(
        OUT, 'sparse_dictionary_registration.npz')
    save = {'words': np.array(words_here),
            'lang': lang, 'cls': cls,
            'headC_L2': headC[0],
            'headC_L16N': headC[1],
            'prof34_L2': prof34[0],
            'prof34_L16N': prof34[1],
            'res34_base_L2': res34_base[0],
            'sample_idx': np.array(sample_idx
                                   if anchor_ok else [])}
    for li in DICT_LAYERS:
        save['atoms_%d' % li] = atoms[li] \
            .astype(np.float32)
    np.savez_compressed(npz_path, **save)
    with open(os.path.join(OUT, 'seal.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'sealed_at':
                   time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'verdict': verdict,
                   'npz_sha256_8':
                       sha8(npz_path),
                   'result_sha256_8':
                       sha8(os.path.join(OUT,
                                         'result.json'))},
                  f, indent=1)
    log('PHASE2992 DONE elapsed=%ss verdict=%s'
        % (elapsed, verdict), lines)
    print('PHASE2992 DONE', verdict, elapsed)


if __name__ == '__main__':
    main()
