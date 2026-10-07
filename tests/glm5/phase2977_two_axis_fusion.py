# -*- coding: utf-8 -*-
"""Phase 2977: Omega-B two-axis fusion algebra - 2^2
factorial causal injection at L17 (preregistered).

Why: plan v3 Omega-B (attachment blank 1). Prior evidence:
2972 language axis dominates static band signature, class
axis English-bound; 2966-2968 language injection at L17
drives B collapse with s_c = 0.657. Open question: when
BOTH axes inject simultaneously, is the layer response the
SUM of single-axis responses (linear fusion) or is there a
nonlinear interaction term (fusion algebra)?

Design (frozen before any observation):
  Stage 1: 74 base single forwards (cells from 2972
     execution.json via 2973 npz identity), 2973 protocol
     verbatim; capture o_proj input pos1 all 36 layers
     (4096-d) AND L17 layer-input residual pos1 (2560-d).
     Derive in-run axis directions at L17 input space:
       d_lang = mean(fr) - mean(en), unit-norm
       d_cls  = mean(C)  - mean(F),  unit-norm
     (quasi-post-hoc derivation note registered; the TEST
     is on new injection responses.)
  Stage 2: 74 words x 4 conditions {sham, lang(s=0.5),
     cls(s=0.5), both} = 296 forwards; injection = in-place
     add s*d at L17 self_attn input pos1 (established
     hook point); dose 0.5 chosen below s_c=0.657 (2966)
     per axis to avoid the saturation regime (2959).
  Response: layer profile prof(l) = x_op(l) . M(l), and
     head grid C(l,h) (2973 machinery).

Tests (frozen):
  T1 interaction (PRIMARY): per layer
     I_wl = prof11 - prof10 - prof01 + prof00;
     stat = median_w I_l; null = per-word sign-flip
     (rng 2977 x10000), maxT family 36; gate p<=0.01
     any layer => nonlinear fusion term exists.
  T2 main effects: median_w (prof10 - prof00) sign-flip
     maxT rng 2978 family 36 (lang); median_w
     (prof01 - prof00) rng 2979 (cls); saturation
     descriptive: median_w(prof11-prof00) vs sum.
  T3 descriptive (quasi-post-hoc): head-grid interaction
     median cells top10, share of |I|.

Anchors (frozen):
  a1 Vt8 rebuild vs 2939 npz < 1e-6
  a2 determinism < 1e-4
  a3 identity vs 2973 npz: norms rel < 1e-4 AND coss
     absdiff < 1e-6 over 74x36 (2970 discipline)
  a4 single-token 74/74
  a5 sham == base bit-level (prof00 vs stage-1 prof)
  a6 injection causality: for word C_en[0], cond lang vs
     sham: layers 0..16 o_proj inputs bit-identical,
     layers >= 17 differ (injection at L17 input is
     strictly downstream-local)

Verdict (frozen):
  anchor fail => anchor_fail_all_void
  T1 sig (>=1 layer) => nonlinear_fusion_interaction_detected
  T1 ns and (T2 lang sig or T2 cls sig) =>
     axes_additive_no_interaction
  else => factorial_injection_all_void
"""
import hashlib
import io
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
EXEC_2972 = os.path.join(BASE, 'phase2972',
                         'two_factor_signature',
                         'execution.json')
OUT = os.path.join(BASE, 'phase2977',
                   'two_axis_fusion_injection')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2977_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
N_PERM = 10000
P_TH = 0.01
RNG_T1 = 2977
RNG_T2L = 2978
RNG_T2C = 2979
S_DOSE_REL = 0.1
L_INJ = 17

PREREG = {
    'mode': 'stage1: 74 base forwards (2973 protocol, '
            'capture o_proj input 36 layers + L17 layer '
            'input); derive unit-norm axis dirs d_lang, '
            'd_cls at L17 input; stage2: 74x4 factorial '
            'injection {sham,lang,cls,both} at L17 input '
            'pos1, s_w=0.1*||x17_w|| per axis',
    'question': 'is the L17-input response to simultaneous '
                'lang+class axis injection the sum of '
                'single-axis responses (linear fusion) or '
                'is there a nonlinear interaction term?',
    'cells_source': 'phase2972 execution.json cells '
                    'verbatim; identity vs 2973 npz order',
    'dose_rationale': 'relative dose: per word inject '
                      's_w*dhat with s_w = 0.1*||x17_w|| '
                      'per axis (10% of own residual '
                      'norm); run3 unit-norm s=0.5 was '
                      '10-60x below the 2966 effect '
                      'regime (floor); single-'
                      'relative-dose design, dose '
                      'dependence NOT mapped (2958 law)',
    'derivation_note': 'axis directions derived in-run '
                       'from base snapshots of the same '
                       '74 words (quasi-post-hoc '
                       'derivation, preregistered test '
                       'on new injection responses)',
    'anchors': {
        'a1': 'Vt8 rebuild vs 2939 npz < 1e-6',
        'a2': 'determinism < 1e-4',
        'a3': 'identity vs 2973 npz norms rel<1e-4 and '
              'coss absdiff<1e-6 (74x36)',
        'a4': 'single-token 74/74',
        'a5': 'sham == base bit-level (74x36 prof)',
        'a6': 'injection locality: layers<17 identical, '
              'layers>=17 differ (word C_en[0])',
    },
    'T0': 'floor gate: max_l |median_w(prof10-prof00)| '
          '>= 0.02 required (lang axis calibration vs '
          '2966 regime; profile/B scale ~1.4); fail => '
          'injection_floor_all_void',
    'T1': 'interaction I=prof11-prof10-prof01+prof00, '
          'median_w per layer; sign-flip null rng 2977 '
          'x10000, maxT family 36; gate p<=0.01 any layer',
    'T2': 'main effects median_w prof10-prof00 (rng 2978) '
          'and prof01-prof00 (rng 2979), sign-flip maxT '
          'family 36; saturation descriptive',
    'T3': 'descriptive head-grid interaction top10 cells '
          '(quasi-post-hoc)',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'T0 fail => injection_floor_all_void; '
               'T1 sig => '
               'nonlinear_fusion_interaction_detected; '
               'T1 ns and (T2 lang sig or T2 cls sig) => '
               'axes_additive_no_interaction; else => '
               'factorial_injection_all_void',
    'correction_note': 'run1: self_attn is called '
                       'with hidden_states as kwarg in '
                       'this transformers version; '
                       'hook_x17/hook_inj read args[0] '
                       'only so x17 capture was empty '
                       '(IndexError). fix: read '
                       'args[0] else '
                       'kwargs[hidden_states] (2952 '
                       'discipline). run2: in-place add '
                       'of numpy direction onto cuda '
                       'tensor raised TypeError; fix: '
                       'torch.as_tensor(d, device, '
                       'dtype) before add. run3: '
                       'completed, anchors 6/6, but all '
                       'effects at floor (lang main top '
                       '0.0085 on profile scale where '
                       'B~1.4; 2966 regime gives '
                       'deltaB~1.3 at s_c) - unit-norm '
                       'dose 0.5 is 10-60x below the '
                       'effective regime; fix: relative '
                       'dose 0.1*||x17|| per word per '
                       'axis + preregistered T0 floor '
                       'gate; artifacts deleted and '
                       'rerun per discipline 3.',
}


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def log(msg, lines):
    lines.append(msg)
    print(msg, flush=True)


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)

    # ---------- cells ----------
    e72 = json.load(open(EXEC_2972, encoding='utf-8'))
    F_EN = e72['cells']['F_en']
    F_FR = e72['cells']['F_fr']
    C_EN = e72['cells']['C_en']
    C_FR = e72['cells']['C_fr']
    assert (len(F_EN), len(F_FR), len(C_EN), len(C_FR)) \
        == (15, 15, 22, 22), 'cell size drift'

    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2977,
                   'name': 'two_axis_fusion_injection',
                   'created': time.strftime(
                       '%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2939': sha8(SRC_2939),
                               's2927': sha8(SRC_2927),
                               's2973': sha8(SRC_2973)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'n_perm': N_PERM,
                   'rng': {'T1': RNG_T1, 'T2lang': RNG_T2L,
                           'T2cls': RNG_T2C},
                   'p_threshold': P_TH,
                   'dose_rel': S_DOSE_REL,
                   'inj_layer': L_INJ,
                   'cells': {'F_en': F_EN, 'F_fr': F_FR,
                             'C_en': C_EN, 'C_fr': C_FR},
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen (dose_rel %.2f, L%d)'
        % (S_DOSE_REL, L_INJ), lines)

    # ---------- sources ----------
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
    z73 = np.load(SRC_2973, allow_pickle=True)
    norms73 = z73['norms'].astype(np.float64)
    coss73 = z73['coss'].astype(np.float64)
    words73 = [str(w) for w in z73['words']]

    # ---------- model ----------
    import sys
    sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
    from phase2662_symmetric_mapping_contract import \
        load_native
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        MD, local_files_only=True, trust_remote_code=True,
        use_fast=True)
    cells = [('F', 'en', w) for w in F_EN] \
        + [('F', 'fr', w) for w in F_FR] \
        + [('C', 'en', w) for w in C_EN] \
        + [('C', 'fr', w) for w in C_FR]
    words77 = ['%s:%s:%s' % c for c in cells]
    ident_ok = bool(words77 == words73)
    log('cell identity vs 2973 npz order: %s' % ident_ok,
        lines)
    assert ident_ok, 'word order drift vs 2973'

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
    n_test = len(cells)
    a4_ok = bool(n_single == n_test)
    log('a4 single-token %d/%d ok=%s'
        % (n_single, n_test, a4_ok), lines)
    ids_the = tok(' the', add_special_tokens=False)[
        'input_ids']
    assert len(ids_the) == 1
    func_tid = int(ids_the[0])

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
                x[:, 1, :] += dt
        return None

    def hook_x17(module, args, kwargs):
        x = hs_of(args, kwargs)
        if x is None or x.dim() < 2:
            return None
        cap_x17.append(
            x[:, 1, :].detach().float().cpu().numpy())
        return None

    def hook_op(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('input')
            if x is None or x.dim() < 2:
                return None
            cap_op[li].append(
                x[:, 1, :].detach().float().cpu().numpy())
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

    def profile(op):
        prof = np.zeros(NL)
        grid = np.zeros((NL, NH))
        for li in range(NL):
            x = op[li].reshape(-1)
            xm = (x * M[li]).reshape(NH, HD)
            grid[li] = xm.sum(axis=1)
            prof[li] = float(grid[li].sum())
        return prof, grid

    # ---------- stage 1: base sweep ----------
    prof_base = np.zeros((n_test, NL))
    norms = np.zeros((n_test, NL))
    coss = np.zeros((n_test, NL))
    X17 = np.zeros((n_test, 2560))
    for i, (_, _, w) in enumerate(cells):
        op = forward1([func_tid, tid_map[w]])
        X17[i] = cap_x17[0].astype(np.float64).reshape(-1)
        for li in range(NL):
            x = op[li].reshape(-1)
            norms[i, li] = float(np.linalg.norm(x))
            coss[i, li] = float(np.dot(x, M[li])) \
                / max(norms[i, li] * Mnorm[li], 1e-30)
        prof_base[i], _ = profile(op)
        if (i + 1) % 20 == 0:
            log('base sweep [%d/%d]' % (i + 1, n_test),
                lines)

    # ---------- a3 identity vs 2973 ----------
    nrm_rel = float(np.max(np.abs(norms - norms73)
                           / np.maximum(norms73, 1e-30)))
    cos_abs = float(np.max(np.abs(coss - coss73)))
    a3_ok = bool(nrm_rel < 1e-4 and cos_abs < 1e-6)
    log('a3 identity vs 2973: norms rel %.2e, coss abs '
        '%.2e ok=%s' % (nrm_rel, cos_abs, a3_ok), lines)

    # ---------- axis directions ----------
    lang = np.array([0 if c[1] == 'en' else 1
                     for c in cells])
    cls = np.array([0 if c[0] == 'F' else 1
                    for c in cells])
    d_lang = X17[lang == 1].mean(axis=0) \
        - X17[lang == 0].mean(axis=0)
    d_cls = X17[cls == 1].mean(axis=0) \
        - X17[cls == 0].mean(axis=0)
    n_lang = float(np.linalg.norm(d_lang))
    n_cls = float(np.linalg.norm(d_cls))
    cos_axes = float(np.dot(d_lang, d_cls)
                     / max(n_lang * n_cls, 1e-30))
    assert n_lang > 0 and n_cls > 0, 'degenerate axis'
    d_lang = d_lang / n_lang
    d_cls = d_cls / n_cls
    log('axes: |d_lang|=%.3f |d_cls|=%.3f cos=%.4f'
        % (n_lang, n_cls, cos_axes), lines)

    # ---------- a2 determinism ----------
    op_a = forward1([func_tid, tid_map[C_EN[0]]])
    op_b = forward1([func_tid, tid_map[C_EN[0]]])
    a2_rel = float(np.abs(op_a[30] - op_b[30]).max()
                   / max(float(np.abs(op_a[30]).max()),
                         1e-30))
    a2_ok = bool(a2_rel < 1e-4)
    log('a2 determinism rel %.2e ok=%s'
        % (a2_rel, a2_ok), lines)

    # ---------- stage 2: factorial injection ----------
    profs = {c: np.zeros((n_test, NL)) for c in
             ('00', '10', '01', '11')}
    n17 = np.linalg.norm(X17, axis=1)
    dose_scales = S_DOSE_REL * n17
    for i, (_, _, w) in enumerate(cells):
        toks = [func_tid, tid_map[w]]
        s_w = float(dose_scales[i])
        op00 = forward1(toks, None)
        op10 = forward1(toks, s_w * d_lang)
        op01 = forward1(toks, s_w * d_cls)
        op11 = forward1(toks,
                        s_w * d_lang + s_w * d_cls)
        for key, op in (('00', op00), ('10', op10),
                        ('01', op01), ('11', op11)):
            profs[key][i], _ = profile(op)
        if (i + 1) % 20 == 0:
            log('factorial [%d/74]' % (i + 1), lines)

    # ---------- a5 sham == base ----------
    a5_diff = float(np.max(np.abs(
        profs['00'] - prof_base)))
    a5_ok = bool(a5_diff == 0.0)
    log('a5 sham vs base max abs diff %.2e ok=%s'
        % (a5_diff, a5_ok), lines)

    # ---------- a6 injection locality ----------
    w0 = C_EN[0]
    toks0 = [func_tid, tid_map[w0]]
    s0 = float(dose_scales[30])
    op_s = forward1(toks0, None)
    op_l = forward1(toks0, s0 * d_lang)
    pre_same = all(float(np.abs(
        op_s[li].astype(np.float64)
        - op_l[li].astype(np.float64)).max()) == 0.0
        for li in range(L_INJ))
    post_diff = any(float(np.abs(
        op_s[li].astype(np.float64)
        - op_l[li].astype(np.float64)).max()) > 0.0
        for li in range(L_INJ, NL))
    a6_ok = bool(pre_same and post_diff)
    log('a6 locality: pre17 identical=%s, post17 '
        'differ=%s ok=%s'
        % (pre_same, post_diff, a6_ok), lines)

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok and a6_ok)
    verdict = None
    t1 = t2 = t3 = None
    p1 = p2l = p2c = None
    st_l = st_c = sat = None
    sig1 = []
    sig_l = []
    sig_c = []

    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        # ---------- T1 interaction ----------
        I = profs['11'] - profs['10'] - profs['01'] \
            + profs['00']
        stat1 = np.median(I, axis=0)
        rng1 = np.random.default_rng(RNG_T1)
        fam1 = np.zeros(N_PERM)
        signs = np.where(rng1.random(
            (N_PERM, n_test)) < 0.5, -1.0, 1.0)
        for k in range(N_PERM):
            fam1[k] = float(np.abs(
                np.median(I * signs[k][:, None],
                          axis=0)).max())
        p1 = np.array([
            (np.sum(fam1 >= abs(stat1[li]) - 1e-12) + 1)
            / (N_PERM + 1) for li in range(NL)])
        sig1 = [li for li in range(NL)
                if p1[li] <= P_TH]
        log('T1 interaction median %s' %
            np.round(stat1, 4).tolist(), lines)
        log('T1 p %s sig %s'
            % (['%.1e' % v for v in p1], sig1), lines)

        # ---------- T2 main effects ----------
        def signflip_maxT(D, seed):
            st = np.median(D, axis=0)
            rg = np.random.default_rng(seed)
            fm = np.zeros(N_PERM)
            sg = np.where(rg.random(
                (N_PERM, n_test)) < 0.5, -1.0, 1.0)
            for k in range(N_PERM):
                fm[k] = float(np.abs(
                    np.median(D * sg[k][:, None],
                              axis=0)).max())
            pp = np.array([
                (np.sum(fm >= abs(st[li]) - 1e-12) + 1)
                / (N_PERM + 1) for li in range(NL)])
            return st, pp

        st_l, p2l = signflip_maxT(
            profs['10'] - profs['00'], RNG_T2L)
        st_c, p2c = signflip_maxT(
            profs['01'] - profs['00'], RNG_T2C)
        sig_l = [li for li in range(NL)
                 if p2l[li] <= P_TH]
        sig_c = [li for li in range(NL)
                 if p2c[li] <= P_TH]
        sat = np.median(profs['11'] - profs['00'],
                        axis=0)
        add_pred = st_l + st_c
        log('T2 lang sig %s top3 %s | cls sig %s top3 %s'
            % (sig_l, [(li, round(float(st_l[li]), 4))
                       for li in np.argsort(
                           -np.abs(st_l))[:3]],
               sig_c, [(li, round(float(st_c[li]), 4))
                       for li in np.argsort(
                           -np.abs(st_c))[:3]]), lines)
        log('T2 saturation: median(observed both) %s vs '
            'additive sum %s'
            % (np.round(sat, 3).tolist(),
               np.round(add_pred, 3).tolist()), lines)

        # ---------- T3 head-grid descriptive ----------
        # layer-level primary; per-head grid recomputation
        # registered as future work (runtime trade-off)
        top_layers = [(int(li),
                       round(float(stat1[li]), 4),
                       float('%.3e' % p1[li]))
                      for li in np.argsort(
                          -np.abs(stat1))[:5]]
        t3 = {'top5_interaction_layers': top_layers,
              'head_grid': 'deferred (layer-level '
                           'primary; grid recomputation '
                           'registered as future work)'}
        log('T3 %s' % json.dumps(t3), lines)

        # ---------- T0 floor gate ----------
        t0_max = float(np.max(np.abs(st_l)))
        t0_ok = bool(t0_max >= 0.02)
        log('T0 floor gate: max|lang main| %.4f ok=%s'
            % (t0_max, t0_ok), lines)

        # ---------- verdict ----------
        if not t0_ok:
            verdict = 'injection_floor_all_void'
        elif len(sig1) >= 1:
            verdict = \
                'nonlinear_fusion_interaction_detected'
        elif len(sig_l) >= 1 or len(sig_c) >= 1:
            verdict = 'axes_additive_no_interaction'
        else:
            verdict = 'factorial_injection_all_void'
        save = {
            'prof00': profs['00'], 'prof10': profs['10'],
            'prof01': profs['01'], 'prof11': profs['11'],
            'prof_base': prof_base,
            'norms': norms, 'coss': coss,
            'Mnorm': Mnorm,
            'd_lang': d_lang * n_lang,
            'd_cls': d_cls * n_cls,
            'axes_cos': np.array([cos_axes]),
            'words': np.array(words77),
            'stat1': stat1, 'p1': p1,
            'st_lang': st_l, 'p_lang': p2l,
            'st_cls': st_c, 'p_cls': p2c,
            'sat': sat, 't0_max': t0_max,
            'n17': n17, 'dose_scales': dose_scales}

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2977, 'model': 'qwen3-4b',
           'prereg': PREREG,
           'anchors': {'a1_diff': float('%.3e' % a1_diff),
                       'a1_ok': a1_ok,
                       'a2_rel': float('%.3e' % a2_rel),
                       'a2_ok': a2_ok,
                       'a3_nrm_rel': float('%.3e'
                                           % nrm_rel),
                       'a3_cos_abs': float('%.3e'
                                           % cos_abs),
                       'a3_ok': a3_ok,
                       'a4_ok': a4_ok,
                       'a5_diff': float('%.3e'
                                        % a5_diff),
                       'a5_ok': a5_ok,
                       'a6_ok': a6_ok,
                       'axes_cos': round(cos_axes, 4),
                       'ok': anchor_ok},
           'T1': {'stat': np.round(stat1, 4).tolist()
                  if p1 is not None else None,
                  'p': [float('%.3e' % v) for v in p1]
                       if p1 is not None else None,
                  'sig_layers':
                      sig1 if p1 is not None else None},
           'T2': {'lang_stat':
                      np.round(st_l, 4).tolist()
                      if p2l is not None else None,
                  'lang_p':
                      [float('%.3e' % v) for v in p2l]
                      if p2l is not None else None,
                  'lang_sig':
                      sig_l if p2l is not None else None,
                  'cls_stat':
                      np.round(st_c, 4).tolist()
                      if p2c is not None else None,
                  'cls_p':
                      [float('%.3e' % v) for v in p2c]
                      if p2c is not None else None,
                  'cls_sig':
                      sig_c if p2c is not None else None,
                  'saturation':
                      np.round(sat, 3).tolist()
                      if p2l is not None else None},
           'T0': {'max_lang_main':
                      None if p2l is None
                      else float('%.4e' % t0_max),
                  'ok':
                      None if p2l is None else t0_ok},
           'T3': t3,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if verdict != 'anchor_fail_all_void':
        np.savez_compressed(os.path.join(
            OUT, 'two_axis_fusion_injection.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2977 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    import torch  # noqa: E402
    main()
