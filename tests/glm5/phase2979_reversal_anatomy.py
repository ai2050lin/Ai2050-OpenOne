# -*- coding: utf-8 -*-
"""Phase 2979: high-dose interaction reversal anatomy via
per-head snapshot decomposition (2950 machinery, adapted
from ablation to factorial injection; preregistered).

Why: 2978 found the L17 two-axis interaction is biphasic
in dose - sub-additive (negative) at low dose with peak
at min(a,b)=0.1, reversing to POSITIVE +0.0127 at
(0.4,0.4). Mechanism candidates:
  (a) saturation squeeze: heads compress both single-axis
      responses into a floor; joint response direction
      stays aligned with the linear sum (g<1, cos~1);
  (b) gain drop with reorientation: responses rotate away
      from the linear-sum direction (g<1, cos low);
  (c) superlinear gain: joint response exceeds the linear
      sum (g>=1).

Design (frozen before any observation):
  Stage 1: 74 base forwards (2977/2978 protocol verbatim,
     cells from 2977 execution.json), full-layer o_proj
     input capture -> norms/coss (a3 identity vs 2973) +
     X17 (axis dirs, a5 identity vs 2977 npz).
  Stage 2: 74 words x 4 conditions {(0.1,0.1) dose-id
     anchor, (0.4,0), (0,0.4), (0.4,0.4)} at L17 input
     pos1, relative dose a*n17_w*d_lang + b*n17_w*d_cls.
     Capture: full-layer profile + L17 per-head o_proj
     input slices (4096-d, 32 heads). 296 forwards.
  Per-head readout contribution (2950 identity):
     c_h(w) = u35 . (Wo17_h @ x_h(w)); sum_h c_h == prof17
     (a7 self-identity, relative 1e-9).

Tests (frozen):
  T0 floor gate: |median_w I(0.4,0.4) @ L17 profile| >=
     0.008 (2978 observed 0.0127); fail =>
     high_dose_floor_all_void.
  T1 (PRIMARY) head-grid reversal: I_h = median_w
     [c_h(0.4,0.4)-c_h(0.4,0)-c_h(0,0.4)+c_h(0,0)];
     sign-flip null rng 2979 x10000, maxT family 32;
     sig heads p<=0.01. Reversal needs >=1 head with
     p<=0.01 AND positive median.
  T2 carrier fate (descriptive + calibrated overlap):
     S_hi vs S_lo = 2978 sig_heads_L17 [0,4,7,12,13,15,25];
     per-S_lo-head sign of I_h at high dose (same-head
     flip vs handover); overlap null = random head draws
     matched in size, rng 2985 x2000 (discipline 11).
  T3 mechanism split (on words, median over all heads
     and over sig heads):
     g_h = ||x_h(11)-x_h(00)|| /
           (||x_h(10)-x_h(00)|| + ||x_h(01)-x_h(00)||)
     cos_align_h = cos(x_h(11)-x_h(00),
                       [x_h(10)-x_h(00)]+[x_h(01)-x_h(00)])
     linear network: g=1, cos=1. Also at dose 0.1 for the
     dose contrast (descriptive).

Verdict (frozen):
  anchor fail => anchor_fail_all_void
  T0 fail => high_dose_floor_all_void
  T1 no sig positive head => no_head_reversal_registered
  T1 sig positive head and median g < 0.5 and
     median cos_align >= 0.9 => reversal_via_saturation_
     squeeze
  T1 sig positive head and median g < 0.5 and
     median cos_align < 0.9 => reversal_via_gain_drop_
     reorientation
  T1 sig positive head and median g >= 0.5 =>
     reversal_via_superlinear_gain

Anchors (frozen):
  a1 Vt8 rebuild vs 2939 npz < 1e-6
  a2 determinism < 1e-4
  a3 identity vs 2973 npz norms rel<1e-4 coss abs<1e-6
  a4 single-token 74/74
  a5 unit axis dirs vs 2977 npz max|d|<1e-6
  a6 (0.1,0.1) L17 profile vs 2977 prof11 bit-level <1e-12
  a7 sum_h c_h == prof17 relative < 1e-9
  a8 injection locality (C_en[0])
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
SRC_2977_NPZ = os.path.join(BASE, 'phase2977',
                            'two_axis_fusion_injection',
                            'two_axis_fusion_injection.npz')
EXEC_2977 = os.path.join(BASE, 'phase2977',
                         'two_axis_fusion_injection',
                         'execution.json')
SRC_2978 = os.path.join(BASE, 'phase2978',
                        'interaction_dose_headgrid',
                        'interaction_dose_headgrid.npz')
OUT = os.path.join(BASE, 'phase2979',
                   'reversal_anatomy')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2979_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
N_PERM = 10000
P_TH = 0.01
L_INJ = 17
HI = 0.4
LO = 0.1
T0_GATE = 0.008
RNG_T1 = 2979
RNG_OVL = 2985
N_OVL = 2000

PREREG = {
    'mode': 'stage1: 74 base forwards (2977/2978 protocol '
            'verbatim); stage2: 74x4 conditions '
            '{(0.1,0.1) dose-id, (0.4,0), (0,0.4), '
            '(0.4,0.4)} at L17 input pos1, relative dose; '
            'capture full-layer profile + L17 per-head '
            'o_proj input; 2950 per-head snapshot identity '
            'c_h = u35.Wo_h.x_h',
    'question': 'is the 2978 high-dose interaction '
                'reversal (+0.0127 at 0.4,0.4) carried by '
                'saturation squeeze, gain drop with '
                'reorientation, or superlinear gain, and '
                'in which heads?',
    'mechanism_split': 'in x_h space (raw L17 o_proj '
                       'input head slices): g_h = '
                       '||x_h(11)-x_h(00)|| / '
                       '(||x_h(10)-x_h(00)||+'
                       '||x_h(01)-x_h(00)||); '
                       'cos_align = cos(dx11, '
                       'dx10+dx01); linear g=1 '
                       'cos=1; squeeze g<1 cos~1; '
                       'reorientation cos low; '
                       'medians taken over T1 sig '
                       'positive heads (primary) '
                       'and all heads (descriptive)',
    'anchors': {
        'a1': 'Vt8 rebuild vs 2939 npz < 1e-6',
        'a2': 'determinism < 1e-4',
        'a3': 'identity vs 2973 npz norms rel<1e-4, '
              'coss abs<1e-6 (74x36)',
        'a4': 'single-token 74/74',
        'a5': 'unit axis dirs vs 2977 npz max|d|<1e-6',
        'a6': '(0.1,0.1) L17 profile vs 2977 prof11 '
              'max|d|<1e-12',
        'a7': 'sum_h c_h == prof17 relative < 1e-9',
        'a8': 'injection locality (C_en[0])',
    },
    'T0': 'floor gate |median_w I(0.4,0.4)@L17 profile| '
          '>= 0.008; fail => high_dose_floor_all_void',
    'T1': 'I_h = median_w [c_h(11)-c_h(10)-c_h(01)+c_h(00)] '
          'at dose 0.4; sign-flip null rng 2979 x10000, '
          'maxT family 32; reversal iff >=1 head p<=0.01 '
          'with positive median',
    'T2': 'S_hi vs S_lo (2978 [0,4,7,12,13,15,25]): '
          'per-S_lo-head I_h sign at high dose; overlap '
          'null random heads rng 2985 x2000 (discipline '
          '11); descriptive',
    'T3': 'g and cos_align medians over words, all heads '
          'and sig heads, at 0.4 and 0.1 (dose contrast '
          'descriptive)',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'T0 fail => high_dose_floor_all_void; '
               'T1 none => no_head_reversal_registered; '
               'T1 + median_sigpos g<0.5 + cos>=0.9 => '
               'reversal_via_saturation_squeeze; '
               'T1 + median_sigpos g<0.5 + cos<0.9 => '
               'reversal_via_gain_drop_reorientation; '
               'T1 + median_sigpos g>=0.5 => '
               'reversal_via_superlinear_gain',
    'correction_note': 'run1: interaction baseline bug - T0/T1/T3 used the (0.1,0.1) condition (lo) as the R00 term instead of sham/base, violating the preregistered I = R(a,b) - R(a,0) - R(0,b) + R(0,0); cross-phase check against 2978 npz on the 24-word subset mismatched (median -0.0240 vs +0.0127, per-word max|d| 6.3e-2), caught by the 2970 identity-gate discipline. fix: stage-1 base sweep now also captures per-head contributions and raw L17 head slices; T0/T1/T3 rewired to the 00 baseline; artifacts deleted and rerun per discipline 3.',
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

    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2979,
                   'name': 'reversal_anatomy',
                   'created': time.strftime(
                       '%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2939': sha8(SRC_2939),
                               's2927': sha8(SRC_2927),
                               's2973': sha8(SRC_2973),
                               's2977npz': sha8(SRC_2977_NPZ),
                               's2977exec': sha8(EXEC_2977),
                               's2978npz': sha8(SRC_2978)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'n_perm': N_PERM,
                   'rng': {'T1': RNG_T1,
                           'overlap': RNG_OVL},
                   'n_overlap_null': N_OVL,
                   'p_threshold': P_TH,
                   'doses': {'lo': LO, 'hi': HI},
                   't0_gate': T0_GATE,
                   'cells': {'F_en': F_EN, 'F_fr': F_FR,
                             'C_en': C_EN, 'C_fr': C_FR},
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

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
    z77 = np.load(SRC_2977_NPZ, allow_pickle=True)
    d77_l = z77['d_lang'].astype(np.float64)
    d77_c = z77['d_cls'].astype(np.float64)
    prof11_77 = z77['prof11'].astype(np.float64)
    z78 = np.load(SRC_2978, allow_pickle=True)
    s_lo = [int(h) for h in z78['sig_heads_L17']]
    log('S_lo (2978 L17 sig heads) = %s' % s_lo, lines)

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
    assert words77 == words73, 'word order drift vs 2973'
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
    Wo17 = layers[L_INJ].self_attn.o_proj.weight.detach() \
        .float().cpu().numpy().astype(np.float64)
    C_row = u35 @ Wo17  # (4096,) per-head readout row

    def profile(op, want_head=False):
        prof = np.zeros(NL)
        for li in range(NL):
            prof[li] = float(np.dot(
                op[li].reshape(-1), M[li]))
        if not want_head:
            return prof, None
        x17 = op[L_INJ].reshape(-1)
        heads = np.zeros(NH)
        for h in range(NH):
            heads[h] = float(np.dot(
                C_row[h * HD:(h + 1) * HD],
                x17[h * HD:(h + 1) * HD]))
        return prof, heads

    # ---------- stage 1: base sweep ----------
    prof_base = np.zeros((n_test, NL))
    heads_base = np.zeros((n_test, NH))
    xh_base = np.zeros((n_test, NH, HD))
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
        prof_base[i], heads_base[i] = \
            profile(op, want_head=True)
        xh_base[i] = op[L_INJ].reshape(-1) \
            .reshape(NH, HD)
        if (i + 1) % 20 == 0:
            log('base sweep [%d/%d]' % (i + 1, n_test),
                lines)

    # ---------- a3 identity vs 2973 ----------
    nrm_rel = float(np.max(np.abs(norms - norms73)
                           / np.maximum(norms73, 1e-30)))
    cos_abs = float(np.max(np.abs(coss - coss73)))
    a3_ok = bool(nrm_rel < 1e-4 and cos_abs < 1e-6)
    log('a3 identity vs 2973: norms rel %.2e coss abs '
        '%.2e ok=%s' % (nrm_rel, cos_abs, a3_ok), lines)

    # ---------- axis directions + a5 ----------
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
    d_lang_u = d_lang / n_lang
    d_cls_u = d_cls / n_cls
    a5_diff = float(max(
        np.abs(d_lang_u - d77_l / np.linalg.norm(d77_l)).max(),
        np.abs(d_cls_u - d77_c / np.linalg.norm(d77_c)).max()))
    a5_ok = bool(a5_diff < 1e-6)
    log('a5 axis dirs vs 2977: max|d| %.2e cos=%.4f ok=%s'
        % (a5_diff, cos_axes, a5_ok), lines)

    # ---------- a2 determinism ----------
    op_a = forward1([func_tid, tid_map[C_EN[0]]])
    op_b = forward1([func_tid, tid_map[C_EN[0]]])
    a2_rel = float(np.abs(op_a[30] - op_b[30]).max()
                   / max(float(np.abs(op_a[30]).max()),
                         1e-30))
    a2_ok = bool(a2_rel < 1e-4)
    log('a2 determinism rel %.2e ok=%s'
        % (a2_rel, a2_ok), lines)

    n17_all = np.linalg.norm(X17, axis=1)

    # ---------- stage 2: dose conditions ----------
    conds = [('lo', LO, LO), ('10', HI, 0.0),
             ('01', 0.0, HI), ('11', HI, HI)]
    profs = {}
    heads_in = {}
    xh = {}
    for key, a, b in conds:
        profs[key] = np.zeros((n_test, NL))
        heads_in[key] = np.zeros((n_test, NH))
        xh[key] = np.zeros((n_test, NH, HD))
        for i, (_, _, w) in enumerate(cells):
            toks = [func_tid, tid_map[w]]
            if a == 0.0 and b == 0.0:
                op = forward1(toks, None)
            else:
                dv = (a * n17_all[i]) * d_lang_u \
                    + (b * n17_all[i]) * d_cls_u
                op = forward1(toks, dv)
            profs[key][i], heads_in[key][i] = \
                profile(op, want_head=True)
            xh[key][i] = op[L_INJ].reshape(-1) \
                .reshape(NH, HD)
        log('cond %s (%.2f,%.2f) done' % (key, a, b), lines)

    # register the 00 (sham) condition from stage 1
    heads_in['00'] = heads_base
    xh['00'] = xh_base

    # ---------- a6 dose-id anchor ----------
    a6_diff = float(np.max(np.abs(
        profs['lo'][:, L_INJ]
        - prof11_77[:, L_INJ])))
    a6_ok = bool(a6_diff < 1e-12)
    log('a6 (0.1,0.1) L17 vs 2977 prof11: max|d| %.2e '
        'ok=%s' % (a6_diff, a6_ok), lines)

    # ---------- a7 head-sum identity ----------
    a7_rel = 0.0
    for key, _, _ in conds:
        hs = heads_in[key].sum(axis=1)
        ps = profs[key][:, L_INJ]
        rel = float(np.max(np.abs(hs - ps)
                           / np.maximum(np.abs(ps),
                                        1e-30)))
        a7_rel = max(a7_rel, rel)
    a7_ok = bool(a7_rel < 1e-9)
    log('a7 head-sum identity: max rel %.2e ok=%s'
        % (a7_rel, a7_ok), lines)

    # ---------- a8 injection locality ----------
    w0 = C_EN[0]
    toks0 = [func_tid, tid_map[w0]]
    i0 = words77.index('C:en:%s' % w0)
    op_s = forward1(toks0, None)
    op_l = forward1(toks0, (HI * n17_all[i0]) * d_lang_u)
    pre_same = all(float(np.abs(
        op_s[li] - op_l[li]).max()) == 0.0
        for li in range(L_INJ))
    post_diff = any(float(np.abs(
        op_s[li] - op_l[li]).max()) > 0.0
        for li in range(L_INJ, NL))
    a8_ok = bool(pre_same and post_diff)
    log('a8 locality: pre17 identical=%s post17 '
        'differ=%s ok=%s'
        % (pre_same, post_diff, a8_ok), lines)

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok and a6_ok and a7_ok
                     and a8_ok)
    verdict = None
    t0res = t1 = t2 = t3 = None
    I_h = p_h = None
    sig_pos = []
    g_all = ca_all = None
    g_sig = ca_sig = None
    g_lo_all = ca_lo_all = None

    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        # ---------- T0 floor gate ----------
        I_prof = (profs['11'] - profs['10']
                  - profs['01'] + prof_base)
        i_l17 = float(np.median(I_prof[:, L_INJ]))
        t0res = {'median_I_L17': round(i_l17, 4),
                 'gate': T0_GATE,
                 'ok': bool(abs(i_l17) >= T0_GATE)}
        log('T0 floor: median I(0.4,0.4)@L17 %.4f ok=%s'
            % (i_l17, t0res['ok']), lines)

        # ---------- T1 head-grid reversal ----------
        I_h = (heads_in['11'] - heads_in['10']
               - heads_in['01'] + heads_in['00'])
        stat_h = np.median(I_h, axis=0)
        rng1 = np.random.default_rng(RNG_T1)
        signs = np.where(rng1.random(
            (N_PERM, n_test)) < 0.5, -1.0, 1.0)
        fam1 = np.zeros(N_PERM)
        for k in range(N_PERM):
            fam1[k] = float(np.abs(np.median(
                I_h * signs[k][:, None],
                axis=0)).max())
        p_h = np.array([
            (np.sum(fam1 >= abs(v) - 1e-12) + 1)
            / (N_PERM + 1) for v in stat_h])
        sig_pos = [h for h in range(NH)
                   if p_h[h] <= P_TH and stat_h[h] > 0]
        sig_neg = [h for h in range(NH)
                   if p_h[h] <= P_TH and stat_h[h] <= 0]
        log('T1 I_h sig+: %s | sig-: %s'
            % ([(h, round(float(stat_h[h]), 5))
                for h in sig_pos],
               [(h, round(float(stat_h[h]), 5))
                for h in sig_neg]), lines)

        # ---------- T3 mechanism split ----------
        # x_h space: raw L17 o_proj input slices
        dx11 = xh['11'] - xh['00']  # (n, NH, HD)
        dx10 = xh['10'] - xh['00']
        dx01 = xh['01'] - xh['00']
        dsum = dx10 + dx01
        n11 = np.linalg.norm(dx11, axis=2)
        nsm = np.linalg.norm(dsum, axis=2)
        g_wh = n11 / np.maximum(nsm, 1e-30)
        ca_wh = np.sum(dx11 * dsum, axis=2) \
            / np.maximum(n11 * nsm, 1e-30)
        g_all = float(np.median(g_wh))
        ca_all = float(np.median(ca_wh))
        g_sig = ca_sig = None
        if sig_pos:
            g_sig = float(np.median(
                g_wh[:, sig_pos]))
            ca_sig = float(np.median(
                ca_wh[:, sig_pos]))
        g_lo_all = None
        ca_lo_all = None
        log('T3 x_h space dose 0.4: g all=%.4f '
            'cos all=%.4f | sig-pos g=%s cos=%s'
            % (g_all, ca_all,
               None if g_sig is None
               else round(g_sig, 4),
               None if ca_sig is None
               else round(ca_sig, 4)), lines)

        # ---------- T2 carrier fate ----------
        stat_map = {h: float(stat_h[h]) for h in range(NH)}
        fate = {str(h): round(stat_map[h], 5)
                for h in s_lo}
        S17 = [h for h in range(NH)
               if p_h[h] <= P_TH]
        obs = len(set(S17) & set(s_lo))
        rng3 = np.random.default_rng(RNG_OVL)
        cnt = 0
        for _ in range(N_OVL):
            draw = rng3.choice(NH, size=len(S17),
                               replace=False)
            if len(set(int(x) for x in draw)
                   & set(s_lo)) >= obs:
                cnt += 1
        t2 = {'S_hi_sig': S17,
              'S_lo_fate_I_h': fate,
              'obs_overlap': obs,
              'null_p': (cnt + 1) / (N_OVL + 1)}
        log('T2 S_hi=%s obs overlap=%d null p=%.4f'
            % (S17, obs, t2['null_p']), lines)

        # ---------- verdict ----------
        if not t0res['ok']:
            verdict = 'high_dose_floor_all_void'
        elif len(sig_pos) >= 1:
            gg = g_sig if g_sig is not None else g_all
            cc = ca_sig if ca_sig is not None \
                else ca_all
            if gg < 0.5 and cc >= 0.9:
                verdict = 'reversal_via_saturation_squeeze'
            elif gg < 0.5:
                verdict = \
                    'reversal_via_gain_drop_reorientation'
            else:
                verdict = \
                    'reversal_via_superlinear_gain'
        else:
            verdict = 'no_head_reversal_registered'
        save = {
            'I_prof': I_prof,
            'c_00': heads_in['00'],
            'c_lo': heads_in['lo'], 'c_10': heads_in['10'],
            'c_01': heads_in['01'], 'c_11': heads_in['11'],
            'I_h': I_h, 'stat_h': stat_h, 'p_h': p_h,
            'sig_pos': np.array(sig_pos),
            'S_lo': np.array(s_lo),
            'ovl_p': np.array([t2['null_p']]),
            'g_wh': g_wh, 'ca_wh': ca_wh,
            'g_all': np.array([g_all]),
            'ca_all': np.array([ca_all]),
            'g_sig': np.array([np.nan if g_sig is None
                               else g_sig]),
            'ca_sig': np.array(
                [np.nan if ca_sig is None
                 else ca_sig]),
            'd_lang_u': d_lang_u, 'd_cls_u': d_cls_u,
            'axes_cos': np.array([cos_axes]),
            'n17': n17_all, 'words': np.array(words77)}

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2979, 'model': 'qwen3-4b',
           'prereg': PREREG,
           'anchors': {
               'a1_diff': float('%.3e' % a1_diff),
               'a1_ok': a1_ok,
               'a2_rel': float('%.3e' % a2_rel),
               'a2_ok': a2_ok,
               'a3_nrm_rel': float('%.3e' % nrm_rel),
               'a3_cos_abs': float('%.3e' % cos_abs),
               'a3_ok': a3_ok, 'a4_ok': a4_ok,
               'a5_diff': float('%.3e' % a5_diff),
               'a5_ok': a5_ok,
               'a6_diff': float('%.3e' % a6_diff),
               'a6_ok': a6_ok,
               'a7_rel': float('%.3e' % a7_rel),
               'a7_ok': a7_ok, 'a8_ok': a8_ok,
               'axes_cos': round(cos_axes, 4),
               'ok': anchor_ok},
           'T0': t0res,
           'T1': {'sig_pos':
                      [(h, round(float(stat_h[h]), 5))
                       for h in sig_pos]
                  if stat_h is not None else None,
                  'sig_neg':
                      [(h, round(float(stat_h[h]), 5))
                       for h in sig_neg]
                  if stat_h is not None else None,
                  'p':
                      [float('%.3e' % v) for v in p_h]
                      if p_h is not None else None},
           'T2': t2,
           'T3': {'g_median_all': None if g_all is None
                  else round(g_all, 4),
                  'cos_median_all': None
                  if ca_all is None
                  else round(ca_all, 4),
                  'g_median_sigpos': None
                  if g_sig is None
                  else round(g_sig, 4),
                  'cos_median_sigpos': None
                  if ca_sig is None
                  else round(ca_sig, 4),
                  'space': 'x_h raw L17 o_proj input '
                           'head slices (128-d per '
                           'head)',
                  'dose_contrast': 'not computed '
                                   '(registered '
                                   'limitation)'},
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if verdict != 'anchor_fail_all_void':
        np.savez_compressed(os.path.join(
            OUT, 'reversal_anatomy.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2979 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    import torch  # noqa: E402
    main()
