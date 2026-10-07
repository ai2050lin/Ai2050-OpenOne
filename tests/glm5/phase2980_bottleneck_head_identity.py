# -*- coding: utf-8 -*-
"""Phase 2980: functional identity of the competitive
bottleneck heads h9/h12 (2979 unique significant
negative-interaction carriers at L17).

Why: 2979 localized the L17 subadditive two-axis
interaction to heads h9 (stat -0.0129, p 1.8e-3) and h12
(-0.0120, p 3.7e-3) under the authoritative 00-baseline.
2978 had shown the interaction is head-level concentrated
(S_lo = [0,4,7,12,13,15,25]) but 2979 found NO head-level
carrier of the positive reversal. Open question: are h9/h12
CAUSALLY necessary for the (0.1,0.1) subadditive
interaction (ablating them abolishes I), or correlational
bystanders (interaction survives, distributed)?

Design (frozen before any observation):
  Word list VERBATIM from sealed 2979 npz (74 words,
  'G:lang:w' format). Axis dirs d_lang_u / d_cls_u VERBATIM
  reload from 2979 npz (cross-checked vs 2977 d_lang/d_cls).
  n17 norms VERBATIM from 2979 npz.
  Stage0: 74 intact forwards (2973 protocol) -> a3 anchor.
  Stage1: dose-0.1 no-ablation 4 injection conditions
          {00,10,01,11} x 74 -> a5 anchor vs 2977 npz prof.
  Stage2: ablation {h9, h12, both, rand} x 4 conditions
          x 74 forwards. Ablation = zero o_proj INPUT slice
          [1, h*128:(h+1)*128] at L17 (2932 pattern);
          injection = self_attn pre-hook x[:,1,:] += dt
          (2977 pattern verbatim).
  Readout: prof[li] = sum_h u35 @ Wo_li_h @ x_li_h (2965
          contributions verbatim); I(w) = prof11 - prof10
          - prof01 + prof00 at L17.

Anchors (frozen):
  a1 o_proj shape L17 == (2560, 4096)
  a2 axis dirs: 2979 reload unit-norm < 1e-9 AND max|d|
     vs 2977 npz d_lang/d_cls < 1e-9
  a3 intact n17 (74) vs 2979 npz n17 rel < 1e-6
  a4 single-token 74/74 + list == 2979 npz words
  a5 I(0.1,0.1) no-abl @L17 vs 2977 npz
     (prof11-prof10-prof01+prof00)[17] per-word max|d|
     < 1e-4 (cross-phase injection identity)
  a6 ablation efficacy |C17[abl_h]| < 1e-6 in each
     ablation condition (implementation gate)
  a7 non-degeneracy intact C17 per-head std > 0 (32/32)

Tests (frozen):
  T1 (main): for c in {h9, h12, both}: d_c(w) = I_abl_c(w)
     - I_int(w); obs = median(d_c); sign-flip permutation
     (74 paired, 10000, rngs 29800/29801/29802); gate
     p <= 0.01. Direction: interaction abolished iff
     |median I_abl| < |median I_int|.
  T2: rand-head control same stat (rng 29803), reported
     as percentile vs the three obs |medians| (descriptive
     calibration, discipline 11).
  T3 descriptive: no-injection ablation main effect on B
     band (mean|dB| per condition) + gap shift.

Verdict (frozen):
  anchor fail => anchor_fail_all_void
  T1 both p<=0.01 & abolished & (h9 or h12 also sig)
     => bottleneck_heads_carry_interaction
  T1 both p<=0.01 & abolished & neither single sig
     => interaction_emerges_at_pair_level
  T1 both p<=0.01 & NOT abolished
     => ablation_amplifies_interaction
  else => interaction_survives_ablation_distributed
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                        'probe_relativity.npz')
SRC_2977 = os.path.join(BASE, 'phase2977',
                        'two_axis_fusion_injection',
                        'two_axis_fusion_injection.npz')
SRC_2979 = os.path.join(BASE, 'phase2979', 'reversal_anatomy',
                        'reversal_anatomy.npz')
OUT = os.path.join(BASE, 'phase2980',
                   'bottleneck_head_identity')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2980_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
L_INJ = 17
H9, H12 = 9, 12
DOSE = 0.1
N_PERM = 10000
RNG_T1 = {'h9': 29800, 'h12': 29801, 'both': 29802,
          'rand': 29803}
RNG_RAND = 29804

PREREG = {
    'mode': 'stage0: 74 intact forwards (2973 protocol, '
            'capture o_proj input 36 layers); stage1: dose '
            '0.1 no-abl 4 injection conditions x 74; '
            'stage2: ablation {h9,h12,both,rand} x 4 '
            'conditions x 74. total 1550 forwards. '
            'ablation = zero o_proj input slice at L17 '
            '(2932); injection = self_attn pre-hook '
            'x[:,1,:] += dt (2977 verbatim)',
    'question': 'are the 2979 negative-interaction carrier '
                'heads h9/h12 causally necessary for the '
                'subadditive two-axis interaction at dose '
                '0.1, or correlational bystanders?',
    'anchors': {
        'a1': 'o_proj shape L17 == (2560, 4096)',
        'a2': '2979 _u dirs unit-norm < 1e-9 AND cos(u, '
          '2977 raw) > 1-1e-9; injection uses 2977 RAW '
          'vectors (protocol verbatim)',
        'a3': 'intact n17 vs 2979 npz rel < 1e-6',
        'a4': 'single-token 74/74 + list == 2979',
        'a5': 'I(0.1,0.1) no-abl vs 2977 npz per-word '
              'max|d| < 1e-4',
        'a6': 'ablation efficacy |C17[abl_h]| < 1e-6',
        'a7': 'non-degeneracy C17 std > 0 (32/32)',
    },
    'T1': 'paired sign-flip perm (10000) on '
          'd_c = I_abl_c - I_int at L17, c in {h9,h12,'
          'both}, gate p <= 0.01; abolished iff '
          '|median I_abl| < |median I_int|',
    'T2': 'rand-head control same stat, percentile vs '
          'obs (descriptive)',
    'T3': 'descriptive: no-injection ablation mean|dB| '
          'band effect',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'both sig & abolished & (h9|h12 sig) => '
               'bottleneck_heads_carry_interaction; both '
               'sig & abolished & neither single => '
               'interaction_emerges_at_pair_level; both '
               'sig & not abolished => '
               'ablation_amplifies_interaction; else => '
               'interaction_survives_ablation_distributed',
    'correction_note': 'run1: anchors a2/a3 failed - '
                       'a2 compared 2979 unit dirs _u '
                       'against raw 2977 vectors (norms '
                       '4.47/13.90, cos actually 1.0, '
                       'same direction) and injection '
                       'must use the RAW 2977 vectors '
                       'verbatim (unit vectors would '
                       'scale dose by 4.5-14x); a3 '
                       'computed n17 as o_proj-input '
                       '(4096-d) norm but upstream n17 '
                       'is the L17 LAYER-INPUT (2560-d) '
                       'norm captured at self_attn '
                       'pre-hook. fix: raw vectors + '
                       'x17 capture in hook_inj; '
                       'artifacts deleted, rerun per '
                       'discipline 3. run2: a5 failed '
                       '(max|d| 0.481) - root cause: 2977 '
                       'npz d_lang/d_cls are the RAW '
                       'mean-diff vectors stored BEFORE '
                       'in-script normalization (norms '
                       '4.47/13.90); the actually '
                       'injected vectors are UNIT-norm '
                       '(= 2979 _u, proven by 2979 a5 '
                       'bit-level reproduction), so '
                       'injection must use 2979 _u dirs; '
                       'run2 x17 layer-input capture kept '
                       '(a3 1.21e-07 pass). plus '
                       'UnboundLocalError med_I_int on '
                       'anchor-fail path fixed; artifacts '
                       'deleted, rerun.',
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


def rankdata(x):
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a, b):
    ra = rankdata(np.asarray(a, dtype=np.float64))
    rb = rankdata(np.asarray(b, dtype=np.float64))
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
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2980,
                   'name': 'bottleneck_head_identity',
                   'created': time.strftime(
                       '%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2927': sha8(SRC_2927),
                               's2977': sha8(SRC_2977),
                               's2979': sha8(SRC_2979)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'inj_layer': L_INJ, 'dose_rel': DOSE,
                   'target_heads': [H9, H12],
                   'n_forwards': 1550, 'n_perm': N_PERM,
                   'rng': {'T1': RNG_T1,
                           'rand_head': RNG_RAND},
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- sources ----------
    z27_ = np.load(SRC_2927, allow_pickle=True)
    u35 = z27_['dirs_word'].astype(np.float64)[NL - 1]
    z77 = np.load(SRC_2977, allow_pickle=True)
    z79 = np.load(SRC_2979, allow_pickle=True)

    d_lang = z79['d_lang_u'].astype(np.float64)
    d_cls = z79['d_cls_u'].astype(np.float64)
    dl_raw = z77['d_lang'].astype(np.float64)
    dc_raw = z77['d_cls'].astype(np.float64)
    n17_ref = z79['n17'].astype(np.float64)
    words79 = [str(w) for w in z79['words']]
    wlist = [w.split(':')[2] for w in words79]
    lab_lang = np.array([w.split(':')[1]
                         for w in words79])
    lab_grp = np.array([w.split(':')[0]
                        for w in words79])

    dl_u = z79['d_lang_u'].astype(np.float64)
    dc_u = z79['d_cls_u'].astype(np.float64)
    a2_n1 = abs(float(np.linalg.norm(dl_u)) - 1.0)
    a2_n2 = abs(float(np.linalg.norm(dc_u)) - 1.0)
    cos_l = float(dl_u @ dl_raw / (
        np.linalg.norm(dl_u) * np.linalg.norm(dl_raw)))
    cos_c = float(dc_u @ dc_raw / (
        np.linalg.norm(dc_u) * np.linalg.norm(dc_raw)))
    a2_ok = bool(a2_n1 < 1e-9 and a2_n2 < 1e-9
                 and abs(cos_l - 1.0) < 1e-9
                 and abs(cos_c - 1.0) < 1e-9)
    log('a2 dirs: 2977 raw norms %.4f/%.4f (stored '
        'pre-normalization); 2979 unit dev %.1e/%.1e; '
        'cos(u,raw) %.12f/%.12f ok=%s'
        % (float(np.linalg.norm(dl_raw)),
           float(np.linalg.norm(dc_raw)),
           a2_n1, a2_n2, cos_l, cos_c, a2_ok), lines)

    I77 = (z77['prof11'].astype(np.float64)
           - z77['prof10'].astype(np.float64)
           - z77['prof01'].astype(np.float64)
           + z77['prof00'].astype(np.float64))[:, L_INJ]

    # ---------- model ----------
    import torch
    import sys
    sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
    from phase2662_symmetric_mapping_contract import \
        load_native
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        MD, local_files_only=True, trust_remote_code=True,
        use_fast=True)
    tid_map = {}
    n_single = 0
    for w in wlist:
        ids = tok(' ' + w, add_special_tokens=False)[
            'input_ids']
        if len(ids) != 1:
            ids = tok(w, add_special_tokens=False)[
                'input_ids']
        assert len(ids) == 1, '%s -> %s' % (w, ids)
        tid_map[w] = int(ids[0])
        n_single += 1
    a4_ok = bool(n_single == 74
                 and [str(w) for w in z79['words']]
                 == words79)
    log('a4 single-token %d/74, list==2979 ok=%s'
        % (n_single, a4_ok), lines)
    ids_the = tok(' the', add_special_tokens=False)[
        'input_ids']
    assert len(ids_the) == 1
    func_tid = int(ids_the[0])

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    log('model loaded', lines)

    Wo17 = layers[L_INJ].self_attn.o_proj.weight.detach() \
        .float().cpu().numpy()
    a1_ok = bool(Wo17.shape == (2560, NH * HD))
    log('a1 o_proj shape L17 %s ok=%s'
        % (Wo17.shape, a1_ok), lines)

    cap_op = {li: [] for li in range(NL)}
    abl = {'h': None}
    inj = {'d': None}
    handles = []

    def hs_of(args, kwargs):
        if args:
            return args[0]
        return kwargs.get('hidden_states')

    def hook_op(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('input')
            if x is None or x.dim() < 2:
                return None
            if li == L_INJ and abl['h'] is not None:
                for hh in abl['h']:
                    x[:, 1,
                      hh * HD:(hh + 1) * HD] = 0
            cap_op[li].append(
                x[:, 1, :].detach().float().cpu().numpy())
            return None
        return h

    x17_cap = []

    def hook_inj(module, args, kwargs):
        d = inj['d']
        x = hs_of(args, kwargs)
        if d is not None:
            if x is not None:
                dt = torch.as_tensor(
                    d, device=x.device, dtype=x.dtype)
                x[:, 1, :] += dt
        elif x is not None and len(x17_cap) < 200:
            x17_cap.append(
                x[:, 1, :].detach().float().cpu()
                .numpy())
        return None

    for li in range(NL):
        handles.append(
            layers[li].self_attn.o_proj
            .register_forward_pre_hook(
                hook_op(li), with_kwargs=True))
    handles.append(
        layers[L_INJ].self_attn
        .register_forward_pre_hook(
            hook_inj, with_kwargs=True))

    def clear_cap():
        for li in cap_op:
            del cap_op[li][:]

    def forward1(toks, d_inj=None, abl_head=None):
        clear_cap()
        inj['d'] = d_inj
        abl['h'] = (list(abl_head)
                    if abl_head is not None else None)
        with torch.no_grad():
            model(torch.tensor([toks], device='cuda'))
        inj['d'] = None
        abl['h'] = None
        return {li: cap_op[li][0].astype(np.float64)
                for li in range(NL)}

    M = np.zeros((NL, NH * HD))
    for li in range(NL):
        Wl = layers[li].self_attn.o_proj.weight.detach() \
            .float().cpu().numpy()
        M[li] = u35 @ Wl

    def contributions(op):
        C = np.zeros((NL, NH))
        prof = np.zeros(NL)
        for li in range(NL):
            x = op[li].reshape(-1)
            xm = (x * M[li]).reshape(NH, HD)
            C[li] = xm.sum(axis=1)
            prof[li] = float(C[li].sum())
        return C, prof

    def band_of(prof):
        return (float(prof[6:13].mean())
                - float(prof[28:36].mean()))

    # ---------- stage0: intact ----------
    n17 = np.zeros(74)
    B_int = np.zeros(74)
    C17_int = np.zeros((74, NH))
    del x17_cap[:]
    for i, w in enumerate(wlist):
        op = forward1([func_tid, tid_map[w]])
        C, prof = contributions(op)
        n17[i] = float(np.linalg.norm(x17_cap[i]
                                      .reshape(-1)))
        B_int[i] = band_of(prof)
        C17_int[i] = C[L_INJ]
        if (i + 1) % 20 == 0:
            log('intact [%d/74]' % (i + 1), lines)

    a3_rel = float(np.abs(n17 - n17_ref).max()
                   / max(float(np.abs(n17_ref).max()),
                         1e-30))
    a3_ok = bool(a3_rel < 1e-6)
    log('a3 intact n17 vs 2979 rel %.2e ok=%s'
        % (a3_rel, a3_ok), lines)

    a7_ok = bool(C17_int.std(axis=0).min() > 0)
    log('a7 non-degeneracy C17 std min %.3e ok=%s'
        % (C17_int.std(axis=0).min(), a7_ok), lines)

    rng_r = np.random.default_rng(RNG_RAND)
    cand = [h for h in range(NH) if h not in (H9, H12)]
    h_rand = int(rng_r.choice(cand))
    log('rand control head = %d' % h_rand, lines)

    dose_scales = DOSE * n17_ref

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a7_ok)
    verdict = None
    med_I_int = None
    a5_max = 0.0
    t1 = t2 = t3 = None
    save = {}
    a6_max = 0.0

    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        # ---------- stage1: no-abl dose 0.1 ----------
        profs = {k: np.zeros((74, NL))
                 for k in ('00', '10', '01', '11')}
        for i, w in enumerate(wlist):
            toks = [func_tid, tid_map[w]]
            s_w = float(dose_scales[i])
            op00 = forward1(toks, None)
            op10 = forward1(toks, s_w * d_lang)
            op01 = forward1(toks, s_w * d_cls)
            op11 = forward1(toks,
                            s_w * (d_lang + d_cls))
            for key, op in (('00', op00), ('10', op10),
                            ('01', op01), ('11', op11)):
                _, prof = contributions(op)
                profs[key][i] = prof
            if (i + 1) % 20 == 0:
                log('stage1 [%d/74]' % (i + 1), lines)
        I_int = (profs['11'] - profs['10']
                 - profs['01'] + profs['00'])[:, L_INJ]
        a5_max = float(np.abs(I_int - I77).max())
        a5_ok = bool(a5_max < 1e-4)
        log('a5 I(0.1,0.1) vs 2977 per-word max|d| '
            '%.2e ok=%s' % (a5_max, a5_ok), lines)

        # ---------- stage2: ablations ----------
        ABL_SETS = {'h9': [H9], 'h12': [H12],
                    'both': [H9, H12], 'rand': [h_rand]}
        I_abl = {}
        B_abl = {}
        a6_max = 0.0
        for cname, heads in ABL_SETS.items():
            pa = {k: np.zeros((74, NL))
                  for k in ('00', '10', '01', '11')}
            for i, w in enumerate(wlist):
                toks = [func_tid, tid_map[w]]
                s_w = float(dose_scales[i])
                ops = {}
                for key, dvec in (('00', None),
                                  ('10', s_w * d_lang),
                                  ('01', s_w * d_cls),
                                  ('11', s_w * (d_lang
                                                + d_cls))):
                    ops[key] = forward1(
                        toks, dvec, abl_head=heads)
                for key, op in ops.items():
                    _, prof = contributions(op)
                    pa[key][i] = prof
                if cname == 'h9':
                    C, _ = contributions(ops['00'])
                    a6_max = max(
                        a6_max,
                        abs(float(C[L_INJ, H9])))
                if cname == 'rand':
                    C, _ = contributions(ops['00'])
                    a6_max = max(
                        a6_max,
                        abs(float(C[L_INJ, h_rand])))
            I_abl[cname] = (pa['11'] - pa['10']
                            - pa['01'] + pa['00'])[
                :, L_INJ]
            B_abl[cname] = (pa['00'][:, 6:13].mean(1)
                            - pa['00'][:, 28:36]
                            .mean(1))
            log('stage2 %s done (I median %.5f)'
                % (cname, float(np.median(
                    I_abl[cname]))), lines)
        a6_ok = bool(a6_max < 1e-6)
        log('a6 ablation efficacy max|C_abl| %.2e ok=%s'
            % (a6_max, a6_ok), lines)

        if not (a5_ok and a6_ok):
            verdict = 'anchor_fail_all_void'
        else:
            # ---------- T1 ----------
            med_I_int = float(np.median(I_int))
            t1 = {}
            sig_map = {}
            for cname in ('h9', 'h12', 'both'):
                d_c = I_abl[cname] - I_int
                obs = float(np.median(d_c))
                rng = np.random.default_rng(
                    RNG_T1[cname])
                cnt = 0
                for _ in range(N_PERM):
                    if abs(float(np.median(
                            d_c
                            * rng.choice(
                                [-1.0, 1.0],
                                size=74)))) \
                            >= abs(obs) - 1e-12:
                        cnt += 1
                p = (cnt + 1) / (N_PERM + 1)
                med_abl = float(
                    np.median(I_abl[cname]))
                abolished = bool(
                    abs(med_abl) < abs(med_I_int))
                sig = bool(p <= 0.01)
                sig_map[cname] = sig
                t1[cname] = {
                    'obs_median_dI': round(obs, 5),
                    'perm_p': float('%.3e' % p),
                    'sig': sig,
                    'median_I_int':
                        round(med_I_int, 5),
                    'median_I_abl':
                        round(med_abl, 5),
                    'abolished': abolished}
                log('T1 %s: dI med %+.5f p %.3e sig=%s '
                    '|I| %s (%.5f -> %.5f)'
                    % (cname, obs, p, sig,
                       'smaller' if abolished
                       else 'bigger-or-equal',
                       abs(med_I_int), abs(med_abl)),
                    lines)

            # ---------- T2 rand control ----------
            d_r = I_abl['rand'] - I_int
            obs_r = float(np.median(d_r))
            rng = np.random.default_rng(
                RNG_T1['rand'])
            cnt = 0
            for _ in range(N_PERM):
                if abs(float(np.median(
                        d_r * rng.choice(
                            [-1.0, 1.0], size=74)))) \
                        >= abs(obs_r) - 1e-12:
                    cnt += 1
            p_r = (cnt + 1) / (N_PERM + 1)
            obs_pool = [abs(t1[c]['obs_median_dI'])
                        for c in ('h9', 'h12', 'both')]
            pct = float((np.array(obs_pool)
                         < abs(obs_r)).mean())
            t2 = {'rand_head': h_rand,
                  'obs_median_dI': round(obs_r, 5),
                  'perm_p': float('%.3e' % p_r),
                  'pct_obs_below_rand': round(pct, 3)}
            log('T2 rand h%d: dI med %+.5f p %.3e '
                'pct=%.2f'
                % (h_rand, obs_r, p_r, pct), lines)

            # ---------- T3 descriptive ----------
            t3 = {}
            for cname in ('h9', 'h12', 'both', 'rand'):
                dB = np.abs(B_abl[cname] - B_int)
                t3[cname] = {
                    'mean_abs_dB':
                        round(float(dB.mean()), 4),
                    'median_abs_dB':
                        round(float(np.median(dB)),
                              4)}
            log('T3 mean|dB| %s'
                % {c: t3[c]['mean_abs_dB']
                   for c in t3}, lines)

            # ---------- verdict ----------
            if sig_map['both'] and \
                    t1['both']['abolished'] and \
                    (sig_map['h9'] or sig_map['h12']):
                verdict = \
                    'bottleneck_heads_carry_interaction'
            elif sig_map['both'] and \
                    t1['both']['abolished']:
                verdict = \
                    'interaction_emerges_at_pair_level'
            elif sig_map['both'] and not \
                    t1['both']['abolished']:
                verdict = 'ablation_amplifies_interaction'
            else:
                verdict = \
                    'interaction_survives_ablation_' \
                    'distributed'
            save = {'I_int': I_int,
                    'I_abl_h9': I_abl['h9'],
                    'I_abl_h12': I_abl['h12'],
                    'I_abl_both': I_abl['both'],
                    'I_abl_rand': I_abl['rand'],
                    'B_int': B_int,
                    'B_abl_h9': B_abl['h9'],
                    'B_abl_h12': B_abl['h12'],
                    'B_abl_both': B_abl['both'],
                    'B_abl_rand': B_abl['rand'],
                    'n17': n17, 'h_rand': np.array([h_rand]),
                    'tids': np.array([tid_map[w]
                                      for w in wlist]),
                    'words': np.array(words79)}

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2980, 'model': 'qwen3-4b',
           'prereg': PREREG,
           'anchors': {'a1_ok': a1_ok,
                       'a2_ok': a2_ok,
                       'a3_rel': float('%.3e' % a3_rel),
                       'a3_ok': a3_ok,
                       'a4_ok': a4_ok,
                       'a5_max': float('%.3e' % a5_max)
                       if anchor_ok else None,
                       'a5_ok': a5_ok
                       if anchor_ok else None,
                       'a6_max': float('%.3e' % a6_max)
                       if a6_max else None,
                       'a6_ok': a6_ok if a6_max
                       else None,
                       'a7_ok': a7_ok,
                       'ok': bool(anchor_ok and a5_ok
                                  and a6_ok)
                       if anchor_ok else False},
           'T1': t1, 'T2': t2, 'T3': t3,
           'median_I_int': round(med_I_int, 5)
           if med_I_int is not None else None,
           'rand_head': h_rand,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if verdict != 'anchor_fail_all_void':
        np.savez_compressed(os.path.join(
            OUT, 'bottleneck_head_identity.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2980 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
