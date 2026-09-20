# -*- coding: utf-8 -*-
"""Phase 2949: head-level dose sufficiency of the regime switch.

Why: 2947 localized the switch effect to few high-leverage
heads per layer (top1 share 0.18-0.25); 2948 showed the head
rank order is predicted by linear W_ov readout gain (rho
0.38-0.56, perm-p < 0.02). Open question: are the top-W_ov-
gain heads SUFFICIENT to carry the switch (switch fires when
only they remain) and NECESSARY (ablating them blocks it)?

Mode: ONE model (qwen3-4b), forward family. xdir injection
(v1/v2/v5, 2942/2945 verbatim) at switch doses L17 s=1.0 and
L16 s=2.0, crossed with head-GROUP ablation at the injected
layer's o_proj input (2947 run2 verbatim mechanics):
  I0  inject, no ablation            (switch reference)
  I1  inject, ablate top5_g heads    (necessity)
  I2  inject, ablate other 27 heads  (sufficiency)
  B1  no inject, ablate top5_g       (baseline gate)
  B2  no inject, ablate other 27     (baseline gate)
  B0  no inject, no ablation         (= func baseline)
K=3 same-session repeats, median readouts.

top5_g frozen from 2948 result.json D2_top5 BEFORE any
observation of this phase:
  L17: [0, 7, 24, 22, 19]
  L16: [13, 16, 1, 17, 6]

Main tests (frozen; switch threshold 100.0 is the 2945
first-crossing threshold, baseline gate 150.0 vs B0 ~185.7):
  T1 sufficiency L17: sep(B2_17) > 150 AND sep(I2_17) < 100
  T2 necessity   L17: sep(I1_17) - sep(I0_17) > 20
  T3 sufficiency L16: sep(B2_16) > 150 AND sep(I2_16) < 100
  T4 necessity   L16: sep(I1_16) - sep(I0_16) > 20
  (necessity margin 20 vs single-head maxD 17.5/50.7 and
   control max|Cc| 1.75 from 2947)

Verdict (frozen):
  anchor fail            => anchor_fail_all_void
  T1&T2&T3&T4            => head_dose_sufficient_necessary
  T1&T2 xor T3&T4        => head_dose_layer_asymmetric
  (T1&T3) & not(T2&T4)   => head_dose_sufficient_only
  (T2&T4) & not(T1&T3)   => head_dose_necessary_only
  else                   => head_dose_not_carried

Anchors (frozen): a1 dirs rebuild < 1e-5 (continuous-forward
chain); a2 baseline determinism < 1e-4; a3 Vt8 < 1e-6;
a4/a5 proj vs 2935 < 1e-4; a6 sep_func > 0; a7 xdir
self-check < 1e-9; a8 same-session determinism < 1e-6;
a9 ablation-slice self-check + o_proj in_features gate;
a10 REF17 sep vs 2945 D1_sep[L17]@1.0 < 1.0; a11 REF16 vs
2945 D1_sep[L16]@2.0 < 1.0; a12 group-ablation mask
self-check on synthetic tensor < 1e-12.

Output: phase2949/head_dose_sufficiency/.
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2887 = os.path.join(BASE, 'phase2887', 'language_axis_mlp',
                        'language_axis_mlp.npz')
SRC_2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                        'probe_relativity.npz')
SRC_2935 = os.path.join(BASE, 'phase2935', 'null_amp_anatomy',
                        'null_amp_anatomy.npz')
SRC_2939 = os.path.join(BASE, 'phase2939', 'rotation_target',
                        'rotation_target.npz')
SRC_2945 = os.path.join(BASE, 'phase2945', 'threshold_curves',
                        'result.json')
SRC_2948 = os.path.join(BASE, 'phase2948', 'wov_head_gain',
                        'result.json')
OUT = os.path.join(BASE, 'phase2949', 'head_dose_sufficiency')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2949_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
SEED = 2896
NH, HD = 32, 128
NL = 36
VOCAB = 151936
S_IDX = (0, 1, 4)
K_REPEAT = 3
LI_SWITCH = 17
LI_GRAD = 16
S_SWITCH = 1.0
S_GRAD = 2.0
TOP5_G = {LI_SWITCH: [0, 7, 24, 22, 19],
          LI_GRAD: [13, 16, 1, 17, 6]}
SWITCH_THR = 100.0
BASELINE_MIN = 150.0
NEC_MARGIN = 20.0
A_REF_TOL = 1.0

PREREG = {
    'mode': 'forward family: xdir injection at L17 s=1.0 and '
            'L16 s=2.0 crossed with head-group ablation at '
            'the injected layer o_proj input {I0 no-abl, I1 '
            'abl top5_g, I2 abl other-27} plus uninjected '
            'baselines {B0, B1, B2}; K=3 same-session '
            'repeats, median readouts',
    'question': 'are the top-W_ov-gain heads (2948) '
                'sufficient to carry the switch and '
                'necessary for it?',
    'top5_g_frozen': {str(k): v for k, v in TOP5_G.items()},
    'anchors': {
        'a1': 'dirs_word rebuild vs 2927 npz < 1e-5',
        'a2': 'func baseline determinism < 1e-4',
        'a3': 'Vt8 rebuild vs 2939 npz < 1e-6',
        'a4': 'proj_func vs 2935 s_base[func] < 1e-4',
        'a5': 'proj_null0 vs 2935 s_base[null0] < 1e-4',
        'a6': 'func separation > 0',
        'a7': 'xdir construction self-check < 1e-9',
        'a8': 'same-session repeat determinism < 1e-6',
        'a9': 'ablation-slice self-check < 1e-12 + o_proj '
              'in_features == 4096 gate',
        'a10': 'REF17 sep vs 2945 L17@1.0 < 1.0',
        'a11': 'REF16 sep vs 2945 L16@2.0 < 1.0',
        'a12': 'group-ablation mask self-check < 1e-12',
    },
    'T1': 'sufficiency L17: sep(B2_17) > 150 AND '
          'sep(I2_17) < 100',
    'T2': 'necessity L17: sep(I1_17) - sep(I0_17) > 20',
    'T3': 'sufficiency L16: sep(B2_16) > 150 AND '
          'sep(I2_16) < 100',
    'T4': 'necessity L16: sep(I1_16) - sep(I0_16) > 20',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'T1&T2&T3&T4 => '
               'head_dose_sufficient_necessary; exactly one '
               'layer with both => '
               'head_dose_layer_asymmetric; sufficiency '
               'both, necessity not both => '
               'head_dose_sufficient_only; necessity both, '
               'sufficiency not both => '
               'head_dose_necessary_only; else => '
               'head_dose_not_carried',
}


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-30)


def log(msg, lines):
    lines.append(msg)
    print(msg, flush=True)


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2949,
                   'name': 'head_dose_sufficiency',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2887': sha8(SRC_2887),
                               's2927': sha8(SRC_2927),
                               's2935': sha8(SRC_2935),
                               's2939': sha8(SRC_2939),
                               's2945': sha8(SRC_2945),
                               's2948': sha8(SRC_2948)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'seed': SEED,
                   'li_switch': LI_SWITCH, 'li_grad': LI_GRAD,
                   's_switch': S_SWITCH, 's_grad': S_GRAD,
                   'top5_g': {str(k): list(v)
                              for k, v in TOP5_G.items()},
                   'switch_thr': SWITCH_THR,
                   'baseline_min': BASELINE_MIN,
                   'nec_margin': NEC_MARGIN,
                   'a_ref_tol': A_REF_TOL,
                   'k_repeat': K_REPEAT,
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- sources ----------
    z87 = np.load(SRC_2887, allow_pickle=True)
    words = [tuple(str(w).split(':')) for w in z87['words']]
    lab_lang = np.asarray(z87['labels_lang']).astype(int)
    n_words = len(words)
    assert n_words == 57

    z27 = np.load(SRC_2927, allow_pickle=True)
    dirs_word_27 = z27['dirs_word'].astype(np.float64)

    z35 = np.load(SRC_2935, allow_pickle=True)
    conds35 = [str(s) for s in z35['cond_names']]
    s_base_35 = z35['s_base'].astype(np.float64)
    ifu35 = conds35.index('func')
    in035 = conds35.index('null0')

    z39 = np.load(SRC_2939, allow_pickle=True)
    Vt8_39 = z39['Vt8'].astype(np.float64)
    coords_39 = z39['coords'].astype(np.float64)
    conds39 = [str(s) for s in z39['cond_names']]
    dcks_39 = coords_39[conds39.index('null0')] \
        - coords_39[conds39.index('func')]

    r45 = json.load(open(SRC_2945, encoding='utf-8'))
    sep45 = r45['D1_sep']
    r48 = json.load(open(SRC_2948, encoding='utf-8'))
    t48 = r48['D2_top5']
    assert list(t48['top5_g_L17']) == TOP5_G[LI_SWITCH], \
        'top5_g L17 drifted from 2948'
    assert list(t48['top5_g_L16']) == TOP5_G[LI_GRAD], \
        'top5_g L16 drifted from 2948'
    log('sources ok (top5_g re-verified against 2948)',
        lines)

    # ---------- model ----------
    import torch
    import sys
    sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
    from phase2662_symmetric_mapping_contract import load_native
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        MD, local_files_only=True, trust_remote_code=True,
        use_fast=True)
    tc = {}

    def tid(t):
        if t not in tc:
            ids = tok(' ' + t, add_special_tokens=False)[
                'input_ids']
            if len(ids) != 1:
                ids = tok(t, add_special_tokens=False)[
                    'input_ids']
            assert len(ids) == 1, '%s -> %s' % (t, ids)
            tc[t] = int(ids[0])
        return tc[t]

    tid_map = {}
    for lang, ck, w in words:
        tid_map[w] = tid(w)
        if lang == 'en':
            assert tid_map[w] == int(ck), 'key mismatch %s' % w
    func_tid = tid('the')

    word_tids = set(tid_map.values())

    def sample_null(seed):
        rng = np.random.default_rng(seed)
        out = []
        while len(out) < n_words:
            r = int(rng.integers(0, VOCAB))
            if r not in word_tids and r > 0:
                out.append(r)
        return out

    null0_tids = sample_null(2896)
    batch = {'func': [[func_tid, tid_map[words[i][2]]]
                      for i in range(n_words)],
             'null0': [[null0_tids[i], tid_map[words[i][2]]]
                       for i in range(n_words)]}

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    log('model loaded (load_native full GPU)', lines)

    cap = {'attnin': {}}
    state_fin = {'on': False}
    fin_cap = {}
    inj = {'coef': None, 'scale': 0.0, 'vec': None}
    abl = {'li': None, 'hset': None}
    handles = []

    def pre_attn(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('hidden_states')
            if x is None or x.dim() < 2:
                return
            if inj['coef'] is not None \
                    and li in inj['coef']:
                c = inj['coef'][li]
                if c != 0.0:
                    x = x.clone()
                    x[:, 1, :] = x[:, 1, :] \
                        + c * inj['scale'] * inj['vec']
                if args:
                    nargs = (x,) + tuple(args[1:])
                    return nargs, kwargs
                nkw = dict(kwargs)
                nkw['hidden_states'] = x
                return args, nkw
            cap['attnin'].setdefault(li, []).append(
                x.detach().float().cpu().numpy())
            return None
        return h

    def pre_oproj(li):
        """Ablate a SET of heads at the o_proj INPUT."""
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('input')
            if x is None or x.dim() < 2:
                return None
            hs = abl['hset'] \
                if abl['li'] == li else None
            if hs:
                x = x.clone()
                for hi in hs:
                    x[:, 1, hi * HD:(hi + 1) * HD] = 0.0
                if args:
                    return (x,) + tuple(args[1:]), kwargs
                nkw = dict(kwargs)
                nkw['input'] = x
                return args, nkw
            return None
        return h

    def pre_norm(module, args, kwargs):
        if state_fin['on']:
            fin_cap['x'] = args[0][:, -1, :].detach() \
                .float().cpu().numpy()

    for li in range(NL):
        handles.append(layers[li].self_attn
                       .register_forward_pre_hook(
                           pre_attn(li), with_kwargs=True))
        handles.append(layers[li].self_attn.o_proj
                       .register_forward_pre_hook(
                           pre_oproj(li), with_kwargs=True))
    handles.append(model.model.norm.register_forward_pre_hook(
        pre_norm, with_kwargs=True))

    def clear_cap():
        for li in cap['attnin']:
            del cap['attnin'][li][:]

    def forward1(toks):
        clear_cap()
        with torch.no_grad():
            model(torch.tensor([toks], device='cuda'))
        return {li: cap['attnin'][li][0]
                for li in cap['attnin']}

    def forward_batch(toks_list, coef=None, scale=0.0,
                      vec=None, abl_li=None, abl_set=None):
        clear_cap()
        fin_cap.pop('x', None)
        state_fin['on'] = True
        inj['coef'] = coef
        inj['scale'] = float(scale)
        inj['vec'] = vec
        abl['li'] = abl_li
        abl['hset'] = abl_set
        with torch.no_grad():
            model(torch.tensor(toks_list, device='cuda'))
        inj['coef'] = None
        state_fin['on'] = False
        abl['li'] = None
        abl['hset'] = None
        fin = fin_cap['x'].astype(np.float64)
        return fin

    # ---------- pass 1: dirs_word rebuild ----------
    attn_store = {}
    for i, (_, _, w) in enumerate(words):
        attnin_all = forward1([func_tid, tid_map[w]])
        for li in range(NL):
            attn_store[(i, li)] = \
                attnin_all[li].astype(np.float32)
        if (i + 1) % 20 == 0:
            log('pass1 words [%d/%d]' % (i + 1, n_words), lines)
    d_dim = attn_store[(0, 0)].shape[-1]
    diffs_w = np.zeros((NL, d_dim))
    for li in range(NL):
        X = np.stack([attn_store[(i, li)][0, 1]
                      for i in range(n_words)]) \
            .astype(np.float64)
        diffs_w[li] = X[lab_lang == 0].mean(0) \
            - X[lab_lang == 1].mean(0)
    dirs_word = np.stack([unit(diffs_w[li]) for li in range(NL)])
    a1_diff = float(np.abs(dirs_word - dirs_word_27).max())
    a1_ok = bool(a1_diff < 1e-5)
    log('a1 dirs_word rebuild diff %.2e ok=%s'
        % (a1_diff, a1_ok), lines)

    _, _, Vt = np.linalg.svd(dirs_word, full_matrices=False)
    Vt8 = Vt[:8]
    a3_diff = float(np.abs(Vt8 - Vt8_39).max())
    a3_ok = bool(a3_diff < 1e-6)
    log('a3 Vt8 rebuild vs 2939 max diff %.2e ok=%s'
        % (a3_diff, a3_ok), lines)

    u35 = dirs_word[NL - 1]

    dcks_S = dcks_39[:, list(S_IDX)]
    Vt8_S = Vt8[list(S_IDX)]
    xdir = dcks_S @ Vt8_S
    a7_diff = float(np.abs(xdir @ Vt8_S.T - dcks_S).max())
    a7_ok = bool(a7_diff < 1e-9)
    log('a7 construction self-check max abs %.2e ok=%s'
        % (a7_diff, a7_ok), lines)
    xdir_t = torch.tensor(xdir, device='cuda',
                          dtype=torch.bfloat16)

    # a9: single-slice self-check + o_proj structural gate
    # a12: group-ablation mask self-check on synthetic
    rng = np.random.default_rng(7)
    syn = rng.standard_normal((2, 2, NH * HD))
    syn2 = syn.copy()
    syn2[:, 1, 5 * HD:6 * HD] = 0.0
    a9_diff = float(
        np.abs(syn2 - syn).max()
        - np.abs(syn[:, 1, 5 * HD:6 * HD]).max())
    oproj_in = layers[0].self_attn.o_proj.in_features
    a9_ok = bool(a9_diff < 1e-12
                 and oproj_in == NH * HD)
    log('a9 ablation-slice self-check %.2e | o_proj in '
        'features %d (expect %d) ok=%s'
        % (a9_diff, oproj_in, NH * HD, a9_ok), lines)

    gset = {0, 7, 30}
    syn3 = syn.copy()
    for hi in gset:
        syn3[:, 1, hi * HD:(hi + 1) * HD] = 0.0
    mask = np.ones(NH * HD, dtype=bool)
    for hi in gset:
        mask[hi * HD:(hi + 1) * HD] = False
    a12_diff = float(np.abs(
        syn3[:, 1, mask] - syn[:, 1, mask]).max())
    a12_ok = bool(a12_diff < 1e-12 and float(
        np.abs(syn3[:, 1, ~mask]).max()) == 0.0)
    log('a12 group-ablation mask self-check %.2e ok=%s'
        % (a12_diff, a12_ok), lines)

    # ---------- baselines ----------
    fin_f1 = forward_batch(batch['func'])
    fin_f2 = forward_batch(batch['func'])
    a2_rel = float(np.abs(fin_f1 - fin_f2).max()
                   / max(float(np.abs(fin_f1).max()), 1e-30))
    a2_ok = bool(a2_rel < 1e-4)
    log('a2 baseline determinism rel %.2e ok=%s'
        % (a2_rel, a2_ok), lines)

    def reads(fin):
        return fin @ u35, fin @ Vt8.T, \
            np.linalg.norm(fin, axis=1)

    proj_f0, c8_f0, nfin_f0 = reads(fin_f1)
    fin_n0 = forward_batch(batch['null0'])
    proj_n0, c8_n0, nfin_n0 = reads(fin_n0)

    a4_diff = float(np.abs(proj_f0 - s_base_35[ifu35]).max())
    a4_ok = bool(a4_diff < 1e-4)
    log('a4 proj_func vs 2935 max abs diff %.2e ok=%s'
        % (a4_diff, a4_ok), lines)
    a5_diff = float(np.abs(proj_n0 - s_base_35[in035]).max())
    a5_ok = bool(a5_diff < 1e-4)
    log('a5 proj_null0 vs 2935 max abs diff %.2e ok=%s'
        % (a5_diff, a5_ok), lines)
    sep_f = float(proj_f0[lab_lang == 0].mean()
                  - proj_f0[lab_lang == 1].mean())
    a6_ok = bool(sep_f > 0.0)
    log('a6 func separation %.4f ok=%s' % (sep_f, a6_ok), lines)
    sep_n = float(proj_n0[lab_lang == 0].mean()
                  - proj_n0[lab_lang == 1].mean())
    log('sep null0 %.3f (func %.3f)' % (sep_n, sep_f), lines)

    anchor_prelim = bool(a1_ok and a2_ok and a3_ok and a4_ok
                         and a5_ok and a6_ok and a7_ok
                         and a9_ok and a12_ok)

    verdict = None
    t1 = t2 = t3 = t4 = d1 = d2 = d3 = None
    save = {}
    a8_diff = a10_diff = a11_diff = None
    a8_ok = a10_ok = a11_ok = False
    cond_med = {}

    if not anchor_prelim:
        verdict = 'anchor_fail_all_void'
    else:
        def sep_of(fin):
            p, c8, nfin = reads(fin)
            return float(p[lab_lang == 0].mean()
                         - p[lab_lang == 1].mean()), p

        def run_cond(tag, coef, scale, abl_li, hset):
            seps = []
            for _ in range(K_REPEAT):
                fin = forward_batch(
                    batch['func'], coef=coef, scale=scale,
                    vec=xdir_t if coef else None,
                    abl_li=abl_li, abl_set=hset)
                seps.append(sep_of(fin)[0])
            med = float(np.median(seps))
            cond_med[tag] = med
            log('%s median sep %.2f (reps %s)'
                % (tag, med,
                   [round(s, 1) for s in seps]), lines)
            return med

        rest17 = [h for h in range(NH)
                  if h not in TOP5_G[LI_SWITCH]]
        rest16 = [h for h in range(NH)
                  if h not in TOP5_G[LI_GRAD]]

        # L17 family
        b2_17 = run_cond('B2_17', None, 0.0, LI_SWITCH,
                         rest17)
        b1_17 = run_cond('B1_17', None, 0.0, LI_SWITCH,
                         TOP5_G[LI_SWITCH])
        i0_17 = run_cond('I0_17', {LI_SWITCH: 1.0}, S_SWITCH,
                         LI_SWITCH, None)
        i1_17 = run_cond('I1_17', {LI_SWITCH: 1.0}, S_SWITCH,
                         LI_SWITCH, TOP5_G[LI_SWITCH])
        i2_17 = run_cond('I2_17', {LI_SWITCH: 1.0}, S_SWITCH,
                         LI_SWITCH, rest17)
        # L16 family
        b2_16 = run_cond('B2_16', None, 0.0, LI_GRAD, rest16)
        b1_16 = run_cond('B1_16', None, 0.0, LI_GRAD,
                         TOP5_G[LI_GRAD])
        i0_16 = run_cond('I0_16', {LI_GRAD: 1.0}, S_GRAD,
                         LI_GRAD, None)
        i1_16 = run_cond('I1_16', {LI_GRAD: 1.0}, S_GRAD,
                         LI_GRAD, TOP5_G[LI_GRAD])
        i2_16 = run_cond('I2_16', {LI_GRAD: 1.0}, S_GRAD,
                         LI_GRAD, rest16)

        # a10/a11: I0 references vs 2945
        a10_diff = abs(i0_17 - sep45['L17']['1.00'])
        a10_ok = bool(a10_diff < A_REF_TOL)
        a11_diff = abs(i0_16 - sep45['L16']['2.00'])
        a11_ok = bool(a11_diff < A_REF_TOL)
        log('a10 I0_17 vs 2945 diff %.4f ok=%s | '
            'a11 I0_16 vs 2945 diff %.4f ok=%s'
            % (a10_diff, a10_ok, a11_diff, a11_ok), lines)

        # a8: same-session repeat determinism
        fin_a = forward_batch(batch['func'],
                              coef={LI_SWITCH: 1.0},
                              scale=S_SWITCH, vec=xdir_t,
                              abl_li=LI_SWITCH,
                              abl_set=TOP5_G[LI_SWITCH])
        fin_b = forward_batch(batch['func'],
                              coef={LI_SWITCH: 1.0},
                              scale=S_SWITCH, vec=xdir_t,
                              abl_li=LI_SWITCH,
                              abl_set=TOP5_G[LI_SWITCH])
        a8_diff = float(np.abs(fin_a - fin_b).max())
        a8_ok = bool(a8_diff < 1e-6)
        log('a8 same-session determinism (I1_17 config) '
            '%.2e ok=%s' % (a8_diff, a8_ok), lines)

        if not (anchor_prelim and a8_ok and a10_ok
                and a11_ok):
            verdict = 'anchor_fail_all_void'
        else:
            t1_pass = bool(b2_17 > BASELINE_MIN
                           and i2_17 < SWITCH_THR)
            t2_pass = bool(i1_17 - i0_17 > NEC_MARGIN)
            t3_pass = bool(b2_16 > BASELINE_MIN
                           and i2_16 < SWITCH_THR)
            t4_pass = bool(i1_16 - i0_16 > NEC_MARGIN)
            t1 = {'sep_B2_17': round(b2_17, 2),
                  'sep_I2_17': round(i2_17, 2),
                  'baseline_min': BASELINE_MIN,
                  'switch_thr': SWITCH_THR,
                  'pass': t1_pass}
            t2 = {'sep_I1_17': round(i1_17, 2),
                  'sep_I0_17': round(i0_17, 2),
                  'delta': round(i1_17 - i0_17, 2),
                  'margin': NEC_MARGIN, 'pass': t2_pass}
            t3 = {'sep_B2_16': round(b2_16, 2),
                  'sep_I2_16': round(i2_16, 2),
                  'baseline_min': BASELINE_MIN,
                  'switch_thr': SWITCH_THR,
                  'pass': t3_pass}
            t4 = {'sep_I1_16': round(i1_16, 2),
                  'sep_I0_16': round(i0_16, 2),
                  'delta': round(i1_16 - i0_16, 2),
                  'margin': NEC_MARGIN, 'pass': t4_pass}
            log('T1 %s | T2 %s | T3 %s | T4 %s'
                % (t1_pass, t2_pass, t3_pass, t4_pass),
                lines)

            suf17, nec17 = t1_pass, t2_pass
            suf16, nec16 = t3_pass, t4_pass
            if suf17 and nec17 and suf16 and nec16:
                verdict = 'head_dose_sufficient_necessary'
            elif (suf17 and nec17) != (suf16 and nec16):
                verdict = 'head_dose_layer_asymmetric'
            elif suf17 and suf16:
                verdict = 'head_dose_sufficient_only'
            elif nec17 and nec16:
                verdict = 'head_dose_necessary_only'
            else:
                verdict = 'head_dose_not_carried'

            d1 = {k: round(v, 2)
                  for k, v in sorted(cond_med.items())}
            d2 = {
                'necessity_delta_L17':
                    round(i1_17 - i0_17, 2),
                'necessity_delta_L16':
                    round(i1_16 - i0_16, 2),
                'sufficiency_depth_L17':
                    round(b2_17 - i2_17, 2),
                'sufficiency_depth_L16':
                    round(b2_16 - i2_16, 2),
                'sep_func': round(sep_f, 2),
                'sep_null0': round(sep_n, 2),
                'note': 'necessity delta > 0: ablating '
                        'top5_g heads weakens the switch; '
                        'sufficiency depth = how far the '
                        'top5-only configuration pushes '
                        'sep below its own baseline',
            }
            d3 = {
                'top5_g_L17': TOP5_G[LI_SWITCH],
                'top5_g_L16': TOP5_G[LI_GRAD],
                'n_rest_L17': len(rest17),
                'n_rest_L16': len(rest16),
                'source': '2948 result.json D2_top5 '
                          '(sha8 %s)' % sha8(SRC_2948),
            }

            save = {
                'words': np.array(['%s:%s:%s' % w
                                   for w in words],
                                  dtype=object),
                'labels_lang': lab_lang,
                'cond_tags': np.array(
                    sorted(cond_med.keys()), dtype=object),
                'cond_seps': np.array(
                    [cond_med[k]
                     for k in sorted(cond_med.keys())]),
                'Vt8': Vt8, 'dirs_word': dirs_word,
            }

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2949, 'model': 'qwen3-4b',
           'prereg': PREREG,
           'anchors': {'a1_diff': float('%.3e' % a1_diff),
                       'a1_ok': a1_ok,
                       'a2_rel': float('%.3e' % a2_rel),
                       'a2_ok': a2_ok,
                       'a3_diff': float('%.3e' % a3_diff),
                       'a3_ok': a3_ok,
                       'a4_diff': float('%.3e' % a4_diff),
                       'a4_ok': a4_ok,
                       'a5_diff': float('%.3e' % a5_diff),
                       'a5_ok': a5_ok,
                       'a6_sep_func': round(sep_f, 4),
                       'a6_ok': a6_ok,
                       'a7_diff': float('%.3e' % a7_diff),
                       'a7_ok': a7_ok,
                       'a9_diff': float('%.3e' % a9_diff),
                       'a9_ok': a9_ok,
                       'a12_diff': float('%.3e' % a12_diff),
                       'a12_ok': a12_ok,
                       'a8_diff': None if a8_diff is None
                       else float('%.3e' % a8_diff),
                       'a8_ok': a8_ok,
                       'a10_diff': None if a10_diff is None
                       else round(a10_diff, 4),
                       'a10_ok': a10_ok,
                       'a11_diff': None if a11_diff is None
                       else round(a11_diff, 4),
                       'a11_ok': a11_ok,
                       'ok': bool(anchor_prelim and a8_ok
                                  and a10_ok and a11_ok)},
           'T1_sufficiency_L17': t1,
           'T2_necessity_L17': t2,
           'T3_sufficiency_L16': t3,
           'T4_necessity_L16': t4,
           'D1_condition_seps': d1, 'D2_effects': d2,
           'D3_groups': d3,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(os.path.join(
            OUT, 'head_dose_sufficiency.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2949 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
