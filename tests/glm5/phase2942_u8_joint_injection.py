# -*- coding: utf-8 -*-
"""Phase 2942: U8 joint injection - is the joint coordinate
displacement (v1/v2/v5 per-word pattern) the causal driver of
the dir35 readout shift?

Why: 2941 preflight (existing artifacts) showed the U8
coordinate shifts (null0 - func, from 2939 npz coords) predict
the actual per-word dir35 readout shift with Spearman 0.994,
but the single v3 term carries only 0.4% of variance - the
per-word displacement mass lives in v2 (mean -19.7), v1
(-11.3), v5 (-8.2) of the actual mean shift -54.0. And 2941
proved a single-direction v3 injection is damped ~26x. The
open causal question: does injecting the JOINT per-word
displacement pattern (S = {v1, v2, v5} = 91% of the U8 mean
contribution) reproduce the null readout shift once the
propagation attenuation is compensated by a gain sweep?

Mode: ONE model (qwen3-4b), forward family. Injection site
verbatim 2927/2941 (attn-input pos-1, L16 = argmax w_li),
2938/2939 batch57 final pre-norm readout.

Injection vector (frozen construction):
  dcks = coords[null0] - coords[func]  (57x8, 2939 npz)
  xdir(w) = sum_{k in S} dcks[w,k] * Vt8[k],  S = {0,1,4}
    (v1, v2, v5 - the three dominant readout-shift bases)
  scale sweep s in {1, 2, 4, 8, 16, 32}, func condition.

Calibration + tests (frozen):
  tau(s)  = median_w ||c_shift_S(w,s)|| / (s * median_w
            ||dcks_S(w)||)   (S-subspace propagation gain)
  ratio(s)= median_w ||c_shift_S(w,s)|| / median_w
            ||dcks_S(w)||;  s* = argmin_s |ratio(s) - 1|
  R1 = Spearman(proj35_inj(s*) - proj35_func0,
                dp35_actual),  dp35_actual =
                proj35_null0_base - proj35_func0 (this run)
  R2 = median(proj35_inj(s*) - proj35_func0)
       / median(dp35_actual)
  R3 (descriptive): sep(s*) position between func 185.7 and
     null0 77.3

Verdict (frozen):
  anchor fail            => anchor_fail_all_void
  R1 < 0.8               => u8_displacement_not_causal
  R1 >= 0.8 AND R2 in
    [0.5, 2.0]           => u8_displacement_causally_sufficient
  else (R1 >= 0.8, R2
    outside)             => u8_displacement_magnitude_mismatch

Anchors (frozen; 2941 run3 values in parentheses):
  a1 dirs_word rebuild vs 2927 npz < 1e-5   (2.17e-08)
  a2 func baseline determinism < 1e-4       (0.0)
  a3 Vt8 rebuild vs 2939 npz < 1e-6         (0.0 bit-level)
  a4 proj_func vs 2935 s_base[func] < 1e-4  (7.2e-06)
  a5 proj_null0 vs 2935 s_base[null0] < 1e-4(6.3e-06)
  a6 func separation > 0                    (185.70)
  a7 c3_func rebuild vs 2940 npz < 1e-3     (3.3e-13)
  a8 injection construction self-check: xdir @ Vt8_S.T vs
     dcks_S max abs < 1e-9 (orthonormal-basis identity)

Descriptive: D1 per-direction propagation slope
  c_shift[:,k]/(s*dcks[:,k]) for k in S; D2 out-of-S
  displacement (c_shift on non-injected bases); D3 sep(s)
  profile; D4 word-structure rho vs func base; D5 fin norm.

Output: phase2942/u8_joint_injection/.
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
SRC_2940 = os.path.join(BASE, 'phase2940', 'v3_decode',
                        'v3_decode.npz')
OUT = os.path.join(BASE, 'phase2942', 'u8_joint_injection')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2942_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
SEED = 2896
NH, HD = 32, 128
NL = 36
VOCAB = 151936
L_INJ = 16
S_IDX = (0, 1, 4)             # v1, v2, v5
SCALES = (1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 8.0, 16.0, 32.0)
R1_MIN = 0.8
R2_LO, R2_HI = 0.5, 2.0

PREREG = {
    'mode': 'forward family: injection of the JOINT per-word '
            'displacement pattern xdir(w) = sum_{k in '
            'S={v1,v2,v5}} dcks[w,k]*Vt8[k] (dcks = 2939 npz '
            'coords null0-func; S = 91% of U8 mean readout '
            'contribution per 2941 preflight), scale sweep s '
            'in {1,2,4,8,16,32}, func condition; site 2927 '
            'verbatim (attn-input pos-1, L16=argmax w_li); '
            'batch57 final pre-norm readout',
    'question': 'is the joint U8 displacement pattern the '
                'causal driver of the dir35 readout shift '
                'under null context: when propagation '
                'attenuation is compensated (s* calibration), '
                'does the injected per-word pattern reproduce '
                'the actual null readout shift?',
    'anchors': {
        'a1': 'dirs_word rebuild vs 2927 npz < 1e-5',
        'a2': 'func baseline determinism < 1e-4',
        'a3': 'Vt8 rebuild vs 2939 npz < 1e-6',
        'a4': 'proj_func vs 2935 s_base[func] < 1e-4',
        'a5': 'proj_null0 vs 2935 s_base[null0] < 1e-4',
        'a6': 'func separation > 0',
        'a7': 'c3_func rebuild vs 2940 npz < 1e-3',
        'a8': 'injection construction self-check xdir@Vt8_S.T '
              'vs dcks_S < 1e-9',
    },
    'calibration': 'tau(s) = S-subspace propagation gain; '
                   'ratio(s) = median||c_shift_S||/median'
                   '||dcks_S||; s* = argmin |ratio-1|',
    'tests': 'R1 = Spearman(proj_inj(s*)-proj_func0, '
             'dp35_actual); R2 = median ratio; R3 sep '
             'position (descriptive)',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'R1 < 0.8 => u8_displacement_not_causal; '
               'R1 >= 0.8 AND R2 in [0.5,2.0] => '
               'u8_displacement_causally_sufficient; else => '
               'u8_displacement_magnitude_mismatch',
    'correction_note': 'run1: (1) R2/D1 denominator bug - '
                       'max(median(dp35),1e-30) clamped the '
                       'NEGATIVE median -29.06 to 1e-30 '
                       '(R2 exploded to -6e29; same clamp hit '
                       'signed per-word dcks in D1 slopes); '
                       'fixed to direct division with '
                       '|denominator| guard. (2) reachability '
                       '(discipline 10, window granularity): '
                       'run1 ratio jumped 0.508 (s=2) -> 1.819 '
                       '(s=4) - the ratio~=1 matching point '
                       'falls between untested scales; grid '
                       'refined to {1,2,2.5,3,3.5,4,8,16,32}. '
                       'run1 registered verdict '
                       'u8_displacement_not_causal (R1 0.713 '
                       'at the undershooting s*=2); run2 '
                       're-evaluates at the matched scale.',
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
        json.dump({'phase': 2942,
                   'name': 'u8_joint_injection',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2887': sha8(SRC_2887),
                               's2927': sha8(SRC_2927),
                               's2935': sha8(SRC_2935),
                               's2939': sha8(SRC_2939),
                               's2940': sha8(SRC_2940)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'seed': SEED, 'layer_inj': L_INJ,
                   's_idx': list(S_IDX),
                   'scales': list(SCALES),
                   'r1_min': R1_MIN, 'r2_lo': R2_LO,
                   'r2_hi': R2_HI,
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
    proj35_39 = z39['proj_dir35'].astype(np.float64)
    if39 = conds39.index('func')
    i039 = conds39.index('null0')
    dcks_39 = coords_39[i039] - coords_39[if39]  # (57, 8)

    z40 = np.load(SRC_2940, allow_pickle=True)
    c3_func_40 = z40['c3_func'].astype(np.float64)
    log('sources ok (dcks from 2939 npz)', lines)

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
    inj = {'on': False, 'scale': 0.0, 'vec': None}
    handles = []

    def pre_attn(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('hidden_states')
            if x is None or x.dim() < 2:
                return
            if li == L_INJ and inj['on']:
                x = x.clone()
                x[:, 1, :] = x[:, 1, :] \
                    + inj['scale'] * inj['vec']
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

    def pre_norm(module, args, kwargs):
        if state_fin['on']:
            fin_cap['x'] = args[0][:, -1, :].detach() \
                .float().cpu().numpy()

    for li in range(NL):
        handles.append(layers[li].self_attn
                       .register_forward_pre_hook(
                           pre_attn(li), with_kwargs=True))
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

    def forward_batch(toks_list, scale=0.0, vec=None):
        clear_cap()
        fin_cap.pop('x', None)
        state_fin['on'] = True
        inj['on'] = vec is not None
        inj['scale'] = float(scale)
        inj['vec'] = vec
        with torch.no_grad():
            model(torch.tensor(toks_list, device='cuda'))
        inj['on'] = False
        state_fin['on'] = False
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

    U_svd, s_svd, Vt = np.linalg.svd(dirs_word,
                                     full_matrices=False)
    Vt8 = Vt[:8]
    a3_diff = float(np.abs(Vt8 - Vt8_39).max())
    a3_ok = bool(a3_diff < 1e-6)
    log('a3 Vt8 rebuild vs 2939 max diff %.2e ok=%s'
        % (a3_diff, a3_ok), lines)

    u35 = dirs_word[NL - 1]

    # ---------- injection vector construction ----------
    dcks_S = dcks_39[:, list(S_IDX)]          # (57, 3)
    Vt8_S = Vt8[list(S_IDX)]                  # (3, 2560)
    xdir = dcks_S @ Vt8_S                     # (57, 2560)
    a8_diff = float(np.abs(xdir @ Vt8_S.T - dcks_S).max())
    a8_ok = bool(a8_diff < 1e-9)
    log('a8 construction self-check max abs %.2e ok=%s'
        % (a8_diff, a8_ok), lines)
    nd = np.linalg.norm(xdir, axis=1)
    log('xdir norms: med %.2f range [%.2f, %.2f]'
        % (float(np.median(nd)), float(nd.min()),
           float(nd.max())), lines)

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
    a7_diff = float(np.abs(c8_f0[:, 2] - c3_func_40).max())
    a7_ok = bool(a7_diff < 1e-3)
    log('a7 c3_func rebuild vs 2940 max abs diff %.2e ok=%s'
        % (a7_diff, a7_ok), lines)
    sep_f = float(proj_f0[lab_lang == 0].mean()
                  - proj_f0[lab_lang == 1].mean())
    a6_ok = bool(sep_f > 0.0)
    log('a6 func separation %.4f ok=%s' % (sep_f, a6_ok), lines)

    d39 = float(np.abs(proj_f0 - proj35_39[if39]).max())
    d39n = float(np.abs(proj_n0 - proj35_39[i039]).max())
    log('descriptive proj vs 2939: func %.2e null0 %.2e'
        % (d39, d39n), lines)

    dp35_actual = proj_n0 - proj_f0
    sep_n = float(proj_n0[lab_lang == 0].mean()
                  - proj_n0[lab_lang == 1].mean())
    log('dp35_actual: med %+.3f | sep null0 %.3f'
        % (float(np.median(dp35_actual)), sep_n), lines)

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok and a6_ok and a7_ok and a8_ok)
    verdict = None
    calib = p1 = d1 = d3 = d4 = d5 = None
    save = {}

    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        # ---------- scale sweep ----------
        proj_inj = {}
        c8_inj = {}
        nfin_inj = {}
        for s in SCALES:
            fin = forward_batch(batch['func'], scale=s,
                                vec=torch.tensor(
                                    xdir, device='cuda',
                                    dtype=torch.bfloat16))
            proj_inj[s], c8_inj[s], nfin_inj[s] = reads(fin)
        log('scale sweep done (%d forwards)' % len(SCALES),
            lines)

        # ---------- calibration ----------
        med_dS = float(np.median(
            np.linalg.norm(dcks_S, axis=1)))
        ratios = {}
        taus = {}
        for s in SCALES:
            cs = c8_inj[s][:, list(S_IDX)] - c8_f0[:, list(S_IDX)]
            med_c = float(np.median(
                np.linalg.norm(cs, axis=1)))
            ratios[s] = med_c / max(med_dS, 1e-30)
            taus[s] = ratios[s] / s
        s_star = min(SCALES, key=lambda s: abs(ratios[s] - 1.0))
        calib = {'ratio_per_s': {str(int(s)):
                                 round(ratios[s], 4)
                                 for s in SCALES},
                 'tau_per_s': {str(int(s)): round(taus[s], 5)
                               for s in SCALES},
                 'med_dS_norm': round(med_dS, 3),
                 's_star': int(s_star),
                 'ratio_s_star': round(ratios[s_star], 4)}
        log('calib ratio %s | s* = %d (ratio %.4f)'
            % (calib['ratio_per_s'], s_star,
               ratios[s_star]), lines)

        # ---------- R1/R2/R3 ----------
        shift_inj = proj_inj[s_star] - proj_f0
        r1 = spearman(shift_inj, dp35_actual)
        med_act = float(np.median(dp35_actual))
        med_sh = float(np.median(shift_inj))
        r2 = med_sh / med_act if abs(med_act) > 1e-12 \
            else float('nan')
        sep_i = float(proj_inj[s_star][lab_lang == 0].mean()
                      - proj_inj[s_star][lab_lang == 1].mean())
        p1 = {'R1_spearman': round(r1, 4),
              'R2_median_ratio': round(r2, 4),
              'median_shift_inj': round(med_sh, 3),
              'median_dp35_actual': round(med_act, 3),
              'sep_inj_sstar': round(sep_i, 2),
              'sep_func': round(sep_f, 2),
              'sep_null0': round(sep_n, 2)}
        log('R1 %.4f | R2 %.4f (shift med %+.3f vs actual '
            'med %+.3f) | sep_inj %.2f (func %.2f / null0 %.2f)'
            % (r1, r2, med_sh, med_act, sep_i, sep_f, sep_n),
            lines)

        if r1 < R1_MIN:
            verdict = 'u8_displacement_not_causal'
        elif r1 >= R1_MIN and R2_LO <= r2 <= R2_HI:
            verdict = 'u8_displacement_causally_sufficient'
        else:
            verdict = 'u8_displacement_magnitude_mismatch'

        # ---------- D1: per-direction slope ----------
        d1 = {}
        for ki, k in enumerate(S_IDX):
            row = {}
            for s in SCALES:
                cs = c8_inj[s][:, k] - c8_f0[:, k]
                den = s * dcks_S[:, ki]
                m = np.abs(den) > 1e-6
                row[str(int(s))] = round(float(np.median(
                    cs[m] / den[m])), 4)
            d1['v%d' % (k + 1)] = row
        log('D1 slopes %s' % json.dumps(d1), lines)

        # ---------- D2: out-of-S displacement ----------
        d2 = {}
        oos = [k for k in range(8) if k not in S_IDX]
        for s in SCALES:
            cs = c8_inj[s][:, oos] - c8_f0[:, oos]
            d2[str(int(s))] = round(float(np.median(
                np.linalg.norm(cs, axis=1))), 3)
        log('D2 out-of-S |c_shift| med per s: %s' % d2, lines)

        # ---------- D3: sep profile ----------
        d3 = {}
        for s in SCALES:
            d3[str(int(s))] = round(float(
                proj_inj[s][lab_lang == 0].mean()
                - proj_inj[s][lab_lang == 1].mean()), 2)
        log('D3 sep(s): %s' % d3, lines)

        # ---------- D4: word-structure rho ----------
        d4 = {}
        for s in SCALES:
            d4[str(int(s))] = round(
                spearman(proj_inj[s], proj_f0), 4)
        log('D4 rho vs func base: %s' % d4, lines)

        # ---------- D5: fin norm ----------
        d5 = {}
        for s in SCALES:
            d5[str(int(s))] = round(float(np.median(
                nfin_inj[s] / np.maximum(nfin_f0, 1e-30))), 5)
        log('D5 norm ratios: %s' % d5, lines)

        save = {
            'words': np.array(['%s:%s:%s' % w
                               for w in words],
                              dtype=object),
            'labels_lang': lab_lang,
            'scales': np.array(SCALES),
            'proj_inj': np.stack([proj_inj[s]
                                  for s in SCALES]),
            'c8_inj': np.stack([c8_inj[s] for s in SCALES]),
            'nfin_inj': np.stack([nfin_inj[s]
                                  for s in SCALES]),
            'proj_func0': proj_f0, 'proj_null0': proj_n0,
            'c8_func0': c8_f0, 'c8_null0': c8_n0,
            'dcks': dcks_39, 'xdir_norm': nd,
            'Vt8': Vt8, 'dirs_word': dirs_word,
            's_star': np.int64(s_star),
        }

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2942, 'model': 'qwen3-4b',
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
                       'a8_diff': float('%.3e' % a8_diff),
                       'a8_ok': a8_ok,
                       'ok': anchor_ok},
           'descriptive_proj_vs_2939': {
               'func': float('%.3e' % d39),
               'null0': float('%.3e' % d39n)},
           'calibration': calib, 'R': p1,
           'D1_slope': d1, 'D2_out_of_S': d2, 'D3_sep': d3,
           'D4_rho': d4, 'D5_norm': d5,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(os.path.join(
            OUT, 'u8_joint_injection.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2942 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
