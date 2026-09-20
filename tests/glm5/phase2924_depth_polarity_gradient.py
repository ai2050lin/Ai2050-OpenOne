# -*- coding: utf-8 -*-
"""Phase 2924: depth-graded polarity code -- zero-forward.

Why: 2922/2923 established that all 13 attribute events are
bipolar contrast detectors (m_hi, m_lo opposite signs) whose
driving sign is layer-structured (LOW-driven shallow 7/8,
HIGH-driven deep 5/5, Fisher p 0.0047, quasi-post-hoc). Open
question (2923 candidate A): is "shallow LOW / deep HIGH" a
FULL-GRID continuous depth gradient, i.e. does the pole-contrast
direction D[h,l] = m_hi[h,l] - m_lo[h,l] (over ALL 32 heads,
not just sig events) rise with depth within each attribute axis?

Mode: zero forward. Artifact domain on the 2921 npz (B_speed/
B_size/B_moist, pole labels) + 2917 npz (lang en/L contrast
profile as non-attribute control) + 2923 npz (d_pole13
reference).

Definitions (frozen):
  m_hi[h,l] = mean over HIGH words of B_ax[h, w, l]; m_lo
  likewise; D = m_hi - m_lo (32x36 per axis). Layer profile
  c_ax[l] = median_h D[h, l] (36 values per axis).
  Main statistic: rho_axis = Spearman(l, c_ax), l = 0..35.
  Direction criterion (frozen BEFORE running): the 2923 layer
  structure predicts LOW-driven (D < 0) shallow and HIGH-driven
  (D > 0) deep, i.e. c_ax INCREASES with l => rho > 0.

Null: 2000 size-preserving pole-label permutations per axis
(fresh default_rng(2899)); under a permutation D, c and rho are
recomputed. p_axis = (#{rho_perm >= rho_obs} + 1)/2001
(one-sided). Per-layer p_l = (#{c_perm[l] >= c_obs[l]}+1)/2001.

Verdict criteria (frozen):
  anchor fail => anchor_fail_all_void;
  >= 2 of 3 attr axes with rho > 0 AND p_axis <= 0.05
      => depth_graded_polarity_confirmed;
  exactly 1 => depth_graded_polarity_partial;
  0 => depth_gradient_absent.

Probes (frozen):
  P1 main per-axis test as above + per-layer significance
     counts (p_l <= 0.05), sign pattern of c_ax (n negative
     layers / n positive layers).
  P2 lang control: en/L contrast profile on the 2917 npz with
     the same machinery (descriptive; lang axis has no pole
     semantics, gradient expected to differ).
  P3 event-level: Spearman(d_pole, event layer l) over the 13
     events (quasi-post-hoc - 2923 already showed the peak
     version) + permutation p via the same global pole
     permutation frame (rng 2900 stream).
  P4 contrast strength: layer profile of median_h |D| per axis
     (descriptive); |d_pole| vs l over the 13 events.

Anchors (frozen):
  a1 2921 sign_M recomputed from npz B per axis (2922 a1
     verbatim);
  a2 d_pole recomputed for the 13 events vs 2923 npz
     d_pole13: |diff| < 1e-9 all;
  a3 2923 npz t5_perm: p50 == 6.0, max == 11.

Output: phase2924/depth_polarity_gradient/.
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC21 = os.path.join(BASE, 'phase2921', 'attr_vocab_expansion',
                     'attr_vocab_expansion.npz')
SRC17 = os.path.join(BASE, 'phase2917', 'event_atlas',
                     'event_atlas.npz')
SRC23 = os.path.join(BASE, 'phase2923', 'polarity_sign_anatomy')
OUT = os.path.join(BASE, 'phase2924', 'depth_polarity_gradient')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2924_run_report.txt')
N_PERM = 2000
FDR_Q = 0.05
NH, NL = 32, 36

PREREG = {
    'mode': 'zero-forward depth-polarity gradient on the 2921 '
            'npz (no model load) + 2917 npz lang control + 2923 '
            'npz reference',
    'question': '2923 candidate A: is "shallow LOW-driven / deep '
                'HIGH-driven" a full-grid continuous depth '
                'gradient - does the pole-contrast direction '
                'D[h,l] = m_hi - m_lo rise with depth across ALL '
                'heads within each attribute axis?',
    'main_stat': 'c_ax[l] = median_h D[h,l], 36 layers; rho_axis '
                 '= Spearman(l, c_ax); direction criterion: c_ax '
                 'increases with l (LOW-driven shallow, HIGH-'
                 'driven deep) => rho > 0; null = 2000 size-'
                 'preserving pole permutations per axis (fresh '
                 'default_rng(2899)); p_axis = one-sided '
                 '(#{rho_perm >= rho_obs}+1)/2001; per-layer p_l '
                 'one-sided on c',
    'verdict': 'anchor fail => anchor_fail_all_void; >= 2/3 attr '
               'axes rho>0 AND p_axis<=0.05 => '
               'depth_graded_polarity_confirmed; exactly 1 => '
               'depth_graded_polarity_partial; 0 => '
               'depth_gradient_absent',
    'anchors': {
        'a1': '2921 sign_M recompute per axis (2922 a1 '
              'verbatim)',
        'a2': '13-event d_pole recompute vs 2923 npz d_pole13: '
              '|diff| < 1e-9 all',
        'a3': '2923 npz t5_perm p50 == 6.0 and max == 11',
    },
    'probes': {
        'P1': 'per-axis main test + per-layer significance '
              'counts + c_ax sign pattern',
        'P2': 'lang en/L contrast profile control (descriptive)',
        'P3': 'Spearman(d_pole, event layer) over 13 events, '
              'permutation p via rng(2900) (quasi-post-hoc)',
        'P4': 'contrast strength profiles median_h |D| + '
              '|d_pole| vs l (descriptive)',
    },
}

EVENTS = {
    'speed': [(18, 16), (14, 12)],
    'size': [(21, 7), (24, 19), (12, 18), (26, 21), (25, 15),
             (11, 23), (11, 27), (22, 2), (18, 7)],
    'moist': [(8, 15), (10, 9)],
}
AX_AI = {'speed': 1, 'size': 2, 'moist': 3}


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def log(msg, lines):
    lines.append(msg)
    print(msg, flush=True)


def masks(lab):
    n = len(lab)
    eye = np.eye(n, dtype=bool)
    same = (lab[:, None] == lab[None, :]) & (~eye)
    diff = (~eye) & (~same)
    return same, diff


def sign_recompute(B, lab):
    same_m, diff_m = masks(lab)
    out = np.zeros((B.shape[0], B.shape[2]))
    for h in range(B.shape[0]):
        for li in range(B.shape[2]):
            s = np.sign(B[h, :, li])
            s[s == 0] = 1.0
            Gm = np.outer(s, s)
            out[h, li] = Gm[same_m].mean() - Gm[diff_m].mean()
    return out


def ranks_ord(M):
    order = np.argsort(M, axis=1)
    r = np.empty(M.shape, dtype=np.float64)
    rows = np.arange(M.shape[0])[:, None]
    r[rows, order] = np.arange(M.shape[1],
                               dtype=np.float64)[None, :] + 1.0
    return r


def spearman_scalar(x, y):
    rx = ranks_ord(np.asarray(x)[None, :])[0]
    ry = ranks_ord(np.asarray(y)[None, :])[0]
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt(rx @ rx * ry @ ry)
    return float(rx @ ry / max(den, 1e-30))


def d_pole_of(r, hi, lo):
    v_hi = r[hi].astype(np.float64)
    v_lo = r[lo].astype(np.float64)
    pooled = np.sqrt(max(v_hi.var() + v_lo.var(), 1e-30))
    return float(v_hi.mean() - v_lo.mean()) / float(pooled)


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2924,
                   'name': 'depth_polarity_gradient',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2921_npz': sha8(SRC21),
                               's2917_npz': sha8(SRC17),
                               's2923_npz': sha8(os.path.join(
                                   SRC23,
                                   'polarity_sign_anatomy.npz'))},
                   'n_perm': N_PERM, 'fdr_q': FDR_Q,
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    z = np.load(SRC21, allow_pickle=True)
    B = {ax: z['B_%s' % ax].astype(np.float64)
         for ax in ('speed', 'size', 'moist')}
    sign_M = z['sign_M'].astype(np.float64)
    p_maxT = z['p_maxT'].astype(np.float64)
    pole = {ax: np.asarray(z['labels_%s' % ax]).astype(int)
            for ax in ('speed', 'size', 'moist')}
    z17 = np.load(SRC17, allow_pickle=True)
    B17 = z17['B_heads'].astype(np.float64)
    lab17 = np.asarray(z17['labels_lang']).astype(int)
    z23 = np.load(os.path.join(SRC23,
                               'polarity_sign_anatomy.npz'),
                  allow_pickle=True)

    # ---------- anchors ----------
    a1 = {}
    a1_ok = True
    for ax in ('speed', 'size', 'moist'):
        sre = sign_recompute(B[ax], pole[ax])
        D = np.abs(sre - sign_M[AX_AI[ax]])
        sc = [(h, li) for h in range(NH) for li in range(NL)
              if p_maxT[AX_AI[ax]][h, li] <= FDR_Q]
        sc_max = max(float(D[h, li]) for (h, li) in sc)
        ok = bool(np.median(D) < 1e-6 and D.max() < 0.15
                  and np.sum(D > 0.02) <= 5 and sc_max < 0.05)
        a1[ax] = {'median': float(np.median(D)),
                  'max': float(D.max()), 'ok': ok}
        a1_ok = a1_ok and ok
    a2_ok = True
    a2_max = 0.0
    oi = 0
    for ax in ('speed', 'size', 'moist'):
        lab = pole[ax]
        hi = lab == 1
        lo = lab == 0
        for (h, li) in EVENTS[ax]:
            d = d_pole_of(B[ax][h, :, li], hi, lo)
            diff = abs(d - float(z23['d_pole13'][oi]))
            a2_max = max(a2_max, diff)
            a2_ok = a2_ok and diff < 1e-9
            oi += 1
    t5 = z23['t5_perm']
    a3_ok = bool(float(np.percentile(t5, 50)) == 6.0
                 and int(t5.max()) == 11)
    a3 = {'t5_p50': float(np.percentile(t5, 50)),
          't5_max': int(t5.max())}
    anchor_ok = bool(a1_ok and a2_ok and a3_ok)
    log('a1 ok=%s | a2 ok=%s max_diff %.2e | a3 %s ok=%s'
        % (a1_ok, a2_ok, a2_max, a3, a3_ok), lines)

    verdict = None
    p1 = p2 = p3 = p4 = None
    save = {}
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        # ---------- P1 per-axis main test ----------
        layers = np.arange(NL, dtype=np.float64)
        p1 = {}
        rho_axis = {}
        c_profiles = {}
        for ax in ('speed', 'size', 'moist'):
            lab = pole[ax]
            hi = lab == 1
            lo = lab == 0
            Dm = (B[ax][:, hi, :].mean(axis=1)
                  - B[ax][:, lo, :].mean(axis=1))
            c = np.median(Dm, axis=0)
            rho_obs = spearman_scalar(layers, c)
            rng = np.random.default_rng(2899)
            rho_perm = np.empty(N_PERM)
            c_perm = np.empty((N_PERM, NL))
            for t in range(N_PERM):
                lab_p = lab[rng.permutation(len(lab))]
                Dp = (B[ax][:, lab_p == 1, :].mean(axis=1)
                      - B[ax][:, lab_p == 0, :].mean(axis=1))
                cp = np.median(Dp, axis=0)
                c_perm[t] = cp
                rho_perm[t] = spearman_scalar(layers, cp)
            p_axis = (int(np.sum(rho_perm >= rho_obs)) + 1.0) \
                / (N_PERM + 1.0)
            p_l = (np.sum(c_perm >= c[None, :], axis=0) + 1.0) \
                / (N_PERM + 1.0)
            n_sig_l = int(np.sum(p_l <= FDR_Q))
            n_neg = int(np.sum(c < 0))
            n_pos = int(np.sum(c > 0))
            rho_axis[ax] = rho_obs
            c_profiles[ax] = c
            p1[ax] = {
                'rho': round(rho_obs, 5),
                'p_axis': round(p_axis, 6),
                'rho_perm_p50': round(
                    float(np.percentile(rho_perm, 50)), 5),
                'n_sig_layers': n_sig_l,
                'sig_layers': [int(li) for li in range(NL)
                               if p_l[li] <= FDR_Q],
                'c_neg_layers': n_neg, 'c_pos_layers': n_pos,
                'c_first12_mean': round(float(c[:12].mean()), 6),
                'c_last12_mean': round(float(c[24:].mean()), 6)}
            log('P1 %s: rho %.4f p_axis %.4f (perm p50 %.4f) | '
                'sig layers %d %s | c<0 %d c>0 %d | shallow12 '
                '%.2e deep12 %.2e'
                % (ax, rho_obs, p_axis,
                   p1[ax]['rho_perm_p50'], n_sig_l,
                   p1[ax]['sig_layers'], n_neg, n_pos,
                   p1[ax]['c_first12_mean'],
                   p1[ax]['c_last12_mean']), lines)
        n_pass = sum(1 for ax in ('speed', 'size', 'moist')
                     if rho_axis[ax] > 0
                     and p1[ax]['p_axis'] <= FDR_Q)
        log('P1 verdict axis: %d/3 axes pass' % n_pass, lines)

        # ---------- P2 lang control ----------
        hi17 = lab17 == 0   # en
        lo17 = lab17 == 1   # L
        D17 = (B17[:, hi17, :].mean(axis=1)
               - B17[:, lo17, :].mean(axis=1))
        c17 = np.median(D17, axis=0)
        rho17 = spearman_scalar(layers, c17)
        rng17 = np.random.default_rng(2899)
        rho17_perm = np.empty(N_PERM)
        for t in range(N_PERM):
            lab_p = lab17[rng17.permutation(len(lab17))]
            Dp = (B17[:, lab_p == 0, :].mean(axis=1)
                  - B17[:, lab_p == 1, :].mean(axis=1))
            rho17_perm[t] = spearman_scalar(
                layers, np.median(Dp, axis=0))
        p17 = (int(np.sum(rho17_perm >= rho17)) + 1.0) \
            / (N_PERM + 1.0)
        p2 = {'rho_en_minus_L': round(rho17, 5),
              'p_axis': round(p17, 6),
              'note': 'en/L contrast profile, descriptive '
                      'control (en=HIGH side by label coding '
                      'here, sign semantics differ from pole '
                      'axes)'}
        log('P2 lang: rho %.4f p %.4f' % (rho17, p17), lines)

        # ---------- P3 event-level d_pole vs layer ----------
        dp13 = z23['d_pole13'].astype(np.float64)
        ev_l = np.array([e[1] for ax in
                         ('speed', 'size', 'moist')
                         for e in EVENTS[ax]], dtype=np.float64)
        rho_ev = spearman_scalar(ev_l, dp13)
        rng3 = np.random.default_rng(2900)
        rho_ev_perm = np.empty(N_PERM)
        for t in range(N_PERM):
            dpp = np.empty(13)
            k = 0
            for ax in ('speed', 'size', 'moist'):
                lab = pole[ax]
                pp = lab[rng3.permutation(len(lab))]
                hi = pp == 1
                lo = pp == 0
                for (h, li) in EVENTS[ax]:
                    dpp[k] = d_pole_of(B[ax][h, :, li], hi, lo)
                    k += 1
            rho_ev_perm[t] = spearman_scalar(ev_l, dpp)
        p_ev = (int(np.sum(rho_ev_perm >= rho_ev)) + 1.0) \
            / (N_PERM + 1.0)
        p3 = {'rho_d_pole_vs_layer': round(rho_ev, 5),
              'p_perm': round(p_ev, 6),
              'note': 'quasi-post-hoc (2923 showed the peak '
                      'version); event layer l used here'}
        log('P3: rho(d_pole, l) %.4f p %.4f'
            % (rho_ev, p_ev), lines)

        # ---------- P4 contrast strength ----------
        p4 = {}
        for ax in ('speed', 'size', 'moist'):
            lab = pole[ax]
            hi = lab == 1
            lo = lab == 0
            Am = np.abs(B[ax][:, hi, :].mean(axis=1)
                        - B[ax][:, lo, :].mean(axis=1))
            cs = np.median(Am, axis=0)
            p4[ax] = {'strength_shallow12': round(
                          float(cs[:12].mean()), 6),
                      'strength_deep12': round(
                          float(cs[24:].mean()), 6),
                      'rho_strength_vs_layer': round(
                          spearman_scalar(layers, cs), 5)}
        abs_dp = np.abs(dp13)
        p4['rho_abs_d_pole_vs_layer'] = round(
            spearman_scalar(ev_l, abs_dp), 5)
        log('P4: %s | rho(|d_pole|, l) %.4f'
            % (p4, p4['rho_abs_d_pole_vs_layer']), lines)

        # ---------- verdict ----------
        if n_pass >= 2:
            verdict = 'depth_graded_polarity_confirmed'
        elif n_pass == 1:
            verdict = 'depth_graded_polarity_partial'
        else:
            verdict = 'depth_gradient_absent'

        save = {
            'c_speed': c_profiles['speed'].astype(np.float64),
            'c_size': c_profiles['size'].astype(np.float64),
            'c_moist': c_profiles['moist'].astype(np.float64),
            'c_lang': c17.astype(np.float64),
            'rho_axis': np.array([rho_axis['speed'],
                                  rho_axis['size'],
                                  rho_axis['moist'],
                                  rho17],
                                 dtype=np.float64),
            'd_pole13_ref': dp13,
            'ev_layers': ev_l.astype(np.int64),
        }

    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2924,
           'model': 'qwen3-4b (zero forward)',
           'prereg': PREREG,
           'anchors': {'a1': a1, 'a1_ok': a1_ok,
                       'a2_ok': a2_ok, 'a2_max_diff': a2_max,
                       'a3': a3, 'a3_ok': a3_ok,
                       'ok': anchor_ok},
           'P1': p1, 'P2': p2, 'P3': p3, 'P4': p4,
           'n_axes_pass': (None if verdict ==
                           'anchor_fail_all_void'
                           else n_pass),
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(
            os.path.join(OUT, 'depth_polarity_gradient.npz'),
            **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2924 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
