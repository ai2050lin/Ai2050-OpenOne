# -*- coding: utf-8 -*-
"""diag_2907b.py -- post-hoc attribution diagnostic for the 2906->2907
M1 verdict flip (both attn groups: below-interval in 2906, inside M1
in 2907).  Zero-forward matrix analysis; NOT a preregistered phase;
results feed the Phase 2907 MEMO section as an errata-style note.

Factors (full 2x2x2 = 8 combinations):
  S  sigma definition : mean_std  (2906 implementation:
                        B[m].std(0).mean())  vs  rms (2907
                        implementation / 2906 prereg TEXT:
                        sqrt(trace(Sigma_c)/d))
  R  rng seed         : 2906 vs 2907
  F  stream structure : mode2906 (rng_main shared across groups,
                        each group consumes ONLY its M1 block; this
                        reproduces 2906 module A exactly)  vs
                        mode2907 (each group consumes M1->M2a->M2b
                        blocks in order, shared stream; reproduces
                        2907 ladder exactly)

Anchors (must reproduce registered intervals exactly):
  (mean_std, 2906, mode2906) == 2906 result.json m1_lo/m1_hi
  (rms,      2907, mode2907) == 2907 result.json M1_interval

Registered numbers to compare against (from disk):
  2906 glm4_attn m1 [0.164931, 0.300391] full=0.121262 BELOW
  2906 qwen_attn m1 [-0.011573, 0.087131] full=-0.019459 BELOW
  2907 glm4_attn M1 [0.091486, 0.226235] full=0.121262 INSIDE
  2907 qwen_attn M1 [-0.019820, 0.063102] full=-0.019459 INSIDE
"""
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC = {
    'glm4': (os.path.join(BASE, 'phase2902',
                          'glm4_channel_jacobian_asymmetry',
                          'glm4_channel_jacobian_asymmetry.npz'),),
    'qwen': (os.path.join(BASE, 'phase2903',
                          'qwen_channel_jacobian_decomposition',
                          'qwen_channel_jacobian_decomposition.npz'),),
}
R2906 = os.path.join(BASE, 'phase2906',
                     'amplitude_law_neuron_attribution',
                     'result.json')
R2907 = os.path.join(BASE, 'phase2907', 'shape_correction_law',
                     'result.json')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\diag_2907b.txt')
N_SYNTH = 400
GROUP_ORDER = ('glm4_mlp', 'glm4_attn', 'qwen_mlp', 'qwen_attn')


def unit_rows(M):
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.maximum(n, 1e-30)


def margin_of_B(B, lab):
    U = unit_rows(B)
    Sm = U @ U.T
    n = len(lab)
    eye = np.eye(n, dtype=bool)
    same = (lab[:, None] == lab[None, :]) & (~eye)
    diff = (~eye) & (~same)
    return float(Sm[same].mean() - Sm[diff].mean())


def interval(ms):
    return (float(np.percentile(ms, 2.5)),
            float(np.percentile(ms, 97.5)))


def cholesky_psd(S):
    ridge = 0.0
    for _ in range(6):
        try:
            return np.linalg.cholesky(
                S + ridge * np.eye(S.shape[0]))
        except np.linalg.LinAlgError:
            ridge = max(ridge * 10, 1e-10)
    raise np.linalg.LinAlgError('chol failed')


def load_groups():
    groups = {}
    for mdl, (npz,) in SRC.items():
        z = np.load(npz, allow_pickle=True)
        lab = np.asarray(z['labels_lang']).astype(int)
        for ch in ('mlp', 'attn'):
            groups['%s_%s' % (mdl, ch)] = {
                'B': z['B_%s' % ch].astype(np.float64),
                'lab': lab,
            }
    return groups


def sigma_of(Bm, sdef):
    if sdef == 'mean_std':
        return float(Bm.std(0).mean())
    S = np.cov(Bm, rowvar=False)
    d = S.shape[0]
    return float(np.sqrt(np.trace(S) / d))


def run_combo(groups, sdef, seed, mode, lines):
    """Reproduce M1 intervals under one (sigma_def, seed, mode)."""
    rng = np.random.default_rng(seed)
    out = {}
    for g in GROUP_ORDER:
        B, lab = groups[g]['B'], groups[g]['lab']
        m_full = margin_of_B(B, lab)
        m0, m1 = lab == 0, lab == 1
        mu0, mu1 = B[m0].mean(0), B[m1].mean(0)
        d = B.shape[1]
        s0 = sigma_of(B[m0], sdef)
        s1 = sigma_of(B[m1], sdef)
        # --- M1 block (2906-style inline synthesis) ---
        ms = []
        for _ in range(N_SYNTH):
            Bs = np.empty_like(B)
            Bs[m0] = mu0 + s0 * rng.normal(
                size=(int(m0.sum()), d))
            Bs[m1] = mu1 + s1 * rng.normal(
                size=(int(m1.sum()), d))
            ms.append(margin_of_B(Bs, lab))
        lo, hi = interval(ms)
        out[g] = {'lo': lo, 'hi': hi,
                  'inside': bool(lo <= m_full <= hi),
                  'full': m_full, 's0': s0, 's1': s1}
        if mode == 'mode2907':
            # consume M2a and M2b blocks on the same stream so the
            # NEXT group's M1 starts where 2907's ladder put it
            S0 = np.cov(B[m0], rowvar=False)
            S1 = np.cov(B[m1], rowvar=False)
            for L0, L1 in (
                (np.diag(np.sqrt(np.diag(S0))),
                 np.diag(np.sqrt(np.diag(S1)))),
                (cholesky_psd(S0), cholesky_psd(S1)),
            ):
                for _ in range(N_SYNTH):
                    z0 = rng.normal(size=(int(m0.sum()), d))
                    z1 = rng.normal(size=(int(m1.sum()), d))
                    _ = z0 @ L0.T, z1 @ L1.T  # consume only
        lines.append(
            '  %s full=%+.6f M1=[%.6f,%.6f] inside=%s '
            '(s0=%.5f s1=%.5f)'
            % (g, m_full, lo, hi, out[g]['inside'], s0, s1))
    n_in = sum(1 for e in out.values() if e['inside'])
    lines.append('  -> inside count %d/4; attn verdict: %s'
                 % (n_in,
                    'BOTH_INSIDE' if (out['glm4_attn']['inside']
                                      and out['qwen_attn']['inside'])
                    else ('SPLIT' if (out['glm4_attn']['inside']
                                      != out['qwen_attn']['inside'])
                          else 'BOTH_BELOW')))
    return out


def main():
    t0 = time.monotonic()
    lines = []
    groups = load_groups()
    lines.append('diag_2907b attribution grid (post-hoc)')
    lines.append('anchors:')
    r06 = json.load(open(R2906, encoding='utf-8'))
    r07 = json.load(open(R2907, encoding='utf-8'))
    for g in GROUP_ORDER:
        e6 = r06['groups'][g]
        e7 = r07['groups'][g]
        lines.append('  registered 2906 %s m1=[%.6f,%.6f] '
                     'inside=%s sigma(mean-std)=%.5f/%.5f'
                     % (g, e6['m1_lo'], e6['m1_hi'],
                        e6['inside_m1'], e6['sigma0_iso'],
                        e6['sigma1_iso']))
        lines.append('  registered 2907 %s M1=[%.6f,%.6f] '
                     'inside=%s sigma(rms)=%.5f/%.5f'
                     % (g, e7['M1_interval'][0],
                        e7['M1_interval'][1], e7['inside_M1'],
                        e7['sigma0_iso'], e7['sigma1_iso']))
    lines.append('sigma ratio rms/mean-std (per group, s0/s1):')
    for g in GROUP_ORDER:
        B, lab = groups[g]['B'], groups[g]['lab']
        m0, m1 = lab == 0, lab == 1
        r0 = (sigma_of(B[m0], 'rms')
              / max(sigma_of(B[m0], 'mean_std'), 1e-30))
        r1 = (sigma_of(B[m1], 'rms')
              / max(sigma_of(B[m1], 'mean_std'), 1e-30))
        lines.append('  %s ratio0=%.4f ratio1=%.4f'
                     % (g, r0, r1))

    results = {}
    for sdef in ('mean_std', 'rms'):
        for seed in (2906, 2907):
            for mode in ('mode2906', 'mode2907'):
                tag = '%s|%d|%s' % (sdef, seed, mode)
                lines.append('---- combo %s ----' % tag)
                results[tag] = run_combo(
                    groups, sdef, seed, mode, lines)

    # anchor checks
    a6 = results['mean_std|2906|mode2906']
    a7 = results['rms|2907|mode2907']
    d6 = max(abs(a6[g]['lo'] - r06['groups'][g]['m1_lo'])
             + abs(a6[g]['hi'] - r06['groups'][g]['m1_hi'])
             for g in GROUP_ORDER)
    d7 = max(abs(a7[g]['lo'] - r07['groups'][g]['M1_interval'][0])
             + abs(a7[g]['hi'] - r07['groups'][g]['M1_interval'][1])
             for g in GROUP_ORDER)
    lines.append('==== anchor 2906 recheck max|diff| = %.3e (want 0)'
                 % d6)
    lines.append('==== anchor 2907 recheck max|diff| = %.3e (want 0)'
                 % d7)
    lines.append('==== cross-combo attn verdicts ====')
    for tag, e in results.items():
        lines.append('  %-28s glm4_attn inside=%-5s qwen_attn '
                     'inside=%-5s'
                     % (tag, e['glm4_attn']['inside'],
                        e['qwen_attn']['inside']))
    lines.append('runtime %.1fs' % (time.monotonic() - t0))
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK', REPORT)


if __name__ == '__main__':
    main()
