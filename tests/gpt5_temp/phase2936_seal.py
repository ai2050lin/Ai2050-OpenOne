# -*- coding: utf-8 -*-
"""Phase 2936 seal: verdict-boundary forensics.

Quantifies the scale-driven vs raw-amplification decomposition
that explains why the additive-anchor battery failed: rel-CI
'amplification' (2934/2935 verdicts) may be denominator-driven
(scale_null << scale_func), not numerator-driven (raw CI
increase). All zero-forward from existing npz.
"""
import hashlib
import json
import os

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
NPZ35 = os.path.join(BASE, 'phase2935', 'null_amp_anatomy',
                     'null_amp_anatomy.npz')
NPZ33 = os.path.join(BASE, 'phase2933', 'full_atlas_ci',
                     'full_atlas_ci.npz')
NPZ34 = os.path.join(BASE, 'phase2934', 'loadband_anatomy',
                     'loadband_anatomy.npz')
OUTD = os.path.join(BASE, 'phase2936', 'anchoring_law')
REP = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2936_seal_report.txt')
LOAD = list(range(6, 13))
DEEP = list(range(28, 36))
SURV = [(1, 6), (5, 6), (7, 19), (8, 2), (14, 9), (20, 8),
        (21, 6)]


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def avg_ranks(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x))
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(x, y):
    rx = avg_ranks(x)
    ry = avg_ranks(y)
    if rx.std() < 1e-30 or ry.std() < 1e-30:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    out = []
    z35 = np.load(NPZ35, allow_pickle=True)
    cells = [tuple(int(v) for v in c) for c in z35['cells']]
    pos = {c: i for i, c in enumerate(cells)}
    conds = [str(s) for s in z35['cond_names']]
    ci = z35['ci_rel'].astype(np.float64)
    scale = z35['scale'].astype(np.float64)
    ifu = conds.index('func')
    inulls = [i for i, s in enumerate(conds)
              if s.startswith('null')]
    f_rel = ci[ifu]
    f_raw = f_rel * scale[ifu]

    out.append('scale (per cond): %s'
               % {conds[i]: round(float(scale[i]), 3)
                  for i in range(len(conds))})
    out.append('scale ratios null/func: %s'
               % {conds[i]: round(float(scale[i]
                                       / scale[ifu]), 3)
                  for i in inulls})
    out.append('')

    # raw vs rel amplification per set + band decomposition
    out.append('per null set: raw amp / rel amp medians + '
               'band decomposition')
    for i in inulls:
        nm = conds[i]
        raw_n = ci[i] * scale[i]
        raw_amp = np.array([raw_n[pos[c]] / max(f_raw[pos[c]],
                                                1e-30)
                            for c in cells])
        rel_amp = ci[i] / np.maximum(f_rel, 1e-30)
        ix_l = [pos[(h, li)] for li in LOAD for h in range(32)]
        ix_d = [pos[(h, li)] for li in DEEP for h in range(32)]
        out.append(
            '%s: raw_amp med %.4f (LOAD %.4f DEEP %.4f) | '
            'rel_amp med %.4f (LOAD %.4f DEEP %.4f)'
            % (nm, float(np.median(raw_amp)),
               float(np.median(raw_amp[ix_l])),
               float(np.median(raw_amp[ix_d])),
               float(np.median(rel_amp)),
               float(np.median(rel_amp[ix_l])),
               float(np.median(rel_amp[ix_d]))))
    out.append('')

    # supp_rel vs ci_rel_func: formal rel-caliber inversion
    out.append('rel-caliber: rho(supp_rel, ci_rel_func) per '
               'null set')
    for i in inulls:
        supp_rel = ci[i] - f_rel
        out.append('  %s: %.4f'
                   % (conds[i], spearman(supp_rel, f_rel)))
    out.append('')

    # supp<=0 cells: layer distribution (null0)
    supp_raw0 = (ci[inulls[0]] * scale[inulls[0]]
                 - f_raw)
    lay_nonpos = {}
    for c in cells:
        if supp_raw0[pos[c]] <= 0:
            lay_nonpos[c[1]] = lay_nonpos.get(c[1], 0) + 1
    out.append('null0 supp_raw<=0 cells by layer: %s'
               % {k: lay_nonpos.get(k, 0)
                  for k in sorted(lay_nonpos)})
    out.append('')

    # survivor core raw behavior
    out.append('survivor core (raw caliber, null0):')
    for c in SURV:
        i0 = inulls[0]
        out.append('  %s: raw_func %.6f raw_null %.6f '
                   'raw_amp %.3f rel_amp %.3f'
                   % (c, f_raw[pos[c]],
                      ci[i0][pos[c]] * scale[i0],
                      ci[i0][pos[c]] * scale[i0]
                      / max(f_raw[pos[c]], 1e-30),
                      ci[i0][pos[c]]
                      / max(f_rel[pos[c]], 1e-30)))
    out.append('')

    # same-condition raw check (2934 pos1 rows)
    z34 = np.load(NPZ34, allow_pickle=True)
    cfgs = [str(s) for s in z34['cfgs']]
    cnds = [str(s) for s in z34['conds']]
    sc34 = z34['scale'].astype(np.float64)
    ci34 = z34['ci_rel'].astype(np.float64)
    ip1 = cfgs.index('pos1')
    r_s = ip1 * 3 + cnds.index('same')
    r_f = ip1 * 3 + cnds.index('func')
    r_n = ip1 * 3 + cnds.index('null')
    raw_s = ci34[r_s] * sc34[cnds.index('same')]
    raw_f34 = ci34[r_f] * sc34[cnds.index('func')]
    raw_n34 = ci34[r_n] * sc34[cnds.index('null')]
    out.append('2934 raw caliber (pos1): scale same %.3f '
               'func %.3f null %.3f'
               % (sc34[cnds.index('same')],
                  sc34[cnds.index('func')],
                  sc34[cnds.index('null')]))
    out.append('  raw amp same/func med %.4f | null/func '
               'med %.4f'
               % (float(np.median(raw_s
                                   / np.maximum(raw_f34,
                                                1e-30))),
                  float(np.median(raw_n34
                                  / np.maximum(raw_f34,
                                               1e-30)))))
    out.append('')

    # SHA registration
    rj = json.load(open(os.path.join(OUTD, 'result.json'),
                        encoding='utf-8'))
    out.append('verdict: %s' % rj['final_verdict'])
    out.append('P1 median rho %.4f | P2 median %.4f | P3 '
               'same: rho %.4f R2lin %.4f a %.6f'
               % (rj['P1']['median_spearman'],
                  rj['P2']['median'],
                  rj['P3']['spearman'],
                  rj['P3']['r2_linear'],
                  rj['P3']['intercept']))
    out.append('SHA8: execution %s result %s npz %s '
               'script %s'
               % (sha8(os.path.join(OUTD, 'execution.json')),
                  sha8(os.path.join(OUTD, 'result.json')),
                  sha8(os.path.join(OUTD, 'anchoring_law.npz')),
                  sha8(r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
                       r'\phase2936_anchoring_law.py')))
    with open(REP, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out) + '\n')
    print('OK seal 2936', flush=True)


if __name__ == '__main__':
    main()
