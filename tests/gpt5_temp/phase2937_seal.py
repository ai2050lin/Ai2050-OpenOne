# -*- coding: utf-8 -*-
"""Phase 2937 seal: rewrite-mechanism forensics."""
import hashlib
import json
import os

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
NPZ = os.path.join(BASE, 'phase2937', 'scale_collapse',
                   'scale_collapse.npz')
OUTD = os.path.join(BASE, 'phase2937', 'scale_collapse')
REP = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2937_seal_report.txt')


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
    z = np.load(NPZ, allow_pickle=True)
    conds = [str(s) for s in z['cond_names']]
    proj = z['proj']  # (5, 57)
    fin_norm = z['fin_norm']  # (5, 57)
    attn_norm = z['attn_norm']  # (5, 36, 57)
    attn_proj = z['attn_proj']  # (5, 36, 57)
    ifu = conds.index('func')
    words = [str(w) for w in z['words']]
    words = [w.split(':')[-1] for w in words]

    cos = proj / np.maximum(fin_norm, 1e-30)
    out.append('cos(final, dirs_word[35]) per cond: '
               'median / p10 / p90')
    for i, cn in enumerate(conds):
        c = cos[i]
        out.append('  %s: %.4f / %.4f / %.4f'
                   % (cn, float(np.median(c)),
                      float(np.percentile(c, 10)),
                      float(np.percentile(c, 90))))
    out.append('')

    # per-layer separation ratio: language-signal collapse
    # layer localization (need labels: lab0 = even index?
    # labels come from 2887; reconstruct via proj structure:
    # use attn_proj func profile sign instead. Load labels.
    z87 = np.load(os.path.join(BASE, 'phase2887',
                               'language_axis_mlp',
                               'language_axis_mlp.npz'),
                  allow_pickle=True)
    lab = np.asarray(z87['labels_lang']).astype(int)
    out.append('per-layer sep ratio sep_c(li)/sep_func(li):')
    sf = np.array([attn_proj[ifu][li][lab == 0].mean()
                   - attn_proj[ifu][li][lab == 1].mean()
                   for li in range(36)])
    rows = {}
    for i, cn in enumerate(conds):
        if cn == 'func':
            continue
        sc = np.array([attn_proj[i][li][lab == 0].mean()
                       - attn_proj[i][li][lab == 1].mean()
                       for li in range(36)])
        r = sc / np.where(np.abs(sf) > 1e-9, sf, np.nan)
        rows[cn] = r
        out.append('  %s: L1 %.3f L4 %.3f L8 %.3f L12 %.3f '
                   'L16 %.3f L20 %.3f L24 %.3f L28 %.3f L32 '
                   '%.3f L35 %.3f'
                   % (cn, r[1], r[4], r[8], r[12], r[16],
                      r[20], r[24], r[28], r[32], r[35]))
    out.append('')
    out.append('sep_func(li) profile (first 12 layers): %s'
               % [round(float(v), 2) for v in sf[1:13]])
    out.append('')

    # per-word rewrite outliers (null0): largest |residual|
    i0 = conds.index('null0')
    fit_x = proj[ifu]
    fit_y = proj[i0]
    b = float(((fit_x - fit_x.mean())
               * (fit_y - fit_y.mean())).sum()
              / ((fit_x - fit_x.mean()) ** 2).sum())
    a = fit_y.mean() - b * fit_x.mean()
    resid = fit_y - (a + b * fit_x)
    ix = np.argsort(-np.abs(resid))[:6]
    out.append('null0 rewrite outliers (|resid| top6):')
    for j in ix:
        out.append('  %s: func %.2f null0 %.2f resid %.2f'
                   % (words[j], fit_x[j], fit_y[j],
                      resid[j]))
    out.append('cross-word spearman proj func vs null0: '
               '%.4f | same: %.4f'
               % (spearman(fit_x, fit_y),
                  spearman(fit_x, proj[conds.index('same')])))
    out.append('')

    # same vs null: word-structure retention with labels
    out.append('within-label correlation (structure '
               'retention):')
    for cn in ['same', 'null0']:
        ic = conds.index(cn)
        r0 = spearman(proj[ifu][lab == 0],
                      proj[ic][lab == 0])
        r1 = spearman(proj[ifu][lab == 1],
                      proj[ic][lab == 1])
        out.append('  %s: lab0 %.4f lab1 %.4f'
                   % (cn, r0, r1))
    out.append('')

    rj = json.load(open(os.path.join(OUTD, 'result.json'),
                        encoding='utf-8'))
    out.append('verdict: %s' % rj['final_verdict'])
    out.append('P2 median beta %.4f | P3 energy_dominant '
               'median %s'
               % (rj['P2']['median_beta_null'],
                  rj['P3']['energy_dominant_median']))
    out.append('SHA8: execution %s result %s npz %s '
               'script %s'
               % (sha8(os.path.join(OUTD, 'execution.json')),
                  sha8(os.path.join(OUTD, 'result.json')),
                  sha8(os.path.join(OUTD, 'scale_collapse.npz')),
                  sha8(r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
                       r'\phase2937_scale_collapse.py')))
    with open(REP, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out) + '\n')
    print('OK seal 2937', flush=True)


if __name__ == '__main__':
    main()
