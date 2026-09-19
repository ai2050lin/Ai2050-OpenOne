# -*- coding: utf-8 -*-
"""diag_2911b.py -- post-hoc factor decomposition of the layer
alternation (descriptive, feeds MEMO 2911 + 2912 design).

margin_j (d=1 cosine separation at layer j) alternates sign for
qwen_attn (8/9 flips, p=0.0195) while delta signs and B column
correlations do not.  Which factor drives it?  For the 2896-family
on a 1-d projection, margin_j ~ f(|delta_j|, scatter_j) with
f increasing in |delta| and decreasing in scatter.  Per layer,
report:
  dnorm_j = |delta_j|                    (class-mean shift)
  spool_j = pooled within-class RMS      (scatter)
  ratio_j = dnorm_j / spool_j            (toy SNR)
and the flip rates of each series, to see which factor carries the
alternation.  All four groups, descriptive only.
"""
import json
import os

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC = {
    'glm4': os.path.join(BASE, 'phase2902',
                         'glm4_channel_jacobian_asymmetry',
                         'glm4_channel_jacobian_asymmetry.npz'),
    'qwen': os.path.join(BASE, 'phase2903',
                         'qwen_channel_jacobian_decomposition',
                         'qwen_channel_jacobian_decomposition.npz'),
}
OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\diag_2911b.txt')
GROUPS = ('glm4_mlp', 'glm4_attn', 'qwen_mlp', 'qwen_attn')


def flips(vals):
    s = np.sign(np.asarray(vals, float))
    f = n = 0
    for j in range(len(s) - 1):
        if s[j] == 0 or s[j + 1] == 0:
            continue
        n += 1
        if s[j] != s[j + 1]:
            f += 1
    return f, n


lines = ['diag_2911b factor decomposition (descriptive)']
for mdl, npz in SRC.items():
    z = np.load(npz, allow_pickle=True)
    lab = np.asarray(z['labels_lang']).astype(int)
    for ch in ('mlp', 'attn'):
        g = '%s_%s' % (mdl, ch)
        B = z['B_%s' % ch].astype(np.float64)
        m0, m1 = lab == 0, lab == 1
        w = B.shape[1]
        dnorm, spool, ratio, margin1d = [], [], [], []
        for j in range(w):
            col = B[:, j]
            dj = abs(col[m1].mean() - col[m0].mean())
            sj = float(np.sqrt(0.5 * (col[m0].var()
                                      + col[m1].var())))
            u = col / max(np.abs(col).max(), 1e-30)
            S = np.outer(u, u)
            n = len(lab)
            eye = np.eye(n, dtype=bool)
            same = (lab[:, None] == lab[None, :]) & (~eye)
            diff = (~eye) & (~same)
            mj = float(S[same].mean() - S[diff].mean())
            dnorm.append(dj)
            spool.append(sj)
            ratio.append(dj / max(sj, 1e-30))
            margin1d.append(mj)
        fd, nd = flips(dnorm)
        fs, ns = flips(spool)
        fr, nr = flips(ratio)
        fm, nm = flips(margin1d)
        lines.append(
            '%s: dnorm=%s spool=%s ratio=%s margin1d=%s'
            % (g,
               ' '.join('%+.4f' % v for v in dnorm),
               ' '.join('%+.4f' % v for v in spool),
               ' '.join('%+.4f' % v for v in ratio),
               ' '.join('%+.4f' % v for v in margin1d)))
        lines.append(
            '    flips: dnorm %d/%d, spool %d/%d, ratio %d/%d, '
            'margin1d %d/%d'
            % (fd, nd, fs, ns, fr, nr, fm, nm))
with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('OK', OUT)
