# -*- coding: utf-8 -*-
"""diag_2911c.py -- sign-structure profile of the layer
alternation (descriptive). 2910's d=1 margin (row-normalised
(57,1) matrix -> every entry becomes sign(v)) is a SIGN
separation metric: margin_j = P(sign agrees | same) -
P(sign agrees | diff).  Its alternation must therefore be carried
by the class-wise sign balance of the raw response column.
Report per layer: pos_frac per class, their gap |gap| (toy driver
of sign separation), and the same-vs-diff agreement margin
recomputed directly from signs (must match 2910 score_true)."""
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
       r'\diag_2911c.txt')
GROUPS = ('glm4_mlp', 'glm4_attn', 'qwen_mlp', 'qwen_attn')

lines = ['diag_2911c sign-structure profile (descriptive)']
for mdl, npz in SRC.items():
    z = np.load(npz, allow_pickle=True)
    lab = np.asarray(z['labels_lang']).astype(int)
    m0, m1 = lab == 0, lab == 1
    n = len(lab)
    eye = np.eye(n, dtype=bool)
    same = (lab[:, None] == lab[None, :]) & (~eye)
    diff = (~eye) & (~same)
    for ch in ('mlp', 'attn'):
        g = '%s_%s' % (mdl, ch)
        B = z['B_%s' % ch].astype(np.float64)
        w = B.shape[1]
        pf0, pf1, gap, marg = [], [], [], []
        for j in range(w):
            v = B[:, j]
            s = np.sign(v)
            s[s == 0] = 1.0  # zeros treated as + (rare)
            Sm = np.outer(s, s)
            marg.append(float(Sm[same].mean()
                              - Sm[diff].mean()))
            a0 = float((s[m0] > 0).mean())
            a1 = float((s[m1] > 0).mean())
            pf0.append(a0)
            pf1.append(a1)
            gap.append(abs(a0 - a1))
        lines.append('%s: pf0=%s' % (g, ' '.join(
            '%.2f' % v for v in pf0)))
        lines.append('        pf1=%s' % ' '.join(
            '%.2f' % v for v in pf1))
        lines.append('        |gap|=%s' % ' '.join(
            '%.2f' % v for v in gap))
        lines.append('        marg=%s' % ' '.join(
            '%+.4f' % v for v in marg))
        # zigzag test on |gap|: adjacent up/down alternation
        d = np.diff(gap)
        zz = int(np.sum(d[:-1] * d[1:] < 0))
        lines.append('        |gap| zigzag %d/%d'
                     % (zz, len(d) - 1))
with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('OK', OUT)
