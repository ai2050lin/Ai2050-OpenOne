# -*- coding: utf-8 -*-
# 3160 探针2b: 修正 rank-1 重算协议 —— 逐行位移矩阵 SVD; 诊断 en nan
import os
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
OUTP = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3160_probe2b_out.txt')
out = []

z = np.load(os.path.join(RDIR, 'phase3156', 'g3p1_position_shift_family', 'qwen3-4b', 'collect.npz'))
H = z['H'].astype(np.float64)
arm = z['arm']; lang = z['lang']; k = z['k']
out.append('row order dump (i, lang, arm, k):')
for i in range(36):
    out.append('  %2d %s %s %d' % (i, lang[i], arm[i], k[i]))

LAYER = 7
for lg in ('zh', 'en'):
    sel0 = (arm == 'A') & (lang == lg) & (k == 0)
    sel1 = (arm == 'A') & (lang == lg) & (k >= 1)
    out.append('[%s] n0=%d n1=%d  H finite: A0=%s A1=%s' % (
        lg, sel0.sum(), sel1.sum(),
        np.isfinite(H[sel0, -1, LAYER, :]).all(), np.isfinite(H[sel1, -1, LAYER, :]).all()))
    base = H[sel0, -1, LAYER, :].mean(0)
    X = H[sel1, -1, LAYER, :] - base[None, :]          # (n1, D) 逐行位移
    out.append('  X shape=%s finite=%s norm_rows=%s' % (
        X.shape, np.isfinite(X).all(),
        np.round(np.linalg.norm(X, axis=1), 2).tolist()))
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = S ** 2
    out.append('  S[:3]=%s top1_share=%.5f top2=%.5f' % (
        np.round(S[:3], 1).tolist(), var[0] / var.sum(),
        (var[0] + var[1]) / var.sum() if len(S) > 1 else 1.0))
    np.save(os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3160_r1_%s.npy' % lg), Vt[0])

with open(OUTP, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out) + '\n')
print('PROBE2B DONE')
