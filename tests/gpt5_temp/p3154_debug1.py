# -*- coding: utf-8 -*-
# p3154_debug1.py: 诊断 G 对比度为 0 的原因 —— 直接从 npz 检查 con/contra 均值差
import numpy as np, json, os

BASE = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913\phase3154\g1p4_mfd_multifactor_disentangle\qwen3-4b\smoke'
z = np.load(os.path.join(BASE, 'collect_smoke.npz'))
H = z['H'].astype(np.float32)
g = z['g'].astype(int)
ho = z['ho'].astype(int)
out = []
out.append('H=%s g=%s ho=%s' % (H.shape, np.bincount(g), np.bincount(ho)))
tr = ho == 0
for k in (0, 3, 18, 35):
    X = H[tr, k, :]
    gg = g[tr]
    dmean = X[gg == 1].mean(0) - X[gg == 0].mean(0)
    out.append('k=%2d ||mean_contra - mean_con||=%.6e  ||mean||=%.4f' %
               (k, float(np.linalg.norm(dmean)), float(np.linalg.norm(X.mean(0)))))
# 检查前 8 行的 g 分布（行序应为 ti,l,s,g,ci）
out.append('g first 16: %s' % g[:16].tolist())
out.append('ho first 16: %s' % ho[:16].tolist())
# 行文本对齐检查: materials.json 前 4 行
mj = json.load(open(os.path.join(BASE, 'materials.json'), encoding='utf-8'))
for r in mj['rows'][:4]:
    out.append('row: ho=%s g=%s %s' % (r['ho'], r['logic'], r['prompt'][:40]))
open(os.path.join(BASE, '..', '..', '..', '..', '..', 'gpt5_temp', 'p3154_debug1_out.txt'),
     'w', encoding='utf-8').write('\n'.join(out))
print('\n'.join(out))
