# -*- coding: utf-8 -*-
# 3160 探针2c: 穷举 rank-1 轴定义 + 检查 en 行是否与 zh 重复
import os
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
OUTP = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3160_probe2c_out.txt')
out = []

z = np.load(os.path.join(RDIR, 'phase3156', 'g3p1_position_shift_family', 'qwen3-4b', 'collect.npz'))
H = z['H'].astype(np.float64)
arm = z['arm']; lang = z['lang']; k = z['k']

def svd_share(X):
    X = X - X.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = S ** 2
    t2 = (var[0] + var[1]) / var.sum() if len(S) > 1 else 1.0
    return var[0] / var.sum() if var.sum() > 0 else float('nan'), t2, Vt[0]

out.append('zh_row1 vs en_row1 identical: %s' % np.array_equal(H[1], H[19]))
out.append('zh_row1 vs en_row1 maxdiff: %.3e' % np.abs(H[1] - H[19]).max())
out.append('zh_A_k0 vs en_A_k0 maxdiff: %.3e' % np.abs(H[0] - H[18]).max())

LAYER = 7
base_zh = H[0, -1, LAYER, :]
base_en = H[18, -1, LAYER, :]
for lg, base in (('zh', base_zh), ('en', base_en)):
    sel = (arm == 'A') & (lang == lg) & (k >= 1)
    # D1: last-token 逐行位移
    X1 = H[sel, -1, LAYER, :] - base[None, :]
    # D2: 全 token 位逐行位移 (n1*ntok, D)
    X2 = H[sel, :, LAYER, :].reshape(-1, H.shape[3]) - base[None, :]
    # D3: k 均值位移 (8, D) —— 全 token
    X3 = np.stack([H[i, :, LAYER, :].mean(0) - base for i in np.where(sel)[0]])
    # D4: 跨层拼接 last-token (k x 3层, D)
    X4 = np.concatenate([H[sel, -1, L, :] - H[0 if lg == 'zh' else 18, -1, L, :] for L in (6, 7, 8)], 0)
    for tag, X in (('D1_last', X1), ('D2_alltok', X2), ('D3_kmean', X3), ('D4_L678', X4)):
        s, t2, _ = svd_share(X)
        out.append('[%s %s] X=%s share=%.5f top2=%.5f' % (lg, tag, X.shape, s, t2))

# B 臂 (位置重置) 与 A 臂差: k>0
for lg in ('zh', 'en'):
    selA = (arm == 'A') & (lang == lg) & (k >= 1)
    selB = (arm == 'B') & (lang == lg) & (k >= 1)
    X5 = H[selA, -1, LAYER, :] - H[selB, -1, LAYER, :]
    s, t2, _ = svd_share(X5)
    out.append('[%s D5_AminusB] X=%s share=%.5f top2=%.5f' % (lg, X5.shape, s, t2))

with open(OUTP, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out) + '\n')
print('PROBE2C DONE')
