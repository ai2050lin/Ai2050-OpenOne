# -*- coding: utf-8 -*-
# 3160 探针2: 验证 3156 H 重算 rank-1 massive 轴协议 (zh/en mid 层 ctx 位移 SVD top-1)
import os, json
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
OUTP = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3160_probe2_out.txt')
out = []

z = np.load(os.path.join(RDIR, 'phase3156', 'g3p1_position_shift_family', 'qwen3-4b', 'collect.npz'))
H = z['H'].astype(np.float64)          # (36, 11, 37, 2560)
arm = z['arm']; lang = z['lang']; k = z['k']; n_tgt = z['n_tgt']
out.append('arm uniq=%s lang uniq=%s k uniq=%s n_tgt uniq=%s' % (
    sorted(set(arm.tolist())), sorted(set(lang.tolist())),
    sorted(set(k.tolist())), sorted(set(n_tgt.tolist()))))
out.append('H shape=%s' % (H.shape,))

sel_A = (arm == 'A')
out.append('A rows=%d B rows=%d' % (sel_A.sum(), (~sel_A).sum()))

def ctx_rank1(layer_list):
    hs = H[:, -1, :, :]                               # last 目标 token: (36, 37, D)
    m0 = np.stack([hs[i, layer_list].mean(0) for i in range(H.shape[0]) if sel_A[i] and k[i] == 0])
    m1 = np.stack([hs[i, layer_list].mean(0) for i in range(H.shape[0]) if sel_A[i] and k[i] >= 1])
    disp = m1.mean(0) - m0.mean(0)                    # (len(layer_list), D)
    disp = disp.reshape(1, -1) if disp.ndim == 1 else disp
    U, S, Vt = np.linalg.svd(disp, full_matrices=False)
    var = S ** 2
    top2 = (var[0] + var[1]) / var.sum() if len(S) > 1 else 1.0
    out.append('layers=%s disp=%s S[:3]=%s top1_share=%.5f top2=%.5f' % (
        layer_list, disp.shape, np.round(S[:3], 1).tolist(),
        var[0] / var.sum(), top2))
    return Vt[0], var[0] / var.sum()

for layers in ([7], [6, 7, 8], [7, 18, 35]):
    r1, sh = ctx_rank1(layers)
    out.append('  -> r1 norm=%.3f share=%.5f' % (np.linalg.norm(r1), sh))

# 分 zh/en 对比 result.json subspace (zh_mid top1=0.99995, en_mid top1=0.99958)
for lg in ('zh', 'en'):
    sel = sel_A & (lang == lg)
    m0 = np.stack([H[i, -1, [7], :].mean(0) for i in range(H.shape[0]) if sel[i] and k[i] == 0])
    m1 = np.stack([H[i, -1, [7], :].mean(0) for i in range(H.shape[0]) if sel[i] and k[i] >= 1])
    disp = m1.mean(0) - m0.mean(0)
    disp = disp.reshape(1, -1) if disp.ndim == 1 else disp
    U, S, Vt = np.linalg.svd(disp, full_matrices=False)
    var = S ** 2
    out.append('%s L7 top1_share=%.5f (expect ~0.999)' % (lg, var[0] / var.sum()))
    np.save(os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3160_r1_%s.npy' % lg), Vt[0])
    out.append('  r1 saved, norm=%.4f absmax=%.1f argmax=%d' % (
        np.linalg.norm(Vt[0]), np.abs(Vt[0]).max(), int(np.abs(Vt[0]).argmax())))

# 3157 H 锚 massive 维度 (per-model 补充量用)
z7 = np.load(os.path.join(RDIR, 'phase3157', 'g2p2_transform_algebra_commutator', 'qwen3-4b', 'collect.npz'))
H7 = z7['H'].astype(np.float64)     # (128, 37, 2560)
am = np.abs(H7[:, 18, :]).mean(0)   # 槽 18 = L_mid
d1 = int(am.argmax())
out.append('3157 qwen3-4b slot18 mean|h| max dim=%d val=%.1f (2nd=%d val=%.1f)' % (
    d1, am[d1], int((-am).argsort()[1]), np.sort(am)[-2]))

with open(OUTP, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out) + '\n')
print('PROBE2 DONE')
