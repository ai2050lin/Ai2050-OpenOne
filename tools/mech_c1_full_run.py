# -*- coding: utf-8 -*-
"""C1 全量运行 + 组方向与层主写方向的重叠诊断（一次加载模型做完）。"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from server import mech_ablate_service as ma
from server import mech_pair_service as mpair

ME = Path('data/mech')
BLOCK = 24

t0 = time.time()
model, tok, device, meta, model_name = mpair._wait_ready()
cj = json.loads(mpair.CLUST_JSON.read_text(encoding='utf-8'))
rec = next(r for r in cj['layers'] if r['block'] == BLOCK)
lab = np.load(mpair.CLUST_NPZ)['lab_%d' % BLOCK].astype(np.int64)
K = int(rec['k'])

Wd = model.model.layers[BLOCK].mlp.down_proj.weight.detach().float()      # [d, d_ff]
U = Wd.t()
U = U / U.norm(dim=1, keepdim=True).clamp_min(1e-9)
# 层的主写方向（前 5 个左奇异向量）
V, S, _ = torch.svd_lowrank(Wd, q=8)
V = V[:, :5]                                                              # [d,5]
top = (V.t() @ U.t()).t().abs()                                           # [d_ff,5] 每单元与主方向的重叠
recs = []
for c in range(K):
    m = lab == c
    uc = U[m].mean(0)
    nrm = float(uc.norm())
    uch = uc / (nrm + 1e-12)
    recs.append({'cluster': c, 'size': int(m.sum()),
                 'mean_write_norm': round(nrm, 5),
                 # 随机方向均值长度的期望 ≈ 1/sqrt(size)
                 'random_expect': round(1.0 / np.sqrt(int(m.sum())), 5),
                 'cos_v1': round(float((uch @ V[:, 0]).abs()), 4),
                 'overlap_top5': round(float(torch.linalg.vector_norm(V.t() @ uch)), 4),
                 'unit_top5_mean': round(float(top[m].mean()), 4)})
diag = {'block': BLOCK, 'k': K,
        'singular_values_top5': [round(float(x), 3) for x in S[:5]],
        'clusters': recs}
(ME / 'cluster_dir_overlap.json').write_text(
    json.dumps(diag, ensure_ascii=False, indent=1), encoding='utf-8')
print('overlap diag done %.1fs' % (time.time() - t0))
for r in recs[:8]:
    print(' C%-2d size %4d |u| %.5f (rand %.5f) cos_v1 %.3f top5 %.3f' % (
        r['cluster'], r['size'], r['mean_write_norm'], r['random_expect'],
        r['cos_v1'], r['overlap_top5']))

ma._run(BLOCK, 'C-Eval val', 0, 2)
print('ablate', ma._STATE['state'], ma._STATE['error'], '%.0fs' % (time.time() - t0))
