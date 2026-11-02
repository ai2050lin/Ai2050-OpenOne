# -*- coding: utf-8 -*-
"""FU-v1 事后诊断（不进判定）：固定语法槽位（g = 动词 / 名词），在 2×2（风格 × 逻辑）内分解：
可加份额 = (SS_s + SS_l) / (SS_s + SS_l + SS_sl)，随机期望 2 / 3。输出（候选对数概率，按格中心化）与各层隐状态，确认集。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from shared.dist_protocol import fusion as F  # noqa: E402
import fu_materials_v1 as MAT  # noqa: E402
import fusion_v1 as FU  # noqa: E402


def two_by_two(Y4):
    """Y4：[4, m]，格序 2s + l。"""
    Yc = Y4 - Y4.mean(0)
    s = np.array([-1, -1, 1, 1.]); l = np.array([-1, 1, -1, 1.])
    est = np.stack([s, l, s * l]) @ Yc / 4
    ss = 4 * (est ** 2).sum(1)
    return ss


def main(model_path):
    snap = FU.snapshot(model_path)
    od = FU.NATIVE / snap[:16] / 'FU-v1'
    H = np.load(od / 'states.npy').astype(np.float64)
    z = np.load(od / 'logp.npz')
    conf = MAT.split()['conf']
    L = H.shape[2] - 1
    out = {}
    for g in (0, 1):
        rows = [c for c in range(8) if (c & 1) == g]                 # 格序 4s + 2l + g → 子格序 2s + l
        so = sum(two_by_two(F.center_cells(z[f'lp{i}'])[rows]) for i in conf)
        st = [sum(two_by_two(H[i, rows, l]) for i in conf) for l in range(L + 1)]
        share = lambda ss: float((ss[0] + ss[1]) / ss.sum())
        out[['verb', 'noun'][g]] = {'output_additive': share(so), 'output_ss_share': (so / so.sum()).tolist(),
                                    'state_additive_by_layer': [share(x) if x.sum() > 0 else None for x in st]}
        print(model_path, ['verb', 'noun'][g], 'out', round(share(so), 3), [round(v, 3) for v in so / so.sum()],
              'states', [round(share(x), 2) if x.sum() > 0 else None for x in st][::3], flush=True)
    out['note'] = '事后诊断，不进 FU-v1 判定'
    FU.save(model_path, {'posthoc_within_slot': out})


if __name__ == '__main__':
    main(sys.argv[1])
