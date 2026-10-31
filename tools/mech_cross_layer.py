# -*- coding: utf-8 -*-
"""跨层对应：不同层的簇，是不是在语料的同一批窗口上亮？

签名 = 该簇所有单元 top-32 命中位置所在的 512-token 窗口集合。
若 L14 的某簇与 L24 的某簇窗口 Jaccard 显著高于随机，说明同一「语境模块」在两层都出现。

产出 tempdata/cluster_cross_layer.json
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from server import mech_pair_service as mpair
from server import neuron_census_service as ncs

SRC_SHIFT = 40
WIN = 512
OUT = Path('tempdata/cluster_cross_layer.json')
BLOCKS = [2, 14, 24, 30, 34]


def wins_of(ti_unit, k=32):
    """unit 的 top-k 全局位置 → 窗口 id 集合。"""
    g = ti_unit[:k].astype(np.int64)
    src = g >> SRC_SHIFT
    off = g & ((1 << SRC_SHIFT) - 1)
    return set((src * (1 << 30) + off // WIN).tolist())


def main():
    run = mpair._pick_run(36)
    cj = json.loads(mpair.CLUST_JSON.read_text(encoding='utf-8'))
    cz = np.load(mpair.CLUST_NPZ)
    per = {}
    for b in BLOCKS:
        rec = next((r for r in cj['layers'] if r['block'] == b), None)
        if rec is None:
            continue
        lab = cz['lab_%d' % b].astype(np.int64)
        ti = np.load(ncs.native_root() / '_census' / run['model_key'] / run['run']
                     / ('L%d.npz' % b))['top_idx']
        K = int(rec['k'])
        sets = []
        for c in range(K):
            idx = np.flatnonzero(lab == c)
            # 每簇抽 300 个单元（足够估计窗口集，且控内存）
            rng = np.random.default_rng(20261031 + c)
            take = idx if idx.size <= 300 else rng.choice(idx, size=300, replace=False)
            s = set()
            for u in take:
                s |= wins_of(ti[u])
            sets.append(s)
        per[b] = {'K': K, 'sizes': [int((lab == c).sum()) for c in range(K)], 'sets': sets}
        print('L%-2d 簇窗口集大小 %s..%s' % (b, min(len(s) for s in sets), max(len(s) for s in sets)), flush=True)

    out = {'window_tokens': WIN, 'sample_units_per_cluster': 300, 'pairs': []}
    for i, b1 in enumerate(BLOCKS):
        for b2 in BLOCKS[i + 1:]:
            if b1 not in per or b2 not in per:
                continue
            A, B = per[b1]['sets'], per[b2]['sets']
            M = np.zeros((len(A), len(B)))
            for x in range(len(A)):
                for y in range(len(B)):
                    inter = len(A[x] & B[y])
                    M[x, y] = inter / max(len(A[x]) + len(B[y]) - inter, 1)
            # 随机基线：把窗口 id 在同层内随机打散后重算（近似用期望值）
            base = float(np.median(M))
            best = np.dstack(np.unravel_index(np.argsort(-M, axis=None)[:6], M.shape))[0]
            out['pairs'].append({
                'a': b1, 'b': b2, 'median_jaccard': round(base, 5),
                'max_jaccard': round(float(M.max()), 5), 'lift': round(float(M.max() / base), 2) if base else None,
                'top': [[int(x), int(y), round(float(M[x, y]), 5)] for x, y in best],
            })
            print('L%-2d×L%-2d  median %.5f  max %.5f  lift %.2f  top %s' % (
                b1, b2, base, M.max(), M.max() / base if base else 0,
                [(int(x), int(y)) for x, y in best[:3]]), flush=True)

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('written', OUT, flush=True)


if __name__ == '__main__':
    main()
