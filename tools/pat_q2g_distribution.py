# -*- coding: utf-8 -*-
"""Q2-G 损伤定律：固定全层覆盖、只扫每层头数，单一变量。

为什么废弃第一版设计
--------------------
第一版想比「128 头集中 4 层」与「128 头散 16 层」，但层集合要在 L0–L19 里随机抽，
而 Q2-D2 已证这个区间层间极不均匀（L4–L7 只掉 0.125，L0–L3 掉 0.674）。
SMOKE 直接踩到：抽到 L5/9/12/15 时 drop=0.000。层身份与分布两个变量混在一起，
这个设计不可用（SMOKE 作废，不入账）。

本版设计：**层集合固定为全 36 层**，每层削 k 头（k 从 2 到 32），头在层内随机。
只有一个变量变——层内覆盖率。层身份混杂被彻底消除。

已有锚点（必须能复现，不复现即设计有错）
--------------------------------------
  k=3.6（128 头随机，Q2-E R128）→ 0.086
  k=7.1（256 头随机，Q2-E R256）→ 0.417
  k=32（4 层集中 100%，Q2-D2）  → 0.674  ← 全层 100% 未必同值，正是本探针要问的

预注册预测
----------
P1 阈值型（我的押注）：k<=4 时 drop<0.05，k=8 后陡升，k=32 达 0.9 以上
P2 线性：drop ≈ c·k，全程无拐点
P3 提前饱和：k=8 之后基本平（drop 台阶在低 k 处走完）

判读（互斥，跑完不许改阈值）
--------------------------
K1 阈值型：存在 k* 使 drop(k*)<0.10 且 drop(2k*)>=0.35（翻倍跨过阈值）
K2 提前饱和：k<=8 段跨度 >=0.30，k>=12 段跨度 <0.15
K3 线性：相邻翻倍的平均比值在 [1.5, 2.5]，且最大单步跨度 <0.30
K4 其他：非上述任一，报实测形状

附带
----
同头数对照：k=4（144 头散 36 层）vs Q2-D2 的 128 头集中 4 层（0.674）。
差值量化「集中 vs 分散」在同一总头数下的效应，这是 D3 定标的直接数据。

用法
----
  SMOKE=1 .venv/Scripts/python.exe -X utf8 tools/pat_q2g_distribution.py
  .venv/Scripts/python.exe -X utf8 tools/pat_q2g_distribution.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'tools')):
    if p not in sys.path:
        sys.path.insert(0, p)

import provenance
from pat_q1_unbind import triples
from pat_q2a_confound import eval_cells, prepare

SEED = 20261109
SMOKE = os.environ.get('SMOKE', '') == '1'
N_TRIP = 4 if SMOKE else 24
REPS = 1 if SMOKE else 3
KS = (2, 8) if SMOKE else (2, 4, 6, 8, 12, 16, 24, 32)


def build_arm(rng, k, n_layers, n_heads):
    """全层覆盖：每层随机 k 头。"""
    sel = []
    for l in range(n_layers):
        hs = rng.choice(n_heads, size=k, replace=False)
        sel.extend((l, int(h)) for h in hs)
    return sel


def main() -> int:
    from server import live_analysis_service as las

    t0 = time.time()
    las._load_async()
    while las._state['status'] in ('idle', 'loading'):
        if time.time() - t0 > 600:
            raise RuntimeError('加载超时')
        time.sleep(2.0)
    model, tok = las._state['model'], las._state['tokenizer']
    dev = next(model.parameters()).device
    meta = las._state['meta']
    n_layers, n_heads = meta['n_layers'], meta['n_heads']

    cells = prepare(tok, triples(N_TRIP))
    acc_base = float(eval_cells(model, dev, cells, [], meta).mean())
    print('评估格 n=%d  基线 %.4f  全 %d 层覆盖  头/层=%d  总头数=%d'
          % (len(cells), acc_base, n_layers, n_heads, n_layers * n_heads))

    rng = np.random.default_rng(SEED)
    out = []
    for k in KS:
        accs = []
        for _ in range(REPS):
            sel = build_arm(rng, k, n_layers, n_heads)
            accs.append(float(eval_cells(model, dev, cells, sel, meta).mean()))
        m = float(np.mean(accs))
        d = acc_base - m
        out.append({'k': k, 'n_heads': k * n_layers, 'coverage': round(k / float(n_heads), 3),
                    'acc': round(m, 4), 'drop': round(d, 4),
                    'range': round(float(max(accs) - min(accs)), 4)})
        print('  k=%-2d  共 %3d 头 (%4.1f%%)  acc=%.4f  drop=%+.4f  极差=%.3f  %s'
              % (k, k * n_layers, 100.0 * k * n_layers / (n_layers * n_heads),
                 m, d, out[-1]['range'], '#' * int(round(d * 40))))

    dk = {x['k']: x['drop'] for x in out}
    print('\n=== 复现锚点核对 ===')
    print('  k=4 -> %.3f（Q2-E 的 k=3.6 实测 0.086，应接近）' % dk.get(4, float('nan')))
    print('  k=8 -> %.3f（Q2-E 的 k=7.1 实测 0.417，应接近）' % dk.get(8, float('nan')))
    print('  全 100%%（k=32）-> %.3f，对比 Q2-D2「4 层集中 100%%」的 0.674' % dk.get(32, float('nan')))

    thr = None
    for x in out:
        if x['k'] * 2 in dk and x['drop'] < 0.10 and dk[x['k'] * 2] >= 0.35:
            thr = x['k']
            break
    low_span = max(dk[k] for k in dk if k <= 8) - min(dk[k] for k in dk if k <= 8)
    high = [dk[k] for k in sorted(dk) if k >= 12]
    high_span = (max(high) - min(high)) if len(high) >= 2 else 0.0
    steps = [dk[sorted(dk)[i + 1]] - dk[sorted(dk)[i]] for i in range(len(dk) - 1)]
    max_step = max(steps) if steps else 0.0

    if thr is not None:
        verdict = 'K1 阈值型：k=%d (drop %.3f) → k=%d (drop %.3f) 跨过阈值' % (thr, dk[thr], thr * 2, dk[thr * 2])
    elif low_span >= 0.30 and high_span < 0.15:
        verdict = 'K2 提前饱和：k<=8 走完 %.3f，k>=12 仅变 %.3f' % (low_span, high_span)
    elif max_step < 0.30:
        verdict = 'K3 线性：最大单步 %.3f，无拐点' % max_step
    else:
        verdict = 'K4 其他：最大单步 %.3f 无翻倍阈值，须按实测形状描述' % max_step
    print('\n=== 判读 ===')
    print('  ' + verdict)

    p = provenance.write_result(
        'pat_q2g_distribution',
        {'n_trip': N_TRIP, 'n_cells': len(cells), 'base_acc': round(acc_base, 4),
         'reps': REPS, 'n_layers': n_layers, 'arms': out,
         'low_span': round(low_span, 4), 'high_span': round(high_span, 4),
         'max_step': round(max_step, 4), 'threshold_k': thr, 'verdict': verdict},
        seed=SEED, tool='tools/pat_q2g_distribution.py')
    print('落盘 %s' % p)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
