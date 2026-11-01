# -*- coding: utf-8 -*-
"""Q2-F 段内异质性：同密度随机抽样为什么能给出 0.29 到 0.94。

起因（Q2-E 观察到的问题，不是新方向）
--------------------------------------
Q2-E 在 L0–L3 内每层随机抽 16 头（层内 50%，共 64 头），三次抽样正确率分别是
0.2917 / 0.3472 / 0.9375。同密度、同层、同样随机，结果跨越 0.65。
因此 Q2-E 的 gap_half（用三次均值算）不可作为定量结论引用，
同时这个跨度本身可能就是「段内少数头承担多数」的直接证据。

本探针只做一件事：把 64 头抽样的分布测出来，判它是抽样噪声还是真异质。

预注册设计
----------
- 12 次独立抽样，L0–L3 每层随机 16 头（保持层平衡，共 64 头 = 层内 50%）
- 12 次独立抽样，L16–L19 每层随机 16 头
- 每次同时算 AB128 = A∪B，得**逐次** gap = drop(AB) - max(drop(A), drop(B))
  （Q2-E 用均值算 gap，本探针改为逐次配对，不会被单次极值带偏）
- 基线只算一次

判读（互斥，跑完不许改阈值）
--------------------------
H1 段内高度异质：某带 12 次抽样极差 >= 0.40，且最小/最大不连续（存在"近乎无损"的抽样）
   => 该带的因果承载集中在少数头上，必须从"整带"转向"少数头定位"
H2 中度异质：极差在 0.15–0.40
H3 抽样噪声：极差 < 0.15，均值可用
H4 双峰抽样：极差 >= 0.40 且中位明显偏离均值（>=0.15），说明存在两类抽样

附带统计
--------
- 每带 12 次的 mean / median / sd / min / max / IQR
- 逐次 gap 的 median 与 [p10, p90] 区间（替代 Q2-E 的均值口径）
- 逐次 drop(AB) 是否稳定低于 drop(A)+drop(B)（共用通路的必要条件）

用法
----
  SMOKE=1 .venv/Scripts/python.exe -X utf8 tools/pat_q2f_hetero.py
  .venv/Scripts/python.exe -X utf8 tools/pat_q2f_hetero.py
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
from pat_q2e_additivity import band_heads, sample_per_layer

SEED = 20261108
SMOKE = os.environ.get('SMOKE', '') == '1'
N_TRIP = 4 if SMOKE else 24
REPS = 4 if SMOKE else 12
BAND_A = (0, 3)
BAND_B = (16, 19)


def stats(x):
    a = np.asarray(x, dtype=np.float64)
    return {'mean': round(float(a.mean()), 4), 'median': round(float(np.median(a)), 4),
            'sd': round(float(a.std(ddof=1)) if len(a) > 1 else 0.0, 4),
            'min': round(float(a.min()), 4), 'max': round(float(a.max()), 4),
            'range': round(float(a.max() - a.min()), 4)}


def verdict_for(sA, sB):
    rng_max = max(sA['range'], sB['range'])
    sk = max(abs(sA['median'] - sA['mean']), abs(sB['median'] - sB['mean']))
    if rng_max >= 0.40 and sk >= 0.15:
        return 'H4 双峰抽样：极差 %.2f 且中位偏离均值 %.2f' % (rng_max, sk)
    if rng_max >= 0.40:
        return 'H1 段内高度异质：极差 %.2f' % rng_max
    if rng_max >= 0.15:
        return 'H2 中度异质：极差 %.2f' % rng_max
    return 'H3 抽样噪声：极差 %.2f，均值可用' % rng_max


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
    print('评估格 n=%d  基线 %.4f  抽样 %d 次  每带 64 头（层内 50%%）'
          % (len(cells), acc_base, REPS))

    rng = np.random.default_rng(SEED)
    rows = []
    for i in range(REPS):
        a64 = sample_per_layer(rng, BAND_A, 16, n_heads)
        b64 = sample_per_layer(rng, BAND_B, 16, n_heads)
        accA = float(eval_cells(model, dev, cells, a64, meta).mean())
        accB = float(eval_cells(model, dev, cells, b64, meta).mean())
        accAB = float(eval_cells(model, dev, cells, a64 + b64, meta).mean())
        dA, dB, dAB = acc_base - accA, acc_base - accB, acc_base - accAB
        gap = dAB - max(dA, dB)
        rows.append({'rep': i, 'acc_A': round(accA, 4), 'acc_B': round(accB, 4),
                     'acc_AB': round(accAB, 4), 'gap': round(gap, 4),
                     'subadditive': bool(dAB < dA + dB - 0.02)})
        print('  rep%-2d  A=%.4f  B=%.4f  AB=%.4f  gap=%+.4f  %s'
              % (i, accA, accB, accAB, gap, '共用' if dAB < dA + dB - 0.02 else '可加?'))

    accA = [r['acc_A'] for r in rows]
    accB = [r['acc_B'] for r in rows]
    gaps = [r['gap'] for r in rows]
    sA, sB = stats(accA), stats(accB)
    print('\n=== 分布 ===')
    for nm, s in (('A64 (L0–L3)', sA), ('B64 (L16–L19)', sB)):
        print('  %-14s mean=%.4f median=%.4f sd=%.4f min=%.4f max=%.4f range=%.4f'
              % (nm, s['mean'], s['median'], s['sd'], s['min'], s['max'], s['range']))
    g = np.asarray(gaps)
    print('  逐次 gap: median=%+.4f  [p10=%+.4f, p90=%+.4f]  gap>0.10 的次数=%d/%d'
          % (float(np.median(g)), float(np.percentile(g, 10)), float(np.percentile(g, 90)),
             int((g > 0.10).sum()), len(g)))
    n_sub = sum(1 for r in rows if r['subadditive'])
    print('  逐次可加性检验：AB 严格小于 A+B 的次数 = %d/%d' % (n_sub, len(rows)))

    v = verdict_for(sA, sB)
    print('\n=== 判读 ===')
    print('  ' + v)

    p = provenance.write_result(
        'pat_q2f_hetero',
        {'n_trip': N_TRIP, 'n_cells': len(cells), 'base_acc': round(acc_base, 4), 'reps': REPS,
         'dist_A64': sA, 'dist_B64': sB, 'rows': rows,
         'gap_median': round(float(np.median(g)), 4),
         'gap_p10': round(float(np.percentile(g, 10)), 4),
         'gap_p90': round(float(np.percentile(g, 90)), 4),
         'n_subadditive': n_sub, 'verdict': v},
        seed=SEED, tool='tools/pat_q2f_hetero.py')
    print('落盘 %s' % p)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
