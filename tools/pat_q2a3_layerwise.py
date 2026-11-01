# -*- coding: utf-8 -*-
"""Q2-A3 逐层全消融：把「层位特异性」从「消融密度」里彻底剥出来。

为什么这一轮是判决性的
----------------------
Q2-A2 用了「带内全部通过头」，但各带头数不等（5/71/73/69/45），密度仍是混杂。
本轮把消融量固定成「整层 32 头」——每层消融的头数、占总头数比例、层内覆盖率
三者全部相同（32/1152 = 2.78%，层内 100%），唯一的变量就是「消哪一层」。

于是曲线形状只可能来自层位本身：
  若 L21–L24 出现显著凹陷 => 承载段真实存在，与特征筛选无关
  若各层凹陷相近        => 承载是连续的，无特异段
  若全平                => Q2-A2 的 0.22 是消融数量的效应，层位也不承载

预注册判据
----------
L1 特异段：存在连续 <=4 层，其 drop 均值 >= 0.10 且为该曲线最大段
L2 连续梯度：drop 随层单调上升/下降，无明显平台（最大层 drop < 0.10 但累积 >= 0.15）
L3 全平：所有层 drop < 0.05

同时给「逐层累积」曲线（L0..l 全部消融），与单层曲线互补：
单层看「独立贡献」，累积看「可补偿性」。累积非单调是允许的，如实报告，不解释。

用法
----
  .venv/Scripts/python.exe -X utf8 tools/pat_q2a3_layerwise.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'tools')):
    if p not in sys.path:
        sys.path.insert(0, p)

import provenance
from pat_q1_unbind import triples
from pat_q2a_confound import eval_cells, prepare

SEED = 20261104
SMOKE = os.environ.get('SMOKE', '') == '1'
N_TRIP = 4 if SMOKE else 24


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
    L, H = meta['n_layers'], meta['n_heads']

    cells = prepare(tok, triples(N_TRIP))
    base = eval_cells(model, dev, cells, [], meta)
    acc_base = float(base.mean())
    print('评估格 n=%d  层数 %d  每层头数 %d  基线 %.4f'
          % (len(cells), L, H, acc_base))

    print('=== 1 单层全消融（每层 32 头，占比恒定 2.78%）===')
    per_layer = []
    for l in range(L):
        acc = float(eval_cells(model, dev, cells, [(l, h) for h in range(H)], meta).mean())
        d = acc_base - acc
        per_layer.append({'layer': l, 'acc': round(acc, 4), 'drop': round(d, 4)})
        bar = '#' * int(round(d * 100 / 4))
        print('  L%-2d  acc=%.4f  drop=%+.4f  %s' % (l, acc, d, bar))

    drops = np.array([x['drop'] for x in per_layer])
    print('  单层最大 drop=%.4f @L%d  均值=%.4f'
          % (drops.max(), int(drops.argmax()), drops.mean()))

    print('=== 2 逐层累积（L0..l 全部消融）===')
    cum = []
    for l in range(L):
        acc = float(eval_cells(model, dev, cells,
                               [(ll, h) for ll in range(l + 1) for h in range(H)], meta).mean())
        cum.append({'upto': l, 'acc': round(acc, 4), 'drop': round(acc_base - acc, 4)})
        if l % 4 == 3 or l == L - 1:
            print('  <=L%-2d  acc=%.4f  掉%.4f' % (l, acc, acc_base - acc))

    print('=== 3 预注册判读 ===')
    # 找 <=4 层的连续段，其 drop 均值最大
    best = (0.0, 0, 0)
    for w in range(1, 5):
        for s in range(0, L - w + 1):
            m = float(drops[s:s + w].mean())
            if m > best[0]:
                best = (m, s, s + w - 1)
    seg_mean, a, b = best
    print('  最佳连续段 L%d–L%d（宽%d）均值 drop=%.4f' % (a, b, b - a + 1, seg_mean))
    if drops.max() < 0.05:
        verdict = 'L3 全平：单层消融均 <0.05，层位不承载（Q2-A2 的 0.22 为消融数量效应）'
    elif seg_mean >= 0.10 and seg_mean >= drops.max() - 1e-9:
        verdict = 'L1 特异段：L%d–L%d 均值 drop %.3f 为全曲线最大段' % (a, b, seg_mean)
    else:
        verdict = 'L2 连续梯度：最大单层 %.3f，未见平台段' % drops.max()
    print('  ' + verdict)

    out = {'n_trip': N_TRIP, 'n_cells': len(cells), 'base_acc': round(acc_base, 4),
           'n_layers': L, 'n_heads': H, 'per_heads_per_layer': H,
           'per_layer': per_layer, 'cumulative': cum,
           'max_drop': round(float(drops.max()), 4),
           'argmax_layer': int(drops.argmax()),
           'best_segment': [a, b, round(seg_mean, 4)],
           'verdict': verdict,
           'verdict_rule': 'L3: max<0.05; L1: 段均值>=0.10 且为该段最大; 其余 L2'}
    p = provenance.write_result('pat_q2a3_layerwise', out, seed=SEED,
                                tool='tools/pat_q2a3_layerwise.py')
    print('落盘 %s' % p)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
