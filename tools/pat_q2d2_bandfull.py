# -*- coding: utf-8 -*-
"""Q2-D2 整带全消融：层位特异性的终版测量。

为什么必须做
------------
Q2-A3 用「单层 32 头」得到「35/36 层 ≤0.007」，据此我写了「唯一层位效应在 L0」。
Q2-D 打翻了这个概括：L21–L24 四层全消（128 头）掉 0.451，L29–L32 四层全消
（同样 128 头）掉 0.000。同层数、同头数、同层内比例，差 0.451。

即：单层消融被补偿掩盖，层位差异只有整带全消才显形。本探针把每个带都做 p=1，
补齐这条曲线，不再从单层数据外推。

预注册判据
----------
F1 承载带唯一：仅 L21–L24 掉 >=0.15，其余带 <0.05
F2 承载带更宽：连续两带（如 L21–L28）都 >=0.15
F3 广泛承载：>=3 个带都 >=0.15
F4 无层位效应：所有带 <0.05（与 Q2-D 的 0.451 vs 0.000 矛盾，须查）

用法
----
  .venv/Scripts/python.exe -X utf8 tools/pat_q2d2_bandfull.py
"""
from __future__ import annotations

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

SEED = 20261106
SMOKE = os.environ.get('SMOKE', '') == '1'
N_TRIP = 4 if SMOKE else 24
REPS = 1 if SMOKE else 3
BANDS = ((0, 3), (4, 7), (8, 11), (12, 15), (16, 19),
         (20, 23), (24, 27), (28, 31), (32, 35))


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
    allh = [(l, h) for l in range(meta['n_layers']) for h in range(meta['n_heads'])]

    cells = prepare(tok, triples(N_TRIP))
    base = eval_cells(model, dev, cells, [], meta)
    acc_base = float(base.mean())
    print('评估格 n=%d  基线 %.4f  每带 4 层 x 32 头 = 128 头 (11.1%%)'
          % (len(cells), acc_base))

    rng = np.random.default_rng(SEED)
    out = []
    for band in BANDS:
        sel = [(l, h) for (l, h) in allh if band[0] <= l <= band[1]]
        accs = [float(eval_cells(model, dev, cells, sel, meta).mean())
                for _ in range(REPS)]
        m = float(np.mean(accs))
        d = acc_base - m
        out.append({'band': list(band), 'n_heads': len(sel),
                    'acc': round(m, 4), 'drop': round(d, 4),
                    'sd': round(float(np.std(accs)), 4)})
        bar = '#' * int(round(d * 100 / 5))
        print('  L%-2d–L%-2d  n=%-3d  acc=%.4f  drop=%+.4f  (sd %.3f) %s'
              % (band[0], band[1], len(sel), m, d, float(np.std(accs)), bar))

    big = [x for x in out if x['drop'] >= 0.15]
    if not big:
        verdict = 'F4 无层位效应（与 Q2-D 的 0.451 vs 0.000 矛盾，须查）'
    elif len(big) >= 3:
        verdict = 'F3 广泛承载：%d 个带均 >=0.15' % len(big)
    elif len(big) == 1:
        verdict = 'F1 承载带唯一：L%d–L%d 掉 %.3f' % (big[0]['band'][0], big[0]['band'][1], big[0]['drop'])
    else:
        verdict = 'F2 承载带更宽：%s' % ', '.join('L%d–L%d' % (x['band'][0], x['band'][1]) for x in big)
    print('=== 判读 ===')
    print('  ' + verdict)

    p = provenance.write_result('pat_q2d2_bandfull',
                                {'n_trip': N_TRIP, 'n_cells': len(cells),
                                 'base_acc': round(acc_base, 4), 'reps': REPS,
                                 'bands': out, 'verdict': verdict},
                                seed=SEED, tool='tools/pat_q2d2_bandfull.py')
    print('落盘 %s' % p)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
