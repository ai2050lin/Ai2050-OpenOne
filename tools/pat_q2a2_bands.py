# -*- coding: utf-8 -*-
"""Q2-A2 层带扫描：角色解绑的因果承载落在哪一段。

为什么需要这一轮
----------------
Q2-A 的 2x2 对照给出了方向但被消融密度混杂：低层那一带每层只有 16 头，
k=60 时等于抽掉该段 42%，随机头也会掉（B 的 5 次重复从 0.438 到 0.993）。
密度一混，层位效应与筛选效应就分不开。

本轮的解法是把 k 固定成「整个层带的全部通过头」，不再抽 k：
每个层带内头数固定、密度口径一致，带与带之间的比较不再受抽样量影响。

预注册判据（跑之前写死）
------------------------
P1 承载段集中：存在一个层带，单独消融即造成 acc 下降 >= 0.15
P2 承载段分散：所有层带单独消融均 < 0.15，但累积消融造成大降
P3 无可点名承载：所有层带（含累积）均 < 0.10

每个层带同时跑「同规模、同层分布的随机头」对照，只有 ok 显著差于 rand 才允许
把该带的效应记在「角色跟随性」上，否则记为「该带被消融的头数本身」。

用法
----
  .venv/Scripts/python.exe -X utf8 tools/pat_q2a2_bands.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'tools')):
    if p not in sys.path:
        sys.path.insert(0, p)

import provenance
from pat_q1_unbind import FORMS, QUERIES, ROLES
from pat_q2a_confound import eval_cells, load_ok_heads, paired, prepare
from pat_q1_unbind import triples

SEED = 20261103
SMOKE = os.environ.get('SMOKE', '') == '1'
N_TRIP = 4 if SMOKE else 24
REPS = 1 if SMOKE else 5
N_BOOT = 400 if SMOKE else 1000
BANDS = ((17, 20), (21, 24), (25, 28), (29, 32), (33, 35))


def in_band(h, band):
    return band[0] <= h[0] <= band[1]


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

    cells = prepare(tok, triples(N_TRIP))
    ok_all = load_ok_heads()
    allh = [(l, h) for l in range(meta['n_layers']) for h in range(meta['n_heads'])]
    rng = np.random.default_rng(SEED)

    base = eval_cells(model, dev, cells, [], meta)
    acc_base = float(base.mean())
    print('评估格 n=%d  基线 %.4f' % (len(cells), acc_base))

    print('=== 1 单带消融（带内全部通过头 vs 同规模同层随机头）===')
    bands_out = []
    for band in BANDS:
        ok_b = [h for h in ok_all if in_band(h, band)]
        if not ok_b:
            print('  L%d–L%d 无通过头，跳过' % band)
            continue
        acc_ok = eval_cells(model, dev, cells, ok_b, meta)
        rands = []
        for _ in range(REPS):
            pool = [h for h in allh if in_band(h, band)]
            idx = rng.choice(len(pool), size=min(len(ok_b), len(pool)), replace=False)
            rands.append(eval_cells(model, dev, cells, [pool[i] for i in idx], meta))
        acc_rand = np.mean(rands, axis=0)
        g = paired(acc_ok, acc_rand, rng)
        drop = acc_base - float(acc_ok.mean())
        drop_r = acc_base - float(acc_rand.mean())
        print('  L%-2d–L%-2d  n_ok=%-3d  ok=%.4f (掉%.3f)  rand=%.4f (掉%.3f)  差=%+.4f CI[%+.4f,%+.4f]'
              % (band[0], band[1], len(ok_b), acc_ok.mean(), drop,
                 acc_rand.mean(), drop_r, g[0], g[1], g[2]))
        bands_out.append({'band': list(band), 'n_ok': len(ok_b),
                          'acc_ok': round(float(acc_ok.mean()), 4),
                          'acc_rand': round(float(acc_rand.mean()), 4),
                          'drop_ok': round(drop, 4), 'drop_rand': round(drop_r, 4),
                          'gap': [round(x, 4) for x in g],
                          'rand_reps': [round(float(x.mean()), 4) for x in rands]})

    print('=== 2 累积消融（自低向高，只消通过头）===')
    cum_out = []
    for band in BANDS:
        cum = [h for h in ok_all if h[0] <= band[1]]
        if not cum:
            continue
        acc_c = eval_cells(model, dev, cells, cum, meta)
        print('  <=L%-2d  n=%-3d  acc=%.4f  掉%.4f'
              % (band[1], len(cum), acc_c.mean(), acc_base - float(acc_c.mean())))
        cum_out.append({'upto': band[1], 'n': len(cum),
                        'acc': round(float(acc_c.mean()), 4),
                        'drop': round(acc_base - float(acc_c.mean()), 4)})

    print('=== 3 预注册判读 ===')
    sig = [b for b in bands_out if b['drop_ok'] >= 0.15 and b['gap'][0] >= 0.10 and b['gap'][1] > 0]
    any_big = [b for b in bands_out if b['drop_ok'] >= 0.15]
    if sig:
        best = max(sig, key=lambda b: b['drop_ok'])
        verdict = ('P1 承载段集中：L%d–L%d 单独消融掉 %.3f，且显著超同规模随机（差%+.3f）'
                   % (best['band'][0], best['band'][1], best['drop_ok'], best['gap'][0]))
    elif any_big:
        verdict = 'P1\' 有单带大掉但不超随机（该带效应可由消融规模解释，非角色跟随性）'
    elif cum_out and cum_out[-1]['drop'] >= 0.15:
        verdict = 'P2 承载段分散：单带均 <0.15，累积掉 %.3f' % cum_out[-1]['drop']
    else:
        verdict = 'P3 无可点名承载'
    print('  ' + verdict)

    out = {'n_trip': N_TRIP, 'n_cells': len(cells), 'base_acc': round(acc_base, 4),
           'reps': REPS, 'bands': bands_out, 'cumulative': cum_out,
           'verdict': verdict,
           'verdict_rule': 'P1: 单带 drop>=0.15 且 gap>=0.10 且 CI 下界>0；否则 P1\'/P2/P3'}
    p = provenance.write_result('pat_q2a2_bands', out, seed=SEED,
                                tool='tools/pat_q2a2_bands.py')
    print('落盘 %s' % p)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
