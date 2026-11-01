# -*- coding: utf-8 -*-
"""Q2-D 部分消融倒 U 曲线：把「部分比全消融更伤」从一次观测变成曲线。

预注册（见 PAT_Q2A_FINDINGS_20261101.md §八，跑前写死）
------------------------------------------------------
固定 4 层一带，层内削弱比例 p ∈ {0, 0.25, 0.5, 0.75, 1.0}，
每层随机消融 round(32p) 个头，每 p 抽 REPS 次取均值。

  预测 U：drop 在 p≈0.5 处取最大，p=1 处回落（倒 U）
  判据 U1：drop(0.5) > drop(0) 且 drop(0.5) > drop(1.0)，两差均 >= 0.05
  若单调增（p=1 最大）=> 倒 U 不成立，Q2-A3 的对比另有解释

两带同时扫（关键）：
  BAND_HOT  = L21–L24（Q2-A2 掉 0.222，观测到部分消融大损的一段）
  BAND_COLD = L29–L32（全消融 0.000，对齐最强但零因果的一段）
若冷带也呈倒 U => 「部分比全消融更伤」是任何层的普遍性质，与承载无关；
若只有热带倒 U => 它是承载段特有的结构性质。

用法
----
  SMOKE=1 .venv/Scripts/python.exe -X utf8 tools/pat_q2d_partial.py
  .venv/Scripts/python.exe -X utf8 tools/pat_q2d_partial.py
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
from pat_q2a_confound import eval_cells, paired, prepare

SEED = 20261105
SMOKE = os.environ.get('SMOKE', '') == '1'
N_TRIP = 4 if SMOKE else 24
REPS = 1 if SMOKE else 5
N_BOOT = 400 if SMOKE else 1000
PS = (0.0, 0.25, 0.5, 0.75, 1.0)
BAND_HOT = (21, 24)
BAND_COLD = (29, 32)


def sample_band(allh, band, p, rng):
    """每层随机消融 round(n_layer*p) 个头。"""
    out = []
    for l in range(band[0], band[1] + 1):
        pool = [h for h in allh if h[0] == l]
        k = int(round(len(pool) * p))
        if k <= 0:
            continue
        idx = rng.choice(len(pool), size=k, replace=False)
        out.extend(pool[i] for i in idx)
    return out


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
    print('评估格 n=%d  基线 %.4f  REPS=%d' % (len(cells), acc_base, REPS))

    rng = np.random.default_rng(SEED)
    out = {'n_trip': N_TRIP, 'n_cells': len(cells), 'base_acc': round(acc_base, 4),
           'reps': REPS, 'bands': {}}

    curves = {}
    for tag, band in (('hot', BAND_HOT), ('cold', BAND_COLD)):
        print('=== %s L%d–L%d ===' % (tag, band[0], band[1]))
        pts = []
        for p in PS:
            if p == 0.0:
                accs = [base]
                nh = 0
            else:
                accs = []
                nh = 0
                for _ in range(REPS):
                    sel = sample_band(allh, band, p, rng)
                    nh = len(sel)
                    accs.append(eval_cells(model, dev, cells, sel, meta))
            m = float(np.mean(accs, axis=0).mean())
            sd = float(np.std([float(a.mean()) for a in accs]))
            pts.append({'p': p, 'n_heads': nh, 'acc': round(m, 4),
                        'drop': round(acc_base - m, 4), 'sd': round(sd, 4)})
            print('  p=%.2f  n=%-4d  acc=%.4f  drop=%+.4f  (sd %.3f)'
                  % (p, nh, m, acc_base - m, sd))
        curves[tag] = pts

        drops = {pt['p']: pt['drop'] for pt in pts}
        mid = drops[0.5]
        u1 = (mid - drops[0.0] >= 0.05) and (mid - drops[1.0] >= 0.05)
        mono = all(pts[i]['drop'] <= pts[i + 1]['drop'] + 1e-9 for i in range(len(pts) - 1))
        peak = max(pts, key=lambda x: x['drop'])
        print('  峰值在 p=%.2f (drop %.4f)  倒U=%s  单调增=%s'
              % (peak['p'], peak['drop'], u1, mono))
        out['bands'][tag] = {'band': list(band), 'points': pts,
                             'peak_p': peak['p'], 'peak_drop': peak['drop'],
                             'U_shape': bool(u1), 'monotone': bool(mono)}

    print('=== 预注册判读 ===')
    hot_u = out['bands']['hot']['U_shape']
    cold_u = out['bands']['cold']['U_shape']
    if hot_u and cold_u:
        verdict = 'U0 倒 U 是任何层的普遍性质（两带皆有），与承载无关'
    elif hot_u and not cold_u:
        verdict = 'U1 倒 U 仅为承载段性质（热带呈倒U、冷带不呈）——部分消融破坏承载结构'
    elif not hot_u and out['bands']['hot']['monotone']:
        verdict = 'U2 单调增：部分消融不比全消融更伤，Q2-A3 的对比需另找解释'
    else:
        verdict = 'U3 非倒 U 非单调：曲线形状不稳定，须加大 REPS 或换带'
    print('  ' + verdict)
    out['verdict'] = verdict

    p = provenance.write_result('pat_q2d_partial', out, seed=SEED,
                                tool='tools/pat_q2d_partial.py')
    print('落盘 %s' % p)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
