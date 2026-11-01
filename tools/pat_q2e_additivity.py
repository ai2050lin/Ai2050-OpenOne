# -*- coding: utf-8 -*-
"""Q2-E 承载段内部结构：双峰是同一通路的两段，还是两条独立通路。

前置事实（Q2-D2）
----------------
整带全消 128 头（11.1%），L0–L3 掉 0.674、L16–L19 掉 0.660，L28–L35 精确 0.000。
承载剖面是宽带双峰。两个峰是不是共用一条残差通路，决定后面该按「一条路」还是
「两条路」建机制模型。

为什么不能只在整带密度上测可加
------------------------------
单带已掉 0.674，基线 0.972，两带叠加的理论上限只到 0.972-0.674=0.298 的余量。
若两路独立可加，预测值 1.334 会撞地板 0.972，与「部分共用」不可分。
**所以主检验放在半密度（每带 64 头，层内 50%）上做，整带密度只作上限核对。**

预注册设计（9 臂，共用同一批 144 格）
------------------------------------
  base      无干预
  A64       L0–L3 每层随机 16 头，共 64 头（层内 50%）
  B64       L16–L19 每层随机 16 头，共 64 头（层内 50%）
  AB128     A64 ∪ B64，共 128 头（层内各 50%）—— 与 A128/B128 密度严格配对
  A128      L0–L3 全消 128 头（层内 100%）
  B128      L16–L19 全消 128 头
  AB256     A128 ∪ B128，256 头（22.2%）
  R128      全局随机 128 头（密度对照）
  R256      全局随机 256 头（密度对照）

两个可加性缺口（都用「相对更强的单带」计，保守）：
  gap_half = drop(AB128) - max(drop(A64), drop(B64))
  gap_full = drop(AB256) - max(drop(A128), drop(B128))

事先写死的预测
--------------
P1 单通路饱和（我的押注）：gap_half ≈ 0，gap_full ≈ 0。
   两带各自削弱的是同一条残差通路的不同段，各削 50% 合计仍是 50%，不叠加。
P2 两路独立可加：gap_half ≈ min(drop(A64), drop(B64))（量级 ~0.3），
   gap_full 撞地板（drop(AB256) >= 0.85）。
P3 部分共用：两项介于两者之间且同向。

判读（互斥，跑完不许改阈值）
--------------------------
E1 单通路：gap_half <= 0.08 且 gap_full <= 0.10
E2 双通路可加：gap_half >= 0.20 且 drop(AB256) >= 0.85
E3 部分共用：其余，且两项同为正
E4 反常（拮抗）：drop(AB128) < max(drop(A64), drop(B64)) - 0.05，即联合比单带更轻
E5 无效：drop(A64)、drop(B64) 均 < 0.05（半密度打不动，主检验无余量，须改设计）

同轮附带（不单独判，作描述统计）
------------------------------
段内线性度 r = drop(A128) / max(drop(A64), 1e-6)。
  r≈2 说明段内损伤随覆盖线性增长（均匀承载）；r≈1 说明前 50% 已打满
  （少数头承担多数）。r>2.5 或 r<1 记异常并查补偿。

用法
----
  SMOKE=1 .venv/Scripts/python.exe -X utf8 tools/pat_q2e_additivity.py
  .venv/Scripts/python.exe -X utf8 tools/pat_q2e_additivity.py
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
from pat_q2a_confound import eval_cells, paired, prepare

SEED = 20261107
SMOKE = os.environ.get('SMOKE', '') == '1'
N_TRIP = 4 if SMOKE else 24
REPS = 1 if SMOKE else 3
N_BOOT = 400 if SMOKE else 1000
BAND_A = (0, 3)
BAND_B = (16, 19)


def band_heads(band, n_layers, n_heads):
    return [(l, h) for l in range(band[0], band[1] + 1) for h in range(n_heads)]


def sample_per_layer(rng, band, per_layer, n_heads):
    """每层随机抽 per_layer 头，保持层平衡（避免又落到层号排序混杂）。"""
    out = []
    for l in range(band[0], band[1] + 1):
        hs = rng.choice(n_heads, size=per_layer, replace=False)
        out.extend((l, int(h)) for h in hs)
    return out


def cov(sel, n_layers, n_heads):
    """D1 三元口径里的层内覆盖率。"""
    tot = len(sel)
    return tot, tot / float(n_layers * n_heads), tot / float(len(set(l for l, _ in sel)) * n_heads)


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
    allh = band_heads((0, n_layers - 1), n_layers, n_heads)

    cells = prepare(tok, triples(N_TRIP))
    base = eval_cells(model, dev, cells, [], meta)
    acc_base = float(base.mean())
    print('评估格 n=%d  基线 %.4f  层=%d 头/层=%d' % (len(cells), acc_base, n_layers, n_heads))

    det_arms = [
        ('A128', band_heads(BAND_A, n_layers, n_heads)),
        ('B128', band_heads(BAND_B, n_layers, n_heads)),
        ('AB256', band_heads(BAND_A, n_layers, n_heads) + band_heads(BAND_B, n_layers, n_heads)),
    ]
    res = {}
    vec = {}
    for name, sel in det_arms:
        a = eval_cells(model, dev, cells, sel, meta)
        vec[name] = a
        res[name] = float(a.mean())
        print('  %-6s n=%-3d acc=%.4f drop=%+.4f' % (name, len(sel), res[name], acc_base - res[name]))

    rng = np.random.default_rng(SEED)
    rnd_names = ('A64', 'B64', 'AB128', 'R128', 'R256')
    accs = {k: [] for k in rnd_names}
    for rep in range(REPS):
        a64 = sample_per_layer(rng, BAND_A, 16, n_heads)
        b64 = sample_per_layer(rng, BAND_B, 16, n_heads)
        arms = {
            'A64': a64,
            'B64': b64,
            'AB128': a64 + b64,
            'R128': [allh[i] for i in rng.choice(len(allh), size=128, replace=False)],
            'R256': [allh[i] for i in rng.choice(len(allh), size=256, replace=False)],
        }
        line = []
        for k in rnd_names:
            a = eval_cells(model, dev, cells, arms[k], meta)
            accs[k].append(float(a.mean()))
            if rep == 0:
                vec[k] = a
            line.append('%s=%.4f' % (k, float(a.mean())))
        print('  rep%d  %s' % (rep, '  '.join(line)))

    for k in rnd_names:
        res[k] = float(np.mean(accs[k]))
    drop = {k: acc_base - v for k, v in res.items()}

    print('\n=== 消融口径（D1 三元）===')
    for k in ('A64', 'B64', 'AB128', 'A128', 'B128', 'AB256', 'R128', 'R256'):
        sel = (band_heads(BAND_A, n_layers, n_heads) + band_heads(BAND_B, n_layers, n_heads)) \
            if k == 'AB256' else band_heads(BAND_A, n_layers, n_heads) if k == 'A128' \
            else band_heads(BAND_B, n_layers, n_heads) if k == 'B128' else None
        if sel is None:
            print('  %-6s acc=%.4f drop=%+.4f  (随机臂，密度见设计)' % (k, res[k], drop[k]))
        else:
            t, fr, lc = cov(sel, n_layers, n_heads)
            print('  %-6s acc=%.4f drop=%+.4f  头=%d 占比=%.1f%% 层内=%.0f%%'
                  % (k, res[k], drop[k], t, fr * 100, lc * 100))

    rng2 = np.random.default_rng(SEED + 1)
    # 主检验用「AB128 vs 更强的单带」的配对差
    stronger = 'A64' if drop['A64'] >= drop['B64'] else 'B64'
    d1, lo1, hi1, n01, n10 = paired(vec[stronger], vec['AB128'], rng2, N_BOOT)
    gap_half = drop['AB128'] - max(drop['A64'], drop['B64'])
    print('\n=== 主检验（半密度，有 headroom）===')
    print('  drop: A64=%.4f B64=%.4f AB128=%.4f  更强单带=%s' % (drop['A64'], drop['B64'], drop['AB128'], stronger))
    print('  gap_half = %.4f   (AB128 - %s 配对差 %.4f, 95%%CI [%.4f, %.4f], McNemar %d/%d)'
          % (gap_half, stronger, d1, lo1, hi1, n01, n10))

    stronger_f = 'A128' if drop['A128'] >= drop['B128'] else 'B128'
    d2, lo2, hi2, m01, m10 = paired(vec[stronger_f], vec['AB256'], rng2, N_BOOT)
    gap_full = drop['AB256'] - max(drop['A128'], drop['B128'])
    print('  gap_full = %.4f   (AB256 - %s 配对差 %.4f, 95%%CI [%.4f, %.4f], McNemar %d/%d)'
          % (gap_full, stronger_f, d2, lo2, hi2, m01, m10))

    print('\n=== 附带：段内线性度 ===')
    r_a = drop['A128'] / max(drop['A64'], 1e-6) if drop['A64'] > 0.02 else None
    r_b = drop['B128'] / max(drop['B64'], 1e-6) if drop['B64'] > 0.02 else None
    print('  r_A=%s  r_B=%s  (≈2 均匀承载 / ≈1 少数头打满；分母 <0.02 记 n.a.)'
          % ('%.2f' % r_a if r_a else 'n.a.', '%.2f' % r_b if r_b else 'n.a.'))

    # 失败集重合度：这才是「同一条通路」的直接证据，比正确率相等更硬
    def fails(name):
        v = vec[name]
        return set(np.flatnonzero(v == 0).tolist())

    fa, fb, fab = fails('A128'), fails('B128'), fails('AB256')
    uni = sorted(fa | fb)
    print('\n=== 失败集结构（直接检验共用通路）===')
    print('  |fail_A|=%d  |fail_B|=%d  |fail_AB|=%d  |A∪B|=%d  |A∩B|=%d'
          % (len(fa), len(fb), len(fab), len(uni), len(fa & fb)))
    jac = len(fa & fb) / float(len(fa | fb)) if (fa | fb) else 0.0
    only = len(fab - (fa | fb))
    print('  Jaccard(A,B)=%.3f   仅AB新增失败=%d   AB 是否等于 A∪B: %s'
          % (jac, only, '是' if fab == (fa | fb) else '否'))

    print('\n=== 密度对照 ===')
    print('  R128 drop=%+.4f  R256 drop=%+.4f  (AB256 应显著高于 R256)' % (drop['R128'], drop['R256']))

    if drop['A64'] < 0.05 and drop['B64'] < 0.05:
        verdict = 'E5 无效：半密度打不动（A64=%.3f B64=%.3f），主检验无余量，须改设计' % (drop['A64'], drop['B64'])
    elif gap_half <= 0.08 and gap_full <= 0.10:
        verdict = 'E1 单通路：两带削的是同一条残差通路的不同段（gap_half=%.3f, gap_full=%.3f）' % (gap_half, gap_full)
    elif gap_half >= 0.20 and drop['AB256'] >= 0.85:
        verdict = 'E2 双通路可加：两带独立，各自足以维持行为（gap_half=%.3f, drop_AB256=%.3f）' % (gap_half, drop['AB256'])
    elif drop['AB128'] < max(drop['A64'], drop['B64']) - 0.05:
        verdict = 'E4 反常拮抗：联合消融比单带更轻（%.3f < %.3f），须查抑制性补偿' % (drop['AB128'], max(drop['A64'], drop['B64']))
    else:
        verdict = 'E3 部分共用：gap_half=%.3f, gap_full=%.3f，两路共享部分通路' % (gap_half, gap_full)
    print('\n=== 判读 ===')
    print('  ' + verdict)

    p = provenance.write_result(
        'pat_q2e_additivity',
        {'n_trip': N_TRIP, 'n_cells': len(cells), 'base_acc': round(acc_base, 4), 'reps': REPS,
         'res': {k: round(v, 4) for k, v in res.items()},
         'drop': {k: round(v, 4) for k, v in drop.items()},
         'gap_half': round(gap_half, 4), 'gap_full': round(gap_full, 4),
         'ci_gap_half': [round(lo1, 4), round(hi1, 4)], 'ci_gap_full': [round(lo2, 4), round(hi2, 4)],
         'mcnemar_half': [n01, n10], 'mcnemar_full': [m01, m10],
         'linearity': [None if r_a is None else round(r_a, 3), None if r_b is None else round(r_b, 3)],
         'fail_sets': {'A': len(fa), 'B': len(fb), 'AB': len(fab),
                       'union': len(uni), 'inter': len(fa & fb),
                       'jaccard': round(jac, 4), 'ab_only_new': only,
                       'ab_equals_union': bool(fab == (fa | fb))},
         'stronger': [stronger, stronger_f],
         'verdict': verdict},
        seed=SEED, tool='tools/pat_q2e_additivity.py')
    print('落盘 %s' % p)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
