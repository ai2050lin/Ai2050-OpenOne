# -*- coding: utf-8 -*-
"""Q2-H 承载段内部结构：L0–L3 里到底有没有"一小批核心头"。

起因
----
Q2-F 已测出：L0–L3 内每层随机抽 16 头（共 64 头，层内 50%），12 次独立抽样
正确率从 0.3403 到 1.0000，极差 0.66，双峰。同样的头数、同样的层、同样的
随机方式，结果能从"完全无损"到"彻底失效"。

这说明承载在段内极不均匀，但没回答"不均匀到什么程度"：
  (a) 段内有一小批核心头，抽中即崩、抽不中则被其余层完全补偿 —— 可点名
  (b) 不存在固定核心，崩溃取决于"抽中的头凑成的总量是否越过阈值" —— 不可点名

这个二分决定后续路线：走"找单元"（继续筛头）还是"测通路"（路径修补）。

预注册判据的逻辑更正（执行前，非事后）
--------------------------------------
原预注册写的是：「极差随子集变小而收窄 ⇒ 存在稳定核心；极差始终不收敛 ⇒ 分布式临界」。
该判据无分辨力，理由：极差 R(k) 在 k=全部头时必然为 0，只要 k 增大就单调收窄，
与"有无核心"无关。若真有 c 个核心头，k 越小越容易漏掉核心，R(k) 反而**更大**，
方向与原文相反。

改为以下三条可判据（互斥，跑完不许改阈值）：
  N1 存在稳定核心：存在某个 k_per 使 12/12 次全部触底（acc<=0.35）
     ⇒ 核心子集 <= 4*k_per 头，可继续用二分法点名。
  N2 阈值型（临界）：无任何 k_per 达 12/12 触底，但存在 k 使 12 次里同时出现
     acc>=0.999 与 acc<=0.35（两极并存），且触底比例随 k 单调上升
     ⇒ 无固定核心，损伤由"凑够的总量"驱动，必须改测通路。
  N3 分布式（可加）：任何 k 都不出现两极，且各 k 的极差 < 0.15
     ⇒ 头近似等价，损伤只依赖数量。
  N4 判据失效：k_per=16（与 Q2-F 同口径的内建复现检查）复现不出极差 >= 0.40
     ⇒ 抽样或评估口径不稳，本轮全部结论作废。

设计
----
- 带：L0–L3（4 层），每层抽 k_per 头，层平衡（避免层号排序混杂，D2）
- k_per ∈ {1, 2, 4, 8, 12, 16}，对应总头数 {4, 8, 16, 32, 48, 64}
- 每个 k_per 独立抽样 12 次（D5：跨抽样取均值前先看分布，故报全分布）
- k_per=16 即 Q2-F 的 A64 臂，用作口径稳定性内建对照
- 每次评估 144 格（24 三元组 x 主动/被动 x 3 查询角色），配对基线
- **完整保存每次抽样的头列表**（Q2-F 没存，导致无法事后定位，本轮补上）

附带离线分析（不额外跑模型）
----------------------------
- 逐 k_per 的头频次富集：出现在"触底抽样"里的次数 / 出现在全部抽样里的次数
  若存在核心，少数头的富集比会显著 > 1（可据此给候选核心排序）
- 触底抽样的交集大小：若多个触底抽样共享同一小批头，交集即核心候选

用法
----
  SMOKE=1 .venv/Scripts/python.exe -X utf8 tools/pat_q2h_core.py
  .venv/Scripts/python.exe -X utf8 tools/pat_q2h_core.py
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
from pat_q2e_additivity import sample_per_layer

SEED = 20261109
SMOKE = os.environ.get('SMOKE', '') == '1'
N_TRIP = 4 if SMOKE else 24
REPS = 3 if SMOKE else 12
BAND = (0, 3)
K_PER = (1, 2, 4, 8, 12, 16)
FLOOR = 0.35
PERFECT = 0.999


def stats(x):
    a = np.asarray(x, dtype=np.float64)
    return {'mean': round(float(a.mean()), 4), 'median': round(float(np.median(a)), 4),
            'sd': round(float(a.std(ddof=1)) if len(a) > 1 else 0.0, 4),
            'min': round(float(a.min()), 4), 'max': round(float(a.max()), 4),
            'range': round(float(a.max() - a.min()), 4)}


def verdict(rows, base_acc):
    """rows: 每 k_per 一条，含 stats / n_perfect / n_floor。"""
    if not rows:
        return 'N4 判据失效：无有效臂'
    ref = [r for r in rows if r['k_per'] == 16]
    if ref:
        r16 = ref[0]
        if r16['stats']['range'] < 0.40 and not SMOKE:
            return ('N4 判据失效：k_per=16 复现极差 %.2f (<0.40)，'
                    '与 Q2-F 的 0.66 不符，抽样或评估口径不稳' % r16['stats']['range'])
        if r16['stats']['max'] < base_acc * 0.95 and not SMOKE:
            return ('N4 判据失效：k_per=16 最大损伤 %.2f，未出现无损抽样，'
                    '与 Q2-F 的双峰不符' % (base_acc - r16['stats']['max']))

    any_all_floor = [r for r in rows if r['n_floor'] == REPS]
    if any_all_floor:
        k = min(r['k_per'] for r in any_all_floor)
        return ('N1 存在稳定核心：k_per=%d（共 %d 头）时 %d/%d 全部触底 '
                '=> 核心子集 <= %d 头，可二分点名'
                % (k, 4 * k, REPS, REPS, 4 * k))

    bipolar = [r for r in rows if r['n_perfect'] > 0 and r['n_floor'] > 0]
    if bipolar:
        ks = ','.join(str(r['k_per']) for r in bipolar)
        return ('N2 阈值型（临界）：k_per∈{%s} 出现无损与触底并存，'
                '无任何 k 全部触底 => 无固定核心，须改测通路' % ks)

    big = [r for r in rows if r['stats']['range'] >= 0.15]
    if big:
        return ('N2 阈值型（临界，弱形）：极差 >=0.15 但未检出两极并存，'
                'k_per∈{%s} => 倾向分布式临界' % ','.join(str(r['k_per']) for r in big))
    return 'N3 分布式（可加）：各 k 极差 <0.15，头近似等价，损伤只依赖数量'


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
    n_heads = meta['n_heads']
    total = (BAND[1] - BAND[0] + 1) * n_heads

    cells = prepare(tok, triples(N_TRIP))
    acc_base = float(eval_cells(model, dev, cells, [], meta).mean())
    print('评估格 n=%d  基线 %.4f  段 L%d–L%d 共 %d 头  每档抽样 %d 次'
          % (len(cells), acc_base, BAND[0], BAND[1], total, REPS))
    print('k_per 档位 %s  => 总头数 %s' % (list(K_PER), [4 * k for k in K_PER]))

    rng = np.random.default_rng(SEED)
    rows, all_draws = [], {}
    for kp in K_PER:
        accs, draws, drops = [], [], []
        for i in range(REPS):
            sel = sample_per_layer(rng, BAND, kp, n_heads)
            acc = float(eval_cells(model, dev, cells, sel, meta).mean())
            accs.append(acc)
            drops.append(acc_base - acc)
            draws.append(sorted('%d-%d' % (l, h) for l, h in sel))
        s = stats(accs)
        n_perfect = int(sum(1 for a in accs if a >= PERFECT))
        n_floor = int(sum(1 for a in accs if a <= FLOOR))
        rows.append({'k_per': kp, 'n_heads': 4 * kp,
                     'coverage': round(4 * kp / float(total), 4),
                     'stats': s, 'n_perfect': n_perfect, 'n_floor': n_floor,
                     'accs': [round(a, 4) for a in accs],
                     'drops': [round(d, 4) for d in drops]})
        all_draws[str(kp)] = draws
        print('  k_per=%-2d (%2d 头, %5.1f%%)  mean=%.4f  min=%.4f  max=%.4f  '
              'range=%.4f  无损%d  触底%d'
              % (kp, 4 * kp, 100.0 * 4 * kp / total, s['mean'], s['min'], s['max'],
                 s['range'], n_perfect, n_floor))

    v = verdict(rows, acc_base)
    print('\n=== 判读 ===')
    print('  ' + v)

    print('\n=== 头频次富集（触底抽样 vs 全部抽样）===')
    enr_rows = []
    for kp in K_PER:
        draws = all_draws[str(kp)]
        r = [x for x in rows if x['k_per'] == kp][0]
        bad_idx = [i for i, a in enumerate(r['accs']) if a <= FLOOR]
        if not bad_idx or len(bad_idx) == len(draws):
            continue
        from collections import Counter
        c_all, c_bad = Counter(), Counter()
        for i, d in enumerate(draws):
            c_all.update(d)
            if i in bad_idx:
                c_bad.update(d)
        exp = float(len(bad_idx)) / len(draws)
        scored = []
        for hid, ca in c_all.items():
            cb = c_bad.get(hid, 0)
            scored.append((cb / float(ca) / exp, hid, cb, ca))
        scored.sort(reverse=True)
        top = scored[:6]
        print('  k_per=%-2d  触底 %d/%d 次  期望命中率 %.2f' % (kp, len(bad_idx), REPS, exp))
        print('           top: ' + '  '.join('%s(%.2fx %d/%d)' % (h, e, cb, ca)
                                             for e, h, cb, ca in top))
        for e, h, cb, ca in top:
            enr_rows.append({'k_per': kp, 'head': h, 'enrich': round(e, 3),
                             'in_bad': cb, 'in_all': ca})

    p = provenance.write_result(
        'pat_q2h_core',
        {'n_trip': N_TRIP, 'n_cells': len(cells), 'base_acc': round(acc_base, 4),
         'reps': REPS, 'band': list(BAND), 'k_per': list(K_PER), 'rows': rows,
         'enrich_top': enr_rows, 'verdict': v},
        seed=SEED, tool='tools/pat_q2h_core.py')
    print('\n落盘 %s' % p)
    print('用时 %.1f 分钟' % ((time.time() - t0) / 60.0))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
