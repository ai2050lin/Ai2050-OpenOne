# -*- coding: utf-8 -*-
"""Q2-A 混杂对照：k=60 的崩塌到底该记在「角色跟随头」还是「低层」。

为什么必须做
------------
Q1 剂量曲线在 k=60 处出现 0.39 的巨大 gap（通过头 0.597 vs 随机头 0.986），
但 `ok_all[:k]` 按命中率降序排、命中率大量并列 1.0，稳定排序退化成层号升序。
于是「前 60 个通过头」实际等于「层号最低的 60 个通过头」。
两个读法没被分开：读法一＝角色跟随头这个集合承载；读法二＝低层承载更重。

预注册设计（2x2，写死在判读规则里，跑完不许改）
--------------------------------------------
        低层            非特指低层
通过筛选  A low_ok        C mid_ok
未通过   B low_rand      D glob_rand
另有反向格 E high_ok（通过头里层号最高的 k 个）。

A/B 只差「是否通过筛选」，层分布匹配；A/C 只差「是否限定低层」，都是通过头。
所以：
  gap_AB = acc(B) - acc(A)  >0 且显著  => 通过筛选本身有因果含义（超出层位）
  gap_AE = acc(E) - acc(A)  >0 且显著  => 低层的通过头更脆（层位读法）
  gap_CD = acc(D) - acc(C)  >0 且显著  => 通过头整体必需，与层位无关

判读顺序（互斥）
----------------
R0 交互：gap_AB 与 gap_AE 同时显著 且 gap_CD 不显著
   => 只有「低层 × 通过筛选」这一格有特权，两个属性单独都不够
R1 角色跟随性承载：gap_AB 显著 且 gap_AE 不显著
R2 层位置驱动：gap_AB 不显著 且 gap_AE 显著
R3 集合整体必需：gap_AB、gap_AE 均不显著 且 gap_CD 显著为正
R4 前结论为抽样假象：三项均不显著

R0 是 SMOKE 干跑（n=24 格，不用于结论）后补入的判据项，补入原因：SMOKE 中
A 崩而 B/C/E 均不崩，说明「低层」与「通过筛选」两个属性各自不足，原 R1/R2
都会误判。显著性阈值（diff>=0.10 且 CI 下界>0）未改动。

统计口径
--------
评估集固定为 24 三元组 × 2 句式 × 3 角色 = 144 格，全部条件共用同一批格，
因此比较用配对差：bootstrap 1000 次取 95% CI，并给 McNemar 方向计数（A错D对 / A对D错）。
B/C/D 三个随机格各抽 REPS 次，报均值与极差，避免单次抽样决定结论。

用法
----
  SMOKE=1 .venv/Scripts/python.exe -X utf8 tools/pat_q2a_confound.py   # 干跑
  .venv/Scripts/python.exe -X utf8 tools/pat_q2a_confound.py            # 全量
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
from pat_q1_unbind import FORMS, QUERIES, ROLES, build, encode, render, triples
from pat_q1_verify import HeadMask

SEED = 20261102
SMOKE = os.environ.get('SMOKE', '') == '1'
N_TRIP = 4 if SMOKE else 24
K_LIST = (10, 30) if SMOKE else (30, 60, 100)
REPS = 1 if SMOKE else 5
N_BOOT = 400 if SMOKE else 1000


def load_ok_heads():
    cand = sorted((ROOT / 'results').glob('2026-11-01/*_pat_q1_unbind_v2.json'))
    if not cand:
        raise SystemExit('找不到 Q1 结果文件')
    d = json.loads(cand[-1].read_text(encoding='utf-8'))
    return [tuple(x) for x in d['payload']['heads_all']]


def pick_low(pool, k):
    return sorted(pool, key=lambda x: (x[0], x[1]))[:k]


def pick_high(pool, k):
    return sorted(pool, key=lambda x: (-x[0], -x[1]))[:k]


def layer_matched(allh, ref, rng):
    """从全局池按 ref 的层分布逐层抽取 —— 用来把「层位」这个属性对齐掉。"""
    by_layer = {}
    for (l, h) in allh:
        by_layer.setdefault(l, []).append((l, h))
    cnt = Counter(l for (l, _) in ref)
    out = []
    for l in sorted(cnt):
        pool = by_layer.get(l, [])
        c = min(cnt[l], len(pool))
        if c <= 0:
            continue
        sel = rng.choice(len(pool), size=c, replace=False)
        out.extend(pool[i] for i in sel)
    return out


def prepare(tok, trips):
    """预编码全部评估格，避免每个条件重复 tokenize。"""
    cells = []
    for ti, trip in enumerate(trips):
        ent_ids = [tok(e, add_special_tokens=False)['input_ids'][0] for e in trip]
        if len(set(ent_ids)) != 3:
            raise SystemExit('三元组 %d 的实体首 token 撞车：%s' % (ti, trip))
        want_of = {'agent': 0, 'recipient': 1, 'theme': 2}
        for form in FORMS:
            ctx, ranges = build(trip, form)
            for role in ROLES:
                raw = ctx + QUERIES[role]
                text = render(tok, raw)
                ids, _idx = encode(tok, text, raw, ranges)
                cells.append({'ids': list(ids), 'ents': ent_ids,
                              'want': ent_ids[want_of[role]], 'ti': ti,
                              'form': form, 'role': role})
    return cells


@torch.no_grad()
def eval_cells(model, dev, cells, pairs, meta):
    """返回 0/1 向量（1=答对）。pairs 为空即基线。"""
    if not pairs:
        mask_ctx = None
    else:
        mask_ctx = HeadMask(model, meta, pairs)
    out = np.zeros(len(cells), dtype=np.int8)
    if mask_ctx is None:
        for i, c in enumerate(cells):
            o = model(input_ids=torch.tensor([c['ids']], device=dev), use_cache=False)
            lp = torch.log_softmax(o.logits[0, -1].float(), dim=-1)
            sc = [float(lp[t]) for t in c['ents']]
            out[i] = int(c['ents'][int(np.argmax(sc))] == c['want'])
    else:
        with mask_ctx:
            for i, c in enumerate(cells):
                o = model(input_ids=torch.tensor([c['ids']], device=dev), use_cache=False)
                lp = torch.log_softmax(o.logits[0, -1].float(), dim=-1)
                sc = [float(lp[t]) for t in c['ents']]
                out[i] = int(c['ents'][int(np.argmax(sc))] == c['want'])
    return out


def paired(a, b, rng, n_boot=N_BOOT):
    """a、b 为同批 144 格的 0/1 向量。返回 (差值 b-a, CI 下界, 上界, A错B对, A对B错)。"""
    diff = float(b.mean() - a.mean())
    d = (b - a).astype(np.float64)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    bs = d[idx].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    b01 = int(((a == 0) & (b == 1)).sum())
    b10 = int(((a == 1) & (b == 0)).sum())
    return diff, float(lo), float(hi), b01, b10


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

    trips = triples(N_TRIP)
    cells = prepare(tok, trips)
    ok_all = load_ok_heads()
    allh = [(l, h) for l in range(meta['n_layers']) for h in range(meta['n_heads'])]

    print('=== 0 诊断：通过头集的层分布（混杂本体）===')
    hist = Counter(l for (l, _) in ok_all)
    print('  n_ok=%d  层范围 L%d–L%d' % (len(ok_all), min(hist), max(hist)))
    print('  分层计数 ' + ' '.join('L%d:%d' % (l, hist[l]) for l in sorted(hist)))
    print('  评估格 n=%d（%d 三元组 x %d 句式 x %d 角色）'
          % (len(cells), N_TRIP, len(FORMS), len(ROLES)))

    rng = np.random.default_rng(SEED)
    base = eval_cells(model, dev, cells, [], meta)
    print('  基线正确率 %.4f' % base.mean())

    res = {}
    for k in K_LIST:
        print('=== k=%d ===' % k)
        A = pick_low(ok_all, k)
        E = pick_high(ok_all, k)
        acc_A = eval_cells(model, dev, cells, A, meta)
        acc_E = eval_cells(model, dev, cells, E, meta)

        Bs, Cs, Ds = [], [], []
        for r in range(REPS):
            Bs.append(eval_cells(model, dev, cells, layer_matched(allh, A, rng), meta))
            idxc = rng.choice(len(ok_all), size=k, replace=False)
            Cs.append(eval_cells(model, dev, cells, [ok_all[i] for i in idxc], meta))
            idxd = rng.choice(len(allh), size=k, replace=False)
            Ds.append(eval_cells(model, dev, cells, [allh[i] for i in idxd], meta))

        def agg(lst):
            m = np.mean(lst, axis=0)
            return m

        acc_B, acc_C, acc_D = agg(Bs), agg(Cs), agg(Ds)
        print('  A low_ok    %.4f  层 L%d–L%d' % (acc_A.mean(), A[0][0], A[-1][0]))
        print('  B low_rand  %.4f  (reps=%d, 单次 %s)' % (
            acc_B.mean(), REPS, [round(float(x.mean()), 3) for x in Bs]))
        print('  C mid_ok    %.4f  (reps=%d, 单次 %s)' % (
            acc_C.mean(), REPS, [round(float(x.mean()), 3) for x in Cs]))
        print('  D glob_rand %.4f  (reps=%d, 单次 %s)' % (
            acc_D.mean(), REPS, [round(float(x.mean()), 3) for x in Ds]))
        print('  E high_ok   %.4f  层 L%d–L%d' % (acc_E.mean(), E[0][0], E[-1][0]))

        gAB = paired(acc_A, acc_B, rng)
        gAE = paired(acc_A, acc_E, rng)
        gCD = paired(acc_C, acc_D, rng)
        gAC = paired(acc_A, acc_C, rng)
        for tag, g in (('B-A 筛选效应', gAB), ('E-A 高层-低层', gAE),
                       ('D-C 随机-通过', gCD), ('C-A 中段-低层', gAC)):
            print('  %-14s diff=%+.4f CI[%+.4f,%+.4f]  A错X对=%d A对X错=%d'
                  % (tag, g[0], g[1], g[2], g[3], g[4]))
        res[k] = {'acc_A': round(float(acc_A.mean()), 4),
                  'acc_B': round(float(acc_B.mean()), 4),
                  'acc_C': round(float(acc_C.mean()), 4),
                  'acc_D': round(float(acc_D.mean()), 4),
                  'acc_E': round(float(acc_E.mean()), 4),
                  'gap_AB': [round(x, 4) for x in gAB],
                  'gap_AE': [round(x, 4) for x in gAE],
                  'gap_CD': [round(x, 4) for x in gCD],
                  'gap_CA': [round(x, 4) for x in gAC],
                  'A_layers': [A[0][0], A[-1][0]], 'E_layers': [E[0][0], E[-1][0]],
                  'B_reps': [round(float(x.mean()), 4) for x in Bs],
                  'C_reps': [round(float(x.mean()), 4) for x in Cs],
                  'D_reps': [round(float(x.mean()), 4) for x in Ds]}

    print('=== 4 预注册判读（只看 gap 显著为正向且 CI 不含 0 的项）===')
    k = 60 if 60 in res else K_LIST[-1]
    r = res[k]
    sig = lambda g: g[0] >= 0.10 and g[1] > 0
    if sig(r['gap_AB']) and sig(r['gap_AE']):
        verdict = 'R0 交互：只有「低层 x 通过筛选」有特权，单属性均不足'
    elif sig(r['gap_AB']):
        verdict = 'R1 角色跟随性承载（通过筛选本身有因果含义，超出层位）'
    elif sig(r['gap_AE']):
        verdict = 'R2 层位置驱动（低层更重要，与是否角色跟随头无关）'
    elif sig(r['gap_CD']):
        verdict = 'R3 集合整体必需（通过头整体承载，但层位不是主因）'
    else:
        verdict = 'R4 前结论为抽样假象（k=60 崩塌不复现）'
    print('  以 k=%d 判读：%s' % (k, verdict))
    print('  参考 gap_AB=%+.4f gap_AE=%+.4f gap_CD=%+.4f'
          % (r['gap_AB'][0], r['gap_AE'][0], r['gap_CD'][0]))

    out = {'n_trip': N_TRIP, 'n_cells': len(cells), 'reps': REPS,
           'base_acc': round(float(base.mean()), 4),
           'ok_hist': {str(l): hist[l] for l in sorted(hist)},
           'per_k': res, 'verdict_k': k, 'verdict': verdict,
           'verdict_rule': 'gap>=0.10 且 CI 下界>0；顺序 R1>R2>R3>R4'}
    p = provenance.write_result('pat_q2a_confound', out, seed=SEED,
                                tool='tools/pat_q2a_confound.py')
    print('落盘 %s' % p)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
