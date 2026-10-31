# -*- coding: utf-8 -*-
"""H13 承载核确认：3 个单元能不能顶替 tailN 的 23 个。

H12（`tools/mech_carrier.py`）在 12 道脆弱题 + 6 道对照题上自适应二分，得到：

  tailN(23, Σ|Δ|=9.43)      打坏 10/17（脆弱 10/12）
    ├ A(11) [83,149,383,583,2423,3311,1944,2600,767,1540,35]   打坏 10
    │   ├ A'(5) [83,149,383,583,2423]                           打坏 2
    │   └ B'(6) [3311,1944,2600,767,1540,35]                    打坏 7
    │       ├ [3311,1944,2600]                                   打坏 2
    │       └ core3 [767,1540,35]                                打坏 7
    └ B(12) [其余 12 个]                                        打坏 1

也就是说 23 个里 3 个（占 Σ|Δ| 的 12.7%）就复现了 7 道，而 Σ|Δ| 是它 2.9 倍的
A'(5) 只打坏 2 道。**这直接否掉「|Δ| 越大越重要」，并给出第一个可破解的编码对象。**

但 H12 的分母只有 17 道题、且判据是自适应贪心的「≥ 半量」阈值，所以必须换到
全量 169 题上重测，才能说「3 个顶 23 个」是不是全量事实而不是小题集巧合。

设计
----
4 个条件 × 169 题，全部 ×0（down_proj 输入置零，与 H5/H6/H8/H10/H12 同一个 hook）：

  base    不干预（不用复用，本轮自带，避免跨轮 base 漂移）
  core3   [767, 1540, 35]             Σ|Δ| 1.196（tailN 的 12.7%）
  core5   [83, 149, 383, 583, 2423]   Σ|Δ| 2.626（tailN 的 27.8%，是 core3 的 2.2×）
  tailN23 H12 的起始集，全量重测以拿到真分母（110 道 base 对的题，不是 17 道）

预注册判据
----------
C1 可复制    tailN23 在全量 169 题上的打坏题集**包含** core3 的打坏题集（分歧 ≤ 1 题）
             否则 H12 的 17 题结论不可外推，整轮降级为「小题集现象」
C2 充分性    |damage(core3)| ≥ 0.6 × |damage(tailN23)|
C3 份额悖论  Σ|Δ|(core3)/Σ|Δ|(tailN23) ≤ 0.20 且 |damage(core3)|/|damage(tailN23)| ≥ 0.60
             两条同时成立 ⇒ 「|Δ| 排序 = 重要性」在全量上被彻底否证（R-MLP-18 强化）
C4 非单调    |damage(core5)| < |damage(core3)|，尽管 Σ|Δ|(core5) = 2.2 × Σ|Δ|(core3)
             这是唯一能区分「载荷量」与「具体单元」的对照：两者都是负号小队单元
C5 特异性    与 H5 的 rand97（打坏 1/110）比，core3 的打坏数须 ≥ 5 才称得上特异

判读
----
C1+C2+C3+C4 全成立 → 「承载」正式成立：L34 的这个功能由 3 个可点名的单元承载，
                     其余 94 个 top-1% 单元（含归因最大的 40 个）是冗余/惰性。
                     下一步：查这 3 个单元在写什么方向、读什么输入（真编码破解起点）。
C2 成立但 C1 不成立 → 只在小题集上成立，全量上另有一批承载单元，需重做自适应二分。
C4 不成立          → 打坏量随 |Δ| 单调，H12 是小样本噪声，回到「载荷量」解释。

用法
----
  SMOKE=1 .venv/Scripts/python.exe -X utf8 tools/mech_core.py    # 3 题 4 条件，验 hook 与 base 对齐
  .venv/Scripts/python.exe -X utf8 tools/mech_core.py            # 169 题 4 条件（约 90 min）
"""
from __future__ import annotations

import json
import os
import time
from math import comb
from pathlib import Path

import numpy as np
import torch

import mech_pair_sign as msign
from server import lang_atlas_service as latlas
from server import mech_pair_service as mpair

ROOT = Path(__file__).resolve().parents[1]
B = int(os.environ.get('B', '34'))
NQ = int(os.environ.get('NQ', '169'))
SMOKE = os.environ.get('SMOKE') == '1'
SEED = 20261101        # 与 H5/H6/H8/H10/H12 相同：题序一致才能跨轮比
MAXNEW = int(os.environ.get('MAXNEW', '260'))
OUT = Path('tempdata') / f'core_L{B}.json'
REF_DOSE = Path('tempdata') / f'dose_L{B}.json'          # H8：base 对齐校验用
REF_CARRIER = Path('tempdata') / f'carrier_L{B}.json'    # H12：core3 / start_set 来源


def build(q) -> str:
    return msign.build(q)


def parse(txt: str):
    return msign.parse(txt)


def load_rows():
    """与 H8 完全相同的题序（同一 SEED、同一 spec）。"""
    rows_all = [r for r in latlas._load_questions() if r.get('source')]
    cm = [r for r in rows_all if r['source'] == 'CMMLU test']
    by_sub = {}
    for r in cm:
        by_sub.setdefault(r['subject'], []).append(r)
    rng = np.random.default_rng(SEED)
    spec = [('high_school_biology', 19), ('high_school_geography', 19),
            ('chinese_history', 10), ('world_history', 10),
            ('chinese_civil_service_exam', 47), ('logical', 22),
            ('modern_chinese', 23), ('ancient_chinese', 19)]
    pool = []
    for s, k in spec:
        p = by_sub[s]
        pool += [p[i] for i in rng.choice(len(p), size=k, replace=False)]
    return pool[:NQ]


def mcnemar(a: int, b: int) -> float:
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def main() -> int:
    t_all = time.time()
    model, tok, device, _meta, model_name = mpair._wait_ready()

    rows = load_rows()
    nq = 3 if SMOKE else NQ
    rows = rows[:nq]

    a = np.load('tempdata/unit_attr_L%d.npz' % B)['attr_sum'].astype(np.float64)
    d_ff = a.size

    # ---- 单元集合：tailN 按 H12 的定义复算，并与 carrier_L34.json 的 start_set 断言一致 ----
    order = np.argsort(-np.abs(a))
    top = order[:97]
    pos_all, neg_all = top[a[top] > 0], top[a[top] < 0]
    m = int(min(pos_all.size, neg_all.size))
    neg45 = neg_all[np.argsort(-np.abs(a[neg_all]))][:m]
    head40 = np.array(sorted(int(u) for u in order[:40]))
    tailN = sorted(int(u) for u in np.setdiff1d(neg45, head40))

    carrier = json.loads(REF_CARRIER.read_text(encoding='utf-8')) if REF_CARRIER.exists() else {}
    core3 = [int(u) for u in carrier.get('final_units', [767, 1540, 35])]
    start = sorted(int(u) for u in carrier.get('start_set', tailN))
    assert start == tailN, 'tailN 复算与 H12 起始集不一致：%s vs %s' % (tailN, start)
    assert set(core3) <= set(tailN), 'core3 不在 tailN 内'
    core5 = [83, 149, 383, 583, 2423]          # H12 里被淘汰的 A' 分支
    assert not (set(core5) & set(core3)), 'core5 与 core3 必须互斥'

    conds = ['base', 'core3', 'core5', 'tailN']
    sets = {'core3': core3, 'core5': core5, 'tailN': tailN}
    masks = {'base': None}
    for nm, us in sets.items():
        masks[nm] = np.isin(np.arange(d_ff), np.array(us))

    sN = float(np.abs(a[tailN]).sum())
    print('ready %.0fs  %d 题' % (time.time() - t_all, len(rows)), flush=True)
    for nm in ('core3', 'core5', 'tailN'):
        us = sets[nm]
        print('  %-6s n=%-3d Σ|Δ|=%.3f  占 tailN %.1f%%  %s'
              % (nm, len(us), np.abs(a[us]).sum(), 100 * np.abs(a[us]).sum() / sN, us), flush=True)
    print('  core3 的 |Δ| 全层排名: %s / %d'
          % (['%d' % (list(order).index(u) + 1) for u in core3], d_ff), flush=True)

    def gen(prompt, mask):
        enc = tok(prompt, return_tensors='pt').to(device)
        hs = []
        if mask is not None:
            msk = torch.as_tensor(mask, device=device)

            def abl(_mod, args):
                v = args[0].clone()
                v[..., msk] = 0.0
                return (v,) + tuple(args[1:])
            hs.append(model.model.layers[B].mlp.down_proj.register_forward_pre_hook(abl))
        try:
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=MAXNEW, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
            return tok.decode(o[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
        finally:
            for h in hs:
                h.remove()

    recs = []
    t0 = time.time()
    for qi, q in enumerate(rows):
        prompt = build(q)
        r = {'qi': qi, 'subject': q['subject'], 'gold': q['answer'], 'out': {}}
        for nm in conds:
            r['out'][nm] = {'pred': parse(gen(prompt, masks[nm]))}
        recs.append(r)
        print('  %d/%d %.0fs  gold %s  %s'
              % (qi + 1, len(rows), time.time() - t0, q['answer'],
                 '  '.join('%s=%s' % (nm, r['out'][nm]['pred']) for nm in conds)), flush=True)

    # ---- 口径（R-MET-06）：base 有效的题作分母；改答率另算 ----
    valid = [i for i, r in enumerate(recs) if r['out']['base']['pred'] is not None]
    n = len(valid)
    base_right = [i for i in valid if recs[i]['out']['base']['pred'] == recs[i]['gold']]
    nb = len(base_right)
    base_acc = nb / n if n else 0.0

    res, dmg_sets = {}, {}
    for nm in ('core3', 'core5', 'tailN'):
        bad = [i for i in base_right
               if recs[i]['out'][nm]['pred'] is None or recs[i]['out'][nm]['pred'] != recs[i]['gold']]
        good = [i for i in valid if i not in base_right
                and recs[i]['out'][nm]['pred'] == recs[i]['gold']]
        ch = sum(1 for i in valid if recs[i]['out'][nm]['pred'] != recs[i]['out']['base']['pred'])
        dmg_sets[nm] = set(bad)
        acc = sum(1 for i in valid if recs[i]['out'][nm]['pred'] == recs[i]['gold']) / n
        res[nm] = {
            'n_units': len(sets[nm]), 'sum_abs': round(float(np.abs(a[sets[nm]]).sum()), 3),
            'share_of_tailN': round(100 * float(np.abs(a[sets[nm]]).sum()) / sN, 1),
            'acc': round(acc, 4), 'delta_pt': round(100 * (acc - base_acc), 2),
            'damaged': len(bad), 'damaged_rate_of_base_right': round(len(bad) / nb, 4) if nb else None,
            'rescued': len(good), 'net': len(good) - len(bad),
            'changed': ch, 'change_rate': round(ch / n, 4),
            'p_vs_base': round(mcnemar(len(bad), len(good)), 6),
            'damaged_qi': sorted(bad),
        }

    # ---- 题集结构 ----
    d3, d5, dN = dmg_sets['core3'], dmg_sets['core5'], dmg_sets['tailN']
    struct = {
        'core3_subset_of_tailN': sorted(d3 - dN),
        'core3_not_in_tailN_n': len(d3 - dN),
        'tailN_not_in_core3': sorted(dN - d3),
        'tailN_not_in_core3_n': len(dN - d3),
        'core5_vs_core3_overlap': sorted(d5 & d3),
        'core5_only': sorted(d5 - d3),
    }

    # ---- base 对齐校验（与 H8 的 dose_L34.json 逐题比，防止题序漂移）----
    align = {'vs_H8': None, 'vs_H8_n': None}
    if REF_DOSE.exists():
        dd = json.loads(REF_DOSE.read_text(encoding='utf-8'))
        rr = {r['qi']: r for r in dd['records']}
        sub = [i for i in valid if i in rr and rr[i]['out'].get('base', {}).get('pred') is not None]
        align = {'vs_H8': sum(1 for i in sub
                              if rr[i]['out']['base']['pred'] == recs[i]['out']['base']['pred']),
                 'vs_H8_n': len(sub)}

    # ---- 判据 ----
    ratio_share = res['core3']['sum_abs'] / res['tailN']['sum_abs'] if res['tailN']['sum_abs'] else 1.0
    ratio_dmg = res['core3']['damaged'] / res['tailN']['damaged'] if res['tailN']['damaged'] else 0.0
    v = {
        'C1_replicable': bool(struct['core3_not_in_tailN_n'] <= 1),
        'C2_sufficient': bool(ratio_dmg >= 0.6),
        'C3_share_paradox': bool(ratio_share <= 0.20 and ratio_dmg >= 0.60),
        'C4_non_monotone': bool(res['core5']['damaged'] < res['core3']['damaged']),
        'C5_specific': bool(res['core3']['damaged'] >= 5),
        'ratio_share': round(ratio_share, 3), 'ratio_damage': round(ratio_dmg, 3),
    }

    ok = all(v[k] for k in ('C1_replicable', 'C2_sufficient', 'C3_share_paradox',
                            'C4_non_monotone', 'C5_specific'))
    if ok:
        reading = ('承载成立：3 个单元（%s，占 tailN 归因量 %.0f%%）在全量 %d 道 base 对的题上'
                   '打坏 %d 道，而归因量是它 %.1f 倍的 core5 只打坏 %d 道。'
                   '「|Δ| 排序 = 重要性」在全量上被否证；L34 的这个功能有可点名的承载核。'
                   % (core3, 100 * ratio_share, nb, res['core3']['damaged'],
                      res['core5']['sum_abs'] / res['core3']['sum_abs'], res['core5']['damaged']))
    elif not v['C1_replicable']:
        reading = ('不可复制：core3 在全量上有 %d 道题是 tailN 打得坏、core3 打不坏的，'
                   'H12 的 17 题结论不能外推，需在全量上重做自适应二分。'
                   % struct['core3_not_in_tailN_n'])
    elif not v['C4_non_monotone']:
        reading = ('非单调对照不成立：core5（归因量 2.2×）打坏数不少于 core3，'
                   '提示打坏量随 |Δ| 单调，H12 是 17 题小样本噪声。')
    else:
        reading = '部分成立：%s' % json.dumps(v, ensure_ascii=False)

    out = {
        'schema': 'mech_core.v1', 'model_name': model_name, 'block': B,
        'max_new_tokens': MAXNEW, 'n_records': len(recs), 'n_valid': n,
        'n_base_right': nb, 'base_acc': round(base_acc, 4),
        'sets': sets, 'results': res, 'structure': struct,
        'base_align': align, 'verdict': v, 'reading': reading,
        'note': ('单元按 |Δ|=|attr_sum| 定义 tailN（与 H8/H10/H12 同源）；'
                 'core3 来自 H12 的 final_units，core5 是 H12 被淘汰的 A\' 分支；'
                 '本轮 base 自带，不复用 H8，避免跨轮 base 漂移。'),
        'records': recs, 'secs': round(time.time() - t_all, 1),
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8', newline='')

    print('\n有效 n=%d  base 对 %d 道（acc %.3f）' % (n, nb, base_acc))
    if align.get('vs_H8_n'):
        print('base 逐题对齐 H8: %s/%s（不一致说明题序没对上，整轮作废）'
              % (align['vs_H8'], align['vs_H8_n']))
    print('%-8s %4s %8s %8s %7s %8s %9s' % ('条件', 'n', 'Σ|Δ|', '份额', '打坏', '改答率', 'Δ'))
    for nm in ('core3', 'core5', 'tailN'):
        c = res[nm]
        print('%-8s %4d %8.3f %7.1f%% %7d %8.3f %+8.2fpt'
              % (nm, c['n_units'], c['sum_abs'], c['share_of_tailN'],
                 c['damaged'], c['change_rate'], c['delta_pt']))
    print('\n题集结构 %s' % json.dumps(struct, ensure_ascii=False))
    print('\n判据 %s' % json.dumps(v, ensure_ascii=False))
    print('读法：%s' % reading)
    print('written %s' % OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
