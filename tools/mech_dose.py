# -*- coding: utf-8 -*-
"""H8 剂量-反应：L34 top-1% 单元的「冗余饱和点」在哪。

H6 已经测出一个反常形状：删 45 个（−5.11 pt）与删 90 个（−5.11 pt）没有任何差别，
而同层随机删 97 个是 +0.73 pt。也就是说这批单元是一个饱和的共享池——删到某个数量就
已经吃掉全部可损伤的功能，再删不额外掉分。这一轮要回答的是**那个数量是多少**。

设计
----
单元按 |Δ|（一阶归因绝对值）降序排列（与 H5 的 top-97 / P 同源，不按符号分组——
H6 已证符号无方向性，R-MLP-14）。取前 k 个，k ∈ DOSE（缺省 10 / 20 / 40），
全部 ×0（down_proj 输入置零，与 H6 同一个 hook）。

已有点（不重跑，作为曲线的高剂量端参照）：H5 的 k=97（−5.84 pt）、H6 的 k=90（−5.11 pt）。

预注册判据
----------
D1 单调      acc(k) 随 k 不增（允许 1 题抖动，即 acc 回升 ≤ 1/169）
D2 饱和点    Δ(k) 达到 Δ(97) 的 80% 时的最小 k，记 k80；Δ 取相对 base 的百分点
D3 有参照    k=10 的 Δ 必须显著区别于 0（|Δ| ≥ 3 pt），否则曲线起点就在噪声里、k80 无意义

判读
----
k80 ≤ 20        → 极强冗余：20 个单元已经承担整池功能，池内其余是备份
20 < k80 ≤ 45   → 中等冗余：与 H6 的「删 45 达饱和」一致
k80 > 45        → 接近线性：这批单元更像独立累加的单位，不是共享池

用法
----
  SMOKE=1 .venv/Scripts/python.exe -X utf8 tools/mech_dose.py            # 3 题 2 剂量，验 hook
  .venv/Scripts/python.exe -X utf8 tools/mech_dose.py                     # 169 题 base+3 剂量（约 90 min）
  DOSE=10,20,40,60 .venv/Scripts/python.exe -X utf8 tools/mech_dose.py    # 自定义剂量
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
DOSE = [int(x) for x in os.environ.get('DOSE', '10,20,40').split(',') if x.strip()]
SEED = 20261101        # 与 H6 相同：题序一致才能对齐 base 与高剂量端参照
MAXNEW = int(os.environ.get('MAXNEW', '260'))
OUT = Path('tempdata') / f'dose_L{B}.json'
REF_PAIR = Path('tempdata') / f'pair_sign_L{B}.json'      # H6，k=90 参照
REF_COT = Path('tempdata') / f'mech_cot_L{B}_n169.json'   # H5，k=97 参照


def parse(txt: str):
    return msign.parse(txt)


def build(q) -> str:
    return msign.build(q)


def load_rows():
    """与 H6 完全相同的题序（同一 SEED、同一 spec），保证两轮可比、base 可对齐。"""
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
    doses = [5, 10] if SMOKE else sorted(set(DOSE))

    a = np.load('tempdata/unit_attr_L%d.npz' % B)['attr_sum'].astype(np.float64)
    d_ff = a.size
    order = np.argsort(-np.abs(a))          # 按 |Δ| 降序，全层
    assert d_ff // 100 == 97, f'预期 top-1% = 97，实际 {d_ff // 100}'

    conds = ['base'] + [f'k{d}' for d in doses]
    masks = {'base': None}
    for d in doses:
        idx = order[:d]
        masks[f'k{d}'] = np.isin(np.arange(d_ff), idx)
    print('ready %.0fs  %d 题 | %s 剂量 Σ|Δ|: %s'
          % (time.time() - t_all, len(rows), doses,
             ', '.join('k%d=%.1f' % (d, np.abs(a[order[:d]]).sum()) for d in doses)), flush=True)

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
            txt = gen(prompt, masks[nm])
            r['out'][nm] = {'pred': parse(txt), 'text': txt.strip()}
        recs.append(r)
        print('  %d/%d %.0fs  gold %s  %s'
              % (qi + 1, len(rows), time.time() - t0, q['answer'],
                 '  '.join('%s=%s' % (nm, r['out'][nm]['pred']) for nm in conds)), flush=True)

    # ---- 严格口径：所有条件 pred 非空 ----
    qs = [i for i, r in enumerate(recs) if all(r['out'][c]['pred'] is not None for c in conds)]
    n = len(qs)

    def acc(c):
        return sum(1 for i in qs if recs[i]['out'][c]['pred'] == recs[i]['gold']) / n

    base = acc('base')
    curve = {}
    for d in doses:
        c = f'k{d}'
        w = sum(1 for i in qs if recs[i]['out']['base']['pred'] == recs[i]['gold']
                and recs[i]['out'][c]['pred'] != recs[i]['gold'])
        rt = sum(1 for i in qs if recs[i]['out']['base']['pred'] != recs[i]['gold']
                 and recs[i]['out'][c]['pred'] == recs[i]['gold'])
        ch = sum(1 for i in qs if recs[i]['out'][c]['pred'] != recs[i]['out']['base']['pred'])
        curve[d] = {'acc': round(acc(c), 4), 'delta_pt': round(100 * (acc(c) - base), 2),
                    'flip': {'wrong': w, 'right': rt}, 'net': rt - w,
                    'changed': ch, 'change_rate': round(ch / n, 4),
                    'p_vs_base': round(mcnemar(w, rt), 6)}

    # ---- 高剂量端参照：H5 k=97、H6 k=90；同时校验题序与 base 是否与 H6 对齐 ----
    # Δ 必须在同一子集上算（分子分母同源），且要用当轮 base 而非全局 base_acc
    def ref_delta(records, cond):
        rr = {r['qi']: r for r in records}
        sub = [i for i in qs if i in rr and rr[i]['out'].get(cond, {}).get('pred') is not None]
        if len(sub) < 0.9 * n:
            return None, len(sub)
        b = sum(1 for i in sub if recs[i]['out']['base']['pred'] == recs[i]['gold']) / len(sub)
        c = sum(1 for i in sub if rr[i]['out'][cond]['pred'] == rr[i]['gold']) / len(sub)
        return round(100 * (c - b), 2), len(sub)

    refs, base_align = {}, {}
    if REF_COT.exists():
        d5 = json.loads(REF_COT.read_text(encoding='utf-8'))
        refs['k97'], nb = ref_delta(d5['records'], 'P')
        base_align['vs_H5'] = sum(1 for r in d5['records']
                                  if r['qi'] < len(recs)
                                  and r['out'].get('base', {}).get('pred')
                                  == recs[r['qi']]['out']['base']['pred'])
        base_align['vs_H5_n'] = sum(1 for r in d5['records'] if r['qi'] < len(recs))
    if REF_PAIR.exists():
        d6 = json.loads(REF_PAIR.read_text(encoding='utf-8'))
        refs['k90'], _ = ref_delta(d6['records'], 'both')

    # ---- 判据 ----
    # Δ(97) 的 100% 基准：H5 严格口径实测 −5.84 pt（tempdata/pair_sign_L34_strict.json 的 P 条件）。
    # 不用当轮 refs['k97']，因为它在小题集上噪声大、且会把 0.8 阈值推到荒谬位置。
    DELTA_FULL = -5.84
    v = {}
    ds = list(doses)
    v['D1_monotone'] = bool(all(curve[ds[i + 1]]['delta_pt'] <= curve[ds[i]]['delta_pt'] + 100.0 / n
                               for i in range(len(ds) - 1)))
    k80 = None
    for d in ds:
        if curve[d]['delta_pt'] <= 0.8 * DELTA_FULL:
            k80 = d
            break
    v['D2_saturation_k80'] = k80
    v['D3_start_beyond_noise'] = bool(abs(curve[ds[0]]['delta_pt']) >= 3.0)
    target = 0.8 * DELTA_FULL
    full = DELTA_FULL
    if k80 is None:
        reading = ('饱和点高于本轮最大剂量 %d：曲线在测到的范围内仍接近线性，'
                   '这批单元更像独立累加的单位，不是共享池。' % max(ds))
    elif k80 <= 20:
        reading = ('极强冗余：删 %d 个就已达到删 97 个的 %.0f%% 损伤（k80=%d）。'
                   '池内其余 %d 个基本是备份，功能主要由最靠前的 %d 个承载。'
                   % (k80, 100 * curve[k80]['delta_pt'] / full, k80, 97 - k80, k80))
    elif k80 <= 45:
        reading = ('中等冗余：k80=%d，与 H6「删 45 个即达饱和」一致；'
                   '池内冗余比例约 %.0f%%。' % (k80, 100 * (1 - k80 / 97)))
    else:
        reading = ('接近线性：k80=%d，删得越多掉得越多，'
                   '提示这批单元更像独立累加而非冗余备份。' % k80)

    out = {
        'schema': 'mech_dose.v1', 'model_name': model_name, 'block': B,
        'max_new_tokens': MAXNEW, 'doses': doses, 'n_records': len(recs), 'n_strict': n,
        'base_acc': round(base, 4), 'curve': curve, 'refs': refs,
        'base_align_vs_H6_and_H5': base_align,
        'delta_full_k97': full, 'k80_threshold_pt': round(target, 2),
        'verdict': v, 'reading': reading,
        'note': '单元按 |Δ| 降序（不按符号）；refs 用 H5/H6 已跑结果作高剂量端，未重跑。',
        'records': recs, 'secs': round(time.time() - t_all, 1),
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8', newline='')

    print('\n严格子集 n=%d  base acc=%.3f' % (n, base))
    if base_align:
        print('base 逐题对齐 H5: %s/%s（不一致说明题序没对上，整轮作废）'
              % (base_align.get('vs_H5'), base_align.get('vs_H5_n')))
    print('%-8s %8s %9s %8s' % ('剂量', '准确率', 'Δ', '改答率'))
    for d in ds:
        c = curve[d]
        print('%-8s %8.3f %+8.2fpt %8.3f' % (f'k{d}', c['acc'], c['delta_pt'], c['change_rate']))
    for k, val in refs.items():
        print('%-8s %9.2fpt（H5/H6 参照）' % (k, val))
    print('\n判据 %s' % json.dumps(v, ensure_ascii=False))
    print('读法：%s' % reading)
    print('written %s' % OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
