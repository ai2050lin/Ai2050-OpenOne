# -*- coding: utf-8 -*-
"""H10 头部惰性 vs 尾部承载：|Δ| 排名到底能不能预测必要性。

H8 摆出一个自相矛盾的事实（同一个 hook、同一批题、同一份归因数组）：

  top-40（按 |Δ| 降序，Σ|Δ| 73.63，占 top-97 的 76%）  →  −0.74 pt，几乎无损
  pos45（全部 45 个正号单元，Σ|Δ| 48.77）              →  −5.11 pt
  neg45（45 个最大负号单元，Σ|Δ| 45.54）               →  −4.38 pt
  随机 97 个（H6 对照）                                →  +0.73 pt

top-40 里已经包含了 pos45 的 18 个、neg45 的 22 个，却什么也没发生。也就是说
「归因幅度最大的 40 个单元」在功能上是惰性的，而效应来自第 41–97 名那批小队单元。
这一轮要分清两种解释：

  A 尾部承载   效应就在「pos45 \\ top40」（27 个）里，符号类的完整性无关；
               那 |Δ| 排序在头部是**反预测**的——归因幅度大 ≠ 必要。
  B 类覆盖面   必须把某一符号类几乎删干净才触发，删 27 个不够；
               那 H6 的「删一半 = 删全部」读法是错的，"饱和"其实是"类覆盖阈值"。

设计
----
四个条件，全部 ×0（down_proj 输入置零，与 H6/H8 同一个 hook）：

  tailP  = pos45 \\ top40   （27 个，正号里排名靠后的部分）
  tailN  = neg45 \\ top40   （23 个，负号里排名靠后的部分）
  k60    = |Δ| 降序前 60    （定位悬崖位置）
  k80    = |Δ| 降序前 80

base **不重跑**，直接复用 tempdata/dose_L34.json 的逐题 base（贪心解码，H5/H6/H8
三轮逐题一致 169/169，已两次验证确定性）。严格子集 = base 有效 且 四个新条件全有效。

预注册判据
----------
T1 尾部承载    |Δ(tailP)| ≥ 3.0pt 或 |Δ(tailN)| ≥ 3.0pt
T2 类覆盖阈值  两个 tail 的 |Δ| 都 < 1.5pt（说明删 27/23 个不够，需接近整类）
C1 悬崖        k ∈ {40,60,80,97} 中 |Δ| 首次 ≥ 3.0pt 的最小 k（97 用 H5 实测 −5.93）

判读
----
T1 成立      → |Δ| 头部惰性、尾部承载；「归因幅度 = 重要性」在 L34 头部失效，
               必须改口径（H5 的 top-97 vs 随机 97 仍然成立，但它测的是"整池"不是"排名"）
T2 成立      → H6 的"饱和池"改为"类覆盖阈值"：效应在删掉某符号类 ~85%+ 时才出现
都不成立      → 效应依赖特定单元组合，需要逐单元留一验证（成本高，另立 Phase）

用法
----
  SMOKE=1 .venv/Scripts/python.exe -X utf8 tools/mech_headtail.py     # 3 题，验 hook 与 base 复用
  .venv/Scripts/python.exe -X utf8 tools/mech_headtail.py              # 169 题 4 条件（约 88 min）
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
SEED = 20261101        # 与 H6/H8 相同：题序一致才能复用 base
MAXNEW = int(os.environ.get('MAXNEW', '260'))
OUT = Path('tempdata') / f'headtail_L{B}.json'
REF_DOSE = Path('tempdata') / f'dose_L{B}.json'          # H8：base 复用 + k10/20/40
REF_COT = Path('tempdata') / f'mech_cot_L{B}_n169.json'  # H5：k=97


def parse(txt: str):
    return msign.parse(txt)


def build(q) -> str:
    return msign.build(q)


def load_rows():
    """与 H6/H8 完全相同的题序（同一 SEED、同一 spec）。"""
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


def masks_for(a: np.ndarray):
    """返回 {'tailP','tailN','k60','k80'} 的布尔掩码与每个掩码的单元清单。"""
    d_ff = a.size
    order = np.argsort(-np.abs(a))
    top = order[:97]
    pos_all = top[a[top] > 0]
    neg_all = top[a[top] < 0]
    m = int(min(pos_all.size, neg_all.size))
    pos = pos_all[np.argsort(-np.abs(a[pos_all]))][:m]
    neg = neg_all[np.argsort(-np.abs(a[neg_all]))][:m]
    head40 = order[:40]
    tailP = np.setdiff1d(pos, head40)
    tailN = np.setdiff1d(neg, head40)
    raw = {'tailP': tailP, 'tailN': tailN, 'k60': order[:60], 'k80': order[:80]}
    masks = {k: np.isin(np.arange(d_ff), v) for k, v in raw.items()}
    return masks, raw, order, pos, neg


def main() -> int:
    t_all = time.time()
    model, tok, device, _meta, model_name = mpair._wait_ready()

    rows = load_rows()
    nq = 3 if SMOKE else NQ
    rows = rows[:nq]
    conds = ['tailP', 'tailN', 'k60', 'k80']

    # ---- base 复用（贪心解码确定性已两次验证） ----
    if not REF_DOSE.exists():
        print('缺少 %s：先跑 H8' % REF_DOSE)
        return 2
    d8 = json.loads(REF_DOSE.read_text(encoding='utf-8'))
    base = {r['qi']: r for r in d8['records']}

    a = np.load('tempdata/unit_attr_L%d.npz' % B)['attr_sum'].astype(np.float64)
    d_ff = a.size
    assert d_ff // 100 == 97, f'预期 top-1% = 97，实际 {d_ff // 100}'
    masks, raw, order, pos, neg = masks_for(a)

    print('ready %.0fs  %d 题 | 单元集：%s'
          % (time.time() - t_all, len(rows),
             '  '.join('%s=%d(Σ|Δ|=%.2f)' % (k, v.size, np.abs(a[v]).sum())
                       for k, v in raw.items())), flush=True)
    print('  tailP ∩ tailN = %d（须为 0）' % np.intersect1d(raw['tailP'], raw['tailN']).size, flush=True)

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
        bpred = (base.get(qi) or {}).get('out', {}).get('base', {}).get('pred')
        print('  %d/%d %.0fs  gold %s  base=%s  %s'
              % (qi + 1, len(rows), time.time() - t0, q['answer'], bpred,
                 '  '.join('%s=%s' % (nm, r['out'][nm]['pred']) for nm in conds)), flush=True)

    # ---- 严格口径：base（复用）与四个新条件 pred 都非空 ----
    qs, miss = [], 0
    for i, r in enumerate(recs):
        bp = (base.get(i) or {}).get('out', {}).get('base', {}).get('pred')
        if bp is None:
            miss += 1
            continue
        if all(r['out'][c]['pred'] is not None for c in conds):
            qs.append(i)
    n = len(qs)

    def acc(c, pool=None):
        idx = pool if pool is not None else qs
        if c == 'base':
            return sum(1 for i in idx if base[i]['out']['base']['pred'] == base[i]['gold']) / len(idx)
        return sum(1 for i in idx if recs[i]['out'][c]['pred'] == recs[i]['gold']) / len(idx)

    b_acc = acc('base')
    res = {}
    for c in conds:
        w = sum(1 for i in qs if base[i]['out']['base']['pred'] == base[i]['gold']
                and recs[i]['out'][c]['pred'] != recs[i]['gold'])
        rt = sum(1 for i in qs if base[i]['out']['base']['pred'] != base[i]['gold']
                 and recs[i]['out'][c]['pred'] == recs[i]['gold'])
        ch = sum(1 for i in qs if recs[i]['out'][c]['pred'] != base[i]['out']['base']['pred'])
        res[c] = {'n_units': int(raw[c].size),
                  'sum_abs_delta': round(float(np.abs(a[raw[c]]).sum()), 2),
                  'acc': round(acc(c), 4),
                  'delta_pt': round(100 * (acc(c) - b_acc), 2),
                  'flip': {'wrong': w, 'right': rt}, 'net': rt - w,
                  'changed': ch, 'change_rate': round(ch / n, 4),
                  'p_vs_base': round(mcnemar(w, rt), 6)}

    # ---- 悬崖定位：把 H8 的 k10/20/40 并进来 ----
    cliff = dict(d8.get('curve') or {})
    cliff['60'] = {'acc': res['k60']['acc'], 'delta_pt': res['k60']['delta_pt']}
    cliff['80'] = {'acc': res['k80']['acc'], 'delta_pt': res['k80']['delta_pt']}
    cliff['97'] = {'delta_pt': (d8.get('refs') or {}).get('k97')}
    ks = sorted(int(k) for k in cliff if cliff[k].get('delta_pt') is not None)
    k_first = next((k for k in ks if cliff[str(k)]['delta_pt'] <= -3.0), None)

    v = {}
    v['T1_tail_carries'] = bool(max(abs(res['tailP']['delta_pt']), abs(res['tailN']['delta_pt'])) >= 3.0)
    v['T2_class_coverage'] = bool(abs(res['tailP']['delta_pt']) < 1.5 and abs(res['tailN']['delta_pt']) < 1.5)
    v['C1_cliff_k_first3pt'] = k_first

    if v['T1_tail_carries']:
        reading = ('尾部承载：把 top-40 拿掉之后剩下的那批小队单元（tailP %d / tailN %d）'
                   '照样能打出 ≥3pt 的损伤。|Δ| 头部在功能上是惰性的——归因幅度大不等于必要，'
                   '「按 |Δ| 排序取 top-k」在 L34 头部失效。'
                   % (res['tailP']['n_units'], res['tailN']['n_units']))
    elif v['T2_class_coverage']:
        reading = ('类覆盖阈值：删 27/23 个尾部队列单元都不动分。要触发效应必须把某一符号类'
                   '删到接近干净（H6 的 pos45=45/45、neg45=45/52）。H6 的「饱和池」应改写为'
                   '「类覆盖阈值」：不是删够了数量，而是某一类的单元被删到近乎为零。')
    else:
        reading = ('两条都不成立：效应既不在尾部队列、也不是单纯类覆盖，'
                   '依赖特定单元组合；需逐单元留一验证（成本高，另立 Phase）。')

    out = {
        'schema': 'mech_headtail.v1', 'model_name': model_name, 'block': B,
        'max_new_tokens': MAXNEW, 'conds': conds,
        'n_records': len(recs), 'n_strict': n, 'n_base_unparsed': miss,
        'base_acc': round(b_acc, 4), 'base_source': str(REF_DOSE),
        'results': res, 'cliff_merged': {str(k): cliff[str(k)] for k in ks},
        'verdict': v, 'reading': reading,
        'note': 'base 复用 H8（贪心解码确定性）；四个条件全部 ×0，down_proj 输入置零，同 H6/H8 hook。',
        'records': recs, 'secs': round(time.time() - t_all, 1),
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8', newline='')

    print('\n严格子集 n=%d  base acc=%.3f（base 未解析 %d 题，单独计数）' % (n, b_acc, miss))
    print('%-8s %7s %9s %9s %8s %9s' % ('条件', '单元数', 'Σ|Δ|', '准确率', 'Δ', '改答率'))
    for c in conds:
        r = res[c]
        print('%-8s %7d %9.2f %9.3f %+8.2fpt %9.3f  (p=%.4f)'
              % (c, r['n_units'], r['sum_abs_delta'], r['acc'], r['delta_pt'],
                 r['change_rate'], r['p_vs_base']))
    print('\n剂量曲线(k): %s' % '  '.join('%s→%+.2fpt' % (k, cliff[str(k)]['delta_pt']) for k in ks))
    print('判据 %s' % json.dumps(v, ensure_ascii=False))
    print('读法：%s' % reading)
    print('written %s' % OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
