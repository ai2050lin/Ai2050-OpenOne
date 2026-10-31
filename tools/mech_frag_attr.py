# -*- coding: utf-8 -*-
"""H11 脆弱题专属归因：那 12 道题到底依赖哪些单元。

H8/H10 摊开的事实：L34 top-1% 的「必要性」不是均匀分布在 169 道题上，而是集中在**极少数
脆弱题**上。base 答对的 110 道题里，只有 20 道会被任何一个 top-1% 子集打坏，其中
qi=73 与 qi=5 被全部 8 个条件打坏（跨 H5/H6/H8/H10 四轮）。而随机删 97 个只打坏 1 道（qi=78）。
所以「−5.9 点」≈「12 道题被打坏」，而各子集之间 −4 ~ −5 的差别其实是
「有没有覆盖到这 12 道题依赖的那组单元」——tailP 只碰坏 3 道就是没覆盖到。

这一轮问的是：这 12 道题依赖的单元，是不是一小撮？它们在已测的哪个子集里？

设计
----
对每题算一阶单元归因（与 C5' 同公式、同 hook）：

  d_i = −Σ_t x_{t,i} · ∂L/∂x_{t,i},   L = logit(正确字母) − logsumexp(其它字母)

在 **L34 的 down_proj 输入**上取 x（与 H6/H8/H10 的消融点是同一个张量），位置取
prompt 末位（即「分析：」之后要发的第一个 token，也就是决策点）。
prompt 用 `msign.build`（与 H5–H10 逐字相同），保证与行为实验同口径。

对象：12 道脆弱题 + 12 道对照题（base 对、且从未被任何条件打坏，按 qi 等距抽取）。

预注册判据
----------
U1 不同单元     脆弱题的 top-97 与全局 top-97（`unit_attr_L34.npz`）的重叠率
                **低于**对照题的重叠率（差 ≥ 10 个百分点）
U2 落在尾负侧   12 道脆弱题 top-20 的并集 S，与 tailN(23) 的重叠数
                **≥ 2 倍** 与 tailP(27) 的重叠数（按集合大小归一后再比）
U3 更集中       脆弱题在 S 上的 Σ|Δ| 占其全层 Σ|Δ| 的比例，**高于**对照题同口径比例
                且相对差 ≥ 20%（判据 v2：v1 只写了「>」，0.093 vs 0.091 这种噪声差也会通过，
                属判据设计缺陷 → R-MET-09）

判读
----
U1 成立          → 这些题用的不是「全库平均意义上重要的单元」，单元级「重要性」随题而变，
                   这就解释了为什么任何子集都能碰坏它们、而平均归因排序对它们没意义
U2 成立          → 承载单元落在负号尾侧（tailN），tailP 惰性是因为压根不在同一批单元上
U3 成立          → 脆弱 = 依赖少数大 |Δ| 单元（低冗余），正常题 = 分散依赖（高冗余）
都不成立          → 「哪个子集碰坏了谁」只是随机覆盖差，需更大样本，不再拆

用法
----
  .venv/Scripts/python.exe -X utf8 tools/mech_frag_attr.py      # 24 题，约 1 分钟
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

import mech_pair_sign as msign
from server import lang_atlas_service as latlas
from server import mech_pair_service as mpair

B = 34
SEED = 20261101
OUT = Path('tempdata') / f'frag_attr_L{B}.json'
REF_DOSE = Path('tempdata') / f'dose_L{B}.json'
REF_ATTR = Path('tempdata') / f'unit_attr_L{B}.npz'

FRAGILE = [73, 5, 94, 78, 29, 125, 41, 25, 68, 97, 55, 21]


def load_rows():
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
    return pool[:169]


def main() -> int:
    t_all = time.time()
    model, tok, device, _meta, model_name = mpair._wait_ready()
    letter_ids = latlas._letter_token_ids(tok)

    rows = load_rows()
    d8 = json.loads(REF_DOSE.read_text(encoding='utf-8'))
    base = {r['qi']: r['out']['base']['pred'] for r in d8['records']}
    gold = {r['qi']: r['gold'] for r in d8['records']}

    # 对照题：base 对、且从未被任何 top-1% 子集打坏，按 qi 等距抽 12 道
    d6 = json.loads((Path('tempdata') / f'pair_sign_L{B}.json').read_text(encoding='utf-8'))
    d5 = json.loads((Path('tempdata') / f'mech_cot_L{B}_n169.json').read_text(encoding='utf-8'))
    d10 = json.loads((Path('tempdata') / f'headtail_L{B}.json').read_text(encoding='utf-8'))
    damaged = set()
    for recs, conds in [(d5['records'], ['P', 'rand97']), (d6['records'], ['pos', 'neg', 'both']),
                        (d8['records'], ['k10', 'k20', 'k40']),
                        (d10['records'], ['tailP', 'tailN', 'k60', 'k80'])]:
        for r in recs:
            q = r['qi']
            if base.get(q) is None or base[q] != gold.get(q):
                continue
            for c in conds:
                p = (r['out'].get(c) or {}).get('pred')
                if p is not None and p != gold[q]:
                    damaged.add(q)
                    break
    ok_all = [q for q in sorted(gold) if base.get(q) is not None and base[q] == gold[q]]
    controls_pool = [q for q in ok_all if q not in damaged and q not in FRAGILE]
    idx = np.linspace(0, len(controls_pool) - 1, 12).round().astype(int)
    controls = [controls_pool[i] for i in sorted(set(idx.tolist()))]
    print('ready %.0fs | 脆弱 %d 道 / 对照 %d 道 / base 对的题 %d / 被任何条件打坏 %d'
          % (time.time() - t_all, len(FRAGILE), len(controls), len(ok_all), len(damaged)), flush=True)

    a_global = np.load(REF_ATTR)['attr_sum'].astype(np.float64)
    d_ff = a_global.size
    g_order = np.argsort(-np.abs(a_global))
    g_top97 = set(int(u) for u in g_order[:97])
    pos_all = g_order[:97][a_global[g_order[:97]] > 0]
    neg_all = g_order[:97][a_global[g_order[:97]] < 0]
    m = int(min(pos_all.size, neg_all.size))
    pos45 = pos_all[np.argsort(-np.abs(a_global[pos_all]))][:m]
    neg45 = neg_all[np.argsort(-np.abs(a_global[neg_all]))][:m]
    head40 = set(int(u) for u in g_order[:40])
    tailP = set(int(u) for u in np.setdiff1d(pos45, np.array(sorted(head40))))
    tailN = set(int(u) for u in np.setdiff1d(neg45, np.array(sorted(head40))))

    def qattr(q):
        prompt = msign.build(q)
        ids = tok(prompt, return_tensors='pt')['input_ids'].to(device)
        ans_tok = next((c for c in letter_ids.get(q['answer'], [])), None)
        others = [next(iter(letter_ids.get(x) or []), None) for x in 'ABCDE'
                  if x != q['answer'] and letter_ids.get(x)]
        others = [o for o in others if o is not None]
        if ans_tok is None or not others:
            return None
        cap = {}

        def grab(_m, args):
            a = args[0]
            a.retain_grad()
            cap['x'] = a
        h = model.model.layers[B].mlp.down_proj.register_forward_pre_hook(grab)
        try:
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                o = model(input_ids=ids, use_cache=False)
                lg = o.logits[0, -1].float()
                target = lg[ans_tok] - torch.logsumexp(lg[others], dim=0)
                target.backward()
        finally:
            h.remove()
        x = cap['x']
        g = x.grad
        if g is None:
            return None
        return -(x.detach().float() * g.float()).sum(dim=(0, 1)).cpu().numpy()

    def prof(qi):
        row = rows[qi]
        d = qattr(row)
        if d is None:
            return None
        ad = np.abs(d)
        tot = float(ad.sum())
        o = np.argsort(-ad)
        t97 = set(int(u) for u in o[:97])
        return {'qi': int(qi), 'subject': row['subject'], 'gold': row['answer'],
                'base': base.get(qi),
                'sum_abs': round(tot, 3),
                'conc_top20': round(float(ad[o[:20]].sum() / tot), 4) if tot > 0 else None,
                'top20': [[int(u), round(float(d[u]), 4)] for u in o[:20]],
                'top97': sorted(t97),
                'overlap_global97': len(t97 & g_top97),
                'in_tailP': len(t97 & tailP), 'in_tailN': len(t97 & tailN),
                'in_head40': len(t97 & head40)}

    t0 = time.time()
    frag = [p for p in (prof(q) for q in FRAGILE) if p]
    print('  脆弱题归因完成 %.0fs' % (time.time() - t0), flush=True)
    t0 = time.time()
    ctrl = [p for p in (prof(q) for q in controls) if p]
    print('  对照题归因完成 %.0fs' % (time.time() - t0), flush=True)

    def mean(ps, k):
        v = [p[k] for p in ps if p[k] is not None]
        return round(float(np.mean(v)), 3) if v else None

    union20 = set()
    for p in frag:
        union20 |= set(u for u, _ in p['top20'])

    S = len(union20)
    res = {
        'fragile': {'n': len(frag), 'overlap_global97': mean(frag, 'overlap_global97'),
                    'conc_top20': mean(frag, 'conc_top20'),
                    'in_tailP': mean(frag, 'in_tailP'), 'in_tailN': mean(frag, 'in_tailN'),
                    'in_head40': mean(frag, 'in_head40')},
        'control': {'n': len(ctrl), 'overlap_global97': mean(ctrl, 'overlap_global97'),
                    'conc_top20': mean(ctrl, 'conc_top20'),
                    'in_tailP': mean(ctrl, 'in_tailP'), 'in_tailN': mean(ctrl, 'in_tailN'),
                    'in_head40': mean(ctrl, 'in_head40')},
        'union_top20': {'size': S, 'in_tailP': len(union20 & tailP), 'in_tailN': len(union20 & tailN),
                        'in_head40': len(union20 & head40), 'in_global97': len(union20 & g_top97),
                        'tailP_size': len(tailP), 'tailN_size': len(tailN)},
    }

    v = {}
    f, c = res['fragile'], res['control']
    v['U1_different_units'] = bool(f['overlap_global97'] is not None and c['overlap_global97'] is not None
                                   and c['overlap_global97'] - f['overlap_global97'] >= 10)
    rate_p = res['union_top20']['in_tailP'] / max(1, len(tailP))
    rate_n = res['union_top20']['in_tailN'] / max(1, len(tailN))
    res['union_top20']['rate_tailP'] = round(rate_p, 4)
    res['union_top20']['rate_tailN'] = round(rate_n, 4)
    v['U2_tailN_over_tailP'] = bool(rate_n >= 2 * rate_p if rate_p > 0 else rate_n > 0)
    v['U3_more_concentrated'] = bool(f['conc_top20'] is not None and c['conc_top20'] is not None
                                     and c['conc_top20'] > 0
                                     and f['conc_top20'] >= 1.20 * c['conc_top20'])
    res['U3_rel_gain'] = (round(f['conc_top20'] / c['conc_top20'], 4)
                          if (f['conc_top20'] and c['conc_top20']) else None)

    if v['U1_different_units']:
        reading = ('脆弱题用的不是全库平均意义上重要的单元（与全局 top-97 重叠 %s，对照题 %s）。'
                   '单元级「重要性」随题而变，这解释了为什么任何子集都能碰坏它们、'
                   '而按平均 |Δ| 排序对它们没有意义。'
                   % (f['overlap_global97'], c['overlap_global97']))
    elif v['U2_tailN_over_tailP']:
        reading = ('承载单元落在负号尾侧：12 道题 top-20 并集共 %d 个单元，'
                   '其中 %d 个在 tailN(23)、%d 个在 tailP(27)，归一化后 %s vs %s。'
                   % (S, res['union_top20']['in_tailN'], res['union_top20']['in_tailP'], rate_n, rate_p))
    elif v['U3_more_concentrated']:
        reading = ('脆弱题更集中：top-20 占其归因总量 %s，对照题 %s —— 脆弱 = 依赖少数大单元（低冗余），'
                   '正常题 = 分散依赖（高冗余）。' % (f['conc_top20'], c['conc_top20']))
    else:
        reading = ('三条都不成立：脆弱题与对照题在归因画像上无法区分'
                   '（与全局 top-97 重叠 %s vs %s，top-20 集中度 %s vs %s、相对差 %s 倍，'
                   'tailN 覆盖反而更低 %s vs %s）。'
                   '⇒ 单元级归因能定位「整池承载功能」，但不能预测「哪道题依赖它」，'
                   '也不能在池内排序。子集间的差异只能算覆盖差，本样本量下不拆。'
                   % (f['overlap_global97'], c['overlap_global97'],
                      f['conc_top20'], c['conc_top20'], res['U3_rel_gain'],
                      f['in_tailN'], c['in_tailN']))

    out = {'schema': 'mech_frag_attr.v1', 'model_name': model_name, 'block': B,
           'target': 'logit(gold_letter) - logsumexp(other_letters) @ prompt 末位',
           'fragile_qs': FRAGILE, 'control_qs': controls,
           'stats': res, 'verdict': v, 'reading': reading,
           'fragile_profiles': frag, 'control_profiles': ctrl,
           'note': '归因在 L34 down_proj 输入上取（与消融同张量）；prompt 用 msign.build，与 H5–H10 同口径。',
           'secs': round(time.time() - t_all, 1)}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8', newline='')

    print('\n%-14s %10s %10s' % ('指标', '脆弱题', '对照题'))
    for k, lab in [('overlap_global97', '与全局top97重叠'), ('conc_top20', 'top20集中度'),
                   ('in_tailP', 'top97∩tailP'), ('in_tailN', 'top97∩tailN'),
                   ('in_head40', 'top97∩head40')]:
        print('%-14s %10s %10s' % (lab, f[k], c[k]))
    print('\ntop-20 并集 %d 个：tailN %d/%d (%.2f)  tailP %d/%d (%.2f)  head40 %d  global97 %d'
          % (S, res['union_top20']['in_tailN'], len(tailN), rate_n,
             res['union_top20']['in_tailP'], len(tailP), rate_p,
             res['union_top20']['in_head40'], res['union_top20']['in_global97']))
    print('判据 %s' % json.dumps(v, ensure_ascii=False))
    print('读法：%s' % reading)
    print('written %s' % OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
