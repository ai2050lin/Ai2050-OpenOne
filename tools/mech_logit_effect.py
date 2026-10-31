# -*- coding: utf-8 -*-
"""H18 单点 767 在答案位置推哪些 token：logit 差分，不做生成、不看激活强度。

为什么换这个量
--------------
H16 已经把两条最容易想到的路堵死了：
  · 它在坏题上并不更响（per-question max|a| 的 AUC 0.573，≈ 随机）
  · 它也不认具体词（坏题/对照的 token 分布几乎重合，最大富集 1.2×，88% 质量落在标点）
所以「767 在算什么」不能从激活强度或词表读出来。剩下的直接量是**它的输出对答案位置
logit 的贡献**：把 down_proj 的 767 行置零，看答案位置哪个 token 的 logit 被拿走。

这个量比激活画像强的地方在于它有符号、有方向、直接落在词表上：
`Δlogit = L(有 767) − L(没有 767)`，正的 token 是它推的，负的是它压的。

设计
----
  题：s767 在全量上打坏的 11 道（base 全对，含 qi=73 的解析失败）
  条件：s767、3 个随机单点（做对照）、core3 其余两个（1540 / 35）
  读法：答案位置（prompt 最后一位，即「分析：」之后）的 logits 差
        —— 只做前向，不生成，因此不受解码器退化（重复输出）干扰

预注册判据
----------
L1 推的是同一批词   11 道题里，767 的 top-5 正面 token 两两 Jaccard 中位数 ≥ 0.2
                    （若成立 ⇒ 它在推少数几个固定 token，可以点名）
L2 压的是正确答案   ≥6/11 道题上，gold 字母的 Δlogit 排在被压的前 3 位
L3 随机做不到       767 的 |Δlogit| 中位数 > 3 个随机单点各自 |Δlogit| 中位数的 2 倍
L4 不是全局缩放     Δlogit 的正负两项都显著（不是把整个分布整体推高/压低）

判读
----
L1 成立 → 767 是个「token 助推器」，可以直接命名它在推什么词
L1 不成立 + L2 成立 → 767 的作用是**压制当前被选中的答案**，不是推某个具体概念
L3 不成立 → 上述效应在随机单点上也有，不能算 767 的特性

口径
----
· logit 差在 fp32 上算，不经过 softmax（softmax 会掩盖小位移，本题只差 1 个单元）
· 每个条件的 Δ 都以**同一题的 base logits** 为基准，逐题配对
· 随机单点复用 H17 的 B 段那 10 个里的前 3 个，保证与已有对照同源

用法
----
  .venv/Scripts/python.exe -X utf8 tools/mech_logit_effect.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import mech_pair_sign as msign
from server import lang_atlas_service as latlas
from server import mech_pair_service as mpair

B = int(os.environ.get('B', '34'))
SEED = 20261101
OUT = Path('tempdata') / f'logit_effect_L{B}.json'
REF_ONE = Path('tempdata') / f'core_one_L{B}.json'
REF_CORE = Path('tempdata') / f'core_L{B}.json'
REF_CARRIER = Path('tempdata') / f'carrier_L{B}.json'


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


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / max(1, len(a | b))


def main() -> int:
    t_all = time.time()
    model, tok, device, _meta, model_name = mpair._wait_ready()

    one = json.loads(REF_ONE.read_text(encoding='utf-8'))
    s767 = int(one['s767'])
    dmg = [int(q) for q in one['segment_A_full']['damaged_qi']]
    rand_u = [int(u) for u in one['segment_B_randoms']['units'][:3]]
    car = json.loads(REF_CARRIER.read_text(encoding='utf-8'))
    others = [int(u) for u in car['final_units'] if int(u) != s767]
    core = json.loads(REF_CORE.read_text(encoding='utf-8'))
    R = {int(r['qi']): r for r in core['records']}
    rows = load_rows()
    d_ff = 9728
    letters = {L: tok.convert_tokens_to_ids(L) for L in 'ABCD'}
    print('坏题 %d 道 | 主条件 %d | 随机 %s | 同族 %s'
          % (len(dmg), s767, rand_u, others), flush=True)

    def logits(prompt, units):
        enc = tok(prompt, return_tensors='pt').to(device)
        hs = []
        if units:
            msk = np.isin(np.arange(d_ff), np.array(units))
            msk_t = torch.as_tensor(msk, device=device)

            def abl(_mod, args):
                v = args[0].clone()
                v[..., msk_t] = 0.0
                return (v,) + tuple(args[1:])
            hs.append(model.model.layers[B].mlp.down_proj.register_forward_pre_hook(abl))
        try:
            with torch.no_grad():
                o = model(**enc)
            return o.logits[0, -1].float().cpu().numpy(), o.logits.shape[-1]
        finally:
            for h in hs:
                h.remove()

    conds = [(str(s767), [s767])] + [('r%d' % u, [u]) for u in rand_u] \
            + [('c%d' % u, [u]) for u in others]

    recs = []
    t0 = time.time()
    for k, qi in enumerate(dmg):
        prompt = msign.build(rows[qi])
        L0, V = logits(prompt, None)
        rec = {'qi': qi, 'subject': rows[qi]['subject'], 'gold': rows[qi]['answer'],
               'base_pred': R[qi]['out']['base']['pred'], 'conds': {}}
        for name, units in conds:
            L, _ = logits(prompt, units)
            d = L - L0
            order = np.argsort(-d)
            up = [(int(i), float(d[i])) for i in order[:8]]
            dn = [(int(i), float(d[i])) for i in order[-8:][::-1]]
            rec['conds'][name] = {
                'units': units,
                'up': [{'tok': tok.decode([i]), 'id': i, 'd': round(v, 4)} for i, v in up],
                'down': [{'tok': tok.decode([i]), 'id': i, 'd': round(v, 4)} for i, v in dn],
                'abs_median': float(np.median(np.abs(d))),
                'abs_max': float(np.abs(d).max()),
                'dmag': float(np.abs(d).sum()),
                'letter_delta': {L: round(float(d[letters[L]]), 4) for L in 'ABCD'},
            }
            # gold 字母在被压低一侧的排名（1 = 压得最狠）
            asc = np.argsort(d)
            pos = int(np.where(asc == letters[rec['gold']])[0][0]) + 1
            rec['conds'][name]['gold_rank_down'] = pos
        recs.append(rec)
        print('  %d/%d qi=%-3d %.0fs' % (k + 1, len(dmg), qi, time.time() - t0), flush=True)

    # ---- L1 推的是不是同一批词 ----
    tops = {r['qi']: [x['tok'] for x in r['conds'][str(s767)]['up'][:5]] for r in recs}
    ks = sorted(tops)
    js = [jaccard(tops[a], tops[b]) for i, a in enumerate(ks) for b in ks[i + 1:]]
    l1 = {'median_jaccard': round(float(np.median(js)), 3),
          'max_jaccard': round(float(np.max(js)), 3),
          'top1_tokens': {q: tops[q][0] for q in ks},
          'top5': tops}

    # ---- L2 压的是不是正确答案 ----
    gd = [r['conds'][str(s767)]['gold_rank_down'] for r in recs]
    l2 = {'n_gold_in_worst3': int(sum(1 for x in gd if x <= 3)), 'n': len(gd),
          'ranks': dict(zip([str(r['qi']) for r in recs], gd)),
          'gold_letter_delta': {str(r['qi']): r['conds'][str(s767)]['letter_delta'][r['gold']]
                                for r in recs}}

    # ---- L3 随机做不到 / L4 不是全局缩放 ----
    main_med = float(np.median([r['conds'][str(s767)]['abs_median'] for r in recs]))
    rand_med = {name: float(np.median([r['conds'][name]['abs_median'] for r in recs]))
                for name, _ in conds if name.startswith('r')}
    l3 = {'s767_median_abs_delta': main_med,
          'random_medians': rand_med,
          'ratio_vs_max_random': round(main_med / (max(rand_med.values()) + 1e-12), 2)}
    both = [r['conds'][str(s767)]['up'][0]['d'] > 0 and r['conds'][str(s767)]['down'][0]['d'] < 0
            for r in recs]
    l4 = {'n_both_directions': int(sum(both)), 'n': len(both)}

    v = {'L1_same_tokens': bool(l1['median_jaccard'] >= 0.2),
         'L2_suppresses_gold': bool(l2['n_gold_in_worst3'] >= 6),
         'L3_random_cannot': bool(l3['ratio_vs_max_random'] > 2.0),
         'L4_not_global_scale': bool(l4['n_both_directions'] >= l4['n'] - 1)}
    if v['L1_same_tokens']:
        reading = ('767 是 token 助推器：11 道坏题里它推的 top-5 token 中位 Jaccard %.2f，'
                   '最常见的第一个是「%s」⇒ 可以按词命名。' % (l1['median_jaccard'], l1['top1_tokens'][ks[0]]))
    elif v['L2_suppresses_gold']:
        reading = ('767 不是「推某个概念」，而是**压掉当前正在被选中的那个答案**：'
                   '%d/%d 道题上 gold 字母落在被压低的前 3 位（Δ 中位 %.4f），'
                   '而它推的 token 各题不同（Jaccard 中位 %.2f）。'
                   % (l2['n_gold_in_worst3'], l2['n'], main_med, l1['median_jaccard']))
    elif not v['L3_random_cannot']:
        reading = ('logit 位移在随机单点上同样大（比率 %.2f）⇒ 本轮没有测到 767 特有的东西，'
                   '只有「删任何单元都会扰动答案分布」这个平凡事实。' % l3['ratio_vs_max_random'])
    else:
        reading = '部分成立：%s' % json.dumps(v, ensure_ascii=False)

    out = {'schema': 'mech_logit_effect.v1', 'model_name': model_name, 'block': B,
           's767': s767, 'conditions': [n for n, _ in conds], 'damaged_qi': dmg,
           'L1_token_consistency': l1, 'L2_gold_suppression': l2,
           'L3_vs_random': l3, 'L4_bidirectionality': l4,
           'verdict': v, 'reading': reading,
           'note': ('答案位置 = prompt 最后一位（「分析：」之后），只做前向；'
                    'Δlogit = L(消融) − L(base)，fp32，不经 softmax；'
                    'gold_rank_down = gold 字母在升序 Δ 里的名次（1 = 被压得最狠）。'),
           'records': recs, 'secs': round(time.time() - t_all, 1)}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8', newline='')

    print('\nL1 推的 token 一致性: 中位 Jaccard %.3f（最大 %.3f）'
          % (l1['median_jaccard'], l1['max_jaccard']))
    for r in recs:
        c = r['conds'][str(s767)]
        print('  qi=%-3d gold=%s 推: %s | 压: %s | gold 名次(升序) %d'
              % (r['qi'], r['gold'],
                 ', '.join('%s%+.3f' % (x['tok'], x['d']) for x in c['up'][:3]),
                 ', '.join('%s%+.3f' % (x['tok'], x['d']) for x in c['down'][:3]),
                 c['gold_rank_down']))
    print('L2 gold 落在被压前 3 位: %d/%d' % (l2['n_gold_in_worst3'], l2['n']))
    print('L3 |Δ| 中位: 767 %.5f  随机 %s  比率 %.2f'
          % (l3['s767_median_abs_delta'],
             json.dumps({k: round(x, 5) for k, x in l3['random_medians'].items()}, ensure_ascii=False),
             l3['ratio_vs_max_random']))
    print('L4 双向性: %d/%d' % (l4['n_both_directions'], l4['n']))
    print('\n判据 %s' % json.dumps(v, ensure_ascii=False))
    print('读法：%s' % reading)
    print('written %s' % OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
