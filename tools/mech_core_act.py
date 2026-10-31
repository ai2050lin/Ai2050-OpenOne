# -*- coding: utf-8 -*-
"""H16 单点 767 的激活画像：这一个单元认什么 token，在坏题上是否被异常点亮。

背景（H15→H17 已定）：L34 的 down_proj 输入里有 1 个单元（767）单独消融就打坏 10 道
base 对的题，与 core3 三个一起（9 道）同量级，随机单点最多 2 道。权重上它与 1540/35
几乎不相似（写方向 |cos| ≤ 0.029）⇒ 不是副本。现在只剩一个问题：**它在算什么**。

捕获的张量 = `layers.34.mlp.down_proj` 的输入 = SwiGLU 输出 `silu(gate·x) * (up·x)`，
形状 [T, 9728]。对单元 i，`a[t,i]` 正是消融时被抹掉的那一份贡献，所以这个量就是
「767 在位置 t 往 residual 写了多少」的直接估计——不需要任何代理指标。

四个问题
--------
  A1 分得开吗      被 767 打坏的题 vs 其余 base 对的题，767 的 per-question max|a| 是否可分
  A2 认哪些 token  767 的激活质量集中在哪些 token 上（按题等权，取 top-20）
  A3 烧到没有      坏题上 767 是否进入该题 9728 个单元的 top-1%（按 max|a|）
  A4 坏题 vs 对照 767 的词表在两组的分布是否不同（描述项，不作判据）

预注册判据
----------
A1 可分     767 的 per-question max|a| 在坏/对照之间 AUC ≥ 0.75
A2 有词表   **已作废**（原写「top-20 覆盖 ≥30%」：12 题试跑里坏组 89.4% / 对照组 87.0%，
            因为中文高频标点（。，：）几乎占满 top-20 ⇒ 该判据对任何单元都恒真，没有区分度。
            与 H14-pre 撤掉 W4 同理。改为描述项 + 探索性富集表，不进判读。）
A2' 题特异  （探索性，阈值人为）存在 token 在坏题上的单位质量是对照的 ≥3 倍且绝对质量 ≥1%
A3 强激活    ≥6/11 坏题至少有 1 个 core3 单元落进该题 top-1%（按 d_ff 排名）

判读
----
A1+A2 成立 → 767 有可读的 token 身份，可往下做「它把什么写进 residual」
A3 不成立  → 打坏不是因为它在坏题上特别火，而是它的输出在这些题上不可替代
A1 不成立  → per-question 峰值不分好坏 ⇒ 别再按「激活强度」找语义，改做写方向投影

口径
----
· 坏题集用 **s767 的 11 道**（含 qi=73 的解析失败），不用 core3 的 9 道——
  767 是实际起作用的那个单元，用它的题集才对得上。qi=73 单列标记（H17 的探针证明
  它对**任意**单点消融都崩，见 probe_frag_L34.json），统计时给两种口径。
· 只做 prefill 前向、不生成；token 聚合按题等权归一，避免长题主导（长题位置多）。
· A4 是描述项不是判据：原先写过「去重 ≤5 个 token」，那是无依据的硬阈值，
  与 H14-pre 撤掉 W4 的理由相同，这里直接降级为描述。

用法
----
  .venv/Scripts/python.exe -X utf8 tools/mech_core_act.py           # 169 题 prefill（约 2 min）
  NQ=30 .venv/Scripts/python.exe -X utf8 tools/mech_core_act.py
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import mech_pair_sign as msign
from server import lang_atlas_service as latlas
from server import mech_pair_service as mpair

B = int(os.environ.get('B', '34'))
NQ = int(os.environ.get('NQ', '169'))
SEED = 20261101
OUT = Path('tempdata') / f'core_act_L{B}.json'
REF_CORE = Path('tempdata') / f'core_L{B}.json'
REF_ONE = Path('tempdata') / f'core_one_L{B}.json'
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


def auc(pos, neg):
    """Mann-Whitney AUC。pos/neg 各自已按升序无关，用秩和算。"""
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return 0.5
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # 平分秩的平均化（否则 AUC 会被并列值系统性偏置）
    srt = allv[order]
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    rp = ranks[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> int:
    t_all = time.time()
    model, tok, device, _meta, model_name = mpair._wait_ready()

    car = json.loads(REF_CARRIER.read_text(encoding='utf-8'))
    core3 = [int(u) for u in car['final_units']]
    one = json.loads(REF_ONE.read_text(encoding='utf-8'))
    s767 = int(one['s767'])
    damaged = set(int(q) for q in one['segment_A_full']['damaged_qi'])
    core = json.loads(REF_CORE.read_text(encoding='utf-8'))
    R = {int(r['qi']): r for r in core['records']}
    base_ok = [q for q, r in R.items()
               if r['out']['base']['pred'] is not None and r['out']['base']['pred'] == r['gold']]
    control = [q for q in base_ok if q not in damaged]
    print('base 对 %d 道 | 767 坏 %d（含 qi=73 解析失败）| 对照 %d'
          % (len(base_ok), len(damaged), len(control)), flush=True)

    rows = load_rows()[:NQ]
    d_ff = 9728
    units = [s767] + [u for u in core3 if u != s767]

    recs = []
    toksum = {'bad': defaultdict(lambda: np.zeros(len(units))),
              'ctl': defaultdict(lambda: np.zeros(len(units)))}
    t0 = time.time()
    for qi, q in enumerate(rows):
        prompt = msign.build(q)
        enc = tok(prompt, return_tensors='pt').to(device)
        box = {}

        def cap(_mod, args):
            box['a'] = args[0].detach()
            return None

        h = model.model.layers[B].mlp.down_proj.register_forward_pre_hook(cap)
        try:
            with torch.no_grad():
                model(**enc)
        finally:
            h.remove()

        A = box['a'][0].float().cpu().numpy()          # [T, d_ff]
        T = A.shape[0]
        ids = enc['input_ids'][0].tolist()
        absA = np.abs(A)
        colmax = absA.max(axis=0)
        order = np.argsort(-colmax)
        pos_of = {int(u): i + 1 for i, u in enumerate(order)}

        per = {}
        for u in units:
            v = absA[:, u]
            tmax = int(v.argmax())
            per[u] = {'max': round(float(v.max()), 4), 'mean': round(float(v.mean()), 5),
                      'sum': round(float(v.sum()), 3),
                      'argmax_tok_id': int(ids[tmax]),
                      'argmax_tok': tok.decode([ids[tmax]]),
                      'argmax_share': round(float(v[tmax] / (v.sum() + 1e-9)), 4),
                      'rank_in_layer': int(pos_of[u])}

        # token 聚合（按题等权：每题总质量归一为 1）
        grp = 'bad' if qi in damaged else ('ctl' if qi in control else None)
        if grp:
            tot = float(absA[:, units].sum()) or 1.0
            for t in range(T):
                toksum[grp][int(ids[t])] += absA[t, units] / tot

        recs.append({'qi': qi, 'subject': q['subject'], 'gold': q['answer'],
                     'grp': grp, 'T': T, 'per_unit': per})
        if (qi + 1) % 20 == 0 or qi + 1 == len(rows):
            print('  %d/%d %.0fs' % (qi + 1, len(rows), time.time() - t0), flush=True)

    def sub(g):
        return [r for r in recs if r['grp'] == g]

    # ---- A1 可分性（坏 vs 对照）----
    a1 = {}
    for u in units:
        bad = [r['per_unit'][u]['max'] for r in sub('bad')]
        oth = [r['per_unit'][u]['max'] for r in sub('ctl')]
        a1[str(u)] = {'auc': round(auc(bad, oth), 3),
                      'mean_bad': round(float(np.mean(bad)), 4) if bad else None,
                      'mean_ctl': round(float(np.mean(oth)), 4) if oth else None,
                      'n_bad': len(bad), 'n_ctl': len(oth)}
    # 去掉通用脆性题（qi=73）的复核
    a1_ex73 = {}
    for u in units:
        bad = [r['per_unit'][u]['max'] for r in sub('bad') if r['qi'] != 73]
        oth = [r['per_unit'][u]['max'] for r in sub('ctl')]
        a1_ex73[str(u)] = {'auc': round(auc(bad, oth), 3), 'n_bad': len(bad)}

    # ---- A2 词表 ----
    def vocab(grp, j, k=20):
        acc = toksum[grp]
        tot = sum(float(v[j]) for v in acc.values()) or 1.0
        ranked = sorted(((float(v[j]), i) for i, v in acc.items()), reverse=True)
        top = [{'tok': tok.decode([i]), 'id': i, 'w': round(w, 4)} for w, i in ranked[:k]]
        return {'top': [x for x in top if x['tok'].strip()],
                'top_share': round(sum(w for w, _ in ranked[:k]) / tot, 3),
                'n_tokens': len(ranked)}

    a2 = {str(u): {'bad': vocab('bad', j), 'ctl': vocab('ctl', j)}
          for j, u in enumerate(units)}

    # 富集表：坏组单位质量 / 对照单位质量（两侧都先归一到各自的题数）
    nb = max(1, len(sub('bad')))
    nc = max(1, len(sub('ctl')))
    enrichment = {}
    for j, u in enumerate(units):
        rows_e = []
        for tid in set(toksum['bad']) | set(toksum['ctl']):
            wb = float(toksum['bad'][tid][j]) / nb
            wc = float(toksum['ctl'][tid][j]) / nc
            if wb < 0.01:
                continue
            tau = wc if wc > 0 else 1e-6
            rows_e.append({'tok': tok.decode([tid]), 'id': tid,
                           'bad': round(wb, 4), 'ctl': round(wc, 4),
                           'ratio': round(wb / tau, 2)})
        rows_e.sort(key=lambda x: -x['ratio'])
        enrichment[str(u)] = rows_e[:15]
    a2p_hit = [x for x in enrichment[str(s767)] if x['ratio'] >= 3.0]

    # ---- A3 强激活覆盖（按 d_ff 排名的 top-1%）----
    thr = max(1, d_ff // 100)
    badr = sub('bad')
    hit = [r['qi'] for r in badr if any(r['per_unit'][u]['rank_in_layer'] <= thr for u in core3)]
    hit767 = [r['qi'] for r in badr if r['per_unit'][s767]['rank_in_layer'] <= thr]
    a3 = {'threshold_rank': thr, 'n_bad': len(badr),
          'core3_hit': sorted(hit), 's767_hit': sorted(hit767),
          'core3_rank_median': float(np.median([r['per_unit'][u]['rank_in_layer']
                                                for r in badr for u in core3]))}

    # ---- A4 描述：坏题最强 token 是否集中 ----
    tops = [r['per_unit'][s767]['argmax_tok'].strip() for r in badr]
    a4 = {'n_positions': len(tops), 'distinct': len(set(tops)),
          'top': sorted(((tops.count(x), x) for x in set(tops)), reverse=True)[:8]}

    v = {'A1_separable': bool(a1[str(s767)]['auc'] >= 0.75),
         'A2p_token_specific_exploratory': bool(len(a2p_hit) >= 1),
         'A3_strong_activation': bool(len(hit) >= 6)}
    if v['A1_separable'] and v['A2p_token_specific_exploratory']:
        reading = ('767 有可能是可读的 token 身份：per-question max|a| AUC %.3f，'
                   '且坏题上存在 %d 个单位质量 ≥3× 的富集 token（探索性判据）。'
                   % (a1[str(s767)]['auc'], len(a2p_hit)))
    elif not v['A3_strong_activation']:
        reading = ('坏题上这 3 个单元并不特别火（767 只落进 %d/%d 道题的 top-1%%，中位排名 %d）⇒ '
                   '打坏不是因为「被额外激活」，而是它们的输出在这些题上不可替代。'
                   % (len(hit767), len(badr), int(a3['core3_rank_median'])))
    else:
        reading = '部分成立：%s' % json.dumps(v, ensure_ascii=False)

    out = {'schema': 'mech_core_act.v1', 'model_name': model_name, 'block': B,
           'units': units, 's767': s767, 'core3': core3,
           'n_questions': len(recs), 'damaged_qi': sorted(damaged),
           'A1_separability': a1, 'A1_ex73': a1_ex73,
           'A2_unit_vocab': a2, 'A2_enrichment_bad_vs_ctl': enrichment,
           'A3_coverage': a3, 'A4_descriptive': a4,
           'verdict': v, 'reading': reading,
           'note': ('捕获 = down_proj 的输入 = 消融时置零的那份 SwiGLU 输出；prefill 不生成；'
                    'token 聚合按题等权；A4 为描述项不作判据（原 ≤5 硬阈值已撤）。'),
           'records': recs, 'secs': round(time.time() - t_all, 1)}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8', newline='')

    print('\nA1 AUC（坏 vs 对照）: %s' % json.dumps({k: x['auc'] for k, x in a1.items()}, ensure_ascii=False))
    print('   去 qi=73:          %s' % json.dumps({k: x['auc'] for k, x in a1_ex73.items()}, ensure_ascii=False))
    for u in units:
        print('A2 %-5d 坏题 top-8: %s'
              % (u, ', '.join('%s(%.3f)' % (x['tok'], x['w']) for x in a2[str(u)]['bad']['top'][:8])))
        print('     对照 top-8: %s' % ', '.join('%s(%.3f)' % (x['tok'], x['w'])
                                            for x in a2[str(u)]['ctl']['top'][:8]))
        print('     top-20 覆盖 坏 %.1f%% / 对照 %.1f%%'
              % (100 * a2[str(u)]['bad']['top_share'], 100 * a2[str(u)]['ctl']['top_share']))
    print('A2\' 富集（坏/对照 ≥3×）单元 767 的前 8: %s'
          % ', '.join('%s(%.1fx)' % (x['tok'], x['ratio']) for x in a2p_hit[:8]))
    print('A3 落进 top-1%%: core3 %d/%d、767 单独 %d/%d（阈排名 ≤%d，中位排名 %d）'
          % (len(hit), len(badr), len(hit767), len(badr), thr, int(a3['core3_rank_median'])))
    print('A4 坏题最强 token: %d 位置 / %d 种 %s' % (a4['n_positions'], a4['distinct'], a4['top'][:5]))
    print('\n判据 %s' % json.dumps(v, ensure_ascii=False))
    print('读法：%s' % reading)
    print('written %s' % OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
