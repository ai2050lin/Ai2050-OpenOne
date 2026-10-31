# -*- coding: utf-8 -*-
"""H12 承载单元定位：tailN 的 23 个单元里，到底是几个在扛那 12 道脆弱题。

前面已经确定的事实（H5/H6/H8/H10/H11）：

  · L34 top-1% 整池（97）打坏 14/110 道 base 对的题；随机 97 只打坏 1 道 → 特异性 ~14×。
  · 池内按 |Δ| 排序没有意义：top-40（占归因量 76%）只打坏 4 道，
    而 tailN 的 23 个「小队负号单元」打坏 11 道 —— 与整池 97 个相当。
  · 被打坏的是**同一小批题**（qi=73、5 被全部 8 个条件打坏），不是各子集打坏各的。
  · 逐题归因（决策点）区分不了脆弱题与对照题（H11 三条判据全不成立）。

那么「池内到底谁在承载」就剩一个可测的问题：把 tailN 的 23 个单元继续拆，
损伤是**跟着具体单元走**，还是**任何达到某个覆盖量的子集都行**。

设计
----
自适应二分，只用 12 道脆弱题 + 6 道对照题（每题生成一次约 8s，cost 极低）：

  round 0   当前集合 = tailN（23 个），记录损伤题数
  round r   把当前集合按 |Δ| 降序对半切成 A / B，各测一次
            子集被接受的条件：损伤题数 ≥ max(3, 0.5 × 父集合损伤题数)
            若两个子集都满足 → 说明不是具体单元而是覆盖量，直接停并记 `coverage_like`
            若只有一个满足 → 进入它，继续
            若都不满足 → 说明这 23 个是协同的，记 `synergistic_at_<n>`
  终止      集合 ≤ 3 个单元，或已判 `coverage_like` / `synergistic`

base 复用 `dose_L34.json`（贪心解码，H5/H6/H8/H10 四轮逐题一致 169/169）。

预注册判据
----------
S1 定位成功    结束时集合 ≤ 3 且损伤题数 ≥ 3（即几个单元就能扛住 12 道题里的 ≥3 道）
S2 覆盖型      任何一半都能维持 ≥ 半量损伤 ⇒ 不是具体单元，是覆盖量
S3 协同型      两半都掉到 < 半量 ⇒ 需要这批单元全体，不存在小承载核

判读
----
S1  → 拿到「承载核」，可以开始问这几个单元编的是什么、读出到哪（真正的编码破解起点）
S2  → 「必要性」是池级的量，池内不可再分；与 R-MLP-18 一起说明单元级排序不可用
S3  → 承载是分布协同的，单点或小集合都无效，只有整体有效

用法
----
  SMOKE=1 .venv/Scripts/python.exe -X utf8 tools/mech_carrier.py    # 2 题 2 条件
  .venv/Scripts/python.exe -X utf8 tools/mech_carrier.py             # 12+6 题，自适应 ≤10 轮
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
SMOKE = os.environ.get('SMOKE') == '1'
MAXNEW = int(os.environ.get('MAXNEW', '260'))
MAX_ROUNDS = int(os.environ.get('MAX_ROUNDS', '8'))
OUT = Path('tempdata') / f'carrier_L{B}.json'
REF_DOSE = Path('tempdata') / f'dose_L{B}.json'
REF_ATTR = Path('tempdata') / f'unit_attr_L{B}.npz'

FRAGILE = [73, 5, 94, 78, 29, 125, 41, 25, 68, 97, 55, 21]
CONTROL = [3, 17, 33, 48, 62, 101]     # 由 H11 的对照池等距抽，base 对、未被打坏


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

    rows = load_rows()
    d8 = json.loads(REF_DOSE.read_text(encoding='utf-8'))
    base = {r['qi']: r['out']['base']['pred'] for r in d8['records']}
    gold = {r['qi']: r['gold'] for r in d8['records']}

    qs = FRAGILE + CONTROL
    if SMOKE:
        qs = qs[:2]
    # 只保留 base 答对的题（打坏 = base 对 → 条件错）
    qs = [q for q in qs if base.get(q) is not None and base[q] == gold.get(q)]
    n_frag = sum(1 for q in qs if q in FRAGILE)
    print('ready %.0fs | 题 %d 道（脆弱 %d）' % (time.time() - t_all, len(qs), n_frag), flush=True)

    a = np.load(REF_ATTR)['attr_sum'].astype(np.float64)
    d_ff = a.size
    order = np.argsort(-np.abs(a))
    top = order[:97]
    pos_all = top[a[top] > 0]
    neg_all = top[a[top] < 0]
    m = int(min(pos_all.size, neg_all.size))
    pos45 = pos_all[np.argsort(-np.abs(a[pos_all]))][:m]
    neg45 = neg_all[np.argsort(-np.abs(a[neg_all]))][:m]
    head40 = set(int(u) for u in order[:40])
    tailN = [int(u) for u in np.setdiff1d(neg45, np.array(sorted(head40)))]
    tailN.sort(key=lambda u: -abs(a[u]))
    print('  tailN 起始 %d 个，Σ|Δ|=%.2f' % (len(tailN), sum(abs(a[u]) for u in tailN)), flush=True)

    def gen(prompt, units):
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
                o = model.generate(**enc, max_new_tokens=MAXNEW, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
            return msign.parse(tok.decode(o[0][enc['input_ids'].shape[1]:], skip_special_tokens=True))
        finally:
            for h in hs:
                h.remove()

    prompts = {q: msign.build(rows[q]) for q in qs}

    def damage(units, tag=''):
        t0 = time.time()
        bad, n_parse = [], 0
        for q in qs:
            p = gen(prompts[q], units)
            if p is None:
                n_parse += 1
                bad.append(q)          # 解析失败按打坏计（与行为口径一致），但单独记数（R-MET-06）
            elif p != gold[q]:
                bad.append(q)
        print('    %-22s 单元%2d  打坏 %2d/%d (脆弱 %2d/%d)  解析失败 %d  %.0fs'
              % (tag or '-', len(units), len(bad), len(qs),
                 sum(1 for q in bad if q in FRAGILE), n_frag, n_parse, time.time() - t0), flush=True)
        return bad

    rounds = []
    cur = list(tailN)
    dmg = damage(cur, 'round0 tailN(23)')
    rounds.append({'round': 0, 'tag': 'tailN', 'size': len(cur), 'damaged': sorted(dmg),
                   'n_damaged': len(dmg), 'units': cur})
    verdict = None
    r = 0
    while len(cur) > 3 and r < MAX_ROUNDS:
        r += 1
        half = len(cur) // 2
        A, Bs = cur[:half], cur[half:]
        dA = damage(A, f'round{r} A({len(A)})')
        dB = damage(Bs, f'round{r} B({len(Bs)})')
        rounds.append({'round': r, 'tag': 'A', 'size': len(A), 'damaged': sorted(dA),
                       'n_damaged': len(dA), 'units': A})
        rounds.append({'round': r, 'tag': 'B', 'size': len(Bs), 'damaged': sorted(dB),
                       'n_damaged': len(dB), 'units': Bs})
        keep = max(3, int(0.5 * len(dmg)))
        okA, okB = len(dA) >= keep, len(dB) >= keep
        if okA and okB:
            verdict = 'coverage_like'
            print('  → 两半都维持 ≥%d 道损伤：覆盖型，不是具体单元' % keep, flush=True)
            break
        if okA and not okB:
            cur, dmg = A, dA
        elif okB and not okA:
            cur, dmg = Bs, dB
        else:
            verdict = 'synergistic_at_%d' % len(cur)
            print('  → 两半都掉到 <%d 道：协同型，需要这批单元全体' % keep, flush=True)
            break
    if verdict is None:
        verdict = 'localized_%d' % len(cur)

    v = {'S1_localized': bool(len(cur) <= 3 and len(dmg) >= 3),
         'S2_coverage_like': verdict == 'coverage_like',
         'S3_synergistic': str(verdict).startswith('synergistic'),
         'final_size': len(cur), 'final_damaged': len(dmg), 'tag': verdict}

    if v['S1_localized']:
        reading = ('定位到承载核：%d 个单元即可打坏 12 道里的 %d 道（%s）。'
                   '可以开始查这几个单元编的是什么、读出到哪。'
                   % (len(cur), len(dmg), cur))
    elif v['S2_coverage_like']:
        reading = ('覆盖型：把 tailN 对半切，两半都还能维持过半损伤 ⇒ '
                   '不存在「小承载核」，池内任何够量的子集都行。'
                   '「必要性」是池级的量，池内不可再分。')
    elif v['S3_synergistic']:
        reading = ('协同型：两半单独都失效、合起来才有效（%s）⇒ 承载是分布协同的，'
                   '单点或小集合都无效。' % verdict)
    else:
        reading = '未收敛（%s），需要更多轮或更大题集。' % verdict

    out = {'schema': 'mech_carrier.v1', 'model_name': model_name, 'block': B,
           'questions': qs, 'fragile': FRAGILE, 'control': CONTROL,
           'start_set': tailN, 'rounds': rounds, 'verdict': v, 'reading': reading,
           'final_units': cur,
           'note': 'base 复用 dose_L34.json；解析失败按打坏计（与行为口径一致），另在 records 里可辨。',
           'secs': round(time.time() - t_all, 1)}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8', newline='')
    print('\n判据 %s' % json.dumps(v, ensure_ascii=False))
    print('读法：%s' % reading)
    print('written %s' % OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
