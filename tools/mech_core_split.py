# -*- coding: utf-8 -*-
"""H15 承载核可再分吗：单点与成对。

H12 的自适应二分到「3 个单元」就停了（`while len(cur) > 3`），所以**只测过三个一起删**，
从没测过单独删 767、也没测过任何一对。H14-pre 已证三个写方向近乎正交（|cos| ≤ 0.029），
说明它们是三条独立通道，但「三条是否都必须」还是未测假设。

设计
----
7 个条件 × 17 题（H12 的 12 脆弱 + 5 对照），全部 ×0（同一个 down_proj hook）：

  s767  s1540  s35          单点
  p767_1540  p767_35  p1540_35   成对
  core3                     对照（应复现 7 道）
  base                      不重跑：复用 H13 的 core_L34.json（同一机型、同一天、贪心解码）

预注册判据（写死）
------------------
P1 全必需    每个单点 < 3 道 且 每个成对 < 3 道，而 core3 ≥ 5 道 ⇒ 三条通道缺一不可（合取依赖）
P2 单点即足  存在单点 ≥ 5 道 ⇒ 承载核是 1 个单元，H12 的 3 是终止条件造成的假象
P3 两点即足  存在成对 ≥ 5 道且所有单点 < 3 ⇒ 核是 2 个单元
P4 均不足    所有单点与成对都 < 3 且 core3 < 5 ⇒ core3 的 7 道不可复现（H12 稳定性存疑）

判读
----
P1 → 7 道题同时依赖三条独立写通道；「承载单元」不是一个，是「一个三元组」，
     破解要问的是「这三条通道分别携带什么、在哪一步被合并」。
P2/P3 → H12 的二分下限截断了答案，需把下限设到 1 重跑（成本另算）。

口径
----
· 题集固定为 H12 的 17 题（脆弱 12 + 对照 5），qi 与 base 取自 core_L34.json，逐题校验 base 一致。
· 损伤 = base 对→条件错；解析失败按打坏计但单独记数（R-MET-06）。
· 17 题的损伤上限是 7（H12 实际值），所以「≥5」已是很强的阈值；本判据不追求显著，
  只做**结构判别**（1 个 / 2 个 / 3 个都必需），显著性留给全量复核。

用法
----
  SMOKE=1 .venv/Scripts/python.exe -X utf8 tools/mech_core_split.py
  .venv/Scripts/python.exe -X utf8 tools/mech_core_split.py
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
SMOKE = os.environ.get('SMOKE') == '1'
MAXNEW = int(os.environ.get('MAXNEW', '260'))
SEED = 20261101
OUT = Path('tempdata') / f'core_split_L{B}.json'
REF_CORE = Path('tempdata') / f'core_L{B}.json'          # H13：base + 逐题 pred（首选）
REF_DOSE = Path('tempdata') / f'dose_L{B}.json'          # H8：base 兜底
REF_CARRIER = Path('tempdata') / f'carrier_L{B}.json'    # H12：core3 / 题集


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

    car = json.loads(REF_CARRIER.read_text(encoding='utf-8'))
    core3 = [int(u) for u in car['final_units']]
    qs = sorted(set(int(q) for q in car['fragile']) | set(int(q) for q in car['control']))
    if SMOKE:
        qs = qs[:2]
    assert len(core3) == 3, core3

    # ---- base：优先 H13，逐个 qi 校验，缺一即退出（不同源的 base 不可拼） ----
    base = {}
    src = None
    if REF_CORE.exists():
        d = json.loads(REF_CORE.read_text(encoding='utf-8'))
        base = {int(r['qi']): r['out']['base']['pred'] for r in d['records']}
        src = 'H13 core_L%d.json' % B
    if not base and REF_DOSE.exists():
        d = json.loads(REF_DOSE.read_text(encoding='utf-8'))
        base = {int(r['qi']): r['out']['base']['pred'] for r in d['records']}
        src = 'H8 dose_L%d.json' % B
    if not base:
        raise SystemExit('没有可用的 base 来源（需先跑 H13 或 H8）')

    rows = load_rows()
    miss = [q for q in qs if q not in base]
    assert not miss, 'base 缺这些题：%s（来源 %s）' % (miss, src)

    u1, u2, u3 = core3
    sets = {
        's%d' % u1: [u1], 's%d' % u2: [u2], 's%d' % u3: [u3],
        'p%d_%d' % (u1, u2): [u1, u2], 'p%d_%d' % (u1, u3): [u1, u3],
        'p%d_%d' % (u2, u3): [u2, u3],
        'core3': core3,
    }
    conds = list(sets)
    d_ff = np.load('tempdata/unit_attr_L%d.npz' % B)['attr_sum'].size
    masks = {nm: np.isin(np.arange(d_ff), np.array(us)) for nm, us in sets.items()}
    print('ready %.0fs | base 来源 %s | 题 %d 道（脆弱 %d）| core3 %s'
          % (time.time() - t_all, src, len(qs),
             sum(1 for q in qs if q in set(int(x) for x in car['fragile'])), core3), flush=True)

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
            return msign.parse(tok.decode(o[0][enc['input_ids'].shape[1]:], skip_special_tokens=True))
        finally:
            for h in hs:
                h.remove()

    prompts = {q: msign.build(rows[q]) for q in qs}
    gold = {q: rows[q]['answer'] for q in qs}
    # 只用 base 答对的题（与 H12 同口径）
    qs = [q for q in qs if base[q] == gold[q]]
    print('  base 对的题 %d 道' % len(qs), flush=True)
    if not qs:
        raise SystemExit('题集被 base 过滤空了，检查 base 来源是否与题序同源')

    res, dmg = {}, {}
    t0 = time.time()
    for nm in conds:
        bad, nfail = [], 0
        for q in qs:
            p = gen(prompts[q], masks[nm])
            if p is None:
                nfail += 1
                bad.append(q)
            elif p != gold[q]:
                bad.append(q)
        dmg[nm] = set(bad)
        res[nm] = {'n_units': len(sets[nm]), 'units': sets[nm], 'damaged': sorted(bad),
                   'n_damaged': len(bad), 'parse_fail': nfail}
        print('  %-12s 单元%d  打坏 %d/%d  解析失败 %d  %.0fs'
              % (nm, len(sets[nm]), len(bad), len(qs), nfail, time.time() - t0), flush=True)

    singles = [nm for nm in conds if nm.startswith('s')]
    pairs = [nm for nm in conds if nm.startswith('p')]
    ms_ = max(res[n]['n_damaged'] for n in singles)
    mp = max(res[n]['n_damaged'] for n in pairs)
    mc = res['core3']['n_damaged']
    v = {
        'P1_all_required': bool(ms_ < 3 and mp < 3 and mc >= 5),
        'P2_single_sufficient': bool(ms_ >= 5),
        'P3_pair_sufficient': bool(mp >= 5 and ms_ < 3),
        'P4_core3_not_replicated': bool(ms_ < 3 and mp < 3 and mc < 5),
        'max_single': ms_, 'max_pair': mp, 'core3': mc,
    }
    if v['P1_all_required']:
        reading = ('三条通道缺一不可：单点最多 %d 道、成对最多 %d 道，而三个一起是 %d 道 ⇒ '
                   '合取依赖——7 道题同时需要三条几何独立的写通道。'
                   '破解的问题因此变成「这三条通道各携带什么、在哪一步被合并」。' % (ms_, mp, mc))
    elif v['P2_single_sufficient']:
        reading = ('承载核其实是 1 个单元（单点最多 %d 道）⇒ H12 的「3」是二分终止条件造成的假象，'
                   '承载核应改写为单点 %s。' % (ms_, [n for n in singles if res[n]['n_damaged'] == ms_]))
    elif v['P3_pair_sufficient']:
        reading = ('核是 2 个单元（成对最多 %d 道且单点都 <3）⇒ 需把 H12 的下限设到 1 重跑。' % mp)
    else:
        reading = ('core3 的损伤没有复现（%d 道 <5），P1/P2/P3 都不成立 ⇒ H12 的 7 道不稳定，'
                   '必须回查 H12 的逐轮记录后才能继续。' % mc)

    out = {'schema': 'mech_core_split.v1', 'model_name': model_name, 'block': B,
           'max_new_tokens': MAXNEW, 'base_source': src,
           'core3': core3, 'sets': sets, 'questions': qs,
           'n_questions': len(qs), 'results': res,
           'overlap_vs_core3': {nm: sorted(dmg[nm] & dmg['core3']) for nm in conds},
           'verdict': v, 'reading': reading,
           'note': '单点/成对/三点共 7 条件；题集与 base 同源于 H12/H13，逐题校验后只用 base 对的题。',
           'secs': round(time.time() - t_all, 1)}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8', newline='')

    print('\n判据 %s' % json.dumps(v, ensure_ascii=False))
    for nm in conds:
        print('  %-12s %d 道  %s' % (nm, res[nm]['n_damaged'], res[nm]['damaged']))
    print('读法：%s' % reading)
    print('written %s' % OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
