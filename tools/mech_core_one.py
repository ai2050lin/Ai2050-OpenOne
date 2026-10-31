# -*- coding: utf-8 -*-
"""H17 单个单元是不是就够：s767 全量 + 随机单点对照。

H15 的意外结局（17 题）：单点 767 打坏 9/17，比 core3 三个一起（H12 记 7/17）还多，
成对 8/17 也没有更好。若这个成立，H12/H13 的「3 个单元承载核」要继续缩到**1 个单元**，
而 H12 的「3」纯粹是二分终止条件（`while len(cur) > 3`）的下限，从没测过单点。

但 H15 的分母只有 17 道（12 脆弱 + 5 对照，脆弱是被富集过的），所以两件事必须补：

  1. s767 在全量 169 题上的真实损伤（对照 H13 的 core3 = 9/110、tailN = 12/110）
  2. 随机单点能不能也做到 —— 「1 个单元很特殊」这句话只有在随机单点做不到时才成立

设计
----
  A 段  s767 × 169 题（base 复用 core_L34.json，逐题校验）
  B 段  10 个随机单点 × 17 题（与 H15 同一批题、同一 base），随机种子写死

预注册判据
----------
O1 可复制    s767 在全量 110 道 base 对的题上打坏 ≥ 7 道（H15 在 17 题上给了 9）
O2 特异性    10 个随机单点在 17 题上打坏的最大值 ≤ 1，而 s767 = 9
O3 主导性    s767 的打坏题集**包含** H13 测到的 core3 那 9 道（若成立，core3 的作用全部由 767 承担）
O4 不劣       s767 全量打坏数 ≥ core3 的 9（三合一还不如一个 ⇒ 「三个都必需」被彻底否掉）

判读
----
O1+O2+O3 全成立 → 承载核 = **单个单元 767**。9 道题的正确性依赖 9728 个单元里的 1 个（0.01%），
                  且随机单点做不到 ⇒ 这是本线目前最强的特异性结论，也是编码破解的最小对象。
O2 不成立       → 任何单点都能打坏几道（删一个单元就够破坏），那「特异性」是题池噪声，回到池级解释。
O3 不成立       → 767 与另外两个分担不同题，核仍是多元的，H15 的 9 里有别的东西。

口径
----
· 解析失败按打坏计但单独记数（R-MET-06）；O1/O3 的「打坏」两种口径都会报（含/不含解析失败）。
· 随机单点从全部 9728 个里均匀抽（含 tailN/top40），与「随机 97」的对照逻辑一致，不挑干净的。
· base 只复用不重跑，逐题校验与 H13 一致。

用法
----
  SMOKE=1 .venv/Scripts/python.exe -X utf8 tools/mech_core_one.py
  .venv/Scripts/python.exe -X utf8 tools/mech_core_one.py      # A 段 22 min + B 段 22 min
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
NQ = int(os.environ.get('NQ', '169'))
NRAND = int(os.environ.get('NRAND', '10'))
MAXNEW = int(os.environ.get('MAXNEW', '260'))
SEED_Q = 20261101
SEED_R = 20261102           # 随机单点的种子，与题序种子分开，写死
OUT = Path('tempdata') / f'core_one_L{B}.json'
REF_CORE = Path('tempdata') / f'core_L{B}.json'
REF_CARRIER = Path('tempdata') / f'carrier_L{B}.json'


def load_rows():
    rows_all = [r for r in latlas._load_questions() if r.get('source')]
    cm = [r for r in rows_all if r['source'] == 'CMMLU test']
    by_sub = {}
    for r in cm:
        by_sub.setdefault(r['subject'], []).append(r)
    rng = np.random.default_rng(SEED_Q)
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
    core = json.loads(REF_CORE.read_text(encoding='utf-8'))
    R = {int(r['qi']): r for r in core['records']}
    base = {q: r['out']['base']['pred'] for q, r in R.items()}
    core3_dmg = set(int(x) for x in core['results']['core3']['damaged_qi'])
    core3_n = core['results']['core3']['damaged']
    rows = load_rows()
    d_ff = np.load('tempdata/unit_attr_L%d.npz' % B)['attr_sum'].size

    s767 = core3[0] if core3[0] == 767 else 767
    print('ready %.0fs | s767=%d | core3=%s（H13 打坏 %d 道）'
          % (time.time() - t_all, s767, core3, core3_n), flush=True)

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

    nq = 3 if SMOKE else NQ
    prompts = {i: msign.build(rows[i]) for i in range(min(NQ, nq))}

    # ---------- A 段：s767 全量 ----------
    recs = []
    t0 = time.time()
    for qi in range(nq):
        pred = gen(prompts[qi], [s767])
        recs.append({'qi': qi, 'subject': rows[qi]['subject'], 'gold': rows[qi]['answer'],
                     'base': base.get(qi), 's767': pred})
        if (qi + 1) % 10 == 0 or qi + 1 == nq:
            print('  A %d/%d %.0fs' % (qi + 1, nq, time.time() - t0), flush=True)

    valid = [r for r in recs if r['base'] is not None]
    base_ok = [r for r in valid if r['base'] == r['gold']]
    dmg_all = {r['qi'] for r in base_ok if r['s767'] != r['gold']}      # 含解析失败
    dmg_strict = {r['qi'] for r in base_ok if r['s767'] is not None and r['s767'] != r['gold']}
    n_parse = len(dmg_all - dmg_strict)
    A = {'n_valid': len(valid), 'n_base_ok': len(base_ok),
         'damaged_incl_parsefail': len(dmg_all), 'damaged_strict': len(dmg_strict),
         'n_parse_fail_in_damaged': n_parse,
         'rate': round(len(dmg_all) / len(base_ok), 4) if base_ok else None,
         'damaged_qi': sorted(dmg_all),
         'delta_pt': round(-100 * len(dmg_all) / len(base_ok), 2) if base_ok else None}
    print('  A 段: 打坏 %d/%d（严格 %d，解析失败 %d）'
          % (len(dmg_all), len(base_ok), len(dmg_strict), n_parse), flush=True)

    # ---------- B 段：随机单点对照（17 题） ----------
    qs17 = sorted(set(int(q) for q in car['fragile']) | set(int(q) for q in car['control']))
    qs17 = [q for q in qs17 if base.get(q) == rows[q]['answer']]
    if SMOKE:
        qs17 = qs17[:2]
    rng = np.random.default_rng(SEED_R)
    rand_units = [int(x) for x in rng.choice(d_ff, size=NRAND, replace=False)]
    print('  B 段: %d 个随机单点 × %d 题' % (len(rand_units), len(qs17)), flush=True)

    Bs, t0 = {}, time.time()
    for k, u in enumerate(rand_units):
        bad = []
        for q in qs17:
            p = gen(prompts[q], [u])
            if p is None or p != rows[q]['answer']:
                bad.append(q)
        Bs[u] = sorted(bad)
        print('  B %d/%d unit %-5d 打坏 %d/%d  %.0fs'
              % (k + 1, len(rand_units), u, len(bad), len(qs17), time.time() - t0), flush=True)

    # 与 s767 在**同一批 17 题**上的损伤对比（用 A 段的逐题结果截取，保证同口径）
    a17 = {r['qi'] for r in recs if r['qi'] in set(qs17)
           and r['base'] == r['gold'] and r['s767'] != r['gold']}
    rand_max = max((len(v) for v in Bs.values()), default=0)

    v = {'O1_replicable': bool(len(dmg_all) >= 7),
         'O2_specific': bool(rand_max <= 1 and len(a17) >= 5),
         'O3_dominant': bool(core3_dmg <= dmg_all),
         'O4_not_worse': bool(len(dmg_all) >= core3_n),
         's767_on_17q': len(a17), 'rand_max_on_17q': rand_max,
         'core3_full': core3_n, 's767_full': len(dmg_all)}
    if v['O1_replicable'] and v['O2_specific'] and v['O3_dominant']:
        reading = ('承载核 = 单个单元 %d。全量打坏 %d/%d（%.1f%%），覆盖 core3 的全部 %d 道，'
                   '而 10 个随机单点在 17 题上最多打坏 %d 道（%d 打坏 %d 道）。'
                   '⇒ 「三个都必需」被否掉，H12 的 3 是二分下限；编码破解的对象缩到 1 个单元。'
                   % (s767, len(dmg_all), len(base_ok), 100 * len(dmg_all) / max(1, len(base_ok)),
                      core3_n, rand_max, s767, len(a17)))
    elif not v['O2_specific']:
        reading = ('特异性不成立：随机单点在 17 题上最多也能打坏 %d 道（s767 = %d）⇒ '
                   '「删一个单元就够破坏这些题」不是 767 独有，特异性是题池噪声。'
                   % (rand_max, len(a17)))
    elif not v['O3_dominant']:
        reading = ('767 不覆盖 core3 的全部损伤：core3 打坏的 %s 里 767 漏了 %s ⇒ '
                   '另外两个单元分担不同题，核仍是多元的。'
                   % (sorted(core3_dmg), sorted(core3_dmg - dmg_all)))
    else:
        reading = '部分成立：%s' % json.dumps(v, ensure_ascii=False)

    out = {'schema': 'mech_core_one.v1', 'model_name': model_name, 'block': B,
           'max_new_tokens': MAXNEW, 's767': s767, 'core3': core3,
           'core3_damaged_H13': sorted(core3_dmg),
           'segment_A_full': A, 'segment_B_randoms': {'seed': SEED_R, 'units': rand_units,
                                                      'damaged': {str(k): x for k, x in Bs.items()},
                                                      'max': rand_max, 'n_questions': len(qs17)},
           'questions_17': qs17, 'verdict': v, 'reading': reading,
           'note': ('A 段全量单点，B 段随机单点对照（同一批 17 题，与 H15 同源）；'
                    'base 复用 core_L34.json 逐题校验；解析失败按打坏计但单独记数（R-MET-06）。'),
           'records_A': recs, 'secs': round(time.time() - t_all, 1)}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8', newline='')

    print('\nA 段（全量）: 打坏 %d/%d，严格 %d，Δ %s pt'
          % (len(dmg_all), len(base_ok), len(dmg_strict), A['delta_pt']))
    print('   漏掉 core3 的题: %s' % sorted(core3_dmg - dmg_all))
    print('B 段（17 题随机单点）: 最多 %d 道；%d 打坏 %d 道' % (rand_max, s767, len(a17)))
    print('\n判据 %s' % json.dumps(v, ensure_ascii=False))
    print('读法：%s' % reading)
    print('written %s' % OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
