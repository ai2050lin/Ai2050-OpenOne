# -*- coding: utf-8 -*-
"""H20 前缀长度扫描：767 的影响是「推理过程」还是「格式」——把饱和读出拆开。

H19 的问题
----------
在 base 自己的完整前缀上做 teacher-forcing，767 消融在 9/9 道坏题上都不改字母 argmax，
KL 中位数 0.000000。**但这个读出是饱和的**：7/9 道题的字母概率 p=1.000。
一旦分析写全，答案几乎是强制的，任何单元的消融都改不动它 ——
所以 H19 只能排除「767 在答案已几乎确定时还改答案」，不能证明「767 不携带内容」。
这个 caveat 必须写进结论，否则就是又一次把弱检验读成强结论。

本轮的判别思路
--------------
把前缀**截短**，让答案重新变得不确定，再看 767 的影响在哪里出现：

  前缀长度 L ∈ {25%, 50%, 75%, 100%} 的 base 分析文本（在 token 边界切）
  每个 L 上做 {base, 767 消融} 两次前向，取 A/B/C/D 的分布

预注册判据
----------
  S1 早段就有影响   L=25% 或 50% 上，767 消融改变字母 argmax 或使 gold−argmax 的间隔变化 ≥1.0
  S2 晚段无影响     L=100%（H19 的饱和点）上不变
  S3 有剂量方向     随 L 增大，767 的影响单调减弱（早段大、晚段小）

判读
----
S1 成立 → 767 参与**推理过程**（在答案未定时影响判断），只是当分析写全后不再需要
          ⇒ 「格式单元」这个命名要改成「轨迹/推理单元」。
S1 不成立 → 连答案未定时 767 也不改判断 ⇒ 它对答案的作用完全经由生成轨迹，
          是干净的格式/风格单元。这条比 H19 强得多，因为它不依赖饱和读出。

用法
----
  .venv/Scripts/python.exe -X utf8 tools/mech_prefix_scan.py
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
MAXNEW = int(os.environ.get('MAXNEW', '260'))
SEED = 20261101
FRACS = [0.25, 0.5, 0.75, 1.0]
OUT = Path('tempdata') / f'prefix_scan_L{B}.json'
REF_ONE = Path('tempdata') / f'core_one_L{B}.json'
REF_CORE = Path('tempdata') / f'core_L{B}.json'


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

    one = json.loads(REF_ONE.read_text(encoding='utf-8'))
    s767 = int(one['s767'])
    dmg = [int(q) for q in one['segment_A_full']['damaged_qi']]
    core = json.loads(REF_CORE.read_text(encoding='utf-8'))
    R = {int(r['qi']): r for r in core['records']}
    rows = load_rows()
    d_ff = 9728
    ids = [tok.convert_tokens_to_ids(L) for L in 'ABCD']
    print('坏题 %d 道 | 主条件 %d | 前缀比例 %s' % (len(dmg), s767, FRACS), flush=True)

    def gen(prompt):
        enc = tok(prompt, return_tensors='pt').to(device)
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=MAXNEW, do_sample=False,
                               pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)

    def fwd(prompt, units=None):
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
            return o.logits[0, -1].float().cpu().numpy()
        finally:
            for h in hs:
                h.remove()

    def probs(L):
        v = np.array([float(L[i]) for i in ids])
        e = np.exp(v - v.max())
        p = e / e.sum()
        return {c: float(p[k]) for k, c in enumerate('ABCD')}, v

    recs = []
    t0 = time.time()
    for k, qi in enumerate(dmg):
        prompt = msign.build(rows[qi])
        full = gen(prompt)
        pos = full.rfind('答案')
        body = full if pos < 0 else full[:pos]
        toks = tok(body)['input_ids']
        row = {'qi': qi, 'gold': rows[qi]['answer'], 'subject': rows[qi]['subject'],
               'body_len_tok': len(toks), 'body_head': body[:40].replace('\n', '\\n'),
               'levels': {}}
        for frac in FRACS:
            m = max(4, int(round(len(toks) * frac)))
            piece = tok.decode(toks[:m])
            pfx = prompt + piece
            pb, vb = probs(fwd(pfx))
            p2, v2 = probs(fwd(pfx, [s767]))
            g = rows[qi]['answer']
            row['levels'][str(frac)] = {
                'n_tok': m,
                'base_argmax': max(pb, key=pb.get), 's767_argmax': max(p2, key=p2.get),
                'p_base': {c: round(pb[c], 4) for c in 'ABCD'},
                'p_s767': {c: round(p2[c], 4) for c in 'ABCD'},
                'argmax_changed': max(pb, key=pb.get) != max(p2, key=p2.get),
                'gold_gap_base': round(float(vb['ABCD'.index(g)] - max(
                    vb[i] for i in range(4) if 'ABCD'[i] != g)), 4),
                'gold_gap_s767': round(float(v2['ABCD'.index(g)] - max(
                    v2[i] for i in range(4) if 'ABCD'[i] != g)), 4),
                'margin_shift': round(float(
                    (vb['ABCD'.index(g)] - max(vb[i] for i in range(4) if 'ABCD'[i] != g))
                    - (v2['ABCD'.index(g)] - max(v2[i] for i in range(4) if 'ABCD'[i] != g))), 4),
            }
        recs.append(row)
        s = ' | '.join('%s:%s→%s(gap %+.2f→%+.2f)' % (f, v['base_argmax'], v['s767_argmax'],
                                                     v['gold_gap_base'], v['gold_gap_s767'])
                       for f, v in row['levels'].items())
        print('  %d/%d qi=%-3d gold=%s  %s' % (k + 1, len(dmg), qi, row['gold'], s), flush=True)

    def at(f):
        vals = [r['levels'][str(f)] for r in recs]
        return {'n': len(vals),
                'n_argmax_changed': sum(1 for x in vals if x['argmax_changed']),
                'median_abs_margin_shift': round(float(np.median([abs(x['margin_shift']) for x in vals])), 3),
                'max_abs_margin_shift': round(float(max(abs(x['margin_shift']) for x in vals)), 3),
                'n_gap_sign_flip': sum(1 for x in vals
                                       if (x['gold_gap_base'] > 0) != (x['gold_gap_s767'] > 0))}

    agg = {str(f): at(f) for f in FRACS}
    early = max(agg['0.25']['n_argmax_changed'], agg['0.5']['n_argmax_changed'])
    v = {'S1_early_effect': bool(early >= 2 or max(agg['0.25']['max_abs_margin_shift'],
                                                   agg['0.5']['max_abs_margin_shift']) >= 1.0),
         'S2_late_no_effect': bool(agg['1.0']['n_argmax_changed'] <= 1),
         'S3_dose_direction': bool(agg['0.25']['median_abs_margin_shift']
                                   >= agg['1.0']['median_abs_margin_shift'])}
    if v['S1_early_effect']:
        reading = ('767 参与推理过程：前缀只给 25–50%% 的分析时它就能改变判断'
                   '（argmax 变化 %d/%d 与 %d/%d，间隔最大位移 %.2f），'
                   '写全后不再改（%d/%d）⇒ 应称「轨迹/推理单元」，而不是纯格式单元。'
                   % (agg['0.25']['n_argmax_changed'], agg['0.25']['n'],
                      agg['0.5']['n_argmax_changed'], agg['0.5']['n'],
                      max(agg['0.25']['max_abs_margin_shift'], agg['0.5']['max_abs_margin_shift']),
                      agg['1.0']['n_argmax_changed'], agg['1.0']['n']))
    else:
        reading = ('连答案尚未确定时 767 也不改判断（25%% 上 argmax 变 %d/%d、间隔中位位移 %.3f）'
                   '⇒ 它对答案的作用完全经由生成轨迹，是干净的格式/风格单元。'
                   % (agg['0.25']['n_argmax_changed'], agg['0.25']['n'],
                      agg['0.25']['median_abs_margin_shift']))

    out = {'schema': 'mech_prefix_scan.v1', 'model_name': model_name, 'block': B,
           's767': s767, 'damaged_qi': dmg, 'fracs': FRACS,
           'agg': agg, 'verdict': v, 'reading': reading,
           'note': ('前缀 = prompt + base 输出里「答案」之前的文本按比例截断（token 边界切）；'
                    'gold_gap = gold 字母 logit 减最强的其他选项；margin_shift = base 的 gap 减消融后的 gap；'
                    '截断到 25% 时答案通常还没定，读出不再饱和，这是本轮相对 H19 的关键改进。'),
           'records': recs, 'secs': round(time.time() - t_all, 1)}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8', newline='')

    print('\n各前缀比例下的影响：')
    for f in FRACS:
        a = agg[str(f)]
        print('  %4.0f%%  argmax 变 %d/%d | 间隔中位位移 %.3f（max %.2f）| 符号翻转 %d'
              % (100 * f, a['n_argmax_changed'], a['n'], a['median_abs_margin_shift'],
                 a['max_abs_margin_shift'], a['n_gap_sign_flip']))
    print('\n判据 %s' % json.dumps(v, ensure_ascii=False))
    print('读法：%s' % reading)
    print('written %s' % OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
