# -*- coding: utf-8 -*-
"""H21 因果闭合：767 消融后，输出到底是不是「先换行再答」，答案是不是跟着次生变化。

到这里已有三段证据，缺的是一个行为层面的闭环：
  H18  logit 层   答案位置把换行族抬高 9–10.7，字母只动 ≤0.12
  H19  前缀固定   9/9 坏题的字母 argmax 不变（但读到饱和 p≈1.0）
  H20  前缀扫描   25/50/75/100% 四档前缀上字母间隔位移中位 0.03（max 0.12）⇒ 无内容效应

如果 767 是格式单元，那么在**自由生成**（它真实作用的场景）里应该看到：
  ① 消融后首 token 明显更常是换行/空白；
  ② 答案改变的那些题，改变发生在「换行之后重写的一段」里，而不是在原来那一步判断上。

本探针直接看输出开头与重复度，不猜。

设计
----
  题：s767 打坏的 11 道 + 改答但未打坏的若干道（合计 ≤16），base 全对
  条件：{base, s767 消融}
  记录：前 3 个 token、前 40 字符、是否含「答案：」、是否出现 3 次以上重复片段、
        最终解析出的字母

预注册判据
----------
  R1 首 token 变了   767 消融后首 token 是空白/换行的题 ≥5 道（base 侧应 ≤1）
  R2 格式崩了       消融后无「答案：」行或含重复片段的题 ≥5 道
  R3 答案跟着变      格式崩的那些题里，答案改变的比例 ≥ 1/2

判读
----
R1+R2+R3 成立 → 闭环：767 的作用是「维持答题格式」，格式崩坏之后答案才跟着变。
               H12–H17 的「承载核」应正名为**格式承载单元**，
               并且「一个单元决定 10 道题的对错」这句话要改成
               「一个单元决定输出格式，格式崩坏时带走 10 道题」。
R1 不成立     → 格式效应没有兑现到行为层，H18 的位移只能读成潜在倾向，
               那么答案改变的机制仍未解释，需另找。

用法
----
  .venv/Scripts/python.exe -X utf8 tools/mech_head_tokens.py
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import mech_pair_sign as msign
from server import lang_atlas_service as latlas
from server import mech_pair_service as mpair

B = int(os.environ.get('B', '34'))
MAXNEW = int(os.environ.get('MAXNEW', '260'))
SEED = 20261101
OUT = Path('tempdata') / f'head_tokens_L{B}.json'
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


def max_repeat(txt: str, n: int = 8):
    """最长重复片段出现次数（n-gram），>2 视为退化。"""
    if len(txt) < n:
        return 0
    c = Counter(txt[i:i + n] for i in range(len(txt) - n + 1))
    return max(c.values())


def main() -> int:
    t_all = time.time()
    model, tok, device, _meta, model_name = mpair._wait_ready()
    one = json.loads(REF_ONE.read_text(encoding='utf-8'))
    s767 = int(one['s767'])
    dmg = [int(q) for q in one['segment_A_full']['damaged_qi']]
    core = json.loads(REF_CORE.read_text(encoding='utf-8'))
    R = {int(r['qi']): r for r in core['records']}
    A = {int(r['qi']): r for r in one['records_A']}
    # 改答但未打坏的题（答案变了但变对了 / 变成另一个错答案）
    changed = [q for q in A if A[q]['base'] == R[q]['gold'] and A[q]['s767'] is not None
               and A[q]['s767'] != R[q]['gold'] and q not in set(dmg)]
    qs = sorted(set(dmg) | set(changed))[:16]
    rows = load_rows()
    d_ff = 9728
    print('题 %d（打坏 %d，其中改答未打坏另有 %d）| 主条件 %d'
          % (len(qs), len(dmg), len(changed), s767), flush=True)

    def run(prompt, units=None):
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
            ids = o[0][enc['input_ids'].shape[1]:].tolist()
            txt = tok.decode(ids, skip_special_tokens=True)
            return {'txt': txt, 'first3': [tok.decode([i]) for i in ids[:3]],
                    'head40': txt[:40], 'has_ans': '答案' in txt,
                    'max_ngram_rep': max_repeat(txt), 'pred': msign.parse(txt)}
        finally:
            for h in hs:
                h.remove()

    recs = []
    t0 = time.time()
    for k, qi in enumerate(qs):
        prompt = msign.build(rows[qi])
        b = run(prompt)
        a = run(prompt, [s767])
        recs.append({'qi': qi, 'gold': rows[qi]['answer'], 'group':
                     'bad' if qi in set(dmg) else 'changed_not_broken',
                     'base': b, 's767': a})
        print('  %d/%d qi=%-3d gold=%s | base 首=%s %s→%s | 767 首=%s %s→%s'
              % (k + 1, len(qs), qi, rows[qi]['answer'],
                 b['first3'][0].encode('unicode_escape').decode()[:12], b['has_ans'], b['pred'],
                 a['first3'][0].encode('unicode_escape').decode()[:12], a['has_ans'], a['pred']),
              flush=True)

    def ws(x):
        return (x or '').strip() == '' or x in ('\n', '\r\n', '\r')

    n_ws_base = sum(1 for r in recs if ws(r['base']['first3'][0]))
    n_ws_abl = sum(1 for r in recs if ws(r['s767']['first3'][0]))
    n_fmt_abl = sum(1 for r in recs
                    if (not r['s767']['has_ans']) or r['s767']['max_ngram_rep'] >= 3)
    fmt_broken = [r for r in recs if (not r['s767']['has_ans']) or r['s767']['max_ngram_rep'] >= 3]
    ans_changed = [r for r in fmt_broken if r['s767']['pred'] != r['base']['pred']]
    v = {'R1_first_tok_changed': bool(n_ws_abl >= 5 and n_ws_base <= 1),
         'R2_format_broken': bool(n_fmt_abl >= 5),
         'R3_answer_follows': bool(len(fmt_broken) and len(ans_changed) / len(fmt_broken) >= 0.5)}
    if v['R1_first_tok_changed'] and v['R2_format_broken'] and v['R3_answer_follows']:
        reading = ('闭环成立：767 消融后 %d/%d 题的首 token 变成空白/换行（base 侧 %d），'
                   '%d 题出现格式崩坏（无「答案」行或重复片段），其中 %d 题的答案随之改变 ⇒ '
                   '767 维持的是答题格式，答案变化是格式崩坏的次生结果。'
                   % (n_ws_abl, len(recs), n_ws_base, n_fmt_abl, len(ans_changed)))
    elif not v['R1_first_tok_changed']:
        reading = ('首 token 变空白的只有 %d/%d（base %d）⇒ H18 的换行位移没有兑现到行为层，'
                   '本轮不能把 767 定性为格式单元；答案改变的机制仍未解释。'
                   % (n_ws_abl, len(recs), n_ws_base))
    else:
        reading = '部分成立：%s' % json.dumps(v, ensure_ascii=False)

    out = {'schema': 'mech_head_tokens.v1', 'model_name': model_name, 'block': B,
           's767': s767, 'questions': qs, 'verdict': v, 'reading': reading,
           'agg': {'n': len(recs), 'ws_base': n_ws_base, 'ws_ablated': n_ws_abl,
                   'fmt_broken': n_fmt_abl, 'ans_changed_among_broken': len(ans_changed),
                   'broken_qi': [r['qi'] for r in fmt_broken],
                   'changed_qi': [r['qi'] for r in ans_changed]},
           'note': ('首 token 的空白判定用 .strip()==""；重复度 = 8-gram 最高出现次数，≥3 视为退化；'
                    'pred 用 msign.parse，与 H17 同口径。'),
           'records': recs, 'secs': round(time.time() - t_all, 1)}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8', newline='')

    print('\n首 token 空白/换行: base %d/%d → 767 %d/%d'
          % (n_ws_base, len(recs), n_ws_abl, len(recs)))
    print('格式崩坏: %d 道 %s' % (n_fmt_abl, [r['qi'] for r in fmt_broken]))
    print('其中答案改变: %d 道 %s' % (len(ans_changed), [r['qi'] for r in ans_changed]))
    print('\n判据 %s' % json.dumps(v, ensure_ascii=False))
    print('读法：%s' % reading)
    print('written %s' % OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
