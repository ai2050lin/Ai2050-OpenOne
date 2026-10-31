# -*- coding: utf-8 -*-
"""H19 767 是「格式单元」还是「知识单元」：在 base 自己的前缀上做 teacher-forcing。

H18 给出了一个刺眼的位移
------------------------
在被 767 打坏的 11 道题的答案位置上，消融 767 把换行族 logit 抬高 9–10.7
（`\\n` / `\\r` / `\\r\\n`，11 道题推的是同一批 token，Jaccard 1.00），而四个选项字母
只动 ±0.06–0.12，小两个数量级；|Δlogit| 中位数是随机单点的 8 倍。

两种互斥解释：

  H-格式  767 是「别换行、接着答」的结构单元。消融后模型倾向另起一行 → 生成轨迹变了
          （推理过程跟着变）→ 在答案裕度薄的题上顺手答错。它不携带知识。
  H-内容  换行是最大的一项而已；767 的输出真的决定答案。

第一版设计（已作废，留档）
--------------------------
原方案是**强制接上「 答案：」前缀**再让模型只吐字母。跑到第 1 题就发现它不中性：
qi=5 上 base 带前缀给出的答案是 **C**，而 gold 与无前缀的 base 都是 **D**。
原因是 `分析：` 之后直接接 ` 答案：` 等于**把推理过程整段删掉**，模型于是在没有分析的情况下作答。
⇒ 「前缀条件」同时改了两个因素（格式 + 推理），不能用来分离格式与内容。**这一版当场废弃。**

本版设计
--------
**不动格式，只在 base 已经写好的前缀上问一次。**

  1. 让 base 正常贪心生成，拿到它自己的输出文本；
  2. 取 `prompt + 文本直到第一个「答案：」` 作为**固定前缀**（格式、推理、长度全部保持 base 原样）；
  3. 在这个前缀上，分别用 {base, 767 消融, 2 个随机单点} 做一次前向；
  4. 只看最后一个位置 A/B/C/D 的分布。

前缀完全一致 ⇒ 格式自由度被完全扣掉。此时字母分布若不变，说明 767 不携带答案信息，
它影响的是**生成轨迹**（先写什么、要不要换行），答案变化是轨迹变化的后果。

预注册判据
----------
  N1 字母不变    固定前缀下，767 消融的字母 argmax 与 base 一致 ≥9/11（坏题）
  N2 改动很小    767 对字母分布的影响 ≤ 随机单点的 2 倍（用 4 类分布的 KL 中位数比）
  N3 对照不动    8 道对照题的字母 argmax 全不变

判读
----
N1+N2 成立 → H-格式成立：767 决定的是「怎么写」而不是「写什么」。
              H12–H17 的「承载核」必须改名为「格式单元」，用答对/答错当因变量的
              单元消融实验存在把格式损伤误读成知识损伤的系统性风险。
N1 不成立   → H-内容成立：格式已固定时 767 仍然改写答案分布，换行只是顺带的副作用。
N2 不成立   → 767 的字母效应虽小但显著大于随机单点，读作「弱内容效应」。

用法
----
  .venv/Scripts/python.exe -X utf8 tools/mech_format_test.py
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
ANS = '答案：'
OUT = Path('tempdata') / f'format_test_L{B}.json'
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


def letter_probs(logits_row, ids):
    v = np.array([float(logits_row[i]) for i in ids], dtype=float)
    e = np.exp(v - v.max())
    p = e / e.sum()
    return {L: float(p[k]) for k, L in enumerate('ABCD')}, v


def kl4(p, q):
    p = np.array([p[L] for L in 'ABCD']) + 1e-12
    q = np.array([q[L] for L in 'ABCD']) + 1e-12
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def main() -> int:
    t_all = time.time()
    model, tok, device, _meta, model_name = mpair._wait_ready()

    one = json.loads(REF_ONE.read_text(encoding='utf-8'))
    s767 = int(one['s767'])
    dmg = [int(q) for q in one['segment_A_full']['damaged_qi']]
    rand_u = [int(u) for u in one['segment_B_randoms']['units'][:2]]
    core = json.loads(REF_CORE.read_text(encoding='utf-8'))
    R = {int(r['qi']): r for r in core['records']}
    base_ok = [q for q, r in R.items()
               if r['out']['base']['pred'] is not None and r['out']['base']['pred'] == r['gold']]
    ctl_pool = [q for q in base_ok if q not in set(dmg)]
    rng = np.random.default_rng(SEED + 7)
    ctl = sorted(int(x) for x in rng.choice(ctl_pool, size=8, replace=False))
    rows = load_rows()
    d_ff = 9728
    ids = [tok.convert_tokens_to_ids(L) for L in 'ABCD']   # 顺序必须是 A,B,C,D（letter_probs 按位置取）
    print('坏题 %d / 对照 %d | 主条件 %d | 随机对照 %s'
          % (len(dmg), len(ctl), s767, rand_u), flush=True)

    def gen(prompt, units=None):
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
            return tok.decode(o[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
        finally:
            for h in hs:
                h.remove()

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

    recs = []
    t0 = time.time()
    for k, qi in enumerate(dmg + ctl):
        grp = 'bad' if qi in set(dmg) else 'ctl'
        prompt = msign.build(rows[qi])
        full = gen(prompt)                      # base 自己写的文本
        pos = full.find(ANS)
        if pos < 0:
            print('  qi=%-3d 无「答案：」行，跳过（记 skip）' % qi, flush=True)
            recs.append({'qi': qi, 'group': grp, 'gold': rows[qi]['answer'],
                         'skipped': True, 'base_text': full[:120]})
            continue
        pref = prompt + full[:pos + len(ANS)]
        pb, vb = letter_probs(fwd(pref), ids)
        rec = {'qi': qi, 'group': grp, 'subject': rows[qi]['subject'],
               'gold': rows[qi]['answer'], 'skipped': False,
               'base_letter_gen': max(pb, key=pb.get),
               'conds': {'base': {'p': {L: round(pb[L], 5) for L in 'ABCD'},
                                  'argmax': max(pb, key=pb.get),
                                  'raw': {L: round(float(vb[i]), 4) for i, L in enumerate('ABCD')}}}}
        for name, units in [('s767', [s767])] + [('r%d' % u, [u]) for u in rand_u]:
            p2, v2 = letter_probs(fwd(pref, units), ids)
            rec['conds'][name] = {'p': {L: round(p2[L], 5) for L in 'ABCD'},
                                  'argmax': max(p2, key=p2.get),
                                  'raw': {L: round(float(v2[i]), 4) for i, L in enumerate('ABCD')},
                                  'kl_vs_base': round(kl4(pb, p2), 5),
                                  'dmax_abs': round(max(abs(v2[i] - vb[i]) for i in range(4)), 4)}
        recs.append(rec)
        c = rec['conds']
        print('  %d/%d qi=%-3d(%s) gold=%s | base %s(p=%.3f) | 767 %s(p=%.3f) KL=%.4f | 随机 %s/%s'
              % (k + 1, len(dmg) + len(ctl), qi, grp, rec['gold'],
                 c['base']['argmax'], c['base']['p'][c['base']['argmax']],
                 c['s767']['argmax'], c['s767']['p'][c['s767']['argmax']],
                 c['s767']['kl_vs_base'],
                 c['r%d' % rand_u[0]]['argmax'], c['r%d' % rand_u[1]]['argmax']), flush=True)

    ok = [r for r in recs if not r.get('skipped')]
    bad = [r for r in ok if r['group'] == 'bad']
    ctlr = [r for r in ok if r['group'] == 'ctl']

    def stat(sub):
        same = sum(1 for r in sub if r['conds']['s767']['argmax'] == r['conds']['base']['argmax'])
        kl767 = [r['conds']['s767']['kl_vs_base'] for r in sub]
        klr = [r['conds']['r%d' % u]['kl_vs_base'] for r in sub for u in rand_u]
        dmax = [r['conds']['s767']['dmax_abs'] for r in sub]
        dmaxr = [r['conds']['r%d' % u]['dmax_abs'] for r in sub for u in rand_u]
        return {'n': len(sub), 'same_argmax': same,
                'kl767_median': round(float(np.median(kl767)), 6) if kl767 else None,
                'kl_rand_median': round(float(np.median(klr)), 6) if klr else None,
                'kl_ratio': round(float(np.median(kl767) / (np.median(klr) + 1e-12)), 2) if klr else None,
                'dmax767_median': round(float(np.median(dmax)), 4) if dmax else None,
                'dmax_rand_median': round(float(np.median(dmaxr)), 4) if dmaxr else None,
                'base_argmax_eq_gold': sum(1 for r in sub if r['conds']['base']['argmax'] == r['gold'])}

    s_bad, s_ctl = stat(bad), stat(ctlr)
    v = {'N1_letter_unchanged': bool(s_bad['same_argmax'] >= 9),
         'N2_effect_vs_random': bool(s_bad['kl_ratio'] is not None and s_bad['kl_ratio'] <= 2.0),
         'N3_control_unchanged': bool(s_ctl['same_argmax'] == s_ctl['n'])}
    if v['N1_letter_unchanged'] and v['N2_effect_vs_random']:
        reading = ('H-格式成立：把 base 自己的前缀原样固定后，767 消融在 %d/%d 道坏题上不再改变字母 '
                   'argmax，字母分布的 KL 中位数只有随机单点的 %.2f 倍 ⇒ '
                   '767 决定的是「怎么写」不是「写什么」，之前的答错是生成轨迹改变后的后果。'
                   % (s_bad['same_argmax'], s_bad['n'], s_bad['kl_ratio']))
    elif not v['N1_letter_unchanged']:
        reading = ('H-内容成立：前缀固定后，767 消融仍改变 %d/%d 道坏题的字母 argmax ⇒ '
                   '它确实写入决定答案的信息，换行只是最大的那个副作用。'
                   % (s_bad['n'] - s_bad['same_argmax'], s_bad['n']))
    else:
        reading = ('弱内容效应：字母 argmax 基本不变（%d/%d），但扰动幅度是随机单点的 %.2f 倍 '
                   '（超过预注册的 2 倍）⇒ 767 对答案分布有可测但不改判的影响。'
                   % (s_bad['same_argmax'], s_bad['n'], s_bad['kl_ratio']))

    out = {'schema': 'mech_format_test.v2', 'model_name': model_name, 'block': B,
           's767': s767, 'random_units': rand_u, 'damaged_qi': dmg, 'control_qi': ctl,
           'agg_bad': s_bad, 'agg_ctl': s_ctl, 'verdict': v, 'reading': reading,
           'design_note': ('v1（强制「 答案：」前缀）已废弃：qi=5 上它把 base 的答案从 D 改成 C，'
                           '因为它把推理整段删掉了，不中性。v2 改用 base 自己的前缀 + teacher forcing。'),
           'note': ('前缀 = prompt + base 贪心输出直到第一个「答案：」；'
                    '字母概率 = 该位置 logits 在 A/B/C/D 四类上的 softmax（只比四类，不比全词表）；'
                    'KL 也是在四类分布上算的。'),
           'records': recs, 'secs': round(time.time() - t_all, 1)}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8', newline='')

    print('\n坏题（n=%d）: 字母不变 %d | KL 767 %.6f vs 随机 %.6f（%.2f×）| base argmax=gold %d'
          % (s_bad['n'], s_bad['same_argmax'], s_bad['kl767_median'] or 0,
             s_bad['kl_rand_median'] or 0, s_bad['kl_ratio'] or 0, s_bad['base_argmax_eq_gold']))
    print('对照（n=%d）: 字母不变 %d | KL 767 %s vs 随机 %s'
          % (s_ctl['n'], s_ctl['same_argmax'], s_ctl['kl767_median'], s_ctl['kl_rand_median']))
    print('\n判据 %s' % json.dumps(v, ensure_ascii=False))
    print('读法：%s' % reading)
    print('written %s' % OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
