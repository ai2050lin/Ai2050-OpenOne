# -*- coding: utf-8 -*-
"""Q1 解绑落点（v2）：疑问词取的是「角色」还是「位置」。

版本记录
--------
v1（裸文本续写）已作废：行为门 0/12，模型把「句子。问句？」当文本续写，
   预测「（」「谁」等，不做问答。属 probe 设计问题，按 AGENTS.md §175 升版本重做。
v2（本文件）三处修改：
   ① 走 chat template + system「你只输出一个词，不要解释。」+ enable_thinking=False
   ② 行为门改**强制选择**口径（三个实体首 token 的 log P 取最大），不依赖模型肯不肯直说
   ③ 注意力位置改在 assistant 起始位（渲染文本末位）
   SMOKE 实测：2 句式 x 3 查询 6/6 全对，log P≈0。含被动句（agent 居中）仍答对
   => 行为层面「角色跟随、位置不决定」已成立，本实验转到注意力层问同一件事。

问题
----
H-UNBIND-01：同一份绑定，换一个查询就取出不同槽。
若成立，注意力在该查询位应落回**角色对应的实体**，且落点应随句式改变而跟着角色走。
若落点只跟位置（永远指向前/中/后某个槽），则「解绑」不成立，只是位置偏置。

设计（2 句式 x 3 查询，一次分开角色与位置）
------------------------------------------
  句 A  「{A}送给了{B}一个{C}。」   位置：前=agent   中=recipient  后=theme
  句 C  「{C}被{A}送给了{B}。」     位置：前=theme   中=agent      后=recipient

  查询  Q_ag「是谁送的？」  Q_re「送给了谁？」  Q_th「送的是什么？」

  | 查询 | 句A 落点 | 句C 落点 |
  |------|---------|---------|
  | agent     | 前 | 中 |
  | recipient | 中 | 后 |
  | theme     | 后 | 前 |

跟角色 => 落点随句式变（前->中 / 中->后 / 后->前）
跟位置 => 落点不变

测量量
------
每个 (层, 头) 取末位（assistant 起始）对各实体 token 集合的注意力质量 mass，
减去**中性问句**同位置质量：Δ = mass(query) − mass(neutral)。
不扣的话，任何头都会出现「落在最后一个实体」的假信号。

预注册判据（跑前冻结，跑后不改）
--------------------------------
D1 解绑成立     存在 (层,头) 使角色命中率 obs ≥ 0.50 且 obs − null_mean ≥ 0.15
                （null = 每三元组内随机置换三个查询与角色的对应，200 次）
D2 跟角色非位置  通过 D1 的头里，≥50% 满足 same_role ≥ 0.66
                （句A 与句C 落在同一角色；因两式位置不同，same_role 即「跟角色」）
D3 层位          通过 D1 的头层中位数 ≥ 24（读出段）。≤21 则与 2831 冲突，须复核
D4 行为门        强制选择正确率 ≥ 0.60

功效三栏（AGENTS.md §13，tools/power_analysis.py）
------------------------------------------------
预期 Δ = 0.25（命中 0.58 vs 机会 0.33）；所需 n = 54（p_d=0.5, α=0.05, power=0.8）
实际 n = 72 三元组 => 检验力 0.851（Δ=0.27 时 0.90）
每三元组 8 次前向（2 句式 x (3 查询 + 1 中性)），共 576 次。

判读
----
D1+D2 成立 => 解绑成立，框架 C（注意力路由）获支持，「指针规则表」路线可继续。
D1 但非 D2 => 落点只是位置偏置，解绑假说在本句式不成立。
D1+D2 成立但 D4 不过 => 表征在、读出被劫持，是更强的发现，须单独记。
D3 ≤21      => 与 2831「问句不改写历史」冲突，先复核 2831 口径再看本结果。

用法
----
  SMOKE=1 .venv/Scripts/python.exe -X utf8 tools/pat_q1_unbind.py
  .venv/Scripts/python.exe -X utf8 tools/pat_q1_unbind.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'tools')):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

import provenance

SMOKE = os.environ.get('SMOKE', '') == '1'
SEED = 20261101
ROLES = ('agent', 'recipient', 'theme')
QUERIES = {'agent': '是谁送的？', 'recipient': '送给了谁？', 'theme': '送的是什么？'}
NEUTRAL = '他昨天在家吗？'
SYS = '你只输出一个词，不要解释。'

PERSONS = ['张三', '李四', '王五', '赵六', '小明', '小红',
           '老王', '小刚', '阿强', '丽丽', '大卫', '张伟']
OBJECTS = ['苹果', '书', '花', '杯子', '蛋糕', '伞']

FORMS = {
    'A': ([('{A}', 'agent'), ('送给了', None), ('{B}', 'recipient'),
           ('一个', None), ('{C}', 'theme'), ('。', None)],
          {'agent': 0, 'recipient': 1, 'theme': 2}),
    'C': ([('{C}', 'theme'), ('被', None), ('{A}', 'agent'), ('送给了', None),
           ('{B}', 'recipient'), ('。', None)],
          {'theme': 0, 'agent': 1, 'recipient': 2}),
}


def triples(n: int):
    out = []
    for i in range(len(PERSONS)):
        for j in range(len(OBJECTS)):
            out.append((PERSONS[i], PERSONS[(i + 3) % len(PERSONS)], OBJECTS[j]))
    return out[:n]


def build(trip, form):
    """返回 (context 文本, {角色: (字符起, 字符止)})。"""
    agent, recipient, theme = trip
    fill = {'A': agent, 'B': recipient, 'C': theme}
    segs, ranges, pos = [], {}, 0
    for raw, role in FORMS[form][0]:
        piece = fill[raw[1]] if raw.startswith('{') else raw
        segs.append(piece)
        if role:
            ranges[role] = (pos, pos + len(piece))
        pos += len(piece)
    return ''.join(segs), ranges


def render(tok, user_text):
    return tok.apply_chat_template(
        [{'role': 'system', 'content': SYS}, {'role': 'user', 'content': user_text}],
        add_generation_prompt=True, tokenize=False, enable_thinking=False)


def encode(tok, text, user_text, ranges):
    """渲染文本 -> ids 与实体 token 索引。offset 是字符级，实体可能跨 token。"""
    base = text.find(user_text)
    if base < 0:
        raise ValueError('渲染文本中找不到用户内容')
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    ids, offs = enc['input_ids'], enc['offset_mapping']
    idx = {}
    for role, (c0, c1) in ranges.items():
        c0, c1 = c0 + base, c1 + base
        sel = [i for i, (a, b) in enumerate(offs) if a < c1 and b > c0]
        if not sel:
            raise ValueError('实体无 token：%s' % role)
        idx[role] = sel
    return ids, idx


@torch.no_grad()
def forward(model, tok, dev, user_text, ranges):
    """返回 ([L,H,3] 注意力质量, {实体: log P 首 token}, T, 末位 token 文本)。"""
    text = render(tok, user_text)
    ids, idx = encode(tok, text, user_text, ranges)
    t = torch.tensor([ids], device=dev)
    out = model(input_ids=t, use_cache=False, output_attentions=True)
    last = len(ids) - 1
    atts = out.attentions
    L = len(atts)
    H = int(atts[0].shape[1])
    M = np.zeros((L, H, len(ROLES)), dtype=np.float32)
    for li, att in enumerate(atts):
        v = att[0, :, last, :].float().cpu().numpy()
        for k, role in enumerate(ROLES):
            M[li, :, k] = v[:, idx[role]].sum(axis=1)
    lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
    return M, lp, len(ids), tok.decode([ids[-1]])


def forced_choice(tok, lp, trip):
    sc = {}
    for e in trip:
        tid = tok(e, add_special_tokens=False)['input_ids'][0]
        sc[e] = float(lp[tid])
    return max(sc, key=sc.get), sc


def main() -> int:
    from server import live_analysis_service as las

    t0 = time.time()
    las._load_async()
    while las._state['status'] in ('idle', 'loading'):
        if time.time() - t0 > 600:
            raise RuntimeError('模型加载超时')
        time.sleep(2.0)
    if las._state['status'] != 'ready':
        raise RuntimeError('模型不可用：%s' % las._state.get('error'))
    model = las._state['model']
    tok = las._state['tokenizer']
    dev = next(model.parameters()).device
    meta = las._state['meta']
    model_name = Path(las._state['dir']).name
    print('模型 %s 层 %d 头 %d' % (model_name, meta['n_layers'], meta['n_heads']))

    n_trip = 2 if SMOKE else 72
    trips = triples(n_trip)
    L, H = meta['n_layers'], meta['n_heads']

    land = {f: {r: np.full((L, H, len(trips)), -1, dtype=np.int8) for r in ROLES} for f in FORMS}
    gate_ok = {f: {r: 0 for r in ROLES} for f in FORMS}
    gate_rows = []

    for ti, trip in enumerate(trips):
        for form in FORMS:
            ctx, ranges = build(trip, form)
            base, _, _, _ = forward(model, tok, dev, ctx + NEUTRAL, ranges)
            for role in ROLES:
                M, lp, ntok, lasttok = forward(model, tok, dev, ctx + QUERIES[role], ranges)
                land[form][role][:, :, ti] = (M - base).argmax(axis=2)
                want = {'agent': trip[0], 'recipient': trip[1], 'theme': trip[2]}[role]
                pick, sc = forced_choice(tok, lp, trip)
                gate_ok[form][role] += int(pick == want)
                if ti < 2:
                    gate_rows.append({'form': form, 'role': role, 'pick': pick,
                                      'want': want, 'ok': pick == want,
                                      'T': ntok, 'last': lasttok,
                                      'score': {k: round(v, 2) for k, v in sc.items()}})
        if SMOKE or (ti + 1) % 12 == 0:
            print('  三元组 %d/%d  %.1fs' % (ti + 1, len(trips), time.time() - t0))

    ridx = {r: i for i, r in enumerate(ROLES)}
    nobs = len(trips) * len(ROLES) * len(FORMS)
    hit = np.zeros((L, H), dtype=np.float32)
    same_role = np.zeros((L, H), dtype=np.float32)
    same_pos = np.zeros((L, H), dtype=np.float32)
    for f in FORMS:
        for r in ROLES:
            hit += (land[f][r] == ridx[r]).sum(axis=2)
    hit /= nobs

    posmap = {f: {i: FORMS[f][1][r] for i, r in enumerate(ROLES)} for f in FORMS}
    for r in ROLES:
        a, c = land['A'][r], land['C'][r]
        same_role += ((a == c) & (a == ridx[r])).sum(axis=2)
        pa = np.vectorize(lambda e: posmap['A'].get(int(e), -9))(a)
        pc = np.vectorize(lambda e: posmap['C'].get(int(e), -8))(c)
        same_pos += (pa == pc).sum(axis=2)
    same_role /= (len(trips) * len(ROLES))
    same_pos /= (len(trips) * len(ROLES))

    rng = np.random.default_rng(SEED)
    NP = 50 if SMOKE else 200
    null = np.zeros((L, H), dtype=np.float32)
    for _ in range(NP):
        acc = np.zeros((L, H), dtype=np.float32)
        for f in FORMS:
            perm = rng.permutation(len(ROLES))
            for k, r in enumerate(ROLES):
                acc += (land[f][r] == perm[k]).sum(axis=2)
        null += acc / nobs
    null /= NP

    ok = (hit >= 0.50) & (hit - null >= 0.15)
    heads = sorted([(int(l), int(h), float(hit[l, h]), float(null[l, h]),
                     float(same_role[l, h]), float(same_pos[l, h])) for l, h in np.argwhere(ok)],
                   key=lambda x: -x[2])
    gate_all = float(np.mean([gate_ok[f][r] for f in FORMS for r in ROLES]) / len(trips))
    lay = np.array([h[0] for h in heads]) if heads else np.array([])

    d1 = bool(len(heads) > 0)
    d2 = bool(len(heads) > 0 and np.mean([h[4] >= 0.66 for h in heads]) >= 0.50)
    d3 = bool(len(heads) > 0 and float(np.median(lay)) >= 24)
    d4 = bool(gate_all >= 0.60)
    if d1 and d2 and d4:
        verdict = 'D1+D2+D4 PASS 解绑成立（C 获支持）'
    elif d1 and not d2:
        verdict = 'D1 PASS D2 FAIL 位置偏置，非解绑'
    elif d1 and d2 and not d4:
        verdict = 'D1+D2 PASS D4 FAIL 表征在、读出被劫持'
    elif not d1:
        verdict = 'D1 FAIL 无角色跟随头'
    else:
        verdict = 'D3 FAIL 层位与 2831 冲突'

    summary = {
        'version': 'v2', 'model': model_name, 'n_layers': L, 'n_heads': H,
        'n_triples': len(trips), 'n_forward': len(trips) * 8,
        'n_heads_ok': len(heads),
        'D1': d1, 'D2': d2, 'D3_median_layer_ge24': d3, 'D4_gate_ge0.60': d4,
        'gate_rate': round(gate_all, 4),
        'median_layer': float(np.median(lay)) if len(lay) else None,
        'top_heads': [{'layer': l, 'head': h, 'hit': round(a, 4), 'null': round(b, 4),
                       'same_role': round(c, 4), 'same_pos': round(d, 4)}
                      for l, h, a, b, c, d in heads[:30]],
        'heads_all': [[l, h] for l, h, *_ in heads],
        'layer_hist': {int(k): int(v) for k, v in zip(*np.unique(lay, return_counts=True))} if len(lay) else {},
        'gate_sample': gate_rows,
        'verdict': verdict,
    }
    print(json.dumps({k: v for k, v in summary.items() if k != 'gate_sample'},
                     ensure_ascii=False, indent=1))

    if not SMOKE:
        p = provenance.write_result('pat_q1_unbind_v2', summary, seed=SEED,
                                    tool='tools/pat_q1_unbind.py')
        print('落盘 %s' % p)
    else:
        print('SMOKE 完成，未落盘')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
