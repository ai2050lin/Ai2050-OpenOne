# -*- coding: utf-8 -*-
"""Q2-C1 前置探查：验证"实体互换"这对 clean/corrupt 是否可用于激活修补。

为什么不能用原预注册的"主动 vs 被动"
------------------------------------
原设计是「把被动句 agent 位置的激活换成主动句的」。
但主动句与被动句的答案**相同**（都是张三），替换后输出本不该变，
干预没有可测效应 —— 这是 D6 要防的情形（判据无法区分备择项）。

改用标准因果追踪的配对：**同句式、实体互换**。
  clean   = 张三送给了李四一个苹果。是谁送的？   -> 张三
  corrupt = 李四送给了张三一个苹果。是谁送的？   -> 李四
两句 token 数相同、实体位置相同，唯一差别是"谁是 agent"，答案相反。
于是"把 clean 在某层某位置的激活搬进 corrupt"就有了明确的期望效应：
  效应 = P(张三) 回升，P(李四) 下降。

本探查只验证三件事，不产出结论
------------------------------
1. clean / corrupt 的 token 长度是否一致（不一致则位置索引全部作废）
2. 实体 token 索引是否对齐（agent 在两个句子里是否落在同一批位置）
3. 强制三选一的行为门：clean 答张三、corrupt 答李四，且都近乎确定

用法
----
  .venv/Scripts/python.exe -X utf8 tools/pat_q2c1_probe.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'tools')):
    if p not in sys.path:
        sys.path.insert(0, p)

from pat_q1_unbind import FORMS, QUERIES, build, encode, render, triples

N_TRIP = 4


def main() -> int:
    from server import live_analysis_service as las

    t0 = time.time()
    las._load_async()
    while las._state['status'] in ('idle', 'loading'):
        if time.time() - t0 > 600:
            raise RuntimeError('加载超时')
        time.sleep(2.0)
    model, tok = las._state['model'], las._state['tokenizer']
    dev = next(model.parameters()).device
    print('模型就绪  设备 %s' % dev)

    trips = triples(N_TRIP)
    for form in ('A', 'C'):
        print('\n===== 句式 %s =====' % form)
        for ti, trip in enumerate(trips):
            agent, recipient, theme = trip
            swapped = (recipient, agent, theme)
            rows = []
            for tag, tr in (('clean', trip), ('corrupt', swapped)):
                ctx, ranges = build(tr, form)
                raw = ctx + QUERIES['agent']
                text = render(tok, raw)
                ids, idx = encode(tok, text, raw, ranges)
                rows.append({'tag': tag, 'ids': ids, 'idx': idx, 'trip': tr, 'raw': raw})
            a, b = rows
            same_len = len(a['ids']) == len(b['ids'])
            same_pos = all(a['idx'][r] == b['idx'][r] for r in ('agent', 'recipient', 'theme'))
            print('  trip%-2d T=%d/%d  等长=%s  实体位置对齐=%s'
                  % (ti, len(a['ids']), len(b['ids']), same_len, same_pos))
            print('         clean  : %s  idx=%s' % (a['raw'], {k: v for k, v in a['idx'].items()}))
            print('         corrupt: %s  idx=%s' % (b['raw'], {k: v for k, v in b['idx'].items()}))

            for r in rows:
                t = torch.tensor([r['ids']], device=dev)
                with torch.no_grad():
                    o = model(input_ids=t, use_cache=False)
                lp = torch.log_softmax(o.logits[0, -1].float(), dim=-1)
                ents = [tok(e, add_special_tokens=False)['input_ids'][0] for e in r['trip']]
                sc = {e: round(float(lp[tok(e, add_special_tokens=False)['input_ids'][0]]), 2)
                      for e in r['trip']}
                pick = max(sc, key=sc.get)
                want = r['trip'][0]
                print('         %-8s 答=%-3s 期望=%-3s %s  %s'
                      % (r['tag'], pick, want, 'OK' if pick == want else 'X', sc))

    print('\n=== 分词粒度（实体是否单 token）===')
    for e in ('张三', '李四', '苹果', '书'):
        ids = tok(e, add_special_tokens=False)['input_ids']
        print('  %-3s -> %d token  %s' % (e, len(ids), [tok.decode([i]) for i in ids]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
