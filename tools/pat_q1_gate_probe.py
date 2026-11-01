# -*- coding: utf-8 -*-
"""Q1 行为门探针：模型在「是谁送的？」后到底预测什么。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from server import live_analysis_service as las

PROMPTS = [
    '张三送给了李四一个苹果。是谁送的？',
    '张三送给了李四一个苹果。送给了谁？',
    '张三送给了李四一个苹果。送的是什么？',
    '苹果被张三送给了李四。是谁送的？',
    '苹果被张三送给了李四。送给了谁？',
    '苹果被张三送给了李四。送的是什么？',
]

las._load_async()
t0 = time.time()
while las._state['status'] in ('idle', 'loading'):
    if time.time() - t0 > 600:
        raise SystemExit('load timeout')
    time.sleep(2.0)
model, tok = las._state['model'], las._state['tokenizer']
dev = next(model.parameters()).device

for p in PROMPTS:
    ids = tok(p, add_special_tokens=False)['input_ids']
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids], device=dev), use_cache=False)
    lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
    top = torch.topk(lp, 6)
    toks = [(tok.decode([int(i)]), round(float(v), 3)) for v, i in zip(top.values, top.indices)]
    print(p)
    print('   n_tok=%d  top=%s' % (len(ids), toks))
