# -*- coding: utf-8 -*-
"""Q8-A 附加检查（跑后追加）：颜色的聚集有多少来自共享汉字。

主脚本只对 FRUIT 做了共享字过滤（下降 24%），没有对 COLOR 做。而颜色词共享字的比例可能高得多
（大红/朱红/绯红/深红 共享「红」）。本脚本只回答一个问题：
  把「共享汉字」的词对全部剔除后，within(COLOR) 还剩多少？两条 within 的相对大小是否改变？

用法：.venv/Scripts/python.exe -X utf8 tests/gpt5/q8a_extra_char_overlap.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / 'results/2026-11-01/091106_pat_q8a.json'
MP = 'D:/develop/model/Qwen/Qwen3-4B'

doc = json.loads(RES.read_text(encoding='utf-8'))
P = doc['payload']

tok = AutoTokenizer.from_pretrained(MP)
model = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.bfloat16).eval()
E = model.get_input_embeddings().weight.detach().float().cpu().numpy()
mu = E.mean(0)


def pooled(words):
    return np.array([E[tok.encode(w, add_special_tokens=False)].mean(0) for w in words]) - mu


def unit(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)


out = {}
for nm in ('COLOR', 'FRUIT'):
    words = P['sets'][nm]['words']
    X = unit(pooled(words))
    S = X @ X.T
    n = len(words)
    iu = np.triu_indices(n, 1)
    sims = S[iu]
    pairs = [(words[i], words[j]) for i, j in zip(*iu)]
    shared = np.array([len(set(a) & set(b)) > 0 for a, b in pairs])
    w_all = float(sims.mean())
    w_ns = float(sims[~shared].mean()) if (~shared).any() else float('nan')
    # 同族（共享字）内部 vs 族间
    fam = float(sims[shared].mean()) if shared.any() else float('nan')
    out[nm] = dict(n=n, n_pairs=n, n_shared=int(shared.sum()),
                   frac_shared=float(shared.mean()),
                   within_all=w_all, within_no_shared=w_ns, within_shared_pairs=fam,
                   drop_frac=float((w_all - w_ns) / w_all))
    print('%-6s pairs=%d 共享字对 %d (%.0f%%)  within 全体 %.4f / 去共享字 %.4f (降 %.1f%%) / 仅共享字对 %.4f'
          % (nm, n, shared.sum(), 100 * shared.mean(), w_all, w_ns,
             100 * out[nm]['drop_frac'], fam), flush=True)

print('\n去共享字后 COLOR−FRUIT = %+.4f（原 %+.4f）'
      % (out['COLOR']['within_no_shared'] - out['FRUIT']['within_no_shared'],
         out['COLOR']['within_all'] - out['FRUIT']['within_all']), flush=True)

# 颜色三族：红/蓝/绿 家族内部的平均相似度（同字）与跨族
import re
for root_ in ('红', '蓝', '绿', '黄'):
    fam = [w for w in P['sets']['COLOR']['words'] if root_ in w and w != root_]
    if len(fam) < 3:
        continue
    V = unit(pooled(fam))
    S = V @ V.T
    n = len(fam)
    iu = np.triu_indices(n, 1)
    print('家族「%s」 n=%d 族内平均 cos %.4f   成员: %s'
          % (root_, n, float(S[iu].mean()), ' '.join(fam)), flush=True)

o = ROOT / 'results/2026-11-01' / 'q8a_extra_char_overlap.json'
o.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8', newline='')
print('OUT', o)
