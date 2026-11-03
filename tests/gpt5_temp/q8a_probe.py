# -*- coding: utf-8 -*-
"""Q8-A 预探：分词形态、单 token 覆盖率、范数-频率初看、各向异性基线。"""
import json
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MP = 'D:/develop/model/Qwen/Qwen3-4B'
tok = AutoTokenizer.from_pretrained(MP)
model = AutoModelForCausalLM.from_pretrained(MP, dtype=torch.bfloat16).eval()
E = model.get_input_embeddings().weight.detach().float().cpu().numpy()
print('emb', E.shape, 'tied', model.config.tie_word_embeddings)
norms = np.linalg.norm(E, axis=1)
print('norm mean %.4f median %.4f min %.3f max %.3f' % (norms.mean(), np.median(norms), norms.min(), norms.max()))

words = ['红', '苹果', '香蕉', '柠檬', '橙子', '深红', '咖啡色', '紫罗兰', '龘', '靐', '猕猴桃',
         '百香果', '火龙果', '牛油果', '橄榄绿', '薰衣草紫', '巧克力色', '的', '一']
for w in words:
    ids = tok.encode(w, add_special_tokens=False)
    toks = [tok.decode([i]) for i in ids]
    print('%-6s n=%d %s' % (w, len(ids), toks))

from wordfreq import zipf_frequency as zf

singles = []
for i in range(E.shape[0]):
    s = tok.decode([i])
    if len(s) == 1 and '\u4e00' <= s <= '\u9fff':
        f = zf(s, 'zh')
        if f > 0.3:
            singles.append((s, i, f))
print('single CJK chars with zipf>0.3:', len(singles))
f = np.array([t[2] for t in singles])
nn = np.array([norms[t[1]] for t in singles])
print('zipf range %.2f-%.2f' % (f.min(), f.max()))
from scipy.stats import spearmanr
print('spearman(log norm, zipf) = %.4f' % spearmanr(nn, f).statistic)

# 各向异性：随机 2000 个 token 的平均两两余弦（原始 / 去均值后）
rng = np.random.default_rng(0)
idx = rng.choice(E.shape[0], 2000, replace=False)
X = E[idx]
Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
C = Xn @ Xn.T
m = ~np.eye(len(idx), dtype=bool)
print('raw mean pairwise cos %.4f' % C[m].mean())
mu = E.mean(0)
Xc = X - mu
Xc = Xc / np.linalg.norm(Xc, axis=1, keepdims=True)
Cc = Xc @ Xc.T
print('centered mean pairwise cos %.4f' % Cc[m].mean())
print('||mu|| %.4f vs mean norm %.4f' % (np.linalg.norm(mu), norms.mean()))
