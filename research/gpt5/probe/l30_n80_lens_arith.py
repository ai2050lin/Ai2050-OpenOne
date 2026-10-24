# -*- coding: utf-8 -*-
"""L30.N80 读方向词表的完整算术过程（零 GPU，只读权重）。

回答两个问题：
  1) 范数 0.8170 是怎么算出来的，能不能逐项核对
  2) 正端 / 负端的 20 个 token 是怎么排出来的，中间每一步做了什么、省掉会怎样
"""
import json
import os

import torch
from safetensors import safe_open

MODEL_DIR = r"D:/AI2050/Ai2050-OpenOne/models/hf/qwen3-4b"
LAYER, CHAN = 30, 80


def wm():
    with open(os.path.join(MODEL_DIR, "model.safetensors.index.json"), encoding="utf-8") as f:
        return json.load(f)["weight_map"]


def get(tag, m):
    return safe_open(os.path.join(MODEL_DIR, m[tag]), framework="pt", device="cpu").get_tensor(tag).float()


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


U2B = {v: k for k, v in bytes_to_unicode().items()}


def dec(s):
    try:
        return bytes(U2B[ch] for ch in s).decode("utf-8")
    except Exception:
        return s


m = wm()
W = get("model.layers.30.mlp.gate_proj.weight", m)   # [9728, 2560]
gamma = get("model.norm.weight", m)                  # [2560]
r = W[CHAN].clone()

# ---------- 1 范数 ----------
ss = (r * r).sum()
nrm = ss.sqrt()
gn = W.norm(dim=1)
print("[1] 范数")
print("  ||r||^2 = sum_i r_i^2 = %.6f" % float(ss))
print("  ||r||   = sqrt(%.6f) = %.6f" % (float(ss), float(nrm)))
print("  逐项抽样 r[0:6] =", [round(float(v), 6) for v in r[:6]])
print("  该 6 项平方和 = %.6f（占总量 %.4f%%）" % (
    float((r[:6] ** 2).sum()), float((r[:6] ** 2).sum() / ss * 100)))
sq = r * r
top = torch.topk(sq, 10)
print("  |r_i| 最大的 10 个坐标（坐标号, r_i, 平方, 占总量%）:")
for v, i in zip(top.values, top.indices):
    print("    dim %4d  r=%+.6f  r^2=%.6f  %.3f%%" % (int(i), float(r[i]), float(v), float(v / ss * 100)))
print("  前 10 个坐标合计占 ||r||^2 的 %.2f%%（其余 2550 维占 %.2f%%）" % (
    float(top.values.sum() / ss * 100), 100 - float(top.values.sum() / ss * 100)))
print("  范数分布：median %.4f  min %.4f  max %.4f  p1 %.4f  p99 %.4f" % (
    float(gn.median()), float(gn.min()), float(gn.max()),
    float(gn.quantile(0.01)), float(gn.quantile(0.99))))
print("  比 0.8170 小的通道数 = %d / 9728 (%.2f%%)" % (int((gn < nrm).sum()), float((gn < nrm).float().mean() * 100)))

# ---------- 2 词表投影 ----------
rh = r / nrm
g = gamma
emb_tag = "model.embed_tokens.weight"
E = safe_open(os.path.join(MODEL_DIR, m[emb_tag]), framework="pt", device="cpu").get_tensor(emb_tag)
V = E.shape[0]
x = (g * rh).to(torch.float32)
s = torch.empty(V, dtype=torch.float32)
for i in range(0, V, 8192):
    s[i:i + 8192] = E[i:i + 8192].float() @ x

with open(os.path.join(MODEL_DIR, "vocab.json"), encoding="utf-8") as f:
    vocab = json.load(f)
id2 = {v: k for k, v in vocab.items()}

print()
print("[2] 词表投影 s = E @ (gamma * r_hat)")
print("  ||gamma|| = %.4f  gamma: min %.4f  max %.4f  mean %.4f" % (
    float(g.norm()), float(g.min()), float(g.max()), float(g.mean())))
print("  s 的统计：min %.4f  max %.4f  std %.4f  mean %.4f" % (
    float(s.min()), float(s.max()), float(s.std()), float(s.mean())))

for tag, sc in (("正端 top-8", s), ("负端 top-8", -s)):
    tt = torch.topk(sc, 8)
    print(" ", tag)
    for v, i in zip(tt.values, tt.indices):
        print("    %-16r s=%+.4f  id=%d" % (dec(id2.get(int(i), "?")), float(v), int(i)))

# 单个 token 的逐项拆解
ti = int(torch.topk(s, 1).indices[0])
contrib = g * rh * E[ti].float()
tc = torch.topk(contrib, 5)
print("  对 top-1 token %r 贡献最大的 5 个坐标（s 应等于全部 2560 项之和）:" % dec(id2.get(ti, "?")))
for v, i in zip(tc.values, tc.indices):
    print("    dim %4d  gamma=%.4f  r_hat=%+.6f  E=%+.4f  ->  %+.6f" % (
        int(i), float(g[i]), float(rh[i]), float(E[ti][i]), float(v)))
print("    top5 贡献 %.4f / s=%.4f = %.2f%%（2560 项分散，无单点主导）" % (
    float(tc.values.sum()), float(s[ti]), float(tc.values.sum() / s[ti] * 100)))

# ---------- 3 每一步省掉会怎样 ----------
print()
print("[3] 消融：每一步省掉后的 top-20 重叠")
def top20(v):
    return set(int(i) for i in torch.topk(v, 20).indices)

s_nog = torch.empty(V, dtype=torch.float32)
for i in range(0, V, 8192):
    s_nog[i:i + 8192] = E[i:i + 8192].float() @ rh.to(torch.float32)

s_norm_g = torch.empty(V, dtype=torch.float32)
rn = r / r.norm()
for i in range(0, V, 8192):
    s_norm_g[i:i + 8192] = E[i:i + 8192].float() @ (g * r).to(torch.float32)

base = top20(s)
print("  基准（单位化 + 乘 gamma）           top-20 = %d/20" % len(base))
print("  不乘 gamma                          top-20 重叠 = %d/20" % len(base & top20(s_nog)))
print("  不单位化、直接乘 gamma               top-20 重叠 = %d/20（仅缩放，排序应不变）" % len(base & top20(s_norm_g)))
print("  正端 20 个 token 的 s 区间: %.4f ~ %.4f" % (
    float(torch.topk(s, 20).values.min()), float(torch.topk(s, 20).values.max())))
print("  负端 20 个 token 的 s 区间: %.4f ~ %.4f" % (
    float(torch.topk(-s, 20).values.min()), float(torch.topk(-s, 20).values.max())))
