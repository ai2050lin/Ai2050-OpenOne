# -*- coding: utf-8 -*-
"""范数 vs 可读性：检验"参数越大 = 神经元越有效"这一直觉（零 GPU，只读权重）。

对 qwen3-4b L30 的 9728 个通道，测四类量并求秩相关：
  A. nrm_read  = ||W_gate[j,:]||          （读方向长度）
  B. nrm_write = ||W_down[:,j]||          （写方向长度 = 单通道最大输出尺寸）
  C. selfalign = cos(读方向, 写方向)      （读的东西是否等于写回去的东西）
  D. 词表可读性三指标（抽样 600 个通道算）：
       lens_coh_pos = ||mean(E[top20])|| / mean(||E[top20]||)   同 S2 的 coh 公式
       lens_coh_neg = 同上，取 bottom20
       lens_sharp   = (mean(s[top20]) - mean(s)) / std(s)

若"越大越有效"成立，则应看到 A/B 与 C/D 强正相关。
"""
import json
import os
import sys

import torch
from safetensors import safe_open

MODEL_DIR = r"D:/AI2050/Ai2050-OpenOne/models/hf/qwen3-4b"
OUT_DIR = r"D:/AI2050/Ai2050-OpenOne/research/gpt5/probe"
LAYER = 30
N_SAMPLE = 600


def shard_map():
    with open(os.path.join(MODEL_DIR, "model.safetensors.index.json"), encoding="utf-8") as f:
        return json.load(f)["weight_map"]


def rank(v):
    """平均秩（处理并列）。"""
    n = v.numel()
    order = torch.argsort(v)
    r = torch.empty(n, dtype=torch.float64)
    r[order] = torch.arange(n, dtype=torch.float64)
    # 并列取平均
    sv, _ = torch.sort(v)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            m = (i + j) / 2.0
            mask = v == sv[i]
            r[mask] = m
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    return float((ra * rb).sum() / (ra.norm() * rb.norm()).clamp_min(1e-12))


def kurt(s):
    s = s.double()
    m = s.mean()
    sd = s.std()
    return float((((s - m) / sd) ** 4).mean() - 3.0)


def main():
    wm = shard_map()
    h = {}

    def get(tag):
        if tag not in h:
            p = os.path.join(MODEL_DIR, wm[tag])
            h[tag] = safe_open(p, framework="pt", device="cpu").get_tensor(tag).float()
        return h[tag]

    g = get("model.layers.%d.mlp.gate_proj.weight" % LAYER)
    d = get("model.layers.%d.mlp.down_proj.weight" % LAYER)
    gamma = get("model.norm.weight")
    emb = get("model.embed_tokens.weight")
    V, Hdim = emb.shape
    N = g.shape[0]

    gn = g.norm(dim=1)
    dn = d.norm(dim=0)
    sa = (g * d.t()).sum(1) / (gn * dn).clamp_min(1e-9)

    # ---- 抽样：范数分层 ----
    order = torch.argsort(gn, descending=True)
    top100 = order[:100]
    bot100 = order[-100:]
    gid = torch.Generator().manual_seed(7)
    midpool = order[100:N - 100]
    mid400 = midpool[torch.randperm(midpool.numel(), generator=gid)[:400]]
    sample = torch.cat([top100, mid400, bot100])

    def lens_stats(vec):
        rhat = vec / vec.norm().clamp_min(1e-12)
        x = (gamma * rhat).float()
        s = torch.empty(V, dtype=torch.float32)
        for i in range(0, V, 16384):
            s[i:i + 16384] = emb[i:i + 16384] @ x
        ti = torch.topk(s, 20).indices
        bi = torch.topk(-s, 20).indices
        ep = emb[ti].double()
        en = emb[bi].double()

        def coh(E):
            mu = E.mean(0)
            return float(mu.norm() / E.norm(dim=1).mean().clamp_min(1e-12))

        sharp = float((s[ti].mean() - s.mean()) / s.std().clamp_min(1e-12))
        return coh(ep), coh(en), sharp, kurt(s)

    rows = []
    for idx, j in enumerate(sample.tolist()):
        cp, cn, sh, ku = lens_stats(g[j])
        rows.append({"chan": int(j), "nrm": float(gn[j]), "nrm_w": float(dn[j]),
                     "selfalign": float(sa[j]), "coh_pos": cp, "coh_neg": cn,
                     "sharp": sh, "kurt": ku})
        if (idx + 1) % 150 == 0:
            print("  lens %d/%d" % (idx + 1, sample.numel()))

    nrm = torch.tensor([r["nrm"] for r in rows])
    nrm_w = torch.tensor([r["nrm_w"] for r in rows])
    sal = torch.tensor([r["selfalign"] for r in rows])
    cp = torch.tensor([r["coh_pos"] for r in rows])
    cn = torch.tensor([r["coh_neg"] for r in rows])
    sh = torch.tensor([r["sharp"] for r in rows])
    ku = torch.tensor([r["kurt"] for r in rows])

    print("\n== 抽样 %d 个通道（top100/mid400/bot100 by ||read||）==" % len(rows))
    print("nrm range %.3f .. %.3f | selfalign median %.4f | coh_pos median %.4f | sharp median %.3f"
          % (nrm.min(), nrm.max(), sal.median(), cp.median(), sh.median()))

    pairs = [("nrm_read", nrm, "selfalign", sal),
             ("nrm_read", nrm, "coh_pos", cp),
             ("nrm_read", nrm, "coh_neg", cn),
             ("nrm_read", nrm, "sharp", sh),
             ("nrm_read", nrm, "kurt", ku),
             ("nrm_write", nrm_w, "selfalign", sal),
             ("nrm_write", nrm_w, "sharp", sh),
             ("nrm_read", nrm, "nrm_write", nrm_w)]
    cors = {}
    print("\n-- Spearman 秩相关 --")
    for n1, v1, n2, v2 in pairs:
        c = spearman(v1, v2)
        cors["%s~%s" % (n1, n2)] = round(c, 4)
        print("  %-12s ~ %-12s  rho = %+.4f" % (n1, n2, c))

    # 分层均值：范数最高的 100 vs 最低的 100
    t = torch.arange(len(rows))
    ti = t[nrm.argsort(descending=True)][:100]
    bi = t[nrm.argsort()][:100]
    bands = {}
    print("\n-- 范数分层（top100 vs bot100）--")
    for name, v in [("selfalign", sal), ("coh_pos", cp), ("coh_neg", cn),
                    ("sharp", sh), ("kurt", ku), ("nrm_write", nrm_w)]:
        a, b = float(v[ti].mean()), float(v[bi].mean())
        bands[name] = {"top100": round(a, 4), "bot100": round(b, 4),
                       "ratio": round(a / b, 3) if abs(b) > 1e-9 else None}
        print("  %-10s top100 %+.4f | bot100 %+.4f" % (name, a, b))

    # 全层（不抽样）范数与自对齐的关系
    print("\n-- 全层 9728 通道 --")
    print("spearman(nrm_read, selfalign) = %+.4f" % spearman(gn, sa))
    q = torch.quantile(gn, torch.tensor([0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]))
    print("nrm_read 分位:", [round(float(x), 4) for x in q])
    for lo, hi, tag in [(0.0, 0.25, "Q1 最低25%"), (0.75, 0.9, "Q4 75-90%"), (0.99, 1.0, "top1%")]:
        m = (gn >= torch.quantile(gn, lo)) & (gn <= torch.quantile(gn, hi))
        if m.sum() > 0:
            print("  %-10s n=%5d  selfalign mean %+.4f  >0.5: %d  <-0.5: %d"
                  % (tag, int(m.sum()), float(sa[m].mean()),
                     int((sa[m] > 0.5).sum()), int((sa[m] < -0.5).sum())))

    res = {"layer": LAYER, "n_sample": len(rows), "spearman": cors, "bands": bands,
           "full_layer": {"spearman_nrm_selfalign": round(spearman(gn, sa), 4),
                          "nrm_quantiles": [round(float(x), 4) for x in q]},
           "rows": rows}
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, "norm_vs_readability.json")
    with open(p, "w", encoding="utf-8", newline="") as fo:
        json.dump(res, fo, ensure_ascii=False, indent=2)
    print("WROTE", p)


if __name__ == "__main__":
    sys.exit(main())
