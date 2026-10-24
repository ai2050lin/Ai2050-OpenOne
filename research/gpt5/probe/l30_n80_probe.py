# -*- coding: utf-8 -*-
"""L30.N80 零 GPU 结构探针 v2（只读权重，无前向，无账本写入）。

在 v1 基础上补三件事：
1) 正确的字节级 BPE 解码（v1 的中文 token 显示为乱码）
2) 跨层同族检测（L26/L28/L29/L31/L32/L34 的读方向 vs L30.N80）—— S9 归族
3) 跨层零分布基线（随机通道对的 cos 均值），用于检验"公共分量抬高 cos"的说法
"""
import json
import os
import sys

import torch
from safetensors import safe_open

MODEL_DIR = r"D:/AI2050/Ai2050-OpenOne/models/hf/qwen3-4b"
OUT_DIR = r"D:/AI2050/Ai2050-OpenOne/research/gpt5/probe"
LAYER = 30
CHAN = 80
CROSS = [26, 28, 29, 31, 32, 34]


def shard_map():
    with open(os.path.join(MODEL_DIR, "model.safetensors.index.json"), encoding="utf-8") as f:
        return json.load(f)["weight_map"]


_handles = {}


def get(tag, wm):
    path = os.path.join(MODEL_DIR, wm[tag])
    if path not in _handles:
        _handles[path] = safe_open(path, framework="pt", device="cpu")
    return _handles[path].get_tensor(tag).float()


def unit(v):
    return v / v.norm().clamp_min(1e-12)


def cos(a, b):
    return float(torch.dot(unit(a), unit(b)))


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


B2U = bytes_to_unicode()
U2B = {v: k for k, v in B2U.items()}


def decode_tok(s):
    try:
        raw = bytes(U2B[ch] for ch in s)
        return raw.decode("utf-8")
    except Exception:
        return s


def main():
    wm = shard_map()
    g = get("model.layers.30.mlp.gate_proj.weight", wm)
    u = get("model.layers.30.mlp.up_proj.weight", wm)
    d = get("model.layers.30.mlp.down_proj.weight", wm)
    gamma = get("model.norm.weight", wm)

    r = g[CHAN].clone()
    urow = u[CHAN].clone()
    w = d[:, CHAN].clone()
    gn = g.norm(dim=1)
    dn = d.norm(dim=0)

    def pct(vals, v):
        return float((vals < v).float().mean() * 100.0)

    res = {
        "layer": LAYER, "chan": CHAN,
        "norm_read": round(float(r.norm()), 4),
        "norm_up": round(float(urow.norm()), 4),
        "norm_write": round(float(w.norm()), 4),
        "median_read_all": round(float(gn.median()), 4),
        "median_write_all": round(float(dn.median()), 4),
        "read_norm_pct": round(pct(gn, r.norm()), 2),
        "write_norm_pct": round(pct(dn, w.norm()), 2),
        "cos_read_write": round(cos(r, w), 4),
    }

    cr = (g @ unit(r)) / gn.clamp_min(1e-9)
    cwf = (d.t() @ unit(r)) / dn.clamp_min(1e-9)
    res["null_read_read"] = {"mean": round(float(cr.mean()), 4),
                             "p95": round(float(cr.quantile(0.95)), 4),
                             "max": round(float(cr.max()), 4)}
    res["null_write_read"] = {"mean": round(float(cwf.mean()), 4),
                              "p95": round(float(cwf.quantile(0.95)), 4),
                              "max": round(float(cwf.max()), 4)}
    res["read_family_same_layer"] = {
        "n_gt_0.5": int((cr > 0.5).sum()), "n_gt_0.7": int((cr > 0.7).sum()),
        "best_other": int(torch.topk(cr, 2).indices[1]),
        "best_other_cos": round(float(torch.topk(cr, 2).values[1]), 4),
    }
    tr = torch.topk(cr, 10)
    res["top_read_same_layer"] = [{"chan": int(i), "cos": round(float(c), 4)}
                                  for c, i in zip(tr.values, tr.indices)]
    tw = torch.topk(cwf, 10)
    res["top_writers_along_read"] = [{"chan": int(i), "cos": round(float(c), 4)}
                                     for c, i in zip(tw.values, tw.indices)]

    # 全通道自对齐分布：每个通道的 读方向 vs 自己的写方向
    sa = (g * d.t()).sum(1) / (gn * dn).clamp_min(1e-9)
    topn_idx = torch.topk(gn, 20).indices
    res["selfalign_all"] = {
        "mean": round(float(sa.mean()), 4), "median": round(float(sa.median()), 4),
        "p95": round(float(sa.quantile(0.95)), 4), "max": round(float(sa.max()), 4),
        "pct_of_80": round(float((sa < sa[CHAN]).float().mean() * 100), 2),
        "n_gt_0.5": int((sa > 0.5).sum()), "n_gt_0.7": int((sa > 0.7).sum()),
    }
    res["selfalign_top_norm20"] = [
        {"chan": int(i), "norm": round(float(gn[i]), 3), "selfalign": round(float(sa[i]), 4)}
        for i in topn_idx]

    gid = torch.Generator().manual_seed(0)
    cross = []
    for L in CROSS:
        gg = get("model.layers.%d.mlp.gate_proj.weight" % L, wm)
        ggn = gg.norm(dim=1)
        c = (gg @ unit(r)) / ggn.clamp_min(1e-9)
        top = torch.topk(c, 3)
        i1 = torch.randint(0, g.shape[0], (2000,), generator=gid)
        i2 = torch.randint(0, gg.shape[0], (2000,), generator=gid)
        base = ((g[i1] * gg[i2]).sum(1) / (gn[i1] * ggn[i2])).mean()
        cross.append({"layer": L, "max_cos": round(float(c.max()), 4),
                      "argmax_chan": int(c.argmax()),
                      "top3": [{"chan": int(i), "cos": round(float(v), 4)}
                               for v, i in zip(top.values, top.indices)],
                      "random_pair_mean_cos": round(float(base), 4),
                      "p99": round(float(c.quantile(0.99)), 4)})
    res["cross_layer"] = cross

    with open(os.path.join(MODEL_DIR, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    id2tok = {v: k for k, v in vocab.items()}
    emb_tag = "model.embed_tokens.weight"
    path = os.path.join(MODEL_DIR, wm[emb_tag])
    fh = safe_open(path, framework="pt", device="cpu")
    emb = fh.get_tensor(emb_tag)
    V = emb.shape[0]

    def lens(vec):
        xh = (gamma * unit(vec)).to(torch.float32)
        out = torch.empty(V, dtype=torch.float32)
        for i in range(0, V, 8192):
            out[i:i + 8192] = emb[i:i + 8192].float() @ xh
        return out

    def show(s, tag):
        top = torch.topk(s, 20)
        bot = torch.topk(-s, 20)
        res[tag] = {
            "pos": [decode_tok(id2tok.get(int(i), "?")) for i in top.indices],
            "neg": [decode_tok(id2tok.get(int(i), "?")) for i in bot.indices],
        }

    show(lens(r), "lens_read")
    show(lens(w), "lens_write")

    print("== L%d.N%d ==" % (LAYER, CHAN))
    print("norm read %.4f (pct %.2f) | write %.4f (pct %.2f) | medians %.4f / %.4f"
          % (res["norm_read"], res["read_norm_pct"], res["norm_write"],
             res["write_norm_pct"], res["median_read_all"], res["median_write_all"]))
    print("cos(read,write)=%.4f  | null(write,read) mean %.4f p95 %.4f max %.4f"
          % (res["cos_read_write"], res["null_write_read"]["mean"],
             res["null_write_read"]["p95"], res["null_write_read"]["max"]))
    print("same-layer read family: >0.5 %d, >0.7 %d, best other N%d cos %.4f"
          % (res["read_family_same_layer"]["n_gt_0.5"], res["read_family_same_layer"]["n_gt_0.7"],
             res["read_family_same_layer"]["best_other"], res["read_family_same_layer"]["best_other_cos"]))
    sa_all = res["selfalign_all"]
    print("self-align all: mean %.4f median %.4f p95 %.4f max %.4f | 80 is at pct %.2f | >0.5 %d, >0.7 %d"
          % (sa_all["mean"], sa_all["median"], sa_all["p95"], sa_all["max"],
             sa_all["pct_of_80"], sa_all["n_gt_0.5"], sa_all["n_gt_0.7"]))
    print("top-norm20 channels (chan/norm/selfalign):",
          [(x["chan"], x["norm"], x["selfalign"]) for x in res["selfalign_top_norm20"][:8]])
    print("top readers:", [(x["chan"], x["cos"]) for x in res["top_read_same_layer"][:6]])
    print("top writers along read:", [(x["chan"], x["cos"]) for x in res["top_writers_along_read"][:6]])
    for c in cross:
        print("L%d: max cos %.4f @N%d | p99 %.4f | random-pair mean %.4f"
              % (c["layer"], c["max_cos"], c["argmax_chan"], c["p99"], c["random_pair_mean_cos"]))
    print("lens_read  + :", " | ".join(res["lens_read"]["pos"][:14]))
    print("lens_read  - :", " | ".join(res["lens_read"]["neg"][:14]))
    print("lens_write + :", " | ".join(res["lens_write"]["pos"][:14]))
    print("lens_write - :", " | ".join(res["lens_write"]["neg"][:14]))

    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, "l30_n80_probe.json")
    with open(p, "w", encoding="utf-8", newline="") as fo:
        json.dump(res, fo, ensure_ascii=False, indent=2)
    print("WROTE", p)


if __name__ == "__main__":
    sys.exit(main())
