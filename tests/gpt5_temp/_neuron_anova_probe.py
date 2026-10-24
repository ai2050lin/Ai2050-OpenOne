# -*- coding: utf-8 -*-
"""残差坐标因子分解探针（严格全交叉 ANOVA，零 GPU）

问题：封存面板恰好是 73 实体 × 10 类别 × 3 模板 = 2190 的完全交叉设计，
      因此可以把每个残差坐标 (layer,dim) 的方差 **精确** 分解为：
        实体主效应 ES / 类别主效应 CS / 模板主效应 TS
        + 两项交互 ES×CS / ES×TS / CS×TS + 三项剩余
      据此回答「这个坐标在做什么」：句式(模板) / 内容(类别) / 身份(实体) / 组合。

与 top-k 纯度法的区别：top-k 在并列值下不稳定，且把「模板主导」误读成「类别纯度高」。
本探针不做任何抽样：每层全部 dim 一次算完。

输出：tests/gpt5_temp/_neuron_anova.json + 控制台简表
纪律：只读封存 npz；不写 registry；全部数字实测。
"""
import io
import json
import os
import zipfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHASE = os.path.join(ROOT, "tests", "glm5", "result",
                     "rdc_query_construction_20260913", "phase3169", "g5a6_oov_panel")
ex = json.load(io.open(os.path.join(PHASE, "execution.json"), encoding="utf-8"))
TPL = ex["tpl"]
CLASSES = list(ex["classes_seen"]) + list(ex["classes_oov"])
ENT = {}
ENT.update({c: list(v) for c, v in ex["ent_seen_full"].items()})
ENT.update({c: list(v) for c, v in ex["ent_oov_full"].items()})
ENTS, CLS_OF = [], []
for ci, c in enumerate(CLASSES):
    for e in ENT[c]:
        ENTS.append(e)
        CLS_OF.append(ci)
NE, NC, NT = len(ENTS), len(CLASSES), len(TPL)
assert (NE, NC, NT) == (73, 10, 3), (NE, NC, NT)
MODELS = ["qwen3-4b", "qwen3-14b", "glm4-9b"]


def load_H(model):
    p = os.path.join(PHASE, "collect_%s.npz" % model)
    with zipfile.ZipFile(p) as z:
        with z.open("H.npy") as f:
            H = np.load(f, allow_pickle=False)
    assert H.ndim == 4 and H.shape[0] == NT and H.shape[1] == NE * NC, H.shape
    return H


def anova(A):
    """A: (T, E, C, D) float32 → 各因子平方和份额（D,）四项 + 明细。

    完全交叉且每格 1 观测 ⇒ 三项交互即残差，分解精确闭合。
    """
    T, E, C, D = A.shape
    N = T * E * C
    g = A.reshape(-1, D).mean(0)
    dev = A - g                                    # (T,E,C,D)

    def ss(x, axes):
        return float(np.prod([A.shape[a] for a in axes])) * (x * x).sum(0)

    m_t = dev.mean((1, 2)); m_e = dev.mean((0, 2)); m_c = dev.mean((0, 1))
    m_te = dev.mean(2); m_tc = dev.mean(1); m_ec = dev.mean(0)
    SS_t = ss(m_t, [1, 2])
    SS_e = ss(m_e, [0, 2])
    SS_c = ss(m_c, [0, 1])
    SS_te = C * ((m_te - m_t[:, None] - m_e[None, :]) ** 2).sum((0, 1))
    SS_tc = E * ((m_tc - m_t[:, None] - m_c[None, :]) ** 2).sum((0, 1))
    SS_ec = T * ((m_ec - m_e[:, None] - m_c[None, :]) ** 2).sum((0, 1))
    SS_tot = (dev * dev).sum((0, 1, 2))
    SS_tec = np.maximum(SS_tot - (SS_t + SS_e + SS_c + SS_te + SS_tc + SS_ec), 0.0)
    eps = 1e-12
    sh = lambda x: x / np.maximum(SS_tot, eps)
    return dict(t=sh(SS_t), e=sh(SS_e), c=sh(SS_c),
                te=sh(SS_te), tc=sh(SS_tc), ec=sh(SS_ec), tec=sh(SS_tec))


def snippet(t, ei, ci):
    return TPL[str(t)].format(e=ENTS[ei], c=CLASSES[ci])


def main():
    out = {"schema": "neuron_anova_v1", "source_phase": 3169,
           "design": "fully crossed 73 entities x 10 classes x 3 templates = 2190 rows",
           "axis_note": "H[T,pair,layer,dim]; pair = entity*10 + class",
           "models": {}}
    for model in MODELS:
        H = load_H(model)
        _, _, nlayers, D = H.shape
        layers = sorted({int(round((nlayers - 1) * f)) for f in (0.25, 0.5, 0.75)})
        print("\n=== %s  H=%s ===" % (model, H.shape))
        print(" layer  模板TS  类别CS  实体ES   T×C   T×E   C×E   余项 | CS>0.5占比  top-CS 坐标(片段)")
        rep = {}
        for L in layers:
            slab = H[:, :, L, :].astype(np.float32)          # (T, 730, D)
            A = slab.reshape(NT, NE, NC, D)
            s = anova(A)
            mean = {k: float(v.mean()) for k, v in s.items()}
            cs = s["c"]
            top = np.argsort(-cs)[:5]
            toplines = []
            for d in top:
                d = int(d)
                # 该坐标最强的单元格：max |A - grand| 的 (t, e, c)
                col = A[:, :, :, d] - A[:, :, :, d].mean()
                t, e, c = np.unravel_index(int(np.argmax(np.abs(col))), col.shape)
                toplines.append(dict(dim=d, cs=round(float(cs[d]), 3),
                                     ts=round(float(s["t"][d]), 3),
                                     es=round(float(s["e"][d]), 3),
                                     cell=snippet(t, e, c),
                                     act=round(float(A[t, e, c, d]), 4)))
            rep[L] = dict(mean=mean, frac_cs_gt_0_5=float((cs > 0.5).mean()),
                          frac_ts_gt_0_5=float((s["t"] > 0.5).mean()),
                          top_cs=toplines)
            print("  %4d   %5.1f  %6.1f  %6.1f  %5.1f %5.1f %5.1f  %6.1f | %8.2f     %d: %.2f  %s"
                  % (L, 100 * mean["t"], 100 * mean["c"], 100 * mean["e"],
                     100 * mean["te"], 100 * mean["tc"], 100 * mean["ec"],
                     100 * mean["tec"], float((cs > 0.5).mean()),
                     toplines[0]["dim"], toplines[0]["cs"], toplines[0]["cell"]))
        out["models"][model] = dict(H_shape=list(H.shape), layers=rep)
        del H

    p = os.path.join(ROOT, "tests", "gpt5_temp", "_neuron_anova.json")
    with io.open(p, "w", encoding="utf-8", newline="") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\nwrote %s" % p)


if __name__ == "__main__":
    main()
