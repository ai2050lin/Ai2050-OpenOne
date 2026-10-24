# -*- coding: utf-8 -*-
"""neuron_panel.py —— 封存面板的行语义与因子分解（服务端助手，只读）

背景：phase3169 (g5a6_oov_panel) 的封存 npz 只有 H 与 marg 两个数组，
      H[NT, PAIRS, layers, dims] 的第 0 维是 **模板轴**（3 个句式），
      第 1 维是 pair = entity_index * NC + class_index（NC=10）。
      行 → 具体文本的映射是确定性的：文本 = TPL[t].format(e=ENTS[i], c=CLASSES[c])，
      而 TPL / 实体表 / 类别表都封存在同一目录的 execution.json 里。

因此「该神经元最强激活的样例文本」不需要重新跑模型，也不需要人工标注——
本模块只做还原，不生成任何标签。

因子分解：面板是 73 实体 × 10 类别 × 3 模板 的完全交叉设计（2190 格，每格 1 观测），
所以每个 (layer,dim) 的方差可精确分解为
    模板 TS / 实体 ES / 类别 CS / 两两交互 / 三项剩余（=残差，无重复观测）
份额之和恒为 1。它回答的问题不是「这个坐标是什么」，而是
「这个坐标的方差主要由哪种因子驱动」——句式、内容、身份，还是它们的组合。

纪律：
  - 只读封存产物；不写 registry / npz / phase 结果；
  - 缺失即返回 None 并说明，不猜、不用记忆值填充；
  - 文本只来自 TPL.format，不拼接任何自然语言解释。
"""
from __future__ import annotations

import io
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_DIR = os.path.join(ROOT, "tests", "glm5", "result",
                         "rdc_query_construction_20260913", "phase3169", "g5a6_oov_panel")
EXEC_PATH = os.path.join(PANEL_DIR, "execution.json")

_INDEX = None


class PanelUnavailable(RuntimeError):
    pass


def load_index() -> dict:
    """构建行语义索引（幂等缓存）。缺 execution.json 时抛 PanelUnavailable。"""
    global _INDEX
    if _INDEX is not None:
        return _INDEX
    if not os.path.isfile(EXEC_PATH):
        raise PanelUnavailable("execution.json missing: %s" % EXEC_PATH)
    ex = json.load(io.open(EXEC_PATH, encoding="utf-8"))
    tpl = ex.get("tpl") or {}
    classes = list(ex.get("classes_seen") or []) + list(ex.get("classes_oov") or [])
    ent = {}
    for c, v in (ex.get("ent_seen_full") or {}).items():
        ent[c] = list(v)
    for c, v in (ex.get("ent_oov_full") or {}).items():
        ent[c] = list(v)
    ents, cls_of = [], []
    for ci, c in enumerate(classes):
        for e in ent[c]:
            ents.append(e)
            cls_of.append(ci)
    ne, nc, nt = len(ents), len(classes), len(tpl)
    if ne * nc != int(ex.get("pairs_full") or 0):
        raise PanelUnavailable("panel mismatch: %d entities x %d classes != pairs_full" % (ne, nc))
    _INDEX = dict(
        tpl={int(k): v for k, v in tpl.items()},
        classes=classes, ents=ents, cls_of=np.asarray(cls_of, dtype=np.int32),
        ne=ne, nc=nc, nt=nt, pairs=ne * nc,
        source=os.path.relpath(EXEC_PATH, ROOT).replace(os.sep, "/"),
        design="fully crossed %d entities x %d classes x %d templates = %d rows" % (ne, nc, nt, ne * nc * nt),
        axis_semantics="H[t, pair, layer, dim]; axis0=template (NOT seed); pair = entity*%d + class" % nc,
    )
    return _INDEX


def snippet(t: int, pair: int) -> str:
    """pair 行在模板 t 下的真实文本（模板来自封存 execution.json）。"""
    ix = load_index()
    i, c = divmod(int(pair), ix["nc"])
    return ix["tpl"][int(t)].format(e=ix["ents"][i], c=ix["classes"][c])


def label_of(pair: int) -> dict:
    """行的因子标签（实体 / 类别 / 是否对角行）。"""
    ix = load_index()
    i, c = divmod(int(pair), ix["nc"])
    return dict(entity=ix["ents"][i], cls=ix["classes"][c],
                entity_class=ix["classes"][int(ix["cls_of"][i])],
                diagonal=bool(ix["cls_of"][i] == c))


def _ss_terms(A: np.ndarray) -> dict:
    """A: (T,E,C) → 各因子平方和（标量）。完全交叉、每格 1 观测 ⇒ 分解精确闭合。"""
    T, E, C = A.shape
    g = A.mean()
    dev = A - g
    m_t = dev.mean((1, 2))
    m_e = dev.mean((0, 2))
    m_c = dev.mean((0, 1))
    ss_t = E * C * float((m_t ** 2).sum())
    ss_e = T * C * float((m_e ** 2).sum())
    ss_c = T * E * float((m_c ** 2).sum())
    m_te = dev.mean(2)
    m_tc = dev.mean(1)
    m_ec = dev.mean(0)
    ss_te = C * float(((m_te - m_t[:, None] - m_e[None, :]) ** 2).sum())
    ss_tc = E * float(((m_tc - m_t[:, None] - m_c[None, :]) ** 2).sum())
    ss_ec = T * float(((m_ec - m_e[:, None] - m_c[None, :]) ** 2).sum())
    ss_tot = float((dev ** 2).sum())
    ss_tec = max(ss_tot - (ss_t + ss_e + ss_c + ss_te + ss_tc + ss_ec), 0.0)
    return dict(total=ss_tot, t=ss_t, e=ss_e, c=ss_c, te=ss_te, tc=ss_tc, ec=ss_ec, tec=ss_tec)


def _as_shares(ss: dict) -> dict:
    tot = ss["total"]
    if tot <= 1e-12:
        return {k: 0.0 for k in ("t", "e", "c", "te", "tc", "ec", "tec")}
    return {k: round(ss[k] / tot, 4) for k in ("t", "e", "c", "te", "tc", "ec", "tec")}


def factor_of_column(col: np.ndarray) -> dict:
    """col: (T, E, C) 或 (T, PAIRS) → 该坐标的因子构成（份额）。

    恒定坐标（面板上方差≈0，如 L0 的 embedding 输出）不参与分解：
    返回 degenerate=True，份额全 0，并明确写出「分解无意义」，不强行给主因子。
    """
    ix = load_index()
    A = col.reshape(ix["nt"], ix["ne"], ix["nc"]).astype(np.float64)
    ss = _ss_terms(A)
    if ss["total"] <= 1e-12:
        return dict(shares={k: 0.0 for k in ("t", "e", "c", "te", "tc", "ec", "tec")},
                    dominant=None, degenerate=True,
                    reading="该坐标在本面板上无方差（所有格子取值相同），因子分解无意义")
    sh = _as_shares(ss)
    main = max(("t", "e", "c"), key=lambda k: sh[k])
    return dict(shares=sh, degenerate=False,
                dominant={"t": "template", "e": "entity", "c": "class"}[main],
                reading="方差主要由%s驱动（主因子份额 %.2f；份额之和为 1，余项=三项交互）"
                        % ({"t": "句式(模板)", "e": "实体(身份)", "c": "类别(内容)"}[main], sh[main]))


def layer_factor_means(slab: np.ndarray) -> dict:
    """slab: (T, PAIRS, D) 整层 → 全维平均份额 + 单义度计数。

    返回的 frac_cs_gt_0_5 = 「类别份额 > 0.5 的坐标占比」，
    是「本层是否存在内容单义坐标」的直接读数（不是标签，是计数）。
    """
    ix = load_index()
    A = slab.reshape(ix["nt"], ix["ne"], ix["nc"], -1).astype(np.float64)
    T, E, C, D = A.shape
    g = A.mean((0, 1, 2), keepdims=True)
    dev = A - g
    m_t = dev.mean((1, 2))
    m_e = dev.mean((0, 2))
    m_c = dev.mean((0, 1))
    ss_t = E * C * (m_t ** 2).sum(0)
    ss_e = T * C * (m_e ** 2).sum(0)
    ss_c = T * E * (m_c ** 2).sum(0)
    m_te = dev.mean(2)
    m_tc = dev.mean(1)
    m_ec = dev.mean(0)
    ss_te = C * ((m_te - m_t[:, None] - m_e[None, :]) ** 2).sum((0, 1))
    ss_tc = E * ((m_tc - m_t[:, None] - m_c[None, :]) ** 2).sum((0, 1))
    ss_ec = T * ((m_ec - m_e[:, None] - m_c[None, :]) ** 2).sum((0, 1))
    ss_tot = (dev ** 2).sum((0, 1, 2))
    ss_tec = np.maximum(ss_tot - (ss_t + ss_e + ss_c + ss_te + ss_tc + ss_ec), 0.0)
    den = np.maximum(ss_tot, 1e-12)
    sh = dict(t=ss_t / den, e=ss_e / den, c=ss_c / den,
              te=ss_te / den, tc=ss_tc / den, ec=ss_ec / den, tec=ss_tec / den)
    return dict(
        n_dims=int(D),
        mean={k: round(float(v.mean()), 4) for k, v in sh.items()},
        frac_cs_gt_0_5=round(float((sh["c"] > 0.5).mean()), 4),
        frac_ts_gt_0_5=round(float((sh["t"] > 0.5).mean()), 4),
        top_class_dims=[int(i) for i in np.argsort(-sh["c"])[:5]],
        design=load_index()["design"],
    )


def poles(H, layer: int, dim: int, k: int = 6) -> dict:
    """该坐标的正极/负极样例（真实文本），供证据卡直接展示。

    返回两极各 k 条：{template, pair, value, text, entity, cls, diagonal}
    排序 = 对该 (layer,dim) 的 NT×PAIRS 个单元格按激活值 argtop / argbottom。
    """
    ix = load_index()
    col = H[:, :, layer, dim].astype(np.float32)      # (T, PAIRS)
    flat = col.reshape(-1)
    n_pairs = col.shape[1]
    n_t = col.shape[0]
    order = np.argsort(flat)

    def one(idx):
        t, p = int(idx // n_pairs), int(idx % n_pairs)
        d = label_of(p)
        return dict(template=t, pair=p, value=round(float(flat[idx]), 4),
                    text=snippet(t, p), **d)

    k = int(min(k, flat.shape[0]))
    pos = [one(i) for i in order[::-1][:k]]
    neg = [one(i) for i in order[:k]]
    return dict(positive=pos, negative=neg,
                method="argtop/argbottom over template×pair；文本由封存 execution.json 的模板与实体表还原",
                axis_semantics=ix["axis_semantics"])
