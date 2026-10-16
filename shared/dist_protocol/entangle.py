# -*- coding: utf-8 -*-
"""纠缠谱与部件归因（entangle.v1）。

问题：不同因子（概念、语言、极性、任务线索）在 HiddenState 中是近似加性可分地组合，还是共享方向、以非加性
方式组合（纠缠）；若是后者，哪些部件（注意力头 / MLP）写入交互项，哪些部件在上下文需要某一维度时把它分离出来。
两种假说都要能被数据区分：加性可分 ⇒ 交互能量在噪声水平、因子方向跨条件不变、未见组合读出无干扰。

输入：逐条向量（items.npz 的 ItemSet，或节点本地原生全场），以及由文法给出的设计（因子、名词、句式变体、名词属性）。
对每个读出位置 × 层 × 因子对 (A, B)：

  函数型方差分解（平衡设计，按格等权）：格均值 M[a,b] = μ + Δa + Δb + I[a,b]
      SS_A = |B|·Σ_a‖Δa‖²，SS_B = |A|·Σ_b‖Δb‖²，SS_I = Σ_ab‖I[a,b]‖²。
  E1 交互能量比 = SS_I / (SS_A + SS_B + SS_I)。原始值含噪声偏差（噪声总在交互项里留下 (|A|−1)(|B|−1) 份方差），
      主值用分半可靠估计：按重复单位（见下）分两半，各自分解，用两半的内积 ⟨I¹, I²⟩ 代替 ‖I‖²（两半噪声独立，期望只剩真实部分）。
  E2 交互张量谱：I 视为 (|A|·|B|) × d 矩阵的奇异值 s_k；p_k = s_k²/Σs²，纠缠熵 H = −Σ p_k ln p_k，有效秩 exp(H)、参与比；
      显著秩 = 超过零分布同序位 (1−α) 分位的奇异值个数（零分布见下）。
  E3 因子子空间主角度：S_A = span{Δa}、S_B = span{Δb}、S_I = I 的主右奇异子空间（累计能量 90%、至多 8 维）；
      报告 S_A 与 S_B 的主角度余弦（共享方向 = 复用），以及交互落在主效应子空间内的比例 frac_I_in_main
      （高 = 交互是对已有方向的增益调制；低 = 交互占用新方向），均附随机子空间基线。
  E4 条件旋转：A 的水平方向组 D_A(b)（在 B=b 内按 A 中心化的格均值）跨 b 的相关，经分半可靠性校正
      corr(b,b') = ⟨D¹(b),D²(b')⟩ / sqrt(⟨D¹(b),D²(b)⟩⟨D¹(b'),D²(b')⟩)；旋转 = 1 − 平均 corr；增益 = 各 b 的可靠范数之比。
  E5 未见组合读出干扰：在 discovery 的 B≠b 上拟合 A 的读出（白化空间水平单位方向），在留出名词上比较
      B≠b（见过的组合）与 B=b（未见组合）的标准化投影间隔；"居中"版本先按各自测试集均值居中，对加性但方向
      不正交的编码不报干扰，只对交互（旋转 / 增益变化）敏感；另给未居中版本。区间为按留出名词整体的自助法。
  线索门控（task_alignment，如 color → task=color）：属性 T 的一个读出方向（discovery 全部线索水平合并拟合），
      在同一批留出名词的各线索水平上的可读出间隔；门控增益 = 对齐水平减其他水平的均值（配对，按留出名词自助法给区间）。

重复单位与零分布（按名词整体 / 句式整体，不逐条打乱）：
  FF（A、B 都是句式级因子，如 lang × task）：单位 = 名词；每个名词有全部格。零分布 = 各名词双重中心化交互的随机翻号。
  NF（A 是名词级标签，如 cat / color；B 是句式级因子）：单位 = 名词；零分布 = 名词整体打乱 A 标签（保持各组名词数）。
  nF（A = 名词本身，B 是句式级因子，如 noun × lang）：单位 = 句式块（除 B 外其余句式因子水平 + 句式变体相同的句式组，
      块内含 B 的全部水平）；零分布 = 各块双重中心化交互的随机翻号。

嵌入层基线：同一读出位置 hs[0] 的 E1 只反映 token 身份（如名词在两种语言中的 token 不同）；"超出 token 身份的纠缠"
要求本层 E1 比 hs[0] 高出 CRITERIA['e1_excess_min']。

部件归因（attribute）：目的层残差的分解 hs[L] = hs[0] + Σ_{b<L}(Σ_h W_O,h z_{b,h} + MLP_b) 是精确的线性分解，
因此每个部件写入的主效应与交互项可以与残差的同名项直接比较（在目的层白化空间内）：
  share_X(c) = ⟨X_c, X_R⟩ / ‖X_R‖²（X ∈ {Δa, Δb, I}，全部部件与嵌入层的份额相加 = 1，闭合误差另报）；
  own_E1(c) = 部件自身写入的分半可靠交互比例；cancel(c) = −⟨I_c, I_in⟩ / ‖I_in‖²（I_in = 该部件输入处残差的交互项）。
  候选"纠缠者"：share_I ≥ share_min 且 own_E1 ≥ own_e1_min；候选"解纠缠者"：cancel ≥ cancel_min 且对主效应份额 ≥ share_min；
  候选"门控者"：因子对含任务对齐属性与任务线索，部件的交互份额 ≥ share_min 且其交互主要落在属性主效应子空间内（增益调制）。
  部件标签只是描述性候选：写入份额不等于因果必要性，须另做预注册干预。

判据固定在 CRITERIA，随结果返回；不得在看到结果后调整。所有结论用「支持 / 未支持 / 注意 / 描述」表述，属于候选稳定观测。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SCHEMA = 'entangle.v1'
NOUN = 'noun'
CRITERIA = {
    'alpha': 0.01,             # 交互能量零分布单侧 p 阈值
    'n_perm': 200,             # 翻号 / 名词整体打乱次数
    'n_split': 8,              # 分半可靠估计的随机分半次数
    'n_boot': 200,             # E5 / 门控按名词整体自助次数
    'e1_min': 0.05,            # 可靠交互比例 ≥ 该值才算「纠缠」（效应量门槛，防止大样本下的微小交互）
    'e1_excess_min': 0.05,     # 超出同一读出位置嵌入层 hs[0] 的可靠交互比例 ≥ 该值才算超出 token 身份
    'rotation_min': 0.2,       # E4 可靠性校正旋转 ≥ 该值记为条件旋转
    'gain_ratio_min': 1.5,     # E4 跨条件增益比 ≥ 该值记为增益调制
    'interference_min': 0.3,   # E5 居中干扰（标准化间隔下降）且自助区间下限 > 0
    'gating_min': 0.3,         # 线索门控增益（标准化间隔）且自助区间下限 > 0
    'subspace_energy': 0.9,    # E3 子空间：累计能量
    'subspace_max_dim': 8,
    'whiten_rank': 256,        # 背景白化：去掉完整条件格均值后的残差协方差，保留的主成分上限
    'whiten_shrink': 1.0,      # 特征值收缩 var + shrink·mean(var)；其余维度按收缩下限缩放（保留全部维度）
    'share_min': 0.1,          # 部件：份额门槛
    'own_e1_min': 0.5,         # 部件：自身写入的交互比例门槛（纠缠者）
    'cancel_min': 0.1,         # 部件：抵消输入交互的比例门槛（解纠缠者）
    'cancel_in_min': 0.1,      # 部件：输入处交互能量 ≥ 该比例 × 目的层交互能量时才计算抵消比例
    'gate_in_main_min': 0.5,   # 部件：交互落在属性主效应子空间内的比例门槛（门控者）
    'min_units': 6,            # 每个因子对至少的重复单位数
}
TOKEN_NOTE = '嵌入层 hs[0] 只含 token 身份；本层可靠交互比例未超出 hs[0] 基线时不计为超出 token 身份的纠缠'


# ---------------------------------------------------------------------------------------------------------------------
# 设计：每条样本的因子标签、名词、句式变体、划分与行为
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Design:
    ids: List[str]
    noun: np.ndarray
    frame: np.ndarray
    variant: np.ndarray
    split: np.ndarray
    labels: Dict[str, np.ndarray]       # 因子 / 属性 / 'noun' → 字符串标签 (N,)
    kind: Dict[str, str]                # 'frame' | 'noun_attr' | 'noun'
    correct: np.ndarray                 # 1 / 0 / -1（无 probe）
    keep: np.ndarray                    # 进入分析的样本
    factors: Tuple[str, ...] = ()       # 文法因子（条件格 = 这些因子的水平组合）

    def frame_factors(self) -> List[str]:
        return [f for f, k in self.kind.items() if k == 'frame']


def _kind_of(lab: np.ndarray, noun: np.ndarray) -> str:
    seen: Dict[str, str] = {}
    for n, v in zip(noun, lab):
        if seen.setdefault(n, v) != v:
            return 'frame'
    return 'noun_attr'


def build_design(ids: Sequence[str], cond: Sequence[Dict[str, Any]], frame: Sequence[str], filler: Sequence[str],
                 split: Sequence[str], correct: Optional[Sequence[int]] = None, spec: Optional[Dict[str, Any]] = None,
                 factors: Optional[Sequence[str]] = None) -> Design:
    """由逐条元数据与文法构造设计。文法提供名词属性（filler.attrs，取 spec.entangle.attrs 列出的键）与句式变体（frame.variant）。"""
    spec = spec or {}
    factors = list(factors or spec.get('factors') or sorted({k for c in cond for k in c}))
    noun = np.array([str(x) for x in filler])
    labels = {f: np.array([str(c.get(f, '')) for c in cond]) for f in factors}
    labels[NOUN] = noun.copy()
    attrs = (spec.get('entangle') or {}).get('attrs') or []
    by_filler = {fl['id']: fl for fl in spec.get('fillers', []) if isinstance(fl, dict)}
    for a in attrs:
        vals = [by_filler.get(n, {}).get('attrs', {}).get(a) for n in noun]
        if all(isinstance(v, (str, int, float)) for v in vals):
            labels[a] = np.array([str(v) for v in vals])
    var_of = {fr['id']: str(fr.get('variant', '')) for fr in spec.get('frames', []) if isinstance(fr, dict)}
    frame = np.array([str(x) for x in frame])
    kind = {f: ('noun' if f == NOUN else _kind_of(lab, noun)) for f, lab in labels.items()}
    n = len(ids)
    corr = np.full(n, -1, dtype=int) if correct is None else np.asarray(correct, dtype=int)
    return Design(list(ids), noun, frame, np.array([var_of.get(f, '') for f in frame]), np.array([str(s) for s in split]),
                  labels, kind, corr, np.ones(n, dtype=bool), tuple(factors))


def design_from_itemset(items, spec: Dict[str, Any]) -> Design:
    """analysis.ItemSet → Design（correct 来自 items.npz）。"""
    return build_design(items.ids, items.cond, items.frame, items.filler, items.split, items.correct, spec)


def design_from_corpus(corpus: Dict[str, Any], spec: Dict[str, Any], ids: Optional[Sequence[str]] = None,
                       correct_by_id: Optional[Dict[str, bool]] = None) -> Design:
    """由文法生成的语料（id 与 runner 相同）构造设计；ids 给定时按其顺序（如原生全场 manifest 的条目顺序）。"""
    by_id = {it['id']: it for it in corpus['items'] if not it.get('canary')}
    ids = list(ids) if ids is not None else list(by_id)
    its = [by_id[i] for i in ids]
    corr = None
    if correct_by_id is not None:
        corr = [(-1 if i not in correct_by_id else int(bool(correct_by_id[i]))) for i in ids]
    return build_design(ids, [it['cond'] for it in its], [it['frame_id'] for it in its],
                        [it['filler_id'] for it in its], [it['split'] for it in its], corr, spec)


def gate_nouns(design: Design, policy: str = 'all_probes', exclude_cells: Optional[Iterable[str]] = None
               ) -> Tuple[Design, Dict[str, Any]]:
    """逐名词行为门：policy='all_probes' 时，某名词任一 probe 判错 → 该名词的全部句式退出分析（保持完全交叉）；
    'none' 不按行为筛选。exclude_cells（中心按格排除的条件格）中的样本也退出；之后每个因子对再单独检查完整性，
    不完整的格组合不做分解（而不是带着不平衡继续算——按格排除造成的不平衡会虚增交互）。"""
    keep = design.keep.copy()
    dropped: List[str] = []
    if policy == 'all_probes':
        bad = {n for n, c in zip(design.noun, design.correct) if c == 0}
        dropped = sorted(bad)
        keep &= ~np.isin(design.noun, list(bad))
    elif policy != 'none':
        raise ValueError(f'unknown noun gate policy {policy!r}')
    n_cells_excl = 0
    if exclude_cells:
        ex = set(exclude_cells)
        fs = list(design.factors)
        ck = np.array(['|'.join(f'{f}={design.labels[f][i]}' for f in sorted(fs)) for i in range(len(design.ids))])
        m = np.isin(ck, list(ex))
        n_cells_excl = int(m.sum())
        keep &= ~m
    out = replace(design, keep=keep)
    probed = sorted({n for n, c in zip(design.noun, design.correct) if c >= 0})
    report = {'policy': policy, 'n_nouns': int(len(set(design.noun))), 'n_probed_nouns': len(probed),
              'dropped_nouns': dropped, 'n_items_kept': int(keep.sum()), 'n_items_excluded_by_cells': n_cells_excl,
              'kept_by_split_label': _count_nouns(out)}
    return out, report


def _count_nouns(design: Design, label: str = 'cat') -> Dict[str, int]:
    lab = design.labels.get(label)
    out: Dict[str, int] = {}
    seen = set()
    for i in np.where(design.keep)[0]:
        key = (design.noun[i])
        if key in seen:
            continue
        seen.add(key)
        k = f'{design.split[i]}/{lab[i] if lab is not None else "-"}'
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


# ---------------------------------------------------------------------------------------------------------------------
# 因子对布局：重复单位 × A × B 的格均值立方体
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Layout:
    kind: str                      # 'FF' | 'NF' | 'nF'
    A: str
    B: str
    a_levels: List[str]
    b_levels: List[str]
    units: List[str]
    item_idx: np.ndarray           # 进入立方体的样本下标
    cell: np.ndarray               # 每个样本的平坦格号 r·|A|·|B| + a·|B| + b
    avail: np.ndarray              # (R, A) 该单位是否含该 A 水平
    strata: np.ndarray             # (R,) 分半分层标签（NF 为单位的 A 水平，否则全 0）
    note: str = ''


def layout(design: Design, A: str, B: str, idx: np.ndarray) -> Optional[Layout]:
    """构造因子对布局；不支持的组合（两个名词级标签）或不完整设计返回 None。"""
    kA, kB = design.kind.get(A), design.kind.get(B)
    if kA is None or kB is None or A == B:
        return None
    if kA == 'frame' and kB in ('noun_attr', 'noun'):
        A, B, kA, kB = B, A, kB, kA
    if kA == 'frame' and kB == 'frame':
        kind = 'FF'
    elif kA == 'noun_attr' and kB == 'frame':
        kind = 'NF'
    elif kA == 'noun' and kB == 'frame':
        kind = 'nF'
    else:
        return None
    la, lb = design.labels[A][idx], design.labels[B][idx]
    a_levels, b_levels = sorted(set(la)), sorted(set(lb))
    if len(a_levels) < 2 or len(b_levels) < 2:
        return None
    if kind == 'nF':
        others = [f for f in design.frame_factors() if f != B]
        ukey = np.array(['|'.join([design.labels[f][i] for f in others] + [design.variant[i]]) for i in idx])
    else:
        ukey = design.noun[idx]
    units = sorted(set(ukey))
    ui = {u: k for k, u in enumerate(units)}
    ai, bi = {a: k for k, a in enumerate(a_levels)}, {b: k for k, b in enumerate(b_levels)}
    R, nA, nB = len(units), len(a_levels), len(b_levels)
    r = np.array([ui[u] for u in ukey])
    a = np.array([ai[v] for v in la])
    b = np.array([bi[v] for v in lb])
    cell = (r * nA + a) * nB + b
    counts = np.bincount(cell, minlength=R * nA * nB).reshape(R, nA, nB)
    avail = counts.sum(2) > 0
    # 平衡：每个格内的样本构成（其余因子 × 句式变体）必须与完整设计相同——用全设计的众数计数核对；
    # 按格排除只去掉一部分样本时（如只排除肯定句），该单位整体退出，而不是带着偏斜的格均值继续算
    nz = counts[counts > 0]
    expected = int(np.bincount(nz).argmax()) if len(nz) else 0
    if kind == 'NF':
        if (avail.sum(1) != 1).any():                     # 名词级标签必须在名词内恒定
            return None
        okR = (counts.sum(1) == expected).all(1)          # 每个名词在自己的 A 水平上含全部 B 水平且计数一致
    else:
        okR = (counts == expected).all((1, 2))
    if not okR.any():
        return None
    keepR = np.where(okR)[0]
    remap = -np.ones(R, dtype=int)
    remap[keepR] = np.arange(len(keepR))
    sel = okR[r]
    r2 = remap[r[sel]]
    cell = (r2 * nA + a[sel]) * nB + b[sel]
    avail = avail[keepR]
    if kind == 'NF':
        # 名词级标签中不足 2 个名词的水平（如行为门之后只剩 1 个名词的颜色）整体退出，其余水平照常分解
        good = np.where(avail.sum(0) >= 2)[0]
        if len(good) < 2:
            return None
        if len(good) < nA:
            dropped_levels = [a_levels[k] for k in range(nA) if k not in set(good)]
            rowsel = avail[:, good].any(1)
            amap = -np.ones(nA, dtype=int)
            amap[good] = np.arange(len(good))
            rr = (cell // nB) // nA
            aa = (cell // nB) % nA
            bb = cell % nB
            keep_items = rowsel[rr] & (amap[aa] >= 0)
            newR = -np.ones(len(rowsel), dtype=int)
            newR[rowsel] = np.arange(rowsel.sum())
            cell = (newR[rr[keep_items]] * len(good) + amap[aa[keep_items]]) * nB + bb[keep_items]
            idx_sel = idx[sel][keep_items]
            avail = avail[rowsel][:, good]
            keepR = keepR[rowsel]
            a_levels = [a_levels[k] for k in good]
            strata = avail.argmax(1)
            note = (f'{len(keepR)}/{R} 个单位完整；不足 2 个名词的水平退出：{dropped_levels}')
            return Layout(kind, A, B, a_levels, b_levels, [units[k] for k in keepR], idx_sel, cell, avail, strata, note)
        strata = avail.argmax(1)
    else:
        strata = np.zeros(len(keepR), dtype=int)
    note = f'{len(keepR)}/{R} 个单位完整' if len(keepR) < R else ''
    return Layout(kind, A, B, a_levels, b_levels, [units[k] for k in keepR], idx[sel], cell, avail, strata, note)


def _group_mean(X: np.ndarray, cell: np.ndarray, n_cells: int) -> Tuple[np.ndarray, np.ndarray]:
    """按格号求均值（排序 + reduceat，O(N·d)）。返回 (n_cells, d) 与计数；空格为 0。"""
    order = np.argsort(cell, kind='stable')
    c = cell[order]
    starts = np.r_[0, np.where(np.diff(c))[0] + 1]
    sums = np.add.reduceat(X[order], starts, axis=0)
    out = np.zeros((n_cells, X.shape[1]), dtype=np.float64)
    cnt = np.zeros(n_cells, dtype=np.int64)
    out[c[starts]] = sums
    cnt[c[starts]] = np.diff(np.r_[starts, len(c)])
    out[cnt > 0] /= cnt[cnt > 0, None]
    return out, cnt


def cube(L: Layout, X: np.ndarray) -> np.ndarray:
    """样本矩阵 X（与 design 同序，(N, d)）→ 立方体 (R, A, B, d)。"""
    R, nA, nB = len(L.units), len(L.a_levels), len(L.b_levels)
    Y, _ = _group_mean(np.asarray(X[L.item_idx], dtype=np.float64), L.cell, R * nA * nB)
    return Y.reshape(R, nA, nB, -1)


def _cell_means(Y: np.ndarray, avail: np.ndarray, rows: Optional[np.ndarray] = None) -> np.ndarray:
    if rows is not None:
        Y, avail = Y[rows], avail[rows]
    w = avail.astype(np.float64)
    den = w.sum(0)
    return np.einsum('ra,rabd->abd', w, Y) / np.maximum(den, 1)[:, None, None]


def decompose(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """格均值 (A, B, d) → (Δa (A,d), Δb (B,d), I (A,B,d))，按格等权。"""
    mu = M.mean((0, 1))
    da = M.mean(1) - mu
    db = M.mean(0) - mu
    return da, db, M - mu - da[:, None] - db[None]


def _energies(da, db, I) -> Tuple[float, float, float]:
    nA, nB = I.shape[0], I.shape[1]
    return float(nB * (da * da).sum()), float(nA * (db * db).sum()), float((I * I).sum())


def _halves(L: Layout, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """按单位分两半（NF 按 A 水平分层），保证每半每个 A 水平都有单位。"""
    h1, h2 = [], []
    for s in np.unique(L.strata):
        u = np.where(L.strata == s)[0]
        u = u[rng.permutation(len(u))]
        k = len(u) // 2
        h1.extend(u[:k]); h2.extend(u[k:2 * k] if len(u) % 2 == 0 else u[k:])
    return np.array(sorted(h1)), np.array(sorted(h2))


def _rel(x1: np.ndarray, x2: np.ndarray) -> float:
    return float((x1 * x2).sum())


# ---------------------------------------------------------------------------------------------------------------------
# 背景白化
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Whitener:
    mean: np.ndarray
    V: np.ndarray        # (d, r)
    w: np.ndarray        # (r,) 前 r 个主成分的缩放
    w_tail: float        # 其余维度的缩放

    def __call__(self, Y: np.ndarray) -> np.ndarray:
        """作用于（已去均值的）效应向量：最后一维为 d。线性，不减均值。"""
        sh = Y.shape
        Y2 = Y.reshape(-1, sh[-1])
        out = self.w_tail * Y2 + ((Y2 @ self.V) * (self.w - self.w_tail)) @ self.V.T
        return out.reshape(sh)


def fit_whitener(X: np.ndarray, design: Design, idx: np.ndarray, rank: int, shrink: float) -> Whitener:
    """背景 = 去掉完整条件格（全部因子水平组合）均值后的 discovery 残差（名词身份、句式变体与噪声）。"""
    fs = list(design.factors) or [f for f in design.labels if f != NOUN and design.kind[f] == 'frame']
    key = np.array(['|'.join(design.labels[f][i] for f in fs) for i in idx])
    _, inv = np.unique(key, return_inverse=True)
    Xd = np.asarray(X[idx], dtype=np.float64)
    G, _ = _group_mean(Xd, inv, inv.max() + 1)
    Rz = Xd - G[inv]
    _, s, Vt = np.linalg.svd(Rz, full_matrices=False)
    var = s ** 2 / max(len(Rz) - 1, 1)
    r = int(min(rank, (var > 1e-12 * max(var.max(), 1e-30)).sum(), len(var)))
    r = max(r, 1)
    # 方差下限：退化读出（如嵌入层的句末 token 在同一语言内完全相同，背景方差为 0）时避免除零
    floor = 1e-6 * max(float((Xd * Xd).sum(1).mean()) / Xd.shape[1], 1e-30)
    lam = max(shrink * float(var[:r].mean()), floor)
    tail = float(var[r:].mean()) if len(var) > r else 0.0
    return Whitener(Xd.mean(0), Vt[:r].T, 1.0 / np.sqrt(var[:r] + lam), 1.0 / np.sqrt(tail + lam))


def identity_whitener(d: int) -> Whitener:
    return Whitener(np.zeros(d), np.zeros((d, 1)), np.ones(1), 1.0)


def spectrum_whitener(X: np.ndarray, design: Design, split: str, whiten: bool) -> Tuple[Whitener, str]:
    """E1–E4 用的白化：在 split 之外的名词（confirm / lockbox）上估计背景（样本分割）。背景若用同一批名词、去掉同一组
    条件格均值来估计，被检验的标签方向上的背景方差会被扣掉、白化后被放大，置换 / 翻号零分布因此偏宽松。"""
    C = CRITERIA
    if not whiten:
        return identity_whitener(X.shape[1]), 'none'
    other = np.where(design.keep & (design.split != split))[0]
    if len(other) >= max(32, len(set(design.noun[other])) * 2):
        return fit_whitener(X, design, other, C['whiten_rank'], C['whiten_shrink']), 'held_out_nouns'
    same = np.where(design.keep & (design.split == split))[0]
    return (fit_whitener(X, design, same, C['whiten_rank'], C['whiten_shrink']),
            'same_sample（留出名词不足；零分布可能偏宽松）')


# ---------------------------------------------------------------------------------------------------------------------
# 单个因子对的纠缠谱
# ---------------------------------------------------------------------------------------------------------------------
def _spectrum(I: np.ndarray) -> np.ndarray:
    M = I.reshape(-1, I.shape[-1])
    if not np.isfinite(M).all():
        return np.zeros(min(M.shape))
    if M.shape[0] <= M.shape[1]:
        ev = np.linalg.eigvalsh(M @ M.T)[::-1]
    else:
        ev = np.linalg.eigvalsh(M.T @ M)[::-1]
    return np.maximum(ev, 0.0)


def _spec_summary(ev: np.ndarray, max_rank: int) -> Dict[str, Any]:
    ev = ev[:max(max_rank, 1)]
    tot = float(ev.sum())
    if tot <= 1e-18:
        return {'sv2_frac': [], 'entropy': None, 'eff_rank': None, 'participation': None, 'max_rank': max_rank}
    p = ev / tot
    nz = p[p > 1e-15]
    H = float(-(nz * np.log(nz)).sum())
    return {'sv2_frac': [round(float(x), 4) for x in p[:10]], 'entropy': H, 'eff_rank': float(np.exp(H)),
            'participation': float(tot ** 2 / (ev ** 2).sum()), 'max_rank': max_rank,
            'entropy_norm': (H / np.log(min(max_rank, len(ev)))) if min(max_rank, len(ev)) > 1 else None}


def _basis(Mat: np.ndarray, energy: float, max_dim: int) -> np.ndarray:
    M = Mat.reshape(-1, Mat.shape[-1])
    if not np.isfinite(M).all() or float((M * M).sum()) <= 1e-18:
        return np.zeros((M.shape[1], 0))
    _, s, Vt = np.linalg.svd(M, full_matrices=False)
    e = np.cumsum(s ** 2) / (s ** 2).sum()
    k = int(min(max_dim, np.searchsorted(e, energy) + 1, (s > 1e-9 * s[0]).sum()))
    return Vt[:k].T


def _angles(Qa: np.ndarray, Qb: np.ndarray, d: int) -> Dict[str, Any]:
    if Qa.shape[1] == 0 or Qb.shape[1] == 0:
        return {'dims': [Qa.shape[1], Qb.shape[1]], 'cos': [], 'mean_cos2': None, 'baseline': None}
    c = np.clip(np.linalg.svd(Qa.T @ Qb, compute_uv=False), 0, 1)
    return {'dims': [Qa.shape[1], Qb.shape[1]], 'cos': [round(float(x), 4) for x in c[:5]],
            'mean_cos2': float((c ** 2).mean()), 'baseline': max(Qa.shape[1], Qb.shape[1]) / d}


def _frac_in(I: np.ndarray, Q: np.ndarray) -> Optional[float]:
    M = I.reshape(-1, I.shape[-1])
    tot = float((M * M).sum())
    if tot <= 1e-18 or Q.shape[1] == 0:
        return None
    P = M @ Q
    return float((P * P).sum() / tot)


def _orth(*Qs: np.ndarray) -> np.ndarray:
    Q = np.concatenate([q for q in Qs if q.shape[1]], axis=1) if any(q.shape[1] for q in Qs) else Qs[0]
    if Q.shape[1] == 0:
        return Q
    U, s, _ = np.linalg.svd(Q, full_matrices=False)
    return U[:, s > 1e-8]


def _rotation(D1: np.ndarray, D2: np.ndarray) -> Dict[str, Any]:
    """D1, D2：(条件 b, 水平, d) 两半的方向组。返回可靠性校正的跨条件相关、旋转与增益。"""
    nb = D1.shape[0]
    C = np.einsum('bad,cad->bc', D1, D2)
    C = (C + C.T) / 2
    diag = np.diag(C)
    gains = np.sqrt(np.maximum(diag, 0))
    corr = []
    for i in range(nb):
        for j in range(i + 1, nb):
            if diag[i] > 1e-18 and diag[j] > 1e-18:
                corr.append(float(np.clip(C[i, j] / np.sqrt(diag[i] * diag[j]), -1.0, 1.0)))
    pos = gains[gains > 1e-12]
    return {'corr_mean': float(np.mean(corr)) if corr else None,
            'rotation': float(1 - np.mean(corr)) if corr else None,
            'gain': [float(g) for g in gains],
            'gain_ratio': float(pos.max() / pos.min()) if len(pos) == nb and nb >= 2 else None}


def _raw_cos(D: np.ndarray) -> Optional[float]:
    nb = D.shape[0]
    v = D.reshape(nb, -1)
    n = np.linalg.norm(v, axis=1)
    vals = [float(v[i] @ v[j] / (n[i] * n[j])) for i in range(nb) for j in range(i + 1, nb) if n[i] > 0 and n[j] > 0]
    return float(np.mean(vals)) if vals else None


def pair_spectrum(Y: np.ndarray, L: Layout, *, seed: int = 0, n_perm: Optional[int] = None,
                  n_split: Optional[int] = None, Yraw: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """一个因子对在一个读出 × 层上的 E1–E4。Y：白化后的立方体 (R, A, B, d)。"""
    C = CRITERIA
    n_perm = C['n_perm'] if n_perm is None else n_perm
    n_split = C['n_split'] if n_split is None else n_split
    rng = np.random.default_rng(seed)
    R, nA, nB, d = Y.shape
    M = _cell_means(Y, L.avail)
    da, db, I = decompose(M)
    ssA, ssB, ssI = _energies(da, db, I)
    tot = ssA + ssB + ssI
    out: Dict[str, Any] = {'layout': L.kind, 'A': L.A, 'B': L.B, 'levels': [nA, nB], 'units': R,
                           'unit_note': L.note, 'ss': {'A': ssA, 'B': ssB, 'I': ssI},
                           'E1_raw': ssI / tot if tot > 0 else None}
    if Yraw is not None:
        sa, sb, si = _energies(*decompose(_cell_means(Yraw, L.avail)))
        out['E1_raw_unwhitened'] = si / (sa + sb + si) if sa + sb + si > 0 else None
    # 分半可靠估计（E1）与 E4
    relA = relB = relI = 0.0
    rotA: List[Dict[str, Any]] = []
    rotB: List[Dict[str, Any]] = []
    for _ in range(n_split):
        h1, h2 = _halves(L, rng)
        if len(h1) == 0 or len(h2) == 0:
            continue
        M1, M2 = _cell_means(Y, L.avail, h1), _cell_means(Y, L.avail, h2)
        a1, b1, i1 = decompose(M1)
        a2, b2, i2 = decompose(M2)
        relA += nB * _rel(a1, a2); relB += nA * _rel(b1, b2); relI += _rel(i1, i2)
        DA1 = (M1 - M1.mean(0, keepdims=True)).transpose(1, 0, 2)     # (B, A, d)：A 在每个 b 内的方向组
        DA2 = (M2 - M2.mean(0, keepdims=True)).transpose(1, 0, 2)
        DB1 = M1 - M1.mean(1, keepdims=True)                            # (A, B, d)：B 在每个 a 内的方向组
        DB2 = M2 - M2.mean(1, keepdims=True)
        rotA.append(_rotation(DA1, DA2))
        if L.kind != 'nF':                                              # 名词作条件时 B 的方向组只有单位内的估计，跳过
            rotB.append(_rotation(DB1, DB2))
    k = max(n_split, 1)
    relA, relB, relI = relA / k, relB / k, relI / k
    den = max(relA, 0) + max(relB, 0) + max(relI, 0)
    out['rel'] = {'A': relA, 'B': relB, 'I': relI}
    out['E1'] = (max(relI, 0) / den) if den > 0 else None

    def _avg(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rs:
            return {}
        keys = ('corr_mean', 'rotation', 'gain_ratio')
        o = {kk: (float(np.mean([r[kk] for r in rs if r[kk] is not None]))
                  if any(r[kk] is not None for r in rs) else None) for kk in keys}
        o['gain'] = [float(x) for x in np.mean([r['gain'] for r in rs], axis=0)]
        return o
    DA = (M - M.mean(0, keepdims=True)).transpose(1, 0, 2)
    out['E4'] = {'A_given_B': {**_avg(rotA), 'raw_cos': _raw_cos(DA), 'levels_B': L.b_levels},
                 'B_given_A': ({**_avg(rotB), 'raw_cos': _raw_cos(M - M.mean(1, keepdims=True)),
                                'levels_A': L.a_levels} if L.kind != 'nF' else None)}
    # 零分布
    null_I, null_A, null_B, null_ev = _null(Y, L, rng, n_perm)
    ev = _spectrum(I)
    max_rank = (nA - 1) * (nB - 1)
    spec = _spec_summary(ev, max_rank)
    if null_ev is not None and len(null_ev):
        kk = min(len(ev), null_ev.shape[1], max_rank)
        thr = np.quantile(null_ev[:, :kk], 1 - C['alpha'], axis=0)
        sig = 0
        for j in range(kk):
            if ev[j] > thr[j]:
                sig += 1
            else:
                break
        spec['signif_rank'] = sig
    out['E2'] = spec
    out['p'] = {'I': _pval(ssI, null_I), 'A': _pval(ssA, null_A), 'B': _pval(ssB, null_B)}
    out['null'] = {'kind': {'FF': 'sign_flip_noun', 'NF': 'permute_label_by_noun', 'nF': 'sign_flip_frame_block'}[L.kind],
                   'n': int(len(null_I)) if null_I is not None else 0,
                   'I_q95': float(np.quantile(null_I, 0.95)) if null_I is not None and len(null_I) else None}
    # E3 子空间
    C3 = C['subspace_energy'], C['subspace_max_dim']
    Qa, Qb, Qi = _basis(da, *C3), _basis(db, *C3), _basis(I, *C3)
    Qab = _orth(Qa, Qb)
    out['E3'] = {'A_vs_B': _angles(Qa, Qb, d), 'I_vs_A': _angles(Qi, Qa, d), 'I_vs_B': _angles(Qi, Qb, d),
                 'frac_I_in_main': _frac_in(I, Qab), 'frac_baseline': Qab.shape[1] / d,
                 'frac_I_in_A': _frac_in(I, Qa), 'frac_I_in_B': _frac_in(I, Qb)}
    return out


def _pval(obs: float, null: Optional[np.ndarray]) -> Optional[float]:
    if null is None or not len(null):
        return None
    return float((1 + (null >= obs - 1e-12 * max(abs(obs), 1)).sum()) / (1 + len(null)))


def _null(Y: np.ndarray, L: Layout, rng: np.random.Generator, n_perm: int):
    """返回 (SS_I 零分布, SS_A 零分布, SS_B 零分布, 交互特征值零分布 (n, k))。"""
    R, nA, nB, d = Y.shape
    if n_perm <= 0:
        return None, None, None, None
    if L.kind in ('FF', 'nF'):
        Ir = Y - Y.mean(1, keepdims=True) - Y.mean(2, keepdims=True) + Y.mean((1, 2), keepdims=True)
        Ar = Y.mean(2) - Y.mean((1, 2))[:, None]            # (R, A, d)
        Br = Y.mean(1) - Y.mean((1, 2))[:, None]            # (R, B, d)
        If = Ir.reshape(R, -1)
        GI, GA, GB = If @ If.T, Ar.reshape(R, -1) @ Ar.reshape(R, -1).T, Br.reshape(R, -1) @ Br.reshape(R, -1).T
        S = rng.choice([-1.0, 1.0], size=(n_perm, R))
        nI = np.einsum('pr,rs,ps->p', S, GI, S) / R ** 2
        nAv = nB * np.einsum('pr,rs,ps->p', S, GA, S) / R ** 2
        nBv = nA * np.einsum('pr,rs,ps->p', S, GB, S) / R ** 2
        AB = nA * nB
        kk = min(AB, (nA - 1) * (nB - 1))
        if AB <= 1024:
            # 翻号后的平均交互 Ī_s = Σ_r s_r I_r / R（分块计算），其特征值 = Ī_s Ī_sᵀ（AB × AB）的特征值
            If_ = Ir.reshape(R, AB * d)
            evs = []
            for chunk in np.array_split(S, max(1, int(np.ceil(len(S) * AB * d / 2e7)))):
                for Mx in (chunk @ If_ / R).reshape(len(chunk), AB, d):
                    evs.append(np.maximum(np.linalg.eigvalsh(Mx @ Mx.T)[::-1][:kk], 0))
            nev = np.array(evs)
        else:
            nev = None
        return nI, nAv, nBv, nev
    # NF：名词整体打乱 A 标签（各组名词数不变）
    grp = L.strata
    Yrow = np.einsum('ra,rabd->rbd', L.avail.astype(float), Y)       # 每个名词自身 A 水平的行 (R, B, d)
    nI, nAv, evs = [], [], []
    kk = (nA - 1) * (nB - 1)
    for _ in range(n_perm):
        g = grp[rng.permutation(R)]
        M = np.stack([Yrow[g == a].mean(0) for a in range(nA)])
        da, db, I = decompose(M)
        sA, _, sI = _energies(da, db, I)
        nI.append(sI); nAv.append(sA)
        evs.append(_spectrum(I)[:kk])
    # B 主效应：各名词 B 剖面中心化后翻号
    Br = Yrow - Yrow.mean(1, keepdims=True)
    GB = Br.reshape(R, -1) @ Br.reshape(R, -1).T
    S = rng.choice([-1.0, 1.0], size=(n_perm, R))
    w = np.zeros(R)
    for a in range(nA):                                               # 按格等权：每组均值再平均
        w[grp == a] = 1.0 / ((grp == a).sum() * nA)
    nBv = nA * np.einsum('pr,r,rs,s,ps->p', S, w, GB, w, S)
    return np.array(nI), np.array(nAv), nBv, np.array(evs)


# ---------------------------------------------------------------------------------------------------------------------
# E5 未见组合读出干扰 与 线索门控
# ---------------------------------------------------------------------------------------------------------------------
def _margins(Ztr, ytr, Zte, yte, levels, center: bool) -> Optional[np.ndarray]:
    g = Ztr.mean(0)
    dirs = {}
    for lv in levels:
        m = ytr == lv
        if m.any():
            v = Ztr[m].mean(0) - g
            n = np.linalg.norm(v)
            if n > 1e-12:
                dirs[lv] = v / n
    if len(dirs) < 2 or len(Zte) < 2:
        return None
    keys = list(dirs)
    U = np.stack([dirs[k] for k in keys])
    P = (Zte - (Zte.mean(0) if center else g)) @ U.T
    col = {k: i for i, k in enumerate(keys)}
    ti = np.array([col.get(t, -1) for t in yte])
    ok = ti >= 0
    s = np.full(len(yte), np.nan)
    own = P[np.where(ok)[0], ti[ok]]
    s[ok] = own - (P[ok].sum(1) - own) / (len(keys) - 1)
    return s


def _sep(s: Optional[np.ndarray]) -> Optional[float]:
    if s is None:
        return None
    s = s[np.isfinite(s)]
    if len(s) < 2 or s.std() < 1e-12:
        return None
    return float(s.mean() / s.std())


def _boot_mean_diff(groups: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], rng, n_boot: int):
    """groups：[(s_seen, nouns_seen, s_unseen, nouns_unseen)]；按名词整体重采样，统计 mean_g(sep_seen − sep_unseen)。"""
    nouns = sorted({n for g in groups for n in list(g[1]) + list(g[3])})
    if not nouns:
        return None
    pos = {n: k for k, n in enumerate(nouns)}
    gidx = [(s1, np.array([pos[n] for n in n1]), s2, np.array([pos[n] for n in n2])) for s1, n1, s2, n2 in groups]
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(len(nouns), size=len(nouns), replace=True)
        w = np.bincount(pick, minlength=len(nouns)).astype(float)
        diffs = []
        for s1, i1, s2, i2 in gidx:
            a = _wsep(s1, w[i1])
            b = _wsep(s2, w[i2])
            if a is not None and b is not None:
                diffs.append(a - b)
        if diffs:
            vals.append(np.mean(diffs))
    if not vals:
        return None
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def _wsep(s: np.ndarray, w: np.ndarray) -> Optional[float]:
    m = np.isfinite(s) & (w > 0)
    if m.sum() < 2:
        return None
    s, w = s[m], w[m]
    mu = float((w * s).sum() / w.sum())
    sd = float(np.sqrt((w * (s - mu) ** 2).sum() / w.sum()))
    return mu / sd if sd > 1e-12 else None


def interference(Z: np.ndarray, design: Design, target: str, context: str, *, held: Sequence[str] = ('confirm',),
                 seed: int = 0, n_boot: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """E5：target 因子的读出在 context 的未见水平上的干扰。Z：白化后的样本矩阵 (N, d)，与 design 同序。"""
    if target == NOUN or context == NOUN or target not in design.labels or context not in design.labels:
        return None
    n_boot = CRITERIA['n_boot'] if n_boot is None else n_boot
    rng = np.random.default_rng(seed)
    keep = design.keep
    D = keep & (design.split == 'discovery')
    H = keep & np.isin(design.split, list(held))
    y, c = design.labels[target], design.labels[context]
    levels = sorted(set(y[D]))
    if len(levels) < 2 or not H.any():
        return None
    rows, groups_c, groups_r = [], [], []
    for lv in sorted(set(c[keep])):
        tr_seen, tr_match = D & (c != lv), D & (c == lv)
        te_seen, te_unseen = H & (c != lv), H & (c == lv)
        if not (tr_seen.any() and tr_match.any() and te_seen.any() and te_unseen.any()):
            continue
        rec = {'context_level': lv}
        for center in (True, False):
            tag = 'centered' if center else 'raw'
            s_seen = _margins(Z[tr_seen], y[tr_seen], Z[te_seen], y[te_seen], levels, center)
            s_un = _margins(Z[tr_seen], y[tr_seen], Z[te_unseen], y[te_unseen], levels, center)
            s_m = _margins(Z[tr_match], y[tr_match], Z[te_unseen], y[te_unseen], levels, center)
            rec[tag] = {'sep_seen': _sep(s_seen), 'sep_unseen': _sep(s_un), 'sep_matched': _sep(s_m)}
            if s_seen is not None and s_un is not None:
                (groups_c if center else groups_r).append(
                    (s_seen, design.noun[te_seen], s_un, design.noun[te_unseen]))
        rows.append(rec)
    if not rows:
        return None

    def summ(tag, groups):
        diffs = [r[tag]['sep_seen'] - r[tag]['sep_unseen'] for r in rows
                 if r[tag]['sep_seen'] is not None and r[tag]['sep_unseen'] is not None]
        tr = [r[tag]['sep_unseen'] / r[tag]['sep_matched'] for r in rows
              if r[tag]['sep_unseen'] is not None and r[tag]['sep_matched']]
        return {'interference': float(np.mean(diffs)) if diffs else None,
                'ci95': _boot_mean_diff(groups, rng, n_boot) if groups else None,
                'transfer_ratio': float(np.mean(tr)) if tr else None}
    return {'target': target, 'context': context, 'held': list(held), 'by_level': rows,
            'centered': summ('centered', groups_c), 'raw': summ('raw', groups_r)}


def gating(Z: np.ndarray, design: Design, attr: str, task_factor: str, aligned: str, *,
           held: Sequence[str] = ('confirm',), seed: int = 0, n_boot: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """线索门控：属性 attr 在任务线索各水平内（discovery 拟合、留出名词检验、居中）的可读出间隔；门控增益 = 对齐水平 − 其余均值。"""
    if attr not in design.labels or task_factor not in design.labels:
        return None
    n_boot = CRITERIA['n_boot'] if n_boot is None else n_boot
    rng = np.random.default_rng(seed)
    keep = design.keep
    D = keep & (design.split == 'discovery')
    H = keep & np.isin(design.split, list(held))
    y, t = design.labels[attr], design.labels[task_factor]
    levels = sorted(set(y[D]))
    per, margins, matched = {}, {}, {}
    # 主统计量：一个读出方向（discovery 全部线索水平合并拟合），在同一批留出名词的各线索水平上比较——
    # 训练噪声对各水平相同，差异只来自测试端，按名词自助是配对的；另记各水平各自拟合的匹配版本作参考
    for lv in sorted(set(t[keep])):
        te = H & (t == lv)
        s = _margins(Z[D], y[D], Z[te], y[te], levels, True) if te.any() else None
        per[lv] = _sep(s)
        if s is not None:
            margins[lv] = (s, design.noun[te])
        tr = D & (t == lv)
        matched[lv] = _sep(_margins(Z[tr], y[tr], Z[te], y[te], levels, True)) if tr.any() and te.any() else None
    if aligned not in per or per[aligned] is None:
        return None
    others = [v for k, v in per.items() if k != aligned and v is not None]
    if not others:
        return None
    gain = per[aligned] - float(np.mean(others))
    groups = [(margins[aligned][0], margins[aligned][1], margins[k][0], margins[k][1])
              for k in margins if k != aligned]
    ci = _boot_mean_diff(groups, rng, n_boot)
    return {'attr': attr, 'task_factor': task_factor, 'aligned': aligned, 'sep_by_task': per,
            'sep_by_task_matched': matched, 'gain': gain, 'ci95': ci}


# ---------------------------------------------------------------------------------------------------------------------
# 场：按 (读出, hs 下标) 取样本矩阵；可选部件
# ---------------------------------------------------------------------------------------------------------------------
class ArrayField:
    """内存中的场：hs[(读出, hs 下标)] = (N, d)；部件 comps[(读出, 块)] = (Z (N, H·hd), M (N, d))，W[块] = (d, H·hd)。"""

    def __init__(self, hs: Dict[Tuple[str, int], np.ndarray], comps=None, W=None, n_heads: int = 0,
                 head_dim: int = 0, post_norm_last: bool = False):
        self.hs, self.comps, self.W = hs, comps or {}, W or {}
        self.n_heads, self.head_dim, self.post_norm_last = n_heads, head_dim, post_norm_last
        self.readouts = sorted({r for r, _ in hs})
        self.n_hs = 1 + max(i for _, i in hs)

    def tags(self) -> List[Tuple[str, int]]:
        return sorted(self.hs, key=lambda t: (t[0], t[1]))

    def X(self, readout: str, hs_index: int) -> np.ndarray:
        return self.hs[(readout, hs_index)]

    def blocks(self, readout: str) -> List[int]:
        return sorted(b for r, b in self.comps if r == readout)

    def comp(self, readout: str, block: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.comps[(readout, block)]

    def W_O(self, block: int) -> np.ndarray:
        return self.W[block]


class NativeField:
    """节点本地原生全场（runner_v3 写出的 shard-*of*/L*.npz 与 components/）。按需逐层读取，多个分片按条目拼接。"""

    def __init__(self, shard_dirs: Sequence[str]):
        self.dirs = [Path(d) for d in shard_dirs]
        self.man = [json.loads((d / 'manifest.json').read_text(encoding='utf-8')) for d in self.dirs]
        self.ids = [i for m in self.man for i in m['ids']]
        self.readouts = list(self.man[0]['readouts'])
        self.layers = sorted({int(f['hs_index']) for f in self.man[0]['files']})
        self.n_hs = max(self.layers) + 1
        boundary = str(self.man[0].get('hidden_state_boundary') or '')
        self.post_norm_last = True if 'synthetic' not in str(self.man[0].get('status')) else False
        self.boundary = boundary
        cm = [d / 'components' / 'components.json' for d in self.dirs]
        self.cman = [json.loads(p.read_text(encoding='utf-8')) for p in cm] if all(p.exists() for p in cm) else None
        if self.cman:
            self.n_heads, self.head_dim = int(self.cman[0]['n_heads']), int(self.cman[0]['head_dim'])
            for m, mm in zip(self.cman, self.man):
                if m['ids'] != mm['ids']:
                    raise ValueError('部件与原生全场的条目顺序不一致')
        self._W: Dict[int, np.ndarray] = {}

    def tags(self) -> List[Tuple[str, int]]:
        return [(r, i) for r in self.readouts for i in self.layers]

    def X(self, readout: str, hs_index: int) -> np.ndarray:
        p = self.readouts.index(readout)
        parts = []
        for d in self.dirs:
            with np.load(d / f'L{hs_index}.npz') as z:
                parts.append(np.asarray(z['X'][p], dtype=np.float32))
        return np.concatenate(parts)

    def blocks(self, readout: str) -> List[int]:
        if not self.cman or readout not in self.cman[0]['readouts']:
            return []
        return list(self.cman[0]['blocks'])

    def comp(self, readout: str, block: int) -> Tuple[np.ndarray, np.ndarray]:
        p = self.cman[0]['readouts'].index(readout)
        Zs, Ms = [], []
        for d in self.dirs:
            with np.load(d / 'components' / f'B{block}.npz') as z:
                Zs.append(np.asarray(z['Z'][p], dtype=np.float32))
                Ms.append(np.asarray(z['M'][p], dtype=np.float32))
        return np.concatenate(Zs), np.concatenate(Ms)

    def W_O(self, block: int) -> np.ndarray:
        if block not in self._W:
            info = self.cman[0]['o_proj']
            f = next(f for f in info['files'] if int(f['block']) == block)
            p = Path(info['dir']) / f['path']
            data = p.read_bytes()
            if hashlib.sha256(data).hexdigest() != f['sha256']:
                raise ValueError(f'o_proj 权重哈希不符：{p}')
            self._W[block] = np.load(p).astype(np.float32)
        return self._W[block]


# ---------------------------------------------------------------------------------------------------------------------
# 主分析：全部读出 × 层 × 因子对
# ---------------------------------------------------------------------------------------------------------------------
def analyze(field, design: Design, pairs: Sequence[Sequence[str]], *, tags: Optional[Sequence[Tuple[str, int]]] = None,
            task_alignment: Optional[Dict[str, str]] = None, held: Sequence[str] = ('confirm',), seed: int = 0,
            n_perm: Optional[int] = None, n_split: Optional[int] = None, n_boot: Optional[int] = None,
            split: str = 'discovery', whiten: bool = True, progress: Optional[Callable[[str], None]] = None
            ) -> Dict[str, Any]:
    """纠缠谱。E1–E4 在 split（缺省 discovery）上计算；E5 与门控用 discovery 拟合、held 检验。"""
    C = CRITERIA
    tags = list(tags or field.tags())
    task_alignment = task_alignment or {}
    idx = np.where(design.keep & (design.split == split))[0]
    out_tags = []
    for ti, (r, li) in enumerate(tags):
        if progress:
            progress(f'{r}@L{li} ({ti + 1}/{len(tags)})')
        X = np.asarray(field.X(r, li), dtype=np.float64)
        W, wnote = spectrum_whitener(X, design, split, whiten)
        Wr = (fit_whitener(X, design, np.where(design.keep & (design.split == 'discovery'))[0], C['whiten_rank'],
                           C['whiten_shrink']) if whiten else identity_whitener(X.shape[1]))
        Zr = Wr(X - Wr.mean)                                   # E5 / 门控：discovery 拟合的白化（测试名词不参与）
        rec: Dict[str, Any] = {'readout': r, 'hs_index': int(li), 'pairs': {}, 'whitener': wnote}
        for A, B in pairs:
            L = layout(design, A, B, idx)
            key = f'{A}×{B}'
            if L is None or len(L.units) < C['min_units']:
                rec['pairs'][key] = {'insufficient': True,
                                     'reason': '设计不完整或重复单位不足（不对不平衡设计做分解）'}
                continue
            Yraw = cube(L, X)
            Y = W(Yraw)
            res = pair_spectrum(Y, L, seed=seed + 7919 * ti, n_perm=n_perm, n_split=n_split, Yraw=Yraw)
            if L.A != NOUN:
                res['E5'] = interference(Zr, design, L.A, L.B, held=held, seed=seed, n_boot=n_boot)
            if L.B != NOUN and L.kind == 'FF':
                res['E5_reverse'] = interference(Zr, design, L.B, L.A, held=held, seed=seed, n_boot=n_boot)
            for attr, al in task_alignment.items():
                tf, _, lv = al.partition('=')
                if {L.A, L.B} == {attr, tf}:
                    res['gating'] = gating(Zr, design, attr, tf, lv, held=held, seed=seed, n_boot=n_boot)
            rec['pairs'][key] = res
        out_tags.append(rec)
    _verdicts(out_tags)
    warnings = []
    np_eff = C['n_perm'] if n_perm is None else n_perm
    if 1.0 / (1 + np_eff) > C['alpha']:
        warnings.append(f'n_perm={np_eff} 时最小可达 p = {1 / (1 + np_eff):.3f} > alpha={C["alpha"]}：本次不可能判定「纠缠」，只作描述')
    return {'schema': SCHEMA, 'criteria': dict(C), 'split': split, 'held': list(held), 'whiten': whiten,
            'warnings': warnings,
            'pairs': [list(p) for p in pairs], 'task_alignment': dict(task_alignment),
            'n_items': int(len(idx)), 'tags': out_tags, 'findings': findings(out_tags),
            'note': ('候选稳定观测：E1–E4 来自 discovery，E5 / 门控在留出名词上检验；零分布按名词整体 / 句式块整体；'
                     '不等于机制证据，须另做干预')}


def _verdicts(out_tags: List[Dict[str, Any]]) -> None:
    """逐 (因子对, 标签) 判定：纠缠 / 仅 token 身份 / 未检出纠缠。基线 = 同一读出位置的 hs[0]。"""
    C = CRITERIA
    base: Dict[Tuple[str, str], Optional[float]] = {}
    for t in out_tags:
        if t['hs_index'] == 0:
            for k, p in t['pairs'].items():
                base[(t['readout'], k)] = None if p.get('insufficient') else p.get('E1')
    for t in out_tags:
        for k, p in t['pairs'].items():
            if p.get('insufficient'):
                continue
            e1, pI = p.get('E1'), p['p'].get('I')
            b = base.get((t['readout'], k))
            sig = e1 is not None and pI is not None and pI <= C['alpha'] and e1 >= C['e1_min']
            excess = None if (b is None or e1 is None) else e1 - b
            p['E1_baseline_hs0'] = b
            p['E1_excess'] = excess
            if not sig:
                v = 'additive'
            elif t['hs_index'] == 0 or (excess is not None and excess < C['e1_excess_min']):
                v = 'token_identity'
            else:
                v = 'entangled'
            p['verdict'] = v
            e4 = (p.get('E4') or {}).get('A_given_B') or {}
            p['rotated'] = bool(e4.get('rotation') is not None and e4['rotation'] >= C['rotation_min'])
            p['gain_modulated'] = bool(e4.get('gain_ratio') is not None and e4['gain_ratio'] >= C['gain_ratio_min'])
            e5 = (p.get('E5') or {}).get('centered') or {}
            ci = e5.get('ci95')
            p['interfered'] = bool(e5.get('interference') is not None and e5['interference'] >= C['interference_min']
                                   and ci is not None and ci[0] > 0)
            g = p.get('gating') or {}
            p['gated'] = bool(g.get('gain') is not None and g['gain'] >= C['gating_min']
                              and g.get('ci95') is not None and g['ci95'][0] > 0)


def findings(out_tags: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按因子对汇总：在哪些读出 × 层上检出纠缠，峰值处的谱、子空间与旋转。"""
    by_pair: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = {}
    for t in out_tags:
        for k, p in t['pairs'].items():
            if not p.get('insufficient'):
                by_pair.setdefault(k, []).append((t, p))
    out = []
    for k, rows in by_pair.items():
        ent = [(t, p) for t, p in rows if p['verdict'] == 'entangled']
        tok = [(t, p) for t, p in rows if p['verdict'] == 'token_identity']
        gated = [f'{t["readout"]}@L{t["hs_index"]}' for t, p in rows if p.get('gated')]
        interf = [f'{t["readout"]}@L{t["hs_index"]}' for t, p in rows if p.get('interfered')]
        if ent:
            t, p = max(ent, key=lambda tp: tp[1]['E1'])
            e3 = p['E3']
            out.append({'id': f'entangled:{k}', 'kind': 'supported',
                        'text': (f'{k}：在 {len(ent)}/{len(rows)} 个读出 × 层上检出超出 token 身份的纠缠；峰值 '
                                 f'{t["readout"]}@L{t["hs_index"]} 可靠交互比例 {p["E1"]:.3f}（hs[0] 基线 '
                                 f'{_f(p.get("E1_baseline_hs0"))}，p={_f(p["p"]["I"])}），显著秩 {p["E2"].get("signif_rank")}'
                                 f'/{p["E2"]["max_rank"]}，交互落在主效应子空间内的比例 {_f(e3.get("frac_I_in_main"))}'
                                 f'（随机基线 {_f(e3.get("frac_baseline"))}）'),
                        'where': [f'{t2["readout"]}@L{t2["hs_index"]}' for t2, _ in ent]})
        elif tok:
            out.append({'id': f'token:{k}', 'kind': 'warning',
                        'text': f'{k}：只在 {len(tok)} 个读出 × 层上检出交互，且未超出嵌入层 hs[0]（token 身份）基线',
                        'where': [f'{t2["readout"]}@L{t2["hs_index"]}' for t2, _ in tok]})
        else:
            out.append({'id': f'additive:{k}', 'kind': 'unsupported',
                        'text': f'{k}：未检出纠缠（交互能量在零分布内或可靠交互比例 < {CRITERIA["e1_min"]}），与加性可分一致',
                        'where': []})
        if gated:
            out.append({'id': f'gated:{k}', 'kind': 'supported', 'text': f'{k}：任务线索对齐时属性更可读出（门控）',
                        'where': gated})
        if interf:
            out.append({'id': f'interference:{k}', 'kind': 'supported',
                        'text': f'{k}：读出在未见组合上下降（居中后仍下降，非加性所致）', 'where': interf})
    return out


def _f(v, d=3):
    return '—' if v is None else f'{v:.{d}f}'


# ---------------------------------------------------------------------------------------------------------------------
# 部件归因
# ---------------------------------------------------------------------------------------------------------------------
def _effects(Y: np.ndarray, L: Layout, halves: List[Tuple[np.ndarray, np.ndarray]]):
    """立方体 → (全量 (Δa, Δb, I), [(两半效应)…])。"""
    full = decompose(_cell_means(Y, L.avail))
    hs = [(decompose(_cell_means(Y, L.avail, h1)), decompose(_cell_means(Y, L.avail, h2))) for h1, h2 in halves]
    return full, hs


def _own_e1(hs, nA: int, nB: int) -> Optional[float]:
    rA = rB = rI = 0.0
    for (a1, b1, i1), (a2, b2, i2) in hs:
        rA += nB * _rel(a1, a2); rB += nA * _rel(b1, b2); rI += _rel(i1, i2)
    den = max(rA, 0) + max(rB, 0) + max(rI, 0)
    return (max(rI, 0) / den) if den > 0 else None


def attribute(field, design: Design, pair: Sequence[str], readout: str, dest: Optional[int] = None, *,
              split: str = 'discovery', seed: int = 0, n_split: Optional[int] = None,
              task_alignment: Optional[Dict[str, str]] = None, whiten: bool = True, top: int = 12) -> Dict[str, Any]:
    """一个因子对、一个读出位置的部件归因（目的层 dest 的残差 = hs[0] + Σ_{b<dest} 部件）。"""
    C = CRITERIA
    n_split = C['n_split'] if n_split is None else n_split
    blocks = field.blocks(readout)
    if not blocks:
        return {'error': '该读出位置没有部件采集'}
    n_blocks = max(blocks) + 1
    dest = n_blocks if dest is None else int(dest)
    if any(b not in blocks for b in range(dest)):
        return {'error': f'目的层 {dest} 之前的块没有全部采集，残差分解不闭合'}
    idx = np.where(design.keep & (design.split == split))[0]
    A, B = pair
    L = layout(design, A, B, idx)
    if L is None:
        return {'error': '设计不完整'}
    nA, nB = len(L.a_levels), len(L.b_levels)
    rng = np.random.default_rng(seed)
    halves = [_halves(L, rng) for _ in range(n_split)]
    # 目的层残差：hs[dest]；dest = 末块之后且 hs[-1] 已过最终归一化时，用 hs[dest−1] + 末块部件重建预归一化残差
    if dest < field.n_hs - 1 or (dest == field.n_hs - 1 and not field.post_norm_last):
        XR = np.asarray(field.X(readout, dest), dtype=np.float64)
        dest_note = f'hs[{dest}]'
    else:
        Zl, Ml = field.comp(readout, dest - 1)
        XR = np.asarray(field.X(readout, dest - 1), dtype=np.float64) + Zl.astype(np.float64) @ field.W_O(dest - 1).T + Ml
        dest_note = f'hs[{dest - 1}] + 块 {dest - 1} 部件（预归一化重建）'
    Wh, wnote = spectrum_whitener(XR, design, split, whiten)
    (daR, dbR, IR), hsR = _effects(Wh(cube(L, XR)), L, halves)
    ssA, ssB, ssI = _energies(daR, dbR, IR)
    QA = _basis(daR, C['subspace_energy'], C['subspace_max_dim'])
    attr_main = None
    for attr, al in (task_alignment or {}).items():
        if {A, B} == {attr, al.partition('=')[0]}:
            attr_main = attr
    Qattr = _basis(daR if L.A == attr_main else dbR, C['subspace_energy'], C['subspace_max_dim']) if attr_main else None

    rows = []
    sumA, sumB, sumI = np.zeros_like(daR), np.zeros_like(dbR), np.zeros_like(IR)
    e1_dest = _own_e1(hsR, nA, nB)

    def _plus(x, y):
        """两组效应（全量与两半）逐项相加。"""
        if x is None:
            return y
        (f1, h1), (f2, h2) = x, y
        return (tuple(u + v for u, v in zip(f1, f2)),
                [(tuple(u + v for u, v in zip(p1, q1)), tuple(u + v for u, v in zip(p2, q2)))
                 for (p1, p2), (q1, q2) in zip(h1, h2)])

    def add(name, block, Yc, inc):
        """inc：该部件输入处残差的效应（全量, 两半）；返回本部件的效应。"""
        nonlocal sumA, sumB, sumI
        eff = _effects(Yc, L, halves)
        (da, db, I), hs = eff
        sumA += da; sumB += db; sumI += I
        e = {'component': name, 'block': block,
             'share_A': float(nB * (da * daR).sum() / ssA) if ssA > 0 else None,
             'share_B': float(nA * (db * dbR).sum() / ssB) if ssB > 0 else None,
             'share_I': float((I * IR).sum() / ssI) if ssI > 0 else None,
             'own_E1': _own_e1(hs, nA, nB), 'I_norm_ratio': float(np.sqrt((I * I).sum() / ssI)) if ssI > 0 else None}
        if inc is not None:
            # 只有输入处确有可靠交互（分半可靠交互比例 ≥ e1_min，且能量 ≥ cancel_in_min × 目的层交互）时，「抵消比例」才有意义
            I_in = inc[0][2]
            nin = float((I_in * I_in).sum())
            e['I_in_E1'] = _own_e1(inc[1], nA, nB)
            e['I_in_ratio'] = nin / ssI if ssI > 0 else None
            ok = (nin > 1e-18 and ssI > 0 and nin >= C['cancel_in_min'] * ssI
                  and (e['I_in_E1'] or 0) >= C['e1_min'])
            e['cancel'] = float(-(I * I_in).sum() / nin) if ok else None
        if Qattr is not None and Qattr.shape[1]:
            e['I_in_attr_main'] = _frac_in(I, Qattr)
        rows.append(e)
        return eff

    X0 = np.asarray(field.X(readout, 0), dtype=np.float64)
    run = add('emb', -1, Wh(cube(L, X0)), None)                       # 残差中已有的效应（逐块累加）
    for b in range(dest):
        Z, Mo = field.comp(readout, b)
        Wb = field.W_O(b).astype(np.float64)                              # (d, H·hd)
        Wt = Wh(Wb.T)                                                     # (H·hd, d)：每行 = 白化后的写入方向
        Yz = cube(L, Z.astype(np.float64))                                # (R, A, B, H·hd)
        hd = field.head_dim
        inc, attn = run, None
        for h in range(field.n_heads):
            sl = slice(h * hd, (h + 1) * hd)
            attn = _plus(attn, add(f'B{b}.H{h}', b, Yz[..., sl] @ Wt[sl], inc))
        mlp_in = _plus(inc, attn)
        run = _plus(mlp_in, add(f'B{b}.MLP', b, Wh(cube(L, Mo.astype(np.float64))), mlp_in))
    closure = {'A': _relerr(sumA, daR), 'B': _relerr(sumB, dbR), 'I': _relerr(sumI, IR)}
    C_ = CRITERIA
    dest_ok = e1_dest is not None and e1_dest >= C_['e1_min']
    for e in rows:
        e['labels'] = []
        if e['component'] == 'emb':                                       # 嵌入层 = token 身份基线，不作部件标签
            continue
        sI = e.get('share_I')
        if (dest_ok and sI is not None and sI >= C_['share_min']
                and (e.get('own_E1') or 0) >= C_['own_e1_min']):
            e['labels'].append('entangler')
        mains = [x for x in (e.get('share_A'), e.get('share_B')) if x is not None]
        if e.get('cancel') is not None and e['cancel'] >= C_['cancel_min'] and mains and max(mains) >= C_['share_min']:
            e['labels'].append('disentangler')
        if (dest_ok and attr_main and sI is not None and sI >= C_['share_min']
                and (e.get('I_in_attr_main') or 0) >= C_['gate_in_main_min']):
            e['labels'].append('gater')
    ranked = sorted(rows, key=lambda e: -abs(e.get('share_I') or 0))
    return {'schema': SCHEMA + '/attribution', 'pair': [L.A, L.B], 'layout': L.kind, 'readout': readout,
            'dest': dest, 'dest_note': dest_note, 'split': split, 'units': len(L.units), 'whitener': wnote,
            'residual': {'ss': {'A': ssA, 'B': ssB, 'I': ssI}, 'E1_rel': e1_dest,
                         'labels_enabled': bool(e1_dest is not None and e1_dest >= CRITERIA['e1_min'])},
            'closure_rel_err': closure, 'attr_aligned': attr_main, 'criteria': dict(CRITERIA),
            'top_by_share_I': ranked[:top],
            'entanglers': [e for e in rows if 'entangler' in e['labels']],
            'disentanglers': [e for e in rows if 'disentangler' in e['labels']],
            'gaters': [e for e in rows if 'gater' in e['labels']],
            'components': rows,
            'note': '份额 = 部件写入向量与目的层残差同名效应的内积占比（目的层白化空间，全部份额相加 = 1 − 闭合误差）；'
                    '标签为描述性候选，不等于因果必要性'}


def _relerr(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    n = float(np.sqrt((b * b).sum()))
    return float(np.sqrt(((a - b) ** 2).sum()) / n) if n > 1e-12 else None


# ---------------------------------------------------------------------------------------------------------------------
# 合成场（已知答案）：低秩纠缠 + 由线索门控的解纠缠；planted=False 为纯加性负对照
# ---------------------------------------------------------------------------------------------------------------------
def synthetic_field(seed: int = 0, planted: bool = True, *, n_cat: int = 4, per_cat: int = 15, d: int = 48,
                    n_blocks: int = 6, n_heads: int = 4, head_dim: int = 6, noise: float = 0.3):
    """返回 (ArrayField, Design, truth)。读出位置只有 'target'；hs[0] = 名词在各语言中的 token 嵌入（语言特异）。

    残差逐块累加：hs[b+1] = hs[b] + Σ_h W_{b,h} z_{b,h} + MLP_b（精确闭合）。各块写入（所有条件都有）：
      块 0 MLP：语言无关的名词概念码 g_u（大于 token 嵌入）+ 类别、语言、极性、任务线索方向（加性）；
      块 1 头 0：颜色属性码 W_{1,0} c_color（加性，两种语言相同）。
    planted=True 另植入：
      块 3 头 1（纠缠者）：z = sign(lang)·F[color]，F 按颜色中心化 → 颜色 × 语言的秩 2 交互，写在新方向上；
      块 4 MLP（解纠缠者，由线索门控）：只在 task=color 时写入 −W_{3,1}·sign(lang)·F[color] + β·W_{1,0} c_color，
        抵消颜色 × 语言交互并放大颜色主效应（颜色 × 任务线索的交互 = 增益调制）。
    每个块的每个部件另加独立噪声（noise / √d 每维）。只用于检验方法能否找回植入结构，不代表任何真实模型。"""
    rng = np.random.default_rng(seed)
    colors = ['c0', 'c1', 'c2']
    nouns, cats, cols = [], [], []
    for c in range(n_cat):
        for k in range(per_cat):
            nouns.append(f'cat{c}.n{k}'); cats.append(f'cat{c}'); cols.append(colors[(k + c) % 3])
    U = len(nouns)
    # 按名词切分 60/20/20（每类内）
    split_of = {}
    for c in range(n_cat):
        ns = [n for n, cc in zip(nouns, cats) if cc == f'cat{c}']
        ns = [ns[i] for i in rng.permutation(len(ns))]
        nd, ncf = int(0.6 * len(ns)), int(0.2 * len(ns))
        for i, n in enumerate(ns):
            split_of[n] = 'discovery' if i < nd else ('confirm' if i < nd + ncf else 'lockbox')
    frames = [(lang, pol, task, var) for lang in ('zh', 'en') for pol in ('aff', 'neg')
              for task in ('color', 'size', 'none') for var in ('v1', 'v2')]

    def unit(*shape):
        v = rng.standard_normal(shape)
        return v / np.linalg.norm(v, axis=-1, keepdims=True)
    tok = {(n, lang): unit(d) * 1.0 for n in nouns for lang in ('zh', 'en')}
    g = {n: unit(d) * 2.0 for n in nouns}
    dir_cat = {c: unit(d) * 1.0 for c in set(cats)}
    dir_lang = {'zh': unit(d) * 1.0, 'en': unit(d) * 1.0}
    dir_pol = {'aff': unit(d) * 0.8, 'neg': unit(d) * 0.8}
    dir_task = {t: unit(d) * 0.8 for t in ('color', 'size', 'none')}
    W = {b: rng.standard_normal((d, n_heads * head_dim)) / np.sqrt(head_dim) for b in range(n_blocks)}
    ccode = {c: rng.standard_normal(head_dim) for c in colors}
    mc = np.mean(list(ccode.values()), axis=0)
    ccode = {c: (v - mc) / np.linalg.norm(v - mc) * 1.2 for c, v in ccode.items()}
    F = {c: rng.standard_normal(head_dim) for c in colors}
    mf = np.mean(list(F.values()), axis=0)
    F = {c: (v - mf) / np.linalg.norm(v - mf) * 1.2 for c, v in F.items()}
    beta = 1.5
    hd = head_dim

    def wh(b, h, z):
        return W[b][:, h * hd:(h + 1) * hd] @ z

    ids, cond, frame_ids, fill, splits = [], [], [], [], []
    N = U * len(frames)
    hs = {i: np.zeros((N, d)) for i in range(n_blocks + 1)}
    Zs = {b: np.zeros((N, n_heads * head_dim)) for b in range(n_blocks)}
    Ms = {b: np.zeros((N, d)) for b in range(n_blocks)}
    k = 0
    for ui, n in enumerate(nouns):
        for (lang, pol, task, var) in frames:
            fid = f'{lang}-{pol}-{task}-{var}'
            ids.append(f'SYN:{fid}:{n}'); frame_ids.append(fid); fill.append(n); splits.append(split_of[n])
            cond.append({'cat': cats[ui], 'lang': lang, 'polarity': pol, 'task': task})
            x = tok[(n, lang)] + rng.standard_normal(d) * noise / np.sqrt(d)
            hs[0][k] = x
            sgn = 1.0 if lang == 'zh' else -1.0
            for b in range(n_blocks):
                z = rng.standard_normal(n_heads * head_dim) * noise / np.sqrt(head_dim)
                m = rng.standard_normal(d) * noise / np.sqrt(d)
                if b == 0:
                    m += g[n] + dir_cat[cats[ui]] + dir_lang[lang] + dir_pol[pol] + dir_task[task]
                if b == 1:
                    z[0 * hd:1 * hd] += ccode[cols[ui]]
                if planted and b == 3:
                    z[1 * hd:2 * hd] += sgn * F[cols[ui]]
                if planted and b == 4 and task == 'color':
                    m += -wh(3, 1, sgn * F[cols[ui]]) + beta * wh(1, 0, ccode[cols[ui]])
                Zs[b][k], Ms[b][k] = z, m
                x = x + W[b] @ z + m
                hs[b + 1][k] = x
            k += 1
    field = ArrayField({('target', i): hs[i] for i in hs}, {('target', b): (Zs[b], Ms[b]) for b in range(n_blocks)},
                       W, n_heads=n_heads, head_dim=head_dim, post_norm_last=False)
    spec = {'factors': ['cat', 'lang', 'polarity', 'task'], 'entangle': {'attrs': ['color']},
            'fillers': [{'id': n, 'attrs': {'color': c}} for n, c in zip(nouns, cols)],
            'frames': [{'id': f'{a}-{b}-{c}-{v}', 'variant': v} for a, b, c, v in frames]}
    design = build_design(ids, cond, frame_ids, fill, splits, None, spec)
    truth = {'planted': planted, 'entangler': 'B3.H1', 'disentangler': 'B4.MLP', 'color_attr_head': 'B1.H0',
             'entangled_pair': ['color', 'lang'], 'gated_pair': ['color', 'task'], 'planted_rank': 2,
             'entangle_from_hs': 4, 'n_blocks': n_blocks}
    return field, design, truth


# ---------------------------------------------------------------------------------------------------------------------
# 命令行：节点本地原生全场 → 纠缠谱 / 部件归因 JSON
# ---------------------------------------------------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import glob
    from . import grammar
    ap = argparse.ArgumentParser(description='entangle.v1：纠缠谱与部件归因（读取节点本地原生全场）')
    ap.add_argument('--spec', required=True, help='文法 JSON（与采集时冻结的版本一致）')
    ap.add_argument('--native', required=True, nargs='+', help='原生全场分片目录（shard-*of*），可用通配符')
    ap.add_argument('--behavior', nargs='*', default=[], help='各分片 outputs/behavior.json（逐名词行为门）')
    ap.add_argument('--out', required=True)
    ap.add_argument('--layer-stride', type=int, default=1, help='纠缠谱只算每隔 k 层（另总含 hs[0] 与末层）')
    ap.add_argument('--attribute', action='store_true', help='另做部件归因（需要 components/）')
    ap.add_argument('--dest', type=int, default=None, help='部件归因的目的层（缺省 = 末块之后）')
    ap.add_argument('--held', nargs='+', default=['confirm'])
    ap.add_argument('--n-perm', type=int, default=None)
    ap.add_argument('--gate', default=None, help='逐名词行为门策略（缺省取文法 entangle.noun_gate）')
    args = ap.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text(encoding='utf-8'))
    dirs = sorted({p for pat in args.native for p in glob.glob(pat)})
    field = NativeField(dirs)
    corr = {}
    for p in args.behavior:
        for b in json.loads(Path(p).read_text(encoding='utf-8')).get('items', []):
            corr[b['id']] = bool(b.get('correct'))
    corpus = grammar.generate(spec, include_canary=False)
    design = design_from_corpus(corpus, spec, field.ids, corr if args.behavior else None)
    ent = spec.get('entangle') or {}
    design, gate_rep = gate_nouns(design, args.gate or ent.get('noun_gate', 'all_probes') if args.behavior else 'none')
    pairs = [tuple(p) for p in ent.get('pairs') or []]
    last = field.n_hs - 1
    layers = sorted(set(list(range(0, field.n_hs, max(args.layer_stride, 1))) + [0, last]))
    tags = [(r, i) for r in field.readouts for i in layers]
    res = analyze(field, design, pairs, tags=tags, task_alignment=ent.get('task_alignment'), held=args.held,
                  n_perm=args.n_perm, progress=lambda s: print('[entangle]', s, flush=True))
    res['gate'] = gate_rep
    res['source'] = {'spec_sha256': grammar.spec_sha(spec), 'native_dirs': dirs,
                     'manifest_sha256': [hashlib.sha256((Path(d) / 'manifest.json').read_bytes()).hexdigest()
                                         for d in dirs]}
    if args.attribute:
        res['attribution'] = [attribute(field, design, p, r, args.dest, task_alignment=ent.get('task_alignment'))
                              for r in field.readouts if field.blocks(r) for p in pairs]
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_json_default), encoding='utf-8')
    print('[entangle] wrote', args.out)
    return 0


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


if __name__ == '__main__':
    raise SystemExit(main())
