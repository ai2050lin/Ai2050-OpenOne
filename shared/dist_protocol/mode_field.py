# -*- coding: utf-8 -*-
"""风格模式场（mode_field.v1）：MF-v1 的统计与判据，纯 numpy；判据固定在 CRITERIA 并随结果返回。

核心约束：风格与词身份高度纠缠（汉字与代码符号在嵌入层就不同），所以「风格可读出」只在跨风格共用 token 的位置上判定——
同一 token 的嵌入层状态在各风格中完全相同，那里读出的风格差异只能来自上下文计算。写入方向也只用共用 token 位置按 token 配平求得。
"""
from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, List, Sequence

import numpy as np

SCHEMA = 'mode_field.v1'
STYLES = ['zh', 'en', 'paper', 'code']
CRITERIA = {
    'schema': SCHEMA,
    'layers_stored': 'range(0, L + 1, 3) 全部位置；共用 token 位置另存全部层',
    'lambdas_rel': [1e-3, 1e-2, 1e-1, 1.0],
    'excl_min_count': 3,          # 风格专属词表：发现集中该风格出现 ≥ 3 次、其他风格 0 次的 token
    'H1_pair_acc_min': 0.8,       # 确认集共用 token 位置：6 个风格对的两两准确率都 ≥ 0.8（层与 λ 按发现集分组交叉验证选定）
    'k_grid': [-6, -3, 0, 3, 6], 'n_random': 2, 'min_swing': 1.0,   # H2：在散文前缀全部位置沿代码方向写入，log M_code 的 steer_verdict
    'H2_sources': ['zh', 'en', 'paper'],
    'top_k': 10, 'n_random_sets': 5,
    'H3_ratio_min': 3.0,          # H3：消融前 10 个「代码模式写入头」，代码文本的 log M_code 与读出间隔的平均下降都 ≥ 同层随机集合最大下降的 3 倍
    'H4_auc_min': 0.8,            # H4：代码方向经最终归一化与 unembedding 后，区分代码专属与其他专属 token 的 AUC
    'H5': '描述：代码文本中单个位置写入 −6σ 代码方向后，log M_code 变化随后续位置的衰减（半衰期）',
    'seed': 0,
}


def exclusive_vocab(tokens_by_style: Dict[str, Sequence[int]], min_count: int) -> Dict[str, List[int]]:
    cnt = {s: Counter(t) for s, t in tokens_by_style.items()}
    out = {}
    for s, c in cnt.items():
        others = set().union(*(cnt[o].keys() for o in cnt if o != s))
        out[s] = sorted(t for t, n in c.items() if n >= min_count and t not in others)
    return out


def shared_tokens(tok: np.ndarray, y: np.ndarray, a: int, b: int) -> set:
    return set(tok[y == a].tolist()) & set(tok[y == b].tolist())


def ridge_multi(X: np.ndarray, y: np.ndarray, n_cls: int, lam_rel: float):
    """一对其余的 ±1 岭回归（原始形式，X 已中心化在内部）。返回 (W, mu, b)。"""
    mu = X.mean(0)
    Xc = X - mu
    Y = np.where(y[:, None] == np.arange(n_cls)[None], 1.0, -1.0)
    G = Xc.T @ Xc
    lam = lam_rel * np.trace(G) / G.shape[0]
    W = np.linalg.solve(G + lam * np.eye(G.shape[0]), Xc.T @ (Y - Y.mean(0)))
    return W, mu, Y.mean(0)


def scores(model, X):
    W, mu, b = model
    return (X - mu) @ W + b


def pairwise_acc(S: np.ndarray, y: np.ndarray, tok: np.ndarray, train_tok: np.ndarray = None, train_y: np.ndarray = None,
                 n_cls: int = 4) -> Dict[str, float]:
    """每个风格对 (a, b)：只取 token 在本集合中同时出现于 a 与 b 的位置，按 S[:, a] > S[:, b] 判 a，平衡准确率。"""
    out = {}
    for a, b in combinations(range(n_cls), 2):
        sh = shared_tokens(tok, y, a, b)
        m = np.isin(tok, list(sh)) & np.isin(y, [a, b])
        if m.sum() == 0:
            out[f'{a}-{b}'] = float('nan')
            continue
        pred_a = S[m, a] > S[m, b]
        ya = y[m] == a
        out[f'{a}-{b}'] = float(0.5 * (pred_a[ya].mean() + (~pred_a[~ya]).mean()))
    return out


def token_matched_direction(X: np.ndarray, y: np.ndarray, tok: np.ndarray, target: int) -> np.ndarray:
    """共用 token 配平的方向：对同时出现在 target 与其他风格的每个 token，取 (target 均值 − 其他均值)，再按 token 平均。"""
    diffs, w = [], []
    for t in np.unique(tok):
        m = tok == t
        a, o = m & (y == target), m & (y != target)
        if a.any() and o.any():
            diffs.append(X[a].mean(0) - X[o].mean(0))
            w.append(min(a.sum(), o.sum()))
    if not diffs:
        return np.zeros(X.shape[1])
    return np.average(np.stack(diffs), axis=0, weights=np.asarray(w, float))


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    allv = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(allv)) + 1.0
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def half_life(curve: Sequence[float]) -> float:
    """curve[0] 为写入位置的效应；返回效应绝对值首次降到一半以下的偏移（线性插值）；始终不降则返回长度。"""
    c = np.abs(np.asarray(curve, float))
    if c[0] <= 0:
        return float('nan')
    for i in range(1, len(c)):
        if c[i] < c[0] / 2:
            return float(i - 1 + (c[i - 1] - c[0] / 2) / max(c[i - 1] - c[i], 1e-12))
    return float(len(c))


def grouped_folds(groups: np.ndarray, k: int, seed: int):
    ug = np.unique(groups)
    parts = np.array_split(np.random.default_rng(seed).permutation(ug), k)
    return [np.isin(groups, p) for p in parts]


def select_layer(Xs: Dict[int, np.ndarray], y, tok, groups, crit=CRITERIA):
    """发现集按组（任务）5 折：每层每 λ 的共用 token 两两准确率均值，取最大。"""
    best = None
    folds = grouped_folds(groups, 5, crit['seed'])
    table = {}
    for l, X in Xs.items():
        for lam in crit['lambdas_rel']:
            accs = []
            for te in folds:
                m = ridge_multi(X[~te], y[~te], 4, lam)
                accs.append(np.nanmean(list(pairwise_acc(scores(m, X[te]), y[te], tok[te]).values())))
            v = float(np.mean(accs))
            table[(l, lam)] = v
            if best is None or v > best[0]:
                best = (v, l, lam)
    return {'cv_acc': best[0], 'layer': best[1], 'lambda_rel': best[2], 'table': {f'{l}/{lam}': v for (l, lam), v in table.items()}}


# ---------------- MF-v2：风格状态的汇集层段、各风格的单侧开关与叠加 ----------------
# 汇集：在一段层上把注意力限制为局部窗口（当前及前 w−1 个位置，外加首位置作注意力汇点），看后段读出层上「深共用」位置的风格读出损失多少。
# 开关：后段（0.5L）每种风格按共用 token 配平的「本风格 − 其他」方向，只在正侧判定（MF-v1 显示负侧有底部效应）。
# 叠加：同时打开两种风格时，各自的风格质量是否都保留单独打开时的效应（专家乘积式叠加），还是只剩一个（胜者通吃）。
CRITERIA_V2 = {
    'schema': 'mode_field.v2',
    'window': 4,                   # 局部窗口：当前及前 3 个位置 + 首位置
    'deep_n': 4,                   # 深共用位置：当前与前 3 个 token 都是共用 token
    'readout_rel': 0.75,           # 读出层 = 网格层中最接近 0.75L 者
    'n_bands': 6,                  # 读出层之前的层均分为 6 段
    'H1_drop_min': 0.2,            # 全部前段层局部窗口：深共用位置两两准确率均值下降 ≥ 0.2
    'H2_band_share_min': 0.5,      # 下降最大的单段 ≥ 全部下降的 50%
    'switch_rel': 0.5,             # 开关写入层 = round(0.5L)
    'k_on': [0, 3, 6], 'n_random': 2,
    'H3_min_combos': 10,           # 12 个（目标风格 × 来源风格）中，Δ(+6) ≥ 1、Δ(+6) > Δ(+3) > 0、Δ(+6) > 每个随机方向 Δ(+6) 的至少 10 个
    'H4_retain_min': 0.5,          # 叠加：两种目标都保留单独打开效应的 ≥ 50%
    'H4_min_triples': 8,           # 12 个（来源 c，目标对 {a, b}）中至少 8 个
    'seed': 0,
}


def window_mask(mask, w: int, q_from: int = 0):
    """mask：[B, 1, T, T] 加性因果掩码（torch）。返回新掩码：查询位置 q 只看 [q − w + 1, q] 与首位置 0。"""
    import torch
    T = mask.shape[-1]
    q = torch.arange(T, device=mask.device)[:, None]
    k = torch.arange(T, device=mask.device)[None, :]
    block = (k < q - w + 1) & (k != 0) & (q >= q_from)
    out = mask.clone()
    out[..., block] = torch.finfo(mask.dtype).min
    return out


def band_localize(full_drop: float, band_drops: Sequence[float], crit=CRITERIA_V2) -> Dict[str, object]:
    b = int(np.argmax(band_drops))
    share = float(band_drops[b] / full_drop) if full_drop > 0 else float('nan')
    return {'band': b, 'share': share, 'pass': bool(full_drop > 0 and share >= crit['H2_band_share_min'])}


def one_sided_verdict(d_op: np.ndarray, d_rand: np.ndarray) -> Dict[str, object]:
    """d_op：[n, 3]（k = 0, 3, 6）；d_rand：[n_rand, n, 3]。只看正侧。"""
    m = d_op.mean(0)
    r6 = [float(r.mean(0)[2]) for r in d_rand]
    ok = bool(m[2] >= 1.0 and m[2] > m[1] > 0 and m[2] > max(r6))
    return {'delta_k3': float(m[1]), 'delta_k6': float(m[2]), 'random_k6': r6, 'pass': ok}


def retain(joint_a: float, single_a: float, joint_b: float, single_b: float, crit=CRITERIA_V2) -> Dict[str, object]:
    ra = joint_a / single_a if single_a > 0 else float('nan')
    rb = joint_b / single_b if single_b > 0 else float('nan')
    return {'retain_a': float(ra), 'retain_b': float(rb),
            'pass': bool(single_a > 0 and single_b > 0 and ra >= crit['H4_retain_min'] and rb >= crit['H4_retain_min'])}


# ---------------- MF-v3：代码压过散文的位置（组件修补）----------------
# 在「代码 + 散文」同时写入的运行里，把某一层的注意力或 MLP 输出（全部位置）换成「只写散文」运行中的值，
# 看被压掉的散文效应恢复多少：恢复比例 = (M_p[修补] − M_p[同时]) / (M_p[只散文] − M_p[同时])。
# 组件只在 A 半（按任务）上选定，在 B 半上检验。
CRITERIA_V3 = {
    'schema': 'mode_field.v3',
    'triples': [['zh', 'en'], ['en', 'zh'], ['paper', 'zh'], ['paper', 'en']],   # [来源, 被压的散文]；与代码同时写入（MF-v2 两模型都压掉的 4 组）
    'k': 6, 'switch_rel': 0.5,
    'H0_prose_retain_max': 0.2, 'H0_code_retain_min': 0.7, 'H0_min_triples': 3,   # 发现集文本上复现压制
    'H1_restore_min': 0.3, 'H1_min_triples': 3,      # A 选出的单个组件在 B 上恢复 ≥ 0.3
    'H2_min_triples': 3,                             # B 上 MLP 组件的正恢复之和 > 注意力组件的正恢复之和
    'seed': 0,
}


def restoration(m_patched: np.ndarray, m_joint: np.ndarray, m_single: np.ndarray) -> np.ndarray:
    """逐条目平均后的恢复比例：m_patched [n, n_comp]，m_joint / m_single [n]。"""
    gap = float(np.mean(m_single - m_joint))
    if gap <= 0:
        return np.full(m_patched.shape[1], np.nan)
    return (m_patched - m_joint[:, None]).mean(0) / gap


def select_component(m_patched, m_joint, m_single, in_A, names, crit=CRITERIA_V3) -> Dict[str, object]:
    rA = restoration(m_patched[in_A], m_joint[in_A], m_single[in_A])
    rB = restoration(m_patched[~in_A], m_joint[~in_A], m_single[~in_A])
    j = int(np.nanargmax(rA))
    pos = lambda r, kind: float(np.nansum([max(v, 0) for v, n in zip(r, names) if n.startswith(kind)]))
    return {'component': names[j], 'restore_A': float(rA[j]), 'restore_B': float(rB[j]),
            'mlp_pos_B': pos(rB, 'mlp'), 'attn_pos_B': pos(rB, 'attn'),
            'top5_B': sorted(((float(v), n) for v, n in zip(rB, names) if v == v), reverse=True)[:5],
            'pass': bool(rB[j] >= crit['H1_restore_min'])}
