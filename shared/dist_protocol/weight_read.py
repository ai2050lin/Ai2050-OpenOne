# -*- coding: utf-8 -*-
"""权重阅读与封闭式编辑（weight_read.v1，WS-v1）：只读权重提出假设，少量前向判定。纯 numpy；判据固定在 CRITERIA。

参考「权重空间编程」（Takehi 等，2026-11）的做法：先从权重结构推出部件读什么、写什么、由谁供给，再用前向与一次封闭式权重编辑检验。
  读出向量：MLP 单元 i 的 gate / up 行乘以该块 post_attention_layernorm 的权重 γ，即残差空间里它读的方向 a_i = γ ⊙ W[i]。
  组合分数（Elhage 等 2021 的 Frobenius 形式）：注意力头 h 的 OV 矩阵 W_OV = W_O^h W_V^h diag(γ_in) 写入的内容被读出矩阵 R 读到的比例
      CS_h = ‖R W_OV‖_F / (‖R‖_F ‖W_OV‖_F)；只依赖权重，不看头注意哪里。
  方向移除：a' = a − (a·v) v，W' = a' / γ（单位向量 v；只改被选单元的行）。
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

SCHEMA = 'weight_read.v1'
CRITERIA = {
    'schema': SCHEMA,
    'n_random_sets': 200,
    'H1_read_ratio_min': 3.0, 'H1_p_max': 0.01,          # 32 个门控单元读代码方向的平均分数 ≥ 同层随机集合的 3 倍，经验 p ≤ 0.01
    'H2_spearman_min': 0.3, 'H2_top_k': 16, 'H2_p_max': 0.05,  # 组合分数与前向实测 |T_h| 的 Spearman ≥ 0.3，且前 16 重合置换 p ≤ 0.05，且优于范数基线
    'H3_restore_frac_min': 0.5,                           # 封闭式编辑的恢复 ≥ MF-v4 激活修补恢复的 0.5 倍
    'H3_control_max_frac': 1 / 3,                         # 随机方向编辑、随机单元编辑各自的恢复 ≤ 真实编辑的 1/3
    'H4_nll_rise_max': 0.01,                              # 自然文本平均 NLL 相对上升 ≤ 1%（全部与代码文本分别）
    'n_perm': 2000,
    'seed': 0,
}


def unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def cosines(A: np.ndarray, v: np.ndarray) -> np.ndarray:
    """A：[n, d] 行向量；返回每行与 v 的余弦。"""
    return (A @ unit(v)) / (np.linalg.norm(A, axis=1) + 1e-12)


def read_scores(Ag: np.ndarray, Au: np.ndarray, v: np.ndarray) -> np.ndarray:
    """gate 与 up 读出向量对方向 v 的合并读分数 sqrt(cos_g² + cos_u²)。"""
    return np.sqrt(cosines(Ag, v) ** 2 + cosines(Au, v) ** 2)


def ratio_vs_random(score_top: float, score_rand: Sequence[float]) -> Dict[str, float]:
    r = np.asarray(score_rand, float)
    return {'top': float(score_top), 'random_mean': float(r.mean()), 'ratio': float(score_top / max(r.mean(), 1e-12)),
            'p': float((1 + (r >= score_top).sum()) / (1 + len(r)))}


def composition(R: np.ndarray, Wov: np.ndarray) -> float:
    """R：[k, d] 读出行；Wov：[d, d]（或低秩因子乘积）。"""
    return float(np.linalg.norm(R @ Wov) / (np.linalg.norm(R) * np.linalg.norm(Wov) + 1e-12))


def composition_lowrank(R: np.ndarray, Wo: np.ndarray, Wv: np.ndarray) -> float:
    """W_OV = Wo @ Wv（Wo：[d, r]，Wv：[r, d]），不显式构造 d×d。"""
    RW = (R @ Wo) @ Wv
    _, s_o = np.linalg.qr(Wo)
    nov = np.linalg.norm(s_o @ Wv)                       # ‖Wo Wv‖_F = ‖R_o Wv‖_F（Wo = Q R_o）
    return float(np.linalg.norm(RW) / (np.linalg.norm(R) * nov + 1e-12))


def remove_direction(W_rows: np.ndarray, gamma: np.ndarray, v: np.ndarray) -> np.ndarray:
    """在残差读出空间里移除方向 v：a = γ ⊙ w，a' = a − (a·v̂) v̂，返回 w' = a' / γ。"""
    vh = unit(v)
    A = W_rows * gamma[None, :]
    A2 = A - np.outer(A @ vh, vh)
    return A2 / gamma[None, :]


def rankdata(x: np.ndarray) -> np.ndarray:
    o = np.argsort(x, kind='mergesort')
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    xs = np.asarray(x)[o]
    i = 0
    while i < len(x):                                   # 并列取平均秩
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(np.asarray(a, float)), rankdata(np.asarray(b, float)))[0, 1])


def topk_overlap(pred: np.ndarray, truth: np.ndarray, k: int, n_perm: int, seed: int) -> Dict[str, float]:
    """预测前 k 与真值前 k 的重合个数及置换 p（打乱预测分数）。"""
    tk = set(np.argsort(-truth)[:k].tolist())
    ov = len(tk & set(np.argsort(-pred)[:k].tolist()))
    rng = np.random.default_rng(seed)
    null = [len(tk & set(rng.permutation(len(pred))[:k].tolist())) for _ in range(n_perm)]
    return {'overlap': ov, 'k': k, 'null_mean': float(np.mean(null)), 'p': float((1 + sum(n >= ov for n in null)) / (1 + n_perm))}


def head_verdict(cs: np.ndarray, base: np.ndarray, truth: np.ndarray, crit=CRITERIA) -> Dict[str, object]:
    rho, rho_b = spearman(cs, truth), spearman(base, truth)
    ov = topk_overlap(cs, truth, crit['H2_top_k'], crit['n_perm'], crit['seed'])
    return {'spearman': rho, 'spearman_norm_baseline': rho_b, 'overlap': ov,
            'pass': bool(rho >= crit['H2_spearman_min'] and rho > rho_b and ov['p'] <= crit['H2_p_max'])}


# ---- WS-v2：上游头的封闭式写出编辑 ----
CRITERIA_V2 = {
    'schema': 'weight_read.v2',
    'n_heads': 8,                       # 发现集上直接贡献（推向代码）最大的 8 个头（块 ≤ 34）
    'n_random_sets': 5,                 # 同层随机头集合
    'H1_reduction_min': 0.2,            # 确认集门控得分「代码 − 散文」差的下降比例 ≥ 0.2，且 ≥ 随机集合平均的 3 倍
    'H1_over_random': 3.0,
    'H2_restore_min': 0.1,              # MF-v4 写入范式下恢复 ≥ 0.1，且 ≥ 随机集合平均的 3 倍
    'H2_over_random': 3.0,
    'H3_nll_rise_max': 0.01,            # 自然文本 NLL 相对上升 ≤ 1%（全部与代码文本分别）
    'locality_window': 16,
    'seed': 0,
}


def remove_write(Wo_h: np.ndarray, V: np.ndarray) -> np.ndarray:
    """从头的写出矩阵 Wo_h：[d, r] 中移除写向子空间 span(V) 的分量（V：[k, d]，行可不正交）。"""
    Q, _ = np.linalg.qr(np.atleast_2d(V).T)
    return Wo_h - Q @ (Q.T @ Wo_h)


def matched_heads(sel, n_heads: int, n_sets: int, seed: int):
    """sel：[(块, 头)]；每个头换成同块另一个未被选中的头。"""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_sets):
        s = []
        for lam, h in sel:
            pool = [x for x in range(n_heads) if (lam, x) not in sel and (lam, x) not in s]
            s.append((lam, int(rng.choice(pool))))
        out.append(s)
    return out


def over_random(v: float, rand, ratio: float, vmin: float) -> Dict[str, object]:
    rm = float(np.mean(rand))
    return {'value': float(v), 'random': [float(x) for x in rand], 'random_mean': rm,
            'pass': bool(v >= vmin and v >= ratio * max(rm, 0.0))}


def locality(attn_row: np.ndarray, t: int, window: int) -> float:
    """位置 t 的注意力行（长度 ≥ t+1）：去掉位置 0（汇聚点）后，落在最近 window 个位置（含自身）的比例。"""
    a = np.asarray(attn_row[1:t + 1], float)
    if a.sum() <= 0:
        return float('nan')
    return float(a[max(0, t - window):].sum() / a.sum())


# ---- WS-v3：上游头的 QK 侧（看哪里、敲掉会怎样） ----
CRITERIA_V3 = {
    'schema': 'weight_read.v3',
    'H1_lookback_min': 1.5,           # md 文本结语散文位置：代码段 token 的注意力质量 / 代码段占可见 token 份额 ≥ 1.5，且高于随机头集合平均
    'H2_syntax_min': 2.0,             # 代码位置：语法骨架 token 的注意力质量 / 其份额 ≥ 2，且高于随机头集合平均
    'H3_drop_min': 0.10,              # 只禁止 8 个头注意语法 token：确认集门控得分差下降 ≥ 10%，
    'H3_over_control': 2.0,           #   且 ≥ 2 × 等数随机 token 禁止的下降、≥ 2 × 随机头禁止语法 token 的下降（5 组平均）
    'min_pos': 8,
    'seed': 0,
}
KEYWORDS = {'return', 'if', 'while', 'for', 'def', 'else', 'elif', 'in', 'and', 'or', 'not', 'import', 'from', 'class', 'True',
            'False', 'None', 'break', 'continue', 'pass', 'lambda', 'yield', 'with', 'as', 'try', 'except'}
OPERATORS = {'=', '+=', '-=', '*=', '/=', '==', '!=', '<=', '>=', '<', '>', '+', '-', '*', '/', '//', '%', '**'}


def is_syntax(tok_str: str) -> bool:
    """预注册的代码语法骨架 token：块头（含冒号且以换行结尾）、Python 关键字、运算符（去掉首尾空白后比较）。"""
    s = tok_str.strip(' \t')
    return (':' in s and tok_str.endswith('\n')) or s.strip() in KEYWORDS or s.strip() in OPERATORS


def mass_ratio(attn_row: np.ndarray, t: int, cls: np.ndarray) -> float:
    """位置 t 的注意力行；cls：长度 ≥ t+1 的布尔类别。去掉位置 0 后，类别上的注意力比例 / 类别在 1..t 中的份额。"""
    a = np.asarray(attn_row[1:t + 1], float)
    c = np.asarray(cls[1:t + 1], bool)
    if a.sum() <= 0 or not c.any() or c.all():
        return float('nan')
    return float((a[c].sum() / a.sum()) / c.mean())


# ---- WS-v4：模式匹配的 QK ----
CRITERIA_V4 = {
    'schema': 'weight_read.v4',
    'rope_deltas': list(range(1, 33)),   # 相对距离 1..32 上平均 RoPE 的双线性作用
    'n_random_dirs': 200,
    'H1_z_min': 2.0,                     # 8 个头代码—代码 QK 对齐相对随机方向的平均 z ≥ 2，且高于随机头的平均 z
    'shift_k': [-4, -2, 0, 2, 4],        # 查询端沿代码方向的移动（单位：自然文本上该方向投影的标准差）
    'H2_ratio_min': 1.5,                 # 散文位置对代码 key 的质量比：+4 / 0 ≥ 1.5，且在 shift_k 上单调（散文与代码位置都单调不减）
    'H2_over_random': 3.0,               #   且 |log 效应| ≥ 3 × 随机方向同样移动的 |log 效应|
    'H3_key_ratio_max': 0.8,             # 代码位置对代码 key 的质量比：代码 key 移动 −4 后 / 原始 ≤ 0.8，且 |log 效应| ≥ 3 × 随机方向
    'seed': 0,
}


def rotate_half(x: np.ndarray) -> np.ndarray:
    h = x.shape[-1] // 2
    return np.concatenate([-x[..., h:], x[..., :h]], -1)


def rope_matrix(inv_freq: np.ndarray, pos: float) -> np.ndarray:
    """HF 约定的 RoPE：rope(x, p) = x·cos(pθ) + rotate_half(x)·sin(pθ)，返回矩阵 P(p)，rope(x, p) = P(p) x。"""
    ang = pos * np.concatenate([inv_freq, inv_freq])
    d = len(ang)
    I = np.eye(d)
    return (I * np.cos(ang)[:, None]) + rotate_half(I.T).T * np.sin(ang)[:, None]


def rope_avg(inv_freq: np.ndarray, deltas) -> np.ndarray:
    """查询在位置 Δ、key 在位置 0 时 q·k = qᵀ P(Δ)ᵀ P(0) k；对 Δ 取平均，P(0) = I。"""
    return np.mean([rope_matrix(inv_freq, d).T for d in deltas], 0)


def qk_alignment(Wq_h, gq, Wk_g, gk, gin, v, B) -> float:
    """方向 v（残差空间）写入的查询与写入的 key 在平均 RoPE 双线性形式下的余弦：cos(γ_q⊙W_q(γ_in⊙v), B (γ_k⊙W_k(γ_in⊙v)))。"""
    u = gin * v
    q = gq * (Wq_h @ u)
    k = B @ (gk * (Wk_g @ u))
    return float(q @ k / (np.linalg.norm(q) * np.linalg.norm(k) + 1e-12))


def z_vs(v: float, null) -> float:
    n = np.asarray(null, float)
    return float((v - n.mean()) / (n.std() + 1e-12))


def monotone(seq, increasing: bool = True) -> bool:
    d = np.diff(np.asarray(seq, float))
    return bool(np.all(d >= 0) if increasing else np.all(d <= 0))


# ---- WS-v5：模式状态的自持（同样内容、不同顺序 + 中性段） ----
CRITERIA_V5 = {
    'schema': 'weight_read.v5',
    'neutral_lengths': [4, 16, 64],     # 中性段（数字列表）token 数
    'H1_gap_min': 0.2,                  # 中性段末位：模式投影差（先散文后代码 − 先代码后散文）/ 自然文本代码—散文均值差 ≥ 0.2（L = 4）
    'H1_persist_min': 0.5,              #   且 gap(64) / gap(4) ≥ 0.5
    'H2_select_ratio_min': 2.0,         # L = 16 末位：8 个头对代码 token 的质量比 PC / CP ≥ 2，且高于随机头的同一比值
    'H3_select_gain_min': 1.5,          # CP 顺序、L = 16：中性段位置上 8 个头查询 +4σ 代码移动后，末位代码质量比 ≥ 1.5 × 原始，
    'H3_state_frac_min': 0.1,           #   且末位模式投影向代码移动 ≥ 0.1 × 该长度下 PC − CP 的差；随机方向的移动 ≤ 1/3
    'H3_control_max_frac': 1 / 3,
    'seed': 0,
}


def neutral_segment(seed_text: str, n_numbers: int) -> str:
    """确定性的数字列表「d, d, d, …」（在代码与散文中都常见）。"""
    import hashlib
    h = hashlib.sha256(seed_text.encode()).digest()
    digits = [str(h[i % len(h)] % 10) for i in range(n_numbers)]
    return ', '.join(digits)


def persistence(gaps_by_len: Dict[int, float], short: int, long: int) -> float:
    g0 = gaps_by_len[short]
    return float(gaps_by_len[long] / g0) if g0 > 0 else float('nan')


# ---- WS-v6：单稳态局部积分模型的拟合与留出预测 ----
CRITERIA_V6 = {
    'schema': 'weight_read.v6',
    'windows': [1, 2, 4, 8, 16, 32, 64],
    'taus': [1, 2, 4, 8, 16, 32, 64],
    'betas': [0.0, 0.25, 0.5, 1.0, 2.0],
    'n_iter': 2,
    'H1_context_gain_min': 0.10,         # 确认集：积分模型（窗口或指数衰减中发现集选出的较好者）R² − 只用本 token 的 R² ≥ 0.10
    'H2_matched_gain_min': 0.02,         # 模式匹配加权（β > 0）比纯指数衰减 R² 高 ≥ 0.02，且发现集选出的 β > 0
    'H3_halflife_range': [4, 32],        # 发现集选出的指数衰减半衰期（token）在 [4, 32] 内
    'H4_mass_r2_min': 0.5,               # 同一积分模型预测下一 token 代码专有词质量的确认集 R² ≥ 0.5
    'smooth': 1.0,
    'seed': 0,
}


def unigram_logodds(code_counts: Dict[int, int], prose_counts: Dict[int, int], smooth: float = 1.0) -> Dict[int, float]:
    """token 的词汇证据：log P(token | 代码) − log P(token | 散文)，加性平滑。"""
    vocab = set(code_counts) | set(prose_counts)
    nc, npr = sum(code_counts.values()), sum(prose_counts.values())
    V = len(vocab)
    return {t: float(np.log((code_counts.get(t, 0) + smooth) / (nc + smooth * V))
                     - np.log((prose_counts.get(t, 0) + smooth) / (npr + smooth * V))) for t in vocab}


def window_feature(e: np.ndarray, k: int) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(e)])
    t = np.arange(len(e))
    lo = np.maximum(0, t - k + 1)
    return (c[t + 1] - c[lo]) / (t + 1 - lo)


def matched_feature(e: np.ndarray, tau: float, beta: float, m_hat: np.ndarray) -> np.ndarray:
    """f_t = Σ_{j≤t} w_tj e_j，w_tj ∝ exp(−(t−j)/τ + β·m̂_t·e_j)。β = 0 时为归一化指数衰减平均。"""
    T = len(e)
    out = np.zeros(T)
    for t in range(T):
        d = t - np.arange(t + 1)
        lw = -d / tau + beta * m_hat[t] * e[:t + 1]
        w = np.exp(lw - lw.max())
        out[t] = (w * e[:t + 1]).sum() / w.sum()
    return out


def fit_linear(x: np.ndarray, y: np.ndarray):
    A = np.stack([np.ones_like(x), x], 1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return coef


def r2(y: np.ndarray, p: np.ndarray) -> float:
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())
