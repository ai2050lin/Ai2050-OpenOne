# -*- coding: utf-8 -*-
"""语言谱图挖掘（atlas.v1）：把分布式测试结果从「每个模式族各自的规律」推进到「跨模式族共享的编码结构」。

输入：同一个模型快照（模型桶）下，若干模式族各自合并后的充分统计量（suffstats.v1，原生坐标，行为门已排除
不达标条件格）。同一模型、同一采样层的 HiddenState 处在同一个坐标系里，所以不同模式族的方向可以直接比较；
不同模型之间不能直接比方向，只比较「图结构」（哪些特征连在一起）。

算法（每个模型、每个采样层）：
  1. 特征 = 水平对比方向。对模式族 f 的因子 F 的水平 ℓ：先按条件格求均值，再对含 ℓ 的格等权平均得 m_ℓ
     （平衡掉其他因子的不均衡），v_ℓ = m_ℓ − mean_ℓ' m_ℓ'。discovery 与 lockbox（未见词表）各估一次。
     稳定特征：cos(v_disc, v_lockbox) ≥ MIN_REP（在没参与发现的词上复现）。
  2. 边 = 跨模式族的稳定特征对，同时满足：
     a) cos(v_a^disc, v_b^disc) ≥ MIN_COS，且 cos(v_a^lock, v_b^lock) ≥ MIN_COS（两组互不重叠的词各自复现）；
     b) 经验零分布：该层所有跨族特征对的 cos 中位数与 MAD 构成零分布（绝大多数特征对互不相关），
        z = (cos − 中位数) / (1.4826·MAD) ≥ MIN_Z。跨族特征对少于 MIN_NULL_PAIRS 时不做 b) 并在结果中标注。
  3. 共享编码轴 = 正边构成的连通分量中覆盖 ≥ 2 个模式族的部分；轴方向 = 成员单位方向的均值。
     留一族迁移：用其他族成员的 discovery 方向预测留出族在 lockbox 上的方向，cos 均值 = transfer。
  4. 预注册概念假设：文法的 concepts 字段（如 "frame=interrog": "疑问"）声明「这些水平应共享编码」。
     每个概念检查被声明的跨族特征对是否成边；没有被声明的边记为「新发现（未预注册）」。
  5. 跨模型：同一对特征（按 族/因子/水平/层 标识）在其他模型里是否也成边 → 跨模型一致率；轴按成员重叠匹配。
  6. 3D 谱图空间：各层所有特征的 discovery 单位方向做经典 MDS（距离 √(2−2cos)）取前三维。
  7. 嵌入层折扣：边所在读出位置有嵌入层标签（读出@L0 / emb）时，计算同一对特征在嵌入层上的原始对比 cos；
     discovery 与留出两次估计都 ≥ MIN_COS（或边本身在嵌入层）记为 token_identity——对齐可由读出位置的 token
     相同解释（如句末同为「？」）。token_identity 边与留出句式预测不计入概念支持；只剩这类证据的概念记为
     token_identity 状态。没有嵌入层标签时 token_identity = None（未检查），在结果中如实标注。

判据固定在 CRITERIA，随结果返回。共享方向说明「这些条件在该层用了相近的表示方向」，不说明共享同一计算机制；
机制需要干预实验（选择性 / 必要性）另行检验。
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SCHEMA = 'atlas.v1'
CRITERIA = {
    'min_rep': 0.5,            # 稳定特征：discovery 与 lockbox 方向 cos 下限
    'min_cos': 0.3,            # 边：discovery 与 lockbox 两次估计都要达到的 cos 下限
    'min_z': 3.0,              # 边：相对该层跨族 cos 经验零分布的稳健 z 下限
    'min_null_pairs': 20,      # 少于该数目的跨族特征对时不使用经验零分布
    'min_transfer': 0.5,       # 共享编码轴：留一族迁移 cos 下限
    'min_families': 2,         # 共享编码轴：至少覆盖的模式族数
    'len_delta_min': 1.0,      # 长度混杂标记：边的两端水平各自的平均 token 数偏离 ≥ 该值，且偏离同向与对齐方向一致（只标记，不排除）
}
_EPS = 1e-12


def _unit(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    return None if n < _EPS or not math.isfinite(n) else v / n


def _cos(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < _EPS or nb < _EPS:
        return None
    return float(np.dot(a, b) / (na * nb))


def _r(v: Optional[float], d: int = 4) -> Optional[float]:
    return None if v is None or not math.isfinite(v) else round(float(v), d)



# ---------------- 1. 特征 ----------------

def level_means(conds: List[Dict[str, Any]], n: np.ndarray, s: np.ndarray, factor: str) -> Dict[str, Tuple[np.ndarray, int]]:
    """单个划分：返回 {level: (m (L,d), n_level)}。先按条件格求均值，再对含该水平的格等权平均（平衡其他因子）。"""
    if not conds:
        return {}
    means = s / np.maximum(n, 1)[None, :, None]
    by_level: Dict[str, List[int]] = {}
    for i, c in enumerate(conds):
        if n[i] > 0 and factor in c:
            by_level.setdefault(str(c[factor]), []).append(i)
    return {lv: (means[:, idx].mean(axis=1), int(n[idx].sum())) for lv, idx in by_level.items()}


def _noise(conds, n, s, q) -> np.ndarray:
    """每层格内方差之和（跨维）的格平均：用于效应 / 噪声比。返回 (L,)。"""
    if not conds:
        return np.zeros(s.shape[0])
    nn = np.maximum(n, 1)[None, :, None]
    var = np.maximum(q / nn - (s / nn) ** 2, 0.0).sum(axis=2)       # (L, C)
    ok = n >= 2
    return var[:, ok].mean(axis=1) if ok.any() else var.mean(axis=1)


def level_len_delta(token_len: Optional[Dict[str, float]], factor: str) -> Dict[str, float]:
    """各水平的平均 token 数相对因子内水平均值的偏离（含该水平的条件格等权平均）。token_len: {条件格键: 平均 token 数}。"""
    if not token_len:
        return {}
    by = {}
    for cell, m in token_len.items():
        kv = dict(x.split('=', 1) for x in cell.split('|') if '=' in x)
        if factor in kv:
            by.setdefault(kv[factor], []).append(float(m))
    if len(by) < 2:
        return {}
    lm = {k: sum(v) / len(v) for k, v in by.items()}
    g = sum(lm.values()) / len(lm)
    return {k: v - g for k, v in lm.items()}


def family_features(family: str, tm_id: str, ss, factors: Sequence[str], exclude_conds: Optional[set] = None,
                    concepts: Optional[Dict[str, str]] = None,
                    token_len: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """从一个模式族的合并统计量取各因子各水平的均值（discovery / 留出划分 / 留出句式），并给出初始稳定性。
    ss = suffstats.SuffStats（同一模型桶内跨分片精确合并的结果）。token_len（可选）：{条件格键: 平均 token 数}，
    用于给每个特征记录句长偏离（长度混杂标记）。"""
    concepts = concepts or {}
    splits = ss.splits
    held = 'lockbox' if 'lockbox' in splits else ('confirm' if 'confirm' in splits else None)
    disc = ss.collapse(['discovery'], exclude_conds=exclude_conds)
    hold = ss.collapse([held], exclude_conds=exclude_conds) if held else None
    lfr = ss.collapse(['lockbox_frame'], exclude_conds=exclude_conds) if 'lockbox_frame' in splits else None
    noise = _noise(*disc)
    groups, feats = [], []
    for f in factors:
        md = level_means(disc[0], disc[1], disc[2], f)
        if len(md) < 2:
            continue
        mh = level_means(hold[0], hold[1], hold[2], f) if hold else {}
        mx = {lv: v for lv, v in (level_means(lfr[0], lfr[1], lfr[2], f) if lfr else {}).items() if lv not in md}
        ld = level_len_delta(token_len, f)
        groups.append({'factor': f, 'md': {lv: v[0] for lv, v in md.items()},
                       'mh': {lv: v[0] for lv, v in mh.items() if lv in md},
                       'mx': {lv: v[0] for lv, v in mx.items()}})
        gd = np.mean(np.stack([v[0] for v in md.values()]), axis=0)
        gh = np.mean(np.stack([v[0] for lv, v in mh.items() if lv in md]), axis=0) if all(lv in mh for lv in md) else None
        for lv, (m, n_lv) in sorted(md.items()):
            for li, tag in enumerate(ss.layer_tags):
                d = m[li] - gd[li]
                h = mh[lv][0][li] - gh[li] if gh is not None else None
                rep = _cos(d, h)
                snr = float(np.dot(d, d) / (noise[li] / max(n_lv, 1))) if noise[li] > _EPS else None
                feats.append({'id': f'{family}:{f}={lv}@{tag}', 'key': f'{family}:{f}={lv}', 'family': family,
                              'tm_id': tm_id, 'factor': f, 'level': lv, 'layer': tag, 'layer_i': li,
                              'concept': concepts.get(f'{f}={lv}'), 'n': n_lv, 'rep': _r(rep), 'snr': _r(snr, 2),
                              'stable': rep is not None and rep >= CRITERIA['min_rep'], 'heldout_only': False,
                              'len_delta': _r(ld.get(lv), 2) if lv in ld else None})
        for lv, (m, n_lv) in sorted(mx.items()):           # 只出现在留出句式里的水平：未见句式预测的检验对象
            for li, tag in enumerate(ss.layer_tags):
                feats.append({'id': f'{family}:{f}={lv}@{tag}', 'key': f'{family}:{f}={lv}', 'family': family,
                              'tm_id': tm_id, 'factor': f, 'level': lv, 'layer': tag, 'layer_i': li,
                              'concept': concepts.get(f'{f}={lv}'), 'n': n_lv, 'rep': None, 'snr': None,
                              'stable': False, 'heldout_only': True,
                              'len_delta': _r(ld.get(lv), 2) if lv in ld else None})
    return {'family': family, 'tm_id': tm_id, 'layers': list(ss.layer_tags), 'dim': int(ss.dim),
            'held_split': held, 'groups': groups, 'features': feats}


# ---------------- 2–6. 图 ----------------

def _robust_null(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if len(values) < CRITERIA['min_null_pairs']:
        return None, None
    arr = np.asarray(values)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med))) * 1.4826
    return med, (mad if mad > 1e-6 else None)


def _mds3(units: List[np.ndarray]) -> List[List[float]]:
    k = len(units)
    if k == 0:
        return []
    U = np.stack(units)
    G = U @ U.T                                                   # 单位向量的 Gram = cos
    D2 = np.maximum(2.0 - 2.0 * G, 0.0)
    J = np.eye(k) - 1.0 / k
    B = -0.5 * J @ D2 @ J
    w, V = np.linalg.eigh(B)
    order = np.argsort(w)[::-1][:3]
    X = V[:, order] * np.sqrt(np.maximum(w[order], 0.0))[None, :]
    if X.shape[1] < 3:
        X = np.hstack([X, np.zeros((k, 3 - X.shape[1]))])
    return [[round(float(x), 4) for x in row] for row in X]


class _Layer:
    """单层的「基线 ↔ 共享轴」交替估计。

    每个 (模式族, 因子) 是一组水平均值 m_ℓ。模型：m_ℓ = μ + a_k（ℓ 属于共享轴 k）或 μ + b_ℓ（ℓ 未解释）。
    基线 μ 只由「未解释」的水平定义（Σb = 0）：已并入共享轴的水平不再进入基线，于是共享成分不会经由
    本族平均泄漏到其他水平的对比里。未解释水平少于 2 个时，μ = 平均(m_ℓ − a_k(ℓ))（由其他族估出的轴补足）。
    a_k = 成员对比向量的平均（最小二乘）。discovery 与留出划分各自独立估计。
    """

    def __init__(self, groups, li):
        self.groups = groups                       # [(family, factor, md{lv:(d,)}, mh{lv:(d,)}, mx)]
        self.li = li
        self.member: Dict[Tuple[str, str, str], int] = {}
        self.A: Dict[int, Tuple[np.ndarray, Optional[np.ndarray]]] = {}

    def _baseline(self, fam, fac, means, which):
        free = [lv for lv in means if (fam, fac, lv) not in self.member]
        if len(free) >= 2:
            return np.mean(np.stack([means[lv] for lv in free]), axis=0)
        adj = []
        for lv, m in means.items():
            k = self.member.get((fam, fac, lv))
            a = self.A[k][which] if k is not None else None
            adj.append(m - a if a is not None else m)
        return np.mean(np.stack(adj), axis=0)

    def contrasts(self) -> Dict[Tuple[str, str, str], Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]]:
        out = {}
        for fam, fac, md, mh, mx in self.groups:
            d_l = {lv: v[self.li] for lv, v in md.items()}
            h_l = {lv: v[self.li] for lv, v in mh.items()}
            bd = self._baseline(fam, fac, d_l, 0)
            bh = self._baseline(fam, fac, h_l, 1) if len(h_l) == len(d_l) else None
            for lv in d_l:
                out[(fam, fac, lv)] = (d_l[lv] - bd, h_l[lv] - bh if bh is not None else None, None)
            for lv, v in mx.items():
                out[(fam, fac, lv)] = (None, None, v[self.li] - bd)
        return out

    def refine(self, rounds: int = 4):
        for _ in range(rounds):
            con = self.contrasts()
            for k in list(self.A):
                mem = [key for key, kk in self.member.items() if kk == k]
                d = np.mean(np.stack([con[x][0] for x in mem]), axis=0)
                hs = [con[x][1] for x in mem if con[x][1] is not None]
                self.A[k] = (d, np.mean(np.stack(hs), axis=0) if len(hs) == len(mem) else None)


_TAG_LAYER = re.compile(r'^(?P<prefix>(?:[^@]+@)?)L(?P<i>\d+)$')


def _baseline_tag(tag: str, layers: Sequence[str]) -> Optional[str]:
    """同一读出位置的嵌入层标签（"读出@L0" 或 "L0" / "emb"）；不存在时返回 None。"""
    if tag == 'emb' or tag.endswith('@emb'):
        return tag
    m = _TAG_LAYER.match(tag)
    if not m:
        return 'emb' if 'emb' in layers else None
    base = f'{m.group("prefix")}L0'
    return base if base in layers else None


def _restrict_family(fam: Dict[str, Any], tags: List[str]) -> Dict[str, Any]:
    """把一个族的特征与水平均值限制到给定标签（按 tags 顺序重新编号 layer_i）。"""
    if list(fam['layers']) == list(tags):
        return fam
    idx = [fam['layers'].index(t) for t in tags]
    pos = {t: i for i, t in enumerate(tags)}
    pick = lambda d: {lv: v[idx] for lv, v in d.items()}  # noqa: E731
    groups = [{**g, 'md': pick(g['md']), 'mh': pick(g['mh']), 'mx': pick(g['mx'])} for g in fam['groups']]
    feats = [{**f, 'layer_i': pos[f['layer']]} for f in fam['features'] if f['layer'] in pos]
    return {**fam, 'layers': list(tags), 'groups': groups, 'features': feats,
            'layers_dropped': [t for t in fam['layers'] if t not in pos]}


def build_model_graph(model: str, families: List[Dict[str, Any]]) -> Dict[str, Any]:
    """单个模型：families = family_features() 的结果列表（同模型）。

    各族的采样标签（读出@层）可以不同（如 suffstats 按体积预算只保留部分层）：只在全部可用族共有的标签上
    建图。维度与多数族不同、或与其他族没有共同标签的族不纳入，并在 skipped 中说明。"""
    usable, skipped = [], []
    dims = [fam['dim'] for fam in families]
    ref_dim = max(set(dims), key=dims.count) if dims else None
    for fam in families:
        if fam['dim'] != ref_dim:
            skipped.append({'family': fam['family'], 'reason': f'向量维度 {fam["dim"]} 与其他族（{ref_dim}）不一致'})
            continue
        if not fam['held_split']:
            skipped.append({'family': fam['family'], 'reason': '没有 lockbox/confirm 划分，无法检验复现'})
            continue
        usable.append(fam)
    common = [t for t in usable[0]['layers'] if all(t in fam['layers'] for fam in usable[1:])] if usable else []
    if usable and not common:
        # 没有全体共有的标签：保留与第一族共有标签最多的那部分族
        keep = [usable[0]] + [fam for fam in usable[1:] if set(fam['layers']) & set(usable[0]['layers'])]
        for fam in usable:
            if fam not in keep:
                skipped.append({'family': fam['family'], 'reason': '与其他族没有共同的采样标签（读出@层）'})
        usable = keep
        common = [t for t in usable[0]['layers'] if all(t in fam['layers'] for fam in usable[1:])]
    usable = [_restrict_family(fam, common) for fam in usable]
    layers = list(common)
    nodes = [f for fam in usable for f in fam['features']]
    node_of = {(f['family'], f['factor'], f['level'], f['layer']): f for f in nodes}
    groups = [(fam['family'], g['factor'], g['md'], g['mh'], g['mx']) for fam in usable for g in fam['groups']]
    edges, nulls, coords, axes, final_con = [], {}, {}, [], {}
    # 嵌入层基线：同一读出位置在 hs[0] 上的原始水平对比（不做共享轴折减）。嵌入层只含 token 身份，
    # 若一对特征在嵌入层上就已对齐，它们在更深层的对齐不能算作上下文计算形成的共享编码。
    base_con: Dict[str, Dict] = {}

    def baseline(tag):
        b = _baseline_tag(tag, layers)
        if b is None:
            return None, None
        if b not in base_con:
            base_con[b] = _Layer(groups, layers.index(b)).contrasts()
        return b, base_con[b]

    def token_identity(tag, x, y):
        b, bc = baseline(tag)
        if b is None:
            return {'baseline': None, 'token_identity': None}
        if b == tag:
            return {'baseline': b, 'token_identity': True, 'baseline_cos': None, 'baseline_cos_lockbox': None}
        bd, bh = _cos(bc[x][0], bc[y][0]), _cos(bc[x][1], bc[y][1])
        ti = bd is not None and bh is not None and bd >= CRITERIA['min_cos'] and bh >= CRITERIA['min_cos']
        return {'baseline': b, 'token_identity': ti, 'baseline_cos': _r(bd), 'baseline_cos_lockbox': _r(bh)}

    for li, tag in enumerate(layers):
        st = _Layer(groups, li)
        con = st.contrasts()
        stable = [k for k in con if con[k][0] is not None and node_of[(*k, tag)]['stable']]
        raw = [_cos(con[x][0], con[y][0]) for i, x in enumerate(stable) for y in stable[i + 1:] if x[0] != y[0]]
        med, scale = _robust_null([c for c in raw if c is not None])
        used = med is not None and scale is not None
        nulls[tag] = {'pairs': len(raw), 'median': _r(med), 'scale': _r(scale), 'used': used}
        rnd = 0
        while True:
            rnd += 1
            pool = [k for k in stable if k not in st.member]
            cand = []
            for i, x in enumerate(pool):
                for y in pool[i + 1:]:
                    if x[0] == y[0]:
                        continue
                    cd, ch = _cos(con[x][0], con[y][0]), _cos(con[x][1], con[y][1])
                    if cd is None or ch is None:
                        continue
                    z = (cd - med) / scale if used else None
                    if cd >= CRITERIA['min_cos'] and ch >= CRITERIA['min_cos'] and (z is None or z >= CRITERIA['min_z']):
                        cand.append((x, y, cd, ch, z))
            if not cand:
                break
            # 贪心生长：以最强的一条边为种子；之后每次加入「与当前成员中严格过半成边」且平均 cos 最高的特征，
            # 每个 (模式族, 因子) 至多一个水平。只靠一条弱边挂上来的特征不会被并入（避免强核心带出假成员）。
            ok = {}
            for x, y, cd, ch, z in cand:
                ok[(x, y)] = ok[(y, x)] = (cd, ch, z)
            seed = max(cand, key=lambda c: c[2])
            comp = [seed[0], seed[1]]
            while True:
                used_groups = {(x[0], x[1]) for x in comp}
                best = None
                for v in pool:
                    if v in comp or (v[0], v[1]) in used_groups:
                        continue
                    hits = [ok[(v, m)][0] for m in comp if (v, m) in ok]
                    if len(hits) * 2 > len(comp) and (best is None or np.mean(hits) > best[1]):
                        best = (v, float(np.mean(hits)))
                if best is None:
                    break
                comp.append(best[0])
            comp = sorted(comp)
            ce = [(x, y, *ok[(x, y)]) for i2, x in enumerate(comp) for y in comp[i2 + 1:] if (x, y) in ok]
            if len({x[0] for x in comp}) < CRITERIA['min_families']:
                break
            mean_cos = float(np.mean([c[2] for c in ce]))
            k = len(st.A)
            for x in comp:
                st.member[x] = k
            st.A[k] = (np.mean(np.stack([con[x][0] for x in comp]), axis=0),
                       np.mean(np.stack([con[x][1] for x in comp]), axis=0))
            st.refine()
            con = st.contrasts()
            fams = sorted({x[0] for x in comp})
            transfers = []                                          # 留一族迁移：其他族的成员方向 → 该族留出划分方向
            for fam in fams:
                others = [u for u in (_unit(con[x][0]) for x in comp if x[0] != fam) if u is not None]
                if not others:
                    continue
                pred = _unit(np.mean(np.stack(others), axis=0))
                transfers += [t for t in (_cos(pred, con[x][1]) for x in comp if x[0] == fam) if t is not None]
            transfer = float(np.mean(transfers)) if transfers else None
            ax_id = f'AX-{model_slug(model)}-{tag}-{rnd}'
            mem_nodes = [node_of[(*x, tag)] for x in comp]
            ax_ti = []
            for x, y, cd, ch, z in ce:
                a, b = node_of[(*x, tag)], node_of[(*y, tag)]
                same = bool(a['concept'] and a['concept'] == b['concept'])
                ti = token_identity(tag, x, y)
                ax_ti.append(ti['token_identity'])
                la, lb = a.get('len_delta'), b.get('len_delta')
                len_conf = (None if la is None or lb is None else
                            bool(abs(la) >= CRITERIA['len_delta_min'] and abs(lb) >= CRITERIA['len_delta_min']
                                 and (la * lb > 0) == (cd > 0)))
                edges.append({'source': a['id'], 'target': b['id'], 'layer': tag, 'axis': ax_id, 'cos': _r(cd),
                              'cos_lockbox': _r(ch), 'cos_refined': _r(_cos(con[x][0], con[y][0])), 'z': _r(z, 2),
                              'declared': same, 'concept': a['concept'] if same else None, **ti,
                              'len_delta': [la, lb], 'length_confound': len_conf})
            supported = transfer is not None and transfer >= CRITERIA['min_transfer']
            axis_ti = (True if ax_ti and all(t is True for t in ax_ti) else
                       None if any(t is None for t in ax_ti) else False)
            axes.append({'id': ax_id, 'layer': tag, 'round': rnd, 'members': [m['id'] for m in mem_nodes], 'families': fams,
                         'concepts': sorted({m['concept'] for m in mem_nodes if m['concept']}), 'mean_cos': _r(mean_cos),
                         'transfer': _r(transfer), 'supported': supported, 'token_identity': axis_ti,
                         'supported_beyond_token': supported and axis_ti is False,
                         'label': ' / '.join(f'{m["family"]}·{m["factor"]}={m["level"]}' for m in mem_nodes)})
        final_con[tag] = con
        keep = [(node_of[(*key, tag)], _unit(v[0] if v[0] is not None else v[2])) for key, v in con.items()]
        keep = [(f, u) for f, u in keep if u is not None]
        coords[tag] = {f['id']: p for (f, _), p in zip(keep, _mds3([u for _, u in keep]))}

    # 4. 预注册概念假设（用最终基线下的对比方向）
    # 嵌入层折扣：token 身份即可解释的边（嵌入层上已对齐，或边本身在嵌入层）不计入概念支持
    edge_set = {(e['source'], e['target']) for e in edges if e.get('token_identity') is not True}
    edge_set |= {(t, s) for s, t in edge_set}
    ti_set = {(e['source'], e['target']) for e in edges if e.get('token_identity') is True}
    ti_set |= {(t, s) for s, t in ti_set}
    by_concept: Dict[str, List[dict]] = {}
    for f in nodes:
        if f['concept']:
            by_concept.setdefault(f['concept'], []).append(f)
    hyps = []
    for tag_c, fl in sorted(by_concept.items()):
        fams = sorted({f['family'] for f in fl})
        if len(fams) < 2:
            continue
        per_layer = []
        for t in layers:
            con = final_con[t]
            members = [f for f in fl if f['layer'] == t and not f['heldout_only']]
            pairs = [(a, b) for i, a in enumerate(members) for b in members[i + 1:] if a['family'] != b['family']]
            linked = [(a, b) for a, b in pairs if (a['id'], b['id']) in edge_set]
            linked_ti = [(a, b) for a, b in pairs if (a['id'], b['id']) in ti_set]
            vec = lambda f: con[(f['family'], f['factor'], f['level'])]  # noqa: E731
            b_tag, bc = baseline(t)
            heldout = []
            for h in (x for x in fl if x['layer'] == t and x['heldout_only']):
                others = [u for u in (_unit(vec(m)[0]) for m in members if m['family'] != h['family'] and m['stable']) if u is not None]
                pred = _cos(np.mean(np.stack(others), axis=0), vec(h)[2]) if others else None
                ok = pred is not None and pred >= CRITERIA['min_transfer']
                base_pred = None
                if b_tag is not None and bc is not None:
                    key = lambda f: (f['family'], f['factor'], f['level'])  # noqa: E731
                    bo = [u for u in (_unit(bc[key(m)][0]) for m in members if m['family'] != h['family']) if u is not None]
                    base_pred = _cos(np.mean(np.stack(bo), axis=0), bc[key(h)][2]) if bo else None
                h_ti = None if b_tag is None else (b_tag == t or (base_pred is not None and base_pred >= CRITERIA['min_transfer']))
                heldout.append({'id': h['id'], 'pred_cos': _r(pred), 'baseline_pred_cos': _r(base_pred),
                                'token_identity': h_ti, 'pass': ok and h_ti is not True, 'pass_raw': ok})
            cs = [c for c in (_cos(vec(a)[0], vec(b)[0]) for a, b in pairs) if c is not None]
            per_layer.append({'layer': t, 'pairs': len(pairs), 'linked': len(linked), 'linked_token_identity': len(linked_ti),
                              'heldout': heldout, 'mean_cos': _r(float(np.mean(cs))) if cs else None,
                              'unstable': [f['id'] for f in members if not f['stable']]})
        full = [p for p in per_layer if p['pairs'] and p['linked'] == p['pairs'] and all(h['pass'] for h in p['heldout'])]
        any_link = any(p['linked'] or any(h['pass'] for h in p['heldout']) for p in per_layer)
        any_ti = any(p['linked_token_identity'] or any(h['pass_raw'] and h['token_identity'] for h in p['heldout'])
                     for p in per_layer)
        status = 'supported' if full else ('partial' if any_link else ('token_identity' if any_ti else 'negative'))
        best = max(per_layer, key=lambda p: (p in full, p['linked'] / p['pairs'] if p['pairs'] else -1,
                                             sum(h['pass'] for h in p['heldout']), p['linked_token_identity'],
                                             p['mean_cos'] or -1)) if per_layer else None
        hyps.append({'concept': tag_c, 'families': fams, 'members': sorted({f['key'] for f in fl}), 'status': status,
                     'layers': per_layer, 'best_layer': best['layer'] if best else None})
    novel = [e for e in edges if not e['declared'] and e.get('token_identity') is not True]
    out_nodes = [{k: v for k, v in f.items() if k != 'layer_i'} for f in nodes]
    return {
        'model': model, 'layers': layers, 'families': [f['family'] for f in usable], 'skipped': skipped,
        'layers_dropped': {f['family']: f['layers_dropped'] for f in usable if f.get('layers_dropped')},
        'nodes': out_nodes, 'edges': edges, 'nulls': nulls, 'coords': coords, 'axes': axes,
        'concept_hypotheses': hyps, 'novel_edges': len(novel),
    }


def model_slug(model: str) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '-', model).strip('-')[:24] or 'M'


def cross_model(graphs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """跨模型：同一特征对（族/因子/水平/层）在其他模型中是否也成边；共享编码轴按成员重叠匹配。"""
    def ekey(e):
        a, b = sorted([e['source'], e['target']])
        return a, b
    sets = {g['model']: {ekey(e) for e in g['edges']} for g in graphs}
    measured = {g['model']: {n['id'] for n in g['nodes'] if n['stable']} for g in graphs}
    rows = []
    for g in graphs:
        others = [h for h in graphs if h['model'] != g['model']]
        for e in g['edges']:
            k = ekey(e)
            testable = [h['model'] for h in others if k[0] in measured[h['model']] and k[1] in measured[h['model']]]
            hit = [m for m in testable if k in sets[m]]
            e['other_models'] = {'testable': len(testable), 'replicated': len(hit)}
            rows.append((len(testable), len(hit)))
        for ax in g['axes']:
            mem = set(ax['members'])
            match = []
            for h in others:
                best = max((len(mem & set(b['members'])) / len(mem | set(b['members'])) for b in h['axes']), default=0.0)
                if best >= 0.5:
                    match.append(h['model'])
            ax['replicated_in'] = match
    tested = sum(1 for t, _ in rows if t)
    rep = sum(1 for t, h in rows if t and h)
    jac = []
    for i, g in enumerate(graphs):
        for h in graphs[i + 1:]:
            u = sets[g['model']] | sets[h['model']]
            jac.append({'a': g['model'], 'b': h['model'],
                        'jaccard': _r(len(sets[g['model']] & sets[h['model']]) / len(u)) if u else None})
    return {'edges_testable': tested, 'edges_replicated': rep,
            'replication_rate': _r(rep / tested) if tested else None, 'pairwise': jac}


# ---------------- 结论 ----------------

def findings(graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    n_stable = sum(1 for n in graph['nodes'] if n['stable'])
    out.append({'id': 'coverage', 'kind': 'info', 'title': '覆盖范围',
                'text': f'{len(graph["families"])} 个模式族、{len(graph["layers"])} 个采样层、{len(graph["nodes"])} 个水平对比特征，'
                        f'其中 {n_stable} 个在未见词表上复现（cos ≥ {CRITERIA["min_rep"]}）。'
                        + (f' 未纳入：{"；".join(s["family"] + "（" + s["reason"] + "）" for s in graph["skipped"])}。' if graph['skipped'] else '')})
    weak_null = [t for t, v in graph['nulls'].items() if not v['used']]
    if weak_null:
        out.append({'id': 'null', 'kind': 'warning', 'title': '经验零分布样本不足',
                    'text': f'层 {", ".join(weak_null)} 的跨族特征对少于 {CRITERIA["min_null_pairs"]} 个，边只按两次估计的 cos 阈值判定；'
                            '加入更多模式族后结论更可靠。'})
    for ax in graph['axes']:
        ti = ax.get('token_identity') is True
        kind = 'warning' if ti and ax['supported'] else ('supported' if ax['supported'] else 'negative')
        rep = f'；在 {len(ax.get("replicated_in", []))} 个其他模型中复现' if 'replicated_in' in ax else ''
        out.append({'id': ax['id'], 'kind': kind, 'level': '候选共享编码轴',
                    'title': f'{ax["layer"]} 层共享方向：{ax["label"]}' + ('（token 身份）' if ti else ''),
                    'text': f'{len(ax["families"])} 个模式族的 {len(ax["members"])} 个特征方向相互对齐（平均 cos {ax["mean_cos"]}），'
                            f'留一族迁移 cos {ax["transfer"]}（判据 ≥ {CRITERIA["min_transfer"]}）{rep}。'
                            + (' 这些特征在同一读出位置的嵌入层（hs[0]）上已经对齐，可由读出 token 相同解释，不计为上下文形成的共享编码。'
                               if ti else '')
                            + (f' 预注册概念：{"、".join(ax["concepts"])}。' if ax['concepts'] else ' 未预注册：新发现候选，需设计专门模板验证。')})
    n_ti = sum(1 for e in graph['edges'] if e.get('token_identity') is True)
    if n_ti:
        out.append({'id': 'token-identity', 'kind': 'warning', 'title': f'{n_ti} 条边可由 token 身份解释',
                    'text': '这些边的两个特征在同一读出位置的嵌入层（hs[0]）上就已对齐（两次估计都 ≥ '
                            f'{CRITERIA["min_cos"]}），或边本身位于嵌入层：对齐来自读出位置的 token 相同（如句末同为「？」），'
                            '不计入概念支持。要检验超出 token 身份的共享编码，需要让对照条件在读出位置使用相同 token。'})
    n_len = sum(1 for e in graph['edges'] if e.get('length_confound') is True)
    if n_len:
        out.append({'id': 'length-confound', 'kind': 'warning', 'title': f'{n_len} 条边可能受句长混杂',
                    'text': f'这些边两端水平的平均 token 数都偏离各自因子均值 ≥ {CRITERIA["len_delta_min"]}，且偏离同向与对齐方向一致：'
                            '对齐可能来自句长 / 位置而非语言变量。只作标记，不从概念判定中排除；需要句长匹配的材料复核。'})
    for h in graph['concept_hypotheses']:
        kind = {'supported': 'supported', 'partial': 'warning', 'negative': 'negative', 'token_identity': 'warning'}[h['status']]
        best = next((p for p in h['layers'] if p['layer'] == h['best_layer']), None)
        out.append({'id': f'concept-{h["concept"]}', 'kind': kind, 'level': '预注册假设',
                    'title': f'概念「{h["concept"]}」跨 {len(h["families"])} 个模式族共享编码',
                    'text': ({'supported': '支持', 'partial': '部分支持', 'negative': '未支持',
                              'token_identity': '仅由 token 身份解释'}[h['status']]
                             + f'：最佳层 {h["best_layer"]}，{best["linked"]}/{best["pairs"]} 对声明特征成边（不含 token 身份边'
                               f' {best["linked_token_identity"]} 对），平均 cos {best["mean_cos"]}。'
                             if best else '')
                            + (''.join(f' 留出句式 {x["id"]}：其他族的同概念方向预测 cos {x["pred_cos"]}'
                                       f'（{"通过" if x["pass"] else "未通过"}，判据 ≥ {CRITERIA["min_transfer"]}'
                                       + (f'；嵌入层基线预测 cos {x["baseline_pred_cos"]}，已能由 token 身份解释'
                                          if x.get('token_identity') and x.get('pass_raw') else '') + '）。'
                                       for x in best['heldout']) if best else '')
                            + (f' 不稳定（未复现）特征：{", ".join(best["unstable"])}。' if best and best['unstable'] else '')})
    if graph['novel_edges']:
        out.append({'id': 'novel', 'kind': 'info', 'title': f'{graph["novel_edges"]} 条未预注册的跨族边',
                    'text': '这些特征对没有被文法 concepts 声明为同一概念，却在两组互不重叠的词上都对齐。它们是新假设的来源：'
                            '把对应水平写进同一个新模式族（或在 concepts 中预注册）后重新测试。'})
    out.append({'id': 'boundary', 'kind': 'info', 'title': '证据边界',
                'text': '共享方向说明这些条件在该层使用了相近的表示方向；是否共享同一计算机制，需要沿轴方向做干预'
                        '（选择性：只改变对应条件的输出；必要性：消去后对应行为下降）。'})
    return out


def build_atlas(per_model: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    graphs = [build_model_graph(m, fams) for m, fams in sorted(per_model.items())]
    xm = cross_model(graphs) if len(graphs) > 1 else None
    for g in graphs:
        g['findings'] = findings(g)
    return {'schema': SCHEMA, 'criteria': CRITERIA, 'models': [g['model'] for g in graphs], 'graphs': graphs,
            'cross_model': xm,
            'policy': '只使用合并口径内的结果（行为门已排除不达标格）；特征方向在原生坐标中比较，只在同一模型内比较方向，'
                      '跨模型只比较图结构。判据固定在 criteria，规律陈述是候选，不自动成为 Evidence。'}


# ---------------- AI 分析 ----------------

AI_SYSTEM = ('你是语言模型可解释性研究助手。你只能根据给定的谱图摘要提出可检验的假设，不能把摘要中没有的数字当作事实。'
             '每个假设必须给出：可被新实验推翻的预测、用来检验它的新模式族设计（因子与水平、句式示例）、以及主要的替代解释。')


def digest(atlas: Dict[str, Any], model: Optional[str] = None, max_nodes: int = 40) -> str:
    """面向 AI（也供人阅读）的谱图摘要（Markdown）。只含统计结果，不含原始向量。"""
    graphs = [g for g in atlas['graphs'] if model is None or g['model'] == model]
    L = [f'# 语言谱图摘要（{SCHEMA}）', '', f'判据：`{json.dumps(atlas["criteria"], ensure_ascii=False)}`', '']
    for g in graphs:
        L += [f'## 模型 {g["model"]}', '', f'- 模式族：{", ".join(g["families"])}；采样层：{", ".join(g["layers"])}']
        if g['skipped']:
            L.append('- 未纳入：' + '；'.join(f'{s["family"]}（{s["reason"]}）' for s in g['skipped']))
        stable = sorted((n for n in g['nodes'] if n['stable']), key=lambda n: -(n['snr'] or 0))[:max_nodes]
        L += ['', '### 稳定特征（未见词表复现；按效应/噪声排序）', '', '| 特征 | 概念 | 复现 cos | 效应/噪声 |', '|---|---|---|---|']
        L += [f'| {n["id"]} | {n["concept"] or "—"} | {n["rep"]} | {n["snr"]} |' for n in stable]
        L += ['', '### 共享编码轴', '']
        L += [f'- {a["id"]}（{a["layer"]}）：{a["label"]}；平均 cos {a["mean_cos"]}；留一族迁移 {a["transfer"]}；'
              f'{"支持" if a["supported"] else "未达判据"}' + (f'；概念 {"、".join(a["concepts"])}' if a['concepts'] else '；未预注册')
              for a in g['axes']] or ['- 无']
        L += ['', '### 预注册概念假设', '']
        L += [f'- 「{h["concept"]}」（{", ".join(h["families"])}）：{h["status"]}；最佳层 {h["best_layer"]}' for h in g['concept_hypotheses']] or ['- 无']
        top = sorted(g['edges'], key=lambda e: -(e['z'] or e['cos'] or 0))[:15]
        L += ['', '### 跨族边（前 15，按 z）', '']
        L += [f'- {e["source"]} ↔ {e["target"]}：cos {e["cos"]} / 未见词表 {e["cos_lockbox"]} / z {e["z"]}'
              + ('' if e['declared'] else '（未预注册）') for e in top] or ['- 无']
        L.append('')
    if atlas.get('cross_model'):
        xm = atlas['cross_model']
        L += ['## 跨模型', '', f'- 可检验的边 {xm["edges_testable"]}，在其他模型也成边 {xm["edges_replicated"]}（{xm["replication_rate"]}）', '']
    return '\n'.join(L)


def ai_prompt(atlas: Dict[str, Any], question: str = '', model: Optional[str] = None) -> str:
    q = question.strip() or '这些共享方向可能反映了什么编码机制？哪些需要新的测试来区分？'
    return (digest(atlas, model) + '\n\n---\n研究问题：' + q + '\n\n'
            '请只输出一个 JSON 数组（最多 5 项），每项字段：\n'
            '{"statement": 假设陈述, "evidence": [引用的特征或轴 id], "prediction": 新实验中可被推翻的预测, '
            '"new_family": {"family_id": 建议 id, "factors": [...], "levels": {因子: [水平]}, "frame_examples": [句式示例], '
            '"concepts": {"因子=水平": 概念}}, "alternative": 主要替代解释}\n'
            '不要输出 JSON 以外的内容。')


def parse_hypotheses(text: str) -> Dict[str, Any]:
    """从模型回复中取出 JSON 数组并校验字段；无法解析时返回原文，状态标为 unparsed。"""
    m = re.search(r'\[.*\]', text or '', re.S)
    if not m:
        return {'status': 'unparsed', 'hypotheses': [], 'raw': text}
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {'status': 'unparsed', 'hypotheses': [], 'raw': text}
    out = []
    for h in arr if isinstance(arr, list) else []:
        if not isinstance(h, dict) or not str(h.get('statement') or '').strip():
            continue
        out.append({
            'statement': str(h.get('statement'))[:600],
            'evidence': [str(x) for x in (h.get('evidence') or [])][:12],
            'prediction': str(h.get('prediction') or '')[:600],
            'new_family': h.get('new_family') if isinstance(h.get('new_family'), dict) else None,
            'alternative': str(h.get('alternative') or '')[:400],
            'status': 'unverified',          # AI 提出的假设：未经检验，必须变成新模板并运行后才有证据
        })
    return {'status': 'ok' if out else 'empty', 'hypotheses': out[:5], 'raw': text if not out else None}
