# -*- coding: utf-8 -*-
"""Q3-T3：稀疏复用——「多模式融合」是否由可复用的字典列构成。

预注册：research/gpt5/docs/PAT_Q3T_PREREG_v1.md（§3）

问题
----
T1 说齿痕几乎恒定（同词 key 两两 cos 0.998），T2 说上下文几乎不改 key
（share_ctx 中位 0.009）、角色选择在 q 侧。两轮都指向同一件事：
**k 侧没有可供"融合"的结构**。那么"多模式融合"若存在，只能出现在输出端——
下一词分布是若干模式的叠加。T3 直接问输出端。

把 144 格相对 uniform 的 logit 差向量 Δ 做稀疏字典分解：若存在 K≤64 的字典、
每格 ≤3 列、重建 R²≥0.9，再看**同一角色的格子是否复用同一批列**。

跑前记录的三处设计决定（只补防伪，不改预注册判据）
------------------------------------------------
偏离 1（三元组构造）：改为 24 个**互不相同**的名词
    triples[i] = (P[i%12], P[(i+3)%12], NOUNS[i])，i=0..23
    理由：若沿用 T2 的「12 人 × 2 名词」，'th' 角色 24 个格子的答案只有 2 个词，
    reuse(th) 会退化到接近上限（所有格子几乎同一向量），判据自动成立却零信息。
    改为 24 名词后，每个角色至少 12 个不同答案。

偏离 2（字典拟合口径）：DictionaryLearning 取 n_components=32 < V=42。
    字典接近过完备时 in-sample R² 必然虚高，因此**主报 held-out R²**：
    按三元组 5 次随机对半切，字典只在训练折拟合，测试折用 OMP(3) 编码。
    in-sample R² 并列报告，只作对照。

偏离 3（复用伪来源控制，预注册补充判据，跑前冻结）：
    Δ 里最大的成分可能是"哪个词表被激活"——'ag' 的答案只能是人名、'th' 只能是名词。
    这会让 reuse(role) 天然很高却毫无信息（等于在说"不同问题答案不同"）。
    ⇒ 除原始 Δ 外另算**去角色均值版** Δc = Δ − mean(同角色其它格，留一)。
      判决口径（冻结）：
        reuse_raw ≥ 3.0 但 reuse_centered < 1.5
        ⇒ 判决降级 NULL，注明"复用全部来自答案词表差异，非融合结构"。
      这是 D6（判据自带备择区分）的直接应用，不是新增判据。

偏离 4（R² 不达标时的参照系，跑前冻结；初稿写错过一次，此处为更正版）：
    "每格 ≤3 列"意味着重建落在**该格所选 3 个原子张成的**3 维子空间里，不同格可选不同原子。
    因此**不存在**"全局 rank-3 PCA 上界"——那个上界只在所有格共享同一子空间时才成立；
    实测 OMP(3) 的 R² 可以高于全局 rank-3 PCA，这不是 bug。
    ⇒ 改用两条参照，都不做上界主张：
      (a) 全局 rank-r PCA 的 held-out R² 曲线（r=1..8）：衡量"**共享**低维模式能解释多少"；
      (b) same_entity / diff_entity 复用比：衡量复用是**按实体**还是**按角色**索引。
    判决：R²(held-out 中位) < 0.90 记 NULL_decomposition（分解前提不成立，reuse 不作为主张），
          但必须同时报 (a)(b)——"为什么达不到"本身就是关于融合结构的正面结论。
    预注册原文的 "≤3 原子 + R²≥0.9" 若经验上不成立，属判据设定问题，须在结论里点明，
    不得事后调 K/原子数去凑（那正是 M2 无限对照）。

判据（预注册 §3，冻结）
----------------------
前提  R²(held-out 中位) ≥ 0.90 且每格非零系数 ≤ 3
主量  reuse = 角色内平均 Jaccard / 随机期望（B=1000 模拟）
        ≥ 3.0（bootstrap 95% CI 下界 > 1.5） → 角色有专属列族
        1.5 – 3.0                            → 弱复用，报条件
        ≤ 1.5                                → 角色无专属方向 → 入 E 集
预测：reuse ≥ 3。

用法
----
  SMOKE=1 .venv/Scripts/python.exe -X utf8 tools/pat_q3t_t3.py
  .venv/Scripts/python.exe -X utf8 tools/pat_q3t_t3.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / 'tools')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import provenance
from pat_q1_unbind import PERSONS, build, render

MODEL_PATH = 'D:/develop/model/Qwen/Qwen3-4B'
SEED = 20261101
SMOKE = os.environ.get('SMOKE', '') == '1'

NOUNS = ['苹果', '香蕉', '桃子', '面包', '蛋糕', '杯子', '碗', '毛巾', '雨伞', '铅笔',
         '书包', '钥匙', '手表', '眼镜', '帽子', '围巾', '手套', '鞋子', '镜子', '想法',
         '计划', '建议', '问题', '答案', '理由', '办法', '习惯', '故事', '照片', '礼物']

QUERIES = {'ag': '是谁送的？', 're': '送给了谁？', 'th': '送的是什么？'}
FORMS = ('A', 'C')
K_DICT = 16 if SMOKE else 32
N_NONZERO = 3
N_SPLIT = 2 if SMOKE else 5
N_BOOT = 200 if SMOKE else 1000


# ---------------------------------------------------------------- 工具

def jacc(a: set, b: set) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def mean_jacc(sets) -> float:
    n = len(sets)
    if n < 2:
        return float('nan')
    tot, cnt = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            tot += jacc(sets[i], sets[j])
            cnt += 1
    return tot / cnt


def mean_jacc_idx(sets, idx) -> float:
    """按索引表算两两 Jaccard，**跳过自配对**（idx 重采样后可能出现重复索引）。

    不跳过的话，(i,i) 的 Jaccard 恒为 1.0，会把 bootstrap 分布整体抬上去，
    出现"CI 下界 > 点估计"这种结构性错误。
    """
    n = len(idx)
    tot, cnt = 0.0, 0
    for a in range(n):
        for b in range(a + 1, n):
            if idx[a] == idx[b]:
                continue
            tot += jacc(sets[idx[a]], sets[idx[b]])
            cnt += 1
    return tot / cnt if cnt else float('nan')


def build_triples(n: int):
    """24 个互不相同的名词（跑前记录：偏离 1）。"""
    out = []
    for i in range(n):
        out.append((PERSONS[i % len(PERSONS)],
                    PERSONS[(i + 3) % len(PERSONS)],
                    NOUNS[i % len(NOUNS)]))
    return out


def candidate_ids(tok):
    """候选词表 = 12 人名 + 30 名词，取首 token，去重后保序。"""
    ids, words = [], []
    seen = set()
    for w in list(PERSONS) + list(NOUNS):
        tid = tok(w, add_special_tokens=False)['input_ids'][0]
        if tid in seen:
            continue
        seen.add(tid)
        ids.append(tid)
        words.append(w)
    return ids, words


def main() -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.decomposition import DictionaryLearning

    t0 = time.time()
    rng = np.random.default_rng(SEED)
    dev = 'cuda'

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16).to(dev).eval()

    cand_ids, cand_words = candidate_ids(tok)
    V = len(cand_ids)
    ix = {w: i for i, w in enumerate(cand_words)}
    print('候选词表 V=%d（人名 %d + 名词 %d，去重后）'
          % (V, len(PERSONS), len(NOUNS)))
    print('字典 K=%d，每格非零 ≤%d，切分 %d 折' % (K_DICT, N_NONZERO, N_SPLIT))

    # ---------------- 阶段 1：144 格 Δ ----------------
    trips = build_triples(6 if SMOKE else 24)
    D_raw, meta = [], []
    beh = {q: [0, 0] for q in QUERIES}
    cov = []

    for ti, trip in enumerate(trips):
        for form in FORMS:
            ctx, _ranges = build(trip, form)
            for qk, qtext in QUERIES.items():
                full = render(tok, ctx + qtext)
                ids = tok(full, add_special_tokens=False)['input_ids']
                with torch.no_grad():
                    out = model(input_ids=torch.tensor([ids], device=dev),
                                use_cache=False)
                lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
                sc = {x: float(lp[tok(x, add_special_tokens=False)['input_ids'][0]])
                      for x in trip}
                want = {'ag': trip[0], 're': trip[1], 'th': trip[2]}[qk]
                beh[qk][1] += 1
                beh[qk][0] += int(max(sc, key=sc.get) == want)

                d = lp[torch.tensor(cand_ids, device=dev)].cpu().numpy()
                d = d + np.log(V)                        # 相对 uniform 的差
                p = lp.exp().cpu().numpy()
                cov.append(float(p[cand_ids].sum()))
                D_raw.append(d.astype(np.float64))
                meta.append({'ti': ti, 'form': form, 'qk': qk,
                             'ans': want, 'triple': trip})

    X = np.stack(D_raw)                              # [N, V]
    N = len(X)
    roles = np.array([m['qk'] for m in meta])
    print('有效格 N=%d，行为门 %s'
          % (N, {q: '%d/%d' % tuple(beh[q]) for q in QUERIES}))
    print('候选集质量覆盖中位 %.3f（低于 0.7 则 Δ 只覆盖部分概率质量）'
          % float(np.median(cov)))

    # ---------------- 阶段 2：数据形状诊断 ----------------
    def cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / max(na * nb, 1e-12))

    wi, cr = [], []
    for i in range(N):
        for j in range(i + 1, N):
            (wi if roles[i] == roles[j] else cr).append(cos(X[i], X[j]))
    cos_within = float(np.mean(wi))
    cos_cross = float(np.mean(cr))
    ev = np.linalg.eigvalsh(np.cov(X.T))
    pr = float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-12))
    print('原始 Δ 余弦：角色内 %.4f / 角色间 %.4f（差距大 ⇒ 词表差异主导）'
          % (cos_within, cos_cross))
    print('有效维度（参与比）= %.2f / V=%d' % (pr, V))

    # 去角色均值（留一）：Δc = Δ − mean(同角色其它格)
    Xc = np.zeros_like(X)
    for r in QUERIES:
        m = roles == r
        idx = np.where(m)[0]
        s = X[idx].sum(0)
        for i in idx:
            Xc[i] = X[i] - (s - X[i]) / max(len(idx) - 1, 1)
    evc = np.linalg.eigvalsh(np.cov(Xc.T))
    prc = float(evc.sum() ** 2 / max((evc ** 2).sum(), 1e-12))
    print('去角色均值后有效维度 = %.2f' % prc)

    # ---------------- 阶段 3：稀疏字典（held-out R²） ----------------
    from sklearn.linear_model import orthogonal_mp

    from sklearn.linear_model import orthogonal_mp
    from sklearn.decomposition import PCA
    import warnings
    warnings.filterwarnings('ignore')

    def fit_encode(Xtr, Xte, K):
        dl = DictionaryLearning(
            n_components=min(K, Xtr.shape[1] - 1), alpha=1.0, max_iter=1500,
            fit_algorithm='cd', transform_algorithm='omp',
            transform_n_nonzero_coefs=N_NONZERO, random_state=SEED)
        dl.fit(Xtr)
        D = dl.components_.T
        code = np.stack([orthogonal_mp(D, x, n_nonzero_coefs=N_NONZERO)
                         for x in Xte])
        return D, code

    def recon_r2(Xtr, Xte, D, code):
        mu = Xtr.mean(0)
        Xh = code @ D.T
        ssr = ((Xte - Xh) ** 2).sum(1)
        sst = ((Xte - mu) ** 2).sum(1)
        return 1.0 - ssr / np.maximum(sst, 1e-12)

    def role_center(Xa, ra, refX, refr):
        """用参考集（训练折）的角色均值中心化，避免测试折泄漏。"""
        mu = {r: refX[refr == r].mean(0) for r in QUERIES if (refr == r).any()}
        out = np.empty_like(Xa)
        for i, r in enumerate(ra):
            out[i] = Xa[i] - mu.get(r, np.zeros(Xa.shape[1]))
        return out

    def pca_curve(Xtr, Xte, ranks):
        mu = Xtr.mean(0)
        A = Xtr - mu
        B = Xte - mu
        p = PCA(n_components=max(ranks), random_state=SEED).fit(A)
        sst = ((Xte - mu) ** 2).sum(1)
        out = []
        for r in ranks:
            V = p.components_[:r]
            Xh = mu + (B @ V.T) @ V
            out.append(1.0 - (((Xte - Xh) ** 2).sum(1) / np.maximum(sst, 1e-12)).mean())
        return out

    RANKS = [1, 2, 3, 4, 5, 6, 8]
    uq = np.unique([m['ti'] for m in meta])
    tids = [int(x) for x in uq]
    r2_ho, r2_in, r2_ho_c, cv_raw, cv_c = [], [], [], [], []
    for s in range(N_SPLIT):
        perm = rng.permutation(tids)
        half = len(perm) // 2
        tr = set(perm[:half].tolist())
        mtr = np.array([m['ti'] in tr for m in meta])
        if mtr.sum() < 8 or (~mtr).sum() < 8:
            continue
        # 原始 Δ
        D, code = fit_encode(X[mtr], X[~mtr], K_DICT)
        r2_ho.append(recon_r2(X[mtr], X[~mtr], D, code))
        _, ci = fit_encode(X[mtr], X[mtr], K_DICT)
        r2_in.append(recon_r2(X[mtr], X[mtr], D, ci))
        # 去角色均值
        Xtr_c = role_center(X[mtr], roles[mtr], X[mtr], roles[mtr])
        Xte_c = role_center(X[~mtr], roles[~mtr], X[mtr], roles[mtr])
        Dc, codec = fit_encode(Xtr_c, Xte_c, K_DICT)
        r2_ho_c.append(recon_r2(Xtr_c, Xte_c, Dc, codec))
        cv_raw.append(pca_curve(X[mtr], X[~mtr], RANKS))
        cv_c.append(pca_curve(Xtr_c, Xte_c, RANKS))
    if not r2_ho:
        print('切分失败')
        return 1
    r2_med = float(np.median([np.median(x) for x in r2_ho]))
    r2_mean = float(np.median([float(np.mean(x)) for x in r2_ho]))
    r2_c_med = float(np.median([np.median(x) for x in r2_ho_c]))
    curve_raw = np.median(np.stack(cv_raw), axis=0)
    curve_c = np.median(np.stack(cv_c), axis=0)
    print('R² held-out 中位 %.3f / 均值 %.3f（每折中位 %s）'
          % (r2_med, r2_mean,
             ' '.join('%.3f' % np.median(x) for x in r2_ho)))
    print('R² held-out 去角色均值 中位 %.3f' % r2_c_med)
    print('R² in-sample 中位 %.3f（过完备对照）'
          % float(np.median([np.median(x) for x in r2_in])))
    print('rank 曲线（held-out R²）')
    print('  r      = ' + ' '.join('%6d' % r for r in RANKS))
    print('  raw    = ' + ' '.join('%6.3f' % x for x in curve_raw))
    print('  centrd = ' + ' '.join('%6.3f' % x for x in curve_c))

    def rank_for(curve, thr=0.90):
        for r, v in zip(RANKS, curve):
            if v >= thr:
                return r
        return None

    r90_raw, r90_c = rank_for(curve_raw), rank_for(curve_c)
    pca3 = float(curve_raw[RANKS.index(3)])
    print('共享 PCA 达到 R²≥0.90 所需维数：raw %s / centered %s；'
          'rank-3 共享子空间 = %.3f（不是上界，仅参照）'
          % (r90_raw, r90_c, pca3))

    # ---------------- 阶段 4：全量字典 + 列集合 ----------------
    def encode_all(Xa, K):
        dl = DictionaryLearning(
            n_components=min(K, Xa.shape[1] - 1), alpha=1.0, max_iter=800,
            fit_algorithm='cd', transform_algorithm='omp',
            transform_n_nonzero_coefs=N_NONZERO, random_state=SEED)
        dl.fit(Xa)
        D = dl.components_.T
        code = np.stack([orthogonal_mp(D, x, n_nonzero_coefs=N_NONZERO)
                         for x in Xa])
        nnz = (np.abs(code) > 1e-9).sum(1)
        sets = []
        for row in code:
            hit = [int(i) for i in np.argsort(-np.abs(row))[:N_NONZERO]
                   if abs(row[i]) > 1e-9]
            sets.append(set(hit))
        return D, code, sets, nnz

    Dl_raw, code_raw, sets_raw, nnz_raw = encode_all(X, K_DICT)
    Dl_c, code_c, sets_c, nnz_c = encode_all(Xc, K_DICT)
    nnz_max = int(max(nnz_raw.max(), nnz_c.max()))

    # 随机期望 E[Jaccard(两个随机 3 元子集)]
    Kr = min(K_DICT, V - 1)
    sim = []
    for _ in range(N_BOOT):
        a = rng.choice(Kr, N_NONZERO, replace=False)
        b = rng.choice(Kr, N_NONZERO, replace=False)
        sim.append(jacc(set(a.tolist()), set(b.tolist())))
    exp_j = float(np.mean(sim))
    print('随机期望 E[J] = %.4f（K=%d，B=%d）' % (exp_j, Kr, N_BOOT))

    def reuse_report(sets, tag):
        out = {}
        for r in QUERIES:
            idx = np.where(roles == r)[0]
            if len(idx) < 3:
                continue
            base = mean_jacc([sets[i] for i in idx])
            boot = []
            for _ in range(N_BOOT):
                pick = rng.choice(idx, len(idx), replace=True)
                v = mean_jacc_idx(sets, pick)
                if not np.isnan(v):
                    boot.append(v)
            lo, hi = (float(np.percentile(boot, 2.5)),
                      float(np.percentile(boot, 97.5)))
            if lo > base + 1e-12:
                print('  [警告] %s：CI 下界 %.4f > 点估计 %.4f，重采样有缺陷' % (r, lo, base))
            out[r] = {'jacc': round(base, 4),
                      'reuse': round(base / exp_j, 3),
                      'ci': [round(lo / exp_j, 3), round(hi / exp_j, 3)]}
        cross = []
        for i in range(N):
            for j in range(i + 1, N):
                if roles[i] != roles[j]:
                    cross.append(jacc(sets[i], sets[j]))
        out['cross'] = {'jacc': round(float(np.mean(cross)), 4),
                        'reuse': round(float(np.mean(cross)) / exp_j, 3)}
        # 同答案实体 vs 不同答案实体（角色内）
        same_e, diff_e = [], []
        for i in range(N):
            for j in range(i + 1, N):
                if roles[i] != roles[j]:
                    continue
                (same_e if meta[i]['ans'] == meta[j]['ans'] else diff_e).append(
                    jacc(sets[i], sets[j]))
        out['same_entity'] = {'jacc': round(float(np.mean(same_e)), 4),
                              'reuse': round(float(np.mean(same_e)) / exp_j, 3),
                              'n': len(same_e)}
        out['diff_entity'] = {'jacc': round(float(np.mean(diff_e)), 4),
                              'reuse': round(float(np.mean(diff_e)) / exp_j, 3),
                              'n': len(diff_e)}
        print('[%s] ' % tag + '  '.join(
            '%s=%.2f%s' % (k, v['reuse'],
                           ('(CI %.2f-%.2f)' % tuple(v['ci'])) if 'ci' in v else '')
            for k, v in out.items() if isinstance(v, dict)))
        return out

    rep_raw = reuse_report(sets_raw, 'raw')
    rep_c = reuse_report(sets_c, 'centered')

    # ---------------- 阶段 5：判决 ----------------
    main_roles = [r for r in QUERIES if r in rep_raw]
    reuse_raw_med = float(np.median([rep_raw[r]['reuse'] for r in main_roles]))
    reuse_c_med = float(np.median([rep_c[r]['reuse'] for r in main_roles]))
    ci_lo_raw = float(np.median([rep_raw[r]['ci'][0] for r in main_roles]))
    if r2_med < 0.90:
        verdict = 'NULL_decomposition'
        note = ('分解前提不成立：held-out R² 中位 %.3f < 0.90（in-sample %.3f 属过完备虚高）。'
                '共享 rank-8 PCA 仅 %.3f ⇒ 不存在可跨三元组泛化的共享低维模式，'
                'Δ 按**实体**索引而非按模式索引。reuse 数值仅作诊断，不作主张。'
                % (r2_med, float(np.median([np.median(x) for x in r2_in])),
                   float(curve_raw[-1])))
    elif reuse_raw_med >= 3.0 and reuse_c_med < 1.5:
        verdict = 'NULL_confound'
        note = ('复用全部来自答案词表差异（raw %.2f → centered %.2f），非融合结构'
                % (reuse_raw_med, reuse_c_med))
    elif reuse_raw_med >= 3.0 and ci_lo_raw > 1.5:
        verdict = 'REUSE_STRONG'
        note = '角色有专属列族'
    elif reuse_raw_med >= 1.5:
        verdict = 'REUSE_WEAK'
        note = '弱复用，须报条件'
    else:
        verdict = 'NO_REUSE'
        note = '角色无专属方向 → 入 E 集'

    se = rep_raw['same_entity']['reuse']
    de = rep_raw['diff_entity']['reuse']
    drv = 'entity' if (de > 0 and se / max(de, 1e-9) >= 2.0) else 'undetermined'
    print('\n判决 = %s' % verdict)
    print('说明：%s' % note)
    print('reuse(raw) 中位 %.2f，reuse(centered) 中位 %.2f，R²(ho) %.3f/%.3f，nnz≤%d'
          % (reuse_raw_med, reuse_c_med, r2_med, r2_c_med, nnz_max))
    print('复用驱动源：same_entity=%.2f vs diff_entity=%.2f ⇒ %s'
          % (se, de, '答案实体' if drv == 'entity' else '未定'))

    rec = {
        'seed': SEED, 'smoke': SMOKE, 'n_cells': N, 'n_triples': len(trips),
        'V': V, 'K_dict': K_DICT, 'n_nonzero': N_NONZERO, 'n_split': N_SPLIT,
        'behaviour_gate': {q: '%d/%d' % tuple(beh[q]) for q in QUERIES},
        'cand_mass_median': round(float(np.median(cov)), 4),
        'cos_within_role': round(cos_within, 4),
        'cos_cross_role': round(cos_cross, 4),
        'eff_dim_raw': round(pr, 2), 'eff_dim_centered': round(prc, 2),
        'r2_holdout_median': round(r2_med, 4),
        'r2_holdout_centered': round(r2_c_med, 4),
        'r2_holdout_folds': [round(float(np.median(x)), 4) for x in r2_ho],
        'r2_insample_median': round(float(np.median([np.median(x) for x in r2_in])), 4),
        'pca_ranks': RANKS,
        'pca_curve_raw': [round(float(x), 4) for x in curve_raw],
        'pca_curve_centered': [round(float(x), 4) for x in curve_c],
        'r2_holdout_mean': round(r2_mean, 4),
        'pca_shared_rank3': round(pca3, 4),
        'rank_for_90_raw': r90_raw, 'rank_for_90_centered': r90_c,
        'nnz_max': nnz_max,
        'random_expectation_jaccard': round(exp_j, 4),
        'reuse_raw': rep_raw, 'reuse_centered': rep_c,
        'reuse_raw_median': round(reuse_raw_med, 3),
        'reuse_centered_median': round(reuse_c_med, 3),
        'reuse_driver': drv,
        'verdict': verdict, 'note': note,
        'predicted': 'reuse >= 3（角色有专属列族）',
        'expected': ('若 R² 与 reuse 双双落空，则与 T1/T2 同向：k 侧无结构、'
                     '输出端复用按实体索引而非按角色索引，'
                     '下一步全押 Q3-Q 的 q 侧分解'),
        'elapsed_s': round(time.time() - t0, 1),
    }
    provenance.write_result('pat_q3t_t3', rec, model_path=MODEL_PATH,
                            seed=SEED, tool='pat_q3t_t3.py')
    print('完成 %.0fs' % (time.time() - t0))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
