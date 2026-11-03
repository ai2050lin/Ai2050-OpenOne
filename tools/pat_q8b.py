# -*- coding: utf-8 -*-
"""Q8-B 频率旋钮的量纲化（Qwen3-4B / GLM-4-9B 通用）

预注册：research/gpt5/docs/PAT_Q8B_PREREG_v1.md（§1–§5 冻结，跑后修订入 §8）

用法：
    .venv/Scripts/python.exe -X utf8 tools/pat_q8b.py
    SMOKE=1 .venv/Scripts/python.exe -X utf8 tools/pat_q8b.py
    DEVICE_MAP=auto MODEL=D:/develop/model/ZhipuAI/glm-4-9b-chat-hf .venv/Scripts/python.exe -X utf8 tools/pat_q8b.py
"""
import os
import sys
import time
import gc
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import provenance                      # noqa: E402
import pat_q8a as A                    # noqa: E402

LOG = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


SMOKE = os.environ.get('SMOKE') == '1'
SEED = int(os.environ.get('SEED', '20261101'))
RNG = np.random.default_rng(SEED)
MODEL_PATH = os.environ.get('MODEL', 'D:/develop/model/Qwen/Qwen3-4B')
N_FOLD = 2 if SMOKE else 5
N_PERM = 5 if SMOKE else 12
LAM = 1e-3


# ---------------------------------------------------------------- 数值工具

def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o


def r2_score(y, p):
    ss = float(((y - p) ** 2).sum())
    st = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss / st if st > 0 else float('nan')


def gram_xy(X, y, chunk=8192):
    d = X.shape[1]
    G = np.zeros((d, d), np.float64)
    b = np.zeros(d, np.float64)
    for i in range(0, len(X), chunk):
        blk = X[i:i + chunk].astype(np.float64)
        G += blk.T @ blk
        b += blk.T @ y[i:i + chunk]
    return G, b


def solve_w(G, b, lam=LAM):
    d = G.shape[0]
    tr = float(np.trace(G)) / d
    return np.linalg.solve(G + lam * tr * np.eye(d), b)


def fkey(ns, folds):
    """缓存键 + 折指纹。

    §8 跑后修订：原缓存 key 只有 (ns, j)，不含"这一折是哪套划分"。
    零分布先用 _f0 调了一次 cv_fast(ns='main')，把 ('f','main',j) 写成 _f0 的 Gram；
    主流程再用 folds 就取到错的 Gf，G_all−Gf ≠ G_train，全维 R² 从 0.826 掉到 0.776。
    缓存键必须带上折的指纹（形状 + 首折元素和），这条并入纪律 R-MTH-24。
    """
    return (ns, len(folds), int(np.asarray(folds[0]).sum()), int(np.asarray(folds[0]).size))


def gmat(X, v):
    """X.T @ v，强制 v 为 float32。

    §8 跑后修订：X 是 float32、v 是 float64 时 numpy 会把整块 X 提升到 float64
    （GLM 下 27 215×4096×8 ≈ 891 MB，每次调用都分配一次），这是之前"卡 10 分钟"的真凶。
    结果再转 float64 供求解。
    """
    return (X.T @ np.asarray(v, np.float32)).astype(np.float64)


_GCACHE = {}


def gcache(key, X):
    """X^T X 的缓存。G 不依赖 y，打乱标签时完全可复用（本轮踩过：重复算是 20 倍浪费）。"""
    if key not in _GCACHE:
        d = X.shape[1]
        G = np.zeros((d, d), np.float64)
        for i in range(0, len(X), 8192):
            blk = X[i:i + 8192].astype(np.float64)
            G += blk.T @ blk
        _GCACHE[key] = G
    return _GCACHE[key]


def cv_fast(X, y, folds, ns, return_w=False):
    """缓存 G 的 K 折 OLS：每次只重算 X^T y。"""
    nk = fkey(ns, folds)
    G_all = gcache(('all', nk), X)
    b_all = gmat(X, y)
    pred = np.zeros(len(y))
    ws = []
    for j, f in enumerate(folds):
        Gf = gcache(('f', nk, j), X[f])
        bf = gmat(X[f], y[f])
        w = solve_w(G_all - Gf, b_all - bf)
        ws.append(w)
        pred[f] = X[f] @ w
    r2 = r2_score(y, pred)
    return (r2, pred, ws) if return_w else (r2, pred)


def cv_res(X, y, folds, u, w, ns_src='main'):
    """回归 y ~ X_res，其中 X_res = X − u·wᵀ（u 为标量协变量，w 为其对 X 的系数）。

    §8 跑后修订：直接构造 X_res 再算 Gram 是 3.2e11 flops 的浪费（一次 3 分钟）。
    X^T X 可闭式展开：
        (X − u wᵀ)ᵀ(X − u wᵀ) = G − (Xᵀu) wᵀ − w (Xᵀu)ᵀ + (uᵀu) w wᵀ
    折内同理。数值上与原算法差 <1e-6。
    """
    u = u.astype(np.float64)
    w = w.astype(np.float64)
    nk = fkey(ns_src, folds)
    G_all = gcache(('all', nk), X)
    Xu = (X.T @ np.asarray(u, np.float32)).astype(np.float64)
    G_res = G_all - np.outer(Xu, w) - np.outer(w, Xu) + float(u @ u) * np.outer(w, w)
    b_res = gmat(X, y) - w * float(u @ y)
    pred = np.zeros(len(y))
    for j, f in enumerate(folds):
        Gf = gcache(('f', nk, j), X[f])
        uf = u[f]
        Xuf = (X[f].T @ np.asarray(uf, np.float32)).astype(np.float64)
        Gf_res = Gf - np.outer(Xuf, w) - np.outer(w, Xuf) + float(uf @ uf) * np.outer(w, w)
        bf = gmat(X[f], y[f]) - w * float(uf @ y[f])
        wsol = solve_w(G_res - Gf_res, b_res - bf)
        pred[f] = (X[f] - np.outer(uf, w)) @ wsol
    return r2_score(y, pred), pred


def folds_of(n):
    return np.array_split(RNG.permutation(n), N_FOLD)


def cv_full(X, y, folds, G_all=None, b_all=None):
    """全空间 OLS 的 held-out R² 与其折内方向（用于生成无泄漏的 t）。"""
    if G_all is None:
        G_all, b_all = gram_xy(X, y)
    pred = np.zeros(len(y))
    for f in folds:
        Gf, bf = gram_xy(X[f], y[f])
        w = solve_w(G_all - Gf, b_all - bf)
        pred[f] = X[f] @ w
    return r2_score(y, pred), pred


def corr_cols(X, y):
    """每个坐标与 y 的相关（X, y 都已中心化）。"""
    yc = np.asarray(y - y.mean(), np.float32)
    Xc = X - X.mean(0)
    num = (Xc.T @ yc).astype(np.float64)
    den = np.linalg.norm(Xc, axis=0) * np.linalg.norm(yc) + 1e-12
    return num / den


def bootstrap_ci(vals, fn=np.mean, n=400):
    vals = np.asarray(vals)
    idx = RNG.integers(0, len(vals), (n, len(vals)))
    bs = np.array([fn(vals[i]) for i in idx])
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def krylov_basis(G, bvec, k_max, tol=1e-10):
    """Krylov 基 span{b, Gb, G²b, ...}。

    §8 跑后修订：PLS-1 的权向量正是张成这个子空间的（PLS-1 的预测 = Krylov 子空间上的 LS 投影），
    但逐个 deflation 在 float32 + 大矩阵上退化到 ~35 MFLOPS（6 分钟没跑完 63 次迭代）。
    改用 G 上的 Arnoldi，每步只有一次 2560² matvec。
    """
    d = G.shape[0]
    V = []
    q = (bvec / (np.linalg.norm(bvec) + 1e-12)).astype(np.float32)
    for _ in range(k_max):
        V.append(q.copy())
        w = G @ q
        for _ in range(2):                      # 两遍 Gram-Schmidt
            for u in V:
                w -= (u @ w) * u
        n = float(np.linalg.norm(w))
        if n < tol:
            break
        q = w / n
    if not V:
        return np.zeros((d, 0), np.float32)
    return np.array(V).T.astype(np.float32, copy=False)


def krylov_curve(X, y, folds, k_list, ns='main', k_max=None):
    """每折内生成 Krylov 基，投影做 OLS，给出 held-out R² 曲线。"""
    k_max = k_max or max(k_list)
    nk = fkey(ns, folds)
    G_all = gcache(('all', nk), X)
    b_all = gmat(X, y)
    n = len(y)
    pred = {k: np.zeros(n) for k in k_list}
    for j, f in enumerate(folds):
        test = np.zeros(n, bool)
        test[f] = True
        G_tr = G_all - gcache(('f', nk, j), X[f])
        b_tr = b_all - gmat(X[f], y[f])
        V = krylov_basis(G_tr, b_tr, k_max)
        Z = X @ V                                   # 全量投影（float32，勿提升 dtype）
        for k in k_list:
            kk = min(k, Z.shape[1])
            A = np.column_stack([np.ones(int((~test).sum())), Z[~test, :kk].astype(np.float64)])
            coef, *_ = np.linalg.lstsq(A, y[~test], rcond=None)
            pred[k][test] = np.column_stack([np.ones(int(test.sum())),
                                             Z[test, :kk].astype(np.float64)]) @ coef
    return [dict(k=int(k), r2=r2_score(y, pred[k])) for k in k_list]


def auc_mwu(score, pos):
    """AUC = P(score_pos > score_neg)，Mann-Whitney。"""
    r = stats.rankdata(score)
    n1 = int(pos.sum())
    n0 = len(pos) - n1
    if n1 == 0 or n0 == 0:
        return float('nan')
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---------------------------------------------------------------- 加载

def load_E():
    """只读 embedding 权重，不加载整模型。

    §8 跑后修订：GLM-4-9B 用 from_pretrained(device_map=auto) 在本机 segfault（分片加载踩到
    多进程/显存边界）。embedding 表是独立张量，直接从 safetensors 取，省 18 GB 内存且更快。
    """
    import json as _json
    from safetensors import safe_open
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    p = Path(MODEL_PATH)
    cfg = _json.loads((p / 'config.json').read_text(encoding='utf-8'))
    tied = bool(cfg.get('tie_word_embeddings', False))
    pats = ('embed_tokens.weight', 'word_embeddings.weight')

    def pick(keys):
        for k in keys:
            if any(k.endswith(pt) for pt in pats):
                return k
        return None

    idxf = p / 'model.safetensors.index.json'
    if idxf.exists():
        mp = _json.loads(idxf.read_text(encoding='utf-8'))['weight_map']
        key = pick(list(mp.keys()))
        if key is None:
            raise RuntimeError('index 里找不到 embedding 键')
        with safe_open(str(p / mp[key]), framework='pt', device='cpu') as f:
            E = f.get_tensor(key).float().numpy()
        say('embedding 来源 %s（%s）' % (mp[key], key))
        return tok, E, tied
    f1 = p / 'model.safetensors'
    if f1.exists():
        with safe_open(str(f1), framework='pt', device='cpu') as f:
            key = pick(list(f.keys()))
            E = f.get_tensor(key).float().numpy()
        say('embedding 来源 %s（%s）' % (f1.name, key))
        return tok, E, tied
    # 兜底：常规加载
    import torch
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16,
                                             device_map='cpu', low_cpu_mem_usage=True)
    m.eval()
    E = m.get_input_embeddings().weight.detach().float().cpu().numpy()
    del m
    gc.collect()
    return tok, E, tied


def cjk1(s):
    return len(s) == 1 and ('\u3400' <= s <= '\u9fff' or '\uf900' <= s <= '\ufaff')


def all_cjk(s):
    return len(s) > 0 and all(cjk1(c) for c in s)


def build_pools(tok, E):
    """按预注册 §1.1 建池。

    关键（本轮踩过）：Qwen 的 convert_ids_to_tokens 返回的是 byte-level BPE 映射串（'çļĦ'），
    必须用 tok.decode([i]) 才拿到真实文本。生僻字被切成多个 token 时，单个 token 解码为 '\ufffd'
    —— 这就是"字节残片"，它没有可查的频率，必须排除出回归（§3 陷阱 1）。
    """
    from wordfreq import zipf_frequency as zf
    V = E.shape[0]
    dec = [tok.decode([i]) for i in range(V)]
    idx_char, z_char = [], []
    idx_word, z_word = [], []
    n_byte = 0
    lat_pool = []
    for i, s in enumerate(dec):
        if (not s) or ('\ufffd' in s):
            n_byte += 1
            continue
        if cjk1(s):
            z = zf(s, 'zh')
            if z > 0:
                idx_char.append(i)
                z_char.append(z)
        elif len(s) >= 2 and all_cjk(s):
            z = zf(s, 'zh')
            if z > 0:
                idx_word.append(i)
                z_word.append(z)
        else:
            t = s.strip()
            if len(t) >= 3 and t.isascii() and t.isalpha():
                lat_pool.append((i, t))
    say('全表解码：CHAR %d / WORD %d / 残片或无文本 %d / 拉丁候选 %d'
        % (len(idx_char), len(idx_word), n_byte, len(lat_pool)))
    if SMOKE:
        for nm in ('char', 'word'):
            arr_i = idx_char if nm == 'char' else idx_word
            arr_z = z_char if nm == 'char' else z_word
            if len(arr_i) > 1500:
                pick = RNG.choice(len(arr_i), 1500, replace=False)
                if nm == 'char':
                    idx_char = [arr_i[j] for j in sorted(pick)]
                    z_char = [arr_z[j] for j in sorted(pick)]
                else:
                    idx_word = [arr_i[j] for j in sorted(pick)]
                    z_word = [arr_z[j] for j in sorted(pick)]
    lat = []
    if len(lat_pool) > 6000:
        pick = RNG.choice(len(lat_pool), 6000, replace=False)
        lat_pool = [lat_pool[j] for j in sorted(pick)]
    for i, t in lat_pool:
        z = zf(t.lower(), 'en')
        if z > 0:
            lat.append((i, t, z))
    say('池：CHAR %d / WORD %d / LATIN %d / 残片 %d'
        % (len(idx_char), len(idx_word), len(lat), n_byte))
    return dict(char=(np.array(idx_char, int), np.array(z_char, float)),
                word=(np.array(idx_word, int), np.array(z_word, float)),
                latin=lat, n_byte=n_byte, V=V, dec=dec)


# ---------------------------------------------------------------- B1 形状

def b1_shape(X, y, folds, G_all, b_all, ns='b1'):
    """t = 折内拟合的频率方向投影；比较线性 / 二次 / 12 分箱查表的 held-out R²。"""
    n = len(y)
    pred = {k: np.zeros(n) for k in ('M1', 'M2', 'M3')}
    slopes = []
    nk = fkey(ns, folds)
    b_all = gmat(X, y)
    for j, f in enumerate(folds):
        # 修（§8 跑后）：原实现把掩码写反（te 实为 train），B1 的所有 R² 是 in-sample。
        test = np.zeros(n, bool)
        test[f] = True
        tr = ~test
        Gf = gcache(('f', nk, j), X[f])
        bf = gmat(X[f], y[f])
        w = solve_w(G_all - Gf, b_all - bf)
        t_tr = X[tr] @ w
        t_te = X[test] @ w
        a, b0 = np.polyfit(t_tr, y[tr], 1)
        slopes.append(float(a))
        pred['M1'][test] = a * t_te + b0
        c2 = np.polyfit(t_tr, y[tr], 2)
        pred['M2'][test] = np.polyval(c2, t_te)
        qs = np.quantile(t_tr, np.linspace(0, 1, 13))
        qs[0] -= 1e-9
        qs[-1] += 1e-9
        bins = np.clip(np.digitize(t_tr, qs[1:-1]), 0, 11)
        means = np.array([y[tr][bins == k].mean() if (bins == k).any() else y[tr].mean()
                          for k in range(12)])
        pred['M3'][test] = means[np.clip(np.digitize(t_te, qs[1:-1]), 0, 11)]
    r = {k: r2_score(y, v) for k, v in pred.items()}
    d = r['M3'] - r['M1']
    # 逐样本 (M3-M1) 差的 bootstrap CI（配对）
    sq1 = (y - pred['M1']) ** 2
    sq3 = (y - pred['M3']) ** 2
    st = float(((y - y.mean()) ** 2).sum())
    diff_terms = (sq1 - sq3) / st
    lo, hi = bootstrap_ci(diff_terms, fn=np.sum)
    out = dict(r2=r['M1'], r2_quad=r['M2'], r2_bin=r['M3'],
               delta_bin_minus_lin=float(d), delta_ci=[lo, hi],
               slope_median=float(np.median(slopes)))
    if d >= 0.05:
        v = 'SATURATING'
    elif d < 0.02 and r['M1'] >= 0.50:
        v = 'LINEAR_LOG_SCALAR'
    elif d < 0.05 and r['M1'] >= 0.50:
        v = 'WEAKLY_NONLINEAR'
    else:
        v = 'WEAK_SCALAR'
    out['verdict'] = v
    say('B1 线性 M1 R²=%.4f | 二次 M2 %.4f | 12箱 M3 %.4f | Δ(M3−M1)=%+.4f CI[%+.4f,%+.4f] => %s'
        % (r['M1'], r['M2'], r['M3'], d, lo, hi, v))
    return out


# ---------------------------------------------------------------- B2 维数

def b2_dim(X, y, folds, G_all, b_all, rho_cols):
    order = np.argsort(-np.abs(rho_cols))
    ks = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    n = len(y)
    nk_b2 = fkey('main', folds)
    Gfs = []
    for j, f in enumerate(folds):
        Gf = gcache(('f', nk_b2, j), X[f])
        bf = gmat(X[f], y[f])
        Gfs.append((f, Gf, bf))
    curve = []
    for k in ks:
        if k > X.shape[1]:
            continue
        sel = order[:k]
        pred = np.zeros(n)
        for f, Gf, bf in Gfs:
            Gs = G_all[np.ix_(sel, sel)] - Gf[np.ix_(sel, sel)]
            bs = b_all[sel] - bf[sel]
            w = solve_w(Gs, bs)
            pred[f] = X[f][:, sel] @ w
        curve.append(dict(k=int(k), r2=r2_score(y, pred)))
    rmax = max(c['r2'] for c in curve)
    k_star = next(c['k'] for c in curve if c['r2'] >= 0.95 * rmax)
    say('B2 坐标曲线（机会 0）：' + ' '.join('k=%d:%.3f' % (c['k'], c['r2']) for c in curve))
    say('B2 k*(95%% 饱和) = %d（max R²=%.4f）' % (k_star, rmax))

    # PLS-1 曲线（§8：Krylov 等价实现，5 折 held-out）
    pls_curve = krylov_curve(X, y, folds, [1, 2, 4, 8, 16, 32], ns='main')
    rmax_p = max(c['r2'] for c in pls_curve)
    kp = next(c['k'] for c in pls_curve if c['r2'] >= 0.95 * rmax_p)
    say('B2 PLS/Krylov 曲线（5 折）：' + ' '.join('k=%d:%.3f' % (c['k'], c['r2']) for c in pls_curve))
    say('B2 PLS k* = %d（max R²=%.4f）' % (kp, rmax_p))

    if k_star <= 4 and rmax >= 0.50:
        v1 = 'SINGLE_SCALAR'
    elif k_star <= 16:
        v1 = 'FEW_DIM'
    else:
        v1 = 'DISTRIBUTED'
    if (k_star <= 4) != (kp <= 4):
        v = 'AMBIGUOUS_DIM'
    else:
        v = v1
    density = ('LOW_DIM_DENSE' if (kp <= 4 and k_star > 16) else
               ('SPARSE_LOW_DIM' if (kp <= 4 and k_star <= 16) else 'DISTRIBUTED'))
    out = dict(coords=curve, k_star=int(k_star), r2_max=float(rmax),
               pls=pls_curve, pls_k_star=int(kp), pls_r2_max=float(rmax_p),
               density=density, verdict_coord=v1, verdict=v)
    say('B2 密度标签 => %s（PLS 低维但坐标分散 = 频率方向是稠密投影，不是某个物理坐标）' % density)
    say('B2 判决 => %s（坐标 %s / PLS k*=%d）' % (v, v1, kp))
    return out


# ---------------------------------------------------------------- B3 范数

def b3_norm(X, y, folds, G_all, b_all, lnorm):
    rho_n = A.spearman(lnorm, y)
    ln_c = lnorm - lnorm.mean()
    _nk = fkey('main', folds)
    w_norm = solve_w(gcache(('all', _nk), X), gmat(X, ln_c))
    w_freq = solve_w(gcache(('all', _nk), X), gmat(X, y))
    cs = float(w_freq @ w_norm / (np.linalg.norm(w_freq) * np.linalg.norm(w_norm) + 1e-12))
    r2_full, _ = cv_fast(X, y, folds, 'main')
    r2_ctrl, _ = cv_res(X, y, folds, ln_c, w_norm, ns_src='main')
    ratio = r2_ctrl / r2_full if r2_full > 1e-9 else float('nan')
    if ratio >= 0.70:
        v = 'FREQ_INDEPENDENT_OF_NORM'
    elif ratio <= 0.50:
        v = 'FREQ_IS_NORM'
    else:
        v = 'PARTIALLY_NORM'
    out = dict(rho_log_norm_zipf=float(rho_n), cos_freq_norm=cs,
               r2_full=float(r2_full), r2_after_removing_norm=float(r2_ctrl),
               ratio=float(ratio), verdict=v)
    say('B3 ρ(log‖e‖,zipf)=%+.4f | cos(w_freq,w_norm)=%+.4f | R²_full %.4f → 扣范数后 %.4f (保留 %.2f) => %s'
        % (rho_n, cs, r2_full, r2_ctrl, ratio, v))
    return out


# ---------------------------------------------------------------- B5 可拧性

def b5_knob(E, mu, Gsub, sub_idx, idx_char, dec, w_freq, slope, n_probe=120):
    """E1 静态预检：沿频率方向平移，最近邻的频率是否跟着动。

    §8 跑后修订 ①：单剂量 δ=1 会把向量推到 2 sd 之外，"最近邻没变"分不清
    「旋钮不接邻域」与「推太远」⇒ 改剂量-反应（0.25/0.5/1/2 个 zipf 单位）。
    §8 跑后修订 ②：全表（151 936×2560）逐向量 gemv ×480 在本机超过 10 分钟，
    改为 ①一次 GEMM ②搜索集限定为「可定义频率的 24 421 行 + 1 724 残片」
    （无频率的行不可能是"频率邻居"，这是口径收窄而非放松）。
    """
    from wordfreq import zipf_frequency as zf
    what = (w_freq / (np.linalg.norm(w_freq) + 1e-12)).astype(np.float32)
    d1 = 1.0 / abs(slope) if slope != 0 else float('nan')
    if not np.isfinite(d1) or d1 > 50:
        return dict(verdict='KNOB_NOT_LOCAL', note='slope 太小，δ 不可用', delta_1zipf=float(d1))
    row_of = {int(i): r for r, i in enumerate(sub_idx)}
    sgn = 1.0 if slope > 0 else -1.0
    pick = RNG.choice(len(idx_char), min(n_probe, len(idx_char)), replace=False)
    doses = [0.25, 0.5, 1.0, 2.0]
    qs, meta = [], []
    for dz in doses:
        for j in pick:
            i = int(idx_char[j])
            v = (E[i] - mu) + np.float32(sgn * dz * d1) * what
            v = (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)
            qs.append(v)
            meta.append((row_of[i], dz, float(zf(dec[i], 'zh'))))
    Q = np.asarray(qs, np.float32)
    S = Gsub @ Q.T
    for k, (r, _, _) in enumerate(meta):
        S[r, k] = -9.0
    nns = np.argmax(S, axis=0)
    del S, Q
    gc.collect()
    curve = []
    for dz in doses:
        arr = []
        for k, (r, dd, z0) in enumerate(meta):
            if dd != dz:
                continue
            z1 = zf(dec[int(sub_idx[int(nns[k])])], 'zh')
            if z1 > 0 and z0 > 0:
                arr.append(z1 - z0)
        arr = np.array(arr)
        curve.append(dict(dose=dz, n=int(len(arr)),
                          median=float(np.median(arr)) if len(arr) else float('nan'),
                          frac_pos=float((arr > 0).mean()) if len(arr) else float('nan'),
                          frac_near=float((np.abs(arr - dz) <= 0.5).mean()) if len(arr) else float('nan')))
        say('B5 剂量 %.2f zipf：Δzipf 中位 %+.3f（同号 %.2f，|Δ−剂量|≤0.5 占 %.2f，n=%d）'
            % (dz, curve[-1]['median'], curve[-1]['frac_pos'], curve[-1]['frac_near'], len(arr)))
    meds = np.array([c['median'] for c in curve])
    mono = bool(np.all(np.diff(meds) > -1e-6)) and bool(meds[-1] > 0.15)
    if mono and meds[2] >= 0.5:
        v = 'SMOOTH_KNOB'
    elif mono:
        v = 'COMPRESSED_KNOB'
    else:
        v = 'KNOB_NOT_LOCAL'
    out = dict(delta_1zipf=float(d1), slope=float(slope), curve=curve,
               monotone=mono, verdict=v, search_set=int(len(sub_idx)),
               note='E1 静态预检，非干预；搜索集=可定义频率行+残片')
    say('B5 判决 => %s（单调 %s；δ=1 时 Δzipf 中位 %+.3f）' % (v, mono, meds[2]))
    return out


# ---------------------------------------------------------------- B6 语义残差

def b6_semantics(tok, E, mu, w_freq, dec, idx_char):
    def matvec(words):
        ws, vs = [], []
        for w in words:
            ids = tok.encode(w, add_special_tokens=False)
            if not ids:
                continue
            ws.append(w)
            vs.append(E[ids].mean(0))
        return ws, np.array(vs)

    what = w_freq / (np.linalg.norm(w_freq) + 1e-12)
    out = {}
    for nm, words in (('COLOR', list(A.COLOR_HEX.keys())), ('FRUIT', A.FRUIT)):
        ws, V = matvec(words)
        X = V - mu
        before = A.within_mean(X)
        t = X @ what
        Xr = X - np.outer(t, what)
        after = A.within_mean(Xr)
        # 零分布：同规模随机在表单字子集（扣频率后的 within）
        nulls = []
        for _ in range(N_PERM):
            s = RNG.choice(idx_char, len(ws), replace=False)
            Z = E[s] - mu
            tz = Z @ what
            nulls.append(A.within_mean(Z - np.outer(tz, what)))
        nul = np.array(nulls)
        ret = after / before if abs(before) > 1e-9 else float('nan')
        if ret >= 0.70:
            v = 'SEMANTICS_ORTHOGONAL_TO_FREQ'
        elif ret <= 0.40:
            v = 'SEMANTICS_ALONG_FREQ'
        else:
            v = 'PARTIALLY_FREQ'
        out[nm] = dict(n=len(ws), within_before=float(before), within_after=float(after),
                       retain=float(ret), null_mean=float(nul.mean()),
                       null_p95=float(np.percentile(nul, 95)),
                       z_vs_null=float((after - nul.mean()) / (nul.std() + 1e-12)),
                       verdict=v)
        say('B6 %s n=%d within %.4f → 扣频率 %.4f（保留 %.2f，同规模随机 %.4f±%.4f）=> %s'
            % (nm, len(ws), before, after, ret, nul.mean(), nul.std(), v))
    return out


# ---------------------------------------------------------------- B7 类型可预测性

def b7_type(t_vals, groups, names):
    """单标量 t 预测单位类型的 one-vs-rest AUC。"""
    out = {}
    pairs = [(0, 1), (0, 2), (1, 2)]
    for a, b in pairs:
        m = (groups == a) | (groups == b)
        score = t_vals[m]
        pos = (groups[m] == a)
        auc = auc_mwu(score, pos)
        bs = []
        idx = np.where(m)[0]
        for _ in range(200 if not SMOKE else 40):
            j = RNG.choice(idx, len(idx), replace=True)
            bs.append(auc_mwu(t_vals[j], groups[j] == a))
        lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        key = '%s_vs_%s' % (names[a], names[b])
        auc_adj = max(auc, 1.0 - auc)
        out[key] = dict(auc=float(auc), auc_direction_free=float(auc_adj), ci=[lo, hi],
                        higher_group=names[a] if auc >= 0.5 else names[b],
                        n_a=int(pos.sum()), n_b=int((~pos).sum()))
        say('B7 AUC(%s) = %.3f（方向自由 %.3f，t 更高的是 %s）CI[%.3f, %.3f]  n=%d/%d'
            % (key, auc, auc_adj, out[key]['higher_group'], lo, hi, pos.sum(), (~pos).sum()))
    a = out['CHAR_vs_WORD']['auc_direction_free']
    out['verdict'] = 'TYPE_IS_FREQUENCY' if a >= 0.90 else (
        'TYPE_HAS_FREQ_FREE_PART' if a < 0.70 else 'TYPE_PARTLY_FREQUENCY')
    say('B7 判决 => %s' % out['verdict'])
    return out


# ---------------------------------------------------------------- main

def main():
    t0 = time.time()
    say('=' * 78)
    say('Q8-B 频率旋钮量纲化  model=%s  SMOKE=%s  seed=%d  folds=%d perm=%d'
        % (MODEL_PATH, SMOKE, SEED, N_FOLD, N_PERM))
    say('=' * 78)
    tok, E, tied = load_E()
    V, d = E.shape
    mu = E.mean(0)
    say('E', E.shape, 'tied', tied)
    say('‖μ‖=%.4f  平均‖e‖=%.4f' % (float(np.linalg.norm(mu)), float(np.linalg.norm(E, axis=1).mean())))

    pools = build_pools(tok, E)
    dec = pools['dec']
    ic, zc = pools['char']
    iw, zw = pools['word']
    idx = np.concatenate([ic, iw])
    y_zipf = np.concatenate([zc, zw])
    y_freq = np.power(10.0, y_zipf - 9.0)          # per-billion 概率
    ylog = y_zipf - y_zipf.mean()

    X = (E[idx] - mu).astype(np.float32)
    say('POOL n=%d  d=%d  zipf 范围 [%.2f, %.2f]' % (len(idx), d, y_zipf.min(), y_zipf.max()))

    # —— 零分布：打乱 y 的 R² 必须≈0，否则 §2 的绝对阈值作废
    perms = []
    _f0 = folds_of(len(y_zipf))
    for _ in range(N_PERM):
        yp = y_zipf - y_zipf.mean()
        RNG.shuffle(yp)
        r, _ = cv_fast(X, yp, _f0, 'null')
        perms.append(r)
    perms = np.array(perms)
    say('零分布（打乱 y）R² 均 %.4f p95 %.4f —— %s'
        % (perms.mean(), np.percentile(perms, 95),
           '≈0，绝对阈值可用' if abs(perms.mean()) < 0.01 else
           '显著为负 ⇒ 本设计 d/n 比下有过拟合，R² 需与零分布并列读（§8）'))

    folds = folds_of(len(idx))
    G_all = gcache(('all', fkey('main', folds)), X)
    b_all = gmat(X, ylog)
    rho_cols = corr_cols(X, ylog)

    payload = dict(config=dict(model=MODEL_PATH, smoke=SMOKE, seed=SEED, folds=N_FOLD,
                               n_perm=N_PERM, n_pool=int(len(idx)),
                               n_char=int(len(ic)), n_word=int(len(iw)),
                               n_latin=len(pools['latin']), n_byte=int(pools['n_byte']),
                               tied=tied, d=int(d), mu_norm=float(np.linalg.norm(mu))),
                   null=dict(r2_mean=float(perms.mean()), r2_p95=float(np.percentile(perms, 95))))

    # B1：zipf 轴 + freq 轴
    say('-' * 78)
    b1 = b1_shape(X, ylog, folds, G_all, b_all)
    ylog_f = y_freq - y_freq.mean()
    r2_freq, _ = cv_fast(X, ylog_f, folds, 'main')
    b1['r2_freq_axis'] = float(r2_freq)
    b1['freq_vs_zipf_gap'] = float(b1['r2'] - r2_freq)
    say('B1 原始频率轴 R²=%.4f（vs zipf 轴 %.4f，差 %+.4f）=> 刻度%s对数'
        % (r2_freq, b1['r2'], b1['freq_vs_zipf_gap'],
           '是' if b1['freq_vs_zipf_gap'] > 0.15 else '不是明显'))
    b1['r2_gain_vs_null'] = float(b1['r2'] - perms.mean())
    payload['B1'] = b1
    say('B1 校准增益 R²_obs − R²_null = %.4f（零分布 %.4f）' % (b1['r2_gain_vs_null'], perms.mean()))

    # B2
    payload['B2'] = b2_dim(X, ylog, folds, G_all, b_all, rho_cols)

    # B3
    lnorm_pool = np.log(np.linalg.norm(E[idx], axis=1))
    payload['B3'] = b3_norm(X, ylog, folds, G_all, b_all, lnorm_pool)

    # B4 的本地素材：单坐标最强坐标
    j_top = int(np.argmax(np.abs(rho_cols)))
    payload['single_coord'] = dict(coord=j_top, pearson=float(rho_cols[j_top]),
                                   spearman=A.spearman(X[:, j_top], y_zipf))
    say('B4 单坐标 %d：Pearson %+.4f / Spearman %+.4f（Q8-A 报的是 Spearman 0.802 量级）'
        % (j_top, rho_cols[j_top], payload['single_coord']['spearman']))

    # B5（需要全表 cos 矩阵）
    say('-' * 78)
    byte_like0 = np.array([i for i, t in enumerate(dec) if ('\ufffd' in t) or (t == '')], int)
    sub_idx = np.concatenate([idx, byte_like0])
    Gsub = (E[sub_idx] - mu).astype(np.float32, copy=False)
    _nrm = np.linalg.norm(Gsub, axis=1, keepdims=True)
    Gsub /= np.maximum(_nrm, 1e-9)
    del _nrm
    gc.collect()
    say('B5 搜索集 %d 行（POOL %d + 残片 %d），已归一化 %.2f GB'
        % (len(sub_idx), len(idx), len(byte_like0), Gsub.nbytes / 1e9))
    w_full = solve_w(G_all, b_all)
    payload['B5'] = b5_knob(E, mu, Gsub, sub_idx, ic, dec, w_full, b1['slope_median'])
    del Gsub
    gc.collect()

    # B6
    payload['B6'] = b6_semantics(tok, E, mu, w_full, dec, ic)

    # B7
    t_all = X @ w_full
    grp = np.concatenate([np.zeros(len(ic), int), np.ones(len(iw), int)])
    byte_like = byte_like0
    say('B7 残片行数 %d（口径：decode 出 \\ufffd 或空）' % len(byte_like))
    if SMOKE:
        byte_like = byte_like[:300]
    t_b = (E[byte_like] - mu) @ w_full if len(byte_like) else np.array([])
    say('B7 字节残片投影 t 中位 %.3f（CHAR %.3f / WORD %.3f）'
        % (np.median(t_b) if len(t_b) else float('nan'),
           np.median(t_all[grp == 0]), np.median(t_all[grp == 1])))
    if len(byte_like) > 30:
        tt = np.concatenate([t_all, t_b])
        gg = np.concatenate([grp, np.full(len(t_b), 2)])
    else:
        tt, gg = t_all, grp
    payload['B7'] = b7_type(tt, gg, ['CHAR', 'WORD', 'BYTE'])

    # LATIN 稳健性（定性对照）：§8 改为小样本单折（本机 float64 BLAS 在大 Gram 上极慢）
    say('-' * 78)
    lat = pools['latin']
    if len(lat) > 300:
        li = np.array([x[0] for x in lat])
        lz = np.array([x[2] for x in lat], float)
        if len(li) > 1500:
            sel = RNG.choice(len(li), 1500, replace=False)
            li, lz = li[sel], lz[sel]
        Xl = (E[li] - mu).astype(np.float32)
        yl = lz - lz.mean()
        perm = RNG.permutation(len(yl))
        cut = int(len(yl) * 0.8)
        tr, te = perm[:cut], perm[cut:]
        Gtr = (Xl[tr].T @ Xl[tr]).astype(np.float64)
        wl = solve_w(Gtr, gmat(Xl[tr], yl[tr]))
        tl_tr, tl_te = Xl[tr] @ wl, Xl[te] @ wl
        a, b0 = np.polyfit(tl_tr, yl[tr], 1)
        lin_te = a * tl_te + b0
        qs = np.quantile(tl_tr, np.linspace(0, 1, 13))
        bins = np.clip(np.digitize(tl_tr, qs[1:-1]), 0, 11)
        means = np.array([yl[tr][bins == j].mean() for j in range(12)])
        bin_te = means[np.clip(np.digitize(tl_te, qs[1:-1]), 0, 11)]
        r_lin = r2_score(yl[te], lin_te)
        r_bin = r2_score(yl[te], bin_te)
        verdict_l = ('SATURATING' if (r_bin - r_lin) >= 0.05 else
                     ('LINEAR_LOG_SCALAR' if (r_bin - r_lin) < 0.02 and r_lin >= 0.50 else
                      ('WEAKLY_NONLINEAR' if r_lin >= 0.50 else 'WEAK_SCALAR')))
        payload['LATIN'] = dict(n=int(len(li)), r2_lin=float(r_lin), r2_bin=float(r_bin),
                                delta=float(r_bin - r_lin), verdict=verdict_l,
                                same_shape_as_cjk=bool(verdict_l == b1['verdict']),
                                note='单折 80/20，n<=1500（快速对照）')
        say('LATIN 对照 n=%d：M1 %.4f / M3 %.4f（Δ %+.4f）判决 %s（与 CJK %s）'
            % (len(li), r_lin, r_bin, r_bin - r_lin, verdict_l,
               '一致' if verdict_l == b1['verdict'] else '不一致'))
    else:
        payload['LATIN'] = dict(n=len(lat), note='样本不足，跳过')

    # 判决汇总
    verd = {}
    for k in ('B1', 'B2', 'B3', 'B5', 'B6', 'B7'):
        v = payload.get(k, {})
        if k == 'B6':
            verd[k] = {kk: vv.get('verdict') for kk, vv in v.items()}
        else:
            verd[k] = v.get('verdict')
    payload['verdicts'] = verd
    say('-' * 78)
    for k, v in verd.items():
        say(k, v)
    payload['runtime_s'] = round(time.time() - t0, 1)
    say('耗时 %.1f s' % payload['runtime_s'])

    path = provenance.write_result('pat_q8b', _jsonable(payload), model_path=MODEL_PATH,
                                   seed=SEED, tool='tools/pat_q8b.py')
    say('RESULT', path)
    txt = path.with_suffix('.log.txt')
    txt.write_text('\n'.join(LOG) + '\n', encoding='utf-8', newline='')
    say('LOG', txt)


if __name__ == '__main__':
    main()
