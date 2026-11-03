# -*- coding: utf-8 -*-
"""Q8-E：频率方向稳定性（Qwen3-4B vs Qwen3-14B，共享词表，纯静态）。

判据见 research/gpt5/docs/PAT_Q8E_PREREG_v1.md。
可跨模型的仪器只有逐 token 标量预测 t = (E - mu) @ w（共享行空间）——
坐标轴跨模型不对齐，cos(w_freq) 无意义（预注册 §0）。
实现注意：pat_q8b 的 Gram 缓存键含 (ns, 折指纹) 但不含 X 身份，
两个模型必须用不同的 ns 前缀，否则跨模型缓存冲突（R-MTH-24 的推论）。
"""
from __future__ import annotations

import os
import sys
import gc
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import pat_q8b as B  # noqa: E402  复用与 Q8-B 完全相同的口径

SMOKE = os.environ.get('SMOKE', '0') == '1'
SEED = int(os.environ.get('SEED', '20261101'))
OUT_DIR = ROOT / 'results' / time.strftime('%Y-%m-%d')
K_LIST = list(range(1, 33)) if not SMOKE else [1, 2, 4, 8, 16]

LOG = []
def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)

MODELS = {
    'qwen3-4b': 'D:/develop/model/Qwen/Qwen3-4B',
    'qwen3-14b': 'D:/develop/model/Qwen/Qwen3-14B',
}


def analyse(name, path):
    """单模型全套静态分析（与 Q8-B 同口径）。返回 dict（含 _t/_rows/_z 供 E4）。"""
    B.MODEL_PATH = path
    B.SMOKE = SMOKE
    B.RNG = np.random.default_rng(SEED + hash(name) % 1000)
    tok, E, tied = B.load_E()
    V, d = E.shape
    say('[%s] V=%d d=%d tied=%s' % (name, V, d, tied))
    pools = B.build_pools(tok, E)
    dec = pools['dec']
    ic, zc = pools['char']
    iw, zw = pools['word']
    idx = np.concatenate([ic, iw])
    z = np.concatenate([zc, zw])
    rows = [dec[i] for i in idx]
    say('[%s] 池 n=%d（CHAR %d / WORD %d）' % (name, len(idx), len(ic), len(iw)))

    mu = E.mean(0)
    X = (E[idx] - mu).astype(np.float32)
    yl = z - z.mean()

    G, bv = B.gram_xy(X, yl)
    w = B.solve_w(G, bv)
    t = (X @ w)
    nu = float(np.linalg.norm(w / (w @ w)))
    med_norm = float(np.median(np.linalg.norm(E[idx], axis=1)))

    folds = B.folds_of(len(idx))
    ns = '%s|' % name
    r2_full, _ = B.cv_fast(X, yl, folds, ns=ns + 'main')
    yl_sh = B.RNG.permutation(yl)
    r2_null, _ = B.cv_fast(X, yl_sh, folds, ns=ns + 'null')
    freq = np.power(10.0, z - 9.0)
    yf = freq - freq.mean()
    # raw 频率轴是噪声地板量：点值依赖求解器数值细节（本轮 cv_fast −0.10 vs 手写 −0.97），
    # 必须 3 个折种子报均值±极差，判决按不等式（无信号 + gap）而非点复现。
    r2_raw_seeds = []
    for sd in range(3):
        B.RNG = np.random.default_rng(SEED + 500 + sd)
        f2 = B.folds_of(len(idx))
        v2_, _ = B.cv_fast(X, yf, f2, ns=ns + 'raw%d' % sd)
        r2_raw_seeds.append(float(v2_))
    r2_raw = float(np.mean(r2_raw_seeds))
    kc = B.krylov_curve(X, yl, folds, K_LIST, ns=ns + 'main')
    sem = B.b6_semantics(tok, E, mu, w, dec, ic)

    # 最强单坐标（全池，R-MET-34 口径）
    Xd = X.astype(np.float64)
    cors = Xd.T @ yl.astype(np.float64)
    sdx = np.sqrt((Xd ** 2).sum(0))
    rr = cors / np.maximum(sdx * np.sqrt((yl.astype(np.float64) ** 2).sum()), 1e-12)
    kbest = int(np.argmax(np.abs(rr)))

    out = dict(name=name, V=int(V), d=int(d), n_pool=int(len(idx)),
               n_char=int(len(ic)), n_word=int(len(iw)),
               w_norm=float(np.linalg.norm(w)), nu=nu, med_norm=med_norm,
               t_std=float(t.std()),
               r2_full=float(r2_full), r2_null=float(r2_null),
               r2_rawfreq=float(r2_raw), r2_rawfreq_seeds=[float(x) for x in r2_raw_seeds],
               krylov=[dict(k=int(e['k']), r2=float(e['r2'])) for e in kc],
               sem=sem, best_coord=kbest, best_coord_rho=float(rr[kbest]),
               _t=t, _rows=rows, _z=z)
    del E, X, Xd, G
    gc.collect()
    return out


def main():
    t0 = time.time()
    say('=' * 78)
    say('Q8-E 频率方向稳定性（静态）  SMOKE=%s  seed=%d' % (SMOKE, SEED))
    say('=' * 78)
    res = {}
    for name, path in MODELS.items():
        res[name] = analyse(name, path)

    # ---------- E4：共享行标量比较（主判）
    a, b = res['qwen3-4b'], res['qwen3-14b']
    common = [r for r in a['_rows'] if r in set(b['_rows'])]
    say('共享 token 文本 %d（4B %d / 14B %d）' % (len(common), a['n_pool'], b['n_pool']))
    ra = {r: k for k, r in enumerate(a['_rows'])}
    rb = {r: k for k, r in enumerate(b['_rows'])}
    ii = np.array([ra[r] for r in common])
    jj = np.array([rb[r] for r in common])
    ta, tb, zz = a['_t'][ii], b['_t'][jj], a['_z'][ii]
    from scipy import stats as st

    def resid(t_, z_):
        A = np.vstack([z_, np.ones_like(z_)]).T
        coef, *_ = np.linalg.lstsq(A, t_, rcond=None)
        return t_ - A @ coef

    rho_raw = float(st.spearmanr(ta, tb).statistic)
    ra_r, rb_r = resid(ta, zz), resid(tb, zz)
    rho_part = float(st.spearmanr(ra_r, rb_r).statistic)
    rng = np.random.default_rng(SEED + 7)
    null = [float(st.spearmanr(ra_r, rng.permutation(rb_r)).statistic) for _ in range(1000)]
    p99 = float(np.percentile(np.abs(null), 99))
    if abs(rho_part) >= 0.50 and abs(rho_part) > p99:
        v4 = 'SCALAR_SHARED'
    elif abs(rho_part) >= 0.20 and abs(rho_part) > p99:
        v4 = 'SCALAR_WEAKLY_SHARED'
    else:
        v4 = 'SCALAR_INDEPENDENT'
    say('E4: rho_raw=%.4f  rho_partial=%.4f  |null| p99=%.4f  -> %s'
        % (rho_raw, rho_part, p99, v4))

    # ---------- 判决
    m = res['qwen3-14b']
    v1 = ('FREQ_PRESENT' if m['r2_full'] >= 0.5 else
          'FREQ_WEAK' if m['r2_full'] >= 0.2 else 'FREQ_ABSENT')
    gap = m['r2_full'] - m['r2_rawfreq']
    v2 = 'LOG_SCALE' if gap >= 0.30 else 'SCALE_UNCLEAR'
    full = m['r2_full']
    kstar = next((e['k'] for e in m['krylov'] if e['r2'] >= full - 0.02), None)
    v3 = 'DISTRIBUTED' if kstar is None else ('LOW_DIM_DENSE' if kstar <= 16 else 'MID_DIM')
    rets = [s['retain'] for s in m['sem'].values() if np.isfinite(s.get('retain', float('nan')))]
    v5 = 'SEM_ORTHOGONAL' if rets and min(rets) >= 0.95 else 'SEM_REDUCTION'

    payload = dict(
        models={k: {kk: vv for kk, vv in v.items() if not kk.startswith('_')}
                for k, v in res.items()},
        E4=dict(n_common=int(len(common)), rho_raw=rho_raw, rho_partial=rho_part,
                null_p99=p99, verdict=v4),
        verdicts=dict(E1=v1, E2=v2, E3=v3, E4=v4, E5=v5,
                      kstar=kstar, scale_gap=float(gap),
                      shape_stable=bool(v1 == 'FREQ_PRESENT' and v2 == 'LOG_SCALE'
                                        and v3 == 'LOW_DIM_DENSE' and v5 == 'SEM_ORTHOGONAL')),
        smoke=SMOKE, seed=SEED, runtime_s=round(time.time() - t0, 1),
    )
    say('判决：E1=%s E2=%s E3=%s(k*=%s) E4=%s E5=%s  shape_stable=%s'
        % (v1, v2, v3, kstar, v4, v5, payload['verdicts']['shape_stable']))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    import provenance
    path = provenance.write_result('pat_q8e', payload,
                                   model_path=MODELS['qwen3-14b'],
                                   seed=SEED, tool='tools/pat_q8e.py')
    say('RESULT', path)
    path.with_suffix('.log.txt').write_text('\n'.join(LOG) + '\n',
                                            encoding='utf-8', newline='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
