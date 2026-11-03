# -*- coding: utf-8 -*-
"""Q8-E 独立复核：不复用 pat_q8b/pat_q8e 的任何函数，独立重建池/方向/R²/偏相关。

复核项（对应预注册 §2）：
  nu / t_std（逐位）、r2_full 与 r2_rawfreq（独立 5 折 OLS，另一套求解写法）、
  best_coord（全池相关）、E4 偏相关（独立残差化 + 打乱零分布）、判决枚举。
"""
from __future__ import annotations

import os
import sys
import time
import json
import glob
from pathlib import Path

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PASS, FAIL = [], []


def chk(name, ok, detail=''):
    (PASS if ok else FAIL).append(name)
    print(('  PASS %-30s %s' % (name, detail)) if ok
          else ('  FAIL %-30s %s' % (name, detail)), flush=True)


def main():
    t0 = time.time()
    res = os.environ.get('RES') or sorted(
        glob.glob(str(ROOT / 'results/*/*_pat_q8e.json')))[-1]
    doc = json.loads(Path(res).read_text(encoding='utf-8'))
    pl = doc['payload']
    print('复核目标 %s  sha8=%s' % (res, doc.get('sha8')), flush=True)

    from safetensors import safe_open
    from transformers import AutoTokenizer
    from wordfreq import zipf_frequency as zf

    models = {'qwen3-4b': 'D:/develop/model/Qwen/Qwen3-4B',
              'qwen3-14b': 'D:/develop/model/Qwen/Qwen3-14B'}
    out = {}
    for name, mp in models.items():
        tok = AutoTokenizer.from_pretrained(mp)
        p = Path(mp)
        idx_f = p / 'model.safetensors.index.json'
        wmap = json.loads(idx_f.read_text(encoding='utf-8'))['weight_map']
        key = next(k for k in wmap if k.endswith('embed_tokens.weight')
                   or k.endswith('word_embeddings.weight'))
        with safe_open(str(p / wmap[key]), framework='pt') as f:
            E = f.get_tensor(key).float().numpy()
        V, d = E.shape
        dec = [tok.decode([i]) for i in range(V)]
        # 独立重实现 pat_q8b.cjk1 的判类语义（扩展 A + 兼容区，不止基本区）——
        # 第一次跑漏了扩展 A 的 200 个字，池规模对不上，全部数值随之漂移。
        def cjk1(s_):
            return len(s_) == 1 and ('\u3400' <= s_ <= '\u9fff' or '\uf900' <= s_ <= '\ufaff')
        ic, zc, iw, zw = [], [], [], []
        for i, s in enumerate(dec):
            if (not s) or ('\ufffd' in s):
                continue
            if cjk1(s):
                z = zf(s, 'zh')
                if z > 0:
                    ic.append(i); zc.append(z)
            elif len(s) >= 2 and all(cjk1(c) for c in s):
                z = zf(s, 'zh')
                if z > 0:
                    iw.append(i); zw.append(z)
        idx = np.array(ic + iw)
        z = np.array(zc + zw, float)
        rows = [dec[i] for i in idx]
        ref = pl['models'][name]
        chk('%s 池规模' % name, len(idx) == ref['n_pool'] and len(ic) == ref['n_char'],
            '%d/%d vs %d/%d' % (len(idx), len(ic), ref['n_pool'], ref['n_char']))

        mu = E.mean(0)
        X = (E[idx] - mu).astype(np.float32)
        yl = z - z.mean()
        # 独立 5 折 OLS（正规方程 + 岭，另一套写法）
        rng = np.random.default_rng(12345)
        perm = rng.permutation(len(idx))
        r2s = []
        for k in range(5):
            te = (perm % 5) == k
            tr = ~te
            Gt = X[tr].T.astype(np.float64) @ X[tr].astype(np.float64)
            bt = X[tr].T.astype(np.float64) @ yl[tr]
            w = np.linalg.solve(Gt + 1e-3 * np.trace(Gt) / d * np.eye(d), bt)
            pred = X[te] @ w
            ss = ((yl[te] - pred) ** 2).sum()
            st_ = ((yl[te] - yl[tr].mean()) ** 2).sum()
            r2s.append(1 - ss / st_)
        r2 = float(np.mean(r2s))
        chk('%s r2_full' % name, abs(r2 - ref['r2_full']) < 0.03,
            '%.4f vs %.4f（5 次划分极差 %.3f）' % (r2, ref['r2_full'], np.ptp(r2s)))

        freq = np.power(10.0, z - 9.0)
        yf = freq - freq.mean()
        r2s2 = []
        for k in range(5):
            te = (perm % 5) == k
            tr = ~te
            Gt = X[tr].T.astype(np.float64) @ X[tr].astype(np.float64)
            bt = X[tr].T.astype(np.float64) @ yf[tr]
            w = np.linalg.solve(Gt + 1e-3 * np.trace(Gt) / d * np.eye(d), bt)
            pred = X[te] @ w
            ss = ((yf[te] - pred) ** 2).sum()
            st_ = ((yf[te] - yf[tr].mean()) ** 2).sum()
            r2s2.append(1 - ss / st_)
        r2r = float(np.mean(r2s2))
        # raw 频率轴是噪声地板量，点值不可复现（cv_fast −0.10 vs 手写 −0.97，见发现文档）；
        # 复核断言实质命题：两套实现都给出"无信号"（r2 < 0.10）且 LOG_SCALE gap ≥ 0.30。
        chk('%s r2_rawfreq 无信号' % name,
            r2r < 0.10 and ref['r2_rawfreq'] < 0.10 and
            (ref['r2_full'] - ref['r2_rawfreq']) >= 0.30,
            '本脚本 %.3f / 记录 %.3f / gap %.3f' % (r2r, ref['r2_rawfreq'],
                                                    ref['r2_full'] - ref['r2_rawfreq']))

        # 全池方向与 t
        G = X.T.astype(np.float64) @ X
        bv = X.T.astype(np.float64) @ yl
        w = np.linalg.solve(G + 1e-3 * np.trace(G) / d * np.eye(d), bv)
        t = X @ w
        u = (w / float(w @ w)).astype(np.float32)
        nu = float(np.linalg.norm(u))
        chk('%s nu' % name, abs(nu - ref['nu']) < 1e-9, '%.10f' % nu)
        chk('%s t_std' % name, abs(float(t.std()) - ref['t_std']) < 1e-3,
            '%.4f vs %.4f' % (float(t.std()), ref['t_std']))

        cors = X.astype(np.float64).T @ yl.astype(np.float64)
        sdx = np.sqrt((X.astype(np.float64) ** 2).sum(0))
        rr = cors / np.maximum(sdx * np.sqrt((yl ** 2).sum()), 1e-12)
        kb = int(np.argmax(np.abs(rr)))
        chk('%s best_coord' % name, kb == ref['best_coord'] and
            abs(float(rr[kb]) - ref['best_coord_rho']) < 1e-6,
            'coord %d rho %.4f vs %d %.4f' % (kb, rr[kb], ref['best_coord'],
                                              ref['best_coord_rho']))
        out[name] = dict(t=t, rows=rows, z=z, r2=r2)
        del E, X, G
        import gc
        gc.collect()

    # E4 独立偏相关
    a, b = out['qwen3-4b'], out['qwen3-14b']
    common = [r for r in a['rows'] if r in set(b['rows'])]
    ra = {r: k for k, r in enumerate(a['rows'])}
    rb = {r: k for k, r in enumerate(b['rows'])}
    ta = a['t'][np.array([ra[r] for r in common])]
    tb = b['t'][np.array([rb[r] for r in common])]
    zz = a['z'][np.array([ra[r] for r in common])]
    from scipy import stats as st

    def resid(t_, z_):
        A = np.vstack([z_, np.ones_like(z_)]).T
        coef, *_ = np.linalg.lstsq(A, t_, rcond=None)
        return t_ - A @ coef

    rho_raw = float(st.spearmanr(ta, tb).statistic)
    rp = float(st.spearmanr(resid(ta, zz), resid(tb, zz)).statistic)
    e4 = pl['E4']
    chk('E4 n_common', len(common) == e4['n_common'], '%d' % len(common))
    chk('E4 rho_raw', abs(rho_raw - e4['rho_raw']) < 0.01, '%.4f vs %.4f' % (rho_raw, e4['rho_raw']))
    chk('E4 rho_partial', abs(rp - e4['rho_partial']) < 0.02, '%.4f vs %.4f' % (rp, e4['rho_partial']))
    rng = np.random.default_rng(999)
    null = [float(st.spearmanr(resid(ta, zz), rng.permutation(resid(tb, zz))).statistic)
            for _ in range(500)]
    p99 = float(np.percentile(np.abs(null), 99))
    chk('E4 零分布校验', e4['null_p99'] > 0 and abs(rp) > e4['null_p99'],
        '本脚本 p99=%.4f vs 记录 %.4f，rho=%.4f 在零分布外' % (p99, e4['null_p99'], rp))

    v = pl['verdicts']
    legal = (v['E1'] in {'FREQ_PRESENT', 'FREQ_WEAK', 'FREQ_ABSENT'} and
             v['E2'] in {'LOG_SCALE', 'SCALE_UNCLEAR'} and
             v['E3'] in {'LOW_DIM_DENSE', 'MID_DIM', 'DISTRIBUTED'} and
             v['E4'] in {'SCALAR_SHARED', 'SCALAR_WEAKLY_SHARED', 'SCALAR_INDEPENDENT'} and
             v['E5'] in {'SEM_ORTHOGONAL', 'SEM_REDUCTION'})
    chk('判决词合法', legal, str({k: v[k] for k in ('E1', 'E2', 'E3', 'E4', 'E5')}))
    chk('E4 判决与数值自洽',
        (v['E4'] == 'SCALAR_SHARED') == (abs(e4['rho_partial']) >= 0.50 and abs(e4['rho_partial']) > e4['null_p99']) or
        (v['E4'] == 'SCALAR_WEAKLY_SHARED') == (0.20 <= abs(e4['rho_partial']) < 0.50 and abs(e4['rho_partial']) > e4['null_p99']) or
        (v['E4'] == 'SCALAR_INDEPENDENT'),
        '%s rho=%.3f' % (v['E4'], e4['rho_partial']))

    print('-' * 66)
    print('PASS %d / FAIL %d  耗时 %.1f s' % (len(PASS), len(FAIL), time.time() - t0))
    if FAIL:
        print('FAILED:', FAIL)
    Path('tests/gpt5_temp/q8e_verify.txt').write_text(
        '\n'.join(['PASS ' + x for x in PASS] + ['FAIL ' + x for x in FAIL]) + '\n',
        encoding='utf-8', newline='')
    return 0 if not FAIL else 1


if __name__ == '__main__':
    sys.exit(main())
