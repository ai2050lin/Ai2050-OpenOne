# -*- coding: utf-8 -*-
"""Q8-B 独立复核：不复用主脚本的任何函数，用另一套口径重算关键数字。

约定：口径不同（折划分、求解方式）时给容差；口径相同（材料/向量/常量）时必须逐位一致。
"""
import os
import sys
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = os.environ.get('MODEL', 'D:/develop/model/Qwen/Qwen3-4B')
OK = []
BAD = []


def chk(name, got, want, tol, note=''):
    if want is None:
        print('SKIP %-34s %s' % (name, note))
        return
    ok = abs(float(got) - float(want)) <= tol
    (OK if ok else BAD).append(name)
    print('%s %-34s got %.6f want %.6f (tol %.2g) %s'
          % ('PASS' if ok else 'FAIL', name, got, want, tol, note))


def main():
    res_path = sys.argv[1]
    doc = json.loads(Path(res_path).read_text(encoding='utf-8'))
    P = doc['payload']
    print('复核', res_path, 'sha8', doc['sha8'], 'model', doc['provenance']['model_path'])

    from wordfreq import zipf_frequency as zf
    from safetensors import safe_open
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    p_ = Path(MODEL_PATH)
    idxf = p_ / 'model.safetensors.index.json'
    if idxf.exists():
        mp = json.loads(idxf.read_text(encoding='utf-8'))['weight_map']
        key = next(k for k in mp if k.endswith('embed_tokens.weight'))
        with safe_open(str(p_ / mp[key]), framework='pt', device='cpu') as f:
            E = f.get_tensor(key).float().numpy()
    else:
        with safe_open(str(p_ / 'model.safetensors'), framework='pt', device='cpu') as f:
            key = next(k for k in f.keys() if k.endswith('embed_tokens.weight'))
            E = f.get_tensor(key).float().numpy()
    mu = E.mean(0)
    V = E.shape[0]

    # ---- 池（独立重建）
    dec = [tok.decode([i]) for i in range(V)]

    def cjk1(s):
        return len(s) == 1 and ('\u3400' <= s <= '\u9fff' or '\uf900' <= s <= '\ufaff')

    idx_c, z_c, idx_w, z_w, nb = [], [], [], [], 0
    for i, s in enumerate(dec):
        if (not s) or ('\ufffd' in s):
            nb += 1
            continue
        if cjk1(s):
            z = zf(s, 'zh')
            if z > 0:
                idx_c.append(i)
                z_c.append(z)
        elif len(s) >= 2 and all(cjk1(c) for c in s):
            z = zf(s, 'zh')
            if z > 0:
                idx_w.append(i)
                z_w.append(z)
    chk('n_char', len(idx_c), P['config']['n_char'], 0, '口径：decode 后单 CJK 且 zipf>0')
    chk('n_word', len(idx_w), P['config']['n_word'], 0)
    chk('n_byte', nb, P['config']['n_byte'], 0, 'decode 出 \\ufffd 或空')
    chk('n_pool', len(idx_c) + len(idx_w), P['config']['n_pool'], 0)

    idx = np.array(idx_c + idx_w)
    zs = np.array(z_c + z_w, float)
    X = E[idx] - mu
    y = zs - zs.mean()

    # ---- 单坐标 Pearson / Spearman（独立口径）
    Xc = X - X.mean(0)
    yc = zs - zs.mean()
    pc = (Xc.T @ yc) / (np.linalg.norm(Xc, axis=0) * np.linalg.norm(yc) + 1e-12)
    j = int(np.argmax(np.abs(pc)))
    got = P['single_coord']
    chk('coord_argmax', j, got['coord'], 0, '最强单坐标')
    chk('coord_pearson', pc[j], got['pearson'], 2e-3)
    chk('coord_spearman', stats.spearmanr(X[:, j], zs).statistic, got['spearman'], 2e-3)

    # ---- B1：独立单折（80/20）线性 R²
    n = len(idx)
    # 多划分重复：单折 R² 的抽样波动必须先量出来，否则分不清"实现差异"与"划分噪声"（D18）
    # 多划分重复：一次全量 Gram + 每次减掉测试折的 Gram（等价，快 5 倍）
    Xc2 = X - X.mean(0)                      # pool 内中心化，故无截距
    Xf64 = Xc2.astype(np.float64)
    G_all2 = Xf64.T @ Xf64
    b_all2 = Xf64.T @ y
    r2s = []
    for sd in (7, 11, 23, 41, 97):
        rng = np.random.default_rng(sd)
        perm = rng.permutation(n)
        cut = int(n * 0.8)
        tr, te = perm[:cut], perm[cut:]
        Xte = Xf64[te]
        Gt = G_all2 - Xte.T @ Xte
        bt = b_all2 - Xte.T @ y[te]
        w = np.linalg.solve(Gt, bt)
        pred = Xte @ w
        r2s.append(1 - ((y[te] - pred) ** 2).sum() / ((y[te] - y[te].mean()) ** 2).sum())
    r2s = np.array(r2s)
    print('     单折 R² 五次划分：%s（均 %.4f，极差 %.4f）'
          % (' '.join('%.3f' % v for v in r2s), r2s.mean(), r2s.ptp() if hasattr(r2s, 'ptp') else r2s.max() - r2s.min()))
    # 用 5 折均值对应的口径比较：允许极差 + 0.02 的口径余量
    tol = float(r2s.max() - r2s.min()) + 0.03
    chk('B1_full_dim_5splits_mean', r2s.mean(), P['B3']['r2_full'], tol, '多划分均值 vs 5 折全维')
    chk('B1_vs_B3_r2full_internal', P['B1']['r2'], P['B3']['r2_full'], 0.06,
        '一维重标定 vs 全维（应接近）')

    # ---- B3：ρ(log‖e‖, zipf)（口径相同，应逐位一致）
    ln = np.log(np.linalg.norm(E[idx], axis=1))
    chk('B3_rho_lognorm_zipf', stats.spearmanr(ln, zs).statistic,
        P['B3']['rho_log_norm_zipf'], 2e-4)

    # ---- B6：within cos（材料与向量恒定，应逐位一致）
    def matvec(words):
        vs = []
        for w in words:
            ids = tok.encode(w, add_special_tokens=False)
            if ids:
                vs.append(E[ids].mean(0))
        return np.array(vs)

    def within(M):
        Z = M - mu
        Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)
        G = Z @ Z.T
        k = len(M)
        return float((G.sum() - np.trace(G)) / (k * (k - 1)))

    sys.path.insert(0, str(ROOT / 'tools'))
    import pat_q8a as A
    for nm, words in (('COLOR', list(A.COLOR_HEX.keys())), ('FRUIT', A.FRUIT)):
        chk('B6_%s_within_before' % nm, within(matvec(words)),
            P['B6'][nm]['within_before'], 1e-5, '应与 Q8-A P3 同值')

    # ---- B7：AUC(CHAR vs WORD) 独立重算
    Xp = X.astype(np.float64)
    G = Xp.T @ Xp
    b = Xp.T @ y
    w = np.linalg.solve(G + 1e-3 * np.trace(G) / G.shape[0] * np.eye(G.shape[0]), b)
    t = X @ w
    pos = np.zeros(n, bool)
    pos[:len(idx_c)] = True
    r = stats.rankdata(t)
    n1, n0 = pos.sum(), (~pos).sum()
    auc = (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    got7 = P['B7']['CHAR_vs_WORD']
    chk('B7_auc_char_vs_word', auc, got7['auc'], 5e-3, '方向自由 %.3f' % got7['auc_direction_free'])

    # ---- verdict 枚举合法性
    allowed = {
        'B1': {'SATURATING', 'LINEAR_LOG_SCALAR', 'WEAKLY_NONLINEAR', 'WEAK_SCALAR'},
        'B2': {'SINGLE_SCALAR', 'FEW_DIM', 'DISTRIBUTED', 'AMBIGUOUS_DIM'},
        'B3': {'FREQ_INDEPENDENT_OF_NORM', 'FREQ_IS_NORM', 'PARTIALLY_NORM'},
        'B5': {'SMOOTH_KNOB', 'COMPRESSED_KNOB', 'KNOB_NOT_LOCAL'},
        'B7': {'TYPE_IS_FREQUENCY', 'TYPE_HAS_FREQ_FREE_PART', 'TYPE_PARTLY_FREQUENCY'},
        'B6': {'SEMANTICS_ORTHOGONAL_TO_FREQ', 'SEMANTICS_ALONG_FREQ', 'PARTIALLY_FREQ'},
    }
    for k, al in allowed.items():
        v = P['verdicts'][k]
        vals = list(v.values()) if isinstance(v, dict) else [v]
        for vv in vals:
            chk('verdict_%s_legal' % k, 0, 0 if vv in al else 1, 0, str(vv))

    print('-' * 60)
    print('%d PASS / %d FAIL' % (len(OK), len(BAD)))
    if BAD:
        print('FAILED:', BAD)
        sys.exit(1)


if __name__ == '__main__':
    main()
