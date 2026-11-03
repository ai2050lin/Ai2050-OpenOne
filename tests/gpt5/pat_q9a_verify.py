# -*- coding: utf-8 -*-
"""Q9-A 独立复核（R-MTH-25：每个 held-out 结论必须有独立代码路径复算并报差值）

口径刻意不同：
  - 全部从磁盘 npz 重算，不读主脚本内存；
  - float64 累加（主脚本 float16 存储 / float32 分析）；
  - 分类器换最近质心（主脚本 LogisticRegression）；
  - 组件比增加**范数归一化版** |proj|/‖comp‖（排除 MLP 输出范数天然更大的混淆，
    K-A 类混淆防线）——这是主脚本没有的新口径，其结果与未归一化版并列报告。
"""
import os
import sys
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261101
rng = np.random.default_rng(SEED + 7)
REG = ['colloq', 'acad', 'code']


def nearest_centroid_acc(Xtr, ytr, Xte, yte):
    """最近质心（逐类均值，余弦相似度）——与主脚本 LR 不同的独立实现。"""
    cents = []
    labs = np.unique(ytr)
    for c in labs:
        mu = Xtr[ytr == c].mean(0)
        cents.append(mu / (np.linalg.norm(mu) + 1e-9))
    C = np.array(cents)
    Xn = Xte / (np.linalg.norm(Xte, axis=1, keepdims=True) + 1e-9)
    pred = (Xn @ C.T).argmax(1)
    return float((labs[pred] == yte).mean())


def run(tag):
    npz = np.load(ROOT / 'tests' / 'gpt5' / 'result' / f'pat_q9a_{tag}.npz')
    R, A, M = npz['R'].astype(np.float64), npz['A'].astype(np.float64), npz['M'].astype(np.float64)
    y_reg, y_lang, is_disc = npz['y_reg'], npz['y_lang'], npz['is_disc']
    N, L, d = R.shape
    tr, te = is_disc, ~is_disc
    out = {'tag': tag, 'N': int(N), 'L': int(L)}

    # 1) 恒等式（float64，逐样本逐层）
    err = R[:, 1:] - (R[:, :-1] + A[:, :-1] + M[:, :-1])
    rel = np.linalg.norm(err, axis=-1) / (np.linalg.norm(R[:, 1:], axis=-1) + 1e-9)
    out['identity_rel_max_f64'] = float(rel.max())

    # Rext 重构（与主脚本同定义）
    Rext = np.zeros((N, L + 1, d))
    Rext[:, :L] = R
    Rext[:, L] = R[:, L - 1] + A[:, L - 1] + M[:, L - 1]

    # 2) 最近质心逐层 3-way（关键层）
    accs = {}
    for l in (0, 1, L // 2, L - 1, L):
        accs[l] = nearest_centroid_acc(Rext[tr, l], y_reg[tr], Rext[te, l], y_reg[te])
    out['nc_acc'] = {str(k): v for k, v in accs.items()}

    # 3) 组件投影：原始 + 范数归一化双口径
    comp = {}
    for lang in ('zh', 'en'):
        ml = y_lang == (0 if lang == 'zh' else 1)
        disc_l, conf_l = tr & ml, te & ml
        col, acad, code = 0, 1, 2
        rows = []
        for l in range(L):
            X = Rext[:, l + 1]
            mc = X[disc_l & (y_reg == col)].mean(0)
            ma = X[disc_l & (y_reg == acad)].mean(0)
            v = mc - ma
            v /= (np.linalg.norm(v) + 1e-9)
            pa = np.abs(A[conf_l, l] @ v)
            pm = np.abs(M[conf_l, l] @ v)
            na = np.linalg.norm(A[conf_l, l], axis=1) + 1e-9
            nm = np.linalg.norm(M[conf_l, l], axis=1) + 1e-9
            pa_n = pa / na   # 归一化：单位分量范数下的风格投影
            pm_n = pm / nm
            rows.append((float(np.median(pa)), float(np.median(pm)),
                         float(np.median(pa_n)), float(np.median(pm_n))))
        arr = np.array(rows)
        for b0, b1 in [(0, L // 4), (L // 4, L // 2), (L // 2, 3 * L // 4), (3 * L // 4, L)]:
            r_raw = np.median(arr[b0:b1, 0]) / (np.median(arr[b0:b1, 1]) + 1e-12)
            r_nrm = np.median(arr[b0:b1, 2]) / (np.median(arr[b0:b1, 3]) + 1e-12)
            comp[f'{lang}_{b0}-{b1}'] = {'raw_attn_over_mlp': float(r_raw),
                                         'normed_attn_over_mlp': float(r_nrm)}
    out['component'] = comp

    # 4) 跨语言迁移（最近质心，二分）
    col, acad = 0, 1
    yb = np.where(y_reg == col, 1, 0)
    l = L  # 末层 pre-norm 输出
    sel_zh = np.isin(y_reg, [col, acad]) & (y_lang == 0)
    sel_en = np.isin(y_reg, [col, acad]) & (y_lang == 1)

    def nc_bin(Xtr, ytr, Xte, yte):
        c1 = Xtr[ytr == 1].mean(0); c0 = Xtr[ytr == 0].mean(0)
        c1 /= np.linalg.norm(c1); c0 /= np.linalg.norm(c0)
        s1 = Xte @ c1; s0 = Xte @ c0
        return float(((s1 > s0).astype(int) == yte).mean())

    out['transfer_nc'] = {
        'in_zh': nc_bin(Rext[tr & sel_zh, l], yb[tr & sel_zh], Rext[te & sel_zh, l], yb[te & sel_zh]),
        'zh2en': nc_bin(Rext[tr & sel_zh, l], yb[tr & sel_zh], Rext[te & sel_en, l], yb[te & sel_en]),
        'in_en': nc_bin(Rext[tr & sel_en, l], yb[tr & sel_en], Rext[te & sel_en, l], yb[te & sel_en]),
        'en2zh': nc_bin(Rext[tr & sel_en, l], yb[tr & sel_en], Rext[te & sel_zh, l], yb[te & sel_zh]),
    }

    # 5) 与主脚本 summary 对照
    import glob
    cands = sorted(glob.glob(str(ROOT / 'results' / '*' / f'*pat_q9a_{tag}.json')))
    if cands:
        doc = json.load(open(cands[-1], encoding='utf-8'))
        res = doc['payload']
        out['main_vs_verify'] = {
            'main_acc0': res['v1']['acc0'], 'verify_nc_acc0': accs[0],
            'main_med_ratio': res['v2']['median_ratio'],
            'verify_med_raw_ratio': float(np.median([c['raw_attn_over_mlp'] for c in comp.values()])),
        }
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


if __name__ == '__main__':
    tags = sys.argv[1:] or ['qwen', 'glm']
    allout = {t: run(t) for t in tags}
    vp = ROOT / 'tests' / 'gpt5' / 'result' / 'pat_q9a_verify_summary.json'
    vp.write_text(json.dumps(allout, ensure_ascii=False, indent=1), encoding='utf-8', newline='')
    print('saved', vp)
