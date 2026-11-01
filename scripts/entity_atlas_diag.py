# -*- coding: utf-8 -*-
"""EA-v1 事后诊断（看到 EA-v1 预注册结果之后设计，不属于预注册；结论只用于设计 EA-v2）。

EA-v1 的 G2 未支持，原因候选是材料功效不足（大小过门后只有 7 个「大」；颜色 9 类、discovery 约 85 个实体）。诊断两件事：
  D1 一对多颜色（只看正例 ≥ 10 的颜色），类别平衡的逻辑回归，正则强度与层由 discovery 交叉验证选定，heldout 评估一次；
     对照：类别多数基线、hs[0]。
  D2 连续目标：用模型自己对 11 种颜色的回答对数概率（问答式提示，完整答案，取各颜色表面形式的最大值，减去 11 色均值）
     作为目标，全部 210 个实体都可用（不依赖人工颜色标签，也不需要过门）。算子 = 岭回归（层与 λ 由 discovery 交叉验证），
     heldout 上报告 R²，对照：类别均值（只知道类别时的最好预测）与 hs[0]。再报告「去掉类别均值后的残差」上的 R²（属性是否落在个体差分里），
     以及中文编码上拟合的算子直接用于英文编码（目标仍是中文回答）的 R²（跨语言复用）。
用法：python scripts/entity_atlas_diag.py --model models/hf/qwen3-4b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from shared.dist_protocol import entity_atlas as E  # noqa: E402
import entity_atlas_run as R  # noqa: E402

PALETTE = ['red', 'yellow', 'green', 'blue', 'white', 'black', 'brown', 'orange', 'purple', 'pink', 'gray']


def color_logprobs(model_path, rows, out):
    f = out / 'color_logprobs.npz'
    if f.exists():
        z = np.load(f)
        return {'zh': z['zh'], 'en': z['en']}
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import build_ent_grammar as B
    torch.set_grad_enabled(False)
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map='cuda').eval()
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    res = {}
    for lang, prompt, forms in (('zh', B.COLOR_PROMPT_ZH, B.COLOR_FORMS_ZH), ('en', B.COLOR_PROMPT_EN, B.COLOR_FORMS_EN)):
        arr = np.zeros((len(rows), len(PALETTE)))
        for i, r in enumerate(rows):
            p = enc(prompt.replace('{X}', r['zh']).replace('{E}', r['en']))
            for j, c in enumerate(PALETTE):
                best = -1e9
                for a in forms[c]:
                    ai = enc(a)
                    lg = model(input_ids=torch.as_tensor([p + ai], device='cuda')).logits[0].float()
                    best = max(best, float(torch.log_softmax(lg[len(p) - 1:-1], -1).gather(-1, torch.as_tensor(ai, device='cuda')[:, None]).sum()))
                arr[i, j] = best
        res[lang] = arr
    np.savez_compressed(f, **res)
    return res


def ridge_cv(Xd, Yd, lams, folds, seed):
    rng = np.random.default_rng(seed)
    parts = np.array_split(rng.permutation(len(Xd)), folds)
    best = None
    for lam in lams:
        sse = 0.0
        for p in parts:
            tr = np.setdiff1d(np.arange(len(Xd)), p)
            m = E.ridge_fit(Xd[tr], Yd[tr], lam)
            sse += float(((E.ridge_predict(m, Xd[p]) - Yd[p]) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, lam)
    return best


def r2(Y, P):
    return float(1 - ((Y - P) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='models/hf/qwen3-4b')
    a = ap.parse_args()
    from shared.dist_protocol.census_runner import model_snapshot
    from sklearn.linear_model import LogisticRegression
    snap = model_snapshot(a.model)['sha256']
    src = R.native_dir(snap[:16] + '-bf16')
    rows = R.lexicon()
    ids = [r['id'] for r in rows]
    cat = np.array([r['cat'] for r in rows])
    split = np.array([E.split_of(i) for i in ids])
    d, h = split == 'discovery', split == 'heldout'
    z = np.load(src / 'codes.npz')
    X = {lang: z[lang].astype(np.float64) for lang in ('zh', 'en')}
    L = X['zh'].shape[1]
    gate = {g['id']: g for g in json.loads((src / 'gate.json').read_text(encoding='utf-8'))}
    lams = [1e-3, 1e-2, 1e-1, 1, 10]
    out = {'note': 'EA-v1 事后诊断：看到预注册结果后设计，不属于预注册，只用于设计 EA-v2', 'model_snapshot': snap}

    # ---- D2：模型自己的颜色回答分布（连续目标）----
    lp = color_logprobs(a.model, rows, src)
    d2 = {}
    for lang in ('zh', 'en'):
        Y = lp[lang] - lp[lang].mean(1, keepdims=True)
        cm = {c: Y[d & (cat == c)].mean(0) for c in set(cat)}
        base = np.stack([cm[c] for c in cat])
        curve = []
        for l in range(L):
            sse, lam = ridge_cv(X[lang][:, l][d], Y[d], lams, 5, 0)
            curve.append((sse, lam))
        lb = int(np.argmin([c[0] for c in curve]))
        lam = curve[lb][1]
        m = E.ridge_fit(X[lang][:, lb][d], Y[d], lam)
        P = E.ridge_predict(m, X[lang][:, lb])
        m0 = E.ridge_fit(X[lang][:, 0][d], Y[d], ridge_cv(X[lang][:, 0][d], Y[d], lams, 5, 0)[1])
        # 残差（个体差分）：目标与编码都减去 discovery 估计的类别均值
        xc = {c: X[lang][:, lb][d & (cat == c)].mean(0) for c in set(cat)}
        Xr = X[lang][:, lb] - np.stack([xc[c] for c in cat])
        Yr = Y - base
        mr = E.ridge_fit(Xr[d], Yr[d], ridge_cv(Xr[d], Yr[d], lams, 5, 0)[1])
        other = 'en' if lang == 'zh' else 'zh'
        d2[lang] = {'layer': lb, 'lambda_rel': lam, 'R2_operator_heldout': r2(Y[h], P[h]),
                    'R2_category_mean_heldout': r2(Y[h], base[h]),
                    'R2_hs0_heldout': r2(Y[h], E.ridge_predict(m0, X[lang][:, 0])[h]),
                    'R2_residual_operator_heldout': r2(Yr[h], E.ridge_predict(mr, Xr)[h]),
                    'R2_xlang_heldout': r2(Y[h], E.ridge_predict(m, X[other][:, lb])[h]),
                    'top1_agree_heldout': float(np.mean(P[h].argmax(1) == Y[h].argmax(1))),
                    'top1_agree_category_mean': float(np.mean(base[h].argmax(1) == Y[h].argmax(1)))}
    out['D2_color_answer_distribution'] = d2

    # ---- D1：一对多颜色，类别平衡逻辑回归 ----
    d1 = {}
    keep = np.array([gate[i]['color_zh'] > 0 and gate[i]['color_en'] > 0 for i in ids])
    col = np.array([r['color'] for r in rows])
    for c in PALETTE:
        y = (col == c).astype(int)
        if (y[keep] == 1).sum() < 10:
            continue
        res = {}
        for lang in ('zh', 'en'):
            best = None
            for l in range(0, L, 2):
                for C in (0.01, 0.1, 1.0):
                    Xd, yd = X[lang][:, l][keep & d], y[keep & d]
                    rng = np.random.default_rng(0); parts = np.array_split(rng.permutation(len(yd)), 5)
                    pred = np.zeros(len(yd), int)
                    for p in parts:
                        tr = np.setdiff1d(np.arange(len(yd)), p)
                        clf = LogisticRegression(C=C, class_weight='balanced', max_iter=2000).fit(Xd[tr], yd[tr])
                        pred[p] = clf.predict(Xd[p])
                    s = E.balanced_acc(yd, pred)
                    if best is None or s > best[0]:
                        best = (s, l, C)
            _, l, C = best
            clf = LogisticRegression(C=C, class_weight='balanced', max_iter=2000).fit(X[lang][:, l][keep & d], y[keep & d])
            hk = keep & h
            maj = E.category_majority_acc(y[keep], split[keep], cat[keep])
            res[lang] = {'layer': l, 'C': C, 'heldout_bal_acc': E.balanced_acc(y[hk], clf.predict(X[lang][:, l][hk])),
                         'category_majority_bal_acc': maj, 'n_pos': int((y[keep] == 1).sum())}
        d1[c] = res
    out['D1_one_vs_rest'] = d1
    dst = ROOT / 'research/entity_atlas/EA-v1-qwen3-4b/diagnostic_posthoc.json'
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
