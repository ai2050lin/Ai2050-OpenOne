# -*- coding: utf-8 -*-
"""MF-v1 事后诊断（不进判定）：H1 在第 6 层即饱和，可能只是前几个 token 的局部信息（n-gram），不是持续的模式状态。
1  n-gram 嵌入基线：把当前与前 n−1 个 token 的嵌入（hs[0]）拼接作特征，同一流程训练、在确认集共用 token 位置评估。
2  「深共用」位置：当前与前 n−1 个 token 都是共用 token 的位置（局部窗口内没有风格专属 token），逐层两两准确率。
3  H4 的逐层 AUC（已在报告 profile 中）与 H2 的单侧效应汇总。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from shared.dist_protocol import mode_field as F  # noqa: E402
import mode_field_v1 as MF  # noqa: E402
import xm_replicate_v1 as XM  # noqa: E402


def main(model_path='models/hf/qwen3-4b', int8=False):
    _, key = XM.key_of(model_path, int8)
    z = MF.load_fields(key)
    rep = json.loads(MF.report_path(model_path, int8).read_text(encoding='utf-8'))
    lam = rep['H1']['lambda_rel']
    G = [int(x) for x in z['grid']]
    y, tok, sp, pos, text, sh = z['style'], z['tok'], z['split'], z['pos'], z['text'], z['shared'].astype(bool)
    D, Cf = sp == 'disc', sp == 'conf'
    H0 = z['Hg'][:, 0].astype(np.float32)
    names = {f'{a}-{b}': f'{MF.STY[a]}-{MF.STY[b]}' for a in range(4) for b in range(a + 1, 4)}
    out = {'ngram_baseline': {}, 'deep_shared': {}}
    # 前 n−1 个 token 的行号（同一段文本内；句首不足的位置用自身代替并标记）
    prev = np.arange(len(tok)) - 1
    ok1 = (pos >= 1)
    for n in (2, 3, 4):
        feats = [H0]
        valid = np.ones(len(tok), bool)
        for j in range(1, n):
            idx = np.arange(len(tok)) - j
            v = pos >= j
            valid &= v
            feats.append(H0[np.where(v, idx, np.arange(len(tok)))])
        X = np.concatenate(feats, 1)
        m = F.ridge_multi(X[D & valid], y[D & valid], 4, lam)
        pa = F.pairwise_acc(F.scores(m, X[Cf & valid]), y[Cf & valid], tok[Cf & valid])
        out['ngram_baseline'][str(n)] = {names[k]: v for k, v in pa.items()}
        print('ngram', n, {names[k]: round(v, 3) for k, v in pa.items()})
    # 深共用：当前与前 n−1 个 token 都是共用 token
    for n in (2, 3, 4, 6):
        deep = sh.copy()
        for j in range(1, n):
            idx = np.arange(len(tok)) - j
            deep &= (pos >= j) & sh[np.where(pos >= j, idx, 0)]
        row = {}
        for gi, l in enumerate(G):
            X = z['Hg'][:, gi].astype(np.float32)
            m = F.ridge_multi(X[D], y[D], 4, lam)
            S = F.scores(m, X[Cf & deep])
            pa = F.pairwise_acc(S, y[Cf & deep], tok[Cf & deep])
            row[str(l)] = {names[k]: v for k, v in pa.items()}
        cnt = {MF.STY[s]: int((Cf & deep & (y == s)).sum()) for s in range(4)}
        out['deep_shared'][str(n)] = {'n_conf_positions': cnt, 'by_layer': row}
        print('deep', n, cnt, {l: round(float(np.nanmean(list(v.values()))), 3) for l, v in row.items()})
    out['H4_profile'] = rep['H4']['profile']
    out['note'] = '事后诊断，不进 MF-v1 判定'
    MF.save(model_path, int8, {'posthoc': out})


if __name__ == '__main__':
    main(*(sys.argv[1:2] or []), int8='--int8' in sys.argv)
