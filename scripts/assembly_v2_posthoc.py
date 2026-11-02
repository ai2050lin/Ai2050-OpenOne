# -*- coding: utf-8 -*-
"""AM-v2 事后诊断（不进判定）：FU-v2 的槽位内可加份额 0.90–0.95 与 P1 补全不优于相邻平均是否矛盾？
可加份额按候选集上的平方偏差计算，尾部低概率 token 的大幅变化占主导；KL 按真实概率加权，由高概率 token 决定。
这里在同一候选集上计算按概率加权的槽位内可加份额（权重 = 该 token 在 4 个格的平均概率），并按概率分层（前 5 / 其余）计算。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

import assembly_v2 as AM  # noqa: E402
import fusion_v1 as FU1  # noqa: E402


def main(model_path):
    od = AM.NATIVE / FU1.snapshot(model_path)[:16] / 'AM-v2'
    out = {}
    for lang in AM.LANGS:
        A = np.load(od / f'logp_{lang}.npy').astype(np.float64)
        n, _, V = A.shape
        Y = A.reshape(n, 2, 2, AM.K, V)
        acc = {'unweighted': [0, 0], 'weighted': [0, 0], 'top5': [0, 0], 'rest': [0, 0]}
        for i in range(n):
            cand = np.unique(np.argsort(-A[i], 1)[:, :20])
            for g in range(AM.K):
                Z = Y[i, :, :, g][:, :, cand]
                Z = Z - Z.mean(2, keepdims=True)                        # 按格中心化
                mu = Z.mean((0, 1), keepdims=True)
                ms, ml = Z.mean(1, keepdims=True), Z.mean(0, keepdims=True)
                add = (np.broadcast_to(ms - mu, Z.shape) ** 2 + np.broadcast_to(ml - mu, Z.shape) ** 2).sum((0, 1))
                inter = ((Z - ms - ml + mu) ** 2).sum((0, 1))
                w = np.exp(Y[i, :, :, g][:, :, cand]).mean((0, 1))
                order = np.argsort(-w)
                top = np.zeros(len(cand), bool); top[order[:5]] = True
                for key, mask, wt in (('unweighted', np.ones(len(cand), bool), np.ones(len(cand))), ('weighted', np.ones(len(cand), bool), w),
                                      ('top5', top, np.ones(len(cand))), ('rest', ~top, np.ones(len(cand)))):
                    acc[key][0] += float((add * wt)[mask].sum()); acc[key][1] += float((inter * wt)[mask].sum())
        out[lang] = {k: a / (a + b) for k, (a, b) in acc.items()}
        print(model_path, lang, {k: round(v, 3) for k, v in out[lang].items()}, flush=True)
    out['note'] = '事后诊断，不进 AM-v2 判定；份额 = (SS_s + SS_l) / (SS_s + SS_l + SS_sl)，随机期望 2/3'
    AM.save(model_path, {'posthoc_weighted_additivity': out})


if __name__ == '__main__':
    main(sys.argv[1])
