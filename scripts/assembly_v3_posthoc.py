# -*- coding: utf-8 -*-
"""AM-v3 事后描述（不参与预注册判定）：事件自身效应里有多少是跨事件共享的，事件特有部分是否是「对自身分布的温度变化」。
  1 共享份额：发现集事件自身状态效应 Δh_j 中平均效应所占平方和份额 ‖Δ̄h‖² / mean‖Δh_j‖²；对数概率效应在各事件候选集上同样计算（候选集内中心化）。
  2 温度：在候选集上把事件自身对数概率效应（中心化）对 Y00（中心化）回归：R² 与斜率（斜率 > 0 = 变尖，< 0 = 变平）。
  3 事后候选 C1：B1 + (s·βs_g + l·βl_g)·(Y00 − 候选均值)，β 只在发现集候选集上拟合，在确认集上计算 KL（候选模型，非预注册）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'tools')]

from shared.dist_protocol import assembly as S  # noqa: E402
import fu_materials_v2 as MAT  # noqa: E402
import fusion_v1 as FU1  # noqa: E402
import assembly_v2 as AM2  # noqa: E402
import assembly_v3 as AM3  # noqa: E402

K = 3


def cz(v):
    return v - v.mean()


def main(model_path):
    crit = json.loads(AM3.PREREG.read_text(encoding='utf-8'))['criteria']
    od = AM2.NATIVE / FU1.snapshot(model_path)[:16]
    sp = AM2.split()
    D, C = sp['disc'], sp['conf']
    out = {}
    for lang in AM3.LANGS:
        A = np.load(od / 'AM-v2' / f'logp_{lang}.npy').astype(np.float32)
        n, _, V = A.shape
        Y = A.reshape(n, 2, 2, K, V)
        Hh = np.load(od / 'AM-v3' / f'h_{lang}.npy').reshape(n, 2, 2, K, -1)
        cand = [np.unique(np.argsort(-A[i], 1)[:, :crit['top_k_cand']]) for i in range(n)]
        r = {}
        for g in range(K):
            oh = np.stack([np.stack(S.own_effects(Hh[j, :, :, g])) for j in D])        # [16, 2, d]
            share_h = [float((oh[:, f].mean(0) ** 2).sum() / (oh[:, f] ** 2).sum(1).mean()) for f in (0, 1)]
            ds, dl, _ = S.slot_effects(Y[D, :, :, g])
            share_y, r2, slope = [], [], []
            X, T = [[], []], [[], []]
            for f, mean_e in ((0, ds), (1, dl)):
                num = den = 0.0
                for j in D:
                    c = cand[j]
                    e = cz(S.own_effects(Y[j, :, :, g])[f][c])
                    x = cz(Y[j, 0, 0, g][c])
                    num += (cz(mean_e[c]) ** 2).sum()
                    den += (e ** 2).sum()
                    b = (x @ e) / (x @ x)
                    slope.append(float(b))
                    r2.append(float(1 - ((e - b * x) ** 2).sum() / (e ** 2).sum()))
                    X[f].append(x)
                    T[f].append(cz((S.own_effects(Y[j, :, :, g])[f] - mean_e)[c]))
                share_y.append(float(num / den))
            beta = [float(np.concatenate(X[f]) @ np.concatenate(T[f]) / (np.concatenate(X[f]) @ np.concatenate(X[f]))) for f in (0, 1)]
            kl_b1, kl_c1 = [], []
            for i in C:
                y00 = Y[i, 0, 0, g]
                xc = y00 - y00[cand[i]].mean()
                for s_, l_ in ((0, 1), (1, 0), (1, 1)):
                    b1 = y00 + s_ * ds + l_ * dl
                    kl_b1.append(S.kl_restricted(Y[i, s_, l_, g], b1, cand[i]))
                    kl_c1.append(S.kl_restricted(Y[i, s_, l_, g], b1 + (s_ * beta[0] + l_ * beta[1]) * xc, cand[i]))
            r[MAT.SLOTS[g]] = {'shared_share_state': dict(zip(('style', 'logic'), share_h)),
                               'shared_share_logit_cand': dict(zip(('style', 'logic'), share_y)),
                               'temp_r2_median': float(np.median(r2)), 'temp_slope_median': float(np.median(slope)),
                               'beta_disc': dict(zip(('style', 'logic'), beta)),
                               'kl_B1': float(np.mean(kl_b1)), 'kl_C1_posthoc': float(np.mean(kl_c1))}
        out[lang] = r
        print(lang, json.dumps(r, ensure_ascii=False), flush=True)
    p = AM3.report_path(model_path)
    rep = json.loads(p.read_text(encoding='utf-8'))
    rep['posthoc'] = out
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding='utf-8')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'models/hf/qwen3-4b')
