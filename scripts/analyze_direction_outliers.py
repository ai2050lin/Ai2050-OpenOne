# -*- coding: utf-8 -*-
"""方向 d 在残差流离群坐标上的能量检查（只读原生全场，不做前向）。

对每个模型、每层（GR-Q 原生全场 final 读出）：
  - 残差状态的逐坐标能量 E_j = mean_i h_ij²，按能量排序的前 k 个坐标（离群坐标）占总能量的比例；
  - d = mean(疑问) − mean(陈述)（discovery）的能量中，落在这些离群坐标上的比例；
  - 去掉离群坐标后的 d'，与 d 的余弦；以及 RMSNorm 之后（h / rms(h)，不乘增益）重算的 d_n 与 d 的余弦；
  - 陈述句在 d̂ 上投影的均值 μ_D 中，来自离群坐标的部分；
  - 各向同性随机方向在离群坐标上的期望能量占比 k/D，作为 INT-Q 随机对照的对比。

用法：python scripts/analyze_direction_outliers.py --out research/interventions/direction_outliers.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.dist_protocol import entangle as E  # noqa: E402

NATIVE = Path(os.environ.get('AI2050_NATIVE_DIR', 'D:/ai2050_native'))
MODELS = {'qwen3-4b': '6be1034f0b55a6df-bf16', 'glm4-9b': '64baf4f7e4fb31ae-bf16', 'qwen3-14b': '1e970d119494affd-bf16'}
KS = (1, 5, 20)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', required=True)
    ap.add_argument('--readout', default='final')
    args = ap.parse_args(argv)
    corpus = json.loads((ROOT / 'distributed_data/bundles/TG-GR-Q/v7/corpus.json').read_text(encoding='utf-8'))
    meta = {it['id']: it for it in corpus['items'] if not it.get('canary')}
    out = {'schema': 'direction_outliers.v1', 'source': 'TG-GR-Q v7 原生全场（discovery，%s 读出）' % args.readout,
           'k': list(KS), 'models': {}}
    for model, key in MODELS.items():
        dirs = sorted(glob.glob(str(NATIVE / 'GR-Q/TG-GR-Q-v7/*' / key / 'shard-*')))
        f = E.NativeField(dirs)
        ids = f.ids
        disc = np.array([meta.get(x, {}).get('split') == 'discovery' for x in ids])
        q = np.array([meta.get(x, {}).get('cond', {}).get('mood') == 'interrog' for x in ids])
        rows = []
        for L in f.layers:
            if L == 0 or L == max(f.layers):
                continue                                   # 嵌入层与最后一层（已过最终归一化）不参与
            X = f.X(args.readout, L)[disc]
            qq = q[disc]
            D = X.shape[1]
            Ej = (X.astype(np.float64) ** 2).mean(0)
            order = np.argsort(-Ej)
            d = X[qq].mean(0) - X[~qq].mean(0)
            dhat = d / np.linalg.norm(d)
            muD = float((X[~qq] @ dhat).mean())
            rec = {'layer_hs': int(L), 'dim': int(D), 'norm_d': float(np.linalg.norm(d)), 'mu_D': muD,
                   'mean_state_norm': float(np.linalg.norm(X, axis=1).mean())}
            for k in KS:
                top = order[:k]
                rec[f'state_energy_top{k}'] = float(Ej[top].sum() / Ej.sum())
                rec[f'd_energy_top{k}'] = float((d[top] ** 2).sum() / (d ** 2).sum())
                rec[f'muD_from_top{k}'] = float((X[~qq][:, top] @ dhat[top]).mean() / muD) if muD else None
                rec[f'random_expect_top{k}'] = k / D
            mask = np.ones(D, bool)
            mask[order[:5]] = False
            dp = d * mask
            rec['cos_d_dminus_top5'] = float(d @ dp / (np.linalg.norm(d) * np.linalg.norm(dp)))
            Xn = X / np.sqrt((X.astype(np.float64) ** 2).mean(1, keepdims=True))
            dn = Xn[qq].mean(0) - Xn[~qq].mean(0)
            rec['cos_d_rmsnormed'] = float(d @ dn / (np.linalg.norm(d) * np.linalg.norm(dn)))
            rec['top5_coords'] = [int(j) for j in order[:5]]
            rows.append(rec)
        out['models'][model] = rows
        print(f'== {model}')
        for r in rows:
            if r['layer_hs'] % 4 == 0:
                print(f"  L{r['layer_hs']:>2} state top1/5 {r['state_energy_top1']:.2f}/{r['state_energy_top5']:.2f} | "
                      f"d top1/5/20 {r['d_energy_top1']:.3f}/{r['d_energy_top5']:.3f}/{r['d_energy_top20']:.3f} "
                      f"(随机期望 top5 {r['random_expect_top5']:.4f}) | μD {r['mu_D']:+.1f}，其中 top5 占 {r['muD_from_top5']:.2f} | "
                      f"cos(d, d去top5) {r['cos_d_dminus_top5']:.3f} cos(d, d_rms) {r['cos_d_rmsnormed']:.3f} | top5 {r['top5_coords']}")
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
