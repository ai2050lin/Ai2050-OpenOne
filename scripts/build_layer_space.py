# -*- coding: utf-8 -*-
"""单层神经元的两种 3D 视图数据（layer_space.v1），供「图谱 · 神经元」页显示。

属性空间：三个轴都是实测量 —— x = 因果作用（L30-v2 fp32 均值消融一阶估计 ΔL̂，symlog），
          y = 激活密度（强激活率，log10），z = 直接读出占比 |D̂| / (|D̂| + |Î|)。
读出语义空间：每个单元的读出向量 r_j = W_U(γ ⊙ W_down[:, j])（151,936 维 logit 方向）单位化后做 PCA 取前 3 维。
          不显式构造 r_j：令 M = diag(γ) W_Uᵀ W_U diag(γ)，B = M^{1/2} W_down，则 B 的列内积等于 r_j 的内积，
          在 2560 维里做 PCA 与在 logit 空间里做等价。每个主轴两端给出它在 logit 空间里最推高的 token，作为轴的语义标注。
两种布局都只是展示：距离不等于模型的计算关系；读出语义空间是降维投影，会失真。

用法：python scripts/build_layer_space.py --layer 30
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import run_layer_attribution as V1  # noqa: E402


def symlog(x, s):
    return np.sign(x) * np.log10(1 + np.abs(x) / s)


def to_unit(v):
    lo, hi = float(np.min(v)), float(np.max(v))
    return (2 * (v - lo) / (hi - lo + 1e-12) - 1), lo, hi


def main() -> int:
    import torch
    from safetensors import safe_open
    from tokenizers import Tokenizer
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--layer', type=int, default=30)
    a = ap.parse_args()
    L = a.layer
    nd = ROOT / 'research' / 'neurons'
    pano = json.loads((nd / f'L{L}-v1-panorama.json').read_text(encoding='utf-8'))
    att = json.loads((nd / f'L{L}-v2-attribution.json').read_text(encoding='utf-8'))
    trig_p = nd / f'L{L}-v3-trigger.json'
    trig = {r['unit']: r for r in json.loads(trig_p.read_text(encoding='utf-8'))['rows']} if trig_p.is_file() else {}
    z = np.load(V1.RUN / f'L{L}.npz')
    tok = Tokenizer.from_file(str(V1.MODEL / 'tokenizer.json'))
    F = len(pano['units'])

    # ---- 属性空间 ----
    tot = np.array([u['total'] for u in att['units']])
    D = np.array([u['direct'] for u in att['units']]); I = tot - D
    share = np.abs(D) / (np.abs(D) + np.abs(I) + 1e-12)
    strong = np.array([u['strong_rate'] for u in pano['units']])
    sx = symlog(tot, 1e-6)
    sy = np.log10(np.maximum(strong, 1e-5))
    px, xlo, xhi = to_unit(sx); py, ylo, yhi = to_unit(sy); pz, zlo, zhi = to_unit(share)
    rank = np.empty(F, dtype=int); rank[np.argsort(-np.abs(tot))] = np.arange(1, F + 1)

    def ticks_symlog(lo, hi):
        out = []
        for v in (-1e-3, -1e-4, -1e-5, 0.0, 1e-5, 1e-4, 1e-3):
            t = float(symlog(np.array([v]), 1e-6)[0])
            if lo <= t <= hi:
                out.append({'pos': 2 * (t - lo) / (hi - lo) - 1, 'label': '0' if v == 0 else f'{v:+.0e}'})
        return out

    def ticks_log(lo, hi):
        return [{'pos': 2 * (np.log10(v) - lo) / (hi - lo) - 1, 'label': f'{v:g}'} for v in (0.001, 0.01, 0.03, 0.1, 0.3)
                if lo <= np.log10(v) <= hi]
    prop_axes = [
        {'axis': 'x', 'name': '因果作用 ΔL̂（均值消融，nats/token，symlog）', 'source': f'L{L}-v2', 'ticks': ticks_symlog(xlo, xhi),
         'neg': '换成均值后损失下降', 'pos': '换成均值后损失上升'},
        {'axis': 'y', 'name': '激活密度（强激活率 P(A > 0.316)，log）', 'source': f'L{L}-v1 普查', 'ticks': ticks_log(ylo, yhi)},
        {'axis': 'z', 'name': '直接读出占比 |D̂| / (|D̂| + |Î|)', 'source': f'L{L}-v2',
         'ticks': [{'pos': 2 * (v - zlo) / (zhi - zlo) - 1, 'label': f'{v:g}'} for v in (0.0, 0.25, 0.5, 0.75, 1.0) if zlo <= v <= zhi]},
    ]

    # ---- 读出语义空间 ----
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    idx = json.loads((V1.MODEL / 'model.safetensors.index.json').read_text(encoding='utf-8'))['weight_map']

    def tensor(name):
        with safe_open(str(V1.MODEL / idx[name]), 'pt') as fh:
            return fh.get_tensor(name)
    WU = tensor('model.embed_tokens.weight').to(dev, torch.float32)       # 共享词嵌入 = lm_head
    gamma = tensor('model.norm.weight').to(dev, torch.float32)
    Wd = tensor(f'model.layers.{L}.mlp.down_proj.weight').to(dev, torch.float32)   # (d, F)
    G = WU.T @ WU                                                        # (d, d)
    M = gamma[:, None] * G * gamma[None, :]
    evals, Q = torch.linalg.eigh(M.double())
    evals = evals.clamp_min(0)
    Mh = (Q * evals.sqrt()) @ Q.T                                        # M^{1/2}
    keep = evals > evals.max() * 1e-10
    Mih = (Q[:, keep] * (1 / evals[keep].sqrt())) @ Q[:, keep].T          # M^{-1/2}（伪逆）
    B = (Mh.float() @ Wd)                                                # (d, F)：列内积 = r_j 的内积
    B = B / B.norm(dim=0, keepdim=True)
    Bc = B - B.mean(1, keepdim=True)
    U_, S_, _ = torch.linalg.svd(Bc, full_matrices=False)
    P = U_[:, :3]                                                        # 主方向（B 空间）
    coords = (P.T @ Bc).T.cpu().numpy()                                  # (F, 3)
    var = (S_ ** 2 / (S_ ** 2).sum()).cpu().numpy()
    sem = coords / (np.abs(coords).max(0, keepdims=True) + 1e-12)
    sem_axes = []
    for k in range(3):
        dlogit = WU @ (gamma * (Mih.float() @ P[:, k]))                  # 主方向在 logit 空间的对应方向
        top = torch.topk(dlogit, 8).indices.tolist(); bot = torch.topk(-dlogit, 8).indices.tolist()
        ext_p = np.argsort(-coords[:, k])[:4].tolist(); ext_n = np.argsort(coords[:, k])[:4].tolist()
        sem_axes.append({'axis': 'xyz'[k], 'name': f'读出 PC{k + 1}（解释方差 {100 * var[k]:.1f}%）',
                         'pos_tokens': [tok.decode([t]) for t in top], 'neg_tokens': [tok.decode([t]) for t in bot],
                         'pos_units': ext_p, 'neg_units': ext_n})

    units = []
    for j, u in enumerate(pano['units']):
        t = trig.get(j, {})
        units.append({'u': j, 't': u['type'][:2], 'p': [round(float(px[j]), 4), round(float(py[j]), 4), round(float(pz[j]), 4)],
                      's': [round(float(v), 4) for v in sem[j]], 'tot': float(tot[j]), 'rk': int(rank[j]), 'sr': float(strong[j]),
                      'ds': float(share[j]), 'ro': u['promotes'][:3], 'rc': u['readout_category'],
                      'mt': u['pos_mode_token'] if u['type'].startswith('T1') else (u['pos_context_token'] if u['type'].startswith('T2') else None),
                      'ok': bool(t.get('sig_p_R')), 'dR': t.get('d_R')})
    out = {'schema': 'layer_space.v1', 'model_key': att['model_key'], 'layer': L, 'n_units': F,
           'sources': {'panorama': f'L{L}-v1', 'attribution': f'L{L}-v2', 'trigger': f'L{L}-v3' if trig else None},
           'property_axes': prop_axes, 'semantic_axes': sem_axes, 'semantic_var_ratio_top10': [float(v) for v in var[:10]],
           'types': ['T1', 'T2', 'T3', 'T4', 'T5', 'T6'],
           'type_names': {'T1': '单 token 检测器', 'T2': '上下文 token 检测器', 'T3': '稀疏强激活', 'T4': '稠密', 'T5': '负极主导', 'T6': '其他'},
           'boundary': '布局只是展示：属性空间的轴是实测量，读出语义空间是降维投影（会失真）；点与点的距离不代表模型内部的计算关系。',
           'units': units}
    path = nd / f'L{L}-space.json'
    path.write_text(json.dumps(out, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    print(path, f'{path.stat().st_size / 2**20:.1f} MB', 'PC 解释方差', [round(float(v), 4) for v in var[:5]])
    for ax in sem_axes:
        print(ax['name'], '+', ax['pos_tokens'][:6], '−', ax['neg_tokens'][:6])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
