# -*- coding: utf-8 -*-
"""单层 MLP 单元全景分型（L30-v1 第 1 部分，预注册 research/neurons/L30-v1.prereg.json）。

只用普查结果（mlp_census.v1 的 L<b>.npz）、语料 token 流与模型权重中的 W_down，不做新的前向计算：
逐单元计算激活率、符号区占比、尾部比、两极片段的 token 集中度，按预注册顺序规则分型；
读出 token（普查已算好的 W_U(γ⊙W_down[:,j]) 前 10）按字符规则分类；对单位化 W_down 列做球面 k-means。
描述统计，不产生证据等级。

用法：python scripts/run_layer_panorama.py --layer 30 --out research/neurons/L30-v1-panorama.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.dist_protocol import census as C  # noqa: E402

NATIVE = Path(os.environ.get('AI2050_NATIVE_DIR', 'D:/ai2050_native'))
RUN = NATIVE / '_census' / '6be1034f0b55a6df-bf16' / '2ca5f0a756d36c8a-20000000'
MODEL = ROOT / 'models' / 'hf' / 'qwen3-4b'
PREREG = ROOT / 'research' / 'neurons' / 'L30-v1.prereg.json'

STRONG = 10 ** -0.5                     # 普查直方图箱边界 10^(−3 + 2.5)
TYPE_ORDER = ['T1_单token检测器', 'T2_上下文token检测器', 'T3_稀疏强激活', 'T4_稠密', 'T5_负极主导', 'T6_其他']
READ_CATS = ['中文字词', '英文整词', '英文子词片段', '数字', '标点与空白', '代码符号', '其他文字', '特殊token']
CODE_CHARS = set('{}()[];=<>/\\_#$@`|&^~*+%')


def strong_bin_index() -> int:
    """正侧第一个下边界 ≥ 10^-0.5 的箱（与 census.summarize_layer 的 strong 口径一致）。"""
    return C.N_SIDE + 1 + C.N_SIDE // 2


def token_category(raw: str, text: str, special: bool) -> str:
    if special:
        return '特殊token'
    s = text.strip()
    if not s:
        return '标点与空白'
    if any('\u4e00' <= ch <= '\u9fff' or '\u3400' <= ch <= '\u4dbf' for ch in s):
        return '中文字词'
    if s.isdigit():
        return '数字'
    if all(unicodedata.category(ch)[0] in 'PZS' for ch in s) and not any(ch in CODE_CHARS for ch in s):
        return '标点与空白'
    if any(ch in CODE_CHARS for ch in s):
        return '代码符号'
    if all(ch.isascii() and ch.isalpha() for ch in s):
        return '英文整词' if raw.startswith('Ġ') or text.startswith(' ') else '英文子词片段'
    return '其他文字'


def spherical_kmeans(X: np.ndarray, k: int, seed: int, iters: int = 50):
    rng = np.random.default_rng(seed)
    cent = X[rng.choice(len(X), k, replace=False)].copy()
    lab = None
    for _ in range(iters):
        sim = X @ cent.T
        new = sim.argmax(1)
        if lab is not None and np.array_equal(new, lab):
            break
        lab = new
        for c in range(k):
            m = X[lab == c]
            if len(m):
                v = m.sum(0)
                cent[c] = v / (np.linalg.norm(v) + 1e-12)
    sim = X @ cent.T
    return lab, cent, float(sim[np.arange(len(X)), lab].sum())


def silhouette_cosine(X: np.ndarray, lab: np.ndarray, k: int, chunk: int = 1024) -> float:
    """余弦距离的平均轮廓系数（逐块计算，不存完整距离矩阵）。"""
    n = len(X)
    sizes = np.bincount(lab, minlength=k).astype(np.float64)
    out = np.empty(n)
    onehot = np.eye(k)[lab]                                    # (n, k)
    for i in range(0, n, chunk):
        D = 1.0 - X[i:i + chunk] @ X.T                         # (b, n)
        sums = D @ onehot                                      # (b, k)
        own = lab[i:i + chunk]
        cnt = sizes[None, :].repeat(len(own), 0)
        cnt[np.arange(len(own)), own] -= 1
        mean = sums / np.maximum(cnt, 1)
        a = mean[np.arange(len(own)), own]
        mean[np.arange(len(own)), own] = np.inf
        b = mean.min(1)
        s = (b - a) / np.maximum(np.maximum(a, b), 1e-12)
        s[cnt[np.arange(len(own)), own] == 0] = 0.0
        out[i:i + chunk] = s
    return float(out.mean())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--layer', type=int, default=30)
    ap.add_argument('--out', default=str(ROOT / 'research' / 'neurons' / 'L30-v1-panorama.json'))
    a = ap.parse_args(argv)
    t0 = time.time()
    from safetensors import safe_open
    from tokenizers import Tokenizer
    pre = json.loads(PREREG.read_text(encoding='utf-8'))
    z = np.load(RUN / f'L{a.layer}.npz')
    man = json.loads((RUN / 'manifest.json').read_text(encoding='utf-8'))
    cdir = NATIVE / '_census' / 'corpus' / man['corpus_fp']
    cman = json.loads((cdir / 'manifest.json').read_text(encoding='utf-8'))
    streams = [np.load(cdir / s['file'])['tokens'] for s in cman['sources']]
    tok = Tokenizer.from_file(str(MODEL / 'tokenizer.json'))
    n_special = 151643                                          # Qwen3：≥ 此 id 为特殊 token

    F = z['s1'].shape[0]
    n = float(z['n'])
    hist = z['hist'].astype(np.float64)                         # (F, 65, 4)
    sb = strong_bin_index()
    strong = hist[:, sb:, :].sum((1, 2)) / n
    neg_strong = hist[:, :C.N_SIDE - C.N_SIDE // 2, :].sum((1, 2)) / n   # 对称：A < −10^-0.5
    pos_rate = hist[:, C.N_SIDE + 1:, :].sum((1, 2)) / n
    regime = hist.sum(1) / n                                    # (F, 4)
    mean = z['s1'] / n
    std = np.sqrt(np.maximum(z['s2'] / n - mean ** 2, 0))
    tail = z['amax'] / (std + 1e-6)

    def ids_at(gidx):
        src, off = C.split_index(gidx)
        return np.array([streams[s][o] if o >= 0 else -1 for s, o in zip(src.ravel(), off.ravel())]).reshape(gidx.shape)

    def prev_at(gidx):
        src, off = C.split_index(gidx)
        return np.array([streams[s][o - 1] if o > 0 else -1 for s, o in zip(src.ravel(), off.ravel())]).reshape(gidx.shape)

    def concentration(M):
        out = np.empty(len(M)); mode = np.empty(len(M), dtype=np.int64)
        for i, row in enumerate(M):
            row = row[row >= 0]
            c = Counter(row.tolist()).most_common(1)
            mode[i], out[i] = (c[0][0], c[0][1] / len(row)) if c else (-1, 0.0)
        return out, mode
    pos_ids = ids_at(z['top_idx']); neg_ids = ids_at(z['bot_idx']); pos_prev = prev_at(z['top_idx'])
    pos_conc, pos_mode = concentration(pos_ids)
    neg_conc, _ = concentration(neg_ids)
    ctx_conc, ctx_mode = concentration(pos_prev)

    types = []
    for j in range(F):
        if pos_conc[j] >= 0.75:
            t = TYPE_ORDER[0]
        elif ctx_conc[j] >= 0.75:
            t = TYPE_ORDER[1]
        elif strong[j] < 0.001 and tail[j] >= 20:
            t = TYPE_ORDER[2]
        elif strong[j] >= 0.10:
            t = TYPE_ORDER[3]
        elif neg_strong[j] > strong[j]:
            t = TYPE_ORDER[4]
        else:
            t = TYPE_ORDER[5]
        types.append(t)

    vocab_cat = {}

    def cat_of(i):
        i = int(i)
        if i not in vocab_cat:
            vocab_cat[i] = token_category(tok.id_to_token(i) or '', tok.decode([i]), i >= n_special)
        return vocab_cat[i]
    read_top = z['readout_top']; read_bot = z['readout_bot']
    read_cat = []
    read_dist = []
    for j in range(F):
        cs = Counter(cat_of(i) for i in read_top[j])
        read_dist.append(dict(cs))
        read_cat.append(cs.most_common(1)[0][0])

    with safe_open(str(MODEL / 'model-00002-of-00003.safetensors'), 'pt') as fh:
        Wd = fh.get_tensor(f'model.layers.{a.layer}.mlp.down_proj.weight').float().numpy()      # (2560, F)
    X = (Wd / (np.linalg.norm(Wd, axis=0, keepdims=True) + 1e-12)).T.astype(np.float32)        # (F, 2560)
    clusters = {}
    for k in (16, 32, 64):
        best = max((spherical_kmeans(X, k, s) for s in range(5)), key=lambda r: r[2])
        lab, cent, obj = best
        sil = silhouette_cosine(X, lab, k)
        members = []
        for c in range(k):
            idx = np.where(lab == c)[0]
            if not len(idx):
                continue
            near = idx[np.argsort(-(X[idx] @ cent[c]))[:5]]
            members.append({'cluster': c, 'n': int(len(idx)),
                            'types': dict(Counter(types[i] for i in idx)),
                            'readout_categories': dict(Counter(read_cat[i] for i in idx)),
                            'representatives': [{'unit': int(u), 'promotes': [tok.decode([int(t)]) for t in read_top[u][:6]]} for u in near]})
        clusters[str(k)] = {'silhouette_cosine': sil, 'objective': obj, 'labels': lab.tolist(), 'clusters': members}
        print(f'[panorama] k={k} silhouette={sil:.4f}', flush=True)

    units = []
    for j in range(F):
        units.append({'unit': j, 'type': types[j], 'strong_rate': float(strong[j]), 'pos_rate': float(pos_rate[j]),
                      'neg_strong_rate': float(neg_strong[j]), 'regime_share': [float(x) for x in regime[j]],
                      'tail_ratio': float(tail[j]), 'amax': float(z['amax'][j]), 'amin': float(z['amin'][j]),
                      'mean': float(mean[j]), 'std': float(std[j]),
                      'pos_token_concentration': float(pos_conc[j]), 'pos_mode_token': tok.decode([int(pos_mode[j])]) if pos_mode[j] >= 0 else None,
                      'neg_token_concentration': float(neg_conc[j]),
                      'pos_context_concentration': float(ctx_conc[j]), 'pos_context_token': tok.decode([int(ctx_mode[j])]) if ctx_mode[j] >= 0 else None,
                      'readout_category': read_cat[j], 'readout_category_dist': read_dist[j],
                      'promotes': [tok.decode([int(t)]) for t in read_top[j]], 'suppresses': [tok.decode([int(t)]) for t in read_bot[j]],
                      'cluster_k32': int(clusters['32']['labels'][j])})
    type_counts = Counter(types)
    summary = {
        'n_units': F, 'n_tokens': int(n),
        'type_counts': {t: int(type_counts.get(t, 0)) for t in TYPE_ORDER},
        'readout_category_counts': dict(Counter(read_cat)),
        'strong_rate_quantiles': {q: float(np.quantile(strong, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        'regime_share_mean': dict(zip(C.REGIMES, map(float, regime.mean(0)))),
        'silhouette_by_k': {k: clusters[k]['silhouette_cosine'] for k in clusters},
    }
    out = {'schema': 'layer_panorama.v1', 'prereg': 'L30-v1', 'prereg_sha256': (PREREG.with_suffix('.sha256')).read_text().split()[0],
           'model_key': man['model_key'], 'layer': a.layer, 'census_manifest': str(RUN / 'manifest.json'),
           'type_rules': pre['part1_panorama']['types'], 'summary': summary,
           'clusters': {k: {kk: vv for kk, vv in v.items() if kk != 'labels'} for k, v in clusters.items()},
           'units': units, 'seconds': round(time.time() - t0, 1),
           'boundary': '描述统计：类型与簇只描述激活形状与输出方向，不说明单元的功能或共享机制。'}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
