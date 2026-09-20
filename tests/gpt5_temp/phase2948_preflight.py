# -*- coding: utf-8 -*-
"""Phase 2948 preflight: linear W_ov head-gain vs 2947 D_h.

Zero-forward probe. g_h = median_w u35.(Wov_h @ xdir_w);
compare rank(g) with rank(D) from 2947 run2 (quasi-post-hoc
preview; formal thresholds will be permutation-null based).
"""
import json
import os

import numpy as np

MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                        'probe_relativity.npz')
SRC_2939 = os.path.join(BASE, 'phase2939', 'rotation_target',
                        'rotation_target.npz')
SRC_2947 = os.path.join(BASE, 'phase2947', 'head_anatomy',
                        'result.json')
NH, HD = 32, 128
NL = 36
out = []

# ---- locate safetensors shards ----
idx_path = os.path.join(MD, 'model.safetensors.index.json')
weight_map = json.load(open(idx_path, encoding='utf-8'))[
    'weight_map']
need = {}
for li in (16, 17):
    for proj in ('v_proj', 'o_proj'):
        k = 'model.layers.%d.self_attn.%s.weight' % (li, proj)
        f = weight_map[k]
        need.setdefault(f, []).append(k)
out.append('shards: %s' % {k: len(v)
                           for k, v in need.items()})

from safetensors import safe_open

tensors = {}
for f, keys in need.items():
    with safe_open(os.path.join(MD, f), framework='pt') as sf:
        for k in keys:
            tensors[k] = sf.get_tensor(k).float().numpy()
out.append('tensors loaded: %s' % {
    k: v.shape for k, v in tensors.items()})

# ---- frozen artifacts ----
z27 = np.load(SRC_2927, allow_pickle=True)
dirs_word = z27['dirs_word'].astype(np.float64)
u35 = dirs_word[NL - 1]
z39 = np.load(SRC_2939, allow_pickle=True)
Vt8 = z39['Vt8'].astype(np.float64)
coords = z39['coords'].astype(np.float64)
conds39 = [str(s) for s in z39['cond_names']]
dcks = coords[conds39.index('null0')] \
    - coords[conds39.index('func')]
S_IDX = (0, 1, 4)
dcks_S = dcks[:, list(S_IDX)]
Vt8_S = Vt8[list(S_IDX)]
xdir = dcks_S @ Vt8_S  # (57, 2560)
r47 = json.load(open(SRC_2947, encoding='utf-8'))
D17 = np.array(r47['D1_vectors']['L17']['D'])
D16 = np.array(r47['D1_vectors']['L16']['D'])
out.append('xdir %s | med|xdir| %.4f'
           % (xdir.shape, float(np.median(
               np.linalg.norm(xdir, axis=1)))))


def rankdata(x):
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a, b):
    ra = rankdata(np.asarray(a, float))
    rb = rankdata(np.asarray(b, float))
    ra -= ra.mean(); rb -= rb.mean()
    den = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return 0.0 if den < 1e-30 else float((ra * rb).sum() / den)


def head_gains(li):
    Wv = tensors['model.layers.%d.self_attn.v_proj.weight'
                 % li]          # (1024, 2560): 8 KV heads
    Wo = tensors['model.layers.%d.self_attn.o_proj.weight'
                 % li]          # (2560, 4096)
    gs = np.zeros(NH)
    for h in range(NH):
        kv = h // (NH // 8)     # GQA group: 4 q-heads/kv
        Wvh = Wv[kv * HD:(kv + 1) * HD]    # (128, 2560)
        Woh = Wo[:, h * HD:(h + 1) * HD]   # (2560, 128)
        ov = Woh @ Wvh                      # (2560, 2560)
        proj = xdir @ ov.T                  # (57, 2560)
        gs[h] = float(np.median(proj @ u35))
    return gs


def perm_p(obs, D, n=20000, seed=2904):
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(n):
        if spearman(rng.permutation(D), _g_cache) >= obs:
            cnt += 1
    return (cnt + 1) / (n + 1)


for li, D in ((17, D17), (16, D16)):
    g = head_gains(li)
    _g_cache = g
    rho = spearman(g, D)
    # perm null: shuffle D
    rng = np.random.default_rng(2904)
    null = [spearman(g, rng.permutation(D))
            for _ in range(2000)]
    p = float((np.sum(np.array(null) >= rho) + 1) / 2001)
    top_g = list(np.argsort(g)[::-1][:5])
    out.append('L%d: rho(g,D) %.4f perm-p %.4f | g range '
               '%.4f..%.4f | top5 g heads %s | top5 D heads '
               '%s | g(h19) %.4f rank %d | g(h22) %.4f rank %d'
               % (li, rho, p, float(g.min()), float(g.max()),
                  top_g, list(np.argsort(D)[::-1][:5]),
                  float(g[19]), int(np.argsort(-g).tolist().index(19)) + 1,
                  float(g[22]), int(np.argsort(-g).tolist().index(22)) + 1))
    # alignment of ov(xdir) with v3 for top heads
    v3 = Vt8[2]
    for h in (19, 22):
        Wv = tensors['model.layers.%d.self_attn.v_proj.weight'
                     % li]
        Wo = tensors['model.layers.%d.self_attn.o_proj.weight'
                     % li]
        kv = h // (NH // 8)
        ovx = (Wo[:, h * HD:(h + 1) * HD]
               @ Wv[kv * HD:(kv + 1) * HD]) @ xdir.mean(0)
        c_u35 = float(ovx @ u35
                      / max(np.linalg.norm(ovx)
                            * np.linalg.norm(u35), 1e-30))
        c_v3 = float(ovx @ v3
                     / max(np.linalg.norm(ovx)
                           * np.linalg.norm(v3), 1e-30))
        out.append('  L%d h%d: cos(ov(xdir_mean), u35) %.4f '
                   '| cos(..., v3) %.4f' % (li, h, c_u35, c_v3))

open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
     r'\phase2948_preflight_report.txt', 'w',
     encoding='utf-8').write('\n'.join(out) + '\n')
print('preflight done')
