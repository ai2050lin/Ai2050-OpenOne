# -*- coding: utf-8 -*-
"""H14-pre 承载核的权重几何：core3 是不是近重复单元。

H12 定位到 [767, 1540, 35] 三个单元就能打坏 tailN 能打坏的同一批 7 道题。最省事的解释是
「这三个是彼此的近重复，删掉任一等于没删，删掉三个才等于删掉一个功能」——如果是这样，
「承载核」其实是「一个功能的三个副本」，编码破解的对象就只剩一个单元。

这一轮只读权重、不跑前向（CPU 侧，safetensors 单分片按需取三张矩阵），4 个问题：

  W1 三个单元之间是不是近重复      cos(write_i, write_j) 是否 ≥0.9
  W2 全层最近邻是谁                每个 core3 单元的 write 方向在 9727 个里的 top-5 |cos|
  W3 读取侧是否也重合              gate 行 / up 行之间的 cos（SwiGLU 里读侧是 gate/up 两行）
  W4 范数是否异常                  三张矩阵行/列范数的全层百分位（**不作判据**，仅记录）

先验（要与实测对照）：若 W1 的 cos 很低（<0.3）且 W2 的最近邻不在 tailN 内，
则「三个副本」这个解释被否掉，承载核是**三个分工不同的单元**，破解要分别做。

口径纪律（写死，防误读）
------------------------
· SwiGLU 有正标度规范自由度：gate 行 ×a、down 列 ÷a 张量不变 ⇒ **范数不是内禀量**（R-MLP-13）。
  所以 W4 只作参考，判据不用范数，只用对正标度不变的 cos。
· 同层单元共享同一坐标系，cos 可用；**不跨层比**（R-MET-05）。
· 权重几何看不见「输入依赖」：cos 只说明两个单元写同一个方向，不说明它们被同一批 token 激活。
  激活侧的判据必须等 H13 收线后再跑（需要前向）。

用法
----
  .venv/Scripts/python.exe -X utf8 tools/mech_core_wiring.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path('D:/develop/model/Qwen/Qwen3-4B')
B = 34
OUT = Path('tempdata') / f'core_wiring_L{B}.json'
CARRIER = Path('tempdata') / f'carrier_L{B}.json'


def load_shard(name: str, keys) -> dict:
    """按需取张量（bf16 → fp32），numpy 不支持 bf16，所以走 torch 再转 numpy。"""
    import torch
    from safetensors import safe_open
    out = {}
    with safe_open(str(MODEL_DIR / name), framework='pt', device='cpu') as f:
        for k in keys:
            if k in f.keys():
                out[k] = f.get_tensor(k).float().numpy()
    return out


def cos_rows(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return M / n


def main() -> int:
    idx = json.loads((MODEL_DIR / 'model.safetensors.index.json').read_text(encoding='utf-8'))
    keys = [f'model.layers.{B}.mlp.{p}.weight' for p in ('gate_proj', 'up_proj', 'down_proj')]
    shards = sorted({idx['weight_map'][k] for k in keys})
    print('分片: %s' % shards)
    W = {}
    for s in shards:
        W.update(load_shard(s, [k for k in keys if idx['weight_map'][k] == s]))
    gate = W[keys[0]].astype(np.float64)      # [d_ff, d_model]
    up = W[keys[1]].astype(np.float64)        # [d_ff, d_model]
    down = W[keys[2]].astype(np.float64)      # [d_model, d_ff]
    d_ff = gate.shape[0]
    print('L%d gate %s up %s down %s' % (B, gate.shape, up.shape, down.shape))

    car = json.loads(CARRIER.read_text(encoding='utf-8'))
    core3 = [int(u) for u in car['final_units']]
    core5 = [83, 149, 383, 583, 2423]
    tailN = sorted(int(u) for u in car['start_set'])
    others = [u for u in tailN if u not in core3]
    print('core3 %s | tailN 其余 %d 个' % (core3, len(others)))

    # ---- 写方向 = down 的第 i 列；读方向 = gate/up 的第 i 行 ----
    wdir = cos_rows(down.T.copy())            # [d_ff, d_model]
    gdir = cos_rows(gate.copy())
    udir = cos_rows(up.copy())

    def pairwise(dirs, ids):
        out = {}
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                out['%d-%d' % (a, b)] = round(float(dirs[a] @ dirs[b]), 4)
        return out

    pw = {'write': pairwise(wdir, core3), 'gate': pairwise(gdir, core3), 'up': pairwise(udir, core3)}
    print('\nW1 三单元两两 cos: %s' % json.dumps(pw, ensure_ascii=False))

    # ---- W2 全层最近邻 ----
    nn = {}
    for u in core3:
        sim = wdir @ wdir[u]
        sim[u] = -9
        top = np.argsort(-sim)[:5]
        nn[u] = [{'id': int(t), 'cos': round(float(sim[t]), 4),
                  'in_tailN': int(t) in tailN, 'in_core3': int(t) in core3} for t in top]
    print('\nW2 写方向最近邻:')
    for u, lst in nn.items():
        print('  %-5d %s' % (u, ', '.join('%d(%.3f%s)' % (x['id'], x['cos'],
                                                          '*' if x['in_tailN'] else '')
                                          for x in lst)))

    # ---- W3 对照组：tailN 其余单元的最近邻 cos（看 core3 的最近邻强度是否异常） ----
    def nn_max(u):
        s = wdir @ wdir[u]
        s[u] = -9
        return float(s.max())

    base_nn = [nn_max(u) for u in others]
    core_nn = [nn_max(u) for u in core3]
    w3 = {'core3_nn_max_cos': [round(x, 4) for x in core_nn],
          'tailN_rest_nn_max_cos_mean': round(float(np.mean(base_nn)), 4),
          'tailN_rest_nn_max_cos_max': round(float(np.max(base_nn)), 4),
          'all_units_nn_max_cos_mean': round(float(np.mean([nn_max(u) for u in range(0, d_ff, 97)])), 4)}

    # ---- W4 范数百分位（仅参考，不作判据） ----
    pct = {}
    for nm, M, ids in (('gate_row', gate, core3), ('up_row', up, core3),
                       ('down_col', down.T.copy(), core3),
                       ('gate_row', gate, others), ('up_row', up, others),
                       ('down_col', down.T.copy(), others)):
        v = np.linalg.norm(M, axis=1)
        o = np.argsort(-v)
        rank = {int(u): int(list(o).index(u)) + 1 for u in ids}
        for u in ids:
            pct.setdefault('%s:%d' % (nm, u), rank[u])
    w4 = {'core3': {k: v for k, v in pct.items() if int(k.split(':')[1]) in core3},
          'tailN_other_mean_rank': {nm: round(float(np.mean([pct['%s:%d' % (nm, u)] for u in others])), 1)
                                    for nm in ('gate_row', 'up_row', 'down_col')}}

    # ---- 判据 ----
    max_write = max(pw['write'].values())
    max_gate = max(pw['gate'].values())
    max_up = max(pw['up'].values())
    v = {
        'W1_near_duplicate': bool(max_write >= 0.9),
        'W2_neighbor_in_tailN': bool(any(x['in_tailN'] for lst in nn.values() for x in lst[:3])),
        'W3_read_overlap': bool(max_gate >= 0.9 or max_up >= 0.9),
        'max_write_cos': round(max_write, 4),
        'max_gate_cos': round(max_gate, 4),
        'max_up_cos': round(max_up, 4),
    }

    if v['W1_near_duplicate']:
        reading = ('三个单元互为近重复（max cos %.3f）⇒「承载核」其实是一个功能的三个副本，'
                   '编码破解只需解一个单元。' % max_write)
    elif v['W2_neighbor_in_tailN']:
        reading = ('三个单元不是彼此的副本（max cos %.3f），但各自的最近邻落在 tailN 内 ⇒ '
                   '池内存在一小簇几何相近单元，承载核可能是这一簇的代表。' % max_write)
    else:
        reading = ('三个单元既不是彼此的副本（write cos %.3f / gate %.3f / up %.3f），'
                   '最近邻也不在 tailN 内 ⇒ 三条几何独立的写通道，不是「一个功能的三个副本」。'
                   '但「三路是否都必须」本轮没测（H12 的二分到 3 个就停），'
                   '所以只能说「三个一起删会打坏那 7 道」，不能说「每个都必要」；'
                   '单点/成对判别见 H15 预注册。'
                   % (max_write, max_gate, max_up))

    # W4 不进判据：范数在 SwiGLU 下有正标度规范自由度（R-MLP-13），
    # 三个单元的范数排名从 7 到 9709 都出现，本身就是「范数不可作判据」的直接演示。
    norm_spread = sorted(w4['core3'].items(), key=lambda kv: kv[1])

    out = {'schema': 'mech_core_wiring.v1', 'block': B, 'd_ff': d_ff,
           'core3': core3, 'core5': core5, 'tailN_rest': others,
           'pairwise_cos': pw, 'nearest_neighbors': nn, 'nn_strength': w3,
           'norm_rank_descriptive_only': w4,
           'norm_rank_spread': {'min': norm_spread[0], 'max': norm_spread[-1],
                               'note': '三个功能同一批题的单元，范数排名横跨 7–9709，'
                                       '是规范自由度的直接演示（R-MLP-13），不作判据'},
           'verdict': v, 'reading': reading,
           'note': ('只读权重、不跑前向；cos 对正标度不变，是 SwiGLU 下唯一可用的几何量；'
                    '范数受规范自由度污染，仅列出不作判据（R-MLP-13）。')}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8', newline='')

    print('\nW3 最近邻强度: core3 %s vs tailN 其余 均值 %.3f / 最大 %.3f'
          % (w3['core3_nn_max_cos'], w3['tailN_rest_nn_max_cos_mean'], w3['tailN_rest_nn_max_cos_max']))
    print('W4 范数排名（不作判据）: %s' % json.dumps(w4['core3'], ensure_ascii=False))
    print('   横跨 %d–%d / %d ⇒ 规范自由度的直接演示' % (norm_spread[0][1], norm_spread[-1][1], d_ff))
    print('\n判据 %s' % json.dumps(v, ensure_ascii=False))
    print('读法：%s' % reading)
    print('written %s' % OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
