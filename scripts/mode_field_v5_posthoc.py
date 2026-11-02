# -*- coding: utf-8 -*-
"""MF-v5 事后诊断（不进判定）：H1 的门控得分在全部位置上被两件事左右——
(1) 少数大幅度单元（Qwen 低索引单元标准差是其他单元的数倍）主导原始加权和；(2) 当前 token 身份（代码符号 vs 汉字）本身改变单元激活。
这里重新采集 32 个单元在混合文本上的激活：
  a  按单元标准差标准化后再加权求和（z 得分）；
  b  只在「共用 token」位置比较：同一 token 在本类文本的两种段落中都出现（与 MF-v1 / MF-v4 H3b 的口径相同）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from shared.dist_protocol import mode_field as F  # noqa: E402
import mf_materials_v1 as MAT  # noqa: E402
import mode_field_v1 as MF1  # noqa: E402
import mode_field_v4 as MF4  # noqa: E402
import mode_field_v5 as MF5  # noqa: E402
import xm_replicate_v1 as XM  # noqa: E402


def main(model_path='models/hf/qwen3-4b', int8=False):
    import torch
    _, key = XM.key_of(model_path, int8)
    od = MF1.out_dir(key)
    v4 = json.loads(MF4.report_path(model_path, int8).read_text(encoding='utf-8'))
    TL, off = v4['layers'], {int(k): v for k, v in v4['unit_offsets'].items()}
    U = np.load(od / 'mf_v4_units.npz')
    top, layer_of, rd = [int(x) for x in U['top']], U['layer_of'], U['rd_mean']
    sign = np.sign(rd[top])
    g5 = np.load(od / 'mf_v5_gate.npz')
    sd = g5['sd'][top]
    tok, model = XM.load(model_path, int8)
    R = XM.Runner(tok, model)
    cap = {}
    hooks = [R.blocks[l].mlp.down_proj.register_forward_pre_hook(lambda m, a, l=l: cap.__setitem__(l, a[0].detach())) for l in TL]
    C = MAT.mixed_corpus()
    ids = [tok(c['text'], add_special_tokens=False).input_ids for c in C]
    segs = [MF5.seg_labels(tok, c) for c in C]
    A = []
    for j in range(0, len(C), 16):
        chunk = list(range(j, min(j + 16, len(C))))
        R.forward([ids[i] for i in chunk])
        for b, i in enumerate(chunk):
            n = len(ids[i])
            cols = [cap[int(layer_of[u])][b, :n, u - off[int(layer_of[u])]].float().cpu().numpy() for u in top]
            A.append(np.stack(cols, 1))
    for h in hooks:
        h.remove()
    A = np.concatenate(A)
    seg = np.concatenate(segs)
    kind = np.concatenate([[c['kind']] * len(ids[i]) for i, c in enumerate(C)])
    tk = np.concatenate([np.array(x) for x in ids])
    raw = A @ sign
    mu = A.mean(0)
    z = ((A - mu) / np.maximum(sd, 1e-6)) @ sign
    out = {}
    for k_, pos_seg, neg_seg in (('md', 'code', 'prose'), ('cmt', 'code', 'comment')):
        m = kind == k_
        shared = set(tk[m & (seg == pos_seg)].tolist()) & set(tk[m & (seg == neg_seg)].tolist())
        sh = m & np.isin(tk, list(shared))
        res = {}
        for name, G in (('raw', raw), ('z', z)):
            res[name] = {'all_positions': F.segment_auc(G[m], seg[m], pos_seg, neg_seg),
                         'shared_token_positions': F.segment_auc(G[sh], seg[sh], pos_seg, neg_seg)}
        res['n_shared_positions'] = {pos_seg: int((sh & (seg == pos_seg)).sum()), neg_seg: int((sh & (seg == neg_seg)).sum())}
        res['n_shared_tokens'] = len(shared)
        out[k_] = res
        print(k_, json.dumps(res), flush=True)
    out['note'] = '事后诊断，不进 MF-v5 判定'
    MF5.save(model_path, int8, {'posthoc': out})


if __name__ == '__main__':
    main(*(sys.argv[1:2] or []), int8='--int8' in sys.argv)
