# -*- coding: utf-8 -*-
"""C19 深挖：L24 那个「6.7% 单元扛 37% 整层效应」的组，到底在算什么。

只用磁盘上的普查数据 + tokenizer，不占 GPU（与后台 C2 并行）。

产出 tempdata/c19_dive.json：
  · 每个组的规模、激活均值/标准差、语料来源分布（en/zh）、单元编号分布
  · 每组 top-1 promoted token 的一致性（众数占比 vs 全层基线）
  · C19 的高激活单元的 promoted 词表与真实上下文片段
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from server import mech_pair_service as mpair
from server import neuron_census_service as ncs

ME = Path('data/mech')
OUT = Path('tempdata/c19_dive.json')
SRC_SHIFT = 40
SRC_NAMES = ['fineweb-en', 'wiki-en', 'fineweb-zh', 'wiki-zh']


def layer_path(run, b):
    return ncs.native_root() / '_census' / run['model_key'] / run['run'] / ('L%d.npz' % b)


def main():
    run = mpair._pick_run(36)
    man = json.loads((layer_path(run, 0).parent / 'manifest.json').read_text(encoding='utf-8'))
    fp = man['corpus_fp']
    tok = ncs.tokenizer(man['model_path'])
    _cman, streams = ncs.corpus(fp)
    n_tok = int(man['n_tokens_counted'])

    cj = json.loads(mpair.CLUST_JSON.read_text(encoding='utf-8'))
    cz = np.load(mpair.CLUST_NPZ)

    res = {'corpus': fp, 'n_tokens': n_tok, 'layers': {}}

    for rec in cj['layers']:
        b, K = rec['block'], int(rec['k'])
        lab = cz['lab_%d' % b].astype(np.int64)
        z = np.load(layer_path(run, b))
        s1, s2 = z['s1'], z['s2']
        mean = s1 / n_tok
        std = np.sqrt(np.maximum(s2 / n_tok - mean ** 2, 0.0))
        rt = z['readout_top']
        ti = z['top_idx']

        # 全层基线：top-1 promoted token 的众数占比
        _, cnt = np.unique(rt[:, 0], return_counts=True)
        base_share = float(cnt.max() / cnt.sum())
        # 全层来源分布
        src_all = np.bincount((ti.ravel() >> SRC_SHIFT).astype(np.int64), minlength=4)[:4]
        src_all = src_all / max(src_all.sum(), 1)

        rows = []
        for c in range(K):
            m = lab == c
            idx = np.flatnonzero(m)
            # 组内 top-1 词一致性
            _, cc = np.unique(rt[idx, 0], return_counts=True)
            share = float(cc.max() / max(cc.sum(), 1))
            # 来源分布
            ss = np.bincount((ti[idx].ravel() >> SRC_SHIFT).astype(np.int64), minlength=4)[:4]
            ss = ss / max(ss.sum(), 1)
            rows.append({
                'cluster': c, 'size': int(idx.size),
                'act_mean': round(float(mean[idx].mean()), 4),
                'act_std': round(float(std[idx].mean()), 4),
                'readout_top1_share': round(share, 4),
                'readout_top1_lift': round(share / base_share, 2) if base_share else None,
                'src': {SRC_NAMES[i]: round(float(ss[i]), 3) for i in range(4)},
                'src_zh': round(float(ss[2] + ss[3]), 3),
                'unit_q': [int(np.percentile(idx, q)) for q in (0, 5, 50, 95, 100)],
                'unit_span_frac': round(float((idx.max() - idx.min() + 1) / lab.size), 3),
                'unit_fill': round(float(idx.size / max(idx.max() - idx.min() + 1, 1)), 3),
            })
        res['layers'][str(b)] = {
            'k': K, 'base_readout_share': round(base_share, 4),
            'src_all': {SRC_NAMES[i]: round(float(src_all[i]), 3) for i in range(4)},
            'clusters': rows}
        print('L%-2d base_share %.4f  src_zh %.2f' % (b, base_share, src_all[2] + src_all[3]), flush=True)

    # ---------------- C19 取证：promoted 词表 + 真实上下文
    b = 24
    lab = cz['lab_%d' % b].astype(np.int64)
    z = np.load(layer_path(run, b))
    s1, rt, ti, tv = z['s1'], z['readout_top'], z['top_idx'], z['top_val']
    n_tok = int(np.load(layer_path(run, b))['n'].item()) if 'n' in z.files else n_tok
    m = lab == 19
    idx = np.flatnonzero(m)

    words = {}
    for u in idx:
        for w in rt[u]:
            s = ncs._decode(tok, [int(w)])
            words[s] = words.get(s, 0) + 1
    top_words = sorted(words.items(), key=lambda kv: -kv[1])[:30]
    # 对照：随机 655 个单元
    rng = np.random.default_rng(7)
    rw = {}
    for u in rng.choice(lab.size, size=int(idx.size), replace=False):
        for w in rt[u]:
            s = ncs._decode(tok, [int(w)])
            rw[s] = rw.get(s, 0) + 1
    rand_words = sorted(rw.items(), key=lambda kv: -kv[1])[:20]

    # 高激活单元（按平均激活）取前 12 个，各取最强 2 个位置看上下文
    mean_act = s1 / n_tok
    order = idx[np.argsort(-mean_act[idx])][:12]
    ctxs = []
    for u in order:
        for g, v in zip(ti[u][:2], tv[u][:2]):
            src, off = int(g) >> SRC_SHIFT, int(g) & ((1 << SRC_SHIFT) - 1)
            if src >= len(streams):
                continue
            ctxt = ncs.context(tok, streams[src], off)
            ctxs.append({'unit': int(u), 'act_mean': round(float(mean_act[u]), 3),
                         'val': round(float(v), 2), 'src': SRC_NAMES[src],
                         'text': (ctxt['left'][-90:] + '【' + ctxt['hit'] + '】' + ctxt['right'][:60])})

    res['c19'] = {'block': b, 'size': int(idx.size), 'n_tokens': n_tok,
                  'top_words': [[w, n] for w, n in top_words],
                  'rand_words': [[w, n] for w, n in rand_words],
                  'contexts': ctxs}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('written', OUT, OUT.stat().st_size, flush=True)


if __name__ == '__main__':
    main()
