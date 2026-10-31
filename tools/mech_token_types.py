# -*- coding: utf-8 -*-
"""量化「这个簇在什么 token 上最亮」。

对每个单元取 top-32 命中位置 → 从语料 token 流取真实 token → 分类：
  符号/标点（不含任何字母、汉字、数字） / 数字 / 拉丁词 / 汉字 / 其他
按 A2 的簇标签聚合 → 每簇的「峰值 token 类型构成」，与全层基线比。

这一层检验用来判定：L24 那个最致因的簇（C19）是不是真的偏「排版/标点」，
还是所有簇都这样（那就等于没区别）。

产出 tempdata/cluster_token_types.json
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from server import mech_pair_service as mpair
from server import neuron_census_service as ncs

SRC_SHIFT = 40
OUT = Path('tempdata/cluster_token_types.json')
BLOCKS = [2, 14, 24, 30, 34]
PUNCT = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \n\t\r')
CJK = re.compile(r'[\u4e00-\u9fff]')
LAT = re.compile(r'[A-Za-z]')
NUM = re.compile(r'\d')


def classify(s: str) -> str:
    t = s.strip()
    if not t:
        return 'space'
    if CJK.search(t):
        return 'han'
    if NUM.search(t) and not LAT.search(t):
        return 'num'
    if LAT.search(t):
        return 'latin'
    if all(ch in PUNCT for ch in t):
        return 'punct'
    return 'other'


def main():
    run = mpair._pick_run(36)
    man = json.loads((ncs.native_root() / '_census' / run['model_key'] / run['run']
                      / 'manifest.json').read_text(encoding='utf-8'))
    tok = ncs.tokenizer(man['model_path'])
    _cman, streams = ncs.corpus(man['corpus_fp'])
    cj = json.loads(mpair.CLUST_JSON.read_text(encoding='utf-8'))
    cz = np.load(mpair.CLUST_NPZ)
    labels = ['punct', 'num', 'latin', 'han', 'other', 'space']

    out = {'blocks': {}}
    for b in BLOCKS:
        rec = next((r for r in cj['layers'] if r['block'] == b), None)
        if rec is None:
            continue
        lab = cz['lab_%d' % b].astype(np.int64)
        ti = np.load(ncs.native_root() / '_census' / run['model_key'] / run['run']
                     / ('L%d.npz' % b))['top_idx']
        n_u, k = ti.shape
        src = (ti >> SRC_SHIFT).astype(np.int64)
        off = (ti & ((1 << SRC_SHIFT) - 1)).astype(np.int64)
        toks = np.zeros((n_u, k), dtype=np.int64)
        for s in range(len(streams)):
            m = src == s
            if m.any():
                toks[m] = streams[s][off[m]]
        flat = toks.ravel()
        uniq, inv = np.unique(flat, return_inverse=True)
        cls = np.array([classify(ncs._decode(tok, [int(u)])) for u in uniq])
        code = {c: i for i, c in enumerate(labels)}
        codes = np.array([code.get(c, code['other']) for c in cls], dtype=np.int8)[inv].reshape(n_u, k)

        layer_cnt = np.bincount(codes.ravel(), minlength=len(labels)).astype(float)
        layer_share = layer_cnt / layer_cnt.sum()
        rows = []
        for c in range(int(rec['k'])):
            m = (lab == c)
            sub = codes[m].ravel()
            cnt = np.bincount(sub, minlength=len(labels)).astype(float)
            sh = cnt / max(cnt.sum(), 1)
            rows.append({'cluster': c, 'size': int(m.sum()),
                         **{labels[i]: round(float(sh[i]), 4) for i in range(len(labels))},
                         'punct_lift': round(float(sh[0] / max(layer_share[0], 1e-9)), 2)})
        out['blocks'][str(b)] = {
            'layer_share': {labels[i]: round(float(layer_share[i]), 4) for i in range(len(labels))},
            'clusters': rows}
        top = sorted(rows, key=lambda r: -r['punct'])[:1][0]
        print('L%-2d 全层 punct %.3f | 最偏标点的簇 C%d punct %.3f (lift %.2f) | 最偏汉字 C%d %.3f'
              % (b, layer_share[0], top['cluster'], top['punct'], top['punct_lift'],
                 sorted(rows, key=lambda r: -r['han'])[0]['cluster'],
                 sorted(rows, key=lambda r: -r['han'])[0]['han']), flush=True)

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('written', OUT, flush=True)


if __name__ == '__main__':
    main()
