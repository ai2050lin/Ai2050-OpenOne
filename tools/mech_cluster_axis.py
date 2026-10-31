# -*- coding: utf-8 -*-
"""共现聚类的「轴」是什么？——用互信息检验。

假设：24 个簇主要按「字符/语种」分开（拉丁 vs 汉字、punct vs 正文），而不是按能力。
做法：给每个单元打两个外在标签（都不来自聚类本身）：
  script：它 top-32 峰值里汉字 vs 拉丁谁多（han / latin / other）
  src   ：峰值落在英文语料还是中文语料（en / zh）
再算 adjusted_mutual_info_score(簇标签, 外在标签)，以及打散标签的零基线。

AMI 越接近 1，说明「簇 ≈ 这个外在标签」，聚类的轴就是这个标签。

产出 tempdata/cluster_axis.json
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score as ami

from server import mech_pair_service as mpair
from server import neuron_census_service as ncs

SRC_SHIFT = 40
OUT = Path('tempdata/cluster_axis.json')
BLOCKS = [2, 14, 24, 30, 34]
CJK = re.compile(r'[\u4e00-\u9fff]')
LAT = re.compile(r'[A-Za-z]')


def main():
    run = mpair._pick_run(36)
    man = json.loads((ncs.native_root() / '_census' / run['model_key'] / run['run']
                      / 'manifest.json').read_text(encoding='utf-8'))
    tok = ncs.tokenizer(man['model_path'])
    _cman, streams = ncs.corpus(man['corpus_fp'])
    cj = json.loads(mpair.CLUST_JSON.read_text(encoding='utf-8'))
    cz = np.load(mpair.CLUST_NPZ)
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
        uniq, inv = np.unique(toks.ravel(), return_inverse=True)
        is_han = np.array([bool(CJK.search(ncs._decode(tok, [int(u)]))) for u in uniq])
        is_lat = np.array([bool(LAT.search(ncs._decode(tok, [int(u)]))) for u in uniq])
        h = is_han[inv].reshape(n_u, k).mean(axis=1)
        l = is_lat[inv].reshape(n_u, k).mean(axis=1)
        script = np.where(h > 0.5, 'han', np.where(l > 0.5, 'latin', 'other'))
        zh_src = ((src == 2) | (src == 3)).mean(axis=1)
        srclab = np.where(zh_src > 0.5, 'zh', np.where(zh_src < 0.5, 'en', 'mix'))
        rng = np.random.default_rng(20261031)
        row = {'block': b, 'k': int(rec['k']),
               'ami_script': round(float(ami(lab, script)), 4),
               'ami_source_lang': round(float(ami(lab, srclab)), 4),
               'ami_null': round(float(ami(lab, rng.permutation(script))), 4),
               'frac_units_han': round(float((script == 'han').mean()), 3),
               'frac_units_zh_src': round(float((srclab == 'zh').mean()), 3)}
        # 每个簇内部的「纯汉字 / 纯英文」占比，用来看簇是不是语言纯的
        pure = []
        for c in range(int(rec['k'])):
            m = lab == c
            if m.sum() < 5:
                continue
            pure.append(float((script[m] == 'han').mean()))
        row['cluster_han_share_min'] = round(float(np.min(pure)), 3)
        row['cluster_han_share_max'] = round(float(np.max(pure)), 3)
        out['blocks'][str(b)] = row
        print('L%-2d AMI(簇,字符)=%.3f  AMI(簇,语种来源)=%.3f  零基线 %.4f | 簇内纯汉字占比 %.2f..%.2f'
              % (b, row['ami_script'], row['ami_source_lang'], row['ami_null'],
                 row['cluster_han_share_min'], row['cluster_han_share_max']), flush=True)

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('written', OUT, flush=True)


if __name__ == '__main__':
    main()
