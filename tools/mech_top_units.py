# -*- coding: utf-8 -*-
"""把「效应最大的那批单元」翻译成人话：它们推什么词、在什么上下文里亮。

输入 tempdata/unit_attr_L{b}.npz（tools/mech_unit_attr.py 落盘）+ 普查数据。
输出 tempdata/top_units_L{b}.md：每个单元一行（效应、平均激活、推的词、一条真实上下文）。

用法：.venv/Scripts/python.exe -X utf8 tools/mech_top_units.py 24 25
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from server import mech_pair_service as mpair
from server import neuron_census_service as ncs

SRC_SHIFT = 40
SRC_NAMES = ['fineweb-en', 'wiki-en', 'fineweb-zh', 'wiki-zh']


def main():
    block = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    topn = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    run = mpair._pick_run(36)
    z = np.load(Path('tempdata') / ('unit_attr_L%d.npz' % block))
    attr = z['attr_sum']
    man = json.loads((ncs.native_root() / '_census' / run['model_key'] / run['run']
                      / 'manifest.json').read_text(encoding='utf-8'))
    tok = ncs.tokenizer(man['model_path'])
    _cman, streams = ncs.corpus(man['corpus_fp'])
    L = np.load(ncs.native_root() / '_census' / run['model_key'] / run['run'] / ('L%d.npz' % block))
    n_tok = int(L['n'].item()) if 'n' in L.files else 19042014
    mean_act = L['s1'] / n_tok
    rt, ti = L['readout_top'], L['top_idx']

    order = np.argsort(-np.abs(attr))[:topn]
    lines = ['# L%d 效应最大的 %d 个单元' % (block, topn), '',
             '| unit | Δ(答案 logit，一阶) | 平均激活 | 推的前 5 个词 | 一条真实上下文 |',
             '|---|---|---|---|---|']
    for u in order:
        words = ' / '.join(ncs._decode(tok, [int(w)]).replace('|', '¦').strip() or '␣'
                           for w in rt[u][:5])
        g = int(ti[u][0])
        src, off = g >> SRC_SHIFT, g & ((1 << SRC_SHIFT) - 1)
        ctxt = ncs.context(tok, streams[src], off) if src < len(streams) else {'left': '', 'hit': '', 'right': ''}
        text = (ctxt['left'][-45:] + '【' + ctxt['hit'] + '】' + ctxt['right'][:30]).replace('|', '¦').replace('\n', ' ')
        lines.append('| %d | %+.2f | %.3f | %s | %s |' % (u, attr[u], mean_act[u], words, text))

    p = Path('tempdata') / ('top_units_L%d.md' % block)
    p.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines[:6]))
    print('... written', p)


if __name__ == '__main__':
    main()
