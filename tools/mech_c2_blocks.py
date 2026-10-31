# -*- coding: utf-8 -*-
"""C2：把 C1 的因果消融扩到多个层，回答「关键组是跨层稳定，还是每层各一套」。

L24 已单独跑过（ablate_L24.json）。本脚本一次加载模型，依次跑给定层，
每层落盘 ablate_L{b}.json，最后写 ablate_by_layer.json 索引。

用法：
  .venv/Scripts/python.exe -X utf8 tools/mech_c2_blocks.py 2,14,30,34
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server import mech_ablate_service as ma
from server import mech_pair_service as mpair

BLOCKS = [int(x) for x in (sys.argv[1].split(',') if len(sys.argv) > 1 else ['14', '30', '34'])]
NQ = int(sys.argv[2]) if len(sys.argv) > 2 else 0        # 0 = 全量 169 题

t_all = time.time()
model, tok, device, meta, model_name = mpair._wait_ready()
print('model ready %.0fs  L=%s' % (time.time() - t_all, meta['n_layers']), flush=True)

cj = json.loads(mpair.CLUST_JSON.read_text(encoding='utf-8'))
have = [r['block'] for r in cj['layers']]
print('clusters available:', have, flush=True)

for b in BLOCKS:
    if b not in have:
        print('skip L%d：unit_clusters.json 里没有' % b, flush=True)
        continue
    t0 = time.time()
    ma._run(b, 'C-Eval val', NQ, 2)
    print('L%-2d %s %s %.0fs' % (b, ma._STATE['state'], ma._STATE['error'],
                                 time.time() - t0), flush=True)
    if ma._STATE['state'] == 'error':
        print(ma._STATE['note'][-600:], flush=True)

print('all done %.0fs' % (time.time() - t_all), flush=True)
