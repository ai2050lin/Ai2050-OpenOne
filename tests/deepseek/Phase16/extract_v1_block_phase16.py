# -*- coding: utf-8 -*-
"""[证据] 把当前 MEMO 中**已追加**的 Phase 16 节原样抽出，存为 memo_append_phase16_v1_asappended.md，
供「同轮重写」留痕（对应 P15 的首轮作废日志保留惯例）。
"""
import io
import os
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
OUT = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase16', 'memo_append_phase16_v1_asappended.md')

mt = open(MEMO, 'rb').read().decode('utf-8-sig')
lines = mt.split('\r\n')
i = next(k for k, l in enumerate(lines) if l.startswith('## Phase 16'))
blk = '\r\n'.join(lines[i:]).rstrip('\r\n')
io.open(OUT, 'w', encoding='utf-8', newline='\n').write(blk)
bb = blk.encode('utf-8')
print('v1 block lines=%d bytes(lf)=%d sha8=%s' % (len(blk.split('\n')), len(bb), hashlib.sha256(bb).hexdigest()[:8]))
print('starts:', blk.split('\r\n')[0][:70])
print('ends  :', blk.split('\r\n')[-1][:70])
print('OUT ->', OUT)
