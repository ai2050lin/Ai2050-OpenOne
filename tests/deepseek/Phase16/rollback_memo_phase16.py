# -*- coding: utf-8 -*-
"""把 MEMO 回滚到「Phase 16 追加前」的**精确字节**状态（对 PRE 基线 sha256 验证）。
依据：PRE = memo_baseline_preappend_phase16.json（sha256 冻结）。
做法：切掉已追加的 Phase 16 节（含分隔 CRLF），再解出原始尾随 CRLF 数 k，使 sha256 == PRE.sha256。
"""
import io
import os
import json
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P16T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase16')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
PRE = json.load(io.open(os.path.join(P16T, 'memo_baseline_preappend_phase16.json'), encoding='utf-8'))

mb = open(MEMO, 'rb').read()
mt = mb.decode('utf-8-sig')
lines = mt.split('\r\n')
i = next(k for k, l in enumerate(lines) if l.startswith('## Phase 16'))
T = '\r\n'.join(lines[:i]).rstrip('\r\n')          # 追加前的正文（无尾随 CRLF）
found = None
for k in (1, 2, 3, 4):
    cand = b'\xef\xbb\xbf' + (T + '\r\n' * k).encode('utf-8')
    if hashlib.sha256(cand).hexdigest() == PRE['sha256']:
        found = cand
        break
assert found is not None, '无法复现 PRE 基线 sha256 —— 中止'
open(MEMO, 'wb').write(found)
rb = open(MEMO, 'rb').read()
h = hashlib.sha256(rb).hexdigest()
print('rollback: %d B -> %d B' % (len(mb), len(rb)))
print('sha256 = %s' % h)
print('== PRE.sha256 : %s' % (h == PRE['sha256']))
print('bytes == PRE  : %s (%d)' % (len(rb) == PRE['bytes'], len(rb)))
mt2 = rb.decode('utf-8-sig')
print('contains Phase16 :', '## Phase 16' in mt2)
print('phase headings   :', len([1 for l in mt2.split('\r\n') if l.startswith('## Phase ')]))
print('bare_lf          :', rb.count(b'\n') - rb.count(b'\r\n'))
print('ROLLBACK MEMO OK')
