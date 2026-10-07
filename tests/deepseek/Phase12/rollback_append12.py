# -*- coding: utf-8 -*-
"""回滚 Phase 12 备忘录追加：截断回 pre-append 基线（bytes 236092 / sha8 277f49da），全字段核验。"""
import os, io, hashlib, shutil

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
BASE = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase12', 'memo_baseline_preappend_phase12.json')
BK = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase12', '_memo_with12_backup.md')

import json
B = json.load(io.open(BASE, encoding='utf-8'))
EXP_B, EXP_SHA = int(B['bytes']), B['sha256']
print('baseline expect bytes', EXP_B, 'sha', EXP_SHA[:8])

cur = open(MEMO, 'rb').read()
print('current bytes', len(cur), 'sha8', hashlib.sha256(cur).hexdigest()[:8])
open(BK, 'wb').write(cur)          # 备份“已追加”版本
print('backup written ->', BK, len(cur))

pre = cur[:EXP_B]
assert hashlib.sha256(pre).hexdigest() == EXP_SHA, '前 %d 字节不等于基线 sha' % EXP_B
open(MEMO, 'wb').write(pre)

rb = open(MEMO, 'rb').read()
t2 = rb.decode('utf-8-sig')
lines = t2.splitlines()
print('restored bytes', len(rb), 'lines', len(lines), 'sha8', hashlib.sha256(rb).hexdigest()[:8])
print('bom=%s crlf=%d bare_lf=%d' % (rb[:3] == b'\xef\xbb\xbf', rb.count(b'\r\n'), rb.count(b'\n') - rb.count(b'\r\n')))
print('phase headings', [i + 1 for i, l in enumerate(lines) if l.startswith('## Phase ')])
print('Phase12 heading present =', any(l.startswith('## Phase 12:') for l in lines))
print('prefix_exact =', hashlib.sha256(rb).hexdigest() == EXP_SHA)
