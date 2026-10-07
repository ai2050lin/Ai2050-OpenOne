# -*- coding: utf-8 -*-
"""探测 memo_append 的 BOM 状态，并把 do_append 中读 APP 的编码改为 utf-8-sig（免疫 BOM 注入）。"""
import io
import os
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC = os.path.join(ROOT, 'tests', 'deepseek', 'Phase21', 'do_append_phase21.py')
APP = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase21', 'memo_append_phase21.md')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
OUT = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase21', '_patch_enc_report.txt')

o = []


def w(s=''):
    o.append(str(s))


B = b'\xef\xbb\xbf'
raw_app = open(APP, 'rb').read()
raw_memo = open(MEMO, 'rb').read()
w('=== BOM 探测 ===')
w('memo_append  bytes=%d  bom=%s  sha8=%s' % (len(raw_app), raw_app[:3] == B, hashlib.sha256(raw_app).hexdigest()[:8]))
w('MEMO         bytes=%d  bom=%s  sha8=%s' % (len(raw_memo), raw_memo[:3] == B, hashlib.sha256(raw_memo).hexdigest()[:8]))
w('memo_append 首 40 字节 = %r' % raw_app[:40])

OLD = "io.open(APP, encoding='utf-8')"
NEW = "io.open(APP, encoding='utf-8-sig')"
raw_src = open(SRC, 'rb').read()
bom_src = raw_src[:3] == B
t = raw_src.decode('utf-8-sig')
n = t.count(OLD)
w('')
w('=== 编码补丁 ===')
w("do_append 中 io.open(APP, encoding='utf-8') 出现 %d 次" % n)
assert n == 2, '期望 2 处，实为 %d' % n
t2 = t.replace(OLD, NEW)
assert t2.count(NEW) == 2
raw_src_new = (('\ufeff' if bom_src else '') + t2).encode('utf-8')
open(SRC, 'wb').write(raw_src_new)
# 回读
rb = open(SRC, 'rb').read().decode('utf-8-sig')
w('after: OLD=%d NEW=%d  bytes %d -> %d' % (rb.count(OLD), rb.count(NEW), len(raw_src), len(raw_src_new)))
w('sha8 %s -> %s' % (hashlib.sha256(raw_src).hexdigest()[:8], hashlib.sha256(raw_src_new).hexdigest()[:8]))
io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(o) + '\n')
print('ENC PATCH OK')
