# -*- coding: utf-8 -*-
"""把 Phase 13 备忘录节以 UTF-8+BOM+CRLF 追加到 AGI_DEEPSEEK_MEMO.md，并自检落盘。

Phase 12 的教训（已写入技能 rdc-phase-closeout）：**自检锚点必须先对追加源文件预检**，
否则会出现「追加成功但锚点缺失」→ 不得不按字节回滚重来。本脚本先预检、再追加。
"""
import os
import io
import time
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
APP = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase13', 'memo_append_phase13.md')
OUT = os.path.join(ROOT, 'tests', 'deepseek', 'Phase13', 'verify_append_phase13.txt')

o = []
def w(s=''):
    o.append(str(s)); print(s)


ANCHORS = [
    'Phase 13: 位点间配对 bootstrap', 'N2h1-α-6',
    'CONCENTRATION_COORDINATE_DEPENDENT', 'PAIRED_TEST_INFORMATIVE',
    '0.000e+00', '10 / 17', '2 / 17', '4 / 17', '0.5475',
    '0.5745', '0.3867', '0.8316', '0.3790',
    '0.7953', '0.5961', '0.8986', '0.9730',
    '0.7505', '0.7390', '0.064481', '0.078529', '0.051705', '0.1593',
    '2.842e-14', '3.55e-15', '0.1007', '0.1094', '0.5707',
    '(t)', '(u)', 'prefix swap', 'Phase 14',
    'F14', 'F15', 'F16', 'F17', 'F19', 'F21',
    '808c4575', '7b5b73ef', '587e7880', '1d95cd24',
    '296', '5990f956', '8021f648', 'e5e1f8dd', '273579', '2858',
    '2.11 s', 'HARKing', '2000', '第一性原理', '坐标系依赖', 'argmax',
]

# ---------- 0. 预检：锚点必须已存在于追加源文件 ----------
src = io.open(APP, encoding='utf-8').read()
pre_miss = [k for k in ANCHORS if k not in src]
w('=== [0] 追加源预检 ===')
w('源文件 %s bytes=%d' % (os.path.basename(APP), len(src.encode('utf-8'))))
for k in ANCHORS:
    c = src.count(k)
    if c < 1:
        w('  !! 源文件缺少锚点: %r' % k)
w('源文件缺失锚点数 = %d' % len(pre_miss))
assert not pre_miss, '追加源文件缺少锚点（先修源文件再追加）: %s' % pre_miss
w('  ==> 预检通过（%d 个锚点全部存在于源文件）' % len(ANCHORS))

# ---------- 1. 追加 ----------
raw0 = open(MEMO, 'rb').read()
assert raw0[:3] == b'\xef\xbb\xbf', 'BOM 丢失'
txt = raw0.decode('utf-8-sig')
n0_b, n0_l = len(raw0), len(raw0.split(b'\r\n'))
body = io.open(APP, encoding='utf-8').read().strip('\r\n')
new = txt.rstrip('\r\n') + '\r\n\r\n' + body + '\r\n'
out = b'\xef\xbb\xbf' + new.replace('\r\n', '\n').replace('\n', '\r\n').encode('utf-8')
open(MEMO, 'wb').write(out)

# ---------- 2. 落盘复核（真实磁盘） ----------
rb = open(MEMO, 'rb').read()
t2 = rb.decode('utf-8-sig')
lines = t2.splitlines()
hdr = [i + 1 for i, l in enumerate(lines) if l.startswith('## Phase 13:')]
allh = [i + 1 for i, l in enumerate(lines) if l.startswith('## Phase ')]
w('')
w('=== [1] 追加结果 ===')
w('append: bytes %d -> %d (+%d) ; lines %d -> %d' % (n0_b, len(rb), len(rb) - n0_b, n0_l, len(lines)))
w('bom=%s crlf=%d bare_lf=%d' % (rb[:3] == b'\xef\xbb\xbf', rb.count(b'\r\n'),
                                  rb.count(b'\n') - rb.count(b'\r\n')))
w('sha256 = %s' % hashlib.sha256(rb).hexdigest())
w('sha8   = %s' % hashlib.sha256(rb).hexdigest()[:8])
w('Phase 13 标题行 = %s' % hdr)
w('全部 Phase 标题行 (%d) = %s' % (len(allh), allh))
w('新增行数 = %d' % (len(lines) - n0_l))
prefix = b'\xef\xbb\xbf' + txt.rstrip('\r\n').replace('\r\n', '\n').replace('\n', '\r\n').encode('utf-8')
w('前缀逐字节未变 = %s' % rb.startswith(prefix))

w('')
w('=== [2] 锚点落盘复核 ===')
miss = []
for k in ANCHORS:
    c = t2.count(k)
    w('  anchor %-34s count=%d %s' % (k, c, 'OK' if c >= 1 else '!! MISSING'))
    if c < 1:
        miss.append(k)
w('缺失锚点数 = %d' % len(miss))
w('clock %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o) + '\n')

assert not miss, '锚点缺失: %s' % miss
assert len(hdr) == 1, 'Phase 13 标题不唯一: %s' % hdr
assert len(allh) == 13, 'Phase 标题数应为 13，实为 %d' % len(allh)
assert rb[:3] == b'\xef\xbb\xbf'
assert rb.count(b'\n') - rb.count(b'\r\n') == 0, 'bare_lf 不为 0'
assert rb.startswith(prefix), '前缀被改动'
print('ALL CHECKS PASSED ->', OUT)
