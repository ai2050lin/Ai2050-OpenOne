# -*- coding: utf-8 -*-
"""把 Phase 14 备忘录节以 UTF-8+BOM+CRLF 追加到 AGI_DEEPSEEK_MEMO.md，并自检落盘。

纪律（Phase 12 教训 #12）：**自检锚点必须先对「追加源文件」预检**，
否则会出现「追加成功但锚点缺失」→ 不得不按字节回滚重来。本脚本先预检、再追加。
"""
import os
import io
import time
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
APP = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase14', 'memo_append_phase14.md')
OUT = os.path.join(ROOT, 'tests', 'deepseek', 'Phase14', 'verify_append_phase14.txt')

o = []
def w(s=''):
    o.append(str(s)); print(s)


ANCHORS = [
    '## Phase 14: 逐层累积代换 + 双坐标集中度', 'N2h1-α-7',
    'CONCENTRATION_COORDINATE_DEPENDENT', 'FAMILY_TRANSFER_BOTH', 'POS_LAST_POSITION_DOMINANT',
    '0.000e+00', '0.002722', '0.2722%', '0.000888', '0.5745', '0.6998', '0.7953', '0.7472', '0.0480',
    '0.499289', '0.465023', '25.3511', '1.0624', '0.109745', '432',
    '074fc963', '0b0276b8', 'e0336dff', '958e21e2', '9035e1f4', 'c62ae3ba',
    'F29', 'F30a', 'F30b', 'head_dim', '置换零假设', '极值型统计量', '第一性原理',
    '(v)', '(w)', '(x)', '(y)', 'Phase 15', 'HARKing', '逐位',
]
OPTIONAL = ['HARKing']          # 若正文未出现，仅提示不阻断
ANCHORS = [k for k in ANCHORS if k not in OPTIONAL] + [k for k in OPTIONAL if True]

# ---------- 0. 预检：锚点必须已存在于追加源文件 ----------
src = io.open(APP, encoding='utf-8').read()
REQ = [k for k in ANCHORS if k not in OPTIONAL]
pre_miss = [k for k in REQ if k not in src]
w('=== [0] 追加源预检 ===')
w('源文件 %s bytes=%d' % (os.path.basename(APP), len(src.encode('utf-8'))))
for k in ANCHORS:
    c = src.count(k)
    flag = 'OK' if c >= 1 else ('(optional)' if k in OPTIONAL else '!! MISSING')
    w('  %-40s count=%d %s' % (k, c, flag))
w('源文件缺失（必选）锚点数 = %d' % len(pre_miss))
assert not pre_miss, '追加源文件缺少锚点（先修源文件再追加）: %s' % pre_miss
w('  ==> 预检通过（%d 个必选锚点全部存在于源文件）' % len(REQ))

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
hdr = [i + 1 for i, l in enumerate(lines) if l.startswith('## Phase 14')]
allh = [i + 1 for i, l in enumerate(lines) if l.startswith('## Phase ')]
w('')
w('=== [1] 追加结果 ===')
w('append: bytes %d -> %d (+%d) ; lines %d -> %d' % (n0_b, len(rb), len(rb) - n0_b, n0_l, len(lines)))
w('bom=%s crlf=%d bare_lf=%d' % (rb[:3] == b'\xef\xbb\xbf', rb.count(b'\r\n'),
                                  rb.count(b'\n') - rb.count(b'\r\n')))
w('sha256 = %s' % hashlib.sha256(rb).hexdigest())
w('sha8   = %s' % hashlib.sha256(rb).hexdigest()[:8])
w('Phase 14 标题行 = %s' % hdr)
w('全部 Phase 标题行 (%d) = %s' % (len(allh), allh))
w('新增行数 = %d' % (len(lines) - n0_l))
prefix = b'\xef\xbb\xbf' + txt.rstrip('\r\n').replace('\r\n', '\n').replace('\n', '\r\n').encode('utf-8')
w('前缀逐字节未变 = %s' % rb.startswith(prefix))
w('pre-append 基线前缀锚（308863 B / 4f9b4574）= %s'
  % (hashlib.sha256(rb[:308863]).hexdigest()[:8] == '4f9b4574'))

w('')
w('=== [2] 锚点落盘复核 ===')
miss = []
for k in ANCHORS:
    c = t2.count(k)
    w('  anchor %-42s count=%d %s' % (k, c, 'OK' if c >= 1 else ('(optional)' if k in OPTIONAL else '!! MISSING')))
    if c < 1 and k not in OPTIONAL:
        miss.append(k)
w('缺失（必选）锚点数 = %d' % len(miss))
w('clock %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o) + '\n')

assert not miss, '锚点缺失: %s' % miss
assert len(hdr) == 1, 'Phase 14 标题不唯一: %s' % hdr
assert len(allh) == 14, 'Phase 标题数应为 14，实为 %d' % len(allh)
assert rb[:3] == b'\xef\xbb\xbf'
assert rb.count(b'\n') - rb.count(b'\r\n') == 0, 'bare_lf 不为 0'
assert rb.startswith(prefix), '前缀被改动'
print('ALL CHECKS PASSED ->', OUT)
