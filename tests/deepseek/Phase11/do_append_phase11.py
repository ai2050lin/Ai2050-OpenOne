# -*- coding: utf-8 -*-
"""把 Phase 11 备忘录节以 UTF-8+BOM+CRLF 追加到 AGI_DEEPSEEK_MEMO.md，并自检落盘。"""
import os, io, hashlib, time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
APP = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase11', 'memo_append_phase11.md')
OUT = os.path.join(ROOT, 'tests', 'deepseek', 'Phase11', 'verify_append_phase11.txt')

o = []


def w(s=''):
    o.append(str(s)); print(s)


raw0 = open(MEMO, 'rb').read()
assert raw0[:3] == b'\xef\xbb\xbf', 'BOM 丢失'
txt = raw0.decode('utf-8-sig')
n0_b, n0_l = len(raw0), len(raw0.split(b'\r\n'))
body = io.open(APP, encoding='utf-8').read().strip('\r\n')
new = txt.rstrip('\r\n') + '\r\n\r\n' + body + '\r\n'
out = b'\xef\xbb\xbf' + new.replace('\r\n', '\n').replace('\n', '\r\n').encode('utf-8')
open(MEMO, 'wb').write(out)

# ---- 落盘复核（真实磁盘）----
rb = open(MEMO, 'rb').read()
t2 = rb.decode('utf-8-sig')
lines = t2.splitlines()
hdr = [i + 1 for i, l in enumerate(lines) if l.startswith('## Phase 11:')]
allh = [i + 1 for i, l in enumerate(lines) if l.startswith('## Phase ')]
w('append: bytes %d -> %d (+%d) ; lines %d -> %d' % (n0_b, len(rb), len(rb) - n0_b, n0_l, len(lines)))
w('bom=%s crlf=%d bare_lf=%d' % (rb[:3] == b'\xef\xbb\xbf', rb.count(b'\r\n'), rb.count(b'\n') - rb.count(b'\r\n')))
w('sha256 = %s' % hashlib.sha256(rb).hexdigest())
w('Phase 11 标题行 = %s' % hdr)
w('全部 Phase 标题行 = %s' % allh)
w('新增行数 = %d' % (len(lines) - n0_l))
# 前缀逐字节未变（append-only）
prefix = b'\xef\xbb\xbf' + txt.rstrip('\r\n').replace('\r\n', '\n').replace('\n', '\r\n').encode('utf-8')
w('前缀逐字节未变 = %s' % rb.startswith(prefix))

need = ['Phase 11: 逐层累积在噪声带下成立',
        '10.574739583333335', 'Q2_ESTABLISHED_WITH_BAND',
        '0.8720', '0.9897', '5.587', '0.4696', '0.4655',
        '329.5 s', '6/18', 'F9', 'F7', 'Phase 12 候选',
        '逐对 dDonor', '自基全剖面', '置换零假设', 'BASIS_SENSITIVE',
        '850987f3', 'a9c59555', '6fb3ef82', '294']
miss = []
for k in need:
    c = t2.count(k)
    w('  anchor %-40s count=%d %s' % (k, c, 'OK' if c >= 1 else '!! MISSING'))
    if c < 1:
        miss.append(k)
w('clock %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o))
assert not miss, '锚点缺失: %s' % miss
assert len(hdr) == 1, 'Phase 11 标题不唯一'
print('DONE ->', OUT)
