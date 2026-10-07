# -*- coding: utf-8 -*-
"""把 Phase 10 备忘录节以 UTF-8+BOM+CRLF 追加到 AGI_DEEPSEEK_MEMO.md，并自检落盘。"""
import os, io, hashlib, time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
APP = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase10', 'memo_append_phase10.md')
OUT = os.path.join(ROOT, 'tests', 'deepseek', 'Phase10', 'verify_append_phase10.txt')

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
hdr = [i + 1 for i, l in enumerate(lines) if l.startswith('## Phase 10:')]
allh = [i + 1 for i, l in enumerate(lines) if l.startswith('## Phase ')]
w('append: bytes %d -> %d (+%d) ; lines %d -> %d' % (n0_b, len(rb), len(rb) - n0_b, n0_l, len(lines)))
w('bom=%s crlf=%d bare_lf=%d' % (rb[:3] == b'\xef\xbb\xbf', rb.count(b'\r\n'), rb.count(b'\n') - rb.count(b'\r\n')))
w('sha256 = %s' % hashlib.sha256(rb).hexdigest())
w('Phase 10 标题行 = %s' % hdr)
w('全部 Phase 标题行 = %s' % allh)
w('新增行数 = %d' % (len(lines) - n0_l))
# 关键锚点必须在盘上
need = ['Phase 10: 软阈值是逐层累积的', 'E0（ℓ=6, α=1）= 10.574739583333335',
        'Q2_accumulate', 'Q_ROBUST', 'R²_lin = 1.0000', 'overlap(U_ℓ, U6)',
        '0.6892', '0.0298', '253.0 s', '49f3ebda', '栈=软门', 'Phase 11 候选']
for k in need:
    c = t2.count(k)
    w('  anchor %-42s count=%d %s' % (k, c, 'OK' if c >= 1 else '!! MISSING'))
    assert c >= 1, '锚点缺失: %s' % k
w('clock %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o))
print('DONE ->', OUT)
