# -*- coding: utf-8 -*-
"""把 Phase 12 备忘录节以 UTF-8+BOM+CRLF 追加到 AGI_DEEPSEEK_MEMO.md，并自检落盘。"""
import os, io, hashlib, time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
APP = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase12', 'memo_append_phase12.md')
OUT = os.path.join(ROOT, 'tests', 'deepseek', 'Phase12', 'verify_append_phase12.txt')

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
hdr = [i + 1 for i, l in enumerate(lines) if l.startswith('## Phase 12:')]
allh = [i + 1 for i, l in enumerate(lines) if l.startswith('## Phase ')]
w('append: bytes %d -> %d (+%d) ; lines %d -> %d' % (n0_b, len(rb), len(rb) - n0_b, n0_l, len(lines)))
w('bom=%s crlf=%d bare_lf=%d' % (rb[:3] == b'\xef\xbb\xbf', rb.count(b'\r\n'), rb.count(b'\n') - rb.count(b'\r\n')))
w('sha256 = %s' % hashlib.sha256(rb).hexdigest())
w('sha8   = %s' % hashlib.sha256(rb).hexdigest()[:8])
w('Phase 12 标题行 = %s' % hdr)
w('全部 Phase 标题行 (%d) = %s' % (len(allh), allh))
w('新增行数 = %d' % (len(lines) - n0_l))
prefix = b'\xef\xbb\xbf' + txt.rstrip('\r\n').replace('\r\n', '\n').replace('\n', '\r\n').encode('utf-8')
w('前缀逐字节未变 = %s' % rb.startswith(prefix))

need = ['Phase 12: 逐层残差替换',
        'N2h1-α-5', 'ALLOCATION_AMBIGUOUS', 'LAST_POS_STATE_SUFFICIENT',
        '10.574739583333335', 'FULL_SWAP', '0.000e+00',
        '25.35', '1.06', '0.7833', '0.9154', '0.5500', '0.8741',
        '0.9955', '0.0048', '0.5745', '0.3867', '0.8316', '0.5895',
        '0.1094', '0.4675', '0.4757',
        'proj_share_u6', '0.680', '0.148',
        'E2b', 'E3', 'E4', 'E5', 'E6', 'F11', 'F12', 'F13', 'F7',
        '264.2 s', 'cross_alpha', 'first_reach', 'HARKing',
        '(r)', '(s)', '位点间配对检验', 'Phase 13 候选',
        'f2311d38', '7bf4510a', 'e58513a8', '67f38c53', '4280b23c', 'f13ea993',
        '295', '9605b42e', '277f49da', '236092']
miss = []
for k in need:
    c = t2.count(k)
    w('  anchor %-34s count=%d %s' % (k, c, 'OK' if c >= 1 else '!! MISSING'))
    if c < 1:
        miss.append(k)
w('clock %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o))
assert not miss, '锚点缺失: %s' % miss
assert len(hdr) == 1, 'Phase 12 标题不唯一'
assert len(allh) == 12, 'Phase 标题数应为 12，实为 %d' % len(allh)
print('DONE ->', OUT)
