# -*- coding: utf-8 -*-
"""把 Phase 8 节追加到 deepseek 备忘录（append-only，幂等，落盘复核）。"""
import os, io, hashlib, time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
SRC = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase8', 'memo_append_phase8.md')
OUT = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase8', 'verify_append_phase8.txt')
HDR = '## Phase 8:'

raw = open(SRC, 'rb').read()
text = raw.decode('utf-8').replace('\r\n', '\n').replace('\n', '\r\n')
if not text.startswith('\r\n'):
    text = '\r\n' + text
add = text.encode('utf-8')

b0 = open(MEMO, 'rb').read()
n0 = len(b0)
h0 = hashlib.sha256(b0).hexdigest()
L0 = b0.decode('utf-8-sig').splitlines()

o = []
o.append('append at %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
o.append('src bytes %d' % len(raw))
o.append('memo before_bytes %d before_lines %d sha8 %s' % (n0, len(L0), h0[:8]))
o.append('hdr_already_count %d' % b0.decode('utf-8-sig').count(HDR))

if b0.decode('utf-8-sig').count(HDR) > 0:
    o.append('SKIP: header already present (idempotent)')
else:
    with open(MEMO, 'ab') as f:
        f.write(add)

b1 = open(MEMO, 'rb').read()
T1 = b1.decode('utf-8-sig')
L1 = T1.splitlines()
o.append('memo after_bytes  %d after_lines  %d sha8 %s' % (len(b1), len(L1), hashlib.sha256(b1).hexdigest()[:8]))
o.append('delta_bytes %d (appended %d)' % (len(b1) - n0, len(add)))
o.append('prefix_unchanged %s' % (b1[:n0] == b0))
o.append('bom_preserved %s' % (b1[:3] == b'\xef\xbb\xbf'))
o.append('utf8_strict_ok %s' % bool(b1.decode('utf-8')))
o.append('hdr_occurrences %d ; hdr_lines %s' %
         (T1.count(HDR), [i + 1 for i, l in enumerate(L1) if l.startswith(HDR)]))
o.append('bare_lf %d crlf %d' % (b1.count(b'\n') - b1.count(b'\r\n'), b1.count(b'\r\n')))
o.append('--- sections after append ---')
for l in L1:
    if l.startswith('## '):
        o.append('   ' + l[:95])
o.append('--- key numbers present ---')
for k in ['0.0742', '0.4717', 'I_nl = 6.85', 'G1 分布式搬运', '总 99.8', '99.8 s', '37c1a609']:
    o.append('   %-16s -> %d' % (k, T1.count(k)))
o.append('--- tail 3 ---')
for l in L1[-3:]:
    o.append('   ' + l[:140])

io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o))
print('\n'.join(o))
