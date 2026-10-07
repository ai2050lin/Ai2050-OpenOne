# -*- coding: utf-8 -*-
"""把 Phase 9 节追加到 deepseek 备忘录（append-only，幂等，落盘复核）。"""
import os, io, hashlib, time, json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
SRC = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase9', 'memo_append_phase9.md')
OUT = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase9', 'verify_append_phase9.txt')
INFRA = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra')
HDR = '## Phase 9:'

raw = open(SRC, 'rb').read()
text = raw.decode('utf-8').replace('\r\n', '\n').replace('\n', '\r\n')
if not text.startswith('\r\n'):
    text = '\r\n' + text
add = text.encode('utf-8')

b0 = open(MEMO, 'rb').read()
n0 = len(b0)
h0 = hashlib.sha256(b0).hexdigest()
T0 = b0.decode('utf-8-sig')

o = []
o.append('append at %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
o.append('src bytes %d ; src lines %d' % (len(raw), len(raw.decode('utf-8').splitlines())))
o.append('memo before_bytes %d before_lines %d sha8 %s bare_lf %d' %
         (n0, len(T0.splitlines()), h0[:8], b0.count(b'\n') - b0.count(b'\r\n')))
o.append('hdr_already_count %d' % T0.count(HDR))

# 基线对账（pre-append 基线）
P9T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase9')
bp = os.path.join(P9T, 'memo_baseline_preappend_phase9.json')
if os.path.isfile(bp):
    B = json.load(io.open(bp, encoding='utf-8'))
    o.append('baseline_match bytes %s sha256 %s' %
             (B['bytes'] == n0, B['sha256'] == h0))

if T0.count(HDR) > 0:
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
o.append('phase1_8_hdrs_present %s' %
         all(T1.count('## Phase %d:' % k) == 1 for k in range(1, 9)))
o.append('--- sections after append ---')
for l in L1:
    if l.startswith('## '):
        o.append('   ' + l[:100])
o.append('--- key numbers present (count) ---')
for k in ['10.574739583333335', '1.2624', '1.3458', 'J=1.42', 'J=5.41', 'x* ≈ 0.6', '0.0578',
          '0.169', '0.266', '4.224', '41.9 s', 'H0_no_verdict', 'e99ebd02', 'ece7ed8c']:
    o.append('   %-22s -> %d' % (k, T1.count(k)))
o.append('--- tail 3 ---')
for l in L1[-3:]:
    o.append('   ' + l[:140])

io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o))
print('\n'.join(o))
