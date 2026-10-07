# -*- coding: utf-8 -*-
"""把「登记约定变更 v2」节追加到 deepseek 备忘录（append-only，幂等）。

纪律：统一 CRLF、保持 BOM、前缀逐字节不变、追加后复核真实磁盘。
"""
import os, io, hashlib, time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
SRC = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra', 'memo_append_phasedirs_v2.md')
OUT = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra', 'verify_append_phasedirs.txt')
HDR = '## 登记约定变更 v2'

raw = open(SRC, 'rb').read()
text = raw.decode('utf-8')
text = text.replace('\r\n', '\n').replace('\n', '\r\n')
if not text.startswith('\r\n'):
    text = '\r\n' + text
add = text.encode('utf-8')

b0 = open(MEMO, 'rb').read()
n0 = len(b0)
h0 = hashlib.sha256(b0).hexdigest()
L0 = b0.decode('utf-8-sig').splitlines()

out = []
out.append('append at %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
out.append('src bytes %d' % len(raw))
out.append('memo before_bytes %d before_lines %d sha8 %s' % (n0, len(L0), h0[:8]))
out.append('hdr_already_count %d' % b0.decode('utf-8-sig').count(HDR))

if b0.decode('utf-8-sig').count(HDR) > 0:
    out.append('SKIP: header already present (idempotent)')
else:
    with open(MEMO, 'ab') as f:
        f.write(add)

b1 = open(MEMO, 'rb').read()
T1 = b1.decode('utf-8-sig')
L1 = T1.splitlines()
out.append('memo after_bytes  %d after_lines  %d sha8 %s' % (len(b1), len(L1), hashlib.sha256(b1).hexdigest()[:8]))
out.append('delta_bytes %d (appended %d)' % (len(b1) - n0, len(add)))
out.append('prefix_unchanged %s' % (b1[:n0] == b0))
out.append('bom_preserved %s' % (b1[:3] == b'\xef\xbb\xbf'))
out.append('utf8_strict_ok %s' % bool(b1.decode('utf-8')))
out.append('hdr_occurrences %d' % T1.count(HDR))
idx = [i + 1 for i, l in enumerate(L1) if l.startswith(HDR)]
out.append('hdr_line %s' % idx)
out.append('bare_lf %d crlf %d' % (b1.count(b'\n') - b1.count(b'\r\n'), b1.count(b'\r\n')))
out.append('--- phase headers after append ---')
for l in L1:
    if l.startswith('## Phase') or l.startswith('## 复核') or l.startswith('## 登记'):
        out.append('   ' + l[:90])
out.append('--- tail 4 ---')
for l in L1[-4:]:
    out.append('   ' + l[:150])

io.open(OUT, 'w', encoding='utf-8').write('\n'.join(out))
print('\n'.join(out))
