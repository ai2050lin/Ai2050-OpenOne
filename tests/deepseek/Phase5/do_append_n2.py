# -*- coding: utf-8 -*-
"""把 Phase 5 (N2) 节 append 到 research/deepseek/docs/AGI_DEEPSEEK_MEMO.md，并逐字节复核前缀不变。"""
import os, time, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
SRC = os.path.join(ROOT, 'gpt5_temp', 'memo_append_n2.md')
VER = os.path.join(ROOT, 'gpt5_temp', 'verify_append_n2.txt')

b0 = open(MEMO, 'rb').read()
h0 = hashlib.sha256(b0).hexdigest()

add = open(SRC, encoding='utf-8').read()
ts = time.strftime('%Y-%m-%d %H:%M')
add = add.replace('{{TS}}', ts)
# 归一化换行 -> CRLF（与原文件一致），并去掉源文件尾部多余换行，保证恰好一行分隔
add = add.replace('\r\n', '\n').rstrip('\n')
add = add.replace('\n', '\r\n')
blob = b'\r\n\r\n' + add.encode('utf-8') + b'\r\n'

with open(MEMO, 'ab') as f:
    f.write(blob)

b1 = open(MEMO, 'rb').read()
T = b1.decode('utf-8-sig', errors='replace')
L = T.splitlines()

out = []
out.append('ts %s' % ts)
out.append('before bytes %d sha8 %s' % (len(b0), h0[:8]))
out.append('after  bytes %d sha8 %s' % (len(b1), hashlib.sha256(b1).hexdigest()[:8]))
out.append('prefix_identical %s' % b1.startswith(b0))
out.append('prefix_sha_match %s' % (hashlib.sha256(b1[:len(b0)]).hexdigest() == h0))
out.append('delta bytes %d (expected %d)' % (len(b1) - len(b0), len(blob)))
hdr = '## Phase 5： 探索性探针 N2'
out.append('header_occurrences %d' % T.count(hdr))
out.append('total_lines %d' % len(L))
out.append('has_BOM %s' % b1.startswith(b'\xef\xbb\xbf'))
out.append('CRLF %d ; lone_LF %d' % (b1.count(b'\r\n'), b1.count(b'\n') - b1.count(b'\r\n')))
out.append('--- tail 10 ---')
for l in L[-10:]:
    out.append('  ' + l[:150])
open(VER, 'w', encoding='utf-8').write('\n'.join(out))
print('\n'.join(out))
