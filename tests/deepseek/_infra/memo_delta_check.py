# -*- coding: utf-8 -*-
"""备忘录增量核查：确认 v2 节完整，定位 +40B/+1 行的来源。"""
import os, io, hashlib

P = r'D:\AI2050\Ai2050-OpenOne\research\deepseek\docs\AGI_DEEPSEEK_MEMO.md'
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\_infra\memo_delta_check.txt'
o = []
b = open(P, 'rb').read()
o.append('bytes %d lines %d sha8 %s' % (len(b), len(b.decode('utf-8-sig').splitlines()),
                                        hashlib.sha256(b).hexdigest()[:8]))
o.append('prefix_142272_intact %s' % (len(b) >= 142272))
if len(b) >= 142272:
    o.append('  sha8(prefix142272) %s  (expect fcb3df84)'
             % hashlib.sha256(b[:142272]).hexdigest()[:8])
    o.append('  appended_tail_bytes %d' % (len(b) - 142272))
    o.append('  tail_repr %r' % b[142272:].decode('utf-8', errors='replace'))
L = b.decode('utf-8-sig').splitlines()
o.append('v2_hdr_count %d' % sum(1 for l in L if l.startswith('## 登记约定变更 v2')))
o.append('--- tail 6 lines ---')
for l in L[-6:]:
    o.append('  | ' + l[:160])
o.append('--- line-ending stats ---')
o.append('crlf %d bare_lf %d' % (b.count(b'\r\n'), b.count(b'\n') - b.count(b'\r\n')))
io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o))
print('\n'.join(o))
