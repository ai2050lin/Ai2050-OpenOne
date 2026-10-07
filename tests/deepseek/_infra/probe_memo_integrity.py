# -*- coding: utf-8 -*-
import os, re, time, hashlib
ROOT = r'D:\AI2050\Ai2050-OpenOne'
p = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
out = []
st = os.stat(p)
out.append('size=%d mtime=%s ctime=%s' % (st.st_size,
           time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime)),
           time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_ctime))))
b = open(p, 'rb').read()
out.append('sha8=%s' % hashlib.sha256(b).hexdigest()[:8])
out.append('has_BOM=%s' % (b[:3] == b'\xef\xbb\xbf'))
out.append('crlf=%d  lf_only=%d' % (b.count(b'\r\n'), b.count(b'\n') - b.count(b'\r\n')))
T = b.decode('utf-8-sig')
L = T.splitlines()
out.append('splitlines=%d  newline_count=%d' % (len(L), T.count('\n')))
out.append('')
out.append('--- header inventory (last 14) ---')
hs = [(i + 1, l) for i, l in enumerate(L) if l.startswith('## ')]
for i, l in hs[-14:]:
    out.append('  L%-6d %s' % (i, l[:120]))
out.append('')
out.append('--- key section presence ---')
for k in ['## Phase 3149', '## Phase 3150', '## 设计草案', '## 探索性探针 E1', '## 探索性探针 N1',
          '3150（Ω-P148）预注册', 'Ω-P149']:
    idx = [i + 1 for i, l in enumerate(L) if k in l]
    out.append('  %-28s count=%d lines=%s' % (k, T.count(k), idx[:6]))
open(os.path.join(ROOT, 'gpt5_temp', 'probe_memo_integrity.txt'), 'w', encoding='utf-8').write('\n'.join(out))
print('ok')
