# -*- coding: utf-8 -*-
"""交付前回滚：把 MEMO 截回 `pre-append-phase20` 基线（逐字节），以便用修正后的生成器重追加。

安全性：断言「当前 MEMO 的前 N 字节 == 基线字节」（即新节只在尾部，前缀未动）后再截断。
"""
import io
import json
import os
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P20T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase20')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
BP = os.path.join(P20T, 'memo_baseline_preappend_phase20.json')

B = json.load(io.open(BP, encoding='utf-8'))
raw = open(MEMO, 'rb').read()
n = int(B['bytes'])
assert len(raw) > n, '当前 MEMO 不比基线长，无需回滚'
assert hashlib.sha256(raw[:n]).hexdigest()[:8] == B['sha8'], \
    '前缀锚不符：前 %d 字节不是基线内容' % n
assert b'## Phase 20' in raw, '当前 MEMO 里没有 Phase 20 节'
# 只在「Phase 20 节起点」处切，确保截断点干净
i = raw.index('## Phase 20'.encode('utf-8'))
# 回退到该标题前的分隔（\r\n\r\n）
j = raw.rindex(b'\r\n\r\n', 0, i) + 4 - 4  # 指向标题行起点之前的 CRLF 对起点
seg = raw[:j] if j > 0 else raw[:n]
# 优先用精确字节数（基线）作为目标
target = raw[:n]
assert target.endswith(b'\r\n'), '截断点不是 CRLF 结尾'
new = target

old_b8 = hashlib.sha256(raw).hexdigest()[:8]
open(MEMO, 'wb').write(new)
chk = open(MEMO, 'rb').read()
t = chk.decode('utf-8-sig')
print('bytes %d -> %d' % (len(raw), len(chk)))
print('sha8 %s -> %s  (基线 %s)' % (old_b8, hashlib.sha256(chk).hexdigest()[:8], B['sha8']))
print('前缀锚恢复 =', hashlib.sha256(chk).hexdigest()[:8] == B['sha8'])
print('bom=%s bare_lf=%d crlf=%d' % (chk[:3] == b'\xef\xbb\xbf',
                                     chk.count(b'\n') - chk.count(b'\r\n'), chk.count(b'\r\n')))
print('Phase 标题数 =', sum(1 for l in t.split('\r\n') if l.startswith('## Phase ')))
print('含 Phase 20 节 =', '## Phase 20' in t)
