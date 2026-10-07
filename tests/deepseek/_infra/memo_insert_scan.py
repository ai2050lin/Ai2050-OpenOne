# -*- coding: utf-8 -*-
"""定位备忘录中被外部进程插入的 +1 行/~40B（在 v2 节之前，即 L1–L1567）。"""
import io, os, hashlib
from collections import Counter

P = r'D:\AI2050\Ai2050-OpenOne\research\deepseek\docs\AGI_DEEPSEEK_MEMO.md'
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\_infra\memo_insert_scan.txt'
b = open(P, 'rb').read()
raw = b.decode('utf-8-sig')
L = raw.split('\r\n')
o = []
o.append('bytes %d crlf_lines %d' % (len(b), len(L)))

# v2 节起点
v2 = [i for i, l in enumerate(L) if l.startswith('## 登记约定变更 v2')][0]
o.append('v2 header at 1-based line %d' % (v2 + 1))
head = L[:v2]
o.append('lines before v2 header: %d' % len(head))

# 1) 精确重复行（非空、长度<80）
c = Counter(x for x in head if x.strip() and len(x) < 80)
dup = [(k, v) for k, v in c.items() if v >= 2]
o.append('--- exact duplicate short lines (%d) ---' % len(dup))
for k, v in sorted(dup, key=lambda z: -z[1])[:25]:
    idx = [i + 1 for i, x in enumerate(head) if x == k]
    o.append('  x%d  L%s  | %s' % (v, idx, k[:80]))

# 2) 可疑短行（1..30 字符，非空、非表格/标题/列表符开头）
susp = [(i + 1, x) for i, x in enumerate(head)
        if x.strip() and len(x.strip()) <= 30
        and not x.lstrip().startswith(('|', '#', '-', '*', '>', '`', '1.', '2.', '3.', '4.', '5.',
                                       '6.', '7.', '8.', '9.', '0.', '---', '```'))]
o.append('--- suspicious short lines (%d) ---' % len(susp))
for i, x in susp[:60]:
    o.append('  L%-5d | %r' % (i, x[:60]))

# 3) 每 100 行快速摘要（供人工扫视）
o.append('--- structural digest: all "## " & "### " headers with line numbers ---')
for i, x in enumerate(L):
    if x.startswith('## ') or x.startswith('### '):
        o.append('  L%-5d %s' % (i + 1, x[:90]))

io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o))
print('wrote', OUT, len(o), 'lines')
print('\n'.join(o[:40]))
