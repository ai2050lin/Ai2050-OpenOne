# -*- coding: utf-8 -*-
import os
root = r'D:\AI2050\Ai2050-OpenOne'
d = os.path.join(root, '.workbuddy', 'memory')
os.makedirs(d, exist_ok=True)
dst = os.path.join(d, '2026-10-01.md')
src = os.path.join(root, 'gpt5_temp', 'wlog_n3.md')
add = open(src, encoding='utf-8').read()
if not os.path.exists(dst):
    open(dst, 'w', encoding='utf-8').write('# 2026-10-01 工作日志\n')
b0 = os.path.getsize(dst)
with open(dst, 'a', encoding='utf-8') as f:
    f.write(add)
b1 = os.path.getsize(dst)
T = open(dst, encoding='utf-8').read()
mem = open(os.path.join(d, 'MEMORY.md'), encoding='utf-8').read()
out = []
out.append('daily before %d after %d delta %d' % (b0, b1, b1 - b0))
out.append('has_section %s' % ('## Phase 7（deepseek 线）' in T))
out.append('tail_ok %s' % T.rstrip().endswith('零 OOM。'))
out.append('memory bytes %d' % len(mem.encode('utf-8')))
for k in ['主轴三段 / 条件子空间（N1–N3', '跨族近乎正交', '装置铁律（N3 新增）', '机制链状态（G1 线，3151–3153）', 'N 系列死线']:
    out.append('mem has %-34s -> %d' % (k, mem.count(k)))
open(os.path.join(root, 'gpt5_temp', 'verify_memory_n3.txt'), 'w', encoding='utf-8').write('\n'.join(out))
print('done')
