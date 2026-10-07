# -*- coding: utf-8 -*-
"""Extract structure of AGI_DEEPSEEK_MEMO.md -> report file."""
import os

SRC = r'D:\AI2050\Ai2050-OpenOne\research\deepseek\docs\AGI_DEEPSEEK_MEMO.md'
OUT_DIR = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\memo_review_20261001'
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, 'structure_report.txt')

report = []
report.append('exists: %s' % os.path.exists(SRC))
if os.path.exists(SRC):
    report.append('size_bytes: %d' % os.path.getsize(SRC))
    with open(SRC, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    report.append('total_lines: %d' % len(lines))
    report.append('')
    for i, ln in enumerate(lines, 1):
        s = ln.rstrip('\n').rstrip('\r')
        if s.startswith('#'):
            report.append('%6d | %s' % (i, s))

with open(OUT, 'w', encoding='utf-8') as w:
    w.write('\n'.join(report))
print('OK ->', OUT)
