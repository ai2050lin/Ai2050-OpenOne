# -*- coding: utf-8 -*-
"""Locate theory/formula sections in GPT MEMO."""
import io

MEMO = r'research\gpt5\docs\AGI_GPT5_MEMO.md'
OUT = r'tests\gpt5_temp\p3103_theory_locate.txt'

kws = ['Unified Theory', '统一机制公式', '核心公式',
       '拼图清单', 'RDC 主', '主公式', '公理',
       'invariant semantic', '不变语义基',
       '证据等级', 'evidence_grade']

lines = []
with io.open(MEMO, encoding='utf-8',
             errors='ignore') as f:
    all_lines = f.readlines()
lines.append('total lines %d' % len(all_lines))

cur_phase = ''
for i, ln in enumerate(all_lines, 1):
    if ln.startswith('## Phase '):
        cur_phase = ln.strip()[:110]
    low = ln.lower()
    for kw in kws:
        if kw.lower() in low:
            lines.append('L%d [%s] kw=%s | %s'
                         % (i, cur_phase[:70], kw,
                            ln.strip()[:150]))
            break

with io.open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('OK hits=%d' % (len(lines) - 1))
