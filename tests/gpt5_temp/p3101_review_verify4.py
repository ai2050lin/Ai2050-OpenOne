# -*- coding: utf-8 -*-
"""Extract keyword snippets from GPT MEMO lines for R54/R48/R44 checks."""
import io

MEMO = r'research\gpt5\docs\AGI_GPT5_MEMO.md'
OUT = r'tests\gpt5_temp\p3101_review_verify4.txt'

targets = {
    11851: ['96.5', '26.55', '41.18', '6%'],
    11885: ['96.5', '26.55', '41.18', '6%'],
    11888: ['96.5', '26.55', '41.18', '6%'],
    11903: ['96.5', '26.55', '41.18', '6%'],
    13121: ['96.5', '26.55', '41.18', '6%'],
    12041: ['submodular', '467', '0.02'],
    12046: ['submodular', '467', '0.02'],
    12053: ['submodular', '467', '0.02'],
    12056: ['submodular', '467', '0.02'],
    12085: ['0.46', '0.93', 'ties', 'rank'],
    12092: ['0.46', '0.93', 'ties', 'rank'],
    12095: ['0.46', '0.93', 'ties', 'rank'],
}

lines = []
with io.open(MEMO, encoding='utf-8', errors='ignore') as f:
    all_lines = f.readlines()

for ln_no, kws in sorted(targets.items()):
    if ln_no > len(all_lines):
        lines.append('L%d: OUT OF RANGE (total %d)' % (ln_no, len(all_lines)))
        continue
    t = all_lines[ln_no - 1]
    lines.append('--- L%d (len=%d) ---' % (ln_no, len(t)))
    low = t.lower()
    shown = set()
    for kw in kws:
        i = low.find(kw.lower())
        if i < 0:
            continue
        s = max(0, i - 120)
        e = min(len(t), i + 220)
        seg = t[s:e].replace('\n', ' ')
        key = seg[:40]
        if key in shown:
            continue
        shown.add(key)
        lines.append('  [%s] ...%s...' % (kw, seg))
    if not shown:
        lines.append('  (no keyword hit) head: %s' % t[:200].replace('\n', ' '))

# Phase header context: find nearest '## Phase' above each target
for ln_no in sorted(targets):
    for j in range(ln_no - 1, max(0, ln_no - 1200), -1):
        if all_lines[j].startswith('## Phase'):
            lines.append('L%d belongs to: %s'
                         % (ln_no, all_lines[j].strip()[:100]))
            break

with io.open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('OK')
