# -*- coding: utf-8 -*-
"""Probe 3: scan every str-% BinOp in
phase3095_closeout.py with a faithful
C-style percent scanner."""
import ast
import io

SRC = (r'D:\AI2050\Ai2050-OpenOne'
       r'\tests\gpt5_temp'
       r'\phase3095_closeout.py')
OUT = (r'D:\AI2050\Ai2050-OpenOne'
       r'\tests\gpt5_temp'
       r'\probe_pct3_3095.txt')
src = io.open(SRC, encoding='utf-8').read()
tree = ast.parse(src)
CONVS = set('sdfregxXoce%')
FLAGS = set('+- #0')
lines = []


def scan(fmt, tag):
    i = 0
    n = len(fmt)
    while i < n:
        if fmt[i] != '%':
            i += 1
            continue
        j = i + 1
        while j < n and fmt[j] in FLAGS:
            j += 1
        while j < n and fmt[j].isdigit():
            j += 1
        if j < n and fmt[j] == '.':
            j += 1
            while j < n and fmt[j].isdigit():
                j += 1
        if j >= n:
            lines.append('%s: trailing %% at %d'
                         % (tag, i))
            break
        conv = fmt[j]
        if conv in CONVS:
            i = j + 1
            continue
        lines.append(
            '%s: BAD %% at %d conv=%r ctx=%r'
            % (tag, i, conv,
               fmt[max(0, i - 45):i + 12]))
        i = j + 1


for node in ast.walk(tree):
    if (isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Mod)
            and isinstance(node.left,
                           ast.Constant)
            and isinstance(node.left.value,
                           str)):
        scan(node.left.value,
             'len=%d head=%r'
             % (len(node.left.value),
                node.left.value[:40]))

with io.open(OUT, 'w',
             encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('PROBE3_DONE bad=%d' % len(lines))
