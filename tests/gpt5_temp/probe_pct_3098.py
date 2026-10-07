# -*- coding: utf-8 -*-
"""ast probe: scan all str-% BinOps in
phase3098_closeout.py for illegal
literal % conversions."""
import ast
import io

P = (r'D:\AI2050\Ai2050-OpenOne'
     r'\tests\gpt5_temp'
     r'\phase3098_closeout.py')
src = io.open(P, encoding='utf-8').read()
tree = ast.parse(src)
bad = []


def scan_node(node, fmt):
    if isinstance(node, ast.Constant) \
            and isinstance(node.value, str):
        s = node.value
        i = 0
        while i < len(s):
            if s[i] == '%':
                j = i + 1
                while j < len(s) and (
                        s[j] in '0123456789'
                        or s[j] in '-+ #'
                        or s[j] == '.'):
                    j += 1
                if j >= len(s) or \
                        s[j] not in \
                        'sdfgeEfFgxXoRrc%':
                    bad.append(
                        (node.lineno,
                         s[max(0, i - 12):
                           j + 12]))
                i = j + 1
            else:
                i += 1


for node in ast.walk(tree):
    if isinstance(node, ast.BinOp) \
            and isinstance(node.op,
                           ast.Mod):
        scan_node(node.left, 'fmt')
        if isinstance(node.right,
                      ast.Tuple):
            for e in node.right.elts:
                scan_node(e, 'arg')
        else:
            scan_node(node.right, 'arg')
rep = 'ILLEGAL %d: %s' % (
    len(bad), bad[:10])
with io.open(
        P.replace('_closeout.py',
                  '_probe_pct.txt'),
        'w', encoding='utf-8') as f:
    f.write(rep + '\n')
print('PCT_SCAN_DONE %d' % len(bad))
