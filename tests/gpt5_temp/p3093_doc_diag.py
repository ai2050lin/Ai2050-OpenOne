# -*- coding: utf-8 -*-
"""Diagnose DOC template %-directive order in
p3093_patch_a2.py vs its arg tuple."""
import io
import re

src = io.open(
    r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
    r'\p3093_patch_a2.py',
    encoding='utf-8').read()
i0 = src.index("DOC = '''") + len("DOC = '''")
i1 = src.index("''' % (", i0)
tpl = src[i0:i1]

# find %-directives with context
out = []
for m in re.finditer(r'%[sd]', tpl):
    a = max(0, m.start() - 45)
    ctx = tpl[a:m.end() + 8]
    ctx = ctx.replace('\n', '|')
    out.append('%3d  %s' % (m.start(), ctx))
argline = src[i1:i1 + 200].split('\n')
rep = ['DIRECTIVES=%d' % len(out)]
rep += out
rep.append('ARGS=' + ' '.join(argline))
io.open(
    r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
    r'\p3093_doc_diag.txt', 'w',
    encoding='utf-8').write('\n'.join(rep) + '\n')
print('DIAG_OK')
