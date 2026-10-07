# -*- coding: utf-8 -*-
"""p3143 closeout syntax fix: close the
summary paren before dict brace."""
import io
import py_compile

SRC = (r'D:\AI2050\Ai2050-OpenOne'
       r'\tests\gpt5_temp'
       r'\p3143_closeout.py')
txt = io.open(SRC, encoding='utf-8').read()
old = """            '+1 by preregistered anchor.'}
    led['measurements'].append(entry)"""
new = """            '+1 by preregistered '
            'anchor.')}
    led['measurements'].append(entry)"""
n = txt.count(old)
assert n == 1, 'count %d' % n
txt = txt.replace(old, new)
io.open(SRC, 'w', encoding='utf-8').write(txt)
txt2 = io.open(SRC, encoding='utf-8').read()
assert txt2.count(new) == 1
py_compile.compile(SRC, doraise=True)
print('syntax fix OK')
