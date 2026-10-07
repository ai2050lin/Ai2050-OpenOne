# -*- coding: utf-8 -*-
"""Runner: exec phase3130_closeout.py, capture
stdout+traceback to file."""
import io
import traceback
import contextlib

SRC = (r'D:\AI2050\Ai2050-OpenOne\tests'
       r'\gpt5_temp\phase3130_closeout.py')
OUT = (r'D:\AI2050\Ai2050-OpenOne\tests'
       r'\gpt5_temp\p3130_closeout_out.txt')

src = io.open(SRC, encoding='utf-8').read()
buf = io.StringIO()
err = ''
try:
    with contextlib.redirect_stdout(buf):
        exec(compile(src, 'phase3130_closeout.py',
                     'exec'),
             {'__name__': '__main__'})
except Exception:
    err = traceback.format_exc()

out = buf.getvalue()
if err:
    out += '\n[EXC]\n' + err
with io.open(OUT, 'w', encoding='utf-8') as f:
    f.write(out)
print('WROTE %d chars' % len(out))
