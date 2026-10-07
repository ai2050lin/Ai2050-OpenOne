# -*- coding: utf-8 -*-
"""Runner: execute phase3082 main, capture
stdout+traceback to file."""
import io
import traceback

SRC = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
       r'\phase3082_omega_p79_ds7b_negative_'
       r'anatomy.py')
OUT = (r'D:\AI2050\Ai2050-OpenOne\tests'
       r'\gpt5_temp\p3082_run.txt')

src = io.open(SRC, encoding='utf-8').read()
g = {'__name__': '__main__',
     '__file__': SRC}
import contextlib
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        exec(compile(src, SRC, 'exec'), g)
    msg = 'STDOUT: ' + buf.getvalue() + 'RUN_OK'
except Exception:
    msg = ('STDOUT: ' + buf.getvalue()
           + 'RUN_FAIL\n'
           + traceback.format_exc())
io.open(OUT, 'w', encoding='utf-8').write(
    msg + '\n')
