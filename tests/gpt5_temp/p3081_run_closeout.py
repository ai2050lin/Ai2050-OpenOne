# -*- coding: utf-8 -*-
"""Runner: execute phase3081_closeout.py, capture
traceback to file (bash shim loses stdout)."""
import io
import traceback

SRC = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase3081_closeout.py')
OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\p3081_closeout_run.txt')

src = io.open(SRC, encoding='utf-8').read()
g = {'__name__': '__main__',
     '__file__': SRC}
try:
    exec(compile(src, SRC, 'exec'), g)
    msg = 'RUN_OK'
except Exception:
    msg = 'RUN_FAIL\n' + traceback.format_exc()
io.open(OUT, 'w', encoding='utf-8').write(
    msg + '\n')
