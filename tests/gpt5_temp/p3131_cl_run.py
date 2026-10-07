# -*- coding: utf-8 -*-
"""Runner: exec phase3131 closeout with
stdout + traceback capture."""
import io
import sys
import traceback

sys.argv = ['phase3131_closeout']
SRC = (r'D:\AI2050\Ai2050-OpenOne\tests'
       r'\gpt5_temp\phase3131_closeout.py')
OUTP = (r'D:\AI2050\Ai2050-OpenOne\tests'
        r'\gpt5_temp\p3131_closeout_out.txt')
src = io.open(SRC, encoding='utf-8').read()
buf = io.StringIO()
try:
    g = {'__name__': '__main__'}
    exec(compile(src, SRC, 'exec'), g)
except BaseException:
    buf.write(traceback.format_exc())
    buf.write('\n[EXC]\n')
with io.open(OUTP, 'w',
             encoding='utf-8') as f:
    f.write(buf.getvalue())
print('WROTE %d chars'
      % len(buf.getvalue()))
