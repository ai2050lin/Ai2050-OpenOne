# -*- coding: utf-8 -*-
"""Runner: exec phase3130 with stdout +
traceback capture (SMOKE via env)."""
import io
import os
import sys
import traceback

sys.argv = ['phase3130']
os.environ['P3130_SMOKE'] = os.environ.get(
    'P3130_SMOKE', '1')
SRC = (r'D:\AI2050\Ai2050-OpenOne\tests'
       r'\glm5\phase3130_omega_p128_'
       r'positionspectrum_a1fit_single_'
       r'swap_dyn.py')
OUTP = (r'D:\AI2050\Ai2050-OpenOne\tests'
        r'\gpt5_temp\p3130_run_out.txt')
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
