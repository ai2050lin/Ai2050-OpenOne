# -*- coding: utf-8 -*-
import io
import os
import traceback

os.environ['P3132_SMOKE'] = \
    os.environ.get('P3132_SMOKE', '1')
SRC = (r'D:\AI2050\Ai2050-OpenOne\tests'
       r'\glm5\phase3132_omega_p130_'
       r'forkcausal_single256.py')
OUTF = (r'D:\AI2050\Ai2050-OpenOne'
        r'\tests\gpt5_temp'
        r'\p3132_run_out.txt')
buf = []
try:
    src = io.open(SRC, encoding='utf-8')\
        .read()
    g = {'__name__': '__main__',
         '__file__': SRC}
    exec(compile(src, SRC, 'exec'), g)
    buf.append('EXEC OK')
except BaseException:
    buf.append('EXEC FAIL')
    buf.append(traceback.format_exc())
with io.open(OUTF, 'w',
             encoding='utf-8') as f:
    f.write('\n'.join(buf) + '\n')
print('runner done')
