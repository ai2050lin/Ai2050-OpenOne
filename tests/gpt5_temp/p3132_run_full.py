# -*- coding: utf-8 -*-
"""Full-run launcher: P3132_SMOKE=0."""
import os

os.environ['P3132_SMOKE'] = '0'
SRC = (r'D:\AI2050\Ai2050-OpenOne\tests'
       r'\gpt5_temp\p3132_run.py')
g = {'__name__': '__main__',
     '__file__': SRC}
exec(compile(open(SRC, encoding='utf-8')
             .read(), SRC, 'exec'), g)
