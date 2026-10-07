# -*- coding: utf-8 -*-
"""Compile check for phase3131 closeout."""
import io
import py_compile

SRC = (r'D:\AI2050\Ai2050-OpenOne\tests'
       r'\gpt5_temp\phase3131_closeout.py')
OUTP = (r'D:\AI2050\Ai2050-OpenOne\tests'
        r'\gpt5_temp\p3131_cl_compile_out.txt')
try:
    py_compile.compile(SRC, doraise=True)
    msg = 'PY_COMPILE: OK'
except Exception as e:
    msg = 'PY_COMPILE: FAIL\n' + repr(e)
io.open(OUTP, 'w', encoding='utf-8').write(msg)
print('done')
