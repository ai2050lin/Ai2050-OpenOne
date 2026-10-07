# -*- coding: utf-8 -*-
"""p3086 compile probe: py_compile the omega_p83 main script, write result to file."""
import py_compile
import traceback

SRC = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\phase3086_omega_p83_continuum_test.py'
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3086_compile.txt'

with open(OUT, 'w', encoding='utf-8') as f:
    try:
        py_compile.compile(SRC, doraise=True)
        f.write('COMPILE_OK\n')
    except Exception:
        f.write('COMPILE_FAIL\n')
        f.write(traceback.format_exc())
