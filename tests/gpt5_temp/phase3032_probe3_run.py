# -*- coding: utf-8 -*-
"""Run probe3, capture traceback to file."""
import io
import traceback

import runpy

try:
    runpy.run_path(
        r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
        r'\phase3032_probe3.py',
        run_name='__main__')
except Exception:
    tb = traceback.format_exc()
    io.open(r'D:\AI2050\Ai2050-OpenOne\tests'
            r'\gpt5_temp\probe3_trace.txt', 'w',
            encoding='utf-8').write(tb)
    print('TRACED')
