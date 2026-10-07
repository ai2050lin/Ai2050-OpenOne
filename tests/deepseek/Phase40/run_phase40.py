# -*- coding: utf-8 -*-
"""Phase40 驱动：run_phase40.py smoke|formal（UTF-8 log；绕开 shim 重定向 UTF-16 问题）。"""
import os, sys, runpy, traceback, io

ROOT = r'D:\AI2050\Ai2050-OpenOne'
PH = os.path.join(ROOT, 'tests', 'deepseek', 'Phase40')
mode = sys.argv[1] if len(sys.argv) > 1 else 'smoke'
os.environ['SMOKE'] = '1' if mode == 'smoke' else '0'
os.chdir(ROOT)

buf = io.StringIO()
code = None
try:
    runpy.run_path(os.path.join(PH, 'q06_steer_base.py'), run_name='__main__')
    code = 0
except Exception:
    traceback.print_exc(file=buf)
    code = 1
tb = buf.getvalue()
if tb:
    p = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase40',
                     ('smoke' if mode == 'smoke' else ''), '_q06_traceback.txt')
    with open(p, 'a', encoding='utf-8') as f:
        f.write(tb + '\n')
    print(tb)
sys.exit(code)
