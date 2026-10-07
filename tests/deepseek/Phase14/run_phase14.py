# -*- coding: utf-8 -*-
"""Phase 14 运行包装器：runpy + Tee（自写 UTF-8 日志，规避 `> log` 产 UTF-16 的装置缺陷）。

用法：python run_phase14.py [--smoke]
"""
import io
import os
import sys
import runpy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r'D:\AI2050\Ai2050-OpenOne'
T14 = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase14')
SMOKE = ('--smoke' in sys.argv)
LOG = os.path.join(T14, 'smoke' if SMOKE else '',
                   '_smoke_stdout.log' if SMOKE else '_formal_stdout.log')
os.makedirs(os.path.dirname(LOG), exist_ok=True)
if SMOKE:
    os.environ['SMOKE'] = '1'

TARGET = os.path.join(HERE, 'n2h1a7_prefix_swap.py')
buf = []


class Tee(object):
    def __init__(self, fh):
        self.fh = fh

    def write(self, s):
        self.fh.write(s)
        buf.append(s)

    def flush(self):
        self.fh.flush()


fh = io.open(LOG, 'w', encoding='utf-8')
old = sys.stdout
sys.stdout = Tee(fh)
try:
    sys.argv = [TARGET] + (['--smoke'] if SMOKE else [])
    runpy.run_path(TARGET, run_name='__main__')
finally:
    sys.stdout = old
    fh.write('\n')
    fh.close()

b = io.open(LOG, 'rb').read()
print('LOG %s bytes=%d lines=%d utf16_bom=%s' % (
    LOG, len(b), b.count(b'\n'), b[:2] in (b'\xff\xfe', b'\xfe\xff')))
