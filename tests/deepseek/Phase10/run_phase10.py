# -*- coding: utf-8 -*-
"""Phase 10 运行包装器：把主脚本 stdout 全量落成 UTF-8 日志（smoke/formal 各一份）。

本机 bash shim 会把 `python x.py > log` 的日志写成 UTF-16 且路径转换偶发错乱，
故统一由本包装器自己写日志，保证 _smoke_stdout.log / _formal_stdout.log 为 UTF-8。
用法：python run_phase10.py smoke | python run_phase10.py formal
"""
import os, sys, io, runpy, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
P10T = os.path.abspath(os.path.join(HERE, '..', '..', 'deepseek_temp', 'Phase10'))

mode = (sys.argv[1] if len(sys.argv) > 1 else 'formal').lower()
assert mode in ('smoke', 'formal'), mode
if mode == 'smoke':
    os.environ['SMOKE'] = '1'

LOG = os.path.join(P10T, '_smoke_stdout.log' if mode == 'smoke' else '_formal_stdout.log')
os.makedirs(P10T, exist_ok=True)

buf = io.StringIO()
real = sys.stdout


class Tee(object):
    def write(self, s):
        buf.write(s); real.write(s)

    def flush(self):
        real.flush()


sys.stdout = Tee()
rc = 0
try:
    runpy.run_path(os.path.join(HERE, 'n2h1a3_depth_locate.py'), run_name='__main__')
except SystemExit as e:
    rc = int(e.code) if e.code is not None else 0
except BaseException:
    buf.write('\n!!! EXCEPTION !!!\n')
    buf.write(traceback.format_exc())
    rc = 1
finally:
    sys.stdout = real

io.open(LOG, 'w', encoding='utf-8').write(buf.getvalue())
print('mode=%s RC=%d log=%s (%d chars)' % (mode, rc, LOG, len(buf.getvalue())))
sys.exit(rc)
