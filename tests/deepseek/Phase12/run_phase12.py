# -*- coding: utf-8 -*-
"""
Phase 12 运行包装器。
用法：
  python run_phase12.py smoke    -> SMOKE=1（面板/网格/位点截断，产物落 smoke/）
  python run_phase12.py formal   -> 正式运行

本包装器自己写 UTF-8 日志（避免 Windows 重定向生成 UTF-16），并把子脚本的
stdout 同时透传到控制台。
"""
import os, sys, io, runpy, time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
HERE = os.path.join(ROOT, 'tests', 'deepseek', 'Phase12')
TEMP = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase12')
SCRIPT = os.path.join(HERE, 'n2h1a5_swap_alloc.py')

mode = (sys.argv[1] if len(sys.argv) > 1 else 'smoke').lower()
assert mode in ('smoke', 'formal'), 'mode 必须是 smoke|formal，得到 %r' % mode
if mode == 'smoke':
    os.environ['SMOKE'] = '1'
    log = os.path.join(TEMP, '_smoke_stdout.log')
else:
    os.environ.pop('SMOKE', None)
    log = os.path.join(TEMP, '_formal_stdout.log')

os.makedirs(TEMP, exist_ok=True)


class Tee(object):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
            except Exception:
                pass

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


t0 = time.time()
with io.open(log, 'w', encoding='utf-8') as f:
    old = sys.stdout
    sys.stdout = Tee(old, f)
    try:
        runpy.run_path(SCRIPT, run_name='__main__')
    finally:
        sys.stdout = old

print('[run12] mode=%s elapsed=%.1fs log=%s' % (mode, time.time() - t0, log))
print('[run12] log bytes=%d' % os.path.getsize(log))
