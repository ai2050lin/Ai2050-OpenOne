# -*- coding: utf-8 -*-
"""Runner for phase3087 A1 layer scan (smoke/full).
Usage: python p3087_run.py smoke|full
Status + stdout + traceback -> file."""
import io
import os
import sys
import time
import traceback

MODE = sys.argv[1]
ROOT = r'D:\AI2050\Ai2050-OpenOne'
SCRIPT = os.path.join(
    ROOT, 'tests', 'glm5',
    'phase3087_omega_p84_glm4_layer_scan.py')
OUTDIR = os.path.join(
    ROOT, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913',
    'phase3087',
    'omega_p84_glm4_layer_scan')
if MODE == 'smoke':
    os.environ['SMOKE'] = '1'
    OUTDIR = os.path.join(OUTDIR, 'smoke')
    STATUS = os.path.join(
        ROOT, 'tests', 'gpt5_temp',
        'p3087_scan_smoke_run.txt')
else:
    os.environ['SMOKE'] = '0'
    STATUS = os.path.join(
        ROOT, 'tests', 'gpt5_temp',
        'p3087_scan_full_run.txt')

g = {'__name__': '__main__',
     '__file__': SCRIPT}
t0 = time.time()
buf = io.StringIO()
old = sys.stdout
sys.stdout = buf
ok = False
err = ''
try:
    src = io.open(SCRIPT,
                  encoding='utf-8').read()
    exec(compile(src, SCRIPT, 'exec'), g)
    ok = True
except Exception:
    err = traceback.format_exc()
finally:
    sys.stdout = old
el = time.time() - t0
with io.open(STATUS, 'w',
             encoding='utf-8') as f:
    f.write('MODE=%s elapsed=%.1fs ok=%s\n'
            % (MODE, el, ok))
    if ok:
        f.write(buf.getvalue())
    else:
        f.write(err)
print('RUNNER_DONE %s ok=%s %.1fs'
      % (MODE, ok, el))
