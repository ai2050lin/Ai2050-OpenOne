# -*- coding: utf-8 -*-
"""Runner for phase3087 A2 full arbitration
(smoke/full). Usage:
python p3087_run_full.py smoke|full <script_name>
Status + stdout + traceback -> file."""
import io
import os
import sys
import time
import traceback

MODE = sys.argv[1]
SNAME = sys.argv[2]
ROOT = r'D:\AI2050\Ai2050-OpenOne'
SCRIPT = os.path.join(ROOT, 'tests', 'glm5',
                      SNAME)
if MODE == 'smoke':
    os.environ['SMOKE'] = '1'
    STATUS = os.path.join(
        ROOT, 'tests', 'gpt5_temp',
        'p3087_arb_smoke_run.txt')
else:
    os.environ['SMOKE'] = '0'
    STATUS = os.path.join(
        ROOT, 'tests', 'gpt5_temp',
        'p3087_arb_full_run.txt')

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
    f.write('MODE=%s script=%s elapsed=%.1fs '
            'ok=%s\n' % (MODE, SNAME, el, ok))
    if ok:
        f.write(buf.getvalue())
    else:
        f.write(err)
print('RUNNER_DONE %s ok=%s %.1fs'
      % (MODE, ok, el))
