# -*- coding: utf-8 -*-
"""Phase 3093 A2 chain orchestrator.
Precondition check (A1 complete) ->
run p3093_patch_a2.py on real A1 data ->
run the generated A2 script in smoke mode
(SMOKE=1) -> verify smoke anchors ->
STOP for human confirmation before the
authoritative run (project discipline: smoke
anchors must be checked by hand; the chain
never launches the ~3.8h authoritative run
by itself).
Exit codes: 0 CHAIN_SMOKE_OK; 1 CHAIN_FAIL;
3 CHAIN_A1_<verdict> (A2 gate refused -
re-decide design; closeout single-arm branch
ready)."""
import io
import json
import os
import re
import subprocess
import sys

ROOT = r'D:\AI2050\Ai2050-OpenOne'
PY = ROOT + r'\.venv\Scripts\python.exe'
TMP = ROOT + r'\tests\gpt5_temp'
A1STD = TMP + r'\p3093_run_stdout.txt'
A1RES = (ROOT + r'\tests\glm5\result'
         r'\rdc_query_construction_20260913'
         r'\phase3093\omega_p90_qwen14b_layer_'
         r'scan\result.json')
PATCH = TMP + r'\p3093_patch_a2.py'
CHAINLOG = TMP + r'\p3093_a2_chain_log.txt'
o = []


def flush():
    io.open(CHAINLOG, 'w',
            encoding='utf-8').write(
        '\n'.join(o) + '\n')


def fail(tag, detail=''):
    o.append('CHAIN_FAIL %s %s'
             % (tag, detail))
    flush()
    print('CHAIN_FAIL %s' % tag)
    sys.exit(1)


# ---- 1. A1 complete? ----
if not os.path.exists(A1RES):
    fail('A1_NOT_DONE', 'no result.json')
std = io.open(A1STD,
              encoding='utf-8').read()
if 'RUN_COMPLETE' not in std:
    fail('A1_NOT_DONE',
         'no RUN_COMPLETE in stdout')
a1 = json.load(io.open(A1RES,
                       encoding='utf-8'))
v1 = a1['verdict']
o.append('A1 verdict=%s' % v1)
if v1 != 'layer_rescue':
    o.append('A2 gate refused (design '
             're-decision needed); closeout '
             'single-arm branch is ready')
    flush()
    print('CHAIN_A1_%s' % v1)
    sys.exit(3)

# ---- 2. patch (real A1 data) ----
try:
    pr = subprocess.run(
        [PY, PATCH], capture_output=True,
        text=True, timeout=300)
except subprocess.TimeoutExpired:
    fail('PATCH_TIMEOUT')
out = (pr.stdout or '') + (pr.stderr or '')
io.open(TMP + r'\p3093_a2_chain_patch.txt',
        'w', encoding='utf-8').write(
    out[-4000:])
m = re.search(r'PATCH_A2_OK L=(\d+)', out)
if pr.returncode != 0 or not m:
    fail('PATCH', out[-2000:])
lb = int(m.group(1))
o.append('patch ok L=%d' % lb)
a2py = (ROOT + r'\tests\glm5'
        r'\phase3093_omega_p91_qwen14b_l%d_'
        r'full_arbitration.py' % lb)
if not os.path.exists(a2py):
    fail('PATCH_DST_MISSING', a2py)

# ---- 3. smoke ----
env = dict(os.environ)
env['SMOKE'] = '1'
so = TMP + r'\p3093_a2_smoke_stdout.txt'
se = TMP + r'\p3093_a2_smoke_stderr.txt'
try:
    with io.open(so, 'w',
                 encoding='utf-8') as f1, \
         io.open(se, 'w',
                 encoding='utf-8') as f2:
        sr = subprocess.run(
            [PY, a2py], stdout=f1,
            stderr=f2, env=env,
            timeout=3600)
except subprocess.TimeoutExpired:
    fail('SMOKE_TIMEOUT')
std2 = io.open(so,
               encoding='utf-8').read()
if sr.returncode != 0 or (
        'RUN_COMPLETE smoke_pending'
        not in std2):
    err = io.open(se,
                  encoding='utf-8').read()
    fail('SMOKE',
         (std2[-1200:] + err[-1200:]))

# ---- 4. smoke anchors ----
slog = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3093\omega_p91_qwen14b_l%d_'
        r'full_arbitration\smoke\run_log.txt'
        % lb)
lg = io.open(slog, encoding='utf-8').read()
ok_line = [l for l in lg.splitlines()
           if 'setup_ok_all' in l]
o.append('smoke setup line: %s'
         % (ok_line[-1] if ok_line
            else 'NONE'))
if not ok_line or ('setup_ok_all=True'
                   not in ok_line[-1]):
    fail('SMOKE_ANCHOR',
         'setup_ok_all not True')
o.append('smoke run_log %d lines; manual '
         'anchor check required before '
         'authoritative run'
         % len(lg.splitlines()))
o.append('NEXT: verify smoke anchors, then '
         'launch authoritative: %s' % a2py)
flush()
print('CHAIN_SMOKE_OK L=%d' % lb)
