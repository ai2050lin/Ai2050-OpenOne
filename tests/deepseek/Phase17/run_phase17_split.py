# -*- coding: utf-8 -*-
"""
Phase 17 正式运行驱动（进程隔离版 + SMOKE）。
=============================================================================
同一进程内连续加载 3 个 nf4 大模型会在第 3 个加载中途段错误（Phase 15 实测 EXIT=139）。
对策：每臂一个独立进程（`ARMS=<arm> SPLIT_PARTIAL=1` -> `_armrec17_<arm>.json`），
再用 `MERGE=1` 在一个**不加载任何模型**的进程里组装 `result_phase17.json`。

stdout 由本脚本以 **UTF-8** 写盘（Windows shim 下 `> log` 可能是 UTF-16）。
用法：python tests/deepseek/Phase17/run_phase17_split.py smoke     # SMOKE（A0，小网格）
      python tests/deepseek/Phase17/run_phase17_split.py           # 全三臂 + 合并
      python tests/deepseek/Phase17/run_phase17_split.py A2_qwen3-14b-nf4   # 只跑某臂
"""
import io
import os
import sys
import time
import subprocess

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P17T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase17')
PY = os.path.join(ROOT, '.venv', 'Scripts', 'python.exe')
SCRIPT = 'tests/deepseek/Phase17/n2h1a10_writevec_centroid.py'
ARMS = ['A0_calib_qwen3-4b-nf4', 'A1_glm4-9b-nf4', 'A2_qwen3-14b-nf4']

args = sys.argv[1:]
MODE_SMOKE = bool(args) and args[0] == 'smoke'
if MODE_SMOKE:
    args = args[1:]


def run(env_extra, logname):
    env = dict(os.environ)
    env.pop('SMOKE', None)
    env.pop('MERGE', None)
    env.pop('SPLIT_PARTIAL', None)
    env.pop('ARMS', None)
    env.update(env_extra)
    t0 = time.time()
    p = subprocess.run([PY, SCRIPT], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    dt = time.time() - t0
    txt = p.stdout.decode('utf-8', errors='replace')
    keep = [ln for ln in txt.splitlines() if 'Loading weights' not in ln]
    out = os.path.join(P17T, logname)
    io.open(out, 'w', encoding='utf-8', newline='\n').write('\n'.join(keep) + '\n')
    print('[%s] exit=%d  %.1fs  -> %s (%d lines)' % (logname, p.returncode, dt, out, len(keep)))
    return p.returncode


rc = 0
if MODE_SMOKE:
    rc |= run({'SMOKE': '1', 'ARMS': 'A0'}, '_smoke_stdout.log')
    rc |= run({'SMOKE': '1', 'MERGE': '1'}, '_merge_smoke_stdout.log')
else:
    only = args if args else None
    todo = ([a for a in ARMS if a in only] if only else list(ARMS))
    for a in todo:
        rc |= run({'ARMS': a, 'SPLIT_PARTIAL': '1'}, '_run_%s_stdout.log' % a)
    if not only:
        rc |= run({'MERGE': '1'}, '_merge_stdout.log')

print('ALL DONE rc=%d' % rc)
