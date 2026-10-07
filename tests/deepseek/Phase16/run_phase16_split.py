# -*- coding: utf-8 -*-
"""
Phase 16 正式运行驱动（进程隔离版）。
=============================================================================
同一进程内连续加载 3 个 nf4 大模型会在第 3 个加载中途段错误（Phase 15 实测 EXIT=139）。
对策：每臂一个独立进程（`ARMS=<arm> SPLIT_PARTIAL=1` -> `_armrec16_<arm>.json`），
再用 `MERGE=1` 在一个**不加载任何模型**的进程里组装 `result_phase16.json`。

stdout 由本脚本以 **UTF-8** 写盘（Windows shim 下 `> log` 可能是 UTF-16）。
用法：python tests/deepseek/Phase16/run_phase16_split.py               # 全三臂 + 合并
      python tests/deepseek/Phase16/run_phase16_split.py A2_qwen3-14b-nf4  # 只跑某臂
"""
import io
import os
import sys
import time
import subprocess

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P16T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase16')
PY = os.path.join(ROOT, '.venv', 'Scripts', 'python.exe')
SCRIPT = 'tests/deepseek/Phase16/n2h1a9_writewin_origin_profile.py'
ARMS = ['A0_calib_qwen3-4b-nf4', 'A1_glm4-9b-nf4', 'A2_qwen3-14b-nf4']

only = sys.argv[1:] if len(sys.argv) > 1 else None
todo = ([a for a in ARMS if a in only] if only else list(ARMS))


def run(env_extra, logname):
    env = dict(os.environ)
    env.pop('SMOKE', None)
    env.update(env_extra)
    t0 = time.time()
    p = subprocess.run([PY, SCRIPT], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    dt = time.time() - t0
    txt = p.stdout.decode('utf-8', errors='replace')
    keep = [ln for ln in txt.splitlines() if 'Loading weights' not in ln]
    out = os.path.join(P16T, logname)
    io.open(out, 'w', encoding='utf-8', newline='\n').write('\n'.join(keep) + '\n')
    print('[%s] exit=%d  %.1fs  -> %s (%d lines)' % (logname, p.returncode, dt, out, len(keep)))
    return p.returncode


rc = 0
for a in todo:
    rc |= run({'ARMS': a, 'SPLIT_PARTIAL': '1'}, '_run_%s_stdout.log' % a)

if not only:
    rc |= run({'MERGE': '1'}, '_merge_stdout.log')

print('ALL DONE rc=%d' % rc)
