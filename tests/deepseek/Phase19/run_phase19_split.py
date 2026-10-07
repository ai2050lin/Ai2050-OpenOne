# -*- coding: utf-8 -*-
"""
Phase 19 正式运行驱动（进程隔离版 + SMOKE）。
=============================================================================
同一进程内连续加载多个大模型会段错误（Phase 15/18/19 实测：A2-bf16 在加载期 segfault）。
对策：每臂一个独立进程（`ARMS=<arm> SPLIT_PARTIAL=1` -> `_armrec19_<arm>.json`），
再用 `MERGE=1` 在一个**不加载任何模型**的进程里组装 `result_phase19.json`。
MERGE 时把四臂总耗时通过 ELAPSED_TOTAL 注入。

stdout 由本脚本以 UTF-8 写盘（Windows shim 下 `> log` 可能是 UTF-16）。
用法：python tests/deepseek/Phase19/run_phase19_split.py smoke      # SMOKE（A0_nf4，小网格）
      python tests/deepseek/Phase19/run_phase19_split.py            # 全四臂 + 合并
      python tests/deepseek/Phase19/run_phase19_split.py A1_bf16    # 只跑某臂
"""
import io
import os
import sys
import time
import subprocess

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P19T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase19')
PY = os.path.join(ROOT, '.venv', 'Scripts', 'python.exe')
SCRIPT = 'tests/deepseek/Phase19/n2h1a12_quant_scheme_robustness.py'
ARMS = ['A0_nf4', 'A0_bf16', 'A1_nf4', 'A1_bf16']

args = sys.argv[1:]
MODE_SMOKE = bool(args) and args[0] == 'smoke'
if MODE_SMOKE:
    args = args[1:]


def run(env_extra, logname):
    env = dict(os.environ)
    for k in ('SMOKE', 'MERGE', 'SPLIT_PARTIAL', 'ARMS', 'ELAPSED_TOTAL'):
        env.pop(k, None)
    env.update(env_extra)
    t0 = time.time()
    p = subprocess.run([PY, SCRIPT], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    dt = time.time() - t0
    txt = p.stdout.decode('utf-8', errors='replace')
    keep = [ln for ln in txt.splitlines() if 'Loading weights' not in ln]
    out = os.path.join(P19T, logname)
    io.open(out, 'w', encoding='utf-8', newline='\n').write('\n'.join(keep) + '\n')
    print('[%s] exit=%d  %.1fs  -> %s (%d lines)' % (logname, p.returncode, dt, out, len(keep)))
    return p.returncode, dt


rc = 0
if MODE_SMOKE:
    _r, _ = run({'SMOKE': '1', 'ARMS': 'A0_nf4', 'SPLIT_PARTIAL': '1'}, '_smoke_stdout.log')
    rc |= _r
    _r, _ = run({'SMOKE': '1', 'MERGE': '1'}, '_merge_smoke_stdout.log')
    rc |= _r
else:
    only = args if args else None
    todo = ([a for a in ARMS if a in only] if only else list(ARMS))
    total = 0.0
    for a in todo:
        _r, _dt = run({'ARMS': a, 'SPLIT_PARTIAL': '1'}, '_run_%s_stdout.log' % a)
        rc |= _r
        total += _dt
    if not only:
        _r, _dt = run({'MERGE': '1', 'ELAPSED_TOTAL': '%.1f' % total}, '_merge_stdout.log')
        rc |= _r
        total += _dt
    print('四臂合计 %.1fs' % total)

print('ALL DONE rc=%d' % rc)
