# -*- coding: utf-8 -*-
"""
Phase 20 正式运行驱动（进程隔离 + SMOKE + PROBE）。
=============================================================================
同一进程内连续加载多个大模型会段错误（Phase 15/18/19 实测）⇒ 每臂一个独立进程。
stdout 由本脚本以 UTF-8 写盘（Windows shim 下 `> log` 可能是 UTF-16）。

用法：python tests/deepseek/Phase20/run_phase20_split.py probe   # 可行性探针（A0 双口径）
      python tests/deepseek/Phase20/run_phase20_split.py smoke   # SMOKE（A0_nf4，小网格）
      python tests/deepseek/Phase20/run_phase20_split.py         # 全四臂 + 合并
      python tests/deepseek/Phase20/run_phase20_split.py A1_bf16 # 只跑某臂
"""
import io
import os
import sys
import time
import subprocess

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P20T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase20')
PY = os.path.join(ROOT, '.venv', 'Scripts', 'python.exe')
SCRIPT = 'tests/deepseek/Phase20/n2h1a13_quant_scheme_robustness.py'
ARMS = ['A0_nf4', 'A0_bf16', 'A1_nf4', 'A1_bf16']

args = sys.argv[1:]
MODE_SMOKE = bool(args) and args[0] == 'smoke'
MODE_PROBE = bool(args) and args[0] == 'probe'
if MODE_SMOKE or MODE_PROBE:
    args = args[1:]


def run(env_extra, logname):
    env = dict(os.environ)
    for k in ('SMOKE', 'MERGE', 'SPLIT_PARTIAL', 'ARMS', 'ELAPSED_TOTAL', 'PROBE'):
        env.pop(k, None)
    env.update(env_extra)
    t0 = time.time()
    p = subprocess.run([PY, SCRIPT], cwd=ROOT, env=env, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    dt = time.time() - t0
    txt = p.stdout.decode('utf-8', errors='replace')
    keep = [ln for ln in txt.splitlines() if 'Loading weights' not in ln]
    out = os.path.join(P20T, logname)
    io.open(out, 'w', encoding='utf-8', newline='\n').write('\n'.join(keep) + '\n')
    print('[%s] exit=%d  %.1fs  -> %s (%d lines)' % (logname, p.returncode, dt, out, len(keep)))
    return p.returncode, dt


rc = 0
if MODE_PROBE:
    _r, _ = run({'PROBE': '1', 'SPLIT_PARTIAL': '1'}, '_probe_stdout.log')
    rc |= _r
    _r, _ = run({'PROBE': '1', 'MERGE': '1'}, '_merge_probe_stdout.log')
    rc |= _r
elif MODE_SMOKE:
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
