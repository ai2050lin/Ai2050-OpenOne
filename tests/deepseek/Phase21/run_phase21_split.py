# -*- coding: utf-8 -*-
"""
Phase 21 正式运行驱动（进程隔离 + SMOKE）。
同一进程连续加载多个大模型会段错误（P15/18/19 实测）⇒ 每臂一个独立进程。
stdout 以 UTF-8 由本脚本写盘（Windows shim 下 `> log` 可能是 UTF-16）。

用法：python tests/deepseek/Phase21/run_phase21_split.py smoke        # A0_nf4 小面板
      python tests/deepseek/Phase21/run_phase21_split.py               # 全四臂 + 合并
      python tests/deepseek/Phase21/run_phase21_split.py A1_bf16       # 只跑某臂
      NOFX=1 python tests/deepseek/Phase21/run_phase21_split.py        # 跳过效应侧（更快）
"""
import io
import os
import sys
import time
import subprocess

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P21T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase21')
PY = os.path.join(ROOT, '.venv', 'Scripts', 'python.exe')
SCRIPT = 'tests/deepseek/Phase21/n2h1a14_component_budget_precision.py'
ARMS = ['A0_nf4', 'A0_bf16', 'A1_nf4', 'A1_bf16']
NOFX = os.environ.get('NOFX', '0') == '1'

args = sys.argv[1:]
MODE_SMOKE = bool(args) and args[0] == 'smoke'
if MODE_SMOKE:
    args = args[1:]


def run(env_extra, logname):
    env = dict(os.environ)
    for k in ('SMOKE', 'MERGE', 'SPLIT_PARTIAL', 'ARMS'):
        env.pop(k, None)
    env.update(env_extra)
    t0 = time.time()
    p = subprocess.run([PY, SCRIPT], cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    txt = p.stdout.decode('utf-8', errors='replace')
    keep = [ln for ln in txt.splitlines() if 'Loading weights' not in ln]
    out = os.path.join(P21T, logname)
    io.open(out, 'w', encoding='utf-8', newline='\n').write('\n'.join(keep) + '\n')
    print('[%s] exit=%d  %.1fs  -> %s (%d lines)' % (logname, p.returncode, dt, out, len(keep)))
    return p.returncode, dt


rc = 0
if MODE_SMOKE:
    _r, _ = run({'SMOKE': '1', 'ARMS': 'A0_nf4', 'SPLIT_PARTIAL': '1'}, '_smoke_stdout.log')
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
        _r, _dt = run({'MERGE': '1'}, '_merge_stdout.log')
        rc |= _r
        total += _dt
    print('四臂合计 %.1fs' % total)

print('ALL DONE rc=%d' % rc)
