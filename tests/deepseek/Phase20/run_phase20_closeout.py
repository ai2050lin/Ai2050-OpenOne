# -*- coding: utf-8 -*-
"""Phase 20 收尾链顺序驱动（fail-fast）。

顺序（严格；每步 stdout 以 UTF-8 落盘到 tests/deepseek_temp/Phase20/）：
  1. publish_probe_phase20.py      探针件发布到 seal 声明路径（幂等）
  2. closeout_phase20.py           Ledger 补登 + 预追加基线快照
  3. gen_memo_phase20.py           MEMO 追加节生成（数据驱动）
  4. do_append_phase20.py          MEMO 追加 + 落盘自检
  5. closeout_docs_phase20.py      wlog 追加 + _infra 基线刷新（含 drift_events / stale）
  6. disk_verify_phase20.py        独立磁盘复核（要求 0 FAIL）
  7. gen_present_phase20.py        展示页渲染

用法：python tests/deepseek/Phase20/run_phase20_closeout.py
      python tests/deepseek/Phase20/run_phase20_closeout.py 4    # 从第 4 步开始（幂等重跑用）
"""
import io
import os
import sys
import time
import subprocess

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P20T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase20')
PY = os.path.join(ROOT, '.venv', 'Scripts', 'python.exe')
D = os.path.join(ROOT, 'tests', 'deepseek', 'Phase20')

STEPS = [
    ('publish', 'publish_probe_phase20.py', '_closeout_1_publish.log'),
    ('ledger ', 'closeout_phase20.py', '_closeout_2_ledger.log'),
    ('memoGen', 'gen_memo_phase20.py', '_closeout_3_memogen.log'),
    ('append ', 'do_append_phase20.py', '_closeout_4_append.log'),
    ('docs   ', 'closeout_docs_phase20.py', '_closeout_5_docs.log'),
    ('verify ', 'disk_verify_phase20.py', '_closeout_6_verify.log'),
    ('present', 'gen_present_phase20.py', '_closeout_7_present.log'),
]

start = 1
args = sys.argv[1:]
if args and args[0].isdigit():
    start = int(args[0])

summary = []
overall = 0
for i, (tag, script, logname) in enumerate(STEPS, 1):
    if i < start:
        summary.append((i, tag, script, 'SKIP'))
        continue
    t0 = time.time()
    p = subprocess.run([PY, os.path.join('tests', 'deepseek', 'Phase20', script)],
                       cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    txt = p.stdout.decode('utf-8', errors='replace')
    io.open(os.path.join(P20T, logname), 'w', encoding='utf-8', newline='\n').write(txt)
    st = 'OK' if p.returncode == 0 else 'FAIL(%d)' % p.returncode
    summary.append((i, tag, script, st))
    print('[%d/7] %-8s %-28s %-9s %6.1fs' % (i, tag, script, st, dt))
    if p.returncode != 0:
        print('---- 末尾 40 行 ----')
        for ln in txt.splitlines()[-40:]:
            print('   ' + ln)
        overall = 1
        break

print()
print('==== 收尾链汇总 ====')
for i, tag, script, st in summary:
    print('  %d. %-28s %s' % (i, script, st))
print('OVERALL rc=%d' % overall)
