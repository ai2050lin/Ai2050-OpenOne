# -*- coding: utf-8 -*-
"""紧凑等待器：轮询 4 个 FULL arm 的 result.json 是否齐全，只打印短进度。"""
import os, sys, time, glob, json
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek\result'
NEED = ['q05_qwen3-4b__bf16_result.json', 'q05_qwen3-4b__nf4_result.json',
        'q05_qwen3-14b__nf4_result.json', 'q05_glm4-9b__nf4_result.json']
maxwait = float(os.environ.get('P_WAIT_MIN', '9')) * 60
t0 = time.time()
while time.time() - t0 < maxwait:
    have = [f for f in NEED if os.path.exists(os.path.join(OUT, f))]
    run = glob.glob(os.path.join(OUT, 'q05_run_*__*.txt'))
    run = sorted(run, key=os.path.getmtime)
    tail = ''
    if run:
        last = os.path.basename(run[-1])
        ls = [l for l in open(os.path.join(OUT, last), encoding='utf-8', errors='replace').read().strip().split('\n') if l]
        tail = '%s :: %s' % (last, ls[-1][:60] if ls else '')
    print('[%5.1fs] done=%d/4  %s | %s' % (time.time() - t0, len(have), ','.join(os.path.basename(h)[4:-12] for h in have), tail), flush=True)
    if len(have) == 4:
        print('ALL4_READY')
        sys.exit(0)
    time.sleep(30)
print('WAIT_TIMEOUT done=%d/4' % len([f for f in NEED if os.path.exists(os.path.join(OUT, f))]))
