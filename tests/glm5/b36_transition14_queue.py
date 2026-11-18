"""Finite admission queue; the scientific runner reacquires the GPU lease."""
import time,sys,subprocess
import b36_transition14 as M
S=M.S
S.configure();start=time.monotonic();ready=False
for attempt in range(121):
    try:
        with S.GpuLease():pass
        ready=True;break
    except RuntimeError as e:
        if 'leased by another' not in str(e):raise
        S.log('gpu_admission_wait',attempt,120,start,status='another process owns GPU lease')
        if attempt<120:time.sleep(60)
if not ready:
    S.save('admission-timeout.json',dict(reason='GPU lease still held after120 minutes; no bypass or termination.',elapsed_seconds=time.monotonic()-start));sys.exit(3)
S.save('admission-resume.json',dict(wait_seconds=time.monotonic()-start,scope='Readiness probe only; frozen runner reacquires the advisory lock.'))
with (S.OUT/'generation.log').open('w',encoding='utf8') as log:
    result=subprocess.run([sys.executable,'-B','-X','utf8','-u',str(S.ROOT/'tests/glm5/b36_transition14_run.py')],stdout=log,stderr=subprocess.STDOUT)
sys.exit(result.returncode)
