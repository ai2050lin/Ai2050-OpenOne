"""Serial, finite all-source follow-up. Actual frozen material and memory guards are mandatory."""
import subprocess
import sys
import psutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'glm5'))
from rdc_prefix_common import *


def main():
    pending=(56640,51572);started=time.monotonic()
    while any(psutil.pid_exists(pid) for pid in pending):
        for pid in pending:
            if psutil.pid_exists(pid):assert 'phase2714_rdc_full_source_history.py' in ' '.join(psutil.Process(pid).cmdline()),'PID identity changed'
        assert time.monotonic()-started<1800;time.sleep(5)
    out=CAMPAIGN/'full_source_history';assert len(list((out/'main/commits').glob('*.json')))==496
    tasks=[('source_kernel_fit',['tests/glm5/phase2714_rdc_source_kernels.py']),
      ('source_fresh_capture',['tests/glm5/phase2714_rdc_full_source_history.py','--fresh']),
      ('source_fresh_frozen_evaluation',['tests/glm5/phase2714_rdc_source_kernels.py','--fresh']),
      ('source_probability',['tests/glm5/phase2714_rdc_source_probability.py'])]
    records=[]
    for name,args in tasks:
        guard(12*1024**2);status('source_serial_tail',state='running',task=name);begin=time.monotonic()
        so=CAMPAIGN/f'jobs/{name}.stdout.log';se=CAMPAIGN/f'jobs/{name}.stderr.log'
        with so.open('w',encoding='utf-8') as f,se.open('w',encoding='utf-8') as g:
            result=subprocess.run([sys.executable,'-X','utf8',*args],cwd=ROOT,stdout=f,stderr=g,timeout=1800)
        records.append({'task':name,'command':args,'exit_code':result.returncode,'seconds':time.monotonic()-begin})
        save(out/'serial_jobs.json',records);assert result.returncode==0,(name,result.returncode)
    status('source_serial_tail',state='complete',tasks=4);print('SOURCE_SERIAL_TAIL_COMPLETE',flush=True)


if __name__=='__main__':main()
