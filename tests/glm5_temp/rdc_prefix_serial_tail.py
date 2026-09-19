"""Finite serial continuation of the explicitly scoped prefix campaign; one CUDA process at a time."""
import subprocess
import sys
import time
from pathlib import Path
import psutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'glm5'))
from rdc_prefix_common import *


def main():
    pending={23168:'phase2711_rdc_prefix_capture.py',33488:'phase2711_rdc_prefix_capture.py'}
    started=time.monotonic()
    while any(psutil.pid_exists(pid) for pid in pending):
        for pid,expected in pending.items():
            if psutil.pid_exists(pid):
                cmd=' '.join(psutil.Process(pid).cmdline())
                assert expected in cmd and '--model qwen14 --limit 64' in cmd,(pid,'Process identity changed')
        assert time.monotonic()-started<5400
        status('serial_tail',state='waiting_qwen14',watched_processes=list(pending));time.sleep(5)
    assert read(CAMPAIGN/'qwen14/status.json')['state']=='captured'
    assert len(list((CAMPAIGN/'qwen14/commits').glob('*.json')))==64
    tasks=[('glm4_pilot',['tests/glm5/phase2711_rdc_prefix_capture.py','--model','glm4','--limit','4']),
      ('glm4_main',['tests/glm5/phase2711_rdc_prefix_capture.py','--model','glm4','--limit','64']),
      ('causal_control_probability',['tests/glm5/phase2712_rdc_prefix_native_probability.py','--causal-controls']),
      ('causal_control_confirmation_probability',['tests/glm5/phase2712_rdc_prefix_native_probability.py','--confirmation','--causal-controls'])]
    records=[]
    for name,args in tasks:
        guard(40*1024**2)
        if name=='glm4_main':
            pilot=read(CAMPAIGN/'glm4/pilot_audit.json');assert pilot['passed']
            existing=sum(p.stat().st_size for p in (CAMPAIGN/'glm4').rglob('*') if p.is_file())
            guard(max(0,pilot['projected_this_run_bytes']-existing)+40*1024**2)
        out=CAMPAIGN/f'jobs/{name}.stdout.log';err=CAMPAIGN/f'jobs/{name}.stderr.log';out.parent.mkdir(parents=True,exist_ok=True)
        status('serial_tail',state='running',task=name)
        begin=time.monotonic()
        with out.open('w',encoding='utf-8') as so,err.open('w',encoding='utf-8') as se:
            result=subprocess.run([sys.executable,'-X','utf8',*args],cwd=ROOT,stdout=so,stderr=se,timeout=5400)
        records.append({'task':name,'command':args,'exit_code':result.returncode,'seconds':time.monotonic()-begin})
        save(CAMPAIGN/'serial_jobs.json',records)
        assert result.returncode==0,(name,result.returncode,'See saved task logs')
    status('serial_tail',state='complete',tasks=len(tasks));print('PREFIX_SERIAL_TAIL_COMPLETE',flush=True)


if __name__=='__main__':main()
