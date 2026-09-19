"""Finite local queue: wait existing capture, CPU analysis, serial GPU audit and long pilot."""
import os,sys,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/glm5'))
from rdc_conditional_common import CAMPAIGN,read,save,stamp


def launch(name,script,args=()):
    stdout=(ROOT/f'tests/glm5_temp/{name}.stdout.log').open('w',encoding='utf-8')
    stderr=(ROOT/f'tests/glm5_temp/{name}.stderr.log').open('w',encoding='utf-8')
    p=subprocess.Popen([sys.executable,str(ROOT/script),*args],cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
    stdout.close();stderr.close()
    save(CAMPAIGN/f'jobs/{name}.json',{'timestamp':stamp(),'pid':p.pid,'script':script,'args':list(args)})
    return p


def main():
    import psutil
    started=time.monotonic();cap=CAMPAIGN/'i_factorial/status.json'
    while True:
        if (CAMPAIGN/'pause_requested').exists():return
        if cap.exists() and read(cap).get('state')=='captured':break
        if time.monotonic()-started>3600:raise RuntimeError('Capture wait budget exceeded')
        time.sleep(3)
    # The status is written just before main returns; wait until this exact script's process exits.
    while any('phase2703_rdc_factorial_capture.py' in ' '.join(p.info.get('cmdline') or []) for p in psutil.process_iter(['cmdline']) if p.pid!=os.getpid()):time.sleep(1)
    analysis=launch('phase2703_analysis','tests/glm5/phase2703_rdc_factorial_analysis.py')
    operator=launch('phase2703_operator','tests/glm5/phase2703_rdc_operator_audit.py')
    assert operator.wait()==0,'Numerical operator audit failed; long pilot not launched'
    if (CAMPAIGN/'pause_requested').exists():return
    pilot=launch('phase2705_pilot','tests/glm5/phase2705_rdc_long_capture.py',('--limit','16'))
    assert pilot.wait()==0,'Long pilot failed'
    assert analysis.wait()==0,'Factorial analysis failed'
    save(CAMPAIGN/'jobs/first_tail_complete.json',{'timestamp':stamp(),'state':'finite_queue_complete','next':'Review long pilot; append2703; execute2704 and remaining verified stages.'})


if __name__=='__main__':main()
