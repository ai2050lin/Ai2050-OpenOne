"""Bounded serial continuation; waits only for current owned Q14 job, never starts two models."""
import sys,time,subprocess,json
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from rdc_continuity_common import CAMPAIGN,save,read,stamp
python=ROOT/'.venv/Scripts/python.exe'

def run(script,args,stem):
    with (ROOT/f'tests/glm5_temp/{stem}.stdout.log').open('w',encoding='utf8') as out,(ROOT/f'tests/glm5_temp/{stem}.stderr.log').open('w',encoding='utf8') as err:
        completed=subprocess.run([str(python),str(ROOT/'tests/glm5'/script),*args],cwd=ROOT,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
    if completed.returncode:raise RuntimeError(f'{script} failed: exit {completed.returncode}; see {stem}.stderr.log')

def main():
    started=time.monotonic();save(CAMPAIGN/'serial_tail.json',{'state':'waiting_for_owned_qwen14','started':stamp(),'remaining_plan':['glm4 pilot/full if resource gate','scale analysis'],'not_an_unlimited_queue':True})
    while True:
        active=[]
        for p in psutil.process_iter(['pid','name','cmdline']):
            cmd=' '.join(p.info.get('cmdline') or [])
            if 'python' in (p.info.get('name') or '').lower() and 'phase2702_rdc_serial_scale.py' in cmd and 'qwen14' in cmd:active.append(p.pid)
        if not active:break
        if time.monotonic()-started>3600:raise RuntimeError('Q14 exceeded bounded wait; no nextmodel started')
        time.sleep(5)
    q=CAMPAIGN/'h_scale/qwen14';assert len(list((q/'commits').glob('*.json')))==256,'Q14 incomplete: do not labelcomplete or silently continue'
    save(CAMPAIGN/'serial_tail.json',{'state':'glm4_pilot','updated_at':stamp()})
    run('phase2702_rdc_serial_scale.py',['glm4','--limit','4'],'phase2702_glm_pilot')
    t=read(CAMPAIGN/'h_scale/glm4/capture_4.json')['mean_seconds'];estimate=256*t
    assert estimate<=3600,f'GLM4 conservative estimate {estimate} exceeds protocol budget'
    save(CAMPAIGN/'scale_resource_decision.json',{'glm4_pilot_seconds':t,'conservative_256_estimate':estimate,'decision':'complete256, no quantization or parallel model','updated_at':stamp()})
    save(CAMPAIGN/'serial_tail.json',{'state':'glm4_full','updated_at':stamp()})
    run('phase2702_rdc_serial_scale.py',['glm4','--limit','256'],'phase2702_glm_capture')
    save(CAMPAIGN/'serial_tail.json',{'state':'scale_analysis','updated_at':stamp()})
    run('phase2702_rdc_scale_analysis.py',[],'phase2702_scale_analysis')
    save(CAMPAIGN/'serial_tail.json',{'state':'complete','updated_at':stamp(),'elapsed_seconds':time.monotonic()-started,'scientific_goal_solved':False})

if __name__=='__main__':
    try:main()
    except BaseException as e:save(CAMPAIGN/'serial_tail.json',{'state':'failed','error':repr(e),'updated_at':stamp()});raise
