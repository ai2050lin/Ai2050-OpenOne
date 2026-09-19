"""Finite serial queue: wait verified Q14, GLM pilot/main, CPU L analysis, then M pilot only."""
import sys,subprocess,time,os
from pathlib import Path
import psutil
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from rdc_conditional_common import *


def run(name,args):
    path=ROOT/'tests/glm5_temp';start=time.monotonic()
    with open(path/(name+'.stdout.log'),'a',encoding='utf-8') as stdout,open(path/(name+'.stderr.log'),'a',encoding='utf-8') as stderr:
        process=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u',*args],cwd=ROOT,stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
        save(CAMPAIGN/f'jobs/{name}.json',{'pid':process.pid,'args':args,'started_at':stamp(),'state':'running'})
        code=process.wait()
    save(CAMPAIGN/f'jobs/{name}.json',{'pid':process.pid,'args':args,'finished_at':stamp(),'exit_code':code,'elapsed_seconds':time.monotonic()-start,'state':'complete' if code==0 else 'failed'})
    assert code==0,(name,code)


def main():
    targets=[]
    for p in psutil.process_iter(['pid','name','cmdline','create_time']):
        cmd=' '.join(p.info['cmdline'] or [])
        if 'python' in (p.info['name'] or '').lower() and 'phase2706_rdc_content_aligned_capture.py' in cmd and 'qwen14' in cmd and '256' in cmd:targets.append(p)
    assert targets or read(CAMPAIGN/'l_aligned_qwen14/status.json')['completed']==256
    save(CAMPAIGN/'jobs/finite_serial_queue.json',{'state':'waiting_qwen14','timestamp':stamp(),'verified_pids':[p.pid for p in targets],
      'queue':['glm4pilot4','glm4main256 ifpilotpassed','alignedCPUanalysis','orderpilot36 only'],'no_automatic_order_expansion':True})
    psutil.wait_procs(targets,timeout=7200)
    assert all(not p.is_running() for p in targets),'Q14 process exceeded finite wait boundary'
    status=read(CAMPAIGN/'l_aligned_qwen14/status.json');assert status['state']=='captured' and status['completed']==256,status
    print('Q14_FINISHED',stamp(),flush=True)
    run('phase2706_glm_pilot',['tests/glm5/phase2706_rdc_content_aligned_capture.py','glm4','--limit','4'])
    out=CAMPAIGN/'l_aligned/glm4';cost=read(out/'capture_4.json');rows=read(out/'prefixes.json')[:4]
    assert cost['mean_forward_seconds']*256<5400
    assert all(read(out/f'behavior/{r["sample_id"]}.json')['native_noop'] for r in rows)
    print('GLM_PILOT_PASS',cost['mean_forward_seconds'],flush=True)
    run('phase2706_glm_main',['tests/glm5/phase2706_rdc_content_aligned_capture.py','glm4','--limit','256'])
    run('phase2706_aligned_analysis',['tests/glm5/phase2706_rdc_content_aligned_analysis.py'])
    run('phase2707_order_pilot',['tests/glm5/phase2707_rdc_output_order_capture.py','--limit','36'])
    save(CAMPAIGN/'jobs/finite_serial_queue.json',{'state':'complete','timestamp':stamp(),'next':'Inspect order pilot behavior/resources before full288'})
    print('FINITE_QUEUE_COMPLETE',stamp(),flush=True)


if __name__=='__main__':main()
