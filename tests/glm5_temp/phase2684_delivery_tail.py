"""Continue actual CPU audits/publication after every sequential model exits.

Does not claim browser verification or clean raw fields automatically. The
active research task performs real browser QA before those terminal actions.
"""
import json,os,socket,subprocess,time,urllib.request
from datetime import datetime
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parents[2];PY=ROOT/'.venv/Scripts/python.exe';RESULT=ROOT/'tests/glm5/result';OUT=RESULT/'phase2684_source_campaign_delivery'


def status(**data):
    data['time']=datetime.now().astimezone().isoformat();path=OUT/'analysis/delivery_tail.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def upstream():
    names=('/phase2683_crossmodel_function_atlas.py','/phase2684_sequential_continuation.py')
    return [p.pid for p in psutil.process_iter(['name','cmdline']) if (p.info['name'] or '').lower()=='python.exe'
            and any(str(s).replace('\\','/').endswith(names) for s in (p.info['cmdline'] or []))]


def checked(script,*args):
    log=OUT/'runtime'/('tail_'+Path(script).stem+('_'+args[0].replace('--','') if args else '')+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    status(stage='actual_CPU_step',script=script,args=args,log=str(log))
    with log.open('ab') as stream:
        p=subprocess.Popen([str(PY),'-u',script,*args],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        code=p.wait()
    if code:status(stage='failed_CPU_step',script=script,args=args,exit_code=code,log=str(log));raise SystemExit(code)


def port_used():
    try:
        with socket.create_connection(('127.0.0.1',5001),timeout=1):return True
    except OSError:return False


def main():
    status(stage='waiting_for_all4_actual_protocol_runs',upstream_pids=upstream())
    while upstream():time.sleep(10)
    final=RESULT/'phase2683_crossmodel_function_atlas/analysis/final.json'
    if not final.exists() or not json.loads(final.read_text(encoding='utf-8'))['all_checks_passed']:
        status(stage='upstream_not_completed_no_publication_or_cleanup',model_started=False);raise SystemExit(1)
    checked('tests/glm5_temp/phase2684_scientific_checks.py')
    checked('tests/glm5/phase2684_source_campaign_delivery.py','--action','review_and_plan')
    checked('tests/glm5/phase2684_source_campaign_delivery.py')
    checked('tests/glm5_temp/phase2684_source_delivery_checks.py','--build')
    backend=None
    if not port_used():
        env=os.environ.copy();env.update(AI2050_SKIP_MODEL_LOAD='1',CUDA_VISIBLE_DEVICES='-1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',AI2050_HOST='127.0.0.1',AI2050_PORT='5001')
        path=OUT/'runtime/cpu_artifact_backend.log'
        with path.open('ab') as log:
            p=subprocess.Popen([str(PY),'server/server.py'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        backend={'launcher_pid':p.pid,'command':'server/server.py','started_by_this_tail':True,'CPU_only':True,'log':str(path),'started':datetime.now().astimezone().isoformat()}
        (OUT/'analysis/owned_backend.json').write_text(json.dumps(backend,indent=2)+'\n',encoding='utf-8')
    # An existing user server is never killed/reconfigured. Failed checks require
    # inspection, not a hidden replacement or an unbounded retry loop.
    ready=False
    for _ in range(18):
        try:
            with urllib.request.urlopen('http://127.0.0.1:5001/api/research-assets/native-source-cases',timeout=5) as response:
                ready=len(json.load(response)['cases'])==128
        except Exception:ready=False
        if ready:break
        time.sleep(5)
    if not ready:status(stage='backend_not_ready_inspect_without_killing_user_services',owned_backend=backend);raise SystemExit(1)
    checked('tests/glm5_temp/phase2684_live_api_checks.py')
    status(stage='ready_for_actual_browser_QA',owned_backend=backend,
           remaining='Active task must inspect actual browser source/parameter/numeric/full-width heatmaps, then record honest browser_checks; cleanup onlyafterallfourpassed; postcleanupHTTP/hash+terminalaudit,finalize2684,and samegoal2685 onward. No browser result or phase2684completion has been fabricated.')


if __name__=='__main__':main()
