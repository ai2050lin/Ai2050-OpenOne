"""Owned serial model coordinator; no concurrent model loads or silent retries."""
import json,subprocess,time
from datetime import datetime
from pathlib import Path
import psutil

ROOT=Path(__file__).resolve().parents[2];PY=ROOT/'.venv/Scripts/python.exe';OUT=ROOT/'tests/glm5/result/phase2684_source_campaign_delivery'


def status(**obj):
    obj['time']=datetime.now().astimezone().isoformat();path=OUT/'analysis/sequential_continuation.json';path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8')


def running_scalar():
    return [p.pid for p in psutil.process_iter(['cmdline','name']) if (p.info['name'] or '').lower()=='python.exe' and any(str(s).replace('\\','/').endswith('/phase2682_resolved_scalar_paths.py') for s in (p.info['cmdline'] or []))]


def main():
    status(stage='waiting_for_actual2682_process_exit',model_pids=running_scalar())
    while running_scalar():time.sleep(10)
    final=ROOT/'tests/glm5/result/phase2682_resolved_scalar_paths/analysis/final.json'
    if not final.exists() or not json.loads(final.read_text(encoding='utf-8'))['all_checks_passed']:
        status(stage='failed_upstream2682',no_model_launched=True);raise SystemExit(1)
    logdir=OUT/'runtime';logdir.mkdir(parents=True,exist_ok=True)
    for action in ('qwen14','glm4','ds7','ds7_answer','finalize'):
        status(stage='2683_'+action,model_sequential=True)
        with (logdir/f'phase2683_{action}.log').open('ab') as log:
            p=subprocess.Popen([str(PY),'-u','tests/glm5/phase2683_crossmodel_function_atlas.py',action],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            status(stage='2683_'+action,child_launcher=p.pid,model_sequential=True)
            code=p.wait()
        if code:
            status(stage='failed2683_'+action,exit_code=code,log=str(logdir/f'phase2683_{action}.log'));raise SystemExit(code)
    status(stage='2682_2683_complete_2684_delivery_still_required',same_goal_next='Complete actual UI/publication/cleanup/audit; do not mark whole campaign complete yet.')


if __name__=='__main__':main()
