"""Actual packaged node + isolated real HTTP center; NO model tasks or research outputs.

The first input waits for the operator to click Start in the packaged client.
All credentials remain in memory or the disposable node service directory.
"""
from pathlib import Path
import json
import os
import secrets
import socket
import sqlite3
import subprocess
import sys
import time
import urllib.request
import urllib.error

ROOT=Path(__file__).resolve().parents[2]
HOME=ROOT/'tests/glm5_temp/phase2765_node'
WORK=ROOT/'tests/glm5_temp'/('phase2765_smoke_'+secrets.token_hex(4))
OUT=ROOT/'tests/glm5/result/phase2765'


def request(base,path,method='GET',body=None,headers=None):
    headers={'Content-Type':'application/json',**(headers or {})}
    req=urllib.request.Request(base+path,data=json.dumps(body).encode() if body is not None else None,method=method,headers=headers)
    with urllib.request.urlopen(req,timeout=10) as response:return json.load(response)


def wait_state(local,headers,states,timeout=18):
    started=time.monotonic()
    while time.monotonic()-started<timeout:
        value=request(local,'/status',headers=headers)
        if value['state'] in states:return value
        time.sleep(.25)
    raise AssertionError('Node did not reach expected state: '+str(states))


def main():
    WORK.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    info=json.loads((HOME/'service.json').read_text(encoding='utf-8'))
    local='http://127.0.0.1:'+str(info['port']);lh={'X-Local-Token':info['token']}
    original=request(local,'/config',headers=lh)
    assert original['task_kinds']==[],'QA node must not claim any tasks'
    assert request(local,'/status',headers=lh)['state']=='stopped'
    sys.path.insert(0,str(ROOT/'deploy'));import node_agent as A
    credpath=A.creds_path(HOME);backup=credpath.read_bytes() if credpath.exists() else None
    with socket.socket() as portpick:
        portpick.bind(('127.0.0.1',0));port=portpick.getsockname()[1]
    center='http://127.0.0.1:'+str(port)
    env={**os.environ,'AI2050_DIST_DIR':str(WORK/'center'),'AI2050_NATIVE_DIR':str(WORK/'native'),
         'AI2050_RESEARCH_DIR':str(WORK/'research'),'AI2050_DIST_ADMIN_TOKEN':secrets.token_hex(24),
         'AI2050_SNAPSHOT_DISABLE':'1','AI2050_DIST_ALLOW_SYNTHETIC':'1','PYTHONIOENCODING':'utf-8'}
    report={'created_at':time.time(),'kind':'software_e2e','isolated_center':True,'model_tasks_allowed':False,'actions':[]}
    with (OUT/'smoke-center.log').open('w',encoding='utf-8') as log:
        proc=subprocess.Popen([sys.executable,str(ROOT/'deploy/center_app.py'),'--host','127.0.0.1','--port',str(port)],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        try:
            for _ in range(80):
                try:request(center,'/api/health');break
                except (OSError,urllib.error.URLError):time.sleep(.25)
            else:raise AssertionError('QA center not ready')
            owner=request(center,'/api/auth/register','POST',{'username':'visual_qa_owner','password':secrets.token_urlsafe(24),'display_name':'本机验收账号'})
            owner_headers={'Authorization':'Bearer '+owner['token']}
            binding=request(center,'/api/auth/node-code','POST',{},owner_headers)
            request(local,'/config','PUT',{'server':center,'participation_mode':'managed','node_name':'QA桌面节点（不领取任务）','limits':{'max_session_minutes':5,'yield_gpu_util':0,'pause_on_battery':False,'min_free_disk_gib':0}},lh)
            bound=request(local,'/bind','POST',{'bind_code':binding['code']},lh)
            assert bound['owner_bound'];node=bound['node_id']
            print('QA_READY: isolated owner-bound node; no task kinds enabled. Click Start in the packaged client.',flush=True)
            if input().strip()!='GO':raise AssertionError('Expected GO after GUI start')
            status=wait_state(local,lh,{'running','waiting'});assert status['remote_granted']
            for _ in range(60):
                status=request(local,'/status',headers=lh)
                if any('尚未选择任务类型' in reason for reason in status.get('reasons',[])):break
                time.sleep(.25)
            assert any('尚未选择任务类型' in reason for reason in status.get('reasons',[]))
            report['empty_selection_reason']=status['reasons']
            for action,states in [('pause',{'paused'}),('resume',{'running','waiting'}),('stop',{'stopped'})]:
                command=None
                for _ in range(40):
                    try:command=request(center,f'/api/research-nodes/{node}/commands','POST',{'action':action},owner_headers);break
                    except urllib.error.HTTPError as exc:
                        if exc.code!=409:raise
                        time.sleep(.25)
                assert command
                state=wait_state(local,lh,states)
                # Applied is distinct from delivered: wait for the node's next acknowledgement.
                for _ in range(40):
                    rows=request(center,'/api/research-nodes',headers=owner_headers)['nodes']
                    record=next((c for n in rows for c in n['history'] if c['id']==command['id']),None)
                    if record and record['status']=='applied':break
                    time.sleep(.25)
                assert record and record['status']=='applied'
                report['actions'].append({'action':action,'node_state':state['state'],'receipt':record['status']})
                print('QA_REMOTE '+action+': applied; node='+state['state'],flush=True)
            assert not request(local,'/status',headers=lh)['remote_granted']
            with sqlite3.connect(WORK/'center/dist.sqlite3') as db:
                report['model_results']=db.execute('SELECT COUNT(*) FROM results').fetchone()[0]
            assert report['model_results']==0
            report['counts']=request(local,'/status',headers=lh)['counts'];assert not any(report['counts'].values())
            report['passed']=True
        finally:
            request(local,'/stop','POST',{},lh)
            request(local,'/config','PUT',original,lh)
            if backup is None:credpath.unlink(missing_ok=True)
            else:credpath.write_bytes(backup)
            proc.terminate();proc.wait(timeout=15)
    (OUT/'node-smoke.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('QA_DONE: pause/resume/stop applied; 0 model results; original node config restored.',flush=True)


if __name__=='__main__':main()
