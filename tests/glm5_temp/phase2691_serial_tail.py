"""Wait for prior owned serial queue; then run crossmodels one at a time."""
import os,re,socket,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'glm5'))
import psutil
from phase2620_native_coordinate_contract import *
from phase2687_serial_continuation import alive_with,verified_final

OUT=RESULT/'phase2691_crossmodel_role_confirmation'
SCRIPT=TESTS/'phase2691_crossmodel_role_confirmation.py'
KEYS=('qwen14','glm4','ds7','ds7_answer')


def status(**values):save(OUT/'analysis/serial_tail.json',{'timestamp':datetime.now().astimezone().isoformat(),'pid':os.getpid(),**values})


def pause_owned_backend():
    path=OUT/'analysis/owned_backend.json'
    if not path.exists():path=RESULT/'phase2684_source_campaign_delivery/analysis/owned_backend.json'
    owned=read(path)
    assert owned['started_by_this_tail'] and owned['CPU_only'] and owned['command']=='server/server.py'
    try:p=psutil.Process(owned['launcher_pid'])
    except psutil.NoSuchProcess:return False
    expected=datetime.fromisoformat(owned['started']).timestamp()
    assert abs(p.create_time()-expected)<15,'Backend PID reused; refuse to stop another process'
    descendants=[*p.children(recursive=True),p];target=[];helpers=[]
    console=(Path(os.environ['SystemRoot'])/'System32/conhost.exe').resolve()
    for child in descendants:
        if any(str(x).replace('\\','/').endswith('server/server.py') for x in child.cmdline()):
            target.append(child)
        else:
            assert child.ppid()==p.pid and Path(child.exe()).resolve()==console and abs(child.create_time()-p.create_time())<15, 'Not the owned artifact backend or verified console helper'
            helpers.append(child.pid)  # OS releases its console; do not target the helper.
    save(OUT/'analysis/paused_backend.json',{'reason':'Release owned CPU artifact server for native14B checkpoint/offload; frontend and files preserved.',
        'actual_pids':[x.pid for x in target],'verified_console_helpers_not_targeted':helpers,'old_ownership':owned,'time':datetime.now().astimezone().isoformat()})
    for child in target:
        try:child.terminate()
        except psutil.NoSuchProcess:pass
    _,alive=psutil.wait_procs(target,timeout=15);assert not alive
    return True


def restore_backend():
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1',5001))==0:return {'restored_by_tail':False,'reason':'Port5001alreadyserved; no overwrite'}
    logpath=OUT/'runtime/restored_cpu_artifact_backend.log';env=os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='-1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    with logpath.open('a',encoding='utf-8') as log:
        process=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'server/server.py'],cwd=ROOT,env=env,
            stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    report={'launcher_pid':process.pid,'command':'server/server.py','started_by_this_tail':True,'CPU_only':True,
        'log':str(logpath),'started':datetime.now().astimezone().isoformat()}
    save(OUT/'analysis/owned_backend.json',report);return report


def main():
    assert not set(alive_with(Path(__file__).name))-{os.getpid(),os.getppid()},'Duplicate crossmodel tail'
    preflight=read(OUT/'analysis/material_preflight.json');assert preflight['all_checks_passed']
    source_sha=preflight['code_sha256'];assert sha(SCRIPT)==source_sha
    (OUT/'runtime').mkdir(parents=True,exist_ok=True)
    status(stage='waiting_actual2690_and_prior_queue_exit',source_sha256=source_sha)
    while alive_with('phase2687_serial_continuation.py'):time.sleep(5)
    assert verified_final(2690,'phase2690_fresh_role_qkv_confirmation'),'Prior queue did not complete; diagnose actual status, do not skip phases'
    assert not any(alive_with(s) for s in ('phase2687_role_qkv_field.py','phase2689_native_qkv_scalar.py','phase2690_fresh_role_qkv_confirmation.py'))
    paused=False
    try:
        paused=pause_owned_backend()
        assert psutil.virtual_memory().available>14*1024**3,'Insufficient host memory for amended native14B CPU10GiB/checkpoint-backed disk preflight; no user processes changed'
        for key in KEYS:
            assert sha(SCRIPT)==source_sha
            completion=OUT/key/'analysis/completion.json'
            if completion.exists():assert read(completion)['all_checks_passed'];continue
            assert not alive_with(SCRIPT.name) and not alive_with('phase2691_resource_runner.py'),'Another model protocol already running'
            status(stage='running_model',protocol=key,owned_backend_paused=paused)
            with (OUT/f'runtime/{key}.log').open('a',encoding='utf-8') as log:
                runner=ROOT/'tests/glm5_temp/phase2691_resource_runner.py'
                save(OUT/'protocol/resource_runner_source.json',{'path':str(runner),'sha256':sha(runner),'formal_scientific_script_unchanged':source_sha})
                p=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u',str(runner),key],cwd=ROOT,
                    stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                status(stage='running_model',protocol=key,child_pid=p.pid,owned_backend_paused=paused)
                code=p.wait()
            assert code==0 and completion.exists() and read(completion)['all_checks_passed'],f'{key} failedexit{code}; logs retained'
        with (OUT/'runtime/finalize.log').open('a',encoding='utf-8') as log:
            p=subprocess.run([str(ROOT/'.venv/Scripts/python.exe'),'-u',str(ROOT/'tests/glm5_temp/phase2691_resource_runner.py'),'finalize'],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        assert p.returncode==0 and verified_final(2691,'phase2691_crossmodel_role_confirmation')
    finally:
        if paused and not alive_with(SCRIPT.name) and not alive_with('phase2691_resource_runner.py'):save(OUT/'analysis/backend_restore_result.json',restore_backend())
    status(stage='crossmodels_complete_next_same_goal',complete_through=2691,same_goal=True,
        next='2692native linked rounding ledger and2693actualclient QA/cleanup remain. Existingheartbeat must implement/execute; do not claim campaignfinished.')


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        status(stage='needs_local_diagnosis',error=repr(exc),future_phases_completed=False)
        raise
