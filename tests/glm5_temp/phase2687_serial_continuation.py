"""Owned sequential continuation only; do not run duplicate model processes."""
import os,re,sys,time,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'glm5'))
import psutil
from phase2620_native_coordinate_contract import *

OUT=RESULT/'phase2687_role_qkv_field'
RUNTIME=OUT/'runtime'
STEPS=[(2688,'phase2688_native_qkv_terms.py','phase2688_native_qkv_terms'),
       (2689,'phase2689_native_qkv_scalar.py','phase2689_native_qkv_scalar'),
       (2690,'phase2690_fresh_role_qkv_confirmation.py','phase2690_fresh_role_qkv_confirmation')]


def alive_with(name):
    result=[]
    for p in psutil.process_iter(['pid','ppid','cmdline']):
        try:
            command=p.info.get('cmdline') or []
            if any(str(x).replace('\\','/').endswith('/'+name) or str(x)==name for x in command):result.append(p.info['pid'])
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return result


def verified_final(phase,folder):
    path=RESULT/folder/'analysis/final.json'
    if not path.exists():return False
    r=read(path);assert r['phase']==phase and r['all_checks_passed']
    phases=[int(s) for s in re.findall(r'^## Phase (\d+):',MEMO.read_text(encoding='utf-8-sig'),re.M)]
    assert phases.count(phase)==1 and phase<=phases[-1], 'Final file alone does not establish MEMO completion'
    return True


def state(**kwargs):save(OUT/'analysis/serial_continuation.json',{'timestamp':datetime.now().astimezone().isoformat(),'pid':os.getpid(),**kwargs})


def main():
    assert not set(alive_with(Path(__file__).name))-{os.getpid(),os.getppid()},'Duplicate continuation'
    RUNTIME.mkdir(parents=True,exist_ok=True)
    assert read(RESULT/'phase2688_native_qkv_terms/analysis/cpu_preflight.json')['all_checks_passed']
    assert read(RESULT/'phase2689_native_qkv_scalar/analysis/cpu_preflight.json')['all_checks_passed']
    frozen={script:sha(TESTS/script) for _,script,_ in STEPS};save(OUT/'protocol/continuation_sources.json',frozen)
    state(stage='waiting_actual2687_completion',steps=STEPS)
    while alive_with('phase2687_role_qkv_field.py'):time.sleep(5)
    assert verified_final(2687,'phase2687_role_qkv_field'),'2687 exited without actual completed phase; inspect failure, never invent completion'
    for phase,script,folder in STEPS:
        assert sha(TESTS/script)==frozen[script],f'Queued code changed: {script}; re-audit before launch'
        if verified_final(phase,folder):continue
        assert not alive_with(script),'Refuse duplicate child'
        state(stage='running',phase=phase,script=script)
        with (RUNTIME/f'phase{phase}.log').open('a',encoding='utf-8') as log:
            process=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-u',str(TESTS/script)],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            state(stage='running',phase=phase,script=script,child_pid=process.pid)
            code=process.wait()
        assert code==0 and verified_final(phase,folder),f'Phase{phase} failedexit{code}; actual logs retained'
    state(stage='implemented_queue_complete',complete_through=2690,same_goal=True,
        next_action='Continue2691 crossmodel>=4096Q14/512GLM/512DS sequential native, then2692ledger2693actualclientQA/cleanup. Not entirecampaign complete; currentqueue exits to existingheartbeat for next implementation.')


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        state(stage='needs_local_diagnosis',error=repr(exc),future_phases_completed=False)
        raise
