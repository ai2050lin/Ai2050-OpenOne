"""Sequential fail-closed campaign driver; model leases remain inside each GPU stage."""
import subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import source_relay2_design as D
from deploy.storage_policy import process_env

def execute(name,script,*args):
    assert not (D.OUT/'stop-request.json').exists(),'Campaign stop requested'
    path=D.OUT/(name+'-console.log');assert not path.exists(),path
    print('START '+name,flush=True)
    with path.open('w',encoding='utf-8') as stream:
        result=subprocess.run([sys.executable,'-B','-u',str(D.ROOT/'tests/glm5'/script),*args],cwd=D.ROOT,
            env=process_env(D.OUT),stdout=stream,stderr=subprocess.STDOUT)
    assert result.returncode==0,(name,result.returncode)
    print('DONE '+name,flush=True)

def main():
    start=time.monotonic();gate=D.OUT/D.MODEL/'discovery-gate.json'
    while not gate.exists():
        D.log('campaign/wait-discovery-behavior',0,1,start);time.sleep(30)
    if not D.read(gate)['passed']:
        D.save('campaign-stopped.json',{'reason':'discovery behavior gate failed','internal_started':False});return
    execute('internal-freeze','source_relay2_native.py','freeze')
    execute('discovery-native','source_relay2_native.py','collect','--split','discovery')
    execute('fit','source_relay2_fit.py')
    execute('discovery-forecast','source_relay2_evaluate.py','--split','discovery')
    execute('discovery-parameters','source_relay2_parameters.py','--split','discovery')
    execute('update-fit','source_relay2_update.py','fit')
    # The discovery recurrence is descriptive and never used to choose a version.
    execute('discovery-update','source_relay2_update.py','evaluate','--split','discovery')
    execute('discovery-update-output','source_relay2_update.py','output','--split','discovery')
    panels={}
    for split in ['confirm','lockbox_frame']:
        execute(split+'-behavior','source_relay2_behavior.py','--model',D.MODEL,'--split',split)
        admitted=D.read(D.OUT/D.MODEL/f'{split}-gate.json')['passed'];panels[split]={'behavior_passed':admitted}
        if not admitted:continue
        execute(split+'-native','source_relay2_native.py','collect','--split',split)
        execute(split+'-forecast','source_relay2_evaluate.py','--split',split)
        execute(split+'-parameters','source_relay2_parameters.py','--split',split)
        execute(split+'-update','source_relay2_update.py','evaluate','--split',split)
        execute(split+'-update-output','source_relay2_update.py','output','--split',split)
        execute(split+'-causal','source_relay2_causal.py','--split',split)
        panels[split]['internal_complete']=True
    D.save('campaign-complete.json',{'panels':panels,'elapsed_seconds':time.monotonic()-start,'scientific_promotion':0})

if __name__=='__main__':
    if '--bootstrap' in sys.argv:
        waiting=time.monotonic();diagnostic=D.ROOT/'tests/glm5/result/source_relay_v1/padding-diagnostic.json'
        while not diagnostic.exists():D.log('bootstrap/wait-padding-calibration',0,1,waiting);time.sleep(30)
        assert D.read(diagnostic)['canonical_test_passed'],'Canonical numeric calibration failed'
        execute('prepare','source_relay2_design.py','prepare')
        execute('discovery-behavior','source_relay2_behavior.py','--model',D.MODEL,'--split','discovery')
    main()
