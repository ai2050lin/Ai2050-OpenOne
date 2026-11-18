"""Sequential model campaign; every internal operation checks its own admission."""
import argparse,contextlib,gc,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import source_formation_design as D
from source_formation_queue import run as queued
from source_formation_analysis import fit,evaluate,replication
from source_formation_update import fit as update_fit,evaluate as update_evaluate

def invoke(label,fn,*args,**kwargs):
    path=D.OUT/(label+'-campaign-console.log');assert not path.exists(),path
    print('START '+label,flush=True);started=time.monotonic()
    with path.open('w',encoding='utf-8',buffering=1) as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        fn(*args,**kwargs)
    print(f'COMPLETE {label} {time.monotonic()-started:.1f}s',flush=True);gc.collect()

def panel(model,split,behavior_done=False):
    root=D.OUT/model
    if not behavior_done:invoke(f'{model}-{split}-behavior',queued,'behavior',model,split)
    if not D.read(root/f'{split}-gate.json')['passed']:
        D.save(f'{model}/{split}-internal-skipped.json',{'reason':'complete behavioral gate failed; no internal collection, fit, causal or output step'},model)
        print('BEHAVIOR_GATE_FAILED '+model+'/'+split,flush=True)
        return False
    invoke(f'{model}-{split}-native',queued,'native',model,split)
    if split=='discovery':invoke(f'{model}-fit',fit,model)
    invoke(f'{model}-{split}-formation',evaluate,model,split)
    if split=='discovery':invoke(f'{model}-update-fit',update_fit,model)
    invoke(f'{model}-{split}-update',update_evaluate,model,split)
    if split!='discovery':
        invoke(f'{model}-{split}-output',queued,'output',model,split)
        invoke(f'{model}-{split}-causal',queued,'causal',model,split)
    return True

def run():
    assert D.read(D.OUT/'qwen14/confirm-behavior-queue-complete.json')['completed'],'First confirmation behavior must complete'
    ok1=panel('qwen14','confirm',True)
    ok2=panel('qwen14','lockbox_frame')
    if ok1 and ok2:replication()
    else:D.save('qwen14/replication-decision.json',{'glm_eligible':False,'reason':'At least one complete heldout behavior gate failed; required mechanism panel missing'},'qwen14')
    eligible=D.read(D.OUT/'qwen14/replication-decision.json')['glm_eligible']
    if eligible:
        if panel('glm9','discovery'):
            panel('glm9','confirm');panel('glm9','lockbox_frame')
        else:print('GLM_INTERNAL_REPLICATION_STOPPED_AT_BEHAVIOR',flush=True)
    else:print('GLM_NOT_ADMITTED_BY_QWEN_FORMATION_GATE',flush=True)
    D.save('campaign-complete.json',{'completed':True,'qwen_behavior_panels':[ok1,ok2],'glm_eligible':eligible,
        'all_models_sequential':True,'controller_sha256':D.sha(Path(__file__))})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['run']);ap.parse_args();run()
