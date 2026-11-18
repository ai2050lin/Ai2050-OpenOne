"""Wait for the shared GPU lease and retain it across preparation and execution."""
import argparse,contextlib,json,sys,time
from pathlib import Path
from datetime import datetime
sys.path.insert(0,str(Path(__file__).parent))
import source_formation_design as D

def run(stage,model,split,recover=False):
    from shared.dist_protocol import mechanism_trace as trace
    from source_binding_resources import resources
    name=f'{model}/{split}-{stage}-queue'
    assert not (D.OUT/(name+'-complete.json')).exists(),'This queued operation already completed'
    if recover:
        assert stage=='behavior' and model=='qwen14' and split=='confirm'
        root=D.OUT/model;start=root/f'{split}-behavior-v2-started.json'
        log=D.OUT/'qwen14-confirm-behavior-console.log'
        assert not (root/f'{split}-behavior.json').exists()
        assert 'RuntimeError: GPU is leased by another research/live process' in log.read_text(encoding='utf-8')
        backup=root/f'{split}-behavior-v2-lease-blocked.json';assert not backup.exists()
        assert start.resolve().is_relative_to(root.resolve()) and backup.resolve().is_relative_to(root.resolve())
        before=D.sha(start);start.rename(backup)
        D.save(name+'-recovery.json',{'reason':'First attempt failed acquiring lease before model forward','preserved_start':str(backup),
            'start_sha256':before,'log_sha256':D.sha(log),'scientific_inputs_scoring_candidates_unchanged':True},model)
    original=trace.GpuLease;lease=original();waiting=time.monotonic()
    while True:
        try:lease.acquire();break
        except RuntimeError as exc:
            if 'GPU is leased' not in str(exc):raise
            row={'time':datetime.now().astimezone().isoformat(),'stage':f'{model}/{split}/{stage}/gpu_wait','done':0,'total':1,
                'elapsed_seconds':time.monotonic()-waiting,'eta_seconds':None,**resources()}
            print(json.dumps(row),flush=True)
            with (D.OUT/'progress.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
            time.sleep(30)
    D.save(name+'-acquired.json',{'wait_seconds':time.monotonic()-waiting,'script_sha256':D.sha(Path(__file__)),
        'contract':'Own shared mechanism-gpu.lock continuously; reuse this same held lease inside same-process runner. Never start concurrent model.'},model)
    class HeldLease:
        def __enter__(self):
            assert lease.handle is not None
            return self
        def __exit__(self,*args):return False
    trace.GpuLease=HeldLease
    try:
        if stage=='behavior':
            from source_formation_behavior_v2 import behavior
            behavior(model,split)
        elif stage=='native':
            from source_formation_native import collect
            collect(model,split)
        elif stage=='causal':
            from source_formation_causal import run as execute
            execute(model,split)
        else:
            from source_formation_update import output
            output(model,split)
        D.save(name+'-complete.json',{'completed':True},model)
    finally:
        trace.GpuLease=original;lease.release()

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['behavior','native','causal','output'])
    ap.add_argument('--model',required=True);ap.add_argument('--split',required=True);ap.add_argument('--recover-lease',action='store_true')
    a=ap.parse_args();run(a.stage,a.model,a.split,a.recover_lease)
