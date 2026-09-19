"""Preserve a resource-stalled native attempt; change only offload file I/O backend."""
import argparse
from rdc_update_common import *


def stop_owned_worker(pid):
    import psutil
    p=psutil.Process(pid);cmd=p.cmdline();joined=' '.join(cmd)
    assert any(n in joined for n in ('phase2738_rdc_update_scale.py','phase2739_rdc_update_scale_recovery.py')) and '--model qwen14' in joined,cmd
    assert p.cwd().lower()==str(ROOT).lower(),p.cwd()
    out=BASE/'scale/qwen14';recovery=BASE/'scale_recovery'
    if (recovery/'interrupted_attempt.json').exists():recovery=recovery/('attempt_'+str(pid))
    assert not (out/'result.json').exists()
    before=[]
    for path in out.rglob('*'):
        if path.is_file():
            dest=recovery/'initial_attempt'/path.relative_to(out);dest.parent.mkdir(parents=True,exist_ok=True)
            assert not dest.exists();shutil.copyfile(path,dest);assert sha(dest)==sha(path)
            before.append({'original':str(path.relative_to(BASE)),'preserved':str(dest.relative_to(BASE)),'sha256':sha(path)})
    elapsed=time.time()-p.create_time();info=p.memory_info()._asdict()
    receipt={'timestamp':stamp(),'source':snapshot(__file__),'process_id':pid,'command':cmd,
      'creation_time_unix':p.create_time(),'observed_elapsed_seconds_before_stop':elapsed,'memory':info,
      'preserved':before,'completed_commits':len(list((out/'commits').glob('*.json'))),
      'native_source':snapshot(ROOT/'tests/glm5/phase2738_rdc_update_scale.py'),
      'offload_source':snapshot(ROOT/'.venv/Lib/site-packages/accelerate/utils/offload.py'),
      'reason':('Observed GPU12/pread first16generation steps took259.9seconds; recover with independent-row batching and explicit native execution-shape audits.' if '--gpu12' in cmd else ('No first record completed after >15minutes; observed high Windows page faults.' if elapsed>900 else 'First16generation steps took214.9seconds in the preceding observed telemetry; lower GPU residency by1GiB using the existing conservative loader profile before changing batch or precision.'))+' Stop only this owned attempt; keep all completed evidence and original weights/materials/dtype/token cap. Any subsequent batch change must have its own saved protocol.',
      'status':'stop_requested_no_scientific_result_claimed'}
    assert elapsed>300
    save(recovery/'interrupted_attempt.json',receipt);p.terminate();p.wait(timeout=30)
    receipt.update(status='owned_worker_stopped',wall_seconds=time.time()-receipt['creation_time_unix'])
    save(recovery/'interrupted_attempt.json',receipt)
    ledger('qwen14_interrupted_resource_attempt',receipt['wall_seconds'],completed_commits=receipt['completed_commits'])
    print('OWNED_SCALE_WORKER_STOPPED',pid,receipt['wall_seconds'],flush=True)


def run(key,gpu12=False):
    import torch
    import accelerate.utils.offload as offload
    import phase2738_rdc_update_scale as experiment
    # Only this subprocess binding is changed; no installed source file is edited.
    original=offload.safe_open
    def pread(*args,**kwargs):
        kwargs['backend']='pread'
        return original(*args,**kwargs)
    name={'qwen14':'Qwen3-14B','glm4':'glm4-9b-chat-hf'}[key];folder=ROOT/'models/hf'/name
    mapping=read(folder/'model.safetensors.index.json')['weight_map']
    weight='model.layers.39.mlp.gate_proj.weight' if key=='qwen14' else 'model.layers.39.mlp.gate_up_proj.weight'
    if weight not in mapping:weight=next(k for k in sorted(mapping) if '.mlp.' in k and k.endswith('.weight'))
    path=folder/mapping[weight];start=time.monotonic()
    with original(str(path),framework='pt',device='cpu') as f:a=f.get_tensor(weight).clone()
    with pread(str(path),framework='pt',device='cpu') as f:b=f.get_tensor(weight)
    assert torch.equal(a,b) and a.dtype==torch.bfloat16
    audit={'timestamp':stamp(),'source':snapshot(__file__),'weight':weight,'shape':list(a.shape),
      'all_weight_entries_backend_exact':True,'dtype':str(a.dtype),'array_identity':identity(bits(a)),
      'seconds':time.monotonic()-start,'change':'Runtime safe_open backend for accelerate offloaded tensor reads only; all parameters, batch1, eager attention, material IDs and128-token cap unchanged.'}
    save(BASE/'scale_recovery'/f'{key}_backend_audit.json',audit);del a,b
    offload.safe_open=pread
    import rdc_operator_model as loader
    original_load=loader.load
    if gpu12:
        # Established loader's conservative profile; do not alter original loader
        # or OS memory settings, and do not silently call this a model change.
        def conservative(key,out,**kwargs):return original_load(key,out,cpu_gib=6)
        loader.load=conservative
        save(BASE/'scale_recovery'/f'{key}_GPU12_protocol.json',{'timestamp':stamp(),'source':snapshot(__file__),
          'dispatch':'Existing12GiB_GPU/6GiB_CPU original-BF16 profile',
          'reason':'13GiB dispatch plus separately resident original output head leaves about1GiB GPU headroom. Observed first16generation steps took214.9seconds; reserve one additional GiB before considering a batch/precision change.',
          'unchanged':['checkpoint weights','single-sample execution','native eager operators','input/tokenizer/positions','128generation cap','all128materials']})
    try:experiment.main(key)
    finally:offload.safe_open=original;loader.load=original_load


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stop-owned-qwen14',type=int);p.add_argument('--model',choices=['qwen14','glm4'])
    p.add_argument('--gpu12',action='store_true');args=p.parse_args()
    if args.stop_owned_qwen14:stop_owned_worker(args.stop_owned_qwen14)
    else:
        assert args.model;run(args.model,args.gpu12)
