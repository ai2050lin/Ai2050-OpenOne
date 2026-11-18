"""Explicit resource-only amendment and cooperative single-model entry point."""
import argparse,shutil,time
from datetime import datetime
import source_output_core as S
from source_output_core import C
from shared.dist_protocol.mechanism_trace import GpuLease

def freeze():
    p=C.checked();assert not list(C.OUT.glob('behavior-calibration-[0-9]*.json'))
    assert not (C.OUT/'resource-amendment.json').exists()
    C.save('resource-amendment.json',{'schema':'source_output_resource_amendment.v1','created_at':datetime.now().astimezone().isoformat(),
        'base_plan_sha256':C.sha(C.OUT/'plan.json'),'launcher_sha256':C.sha(__file__),
        'change':'Minimum free C volume reserve 30 -> 12 GiB; start requires reserve plus remaining 3.8 GiB run budget.',
        'reason':'An independently authorized Qwen3-14B job is using the shared C volume. Preserve both jobs and all previous evidence; the original 30 GiB guard was inherited from a much larger earlier experiment.',
        'minimum_free_GiB':12,'budget_GiB':p['resources']['budget_GiB'],'observed_free_GiB':shutil.disk_usage(C.OUT).free/2**30,
        'exposure':'No Qwen3-4B behavioral or native outcomes from this run have been produced.',
        'unchanged':['materials','candidates','behavior gates','scientific analysis','model precision','single CUDA model rule'],
        'preserved_original_plan':True,'cleanup_of_other_jobs':False})

def resources():
    path=C.OUT/'resource-amendment-behavior-only.json'
    a=C.read(path if path.exists() else C.OUT/'resource-amendment.json')
    assert a['base_plan_sha256']==C.sha(C.OUT/'plan.json') and a['launcher_sha256']==C.sha(__file__)
    if path.exists():
        assert a['original_amendment_sha256']==C.sha(C.OUT/'resource-amendment.json')
        assert a['original_launcher_sha256']==C.sha(C.OUT/'resource-launcher-v1.py')
        assert a['calibration_sha256']==C.sha(C.OUT/'behavior-calibration.json')
        assert not any(v['passed'] for v in C.read(C.OUT/'behavior-calibration.json')['branches'].values())
    return a

def guard():
    a=resources()
    used=sum(path.stat().st_size for path in C.OUT.rglob('*') if path.is_file());free=shutil.disk_usage(C.OUT).free
    assert used<a['budget_GiB']*2**30,('Own frozen storage budget exceeded',used)
    assert free>a['minimum_free_GiB']*2**30,('Shared volume below explicit amended reserve; completed evidence kept',free)
    return free/2**30

def execute(stage,split):
    C.checked();guard();C.reserve=guard;start=time.monotonic()
    while True:
        try:
            with GpuLease():pass
            break
        except RuntimeError as error:
            C.log('waiting_for_gpu',0,1,start,reason=str(error));assert time.monotonic()-start<3600;time.sleep(30)
    a=resources();used=sum(path.stat().st_size for path in C.OUT.rglob('*') if path.is_file())
    assert shutil.disk_usage(C.OUT).free>(a['minimum_free_GiB']+a['budget_GiB'])*2**30-used,'Insufficient space for remaining own budget plus reserve'
    if stage=='causal':
        from source_output_causal import run
        run()
    else:
        import source_output_runner as runner
        getattr(runner,stage)(split)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','behavior','collect','causal']);parser.add_argument('split',nargs='?',choices=['calibration','confirm']);a=parser.parse_args()
    if a.stage=='freeze':freeze()
    else:execute(a.stage,a.split)
