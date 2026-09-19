"""Ordered-source/query campaign; previous scientific result trees are read-only."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS','1')
import gc, gzip, hashlib, json, shutil, time
from pathlib import Path
import numpy as np
from rdc_feature_common import ROOT, RESULT, stamp, sha, read, save, immutable, bits, unbits, npz
from rdc_law_common import gzread, compressed, clustered, identity

BASE=RESULT/'rdc_query_campaign_20260913'
PRIOR=RESULT/'rdc_update_campaign_20260913'
LAW=RESULT/'rdc_law_campaign_20260911'
MEMO=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'
MODELS={'qwen4':'qwen3-4b','qwen14':'Qwen3-14B','glm4':'glm4-9b-chat-hf'}

def rank(value):return hashlib.sha256(('query2740/'+str(value)).encode()).hexdigest()

def snapshot(path):
    path=Path(path);digest=sha(path);dest=BASE/'sources'/(path.stem+'_'+digest[:16]+path.suffix)
    if not dest.exists():dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    assert sha(dest)==digest
    return {'path':str(path),'sha256':digest,'snapshot':str(dest.relative_to(BASE))}

def usage():return sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file()) if BASE.exists() else 0

def guard(expected=0):
    r=read(BASE/'resources.json')
    assert usage()+expected<r['result_ceiling_bytes'],('Result budget',usage(),expected)
    assert shutil.disk_usage(ROOT).free-expected>r['disk_floor_bytes'],('Disk reserve',shutil.disk_usage(ROOT).free)
    if (BASE/'compute_ledger.json').exists():assert sum(r['seconds'] for r in read(BASE/'compute_ledger.json'))<r['compute_ceiling_seconds']

def ledger(kind,seconds,**extra):
    p=BASE/'compute_ledger.json';lock=BASE/'compute_ledger.lock';fd=None;start=time.monotonic()
    while fd is None:
        try:fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        except FileExistsError:
            assert time.monotonic()-start<15,('Ledger lock held too long',str(lock));time.sleep(.025)
    try:
        records=read(p) if p.exists() else []
        records.append({'timestamp':stamp(),'kind':kind,'seconds':seconds,**extra});save(p,records)
    finally:os.close(fd);lock.unlink()
    guard()

def failure(out,start,exc):
    import traceback
    seconds=time.monotonic()-start
    save(Path(out)/('failure_'+str(time.time_ns())+'.json'),{'timestamp':stamp(),'seconds':seconds,'error':str(exc),'traceback':traceback.format_exc()})
    ledger('failed_'+Path(out).name,seconds)

def rms(a,axis=-1,keepdims=True):return np.sqrt(np.mean(np.asarray(a,dtype=np.float64)**2,axis=axis,keepdims=keepdims)).clip(1e-12)

def load(key,out):
    """Reuse tested dispatch; redirect its source snapshots to this campaign only."""
    import rdc_operator_model as dispatch
    dispatch.snapshot=snapshot
    model,tok=dispatch.load(key,Path(out),cpu_gib=6)
    import torch
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    assert not getattr(model,'is_quantized',False)
    save(Path(out)/'original_precision.json',{'timestamp':stamp(),'model':key,'dtype':str(model.dtype),'quantization':False,
      'device_map':getattr(model,'hf_device_map',None),
      'actual_parameter_devices':sorted({str(p.device) for p in model.parameters()}),
      'device_map_note':'All-resident current Transformers loads may omit hf_device_map; actual parameter devices are always measured.',
      'configuration_sha256':sha(ROOT/'models/hf'/MODELS[key]/'config.json'),
      'load_source':snapshot(Path(dispatch.__file__))})
    return model,tok

def clone_cache(cache,config,repeats=1):
    from transformers.cache_utils import DynamicCache
    # Own tensors, no mutation of the saved prefix and no retention of another branch.
    return DynamicCache([(l.keys.repeat_interleave(repeats,dim=0),l.values.repeat_interleave(repeats,dim=0)) for l in cache.layers],config=config)

def cache_id(cache):return [dict(block=i,keys=identity(bits(l.keys)),values=identity(bits(l.values))) for i,l in enumerate(cache.layers)]
