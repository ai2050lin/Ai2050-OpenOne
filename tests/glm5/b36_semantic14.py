"""Shared identity and storage guard for semantic/answer-code separation."""
import os,sys,time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
from deploy.storage_policy import check_write
from shared.dist_protocol.mechanism_trace import file_sha,digest,save_json,unbits,GpuLease
from source_binding_resources import resources
import json

OUT=ROOT/'tests/glm5/result/b36_semantic14_20261113_v1'
TEMP=ROOT/'tests/glm5_temp/b36_semantic14_20261113_v1'
MODEL=ROOT/'models/hf/Qwen3-14B'
PREVIOUS=ROOT/'tests/glm5/result/branch_reuse14_20261113_v3'
SNAPSHOT='1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce'
RUN='B36-SEMANTIC14-20261113-v1';SEED=202611132010
LENGTH=256;BATCH=4;CHUNK=64;DIM=5120;WIDTH=17408

def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def now():return datetime.now(ZoneInfo('America/Chicago')).isoformat(timespec='minutes')
def group():return read(PREVIOUS/'frozen-selection.json')['graphs']['parameter']['edges'][-1]['parents']
def matched():return read(PREVIOUS/'frozen-selection.json')['graphs']['parameter']['edges'][-1]['random'][0]
def configure(peak=30*2**20):
    check_write(OUT,peak,operation='B36 semantic/format study')
    for p in [OUT,TEMP]:check_write(p);p.mkdir(parents=True,exist_ok=True)
    for k in ['TEMP','TMP','TMPDIR','HF_HOME','HF_HUB_CACHE','HF_DATASETS_CACHE','TORCH_HOME','XDG_CACHE_HOME','MPLCONFIGDIR']:
        os.environ[k]=str(TEMP)
    os.environ.update(PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='safe.directory',GIT_CONFIG_VALUE_0=str(ROOT),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
def save(name,data):
    from tools import provenance
    name=str(name).replace('\\','/');path=check_write(OUT/name);provenance.RESULTS=OUT/'provenance';check_write(provenance.RESULTS)
    helper=provenance.write_result('b36_semantic14_'+name.replace('/','_'),dict(payload_sha256=digest(data),run_id=RUN),model_path=MODEL,seed=SEED,tool='b36_semantic14')
    prov=read(helper)['provenance'];assert not provenance.check_provenance({'provenance':prov})
    save_json(path,dict(**data,run_id=RUN,model_snapshot=SNAPSHOT,created_at=now(),provenance=prov,provenance_helper=str(helper.relative_to(ROOT))))
def bind():
    import recursive_branch14 as R
    R.LENGTH=LENGTH;R.BATCH=BATCH;R.MODEL=MODEL;R.C.MODEL=MODEL
    return R
def log(stage,done,total,start,**extra):
    elapsed=time.monotonic()-start;r=dict(stage=stage,done=done,total=total,seconds=round(elapsed,2),eta_seconds=round(elapsed/done*(total-done),1) if done else None,**resources(),**extra)
    if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
        t=sys.modules['torch'];r.update(cuda_allocated=t.cuda.memory_allocated(),cuda_peak=t.cuda.max_memory_allocated())
    assert r.get('shared_gpu_bytes') is not None and r['shared_gpu_bytes']<2**30
    print(json.dumps(r,ensure_ascii=False),flush=True)
    with (OUT/'progress.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    check_write(OUT)

