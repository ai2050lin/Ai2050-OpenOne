"""Shared identity, storage and provenance for branch-reuse development/confirmation."""
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
from deploy.storage_policy import check_write
from shared.dist_protocol.mechanism_trace import file_sha,digest,save_json,unbits,GpuLease
from source_binding_resources import resources

OUT=ROOT/'tests/glm5/result/branch_reuse14_20261113_v3'
TEMP=ROOT/'tests/glm5_temp/branch_reuse14_20261113_v3'
PREVIOUS=ROOT/'tests/glm5/result/recursive_branch14_20261113_v6'
MODEL=ROOT/'models/hf/Qwen3-14B'
SNAPSHOT='1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce'
RUN='BRANCH-REUSE14-20261113-v3'
SEED=202611131530


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def configure(peak=100*2**20):
    check_write(OUT,peak,operation='Qwen14 branch-reuse development/confirmation')
    for path in [OUT,TEMP]:check_write(path);path.mkdir(parents=True,exist_ok=True)
    for name in ['TEMP','TMP','TMPDIR','HF_HOME','HF_HUB_CACHE','HF_DATASETS_CACHE','TORCH_HOME','XDG_CACHE_HOME','MPLCONFIGDIR']:
        os.environ[name]=str(TEMP)
    os.environ.update(PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',GIT_CONFIG_COUNT='1',
        GIT_CONFIG_KEY_0='safe.directory',GIT_CONFIG_VALUE_0=str(ROOT),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')


def save(name,data):
    from tools import provenance
    path=check_write(OUT/name);provenance.RESULTS=OUT/'provenance';check_write(provenance.RESULTS)
    helper=provenance.write_result('branch_reuse14_'+name.replace('/','_'),
        dict(run_id=RUN,payload_sha256=digest(data),result_path=str(path)),model_path=MODEL,seed=SEED,tool='branch_reuse14')
    prov=read(helper)['provenance'];assert not provenance.check_provenance({'provenance':prov})
    save_json(path,dict(**data,run_id=RUN,model_snapshot=SNAPSHOT,provenance=prov,
        provenance_helper=str(helper.relative_to(ROOT))))


def log(stage,done,total,start,**kwargs):
    elapsed=time.monotonic()-start
    item=dict(stage=stage,done=done,total=total,seconds=round(elapsed,2),
        eta_seconds=round(elapsed/done*(total-done),1) if done else None,**resources(),**kwargs)
    if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
        t=sys.modules['torch'];item.update(cuda_allocated=t.cuda.memory_allocated(),cuda_peak=t.cuda.max_memory_allocated())
        assert item['shared_gpu_bytes']<2**30,'Shared GPU memory exceeded 1 GiB trial limit'
    print(json.dumps(item,ensure_ascii=False),flush=True)
    with (OUT/'progress.jsonl').open('a',encoding='utf8') as stream:stream.write(json.dumps(item,ensure_ascii=False)+'\n')
    check_write(OUT)

