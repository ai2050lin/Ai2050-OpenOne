"""Identity, D-drive storage and provenance for the fixed B36 group study."""
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(Path(__file__).parent)]
from deploy.storage_policy import check_write
from shared.dist_protocol.mechanism_trace import file_sha,digest,save_json,unbits,GpuLease
from source_binding_resources import resources

OUT=ROOT/'tests/glm5/result/b36_composition14_20261113_v3'
TEMP=ROOT/'tests/glm5_temp/b36_composition14_20261113_v3'
PREVIOUS=ROOT/'tests/glm5/result/branch_reuse14_20261113_v3'
MODEL=ROOT/'models/hf/Qwen3-14B'
SNAPSHOT='1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce'
RUN='B36-COMPOSITION14-20261113-v3'
SEED=202611131730
LENGTH=256;BATCH=4;CHUNK=32;DIM=5120;WIDTH=17408;DEPTH=40


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def now():return datetime.now(ZoneInfo('America/Chicago')).isoformat(timespec='minutes')
def group():return read(PREVIOUS/'frozen-selection.json')['graphs']['parameter']['edges'][-1]['parents']
def matched():return read(PREVIOUS/'frozen-selection.json')['graphs']['parameter']['edges'][-1]['random'][0]
def roots():return read(PREVIOUS/'frozen-selection.json')['graphs']['parameter']['roots']
def tracked():return sorted(set(group()+matched()+roots()))


def configure(peak=100*2**20):
    check_write(OUT,peak,operation='Fixed B36 composition study')
    for p in [OUT,TEMP]:check_write(p);p.mkdir(parents=True,exist_ok=True)
    for key in ['TEMP','TMP','TMPDIR','HF_HOME','HF_HUB_CACHE','HF_DATASETS_CACHE','TORCH_HOME','XDG_CACHE_HOME','MPLCONFIGDIR']:
        os.environ[key]=str(TEMP)
    os.environ.update(PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='safe.directory',GIT_CONFIG_VALUE_0=str(ROOT),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')


def save(name,data):
    from tools import provenance
    name=str(name).replace('\\','/');path=check_write(OUT/name)
    provenance.RESULTS=OUT/'provenance';check_write(provenance.RESULTS)
    helper=provenance.write_result('b36_composition14_'+name.replace('/','_'),dict(run_id=RUN,payload_sha256=digest(data),result_path=str(path)),model_path=MODEL,seed=SEED,tool='b36_composition14')
    prov=read(helper)['provenance'];assert not provenance.check_provenance({'provenance':prov})
    save_json(path,dict(**data,run_id=RUN,model_snapshot=SNAPSHOT,created_at=now(),provenance=prov,provenance_helper=str(helper.relative_to(ROOT))))


def log(stage,done,total,start,**extra):
    elapsed=time.monotonic()-start
    row=dict(stage=stage,done=done,total=total,seconds=round(elapsed,2),eta_seconds=round(elapsed/done*(total-done),1) if done else None,**resources(),**extra)
    if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
        t=sys.modules['torch'];row.update(cuda_allocated=t.cuda.memory_allocated(),cuda_peak=t.cuda.max_memory_allocated())
    assert row.get('shared_gpu_bytes') is not None and row['shared_gpu_bytes']<2**30
    print(json.dumps(row,ensure_ascii=False),flush=True)
    with (OUT/'progress.jsonl').open('a',encoding='utf8') as out:out.write(json.dumps(row,ensure_ascii=False)+'\n')
    check_write(OUT)


def arrays(path,values):
    import numpy as np
    path=check_write(path)
    if path.exists():
        with np.load(path) as existing:
            assert set(existing.files)==set(values)
            for k in values:assert np.array_equal(existing[k],values[k]),(path,k)
        return
    partial=path.with_suffix('.partial');assert not partial.exists()
    with partial.open('wb') as out:np.savez_compressed(out,**values)
    partial.replace(path)


def bind():
    import recursive_branch14 as R
    R.LENGTH=LENGTH;R.BATCH=BATCH;R.MODEL=MODEL;R.C.MODEL=MODEL
    return R
