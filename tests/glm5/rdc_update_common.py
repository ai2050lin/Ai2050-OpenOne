"""Relation-update campaign; every earlier campaign is read-only evidence."""
from rdc_law_common import (ROOT, RESULT, Path, np, time, json, stamp, sha, read,
    save, immutable, bits, unbits, npz, gzread, compressed, clustered, identity)
import hashlib
import shutil
import os
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')

BASE = RESULT / 'rdc_update_campaign_20260913'
PRIOR = RESULT / 'rdc_binding_campaign_20260912'
LAW = RESULT / 'rdc_law_campaign_20260911'
MEMO = ROOT / 'research/glm5/docs/AGI_GLM5_MEMO.md'

def ranked(value):
    return hashlib.sha256(('update2736:' + value).encode()).hexdigest()

def snapshot(path):
    path=Path(path);digest=sha(path)
    dest=BASE/'sources'/(path.stem+'_'+digest[:16]+path.suffix)
    dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists():shutil.copyfile(path,dest)
    assert sha(dest)==digest
    return {'path':str(path),'sha256':digest,'snapshot':str(dest.relative_to(BASE))}

def usage():
    return sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file())

def guard(expected=0):
    r=read(BASE/'resources.json')
    assert usage()+expected<r['result_ceiling_bytes']
    assert shutil.disk_usage(ROOT).free-expected>r['disk_floor_bytes']
    ledger_path=BASE/'compute_ledger.json'
    if ledger_path.exists():assert sum(x['seconds'] for x in read(ledger_path))<r['compute_ceiling_seconds']

def ledger(kind,seconds,**fields):
    path=BASE/'compute_ledger.json';entries=read(path) if path.exists() else []
    entries.append(dict(timestamp=stamp(),kind=kind,seconds=seconds,**fields));save(path,entries);guard()

def prior_natural():
    from rdc_binding_common import signed_rows
    return gzread(PRIOR/'natural_discovery.json.gz')+gzread(PRIOR/'natural_confirmation.json.gz')+signed_rows()

def native_path(row):
    mode=row.get('capture_mode','main')
    if mode=='binding':return PRIOR/'capture/natural'/f'{row["sample_id"]}.npz'
    if mode=='signed':return PRIOR/'signed_source/fields'/f'{row["sample_id"]}.npz'
    if mode=='update':return BASE/'capture/qwen4'/f'{row["sample_id"]}.npz'
    if mode=='language':return BASE/'language_capture/fields'/f'{row["sample_id"]}.npz'
    if mode=='fresh_update':return BASE/'fresh_graph/fields'/f'{row["sample_id"]}.npz'
    return LAW/'capture'/mode/'fields'/f'{row["sample_id"]}.npz'

def sources(row):
    mode=row.get('capture_mode','main');path=native_path(row)
    if mode not in ('binding','signed','update','language','fresh_update'):path=LAW/'capture'/mode/'sources'/f'{row["sample_id"]}.npz'
    with np.load(path) as z:return unbits(z['H12_sources'])

def failure(out,start,exc):
    import traceback
    seconds=time.monotonic()-start
    save(Path(out)/('failure_'+str(time.time_ns())+'.json'),dict(timestamp=stamp(),error=str(exc),traceback=traceback.format_exc(),seconds=seconds))
    ledger('failed_'+Path(out).name,seconds)
