"""Source/role binding campaign. Prior campaigns are strictly read-only."""
from rdc_law_common import (ROOT, RESULT, Path, np, time, json, stamp, sha, read, save,
    immutable, bits, unbits, npz, gzread, compressed, clustered, identity)
import hashlib
import shutil
import os
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')

BASE = RESULT / 'rdc_binding_campaign_20260912'
LAW = RESULT / 'rdc_law_campaign_20260911'
MEMO = ROOT / 'research/glm5/docs/AGI_GLM5_MEMO.md'

def snapshot(path):
    path = Path(path); digest = sha(path)
    target = BASE / 'sources' / (path.stem + '_' + digest[:16] + path.suffix)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists(): shutil.copyfile(path, target)
    assert sha(target) == digest
    return {'path': str(path), 'sha256': digest, 'snapshot': str(target.relative_to(BASE))}

def usage():
    return sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file())

def guard(expected=0):
    limits = read(BASE / 'resources.json')
    assert usage() + expected < limits['result_ceiling_bytes']
    assert shutil.disk_usage(ROOT).free - expected > limits['disk_floor_bytes']

def ledger(kind, seconds, **fields):
    path = BASE / 'compute_ledger.json'
    records = read(path) if path.exists() else []
    records.append(dict(timestamp=stamp(), kind=kind, seconds=seconds, **fields))
    save(path, records)
    assert sum(r['seconds'] for r in records) < read(BASE/'resources.json')['compute_ceiling_seconds']

def ranked(s):
    return hashlib.sha256(('binding2732:' + s).encode()).hexdigest()

def signed_rows():
    """Occurrence identities corrected by an append-only audited material view."""
    corrected=BASE/'signed_source/identity_recovery/resolved_material.json.gz'
    return gzread(corrected if corrected.exists() else BASE/'signed_source/natural_material.json.gz')

def source_path(row, kind='sources'):
    mode = row.get('capture_mode', 'main')
    return LAW / 'capture' / mode / kind / (row['sample_id'] + '.npz')
