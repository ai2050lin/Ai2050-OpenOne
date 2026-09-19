"""Formation/operation/composition campaign, isolated from every prior result tree."""
import gzip
import hashlib
import shutil
from rdc_feature_common import ROOT, RESULT, Path, np, time, json, stamp, sha, read, save, immutable, bits, unbits, npz

BASE = RESULT / 'rdc_law_campaign_20260911'
OPERATOR = RESULT / 'rdc_operator_atlas_20260911'
JOINT = RESULT / 'rdc_joint_atlas_20260911'
MEMO = ROOT / 'research/glm5/docs/AGI_GLM5_MEMO.md'


def gzread(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def compressed(path, value):
    path = Path(path)
    payload = gzip.compress(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8'), mtime=0)
    if path.exists():
        assert path.read_bytes() == payload, ('Frozen material changed', path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def rank(value):
    return hashlib.sha256(('2728:' + value).encode('utf-8')).hexdigest()


def snapshot(path):
    path = Path(path)
    digest = sha(path)
    dest = BASE / 'sources' / (path.stem + '_' + digest[:16] + path.suffix)
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    assert sha(dest) == digest
    return {'path': str(path.relative_to(ROOT)), 'sha256': digest, 'snapshot': str(dest.relative_to(BASE))}


def usage():
    return sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file()) if BASE.exists() else 0


def guard(expected=0):
    r = read(BASE / 'resources.json')
    assert usage() + expected < r['result_ceiling_bytes']
    assert shutil.disk_usage(ROOT).free - expected > r['disk_floor_bytes']


def ledger(kind, seconds, **fields):
    path = BASE / 'compute_ledger.json'
    records = read(path) if path.exists() else []
    records.append({'timestamp': stamp(), 'kind': kind, 'seconds': seconds, **fields})
    save(path, records)
    assert sum(r['seconds'] for r in records) < read(BASE / 'resources.json')['compute_ceiling_seconds']


def clustered(values, groups, seed=2728):
    from collections import defaultdict
    d = defaultdict(list)
    for value, group in zip(values, groups):
        d[group].append(float(value))
    a = np.array([np.mean(d[g]) for g in sorted(d)])
    if not len(a):
        return {'groups': 0, 'mean': None, 'interval95': None}
    rng = np.random.default_rng(seed)
    boot = a[rng.integers(len(a), size=(2000, len(a)))].mean(1)
    return {'groups': len(a), 'mean': float(a.mean()), 'interval95': np.quantile(boot, [.025, .975]).tolist(),
            'scope': 'Source-cluster means; conditional on the frozen sample and fit, not independent coordinate replicates.'}


def identity(a):
    a = np.ascontiguousarray(a)
    return {'shape': list(a.shape), 'dtype': str(a.dtype), 'sha256': hashlib.sha256(a.tobytes()).hexdigest()}
