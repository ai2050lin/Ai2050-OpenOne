"""Ordinary-position conditional operators: isolated artifacts and bounded resources."""
import gzip
import shutil
from rdc_feature_common import ROOT, RESULT, Path, np, time, json, stamp, sha, read, save, immutable, bits, unbits, npz

BASE = RESULT / 'rdc_operator_atlas_20260911'
PRIOR = RESULT / 'rdc_joint_atlas_20260911'


def compressed(path, value):
    path = Path(path)
    data = gzip.compress(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8'), mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == data, ('Immutable material changed', path)
    else:
        path.write_bytes(data)


def gzread(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))


def rows():
    return gzread(BASE / 'material.json.gz')


def usage():
    return sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file()) if BASE.exists() else 0


def guard(expected=0):
    config = read(BASE / 'resources.json')
    assert usage() + expected < config['result_ceiling_bytes'], ('Result envelope', usage(), expected)
    assert shutil.disk_usage(ROOT).free - expected > config['disk_floor_bytes']


def snapshot(path):
    path = Path(path)
    digest = sha(path)
    target = BASE / 'source_snapshots' / (path.stem + '_' + digest[:16] + path.suffix)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copyfile(path, target)
    assert sha(target) == digest
    return {'path': str(path.relative_to(ROOT)), 'sha256': digest, 'snapshot': str(target.relative_to(BASE))}


def ledger(kind, seconds, **kw):
    path = BASE / 'compute_ledger.json'
    records = read(path) if path.exists() else []
    records.append({'timestamp': stamp(), 'kind': kind, 'seconds': seconds, **kw})
    save(path, records)
    assert sum(r['seconds'] for r in records) < read(BASE / 'resources.json')['compute_ceiling_seconds']


def rank(text):
    import hashlib
    return hashlib.sha256(('2724:' + text).encode('utf-8')).hexdigest()


def identity(array):
    import hashlib
    a = np.asarray(array)
    return {'shape': list(a.shape), 'dtype': str(a.dtype), 'sha256': hashlib.sha256(a.tobytes()).hexdigest()}


def clustered(values, groups, seed=2724):
    from collections import defaultdict
    d = defaultdict(list)
    for value, group in zip(values, groups):
        d[group].append(value)
    a = np.array([np.mean(d[g]) for g in sorted(d)], dtype=float)
    if not len(a):
        return {'groups': 0, 'mean': None, 'interval95': None}
    rng = np.random.default_rng(seed)
    b = a[rng.integers(len(a), size=(2000, len(a)))].mean(1)
    return {'groups': len(a), 'mean': float(a.mean()), 'interval95': np.quantile(b, [.025, .975]).tolist(),
            'scope': 'Document/content cluster means, conditional on the fixed fit; not a universal semantic law.'}
