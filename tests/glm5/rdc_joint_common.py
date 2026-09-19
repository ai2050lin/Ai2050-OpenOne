"""Independent, native-coordinate joint-language campaign; no implicit old-run mutation."""
import shutil
import gzip
from rdc_feature_common import ROOT, RESULT, Path, np, time, json, stamp, sha, read, save, immutable, bits, unbits, npz

BASE = RESULT / 'rdc_joint_atlas_20260911'
PREVIOUS = RESULT / 'rdc_relation_dynamics_20260910'
PREFIX = RESULT / 'rdc_prefix_atlas_20260910'
D_FLOOR = 8 * 1024**3


def usage():
    return sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file()) if BASE.exists() else 0


def guard(expected=0, *, preflight=False):
    """Large experiments require a separate, explicit allocation, not an old budget increase."""
    config_path = BASE / 'resource_allocation.json'
    if not config_path.exists():
        assert preflight and expected <= 4 * 1024**2, 'Storage choice/allocation required before model capture'
        assert shutil.disk_usage(ROOT).free - expected > D_FLOOR
        return
    config = read(config_path)
    assert usage() + expected < config['result_ceiling_bytes'], ('New-run allocation', usage(), expected)
    location = BASE.resolve()
    assert shutil.disk_usage(location).free - expected > config['result_volume_floor_bytes']
    if location.drive.lower() != ROOT.drive.lower():
        assert shutil.disk_usage(ROOT).free > config['workspace_volume_floor_bytes']


def snapshot(path):
    path = Path(path)
    digest = sha(path)
    dest = BASE / 'source_snapshots' / f'{path.stem}_{digest[:16]}{path.suffix}'
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
    assert sha(dest) == digest
    return {'path': str(path.relative_to(ROOT)), 'sha256': digest,
            'snapshot': str(dest.relative_to(BASE))}


def status(name, **values):
    save(BASE / name / 'status.json', {'timestamp': stamp(), **values})


def old_rows():
    return read(PREVIOUS / 'material.json') + read(PREVIOUS / 'fresh_material.json')


def rows(fresh=False):
    path = material_path(fresh)
    return json.loads(gzip.decompress(path.read_bytes()).decode('utf-8'))


def material_path(fresh=False):
    return BASE / ('fresh_material.json.gz' if fresh else 'material.json.gz')


def compressed_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
    payload = gzip.compress(data, compresslevel=6, mtime=0)
    if path.exists():
        assert path.read_bytes() == payload, ('Immutable compressed material', path)
    else:
        path.write_bytes(payload)


def field(row, fresh=False):
    with np.load(BASE / ('fresh' if fresh else 'main') / 'fields' / f'{row["sample_id"]}.npz') as z:
        return {k: z[k] for k in z.files}


def paired_summary(values, groups, seed=2719, nboot=2000):
    """Resample independent declared document/content groups, never coordinates or token pieces."""
    from collections import defaultdict
    bucket = defaultdict(list)
    for value, group in zip(values, groups):
        bucket[group].append(float(value))
    x = np.array([np.mean(bucket[k]) for k in sorted(bucket)], dtype=float)
    if not len(x):
        return {'groups': 0, 'mean': None, 'interval95': None}
    rng = np.random.default_rng(seed)
    means = np.mean(x[rng.integers(len(x), size=(nboot, len(x)))], axis=1)
    return {'groups': len(x), 'mean': float(x.mean()),
            'interval95': np.quantile(means, [.025, .975]).tolist(),
            'scope': 'Conditional on the frozen training fit; cluster bootstrap, no universal semantic claim.'}
