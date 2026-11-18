"""Inventory the local model stores and check the completed storage migration."""
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

from deploy.parameter_reader import read_slice


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tests/glm5/result/model_cache_merge_20261109'
HF_HOME = Path(r'C:\Users\Admin\.cache\huggingface')
HUB = HF_HOME / 'hub'
PRIMARY = ROOT / 'models/hf/Qwen3-14B'
LEGACY = Path('D:/develop/model/Qwen/Qwen3-14B')
ALIAS_STORE = Path('D:/develop/model/hf-cache-blobs')


def describe(path):
    real = path.resolve(strict=True)
    stat = real.stat()
    return {
        'path': str(path), 'physical_path': str(real), 'bytes': stat.st_size,
        'volume': real.drive, 'file_id': [stat.st_dev, stat.st_ino],
        'hardlink_count': stat.st_nlink,
    }


def group(name, path, files):
    entries = [describe(file) for file in sorted(set(files))]
    sizes = {}
    seen = set()
    for entry in entries:
        identity = tuple(entry['file_id'])
        if identity in seen:
            continue
        seen.add(identity)
        volume = entry['volume']
        sizes[volume] = sizes.get(volume, 0) + entry['bytes']
    return {'model': name, 'path': str(path), 'distinct_bytes_by_volume': sizes, 'files': entries}


def files_under(root):
    if not root.is_dir():
        return []
    return [path for path in root.rglob('*') if path.is_file()]


def collect():
    caches = []
    for path in sorted(HUB.glob('models--*')):
        members = []
        for part in ['blobs', 'refs', 'snapshots']:
            members.extend(files_under(path / part))
        caches.append(group(path.name.removeprefix('models--').replace('--', '/'), path, members))
    gpt2 = HF_HOME / 'gpt2-local'
    caches.append(group('GPT-2 local', gpt2, files_under(gpt2)))

    installed = []
    roots = sorted((ROOT / 'models/hf').iterdir())
    develop = Path('D:/develop/model')
    for publisher in ['Qwen', 'ZhipuAI', 'deepseek-ai']:
        roots.extend(sorted((develop / publisher).iterdir()))
    for path in roots:
        if path.is_dir():
            files = [item for item in path.iterdir() if item.is_file()]
            installed.append(group(path.name, path, files))

    return {
        'time': datetime.now().astimezone().isoformat(),
        'units': 'file bytes; group totals count each NTFS file identity once within that group',
        'note': 'Groups can share hardlinks or directory aliases. Group totals must not be added as physical totals.',
        'huggingface_model_caches': caches,
        'installed_model_folders': installed,
        'derived_native_weights': group('Derived native parameter bank', Path('D:/ai2050_native/_weights'),
                                        files_under(Path('D:/ai2050_native/_weights'))),
        'offload_cache': group('Disk offload', Path('D:/ai2050_offload'),
                               files_under(Path('D:/ai2050_offload'))),
        'free_bytes': {drive: shutil.disk_usage(drive + '/').free for drive in ['C:', 'D:']},
    }


def verify(journal, inventory):
    events = [json.loads(line) for line in journal.read_text(encoding='utf-8-sig').splitlines()]
    complete = [event for event in events if event['state'] == 'completed']
    assert [event['stage'] for event in complete] == ['Weights', 'Cache']
    shards = [event for event in events if event['state'] == 'linked']
    assert len(shards) == 8
    for event in shards:
        src, dest = describe(Path(event['canonical'])), describe(Path(event['path']))
        assert src['file_id'] == dest['file_id'] and src['hardlink_count'] >= 2
        assert src['bytes'] == dest['bytes'] == event['released_logical_bytes']

    snapshots = []
    for event in events:
        if event['state'] != 'cache_verified':
            continue
        repo = event['repo']
        source = HUB / repo / 'blobs'
        assert source.is_junction() and source.resolve() == ALIAS_STORE / repo
        for entry in event['records']:
            visible = describe(source / entry['name'])
            assert visible['bytes'] == entry['bytes'] and visible['volume'] == 'D:'
            if entry['source']:
                assert visible['file_id'] == describe(Path(entry['source']))['file_id']
        snapshots.append(repo)
    assert len(snapshots) == 2

    index_file = 'model.safetensors.index.json'
    primary_index = json.loads((PRIMARY / index_file).read_text(encoding='utf-8'))
    assert primary_index == json.loads((LEGACY / index_file).read_text(encoding='utf-8'))
    assert len(set(primary_index['weight_map'].values())) == 8
    for path in [PRIMARY, LEGACY]:
        assert all((path / name).is_file() for name in set(primary_index['weight_map'].values()))
    parameter = 'model.layers.35.mlp.gate_proj.weight'
    one = read_slice(PRIMARY, parameter, 0, 0, 1, 4)
    two = read_slice(LEGACY, parameter, 0, 0, 1, 4)
    assert one == two and one['dtype'] == 'BF16'
    return {
        'completed': True, 'duplicate_d_shards': len(shards), 'cache_snapshots_preserved': snapshots,
        'directory_resolves': {str(path): str(path.resolve()) for path in [PRIMARY, LEGACY]},
        'same_bf16_slice_at_both_paths': {'tensor': parameter, 'shape': one['actual_shape']},
        'stage_results': complete,
    }


def check_package():
    root = ROOT / 'tests/glm5/result/qwen14_mlp_atlas_v1'
    manifest = json.loads((root / 'delivery.json').read_text(encoding='utf-8-sig'))
    record = next(item for item in manifest['packages'] if item['path'].endswith('ai2050-qwen14-native.tar.gz'))
    package = Path(record['path'])
    sidecar = Path(record['sha256_file'])
    assert package.stat().st_size == record['bytes']
    assert sidecar.read_text(encoding='ascii').split()[0] == record['sha256']
    with package.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert digest == record['sha256']
    return {'file': str(package), 'bytes': record['bytes'], 'sha256': digest,
            'members': manifest['native_files'], 'unpacked_bytes': manifest['native_unpacked_bytes'],
            'remote_upload_receipt_in_delivery': manifest['remote_upload_receipt']}


def main():
    inventory = collect()
    checks = verify(OUT / 'merge-events.jsonl', inventory)
    checks['native_package'] = check_package()
    for name, document in [('model-cache-inventory.json', inventory), ('storage-verification.json', checks)]:
        path = OUT / name
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)
    print(json.dumps({'verification': checks, 'free_bytes': inventory['free_bytes']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
