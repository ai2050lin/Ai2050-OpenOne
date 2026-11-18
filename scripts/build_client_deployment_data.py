"""Build the public client cache release without copying accounts, weights or native tensors.

Existing cache entries retain their original source version. Mainline views and
proofs are refreshed from a temporary index, and the language map from its manifest.
Large extra packs are streamed into the archive; no 85 GiB tensor copy is made.
"""
import gzip
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import sys
import tarfile
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'deploy')]
from deploy.storage_policy import StorageBlocked, check_write, checked_dir, pack_target, staging_file, stop_with_notice, storage_dir
OUT = ROOT / 'release/deployment-20261108'
WORK = ROOT / 'tests/glm5/result/deployment_bundle_20261108'


def checksum(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def build(stage):
    check_write(OUT, operation='离线发布包')
    checked_dir(WORK, operation='发布核对记录')
    source = ROOT / 'distributed_data'
    cache = source / 'datapack'
    seed = json.loads((ROOT / 'shared/dist_protocol/mainline_release.json').read_text(encoding='utf-8'))
    from deploy.research_library import Library
    from deploy.datapack import lang_map_plan
    from shared.dist_protocol.mainline_state import MainlineStore
    db_size = (source / 'research_library/library.sqlite3').stat().st_size
    check_write(stage, db_size * 2 + (cache / 'datapack.json.gz').stat().st_size * 2, operation='发布包暂存')
    index = stage / 'research_library'
    index.mkdir(exist_ok=True)
    with sqlite3.connect(f'file:{(source / "research_library/library.sqlite3").as_posix()}?mode=ro', uri=True) as src:
        with sqlite3.connect(index / 'library.sqlite3') as dest:
            src.backup(dest)

    class CacheLibrary(Library):
        def artifact(self, sha):
            path = self.root / 'objects' / sha[:2] / (sha + '.json')
            if path.exists():
                return super().artifact(sha)
            original = source / 'research_library/objects' / sha[:2] / (sha + '.json')
            raw = original.read_bytes()
            if hashlib.sha256(raw).hexdigest() != sha:
                raise ValueError('source evidence checksum differs: ' + sha)
            return json.loads(raw)

    lib = CacheLibrary(index)
    snapshot = MainlineStore(lib).ensure()
    import server.research_space_service as space
    space.library = lambda: lib
    view = space.research_space()
    assert view['mainline_revision'] == snapshot['revision']
    with gzip.open(cache / 'datapack.json.gz', 'rt', encoding='utf-8') as stream:
        doc = json.load(stream)
    source_version, source_created = doc['version'], doc['created_at']
    entries = doc['entries']
    entries['/api/research/mainline'] = snapshot
    entries['/api/traces/research-space'] = view
    entries['/api/research-library/workpackages'] = {**lib.workpackages(), 'mainline_revision': snapshot['revision']}
    entries['/api/research-library/overview'] = lib.summary()
    entries['/api/research-library/fragments?limit=50&offset=0'] = {**lib.fragments(), 'mainline_revision': snapshot['revision']}
    for sha, proof in seed['artifacts'].items():
        entries['/api/research-library/artifacts/' + sha] = proof
    lang_map = {path: read() for path, read in lang_map_plan()}
    entries.update(lang_map)
    map_files = lang_map['/api/lang/map']['manifest']['files']
    map_stamp = hashlib.sha256(json.dumps(map_files, sort_keys=True).encode('utf-8')).hexdigest()[:12]
    doc.update(version=source_version + '-mainline-' + snapshot['revision'][:12] + '-langmap-' + map_stamp, created_at=time.time(),
               source_cache={'version': source_version, 'created_at': source_created,
                             'note': '刷新主线状态、索引与语言图谱结构；其他缓存条目保留原始来源口径。'},
               mainline_revision=snapshot['revision'])
    newcache = stage / 'datapack'
    newcache.mkdir(exist_ok=True)
    raw = json.dumps(doc, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    (newcache / 'datapack.json.gz').write_bytes(gzip.compress(raw, mtime=0))
    meta = {k: doc[k] for k in ('schema', 'version', 'created_at', 'mainline_revision', 'source_cache')}
    meta.update(n_entries=len(entries), bytes=(newcache / 'datapack.json.gz').stat().st_size,
                raw_bytes=len(raw), errors=doc.get('errors', []))
    (newcache / 'datapack.meta.json').write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
    files = {f'datapack/{p.name}': p for p in newcache.iterdir() if p.is_file()}
    files['datapack/extras.json'] = cache / 'extras.json'
    for p in sorted((cache / 'extra').glob('*.json.gz')):
        files['datapack/extra/' + p.name] = p
    for p in sorted((ROOT / 'release').glob('ai2050-research-*.tar.gz')):
        files['research_uploads/' + p.name] = p
    files['mainline_release.json'] = ROOT / 'shared/dist_protocol/mainline_release.json'
    files['apply_data.py'] = ROOT / 'scripts/apply_client_deployment_data.py'
    files['README.md'] = ROOT / 'deploy/README_RELEASE_20261108.md'
    manifest = {'schema': 'client_deployment_data.v1', 'server_version': '1.0.29', 'client_version': '1.0.8',
                'mainline_revision': snapshot['revision'], 'source_cache_version': source_version,
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'files': {},
                'scope': '主包、15个附加包、已发布研究上传包及主线记录；不含账号、模型权重、封存正文和原生张量归档。'}
    path = OUT / 'ai2050-data-20261108-mainline4.tar'
    if path.exists():
        raise FileExistsError(path)
    path = pack_target(path, sum(file.stat().st_size for file in files.values()), len(files))
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with staging_file(path) as partial, tarfile.open(partial, 'w') as tar:
        for i, (name, file) in enumerate(files.items(), 1):
            check_write(partial, file.stat().st_size, operation='继续打包离线发布包')
            sha = checksum(file)
            manifest['files'][name] = {'sha256': sha, 'bytes': file.stat().st_size}
            tar.add(file, arcname=name, recursive=False)
            if checksum(file) != sha:
                raise RuntimeError('Source changed while packaging: ' + name)
            print(f'progress {i}/{len(files)} {name} elapsed={time.monotonic()-started:.1f}s', flush=True)
        content = json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8')
        info = tarfile.TarInfo('DATA_MANIFEST.json')
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    sha = checksum(path)
    path.with_name(path.name + '.sha256').write_bytes((sha + '  ' + path.name + '\n').encode('utf-8'))
    (WORK / 'data-release.json').write_text(json.dumps({**manifest, 'archive': str(path), 'sha256': sha,
        'archive_bytes': path.stat().st_size}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Data release verified', path, path.stat().st_size, sha, flush=True)


if __name__ == '__main__':
    try:
        root = checked_dir(storage_dir('staging', 'deployment-data'), operation='发布包暂存')
        with tempfile.TemporaryDirectory(prefix='run-', dir=root) as directory:
            build(Path(directory))
    except StorageBlocked as error:
        stop_with_notice(error)
