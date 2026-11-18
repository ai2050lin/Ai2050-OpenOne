"""Assemble verified client, center and cache deliverables; never publishes remotely."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time
import zipfile
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from deploy.storage_policy import StorageBlocked, archive_budget, check_write, checked_dir, pack_target, staging_file, stop_with_notice
OUT = ROOT / 'release/deployment-20261108'
EVIDENCE = ROOT / 'tests/glm5/result/deployment_bundle_20261108'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def assemble():
    selected = {
        'AI2050-Setup-1.0.8-x64.exe': ROOT / 'release/desktop-1.0.8/AI2050-Setup-1.0.8-x64.exe',
        'AI2050-Setup-1.0.8-x64.exe.blockmap': ROOT / 'release/desktop-1.0.8/AI2050-Setup-1.0.8-x64.exe.blockmap',
        'ai2050-center-1.0.29.tar.gz': ROOT / 'release/ai2050-center-1.0.29.tar.gz',
        'ai2050-center-1.0.29.tar.gz.sha256': ROOT / 'release/ai2050-center-1.0.29.tar.gz.sha256',
        'upgrade_center.sh': ROOT / 'release/upgrade_center.sh',
        'apply_data.py': ROOT / 'scripts/apply_client_deployment_data.py',
        'README.md': ROOT / 'deploy/README_RELEASE_20261108.md',
    }
    existing = [OUT / 'ai2050-data-20261108-mainline4.tar', OUT / 'ai2050-data-20261108-mainline4.tar.sha256']
    archive = ROOT / 'release/AI2050-release-20261108-client1.0.8-center1.0.29.zip'
    if archive.exists():
        raise FileExistsError(archive)
    additional = sum(source.stat().st_size for source in selected.values())
    estimate = archive_budget(additional + sum(p.stat().st_size for p in existing), len(selected) + len(existing) + 2)
    check_write(archive, estimate, operation='发布组合包')
    checked_dir(OUT, additional + estimate, operation='发布组合包峰值')
    for name, source in selected.items():
        dest = check_write(OUT / name, source.stat().st_size, operation='发布组合包副本')
        shutil.copyfile(source, dest)
    paths = [OUT / name for name in selected]
    paths += existing
    # Check the center is exactly the already-tested build, including its web client.
    with tarfile.open(OUT / 'ai2050-center-1.0.29.tar.gz') as tar:
        metadata = json.load(tar.extractfile('ai2050-center-1.0.29/RELEASE_MANIFEST.json'))
        for name, checksum in metadata['files'].items():
            assert sha(ROOT / name) == checksum, 'center source differs: ' + name
    desktop = json.loads((EVIDENCE / 'desktop-payload.json').read_text())
    data = json.loads((EVIDENCE / 'data-release.json').read_text(encoding='utf-8'))
    manifest = {'schema': 'deployment_release.v1', 'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'client_version': '1.0.8', 'server_version': '1.0.29',
                'mainline_revision': data['mainline_revision'], 'native_tensor_archive_included': False,
                'model_weights_included': False, 'authenticode': 'NotSigned',
                'desktop': desktop, 'validation': {'renderer_tests': 48, 'electron_tests': 11,
                    'cache_apply_tests': 7, 'server_regression_previously_passed': 324,
                    'server_package_smoke_previously_passed': True},
                'formal_server_modified': False, 'files': {}}
    for p in paths:
        manifest['files'][p.name] = {'bytes': p.stat().st_size, 'sha256': sha(p)}
    manifest_path = check_write(OUT / 'RELEASE_MANIFEST.json', operation='发布组合包清单')
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    paths.append(manifest_path)
    sums = check_write(OUT / 'SHA256SUMS.txt', operation='发布组合包校验')
    sums.write_bytes(''.join(sha(p) + '  ' + p.name + '\n' for p in sorted(paths)).encode('utf-8'))
    paths.append(sums)
    archive = pack_target(archive, sum(p.stat().st_size for p in paths), len(paths))
    with staging_file(archive) as partial, zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as zip:
        for i, p in enumerate(paths, 1):
            check_write(partial, p.stat().st_size, operation='继续打包发布组合包')
            zip.write(p, arcname='ai2050-deployment-20261108/' + p.name)
            print(f'progress {i}/{len(paths)} {p.name}', flush=True)
    with zipfile.ZipFile(archive) as zip:
        assert zip.testzip() is None
    checksum = sha(archive)
    archive.with_name(archive.name + '.sha256').write_bytes((checksum + '  ' + archive.name + '\n').encode('utf-8'))
    report = {'archive': str(archive), 'bytes': archive.stat().st_size, 'sha256': checksum,
              'file_count': len(paths), 'zip_crc_passed': True, 'files': manifest['files']}
    check_write(EVIDENCE / 'delivery.json', operation='发布核对记录').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Delivery ready', archive, archive.stat().st_size, checksum, flush=True)


if __name__ == '__main__':
    try:
        assemble()
    except StorageBlocked as error:
        stop_with_notice(error)
