"""Exercise the packaged center and uploader through its HTTP routes via ASGI."""

import hashlib
import io
import json
import os
import secrets
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.parse import urlsplit
from urllib.request import Request

CENTER = 'ai2050-center-1.0.40.tar.gz'
UPLOAD = 'goodfire-phases-2970-2976-research-upload.tar.gz'
WEB = 'ai2050-web-1.0.40.zip'
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'release/deployment-20261118-goodfire'
OLD = ROOT / 'release/deployment-20261117'
BASE = 'http://127.0.0.1:5010'
CHECK = ROOT / 'tests/glm5/result/goodfire_release_20261118/validation.json'


class ASGIOpener:
    """Adapter for the uploader's urllib interface. Routing and auth stay in the center."""

    def __init__(self, client):
        self.client = client

    def open(self, req, timeout):
        assert timeout > 0 and req.full_url.startswith(BASE + '/')
        url = urlsplit(req.full_url)
        response = self.client.request(req.get_method(), url.path + ('?' + url.query if url.query else ''),
                                       content=req.data, headers=dict(req.header_items()), follow_redirects=False)
        if not response.is_success:
            raise HTTPError(req.full_url, response.status_code, response.reason_phrase, response.headers, None)
        return io.BytesIO(response.content)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def source_audit():
    from package_goodfire_release import SOURCE, STUDIES

    checks = {}
    for phase, suffix, *_ in STUDIES:
        source, = SOURCE.glob(f'paradigm_{suffix}_v1')
        manifest = json.loads((source / 'manifest.json').read_bytes())
        for name, meta in manifest['files'].items():
            path = source / name
            with path.open('rb') as handle:
                checksum = hashlib.file_digest(handle, 'sha256').hexdigest()
            assert path.stat().st_size == meta['bytes'] and checksum == meta['sha256'], path
        checks[phase] = len(manifest['files'])
        print(f'Phase {phase}: verified {checks[phase]} original result files', flush=True)
    return checks


def upgrade_check():
    sys.path.insert(0, str(OUT))
    try:
        import upgrade_goodfire as upgrade
    finally:
        sys.path.remove(str(OUT))
    health = dict(service='ai2050-center', data_dir_in_release=False)
    cases = [('1.0.39', [], False), ('1.0.39', ['--apply'], True),
             ('1.0.40', ['--apply'], False), ('1.0.100', ['--apply'], False)]
    for version, args, called in cases:
        with (patch.object(upgrade, 'verify_files'), patch.object(upgrade, 'read_health', return_value={**health, 'version': version}),
              patch.object(upgrade.subprocess, 'run') as system, patch.object(sys, 'argv', ['upgrade_goodfire.py', *args])):
            upgrade.main()
            if called:
                system.assert_called_once_with(['bash', str(OUT / 'upgrade_center.sh'), str(OUT / CENTER)], check=True)
            else:
                system.assert_not_called()
    for invalid in ({**health, 'version': 'unknown'}, {**health, 'version': '1.0.39', 'data_dir_in_release': True}):
        try:
            upgrade.upgrade_needed(invalid)
        except ValueError:
            continue
        raise AssertionError('Unsafe health should fail the upgrade check')
    return dict(default_is_read_only=True, older_explicit_passes_to_upgrade=True, current_or_newer_is_kept=True,
                invalid_health_and_internal_data_path_rejected=True, linux_systemd_live_test=False)


def extract_center(temp):
    prefix = CENTER.removesuffix('.tar.gz') + '/'
    with tarfile.open(OUT / CENTER, 'r:gz') as archive:
        manifest = json.load(archive.extractfile(prefix + 'RELEASE_MANIFEST.json'))
        assert set(archive.getnames()) == {prefix + name for name in (*manifest['files'], 'RELEASE_MANIFEST.json')}
        for name, digest in manifest['files'].items():
            member = archive.getmember(prefix + name)
            assert member.isfile() and '..' not in Path(name).parts
            assert sha(archive.extractfile(member).read()) == digest, name
        archive.extractall(temp, filter='data')
    with zipfile.ZipFile(OUT / WEB) as web:
        expected = {n: v for n, v in manifest['files'].items() if n.startswith('frontend/dist/')}
        assert len(web.namelist()) == len(expected)
        assert {n: sha(web.read(n)) for n in web.namelist()} == expected
    assert sha((OUT / 'upgrade_center.sh').read_bytes()) == manifest['files']['deploy/upgrade_center.sh']
    assert sha((OUT / 'research_ingest_bundle.py').read_bytes()) == manifest['files']['deploy/research_ingest_bundle.py']
    return temp / prefix, manifest['files'], expected


def readback(request, studies, manifest, first_record):
    try:
        request('/api/research-library/ingest', first_record)
    except HTTPError as exc:
        assert exc.code in (401, 403), exc.code
    else:
        raise AssertionError('Anonymous ingest succeeded')
    report = []
    expected_levels = {'observation': 2, 'negative': 3, 'intervention': 1}
    card_levels = {}
    for expected in studies:
        q = urlencode({'run_id': expected['run_id'], 'model': expected['model_snapshot']})
        index = json.loads(request('/api/traces/research-space?' + q))
        study, = [s for s in index['studies'] if s['run_id'] == expected['run_id']]
        assert all(study[key] == expected[key] for key in ('run_id', 'model_snapshot', 'report_sha256', 'binding_sha256'))
        assert study['stages']['independent_recomputation']['status'] == 'pending'
        doc = json.loads(request('/api/research-library/artifacts/' + study['report_sha256']))
        assert doc['counterexamples'] and doc['evidence_sha256'] == expected['evidence_sha256']
        assert all(item['model_snapshot'] == expected['model_snapshot'] for item in index['objects'])
        for item in index['objects']:
            level = item['body']['evidence']['level']
            card_levels[level] = card_levels.get(level, 0) + 1
        report.append(dict(run_id=study['run_id'], report_sha256=study['report_sha256'], cards=len(index['objects'])))
    for entry in manifest['objects']:
        digest = entry['sha256']
        raw = request('/api/research-library/artifacts/' + digest)
        doc = json.dumps(json.loads(raw), ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
        assert sha(doc.encode('utf-8')) == digest
    assert card_levels == expected_levels, card_levels
    result_2976 = json.loads(request('/api/research-library/artifacts/' + studies[-1]['evidence_sha256']))
    assert result_2976['phase'] == 2976 and result_2976['measurements']['native']['counterexamples']
    return report, card_levels


def worker(temp):
    sys.path.insert(0, str(Path.cwd()))
    from deploy import research_ingest_bundle
    from deploy.center_app import create_app
    from fastapi.testclient import TestClient

    assert Path(research_ingest_bundle.__file__).is_relative_to(Path.cwd())
    opener = ASGIOpener(TestClient(create_app(), base_url=BASE))
    research_ingest_bundle.build_opener = lambda *handlers: opener

    def request(path, data=None):
        with opener.open(Request(BASE + path, data, {'Content-Type': 'application/json'}), timeout=8) as res:
            return res.read()

    def ingest(path):
        research_ingest_bundle.run(SimpleNamespace(bundle=path, verify_only=False, center=BASE,
                                                   admin_token_file=temp / 'smoke-token.txt'))

    bundle = OUT / UPLOAD
    with tarfile.open(bundle, 'r:gz') as archive:
        index = json.load(archive.extractfile('MANIFEST.json'))
        first_record = archive.extractfile(f'objects/{index["objects"][0]["sha256"]}.json').read()
    health = json.loads(request('/api/health'))
    assert health['version'] == '1.0.40' and health['data_dir_in_release'] is False
    print('Packaged center API healthy through ASGI', flush=True)
    ingest(OLD / 'phase-2955-research-upload.tar.gz')
    ingest(bundle)
    before = json.loads(request('/api/traces/research-space'))
    ingest(bundle)
    after = json.loads(request('/api/traces/research-space'))
    assert before['fingerprint'] == after['fingerprint'] and before['counts'] == after['counts']
    assert any(s['run_id'] == 'KV-RECEIPT-20261117-v1' for s in after['studies'])
    reports, levels = readback(request, index['studies'], index, first_record)
    with zipfile.ZipFile(OUT / WEB) as archive:
        for name in archive.namelist():
            assert sha(request('/' + name.removeprefix('frontend/dist/'))) == sha(archive.read(name)), name
        assert sha(request('/workbench')) == sha(archive.read('frontend/dist/index.html'))
        web_count = len(archive.namelist())
    try:
        request('/api/research-library/artifacts/' + '0' * 64)
    except HTTPError as exc:
        assert exc.code == 404
    else:
        raise AssertionError('Missing artifact did not return 404')
    validation = dict(health=health, web_http_files=web_count, studies=reports, card_levels=levels,
                      public_artifacts=len(index['objects']), existing_study_retained=True,
                      anonymous_ingest_rejected=True, duplicate_ingest_preserved_counts=True, missing_artifact_404=True)
    (temp / 'api-readback.json').write_bytes(json.dumps(validation).encode('utf-8'))


def smoke(temp):
    center, files, _ = extract_center(temp)
    token_file = temp / 'smoke-token.txt'
    token = secrets.token_urlsafe(32)
    token_file.write_text(token, encoding='ascii')
    env = {**os.environ, 'AI2050_DIST_DIR': str(temp / 'data'), 'AI2050_DIST_ADMIN_TOKEN': token,
           'AI2050_RESEARCH_DIR': str(temp / 'empty-research'), 'PYTHONDONTWRITEBYTECODE': '1',
           'PYTHONIOENCODING': 'utf-8'}
    subprocess.run([sys.executable, '-B', '-u', str(Path(__file__).resolve()), '--worker', str(temp)],
                   cwd=center, env=env, check=True, timeout=45)
    result = json.loads((temp / 'api-readback.json').read_bytes())
    return dict(result, center_files=len(files), transport='ASGI test client, no live sockets')


def main():
    from package_goodfire_release import check_write, write

    checked = source_audit()
    scratch = check_write(ROOT / 'tests/glm5_temp', 150 * 2**20)
    with tempfile.TemporaryDirectory(prefix='goodfire-release-', dir=scratch) as path:
        validation = smoke(Path(path))
    validation.update(source_run_files=checked, upgrade_policy=upgrade_check(),
                      upload_sha256=sha((OUT / UPLOAD).read_bytes()))
    write(CHECK, json.dumps(validation, indent=2).encode('utf-8') + b'\n')
    print(CHECK, flush=True)


if __name__ == '__main__':
    if sys.argv[1:2] == ['--worker']:
        worker(Path(sys.argv[2]))
    else:
        main()
