"""Verify every release member and run the packaged, GPU-free center in isolation."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tests/glm5/result/mainline_sync_20261108'
name = 'ai2050-center-' + (ROOT / 'VERSION').read_text().strip()
archive = ROOT / 'release' / (name + '.tar.gz')
actual_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
assert (ROOT / 'release' / (name + '.tar.gz.sha256')).read_text().split()[0] == actual_sha
target = OUT / 'package-smoke'
target.mkdir(exist_ok=True)
with tarfile.open(archive) as tar:
    manifest = json.load(tar.extractfile(name + '/RELEASE_MANIFEST.json'))
    assert manifest['version'] == '1.0.29'
    assert len(manifest['git_commit']) == 40 and manifest['git_dirty'] is True
    for path, checksum in manifest['files'].items():
        raw = tar.extractfile(name + '/' + path).read()
        assert hashlib.sha256(raw).hexdigest() == checksum, path
    for member in tar.getmembers():
        dest = (target / member.name).resolve()
        assert dest.is_relative_to(target.resolve()) and member.isfile()
    tar.extractall(target, filter='data')
package = target / name
env = {**os.environ, 'PYTHONPATH': str(package), 'AI2050_SNAPSHOT_DISABLE': '1',
       'AI2050_DIST_DIR': str(target / 'center-data'), 'AI2050_RESEARCH_DIR': str(target / 'empty-research'),
       'AI2050_LOCAL_TOOLS': '0', 'AI2050_DEV': '1', 'AI2050_FRONTEND_DIST': str(package / 'frontend/dist')}
code = '''
import json
from fastapi.testclient import TestClient
import center_app
app = TestClient(center_app.create_app())
assert app.get('/api/health').json()['version'] == '1.0.29'
env = app.get('/api/research/mainline').json()
record = json.loads(env['record'])
assert record['active_branch']['id'] == 'H3b'
for path in ['/api/research-library/workpackages', '/api/research-library/fragments', '/api/traces/research-space']:
    result = app.get(path)
    assert result.status_code == 200, result.text
    assert result.json()['mainline_revision'] == env['revision']
assert app.get('/workbench').status_code == 200
assert app.get('/api/models/local').status_code == 403
print(json.dumps({'packaged_health':'1.0.29','revision':env['revision'],'same_revision':True,'local_tools_guarded':True}))
'''
result = subprocess.run([sys.executable, '-c', code], cwd=package / 'deploy', env=env,
                        capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60)
(OUT / 'package-smoke.log').write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
assert result.returncode == 0, result.stderr
summary = {'archive': str(archive), 'sha256': actual_sha, 'version': manifest['version'],
           'members_verified': len(manifest['files']), 'git_commit': manifest['git_commit'],
           'git_dirty': manifest['git_dirty'], 'packaged_api_smoke_passed': True}
(OUT / 'release.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary))
