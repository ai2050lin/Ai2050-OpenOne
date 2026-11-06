"""Test the extracted production server, then optionally ingest real results over HTTP."""
import argparse
import json
import os
from pathlib import Path
import secrets
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'tests/glm5/result/mlp_factor_steer_v1'
RELEASE = ROOT / 'release'
DATA = RELEASE / 'ai2050-research-factor-steer.tar.gz'


def port_number():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def verify_sha(path):
    import hashlib
    with path.open('rb') as stream:
        checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert checksum == Path(str(path) + '.sha256').read_text(encoding='ascii').split()[0]
    return checksum


def check(version, with_data, port, hold):
    program = RELEASE / f'ai2050-center-{version}.tar.gz'
    proof = {'package_sha256': verify_sha(program)}
    with tempfile.TemporaryDirectory(dir=ROOT / 'tests/glm5_temp', prefix=f'center-{version}-api-') as tmp:
        tmp = Path(tmp)
        with tarfile.open(program) as tar:
            tar.extractall(tmp, filter='data')
        root = tmp / f'ai2050-center-{version}'
        token = secrets.token_hex(24)
        env = {**os.environ, 'AI2050_DIST_DIR': str(tmp / 'center'), 'AI2050_NATIVE_DIR': str(tmp / 'native'),
               'AI2050_RESEARCH_DIR': str(tmp / 'research'), 'AI2050_SNAPSHOT_DISABLE': '1',
               'AI2050_LOCAL_TOOLS': '0', 'AI2050_DIST_ADMIN_TOKEN': token, 'PYTHONUTF8': '1'}
        for name in ('AI2050_DIST_ALLOW_SYNTHETIC', 'AI2050_FRONTEND_DIST', 'PYTHONPATH'):
            env.pop(name, None)
        base = f'http://127.0.0.1:{port}'

        def raw(path, body=None):
            headers = {'Content-Type': 'application/json'} if body else {}
            with urlopen(Request(base + path, data=body, headers=headers), timeout=8) as r:
                return r.read()

        def get(path):
            return json.loads(raw(path))

        def status(path, body=None):
            try:
                raw(path, body)
                return 200
            except HTTPError as exc:
                return exc.code

        with (OUT / 'isolated-release-server.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(
                [sys.executable, 'deploy/center_app.py', '--host', '127.0.0.1', '--port', str(port)],
                env=env, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            try:
                for i in range(120):
                    if process.poll() is not None:
                        raise RuntimeError('Packaged center exited before health check')
                    try:
                        health = get('/api/health')
                        break
                    except (URLError, TimeoutError):
                        time.sleep(.25)
                else:
                    raise RuntimeError('Packaged center did not start')
                index_path = '/api/traces/research-space'
                assert health['version'] == version and not health['data_dir_in_release'], health
                assert get(index_path)['schema'] == 'research_space_index.v1'
                assert status('/api/traces/raw/not-public') == 403
                assert status('/api/research-library/ingest', b'{}') in (401, 403)
                assert b'/assets/' in raw('/workbench')
                proof.update(health=health, public_index=True, same_origin_workbench=True,
                             raw_trace_guard=True, anonymous_ingest_guard=True)
                if with_data:
                    proof['data_package_sha256'] = verify_sha(DATA)
                    env['AI2050_RESEARCH_ADMIN_TOKEN'] = token
                    uploader = [sys.executable, 'deploy/research_ingest_bundle.py', str(DATA)]
                    subprocess.run(uploader + ['--verify-only'], env=env, cwd=root, check=True, timeout=100)
                    for _ in range(2):
                        subprocess.run(uploader + ['--center', base], env=env, cwd=root, check=True, timeout=300)
                    reg = json.loads((OUT / 'registration-final.json').read_text(encoding='utf-8'))
                    index = get(index_path)
                    study = next(x for x in index['studies'] if x['run_id'] == reg['run_id'])
                    assert study['localization'] == 'native_bank' and study['report_sha256'] == reg['report_sha256']
                    model = next(m for m in index['models'] if m['snapshot'] == reg['model_snapshot'])
                    assert (model['n_layers'], model['n_heads'], model['d_ff'], model['d_model']) == (36, 32, 9728, 2560)
                    prefix = '/api/research-library/artifacts/'
                    report = get(prefix + study['report_sha256'])
                    for name in ('lookup', 'dispatch'):
                        native = get(prefix + report['native'][name]['chunks'][0])['cases'][0]
                        case = get(prefix + report['cases'][name]['chunks'][0])['cases'][0]
                        assert native['id'] == case['id']
                        assert len(native['clean']['attribute']['keys']) == 8
                        assert len(native['edited']['tail_b']['values'][7]) == 128
                    graph = get(prefix + reg['graph_sha256'])
                    assert len(graph['feature_cards']) == 33
                    assert len(graph['graph_edges']) == 60
                    proof.update(endpoint=base, run_id=reg['run_id'], index_counts=index['counts'],
                                 report_sha256=reg['report_sha256'], native_values=True, idempotent=True)
                path = OUT / ('api-upload-proof.json' if with_data else 'server-package-proof.json')
                path.write_text(json.dumps(proof, indent=2, ensure_ascii=False), encoding='utf-8')
                print(json.dumps(proof, ensure_ascii=False), flush=True)
                if hold:
                    time.sleep(hold)
            finally:
                process.terminate()
                process.wait(timeout=20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', choices=('1.0.25', '1.0.26'), default='1.0.26')
    parser.add_argument('--with-data', action='store_true')
    parser.add_argument('--port', type=int)
    parser.add_argument('--hold', type=int, default=0)
    args = parser.parse_args()
    check(args.version, args.with_data, args.port or port_number(), args.hold)
