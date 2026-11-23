"""Preflight, submit explicit artifacts, verify receipts; standard-library client."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from deploy.research_ingest_bundle import NoRedirect, endpoint
from deploy.storage_policy import check_write
from shared.dist_protocol.test_archive import canonical, digest, file_sha, validate_plan, ROLES


def runtime_identity():
    return {'python': sys.version, 'platform': sys.platform,
            'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions() if d.metadata['Name']}}


def save(path, value):
    path = check_write(path, len(canonical(value)) + 4096, operation='测试归档回执')
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + '.partial')
    with staging.open('wb') as stream:
        stream.write(canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    staging.replace(path)


class Client:
    def __init__(self, center='https://server.ai2050.com', token_file=None, *, session=None):
        self.base = endpoint(center)
        if session is not None:
            if token_file is not None:
                raise ValueError('choose one archive credential source')
            token = ''
        else:
            token = Path(token_file).read_text(encoding='utf-8').strip() if token_file else (
                os.environ.get('AI2050_RESEARCH_ADMIN_TOKEN') or os.environ.get('AI2050_DIST_ADMIN_TOKEN', ''))
            session = os.environ.get('AI2050_RESEARCH_SESSION', '')
        if not token and not session:
            raise ValueError('authorized admin token/session required; never pass credentials in URLs')
        self.auth = {'X-Admin-Token': token} if token else {'Authorization': 'Bearer ' + session}
        self.http = build_opener(NoRedirect())

    def request(self, path, doc=None, *, method=None, raw=None):
        payload = canonical(doc) if doc is not None else raw
        headers = {**self.auth, 'Content-Type': 'application/json' if raw is None else 'application/octet-stream'}
        for attempt in range(4):
            try:
                request = Request(self.base + '/api/test-archive' + path, data=payload, headers=headers,
                                  method=method or ('POST' if payload is not None else 'GET'))
                with self.http.open(request, timeout=3600) as response:
                    return json.load(response)
            except HTTPError as exc:
                if (exc.code < 500 and exc.code != 429) or attempt == 3:
                    # Never print request headers, tokens, or arbitrary server error bodies.
                    raise RuntimeError(f'archive HTTP {exc.code} at {path.split("?")[0]}; local data retained') from None
            except (URLError, TimeoutError, ConnectionError):
                if attempt == 3:
                    raise RuntimeError('archive connection failed; local data retained') from None
            time.sleep(min(2**attempt, 4))

    def preflight(self, plan):
        validate_plan(plan)
        result = self.request('/preflight', plan)
        if result.get('plan_sha256') != digest(plan):
            raise ValueError('preflight plan hash differs')
        return result

    def start(self, preflight, run_id, decision, reason, source_run_id=''):
        return self.request('/runs', {'run_id': run_id, 'preflight_id': preflight['preflight_id'],
                                     'decision': decision, 'reason': reason, 'source_run_id': source_run_id})

    def upload(self, path):
        before = path.stat()
        value, size = file_sha(path), before.st_size
        up = self.request('/uploads', {'sha256': value, 'size': size})
        if not up.get('available'):
            uid = up['upload_id']
            with path.open('rb') as stream:
                offset, last = up['offset'], time.monotonic()
                stream.seek(offset)
                while chunk := stream.read(up['chunk_bytes']):
                    result = self.request(f'/uploads/{uid}?offset={offset}', method='PUT', raw=chunk)
                    offset = result['offset']
                    stream.seek(offset)
                    if time.monotonic() - last > 30:
                        print(f'[archive] upload {path.name}: {offset}/{size} bytes', flush=True)
                        last = time.monotonic()
            self.request(f'/uploads/{uid}/finish', {}, method='POST')
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or file_sha(path) != value:
            raise ValueError('local artifact changed during upload; do not finalize or clean')
        return {'sha256': value, 'size': size}

    def submit(self, run_id, manifest_path, status, summary, parent_manifest_sha=''):
        manifest_path = Path(manifest_path).resolve(strict=True)
        spec = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
        if spec.get('schema') != 'research_test_files.v1' or not isinstance(spec.get('files'), list):
            raise ValueError('explicit research_test_files.v1 manifest required')
        files, names = [], set()
        for entry in spec['files']:
            name = entry.get('path')
            if not isinstance(name, str) or name in names or '\\' in name or ':' in name or name.startswith('/'):
                raise ValueError('unique relative manifest paths required')
            names.add(name)
            if entry.get('role') not in ROLES or entry.get('retention') not in {'core', 'important', 'reuse', 'scratch'}:
                raise ValueError('explicit file role/retention required')
            path = (manifest_path.parent / name).resolve(strict=True)
            if not path.is_relative_to(manifest_path.parent) or not path.is_file():
                raise ValueError('manifest path escapes its directory')
            files.append((entry, path))
        # Validate ALL paths before the first upload. Upload only explicitly listed retained files.
        artifacts = []
        for entry, path in files:
            if entry['retention'] == 'scratch':
                continue
            artifacts.append({'name': entry['path'], 'role': entry['role'], 'retention': entry['retention'],
                              'availability': 'uploaded', **self.upload(path)})
            print(f'[archive] saved {len(artifacts)} retained files', flush=True)
        final = {'status': status, 'summary': summary, 'artifacts': artifacts}
        if parent_manifest_sha:
            self.request('/runs/' + run_id + '/revisions', {'parent_manifest_sha256': parent_manifest_sha, 'final': final})
        else:
            self.request('/runs/' + run_id + '/finalize', final)
        receipt = self.request('/runs/' + run_id + '/receipt?verify=true')
        immutable = {k: v for k, v in receipt.items() if k not in {'receipt_sha256', 'readback_verified', 'cleanup_allowed'}}
        # cleanup_allowed belongs to the original signed-by-hash receipt too.
        immutable['cleanup_allowed'] = not receipt.get('missing_core_roles')
        if receipt.get('manifest_sha256') != digest(final) or receipt.get('run_id') != run_id or digest(immutable) != receipt.get('receipt_sha256'):
            raise ValueError('receipt identity or hash differs; do not clean')
        save(manifest_path.parent / 'server-receipt.json', receipt)
        return receipt


def cleanup_scratch(manifest_path, receipt, active_files=()):
    """Delete only explicit scratch files after a fresh remote receipt; never recurse."""
    if not receipt.get('readback_verified') or not receipt.get('cleanup_allowed'):
        raise ValueError('fresh verified server receipt required before cleanup')
    base = Path(manifest_path).resolve(strict=True).parent
    spec = json.loads(Path(manifest_path).read_text(encoding='utf-8-sig'))
    retained = {a['name']: a for a in receipt['artifacts'] if a['availability'] == 'uploaded'}
    candidates = []
    for entry in spec['files']:
        path = base / entry['path']
        resolved = path.resolve(strict=True)
        if path.is_symlink() or not resolved.is_relative_to(base) or not resolved.is_file():
            raise ValueError('cleanup path is not an owned regular file')
        if entry['retention'] == 'scratch':
            if entry.get('in_use') is not False or not entry.get('rebuild') or str(resolved) in active_files:
                raise ValueError('scratch rebuild/dependency review required')
            candidates.append(resolved)
        else:
            proof = retained.get(entry['path'])
            if not proof or file_sha(resolved) != proof['sha256']:
                raise ValueError('local retained file lacks matching server evidence')
    # Save a reviewable exact deletion list BEFORE removing any file.
    report = {'schema': 'research_test_cleanup.v1', 'receipt_sha256': receipt['receipt_sha256'],
              'files': [{'path': str(p), 'size': p.stat().st_size, 'sha256': file_sha(p)} for p in candidates],
              'state': 'prepared'}
    save(base / 'cleanup-record.json', report)
    for p in candidates:
        p.unlink()
    report['state'] = 'completed'
    report['bytes_removed'] = sum(f['size'] for f in report['files'])
    save(base / 'cleanup-record.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--center', default='https://server.ai2050.com')
    parser.add_argument('--token-file', type=Path)
    sub = parser.add_subparsers(dest='command', required=True)
    check = sub.add_parser('check');check.add_argument('--plan', type=Path, required=True);check.add_argument('--output', type=Path, required=True)
    query = sub.add_parser('query')
    for k in ('q', 'domain', 'model', 'state', 'family'):
        query.add_argument('--' + k, default='')
    start = sub.add_parser('start');start.add_argument('--preflight', type=Path, required=True);start.add_argument('--run-id', required=True);start.add_argument('--decision', required=True);start.add_argument('--reason', required=True);start.add_argument('--source-run-id', default='')
    submit = sub.add_parser('submit');submit.add_argument('--run-id', required=True);submit.add_argument('--manifest', type=Path, required=True);submit.add_argument('--status', choices=sorted({'completed','failed','interrupted'}), required=True);submit.add_argument('--summary', required=True);submit.add_argument('--clean-scratch', action='store_true');submit.add_argument('--parent-manifest-sha', default='')
    args = parser.parse_args()
    try:
        client = Client(args.center, args.token_file)
        if args.command == 'check':
            result = client.preflight(json.loads(args.plan.read_text(encoding='utf-8-sig')));save(args.output, result)
        elif args.command == 'query':
            result = client.request('/runs?' + urlencode({k: getattr(args,k) for k in ('q','domain','model','state','family')}))
        elif args.command == 'start':
            result = client.start(json.loads(args.preflight.read_text(encoding='utf-8-sig')), args.run_id, args.decision, args.reason, args.source_run_id)
        else:
            result = client.submit(args.run_id, args.manifest, args.status, args.summary, args.parent_manifest_sha)
            if args.clean_scratch:
                cleanup_scratch(args.manifest, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from None


if __name__ == '__main__':
    main()
