"""Verify and ingest an ordered research release through the center's authenticated API."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import tarfile
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, url):
        return None


def read_member(archive, name, limit):
    member = archive.getmember(name)
    if not member.isfile() or member.size > limit:
        raise ValueError(f'invalid bundle member {name}')
    with archive.extractfile(member) as stream:
        return stream.read()


def verified_objects(archive, manifest):
    seen = set()
    for entry in manifest['objects']:
        sha = entry['sha256']
        if not re.fullmatch('[0-9a-f]{64}', sha) or sha in seen:
            raise ValueError('invalid or duplicate object hash')
        seen.add(sha)
        raw = read_member(archive, f'objects/{sha}.json', 32 * 2**20)
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError(f'object hash mismatch {sha}')
        record = json.loads(raw)
        if record.get('kind') != 'real':
            raise ValueError(f'non-real object {sha}')
        yield sha, raw


def endpoint(value):
    url = urlsplit(value)
    if url.scheme != 'https' and not (url.scheme == 'http' and url.hostname in {'127.0.0.1', 'localhost', '::1'}):
        raise ValueError('use HTTPS or local loopback HTTP')
    if not url.hostname or url.username or url.password or url.query or url.fragment or url.path not in ('', '/'):
        raise ValueError('center must be an origin')
    return value.rstrip('/')


def run(args):
    with tarfile.open(args.bundle, 'r:gz') as archive:
        manifest = json.loads(read_member(archive, 'MANIFEST.json', 4 * 2**20))
        if manifest.get('schema') != 'ai2050_research_upload.v1' or not manifest.get('objects'):
            raise ValueError('invalid research upload manifest')
        # Check every byte before making the first server write.
        count = sum(1 for _ in verified_objects(archive, manifest))
        if args.verify_only:
            print(f'verified {count} immutable objects')
            return
        base = endpoint(args.center)
        if args.admin_token_file:
            token = Path(args.admin_token_file).read_text(encoding='utf-8').strip()
        else:
            token = os.environ.get('AI2050_RESEARCH_ADMIN_TOKEN', '').strip()
        session = os.environ.get('AI2050_RESEARCH_SESSION', '').strip()
        if not token and not session:
            raise ValueError('provide --admin-token-file or an authorized admin token/session via environment')
        auth = {'X-Admin-Token': token} if token else {'Authorization': f'Bearer {session}'}
        client = build_opener(NoRedirect())

        def request(path, raw=None, headers=None):
            with client.open(Request(base + path, data=raw, headers=headers or {}), timeout=90) as response:
                return json.load(response)

        public = request('/api/traces/research-space')
        if public.get('schema') != 'research_space_index.v1':
            raise ValueError('center does not expose the public archive index')
        for i, (sha, raw) in enumerate(verified_objects(archive, manifest), 1):
            try:
                present = request('/api/research-library/artifacts/' + sha)
                # The artifact route is content addressed. Its presence makes retry idempotent.
                if not isinstance(present, dict):
                    raise ValueError(f'invalid existing object {sha}')
            except HTTPError as exc:
                if exc.code != 404:
                    raise
                result = request('/api/research-library/ingest', raw, {**auth, 'Content-Type': 'application/json'})
                if result.get('sha256') != sha:
                    raise ValueError(f'server returned a different hash for {sha}')
            if i % 16 == 0 or i == count:
                print(f'{i}/{count} objects verified on center', flush=True)
        published = request('/api/traces/research-space')
        lookup = {(s['run_id'], s['model_snapshot']): s for s in published.get('studies', [])}
        for study in manifest['studies']:
            identity = study['run_id'], study['model_snapshot']
            if lookup.get(identity, {}).get('report_sha256') != study['report_sha256']:
                raise ValueError(f'study is not yet visible in the public index: {identity[0]}')
            request('/api/research-library/artifacts/' + study['report_sha256'])
            print(f'{identity[0]} publicly indexed at {base}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle', type=Path)
    parser.add_argument('--center', default='https://server.ai2050.com')
    parser.add_argument('--admin-token-file', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    try:
        run(parser.parse_args())
    except HTTPError as exc:
        raise SystemExit(f'center API HTTP {exc.code} for {urlsplit(exc.url).path}') from None
    except (URLError, ValueError, OSError, KeyError, tarfile.TarError) as exc:
        raise SystemExit(str(exc)) from None


if __name__ == '__main__':
    main()
