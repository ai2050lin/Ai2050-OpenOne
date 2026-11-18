"""Verify and merge a client cache release. Does not replace research/account databases.

Default is verify-only. Stop the center before --apply; existing replaced cache
files are moved to a dated backup for rollback. Research upload archives remain
separate and use the existing authenticated client protocol.
"""
import argparse
from contextlib import closing
from datetime import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import tarfile
import tempfile


def digest(doc):
    return hashlib.sha256(json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def verify(archive):
    names = archive.getnames()
    if len(names) != len(set(names)) or len(names) > 10000:
        raise ValueError('duplicate or excessive archive members')
    meta = archive.getmember('DATA_MANIFEST.json')
    if not meta.isfile() or meta.size > 4 * 2**20:
        raise ValueError('invalid data manifest')
    manifest = json.load(archive.extractfile(meta))
    if manifest.get('schema') != 'client_deployment_data.v1':
        raise ValueError('unsupported data release')
    if set(names) != set(manifest['files']) | {'DATA_MANIFEST.json'}:
        raise ValueError('manifest and archive differ')
    for i, (name, entry) in enumerate(manifest['files'].items(), 1):
        p = PurePosixPath(name)
        member = archive.getmember(name)
        if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name or not member.isfile():
            raise ValueError('unsafe archive member: ' + name)
        if member.size != entry['bytes']:
            raise ValueError('file size differs: ' + name)
        h = hashlib.sha256()
        with archive.extractfile(member) as stream:
            for chunk in iter(lambda: stream.read(4 * 2**20), b''):
                h.update(chunk)
        if h.hexdigest() != entry['sha256']:
            raise ValueError('file checksum differs: ' + name)
        print(f'verified {i}/{len(manifest["files"])} {name}', flush=True)
    state = json.load(archive.extractfile('mainline_release.json'))['state']
    if digest(state) != manifest['mainline_revision']:
        raise ValueError('mainline revision differs')
    return manifest, state


def apply(archive_path, data_dir, *, write=False):
    data_dir = Path(data_dir).resolve()
    with tarfile.open(archive_path, 'r:') as archive:
        manifest, state = verify(archive)
        db = data_dir / 'research_library/library.sqlite3'
        if db.is_file():
            with closing(sqlite3.connect(f'file:{db.as_posix()}?mode=ro', uri=True)) as conn:
                row = conn.execute("SELECT value FROM metadata WHERE key='research_mainline.current'").fetchone()
            if row:
                p = data_dir / 'research_library/objects' / row[0][:2] / (row[0] + '.json')
                current = json.loads(p.read_text(encoding='utf-8'))
                if current['sequence'] > state['sequence'] or (current['sequence'] == state['sequence'] and digest(current) != digest(state)):
                    raise ValueError('center research state is newer or conflicting; cache downgrade refused')
        if not write:
            return {'verified': True, 'applied': False, 'revision': manifest['mainline_revision']}
        if not data_dir.is_dir():
            raise ValueError('specify the existing AI2050_DIST_DIR; it must already exist')
        stage = Path(tempfile.mkdtemp(prefix='.client-data-stage-', dir=data_dir))
        backup = data_dir / ('client-data-backup-' + datetime.now().strftime('%Y%m%dT%H%M%S%f'))
        moved = []
        try:
            for name in manifest['files']:
                if not name.startswith(('datapack/', 'research_uploads/')):
                    continue
                target = (stage / name).resolve()
                if not target.is_relative_to(stage):
                    raise ValueError('invalid extraction target')
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(name) as src, target.open('wb') as dst:
                    shutil.copyfileobj(src, dst, 4 * 2**20)
            incoming = json.loads((stage / 'datapack/extras.json').read_text(encoding='utf-8'))
            existing_file = data_dir / 'datapack/extras.json'
            existing = json.loads(existing_file.read_text(encoding='utf-8')) if existing_file.is_file() else {}
            preserved = set()
            for key, old in existing.items():
                new = incoming.get(key)
                import re
                extra_rel = 'datapack/extra/' + re.sub(r'[^A-Za-z0-9._-]', '_', key) + '.json.gz'
                if new is None or (old.get('created_at', 0) > new.get('created_at', 0) and (data_dir / extra_rel).is_file()):
                    incoming[key] = old
                    if new is not None:
                        preserved.add(extra_rel)
            (stage / 'datapack/extras.json').write_text(json.dumps(incoming, ensure_ascii=False), encoding='utf-8')
            with gzip.open(stage / 'datapack/datapack.json.gz', 'rt', encoding='utf-8') as stream:
                pack = json.load(stream)
            if pack['mainline_revision'] != manifest['mainline_revision']:
                raise ValueError('main pack revision differs')
            files = sorted(p for p in stage.rglob('*') if p.is_file())
            # Metadata is switched after the actual compressed files.
            files.sort(key=lambda p: p.name in ('extras.json', 'datapack.meta.json'))
            owner = data_dir.stat()
            for p in files:
                rel = p.relative_to(stage)
                if rel.as_posix() in preserved:
                    continue
                target = (data_dir / rel).resolve()
                if not target.is_relative_to(data_dir):
                    raise ValueError('destination escapes data directory')
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_file():
                    def file_sha(file):
                        h = hashlib.sha256()
                        with file.open('rb') as stream:
                            for chunk in iter(lambda: stream.read(4 * 2**20), b''):
                                h.update(chunk)
                        return h.hexdigest()
                    if file_sha(target) == file_sha(p):
                        continue
                saved = backup / rel
                had_old = target.exists()
                if had_old:
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(target, saved)
                moved.append((target, saved, had_old))
                os.replace(p, target)
                if hasattr(os, 'geteuid') and os.geteuid() == 0:
                    os.chown(target, owner.st_uid, owner.st_gid)
                    parent = target.parent
                    while parent != data_dir:
                        os.chown(parent, owner.st_uid, owner.st_gid)
                        parent = parent.parent
            return {'verified': True, 'applied': True, 'revision': manifest['mainline_revision'],
                    'backup': str(backup), 'files': len(moved), 'preserved_newer_extras': len(preserved)}
        except Exception:
            for target, saved, had_old in reversed(moved):
                target.unlink(missing_ok=True)
                if had_old and saved.exists():
                    os.replace(saved, target)
            raise
        finally:
            if stage.resolve().parent != data_dir or not stage.name.startswith('.client-data-stage-'):
                raise RuntimeError('invalid cleanup directory')
            shutil.rmtree(stage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(apply(args.archive, args.data_dir, write=args.apply), ensure_ascii=False, indent=2))
