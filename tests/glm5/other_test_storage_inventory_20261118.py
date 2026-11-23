"""Metadata-only inventory for the authorized continuation of test-data cleanup."""
from __future__ import annotations

import collections
import csv
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deploy.storage_policy import checked_dir

OUT = ROOT/'tests/glm5/result/other_test_storage_cleanup_20261118_v1'
SOURCES = [ROOT/'tests', ROOT/'results', ROOT/'research', Path('D:/ai2050_native')]


def main():
    checked_dir(OUT, 128*2**20, operation='其他测试存储盘点')
    if (OUT/'inventory-files.jsonl').exists():
        raise RuntimeError('Inventory already exists; do not overwrite')
    start = time.monotonic()
    groups = collections.defaultdict(lambda: {'files': 0, 'bytes': 0, 'extensions': collections.defaultdict(lambda: [0, 0]), 'newest_mtime': 0})
    skipped = []
    total = 0
    last = start
    with (OUT/'inventory-files.jsonl').open('x', encoding='utf-8') as output, (OUT/'inventory-progress.jsonl').open('x', encoding='utf-8') as progress:
        def log(stage, **extra):
            row = {'stage': stage, 'files': total, 'elapsed_seconds': round(time.monotonic()-start, 2), **extra}
            line = json.dumps(row, ensure_ascii=False)
            print(line, flush=True)
            progress.write(line+'\n')
            progress.flush()
        log('start', roots=list(map(str, SOURCES)), free_bytes=shutil.disk_usage(ROOT).free)
        for source in SOURCES:
            if not source.exists():
                continue
            stack = [source]
            while stack:
                folder = stack.pop()
                if folder == OUT:
                    continue
                try:
                    with os.scandir(folder) as items:
                        for item in items:
                            try:
                                st = item.stat(follow_symlinks=False)
                                path = Path(item.path)
                                if item.is_symlink() or getattr(st, 'st_file_attributes', 0) & 0x400:
                                    skipped.append({'path': str(path), 'reason': 'reparse point; not followed'})
                                    continue
                                if item.is_dir(follow_symlinks=False):
                                    if item.name not in {'__pycache__', 'node_modules', '.git', '.venv'}:
                                        stack.append(path)
                                    continue
                                if not item.is_file(follow_symlinks=False):
                                    continue
                                rel = path.relative_to(source)
                                if source == ROOT/'tests':
                                    if len(rel.parts)>2 and rel.parts[1] in {'result', 'results'}:
                                        key = '/'.join(rel.parts[:3])
                                    elif rel.parts[0].endswith('_temp'):
                                        key = '/'.join(rel.parts[:2]) if len(rel.parts)>2 else rel.parts[0]
                                    else:
                                        key = '/'.join(rel.parts[:2]) if len(rel.parts)>2 else '<test-sources>'
                                elif source.name == 'ai2050_native':
                                    key = rel.parts[0]
                                else:
                                    key = '/'.join(rel.parts[:2]) if len(rel.parts)>2 else rel.parts[0]
                                group = str(source)+'::'+key
                                row = {'path': str(path.resolve()), 'source': str(source), 'group': group, 'bytes': st.st_size,
                                       'mtime_ns': st.st_mtime_ns, 'device': st.st_dev, 'inode': st.st_ino, 'nlink': st.st_nlink,
                                       'suffix': path.suffix.lower()}
                                output.write(json.dumps(row, ensure_ascii=False)+'\n')
                                total += 1
                                g = groups[group]
                                g['files'] += 1
                                g['bytes'] += st.st_size
                                g['extensions'][path.suffix.lower()][0] += 1
                                g['extensions'][path.suffix.lower()][1] += st.st_size
                                g['newest_mtime'] = max(g['newest_mtime'], st.st_mtime_ns)
                            except OSError as exc:
                                skipped.append({'path': item.path, 'reason': str(exc)})
                except OSError as exc:
                    skipped.append({'path': str(folder), 'reason': str(exc)})
                if time.monotonic()-last>=20:
                    output.flush()
                    log('inventory', current=str(folder), remaining_directories=len(stack))
                    last = time.monotonic()
            log('root_complete', root=str(source))
        output.flush()
        log('complete', groups=len(groups), skipped=len(skipped), free_bytes=shutil.disk_usage(ROOT).free)
    ordered = [{'group': k, **dict(v), 'extensions': dict(v['extensions'])} for k,v in groups.items()]
    ordered.sort(key=lambda x: x['bytes'], reverse=True)
    (OUT/'inventory-groups.json').write_text(json.dumps(ordered, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (OUT/'inventory-skipped.json').write_text(json.dumps(skipped, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    for row in ordered[:60]:
        print(json.dumps({'group': row['group'], 'files': row['files'], 'GiB': round(row['bytes']/2**30, 4),
                          'extensions': row['extensions']}, ensure_ascii=False), flush=True)


if __name__=='__main__':
    main()
