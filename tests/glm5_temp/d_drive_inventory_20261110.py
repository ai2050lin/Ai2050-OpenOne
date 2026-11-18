"""Read-only D: inventory. Skip reparse targets and report inaccessible paths."""
from collections import defaultdict
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import time


ROOT = Path('D:/')
RESULT = Path(__file__).resolve().parents[1] / 'glm5' / 'result' / 'd_drive_review_20261110'
REPARSE = 1024
DIRECTORY = 16
GIB = 2**30


def scan():
    groups = defaultdict(lambda: [0, 0, 0.0])
    large, reparse, problems, partial = [], [], [], []
    stack = [ROOT]
    directories = files = size = 0
    started = last = time.monotonic()
    print('scan 0 directories, 0 files', flush=True)
    while stack:
        folder = stack.pop()
        directories += 1
        try:
            with os.scandir(folder) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    try:
                        stat = entry.stat(follow_symlinks=False)
                    except OSError as exc:
                        problems.append((str(path), str(exc)))
                        continue
                    parts = path.relative_to(ROOT).parts
                    if stat.st_file_attributes & REPARSE:
                        reparse.append(str(path))
                    elif stat.st_file_attributes & DIRECTORY:
                        stack.append(path)
                    else:
                        files += 1
                        size += stat.st_size
                        for length in range(1, min(len(parts), 7)):
                            group = groups[parts[:length]]
                            group[0] += stat.st_size
                            group[1] += 1
                            group[2] = max(group[2], stat.st_mtime)
                        if len(parts) == 1 or stat.st_size >= 256 * 2**20:
                            large.append((stat.st_size, str(path), stat.st_mtime))
                        if path.suffix.lower() in ('.part', '.partial', '.incomplete', '.tmp', '.retired'):
                            partial.append((stat.st_size, str(path), stat.st_mtime))
        except OSError as exc:
            problems.append((str(folder), str(exc)))
        now = time.monotonic()
        if now - last >= 25:
            print(f'scan {directories} directories, {files} files, {size/GIB:.2f} logical GiB, '
                  f'{now-started:.0f}s elapsed; remaining directory stack {len(stack)}', flush=True)
            last = now
    result = {
        'timestamp': datetime.now().astimezone().isoformat(timespec='seconds'),
        'free_bytes': shutil.disk_usage(ROOT).free, 'logical_bytes': size,
        'directories': directories, 'files': files, 'scan_seconds': round(time.monotonic() - started, 2),
        'groups': [[str(ROOT.joinpath(*p)), *s] for p, s in sorted(groups.items(), key=lambda pair: -pair[1][0])
                   if s[0] >= 32 * 2**20 or len(p) <= 2],
        'large': sorted(large, reverse=True),
        'partial': sorted(partial, reverse=True)[:300],
        'reparse': reparse,
        'problems': problems,
    }
    RESULT.mkdir(parents=True, exist_ok=True)
    (RESULT / 'inventory.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('free_bytes', 'logical_bytes', 'directories', 'files', 'scan_seconds')} |
                     {'reparse_count': len(reparse), 'problems_count': len(problems)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    scan()
