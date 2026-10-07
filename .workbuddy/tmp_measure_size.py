# -*- coding: utf-8 -*-
"""Measure workspace and git repo sizes for the public-release slimming plan."""
import io, os, subprocess, stat
from collections import defaultdict

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\tmp_size_report.txt'
REPARSE = 0x400  # FILE_ATTRIBUTE_REPARSE_POINT
SKIP_TOP = {'.git'}

def human(n):
    for u in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or u == 'GB':
            return '%.1f%s' % (n, u)
        n /= 1024.0

# ---------- 1. workspace walk (skip .git, reparse dirs) ----------
dir_stats = {}          # top -> [files, bytes]
dir2_stats = defaultdict(lambda: [0, 0])  # top/second -> [files, bytes]
ext_stats = defaultdict(lambda: [0, 0])
total = [0, 0]
big = []                # (bytes, relpath)
for dirpath, dirnames, filenames in os.walk(ROOT):
    keep = []
    for d in dirnames:
        p = os.path.join(dirpath, d)
        try:
            if os.lstat(p).st_file_attributes & REPARSE:
                continue
        except OSError:
            continue
        if d == '.git':
            continue
        keep.append(d)
    dirnames[:] = keep
    for fn in filenames:
        p = os.path.join(dirpath, fn)
        try:
            sz = os.path.getsize(p)
        except OSError:
            continue
        rel = os.path.relpath(p, ROOT)
        parts = rel.split(os.sep)
        top = parts[0]
        dir_stats.setdefault(top, [0, 0])
        dir_stats[top][0] += 1
        dir_stats[top][1] += sz
        if len(parts) >= 2 and top in ('tests', 'frontend'):
            k = top + '/' + parts[1]
            dir2_stats[k][0] += 1
            dir2_stats[k][1] += sz
        ext = os.path.splitext(fn)[1].lower() or '(noext)'
        ext_stats[ext][0] += 1
        ext_stats[ext][1] += sz
        total[0] += 1
        total[1] += sz
        big.append((sz, rel))

# ---------- 2. git index ----------
def git(*args):
    return subprocess.run(['git', '-C', ROOT] + list(args),
                          capture_output=True)

r = git('ls-files', '-z')
tracked = [t.decode('utf-8', 'replace') for t in r.stdout.split(b'\x00') if t]
tracked_size = 0
tracked_top = defaultdict(lambda: [0, 0])
tracked_big = []
missing = []
for t in tracked:
    p = os.path.join(ROOT, t.replace('/', os.sep))
    try:
        sz = os.path.getsize(p)
    except OSError:
        missing.append(t)
        sz = 0
    tracked_size += sz
    top = t.split('/')[0]
    tracked_top[top][0] += 1
    tracked_top[top][1] += sz
    tracked_big.append((sz, t))

# ---------- 3. .git object store ----------
git_dir = os.path.join(ROOT, '.git')
git_size = [0, 0]
for dirpath, dirnames, filenames in os.walk(git_dir):
    for fn in filenames:
        try:
            git_size[0] += 1
            git_size[1] += os.path.getsize(os.path.join(dirpath, fn))
        except OSError:
            pass
co = git('count-objects', '-v').stdout.decode('utf-8', 'replace')

# ---------- 4. report ----------
L = []
A = L.append
A('=== WORKSPACE TOTAL ===')
A('files=%d  bytes=%d (%s)' % (total[0], total[1], human(total[1])))
A('')
A('=== TOP-LEVEL DIRS (workspace) ===')
for k, v in sorted(dir_stats.items(), key=lambda x: -x[1][1])[:25]:
    A('%-38s %8d files  %12s' % (k, v[0], human(v[1])))
A('')
A('=== tests/* and frontend/* SECOND-LEVEL (workspace, top 15) ===')
for k, v in sorted(dir2_stats.items(), key=lambda x: -x[1][1])[:15]:
    A('%-44s %8d files  %12s' % (k, v[0], human(v[1])))
A('')
A('=== EXTENSIONS (workspace, top 20 by bytes) ===')
for k, v in sorted(ext_stats.items(), key=lambda x: -x[1][1])[:20]:
    A('%-10s %8d files  %12s' % (k, v[0], human(v[1])))
A('')
A('=== GIT INDEX (tracked) ===')
A('tracked=%d files  bytes=%d (%s)  missing_on_disk=%d' %
  (len(tracked), tracked_size, human(tracked_size), len(missing)))
A('--- tracked by top dir ---')
for k, v in sorted(tracked_top.items(), key=lambda x: -x[1][1])[:20]:
    A('%-38s %8d files  %12s' % (k, v[0], human(v[1])))
A('--- tracked big files top 25 ---')
for sz, t in sorted(tracked_big, key=lambda x: -x[0])[:25]:
    A('%12s  %s' % (human(sz), t))
A('')
A('=== GIT OBJECT STORE ===')
A('.git dir: %d files  %s' % (git_size[0], human(git_size[1])))
A('count-objects -v:')
for line in co.strip().splitlines():
    A('  ' + line)
A('')
A('=== WORKSPACE BIG FILES top 30 (any dir) ===')
for sz, p in sorted(big, key=lambda x: -x[0])[:30]:
    A('%12s  %s' % (human(sz), p))

io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n')
print('REPORT_WRITTEN')
