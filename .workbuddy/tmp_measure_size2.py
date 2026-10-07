# -*- coding: utf-8 -*-
"""Round-2 diagnosis: .git internals + tracked breakdown by second-level dir."""
import io, os, subprocess
from collections import defaultdict

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\tmp_size_report2.txt'

def human(n):
    for u in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or u == 'GB':
            return '%.1f%s' % (n, u)
        n /= 1024.0

def git(*args):
    return subprocess.run(['git', '-C', ROOT] + list(args), capture_output=True)

# ---------- .git top-level subdirs ----------
L = []
A = L.append
git_dir = os.path.join(ROOT, '.git')
A('=== .git TOP-LEVEL ===')
for name in sorted(os.listdir(git_dir)):
    p = os.path.join(git_dir, name)
    if os.path.isdir(p) and not os.path.islink(p):
        tot = 0; cnt = 0
        for dp, dn, fn in os.walk(p):
            for f in fn:
                try:
                    tot += os.path.getsize(os.path.join(dp, f)); cnt += 1
                except OSError:
                    pass
        A('%-24s %10d files  %12s' % (name, cnt, human(tot)))
    else:
        try:
            A('%-24s %10s files  %12s' % (name, '1', human(os.path.getsize(p))))
        except OSError:
            pass

# ---------- tracked by second-level ----------
r = git('ls-files', '-z')
tracked = [t.decode('utf-8', 'replace') for t in r.stdout.split(b'\x00') if t]
by2 = defaultdict(lambda: [0, 0])
big = []
for t in tracked:
    p = os.path.join(ROOT, t.replace('/', os.sep))
    try:
        sz = os.path.getsize(p)
    except OSError:
        sz = 0
    parts = t.split('/')
    key = '/'.join(parts[:2]) if len(parts) > 1 else parts[0]
    by2[key][0] += 1
    by2[key][1] += sz
    if sz > 1024*1024:
        big.append((sz, t))

A('')
A('=== TRACKED BY SECOND-LEVEL DIR (size>1MB dirs only, top 40) ===')
for k, v in sorted(by2.items(), key=lambda x: -x[1][1])[:40]:
    if v[1] > 1024*1024:
        A('%-52s %7d files  %12s' % (k, v[0], human(v[1])))
A('')
A('=== TRACKED FILES >1MB (top 60) ===')
for sz, t in sorted(big, key=lambda x: -x[0])[:60]:
    A('%12s  %s' % (human(sz), t))
A('')
A('count >1MB: %d  sum=%s' % (len(big), human(sum(s for s, _ in big))))

# ---------- extension breakdown of tracked ----------
ext = defaultdict(lambda: [0, 0])
for t in tracked:
    e = os.path.splitext(t)[1].lower() or '(noext)'
    p = os.path.join(ROOT, t.replace('/', os.sep))
    try:
        sz = os.path.getsize(p)
    except OSError:
        sz = 0
    ext[e][0] += 1
    ext[e][1] += sz
A('')
A('=== TRACKED EXTENSIONS top 25 ===')
for k, v in sorted(ext.items(), key=lambda x: -x[1][1])[:25]:
    A('%-10s %7d files  %12s' % (k, v[0], human(v[1])))

# ---------- remote state ----------
r = git('remote', '-v')
A('')
A('=== REMOTES ===')
A(r.stdout.decode('utf-8', 'replace').strip() or '(none)')
r = git('branch', '-a')
A('=== BRANCHES ===')
A(r.stdout.decode('utf-8', 'replace').strip())
r = git('log', '--oneline', '-3')
A('=== LAST COMMITS ===')
A(r.stdout.decode('utf-8', 'replace').strip())

io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n')
print('OK2')
