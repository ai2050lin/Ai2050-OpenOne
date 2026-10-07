# -*- coding: utf-8 -*-
"""Round-3: LFS manifest + big tracked files in research/ and tests/ script dirs."""
import io, os, subprocess
from collections import defaultdict

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\tmp_size_report3.txt'

def human(n):
    for u in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or u == 'GB':
            return '%.1f%s' % (n, u)
        n /= 1024.0

def git(*args):
    return subprocess.run(['git', '-C', ROOT] + list(args), capture_output=True)

L = []
A = L.append

# ---------- .gitattributes / lfs config ----------
attr = os.path.join(ROOT, '.gitattributes')
A('=== .gitattributes ===')
if os.path.exists(attr):
    A(io.open(attr, encoding='utf-8', errors='replace').read())
else:
    A('(not found)')
lcfg = os.path.join(ROOT, '.lfsconfig')
A('=== .lfsconfig ===')
A(io.open(lcfg, encoding='utf-8', errors='replace').read() if os.path.exists(lcfg) else '(not found)')

# ---------- lfs ls-files ----------
r = git('lfs', 'ls-files', '-l')  # sha-type size path? -l adds size? use -s
if r.returncode != 0 or not r.stdout:
    r = git('lfs', 'ls-files')
lfs_out = r.stdout.decode('utf-8', 'replace').splitlines()
A('=== LFS tracked files: %d ===' % len(lfs_out))
tot = 0
for line in lfs_out:
    parts = line.split()
    sz = 0
    for i, p in enumerate(parts):
        if p.isdigit():
            sz = int(p)
    tot += sz
    A(line)
A('LFS total (reported sizes): %s' % human(tot))

# ---------- big tracked files grouped for target dirs ----------
r = git('ls-files', '-z')
tracked = [t.decode('utf-8', 'replace') for t in r.stdout.split(b'\x00') if t]
groups = defaultdict(lambda: [0, 0])
big = []
for t in tracked:
    p = os.path.join(ROOT, t.replace('/', os.sep))
    try:
        sz = os.path.getsize(p)
    except OSError:
        sz = 0
    if t.startswith('research/') or t.startswith('tests/glm5/') or \
       t.startswith('tests/codex/') or t.startswith('tests/gpt5/') or \
       t.startswith('tests/deepseek') or t.startswith('tests/MainAnalysis') or \
       t.startswith('frontend/src') or t.startswith('frontend/public') or \
       t.startswith('server/') or t.startswith('scripts/'):
        parts = t.split('/')
        key = '/'.join(parts[:2]) if len(parts) > 2 else t
        groups[key][0] += 1
        groups[key][1] += sz
        if sz >= 500 * 1024:
            big.append((sz, t))

A('')
A('=== TARGET DIRS grouped (research/*, tests/*, frontend/src|public, server, scripts) ===')
for k, v in sorted(groups.items(), key=lambda x: -x[1][1]):
    A('%-48s %6d files  %10s' % (k, v[0], human(v[1])))
A('')
A('=== FILES >=500KB in target dirs (top 60) ===')
for sz, t in sorted(big, key=lambda x: -x[0])[:60]:
    A('%10s  %s' % (human(sz), t))
A('count>=500KB: %d  sum=%s' % (len(big), human(sum(s for s, _ in big))))

# ---------- md / txt big ones ----------
A('')
A('=== tracked .md >=1MB ===')
n = 0; s = 0
for t in tracked:
    if t.endswith('.md'):
        p = os.path.join(ROOT, t.replace('/', os.sep))
        try:
            sz = os.path.getsize(p)
        except OSError:
            sz = 0
        if sz >= 1024*1024:
            n += 1; s += sz
            A('%10s  %s' % (human(sz), t))
A('md>=1MB: %d sum=%s' % (n, human(s)))
A('')
A('=== tracked .txt >=500KB ===')
n = 0; s = 0
for t in tracked:
    if t.endswith('.txt'):
        p = os.path.join(ROOT, t.replace('/', os.sep))
        try:
            sz = os.path.getsize(p)
        except OSError:
            sz = 0
        if sz >= 500*1024:
            n += 1; s += sz
            A('%10s  %s' % (human(sz), t))
A('txt>=500KB: %d sum=%s' % (n, human(s)))

io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n')
print('OK3')
