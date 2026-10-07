# -*- coding: utf-8 -*-
"""Remove result-data/junk from git index (keep worktree files untouched).

Rules mirror the .gitignore public-release section exactly.
Output: removal list stats + post-clean tracked totals.
"""
import io, os, subprocess

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\tmp_rm_report.txt'

def git(args, **kw):
    return subprocess.run(['git', '-C', ROOT] + args, capture_output=True, **kw)

def human(n):
    for u in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or u == 'GB':
            return '%.1f%s' % (n, u)
        n /= 1024.0

DROP_PREFIX = (
    'tests/result/', 'tests/nfb_data/', 'data/MNIST/', 'frontend/dist/',
    'research/gpt5/data/', 'research/glm5/log/', 'node_modules/',
)
TEMP_DIRS = ('tests/glm5_temp/', 'tests/codex_temp/', 'tests/gpt5_temp/',
             'tests/deepseek_temp/')
DROP_EXT = ('.log', '.npy', '.csv', '.gz', '.pkl', '.exe', '.bak_20260918')
KEEP_JSONL = ('data/iso_corpus.jsonl', 'shared/data/iso_corpus.jsonl')
SNAPSHOT_MD = ('research/glm5/docs/AGI_GLM5_MEMO_2026', 'research/gpt5/docs/AGI_GPT5_MEMO_2026')

def should_drop(t):
    if t in KEEP_JSONL:
        return False
    for p in DROP_PREFIX:
        if t.startswith(p):
            return True
    for d in TEMP_DIRS:
        if t.startswith(d) and not t.endswith('.py'):
            return True
    for e in DROP_EXT:
        if t.endswith(e):
            return True
    if t.endswith('.jsonl'):
        return True
    for s in SNAPSHOT_MD:
        if t.startswith(s):
            return True
    if t == 'tests/codex/encoding_mechanism_large_scale_data_stage421.json':
        return True
    if t.startswith('research/gpt5/tests/glm5/probe'):
        return True
    if t.startswith('frontend/website/') and t.endswith('.png'):
        return True
    return False

r = git(['ls-files', '-z'])
tracked = [t.decode('utf-8', 'replace') for t in r.stdout.split(b'\x00') if t]
drop = [t for t in tracked if should_drop(t)]
keep = [t for t in tracked if not should_drop(t)]

drop_bytes = 0
paths = []
for t in drop:
    p = os.path.join(ROOT, t.replace('/', os.sep))
    try:
        drop_bytes += os.path.getsize(p)
    except OSError:
        pass
    paths.append(t)

# write pathspec file (NUL-separated, forward slashes)
spec = os.path.join(ROOT, '.workbuddy', 'tmp_pathspec.txt')
with io.open(spec, 'wb') as f:
    f.write(b'\x00'.join(t.encode('utf-8') for t in paths))

L = []
A = L.append
A('tracked_before=%d  drop=%d (%s)  keep=%d' %
  (len(tracked), len(drop), human(drop_bytes), len(keep)))
A('git version: %s' % git(['--version']).stdout.decode().strip())

# batch git rm --cached
BATCH = 1500
fails = []
for i in range(0, len(paths), BATCH):
    chunk = paths[i:i + BATCH]
    with io.open(spec, 'wb') as f:
        f.write(b'\x00'.join(t.encode('utf-8') for t in chunk))
    r = git(['rm', '-q', '--cached', '--pathspec-from-file=' + spec.replace('\\', '/'),
             '--pathspec-file-nul', '--'])
    if r.returncode != 0:
        fails.append((i, r.stderr.decode('utf-8', 'replace')[:400]))

A('batch fails: %d' % len(fails))
for i, e in fails[:5]:
    A('  batch@%d: %s' % (i, e))

# post state
r = git(['ls-files', '-z'])
tracked2 = [t.decode('utf-8', 'replace') for t in r.stdout.split(b'\x00') if t]
sz2 = 0
for t in tracked2:
    p = os.path.join(ROOT, t.replace('/', os.sep))
    try:
        sz2 += os.path.getsize(p)
    except OSError:
        pass
A('tracked_after=%d files  bytes=%d (%s)' % (len(tracked2), sz2, human(sz2)))

# top remaining by ext
from collections import defaultdict
ext = defaultdict(lambda: [0, 0])
for t in tracked2:
    p = os.path.join(ROOT, t.replace('/', os.sep))
    try:
        sz = os.path.getsize(p)
    except OSError:
        sz = 0
    e = os.path.splitext(t)[1].lower() or '(noext)'
    ext[e][0] += 1
    ext[e][1] += sz
A('--- remaining extensions top 15 ---')
for k, v in sorted(ext.items(), key=lambda x: -x[1][1])[:15]:
    A('%-10s %7d files  %10s' % (k, v[0], human(v[1])))

# verify critical keeps exist in index
for must in ('data/iso_corpus.jsonl', 'shared/data/iso_corpus.jsonl',
             'research/gpt5/docs/AGI_GPT5_MEMO.md',
             'research/glm5/docs/AGI_GLM5_MEMO.md',
             'ai2050_research_os/registry/industry.json',
             'ai2050_research_os/registry/cases.json'):
    A('kept[%s]=%s' % (must, 'YES' if must in tracked2 else 'NO'))

io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n')
print('RM_DONE')
