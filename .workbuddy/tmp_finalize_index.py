# -*- coding: utf-8 -*-
"""Finalize index: residual cleanup + add new platform artifacts + stats."""
import io, os, subprocess, glob

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\tmp_final_index.txt'

def git(args, **kw):
    return subprocess.run(['git', '-C', ROOT] + args, capture_output=True, **kw)

def human(n):
    for u in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or u == 'GB':
            return '%.1f%s' % (n, u)
        n /= 1024.0

L = []
A = L.append

# ---- 1. residual removals (log needs --sparse; bak variants) ----
r = git(['rm', '-q', '--cached', '--sparse', '--',
         'server/server_5001.err.log', 'server/server_5001.out.log'])
A('rm logs: rc=%d %s' % (r.returncode, r.stderr.decode('utf-8', 'replace')[:200]))

r = git(['ls-files', '-z'])
tracked = [t.decode('utf-8', 'replace') for t in r.stdout.split(b'\x00') if t]
bak = [t for t in tracked if '.bak_' in t]
wb = [t for t in tracked if t.startswith('.workbuddy/') or t.startswith('.codebuddy/')]
junk = bak + wb
if junk:
    spec = os.path.join(ROOT, '.workbuddy', 'tmp_pathspec.txt')
    with io.open(spec, 'wb') as f:
        f.write(b'\x00'.join(t.encode('utf-8') for t in junk))
    r = git(['rm', '-q', '--cached', '--sparse', '--pathspec-from-file=' +
             spec.replace('\\', '/'), '--pathspec-file-nul', '--'])
    A('rm bak/workbuddy/codebuddy (%d): rc=%d %s' %
      (len(junk), r.returncode, r.stderr.decode('utf-8', 'replace')[:200]))

# ---- 2. add new platform artifacts ----
ADD_PATHS = [
    '.gitignore', '.gitattributes', 'README.md', 'AGENTS.md',
    'ai2050_research_os',
    'research/deepseek/atlas',
    'research/deepseek/docs/AGI_DEEPSEEK_MEMO.md',
]
for p in ADD_PATHS:
    r = git(['add', '--', p])
    A('add %-46s rc=%d %s' % (p, r.returncode, r.stderr.decode('utf-8', 'replace')[:120]))

# small contract/design json under tests/deepseek/result (skip big data)
contract_json = []
for pat in ('tests/deepseek/result/*.json', 'tests/deepseek/result/**/*.json'):
    contract_json += glob.glob(os.path.join(ROOT, pat.replace('/', os.sep)), recursive=True)
contract_json = sorted(set(contract_json))
small = [p for p in contract_json if os.path.getsize(p) <= 512 * 1024]
if small:
    r = git(['add', '--'] + [os.path.relpath(p, ROOT).replace('\\', '/') for p in small])
    A('add contracts (%d files): rc=%d %s' %
      (len(small), r.returncode, r.stderr.decode('utf-8', 'replace')[:120]))
A('contract jsons found=%d small=%d' % (len(contract_json), len(small)))

# ---- 3. final stats ----
r = git(['ls-files', '-z'])
tracked = [t.decode('utf-8', 'replace') for t in r.stdout.split(b'\x00') if t]
sz = 0
for t in tracked:
    p = os.path.join(ROOT, t.replace('/', os.sep))
    try:
        sz += os.path.getsize(p)
    except OSError:
        pass
A('FINAL tracked=%d files  bytes=%d (%s)' % (len(tracked), sz, human(sz)))

st = git(['status', '--porcelain', '-b'])
lines = st.stdout.decode('utf-8', 'replace').splitlines()
A('status lines=%d' % len(lines))
A('status head:')
for ln in lines[:6]:
    A('  ' + ln[:150])

# ensure .workbuddy not in index anymore
A('workbuddy_in_index=%s' % ('YES' if any(t.startswith('.workbuddy/') for t in tracked) else 'NO'))
A('bak_in_index=%s' % ('YES' if any('.bak_' in t for t in tracked) else 'NO'))

io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n')
print('FINALIZED')
