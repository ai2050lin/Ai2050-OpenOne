# -*- coding: utf-8 -*-
"""Build a clean single-commit release repo from current HEAD."""
import io, os, subprocess, tarfile, shutil

ROOT = r'D:\AI2050\Ai2050-OpenOne'
REL = r'D:\AI2050\Ai2050-OpenOne-publish'
TAR = os.path.join(ROOT, '.workbuddy', 'tmp_release_tree.tar')
OUT = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\tmp_release_build.txt'

def git(args, cwd=None, **kw):
    return subprocess.run(['git', '-C', cwd or ROOT] + args,
                          capture_output=True, **kw)

def human(n):
    for u in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or u == 'GB':
            return '%.1f%s' % (n, u)
        n /= 1024.0

L = []
A = L.append

# 1. export tree from HEAD (no pipes; archive to file)
r = git(['archive', '--format=tar', '-o', TAR, 'HEAD'])
A('archive rc=%d %s' % (r.returncode, r.stderr.decode('utf-8', 'replace')[:200]))

# 2. fresh release dir
if os.path.exists(REL):
    A('release dir exists -> wiping content')
    shutil.rmtree(REL, ignore_errors=True)
os.makedirs(REL, exist_ok=True)
with tarfile.open(TAR) as tf:
    tf.extractall(REL)  # noqa: S202 - trusted local archive
n_files = sum(len(fs) for _, _, fs in os.walk(REL))
A('extracted files=%d' % n_files)

# 3. git identity from source repo (fallback fixed)
un = git(['config', '--get', 'user.name']).stdout.decode().strip() or 'AI2050 Research'
ue = git(['config', '--get', 'user.email']).stdout.decode().strip() or 'ai2050@research.local'
A('identity: %s <%s>' % (un, ue))

# 4. init + single commit
r = git(['init', '-b', 'main'], cwd=REL)
A('init rc=%d %s' % (r.returncode, r.stderr.decode('utf-8', 'replace')[:150]))
r = git(['add', '-A'], cwd=REL)
A('add rc=%d %s' % (r.returncode, r.stderr.decode('utf-8', 'replace')[:150]))
r = git(['-c', 'user.name=%s' % un, '-c', 'user.email=%s' % ue,
         'commit', '-q', '-m',
         'AI2050 Open Mechanistic-Interpretability Learning Platform - release v1',
         ], cwd=REL)
A('commit rc=%d %s' % (r.returncode, r.stderr.decode('utf-8', 'replace')[:150]))

# 5. aggressive gc -> measure pack
r = git(['gc', '--aggressive', '--prune=now'], cwd=REL)
A('gc rc=%d' % r.returncode)
r = git(['count-objects', '-v'], cwd=REL)
A('count-objects:')
for line in r.stdout.decode('utf-8', 'replace').strip().splitlines():
    A('  ' + line)

# 6. verify critical assets present in release tree
CHECKS = [
    'README.md', 'AGENTS.md', '.gitignore', '.gitattributes',
    'ai2050_research_os/README.md',
    'ai2050_research_os/registry/industry.json',
    'ai2050_research_os/registry/cases.json',
    'ai2050_research_os/registry/visualization_specs.json',
    'ai2050_research_os/schemas/snapshot.v2.schema.json',
    'ai2050_research_os/config/industry_sources.json',
    'ai2050_research_os/scripts/fetch_industry.py',
    'ai2050_research_os/scripts/researchctl.py',
    'research/deepseek/docs/AGI_DEEPSEEK_MEMO.md',
    'research/gpt5/docs/AGI_GPT5_MEMO.md',
    'research/glm5/docs/AGI_GLM5_MEMO.md',
    'research/deepseek/atlas/phase_queue_v1.json',
    'research/deepseek/atlas/metric_dict.json',
    'tests/deepseek/result/q06_prereg_design_v1.json',
    'data/iso_corpus.jsonl',
    'frontend/package.json', 'frontend/vite.config.js',
    'server/server.py',
]
A('--- presence checks ---')
miss = 0
for c in CHECKS:
    ok = os.path.exists(os.path.join(REL, c.replace('/', os.sep)))
    miss += 0 if ok else 1
    A('%s %s' % ('OK ' if ok else 'MISS', c))
A('missing=%d' % miss)

# negative checks: heavy stuff must be absent
NEG = [
    'node_modules', 'tests/result', 'data/MNIST', 'tests/nfb_data',
    'frontend/node_modules',
]
A('--- absence checks ---')
for c in NEG:
    A('%s %s' % ('PRESENT(BAD)' if os.path.exists(os.path.join(REL, c.replace('/', os.sep))) else 'absent', c))

# big files in release tree
big = []
for dp, dn, fn in os.walk(REL):
    dn[:] = [d for d in dn if d != '.git']
    for f in fn:
        p = os.path.join(dp, f)
        try:
            sz = os.path.getsize(p)
        except OSError:
            continue
        if sz > 2 * 1024 * 1024:
            big.append((sz, os.path.relpath(p, REL)))
A('--- files >2MB in release tree: %d ---' % len(big))
for sz, p in sorted(big, key=lambda x: -x[0])[:15]:
    A('%10s  %s' % (human(sz), p))
A('release tree bytes (excl .git): %s' % human(
    sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, fn in os.walk(REL)
        if '.git' not in dp for f in fn)))

# .git dir size
gd = os.path.join(REL, '.git')
gs = sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, fn in os.walk(gd) for f in fn)
A('release .git size: %s' % human(gs))

io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n')
print('RELEASE_BUILT')
