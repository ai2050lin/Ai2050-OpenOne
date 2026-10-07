import os, hashlib, json

root = r'D:\AI2050\Ai2050-OpenOne'
src = os.path.join(root, 'tests', 'gpt5_temp')
out = []

out.append('src exists %s' % os.path.isdir(src))
out.append('tests/deepseek exists %s' % os.path.isdir(os.path.join(root, 'tests', 'deepseek')))
out.append('tests/deepseek_temp exists %s' % os.path.isdir(os.path.join(root, 'tests', 'deepseek_temp')))
out.append('tests dirs: ' + ', '.join(sorted([d for d in os.listdir(os.path.join(root, 'tests')) if os.path.isdir(os.path.join(root,'tests',d))])))
out.append('')

# files in src
files = [f for f in sorted(os.listdir(src)) if os.path.isfile(os.path.join(src, f))]
dirs = [d for d in sorted(os.listdir(src)) if os.path.isdir(os.path.join(src, d))]
out.append('src files %d dirs %d' % (len(files), len(dirs)))
out.append('--- dirs ---')
for d in dirs:
    p = os.path.join(src, d)
    n = 0; sz = 0
    for dp, dn, fn in os.walk(p):
        for f in fn:
            n += 1; sz += os.path.getsize(os.path.join(dp, f))
    out.append('  D %-40s files=%d bytes=%d' % (d, n, sz))

# classify N-series artifacts
import re
PATS = [
    ('E1', r'^e1_'), ('E2', r'^e2_'), ('E3', r'^e3'), ('N1', r'^n1'), ('N2', r'^n2'),
    ('N3', r'^n3'), ('probe', r'^probe_'), ('seal', r'^(N1|N2h1|N3)_design_seal'),
    ('hash', r'^hash_'), ('do_', r'^do_'), ('memo_append', r'^memo_append'),
    ('wlog', r'^wlog_'), ('memory', r'^memory_'), ('verify', r'^verify_'),
    ('review', r'^memo_review'),
]
buckets = {k: [] for k, _ in PATS}
other = []
for f in files:
    hit = None
    for k, pat in PATS:
        if re.search(pat, f, re.I):
            hit = k; break
    if hit:
        buckets[hit].append(f)
    else:
        other.append(f)

out.append('')
out.append('--- classified N-series artifacts ---')
for k, _ in PATS:
    if buckets[k]:
        out.append('[%s] %d' % (k, len(buckets[k])))
        for f in buckets[k]:
            out.append('    %-52s %8d' % (f, os.path.getsize(os.path.join(src, f))))
out.append('[OTHER] %d' % len(other))
for f in other:
    out.append('    %-52s %8d' % (f, os.path.getsize(os.path.join(src, f))))

open(os.path.join(root, 'gpt5_temp', 'inv_products.txt'), 'w', encoding='utf-8').write('\n'.join(out))
print('ok', len(files))
