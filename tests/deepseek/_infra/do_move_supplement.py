import os, re, hashlib, shutil, time

root = r'D:\AI2050\Ai2050-OpenOne'
T = os.path.join(root, 'tests', 'gpt5_temp')
DST_TEMP = os.path.join(root, 'tests', 'deepseek_temp')

pat = re.compile(r'^(e[123]b?|n[123])[\w\-\.]*\.(py|txt)$')
pats = [r'^e[123]b?_[\w\-\.]*\.(py|txt)$', r'^n[123][\w\-\.]*\.(py|txt)$']

def collect(d):
    hits = []
    for f in sorted(os.listdir(d)):
        p = os.path.join(d, f)
        if os.path.isfile(p) and any(re.search(x, f) for x in pats):
            hits.append(f)
    return hits

todo = collect(T)
print('remaining in tests/gpt5_temp:', todo)

def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()

moved = []
for f in todo:
    src = os.path.join(T, f)
    dst = os.path.join(DST_TEMP, f)
    if os.path.exists(dst):
        print('SKIP exists', f); continue
    h0 = sha(src); n0 = os.path.getsize(src)
    shutil.move(src, dst)
    h1 = sha(dst)
    moved.append((f, n0, h0 == h1, not os.path.exists(src)))
    print('moved %-52s %7d hash_ok=%s' % (f, n0, h0 == h1))

lines = ['# 补搬（含点号文件名的报告） %s' % time.strftime('%Y-%m-%d %H:%M:%S')]
for f, n, h, g in moved:
    lines.append('%-52s %7d hash_ok=%s src_gone=%s' % (f, n, h, g))
lines.append('total %d ; all_ok %s' % (len(moved), all(h and g for _, _, h, g in moved)))
open(os.path.join(root, 'gpt5_temp', 'move_manifest_phase1_7_supplement.txt'), 'w', encoding='utf-8').write('\n'.join(lines))
print('done', len(moved))
