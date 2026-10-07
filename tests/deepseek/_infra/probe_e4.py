import os, time, hashlib, subprocess, glob
root = r'D:\AI2050\Ai2050-OpenOne'
out = []
out.append('now %s' % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
try:
    import torch
    out.append('torch %s cuda %s' % (torch.__version__, torch.cuda.is_available()))
    if torch.cuda.is_available():
        free, tot = torch.cuda.mem_get_info()
        out.append('gpu free %.2f GB / total %.2f GB' % (free / 1e9, tot / 1e9))
except Exception as e:
    out.append('torch fail %r' % e)
try:
    r = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'],
                       capture_output=True, text=True, timeout=90)
    lines = [l for l in (r.stdout or '').splitlines() if 'python' in l.lower()]
    out.append('python procs %d' % len(lines))
    for l in lines[:12]:
        out.append('   ' + l[:170])
except Exception as e:
    out.append('tasklist fail %r' % e)

p = os.path.join(root, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
b = open(p, 'rb').read()
out.append('memo bytes %d sha8 %s' % (len(b), hashlib.sha256(b).hexdigest()[:8]))
out.append('has BOM %s' % b.startswith(b'\xef\xbb\xbf'))
out.append('CRLF count %d  LF-only %d' % (b.count(b'\r\n'), b.count(b'\n') - b.count(b'\r\n')))
T = b.decode('utf-8-sig')
for h in ['## Phase 3', '## Phase 4', '## Phase 5', '## Phase 6']:
    out.append('  hdr %-14s -> %d' % (h, T.count(h)))
L = T.splitlines()
out.append('lines %d' % len(L))
out.append('last_line %r' % L[-1][:120])

d = os.path.join(root, 'tests', 'gpt5_temp')
out.append('--- n2h1 artifacts ---')
for pat in ['n2h1*.py', 'n2h1*.txt', 'N2h1*.json', 'n2h1c*.txt', 'n2h1b*.txt']:
    for q in sorted(glob.glob(os.path.join(d, pat))):
        bb = open(q, 'rb').read()
        out.append('  %-44s %7d B %s' % (os.path.basename(q), len(bb), hashlib.sha256(bb).hexdigest()[:8]))
out.append('--- models ---')
h = os.path.join(root, 'models', 'hf')
if os.path.isdir(h):
    for m in sorted(os.listdir(h)):
        out.append('  ' + m)

open(os.path.join(root, 'gpt5_temp', 'probe_e4.txt'), 'w', encoding='utf-8').write('\n'.join(out))
print('written probe_e4.txt')
