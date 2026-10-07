# -*- coding: utf-8 -*-
import os, re, json, time, glob, subprocess
root = r'D:\AI2050\Ai2050-OpenOne'
out = []
def w(s): out.append(str(s))

w('=== time ===')
w(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

w('=== memo ===')
p = os.path.join(root, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
T = open(p, encoding='utf-8').read()
L = T.splitlines()
w('bytes %d lines %d' % (len(T.encode('utf-8')), len(L)))
w('tail 10:')
for l in L[-10:]:
    w('  ' + l[:170])
w('recent ## headers:')
hs = [(i + 1, l) for i, l in enumerate(L) if l.startswith('## ')]
for i, l in hs[-8:]:
    w('  L%d %s' % (i, l[:130]))

w('=== models/hf ===')
h = os.path.join(root, 'models', 'hf')
if os.path.isdir(h):
    for d in sorted(os.listdir(h)):
        dd = os.path.join(h, d)
        if os.path.isdir(dd):
            sz = 0
            for dp, dn, fn in os.walk(dd):
                for f in fn:
                    try: sz += os.path.getsize(os.path.join(dp, f))
                    except: pass
            w('  %s %.2f GB' % (d, sz / 1e9))
else:
    w('  MISSING')

w('=== e2 report ===')
p2 = os.path.join(root, 'tests', 'gpt5_temp', 'e2_report.txt')
if os.path.exists(p2):
    w(open(p2, encoding='utf-8').read()[:3200])
else:
    w('  MISSING')

w('=== gpu ===')
for exe in [r'C:\Windows\System32\nvidia-smi.exe', 'nvidia-smi']:
    try:
        r = subprocess.run([exe, '--query-gpu=name,memory.total,memory.used', '--format=csv'], capture_output=True, text=True, timeout=30)
        w(r.stdout.strip()); w('err:' + r.stderr.strip()[:200])
        break
    except Exception as e:
        w('  fail %s %r' % (exe, e))

w('=== 3149 script load pattern ===')
cands = sorted(glob.glob(os.path.join(root, 'tests', 'glm5', 'phase3149*.py')))
w('scripts %s' % cands)
for c in cands[:1]:
    txt = open(c, encoding='utf-8', errors='replace').read()
    w('  lines %d' % txt.count(chr(10)))
    for l in txt.splitlines():
        if re.search(r'from_pretrained|device_map|torch_dtype|dtype=|bfloat|float16|load_in|MODEL|model_dir|AutoModel', l):
            w('   ' + l.strip()[:170])

w('=== existing n1/e3 scripts ===')
for pat in ['*n1*', '*e3*', '*embed*audit*']:
    w('  %s -> %s' % (pat, glob.glob(os.path.join(root, 'tests', 'gpt5_temp', pat))))

open(os.path.join(root, 'gpt5_temp', 'probe_p1.txt'), 'w', encoding='utf-8').write('\n'.join(out))
print('ok')
