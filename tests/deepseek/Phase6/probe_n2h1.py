import os, time, json, glob, subprocess

root = r'D:\AI2050\Ai2050-OpenOne'
out = []
out.append('now %s' % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))

# GPU
try:
    r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
                        '--format=csv,noheader'], capture_output=True, text=True, timeout=30)
    out.append('gpu: ' + (r.stdout or r.stderr).strip())
except Exception as e:
    out.append('gpu query fail %r' % e)

# python procs
try:
    import psutil
    ps = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cl = ' '.join(p.info.get('cmdline') or [])
            if 'python' in (p.info.get('name') or '').lower() and 'phase' in cl.lower():
                ps.append('pid=%s %s' % (p.info['pid'], cl[:150]))
        except Exception:
            pass
    out.append('python phase procs %d' % len(ps))
    for s in ps[:8]:
        out.append('  ' + s)
except Exception as e:
    out.append('psutil fail %r' % e)

# target memo
memo = os.path.join(root, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
if os.path.exists(memo):
    b = open(memo, 'rb').read()
    out.append('memo bytes %d' % len(b))
    T = b.decode('utf-8-sig')
    out.append('memo lines %d' % (T.count(chr(10)) + 1))
    hdrs = [l for l in T.splitlines() if l.startswith('## ')]
    out.append('memo hdrs: ' + ' | '.join(h[:60] for h in hdrs))
else:
    out.append('memo MISSING')

# n2 artifacts present
d = os.path.join(root, 'tests', 'gpt5_temp')
for pat in ['n2c_*', 'n2d_*', 'n2g_*']:
    fs = sorted(glob.glob(os.path.join(d, pat)))
    out.append('%s -> %s' % (pat, ', '.join(os.path.basename(f) for f in fs)))

# models available
h = os.path.join(root, 'models', 'hf')
if os.path.isdir(h):
    out.append('models: ' + ', '.join(sorted(os.listdir(h))))

open(os.path.join(root, 'gpt5_temp', 'probe_n2h1.txt'), 'w', encoding='utf-8').write(chr(10).join(out))
print('ok')
