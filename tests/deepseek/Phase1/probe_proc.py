# -*- coding: utf-8 -*-
import os, time, subprocess, json
root = r'D:\AI2050\Ai2050-OpenOne'
out = []
def w(s): out.append(str(s))
w('time %s' % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
try:
    import psutil
    rows = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            cl = p.info['cmdline'] or []
            s = ' '.join(cl)
            if 'python' in (p.info['name'] or '').lower():
                rows.append('pid=%s rss=%.0fMB start=%s cmd=%s' % (
                    p.info['pid'], p.memory_info().rss / 1e6,
                    time.strftime('%H:%M:%S', time.localtime(p.info['create_time'])),
                    s[:200]))
        except Exception:
            pass
    w('python procs %d' % len(rows))
    for r in rows: w('  ' + r)
except Exception as e:
    w('psutil fail %r' % e)
# 3149 phase dir state
d = os.path.join(root, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913', 'phase3149')
for dp, dn, fn in os.walk(d):
    for f in fn:
        p = os.path.join(dp, f)
        st = os.stat(p)
        if f in ('result.json', 'design_seal.json', 'run_log.txt'):
            w('3149 %s bytes=%d mtime=%s' % (f, st.st_size, time.strftime('%m-%d %H:%M:%S', time.localtime(st.st_mtime))))
# check phase3150 dir
d50 = os.path.join(root, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913', 'phase3150')
w('phase3150 dir exists %s' % os.path.isdir(d50))
if os.path.isdir(d50):
    for dp, dn, fn in os.walk(d50):
        for f in fn: w('  3150 ' + f)
open(os.path.join(root, 'gpt5_temp', 'probe_proc.txt'), 'w', encoding='utf-8').write('\n'.join(out))
print('ok')
