# -*- coding: utf-8 -*-
"""Age + activity probe for the stale phase2751 rebuild processes."""
import datetime as dt
import io
import os

import psutil

OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3101_stale_probe.txt'
now = dt.datetime.now()
lines = []
for pid in (30012, 30656):
    try:
        p = psutil.Process(pid)
        ct = dt.datetime.fromtimestamp(p.create_time())
        cpu = p.cpu_times()
        lines.append(
            'pid=%d start=%s age=%.1fh cpu_total=%.1fs '
            'status=%s threads=%d mem=%.0fMB'
            % (pid, ct.strftime('%m-%d %H:%M'),
               (now - ct).total_seconds() / 3600.0,
               (cpu.user + cpu.system),
               p.status(), p.numthreads(),
               p.memory_info().rss / 1e6))
    except Exception as e:
        lines.append('pid=%d ERR %r' % (pid, e))
# recent writes under result dir for phase2751-ish artifacts
root = r'D:\AI2050\Ai2050-OpenOne\tests'
recent = []
cutoff = now - dt.timedelta(hours=6)
for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = [d for d in dirnames
                   if d not in ('node_modules',)]
    for fn in filenames:
        fp = os.path.join(dirpath, fn)
        try:
            mt = dt.datetime.fromtimestamp(
                os.path.getmtime(fp))
        except OSError:
            continue
        if mt >= cutoff and ('2751' in fn
                             or '2751' in dirpath):
            recent.append('%s  %s' % (
                mt.strftime('%m-%d %H:%M'), fp))
lines.append('recent 2751-tagged writes (6h): %d'
             % len(recent))
lines.extend(sorted(recent)[-20:])
with io.open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('OK')
