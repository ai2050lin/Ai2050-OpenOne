# -*- coding: utf-8 -*-
"""Kill leftover 3129 python processes
(excluding self and the killer itself)."""
import io
import os
import psutil

ME = os.getpid()
killed = []
for p in psutil.process_iter(
        ['pid', 'name', 'cmdline']):
    try:
        if p.info['pid'] == ME:
            continue
        j = ' '.join(str(x) for x in
                     (p.info['cmdline'] or []))
        if 'python' in (p.info['name']
                        or '').lower() \
                and ('3129_omega_p127' in j
                     or 'phase3129' in j):
            p.kill()
            killed.append('pid=%s'
                          % p.info['pid'])
    except Exception:
        pass
gpu = os.popen(
    r'"C:\Windows\System32\nvidia-smi.exe"'
    ' --query-gpu=memory.used'
    ' --format=csv,noheader').read().strip()
with io.open(r'D:\AI2050\Ai2050-OpenOne\tests'
             r'\gpt5_temp\p3129_kill.txt',
             'w') as f:
    f.write('killed=%s\ngpu=%s\n'
            % (killed or 'none', gpu))
print('KILL_DONE me=%d' % ME)
