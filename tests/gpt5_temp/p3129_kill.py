# -*- coding: utf-8 -*-
"""Kill leftover 3129 python processes."""
import io
import os
import psutil

killed = []
for p in psutil.process_iter(
        ['pid', 'name', 'cmdline']):
    try:
        j = ' '.join(str(x) for x in
                     (p.info['cmdline'] or []))
        if 'python' in (p.info['name']
                        or '').lower() \
                and '3129' in j:
            p.kill()
            killed.append('pid=%s'
                          % p.info['pid'])
    except Exception:
        pass
gone = []
for pid in killed:
    try:
        psutil.Process(int(pid.split('=')[1])
                       ).wait(5)
        gone.append(pid)
    except Exception:
        pass
gpu = os.popen(
    r'"C:\Windows\System32\nvidia-smi.exe"'
    ' --query-gpu=memory.used'
    ' --format=csv,noheader').read().strip()
with io.open(r'D:\AI2050\Ai2050-OpenOne\tests'
             r'\gpt5_temp\p3129_kill.txt',
             'w') as f:
    f.write('killed=%s\nall_gone=%s\ngpu=%s\n'
            % (killed or 'none',
               len(gone) == len(killed)
               and killed != [], gpu))
print('KILL_DONE')
