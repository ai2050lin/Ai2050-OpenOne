# -*- coding: utf-8 -*-
"""Probe the foreign python process holding GPU memory (PID 30656)."""
import io
import json

import psutil

OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3101_gpu_proc_probe.txt'
rows = []
for p in psutil.process_iter(['pid', 'name', 'cmdline',
                              'create_time', 'memory_info']):
    try:
        nm = (p.info['name'] or '').lower()
        if 'python' in nm:
            cl = ' '.join(p.info['cmdline'] or [])[:400]
            rows.append({'pid': p.info['pid'],
                         'name': p.info['name'],
                         'cmdline': cl})
    except Exception:
        pass
with io.open(OUT, 'w', encoding='utf-8') as f:
    f.write('python processes:\n')
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print('WROTE', len(rows))
