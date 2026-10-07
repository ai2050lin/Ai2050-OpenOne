# -*- coding: utf-8 -*-
"""GPU pre-check for Phase 3129: free VRAM
+ leftover python processes. Output to file."""
import io
import os

out = []

try:
    import psutil
    for p in psutil.process_iter(
            ['pid', 'name', 'cmdline']):
        try:
            cl = p.info['cmdline'] or []
            j = ' '.join(str(x) for x in cl)
            if 'python' in (p.info['name']
                            or '').lower() \
                    and '3128' in j:
                out.append('LEFTOVER: pid=%s %s'
                           % (p.info['pid'], j[:120]))
        except Exception:
            pass
except Exception as e:
    out.append('psutil err: %r' % e)

r = os.popen(
    r'"C:\Windows\System32\nvidia-smi.exe"'
    ' --query-gpu=memory.used,memory.total'
    ' --format=csv').read()
out.append('nvidia-smi: ' + r.strip())

with io.open(
        r'D:\AI2050\Ai2050-OpenOne\tests'
        r'\gpt5_temp\p3129_gpu_check.txt',
        'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
print('GPUCHECK_DONE')
