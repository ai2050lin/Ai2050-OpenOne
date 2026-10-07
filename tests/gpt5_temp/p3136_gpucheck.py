# -*- coding: utf-8 -*-
import io
import psutil
out = []
n = 0
for p in psutil.process_iter(['pid', 'name', 'cmdline']):
    try:
        cl = ' '.join(p.info['cmdline'] or [])
        nm = (p.info['name'] or '').lower()
        if 'python' in nm and ('phase31' in cl or 'phase30' in cl):
            out.append('KILL %d %s' % (p.info['pid'], cl[:120]))
            p.kill()
            n += 1
    except Exception:
        pass
out.append('killed %d' % n)
try:
    import torch
except Exception as e:
    out.append('torch import fail: %r' % e)
else:
    out.append('cuda_avail=%s' % torch.cuda.is_available())
    if torch.cuda.is_available():
        out.append('mem=%.1fMB/%.1fMB' % (torch.cuda.memory_allocated() / 1e6, torch.cuda.get_device_properties(0).total_memory / 1e6))
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3136_gpucheck.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('OK')
