# -*- coding: utf-8 -*-
"""p3086 disk recheck: confirm all closeout
writes landed on real disk."""
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913'
     r'\phase3086\omega_p83_continuum_test')
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3086_disk_check.txt'
o = []

# 1. MEMO tail
p = ROOT + r'\research\gpt5\docs\AGI_GPT5_MEMO.md'
t = io.open(p, encoding='utf-8').read()
i = t.rfind('## Phase 3086:')
o.append('MEMO len=%d phase3086_at=%d '
         'menu3087=%s tail_ok=%s'
         % (len(t), i,
            '接续 3087 菜单' in t,
            t.rstrip().endswith('3087 A.')))

# 2. audit
p = (ROOT + r'\research\gpt5\docs'
     r'\hdmcc_knowledge_map_review_20260921.md')
t = io.open(p, encoding='utf-8').read()
o.append('AUDIT len=%d 48=%s rho=%s'
         % (len(t), '## 四十八、3086' in t,
            '+0.867' in t))

# 3. wlog
p = ROOT + r'\.workbuddy\memory\2026-09-22.md'
t = io.open(p, encoding='utf-8').read()
o.append('WLOG len=%d p3086=%s'
         % (len(t), 'Phase 3086' in t))

# 4. MEMORY
p = ROOT + r'\.workbuddy\memory\MEMORY.md'
t = io.open(p, encoding='utf-8').read()
o.append('MEMORY len=%d max3086=%s'
         % (len(t), 'max=3086' in t))

# 5. ledger
p = (ROOT + r'\research\gpt5\atlas'
     r'\atlas_ledger.json')
led = json.load(io.open(p, encoding='utf-8'))
o.append('LEDGER n=%d sha=%s p3086=%s'
         % (len(led['measurements']),
            led.get('ledger_sha256_8'),
            any(isinstance(m, dict)
                and m.get('phase') == 3086
                for m in
                led['measurements'])))

# 6. artifacts
for sub in ('', r'\smoke'):
    d = R + sub
    fs = sorted(os.listdir(d))
    o.append('DIR %s: %d files: %s'
             % (sub or '.\\', len(fs),
                ','.join(fs)))

# 7. scripts exist
for s in (r'\tests\glm5'
          r'\phase3086_omega_p83_continuum_'
          r'test.py',
          r'\tests\gpt5_temp'
          r'\phase3086_closeout.py',
          r'\tests\gpt5_temp'
          r'\phase3086_verify.py'):
    o.append('SCRIPT %s exists=%s size=%d'
             % (s.split('\\')[-1],
                os.path.exists(ROOT + s),
                os.path.getsize(ROOT + s)))

io.open(OUT, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
