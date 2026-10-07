# -*- coding: utf-8 -*-
"""Post-closeout disk verification."""
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
out = []

# 1. MEMO tail: Phase headers
memo = io.open(ROOT + r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md',
               encoding='utf-8', errors='ignore').read()
out.append('MEMO len=%d' % len(memo))
out.append('has P3101 header=%s' %
           ('## Phase 3101:' in memo))
out.append('has P3102 header=%s' %
           ('## Phase 3102:' in memo))
i1 = memo.find('## Phase 3101:')
out.append('P3101 head=%s' %
           memo[i1:i1 + 90].replace('\n', ' '))
i2 = memo.find('## Phase 3102:')
out.append('P3102 head=%s' %
           memo[i2:i2 + 80].replace('\n', ' '))
out.append('P3102 has 467/1792=%s' %
           ('467/1792' in memo[i2:]))
out.append('P3102 has table=%s' %
           ('| R53 |' in memo[i2:]))

# 2. Ledger meas3101
led = json.load(io.open(ROOT + r'\research\gpt5\atlas'
                        r'\atlas_ledger.json',
                        encoding='utf-8'))
m1 = [m for m in led['measurements']
      if m.get('phase') == 3101]
out.append('ledger n=%d meas3101=%d sha=%s'
           % (len(led['measurements']), len(m1),
              led.get('ledger_sha256_8')))
if m1:
    out.append('verdict=%s' % m1[0]['verdict'])
    out.append('npz8=%s' %
               m1[0]['hashes']['npz_sha256_8'])
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
out.append('l14 tail=%s' % l14['connects'][-1])

# 3. MEMORY.md
memw = io.open(ROOT + r'\.workbuddy\memory\MEMORY.md',
               encoding='utf-8').read()
out.append('MEMO_W len=%d max3102=%s'
           % (len(memw), 'max=3102' in memw))
out.append('MEMO_W has carrier_absent=%s'
           % ('seventh_carrier_absent' in memw))

# 4. wlog
wl = io.open(ROOT + r'\.workbuddy\memory'
             r'\2026-09-23.md',
             encoding='utf-8',
             errors='ignore').read()
out.append('wlog_D has P3101=%s len=%d'
           % ('Phase 3101' in wl, len(wl)))
wl_c = io.open(r'C:\Users\Admin\WorkBuddy'
               r'\2026-09-17-01-30-05\.workbuddy'
               r'\memory\2026-09-23.md',
               encoding='utf-8',
               errors='ignore').read()
out.append('wlog_C has P3101=%s' %
           ('Phase 3101' in wl_c))

with io.open(ROOT + r'\tests\gpt5_temp'
             r'\p3101_disk_verify.txt', 'w',
             encoding='utf-8') as f:
    f.write('\n'.join(out) + '\n')
print('OK')
