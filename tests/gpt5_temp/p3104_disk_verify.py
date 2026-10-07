# -*- coding: utf-8 -*-
"""3104 disk verification (real-disk probe)."""
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
checks = {}

# 1. MEMO tail
memo = io.open(ROOT + r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md',
               encoding='utf-8').read()
checks['memo_len'] = len(memo)
checks['memo_has_3104'] = '## Phase 3104:' in memo
i = memo.rindex('## Phase 3104:')
checks['memo_3104_title'] = memo[i:i + 120].splitlines()[0]
checks['memo_3104_has_verdict'] = (
    'K3 真值门不适定' in memo[i:])
checks['memo_ends_3104'] = memo.rstrip().endswith(
    'bit 级确定性通过。')

# 2. ledger
led = json.load(io.open(
    ROOT + r'\research\gpt5\atlas\atlas_ledger.json',
    encoding='utf-8'))
checks['ledger_n'] = len(led['measurements'])
checks['ledger_sha'] = led.get('ledger_sha256_8')
m4 = [m for m in led['measurements']
      if m.get('phase') == 3104]
checks['ledger_meas3104'] = len(m4) == 1
checks['ledger_meas3104_id'] = (
    m4[0]['meas_id'] if m4 else None)
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
checks['l14_tail'] = l14['connects'][-1]

# 3. wlogs both disks
for tag, wl in (('wlog_D', ROOT + r'\.workbuddy'
                 r'\memory\2026-09-23.md'),
                ('wlog_C', r'C:\Users\Admin'
                 r'\WorkBuddy\2026-09-17-01-30-05'
                 r'\.workbuddy\memory'
                 r'\2026-09-23.md')):
    t = io.open(wl, encoding='utf-8').read()
    checks[tag] = 'Phase 3104 Omega-P102' in t

# 4. MEMORY.md
memw = io.open(ROOT + r'\.workbuddy\memory\MEMORY.md',
               encoding='utf-8').read()
checks['memw_max3104'] = 'max=3104' in memw
checks['memw_len'] = len(memw)
checks['memw_next3105'] = '下一 3105' in memw

# 5. artifacts on disk
import os
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3104'
        r'\omega_p102_relation_vs_endpoint')
for f in ('result.json', 'material.json',
          'design_seal.json', 'confound_check.json',
          'capture.npz', 'run_log.txt',
          'closeout_log.txt'):
    checks['file_' + f] = os.path.getsize(
        os.path.join(OUTD, f))
smoke = os.path.join(OUTD, 'smoke')
checks['smoke_files'] = sorted(
    os.listdir(smoke))[:8]

with io.open(ROOT + r'\tests\gpt5_temp'
             r'\p3104_disk_verify_out.txt', 'w',
             encoding='utf-8') as f:
    json.dump(checks, f, indent=1,
              ensure_ascii=False)
print('verify done')
