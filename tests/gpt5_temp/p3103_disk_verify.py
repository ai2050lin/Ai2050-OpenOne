# -*- coding: utf-8 -*-
"""Final disk verification for Phase 3103."""
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
out = []

memo = io.open(ROOT + r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md',
               encoding='utf-8',
               errors='ignore').read()
i3 = memo.find('## Phase 3103:')
out.append('MEMO len=%d P3101=%s P3102=%s P3103=%s'
           % (len(memo),
              '## Phase 3101:' in memo,
              '## Phase 3102:' in memo,
              i3 > 0))
seg = memo[i3:]
checks = ['A5/B34/C14/D21/E20', 'PA-01', 'PA-05',
          'add57ba7', 'consumer_grep', '3076',
          '待填充研究接口', '3104']
for c in checks:
    out.append('P3103 has [%s]=%s' % (c, c in seg))
out.append('P3103 len=%d' % len(seg))
out.append('P3103 tail=%s'
           % seg[-120:].replace('\n', ' | '))

led = json.load(io.open(ROOT + r'\research\gpt5\atlas'
                        r'\atlas_ledger.json',
                        encoding='utf-8'))
m3 = [m for m in led['measurements']
      if m.get('phase') == 3103]
out.append('ledger n=%d meas3103=%d sha=%s'
           % (len(led['measurements']), len(m3),
              led.get('ledger_sha256_8')))
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
out.append('l14 tail2=%s' % ' | '.join(
    l14['connects'][-2:]))

memw = io.open(ROOT + r'\.workbuddy\memory\MEMORY.md',
               encoding='utf-8').read()
out.append('MEMO_W len=%d max3103=%s has2103sec=%s'
           % (len(memw), 'max=3103' in memw,
              '命题账本 62' in memw))

for tag, p in (('D', ROOT + r'\.workbuddy\memory'
                r'\2026-09-23.md'),
               ('C', r'C:\Users\Admin\WorkBuddy'
                     r'\2026-09-17-01-30-05'
                     r'\.workbuddy\memory'
                     r'\2026-09-23.md')):
    wl = io.open(p, encoding='utf-8',
                 errors='ignore').read()
    out.append('wlog_%s has P3103Omega=%s'
               % (tag, 'Phase 3103 Omega-P101' in wl))

ld = json.load(io.open(
    ROOT + r'\tests\glm5\result'
    r'\rdc_query_construction_20260913'
    r'\phase3103\omega_p101_formula_audit'
    r'\proposition_ledger.json',
    encoding='utf-8'))
out.append('artifact: props R=%d PA=%d grades=%s'
           % (len(ld['propositions_review']),
              len(ld['propositions_new']),
              ld['grade_distribution']))

with io.open(ROOT + r'\tests\gpt5_temp'
             r'\p3103_disk_verify.txt', 'w',
             encoding='utf-8') as f:
    f.write('\n'.join(out) + '\n')
print('OK')
