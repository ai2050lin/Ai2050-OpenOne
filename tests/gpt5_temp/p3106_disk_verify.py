# -*- coding: utf-8 -*-
"""Phase 3106 disk verification: confirm all five writes landed."""
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3106'
        r'\omega_p104_composition_dose_depth')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
WLOG_D = ROOT + r'\.workbuddy\memory\2026-09-23.md'
WLOG_C = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
          r'\.workbuddy\memory\2026-09-23.md')
MEMO_W = (ROOT + r'\.workbuddy\memory\MEMORY.md')

lines = []
ok_all = True


def check(name, cond, detail=''):
    global ok_all
    status = 'PASS' if cond else 'FAIL'
    if not cond:
        ok_all = False
    lines.append('[%s] %s %s' % (status, name, detail))


memo = io.open(MEMO, encoding='utf-8').read()
idx = memo.find('## Phase 3106:')
check('MEMO contains Phase 3106 section', idx > 0,
      'at char %d of %d' % (idx, len(memo)))
check('MEMO 3106 near tail', idx >= len(memo) - 6000,
      'starts %d from end' % (len(memo) - idx))
check('MEMO 3106 sha8 recorded',
      idx > 0 and 'material=1437bf61' in memo[idx:])
check('MEMO 3106 first-run invalidation logged',
      idx > 0 and 'a073c0a5' in memo[idx:])
check('MEMO 3106 frequency negation 1.000',
      idx > 0 and '1.000（29/29）' in memo[idx:])

led = json.load(io.open(LEDGER, encoding='utf-8'))
m3106 = [m for m in led['measurements']
         if m.get('phase') == 3106]
check('ledger has meas3106', len(m3106) == 1)
if m3106:
    c = m3106[0]['claim']
    check('claim has 0.930/0.958', '0.930 test' in c)
    check('claim has freq negation', '-9.76' in c)
    check('claim has passive +0.36', '+0.36' in c)
    check('claim has G4 L8', 'crit_obj|L8' in c)
    check('claim has Pearson 0.960', '0.960' in c)
    check('claim has superseded run',
          'a073c0a5' in c)
    check('claim NEXT 3107', 'NEXT 3107' in c)
check('ledger measurements n=243',
      len(led['measurements']) == 243,
      'n=%d' % len(led['measurements']))
l14 = [l for l in led['linkage']
       if l.get('link_id') == 'L14_readout_spectrum_cross_model']
check('L14 includes meas3106',
      len(l14) == 1 and any('meas3106' in x
                            for x in l14[0]['connects']),
      'len=%d' % (len(l14[0]['connects'])
                  if l14 else -1))

for tag, p in (('wlog D', WLOG_D), ('wlog C', WLOG_C)):
    try:
        w = io.open(p, encoding='utf-8').read()
        check('%s has Phase 3106 entry' % tag,
              'Phase 3106 Omega-P104' in w)
        check('%s records sha8' % tag,
              '1437bf61' in w)
    except IOError as e:
        check('%s readable' % tag, False, repr(e))

mem = io.open(MEMO_W, encoding='utf-8').read()
check('MEMORY max=3106', 'max=3106' in mem)
check('MEMORY no max=3105 stale', 'max=3105' not in mem)
check('MEMORY next=3107', '下一 3107' in mem)
check('MEMORY 3106 block', '机制链状态（3106）' in mem)
lines.append('MEMORY.md len = %d' % len(mem))

for fn, minsize in (('result.json', 500),
                    ('material.json', 100000),
                    ('capture.npz', 100000000),
                    ('design_seal.json', 200),
                    ('closeout_log.txt', 100),
                    ('run_log.txt', 100)):
    p = OUTD + '\\' + fn
    try:
        sz = os.path.getsize(p)
        check('artifact %s' % fn, sz >= minsize,
              'size=%d' % sz)
    except OSError as e:
        check('artifact %s' % fn, False, repr(e))

lines.append('')
lines.append('ALL PASS' if ok_all else 'SOME CHECKS FAILED')
io.open(ROOT + r'\tests\gpt5_temp\p3106_disk_verify_out.txt',
        'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('verify done')
