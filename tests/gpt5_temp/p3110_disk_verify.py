# -*- coding: utf-8 -*-
"""Phase 3110 disk verification: confirm all five writes landed."""
import hashlib
import io
import json
import os
import re

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3110'
        r'\omega_p108_ksweep_minport')
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
idx = memo.find('## Phase 3110:')
check('MEMO contains Phase 3110 section', idx > 0,
      'at char %d of %d' % (idx, len(memo)))
check('MEMO 3110 near tail', idx >= len(memo) - 6000,
      'starts %d from end' % (len(memo) - idx))
tail = memo[idx:] if idx > 0 else ''
check('MEMO 3110 d_min 5', 'd_min = 5' in tail)
check('MEMO 3110 K5 AUC 0.9942', '0.9942' in tail)
check('MEMO 3110 rel K5 0.6014', '0.6014' in tail)
check('MEMO 3110 d_min rel 400', 'd_min = **400**'
      in tail)
check('MEMO 3110 specificity',
      'truth_specific_diffuseness' in tail)
check('MEMO 3110 broadcast candidate',
      '全局一阶矩信号' in tail)
check('MEMO 3111 prereg', '3111 预注册' in tail)

led = json.load(io.open(LEDGER, encoding='utf-8'))
m3110 = [m for m in led['measurements']
         if m.get('phase') == 3110]
check('ledger has meas3110', len(m3110) == 1)
if m3110:
    c = m3110[0]['claim']
    check('claim verdict', 'sharp_core' in c
          and 'truth_specific_diffuseness' in c)
    check('claim d_min_truth 5', 'd_min_truth = 5' in c)
    check('claim K5 0.9942', '0.9942' in c)
    check('claim rel 0.6014', '0.6014' in c)
    check('claim d_min_rel 400', 'd_min_rel = 400' in c)
    check('claim 80x', '80x' in c)
    check('claim first-moment candidate',
          'first-moment' in c)
    check('claim NEXT 3111', 'NEXT 3111' in c)
check('ledger measurements n=247',
      len(led['measurements']) == 247,
      'n=%d' % len(led['measurements']))
l14 = [l for l in led['linkage']
       if l.get('link_id') == 'L14_readout_spectrum_cross_model']
check('L14 includes meas3110',
      len(l14) == 1 and any('meas3110' in x
                            for x in l14[0]['connects']),
      'len=%d' % (len(l14[0]['connects'])
                  if l14 else -1))
sha_log = io.open(OUTD + r'\closeout_log.txt',
                  encoding='utf-8').read()
saved = led.pop('ledger_sha256_8', None)
blob = json.dumps(led, sort_keys=True,
                  ensure_ascii=False)
sha8 = hashlib.sha256(
    blob.encode('utf-8')).hexdigest()[:8]
mm = re.search(r'sha=([0-9a-f]{8})', sha_log)
check('ledger sha8 matches closeout log',
      mm is not None and sha8 == mm.group(1),
      'computed=%s log=%s'
      % (sha8, mm.group(1) if mm else '?'))

for tag, p in (('wlog D', WLOG_D), ('wlog C', WLOG_C)):
    try:
        w = io.open(p, encoding='utf-8').read()
        check('%s has Phase 3110 entry' % tag,
              'Phase 3110 Omega-P108' in w)
        check('%s records d_min=5' % tag,
              'd_min=5' in w)
    except IOError as e:
        check('%s readable' % tag, False, repr(e))

mem = io.open(MEMO_W, encoding='utf-8').read()
check('MEMORY max=3110', 'max=3110' in mem)
check('MEMORY no max=3109 stale', 'max=3109' not in mem)
check('MEMORY next=3111', '下一 3111' in mem)
check('MEMORY 3110 block', '机制链状态（3110）' in mem)
check('MEMORY no next-3110 stale',
      '下一 3110' not in mem)
lines.append('MEMORY.md len = %d' % len(mem))

for fn, minsize in (('result.json', 1500),
                    ('design_seal.json', 500),
                    ('run_log.txt', 800),
                    ('closeout_log.txt', 100)):
    p = OUTD + '\\' + fn
    try:
        sz = os.path.getsize(p)
        check('artifact %s' % fn, sz >= minsize,
              'size=%d' % sz)
    except OSError as e:
        check('artifact %s' % fn, False, repr(e))

lines.append('')
lines.append('ALL PASS' if ok_all else 'SOME CHECKS FAILED')
io.open(ROOT + r'\tests\gpt5_temp\p3110_disk_verify_out.txt',
        'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('verify done')
