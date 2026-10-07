# -*- coding: utf-8 -*-
"""Phase 3109 disk verification: confirm all five writes landed."""
import hashlib
import io
import json
import os
import re

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3109'
        r'\omega_p107_underdetermination_verdict')
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
idx = memo.find('## Phase 3109:')
check('MEMO contains Phase 3109 section', idx > 0,
      'at char %d of %d' % (idx, len(memo)))
check('MEMO 3109 near tail', idx >= len(memo) - 6000,
      'starts %d from end' % (len(memo) - idx))
tail = memo[idx:] if idx > 0 else ''
check('MEMO 3109 verdict',
      'stability_not_n_limited' in tail)
check('MEMO 3109 floor 0.0194', '0.0194' in tail)
check('MEMO 3109 rho -1.000',
      '-1.000' in tail or '−1.000' in tail)
check('MEMO 3109 rand 0.9997', '0.9997' in tail)
check('MEMO 3109 J 0.070', '0.070' in tail)
check('MEMO 3109 diffuse redundancy',
      '弥散冗余' in tail)
check('MEMO 3110 prereg', '3110 预注册' in tail)

led = json.load(io.open(LEDGER, encoding='utf-8'))
m3109 = [m for m in led['measurements']
         if m.get('phase') == 3109]
check('ledger has meas3109', len(m3109) == 1)
if m3109:
    c = m3109[0]['claim']
    check('claim has verdict',
          'stability_not_n_limited' in c)
    check('claim has floor curve', '0.2355' in c
          and '0.0194' in c)
    check('claim has rho -1.000', '-1.000' in c)
    check('claim has random 0.9997', '0.9997' in c)
    check('claim has random_sufficient',
          'random_sufficient' in c)
    check('claim has transfer 0.9998', '0.9998' in c)
    check('claim has diffuse', 'DIFFUSELY' in c)
    check('claim NEXT 3110', 'NEXT 3110' in c)
check('ledger measurements n=246',
      len(led['measurements']) == 246,
      'n=%d' % len(led['measurements']))
l14 = [l for l in led['linkage']
       if l.get('link_id') == 'L14_readout_spectrum_cross_model']
check('L14 includes meas3109',
      len(l14) == 1 and any('meas3109' in x
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
# closeout_log was overwritten by the idempotent re-run
# (memory rewrite after a 3000-char assert), so the
# first-run 'sha=' line is gone; verify self-consistency
# of the stored hash instead.
check('ledger sha8 self-consistent',
      saved == sha8,
      'stored=%s computed=%s' % (saved, sha8))

for tag, p in (('wlog D', WLOG_D), ('wlog C', WLOG_C)):
    try:
        w = io.open(p, encoding='utf-8').read()
        check('%s has Phase 3109 entry' % tag,
              'Phase 3109 Omega-P107' in w)
        check('%s records 0.9997' % tag,
              '0.9997' in w)
    except IOError as e:
        check('%s readable' % tag, False, repr(e))

mem = io.open(MEMO_W, encoding='utf-8').read()
check('MEMORY max=3109', 'max=3109' in mem)
check('MEMORY no max=3108 stale', 'max=3108' not in mem)
check('MEMORY next=3110', '下一 3110' in mem)
check('MEMORY 3109 block', '机制链状态（3109）' in mem)
check('MEMORY no dup 3106 lines',
      mem.count('- 3106：') == 1)
check('MEMORY no dup 3105 lines',
      mem.count('- 3105：') == 1)
lines.append('MEMORY.md len = %d' % len(mem))

for fn, minsize in (('result.json', 1500),
                    ('design_seal.json', 1000),
                    ('run_log.txt', 1000),
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
io.open(ROOT + r'\tests\gpt5_temp\p3109_disk_verify_out.txt',
        'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('verify done')
