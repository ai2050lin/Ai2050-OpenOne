# -*- coding: utf-8 -*-
"""Phase 3111 disk verification: confirm all five writes landed."""
import hashlib
import io
import json
import os
import re

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3111'
        r'\omega_p109_broadcast_verdict')
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
idx = memo.find('## Phase 3111:')
check('MEMO contains Phase 3111 section', idx > 0,
      'at char %d of %d' % (idx, len(memo)))
check('MEMO 3111 near tail', idx >= len(memo) - 6000,
      'starts %d from end' % (len(memo) - idx))
tail = memo[idx:] if idx > 0 else ''
check('MEMO 3111 verdict',
      'mixed_broadcast_plus_distribution' in tail)
check('MEMO 3111 mean 0.9439', '0.9439' in tail)
check('MEMO 3111 single 0.9222', '0.9222' in tail)
check('MEMO 3111 holographic',
      '全息式冗余' in tail)
check('MEMO 3111 belief broadcast',
      '全局信念状态' in tail)
check('MEMO 3112 prereg', '3112 预注册' in tail)

led = json.load(io.open(LEDGER, encoding='utf-8'))
m3111 = [m for m in led['measurements']
         if m.get('phase') == 3111]
check('ledger has meas3111', len(m3111) == 1)
if m3111:
    c = m3111[0]['claim']
    check('claim verdict',
          'mixed_broadcast_plus_distribution' in c)
    check('claim mean 0.9439', '0.9439' in c)
    check('claim single 0.9222', '0.9222' in c)
    check('claim d_min 5 unchanged',
          'd_min stays 5' in c)
    check('claim 80.6pct', '80.6pct' in c)
    check('claim holographic', 'holographic' in c)
    check('claim belief state',
          'BELIEF STATE' in c)
    check('claim NEXT 3112', 'NEXT 3112' in c)
check('ledger measurements n=248',
      len(led['measurements']) == 248,
      'n=%d' % len(led['measurements']))
l14 = [l for l in led['linkage']
       if l.get('link_id') == 'L14_readout_spectrum_cross_model']
check('L14 includes meas3111',
      len(l14) == 1 and any('meas3111' in x
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
        check('%s has Phase 3111 entry' % tag,
              'Phase 3111 Omega-P109' in w)
        check('%s records 0.9222' % tag,
              '0.9222' in w)
    except IOError as e:
        check('%s readable' % tag, False, repr(e))

mem = io.open(MEMO_W, encoding='utf-8').read()
check('MEMORY max=3111', 'max=3111' in mem)
check('MEMORY no max=3110 stale', 'max=3110' not in mem)
check('MEMORY next=3112', '下一 3112' in mem)
check('MEMORY 3111 block', '机制链状态（3111）' in mem)
check('MEMORY no next-3111 stale',
      '下一 3111' not in mem)
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
io.open(ROOT + r'\tests\gpt5_temp\p3111_disk_verify_out.txt',
        'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('verify done')
