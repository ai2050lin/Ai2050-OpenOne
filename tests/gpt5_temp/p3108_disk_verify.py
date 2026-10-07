# -*- coding: utf-8 -*-
"""Phase 3108 disk verification: confirm all five writes landed."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3108'
        r'\omega_p106_degeneracy_subspace')
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
idx = memo.find('## Phase 3108:')
check('MEMO contains Phase 3108 section', idx > 0,
      'at char %d of %d' % (idx, len(memo)))
check('MEMO 3108 near tail', idx >= len(memo) - 6000,
      'starts %d from end' % (len(memo) - idx))
tail = memo[idx:] if idx > 0 else ''
check('MEMO 3108 verdict', 'per_material_readout'
      in tail)
check('MEMO 3108 M1 J 0.118', '0.118' in tail)
check('MEMO 3108 M2 cos 0.0039', '0.0039' in tail)
check('MEMO 3108 sanity 0/3', '0/3' in tail)
check('MEMO 3108 probe cos 0.041', '0.041' in tail)
check('MEMO 3108 PR 15.33', '15.33' in tail)
check('MEMO 3108 underdetermined 1459', '1459' in tail)
check('MEMO 3109 prereg', '3109 预注册' in tail)

led = json.load(io.open(LEDGER, encoding='utf-8'))
m3108 = [m for m in led['measurements']
         if m.get('phase') == 3108]
check('ledger has meas3108', len(m3108) == 1)
if m3108:
    c = m3108[0]['claim']
    check('claim has verdict', 'per_material_'
          'readout' in c)
    check('claim has M1 0.118', '0.118' in c)
    check('claim has P1 0.0039', '0.0039' in c)
    check('claim has internal_inconsistency',
          'internal_inconsistency' in c)
    check('claim has probe 0.041', '0.041' in c)
    check('claim has PR 15.33', '15.33' in c)
    check('claim has 1459', '1459' in c)
    check('claim NEXT 3109', 'NEXT 3109' in c)
check('ledger measurements n=245',
      len(led['measurements']) == 245,
      'n=%d' % len(led['measurements']))
l14 = [l for l in led['linkage']
       if l.get('link_id') == 'L14_readout_spectrum_cross_model']
check('L14 includes meas3108',
      len(l14) == 1 and any('meas3108' in x
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
led['ledger_sha256_8'] = saved
import re
mm = re.search(r'sha=([0-9a-f]{8})', sha_log)
check('ledger sha8 matches closeout log',
      mm is not None and sha8 == mm.group(1),
      'computed=%s log=%s'
      % (sha8, mm.group(1) if mm else '?'))

for tag, p in (('wlog D', WLOG_D), ('wlog C', WLOG_C)):
    try:
        w = io.open(p, encoding='utf-8').read()
        check('%s has Phase 3108 entry' % tag,
              'Phase 3108 Omega-P106' in w)
        check('%s records 0.118' % tag, '0.118' in w)
    except IOError as e:
        check('%s readable' % tag, False, repr(e))

mem = io.open(MEMO_W, encoding='utf-8').read()
check('MEMORY max=3108', 'max=3108' in mem)
check('MEMORY no max=3107 stale', 'max=3107' not in mem)
check('MEMORY next=3109', '下一 3109' in mem)
check('MEMORY 3108 block', '机制链状态（3108）' in mem)
check('MEMORY no next-3108 stale',
      '下一 3108' not in mem)
lines.append('MEMORY.md len = %d' % len(mem))

for fn, minsize in (('result.json', 2000),
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
io.open(ROOT + r'\tests\gpt5_temp\p3108_disk_verify_out.txt',
        'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('verify done')
