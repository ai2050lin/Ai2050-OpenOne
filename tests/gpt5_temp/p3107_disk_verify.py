# -*- coding: utf-8 -*-
"""Phase 3107 disk verification: confirm all five writes landed."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3107'
        r'\omega_p105_writehead_mapping')
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
idx = memo.find('## Phase 3107:')
check('MEMO contains Phase 3107 section', idx > 0,
      'at char %d of %d' % (idx, len(memo)))
check('MEMO 3107 near tail', idx >= len(memo) - 4000,
      'starts %d from end' % (len(memo) - idx))
tail = memo[idx:] if idx > 0 else ''
check('MEMO 3107 verdict string',
      'sparse_but_material_specific' in tail)
check('MEMO 3107 Jaccard 0.0526', '0.0526' in tail)
check('MEMO 3107 Spearman 0.6545', '0.6545' in tail)
check('MEMO 3107 cos 0.0050', '0.0050' in tail)
check('MEMO 3107 multi-route readout',
      '多路读出' in tail)
check('MEMO 3107 G1 table 0.9999', '0.9999' in tail)
check('MEMO 3108 prereg', '3108 = 简并分离与子空间角度'
      in tail)

led = json.load(io.open(LEDGER, encoding='utf-8'))
m3107 = [m for m in led['measurements']
         if m.get('phase') == 3107]
check('ledger has meas3107', len(m3107) == 1)
if m3107:
    c = m3107[0]['claim']
    check('claim has verdict', 'sparse_but_material_'
          'specific' in c)
    check('claim has G1 5/5', '5/5' in c)
    check('claim has Jaccard 0.0526', '0.0526' in c)
    check('claim has random 0.042', '0.042' in c)
    check('claim has cos 0.0050', '0.0050' in c)
    check('claim has Spearman 0.6545', '0.6545' in c)
    check('claim has subspace-level', 'SUBSPACE'
          in c)
    check('claim NEXT 3108', 'NEXT' in c and '3108' in c)
check('ledger measurements n=244',
      len(led['measurements']) == 244,
      'n=%d' % len(led['measurements']))
l14 = [l for l in led['linkage']
       if l.get('link_id') == 'L14_readout_spectrum_cross_model']
check('L14 includes meas3107',
      len(l14) == 1 and any('meas3107' in x
                            for x in l14[0]['connects']),
      'len=%d' % (len(l14[0]['connects'])
                  if l14 else -1))
# recompute ledger sha8 (same method as closeout)
saved = led.pop('ledger_sha256_8', None)
blob = json.dumps(led, sort_keys=True,
                  ensure_ascii=False)
sha8 = hashlib.sha256(
    blob.encode('utf-8')).hexdigest()[:8]
led['ledger_sha256_8'] = saved
check('ledger sha8 matches closeout log',
      sha8 == 'e4586ded', 'computed=%s' % sha8)

for tag, p in (('wlog D', WLOG_D), ('wlog C', WLOG_C)):
    try:
        w = io.open(p, encoding='utf-8').read()
        check('%s has Phase 3107 entry' % tag,
              'Phase 3107 Omega-P105' in w)
        check('%s records Jaccard' % tag,
              '0.053' in w)
    except IOError as e:
        check('%s readable' % tag, False, repr(e))

mem = io.open(MEMO_W, encoding='utf-8').read()
check('MEMORY max=3107', 'max=3107' in mem)
check('MEMORY no max=3106 stale', 'max=3106' not in mem)
check('MEMORY next=3108', '下一 3108' in mem)
check('MEMORY 3107 block', '机制链状态（3107）' in mem)
check('MEMORY no next-3107 stale',
      '下一 3107' not in mem)
lines.append('MEMORY.md len = %d' % len(mem))

for fn, minsize in (('result.json', 500),
                    ('design_seal.json', 200),
                    ('run_log.txt', 100),
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
io.open(ROOT + r'\tests\gpt5_temp\p3107_disk_verify_out.txt',
        'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('verify done')
