# -*- coding: utf-8 -*-
"""Phase 3105 disk verification: confirm all five writes landed on real disk."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3105'
        r'\omega_p103_incontext_truth_consistency')
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

# 1. MEMO tail section
memo = io.open(MEMO, encoding='utf-8').read()
idx = memo.find('## Phase 3105:')
tail_pos = len(memo) - 5000
check('MEMO contains Phase 3105 section', idx > 0,
      'at char %d of %d' % (idx, len(memo)))
check('MEMO Phase 3105 in tail', idx >= tail_pos - 8000,
      'section starts %d chars from end' % (len(memo) - idx))
check('MEMO has dissociation 3x repetition',
      memo.count('in-context 一致性而非外部图成员') >= 3,
      'count=%d' % memo.count('in-context 一致性而非外部图成员'))
check('MEMO 3105 sha8=9dde1cb0 recorded',
      'material=9dde1cb0' in memo[idx:] if idx > 0 else False)
check('MEMO ends with 3105 script line',
      'phase3105_omega_p103_incontext_truth_consistency.py' in memo[idx:])

# 2. Ledger
led = json.load(io.open(LEDGER, encoding='utf-8'))
m3105 = [m for m in led['measurements'] if m.get('phase') == 3105]
check('ledger has meas3105', len(m3105) == 1,
      'n_matches=%d' % len(m3105))
if m3105:
    c = m3105[0]['claim']
    check('ledger claim G1 0.990', '0.990' in c)
    check('ledger claim 74/74', '74/74' in c)
    check('ledger claim decoy -6.77', '-6.77' in c)
    check('ledger claim G2 artifact', 'tie-break' in c)
    check('ledger claim sign correction',
          '0.497 becomes 0.503' in c)
    check('ledger claim DISSOCIATION', 'DISSOCIATION' in c)
    check('ledger anchors sha8', '9dde1cb0' in m3105[0]['anchors'])
blob = json.dumps(led, sort_keys=True, ensure_ascii=False)
sha = hashlib.sha256(blob.encode('utf-8')).hexdigest()[:8]
# ledger file may have key order differences from blob; recompute including sha key
check('ledger measurements n=242',
      len(led['measurements']) == 242,
      'n=%d' % len(led['measurements']))
l14 = [l for l in led['linkage']
       if l.get('link_id') == 'L14_readout_spectrum_cross_model'][0]
check('L14 connects includes meas3105',
      any('meas3105' in x for x in l14['connects']),
      'len=%d' % len(l14['connects']))
lines.append('ledger recomputed sha8 (claim-order) = %s ; stored = %s'
             % (sha, led.get('ledger_sha256_8')))

# 3. Both wlogs
for tag, p in (('wlog D', WLOG_D), ('wlog C', WLOG_C)):
    try:
        w = io.open(p, encoding='utf-8').read()
        check('%s has Phase 3105 entry' % tag,
              'Phase 3105 Omega-P103' in w)
        check('%s records 0.990' % tag, '0.990' in w)
        check('%s records sha8' % tag, '9dde1cb0' in w)
    except IOError as e:
        check('%s readable' % tag, False, repr(e))

# 4. MEMORY.md
mem = io.open(MEMO_W, encoding='utf-8').read()
check('MEMORY max=3105', 'max=3105' in mem)
check('MEMORY no max=3104 stale', 'max=3104' not in mem)
check('MEMORY next=3106', '下一 3106' in mem)
check('MEMORY mechanism-chain 3105 block',
      '机制链状态（3105）' in mem)
lines.append('MEMORY.md len = %d' % len(mem))

# 5. Artifacts exist with plausible sizes
for fn, minsize in (('result.json', 500), ('material.json', 100000),
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
io.open(ROOT + r'\tests\gpt5_temp\p3105_disk_verify_out.txt',
        'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('verify done')
