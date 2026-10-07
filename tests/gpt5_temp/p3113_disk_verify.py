# -*- coding: utf-8 -*-
"""Phase 3113 disk verification."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3113'
        r'\omega_p111_artifact_writein')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
WLOG_D = ROOT + r'\.workbuddy\memory\2026-09-23.md'
WLOG_C = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
          r'\.workbuddy\memory\2026-09-23.md')
MEMO_W = ROOT + r'\.workbuddy\memory\MEMORY.md'
REPF = OUTD + r'\disk_verify_out.txt'
o = []
n_pass = 0
n_fail = 0


def check(name, cond):
    global n_pass, n_fail
    if cond:
        n_pass += 1
        o.append('PASS %s' % name)
    else:
        n_fail += 1
        o.append('FAIL %s' % name)


# ---- artifacts ----
for fn, mn in (('result.json', 2000),
               ('design_seal.json', 600),
               ('run_log.txt', 3000),
               ('closeout_log.txt', 100),
               ('capture_b.npz', 100000000)):
    p = OUTD + '\\' + fn
    sz = os.path.getsize(p) if os.path.exists(p) else -1
    check('file %s size>=%d (got %d)' % (fn, mn, sz),
          sz >= mn)

# ---- ledger ----
led = json.load(io.open(LEDGER, encoding='utf-8'))
n_m = len(led['measurements'])
check('ledger n==250 (got %d)' % n_m, n_m == 250)
m13 = [m for m in led['measurements']
       if m.get('phase') == 3113]
check('ledger has phase 3113 (got %d)' % len(m13),
      len(m13) == 1)
check('meas_id meas3113',
      m13 and m13[0]['meas_id'] ==
      'meas3113_omega_p111_artifact_writein')
check('meas3113 verdict',
      m13 and m13[0]['verdict'] ==
      'belief_robust|within_unit_replicated|'
      'write_in_concentrated')
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
n_l14 = len(l14['connects'])
check('L14 connects==218 (got %d)' % n_l14,
      n_l14 == 218)
check('L14 last is meas3113',
      l14['connects'][-1] ==
      'meas3113_omega_p111_artifact_writein')
saved = led.pop('ledger_sha256_8', None)
blob = json.dumps(led, sort_keys=True,
                  ensure_ascii=False)
sha8 = hashlib.sha256(
    blob.encode('utf-8')).hexdigest()[:8]
led['ledger_sha256_8'] = saved
check('ledger sha8 self-consistent (got %s)' % sha8,
      saved == sha8)
cl = io.open(OUTD + r'\closeout_log.txt',
             encoding='utf-8').read()
check('closeout_log sha matches ledger',
      ('sha=' + saved) in cl)

# ---- MEMO ----
memo = io.open(MEMO, encoding='utf-8').read()
n_char = len(memo)
check('MEMO chars >= 957000 (got %d)' % n_char,
      n_char >= 957000)
check('MEMO has Phase 3113 title',
      '## Phase 3113:' in memo)
i12 = memo.find('## Phase 3112:')
i13 = memo.find('## Phase 3113:')
check('MEMO 3112 before 3113', 0 <= i12 < i13)
tail = memo[-4200:]
check('MEMO tail within-pair 0.952', '0.952' in tail)
check('MEMO tail cross-pair 0.920', '0.920' in tail)
check('MEMO tail ds_mlp -3.970',
      '-3.970' in tail or '\u22123.970' in tail)
check('MEMO tail top8 0.556', '0.556' in tail)
check('MEMO tail slot clarification (L6=layer 28)',
      '模型层 28' in tail or '层 28' in tail)
check('MEMO tail 3114 preregistration',
      '3114' in tail and '因果消融' in tail)
check('MEMO tail hard-caveats', '硬伤' in tail)

# ---- workspace logs ----
for wl in (WLOG_D, WLOG_C):
    try:
        prev = io.open(wl, encoding='utf-8').read()
        ok = ('Phase 3113 Omega-P111' in prev and
              'belief_robust' in prev)
        check('wlog %s has 3113 entry' % wl[:40], ok)
    except IOError:
        check('wlog %s readable' % wl[:40], False)

# ---- MEMORY.md ----
mem = io.open(MEMO_W, encoding='utf-8').read()
check('MEMORY chars < 3000 (got %d)' % len(mem),
      len(mem) < 3000)
check('MEMORY max=3113', 'max=3113' in mem)
check('MEMORY no stale max=3112',
      'max=3112' not in mem)
check('MEMORY 机制链状态（3113）',
      '机制链状态（3113）' in mem)
check('MEMORY 3113 line has 0.952', '0.952' in mem)
check('MEMORY 3113 line has erase -3.97',
      '-3.97' in mem or '\u22123.97' in mem)
check('MEMORY next 3114', '下一 3114' in mem)
check('MEMORY no stale 下一 3113',
      '下一 3113' not in mem)

o.append('SUMMARY pass=%d fail=%d' % (n_pass, n_fail))
io.open(REPF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('verify done pass=%d fail=%d'
      % (n_pass, n_fail))
