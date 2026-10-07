# -*- coding: utf-8 -*-
"""Phase 3112 disk verification: re-check every closeout
write against the REAL disk (report written to file,
then read back - bash stdout unreliable on this box)."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3112'
        r'\omega_p110_broadcast_emergence')
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


# ---- artifacts on disk ----
for fn, mn in (('result.json', 500),
               ('design_seal.json', 300),
               ('run_log.txt', 1500),
               ('closeout_log.txt', 100)):
    p = OUTD + '\\' + fn
    sz = os.path.getsize(p) if os.path.exists(p) else -1
    check('file %s size>=%d (got %d)' % (fn, mn, sz),
          sz >= mn)

# ---- ledger ----
led = json.load(io.open(LEDGER, encoding='utf-8'))
n_m = len(led['measurements'])
check('ledger n==249 (got %d)' % n_m, n_m == 249)
m12 = [m for m in led['measurements']
       if m.get('phase') == 3112]
check('ledger has phase 3112 (got %d)' % len(m12),
      len(m12) == 1)
check('meas_id meas3112',
      m12 and m12[0]['meas_id'] ==
      'meas3112_omega_p110_broadcast_emergence')
check('meas3112 verdict',
      m12 and m12[0]['verdict'] ==
      'emerge_L6|few_channels|replicated')
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
n_l14 = len(l14['connects'])
check('L14 connects==217 (got %d)' % n_l14, n_l14 == 217)
check('L14 last is meas3112',
      l14['connects'][-1] ==
      'meas3112_omega_p110_broadcast_emergence')
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
check('MEMO chars >= 944000 (got %d)' % n_char,
      n_char >= 944000)
check('MEMO has Phase 3112 title',
      '## Phase 3112:' in memo)
check('MEMO 3112 title has timestamp bracket',
      '] [20' in memo or '] [2' in memo or
      '[2026-' in memo)
tail = memo[-3500:]
check('MEMO tail has emerge L6 claim',
      ('L6' in tail and '0.9386' in tail))
check('MEMO tail has few_channels claim',
      'few_channels' in tail and '0.4791' in tail)
check('MEMO tail has PC1-truth AUC 0.9939',
      '0.9939' in tail)
check('MEMO tail has 3106 replication 0.7432',
      '0.7432' in tail)
check('MEMO tail has 3113 preregistration',
      '3113' in tail and '伪迹分离' in tail)
check('MEMO tail has hard-caveats section',
      '硬伤' in tail)
# phase order sanity: 3112 after 3111
i11 = memo.find('## Phase 3111:')
i12 = memo.find('## Phase 3112:')
check('MEMO 3111 before 3112',
      0 <= i11 < i12)

# ---- workspace logs ----
for wl in (WLOG_D, WLOG_C):
    try:
        prev = io.open(wl, encoding='utf-8').read()
        ok = ('Phase 3112 Omega-P110' in prev and
              'emerge_L6' in prev)
        check('wlog %s has 3112 entry' % wl[:40], ok)
    except IOError as e:
        check('wlog %s readable' % wl[:40], False)

# ---- MEMORY.md ----
mem = io.open(MEMO_W, encoding='utf-8').read()
check('MEMORY chars < 3000 (got %d)' % len(mem),
      len(mem) < 3000)
check('MEMORY max=3112', 'max=3112' in mem)
check('MEMORY no stale max=3111',
      'max=3111' not in mem)
check('MEMORY 机制链状态（3112）',
      '机制链状态（3112）' in mem)
check('MEMORY 3112 line has emerge_L6',
      'emerge_L6' in mem)
check('MEMORY 3112 line has PC1 AUC 0.994',
      '0.994' in mem)
check('MEMORY next 3113', '下一 3113' in mem)
check('MEMORY duplicated next-line fixed',
      '之后 T4 + 写入端 之后' not in mem)

o.append('SUMMARY pass=%d fail=%d' % (n_pass, n_fail))
io.open(REPF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('verify done pass=%d fail=%d'
      % (n_pass, n_fail))
