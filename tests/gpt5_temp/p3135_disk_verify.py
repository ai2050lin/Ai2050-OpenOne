# -*- coding: utf-8 -*-
"""Phase 3135 disk verify: independent
re-read of all five write targets.
~22 checks; prints PASS/FAIL per check;
exit 1 on any FAIL."""
import hashlib
import io
import json
import sys

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3135'
        r'\omega_p133_conduction_'
        r'co36ablation_window')
LEDGER = (ROOT + r'\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (ROOT + r'\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
MEMW = ROOT + r'\.workbuddy\memory\MEMORY.md'
WLOG1 = (ROOT + r'\.workbuddy\memory'
         r'\2026-09-27.md')
WLOG2 = (r'C:\Users\Admin\WorkBuddy'
         r'\2026-09-17-01-30-05'
         r'\.workbuddy\memory'
         r'\2026-09-27.md')

fails = []


def chk(name, ok, detail=''):
    tag = 'PASS' if ok else 'FAIL'
    print('%s %s %s' % (tag, name, detail))
    if not ok:
        fails.append(name)


# 1-3 result.json + verdict integrity
r = json.load(io.open(
    OUTD + r'\result.json',
    encoding='utf-8'))
chk('01 smoke_false', r['smoke'] is False)
V = r['verdict']
chk('02 verdict_8tok',
    len(V.split('|')) == 8
    and V.split('|')[7]
    == 'coverage_full', V[:50])
raw = io.open(OUTD + r'\result.json',
              'rb').read()
sha8 = hashlib.sha256(raw).hexdigest()[:8]

# 4-6 ledger
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
ms = led['measurements']
chk('04 ledger_n272', len(ms) == 272,
    'n=%d' % len(ms))
last = ms[-1]
chk('05 last_meas3135',
    last['meas_id']
    == 'meas3135_omega_p133_'
    'conduction_co36ablation_window')
led2 = {k: v for k, v in led.items()
        if k != 'ledger_sha256_8'}
# mirror closeout: blob was hashed while
# the PREVIOUS phase sha was still in
# the dict (3134 closeout pattern)
led2['ledger_sha256_8'] = '3f905773'
blob = json.dumps(led2, sort_keys=True,
                  ensure_ascii=False)
re8 = hashlib.sha256(
    blob.encode('utf-8')).hexdigest()[:8]
chk('06 ledger_sha8',
    re8 == led.get('ledger_sha256_8'),
    '%s vs %s' % (re8,
                  led.get('ledger_sha256_8')))
chk('07 ledger_verdict_match',
    last['verdict'] == V)
chk('08 ledger_result_sha',
    last['hashes']['result_sha256_8']
    == sha8,
    '%s vs %s' % (last['hashes']
                  ['result_sha256_8'],
                  sha8))

# 9-12 MEMO
memo = io.open(MEMO,
               encoding='utf-8').read()
n35 = memo.count('## Phase 3135:')
chk('09 memo_3135_once', n35 == 1,
    'count=%d' % n35)
i35 = memo.find('## Phase 3135:')
i34 = memo.find('## Phase 3134:')
chk('10 memo_append_order', i35 > i34 > 0,
    'i34=%d i35=%d' % (i34, i35))
head = memo[i35:i35 + 120]
chk('11 memo_title_short',
    head.split('\n')[0][-1] == ']'
    and len(head.split('\n')[0]) < 110,
    head.split('\n')[0][:60])
chk('12 memo_sections',
    all(s in memo[i35:] for s in (
        '### 1. 三大发现', '### 2. 关键数值',
        '### 3. 硬伤', '### 4. 机制拼图更新',
        '### 5. 3136 预注册')))

# 13-15 wlogs
for nm, wl in (('13 wlog_D', WLOG1),
               ('14 wlog_C', WLOG2)):
    try:
        t = io.open(wl,
                    encoding='utf-8').read()
        chk(nm,
            'Phase 3135 Omega-P133 closeout'
            in t)
    except IOError as e:
        chk(nm, False, str(e))

# 16-22 MEMORY.md
mem = io.open(MEMW,
              encoding='utf-8').read()
chk('16 mem_len', len(mem) < 3000,
    'len=%d' % len(mem))
for a in ('3135（T4）：', '下一 3136',
          'max=3135'):
    chk('17 anchor %s' % a,
        mem.count(a) == 1,
        'n=%d' % mem.count(a))
anchors = ['3134（T4）：', '3133（T4）：',
           '3132（T4）：', '3131（T4）：',
           '3130（T4）：', '3129（T4）：',
           '3128（T4）：', '3127（T4）：',
           '3126（T4）：', '3125（T4）：',
           '3124（T4）：', '3123（T4）：',
           '3121–3122：', '3118–3120：',
           '3113–3117：', '3110–3112：',
           '3109：', '3108：', '3107：',
           '3106：', '3105：',
           '3104/3103/3101：']
bad = [a for a in anchors
       if mem.count('- ' + a) != 1]
chk('18 mem_anchors_once',
    not bad, str(bad))
ls = mem.split('\n')
i34l = [i for i, e in enumerate(ls)
        if e.startswith('- 3134（T4）：')]
i35l = [i for i, e in enumerate(ls)
        if e.startswith('- 3135（T4）：')]
chk('19 mem_3135_after_3134',
    len(i34l) == 1 and len(i35l) == 1
    and i35l[0] == i34l[0] + 1)
chk('20 mem_no_3134_next',
    '下一 3135' not in mem)

# 21-22 artifacts on disk
try:
    z = np.load(OUTD
                + r'\p133_readout.npz',
                allow_pickle=False)
    chk('21 npz_29keys',
        len(z.files) == 29
        and 'co36' in z.files
        and 'cos_38' in z.files,
        'n=%d' % len(z.files))
except Exception as e:
    chk('21 npz_29keys', False, str(e))
import os
seal = os.path.exists(OUTD
                      + r'\design_seal.json')
rl = io.open(OUTD + r'\run_log.txt',
             encoding='utf-8').read()
chk('22 seal_and_done',
    seal and 'P3135 DONE' in rl)
chk('23 ckpt_removed',
    not os.path.exists(OUTD
                       + r'\p133_ckpt.pkl'))

print('VERIFY_%s (%d fails)'
      % ('OK' if not fails else 'FAIL',
         len(fails)))
sys.exit(1 if fails else 0)
