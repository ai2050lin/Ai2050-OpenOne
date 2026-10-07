# -*- coding: utf-8 -*-
"""Phase 3129 post-verify: MEMO errata
(design_seal.json claim) + wlog verify lines."""
import datetime
import io

import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
LEDGER = (ROOT + r'\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (ROOT + r'\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WLOGS = [ROOT + r'\.workbuddy\memory',
         (r'C:\Users\Admin\WorkBuddy'
          r'\2026-09-17-01-30-05'
          r'\.workbuddy\memory')]
NOW = datetime.datetime.now()
STAMP = NOW.strftime('%Y-%m-%d %H:%M')
sha_led = json.load(io.open(
    LEDGER, encoding='utf-8'))['ledger_sha256_8']

memo = io.open(MEMO, encoding='utf-8').read()
if '3129 纠错' not in memo:
    line = ('\n\n**3129 纠错 [%s]**：上节产物'
            '列表笔误——`design_seal.json` '
            '不在 3129 产物中（3129 脚本未'
            '单独立 seal 文件；预注册冻结由'
            ' 3128 MEMO §5 与 3129 脚本常量'
            '承担）。verify 19/19 PASS 已按'
            '实际产物清单（result.json、'
            'p127_readout.npz、run_log.txt）'
            '复核，ledger sha8 %s。'
            % (STAMP, sha_led))
    with io.open(MEMO, 'a',
                 encoding='utf-8') as f:
        f.write(line)
    print('memo errata appended')
else:
    print('memo errata exists, skip')

wline = ('- Phase 3129 verify: 19/19 PASS '
         '(A1-A4 B1-B3 C1-C3 D1-D3 E1-E2 '
         'F1-F3 G1), fails=0; gates f32 '
         'maxd=0.00e+00; dose f10 3128 '
         'repro exact; ledger sha8 %s; '
         'seal errata appended.'
         % sha_led)
for wd in WLOGS:
    wl = wd + '\\' + '2026-09-25.md'
    try:
        prev = io.open(wl,
                       encoding='utf-8').read()
    except IOError:
        prev = ''
    if 'Phase 3129 verify' not in prev:
        with io.open(wl, 'a',
                     encoding='utf-8') as f:
            f.write(wline + '\n')
        print('wlog verify: %s'
              % wl[:40])
    else:
        print('wlog verify exists: %s'
              % wl[:40])
print('POSTVERIFY_OK')
