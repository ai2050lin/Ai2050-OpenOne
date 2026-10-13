# -*- coding: utf-8 -*-
"""3178 closeout self-check v2: corrected integrity checks (no data writes).

The v1 check 2/5 formula double-counted a separator newline (+1 that never
happened: the MEMO physical tail already ended with CRLF, so sep='').
This script re-runs all 5 checks with structural invariants instead of
arithmetic that depends on the original file's exact trailing-newline count.
"""
import datetime
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
LEDGER = ROOT + r'\research\gpt5\atlas\atlas_ledger.json'
MEMO = ROOT + r'\research\gpt5\docs\AGI_GPT5_MEMO.md'
DAILY = ROOT + r'\.workbuddy\memory\2026-10-10.md'
WMEM = ROOT + r'\.workbuddy\memory\MEMORY.md'
S = ROOT + r'\tests\glm5\result\rdc_query_construction_20260913\phase3178\g5b3_ftr14_a2_attribution'

LOG = []
ok = 0


def chk(name, cond, extra=''):
    global ok
    assert cond, (name, extra)
    ok += 1
    LOG.append('PASS %s %s' % (name, extra))
    print('PASS %s %s' % (name, extra), flush=True)


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


# 1. ledger
led = json.load(io.open(LEDGER, encoding='utf-8'))
chk('1a ledger n=330', len(led['measurements']) == 330, str(len(led['measurements'])))
chk('1b last phase 3178', led['measurements'][-1]['phase'] == 3178)
chk('1c verdict field', 'relation_specific_residual_confirmed' in led['measurements'][-1]['verdict'])
led_tmp = json.loads(json.dumps(led, ensure_ascii=False))
led_tmp.pop('ledger_sha256_8', None)
recomputed = sha8(json.dumps(led_tmp, ensure_ascii=False, indent=1, sort_keys=False).encode('utf-8'))
chk('1d chain', led['ledger_sha256_8'] == recomputed, led['ledger_sha256_8'])

# 2. MEMO structural integrity
raw = open(MEMO, 'rb').read()
chk('2a MEMO BOM', raw.startswith(b'\xef\xbb\xbf'))
text = raw.decode('utf-8').lstrip('\ufeff')
chk('2b no bare LF (all CRLF)', text.count('\n') == text.count('\r\n'),
    'lf=%d crlf=%d' % (text.count('\n'), text.count('\r\n')))
chk('2c 3178 heading once', text.count('## Phase 3178: G5-B3 FTR-14 A2 归因分解') == 1)
i_prereg3178 = text.find('### 接续：预注册 3178')
i_3178 = text.find('## Phase 3178: G5-B3')
chk('2d 3178 section after 3177 prereg', 0 < i_prereg3178 < i_3178)
chk('2e 3177 heading intact', '## Phase 3177: G5-B2 FTR-14 held_out 关系留出臂' in text)
chk('2f tail sentence', text.endswith('水果类、K4。\r\n'))
for anchor in ('relation_specific_residual_confirmed', 'r(m)=g−p=0.0918/0.0951/0.1289',
               'p fold min | -0.0308 / -0.0458 | -0.0491 / -0.0529 | -0.0351 / -0.0414',
               '### 接续：预注册 3179', 'FTR-23 立项入册', 'res 4f521d9b / seal 1f166645'):
    chk('2g MEMO contains: ' + anchor[:30], anchor in text)

# 3. daily
d2 = io.open(DAILY, encoding='utf-8').read()
chk('3a daily 3178 entry', '3178 G5-B3' in d2 and '4f521d9b/seal 1f166645' in d2)
chk('3b daily prereg 3179', '预注册 3179=G5-B4 FTR-23 立项' in d2)

# 4. workspace MEMORY
w2 = open(WMEM, 'rb').read().decode('utf-8')
chk('4a MEMORY typo fixed', '字节恒乙' not in w2 and '字节恒等' in w2)
chk('4b MEMORY 3178 entry', '✅ 3178 FTR-14 A2 归因分解闭环' in w2 and '329→**330**' in w2)
chk('4c MEMORY 3179 next', '下一步 3179=G5-B4 FTR-23 立项入册' in w2)

# 5. artifacts
a = sha8(open(S + r'\smoke_atlas_registry_v1_5.json', 'rb').read())
b = sha8(open(S + r'\atlas_registry_v1_5.json', 'rb').read())
chk('5a registry SMOKE/FULL byte-identical', a == b == 'e1942d03', '%s/%s' % (a, b))
r_full = json.load(io.open(S + r'\result.json', encoding='utf-8'))
chk('5b FULL res/seal', r_full['res_sha8'] == '4f521d9b' and r_full['seal_sha8'] == '1f166645')
chk('5c verdict state', r_full['arm']['calibration_pass'] is False
    and r_full['gates']['calibration_bar'] is False
    and r_full['gates']['device_3157'] is True and r_full['gates']['replication_3177'] is True)
chk('5d registry v1.5 invariants', r_full['counts']['features'] == 22
    and r_full['counts']['failures'] == 16 and r_full['counts']['upgrades'] == 5)
h = sha8(open(S + r'\atlas_v1_6.html', 'rb').read())
chk('5e html sha + fields', h == 'b95e67a2' and r_full['field_check']['n_expected'] == 669
    and r_full['field_check']['mismatch'] == 0 and r_full['field_check']['missing'] == 0
    and r_full['field_check']['extra'] == 0 and r_full['field_check']['dups'] == 0)

LOG.append('SELF-CHECK v2 %d/%d ALL PASS' % (ok, ok))
print('SELF-CHECK v2 %d/%d ALL PASS' % (ok, ok))
with io.open(ROOT + r'\tests\gpt5_temp\p3178_closeout_out.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(LOG) + '\n')
print('CHECK2 DONE')
