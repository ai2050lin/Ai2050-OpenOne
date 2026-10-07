# -*- coding: utf-8 -*-
"""p3149 disk verify: independent on-disk
check of all five writes + anchors."""
import hashlib
import io
import json
import os
import re

ROOT = r'D:\AI2050\Ai2050-OpenOne'
D49 = (ROOT + r'\tests\glm5\result'
       + r'\rdc_query_construction_20260913'
       + r'\phase3149'
       + r'\omega_p147_carrier_dlogit_'
       + r'poslate_kdose_v3amp')
RES_SHA = '108d044e'
SEAL = '29d924e2'
MEMO = (ROOT + r'\research\gpt5\docs'
        + r'\AGI_GPT5_MEMO.md')
DAILY = (ROOT + r'\.workbuddy\memory'
         + r'\2026-10-01.md')
MEM = (ROOT + r'\.workbuddy\memory'
       + r'\MEMORY.md')
LEDGER = (ROOT + r'\research\gpt5\atlas'
          + r'\atlas_ledger.json')
SCRIPT = (ROOT + r'\tests\glm5'
          + r'\phase3149_omega_p147_'
          r'carrier_dlogit_poslate_kdose_'
          r'v3amp.py')

npass = 0
nfail = 0


def chk(tag, cond):
    global npass, nfail
    if cond:
        npass += 1
        print('PASS', tag)
    else:
        nfail += 1
        print('FAIL', tag)


# 1. result.json integrity
raw = io.open(D49 + r'\result.json',
              'rb').read()
sha = hashlib.sha256(raw).hexdigest()[:8]
chk('result sha8 == 108d044e',
    sha == RES_SHA)
r = json.loads(raw.decode('utf-8'))
chk('verdict starts a_3148_ok',
    r['verdict'].startswith('a_3148_ok'))
chk('repro_bit_8',
    'repro_bit_8' in r['verdict'])
chk('repro_bit_ok',
    'repro_bit_ok' in r['verdict'])
chk('carrier_common_found',
    'carrier_common_found' in
    r['verdict'])
chk('poslate_fixed_early',
    'poslate_fixed_early' in
    r['verdict'])
chk('kx_interchangeable',
    'kx_interchangeable' in
    r['verdict'])
chk('v3_bias_growing',
    'v3_bias_growing' in r['verdict'])
chk('xphase_ok', 'xphase_ok'
    in r['verdict'])
chk('runtime_s ~ 2498',
    abs(r['runtime_s'] - 2498.3) < 1.0)
chk('seal_sha8 == 29d924e2',
    str(r['seal_sha8']) == SEAL)

# 2. bit anchors all match
ba = r['part_bits']['bit_anchors']
chk('8 bit anchors',
    len(ba) == 8)
chk('all bit match',
    all(v['match'] is True
        for v in ba.values()))

# 3. part values
t2 = r['part_t2']
chk('neg common 13',
    len(t2['common_neg']) == 13)
chk('pos common 9',
    len(t2['common_pos']) == 9)
chk('shared empty',
    t2['shared'] == [])
chk('d131 med neg ~ 0.102',
    abs(t2['stats']['neg']
        ['d131_med'] - 0.1017) < 0.001)
chk('d131 med pos ~ -0.025',
    abs(t2['stats']['pos']
        ['d131_med'] + 0.0245) < 0.001)
l = r['part_l']
chk('flp_rows 13',
    len(l['flp_rows']) == 13)
chk('frac_fixed ~ 0.769',
    abs(l['frac_fixed'] - 0.7692)
    < 0.001)
chk('l_tag poslate_fixed_early',
    l['l_tag'] == 'poslate_fixed_early')
k = r['part_k']
chk('best_gap ~ 0.0234',
    abs(k['best_gap'] - 0.0234) < 0.001)
chk('best_pair top10@d1~full@d0.5',
    k['best_pair'] == [10, 1.0, 0.5])
v3 = r['part_v3']
chk('sym 0.05 == 1.0',
    abs(v3['sym_curve']['0.05'] - 1.0)
    < 1e-9)
chk('sym 0.10 == 1.0',
    abs(v3['sym_curve']['0.1'] - 1.0)
    < 1e-9)
chk('v3_tag growing',
    v3['v3_tag'] == 'v3_bias_growing')

# 4. seal + npz on disk
chk('design_seal.json exists',
    os.path.exists(D49 +
                   r'\design_seal.json'))
chk('p147_readout.npz exists',
    os.path.exists(D49 +
                   r'\p147_readout.npz'))
chk('no ckpt leftover',
    not os.path.exists(D49 +
                       r'\p147_ckpt.pkl'))

# 5. ledger n=286 + last phase 3149
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
chk('ledger n == 286',
    len(led['measurements']) == 286)
chk('last entry phase 3149',
    led['measurements'][-1]['phase']
    == 3149)
chk('ledger res sha matches',
    led['measurements'][-1]
    ['result_sha8'] == RES_SHA)

# 6. MEMO section
mt = io.open(MEMO, encoding='utf-8').read()
chk('MEMO Phase 3149 title',
    '## Phase 3149: 载体分离+早期固化'
    '+坐标剂量互换+偏置增长' in mt)
chk('MEMO prereg 3150',
    '3150（Ω-P148）预注册' in mt)
chk('MEMO res sha 108d044e',
    '108d044e' in mt)
chk('MEMO ledger n=286',
    'ledger n=286' in mt)
i49 = mt.index('## Phase 3149:')
i48 = mt.index('## Phase 3148:')
chk('MEMO 3149 after 3148', i49 > i48)

# 7. daily log
dt = io.open(DAILY, encoding='utf-8').read()
chk('daily has 3149 section',
    '## Phase 3149 闭环' in dt)
chk('daily has sha',
    '108d044e' in dt)

# 8. MEMORY.md line
mm = io.open(MEM, encoding='utf-8').read()
chk('MEMORY has 3149 entry',
    '3149（T4）' in mm)
chk('MEMORY has carrier_common',
    'carrier_common_found' in mm)
chk('MEMORY has res sha',
    '108d044e' in mm)

# 9. script rev markers
st = io.open(SCRIPT,
             encoding='utf-8').read()
for rev in ('rev-3149a', 'rev-3149b',
            'rev-3149c', 'rev-3149d',
            'rev-3149e'):
    chk('script %s marker' % rev,
        rev in st)
chk('script name p147',
    'omega_p147_carrier_dlogit' in st)

# 10. smoke artifacts intact
SMK = D49 + r'\smoke'
chk('smoke result exists',
    os.path.exists(SMK +
                   r'\result.json'))

print('==== %d/%d PASS ===='
      % (npass, npass + nfail))
if nfail:
    raise SystemExit('VERIFY FAILED')
print('DISK VERIFY COMPLETE')
