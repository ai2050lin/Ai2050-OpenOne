# -*- coding: utf-8 -*-
"""Independent on-disk verification for
Phase 3144 closeout (reads REAL disk
state; nothing taken on trust)."""
import hashlib
import io
import json
import os

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        + r'\rdc_query_construction_20260913')
D44 = os.path.join(
    RDIR, 'phase3144',
    'omega_p142_readouttraj_co36sign_'
    'unembed_d19resid')
LEDGER = (ROOT + r'\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (ROOT + r'\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
PLOG = (ROOT + r'\.workbuddy\memory'
        r'\2026-09-29.md')
PMEM = (ROOT + r'\.workbuddy\memory'
        r'\MEMORY.md')

n_pass = 0
n_fail = 0


def chk(name, cond, detail=''):
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print('PASS %s %s' % (name, detail))
    else:
        n_fail += 1
        print('FAIL %s %s' % (name, detail))


# 1. result.json on disk
rf = os.path.join(D44, 'result.json')
chk('01 result exists', os.path.exists(rf))
raw = io.open(rf, 'rb').read()
sha = hashlib.sha256(raw).hexdigest()[:8]
chk('02 result sha8', sha == 'def947d8',
    sha)
res = json.loads(raw.decode('utf-8'))
chk('03 smoke false',
    res['smoke'] is False)
EXP = ('a_3143_ok|repro_bit_6|'
       'repro_bit_ok|dvec19_repro_6a0332|'
       'field_self_ok|z35_cos17_ok|'
       'traj_gradual|pc1_channel_format|'
       'tail_neg_confirmed|tail_sign_sym|'
       'identity_unembed_orthogonal|'
       'd19resid_readout|xphase_ok|'
       'coverage_full')
chk('04 verdict', res['verdict'] == EXP)
chk('05 seal sha8',
    str(res['seal_sha8']) == 'bcd6fc5e',
    res['seal_sha8'])
chk('06 runtime',
    abs(float(res['runtime_s']) - 4472.1)
    < 60.0, res['runtime_s'])
chk('07 res43 sha link',
    res['part_a']['res43_sha8']
    == 'ce348ff5')
chk('08 xphase 1.0',
    float(res['part_a']['xphase_P']) == 1.0
    and float(res['part_a']['xphase_A1'])
    == 1.0)
# 2. seal file
sf = os.path.join(D44, 'design_seal.json')
seal = json.load(io.open(sf,
                         encoding='utf-8'))
chk('09 seal phase', seal['phase'] == 3144)
chk('10 seal smoke',
    seal['smoke'] is False)
chk('11 seal anchors len',
    len(seal['anchors']) >= 20)
# 3. npz
npzf = os.path.join(D44, 'p142_readout.npz')
chk('12 npz exists', os.path.exists(npzf))
z = np.load(npzf, allow_pickle=False)
for k in ('traj_pc1_wdn', 'traj_joint_wdn',
          'gap', 'perp_share',
          'resid_field_cos', 'cos19',
          'cos17', 'dvec19_sha',
          'd_head_tail_tailpos'):
    chk('13 npz key %s' % k, k in z.files)
chk('14 npz dvec19 sha',
    str(z['dvec19_sha'][0]) == '6a0332a6')
chk('15 npz perp share 38',
    abs(float(z['perp_share'][18])
        - 0.6405) < 1e-3,
    float(z['perp_share'][18]))
# 4. ckpt cleaned
ckf = os.path.join(D44, 'p142_ckpt.pkl')
chk('16 ckpt cleaned',
    not os.path.exists(ckf)
    or os.path.getsize(ckf) == 0)
# 5. run log
lg = io.open(os.path.join(D44,
                          'run_log.txt'),
             encoding='utf-8').read()
chk('17 log verdict line',
    'VERDICT: ' + EXP in lg)
chk('18 log DONE', 'DONE (verdict' in lg)
chk('19 log no traceback',
    'Traceback' not in lg)
chk('20 log 6/6 anchors',
    'B1 3142 repro: 3/3 bit-match' in lg
    and 'D repro: 3/3 bit-match' in lg)
# 6. result internal anchors
pt = res['part_traj']
chk('21 traj gap max',
    abs(pt['max_gap'] - (-0.2186)) < 0.01,
    pt['max_gap'])
chk('22 bit anchors B 3/3',
    len(pt['bit_anchors_3142']) == 3
    and all(v['match'] for v in
            pt['bit_anchors_3142'].values()))
pd = res['part_co36sign']
chk('23 bit anchors D 3/3',
    len(pd['bit_anchors']) == 3
    and all(v['match'] for v in
            pd['bit_anchors'].values()))
chk('24 d_head k25',
    abs(pd['d_head'] - 0.21875) < 1e-12)
chk('25 co50ex bit',
    abs(pd['d_co50ex'] - 0.421875)
    < 1e-12)
chk('26 ident overlap 16',
    pd['ident']['n_head_gt'] == 16)
chk('27 unemb orthogonal',
    res['part_unembed']['self_minus_rand']
    < 0)
pr = res['part_resid']
chk('28 resid tag',
    pr['resid_tag'] == 'd19resid_readout')
chk('29 perp wdn L39 dominant',
    pr['resid']['39']['perp_wdn_med']
    > pr['resid']['39']['dh19_wdn_med'],
    '%.3f vs %.3f'
    % (pr['resid']['39']['perp_wdn_med'],
       pr['resid']['39']['dh19_wdn_med']))
chk('30 dvec19 sha',
    res['part_field']['dvec19_sha8']
    == '6a0332a6')
# 7. ledger
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
chk('31 ledger n=281',
    len(led['measurements']) == 281,
    len(led['measurements']))
last = led['measurements'][-1]
chk('32 ledger last 3144',
    last['phase'] == 3144)
chk('33 ledger hashes',
    last['hashes']['result_sha256_8']
    == 'def947d8'
    and last['hashes']['seal_sha256_8']
    == 'bcd6fc5e')
# 8. MEMO
mt = io.open(MEMO, encoding='utf-8').read()
chk('34 memo 3144 once',
    mt.count('## Phase 3144') == 1)
chk('35 memo 3145 prereg',
    '3145（Ω-P143）预注册' in mt)
chk('36 memo anchors',
    'def947d8' in mt and 'bcd6fc5e' in mt
    and 'ledger n=281' in mt)
chk('37 memo key numbers',
    '0.21875' in mt and '2.4955' in mt
    and '−69.51' in mt)
# 9. daily log
dt = io.open(PLOG, encoding='utf-8').read()
chk('38 daily 3144',
    'Phase 3144 (Omega-P142) 闭环' in dt)
chk('39 daily sha',
    'def947d8' in dt)
# 10. MEMORY.md
mm = io.open(PMEM, encoding='utf-8').read()
chk('40 memory next 3145',
    'max=3144，下一 3145' in mm)
chk('41 memory findings',
    'd19resid_readout' in mm
    and 'identity_unembed_orthogonal'
    in mm.replace('unembed 正交',
                  'identity_unembed_'
                  'orthogonal'))
print('=== VERIFY %d/%d PASS, %d FAIL ==='
      % (n_pass, n_pass + n_fail, n_fail))
if n_fail:
    raise SystemExit(1)
