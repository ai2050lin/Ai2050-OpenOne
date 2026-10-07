# -*- coding: utf-8 -*-
"""Phase 3143 independent disk verify."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        + r'\rdc_query_construction_20260913')
D43 = os.path.join(
    RDIR, 'phase3143',
    'omega_p141_d19field_readout_topk_'
    'newI')
LEDGER = (ROOT + r'\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (ROOT + r'\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
out = []
nok = 0


def chk(name, cond):
    global nok
    out.append('%s %s' % (
        'PASS' if cond else 'FAIL', name))
    if cond:
        nok += 1


# artifacts on disk
for fn in ('result.json', 'run_log.txt',
           'design_seal.json',
           'p141_readout.npz'):
    fp = os.path.join(D43, fn)
    chk('artifact %s' % fn,
        os.path.exists(fp) and
        os.path.getsize(fp) > 0)
raw = io.open(os.path.join(D43,
                           'result.json'),
              'rb').read()
chk('result sha8 ce348ff5',
    hashlib.sha256(raw).hexdigest()[:8]
    == 'ce348ff5')
res = json.loads(raw.decode('utf-8'))
chk('verdict exact', res['verdict'] ==
    ('a_3142_ok|repro_bit_5|'
     'repro_bit_ok|dvec19_repro_6a0332|'
     'field_self_ok|z35_cos17_ok|'
     'd19_path_partial|d19_direct_hi|'
     'stat_add|readout_comp_absent|'
     'pc1_readout_dark|head_conc_k13|'
     'iself_session|xphase_ok|'
     'coverage_full'))
chk('seal sha8 e7f52d03',
    str(res['seal_sha8']) == 'e7f52d03')
chk('smoke False', res['smoke'] is False)
chk('pc1_sign +1',
    res['part_readout']['pc1_sign'] == 1)
chk('xphase P/A1 = 1.0',
    res['part_a']['xphase_P'] == 1.0 and
    res['part_a']['xphase_A1'] == 1.0)
chk('5/5 bit anchors',
    all(v['match'] for v in
        res['part_readout']
        ['bit_anchors_3142'].values())
    and all(v['match'] for v in
            res['part_topk']
            ['bit_anchors'].values())
    and res['part_newI']
    ['bit_anchors']['f1_cross_retr']
    ['match'] is True)
chk('dvec19 sha 6a0332a6',
    res['part_field']['dvec19_sha8']
    == '6a0332a6' and
    res['part_field']['dvec19_sha_ok']
    is True)
# npz content
import numpy as np  # noqa: E402
z = np.load(os.path.join(D43,
                         'p141_readout.npz'),
            allow_pickle=False)
chk('npz cos19 shape',
    z['cos19'].shape == (40, 128))
d19np = z['dvec19_out'].astype(np.float32)
chk('npz dvec19 sha 6a0332a6',
    hashlib.sha256(
        d19np.tobytes()).hexdigest()[:8]
    == '6a0332a6')
# run_log verdict line
rl = io.open(os.path.join(D43,
                          'run_log.txt'),
             encoding='utf-8').read()
chk('run_log VERDICT line',
    'VERDICT: ' + res['verdict'] in rl)
chk('run_log ckpt cleaned',
    'ckpt cleaned (final)' in rl)
chk('run_log 5/5', 'repro: 5/5' in rl
    or '2/2 bit-match' in rl)
# ledger
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
chk('ledger n=280',
    len(led['measurements']) == 280)
last = led['measurements'][-1]
chk('ledger last phase 3143',
    last.get('phase') == 3143)
chk('ledger res sha',
    last['hashes']['result_sha256_8']
    == 'ce348ff5')
chk('ledger seal sha',
    last['hashes']['seal_sha256_8']
    == 'e7f52d03')
# MEMO
t = io.open(MEMO, encoding='utf-8').read()
chk('MEMO Phase 3143 x1',
    t.count('## Phase 3143') == 1)
chk('MEMO 3144 prereg',
    '3144（Ω-P142）预注册' in t)
chk('MEMO k25 finding',
    'k25 0.2188' in t)
chk('MEMO iself_session',
    t.count('iself_session') >= 1)
# daily log
p_log = (ROOT + r'\.workbuddy\memory'
         r'\2026-09-29.md')
tl = io.open(p_log, encoding='utf-8').read()
chk('daily log 3143 closed',
    'Phase 3143 (Omega-P141) 闭环' in tl)
# MEMORY.md
p_mem = (ROOT + r'\.workbuddy\memory'
         r'\MEMORY.md')
tm = io.open(p_mem, encoding='utf-8').read()
chk('MEMORY max=3143', 'max=3143' in tm)
chk('MEMORY 3144 next', '下一 3144' in tm)
out.append('TOTAL %d checks, %d PASS'
           % (len(out), nok))
io.open((ROOT + r'\tests\gpt5_temp'
         r'\p3143_disk_verify_report.txt'),
        'w', encoding='utf-8').write(
    chr(10).join(out))
print('OK %d/%d' % (nok, len(out) - 1))
