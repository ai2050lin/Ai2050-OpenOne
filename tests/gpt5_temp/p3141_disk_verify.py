# -*- coding: utf-8 -*-
"""Phase 3141 independent disk verify."""
import hashlib
import io
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        + r'\rdc_query_construction_20260913')
D14 = os.path.join(
    RDIR, 'phase3141',
    'omega_p139_cinjrefine_wrjoint_'
    'coenrich_writeinduce')
REP = (ROOT + r'\tests\gpt5_temp'
       r'\p3141_disk_verify_report.txt')
ok = []
bad = []


def chk(name, cond):
    if cond:
        ok.append(name)
    else:
        bad.append(name)


# --- artifacts on disk ---
for fn in ('result.json', 'design_seal.json',
           'p139_readout.npz', 'run_log.txt'):
    chk('file:' + fn,
        os.path.exists(os.path.join(D14, fn)))
raw = io.open(os.path.join(D14,
                           'result.json'),
              'rb').read()
chk('res sha8 a93c8892',
    hashlib.sha256(raw).hexdigest()[:8]
    == 'a93c8892')
resj = json.loads(raw.decode('utf-8'))
chk('verdict match',
    resj['verdict'] == (
        'a_3140_ok|repro_bit_9|'
        'repro_bit_ok|step1_peak_l19|'
        'step1_below_all|step1_dose_flat|'
        'peak_sharpened|dvec29_active|'
        'joint_blocking|'
        'co_enrich_causal_mixed|'
        'xphase_ok|v1fwd_bit_ok|'
        'writeinduce_flat|gen_retr_below|'
        'coverage_full'))
chk('seal sha 7ccf7ffc',
    str(resj['seal_sha8']) == '7ccf7ffc')
chk('smoke False', resj['smoke'] is False)
pc = resj['part_c']
chk('xphase P=1.0', pc['xphase_P'] == 1.0)
chk('xphase A1=1.0',
    pc['xphase_A1'] == 1.0)
chk('iinj17 bit', abs(
    pc['iinj17_d1'] - 0.328125) < 1e-12)
ba = pc['bit_anchors_3140']
chk('5 C bit anchors',
    all(v['match'] for v in ba.values())
    and len(ba) == 5)
chk('all_d2 L26 bit', abs(
    pc['allstep_d2']['26'] - 0.2890625)
    < 1e-12)
chk('all_d2 L24 0.3203', abs(
    pc['allstep_d2']['24'] - 0.3203125)
    < 1e-12)
chk('s1 L19 d2 0.1406', abs(
    pc['step1']['2.0']['19'] - 0.140625)
    < 1e-12)
chk('sharp 2.0', abs(pc['sharp_s1']
                     - 2.0) < 1e-9)
pd_ = resj['part_d']
chk('wrpc1 d2 bit', abs(
    pd_['pc1_d2'] - 0.203125) < 1e-12)
chk('dvec29 d2 0.6406', abs(
    pd_['dvec29_d2'] - 0.640625) < 1e-12)
chk('joint d2 0.4766', abs(
    pd_['joint_d2'] - 0.4765625) < 1e-12)
chk('law blocking',
    pd_['law_d2'] == 'joint_blocking')
pe = resj['part_e']
chk('co36 d1 bit', abs(
    pe['co36_d1'] - 0.140625) < 1e-12)
chk('co36 d2 bit', abs(
    pe['co36_d2'] - 0.2265625) < 1e-12)
chk('co50 d1 bit', abs(
    pe['co50_d1'] - 0.421875) < 1e-12)
chk('e_bit 3/3', pe['e_bit_3137'] == 3)
chk('hi_d1 0.2188', abs(
    pe['hi_d1'] - 0.21875) < 1e-12)
chk('lo_d1 0.0938', abs(
    pe['lo_d1'] - 0.09375) < 1e-12)
chk('high25 len 25', len(pe['high25']) == 25)
chk('low25 len 25', len(pe['low25']) == 25)
chk('high/low disjoint',
    not set(pe['high25'])
    & set(pe['low25']))
pf = resj['part_f']
chk('n_pairs 84', pf['n_pairs'] == 84)
chk('V1 fwd bit 3140', (
    abs(pf['retr_fwd_v1']['17']
        - 0.05952380952380952) < 1e-12
    and abs(pf['retr_fwd_v1']['29']
            - 0.03571428571428571) < 1e-12
    and abs(pf['retr_fwd_v1']['38']
            - 0.05952380952380952) < 1e-12))
chk('v1fwd_bit flag',
    pf['v1_fwd_bit_3140'] is True)
chk('gen best 0.0595', abs(
    pf['gen_best'] - 0.05952380952380952)
    < 1e-12)

# --- ledger ---
led = json.load(io.open(
    ROOT + r'\research\gpt5\atlas'
    r'\atlas_ledger.json',
    encoding='utf-8'))
chk('ledger n=278',
    len(led['measurements']) == 278)
last = led['measurements'][-1]
chk('ledger last phase 3141',
    last.get('phase') == 3141)
chk('ledger sha8 a93c8892',
    last['hashes']['result_sha256_8']
    == 'a93c8892')

# --- MEMO ---
t = io.open(ROOT + r'\research\gpt5\docs'
            r'\AGI_GPT5_MEMO.md',
            encoding='utf-8').read()
chk('MEMO has 3141 hdr',
    t.count('## Phase 3141') == 1)
chk('MEMO T4 第24Phase',
    'T4 第24' in t)
chk('MEMO prereg 3142',
    '3142（Ω-P140）预注册' in t)
chk('MEMO finding x3',
    t.count('发现 1（×3）') >= 1
    and t.count('发现 4（×3）') >= 1)

# --- wlog ---
tw = io.open(ROOT + r'\.workbuddy\memory'
             r'\2026-09-29.md',
             encoding='utf-8').read()
chk('wlog 3141 closed',
    'Phase 3141 (Omega-P139) 闭环' in tw)

# --- MEMORY.md ---
tm = io.open(ROOT + r'\.workbuddy\memory'
             r'\MEMORY.md',
             encoding='utf-8').read()
chk('MEMORY max=3141', 'max=3141' in tm)
chk('MEMORY 3141 line',
    '3141（T4）' in tm)

# --- run_log verdict ---
rl = io.open(os.path.join(D14,
                          'run_log.txt'),
             encoding='utf-8').read()
chk('run_log VERDICT line',
    'VERDICT: ' + resj['verdict'] in rl)
chk('run_log COMPLETE',
    'PHASE 3141 COMPLETE' in rl)

rep = ['PASS %d' % len(ok)]
rep += ['  ' + s for s in ok]
if bad:
    rep.append('FAIL %d' % len(bad))
    rep += ['  ' + s for s in bad]
else:
    rep.append('FAIL 0')
rep.append('TOTAL %d' % (len(ok) + len(bad)))
io.open(REP, 'w',
        encoding='utf-8').write(
    '\n'.join(rep) + '\n')
print('verify done: %d PASS, %d FAIL'
      % (len(ok), len(bad)))
