# -*- coding: utf-8 -*-
"""Phase 3133 disk verify: independent
re-derivation from frozen artifacts.
Sections: A meta/verdict/seal/npz,
B part_b recompute (fstep-based),
C part_c recompute, D ledger chain,
E docs."""
import hashlib
import io
import json
import os

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUTD = (RDIR + r'\phase3133'
        r'\omega_p131_transplant_'
        r'a1fork_migrate')
D32 = (RDIR + r'\phase3132'
       r'\omega_p130_forkcausal_single256')
LEDGER = (ROOT + r'\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (ROOT + r'\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WLOGS = [ROOT + r'\.workbuddy\memory',
         (r'C:\Users\Admin\WorkBuddy'
          r'\2026-09-17-01-30-05'
          r'\.workbuddy\memory')]
MEMW = ROOT + r'\.workbuddy\memory\MEMORY.md'
OUTF = (ROOT + r'\tests\gpt5_temp'
        r'\p3133_verify_out.txt')

REF32_CHG256 = {4: 0.96875, 8: 0.7890625,
                9: 0.76171875, 13: 0.4921875,
                17: 0.8828125, 29: 0.1875,
                33: 0.1640625, 38: 0.9921875}
LAYERS_C = [4, 8, 9, 13, 17, 29, 33, 38]
CAP_L = [17, 29, 33, 38]
TRIALS_B = ['t17_s0_d05', 't17_s0_d10',
            't17_s0_d20', 't17_as_d10',
            't38_s0_d10', 'tJ1_s0', 'tJ2_s0']
SEL64_SHA_REF = '9efd0f88'
SEL256_SHA_REF = 'e34d2588'
CO50_SHA_REF = '52b126af'
TR_FULL = 0.90
TR_PART = 0.10
ADD_TOL = 0.15
DOSE_SLACK = 0.02
RHO_PROF = 0.70
RHO_TRANS = 0.70
MED_E_TOL = 3
NP = 672
PREV_LED_SHA = '65d93639'

res = []
ok_all = True


def chk(sec, cond, msg):
    global ok_all
    tag = 'GREEN' if cond else 'RED'
    if not cond:
        ok_all = False
    res.append('%s %s: %s' % (tag, sec, msg))


r = json.load(io.open(
    OUTD + r'\result.json',
    encoding='utf-8'))
vd = r['verdict'].split('|')
pb = r['part_b']
pc = r['part_c']
z = np.load(OUTD + r'\p131_readout.npz',
            allow_pickle=False)
z32 = np.load(D32 + r'\p130_readout.npz',
              allow_pickle=False)

# ---------- A: meta / verdict / seal
chk('A1', r['name']
     == 'omega_p131_transplant_'
     'a1fork_migrate'
     and r['phase'] == 3133
     and r['smoke'] is False
     and r['runtime_s'] > 2000,
     'meta')
chk('A2', len(vd) == 10
     and vd[0] == 'a_3132_ok'
     and vd[1] in ('transplant_l17_full',
                   'transplant_l17_partial',
                   'transplant_l17_null')
     and vd[2] in ('dose_monotone',
                   'dose_nonmonotone')
     and vd[3] in ('l38_transplant_lower',
                   'l38_transplant_'
                   'comparable')
     and vd[4] in ('joint_additive',
                   'joint_superadditive',
                   'joint_subadditive')
     and vd[5] in ('step_transplant_'
                   'allstep_higher',
                   'step_transplant_'
                   'equivalent')
     and vd[6] in ('a1_forklayer_'
                   'symmetric',
                   'a1_forklayer_'
                   'asymmetric')
     and vd[7] in ('a1_transfer_high',
                   'a1_transfer_low')
     and vd[8] in ('a1_l17coord_rewrites',
                   'a1_l17coord_null')
     and vd[9] == 'coverage_full',
     'verdict 10-seg gates')
sha_seal = hashlib.sha256(io.open(
    OUTD + r'\design_seal.json', 'rb'
).read()).hexdigest()[:8]
chk('A3', r['seal_sha8'] == sha_seal,
    'seal sha8 recompute %s' % sha_seal)
need = {'co50', 'tf_idx', 'sel256',
        'dvec_med_norm',
        'fdm0_P', 'fdm0_A1',
        'fdiv_P', 'fdiv_A1',
        'emerge_P', 'emerge_A1'}
for l in CAP_L:
    need.add('dvec%d' % l)
    need.add('dvec_full_%d' % l)
for l in LAYERS_C:
    need.add('chgA1_%d' % l)
for t in TRIALS_B:
    need.add('fstep_%s' % t)
for s in (1, -1):
    need.add('fstep_c3_sgn%+d' % s)
chk('A4', need <= set(z.files),
    'npz keys (%d, need %d)'
    % (len(z.files), len(need)))

# ---------- B: part_b recompute
co50 = z['co50'].astype(np.int64)
sha_co = hashlib.sha256(
    co50.tobytes()).hexdigest()[:8]
chk('B1', sha_co == CO50_SHA_REF,
    'co50 sha8 %s' % sha_co)
dn = pb['dvec_med_norm']
ok_n = True
for l in CAP_L:
    d = z['dvec_full_%d' % l] \
        .astype(np.float64)
    med = float(np.median(
        np.linalg.norm(d, axis=1)))
    if abs(med - dn[str(l)]) > 0.5:
        ok_n = False
d17 = z['dvec17'].astype(np.float64)
med17 = float(np.median(
    np.linalg.norm(d17, axis=1)))
ok_n = ok_n and abs(med17
                    - dn['17']) < 1e-6
e_full = float((d17 * d17).sum(1).mean())
e_co = float((d17[:, co50] ** 2)
             .sum(1).mean())
frac = e_co / max(e_full, 1e-12)
ok_n = ok_n and abs(
    frac - pb['dvec17_co50_energy']) \
    < 1e-6
chk('B2', ok_n,
    'dvec norms + co50 energy recompute')
tt = pb['trials']
ok_f = set(tt) == set(TRIALS_B)
for t in TRIALS_B:
    fs = z['fstep_%s' % t]
    chg_re = 1.0 - float(
        (fs == -1).mean())
    first_re = int((fs == 0).sum())
    if abs(chg_re
           - tt[t]['chg']) > 1e-9:
        ok_f = False
    if first_re != tt[t]['first']:
        ok_f = False
chk('B3', ok_f,
    'trial chg/first recompute from '
    'fstep npz')
chg17 = pb['chg17_s0_d10']
chg17_05 = pb['chg17_s0_d05']
chg17_20 = pb['chg17_s0_d20']
chg17_as = pb['chg17_as_d10']
chg38 = pb['chg38_s0_d10']
chgJ1 = pb['chgJ1_s0']
chgJ2 = pb['chgJ2_s0']
b_trans_exp = ('transplant_l17_full'
               if chg17 >= TR_FULL
               else ('transplant_l17_partial'
                     if chg17 >= TR_PART
                     else 'transplant_l17_'
                          'null'))
chk('B4', vd[1] == b_trans_exp,
    'transplant gate %s' % vd[1])
dose_ok = (chg17_05 <= chg17 + DOSE_SLACK
           and chg17 <= chg17_20
           + DOSE_SLACK)
chk('B5', vd[2] == ('dose_monotone'
                    if dose_ok
                    else 'dose_nonmonotone'),
    'dose gate %s' % vd[2])
chk('B6', vd[3] == ('l38_transplant_lower'
                    if chg38 < 0.9
                    * max(chg17, 1e-9)
                    else 'l38_transplant_'
                    'comparable'),
    'L38 gate %s' % vd[3])
predJ1 = 1.0 - (1.0 - chg17) \
    * (1.0 - chg38)
joint_exp = ('joint_additive'
             if abs(chgJ1 - predJ1)
             <= ADD_TOL
             else ('joint_superadditive'
                   if chgJ1 > predJ1
                   else 'joint_subadditive'))
chk('B7', vd[4] == joint_exp
     and abs(predJ1
             - pb['predJ1_indep']) < 1e-12,
    'joint gate %s (pred %.4f)'
    % (vd[4], predJ1))
chk('B8', vd[5] == ('step_transplant_'
                    'allstep_higher'
                    if chg17_as - chg17
                    > 0.10
                    else 'step_transplant_'
                    'equivalent'),
    'step gate %s' % vd[5])

# ---------- C: part_c recompute
tf_idx = z['tf_idx']
chk('C1', len(tf_idx) == 128
     and len(set(tf_idx.tolist())) == 128
     and list(tf_idx) == pc['tf_idx'],
    'tf_idx consistency')
fdA1 = z['fdiv_A1']
fdP = z['fdiv_P']
medA1_re = np.median(
    np.abs(z['fdm0_A1'].astype(
        np.float64)), axis=0)
medP_re = np.median(
    np.abs(z['fdm0_P'].astype(
        np.float64)), axis=0)
ok_m = (np.allclose(medA1_re,
                    pc['med_dm0_A1'],
                    atol=1e-4)
        and np.allclose(medP_re,
                        pc['med_dm0_P'],
                        atol=1e-4))


def rankv(x):
    return np.argsort(
        np.argsort(np.asarray(
            x, dtype=np.float64)))


sp_prof_re = float(np.corrcoef(
    rankv(medA1_re),
    rankv(medP_re))[0, 1])
ok_m = ok_m and abs(
    sp_prof_re - pc['sp_prof']) < 1e-6
_a1pos = fdA1[fdA1 >= 0].astype(np.int32)
_ppos = fdP[fdP >= 0].astype(np.int32)
medE_A1_re = (int(np.median(_a1pos))
              if _a1pos.size else -1)
medE_P_re = (int(np.median(_ppos))
             if _ppos.size else -1)
ok_m = ok_m and medE_A1_re \
    == pc['medE_A1'] \
    and medE_P_re == pc['medE_P']
sym_exp = ('a1_forklayer_symmetric'
           if pc['sp_prof'] >= RHO_PROF
           and abs(pc['medE_A1']
                   - pc['medE_P'])
           <= MED_E_TOL
           else 'a1_forklayer_asymmetric')
ok_m = ok_m and vd[6] == sym_exp
chk('C2', ok_m,
    'fork profile recompute + gate %s '
    '(medE %d/%d)'
    % (vd[6], pc['medE_A1'],
       pc['medE_P']))
sel64_re = np.sort(
    np.random.default_rng(3131).choice(
        NP, 64, replace=False)).astype(
    np.int64)
rest = np.setdiff1d(np.arange(NP),
                    sel64_re)
sel192_re = np.sort(
    np.random.default_rng(3132).choice(
        rest, 192, replace=False)).astype(
    np.int64)
sel256_re = np.sort(np.concatenate(
    [sel64_re, sel192_re])).astype(
    np.int64)
sha256_re = hashlib.sha256(
    sel256_re.tobytes()).hexdigest()[:8]
chk('C3', sha256_re
     == SEL256_SHA_REF
     and np.array_equal(sel256_re,
                        z['sel256'])
     and sha256_re == pc['sel_sha8'],
    'sel256 chain (paired with 3132)')
c2a = pc['chgA1']
ok_s = True
for l in LAYERS_C:
    if abs(float(z['chgA1_%d' % l])
           - c2a[str(l)]) > 1e-12:
        ok_s = False
    if not 0.0 <= c2a[str(l)] <= 1.0:
        ok_s = False
vecA1 = [c2a[str(l)] for l in LAYERS_C]
vecP = [REF32_CHG256[l]
        for l in LAYERS_C]
sp_spec_re = float(np.corrcoef(
    rankv(vecA1), rankv(vecP))[0, 1])
ok_s = ok_s and abs(
    sp_spec_re - pc['sp_spec']) < 1e-6
ok_s = ok_s and vd[7] == (
    'a1_transfer_high'
    if pc['sp_spec'] >= RHO_TRANS
    else 'a1_transfer_low')
chk('C4', ok_s,
    'A1 spectrum recompute + gate %s '
    '(sp %.3f)' % (vd[7], pc['sp_spec']))
ok_c3 = True
for s in (1, -1):
    fs = z['fstep_c3_sgn%+d' % s]
    chg_re = 1.0 - float(
        (fs == -1).mean())
    if abs(chg_re
           - pc['c3_trials']['sgn%+d' % s]
           ['chg']) > 1e-9:
        ok_c3 = False
c3_best = max(v['chg'] for v in
              pc['c3_trials'].values())
ok_c3 = ok_c3 and vd[8] == (
    'a1_l17coord_rewrites'
    if c3_best >= TR_PART
    else 'a1_l17coord_null')
row_l2 = np.linalg.norm(
    z32['hout_l17'][:672].astype(
        np.float64), axis=1)
delta_re = 0.10 * float(row_l2.mean())
ok_c3 = ok_c3 and abs(
    pc['delta_l17'] - delta_re) < 1e-3
chk('C5', ok_c3,
    'C3 coords recompute + delta + gate '
    '%s' % vd[8])

# ---------- D: ledger
led = json.load(io.open(
    LEDGER, encoding='utf-8'))
ms = led['measurements']
ids = [m.get('meas_id', '')
       for m in ms]
chk('D1', ids.count(
    'meas3133_omega_p131_transplant_'
    'a1fork_migrate') == 1,
    'ledger entry unique')
led_rt = json.loads(json.dumps(led))
led_rt['ledger_sha256_8'] = PREV_LED_SHA
blob = json.dumps(led_rt, sort_keys=True,
                  ensure_ascii=False)
sha_led = hashlib.sha256(
    blob.encode('utf-8')).hexdigest()[:8]
chk('D2', sha_led
     == led['ledger_sha256_8'],
    'ledger sha chain (prev %s -> %s)'
    % (PREV_LED_SHA, sha_led))
chk('D3', len(ms) == 270,
    'ledger n=%d' % len(ms))

# ---------- E: docs
memo = io.open(MEMO,
               encoding='utf-8').read()
i3133 = memo.find('## Phase 3133:')
sec_ok = False
if i3133 >= 0:
    head = memo[i3133:memo.find(
        '\n', i3133)]
    seg = memo[i3133:i3133 + 20000]
    sec_ok = (len(head) < 110
              and all(('### %d.' % k)
                      in seg
                      for k in (1, 2, 3, 4,
                                5)))
chk('E1', sec_ok,
    'MEMO 3133 title<110 + 5 sections')
import datetime
WDATE = datetime.datetime.now()\
    .strftime('%Y-%m-%d')
w_ok = True
for wd in WLOGS:
    try:
        txt = io.open(
            wd + '\\' + WDATE + '.md',
            encoding='utf-8').read()
        w_ok = w_ok and (
            'Phase 3133 Omega-P131 closeout'
            in txt)
    except IOError:
        w_ok = False
chk('E2', w_ok, 'wlog x2 closeout lines')
mem = io.open(MEMW,
              encoding='utf-8').read()
chk('E3', ('3133\uff08T4\uff09' in mem)
     and ('max=3133' in mem)
     and ('下一 3134' in mem)
     and len(mem) < 3000,
     'MEMORY 3133 line + max + next '
     '(%d chars)' % len(mem))

res.append('VERIFY: %s (%d checks)'
           % ('GREEN' if ok_all else 'RED',
              len(res)))
with io.open(OUTF, 'w',
             encoding='utf-8') as f:
    f.write('\n'.join(res) + '\n')
print('verify done')
