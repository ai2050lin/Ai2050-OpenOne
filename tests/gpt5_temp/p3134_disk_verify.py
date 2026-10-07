# -*- coding: utf-8 -*-
"""Phase 3134 disk verify: independent
re-derivation from frozen artifacts.
Sections: A meta/verdict/seal/npz,
B part_b recompute (fstep-based),
C part_c/d recompute, D ledger chain,
E docs."""
import hashlib
import io
import json
import os

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUTD = (RDIR + r'\phase3134'
        r'\omega_p132_carrier_matrix_'
        r'forkcoord_stepscan')
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
        r'\p3134_verify_out.txt')

CAP_L = [17, 29, 33, 38]
DOSES = (0.5, 1.0, 2.0)
PAIRS = [(17, 29), (17, 33), (29, 33)]
STEPS_SCAN = ['0', '1', '2', '3', '5', '8',
              '11']
CO50_SHA_REF = '52b126af'
TR_PART = 0.10
ADD_TOL = 0.15
DOSE_SLACK = 0.02
J2_TOL = 0.05
STEP_GATE = 0.20
FLIP_TOL = 3
FORK_PROF_IDX = 36
CHGJ2_REF = 258 / 672.0
PREV_LED_SHA = 'd3c75ef9'

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
pd_ = r['part_d']
z = np.load(OUTD + r'\p132_readout.npz',
            allow_pickle=False)

# ---------- A: meta / verdict / seal
chk('A1', r['name']
     == 'omega_p132_carrier_matrix_'
     'forkcoord_stepscan'
     and r['phase'] == 3134
     and r['smoke'] is False
     and r['runtime_s'] > 2000,
     'meta')
chk('A2', len(vd) == 9
     and vd[0] == 'a_3133_ok'
     and vd[1] in ('carrier_l17_max',
                   'carrier_dispersed',
                   'carrier_mixed')
     and vd[2] in ('dose_monotone_all',
                   'dose_nonmonotone')
     and vd[3] in ('j2_session_match',
                   'j2_session_drift')
     and vd[4] in ('pair_additive',
                   'pair_superadditive')
     and vd[5] in ('forkcoord_p_rewrite',
                   'forkcoord_p_null')
     and vd[6] in ('stepscan_effective',
                   'stepscan_weak')
     and vd[7].startswith('traj_swap_')
     and vd[8] == 'coverage_full',
     'verdict 9-seg gates')
sha_seal = hashlib.sha256(io.open(
    OUTD + r'\design_seal.json', 'rb'
).read()).hexdigest()[:8]
chk('A3', r['seal_sha8'] == sha_seal,
    'seal sha8 recompute %s' % sha_seal)
need = {'tf_idx', 'co36', 'dvec_med_norm',
        'med_traj_P', 'med_traj_A1',
        'dm_swap_traj', 'dm_inj_traj',
        'second_diff',
        'fstep_stepscan_full'}
for l in CAP_L:
    need.add('dvec%d' % l)
    need.add('dvec_full_%d' % l)
exp_names = set()
for l in CAP_L:
    for d in DOSES:
        exp_names.add('l%02d_d0%d'
                      % (l, int(d * 10)))
for (a, b) in PAIRS:
    exp_names.add('p%02d_%02d' % (a, b))
exp_names.add('j2_s0')
for t in exp_names:
    need.add('fstep_%s' % t)
chk('A4', need <= set(z.files),
    'npz keys (%d, need %d)'
    % (len(z.files), len(need)))

# ---------- B: part_b recompute
co50_ref = np.load(
    RDIR + r'\phase3132'
    r'\omega_p130_forkcausal_single256'
    + r'\p130_readout.npz',
    allow_pickle=False)['co50']
sha_co = hashlib.sha256(
    co50_ref.astype(np.int64).tobytes()
).hexdigest()[:8]
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
chk('B2', ok_n,
    'dvec norms recompute (fp16/full '
    'tol 0.5, fp32/B tol 1e-6)')
tt = pb['trials']
ok_f = set(tt) == exp_names
for t in exp_names:
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
    '16-trial chg/first recompute from '
    'fstep npz')
cm = pb['chg_matrix']
d2s = {l: cm[str(l)][2] for l in CAP_L}
if d2s[17] >= max(d2s.values()):
    carrier_exp = 'carrier_l17_max'
elif max(d2s[l] for l in CAP_L
         if l != 17) > 2.0 * d2s[17]:
    carrier_exp = 'carrier_dispersed'
else:
    carrier_exp = 'carrier_mixed'
chk('B4', vd[1] == carrier_exp,
    'carrier gate %s (d2 L17 %.4f max '
    'other %.4f)' % (vd[1], d2s[17],
                     max(d2s[l] for l in
                         CAP_L
                         if l != 17)))
mono_ok = all(
    cm[str(l)][0] <= cm[str(l)][1]
    + DOSE_SLACK
    and cm[str(l)][1] <= cm[str(l)][2]
    + DOSE_SLACK for l in CAP_L)
chk('B5', vd[2] == ('dose_monotone_all'
                    if mono_ok
                    else 'dose_nonmonotone')
     and vd[3] == ('j2_session_match'
                   if abs(
                       tt['j2_s0']['chg']
                       - CHGJ2_REF)
                   <= J2_TOL
                   else 'j2_session_drift'),
    'dose+j2 gates %s|%s' % (vd[2], vd[3]))
pair_ok = True
for (a, b) in PAIRS:
    pa = tt['l%02d_d010' % a]['chg']
    pbb = tt['l%02d_d010' % b]['chg']
    pred = 1.0 - (1.0 - pa) * (1.0 - pbb)
    if abs(pb['pair_pred']
           ['%02d_%02d' % (a, b)]
           - pred) > 1e-12:
        pair_ok = False
    if abs(tt['p%02d_%02d' % (a, b)]
           ['chg'] - pred) > ADD_TOL:
        pair_ok = False
chk('B6', vd[4] == ('pair_additive'
                    if pair_ok
                    else 'pair_'
                    'superadditive'),
    'pair gate %s' % vd[4])

# ---------- C: part_c/d recompute
co36 = z['co36']
chk('C1', len(co36) == 50
     and len(set(co36.tolist())) == 50
     and hashlib.sha256(
         co36.tobytes()).hexdigest()[:8]
     == pc['co36_sha8']
     and pc['fork_layer'] == 35
     and pc['fork_prof_idx'] == 36
     and vd[5] == (
         'forkcoord_p_rewrite'
         if pc['c1_trials']['P']['chg']
         >= TR_PART
         else 'forkcoord_p_null'),
    'co36 chain + fork gate %s (P chg '
    '%.4f)' % (vd[5],
               pc['c1_trials']['P']['chg']))
scan = pc['scan']
best_s = max(scan.keys(),
             key=lambda k: scan[k]['chg'])
fs_full = z['fstep_stepscan_full']
chg_full_re = 1.0 - float(
    (fs_full == -1).mean())
ok_s = (pc['best_step'] == best_s
        and vd[6] == ('stepscan_effective'
                      if scan[best_s]['chg']
                      >= STEP_GATE
                      else 'stepscan_weak')
        and pc['c3_full']['step']
        == int(best_s)
        and abs(chg_full_re
                - pc['c3_full']['chg'])
        < 1e-9
        and abs(scan[best_s]['chg']
                - pc['c3_full']['chg'])
        < 1e-9)
chk('C2', ok_s,
    'step scan recompute + gate %s (best '
    '%s chg %.4f)' % (vd[6], best_s,
                      scan[best_s]['chg']))
tf_idx = z['tf_idx']
chk('C3', len(tf_idx) == 128
     and len(set(tf_idx.tolist())) == 128
     and list(tf_idx) == pd_['tf_idx'],
    'tf_idx consistency')
dm_swap = z['dm_swap_traj']
dm_inj = z['dm_inj_traj']
sd_re = dm_swap - dm_inj
# [rev-3134a] mirror the soft band gate:
# peak in [33, 40] -> traj_swap_peak%d;
# otherwise traj_swap_early%d (recorded
# refutation, not a crash). The old
# FLIP_TOL assert conflated emergence
# (medE=36) with |dm| peak.
_k0 = int(np.argmax(
    np.abs(dm_swap[:, 0])))
_exp_tok = ('traj_swap_peak%d' % _k0
            if 33 <= _k0 <= 40
            else 'traj_swap_early%d'
            % _k0)
ok_d = (dm_swap.shape == (41, 13)
        and dm_inj.shape == (41, 13)
        and z['med_traj_P'].shape == (41, 13)
        and z['med_traj_A1'].shape == (41, 13)
        and np.allclose(
            sd_re, z['second_diff'],
            atol=1e-6)
        and _k0 == pd_['k0_swap_peak']
        and vd[7] == _exp_tok)
chk('C4', ok_d,
    'traj profiles + second-diff + swap '
    'peak %d token %s' % (_k0, vd[7]))
sds = pd_['second_diff_summary']
chk('C5', abs(sds['med_abs']
              - float(np.median(
                  np.abs(sd_re)))) < 1e-6
     and abs(sds['max_abs']
             - float(np.max(
                 np.abs(sd_re)))) < 1e-6
     and sds['max_abs'] >= sds['med_abs'],
    'second-diff summary recompute')

# ---------- D: ledger
led = json.load(io.open(
    LEDGER, encoding='utf-8'))
ms = led['measurements']
ids = [m.get('meas_id', '')
       for m in ms]
chk('D1', ids.count(
    'meas3134_omega_p132_carrier_matrix_'
    'forkcoord_stepscan') == 1,
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
chk('D3', len(ms) == 271,
    'ledger n=%d' % len(ms))

# ---------- E: docs
memo = io.open(MEMO,
               encoding='utf-8').read()
i3134 = memo.find('## Phase 3134:')
sec_ok = False
if i3134 >= 0:
    head = memo[i3134:memo.find(
        '\n', i3134)]
    seg = memo[i3134:i3134 + 20000]
    sec_ok = (len(head) < 110
              and all(('### %d.' % k)
                      in seg
                      for k in (1, 2, 3, 4,
                                5)))
chk('E1', sec_ok,
    'MEMO 3134 title<110 + 5 sections')
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
            'Phase 3134 Omega-P132 closeout'
            in txt)
    except IOError:
        w_ok = False
chk('E2', w_ok, 'wlog x2 closeout lines')
mem = io.open(MEMW,
              encoding='utf-8').read()
chk('E3', ('3134\uff08T4\uff09' in mem)
     and ('max=3134' in mem)
     and ('下一 3135' in mem)
     and len(mem) < 3000,
     'MEMORY 3134 line + max + next '
     '(%d chars)' % len(mem))

res.append('VERIFY: %s (%d checks)'
           % ('GREEN' if ok_all else 'RED',
              len(res)))
with io.open(OUTF, 'w',
             encoding='utf-8') as f:
    f.write('\n'.join(res) + '\n')
print('verify done')
