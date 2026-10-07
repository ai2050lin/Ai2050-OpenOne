# -*- coding: utf-8 -*-
"""Phase 3132 disk verify: independent
re-derivation from frozen artifacts.
Sections: A meta/verdict/seal/npz,
B part_b recompute, C part_c recompute,
D ledger chain, E docs."""
import hashlib
import io
import json
import os

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUTD = (RDIR + r'\phase3132'
        r'\omega_p130_forkcausal_'
        r'single256')
D26 = (RDIR + r'\phase3126'
       r'\omega_p124_glm4_anchoredlast_'
       r'regen_writechain')
D31 = (RDIR + r'\phase3131'
       r'\omega_p129_layerwindow_'
       r'single40_a1gen_forklayer')
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
        r'\p3132_verify_out.txt')

REF30 = {8: 208 / 256.0, 9: 205 / 256.0,
         13: 137 / 256.0, 29: 45 / 256.0}
CHG40_REF = {4: 0.96875, 8: 0.734375,
             9: 0.78125, 13: 0.421875,
             17: 0.84375, 29: 0.109375,
             33: 0.09375, 38: 1.0}
LAYERS_C = [4, 8, 9, 13, 17, 29, 33, 38]
SEL64_SHA_REF = '9efd0f88'
SPEC256_TOL = 0.08
SUB64_TOL = 0.033
FORK_LO = 15
FORK_HI = 19
PREV_LED_SHA = '4eacba5f'
NP = 672

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
z = np.load(OUTD + r'\p130_readout.npz',
            allow_pickle=False)
z31 = np.load(D31 + r'\p129_readout.npz',
              allow_pickle=False)

# ---------- A: meta / verdict / seal
chk('A1', r['name']
     == 'omega_p130_forkcausal_single256'
     and r['phase'] == 3132
     and r['smoke'] is False
     and r['runtime_s'] > 2000,
     'meta')
chk('A2', len(vd) == 7
     and vd[0] == 'a_3131_ok'
     and vd[1] in ('l17_inj_rewrites',
                   'l17_inj_ambiguous',
                   'l17_inj_null')
     and vd[2] in ('rescue_l17_present',
                   'rescue_not_specific',
                   'rescue_absent')
     and vd[3] in ('ctrl33_quiescent',
                   'ctrl33_active')
     and vd[4] in ('spec256_ok',
                   'spec256_dev')
     and vd[5] in ('fork_l17_gate',
                   'fork_l17_amp',
                   'fork_anom')
     and vd[6] == 'coverage_full',
     'verdict 7-seg gates')
sha_seal = hashlib.sha256(io.open(
    OUTD + r'\design_seal.json', 'rb'
).read()).hexdigest()[:8]
chk('A3', r['seal_sha8'] == sha_seal,
    'seal sha8 recompute %s' % sha_seal)
need = {'co50', 'rho_l17', 'hout_l17',
        'dm31_med17', 'dm31_frac_neg17',
        'tf_idx', 'med_tf_p1', 'med_tf_m1',
        'med_tf_avg', 'sel64', 'sel256'}
for l in LAYERS_C:
    need.add('chg256_%d' % l)
    need.add('same256_%d' % l)
# [rev-3132a-fix] ctrl layers 38/33 have
# step0-only trials (frozen design);
# B3 rescue likewise has no 38|allstep|.
# tf_* keys added by rev-3132a.
for k in ('17', '38', '33'):
    ms = (('step0', 'allstep')
          if k == '17' else ('step0',))
    for m in ms:
        for s in ('p1', 'm1'):
            need.add('same_L%s_%s_%s'
                     % (k, m, s))
for k in ('17', '38'):
    ms = (('step0', 'allstep')
          if k == '17' else ('step0',))
    for m in ms:
        for s in ('p1', 'm1'):
            need.add('sameR_L%s_%s_%s'
                     % (k, m, s))
need |= {'tf_df_last_p1', 'tf_df_last_m1',
         'tf_dk0_med_p1', 'tf_dk0_med_m1'}
chk('A4', need <= set(z.files),
    'npz keys (%d, need %d)'
    % (len(z.files), len(need)))

# ---------- B: part_b recompute
chk('B1', abs(pb['delta_l17']
              - 0.10 * pb['mean_row_l2'])
     < 1e-12 and len(pb['co50_sha8']) == 8,
     'delta_l17 consistency')
it = pb['inj_trials']
k17 = ['17|step0|+1', '17|step0|-1',
       '17|allstep|+1', '17|allstep|-1']
k38 = ['38|step0|+1', '38|step0|-1']
k33 = ['33|step0|+1', '33|step0|-1']
chk('B2', len(it) == 8
     and set(it) == set(k17 + k38 + k33)
     and abs(pb['chg_l17_best']
             - max(it[k]['chg']
                   for k in k17)) < 1e-12
     and abs(pb['chg_l38_best']
             - max(it[k]['chg']
                   for k in k38)) < 1e-12
     and abs(pb['chg_l33_best']
             - max(it[k]['chg']
                   for k in k33)) < 1e-12,
     'inj trials best recompute')
chg17 = pb['chg_l17_best']
chg33 = pb['chg_l33_best']
l17_exp = ('l17_inj_rewrites'
           if chg17 >= 0.10 and chg17 > chg33
           else ('l17_inj_ambiguous'
                 if chg17 >= 0.10
                 else 'l17_inj_null'))
chk('B3', vd[1] == l17_exp,
    'l17 gate %s' % vd[1])
ctrl_exp = ('ctrl33_quiescent'
            if chg33 < 0.5 * max(chg17, 0.10)
            else 'ctrl33_active')
chk('B4', vd[3] == ctrl_exp,
    'ctrl gate %s' % vd[3])
rt = pb['resc_trials']
rk17 = k17
rk38 = k38
resc17 = max(rt[k]['rescue']
             for k in rk17)
resc38 = max(rt[k]['rescue']
             for k in rk38)
chk('B5', len(rt) == 6
     and abs(pb['rescue_l17'] - resc17)
     < 1e-12
     and abs(pb['rescue_l38'] - resc38)
     < 1e-12,
     'rescue recompute')
resc_exp = ('rescue_l17_present'
            if resc17 >= 0.10
            and resc17 > resc38
            else ('rescue_not_specific'
                  if resc38 >= 0.10
                  else 'rescue_absent'))
chk('B6', vd[2] == resc_exp,
    'rescue gate %s' % vd[2])
mta = pb['med_tf_avg']
# [rev-3132a] zero region = index 0..18
# (inject at layers[17] output pos0,
# track-last profile)
chk('B7', len(mta) == 41
     and all(abs(v) < 1e-9
             for v in mta[:19])
     and mta[19] > 0
     and pb['peak_tf'] == int(max(
         range(41),
         key=lambda i: mta[i])),
     'tf profile zero-region + peak')
pk_tf = pb['peak_tf']
fork_exp = ('fork_l17_gate'
            if 15 <= pk_tf <= 19
            else ('fork_l17_amp'
                  if pk_tf > 19
                  else 'fork_anom'))
chk('B8', vd[5] == fork_exp,
    'fork gate %s (peak L%02d)'
    % (vd[5], pk_tf))
mad31 = z31['med_abs_dm'].astype(np.float64)


def rankv(x):
    return np.argsort(
        np.argsort(x)).astype(np.float64)


sp_re = float(np.corrcoef(
    rankv(np.asarray(mta)),
    rankv(mad31))[0, 1])
chk('B9', abs(sp_re
              - pb['spearman_tf_vs_3131'])
     < 1e-6,
     'spearman recompute %.6f' % sp_re)
chk('B10', abs(float(z['dm31_med17'])
               - pb['dm_med17']) < 1e-12
     and abs(float(z['dm31_frac_neg17'])
             - pb['dm_frac_neg17']) < 1e-12,
     'dm sign npz vs result')
dm31 = z31['dm'].astype(np.float64)
chk('B11', abs(float(np.median(dm31[:, 17]))
               - pb['dm_med17']) < 1e-9,
     'dm med17 vs 3131 npz')

# ---------- C: part_c recompute
sel64_re = np.sort(
    np.random.default_rng(3131).choice(
        NP, 64, replace=False)).astype(
    np.int64)
sha64 = hashlib.sha256(
    sel64_re.tobytes()).hexdigest()[:8]
rest = np.setdiff1d(np.arange(NP), sel64_re)
sel192_re = np.sort(
    np.random.default_rng(3132).choice(
        rest, 192, replace=False)).astype(
    np.int64)
sel256_re = np.sort(np.concatenate(
    [sel64_re, sel192_re])).astype(np.int64)
sha256_re = hashlib.sha256(
    sel256_re.tobytes()).hexdigest()[:8]
chk('C1', sha64 == SEL64_SHA_REF
     and np.array_equal(sel64_re,
                        z['sel64'])
     and np.array_equal(sel256_re,
                        z['sel256'])
     and sha256_re == pc['sel_sha8']
     and bool(np.all(np.isin(sel64_re,
                             sel256_re))),
     'sel64/sel256 chain (nested)')
c2 = pc['chg256']
ok_c2 = True
for l in LAYERS_C:
    sm = z['same256_%d' % l]
    chg_re = 1.0 - float(sm.mean())
    if abs(chg_re - c2[str(l)]) > 1e-12:
        ok_c2 = False
    if abs(float(z['chg256_%d' % l])
           - c2[str(l)]) > 1e-12:
        ok_c2 = False
chk('C2', ok_c2,
    'chg256 recompute from same256 npz')
dev30_re = {str(l): abs(c2[str(l)]
                        - REF30[l])
            for l in REF30}
chk('C3', all(abs(dev30_re[k]
                  - pc['dev30'][k]) < 1e-12
              for k in dev30_re)
     and (vd[4] == 'spec256_ok')
     == all(v <= SPEC256_TOL
            for v in dev30_re.values()),
     'dev30 recompute + spec gate %s'
     % vd[4])
pos64 = np.array(
    [int(np.where(sel256_re == j)[0][0])
     for j in sel64_re], dtype=np.int64)
dev31_re = {}
for l in LAYERS_C:
    sm = z['same256_%d' % l][pos64]
    chg64 = 1.0 - float(sm.mean())
    dev31_re[str(l)] = abs(
        chg64 - CHG40_REF[l])
ok_c4 = all(abs(dev31_re[k]
                - pc['dev31'][k]) < 1e-9
            for k in dev31_re)
n_over = sum(1 for v in dev31_re.values()
             if v > SUB64_TOL)
chk('C4', ok_c4
     and pc['n_dev31_over'] == n_over,
     'dev31 recompute (64-subset, '
     'over=%d, max=%.4f)'
     % (n_over, max(dev31_re.values())))

# ---------- D: ledger
led = json.load(io.open(
    LEDGER, encoding='utf-8'))
ms = led['measurements']
ids = [m.get('meas_id', '')
       for m in ms]
chk('D1', ids.count(
    'meas3132_omega_p130_forkcausal_'
    'single256') == 1,
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
chk('D3', len(ms) == 269,
    'ledger n=%d' % len(ms))

# ---------- E: docs
memo = io.open(MEMO,
               encoding='utf-8').read()
i3132 = memo.find('## Phase 3132:')
sec_ok = False
if i3132 >= 0:
    head = memo[i3132:memo.find(
        '\n', i3132)]
    seg = memo[i3132:i3132 + 20000]
    sec_ok = (len(head) < 110
              and all(('### %d.' % k)
                      in seg
                      for k in (1, 2, 3, 4,
                                5)))
chk('E1', sec_ok,
    'MEMO 3132 title<110 + 5 sections')
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
            'Phase 3132 Omega-P130 closeout'
            in txt)
    except IOError:
        w_ok = False
chk('E2', w_ok, 'wlog x2 closeout lines')
mem = io.open(MEMW,
              encoding='utf-8').read()
chk('E3', ('3132\uff08T4\uff09' in mem)
     and ('max=3132' in mem)
     and ('\u4e0b\u4e00 3133' in mem)
     and len(mem) < 3000,
     'MEMORY 3132 line + max + next '
     '(%d chars)' % len(mem))

res.append('VERIFY: %s (%d checks)'
           % ('GREEN' if ok_all else 'RED',
              len(res)))
with io.open(OUTF, 'w',
             encoding='utf-8') as f:
    f.write('\n'.join(res) + '\n')
print('verify done')
