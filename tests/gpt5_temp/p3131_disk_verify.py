# -*- coding: utf-8 -*-
"""Phase 3131 disk verify: independent
re-derivation from disk artifacts.
Sections: A meta, B window, C part-c,
D ledger, E memo/wlog/memory."""
import hashlib
import io
import json

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
BASE = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3131'
        r'\omega_p129_layerwindow_'
        r'single40_a1gen_forklayer')
D30 = (ROOT + r'\tests\glm5\result'
       r'\rdc_query_construction_20260913'
       r'\phase3130'
       r'\omega_p128_positionspectrum_'
       r'a1fit_single_swap_dyn')
LEDGER = (ROOT + r'\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (ROOT + r'\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WLOGS = [ROOT + r'\.workbuddy\memory',
         (r'C:\Users\Admin\WorkBuddy'
          r'\2026-09-17-01-30-05'
          r'\.workbuddy\memory')]
MEMW = ROOT + r'\.workbuddy\memory\MEMORY.md'
OUTP = (ROOT + r'\tests\gpt5_temp'
        r'\p3131_verify_out.txt')

F10_REF = 0.18229166666666666
MEDDM_REF = 1.0975285470485687
DELTA_Q_REF = 10.694569431733811
WIN_LAYERS = [20, 22, 24, 26, 28]

res = json.load(io.open(
    BASE + r'\result.json',
    encoding='utf-8'))
zn = np.load(BASE + r'\p129_readout.npz',
             allow_pickle=False)
z30 = np.load(D30 + r'\p128_readout.npz',
              allow_pickle=False)

P = []
ok = []


def chk(sec, cond, note=''):
    P.append('%s %s %s'
             % ('PASS' if cond else 'FAIL',
                sec, note))
    ok.append(bool(cond))


# ---- A meta ----
chk('A1', res['smoke'] is False
    and res['phase'] == 3131
    and res['runtime_s'] > 5000,
    'meta smoke=False runtime>5000')
V = res['verdict']
vd = V.split('|')
gates_ok = (
    len(vd) == 7 and vd[0] == 'a_3130_ok'
    and vd[1] in ('qwen_win_wide',
                  'qwen_win_narrow',
                  'qwen_win_partial')
    and vd[2] == 'gen_swap_3130_bitexact'
    and vd[3] in ('single40_dominant',
                  'single40_distributed')
    and vd[4] in ('a1_gen_symmetric',
                  'a1_gen_asymmetric')
    and vd[5] in ('fork_late', 'fork_mid',
                  'fork_early')
    and vd[6] == 'coverage_full')
chk('A2', gates_ok, 'verdict 7-seg gates')
sha8 = hashlib.sha256(io.open(
    BASE + r'\result.json', 'rb').read()
).hexdigest()[:8]
sha_seal = hashlib.sha256(io.open(
    BASE + r'\design_seal.json', 'rb')
    .read()).hexdigest()[:8]
chk('A3', res['seal_sha8'] == sha_seal,
    'seal file sha8 recompute %s '
    '(result.json sha8 %s)'
    % (sha_seal, sha8))
keys = ['win_ks', 'win_flip',
        'win_med_dm', 'gen_same_p',
        'gen_same_a1', 'hist_a1', 'sel',
        'chg40', 'first40', 'dm',
        'med_abs_dm', 'med_dm_first',
        'med_dm_rest']
chk('A4', sorted(zn.files)
    == sorted(keys), 'npz 13 keys')

# ---- B window ----
wf = np.asarray(zn['win_flip'])
wm = np.asarray(zn['win_med_dm'])
ks = [int(v) for v in zn['win_ks']]
pb = res['part_b']
chk('B1', ks == WIN_LAYERS
    and np.allclose(
        wf, pb['win_flip'], atol=1e-12)
    and np.allclose(
        wm, pb['win_med_dm'], atol=1e-12),
    'win 5-pt npz==result')
i24 = WIN_LAYERS.index(24)
chk('B2', abs(wf[i24] - F10_REF) < 1e-9
    and abs(wm[i24] - MEDDM_REF) < 1e-9
    and abs(pb['delta_q']
            - DELTA_Q_REF) < 1e-9
    and pb['l24_anchor_ok'] is True,
    'L24 double anchor + delta_q 1e-9')
n_win = int(sum(
    1 for v in wf if v >= 0.5 * F10_REF))
win_exp = ('qwen_win_wide'
           if n_win >= 3
           else ('qwen_win_narrow'
                 if n_win <= 1
                 else 'qwen_win_partial'))
chk('B3', n_win == pb['n_win']
    and vd[1] == win_exp,
    'n_win=%d gate consistent' % n_win)

# ---- C part-c ----
pc = res['part_c']
gsp = np.asarray(zn['gen_same_p'])
g30 = np.asarray(
    z30['gen_same_full']).astype(bool)
chk('C1', gsp.shape == (672,)
    and np.array_equal(gsp.astype(bool),
                       g30)
    and pc['full_swap_chg'] == 1.0
    and pc['full_swap_first'] == 178
    and pc['full_swap_bitexact_3130']
    is True,
    'full swap P bitexact vs 3130')
sel_exp = np.sort(
    np.random.default_rng(3131).choice(
        672, size=64,
        replace=False)).astype(np.int64)
sel = np.asarray(zn['sel'])
sel_sha = hashlib.sha256(
    sel.tobytes()).hexdigest()[:8]
chk('C2', np.array_equal(sel, sel_exp)
    and sel_sha == pc['sel64_sha8'],
    'sel64 re-derive + sha8 %s'
    % sel_sha)
c40 = np.asarray(zn['chg40'])
f40 = np.asarray(zn['first40'])
chk('C3', len(c40) == 40 and len(f40) == 40
    and np.allclose(
        c40, pc['chg40'], atol=1e-12)
    and np.array_equal(
        f40, np.asarray(pc['first40']))
    and pc['peak40'] == int(np.argmax(c40))
    and abs(pc['dominance40']
            - float(c40.max())) < 1e-12
    and vd[3] == ('single40_dominant'
                  if pc['dominance40'] >= 0.5
                  else 'single40_'
                       'distributed'),
    'chg40/peak/dominance re-derive')
ha1 = np.asarray(zn['hist_a1'])
gsa1 = np.asarray(zn['gen_same_a1'])
chk('C4', int(ha1.sum()) == 672
    and int(ha1[0]) == 0
    and pc['first_a1'] == int(ha1[1])
    and abs(pc['chg_a1']
            - (1.0 - float(gsa1.mean())))
    < 1e-12
    and vd[4] == ('a1_gen_symmetric'
                  if pc['chg_a1'] >= 0.95
                  else 'a1_gen_asymmetric'),
    'a1 hist structure + chg gate')
dm = np.asarray(zn['dm'])
med_abs = np.asarray(zn['med_abs_dm'])
med_r = float(np.median(
    np.abs(dm.astype(np.float64)),
    axis=0).max()
    - med_abs.max())
chk('C5', dm.shape == (672, 41)
    and med_abs.shape == (41,)
    and abs(med_r) < 1e-4,
    'med_abs_dm re-derive (f32 tol)')
medf = np.asarray(zn['med_dm_first'])
medr = np.asarray(zn['med_dm_rest'])
pk = pc['pk_dm']
chk('C6', len(medf) == 41 and len(medr) == 41
    and pk == int(np.argmax(med_abs))
    and pc['n_first'] == 178
    and pc['pk_dm_first'] == (
        int(np.argmax(medf))
        if pc['n_first'] > 0 else -1)
    and vd[5] == ('fork_late' if pk >= 38
                  else ('fork_mid'
                        if pk >= 20
                        else 'fork_early')),
    'fork pk/gate/first-set consistent')

# ---- D ledger ----
led = json.load(io.open(
    LEDGER, encoding='utf-8'))
ms = led['measurements']
m31 = [m for m in ms
       if m.get('meas_id', '')
       == 'meas3131_omega_p129_layer'
       'window_single40_a1gen_'
       'forklayer']
chk('D1', len(m31) == 1
    and m31[0]['phase'] == 3131
    and m31[0]['verdict'] == V
    and pc['sel64_sha8']
    in m31[0]['claim'],
    'ledger entry unique + claim')
blob = json.dumps(
    {k: v for k, v in led.items()
     if k != 'ledger_sha256_8'},
    sort_keys=True, ensure_ascii=False)
# chain recompute: closeout hashed the
# dict carrying the PREVIOUS round's
# sha field value (3130 round wrote
# 3b0e4a9b, n=267). Restore it, then
# re-hash and compare with current.
PREV_SHA = '3b0e4a9b'
cur = led['ledger_sha256_8']
led2 = json.loads(json.dumps(
    led))
n_before = len(led2['measurements'])
led2['ledger_sha256_8'] = PREV_SHA
blob2 = json.dumps(
    led2, sort_keys=True,
    ensure_ascii=False)
sha_led = hashlib.sha256(
    blob2.encode('utf-8')).hexdigest()[:8]
chk('D2', sha_led == cur
    and n_before == 268,
    'ledger sha chain recompute '
    '%s->%s' % (PREV_SHA, sha_led))
chk('D3', len(ms) == 268,
    'ledger n=%d' % len(ms))

# ---- E memo/wlog/memory ----
memo = io.open(MEMO,
               encoding='utf-8').read()
i31 = memo.find('## Phase 3131:')
tit = (memo[i31:memo.find('\n', i31)]
       if i31 >= 0 else '')
chk('E1', i31 >= 0 and len(tit) < 110
    and memo.count('## Phase 3131:') == 1
    and all(s in memo[i31:] for s in
            ('### 1.', '### 2.',
             '### 3.', '### 4.',
             '### 5.')),
    'memo 3131 title<110 + 5 sections')
wstat = []
import os
for wd in WLOGS:
    hit = False
    for fn in os.listdir(wd):
        if fn.endswith('.md') \
                and fn[:4] == '2026':
            p = wd + '\\' + fn
            try:
                t = io.open(
                    p,
                    encoding='utf-8').read()
            except Exception:
                continue
            if 'Phase 3131 Omega-P129 ' \
                    'closeout' in t:
                hit = True
                break
    wstat.append(hit)
chk('E2', all(wstat),
    'both wlogs contain closeout line')
mem = io.open(MEMW,
              encoding='utf-8').read()
chk('E3', '3131\uff08T4\uff09' in mem
    and 'max=3131' in mem
    and '3132' in mem and len(mem) < 3000,
    'memory 3131 line + max + <3000 '
    '(%d chars)' % len(mem))

n_fail = ok.count(False)
out = ('VERIFY %s (%d/%d PASS)\n\n'
       % ('GREEN' if n_fail == 0
          else 'RED', len(ok) - n_fail,
          len(ok))) + '\n'.join(P)
io.open(OUTP, 'w',
        encoding='utf-8').write(out)
print('done %d/%d'
      % (len(ok) - n_fail, len(ok)))
