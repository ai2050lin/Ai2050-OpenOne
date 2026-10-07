# -*- coding: utf-8 -*-
"""Phase 3078 independent verify: recompute all
statistics from frozen npz files, replay verdict,
check anchors reachable without forwards, verify
Ledger/MEMO/wlog on-disk state."""
import hashlib
import io
import json
import os

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
R = RDIR + (r'\phase3078'
            r'\omega_p75_routing_timing')
P76 = RDIR + (r'\phase3076'
              r'\omega_p73_cross_prompt_family')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
WLOG = ROOT + (r'\.workbuddy\memory'
               r'\2026-09-21.md')
chk = []


def ck(name, cond):
    chk.append((name, bool(cond)))


res = json.load(io.open(
    R + r'\result.json', encoding='utf-8'))
seal = json.load(io.open(
    R + r'\seal.json', encoding='utf-8'))
z8 = np.load(R + (r'\omega_p75_routing_timing'
                  r'.npz'))
z76 = np.load(P76 + (r'\omega_p73_cross_prompt_'
                     r'family.npz'))

FKEYS = ('A', 'B', 'C')
LYRS = [int(v) for v in z8['LYRS']]
NLYR = len(LYRS)

# ---- E1 anchor replay (no forwards needed) --
for fk in FKEYS:
    d34 = z8['DAH34_RE_' + fk]
    d35 = z8['DAH35_RE_' + fk]
    ck('a1_34_' + fk,
       float(np.max(np.abs(
           d34 - z76['DAH34_' + fk]))) == 0.0)
    ck('a1_35_' + fk,
       float(np.max(np.abs(
           d35 - z76['DAH35_' + fk]))) == 0.0)
    ck('a1m_34_' + fk,
       float(np.max(np.abs(
           np.median(d34, axis=0)
           - z76['DAH34_MED_' + fk]))) == 0.0)
    ck('a1m_35_' + fk,
       float(np.max(np.abs(
           np.median(d35, axis=0)
           - z76['DAH35_MED_' + fk]))) == 0.0)
r34_71 = np.array(json.load(io.open(
    ROOT + (r'\tests\glm5\result'
            r'\rdc_query_construction_20260913'
            r'\phase3071'
            r'\omega_p68_attn_head_decomp'
            r'\result.json'),
    encoding='utf-8'))['stats']['head']['r34'],
    dtype=np.float64)
ck('a2_R1A_vs_3071',
   float(np.max(np.abs(
       z76['R1_ALL32_A'] - r34_71))) == 0.0)
for fk in FKEYS:
    ck('b0_' + fk,
       float(z8['B0_DIFF_' + fk]) == 0.0)
    ck('a0_ok_' + fk, bool(z8['A0_OK_' + fk]))

# ---- statistics replay ----


def spearman(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    ra = np.argsort(np.argsort(a)) \
        .astype(np.float64)
    rb = np.argsort(np.argsort(b)) \
        .astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum()
                  * (rb * rb).sum())
    if den == 0:
        return 0.0
    return float((ra * rb).sum() / den)


def icc_head(M):
    M = np.asarray(M, np.float64)
    gm = M.mean()
    alpha = M.mean(axis=0) - gm
    beta = M.mean(axis=1) - gm
    eps = M - gm - alpha[None, :] \
        - beta[:, None]
    nf, nh = M.shape
    v_head = float((alpha ** 2).sum()
                   / (nh - 1))
    v_eps = float((eps ** 2).sum()
                  / ((nf - 1) * (nh - 1)))
    return v_head / max(v_head + v_eps, 1e-30)


R1 = {f: z76['R1_ALL32_' + f]
      .astype(np.float64) for f in FKEYS}
DM = {f: {li: z8['DM_L%d_%s' % (li, f)]
          .astype(np.float64)
          for li in LYRS} for f in FKEYS}
OM = {f: {li: z8['OM_L%d_%s' % (li, f)]
          .astype(np.float64)
          for li in LYRS} for f in FKEYS}
si = np.zeros((NLYR, 3))
sa = np.zeros((NLYR, 3))
sc = np.zeros((NLYR, 3))
ic = np.zeros(NLYR)
for i, li in enumerate(LYRS):
    for fi, fk in enumerate(FKEYS):
        si[i, fi] = spearman(
            np.abs(DM[fk][li]), R1[fk])
        sa[i, fi] = spearman(
            OM[fk][li], R1[fk])
    for pi, (fa, fb) in enumerate(
            (('A', 'B'), ('A', 'C'),
             ('B', 'C'))):
        sc[i, pi] = spearman(
            np.abs(DM[fa][li]),
            np.abs(DM[fb][li]))
    ic[i] = icc_head(np.stack(
        [np.abs(DM[fk][li])
         for fk in FKEYS]))
ck('sp_intra_bit', float(np.max(
    np.abs(si - np.array(res['stats']
                         ['sp_intra'])))) == 0.0)
ck('sp_amp_bit', float(np.max(
    np.abs(sa - np.array(res['stats']
                         ['sp_amp'])))) == 0.0)
ck('sp_cross_bit', float(np.max(
    np.abs(sc - np.array(res['stats']
                         ['sp_cross'])))) == 0.0)
ck('icc_bit', float(np.max(
    np.abs(ic - np.array(res['stats']
                         ['icc'])))) == 0.0)
sd34b = [spearman(np.abs(DM[fk][34]),
                  np.abs(z76['DAH34_MED_' + fk]))
         for fk in FKEYS]
ck('sp_d34b_bit', float(np.max(np.abs(
    np.array(sd34b)
    - np.array(res['stats']['sp_d34b']))))
    == 0.0)

# ---- verdict replay ----
gmax = float(np.max(np.abs(si)))
ck('gmax_value', abs(gmax
   - 0.22067448680351906) < 1e-15)
ck('gmax_layer_L33C',
   tuple(divmod(int(np.argmax(np.abs(si))), 3))
   == (6, 2))
S = [li for i, li in enumerate(LYRS)
     if np.min(np.abs(si[i])) >= 0.35]
ck('S_empty', S == [])
ck('verdict', res['verdict']
   == 'routing_signal_absent')
ck('verdict_npz', str(z8['VERDICT'])
   == 'routing_signal_absent')
ck('forwards', res['forwards'] == 99)
ppmin = min(min(r) for r
            in res['stats']['pp_intra'])
ck('pp_min', abs(ppmin - 0.22625) < 1e-12
   and ppmin > 0.05)
ck('amp_null', float(np.max(np.abs(sa)))
   < 0.35)
ck('cross_high_L34', sc[7].min() > 0.8)

# ---- gates + seal ----
g = res['gates']
ck('gates', g['S_layers'] == []
   and g['early_layers'] == []
   and abs(g['global_max_abs_sp']
           - 0.22067448680351906) < 1e-15)
for k in ('npz_sha256_8', 'result_sha256_8',
          'exec_sha256_8', 'script_sha256_8'):
    if k == 'npz_sha256_8':
        p = R + (r'\omega_p75_routing_timing'
                 r'.npz')
    elif k == 'result_sha256_8':
        p = R + r'\result.json'
    elif k == 'exec_sha256_8':
        p = R + r'\execution.json'
    else:
        p = ROOT + (r'\tests\glm5'
                    r'\phase3078_omega_p75_'
                    r'routing_timing.py')
    h = hashlib.sha256(io.open(
        p, 'rb').read()).hexdigest()[:8]
    ck(k, h == seal[k])

# ---- ledger / memo / wlog ----
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
m3078 = [m for m in led['measurements']
         if isinstance(m, dict)
         and m.get('phase') == 3078]
ck('ledger_n', len(led['measurements']) == 217)
ck('ledger_meas', len(m3078) == 1
   and m3078[0]['verdict']
   == 'routing_signal_absent')
l14 = [l for l in led['linkage']
       if isinstance(l, dict)
       and l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
ck('ledger_l14', len(l14['connects']) == 185
   and l14['connects'][-1]['phase'] == 3078)
led2 = json.loads(json.dumps(led))
led2.pop('ledger_sha256_8')
blob = json.dumps(led2, sort_keys=True,
                  ensure_ascii=False)
ck('ledger_sha', hashlib.sha256(
    blob.encode('utf-8')).hexdigest()[:8]
   == led['ledger_sha256_8'])
memo = io.open(MEMO, encoding='utf-8').read()
ck('memo_phase3078', '## Phase 3078:' in memo)
ck('memo_verdict', 'routing_signal_absent'
   in memo)
ck('wlog', 'Phase 3078' in io.open(
    WLOG, encoding='utf-8').read())

# ---- report ----
bad = [n for n, okc in chk if not okc]
io.open(R + r'\verify_log.txt', 'w',
        encoding='utf-8').write(
    '%d checks, %d failed\n'
    % (len(chk), len(bad))
    + ('\n'.join('FAIL ' + n for n in bad)
       if bad else 'ALL OK') + '\n')
print('VERIFY_%s (%d/%d)'
      % ('OK' if not bad else 'FAIL',
         len(chk) - len(bad), len(chk)))
