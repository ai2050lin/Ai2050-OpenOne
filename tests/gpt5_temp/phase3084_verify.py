# -*- coding: utf-8 -*-
"""Phase 3084 independent verify: seal sha8,
npz bit-replay (med_c / r1 / top8 / n_neg /
capture8 / verdict tree), ledger, documents."""
import hashlib
import io
import json
import os

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913'
     r'\phase3084'
     r'\omega_p81_3b_layer_scan')
SCRIPT = (ROOT + r'\tests\glm5'
          r'\phase3084_omega_p81_3b_layer_'
          r'scan.py')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
AUDIT = ROOT + (r'\research\gpt5\docs'
                r'\hdmcc_knowledge_map_review_'
                r'20260921.md')
WLOG = ROOT + r'\.workbuddy\memory\2026-09-22.md'
MEMW = ROOT + r'\.workbuddy\memory\MEMORY.md'
REPF = (ROOT + r'\tests\gpt5_temp'
        r'\p3084_verify_report.txt')

ok = []
fail = []


def chk(cond, tag):
    if cond:
        ok.append(tag)
    else:
        fail.append(tag)


def sha8(path):
    with io.open(path, 'rb') as f:
        return hashlib.sha256(
            f.read()).hexdigest()[:8]


seal = json.load(io.open(
    R + r'\seal.json', encoding='utf-8'))
res = json.load(io.open(
    R + r'\result.json', encoding='utf-8'))
exe = json.load(io.open(
    R + r'\execution.json', encoding='utf-8'))
chk(seal['npz_sha256_8'] == sha8(
    R + r'\omega_p81_3b_layer_scan.npz'),
    'seal npz8')
chk(seal['result_sha256_8'] == sha8(
    R + r'\result.json'), 'seal result8')
chk(seal['exec_sha256_8'] == sha8(
    R + r'\execution.json'), 'seal exec8')
chk(seal['script_sha256_8'] == sha8(
    SCRIPT), 'seal script8')
chk(seal['verdict'] == res['verdict'],
    'seal verdict==result')
chk(res['phase'] == 3084
    and res['name']
    == 'omega_p81_3b_layer_scan',
    'res phase/name')
chk(exe['created'] == res['created']
    == seal['created'], 'created align')
chk(exe.get('prereg') == res.get('prereg'),
    'prereg frozen')

verdict = res['verdict']
chk(verdict in {'setup_failed',
                'layer_rescue',
                'layer_partial',
                'layer_absent'},
    'verdict allowed')

z = np.load(R + r'\omega_p81_3b_layer_scan.npz',
            allow_pickle=False)
chk(bool(z['SMOKE']) is False, 'not smoke')
chk(int(z['FORWARDS']) > 4000,
    'forwards >4k')
chk(bool(z['SETUP_OK']), 'SETUP_OK')
chk(bool(z['TIED']), 'TIED recorded')
chk(list(z['L_CAND']) == [28, 31, 33, 34],
    'L_CAND frozen')
chk(int(z['L_INJ_3083']) == 33,
    '3083 position recorded')

FKEYS = ('A', 'B', 'C')
LC = (28, 31, 33, 34)
for fk in FKEYS:
    for bk in ('B0', 'B1', 'B6',
               'B7A'):
        chk(bool(z[bk + '_OK_' + fk]),
            '%s_OK_%s' % (bk, fk))
        chk(float(z[bk + '_DIFF_' + fk])
            == 0.0,
            '%s_DIFF0_%s' % (bk, fk))
    chk(bool(z['B3_OK_' + fk]),
        'B3_OK_%s' % fk)

NN = {}
rescue = []
for L in LC:
    for fk in FKEYS:
        pfx = 'L%d_%s' % (L, fk)
        chk(bool(z['B4_OK_' + pfx]),
            'B4_OK_%s' % pfx)
        chk(float(z['B4_DIFF_' + pfx])
            == 0.0, 'B4_DIFF0_%s' % pfx)
        chk(bool(z['B8_OK_' + pfx]),
            'B8_OK_%s' % pfx)
        chk(float(z['B8_DIFF_' + pfx])
            == 0.0, 'B8_DIFF0_%s' % pfx)
        # med_c replay from COS_LAD
        mc = float(np.median(
            z['COS_LAD_' + pfx]))
        chk(mc == float(z['MED_C_'
                        + pfx]),
            'med_c bit %s' % pfx)
        # r1 replay from CS1H - med_c
        r1 = np.median(z['CS1H_' + pfx],
                       axis=1) - mc
        chk(np.allclose(
            r1, z['R1_NH_' + pfx],
            rtol=0, atol=0),
            'r1 bit %s' % pfx)
        n_neg = int((r1 < 0).sum())
        chk(n_neg == int(z['N_NEG_'
                         + pfx]),
            'n_neg bit %s' % pfx)
        order = np.argsort(r1)
        topk = min(8, n_neg)
        top8 = [int(h)
                for h in order[:topk]]
        chk(top8 == list(z['TOP8_'
                         + pfx]),
            'top8 bit %s' % pfx)
        sel_ok = bool(topk == 8
                      and bool(
                          (r1[top8] < 0)
                          .all()))
        chk(sel_ok == bool(
            z['TOP8_SEL_OK_' + pfx]),
            'sel_ok bit %s' % pfx)
        neg = r1[r1 < 0]
        if len(neg) > 0:
            cap = abs(float(
                r1[top8].sum())) \
                / abs(float(neg.sum()))
            mn = float(np.median(neg))
        else:
            cap = float('nan')
            mn = float('nan')
        chk(cap == float(z['CAPTURE8_'
                          + pfx]),
            'capture8 bit %s' % pfx)
        chk(mn == float(z['MED_NEG_R1_'
                         + pfx]),
            'med_neg bit %s' % pfx)
        rall = float(z['R_ALL_' + pfx])
        chk(np.isfinite(rall)
            and -1.0 < rall < 1.0,
            'R_ALL sane %s' % pfx)
        NN[(L, fk)] = n_neg

# verdict tree replay
if verdict != 'setup_failed':
    rescue = [L for L in LC
              if all(NN[(L, f)] >= 8
                     for f in FKEYS)]
    if rescue:
        vr = 'layer_rescue'
    elif any(NN[(L, 'C')] >= 8
             for L in LC):
        vr = 'layer_partial'
    else:
        vr = 'layer_absent'
    chk(vr == verdict, 'verdict replay')
    if rescue:
        best = int(max(rescue,
                       key=lambda L: min(
                           NN[(L, f)]
                           for f in FKEYS)))
        chk(res['stats']['rescue_best']
            == best,
            'rescue_best replay')
    chk(res['stats']['rescue_layers']
        == [int(L) for L in rescue],
        'rescue_layers replay')

# ==== ledger ====
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
meas = [m for m in led['measurements']
        if isinstance(m, dict)
        and m.get('phase') == 3084]
chk(len(meas) == 1, 'ledger meas3084')
chk(len(led['measurements']) == 223,
    'ledger n=223')
l14 = [l for l in led['linkage']
       if isinstance(l, dict)
       and l.get('link_id')
       == 'L14_readout_spectrum_cross_model']
chk(len(l14) == 1, 'ledger L14 exists')
c3084 = [c for c in l14[0]['connects']
         if isinstance(c, dict)
         and c.get('phase') == 3084]
chk(len(c3084) == 1, 'L14 connects 3084')
if 'ledger_sha256_8' in led:
    saved = led.pop('ledger_sha256_8')
    blob = json.dumps(led, sort_keys=True,
                      ensure_ascii=False)
    chk(hashlib.sha256(blob.encode(
        'utf-8')).hexdigest()[:8] == saved,
        'ledger sha replay')
    led['ledger_sha256_8'] = saved

# ==== documents ====
memo = io.open(MEMO, encoding='utf-8').read()
chk('## Phase 3084:' in memo,
    'MEMO has Phase 3084')
if '## Phase 3084:' in memo:
    tail = memo[memo.rindex('## Phase 3084:'):]
    chk('layer' in tail, 'MEMO tail content')
else:
    chk(False, 'MEMO tail content')
aud = io.open(AUDIT,
              encoding='utf-8').read()
chk('## 四十六、3084' in aud,
    'audit 46 present')
wl = io.open(WLOG, encoding='utf-8').read()
chk('Phase 3084' in wl, 'wlog 3084')
mem = io.open(MEMW, encoding='utf-8').read()
chk('max=3084' in mem, 'MEMORY max=3084')

rep = ['VERIFY_OK %d/%d'
       % (len(ok), len(ok) + len(fail))]
if fail:
    rep.append('FAILED: ' + '; '.join(fail))
io.open(REPF, 'w', encoding='utf-8').write(
    '\n'.join(rep) + '\n')
print('VERIFY_DONE %d/%d'
      % (len(ok), len(ok) + len(fail)))
