# -*- coding: utf-8 -*-
"""Phase 3151 independent disk verify."""
import hashlib
import json
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = os.path.join(ROOT, 'tests', 'glm5', 'result',
                    'rdc_query_construction_20260913',
                    'phase3151', 'g1p1_combo_additive_vs_interaction')
OK = [0]
BAD = [0]

def chk(name, cond):
    if cond:
        OK[0] += 1
        print('PASS %s' % name)
    else:
        BAD[0] += 1
        print('FAIL %s' % name)

def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

# artifacts
for f in ['result.json', 'result_v2fix.json', 'result_rev3151b.json',
          'execution.json', 'collect.npz', 'run_log.txt',
          'v2fix_log.txt']:
    chk('exists %s' % f, os.path.exists(os.path.join(OUTD, f)))
chk('smoke dir', os.path.isdir(os.path.join(OUTD, 'smoke')))

# main result
p = os.path.join(OUTD, 'result.json')
chk('main disk sha8', sha8(p) == 'dab41389')
r = json.load(open(p, encoding='utf-8'))
chk('main res_sha8', r['res_sha8'] == '3b4344c6')
chk('main seal', r['seal_sha8'] == 'a566dc96')
chk('panel 738', r['panel_rows'] == 738)
chk('panel >= 672 constraint', r['panel_rows'] >= 672)
chk('v0 ok', r['v0_material']['ok'] is True)
chk('gates present', len(r['v1']['gates']) == 4)
chk('v3 worst fruit', r['v3']['worst_class_b4'] == u'\u6c34\u679c')
chk('v5 k3 energy', abs(r['v5']['class_subspace_energy_k3'] -
                        0.8813) < 5e-4)

# v2fix
p = os.path.join(OUTD, 'result_v2fix.json')
chk('v2fix disk sha8', sha8(p) == 'd8cef721')
v = json.load(open(p, encoding='utf-8'))
chk('v2fix res', v['res_sha8'] == '2a26190f')
chk('v2fix seal', v['seal_sha8'] == '44f17187')
chk('M1vB4 pass', v['v2_k3']['M1_vs_B4']['pass_gate'] is True)
chk('M1vB5 pass', v['v2_k3']['M1_vs_B5']['pass_gate'] is True)
chk('M2vB4 fail', v['v2_k3']['M2_vs_B4']['pass_gate'] is False)

# rev3151b
p = os.path.join(OUTD, 'result_rev3151b.json')
rb = json.load(open(p, encoding='utf-8'))
chk('revb seal', rb['seal_sha8'] == '6409274e')
chk('revb verdict', rb['corrected_verdict'].startswith(
    'a_3150_ok|interaction_pair_generalizes_at_k3_only'))

# execution
e = json.load(open(os.path.join(OUTD, 'execution.json'),
                   encoding='utf-8'))
chk('exe design sha', e['design_sha'].startswith('e48acffd'))
chk('exe panel', e['design']['panel_rows'] == 738)

# collect npz
import numpy as np
z = np.load(os.path.join(OUTD, 'collect.npz'))
chk('collect shape', z['H'].shape == (3, 246, 41, 4096))

# ledger
led = json.load(open(os.path.join(ROOT, 'research', 'gpt5',
                                  'atlas', 'atlas_ledger.json'),
                     encoding='utf-8'))
chk('ledger n=288', len(led['measurements']) == 288)
m3151 = [m for m in led['measurements'] if m.get('phase') == 3151]
chk('ledger 3151 entry', len(m3151) == 1)
chk('ledger 3151 verdict', m3151[0]['verdict'].startswith(
    'a_3150_ok|interaction_pair_generalizes_at_k3_only'))
chk('ledger 3151 rows', m3151[0]['n_rows'] == 738)

# MEMO
memo = open(os.path.join(ROOT, 'research', 'gpt5', 'docs',
                         'AGI_GPT5_MEMO.md'),
            encoding='utf-8').read()
chk('MEMO 3151 section', u'## Phase 3151:' in memo)
chk('MEMO 3152 prereg', u'### \u9884\u6ce8\u518c Phase 3152' in memo)
chk('MEMO verdict str',
    u'interaction_pair_generalizes_at_k3_only' in memo)

# daily
daily = open(os.path.join(ROOT, '.workbuddy', 'memory',
                          '2026-10-01.md'),
             encoding='utf-8').read()
chk('daily 3151', u'Phase 3151 (gpt5 \u7ebf)' in daily)

# MEMORY
mem = open(os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md'),
           encoding='utf-8').read()
chk('MEMORY n=288', u'n=288 @3151' in mem)
chk('MEMORY 3151 line', u'3151\uff1aG1-P1' in mem)

# adjudication doc
chk('adjudication doc', os.path.exists(os.path.join(
    ROOT, 'research', 'gpt5', 'docs',
    'UNIFIED_REVIEW_ADJUDICATION_v1.md')))

print('VERIFY %d/%d PASS' % (OK[0], OK[0] + BAD[0]))
