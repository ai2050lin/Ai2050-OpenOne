# -*- coding: utf-8 -*-
"""Mock dry-run for phase3093_closeout.py.
Builds a self-consistent mock result tree
(A1 layer_rescue best=L38 + A2
fifth_mixed_absent), redirects every path
const to a temp root, execs the closeout
source twice (idempotency check), then prints
artifact tails.  No real research file is
touched."""
import io
import json
import os
import shutil
import sys

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne'
        r'\tests\gpt5_temp'
        r'\p3093_mock_closeout')
SRC = (r'D:\AI2050\Ai2050-OpenOne'
       r'\tests\gpt5_temp'
       r'\phase3093_closeout.py')
rep = []

if os.path.exists(BASE):
    shutil.rmtree(BASE)
P3093 = os.path.join(
    BASE, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913',
    'phase3093')
A1D = os.path.join(
    P3093, 'omega_p90_qwen14b_layer_scan')
A2D = os.path.join(
    P3093,
    'omega_p91_qwen14b_l38_full_arbitration')
os.makedirs(A1D)
os.makedirs(A2D)
DOCS = os.path.join(
    BASE, 'research', 'gpt5', 'docs')
ATLAS = os.path.join(
    BASE, 'research', 'gpt5', 'atlas')
os.makedirs(DOCS)
os.makedirs(ATLAS)
WB = os.path.join(
    BASE, '.workbuddy', 'memory')
os.makedirs(WB)
rep.append('mock tree built at %s' % BASE)

# ---- mock ledger (230 meas / 198 conns) ----
led = {
    'measurements': [{'phase': 0}
                     for _ in range(230)],
    'linkage': [
        {'link_id':
         'L14_readout_spectrum_cross_model',
         'connects': [{'old': i}
                      for i in range(198)]},
        {'link_id': 'other'}],
    'ledger_sha256_8': 'deadbeef'}
with io.open(os.path.join(ATLAS,
                          'atlas_ledger.json'),
             'w', encoding='utf-8') as f:
    json.dump(led, f, ensure_ascii=False)

# ---- mock memo / audit / wlog / MEMORY ----
memo = ('pre\n## Phase 3091: x [t]\n'
        '## Phase 3092: y [t]\ntail\n')
with io.open(os.path.join(
        DOCS, 'AGI_GPT5_MEMO.md'), 'w',
        encoding='utf-8') as f:
    f.write(memo)
with io.open(os.path.join(
        DOCS, 'hdmcc_knowledge_map_review_'
        '20260921.md'), 'w',
        encoding='utf-8') as f:
    f.write('audit base\n五十三 earlier\n')
with io.open(os.path.join(
        WB, '2026-09-22.md'), 'w',
        encoding='utf-8') as f:
    f.write('- Phase 3092 line\n')
mem_txt = (
    '# RDC\n'
    '- LPF v5.3 ...；repro 锚 bit 级通过）。\n'
    '- 泛化声明分层。\n'
    '- ...3092 G_DS 门敏感性 '
    'gate_sensitive_substantive'
    '（无符号 Stouffer 否决）**。\n'
    '## 下一步\n'
    '- max=3092，下一个 3093（A qwen3-14b '
    '第五谱点 ~21k 前向 GPU；B 4B 主干解剖；'
    'C R1 复用拓扑全景——reuse_inventory '
    '底册已备）。\n')
with io.open(os.path.join(WB, 'MEMORY.md'),
             'w', encoding='utf-8') as f:
    f.write(mem_txt)

# ---- mock A1 (rescue, best L38) ----
NN = {31: {'A': 20, 'B': 15, 'C': 12},
      34: {'A': 15, 'B': 9, 'C': 7},
      37: {'A': 6, 'B': 5, 'C': 4},
      38: {'A': 20, 'B': 18, 'C': 16}}
FL = {}
for fk in ('A', 'B', 'C'):
    FL[fk] = {}
    for L in (31, 34, 37, 38):
        FL[fk][str(L)] = {
            'med_c': 0.24 + 0.001 * L,
            'n_neg': NN[L][fk],
            'top8': [1, 2, 3, 4, 5, 6,
                     7, 8],
            'top8_sel_ok': True,
            'capture8': 0.79,
            'med_neg_r1': -0.017,
            'r_all': -0.26,
            'b4_ok': True, 'b8_ok': True}
FA = {}
for fk in ('A', 'B', 'C'):
    FA[fk] = {'b0_diff': 0.0, 'b0_ok': True,
              'b1_diff': 0.0, 'b1_ok': True,
              'b3_ok': True, 'b4_diff': 0.001,
              'b4_ok': True, 'b6_diff': 0.0,
              'b6_ok': True,
              'b7a_diff': 0.0, 'b7a_ok': True}
res1 = {
    'phase': 3093,
    'name': 'omega_p90_qwen14b_layer_scan',
    'forwards': 12210, 'elapsed': 7441.0,
    'verdict': 'layer_rescue',
    'stats': {
        'families_layers': FL,
        'fam_anchors': FA,
        'setup_ok_all': True,
        'rescue_layers': [31, 38],
        'rescue_best': 38}}
with io.open(A1D + r'\result.json', 'w',
             encoding='utf-8') as f:
    json.dump(res1, f)
seal1 = {'npz_sha256_8': 'a1npz08',
         'result_sha256_8': 'a1res08',
         'script_sha256_8': 'a1scr08'}
with io.open(A1D + r'\seal.json', 'w',
             encoding='utf-8') as f:
    json.dump(seal1, f)
np.savez(A1D + (r'\omega_p90_qwen14b_'
                r'layer_scan.npz'),
         VERDICT=np.array('layer_rescue'),
         SMOKE=np.bool_(False),
         SETUP_OK=np.bool_(True),
         L_CAND=np.array([31, 34, 37, 38],
                         dtype=np.int64))

# ---- mock A2 (fifth_mixed_absent) ----
def fam(dom):
    return {'domain': dom, 'med_c': 0.31,
            'med_tt_norm': 12.1,
            'r_all': -0.19, 'a_u8': 0.5,
            'n_neg': 20,
            'top8': [1, 2, 3, 4, 5, 6, 7, 8],
            'top8_sel_ok': True,
            'capture8': 0.81,
            'b_anchors': dict(FA['A'])}
e3 = {}
for pr in ('T', 'U'):
    for pr2 in ('AB', 'AC', 'BC'):
        e3['f2_cTT-%s-%s' % (pr, pr2)] = {
            'sp': 0.11, 'p': 0.21}
res2 = {
    'phase': 3093,
    'name': ('omega_p91_qwen14b_l38_full_'
             'arbitration'),
    'forwards': 20922, 'elapsed': 13800.0,
    'verdict': 'fifth_mixed_absent',
    'stats': {
        'repro': {'A': {'medc_diff': 0.0,
                        'rall_diff': 0.0},
                  'B': {'medc_diff': 0.0,
                        'rall_diff': 0.0},
                  'C': {'medc_diff': 0.0,
                        'rall_diff': 0.0}},
        'families': {
            'A': fam('everyday-causal'),
            'B': fam('abstract-nouns'),
            'C': fam('function-words')},
        'cross': {
            'e3': e3,
            'g_ds_count': 1, 'g_ds_n_bonf': 0,
            'g_ds_min_sp': -0.2443,
            'stouffer_z': 1.2}},
    'gates': {
        'G_DS': False, 'count_sig_pos': 1,
        'n_bonf': 0, 'min_sp': -0.2443,
        'stouffer_z': 1.2,
        'f2_TAB_sp': 0.11, 'f2_TAB_p': 0.21,
        'G1': False, 'G1_sp': 0.05,
        'G1_p': 0.4, 'G1_resp': 'T',
        'G1_pair': 'AB',
        'f1_sign_positive': False,
        'spec_class': 'mixed',
        'spectrum': {}}}
with io.open(A2D + r'\result.json', 'w',
             encoding='utf-8') as f:
    json.dump(res2, f)
seal2 = {'npz_sha256_8': 'a2npz08',
         'result_sha256_8': 'a2res08',
         'script_sha256_8': 'a2scr08'}
with io.open(A2D + r'\seal.json', 'w',
             encoding='utf-8') as f:
    json.dump(seal2, f)
np.savez(A2D + (r'\omega_p91_qwen14b_l38_'
                r'full_arbitration.npz'),
         VERDICT=np.array('fifth_mixed_absent'),
         SMOKE=np.bool_(False))

# ---- redirect paths & exec twice ----
src = io.open(SRC, encoding='utf-8').read()
old_root = "ROOT = r'D:\\AI2050\\Ai2050-OpenOne'"
assert old_root in src
src2 = src.replace(
    old_root, "ROOT = r'%s'" % BASE)
assert src2 != src
g = {'__name__': '__main__'}
try:
    exec(compile(src2, SRC, 'exec'), g)
    rep.append('RUN1 CLOSEOUT_OK')
except SystemExit as e:
    rep.append('RUN1 SystemExit(%s)' % e.code)
    rep.append('closeout_log: %s'
               % io.open(os.path.join(
                   A1D, 'closeout_log.txt'),
                   encoding='utf-8').read())
    print('\n'.join(rep))
    sys.exit(1)
g2 = {'__name__': '__main__'}
try:
    exec(compile(src2, SRC, 'exec'), g2)
    rep.append('RUN2 CLOSEOUT_OK (idempotent)')
except SystemExit as e:
    rep.append('RUN2 SystemExit(%s)' % e.code)

# ---- verify artifacts ----
m2 = io.open(os.path.join(
    DOCS, 'AGI_GPT5_MEMO.md'),
    encoding='utf-8').read()
assert '## Phase 3093:' in m2
assert 'fifth_mixed_absent' in m2
assert '### 3. A2 全仲裁（L38）' in m2
i93 = m2.rindex('## Phase 3093:')
assert i93 > m2.rindex('## Phase 3092:')
led2 = json.load(io.open(
    os.path.join(ATLAS, 'atlas_ledger.json'),
    encoding='utf-8'))
assert len(led2['measurements']) == 231
l14c = [l for l in led2['linkage']
        if l.get('link_id')
        == 'L14_readout_spectrum_cross_model'
        ][0]['connects']
assert len(l14c) == 199
aud2 = io.open(os.path.join(
    DOCS, 'hdmcc_knowledge_map_review_'
    '20260921.md'), encoding='utf-8').read()
assert '五十四' in aud2
wl2 = io.open(os.path.join(
    WB, '2026-09-22.md'),
    encoding='utf-8').read()
assert '3093' in wl2
mem2 = io.open(os.path.join(WB, 'MEMORY.md'),
               encoding='utf-8').read()
assert 'max=3093' in mem2
assert 'max=3092' not in mem2
assert len(mem2) < 3000
assert 'qwen3-14b（40L' in mem2
rep.append('VERIFY_ALL_OK (memo/ledger/audit/'
           'wlog/memory)')
print('\n'.join(rep))
print('---- MEMO 3093 head ----')
print(m2[i93:i93 + 1600])
