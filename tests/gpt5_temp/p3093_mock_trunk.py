# -*- coding: utf-8 -*-
"""Trunk-variant mock dry-run for
phase3093_closeout.py.

The original mock (p3093_closeout_mock.py)
only exercises the fifth_mixed_absent
narrative branch.  The authoritative smoke
preview shows spec_class=trunk, so the
real verdict will be fifth_trunk_migrates
or fifth_trunk_no_migrate -- branches
never mock-tested.  This script builds two
self-consistent mock trees (A1
layer_rescue best=L38 + A2 trunk verdict),
execs the closeout source twice per
variant (idempotency), and asserts the
memo narrative/tables render for each
branch.  No real research file is touched.

NOTE: A1/A2 mock verdict keys use L38
(mock rescue best) -- the real run is L37;
closeout renders LB dynamically from
rescue_best so both are covered."""
import io
import json
import os
import shutil
import sys

import numpy as np

SRC = (r'D:\AI2050\Ai2050-OpenOne'
       r'\tests\gpt5_temp'
       r'\phase3093_closeout.py')
ROOTM = (r'D:\AI2050\Ai2050-OpenOne'
         r'\tests\gpt5_temp')
rep = []


def fam(dom, n_neg, capture8):
    return {'domain': dom, 'med_c': 0.31,
            'med_tt_norm': 12.1,
            'r_all': -0.19, 'a_u8': 0.5,
            'n_neg': n_neg,
            'top8': [1, 2, 3, 4, 5, 6, 7, 8],
            'top8_sel_ok': True,
            'capture8': capture8,
            'b_anchors': {
                'b0_diff': 0.0, 'b0_ok': True,
                'b1_diff': 0.0, 'b1_ok': True,
                'b3_ok': True, 'b4_diff': 0.001,
                'b4_ok': True, 'b6_diff': 0.0,
                'b6_ok': True,
                'b7a_diff': 0.0,
                'b7a_ok': True}}


def build_tree(base, verdict, gates):
    if os.path.exists(base):
        shutil.rmtree(base)
    P3093 = os.path.join(
        base, 'tests', 'glm5', 'result',
        'rdc_query_construction_20260913',
        'phase3093')
    A1D = os.path.join(
        P3093, 'omega_p90_qwen14b_layer_scan')
    A2D = os.path.join(
        P3093, 'omega_p91_qwen14b_l38_full_'
        'arbitration')
    os.makedirs(A1D)
    os.makedirs(A2D)
    DOCS = os.path.join(
        base, 'research', 'gpt5', 'docs')
    ATLAS = os.path.join(
        base, 'research', 'gpt5', 'atlas')
    os.makedirs(DOCS)
    os.makedirs(ATLAS)
    WB = os.path.join(
        base, '.workbuddy', 'memory')
    os.makedirs(WB)

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
    with io.open(os.path.join(
            ATLAS, 'atlas_ledger.json'), 'w',
            encoding='utf-8') as f:
        json.dump(led, f, ensure_ascii=False)

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
        f.write('audit base\n')
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
    with io.open(os.path.join(
            WB, 'MEMORY.md'), 'w',
            encoding='utf-8') as f:
        f.write(mem_txt)

    # ---- A1 (rescue best=L38) ----
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
        FA[fk] = {
            'b0_diff': 0.0, 'b0_ok': True,
            'b1_diff': 0.0, 'b1_ok': True,
            'b3_ok': True, 'b4_diff': 0.001,
            'b4_ok': True, 'b6_diff': 0.0,
            'b6_ok': True, 'b7a_diff': 0.0,
            'b7a_ok': True}
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
    with io.open(A1D + r'\seal.json', 'w',
                 encoding='utf-8') as f:
        json.dump({'npz_sha256_8': 'a1npz08',
                   'result_sha256_8': 'a1res08',
                   'script_sha256_8': 'a1scr08'},
                  f)
    np.savez(A1D + (r'\omega_p90_qwen14b_'
                    r'layer_scan.npz'),
             VERDICT=np.array('layer_rescue'),
             SMOKE=np.bool_(False),
             SETUP_OK=np.bool_(True),
             L_CAND=np.array([31, 34, 37, 38],
                             dtype=np.int64))

    # ---- A2 (parameterized trunk verdict) --
    e3 = {}
    for pr in ('T', 'U'):
        for pr2 in ('AB', 'AC', 'BC'):
            sig = (pr == 'T' and pr2 == 'AB'
                   and gates['G_DS'])
            e3['f2_cTT-%s-%s' % (pr, pr2)] = {
                'sp': 0.42 if sig else 0.11,
                'p': 0.003 if sig else 0.21}
    res2 = {
        'phase': 3093,
        'name': ('omega_p91_qwen14b_l38_full_'
                 'arbitration'),
        'forwards': 20922, 'elapsed': 13800.0,
        'verdict': verdict,
        'stats': {
            'repro': {
                fk: {'medc_diff': 0.0,
                     'rall_diff': 0.0}
                for fk in ('A', 'B', 'C')},
            'families': {
                'A': fam('everyday-causal',
                         20, 0.81),
                'B': fam('abstract-nouns',
                         18, 0.72),
                'C': fam('function-words',
                         16, 0.66)},
            'cross': {
                'e3': e3,
                'g_ds_count': gates['count'],
                'g_ds_n_bonf': gates['n_bonf'],
                'g_ds_min_sp': gates['min_sp'],
                'stouffer_z': gates['z']}},
        'gates': {
            'G_DS': gates['G_DS'],
            'count_sig_pos': gates['count'],
            'n_bonf': gates['n_bonf'],
            'min_sp': gates['min_sp'],
            'stouffer_z': gates['z'],
            'f2_TAB_sp': gates['f2_sp'],
            'f2_TAB_p': gates['f2_p'],
            'G1': gates['G1'], 'G1_sp': 0.4,
            'G1_p': 0.02, 'G1_resp': 'T',
            'G1_pair': 'AB',
            'f1_sign_positive': True,
            'spec_class': 'trunk',
            'spectrum': {}}}
    with io.open(A2D + r'\result.json', 'w',
                 encoding='utf-8') as f:
        json.dump(res2, f)
    with io.open(A2D + r'\seal.json', 'w',
                 encoding='utf-8') as f:
        json.dump({'npz_sha256_8': 'a2npz08',
                   'result_sha256_8': 'a2res08',
                   'script_sha256_8': 'a2scr08'},
                  f)
    np.savez(A2D + (r'\omega_p91_qwen14b_l38_'
                    r'full_arbitration.npz'),
             VERDICT=np.array(verdict),
             SMOKE=np.bool_(False))
    return A1D, DOCS, ATLAS, WB


def run_variant(tag, verdict, gates,
                expect_frag):
    base = os.path.join(
        ROOTM, 'p3093_mock_%s' % tag)
    A1D, DOCS, ATLAS, WB = build_tree(
        base, verdict, gates)
    src = io.open(SRC,
                  encoding='utf-8').read()
    old = ("ROOT = r'D:\\AI2050"
           "\\Ai2050-OpenOne'")
    assert old in src
    src2 = src.replace(
        old, "ROOT = r'%s'" % base)
    assert src2 != src
    for attempt in (1, 2):
        g = {'__name__': '__main__'}
        try:
            exec(compile(src2, SRC, 'exec'), g)
            rep.append('%s RUN%d CLOSEOUT_OK'
                       % (tag, attempt))
        except SystemExit as e:
            rep.append('%s RUN%d SystemExit(%s)'
                       % (tag, attempt, e.code))
            log = io.open(os.path.join(
                A1D, 'closeout_log.txt'),
                encoding='utf-8').read()
            rep.append('closeout_log: %s' % log)
            print('\n'.join(rep))
            sys.exit(1)
    m2 = io.open(os.path.join(
        DOCS, 'AGI_GPT5_MEMO.md'),
        encoding='utf-8').read()
    assert '## Phase 3093:' in m2
    assert verdict in m2, verdict
    assert '### 3. A2 全仲裁（L38）' in m2
    assert expect_frag in m2, expect_frag
    assert 'spec_class=trunk' in m2
    led2 = json.load(io.open(
        os.path.join(ATLAS,
                     'atlas_ledger.json'),
        encoding='utf-8'))
    assert len(led2['measurements']) == 231
    l14c = [l for l in led2['linkage']
            if l.get('link_id')
            == 'L14_readout_spectrum_'
            'cross_model'][0]['connects']
    assert len(l14c) == 199
    aud2 = io.open(os.path.join(
        DOCS, 'hdmcc_knowledge_map_review_'
        '20260921.md'),
        encoding='utf-8').read()
    assert '五十四' in aud2
    mem2 = io.open(os.path.join(
        WB, 'MEMORY.md'),
        encoding='utf-8').read()
    assert 'max=3093' in mem2
    assert 'max=3092' not in mem2
    assert len(mem2) < 3000
    i93 = m2.rindex('## Phase 3093:')
    rep.append('%s VERIFY_ALL_OK' % tag)
    return m2[i93:i93 + 1400]


# ---- variant 1: fifth_trunk_migrates ----
g_mig = {'G_DS': True, 'count': 5,
         'n_bonf': 2, 'min_sp': 0.31,
         'z': 2.81, 'f2_sp': 0.42,
         'f2_p': 0.003, 'G1': True}
m_mig = run_variant(
    'mig', 'fifth_trunk_migrates', g_mig,
    'G_DS 迁移信号首次出现在第五谱点')

# ---- variant 2: fifth_trunk_no_migrate --
g_nom = {'G_DS': False, 'count': 1,
         'n_bonf': 0, 'min_sp': -0.12,
         'z': 1.09, 'f2_sp': 0.05,
         'f2_p': 0.31, 'G1': False}
m_nom = run_variant(
    'nom', 'fifth_trunk_no_migrate', g_nom,
    '缺席不依赖谱位形态')

print('\n'.join(rep))
print('---- MEMO 3093 head (migrates) ----')
print(m_mig[:1400])
print('---- narrative (no_migrate) ----')
k = m_nom.find('A2 在 L38 给出 fifth_trunk_'
               'no_migrate')
print(m_nom[k:k + 420])
