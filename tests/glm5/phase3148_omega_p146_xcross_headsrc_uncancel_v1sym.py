# -*- coding: utf-8 -*-
"""Phase 3148 (Omega-P146): tail sign
crossing fill (TAIL25 pos/neg mid-dose
d{2.5,3,3.5} -> locate the sign-crossing
dose d* between d2 (pos>neg, 3145/3147)
and d4 (pos<neg, 3146) + fstep/first
structure of pos_d1 vs neg_d4 flip
rows -> format-channel vs answer-side
switching) + top-2 super-source
write-side identity (order_ex[:2]=
{2530,3755} solo+pair dose d{0.5,1,2}
@L17 + L17 per-head o_proj-input
ablation over 32 heads -> per-head
contribution to the two coordinates +
dvec29/dvec19 top-50 overlap) +
unembed-cancel causality (neg d0.5
@L38 allstep + L39 projection removal
of w131401 vs random dir vs w_dn ->
is the format shift a sufficient cause
of the flip) + v1 perturbation
symmetry (amp alpha{-0.25,-0.5} vs
clip alpha{+0.25,+0.5} @L38 all ->
amplitude dependence vs direction
specificity).

Preregistered in 3147 closeout
(Omega-P146). 18 bit-anchor replays:
11 from 3147 (b_pc1/b_dvec29/b_joint,
d_tbot15_d4/d_ttop10_d4, d_co50ex
d{1,2,4}, t_pc1_inst, e_pcres38_d1
pos/neg) + v2_clip_a025/a050 (3146
windows, 2nd replay) + tail sign
curve bits (s_tailpos_d1.0/d_tail_d4.0
3rd cross-phase, s_tailpos_d4.0 +
d_tail_d1.0/d_tail_d2.0 2nd replay).
Bank shards REUSED from 3138. Frozen
dvecs from 3135 (sha-anchored).
Session baselines P+A1 with xphase
recording. Frozen before observation."""
import gc
import hashlib
import io
import json
import os
import pickle
import random as _rnd
import time
import zlib

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        + r'\rdc_query_construction_20260913')
NAME = ('omega_p146_xcross_headsrc_'
        'uncancel_v1sym')
SMOKE = os.environ.get('P3148_SMOKE',
                       '') == '1'

D05 = os.path.join(RDIR, 'phase3105',
                   'omega_p103_incontext_truth_'
                   'consistency')
D26 = RDIR + r'\phase3126' \
      r'\omega_p124_glm4_anchoredlast_' \
      r'regen_writechain'
D35 = RDIR + r'\phase3135' \
      r'\omega_p133_conduction_' \
      r'co36ablation_window'
D36 = RDIR + r'\phase3136' \
      r'\omega_p134_conddose_crossmatrix_w8drop'
D38 = RDIR + r'\phase3138' \
      r'\omega_p136_statebank_' \
      r'bicdecomp_probecontrast'
D47 = RDIR + r'\phase3147' \
      r'\omega_p145_tbsym_co50exk_' \
      r'negmech_v1micro'
MDIR_G = os.path.join(ROOT, 'models', 'hf',
                      'glm4-9b-chat-hf')
OUT = os.path.join(RDIR, 'phase3148', NAME)
if SMOKE:
    OUT = os.path.join(OUT, 'smoke')
os.makedirs(OUT, exist_ok=True)
LOGF = os.path.join(OUT, 'run_log.txt')
T0 = time.time()


def log(msg):
    line = '[%7.1fs] %s' % (time.time() - T0,
                            msg)
    with io.open(LOGF, 'a',
                 encoding='utf-8') as f:
        f.write(line + '\n')
    print(line, flush=True)


# ---------------- crash-resume ckpt -----
CKPTF = os.path.join(OUT, 'p146_ckpt.pkl')
CK = {'done': [], 'data': {}, 'meta': {}}
if os.path.exists(CKPTF):
    try:
        with io.open(CKPTF, 'rb') as _fh:
            CK = pickle.load(_fh)
        log('RESUME: ckpt stages=%s'
            % sorted(CK['done']))
    except Exception as _e:
        log('RESUME: ckpt load failed (%s), '
            'starting fresh' % _e)
        CK = {'done': [], 'data': {},
              'meta': {}}


def ck_save(stage, data):
    CK['data'][stage] = data
    if stage not in CK['done']:
        CK['done'].append(stage)
    _tmp = CKPTF + '.tmp'
    with io.open(_tmp, 'wb') as _fh:
        pickle.dump(CK, _fh, protocol=4)
    os.replace(_tmp, CKPTF)
    log('CKPT saved: %s' % stage)


# ---------------- frozen constants ------
NP = 672
N_T = 4
DIRS = ('P', 'A1')
ALL_L = list(range(40))
KEY_L = (17, 29, 33, 38)
SWAP_L = [8, 9, 13, 29]
D19_L = 19
D17_L = 17
NCAP = 4 if SMOKE else 128
N19_ROWS = 4 if SMOKE else NP
GEN_BATCH = 4 if SMOKE else 32
SPEC_ROWS = 4 if SMOKE else 16
DOSE_COND = 2.0
DOSE_C = 1.0
DEC_L = (17, 29, 38)
FIELD_SELF_COS = 0.99
Z35_TOL = 0.08
SHARE38_TOL = 0.02
WDN39_TOL = 0.15
ALIGN_TOL = 0.03
PNORM38_TOL = 1.0
SPEC_TOKS = [131401, 134772, 23386]
SIGN_TOL = 0.03
XDOSE_MID = (2.5, 3.0, 3.5)
H_DOSES = (0.5, 1.0, 2.0)
HEAD_COORDS = [2530, 3755]
NHEADS = 32
HEAD_ROWS = 4 if SMOKE else 32
V2_ALPHA = ((-0.25, 'v2_amp_a025'),
            (-0.5, 'v2_amp_a050'),
            (0.25, 'v2_clip_a025'),
            (0.5, 'v2_clip_a050'))
UCANCEL_TOL = 0.05
RNG_SEED = 3148

# ---------------- frozen 3147 anchors --
EXP_V47 = ('a_3146_ok|repro_bit_11|'
           'repro_bit_ok|dvec19_repro_6a0332|'
           'field_self_ok|z35_cos17_ok|'
           'resid_anchor_ok|tail_sign_cross|'
           'tb15_sym_pos|tail_sub_bot10|'
           'co50ex_threshold|co50ex_locus_top|'
           'enrich_inverse_absent|'
           'neg_late_format_shift|'
           'neg_repro_3146|'
           'v1_micro_continuous|'
           'v1_amp_stable|xphase_ok|'
           'coverage_full')
RES47_SHA = '63b9885d'
SEAL47 = '66a7b2d9'
XPHASE47 = 1.0
DVEC19_SHA = '6a0332a6'
MEDNORM19 = 5.643608093261719
# 11 bit anchors replayed from 3147
BIT11 = {'b_pc1_l29_d2.0': 0.203125,
         'b_dvec29_l29_d2.0': 0.640625,
         'b_joint_l29_d2.0': 0.4765625,
         'd_tbot15_d4.0': 0.359375,
         'd_ttop10_d4.0': 0.2421875,
         'd_co50ex_d1.0': 0.421875,
         'd_co50ex_d2.0': 0.8515625,
         'd_co50ex_d4.0': 0.9921875,
         't_pc1_l29_inst': 0.0625,
         'e_pcres38_d1.0_pos': 0.984375,
         'e_pcres38_d1.0_neg': 0.421875}
# +7 further bit wants (cross-phase)
WANT_V2 = {'v2_clip_a025': 0.296875,
           'v2_clip_a050': 0.6640625}
WANT_TAIL = {'s2_tailpos_d1': 0.1015625,
             's2_tailpos_d4': 0.234375,
             's2_tailneg_d1': 0.09375,
             's2_tailneg_d2': 0.171875,
             's2_tailneg_d4': 0.3046875}
N_BIT_TOT = 18
# 3146 dose dvals (subset used here)
DVALS46 = {'d_co50ex_d2.0': 0.8515625}
ORDER_EX2 = [2530, 3755]
# 3147 part_s exact values (link)
S47 = {'tail_pos_1': 0.1015625,
       'tail_pos_2': 0.2578125,
       'tail_pos_4': 0.234375,
       'tail_neg_1': 0.09375,
       'tail_neg_2': 0.171875,
       'tail_neg_4': 0.3046875,
       'tb15_pos_d1': 0.15625,
       'tb15_pos_d2': 0.359375,
       'tb15_neg_d1': 0.109375,
       'tb15_neg_d2': 0.09375,
       'chg_tb5_d4': 0.15625,
       'chg_tb10_d4': 0.2109375,
       'chg_tbot15_d4': 0.359375,
       'tailpos_d2_got': 0.2578125}
S47_TAGS = {'tail_sign': 'tail_sign_cross',
            'tb15_sym': 'tb15_sym_pos',
            'tail_sub': 'tail_sub_bot10',
            'tailpos_d2_repro': True}
# 3147 part_x exact values (link)
X47 = {'dose_05': 0.140625,
       'dose_1': 0.421875,
       'dose_2': 0.8515625,
       'dose_4': 0.9921875,
       'ktop2': 0.4296875,
       'ktop5': 0.5078125,
       'ktop10': 0.890625,
       'ktop15': 0.953125,
       'kbot2': 0.3671875,
       'kbot5': 0.2421875,
       'kbot10': 0.390625,
       'kbot15': 0.4375}
X47_TAGS = {'x_dose': 'co50ex_threshold',
            'x_locus': 'co50ex_locus_top',
            'enrich_inverse': False}
# 3147 part_n exact values (link)
N47 = {'chg_neg_d05': 0.2421875,
       'n_seqs': 44,
       'flip_rows': [0, 3, 4, 10, 14, 17,
                     20, 28],
       'fsteps': [4, 6, 4, 4, 4, 4, 6, 4],
       'dlog0': -0.4124155986937694,
       'dlog2': 1.4527831124141812}
N47_TAGS = {'n_repro_ok': True,
            'neg_tag':
                'neg_late_format_shift'}
# 3147 part_v exact values (link)
V47 = {'a005': 0.0703125,
       'a010': 0.078125,
       'a015': 0.1328125,
       'micro_min': 0.0703125,
       'amp_ratio': 0.8310960572684135}
V47_TAGS = {'v_micro': 'v1_micro_continuous',
            'v_traj': 'v1_amp_stable'}
# 3144 resid anchors (C4 recompute gates)
PSHARE_38 = 0.640474
PWDN_39 = 2.495525
PNORM_38 = 55.263357
RA_V1_44 = 0.228672
# 3143/3144 coordinate lists
DELTA_L17 = 0.637683315669971
HIGH25_43 = [2319, 83, 2309, 3140, 3099,
             2491, 2097, 999, 2702, 1939,
             1235, 2617, 161, 3426, 1877,
             551, 3301, 2011, 2560, 1008,
             2605, 78, 1357, 3818, 207]
LOW25_43 = [381, 1179, 1004, 1578, 2911,
            1626, 1973, 642, 1284, 2442,
            1306, 2890, 2664, 317, 1757,
            2185, 1991, 3131, 3756, 294,
            1634, 807, 1059, 3499, 3432]
G_TOP25 = [2319, 642, 1634, 2097, 1008,
           1357, 2664, 2617, 83, 294,
           3818, 1877, 3499, 2011, 3301,
           161, 317, 999, 3099, 1235,
           807, 1626, 2309, 381, 1939]
G_BOT25 = [3432, 3426, 2911, 3131, 3140,
           1004, 1578, 1284, 2185, 1973,
           2442, 1059, 551, 207, 2702,
           3756, 78, 2560, 1306, 1179,
           1757, 2890, 2491, 1991, 2605]
DVEC_SHA = {17: '5e4c3085',
            29: 'ee9484b2',
            33: '59fbe0d3',
            38: 'aced803b'}
LEDGER_N = 284

# ---------------- seal ------------------
SEAL = {
    'phase': 3148,
    'name': NAME,
    'created': time.strftime(
        '%Y-%m-%d %H:%M:%S'),
    'smoke': SMOKE,
    'constants': {
        'NP': NP, 'N_T': N_T,
        'DIRS': list(DIRS),
        'ALL_L': ALL_L,
        'KEY_L': list(KEY_L),
        'SWAP_L': SWAP_L,
        'D19_L': D19_L, 'D17_L': D17_L,
        'NCAP': NCAP,
        'N19_ROWS': N19_ROWS,
        'GEN_BATCH': GEN_BATCH,
        'SPEC_ROWS': SPEC_ROWS,
        'DOSE_COND': DOSE_COND,
        'DOSE_C': DOSE_C,
        'DEC_L': list(DEC_L),
        'SIGN_TOL': SIGN_TOL,
        'XDOSE_MID': list(XDOSE_MID),
        'H_DOSES': list(H_DOSES),
        'HEAD_COORDS': HEAD_COORDS,
        'NHEADS': NHEADS,
        'HEAD_ROWS': HEAD_ROWS,
        'V2_ALPHA': [list(v) for v
                     in V2_ALPHA],
        'UCANCEL_TOL': UCANCEL_TOL,
        'SPEC_TOKS': SPEC_TOKS,
        'RNG_SEED': RNG_SEED,
        'BANK_REUSE_3138': True},
    'anchors': {
        'res47_verdict': EXP_V47,
        'res47_sha8': RES47_SHA,
        'seal47': SEAL47,
        'xphase47': XPHASE47,
        'dvec19_sha8': DVEC19_SHA,
        'mednorm19': MEDNORM19,
        'bit11': BIT11,
        'want_v2': WANT_V2,
        'want_tail': WANT_TAIL,
        'n_bit_tot': N_BIT_TOT,
        'order_ex2': ORDER_EX2,
        's47': S47, 's47_tags': S47_TAGS,
        'x47': X47, 'x47_tags': X47_TAGS,
        'n47': N47, 'n47_tags': N47_TAGS,
        'v47': V47, 'v47_tags': V47_TAGS,
        'pshare_38': PSHARE_38,
        'pwdn_39': PWDN_39,
        'pnorm_38': PNORM_38,
        'ra_v1_44': RA_V1_44,
        'delta_l17': DELTA_L17,
        'high25_43': HIGH25_43,
        'low25_43': LOW25_43,
        'g_top25': G_TOP25,
        'g_bot25': G_BOT25,
        'dvec_sha8': {str(k): v for k, v
                      in DVEC_SHA.items()},
        'ledger_n': LEDGER_N},
    'prereg': ('3147 closeout Omega-P146: '
               '(1) tail sign-cross fill: '
               'TAIL25 pos/neg mid-dose '
               'd{2.5,3,3.5} + full curve '
               'd{1,2,2.5,3,3.5,4} both sgn '
               '-> locate crossing d* + '
               'fstep/first structure of '
               'pos_d1 vs neg_d4 flip rows '
               '(format vs answer-side '
               'switching); (2) top-2 source '
               'identity: order_ex[:2]='
               '{2530,3755} solo+pair dose '
               'd{0.5,1,2} @L17 sgn-1 + L17 '
               'per-head o_proj-input '
               'ablation x32 heads -> '
               'per-head contribution to '
               'the two coords + dvec29/'
               'dvec19 top-50 overlap; '
               '(3) unembed-cancel: neg d0.5 '
               '@L38 allstep + L39 removal '
               'of w131401 vs random vs '
               'w_dn -> format shift '
               'sufficient cause test; '
               '(4) v1 symmetry: amp '
               'alpha{-0.25,-0.5} vs clip '
               'alpha{+0.25,+0.5} @L38 all. '
               '18 bit replays. Bank reused '
               '3138. Frozen before '
               'observation.')}
SEALF = os.path.join(OUT, 'design_seal.json')
if os.path.exists(SEALF):
    prev = json.load(io.open(SEALF,
                             encoding='utf-8'))
    seal_rt = json.loads(json.dumps(SEAL))
    assert prev['constants'] == \
        seal_rt['constants'], 'seal drift'
    assert prev['anchors'] == \
        seal_rt['anchors'], 'seal drift'
    log('seal ok (existing, constants match)')
else:
    with io.open(SEALF, 'w',
                 encoding='utf-8') as f:
        json.dump(SEAL, f,
                  ensure_ascii=False,
                  indent=1)
    log('seal written (constants frozen)')

# ---------------- frozen inputs ---------
mat5 = json.load(io.open(
    os.path.join(D05, 'material.json'),
    encoding='utf-8'))
z26 = np.load(D26 + r'\p124_readout.npz',
              allow_pickle=False)
assert z26['mlg_s0_P'].shape == (672, 41, 13)
assert z26['gen_base_P'].shape == (672, 12)
assert z26['gen_base_A1'].shape == (672, 12)
z35 = np.load(D35 + r'\p133_readout.npz',
              allow_pickle=False)
dvec = {}
for l in KEY_L:
    a = z35['dvec%d' % l].astype(np.float32)
    got = hashlib.sha256(
        a.tobytes()).hexdigest()[:8]
    assert got == DVEC_SHA[l], (l, got)
    assert a.shape == (NP, 4096)
    dvec[l] = a
    log('frozen dvec L%02d sha8 %s'
        % (l, DVEC_SHA[l]))
G_ORDER = z35['co36_rank'].astype(np.int64)
assert len(G_ORDER) == 50
assert list(G_ORDER[:25]) == G_TOP25, \
    'co36_rank top25 drift vs 3137'
assert list(G_ORDER[25:]) == G_BOT25, \
    'co36_rank bot25 drift vs 3137'
log('co36_rank order exact == 3137 G '
    'top25/bot25 lists')
z136 = np.load(D36 + r'\p134_readout.npz',
               allow_pickle=False)
CO_SETS = {}
for cn in ('co36', 'co50', 'union'):
    _c = np.asarray(z136[cn]).astype(np.int64)
    assert _c.ndim == 1, (cn, _c.shape)
    assert _c.min() >= 0 and _c.max() < 4096
    CO_SETS[cn] = _c
assert len(CO_SETS['co36']) == 50
assert set(G_ORDER.tolist()) == \
    set(CO_SETS['co36'].tolist()), \
    'co36_rank set != co36 set'
led = json.load(io.open(
    ROOT + r'\research\gpt5\atlas'
    r'\atlas_ledger.json',
    encoding='utf-8'))
assert len(led['measurements']) == LEDGER_N
log('frozen inputs ok (26/35/36 + ledger '
    'n=%d)' % LEDGER_N)

# ================================================================
# PART A: 3147 link asserts
# ================================================================
log('== PART A: link asserts ==')
res47 = json.load(io.open(
    D47 + r'\result.json', encoding='utf-8'))
assert res47['smoke'] is False
V47g = res47['verdict']
assert V47g == EXP_V47, V47g
raw47 = io.open(
    D47 + r'\result.json', 'rb').read()
sha47 = hashlib.sha256(raw47).hexdigest()[:8]
assert sha47 == RES47_SHA, sha47
assert str(res47['seal_sha8']) == SEAL47
pa47 = res47['part_a']
assert abs(float(pa47['xphase_P'])
           - XPHASE47) < 1e-12
assert abs(float(pa47['xphase_A1'])
           - XPHASE47) < 1e-12
pb47 = res47['part_bits']['bit_anchors']
assert len(pb47) == 11
for tn, v in BIT11.items():
    assert abs(float(pb47[tn]['got'])
               - v) < 1e-9, tn
    assert abs(float(pb47[tn]['want'])
               - v) < 1e-9, tn
    assert pb47[tn]['match'] is True, tn
ps47 = res47['part_s']
assert ps47['tail_pos'] == {
    '1': S47['tail_pos_1'],
    '2': S47['tail_pos_2'],
    '4': S47['tail_pos_4']}
assert ps47['tail_neg'] == {
    '1': S47['tail_neg_1'],
    '2': S47['tail_neg_2'],
    '4': S47['tail_neg_4']}
for k in ('tb15_pos_d1', 'tb15_pos_d2',
          'tb15_neg_d1', 'tb15_neg_d2',
          'chg_tb5_d4', 'chg_tb10_d4',
          'chg_tbot15_d4',
          'tailpos_d2_got'):
    assert abs(float(ps47[k]) - S47[k]) \
        < 1e-9, k
for k, v in S47_TAGS.items():
    assert ps47[k] == v, k
px47 = res47['part_x']
assert [int(c) for c in
        px47['order_ex'][:2]] == ORDER_EX2
assert len(px47['order_ex']) == 50
assert abs(float(
    px47['dose_curve']['0.5'])
    - X47['dose_05']) < 1e-9
assert abs(float(px47['dose_curve']['1'])
           - X47['dose_1']) < 1e-9
assert abs(float(px47['dose_curve']['2'])
           - X47['dose_2']) < 1e-9
assert abs(float(px47['dose_curve']['4'])
           - X47['dose_4']) < 1e-9
for k, v in (('2', 'ktop2'), ('5', 'ktop5'),
             ('10', 'ktop10'),
             ('15', 'ktop15')):
    assert abs(float(px47['k_top'][k])
               - X47[v]) < 1e-9, v
for k, v in (('2', 'kbot2'), ('5', 'kbot5'),
             ('10', 'kbot10'),
             ('15', 'kbot15')):
    assert abs(float(px47['k_bot'][k])
               - X47[v]) < 1e-9, v
for k, v in X47_TAGS.items():
    assert px47[k] == v, k
pn47 = res47['part_n']
assert abs(float(pn47['chg_neg_d05'])
           - N47['chg_neg_d05']) < 1e-9
assert int(pn47['n_seqs']) == N47['n_seqs']
assert [int(x) for x in
        pn47['flip_rows']] == \
    N47['flip_rows']
assert [int(x) for x in
        pn47['fsteps']] == N47['fsteps']
assert abs(float(pn47['dlog_traj']['0'])
           - N47['dlog0']) < 1e-9
assert abs(float(pn47['dlog_traj']['2'])
           - N47['dlog2']) < 1e-9
for k, v in N47_TAGS.items():
    assert pn47[k] == v, k
pv47 = res47['part_v']
assert abs(float(pv47['micro']
                 ['v_base_a005'])
           - V47['a005']) < 1e-9
assert abs(float(pv47['micro']
                 ['v_base_a010'])
           - V47['a010']) < 1e-9
assert abs(float(pv47['micro']
                 ['v_base_a015'])
           - V47['a015']) < 1e-9
assert abs(float(pv47['micro_min'])
           - V47['micro_min']) < 1e-9
assert abs(float(pv47['amp_ratio'])
           - V47['amp_ratio']) < 1e-9
for k, v in V47_TAGS.items():
    assert pv47[k] == v, k
log('A hard asserts ok (3147 sha8 %s seal '
    '%s: xphase/bit11/part_s/part_x '
    'order_ex2=%s/part_n/part_v all '
    'verified)' % (sha47, SEAL47,
                   ORDER_EX2))


# p3148 patch1 applied

# ================================================================
# materials factory (3142 lineage)
# ================================================================
p2r = mat5['pair2rel']
frel = mat5['false_rels']
ents_all = mat5['entities']
PREDS_all = mat5['predicates']
pks = sorted(p2r.keys())
assert len(pks) == NP
ENTS_N = len(ents_all)


def build_prompt(mat, s, o, lrel, qrel):
    ents = mat['entities']
    PREDS = mat['predicates']
    D = [tuple(d) for d in
         mat['distractors']['%d_%d' % (s, o)]]
    k = mat['kline']['%d_%d' % (s, o)]
    lines = [(s, lrel, o)] + list(D)
    rng2 = _rnd.Random(zlib.crc32(
        ('%d_%d_ord5' % (s, o))
        .encode('ascii')))
    order = list(range(8))
    rng2.shuffle(order)
    lines = [lines[i] for i in order]
    ci = lines.index((s, lrel, o))
    lines[ci], lines[k] = lines[k], lines[ci]
    text = 'Facts:'
    for (ls, lr, lo) in lines:
        text += ' The %s %s the %s.' % (
            ents[ls], PREDS[lr], ents[lo])
    text += (' Query: The %s %s the %s. Is this '
             'query true? Answer:'
             % (ents[s], PREDS[qrel], ents[o]))
    return text


def make_materials(tok, gens):
    texts = {}
    PID_T = {}
    for dc in DIRS:
        texts[dc] = {}
        ids_l = []
        for pk in pks:
            (s, o) = (int(v)
                      for v in pk.split('_'))
            r = p2r[pk]
            ri1, ri2 = frel[pk]
            if dc == 'P':
                (qrel, lrel) = (r, r)
            else:
                (qrel, lrel) = (r, ri1)
            t_ = build_prompt(mat5, s, o,
                              lrel, qrel)
            texts[dc][pk] = t_
            ids_l.append(list(
                tok(t_,
                    add_special_tokens=False)
                ['input_ids']))
        PID_T[dc] = ids_l
    DOT = int(tok('.',
                  add_special_tokens=False)
              ['input_ids'][0])
    return {'texts': texts, 'PID_T': PID_T,
            'DOT': DOT}


log('factory core defined')

# ================================================================
# model load
# ================================================================
log('== load glm4-9b ==')
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, \
    AutoTokenizer  # noqa: E402

gens_g = {'P': z26['gen_base_P'],
          'A1': z26['gen_base_A1']}
tok_g = AutoTokenizer.from_pretrained(
    MDIR_G, trust_remote_code=True)
model_g = AutoModelForCausalLM.from_pretrained(
    MDIR_G, torch_dtype=torch.bfloat16,
    attn_implementation='eager',
    trust_remote_code=True).to('cuda').eval()
NLG = len(model_g.model.layers)
assert NLG == 40
HIDG = int(model_g.config.hidden_size)
assert HIDG == 4096
log('glm4 loaded NLG=%d HIDG=%d'
    % (NLG, HIDG))
WUG = model_g.lm_head.weight.detach()
VOCAB = int(WUG.shape[0])
YES_G = int(tok_g(' yes',
                  add_special_tokens=False)
            ['input_ids'][0])
NO_G = int(tok_g(' no',
                 add_special_tokens=False)
           ['input_ids'][0])
w_dn_g = (WUG[YES_G] - WUG[NO_G]) \
    .float().cpu().numpy()
DOT_G = int(tok_g('.',
                  add_special_tokens=False)
            ['input_ids'][0])
norm_g = model_g.model.norm
PADG = tok_g.pad_token_id
if PADG is None:
    PADG = model_g.config.pad_token_id
tok_g.padding_side = 'left'
matg = make_materials(tok_g, gens_g)
pids_P_all = matg['PID_T']['P']
pids_A1_all = matg['PID_T']['A1']
_genenc = tok_g(
    matg['texts']['P'][pks[0]])['input_ids']
PREFIX_IDS = [int(t) for t in
              _genenc[:len(_genenc)
                      - len(matg['PID_T']
                            ['P'][0])]]
log('gen-prefix ready (%d ids; 3142 '
    'BOS semantics)' % len(PREFIX_IDS))
log('materials ready (672x2)')


def pad12(ids_r):
    v = list(ids_r)[:N_NEW]
    return v + [DOT_G] * (N_NEW - len(v))


def _pad_batch(rows):
    chunk = [list(PREFIX_IDS) + list(p)
             for p in rows]
    maxlen = max(len(p) for p in chunk)
    ids_p = np.full((len(chunk), maxlen),
                    PADG, dtype=np.int64)
    mask_p = np.zeros((len(chunk), maxlen),
                      dtype=np.int64)
    for i, p in enumerate(chunk):
        ids_p[i, maxlen - len(p):] = p
        mask_p[i, maxlen - len(p):] = 1
    return ids_p, mask_p


N_NEW = 12


def gen_batch_g2(pids_list,
                 inj=None,
                 inj_vec=None,
                 clip=None):
    """Generate N_NEW tokens per row from
    prompt ids. inj_vec: (il, dvec_batch
    (n,HID) fp32, scale, mode). mode:
    'allstep' = every forward; int s =
    prompt forward + decode step s (3135
    semantics). inj: (il, coords, delta,
    sgn, mode) coordinate injection
    (3137 G semantics). clip: list of
    (il, direction np[HID], alpha, cmode)
    v1-projection hooks on layer outputs;
    cmode 'all' = every forward (3145
    semantics at alpha 1.0), 'decode' =
    skip prompt forward (step 0)."""
    ids_p, mask_p = _pad_batch(pids_list)
    hooks = []
    if clip:
        for (il, direction, alpha,
             cmode) in clip:
            lyr = model_g.model.layers[il]
            v_t = torch.as_tensor(
                np.asarray(direction,
                           dtype=np.float32),
                device='cuda')
            v_t = v_t / (v_t.norm() + 1e-12)
            _af = float(alpha)
            st = {'step': 0}

            def _clip(mod, inp, out,
                      _v=v_t, _a=_af,
                      _m=cmode, _st=st):
                o2 = out[0] \
                    if isinstance(out,
                                  tuple) \
                    else out
                if _m == 'decode' \
                        and _st['step'] == 0:
                    _st['step'] += 1
                    return None
                h = o2[:, -1, :].float()
                proj = (h @ _v) \
                    .unsqueeze(-1)
                o2[:, -1, :] = (
                    h - _a * proj
                    * _v.unsqueeze(0)) \
                    .to(o2.dtype)
                _st['step'] += 1
                return None

            hooks.append(
                lyr.register_forward_hook(
                    _clip))
    if inj is not None:
        inj_list = inj if isinstance(
            inj, (list, tuple)) \
            and inj and isinstance(
                inj[0], (list, tuple)) \
            else [inj]
        for (il, coords, delta, sgn,
             mode) in inj_list:
            lyr = model_g.model.layers[il]
            co_t = torch.as_tensor(
                np.asarray(coords,
                           dtype=np.int64),
                device='cuda')
            dv_t = torch.full(
                (len(coords),),
                float(delta) * float(sgn),
                device='cuda',
                dtype=torch.bfloat16)
            st = {'step': 0}

            def _inj(mod, inp, out,
                     _co=co_t, _dv=dv_t,
                     _m=mode, _st=st):
                o2 = out[0] \
                    if isinstance(out,
                                  tuple) \
                    else out
                if _m == 'allstep' \
                        or _st['step'] == 0 \
                        or _st['step'] == _m:
                    o2[:, -1, _co] += _dv
                _st['step'] += 1
                return None

            hooks.append(
                lyr.register_forward_hook(
                    _inj))
    if inj_vec:
        for (il, dvec_batch, scale,
             mode) in inj_vec:
            lyr = model_g.model.layers[il]
            dv_t = torch.as_tensor(
                np.asarray(dvec_batch,
                           dtype=np.float32),
                device='cuda') \
                .to(torch.bfloat16) \
                * float(scale)
            st = {'step': 0}

            def _injv(mod, inp, out,
                      _dv=dv_t, _m=mode,
                      _st=st):
                o2 = out[0] \
                    if isinstance(out,
                                  tuple) \
                    else out
                if (_m == 'allstep'
                        or _st['step'] == 0
                        or _st['step']
                        == _m):
                    o2[:, -1, :] += _dv
                _st['step'] += 1
                return None

            hooks.append(
                lyr.register_forward_hook(
                    _injv))
    t_ids = torch.tensor(ids_p,
                         device='cuda')
    t_mask = torch.tensor(mask_p,
                          device='cuda')
    with torch.inference_mode():
        outg = model_g.generate(
            input_ids=t_ids,
            attention_mask=t_mask,
            max_new_tokens=N_NEW,
            do_sample=False, num_beams=1,
            pad_token_id=PADG)
    for hk in hooks:
        hk.remove()
    newg = outg[:, ids_p.shape[1]:]
    res = []
    for row in newg:
        ids_r = [int(t) for t in row]
        if PADG in ids_r:
            ids_r = ids_r[
                :ids_r.index(PADG)]
        for e in model_g.config.eos_token_id \
                if isinstance(
                    model_g.config
                    .eos_token_id,
                    list) else [
            model_g.config.eos_token_id]:
            if e in ids_r:
                ids_r = ids_r[
                    :ids_r.index(e)]
                break
        res.append(ids_r)
    return res


def capture_states(rows_all, cap_layers,
                   swap_layers=None):
    """Last-position states at cap_layers.
    SINGLE-SAMPLE loop (batch=1, zero
    padding) replicating z26/3140
    semantics exactly. swap_layers:
    layer-skip hooks (3135 dvec
    semantics)."""
    OUTS = {l: np.zeros((len(rows_all),
                         HIDG),
                        dtype=np.float32)
            for l in cap_layers}
    for j in range(len(rows_all)):
        ids_r = rows_all[j]
        t_in = torch.tensor(
            [list(ids_r)], device='cuda')
        feats = {l: None
                 for l in cap_layers}
        hooks = []

        def _mk(_l):
            def hook(mod, inp, out):
                o2 = out[0] \
                    if isinstance(out,
                                  tuple) \
                    else out
                feats[_l] = \
                    o2[0, -1, :].detach()
            return hook

        for l in cap_layers:
            lyr = model_g.model.layers[l]
            hooks.append(
                lyr.register_forward_hook(
                    _mk(l)))
        if swap_layers:
            for l in swap_layers:
                lyr = model_g.model.layers[l]

                def _swap(mod, inp, out):
                    o2 = out[0] \
                        if isinstance(
                            out, tuple) \
                        else out
                    i2 = inp[0] \
                        if isinstance(
                            inp, tuple) \
                        else inp
                    o2.copy_(
                        i2.to(o2.dtype))

                hooks.append(
                    lyr.register_forward_hook(
                        _swap))
        with torch.inference_mode():
            model_g(t_in, use_cache=False)
            for hk in hooks:
                hk.remove()
            for l in cap_layers:
                OUTS[l][j] = \
                    feats[l].float() \
                    .cpu().numpy()
        del feats
    return OUTS


def capture_states_inject2(rows_all,
                           cap_layers,
                           inj_list):
    """Single-sample forwards with per-row
    dvec injections (3135
    capture_states_inject semantics,
    multi-injection extended). inj_list:
    list of (il, dv_rows (n,HID) fp32,
    scale). Injection is in-place on the
    layer output; capture hooks registered
    after injection hooks read the
    post-injection field."""
    OUTS = {l: np.zeros((len(rows_all),
                         HIDG),
                        dtype=np.float32)
            for l in cap_layers}
    specs = [(model_g.model.layers[il],
              dvs, sc)
             for (il, dvs, sc) in inj_list]
    for j in range(len(rows_all)):
        feats = {l: None
                 for l in cap_layers}
        hooks = []
        for (lyr, dvs, sc) in specs:
            dv_t = torch.as_tensor(
                np.asarray(dvs[j],
                           dtype=np.float32),
                device='cuda') \
                .to(torch.bfloat16) \
                * float(sc)

            def _inj(mod, inp, out,
                     _dv=dv_t):
                o2 = out[0] \
                    if isinstance(out,
                                  tuple) \
                    else out
                o2[0, -1, :] += _dv
                return None

            hooks.append(
                lyr.register_forward_hook(
                    _inj))

        def _mk(_l):
            def hook(mod, inp, out):
                o2 = out[0] \
                    if isinstance(out,
                                  tuple) \
                    else out
                feats[_l] = \
                    o2[0, -1, :].detach()
            return hook

        for l in cap_layers:
            hooks.append(
                model_g.model.layers[l]
                .register_forward_hook(
                    _mk(l)))
        try:
            with torch.inference_mode():
                model_g(
                    torch.tensor(
                        [list(rows_all[j])],
                        device='cuda'),
                    use_cache=False)
        finally:
            for hk in hooks:
                hk.remove()
        for l in cap_layers:
            OUTS[l][j] = feats[l].float() \
                .cpu().numpy()
        del feats
    return OUTS


def trial_metrics(gen_l, base12_l):
    same_l = np.zeros(len(gen_l),
                      dtype=bool)
    fstep = np.full(len(gen_l), -1,
                    dtype=np.int8)
    first_l = 0
    for j in range(len(gen_l)):
        g12 = pad12(gen_l[j])
        b12 = base12_l[j]
        same_l[j] = (g12 == b12)
        if g12 != b12:
            for t in range(N_NEW):
                if g12[t] != b12[t]:
                    fstep[j] = t
                    break
        first_l += int(g12[0] != b12[0])
    chg_l = 1.0 - float(same_l.mean())
    return chg_l, first_l, fstep


def rankdata_avg(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind='stable')
    ranks = np.empty(len(x),
                     dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) \
                and sx[j + 1] == sx[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman(a, b):
    ra = rankdata_avg(a)
    rb = rankdata_avg(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = np.sqrt((ra ** 2).sum()
                  * (rb ** 2).sum())
    if den < 1e-12:
        return 0.0
    return float((ra * rb).sum() / den)


# ================================================================
# PART C0: session baselines P+A1 + xphase
# ================================================================
log('== PART C0: baselines P+A1 ==')
CKM = {'smoke': SMOKE, 'ncap': NCAP,
       'n19': N19_ROWS,
       'gen_prefix': True}
_KEEP_BOS_FREE = ('dvec19', 'field',
                  'cond19', 'cond17',
                  'nbase', 'ninj', 'nstepb',
                  'nstepi')
if CK['meta'] and CK['meta'] != CKM:
    _nd = {k: v for k, v in
           CK['data'].items()
           if k in _KEEP_BOS_FREE}
    _nold = len(CK['data'])
    CK = {'done': sorted(_nd.keys()),
          'data': _nd, 'meta': {}}
    log('CKPT meta mismatch -> kept %d '
        'BOS-independent capture stages, '
        'dropped %d gen stages'
        % (len(_nd), _nold - len(_nd)))
CK['meta'] = CKM
rows_scan = [pids_P_all[j]
             for j in range(NCAP)]
_EB = CK['data'].get('e_base_P')
if _EB is not None:
    gen_base_P = _EB['gens']
    xphase_P = _EB['xphase']
    log('base P RESUMED (xphase=%.4f)'
        % xphase_P)
else:
    gen_base_P = []
    for b0 in range(0, NCAP, GEN_BATCH):
        gen_base_P.extend(gen_batch_g2(
            rows_scan[b0:b0 + GEN_BATCH]))
    _xp = [pad12(gen_base_P[j]) ==
           pad12([int(v) for v in
                  z26['gen_base_P'][j]])
           for j in range(NCAP)]
    xphase_P = float(np.mean(_xp))
    log('xphase P: session-base vs z26 '
        'bit-match %.4f (%d/%d)'
        % (xphase_P, int(np.sum(_xp)),
           NCAP))
    ck_save('e_base_P', {
        'gens': gen_base_P,
        'xphase': xphase_P})
base12_P = [pad12(gen_base_P[j])
            for j in range(NCAP)]
rows_A1 = [pids_A1_all[j]
           for j in range(NCAP)]
_EB1 = CK['data'].get('e_base_A1')
if _EB1 is not None:
    gen_base_A1 = _EB1['gens']
    xphase_A1 = _EB1['xphase']
    log('base A1 RESUMED (xphase=%.4f)'
        % xphase_A1)
else:
    gen_base_A1 = []
    for b0 in range(0, NCAP, GEN_BATCH):
        gen_base_A1.extend(gen_batch_g2(
            rows_A1[b0:b0 + GEN_BATCH]))
    _xa = [pad12(gen_base_A1[j]) ==
           pad12([int(v) for v in
                  z26['gen_base_A1'][j]])
           for j in range(NCAP)]
    xphase_A1 = float(np.mean(_xa))
    log('xphase A1: session-base vs z26 '
        'bit-match %.4f (%d/%d)'
        % (xphase_A1, int(np.sum(_xa)),
           NCAP))
    ck_save('e_base_A1', {
        'gens': gen_base_A1,
        'xphase': xphase_A1})
base12_A1 = [pad12(gen_base_A1[j])
             for j in range(NCAP)]
xphase_ok = (xphase_P == 1.0) \
    and (xphase_A1 == 1.0)

# ================================================================
# PART C1: dvec19 re-capture (sha assert)
# ================================================================
log('== PART C1: dvec19 re-capture ==')
_K19 = CK['data'].get('dvec19')
if _K19 is not None \
        and _K19.get('sha8') == DVEC19_SHA:
    dvec19 = _K19['dvec']
    dvec19_sha = _K19['sha8']
    log('dvec19 RESUMED sha8=%s'
        % dvec19_sha)
else:
    rows_d19 = [pids_P_all[j]
                for j in range(N19_ROWS)]
    base19 = capture_states(
        rows_d19, [D19_L],
        swap_layers=None)[D19_L]
    log('dvec19 base capture done '
        '(%d rows)' % N19_ROWS)
    swap19 = capture_states(
        rows_d19, [D19_L],
        swap_layers=SWAP_L)[D19_L]
    dvec19 = (swap19 - base19) \
        .astype(np.float32)
    dvec19_sha = hashlib.sha256(
        dvec19.tobytes()).hexdigest()[:8]
    ck_save('dvec19', {
        'dvec': dvec19,
        'sha8': dvec19_sha})
dvec19_medn = float(np.median(
    np.linalg.norm(dvec19, axis=1)))
d19_sha_ok = (dvec19_sha == DVEC19_SHA)
d19_medn_drift = abs(
    dvec19_medn - MEDNORM19)
log('dvec19: sha8=%s (want %s, match %s) '
    'med||d||=%.4f (drift %.2e, n=%d)'
    % (dvec19_sha, DVEC19_SHA, d19_sha_ok,
       dvec19_medn, d19_medn_drift,
       dvec19.shape[0]))
assert d19_medn_drift < 0.05, \
    d19_medn_drift

# ================================================================
# PART C2: bank load + decomposition @DEC_L
# ================================================================
log('== PART C2: bank + decomposition ==')
H_ld = {}
H_sc = {}
for dc in DIRS:
    H_ld[dc] = {}
    H_sc[dc] = {}
    for t in range(N_T):
        fp = os.path.join(
            D38, 'bank_%s_T%d.npz'
            % (dc, t))
        assert os.path.exists(fp), fp
        zt = np.load(fp,
                     allow_pickle=False)
        _a = zt['states'].astype(np.float32)
        assert np.isfinite(_a).all(), \
            'bank shard non-finite'
        assert _a.shape == (NP, 40, HIDG)
        H_ld[dc][t] = _a
        H_sc[dc][t] = zt['scale'] \
            .astype(np.float32)
log('bank loaded (8 shards fp16, reused)')
IDEINT_P = {}
BANK_B = {}
PC = {}
for l in DEC_L:
    X = np.zeros((2, N_T, NP, HIDG),
                 dtype=np.float32)
    for di, dc in enumerate(DIRS):
        for t in range(N_T):
            X[di, t] = \
                H_ld[dc][t][:, l, :] \
                * H_sc[dc][t][l]
    B = X.mean(axis=(0, 1, 2))
    Ideint = X.mean(axis=1) - B
    C = X.mean(axis=(0, 2)) - B
    IDEINT_P[l] = Ideint[0].copy()
    BANK_B[l] = B.copy()
    if l == 29:
        WR = X - B[None, None, None, :] \
            - Ideint[:, None, :, :] \
            - C[None, :, None, :]
        WRm = WR.reshape(-1, HIDG)
        mu = WRm.mean(axis=0)[None, :]
        WRc = WRm - mu
        _U, _S, _Vt = np.linalg.svd(
            WRc.astype(np.float64),
            full_matrices=False)
        _var = (_S[:4] ** 2
                / np.sum(_S ** 2)).tolist()
        PC[29] = {
            'V': _Vt[:4].astype(np.float32),
            'var': _var,
            'mnorm': float(np.median(
                np.linalg.norm(
                    WRc, axis=1)))}
        log('C2 L29 WR PC var %s mnorm '
            '%.3f' % (json.dumps(
                [round(v, 4)
                 for v in _var]),
                PC[29]['mnorm']))
        del WR, WRm, mu, WRc, _U, _S, _Vt
    del X, B, Ideint, C
    gc.collect()
log('C2 done (%d layers)' % len(DEC_L))

# ================================================================
# PART C3: full-field rebuild (128 rows)
# ================================================================
log('== PART C3: field rebuild ==')
rows_cap = [pids_P_all[j]
            for j in range(NCAP)]
_KF = CK['data'].get('field')
if _KF is not None:
    base_states = {int(k):
                   v.astype(np.float32)
                   for k, v in
                   _KF['base'].items()}
    swap_states = {int(k):
                   v.astype(np.float32)
                   for k, v in
                   _KF['swap'].items()}
    log('field RESUMED')
else:
    base_states = capture_states(
        rows_cap, ALL_L, swap_layers=None)
    log('field base capture done (%d rows '
        'x %d layers)' % (NCAP, len(ALL_L)))
    swap_states = capture_states(
        rows_cap, ALL_L,
        swap_layers=SWAP_L)
    log('field swap4 capture done')
    ck_save('field', {
        'base': {str(l): base_states[l]
                 .astype(np.float16)
                 for l in ALL_L},
        'swap': {str(l): swap_states[l]
                 .astype(np.float16)
                 for l in ALL_L}})
field = {}
for l in ALL_L:
    field[l] = (swap_states[l]
                - base_states[l]) \
        .astype(np.float32)


def _rowcos(A, Bm):
    num = np.sum(A.astype(np.float64)
                 * Bm.astype(np.float64),
                 axis=1)
    den = (np.linalg.norm(
        A.astype(np.float64), axis=1)
        * np.linalg.norm(
            Bm.astype(np.float64), axis=1))
    with np.errstate(invalid='ignore',
                     divide='ignore'):
        return np.where(den > 1e-12,
                        num / den, 0.0)


fs17 = float(np.median(
    _rowcos(field[17], dvec[17][:NCAP])))
fs19 = float(np.median(
    _rowcos(field[19], dvec19[:NCAP])))
field_self_ok = (fs17 >= FIELD_SELF_COS
                 and fs19 >= FIELD_SELF_COS)
log('field self-consistency: cos(field17,'
    'dvec17)=%.4f cos(field19,dvec19)='
    '%.4f (gate %.2f -> %s)'
    % (fs17, fs19, FIELD_SELF_COS,
       field_self_ok))

# ================================================================
# PART C4: cond19/cond17 captures -> perp38
# top-PC + resid anchors
# ================================================================
log('== PART C4: cond captures + resid ==')
_KA = CK['data'].get('cond19')
if _KA is not None:
    h19 = {int(k):
           v.astype(np.float32)
           for k, v in
           _KA['h'].items()}
    log('cond19 RESUMED')
else:
    h19 = capture_states_inject2(
        rows_cap, ALL_L,
        [(D19_L, dvec19[:NCAP], DOSE_COND)])
    log('cond19 injection capture done')
    ck_save('cond19', {
        'h': {str(l): h19[l]
              .astype(np.float16)
              for l in ALL_L}})
_KA7 = CK['data'].get('cond17')
if _KA7 is not None:
    h17 = {int(k):
           v.astype(np.float32)
           for k, v in
           _KA7['h'].items()}
    log('cond17 RESUMED')
else:
    h17 = capture_states_inject2(
        rows_cap, ALL_L,
        [(D17_L, dvec[17][:NCAP],
          DOSE_COND)])
    log('cond17 injection capture done')
    ck_save('cond17', {
        'h': {str(l): h17[l]
              .astype(np.float16)
              for l in ALL_L}})
cos19m = {}
for r in range(20, 40):
    dh = (h19[r].astype(np.float64)
          - base_states[r].astype(np.float64))
    dv = field[r].astype(np.float64)
    cos19m[r] = _rowcos(dh, dv)
z35_c17_med = np.array(
    [float(np.nanmedian(z35['cos_17'][r]))
     for r in range(20, 40)])
h17_cos = {}
for r in range(20, 40):
    dh = (h17[r].astype(np.float64)
          - base_states[r].astype(np.float64))
    dv = field[r].astype(np.float64)
    h17_cos[r] = _rowcos(dh, dv)
ours_c17_med = np.array(
    [float(np.nanmedian(h17_cos[r]))
     for r in range(20, 40)])
z35_diff = float(np.max(np.abs(
    ours_c17_med - z35_c17_med)))
z35_ok = z35_diff < Z35_TOL
log('z35 cos_17 soft anchor: max|diff| '
    'over L20-39 = %.4f (tol %.2f -> %s)'
    % (z35_diff, Z35_TOL, z35_ok))


def _perp_at(r):
    dh19r = (h19[r].astype(np.float64)
             - base_states[r]
             .astype(np.float64))
    dh17r = (h17[r].astype(np.float64)
             - base_states[r]
             .astype(np.float64))
    n17 = np.sum(dh17r * dh17r, axis=1,
                 keepdims=True)
    coef = np.where(
        n17 > 1e-12,
        np.sum(dh19r * dh17r, axis=1,
               keepdims=True)
        / np.maximum(n17, 1e-12), 0.0)
    perp = dh19r - coef * dh17r
    n19 = np.linalg.norm(dh19r, axis=1)
    npp = np.linalg.norm(perp, axis=1)
    share = np.where(
        n19 > 1e-12, npp / np.maximum(
            n19, 1e-12), 0.0)
    return perp, share, dh19r, dh17r


perp38, share38_arr, dh19r38, dh17r38 = \
    _perp_at(38)
perp39, share39_arr, dh19r39, dh17r39 = \
    _perp_at(39)
share38 = float(np.median(share38_arr))
wdn_p38 = float(np.median(
    perp38 @ w_dn_g.astype(np.float64)))
wdn_p39 = float(np.median(
    perp39 @ w_dn_g.astype(np.float64)))
dh19_wdn39 = float(np.median(
    dh19r39 @ w_dn_g.astype(np.float64)))
pnorm38 = float(np.median(
    np.linalg.norm(perp38, axis=1)))
pc = perp38 - perp38.mean(axis=0)
_U, _S, _Vt = np.linalg.svd(
    pc, full_matrices=False)
pc1_res = _Vt[0].astype(np.float64)
dv29_mean = dvec[29][:NCAP].astype(
    np.float64).mean(axis=0)
v1 = PC[29]['V'][0].astype(np.float64)
ra_v1 = float(abs(pc1_res @ v1)
              / max(np.linalg.norm(pc1_res),
                    1e-12))
ra_wdn = float(abs(pc1_res
                   @ w_dn_g.astype(np.float64))
               / max(np.linalg.norm(pc1_res),
                     1e-12))
ra_dv29 = float(abs(pc1_res @ dv29_mean)
                / max(np.linalg.norm(pc1_res)
                      * np.linalg.norm(
                          dv29_mean), 1e-12))
# sign convention: median projection of
# perp38 rows onto pc1_res is positive
proj_sign = float(np.median(
    perp38 @ pc1_res))
if proj_sign < 0:
    pc1_res = -pc1_res
    log('C4 pc1_res sign flipped '
        '(median proj %.3e -> +)'
        % proj_sign)
resid_anchor_ok = (
    abs(share38 - PSHARE_38)
    < SHARE38_TOL
    and abs(wdn_p39 - PWDN_39)
    < WDN39_TOL
    and abs(ra_v1 - RA_V1_44) < ALIGN_TOL
    and abs(pnorm38 - PNORM_38)
    < PNORM38_TOL)
log('C4 resid anchors: share38 %.4f '
    '(want %.4f) wdn_p39 %.4f (want %.4f) '
    'ra_v1 %.4f (want %.4f) pnorm38 %.3f '
    '(want %.3f) -> %s'
    % (share38, PSHARE_38, wdn_p39,
       PWDN_39, ra_v1, RA_V1_44, pnorm38,
       PNORM_38, resid_anchor_ok))
ra_all = {'pc1_res_vs_v1': ra_v1,
          'pc1_res_vs_wdn': ra_wdn,
          'pc1_res_vs_dv29mean': ra_dv29}

# ================================================================
# PART B: bit replays (3 gen trials)
# ================================================================
log('== PART B: bit replays ==')
mn29 = PC[29]['mnorm']
dv_pc1_pos = np.tile(
    (PC[29]['V'][0] * mn29 * 2.0)[None, :],
    (NCAP, 1)).astype(np.float32)
dv_pc1_neg = -dv_pc1_pos
dv_pc1 = dv_pc1_pos
dv_dv29 = (dvec[29][:NCAP] * 2.0) \
    .astype(np.float32)
E_res = {}
fstep_store = {}


def _run_vec_trial(tname, il, dv_batch,
                   scale, mode, rows,
                   base12, save_gens=False,
                   clip=None):
    _K = CK['data'].get(tname)
    if _K is not None:
        E_res[tname] = _K['res']
        fstep_store[tname] = _K.get(
            'fstep')
        log('%s RESUMED chg=%.4f'
            % (tname, E_res[tname]['chg']))
        return
    gen_l = []
    for b0 in range(0, len(rows),
                    GEN_BATCH):
        batch = rows[b0:b0 + GEN_BATCH]
        gen_l.extend(gen_batch_g2(
            batch,
            inj_vec=[(il,
                      dv_batch[b0:b0
                               + len(batch)],
                      scale, mode)]
            if dv_batch is not None
            else None,
            clip=clip))
    chg_l, first_l, fs = trial_metrics(
        gen_l, base12)
    E_res[tname] = {'chg': chg_l,
                    'first': int(first_l)}
    fstep_store[tname] = fs
    if save_gens:
        E_res[tname]['gens'] = gen_l
    log('%s: chg=%.4f first=%d'
        % (tname, chg_l, first_l))
    _rec = {'res': {'chg': chg_l,
                    'first': int(first_l)},
            'fstep': fs.astype(np.int8)
            .tolist()}
    if save_gens:
        _rec['res']['gens'] = gen_l
    ck_save(tname, _rec)


def _run_coord_trial(tname, coords, sgn,
                     dsc, rows, base12,
                     il=D17_L):
    _K = CK['data'].get(tname)
    if _K is not None:
        E_res[tname] = _K['res']
        fstep_store[tname] = _K.get(
            'fstep')
        log('%s RESUMED chg=%.4f'
            % (tname, E_res[tname]['chg']))
        return
    gen_l = []
    for b0 in range(0, len(rows),
                    GEN_BATCH):
        batch = rows[b0:b0 + GEN_BATCH]
        gen_l.extend(gen_batch_g2(
            batch,
            inj=[(il, coords,
                  dsc * DELTA_L17, sgn,
                  0)]))
    chg_l, first_l, fs = trial_metrics(
        gen_l, base12)
    E_res[tname] = {'chg': chg_l,
                    'first': int(first_l)}
    fstep_store[tname] = fs
    log('%s: chg=%.4f first=%d'
        % (tname, chg_l, first_l))
    ck_save(tname, {
        'res': E_res[tname],
        'fstep': fs.astype(np.int8)
        .tolist()})


# B1: replay 3 generation bit anchors
_run_vec_trial('b_pc1_l29_d2.0', 29,
               dv_pc1_pos, 1.0, 'allstep',
               rows_scan, base12_P,
               save_gens=True)
pc1_sign = None
if not SMOKE and abs(
        E_res['b_pc1_l29_d2.0']['chg']
        - BIT11['b_pc1_l29_d2.0']) >= 1e-9:
    _run_vec_trial('b_pc1neg_l29_d2.0',
                   29, dv_pc1_neg, 1.0,
                   'allstep', rows_scan,
                   base12_P)
    if abs(E_res['b_pc1neg_l29_d2.0']
           ['chg']
           - BIT11['b_pc1_l29_d2.0']) < 1e-9:
        pc1_sign = -1
    else:
        raise AssertionError(
            'pc1 sign unresolved')
else:
    if not SMOKE:
        pc1_sign = 1
if pc1_sign == -1:
    dv_pc1 = dv_pc1_neg
    log('B1 pc1 sign resolved: -1')
elif pc1_sign == 1:
    log('B1 pc1 sign resolved: +1')
_run_vec_trial('b_dvec29_l29_d2.0', 29,
               dv_dv29, 1.0, 'allstep',
               rows_scan, base12_P)
_KJ = CK['data'].get('b_joint_l29_d2.0')
if _KJ is None:
    gen_l = []
    for b0 in range(0, NCAP, GEN_BATCH):
        batch = rows_scan[b0:b0 + GEN_BATCH]
        gen_l.extend(gen_batch_g2(
            batch,
            inj_vec=[(29, dv_pc1[b0:b0
                                + len(batch)],
                      1.0, 'allstep'),
                     (29, dv_dv29[b0:b0
                                  + len(batch)],
                      1.0, 'allstep')]))
    chg_l, first_l, fs = trial_metrics(
        gen_l, base12_P)
    E_res['b_joint_l29_d2.0'] = {
        'chg': chg_l, 'first': int(first_l),
        'gens': gen_l}
    fstep_store['b_joint_l29_d2.0'] = fs
    log('b_joint_l29_d2.0: chg=%.4f '
        'first=%d' % (chg_l, first_l))
    ck_save('b_joint_l29_d2.0', {
        'res': E_res['b_joint_l29_d2.0']})
else:
    E_res['b_joint_l29_d2.0'] = _KJ['res']
    log('b_joint_l29_d2.0 RESUMED '
        'chg=%.4f'
        % E_res['b_joint_l29_d2.0']['chg'])
n_bit_all = 0
bit_anchors = {}
if not SMOKE:
    for tn in ('b_pc1_l29_d2.0',
               'b_dvec29_l29_d2.0',
               'b_joint_l29_d2.0'):
        got = E_res[tn]['chg']
        m = abs(got - BIT11[tn]) < 1e-9
        n_bit_all += int(m)
        bit_anchors[tn] = {
            'got': got, 'want': BIT11[tn],
            'match': bool(m)}
    log('B1 3142 repro: %d/3 bit-match'
        % n_bit_all)
else:
    log('B1 3142 repro SKIPPED (smoke)')

# ================================================================
# PART S: bot15 sign matrix + subdivision
# ================================================================
log('== PART S: tb sign matrix ==')
I17 = IDEINT_P[17]
fr = I17 ** 2 / np.maximum(
    np.sum(I17 ** 2, axis=1,
           keepdims=True), 1e-12)
co36 = CO_SETS['co36']
assert len(co36) == 50
score = np.mean(fr[:, co36], axis=0)
order_e = co36[np.argsort(-score)]
assert list(order_e[:25]) == HIGH25_43, \
    'order_e drift vs 3141 high25'
assert list(order_e[25:]) == LOW25_43, \
    'order_e drift vs 3141 low25'
log('order_e equals 3141 high25/low25 '
    '(pure-numpy exact)')
HEAD25 = list(order_e[:25])
TAIL25 = list(order_e[25:])
TTOP10 = list(order_e[25:35])
TBOT15 = list(order_e[35:])
TB5 = list(order_e[35:40])
TB10 = list(order_e[40:])
# 2 bit replays @d4 sgn-1 (3146, 2nd)
_run_coord_trial('d_tbot15_d4.0', TBOT15,
                 -1, 4.0, rows_A1,
                 base12_A1)
_run_coord_trial('d_ttop10_d4.0', TTOP10,
                 -1, 4.0, rows_A1,
                 base12_A1)
if not SMOKE:
    for tn in ('d_tbot15_d4.0',
               'd_ttop10_d4.0'):
        got = E_res[tn]['chg']
        m = abs(got - BIT11[tn]) < 1e-9
        n_bit_all += int(m)
        bit_anchors[tn] = {
            'got': got, 'want': BIT11[tn],
            'match': bool(m)}
    log('S replay: +2 bit anchors (total '
        '%d/%d)' % (n_bit_all,
                    N_BIT_TOT))
# 3148: tail sign curve moved to
# PART X2 (s2_* namespace, full
# d{1,2,2.5,3,3.5,4} both signs)


def _chg(tn):
    return E_res[tn]['chg']


# ================================================================
# PART X: co50ex superlinear anatomy
# ================================================================
log('== PART X: co50ex anatomy ==')
co50 = CO_SETS['co50']
co50ex = np.asarray(
    sorted(set(co50.tolist())
           - set(co36.tolist())),
    dtype=np.int64)
assert co50ex.ndim == 1
assert len(co50ex) == 50, len(co50ex)
assert len(set(co50ex.tolist())
           & set(co36.tolist())) == 0, \
    'co50ex must be disjoint from co36' \
    ' (3135)'
score_ex = np.mean(fr[:, co50ex], axis=0)
order_ex = co50ex[np.argsort(-score_ex)]
log('X order_ex (IDEINT@L17 enrichment '
    'desc): %s' % json.dumps(
        [int(c) for c in order_ex]))
# 3 bit replays + low-dose fill
_run_coord_trial('d_co50ex_d1.0', co50ex,
                 -1, 1.0, rows_A1, base12_A1)
_run_coord_trial('d_co50ex_d2.0', co50ex,
                 -1, 2.0, rows_A1, base12_A1)
_run_coord_trial('d_co50ex_d4.0', co50ex,
                 -1, 4.0, rows_A1, base12_A1)
if not SMOKE:
    for tn in ('d_co50ex_d1.0',
               'd_co50ex_d2.0',
               'd_co50ex_d4.0'):
        got = E_res[tn]['chg']
        m = abs(got - BIT11[tn]) < 1e-9
        n_bit_all += int(m)
        bit_anchors[tn] = {
            'got': got,
            'want': BIT11[tn],
            'match': bool(m)}
    log('X replay: +3 bit anchors '
        '(total %d/%d)'
        % (n_bit_all, N_BIT_TOT))
assert [int(c) for c in order_ex[:2]] == ORDER_EX2, list(order_ex[:2])
log('order_ex top2 == 3147 %s (enrichment desc)' % ORDER_EX2)

# ================================================================
# PART T: pc1 instant replay (1 bit)
# ================================================================
log('== PART T: instant replay ==')
_run_vec_trial('t_pc1_l29_inst', 29,
               dv_pc1, 1.0, 0,
               rows_scan, base12_P)
if not SMOKE:
    got = E_res['t_pc1_l29_inst']['chg']
    m = abs(got - BIT11['t_pc1_l29_inst']) \
        < 1e-9
    n_bit_all += int(m)
    bit_anchors['t_pc1_l29_inst'] = {
        'got': got,
        'want': BIT11['t_pc1_l29_inst'],
        'match': bool(m)}
    log('T replay: +1 bit anchor (total '
        '%d/11)' % n_bit_all)

# ================================================================
# PART N: pcres d1 replays + neg mechanism
# ================================================================
log('== PART N: pcres replay + neg ==')
dv100 = (pc1_res
         / max(np.linalg.norm(pc1_res),
               1e-12) * pnorm38 * 1.0)     .astype(np.float32)
dv100_tile = np.tile(dv100[None, :],
                     (NCAP, 1))
_run_vec_trial('e_pcres38_d1.0_pos', 38,
               dv100_tile, 1.0, 'allstep',
               rows_scan, base12_P)
_run_vec_trial('e_pcres38_d1.0_neg', 38,
               -dv100_tile, 1.0, 'allstep',
               rows_scan, base12_P)
if not SMOKE:
    for tn in ('e_pcres38_d1.0_pos',
               'e_pcres38_d1.0_neg'):
        got = E_res[tn]['chg']
        m = abs(got - BIT11[tn]) < 1e-9
        n_bit_all += int(m)
        bit_anchors[tn] = {
            'got': got, 'want': BIT11[tn],
            'match': bool(m)}
    log('N replay: +2 bit anchors (total '
        '%d/%d)' % (n_bit_all,
                    N_BIT_TOT))
# N1: neg_d0.5 replay (soft anchor
# 0.2421875 from 3146)
dv_n = (pc1_res
        / max(np.linalg.norm(pc1_res),
              1e-12) * pnorm38 * 0.5) \
    .astype(np.float32)
dv_n_tile = np.tile(dv_n[None, :],
                    (NCAP, 1))
_run_vec_trial('n_neg_d0.5', 38,
               -dv_n_tile, 1.0, 'allstep',
               rows_scan, base12_P,
               save_gens=True)
chg_n = E_res['n_neg_d0.5']['chg']
_gen_n = E_res['n_neg_d0.5']['gens']
fs_n = fstep_store['n_neg_d0.5']
log('N1 n_neg_d0.5 done (3146 want '
    '0.2422)')
n_repro_ok = abs(chg_n - 0.2421875) < 1e-9
log('N1 soft anchor vs 3146: %s'
    % n_repro_ok)
# flip rows (3147 frozen list)
flip_rows = []
for j in range(NCAP):
    if fstep_store['n_neg_d0.5'][j] >= 0:
        flip_rows.append(j)
flip_rows = flip_rows[:8]
NF = len(flip_rows)
log('N2: %d flip rows %s'
    % (NF, flip_rows))
if not SMOKE:
    assert flip_rows == N47['flip_rows'], \
        flip_rows
    assert [int(fstep_store['n_neg_d0.5'][j])
            for j in flip_rows] == \
        N47['fsteps'], 'fsteps drift'
    log('N2 frozen-list match ok (3147)')

# ================================================================
# PART X2: tail sign-crossing mid-dose fill
# ================================================================
log('== PART X2: sign-cross fill ==')
TAIL_DOSES = (1.0, 2.0) + tuple(XDOSE_MID) \
    + (4.0,)
for dd in TAIL_DOSES:
    _run_coord_trial('s2_tailpos_d%g' % dd,
                     TAIL25, 1, dd, rows_A1,
                     base12_A1)
    _run_coord_trial('s2_tailneg_d%g' % dd,
                     TAIL25, -1, dd, rows_A1,
                     base12_A1)
if not SMOKE:
    for tn, wv in WANT_TAIL.items():
        got = E_res[tn]['chg']
        m = abs(got - wv) < 1e-9
        n_bit_all += int(m)
        bit_anchors[tn] = {
            'got': got, 'want': wv,
            'match': bool(m)}
    log('X2 replay: +5 bit anchors (total '
        '%d/%d)' % (n_bit_all, N_BIT_TOT))
posc = {dd: _chg('s2_tailpos_d%g' % dd)
        for dd in TAIL_DOSES}
negc = {dd: _chg('s2_tailneg_d%g' % dd)
        for dd in TAIL_DOSES}
diffc = {dd: posc[dd] - negc[dd]
         for dd in TAIL_DOSES}
tp2_repro = abs(posc[2.0] - 0.2578125) \
    < 1e-9
log('X2-SOFT: pos_d2 %.4f vs 3147 '
    '0.2578 repro %s' % (posc[2.0],
                         tp2_repro))
d_lo = None
d_hi = None
for dd in TAIL_DOSES:
    if diffc[dd] > SIGN_TOL:
        d_lo = dd
    elif d_lo is not None \
            and d_hi is None \
            and diffc[dd] < -SIGN_TOL:
        d_hi = dd
        break
if d_lo is not None and d_hi is not None:
    xcross_tag = 'tail_xcross_located'
    xcross_int = [float(d_lo),
                  float(d_hi)]
else:
    xcross_tag = 'tail_xcross_gradual'
    xcross_int = None
log('X-GATE2: pos %s'
    % json.dumps({'%g' % k: round(v, 4)
                  for k, v
                  in sorted(posc.items())}))
log('X-GATE2: neg %s'
    % json.dumps({'%g' % k: round(v, 4)
                  for k, v
                  in sorted(negc.items())}))
log('X-GATE2: diff %s -> %s (interval %s)'
    % (json.dumps({'%g' % k: round(v, 4)
                     for k, v
                     in sorted(diffc.items())}),
       xcross_tag, xcross_int))
# fstep structure: pos_d1 flips vs
# neg_d4 flips (format vs answer-side)
fs_p1 = fstep_store['s2_tailpos_d1']
fs_n4 = fstep_store['s2_tailneg_d4']
fl_p = [int(fs_p1[j]) for j in range(NCAP)
        if fs_p1[j] >= 0]
fl_n = [int(fs_n4[j]) for j in range(NCAP)
        if fs_n4[j] >= 0]
med_fp = float(np.median(fl_p)) \
    if fl_p else -1.0
med_fn = float(np.median(fl_n)) \
    if fl_n else -1.0
early_p = float(np.mean([v <= 1
                         for v in fl_p])) \
    if fl_p else 0.0
early_n = float(np.mean([v <= 1
                         for v in fl_n])) \
    if fl_n else 0.0
if med_fn > med_fp:
    fstep_tag = 'xcross_fstep_neglate'
elif med_fp > med_fn:
    fstep_tag = 'xcross_fstep_poslate'
else:
    fstep_tag = 'xcross_fstep_equal'
log('X-SOFT2: pos_d1 flips n=%d med_fs='
    '%.1f early=%.2f | neg_d4 flips n=%d '
    'med_fs=%.1f early=%.2f -> %s'
    % (len(fl_p), med_fp, early_p,
       len(fl_n), med_fn, early_n,
       fstep_tag))

# ================================================================
# PART H: top-2 source write-side identity
# ================================================================
log('== PART H: top-2 identity ==')
for dd in H_DOSES:
    _run_coord_trial('x_h1_d%g' % dd,
                     [HEAD_COORDS[0]], -1, dd,
                     rows_A1, base12_A1)
    _run_coord_trial('x_h2_d%g' % dd,
                     [HEAD_COORDS[1]], -1, dd,
                     rows_A1, base12_A1)
    _run_coord_trial('x_h12_d%g' % dd,
                     HEAD_COORDS, -1, dd,
                     rows_A1, base12_A1)
c1_2 = _chg('x_h1_d2')
c2_2 = _chg('x_h2_d2')
c12_2 = _chg('x_h12_d2')
resid_pair = c12_2 - (c1_2 + c2_2)
if abs(resid_pair) < SIGN_TOL:
    h_pair = 'h_pair_additive'
elif resid_pair > SIGN_TOL:
    h_pair = 'h_pair_super'
else:
    h_pair = 'h_pair_sub'
share_top2 = c12_2 / max(
    DVALS46['d_co50ex_d2.0'], 1e-9)
log('H-GATE1: solo1 %.4f solo2 %.4f pair '
    '%.4f resid %+.4f -> %s (share of '
    'co50ex_d2 %.3f)'
    % (c1_2, c2_2, c12_2, resid_pair,
       h_pair, share_top2))


def _mk_head_abl(head_h):
    def pre(mod, args):
        x = args[0]
        xb = x.view(x.shape[0], x.shape[1],
                    NHEADS, -1).clone()
        xb[:, :, head_h, :] = 0
        return (xb.view(x.shape),)
    return pre


def _cap_swap_head(rows_all, cap_layers,
                   head_h):
    """Single-sample capture with swap4
    layer-skip + o_proj-input head
    ablation (R55-legal)."""
    OUTS = {l: np.zeros((len(rows_all),
                         HIDG),
                        dtype=np.float32)
            for l in cap_layers}
    lyr17 = model_g.model.layers[17]
    for j in range(len(rows_all)):
        feats = {l: None
                 for l in cap_layers}
        hooks = []
        hooks.append(
            lyr17.self_attn.o_proj
            .register_forward_pre_hook(
                _mk_head_abl(head_h)))
        for l in SWAP_L:
            lyr = model_g.model.layers[l]

            def _swap(mod, inp, out):
                o2 = out[0] \
                    if isinstance(out,
                                  tuple) \
                    else out
                i2 = inp[0] \
                    if isinstance(inp,
                                  tuple) \
                    else inp
                o2.copy_(
                    i2.to(o2.dtype))

            hooks.append(
                lyr.register_forward_hook(
                    _swap))

        def _mk(_l):
            def hook(mod, inp, out):
                o2 = out[0] \
                    if isinstance(out,
                                  tuple) \
                    else out
                feats[_l] = \
                    o2[0, -1, :].detach()
            return hook

        for l in cap_layers:
            hooks.append(
                model_g.model.layers[l]
                .register_forward_hook(
                    _mk(l)))
        try:
            with torch.inference_mode():
                model_g(
                    torch.tensor(
                        [list(rows_all[j])],
                        device='cuda'),
                    use_cache=False)
        finally:
            for hk in hooks:
                hk.remove()
        for l in cap_layers:
            OUTS[l][j] = feats[l].float() \
                .cpu().numpy()
        del feats
    return OUTS


rows_head = rows_cap[:HEAD_ROWS]
_KH = CK['data'].get('headab')
if _KH is not None:
    head_contrib = np.asarray(
        _KH['contrib'],
        dtype=np.float32)
    log('headab RESUMED (%d heads)'
        % head_contrib.shape[0])
else:
    head_contrib = np.zeros(
        (NHEADS, HEAD_ROWS, 2),
        dtype=np.float32)
    for h in range(NHEADS):
        sw = _cap_swap_head(
            rows_head, [19], h)[19]
        fh = (sw - base_states[19]
              [:HEAD_ROWS]) \
            .astype(np.float32)
        for k, c in enumerate(
                HEAD_COORDS):
            head_contrib[h, :, k] = (
                np.abs(field[19]
                       [:HEAD_ROWS, c]
                       .astype(np.float64))
                - np.abs(fh[:, c]
                         .astype(
                             np.float64))
                ).astype(np.float32)
        log('headab h=%02d done' % h)
    ck_save('headab', {
        'contrib': head_contrib})
contrib_head = np.median(head_contrib,
                         axis=1).sum(axis=1)
order_head = np.argsort(-contrib_head)
pos_sum = float(np.sum(
    np.maximum(contrib_head, 0.0)))
if pos_sum > 1e-9:
    frac_top2 = float(
        contrib_head[order_head[:2]]
        .sum() / pos_sum)
else:
    frac_top2 = 0.0
if frac_top2 > 0.4:
    h_head = 'head_conc_top2'
elif frac_top2 > 0.25:
    h_head = 'head_conc_moderate'
else:
    h_head = 'head_conc_diffuse'
log('H-GATE2: per-head contrib top6 %s '
    'frac_top2 %.3f -> %s'
    % (json.dumps([round(
        float(contrib_head[h]), 4)
        for h in order_head[:6]]),
       frac_top2, h_head))
# H3: top-50 overlap with frozen dvecs
dv29_mean = np.abs(
    dvec[29].astype(np.float64)
    .mean(axis=0))
dv19_mean = np.abs(
    dvec19.astype(np.float64)
    .mean(axis=0))
top29 = np.argsort(-dv29_mean)[:50]
top19 = np.argsort(-dv19_mean)[:50]
s29 = set(top29.tolist())
s19 = set(top19.tolist())
in29 = [int(c) in s29
        for c in HEAD_COORDS]
in19 = [int(c) in s19
        for c in HEAD_COORDS]
rank29 = {}
rank19 = {}
for c in HEAD_COORDS:
    c = int(c)
    rank29[c] = (int(np.where(
        top29 == c)[0][0]) + 1
        if c in s29 else -1)
    rank19[c] = (int(np.where(
        top19 == c)[0][0]) + 1
        if c in s19 else -1)
n_in = int(sum(in29) + sum(in19))
if n_in >= 2:
    h_overlap = 'src_top_overlap'
elif n_in == 1:
    h_overlap = 'src_top_partial'
else:
    h_overlap = 'src_top_absent'
log('H-SOFT3: ranks dv29 %s dv19 %s -> '
    '%s' % (json.dumps(rank29),
            json.dumps(rank19),
            h_overlap))

# ================================================================
# PART U: unembed-cancel causality
# ================================================================
log('== PART U: unembed cancel ==')
w_tok = WUG[SPEC_TOKS[0]] \
    .to(torch.float32).cpu().numpy() \
    .astype(np.float64)
w_tok_u = (w_tok / max(np.linalg.norm(
    w_tok), 1e-12)).astype(np.float32)
_rng_u = np.random.RandomState(RNG_SEED)
_r_u = _rng_u.randn(4096)
r_u = (_r_u / np.linalg.norm(_r_u)) \
    .astype(np.float32)
wdn_u = (w_dn_g.astype(np.float64)
         / max(np.linalg.norm(w_dn_g),
               1e-12)).astype(np.float32)
chg_only = chg_n
_run_vec_trial('u_neg_cancel', 38,
               -dv_n_tile, 1.0, 'allstep',
               rows_scan, base12_P,
               clip=[(39, w_tok_u, 1.0,
                      'all')])
_run_vec_trial('u_neg_rand', 38,
               -dv_n_tile, 1.0, 'allstep',
               rows_scan, base12_P,
               clip=[(39, r_u, 1.0, 'all')])
_run_vec_trial('u_neg_wdn', 38,
               -dv_n_tile, 1.0, 'allstep',
               rows_scan, base12_P,
               clip=[(39, wdn_u, 1.0,
                      'all')])
d_cancel = chg_only - _chg('u_neg_cancel')
d_rand = chg_only - _chg('u_neg_rand')
d_wdn = chg_only - _chg('u_neg_wdn')
if d_cancel > UCANCEL_TOL \
        and d_cancel > 2.0 * max(d_rand,
                                 0.01) \
        and d_cancel > d_wdn + 0.02:
    u_tag = 'uncancel_format_sufficient'
elif d_cancel > UCANCEL_TOL \
        and d_wdn > UCANCEL_TOL:
    u_tag = 'uncancel_nonspecific'
elif d_cancel <= UCANCEL_TOL:
    u_tag = 'uncancel_insufficient'
else:
    u_tag = 'uncancel_partial'
fs_can = fstep_store['u_neg_cancel']
fs_rnd = fstep_store['u_neg_rand']
rec_can = float(np.mean(
    [fs_can[j] == -1
     for j in flip_rows])) \
    if flip_rows else 0.0
rec_rnd = float(np.mean(
    [fs_rnd[j] == -1
     for j in flip_rows])) \
    if flip_rows else 0.0
log('U-GATE: only %.4f cancel %.4f rand '
    '%.4f wdn %.4f (d %+.4f/%+.4f/'
    '%+.4f) flip-rec can %.2f rnd %.2f '
    '-> %s'
    % (chg_only, _chg('u_neg_cancel'),
       _chg('u_neg_rand'),
       _chg('u_neg_wdn'), d_cancel,
       d_rand, d_wdn, rec_can, rec_rnd,
       u_tag))

# ================================================================
# PART V2: v1 perturbation symmetry
# ================================================================
log('== PART V2: v1 symmetry ==')
v1f = v1.astype(np.float32)
for (a, tname) in V2_ALPHA:
    _run_vec_trial(tname, 29, None, 1.0,
                   'allstep', rows_scan,
                   base12_P,
                   clip=[(38, v1f, a,
                          'all')])
if not SMOKE:
    for tn, wv in WANT_V2.items():
        got = E_res[tn]['chg']
        m = abs(got - wv) < 1e-9
        n_bit_all += int(m)
        bit_anchors[tn] = {
            'got': got, 'want': wv,
            'match': bool(m)}
    log('V2 replay: +2 bit anchors (total '
        '%d/%d)' % (n_bit_all, N_BIT_TOT))
sym25 = _chg('v2_amp_a025') / max(
    _chg('v2_clip_a025'), 1e-9)
sym50 = _chg('v2_amp_a050') / max(
    _chg('v2_clip_a050'), 1e-9)
if sym25 > 0.8 and sym50 > 0.8:
    v_sym = 'v1_sym_amplitude'
elif sym25 < 0.5 and sym50 < 0.5:
    v_sym = 'v1_sym_directional'
else:
    v_sym = 'v1_sym_mixed'
log('V-GATE2: amp025 %.4f vs clip025 '
    '%.4f (sym %.2f); amp050 %.4f vs '
    'clip050 %.4f (sym %.2f) -> %s'
    % (_chg('v2_amp_a025'),
       _chg('v2_clip_a025'), sym25,
       _chg('v2_amp_a050'),
       _chg('v2_clip_a050'), sym50,
       v_sym))

# ================================================================
# verdict + result.json + npz
# ================================================================
tags = ['a_3147_ok']
if not SMOKE:
    tags.append('repro_bit_%d' % n_bit_all)
    tags.append('repro_bit_ok'
                if n_bit_all == N_BIT_TOT
                else 'repro_bit_drift')
else:
    tags.append('repro_smoke_skipped')
tags.append('dvec19_repro_%s'
            % dvec19_sha[:6])
tags.append('field_self_ok'
            if field_self_ok
            else 'field_self_drift')
tags.append('z35_cos17_ok' if z35_ok
            else 'z35_cos17_drift')
tags.append('resid_anchor_ok'
            if resid_anchor_ok
            else 'resid_anchor_drift')
tags.append(xcross_tag)
tags.append(fstep_tag)
tags.append(h_pair)
tags.append(h_head)
tags.append(h_overlap)
tags.append(u_tag)
tags.append(v_sym)
tags.append('xphase_ok' if xphase_ok
            else 'xphase_drift')
tags.append('coverage_full')
verdict = '|'.join(tags)
log('VERDICT: %s' % verdict)

result = {
    'phase': 3148,
    'name': NAME,
    'smoke': SMOKE,
    'created': time.strftime(
        '%Y-%m-%d %H:%M:%S'),
    'runtime_s': time.time() - T0,
    'seal_sha8': hashlib.sha256(
        json.dumps(SEAL, sort_keys=True,
                   ensure_ascii=False)
        .encode('utf-8')).hexdigest()[:8],
    'verdict': verdict,
    'constants': SEAL['constants'],
    'part_a': {
        'res47_sha8': sha47,
        'seal47': SEAL47,
        'xphase_P': xphase_P,
        'xphase_A1': xphase_A1},
    'part_field': {
        'dvec19_sha8': dvec19_sha,
        'dvec19_sha_ok':
            bool(d19_sha_ok),
        'dvec19_medn': dvec19_medn,
        'field17_vs_dvec17_cos': fs17,
        'field19_vs_dvec19_cos': fs19,
        'field_self_ok':
            bool(field_self_ok),
        'z35_c17_maxdiff': z35_diff,
        'z35_ok': bool(z35_ok)},
    'part_resid': {
        'share38': share38,
        'wdn_p38': wdn_p38,
        'wdn_p39': wdn_p39,
        'dh19_wdn39': dh19_wdn39,
        'pnorm38': pnorm38,
        'resid_align': ra_all,
        'anchor_ok':
            bool(resid_anchor_ok)},
    'part_bits': {'bit_anchors':
                  bit_anchors},
    'part_x2': {
        'pos_curve': {'%g' % k:
                      float(v) for k, v
                      in sorted(
                          posc.items())},
        'neg_curve': {'%g' % k:
                      float(v) for k, v
                      in sorted(
                          negc.items())},
        'diff_curve': {'%g' % k:
                       float(v) for k, v
                       in sorted(
                           diffc.items())},
        'xcross_interval': xcross_int,
        'xcross_tag': xcross_tag,
        'tailpos_d2_repro':
            bool(tp2_repro),
        'flip_pos_d1': {
            'n': len(fl_p),
            'med_fstep': med_fp,
            'early_frac': early_p},
        'flip_neg_d4': {
            'n': len(fl_n),
            'med_fstep': med_fn,
            'early_frac': early_n},
        'fstep_tag': fstep_tag},
    'part_h': {
        'solo1_d2': c1_2,
        'solo2_d2': c2_2,
        'pair_d2': c12_2,
        'resid_pair': resid_pair,
        'h_pair': h_pair,
        'share_top2_vs_co50ex_d2':
            share_top2,
        'dose_solo1': {'%g' % dd:
                       _chg('x_h1_d%g' % dd)
                       for dd in H_DOSES},
        'dose_solo2': {'%g' % dd:
                       _chg('x_h2_d%g' % dd)
                       for dd in H_DOSES},
        'dose_pair': {'%g' % dd:
                      _chg('x_h12_d%g' % dd)
                      for dd in H_DOSES},
        'head_order_top6': [int(h) for h
                            in order_head
                            [:6]],
        'head_contrib_top6': [
            float(contrib_head[h])
            for h in order_head[:6]],
        'frac_top2': frac_top2,
        'h_head': h_head,
        'rank_dv29': rank29,
        'rank_dv19': rank19,
        'in_top50': {'dv29':
                     [bool(v) for v
                      in in29],
                     'dv19':
                     [bool(v) for v
                      in in19]},
        'h_overlap': h_overlap},
    'part_u': {
        'chg_only': chg_only,
        'chg_cancel': _chg('u_neg_cancel'),
        'chg_rand': _chg('u_neg_rand'),
        'chg_wdn': _chg('u_neg_wdn'),
        'd_cancel': d_cancel,
        'd_rand': d_rand,
        'd_wdn': d_wdn,
        'rec_cancel': rec_can,
        'rec_rand': rec_rnd,
        'u_tag': u_tag},
    'part_v2': {
        'chg_amp025': _chg('v2_amp_a025'),
        'chg_amp050': _chg('v2_amp_a050'),
        'chg_clip025': _chg(
            'v2_clip_a025'),
        'chg_clip050': _chg(
            'v2_clip_a050'),
        'sym25': sym25,
        'sym50': sym50,
        'v_sym': v_sym},
    }
RF = os.path.join(OUT, 'result.json')
with io.open(RF, 'w',
             encoding='utf-8') as f:
    json.dump(result, f,
              ensure_ascii=False,
              indent=1)
npz_out = {
    'dvec19_sha': np.array([dvec19_sha]),
    'head_contrib': head_contrib,
    'order_head':
        order_head.astype(np.int64),
    'tail_doses': np.array(TAIL_DOSES),
    'pos_curve': np.array(
        [posc[d] for d in TAIL_DOSES]),
    'neg_curve': np.array(
        [negc[d] for d in TAIL_DOSES]),
    'sym_ratios': np.array([sym25,
                            sym50]),
    'pc1_res': pc1_res.astype(
        np.float32)}
np.savez(os.path.join(OUT,
                      'p146_readout.npz'),
         **npz_out)
if not SMOKE:
    try:
        os.remove(CKPTF)
        log('ckpt cleaned (final)')
    except OSError:
        pass
log('result.json + npz written. DONE '
    '(verdict %s)' % verdict)

# p3148 patch2 applied
# p3148 patch3 applied
# rev-3148a patch1: flip_rows cap 8 (3147 NFLIP_CAP semantics)
