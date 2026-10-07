# -*- coding: utf-8 -*-
"""p3148 patch1: replace head section
(docstring .. PART A) with 3148 anchors
+ 3147 link asserts. Idempotent."""
import io

FP = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
      r'\phase3148_omega_p146_xcross_'
      r'headsrc_uncancel_v1sym.py')

s = io.open(FP, encoding='utf-8').read()
if 'p3148 patch1 applied' in s:
    print('ALREADY_APPLIED')
    raise SystemExit

MARK = ('# ================================================================\n'
        '# materials factory (3142 lineage)')
idx = s.index(MARK)
tail = s[idx:]

NEW_HEAD = r'''# -*- coding: utf-8 -*-
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
NCAP = 6 if SMOKE else 128
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

'''

out = NEW_HEAD + '\n# p3148 patch1 applied\n\n' + tail
io.open(FP, 'w', encoding='utf-8',
        newline='\n').write(out)
print('PATCH1_OK head_len=%d total=%d'
      % (len(NEW_HEAD), len(out)))
