# -*- coding: utf-8 -*-
"""Phase 3146 (Omega-P144): tail-signal locus
subdivision (TAIL25 top10/bottom15 by IDEINT
enrichment @d4 + full dose matrix head/tail/
gbot/co36full/co50ex x d{1,2,4} A1 rows @L17)
+ pc1 accumulation anatomy (hist-condition
full-layer spectrum L20-39: pc1-hist replay
injection vs base-hist injection + final-norm
locus decomposition of the L39 1.9x logit
jump) + pcres dose curve (@L38 d{0.25,0.5,1.0}
+/-, pos-neg gap evolution + neg first-step
flip structure) + v1 clean-window intervention
(partial-magnitude clip alpha{0.25,0.5} +
decode-only clip -> non-dirty window search,
then joint/dv29 gap re-test under best window).

Preregistered in 3145 closeout (Omega-P144):
(1) tail locus: d4 subdivision TTOP10 vs
TBOT15 (sgn-1) -> top10/bot15/diffuse; dose
matrix head/headpos/gbot/co36full/co50ex/
tail d{1,2,4} -> mono/sat/flat per set;
(2) hist spectrum: 16 diverged rows tf
captures L20-39 under base-hist and pc1-hist
prefixes, injection dv_pc1@L29 -> per-layer
dlogit SPEC_TOKS + w_dn + ||dh|| growth;
final-norm decomposition: dlogit(normed dh39)
vs dlogit(raw dh39) -> jump locus tag;
(3) pcres dose: @L38 sgn +/- x d{0.25,0.5,1.0}
(pnorm38 scale) -> gap(d)=pos-neg monotone
gate + neg fstep median early/late; d1.0
replayed as 2 bit anchors (0.984375/0.421875);
(4) clean clip: base+{a025,a050,decode-only}
@38 windows (CLEAN_GATE 0.05) -> best clean
window -> joint/dv29 + window -> shrink2 =
1 - delta_post2/delta_pre(0.1640625) gate
0.5; no clean window -> v1_no_clean_window.

12 bit-anchor replays: b_pc1/b_dvec29/b_joint
(3142, 4th), d_head/d_gbot (3137, 3rd),
d_co36full/d_co50ex/d_tail_d1 (3144, 2nd),
d_tail_d2/d_tail_d4/t_pc1_inst/d_headpos_d1
(3145, 2nd). Bank shards REUSED from 3138.
Frozen dvecs from 3135 (sha-anchored).
Session baselines P+A1 with xphase recording.
Frozen before observation."""
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
NAME = ('omega_p144_taillocus_histfield_'
        'pcresdose_cleanclip')
SMOKE = os.environ.get('P3146_SMOKE',
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
D45 = RDIR + r'\phase3145' \
      r'\omega_p143_v1clip_pc1spec_' \
      r'headsign_residcausal'
MDIR_G = os.path.join(ROOT, 'models', 'hf',
                      'glm4-9b-chat-hf')
OUT = os.path.join(RDIR, 'phase3146', NAME)
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
CKPTF = os.path.join(OUT, 'p144_ckpt.pkl')
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
SHRINK2_GATE = 0.5
CLEAN_GATE = 0.05
ALPHA_GRID = (0.25, 0.5)
ONSET_TOL = 0.10
SIGN_TOL = 0.03
PCRES_DOSES = (0.25, 0.5, 1.0)
Z35_TOL = 0.08
FIELD_SELF_COS = 0.99
SHARE38_TOL = 0.02
WDN39_TOL = 0.15
ALIGN_TOL = 0.03
PNORM38_TOL = 1.0
SPEC_TOKS = [131401, 134772, 23386]
RNG_SEED = 3146

# ---------------- frozen 3145 anchors --
EXP_V45 = ('a_3144_ok|repro_bit_5|'
           'repro_bit_ok|dvec19_repro_6a0332|'
           'field_self_ok|z35_cos17_ok|'
           'resid_anchor_ok|v1_amp_dv29dom|'
           'v1_causal_dirty|pc1_onset_l29|'
           'pc1_history_dependent|'
           'pc1_hist_lock|'
           'pc1_replay_readout_sim|'
           'head_sign_asym|tail_dose_mono|'
           'tail_sym_break|resid_causal_active|'
           'answer_side_yes|xphase_ok|'
           'coverage_full')
RES45_SHA = '9d9cc6d1'
SEAL45 = '46c0187b'
XPHASE45 = 1.0
DVEC19_SHA = '6a0332a6'
MEDNORM19 = 5.643608093261719
# 12 bit anchors to replay on-site
BIT12 = {'b_pc1_l29_d2.0': 0.203125,
         'b_dvec29_l29_d2.0': 0.640625,
         'b_joint_l29_d2.0': 0.4765625,
         'd_head_d1.0': 0.21875,
         'd_gbot_d1.0': 0.1171875,
         'd_co36full_d1.0': 0.140625,
         'd_co50ex_d1.0': 0.421875,
         'd_tail_d1.0': 0.09375,
         'd_tail_d2.0': 0.171875,
         'd_tail_d4.0': 0.3046875,
         'd_headpos_d1.0': 0.140625,
         't_pc1_l29_inst': 0.0625}
# 3145 part_v1clip anchors
A45_JOINT = -59.116463
A45_DV29 = -67.020531
A45_PC1C = 7.904068
A45_RATIO = 8.479245
CLIP45 = {'jc_l30': 0.6640625,
          'jc_l33': 0.7265625,
          'jc_l36': 0.7890625,
          'jc_l38': 0.9609375,
          'jc_l39': 0.65625,
          'bc_l38': 0.8046875,
          'dc_l38': 0.9609375}
CHG_DV29_45 = 0.640625
CHG_JOINT_45 = 0.4765625
DELTA_PRE45 = 0.1640625
CHG_BC_45 = 0.8046875
# 3145 part_spec anchors
CHG_INST_45 = 0.0625
CHG_ALL_45 = 0.203125
INST_RATIO_45 = 0.3076923076923077
SPEC45 = {'131401': {29: 1.2012430724807928,
                     38: 1.474479109456297,
                     39: 2.832020115747582},
          '134772': {39: 2.408350798941683},
          '23386': {39: 2.5159921381218737}}
ONSET45 = 29
MED_WDN39_45 = 0.7459101974964142
LOCK45 = 0.875
# 3145 part_headtail anchors
HT45 = {'d_head': 0.21875,
        'd_headpos': 0.140625,
        'd_gbot': 0.1171875,
        'd_tail_d1': 0.09375,
        'd_tail_d2': 0.171875,
        'd_tail_d4': 0.3046875,
        'd_tailpos_d2': 0.2578125}
# 3145 part_residcausal anchors
RC45 = {'e_pcres_l19_pos': 1.0,
        'e_pcres_l19_neg': 1.0,
        'e_pcres_l38_pos': 0.984375,
        'e_pcres_l38_neg': 0.421875}
DYES45 = 1.119962104864634
DNO45 = -1.1540060264129592
WDNCHK45 = 2.2739681312775932
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
LEDGER_N = 282

# ---------------- seal ------------------
SEAL = {
    'phase': 3146,
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
        'SHRINK2_GATE': SHRINK2_GATE,
        'CLEAN_GATE': CLEAN_GATE,
        'ALPHA_GRID': list(ALPHA_GRID),
        'ONSET_TOL': ONSET_TOL,
        'SIGN_TOL': SIGN_TOL,
        'PCRES_DOSES': list(PCRES_DOSES),
        'Z35_TOL': Z35_TOL,
        'FIELD_SELF_COS': FIELD_SELF_COS,
        'SPEC_TOKS': SPEC_TOKS,
        'RNG_SEED': RNG_SEED,
        'BANK_REUSE_3138': True},
    'anchors': {
        'res45_verdict': EXP_V45,
        'res45_sha8': RES45_SHA,
        'seal45': SEAL45,
        'xphase45': XPHASE45,
        'dvec19_sha8': DVEC19_SHA,
        'mednorm19': MEDNORM19,
        'bit12': BIT12,
        'a45_joint': A45_JOINT,
        'a45_dv29': A45_DV29,
        'a45_pc1c': A45_PC1C,
        'a45_ratio': A45_RATIO,
        'clip45': CLIP45,
        'chg_dv29_45': CHG_DV29_45,
        'chg_joint_45': CHG_JOINT_45,
        'delta_pre45': DELTA_PRE45,
        'chg_bc45': CHG_BC_45,
        'chg_inst45': CHG_INST_45,
        'chg_all45': CHG_ALL_45,
        'inst_ratio45': INST_RATIO_45,
        'spec45': {k: {str(a): v for a, v
                       in d.items()}
                   for k, d in
                   SPEC45.items()},
        'onset45': ONSET45,
        'med_wdn39_45': MED_WDN39_45,
        'lock45': LOCK45,
        'ht45': HT45,
        'rc45': RC45,
        'dyes45': DYES45, 'dno45': DNO45,
        'wdnchk45': WDNCHK45,
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
    'prereg': ('3145 closeout Omega-P144: '
               '(1) tail locus: TTOP10 vs '
               'TBOT15 sgn-1 @d4 A1 rows L17 '
               '-> top10/bot15/diffuse; dose '
               'matrix head/headpos/gbot/'
               'co36full/co50ex/tail d{1,2,4} '
               '-> mono/sat/flat per set; '
               '12 bit replays; (2) hist '
               'spectrum: 16 diverged rows '
               'tf captures L20-39 base-hist '
               'vs pc1-hist injection dv_pc1@'
               'L29 -> dlogit SPEC_TOKS + '
               'w_dn + norm growth; final-'
               'norm decomposition dlogit('
               'normed dh39) vs raw -> jump '
               'locus finalnorm/downstream/'
               'damped; (3) pcres @L38 +/- x '
               'd{0.25,0.5,1.0} -> gap '
               'monotone gate + neg fstep '
               'median early/late; d1.0 '
               'replayed 2 bit anchors; '
               '(4) clean clip: base+{a025,'
               'a050,decode}@38 windows '
               '(CLEAN_GATE 0.05) -> best '
               'window -> joint/dv29 gap '
               're-test shrink2 = '
               '1-delta_post2/0.1640625 gate '
               '0.5; no clean window -> '
               'v1_no_clean_window. Bank '
               'reused 3138. Frozen before '
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
# PART A: 3145 link asserts
# ================================================================
log('== PART A: link asserts ==')
res45 = json.load(io.open(
    D45 + r'\result.json', encoding='utf-8'))
assert res45['smoke'] is False
V45 = res45['verdict']
assert V45 == EXP_V45, V45
raw45 = io.open(
    D45 + r'\result.json', 'rb').read()
sha45 = hashlib.sha256(raw45).hexdigest()[:8]
assert sha45 == RES45_SHA, sha45
assert str(res45['seal_sha8']) == SEAL45
pa45 = res45['part_a']
assert abs(float(pa45['xphase_P'])
           - XPHASE45) < 1e-12
assert abs(float(pa45['xphase_A1'])
           - XPHASE45) < 1e-12
pv45 = res45['part_v1clip']
for tn, v in BIT12.items():
    if tn in ('d_head_d1.0', 'd_gbot_d1.0'):
        continue
    if tn in pv45['bit_anchors']:
        assert abs(float(
            pv45['bit_anchors'][tn]
            ['got']) - v) < 1e-9, tn
assert pv45['bit_anchors']['d_head_d1.0'][
    'got'] == 0.21875
assert pv45['bit_anchors']['d_gbot_d1.0'][
    'got'] == 0.1171875
assert abs(float(pv45['amp_joint'])
           - A45_JOINT) < 1e-4
assert abs(float(pv45['amp_dv29'])
           - A45_DV29) < 1e-4
assert abs(float(pv45['amp_pc1contrib'])
           - A45_PC1C) < 1e-4
assert abs(float(pv45['amp_ratio'])
           - A45_RATIO) < 1e-4
assert pv45['amp_tag'] == 'v1_amp_dv29dom'
for k, v in CLIP45.items():
    assert abs(float(pv45['clip_res'][k])
               - v) < 1e-9, k
assert abs(float(pv45['chg_dv29'])
           - CHG_DV29_45) < 1e-9
assert abs(float(pv45['chg_joint'])
           - CHG_JOINT_45) < 1e-9
assert abs(float(pv45['delta_pre'])
           - DELTA_PRE45) < 1e-9
assert abs(float(pv45['chg_base_clip'])
           - CHG_BC_45) < 1e-9
assert pv45['v1_tag'] == 'v1_causal_dirty'
ps45 = res45['part_spec']
assert abs(float(ps45['chg_inst'])
           - CHG_INST_45) < 1e-9
assert abs(float(ps45['chg_allstep'])
           - CHG_ALL_45) < 1e-9
assert abs(float(ps45['inst_ratio'])
           - INST_RATIO_45) < 1e-9
assert ps45['inst_tag'] == \
    'pc1_history_dependent'
for tok, dd in SPEC45.items():
    for r, v in dd.items():
        assert abs(float(
            ps45['spec'][tok][str(r)])
            - v) < 1e-9, (tok, r)
assert int(ps45['onset_layer']) == ONSET45
assert ps45['onset_tag'] == 'pc1_onset_l29'
assert abs(float(ps45['med_wdn39_base_hist'])
           - MED_WDN39_45) < 1e-9
assert abs(float(ps45['lock_bh'])
           - LOCK45) < 1e-9
assert abs(float(ps45['lock_ph'])
           - LOCK45) < 1e-9
assert ps45['hist_tag'] == 'pc1_hist_lock'
assert ps45['drop_tag'] == \
    'pc1_replay_readout_sim'
ph45 = res45['part_headtail']
for k, v in HT45.items():
    assert abs(float(ph45[k]) - v) < 1e-9, k
assert ph45['head_tag'] == 'head_sign_asym'
assert ph45['dose_tag'] == 'tail_dose_mono'
assert ph45['sym_tag'] == 'tail_sym_break'
pc45 = res45['part_residcausal']
for k, v in RC45.items():
    assert abs(float(pc45['trials'][k])
               - v) < 1e-9, k
assert pc45['causal_tag'] == \
    'resid_causal_active'
assert abs(float(pc45['dyes39'])
           - DYES45) < 1e-9
assert abs(float(pc45['dno39'])
           - DNO45) < 1e-9
assert abs(float(pc45['wdn_check'])
           - WDNCHK45) < 1e-9
assert pc45['side_tag'] == 'answer_side_yes'
pf45 = res45['part_field']
assert str(pf45['dvec19_sha8']) == \
    DVEC19_SHA
assert abs(float(pf45['dvec19_medn'])
           - MEDNORM19) < 1e-9
assert pf45['field_self_ok'] is True
assert pf45['z35_ok'] is True
pr45 = res45['part_resid']
assert abs(float(pr45['share38'])
           - PSHARE_38) < 1e-4
assert abs(float(pr45['wdn_p39'])
           - PWDN_39) < 1e-4
assert abs(float(pr45['pnorm38'])
           - PNORM_38) < 1e-3
assert pr45['anchor_ok'] is True
log('A hard asserts ok (3145 sha8 %s seal '
    '%s: xphase/bit5/amp/clip_res/spec/'
    'headtail/residcausal/field/resid '
    'anchors all verified)' % (sha45,
                               SEAL45))

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
                  'tfbase', 'tfinj',
                  'tfpc1b', 'tfpc1i')
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
# PART B: bit replays + clean-window clip
# ================================================================
log('== PART B: bit replays + clean clip ==')
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


# B1: replay 3 generation bit anchors
_run_vec_trial('b_pc1_l29_d2.0', 29,
               dv_pc1_pos, 1.0, 'allstep',
               rows_scan, base12_P,
               save_gens=True)
pc1_sign = None
if not SMOKE and abs(
        E_res['b_pc1_l29_d2.0']['chg']
        - BIT12['b_pc1_l29_d2.0']) >= 1e-9:
    _run_vec_trial('b_pc1neg_l29_d2.0',
                   29, dv_pc1_neg, 1.0,
                   'allstep', rows_scan,
                   base12_P)
    if abs(E_res['b_pc1neg_l29_d2.0']
           ['chg']
           - BIT12['b_pc1_l29_d2.0']) < 1e-9:
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
        m = abs(got - BIT12[tn]) < 1e-9
        n_bit_all += int(m)
        bit_anchors[tn] = {
            'got': got, 'want': BIT12[tn],
            'match': bool(m)}
    log('B1 3142 repro: %d/3 bit-match'
        % n_bit_all)
else:
    log('B1 3142 repro SKIPPED (smoke)')

# B4: clean-window clip intervention
v1f = v1.astype(np.float32)
WINDOWS = {
    'a025': [(38, v1f, 0.25, 'all')],
    'a050': [(38, v1f, 0.5, 'all')],
    'deco': [(38, v1f, 1.0, 'decode')]}
for wn, cl in WINDOWS.items():
    _run_vec_trial('b4_base_%s' % wn, 29,
                   None, 1.0, 'allstep',
                   rows_scan, base12_P,
                   clip=cl)
clean_ws = [wn for wn in WINDOWS
            if E_res['b4_base_%s' % wn]
            ['chg'] < CLEAN_GATE]
log('B4 windows: %s (clean gate %.2f)'
    % (json.dumps({wn: round(
        E_res['b4_base_%s' % wn]['chg'], 4)
        for wn in WINDOWS}), CLEAN_GATE))
shrink2 = None
delta_post2 = None
if clean_ws:
    best_w = min(clean_ws,
                 key=lambda wn:
                 E_res['b4_base_%s' % wn]
                 ['chg'])
    log('B4 best clean window: %s'
        % best_w)
    for cond, dvb in (('joint', None),
                      ('dv29', None)):
        tname = 'b4_%s_%s' % (cond, best_w)
        _K = CK['data'].get(tname)
        if _K is not None:
            E_res[tname] = _K['res']
            log('%s RESUMED chg=%.4f'
                % (tname,
                   E_res[tname]['chg']))
            continue
        gen_l = []
        for b0 in range(0, NCAP,
                        GEN_BATCH):
            batch = rows_scan[
                b0:b0 + GEN_BATCH]
            if cond == 'joint':
                iv = [(29, dv_pc1[b0:b0
                                  + len(batch)],
                       1.0, 'allstep'),
                      (29, dv_dv29[b0:b0
                                   + len(batch)],
                       1.0, 'allstep')]
            else:
                iv = [(29, dv_dv29[b0:b0
                                   + len(batch)],
                       1.0, 'allstep')]
            gen_l.extend(gen_batch_g2(
                batch, inj_vec=iv,
                clip=WINDOWS[best_w]))
        chg_l, first_l, fs = \
            trial_metrics(gen_l, base12_P)
        E_res[tname] = {'chg': chg_l,
                        'first':
                        int(first_l)}
        fstep_store[tname] = fs
        log('%s: chg=%.4f first=%d'
            % (tname, chg_l, first_l))
        ck_save(tname, {
            'res': E_res[tname],
            'fstep': fs.astype(np.int8)
            .tolist()})
    chg_j2 = E_res['b4_joint_%s'
                    % best_w]['chg']
    chg_d2 = E_res['b4_dv29_%s'
                    % best_w]['chg']
    delta_post2 = chg_d2 - chg_j2
    shrink2 = (1.0 - delta_post2
               / DELTA_PRE45) \
        if abs(DELTA_PRE45) > 1e-9 else 0.0
    clean_tag = (
        'v1_partial_clean_confirmed'
        if shrink2 >= SHRINK2_GATE
        else 'v1_partial_clean_absent')
    log('B4-GATE: window %s dv29 %.4f '
        'joint %.4f delta_post2 %+.4f '
        'shrink2 %.3f (gate %.2f) -> %s'
        % (best_w, chg_d2, chg_j2,
           delta_post2, shrink2,
           SHRINK2_GATE, clean_tag))
else:
    clean_tag = 'v1_no_clean_window'
    best_w = None
    log('B4-GATE: no clean window under '
        'alpha{0.25,0.5}/decode-only -> '
        'v1_no_clean_window')

# ================================================================
# PART T: pc1 accumulation anatomy
# ================================================================
log('== PART T: pc1 accumulation ==')
# T1: instant injection replay (bit)
_run_vec_trial('t_pc1_l29_inst', 29,
               dv_pc1, 1.0, 0,
               rows_scan, base12_P)
if not SMOKE:
    got = E_res['t_pc1_l29_inst']['chg']
    m = abs(got - BIT12['t_pc1_l29_inst']) \
        < 1e-9
    n_bit_all += int(m)
    bit_anchors['t_pc1_l29_inst'] = {
        'got': got,
        'want': BIT12['t_pc1_l29_inst'],
        'match': bool(m)}
    log('T1 replay: +1 bit anchor (total '
        '%d/12)' % n_bit_all)
gens_pc1 = E_res['b_pc1_l29_d2.0']['gens']
div_rows = []
for j in range(NCAP):
    g12 = pad12(gens_pc1[j])
    if g12 != base12_P[j]:
        div_rows.append(j)
    if len(div_rows) >= SPEC_ROWS:
        break
TN = len(div_rows)
log('T4: %d diverged rows (cap %d)'
    % (TN, SPEC_ROWS))
tf_rows = []
tf_pc1_rows = []
tstar_tok = []
for j in div_rows:
    tstar = -1
    g12 = pad12(gens_pc1[j])
    for t in range(N_NEW):
        if g12[t] != base12_P[j][t]:
            tstar = t
            break
    assert tstar >= 0
    tf_rows.append(list(PREFIX_IDS)
                   + list(rows_scan[j])
                   + list(base12_P[j][:tstar]))
    tf_pc1_rows.append(
        list(PREFIX_IDS)
        + list(rows_scan[j])
        + list(gens_pc1[j][:tstar]))
    tstar_tok.append(int(base12_P[j][tstar]))
SPEC_CAP_L = list(range(20, 40))
_KT = CK['data'].get('tfbase')
if _KT is not None:
    h_tf_base = {int(k):
                 v.astype(np.float32)
                 for k, v in
                 _KT['h'].items()}
    log('tfbase RESUMED')
else:
    h_tf_base = capture_states(
        tf_rows, SPEC_CAP_L)
    ck_save('tfbase', {
        'h': {str(l): h_tf_base[l]
              .astype(np.float16)
              for l in SPEC_CAP_L}})
_KI = CK['data'].get('tfinj')
if _KI is not None:
    h_tf_inj = {int(k):
                v.astype(np.float32)
                for k, v in
                _KI['h'].items()}
    log('tfinj RESUMED')
else:
    h_tf_inj = capture_states_inject2(
        tf_rows, SPEC_CAP_L,
        [(29, dv_pc1[:TN], 1.0)])
    ck_save('tfinj', {
        'h': {str(l): h_tf_inj[l]
              .astype(np.float16)
              for l in SPEC_CAP_L}})
_K4 = CK['data'].get('tfpc1b')
if _K4 is not None:
    h_pc1_base = {int(k):
                  v.astype(np.float32)
                  for k, v in
                  _K4['h'].items()}
    log('tfpc1b RESUMED')
else:
    h_pc1_base = capture_states(
        tf_pc1_rows, SPEC_CAP_L)
    ck_save('tfpc1b', {
        'h': {str(l): h_pc1_base[l]
              .astype(np.float16)
              for l in SPEC_CAP_L}})
_K5 = CK['data'].get('tfpc1i')
if _K5 is not None:
    h_pc1_inj = {int(k):
                 v.astype(np.float32)
                 for k, v in
                 _K5['h'].items()}
    log('tfpc1i RESUMED')
else:
    h_pc1_inj = capture_states_inject2(
        tf_pc1_rows, SPEC_CAP_L,
        [(29, dv_pc1[:TN], 1.0)])
    ck_save('tfpc1i', {
        'h': {str(l): h_pc1_inj[l]
              .astype(np.float16)
              for l in SPEC_CAP_L}})
w_rows_np = {}
for tok in SPEC_TOKS:
    w_rows_np[tok] = \
        WUG[tok].to(torch.float32) \
        .cpu().numpy() \
        .astype(np.float64)
dl_base = {}
dl_hist = {}
for tok in SPEC_TOKS:
    dl_base[tok] = {}
    dl_hist[tok] = {}
    for r in SPEC_CAP_L:
        dhb = (h_tf_inj[r].astype(np.float64)
               - h_tf_base[r]
               .astype(np.float64))
        dhh = (h_pc1_inj[r].astype(np.float64)
               - h_pc1_base[r]
               .astype(np.float64))
        dl_base[tok][r] = float(np.median(
            dhb @ w_rows_np[tok]))
        dl_hist[tok][r] = float(np.median(
            dhh @ w_rows_np[tok]))
wdn_base = {}
wdn_hist = {}
nrm_base = {}
nrm_hist = {}
for r in SPEC_CAP_L:
    dhb = (h_tf_inj[r].astype(np.float64)
           - h_tf_base[r].astype(np.float64))
    dhh = (h_pc1_inj[r].astype(np.float64)
           - h_pc1_base[r].astype(np.float64))
    wdn_base[r] = float(np.median(
        dhb @ w_dn_g.astype(np.float64)))
    wdn_hist[r] = float(np.median(
        dhh @ w_dn_g.astype(np.float64)))
    nrm_base[r] = float(np.median(
        np.linalg.norm(dhb, axis=1)))
    nrm_hist[r] = float(np.median(
        np.linalg.norm(dhh, axis=1)))
onset_hist = None
for r in SPEC_CAP_L:
    if dl_hist[SPEC_TOKS[0]][r] > ONSET_TOL:
        onset_hist = r
        break
onset_hist_tag = (
    'hist_onset_l%02d' % onset_hist
    if onset_hist is not None
    else 'hist_onset_none')
tk0 = SPEC_TOKS[0]
jump_hist = dl_hist[tk0][39] / max(
    abs(dl_hist[tk0][38]), 0.1)
log('T4-SOFT: hist dlogit(%d) %s'
    % (tk0, json.dumps(
        {str(r): round(dl_hist[tk0][r], 3)
         for r in SPEC_CAP_L})))
log('T4-SOFT: base dlogit(%d) %s'
    % (tk0, json.dumps(
        {str(r): round(dl_base[tk0][r], 3)
         for r in SPEC_CAP_L})))
log('T4-SOFT: wdn_hist %s'
    % json.dumps({str(r): round(v, 3)
                  for r, v in
                  wdn_hist.items()}))
log('T4-SOFT: ||dh||_hist %s'
    % json.dumps({str(r): round(v, 2)
                  for r, v in
                  nrm_hist.items()}))
log('T4-GATE: hist onset (first med '
    'dlogit(%d)>%.2f) -> %s; jump39_hist '
    '%.2f' % (tk0, ONSET_TOL,
              onset_hist_tag, jump_hist))
# T5: final-norm locus decomposition
def _nrm_rows(H):
    t = torch.as_tensor(
        np.asarray(H, dtype=np.float32),
        device='cuda') \
        .to(torch.bfloat16)
    with torch.inference_mode():
        o = norm_g(t)
    return o.float().cpu().numpy() \
        .astype(np.float64)


dl_norm = {}
for tok in SPEC_TOKS:
    dl_norm[tok] = {}
    for r in (38, 39):
        nb = _nrm_rows(h_tf_base[r])
        ni = _nrm_rows(h_tf_inj[r])
        dhn = (ni - nb)
        dl_norm[tok][r] = float(np.median(
            dhn @ w_rows_np[tok]))
jump_raw = dl_base[tk0][39] / max(
    abs(dl_base[tk0][38]), 0.1)
jump_norm = dl_norm[tk0][39] / max(
    abs(dl_norm[tk0][38]), 0.1)
if jump_norm > jump_raw * 1.15:
    l39_tag = 'l39_jump_finalnorm'
elif jump_norm < jump_raw * 0.85:
    l39_tag = 'l39_jump_damped_by_norm'
else:
    l39_tag = 'l39_jump_downstream'
log('T5-GATE: dlogit(%d) raw38 %.3f '
    'raw39 %.3f (jump %.2f) norm38 %.3f '
    'norm39 %.3f (jump %.2f) -> %s'
    % (tk0, dl_base[tk0][38],
       dl_base[tk0][39], jump_raw,
       dl_norm[tk0][38], dl_norm[tk0][39],
       jump_norm, l39_tag))

# ================================================================
# PART D: tail locus + dose matrix
# ================================================================
log('== PART D: tail locus + dose ==')
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
GB25 = list(G_BOT25)
CO36L = [int(c) for c in co36]
co50 = CO_SETS['co50']
co50ex = sorted(set(co50.tolist())
                - set(co36.tolist()))
D_TRIALS = [
    # bit replays
    ('d_head_d1.0', HEAD25, -1, 1.0),
    ('d_gbot_d1.0', GB25, -1, 1.0),
    ('d_co36full_d1.0', CO36L, -1, 1.0),
    ('d_co50ex_d1.0', co50ex, -1, 1.0),
    ('d_tail_d1.0', TAIL25, -1, 1.0),
    ('d_tail_d2.0', TAIL25, -1, 2.0),
    ('d_tail_d4.0', TAIL25, -1, 4.0),
    ('d_headpos_d1.0', HEAD25, 1, 1.0),
    # dose matrix extension
    ('d_head_d2.0', HEAD25, -1, 2.0),
    ('d_head_d4.0', HEAD25, -1, 4.0),
    ('d_headpos_d2.0', HEAD25, 1, 2.0),
    ('d_headpos_d4.0', HEAD25, 1, 4.0),
    ('d_gbot_d2.0', GB25, -1, 2.0),
    ('d_gbot_d4.0', GB25, -1, 4.0),
    ('d_co36full_d2.0', CO36L, -1, 2.0),
    ('d_co36full_d4.0', CO36L, -1, 4.0),
    ('d_co50ex_d2.0', co50ex, -1, 2.0),
    ('d_co50ex_d4.0', co50ex, -1, 4.0),
    ('d_tailpos_d4.0', TAIL25, 1, 4.0),
    # tail locus subdivision @d4
    ('d_ttop10_d4.0', TTOP10, -1, 4.0),
    ('d_tbot15_d4.0', TBOT15, -1, 4.0)]
for (tname, coords, sgn, dsc) in D_TRIALS:
    _K = CK['data'].get(tname)
    if _K is not None:
        E_res[tname] = _K['res']
        log('%s RESUMED chg=%.4f'
            % (tname, E_res[tname]['chg']))
        continue
    gen_l = []
    for b0 in range(0, NCAP, GEN_BATCH):
        batch = rows_A1[b0:b0 + GEN_BATCH]
        gen_l.extend(gen_batch_g2(
            batch,
            inj=[(17, coords,
                  dsc * DELTA_L17, sgn, 0)]))
    chg_l, first_l, fs = trial_metrics(
        gen_l, base12_A1)
    E_res[tname] = {'chg': chg_l,
                    'first': int(first_l)}
    fstep_store[tname] = fs
    log('%s: chg=%.4f first=%d'
        % (tname, chg_l, first_l))
    ck_save(tname, {'res': E_res[tname],
                    'fstep': fs.astype(
                        np.int8).tolist()})
if not SMOKE:
    for tn in ('d_head_d1.0',
               'd_gbot_d1.0',
               'd_co36full_d1.0',
               'd_co50ex_d1.0',
               'd_tail_d1.0',
               'd_tail_d2.0',
               'd_tail_d4.0',
               'd_headpos_d1.0'):
        got = E_res[tn]['chg']
        m = abs(got - BIT12[tn]) < 1e-9
        n_bit_all += int(m)
        bit_anchors[tn] = {
            'got': got, 'want': BIT12[tn],
            'match': bool(m)}
    log('D replay: +8 bit anchors (total '
        '%d/12)' % n_bit_all)


def _dose_shape(c1, c2, c4):
    if c2 > c1 + 0.02 \
            and c4 > c2 + 0.02:
        return 'mono'
    if c2 > c1 + 0.02 \
            and abs(c4 - c2) <= 0.02:
        return 'sat'
    if c2 < c1 + 0.02 \
            and c4 < c1 + 0.02:
        return 'flat'
    return 'mixed'


dose_shapes = {
    'head': _dose_shape(
        E_res['d_head_d1.0']['chg'],
        E_res['d_head_d2.0']['chg'],
        E_res['d_head_d4.0']['chg']),
    'headpos': _dose_shape(
        E_res['d_headpos_d1.0']['chg'],
        E_res['d_headpos_d2.0']['chg'],
        E_res['d_headpos_d4.0']['chg']),
    'gbot': _dose_shape(
        E_res['d_gbot_d1.0']['chg'],
        E_res['d_gbot_d2.0']['chg'],
        E_res['d_gbot_d4.0']['chg']),
    'co36full': _dose_shape(
        E_res['d_co36full_d1.0']['chg'],
        E_res['d_co36full_d2.0']['chg'],
        E_res['d_co36full_d4.0']['chg']),
    'co50ex': _dose_shape(
        E_res['d_co50ex_d1.0']['chg'],
        E_res['d_co50ex_d2.0']['chg'],
        E_res['d_co50ex_d4.0']['chg']),
    'tail': _dose_shape(
        E_res['d_tail_d1.0']['chg'],
        E_res['d_tail_d2.0']['chg'],
        E_res['d_tail_d4.0']['chg'])}
chg_tt = E_res['d_ttop10_d4.0']['chg']
chg_tb = E_res['d_tbot15_d4.0']['chg']
if chg_tt > chg_tb + 0.03:
    tail_locus = 'tail_locus_top10'
elif chg_tb > chg_tt + 0.03:
    tail_locus = 'tail_locus_bot15'
else:
    tail_locus = 'tail_locus_diffuse'
log('D-GATE: dose shapes %s'
    % json.dumps(dose_shapes))
log('D-GATE: tail locus @d4: ttop10 '
    '%.4f vs tbot15 %.4f (tail_d4 %.4f) '
    '-> %s' % (chg_tt, chg_tb,
               E_res['d_tail_d4.0']['chg'],
               tail_locus))
head_dose_ratio = E_res[
    'd_head_d4.0']['chg'] / max(
    E_res['d_head_d1.0']['chg'], 1e-9)
tail_dose_ratio = E_res[
    'd_tail_d4.0']['chg'] / max(
    E_res['d_tail_d1.0']['chg'], 1e-9)
log('D-SOFT: head dose ratio d4/d1 %.2f '
    'vs tail %.2f' % (head_dose_ratio,
                      tail_dose_ratio))

# ================================================================
# PART E: pcres dose curve @L38
# ================================================================
log('== PART E: pcres dose curve ==')
E_TRIALS = [
    ('e_pcres38_d0.25_pos', 38, 1.0,
     0.25),
    ('e_pcres38_d0.25_neg', 38, -1.0,
     0.25),
    ('e_pcres38_d0.5_pos', 38, 1.0, 0.5),
    ('e_pcres38_d0.5_neg', 38, -1.0,
     0.5),
    ('e_pcres38_d1.0_pos', 38, 1.0, 1.0),
    ('e_pcres38_d1.0_neg', 38, -1.0,
     1.0)]
for (tname, il, sg, dsc) in E_TRIALS:
    _K = CK['data'].get(tname)
    if _K is not None:
        E_res[tname] = _K['res']
        fstep_store[tname] = np.asarray(
            _K.get('fstep', []),
            dtype=np.int8)
        log('%s RESUMED chg=%.4f'
            % (tname, E_res[tname]['chg']))
        continue
    dv_dose = (pc1_res
               / max(np.linalg.norm(pc1_res),
                     1e-12)
               * pnorm38 * float(dsc)) \
        .astype(np.float32)
    dv_tile = np.tile(dv_dose[None, :],
                      (NCAP, 1))
    gen_l = []
    for b0 in range(0, NCAP, GEN_BATCH):
        batch = rows_scan[b0:b0 + GEN_BATCH]
        gen_l.extend(gen_batch_g2(
            batch,
            inj_vec=[(il,
                      sg * dv_tile[
                          b0:b0 + len(batch)],
                      1.0, 'allstep')]))
    chg_l, first_l, fs = trial_metrics(
        gen_l, base12_P)
    E_res[tname] = {'chg': chg_l,
                    'first': int(first_l)}
    fstep_store[tname] = fs
    log('%s: chg=%.4f first=%d'
        % (tname, chg_l, first_l))
    ck_save(tname, {'res': E_res[tname],
                    'fstep': fs.astype(
                        np.int8).tolist()})
if not SMOKE:
    for tn, v in (
            ('e_pcres38_d1.0_pos',
             RC45['e_pcres_l38_pos']),
            ('e_pcres38_d1.0_neg',
             RC45['e_pcres_l38_neg'])):
        got = E_res[tn]['chg']
        m = abs(got - v) < 1e-9
        n_bit_all += int(m)
        bit_anchors[tn] = {
            'got': got, 'want': v,
            'match': bool(m)}
    log('E replay: +2 bit anchors (total '
        '%d/14)' % n_bit_all)
def _chg(tn):
    return E_res[tn]['chg']
gap_by_dose = {}
for dsc, tpos, tneg in (
        (0.25, 'e_pcres38_d0.25_pos',
         'e_pcres38_d0.25_neg'),
        (0.5, 'e_pcres38_d0.5_pos',
         'e_pcres38_d0.5_neg'),
        (1.0, 'e_pcres38_d1.0_pos',
         'e_pcres38_d1.0_neg')):
    gap_by_dose[dsc] = _chg(tpos) \
        - _chg(tneg)
g025 = gap_by_dose[0.25]
g050 = gap_by_dose[0.5]
g100 = gap_by_dose[1.0]
if g050 > g025 + 0.02 \
        and g100 > g050 + 0.02:
    gap_tag = 'pcres_gap_mono'
else:
    gap_tag = 'pcres_gap_nonmono'
fs_neg = []
for tn in ('e_pcres38_d0.25_neg',
           'e_pcres38_d0.5_neg'):
    fs = fstep_store.get(tn)
    if fs is None:
        continue
    fs_neg.extend(int(v) for v in fs
                  if v >= 0)
med_fs_neg = float(np.median(fs_neg)) \
    if fs_neg else -1.0
neg_flip_tag = (
    'neg_flip_early'
    if 0 <= med_fs_neg <= 2
    else 'neg_flip_late'
    if med_fs_neg > 2
    else 'neg_flip_none')
log('E-GATE: gap(d) %s -> %s; neg flips '
    'n=%d med_fstep=%.1f -> %s'
    % (json.dumps({'0.25': round(g025, 4),
                   '0.5': round(g050, 4),
                   '1.0': round(g100, 4)}),
       gap_tag, len(fs_neg), med_fs_neg,
       neg_flip_tag))

# ================================================================
# verdict + result.json + npz
# ================================================================
tags = ['a_3145_ok']
if not SMOKE:
    tags.append('repro_bit_%d' % n_bit_all)
    tags.append('repro_bit_ok'
                if n_bit_all == 14
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
tags.append('head_dose_%s'
            % dose_shapes['head'])
tags.append('co36full_dose_%s'
            % dose_shapes['co36full'])
tags.append('co50ex_dose_%s'
            % dose_shapes['co50ex'])
tags.append('gbot_dose_%s'
            % dose_shapes['gbot'])
tags.append('tail_dose_%s'
            % dose_shapes['tail'])
tags.append(tail_locus)
tags.append(onset_hist_tag)
tags.append(l39_tag)
tags.append(gap_tag)
tags.append(neg_flip_tag)
tags.append(clean_tag)
tags.append('xphase_ok' if xphase_ok
            else 'xphase_drift')
tags.append('coverage_full')
verdict = '|'.join(tags)
log('VERDICT: %s' % verdict)

result = {
    'phase': 3146,
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
        'res45_sha8': sha45,
        'seal45': SEAL45,
        'xphase_P': xphase_P,
        'xphase_A1': xphase_A1},
    'part_field': {
        'dvec19_sha8': dvec19_sha,
        'dvec19_sha_ok': bool(d19_sha_ok),
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
    'part_cleanclip': {
        'bit_anchors': bit_anchors,
        'windows': {wn: E_res[
            'b4_base_%s' % wn]['chg']
            for wn in WINDOWS},
        'clean_windows': clean_ws,
        'best_window': best_w,
        'chg_joint_best':
            E_res['b4_joint_%s' % best_w]
            ['chg'] if best_w else None,
        'chg_dv29_best':
            E_res['b4_dv29_%s' % best_w]
            ['chg'] if best_w else None,
        'delta_pre': DELTA_PRE45,
        'delta_post2': delta_post2,
        'shrink2': shrink2,
        'clean_tag': clean_tag},
    'part_hist': {
        'dl_base': {str(tok):
                    {str(r): dl_base[tok][r]
                     for r in SPEC_CAP_L}
                    for tok in SPEC_TOKS},
        'dl_hist': {str(tok):
                    {str(r): dl_hist[tok][r]
                     for r in SPEC_CAP_L}
                    for tok in SPEC_TOKS},
        'dl_norm': {str(tok):
                    {str(r): dl_norm[tok][r]
                     for r in (38, 39)}
                    for tok in SPEC_TOKS},
        'wdn_base': {str(r): wdn_base[r]
                     for r in SPEC_CAP_L},
        'wdn_hist': {str(r): wdn_hist[r]
                     for r in SPEC_CAP_L},
        'nrm_base': {str(r): nrm_base[r]
                     for r in SPEC_CAP_L},
        'nrm_hist': {str(r): nrm_hist[r]
                     for r in SPEC_CAP_L},
        'onset_hist': onset_hist,
        'onset_hist_tag': onset_hist_tag,
        'jump_raw': jump_raw,
        'jump_norm': jump_norm,
        'jump_hist': jump_hist,
        'l39_tag': l39_tag,
        'tstar_tok': tstar_tok},
    'part_dose': {
        'shapes': dose_shapes,
        'dvals': {tn: E_res[tn]['chg']
                  for tn, _, _, _
                  in D_TRIALS},
        'head_dose_ratio': head_dose_ratio,
        'tail_dose_ratio': tail_dose_ratio,
        'tail_locus': tail_locus,
        'chg_ttop10_d4': chg_tt,
        'chg_tbot15_d4': chg_tb},
    'part_pcres': {
        'trials': {tn: E_res[tn]['chg']
                   for tn, _, _, _
                   in E_TRIALS},
        'gap_by_dose': {str(k): v for k, v
                        in gap_by_dose.items()},
        'gap_tag': gap_tag,
        'neg_flip_tag': neg_flip_tag,
        'med_fstep_neg': med_fs_neg,
        'neg_fstep_hist': {
            tn: [int(v) for v in
                 fstep_store.get(tn, [])
                 if v >= 0]
            for tn in ('e_pcres38_d0.25_neg',
                       'e_pcres38_d0.5_neg')}},
    }
RF = os.path.join(OUT, 'result.json')
with io.open(RF, 'w',
             encoding='utf-8') as f:
    json.dump(result, f,
              ensure_ascii=False,
              indent=1)
npz_out = {
    'dl_base_131401': np.array(
        [dl_base[131401][r]
         for r in SPEC_CAP_L]),
    'dl_hist_131401': np.array(
        [dl_hist[131401][r]
         for r in SPEC_CAP_L]),
    'dl_norm_131401': np.array(
        [dl_norm[131401][r]
         for r in (38, 39)]),
    'wdn_hist': np.array(
        [wdn_hist[r] for r in SPEC_CAP_L]),
    'nrm_hist': np.array(
        [nrm_hist[r] for r in SPEC_CAP_L]),
    'spec_layers': np.array(SPEC_CAP_L),
    'dvec19_sha': np.array([dvec19_sha]),
    'dose_vals': np.array(
        [E_res[tn]['chg']
         for tn, _, _, _ in D_TRIALS]),
    'dose_names': np.array(
        [tn for tn, _, _, _ in D_TRIALS]),
    'pcres_vals': np.array(
        [E_res[tn]['chg']
         for tn, _, _, _ in E_TRIALS]),
    'gap_by_dose': np.array(
        [gap_by_dose[d]
         for d in PCRES_DOSES]),
    'pc1_res': pc1_res.astype(np.float32)}
np.savez(os.path.join(OUT,
                      'p144_readout.npz'),
         **npz_out)
if not SMOKE:
    try:
        os.remove(CKPTF)
        log('ckpt cleaned (final)')
    except OSError:
        pass
log('result.json + npz written. DONE '
    '(verdict %s)' % verdict)
