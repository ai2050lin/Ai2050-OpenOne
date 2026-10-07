# -*- coding: utf-8 -*-
"""Phase 3145 (Omega-P143): v1-amplification
causal clip (is the pc1-channel downstream
amplification the carrier of joint
blocking?) + pc1-channel token spectrum
(per-layer logit displacement of ' preoc'
-family tokens + allstep-history replay vs
instant injection) + head sign completion
(d_headpos sgn+1) + tail dose curve
(d{2,4}: dose saturation / sign symmetry
under dose) + dh19-residual top-PC causal
injection (@L19/@L38 +/-) + perp@L39
yes/no side decomposition.

Preregistered in 3144 closeout (Omega-
P143):
(1) v1 clip: joint/dv29/base injections
at L29 (P rows, dose 2.0) with v1-projection
clips at CLIP_L {30,33,36,38,39} (joint) +
base+clip@38 / dv29+clip@38 controls ->
shrink(38) = 1 - delta_post/delta_pre
gates v1_causal_confirmed / absent /
dirty; B2 attribution: amp_dv29 vs
amp_pc1contrib from 3144 traj anchors;
(2) token spectrum: teacher-forced 16
diverged rows, captures L20-39 base/inj
(dv_pc1@L29) -> per-layer dlogit of
SPEC_TOKS [131401,134772,23386] -> onset
layer (first med>0.10); instant trial
(pc1 @L29 mode 0) vs allstep 0.203125
ratio gate; pc1-history replay (tf on
pc1-generated prefix + injection) ->
hist_lock / release + readout drop;
(3) head/tail: d_headpos_d1.0 (HEAD25
sgn+1) vs d_head 0.21875 -> sign gate;
tail dose d{2,4} (TAIL25 sgn-1) +
d_tailpos_d2.0 -> monotone/saturating/
flat + symmetry keep/break; d_head/d_gbot
replayed as bit anchors;
(4) residual causal: perp38 top-PC
unit direction x perp_norm_med38 @L19/@L38
(sgn +/-) injections on P rows ->
resid_causal_active (chg>=0.05) / dead;
perp39 yes/no decomposition (med perp@W_yes
- @W_no self-consistency vs perp_wdn39).

Bank shards REUSED from 3138 full run.
Frozen dvecs from 3135 (sha-anchored).
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
NAME = ('omega_p143_v1clip_pc1spec_'
        'headsign_residcausal')
SMOKE = os.environ.get('P3145_SMOKE',
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
D44 = RDIR + r'\phase3144' \
      r'\omega_p142_readouttraj_' \
      r'co36sign_unembed_d19resid'
MDIR_G = os.path.join(ROOT, 'models', 'hf',
                      'glm4-9b-chat-hf')
OUT = os.path.join(RDIR, 'phase3145', NAME)
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
CKPTF = os.path.join(OUT, 'p143_ckpt.pkl')
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
CLIP_L = (30, 33, 36, 38, 39)
CLIP_MAIN = 38
SHRINK_GATE = 0.5
BASE_CLIP_DIRTY = 0.05
ONSET_TOL = 0.10
HIST_LOCK_FRAC = 0.5
SIGN_TOL = 0.03
DOSE_T = (2.0, 4.0)
PCRES_DOSE = 1.0
RESID_GATE = 0.05
READOUT_DROP = 0.5
Z35_TOL = 0.08
FIELD_SELF_COS = 0.99
SHARE38_TOL = 0.02
WDN39_TOL = 0.15
ALIGN_TOL = 0.03
PNORM38_TOL = 1.0
SPEC_TOKS = [131401, 134772, 23386]
RNG_SEED = 3145

# ---------------- frozen 3144 anchors --
EXP_V44 = ('a_3143_ok|repro_bit_6|'
           'repro_bit_ok|dvec19_repro_6a0332|'
           'field_self_ok|z35_cos17_ok|'
           'traj_gradual|pc1_channel_format|'
           'tail_neg_confirmed|tail_sign_sym|'
           'identity_unembed_orthogonal|'
           'd19resid_readout|xphase_ok|'
           'coverage_full')
RES44_SHA = 'def947d8'
SEAL44 = 'bcd6fc5e'
XPHASE44 = 1.0
DVEC19_SHA = '6a0332a6'
MEDNORM19 = 5.643608093261719
PC1_SIGN_44 = 1
# 3142/3144 bit anchors to replay (5)
BIT5 = {'b_pc1_l29_d2.0': 0.203125,
        'b_dvec29_l29_d2.0': 0.640625,
        'b_joint_l29_d2.0': 0.4765625,
        'd_head_d1.0': 0.21875,
        'd_gbot_d1.0': 0.1171875}
# traj anchors (3144 result.json)
JV1_29 = -10.391924
JV1_39 = -69.508387
DV1_29 = -20.869472
DV1_39 = -87.890003
PV1_29 = 10.480455
PV1_38 = 16.066369
PV1_39 = 5.685421
JNORM_39 = 212.556174
JWDN_39 = 4.737774
GAP_38 = -1.806777
# tokdec anchors
DLOGIT_STAR_44 = 0.596221
WDN39_B3_44 = 0.755293
TOP1D_44 = [23386, 131401, 131401, 131401,
            131401, 134772, 131401, 131401]
# co36sign anchors
D_TAIL_44 = 0.09375
D_TAILPOS_44 = 0.1015625
D_CO36FULL_44 = 0.140625
D_CO50EX_44 = 0.421875
# resid anchors (3144 part_resid)
PSHARE_38 = 0.640474
PWDN_38 = 1.576030
PWDN_39 = 2.495525
DH19WDN_39 = 1.404258
PNORM_38 = 55.263357
RA_V1_44 = 0.228672
RA_WDN_44 = 0.000405
RA_DV29_44 = 0.090669
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
LEDGER_N = 281

# ---------------- seal ------------------
SEAL = {
    'phase': 3145,
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
        'CLIP_L': list(CLIP_L),
        'CLIP_MAIN': CLIP_MAIN,
        'SHRINK_GATE': SHRINK_GATE,
        'BASE_CLIP_DIRTY': BASE_CLIP_DIRTY,
        'ONSET_TOL': ONSET_TOL,
        'HIST_LOCK_FRAC': HIST_LOCK_FRAC,
        'SIGN_TOL': SIGN_TOL,
        'DOSE_T': list(DOSE_T),
        'PCRES_DOSE': PCRES_DOSE,
        'RESID_GATE': RESID_GATE,
        'READOUT_DROP': READOUT_DROP,
        'Z35_TOL': Z35_TOL,
        'FIELD_SELF_COS': FIELD_SELF_COS,
        'SPEC_TOKS': SPEC_TOKS,
        'RNG_SEED': RNG_SEED,
        'BANK_REUSE_3138': True},
    'anchors': {
        'res44_verdict': EXP_V44,
        'res44_sha8': RES44_SHA,
        'seal44': SEAL44,
        'xphase44': XPHASE44,
        'dvec19_sha8': DVEC19_SHA,
        'mednorm19': MEDNORM19,
        'pc1_sign_44': PC1_SIGN_44,
        'bit5': BIT5,
        'jv1_29': JV1_29, 'jv1_39': JV1_39,
        'dv1_29': DV1_29, 'dv1_39': DV1_39,
        'pv1_29': PV1_29, 'pv1_38': PV1_38,
        'pv1_39': PV1_39,
        'jnorm_39': JNORM_39,
        'jwdn_39': JWDN_39,
        'gap_38': GAP_38,
        'dlogit_star_44': DLOGIT_STAR_44,
        'wdn39_b3_44': WDN39_B3_44,
        'top1d_44': TOP1D_44,
        'd_tail_44': D_TAIL_44,
        'd_tailpos_44': D_TAILPOS_44,
        'd_co36full_44': D_CO36FULL_44,
        'd_co50ex_44': D_CO50EX_44,
        'pshare_38': PSHARE_38,
        'pwdn_38': PWDN_38,
        'pwdn_39': PWDN_39,
        'dh19wdn_39': DH19WDN_39,
        'pnorm_38': PNORM_38,
        'ra_v1_44': RA_V1_44,
        'ra_wdn_44': RA_WDN_44,
        'ra_dv29_44': RA_DV29_44,
        'delta_l17': DELTA_L17,
        'high25_43': HIGH25_43,
        'low25_43': LOW25_43,
        'g_top25': G_TOP25,
        'g_bot25': G_BOT25,
        'dvec_sha8': {str(k): v for k, v
                      in DVEC_SHA.items()},
        'ledger_n': LEDGER_N},
    'prereg': ('3144 closeout Omega-P143: '
               '(1) v1 clip causal: joint/'
               'dv29/base @L29 dose2.0 P rows '
               '+ v1-projection clips @CLIP_L '
               '{30,33,36,38,39} (joint) + '
               'base/dv29+clip@38 controls -> '
               'shrink(38)>=0.5 & base_clip '
               'clean -> v1_causal_confirmed; '
               'B2 attribution amp_dv29 vs '
               'amp_pc1contrib from 3144 '
               'traj; (2) pc1 token spectrum: '
               'tf 16 diverged rows captures '
               'L20-39 base/inj -> per-layer '
               'dlogit SPEC_TOKS onset (med>'
               '0.10); instant (mode 0) vs '
               'allstep ratio gate; pc1-hist '
               'replay -> hist_lock/release + '
               'readout drop; (3) d_headpos '
               'sgn+1 vs d_head sign gate; '
               'tail dose d{2,4} + tailpos_d2 '
               '-> mono/sat/flat + sym keep/'
               'break; d_head/d_gbot replay '
               'bit anchors (5 total); '
               '(4) perp38 top-PC x '
               'perp_norm_med38 @L19/@L38 +/- '
               '-> resid_causal_active/dead; '
               'perp39 yes/no decomposition '
               'self-consistency. Bank reused '
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
# PART A: 3144 link asserts
# ================================================================
log('== PART A: link asserts ==')
res44 = json.load(io.open(
    D44 + r'\result.json', encoding='utf-8'))
assert res44['smoke'] is False
V44 = res44['verdict']
assert V44 == EXP_V44, V44
raw44 = io.open(
    D44 + r'\result.json', 'rb').read()
sha44 = hashlib.sha256(raw44).hexdigest()[:8]
assert sha44 == RES44_SHA, sha44
assert str(res44['seal_sha8']) == SEAL44
pa44 = res44['part_a']
assert abs(float(pa44['xphase_P'])
           - XPHASE44) < 1e-12
assert abs(float(pa44['xphase_A1'])
           - XPHASE44) < 1e-12
pt44 = res44['part_traj']
assert int(pt44['pc1_sign']) == PC1_SIGN_44
for tn, v in BIT5.items():
    if tn in pt44['bit_anchors_3142']:
        assert abs(float(
            pt44['bit_anchors_3142'][tn]
            ['got']) - v) < 1e-9, tn
ptj = pt44['traj']


def _tv(cn, r, f):
    return float(ptj[cn][str(r)][f])


assert abs(_tv('joint', 29, 'v1')
           - JV1_29) < 1e-5
assert abs(_tv('joint', 39, 'v1')
           - JV1_39) < 1e-5
assert abs(_tv('dv29', 29, 'v1')
           - DV1_29) < 1e-5
assert abs(_tv('dv29', 39, 'v1')
           - DV1_39) < 1e-5
assert abs(_tv('pc1', 29, 'v1')
           - PV1_29) < 1e-5
assert abs(_tv('pc1', 38, 'v1')
           - PV1_38) < 1e-5
assert abs(_tv('pc1', 39, 'v1')
           - PV1_39) < 1e-5
assert abs(_tv('joint', 39, 'norm')
           - JNORM_39) < 1e-4
assert abs(_tv('joint', 39, 'wdn')
           - JWDN_39) < 1e-5
assert abs(float(pt44['gap']['38'])
           - GAP_38) < 1e-5
assert pt44['traj_tag'] == 'traj_gradual'
ptd44 = res44['part_tokdec']
assert ptd44['pc1_channel'] == \
    'pc1_channel_format'
b3 = ptd44['b3']
assert abs(float(b3['med_dlogit_star'])
           - DLOGIT_STAR_44) < 1e-5
assert abs(float(b3['med_dh_wdn39'])
           - WDN39_B3_44) < 1e-5
assert [int(t) for t
        in b3['top1d_token_ids']] == TOP1D_44
pcs44 = res44['part_co36sign']
assert abs(float(pcs44['d_tail'])
           - D_TAIL_44) < 1e-9
assert abs(float(pcs44['d_tailpos'])
           - D_TAILPOS_44) < 1e-9
assert abs(float(pcs44['d_co36full'])
           - D_CO36FULL_44) < 1e-9
assert abs(float(pcs44['d_co50ex'])
           - D_CO50EX_44) < 1e-9
assert pcs44['tail_tag'] == \
    'tail_neg_confirmed'
assert pcs44['sign_tag'] == 'tail_sign_sym'
for tn, v in (('d_head_d1.0', 0.21875),
              ('d_gbot_d1.0', 0.1171875),
              ('d_co36full_d1.0',
               D_CO36FULL_44)):
    assert abs(float(
        pcs44['bit_anchors'][tn]['got'])
        - v) < 1e-9, tn
pr44 = res44['part_resid']
r38 = pr44['resid']['38']
r39 = pr44['resid']['39']
assert abs(float(r38['perp_share'])
           - PSHARE_38) < 1e-5
assert abs(float(r38['perp_wdn_med'])
           - PWDN_38) < 1e-5
assert abs(float(r38['perp_norm_med'])
           - PNORM_38) < 1e-4
assert abs(float(r39['perp_wdn_med'])
           - PWDN_39) < 1e-5
assert abs(float(r39['dh19_wdn_med'])
           - DH19WDN_39) < 1e-5
ra = pr44['resid_align']
assert abs(float(ra['pc1_res_vs_v1'])
           - RA_V1_44) < 1e-5
assert abs(float(ra['pc1_res_vs_wdn'])
           - RA_WDN_44) < 1e-5
assert abs(float(ra['pc1_res_vs_dv29mean'])
           - RA_DV29_44) < 1e-5
assert pr44['resid_tag'] == \
    'd19resid_readout'
pf44 = res44['part_field']
assert str(pf44['dvec19_sha8']) == \
    DVEC19_SHA
assert abs(float(pf44['dvec19_medn'])
           - MEDNORM19) < 1e-9
assert pf44['field_self_ok'] is True
assert pf44['z35_ok'] is True
log('A hard asserts ok (3144 sha8 %s seal '
    '%s: xphase/dvec19/traj/tokdec/'
    'co36sign/resid anchors; 6-bit chain '
    'verified)' % (sha44, SEAL44))

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
    (il, direction np[HID]) v1-projection
    removal hooks on layer outputs."""
    ids_p, mask_p = _pad_batch(pids_list)
    hooks = []
    if clip:
        for (il, direction) in clip:
            lyr = model_g.model.layers[il]
            v_t = torch.as_tensor(
                np.asarray(direction,
                           dtype=np.float32),
                device='cuda')
            v_t = v_t / (v_t.norm() + 1e-12)

            def _clip(mod, inp, out,
                      _v=v_t):
                o2 = out[0] \
                    if isinstance(out,
                                  tuple) \
                    else out
                h = o2[:, -1, :].float()
                proj = (h @ _v) \
                    .unsqueeze(-1)
                o2[:, -1, :] = (
                    h - proj
                    * _v.unsqueeze(0)) \
                    .to(o2.dtype)
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
                  'tfbase_pc1')
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
# PART B: v1 clip causal
# ================================================================
log('== PART B: v1 clip causal ==')
mn29 = PC[29]['mnorm']
dv_pc1_pos = np.tile(
    (PC[29]['V'][0] * mn29 * 2.0)[None, :],
    (NCAP, 1)).astype(np.float32)
dv_pc1_neg = -dv_pc1_pos
dv_pc1 = dv_pc1_pos
dv_dv29 = (dvec[29][:NCAP] * 2.0) \
    .astype(np.float32)
E_res = {}


def _run_vec_trial(tname, il, dv_batch,
                   scale, mode, rows,
                   base12, clip=None):
    _K = CK['data'].get(tname)
    if _K is not None:
        E_res[tname] = _K['res']
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
                      scale, mode)],
            clip=clip))
    chg_l, first_l, _fs = trial_metrics(
        gen_l, base12)
    E_res[tname] = {'chg': chg_l,
                    'first': int(first_l),
                    'gens': gen_l}
    log('%s: chg=%.4f first=%d'
        % (tname, chg_l, first_l))
    ck_save(tname, {'res': E_res[tname]})


# B1: replay 3 generation bit anchors
_run_vec_trial('b_pc1_l29_d2.0', 29,
               dv_pc1_pos, 1.0, 'allstep',
               rows_scan, base12_P)
pc1_sign = None
if not SMOKE and abs(
        E_res['b_pc1_l29_d2.0']['chg']
        - BIT5['b_pc1_l29_d2.0']) >= 1e-9:
    _run_vec_trial('b_pc1neg_l29_d2.0',
                   29, dv_pc1_neg, 1.0,
                   'allstep', rows_scan,
                   base12_P)
    if abs(E_res['b_pc1neg_l29_d2.0']
           ['chg']
           - BIT5['b_pc1_l29_d2.0']) < 1e-9:
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
    chg_l, first_l, _fs = trial_metrics(
        gen_l, base12_P)
    E_res['b_joint_l29_d2.0'] = {
        'chg': chg_l, 'first': int(first_l),
        'gens': gen_l}
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
        m = abs(got - BIT5[tn]) < 1e-9
        n_bit_all += int(m)
        bit_anchors[tn] = {
            'got': got, 'want': BIT5[tn],
            'match': bool(m)}
    log('B1 3142 repro: %d/3 bit-match'
        % n_bit_all)
else:
    log('B1 3142 repro SKIPPED (smoke)')

# B2: v1 amplification attribution
# (pure computation from 3144 anchors)
amp_joint = JV1_39 - JV1_29
amp_dv29 = DV1_39 - DV1_29
amp_pc1c = (JV1_39 - DV1_39) \
    - (JV1_29 - DV1_29)
amp_ratio = abs(amp_dv29) / max(
    abs(amp_pc1c), 1e-9)
amp_tag = ('v1_amp_dv29dom'
           if amp_ratio > 3.0
           else 'v1_amp_pc1dom')
log('B2-SOFT: amp_joint %.2f amp_dv29 '
    '%.2f amp_pc1contrib %+.2f ratio '
    '|dv29/pc1c| %.2f -> %s (3144 '
    'attribution refined: dv29 carrier '
    'own downstream dynamics dominate '
    'the v1 growth)' % (amp_joint,
                        amp_dv29,
                        amp_pc1c, amp_ratio,
                        amp_tag))

# B3: v1 clip generation trials
chg_dv29 = E_res['b_dvec29_l29_d2.0']['chg']
chg_joint = E_res['b_joint_l29_d2.0']['chg']
delta_pre = chg_dv29 - chg_joint
clip_res = {}
CLIP_TRIALS = ([('jc_l%02d' % l,
                 [(l, v1.astype(np.float32))])
                for l in CLIP_L]
               + [('bc_l%02d' % CLIP_MAIN,
                   [(CLIP_MAIN,
                     v1.astype(np.float32))]),
                  ('dc_l%02d' % CLIP_MAIN,
                   [(CLIP_MAIN,
                     v1.astype(np.float32))])])
for (tname, clip_spec) in CLIP_TRIALS:
    _K = CK['data'].get('clip_' + tname)
    if _K is not None:
        clip_res[tname] = _K['res']
        log('clip %s RESUMED chg=%.4f'
            % (tname,
               clip_res[tname]['chg']))
        continue
    if tname.startswith('jc_'):
        _gen = []
        for b0 in range(0, NCAP,
                        GEN_BATCH):
            batch = rows_scan[b0:b0
                              + GEN_BATCH]
            _gen.extend(gen_batch_g2(
                batch,
                inj_vec=[(29, dv_pc1[b0:b0
                                    + len(batch)],
                          1.0, 'allstep'),
                         (29, dv_dv29[b0:b0
                                      + len(batch)],
                          1.0, 'allstep')],
                clip=clip_spec))
    elif tname.startswith('dc_'):
        _gen = []
        for b0 in range(0, NCAP,
                        GEN_BATCH):
            batch = rows_scan[
                b0:b0 + GEN_BATCH]
            _gen.extend(gen_batch_g2(
                batch,
                inj_vec=[(29, dv_dv29[b0:b0
                                      + len(batch)],
                          1.0, 'allstep')],
                clip=clip_spec))
    else:
        _gen = []
        for b0 in range(0, NCAP,
                        GEN_BATCH):
            batch = rows_scan[
                b0:b0 + GEN_BATCH]
            _gen.extend(gen_batch_g2(
                batch,
                clip=clip_spec))
    chg_l, first_l, _fs = trial_metrics(
        _gen, base12_P)
    clip_res[tname] = {'chg': chg_l,
                       'first': int(first_l)}
    log('clip %s: chg=%.4f first=%d'
        % (tname, chg_l, first_l))
    ck_save('clip_' + tname,
            {'res': clip_res[tname]})
_m38 = CLIP_MAIN
chg_bc = clip_res['bc_l%02d' % _m38]['chg']
chg_dc = clip_res['dc_l%02d' % _m38]['chg']
chg_jc = clip_res['jc_l%02d' % _m38]['chg']
delta_post = chg_dc - chg_jc
shrink_main = (1.0 - delta_post
               / delta_pre) \
    if abs(delta_pre) > 1e-9 else 0.0
shrink_spec = {}
for l in CLIP_L:
    dp = chg_dc - clip_res[
        'jc_l%02d' % l]['chg']
    shrink_spec[l] = (1.0 - dp
                      / delta_pre) \
        if abs(delta_pre) > 1e-9 else 0.0
clip_dirty = chg_bc >= BASE_CLIP_DIRTY
if clip_dirty:
    v1_tag = 'v1_causal_dirty'
elif shrink_main >= SHRINK_GATE:
    v1_tag = 'v1_causal_confirmed'
else:
    v1_tag = 'v1_causal_absent'
log('B3-SOFT: shrink spec %s'
    % json.dumps({str(l): round(v, 3)
                  for l, v
                  in shrink_spec.items()}))
log('B3-GATE: base_clip %.4f (dirty gate '
    '%.2f) dv29+clip %.4f joint+clip %.4f '
    'delta_pre %.4f delta_post %.4f '
    'shrink(38) %.3f -> %s'
    % (chg_bc, BASE_CLIP_DIRTY, chg_dc,
       chg_jc, delta_pre, delta_post,
       shrink_main, v1_tag))

# ================================================================
# PART T: pc1 token spectrum + instant
# vs allstep + history replay
# ================================================================
log('== PART T: pc1 spectrum ==')
# T1: instant injection (mode 0 = prompt
# forward only)
_run_vec_trial('t_pc1_l29_inst', 29,
               dv_pc1, 1.0, 0,
               rows_scan, base12_P)
chg_inst = E_res['t_pc1_l29_inst']['chg']
chg_all = E_res['b_pc1_l29_d2.0']['chg']
inst_ratio = chg_inst / max(chg_all, 1e-9)
inst_tag = ('pc1_history_dependent'
            if inst_ratio < 0.5
            else 'pc1_step_independent')
log('T1-GATE: instant chg %.4f vs '
    'allstep %.4f ratio %.3f -> %s'
    % (chg_inst, chg_all, inst_ratio,
       inst_tag))
# interfering-token classification for
# the instant trial (compare 3144 allstep
# format 17/other 9)
gens_inst = E_res['t_pc1_l29_inst']['gens']
counts_i = {'answer': 0, 'format': 0,
            'other': 0, 'none': 0}
for j in range(len(gens_inst)):
    g12 = pad12(gens_inst[j])
    if g12 == base12_P[j]:
        counts_i['none'] += 1
        continue
    for t in range(N_NEW):
        if g12[t] != base12_P[j][t]:
            counts_i[
                'answer'
                if g12[t] in (YES_G, NO_G)
                else 'format'
                if g12[t] == DOT_G
                else 'other'] += 1
            break
log('T1-SOFT: instant tokcls %s'
    % json.dumps(counts_i))
# T2: teacher-forced per-layer captures
gens_pc1 = E_res['b_pc1_l29_d2.0']['gens']
div_rows = []
for j in range(NCAP):
    g12 = pad12(gens_pc1[j])
    if g12 != base12_P[j]:
        div_rows.append(j)
    if len(div_rows) >= SPEC_ROWS:
        break
TN = len(div_rows)
log('T2: %d diverged rows (cap %d)'
    % (TN, SPEC_ROWS))
tf_rows = []
tstar_tok = []
tf_pc1_rows = []
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
spec_l = {}
for tok in SPEC_TOKS:
    w_row = WUG[tok].to(torch.float32) \
        .cpu().numpy().astype(np.float64)
    vals = {}
    for r in SPEC_CAP_L:
        dh = (h_tf_inj[r].astype(np.float64)
              - h_tf_base[r]
              .astype(np.float64))
        vals[r] = float(np.median(
            dh @ w_row))
    spec_l[tok] = vals
onset_l = None
for r in SPEC_CAP_L:
    if spec_l[SPEC_TOKS[0]][r] > ONSET_TOL:
        onset_l = r
        break
onset_tag = ('pc1_onset_l%02d' % onset_l
             if onset_l is not None
             else 'pc1_onset_none')
log('T2-SOFT: dlogit spec %s'
    % json.dumps({
        str(tok): {str(r): round(v, 3)
                   for r, v
                   in spec_l[tok].items()}
        for tok in SPEC_TOKS}))
log('T2-GATE: onset (first L20+ with '
    'med dlogit(%d) > %.2f) -> %s'
    % (SPEC_TOKS[0], ONSET_TOL, onset_tag))
# T3: pc1-history replay
_K3 = CK['data'].get('tfbase_pc1')
if _K3 is not None:
    h_tf_pc1 = {int(k):
                v.astype(np.float32)
                for k, v in
                _K3['h'].items()}
    log('tfbase_pc1 RESUMED')
else:
    h_tf_pc1 = capture_states(
        tf_pc1_rows, SPEC_CAP_L)
    ck_save('tfbase_pc1', {
        'h': {str(l): h_tf_pc1[l]
              .astype(np.float16)
              for l in SPEC_CAP_L}})
h_tf_pc1_inj = capture_states_inject2(
    tf_pc1_rows, [39],
    [(29, dv_pc1[:TN], 1.0)])[39]
dh_base_hist = (h_tf_inj[39].astype(
    np.float64)
    - h_tf_base[39].astype(np.float64))
dh_pc1_hist = (h_tf_pc1_inj.astype(
    np.float64)
    - h_tf_pc1[39].astype(np.float64))
wdn_bh = dh_base_hist @ w_dn_g.astype(
    np.float64)
wdn_ph = dh_pc1_hist @ w_dn_g.astype(
    np.float64)
med_wdn_bh = float(np.median(wdn_bh))
med_wdn_ph = float(np.median(wdn_ph))
drop_tag = ('pc1_replay_readout_drop'
            if med_wdn_bh - med_wdn_ph
            > READOUT_DROP
            else 'pc1_replay_readout_sim')


def _chunked_top1(DH):
    n = DH.shape[0]
    W = WUG.detach()
    best_val = np.full(n, -np.inf)
    am = np.zeros(n, dtype=np.int64)
    CH = 16384
    for s0 in range(0, VOCAB, CH):
        s1 = min(s0 + CH, VOCAB)
        Wc = W[s0:s1].to(torch.float32) \
            .cpu().numpy() \
            .astype(np.float64)
        LG = DH @ Wc.T
        mx = LG.max(axis=1)
        upd = mx > best_val
        am[upd] = s0 + np.argmax(
            LG, axis=1)[upd]
        best_val[upd] = mx[upd]
    return am


am_bh = _chunked_top1(dh_base_hist)
am_ph = _chunked_top1(dh_pc1_hist)
spec_set = set(SPEC_TOKS)
lock_bh = float(np.mean(
    [int(int(t) in spec_set)
     for t in am_bh]))
lock_ph = float(np.mean(
    [int(int(t) in spec_set)
     for t in am_ph]))
hist_tag = ('pc1_hist_lock'
            if lock_ph >= HIST_LOCK_FRAC
            else 'pc1_hist_release')
log('T3-SOFT: wdn39 base-hist %.3f vs '
    'pc1-hist %.3f (drop gate %.2f) '
    'top1d in-spec base-hist %.2f pc1-hist '
    '%.2f -> %s / %s'
    % (med_wdn_bh, med_wdn_ph,
       READOUT_DROP, lock_bh, lock_ph,
       drop_tag, hist_tag))

# ================================================================
# PART D: head sign + tail dose
# ================================================================
log('== PART D: head sign + tail dose ==')
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
GB25 = list(G_BOT25)
D_TRIALS = [
    ('d_head_d1.0', HEAD25, -1, DOSE_C),
    ('d_gbot_d1.0', GB25, -1, DOSE_C),
    ('d_headpos_d1.0', HEAD25, 1, DOSE_C),
    ('d_tail_d2.0', TAIL25, -1, 2.0),
    ('d_tail_d4.0', TAIL25, -1, 4.0),
    ('d_tailpos_d2.0', TAIL25, 1, 2.0)]
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
    chg_l, first_l, _fs = trial_metrics(
        gen_l, base12_A1)
    E_res[tname] = {'chg': chg_l,
                    'first': int(first_l)}
    log('%s: chg=%.4f first=%d'
        % (tname, chg_l, first_l))
    ck_save(tname, {'res': E_res[tname]})
if not SMOKE:
    for tn in ('d_head_d1.0',
               'd_gbot_d1.0'):
        got = E_res[tn]['chg']
        m = abs(got - BIT5[tn]) < 1e-9
        n_bit_all += int(m)
        bit_anchors[tn] = {
            'got': got, 'want': BIT5[tn],
            'match': bool(m)}
    log('D replay: +2 bit anchors (total '
        '%d/5)' % n_bit_all)
d_head = E_res['d_head_d1.0']['chg']
d_gbot = E_res['d_gbot_d1.0']['chg']
d_headpos = E_res['d_headpos_d1.0']['chg']
d_t1 = D_TAIL_44
d_t2 = E_res['d_tail_d2.0']['chg']
d_t4 = E_res['d_tail_d4.0']['chg']
d_tp2 = E_res['d_tailpos_d2.0']['chg']
head_asym = abs(d_headpos - d_head) \
    > SIGN_TOL
head_tag = ('head_sign_asym'
            if head_asym
            else 'head_sign_sym')
if d_t2 > d_t1 + 0.02 \
        and d_t4 > d_t2 + 0.02:
    dose_tag = 'tail_dose_mono'
elif abs(d_t4 - d_t2) < 0.02 \
        and d_t2 > d_t1 + 0.02:
    dose_tag = 'tail_dose_saturating'
elif d_t2 < d_t1 + 0.02 \
        and d_t4 < d_t1 + 0.02:
    dose_tag = 'tail_dose_flat'
else:
    dose_tag = 'tail_dose_mixed'
sym_d2 = abs(d_tp2 - d_t2) < SIGN_TOL
sym_tag = ('tail_sym_keep'
           if sym_d2 else 'tail_sym_break')
log('D-GATE: head %.4f headpos %.4f '
    '(tol %.2f) -> %s | tail d1 %.4f d2 '
    '%.4f d4 %.4f tailpos_d2 %.4f -> %s '
    '/ %s'
    % (d_head, d_headpos, SIGN_TOL,
       head_tag, d_t1, d_t2, d_t4, d_tp2,
       dose_tag, sym_tag))

# ================================================================
# PART E: residual top-PC causal injection
# ================================================================
log('== PART E: resid causal ==')
dv_pcres = (pc1_res
            / max(np.linalg.norm(pc1_res),
                  1e-12)
            * pnorm38 * PCRES_DOSE) \
    .astype(np.float32)
dv_pcres_tile = np.tile(
    dv_pcres[None, :], (NCAP, 1))
E_TRIALS = [('e_pcres_l19_pos', D19_L,
             1.0),
            ('e_pcres_l19_neg', D19_L,
             -1.0),
            ('e_pcres_l38_pos', 38, 1.0),
            ('e_pcres_l38_neg', 38, -1.0)]
for (tname, il, sg) in E_TRIALS:
    _K = CK['data'].get(tname)
    if _K is not None:
        E_res[tname] = _K['res']
        log('%s RESUMED chg=%.4f'
            % (tname, E_res[tname]['chg']))
        continue
    gen_l = []
    for b0 in range(0, NCAP, GEN_BATCH):
        batch = rows_scan[b0:b0 + GEN_BATCH]
        gen_l.extend(gen_batch_g2(
            batch,
            inj_vec=[(il,
                      sg * dv_pcres_tile[
                          b0:b0 + len(batch)],
                      1.0, 'allstep')]))
    chg_l, first_l, _fs = trial_metrics(
        gen_l, base12_P)
    E_res[tname] = {'chg': chg_l,
                    'first': int(first_l)}
    log('%s: chg=%.4f first=%d'
        % (tname, chg_l, first_l))
    ck_save(tname, {'res': E_res[tname]})
e_max = max(E_res[t]['chg']
            for t, _, _ in E_TRIALS)
resid_causal_tag = (
    'resid_causal_active'
    if e_max >= RESID_GATE
    else 'resid_causal_dead')
log('E-GATE: pcres injections chg %s '
    '(gate %.2f) -> %s'
    % (json.dumps({t: round(
        E_res[t]['chg'], 4)
        for t, _, _ in E_TRIALS}),
       RESID_GATE, resid_causal_tag))
# yes/no decomposition of perp39
w_yes = WUG[YES_G].to(torch.float32) \
    .cpu().numpy().astype(np.float64)
w_no = WUG[NO_G].to(torch.float32) \
    .cpu().numpy().astype(np.float64)
dyes39 = float(np.median(
    perp39 @ w_yes))
dno39 = float(np.median(
    perp39 @ w_no))
wdn_check = dyes39 - dno39
if dyes39 > 0 and dno39 < 0:
    side_tag = 'answer_side_yes'
elif dyes39 < 0 and dno39 > 0:
    side_tag = 'answer_side_no'
elif abs(dyes39) > abs(dno39):
    side_tag = 'answer_side_yes_dom'
else:
    side_tag = 'answer_side_no_dom'
log('E-SOFT: perp39 yes/no decomposition: '
    'dyes %+.3f dno %+.3f (diff %.3f vs '
    'perp_wdn39 %.3f) -> %s'
    % (dyes39, dno39, wdn_check, wdn_p39,
       side_tag))

# ================================================================
# verdict + result.json + npz
# ================================================================
tags = ['a_3144_ok']
if not SMOKE:
    tags.append('repro_bit_%d' % n_bit_all)
    tags.append('repro_bit_ok'
                if n_bit_all == 5
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
tags.append(amp_tag)
tags.append(v1_tag)
tags.append(onset_tag)
tags.append(inst_tag)
tags.append(hist_tag)
tags.append(drop_tag)
tags.append(head_tag)
tags.append(dose_tag)
tags.append(sym_tag)
tags.append(resid_causal_tag)
tags.append(side_tag)
tags.append('xphase_ok' if xphase_ok
            else 'xphase_drift')
tags.append('coverage_full')
verdict = '|'.join(tags)
log('VERDICT: %s' % verdict)

result = {
    'phase': 3145,
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
        'res44_sha8': sha44,
        'seal44': SEAL44,
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
    'part_v1clip': {
        'amp_joint': amp_joint,
        'amp_dv29': amp_dv29,
        'amp_pc1contrib': amp_pc1c,
        'amp_ratio': amp_ratio,
        'amp_tag': amp_tag,
        'bit_anchors': bit_anchors,
        'chg_dv29': chg_dv29,
        'chg_joint': chg_joint,
        'delta_pre': delta_pre,
        'clip_res': {k: v['chg']
                     for k, v
                     in clip_res.items()},
        'delta_post': delta_post,
        'shrink_main': shrink_main,
        'shrink_spec': {str(k): v for k, v
                        in shrink_spec.items()},
        'chg_base_clip': chg_bc,
        'v1_tag': v1_tag},
    'part_spec': {
        'tokcls_inst': counts_i,
        'chg_inst': chg_inst,
        'chg_allstep': chg_all,
        'inst_ratio': inst_ratio,
        'inst_tag': inst_tag,
        'spec': {str(tok):
                 {str(r): spec_l[tok][r]
                  for r in SPEC_CAP_L}
                 for tok in SPEC_TOKS},
        'onset_layer': onset_l,
        'onset_tag': onset_tag,
        'med_wdn39_base_hist': med_wdn_bh,
        'med_wdn39_pc1_hist': med_wdn_ph,
        'lock_bh': lock_bh,
        'lock_ph': lock_ph,
        'hist_tag': hist_tag,
        'drop_tag': drop_tag,
        'tstar_tok': tstar_tok},
    'part_headtail': {
        'd_head': d_head,
        'd_headpos': d_headpos,
        'd_gbot': d_gbot,
        'd_tail_d1': d_t1,
        'd_tail_d2': d_t2,
        'd_tail_d4': d_t4,
        'd_tailpos_d2': d_tp2,
        'head_tag': head_tag,
        'dose_tag': dose_tag,
        'sym_tag': sym_tag},
    'part_residcausal': {
        'trials': {t: E_res[t]['chg']
                   for t, _, _ in E_TRIALS},
        'e_max': e_max,
        'causal_tag': resid_causal_tag,
        'dyes39': dyes39,
        'dno39': dno39,
        'wdn_check': wdn_check,
        'side_tag': side_tag}}
RF = os.path.join(OUT, 'result.json')
with io.open(RF, 'w',
             encoding='utf-8') as f:
    json.dump(result, f,
              ensure_ascii=False,
              indent=1)
spec_out = {}
for tok in SPEC_TOKS:
    spec_out['spec_%d' % tok] = np.array(
        [spec_l[tok][r]
         for r in SPEC_CAP_L])
npz_out = {
    'spec_layers': np.array(SPEC_CAP_L),
    'shrink_spec': np.array(
        [shrink_spec[l] for l in CLIP_L]),
    'clip_layers': np.array(CLIP_L),
    'share38_arr': share38_arr.astype(
        np.float32),
    'perp_wdn_row': (perp39
                     @ w_dn_g.astype(
                         np.float64)
                     ).astype(np.float32),
    'dvec19_sha': np.array([dvec19_sha]),
    'head_tail': np.array(
        [d_head, d_headpos, d_t1, d_t2,
         d_t4, d_tp2]),
    'pcres_chg': np.array(
        [E_res[t]['chg']
         for t, _, _ in E_TRIALS]),
    'pc1_res': pc1_res.astype(np.float32)}
npz_out.update(spec_out)
np.savez(os.path.join(OUT,
                      'p143_readout.npz'),
         **npz_out)
if not SMOKE:
    try:
        os.remove(CKPTF)
        log('ckpt cleaned (final)')
    except OSError:
        pass
log('result.json + npz written. DONE '
    '(verdict %s)' % verdict)
