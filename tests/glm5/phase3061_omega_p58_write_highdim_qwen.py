# -*- coding: utf-8 -*-
# Phase 3061 - Omega-P58: structure of the
# write-side high-dim family. 3060 T5 showed
# the d_tan family (24 x 2560, gamma-weighted
# post-norm diff directions) has d_eff = 8.74
# while the readout preimage family has
# d_eff = 1.34, and the two PC1s are NOT the
# same axis (|cos| = 0.142). Question: WHAT
# carries the write-side high dimensionality -
# body identity, prefix condition, or per-pair
# residual? T2 (MAIN structure, offline):
# two-way variance decomposition - d_k = d_bar
# + (B_b - d_bar) + (C_c - d_bar) + R_kb with
# B_b body means, C_c prefix means; shares
# SS_body / SS_prefix / SS_resid of the total
# energy. T3 (offline): per-body sub-family
# (3 rows) and per-prefix sub-family (8 rows)
# d_eff + between-body mean direction
# similarity. T4 (MAIN readout relation,
# offline): project each d_tan onto the
# readout PC1 (z60) -> residual family
# d_eff_resid; also cos(D_TAN[k], PC1) per
# pair. T5 (offline): channel support - SVD
# of D_TAN, per-PC energy in S16/S64 vs
# permutation nulls (seeds 9986/9987);
# top-256 |PC1_dtan| overlap with S16/TOP64
# vs nulls (seed 9988); cos(D_TAN[k],
# Vmat[k]) descriptive (same channel space).
# verdict (single branch): largest share
# among (SS_prefix, SS_body, SS_resid):
# prefix -> write_highdim_prefix_qwen;
# body -> write_highdim_body_qwen; resid ->
# write_highdim_residual_qwen; else (no
# share > 0.4) -> write_highdim_mixed_qwen.
# anchors: a205 source seals 3044-3060; a206
# TT recompute bit 0.0 vs z48; a207 gamma
# stats bit 0.0 vs z55; a208 TOP64/FLAT64
# bit 0.0 vs z58; a209 D_TAN bit 0.0 vs
# z59; a210 Vmat/SV/d_eff bit 0.0 vs z59;
# a211 readout PC1 bit 0.0 vs z60 (sign-
# invariant: |min(PC1, -PC1)| both checked).
import os
import json
import time
import hashlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHASE = 3061
NAME = 'omega_p58_write_highdim_qwen'
BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase%d' % PHASE, NAME)
LOG = os.path.join(OUT, 'run_log.txt')
MODEL_DIR = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL = 36
KV_HEAD = 8
HDIM = 128
SEED_MAIN = 3010
N_PERM = 2000

PREREG = {
    'mode': 'fp32 MODEL weights only (torch.float32, '
            'seed 3010, zero forwards); all arrays '
            'from the z48/z55/z58/z59/z60 npzs; '
            'statistics on the 24 old pairs (rows '
            'c*8+b of D_TAN, c in 1..3)',
    'question': '3061 A main line: the d_tan write '
                'family has d_eff = 8.74 vs the '
                'readout family 1.34 and the two '
                'PC1s differ (3060). WHAT carries '
                'the write-side high dimension: '
                'body identity, prefix condition, '
                'or per-pair residual? And does '
                'the write family share the '
                'readout single axis?',
    'T2_two_way_decomposition': 'MAIN (offline): '
                                'd_bar = mean(D_TAN); '
                                'B_b = mean over c of '
                                'rows (c,b); C_c = '
                                'mean over b; R_kb = '
                                'd_kb - B_b - C_c + '
                                'd_bar; SS_body = '
                                'sum_b 3*||B_b - '
                                'd_bar||^2; SS_prefix '
                                '= sum_c 8*||C_c - '
                                'd_bar||^2; SS_resid = '
                                'sum_kb ||R_kb||^2; '
                                'shares of the total; '
                                'verdict by the largest '
                                'share unless it < 0.4',
    'T3_subfamily_spectrum': 'offline: per-body '
                             'sub-family (3 rows) '
                             'd_eff_b and per-prefix '
                             'sub-family (8 rows) '
                             'd_eff_c; between-body '
                             'mean-direction cos '
                             'matrix (|cos|)',
    'T4_readout_projection': 'MAIN (offline): PC1_ro '
                             'from z60 (Vmat SVD); '
                             'DRES = D_TAN - '
                             '(D_TAN @ PC1_ro) '
                             'PC1_ro; d_eff_resid '
                             'via SVD of DRES; '
                             'per-pair cos(D_TAN[k], '
                             'PC1_ro) with sign-'
                             'invariant |cos| for '
                             'reporting',
    'T5_channel_support': 'offline: SVD of D_TAN -> '
                          'per-PC energy in S16/S64 '
                          'vs 16/64-of-2560 '
                          'permutation nulls (seeds '
                          '9986/9987, R = 2000, '
                          'one-sided p); top-256 '
                          '|PC1_dtan| channels '
                          'overlap with S16/TOP64 '
                          'vs nulls (seed 9988); '
                          'cos(D_TAN[k], TT[k]) '
                          'descriptive',
    'verdict': 'largest of (SS_prefix, SS_body, '
               'SS_resid) -> write_highdim_prefix/'
               'body/residual_qwen; if max share '
               '< 0.4 -> write_highdim_mixed_qwen; '
               'single branch assigned inside the '
               'criteria',
    'anchors': 'a205 source seals 3044-3060 (result.'
               'json sha256_8 vs seal.json); a206 TT '
               'recompute bit 0.0 vs z48; a207 gamma '
               'stats bit 0.0 vs z55 GAMMA_STATS; '
               'a208 TOP64/FLAT64 recompute bit 0.0 '
               'vs z58; a209 D_TAN bit 0.0 vs z59; '
               'a210 Vmat/SV/d_eff recompute bit 0.0 '
               'vs z59; a211 readout PC1 bit 0.0 vs '
               'z60 (sign-invariant: min over '
               'both signs)',
    'statistics_discipline': 'all vectors in the '
                             'same gamma-weighted '
                             'channel space; SVD sign '
                             'is arbitrary - all '
                             'cross-axis cosines '
                             'reported and judged in '
                             'absolute value; nulls '
                             'are channel-set '
                             'permutations on frozen '
                             'seeds; verdict in one '
                             'branch',
    'corrections': 'run1 (29s) crashed pre-verdict '
                   'at the T2 additivity assert: the '
                   'identity d_kb = d_bar + (B_b - '
                   'd_bar) + (C_c - d_bar) + R_kb '
                   'expands to sum ||d_kb||^2 = 24*||'
                   'd_bar||^2 + SS_body + SS_prefix + '
                   'SS_resid, so the three SS terms '
                   'add up to the CENTERED total sum '
                   'of squares sum ||d_kb - d_bar||^2, '
                   'not the raw one; the assert '
                   'compared against the raw total. '
                   'Statistics unobserved at crash '
                   'time (T2 shares not logged; '
                   'a205-a211 all passed, a206-a211 '
                   'bit 0.0). Fix: total_ss is the '
                   'centered sum of squares. run2 '
                   '(28s) crashed pre-verdict at the '
                   'T5 descriptive metric: cos(D_TAN['
                   'k], TT[k]) is a CROSS-SPACE '
                   'cosine - D_TAN lives in the 2560 '
                   'channel space while TT[k] is a '
                   '151936-dim logits vector; the '
                   'gufunc core-dim mismatch raised. '
                   'T2/T3/T4 statistics were already '
                   'produced (SS_body = 0.8267 '
                   'dominant) but the verdict line '
                   'was not reached. Fix: descriptive '
                   'metric replaced with cos(D_TAN[k]'
                   ', Vmat[k]) - same channel space, '
                   'write direction vs its own pair '
                   'readout preimage. run3 (30s) '
                   'completed all measurements and '
                   'the verdict (write_highdim_body_'
                   'qwen, anchors ok) but crashed at '
                   'npz serialization: keys B_B/C_C '
                   'referenced the undefined names - '
                   'the variables are B_b/C_c. Fix: '
                   'npz keys renamed B_BODY/C_PREFIX '
                   'bound to the correct variables. '
                   'run4 (28s) crashed pre-verdict at '
                   'the T5 overlap-null allocation: '
                   'argsort of a (2000, 2560) int64 '
                   'array (39.1 MiB) OOMed - the host '
                   'was near the memory ceiling (the '
                   'fp64 W_U copy is 3.1 GB and '
                   'unused after a210). Fixes: del '
                   'wud after a210; the seed-9988 '
                   'permutation draws generated '
                   'row-by-row (same generator '
                   'stream and per-row argsort, so '
                   'the null draws are bit-identical '
                   'to the batched version - no '
                   'statistical deviation). run5 (27s) '
                   'reached the same npz line and '
                   'crashed again on NameError: the '
                   'run3 fix renamed the KEYS only and '
                   'kept the wrong right-hand sides '
                   '(B_BODY = B_B / C_PREFIX = C_C); '
                   'the variables are B_b / C_c. Fix: '
                   'RHS corrected. run6 (25s) reached '
                   'the NEXT npz line and crashed on '
                   'the same error class: R_KB = R_KB '
                   '(variable is R_kb); a full '
                   'line-by-line audit of the savez '
                   'block confirmed all other keys '
                   'bind correctly. Fix: key R_RESID '
                   '= R_kb. run7 authoritative if '
                   'anchors pass.',
}

lines = []


def log(msg):
    lines.append(str(msg))
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(str(msg) + '\n')
    print(msg)


def sha8(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]


os.makedirs(OUT, exist_ok=True)
if os.path.exists(LOG):
    os.remove(LOG)
for fn in ('execution.json', 'result.json', 'seal.json',
           NAME + '.npz'):
    p = os.path.join(OUT, fn)
    if os.path.exists(p):
        os.remove(p)
t0 = time.time()
created = time.strftime('%Y-%m-%d %H:%M:%S')
execution = {'phase': PHASE, 'name': NAME,
             'created': created, 'prereg': PREREG}
with open(os.path.join(OUT, 'execution.json'), 'w',
          encoding='utf-8') as f:
    json.dump(execution, f, ensure_ascii=False, indent=1)
log('execution.json written (prereg frozen) %s' % created)

torch.manual_seed(SEED_MAIN)
np.random.seed(SEED_MAIN)

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.float32,
    attn_implementation='eager').to('cuda').eval()
assert len(model.model.layers) == NL
assert int(model.config.num_key_value_heads) == KV_HEAD
NVOC = int(model.config.vocab_size)
log('model loaded fp32 weights-only (vocab=%d)' % NVOC)

# ---------- chain: load banks ----------
z48 = np.load(os.path.join(
    BASE, 'phase3048',
    'omega_p45_kvpos_full_replay_qwen',
    'omega_p45_kvpos_full_replay_qwen.npz'),
    allow_pickle=True)
TT = z48['TT']
LG = z48['LG']
z55 = np.load(os.path.join(
    BASE, 'phase3055',
    'omega_p52_gamma_prealign_qwen',
    'omega_p52_gamma_prealign_qwen.npz'),
    allow_pickle=True)
z58 = np.load(os.path.join(
    BASE, 'phase3058',
    'omega_p55_payload_channel_identity_qwen',
    'omega_p55_payload_channel_identity_qwen.npz'),
    allow_pickle=True)
z59 = np.load(os.path.join(
    BASE, 'phase3059',
    'omega_p56_payload_subspace_qwen',
    'omega_p56_payload_subspace_qwen.npz'),
    allow_pickle=True)
z60 = np.load(os.path.join(
    BASE, 'phase3060',
    'omega_p57_pc1_identity_qwen',
    'omega_p57_pc1_identity_qwen.npz'),
    allow_pickle=True)
log('z48/z55/z58/z59/z60 banks loaded')

seal_detail = []
for ph, nm in (
        (3044, 'omega_p41_field_axis_injection_'
               'qwen'),
        (3045, 'omega_p42_l20_axis_anatomy_qwen'),
        (3046, 'omega_p43_kfield_injection_qwen'),
        (3047, 'omega_p44_kv_joint_replay_qwen'),
        (3048, 'omega_p45_kvpos_full_replay_qwen'),
        (3049, 'omega_p46_kvload_localization_'
               'qwen'),
        (3050, 'omega_p47_kvdeep_dissection_'
               'qwen'),
        (3051, 'omega_p48_l35_anatomy_qwen'),
        (3052, 'omega_p49_kvhead_identity_qwen'),
        (3053, 'omega_p50_gate_source_qwen'),
        (3054, 'omega_p51_norm_projection_qwen'),
        (3055, 'omega_p52_gamma_prealign_qwen'),
        (3056, 'omega_p53_gamma_spectrum_qwen'),
        (3057, 'omega_p54_whiten_geometry_qwen'),
        (3058, 'omega_p55_payload_channel_'
               'identity_qwen'),
        (3059, 'omega_p56_payload_subspace_'
               'qwen'),
        (3060, 'omega_p57_pc1_identity_qwen')):
    d = os.path.join(BASE, 'phase%d' % ph, nm)
    with open(os.path.join(d, 'seal.json'),
              encoding='utf-8') as f:
        sealj = json.load(f)
    with open(os.path.join(d, 'result.json'),
              'rb') as f:
        s = hashlib.sha256(f.read()).hexdigest()[:8]
    seal_detail.append(bool(
        s == sealj['result_sha256_8']))
a205_ok = bool(seal_detail) and all(seal_detail)
log('a205 source seals ok=%s' % a205_ok)

# a206: TT recompute (cond-major, prompt = b*4+c)
LGd = np.stack([LG[b * 4 + c] - LG[b * 4]
                for c in (1, 2, 3)
                for b in range(8)])
a206_diff = float(np.max(np.abs(LGd - TT)))
a206_ok = bool(a206_diff == 0.0)
log('a206 TT recompute diff=%.3e ok=%s'
    % (a206_diff, a206_ok))

# ---------- gamma / weights ----------
GAMMA_T = model.model.norm.weight.detach().clone()
GAMMA = GAMMA_T.double().cpu().numpy()
assert GAMMA.shape == (2560,)
gm_mean = float(GAMMA.mean())

gs_now = np.array([GAMMA.min(), GAMMA.max(),
                   GAMMA.mean(), GAMMA.std()])
a207_diff = float(np.max(np.abs(
    gs_now - z55['GAMMA_STATS'])))
a207_ok = bool(a207_diff == 0.0)
log('a207 gamma stats diff=%.3e ok=%s'
    % (a207_diff, a207_ok))

order_shape = np.argsort(
    np.abs(GAMMA - gm_mean))[::-1]
TOP64 = order_shape[:64].astype(np.int64)
FLAT64 = order_shape[::-1][:64].astype(np.int64)
S16 = TOP64[:16].astype(np.int64)
a208_diff = max(
    float(np.max(np.abs(TOP64
                        - z58['TOP64']))),
    float(np.max(np.abs(FLAT64
                        - z58['FLAT64']))))
a208_ok = bool(a208_diff == 0.0)
log('a208 TOP64/FLAT64 recompute diff=%.3e ok=%s'
    % (a208_diff, a208_ok))

# a209: D_TAN bit vs z59
D_TAN = z59['D_TAN']
assert D_TAN.shape == (24, 2560)
a209_diff = float(np.max(np.abs(
    D_TAN - z59['D_TAN'])))
a209_ok = bool(a209_diff == 0.0)
log('a209 D_TAN bit vs z59 diff=%.3e ok=%s'
    % (a209_diff, a209_ok))

# a210: Vmat/SV/d_eff recompute vs z59
WU = model.lm_head.weight.detach()  # (V, 2560)
wud = WU.double().cpu().numpy()
NP_ = 24
Vmat = np.zeros((NP_, 2560))
for k in range(NP_):
    w_t = wud.T @ TT[k]
    Vmat[k] = GAMMA * w_t
U_r, sv_r, Vt_r = np.linalg.svd(
    Vmat, full_matrices=False)
d_eff_ro = float(sv_r.sum() ** 2 / (sv_r ** 2).sum())
a210_diff = max(
    float(np.max(np.abs(Vmat - z59['Vmat']))),
    float(np.max(np.abs(sv_r - z59['SV']))),
    abs(d_eff_ro - float(z59['D_EFF'])))
a210_ok = bool(a210_diff == 0.0)
log('a210 Vmat/SV/d_eff recompute diff=%.3e ok=%s'
    % (a210_diff, a210_ok))
del wud  # free the 3.1 GB fp64 copy (unused below)

# a211: readout PC1 bit vs z60 (sign-invariant)
PC1_RO = z60['PC1']
diff_p = float(np.max(np.abs(PC1_RO - Vt_r[0])))
diff_m = float(np.max(np.abs(PC1_RO + Vt_r[0])))
a211_diff = float(min(diff_p, diff_m))
a211_ok = bool(a211_diff == 0.0)
log('a211 readout PC1 vs z60 diff=%.3e ok=%s'
    % (a211_diff, a211_ok))


def cos0(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b) / (na * nb)


# row index maps: k = c_idx*8 + b, c in 1..3
bidx = np.array([b for c in (1, 2, 3)
                 for b in range(8)])
cidx = np.array([c for c in (1, 2, 3)
                 for b in range(8)])

# ---------- T2: two-way decomposition ----------
d_bar = D_TAN.mean(axis=0)
B_b = np.stack([D_TAN[bidx == b].mean(axis=0)
                for b in range(8)])
C_c = np.stack([D_TAN[cidx == c].mean(axis=0)
                for c in (1, 2, 3)])
R_kb = np.zeros_like(D_TAN)
for k in range(NP_):
    R_kb[k] = (D_TAN[k] - B_b[bidx[k]]
               - C_c[cidx[k] - 1] + d_bar)
total_ss = float(((D_TAN - d_bar) ** 2).sum())
ss_body = float(sum(
    3 * float(((B_b[b] - d_bar) ** 2).sum())
    for b in range(8)))
ss_prefix = float(sum(
    8 * float(((C_c[c] - d_bar) ** 2).sum())
    for c in range(3)))
ss_resid = float((R_kb ** 2).sum())
assert abs(ss_body + ss_prefix + ss_resid
           - total_ss) < 1e-6 * total_ss
sh_body = ss_body / total_ss
sh_prefix = ss_prefix / total_ss
sh_resid = ss_resid / total_ss
log('T2 two-way: SS_body=%.4f SS_prefix=%.4f '
    'SS_resid=%.4f (sum=%.4f)'
    % (sh_body, sh_prefix, sh_resid,
       sh_body + sh_prefix + sh_resid))

# ---------- T3: sub-family spectra ----------
d_eff_body = np.zeros(8)
for b in range(8):
    _, svb, _ = np.linalg.svd(
        D_TAN[bidx == b], full_matrices=False)
    d_eff_body[b] = (svb.sum() ** 2
                     / (svb ** 2).sum())
d_eff_pref = np.zeros(3)
for ci, c in enumerate((1, 2, 3)):
    _, svc, _ = np.linalg.svd(
        D_TAN[cidx == c], full_matrices=False)
    d_eff_pref[ci] = (svc.sum() ** 2
                      / (svc ** 2).sum())
cos_bodies = np.zeros((8, 8))
for b1 in range(8):
    for b2 in range(8):
        cos_bodies[b1, b2] = abs(
            cos0(B_b[b1], B_b[b2]))
off_diag = cos_bodies[
    ~np.eye(8, dtype=bool)]
log('T3: d_eff_body=%s (med %.2f); '
    'd_eff_prefix=%s (med %.2f); between-body '
    'mean |cos| med=%.3f min=%.3f'
    % (np.array2string(d_eff_body, precision=2),
       float(np.median(d_eff_body)),
       np.array2string(d_eff_pref, precision=2),
       float(np.median(d_eff_pref)),
       float(np.median(off_diag)),
       float(off_diag.min())))

# ---------- T4: readout projection ----------
coef = D_TAN @ PC1_RO
DRES = D_TAN - coef[:, None] * PC1_RO[None, :]
_, sv_res, _ = np.linalg.svd(
    DRES, full_matrices=False)
d_eff_resid = float(
    (sv_res.sum() ** 2) / (sv_res ** 2).sum())
cos_dtan_pc1 = np.array([
    abs(cos0(D_TAN[k], PC1_RO))
    for k in range(NP_)])
log('T4: d_eff_resid after removing readout '
    'PC1 = %.2f (was %.2f); |cos(d_tan, '
    'PC1_ro)| med=%.4f max=%.4f'
    % (d_eff_resid, 8.74,
       float(np.median(cos_dtan_pc1)),
       float(cos_dtan_pc1.max())))

# ---------- T5: channel support ----------
Ud, svd_w, Vtd = np.linalg.svd(
    D_TAN, full_matrices=False)
d_eff_dtan = float(
    (svd_w.sum() ** 2) / (svd_w ** 2).sum())
PC1_D = Vtd[0]
pc_e16_w = (Vtd[:, S16] ** 2).sum(axis=1)
pc_e64_w = (Vtd[:, TOP64] ** 2).sum(axis=1)
e16_pc1d = float((PC1_D[S16] ** 2).sum())
e64_pc1d = float((PC1_D[TOP64] ** 2).sum())
rng1 = np.random.default_rng(9986)
pm1 = np.argsort(
    rng1.random((N_PERM, 2560)), axis=1)[:, :16]
null16_w = np.array([
    float((PC1_D[pm] ** 2).sum()) for pm in pm1])
p_e16_w = float((null16_w >= e16_pc1d).sum()
                + 1) / (N_PERM + 1)
rng2 = np.random.default_rng(9987)
pm2 = np.argsort(
    rng2.random((N_PERM, 2560)), axis=1)[:, :64]
null64_w = np.array([
    float((PC1_D[pm] ** 2).sum()) for pm in pm2])
p_e64_w = float((null64_w >= e64_pc1d).sum()
                + 1) / (N_PERM + 1)
top256_w = np.argsort(np.abs(PC1_D))[::-1][:256]
ov16_w = int(len(set(top256_w.tolist())
                 & set(S16.tolist())))
ov64_w = int(len(set(top256_w.tolist())
                 & set(TOP64.tolist())))
rng3 = np.random.default_rng(9988)
pset = np.empty((N_PERM, 256), dtype=np.int64)
for i in range(N_PERM):
    pset[i] = np.argsort(
        rng3.random(2560))[:256]
ov16_null = np.array([float(len(
    set(ps.tolist()) & set(S16.tolist())))
    for ps in pset])
ov64_null = np.array([float(len(
    set(ps.tolist()) & set(TOP64.tolist())))
    for ps in pset])
p_ov16_w = float((ov16_null >= ov16_w).sum()
                 + 1) / (N_PERM + 1)
p_ov64_w = float((ov64_null >= ov64_w).sum()
                 + 1) / (N_PERM + 1)
cos_dtan_vmat = np.array([
    cos0(D_TAN[k], Vmat[k]) for k in range(NP_)])
log('T5: d_eff_dtan=%.2f; PC1_d E16=%.4f vs null '
    'med=%.4f (p=%.4f), E64=%.4f vs med=%.4f '
    '(p=%.4f); top256 overlap S16=%d (null '
    'med=%.1f, p=%.4f) TOP64=%d (null med=%.1f, '
    'p=%.4f); cos(d_tan, t_k) med=%.4f'
    % (d_eff_dtan, e16_pc1d,
       float(np.median(null16_w)), p_e16_w,
       e64_pc1d, float(np.median(null64_w)),
       p_e64_w, ov16_w,
       float(np.median(ov16_null)), p_ov16_w,
       ov64_w, float(np.median(ov64_null)),
       p_ov64_w, float(np.median(cos_dtan_vmat))))

# ---------- verdict ----------
shares = {'prefix': sh_prefix,
          'body': sh_body,
          'resid': sh_resid}
top_key = max(shares, key=shares.get)
top_share = shares[top_key]
if top_share < 0.4:
    verdict = 'write_highdim_mixed_qwen'
elif top_key == 'prefix':
    verdict = 'write_highdim_prefix_qwen'
elif top_key == 'body':
    verdict = 'write_highdim_body_qwen'
else:
    verdict = 'write_highdim_residual_qwen'
log('VERDICT: %s (top share %s=%.4f)'
    % (verdict, top_key, top_share))

anchor_core_ok = bool(a205_ok and a206_ok
                      and a207_ok and a208_ok
                      and a209_ok and a210_ok
                      and a211_ok)
log('anchors core ok=%s' % anchor_core_ok)

elapsed = time.time() - t0
npz_path = os.path.join(OUT, NAME + '.npz')
np.savez(npz_path,
         SH_BODY=np.float64(sh_body),
         SH_PREFIX=np.float64(sh_prefix),
         SH_RESID=np.float64(sh_resid),
         SS_BODY=np.float64(ss_body),
         SS_PREFIX=np.float64(ss_prefix),
         SS_RESID=np.float64(ss_resid),
         TOTAL_SS=np.float64(total_ss),
         D_EFF_BODY=d_eff_body,
         D_EFF_PREFIX=d_eff_pref,
         COS_BODIES=cos_bodies,
         D_EFF_RESID=np.float64(d_eff_resid),
         COS_DTAN_PC1=cos_dtan_pc1,
         D_EFF_DTAN=np.float64(d_eff_dtan),
         E16_PC1D=np.float64(e16_pc1d),
         E64_PC1D=np.float64(e64_pc1d),
         NULL16_W=null16_w,
         NULL64_W=null64_w,
         TOP256_W=top256_w,
         OV16_W=np.float64(ov16_w),
         OV64_W=np.float64(ov64_w),
         OV16_NULL=ov16_null,
         OV64_NULL=ov64_null,
         COS_DTAN_VMAT=cos_dtan_vmat,
         B_BODY=B_b,
         C_PREFIX=C_c,
         R_RESID=R_kb,
         PC1_D=PC1_D,
         a205_ok=np.bool_(a205_ok),
         a206_diff=np.float64(a206_diff),
         a207_diff=np.float64(a207_diff),
         a208_diff=np.float64(a208_diff),
         a209_diff=np.float64(a209_diff),
         a210_diff=np.float64(a210_diff),
         a211_diff=np.float64(a211_diff),
         anchor_core_ok=np.bool_(
             anchor_core_ok),
         verdict=np.array(verdict),
         elapsed=np.float64(elapsed))

stats = {
    'T2_two_way_decomposition': {
        'share_body': sh_body,
        'share_prefix': sh_prefix,
        'share_resid': sh_resid},
    'T3_subfamily_spectrum': {
        'd_eff_body': [float(v)
                       for v in d_eff_body],
        'd_eff_prefix': [float(v)
                         for v in d_eff_pref],
        'between_body_mean_abs_cos_med':
            float(np.median(off_diag)),
        'between_body_mean_abs_cos_min':
            float(off_diag.min())},
    'T4_readout_projection': {
        'd_eff_resid': d_eff_resid,
        'cos_dtan_pc1_med':
            float(np.median(cos_dtan_pc1)),
        'cos_dtan_pc1_max':
            float(cos_dtan_pc1.max())},
    'T5_channel_support': {
        'd_eff_dtan': d_eff_dtan,
        'E16_pc1d': e16_pc1d,
        'null16_med':
            float(np.median(null16_w)),
        'p_E16': p_e16_w,
        'E64_pc1d': e64_pc1d,
        'null64_med':
            float(np.median(null64_w)),
        'p_E64': p_e64_w,
        'ov16': ov16_w,
        'ov16_null_med':
            float(np.median(ov16_null)),
        'p_ov16': p_ov16_w,
        'ov64': ov64_w,
        'ov64_null_med':
            float(np.median(ov64_null)),
        'p_ov64': p_ov64_w,
        'cos_dtan_vmat_med':
            float(np.median(cos_dtan_vmat))},
    'anchors': {'a205_seals_ok': a205_ok,
                'a206_tt_diff': a206_diff,
                'a207_gamma_stats_diff':
                    a207_diff,
                'a208_top64_diff': a208_diff,
                'a209_dtan_diff': a209_diff,
                'a210_vmat_sv_diff':
                    a210_diff,
                'a211_pc1_diff': a211_diff,
                'anchor_core_ok':
                    anchor_core_ok},
}
result = {'phase': PHASE, 'name': NAME,
          'created': created, 'elapsed': elapsed,
          'run': 'run7 authoritative (fp32 '
                 'weights-only, zero forwards; '
                 'arrays from z48/z55/z58/z59/z60 '
                 'npzs; readout PC1 recomputed and '
                 'bit-anchored vs z60 sign-'
                 'invariantly)',
          'prereg': PREREG, 'stats': stats,
          'verdict': verdict}
res_path = os.path.join(OUT, 'result.json')
with open(res_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=1)

seal = {
    'phase': PHASE, 'name': NAME, 'created': created,
    'npz_sha256_8': sha8(npz_path),
    'result_sha256_8': sha8(res_path),
    'exec_sha256_8': sha8(os.path.join(
        OUT, 'execution.json')),
    'script_sha256_8': sha8(os.path.abspath(__file__)),
    'verdict': verdict,
    'anchor_core_ok': anchor_core_ok,
}
with open(os.path.join(OUT, 'seal.json'), 'w',
          encoding='utf-8') as f:
    json.dump(seal, f, ensure_ascii=False, indent=1)
log('sealed npz8=%s result8=%s exec8=%s '
    'script8=%s elapsed=%.1fs'
    % (seal['npz_sha256_8'],
       seal['result_sha256_8'],
       seal['exec_sha256_8'],
       seal['script_sha256_8'], elapsed))
log('sealed')
