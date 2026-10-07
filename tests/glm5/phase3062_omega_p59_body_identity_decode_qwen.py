# -*- coding: utf-8 -*-
# Phase 3062 - Omega-P59: identity decoding of
# the 8 body write directions. 3061 verdict
# write_highdim_body_qwen: 83 pct of the
# write-side variance is body identity
# (B_b - d_bar, ~5 dims). Question: WHAT is
# the semantic content of the body-identity
# dimensions? T2 (MAIN decode, offline):
# logits_ID = W_U @ (B_b - d_bar) -> top-15
# tokens per body + margin vs a rotation null
# (unit gaussian dirs scaled by ||B_ID_b|| -
# logits are linear in the direction, so unit
# logits are computed once and scaled) +
# SELF-token test: does body b's own
# connective token (' so'...' thus') rank in
# its own decode? + cross-body specificity.
# T3 (MAIN coupling, offline): M[b1,b2] =
# mean_c cos(B_ID[b1], Vmat[c*8+b2]) -
# diagonal dominance vs body-label
# permutation. T4 (MAIN channels, offline):
# pooled identity profile P = sqrt(mean_b
# B_ID_b^2); top-256 channels overlap vs
# S16 / TOP64 / 3022 coalition write-out
# top-64 (recomputed, bit-anchored vs z58
# COAL_TOP64) / 2802 sig channels, all vs
# 256-of-2560 permutation nulls; Spearman(P,
# |votes2802|). T5 (offline): per-body
# top-256 support Jaccard vs null (distinct
# vs shared channels) + descriptive decode
# of d_bar and raw B_b. verdict (single
# branch): count_self >= 6 (p_self <= 0.05
# and spec > 0) and diag p <= 0.005 ->
# body_identity_selftoken_qwen; else count_
# margin >= 6 (p_margin <= 0.005) and >= 1
# channel overlap p <= 0.005 and diag p <=
# 0.005 -> body_identity_channel_semantic_
# qwen; else count_margin >= 6 ->
# body_identity_decoded_unplaced_qwen;
# else -> body_identity_opaque_qwen.
# anchors: a212 source seals 3044-3061;
# a213 TT recompute bit 0.0 vs z48; a214
# gamma stats bit 0.0 vs z55; a215 TOP64/
# FLAT64 bit 0.0 vs z58; a216 D_TAN bit 0.0
# vs z59; a217 B_BODY/C_PREFIX recompute +
# write PC1 sign-invariant vs z61; a218
# Vmat/SV/d_eff recompute bit 0.0 vs z59
# (W_U identity proof); a219 COAL_TOP64
# recompute bit 0.0 vs z58.
import os
import json
import time
import hashlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHASE = 3062
NAME = 'omega_p59_body_identity_decode_qwen'
BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase%d' % PHASE, NAME)
LOG = os.path.join(OUT, 'run_log.txt')
MODEL_DIR = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL = 36
KV_HEAD = 8
SEED_MAIN = 3010
N_PERM = 2000
R_ROT = 1000
ROT_CHUNK = 200
TARGETS = ('so', 'because', 'therefore', 'however',
           'while', 'yet', 'although', 'thus')

PREREG = {
    'mode': 'fp32 MODEL weights only (torch.float32, '
            'seed 3010, zero forwards); all arrays '
            'from the z48/z55/z58/z59/z60/z61 npzs '
            'plus fp64 recomputes of Vmat (a218) '
            'and the 3022 coalition write-out '
            'channels (a219); statistics offline',
    'question': '3062 A main line: 83 pct of the '
                'write-side variance is body '
                'identity (B_b - d_bar, ~5 dims). '
                'WHAT semantic content lives in '
                'these body-identity dimensions: '
                'does body b write toward its own '
                'connective (self-token decode), '
                'does its write direction couple '
                'to its own readout preimages '
                '(diagonal dominance), and whose '
                'channels do the identity '
                'dimensions occupy (pipe S16/'
                'TOP64 vs 2802 voting dims vs '
                '3022 relay coalition)?',
    'T2_vocab_decode': 'MAIN (offline): B_ID = '
                       'B_BODY - d_bar (d_bar = '
                       'mean of 24 D_TAN rows); '
                       'logits_ID = W_U @ B_ID '
                       '(fp64, 8 x 151936); top-15 '
                       'tokens per body (descriptive '
                       'report); margin_b = top1 - '
                       'top2; rotation null: R = '
                       '1000 unit gaussian dirs '
                       '(seed 9991), logits linear '
                       'in the direction so unit '
                       'logits computed once and '
                       'scaled by ||B_ID_b||; '
                       'p_margin_b one-sided; '
                       'SELF test: tok_b = first '
                       'token of " " + TARGETS[b]; '
                       'p_self_b = fraction of '
                       'scaled null self-logits '
                       '>= observed; spec_b = '
                       'self_b - mean(other 7 '
                       'body tokens); self pct '
                       'rank descriptive',
    'T3_write_read_coupling': 'MAIN (offline): '
                              'M[b1,b2] = mean over '
                              'c in 1..3 of cos(B_ID['
                              'b1], Vmat[c*8+b2]); '
                              'obs = mean diagonal; '
                              'null = body-label '
                              'permutation of the '
                              '8 columns (R = 2000, '
                              'seed 9992); p one-sided',
    'T4_channel_identity': 'MAIN (offline): P = '
                           'sqrt(mean_b B_ID_b^2); '
                           'top256_P overlap with '
                           'S16 (seed 9993), TOP64 '
                           '(seed 9994), COAL_TOP64 '
                           'recomputed vs z58 (seed '
                           '9995), sig2802 (seed '
                           '9996) - each vs '
                           '256-of-2560 permutation '
                           'nulls (R = 2000, row-'
                           'wise draws, one-sided); '
                           'Spearman(P, |votes2802|)',
    'T5_support_distinctness': 'offline: per-body '
                               'top-256 |B_ID_b| sets - '
                               '28 pairwise Jaccards '
                               'med vs null of two '
                               'random 256-sets (R = '
                               '2000, seed 9997); '
                               'descriptive decode of '
                               'd_bar and raw B_b '
                               'top-8',
    'verdict': 'count_self = #{b: p_self_b <= 0.05 '
               'and spec_b > 0}; count_margin = '
               '#{b: p_margin_b <= 0.005}; sig_ch = '
               '#{channel overlaps with p <= 0.005}; '
               'count_self >= 6 and p_diag <= 0.005 '
               '-> body_identity_selftoken_qwen; '
               'elif count_margin >= 6 and sig_ch '
               '>= 1 and p_diag <= 0.005 -> '
               'body_identity_channel_semantic_qwen;'
               ' elif count_margin >= 6 -> '
               'body_identity_decoded_unplaced_qwen;'
               ' else -> body_identity_opaque_qwen; '
               'single branch assigned inside the '
               'criteria',
    'anchors': 'a212 source seals 3044-3061; a213 '
               'TT recompute bit 0.0 vs z48; a214 '
               'gamma stats bit 0.0 vs z55; a215 '
               'TOP64/FLAT64 recompute bit 0.0 vs '
               'z58; a216 D_TAN bit 0.0 vs z59; '
               'a217 B_BODY/C_PREFIX recompute + '
               'write PC1 sign-invariant vs z61; '
               'a218 Vmat/SV/d_eff recompute bit '
               '0.0 vs z59; a219 COAL_TOP64 '
               'recompute bit 0.0 vs z58',
    'statistics_discipline': 'all vectors in the '
                             'same gamma-weighted '
                             'channel space; logits '
                             'decode is W_U applied '
                             'IN that space (linear '
                             'map, same convention as '
                             '3060 T4); SVD signs '
                             'arbitrary - PC1 anchor '
                             'sign-invariant (min '
                             'over both signs); nulls '
                             'are permutation/rotation '
                             'on frozen seeds; '
                             'verdict in one branch',
    'memory_discipline': 'wud fp64 (3.1 GB) held '
                         'only while W_U products are '
                         'computed (a218, T2 decode, '
                         'rotation null) then del; '
                         'permutation nulls generated '
                         'row-by-row (same generator '
                         'stream + per-row argsort, '
                         'bit-equivalent to batched); '
                         'dh fp64 (200 MB) del after '
                         'a219',
    'corrections': 'run1 (35s) crashed '
                   'pre-verdict at the '
                   'decode report: the '
                   'd_bar/raw B_b '
                   'descriptive decode '
                   'block was lost to a '
                   'phantom Edit (report '
                   'section landed, the '
                   'computing block did '
                   'not - local phantom-'
                   'edit defect); '
                   'NameError top8_dbar '
                   'after T2 was logged '
                   '(count_margin=0, '
                   'count_self=1, T3-T5 '
                   'unreached). Fix: '
                   'block re-inserted '
                   'via a Python patch '
                   'with a count==1 '
                   'assert before del '
                   'wud. run2 '
                   '(34s) crashed pre-'
                   'verdict at T3: row '
                   'indexing - D_TAN/'
                   'Vmat rows are c-'
                   'major with c_idx '
                   '0..2 (k = c_idx*8 '
                   '+ b), the loop '
                   'used c in 1..3 '
                   'giving k up to 31 '
                   '(IndexError 24); '
                   'rows for body b '
                   'are b, 8+b, 16+b.'
                   ' Fix applied. '
                   'run3 authoritative '
                   'if anchors pass.',
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
           NAME + '.npz', 'decode_report.txt'):
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
z61 = np.load(os.path.join(
    BASE, 'phase3061',
    'omega_p58_write_highdim_qwen',
    'omega_p58_write_highdim_qwen.npz'),
    allow_pickle=True)
z22 = np.load(os.path.join(
    BASE, 'phase3022',
    'omega_p2p_l3_relay_neurons_qwen',
    'omega_p2p_l3_relay_neurons_qwen.npz'),
    allow_pickle=True)
z2802 = np.load(os.path.join(
    BASE, 'phase2802', 'qwen4_polysemy_spectrum',
    'polysemy.npz'), allow_pickle=True)
log('z48/z55/z58/z59/z60/z61/z22/z2802 banks loaded')

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
        (3060, 'omega_p57_pc1_identity_qwen'),
        (3061, 'omega_p58_write_highdim_qwen')):
    d = os.path.join(BASE, 'phase%d' % ph, nm)
    with open(os.path.join(d, 'seal.json'),
              encoding='utf-8') as f:
        sealj = json.load(f)
    with open(os.path.join(d, 'result.json'),
              'rb') as f:
        s = hashlib.sha256(f.read()).hexdigest()[:8]
    seal_detail.append(bool(
        s == sealj['result_sha256_8']))
a212_ok = bool(seal_detail) and all(seal_detail)
log('a212 source seals ok=%s' % a212_ok)

# a213: TT recompute (cond-major, prompt = b*4+c)
LGd = np.stack([LG[b * 4 + c] - LG[b * 4]
                for c in (1, 2, 3)
                for b in range(8)])
a213_diff = float(np.max(np.abs(LGd - TT)))
a213_ok = bool(a213_diff == 0.0)
log('a213 TT recompute diff=%.3e ok=%s'
    % (a213_diff, a213_ok))

# ---------- gamma / weights ----------
GAMMA_T = model.model.norm.weight.detach().clone()
GAMMA = GAMMA_T.double().cpu().numpy()
assert GAMMA.shape == (2560,)
gm_mean = float(GAMMA.mean())

gs_now = np.array([GAMMA.min(), GAMMA.max(),
                   GAMMA.mean(), GAMMA.std()])
a214_diff = float(np.max(np.abs(
    gs_now - z55['GAMMA_STATS'])))
a214_ok = bool(a214_diff == 0.0)
log('a214 gamma stats diff=%.3e ok=%s'
    % (a214_diff, a214_ok))

order_shape = np.argsort(
    np.abs(GAMMA - gm_mean))[::-1]
TOP64 = order_shape[:64].astype(np.int64)
FLAT64 = order_shape[::-1][:64].astype(np.int64)
S16 = TOP64[:16].astype(np.int64)
a215_diff = max(
    float(np.max(np.abs(TOP64
                        - z58['TOP64']))),
    float(np.max(np.abs(FLAT64
                        - z58['FLAT64']))))
a215_ok = bool(a215_diff == 0.0)
log('a215 TOP64/FLAT64 recompute diff=%.3e ok=%s'
    % (a215_diff, a215_ok))

# a216: D_TAN bit vs z59
D_TAN = z59['D_TAN']
assert D_TAN.shape == (24, 2560)
a216_diff = float(np.max(np.abs(
    D_TAN - z59['D_TAN'])))
a216_ok = bool(a216_diff == 0.0)
log('a216 D_TAN bit vs z59 diff=%.3e ok=%s'
    % (a216_diff, a216_ok))

# a217: B_BODY/C_PREFIX recompute + write PC1
bidx = np.array([b for c in (1, 2, 3)
                 for b in range(8)])
cidx = np.array([c for c in (1, 2, 3)
                 for b in range(8)])
d_bar = D_TAN.mean(axis=0)
B_b = np.stack([D_TAN[bidx == b].mean(axis=0)
                for b in range(8)])
C_c = np.stack([D_TAN[cidx == c].mean(axis=0)
                for c in (1, 2, 3)])
Uw, svw, Vtw = np.linalg.svd(
    D_TAN, full_matrices=False)
PC1_D = Vtw[0]
diff_pc_p = float(np.max(np.abs(
    PC1_D - z61['PC1_D'])))
diff_pc_m = float(np.max(np.abs(
    PC1_D + z61['PC1_D'])))
a217_diff = max(
    float(np.max(np.abs(B_b - z61['B_BODY']))),
    float(np.max(np.abs(C_c - z61['C_PREFIX']))),
    float(min(diff_pc_p, diff_pc_m)))
a217_ok = bool(a217_diff == 0.0)
log('a217 B_BODY/C_PREFIX/PC1 recompute vs z61 '
    'diff=%.3e ok=%s' % (a217_diff, a217_ok))

# a218: Vmat/SV/d_eff recompute vs z59 (fp64 W_U)
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
a218_diff = max(
    float(np.max(np.abs(Vmat - z59['Vmat']))),
    float(np.max(np.abs(sv_r - z59['SV']))),
    abs(d_eff_ro - float(z59['D_EFF'])))
a218_ok = bool(a218_diff == 0.0)
log('a218 Vmat/SV/d_eff recompute diff=%.3e ok=%s'
    % (a218_diff, a218_ok))

# a219: COAL_TOP64 recompute vs z58 (3022 alliance)
dh = (model.layers[3].mlp.down_proj.weight.detach()
      if hasattr(model, 'layers')
      else model.model.layers[3].mlp.down_proj.weight
      .detach()).double().cpu().numpy()
s_relay = z22['s_relay'].astype(np.float64)
assert s_relay.shape == (11, 9728)
w32_mean = np.zeros(2560)
for ti in range(11):
    s = s_relay[ti]
    jtop = np.argsort(s)[::-1][:32]
    w32_mean += np.abs(dh[:, jtop] @ s[jtop])
w32_mean /= 11.0
COAL_TOP64 = np.argsort(w32_mean)[::-1][:64] \
    .astype(np.int64)
a219_diff = float(np.max(np.abs(
    COAL_TOP64 - z58['COAL_TOP64'])))
a219_ok = bool(a219_diff == 0.0)
log('a219 COAL_TOP64 recompute diff=%.3e ok=%s'
    % (a219_diff, a219_ok))
del dh

# ---------- body identity component ----------
B_ID = B_b - d_bar[None, :]
norm_bid = np.linalg.norm(B_ID, axis=1)
log('body identity: ||B_ID|| = %s'
    % np.array2string(norm_bid, precision=3))

# ---------- T2: vocab decode ----------
logits_ID = (wud @ B_ID.T).T  # (8, V) fp64
tok_ids = np.zeros(8, dtype=np.int64)
tok_n = np.zeros(8, dtype=np.int64)
for b in range(8):
    ids = tok.encode(' ' + TARGETS[b],
                     add_special_tokens=False)
    tok_n[b] = len(ids)
    tok_ids[b] = ids[0]
log('self tokens: %s (n=%s)'
    % (np.array2string(tok_ids),
       np.array2string(tok_n)))
margin_obs = np.zeros(8)
self_obs = np.zeros(8)
self_pct = np.zeros(8)
spec_obs = np.zeros(8)
for b in range(8):
    lg = logits_ID[b]
    part = np.partition(lg, -2)
    margin_obs[b] = part[-1] - part[-2]
    self_obs[b] = lg[tok_ids[b]]
    self_pct[b] = float((lg < self_obs[b]).sum()) \
        / (NVOC - 1)
    others = [tok_ids[bb] for bb in range(8)
              if bb != b]
    spec_obs[b] = self_obs[b] \
        - float(np.mean(lg[others]))

# rotation null: unit dirs, logits ONCE, scaled
rng4 = np.random.default_rng(9991)
margins_u = np.zeros(R_ROT)
self_u = np.zeros((R_ROT, 8))
for lo in range(0, R_ROT, ROT_CHUNK):
    U = rng4.standard_normal(
        (ROT_CHUNK, 2560))
    U /= np.linalg.norm(U, axis=1,
                        keepdims=True)
    LU = U @ wud.T  # (chunk, V) fp64
    part = np.partition(LU, -2, axis=1)
    margins_u[lo:lo + ROT_CHUNK] = \
        part[:, -1] - part[:, -2]
    self_u[lo:lo + ROT_CHUNK] = LU[:, tok_ids]
# descriptive decode of d_bar and raw B_b
# (shared-component controls, BEFORE del)
lg_dbar = wud @ d_bar
top8_dbar = np.argsort(lg_dbar)[::-1][:8]
logits_raw = (wud @ B_b.T).T  # (8, V) fp64
top8_raw = np.argsort(logits_raw,
                      axis=1)[:, ::-1][:, :8]
del wud  # free the 3.1 GB fp64 copy
p_margin = np.zeros(8)
p_self = np.zeros(8)
for b in range(8):
    thr_m = margin_obs[b] / norm_bid[b]
    p_margin[b] = float(
        (margins_u >= thr_m).sum() + 1) \
        / (R_ROT + 1)
    thr_s = self_obs[b] / norm_bid[b]
    p_self[b] = float(
        (self_u[:, b] >= thr_s).sum() + 1) \
        / (R_ROT + 1)
count_self = int(sum(
    1 for b in range(8)
    if p_self[b] <= 0.05 and spec_obs[b] > 0))
count_margin = int(sum(
    1 for b in range(8)
    if p_margin[b] <= 0.005))
log('T2: margin obs=%s p=%s (count_margin=%d); '
    'self pct=%s p_self=%s spec=%s (count_self=%d)'
    % (np.array2string(margin_obs, precision=1),
       np.array2string(p_margin, precision=4),
       count_margin,
       np.array2string(self_pct, precision=3),
       np.array2string(p_self, precision=4),
       np.array2string(spec_obs, precision=1),
       count_self))

# decode report (descriptive)
top15 = np.argsort(logits_ID, axis=1)[:, ::-1][:, :15]
rep = ['Phase 3062 decode report (W_U @ '
       '(B_b - d_bar), top-15 per body)']
for b in range(8):
    toks = [tok.decode([int(i)]) for i in top15[b]]
    rep.append('body %d (%s): self=%r rank_pct='
               '%.4f | %s'
               % (b, TARGETS[b], TARGETS[b],
                  self_pct[b],
                  ' '.join(repr(t) for t in toks)))
rep.append('d_bar (shared component) top-8: %s'
           % ' '.join(repr(tok.decode([int(i)]))
                      for i in top8_dbar))
for b in range(8):
    rep.append('raw B_%d top-8: %s'
               % (b, ' '.join(
                   repr(tok.decode([int(i)]))
                   for i in top8_raw[b])))
with open(os.path.join(OUT, 'decode_report.txt'),
          'w', encoding='utf-8') as f:
    f.write('\n'.join(rep) + '\n')
log('decode_report.txt written')

# ---------- T3: write-read coupling ----------
M = np.zeros((8, 8))
for b1 in range(8):
    nb = float(np.linalg.norm(B_ID[b1]))
    for b2 in range(8):
        rows = [ci * 8 + b2
                for ci in range(3)]
        cs = [float(B_ID[b1] @ Vmat[k])
              / (nb * float(np.linalg.norm(
                  Vmat[k])))
              for k in rows]
        M[b1, b2] = float(np.mean(cs))
diag_obs = float(np.mean(np.diag(M)))
off_obs = float(np.mean(M[~np.eye(8, dtype=bool)]))
rng5 = np.random.default_rng(9992)
diag_null = np.array([
    float(np.mean([M[b, rng5.permutation(8)[b]]
                   for b in range(8)]))
    for _ in range(N_PERM)])
p_diag = float((diag_null >= diag_obs).sum()
               + 1) / (N_PERM + 1)
log('T3 coupling: diag=%.4f off=%.4f (gap %+.4f) '
    'p_diag=%.4f' % (diag_obs, off_obs,
                     diag_obs - off_obs, p_diag))

# ---------- T4: channel identity ----------
P_prof = np.sqrt((B_ID ** 2).mean(axis=0))
top256_p = np.argsort(P_prof)[::-1][:256]
sig2802 = z2802['sig'].astype(bool)
votes2802 = np.abs(
    z2802['votes'].astype(np.float64))


def overlap_null(seed, ref_set, obs):
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(N_PERM):
        ps = np.argsort(rng.random(2560))[:256]
        if len(set(ps.tolist())
               & set(ref_set.tolist())) >= obs:
            cnt += 1
    return (float(cnt) + 1.0) / (N_PERM + 1.0)


ov_s16 = int(len(set(top256_p.tolist())
                 & set(S16.tolist())))
ov_t64 = int(len(set(top256_p.tolist())
                 & set(TOP64.tolist())))
ov_coal = int(len(set(top256_p.tolist())
                  & set(COAL_TOP64.tolist())))
ov_sig = int(sig2802[top256_p].sum())
p_s16 = overlap_null(9993, S16, ov_s16)
p_t64 = overlap_null(9994, TOP64, ov_t64)
p_coal = overlap_null(9995, COAL_TOP64, ov_coal)
p_sig = overlap_null(9996,
                     np.where(sig2802)[0], ov_sig)


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx @ ry)
                 / np.sqrt((rx @ rx) * (ry @ ry)))


sp_votes = spearman(P_prof, votes2802)
sig_ch = int(sum(
    1 for p in (p_s16, p_t64, p_coal, p_sig)
    if p <= 0.005))
log('T4: top256_P overlap S16=%d (p=%.4f) '
    'TOP64=%d (p=%.4f) COAL64=%d (p=%.4f) '
    'sig2802=%d/%d (p=%.4f); Spearman(P, votes)'
    '=%.4f (sig_ch=%d)'
    % (ov_s16, p_s16, ov_t64, p_t64, ov_coal,
       p_coal, ov_sig, int(sig2802.sum()), p_sig,
       sp_votes, sig_ch))

# ---------- T5: support distinctness ----------
tsets = [set(np.argsort(np.abs(B_ID[b]))[::-1]
             [:256].tolist())
         for b in range(8)]
jacs = []
for b1 in range(8):
    for b2 in range(b1 + 1, 8):
        inter = len(tsets[b1] & tsets[b2])
        uni = len(tsets[b1] | tsets[b2])
        jacs.append(inter / uni)
jacc_med = float(np.median(jacs))
rng7 = np.random.default_rng(9997)
jacc_null = np.zeros(N_PERM)
for i in range(N_PERM):
    idx = rng7.choice(2560, 512, replace=False)
    s1 = set(idx[:256].tolist())
    s2 = set(idx[256:].tolist())
    jacc_null[i] = len(s1 & s2) / len(s1 | s2)
p_jacc = float((jacc_null >= jacc_med).sum()
               + 1) / (N_PERM + 1)
log('T5: per-body top-256 Jaccard med=%.4f '
    '(null med=%.4f, p=%.4f); norms range '
    '%.3f-%.3f'
    % (jacc_med, float(np.median(jacc_null)),
       p_jacc, float(norm_bid.min()),
       float(norm_bid.max())))

# ---------- verdict ----------
if count_self >= 6 and p_diag <= 0.005:
    verdict = 'body_identity_selftoken_qwen'
elif (count_margin >= 6 and sig_ch >= 1
      and p_diag <= 0.005):
    verdict = 'body_identity_channel_semantic_qwen'
elif count_margin >= 6:
    verdict = 'body_identity_decoded_unplaced_qwen'
else:
    verdict = 'body_identity_opaque_qwen'
log('VERDICT: %s (count_self=%d count_margin=%d '
    'sig_ch=%d p_diag=%.4f)'
    % (verdict, count_self, count_margin,
       sig_ch, p_diag))

anchor_core_ok = bool(a212_ok and a213_ok
                      and a214_ok and a215_ok
                      and a216_ok and a217_ok
                      and a218_ok and a219_ok)
log('anchors core ok=%s' % anchor_core_ok)

elapsed = time.time() - t0
npz_path = os.path.join(OUT, NAME + '.npz')
np.savez(npz_path,
         B_ID=B_ID,
         DBAR=d_bar,
         NORM_BID=norm_bid,
         LOGITS_ID=logits_ID,
         TOP15=top15.astype(np.int64),
         TOP8_DBAR=top8_dbar.astype(np.int64),
         TOP8_RAW=top8_raw.astype(np.int64),
         TOK_IDS=tok_ids,
         TOK_N=tok_n,
         MARGIN=margin_obs,
         MARGIN_P=p_margin,
         SELF_LOGIT=self_obs,
         SELF_PCT=self_pct,
         SELF_P=p_self,
         SPEC=spec_obs,
         MARGINS_U=margins_u,
         SELF_U=self_u,
         M_CPL=M,
         DIAG_MEAN=np.float64(diag_obs),
         DIAG_OFF=np.float64(off_obs),
         DIAG_NULL=diag_null,
         DIAG_P=np.float64(p_diag),
         PROFILE_P=P_prof,
         TOP256_P=top256_p.astype(np.int64),
         OV_S16=np.int64(ov_s16),
         OV_TOP64=np.int64(ov_t64),
         OV_COAL=np.int64(ov_coal),
         OV_SIG=np.int64(ov_sig),
         NULL_S16_P=np.float64(p_s16),
         NULL_TOP64_P=np.float64(p_t64),
         NULL_COAL_P=np.float64(p_coal),
         NULL_SIG_P=np.float64(p_sig),
         SPEAR_P_VOTES=np.float64(sp_votes),
         JACC_MED=np.float64(jacc_med),
         JACC_NULL=jacc_null,
         JACC_P=np.float64(p_jacc),
         COAL_TOP64=COAL_TOP64,
         COUNT_SELF=np.int64(count_self),
         COUNT_MARGIN=np.int64(count_margin),
         SIG_CH=np.int64(sig_ch),
         a212_ok=np.bool_(a212_ok),
         a213_diff=np.float64(a213_diff),
         a214_diff=np.float64(a214_diff),
         a215_diff=np.float64(a215_diff),
         a216_diff=np.float64(a216_diff),
         a217_diff=np.float64(a217_diff),
         a218_diff=np.float64(a218_diff),
         a219_diff=np.float64(a219_diff),
         anchor_core_ok=np.bool_(
             anchor_core_ok),
         verdict=np.array(verdict),
         elapsed=np.float64(elapsed))

stats = {
    'T2_vocab_decode': {
        'margin_obs': [float(v)
                       for v in margin_obs],
        'p_margin': [float(v)
                     for v in p_margin],
        'count_margin': count_margin,
        'self_pct': [float(v)
                     for v in self_pct],
        'p_self': [float(v)
                   for v in p_self],
        'spec_obs': [float(v)
                     for v in spec_obs],
        'count_self': count_self,
        'tok_ids': [int(v) for v in tok_ids],
        'tok_n': [int(v) for v in tok_n]},
    'T3_write_read_coupling': {
        'diag_mean': diag_obs,
        'off_mean': off_obs,
        'p_diag': p_diag},
    'T4_channel_identity': {
        'ov_s16': ov_s16, 'p_s16': p_s16,
        'ov_top64': ov_t64, 'p_top64': p_t64,
        'ov_coal': ov_coal, 'p_coal': p_coal,
        'ov_sig': ov_sig, 'p_sig': p_sig,
        'sig2802_total': int(sig2802.sum()),
        'spearman_p_votes': sp_votes,
        'sig_ch': sig_ch},
    'T5_support_distinctness': {
        'jaccard_med': jacc_med,
        'jaccard_null_med':
            float(np.median(jacc_null)),
        'p_jaccard': p_jacc},
    'anchors': {'a212_seals_ok': a212_ok,
                'a213_tt_diff': a213_diff,
                'a214_gamma_stats_diff':
                    a214_diff,
                'a215_top64_diff': a215_diff,
                'a216_dtan_diff': a216_diff,
                'a217_bbody_pc1_diff':
                    a217_diff,
                'a218_vmat_sv_diff':
                    a218_diff,
                'a219_coal_diff': a219_diff,
                'anchor_core_ok':
                    anchor_core_ok},
}
result = {'phase': PHASE, 'name': NAME,
          'created': created, 'elapsed': elapsed,
          'run': 'run3 authoritative (fp32 '
                 'weights-only, zero forwards; '
                 'arrays from z48/z55/z58/z59/'
                 'z60/z61/z22/z2802 npzs; W_U '
                 'identity proven by a218 Vmat '
                 'bit recompute; rotation null '
                 'uses linearity of logits in '
                 'the direction)',
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
