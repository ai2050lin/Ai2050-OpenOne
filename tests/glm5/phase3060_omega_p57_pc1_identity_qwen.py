# -*- coding: utf-8 -*-
# Phase 3060 - Omega-P57: identity of the main
# readout axis (PC1). 3059 showed the 24 gamma-
# weighted readout preimages v_k = gamma*w_t form
# a nearly one-dimensional family (d_eff = 1.34)
# whose energy concentrates in the S16 channel
# support (med E16 = 0.518 vs null 0.144, p =
# 0.0005). Question: WHAT is the shared axis?
# T2 (offline): shared-component decomposition -
# mean v_bar, cos(PC1, v_bar); per-body sub-
# families (8 bodies x 3 prefix conditions) ->
# d_eff_b and cos(PC1_b, PC1_global) + leave-one-
# body-out stability -> why do eight bodies share
# one axis. T3 (MAIN, offline): PC1 channel
# identity - energy of PC1 in S16/S64 vs
# permutation nulls (seeds 9983/9984); overlap of
# top-|PC1| channels with S16/S64 vs nulls;
# Spearman(|PC1|, |votes2802|) + top-256 sig
# enrichment. T4 (offline): token-space decode -
# logits_dir = W_U @ PC1, top-15 tokens;
# cos(PC1, w_t) spectrum. T5 (offline): write-
# side axis - SVD of the z59 D_TAN family ->
# d_eff_dtan + cos(PC1_dtan, PC1_readout).
# verdict (single branch): min_b cos(PC1_b, PC1)
# >= 0.9 AND p_S16 < 0.05 -> pc1_shared_piped_
# qwen; min_b >= 0.9 -> pc1_shared_loose_qwen;
# else -> pc1_fragmented_qwen.
# anchors: a198 source seals 3044-3059; a199 TT
# recompute bit 0.0 vs z48; a200 gamma stats bit
# 0.0 vs z55; a201 TOP64/FLAT64 bit 0.0 vs z58;
# a202 Vmat recompute bit 0.0 vs z59; a203 SV/
# d_eff bit 0.0 vs z59; a204 D_TAN bit 0.0 vs
# z59.
import os
import json
import time
import hashlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHASE = 3060
NAME = 'omega_p57_pc1_identity_qwen'
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
BODIES = (
    'The weather was cold, so',
    'He studied every night because',
    'The experiment failed, therefore',
    'He missed the train, however',
    'The garden grows quickly while',
    'The price was high, yet',
    'She speaks French, although',
    'The road was closed, thus',)
TARGETS = ('so', 'because', 'therefore', 'however',
           'while', 'yet', 'although', 'thus')
PREFIXES = ('', 'In a formal style,',
            'In Shakespearean style,',
            'Regarding the weather,')

PREREG = {
    'mode': 'fp32 MODEL weights only (torch.float32, '
            'seed 3010, zero forwards); all arrays '
            'from the z48/z55/z58/z59/z2802 npzs; '
            'statistics on the 24 old pairs',
    'question': '3060 A main line: WHAT is the main '
                'readout axis PC1 of the 24 preimage '
                'family (3059: d_eff = 1.34, S16 '
                'support)? Is it a shared cross-body '
                'axis, what channels/token directions '
                'does it live in, is it the same axis '
                'as the d_tan write-side PC1, and '
                'does it relate to the 2802 vote '
                'channels?',
    'T2_shared_decomposition': 'offline: v_bar = '
                               'mean(Vmat); cos(Vt[0], '
                               'v_bar); per-body sub-'
                               'family (rows c*8+b, c '
                               'in 0..2) d_eff_b + '
                               'cos(PC1_b, PC1); leave-'
                               'one-body-out PC1 '
                               'stability',
    'T3_pc1_channels': 'MAIN (offline): E16_pc1 = '
                       '||PC1[S16]||^2, E64_pc1 vs '
                       '16/64-of-2560 permutation '
                       'nulls (seeds 9983/9984, R = '
                       '2000, one-sided); top-256 '
                       '|PC1| channels overlap with '
                       'S16/S64 vs nulls; '
                       'Spearman(|PC1|, |votes|) + '
                       'top-256 |PC1| enrichment of '
                       '2802 sig channels vs null '
                       '(seed 9985)',
    'T4_token_decode': 'offline: logits_dir = W_U @ '
                       'PC1 (151936,); top-15 token '
                       'decode; cos(PC1, w_t) '
                       'spectrum per pair + per '
                       'body',
    'T5_write_side_axis': 'offline: SVD of D_TAN '
                          '(24 x 2560, z59) -> '
                          'd_eff_dtan + cos(PC1_dtan, '
                          'PC1_readout); also cos of '
                          'the two family means',
    'verdict': 'min_b cos(PC1_b, PC1_global) >= 0.9 '
               'AND p_E16 < 0.05 -> '
               'pc1_shared_piped_qwen; min_b >= 0.9 '
               '-> pc1_shared_loose_qwen; else -> '
               'pc1_fragmented_qwen; single branch '
               'assigned inside the criteria',
    'anchors': 'a198 source seals 3044-3059 (result.'
               'json sha256_8 vs seal.json); a199 TT '
               'recompute bit 0.0 vs z48; a200 gamma '
               'stats bit 0.0 vs z55 GAMMA_STATS; '
               'a201 TOP64/FLAT64 recompute bit 0.0 '
               'vs z58; a202 Vmat recompute bit 0.0 '
               'vs z59; a203 SV/d_eff bit 0.0 vs '
               'z59; a204 D_TAN bit 0.0 vs z59',
    'statistics_discipline': 'all vectors in the '
                             'same gamma-weighted '
                             'channel space; nulls are '
                             'channel-set permutations '
                             'on frozen seeds; verdict '
                             'in one branch',
    'corrections': 'run1 (30s) crashed pre-verdict '
                   'at the T4 per-body block: '
                   'wud.T @ TT[rows] tried to matmul '
                   '(2560, V) against (3, V) - the '
                   'TT row slice must be transposed '
                   'first. Fix: transpose. run2 '
                   '(32s) completed all measurements '
                   'and printed the verdict but '
                   'crashed at result.json '
                   'serialization (sig256 is '
                   'numpy.int64 -> int) AND the '
                   'verdict criteria were sign-'
                   'naive: SVD sign is arbitrary, '
                   'per-body/LOO cosines of '
                   '-0.998/-1.0 are |cos| = 1 (a '
                   'perfectly shared axis), so the '
                   'criteria must use |cos|; '
                   'sign-fragile fragment verdict is '
                   'invalid. Fix: abs() in criteria, '
                   'int(sig256). run3 authoritative '
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
log('z48/z55/z58/z59 banks loaded')

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
               'qwen')):
    d = os.path.join(BASE, 'phase%d' % ph, nm)
    with open(os.path.join(d, 'seal.json'),
              encoding='utf-8') as f:
        sealj = json.load(f)
    with open(os.path.join(d, 'result.json'),
              'rb') as f:
        s = hashlib.sha256(f.read()).hexdigest()[:8]
    seal_detail.append(bool(
        s == sealj['result_sha256_8']))
a198_ok = bool(seal_detail) and all(seal_detail)
log('a198 source seals ok=%s' % a198_ok)

# a199: TT recompute (cond-major, prompt = b*4+c)
LGd = np.stack([LG[b * 4 + c] - LG[b * 4]
                for c in (1, 2, 3)
                for b in range(8)])
a199_diff = float(np.max(np.abs(LGd - TT)))
a199_ok = bool(a199_diff == 0.0)
log('a199 TT recompute diff=%.3e ok=%s'
    % (a199_diff, a199_ok))

# ---------- gamma / weights ----------
GAMMA_T = model.model.norm.weight.detach().clone()
GAMMA = GAMMA_T.double().cpu().numpy()
assert GAMMA.shape == (2560,)
gm_mean = float(GAMMA.mean())

gs_now = np.array([GAMMA.min(), GAMMA.max(),
                   GAMMA.mean(), GAMMA.std()])
a200_diff = float(np.max(np.abs(
    gs_now - z55['GAMMA_STATS'])))
a200_ok = bool(a200_diff == 0.0)
log('a200 gamma stats diff=%.3e ok=%s'
    % (a200_diff, a200_ok))

order_shape = np.argsort(
    np.abs(GAMMA - gm_mean))[::-1]
TOP64 = order_shape[:64].astype(np.int64)
FLAT64 = order_shape[::-1][:64].astype(np.int64)
S16 = TOP64[:16].astype(np.int64)
a201_diff = max(
    float(np.max(np.abs(TOP64
                        - z58['TOP64']))),
    float(np.max(np.abs(FLAT64
                        - z58['FLAT64']))))
a201_ok = bool(a201_diff == 0.0)
log('a201 TOP64/FLAT64 recompute diff=%.3e ok=%s'
    % (a201_diff, a201_ok))

WU = model.lm_head.weight.detach()  # (V, 2560)
wud = WU.double().cpu().numpy()

# a202: Vmat recompute vs z59
NP_ = 24
Vmat = np.zeros((NP_, 2560))
for k in range(NP_):
    w_t = wud.T @ TT[k]
    Vmat[k] = GAMMA * w_t
a202_diff = float(np.max(np.abs(Vmat
                                - z59['Vmat'])))
a202_ok = bool(a202_diff == 0.0)
log('a202 Vmat recompute diff=%.3e ok=%s'
    % (a202_diff, a202_ok))

# a203: SVD recompute vs z59
U, sv, Vt = np.linalg.svd(Vmat,
                          full_matrices=False)
d_eff = float(sv.sum() ** 2 / (sv ** 2).sum())
a203_diff = max(
    float(np.max(np.abs(sv - z59['SV']))),
    abs(d_eff - float(z59['D_EFF'])))
a203_ok = bool(a203_diff == 0.0)
log('a203 SV/d_eff recompute diff=%.3e ok=%s '
    '(d_eff=%.4f)' % (a203_diff, a203_ok, d_eff))

# a204: D_TAN bit vs z59
D_TAN = z59['D_TAN']
assert D_TAN.shape == (NP_, 2560)
a204_diff = float(np.max(np.abs(
    D_TAN - z59['D_TAN'])))
a204_ok = bool(a204_diff == 0.0)
log('a204 D_TAN bit vs z59 diff=%.3e ok=%s'
    % (a204_diff, a204_ok))


def cos0(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b) / (na * nb)


PC1 = Vt[0]

# ---------- T2: shared decomposition ----------
v_bar = Vmat.mean(axis=0)
cos_pc1_vbar = cos0(PC1, v_bar)
sv_share0 = float(sv[0] ** 2 / (sv ** 2).sum())
log('T2: cos(PC1, v_bar)=%.4f; PC1 var share='
    '%.4f' % (cos_pc1_vbar, sv_share0))
# per-body sub-families (rows c*8+b)
bidx = np.array([b for c in (1, 2, 3)
                 for b in range(8)])
cidx = np.array([c for c in (1, 2, 3)
                 for b in range(8)])
d_eff_b = np.zeros(8)
cos_pc1_b = np.zeros(8)
cos_vbar_b = np.zeros(8)
for b in range(8):
    rows = np.where(bidx == b)[0]
    Vb = Vmat[rows]
    _, svb, Vtb = np.linalg.svd(
        Vb, full_matrices=False)
    d_eff_b[b] = (svb.sum() ** 2
                  / (svb ** 2).sum())
    cos_pc1_b[b] = cos0(Vtb[0], PC1)
    cos_vbar_b[b] = cos0(
        Vb.mean(axis=0), v_bar)
log('T2 per-body: d_eff_b=%s; cos(PC1_b,PC1)=%s; '
    'cos(vbar_b,vbar)=%s'
    % (np.array2string(d_eff_b, precision=2),
       np.array2string(cos_pc1_b, precision=3),
       np.array2string(cos_vbar_b, precision=3)))
min_pc1_b = float(np.abs(cos_pc1_b).min())
# leave-one-body-out PC1 stability
loo_cos = np.zeros(8)
for b in range(8):
    rows = np.where(bidx != b)[0]
    _, _, Vtb = np.linalg.svd(
        Vmat[rows], full_matrices=False)
    loo_cos[b] = cos0(Vtb[0], PC1)
min_loo = float(np.abs(loo_cos).min())
log('T2 leave-one-body-out cos(PC1_b,PC1)=%s '
    '(min=%.4f)'
    % (np.array2string(loo_cos, precision=3),
       min_loo))

# ---------- T3: PC1 channel identity ----------
e16_pc1 = float((PC1[S16] ** 2).sum())
e64_pc1 = float((PC1[TOP64] ** 2).sum())
rng1 = np.random.default_rng(9983)
pm1 = np.argsort(
    rng1.random((N_PERM, 2560)), axis=1)[:, :16]
null16 = np.array([
    float((PC1[pm] ** 2).sum()) for pm in pm1])
p_e16 = float((null16 >= e16_pc1).sum()
              + 1) / (N_PERM + 1)
rng2 = np.random.default_rng(9984)
pm2 = np.argsort(
    rng2.random((N_PERM, 2560)), axis=1)[:, :64]
null64 = np.array([
    float((PC1[pm] ** 2).sum()) for pm in pm2])
p_e64 = float((null64 >= e64_pc1).sum()
              + 1) / (N_PERM + 1)
log('T3: E16_pc1=%.4f vs null med=%.4f (p=%.4f); '
    'E64_pc1=%.4f vs null med=%.4f (p=%.4f)'
    % (e16_pc1, float(np.median(null16)), p_e16,
       e64_pc1, float(np.median(null64)), p_e64))
# top-|PC1| channel overlap with S16/S64
top256 = np.argsort(np.abs(PC1))[::-1][:256]
ov16 = int(len(set(top256.tolist())
               & set(S16.tolist())))
ov64 = int(len(set(top256.tolist())
               & set(TOP64.tolist())))
rng3 = np.random.default_rng(9985)
pset = np.argsort(
    rng3.random((N_PERM, 2560)), axis=1)[:, :256]
ov16_null = np.array([float(len(
    set(ps.tolist()) & set(S16.tolist())))
    for ps in pset])
ov64_null = np.array([float(len(
    set(ps.tolist()) & set(TOP64.tolist())))
    for ps in pset])
p_ov16 = float((ov16_null >= ov16).sum()
               + 1) / (N_PERM + 1)
p_ov64 = float((ov64_null >= ov64).sum()
               + 1) / (N_PERM + 1)
log('T3 overlap top256 vs S16: %d (null med=%.1f, '
    'p=%.4f); vs TOP64: %d (null med=%.1f, p=%.4f)'
    % (ov16, float(np.median(ov16_null)), p_ov16,
       ov64, float(np.median(ov64_null)), p_ov64))
# 2802 vote-channel relation
z2802 = np.load(os.path.join(
    BASE, 'phase2802', 'qwen4_polysemy_spectrum',
    'polysemy.npz'), allow_pickle=True)
votes = z2802['votes'].astype(np.float64)
sig = z2802['sig'].astype(bool)
assert votes.shape == (2560,)
from scipy.stats import spearmanr
rho_pc1_votes = float(spearmanr(
    np.abs(PC1), np.abs(votes)).statistic)
sig256 = sig[top256].sum()
sig_expect = float(256 * sig.sum() / 2560)
sig_null = np.array([float(
    sig[pset[i]].sum())
    for i in range(N_PERM)])
p_sig = float((sig_null >= sig256).sum()
              + 1) / (N_PERM + 1)
log('T3 votes: spearman(|PC1|,|votes|)=%.4f; '
    'top256 sig=%d vs expect=%.1f (p=%.4f)'
    % (rho_pc1_votes, sig256, sig_expect, p_sig))

# ---------- T4: token decode ----------
logits_dir = wud @ PC1
top_tok = np.argsort(logits_dir)[::-1][:15]
tok_strs = [tok.decode([int(t)]).strip()
            for t in top_tok]
log('T4 top-15 tokens of W_U@PC1: %s'
    % str(tok_strs))
cos_pc1_wt = np.array([
    cos0(PC1, wud.T @ TT[k]) for k in range(NP_)])
med_cos_pc1_wt = float(np.median(cos_pc1_wt))
# per-body mean w_t alignment
cos_pc1_wbar_b = np.zeros(8)
for b in range(8):
    rows = np.where(bidx == b)[0]
    wbar = (wud.T @ TT[rows].T).mean(axis=1)
    cos_pc1_wbar_b[b] = cos0(PC1, wbar)
log('T4: med cos(PC1, w_t)=%.4f; per-body '
    'cos(PC1, wbar_b)=%s'
    % (med_cos_pc1_wt,
       np.array2string(cos_pc1_wbar_b,
                       precision=3)))

# ---------- T5: write-side axis ----------
Ud, svd_, Vtd = np.linalg.svd(
    D_TAN, full_matrices=False)
d_eff_dtan = float(
    (svd_.sum() ** 2) / (svd_ ** 2).sum())
cos_pc1_pair = cos0(Vtd[0], PC1)
d_bar = D_TAN.mean(axis=0)
cos_dtan_pc1 = cos0(d_bar, PC1)
log('T5: d_eff_dtan=%.2f (svd[0..3]=%s); '
    'cos(PC1_dtan, PC1_readout)=%.4f; '
    'cos(dbar, PC1)=%.4f'
    % (d_eff_dtan,
       np.array2string(svd_[:4], precision=2),
       cos_pc1_pair, cos_dtan_pc1))

# ---------- verdict ----------
consistent = bool(min_pc1_b >= 0.9
                  and min_loo >= 0.9)
enriched = bool(p_e16 < 0.05 and p_e64 < 0.05)
if consistent and enriched:
    verdict = 'pc1_shared_piped_qwen'
elif consistent:
    verdict = 'pc1_shared_loose_qwen'
else:
    verdict = 'pc1_fragmented_qwen'
log('VERDICT: %s' % verdict)

anchor_core_ok = bool(a198_ok and a199_ok
                      and a200_ok and a201_ok
                      and a202_ok and a203_ok
                      and a204_ok)
log('anchors core ok=%s' % anchor_core_ok)

elapsed = time.time() - t0
npz_path = os.path.join(OUT, NAME + '.npz')
np.savez(npz_path,
         COS_PC1_VBAR=np.float64(cos_pc1_vbar),
         SV_SHARE0=np.float64(sv_share0),
         D_EFF_B=d_eff_b,
         COS_PC1_B=cos_pc1_b,
         COS_VBAR_B=cos_vbar_b,
         LOO_COS=loo_cos,
         E16_PC1=np.float64(e16_pc1),
         E64_PC1=np.float64(e64_pc1),
         NULL16=null16,
         NULL64=null64,
         TOP256=top256,
         OV16=np.float64(ov16),
         OV64=np.float64(ov64),
         OV16_NULL=ov16_null,
         OV64_NULL=ov64_null,
         RHO_PC1_VOTES=np.float64(
             rho_pc1_votes),
         SIG256=np.float64(sig256),
         SIG_NULL=sig_null,
         LOGITS_TOP15=np.array(
             ['%d:%s' % (int(t), s)
              for t, s in zip(top_tok,
                              tok_strs)]),
         COS_PC1_WT=cos_pc1_wt,
         COS_PC1_WBAR_B=cos_pc1_wbar_b,
         D_EFF_DTAN=np.float64(d_eff_dtan),
         COS_PC1_PAIR=np.float64(
             cos_pc1_pair),
         COS_DBAR_PC1=np.float64(cos_dtan_pc1),
         PC1=PC1,
         a198_ok=np.bool_(a198_ok),
         a199_diff=np.float64(a199_diff),
         a200_diff=np.float64(a200_diff),
         a201_diff=np.float64(a201_diff),
         a202_diff=np.float64(a202_diff),
         a203_diff=np.float64(a203_diff),
         a204_diff=np.float64(a204_diff),
         anchor_core_ok=np.bool_(
             anchor_core_ok),
         verdict=np.array(verdict),
         elapsed=np.float64(elapsed))

stats = {
    'T2_shared_decomposition': {
        'cos_pc1_vbar': cos_pc1_vbar,
        'pc1_var_share': sv_share0,
        'd_eff_b': [float(v) for v in d_eff_b],
        'cos_pc1_b': [float(v)
                      for v in cos_pc1_b],
        'min_cos_pc1_b': min_pc1_b,
        'min_loo_cos': min_loo},
    'T3_pc1_channels': {
        'E16_pc1': e16_pc1,
        'null16_med': float(np.median(null16)),
        'p_E16': p_e16,
        'E64_pc1': e64_pc1,
        'null64_med': float(np.median(null64)),
        'p_E64': p_e64,
        'ov16': ov16,
        'ov16_null_med':
            float(np.median(ov16_null)),
        'p_ov16': p_ov16,
        'ov64': ov64,
        'ov64_null_med':
            float(np.median(ov64_null)),
        'p_ov64': p_ov64,
        'rho_pc1_votes': rho_pc1_votes,
        'sig256': int(sig256),
        'sig_expect': sig_expect,
        'p_sig': p_sig},
    'T4_token_decode': {
        'top15_tokens': tok_strs,
        'med_cos_pc1_wt': med_cos_pc1_wt,
        'cos_pc1_wbar_b': [float(v) for v in
                           cos_pc1_wbar_b]},
    'T5_write_side_axis': {
        'd_eff_dtan': d_eff_dtan,
        'cos_pc1_dtan_vs_readout': cos_pc1_pair,
        'cos_dbar_pc1': cos_dtan_pc1},
    'anchors': {'a198_seals_ok': a198_ok,
                'a199_tt_diff': a199_diff,
                'a200_gamma_stats_diff':
                    a200_diff,
                'a201_top64_diff': a201_diff,
                'a202_vmat_diff': a202_diff,
                'a203_sv_diff': a203_diff,
                'a204_dtan_diff': a204_diff,
                'anchor_core_ok':
                    anchor_core_ok},
}
result = {'phase': PHASE, 'name': NAME,
          'created': created, 'elapsed': elapsed,
          'run': 'run3 authoritative (fp32 '
                 'weights-only, zero forwards; '
                 'arrays from z48/z55/z58/z59/'
                 'z2802 npzs; PC1/SVD recomputed '
                 'and bit-anchored vs z59)',
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
