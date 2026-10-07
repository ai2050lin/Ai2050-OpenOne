# -*- coding: utf-8 -*-
# Phase 3056 - Omega-P53: gamma pre-alignment
# direction-spectrum inventory. 3055 proved the
# final-norm gamma channel assignment causally
# carries the readout pre-alignment (permute
# 0.6805 -> 0.2031, ones -> 0.1899) with a
# TARGET-SPECIFIC gain (t-gain 0.0843 vs random
# -0.0002). Question: WHICH vocab/semantic
# directions are pre-aligned - inventory the
# combined readout basis gamma*W_U.
# T2 token gain spectrum (offline, weights only):
# g_v = ||gamma*W_U[v,:]|| / ||W_U[v,:]|| for all
# vocab tokens (channel re-weighting gain per
# token readout row); distribution + top/bottom
# decoded (descriptive).
# T3 target-direction mass enrichment (MAIN):
# the 24 target logit directions t_k (z48 TT);
# stat = med_k mass_top(k) with mass_top(k) =
# sum_{v in topG} t_k[v]^2 / sum_v t_k[v]^2,
# G = 200 top-gain tokens; null R = 2000 random
# 200-token sets (seed 9961), one-sided p;
# negative control: bottom-200 tokens.
# T4 2802 voting-spectrum channel overlap:
# z2802 votes (2560,) signed per-channel votes
# + sig mask; Spearman(|gamma|, |votes|) +
# enrichment of sig channels among top-gamma
# channels (permutation test R = 2000).
# T5 cross-layer norm gamma profiles:
# gamma_final vs 72 layer RMSNorm gammas
# (input_layernorm + post_attention_layernorm
# per layer) cos by depth; corr(gamma,
# colstd(W_U)) Pearson + Spearman (does gamma
# amplify high-variance readout channels).
# verdict: p_top < 0.05 AND med_mass_top >
# med_mass_bot -> gamma_vocab_prealigned_qwen;
# p_top < 0.05 -> gamma_vocab_mixed_qwen;
# else -> gamma_vocab_null_qwen.
# anchors: a116d source seals 3044-3055; a174
# z55 T2 replay (medians recomputed from the
# z55 npz arrays bit-equal to the stored
# result.json numbers); a175 gamma stats
# recomputed from model weights bit-equal to
# z55 GAMMA_STATS; a176 TT recomputed from z48
# LG diffs bit 0.0.
import os
import json
import time
import hashlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHASE = 3056
NAME = 'omega_p53_gamma_spectrum_qwen'
BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase%d' % PHASE, NAME)
LOG = os.path.join(OUT, 'run_log.txt')
MODEL_DIR = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL = 36
NVOC = 151936
SEED_MAIN = 3010
G_TOP = 200
R_NULL = 2000
SEED_NULL = 9961
R_ENR = 2000
SEED_ENR = 9962
N_ENR = 256

PREREG = {
    'mode': 'fp32 MODEL (torch.float32, eager, '
            'seed 3010) loaded WEIGHTS-ONLY - this '
            'phase runs zero forwards; all '
            'quantities are offline from the model '
            'weights, the phase3048 npz (TT/LG), '
            'the phase3055 npz (T2 arrays + '
            'GAMMA_STATS) and the phase2802 npz '
            '(votes/sig); chain anchors a174/a175/'
            'a176',
    'question': '3056 A main line: WHICH vocab / '
                'semantic directions does the '
                'final-norm gamma pre-align (3055: '
                'channel assignment is the payload, '
                'gain target-specific)? Inventory '
                'the combined readout basis '
                'gamma*W_U: token gain spectrum, '
                'mass enrichment of the 24 target '
                'directions on top-gain tokens, '
                'overlap with the 2802 signed vote '
                'spectrum, cross-layer norm gamma '
                'profiles',
    'T2_token_gain': 'g_v = ||gamma*W_U[v,:]|| / '
                     '||W_U[v,:]|| over all '
                     'NVOC tokens (GPU chunked); '
                     'distribution stats; top-30 '
                     'and bottom-30 tokens '
                     'decoded (descriptive)',
    'T3_mass_enrich': 'MAIN: t_k from z48 TT '
                      '(24 pairs); mass_top(k) = '
                      'sum of t_k[v]^2 over the '
                      'G_TOP = 200 highest-g_v '
                      'tokens / total t_k[v]^2; '
                      'stat = med_k mass_top(k); '
                      'null R = 2000 random '
                      '200-token sets (seed 9961), '
                      'one-sided p = (1 + #null >= '
                      'stat) / (1 + R); negative '
                      'control: bottom-200 tokens '
                      'mass; verdict: p_top < 0.05 '
                      'AND stat_top > stat_bot -> '
                      'gamma_vocab_prealigned_qwen; '
                      'p_top < 0.05 -> '
                      'gamma_vocab_mixed_qwen; '
                      'else -> gamma_vocab_null_qwen',
    'T4_vote_overlap': 'z2802 votes (2560,) '
                       'signed per-channel votes + '
                       'sig (2560,) bool; Spearman '
                       'corr(|gamma|, |votes|); '
                       'enrichment: mean |votes| '
                       'within the top N_ENR = 256 '
                       'gamma channels vs random '
                       '256-channel sets (R = 2000, '
                       'seed 9962), one-sided p; '
                       'sig-count within top-gamma '
                       'vs expectation '
                       '(descriptive)',
    'T5_layer_profiles': 'gamma_final vs 72 '
                         'layer RMSNorm gammas '
                         '(layers[i].'
                         'input_layernorm.weight '
                         'and .post_attention_'
                         'layernorm.weight): '
                         'plain cos and '
                         'mean-centered cos by '
                         'depth (descriptive); '
                         'corr(gamma, colstd(W_U)) '
                         'Pearson + Spearman',
    'anchors': 'a116d source seals 3044-3055 '
               '(result.json sha256_8 vs seal.json); '
               'a174 z55 T2 replay: medians of '
               'COS_DTAN_G / COS_DTAN_N / GAIN_U '
               'recomputed from the z55 npz arrays '
               'bit-equal to the stored result.json '
               'values; a175 gamma stats recomputed '
               'from model.model.norm.weight '
               'bit-equal to z55 GAMMA_STATS; a176 '
               'TT recomputed from z48 LG diffs '
               'bit 0.0',
    'statistics_discipline': 'null and observed '
                             'share the same mass '
                             'definition (per-token '
                             'squared logit mass), '
                             'the same 24 targets and '
                             'the same G_TOP set size; '
                             'enrichment null permutes '
                             'channel identity; verdict '
                             'in one branch assigned '
                             'inside the criteria',
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
layers = model.model.layers
assert len(layers) == NL
NVOC = int(model.config.vocab_size)
log('model loaded fp32 weights-only (vocab=%d)' % NVOC)

# ---------- chain banks ----------
z48 = np.load(os.path.join(
    BASE, 'phase3048',
    'omega_p45_kvpos_full_replay_qwen',
    'omega_p45_kvpos_full_replay_qwen.npz'),
    allow_pickle=True)
LG = z48['LG']
TT = z48['TT']
assert TT.shape == (24, NVOC), TT.shape
z55 = np.load(os.path.join(
    BASE, 'phase3055',
    'omega_p52_gamma_prealign_qwen',
    'omega_p52_gamma_prealign_qwen.npz'),
    allow_pickle=True)
z2802 = np.load(os.path.join(
    BASE, 'phase2802', 'qwen4_polysemy_spectrum',
    'polysemy.npz'), allow_pickle=True)
VOTES = z2802['votes'].astype(np.float64)
SIG = z2802['sig']
assert VOTES.shape == (2560,)
log('banks loaded: z48 TT%s LG%s; z55 T2 arrays; '
    'z2802 votes/sig' % (TT.shape, LG.shape))

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
        (3055, 'omega_p52_gamma_prealign_qwen')):
    d = os.path.join(BASE, 'phase%d' % ph, nm)
    with open(os.path.join(d, 'seal.json'),
              encoding='utf-8') as f:
        sealj = json.load(f)
    with open(os.path.join(d, 'result.json'),
              'rb') as f:
        s = hashlib.sha256(f.read()).hexdigest()[:8]
    seal_detail.append(bool(
        s == sealj['result_sha256_8']))
a116_ok = bool(seal_detail) and all(seal_detail)
log('a116d source seals ok=%s' % a116_ok)

# a175: gamma stats from model weights
GAMMA = model.model.norm.weight.detach() \
    .double().cpu().numpy()
assert GAMMA.shape == (2560,)
gs_now = np.array([GAMMA.min(), GAMMA.max(),
                   GAMMA.mean(), GAMMA.std()])
gs_z55 = z55['GAMMA_STATS']
a175_diff = float(np.max(np.abs(gs_now - gs_z55)))
a175_ok = bool(a175_diff == 0.0)
log('a175 gamma stats diff=%.3e ok=%s (min=%.4f '
    'max=%.4f mean=%.4f std=%.4f)'
    % (a175_diff, a175_ok, GAMMA.min(),
       GAMMA.max(), GAMMA.mean(), GAMMA.std()))

# a174: z55 T2 replay
with open(os.path.join(BASE, 'phase3055',
                       'omega_p52_gamma_prealign_'
                       'qwen', 'result.json'),
          encoding='utf-8') as f:
    res55 = json.load(f)
t2_55 = res55['stats']['T2_preimage']
med_g = float(np.median(z55['COS_DTAN_G']))
med_n = float(np.median(z55['COS_DTAN_N']))
gain_r = med_g - med_n
med_gu = float(np.median(z55['GAIN_U']))
p95_gu = float(np.percentile(z55['GAIN_U'], 95))
a174_ok = bool(
    med_g == t2_55['med_cos_dtan_gamma_wt']
    and med_n == t2_55['med_cos_dtan_wt']
    and gain_r == t2_55['gain_t']
    and med_gu == t2_55['med_gain_u']
    and p95_gu == t2_55['pct95_gain_u'])
log('a174 z55 T2 replay ok=%s (med_g=%.6f '
    'med_n=%.6f gain=%.6f med_gu=%.6f p95=%.6f)'
    % (a174_ok, med_g, med_n, gain_r,
       med_gu, p95_gu))

# a176: TT recomputed from z48 LG diffs
# prompt index = body*4 + cond (assembled
# body-major); TT is cond-major stacked
LGd = np.stack([LG[b * 4 + c] - LG[b * 4]
                for c in (1, 2, 3)
                for b in range(8)])
a176_diff = float(np.max(np.abs(LGd - TT)))
a176_ok = bool(a176_diff == 0.0)
log('a176 TT recompute diff=%.3e ok=%s'
    % (a176_diff, a176_ok))


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------- T2 token gain spectrum ----------
WU = model.lm_head.weight.detach()  # (V, 2560)
g_num = torch.zeros(NVOC, dtype=torch.float64)
g_den = torch.zeros(NVOC, dtype=torch.float64)
gam_t = torch.tensor(GAMMA, dtype=torch.float32,
                     device='cuda')
CH = 8192
with torch.no_grad():
    for s in range(0, NVOC, CH):
        e = min(s + CH, NVOC)
        blk = WU[s:e].float()
        g_num[s:e] = (blk * gam_t) \
            .double().pow(2).sum(dim=1).sqrt() \
            .cpu()
        g_den[s:e] = blk.double() \
            .pow(2).sum(dim=1).sqrt().cpu()
GAIN_V = (g_num / g_den).numpy()
log('T2 gain_v: min=%.4f p5=%.4f med=%.4f '
    'p95=%.4f max=%.4f mean=%.4f std=%.4f'
    % (GAIN_V.min(),
       float(np.percentile(GAIN_V, 5)),
       float(np.median(GAIN_V)),
       float(np.percentile(GAIN_V, 95)),
       GAIN_V.max(), GAIN_V.mean(), GAIN_V.std()))
order = np.argsort(GAIN_V)[::-1]
top_idx = order[:G_TOP]
bot_idx = order[::-1][:G_TOP]
top30 = order[:30]
bot30 = order[::-1][:30]
top30_tok = [tok.decode([int(v)]).strip()
             for v in top30]
bot30_tok = [tok.decode([int(v)]).strip()
             for v in bot30]
log('T2 top-30 gain tokens: %s'
    % json.dumps(top30_tok, ensure_ascii=False))
log('T2 bottom-30 gain tokens: %s'
    % json.dumps(bot30_tok, ensure_ascii=False))

# ---------- T3 mass enrichment (MAIN) ----------
TTd = TT.astype(np.float64)
denom = (TTd ** 2).sum(axis=1)


def mass_of(idxs):
    sub = TTd[:, idxs]
    return float(np.median(
        (sub ** 2).sum(axis=1) / denom))


stat_top = mass_of(top_idx)
stat_bot = mass_of(bot_idx)
rng_n = np.random.default_rng(SEED_NULL)
null_stat = np.zeros(R_NULL)
for r in range(R_NULL):
    idxs = rng_n.choice(NVOC, size=G_TOP,
                        replace=False)
    null_stat[r] = mass_of(idxs)
p_top = float((1 + (null_stat >= stat_top).sum())
              / (1 + R_NULL))
log('T3: mass_top=%.6f mass_bot=%.6f null med=%.6f '
    'p95=%.6f p_top=%.5f'
    % (stat_top, stat_bot,
       float(np.median(null_stat)),
       float(np.percentile(null_stat, 95)), p_top))

# ---------- verdict ----------
if p_top < 0.05 and stat_top > stat_bot:
    verdict = 'gamma_vocab_prealigned_qwen'
elif p_top < 0.05:
    verdict = 'gamma_vocab_mixed_qwen'
else:
    verdict = 'gamma_vocab_null_qwen'
log('VERDICT: %s' % verdict)

# ---------- T4 2802 vote overlap ----------
rho_gv = spearman(np.abs(GAMMA), np.abs(VOTES))
top_ch = np.argsort(np.abs(GAMMA))[::-1][:N_ENR]
mean_v_top = float(np.mean(np.abs(VOTES[top_ch])))
mean_v_all = float(np.mean(np.abs(VOTES)))
rng_e = np.random.default_rng(SEED_ENR)
null_mean = np.zeros(R_ENR)
for r in range(R_ENR):
    idxs = rng_e.choice(2560, size=N_ENR,
                        replace=False)
    null_mean[r] = float(np.mean(
        np.abs(VOTES[idxs])))
p_enr = float((1 + (null_mean >= mean_v_top).sum())
              / (1 + R_ENR))
sig_in_top = int(SIG[top_ch].sum())
sig_total = int(SIG.sum())
exp_sig = N_ENR * sig_total / 2560.0
log('T4: spearman(|gamma|,|votes|)=%.4f; '
    'mean|votes| top=%d-ch=%.2f vs all=%.2f '
    'p_enr=%.5f; sig in top=%d (exp %.1f, '
    'total %d)'
    % (rho_gv, N_ENR, mean_v_top, mean_v_all,
       p_enr, sig_in_top, exp_sig, sig_total))

# ---------- T5 cross-layer profiles ----------
COS_L_IN = np.zeros(NL)
COS_L_POST = np.zeros(NL)
COSC_IN = np.zeros(NL)
COSC_POST = np.zeros(NL)
gm = GAMMA - GAMMA.mean()
for li in range(NL):
    for which, attr, arr, carr in (
            (0, 'input_layernorm', COS_L_IN,
             COSC_IN),
            (1, 'post_attention_layernorm',
             COS_L_POST, COSC_POST)):
        g_l = getattr(layers[li], attr).weight \
            .detach().double().cpu().numpy()
        arr[li] = float(g_l @ GAMMA) / (
            np.linalg.norm(g_l)
            * np.linalg.norm(GAMMA))
        cl = g_l - g_l.mean()
        carr[li] = float(cl @ gm) / (
            np.linalg.norm(cl)
            * np.linalg.norm(gm))
with torch.no_grad():
    colstd = WU.float().std(dim=0) \
        .double().cpu().numpy()
r_pear = float(np.corrcoef(GAMMA, colstd)[0, 1])
r_spear = spearman(GAMMA, colstd)
log('T5: cos(gamma, layer in) med=%.4f '
    'range=[%.4f, %.4f]; cos(gamma, layer post) '
    'med=%.4f range=[%.4f, %.4f]; centered '
    'med in=%.4f post=%.4f; corr(gamma, '
    'colstd(W_U)) pear=%.4f spear=%.4f'
    % (float(np.median(COS_L_IN)), COS_L_IN.min(),
       COS_L_IN.max(), float(np.median(COS_L_POST)),
       COS_L_POST.min(), COS_L_POST.max(),
       float(np.median(COSC_IN)),
       float(np.median(COSC_POST)),
       r_pear, r_spear))

anchor_core_ok = bool(a116_ok and a174_ok
                      and a175_ok and a176_ok)
log('anchors core ok=%s' % anchor_core_ok)

elapsed = time.time() - t0
npz_path = os.path.join(OUT, NAME + '.npz')
np.savez(npz_path,
         GAIN_V=GAIN_V,
         ORDER=order,
         TOP_IDX=top_idx,
         BOT_IDX=bot_idx,
         TT_MASS_TOP=np.float64(stat_top),
         TT_MASS_BOT=np.float64(stat_bot),
         NULL_STAT=null_stat,
         P_TOP=np.float64(p_top),
         RHO_GV=np.float64(rho_gv),
         MEAN_V_TOP=np.float64(mean_v_top),
         MEAN_V_ALL=np.float64(mean_v_all),
         NULL_MEAN=null_mean,
         P_ENR=np.float64(p_enr),
         SIG_IN_TOP=np.int64(sig_in_top),
         SIG_TOTAL=np.int64(sig_total),
         COS_L_IN=COS_L_IN,
         COS_L_POST=COS_L_POST,
         COSC_IN=COSC_IN,
         COSC_POST=COSC_POST,
         R_PEAR=np.float64(r_pear),
         R_SPEAR=np.float64(r_spear),
         TOP30_TOK=np.array(top30_tok),
         BOT30_TOK=np.array(bot30_tok),
         a175_diff=np.float64(a175_diff),
         a176_diff=np.float64(a176_diff),
         a116_ok=np.bool_(a116_ok),
         a174_ok=np.bool_(a174_ok),
         a175_ok=np.bool_(a175_ok),
         a176_ok=np.bool_(a176_ok),
         verdict=np.array(verdict),
         elapsed=np.float64(elapsed))

stats = {
    'T2_token_gain': {
        'min': float(GAIN_V.min()),
        'p5': float(np.percentile(GAIN_V, 5)),
        'med': float(np.median(GAIN_V)),
        'p95': float(np.percentile(GAIN_V, 95)),
        'max': float(GAIN_V.max()),
        'mean': float(GAIN_V.mean()),
        'std': float(GAIN_V.std())},
    'T3_mass_enrich': {
        'mass_top': stat_top,
        'mass_bot': stat_bot,
        'null_med': float(np.median(null_stat)),
        'null_p95': float(np.percentile(
            null_stat, 95)),
        'p_top': p_top,
        'G_TOP': G_TOP,
        'R_NULL': R_NULL},
    'T4_vote_overlap': {
        'spearman_abs_gamma_abs_votes': rho_gv,
        'mean_abs_votes_top': mean_v_top,
        'mean_abs_votes_all': mean_v_all,
        'p_enr': p_enr,
        'sig_in_top': sig_in_top,
        'sig_expected': exp_sig,
        'sig_total': sig_total},
    'T5_layer_profiles': {
        'med_cos_in': float(np.median(COS_L_IN)),
        'med_cos_post':
            float(np.median(COS_L_POST)),
        'med_cos_centered_in':
            float(np.median(COSC_IN)),
        'med_cos_centered_post':
            float(np.median(COSC_POST)),
        'corr_gamma_colstd_pearson': r_pear,
        'corr_gamma_colstd_spearman': r_spear},
    'anchors': {'a116_seals_ok': a116_ok,
                'a174_z55_replay_ok': a174_ok,
                'a175_gamma_stats_diff':
                    a175_diff,
                'a176_tt_diff': a176_diff,
                'anchor_core_ok':
                    anchor_core_ok},
}
result = {'phase': PHASE, 'name': NAME,
          'created': created, 'elapsed': elapsed,
          'run': 'run1 authoritative (fp32 '
                 'weights-only; zero forwards; '
                 'offline from model weights + z48 '
                 'TT/LG + z55 T2 arrays/GAMMA_STATS '
                 '+ z2802 votes/sig)',
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
