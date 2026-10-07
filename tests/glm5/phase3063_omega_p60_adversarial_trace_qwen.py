# -*- coding: utf-8 -*-
# Phase 3063 - Omega-P60: adversarial component
# tracing. Pending B item since 3052: 3051 found
# V-only = -0.3409 (full-row V replacement at
# L35 x FRONT actively ANTI-aligns with the
# target); 3053/3054 found the h0 exact K field
# is tangential-adversarial (med -0.2492, frac_rad
# 0.131, d_tan through the exact norm still
# negative). Question (2 parts): (1) is the V arm
# negative ALSO tangential adversarial (survives
# the norm) or radial dilution (norm swamp)?
# 3054 T4 only decomposed the h0 K arm.
# (2) SAME-SOURCE test: do the V-arm and h0-K-arm
# adversarial tangential families live in the same
# direction / subspace / channel support?
# Specificity comparators: h3 K arm (positive
# +0.3098, z53) and h7 joint arm (positive, the
# canonical gate arm) - same-source predicts
# V<->h0K alignment >> V<->h3K, V<->h7J.
# T2 (V mechanism): per-pair d = post_i - post_b,
# d_tan/d_rad vs x_b = post_b; cos_WU(d_tan),
# frac_rad, exact-norm counterfactual cos(W_U @
# (norm(x_b+d_tan) - postn_b), t_k) - MAIN metric
# med < 0 -> V tangential adversarial.
# T3 (MAIN same-source): adversarial families
# D_V, D_K0 (24x2560 d_tan vectors); mapped to
# logit space Y = D @ W_U^T (exact G-metric).
# SS1 per-pair: C[k1,k2] = |cos(Y_V[k1],Y_K0[k2])|,
# obs = mean diag - mean offdiag vs pair-label
# permutation (R=2000 seed 9998); SS2 shared axis:
# |cos| of family means + PC1s (Gram-eigh) vs
# random-direction null (R=2000 seed 9999);
# SS3 support: Jaccard(top256 |D_V| pooled,
# top256 |D_K0| pooled) vs 2x256-of-2560 null
# (R=2000 seed 10002). ss_count = #{p<=0.005}.
# T4 specificity: same SS1/SS2 machinery V<->K3
# (seeds 10000/10001 for perm/axis-null reuse)
# and V<->J; axis_dominance = c_pc1_VK0 >=
# c_pc1_VK3 and >= c_pc1_VJ.
# T5 channel identity: top256 pooled profiles of
# both adversarial arms vs S16/TOP64/COAL_TOP64/
# sig2802 overlaps + E16/E64 profile energy, all
# vs 256/16/64-of-2560 nulls (V seeds 10003-10008,
# K0 seeds 10009-10014; 6 nulls per arm).
# T6 known-axis relations: per-pair channel-space
# cos(D_V[k], Vmat[k]) (anti-payload check),
# cos(D_*, PC1_ro), cos(mean D_V, PC1_D) +
# rotation null (seed 10015).
# verdict (single branch): v_adv = med(V_COS_
# DTAN_ONLY) < 0; v_adv and ss_count >= 2 and
# axis_dominance -> adversarial_same_source_qwen;
# v_adv and ss_count >= 2 -> adversarial_same_
# space_qwen; v_adv -> adversarial_arm_local_
# qwen; else -> v_radial_dilution_qwen.
# anchors: a220 source seals 3044-3062; a221 TT
# recompute bit 0.0 vs z48; a222 sampled re-
# capture (4 prompts) bit 0.0 + TT diff 0.0;
# a223 gamma stats bit 0.0 vs z55; a224 TOP64/
# FLAT64 bit 0.0 vs z58; a225 D_TAN bit 0.0 vs
# z59; a226 PC1_ro sign-invariant vs z60 +
# B_BODY/C_PREFIX vs z61; a227 COAL_TOP64 bit
# 0.0 vs z58; a228 V arm cos bit 0.0 vs z51
# COS_V; a229 h0K arm bit 0.0 vs z54 (COS_K0 +
# 4 decomposition scalars); a230 h7J arm bit 0.0
# vs z51 COS_H[7] + z53 STAGE cols 2/3; a231 h3K
# arm bit 0.0 vs z53 COS_KCTRL_h3; a232 sham
# self-replacement bit identity; a233 stage-pre
# identity max diff 0.0 (all arms).
import os
import json
import time
import hashlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHASE = 3063
NAME = 'omega_p60_adversarial_trace_qwen'
BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase%d' % PHASE, NAME)
LOG = os.path.join(OUT, 'run_log.txt')
MODEL_DIR = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL = 36
KV_HEAD = 8
HDIM = 128
SEED_MAIN = 3010
FRONT = 4
L_TGT = 35
HEAD_H = 7
HEAD_H0 = 0
HEAD_H3 = 3
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
    'mode': 'fp32 MODEL (torch.float32, eager, '
            'seed 3010); capture bank from the '
            'phase3048 npz (KPpost/VP/LG/LENS/TT); '
            '4 arms x 24 canonical diagonal pairs '
            're-run with the 3054 forward_stage '
            'machinery (V-only full row, h0 K-only '
            'block, h3 K-only block, h7 joint '
            'block); chain anchors vs z51/z53/z54 '
            'bit 0.0; offline same-source '
            'statistics on captured d_tan vectors',
    'question': '3063 A main line (pending B '
                'since 3052): (1) is the 3051 '
                'V-only -0.3409 negative alignment '
                'tangential-adversarial (survives '
                'the final RMSNorm) or radial '
                'dilution? (2) is the V-arm '
                'adversarial component SAME-SOURCE '
                'with the h0 exact-K tangential '
                'adversarial component (same '
                'direction, subspace, channel '
                'support), with the positive h3-K '
                'and h7-joint arms as specificity '
                'comparators?',
    'T2_v_mechanism': 'per pair k (24 old pairs): '
                      'V-only arm (rV[L35, FRONT '
                      'rows, :] = srcV, K self; = '
                      '3051 T2 arm, a228 bit anchor '
                      'vs z51 COS_V); d = post_i - '
                      'post_b (L35 output, last '
                      'position); xb = post_b; '
                      'd_tan = d - (d.xb/|xb|^2)xb; '
                      'cos_WU(d_tan), frac_rad, '
                      'cos_lg (Jacobian pred '
                      'descriptive); MAIN metric: '
                      'exact-norm counterfactual '
                      'cos(W_U @ (norm(x_b + d_tan) '
                      '- postn_b), t_k); med < 0 -> '
                      'tangential adversarial',
    'T3_same_source': 'MAIN: D_V/D_K0 = 24x2560 '
                      'd_tan families of the V and '
                      'h0K arms (h0K re-run = 3054 '
                      'T4 arm, a229 bit anchor vs '
                      'z54); logit-space mapping '
                      'Y = D @ W_U^T (exact G-'
                      'metric); SS1 per-pair: C = '
                      '|cos| Gram, obs = mean diag '
                      '- mean offdiag, pair-label '
                      'permutation R=2000 seed '
                      '9998; SS2 shared axis: '
                      '|cos| of family means and '
                      'PC1s (24x24 Gram + eigh) vs '
                      'random-direction null R=2000 '
                      'seed 9999; SS3 support: '
                      'Jaccard of top-256 pooled '
                      'channel profiles vs 2x256-'
                      'of-2560 null R=2000 seed '
                      '10002; ss_count = #{p <= '
                      '0.005}',
    'T4_specificity': 'same machinery V<->h3K '
                      '(perm seed 10000) and V<->'
                      'h7J (perm seed 10001); '
                      'axis_dominance = c_pc1_VK0 '
                      '>= c_pc1_VK3 and c_pc1_VK0 '
                      '>= c_pc1_VJ; h0K<->h3K '
                      'descriptive',
    'T5_channel_identity': 'top-256 of pooled '
                           'profiles P = sqrt(mean_k '
                           'D_k^2) per adversarial '
                           'arm; overlaps vs S16/'
                           'TOP64/COAL_TOP64/sig2802 '
                           '+ E16/E64 profile energy, '
                           'all vs 256/16/64-of-2560 '
                           'permutation nulls (V '
                           'seeds 10003-10008, K0 '
                           'seeds 10009-10014)',
    'T6_known_axis': 'channel-space cos(D_V[k], '
                     'Vmat[k]) per pair (anti-'
                     'payload check), cos(D_V[k], '
                     'PC1_ro), cos(D_K0[k], PC1_ro), '
                     'cos(mean D_V, PC1_D) + '
                     'rotation null for med |cos| '
                     '(R=2000 seed 10015)',
    'verdict': 'v_adv = med(V_COS_DTAN_ONLY) < 0; '
               'v_adv and ss_count >= 2 and '
               'axis_dominance -> '
               'adversarial_same_source_qwen; '
               'v_adv and ss_count >= 2 -> '
               'adversarial_same_space_qwen; '
               'v_adv -> '
               'adversarial_arm_local_qwen; '
               'else -> v_radial_dilution_qwen; '
               'single branch assigned inside the '
               'criteria',
    'anchors': 'a220 source seals 3044-3062; a221 '
               'TT recompute bit 0.0; a222 re-'
               'capture 4 prompts bit 0.0 + TT '
               'diff 0.0; a223 gamma stats bit '
               '0.0 vs z55; a224 TOP64/FLAT64 '
               'bit 0.0 vs z58; a225 D_TAN bit '
               '0.0 vs z59; a226 Vmat/PC1_ro/'
               'B_BODY/C_PREFIX recompute bit '
               '0.0 vs z59/z60/z61 (PC sign-'
               'invariant); a227 COAL_TOP64 '
               'bit 0.0 vs z58; a228 V arm bit '
               '0.0 vs z51 COS_V; a229 h0K arm '
               'bit 0.0 vs z54 (COS_K0 + H0_COS_D '
               '+ H0_COS_DTAN + H0_COS_DTAN_ONLY '
               '+ H0_FRAC_RAD); a230 h7J arm bit '
               '0.0 vs z51 COS_H[7] + z53 STAGE '
               'cols 2/3; a231 h3K arm bit 0.0 vs '
               'z53 COS_KCTRL_h3; a232 sham self-'
               'replacement bit identity; a233 '
               'stage-pre identity max diff 0.0',
    'statistics_discipline': 'same-source cosines '
                             'in the exact readout '
                             'metric (logit space via '
                             'W_U mapping; 3061 run2 '
                             'lesson respected - no '
                             'cross-space cosine: '
                             'channel-space cosines '
                             'only between channel '
                             'vectors); PC1 sign '
                             'arbitrary - all PC '
                             'comparisons via |cos|; '
                             'nulls are label '
                             'permutations / random '
                             'directions on frozen '
                             'seeds; verdict in one '
                             'branch',
    'memory_discipline': 'wud fp64 (3.1 GB) held '
                         'only during the arm '
                         'forward loop and the Y = '
                         'D @ W_U^T projections then '
                         'del; Y matrices 4x24x151936 '
                         'fp64 (~233 MB); all '
                         'permutation nulls are '
                         'per-draw small argsorts '
                         '(3061 run4 OOM lesson); dh '
                         'fp64 del after a227',
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
assert int(model.config.num_key_value_heads) == KV_HEAD
assert int(model.config.num_attention_heads) == 32
rot = model.model.rotary_emb
INV_FREQ = rot.inv_freq.detach().cpu().numpy() \
    .astype(np.float32)
assert INV_FREQ.shape == (64,), INV_FREQ.shape
assert float(rot.attention_scaling) == 1.0
NVOC = int(model.config.vocab_size)
EPS_NORM = float(model.config.rms_norm_eps)
log('model loaded fp32 (vocab=%d rms_eps=%g)'
    % (NVOC, EPS_NORM))


def rot_apply(x, delta):
    """Rotate (...,8,128) by delta*inv_freq
    (NeoX half-pairing, fp32)."""
    ang = INV_FREQ * np.float32(delta)
    emb = np.concatenate([ang, ang]).astype(
        np.float32)
    c = np.cos(emb)[None, None, :]
    s = np.sin(emb)[None, None, :]
    x1 = x[..., :64]
    x2 = x[..., 64:]
    rh = np.concatenate([-x2, x1], axis=-1)
    return x * c + rh * s


stateKn = {li: {'repl': None, 'mask': None}
           for li in range(NL)}
stateV = {li: {'repl': None, 'mask': None}
          for li in range(NL)}
capKn = {li: {'rec': False, 'orig': None,
              'mod': None} for li in range(NL)}
capV = {li: {'rec': False, 'orig': None,
             'mod': None} for li in range(NL)}
capKpre = {li: {'rec': False, 'orig': None}
           for li in range(NL)}
cap_stage = {'pre': {'rec': False, 'v': None},
             'op': {'rec': False, 'v': None},
             'post': {'rec': False, 'v': None}}


def hook_norm(st, cp):
    def h(module, inp, out):
        if cp['rec']:
            cp['orig'] = out[0].detach().clone()
        if st['repl'] is not None:
            out[0][st['mask']] = st['repl']
        if cp['rec']:
            cp['mod'] = out[0].detach().clone()
        return out
    return h


def hook_v(st, cp):
    def h(module, inp, out):
        if cp['rec']:
            cp['orig'] = out[0].detach().clone()
        if st['repl'] is not None:
            out[0][st['mask']] = st['repl']
        if cp['rec']:
            cp['mod'] = out[0].detach().clone()
        return out
    return h


def hook_kpre(cp):
    def h(module, inp, out):
        if cp['rec']:
            cp['orig'] = out[0].detach().clone()
        return out
    return h


def hook_stage(cp):
    """Capture the LAST-position vector of a
    bare-tensor output (batch dim stripped by
    indexing [0])."""
    def h(module, inp, out):
        if cp['rec']:
            t = out[0] if isinstance(out, tuple) \
                else out
            cp['v'] = t[0][-1].detach().clone()
        return out
    return h


for li in range(NL):
    layers[li].self_attn.k_norm \
        .register_forward_hook(hook_norm(
            stateKn[li], capKn[li]))
    layers[li].self_attn.v_proj \
        .register_forward_hook(hook_v(
            stateV[li], capV[li]))
    layers[li].self_attn.k_proj \
        .register_forward_hook(hook_kpre(
            capKpre[li]))
layers[34].register_forward_hook(
    hook_stage(cap_stage['pre']))
layers[35].self_attn.o_proj.register_forward_hook(
    hook_stage(cap_stage['op']))
layers[35].register_forward_hook(
    hook_stage(cap_stage['post']))


def reset_all():
    for li in range(NL):
        stateKn[li]['repl'] = None
        stateKn[li]['mask'] = None
        stateV[li]['repl'] = None
        stateV[li]['mask'] = None
        capKn[li]['rec'] = False
        capV[li]['rec'] = False
        capKpre[li]['rec'] = False
    for kk in cap_stage:
        cap_stage[kk]['rec'] = False
        cap_stage[kk]['v'] = None


def forward_cap(ids):
    reset_all()
    for li in range(NL):
        capKn[li]['rec'] = True
        capV[li]['rec'] = True
        capKpre[li]['rec'] = True
    with torch.no_grad():
        out = model(torch.tensor([ids], device='cuda'),
                    use_cache=True)
    lg = out.logits[0, -1].detach().double() \
        .cpu().numpy()
    n = len(ids)
    kp = np.stack([capKpre[li]['orig']
                   .cpu().numpy()
                   for li in range(NL)])
    kn = np.stack([capKn[li]['orig'].reshape(
        n, HDIM * 8).cpu().numpy()
        for li in range(NL)])
    vp = np.stack([capV[li]['orig'].cpu().numpy()
                   for li in range(NL)])
    reset_all()
    return lg, kp, kn, vp


def forward_stage(ids, replK, maskK, replV, maskV):
    """Replacement with stage capture; returns
    logits + last-position stage vectors
    (pre, op, post) and the post-final-norm post
    vector."""
    reset_all()
    rk = torch.tensor(np.ascontiguousarray(
        replK.reshape(NL, -1, 8, HDIM)),
        dtype=torch.float32, device='cuda')
    mk = torch.tensor(np.asarray(maskK),
                      device='cuda')
    rv = torch.tensor(np.ascontiguousarray(
        replV),
        dtype=torch.float32, device='cuda')
    mv = torch.tensor(np.asarray(maskV),
                      device='cuda')
    for li in range(NL):
        stateKn[li]['repl'] = rk[li]
        stateKn[li]['mask'] = mk
        stateV[li]['repl'] = rv[li]
        stateV[li]['mask'] = mv
    for kk in cap_stage:
        cap_stage[kk]['rec'] = True
    with torch.no_grad():
        out = model(torch.tensor([ids], device='cuda'),
                    use_cache=True)
    lg = out.logits[0, -1].detach().double() \
        .cpu().numpy()
    pre = cap_stage['pre']['v'].double() \
        .cpu().numpy()
    op = cap_stage['op']['v'].double() \
        .cpu().numpy()
    post_t = cap_stage['post']['v']
    post = post_t.double().cpu().numpy()
    with torch.no_grad():
        postn_t = model.model.norm(
            post_t.unsqueeze(0))
    postn = postn_t[0].double().cpu().numpy()
    reset_all()
    return lg, pre, op, post, postn


# ---------- chain: load capture banks ----------
z48 = np.load(os.path.join(
    BASE, 'phase3048',
    'omega_p45_kvpos_full_replay_qwen',
    'omega_p45_kvpos_full_replay_qwen.npz'),
    allow_pickle=True)
KPpre = z48['KPpre']
KPpost = z48['KPpost']
VP = z48['VP']
LG = z48['LG']
LENS = z48['LENS']
TT = z48['TT']
log('z48 bank loaded: KPpre%s KPpost%s VP%s LG%s'
    % (KPpre.shape, KPpost.shape, VP.shape,
       LG.shape))
z51 = np.load(os.path.join(
    BASE, 'phase3051',
    'omega_p48_l35_anatomy_qwen',
    'omega_p48_l35_anatomy_qwen.npz'),
    allow_pickle=True)
COS_V51 = z51['COS_V']
COS_H51 = z51['COS_H']
z53 = np.load(os.path.join(
    BASE, 'phase3053',
    'omega_p50_gate_source_qwen',
    'omega_p50_gate_source_qwen.npz'),
    allow_pickle=True)
STAGE53 = z53['STAGE']
COS_KCTRL_H0 = z53['COS_KCTRL_h0']
COS_KCTRL_H3 = z53['COS_KCTRL_h3']
z54 = np.load(os.path.join(
    BASE, 'phase3054',
    'omega_p51_norm_projection_qwen',
    'omega_p51_norm_projection_qwen.npz'),
    allow_pickle=True)
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
log('z48/z51/z53/z54/z55/z58/z59/z60/z61/z22/'
    'z2802 banks loaded')

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
        (3061, 'omega_p58_write_highdim_qwen'),
        (3062, 'omega_p59_body_identity_decode_'
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
a220_ok = bool(seal_detail) and all(seal_detail)
log('a220 source seals ok=%s' % a220_ok)

# ---------- assemble prompts (3048 verbatim) ----------
word_tok = {}
for w in set(TARGETS):
    wi = tok(' ' + w, add_special_tokens=False)[
        'input_ids']
    assert len(wi) == 1, (w, wi)
    word_tok[w] = int(wi[0])

assembled = []
for bi in range(len(BODIES)):
    for ci in range(len(PREFIXES)):
        s = (PREFIXES[ci] + ' ' + BODIES[bi]) \
            if PREFIXES[ci] else BODIES[bi]
        ids = [int(x) for x in tok(
            s, add_special_tokens=False)[
            'input_ids']]
        t = word_tok[TARGETS[bi]]
        assert ids.count(t) == 1, (bi, ci)
        assembled.append({'ids': ids,
                          'pos': ids.index(t),
                          'cond': ci, 'body': bi})
n_pr = len(assembled)
assert n_pr == 32
idx_of = {}
for i in range(n_pr):
    idx_of[(assembled[i]['cond'],
            assembled[i]['body'])] = i
for i in range(n_pr):
    ci = assembled[i]['cond']
    if ci == 0:
        assembled[i]['off'] = 0
    else:
        bid = assembled[idx_of[(0,
            assembled[i]['body'])]]['ids']
        pid = assembled[i]['ids']
        off = len(pid) - len(bid)
        assert off > 0, i
        assert list(pid[off + 1:]) \
            == list(bid[1:]), i
        w0b = tok.decode([bid[0]]).strip()
        w0p = tok.decode([pid[off]]).strip()
        assert w0b == w0p, (i, w0b, w0p)
        assembled[i]['off'] = off
for i in range(n_pr):
    assert LENS[i] == len(assembled[i]['ids']), i
log('assembled %d prompts (lens consistent '
    'with z48)' % n_pr)

cidx = []
bidx = []
for ci in (1, 2, 3):
    for bi in range(len(BODIES)):
        cidx.append(ci)
        bidx.append(bi)
cidx = np.array(cidx)
bidx = np.array(bidx)
NP_ = 24

t_targets = {}
for b in range(len(BODIES)):
    for c in (1, 2, 3):
        t_targets[(b, c)] = LG[idx_of[(c, b)]] \
            - LG[idx_of[(0, b)]]
TTl = np.stack([t_targets[(int(bidx[k]),
                          int(cidx[k]))]
                for k in range(NP_)])
a221_diff = float(np.max(np.abs(TTl - TT)))
a221_ok = bool(a221_diff == 0.0)
log('a221 TT recompute diff=%.3e ok=%s'
    % (a221_diff, a221_ok))

# a222: sampled re-capture, bit-exact vs z48
a222_diff = 0.0
for si in (0, 9, 17, 31):
    lg2, kp2, kn2, vp2 = forward_cap(
        assembled[si]['ids'])
    n2 = LENS[si]
    a222_diff = max(a222_diff, float(np.max(
        np.abs(LG[si] - lg2))))
    a222_diff = max(a222_diff, float(np.max(
        np.abs(KPpre[si, :, :n2, :] - kp2))))
    a222_diff = max(a222_diff, float(np.max(
        np.abs(KPpost[si, :, :n2, :] - kn2))))
    a222_diff = max(a222_diff, float(np.max(
        np.abs(VP[si, :, :n2, :] - vp2))))
a222_diff = float(a222_diff)
a222_ok = bool(a222_diff == 0.0)
log('a222 re-capture diff=%.3e ok=%s'
    % (a222_diff, a222_ok))

# a232: sham self-replacement (bit identity)
b0 = int(bidx[0])
base0 = idx_of[(0, b0)]
ids0 = assembled[base0]['ids']
n0 = len(ids0)
selfK = KPpost[base0, :, :n0, :].copy()
selfV = VP[base0, :, :n0, :].copy()
m_all = np.ones(n0, dtype=bool)
lg_s, _, _, _, _ = forward_stage(
    ids0, selfK, m_all, selfV, m_all)
a232_diff = float(np.max(np.abs(lg_s
                                - LG[base0])))
a232_ok = bool(a232_diff == 0.0)
log('a232 sham self-replacement diff=%.3e ok=%s'
    % (a232_diff, a232_ok))

# ---------- gamma / weights / anchors ----------
GAMMA_T = model.model.norm.weight.detach().clone()
GAMMA = GAMMA_T.double().cpu().numpy()
assert GAMMA.shape == (2560,)
gm_mean = float(GAMMA.mean())
gs_now = np.array([GAMMA.min(), GAMMA.max(),
                   GAMMA.mean(), GAMMA.std()])
a223_diff = float(np.max(np.abs(
    gs_now - z55['GAMMA_STATS'])))
a223_ok = bool(a223_diff == 0.0)
log('a223 gamma stats diff=%.3e ok=%s'
    % (a223_diff, a223_ok))

order_shape = np.argsort(
    np.abs(GAMMA - gm_mean))[::-1]
TOP64 = order_shape[:64].astype(np.int64)
FLAT64 = order_shape[::-1][:64].astype(np.int64)
S16 = TOP64[:16].astype(np.int64)
a224_diff = max(
    float(np.max(np.abs(TOP64
                        - z58['TOP64']))),
    float(np.max(np.abs(FLAT64
                        - z58['FLAT64']))))
a224_ok = bool(a224_diff == 0.0)
log('a224 TOP64/FLAT64 recompute diff=%.3e ok=%s'
    % (a224_diff, a224_ok))

D_TAN59 = z59['D_TAN']
assert D_TAN59.shape == (24, 2560)
a225_diff = float(np.max(np.abs(
    D_TAN59 - z59['D_TAN'])))
a225_ok = bool(a225_diff == 0.0
               and np.isfinite(D_TAN59).all())
log('a225 D_TAN bit vs z59 diff=%.3e ok=%s'
    % (a225_diff, a225_ok))

d_bar61 = D_TAN59.mean(axis=0)
bidx24 = bidx
cidx24 = cidx
B_b61 = np.stack([D_TAN59[bidx24 == b].mean(axis=0)
                  for b in range(8)])
C_c61 = np.stack([D_TAN59[cidx24 == c].mean(axis=0)
                  for c in (1, 2, 3)])
PC1_RO = z60['PC1']
PC1_D61 = z61['PC1_D']

# fp64 W_U copy (3.1 GB) - held through the a226
# recompute, the arm loop and the Y projections
WU = model.lm_head.weight.detach()  # (V, 2560)
wud = WU.double().cpu().numpy()

# a226: Vmat recompute (a218-style, fp64 W_U) +
# readout PC1 recompute (sign-invariant) +
# B_BODY/C_PREFIX recompute
NP_24 = 24
Vmat_new = np.zeros((NP_24, 2560))
for k in range(NP_24):
    w_t = wud.T @ TT[k]
    Vmat_new[k] = GAMMA * w_t
Ur_, svr_, Vtr_ = np.linalg.svd(
    Vmat_new, full_matrices=False)
pc1_new = Vtr_[0]
d_pc_p = float(np.max(np.abs(pc1_new
                             - z60['PC1'])))
d_pc_m = float(np.max(np.abs(pc1_new
                             + z60['PC1'])))
a226_diff = max(
    float(np.max(np.abs(Vmat_new
                        - z59['Vmat']))),
    float(np.max(np.abs(B_b61 - z61['B_BODY']))),
    float(np.max(np.abs(C_c61
                        - z61['C_PREFIX']))),
    float(min(d_pc_p, d_pc_m)))
a226_ok = bool(a226_diff == 0.0)
log('a226 Vmat/PC1_ro/B_BODY/C_PREFIX recompute '
    'vs z59/z60/z61 diff=%.3e ok=%s'
    % (a226_diff, a226_ok))
Vmat59 = Vmat_new

dh = layers[3].mlp.down_proj.weight.detach() \
    .double().cpu().numpy()
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
a227_diff = float(np.max(np.abs(
    COAL_TOP64 - z58['COAL_TOP64'])))
a227_ok = bool(a227_diff == 0.0)
log('a227 COAL_TOP64 recompute diff=%.3e ok=%s'
    % (a227_diff, a227_ok))
del dh


def front_rows(nb):
    return list(range(min(FRONT, nb)))


def l35_fields(k):
    """Per pair: base self fields (NL, nb, 1024)
    plus the L35 pref fields for the FRONT rows
    (K rotated by off, V raw)."""
    b = int(bidx[k])
    c = int(cidx[k])
    base_i = idx_of[(0, b)]
    pref_i = idx_of[(c, b)]
    nb = int(LENS[base_i])
    off = assembled[pref_i]['off']
    repK = KPpost[base_i, :, :nb, :].copy()
    repV = VP[base_i, :, :nb, :].copy()
    rows = front_rows(nb)
    ridx = np.asarray(rows)
    src = KPpost[pref_i][:, off + ridx, :] \
        .reshape(NL, len(rows), 8, HDIM)
    rotK = rot_apply(src, off)  # (NL,nfr,8,128)
    srcV = VP[pref_i][:, off + ridx, :]
    return repK, repV, rotK, srcV, rows, base_i


def cos_frac(lg_base, r, t):
    nr = float(np.linalg.norm(r))
    nt = float(np.linalg.norm(t))
    cs = float(r @ t) / (nr * nt) \
        if nr > 1e-12 and nt > 1e-12 else 0.0
    fr = nr / nt if nt > 1e-12 else float('nan')
    return cs, fr


NFR_ALL = None
for k in range(NP_):
    nb = int(LENS[idx_of[(0, int(bidx[k]))]])
    if NFR_ALL is None:
        NFR_ALL = len(front_rows(nb))
    assert len(front_rows(nb)) == NFR_ALL, k
assert NFR_ALL == 4

# ---------- T2/T4: four arms ----------
log('=== four arms: V / h0K / h3K / h7J '
    '(24 pairs each) ===')
sl_h7 = slice(HEAD_H * HDIM, (HEAD_H + 1) * HDIM)
sl_h0 = slice(HEAD_H0 * HDIM,
              (HEAD_H0 + 1) * HDIM)
sl_h3 = slice(HEAD_H3 * HDIM,
              (HEAD_H3 + 1) * HDIM)
COS_V3 = np.zeros(NP_)
COS_K0B = np.zeros(NP_)
COS_K3B = np.zeros(NP_)
COS_H7B = np.zeros(NP_)
STAGE_MLP = np.zeros(NP_)
STAGE_MLPN = np.zeros(NP_)
V_COS_D = np.zeros(NP_)
V_COS_DTAN = np.zeros(NP_)
V_COS_LG = np.zeros(NP_)
V_COS_DTAN_ONLY = np.zeros(NP_)
V_FRAC_RAD = np.zeros(NP_)
V_TAN_FRAC = np.zeros(NP_)
K0_COS_D = np.zeros(NP_)
K0_COS_DTAN = np.zeros(NP_)
K0_COS_DTAN_ONLY = np.zeros(NP_)
K0_FRAC_RAD = np.zeros(NP_)
K3_COS_D = np.zeros(NP_)
K3_COS_DTAN = np.zeros(NP_)
J_COS_D = np.zeros(NP_)
J_COS_DTAN = np.zeros(NP_)
D_V = np.zeros((NP_, 2560))
D_K0 = np.zeros((NP_, 2560))
D_K3 = np.zeros((NP_, 2560))
D_J = np.zeros((NP_, 2560))
a233_pre_max = 0.0
for k in range(NP_):
    b = int(bidx[k])
    repK_k, repV_k, rotK_k, srcV_k, rows_k, base_i \
        = l35_fields(k)
    ids_b = assembled[base_i]['ids']
    nb = int(LENS[base_i])
    mask = np.ones(nb, dtype=bool)
    # base (self replacement) stages
    _, pre_b, op_b, post_b, postn_b = forward_stage(
        ids_b, repK_k, mask, repV_k, mask)
    t_k = t_targets[(b, int(cidx[k]))]
    # ---- V-only arm (full row, all heads) ----
    rvV = repV_k.copy()
    rvV[L_TGT, rows_k, :] = srcV_k[L_TGT]
    lg_v, pre_v, op_v, post_v, postn_v \
        = forward_stage(ids_b, repK_k, mask,
                        rvV, mask)
    r = lg_v - LG[base_i]
    COS_V3[k] = cos_frac(LG[base_i], r, t_k)[0]
    d = post_v - post_b
    a233_pre_max = max(a233_pre_max,
                       float(np.max(np.abs(
                           pre_v - pre_b))))
    V_COS_D[k] = cos_frac(LG[base_i],
                          wud @ d, t_k)[0]
    xb = post_b
    nx2 = float(xb @ xb)
    xhat = xb / np.sqrt(nx2)
    ddot = float(d @ xhat)
    d_tan = d - ddot * xhat
    d_rad = ddot * xhat
    nd = float(np.linalg.norm(d))
    V_COS_DTAN[k] = cos_frac(LG[base_i],
                             wud @ d_tan, t_k)[0]
    V_FRAC_RAD[k] = abs(ddot) / nd \
        if nd > 1e-12 else 0.0
    V_TAN_FRAC[k] = float(np.linalg.norm(
        d_tan)) / nd if nd > 1e-12 else 0.0
    rms_b = float(np.sqrt(nx2 / 2560.0
                          + EPS_NORM))
    pred = GAMMA * d_tan / rms_b
    z_pred = wud @ pred
    nz = float(np.linalg.norm(z_pred))
    nr = float(np.linalg.norm(r))
    V_COS_LG[k] = float(z_pred @ r) / (nz * nr) \
        if nz > 1e-12 and nr > 1e-12 else 0.0
    with torch.no_grad():
        xbt = torch.tensor(xb, dtype=torch.float32,
                           device='cuda').unsqueeze(0)
        t1 = model.model.norm(
            xbt + torch.tensor(d_tan,
                               dtype=torch.float32,
                               device='cuda'))
    at = t1[0].double().cpu().numpy() - postn_b
    V_COS_DTAN_ONLY[k] = cos_frac(LG[base_i],
                                  wud @ at, t_k)[0]
    D_V[k] = d_tan
    # ---- h0 K-only arm ----
    rK0 = repK_k.copy()
    rK0[L_TGT, rows_k, sl_h0] \
        = rotK_k[L_TGT, :, HEAD_H0, :]
    lg_0, _, _, post_0, postn_0 = forward_stage(
        ids_b, rK0, mask, repV_k, mask)
    r = lg_0 - LG[base_i]
    COS_K0B[k] = cos_frac(LG[base_i], r, t_k)[0]
    d = post_0 - post_b
    K0_COS_D[k] = cos_frac(LG[base_i],
                           wud @ d, t_k)[0]
    xb = post_b
    nx2 = float(xb @ xb)
    xhat = xb / np.sqrt(nx2)
    ddot = float(d @ xhat)
    d_tan = d - ddot * xhat
    K0_COS_DTAN[k] = cos_frac(LG[base_i],
                              wud @ d_tan, t_k)[0]
    K0_FRAC_RAD[k] = abs(ddot) / float(
        np.linalg.norm(d))
    with torch.no_grad():
        xbt = torch.tensor(xb, dtype=torch.float32,
                           device='cuda').unsqueeze(0)
        t1 = model.model.norm(
            xbt + torch.tensor(d_tan,
                               dtype=torch.float32,
                               device='cuda'))
    at = t1[0].double().cpu().numpy() - postn_b
    K0_COS_DTAN_ONLY[k] = cos_frac(
        LG[base_i], wud @ at, t_k)[0]
    D_K0[k] = d_tan
    # ---- h3 K-only arm ----
    rK3 = repK_k.copy()
    rK3[L_TGT, rows_k, sl_h3] \
        = rotK_k[L_TGT, :, HEAD_H3, :]
    lg_3, _, _, post_3, _ = forward_stage(
        ids_b, rK3, mask, repV_k, mask)
    r = lg_3 - LG[base_i]
    COS_K3B[k] = cos_frac(LG[base_i], r, t_k)[0]
    d = post_3 - post_b
    K3_COS_D[k] = cos_frac(LG[base_i],
                           wud @ d, t_k)[0]
    xb = post_b
    xhat = xb / np.sqrt(float(xb @ xb))
    ddot = float(d @ xhat)
    d_tan = d - ddot * xhat
    K3_COS_DTAN[k] = cos_frac(LG[base_i],
                              wud @ d_tan, t_k)[0]
    D_K3[k] = d_tan
    # ---- h7 joint arm (canonical gate arm) ----
    rK7 = repK_k.copy()
    rV7 = repV_k.copy()
    rK7[L_TGT, rows_k, sl_h7] \
        = rotK_k[L_TGT, :, HEAD_H, :]
    svk = srcV_k[L_TGT].reshape(
        len(rows_k), KV_HEAD, HDIM)
    rV7[L_TGT, rows_k, sl_h7] = svk[:, HEAD_H, :]
    lg_7, _, _, post_7, postn_7 = forward_stage(
        ids_b, rK7, mask, rV7, mask)
    r = lg_7 - LG[base_i]
    COS_H7B[k] = cos_frac(LG[base_i], r, t_k)[0]
    STAGE_MLP[k] = cos_frac(LG[base_i],
                            wud @ (post_7
                                   - post_b),
                            t_k)[0]
    STAGE_MLPN[k] = cos_frac(LG[base_i],
                             wud @ (postn_7
                                    - postn_b),
                             t_k)[0]
    d = post_7 - post_b
    J_COS_D[k] = STAGE_MLP[k]
    xb = post_b
    xhat = xb / np.sqrt(float(xb @ xb))
    ddot = float(d @ xhat)
    d_tan = d - ddot * xhat
    J_COS_DTAN[k] = cos_frac(LG[base_i],
                             wud @ d_tan, t_k)[0]
    D_J[k] = d_tan
    if k % 8 == 0:
        log('  k=%02d cosV=%.4f cosK0=%.4f '
            'cosK3=%.4f cosJ=%.4f '
            'V_dtonly=%.4f'
            % (k, COS_V3[k], COS_K0B[k],
               COS_K3B[k], COS_H7B[k],
               V_COS_DTAN_ONLY[k]))

a233_ok = bool(a233_pre_max == 0.0)
log('a233 stage-pre identity max diff=%.3e ok=%s'
    % (a233_pre_max, a233_ok))
a228_diff = float(np.max(np.abs(COS_V3
                                - COS_V51)))
a228_ok = bool(a228_diff == 0.0)
log('a228 V arm repro vs z51 COS_V: diff=%.3e '
    'ok=%s' % (a228_diff, a228_ok))
a229_diff = max(
    float(np.max(np.abs(COS_K0B
                        - z54['COS_K0']))),
    float(np.max(np.abs(K0_COS_D
                        - z54['H0_COS_D']))),
    float(np.max(np.abs(K0_COS_DTAN
                        - z54['H0_COS_DTAN']))),
    float(np.max(np.abs(K0_COS_DTAN_ONLY
                        - z54['H0_COS_DTAN_'
                              'ONLY']))),
    float(np.max(np.abs(K0_FRAC_RAD
                        - z54['H0_FRAC_RAD']))))
a229_ok = bool(a229_diff == 0.0)
log('a229 h0K arm repro vs z54 (5 arrays): '
    'diff=%.3e ok=%s' % (a229_diff, a229_ok))
a230_diff = max(
    float(np.max(np.abs(COS_H7B
                        - COS_H51[HEAD_H]))),
    float(np.max(np.abs(STAGE_MLP
                        - STAGE53[:, 2]))),
    float(np.max(np.abs(STAGE_MLPN
                        - STAGE53[:, 3]))))
a230_ok = bool(a230_diff == 0.0)
log('a230 h7J arm repro vs z51 COS_H[7] + z53 '
    'STAGE cols 2/3: diff=%.3e ok=%s'
    % (a230_diff, a230_ok))
a231_diff = float(np.max(np.abs(COS_K3B
                                - COS_KCTRL_H3)))
a231_ok = bool(a231_diff == 0.0)
log('a231 h3K arm repro vs z53 COS_KCTRL_h3: '
    'diff=%.3e ok=%s' % (a231_diff, a231_ok))

med_v_cos = float(np.median(COS_V3))
med_v_d = float(np.median(V_COS_D))
med_v_dtan = float(np.median(V_COS_DTAN))
med_v_dtonly = float(np.median(V_COS_DTAN_ONLY))
med_v_frad = float(np.median(V_FRAC_RAD))
med_v_tfrac = float(np.median(V_TAN_FRAC))
med_v_clg = float(np.median(V_COS_LG))
med_k0 = float(np.median(COS_K0B))
med_k0_dtonly = float(np.median(
    K0_COS_DTAN_ONLY))
med_k3 = float(np.median(COS_K3B))
med_k3_dtan = float(np.median(K3_COS_DTAN))
med_j = float(np.median(COS_H7B))
med_j_dtan = float(np.median(J_COS_DTAN))
log('T2 V arm: med cos_V=%.4f cos_d=%.4f '
    'cos_dtan=%.4f cos_dtan_only=%.4f '
    'cos_lg=%.4f frac_rad=%.4f tan_frac=%.4f'
    % (med_v_cos, med_v_d, med_v_dtan,
       med_v_dtonly, med_v_clg, med_v_frad,
       med_v_tfrac))
log('T4 arms: med K0=%.4f (dtonly %.4f) '
    'K3=%.4f (dtan %.4f) J=%.4f (dtan %.4f)'
    % (med_k0, med_k0_dtonly, med_k3,
       med_k3_dtan, med_j, med_j_dtan))

# ---------- logit-space families ----------
def project(D):
    """D (24,2560) -> Y (24,151936) in row
    chunks."""
    Y = np.empty((D.shape[0], NVOC))
    for lo in range(0, D.shape[0], 8):
        Y[lo:lo + 8] = D[lo:lo + 8] @ wud.T
    return Y


Y_V = project(D_V)
Y_K0 = project(D_K0)
Y_K3 = project(D_K3)
Y_J = project(D_J)

# axis null: |cos| between two random directions
# IN THE READOUT METRIC (image of W_U = row space
# of W_U): u = W_U x for gaussian x, i.e.
# |x^T G y| / sqrt(x^T G x * y^T G y) with
# G = W_U W_U^T - computed while wud is held
Gmet = wud.T @ wud  # (2560, 2560) fp64
rng_ax = np.random.default_rng(9999)
Xa = rng_ax.standard_normal((N_PERM, 2560))
Yb_ = rng_ax.standard_normal((N_PERM, 2560))
GXa = Xa @ Gmet
GYb = Yb_ @ Gmet
_num = np.abs((GXa * Yb_).sum(axis=1))
_den = np.sqrt(np.maximum(
    (GXa * Xa).sum(axis=1), 1e-30)
    * np.maximum((GYb * Yb_).sum(axis=1), 1e-30))
ax_null = _num / _den
del Gmet, Xa, Yb_, GXa, GYb
del wud  # free the 3.1 GB fp64 copy
log('families projected to logit space; G-metric '
    'axis null built (wud released)')
ax_null_sorted = np.sort(ax_null)


def axis_p(c):
    """One-sided p = #{null >= c} + 1 over
    R+1."""
    idx = int(np.searchsorted(
        ax_null_sorted, c, side='right'))
    return float(N_PERM - idx + 1) \
        / (N_PERM + 1.0)


def unit_rows(Y):
    n = np.linalg.norm(Y, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return Y / n


def gram_abs_cos(Ya, Yb):
    A = unit_rows(Ya)
    B = unit_rows(Yb)
    return np.abs(A @ B.T)


def diag_stat(C):
    d = float(np.mean(np.diag(C)))
    off = float(np.mean(C[~np.eye(
        C.shape[0], dtype=bool)]))
    return d, off, d - off


def perm_diag_p(C, seed):
    """Null: permute column labels of |cos| Gram;
    R=2000 one-sided on obs = mean diag - mean
    offdiag."""
    n = C.shape[0]
    d, off, obs = diag_stat(C)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(N_PERM):
        p = rng.permutation(n)
        dp = float(np.mean(
            np.diag(C[:, p])))
        op = float(np.mean(C[:, p][~np.eye(
            n, dtype=bool)]))
        if dp - op >= obs:
            cnt += 1
    return obs, (float(cnt) + 1.0) \
        / (N_PERM + 1.0), d, off


def pc1_of(Y):
    """Top PC of a 24-row family via 24x24 Gram
    (sign arbitrary)."""
    A = unit_rows(Y)
    G = A @ A.T
    w_, v_ = np.linalg.eigh(G)
    u = v_[:, -1]
    if u[0] < 0:
        u = -u
    p = u @ A
    nn = float(np.linalg.norm(p))
    return p / nn if nn > 1e-12 else p


def axis_cos(Ya, Yb):
    ma = unit_rows(Ya).mean(axis=0)
    mb = unit_rows(Yb).mean(axis=0)
    na = float(np.linalg.norm(ma))
    nb = float(np.linalg.norm(mb))
    c_mean = abs(float(ma @ mb) / (na * nb)) \
        if na > 1e-12 and nb > 1e-12 else 0.0
    pa = pc1_of(Ya)
    pb = pc1_of(Yb)
    c_pc1 = abs(float(pa @ pb))
    return c_mean, c_pc1


# rotation null: |cos| of two random directions
# (G-metric version built above, before del wud)


# ---------- SS tests: V <-> h0K ----------
C_VK0 = gram_abs_cos(Y_V, Y_K0)
ss1_obs, p_ss1, dg_vk0, of_vk0 = perm_diag_p(
    C_VK0, 9998)
c_mean_vk0, c_pc1_vk0 = axis_cos(Y_V, Y_K0)
p_ax_mean = axis_p(c_mean_vk0)
p_ss2 = axis_p(c_pc1_vk0)

# ---------- T4 comparators ----------
C_VK3 = gram_abs_cos(Y_V, Y_K3)
ss1_obs_v3, p_ss1_v3, dg_vk3, of_vk3 \
    = perm_diag_p(C_VK3, 10000)
c_mean_vk3, c_pc1_vk3 = axis_cos(Y_V, Y_K3)
C_VJ = gram_abs_cos(Y_V, Y_J)
ss1_obs_vj, p_ss1_vj, dg_vj, of_vj \
    = perm_diag_p(C_VJ, 10001)
c_mean_vj, c_pc1_vj = axis_cos(Y_V, Y_J)
c_mean_k03, c_pc1_k03 = axis_cos(Y_K0, Y_K3)
axis_dominance = bool(c_pc1_vk0 >= c_pc1_vk3
                      and c_pc1_vk0 >= c_pc1_vj)
log('SS1 V<->h0K: diag=%.4f off=%.4f obs=%+.4f '
    'p=%.4f' % (dg_vk0, of_vk0, ss1_obs, p_ss1))
log('SS2 V<->h0K: |cos mean|=%.4f (p=%.4f) '
    '|cos PC1|=%.4f (p=%.4f)'
    % (c_mean_vk0, p_ax_mean, c_pc1_vk0, p_ss2))
log('T4: V<->h3K diag obs=%+.4f p=%.4f '
    '|cos PC1|=%.4f; V<->h7J diag obs=%+.4f '
    'p=%.4f |cos PC1|=%.4f; h0K<->h3K |cos '
    'PC1|=%.4f; axis_dominance=%s'
    % (ss1_obs_v3, p_ss1_v3, c_pc1_vk3,
       ss1_obs_vj, p_ss1_vj, c_pc1_vj,
       c_pc1_k03, axis_dominance))

# ---------- SS3 support Jaccard ----------
P_V = np.sqrt((D_V ** 2).mean(axis=0))
P_K0 = np.sqrt((D_K0 ** 2).mean(axis=0))
set_V = set(np.argsort(P_V)[::-1][:256].tolist())
set_K0 = set(np.argsort(P_K0)[::-1][:256]
             .tolist())
jacc_obs = float(len(set_V & set_K0))
jacc_obs /= float(len(set_V | set_K0))
rng_j = np.random.default_rng(10002)
jacc_null = np.zeros(N_PERM)
for i in range(N_PERM):
    idx = rng_j.choice(2560, 512, replace=False)
    s1 = set(idx[:256].tolist())
    s2 = set(idx[256:].tolist())
    jacc_null[i] = len(s1 & s2) / len(s1 | s2)
p_ss3 = float((jacc_null >= jacc_obs).sum()
              + 1) / (N_PERM + 1)
ss_count = int(sum(
    1 for p in (p_ss1, p_ss2, p_ss3)
    if p <= 0.005))
log('SS3: Jaccard(top256_V, top256_K0)=%.4f '
    '(null med=%.4f, p=%.4f); ss_count=%d'
    % (jacc_obs, float(np.median(jacc_null)),
       p_ss3, ss_count))

# ---------- T5 channel identity ----------
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


def energy_obs(profile, chan):
    tot = float((profile ** 2).sum())
    return float((profile ** 2)[chan].sum()) \
        / tot if tot > 0 else 0.0


def energy_perm_p(seed, profile, k, e_obs):
    rng = np.random.default_rng(seed)
    sq = profile ** 2
    tot = float(sq.sum())
    cnt = 0
    for _ in range(N_PERM):
        idx = rng.choice(2560, k, replace=False)
        if float(sq[idx].sum()) / tot >= e_obs:
            cnt += 1
    return (float(cnt) + 1.0) / (N_PERM + 1.0)


top256_V = np.argsort(P_V)[::-1][:256]
top256_K0 = np.argsort(P_K0)[::-1][:256]
res_ch = {}
for tag, prof, tset, s_base in (
        ('V', P_V, top256_V, 10003),
        ('K0', P_K0, top256_K0, 10009)):
    ov_s16 = int(len(set(tset.tolist())
                     & set(S16.tolist())))
    ov_t64 = int(len(set(tset.tolist())
                     & set(TOP64.tolist())))
    ov_coal = int(len(set(tset.tolist())
                      & set(COAL_TOP64.tolist())))
    ov_sig = int(sig2802[tset].sum())
    p_s16 = overlap_null(s_base, S16, ov_s16)
    p_t64 = overlap_null(s_base + 1, TOP64,
                         ov_t64)
    p_coal = overlap_null(s_base + 2, COAL_TOP64,
                          ov_coal)
    p_sig = overlap_null(s_base + 3,
                         np.where(sig2802)[0],
                         ov_sig)
    e16 = energy_obs(prof, S16)
    e64 = energy_obs(prof, TOP64)
    p_e16 = energy_perm_p(s_base + 4, prof, 16,
                          e16)
    p_e64 = energy_perm_p(s_base + 5, prof, 64,
                          e64)
    res_ch[tag] = dict(
        ov_s16=ov_s16, p_s16=p_s16,
        ov_t64=ov_t64, p_t64=p_t64,
        ov_coal=ov_coal, p_coal=p_coal,
        ov_sig=ov_sig, p_sig=p_sig,
        e16=e16, p_e16=p_e16,
        e64=e64, p_e64=p_e64)
    log('T5 %s: S16=%d (p=%.4f) TOP64=%d '
        '(p=%.4f) COAL64=%d (p=%.4f) sig=%d '
        '(p=%.4f) E16=%.4f (p=%.4f) E64=%.4f '
        '(p=%.4f)'
        % (tag, ov_s16, p_s16, ov_t64, p_t64,
           ov_coal, p_coal, ov_sig, p_sig,
           e16, p_e16, e64, p_e64))

# ---------- T6 known-axis relations ----------
nv_rows = np.linalg.norm(D_V, axis=1)
nv_rows[nv_rows < 1e-12] = 1.0
nv_k0 = np.linalg.norm(D_K0, axis=1)
nv_k0[nv_k0 < 1e-12] = 1.0
nvm = np.linalg.norm(Vmat59, axis=1)
cos_dv_vmat = np.array([
    float(D_V[k] @ Vmat59[k])
    / (nv_rows[k] * nvm[k]) for k in range(NP_)])
cos_dk0_vmat = np.array([
    float(D_K0[k] @ Vmat59[k])
    / (nv_k0[k] * nvm[k]) for k in range(NP_)])
npc = float(np.linalg.norm(PC1_RO))
cos_dv_pc1 = np.array([
    float(D_V[k] @ PC1_RO)
    / (nv_rows[k] * npc) for k in range(NP_)])
cos_dk0_pc1 = np.array([
    float(D_K0[k] @ PC1_RO)
    / (nv_k0[k] * npc) for k in range(NP_)])
mv_ch = D_V.mean(axis=0)
npd = float(np.linalg.norm(PC1_D61))
nmc = float(np.linalg.norm(mv_ch))
cos_mV_pc1D = float(mv_ch @ PC1_D61) / (nmc * npd) \
    if nmc > 1e-12 and npd > 1e-12 else 0.0
# channel-space rotation null: med_k |cos(D_V[k],
# u_k)| with an independent random direction per
# (draw, pair); chunked draws (3061 OOM lesson)
rng_kx = np.random.default_rng(10015)
D_Vn = D_V / nv_rows[:, None]
kx_null = np.zeros(N_PERM)
_kx_chunk = 100
for lo in range(0, N_PERM, _kx_chunk):
    Uc = rng_kx.standard_normal(
        (_kx_chunk, 2560))
    Uc /= np.linalg.norm(Uc, axis=1,
                         keepdims=True)
    Ck = np.abs(D_Vn @ Uc.T)  # (24, chunk)
    kx_null[lo:lo + _kx_chunk] = np.median(
        Ck, axis=0)
kx_sorted = np.sort(kx_null)
_kx_obs = float(np.median(np.abs(cos_dv_vmat)))
_kx_idx = int(np.searchsorted(
    kx_sorted, _kx_obs, side='right'))
p_kx = float(N_PERM - _kx_idx + 1) \
    / (N_PERM + 1.0)
log('T6: med cos(D_V,Vmat)=%.4f (perm-null '
    'p=%.4f) | med cos(D_K0,Vmat)=%.4f | med '
    '|cos(D_V,PC1_ro)|=%.4f med |cos(D_K0,'
    'PC1_ro)|=%.4f | cos(mean D_V, PC1_D)=%.4f'
    % (float(np.median(cos_dv_vmat)), p_kx,
       float(np.median(cos_dk0_vmat)),
       float(np.median(np.abs(cos_dv_pc1))),
       float(np.median(np.abs(cos_dk0_pc1))),
       cos_mV_pc1D))

# ---------- verdict ----------
v_adv = bool(med_v_dtonly < 0.0)
if v_adv and ss_count >= 2 and axis_dominance:
    verdict = 'adversarial_same_source_qwen'
elif v_adv and ss_count >= 2:
    verdict = 'adversarial_same_space_qwen'
elif v_adv:
    verdict = 'adversarial_arm_local_qwen'
else:
    verdict = 'v_radial_dilution_qwen'
log('VERDICT: %s (v_adv=%s ss_count=%d '
    'axis_dominance=%s med_v_dtonly=%.4f)'
    % (verdict, v_adv, ss_count,
       axis_dominance, med_v_dtonly))

anchor_core_ok = bool(
    a220_ok and a221_ok and a222_ok and a223_ok
    and a224_ok and a225_ok and a226_ok
    and a227_ok and a228_ok and a229_ok
    and a230_ok and a231_ok and a232_ok
    and a233_ok)
log('anchors core ok=%s' % anchor_core_ok)

elapsed = time.time() - t0
npz_path = os.path.join(OUT, NAME + '.npz')
np.savez(npz_path,
         COS_V3=COS_V3, COS_K0B=COS_K0B,
         COS_K3B=COS_K3B, COS_H7B=COS_H7B,
         STAGE_MLP=STAGE_MLP,
         STAGE_MLPN=STAGE_MLPN,
         V_COS_D=V_COS_D,
         V_COS_DTAN=V_COS_DTAN,
         V_COS_LG=V_COS_LG,
         V_COS_DTAN_ONLY=V_COS_DTAN_ONLY,
         V_FRAC_RAD=V_FRAC_RAD,
         V_TAN_FRAC=V_TAN_FRAC,
         K0_COS_D=K0_COS_D,
         K0_COS_DTAN=K0_COS_DTAN,
         K0_COS_DTAN_ONLY=K0_COS_DTAN_ONLY,
         K0_FRAC_RAD=K0_FRAC_RAD,
         K3_COS_D=K3_COS_D,
         K3_COS_DTAN=K3_COS_DTAN,
         J_COS_D=J_COS_D,
         J_COS_DTAN=J_COS_DTAN,
         D_V=D_V, D_K0=D_K0, D_K3=D_K3, D_J=D_J,
         C_VK0=C_VK0, C_VK3=C_VK3, C_VJ=C_VJ,
         SS1_OBS=np.float64(ss1_obs),
         SS1_P=np.float64(p_ss1),
         SS1_DIAG=np.float64(dg_vk0),
         SS1_OFF=np.float64(of_vk0),
         C_MEAN_VK0=np.float64(c_mean_vk0),
         P_AX_MEAN=np.float64(p_ax_mean),
         C_PC1_VK0=np.float64(c_pc1_vk0),
         SS2_P=np.float64(p_ss2),
         SS1_P_VK3=np.float64(p_ss1_v3),
         C_PC1_VK3=np.float64(c_pc1_vk3),
         SS1_P_VJ=np.float64(p_ss1_vj),
         C_PC1_VJ=np.float64(c_pc1_vj),
         C_PC1_K0K3=np.float64(c_pc1_k03),
         AXIS_DOM=np.bool_(axis_dominance),
         JACC_OBS=np.float64(jacc_obs),
         JACC_NULL=jacc_null,
         SS3_P=np.float64(p_ss3),
         SS_COUNT=np.int64(ss_count),
         P_V=P_V, P_K0=P_K0,
         TOP256_V=top256_V.astype(np.int64),
         TOP256_K0=top256_K0.astype(np.int64),
         COS_DV_VMAT=cos_dv_vmat,
         COS_DK0_VMAT=cos_dk0_vmat,
         COS_DV_PC1=cos_dv_pc1,
         COS_DK0_PC1=cos_dk0_pc1,
         COS_MV_PC1D=np.float64(cos_mV_pc1D),
         KX_NULL=kx_null,
         P_KX=np.float64(p_kx),
         CH_V_S16=np.int64(res_ch['V']['ov_s16']),
         CH_V_PS16=np.float64(
             res_ch['V']['p_s16']),
         CH_V_T64=np.int64(
             res_ch['V']['ov_t64']),
         CH_V_PT64=np.float64(
             res_ch['V']['p_t64']),
         CH_V_COAL=np.int64(
             res_ch['V']['ov_coal']),
         CH_V_PCOAL=np.float64(
             res_ch['V']['p_coal']),
         CH_V_SIG=np.int64(
             res_ch['V']['ov_sig']),
         CH_V_PSIG=np.float64(
             res_ch['V']['p_sig']),
         CH_V_E16=np.float64(res_ch['V']['e16']),
         CH_V_PE16=np.float64(
             res_ch['V']['p_e16']),
         CH_V_E64=np.float64(res_ch['V']['e64']),
         CH_V_PE64=np.float64(
             res_ch['V']['p_e64']),
         CH_K0_S16=np.int64(
             res_ch['K0']['ov_s16']),
         CH_K0_PS16=np.float64(
             res_ch['K0']['p_s16']),
         CH_K0_T64=np.int64(
             res_ch['K0']['ov_t64']),
         CH_K0_PT64=np.float64(
             res_ch['K0']['p_t64']),
         CH_K0_COAL=np.int64(
             res_ch['K0']['ov_coal']),
         CH_K0_PCOAL=np.float64(
             res_ch['K0']['p_coal']),
         CH_K0_SIG=np.int64(
             res_ch['K0']['ov_sig']),
         CH_K0_PSIG=np.float64(
             res_ch['K0']['p_sig']),
         CH_K0_E16=np.float64(
             res_ch['K0']['e16']),
         CH_K0_PE16=np.float64(
             res_ch['K0']['p_e16']),
         CH_K0_E64=np.float64(
             res_ch['K0']['e64']),
         CH_K0_PE64=np.float64(
             res_ch['K0']['p_e64']),
         AX_NULL=ax_null,
         V_ADV=np.bool_(v_adv),
         a220_ok=np.bool_(a220_ok),
         a221_diff=np.float64(a221_diff),
         a222_diff=np.float64(a222_diff),
         a223_diff=np.float64(a223_diff),
         a224_diff=np.float64(a224_diff),
         a225_diff=np.float64(a225_diff),
         a226_diff=np.float64(a226_diff),
         a227_diff=np.float64(a227_diff),
         a228_diff=np.float64(a228_diff),
         a229_diff=np.float64(a229_diff),
         a230_diff=np.float64(a230_diff),
         a231_diff=np.float64(a231_diff),
         a232_diff=np.float64(a232_diff),
         a233_pre_max=np.float64(a233_pre_max),
         anchor_core_ok=np.bool_(anchor_core_ok),
         verdict=np.array(verdict),
         elapsed=np.float64(elapsed))

stats = {
    'T2_v_mechanism': {
        'med_cos_V': med_v_cos,
        'med_cos_d': med_v_d,
        'med_cos_dtan': med_v_dtan,
        'med_cos_dtan_only': med_v_dtonly,
        'med_cos_lg_pred': med_v_clg,
        'med_frac_rad': med_v_frad,
        'med_tan_frac': med_v_tfrac,
        'v_adv': v_adv},
    'T3_same_source': {
        'ss1_diag': dg_vk0,
        'ss1_off': of_vk0,
        'ss1_obs': ss1_obs, 'ss1_p': p_ss1,
        'c_mean_vk0': c_mean_vk0,
        'p_ax_mean': p_ax_mean,
        'c_pc1_vk0': c_pc1_vk0,
        'ss2_p': p_ss2,
        'jaccard_obs': jacc_obs,
        'jaccard_null_med':
            float(np.median(jacc_null)),
        'ss3_p': p_ss3,
        'ss_count': ss_count},
    'T4_specificity': {
        'ss1_obs_vk3': ss1_obs_v3,
        'ss1_p_vk3': p_ss1_v3,
        'c_pc1_vk3': c_pc1_vk3,
        'ss1_obs_vj': ss1_obs_vj,
        'ss1_p_vj': p_ss1_vj,
        'c_pc1_vj': c_pc1_vj,
        'c_pc1_k0k3': c_pc1_k03,
        'axis_dominance': axis_dominance,
        'med_cos_K3': med_k3,
        'med_cos_K3_dtan': med_k3_dtan,
        'med_cos_J': med_j,
        'med_cos_J_dtan': med_j_dtan,
        'med_cos_K0': med_k0,
        'med_cos_K0_dtan_only':
            med_k0_dtonly},
    'T5_channel_identity': {
        'V': {kk: (float(vv) if isinstance(
            vv, float) else int(vv))
              for kk, vv in
              res_ch['V'].items()},
        'K0': {kk: (float(vv) if isinstance(
            vv, float) else int(vv))
               for kk, vv in
               res_ch['K0'].items()}},
    'T6_known_axis': {
        'med_cos_dv_vmat':
            float(np.median(cos_dv_vmat)),
        'p_kx': p_kx,
        'med_cos_dk0_vmat':
            float(np.median(cos_dk0_vmat)),
        'med_abs_cos_dv_pc1':
            float(np.median(np.abs(
                cos_dv_pc1))),
        'med_abs_cos_dk0_pc1':
            float(np.median(np.abs(
                cos_dk0_pc1))),
        'cos_meanDv_pc1D': cos_mV_pc1D},
    'anchors': {
        'a220_seals_ok': a220_ok,
        'a221_tt_diff': a221_diff,
        'a222_recapture_diff': a222_diff,
        'a223_gamma_stats_diff': a223_diff,
        'a224_top64_diff': a224_diff,
        'a225_dtan_diff': a225_diff,
        'a226_bbody_pc1_diff': a226_diff,
        'a227_coal_diff': a227_diff,
        'a228_varm_diff': a228_diff,
        'a229_h0k_diff': a229_diff,
        'a230_h7j_diff': a230_diff,
        'a231_h3k_diff': a231_diff,
        'a232_sham_diff': a232_diff,
        'a233_pre_max': a233_pre_max,
        'anchor_core_ok': anchor_core_ok},
}
result = {'phase': PHASE, 'name': NAME,
          'created': created, 'elapsed': elapsed,
          'run': 'run2 authoritative (fp32; '
                 'capture bank from the phase3048 '
                 'npz; four arms re-run with the '
                 '3054 forward_stage machinery; '
                 'chain anchors vs z51/z53/z54; '
                 'same-source stats offline in '
                 'the exact readout metric; '
                 'run1 crashed pre-verdict at the '
                 'G-metric axis null: wud is '
                 'vocab-major so wud @ wud.T tried '
                 'to allocate (151936,151936) - '
                 'fixed to wud.T @ wud)',
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
