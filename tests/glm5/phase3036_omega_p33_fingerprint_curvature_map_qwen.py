# -*- coding: utf-8 -*-
# Phase 3036 - Omega-P33: fingerprint curvature map (H2 decision)
# Per-fingerprint curvature probe: for each prompt's top-8 tokens,
# inject +-m*||h|| along the token's OWN fingerprint direction
# d_t = W_U[t]/||W_U[t]|| at the L35 decoder-layer input, measure the
# symmetric second difference kappa_t of P_t. Compare against the
# logistic operating-point prediction kappa_pred = b_t^2 (1-2P)/(P(1-P))
# (b_t = empirical odd slope, no free curvature parameter). H2 (the
# "high-curvature fingerprint can win from behind" claim) is supported
# iff the standardized residual exceeds the random-direction noise
# floor; otherwise curvature is operating-point-determined (uniform
# logistic readout). Side product: fingerprint pairwise cosine
# (orthogonality claim) vs random-row control.
# PREREG frozen below BEFORE any observation (execution.json written
# pre-run). Machine: 3035 verbatim two-step protocol (re-prefill per
# chain; step-2 token = ids[-1] repeat; captures at last position).
import os
import json
import time
import hashlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHASE = 3036
NAME = 'omega_p33_fingerprint_curvature_map_qwen'
BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase%d' % PHASE, NAME)
LOG = os.path.join(OUT, 'run_log.txt')
MODEL_DIR = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL = 36
HID = 2560
M_ARMS = (0.01, -0.01, 0.02, -0.02)
I_P01 = 0
I_M01 = 1
I_P02 = 2
I_M02 = 3
K_TOP = 8
N_RAND = 10
RAND_SEED = 9036
SITE_LATE = 35
SITES = (SITE_LATE,)
A54_GATE = 2e-2
Z_GATE = 3.0
SIGN_FRAC = 0.8
INFLEC_WIN = 0.05

GEN_PROMPTS = (
    'The weather was cold, so',
    'He studied every night because',
    'She wanted to buy the car, but',
    'The experiment failed, therefore',
    'You should take an umbrella if',
    'The meeting was long, and',
    'He missed the train, however',
    'The garden grows quickly while',
    'The price was high, yet',
    'She speaks French, although',
    'The road was closed, thus',
    'We left early because',)
SHAM_PROMPTS = ('The cat sat on the',
                'They walked home in the',
                'She looked at the sky above')

PREREG = {
    'mode': '3035 verbatim two-step protocol (re-prefill '
            'per chain, step-2 token ids[-1], captures at '
            'last position, eager attention, bf16); '
            'directed residual injection at the L35 '
            'decoder-layer INPUT via forward_pre_hook '
            'registered BEFORE capture hooks; h += m * '
            '||h|| * d computed in float32 then cast '
            'bf16; LATE site only (curvature is a readout '
            'property; mid-field attenuation already '
            'quantified in 3035 atten ~27x)',
    'question': 'H2 decision (attachment claim): do '
                'different fingerprints sit in regions of '
                'different curvature such that a '
                'high-curvature fingerprint could win from '
                'behind, or is readout curvature fully '
                'determined by the operating point (logistic '
                'f2 sign/magnitude), as 3035 indicated for '
                'top-1? Extends the probe to the top-8 token '
                'set with each token probed along its OWN '
                'fingerprint direction',
    'tokens': 'per prompt, top-8 tokens of the BASE step-2 '
              'full-model softmax at the last position (no '
              'exclusion); d_t = W_U[t] row L2-normalized '
              '(float32 lm_head rows)',
    'dose': 'm in {+0.01,-0.01,+0.02,-0.02} (fraction of '
            '||h|| at L35 input); kappa_t(delta) = '
            '[P_t(+d) + P_t(-d) - 2 P_t(0)] / delta^2 in P '
            'units, primary delta=0.01',
    'T1': 'sign test: for (prompt, token) pairs with '
          'P_t outside the near-inflection window '
          '[0.45, 0.55] (|P-0.5| >= 0.05), predicted sign '
          '= +1 if P_t < 0.5 else -1 (logistic f2 = '
          'P(1-P)(1-2P) sign); frac_match over logic '
          'prompts; sham top-8 as chance reference',
    'T2': 'PRIMARY H2 adjudication: kappa_pred_t = '
          'b_t^2 * (1 - 2 P_t) / max(P_t (1 - P_t), 1e-30) '
          'with b_t = (P_t(+0.01) - P_t(-0.01)) / 0.02 '
          '(empirical odd slope, no free curvature '
          'parameter); z_t = |kappa_t - kappa_pred_t| / '
          'theta_tag with theta_tag = 95th percentile of '
          '|kappa_rand| over N_RAND=10 Gaussian random '
          'unit directions (seed 9036) at m=+-0.01 '
          'measured on P_A (top-1 token, same estimator); '
          'med over 11 logic prompts of per-prompt '
          'max_t z_t; H2 supported iff med_max_z > '
          'Z_GATE=3.0',
    'T3': 'DESCRIPTIVE orthogonality: median |cos| over '
          'all pairs within each prompt top-8 fingerprint '
          'rows vs 200 random-row pairs (seed 9036, ids '
          '1000..50000); also curvature specificity ratio '
          'med|kappa_top8| / med|kappa_rand| (logic)',
    'verdict_tree': 'if med_max_z > 3.0 -> '
                    'fp_curvature_hetero_qwen (H2 '
                    'supported); elif frac_sign >= 0.8 -> '
                    'fp_curvature_logistic_uniform_qwen '
                    '(curvature operating-point-determined, '
                    'H2 rejected); else -> '
                    'fp_curvature_mixed_qwen',
    'anchors': 'a52 duplicate base chain logits '
               'bit-identical (0.0); a53 gate-on m=0 vs '
               'gate-off base logits bit-identical (0.0); '
               'a54 injection-entered: duplicate +0.01 late '
               'chain residual snapshot bit-identical AND '
               '||rs_arm - rs_base|| / ||rs_base|| within '
               'A54_GATE=2e-2 of 0.01; a55 W_U row copy '
               'bit-identical to lm_head float row (0.0); '
               'a56 kappa01 recomputed from stored P arrays '
               'bit-identical (0.0); a57 source seals: '
               'sha8(result.json) of phases 3030/3032/3035 '
               'matches their seal.json',
    'control': 'random-direction null: 10 Gaussian unit '
               'vectors per prompt (seed 9036), arms m = '
               '+-0.01 at late site, second difference of '
               'P_A; theta_tag = 95th percentile',
    'corrections': 'run1 fresh; late-site only by design '
                   '(mid attenuation covered by 3035); all '
                   'arrays pre-initialized before loops '
                   '(3020 lesson); injection hooks '
                   'registered before capture hooks; '
                   'prefill-pollution lesson (3032): every '
                   'chain re-prefills',
}

lines = []


def log(msg):
    lines.append(str(msg))
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(str(msg) + '\n')
    print(msg)


os.makedirs(OUT, exist_ok=True)
if os.path.exists(LOG):
    os.remove(LOG)
# rerun discipline: clear old artifacts
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

torch.manual_seed(3009)
np.random.seed(3009)

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.bfloat16,
    attn_implementation='eager').to('cuda').eval()
layers = model.model.layers
assert len(layers) == NL
assert int(model.config.hidden_size) == HID
EPS = float(getattr(model.config, 'rms_norm_eps', 1e-6))
log('model loaded eps=%g' % EPS)

# ---------- hooks ----------
state_r = {'on': False}
rs = {}
state_fin = {'on': False}
fin_cap = {}
state_fp = {'on': False, 'site': None, 'm': 0.0,
            'd': None, 'applied': None}
handles = []


def pre_layer(li):
    def h(module, args, kwargs):
        if state_r['on']:
            x = args[0] if args \
                else kwargs.get('hidden_states')
            if x is not None and x.dim() >= 2:
                rs.setdefault(li, []).append(
                    x[:, -1, :].detach().float()
                    .cpu().numpy().copy())
        return None
    return h


def pre_norm(module, args, kwargs):
    if state_fin['on']:
        fin_cap['x'] = args[0][:, -1, :] \
            .detach().float().cpu().numpy().copy()
    return None


def fp_inject(site):
    def h(module, args, kwargs):
        if not state_fp['on'] \
                or state_fp['site'] != site:
            return None
        x = args[0] if args \
            else kwargs.get('hidden_states')
        if x is None or x.dim() < 2:
            return None
        m = float(state_fp['m'])
        d_cur = state_fp['d']
        if d_cur is None:
            d_cur = torch.zeros(
                HID, device=x.device)
        h32 = x.detach().float()
        nrm = h32[:, -1, :].norm()
        delta = d_cur * (m * nrm)
        h32[:, -1, :] = h32[:, -1, :] + delta
        state_fp['applied'] = float(
            float(m) * float(nrm))
        out = h32.to(x.dtype)
        if args:
            return (out,) + tuple(args[1:]), kwargs
        return args, dict(kwargs,
                          hidden_states=out)
    return h


# injection hooks FIRST (pre-hook chaining: later hooks
# see earlier modifications), then capture hooks
for site in SITES:
    handles.append(layers[site]
                   .register_forward_pre_hook(
                       fp_inject(site),
                       with_kwargs=True))
for li in range(NL):
    handles.append(layers[li]
                   .register_forward_pre_hook(
                       pre_layer(li),
                       with_kwargs=True))
handles.append(model.model.norm
               .register_forward_pre_hook(
                   pre_norm, with_kwargs=True))

W_U = model.lm_head.weight.detach().float() \
    .cpu().numpy()
assert W_U.shape == (int(model.config.vocab_size),
                     HID), W_U.shape


def softmax(lg):
    l = lg - lg.max()
    p = np.exp(l)
    return p / p.sum()


def run_fp_chain(ids, site, m, d_fp, gate):
    rs.clear()
    fin_cap.pop('x', None)
    state_fp['applied'] = None
    with torch.no_grad():
        out = model(torch.tensor([ids],
                                 device='cuda'),
                    use_cache=True)
        past = out.past_key_values
        state_fp['on'] = bool(gate)
        state_fp['site'] = int(site)
        state_fp['m'] = float(m)
        state_fp['d'] = d_fp
        state_r['on'] = True
        state_fin['on'] = True
        out2 = model(
            input_ids=torch.tensor(
                [[int(ids[-1])]], device='cuda'),
            past_key_values=past,
            use_cache=False)
        state_r['on'] = False
        state_fin['on'] = False
    state_fp['on'] = False
    lg = out2.logits[0, -1].detach() \
        .double().cpu().numpy()
    p = softmax(lg.astype(np.float64))
    snap = {li: rs[li][-1].copy()
            for li in SITES
            if li in rs}
    xf = fin_cap['x'][0].copy() \
        if 'x' in fin_cap else None
    applied = state_fp['applied']
    return p, lg, snap, xf, applied


# ---------- tag set from 3028 npz ----------
z28 = np.load(os.path.join(
    BASE, 'phase3028',
    'omega_p2v_dose_symmetry_qwen',
    'omega_p2v_dose_symmetry_qwen.npz'),
    allow_pickle=True)
tags28 = [str(t) for t in z28['tags']]
tag2prompt = {}
for t in tags28:
    tag2prompt[int(t.split(':')[0][1:])] = t
PROMPT_IDX = sorted(tag2prompt.keys())
log('tags28=%s' % tags28)
log('prompt_idx=%s' % PROMPT_IDX)

nT = len(PROMPT_IDX)
nS = len(SHAM_PROMPTS)
nP = nT + nS
row_tag = [tags28[i] for i in range(nT)] + \
    ['sham%d' % i for i in range(nS)]
row_prompt = [GEN_PROMPTS[pi] for pi in PROMPT_IDX] + \
    list(SHAM_PROMPTS)
rng = np.random.default_rng(RAND_SEED)
rand_dirs = rng.standard_normal((N_RAND, HID))
rand_dirs = rand_dirs / np.maximum(
    np.linalg.norm(rand_dirs, axis=1, keepdims=True),
    1e-30)

# pre-init all stores (3020 lesson)
p_cur = np.full((nP, K_TOP, 4), np.nan)
p0_top = np.full((nP, K_TOP), np.nan)
tok_top = np.zeros((nP, K_TOP), dtype=np.int64)
dec_top = [[''] * K_TOP for _ in range(nP)]
kappa01 = np.full((nP, K_TOP), np.nan)
kappa02 = np.full((nP, K_TOP), np.nan)
slope01 = np.full((nP, K_TOP), np.nan)
kpred01 = np.full((nP, K_TOP), np.nan)
zres01 = np.full((nP, K_TOP), np.nan)
sign_pred_arr = np.zeros((nP, K_TOP), dtype=int)
sign_match = np.zeros((nP, K_TOP), dtype=int)
rand_k01 = np.full((nP, N_RAND), np.nan)
theta_tag = np.full(nP, np.nan)
maxz_row = np.full(nP, np.nan)
maxz_tok = [''] * nP
cos_top_med = np.full(nP, np.nan)
top3gap = np.full(nP, np.nan)

# ---------- anchor workspaces ----------
a52_diff = None
a53_diff = None
a54_dup_diff = None
a54_ratio_err = None
a55_diff = None
a56_diff = None
a57_detail = []
verdict = None
anchor_all_ok = False

log('=== prompts (logic then sham) ===')
for ri in range(nP):
    pr = row_prompt[ri]
    ids = tok(pr, add_special_tokens=False)[
        'input_ids']
    # base chain (gate off)
    p_b, lg_b, snap_b, xf_b, _ = run_fp_chain(
        ids, SITE_LATE, 0.0, None, gate=False)
    order = np.argsort(-p_b)
    tok_top[ri] = order[:K_TOP]
    p0_top[ri] = p_b[order[:K_TOP]]
    for k in range(K_TOP):
        dec_top[ri][k] = tok.decode(
            [int(order[k])]).strip()
    if len(order) > 2:
        top3gap[ri] = float(lg_b[order[1]]
                            - lg_b[order[2]])
    # a52 / a53 (row 0 only)
    if ri == 0:
        p_d, lg_d, _, _, _ = run_fp_chain(
            ids, SITE_LATE, 0.0, None, gate=False)
        a52_diff = float(np.max(np.abs(lg_d - lg_b)))
        p_g, lg_g, _, _, _ = run_fp_chain(
            ids, SITE_LATE, 0.0, None, gate=True)
        a53_diff = float(np.max(np.abs(lg_g - lg_b)))
        a55_diff = float(np.max(np.abs(
            W_U[int(order[0])]
            - model.lm_head.weight.detach()
            .float().cpu().numpy()[int(order[0])])))
    # per-fingerprint dose arms (own direction)
    for k in range(K_TOP):
        wt = W_U[int(order[k])]
        d_t = torch.from_numpy(
            wt / max(float(np.linalg.norm(wt)),
                     1e-30)).float().cuda()
        for mi, m in enumerate(M_ARMS):
            p_a, _, _, _, _ = run_fp_chain(
                ids, SITE_LATE, m, d_t, gate=True)
            p_cur[ri, k, mi] = p_a[int(order[k])]
        pk0 = float(p0_top[ri, k])
        dp = float(p_cur[ri, k, I_P01])
        dm = float(p_cur[ri, k, I_M01])
        kappa01[ri, k] = (dp + dm - 2.0 * pk0) \
            / 0.01 ** 2
        slope01[ri, k] = (dp - dm) / 0.02
        dp2 = float(p_cur[ri, k, I_P02])
        dm2 = float(p_cur[ri, k, I_M02])
        kappa02[ri, k] = (dp2 + dm2 - 2.0 * pk0) \
            / 0.02 ** 2
        kpred01[ri, k] = slope01[ri, k] ** 2 \
            * (1.0 - 2.0 * pk0) \
            / max(pk0 * (1.0 - pk0), 1e-30)
        sign_pred_arr[ri, k] = 1 if pk0 < 0.5 else -1
    # random-direction null (measured on P_A)
    tokA = int(order[0])
    pa0 = float(p_b[tokA])
    for rj in range(N_RAND):
        d_r = torch.from_numpy(
            rand_dirs[rj]).float().cuda()
        prp, _, _, _, _ = run_fp_chain(
            ids, SITE_LATE, 0.01, d_r, gate=True)
        prm, _, _, _, _ = run_fp_chain(
            ids, SITE_LATE, -0.01, d_r, gate=True)
        rand_k01[ri, rj] = (
            float(prp[tokA]) + float(prm[tokA])
            - 2.0 * pa0) / 0.01 ** 2
    theta_tag[ri] = float(np.percentile(
        np.abs(rand_k01[ri]), 95))
    # a54 (row 0 only): injection-entered duplicate
    if ri == 0:
        wt0 = W_U[int(order[0])]
        d_t0 = torch.from_numpy(
            wt0 / max(float(np.linalg.norm(wt0)),
                      1e-30)).float().cuda()
        p_r, _, snap_r, _, _ = run_fp_chain(
            ids, SITE_LATE, 0.01, d_t0, gate=True)
        r_b = snap_b[SITE_LATE]
        r_a = snap_r[SITE_LATE]
        num = float(np.linalg.norm(r_a - r_b))
        den = float(np.linalg.norm(r_b))
        a54_ratio_err = float(
            abs(num / max(den, 1e-30) - 0.01) / 0.01)
        p_r2, _, snap_r2, _, _ = run_fp_chain(
            ids, SITE_LATE, 0.01, d_t0, gate=True)
        a54_dup_diff = float(np.max(np.abs(
            snap_r2[SITE_LATE] - snap_r[SITE_LATE])))
    # per-row stats
    zres01[ri] = np.abs(kappa01[ri] - kpred01[ri]) \
        / max(float(theta_tag[ri]), 1e-30)
    for k in range(K_TOP):
        sign_match[ri, k] = int(
            np.sign(kappa01[ri, k])
            == sign_pred_arr[ri, k]
            and abs(kappa01[ri, k]) > 1e-12)
    kmax = int(np.argmax(zres01[ri]))
    maxz_row[ri] = float(zres01[ri, kmax])
    maxz_tok[ri] = dec_top[ri][kmax]
    wr = W_U[tok_top[ri]]
    wrn = wr / np.maximum(
        np.linalg.norm(wr, axis=1, keepdims=True),
        1e-30)
    cm = np.abs(wrn @ wrn.T)
    iu = np.triu_indices(K_TOP, 1)
    cos_top_med[ri] = float(np.median(cm[iu]))
    log('%s A=%r pA0=%.4f theta=%.3e medz=%.2f '
        'maxz=%.2f @%r cos8=%.4f'
        % (row_tag[ri], dec_top[ri][0],
           p0_top[ri, 0], theta_tag[ri],
           float(np.median(zres01[ri])),
           maxz_row[ri], maxz_tok[ri],
           cos_top_med[ri]))

# ---------- a56: recompute identity ----------
rec = (p_cur[:, :, I_P01] + p_cur[:, :, I_M01]
       - 2.0 * p0_top) / 0.01 ** 2
a56_diff = float(np.nanmax(np.abs(rec - kappa01)))

# ---------- a57: source seals ----------
for ph, nm in ((3030, 'omega_p2x_readout_convexity_qwen'),
               (3032, 'omega_p2z_deep_peak_anatomy_qwen'),
               (3035,
                'omega_p32_fingerprint_competition_qwen')):
    d = os.path.join(BASE, 'phase%d' % ph, nm)
    with open(os.path.join(d, 'seal.json'),
              encoding='utf-8') as f:
        sealj = json.load(f)
    with open(os.path.join(d, 'result.json'),
              'rb') as f:
        s = hashlib.sha256(f.read()).hexdigest()[:8]
    a57_detail.append(bool(s == sealj['result_sha256_8']))
a57_ok = bool(a57_detail) and all(a57_detail)

# ---------- a51-style manual recompute (row0 base) ----------
pi0 = PROMPT_IDX[0]
ids0 = tok(GEN_PROMPTS[pi0],
           add_special_tokens=False)['input_ids']
p_b2, lg_b2, _, xf_b2, _ = run_fp_chain(
    ids0, SITE_LATE, 0.0, None, gate=False)
w = model.model.norm.weight.detach() \
    .float().cpu().numpy()
h_fin = xf_b2
var = float((h_fin ** 2).mean())
hn = h_fin / np.sqrt(var + EPS)
lg_man = (W_U @ (hn * w)).astype(np.float64)
a51_maxdiff = float(np.max(np.abs(lg_man - lg_b2)))
ord_m = np.argsort(-lg_man)
ord_o = np.argsort(-lg_b2)
near_tie = np.isfinite(top3gap[0]) and top3gap[0] < 0.05
if near_tie:
    a51_top2_ok = True
    a51_note = 'near-tie skip (gap=%.4f)' % top3gap[0]
else:
    a51_top2_ok = bool(ord_m[0] == ord_o[0]
                       and ord_m[1] == ord_o[1])
    a51_note = ''

# ---------- verdict ----------
a52_ok = a52_diff is not None and a52_diff == 0.0
a53_ok = a53_diff is not None and a53_diff == 0.0
a54_ok = (a54_dup_diff is not None
          and a54_dup_diff == 0.0
          and a54_ratio_err is not None
          and a54_ratio_err <= A54_GATE)
a55_ok = a55_diff is not None and a55_diff == 0.0
a56_ok = a56_diff is not None and a56_diff == 0.0
a51_ok = (a51_top2_ok is True
          and a51_maxdiff is not None
          and a51_maxdiff <= 0.15)
anchor_all_ok = bool(a52_ok and a53_ok and a54_ok
                     and a55_ok and a56_ok and a57_ok
                     and a51_ok)

elig = np.abs(p0_top[:nT] - 0.5) >= INFLEC_WIN
n_elig = int(elig.sum())
n_sign_match = int(sign_match[:nT][elig].sum())
frac_sign = float(n_sign_match) / max(n_elig, 1)
sham_elig = np.abs(p0_top[nT:] - 0.5) >= INFLEC_WIN
sham_match = int(sign_match[nT:][sham_elig].sum())
sham_elig_n = int(sham_elig.sum())
med_max_z = float(np.median(maxz_row[:nT]))
med_theta = float(np.median(theta_tag[:nT]))
med_z = float(np.median(zres01[:nT]))
k_top8_abs = np.abs(kappa01[:nT])
spec_curv = float(np.median(k_top8_abs)
                  / max(float(np.median(
                      np.abs(rand_k01[:nT]))), 1e-30))
med_cos_top = float(np.median(cos_top_med[:nT]))
rid = rng.integers(1000, 50000, size=400)
wr = W_U[rid]
wrn = wr / np.maximum(
    np.linalg.norm(wr, axis=1, keepdims=True), 1e-30)
cmr = np.abs(np.sum(wrn[0::2] * wrn[1::2], axis=1))
med_cos_rand = float(np.median(cmr))

if med_max_z > Z_GATE:
    verdict = 'fp_curvature_hetero_qwen'
elif frac_sign >= SIGN_FRAC:
    verdict = 'fp_curvature_logistic_uniform_qwen'
else:
    verdict = 'fp_curvature_mixed_qwen'

log('=== verdict ===')
log('a52=%r a53=%r a54_dup=%r a54_ratio_err=%r '
    'a55=%r a56=%r a57=%r a51_top2=%r a51_maxdiff=%r'
    % (a52_diff, a53_diff, a54_dup_diff,
       a54_ratio_err, a55_diff, a56_diff, a57_ok,
       a51_top2_ok, a51_maxdiff))
log('sign: n_match=%d/%d eligible (frac=%.3f); sham '
    '%d/%d' % (n_sign_match, n_elig, frac_sign,
               sham_match, sham_elig_n))
log('H2: med_max_z=%.3f (gate %.1f) med_z=%.3f '
    'med_theta=%.3e spec_curv=%.2f'
    % (med_max_z, Z_GATE, med_z, med_theta,
       spec_curv))
log('ortho: med|cos| top8=%.4f random=%.4f'
    % (med_cos_top, med_cos_rand))
log('VERDICT=%s anchor_all_ok=%s'
    % (verdict, anchor_all_ok))
log('correction_note=%s'
    % ('a51 near-tie: ' + a51_note if a51_note
       else 'none'))

elapsed = time.time() - t0

# ---------- npz (flat arrays only) ----------
npz_path = os.path.join(OUT, NAME + '.npz')
np.savez(
    npz_path,
    tags=np.array(row_tag),
    prompt_idx=np.array(PROMPT_IDX + [-1] * nS),
    m_arms=np.array(M_ARMS),
    p_cur=p_cur, p0_top=p0_top, tok_top=tok_top,
    dec_top=np.array(dec_top),
    kappa01=kappa01, kappa02=kappa02,
    slope01=slope01, kpred01=kpred01,
    zres01=zres01, sign_pred=sign_pred_arr,
    sign_match=sign_match,
    rand_k01=rand_k01, theta_tag=theta_tag,
    maxz_row=maxz_row, maxz_tok=np.array(maxz_tok),
    cos_top_med=cos_top_med, top3gap=top3gap,
    a52_diff=np.float64(a52_diff),
    a53_diff=np.float64(a53_diff),
    a54_dup_diff=np.float64(a54_dup_diff),
    a54_ratio_err=np.float64(a54_ratio_err),
    a55_diff=np.float64(a55_diff),
    a56_diff=np.float64(a56_diff),
    a57_ok=np.bool_(a57_ok),
    a51_top2_ok=np.bool_(a51_top2_ok),
    a51_maxdiff=np.float64(a51_maxdiff),
    n_elig=np.int64(n_elig),
    n_sign_match=np.int64(n_sign_match),
    frac_sign=np.float64(frac_sign),
    med_max_z=np.float64(med_max_z),
    med_z=np.float64(med_z),
    med_theta=np.float64(med_theta),
    spec_curv=np.float64(spec_curv),
    med_cos_top=np.float64(med_cos_top),
    med_cos_rand=np.float64(med_cos_rand),
    verdict=np.array(verdict),
    elapsed=np.float64(elapsed))

result = {
    'phase': PHASE, 'name': NAME, 'created': created,
    'final_verdict': verdict,
    'anchor_all_ok': anchor_all_ok,
    'anchors': {
        'a52_dup_base_bit': a52_diff,
        'a53_gate0_bit': a53_diff,
        'a54_dup_bit': a54_dup_diff,
        'a54_ratio_err': a54_ratio_err,
        'a54_gate': A54_GATE,
        'a55_wu_row_bit': a55_diff,
        'a56_kappa_recompute_bit': a56_diff,
        'a57_source_seals': a57_ok,
        'a51_top2_ok': a51_top2_ok,
        'a51_maxdiff': a51_maxdiff,
        'a51_note': a51_note,
    },
    'T1_sign': {
        'n_eligible': n_elig,
        'n_match': n_sign_match,
        'frac_match': frac_sign,
        'sham_match': sham_match,
        'sham_eligible': sham_elig_n,
        'inflection_window': INFLEC_WIN,
        'match_matrix': [[int(v) for v in row]
                         for row in sign_match[:nT]],
    },
    'T2_h2': {
        'med_max_z': med_max_z,
        'z_gate': Z_GATE,
        'med_z': med_z,
        'med_theta': med_theta,
        'maxz_per_row': [float(v) for v in maxz_row],
        'maxz_tok': maxz_tok,
        'theta_per_row': [float(v)
                          for v in theta_tag],
        'spec_curv_ratio': spec_curv,
        'h2_supported': bool(med_max_z > Z_GATE),
    },
    'T3_ortho': {
        'med_abs_cos_top8': med_cos_top,
        'med_abs_cos_random': med_cos_rand,
        'cos_top_med_per_row': [float(v) for v in
                                cos_top_med[:nT]],
    },
    'per_row': [
        {'tag': row_tag[i],
         'prompt': row_prompt[i],
         'top8': [{'tok': dec_top[i][k],
                   'p0': float(p0_top[i, k]),
                   'kappa01': float(kappa01[i, k]),
                   'kpred01': float(kpred01[i, k]),
                   'z': float(zres01[i, k]),
                   'sign_match': int(sign_match[i, k])}
                  for k in range(K_TOP)]}
        for i in range(nP)],
    'correction_note': ('a51 near-tie: ' + a51_note)
    if a51_note else 'none',
    'prereg': PREREG,
    'elapsed_s': round(elapsed, 1),
}
res_path = os.path.join(OUT, 'result.json')
with open(res_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=1)


def sha8(p):
    with open(p, 'rb') as f:
        return hashlib.sha256(f.read()) \
            .hexdigest()[:8]


seal = {
    'phase': PHASE, 'name': NAME, 'created': created,
    'npz_sha256_8': sha8(npz_path),
    'result_sha256_8': sha8(res_path),
    'exec_sha256_8': sha8(os.path.join(
        OUT, 'execution.json')),
    'script_sha256_8': sha8(os.path.abspath(__file__)),
    'verdict': verdict,
    'anchor_all_ok': anchor_all_ok,
}
with open(os.path.join(OUT, 'seal.json'), 'w',
          encoding='utf-8') as f:
    json.dump(seal, f, ensure_ascii=False, indent=1)
log('sealed npz8=%s result8=%s exec8=%s script8=%s '
    'elapsed=%.1fs'
    % (seal['npz_sha256_8'], seal['result_sha256_8'],
       seal['exec_sha256_8'], seal['script_sha256_8'],
       elapsed))
log('sealed')
