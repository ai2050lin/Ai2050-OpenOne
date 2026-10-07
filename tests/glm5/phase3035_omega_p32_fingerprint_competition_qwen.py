# -*- coding: utf-8 -*-
# Phase 3035 - Omega-P32: fingerprint competition anatomy
# Directed readout intervention along d_fp = (w_A - w_B)/||w_A - w||
# at decoder-layer input sites L8 (mid field) and L35 (pre-readout).
# Question: is the P(A)/P(B) migration symmetric (odd) in m, and does
# the even part follow the logistic inflection prediction
# (convex iff P0(A) < 0.5)? Operationalizes the "fingerprint
# competition" hypothesis against the 3028-3030 convexity chain.
# PREREG frozen below BEFORE any observation (execution.json written
# pre-run). Machine: 3032 verbatim two-step protocol (re-prefill per
# chain; step-2 token = ids[-1] repeat; captures at last position).
import os
import json
import time
import hashlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHASE = 3035
NAME = 'omega_p32_fingerprint_competition_qwen'
BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase%d' % PHASE, NAME)
LOG = os.path.join(OUT, 'run_log.txt')
MODEL_DIR = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL = 36
HID = 2560
M_GRID = (-0.05, -0.02, -0.01, 0.0, 0.01, 0.02,
           0.05)
SITES = (8, 35)
A49_GATE = 2e-2
A51_GATE = 0.15
THETA_SHAM = 0.02
T2_MATCH_HI = 9
T2_MATCH_LO = 5
T1_MED_GATE = 0.35

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
    'mode': '3032 verbatim two-step protocol (re-prefill '
            'per chain, step-2 token ids[-1], captures at '
            'last position, eager attention, bf16); NEW: '
            'directed residual injection along '
            'd_fp = (w_A - w_B)/||w_A - w_B|| (float32, '
            'W_U = lm_head rows) applied at decoder-layer '
            'INPUT via forward_pre_hook registered BEFORE '
            'the residual capture hook (so capture sees '
            'injected value); h_last += m * ||h_last|| * '
            'd_fp computed in float32 then cast bf16',
    'question': 'fingerprint competition: when the '
                'residual is pushed along the '
                'fingerprint-difference direction of the '
                'top-2 competing tokens, is the P(A)/P(B) '
                'migration odd-symmetric in dose m, does '
                'the even (curvature) part follow the '
                'logistic inflection prediction (positive '
                'iff P0(A) < 0.5), and is the migration a '
                'two-way competition (mass taken mainly '
                'from the runner-up)?',
    'tokens': 'A, B = top-2 tokens of the BASE step-2 '
              'full-model softmax at the last position '
              '(no exclusion); d_fp from lm_head weight '
              'rows in float32, L2-normalized',
    'dose': 'm in {-0.05,-0.02,-0.01,0.0,0.01,0.02,0.05} '
            '(fraction of ||h_last||), sites L8 and L35 '
            'decoder-layer input; LATE (35) is primary',
    'T1': 'PRIMARY symmetry: per prompt A_idx = '
          'sum|even| / (sum|even| + sum|odd|) over the '
          'three (+m,-m) pairs on P(A), late site; odd = '
          '(dPA(m) - dPA(-m))/2, even = (dPA(m) + '
          'dPA(-m))/2; report med over logic prompts and '
          'sham null',
    'T2': 'PRIMARY inflection: even at m=0.01 (smallest '
          'dose, least contamination); predicted sign = '
          '+1 if P0(A) < 0.5 else -1 (logistic f2 sign = '
          'sigma(1-sigma)(1-2*sigma)); n_match over 11 '
          'logic prompts; sham match count as chance '
          'reference',
    'T3': 'DESCRIPTIVE competition coupling: kappa = '
          '-dPB(+0.05) / dPA(+0.05) at late site (kappa=1 '
          'perfect two-way); monotonicity |dPA(0.05)| > '
          '|dPA(0.02)| > |dPA(0.01)| count; early-vs-late '
          'attenuation ratio med|dPA|',
    'verdict_tree': 'if spec_ratio < 2 -> '
                    'fp_nonspecific_qwen; elif n_match >= '
                    '9 AND med A_idx <= 0.35 -> '
                    'fp_logistic_readout_qwen; elif '
                    'n_match <= 5 -> '
                    'fp_anomalous_asymmetry_qwen; else -> '
                    'fp_mixed_competition_qwen',
    'anchors': 'a47 duplicate base chain logits '
               'bit-identical (0.0); a48 gate-on m=0 '
               'chain vs gate-off base logits '
               'bit-identical (0.0); a49 injection-'
               'entered: duplicate +0.1 late chains rs '
               'bit-identical AND ||rs_arm - rs_base|| / '
               '||rs_base|| within A49_GATE=2e-2 of m; '
               'a51 manual final-norm+lm_head recompute '
               'vs model logits: top-2 identity (skip '
               'note if logit2-logit3 < 0.05 near-tie) '
               'AND max|dlogit| <= A51_GATE=0.15 (bf16 '
               'family, a38 lesson)',
    'control': 'random-direction null: d_rand = '
               '(W_U[r1] - W_U[r2]) normalized, r1/r2 = '
               'base-prob rank 100/101 tokens; arms at '
               'late site m in {-0.02,-0.01,0.01,0.02}; '
               'spec_ratio = min med-ratio over m in '
               '{0.01,0.02}, gate 2',
    'corrections': 'run4 (npz8 1a3c860f): sham gate '
                   'mis-specified - theta on |dPA(0.05)| '
                   'under intervention measures EFFICACY '
                   '(m*||h|| at L35 input large in '
                   'absolute units), not contamination; '
                   'contamination covered by a47/a48 bit '
                   'determinism; run4 verdict '
                   'fp_contaminated_void registered; '
                   'run5 adds random-direction '
                   'specificity control + perturbative '
                   'dose grid, T2 pair 0.05->0.01, T3 at '
                   '0.05; also prefill-pollution lesson '
                   '(3032 run2): every chain re-prefills; '
                   'injection hooks registered before '
                   'capture hooks; all arrays '
                   'pre-initialized before loops '
                   '(3020 lesson)',
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
assert W_U.shape == (int(model.config.vocab_size), HID), W_U.shape


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
            for li in (SITES[0], SITES[1])
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
nM = len(M_GRID)
i_m0 = M_GRID.index(0.0)

# pre-init all stores (3020 lesson)
pA_late = np.full((nT, nM), np.nan)
pB_late = np.full((nT, nM), np.nan)
pA_early = np.full((nT, nM), np.nan)
pB_early = np.full((nT, nM), np.nan)
pA0 = np.full(nT, np.nan)
pB0 = np.full(nT, np.nan)
tokA = np.zeros(nT, dtype=np.int64)
tokB = np.zeros(nT, dtype=np.int64)
decA = []
decB = []
top3gap = np.full(nT, np.nan)
A_idx = np.full(nT, np.nan)
even1 = np.full(nT, np.nan)
sign_pred = np.zeros(nT, dtype=int)
match_flag = np.zeros(nT, dtype=int)
kappa = np.full(nT, np.nan)
mono_ok = np.zeros(nT, dtype=int)
atten = np.full(nT, np.nan)
base_lg_top8 = np.full((nT, 8), np.nan)
base_tok_top8 = np.zeros((nT, 8), dtype=np.int64)
dPA_late = np.full((nT, nM), np.nan)
dPA_rand_late = np.full((nT, 4), np.nan)

# sham stores
nS = len(SHAM_PROMPTS)
pA_sham = np.full((nS, nM), np.nan)
pA0_sham = np.full(nS, np.nan)
A_idx_sham = np.full(nS, np.nan)
even1_sham = np.full(nS, np.nan)
match_sham = np.zeros(nS, dtype=int)

# ---------- anchor workspaces ----------
a47_diff = None
a48_diff = None
a49_dup_diff = None
a49_ratio_err = None
a51_top2_ok = None
a51_maxdiff = None
a51_note = ''
verdict = None
anchor_all_ok = False

log('=== logic prompts ===')
ti = 0
for pi in PROMPT_IDX:
    pr = GEN_PROMPTS[pi]
    ids = tok(pr, add_special_tokens=False)[
        'input_ids']
    tag = tag2prompt[pi]
    # base chain (gate off)
    p_b, lg_b, snap_b, xf_b, _ = run_fp_chain(
        ids, SITES[1], 0.0, None, gate=False)
    order = np.argsort(-p_b)
    tokA[ti] = int(order[0])
    tokB[ti] = int(order[1])
    pA0[ti] = float(p_b[order[0]])
    pB0[ti] = float(p_b[order[1]])
    decA.append(tok.decode([int(order[0])]).strip())
    decB.append(tok.decode([int(order[1])]).strip())
    base_tok_top8[ti] = order[:8]
    base_lg_top8[ti] = lg_b[order[:8]]
    if len(order) > 2:
        top3gap[ti] = float(lg_b[order[1]]
                            - lg_b[order[2]])
    # a47: duplicate base chain (prompt 0 only)
    if ti == 0:
        p_d, lg_d, _, _, _ = run_fp_chain(
            ids, SITES[1], 0.0, None, gate=False)
        a47_diff = float(np.max(np.abs(lg_d - lg_b)))
    # a48: gate-on m=0 vs gate-off base
    p_g, lg_g, _, _, _ = run_fp_chain(
        ids, SITES[1], 0.0, None, gate=True)
    if ti == 0:
        a48_diff = float(np.max(np.abs(lg_g - lg_b)))
    # d_fp
    wA = W_U[int(order[0])]
    wB = W_U[int(order[1])]
    d = wA - wB
    nd = float(np.linalg.norm(d))
    d = d / max(nd, 1e-30)
    d_t = torch.from_numpy(d).float().cuda()
    # dose arms
    for si, site in enumerate(SITES):
        for mi, m in enumerate(M_GRID):
            p_a, _, snap_a, _, _ = run_fp_chain(
                ids, site, m, d_t, gate=True)
            if si == 0:
                pA_early[ti, mi] = p_a[tokA[ti]]
                pB_early[ti, mi] = p_a[tokB[ti]]
            else:
                pA_late[ti, mi] = p_a[tokA[ti]]
                pB_late[ti, mi] = p_a[tokB[ti]]
    # random-direction control arms (late site)
    r1 = int(order[100])
    r2 = int(order[101])
    dr = W_U[r1] - W_U[r2]
    dr = dr / max(float(np.linalg.norm(dr)),
                  1e-30)
    dr_t = torch.from_numpy(dr).float().cuda()
    for ri, mm in ((0, -0.02), (1, -0.01),
                   (2, 0.01), (3, 0.02)):
        p_rc, _, _, _, _ = run_fp_chain(
            ids, SITES[1], mm, dr_t,
            gate=True)
        dPA_rand_late[ti, ri] = \
            p_rc[tokA[ti]] - pA0[ti]
    # a49: duplicate +0.05 late chain + ratio check
    if ti == 0:
        p_r, _, snap_r, _, _ = run_fp_chain(
            ids, SITES[1], 0.05, d_t, gate=True)
        r_b = snap_b[SITES[1]]
        r_a = snap_r[SITES[1]]
        num = float(np.linalg.norm(r_a - r_b))
        den = float(np.linalg.norm(r_b))
        a49_ratio_err = float(abs(num / max(den, 1e-30)
                                  - 0.05) / 0.05)
        # duplicate bit-check against the grid arm
        p_a2, _, snap_a2, _, _ = run_fp_chain(
            ids, SITES[1], 0.05, d_t, gate=True)
        a49_dup_diff = float(np.max(np.abs(
            snap_a2[SITES[1]] - snap_r[SITES[1]])))
    # stats (late primary)
    dPA = pA_late[ti] - pA0[ti]
    dPA_late[ti] = dPA
    im = {m: k for k, m in enumerate(M_GRID)}
    s_ev = 0.0
    s_od = 0.0
    for m in (0.01, 0.02, 0.05):
        dp = dPA[im[m]]
        dm = dPA[im[-m]]
        s_ev += abs(0.5 * (dp + dm))
        s_od += abs(0.5 * (dp - dm))
    A_idx[ti] = s_ev / max(s_ev + s_od, 1e-30)
    e1 = 0.5 * (dPA[im[0.01]] + dPA[im[-0.01]])
    even1[ti] = e1
    sign_pred[ti] = 1 if pA0[ti] < 0.5 else -1
    match_flag[ti] = int(np.sign(e1)
                         == sign_pred[ti]
                         and abs(e1) > 1e-12)
    dpl = dPA[im[0.05]]
    dpb = pB_late[ti, im[0.05]] - pB0[ti]
    if abs(dpl) > 1e-12:
        kappa[ti] = float(-dpb / dpl)
    a5 = abs(dPA[im[0.05]])
    a1 = abs(dPA[im[0.02]])
    a0 = abs(dPA[im[0.01]])
    mono_ok[ti] = int(a5 > a1 > a0)
    dE = pA_early[ti] - pA0[ti]
    atten[ti] = float(abs(dE).mean()
                      / max(abs(dPA).mean(), 1e-30))
    log('%s P0(A)=%.4f A=%r B=%r A_idx=%.3f even1=%+.5f '
        'match=%d kappa=%.3f mono=%d atten=%.3f'
        % (tag, pA0[ti], decA[-1], decB[-1],
           A_idx[ti], even1[ti], match_flag[ti],
           kappa[ti] if np.isfinite(kappa[ti]) else -1,
           mono_ok[ti], atten[ti]))
    ti += 1

# ---------- sham null ----------
log('=== sham prompts ===')
for si_, pr in enumerate(SHAM_PROMPTS):
    ids = tok(pr, add_special_tokens=False)[
        'input_ids']
    p_b, lg_b, _, _, _ = run_fp_chain(
        ids, SITES[1], 0.0, None, gate=False)
    order = np.argsort(-p_b)
    pA0_sham[si_] = float(p_b[order[0]])
    wA = W_U[int(order[0])]
    wB = W_U[int(order[1])]
    d = wA - wB
    d = d / max(float(np.linalg.norm(d)), 1e-30)
    d_t = torch.from_numpy(d).float().cuda()
    for mi, m in enumerate(M_GRID):
        p_a, _, _, _, _ = run_fp_chain(
            ids, SITES[1], m, d_t, gate=True)
        pA_sham[si_, mi] = p_a[int(order[0])]
    dPA = pA_sham[si_] - pA0_sham[si_]
    im = {m: k for k, m in enumerate(M_GRID)}
    s_ev = 0.0
    s_od = 0.0
    for m in (0.01, 0.02, 0.05):
        dp = dPA[im[m]]
        dm = dPA[im[-m]]
        s_ev += abs(0.5 * (dp + dm))
        s_od += abs(0.5 * (dp - dm))
    A_idx_sham[si_] = s_ev / max(s_ev + s_od, 1e-30)
    e1 = 0.5 * (dPA[im[0.01]] + dPA[im[-0.01]])
    even1_sham[si_] = e1
    sp = 1 if pA0_sham[si_] < 0.5 else -1
    match_sham[si_] = int(np.sign(e1) == sp
                          and abs(e1) > 1e-12)
    log('sham%d P0=%.4f A_idx=%.3f even1=%+.5f '
        'match=%d dPA(0.05)=%+.5f'
        % (si_, pA0_sham[si_], A_idx_sham[si_],
           e1, match_sham[si_],
           dPA[im[0.05]]))

# ---------- a51 manual recompute (prompt0 base) ----------
pi0 = PROMPT_IDX[0]
ids0 = tok(GEN_PROMPTS[pi0],
           add_special_tokens=False)['input_ids']
p_b2, lg_b2, _, xf_b2, _ = run_fp_chain(
    ids0, SITES[1], 0.0, None, gate=False)
w = model.model.norm.weight.detach() \
    .float().cpu().numpy()
h_fin = xf_b2
var = float((h_fin ** 2).mean())
hn = h_fin / np.sqrt(var + EPS)
lg_man = (W_U @ (hn * w)).astype(np.float64)
a51_maxdiff = float(np.max(np.abs(lg_man - lg_b2)))
ord_m = np.argsort(-lg_man)
ord_o = np.argsort(-lg_b2)
if top3gap[0] is not None \
        and np.isfinite(top3gap[0]) \
        and top3gap[0] < 0.05:
    a51_top2_ok = True
    a51_note = 'near-tie skip (gap=%.4f)' % top3gap[0]
else:
    a51_top2_ok = bool(ord_m[0] == ord_o[0]
                       and ord_m[1] == ord_o[1])

# ---------- verdict ----------
a47_ok = a47_diff is not None and a47_diff == 0.0
a48_ok = a48_diff is not None and a48_diff == 0.0
a49_ok = (a49_dup_diff is not None
          and a49_dup_diff == 0.0
          and a49_ratio_err is not None
          and a49_ratio_err <= A49_GATE)
a51_ok = (a51_top2_ok is True
          and a51_maxdiff is not None
          and a51_maxdiff <= A51_GATE)
anchor_all_ok = bool(a47_ok and a48_ok
                     and a49_ok and a51_ok)

sham_med_small = float(np.median(np.abs(
    pA_sham[:, M_GRID.index(0.01)]
    - pA0_sham)))
i01 = M_GRID.index(0.01)
i02 = M_GRID.index(0.02)
spec1 = float(np.median(
    np.abs(dPA_late[:, i01])
    / np.maximum(np.abs(dPA_rand_late[:, 2]),
                 1e-12)))
spec2 = float(np.median(
    np.abs(dPA_late[:, i02])
    / np.maximum(np.abs(dPA_rand_late[:, 3]),
                 1e-12)))
spec_ratio = float(min(spec1, spec2))
n_match = int(match_flag.sum())
med_A = float(np.median(A_idx))
n_match_sham = int(match_sham.sum())
med_mono = int(mono_ok.sum())
med_kappa = float(np.median(kappa[np.isfinite(kappa)])) \
    if np.isfinite(kappa).any() else float('nan')
med_atten = float(np.median(atten[np.isfinite(atten)])) \
    if np.isfinite(atten).any() else float('nan')

if spec_ratio < 2.0:
    verdict = 'fp_nonspecific_qwen'
elif n_match >= T2_MATCH_HI \
        and med_A <= T1_MED_GATE:
    verdict = 'fp_logistic_readout_qwen'
elif n_match <= T2_MATCH_LO:
    verdict = 'fp_anomalous_asymmetry_qwen'
else:
    verdict = 'fp_mixed_competition_qwen'

log('=== verdict ===')
log('a47=%r a48=%r a49_dup=%r a49_ratio_err=%r '
    'a51_top2=%r a51_maxdiff=%r'
    % (a47_diff, a48_diff, a49_dup_diff,
       a49_ratio_err, a51_top2_ok, a51_maxdiff))
log('n_match=%d/11 med_A_idx=%.4f sham: med|dPA01|=%.5f '
    'n_match=%d/3' % (n_match, med_A, sham_med_small,
                      n_match_sham))
log('spec_ratio=%.4f (spec1=%.4f spec2=%.4f)'
    % (spec_ratio, spec1, spec2))
log('med_kappa=%.4f mono=%d/11 med_atten=%.4f'
    % (med_kappa, med_mono, med_atten))
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
    tags=np.array(tags28), prompt_idx=np.array(PROMPT_IDX),
    m_grid=np.array(M_GRID), sites=np.array(SITES),
    pA_late=pA_late, pB_late=pB_late,
    pA_early=pA_early, pB_early=pB_early,
    dPA_late=dPA_late, pA0=pA0, pB0=pB0,
    tokA=tokA, tokB=tokB,
    decA=np.array(decA), decB=np.array(decB),
    top3gap=top3gap, base_lg_top8=base_lg_top8,
    base_tok_top8=base_tok_top8,
    A_idx=A_idx, even1=even1, sign_pred=sign_pred,
    match_flag=match_flag, kappa=kappa,
    mono_ok=mono_ok, atten=atten,
    pA_sham=pA_sham, pA0_sham=pA0_sham,
    A_idx_sham=A_idx_sham, even1_sham=even1_sham,
    match_sham=match_sham,
    a47_diff=np.float64(a47_diff),
    a48_diff=np.float64(a48_diff),
    a49_dup_diff=np.float64(a49_dup_diff),
    a49_ratio_err=np.float64(a49_ratio_err),
    a51_top2_ok=np.bool_(a51_top2_ok),
    a51_maxdiff=np.float64(a51_maxdiff),
    sham_med_small=np.float64(sham_med_small),
    n_match=np.int64(n_match), med_A=np.float64(med_A),
    med_kappa=np.float64(med_kappa),
    med_atten=np.float64(med_atten),
    med_mono=np.int64(med_mono),
    dPA_rand_late=dPA_rand_late,
    spec_ratio=np.float64(spec_ratio),
    spec1=np.float64(spec1),
    spec2=np.float64(spec2),
    verdict=np.array(verdict),
    elapsed=np.float64(elapsed))

result = {
    'phase': PHASE, 'name': NAME, 'created': created,
    'final_verdict': verdict,
    'anchor_all_ok': anchor_all_ok,
    'anchors': {
        'a47_dup_base_bit': a47_diff,
        'a48_gate0_bit': a48_diff,
        'a49_dup_bit': a49_dup_diff,
        'a49_ratio_err': a49_ratio_err,
        'a49_gate': A49_GATE,
        'a51_top2_ok': a51_top2_ok,
        'a51_maxdiff': a51_maxdiff,
        'a51_gate': A51_GATE,
        'a51_note': a51_note,
    },
    'T1_symmetry': {
        'med_A_idx_logic': med_A,
        'A_idx_per_tag': [float(v) for v in A_idx],
        'A_idx_sham': [float(v) for v in A_idx_sham],
        'sham_med_abs_dPA05': sham_med_small,
        'theta_sham': THETA_SHAM,
    },
    'T2_inflection': {
        'n_match': n_match, 'n_total': nT,
        'match_per_tag': [int(v) for v in match_flag],
        'even1_per_tag': [float(v) for v in even1],
        'sign_pred_per_tag': [int(v) for v in sign_pred],
        'n_match_sham': n_match_sham,
        'even1_sham': [float(v) for v in even1_sham],
    },
    'T2b_specificity': {
        'spec_ratio': spec_ratio,
        'spec1_m01': spec1,
        'spec2_m02': spec2,
        'dPA_rand_late': [[float(x) for x in row]
                         for row in dPA_rand_late],
    },
    'T3_competition': {
        'med_kappa': med_kappa,
        'kappa_per_tag': [float(v) for v in kappa],
        'mono_count': med_mono,
        'med_atten_early_vs_late': med_atten,
    },
    'per_tag': [
        {'tag': tags28[i] if i < len(tags28) else '',
         'prompt': GEN_PROMPTS[PROMPT_IDX[i]],
         'A': decA[i], 'B': decB[i],
         'pA0': float(pA0[i]), 'pB0': float(pB0[i]),
         'A_idx': float(A_idx[i]),
         'even1': float(even1[i]),
         'match': int(match_flag[i]),
         'kappa': float(kappa[i])
         if np.isfinite(kappa[i]) else None,
         'mono': int(mono_ok[i]),
         'atten': float(atten[i])}
        for i in range(nT)],
    'run4_note': 'run4 (npz8 1a3c860f) verdict '
                 'fp_contaminated_void: preregistered '
                 'sham gate mis-specified (measured '
                 'intervention efficacy, not '
                 'contamination); corrected in run5 '
                 'per 3030 a38 precedent',
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
