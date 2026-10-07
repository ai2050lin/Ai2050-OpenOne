# -*- coding: utf-8 -*-
# Phase 3039 - Omega-P36: direct-protocol logistic
# replication (protocol robustness verdict)
# Phase 3038 established that the two-step re-entrant
# readout (all intervention phases since 3028) is a
# protocol-conditional quantity, systematically flatter
# than the natural direct readout. The logistic-readout
# finding (3035: inflection sign 10/11) and curvature
# findings (3036) were established UNDER the re-entrant
# protocol. This phase replicates the directed injection
# experiment under the DIRECT protocol: the residual
# injection is moved INTO the prefill last position
# (L-1), so the readout is the natural continuation
# distribution. Verbatim machines from 3035/3036
# (fp_inject hook, dose grid, random-direction
# control); new cross-phase anchor vs 3038 direct
# baselines. Saturation window |P0-0.5|<0.05 excluded
# from the inflection sign test (3036 lesson).
# PREREG frozen below BEFORE any observation.
import os
import json
import time
import hashlib
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PHASE = 3039
NAME = 'omega_p36_direct_logistic_replication_qwen'
BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase%d' % PHASE, NAME)
LOG = os.path.join(OUT, 'run_log.txt')
MODEL_DIR = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL = 36
HID = 2560
M_GRID = (-0.05, -0.02, -0.01, 0.0, 0.01, 0.02,
          0.05)
SITE_LATE = 35
A70_GATE = 2e-2
A71_GATE = 0.15
INFLEC_WIN = 0.05
N_MATCH_HI = 10
N_MATCH_LO = 6

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
ROW_PROMPTS = list(GEN_PROMPTS) + list(SHAM_PROMPTS)
nT = len(GEN_PROMPTS)
nS = len(SHAM_PROMPTS)
nP = nT + nS

PREREG = {
    'mode': 'DIRECT-protocol injection: fp_inject '
            'verbatim (3035/3036) applied during the '
            'PREFILL forward at decoder-layer INPUT '
            'site 35, last position = L-1 (the '
            'natural-continuation readout position); '
            'no step-2, no re-feed; logits '
            'out.logits[0,-1] double -> float64 '
            'softmax; eager attention, bf16, seed '
            '3009; every chain = fresh prefill',
    'question': 'protocol robustness of the readout '
                'mechanism chain: does the 3035 '
                'logistic-inflection sign law (positive '
                'curvature iff P0(A) < 0.5), the '
                'fingerprint-direction specificity, and '
                'the multibody coupling kappa replicate '
                'when the intervention and the readout '
                'live in the direct (natural '
                'continuation) protocol instead of the '
                're-entrant two-step protocol?',
    'tokens': 'A, B = top-2 tokens of the DIRECT base '
              'softmax at the last prefill position (no '
              'exclusion); d_fp = (W_U rows) '
              'L2-normalized float32; h_last += m * '
              '||h_last|| * d_fp in float32 then cast '
              'bf16',
    'dose': 'm in {-0.05,-0.02,-0.01,0.0,0.01,0.02,'
            '0.05} (fraction of ||h_last||) at site 35 '
            'decoder-layer input, per token k in '
            '{A,B}; random-direction control d_rand = '
            '(W_U[r1]-W_U[r2]) normalized with '
            'r1/r2 = base-prob rank 100/101 tokens, '
            'arms m in {-0.02,-0.01,0.01,0.02}',
    'T1': 'inflection sign: even1 = (P(+0.01) - 2*P0 '
          '+ P(-0.01)) / 0.01**2 on P(A); predicted '
          'sign = +1 if P0(A) < 0.5 else -1; prompts '
          'with |P0-0.5| < 0.05 EXCLUDED (saturation/'
          'inflection window, 3036 lesson); n_match '
          'over logic prompts, sham match as chance '
          'reference',
    'T2': 'fingerprint specificity: per prompt '
          '|dPA_fp(m)| vs |dPA_rand(m)| for m in '
          '{0.01, 0.02}; spec_m = median over '
          'prompts of the ratio; spec_ratio = '
          'min(spec_m01, spec_m02); gate 2',
    'T3': 'DESCRIPTIVE: kappa = -dPB(+0.05) / '
          'dPA(+0.05) (P(B) read from the A+0.05 '
          'arm full softmax); monotonicity count '
          '|dPA(0.05)| > |dPA(0.02)| > |dPA(0.01)|; '
          'A_idx odd/even symmetry med',
    'verdict_tree': 'if spec_ratio < 2 -> '
                    'direct_nonspecific_qwen; elif '
                    'n_match >= 10 (of eligible) -> '
                    'direct_logistic_robust_qwen; '
                    'elif n_match <= 6 -> '
                    'direct_logistic_breakdown_qwen; '
                    'else -> '
                    'direct_logistic_partial_qwen '
                    '(3035 gates rescaled 9/11 -> '
                    '10/12, 5/11 -> 6/12)',
    'anchors': 'a67 duplicate direct base chain '
               'logits bit-identical (0.0); a68 '
               'gate-on m=0 chain vs gate-off base '
               'bit-identical (0.0); a69 cross-phase: '
               'direct base p[A] and argmax A vs '
               '3038 npz p_dirA/a_dir over the 12 '
               'logic prompts (bit 0.0); a70 '
               'injection-entered: duplicate +0.05 '
               'arm rs bit-identical AND '
               '||rs_arm - rs_base|| / ||rs_base|| '
               'within A70_GATE=2e-2 of m; a71 manual '
               'final-norm+lm_head recompute: top-2 '
               'identity (near-tie skip note if '
               'gap<0.05) AND max|dlogit| <= '
               'A71_GATE=0.15; a72 source seals '
               '3037/3038 sha8(result) match',
    'control': 'random-direction null (3035 run6 '
               'verbatim); sham prompts (3) carry '
               'the same full arm battery as chance '
               'reference',
    'corrections': 'none yet (run1); protocol-'
                   'robustness follow-up of the 3038 '
                   'flattening finding: injection '
                   'moved INTO the prefill last '
                   'position so intervention and '
                   'readout share the direct '
                   'protocol; all arrays pre-'
                   'initialized (3020 lesson)',
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

# ---------- hooks (verbatim 3035/3036) ----------
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


# injection hooks FIRST (pre-hook chaining), then capture
handles.append(layers[SITE_LATE]
               .register_forward_pre_hook(
                   fp_inject(SITE_LATE),
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


def softmax64(lg):
    l = lg - lg.max()
    p = np.exp(l)
    return p / p.sum()


def run_direct_chain(ids, m, d_fp, gate, cap_fin=False):
    rs.clear()
    fin_cap.pop('x', None)
    state_fp['applied'] = None
    state_fp['on'] = bool(gate)
    state_fp['site'] = int(SITE_LATE)
    state_fp['m'] = float(m)
    state_fp['d'] = d_fp
    state_r['on'] = True
    state_fin['on'] = bool(cap_fin)
    with torch.no_grad():
        out = model(torch.tensor([ids],
                                 device='cuda'),
                    use_cache=True)
    state_r['on'] = False
    state_fin['on'] = False
    state_fp['on'] = False
    lg = out.logits[0, -1].detach() \
        .double().cpu().numpy()
    p = softmax64(lg)
    snap = rs[SITE_LATE][-1].copy() \
        if SITE_LATE in rs else None
    xf = fin_cap['x'][0].copy() \
        if 'x' in fin_cap else None
    applied = state_fp['applied']
    return p, lg, snap, xf, applied


tok_ids = []
for pr in ROW_PROMPTS:
    ids = tok(pr, add_special_tokens=False)[
        'input_ids']
    tok_ids.append(list(int(x) for x in ids))

# ---------- main loop ----------
p0_A = np.full(nP, np.nan)
p0_B = np.full(nP, np.nan)
tokA = np.zeros(nP, dtype=np.int64)
tokB = np.zeros(nP, dtype=np.int64)
even1 = np.full(nP, np.nan)
sign_obs = np.zeros(nP, dtype=int)
sign_pred = np.zeros(nP, dtype=int)
sign_match = np.zeros(nP, dtype=int)
a_idx = np.full(nP, np.nan)
kappa = np.full(nP, np.nan)
mono_ok = np.zeros(nP, dtype=bool)
dPA_spec = np.full((nP, 2), np.nan)
dPA_rand = np.full((nP, 4), np.nan)
even1_sham_note = ''
a67_diff = None
a68_diff = None
a69_max = None
a69_tok = None
a70_dup = None
a70_ratio_err = None
a71_maxdiff = None
a71_top2_ok = None
a71_note = ''

nM = len(M_GRID)
mi_of = {m: i for i, m in enumerate(M_GRID)}
for ri in range(nP):
    ids = tok_ids[ri]
    p_b, lg_b, snap_b, xf_b, _ = run_direct_chain(
        ids, 0.0, None, gate=False, cap_fin=(ri == 0))
    order = np.argsort(-p_b)
    A = int(order[0])
    B = int(order[1])
    tokA[ri] = A
    tokB[ri] = B
    p0_A[ri] = float(p_b[A])
    p0_B[ri] = float(p_b[B])
    wA = W_U[A]
    wB = W_U[B]
    dAB = wA - wB
    dAB = dAB / max(float(np.linalg.norm(dAB)), 1e-30)
    d_t = torch.from_numpy(dAB).float().cuda()
    d_armA = None
    # anchors on row 0
    if ri == 0:
        p_d, lg_d, _, _, _ = run_direct_chain(
            ids, 0.0, None, gate=False)
        a67_diff = float(np.max(np.abs(lg_d - lg_b)))
        p_g, lg_g, _, _, _ = run_direct_chain(
            ids, 0.0, None, gate=True)
        a68_diff = float(np.max(np.abs(lg_g - lg_b)))
        # a71 manual recompute
        w_norm = model.model.norm.weight.detach() \
            .float().cpu().numpy()
        h_fin = xf_b
        var = float((h_fin ** 2).mean())
        hn = h_fin / np.sqrt(var + EPS)
        lg_man = (W_U @ (hn * w_norm)) \
            .astype(np.float64)
        a71_maxdiff = float(np.max(
            np.abs(lg_man - lg_b)))
        srt = np.sort(lg_b)
        gap0 = float(srt[-1] - srt[-2])
        if gap0 < 0.05:
            a71_top2_ok = True
            a71_note = ('near-tie skip (gap=%.4f)'
                        % gap0)
        else:
            a71_top2_ok = bool(
                int(np.argmax(lg_man)) == A
                and int(np.argsort(-lg_man)[1]) == B)
            a71_note = ''
    # dose arms per token k in {A, B}
    p_cur = np.full((2, nM), np.nan)
    pB_in_A05 = np.full(nM, np.nan)
    snap_arm = None
    d_armA = None
    for k, tokk in enumerate((A, B)):
        wt = W_U[int(tokk)]
        d_k = torch.from_numpy(
            wt / max(float(np.linalg.norm(wt)),
                     1e-30)).float().cuda()
        if k == 0:
            d_armA = d_k
        for mi, m in enumerate(M_GRID):
            p_a, _, sn, _, _ = run_direct_chain(
                ids, m, d_k, gate=True)
            p_cur[k, mi] = float(p_a[int(tokk)])
            if k == 0:
                pB_in_A05[mi] = float(p_a[B])
            if ri == 0 and k == 0 \
                    and m == 0.05:
                snap_arm = sn
                d_armA_snap = d_k
    # a70 on row 0: duplicate arm + ratio check
    if ri == 0:
        p_d2, _, snap_d2, _, _ = run_direct_chain(
            ids, 0.05, d_armA_snap, gate=True)
        a70_dup = float(np.max(
            np.abs(snap_d2 - snap_arm)))
        num = float(np.linalg.norm(
            snap_arm - snap_b))
        den = float(np.linalg.norm(snap_b))
        a70_ratio_err = float(
            abs(num / max(den, 1e-30) - 0.05)
            / 0.05)
    # T3 (uniform across rows)
    dPA_05 = float(p_cur[0, mi_of[0.05]]) \
        - float(p_b[A])
    dPB_05 = float(pB_in_A05[mi_of[0.05]]) \
        - float(p_b[B])
    # T1 inflection at m=0.01 pair
    p_p = float(p_cur[0, mi_of[0.01]])
    p_m = float(p_cur[0, mi_of[-0.01]])
    p0 = float(p_b[A])
    even1[ri] = (p_p - 2.0 * p0 + p_m) \
        / (0.01 ** 2)
    sign_obs[ri] = 1 if even1[ri] > 0 else (-1
                                            if even1[ri] < 0
                                            else 0)
    sign_pred[ri] = 1 if p0 < 0.5 else -1
    sign_match[ri] = int(sign_obs[ri]
                         == sign_pred[ri])
    # T1 A_idx over 3 pairs
    ev_sum = 0.0
    od_sum = 0.0
    for mp, mm in ((0.01, -0.01), (0.02, -0.02),
                   (0.05, -0.05)):
        dpp = float(p_cur[0, mi_of[mp]]) - p0
        dpm = float(p_cur[0, mi_of[mm]]) - p0
        ev_sum += abs((dpp + dpm) / 2.0)
        od_sum += abs((dpp - dpm) / 2.0)
    a_idx[ri] = ev_sum / max(ev_sum + od_sum, 1e-30)
    # T3
    kappa[ri] = -dPB_05 / dPA_05 \
        if dPA_05 != 0 else float('nan')
    a05 = abs(float(p_cur[0, mi_of[0.05]]) - p0)
    a02 = abs(float(p_cur[0, mi_of[0.02]]) - p0)
    a01 = abs(float(p_cur[0, mi_of[0.01]]) - p0)
    mono_ok[ri] = bool(a05 > a02 > a01)
    # T2 random-direction control
    r1 = int(order[99])
    r2 = int(order[100])
    dr = W_U[r1] - W_U[r2]
    dr = dr / max(float(np.linalg.norm(dr)), 1e-30)
    dr_t = torch.from_numpy(dr).float().cuda()
    for j, m in enumerate((-0.02, -0.01,
                           0.01, 0.02)):
        p_r, _, _, _, _ = run_direct_chain(
            ids, m, dr_t, gate=True)
        dPA_rand[ri, j] = float(
            p_r[A]) - p0
    dPA_spec[ri, 0] = abs(
        float(p_cur[0, mi_of[0.01]]) - p0)
    dPA_spec[ri, 1] = abs(
        float(p_cur[0, mi_of[0.02]]) - p0)
    log('row%d [%s] P0=%.4f even1=%+.4f pred=%+d '
        'obs=%+d match=%d kappa=%.3f mono=%s'
        % (ri, 'logic' if ri < nT else 'sham',
           p0, even1[ri], sign_pred[ri],
           sign_obs[ri], sign_match[ri],
           kappa[ri], bool(mono_ok[ri])))

# ---------- a69 cross-phase vs 3038 ----------
z38 = np.load(os.path.join(
    BASE, 'phase3038',
    'omega_p35_reentrant_readout_qwen',
    'omega_p35_reentrant_readout_qwen.npz'),
    allow_pickle=True)
a38_pdirA = z38['p_dirA']
a38_adir = z38['a_dir']
a69_max = 0.0
for pi in range(nT):
    a69_max = max(a69_max, abs(
        float(p0_A[pi]) - float(a38_pdirA[pi])))
a69_max = float(a69_max)
a69_tok = bool((tokA[:nT] == a38_adir[:nT]).all())
log('a67=%r a68=%r a69 max_dp=%.3e tok_ok=%s '
    'a70 dup=%r ratio_err=%.4f a71 maxdiff=%.4f '
    'top2=%s %s'
    % (a67_diff, a68_diff, a69_max, a69_tok,
       a70_dup, a70_ratio_err, a71_maxdiff,
       a71_top2_ok, a71_note))

# ---------- a72 source seals ----------
a72_detail = []
for ph, nm in ((3037,
                'omega_p34_kv_situational_'
                'specificity_qwen'),
               (3038,
                'omega_p35_reentrant_readout_'
                'qwen')):
    d = os.path.join(BASE, 'phase%d' % ph, nm)
    with open(os.path.join(d, 'seal.json'),
              encoding='utf-8') as f:
        sealj = json.load(f)
    with open(os.path.join(d, 'result.json'),
              'rb') as f:
        s = hashlib.sha256(f.read()).hexdigest()[:8]
    a72_detail.append(bool(
        s == sealj['result_sha256_8']))
a72_ok = bool(a72_detail) and all(a72_detail)

# ---------- statistics ----------
elig = np.abs(p0_A[:nT] - 0.5) >= INFLEC_WIN
n_elig = int(elig.sum())
n_match = int(sign_match[:nT][elig].sum())
n_excl = nT - n_elig
sham_match = int(sign_match[nT:].sum())
med_a_idx = float(np.nanmedian(a_idx[:nT]))
med_kappa = float(np.nanmedian(np.abs(
    kappa[:nT])))
n_mono = int(mono_ok[:nT].sum())
spec_m01 = float(np.median(
    dPA_spec[:nT, 0]
    / np.maximum(np.abs(dPA_rand[:nT, 2]),
                 1e-12)))
spec_m02 = float(np.median(
    dPA_spec[:nT, 1]
    / np.maximum(np.abs(dPA_rand[:nT, 3]),
                 1e-12)))
spec_ratio = float(min(spec_m01, spec_m02))

# ---------- verdict ----------
a_ok = bool(a67_diff == 0.0 and a68_diff == 0.0
            and a69_max == 0.0 and a69_tok
            and a70_dup == 0.0
            and a70_ratio_err <= A70_GATE
            and a71_top2_ok
            and a71_maxdiff <= A71_GATE
            and a72_ok)
if spec_ratio < 2:
    verdict = 'direct_nonspecific_qwen'
elif n_match >= N_MATCH_HI:
    verdict = 'direct_logistic_robust_qwen'
elif n_match <= N_MATCH_LO:
    verdict = 'direct_logistic_breakdown_qwen'
else:
    verdict = 'direct_logistic_partial_qwen'

log('=== verdict ===')
log('a67=%r a68=%r a69=%.3e/%s a70=%r/%.4f '
    'a71=%.4f/%s a72=%r'
    % (a67_diff, a68_diff, a69_max, a69_tok,
       a70_dup, a70_ratio_err, a71_maxdiff,
       a71_top2_ok, a72_ok))
log('eligible=%d excluded=%d n_match=%d '
    'sham_match=%d/%d'
    % (n_elig, n_excl, n_match, sham_match, nS))
log('med_A_idx=%.4f med_|kappa|=%.4f n_mono=%d '
    'spec_m01=%.2f spec_m02=%.2f spec=%.2f'
    % (med_a_idx, med_kappa, n_mono, spec_m01,
       spec_m02, spec_ratio))
log('VERDICT=%s anchor_all_ok=%s'
    % (verdict, a_ok))

elapsed = time.time() - t0

# ---------- npz (flat arrays only) ----------
npz_path = os.path.join(OUT, NAME + '.npz')
np.savez(
    npz_path,
    prompts=np.array(ROW_PROMPTS),
    tokA=tokA, tokB=tokB,
    p0_A=p0_A, p0_B=p0_B,
    even1=even1, sign_obs=sign_obs,
    sign_pred=sign_pred, sign_match=sign_match,
    a_idx=a_idx, kappa=kappa, mono_ok=mono_ok,
    dPA_spec=dPA_spec, dPA_rand=dPA_rand,
    n_elig=np.int64(n_elig), n_excl=np.int64(n_excl),
    n_match=np.int64(n_match),
    sham_match=np.int64(sham_match),
    med_a_idx=np.float64(med_a_idx),
    med_kappa=np.float64(med_kappa),
    n_mono=np.int64(n_mono),
    spec_m01=np.float64(spec_m01),
    spec_m02=np.float64(spec_m02),
    spec_ratio=np.float64(spec_ratio),
    a67_diff=np.float64(a67_diff),
    a68_diff=np.float64(a68_diff),
    a69_max=np.float64(a69_max),
    a69_tok=np.bool_(a69_tok),
    a70_dup=np.float64(a70_dup),
    a70_ratio_err=np.float64(a70_ratio_err),
    a71_maxdiff=np.float64(a71_maxdiff),
    a71_top2_ok=np.bool_(a71_top2_ok),
    a72_ok=np.bool_(a72_ok),
    verdict=np.array(verdict),
    elapsed=np.float64(elapsed))

result = {
    'phase': PHASE, 'name': NAME, 'created': created,
    'final_verdict': verdict,
    'anchor_all_ok': a_ok,
    'anchors': {
        'a67_dup_base_bit': a67_diff,
        'a68_gate_m0_bit': a68_diff,
        'a69_max_dp_3038': a69_max,
        'a69_tok_identity': a69_tok,
        'a70_dup_rs_bit': a70_dup,
        'a70_ratio_err': a70_ratio_err,
        'a70_gate': A70_GATE,
        'a71_maxdiff': a71_maxdiff,
        'a71_top2_ok': a71_top2_ok,
        'a71_gate': A71_GATE,
        'a71_note': a71_note,
        'a72_source_seals': a72_ok,
    },
    'T1_inflection': {
        'n_eligible': n_elig,
        'n_excluded_saturation': n_excl,
        'n_match': n_match,
        'sham_match': sham_match,
        'sham_n': nS,
        'per_row': [{
            'row': i,
            'kind': 'logic' if i < nT else 'sham',
            'P0': float(p0_A[i]),
            'even1': float(even1[i]),
            'sign_pred': int(sign_pred[i]),
            'sign_obs': int(sign_obs[i]),
            'match': int(sign_match[i]),
        } for i in range(nP)],
    },
    'T2_specificity': {
        'spec_m01': spec_m01,
        'spec_m02': spec_m02,
        'spec_ratio': spec_ratio,
    },
    'T3_competition': {
        'med_A_idx': med_a_idx,
        'med_abs_kappa': med_kappa,
        'n_mono': n_mono,
    },
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
    'anchor_all_ok': a_ok,
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
