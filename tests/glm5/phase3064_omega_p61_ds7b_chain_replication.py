# -*- coding: utf-8 -*-
"""Phase 3064: Omega-P61 - DS7B full-chain
replication (KV ladder + last-layer gate anatomy +
gamma/PC1 pipe + write-side ANOVA + adversarial
same-source), deepseek-r1-distill-qwen-7b bf16.

Plan 3064 A (menu of 3063). The qwen3-4b chain
(3044-3063) is replicated on the distill model:
S1 single-layer ladder (V-only / K-only per layer
x 24 pairs), S2 last-layer anatomy (V/K/joint +
per-kv-head), S3 norm/gamma/PC1 pipe with the
exact-norm tangential counterfactual, S4 write-
side two-way ANOVA (body identity), S5 adversarial
same-source (V vs most-adversarial K head).

Qwen2 differences vs Qwen3 (documented): no
q_norm/k_norm -> replacement point is the k_proj
output (pre-RoPE); GQA 28q/4kv (kv head block =
128 dims, 4 blocks); 28 layers; hidden 3584;
vocab 152064; rope_theta 10000. bf16 weights
(7.6B fp32 exceeds the 16 GB GPU) - all bit
anchors are internal same-precision; the RMSNorm
counterfactual is computed manually in fp64 with
the module gamma/eps to avoid bf16 rounding in
the precision-sensitive step.

Verdict (single branch, preregistered):
setup_failed_ds7b / chain_replicated_ds7b (5
stage passes) / chain_partial_ds7b (3-4) /
chain_fragmented_ds7b (1-2) / chain_absent_ds7b
(0).
"""
import hashlib
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, \
    AutoTokenizer

PHASE = 3064
NAME = 'omega_p61_ds7b_chain_replication'
BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase%d' % PHASE, NAME)
LOG = os.path.join(OUT, 'run_log.txt')
MODEL_DIR = (r'D:\AI2050\Ai2050-OpenOne\models\hf'
             r'\deepseek-r1-distill-qwen-7b')

NL = 28
KV_HEAD = 4
NQ_HEAD = 28
HDIM = 128
HID = 3584
SEED_MAIN = 3010
FRONT = 4
L_TGT = 27
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
    'mode': 'bf16 MODEL (native dtype of the '
            'distill checkpoint; 7.6B fp32 exceeds '
            'the 16 GB GPU - precision reduction '
            'registered, all bit anchors internal '
            'same-precision, permutation nulls '
            'same-precision); eager attention, '
            'seed 3010; fresh capture banks (no '
            'historical DS7B bank exists)',
    'question': '3064 A (menu of 3063): is the '
                'qwen3-4b Omega-P2 chain (single-'
                'layer ladder rising to the last '
                'layer, last-layer K-carry + V '
                'adversarial flip, gamma/S16 pipe '
                '+ readout PC1, write-side body '
                'identity, adversarial same-source '
                'of V and K arms) universal or '
                'qwen-specific? Replicated on '
                'deepseek-r1-distill-qwen-7b '
                '(Qwen2 arch, 28L, GQA 28q/4kv).',
    'S1_ladder': 'per layer l (0..27) x 24 pairs: '
                 'V-only arm (rvV[l, FRONT rows, :] '
                 '= srcV[l]) and K-only full-field '
                 'arm (rkK[l, FRONT rows, :] = '
                 'rotK[l]); metric cos(lg - LG0, '
                 'TT[k]) med over pairs; PASS = '
                 'medK(27) > 0.15 AND medV(27) < 0 '
                 'AND mean(medK[l>=14]) > '
                 'mean(medK[l<14])',
    'S2_gate': 'L27 anatomy: joint full K+V at '
               'L27; K_share = medK/medJ >= 0.8; '
               'head concentration: max_h medJh '
               '>= 0.6 * medJ_full; per-kv-head '
               'joint arms (4 heads x 24 pairs); '
               'adversarial head = argmin_h medJh',
    'S3_pipe': 'stage captures at L27 (pre/op/'
               'post + fp64 manual RMSNorm '
               'counterfactual); v_adv_tan = '
               'med(V_COS_DTAN_ONLY) < 0; gamma '
               'skew max/med >= 2; TOP64/FLAT64/'
               'S16 by |gamma - mean|; PC1 of '
               'Vmat family sv_share0 >= 0.2',
    'S4_write': 'two-way centered ANOVA of the '
                'V-arm d_tan family (24x3584): '
                'SS_body >= 0.5 AND mean per-body '
                'd_eff < mean per-prefix d_eff; '
                'between-body |cos| matrix '
                'descriptive',
    'S5_source': 'V arm vs most-adversarial K '
                 'head arm: SS1 diag perm (seed '
                 '9998), SS2 shared axis vs '
                 'W_U-image null (seed 9999), '
                 'SS3 top-256 support Jaccard '
                 '(seed 10002); ss_count = #{p <= '
                 '0.005}; specificity comparator = '
                 'most-positive K head (perm seed '
                 '10000); axis_dominance required; '
                 'channel identity of both '
                 'adversarial arms vs S16/TOP64 '
                 '(seeds 10003-10008 / 10009-'
                 '10014); T6 anti-payload + PC1 '
                 'alignment + rotation null (seed '
                 '10015)',
    'verdict': 'setup anchors fail -> '
               'setup_failed_ds7b; count = '
               '#{S1..S5 pass}: 5 -> '
               'chain_replicated_ds7b; 3-4 -> '
               'chain_partial_ds7b; 1-2 -> '
               'chain_fragmented_ds7b; 0 -> '
               'chain_absent_ds7b; single branch '
               'assigned inside the criteria',
    'anchors': 'b0 recapture 4 prompts bit 0.0 '
               '(LG+KB+VB); b1 sham self-'
               'replacement bit identity; b2 '
               'stage-pre identity under L27 '
               'replacement (max diff 0.0); b3 '
               'gamma/wud/TT finite; b5 '
               'diagnostic past-key validation '
               '(recorded, not asserted - bf16 '
               'rounding)',
    'statistics_discipline': 'same-precision '
                             'permutation nulls on '
                             'frozen seeds; PC1 sign '
                             'arbitrary - |cos| '
                             'everywhere; no cross-'
                             'space cosine (logit '
                             'space only via Y = D '
                             '@ W_U^T); verdict in '
                             'one branch',
    'memory_discipline': 'wud fp64 (4.4 GB) held '
                         'only for Vmat/PC1/Y '
                         'projections then del; '
                         'banks fp64 ~122 MB; '
                         'permutation nulls per-'
                         'draw small argsorts',
}

lines = []


def log(msg):
    lines.append(str(msg))
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(str(msg) + '\n')
    print(msg)


def sha8(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(
            f.read()).hexdigest()[:8]


os.makedirs(OUT, exist_ok=True)
for fn in (NAME + '.npz', 'run_log.txt',
           'execution.json', 'result.json',
           'seal.json'):
    p = os.path.join(OUT, fn)
    if os.path.exists(p):
        os.remove(p)
t0 = time.time()
created = time.strftime('%Y-%m-%d %H:%M:%S')
execution = {'phase': PHASE, 'name': NAME,
             'created': created, 'prereg': PREREG}
with open(os.path.join(OUT, 'execution.json'), 'w',
          encoding='utf-8') as f:
    json.dump(execution, f, ensure_ascii=False,
              indent=1)
log('execution.json written (prereg frozen) %s'
    % created)

torch.manual_seed(SEED_MAIN)
np.random.seed(SEED_MAIN)

tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, torch_dtype=torch.bfloat16,
    attn_implementation='eager').to('cuda').eval()
layers = model.model.layers
assert len(layers) == NL
assert int(model.config.num_key_value_heads) \
    == KV_HEAD
assert int(model.config.num_attention_heads) \
    == NQ_HEAD
assert int(model.config.hidden_size) == HID
rot = model.model.rotary_emb
INV_FREQ = rot.inv_freq.detach().cpu().numpy() \
    .astype(np.float32)
assert INV_FREQ.shape == (HDIM // 2,)
assert float(getattr(rot, 'attention_scaling',
                     1.0)) == 1.0
NVOC = int(model.config.vocab_size)
EPS_NORM = float(model.config.rms_norm_eps)
log('model loaded bf16 (vocab=%d rms_eps=%g '
    'layers=%d kvheads=%d hdim=%d)'
    % (NVOC, EPS_NORM, NL, KV_HEAD, HDIM))

KVW = KV_HEAD * HDIM  # 512


def rot_apply(x, delta):
    """Rotate (..., 4, 128) by delta*inv_freq
    (NeoX half-pairing, fp32)."""
    ang = INV_FREQ * np.float32(delta)
    emb = np.concatenate([ang, ang]).astype(
        np.float32)
    c = np.cos(emb)[None, None, :]
    s = np.sin(emb)[None, None, :]
    h = HDIM // 2
    x1 = x[..., :h]
    x2 = x[..., h:]
    rh = np.concatenate([-x2, x1], axis=-1)
    return x * c + rh * s


stateK = {li: {'repl': None, 'mask': None}
          for li in range(NL)}
stateV = {li: {'repl': None, 'mask': None}
          for li in range(NL)}
capK = {li: {'rec': False, 'orig': None}
        for li in range(NL)}
capV = {li: {'rec': False, 'orig': None}
        for li in range(NL)}
cap_stage = {'pre': {'rec': False, 'v': None},
             'op': {'rec': False, 'v': None},
             'post': {'rec': False, 'v': None}}


def hook_k(st, cp):
    def h(module, inp, out):
        if cp['rec']:
            cp['orig'] = out[0].detach().clone()
        if st['repl'] is not None:
            out[0][st['mask']] = st['repl']
        return out
    return h


def hook_v(st, cp):
    def h(module, inp, out):
        if cp['rec']:
            cp['orig'] = out[0].detach().clone()
        if st['repl'] is not None:
            out[0][st['mask']] = st['repl']
        return out
    return h


def hook_stage(cp):
    def h(module, inp, out):
        if cp['rec']:
            t = out[0] if isinstance(out, tuple) \
                else out
            cp['v'] = t[0][-1].detach().clone()
        return out
    return h


for li in range(NL):
    layers[li].self_attn.k_proj \
        .register_forward_hook(hook_k(
            stateK[li], capK[li]))
    layers[li].self_attn.v_proj \
        .register_forward_hook(hook_v(
            stateV[li], capV[li]))
layers[NL - 2].register_forward_hook(
    hook_stage(cap_stage['pre']))
layers[L_TGT].self_attn.o_proj \
    .register_forward_hook(hook_stage(
        cap_stage['op']))
layers[L_TGT].register_forward_hook(
    hook_stage(cap_stage['post']))


def reset_all():
    for li in range(NL):
        stateK[li]['repl'] = None
        stateK[li]['mask'] = None
        stateV[li]['repl'] = None
        stateV[li]['mask'] = None
        capK[li]['rec'] = False
        capV[li]['rec'] = False
    for kk in cap_stage:
        cap_stage[kk]['rec'] = False
        cap_stage[kk]['v'] = None


def forward_plain(ids):
    reset_all()
    with torch.no_grad():
        out = model(torch.tensor([ids],
                    device='cuda'),
                    use_cache=True)
    lg = out.logits[0, -1].detach().double() \
        .cpu().numpy()
    reset_all()
    return lg


def forward_cap(ids):
    reset_all()
    for li in range(NL):
        capK[li]['rec'] = True
        capV[li]['rec'] = True
    with torch.no_grad():
        out = model(torch.tensor([ids],
                    device='cuda'),
                    use_cache=True)
    lg = out.logits[0, -1].detach().double() \
        .cpu().numpy()
    n = len(ids)
    kb = np.stack([capK[li]['orig'].double()
                   .cpu().numpy()
                   for li in range(NL)])
    vb = np.stack([capV[li]['orig'].double()
                   .cpu().numpy()
                   for li in range(NL)])
    reset_all()
    return lg, kb, vb


def forward_stage(ids, replK, maskK, replV,
                  maskV):
    reset_all()
    rk = torch.tensor(np.ascontiguousarray(
        replK), dtype=torch.bfloat16,
        device='cuda')
    mk = torch.tensor(np.asarray(maskK),
                      device='cuda')
    rv = torch.tensor(np.ascontiguousarray(
        replV), dtype=torch.bfloat16,
        device='cuda')
    mv = torch.tensor(np.asarray(maskV),
                      device='cuda')
    for li in range(NL):
        stateK[li]['repl'] = rk[li]
        stateK[li]['mask'] = mk
        stateV[li]['repl'] = rv[li]
        stateV[li]['mask'] = mv
    for kk in cap_stage:
        cap_stage[kk]['rec'] = True
    with torch.no_grad():
        out = model(torch.tensor([ids],
                    device='cuda'),
                    use_cache=True)
    lg = out.logits[0, -1].detach().double() \
        .cpu().numpy()
    pre = cap_stage['pre']['v'].double() \
        .cpu().numpy()
    op = cap_stage['op']['v'].double() \
        .cpu().numpy()
    post = cap_stage['post']['v'].double() \
        .cpu().numpy()
    reset_all()
    return lg, pre, op, post


# ---------- assemble prompts ----------
word_tok = {}
for w in TARGETS:
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
LENS = np.array([len(assembled[i]['ids'])
                 for i in range(n_pr)])
NMAX = int(LENS.max())
log('assembled %d prompts (lens %d-%d)'
    % (n_pr, int(LENS.min()), NMAX))

# ---------- capture banks (48 seqs) ----------
LG = np.zeros((n_pr, NVOC))
KB = np.zeros((n_pr, NL, NMAX, KVW))
VB = np.zeros((n_pr, NL, NMAX, KVW))
for i in range(n_pr):
    lg, kb, vb = forward_cap(assembled[i]['ids'])
    n = int(LENS[i])
    LG[i] = lg
    KB[i, :, :n, :] = kb
    VB[i, :, :n, :] = vb
log('banks captured: LG%s KB%s VB%s'
    % (LG.shape, KB.shape, VB.shape))

# b0: recapture anchor
b0_diff = 0.0
for si in (0, 9, 17, 31):
    lg2, kb2, vb2 = forward_cap(
        assembled[si]['ids'])
    n2 = int(LENS[si])
    b0_diff = max(b0_diff, float(np.max(
        np.abs(LG[si] - lg2))))
    b0_diff = max(b0_diff, float(np.max(
        np.abs(KB[si, :, :n2, :] - kb2))))
    b0_diff = max(b0_diff, float(np.max(
        np.abs(VB[si, :, :n2, :] - vb2))))
b0_diff = float(b0_diff)
b0_ok = bool(b0_diff == 0.0)
log('b0 recapture diff=%.3e ok=%s'
    % (b0_diff, b0_ok))

cidx = []
bidx = []
for ci in (1, 2, 3):
    for bi in range(len(BODIES)):
        cidx.append(ci)
        bidx.append(bi)
cidx = np.array(cidx)
bidx = np.array(bidx)
NP_ = 24

TT = np.stack([
    LG[idx_of[(int(cidx[k]), int(bidx[k]))]]
    - LG[idx_of[(0, int(bidx[k]))]]
    for k in range(NP_)])


def front_rows(nb):
    return list(range(min(FRONT, nb)))


NFR_ALL = None
for k in range(NP_):
    nb = int(LENS[idx_of[(0, int(bidx[k]))]])
    if NFR_ALL is None:
        NFR_ALL = len(front_rows(nb))
    assert len(front_rows(nb)) == NFR_ALL, k
assert NFR_ALL == FRONT


def pair_fields(k):
    """Base self fields + L_TGT source fields
    for the FRONT rows (K RoPE-compensated by
    off, V raw)."""
    b = int(bidx[k])
    c = int(cidx[k])
    base_i = idx_of[(0, b)]
    pref_i = idx_of[(c, b)]
    nb = int(LENS[base_i])
    off = assembled[pref_i]['off']
    rows = front_rows(nb)
    ridx = np.asarray(rows)
    repK = KB[base_i, :, :nb, :].copy()
    repV = VB[base_i, :, :nb, :].copy()
    srcK = KB[pref_i, :, off + ridx, :] \
        .reshape(NL, len(rows), KV_HEAD, HDIM)
    rotK = rot_apply(srcK, off) \
        .reshape(NL, len(rows), KVW)
    srcV = VB[pref_i][:, off + ridx, :]
    return repK, repV, rotK, srcV, rows, \
        base_i, off


def cos_frac(lg_base, r, t):
    nr = float(np.linalg.norm(r))
    nt = float(np.linalg.norm(t))
    cs = float(r @ t) / (nr * nt) \
        if nr > 1e-12 and nt > 1e-12 else 0.0
    fr = nr / nt if nt > 1e-12 else float('nan')
    return cs, fr


# b1: sham self-replacement (bit identity)
k0 = 0
base0 = idx_of[(0, int(bidx[k0]))]
ids0 = assembled[base0]['ids']
n0 = int(LENS[base0])
selfK = KB[base0, :, :n0, :].copy()
selfV = VB[base0, :, :n0, :].copy()
m_all = np.ones(n0, dtype=bool)
lg_s, _, _, _ = forward_stage(
    ids0, selfK, m_all, selfV, m_all)
b1_diff = float(np.max(np.abs(lg_s
                              - LG[base0])))
b1_ok = bool(b1_diff == 0.0)
log('b1 sham self-replacement diff=%.3e ok=%s'
    % (b1_diff, b1_ok))

# b5 diagnostic: past-key validation (pair 0,
# layer L_TGT, row 0) - recorded, not asserted
repK5, repV5, rotK5, srcV5, rows5, base5, off5 \
    = pair_fields(0)
pref5 = idx_of[(int(cidx[0]), int(bidx[0]))]
ids5 = assembled[base5]['ids']
n5 = int(LENS[base5])
m5 = np.ones(n5, dtype=bool)
rkK = repK5.copy()
rkK[L_TGT, rows5, :] = rotK5[L_TGT]
reset_all()
rk_t = torch.tensor(np.ascontiguousarray(rkK),
                    dtype=torch.bfloat16,
                    device='cuda')
rv_t = torch.tensor(np.ascontiguousarray(repV5),
                    dtype=torch.bfloat16,
                    device='cuda')
mk5 = torch.tensor(m5, device='cuda')
for li in range(NL):
    stateK[li]['repl'] = rk_t[li]
    stateK[li]['mask'] = mk5
    stateV[li]['repl'] = rv_t[li]
    stateV[li]['mask'] = mk5
with torch.no_grad():
    out_r = model(torch.tensor([ids5],
                  device='cuda'), use_cache=True)
# transformers 5.x: only cache.layers[li].keys (probe 2026-09-21)
_kl = out_r.past_key_values.layers[L_TGT].keys
pk_r = _kl[0, :, rows5[0]].double() \
    .cpu().numpy()
reset_all()
with torch.no_grad():
    out_p = model(torch.tensor(
        [assembled[pref5]['ids']],
        device='cuda'), use_cache=True)
_kl = out_p.past_key_values.layers[L_TGT].keys
pk_p = _kl[0, :, off5 + rows5[0]] \
    .double().cpu().numpy()
b5_rel = float(np.linalg.norm(pk_r - pk_p)
               / max(np.linalg.norm(pk_p), 1e-12))
log('b5 diagnostic past-key rel diff=%.3e '
    '(recorded, not asserted)' % b5_rel)

# gamma / weights
GAMMA_T = model.model.norm.weight.detach() \
    .clone()
GAMMA = GAMMA_T.double().cpu().numpy()
assert GAMMA.shape == (HID,)
gm_mean = float(GAMMA.mean())
gamma_skew = float(GAMMA.max()) \
    / max(float(np.median(GAMMA)), 1e-12)
order_shape = np.argsort(
    np.abs(GAMMA - gm_mean))[::-1]
TOP64 = order_shape[:64].astype(np.int64)
FLAT64 = order_shape[::-1][:64].astype(np.int64)
S16 = TOP64[:16].astype(np.int64)
b3_ok = bool(np.isfinite(GAMMA).all()
             and np.isfinite(TT).all())
log('gamma stats: min=%.4f max=%.4f mean=%.4f '
    'std=%.4f skew(max/med)=%.2f'
    % (GAMMA.min(), GAMMA.max(), gm_mean,
       GAMMA.std(), gamma_skew))

WU = model.lm_head.weight.detach()
wud = WU.double().cpu().numpy()  # (NVOC, HID)

Vmat = np.zeros((NP_, HID))
for k in range(NP_):
    Vmat[k] = GAMMA * (wud.T @ TT[k])
Ur_, svr_, Vtr_ = np.linalg.svd(
    Vmat, full_matrices=False)
PC1_RO = Vtr_[0]
sv_share0 = float(svr_[0] ** 2
                  / (svr_ ** 2).sum())
log('Vmat/PC1: sv_share0=%.4f' % sv_share0)


def rms_norm_fp64(x):
    ms = float((x * x).mean()) + EPS_NORM
    return x / np.sqrt(ms) * GAMMA


# ---------- S1: single-layer ladder ----------
log('=== S1 ladder: V-only / K-only per layer '
    '(24 pairs each) ===')
COS_LAD_V = np.zeros((NL, NP_))
COS_LAD_K = np.zeros((NL, NP_))
for k in range(NP_):
    repK, repV, rotK, srcV, rows, base_i, off \
        = pair_fields(k)
    nb = int(LENS[base_i])
    m = np.ones(nb, dtype=bool)
    lg0 = LG[base_i]
    t_k = TT[k]
    for li in range(NL):
        rvV = repV.copy()
        rvV[li, rows, :] = srcV[li]
        lg_v, _, _, _ = forward_stage(
            assembled[base_i]['ids'],
            repK, m, rvV, m)
        COS_LAD_V[li, k] = cos_frac(
            lg0, lg_v - lg0, t_k)[0]
        rkK = repK.copy()
        rkK[li, rows, :] = rotK[li]
        lg_k, _, _, _ = forward_stage(
            assembled[base_i]['ids'],
            rkK, m, repV, m)
        COS_LAD_K[li, k] = cos_frac(
            lg0, lg_k - lg0, t_k)[0]
medV_lad = np.median(COS_LAD_V, axis=1)
medK_lad = np.median(COS_LAD_K, axis=1)
late = np.arange(NL) >= NL // 2
s1_late_gt = bool(medK_lad[late].mean()
                  > medK_lad[~late].mean())
s1_k27 = bool(medK_lad[L_TGT] > 0.15)
s1_v27 = bool(medV_lad[L_TGT] < 0.0)
s1_pass = bool(s1_late_gt and s1_k27
               and s1_v27)
log('S1: medK(27)=%.4f medV(27)=%.4f '
    'late=%.4f early=%.4f peakK=%d pass=%s'
    % (medK_lad[L_TGT], medV_lad[L_TGT],
       medK_lad[late].mean(),
       medK_lad[~late].mean(),
       int(np.argmax(np.abs(medK_lad))),
       s1_pass))
log('S1 profile K: %s'
    % ' '.join('%.2f' % v for v in medK_lad))
log('S1 profile V: %s'
    % ' '.join('%.2f' % v for v in medV_lad))

# ---------- S2: L27 anatomy ----------
log('=== S2 L27 anatomy ===')
COS_V27 = np.zeros(NP_)
COS_KF = np.zeros(NP_)
COS_JF = np.zeros(NP_)
COS_JH = np.zeros((KV_HEAD, NP_))
for k in range(NP_):
    repK, repV, rotK, srcV, rows, base_i, off \
        = pair_fields(k)
    nb = int(LENS[base_i])
    m = np.ones(nb, dtype=bool)
    lg0 = LG[base_i]
    t_k = TT[k]
    rvV = repV.copy()
    rvV[L_TGT, rows, :] = srcV[L_TGT]
    lg_v, _, _, _ = forward_stage(
        assembled[base_i]['ids'],
        repK, m, rvV, m)
    COS_V27[k] = cos_frac(lg0, lg_v - lg0,
                          t_k)[0]
    rkK = repK.copy()
    rkK[L_TGT, rows, :] = rotK[L_TGT]
    lg_k, _, _, _ = forward_stage(
        assembled[base_i]['ids'],
        rkK, m, repV, m)
    COS_KF[k] = cos_frac(lg0, lg_k - lg0,
                         t_k)[0]
    rjK = repK.copy()
    rjV = repV.copy()
    rjK[L_TGT, rows, :] = rotK[L_TGT]
    rjV[L_TGT, rows, :] = srcV[L_TGT]
    lg_j, _, _, _ = forward_stage(
        assembled[base_i]['ids'],
        rjK, m, rjV, m)
    COS_JF[k] = cos_frac(lg0, lg_j - lg0,
                         t_k)[0]
    for hh in range(KV_HEAD):
        sl = slice(hh * HDIM, (hh + 1) * HDIM)
        rhK = repK.copy()
        rhV = repV.copy()
        rhK[L_TGT, rows, sl] = \
            rotK[L_TGT].reshape(
                len(rows), KV_HEAD, HDIM)[:, hh, :]
        rhV[L_TGT, rows, sl] = \
            srcV[L_TGT].reshape(
                len(rows), KV_HEAD, HDIM)[:, hh, :]
        lg_h, _, _, _ = forward_stage(
            assembled[base_i]['ids'],
            rhK, m, rhV, m)
        COS_JH[hh, k] = cos_frac(lg0,
                                 lg_h - lg0,
                                 t_k)[0]
medV = float(np.median(COS_V27))
medK = float(np.median(COS_KF))
medJ = float(np.median(COS_JF))
medJh = np.median(COS_JH, axis=1)
k_share = medK / medJ if abs(medJ) > 1e-12 \
    else float('nan')
head_max = float(np.max(medJh))
h_adv = int(np.argmin(medJh))
h_pos = int(np.argmax(medJh))
s2_share = bool(k_share >= 0.8)
s2_conc = bool(head_max >= 0.6 * abs(medJ)
               and medJ > 0)
s2_pass = bool(s2_share and s2_conc)
log('S2: medV=%.4f medK=%.4f medJ=%.4f '
    'K_share=%.3f medJh=%s pass=%s'
    % (medV, medK, medJ, k_share,
       np.round(medJh, 4).tolist(), s2_pass))
log('S2: adversarial head=%d (med %.4f), '
    'positive head=%d (med %.4f)'
    % (h_adv, medJh[h_adv], h_pos,
       medJh[h_pos]))

# ---------- S3: stage captures + pipe ----------
log('=== S3 stage arms: V / K_adv / K_pos / '
    'joint-h-adv ===')
D_V = np.zeros((NP_, HID))
D_KA = np.zeros((NP_, HID))
D_KP = np.zeros((NP_, HID))
V_COS_D = np.zeros(NP_)
V_COS_DTAN = np.zeros(NP_)
V_COS_DTAN_ONLY = np.zeros(NP_)
V_FRAC_RAD = np.zeros(NP_)
V_TAN_FRAC = np.zeros(NP_)
V_COS_LG = np.zeros(NP_)
KA_COS_D = np.zeros(NP_)
KA_COS_DTAN_ONLY = np.zeros(NP_)
a233_pre = 0.0
for k in range(NP_):
    repK, repV, rotK, srcV, rows, base_i, off \
        = pair_fields(k)
    nb = int(LENS[base_i])
    m = np.ones(nb, dtype=bool)
    ids_b = assembled[base_i]['ids']
    lg_b, pre_b, op_b, post_b = forward_stage(
        ids_b, repK, m, repV, m)
    xb = post_b
    nx2 = float(xb @ xb)
    xhat = xb / np.sqrt(nx2)
    t_k = TT[k]
    lg0 = LG[base_i]
    # V arm with stage capture
    rvV = repV.copy()
    rvV[L_TGT, rows, :] = srcV[L_TGT]
    lg_v, pre_v, op_v, post_v = forward_stage(
        ids_b, repK, m, rvV, m)
    a233_pre = max(a233_pre, float(np.max(
        np.abs(pre_v - pre_b))))
    d = post_v - post_b
    D_V[k] = d - float(d @ xhat) * xhat
    V_COS_D[k] = cos_frac(lg0, wud @ d,
                          t_k)[0]
    ddot = float(d @ xhat)
    nd = float(np.linalg.norm(d))
    V_FRAC_RAD[k] = abs(ddot) / nd \
        if nd > 1e-12 else 0.0
    dt = d - ddot * xhat
    V_TAN_FRAC[k] = float(np.linalg.norm(dt)) \
        / nd if nd > 1e-12 else 0.0
    V_COS_DTAN[k] = cos_frac(lg0, wud @ dt,
                             t_k)[0]
    rms_b = float(np.sqrt(nx2 / HID
                          + EPS_NORM))
    z_pred = wud @ (GAMMA * dt / rms_b)
    r_lg = lg_v - lg0
    nz = float(np.linalg.norm(z_pred))
    nr = float(np.linalg.norm(r_lg))
    V_COS_LG[k] = float(z_pred @ r_lg) \
        / (nz * nr) \
        if nz > 1e-12 and nr > 1e-12 else 0.0
    postn_b = rms_norm_fp64(xb)
    cf = rms_norm_fp64(xb + dt) - postn_b
    V_COS_DTAN_ONLY[k] = cos_frac(
        lg0, wud @ cf, t_k)[0]
    # K adversarial head arm
    sl_a = slice(h_adv * HDIM,
                 (h_adv + 1) * HDIM)
    raK = repK.copy()
    raK[L_TGT, rows, sl_a] = \
        rotK[L_TGT].reshape(
            len(rows), KV_HEAD,
            HDIM)[:, h_adv, :]
    lg_a, _, _, post_a = forward_stage(
        ids_b, raK, m, repV, m)
    da = post_a - post_b
    D_KA[k] = da - float(da @ xhat) * xhat
    KA_COS_D[k] = cos_frac(lg0, wud @ da,
                           t_k)[0]
    KA_COS_DTAN_ONLY[k] = cos_frac(
        lg0, wud @ (rms_norm_fp64(xb
            + (da - float(da @ xhat) * xhat))
            - postn_b), t_k)[0]
    # K positive head arm
    sl_p = slice(h_pos * HDIM,
                 (h_pos + 1) * HDIM)
    rpK = repK.copy()
    rpK[L_TGT, rows, sl_p] = \
        rotK[L_TGT].reshape(
            len(rows), KV_HEAD,
            HDIM)[:, h_pos, :]
    lg_p, _, _, post_p = forward_stage(
        ids_b, rpK, m, repV, m)
    dp = post_p - post_b
    D_KP[k] = dp - float(dp @ xhat) * xhat
med_v_d = float(np.median(V_COS_D))
med_v_dtan = float(np.median(V_COS_DTAN))
med_v_dtonly = float(
    np.median(V_COS_DTAN_ONLY))
med_v_clg = float(np.median(V_COS_LG))
med_v_frad = float(np.median(V_FRAC_RAD))
med_v_tfrac = float(np.median(V_TAN_FRAC))
med_ka = float(np.median(KA_COS_D))
log('S3 V arm: med cos_d=%.4f cos_dtan=%.4f '
    'cos_dtan_only=%.4f cos_lg=%.4f '
    'frac_rad=%.4f tan_frac=%.4f'
    % (med_v_d, med_v_dtan, med_v_dtonly,
       med_v_clg, med_v_frad, med_v_tfrac))
log('S3 K_adv arm (head %d): med cos_d=%.4f '
    'cos_dtan_only=%.4f'
    % (h_adv, med_ka,
       float(np.median(KA_COS_DTAN_ONLY))))
v_adv_tan = bool(med_v_dtonly < 0.0)
s3_pipe = bool(v_adv_tan and gamma_skew >= 2.0
               and sv_share0 >= 0.2)
log('S3: v_adv_tan=%s gamma_skew=%.2f '
    'sv_share0=%.4f pass=%s'
    % (v_adv_tan, gamma_skew, sv_share0,
       s3_pipe))

# ---------- S4: write-side ANOVA ----------
d_bar = D_V.mean(axis=0)
B_b = np.stack([D_V[bidx == b].mean(axis=0)
                for b in range(8)])
C_c = np.stack([D_V[cidx == c].mean(axis=0)
                for c in (1, 2, 3)])
R_kb = D_V - d_bar - (B_b[bidx] - d_bar) \
    - (C_c[cidx - 1] - d_bar)
ss_body = float((np.linalg.norm(
    B_b[bidx] - d_bar, axis=1) ** 2).sum())
ss_pref = float((np.linalg.norm(
    C_c[cidx - 1] - d_bar, axis=1) ** 2).sum())
ss_resid = float((np.linalg.norm(
    R_kb, axis=1) ** 2).sum())
ss_tot = ss_body + ss_pref + ss_resid
sh_body = ss_body / ss_tot
sh_pref = ss_pref / ss_tot
sh_resid = ss_resid / ss_tot


def d_eff_rows(M):
    sv = np.linalg.svd(M,
                       compute_uv=False)
    s2 = sv ** 2
    tot = float(s2.sum())
    return float(s2.sum()) ** 2 \
        / float((s2 ** 2).sum()) \
        if tot > 1e-24 else 0.0


d_eff_body = [d_eff_rows(D_V[bidx == b])
              for b in range(8)]
d_eff_pref = [d_eff_rows(D_V[cidx == c])
              for c in (1, 2, 3)]
U_b = B_b / np.maximum(np.linalg.norm(
    B_b, axis=1, keepdims=True), 1e-12)
cos_bb = np.abs(U_b @ U_b.T)
iu = ~np.eye(8, dtype=bool)
bb_med = float(np.median(cos_bb[iu]))
s4_a = bool(sh_body >= 0.5)
s4_b = bool(np.mean(d_eff_body)
            < np.mean(d_eff_pref))
s4_pass = bool(s4_a and s4_b)
log('S4: SS body=%.4f prefix=%.4f resid=%.4f '
    'd_eff_body=%s d_eff_pref=%s '
    'between-body |cos| med=%.4f pass=%s'
    % (sh_body, sh_pref, sh_resid,
       np.round(d_eff_body, 2).tolist(),
       np.round(d_eff_pref, 2).tolist(),
       bb_med, s4_pass))

# ---------- S5: same-source stats ----------
def project(D):
    Y = np.empty((D.shape[0], NVOC))
    for lo in range(0, D.shape[0], 8):
        Y[lo:lo + 8] = D[lo:lo + 8] @ wud.T
    return Y


Y_V = project(D_V)
Y_KA = project(D_KA)
Y_KP = project(D_KP)

# axis null in the readout metric
Gmet = wud.T @ wud  # (HID, HID) fp64
del wud  # free the 4.4 GB fp64 copy (last use)
rng_ax = np.random.default_rng(9999)
Xa = rng_ax.standard_normal((N_PERM, HID))
Yb_ = rng_ax.standard_normal((N_PERM, HID))
GXa = Xa @ Gmet
GYb = Yb_ @ Gmet
_num = np.abs((GXa * Yb_).sum(axis=1))
_den = np.sqrt(np.maximum(
    (GXa * Xa).sum(axis=1), 1e-30)
    * np.maximum((GYb * Yb_).sum(axis=1),
                 1e-30))
ax_null = _num / _den
del Gmet, Xa, Yb_, GXa, GYb
ax_null_sorted = np.sort(ax_null)


def axis_p(c):
    idx = int(np.searchsorted(
        ax_null_sorted, c, side='right'))
    return float(N_PERM - idx + 1) \
        / (N_PERM + 1.0)


def unit_rows(Y):
    n = np.linalg.norm(Y, axis=1,
                       keepdims=True)
    n[n < 1e-12] = 1.0
    return Y / n


def gram_abs_cos(Ya, Yb):
    A = unit_rows(Ya)
    B = unit_rows(Yb)
    return np.abs(A @ B.T)


def diag_stat(C):
    dd = float(np.mean(np.diag(C)))
    off = float(np.mean(C[~np.eye(
        C.shape[0], dtype=bool)]))
    return dd, off, dd - off


def perm_diag_p(C, seed):
    n = C.shape[0]
    dd, off, obs = diag_stat(C)
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(N_PERM):
        p = rng.permutation(n)
        dp = float(np.mean(np.diag(C[:, p])))
        op = float(np.mean(C[:, p][~np.eye(
            n, dtype=bool)]))
        if dp - op >= obs:
            cnt += 1
    return obs, (float(cnt) + 1.0) \
        / (N_PERM + 1.0), dd, off


def pc1_of(Y):
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
    return c_mean, abs(float(pa @ pb))


C_VKA = gram_abs_cos(Y_V, Y_KA)
ss1_obs, p_ss1, dg_a, of_a = perm_diag_p(
    C_VKA, 9998)
c_mean_ka, c_pc1_ka = axis_cos(Y_V, Y_KA)
p_ax_mean = axis_p(c_mean_ka)
p_ss2 = axis_p(c_pc1_ka)
C_VKP = gram_abs_cos(Y_V, Y_KP)
ss1_obs_p, p_ss1_p, dg_p, of_p = perm_diag_p(
    C_VKP, 10000)
c_mean_p, c_pc1_p = axis_cos(Y_V, Y_KP)
axis_dom = bool(c_pc1_ka >= c_pc1_p)
log('SS1 V<->K_adv: diag=%.4f off=%.4f '
    'obs=%+.4f p=%.4f'
    % (dg_a, of_a, ss1_obs, p_ss1))
log('SS2 V<->K_adv: |cos mean|=%.4f (p=%.4f) '
    '|cos PC1|=%.4f (p=%.4f)'
    % (c_mean_ka, p_ax_mean, c_pc1_ka, p_ss2))
log('T4: V<->K_pos diag obs=%+.4f p=%.4f '
    '|cos PC1|=%.4f; axis_dominance=%s'
    % (ss1_obs_p, p_ss1_p, c_pc1_p, axis_dom))

P_V = np.sqrt((D_V ** 2).mean(axis=0))
P_KA = np.sqrt((D_KA ** 2).mean(axis=0))
set_V = set(np.argsort(P_V)[::-1][:256]
            .tolist())
set_KA = set(np.argsort(P_KA)[::-1][:256]
             .tolist())
jacc_obs = float(len(set_V & set_KA)) \
    / float(len(set_V | set_KA))
rng_j = np.random.default_rng(10002)
jacc_null = np.zeros(N_PERM)
for i in range(N_PERM):
    idx = rng_j.choice(HID, 512,
                       replace=False)
    s1_ = set(idx[:256].tolist())
    s2_ = set(idx[256:].tolist())
    jacc_null[i] = len(s1_ & s2_) \
        / len(s1_ | s2_)
p_ss3 = float((jacc_null >= jacc_obs).sum()
              + 1) / (N_PERM + 1)
ss_count = int(sum(
    1 for p in (p_ss1, p_ss2, p_ss3)
    if p <= 0.005))
log('SS3: Jaccard=%.4f (null med=%.4f, '
    'p=%.4f); ss_count=%d'
    % (jacc_obs, float(np.median(jacc_null)),
       p_ss3, ss_count))

# channel identity of the two adversarial arms
def overlap_null(seed, ref_set, obs):
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(N_PERM):
        ps = np.argsort(rng.random(HID))[:256]
        if len(set(ps.tolist())
               & set(ref_set.tolist())) >= obs:
            cnt += 1
    return (float(cnt) + 1.0) \
        / (N_PERM + 1.0)


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
        idx = rng.choice(HID, k,
                         replace=False)
        if float(sq[idx].sum()) / tot >= e_obs:
            cnt += 1
    return (float(cnt) + 1.0) \
        / (N_PERM + 1.0)


top256_V = np.argsort(P_V)[::-1][:256]
top256_KA = np.argsort(P_KA)[::-1][:256]
res_ch = {}
for tag, prof, tset, s_base in (
        ('V', P_V, top256_V, 10003),
        ('KA', P_KA, top256_KA, 10009)):
    ov_s16 = int(len(set(tset.tolist())
                     & set(S16.tolist())))
    ov_t64 = int(len(set(tset.tolist())
                     & set(TOP64.tolist())))
    p_s16 = overlap_null(s_base, S16, ov_s16)
    p_t64 = overlap_null(s_base + 1, TOP64,
                         ov_t64)
    e16 = energy_obs(prof, S16)
    e64 = energy_obs(prof, TOP64)
    p_e16 = energy_perm_p(s_base + 2, prof,
                          16, e16)
    p_e64 = energy_perm_p(s_base + 3, prof,
                          64, e64)
    res_ch[tag] = dict(
        ov_s16=ov_s16, p_s16=p_s16,
        ov_t64=ov_t64, p_t64=p_t64,
        e16=e16, p_e16=p_e16,
        e64=e64, p_e64=p_e64)
    log('S5 %s: S16=%d (p=%.4f) TOP64=%d '
        '(p=%.4f) E16=%.4f (p=%.4f) '
        'E64=%.4f (p=%.4f)'
        % (tag, ov_s16, p_s16, ov_t64, p_t64,
           e16, p_e16, e64, p_e64))

# T6 known-axis relations
nv_rows = np.linalg.norm(D_V, axis=1)
nv_rows[nv_rows < 1e-12] = 1.0
nvm = np.linalg.norm(Vmat, axis=1)
cos_dv_vmat = np.array([
    float(D_V[k] @ Vmat[k])
    / (nv_rows[k] * nvm[k])
    for k in range(NP_)])
npc = float(np.linalg.norm(PC1_RO))
cos_dv_pc1 = np.array([
    float(D_V[k] @ PC1_RO)
    / (nv_rows[k] * npc)
    for k in range(NP_)])
mv_ch = D_V.mean(axis=0)
nmc = float(np.linalg.norm(mv_ch))
cos_mV_pc1ro = float(mv_ch @ PC1_RO) / (
    nmc * npc) if nmc > 1e-12 else 0.0
rng_kx = np.random.default_rng(10015)
D_Vn = D_V / nv_rows[:, None]
kx_null = np.zeros(N_PERM)
_kx_chunk = 100
for lo in range(0, N_PERM, _kx_chunk):
    Uc = rng_kx.standard_normal(
        (_kx_chunk, HID))
    Uc /= np.linalg.norm(Uc, axis=1,
                         keepdims=True)
    Ck = np.abs(D_Vn @ Uc.T)
    kx_null[lo:lo + _kx_chunk] = np.median(
        Ck, axis=0)
kx_sorted = np.sort(kx_null)
_kx_obs = float(np.median(
    np.abs(cos_dv_vmat)))
_kx_idx = int(np.searchsorted(
    kx_sorted, _kx_obs, side='right'))
p_kx = float(N_PERM - _kx_idx + 1) \
    / (N_PERM + 1.0)
log('T6: med cos(D_V,Vmat)=%.4f (perm-null '
    'p=%.4f) | med |cos(D_V,PC1_ro)|=%.4f | '
    'cos(mean D_V, PC1_ro)=%.4f'
    % (float(np.median(cos_dv_vmat)), p_kx,
       float(np.median(np.abs(cos_dv_pc1))),
       cos_mV_pc1ro))

# ---------- verdict ----------
setup_ok = bool(b0_ok and b1_ok
                and a233_pre == 0.0 and b3_ok)
s5_pass = bool(ss_count >= 2 and axis_dom
               and med_ka < 0.0)
stage_pass = [bool(s1_pass), bool(s2_pass),
              bool(s3_pipe), bool(s4_pass),
              s5_pass]
if not setup_ok:
    verdict = 'setup_failed_ds7b'
else:
    cnt_pass = int(sum(stage_pass))
    if cnt_pass == 5:
        verdict = 'chain_replicated_ds7b'
    elif cnt_pass >= 3:
        verdict = 'chain_partial_ds7b'
    elif cnt_pass >= 1:
        verdict = 'chain_fragmented_ds7b'
    else:
        verdict = 'chain_absent_ds7b'
log('VERDICT: %s (stages S1-S5 = %s, '
    'setup_ok=%s)'
    % (verdict, stage_pass, setup_ok))

elapsed = time.time() - t0
npz_path = os.path.join(OUT, NAME + '.npz')
np.savez(npz_path,
         COS_LAD_V=COS_LAD_V,
         COS_LAD_K=COS_LAD_K,
         COS_V27=COS_V27, COS_KF=COS_KF,
         COS_JF=COS_JF, COS_JH=COS_JH,
         D_V=D_V, D_KA=D_KA, D_KP=D_KP,
         Vmat=Vmat, PC1_RO=PC1_RO,
         TT=TT, GAMMA=GAMMA,
         TOP64=TOP64.astype(np.int64),
         FLAT64=FLAT64.astype(np.int64),
         S16=S16.astype(np.int64),
         V_COS_D=V_COS_D,
         V_COS_DTAN=V_COS_DTAN,
         V_COS_DTAN_ONLY=V_COS_DTAN_ONLY,
         V_FRAC_RAD=V_FRAC_RAD,
         V_TAN_FRAC=V_TAN_FRAC,
         V_COS_LG=V_COS_LG,
         KA_COS_D=KA_COS_D,
         KA_COS_DTAN_ONLY=KA_COS_DTAN_ONLY,
         C_VKA=C_VKA, C_VKP=C_VKP,
         SS1_OBS=np.float64(ss1_obs),
         SS1_P=np.float64(p_ss1),
         SS1_DIAG=np.float64(dg_a),
         SS1_OFF=np.float64(of_a),
         C_MEAN_KA=np.float64(c_mean_ka),
         P_AX_MEAN=np.float64(p_ax_mean),
         C_PC1_KA=np.float64(c_pc1_ka),
         SS2_P=np.float64(p_ss2),
         SS1_P_KP=np.float64(p_ss1_p),
         C_PC1_KP=np.float64(c_pc1_p),
         AXIS_DOM=np.bool_(axis_dom),
         JACC_OBS=np.float64(jacc_obs),
         JACC_NULL=jacc_null,
         SS3_P=np.float64(p_ss3),
         SS_COUNT=np.int64(ss_count),
         P_V=P_V, P_KA=P_KA,
         TOP256_V=top256_V.astype(np.int64),
         TOP256_KA=top256_KA.astype(np.int64),
         COS_DV_VMAT=cos_dv_vmat,
         COS_DV_PC1=cos_dv_pc1,
         COS_MV_PC1RO=np.float64(
             cos_mV_pc1ro),
         KX_NULL=kx_null,
         P_KX=np.float64(p_kx),
         MEDV_LAD=medV_lad, MEDK_LAD=medK_lad,
         MEDJH=medJh,
         SS_BODY=np.float64(sh_body),
         SS_PREFIX=np.float64(sh_pref),
         SS_RESID=np.float64(sh_resid),
         D_EFF_BODY=np.array(d_eff_body),
         D_EFF_PREF=np.array(d_eff_pref),
         BB_MED=np.float64(bb_med),
         B_BODY=B_b, C_PREFIX=C_c,
         CH_V_S16=np.int64(
             res_ch['V']['ov_s16']),
         CH_V_PS16=np.float64(
             res_ch['V']['p_s16']),
         CH_V_T64=np.int64(
             res_ch['V']['ov_t64']),
         CH_V_PT64=np.float64(
             res_ch['V']['p_t64']),
         CH_V_E16=np.float64(
             res_ch['V']['e16']),
         CH_V_PE16=np.float64(
             res_ch['V']['p_e16']),
         CH_V_E64=np.float64(
             res_ch['V']['e64']),
         CH_V_PE64=np.float64(
             res_ch['V']['p_e64']),
         CH_KA_S16=np.int64(
             res_ch['KA']['ov_s16']),
         CH_KA_PS16=np.float64(
             res_ch['KA']['p_s16']),
         CH_KA_T64=np.int64(
             res_ch['KA']['ov_t64']),
         CH_KA_PT64=np.float64(
             res_ch['KA']['p_t64']),
         CH_KA_E16=np.float64(
             res_ch['KA']['e16']),
         CH_KA_PE16=np.float64(
             res_ch['KA']['p_e16']),
         CH_KA_E64=np.float64(
             res_ch['KA']['e64']),
         CH_KA_PE64=np.float64(
             res_ch['KA']['p_e64']),
         AX_NULL=ax_null,
         B0_DIFF=np.float64(b0_diff),
         B1_DIFF=np.float64(b1_diff),
         B5_REL=np.float64(b5_rel),
         A233_PRE=np.float64(a233_pre),
         GAMMA_SKEW=np.float64(gamma_skew),
         SV_SHARE0=np.float64(sv_share0),
         STAGE_PASS=np.array(stage_pass),
         V_ADV_TAN=np.bool_(v_adv_tan),
         SETUP_OK=np.bool_(setup_ok),
         verdict=np.array(verdict),
         elapsed=np.float64(elapsed))

stats = {
    'S1_ladder': {
        'medK_27': float(medK_lad[L_TGT]),
        'medV_27': float(medV_lad[L_TGT]),
        'medK_late_mean':
            float(medK_lad[late].mean()),
        'medK_early_mean':
            float(medK_lad[~late].mean()),
        'pass': s1_pass},
    'S2_gate': {
        'medV': medV, 'medK': medK,
        'medJ': medJ, 'k_share':
            float(k_share),
        'medJh': [float(v) for v in medJh],
        'h_adv': int(h_adv),
        'h_pos': int(h_pos),
        'pass': s2_pass},
    'S3_pipe': {
        'med_cos_d': med_v_d,
        'med_cos_dtan': med_v_dtan,
        'med_cos_dtan_only': med_v_dtonly,
        'med_cos_lg_pred': med_v_clg,
        'med_frac_rad': med_v_frad,
        'med_tan_frac': med_v_tfrac,
        'med_ka_cos_d': med_ka,
        'gamma_skew': gamma_skew,
        'sv_share0': sv_share0,
        'v_adv_tan': v_adv_tan,
        'pass': s3_pipe},
    'S4_write': {
        'share_body': sh_body,
        'share_prefix': sh_pref,
        'share_resid': sh_resid,
        'd_eff_body': [float(v) for v in
                       d_eff_body],
        'd_eff_pref': [float(v) for v in
                       d_eff_pref],
        'between_body_cos_med': bb_med,
        'pass': s4_pass},
    'S5_source': {
        'ss1_diag': dg_a, 'ss1_off': of_a,
        'ss1_obs': ss1_obs, 'ss1_p': p_ss1,
        'c_mean_ka': c_mean_ka,
        'p_ax_mean': p_ax_mean,
        'c_pc1_ka': c_pc1_ka,
        'ss2_p': p_ss2,
        'ss1_p_kpos': p_ss1_p,
        'c_pc1_kpos': c_pc1_p,
        'axis_dominance': axis_dom,
        'jaccard_obs': jacc_obs,
        'jaccard_null_med':
            float(np.median(jacc_null)),
        'ss3_p': p_ss3, 'ss_count': ss_count,
        'med_ka': med_ka,
        'channel_V': {kk: float(vv) for kk,
                      vv in
                      res_ch['V'].items()},
        'channel_KA': {kk: float(vv) for kk,
                       vv in
                       res_ch['KA'].items()},
        'med_cos_dv_vmat':
            float(np.median(cos_dv_vmat)),
        'p_kx': p_kx,
        'med_abs_cos_dv_pc1':
            float(np.median(
                np.abs(cos_dv_pc1))),
        'pass': s5_pass},
    'anchors': {
        'b0_recapture_diff': b0_diff,
        'b1_sham_diff': b1_diff,
        'b5_pastkey_rel': b5_rel,
        'a233_pre_max': a233_pre,
        'setup_ok': setup_ok},
}
result = {'phase': PHASE, 'name': NAME,
          'created': created, 'elapsed': elapsed,
          'run': 'run1 authoritative (bf16 '
                 'native; fresh DS7B capture '
                 'banks; Qwen2 adaptations: no '
                 'k_norm -> k_proj replacement '
                 'point, 4 kv heads, fp64 manual '
                 'RMSNorm counterfactual)',
          'prereg': PREREG, 'stats': stats,
          'verdict': verdict}
res_path = os.path.join(OUT, 'result.json')
with open(res_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False,
              indent=1)

seal = {
    'phase': PHASE, 'name': NAME,
    'created': created,
    'npz_sha256_8': sha8(npz_path),
    'result_sha256_8': sha8(res_path),
    'exec_sha256_8': sha8(os.path.join(
        OUT, 'execution.json')),
    'script_sha256_8': sha8(
        os.path.abspath(__file__)),
    'verdict': verdict,
    'setup_ok': setup_ok,
}
with open(os.path.join(OUT, 'seal.json'), 'w',
          encoding='utf-8') as f:
    json.dump(seal, f, ensure_ascii=False,
              indent=1)
log('sealed npz8=%s result8=%s exec8=%s '
    'script8=%s elapsed=%.1fs'
    % (seal['npz_sha256_8'],
       seal['result_sha256_8'],
       seal['exec_sha256_8'],
       seal['script_sha256_8'], elapsed))
log('sealed')
