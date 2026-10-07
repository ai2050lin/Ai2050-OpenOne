"""
Phase 3078 A (menu of 3077): omega_p75_routing_timing.

Question: at which layer does the routing signal (the
head-by-family pattern of causal contribution R1, shown
by 3076/3077 to be head x family specific with
ICC_head(R1)=0.45 while 12 static geometry features all
failed to predict it) become readable from the BASE
(prompt-conditioned) representation?

Constraint that motivates the design: the 3076 injection
is a V-replacement inside L34, so every layer before L34
is bit-identical between injected and base forwards.
Any routing signal that exists prior to the injection
must therefore live in the base representation itself.

Method (no injections, base forwards only):
  For each family f in {A, B, C} (A everyday-causal /
  3074 bit-anchor family, B science-causal, C social-
  affective, same 32-prompt frame: 8 bodies x 4 prefix
  conditions), run base forwards and capture the last-
  position attention o_proj INPUT zH_l (NQW=4096, the
  per-head concat) at layers
  LYRS=(24,26,28,30,31,32,33,34,35).
  For each pair k (pref_i, base_i) and head h:
    dAh_l[k,h] = ((zh_l[h] @ WoT_h) @ W32 @ TT64[k])
                 / ||TT64[k]||
  where WoT = o_proj.W^T reshaped (32,128,2560) is the
  per-head W_O block and W32 is the fp32 output
  embedding (3071/3076-identical lens math).  TT64 is
  rebuilt float64 from the float64 logits (LG64); the
  3076 npz only stored a float32 TT copy, and the a1
  anchor below needs the exact float64 t used by 3076.
  Family readout spectrum Dm_l[f] = median_k dAh_l.

Anchors (hard, bit):
  a1  per-k DAH34/DAH35 recomputed as
      headproj(ZH34/35_npz[k] - ZHbase[base_k, L34/35])
      vs 3076 npz DAH34/35 rows: max|d| must be 0.0.
      This simultaneously proves (i) 3078 base forwards
      are bit-identical to 3076 base banks at L34/L35
      and (ii) the 3078 headproj math is bit-identical
      to 3076 head_decomp.
  a1m (authoritative only) median DAH34/35 vs npz
      DAH34/35_MED: max|d| = 0.0.
  a2  R1_ALL32_A vs 3071 result.json stats.head.r34:
      max|d| = 0.0.
  a0  provenance (loose gate): max|TT64 - TT32_npz| and
      max|LG64 - LG32_npz| must be float32-rounding
      scale (< 1e-3), proving 3078 TT64/LG64 match the
      3076 run up to the stored float32 rounding.
  b0  in-process recapture of base0 zH: bit 0.0.

Statistics:
  E2 intra-family readability curve:
    sp_intra[l,f] = spearman(|Dm_l[f]|, R1_f),
    two-sided permutation p (n_perm=20000, head
    permutation).
  E3 cross-family spectrum similarity:
    sp_cross[l,(f,g)] = spearman(|Dm_l[f]|, |Dm_l[g]|).
  E4 head x family interaction per layer:
    ICC_head of anova2 on M_l = [|Dm_l[f]|]_(3,32)
    (signed-D variant recorded as icc_sig).
  E6 amplitude control:
    sp_amp[l,f] = spearman(median_k ||outh||, R1_f) --
    separates "aligned with family TT direction" from
    "writes a lot at all".
  E7 base-vs-delta relationship at L34/L35:
    sp_d34b[f] = spearman(|Dm_34[f]|,
    |DAH34_MED_npz[f]|).

Verdict gates (preregistered):
  S = {l in LYRS : min_f |sp_intra[l,f]| >= 0.35 AND
                    max_f pp_intra[l,f] < 0.05}
  early_routing        : S contains some l <= 32
  routing_late_emergent: S non-empty and all l >= 33
  routing_signal_absent: S empty AND global max
                         |sp_intra| < 0.35
  routing_partial      : otherwise.

limitations (recorded): n_heads=32 so spearman power is
limited (|sp|>=0.35 ~ p~0.05 at n=32); the 0.05
permutation gate is not multiplicity-corrected across
9x3 tests (the min-over-families requirement is the
main multiplicity guard); causal-connective paradigm
only, shared syntactic frame; single model qwen3-4b;
the readout is linear (head write projected on the
family TT direction) - a nonlinear routing signal would
be invisible to it.

memory discipline: single model; W32 fp32 resident
(~246 MB); Wo/WoT loaded per layer then deleted; LG64/
TT64 per family freed at family end; del + gc +
empty_cache at family end and run end.
"""
import gc
import hashlib
import io
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, \
    AutoModelForCausalLM

PHASE = 3078
NAME = 'omega_p75_routing_timing'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(
    ROOT, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913',
    'phase3078', NAME)
SMOKE = os.environ.get('SMOKE', '0') == '1'
if SMOKE:
    OUT = os.path.join(OUT, 'smoke')
SCRIPT = os.path.join(
    ROOT, 'tests', 'glm5',
    'phase3078_omega_p75_routing_timing.py')

MDIR = os.path.join(ROOT, 'models', 'hf',
                    'qwen3-4b')
NL, HID, KV_HEAD, NQ = 36, 2560, 8, 32
HDIM = 128
NQW = NQ * HDIM
L34 = 34
L35 = 35
LYRS = (24, 26, 28, 30, 31, 32, 33, 34, 35)
NLYR = len(LYRS)
SEED = 3078
N_PERM = 20000
GATE_SP = 0.35
GATE_P = 0.05
FKEYS = ('A', 'B', 'C')
TARGETS = ('so', 'because', 'therefore',
           'however', 'while', 'yet',
           'although', 'thus')
FAM = {
    'A': {
        'domain': 'everyday-causal (3074 texts '
                  'rerun, bit-anchor family)',
        'bodies': (
            'The weather was cold, so',
            'He studied every night because',
            'The experiment failed, therefore',
            'He missed the train, however',
            'The garden grows quickly while',
            'The price was high, yet',
            'She speaks French, although',
            'The road was closed, thus',),
        'prefixes': ('', 'In a formal style,',
                     'In Shakespearean style,',
                     'Regarding the weather,'),
    },
    'B': {
        'domain': 'science-causal',
        'bodies': (
            'The solution turned acidic, so',
            'The sample was heated because',
            'The catalyst degraded, therefore',
            'The vacuum leaked, however',
            'The crystals formed while',
            'The pressure dropped, yet',
            'The alloy expanded, although',
            'The circuit overheated, thus',),
        'prefixes': ('', 'In a formal style,',
                     'In Shakespearean style,',
                     'Regarding the '
                     'experiment,'),
    },
    'C': {
        'domain': 'social-emotional-causal',
        'bodies': (
            'She felt deeply betrayed, so',
            'He apologized to her because',
            'They reconciled after the '
            'quarrel, therefore',
            'She stormed out of the room, '
            'however',
            'He listened quietly to every '
            'word while',
            'The gift was cheap and hasty, '
            'yet',
            'She forgave him in the end, '
            'although',
            'The friendship ended without '
            'warning, thus',),
        'prefixes': ('', 'In a formal style,',
                     'In Shakespearean style,',
                     'Regarding the '
                     'conversation,'),
    },
}
RDIR = os.path.join(
    ROOT, 'tests', 'glm5', 'result',
    'rdc_query_construction_20260913')
P76 = os.path.join(
    RDIR, 'phase3076',
    'omega_p73_cross_prompt_family')
P71 = os.path.join(
    RDIR, 'phase3071',
    'omega_p68_attn_head_decomp')

PREREG = {
    'mode': 'single model qwen3-4b bf16 eager; '
            'base forwards only (no injection); '
            'frozen inputs from 3076 npz (ZH34/'
            'ZH35/DAH34/DAH35/DAH34_MED/DAH35_MED/'
            'R1_ALL32/TT32/LG32) and 3071 result '
            'r34; float64 TT64/LG64 rebuilt from '
            'fresh base forwards for bit anchors; '
            'smoke mode optional (SMOKE=1: 8 '
            'pairs k=0-7, forwards 8 bases + 8 '
            'ci=1 prefs per family, a1 on k=0-7 '
            'rows, a1m off, verdict smoke_pending)',
    'question': '3078 A (menu of 3077): the '
                'routing signal (head x family '
                'causal contribution pattern R1) '
                'is not predictable from 12 static '
                'geometry features (3077 E4) and '
                'is head-family specific (ICC 0.45 '
                'vs 0.87).  At which layer does a '
                'linear readout of the base '
                'representation (per-head o_proj '
                'input projected on the family TT '
                'direction) become informative '
                'about R1?  Early (L<=32) readability '
                'would mean routing lives in the '
                'base representation before the '
                'injection; late-only readability '
                '(L>=33) would mean it emerges with '
                'the injection; absence everywhere '
                'extends the 3077 observation-'
                'causation decoupling to the whole '
                'time course.',
    'definitions': {
        'zh_l': 'last-position attention o_proj '
                'INPUT (per-head concat, NQW=4096) '
                'at layer l, base forward',
        'dAh_l[k,h]': '((zh_l[h] @ WoT_h) @ W32 @ '
                       'TT64[k]) / ||TT64[k]||, '
                       'fp32 lens math identical to '
                       '3076 head_decomp',
        'Dm_l[f]': 'median_k dAh_l (signed, (32,))',
        'Om_l[f]': 'median_k ||outh|| head write '
                   'amplitude control',
        'R1_f': '3076 npz R1_ALL32_f (32,), the '
                'per-head causal readout',
        'TT64[k]': 'LG64[pref_k] - LG64[base_k] in '
                   'float64 (3076 computed E2H with '
                   'this exact float64 object)',
    },
    'anchors': {
        'a1': 'per-k DAH34/DAH35 recomputed vs '
              '3076 npz rows: max|d| = 0.0 (hard)',
        'a1m': 'median DAH34/35 vs 3076 npz '
               'DAH34/35_MED: max|d| = 0.0 '
               '(authoritative only)',
        'a2': 'R1_ALL32_A vs 3071 r34: '
              'max|d| = 0.0 (hard)',
        'a0': 'max|TT64-TT32_npz| < 1e-3 and '
              'max|LG64-LG32_npz| < 1e-3 '
              '(float32-rounding provenance gate)',
        'b0': 'in-process base0 zH recapture '
              'bit = 0.0',
    },
    'gates': {
        'S': 'layers with min_f |sp_intra| >= 0.35 '
             'AND max_f perm_p < 0.05',
        'early_routing': 'some l in S with l <= 32',
        'routing_late_emergent': 'S non-empty and '
                                 'all l >= 33',
        'routing_signal_absent': 'S empty and '
                                 'global max |sp| < '
                                 '0.35',
        'routing_partial': 'otherwise',
    },
    'statistics_discipline': 'spearman via stable '
        'argsort ranks; two-sided permutation p '
        '(20000, seed 3078, vectorized row '
        'permutations); ICC via two-way no-rep '
        'anova2 (head effect = column mean); '
        'verdict scalars fp64; no multiplicity '
        'correction beyond the min-over-families '
        'requirement (recorded as limitation)',
    'limitations': 'n_heads=32 limits spearman '
        'power; linear readout only (TT-direction '
        'projection); causal-connective paradigm '
        'only, shared syntactic frame; single '
        'model; |sp|>=0.35 corresponds to p~0.05 '
        'at n=32',
    'memory_discipline': 'W32 fp32 resident; Wo '
        'per layer then deleted; LG64/TT64 freed '
        'per family; del+gc+empty_cache at family '
        'end and run end',
}

os.makedirs(OUT, exist_ok=True)
LOGF = os.path.join(OUT, 'run_log.txt')
_o = []


def log(msg):
    _o.append(str(msg))


t0 = time.time()

# ---------- execution.json (prereg freeze) ----------
exep = os.path.join(OUT, 'execution.json')
resp = os.path.join(OUT, 'result.json')
for p_ in (exep, resp):
    if os.path.exists(p_):
        raise SystemExit(
            'stale %s exists; delete it before '
            'rerun (prereg discipline)' % p_)
exe = {'phase': PHASE, 'name': NAME,
       'created': time.strftime(
           '%Y-%m-%d %H:%M:%S'),
       'smoke': SMOKE, 'prereg': PREREG}
with io.open(exep, 'w', encoding='utf-8') as f:
    json.dump(exe, f, ensure_ascii=False, indent=1)
log('execution.json written (prereg frozen) %s '
    'smoke=%s' % (exe['created'], SMOKE))

# ---------- load model ----------
tok = AutoTokenizer.from_pretrained(MDIR)
model = AutoModelForCausalLM.from_pretrained(
    MDIR, torch_dtype=torch.bfloat16,
    attn_implementation='eager').to('cuda').eval()
layers = model.model.layers
assert len(layers) == NL
assert int(model.config.num_attention_heads) == NQ
assert int(model.config.hidden_size) == HID
NVOC = int(model.config.vocab_size)
W32 = model.get_output_embeddings() \
    .weight.detach().float()
assert int(W32.shape[0]) == NVOC
log('model loaded NVOC=%d W32%s'
    % (NVOC, tuple(W32.shape)))

# ---------- capture hooks (o_proj input only) ----
capH = {li: {'rec': False, 'v': None}
        for li in range(NL)}


def hook_in_last(cp):
    def h(module, inp, out):
        if cp['rec']:
            cp['v'] = inp[0][0, -1] \
                .detach().clone()
        return out
    return h


for li in range(NL):
    layers[li].self_attn.o_proj \
        .register_forward_hook(hook_in_last(
            capH[li]))


def reset_capH():
    for li in range(NL):
        capH[li]['rec'] = False
        capH[li]['v'] = None


def forward_base(ids):
    """3076/3071-identical base forward (no
    injection).  Returns float64 last-position
    logits and zH at LYRS layers (fp64 numpy)."""
    reset_capH()
    for li in LYRS:
        capH[li]['rec'] = True
    with torch.no_grad():
        out = model(torch.tensor(
            [ids], device='cuda'),
            use_cache=False)
    lg = out.logits[0, -1].detach() \
        .double().cpu().numpy()
    zhs = {}
    for li in LYRS:
        zhs[li] = capH[li]['v'].double() \
            .cpu().numpy()
    reset_capH()
    return lg, zhs


# ---------- frozen data ----------
z76 = np.load(os.path.join(
    P76, 'omega_p73_cross_prompt_family.npz'))
ZH34_NPZ = {f: z76['ZH34_' + f]
            .astype(np.float64)
            for f in FKEYS}
ZH35_NPZ = {f: z76['ZH35_' + f]
            .astype(np.float64)
            for f in FKEYS}
DAH34_NPZ = {f: z76['DAH34_' + f]
             .astype(np.float64)
             for f in FKEYS}
DAH35_NPZ = {f: z76['DAH35_' + f]
             .astype(np.float64)
             for f in FKEYS}
DAH34M_NPZ = {f: z76['DAH34_MED_' + f]
              .astype(np.float64)
              for f in FKEYS}
DAH35M_NPZ = {f: z76['DAH35_MED_' + f]
              .astype(np.float64)
              for f in FKEYS}
R1 = {f: z76['R1_ALL32_' + f]
      .astype(np.float64) for f in FKEYS}
TT32_NPZ = {f: z76['TT_' + f]
            .astype(np.float32) for f in FKEYS}
LG32_NPZ = {f: z76['LG_' + f]
            .astype(np.float32) for f in FKEYS}
j71 = json.load(io.open(
    os.path.join(P71, 'result.json'),
    encoding='utf-8'))
R34_71 = np.array(
    j71['stats']['head']['r34'],
    dtype=np.float64)

a2_diff = float(np.max(np.abs(
    R1['A'] - R34_71)))
a2_ok = bool(a2_diff == 0.0)
log('a2 R1_ALL32_A vs 3071 r34: max|d|=%.3e '
    'ok=%s' % (a2_diff, a2_ok))

# ---------- statistics helpers (3077-identical) --


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ra = np.argsort(np.argsort(a)) \
        .astype(np.float64)
    rb = np.argsort(np.argsort(b)) \
        .astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum()
                  * (rb * rb).sum())
    if den == 0:
        return 0.0
    return float((ra * rb).sum() / den)


def perm_p(a, b, n_perm=N_PERM, seed=SEED):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    obs = abs(spearman(a, b))
    if n_perm <= 0:
        return 1.0
    ra = np.argsort(np.argsort(a)) \
        .astype(np.float64)
    ra -= ra.mean()
    B = np.tile(b, (n_perm, 1))
    B = rng.permuted(B, axis=1)
    rb = np.argsort(np.argsort(B, axis=1),
                    axis=1).astype(np.float64)
    rb -= rb.mean(axis=1, keepdims=True)
    num = (rb * ra[None, :]).sum(axis=1)
    den = np.sqrt(
        (rb * rb).sum(axis=1)
        * float((ra * ra).sum()))
    den[den == 0] = 1.0
    stats = np.abs(num / den)
    return float((stats >= obs - 1e-12)
                 .mean())


def anova2(M):
    M = np.asarray(M, dtype=np.float64)
    gm = M.mean()
    alpha = M.mean(axis=0) - gm
    beta = M.mean(axis=1) - gm
    eps = M - gm - alpha[None, :] \
        - beta[:, None]
    nf, nh = M.shape
    v_head = float((alpha ** 2).sum()
                   / (nh - 1))
    v_fam = float((beta ** 2).sum()
                  / (nf - 1))
    v_eps = float((eps ** 2).sum()
                  / ((nf - 1) * (nh - 1)))
    return {'alpha': alpha, 'beta': beta,
            'v_head': v_head, 'v_fam': v_fam,
            'v_eps': v_eps,
            'icc_head': v_head
            / max(v_head + v_eps, 1e-30)}


# ---------- head projection (3076 bit math) ----


def headproj(dzH, WoT, t, tn):
    dhr = torch.tensor(
        np.ascontiguousarray(
            dzH.reshape(NQ, HDIM)),
        device='cuda').float()
    with torch.no_grad():
        outh = torch.einsum(
            'hd,hdj->hj', dhr, WoT)
        un = F.linear(outh, W32)
    out_np = outh.float().cpu().numpy()
    un_np = un.double().cpu().numpy()
    dAh = (un_np @ t) / tn
    return dAh, np.linalg.norm(
        out_np, axis=1)


FW = [0]

# ---------- per-family run ----------
RES_F = {}
for fk in FKEYS:
    fam = FAM[fk]
    bodies = fam['bodies']
    prefixes = fam['prefixes']
    log('[%s] ==== family begin (%s) ===='
        % (fk, fam['domain']))

    # assembly (3076-identical)
    word_tok = {}
    for w in TARGETS:
        wi = tok(' ' + w,
                 add_special_tokens=False)[
            'input_ids']
        assert len(wi) == 1, (fk, w, wi)
        word_tok[w] = int(wi[0])
    assembled = []
    for bi in range(len(bodies)):
        for ci in range(len(prefixes)):
            s = (prefixes[ci] + ' '
                 + bodies[bi]) if prefixes[ci] \
                else bodies[bi]
            ids = [int(x) for x in tok(
                s, add_special_tokens=False)[
                'input_ids']]
            t_ = word_tok[TARGETS[bi]]
            assert ids.count(t_) == 1, \
                (fk, bi, ci)
            assembled.append(
                {'ids': ids, 'cond': ci,
                 'body': bi})
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
            assert off > 0, (fk, i)
            assert list(pid[off + 1:]) \
                == list(bid[1:]), (fk, i)
    LENS = np.array([len(assembled[i]['ids'])
                     for i in range(n_pr)])
    log('[%s] assembled 32 prompts (lens %d-%d)'
        % (fk, int(LENS.min()), int(LENS.max())))

    cidx = []
    bidx = []
    for ci in (1, 2, 3):
        for bi in range(8):
            cidx.append(ci)
            bidx.append(bi)
    cidx = np.array(cidx)
    bidx = np.array(bidx)
    NP_ = 24
    K_USE = 8 if SMOKE else NP_
    BASE_K = np.array([
        idx_of[(0, int(bidx[k]))]
        for k in range(NP_)])

    # forward set
    fwd_idx = [idx_of[(0, bi)]
               for bi in range(8)]
    if SMOKE:
        for bi in range(8):
            fwd_idx.append(idx_of[(1, bi)])
    else:
        for ci in (1, 2, 3):
            for bi in range(8):
                fwd_idx.append(
                    idx_of[(ci, bi)])
    LG64 = {}
    ZHb = np.zeros((8, NLYR, NQW))
    for i in fwd_idx:
        lg, zhs = forward_base(
            assembled[i]['ids'])
        FW[0] += 1
        LG64[i] = lg
        if assembled[i]['cond'] == 0:
            ZHb[assembled[i]['body']] = \
                np.stack([zhs[li]
                          for li in LYRS])
    log('[%s] base forwards done (n=%d, '
        'forwards=%d)'
        % (fk, len(fwd_idx), FW[0]))

    # b0 in-process recapture
    b0 = idx_of[(0, 0)]
    lg2, zhs2 = forward_base(
        assembled[b0]['ids'])
    FW[0] += 1
    b0_diff = float(np.max(np.abs(
        LG64[b0] - lg2)))
    for li in LYRS:
        b0_diff = max(b0_diff, float(
            np.max(np.abs(
                ZHb[0, LYRS.index(li)]
                - zhs2[li]))))
    b0_ok = bool(b0_diff == 0.0)
    log('[%s] b0 recapture diff=%.3e ok=%s'
        % (fk, b0_diff, b0_ok))

    # TT64 rebuild + a0 provenance
    TT64 = np.stack([
        LG64[idx_of[(int(cidx[k]),
                     int(bidx[k]))]]
        - LG64[idx_of[(0, int(bidx[k]))]]
        for k in range(K_USE)])
    a0_tt = float(np.max(np.abs(
        TT64 - TT32_NPZ[fk][:K_USE]
        .astype(np.float64))))
    a0_lg = max(
        float(np.max(np.abs(
            LG64[i] - LG32_NPZ[fk][i]
            .astype(np.float64))))
        for i in fwd_idx)
    a0_ok = bool(a0_tt < 1e-3 and a0_lg < 1e-3)
    log('[%s] a0 provenance: max|TT64-TT32|='
        '%.3e max|LG64-LG32|=%.3e ok=%s'
        % (fk, a0_tt, a0_lg, a0_ok))

    # per-layer spectra + delta anchors
    D = {li: np.zeros((K_USE, NQ))
         for li in LYRS}
    O = {li: np.zeros((K_USE, NQ))
         for li in LYRS}
    DAH34_RE = None
    DAH35_RE = None
    a1_diff = 0.0
    a1m_diff = None
    for li in LYRS:
        Wo = layers[li].self_attn.o_proj \
            .weight.detach().float()
        WoT = Wo.t().reshape(NQ, HDIM, HID)
        for k in range(K_USE):
            bk = int(BASE_K[k])
            t = TT64[k]
            tn = max(float(np.linalg.norm(t)),
                     1e-12)
            dAh, ohn = headproj(
                ZHb[int(bidx[k]),
                    LYRS.index(li)],
                WoT, t, tn)
            D[li][k] = dAh
            O[li][k] = ohn
        if li in (L34, L35):
            ZN = (ZH34_NPZ[fk] if li == L34
                  else ZH35_NPZ[fk])
            RE = np.zeros((K_USE, NQ))
            for k in range(K_USE):
                bk = int(BASE_K[k])
                t = TT64[k]
                tn = max(
                    float(np.linalg.norm(t)),
                    1e-12)
                RE[k] = headproj(
                    ZN[k] - ZHb[int(bidx[k]),
                                LYRS.index(li)],
                    WoT, t, tn)[0]
            REF = (DAH34_NPZ[fk] if li == L34
                   else DAH35_NPZ[fk])
            d_ = float(np.max(np.abs(
                RE - REF[:K_USE])))
            a1_diff = max(a1_diff, d_)
            if li == L34:
                DAH34_RE = RE
            else:
                DAH35_RE = RE
        del Wo, WoT
        gc.collect()
        torch.cuda.empty_cache()
    a1_ok = bool(a1_diff == 0.0)
    log('[%s] a1 per-k DAH34/35 recompute vs '
        '3076 npz: max|d|=%.3e ok=%s'
        % (fk, a1_diff, a1_ok))
    if not SMOKE:
        m34 = np.median(DAH34_RE, axis=0)
        m35 = np.median(DAH35_RE, axis=0)
        a1m_diff = max(
            float(np.max(np.abs(
                m34 - DAH34M_NPZ[fk]))),
            float(np.max(np.abs(
                m35 - DAH35M_NPZ[fk]))))
        log('[%s] a1m DAH medians vs 3076 npz: '
            'max|d|=%.3e ok=%s'
            % (fk, a1m_diff,
               bool(a1m_diff == 0.0)))

    Dm = {li: np.median(D[li], axis=0)
          for li in LYRS}
    Om = {li: np.median(O[li], axis=0)
          for li in LYRS}
    log('[%s] spectra done: D_L24 top %s | '
        'D_L34 top %s'
        % (fk,
           np.argsort(np.abs(Dm[24]))[::-1][:4]
           .tolist(),
           np.argsort(np.abs(Dm[34]))[::-1][:4]
           .tolist()))

    RES_F[fk] = {
        'D': D, 'O': O, 'Dm': Dm, 'Om': Om,
        'DAH34_RE': DAH34_RE,
        'DAH35_RE': DAH35_RE,
        'a0_tt': a0_tt, 'a0_lg': a0_lg,
        'a0_ok': a0_ok, 'a1_diff': a1_diff,
        'a1_ok': a1_ok,
        'a1m_diff': a1m_diff,
        'b0_diff': b0_diff, 'b0_ok': b0_ok,
        'K_USE': K_USE,
    }
    del LG64, ZHb, TT64
    gc.collect()
    torch.cuda.empty_cache()

# ---------- E2-E7 statistics ----------
sp_intra = np.zeros((NLYR, 3))
pp_intra = np.zeros((NLYR, 3))
sp_amp = np.zeros((NLYR, 3))
sp_cross = np.zeros((NLYR, 3))
icc = np.zeros(NLYR)
icc_sig = np.zeros(NLYR)
CPAIRS = (('A', 'B'), ('A', 'C'), ('B', 'C'))
for li_i, li in enumerate(LYRS):
    for fi, fk in enumerate(FKEYS):
        s_ = spearman(
            np.abs(RES_F[fk]['Dm'][li]),
            R1[fk])
        sp_intra[li_i, fi] = s_
        pp_intra[li_i, fi] = perm_p(
            np.abs(RES_F[fk]['Dm'][li]),
            R1[fk])
        sp_amp[li_i, fi] = spearman(
            RES_F[fk]['Om'][li], R1[fk])
    for pi_, (fa, fb) in enumerate(CPAIRS):
        sp_cross[li_i, pi_] = spearman(
            np.abs(RES_F[fa]['Dm'][li]),
            np.abs(RES_F[fb]['Dm'][li]))
    M_ = np.stack([
        np.abs(RES_F[fk]['Dm'][li])
        for fk in FKEYS])
    icc[li_i] = anova2(M_)['icc_head']
    M2 = np.stack([
        RES_F[fk]['Dm'][li] for fk in FKEYS])
    icc_sig[li_i] = anova2(M2)['icc_head']
sp_d34b = [float(spearman(
    np.abs(RES_F[fk]['Dm'][L34]),
    np.abs(DAH34M_NPZ[fk])))
    for fk in FKEYS]
sp_d35b = [float(spearman(
    np.abs(RES_F[fk]['Dm'][L35]),
    np.abs(DAH35M_NPZ[fk])))
    for fk in FKEYS]
log('E2 sp_intra by layer (rows %s):'
    % (LYRS,))
for li_i, li in enumerate(LYRS):
    log('  L%d intra %s pp %s cross %s '
        'icc=%.3f amp %s'
        % (li,
           ['%.3f' % v for v in
            sp_intra[li_i]],
           ['%.3f' % v for v in
            pp_intra[li_i]],
           ['%.3f' % v for v in
            sp_cross[li_i]],
           icc[li_i],
           ['%.3f' % v for v in
            sp_amp[li_i]]))
log('E7 sp_d34b %s | sp_d35b %s'
    % (['%.3f' % v for v in sp_d34b],
       ['%.3f' % v for v in sp_d35b]))

# ---------- verdict ----------
verdict = 'smoke_pending' if SMOKE else None
gates = {}
if not SMOKE:
    S = [li for li_i, li in enumerate(LYRS)
         if float(np.min(np.abs(
             sp_intra[li_i]))) >= GATE_SP
         and float(np.max(
             pp_intra[li_i])) < GATE_P]
    gmax = float(np.max(np.abs(sp_intra)))
    early = [li for li in S if li <= 32]
    gates = {'S_layers': S,
             'global_max_abs_sp': gmax,
             'early_layers': early}
    if early:
        verdict = 'routing_early_readable'
    elif S and all(li >= 33 for li in S):
        verdict = 'routing_late_emergent'
    elif not S and gmax < GATE_SP:
        verdict = 'routing_signal_absent'
    else:
        verdict = 'routing_partial'
    log('VERDICT: %s (S=%s gmax=%.3f)'
        % (verdict, S, gmax))
else:
    log('VERDICT: smoke_pending')

# ---------- npz ----------
npz_path = os.path.join(OUT, NAME + '.npz')
save = {
    'VERDICT': np.array(verdict),
    'ELAPSED': np.float64(time.time() - t0),
    'SMOKE': np.bool_(SMOKE),
    'FORWARDS': np.int64(FW[0]),
    'LYRS': np.array(LYRS, dtype=np.int64),
    'SP_INTRA': sp_intra,
    'PP_INTRA': pp_intra,
    'SP_AMP': sp_amp,
    'SP_CROSS': sp_cross,
    'ICC': icc,
    'ICC_SIG': icc_sig,
    'SP_D34B': np.array(sp_d34b),
    'SP_D35B': np.array(sp_d35b),
    'GATES_S': np.array(
        gates.get('S_layers', []),
        dtype=np.int64),
    'GATES_GMAX': np.float64(
        gates.get('global_max_abs_sp',
                  np.nan)),
}
for fk in FKEYS:
    R = RES_F[fk]
    for li in LYRS:
        save['D_L%d_%s' % (li, fk)] = R['D'][li]
        save['O_L%d_%s' % (li, fk)] = R['O'][li]
        save['DM_L%d_%s' % (li, fk)] = R['Dm'][li]
        save['OM_L%d_%s' % (li, fk)] = R['Om'][li]
    save['DAH34_RE_' + fk] = R['DAH34_RE']
    save['DAH35_RE_' + fk] = R['DAH35_RE']
    save['A0_TT_' + fk] = np.float64(
        R['a0_tt'])
    save['A0_LG_' + fk] = np.float64(
        R['a0_lg'])
    save['A0_OK_' + fk] = np.bool_(R['a0_ok'])
    save['A1_DIFF_' + fk] = np.float64(
        R['a1_diff'])
    save['A1_OK_' + fk] = np.bool_(R['a1_ok'])
    save['B0_DIFF_' + fk] = np.float64(
        R['b0_diff'])
    save['B0_OK_' + fk] = np.bool_(R['b0_ok'])
    if not SMOKE:
        save['A1M_DIFF_' + fk] = np.float64(
            R['a1m_diff'])
np.savez(npz_path, **save)
log('npz saved %s' % npz_path)

# ---------- result.json ----------
def f64(x):
    x = float(x)
    return x if np.isfinite(x) else None


res = {
    'phase': PHASE,
    'name': NAME,
    'smoke': SMOKE,
    'elapsed_s': f64(time.time() - t0),
    'forwards': int(FW[0]),
    'verdict': verdict,
    'prereg': PREREG,
    'anchors': {
        'a2_diff': a2_diff, 'a2_ok': a2_ok,
        'a1_diff': {fk: f64(
            RES_F[fk]['a1_diff'])
            for fk in FKEYS},
        'a1_ok': {fk: bool(
            RES_F[fk]['a1_ok'])
            for fk in FKEYS},
        'a1m_diff': (None if SMOKE else
                     {fk: f64(
                         RES_F[fk]['a1m_diff'])
                      for fk in FKEYS}),
        'a0_tt': {fk: f64(
            RES_F[fk]['a0_tt'])
            for fk in FKEYS},
        'a0_lg': {fk: f64(
            RES_F[fk]['a0_lg'])
            for fk in FKEYS},
        'a0_ok': {fk: bool(
            RES_F[fk]['a0_ok'])
            for fk in FKEYS},
        'b0_diff': {fk: f64(
            RES_F[fk]['b0_diff'])
            for fk in FKEYS},
        'b0_ok': {fk: bool(
            RES_F[fk]['b0_ok'])
            for fk in FKEYS},
    },
    'stats': {
        'layers': list(LYRS),
        'sp_intra': [[f64(v) for v in row]
                     for row in sp_intra],
        'pp_intra': [[f64(v) for v in row]
                     for row in pp_intra],
        'sp_amp': [[f64(v) for v in row]
                   for row in sp_amp],
        'sp_cross': [[f64(v) for v in row]
                     for row in sp_cross],
        'icc': [f64(v) for v in icc],
        'icc_sig': [f64(v) for v in icc_sig],
        'sp_d34b': [f64(v) for v in sp_d34b],
        'sp_d35b': [f64(v) for v in sp_d35b],
    },
    'gates': gates,
    'note': '3078 A: routing timing.  Base '
            'per-head TT-projection spectra '
            'Dm_l tested against R1 per layer; '
            'a1 bit anchors prove base forwards '
            'and headproj math identical to '
            '3076.',
}
res_path = os.path.join(OUT, 'result.json')
with io.open(res_path, 'w',
             encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False,
              indent=1)
log('result.json written')

# ---------- seal ----------
def fsha8(p):
    return hashlib.sha256(
        io.open(p, 'rb').read()) \
        .hexdigest()[:8]


seal = {
    'phase': PHASE,
    'name': NAME,
    'npz_sha256_8': fsha8(npz_path),
    'result_sha256_8': fsha8(res_path),
    'exec_sha256_8': fsha8(exep),
    'script_sha256_8': fsha8(SCRIPT),
}
with io.open(os.path.join(OUT, 'seal.json'),
             'w', encoding='utf-8') as f:
    json.dump(seal, f, ensure_ascii=False,
              indent=1)
log('sealed npz8=%s result8=%s exec8=%s '
    'script8=%s'
    % (seal['npz_sha256_8'],
       seal['result_sha256_8'],
       seal['exec_sha256_8'],
       seal['script_sha256_8']))

with io.open(LOGF, 'w',
             encoding='utf-8') as f:
    f.write('\n'.join(_o) + '\n')
print('PHASE%d DONE verdict=%s elapsed=%.1fs '
      'forwards=%d'
      % (PHASE, verdict, time.time() - t0,
         FW[0]))
