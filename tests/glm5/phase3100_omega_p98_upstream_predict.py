# -*- coding: utf-8 -*-
"""Phase 3100  Omega-P98: upstream-to-
rewriter coupling - first-order
predictability of the last-block MLP
increment and the natural carrier test
of the 3093 focal heads (4B block35 /
14B block39, dual-arm, native only).

Chain so far: 3098 proved the
conditional TT rewrite is performed by
the LAST-block MLP substep; 3099
showed the rewrite is diffuse but
linearly decomposable (top-1024
rebuild cos 0.965-0.996) with
direction-domain conditional commonality
(nf2 inversion) and near first-order
readout bridging.  3093 (14B, L_INJ=37)
found per-head causal focal sets: the
8 heads whose post-o_proj last-position
output blocks (HEAD_IDX = 128-wide
coordinate slices of the 512-d... of
the 5120-d attention output) carry the
family-redirect recovery (TOP8 by the
3071 criterion on r1_nh).  Missing
link: the NATURAL forward coupling
from the upstream increments (Delta-h0,
Delta-a at the last block; per-head
increments at L37) to the rewriter
increment Delta-m.

Questions (frozen):
  Q1 (first-order MLP): is the MLP
      increment Delta-m predicted by
      the one-step Jacobian at the
      base working point, with the
      EXACT linear upstream map
      (Delta-g, Delta-u from bank
      differences, so only the silu
      nonlinearity is first-order)?
      Delta-act_pred = silu'(g_b) *
      (u_b * Dg + g_b * Du);
      Delta-m_pred = W_down @
      Delta-act_pred (f64 CPU exact).
  Q2 (channel split): splitting
      Delta-x first-order through the
      RMSNorm Jacobian into the
      attention channel J@Delta-a and
      the residual channel J@Delta-h0,
      what share of ||Delta-m_pred||^2
      is carried by the attention
      channel?
  Q3 (natural carrier, 14B only): are
      the 3093 causal focal heads
      (TOP8_A/B/C) ALSO the largest
      natural per-head increments at
      L37 (Delta-BH37 per-128-block L2
      share, top-8 by share)?  overlap
      per family + Spearman(share,
      r1_nh) recorded.

Pipeline (per arm, native only,
frozen):  3076-identical prompts
(assemble identical to 3099);  per
prompt ONE native forward with hooks
on the LAST block (a / m / h2 whole
sequence) + o_proj forward_pre hook
(pre-o_proj concat, last position) +
the L37 self_attn hook (14B only,
last position, post-o_proj 5120);
h0 = hidden_states[LB].  Manual MLP
replay WHOLE-sequence (3099 GEMM
lesson: same shapes as internals,
bit-0).  Banks per family (fp32):
LG, ACT, M, H2, H0, A, H1, X (ln2
out), G (gate pre), U (up pre),
CONC (pre-o_proj, last pos),
BH37 (14B).  All Jacobian statistics
in f64 on CPU with W_gate/W_up/
W_down materialized f64 (exact
linear algebra; avoids the bf16 GEMM
shape trap entirely - the trap only
affects bit-level anchors, which use
the whole-sequence replay).

Anchors (frozen):
  d1: m_man == hooked m, bit 0.
  d2: (h0+a)+m_man == hooked h2,
      bit 0.
  d3: F2I native recompute vs 3098
      sealed F2I[key][:,0] <= 1e-9
      (both arms).
  d4: 3098 npz sha8 == eeba6c18 and
      3093 npz sha8 == its seal.json.
  d5: head(norm(h2)) vs native
      logits, bit 0.
  d6 (14B, formal only): LG bank and
      recomputed TT rows vs the 3093
      sealed LG_A/B/C and TT_A/B/C
      <= 1e-3 (f32 storage tolerance;
      row order identical by
      construction: bi-major, ci
      minor, 32 prompts).

Gates (frozen):
  H_F1: median over the pooled
      pred_cos (2 arms x 3 families
      x 24 pairs) >= 0.7.
  H_F2: median over the pooled
      a-share (energy share
      ||Dm_pred_a||^2/||Dm_pred||^2)
      >= 0.5.
  H_F3 (14B): median over the 3
      families of overlap(top8_nat,
      TOP8_3093) >= 3 (of 8).
verdict ladder (frozen):
  not H_F1 -> sixth_mlp_nonlinearity;
  H_F1 and not H_F2 ->
  sixth_upstream_residual;
  H_F1 and H_F2 and not H_F3 ->
  sixth_head_mismatch;
  all True -> sixth_upstream_coupled.
4B sub-verdict (no H_F3, recorded
not gating the main verdict):
  not H_F1(4B) -> sixth_mlp_
  nonlinearity; else not H_F2(4B) ->
  sixth_upstream_residual; else
  sixth_upstream_coupled_nof3.
SMOKE: prefixes 0..1 only (NP_=8),
d6 skipped, verdict
sixth_upstream_smoke.

Limitations (recorded): single last
position; ln2 nonlinearity only
first-order (x_pred_cos recorded to
attribute H_F1 failures); Q3 is a
natural-carrier correspondence, not
a causal intervention (3093 supplies
the causal side); o_proj mixing
means the 128-wide blocks are
coordinate slices, not isolated
heads (3093-identical convention);
gamma of post_attention_layernorm
included in the Jacobian; eps from
config.

Memory discipline: 4B arm completes
and frees BEFORE the 14B load; f64
weight copies (14B ~2.1 GB CPU) are
per-arm and deleted; banks fp32;
no quantization.
"""
import gc
import hashlib
import io
import json
import os
from datetime import datetime

import numpy as np
import torch
from transformers import AutoModelForCausalLM, \
    AutoTokenizer

PHASE = 3100
NAME = 'omega_p98_upstream_predict'
SEED = 3100
ROOT = r'D:\AI2050\Ai2050-OpenOne'
R13 = (ROOT + r'\tests\glm5\result'
       r'\rdc_query_construction_20260913')
P98 = (R13 + r'\phase3098'
       r'\omega_p96_lastblock_substep_'
       r'anatomy'
       r'\omega_p96_lastblock_substep_'
       r'anatomy.npz')
P98SEAL = (R13 + r'\phase3098'
           r'\omega_p96_lastblock_substep_'
           r'anatomy\seal.json')
P93 = (R13 + r'\phase3093'
       r'\omega_p91_qwen14b_l37_full_'
       r'arbitration'
       r'\omega_p91_qwen14b_l37_full_'
       r'arbitration.npz')
P93SEAL = (R13 + r'\phase3093'
           r'\omega_p91_qwen14b_l37_full_'
           r'arbitration\seal.json')
SMOKE = os.environ.get('SMOKE', '0') == '1'
OUT = (R13 + r'\phase3100' + '\\' + NAME)
if SMOKE:
    OUT = OUT + r'\smoke'
LOGF = OUT + r'\run_log.txt'
LOGS = []


def log(msg):
    LOGS.append(msg)


def flush_log():
    with io.open(LOGF, 'w',
                 encoding='utf-8') as f:
        f.write('\n'.join(LOGS) + '\n')


def h8(path):
    h = hashlib.sha256()
    with io.open(path, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 20),
                        b''):
            h.update(blk)
    return h.hexdigest()[:8]


os.makedirs(OUT, exist_ok=True)
log('execution frozen smoke=%s' % SMOKE)

# frozen 3076 texts (bit-identical)
TARGETS = ('so', 'because', 'therefore',
           'however', 'while', 'yet',
           'although', 'thus')
FAM = {
    'A': {
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
                     'Regarding the weather,')},
    'B': {
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
                     'experiment,')},
    'C': {
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
                     'conversation,')}}
FKEYS = ('A', 'B', 'C')
CPAIRS = (('A', 'B'), ('A', 'C'), ('B', 'C'))
CONS = [0, 1] if SMOKE else [0, 1, 2, 3]
NP_ = 8 if SMOKE else 24
NCI = len(CONS) - 1

ARM = {
    '4B': {'mdir': os.path.join(
               ROOT, 'models', 'hf',
               'qwen3-4b'),
           'nl': 36, 'hid': 2560,
           'inter': 9728, 'nq': 32,
           'hd': 128, 'l_bh': None},
    '14B': {'mdir': os.path.join(
                ROOT, 'models', 'hf',
                'Qwen3-14B'),
            'nl': 40, 'hid': 5120,
            'inter': 17408, 'nq': 40,
            'hd': 128, 'l_bh': 37}}

GATES = {
    'H_F1_pred_cos_ge': 0.7,
    'H_F2_ashare_ge': 0.5,
    'H_F3_overlap_ge': 3,
    'TOPN': 8,
    'D6_TOL': 1e-3}

z98 = np.load(P98, allow_pickle=False)
seal98 = json.load(io.open(
    P98SEAL, encoding='utf-8'))
z93 = np.load(P93, allow_pickle=False)
seal93 = json.load(io.open(
    P93SEAL, encoding='utf-8'))
d4a = (h8(P98) == seal98['npz_sha256_8'])
d4b = (seal98['npz_sha256_8']
       == 'eeba6c18')
d4c = (h8(P93) == seal93['npz_sha256_8'])
log('d4 3098 sha8 match=%s (==eeba6c18=%s)'
    ' 3093 sha8 match seal=%s'
    % (d4a, d4b, d4c))

exec_doc = {
    'phase': PHASE, 'name': NAME,
    'frozen_before_compute': True,
    'question': ('is the last-block MLP '
                 'increment first-order '
                 'predictable from the '
                 'upstream increments; which '
                 'channel (attention vs h0 '
                 'residual) carries it; are '
                 'the 3093 causal focal heads '
                 'the natural carriers at L37'),
    'pipeline': {
        'replay': ('whole-sequence MLP '
                   'replay (3099 GEMM lesson); '
                   'd1/d2 bit 0'),
        'jacobian': ('Delta-act_pred = '
                     'silu\'(g_b)*(u_b*Dg+'
                     'g_b*Du) with Dg/Du = '
                     'bank differences '
                     '(exact linear upstream, '
                     'only silu first-order); '
                     'Delta-m_pred = W_down @ '
                     'Delta-act_pred in f64 '
                     'CPU; channel split via '
                     'the RMSNorm Jacobian '
                     '(gamma included) into '
                     'J@Delta-a and '
                     'J@Delta-h0'),
        'carrier': ('14B L37 post-o_proj '
                    'last-position 5120 vector '
                    'split into 40x128 '
                    'coordinate blocks '
                    '(3093 HEAD_IDX '
                    'convention); per-block '
                    'L2 share of Delta-BH37; '
                    'top8 vs 3093 TOP8'),
        'scope': ('last block + L37 '
                  'observation only, native '
                  'forwards only, final '
                  'position only')},
    'gates': GATES,
    'anchors': {
        'd1': 'm_man == hook m bit 0',
        'd2': '(h0+a)+m_man == hook h2 '
              'bit 0',
        'd3': 'F2I native recompute vs '
              '3098 sealed <=1e-9',
        'd4': '3098 sha8==eeba6c18; 3093 '
              'sha8==seal',
        'd5': 'head(norm(h2)) vs native '
              'logits bit 0',
        'd6': '14B LG/TT vs 3093 sealed '
              '<=1e-3 (formal only)'},
    'inputs': {
        'p3098_npz8': h8(P98),
        'p3098_seal8':
            seal98['npz_sha256_8'],
        'p3093_npz8': h8(P93),
        'p3093_seal8':
            seal93['npz_sha256_8']},
    'models': {k: {'mdir': v['mdir'],
                   'nl': v['nl'],
                   'hid': v['hid'],
                   'inter': v['inter'],
                   'nq': v['nq'],
                   'l_bh': v['l_bh']}
               for k, v in ARM.items()},
    'forwards_per_arm': len(CONS) * 8 * 3,
    'smoke': SMOKE}
with io.open(OUT + r'\execution.json',
             'w', encoding='utf-8') as f:
    json.dump(exec_doc, f, indent=1,
              ensure_ascii=False)
log('execution.json written')


def cosv(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b) / (na * nb)


def silu_deriv(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return s * (1.0 + x * (1.0 - s))


def rms_jv(h, gamma, v, eps, hid):
    # Jacobian-vector product of
    # y = h / rms(h) * gamma at h,
    # rms = sqrt(mean(h^2)+eps).
    ms = float(np.mean(h * h)) + eps
    rms = np.sqrt(ms)
    hv = float(h @ v)
    return gamma * (v / rms
                    - h * hv / (rms ** 3
                                * hid))


def spearman(x, y):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def assemble(fkey, tok):
    fam = FAM[fkey]
    bodies = fam['bodies']
    prefixes = fam['prefixes']
    word_tok = {}
    for w in TARGETS:
        wi = tok(' ' + w,
                 add_special_tokens=False)[
            'input_ids']
        assert len(wi) == 1, (fkey, w)
        word_tok[w] = int(wi[0])
    assembled = []
    for bi in range(len(bodies)):
        for ci in CONS:
            s = (prefixes[ci] + ' '
                 + bodies[bi]) \
                if prefixes[ci] else bodies[bi]
            ids = [int(x) for x in tok(
                s,
                add_special_tokens=False)[
                    'input_ids']]
            t = word_tok[TARGETS[bi]]
            assert ids.count(t) == 1, \
                (fkey, bi, ci)
            assembled.append(
                {'ids': ids, 'cond': ci,
                 'body': bi})
    idx_of = {}
    for i in range(len(assembled)):
        idx_of[(assembled[i]['cond'],
                assembled[i]['body'])] = i
    for it in assembled:
        ci = it['cond']
        if ci == 0:
            it['off'] = 0
        else:
            bid = assembled[idx_of[(0,
                it['body'])]]['ids']
            pid = it['ids']
            off = len(pid) - len(bid)
            assert off > 0, (fkey, ci)
            assert list(pid[off + 1:]) \
                == list(bid[1:]), (fkey, ci)
            it['off'] = off
    cidx = []
    bidx = []
    for ci in (1, 2, 3)[:NCI]:
        for bi in range(len(bodies)):
            cidx.append(ci)
            bidx.append(bi)
    assert len(cidx) == NP_
    return assembled, idx_of, \
        np.array(cidx), np.array(bidx)


def mk_rec(store, tag):
    # whole-sequence record: the manual
    # replay must see the same GEMM
    # shapes as the block internals
    # (3099 lesson).
    def hook(module, args, output):
        t = output[0] \
            if isinstance(output, tuple) \
            else output
        store[tag] = t.detach()
    return hook


def mk_pre(store, tag):
    # forward_pre_hook on o_proj: the
    # input is the pre-o_proj concat
    # (1, seq, NQ*HD).
    def hook(module, args):
        store[tag] = args[0].detach()
    return hook


def f64_cpu(mat, chunks=1024):
    # materialize a bf16 GPU linear as
    # an f64 CPU array, chunked.
    rows, cols = mat.shape
    out = np.empty((rows, cols),
                   np.float64)
    with torch.no_grad():
        for a0 in range(0, cols, chunks):
            a1 = min(a0 + chunks, cols)
            out[:, a0:a1] = mat[:,
                                a0:a1] \
                .double().cpu().numpy()
    return out


def run_arm(side):
    cfg = ARM[side]
    NL = cfg['nl']
    LB = NL - 1
    HID = cfg['hid']
    NQ = cfg['nq']
    HD = cfg['hd']
    NQW = NQ * HD
    L_BH = cfg['l_bh']
    log('[%s] ==== arm begin (%s) ===='
        % (side, cfg['mdir']))
    tok = AutoTokenizer.from_pretrained(
        cfg['mdir'])
    model = AutoModelForCausalLM \
        .from_pretrained(
            cfg['mdir'],
            torch_dtype=torch.bfloat16,
            attn_implementation='eager') \
        .to('cuda').eval()
    layers = model.model.layers
    assert len(layers) == NL
    assert int(model.config.hidden_size) \
        == HID
    INTER = int(
        model.config.intermediate_size)
    assert INTER == cfg['inter'], \
        (side, INTER, cfg['inter'])
    assert int(model.config.vocab_size) \
        == 151936
    blk = layers[LB]
    norm_mod = model.model.norm
    head = model.lm_head
    mlp = blk.mlp
    Wd = mlp.down_proj.weight
    assert tuple(Wd.shape) == (HID, INTER)
    ln2 = blk.post_attention_layernorm
    eps = float(model.config
                .rms_norm_eps)
    gamma = ln2.weight.detach() \
        .double().cpu().numpy()
    log('[%s] loaded nl=%d last_block=%d '
        'inter=%d nq=%d l_bh=%s'
        % (side, NL, LB, INTER, NQ,
           str(L_BH)))
    # f64 CPU weight copies (exact
    # linear algebra for the Jacobian
    # statistics; avoids bf16 GEMM
    # shape effects on statistics)
    Wd64 = f64_cpu(Wd)
    Wg64 = f64_cpu(mlp.gate_proj.weight)
    Wu64 = f64_cpu(mlp.up_proj.weight)
    log('[%s] f64 weights materialized'
        % side)
    n_pr = len(CONS) * 8
    A = {'d1': 0.0, 'd2': 0.0,
         'd3': 0.0, 'd5': 0.0,
         'd6a': 0.0, 'd6b': 0.0}
    F2IN = {fa + fb: np.zeros(NP_)
            for (fa, fb) in CPAIRS}
    TTI_BANK = {}
    PRED = {fa: np.zeros(NP_)
            for fa in FKEYS}
    ACTC = {fa: np.zeros(NP_)
            for fa in FKEYS}
    RESD = {fa: np.zeros(NP_)
            for fa in FKEYS}
    XPC = {fa: np.zeros(NP_)
           for fa in FKEYS}
    ASH = {fa: np.zeros(NP_)
           for fa in FKEYS}
    HSH = {fa: np.zeros(NP_)
           for fa in FKEYS}
    BHSH = {fa: np.zeros((NP_, NQ))
            for fa in FKEYS}
    T8N = {fa: np.zeros(GATES['TOPN'],
                        np.int64)
           for fa in FKEYS}
    OVL = {fa: 0 for fa in FKEYS}
    SP = {fa: 0.0 for fa in FKEYS}
    for fkey in FKEYS:
        assembled, idx_of, cidx, bidx = \
            assemble(fkey, tok)
        lg = np.zeros((n_pr, 151936),
                      dtype=np.float32)
        act_b = np.zeros((n_pr, INTER),
                         dtype=np.float32)
        m_b = np.zeros((n_pr, HID),
                       dtype=np.float32)
        h2_b = np.zeros((n_pr, HID),
                        dtype=np.float32)
        h0_b = np.zeros((n_pr, HID),
                        dtype=np.float32)
        a_b = np.zeros((n_pr, HID),
                       dtype=np.float32)
        x_b = np.zeros((n_pr, HID),
                       dtype=np.float32)
        g_b = np.zeros((n_pr, INTER),
                       dtype=np.float32)
        u_b = np.zeros((n_pr, INTER),
                       dtype=np.float32)
        conc_b = np.zeros((n_pr, NQW),
                          dtype=np.float32)
        bh37 = np.zeros((n_pr, NQW),
                        dtype=np.float32) \
            if L_BH is not None else None
        hook37 = (layers[L_BH].self_attn
                  if L_BH is not None
                  else None)
        for i, it in enumerate(assembled):
            ids_t = torch.tensor(
                [it['ids']], device='cuda')
            S = {}
            hs_ = [
                blk.self_attn
                .register_forward_hook(
                    mk_rec(S, 'a')),
                blk.mlp
                .register_forward_hook(
                    mk_rec(S, 'm')),
                blk.register_forward_hook(
                    mk_rec(S, 'h2')),
                blk.self_attn.o_proj
                .register_forward_pre_hook(
                    mk_pre(S, 'c'))]
            if hook37 is not None:
                hs_.append(
                    hook37
                    .register_forward_hook(
                        mk_rec(S, 'b37')))
            with torch.no_grad():
                out = model(
                    ids_t, use_cache=False,
                    output_hidden_states=True)
            for hd_ in hs_:
                hd_.remove()
            h0f = out.hidden_states[LB] \
                .detach()
            af = S['a']
            mf = S['m']
            h2f = S['h2']
            h1f = h0f + af
            with torch.no_grad():
                x = ln2(h1f)
                g = mlp.gate_proj(x)
                u = mlp.up_proj(x)
                actf = mlp.act_fn(g) * u
                mmf = mlp.down_proj(actf)
            h2mf = h1f + mmf
            d1v = float((mmf[0, -1] \
                - mf[0, -1]) \
                .abs().max().item())
            A['d1'] = max(A['d1'], d1v)
            d2v = float((h2mf[0, -1] \
                - h2f[0, -1]) \
                .abs().max().item())
            A['d2'] = max(A['d2'], d2v)
            lg[i] = out.logits[0, -1] \
                .detach().float() \
                .cpu().numpy()
            act_b[i] = actf[0, -1] \
                .detach().float() \
                .cpu().numpy()
            m_b[i] = mmf[0, -1].detach() \
                .float().cpu().numpy()
            h2_b[i] = h2f[0, -1].detach() \
                .float().cpu().numpy()
            h0_b[i] = h0f[0, -1].detach() \
                .float().cpu().numpy()
            a_b[i] = af[0, -1].detach() \
                .float().cpu().numpy()
            x_b[i] = x[0, -1].detach() \
                .float().cpu().numpy()
            g_b[i] = g[0, -1].detach() \
                .float().cpu().numpy()
            u_b[i] = u[0, -1].detach() \
                .float().cpu().numpy()
            conc_b[i] = S['c'][0, -1] \
                .detach().float() \
                .cpu().numpy()
            if bh37 is not None:
                bh37[i] = S['b37'][0, -1] \
                    .detach().float() \
                    .cpu().numpy()
            del out, S, h0f, af, mf, h2f
            del h1f, x, g, u, actf, mmf
            del h2mf
            if i % 8 == 7:
                log('[%s][%s] prompts %d/1cfg'
                    % (side, fkey, i + 1))
                flush_log()
        # d5 readout anchor
        with torch.no_grad():
            z2 = head(norm_mod(
                torch.from_numpy(h2_b)
                .to('cuda')
                .to(torch.bfloat16)))
        A['d5'] = max(A['d5'], float(
            np.max(np.abs(
                z2.double().cpu().numpy()
                - lg.astype(np.float64)))))
        del z2
        # d6: LG/TT vs 3093 sealed (14B,
        # formal only; identical row
        # order by construction)
        if side == '14B' and not SMOKE:
            lg93 = z93['LG_' + fkey] \
                .astype(np.float64)
            d6a = float(np.max(np.abs(
                lg.astype(np.float64)
                - lg93)))
            tt93 = z93['TT_' + fkey] \
                .astype(np.float64)
            ttr = np.zeros((NP_, 151936))
            for k in range(NP_):
                crow = idx_of[(int(
                    cidx[k]),
                    int(bidx[k]))]
                brow = idx_of[(0,
                    int(bidx[k]))]
                ttr[k] = lg[crow] \
                    .astype(np.float64) \
                    - lg[brow] \
                    .astype(np.float64)
            d6b = float(np.max(np.abs(
                ttr - tt93)))
            # sealed self-check: TT rows
            # must equal the LG pref-base
            # differences inside 3093
            # (row order bi-major ci
            # minor: pref rows are
            # 1::4, 2::4, 3::4)
            pr = np.concatenate(
                [lg93[1::4], lg93[2::4],
                 lg93[3::4]], axis=0)
            br = np.tile(lg93[0::4], (3, 1))
            tt_sc = float(np.max(np.abs(
                tt93 - (pr - br)[:NP_])))
            A['d6a'] = d6a
            A['d6b'] = d6b
            log('[%s][%s] d6 lg_diff=%.3e '
                'tt_diff=%.3e '
                'tt_selfcheck=%.3e'
                % (side, fkey, d6a, d6b,
                   tt_sc))
            del lg93, tt93, ttr
        # delta rows + Jacobian statistics
        da = np.zeros((NP_, INTER),
                      np.float64)
        dm = np.zeros((NP_, HID),
                      np.float64)
        tti = np.zeros((NP_, 151936),
                       np.float64)
        for k in range(NP_):
            crow = idx_of[(int(cidx[k]),
                           int(bidx[k]))]
            brow = idx_of[(0, int(bidx[k]))]
            da[k] = act_b[crow] \
                .astype(np.float64) \
                - act_b[brow] \
                .astype(np.float64)
            dm[k] = m_b[crow] \
                .astype(np.float64) \
                - m_b[brow].astype(np.float64)
            tti[k] = lg[crow] \
                .astype(np.float64) \
                - lg[brow].astype(np.float64)
            # --- Q1 first-order MLP ---
            gp = g_b[brow] \
                .astype(np.float64)
            up_ = u_b[brow] \
                .astype(np.float64)
            dgp = g_b[crow] \
                .astype(np.float64) - gp
            dup = u_b[crow] \
                .astype(np.float64) - up_
            sp = silu_deriv(gp)
            dap = sp * (up_ * dgp
                        + gp * dup)
            dmp = Wd64 @ dap
            PRED[fkey][k] = cosv(dmp, dm[k])
            ACTC[fkey][k] = cosv(dap, da[k])
            RESD[fkey][k] = \
                float(np.linalg.norm(
                    dm[k] - dmp)) \
                / max(float(np.linalg.norm(
                    dm[k])), 1e-12)
            # --- Q2 channel split ---
            h1row = h0_b[brow] \
                .astype(np.float64) \
                + a_b[brow] \
                .astype(np.float64)
            dh0 = h0_b[crow] \
                .astype(np.float64) \
                - h0_b[brow] \
                .astype(np.float64)
            daa = a_b[crow] \
                .astype(np.float64) \
                - a_b[brow] \
                .astype(np.float64)
            dx_t = x_b[crow] \
                .astype(np.float64) \
                - x_b[brow].astype(np.float64)
            dx_a = rms_jv(h1row, gamma, daa,
                          eps, HID)
            dx_h0 = rms_jv(h1row, gamma,
                           dh0, eps, HID)
            XPC[fkey][k] = cosv(
                dx_a + dx_h0, dx_t)
            dga = Wg64 @ dx_a
            dua = Wu64 @ dx_a
            dap_a = sp * (up_ * dga
                          + gp * dua)
            dmp_a = Wd64 @ dap_a
            dgh = Wg64 @ dx_h0
            duh = Wu64 @ dx_h0
            dap_h = sp * (up_ * dgh
                          + gp * duh)
            dmp_h = Wd64 @ dap_h
            e_a = float(dmp_a @ dmp_a)
            e_h = float(dmp_h @ dmp_h)
            e_t = float(dmp @ dmp)
            ASH[fkey][k] = e_a / max(e_t,
                                     1e-24)
            HSH[fkey][k] = e_h / max(e_t,
                                     1e-24)
            # --- Q3 per-head at L37 ---
            if bh37 is not None:
                dbh = bh37[crow] \
                    .astype(np.float64) \
                    - bh37[brow] \
                    .astype(np.float64)
                nh = np.sqrt(
                    (dbh.reshape(NQ, HD)
                     ** 2).sum(axis=1))
                ssum = float(nh.sum())
                if ssum > 0:
                    BHSH[fkey][k] = nh \
                        / ssum
                del dbh, nh
            del dap, dmp, sp, gp, up_, dgp
            del dup, dh0, daa, dx_t, dx_a
            del dx_h0, dga, dua, dap_a
            del dmp_a, dgh, duh, dap_h
            del dmp_h
        TTI_BANK[fkey] = tti
        # 3093 carrier comparison (14B)
        if bh37 is not None and not SMOKE:
            r193 = z93['R1_ALLNH_'
                       + fkey] \
                .astype(np.float64)
            t893 = z93['TOP8_' + fkey] \
                .astype(np.int64)
            sh_med = np.median(
                BHSH[fkey], axis=0)
            top = np.argsort(-sh_med,
                             kind='stable'
                             )[:GATES['TOPN']]
            T8N[fkey] = top \
                .astype(np.int64)
            OVL[fkey] = len(set(
                int(x) for x in top)
                & set(int(x)
                      for x in t893))
            SP[fkey] = spearman(sh_med,
                                r193)
            log('[%s][%s] Q3 top8_nat=%s '
                'top8_3093=%s overlap=%d '
                'spearman=%.3f'
                % (side, fkey,
                   list(map(int, top)),
                   list(map(int, t893)),
                   OVL[fkey], SP[fkey]))
            del r193, t893, sh_med, top
        elif bh37 is not None:
            sh_med = np.median(
                BHSH[fkey], axis=0)
            top = np.argsort(-sh_med,
                             kind='stable'
                             )[:GATES['TOPN']]
            T8N[fkey] = top \
                .astype(np.int64)
            log('[%s][%s] Q3(smoke) '
                'top8_nat=%s'
                % (side, fkey,
                   list(map(int, top))))
            del sh_med, top
        log('[%s][%s] family done: d1='
            '%.1e d2=%.1e d5=%.1e '
            'pred_med=%.4f ashare_med='
            '%.4f'
            % (side, fkey, A['d1'],
               A['d2'], A['d5'],
               float(np.median(
                   PRED[fkey])),
               float(np.median(
                   ASH[fkey]))))
        flush_log()
        del lg, act_b, m_b, h2_b, h0_b
        del a_b, x_b, g_b, u_b, conc_b
        del da, dm
        if bh37 is not None:
            del bh37
        gc.collect()
        torch.cuda.empty_cache()
    # cross-family F2I native (d3):
    # TTI banks retained per family.
    for (fa, fb) in CPAIRS:
        key = fa + fb
        for k in range(NP_):
            F2IN[key][k] = cosv(
                TTI_BANK[fa][k],
                TTI_BANK[fb][k])
            A['d3'] = max(A['d3'], abs(
                F2IN[key][k]
                - float(z98['F2I_%s_%s'
                             % (side, key)]
                        [k, 0])))
    log('[%s] pairs done: d3=%.1e'
        % (side, A['d3']))
    flush_log()
    del TTI_BANK
    del model, tok, norm_mod, head, blk
    del mlp, Wd, ln2, gamma, Wd64
    del Wg64, Wu64
    gc.collect()
    torch.cuda.empty_cache()
    log('[%s] arm done, memory freed'
        % side)
    return {'PRED': PRED, 'ACTC': ACTC,
            'RESD': RESD, 'XPC': XPC,
            'ASH': ASH, 'HSH': HSH,
            'BHSH': BHSH, 'T8N': T8N,
            'OVL': OVL, 'SP': SP,
            'F2IN': F2IN, 'A': A,
            'nl': NL}


RES = {}
RES['4B'] = run_arm('4B')
RES['14B'] = run_arm('14B')
flush_log()

# ==== verdict (frozen gates) ====
all_pred = np.concatenate(
    [RES[s]['PRED'][fa]
     for s in ('4B', '14B')
     for fa in FKEYS])
all_ash = np.concatenate(
    [RES[s]['ASH'][fa]
     for s in ('4B', '14B')
     for fa in FKEYS])
med_pred = float(np.median(all_pred))
med_ash = float(np.median(all_ash))
H_F1 = bool(med_pred
            >= GATES['H_F1_pred_cos_ge'])
H_F2 = bool(med_ash
            >= GATES['H_F2_ashare_ge'])
ovl = [RES['14B']['OVL'][fa]
       for fa in FKEYS]
med_ovl = float(np.median(ovl))
H_F3 = bool(med_ovl
            >= GATES['H_F3_overlap_ge'])
# per-arm gate values
med_pred_4b = float(np.median(
    np.concatenate([RES['4B']['PRED'][fa]
                    for fa in FKEYS])))
med_pred_14b = float(np.median(
    np.concatenate(
        [RES['14B']['PRED'][fa]
         for fa in FKEYS])))
med_ash_4b = float(np.median(
    np.concatenate([RES['4B']['ASH'][fa]
                    for fa in FKEYS])))
med_ash_14b = float(np.median(
    np.concatenate(
        [RES['14B']['ASH'][fa]
         for fa in FKEYS])))
setup_ok = bool(
    d4a and d4b and d4c
    and RES['4B']['A']['d1'] == 0.0
    and RES['4B']['A']['d2'] == 0.0
    and RES['4B']['A']['d3'] <= 1e-9
    and RES['4B']['A']['d5'] == 0.0
    and RES['14B']['A']['d1'] == 0.0
    and RES['14B']['A']['d2'] == 0.0
    and RES['14B']['A']['d3'] <= 1e-9
    and RES['14B']['A']['d5'] == 0.0)
if not SMOKE:
    setup_ok = setup_ok and bool(
        RES['14B']['A']['d6a']
        <= GATES['D6_TOL']
        and RES['14B']['A']['d6b']
        <= GATES['D6_TOL'])
log('setup_ok=%s' % setup_ok)
if SMOKE:
    verdict = 'sixth_upstream_smoke'
elif not setup_ok:
    verdict = 'sixth_setup_failed'
else:
    if not H_F1:
        verdict = 'sixth_mlp_nonlinearity'
    elif not H_F2:
        verdict = 'sixth_upstream_residual'
    elif not H_F3:
        verdict = 'sixth_head_mismatch'
    else:
        verdict = 'sixth_upstream_coupled'
# 4B sub-verdict (no H_F3)
if not (med_pred_4b
        >= GATES['H_F1_pred_cos_ge']):
    verdict_4b = 'sixth_mlp_nonlinearity'
elif not (med_ash_4b
          >= GATES['H_F2_ashare_ge']):
    verdict_4b = 'sixth_upstream_residual'
else:
    verdict_4b = \
        'sixth_upstream_coupled_nof3'
log('GATES: med_pred=%.4f med_ash='
    '%.4f med_ovl=%.1f'
    % (med_pred, med_ash, med_ovl))
log('H_F1=%s H_F2=%s H_F3=%s'
    % (H_F1, H_F2, H_F3))
log('per-arm: pred 4B=%.4f 14B=%.4f '
    'ash 4B=%.4f 14B=%.4f'
    % (med_pred_4b, med_pred_14b,
       med_ash_4b, med_ash_14b))
log('VERDICT: %s (4B sub: %s)'
    % (verdict, verdict_4b))

# ==== persist ====
save = {
    'VERDICT': np.array(verdict),
    'VERDICT_4B': np.array(verdict_4b),
    'SMOKE': np.bool_(SMOKE),
    'PHASE': np.int64(PHASE)}
for s in ('4B', '14B'):
    for fa in FKEYS:
        save['PRED_COS_%s_%s' % (s, fa)] \
            = RES[s]['PRED'][fa] \
            .astype(np.float64)
        save['ACT_COS_%s_%s' % (s, fa)] \
            = RES[s]['ACTC'][fa] \
            .astype(np.float64)
        save['RESID_%s_%s' % (s, fa)] \
            = RES[s]['RESD'][fa] \
            .astype(np.float64)
        save['XPCOS_%s_%s' % (s, fa)] \
            = RES[s]['XPC'][fa] \
            .astype(np.float64)
        save['ASHARE_%s_%s' % (s, fa)] \
            = RES[s]['ASH'][fa] \
            .astype(np.float64)
        save['HSHARE_%s_%s' % (s, fa)] \
            = RES[s]['HSH'][fa] \
            .astype(np.float64)
        save['BHSHARE_%s_%s' % (s, fa)] \
            = RES[s]['BHSH'][fa] \
            .astype(np.float32)
        save['TOP8NAT_%s_%s' % (s, fa)] \
            = RES[s]['T8N'][fa] \
            .astype(np.int64)
for fa in FKEYS:
    save['OVERLAP_14B_%s' % fa] = \
        np.int64(RES['14B']['OVL'][fa])
    save['SPEAR_14B_%s' % fa] = \
        np.float64(RES['14B']['SP'][fa])
for (fa, fb) in CPAIRS:
    for s in ('4B', '14B'):
        save['F2IN_%s_%s' % (s, fa + fb)] = \
            RES[s]['F2IN'][fa + fb] \
            .astype(np.float64)
for s in ('4B', '14B'):
    for kk, vv in RES[s]['A'].items():
        save['ANCH_%s_%s' % (s, kk)] = \
            np.float64(vv)
save['MED_PRED'] = np.float64(med_pred)
save['MED_ASHARE'] = np.float64(med_ash)
save['MED_OVL'] = np.float64(med_ovl)
save['MED_PRED_4B'] = \
    np.float64(med_pred_4b)
save['MED_PRED_14B'] = \
    np.float64(med_pred_14b)
save['MED_ASHARE_4B'] = \
    np.float64(med_ash_4b)
save['MED_ASHARE_14B'] = \
    np.float64(med_ash_14b)
save['H_F1'] = np.bool_(H_F1)
save['H_F2'] = np.bool_(H_F2)
save['H_F3'] = np.bool_(H_F3)
save['GATES'] = np.array(json.dumps(
    GATES))
NPZ = OUT + '\\' + NAME + '.npz'
np.savez_compressed(NPZ, **save)
npz8 = h8(NPZ)
log('npz saved sha8=%s' % npz8)

res_doc = {
    'phase': PHASE, 'name': NAME,
    'verdict': verdict,
    'verdict_4b': verdict_4b,
    'smoke': SMOKE,
    'gates': {
        'med_pred': med_pred,
        'med_ashare': med_ash,
        'med_overlap': med_ovl,
        'H_F1': H_F1, 'H_F2': H_F2,
        'H_F3': H_F3,
        'per_arm': {
            'med_pred_4b': med_pred_4b,
            'med_pred_14b': med_pred_14b,
            'med_ash_4b': med_ash_4b,
            'med_ash_14b': med_ash_14b},
        'overlap_14b': ovl,
        'spearman_14b': {
            fa: RES['14B']['SP'][fa]
            for fa in FKEYS},
        'top8_nat_14b': {
            fa: [int(x) for x in
                 RES['14B']['T8N'][fa]]
            for fa in FKEYS},
        'top8_3093': {
            fa: [int(x) for x in
                 z93['TOP8_' + fa]]
            for fa in FKEYS}},
    'anchors': {
        'd1_max': {s: RES[s]['A']['d1']
                   for s in ('4B', '14B')},
        'd2_max': {s: RES[s]['A']['d2']
                   for s in ('4B', '14B')},
        'd3_max': {s: RES[s]['A']['d3']
                   for s in ('4B', '14B')},
        'd4': {'p98_ok': bool(d4a and d4b),
               'p93_ok': bool(d4c)},
        'd5_max': {s: RES[s]['A']['d5']
                   for s in ('4B', '14B')},
        'd6_14B': {
            'lg_diff': RES['14B']['A'][
                'd6a'],
            'tt_diff': RES['14B']['A'][
                'd6b']}},
    'medians': {
        fa: {'pred_cos_4b': float(
                 np.median(
                     RES['4B']['PRED'][fa])),
             'pred_cos_14b': float(
                 np.median(RES['14B']
                           ['PRED'][fa])),
             'ashare_4b': float(
                 np.median(
                     RES['4B']['ASH'][fa])),
             'ashare_14b': float(
                 np.median(RES['14B']
                           ['ASH'][fa])),
             'act_cos_4b': float(
                 np.median(
                     RES['4B']['ACTC'][fa])),
             'act_cos_14b': float(
                 np.median(RES['14B']
                           ['ACTC'][fa])),
             'xpcos_4b': float(
                 np.median(
                     RES['4B']['XPC'][fa])),
             'xpcos_14b': float(
                 np.median(RES['14B']
                           ['XPC'][fa]))}
        for fa in FKEYS},
    'inputs': {
        'p3098_npz8': h8(P98),
        'p3093_npz8': h8(P93)},
    'npz_sha256_8': npz8,
    'frozen_inputs': exec_doc['inputs'],
    'finished_at': datetime.now() \
        .strftime('%Y-%m-%d %H:%M:%S')}
RESF = OUT + r'\result.json'
with io.open(RESF, 'w',
             encoding='utf-8') as f:
    json.dump(res_doc, f, indent=1,
              ensure_ascii=False)
result8 = h8(RESF)
SEALF = OUT + r'\seal.json'
with io.open(SEALF, 'w',
             encoding='utf-8') as f:
    json.dump({'phase': PHASE,
               'npz_sha256_8': npz8,
               'result_sha256_8': result8},
              f, indent=1)
log('result saved sha8=%s' % result8)
log('RUN_COMPLETE %s' % verdict)
flush_log()
print('RUN_COMPLETE %s' % verdict)
