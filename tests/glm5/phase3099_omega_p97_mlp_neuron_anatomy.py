# -*- coding: utf-8 -*-
"""Phase 3099  Omega-P97: inside the
last-block MLP - neuron-level anatomy
of the conditional TT rewrite (4B
block35 / 14B block39, dual-arm).

3098 proved (verdict
fifth_lastblock_mlp_rewrite): the
conditional TT rewrite of the last
decoder block is performed by the MLP
substep (frac_mlp 0.947/0.949; H_D3
True/True), with family-common vs
family-idiosyncratic MLP increments
reversing with condition (f2dm ci2
0.57-0.99 vs 14B formal AC/BC -0.11/
-0.17).  Open questions: (Q1) is the
rewrite carried by a CONCENTRATED set
of intermediate neurons or diffuse;
(Q2) do style conditions (ci2,
Shakespearean) reuse a CROSS-FAMILY
neuron set while formal conditions
stay family-idiosyncratic; (Q3) does
the MLP increment Delta-m first-order
bridge to the logit-space TT (i.e. is
the final-norm nonlinearity
negligible for the rewrite).

Pipeline (per arm, native only,
frozen):  assemble the 3076-identical
prompts;  per prompt ONE native
forward with record hooks on the LAST
block (index NL-1):  a = self_attn
out, m = mlp out, h2 = block out,
h0 = hidden_states[NL-1].  Manual MLP
replay on the final position:
  h1  = h0 + a
  x   = blk.post_attention_layernorm
        (h1)
  act = act_fn(gate_proj(x)) *
        up_proj(x)
  m_man = down_proj(act)
Banks kept per family (fp32, bf16
values lossless):  LG native logits,
ACT replayed activations (inter),
TTI native logit-space TT rows (f64),
DA Delta-act rows (f64, cross-family
kept);  DM Delta-m rows are transient
per family.  Manual MLP replay uses
WHOLE-sequence GEMMs (same shapes as
the block internals) so that it is
bit-exact: a per-position GEMV replay
differs by up to 0.25 in bf16
(observed in smoke) because cuBLAS
accumulation order changes with the
GEMM shape.

Anchors (frozen):
  d1: m_man == hooked m, bit 0 per
      prompt (MLP replay exact).
  d2: (h0+a)+m_man == hooked h2,
      bit 0 (block reassembly exact).
  d3: re-computed cross-family cos of
      native logit TT vs 3098 sealed
      F2I[key][:,0] <= 1e-9 (same
      prompts, same readout path).
  d4: 3098 npz sha8 == its seal.json
      (== eeba6c18) - sealed inputs
      untampered.
  d5: head(norm(h2)) vs native
      logits, bit 0 (readout path
      exact).

Neuron analysis (per side):
  Delta-act[k] = act[pref_k] -
  act[base_k] in R^inter;
  nf2[key][k] = cos(Delta-act_fa,
  Delta-act_fb) cross-family;
  shareK[k] = L1 share of the top-K
  |Delta-act| entries, K in
  {1,8,64,256,1024};
  consensus top-64 per (family, ci):
  median |Delta-act| over the 8
  bodies of the group, stable argsort
  descending;
  J64[key][ci] = Jaccard(top64_fa,
  top64_fb);
  JCROSS[fa][ci_a,ci_b] = Jaccard
  within family across conditions;
  Delta-m[k] = m[pref_k] - m[base_k]
  (hook m difference; exactness of
  the replay guaranteed by d1);
  contribution c_j = |Delta-act_j| *
  ||W_down[:,j]||_2;  top-1024 rebuild
  r = sum_{j in top1024} Delta-act_j *
  W_down[:,j];  SHM = ||r||/||Delta-m||;
  COSM = cos(r, Delta-m);
  bridge[k] = cos(head(norm(Delta-m)),
  TT_logit[k]) with TT_logit = native
  logit-space TT (3096/3098 def).

Preregistered decision gates
(frozen):  active group (from 3098
sealed F2S): |med(post)-med(pre)|
>= 0.10.
  H_E1 (concentration): median over
  the pooled per-group med share-256
  (2 arms x 9 groups; group pool = 8
  bodies x 2 families) >= 0.5.
  H_E2a (style-shared neurons):
  median J64 over active ci2
  (key,side) groups >= 3 x median
  J64 of the 14B formal AC/BC ci1
  control groups; empty pool -> False.
  H_E3 (first-order bridge): median
  over the pooled bridge values
  (2 arms x 3 families x 24 pairs)
  >= 0.8.
  verdict ladder:
    not H_E1 ->
    fifth_mlp_neuron_diffuse;
    H_E1 and not H_E2a ->
    fifth_mlp_neuron_family_
    idiosyncratic;
    H_E1 and H_E2a and not H_E3 ->
    fifth_mlp_neuron_nobridge;
    all True ->
    fifth_mlp_neuron_style_common.
SMOKE: prefixes 0..1 only (ci1
pairs), verdict
fifth_mlp_neuron_smoke.

Limitations (recorded): top-K shares
are L1-based; consensus sets use
median |Delta-act| (magnitude, not
sign); rebuild rank capped at 1024;
single last position; n=8 bodies per
group; causal-connective paradigm
only; zero-side null not modeled.

Memory discipline: 4B arm completes
and frees (del + gc + empty_cache)
BEFORE the 14B load (model-sequential
OOM rule).  banks fp32 (bf16 values
lossless); W_down column norms and
top-M rebuild done in fp32/bf16 GPU
chunks; no quantization.
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

PHASE = 3099
NAME = 'omega_p97_mlp_neuron_anatomy'
SEED = 3099
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
SMOKE = os.environ.get('SMOKE', '0') == '1'
OUT = (R13 + r'\phase3099' + '\\' + NAME)
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
           'inter': 9728},
    '14B': {'mdir': os.path.join(
                ROOT, 'models', 'hf',
                'Qwen3-14B'),
            'nl': 40, 'hid': 5120,
            'inter': 17408}}

GATES = {
    'active_abs_ge': 0.10,
    'H_E1_share256_ge': 0.5,
    'H_E2a_ratio_ge': 3.0,
    'H_E3_bridge_ge': 0.8,
    'K_LIST': (1, 8, 64, 256, 1024),
    'TOP64': 64,
    'TOPM': 1024}

z98 = np.load(P98, allow_pickle=False)
seal98 = json.load(io.open(
    P98SEAL, encoding='utf-8'))
d4a = (h8(P98) == seal98['npz_sha256_8'])
d4b = (seal98['npz_sha256_8']
       == 'eeba6c18')
log('d4 3098 npz sha8 match=%s '
    'seal8==eeba6c18=%s' % (d4a, d4b))

exec_doc = {
    'phase': PHASE, 'name': NAME,
    'frozen_before_compute': True,
    'question': ('which intermediate '
                 'neurons of the last-block '
                 'MLP carry the conditional '
                 'TT rewrite; is the neuron '
                 'set style-shared across '
                 'families; does Delta-m '
                 'first-order bridge to the '
                 'logit TT'),
    'pipeline': {
        'replay': ('h1=h0+a; x=ln2(h1); '
                   'act=act_fn(gate(x))*'
                   'up(x); m_man=down(act); '
                   'd1 m_man==hook m bit0; '
                   'd2 (h0+a)+m_man==hook '
                   'h2 bit0'),
        'neuron': ('Delta-act = act[pref]-'
                   'act[base]; nf2 cross-'
                   'family cos; shareK L1 '
                   'top-K; consensus top-64 '
                   'median |Delta-act|; '
                   'J64 Jaccard; JCROSS '
                   'within family; Delta-m '
                   'top-1024 rebuild by '
                   '|Delta-act_j|*'
                   '||W_down[:,j]||; bridge '
                   'cos(head(norm(Delta-m)),'
                   ' TT_logit)'),
        'scope': ('last block only, '
                  'native forwards only, '
                  'final position only')},
    'gates': GATES,
    'anchors': {
        'd1': 'm_man == hook m bit 0',
        'd2': '(h0+a)+m_man == hook h2 '
              'bit 0',
        'd3': 'F2I native recompute vs '
              '3098 sealed <=1e-9',
        'd4': '3098 npz sha8 vs seal '
              '(eeba6c18)',
        'd5': 'head(norm(h2)) vs native '
              'logits bit 0'},
    'inputs': {
        'p3098_npz8': h8(P98),
        'p3098_seal8':
            seal98['npz_sha256_8']},
    'models': {k: {'mdir': v['mdir'],
                   'nl': v['nl'],
                   'hid': v['hid'],
                   'inter': v['inter']}
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


def jac(s1, s2):
    s1 = set(int(x) for x in s1)
    s2 = set(int(x) for x in s2)
    u = s1 | s2
    if not u:
        return 0.0
    return float(len(s1 & s2)) \
        / float(len(u))


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
    # shapes as the block internals.
    def hook(module, args, output):
        t = output[0] \
            if isinstance(output, tuple) \
            else output
        store[tag] = t.detach()
    return hook


def run_arm(side):
    cfg = ARM[side]
    NL = cfg['nl']
    LB = NL - 1
    HID = cfg['hid']
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
    log('[%s] loaded nl=%d last_block=%d '
        'inter=%d'
        % (side, NL, LB, INTER))
    # W_down column norms, fp32 chunked
    colnorm = np.zeros(INTER, np.float64)
    CH = 2048
    with torch.no_grad():
        for a0 in range(0, INTER, CH):
            a1 = min(a0 + CH, INTER)
            cn = torch.linalg.norm(
                Wd[:, a0:a1].float(), dim=0)
            colnorm[a0:a1] = cn.double() \
                .cpu().numpy()
    log('[%s] W_down colnorm done' % side)
    n_pr = len(CONS) * 8
    A = {'d1': 0.0, 'd2': 0.0,
         'd3': 0.0, 'd5': 0.0}
    F2IN = {fa + fb: np.zeros(NP_)
            for (fa, fb) in CPAIRS}
    NF2 = {fa + fb: np.zeros(NP_)
           for (fa, fb) in CPAIRS}
    SHARE = {fa: {K: np.zeros(NP_)
                  for K in GATES['K_LIST']}
             for fa in FKEYS}
    SHM = {fa: np.zeros(NP_)
           for fa in FKEYS}
    COSM = {fa: np.zeros(NP_)
            for fa in FKEYS}
    BRIDGE = {fa: np.zeros(NP_)
              for fa in FKEYS}
    TOP64 = {fa: {ci: None
                  for ci in range(1, NCI + 1)}
             for fa in FKEYS}
    TTI = {}
    DA = {}
    for fkey in FKEYS:
        assembled, idx_of, cidx, bidx = \
            assemble(fkey, tok)
        lg = np.zeros((n_pr, 151936),
                      dtype=np.float32)
        act_b = np.zeros((n_pr, INTER),
                         dtype=np.float32)
        m_b = np.zeros((n_pr, HID),
                       dtype=np.float32)
        hpost = np.zeros((n_pr, HID),
                         dtype=np.float32)
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
                    mk_rec(S, 'h2'))]
            with torch.no_grad():
                out = model(
                    ids_t, use_cache=False,
                    output_hidden_states=True)
            for hd in hs_:
                hd.remove()
            # full-sequence tensors: the
            # replay must use the SAME GEMM
            # shapes as the block internals,
            # otherwise bf16 kernels differ
            # (per-position GEMV vs whole-
            # sequence GEMM changed bits by
            # up to 0.25 in smoke).
            h0f = out.hidden_states[LB] \
                .detach()
            af = S['a']
            mf = S['m']
            h2f = S['h2']
            # manual MLP replay (whole
            # sequence, same shapes)
            h1f = h0f + af
            with torch.no_grad():
                x = blk \
                    .post_attention_layernorm(
                        h1f)
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
            hpost[i] = h2f[0, -1].detach() \
                .float().cpu().numpy()
            del out, S, h0f, af, mf, h2f
            del h1f, x, g, u, actf, mmf, h2mf
            if i % 8 == 7:
                log('[%s][%s] prompts %d/1cfg'
                    % (side, fkey, i + 1))
                flush_log()
        # d5 readout anchor
        with torch.no_grad():
            z2 = head(norm_mod(
                torch.from_numpy(hpost)
                .to('cuda')
                .to(torch.bfloat16)))
        A['d5'] = max(A['d5'], float(
            np.max(np.abs(
                z2.double().cpu().numpy()
                - lg.astype(np.float64)))))
        del z2, hpost
        # banks + delta rows
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
        DA[fkey] = da
        TTI[fkey] = tti
        # bridge batched
        dmt = torch.from_numpy(dm) \
            .to('cuda').to(torch.bfloat16)
        with torch.no_grad():
            bv = head(norm_mod(dmt))
        BV = bv.double().cpu().numpy()
        del dmt, bv
        for k in range(NP_):
            BRIDGE[fkey][k] = cosv(
                BV[k], tti[k])
        # shareK per k
        for k in range(NP_):
            mag = np.abs(da[k])
            s1 = float(mag.sum())
            if s1 <= 0:
                for K in GATES['K_LIST']:
                    SHARE[fkey][K][k] = 0.0
                continue
            srt = np.sort(mag)[::-1]
            csum = np.cumsum(srt)
            for K in GATES['K_LIST']:
                SHARE[fkey][K][k] = \
                    float(csum[K - 1]) / s1
        # top-M rebuild per k
        TM = GATES['TOPM']
        for k in range(NP_):
            cj = np.abs(da[k]) * colnorm
            S_ix = np.argsort(-cj,
                              kind='stable'
                              )[:TM]
            da_t = torch.tensor(
                da[k][S_ix],
                dtype=torch.float32,
                device='cuda') \
                .to(torch.bfloat16)
            with torch.no_grad():
                r = Wd[:, S_ix] @ da_t
            r64 = r.double().cpu().numpy()
            SHM[fkey][k] = float(
                np.linalg.norm(r64)) \
                / max(float(np.linalg.norm(
                    dm[k])), 1e-12)
            COSM[fkey][k] = cosv(
                r64, dm[k])
            del da_t, r
        # consensus top-64 per ci
        for ci in range(1, NCI + 1):
            ks = [k for k in range(NP_)
                  if int(cidx[k]) == ci]
            magm = np.median(
                np.abs(da[ks]), axis=0)
            top = np.argsort(-magm,
                             kind='stable'
                             )[:GATES['TOP64']]
            TOP64[fkey][ci] = top \
                .astype(np.int64)
        log('[%s][%s] family done: d1='
            '%.1e d2=%.1e d5=%.1e '
            'shm_med=%.4f br_med=%.4f'
            % (side, fkey, A['d1'],
               A['d2'], A['d5'],
               float(np.median(SHM[fkey])),
               float(np.median(
                   BRIDGE[fkey]))))
        flush_log()
        del dm, lg, act_b, m_b, da, tti
        del BV
        gc.collect()
        torch.cuda.empty_cache()
    # cross-family: nf2 + F2I native d3
    for (fa, fb) in CPAIRS:
        key = fa + fb
        for k in range(NP_):
            NF2[key][k] = cosv(
                DA[fa][k], DA[fb][k])
            F2IN[key][k] = cosv(
                TTI[fa][k], TTI[fb][k])
            A['d3'] = max(A['d3'], abs(
                F2IN[key][k]
                - float(z98['F2I_%s_%s'
                             % (side, key)]
                        [k, 0])))
    # J64 / JCROSS
    J64D = {fa + fb: np.zeros(NCI)
            for (fa, fb) in CPAIRS}
    JCR = {}
    for (fa, fb) in CPAIRS:
        key = fa + fb
        for ci in range(1, NCI + 1):
            J64D[key][ci - 1] = jac(
                TOP64[fa][ci],
                TOP64[fb][ci])
    for fa in FKEYS:
        JCR[fa] = np.zeros((NCI, NCI))
        for ca in range(1, NCI + 1):
            for cb in range(1, NCI + 1):
                JCR[fa][ca - 1, cb - 1] = jac(
                    TOP64[fa][ca],
                    TOP64[fa][cb])
    log('[%s] pairs done: d3=%.1e '
        'nf2_med=%.4f'
        % (side, A['d3'], float(
            np.median(np.concatenate(
                [NF2[key]
                 for key in ('AB', 'AC',
                             'BC')])))))
    flush_log()
    del DA, TTI
    del model, tok, norm_mod, head, blk
    del mlp, Wd, colnorm
    gc.collect()
    torch.cuda.empty_cache()
    log('[%s] arm done, memory freed'
        % side)
    return {'F2IN': F2IN, 'NF2': NF2,
            'SHARE': SHARE, 'SHM': SHM,
            'COSM': COSM,
            'BRIDGE': BRIDGE,
            'TOP64': TOP64,
            'J64': J64D, 'JCR': JCR,
            'A': A, 'nl': NL}


RES = {}
RES['4B'] = run_arm('4B')
RES['14B'] = run_arm('14B')
flush_log()

ok = True
for side in ('4B', '14B'):
    A = RES[side]['A']
    assert A['d1'] == 0.0, (side, A['d1'])
    assert A['d2'] == 0.0, (side, A['d2'])
    assert A['d5'] == 0.0, (side, A['d5'])
    assert A['d3'] <= 1e-9, (side, A['d3'])
assert d4a and d4b
log('anchors ok d1/d2/d5 bit0, d3 '
    '<=1e-9, d4 sha match + eeba6c18')
flush_log()

CIDX = np.array([k // 8 + 1
                 for k in range(24)])
if SMOKE:
    VERDICT = 'fifth_mlp_neuron_smoke'
    log('SMOKE verdict: %s' % VERDICT)
    flush_log()
    E = {}
else:
    E = {'active': {}, 'share': {},
         'j64': {}, 'jcross': {},
         'nf2': {}, 'bridge': {},
         'shm': {}, 'cosm': {},
         'gates': {}, 'decision': {}}
    # active groups from 3098 sealed F2S
    ACTV = {}
    for side in ('4B', '14B'):
        for key in ('AB', 'AC', 'BC'):
            f2s = z98['F2S_%s_%s'
                      % (side, key)]
            for ci in (1, 2, 3):
                m = CIDX == ci
                ck = '%s_%s_ci%d' % (
                    side, key, ci)
                cliff = float(
                    np.median(f2s[m, 2])
                    - np.median(f2s[m, 0]))
                is_act = abs(cliff) \
                    >= GATES['active_abs_ge']
                ACTV[ck] = bool(is_act)
                E['active'][ck] = {
                    'active': bool(is_act),
                    'cliff_3098': cliff}
    n_act = sum(ACTV.values())
    log('active groups (from 3098 '
        'sealed): %d/18' % n_act)
    # per-group medians
    for side in ('4B', '14B'):
        R = RES[side]
        for key in ('AB', 'AC', 'BC'):
            for ci in (1, 2, 3):
                m = CIDX == ci
                ck = '%s_%s_ci%d' % (
                    side, key, ci)
                sp = np.concatenate(
                    [R['SHARE'][fa][256][m]
                     for fa in key])
                E['share'][ck] = float(
                    np.median(sp))
                E['nf2'][ck] = float(
                    np.median(
                        R['NF2'][key][m]))
                E['bridge'][ck] = float(
                    np.median(
                        np.concatenate(
                            [R['BRIDGE'][fa][m]
                             for fa in key])))
                E['shm'][ck] = float(
                    np.median(
                        np.concatenate(
                            [R['SHM'][fa][m]
                             for fa in key])))
                E['cosm'][ck] = float(
                    np.median(
                        np.concatenate(
                            [R['COSM'][fa][m]
                             for fa in key])))
                E['j64']['%s_%s_ci%d'
                         % (side, key, ci)] \
                    = float(R['J64'][key]
                            [ci - 1])
        for fa in FKEYS:
            E['jcross']['%s_%s'
                        % (side, fa)] = \
                R['JCR'][fa].tolist()
    # H_E1
    pool1 = [E['share']['%s_%s_ci%d'
                        % (side, key, ci)]
             for side in ('4B', '14B')
             for key in ('AB', 'AC', 'BC')
             for ci in (1, 2, 3)]
    v1 = float(np.median(pool1))
    H_E1 = bool(v1 >= GATES
                ['H_E1_share256_ge'])
    # H_E2a
    pool2 = [E['j64']['%s_%s_ci2'
                      % (side, key)]
             for side in ('4B', '14B')
             for key in ('AB', 'AC', 'BC')
             if ACTV['%s_%s_ci2'
                     % (side, key)]]
    ctrl = [E['j64']['14B_%s_ci1'
                     % key]
            for key in ('AC', 'BC')]
    if len(pool2) == 0:
        v2 = None
        v2c = float(np.median(ctrl))
        H_E2a = False
    else:
        v2 = float(np.median(pool2))
        v2c = float(np.median(ctrl))
        H_E2a = bool(v2 >= GATES
                     ['H_E2a_ratio_ge']
                     * v2c)
    # H_E3
    pool3 = [float(RES[side]['BRIDGE'][fa][k])
             for side in ('4B', '14B')
             for fa in FKEYS
             for k in range(24)]
    v3 = float(np.median(pool3))
    H_E3 = bool(v3 >= GATES
                ['H_E3_bridge_ge'])
    E['gates'] = {
        'H_E1': H_E1, 'H_E1_val': v1,
        'H_E2a': H_E2a,
        'H_E2a_pool': v2,
        'H_E2a_ctrl': v2c,
        'H_E2a_pool_n': len(pool2),
        'H_E3': H_E3, 'H_E3_val': v3}
    log('gates: H_E1=%s (%.4f) H_E2a=%s '
        '(%s vs ctrl %.4f, n=%d) H_E3=%s '
        '(%.4f)'
        % (H_E1, v1, H_E2a,
           ('%.4f' % v2) if v2 is not None
           else 'NA', v2c, len(pool2),
           H_E3, v3))
    # verdict ladder
    if not H_E1:
        VERDICT = \
            'fifth_mlp_neuron_diffuse'
    elif not H_E2a:
        VERDICT = \
            'fifth_mlp_neuron_family_' \
            'idiosyncratic'
    elif not H_E3:
        VERDICT = \
            'fifth_mlp_neuron_nobridge'
    else:
        VERDICT = \
            'fifth_mlp_neuron_style_common'
    E['decision'] = {
        'H_E1': H_E1, 'H_E2a': H_E2a,
        'H_E3': H_E3}
    log('VERDICT: %s' % VERDICT)

# ---------- persist ----------
save = {
    'VERDICT': np.array(VERDICT),
    'SMOKE': np.bool_(SMOKE),
    'PHASE': np.int64(PHASE),
    'SEED': np.int64(SEED),
    'D4_OK': np.bool_(bool(d4a and d4b))}
for side in ('4B', '14B'):
    R = RES[side]
    for tag in ('d1', 'd2', 'd3', 'd5'):
        save['%s_%s' % (tag.upper(), side)] \
            = np.float64(R['A'][tag])
    for key in ('AB', 'AC', 'BC'):
        save['F2IN_%s_%s' % (side, key)] = \
            R['F2IN'][key].astype(np.float64)
        save['NF2_%s_%s' % (side, key)] = \
            R['NF2'][key].astype(np.float64)
        save['J64_%s_%s' % (side, key)] = \
            R['J64'][key].astype(np.float64)
    for fa in FKEYS:
        save['SHM_%s_%s' % (side, fa)] = \
            R['SHM'][fa].astype(np.float64)
        save['COSM_%s_%s' % (side, fa)] = \
            R['COSM'][fa].astype(np.float64)
        save['BRIDGE_%s_%s'
             % (side, fa)] = \
            R['BRIDGE'][fa].astype(
                np.float64)
        save['JCROSS_%s_%s'
             % (side, fa)] = \
            R['JCR'][fa].astype(np.float64)
        for K in GATES['K_LIST']:
            save['SHARE_%s_%s_K%d'
                 % (side, fa, K)] = \
                R['SHARE'][fa][K].astype(
                    np.float64)
        for ci in range(1, NCI + 1):
            save['TOP64_%s_%s_ci%d'
                 % (side, fa, ci)] = \
                R['TOP64'][fa][ci]
npz_path = OUT + r'\%s.npz' % NAME
np.savez(npz_path, **save)

res = {
    'phase': PHASE, 'name': NAME,
    'verdict': VERDICT,
    'smoke': SMOKE,
    'forwards': len(CONS) * 8 * 3 * 2,
    'anchors': {
        'd1_4B': RES['4B']['A']['d1'],
        'd1_14B': RES['14B']['A']['d1'],
        'd2_4B': RES['4B']['A']['d2'],
        'd2_14B': RES['14B']['A']['d2'],
        'd3_4B': RES['4B']['A']['d3'],
        'd3_14B': RES['14B']['A']['d3'],
        'd5_4B': RES['4B']['A']['d5'],
        'd5_14B': RES['14B']['A']['d5'],
        'd4_ok': bool(d4a and d4b),
        'ok': True}}
if not SMOKE:
    res['stats'] = {
        k: v for k, v in E.items()
        if k != 'decision'}
with io.open(OUT + r'\result.json', 'w',
             encoding='utf-8') as f:
    json.dump(res, f, indent=1,
              ensure_ascii=False)
seal = {
    'npz_sha256_8': h8(npz_path),
    'result_sha256_8': h8(
        OUT + r'\result.json'),
    'script_sha256_8': h8(
        os.path.abspath(__file__)),
    'execution_sha256_8': h8(
        OUT + r'\execution.json')}
with io.open(OUT + r'\seal.json', 'w',
             encoding='utf-8') as f:
    json.dump(seal, f, indent=1)
log('sealed npz8=%s result8=%s'
    % (seal['npz_sha256_8'],
       seal['result_sha256_8']))
with io.open(LOGF, 'w',
             encoding='utf-8') as f:
    f.write('\n'.join(LOGS) + '\n')
print('RUN_COMPLETE %s' % VERDICT)
