# -*- coding: utf-8 -*-
"""Phase 3098  Omega-P96: last-block
substep anatomy + intervention (4B
block35 / 14B block39, dual-arm).

3097 proved: the 14B f2 cliff is ONE
step at L39->L40 and that step IS the
last decoder block (block39) computation
(same norm+head readout path on both
sides).  Open questions: (Q1) WHICH
substep of the block rewrites the TT
directions - attention or MLP; (Q2) is
the rewrite causally material to the
output distribution.

Pipeline (per arm, frozen):
  assemble the 3076-identical prompts
  (4 prefixes x 8 causal-connective
  bodies per family); per prompt run 4
  forwards that differ ONLY in hooks on
  the LAST block (index NL-1):
    native   : record h0 (=hidden_states
               [NL-1]), attn out a, mlp out
               m, block out h2, logits;
    no_attn  : self_attn output zeroed
               (h1=h0, then normal mlp);
    no_mlp   : mlp output zeroed (h2=h1);
    skip     : whole block returns its
               input (h2=h0).
  substep readouts Z_s = head(norm(h_s)),
  s in {pre, att, post} (h2 is pre-final-
  norm so norm+head = native path);
  TT_s[k] = Z_s[pref_k] - Z_s[base_k];
  F2S[key][k,s] = cos(TT_f_s, TT_g_s)
  (same-model cross-family, 3096 def);
  F2I[key][k,c] = f2 from the 4 config
  logit sets (native/no_attn/no_mlp/
  skip).  Cliff decomposition:
  attn_step = med(att)-med(pre),
  mlp_step = med(post)-med(att) per
  (pair,ci) group; frac_mlp = |mlp_step|
  /(|attn_step|+|mlp_step|).

Anchors (frozen):
  c0: (h0+a)+m == h2 hook, bit 0 (same
      bf16 adds as the block internals).
  c1: head(norm(h2)) vs native logits
      <=1e-6, expect bit 0.
  c2: F2I native col vs sealed F2_CTT
      (3079/3093) <=1e-9; AND F2S post
      col == F2I native col bit 0.
  c3: F2S pre col vs 3096 sealed F2P
      [:, NL-1] <=1e-9.
  c4: 3096 npz sha8 == its seal.json.
  c5: skip logits vs head(norm(h0))
      bit 0 (skip = readout of h0).

Preregistered decision gates (frozen):
  active group: |med(post)-med(pre)|
  >= 0.10.  attribution per model:
  med(frac_mlp | active) >= 0.7 -> mlp;
  <= 0.3 -> attn; else mixed; no active
  -> no_active.
  H_D3 (causal materiality, per model,
  attribution in {mlp,attn}): med KL
  (native||no_dom) > med KL(native||
  no_oth) AND mean top1 agree(no_dom) <
  mean top1 agree(no_oth).
  verdict:
    fifth_lastblock_no_active /
    fifth_lastblock_mlp_rewrite /
    fifth_lastblock_attn_rewrite /
    fifth_lastblock_mixed_rewrite /
    fifth_lastblock_attribution_split.
SMOKE: prefixes 0..1 only (ci1 pairs),
verdict fifth_lastblock_smoke.

Stats (no gate): TT-norm ratio post/
pre per family (shrink vs amplify of
the condition effect), KEEP cos(TT_pre,
TT_post) per family (does a family's
own condition effect survive), and f2
of the readout-space step vectors
DTT_att/DTT_mlp (is the rewrite family-
common or family-idiosyncratic).

Limitations (recorded): zeroing a
substep output is an ablation, not a
minimal intervention (downstream ln2
sees h0/h1 instead of h1/h2); last
position only; n=8 bodies per ci group;
causal-connective paradigm only.

Memory discipline: 4B arm completes and
frees (del + gc + empty_cache) BEFORE
the 14B load (model-sequential OOM
rule).  per-config logits kept fp32
transiently per family; hidden banks
fp32 (bf16 values stored losslessly).
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

PHASE = 3098
NAME = 'omega_p96_lastblock_substep_' \
       'anatomy'
SEED = 3098
ROOT = r'D:\AI2050\Ai2050-OpenOne'
R13 = (ROOT + r'\tests\glm5\result'
       r'\rdc_query_construction_20260913')
P96 = (R13 + r'\phase3096'
       r'\omega_p94_f2_lens_layer_profile'
       r'\omega_p94_f2_lens_layer_'
       r'profile.npz')
P96SEAL = (R13 + r'\phase3096'
           r'\omega_p94_f2_lens_layer_'
           r'profile\seal.json')
P4B79 = (R13 + r'\phase3079'
         r'\omega_p76_migration_lock'
         r'\omega_p76_migration_lock.npz')
P14B = (R13 + r'\phase3093'
        r'\omega_p91_qwen14b_l37_full_'
        r'arbitration'
        r'\omega_p91_qwen14b_l37_full_'
        r'arbitration.npz')
SMOKE = os.environ.get('SMOKE', '0') == '1'
OUT = (R13 + r'\phase3098' + '\\' + NAME)
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

ARM = {
    '4B': {'mdir': os.path.join(
               ROOT, 'models', 'hf',
               'qwen3-4b'),
           'nl': 36, 'hid': 2560,
           'nq': 32, 'kv': 8,
           'sealed': P4B79},
    '14B': {'mdir': os.path.join(
                ROOT, 'models', 'hf',
                'Qwen3-14B'),
            'nl': 40, 'hid': 5120,
            'nq': 40, 'kv': 8,
            'sealed': P14B}}

GATES = {
    'active_abs_ge': 0.10,
    'mlp_dominant_frac_ge': 0.7,
    'attn_dominant_frac_le': 0.3,
    'H_D3': ('med KL(nat||no_dom) > '
             'med KL(nat||no_oth) AND '
             'mean top1 agree(no_dom) < '
             'mean top1(no_oth)')}
CFG_NAMES = ('native', 'no_attn',
             'no_mlp', 'skip')

z96 = np.load(P96, allow_pickle=False)
seal96 = json.load(io.open(
    P96SEAL, encoding='utf-8'))
c4_ok = (h8(P96) == seal96['npz_sha256_8'])
log('c4 3096 npz sha8 match=%s' % c4_ok)

exec_doc = {
    'phase': PHASE, 'name': NAME,
    'frozen_before_compute': True,
    'question': ('which substep (attn vs '
                 'mlp) of the last decoder '
                 'block performs the '
                 'conditional TT rewrite, '
                 'and is it causally '
                 'material to the output '
                 'distribution'),
    'pipeline': {
        'substeps': ('h0=hs[NL-1]; h1=h0+a; '
                     'h2=h1+m (block out, '
                     'pre-final-norm); '
                     'Z_s=head(norm(h_s)); '
                     'TT_s[k]=Z_s[pref_k]-'
                     'Z_s[base_k]; F2S=cos '
                     'cross-family'),
        'interventions': CFG_NAMES,
        'hook': ('zero substep output / '
                 'block returns input; '
                 'last block only '
                 '(index NL-1)')},
    'gates': GATES,
    'anchors': {
        'c0': '(h0+a)+m == h2 bit 0',
        'c1': 'head(norm(h2)) vs native '
              'logits <=1e-6 (bit 0)',
        'c2': 'F2I native vs sealed '
              'F2_CTT <=1e-9; F2S post '
              '== F2I native bit 0',
        'c3': 'F2S pre vs 3096 F2P[:,'
              'NL-1] <=1e-9',
        'c4': '3096 npz sha8 vs seal',
        'c5': 'skip logits vs '
              'head(norm(h0)) bit 0'},
    'inputs': {
        'p3096_npz8': h8(P96),
        'p3096_seal8':
            seal96['npz_sha256_8'],
        'p4b79_npz8': h8(P4B79),
        'p14b_npz8': h8(P14B)},
    'models': {k: {'mdir': v['mdir'],
                   'nl': v['nl'],
                   'hid': v['hid']}
               for k, v in ARM.items()},
    'forwards_per_arm': len(CONS) * 8
                        * 4 * 3,
    'smoke': SMOKE}
with io.open(OUT + r'\execution.json',
             'w', encoding='utf-8') as f:
    json.dump(exec_doc, f, indent=1,
              ensure_ascii=False)
log('execution.json written')

SEALED = {'4B': np.load(P4B79,
                        allow_pickle=False),
          '14B': np.load(P14B,
                         allow_pickle=False)}


def cosv(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b) / (na * nb)


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
    for ci in (1, 2, 3)[:len(CONS) - 1]:
        for bi in range(len(bodies)):
            cidx.append(ci)
            bidx.append(bi)
    assert len(cidx) == NP_
    return assembled, idx_of, \
        np.array(cidx), np.array(bidx)


def mk_rec(store, tag):
    def hook(module, args, output):
        t = output[0] \
            if isinstance(output, tuple) \
            else output
        store[tag] = t[0, -1].detach()
    return hook


def mk_zero(module, args, output):
    if isinstance(output, tuple):
        return tuple(
            torch.zeros_like(x)
            if torch.is_tensor(x) else x
            for x in output)
    return torch.zeros_like(output)


def mk_skip(module, args, output):
    # this transformers version
    # assigns the layer output directly
    # (hidden_states = decoder_layer
    # (...)), so the hook must return
    # the bare input tensor.
    return args[0]


def run_arm(side):
    cfg = ARM[side]
    NL = cfg['nl']
    LB = NL - 1
    z_sealed = SEALED[side]
    log('[%s] ==== arm begin (%s) ===='
        % (side, cfg['mdir']))
    tok = AutoTokenizer.from_pretrained(
        cfg['mdir'])
    model = AutoModelForCausalLM.from_pretrained(
        cfg['mdir'],
        torch_dtype=torch.bfloat16,
        attn_implementation='eager') \
        .to('cuda').eval()
    layers = model.model.layers
    assert len(layers) == NL
    assert int(model.config.hidden_size) \
        == cfg['hid']
    assert int(model.config.vocab_size) \
        == 151936
    blk = layers[LB]
    norm_mod = model.model.norm
    head = model.lm_head
    log('[%s] loaded nl=%d last_block=%d'
        % (side, NL, LB))
    n_pr = len(CONS) * 8
    A = {'c0': 0.0, 'c1': 0.0,
         'c2': 0.0, 'c2b': 0.0,
         'c3': 0.0, 'c5': 0.0}
    F2S = {fa + fb: np.zeros((NP_, 3))
           for (fa, fb) in CPAIRS}
    F2I = {fa + fb: np.zeros((NP_, 4))
           for (fa, fb) in CPAIRS}
    F2DA = {fa + fb: np.zeros(NP_)
            for (fa, fb) in CPAIRS}
    F2DM = {fa + fb: np.zeros(NP_)
            for (fa, fb) in CPAIRS}
    NRM = {fa: np.zeros((NP_, 2))
           for fa in FKEYS}
    KEEP = {fa: np.zeros(NP_)
            for fa in FKEYS}
    KL = {fa: np.zeros((n_pr, 3))
          for fa in FKEYS}
    T1 = {fa: np.zeros((n_pr, 3))
          for fa in FKEYS}
    TT_F = {}
    TT_I = {}
    for fkey in FKEYS:
        assembled, idx_of, cidx, bidx = \
            assemble(fkey, tok)
        H = {s: np.zeros((n_pr, cfg['hid']),
                         dtype=np.float32)
             for s in ('pre', 'att', 'post')}
        LG = {c: np.zeros((n_pr, 151936),
                          dtype=np.float32)
              for c in CFG_NAMES}
        for i, it in enumerate(assembled):
            ids_t = torch.tensor(
                [it['ids']], device='cuda')
            # ---- native ----
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
            h0 = out.hidden_states[LB][0, -1] \
                .detach()
            a = S['a']
            m = S['m']
            h2 = S['h2']
            h1 = h0 + a
            h2m = h1 + m
            d0 = float((h2m - h2) \
                .abs().max().item())
            A['c0'] = max(A['c0'], d0)
            LG['native'][i] = out.logits[0, -1] \
                .detach().float() \
                .cpu().numpy()
            H['pre'][i] = h0.float() \
                .cpu().numpy()
            H['att'][i] = h1.float() \
                .cpu().numpy()
            H['post'][i] = h2.float() \
                .cpu().numpy()
            del out, S
            # ---- no_attn ----
            hd_ = blk.self_attn \
                .register_forward_hook(mk_zero)
            with torch.no_grad():
                out = model(ids_t,
                            use_cache=False)
            hd_.remove()
            LG['no_attn'][i] = \
                out.logits[0, -1].detach() \
                .float().cpu().numpy()
            del out
            # ---- no_mlp ----
            hd_ = blk.mlp \
                .register_forward_hook(mk_zero)
            with torch.no_grad():
                out = model(ids_t,
                            use_cache=False)
            hd_.remove()
            LG['no_mlp'][i] = \
                out.logits[0, -1].detach() \
                .float().cpu().numpy()
            del out
            # ---- skip ----
            hd_ = blk \
                .register_forward_hook(mk_skip)
            with torch.no_grad():
                out = model(ids_t,
                            use_cache=False)
            hd_.remove()
            LG['skip'][i] = \
                out.logits[0, -1].detach() \
                .float().cpu().numpy()
            del out
            if i % 8 == 7:
                log('[%s][%s] prompts %d/4cfg'
                    % (side, fkey, i + 1))
                flush_log()
        # c5 skip logits vs head(norm(h0))
        h0b = torch.from_numpy(
            H['pre']).to('cuda') \
            .to(torch.bfloat16)
        with torch.no_grad():
            zpre = head(norm_mod(h0b))
        A['c5'] = max(A['c5'], float(
            np.max(np.abs(
                zpre.double().cpu().numpy()
                - LG['skip'].astype(
                    np.float64)))))
        # c1 Z_post vs native logits
        with torch.no_grad():
            z2 = head(norm_mod(
                torch.from_numpy(H['post'])
                .to('cuda')
                .to(torch.bfloat16)))
        A['c1'] = max(A['c1'], float(
            np.max(np.abs(
                z2.double().cpu().numpy()
                - LG['native'].astype(
                    np.float64)))))
        # substep Z (this family only)
        ZS = {}
        for s in ('pre', 'att', 'post'):
            hb = torch.from_numpy(H[s]) \
                .to('cuda') \
                .to(torch.bfloat16)
            with torch.no_grad():
                z = head(norm_mod(hb))
            ZS[s] = z.double() \
                .cpu().numpy()
            del hb, z
        LGD = {c: LG[c].astype(np.float64)
               for c in CFG_NAMES}
        # per-family TT rows (kept for
        # cross-family pair f2 later)
        TT_F[fkey] = {}
        for s in ('pre', 'att', 'post'):
            TT_F[fkey][s] = np.stack([
                ZS[s][idx_of[(
                    int(cidx[k]),
                    int(bidx[k]))]]
                - ZS[s][idx_of[(
                    0, int(bidx[k]))]]
                for k in range(NP_)])
        TT_I[fkey] = {}
        for c in CFG_NAMES:
            TT_I[fkey][c] = np.stack([
                LGD[c][idx_of[(
                    int(cidx[k]),
                    int(bidx[k]))]]
                - LGD[c][idx_of[(
                    0, int(bidx[k]))]]
                for k in range(NP_)])
        # within-family stats
        for k in range(NP_):
            NRM[fkey][k, 0] = float(
                np.linalg.norm(
                    TT_F[fkey]['pre'][k]))
            NRM[fkey][k, 1] = float(
                np.linalg.norm(
                    TT_F[fkey]['post'][k]))
            KEEP[fkey][k] = cosv(
                TT_F[fkey]['pre'][k],
                TT_F[fkey]['post'][k])
        # KL / top1 (within family)
        lp = LGD['native'] \
            - LGD['native'].max(
                axis=1, keepdims=True)
        lp = lp - np.log(np.exp(lp) \
            .sum(axis=1, keepdims=True))
        for ci_, c in enumerate(
                ('no_attn', 'no_mlp',
                 'skip')):
            lq = LGD[c] - LGD[c].max(
                axis=1, keepdims=True)
            lq = lq - np.log(np.exp(lq) \
                .sum(axis=1,
                     keepdims=True))
            p = np.exp(lp)
            KL[fkey][:, ci_] = np.sum(
                p * (lp - lq), axis=1)
            T1[fkey][:, ci_] = (
                LGD[c].argmax(axis=1)
                == LGD['native'].argmax(
                    axis=1)).astype(
                np.float64)
        log('[%s][%s] family done: c0='
            '%.1e c1=%.1e c5=%.1e'
            % (side, fkey, A['c0'],
               A['c1'], A['c5']))
        flush_log()
        del H, LG, LGD, ZS
        gc.collect()
        torch.cuda.empty_cache()
    # cross-family pairs (all TT in RAM)
    for (fa, fb) in CPAIRS:
        key = fa + fb
        for k in range(NP_):
            for si, s in enumerate(
                    ('pre', 'att', 'post')):
                F2S[key][k, si] = cosv(
                    TT_F[fa][s][k],
                    TT_F[fb][s][k])
            for ci_, c in enumerate(
                    CFG_NAMES):
                F2I[key][k, ci_] = cosv(
                    TT_I[fa][c][k],
                    TT_I[fb][c][k])
            da_f = TT_F[fa]['att'][k] \
                - TT_F[fa]['pre'][k]
            da_g = TT_F[fb]['att'][k] \
                - TT_F[fb]['pre'][k]
            dm_f = TT_F[fa]['post'][k] \
                - TT_F[fa]['att'][k]
            dm_g = TT_F[fb]['post'][k] \
                - TT_F[fb]['att'][k]
            F2DA[key][k] = cosv(da_f,
                                da_g)
            F2DM[key][k] = cosv(dm_f,
                                dm_g)
    # c2 sealed + c2b bit
    for (fa, fb) in CPAIRS:
        key = fa + fb
        for k in range(NP_):
            A['c2'] = max(A['c2'], abs(
                F2I[key][k, 0]
                - float(z_sealed[
                    'F2_CTT_' + key][k])))
            A['c2b'] = max(A['c2b'], abs(
                F2S[key][k, 2]
                - F2I[key][k, 0]))
    # c3 pre vs 3096 F2P[:, NL-1]
    for (fa, fb) in CPAIRS:
        key = fa + fb
        col = z96['F2P_%s_%s'
                  % (side, key)][:, LB]
        for k in range(NP_):
            A['c3'] = max(A['c3'], abs(
                F2S[key][k, 0]
                - float(col[k])))
    log('[%s] pairs done: c2=%.1e '
        'c2b=%.1e c3=%.1e'
        % (side, A['c2'], A['c2b'],
           A['c3']))
    flush_log()
    del TT_F, TT_I
    del model, tok, norm_mod, head, blk
    gc.collect()
    torch.cuda.empty_cache()
    log('[%s] arm done, memory freed'
        % side)
    return {'F2S': F2S, 'F2I': F2I,
            'F2DA': F2DA, 'F2DM': F2DM,
            'NRM': NRM, 'KEEP': KEEP,
            'KL': KL, 'T1': T1,
            'A': A, 'nl': NL}


RES = {}
RES['4B'] = run_arm('4B')
RES['14B'] = run_arm('14B')
flush_log()

ok = True
for side in ('4B', '14B'):
    A = RES[side]['A']
    assert A['c0'] == 0.0, (side, A['c0'])
    assert A['c5'] == 0.0, (side, A['c5'])
    assert A['c2b'] == 0.0, \
        (side, A['c2b'])
    assert A['c1'] <= 1e-6, \
        (side, A['c1'])
    assert A['c2'] <= 1e-9, \
        (side, A['c2'])
    assert A['c3'] <= 1e-9, \
        (side, A['c3'])
assert c4_ok
log('anchors ok c0/c5/c2b bit0, c1/c2/c3 '
    '<=1e-6/1e-9/1e-9, c4 sha match')
flush_log()

if SMOKE:
    VERDICT = 'fifth_lastblock_smoke'
    log('SMOKE verdict: %s' % VERDICT)
    flush_log()
    E = {}
else:
    CIDX = np.array([k // 8 + 1
                     for k in range(24)])
    E = {'gs': {}, 'frac': {},
         'active': {}, 'attrib': {},
         'hd3': {}, 'agg': {},
         'rr': {}, 'keep': {},
         'f2da': {}, 'f2dm': {}}
    for side in ('4B', '14B'):
        NL = RES[side]['nl']
        fr = []
        n_act = 0
        for key in ('AB', 'AC', 'BC'):
            for ci in (1, 2, 3):
                m = CIDX == ci
                ck = '%s_ci%d' % (key, ci)
                g = [float(np.median(
                    RES[side]['F2S'][key][m,
                                           s]))
                    for s in (0, 1, 2)]
                E['gs']['%s_%s'
                        % (side, ck)] = g
                cliff = g[2] - g[0]
                is_act = abs(cliff) \
                    >= GATES['active_abs_ge']
                E['active']['%s_%s'
                            % (side, ck)] = \
                    bool(is_act)
                if is_act:
                    n_act += 1
                    ast = g[1] - g[0]
                    mst = g[2] - g[1]
                    den = abs(ast) + abs(mst)
                    fr.append(
                        abs(mst) / den
                        if den > 1e-12
                        else np.nan)
        if n_act == 0:
            att = 'no_active'
        else:
            fm = float(np.nanmedian(fr))
            if fm >= GATES[
                    'mlp_dominant_frac_ge']:
                att = 'mlp'
            elif fm <= GATES[
                    'attn_dominant_frac_le']:
                att = 'attn'
            else:
                att = 'mixed'
        E['attrib'][side] = att
        E['frac']['%s_med_frac_mlp'
                  % side] = (
            float(np.nanmedian(fr))
            if n_act else None)
        E['agg']['%s_n_active' % side] = \
            n_act
        log('[%s] attribution=%s '
            'n_active=%d med_frac_mlp=%s'
            % (side, att, n_act,
               ('%.3f'
                % E['frac']['%s_med_frac_'
                            'mlp' % side])
               if n_act else 'NA'))
        # H_D3
        if att in ('mlp', 'attn'):
            dom = 0 if att == 'attn' \
                else 1
            oth = 1 if att == 'attn' \
                else 0
            kld = np.median(np.concatenate(
                [RES[side]['KL'][fa][:, dom]
                 for fa in FKEYS]))
            klo = np.median(np.concatenate(
                [RES[side]['KL'][fa][:, oth]
                 for fa in FKEYS]))
            t1d = np.mean(np.concatenate(
                [RES[side]['T1'][fa][:, dom]
                 for fa in FKEYS]))
            t1o = np.mean(np.concatenate(
                [RES[side]['T1'][fa][:, oth]
                 for fa in FKEYS]))
            kls = np.median(
                np.concatenate(
                    [RES[side]['KL'][fa][:, 2]
                     for fa in FKEYS]))
            t1s = np.mean(np.concatenate(
                [RES[side]['T1'][fa][:, 2]
                 for fa in FKEYS]))
            E['hd3'][side] = bool(
                kld > klo and t1d < t1o)
            E['agg'].update({
                '%s_kl_dom' % side:
                    float(kld),
                '%s_kl_oth' % side:
                    float(klo),
                '%s_t1_dom' % side:
                    float(t1d),
                '%s_t1_oth' % side:
                    float(t1o),
                '%s_kl_skip' % side:
                    float(kls),
                '%s_t1_skip' % side:
                    float(t1s)})
            log('[%s] H_D3=%s kl_dom=%.4g '
                'kl_oth=%.4g t1_dom=%.3f '
                't1_oth=%.3f | skip '
                'kl=%.4g t1=%.3f'
                % (side, E['hd3'][side],
                   kld, klo, t1d, t1o,
                   kls, t1s))
        # stats: rr / keep / f2d meds
        for key in ('AB', 'AC', 'BC'):
            for ci in (1, 2, 3):
                m = CIDX == ci
                ck = '%s_%s_ci%d' % (
                    side, key, ci)
                for fa in key:
                    rr = np.median(
                        RES[side]['NRM'][fa][
                            m, 1]
                        / np.maximum(
                            RES[side]['NRM']
                            [fa][m, 0],
                            1e-12))
                    kp = float(np.median(
                        RES[side]['KEEP']
                        [fa][m]))
                    E['rr'][ck + '_' + fa] = \
                        float(rr)
                    E['keep'][ck + '_' + fa] \
                        = kp
                E['f2da'][ck] = \
                    float(np.median(
                        RES[side]['F2DA']
                        [key][m]))
                E['f2dm'][ck] = \
                    float(np.median(
                        RES[side]['F2DM']
                        [key][m]))
    a4 = E['attrib']['4B']
    a14 = E['attrib']['14B']
    if 'no_active' in (a4, a14):
        VERDICT = \
            'fifth_lastblock_no_active'
    elif a4 == a14:
        VERDICT = \
            'fifth_lastblock_%s_rewrite' \
            % a4
    else:
        VERDICT = \
            'fifth_lastblock_' \
            'attribution_split'
    E['decision'] = {
        'attrib_4B': a4,
        'attrib_14B': a14,
        'H_D3_4B': E['hd3'].get('4B'),
        'H_D3_14B': E['hd3'].get('14B')}
    log('decision: attrib 4B=%s 14B=%s '
        'H_D3 %s/%s -> VERDICT: %s'
        % (a4, a14, E['hd3'].get('4B'),
           E['hd3'].get('14B'), VERDICT))
    log('VERDICT: %s' % VERDICT)

# ---------- persist ----------
save = {
    'VERDICT': np.array(VERDICT),
    'SMOKE': np.bool_(SMOKE),
    'PHASE': np.int64(PHASE),
    'SEED': np.int64(SEED),
    'C4_OK': np.bool_(c4_ok)}
for side in ('4B', '14B'):
    R = RES[side]
    for tag in ('c0', 'c1', 'c2', 'c2b',
                'c3', 'c5'):
        save['%s_%s' % (tag.upper(), side)] \
            = np.float64(R['A'][tag])
    for key in ('AB', 'AC', 'BC'):
        save['F2S_%s_%s' % (side, key)] = \
            R['F2S'][key].astype(np.float64)
        save['F2I_%s_%s' % (side, key)] = \
            R['F2I'][key].astype(np.float64)
        save['F2DA_%s_%s' % (side, key)] = \
            R['F2DA'][key].astype(np.float64)
        save['F2DM_%s_%s' % (side, key)] = \
            R['F2DM'][key].astype(np.float64)
    for fa in FKEYS:
        save['NRM_%s_%s' % (side, fa)] = \
            R['NRM'][fa].astype(np.float64)
        save['KEEP_%s_%s' % (side, fa)] = \
            R['KEEP'][fa].astype(np.float64)
        save['KL_%s_%s' % (side, fa)] = \
            R['KL'][fa].astype(np.float64)
        save['T1_%s_%s' % (side, fa)] = \
            R['T1'][fa].astype(np.float64)
if not SMOKE:
    for ck, g in E['gs'].items():
        save['GMED_' + ck] = np.array(g)
npz_path = OUT + r'\%s.npz' % NAME
np.savez(npz_path, **save)

res = {
    'phase': PHASE, 'name': NAME,
    'verdict': VERDICT,
    'smoke': SMOKE,
    'forwards': len(CONS) * 8 * 4 * 3
                * 2,
    'anchors': {
        'c0_4B': RES['4B']['A']['c0'],
        'c0_14B': RES['14B']['A']['c0'],
        'c1_4B': RES['4B']['A']['c1'],
        'c1_14B': RES['14B']['A']['c1'],
        'c2_4B': RES['4B']['A']['c2'],
        'c2_14B': RES['14B']['A']['c2'],
        'c2b_4B': RES['4B']['A']['c2b'],
        'c2b_14B': RES['14B']['A']['c2b'],
        'c3_4B': RES['4B']['A']['c3'],
        'c3_14B': RES['14B']['A']['c3'],
        'c5_4B': RES['4B']['A']['c5'],
        'c5_14B': RES['14B']['A']['c5'],
        'c4_ok': bool(c4_ok),
        'ok': True}}
if not SMOKE:
    res['stats'] = {
        k: v for k, v in E.items()
        if k != 'gs'}
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
