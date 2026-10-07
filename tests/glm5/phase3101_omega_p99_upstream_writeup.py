# -*- coding: utf-8 -*-
"""Phase 3101  Omega-P99: upstream write
localization and the fair carrier test of
the 3093 focal heads (14B main arm +
4B profile control; native + per-head
swap forwards).

3100 (sixth_upstream_residual) showed:
the last-block MLP increment is near
first-order predictable (pred_cos
0.9404), its input arrives ~96.5 pct
through the h0 RESIDUAL channel, and
the L37 natural-increment top-8 heads
(post-o_proj convention) do not
overlap the 3093 causal focal TOP8.
CORRECTION FOUND IN DESIGN: the 3093
BH bank and the attn_swaps operate at
the o_proj INPUT (pre-o_proj concat,
40x128 PURE per-head blocks; capH =
hook_in_last on o_proj), while 3100 Q3
used the post-o_proj module output
(o_proj-mixed coordinate slices) - a
different coordinate object.  Q_C
below re-tests the natural shares in
the correct pre-o_proj convention.

Questions (frozen):
  Q_A (fair natural shares, 14B):
      split the L37 PRE-o_proj concat
      last position into 40x128 pure
      per-head blocks (3093 HEAD_IDX
      convention), take the natural
      Delta-BH37 L2 share top-8 per
      family, compare with the 3093
      causal focal TOP8 (H_G1).
  Q_B (swap downstream path, 14B):
      JOINT intervention, bit-identical
      to the 3093 E3 scan (smoke 5 of
      this phase exposed that 3093
      CS1H is NOT a pure head swap):
      per pair k first build repV37 =
      base own L37 V (full sequence)
      with positions [0, FRONT=4)
      replaced by the prefix V at
      [off, off+FRONT) (3093 repv_of),
      inject via the L37 v_proj hook
      (all positions); then for EVERY
      head h in 0..39, replace its L37
      pre-o_proj last-position block by
      the BASE natural value (3093
      attn_swaps: identity-restoring
      swap; 24 pairs x 3 families =
      2880 swap forwards) and measure
      cos(swap-induced increment,
      natural increment) at L39 h0
      (= hidden_states[39]), last-block
      MLP out m39, and last-block attn
      out a39.  Focal (TOP8) vs
      non-focal split (H_G2).
      The recomputed CS1H cos matrix
      must reproduce the 3093 sealed
      CS1H (d2 anchor - end-to-end
      validation of the joint
      intervention + LG + TT chain);
      the pure-repV E1 ladder must
      reproduce sealed COS_LAD / MED_C
      (d2b anchor - validates the repV
      injection machinery separately).
      NOTE (semantics, decisive probe
      2026-09-23): the 3093 V
      intervention replaces the FULL V
      stream at ALL layers with the
      base natural bank V; only L37
      rows [0,FRONT) carry prefix V.
      This is NOT equivalent to an
      L37-only replacement: downstream
      layers' V would otherwise
      recompute from the modified
      state.  The full-stream form
      makes the intervention exactly
      localized (downstream V =
      natural) and reproduces the
      sealed COS_LAD to 1e-12; the
      L37-only form deviates by up to
      0.077 in cos.
  Q_C (write localization, both
      arms): per-layer profile of the
      natural conditional increment -
      ||Delta-h2(l)|| over all blocks,
      plus per-layer Delta-a / Delta-m
      / Delta-h0 norms; late-write
      share of the last 3 blocks
      (H_G3, descriptive).

Pipeline:  14B arm first: 96 native
forwards (banks LG / TT / BH_PRE /
VB37 = L37 v_proj output full
sequence / hs38 / hs39 / a39 / m39 +
per-layer profile), then per family
1 V self-replacement identity
forward + E1 pure-repV ladder + 2880
joint swap forwards (for (family,
head, k): base text, repV37 V
injection at L37 + stateATN[37]
replaces HEAD_IDX[h] block at o_proj
input by the BASE natural value;
record lg, swap-effective block,
V-effective rows, hs38/hs39/a39/
m39).  4B arm: 96 native forwards,
profile only.  All statistics f64
CPU.

Anchors (frozen):
  d1: swap-effective anchor - the
      captured L37 pre-o_proj block
      at the swapped coordinates
      equals the replacement value
      (BASE natural), bit 0 (every
      swap forward).
  d1b: V self-replacement identity
      - forward with L37 V replaced
      by base own V reproduces the
      native logits bit 0 (one per
      family; validates the L37-only
      replacement mechanism).
  d1v: V-injection effective - the
      captured post-hook L37 V rows
      equal repV37 bit 0 (every swap
      forward).
  d2: CS1H_NEW == 3093 sealed
      CS1H[fa] (max abs diff; bit 0
      expected, tolerance 1e-6).
  d2b: COS_LAD_NEW == 3093 sealed
      COS_LAD[fa] and (formal)
      med_c_new == MED_C[fa]
      (tolerance 1e-9).
  d3: LG bank == 3093 sealed LG
      (lg bit 0 expected, tol 1e-3;
      TT rows tol 1e-3).
  d4: 3093 npz sha8 == seal.json;
      3098 npz sha8 == eeba6c18
      (inputs untampered).
  d5: head(norm(h2_LB)) vs native
      logits, bit 0 (native readout
      path).
  d6: determinism - 2 re-run
      prompts reproduce lg / hs39
      bit 0.

Gates (frozen):
  H_G1 (fair natural carrier):
      median over the 3 families of
      overlap(top8_nat_pre, TOP8_3093)
      >= 3 (of 8).
  H_G2 (focal path coupling): median
      over families of
      focal_med/nonfocal_med >= 2,
      where the medians are over
      (head in group, 24 pairs) of
      PATH_H0.  (PATH_M / PATH_A
      recorded, not gating.)
  H_G3 (late write): median over
      pooled per-pair last-3-block
      share of sum_l ||Delta-h2(l)||
      >= 0.5 (both arms pooled;
      descriptive).
verdict ladder (frozen, 14B main):
  setup failure -> seventh_setup_
  failed;  H_G1 and H_G2 ->
  seventh_carrier_coupled;  H_G1
  and not H_G2 -> seventh_share_
  only;  not H_G1 and H_G2 ->
  seventh_path_only;  neither ->
  seventh_carrier_absent.
  SMOKE -> seventh_smoke.
  4B sub-verdict: profile only ->
  seventh_profile_only.

SMOKE: CONS 0..1, first 4 pairs per
family, head subset {TOP8_A[0],
TOP8_B[0], 0, 39}, verdict
seventh_smoke; d2 on the subset.

Limitations (recorded): single last
position; swap at one layer (L37)
only; PATH cosines measure direction
alignment, not magnitude recovery;
the natural increment includes all
prefix effects (the diff/swap
asymmetry noted in 3100 remains,
but Q_A now uses the SAME coordinate
object as 3093); causal-connective
paradigm; H_G3 threshold descriptive.

Memory discipline: swap loop is
transient per (family, head); profile
hooks gated by a global switch; 4B
frees before 14B; no quantization.
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

PHASE = 3101
NAME = 'omega_p99_upstream_writeup'
SEED = 3101
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
OUT = (R13 + r'\phase3101' + '\\' + NAME)
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
CONS = [0, 1] if SMOKE else [0, 1, 2, 3]
NP_ = 8 if SMOKE else 24
NCI = len(CONS) - 1
NK = 4 if SMOKE else NP_

ARM = {
    '4B': {'mdir': os.path.join(
               ROOT, 'models', 'hf',
               'qwen3-4b'),
           'nl': 36, 'hid': 2560,
           'nq': 32, 'hd': 128,
           'l_inj': None, 'swap': False},
    '14B': {'mdir': os.path.join(
                ROOT, 'models', 'hf',
                'Qwen3-14B'),
            'nl': 40, 'hid': 5120,
            'nq': 40, 'hd': 128,
            'l_inj': 37, 'swap': True}}

GATES = {
    'H_G1_overlap_ge': 3,
    'H_G2_ratio_ge': 2.0,
    'H_G3_late_ge': 0.5,
    'TOPN': 8,
    'D2_TOL': 1e-6,
    'D2B_TOL': 1e-9,
    'D3_TOL': 1e-3}

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
    'question': ('are the 3093 causal '
                 'focal heads the natural '
                 'carriers in the CORRECT '
                 'pre-o_proj per-head '
                 'convention; does a focal '
                 'head swap propagate along '
                 'the natural downstream '
                 'path; which layers write '
                 'the conditional increment'),
    'correction': ('3093 BH/attn_swaps '
                   'operate at the o_proj '
                   'INPUT (pre-o_proj concat '
                   'pure per-head blocks); '
                   '3100 Q3 used post-o_proj '
                   'slices - Q_A re-tests '
                   'natural shares in the '
                   'correct convention.  '
                   '3093 CS1H is the JOINT '
                   'repV V-injection + '
                   'single-head identity-'
                   'restoring swap recovery '
                   'spectrum, not a pure '
                   'head swap; the swap '
                   'value is the BASE '
                   'natural block.  3101 '
                   'reproduces the joint '
                   'intervention (repV37: '
                   'L37 V = base own V with '
                   'first FRONT=4 positions '
                   'from prefix V)'),
    'pipeline': {
        'native': ('96 forwards/arm: LG, '
                   'TT, BH_PRE (L37 pre-o_'
                   'proj), VB37 (L37 v_proj '
                   'out full seq), hs38/hs39/'
                   'a39/m39, per-layer profile',
                   ),
        'swap': ('joint 3093-E3 forwards '
                 '(14B): per pair FULL V-'
                 'stream replacement (all '
                 'layers = base natural '
                 'bank V; L37 rows '
                 '[0,FRONT) = prefix V) '
                 '+ every head h: L37 '
                 'o_proj input block '
                 'HEAD_IDX[h] replaced by '
                 'BASE natural value; '
                 'record lg + path '
                 'increments; plus per-'
                 'family V self-replacement '
                 'identity forward and E1 '
                 'pure-repV ladder '
                 '(COS_LAD_NEW)'),
        'profile': ('per-layer Delta-h2 '
                    'norms + Delta-a/Delta-m/'
                    'Delta-h0 norms; last-3 '
                    'share')},
    'gates': GATES,
    'anchors': {
        'd1': 'head-swap effective block '
              'bit 0 (base value)',
        'd1b': 'V self-replacement '
               'identity bit 0',
        'd1v': 'V-injection effective '
               'rows bit 0',
        'd2': 'CS1H_NEW == 3093 sealed '
              '(tol 1e-6)',
        'd2b': 'COS_LAD_NEW/MED_C == '
               '3093 sealed (tol 1e-9)',
        'd3': 'LG/TT vs 3093 sealed '
              '(lg bit0, tol 1e-3)',
        'd4': '3098 eeba6c18; 3093 seal',
        'd5': 'head(norm(h2)) bit 0',
        'd6': 'determinism re-run bit 0'},
    'inputs': {
        'p3098_npz8': h8(P98),
        'p3093_npz8': h8(P93)},
    'models': {k: {'mdir': v['mdir'],
                   'nl': v['nl'],
                   'hid': v['hid'],
                   'nq': v['nq'],
                   'l_inj': v['l_inj'],
                   'swap': v['swap']}
               for k, v in ARM.items()},
    'forwards': {
        '4B': len(CONS) * 8 * 3,
        '14B': (len(CONS) * 8 * 3
                + (NK if SMOKE else NP_)
                * 3 + 3
                + (4 if SMOKE else 40)
                * NK * 3)},
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


# global profile switch + capture
PROF = {'on': False}
SWAP_ST = {'repl': None, 'mask': None}
# 3093-identical V-injection state:
# per-layer repl/mask (full V-stream
# replacement) + per-layer capture
STV = {}
CAPV = {}
V_EFF = {'on': False, 'v': None}


def mk_last(store, tag):
    def hook(module, args, output):
        t = output[0] \
            if isinstance(output, tuple) \
            else output
        store[tag] = t[0, -1].detach()
    return hook


def mk_block_last(store, tag):
    def hook(module, args, output):
        t = output[0] \
            if isinstance(output, tuple) \
            else output
        if PROF['on']:
            store.setdefault(
                tag, []).append(
                t[0, -1].detach())
        else:
            store[tag] = t[0, -1].detach()
    return hook


def mk_in_last(store, tag):
    # capture the o_proj INPUT last
    # position (pre-o_proj concat).
    def hook(module, args, output):
        store[tag] = args[0][0, -1] \
            .detach().clone()
    return hook


def mk_swap_pre():
    # 3093 hook_pre_last: replace the
    # o_proj input last position at the
    # masked coordinates.
    def hook(module, args):
        if SWAP_ST['repl'] is None:
            return None
        a = args[0].clone()
        a[0, -1, SWAP_ST['mask']] = \
            SWAP_ST['repl']
        return (a,)
    return hook


def mk_v_hook(cap, st):
    # 3093 hook_v: capture the NATURAL
    # v_proj output (before replacement)
    # and/or replace rows by st['repl'].
    def hook(module, inp, out):
        if cap['rec']:
            cap['orig'] = out[0] \
                .detach().clone()
        if st['repl'] is not None:
            out[0][st['mask']] = st['repl']
        return out
    return hook


def mk_v_eff():
    # registered AFTER mk_v_hook on the
    # same module: sees the post-
    # replacement v_proj output.
    def hook(module, inp, out):
        if V_EFF['on']:
            V_EFF['v'] = out[0] \
                .detach().clone()
        return out
    return hook


def run_arm(side):
    cfg = ARM[side]
    NL = cfg['nl']
    HID = cfg['hid']
    NQ = cfg['nq']
    HD = cfg['hd']
    NQW = NQ * HD
    L_INJ = cfg['l_inj']
    DO_SWAP = cfg['swap']
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
    assert int(model.config.vocab_size) \
        == 151936
    norm_mod = model.model.norm
    head = model.lm_head
    A = {'d1': 0.0, 'd1b': 0.0,
         'd1v': 0.0, 'd2': 0.0,
         'd2b': 0.0, 'd3a': 0.0,
         'd3b': 0.0, 'd5': 0.0,
         'd6': 0.0}
    # constant hooks (installed once)
    prof_h2 = {'v': []}
    prof_a = {'v': []}
    prof_m = {'v': []}
    hs_store = {}
    cap_in = {}
    handles = []
    for li in range(NL):
        handles.append(
            layers[li]
            .register_forward_hook(
                mk_block_last(prof_h2,
                              'l%d' % li)))
        handles.append(
            layers[li].self_attn
            .register_forward_hook(
                mk_block_last(prof_a,
                              'a%d' % li)))
        handles.append(
            layers[li].mlp
            .register_forward_hook(
                mk_block_last(prof_m,
                              'm%d' % li)))
    handles.append(
        layers[NL - 1]
        .register_forward_hook(
            mk_last(hs_store, 'h2lb')))
    if L_INJ is not None:
        KVW = int(model.config
                  .num_key_value_heads) * HD
        assert KVW == 1024, KVW
        STV.clear()
        CAPV.clear()
        for li in range(NL):
            STV[li] = {'repl': None,
                       'mask': None}
            CAPV[li] = {'rec': False,
                        'orig': None}
            handles.append(
                layers[li].self_attn.v_proj
                .register_forward_hook(
                    mk_v_hook(CAPV[li],
                              STV[li])))
        handles.append(
            layers[L_INJ].self_attn.v_proj
            .register_forward_hook(
                mk_v_eff()))
        handles.append(
            layers[L_INJ].self_attn.o_proj
            .register_forward_hook(
                mk_in_last(cap_in, 'bhin')))
        handles.append(
            layers[L_INJ].self_attn.o_proj
            .register_forward_pre_hook(
                mk_swap_pre()))
    log('[%s] hooks installed (%d)'
        % (side, len(handles)))
    # ---- native pass ----
    LG = {}
    TT = {}
    BH_PRE = {}
    VB37 = {}
    VBF = {}
    HS39 = {}
    A39 = {}
    M39 = {}
    H2LB = {}
    PROF_H2 = {}
    PROF_A = {}
    PROF_M = {}
    for fkey in FKEYS:
        assembled, idx_of, cidx, bidx = \
            assemble(fkey, tok)
        n_pr = len(assembled)
        lg = np.zeros((n_pr, 151936),
                      dtype=np.float32)
        bhp = np.zeros((n_pr, NQW),
                       dtype=np.float32) \
            if L_INJ is not None else None
        vfull = None
        if L_INJ is not None:
            lens_arr = np.array(
                [len(it['ids'])
                 for it in assembled])
            nmax = int(lens_arr.max())
            vfull = np.zeros(
                (n_pr, NL, nmax, KVW),
                dtype=np.float32)
        hs39 = np.zeros((n_pr, HID),
                        dtype=np.float32)
        a39 = np.zeros((n_pr, HID),
                       dtype=np.float32)
        m39 = np.zeros((n_pr, HID),
                       dtype=np.float32)
        h2lb = np.zeros((n_pr, HID),
                        dtype=np.float32)
        ph2 = np.zeros((n_pr, NL, HID),
                       dtype=np.float32)
        pa = np.zeros((n_pr, NL, HID),
                      dtype=np.float32)
        pm = np.zeros((n_pr, NL, HID),
                      dtype=np.float32)
        PROF['on'] = True
        for i, it in enumerate(assembled):
            for k in list(prof_h2):
                prof_h2[k] = []
            for k in list(prof_a):
                prof_a[k] = []
            for k in list(prof_m):
                prof_m[k] = []
            ids_t = torch.tensor(
                [it['ids']], device='cuda')
            if bhp is not None:
                for li in range(NL):
                    CAPV[li]['rec'] = True
            with torch.no_grad():
                out = model(
                    ids_t, use_cache=False,
                    output_hidden_states=False)
            lg[i] = out.logits[0, -1] \
                .detach().float() \
                .cpu().numpy()
            del out
            if bhp is not None:
                n_i = len(it['ids'])
                for li in range(NL):
                    CAPV[li]['rec'] = False
                    vfull[i, li, :n_i, :] = \
                        CAPV[li]['orig'] \
                        .float().cpu() \
                        .numpy()
                    CAPV[li]['orig'] = None
                bhp[i] = cap_in['bhin'] \
                    .float().cpu().numpy()
            hs39[i] = prof_h2['l%d'
                              % (NL - 2)][0] \
                .float().cpu().numpy()
            a39[i] = prof_a['a%d'
                            % (NL - 1)][0] \
                .float().cpu().numpy()
            m39[i] = prof_m['m%d'
                            % (NL - 1)][0] \
                .float().cpu().numpy()
            h2lb[i] = hs_store['h2lb'] \
                .float().cpu().numpy()
            for li in range(NL):
                ph2[i, li] = \
                    prof_h2['l%d' % li][0] \
                    .float().cpu().numpy()
                pa[i, li] = \
                    prof_a['a%d' % li][0] \
                    .float().cpu().numpy()
                pm[i, li] = \
                    prof_m['m%d' % li][0] \
                    .float().cpu().numpy()
            if i % 8 == 7:
                log('[%s][%s] native %d/1cfg'
                    % (side, fkey, i + 1))
                flush_log()
        PROF['on'] = False
        # d5 readout anchor
        with torch.no_grad():
            z5 = head(norm_mod(
                torch.from_numpy(h2lb)
                .to('cuda')
                .to(torch.bfloat16)))
        A['d5'] = max(A['d5'], float(
            np.max(np.abs(
                z5.double().cpu().numpy()
                - lg.astype(np.float64)))))
        del z5
        # d3: LG/TT vs 3093 sealed
        # (formal only: identical 32-row
        # bi-major order; SMOKE has a
        # 16-row different-order bank).
        # TT rows are recomputed for the
        # swap pass in BOTH modes.
        if L_INJ is not None:
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
            if not SMOKE:
                lg93 = z93['LG_' + fkey] \
                    .astype(np.float64)
                d3a = float(np.max(np.abs(
                    lg.astype(np.float64)
                    - lg93)))
                tt93 = z93['TT_' + fkey] \
                    .astype(np.float64)
                d3b = float(np.max(np.abs(
                    ttr - tt93)))
                A['d3a'] = max(A['d3a'],
                               d3a)
                A['d3b'] = max(A['d3b'],
                               d3b)
                log('[%s][%s] d3 lg_diff='
                    '%.3e tt_diff=%.3e'
                    % (side, fkey, d3a,
                       d3b))
                del lg93, tt93
            TT[fkey] = ttr
        else:
            TT[fkey] = None
        LG[fkey] = lg
        BH_PRE[fkey] = bhp
        VBF[fkey] = vfull
        # L37 slice kept for persist
        if vfull is not None:
            VB37[fkey] = vfull[:, L_INJ] \
                .copy()
        HS39[fkey] = hs39
        A39[fkey] = a39
        M39[fkey] = m39
        H2LB[fkey] = h2lb
        PROF_H2[fkey] = ph2
        PROF_A[fkey] = pa
        PROF_M[fkey] = pm
        log('[%s][%s] native done'
            % (side, fkey))
        flush_log()
    # d6 determinism: re-run 2 prompts
    fkey0 = 'A'
    assembled0, _, _, _ = assemble(
        fkey0, tok)
    for si in (0, 9):
        ids_t = torch.tensor(
            [assembled0[si]['ids']],
            device='cuda')
        with torch.no_grad():
            out = model(ids_t,
                        use_cache=False)
        lg2 = out.logits[0, -1].detach() \
            .float().cpu().numpy()
        del out
        A['d6'] = max(A['d6'], float(
            np.max(np.abs(
                lg2.astype(np.float64)
                - LG[fkey0][si]
                .astype(np.float64)))))
    log('[%s] d6 determinism=%.3e'
        % (side, A['d6']))
    # ---- layer profile (Q_C) ----
    NL_prof = NL
    late_share = []
    am_norm = np.zeros((len(FKEYS),
                        NL_prof, 3))
    for fi, fkey in enumerate(FKEYS):
        ph2 = PROF_H2[fkey]
        pa = PROF_A[fkey]
        pm = PROF_M[fkey]
        idx_of = {}
        asm, idx_of, cidx, bidx = \
            assemble(fkey, tok)
        for k in range(NP_):
            crow = idx_of[(int(cidx[k]),
                           int(bidx[k]))]
            brow = idx_of[(0,
                           int(bidx[k]))]
            dh2 = np.linalg.norm(
                ph2[crow].astype(np.float64)
                - ph2[brow].astype(
                    np.float64), axis=1)
            # dh2[li] = block li write;
            # late = last 3 blocks
            tot = float(dh2.sum())
            late = float(
                dh2[NL_prof - 3:].sum())
            if tot > 0:
                late_share.append(
                    late / tot)
            da_n = np.linalg.norm(
                pa[crow].astype(np.float64)
                - pa[brow].astype(
                    np.float64), axis=1)
            dm_n = np.linalg.norm(
                pm[crow].astype(np.float64)
                - pm[brow].astype(
                    np.float64), axis=1)
            am_norm[fi, :, 0] += da_n / NP_
            am_norm[fi, :, 1] += dm_n / NP_
            am_norm[fi, :, 2] += dh2 / NP_
    late_med = float(np.median(
        late_share)) if late_share else 0.0
    H_G3 = bool(late_med
                >= GATES['H_G3_late_ge'])
    log('[%s] H_G3 late-3 share=%.4f '
        '(n=%d) -> %s'
        % (side, late_med,
           len(late_share), H_G3))
    am_norm /= len(FKEYS)
    # ---- swap pass (14B) ----
    CS1H_NEW = {}
    PATH_H0 = {}
    PATH_M = {}
    PATH_A = {}
    T8N_PRE = {}
    OVL_PRE = {}
    BHSH_PRE = {}
    COS_LAD_NEW = {}
    MED_C_NEW = {}
    R1_NH_NEW = {}
    T8_R1_NEW = {}
    if DO_SWAP:
        FRONT = 4  # 3093 constant
        ridx = np.arange(FRONT)
        HD_IDX = [np.arange(h * HD,
                            (h + 1) * HD)
                  for h in range(NQ)]
        if SMOKE:
            head_list = sorted(set(
                [int(z93['TOP8_A'][0]),
                 int(z93['TOP8_B'][0]),
                 0, 39]))
        else:
            head_list = list(range(NQ))
        NH_RUN = len(head_list)

        def repv_of(k, assembled,
                    idx_of, cidx, bidx,
                    fkey, lens_arr):
            # 3093 repv_of EXACT: the FULL
            # V stream (all layers, all
            # base positions) is replaced
            # by the base natural bank V,
            # EXCEPT L_INJ rows [0,FRONT)
            # which carry the prefix V at
            # [off, off+FRONT).  The
            # intervention is therefore
            # exactly localized: downstream
            # V stays base-natural.
            # (Probe 2026-09-23: this
            # reproduces the 3093 sealed
            # COS_LAD to 1e-12; an L37-only
            # replacement does NOT - the
            # downstream V would recompute
            # from the modified state.)
            b_ = int(bidx[k])
            c_ = int(cidx[k])
            base_i = idx_of[(0, b_)]
            pref_i = idx_of[(c_, b_)]
            nb = int(lens_arr[base_i])
            off = int(assembled[
                pref_i]['off'])
            assert nb >= FRONT, \
                (fkey, k, nb)
            repV = VBF[fkey][base_i,
                             :, :nb, :] \
                .copy()
            repV[L_INJ, ridx, :] = \
                VBF[fkey][pref_i,
                          L_INJ,
                          off + ridx, :]
            return base_i, pref_i, repV

        def run_fwd(ids, repV,
                    sw_repl=None,
                    sw_mask=None):
            # 3093 forward_gen semantics:
            # every layer's v_proj output
            # replaced by repV[li] (all
            # positions); optional head
            # swap at the L_INJ o_proj
            # input.  Returns lg f32 and
            # the post-replacement L37 V
            # rows (nb, KVW) f32.
            nb = repV.shape[1]
            rt = torch.tensor(
                np.ascontiguousarray(repV),
                dtype=torch.bfloat16,
                device='cuda')
            m_all = torch.ones(
                nb, dtype=torch.bool,
                device='cuda')
            for li in range(NL):
                STV[li]['repl'] = rt[li]
                STV[li]['mask'] = m_all
            V_EFF['on'] = True
            if sw_repl is not None:
                SWAP_ST['repl'] = sw_repl
                SWAP_ST['mask'] = sw_mask
            ids_t = torch.tensor(
                [ids], device='cuda')
            with torch.no_grad():
                out = model(
                    ids_t, use_cache=False)
            lgw = out.logits[0, -1] \
                .detach().float() \
                .cpu().numpy()
            del out
            veff = V_EFF['v'].float() \
                .cpu().numpy()
            V_EFF['on'] = False
            V_EFF['v'] = None
            for li in range(NL):
                STV[li]['repl'] = None
                STV[li]['mask'] = None
            SWAP_ST['repl'] = None
            SWAP_ST['mask'] = None
            return lgw, veff

        for fkey in FKEYS:
            assembled, idx_of, cidx, \
                bidx = assemble(fkey, tok)
            lens_arr = np.array(
                [len(it['ids'])
                 for it in assembled])
            cs1 = np.full((NQ, NP_),
                          np.nan)
            ph0 = np.full((NQ, NP_),
                          np.nan)
            pm_ = np.full((NQ, NP_),
                          np.nan)
            pa_ = np.full((NQ, NP_),
                          np.nan)
            # d1b: V self-replacement
            # identity (3093 b1 anchor,
            # full V stream; bit 0)
            b0_i = idx_of[(0, int(
                bidx[0]))]
            nb0 = int(lens_arr[b0_i])
            selfv = VBF[fkey][b0_i, :,
                              :nb0, :] \
                .copy()
            lg_s, veff_s = run_fwd(
                assembled[b0_i]['ids'],
                selfv)
            d1bv = float(np.max(np.abs(
                lg_s.astype(np.float64)
                - LG[fkey][b0_i].astype(
                    np.float64))))
            A['d1b'] = max(A['d1b'], d1bv)
            d1vs = float(np.max(np.abs(
                veff_s
                - selfv[L_INJ])))
            A['d1v'] = max(A['d1v'], d1vs)
            log('[%s][%s] d1b self-V '
                'identity diff=%.3e '
                '(veff %.3e)'
                % (side, fkey, d1bv,
                   d1vs))
            # E1: pure-repV ladder
            # (3093 E1; d2b vs sealed
            # COS_LAD / MED_C)
            n_e1 = NK if SMOKE else NP_
            clad = np.full(NP_, np.nan)
            for k in range(n_e1):
                base_i, pref_i, repV = \
                    repv_of(k, assembled,
                            idx_of, cidx,
                            bidx, fkey,
                            lens_arr)
                lgw, veff_k = run_fwd(
                    assembled[base_i]['ids'],
                    repV)
                d1vk = float(np.max(
                    np.abs(veff_k
                           - repV[L_INJ])))
                A['d1v'] = max(A['d1v'],
                               d1vk)
                dlg = lgw.astype(
                    np.float64) \
                    - LG[fkey][base_i] \
                    .astype(np.float64)
                clad[k] = cosv(
                    dlg, TT[fkey][k])
            n_use = int(np.sum(
                ~np.isnan(clad)))
            clad93 = z93['COS_LAD_'
                         + fkey] \
                .astype(np.float64)
            d2bv = float(np.max(np.abs(
                clad[:n_use]
                - clad93[:n_use])))
            A['d2b'] = max(A['d2b'], d2bv)
            med_c_new = float(
                np.nanmedian(clad))
            if not SMOKE:
                mc93 = float(
                    z93['MED_C_' + fkey])
                d2bm = abs(med_c_new
                           - mc93)
                A['d2b'] = max(A['d2b'],
                               d2bm)
            COS_LAD_NEW[fkey] = clad
            MED_C_NEW[fkey] = med_c_new
            log('[%s][%s] d2b cos_lad '
                'diff=%.3e med_c_new='
                '%.4f'
                % (side, fkey, d2bv,
                   med_c_new))
            del clad93
            for hi, h in enumerate(
                    head_list):
                blk_idx = HD_IDX[h]
                for k in range(NK):
                    base_i, pref_i, \
                        repV = repv_of(
                            k, assembled,
                            idx_of, cidx,
                            bidx, fkey,
                            lens_arr)
                    # identity-restoring
                    # head swap (3093
                    # attn_swaps: BASE
                    # natural block)
                    sw_repl = torch.tensor(
                        BH_PRE[fkey][
                            base_i,
                            blk_idx],
                        dtype=torch.bfloat16,
                        device='cuda')
                    sw_mask = torch.tensor(
                        blk_idx,
                        dtype=torch.long,
                        device='cuda')
                    lgw, veff_k = run_fwd(
                        assembled[base_i]
                        ['ids'], repV,
                        sw_repl=sw_repl,
                        sw_mask=sw_mask)
                    d1vk = float(np.max(
                        np.abs(veff_k
                               - repV[
                                   L_INJ])))
                    A['d1v'] = max(A['d1v'],
                                   d1vk)
                    # d1 swap-effective
                    eff = cap_in['bhin'] \
                        .float().cpu() \
                        .numpy()
                    d1v_ = float(np.max(
                        np.abs(
                            eff[blk_idx]
                            - BH_PRE[fkey][
                                base_i,
                                blk_idx] \
                            .astype(
                                np.float32))))
                    A['d1'] = max(A['d1'],
                                  d1v_)
                    crow = pref_i
                    brow = base_i
                    cs1[h, k] = cosv(
                        lgw.astype(
                            np.float64)
                        - LG[fkey][brow]
                        .astype(np.float64),
                        TT[fkey][k])
                    dh_nat = HS39[fkey][crow] \
                        .astype(np.float64) \
                        - HS39[fkey][brow] \
                        .astype(np.float64)
                    # PROF off: single-tensor
                    # stores this forward
                    hs39_w = prof_h2[
                        'l%d' % (NL - 2)] \
                        .float() \
                        .cpu().numpy()
                    ph0[h, k] = cosv(
                        hs39_w.astype(
                            np.float64)
                        - HS39[fkey][brow]
                        .astype(np.float64),
                        dh_nat)
                    m39_w = prof_m[
                        'm%d' % (NL - 1)] \
                        .float() \
                        .cpu().numpy()
                    dm_nat = M39[fkey][crow] \
                        .astype(np.float64) \
                        - M39[fkey][brow] \
                        .astype(np.float64)
                    pm_[h, k] = cosv(
                        m39_w.astype(
                            np.float64)
                        - M39[fkey][brow]
                        .astype(np.float64),
                        dm_nat)
                    a39_w = prof_a[
                        'a%d' % (NL - 1)] \
                        .float() \
                        .cpu().numpy()
                    da_nat = A39[fkey][crow] \
                        .astype(np.float64) \
                        - A39[fkey][brow] \
                        .astype(np.float64)
                    pa_[h, k] = cosv(
                        a39_w.astype(
                            np.float64)
                        - A39[fkey][brow]
                        .astype(np.float64),
                        da_nat)
                log('[%s][%s] swap heads '
                    '%d/%d done (d1=%.1e)'
                    % (side, fkey, hi + 1,
                       NH_RUN, A['d1']))
                flush_log()
            CS1H_NEW[fkey] = cs1
            PATH_H0[fkey] = ph0
            PATH_M[fkey] = pm_
            PATH_A[fkey] = pa_
            # r1_nh_new (3093 r1_nh
            # convention: per-head median
            # over pairs minus med_c)
            hh = np.array(head_list)
            r1n = np.full(NQ, np.nan)
            r1n[hh] = np.nanmedian(
                cs1[hh, :NK], axis=1) \
                - med_c_new
            R1_NH_NEW[fkey] = r1n
            if not SMOKE:
                t8r = np.argsort(
                    -r1n, kind='stable'
                    )[:8]
                T8_R1_NEW[fkey] = t8r \
                    .astype(np.int64)
                t893r = z93['TOP8_'
                            + fkey] \
                    .astype(np.int64)
                ovlr = len(set(int(x)
                               for x in t8r)
                           & set(int(x)
                                 for x in
                                 t893r))
                log('[%s][%s] TOP8_R1 '
                    'sanity overlap vs '
                    '3093=%d/8'
                    % (side, fkey, ovlr))
                del t893r
            # d2: CS1H_NEW vs 3093 sealed
            cs193 = z93['CS1H_' + fkey] \
                .astype(np.float64)
            if SMOKE:
                sub = np.array(
                    [cs193[h, :NK]
                     for h in head_list])
                d2v = float(np.max(
                    np.abs(
                        cs1[head_list, :NK]
                        - sub)))
            else:
                d2v = float(np.max(
                    np.abs(cs1 - cs193)))
            A['d2'] = max(A['d2'], d2v)
            log('[%s][%s] d2 cs1h_diff='
                '%.3e' % (side, fkey, d2v))
            del cs193
            # Q_A: pre-o_proj natural
            # shares
            bhp = BH_PRE[fkey]
            sh24 = np.zeros((NP_, NQ))
            for k in range(NP_):
                crow = idx_of[(int(
                    cidx[k]),
                    int(bidx[k]))]
                brow = idx_of[(0,
                    int(bidx[k]))]
                dbh = bhp[crow] \
                    .astype(np.float64) \
                    - bhp[brow] \
                    .astype(np.float64)
                nh = np.sqrt(
                    (dbh.reshape(NQ, HD)
                     ** 2).sum(axis=1))
                ss = float(nh.sum())
                if ss > 0:
                    sh24[k] = nh / ss
            BHSH_PRE[fkey] = sh24
            sh_med = np.median(sh24,
                               axis=0)
            top = np.argsort(-sh_med,
                             kind='stable'
                             )[:GATES['TOPN']]
            T8N_PRE[fkey] = top \
                .astype(np.int64)
            t893 = z93['TOP8_' + fkey] \
                .astype(np.int64)
            OVL_PRE[fkey] = len(set(
                int(x) for x in top)
                & set(int(x)
                      for x in t893))
            log('[%s][%s] Q_A top8_nat_pre='
                '%s top8_3093=%s overlap='
                '%d'
                % (side, fkey,
                   list(map(int, top)),
                   list(map(int, t893)),
                   OVL_PRE[fkey]))
            del sh24, sh_med, top, t893
            gc.collect()
            torch.cuda.empty_cache()
    # focal vs non-focal (H_G2)
    H_G2 = False
    ratios = []
    if DO_SWAP and not SMOKE:
        for fkey in FKEYS:
            t893 = z93['TOP8_' + fkey] \
                .astype(np.int64)
            ph0 = PATH_H0[fkey]
            vals_f = []
            vals_n = []
            for h in range(NQ):
                row = ph0[h, :NK] \
                    if SMOKE else ph0[h]
                row = row[
                    ~np.isnan(row)]
                if row.size == 0:
                    continue
                if h in set(int(x)
                            for x in t893):
                    vals_f.extend(row)
                else:
                    vals_n.extend(row)
            mf = float(np.median(vals_f)) \
                if vals_f else 0.0
            mn = float(np.median(vals_n)) \
                if vals_n else 0.0
            r = (mf / mn) if abs(mn) > 1e-9 \
                else 0.0
            ratios.append(r)
            log('[%s][%s] H_G2 focal='
                '%.4f nonfocal=%.4f '
                'ratio=%.3f'
                % (side, fkey, mf, mn, r))
            del t893
        H_G2 = bool(np.median(ratios)
                    >= GATES['H_G2_ratio_ge'])
    H_G1 = False
    if DO_SWAP and not SMOKE:
        ovl = [OVL_PRE[fa]
               for fa in FKEYS]
        H_G1 = bool(np.median(ovl)
                    >= GATES['H_G1_overlap_ge'])
        log('[%s] H_G1 overlaps=%s -> %s'
            % (side, ovl, H_G1))
    flush_log()
    # free (ret holds the references;
    # deleting the local names does not
    # affect the returned dict)
    ret = {'A': A, 'late_med': late_med,
           'H_G3': H_G3,
           'am_norm': am_norm,
           'H_G1': H_G1, 'H_G2': H_G2,
           'ratios': ratios,
           'OVL_PRE': OVL_PRE,
           'T8N_PRE': T8N_PRE,
           'BHSH_PRE': BHSH_PRE,
           'CS1H_NEW': CS1H_NEW,
           'PATH_H0': PATH_H0,
           'PATH_M': PATH_M,
           'PATH_A': PATH_A,
           'COS_LAD_NEW': COS_LAD_NEW,
           'MED_C_NEW': MED_C_NEW,
           'R1_NH_NEW': R1_NH_NEW,
           'T8_R1_NEW': T8_R1_NEW,
           'VB37': VB37,
           'NL': NL}
    del LG, TT, BH_PRE, HS39, A39, M39
    del H2LB, PROF_H2, PROF_A, PROF_M
    if L_INJ is not None:
        del VBF
    if DO_SWAP:
        del CS1H_NEW, PATH_H0, PATH_M
        del PATH_A
    del model, tok, norm_mod, head
    del layers
    gc.collect()
    torch.cuda.empty_cache()
    log('[%s] arm done, memory freed'
        % side)
    return ret


RES = {}
RES['4B'] = run_arm('4B')
RES['14B'] = run_arm('14B')
flush_log()

# ==== verdict ====
setup_ok = bool(
    d4a and d4b and d4c
    and RES['4B']['A']['d5'] == 0.0
    and RES['4B']['A']['d6'] == 0.0
    and RES['14B']['A']['d1'] == 0.0
    and RES['14B']['A']['d1b'] == 0.0
    and RES['14B']['A']['d1v'] == 0.0
    and RES['14B']['A']['d2']
    <= GATES['D2_TOL']
    and RES['14B']['A']['d2b']
    <= GATES['D2B_TOL']
    and RES['14B']['A']['d3a'] == 0.0
    and RES['14B']['A']['d3b']
    <= GATES['D3_TOL']
    and RES['14B']['A']['d5'] == 0.0
    and RES['14B']['A']['d6'] == 0.0)
log('setup_ok=%s' % setup_ok)
if SMOKE:
    verdict = 'seventh_smoke'
elif not setup_ok:
    verdict = 'seventh_setup_failed'
else:
    g1 = RES['14B']['H_G1']
    g2 = RES['14B']['H_G2']
    if g1 and g2:
        verdict = 'seventh_carrier_coupled'
    elif g1:
        verdict = 'seventh_share_only'
    elif g2:
        verdict = 'seventh_path_only'
    else:
        verdict = 'seventh_carrier_absent'
verdict_4b = 'seventh_profile_only'
log('VERDICT: %s (4B: %s)'
    % (verdict, verdict_4b))

# ==== persist ====
save = {
    'VERDICT': np.array(verdict),
    'VERDICT_4B': np.array(verdict_4b),
    'SMOKE': np.bool_(SMOKE),
    'PHASE': np.int64(PHASE)}
for s in ('4B', '14B'):
    save['LATE3_%s' % s] = np.float64(
        RES[s]['late_med'])
    save['H_G3_%s' % s] = np.bool_(
        RES[s]['H_G3'])
    save['AMNORM_%s' % s] = \
        RES[s]['am_norm'].astype(
            np.float64)
    for kk, vv in RES[s]['A'].items():
        save['ANCH_%s_%s' % (s, kk)] = \
            np.float64(vv)
for s in ('14B',):
    for fa in FKEYS:
        if fa in RES[s]['CS1H_NEW']:
            save['CS1H_NEW_%s_%s'
                 % (s, fa)] = \
                RES[s]['CS1H_NEW'][fa] \
                .astype(np.float64)
            save['PATH_H0_%s_%s'
                 % (s, fa)] = \
                RES[s]['PATH_H0'][fa] \
                .astype(np.float64)
            save['PATH_M_%s_%s'
                 % (s, fa)] = \
                RES[s]['PATH_M'][fa] \
                .astype(np.float64)
            save['PATH_A_%s_%s'
                 % (s, fa)] = \
                RES[s]['PATH_A'][fa] \
                .astype(np.float64)
            save['BHSHARE_PRE_%s_%s'
                 % (s, fa)] = \
                RES[s]['BHSH_PRE'][fa] \
                .astype(np.float32)
            save['TOP8NAT_PRE_%s_%s'
                 % (s, fa)] = \
                RES[s]['T8N_PRE'][fa] \
                .astype(np.int64)
            save['OVERLAP_PRE_%s_%s'
                 % (s, fa)] = np.int64(
                RES[s]['OVL_PRE'][fa])
            save['COS_LAD_NEW_%s_%s'
                 % (s, fa)] = \
                RES[s]['COS_LAD_NEW'][fa] \
                .astype(np.float64)
            save['MED_C_NEW_%s_%s'
                 % (s, fa)] = np.float64(
                RES[s]['MED_C_NEW'][fa])
            save['R1_NH_NEW_%s_%s'
                 % (s, fa)] = \
                RES[s]['R1_NH_NEW'][fa] \
                .astype(np.float64)
            if fa in RES[s]['T8_R1_NEW']:
                save['T8_R1_NEW_%s_%s'
                     % (s, fa)] = \
                    RES[s]['T8_R1_NEW'][fa] \
                    .astype(np.int64)
            if fa in RES[s]['VB37']:
                save['VB37_%s_%s'
                     % (s, fa)] = \
                    RES[s]['VB37'][fa] \
                    .astype(np.float32)
if '14B' in RES and RES['14B']['ratios']:
    save['H_G2_RATIOS'] = np.array(
        RES['14B']['ratios'],
        np.float64)
save['H_G1'] = np.bool_(
    RES['14B']['H_G1'])
save['H_G2'] = np.bool_(
    RES['14B']['H_G2'])
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
        'H_G1': RES['14B']['H_G1'],
        'H_G2': RES['14B']['H_G2'],
        'H_G3_14b': RES['14B']['H_G3'],
        'H_G3_4b': RES['4B']['H_G3'],
        'late3_14b': RES['14B'][
            'late_med'],
        'late3_4b': RES['4B']['late_med'],
        'g2_ratios': RES['14B']['ratios'],
        'overlap_pre': {
            fa: int(RES['14B']['OVL_PRE'][
                fa])
            for fa in FKEYS
            if fa in RES['14B']['OVL_PRE']},
        'top8_nat_pre': {
            fa: [int(x) for x in
                 RES['14B']['T8N_PRE'][fa]]
            for fa in FKEYS
            if fa in RES['14B']
            ['T8N_PRE']},
        'top8_3093': {
            fa: [int(x) for x in
                 z93['TOP8_' + fa]]
            for fa in FKEYS}},
    'anchors': {
        'd1_max': RES['14B']['A']['d1'],
        'd1b_max': RES['14B']['A'][
            'd1b'],
        'd1v_max': RES['14B']['A'][
            'd1v'],
        'd2_max': RES['14B']['A']['d2'],
        'd2b_max': RES['14B']['A'][
            'd2b'],
        'd3a_max': RES['14B']['A'][
            'd3a'],
        'd3b_max': RES['14B']['A'][
            'd3b'],
        'd4': {'p98_ok': bool(d4a and d4b),
               'p93_ok': bool(d4c)},
        'd5_max': {s: RES[s]['A']['d5']
                   for s in ('4B', '14B')},
        'd6_max': {s: RES[s]['A']['d6']
                   for s in ('4B', '14B')}},
    'inputs': {
        'p3098_npz8': h8(P98),
        'p3093_npz8': h8(P93)},
    'npz_sha256_8': npz8,
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
