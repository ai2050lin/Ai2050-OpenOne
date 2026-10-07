# -*- coding: utf-8 -*-
"""Phase 3096  Omega-P94: f2 logit-lens
layer profile (4B + 14B, dual-arm).

3094/3095 found: on qwen3-14b the TT
angle factor f2 collapses on AC/BC
(med 0.595->0.169, 0.598->0.183) with
survival prefix-dominated (Shakespearean
overall + formal x AB).  Open question:
WHERE in the 40-layer stack does the
collapse emerge -- representation level
(early) or readout assembly (late)?

Pipeline (per arm, frozen):
  assemble the 3076-identical 32
  prompts per family (4 prefixes x 8
  causal-connective bodies); forward
  each prompt once with
  output_hidden_states=True; keep the
  LAST-position hidden state of every
  layer L=0..NL (L=0 embedding output,
  L=NL final layer output);
  logit-lens: Z_L = lm_head(norm(h_L));
  TT_L[k] = Z_L[pref_k] - Z_L[base_k]
  (same pref/base indexing as 3076:
  cond=cidx[k] in 1..3, body=bidx[k],
  base cond=0); f2_L[fa,fb,k] =
  cos(TT_f_L[k], TT_g_L[k]);
  d = L/NL normalized depth, common
  grid 0.0..1.0 step 0.1 (nearest
  layer) for the 36L vs 40L comparison.

Anchors (frozen):
  a1: lens(final layer) vs native
      out.logits[0,-1] <= 1e-6
      (expect bit 0: same norm + head
      modules, same bf16 order).
  a2: final-layer f2 recomputed from
      native logits vs sealed F2_CTT
      (3079 npz for 4B / 3093 npz for
      14B) <= 1e-9 (same cosv path as
      3076/3093; 3094 proved replay
      bit-0).
  a2b: final-layer f2 from the lens
      path vs sealed F2_CTT <= 1e-9
      (locks the profile endpoint).

Preregistered decision gates (frozen;
d_div[key][ci] = first grid d with
(M4B - M14B) >= 0.2 and tail mean of
the delta over later grid points
>= 0.15; none -> 1.0):
  H_B1 late_assembly:
      min(d_div[AC,ci1], d_div[BC,ci1])
      >= 0.6  -> the 14B collapse is a
      late (readout-assembly) event.
  H_B2 early_rep:
      max(d_div over AC/BC, all ci)
      <= 0.3  -> the collapse is a
      representation-level difference.
  H_B3 style_split:
      median(d_div[*,ci2]) >=
      median(d_div[*,ci1]) + 0.15
      -> the Shakespearean alignment
      separates later than formal.
  order: H_B1 -> H_B2 -> H_B3 ->
  mixed -> inconclusive; verdicts
  fifth_lens_late_assembly /
  fifth_lens_early_rep /
  fifth_lens_style_split /
  fifth_lens_mixed /
  fifth_lens_inconclusive.
SMOKE: prefixes 0..1 only (ci1 pairs),
verdict fifth_lens_smoke.

Limitations (recorded): logit-lens
mid-layer readouts are a probe, not the
model's actual computation path (known
lens caveat); d-grid nearest-layer
mapping coarsens the 36L/40L alignment;
n=8 bodies per ci group; causal-
connective paradigm only.

Memory discipline: 4B arm completes and
frees (del + gc + empty_cache) BEFORE
the 14B load (model-sequential OOM
rule).  h_bank fp32 (bf16 values stored
losslessly); per-layer Z discarded
after f2 extraction.
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

PHASE = 3096
NAME = 'omega_p94_f2_lens_layer_profile'
SEED = 3096
ROOT = r'D:\AI2050\Ai2050-OpenOne'
R13 = (ROOT + r'\tests\glm5\result'
       r'\rdc_query_construction_20260913')
P4B79 = (R13 + r'\phase3079'
         r'\omega_p76_migration_lock'
         r'\omega_p76_migration_lock.npz')
P14B = (R13 + r'\phase3093'
        r'\omega_p91_qwen14b_l37_full_arbitration'
        r'\omega_p91_qwen14b_l37_full_'
        r'arbitration.npz')
SMOKE = os.environ.get('SMOKE', '0') == '1'
OUT = (R13 + r'\phase3096' + '\\' + NAME)
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

D_GRID = [round(0.1 * i, 1)
          for i in range(11)]
GATES = {
    'H_B1_late_assembly': {
        'min_d_div_acbc_ci1_ge': 0.6},
    'H_B2_early_rep': {
        'max_d_div_acbc_all_ci_le': 0.3},
    'H_B3_style_split': {
        'd_div_ci2_median_ge_ci1_median_'
        'plus': 0.15},
    'd_div_def': {'delta_ge': 0.2,
                  'tail_mean_ge': 0.15,
                  'no_div_value': 1.0},
    'd_half_def': {'below': 0.35}}

exec_doc = {
    'phase': PHASE, 'name': NAME,
    'frozen_before_compute': True,
    'question': ('at which normalized '
                 'depth does the 14B f2 '
                 'collapse emerge '
                 '(representation vs '
                 'readout assembly)'),
    'pipeline': {
        'lens': ('Z_L=lm_head(norm(h_L)) '
                 'last position for L<NL '
                 '(pre-norm states); L=NL '
                 'is the post-final-norm '
                 'state so Z_NL=lm_head(h) '
                 '(native logits path); '
                 'TT_L[k]=Z_L[pref_k]-'
                 'Z_L[base_k]; f2_L=cos'),
        'd_grid': D_GRID,
        'grid_map': 'nearest layer'},
    'gates': GATES,
    'decision_order': ['H_B1', 'H_B2',
                       'H_B3', 'mixed',
                       'inconclusive'],
    'anchors': {
        'a1': ('lens(final) vs native '
               'logits <=1e-6, expect '
               'bit 0'),
        'a2': ('native final f2 vs '
               'sealed F2_CTT '
               '(3079/3093) <=1e-9'),
        'a2b': ('lens final f2 vs '
                'sealed <=1e-9')},
    'inputs': {'p4b79_npz8': h8(P4B79),
               'p14b_npz8': h8(P14B)},
    'models': {k: {'mdir': v['mdir'],
                   'nl': v['nl'],
                   'hid': v['hid'],
                   'nq': v['nq'],
                   'kv': v['kv']}
               for k, v in
               ARM.items()},
    'forwards_per_arm': 96,
    'smoke': SMOKE}
with io.open(OUT + r'\execution.json',
             'w', encoding='utf-8') as f:
    json.dump(exec_doc, f, indent=1,
              ensure_ascii=False)
log('execution.json written')

z4_79 = np.load(P4B79, allow_pickle=False)
z14 = np.load(P14B, allow_pickle=False)
SEALED = {'4B': z4_79, '14B': z14}


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


def run_arm(side):
    cfg = ARM[side]
    NL = cfg['nl']
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
    tied = bool(getattr(
        model.config,
        'tie_word_embeddings', False))
    log('[%s] loaded nl=%d hid=%d tied=%s'
        % (side, NL, cfg['hid'], tied))
    norm_mod = model.model.norm
    head = model.lm_head
    n_pr = len(CONS) * 8
    h_bank = {}
    LG = {}
    idx_all = {}
    cid_all = {}
    bid_all = {}
    for fkey in FKEYS:
        assembled, idx_of, cidx, bidx = \
            assemble(fkey, tok)
        idx_all[fkey] = idx_of
        cid_all[fkey] = cidx
        bid_all[fkey] = bidx
        hb = np.zeros((n_pr, NL + 1,
                       cfg['hid']),
                      dtype=np.float32)
        lg = np.zeros((n_pr, 151936))
        for i, it in enumerate(assembled):
            with torch.no_grad():
                out = model(torch.tensor(
                    [it['ids']],
                    device='cuda'),
                    use_cache=False,
                    output_hidden_states=True)
            lg[i] = out.logits[0, -1] \
                .detach().double() \
                .cpu().numpy()
            hs = out.hidden_states
            assert len(hs) == NL + 1
            hb[i] = np.stack([
                hs[j][0, -1].detach()
                .float().cpu().numpy()
                for j in range(NL + 1)])
        h_bank[fkey] = hb
        LG[fkey] = lg
        log('[%s][%s] forwarded %d prompts'
            % (side, fkey, n_pr))
    # a1 lens(final) vs native logits
    # NOTE: hidden_states[NL] is the
    # POST-final-norm state (HF appends
    # it outside the layer loop), so the
    # lens at L=NL is head(h) directly;
    # L<NL states are pre-norm.
    a1_diff = 0.0
    for fkey in FKEYS:
        h = torch.from_numpy(
            h_bank[fkey][:, NL, :]) \
            .to('cuda') \
            .to(torch.bfloat16)
        with torch.no_grad():
            z = head(h)
        zn = z.double().cpu().numpy()
        a1_diff = max(a1_diff, float(
            np.max(np.abs(
                zn - LG[fkey]))))
    a1_ok = a1_diff <= 1e-6
    log('[%s] a1 lens(final) vs native '
        'diff=%.3e ok=%s'
        % (side, a1_diff, a1_ok))
    # per-layer profile
    F2P = {key: np.zeros((NP_, NL + 1))
           for key in ('AB', 'AC', 'BC')}
    for L in range(NL + 1):
        Z = {}
        for fkey in FKEYS:
            h = torch.from_numpy(
                h_bank[fkey][:, L, :]) \
                .to('cuda') \
                .to(torch.bfloat16)
            with torch.no_grad():
                z = head(h) if L == NL \
                    else head(norm_mod(h))
            Z[fkey] = z.double() \
                .cpu().numpy()
        cidx = cid_all['A']
        bidx = bid_all['A']
        for (fa, fb) in CPAIRS:
            key = fa + fb
            for k in range(NP_):
                pf = Z[fa][idx_all[fa][(
                    int(cidx[k]),
                    int(bidx[k]))]]
                pb = Z[fa][idx_all[fa][(
                    0, int(bidx[k]))]]
                gf = Z[fb][idx_all[fb][(
                    int(cidx[k]),
                    int(bidx[k]))]]
                gb = Z[fb][idx_all[fb][(
                    0, int(bidx[k]))]]
                F2P[key][k, L] = cosv(
                    pf - pb, gf - gb)
    # a2 native final f2 vs sealed
    # a2b lens final f2 vs sealed
    a2_diff = 0.0
    a2b_diff = 0.0
    cidx = cid_all['A']
    bidx = bid_all['A']
    for (fa, fb) in CPAIRS:
        key = fa + fb
        for k in range(NP_):
            pf = LG[fa][idx_all[fa][(
                int(cidx[k]),
                int(bidx[k]))]]
            pb = LG[fa][idx_all[fa][(
                0, int(bidx[k]))]]
            gf = LG[fb][idx_all[fb][(
                int(cidx[k]),
                int(bidx[k]))]]
            gb = LG[fb][idx_all[fb][(
                0, int(bidx[k]))]]
            f2n = cosv(pf - pb, gf - gb)
            a2_diff = max(a2_diff, abs(
                f2n - float(
                    z_sealed['F2_CTT_'
                             + key][k])))
            a2b_diff = max(a2b_diff, abs(
                F2P[key][k, NL] - float(
                    z_sealed['F2_CTT_'
                             + key][k])))
    a2_ok = a2_diff <= 1e-9
    a2b_ok = a2b_diff <= 1e-9
    log('[%s] a2 native f2 vs sealed '
        'diff=%.3e ok=%s | a2b lens final '
        'diff=%.3e ok=%s'
        % (side, a2_diff, a2_ok,
           a2b_diff, a2b_ok))
    del model, tok, norm_mod, head
    del h_bank, LG
    gc.collect()
    torch.cuda.empty_cache()
    log('[%s] arm done, memory freed'
        % side)
    return {'F2P': F2P, 'nl': NL,
            'a1_ok': a1_ok,
            'a2_ok': a2_ok,
            'a2b_ok': a2b_ok,
            'a1_diff': a1_diff,
            'a2_diff': a2_diff,
            'a2b_diff': a2b_diff}


RES = {}
RES['4B'] = run_arm('4B')
RES['14B'] = run_arm('14B')
flush_log()

assert RES['4B']['a1_ok'] \
    and RES['14B']['a1_ok']
assert RES['4B']['a2_ok'] \
    and RES['14B']['a2_ok']
assert RES['4B']['a2b_ok'] \
    and RES['14B']['a2b_ok']
log('anchors ok (both arms)')
flush_log()

if SMOKE:
    VERDICT = 'fifth_lens_smoke'
    log('SMOKE verdict: %s' % VERDICT)
    flush_log()
else:
    # ---------- E1 curves ----------
    CIDX = np.array([k // 8 + 1
                     for k in range(24)])
    E = {'curves': {}, 'd_div': {},
         'd_half': {}, 'auc': {}}

    def curve(side, key, ci):
        F2P = RES[side]['F2P'][key]
        NL = RES[side]['nl']
        out = []
        for d in D_GRID:
            Lq = int(round(d * NL))
            m = CIDX == ci
            out.append(float(np.median(
                F2P[m, Lq])))
        return np.array(out)

    for key in ('AB', 'AC', 'BC'):
        for ci in (1, 2, 3):
            c4 = curve('4B', key, ci)
            c14 = curve('14B', key, ci)
            E['curves']['%s_ci%d'
                        % (key, ci)] = {
                'grid': D_GRID,
                'm4B': c4.tolist(),
                'm14B': c14.tolist()}
            dlt = c4 - c14
            ddiv = 1.0
            for gi, d in enumerate(
                    D_GRID):
                if dlt[gi] >= 0.2:
                    tail = dlt[gi:]
                    if float(
                            tail.mean()) \
                            >= 0.15:
                        ddiv = d
                        break
            E['d_div']['%s_ci%d'
                       % (key, ci)] = ddiv
            dh = 1.0
            for gi, d in enumerate(
                    D_GRID):
                if c14[gi] < 0.35:
                    dh = d
                    break
            E['d_half']['14B_%s_ci%d'
                        % (key, ci)] = dh
            E['auc']['14B_%s_ci%d'
                     % (key, ci)] = float(
                np.trapezoid(c14, dx=0.1))
    for key in ('AB', 'AC', 'BC'):
        for ci in (1, 2, 3):
            log('E1 %s ci%d: 4B %s | 14B %s'
                % (key, ci,
                   ' '.join('%.2f' % x for x
                            in E['curves']
                            ['%s_ci%d'
                             % (key, ci)]
                            ['m4B']),
                   ' '.join('%.2f' % x for x
                            in E['curves']
                            ['%s_ci%d'
                             % (key, ci)]
                            ['m14B'])))
    log('E1 d_div: %s'
        % json.dumps(E['d_div']))
    # ---------- decision ----------
    H_B1 = min(E['d_div']['AC_ci1'],
               E['d_div']['BC_ci1']) >= 0.6
    H_B2 = max(
        E['d_div']['AC_ci%d' % ci]
        for ci in (1, 2, 3)) <= 0.3 and \
        max(E['d_div']['BC_ci%d' % ci]
            for ci in (1, 2, 3)) <= 0.3
    med_ci2 = float(np.median(
        [E['d_div']['%s_ci2' % key]
         for key in ('AC', 'BC')]))
    med_ci1 = float(np.median(
        [E['d_div']['%s_ci1' % key]
         for key in ('AC', 'BC')]))
    H_B3 = med_ci2 >= med_ci1 + 0.15
    log('decision: H_B1=%s H_B2=%s '
        'H_B3=%s (med_ci2=%.2f '
        'med_ci1=%.2f)'
        % (H_B1, H_B2, H_B3, med_ci2,
           med_ci1))
    if H_B1:
        VERDICT = \
            'fifth_lens_late_assembly'
    elif H_B2:
        VERDICT = 'fifth_lens_early_rep'
    elif H_B3:
        VERDICT = 'fifth_lens_style_split'
    elif not (H_B1 or H_B2 or H_B3):
        VERDICT = 'fifth_lens_mixed'
    else:
        VERDICT = \
            'fifth_lens_inconclusive'
    log('VERDICT: %s' % VERDICT)
    E['decision'] = {
        'H_B1': bool(H_B1),
        'H_B2': bool(H_B2),
        'H_B3': bool(H_B3),
        'med_ci2': med_ci2,
        'med_ci1': med_ci1}

# ---------- persist ----------
save = {
    'VERDICT': np.array(VERDICT),
    'SMOKE': np.bool_(SMOKE),
    'PHASE': np.int64(PHASE),
    'SEED': np.int64(SEED),
    'NL_4B': np.int64(RES['4B']['nl']),
    'NL_14B': np.int64(RES['14B']['nl']),
    'A1_4B': np.float64(
        RES['4B']['a1_diff']),
    'A1_14B': np.float64(
        RES['14B']['a1_diff']),
    'A2_4B': np.float64(
        RES['4B']['a2_diff']),
    'A2_14B': np.float64(
        RES['14B']['a2_diff']),
    'A2B_4B': np.float64(
        RES['4B']['a2b_diff']),
    'A2B_14B': np.float64(
        RES['14B']['a2b_diff'])}
for side in ('4B', '14B'):
    for key in ('AB', 'AC', 'BC'):
        save['F2P_%s_%s' % (side, key)] = \
            RES[side]['F2P'][key] \
            .astype(np.float64)
if not SMOKE:
    save['D_GRID'] = np.array(D_GRID)
    for key in ('AB', 'AC', 'BC'):
        for ci in (1, 2, 3):
            ck = '%s_ci%d' % (key, ci)
            save['M4B_' + ck] = np.array(
                E['curves'][ck]['m4B'])
            save['M14B_' + ck] = np.array(
                E['curves'][ck]['m14B'])
            save['DDIV_' + ck] = \
                np.float64(
                    E['d_div'][ck])
npz_path = OUT + r'\%s.npz' % NAME
np.savez(npz_path, **save)

res = {
    'phase': PHASE, 'name': NAME,
    'verdict': VERDICT,
    'smoke': SMOKE,
    'forwards': 96 * 2,
    'anchors': {
        'a1_diff_4B': RES['4B']['a1_diff'],
        'a1_diff_14B':
            RES['14B']['a1_diff'],
        'a2_diff_4B': RES['4B']['a2_diff'],
        'a2_diff_14B':
            RES['14B']['a2_diff'],
        'a2b_diff_4B':
            RES['4B']['a2b_diff'],
        'a2b_diff_14B':
            RES['14B']['a2b_diff'],
        'ok': True}}
if not SMOKE:
    res['stats'] = E
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
