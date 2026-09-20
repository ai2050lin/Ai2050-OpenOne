# -*- coding: utf-8 -*-
"""Phase 2941: causal injection along v3 - is the v3 push a
passive geometric displacement or an amplified readout-failure
mechanism?

Why: 2939 identified v3 (3rd SVD basis of stacked dirs_word) as
the top net-energy inflow under null context (delta_e_med
+0.105, p 1e-4); 2940 found v3 is owned by mid layers L14-L18
(w_li top L16/L17/L18) and is word-attribute blind. MEMO 2940
states "null pushes the last-position residual along v3 away
from the dir35 axis". Preflight (discipline 10, on existing
artifacts only) established:
  cos(v3, dirs_word[35]) = -0.227 (direct linear readout
  effect -0.227*delta exists and is predictable);
  v3 term explains only 0.4% of the per-word variance of the
  actual dir35 readout shift (U8 total: Spearman 0.994);
  v3 class-shift contribution to sep displacement: -0.7 of
  the actual -108.4.
So the DECIDABLE causal question is: does injecting delta*v3
at the L16 attn-input (pos 1) produce a dir35 readout effect
equal to the linear direct prediction (passive), amplified
(nonlinear mechanism support), or attenuated (damped)?

Mode: ONE model (qwen3-4b), forward family. 2927 injection
site verbatim (attn-input pos-1, layer L16 = argmax w_li),
2938/2939 batch57 readout (final pre-norm residual).

Injection grid: delta in {0, +-2, +-4, +-8, +-16, +-32} for
conditions func and null0 (null0 = 2939 sample_null(2896)
verbatim). 22 batch57 forwards.

Anchors (frozen; reachability preflighted):
  a1 dirs_word rebuild vs 2927 npz < 1e-5   (2939: 2.17e-08)
  a2 func baseline determinism < 1e-4       (2939: 0.0)
  a3 Vt8 rebuild vs 2939 npz < 1e-6         (2940: 3.04e-08)
  a4 proj_func vs 2935 s_base[func] < 1e-4  (2939: 7.2e-06)
  a5 proj_null0 vs 2935 s_base[null0] < 1e-4(2939: 6.3e-06)
  a6 func separation > 0                    (2939: 185.7)
  a7 c3_func rebuild vs 2940 npz < 1e-3 (reachability: dir
    noise 3e-08 * fin norm ~730 => ~2e-05; threshold set at
    1e-3, catches implementation errors not bf16 noise)

Main tests (frozen):
  P1 (func, +delta along v3): per-unit gain
     g(delta) = (proj35_inj(delta) - proj35_base) / delta,
     delta in {2,4,8,16,32}; gain_med = median.
     linear prediction = cos(v3, u35) ~= -0.227 (this run's
     rebuild value used).
     amplified : |gain_med| >= 2*|cos|
     linear    : 0.5*|cos| <= |gain_med| < 2*|cos|
     attenuated: |gain_med| < 0.5*|cos|
  P2 (null0, -delta cancellation): rec(16) =
     (proj35_inj(-16) - proj35_null0_base) / (-16*cos)
     (linear additive prediction denominator, positive).
     additive    : 0.5 <= rec16 <= 2.0
     superadditive: rec16 > 2.0
     subadditive : rec16 < 0.5

Verdict (frozen):
  anchor fail                     => anchor_fail_all_void
  P1 amplified                    => v3_push_causally_amplified
  P1 linear AND P2 additive       => v3_push_geometric_passive
  P1 attenuated AND P2 subadditive=> v3_push_damped
  else                            => v3_push_mixed

Descriptive: D1 c3 propagation slope per delta (func/null0);
D2 sep(delta); D3 word-level structure rho vs func base;
D4 fin norm; D5 odd-order symmetry (gain at -delta).

Output: phase2941/v3_causal_injection/.
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2887 = os.path.join(BASE, 'phase2887', 'language_axis_mlp',
                        'language_axis_mlp.npz')
SRC_2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                        'probe_relativity.npz')
SRC_2935 = os.path.join(BASE, 'phase2935', 'null_amp_anatomy',
                        'null_amp_anatomy.npz')
SRC_2939 = os.path.join(BASE, 'phase2939', 'rotation_target',
                        'rotation_target.npz')
SRC_2940 = os.path.join(BASE, 'phase2940', 'v3_decode',
                        'v3_decode.npz')
OUT = os.path.join(BASE, 'phase2941', 'v3_causal_injection')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2941_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
SEED = 2896
NH, HD = 32, 128
NL = 36
VOCAB = 151936
L_INJ = 16                    # argmax w_li (2940: w=0.625)
DELTAS = (2.0, 4.0, 8.0, 16.0, 32.0)
AMP_F = 2.0
LIN_F = 0.5
REC_LO, REC_HI = 0.5, 2.0
REC_DELTA = 16.0

PREREG = {
    'mode': 'forward family: 2927 injection site verbatim '
            '(attn-input pos-1, layer L16=argmax w_li), '
            '2938/2939 batch57 final pre-norm readout; '
            'delta in {0,+-2,+-4,+-8,+-16,+-32} x '
            '{func, null0(2939 sample_null(2896) verbatim)}',
    'question': 'is the null-context v3 push a passive '
                'geometric displacement of the dir35 readout '
                '(causal gain == linear direct prediction '
                'cos(v3,u35)=-0.227) or an amplified '
                'readout-failure mechanism (gain >= 2x)?',
    'anchors': {
        'a1': 'dirs_word rebuild vs 2927 npz < 1e-5',
        'a2': 'func baseline determinism < 1e-4',
        'a3': 'Vt8 rebuild vs 2939 npz < 1e-6',
        'a4': 'proj_func vs 2935 s_base[func] < 1e-4',
        'a5': 'proj_null0 vs 2935 s_base[null0] < 1e-4',
        'a6': 'func separation > 0',
        'a7': 'c3_func rebuild vs 2940 npz < 1e-3',
    },
    'P1': 'func +delta: gain_med over {2,4,8,16,32} vs '
          'cos(v3,u35); amplified >=2x, linear 0.5-2x, '
          'attenuated <0.5x',
    'P2': 'null0 -delta cancellation: rec16 = '
          '(proj(-16)-proj_base)/(-16*cos); additive '
          '[0.5,2.0], superadditive >2, subadditive <0.5',
    'D1-D5': 'c3 slope, sep(delta), word-structure rho, '
             'fin norm, odd-order symmetry (descriptive)',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'P1 amplified => v3_push_causally_amplified; '
               'P1 linear AND P2 additive => '
               'v3_push_geometric_passive; P1 attenuated AND '
               'P2 subadditive => v3_push_damped; else => '
               'v3_push_mixed',
}


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-30)


def log(msg, lines):
    lines.append(msg)
    print(msg, flush=True)


def rankdata(x):
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(a, b):
    ra = rankdata(np.asarray(a, dtype=np.float64))
    rb = rankdata(np.asarray(b, dtype=np.float64))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    if den < 1e-30:
        return 0.0
    return float((ra * rb).sum() / den)


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2941,
                   'name': 'v3_causal_injection',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2887': sha8(SRC_2887),
                               's2927': sha8(SRC_2927),
                               's2935': sha8(SRC_2935),
                               's2939': sha8(SRC_2939),
                               's2940': sha8(SRC_2940)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'seed': SEED, 'layer_inj': L_INJ,
                   'deltas': list(DELTAS),
                   'amp_factor': AMP_F, 'lin_factor': LIN_F,
                   'rec_lo': REC_LO, 'rec_hi': REC_HI,
                   'rec_delta': REC_DELTA,
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- sources ----------
    z87 = np.load(SRC_2887, allow_pickle=True)
    words = [tuple(str(w).split(':')) for w in z87['words']]
    lab_lang = np.asarray(z87['labels_lang']).astype(int)
    n_words = len(words)
    assert n_words == 57

    z27 = np.load(SRC_2927, allow_pickle=True)
    dirs_word_27 = z27['dirs_word'].astype(np.float64)

    z35 = np.load(SRC_2935, allow_pickle=True)
    conds35 = [str(s) for s in z35['cond_names']]
    s_base_35 = z35['s_base'].astype(np.float64)
    ifu35 = conds35.index('func')
    in035 = conds35.index('null0')

    z39 = np.load(SRC_2939, allow_pickle=True)
    Vt8_39 = z39['Vt8'].astype(np.float64)
    _ = z39['coords'].astype(np.float64)   # source-chain pin
    conds39 = [str(s) for s in z39['cond_names']]
    proj35_39 = z39['proj_dir35'].astype(np.float64)
    if39 = conds39.index('func')
    i039 = conds39.index('null0')

    z40 = np.load(SRC_2940, allow_pickle=True)
    c3_func_40 = z40['c3_func'].astype(np.float64)
    log('sources ok', lines)

    # ---------- model ----------
    import torch
    import sys
    sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
    from phase2662_symmetric_mapping_contract import load_native
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        MD, local_files_only=True, trust_remote_code=True,
        use_fast=True)
    tc = {}

    def tid(t):
        if t not in tc:
            ids = tok(' ' + t, add_special_tokens=False)[
                'input_ids']
            if len(ids) != 1:
                ids = tok(t, add_special_tokens=False)[
                    'input_ids']
            assert len(ids) == 1, '%s -> %s' % (t, ids)
            tc[t] = int(ids[0])
        return tc[t]

    tid_map = {}
    for lang, ck, w in words:
        tid_map[w] = tid(w)
        if lang == 'en':
            assert tid_map[w] == int(ck), 'key mismatch %s' % w
    func_tid = tid('the')

    word_tids = set(tid_map.values())

    def sample_null(seed):
        rng = np.random.default_rng(seed)
        out = []
        while len(out) < n_words:
            r = int(rng.integers(0, VOCAB))
            if r not in word_tids and r > 0:
                out.append(r)
        return out

    null0_tids = sample_null(2896)
    batch = {'func': [[func_tid, tid_map[words[i][2]]]
                      for i in range(n_words)],
             'null0': [[null0_tids[i], tid_map[words[i][2]]]
                       for i in range(n_words)]}

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    log('model loaded (load_native full GPU)', lines)

    cap = {'attnin': {}}
    state_fin = {'on': False}
    fin_cap = {}
    inj = {'on': False, 'delta': 0.0, 'vec': None}
    handles = []

    def pre_attn(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('hidden_states')
            if x is None or x.dim() < 2:
                return
            if li == L_INJ and inj['on']:
                x = x.clone()
                x[:, 1, :] = x[:, 1, :] \
                    + inj['delta'] * inj['vec']
                if args:
                    nargs = (x,) + tuple(args[1:])
                    return nargs, kwargs
                nkw = dict(kwargs)
                nkw['hidden_states'] = x
                return args, nkw
            cap['attnin'].setdefault(li, []).append(
                x.detach().float().cpu().numpy())
            return None
        return h

    def pre_norm(module, args, kwargs):
        if state_fin['on']:
            fin_cap['x'] = args[0][:, -1, :].detach() \
                .float().cpu().numpy()

    for li in range(NL):
        handles.append(layers[li].self_attn
                       .register_forward_pre_hook(
                           pre_attn(li), with_kwargs=True))
    handles.append(model.model.norm.register_forward_pre_hook(
        pre_norm, with_kwargs=True))

    def clear_cap():
        for li in cap['attnin']:
            del cap['attnin'][li][:]

    def forward1(toks):
        clear_cap()
        with torch.no_grad():
            model(torch.tensor([toks], device='cuda'))
        return {li: cap['attnin'][li][0]
                for li in cap['attnin']}

    def forward_batch(toks_list, delta=0.0, vec=None):
        clear_cap()
        fin_cap.pop('x', None)
        state_fin['on'] = True
        inj['on'] = vec is not None
        inj['delta'] = float(delta)
        inj['vec'] = vec
        with torch.no_grad():
            model(torch.tensor(toks_list, device='cuda'))
        inj['on'] = False
        state_fin['on'] = False
        fin = fin_cap['x'].astype(np.float64)
        return fin

    vdt = next(layers[0].mlp.parameters()).dtype
    ldev = next(layers[L_INJ].mlp.parameters()).device

    # ---------- pass 1: dirs_word rebuild ----------
    attn_store = {}
    for i, (_, _, w) in enumerate(words):
        attnin_all = forward1([func_tid, tid_map[w]])
        for li in range(NL):
            attn_store[(i, li)] = \
                attnin_all[li].astype(np.float32)
        if (i + 1) % 20 == 0:
            log('pass1 words [%d/%d]' % (i + 1, n_words), lines)
    d_dim = attn_store[(0, 0)].shape[-1]
    diffs_w = np.zeros((NL, d_dim))
    for li in range(NL):
        X = np.stack([attn_store[(i, li)][0, 1]
                      for i in range(n_words)]) \
            .astype(np.float64)
        diffs_w[li] = X[lab_lang == 0].mean(0) \
            - X[lab_lang == 1].mean(0)
    dirs_word = np.stack([unit(diffs_w[li]) for li in range(NL)])
    a1_diff = float(np.abs(dirs_word - dirs_word_27).max())
    a1_ok = bool(a1_diff < 1e-5)
    log('a1 dirs_word rebuild diff %.2e ok=%s'
        % (a1_diff, a1_ok), lines)

    U_svd, s_svd, Vt = np.linalg.svd(dirs_word,
                                     full_matrices=False)
    Vt8 = Vt[:8]
    a3_diff = float(np.abs(Vt8 - Vt8_39).max())
    a3_ok = bool(a3_diff < 1e-6)
    log('a3 Vt8 rebuild vs 2939 max diff %.2e ok=%s'
        % (a3_diff, a3_ok), lines)

    u35 = dirs_word[NL - 1]
    v3 = Vt8[2]
    cos_v3_u35 = float(v3 @ u35)
    log('cos(v3, u35) this run = %.6f' % cos_v3_u35, lines)

    v3_t = torch.tensor(v3, device=ldev, dtype=vdt)

    # ---------- baseline + determinism ----------
    fin_f1 = forward_batch(batch['func'])
    fin_f2 = forward_batch(batch['func'])
    a2_rel = float(np.abs(fin_f1 - fin_f2).max()
                   / max(float(np.abs(fin_f1).max()), 1e-30))
    a2_ok = bool(a2_rel < 1e-4)
    log('a2 baseline determinism rel %.2e ok=%s'
        % (a2_rel, a2_ok), lines)

    proj = {}
    c3 = {}
    nfin = {}
    proj['func0'] = fin_f1 @ u35
    c3['func0'] = fin_f1 @ v3
    nfin['func0'] = np.linalg.norm(fin_f1, axis=1)
    fin_n0 = forward_batch(batch['null0'])
    proj['null0_base'] = fin_n0 @ u35
    c3['null0_base'] = fin_n0 @ v3
    nfin['null0_base'] = np.linalg.norm(fin_n0, axis=1)

    a4_diff = float(np.abs(proj['func0'] - s_base_35[ifu35])
                    .max())
    a4_ok = bool(a4_diff < 1e-4)
    log('a4 proj_func vs 2935 max abs diff %.2e ok=%s'
        % (a4_diff, a4_ok), lines)
    a5_diff = float(np.abs(proj['null0_base'] - s_base_35[in035])
                    .max())
    a5_ok = bool(a5_diff < 1e-4)
    log('a5 proj_null0 vs 2935 max abs diff %.2e ok=%s'
        % (a5_diff, a5_ok), lines)
    a7_diff = float(np.abs(c3['func0'] - c3_func_40).max())
    a7_ok = bool(a7_diff < 1e-3)
    log('a7 c3_func rebuild vs 2940 max abs diff %.2e ok=%s'
        % (a7_diff, a7_ok), lines)

    sep_f = float(proj['func0'][lab_lang == 0].mean()
                  - proj['func0'][lab_lang == 1].mean())
    a6_ok = bool(sep_f > 0.0)
    log('a6 func separation %.4f ok=%s' % (sep_f, a6_ok), lines)

    # cross-check vs 2939 projections (descriptive)
    d39 = float(np.abs(proj['func0'] - proj35_39[if39]).max())
    d39n = float(np.abs(proj['null0_base']
                        - proj35_39[i039]).max())
    log('descriptive proj vs 2939: func %.2e null0 %.2e'
        % (d39, d39n), lines)

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok and a6_ok and a7_ok)
    verdict = None
    p1 = p2 = d1 = d2 = d3 = d4 = d5 = None
    save = {}

    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        # ---------- injection sweep ----------
        keys = []
        for cn in ('func', 'null0'):
            for sgn in ('+', '-'):
                for delta in DELTAS:
                    key = '%s_%s%d' % (cn, sgn, int(delta))
                    keys.append(key)
                    fin = forward_batch(
                        batch[cn], delta=(delta if sgn == '+'
                                          else -delta),
                        vec=v3_t)
                    proj[key] = fin @ u35
                    c3[key] = fin @ v3
                    nfin[key] = np.linalg.norm(fin, axis=1)
        log('injection sweep done (%d forwards)'
            % len(keys), lines)

        base_f = proj['func0']
        base_n = proj['null0_base']

        # ---------- P1: func +delta gain ----------
        gains = {}
        for delta in DELTAS:
            key = 'func_+%d' % int(delta)
            gains[delta] = (proj[key] - base_f) / delta
        gain_med = float(np.median(
            [float(np.median(gains[d])) for d in DELTAS]))
        ac = abs(cos_v3_u35)
        if gain_med >= AMP_F * ac or gain_med <= -AMP_F * ac:
            p1_class = 'amplified'
        elif (LIN_F * ac <= abs(gain_med) < AMP_F * ac):
            p1_class = 'linear_passive'
        else:
            p1_class = 'attenuated'
        p1 = {'gain_per_delta': {
                  str(int(d)): round(float(np.median(gains[d])),
                                     5) for d in DELTAS},
              'gain_med': round(gain_med, 5),
              'cos_v3_u35': round(cos_v3_u35, 6),
              'class': p1_class}
        log('P1 gains %s | med %.5f vs cos %.5f -> %s'
            % (p1['gain_per_delta'], gain_med, cos_v3_u35,
               p1_class), lines)

        # ---------- P2: null0 -delta cancellation ----------
        denom = -REC_DELTA * cos_v3_u35  # linear prediction (>0)
        recs = {}
        for delta in DELTAS:
            key = 'null0_-%d' % int(delta)
            obs = float(np.median(proj[key] - base_n))
            recs[delta] = obs / denom
        rec16 = recs[REC_DELTA]
        if REC_LO <= rec16 <= REC_HI:
            p2_class = 'additive'
        elif rec16 > REC_HI:
            p2_class = 'superadditive'
        else:
            p2_class = 'subadditive'
        p2 = {'rec_per_delta': {
                  str(int(d)): round(recs[d], 4)
                  for d in DELTAS},
              'rec16': round(rec16, 4),
              'denom_linear': round(denom, 4),
              'class': p2_class}
        log('P2 rec %s | rec16 %.4f (denom %.4f) -> %s'
            % (p2['rec_per_delta'], rec16, denom, p2_class),
            lines)

        # ---------- D1: c3 propagation slope ----------
        d1 = {}
        for cn, bk in (('func', 'func0'),
                       ('null0', 'null0_base')):
            sl = {}
            for sgn in ('+', '-'):
                for delta in DELTAS:
                    key = '%s_%s%d' % (cn, sgn, int(delta))
                    d = delta if sgn == '+' else -delta
                    sl['%s%d' % (sgn, int(delta))] = round(
                        float(np.median(
                            (c3[key] - c3[bk]) / d)), 4)
            d1[cn] = sl
        log('D1 c3 slopes: func %s | null0 %s'
            % (d1['func'], d1['null0']), lines)

        # ---------- D2: sep(delta) ----------
        m0 = lab_lang == 0
        m1 = lab_lang == 1
        d2 = {}
        for cn in ('func', 'null0'):
            row = {}
            for sgn in ('+', '-'):
                for delta in DELTAS:
                    key = '%s_%s%d' % (cn, sgn, int(delta))
                    row['%s%d' % (sgn, int(delta))] = round(
                        float(proj[key][m0].mean()
                              - proj[key][m1].mean()), 2)
            row['base'] = round(float(
                proj['func0' if cn == 'func'
                     else 'null0_base'][m0].mean()
                - proj['func0' if cn == 'func'
                       else 'null0_base'][m1].mean()), 2)
            d2[cn] = row
        log('D2 sep: func %s | null0 %s'
            % (d2['func'], d2['null0']), lines)

        # ---------- D3: word-structure rho vs func base ----
        d3 = {}
        for cn in ('func', 'null0'):
            row = {}
            for sgn in ('+', '-'):
                for delta in DELTAS:
                    key = '%s_%s%d' % (cn, sgn, int(delta))
                    row['%s%d' % (sgn, int(delta))] = round(
                        spearman(proj[key], base_f), 4)
            d3[cn] = row
        log('D3 rho vs func base: func %s | null0 %s'
            % (d3['func'], d3['null0']), lines)

        # ---------- D4: fin norm ----------
        d4 = {}
        for cn, bk in (('func', 'func0'),
                       ('null0', 'null0_base')):
            row = {}
            for sgn in ('+', '-'):
                for delta in DELTAS:
                    key = '%s_%s%d' % (cn, sgn, int(delta))
                    ratio = float(np.median(
                        nfin[key]
                        / np.maximum(nfin[bk], 1e-30)))
                    row['%s%d' % (sgn, int(delta))] = round(
                        ratio, 5)
            d4[cn] = row
        log('D4 norm ratios: func %s | null0 %s'
            % (d4['func'], d4['null0']), lines)

        # ---------- D5: odd-order symmetry ----------
        d5 = {}
        for cn in ('func', 'null0'):
            row = {}
            for delta in DELTAS:
                kp = '%s_+%d' % (cn, int(delta))
                km = '%s_-%d' % (cn, int(delta))
                sp = float(np.median(proj[kp] - base_f
                                     if cn == 'func'
                                     else proj[kp] - base_n))
                sm = float(np.median(proj[km] - base_f
                                     if cn == 'func'
                                     else proj[km] - base_n))
                lin_r = (abs(sp) + abs(sm)) / max(abs(sp - sm),
                                                  1e-30)
                row[str(int(delta))] = {
                    'shift+': round(sp, 3),
                    'shift-': round(sm, 3),
                    'evenness': round(lin_r, 3)}
            d5[cn] = row
        log('D5 symmetry func: %s'
            % {k: v['evenness']
               for k, v in d5['func'].items()}, lines)

        # ---------- verdict ----------
        if p1_class == 'amplified':
            verdict = 'v3_push_causally_amplified'
        elif p1_class == 'linear_passive' \
                and p2_class == 'additive':
            verdict = 'v3_push_geometric_passive'
        elif p1_class == 'attenuated' \
                and p2_class == 'subadditive':
            verdict = 'v3_push_damped'
        else:
            verdict = 'v3_push_mixed'

        save = {
            'words': np.array(['%s:%s:%s' % w
                               for w in words],
                              dtype=object),
            'labels_lang': lab_lang,
            'keys': np.array(keys, dtype=object),
            'proj': np.stack([proj[k] for k in keys]),
            'c3': np.stack([c3[k] for k in keys]),
            'fin_norm': np.stack([nfin[k] for k in keys]),
            'base_keys': np.array(['func0', 'null0_base'],
                                  dtype=object),
            'proj_base': np.stack([proj['func0'],
                                   proj['null0_base']]),
            'c3_base': np.stack([c3['func0'],
                                 c3['null0_base']]),
            'Vt8': Vt8,
            'dirs_word': dirs_word,
        }

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2941, 'model': 'qwen3-4b',
           'prereg': PREREG,
           'anchors': {'a1_diff': float('%.3e' % a1_diff),
                       'a1_ok': a1_ok,
                       'a2_rel': float('%.3e' % a2_rel),
                       'a2_ok': a2_ok,
                       'a3_diff': float('%.3e' % a3_diff),
                       'a3_ok': a3_ok,
                       'a4_diff': float('%.3e' % a4_diff),
                       'a4_ok': a4_ok,
                       'a5_diff': float('%.3e' % a5_diff),
                       'a5_ok': a5_ok,
                       'a6_sep_func': round(sep_f, 4),
                       'a6_ok': a6_ok,
                       'a7_diff': float('%.3e' % a7_diff),
                       'a7_ok': a7_ok,
                       'ok': anchor_ok},
           'descriptive_proj_vs_2939': {
               'func': float('%.3e' % d39),
               'null0': float('%.3e' % d39n)},
           'P1': p1, 'P2': p2,
           'D1_c3_slope': d1, 'D2_sep': d2, 'D3_rho': d3,
           'D4_norm': d4, 'D5_symmetry': d5,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(os.path.join(
            OUT, 'v3_causal_injection.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2941 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()
