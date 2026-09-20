# -*- coding: utf-8 -*-
"""Phase 2968: is the h15 biphasic peak position an
INDIVIDUAL property (locked to s_c per word) or a median
artifact of a heterogeneous mixture? And does the biphasic
shape generalize to the reverse (enhancement) direction?

Why: 2966 found the h15 C-dose curve at L34 is biphasic
(0.438 -> peak 0.714 at s=0.625 -> -0.064) with the peak
at the L17 switch threshold s_c=0.6567 - but that was the
MEDIAN-word curve. A median peak can arise from a mixture
of monotone risers and fallers with no individual peaks.
2967 pre-check (design variable only): 40/57 words have an
inner argmax peak - reachability OK (discipline 10).

Mode: two forward families, protocol identical to 2966/2967
(57-word batch verbatim, 2953 GRID17 + s=0, K=1):
  family A: +xdir (replication anchor, bit-level vs 2967)
  family B: -xdir (enhancement side; sep expected to rise,
            no switch crossing expected)
Captures: o_proj input all layers (C34 kept), v_proj
L17+L34 (A11 recovery), final-norm input.

Anchors (frozen):
  a1 dirs_word vs 2927 < 1e-5
  a2 determinism rel < 1e-4
  a3 Vt8 vs 2939 < 1e-6
  a4 proj_func vs 2935 < 1e-4
  a5 proj_null0 vs 2935 < 1e-4
  a6 sep_func > 0
  a7 xdir self-check < 1e-9
  a8 GQA gates
  a9 A11_L17 vs 2953 < 1e-6
  a10 L17 recovery residual < 0.3
  a11 sep shared grid vs 2953 < 0.05
  a12 C34 curves family A vs 2967 npz bit 0 (< 1e-6)
  a13 sep_curve family A vs 2967 npz bit 0 (< 1e-6)

Tests (frozen):
  T1 (PRIMARY, family A per-word peaks):
      peak word = argmax at inner grid point (1..9);
      gate G1: n_peak >= 20/57 (pre-checked 40, disc. 10);
      continuous peak via 3-point parabola on inner argmax,
      clipped to neighbour interval; gate G2:
      |median(peak_loc over peak words) - s_c_insession|
      < 0.3 (LOCK_TOL of 2966); bootstrap 95% CI of the
      median (word resample, rng 2977 x10000) reported.
      degeneracy: flat curve (std < 1e-12) excluded and
      counted.
  T2 (family B dose response): rho(s, median-word C15_B),
      two-sided perm rng 2978 x10000, gate p <= 0.01;
      rho > 0 significant => enhancement_monotone;
      rho < 0 significant => significant_negative;
      else ns. Effect size |C15_B(end)-C15_B(0)| rel std0.
  T3 descriptive: peak-bucket histogram; per-word peak
      comparison h15 vs h8 vs h21 (L34); per-word
      A11_L34_h15 curve peak vs C15 peak (rho, descriptive);
      family B sep/B end values; A11_L17 recovery family B.

Verdict (frozen):
  anchor fail => anchor_fail_all_void
  T1 pass & T2 enhancement_monotone =>
      biphasic_direction_gated_locked
  T1 pass & T2 significant_negative =>
      biphasic_locked_biphasic_both
  T1 pass & T2 ns => biphasic_locked_no_reverse
  T1 fail => biphasic_median_artifact
  Reachability: G1 pre-checked; G2 and T2 branches all
  open (per-word peak positions never observed before).
"""
import hashlib
import json
import os
import sys
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
SRC_2953 = os.path.join(BASE, 'phase2953', 'a11_s_response',
                        'a11_s_response.npz')
SRC_2967 = os.path.join(BASE, 'phase2967',
                        'collapse_carrier_anatomy',
                        'collapse_carrier.npz')
OUT = os.path.join(BASE, 'phase2968', 'h15_peak_anatomy')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2968_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
VOCAB = 151936
SEED = 2896
LI_INJ = 17
LI_TGT = 34
H_TGT = 15
GRID17 = (0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.25,
          1.5, 2.0)
S_IDX = (0, 1, 4)
N_PERM = 10000
N_BOOT = 10000
RNG_BOOT, RNG_T2 = 2977, 2978
SEP_THRESHOLD = 100.0
LOCK_TOL = 0.3
N_PEAK_MIN = 20
BIT_TOL = 1e-6
DEGEN_EPS = 1e-12

PREREG = {
    'mode': 'two forward families protocol-identical to '
            '2966/2967: family A +xdir (replication anchor), '
            'family B -xdir (enhancement side); 57-word '
            'batch verbatim, GRID17+s0, K=1; C34 + A11 '
            'L17/L34 + final-norm captures',
    'question': 'is the h15 biphasic peak an individual '
                'property locked to s_c per word (or a '
                'median artifact), and does the shape gate '
                'on injection direction?',
    'anchors': {
        'a1': 'dirs_word vs 2927 < 1e-5',
        'a2': 'determinism rel < 1e-4',
        'a3': 'Vt8 vs 2939 < 1e-6',
        'a4': 'proj_func vs 2935 < 1e-4',
        'a5': 'proj_null0 vs 2935 < 1e-4',
        'a6': 'sep_func > 0',
        'a7': 'xdir self-check < 1e-9',
        'a8': 'GQA gates',
        'a9': 'A11_L17 vs 2953 < 1e-6',
        'a10': 'L17 recovery residual < 0.3',
        'a11': 'sep shared grid vs 2953 < 0.05',
        'a12': 'C34 family A vs 2967 npz < 1e-6',
        'a13': 'sep family A vs 2967 npz < 1e-6',
    },
    'T1': 'per-word peaks family A: gate G1 n_peak >= 20 '
          '(pre-checked 40); continuous parabola peak, '
          'gate G2 |median_peak - s_c| < 0.3; bootstrap '
          'CI rng 2977 x10000; flat-curve exclusion '
          '(std < 1e-12)',
    'T2': 'family B rho(s, median C15) two-sided perm '
          'rng 2978 x10000 p<=0.01; sign decides branch; '
          'effect size rel std0 reported',
    'T3': 'descriptive: peak buckets; h8/h21 peaks; '
          'A11_L34_h15 peak vs C15 peak rho; family B '
          'sep/B endpoints',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'T1 pass & T2 enhancement_monotone => '
               'biphasic_direction_gated_locked; T1 pass '
               '& T2 significant_negative => '
               'biphasic_locked_biphasic_both; T1 pass & '
               'T2 ns => biphasic_locked_no_reverse; '
               'T1 fail => biphasic_median_artifact',
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


def skey(li, s):
    return 'L%d|%.4f' % (li, float(s))


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


def peak_loc(s_grid, y):
    """Inner argmax -> 3-point parabola vertex; None for
    endpoint peaks."""
    am = int(np.argmax(y))
    if am == 0 or am == len(y) - 1:
        return None
    den = y[am - 1] - 2.0 * y[am] + y[am + 1]
    if abs(den) < 1e-30:
        return float(s_grid[am])
    off = 0.5 * (y[am - 1] - y[am + 1]) / den
    return float(s_grid[am] + np.clip(off, -0.5, 0.5)
                 * (s_grid[am + 1] - s_grid[am - 1]))


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2968,
                   'name': 'h15_peak_anatomy',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2887': sha8(SRC_2887),
                               's2927': sha8(SRC_2927),
                               's2935': sha8(SRC_2935),
                               's2939': sha8(SRC_2939),
                               's2953': sha8(SRC_2953),
                               's2967': sha8(SRC_2967)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'seed': SEED, 'inj_layer': LI_INJ,
                   'grid17': list(GRID17), 'k_repeat': 1,
                   'n_perm': N_PERM, 'n_boot': N_BOOT,
                   'rng': {'boot': RNG_BOOT, 'T2': RNG_T2},
                   'sep_threshold': SEP_THRESHOLD,
                   'lock_tol': LOCK_TOL,
                   'n_peak_min': N_PEAK_MIN,
                   'bit_tol': BIT_TOL,
                   'degen_eps': DEGEN_EPS,
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
    coords_39 = z39['coords'].astype(np.float64)
    conds39 = [str(s) for s in z39['cond_names']]
    dcks_39 = coords_39[conds39.index('null0')] \
        - coords_39[conds39.index('func')]
    z53 = np.load(SRC_2953, allow_pickle=True)
    z67 = np.load(SRC_2967, allow_pickle=True)
    log('sources ok', lines)

    # ---------- model ----------
    import torch
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
            assert len(ids) == 1
            tc[t] = int(ids[0])
        return tc[t]

    tid_map = {}
    for lang, ck, w in words:
        tid_map[w] = tid(w)
    func_tid = tid('the')
    batch = [[func_tid, tid_map[words[i][2]]]
             for i in range(n_words)]

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    log('model loaded', lines)

    a8_ok = bool(
        layers[0].self_attn.v_proj.weight.shape[0] == 1024
        and layers[0].self_attn.o_proj.in_features == NH * HD)
    log('a8 GQA gates ok=%s' % a8_ok, lines)

    cap_in = {'on': False, 'store': {}}
    cap_v = {}
    cap_x = {}
    fin_cap = {}
    state_fin = {'on': False}
    inj = {'li': None, 'scale': 0.0, 'vec': None}
    handles = []

    V_HOOK_LAYERS = (LI_INJ, LI_TGT)

    def pre_attn(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('hidden_states')
            if x is None or x.dim() < 2:
                return
            if inj['li'] == li and inj['vec'] is not None:
                x = x.clone()
                x[:, 1, :] = x[:, 1, :] \
                    + inj['scale'] * inj['vec']
                if args:
                    return (x,) + tuple(args[1:]), kwargs
                nkw = dict(kwargs)
                nkw['hidden_states'] = x
                return args, nkw
            if cap_in['on']:
                cap_in['store'].setdefault(li, []).append(
                    x[:, 1, :].detach().float()
                    .cpu().numpy())
            return None
        return h

    def hook_v(li):
        def h(module, args, output):
            if li in V_HOOK_LAYERS:
                o = output.detach().float().cpu().numpy()
                cap_v.setdefault(li, []).append(
                    (o[:, 0, :].copy(), o[:, 1, :].copy()))
            return None
        return h

    def hook_x(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('input')
            if x is None or x.dim() < 2:
                return None
            cap_x.setdefault(li, []).append(
                x[:, 1, :].detach().float().cpu().numpy())
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
        handles.append(layers[li].self_attn.o_proj
                       .register_forward_pre_hook(
                           hook_x(li), with_kwargs=True))
    for li in V_HOOK_LAYERS:
        handles.append(layers[li].self_attn.v_proj
                       .register_forward_hook(hook_v(li)))
    handles.append(model.model.norm.register_forward_pre_hook(
        pre_norm, with_kwargs=True))

    # ---------- pass 1: dirs_word rebuild (a1/a3) ----------
    attn_store = {}
    cap_in['on'] = True
    for i, (_, _, w) in enumerate(words):
        cap_in['store'].clear()
        with torch.no_grad():
            model(torch.tensor([[func_tid,
                                 tid_map[words[i][2]]]],
                               device='cuda'))
        for li in range(NL):
            attn_store[(i, li)] = \
                cap_in['store'][li][0].astype(np.float32)
        if (i + 1) % 20 == 0:
            log('pass1 [%d/%d]' % (i + 1, n_words), lines)
    cap_in['on'] = False

    d_dim = attn_store[(0, 0)].shape[-1]
    diffs_w = np.zeros((NL, d_dim))
    for li in range(NL):
        X = np.stack([attn_store[(i, li)]
                      for i in range(n_words)]) \
            .astype(np.float64)
        diffs_w[li] = X[lab_lang == 0].mean(0) \
            - X[lab_lang == 1].mean(0)
    dirs_word = np.stack([unit(diffs_w[li]) for li in range(NL)])
    a1_diff = float(np.abs(dirs_word - dirs_word_27).max())
    a1_ok = bool(a1_diff < 1e-5)
    log('a1 dirs rebuild diff %.2e ok=%s' % (a1_diff, a1_ok),
        lines)
    _, _, Vt = np.linalg.svd(dirs_word, full_matrices=False)
    Vt8 = Vt[:8]
    a3_diff = float(np.abs(Vt8 - Vt8_39).max())
    a3_ok = bool(a3_diff < 1e-6)
    log('a3 Vt8 vs 2939 %.2e ok=%s' % (a3_diff, a3_ok), lines)
    u35 = dirs_word[NL - 1]
    xdir = dcks_39[:, list(S_IDX)] @ Vt8[list(S_IDX)]
    a7_diff = float(np.abs(
        xdir @ Vt8[list(S_IDX)].T
        - dcks_39[:, list(S_IDX)]).max())
    a7_ok = bool(a7_diff < 1e-9)
    log('a7 xdir self-check %.2e ok=%s' % (a7_diff, a7_ok),
        lines)
    xdir_t = torch.tensor(xdir, device='cuda',
                          dtype=torch.bfloat16)
    neg_t = torch.tensor(-xdir, device='cuda',
                         dtype=torch.bfloat16)

    M = np.zeros((NL, NH * HD))
    for li in range(NL):
        Wl = layers[li].self_attn.o_proj.weight.detach() \
            .float().cpu().numpy()
        M[li] = u35 @ Wl

    def forward_batch(toks_list, scale=0.0, inj_li=None,
                      vec_t=None):
        cap_v.clear()
        cap_x.clear()
        fin_cap.pop('x', None)
        inj['li'] = inj_li
        inj['scale'] = float(scale)
        inj['vec'] = vec_t if scale else None
        state_fin['on'] = True
        with torch.no_grad():
            model(torch.tensor(toks_list, device='cuda'))
        inj['li'] = None
        inj['scale'] = 0.0
        inj['vec'] = None
        state_fin['on'] = False
        fin = fin_cap['x'].astype(np.float64)
        v = {li: (np.stack([a for a, b in cap_v[li]]),
                  np.stack([b for a, b in cap_v[li]]))
             for li in cap_v}
        x = {li: np.stack(cap_x[li])[0].astype(np.float64)
             for li in cap_x}
        return fin, v, x

    # ---------- baselines (a2/a4/a5/a6) ----------
    fin_f1, _, _ = forward_batch(batch)
    fin_f2, _, _ = forward_batch(batch)
    a2_rel = float(np.abs(fin_f1 - fin_f2).max()
                   / max(float(np.abs(fin_f1).max()), 1e-30))
    a2_ok = bool(a2_rel < 1e-4)
    log('a2 baseline determinism rel %.2e ok=%s'
        % (a2_rel, a2_ok), lines)
    proj_f0 = fin_f1 @ u35
    a4_diff = float(np.abs(proj_f0 - s_base_35[ifu35]).max())
    a4_ok = bool(a4_diff < 1e-4)
    log('a4 proj_func vs 2935 %.2e ok=%s'
        % (a4_diff, a4_ok), lines)
    sep_f = float(proj_f0[lab_lang == 0].mean()
                  - proj_f0[lab_lang == 1].mean())
    a6_ok = bool(sep_f > 0.0)
    log('a6 sep_func %.4f ok=%s' % (sep_f, a6_ok), lines)

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
    batch_n0 = [[null0_tids[i], tid_map[words[i][2]]]
                for i in range(n_words)]
    fin_n0, _, _ = forward_batch(batch_n0)
    proj_n0 = fin_n0 @ u35
    a5_diff = float(np.abs(proj_n0 - s_base_35[in035]).max())
    a5_ok = bool(a5_diff < 1e-4)
    log('a5 proj_null0 vs 2935 %.2e ok=%s'
        % (a5_diff, a5_ok), lines)

    # ---------- dose families ----------
    NKV = 1024 // HD
    HPG = NH // NKV

    def recover(Xf, v0r, v1r):
        A = np.zeros((n_words, NH))
        res = 0.0
        for hh in range(NH):
            k = hh // HPG
            d = v1r[:, k, :] - v0r[:, k, :]
            den_w = (d * d).sum(1)
            num = ((Xf[:, hh, :] - v0r[:, k, :]) * d).sum(1)
            A[:, hh] = num / np.maximum(den_w, 1e-30)
            rec = v0r[:, k, :] + A[:, hh:hh + 1] * d
            res = max(res, float(np.abs(
                rec - Xf[:, hh, :]).max()))
        return A, res

    _, v_base, x_base = forward_batch(batch)
    A11b = {}
    res17 = res34 = 0.0
    A, r = recover(
        x_base[LI_INJ].reshape(n_words, NH, HD),
        v_base[LI_INJ][0].reshape(n_words, NKV, HD),
        v_base[LI_INJ][1].reshape(n_words, NKV, HD))
    A11b[LI_INJ] = A
    res17 = r
    A, r = recover(
        x_base[LI_TGT].reshape(n_words, NH, HD),
        v_base[LI_TGT][0].reshape(n_words, NKV, HD),
        v_base[LI_TGT][1].reshape(n_words, NKV, HD))
    A11b[LI_TGT] = A
    res34 = r
    a10_ok = bool(res17 < 0.3)
    log('a10 L17 recovery residual %.2e ok=%s'
        % (res17, a10_ok), lines)

    s_grid = np.array([0.0] + [float(s) for s in GRID17])
    nS = len(s_grid)

    def run_family(vec_t):
        sep_c = {}
        A11_c = {}
        C34_c = {}
        B_c = {}
        for s in s_grid:
            fin, vv, xx = forward_batch(
                batch, scale=float(s),
                inj_li=(LI_INJ if s > 0 else None),
                vec_t=vec_t)
            P = fin @ u35
            sep_c[skey(LI_INJ, s)] = float(
                P[lab_lang == 0].mean()
                - P[lab_lang == 1].mean())
            if s > 0:
                A, r = recover(
                    xx[LI_INJ].reshape(n_words, NH, HD),
                    vv[LI_INJ][0].reshape(n_words, NKV, HD),
                    vv[LI_INJ][1].reshape(n_words, NKV, HD))
                A11_c[skey(LI_INJ, s)] = A
            A, r = recover(
                xx[LI_TGT].reshape(n_words, NH, HD),
                vv[LI_TGT][0].reshape(n_words, NKV, HD),
                vv[LI_TGT][1].reshape(n_words, NKV, HD))
            A11_c[skey(LI_TGT, s)] = A
            prof = np.zeros(NL)
            for li in range(NL):
                X = xx[li]
                xm = (X * M[li]).reshape(n_words, NH, HD)
                ch = xm.sum(axis=2)
                if li == LI_TGT:
                    C34_c[s] = ch
                prof[li] = float(ch.sum(axis=1).mean())
            B_c[s] = float(prof[6:13].mean()
                           - prof[28:36].mean())
            log('  s=%.4f sep=%.1f B=%.4f'
                % (s, sep_c[skey(LI_INJ, s)], B_c[s]), lines)
        return sep_c, A11_c, C34_c, B_c

    log('family A (+xdir)', lines)
    sepA, A11A, C34A, BA = run_family(xdir_t)
    log('family B (-xdir)', lines)
    sepB, A11B, C34B, BB = run_family(neg_t)

    # NOTE: A11 drift vs baseline is DESCRITIVE here, not
    # an anchor - injection is SUPPOSED to change A11
    # (2966: L34 h15 curve 0.35->0.45); the recovery
    # residual itself stays the pass1 baseline value.
    drift17 = max(
        float(np.abs(A11A[skey(LI_INJ, s)]
                     - A11b[LI_INJ]).max())
        for s in s_grid if s > 0)
    log('A11_L17 drift vs baseline max %.2e (descriptive, '
        'not an anchor)' % drift17, lines)

    # ---------- cross-phase anchors ----------
    a9_diff = float(np.abs(
        A11b[LI_INJ] - z53['A11b_L17']).max())
    for s in GRID17:
        a9_diff = max(a9_diff, float(np.abs(
            A11A[skey(LI_INJ, s)]
            - z53['A11_%s' % skey(LI_INJ, s)]).max()))
    a9_ok = bool(a9_diff < BIT_TOL)
    log('a9 A11_L17 vs 2953 npz (all 11 s) %.2e ok=%s'
        % (a9_diff, a9_ok), lines)
    sep53 = z53['sep_curves'].astype(np.float64)
    a11_diff = 0.0
    for j, s in enumerate(GRID17):
        a11_diff = max(a11_diff, abs(
            sepA[skey(LI_INJ, s)] - float(sep53[0, j])))
    a11_ok = bool(a11_diff < 0.05)
    log('a11 sep shared grid vs 2953 L17 row %.2e ok=%s'
        % (a11_diff, a11_ok), lines)
    C34_A = np.stack([C34A[s] for s in s_grid])
    a12_diff = float(np.abs(
        C34_A - z67['C_all'][:, LI_TGT]).max()
        / max(float(np.abs(z67['C_all'][:, LI_TGT]).max()),
              1e-30))
    a12_ok = bool(a12_diff < BIT_TOL)
    log('a12 C34 family A vs 2967 npz %.2e ok=%s'
        % (a12_diff, a12_ok), lines)
    sepA_curve = np.array([sepA[skey(LI_INJ, s)]
                           for s in s_grid])
    a13_diff = float(np.abs(
        sepA_curve - z67['sep_curve']).max()
        / max(float(np.abs(z67['sep_curve']).max()), 1e-30))
    a13_ok = bool(a13_diff < BIT_TOL)
    log('a13 sep family A vs 2967 npz %.2e ok=%s'
        % (a13_diff, a13_ok), lines)

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok and a6_ok and a7_ok and a8_ok
                     and a9_ok and a10_ok and a11_ok
                     and a12_ok and a13_ok)
    verdict = None
    t1 = t2 = t3 = None
    save = {}
    s_c = None

    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        # in-session s_c (2945 rule, family A)
        pts = sorted([(float(s), sepA[skey(LI_INJ, s)])
                      for s in s_grid])
        cross = [i for i, (sv, sp) in enumerate(pts)
                 if sp < SEP_THRESHOLD]
        if not cross:
            s_c = None
        else:
            i0 = cross[0]
            if i0 == 0:
                s_c = pts[0][0]
            else:
                s0, sp0 = pts[i0 - 1]
                s1_, sp1 = pts[i0]
                w = (sp0 - SEP_THRESHOLD) \
                    / max(sp0 - sp1, 1e-30)
                s_c = s0 + w * (s1_ - s0)
        log('in-session s_c(L17) = %s'
            % (None if s_c is None
               else round(float(s_c), 4)), lines)

        # ---------- T1 per-word peaks (family A) ----------
        C15A = C34_A[:, :, H_TGT]  # (11, 57)
        flat = C15A.std(axis=0) < DEGEN_EPS
        pk = np.array([peak_loc(s_grid, C15A[:, i])
                       for i in range(n_words)],
                      dtype=object)
        peak_words = [i for i in range(n_words)
                      if pk[i] is not None and not flat[i]]
        n_peak = len(peak_words)
        g1_ok = bool(n_peak >= N_PEAK_MIN)
        pv = np.array([float(pk[i]) for i in peak_words])
        med_peak = float(np.median(pv)) if n_peak else None
        g2_ok = bool(s_c is not None and med_peak is not None
                     and abs(med_peak - float(s_c))
                     < LOCK_TOL)
        boot_med = None
        ci_lo = ci_hi = None
        if n_peak >= 5:
            rngb = np.random.default_rng(RNG_BOOT)
            bs = np.empty(N_BOOT)
            for b in range(N_BOOT):
                idx = rngb.integers(0, n_peak, n_peak)
                bs[b] = np.median(pv[idx])
            boot_med = float(np.median(bs))
            ci_lo = float(np.percentile(bs, 2.5))
            ci_hi = float(np.percentile(bs, 97.5))
        t1_pass = bool(g1_ok and g2_ok)
        t1 = {'n_peak_words': n_peak,
              'n_flat': int(flat.sum()),
              'g1_ok': g1_ok,
              'median_peak_loc': (None if med_peak is None
                                  else round(med_peak, 4)),
              's_c_insession': (None if s_c is None
                                else round(float(s_c), 4)),
              'g2_ok': g2_ok,
              'boot_median': (None if boot_med is None
                              else round(boot_med, 4)),
              'boot_ci95': (None if ci_lo is None else
                            [round(ci_lo, 4),
                             round(ci_hi, 4)]),
              'pass': t1_pass}
        log('T1 n_peak=%d flat=%d g1=%s med_peak=%s '
            's_c=%s g2=%s CI=%s pass=%s'
            % (n_peak, int(flat.sum()), g1_ok,
               None if med_peak is None
               else round(med_peak, 4),
               None if s_c is None
               else round(float(s_c), 4), g2_ok,
               None if ci_lo is None else
               [round(ci_lo, 4), round(ci_hi, 4)],
               t1_pass), lines)

        # ---------- T2 family B dose response ----------
        C15B = np.stack(
            [C34B[s][:, H_TGT] for s in s_grid])
        curveB = np.array([float(np.median(C15B[i]))
                           for i in range(nS)])
        rho2 = spearman(s_grid, curveB)
        rng2 = np.random.default_rng(RNG_T2)
        cnt2 = 0
        for _ in range(N_PERM):
            if abs(spearman(rng2.permutation(s_grid),
                            curveB)) \
                    >= abs(rho2) - 1e-12:
                cnt2 += 1
        p2 = (cnt2 + 1) / (N_PERM + 1)
        effB = float(abs(curveB[-1] - curveB[0]))
        sd0B = float(C15B[0].std())
        if p2 > 0.01:
            t2_branch = 'ns'
        elif rho2 > 0:
            t2_branch = 'enhancement_monotone'
        else:
            t2_branch = 'significant_negative'
        t2 = {'rho': round(float(rho2), 4),
              'perm_p': float('%.3e' % p2),
              'branch': t2_branch,
              'curve': [round(float(v), 4)
                        for v in curveB],
              'effect_abs': round(effB, 4),
              'effect_rel_std0': round(
                  effB / max(sd0B, 1e-30), 4),
              'sep_B_end': round(float(
                  sepB[skey(LI_INJ, 2.0)]), 1),
              'B_B_end': round(float(BB[2.0]), 4)}
        log('T2 family B rho %.4f p %.3e branch=%s '
            'curve=%s sep_end=%.1f B_end=%.4f'
            % (rho2, p2, t2_branch, t2['curve'],
               t2['sep_B_end'], t2['B_B_end']), lines)

        # ---------- T3 descriptive ----------
        buckets = {}
        for i in peak_words:
            b = round(float(pk[i]), 2)
            buckets[b] = buckets.get(b, 0) + 1
        h8_pk = [peak_loc(s_grid, C34_A[:, i, 8])
                 for i in range(n_words)]
        h21_pk = [peak_loc(s_grid, C34_A[:, i, 21])
                  for i in range(n_words)]
        h8_inner = sum(1 for v in h8_pk if v is not None)
        h21_inner = sum(1 for v in h21_pk
                        if v is not None)
        # A11_L34_h15 per-word peak vs C15 per-word peak
        both = []
        for i in peak_words:
            ap = peak_loc(s_grid, np.stack(
                [A11A[skey(LI_TGT, s)][:, H_TGT]
                 for s in s_grid])[:, i])
            if ap is not None:
                both.append((float(pk[i]), ap))
        rho_pk = (spearman([a for a, _ in both],
                           [b for _, b in both])
                  if len(both) >= 5 else None)
        t3 = {'peak_buckets': {str(k): v for k, v
                               in sorted(buckets.items())},
              'h8_inner_peaks': h8_inner,
              'h21_inner_peaks': h21_inner,
              'A11pk_vs_C15pk_pairs': len(both),
              'A11pk_vs_C15pk_rho': (None if rho_pk is None
                                     else round(
                                         float(rho_pk), 4)),
              'familyB_A11_L17_resid': float(
                  '%.3e' % max(
                      float(np.abs(A11B[skey(LI_INJ, s)]
                                   - A11b[LI_INJ]).max())
                      for s in s_grid if s > 0))}
        log('T3 buckets=%s h8_inner=%d h21_inner=%d '
            'A11pk_rho=%s'
            % (t3['peak_buckets'], h8_inner, h21_inner,
               t3['A11pk_vs_C15pk_rho']), lines)

        if t1_pass and t2_branch == 'enhancement_monotone':
            verdict = 'biphasic_direction_gated_locked'
        elif t1_pass and t2_branch == 'significant_negative':
            verdict = 'biphasic_locked_biphasic_both'
        elif t1_pass:
            verdict = 'biphasic_locked_no_reverse'
        else:
            verdict = 'biphasic_median_artifact'

        save = {'s_grid': s_grid,
                'C34_A': C34_A,
                'C34_B': np.stack(
                    [C34B[s] for s in s_grid]),
                'A11_L34_A': np.stack(
                    [A11A[skey(LI_TGT, s)]
                     for s in s_grid]),
                'A11_L34_B': np.stack(
                    [A11B[skey(LI_TGT, s)]
                     for s in s_grid]),
                'sep_A': sepA_curve,
                'sep_B': np.array(
                    [sepB[skey(LI_INJ, s)]
                     for s in s_grid]),
                'B_A': np.array([BA[s] for s in s_grid]),
                'B_B': np.array([BB[s] for s in s_grid]),
                'labels_lang': lab_lang,
                'words': np.array(
                    ['%s:%s:%s' % w for w in words],
                    dtype=object)}

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2968, 'model': 'qwen3-4b',
           'prereg': PREREG,
           'anchors': {
               'a1_diff': float('%.3e' % a1_diff),
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
               'a8_ok': a8_ok,
               'a9_diff': float('%.3e' % a9_diff),
               'a9_ok': a9_ok,
               'a10_res17': float('%.3e' % res17),
               'a10_ok': a10_ok,
               'a11_diff': float('%.3e' % a11_diff),
               'a11_ok': a11_ok,
               'a12_diff': float('%.3e' % a12_diff),
               'a12_ok': a12_ok,
               'a13_diff': float('%.3e' % a13_diff),
               'a13_ok': a13_ok,
               'ok': anchor_ok},
           's_c_insession': (None if s_c is None
                             else round(float(s_c), 4)),
           'T1_peaks': t1, 'T2_familyB': t2,
           'T3_descriptive': t3,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if verdict != 'anchor_fail_all_void':
        np.savez_compressed(os.path.join(
            OUT, 'h15_peak.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2968 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
