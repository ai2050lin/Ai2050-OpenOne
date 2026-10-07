"""Phase 2988 (plan v4 P0): domain-of-validity census.

Retests two forward cards of the 34-card primitive set under
L16 neutral context, plus the a1 direction anchor:

  Card M2968 (h15 peak locked to s_c): same 2953/2968 injection
  instrument (xdir at L17 word position, GRID17 + 0), two arms:
  L2 (verbatim 2968, bit-level anchors) and L16N (14 neutral
  filler tokens prepended, 2987 filler). T1 = per-word C34/h15
  peaks + lock to in-session s_c (2945 rule sep<100).

  Card M2940 (word-blind class axis): baseline (s=0) u35 readout
  on the 57-word 2887 list. WB1 = language class decode (expect
  significant both arms); WB2 = concept-group ICC of class-
  centered readout (blindness core; 22 cross-lang groups cover
  57/57 words, reachability pre-checked). Blind = p >= 0.05.

Verdict = T1 branch __ WB branch (enumerated in prereg).
Census card set v2 assembled in closeout.

Discipline: execution.json frozen before any forward; anchors
pre-initialized on the anchor_fail path; verdict assigned in
every branch; npz arrays pre-initialized; JSON scalars cast.
"""

import hashlib
import io
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
SRC_2953 = os.path.join(BASE, 'phase2953', 'a11_s_response',
                        'a11_s_response.npz')
SRC_2967 = os.path.join(BASE, 'phase2967',
                        'collapse_carrier_anatomy',
                        'collapse_carrier.npz')
SRC_2968 = os.path.join(BASE, 'phase2968', 'h15_peak_anatomy',
                        'h15_peak.npz')
OUT = os.path.join(BASE, 'phase2988', 'context_domain_census')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'

NH, HD = 32, 128
NL = 36
VOCAB = 151936
LI_INJ = 17
LI_TGT = 34
H_TGT = 15
GRID17 = (0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0, 1.25,
          1.5, 2.0)
S_IDX = (0, 1, 4)
N_PERM = 10000
N_BOOT = 10000
RNG_BOOT, RNG_WB1, RNG_WB2 = 2977, 2998, 2999
SEP_THRESHOLD = 100.0
LOCK_TOL = 0.3
N_PEAK_MIN = 20
BIT_TOL = 1e-6
DEGEN_EPS = 1e-12
ICC_P_GATE = 0.05
FILLER_NEUTRAL = (' The sun rises in the east and sets in '
                  'the west .')


def sha8(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]


def log(msg, lines):
    lines.append('[%s] %s' % (time.strftime('%H:%M:%S'), msg))
    with open(os.path.join(OUT, 'run_log.txt'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-30)


def skey(li, s):
    return 'L%d|%.4f' % (li, float(s))


def rankdata(x):
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and sx[j] == sx[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1.0
        i = j
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
    am = int(np.argmax(y))
    if am == 0 or am == len(y) - 1:
        return None
    den = y[am - 1] - 2.0 * y[am] + y[am + 1]
    if abs(den) < 1e-30:
        return float(s_grid[am])
    off = 0.5 * (y[am - 1] - y[am + 1]) / den
    off = float(np.clip(off, -0.5, 0.5))
    return float(s_grid[am] + off * (s_grid[am + 1]
                                     - s_grid[am - 1]))


def crossing(pts, thr):
    """2945 rule: first downward crossing of thr on sorted
    (s, sep) points, linear interpolation; None if never."""
    cross = [i for i, (sv, sp) in enumerate(pts) if sp < thr]
    if not cross:
        return None
    i0 = cross[0]
    if i0 == 0:
        return float(pts[0][0])
    s0, sp0 = pts[i0 - 1]
    s1, sp1 = pts[i0]
    w = (sp0 - thr) / max(sp0 - sp1, 1e-30)
    return float(s0 + w * (s1 - s0))


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)

    # ---------- word list ----------
    z87 = np.load(SRC_2887, allow_pickle=True)
    words = [tuple(str(w).split(':')) for w in z87['words']]
    lab_lang = np.asarray(z87['labels_lang']).astype(int)
    n_words = len(words)
    assert n_words == 57, 'word count drift'
    cks = [w[1] for w in words]
    uniq_cks = sorted(set(cks))
    groups = [np.array([i for i in range(n_words)
                        if cks[i] == u]) for u in uniq_cks
              if sum(1 for i in range(n_words)
                     if cks[i] == u) >= 2]
    covered = sum(len(g) for g in groups)
    assert len(groups) >= 10 and covered == n_words, \
        'WB2 reachability fail'

    prereg = {
        'design': 'two arms x 57-word 2887 list; arm L2 = '
                  'batch [func,word] verbatim 2968; arm L16N '
                  '= 14 neutral filler tokens (2987 sentence) '
                  '+ [func,word]; injection xdir at L17 word '
                  'position (last-token indexing, bit-equal '
                  'on L2 by construction), s_grid = 0 + '
                  'GRID17, families A(+xdir)/B(-xdir)',
        'cards_retested': 'M2968 peak-lock (T1), M2940 '
                          'word-blind (T2), M2936/M2939 '
                          'direction anchors (a1/a3)',
        'T1': 'per-word C34/h15 peaks family A per arm: '
              'inner-argmax parabola, flat exclusion '
              'std<1e-12; gate G1 n_peak >= %d; in-session '
              's_c = 2945 rule sep<%.1f; gate G2 '
              '|median_peak - s_c| < %.1f; bootstrap CI '
              'rng %d x%d; branches: lock (G1&G2), '
              'unlocked (G1 only), artifact (!G1), '
              'threshold_absent (s_c None)'
              % (N_PEAK_MIN, SEP_THRESHOLD, LOCK_TOL,
                 RNG_BOOT, N_BOOT),
        'T2': 'baseline s=0 u35 readout per arm; WB1 lang '
              'median-difference two-sided perm rng %d x%d '
              '(secondary, expect significant both arms); '
              'WB2 PRIMARY: ICC of class-centered readout '
              'over %d ck groups (size>=2, cover %d words); '
              'null = within-class permutation rng %d x%d; '
              'blind = p >= %.2f; branches: blind_both / '
              'structure_L16 / structure_L2 / '
              'structure_both'
              % (RNG_WB1, N_PERM, len(groups), covered,
                 RNG_WB2, N_PERM, ICC_P_GATE),
        'T3': 'descriptive: s_c per arm, sep curve '
              'endpoints, B band endpoints, peak buckets',
        'verdict': 'anchor fail => anchor_fail_all_void; '
                   'else composite T1branch__WBbranch, all '
                   '16 combos enumerated as strings',
        'anchors': {
            'a1': 'dirs_word rebuild vs 2927 < 1e-5',
            'a2': 'L2 determinism rel < 1e-4',
            'a3': 'Vt8 vs 2939 < 1e-6',
            'a4': 'proj_func vs 2935 < 1e-4',
            'a5': 'proj_null0 vs 2935 < 1e-4',
            'a6': 'sep_func > 0',
            'a7': 'xdir self-check < 1e-9',
            'a8': 'GQA gates',
            'a9': 'A11_L17 L2-A vs 2953 bit < 1e-6 (all s)',
            'a10': 'L17 recovery residual < 0.3',
            'a11': 'sep L2-A shared grid vs 2953 < 0.05',
            'a12': 'C34_A L2 vs 2967 npz bit < 1e-6',
            'a13': 'sep_A L2 vs 2967 npz bit < 1e-6',
            'a14': 'C34_B L2 vs 2968 npz bit < 1e-6',
            'a15': 'sep_B L2 vs 2968 npz bit < 1e-6',
            'a16': 'L16N determinism rel < 1e-4',
            'a17': 'B_A L2 vs 2968 npz bit < 1e-6'},
        'stats': 'raw p, family = T2 4 tests (WB2 primary '
                 'declared); T1 gates are count/threshold '
                 'based',
    }

    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2988,
                   'name': 'context_domain_census',
                   'created': time.strftime(
                       '%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2887': sha8(SRC_2887),
                               's2927': sha8(SRC_2927),
                               's2935': sha8(SRC_2935),
                               's2939': sha8(SRC_2939),
                               's2953': sha8(SRC_2953),
                               's2967': sha8(SRC_2967),
                               's2968': sha8(SRC_2968)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'li_inj': LI_INJ, 'li_tgt': LI_TGT,
                   'h_tgt': H_TGT, 'grid17': list(GRID17),
                   's_idx': list(S_IDX),
                   'sep_threshold': SEP_THRESHOLD,
                   'lock_tol': LOCK_TOL,
                   'n_peak_min': N_PEAK_MIN,
                   'filler_neutral': FILLER_NEUTRAL,
                   'n_groups_wb2': len(groups),
                   'prereg': prereg},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- sources ----------
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
    z68 = np.load(SRC_2968, allow_pickle=True)
    log('sources ok', lines)

    # ---------- model ----------
    import sys
    sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
    from phase2662_symmetric_mapping_contract import \
        load_native
    from transformers import AutoTokenizer
    import torch

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
    batch_l2 = [[func_tid, tid_map[words[i][2]]]
                for i in range(n_words)]

    neutral_pool = tok(FILLER_NEUTRAL,
                       add_special_tokens=False)['input_ids']
    assert len(neutral_pool) >= 1
    n14 = [int(neutral_pool[k % len(neutral_pool)])
           for k in range(14)]
    batch_l16 = [list(n14) + [func_tid, tid_map[words[i][2]]]
                 for i in range(n_words)]
    assert all(len(r) == 16 for r in batch_l16)

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    log('model loaded (n14 head %s)' % n14[:4], lines)

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
                # word = last token in both arms (L2: idx 1)
                x[:, -1, :] = x[:, -1, :] \
                    + inj['scale'] * inj['vec']
                if args:
                    return (x,) + tuple(args[1:]), kwargs
                nkw = dict(kwargs)
                nkw['hidden_states'] = x
                return args, nkw
            if cap_in['on']:
                cap_in['store'].setdefault(li, []).append(
                    x[:, -1, :].detach().float()
                    .cpu().numpy())
            return None
        return h

    def hook_v(li):
        def h(module, args, output):
            if li in V_HOOK_LAYERS:
                o = output.detach().float().cpu().numpy()
                # (func-pos, word-pos) = (-2, -1) both arms
                cap_v.setdefault(li, []).append(
                    (o[:, -2, :].copy(), o[:, -1, :].copy()))
            return None
        return h

    def hook_x(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('input')
            if x is None or x.dim() < 2:
                return None
            cap_x.setdefault(li, []).append(
                x[:, -1, :].detach().float().cpu().numpy())
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
    log('a7 xdir self-check %.2e ok=%s'
        % (a7_diff, a7_ok), lines)
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

    # ---------- baselines ----------
    fin_f1, _, _ = forward_batch(batch_l2)
    fin_f2, _, _ = forward_batch(batch_l2)
    a2_rel = float(np.abs(fin_f1 - fin_f2).max()
                   / max(float(np.abs(fin_f1).max()), 1e-30))
    a2_ok = bool(a2_rel < 1e-4)
    log('a2 L2 determinism rel %.2e ok=%s'
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

    a8_ok = bool(
        layers[0].self_attn.v_proj.weight.shape[0] == 1024
        and layers[0].self_attn.o_proj.in_features == NH * HD)
    log('a8 GQA gates ok=%s' % a8_ok, lines)

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

    # ---------- L16N baseline + determinism ----------
    fin_g1, _, _ = forward_batch(batch_l16)
    fin_g2, _, _ = forward_batch(batch_l16)
    a16_rel = float(np.abs(fin_g1 - fin_g2).max()
                    / max(float(np.abs(fin_g1).max()), 1e-30))
    a16_ok = bool(a16_rel < 1e-4)
    log('a16 L16N determinism rel %.2e ok=%s'
        % (a16_rel, a16_ok), lines)
    proj_g0 = fin_g1 @ u35
    sep_g = float(proj_g0[lab_lang == 0].mean()
                  - proj_g0[lab_lang == 1].mean())
    log('L16N baseline sep %.4f (descriptive)' % sep_g, lines)

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

    _, v_base2, x_base2 = forward_batch(batch_l2)
    A11b2 = {}
    res17 = res34 = 0.0
    A, r = recover(
        x_base2[LI_INJ].reshape(n_words, NH, HD),
        v_base2[LI_INJ][0].reshape(n_words, NKV, HD),
        v_base2[LI_INJ][1].reshape(n_words, NKV, HD))
    A11b2[LI_INJ] = A
    res17 = r
    A, r = recover(
        x_base2[LI_TGT].reshape(n_words, NH, HD),
        v_base2[LI_TGT][0].reshape(n_words, NKV, HD),
        v_base2[LI_TGT][1].reshape(n_words, NKV, HD))
    A11b2[LI_TGT] = A
    res34 = r
    a10_ok = bool(res17 < 0.3)
    log('a10 L17 recovery residual %.2e ok=%s'
        % (res17, a10_ok), lines)

    s_grid = np.array([0.0] + [float(s) for s in GRID17])
    nS = len(s_grid)

    def run_family(batch, vec_t):
        sep_c = {}
        C34_c = {}
        B_c = {}
        A17_c = {}
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
                A, _ = recover(
                    xx[LI_INJ].reshape(n_words, NH, HD),
                    vv[LI_INJ][0].reshape(n_words, NKV, HD),
                    vv[LI_INJ][1].reshape(n_words, NKV, HD))
                A17_c[s] = A
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
        return sep_c, C34_c, B_c, A17_c

    log('family A L2 (+xdir)', lines)
    sepA2, C34A2, BA2, A17A2 = run_family(batch_l2, xdir_t)
    log('family B L2 (-xdir)', lines)
    sepB2, C34B2, BB2, A17B2 = run_family(batch_l2, neg_t)
    log('family A L16N (+xdir)', lines)
    sepA16, C34A16, BA16, A17A16 = run_family(batch_l16,
                                              xdir_t)
    log('family B L16N (-xdir)', lines)
    sepB16, C34B16, BB16, A17B16 = run_family(batch_l16,
                                              neg_t)
    C34_A16 = np.stack([C34A16[s] for s in s_grid])
    C34_B16 = np.stack([C34B16[s] for s in s_grid])

    # ---------- cross-phase anchors ----------
    a9_diff = float(np.abs(
        A11b2[LI_INJ] - z53['A11b_L17']).max())
    for s in GRID17:
        a9_diff = max(a9_diff, float(np.abs(
            A17A2[s] - z53['A11_%s' % skey(LI_INJ, s)])
            .max()))
    a9_ok = bool(a9_diff < BIT_TOL)
    log('a9 A11_L17 L2-A vs 2953 (baseline+10 s) %.2e ok=%s'
        % (a9_diff, a9_ok), lines)
    sep53 = z53['sep_curves'].astype(np.float64)
    a11_diff = 0.0
    for j, s in enumerate(GRID17):
        a11_diff = max(a11_diff, abs(
            sepA2[skey(LI_INJ, s)] - float(sep53[0, j])))
    a11_ok = bool(a11_diff < 0.05)
    log('a11 sep L2-A shared grid vs 2953 %.2e ok=%s'
        % (a11_diff, a11_ok), lines)
    C34_A2 = np.stack([C34A2[s] for s in s_grid])
    C34_B2 = np.stack([C34B2[s] for s in s_grid])
    a12_diff = float(np.abs(
        C34_A2 - z67['C_all'][:, LI_TGT]).max()
        / max(float(np.abs(z67['C_all'][:, LI_TGT]).max()),
              1e-30))
    a12_ok = bool(a12_diff < BIT_TOL)
    log('a12 C34_A L2 vs 2967 %.2e ok=%s'
        % (a12_diff, a12_ok), lines)
    sepA_curve2 = np.array([sepA2[skey(LI_INJ, s)]
                            for s in s_grid])
    a13_diff = float(np.abs(
        sepA_curve2 - z67['sep_curve']).max()
        / max(float(np.abs(z67['sep_curve']).max()), 1e-30))
    a13_ok = bool(a13_diff < BIT_TOL)
    log('a13 sep_A L2 vs 2967 %.2e ok=%s'
        % (a13_diff, a13_ok), lines)
    a14_diff = float(np.abs(
        C34_B2 - z68['C34_B']).max()
        / max(float(np.abs(z68['C34_B']).max()), 1e-30))
    a14_ok = bool(a14_diff < BIT_TOL)
    log('a14 C34_B L2 vs 2968 %.2e ok=%s'
        % (a14_diff, a14_ok), lines)
    sepB_curve2 = np.array([sepB2[skey(LI_INJ, s)]
                            for s in s_grid])
    a15_diff = float(np.abs(
        sepB_curve2 - z68['sep_B']).max()
        / max(float(np.abs(z68['sep_B']).max()), 1e-30))
    a15_ok = bool(a15_diff < BIT_TOL)
    log('a15 sep_B L2 vs 2968 %.2e ok=%s'
        % (a15_diff, a15_ok), lines)
    BA_curve2 = np.array([BA2[s] for s in s_grid])
    a17_diff = float(np.abs(
        BA_curve2 - z68['B_A']).max()
        / max(float(np.abs(z68['B_A']).max()), 1e-30))
    a17_ok = bool(a17_diff < BIT_TOL)
    log('a17 B_A L2 vs 2968 %.2e ok=%s'
        % (a17_diff, a17_ok), lines)

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok and a6_ok and a7_ok and a8_ok
                     and a9_ok and a10_ok and a11_ok
                     and a12_ok and a13_ok and a14_ok
                     and a15_ok and a16_ok and a17_ok)
    verdict = None
    t1 = {}
    t2 = {}
    t3 = {}
    s_c2 = None
    s_c16 = None

    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        # ---------- in-session s_c per arm ----------
        pts2 = sorted([(float(s), sepA2[skey(LI_INJ, s)])
                       for s in s_grid])
        pts16 = sorted([(float(s), sepA16[skey(LI_INJ, s)])
                        for s in s_grid])
        s_c2 = crossing(pts2, SEP_THRESHOLD)
        s_c16 = crossing(pts16, SEP_THRESHOLD)
        log('in-session s_c L2 = %s | L16N = %s'
            % (None if s_c2 is None else round(s_c2, 4),
               None if s_c16 is None else round(s_c16, 4)),
            lines)

        # ---------- T1 per-word peaks per arm ----------
        def t1_arm(C34_stack, s_c):
            C15 = C34_stack[:, :, H_TGT]
            flat = C15.std(axis=0) < DEGEN_EPS
            pk = np.array([peak_loc(s_grid, C15[:, i])
                           for i in range(n_words)],
                          dtype=object)
            peak_words = [i for i in range(n_words)
                          if pk[i] is not None and not flat[i]]
            n_peak = len(peak_words)
            g1 = bool(n_peak >= N_PEAK_MIN)
            pv = np.array([float(pk[i]) for i in peak_words]) \
                if n_peak else np.array([])
            med_peak = float(np.median(pv)) if n_peak else None
            boot = None
            if n_peak >= 5:
                rngb = np.random.default_rng(RNG_BOOT)
                bs = np.empty(N_BOOT)
                for b in range(N_BOOT):
                    idx = rngb.integers(0, n_peak, n_peak)
                    bs[b] = float(np.median(pv[idx]))
                boot = [float(np.percentile(bs, 2.5)),
                        float(np.percentile(bs, 97.5))]
            g2 = bool(s_c is not None
                      and med_peak is not None
                      and abs(med_peak - float(s_c))
                      < LOCK_TOL)
            if s_c is None:
                br = 'threshold_absent'
            elif g1 and g2:
                br = 'lock'
            elif g1:
                br = 'unlocked'
            else:
                br = 'artifact'
            return {'branch': br, 'n_peak': int(n_peak),
                    'med_peak': None if med_peak is None
                    else round(med_peak, 4),
                    'boot_ci': boot, 'g1': g1, 'g2': g2,
                    'peak_words': [int(i)
                                   for i in peak_words]}

        t1['L2'] = t1_arm(C34_A2, s_c2)
        t1['L16N'] = t1_arm(C34_A16, s_c16)
        log('T1 L2 %s | L16N %s'
            % (t1['L2']['branch'], t1['L16N']['branch']),
            lines)

        # ---------- T2 word-blind battery ----------
        def wb_arm(proj):
            m0 = lab_lang == 0
            m1 = lab_lang == 1
            n0 = int(m0.sum())
            obs_lang = float(np.median(proj[m0])
                             - np.median(proj[m1]))
            rng1 = np.random.default_rng(RNG_WB1)
            cnt1 = 0
            for _ in range(N_PERM):
                r = rng1.permutation(n_words)
                v = float(np.median(proj[r[:n0]])
                          - np.median(proj[r[n0:]]))
                if abs(v) >= abs(obs_lang) - 1e-12:
                    cnt1 += 1
            p_lang = (cnt1 + 1) / (N_PERM + 1)
            # class-centered residuals
            e = proj.copy()
            e[m0] = proj[m0] - np.median(proj[m0])
            e[m1] = proj[m1] - np.median(proj[m1])
            grand = float(e.mean())

            def icc(vals):
                ssb = 0.0
                for g in groups:
                    mu = float(vals[g].mean())
                    ssb += len(g) * (mu - grand) ** 2
                sst = float(((vals - grand) ** 2).sum())
                return ssb / max(sst, 1e-30)

            icc_obs = icc(e)
            rng2 = np.random.default_rng(RNG_WB2)
            cnt2 = 0
            idx0 = np.nonzero(m0)[0]
            idx1 = np.nonzero(m1)[0]
            for _ in range(N_PERM):
                ep = e.copy()
                ep[idx0] = e[rng2.permutation(idx0)]
                ep[idx1] = e[rng2.permutation(idx1)]
                if icc(ep) >= icc_obs - 1e-12:
                    cnt2 += 1
            p_icc = (cnt2 + 1) / (N_PERM + 1)
            return {'obs_lang': round(obs_lang, 4),
                    'p_lang': float('%.3e' % p_lang),
                    'icc': round(icc_obs, 4),
                    'p_icc': float('%.3e' % p_icc),
                    'n_groups': len(groups)}

        wb2 = wb_arm(proj_f0)
        wb16 = wb_arm(proj_g0)
        t2['L2'] = wb2
        t2['L16N'] = wb16
        if wb2['p_icc'] >= ICC_P_GATE \
                and wb16['p_icc'] >= ICC_P_GATE:
            wb_br = 'blind_both'
        elif wb2['p_icc'] >= ICC_P_GATE \
                and wb16['p_icc'] < ICC_P_GATE:
            wb_br = 'structure_L16'
        elif wb2['p_icc'] < ICC_P_GATE \
                and wb16['p_icc'] < ICC_P_GATE:
            wb_br = 'structure_both'
        else:
            wb_br = 'structure_L2'
        log('T2 L2 icc %.4f p %.3e | L16N icc %.4f p %.3e '
            '-> %s' % (wb2['icc'], wb2['p_icc'],
                       wb16['icc'], wb16['p_icc'], wb_br),
            lines)

        # ---------- T3 descriptive ----------
        t3 = {'s_c_L2': None if s_c2 is None
              else round(s_c2, 4),
              's_c_L16N': None if s_c16 is None
              else round(s_c16, 4),
              'sepA_endpoints_L2': [round(sepA_curve2[0], 2),
                                    round(float(sepA_curve2[-1]),
                                          2)],
              'sepA_endpoints_L16N': [
                  round(float(np.array(
                      [sepA16[skey(LI_INJ, s)]
                       for s in s_grid])[0]), 2),
                  round(float(np.array(
                      [sepA16[skey(LI_INJ, s)]
                       for s in s_grid])[-1]), 2)],
              'B_endpoints_L2': [round(BA_curve2[0], 4),
                                 round(float(BA_curve2[-1]),
                                       4)],
              'B_endpoints_L16N': [
                  round(float(np.array(
                      [BA16[s] for s in s_grid])[0]), 4),
                  round(float(np.array(
                      [BA16[s] for s in s_grid])[-1]), 4)],
              'L16N_baseline_sep': round(sep_g, 4)}
        log('T3 %s' % json.dumps(t3), lines)

        t1_br = t1['L16N']['branch']
        verdict = '%s__%s' % (t1_br, wb_br)
        log('==== VERDICT: %s ====' % verdict, lines)

    # ---------- persist ----------
    elapsed = round(time.monotonic() - t0, 1)
    result = {'phase': 2988,
              'name': 'context_domain_census',
              'final_verdict': verdict,
              'anchor_all_ok': anchor_ok,
              'anchors': {
                  'a1_dirs_diff': a1_diff, 'a1_ok': a1_ok,
                  'a2_rel': a2_rel, 'a2_ok': a2_ok,
                  'a3_diff': a3_diff, 'a3_ok': a3_ok,
                  'a4_diff': a4_diff, 'a4_ok': a4_ok,
                  'a5_diff': a5_diff, 'a5_ok': a5_ok,
                  'a6_sep': sep_f, 'a6_ok': a6_ok,
                  'a7_diff': a7_diff, 'a7_ok': a7_ok,
                  'a8_ok': a8_ok,
                  'a9_diff': a9_diff, 'a9_ok': a9_ok,
                  'a10_res': res17, 'a10_ok': a10_ok,
                  'a11_diff': a11_diff, 'a11_ok': a11_ok,
                  'a12_diff': a12_diff, 'a12_ok': a12_ok,
                  'a13_diff': a13_diff, 'a13_ok': a13_ok,
                  'a14_diff': a14_diff, 'a14_ok': a14_ok,
                  'a15_diff': a15_diff, 'a15_ok': a15_ok,
                  'a16_rel': a16_rel, 'a16_ok': a16_ok,
                  'a17_diff': a17_diff, 'a17_ok': a17_ok},
              'T1': t1, 'T2': t2, 'T3': t3,
              'elapsed_s': elapsed,
              'prereg': prereg}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    sepA16_curve = np.array([sepA16[skey(LI_INJ, s)]
                             for s in s_grid])
    sepB16_curve = np.array([sepB16[skey(LI_INJ, s)]
                             for s in s_grid])
    BA16_curve = np.array([BA16[s] for s in s_grid])
    BB16_curve = np.array([BB16[s] for s in s_grid])
    BB2_curve = np.array([BB2[s] for s in s_grid])
    save = {
        's_grid': s_grid,
        'labels_lang': lab_lang,
        'words': np.array(['%s:%s:%s' % w for w in words]),
        'proj_L2_base': proj_f0,
        'proj_L16N_base': proj_g0,
        'sep_A_L2': sepA_curve2,
        'sep_B_L2': sepB_curve2,
        'sep_A_L16N': sepA16_curve,
        'sep_B_L16N': sepB16_curve,
        'C34_A_L2': C34_A2,
        'C34_B_L2': C34_B2,
        'C34_A_L16N': C34_A16,
        'C34_B_L16N': C34_B16,
        'B_A_L2': BA_curve2,
        'B_B_L2': BB2_curve,
        'B_A_L16N': BA16_curve,
        'B_B_L16N': BB16_curve,
        'A11b_L17': A11b2[LI_INJ],
        'A11b_L34': A11b2[LI_TGT],
    }
    if verdict != 'anchor_fail_all_void':
        save['pk_L2'] = np.array(
            [float(pk) if pk is not None else np.nan
             for pk in
             [peak_loc(s_grid, C34_A2[:, :, H_TGT][:, i])
              for i in range(n_words)]])
        save['pk_L16N'] = np.array(
            [float(pk) if pk is not None else np.nan
             for pk in
             [peak_loc(s_grid, C34_A16[:, :, H_TGT][:, i])
              for i in range(n_words)]])
    np.savez_compressed(os.path.join(
        OUT, 'context_domain_census.npz'), **save)
    log('==== PHASE2988 DONE elapsed %.1fs ====' % elapsed,
        lines)
    print('OK phase2988 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()

