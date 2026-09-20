# -*- coding: utf-8 -*-
"""Phase 2950: head-level competitive rebalancing anatomy.

Why: 2949 found that group-ablating the top-W_ov-gain
heads DEEPENS the collapse (L17 sep 21.5 -> -9.5, L16
84.8 -> 48.1) - the opposite of the single-head order of
2947. Open question: is the deepening carried by an ACTIVE
compensatory rebalancing of the remaining heads (and
downstream layers), beyond the passive loss of the
ablated heads' contribution?

Decomposition (exact identity at the injected layer's
o_proj output snapshot; per-head contribution c_h(w) =
u35 . (Wo_h @ x_h(w)), linear in proj, hence in sep):
  D_abl  = sum_{h in top5_g} sep_c_h  (direct loss when
           the promoting heads are removed; promoting
           heads push proj down, so -D_abl > 0 would
           RAISE sep)
  dSep   = sep(I1) - sep(I0)          (total change,
           ~ -30.9 / -36.7 from 2949)
  D_nonlin = dSep + D_abl             (everything not
           explained by the passive loss: remaining-head
           recomputation + downstream nonlinear response)
  Active rebalancing iff D_nonlin < 0 (pushes sep further
  down) AND |D_nonlin| > |D_abl| (indirect response
  dominates the passive loss) AND specificity gate
  |sep(B1) - sep(B0)| < 10 (no-injection ablation must
  leave the baseline readout near-intact).

Mode: ONE model, forward family. Conditions x {L17 s=1.0,
L16 s=2.0}: B1 (no inj, abl top5_g), I0 (inj, no abl),
I1 (inj, abl top5_g). K=3 same-session repeats, median.
Captures: per-head o_proj-INPUT slices at the injected
layer (post-ablation) -> c_h maps; attnin u35 projections
per layer -> rebalancing propagation profile L18..L35.

Main tests (frozen):
  T1 (L17): D_nonlin17 < 0 AND |D_nonlin17| > |D_abl17|
            AND |sep(B1_17) - sep(B0)| < 10
  T2 (L16): same with 16

Verdict (frozen):
  anchor fail      => anchor_fail_all_void
  T1 and T2        => rebalancing_compensatory
  exactly one      => rebalancing_layer_asymmetric
  else             => no_active_rebalancing

Anchors (frozen): a1 dirs rebuild < 1e-5 (continuous-
forward chain); a2 determinism < 1e-4; a3 Vt8 < 1e-6;
a4/a5 proj vs 2935 < 1e-4; a6 sep_func > 0; a7 xdir
self-check < 1e-9; a9 slice self-check + o_proj gate;
a10 I0_17 vs 2945 < 1.0; a11 I0_16 vs 2945 < 1.0; a12
group-ablation mask self-check < 1e-12; a13 ablated-head
capture-zero check (captured o_proj input: top5_g slices
exactly zero under I1); a8 same-session determinism < 1e-6.

Output: phase2950/rebalance_anatomy/.
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
SRC_2945 = os.path.join(BASE, 'phase2945', 'threshold_curves',
                        'result.json')
SRC_2948 = os.path.join(BASE, 'phase2948', 'wov_head_gain',
                        'result.json')
OUT = os.path.join(BASE, 'phase2950', 'rebalance_anatomy')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2950_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
SEED = 2896
NH, HD = 32, 128
NL = 36
VOCAB = 151936
S_IDX = (0, 1, 4)
K_REPEAT = 3
LI_SWITCH = 17
LI_GRAD = 16
S_SWITCH = 1.0
S_GRAD = 2.0
TOP5_G = {LI_SWITCH: [0, 7, 24, 22, 19],
          LI_GRAD: [13, 16, 1, 17, 6]}
SPEC_GATE = 10.0
A_REF_TOL = 1.0

PREREG = {
    'mode': 'forward family: xdir injection at L17 s=1.0 / '
            'L16 s=2.0 x head-group ablation {B1 no-inj-abl, '
            'I0 inj, I1 inj+abl top5_g}; per-head o_proj-'
            'input capture at the injected layer; K=3 '
            'same-session repeats, median',
    'question': 'is the 2949 group-ablation deepening '
                'carried by active compensatory '
                'rebalancing beyond the passive loss?',
    'top5_g_frozen': {str(k): v for k, v in TOP5_G.items()},
    'decomposition': 'D_abl = sum_{h in top5_g} sep_c_h; '
                     'dSep = sep(I1) - sep(I0); '
                     'D_nonlin = dSep + D_abl',
    'anchors': {
        'a1': 'dirs_word rebuild vs 2927 npz < 1e-5',
        'a2': 'func baseline determinism < 1e-4',
        'a3': 'Vt8 rebuild vs 2939 npz < 1e-6',
        'a4': 'proj_func vs 2935 s_base[func] < 1e-4',
        'a5': 'proj_null0 vs 2935 s_base[null0] < 1e-4',
        'a6': 'func separation > 0',
        'a7': 'xdir construction self-check < 1e-9',
        'a9': 'ablation-slice self-check < 1e-12 + o_proj '
              'in_features == 4096 gate',
        'a10': 'I0_17 sep vs 2945 L17@1.0 < 1.0',
        'a11': 'I0_16 sep vs 2945 L16@2.0 < 1.0',
        'a12': 'group-ablation mask self-check < 1e-12',
        'a13': 'ablated-head capture-zero check (I1 '
               'o_proj input: top5_g slices max abs 0)',
        'a8': 'same-session repeat determinism < 1e-6',
    },
    'T1': 'L17: D_nonlin < 0 AND |D_nonlin| > |D_abl| AND '
          '|sep(B1_17) - sep(B0)| < 10',
    'T2': 'L16: same',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'T1 and T2 => rebalancing_compensatory; '
               'exactly one => rebalancing_layer_asymmetric; '
               'else => no_active_rebalancing',
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


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2950,
                   'name': 'rebalance_anatomy',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2887': sha8(SRC_2887),
                               's2927': sha8(SRC_2927),
                               's2935': sha8(SRC_2935),
                               's2939': sha8(SRC_2939),
                               's2945': sha8(SRC_2945),
                               's2948': sha8(SRC_2948)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'seed': SEED,
                   'li_switch': LI_SWITCH, 'li_grad': LI_GRAD,
                   's_switch': S_SWITCH, 's_grad': S_GRAD,
                   'top5_g': {str(k): list(v)
                              for k, v in TOP5_G.items()},
                   'spec_gate': SPEC_GATE,
                   'a_ref_tol': A_REF_TOL,
                   'k_repeat': K_REPEAT,
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

    r45 = json.load(open(SRC_2945, encoding='utf-8'))
    sep45 = r45['D1_sep']
    r48 = json.load(open(SRC_2948, encoding='utf-8'))
    t48 = r48['D2_top5']
    assert list(t48['top5_g_L17']) == TOP5_G[LI_SWITCH]
    assert list(t48['top5_g_L16']) == TOP5_G[LI_GRAD]
    log('sources ok (top5_g re-verified against 2948)',
        lines)

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
    ocap = {'li': {}, 'on': False}
    state_fin = {'on': False}
    fin_cap = {}
    inj = {'coef': None, 'scale': 0.0, 'vec': None}
    abl = {'li': None, 'hset': None}
    handles = []

    def pre_attn(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('hidden_states')
            if x is None or x.dim() < 2:
                return
            if inj['coef'] is not None \
                    and li in inj['coef']:
                c = inj['coef'][li]
                if c != 0.0:
                    x = x.clone()
                    x[:, 1, :] = x[:, 1, :] \
                        + c * inj['scale'] * inj['vec']
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

    def pre_oproj(li):
        """Ablate head SET at o_proj input; capture the
        post-ablation input when ocap is on."""
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('input')
            if x is None or x.dim() < 2:
                return None
            hs = abl['hset'] \
                if abl['li'] == li else None
            if hs:
                x = x.clone()
                for hi in hs:
                    x[:, 1, hi * HD:(hi + 1) * HD] = 0.0
            else:
                x = x
            if ocap['on'] and li in (LI_SWITCH, LI_GRAD):
                ocap['li'].setdefault(li, []).append(
                    x[:, 1, :].detach().float()
                    .cpu().numpy())
            if hs:
                if args:
                    return (x,) + tuple(args[1:]), kwargs
                nkw = dict(kwargs)
                nkw['input'] = x
                return args, nkw
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
                           pre_oproj(li), with_kwargs=True))
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

    def forward_batch(toks_list, coef=None, scale=0.0,
                      vec=None, abl_li=None, abl_set=None,
                      capture_heads=False):
        clear_cap()
        ocap['li'] = {}
        ocap['on'] = bool(capture_heads)
        fin_cap.pop('x', None)
        state_fin['on'] = True
        inj['coef'] = coef
        inj['scale'] = float(scale)
        inj['vec'] = vec
        abl['li'] = abl_li
        abl['hset'] = abl_set
        with torch.no_grad():
            model(torch.tensor(toks_list, device='cuda'))
        inj['coef'] = None
        state_fin['on'] = False
        abl['li'] = None
        abl['hset'] = None
        ocap['on'] = False
        fin = fin_cap['x'].astype(np.float64)
        heads = {li: np.stack(v).astype(np.float64)
                 for li, v in ocap['li'].items()}
        attnin = {li: np.stack(v).astype(np.float64)
                  for li, v in cap['attnin'].items() if v}
        return fin, heads, attnin

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

    _, _, Vt = np.linalg.svd(dirs_word, full_matrices=False)
    Vt8 = Vt[:8]
    a3_diff = float(np.abs(Vt8 - Vt8_39).max())
    a3_ok = bool(a3_diff < 1e-6)
    log('a3 Vt8 rebuild vs 2939 max diff %.2e ok=%s'
        % (a3_diff, a3_ok), lines)

    u35 = dirs_word[NL - 1]

    dcks_S = dcks_39[:, list(S_IDX)]
    Vt8_S = Vt8[list(S_IDX)]
    xdir = dcks_S @ Vt8_S
    a7_diff = float(np.abs(xdir @ Vt8_S.T - dcks_S).max())
    a7_ok = bool(a7_diff < 1e-9)
    log('a7 construction self-check max abs %.2e ok=%s'
        % (a7_diff, a7_ok), lines)
    xdir_t = torch.tensor(xdir, device='cuda',
                          dtype=torch.bfloat16)

    # a9 / a12 self-checks
    rng = np.random.default_rng(7)
    syn = rng.standard_normal((2, 2, NH * HD))
    syn2 = syn.copy()
    syn2[:, 1, 5 * HD:6 * HD] = 0.0
    a9_diff = float(
        np.abs(syn2 - syn).max()
        - np.abs(syn[:, 1, 5 * HD:6 * HD]).max())
    oproj_in = layers[0].self_attn.o_proj.in_features
    a9_ok = bool(a9_diff < 1e-12
                 and oproj_in == NH * HD)
    log('a9 ablation-slice self-check %.2e | o_proj in '
        'features %d (expect %d) ok=%s'
        % (a9_diff, oproj_in, NH * HD, a9_ok), lines)

    gset = {0, 7, 30}
    syn3 = syn.copy()
    for hi in gset:
        syn3[:, 1, hi * HD:(hi + 1) * HD] = 0.0
    mask = np.ones(NH * HD, dtype=bool)
    for hi in gset:
        mask[hi * HD:(hi + 1) * HD] = False
    a12_diff = float(np.abs(
        syn3[:, 1, mask] - syn[:, 1, mask]).max())
    a12_ok = bool(a12_diff < 1e-12 and float(
        np.abs(syn3[:, 1, ~mask]).max()) == 0.0)
    log('a12 group-ablation mask self-check %.2e ok=%s'
        % (a12_diff, a12_ok), lines)

    # ---------- baselines ----------
    fin_f1, _, _ = forward_batch(batch['func'])
    fin_f2, _, _ = forward_batch(batch['func'])
    a2_rel = float(np.abs(fin_f1 - fin_f2).max()
                   / max(float(np.abs(fin_f1).max()), 1e-30))
    a2_ok = bool(a2_rel < 1e-4)
    log('a2 baseline determinism rel %.2e ok=%s'
        % (a2_rel, a2_ok), lines)

    def reads(fin):
        return fin @ u35, fin @ Vt8.T, \
            np.linalg.norm(fin, axis=1)

    proj_f0, _, _ = reads(fin_f1)
    fin_n0, _, _ = forward_batch(batch['null0'])
    proj_n0, _, _ = reads(fin_n0)

    a4_diff = float(np.abs(proj_f0 - s_base_35[ifu35]).max())
    a4_ok = bool(a4_diff < 1e-4)
    log('a4 proj_func vs 2935 max abs diff %.2e ok=%s'
        % (a4_diff, a4_ok), lines)
    a5_diff = float(np.abs(proj_n0 - s_base_35[in035]).max())
    a5_ok = bool(a5_diff < 1e-4)
    log('a5 proj_null0 vs 2935 max abs diff %.2e ok=%s'
        % (a5_diff, a5_ok), lines)
    sep_f = float(proj_f0[lab_lang == 0].mean()
                  - proj_f0[lab_lang == 1].mean())
    a6_ok = bool(sep_f > 0.0)
    log('a6 func separation %.4f ok=%s' % (sep_f, a6_ok), lines)
    sep_n = float(proj_n0[lab_lang == 0].mean()
                  - proj_n0[lab_lang == 1].mean())
    log('sep null0 %.3f (func %.3f)' % (sep_n, sep_f), lines)

    anchor_prelim = bool(a1_ok and a2_ok and a3_ok and a4_ok
                         and a5_ok and a6_ok and a7_ok
                         and a9_ok and a12_ok)

    verdict = None
    t1 = t2 = d1 = d2 = d3 = None
    save = {}
    a8_diff = a10_diff = a11_diff = None
    a8_ok = a10_ok = a11_ok = a13_ok = False
    a13_diff = None
    W_cache = {}

    if not anchor_prelim:
        verdict = 'anchor_fail_all_void'
    else:
        def sep_of(fin):
            p, _, _ = reads(fin)
            return float(p[lab_lang == 0].mean()
                         - p[lab_lang == 1].mean()), p

        def head_sep_c(xin, li):
            """xin: (K, 57, 4096) o_proj input at pos1.
            Returns per-head sep contribution, median
            over K reps: (32,)."""
            Wo = W_cache[li]
            seps = []
            for k in range(xin.shape[0]):
                X = xin[k].reshape(n_words, NH, HD)
                c = np.zeros((NH, n_words))
                for hh in range(NH):
                    oh = X[:, hh, :] \
                        @ Wo[:, hh * HD:(hh + 1) * HD].T
                    c[hh] = oh @ u35
                seps.append(
                    c[:, lab_lang == 0].mean(axis=1)
                    - c[:, lab_lang == 1].mean(axis=1))
            return np.median(np.array(seps), axis=0)

        for li in (LI_SWITCH, LI_GRAD):
            W_cache[li] = layers[li].self_attn.o_proj.weight \
                .detach().float().cpu().numpy()

        def run_cond(tag, coef, scale, abl_li, hset):
            fins, seps_l, heads_l = [], [], {}
            attn_last = None
            for _ in range(K_REPEAT):
                fin, heads, attnin = forward_batch(
                    batch['func'], coef=coef, scale=scale,
                    vec=xdir_t if coef else None,
                    abl_li=abl_li, abl_set=hset,
                    capture_heads=True)
                fins.append(sep_of(fin)[0])
                for li in (LI_SWITCH, LI_GRAD):
                    heads_l.setdefault(li, []).append(
                        heads[li][0])
                attn_last = attnin
            med = float(np.median(fins))
            log('%s median sep %.2f (reps %s)'
                % (tag, med,
                   [round(s, 1) for s in fins]), lines)
            hmed = {li: np.median(np.stack(v), axis=0)
                    for li, v in heads_l.items()}
            return med, hmed, attn_last

        # L17 family
        b1_17, h_b1_17, _ = run_cond(
            'B1_17', None, 0.0, LI_SWITCH,
            TOP5_G[LI_SWITCH])
        i0_17, h_i0_17, att_i0_17 = run_cond(
            'I0_17', {LI_SWITCH: 1.0}, S_SWITCH,
            LI_SWITCH, None)
        i1_17, h_i1_17, att_i1_17 = run_cond(
            'I1_17', {LI_SWITCH: 1.0}, S_SWITCH,
            LI_SWITCH, TOP5_G[LI_SWITCH])
        # L16 family
        b1_16, h_b1_16, _ = run_cond(
            'B1_16', None, 0.0, LI_GRAD, TOP5_G[LI_GRAD])
        i0_16, h_i0_16, att_i0_16 = run_cond(
            'I0_16', {LI_GRAD: 1.0}, S_GRAD,
            LI_GRAD, None)
        i1_16, h_i1_16, att_i1_16 = run_cond(
            'I1_16', {LI_GRAD: 1.0}, S_GRAD,
            LI_GRAD, TOP5_G[LI_GRAD])

        # a13: ablated slices exactly zero in captured I1
        zmax = 0.0
        for li, top in ((LI_SWITCH, TOP5_G[LI_SWITCH]),
                        (LI_GRAD, TOP5_G[LI_GRAD])):
            X = h_i1_17[li] if li == LI_SWITCH \
                else h_i1_16[li]
            Xr = X.reshape(n_words, NH, HD)
            for hh in top:
                zmax = max(zmax, float(
                    np.abs(Xr[:, hh, :]).max()))
        a13_diff = zmax
        a13_ok = bool(zmax == 0.0)
        log('a13 ablated-head capture-zero %.2e ok=%s'
            % (zmax, a13_ok), lines)

        # a10/a11
        a10_diff = abs(i0_17 - sep45['L17']['1.00'])
        a10_ok = bool(a10_diff < A_REF_TOL)
        a11_diff = abs(i0_16 - sep45['L16']['2.00'])
        a11_ok = bool(a11_diff < A_REF_TOL)
        log('a10 I0_17 vs 2945 diff %.4f ok=%s | '
            'a11 I0_16 vs 2945 diff %.4f ok=%s'
            % (a10_diff, a10_ok, a11_diff, a11_ok), lines)

        # a8: same-session determinism
        fin_a, _, _ = forward_batch(
            batch['func'], coef={LI_SWITCH: 1.0},
            scale=S_SWITCH, vec=xdir_t,
            abl_li=LI_SWITCH, abl_set=TOP5_G[LI_SWITCH])
        fin_b, _, _ = forward_batch(
            batch['func'], coef={LI_SWITCH: 1.0},
            scale=S_SWITCH, vec=xdir_t,
            abl_li=LI_SWITCH, abl_set=TOP5_G[LI_SWITCH])
        a8_diff = float(np.abs(fin_a - fin_b).max())
        a8_ok = bool(a8_diff < 1e-6)
        log('a8 same-session determinism (I1_17) %.2e ok=%s'
            % (a8_diff, a8_ok), lines)

        if not (anchor_prelim and a8_ok and a10_ok
                and a11_ok and a13_ok):
            verdict = 'anchor_fail_all_void'
        else:
            def family(li, sep_b1, h_b1, sep_i0, h_i0,
                       sep_i1, h_i1):
                top = TOP5_G[li]
                sc_i0 = head_sep_c(h_i0[li][None], li)
                sc_i1 = head_sep_c(h_i1[li][None], li)
                sc_b1 = head_sep_c(h_b1[li][None], li)
                d_abl = float(sum(sc_i0[h] for h in top))
                d_sep = sep_i1 - sep_i0
                d_nonlin = d_sep + d_abl
                return {
                    'D_abl': round(d_abl, 3),
                    'dSep_total': round(d_sep, 3),
                    'D_nonlin': round(d_nonlin, 3),
                    'sep_c_top5_I0':
                        {str(h): round(float(sc_i0[h]), 3)
                         for h in top},
                    'sep_c_top5_B1':
                        {str(h): round(float(sc_b1[h]), 3)
                         for h in top},
                    'sc_i0': sc_i0, 'sc_i1': sc_i1,
                    'sc_b1': sc_b1,
                    'sep_b1': sep_b1, 'sep_i0': sep_i0,
                    'sep_i1': sep_i1,
                }

            f17 = family(LI_SWITCH, b1_17, h_b1_17,
                         i0_17, h_i0_17, i1_17, h_i1_17)
            f16 = family(LI_GRAD, b1_16, h_b1_16,
                         i0_16, h_i0_16, i1_16, h_i1_16)

            def decide(fam):
                d_abl = fam['D_abl']
                d_nl = fam['D_nonlin']
                spec = abs(fam['sep_b1'] - sep_f) \
                    < SPEC_GATE
                passed = bool(d_nl < 0
                              and abs(d_nl) > abs(d_abl)
                              and spec)
                return {'D_nonlin': d_nl, 'D_abl': d_abl,
                        'spec_gate_pass': spec,
                        'threshold':
                            'D_nonlin<0, |D_nonlin|>|D_abl|, '
                            '|sep(B1)-sep(B0)|<10',
                        'pass': passed}

            t1 = decide(f17)
            t2 = decide(f16)
            log('T1 %s (D_nonlin %.2f, D_abl %.2f) | '
                'T2 %s (D_nonlin %.2f, D_abl %.2f)'
                % (t1['pass'], t1['D_nonlin'], t1['D_abl'],
                   t2['pass'], t2['D_nonlin'], t2['D_abl']),
                lines)

            if t1['pass'] and t2['pass']:
                verdict = 'rebalancing_compensatory'
            elif t1['pass'] != t2['pass']:
                verdict = 'rebalancing_layer_asymmetric'
            else:
                verdict = 'no_active_rebalancing'

            # propagation profile: attnin u35 projection
            # median per layer, I1 vs I0 (L18..L35)
            def profile(att0, att1):
                out = {}
                for li in range(LI_SWITCH + 1, NL):
                    if li not in att0 or li not in att1:
                        continue
                    p0 = att0[li][:, 0, 1, :] @ u35
                    p1 = att1[li][:, 0, 1, :] @ u35
                    out[str(li)] = round(float(
                        np.median(p1 - p0)), 3)
                return out

            d1 = {
                'L17': {'sc_I0': [round(float(v), 3)
                                  for v in f17['sc_i0']],
                        'sc_I1': [round(float(v), 3)
                                  for v in f17['sc_i1']],
                        'sc_B1': [round(float(v), 3)
                                  for v in f17['sc_b1']]},
                'L16': {'sc_I0': [round(float(v), 3)
                                  for v in f16['sc_i0']],
                        'sc_I1': [round(float(v), 3)
                                  for v in f16['sc_i1']],
                        'sc_B1': [round(float(v), 3)
                                  for v in f16['sc_b1']]},
            }
            d2 = {
                'prop_L17_I1_minus_I0':
                    profile(att_i0_17, att_i1_17),
                'prop_L16_I1_minus_I0':
                    profile(att_i0_16, att_i1_16),
                'sep_func': round(sep_f, 2),
                'sep_null0': round(sep_n, 2),
            }
            d3 = {
                'top5_g_L17': TOP5_G[LI_SWITCH],
                'top5_g_L16': TOP5_G[LI_GRAD],
                'spec_gate': SPEC_GATE,
                'seps': {'B1_17': round(b1_17, 2),
                         'I0_17': round(i0_17, 2),
                         'I1_17': round(i1_17, 2),
                         'B1_16': round(b1_16, 2),
                         'I0_16': round(i0_16, 2),
                         'I1_16': round(i1_16, 2)},
            }

            save = {
                'words': np.array(['%s:%s:%s' % w
                                   for w in words],
                                  dtype=object),
                'labels_lang': lab_lang,
                'sc_I0_L17': f17['sc_i0'],
                'sc_I1_L17': f17['sc_i1'],
                'sc_B1_L17': f17['sc_b1'],
                'sc_I0_L16': f16['sc_i0'],
                'sc_I1_L16': f16['sc_i1'],
                'sc_B1_L16': f16['sc_b1'],
                'Vt8': Vt8, 'dirs_word': dirs_word,
            }

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2950, 'model': 'qwen3-4b',
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
                       'a9_diff': float('%.3e' % a9_diff),
                       'a9_ok': a9_ok,
                       'a12_diff': float('%.3e' % a12_diff),
                       'a12_ok': a12_ok,
                       'a13_diff': None if a13_diff is None
                       else float('%.3e' % a13_diff),
                       'a13_ok': a13_ok,
                       'a8_diff': None if a8_diff is None
                       else float('%.3e' % a8_diff),
                       'a8_ok': a8_ok,
                       'a10_diff': None if a10_diff is None
                       else round(a10_diff, 4),
                       'a10_ok': a10_ok,
                       'a11_diff': None if a11_diff is None
                       else round(a11_diff, 4),
                       'a11_ok': a11_ok,
                       'ok': bool(anchor_prelim and a8_ok
                                  and a10_ok and a11_ok
                                  and a13_ok)},
           'T1_L17': None if t1 is None else {
               k: v for k, v in t1.items()},
           'T2_L16': None if t2 is None else {
               k: v for k, v in t2.items()},
           'D1_head_sep_c': d1, 'D2_profiles': d2,
           'D3_refs': d3,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(os.path.join(
            OUT, 'rebalance_anatomy.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2950 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
