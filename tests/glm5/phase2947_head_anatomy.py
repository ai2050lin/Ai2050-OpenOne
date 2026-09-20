# -*- coding: utf-8 -*-
"""Phase 2947: head-level anatomy of the regime switch.

Why: 2944-2946 established the switch as a per-layer
concentration-gated threshold phenomenon with nonlinear
two-layer interference; L15/L17 are switch-type (steep
monotone), L16 gradual. Open question: at the head level,
is the switch carried by few dominant heads (localized) in
switch-type layers and spread over many heads in gradual
layers?

Mode: ONE model (qwen3-4b), forward family. Injection
xdir (v1/v2/v5, 2942/2945/2946 verbatim) at L17 s=1.0
(switch engaged) and L16 s=2.0 (gradual mid). For each head
h in 0..31, ablate head h at the injected layer (zero the
h-th 128-dim slice of the attention output at pos 1 before
o_proj) and measure sep; control layer L10 gets the same
ablation sweep while injection stays at L17 (baseline head
importance without carrying the injection). K=3 same-session
repeats, median readouts.

Per-head quantities (frozen definitions):
  C_h(L17)  = sep_REF17 - sep_A17_h   (drop of collapse
              when head h removed; >0 promotes collapse)
  Cc_h(L10) = sep_REF17 - sep_A10_h   (control baseline)
  D_h(L17)  = C_h(L17) - Cc_h(L10)    (injection-specific)
  D_h(L16)  = C_h(L16) - Cc_h(L10)
  share P   = max_h D_h / sum_h |D_h|
  effN      = (sum_h |D_h|)^2 / sum_h D_h^2
            (participation ratio; 1 => single head)

Anchors (frozen; reference values from 2945/2946):
  a1 dirs_word rebuild vs 2927 npz < 1e-5     (2.17e-08)
  a2 func baseline determinism < 1e-4         (0.0)
  a3 Vt8 rebuild vs 2939 npz < 1e-6           (0.0)
  a4 proj_func vs 2935 s_base[func] < 1e-4    (7.2e-06)
  a5 proj_null0 vs 2935 s_base[null0] < 1e-4  (6.3e-06)
  a6 func separation > 0                      (185.70)
  a7 injection construction self-check < 1e-9 (9.9e-14)
  a8 same-session repeat determinism < 1e-6   (2.8e-14)
  a9 ablation-slice self-check on synthetic tensor < 1e-12
  a10 REF17 sep vs 2945 D1_sep[L17]@1.0 < 1.0  (21.5)
  a11 REF16 sep vs 2945 D1_sep[L16]@2.0 < 1.0  (84.8)

Main tests (frozen):
  T0 effect gate: max_h D_h(L17) >= 5.0 (else
     head_effects_below_noise).
  T1 concentration: P(L17) > P(L16) + 0.05.
  T2 distribution: effN(L16) > 1.2 * effN(L17).

Verdict (frozen):
  anchor fail           => anchor_fail_all_void
  T0 fail               => head_effects_below_noise
  T1 pass AND T2 pass   => switch_head_localized
  T1 pass else          => switch_head_concentrated_only
  else                  => switch_head_diffuse

Descriptive: D1 C/Cc/D vectors per layer; D2 top-5 heads;
D3 effN table; D4 control-layer distribution.

Output: phase2947/head_anatomy/.
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
OUT = os.path.join(BASE, 'phase2947', 'head_anatomy')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2947_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
SEED = 2896
NH, HD = 32, 128
NL = 36
VOCAB = 151936
S_IDX = (0, 1, 4)
K_REPEAT = 3
LI_SWITCH = 17
LI_GRAD = 16
LI_CTRL = 10
S_SWITCH = 1.0
S_GRAD = 2.0
GATE_MIN = 5.0
SHARE_MARGIN = 0.05
EFFN_MARGIN = 1.2
A_REF_TOL = 1.0

PREREG = {
    'mode': 'forward family: xdir injection at L17 s=1.0 / '
            'L16 s=2.0 with per-head ablation (attn-output '
            'pos-1 head slice zeroed pre-o_proj), 32 heads x '
            '{inj L17, inj L16, ctrl L10}, K=3 same-session '
            'repeats, median readouts',
    'question': 'is the switch carried by few dominant heads '
                'in switch-type L17 and spread over many '
                'heads in gradual L16?',
    'anchors': {
        'a1': 'dirs_word rebuild vs 2927 npz < 1e-5',
        'a2': 'func baseline determinism < 1e-4',
        'a3': 'Vt8 rebuild vs 2939 npz < 1e-6',
        'a4': 'proj_func vs 2935 s_base[func] < 1e-4',
        'a5': 'proj_null0 vs 2935 s_base[null0] < 1e-4',
        'a6': 'func separation > 0',
        'a7': 'injection construction self-check < 1e-9',
        'a8': 'same-session repeat determinism < 1e-6',
        'a9': 'ablation-slice self-check < 1e-12',
        'a10': 'REF17 sep vs 2945 L17@1.0 < 1.0',
        'a11': 'REF16 sep vs 2945 L16@2.0 < 1.0',
    },
    'quantities': 'C_h = sep_ref - sep_abl(h); Cc from '
                  'control L10; D_h = C_h - Cc_h; '
                  'P = max D / sum|D|; effN = (sum|D|)^2 / '
                  'sum D^2',
    'T0': 'max_h D_h(L17) >= 5.0',
    'T1': 'P(L17) > P(L16) + 0.05',
    'T2': 'effN(L16) > 1.2 * effN(L17)',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'T0 fail => head_effects_below_noise; '
               'T1 and T2 => switch_head_localized; '
               'T1 only => switch_head_concentrated_only; '
               'else => switch_head_diffuse',
    'correction_note': 'run1 ablation hook operated on the '
                       'attention-module OUTPUT (2560-d '
                       'hidden AFTER o_proj; qwen3-4b '
                       'hidden=2560, not 4096) where the '
                       '32x128 head structure does not '
                       'exist: heads 0-19 were hidden-'
                       'dim slices, heads 20-31 were '
                       'empty slices (bit-level zero). '
                       'run2 moves ablation to the o_proj '
                       'INPUT (4096-d = NH*HD) via '
                       'forward_pre_hook - the correct '
                       'head structure. run2 is the '
                       'authoritative run.',
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
        json.dump({'phase': 2947,
                   'name': 'head_anatomy',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2887': sha8(SRC_2887),
                               's2927': sha8(SRC_2927),
                               's2935': sha8(SRC_2935),
                               's2939': sha8(SRC_2939),
                               's2945': sha8(SRC_2945)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'seed': SEED,
                   'li_switch': LI_SWITCH,
                   'li_grad': LI_GRAD, 'li_ctrl': LI_CTRL,
                   's_switch': S_SWITCH, 's_grad': S_GRAD,
                   'gate_min': GATE_MIN,
                   'share_margin': SHARE_MARGIN,
                   'effn_margin': EFFN_MARGIN,
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
    log('sources ok (2945 ref loaded)', lines)

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
    inj = {'coef': None, 'scale': 0.0, 'vec': None}
    abl = {'li': None, 'h': None}
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
        """Ablate head h at the o_proj INPUT (NH*HD dims)."""
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('input')
            if x is None or x.dim() < 2:
                return None
            if abl['li'] == li and abl['h'] is not None:
                hi = abl['h']
                x = x.clone()
                x[:, 1, hi * HD:(hi + 1) * HD] = 0.0
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
                      vec=None, abl_li=None, abl_h=None):
        clear_cap()
        fin_cap.pop('x', None)
        state_fin['on'] = True
        inj['coef'] = coef
        inj['scale'] = float(scale)
        inj['vec'] = vec
        abl['li'] = abl_li
        abl['h'] = abl_h
        with torch.no_grad():
            model(torch.tensor(toks_list, device='cuda'))
        inj['coef'] = None
        state_fin['on'] = False
        abl['li'] = None
        abl['h'] = None
        fin = fin_cap['x'].astype(np.float64)
        return fin

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

    # a9: ablation-slice self-check on synthetic tensor +
    # structural gate: o_proj input must be NH*HD wide
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

    # ---------- baselines ----------
    fin_f1 = forward_batch(batch['func'])
    fin_f2 = forward_batch(batch['func'])
    a2_rel = float(np.abs(fin_f1 - fin_f2).max()
                   / max(float(np.abs(fin_f1).max()), 1e-30))
    a2_ok = bool(a2_rel < 1e-4)
    log('a2 baseline determinism rel %.2e ok=%s'
        % (a2_rel, a2_ok), lines)

    def reads(fin):
        return fin @ u35, fin @ Vt8.T, \
            np.linalg.norm(fin, axis=1)

    proj_f0, c8_f0, nfin_f0 = reads(fin_f1)
    fin_n0 = forward_batch(batch['null0'])
    proj_n0, c8_n0, nfin_n0 = reads(fin_n0)

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
                         and a9_ok)

    verdict = None
    t0_ = t1 = d1 = d2 = d3 = d4 = None
    save = {}
    a10_diff = a11_diff = None
    a10_ok = a11_ok = False

    if not anchor_prelim:
        verdict = 'anchor_fail_all_void'
    else:
        def sep_of(fin):
            p, c8, nfin = reads(fin)
            return float(p[lab_lang == 0].mean()
                         - p[lab_lang == 1].mean()), p

        # references
        ref17 = []
        for _ in range(K_REPEAT):
            fin = forward_batch(batch['func'],
                                coef={LI_SWITCH: 1.0},
                                scale=S_SWITCH, vec=xdir_t)
            ref17.append(sep_of(fin)[0])
        sep_r17 = float(np.median(ref17))
        ref16 = []
        for _ in range(K_REPEAT):
            fin = forward_batch(batch['func'],
                                coef={LI_GRAD: 1.0},
                                scale=S_GRAD, vec=xdir_t)
            ref16.append(sep_of(fin)[0])
        sep_r16 = float(np.median(ref16))
        log('REF L17@%.1f sep %.2f | REF L16@%.1f sep %.2f'
            % (S_SWITCH, sep_r17, S_GRAD, sep_r16), lines)

        a10_diff = abs(sep_r17 - sep45['L17']['1.00'])
        a10_ok = bool(a10_diff < A_REF_TOL)
        a11_diff = abs(sep_r16 - sep45['L16']['2.00'])
        a11_ok = bool(a11_diff < A_REF_TOL)
        log('a10 REF17 vs 2945 diff %.4f ok=%s | '
            'a11 REF16 vs 2945 diff %.4f ok=%s'
            % (a10_diff, a10_ok, a11_diff, a11_ok), lines)

        # a8: repeat determinism (reuse ref repeats)
        fin_a = forward_batch(batch['func'],
                              coef={LI_SWITCH: 1.0},
                              scale=S_SWITCH, vec=xdir_t)
        fin_b = forward_batch(batch['func'],
                              coef={LI_SWITCH: 1.0},
                              scale=S_SWITCH, vec=xdir_t)
        a8_diff = float(np.abs(fin_a - fin_b).max())
        a8_ok = bool(a8_diff < 1e-6)
        log('a8 same-session determinism %.2e ok=%s'
            % (a8_diff, a8_ok), lines)

        if not (anchor_prelim and a8_ok and a10_ok
                and a11_ok):
            verdict = 'anchor_fail_all_void'
        else:
            def sweep(target_li, coef, scale, tag):
                out = {}
                projs_store = {}
                for hh in range(NH):
                    seps = []
                    for _ in range(K_REPEAT):
                        fin = forward_batch(
                            batch['func'], coef=coef,
                            scale=scale, vec=xdir_t,
                            abl_li=target_li, abl_h=hh)
                        seps.append(sep_of(fin)[0])
                    out[hh] = float(np.median(seps))
                log('%s sweep done (32 heads)' % tag, lines)
                return out

            s17 = sweep(LI_SWITCH, {LI_SWITCH: 1.0},
                        S_SWITCH, 'A17')
            s16 = sweep(LI_GRAD, {LI_GRAD: 1.0},
                        S_GRAD, 'A16')
            s10 = sweep(LI_CTRL, {LI_SWITCH: 1.0},
                        S_SWITCH, 'A10ctrl')

            C17 = np.array([sep_r17 - s17[h]
                            for h in range(NH)])
            C16 = np.array([sep_r16 - s16[h]
                            for h in range(NH)])
            Cc = np.array([sep_r17 - s10[h]
                           for h in range(NH)])
            D17 = C17 - Cc
            D16 = C16 - Cc

            def stats(D):
                ad = np.abs(D)
                P = float(ad.max()) / max(ad.sum(), 1e-30)
                effN = float(ad.sum() ** 2
                             / max((D ** 2).sum(), 1e-30))
                return P, effN

            P17, effN17 = stats(D17)
            P16, effN16 = stats(D16)
            maxD17 = float(D17.max())
            log('L17: P %.4f effN %.2f maxD %.2f | '
                'L16: P %.4f effN %.2f maxD %.2f | '
                'ctrl max|Cc| %.2f'
                % (P17, effN17, maxD17, P16, effN16,
                   float(D16.max()), float(np.abs(Cc).max())),
                lines)

            t0_pass = bool(maxD17 >= GATE_MIN)
            t1_pass = bool(P17 > P16 + SHARE_MARGIN)
            t2_pass = bool(effN16 > EFFN_MARGIN * effN17)
            t0_ = {'max_D17': round(maxD17, 2),
                   'gate': GATE_MIN, 'pass': t0_pass}
            t1 = {'P_L17': round(P17, 4),
                  'P_L16': round(P16, 4),
                  'margin': SHARE_MARGIN,
                  'pass': t1_pass}
            t2 = {'effN_L17': round(effN17, 3),
                  'effN_L16': round(effN16, 3),
                  'margin': EFFN_MARGIN, 'pass': t2_pass}
            log('T0 %s | T1 %s | T2 %s'
                % (t0_pass, t1_pass, t2_pass), lines)

            if not t0_pass:
                verdict = 'head_effects_below_noise'
            elif t1_pass and t2_pass:
                verdict = 'switch_head_localized'
            elif t1_pass:
                verdict = 'switch_head_concentrated_only'
            else:
                verdict = 'switch_head_diffuse'

            top17 = [int(i) for i in np.argsort(
                D17)[::-1][:5]]
            top16 = [int(i) for i in np.argsort(
                D16)[::-1][:5]]
            d1 = {
                'L17': {'C': [round(float(v), 2)
                              for v in C17],
                        'D': [round(float(v), 2)
                              for v in D17]},
                'L16': {'C': [round(float(v), 2)
                              for v in C16],
                        'D': [round(float(v), 2)
                              for v in D16]},
                'ctrl_Cc': [round(float(v), 2)
                            for v in Cc],
            }
            d2 = {'top5_L17': top17,
                  'top5_L16': top16,
                  'D_top_L17': [round(float(D17[i]), 2)
                                for i in top17],
                  'D_top_L16': [round(float(D16[i]), 2)
                                for i in top16]}
            d3 = {'P_L17': round(P17, 4),
                  'P_L16': round(P16, 4),
                  'effN_L17': round(effN17, 3),
                  'effN_L16': round(effN16, 3),
                  'ctrl_max_abs_Cc':
                      round(float(np.abs(Cc).max()), 2)}
            d4 = {'sep_ref_L17': round(sep_r17, 2),
                  'sep_ref_L16': round(sep_r16, 2),
                  'sep_func': round(sep_f, 2),
                  'sep_null0': round(sep_n, 2),
                  'note': 'C_h > 0: head promotes the '
                          'collapse (ablating restores '
                          'sep); C_h < 0: head resists'}

            save = {
                'words': np.array(['%s:%s:%s' % w
                                   for w in words],
                                  dtype=object),
                'labels_lang': lab_lang,
                'C_L17': C17, 'C_L16': C16, 'Cc': Cc,
                'D_L17': D17, 'D_L16': D16,
                'Vt8': Vt8, 'dirs_word': dirs_word,
            }

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2947, 'model': 'qwen3-4b',
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
                       'a8_diff': float('%.3e' % a8_diff),
                       'a8_ok': a8_ok,
                       'a9_diff': float('%.3e' % a9_diff),
                       'a9_ok': a9_ok,
                       'a10_diff': None if a10_diff is None
                       else round(a10_diff, 4),
                       'a10_ok': a10_ok,
                       'a11_diff': None if a11_diff is None
                       else round(a11_diff, 4),
                       'a11_ok': a11_ok,
                       'ok': bool(anchor_prelim and a8_ok
                                  and a10_ok and a11_ok)},
           'T0_gate': t0_, 'T1_share': t1,
           'T2_effn': t2,
           'D1_vectors': d1, 'D2_top5': d2,
           'D3_stats': d3, 'D4_refs': d4,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(os.path.join(
            OUT, 'head_anatomy.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2947 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
