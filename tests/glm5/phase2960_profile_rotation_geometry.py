# -*- coding: utf-8 -*-
"""Phase 2960: profile rotation geometry - dissect the L17
profile_rotating signature found in 2958 (cos(S_mlp(k=.5),
S_mlp(k=0)) < 0.9). Decompose the downstream S_mlp(k)
profile family into a fixed component (mean over k) plus a
deviation component, then SVD the deviations:

  T1 rotation dimensionality: energy of top-1 PC of the
     centered deviations / total >= 0.8 => rank1_rotation
     (single rotation axis) else multi_rank_rotation.
  T2 trajectory shape: per-layer linear fit in k; relative
     residual ||resid||_F / ||centered||_F <= 0.15 =>
     trajectory_linear else trajectory_curved.
  T3 fixed component: ||mean_k s_k|| / mean_k ||s_k|| >= 0.5
     => fixed_dominant else rotation_dominant.

Family agreement per axis (else mixed_<axis>); verdict
f'{t1_all}_{t2_all}_{t3_all}'.

Design (2958 verbatim): session B0(det) + {I0,I1} x {L17,L16}
K=3 + patch curve k in {0,.25,.5,.75,1.0} K=2 per family,
capture rep0. Downstream: L17 [18,35], L16 [17,35].

Anchors (2958 verbatim thresholds + a22):
  a1/a2/a3/a7/a9/a12/a13/a19/a16/a14/a15/a10/a11/a17/a18/a21
  (identical definitions to phase2958);
  a22 R_mlp(k)/sep(k)/S_mlp(k=0.5) vs 2958 npz bit-level.

Output: phase2960/profile_rotation_geometry/.
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
SRC_2939 = os.path.join(BASE, 'phase2939', 'rotation_target',
                        'rotation_target.npz')
SRC_2950 = os.path.join(BASE, 'phase2950',
                        'rebalance_anatomy', 'result.json')
SRC_2957 = os.path.join(BASE, 'phase2957',
                        'rebalance_mlp_constancy',
                        'rebalance_mlp_constancy.npz')
SRC_2957R = os.path.join(BASE, 'phase2957',
                         'rebalance_mlp_constancy',
                         'result.json')
SRC_2958 = os.path.join(BASE, 'phase2958',
                        'imprint_dose_response',
                        'imprint_dose_response.npz')
OUT = os.path.join(BASE, 'phase2960',
                   'profile_rotation_geometry')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2960_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
SEED = 2896
NH, HD = 32, 128
NL = 36
S_IDX = (0, 1, 4)
K_REPEAT = 3
LI_SWITCH = 17
LI_GRAD = 16
S_SWITCH = 1.0
S_GRAD = 2.0
TOP5_G = {LI_SWITCH: [0, 7, 24, 22, 19],
          LI_GRAD: [13, 16, 1, 17, 6]}
KS = [0.0, 0.25, 0.5, 0.75, 1.0]
E_RANK1 = 0.8
RES_LIN = 0.15
FIX_DOM = 0.5

PREREG = {
    'mode': '2958 verbatim base (same seed/batch/session '
            'structure); new analysis only: decompose the '
            'downstream S_mlp(k) profile family into fixed '
            '(mean over k) + deviation, SVD the deviations',
    'question': 'is the 2958 L17 profile rotation a rank-1 '
                'rotation about a single axis with a linear '
                'trajectory in k, and is the fixed component '
                'dominant?',
    'top5_g_frozen': {str(k): v for k, v in TOP5_G.items()},
    'region': 'downstream of dose: L17 [18,35], L16 [17,35]',
    'anchors': {
        'a1': 'dirs_word rebuild vs 2927 < 1e-5',
        'a2': 'base determinism < 1e-4',
        'a3': 'Vt8 vs 2939 < 1e-6',
        'a7': 'xdir self-check < 1e-9',
        'a8': 'all same-session repeats (K=3 conds + K=2 '
              'curve) determinism < 1e-6',
        'a9': 'ablation-slice self-check < 1e-12 + o_proj '
              'in_features == 4096',
        'a10': 'dSep_fin L17 vs 2950 < 1.0',
        'a11': 'dSep_fin L16 vs 2950 < 1.0',
        'a12': 'group-ablation mask self-check < 1e-12',
        'a13': 'ablated-head capture-zero (I1, k=None) == 0',
        'a14': 'patch closure: fin(k=1) == fin(I0) and '
               'fin(k=0) == fin(I1), both families, max abs '
               '< 1e-6',
        'a15': 'S_att/S_mlp(k=0) profiles vs 2957 npz < 1e-6 '
               '(bit); D_abl vs 2957 D1 (3dp) < 1e-3',
        'a16': 'residual identity bf16-bound ratio <= 2.0',
        'a17': 'module accounting rel < 0.01 (k=0)',
        'a18': 'causal null: C_l < 1e-12 for l < dose layer',
        'a19': 'non-ablated o_proj-input slices bit-identical '
               'I0 vs I1',
        'a21': 'passive-linear chain: |S_att[dose](k) + '
               '(1-k)*D_abl| < 1.0 all k both families',
        'a22': 'R_mlp(k), S_mlp(k=0.5) vs 2958 npz < 1e-3 '
               '(3dp-rounded storage, gate 5.01e-4*2); '
               'sep(k) vs 2958 npz < 5.01e-3 (2dp-rounded '
               'storage, discipline-2948 rounding gate)',
    },
    'T1': 'energy(top-1 PC of centered S_mlp(k) deviations) '
          '/ total >= 0.8 => rank1_rotation else '
          'multi_rank_rotation (downstream region)',
    'T2': 'per-layer linear fit in k; rel residual '
          '||resid||_F/||centered||_F <= 0.15 => '
          'trajectory_linear else trajectory_curved',
    'T3': '||mean_k s_k|| / mean_k ||s_k|| >= 0.5 => '
          'fixed_dominant else rotation_dominant',
    'verdict': 'anchor fail => anchor_fail_all_void; else '
               '{t1_all}_{t2_all}_{t3_all} (family agreement, '
               'else mixed_<axis>)',
    'correction_note': 'run1 a22 anchor fail (4.66e-03 vs '
                       '1e-6): the 2958 npz stores R_mlp_k '
                       'and S_mlp_k05 rounded to 3dp and '
                       'sep_k rounded to 2dp, so a full-'
                       'precision recompute can never match '
                       'at bit level - the observed 4.66e-03 '
                       'sits inside the 2dp rounding gate '
                       '(5e-3), exactly the discipline-2948 '
                       'trap (full-precision vs round(v,2) '
                       'json is not bit-comparable; rounding '
                       'gate 5.01e-3). run2 splits a22 into '
                       'R/S05 < 1e-3 (3dp) and sep < 5.01e-3 '
                       '(2dp). All other anchors passed in '
                       'run1 and are deterministic. Old '
                       'execution/result/npz deleted per '
                       'discipline 3; run2 is authoritative. '
                       'run2: a22 passed under the split '
                       'gates (R/S05 4.98e-04 < 1e-3, sep '
                       '4.66e-03 < 5.01e-3) but crashed in '
                       'T2 - lstsq was called with b = '
                       'Sfam.T (n_l, 5) against A (5, 2); '
                       'the correct call is lstsq(A, Sfam) '
                       'with Sfam (5, n_l), solving per-layer '
                       'coefficients jointly. No test results '
                       'were computed in run2; run3 is '
                       'authoritative.',
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
        json.dump({'phase': 2960,
                   'name': 'profile_rotation_geometry',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2887': sha8(SRC_2887),
                               's2927': sha8(SRC_2927),
                               's2939': sha8(SRC_2939),
                               's2950': sha8(SRC_2950),
                               's2957': sha8(SRC_2957),
                               's2957r': sha8(SRC_2957R),
                               's2958': sha8(SRC_2958)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'seed': SEED, 'k_repeat': K_REPEAT,
                   'li_switch': LI_SWITCH, 'li_grad': LI_GRAD,
                   's_switch': S_SWITCH, 's_grad': S_GRAD,
                   'top5_g': {str(k): list(v)
                              for k, v in TOP5_G.items()},
                   'ks': KS, 'e_rank1': E_RANK1,
                   'res_lin': RES_LIN, 'fix_dom': FIX_DOM,
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
    z39 = np.load(SRC_2939, allow_pickle=True)
    Vt8_39 = z39['Vt8'].astype(np.float64)
    coords_39 = z39['coords'].astype(np.float64)
    conds39 = [str(s) for s in z39['cond_names']]
    dcks_39 = coords_39[conds39.index('null0')] \
        - coords_39[conds39.index('func')]
    r50 = json.load(open(SRC_2950, encoding='utf-8'))
    seps50 = r50['D3_refs']['seps']
    dsep50 = {'L17': seps50['I1_17'] - seps50['I0_17'],
              'L16': seps50['I1_16'] - seps50['I0_16']}
    z57 = np.load(SRC_2957, allow_pickle=True)
    r57 = json.load(open(SRC_2957R, encoding='utf-8'))
    z58 = np.load(SRC_2958, allow_pickle=True)

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

    # ---------- hooks (2958 verbatim) ----------
    cap_ln = {'on': False, 'store': {}}
    cap_nm = {'on': False, 'store': {}}
    cap_a = {'on': False, 'store': {}}
    cap_m = {'on': False, 'store': {}}
    ocap = {'li': {}, 'on': False}
    fin_cap = {}
    state_fin = {'on': False}
    inj = {'li': None, 'scale': 0.0, 'vec': None}
    abl = {'li': None, 'hset': None, 'k': None,
           'xorig': None}
    handles = []

    def pre_iln(li):
        def h(module, args, kwargs):
            if not cap_ln['on']:
                return None
            x = args[0] if args \
                else kwargs.get('hidden_states')
            if x is None or x.dim() < 2:
                return None
            cap_ln['store'].setdefault(li, []).append(
                x[:, 1, :].detach().float()
                .cpu().numpy())
            return None
        return h

    def pre_attn(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('hidden_states')
            if x is None or x.dim() < 2:
                return
            if inj['li'] == li and inj['vec'] is not None:
                xuse = x.clone()
                xuse[:, 1, :] = xuse[:, 1, :] \
                    + inj['scale'] * inj['vec']
                if cap_nm['on']:
                    cap_nm['store'].setdefault(
                        li, []).append(
                        xuse[:, 1, :].detach().float()
                        .cpu().numpy())
                if args:
                    return (xuse,) + tuple(args[1:]), kwargs
                nkw = dict(kwargs)
                nkw['hidden_states'] = xuse
                return args, nkw
            if cap_nm['on']:
                cap_nm['store'].setdefault(li, []).append(
                    x[:, 1, :].detach().float()
                    .cpu().numpy())
            return None
        return h

    def hook_attn_out(li):
        def h(module, args, output):
            if cap_a['on']:
                o = output[0] if isinstance(output, tuple) \
                    else output
                cap_a['store'].setdefault(li, []).append(
                    o[:, 1, :].detach().float()
                    .cpu().numpy())
            return None
        return h

    def hook_mlp_out(li):
        def h(module, args, output):
            if cap_m['on']:
                o = output[0] if isinstance(output, tuple) \
                    else output
                cap_m['store'].setdefault(li, []).append(
                    o[:, 1, :].detach().float()
                    .cpu().numpy())
            return None
        return h

    def pre_oproj(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('input')
            if x is None or x.dim() < 2:
                return None
            hs = abl['hset'] \
                if abl['li'] == li else None
            if hs:
                x = x.clone()
                if abl['k'] is None:
                    for hi in hs:
                        x[:, 1, hi * HD:(hi + 1) * HD] = 0.0
                else:
                    xo = abl['xorig']
                    for hi in hs:
                        sl = slice(hi * HD, (hi + 1) * HD)
                        x[:, 1, sl] = (
                            abl['k'] * xo[:, sl]) \
                            .to(x.dtype)
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
        handles.append(layers[li].input_layernorm
                       .register_forward_pre_hook(
                           pre_iln(li), with_kwargs=True))
        handles.append(layers[li].self_attn
                       .register_forward_pre_hook(
                           pre_attn(li), with_kwargs=True))
        handles.append(layers[li].self_attn
                       .register_forward_hook(
                           hook_attn_out(li)))
        handles.append(layers[li].mlp
                       .register_forward_hook(
                           hook_mlp_out(li)))
        handles.append(layers[li].self_attn.o_proj
                       .register_forward_pre_hook(
                           pre_oproj(li), with_kwargs=True))
    handles.append(model.model.norm.register_forward_pre_hook(
        pre_norm, with_kwargs=True))

    # ---------- pass 1: dirs rebuild (a1/a3) ----------
    attn_store = {}
    inj['li'] = -1
    cap_nm['on'] = True
    for i, (_, _, w) in enumerate(words):
        with torch.no_grad():
            model(torch.tensor([[func_tid,
                                 tid_map[words[i][2]]]],
                               device='cuda'))
        for li in range(NL):
            attn_store[(i, li)] = \
                cap_nm['store'][li][-1].astype(np.float32)
        cap_nm['store'].clear()
        if (i + 1) % 20 == 0:
            log('pass1 [%d/%d]' % (i + 1, n_words), lines)
    inj['li'] = None
    cap_nm['on'] = False

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
    log('a1 dirs rebuild %.2e ok=%s' % (a1_diff, a1_ok), lines)
    _, _, Vt = np.linalg.svd(dirs_word, full_matrices=False)
    Vt8 = Vt[:8]
    a3_diff = float(np.abs(Vt8 - Vt8_39).max())
    a3_ok = bool(a3_diff < 1e-6)
    log('a3 Vt8 vs 2939 %.2e ok=%s' % (a3_diff, a3_ok), lines)
    u35 = dirs_word[NL - 1]
    dcks_S = dcks_39[:, list(S_IDX)]
    Vt8_S = Vt8[list(S_IDX)]
    xdir = dcks_S @ Vt8_S
    a7_diff = float(np.abs(xdir @ Vt8_S.T - dcks_S).max())
    a7_ok = bool(a7_diff < 1e-9)
    log('a7 xdir self-check %.2e ok=%s' % (a7_diff, a7_ok),
        lines)
    xdir_t = torch.tensor(xdir, device='cuda',
                          dtype=torch.bfloat16)

    rng = np.random.default_rng(7)
    syn = rng.standard_normal((2, 2, NH * HD))
    syn2 = syn.copy()
    syn2[:, 1, 5 * HD:6 * HD] = 0.0
    a9_diff = float(
        np.abs(syn2 - syn).max()
        - np.abs(syn[:, 1, 5 * HD:6 * HD]).max())
    oproj_in = layers[0].self_attn.o_proj.in_features
    a9_ok = bool(a9_diff < 1e-12 and oproj_in == NH * HD)
    log('a9 slice self-check %.2e | o_proj in %d ok=%s'
        % (a9_diff, oproj_in, a9_ok), lines)
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
    log('a12 group mask self-check %.2e ok=%s'
        % (a12_diff, a12_ok), lines)

    # ---------- forwards ----------
    def forward_batch(coef=None, scale=0.0, abl_li=None,
                      abl_set=None, abl_k=None, abl_xo=None,
                      capture=False):
        cap_ln['store'].clear()
        cap_a['store'].clear()
        cap_m['store'].clear()
        ocap['li'] = {}
        fin_cap.pop('x', None)
        cap_ln['on'] = capture
        cap_a['on'] = capture
        cap_m['on'] = capture
        ocap['on'] = capture
        state_fin['on'] = True
        inj['li'] = li_of(coef)
        inj['scale'] = float(scale)
        inj['vec'] = xdir_t if coef else None
        abl['li'] = abl_li
        abl['hset'] = abl_set
        abl['k'] = abl_k
        abl['xorig'] = abl_xo
        with torch.no_grad():
            model(torch.tensor(batch, device='cuda'))
        inj['li'] = None
        inj['vec'] = None
        state_fin['on'] = False
        abl['li'] = None
        abl['hset'] = None
        abl['k'] = None
        abl['xorig'] = None
        cap_ln['on'] = cap_a['on'] = cap_m['on'] = False
        ocap['on'] = False
        fin = fin_cap['x'].astype(np.float64)
        if not capture:
            return fin, None, None, None
        x = {li: np.stack(v)[0].astype(np.float64)
             for li, v in cap_ln['store'].items()}
        a = {li: np.stack(v)[0].astype(np.float64)
             for li, v in cap_a['store'].items()}
        m = {li: np.stack(v)[0].astype(np.float64)
             for li, v in cap_m['store'].items()}
        heads = {li: np.stack(v)[0].astype(np.float64)
                 for li, v in ocap['li'].items()}
        return fin, x, (a, m), heads

    def li_of(coef):
        if not coef:
            return None
        ks = [k for k, v in coef.items() if v]
        return ks[0] if ks else None

    fin_b1, _, _, _ = forward_batch()
    fin_b2, _, _, _ = forward_batch()
    a2_rel = float(np.abs(fin_b1 - fin_b2).max()
                   / max(float(np.abs(fin_b1).max()), 1e-30))
    a2_ok = bool(a2_rel < 1e-4)
    log('a2 base determinism rel %.2e ok=%s'
        % (a2_rel, a2_ok), lines)

    conds = [('I0_17', {LI_SWITCH: 1.0}, S_SWITCH, None, None),
             ('I1_17', {LI_SWITCH: 1.0}, S_SWITCH, LI_SWITCH,
              TOP5_G[LI_SWITCH]),
             ('I0_16', {LI_GRAD: 1.0}, S_GRAD, None, None),
             ('I1_16', {LI_GRAD: 1.0}, S_GRAD, LI_GRAD,
              TOP5_G[LI_GRAD])]
    runs = {}
    a8_diff = 0.0
    for cname, coef, s, ali, aset in conds:
        fins = []
        cap0 = None
        for rep in range(K_REPEAT):
            fin, x, am, heads = forward_batch(
                coef=coef, scale=s, abl_li=ali, abl_set=aset,
                capture=(rep == 0))
            fins.append(float((fin @ u35)[lab_lang == 0].mean()
                              - (fin @ u35)[lab_lang == 1]
                              .mean()))
            if rep == 0:
                cap0 = (x, am, heads, fin)
            else:
                a8_diff = max(a8_diff, float(
                    np.abs(fin - cap0[3]).max()))
        runs[cname] = {'sep_med': float(np.median(fins)),
                       'fins': fins, 'cap': cap0}
        log('%s seps %s median %.2f'
            % (cname, [round(v, 1) for v in fins],
               runs[cname]['sep_med']), lines)
    a8_ok = bool(a8_diff < 1e-6)
    log('a8 same-session rep determinism %.2e ok=%s'
        % (a8_diff, a8_ok), lines)

    # a19: non-ablated o_proj-input slices I0 vs I1 identical
    a19_diff = 0.0
    for tag, li in (('L17', LI_SWITCH), ('L16', LI_GRAD)):
        X0 = runs['I0_%d' % li]['cap'][2][li]
        X1 = runs['I1_%d' % li]['cap'][2][li]
        keep = [h for h in range(NH) if h not in TOP5_G[li]]
        for hh in keep:
            a19_diff = max(a19_diff, float(np.abs(
                X0.reshape(n_words, NH, HD)[:, hh, :]
                - X1.reshape(n_words, NH, HD)[:, hh, :])
                .max()))
    a19_ok = bool(a19_diff == 0.0)
    log('a19 non-ablated slices I0 vs I1 %.2e ok=%s'
        % (a19_diff, a19_ok), lines)

    # a13: ablated slices zero in I1 (k=None) capture
    zmax = 0.0
    for cname, li in (('I1_17', LI_SWITCH),
                      ('I1_16', LI_GRAD)):
        X = runs[cname]['cap'][2][li]
        Xr = X.reshape(n_words, NH, HD)
        for hh in TOP5_G[li]:
            zmax = max(zmax, float(np.abs(Xr[:, hh, :]).max()))
    a13_diff = zmax
    a13_ok = bool(zmax == 0.0)
    log('a13 ablated-head capture-zero %.2e ok=%s'
        % (zmax, a13_ok), lines)

    # a16: residual identity chain (I0/I1/B0)
    id_ratio = 0.0
    for cname in ('I0_17', 'I1_17', 'I0_16', 'I1_16'):
        x, (a, m), _, _ = runs[cname]['cap']
        for li in range(NL - 1):
            rec = x[li] + a[li] + m[li]
            diff = np.abs(x[li + 1] - rec)
            bound = (2.0 ** -7) * (np.abs(x[li])
                                   + np.abs(a[li])
                                   + np.abs(m[li]))
            r = float((diff
                       / np.maximum(bound, 1e-30)).max())
            id_ratio = max(id_ratio, r)
    fin_b0, x_b0, am_b0, _ = forward_batch(capture=True)
    for li in range(NL - 1):
        rec = x_b0[li] + am_b0[0][li] + am_b0[1][li]
        diff = np.abs(x_b0[li + 1] - rec)
        bound = (2.0 ** -7) * (np.abs(x_b0[li])
                               + np.abs(am_b0[0][li])
                               + np.abs(am_b0[1][li]))
        r = float((diff / np.maximum(bound, 1e-30)).max())
        id_ratio = max(id_ratio, r)
    a16_ok = bool(id_ratio <= 2.0)
    log('a16 identity bf16-bound ratio max %.3f ok=%s'
        % (id_ratio, a16_ok), lines)

    # ---------- per-head snapshot sc maps ----------
    Wo_cache = {li: layers[li].self_attn.o_proj.weight
                .detach().float().cpu().numpy().astype(
                    np.float64)
                for li in (LI_SWITCH, LI_GRAD)}

    def head_sep_c(xin, li):
        Wo = Wo_cache[li]
        X = xin.reshape(n_words, NH, HD)
        c = np.zeros((NH, n_words))
        for hh in range(NH):
            oh = X[:, hh, :] \
                @ Wo[:, hh * HD:(hh + 1) * HD].T
            c[hh] = oh @ u35
        return c[:, lab_lang == 0].mean(axis=1) \
            - c[:, lab_lang == 1].mean(axis=1)

    sc_I0 = {'L17': head_sep_c(runs['I0_17']['cap'][2][
                                   LI_SWITCH], LI_SWITCH),
             'L16': head_sep_c(runs['I0_16']['cap'][2][
                                   LI_GRAD], LI_GRAD)}
    D_abl = {'L17': float(sum(sc_I0['L17'][h]
                              for h in TOP5_G[LI_SWITCH])),
             'L16': float(sum(sc_I0['L16'][h]
                              for h in TOP5_G[LI_GRAD]))}

    def csep(v):
        p = v @ u35
        return float(p[lab_lang == 0].mean()
                     - p[lab_lang == 1].mean())

    anchor_prelim = bool(a1_ok and a2_ok and a3_ok and a7_ok
                         and a9_ok and a12_ok and a8_ok
                         and a13_ok and a19_ok and a16_ok)

    verdict = None
    t1r = t2r = t3r = None
    d1 = d2 = d3 = None
    curves = {}
    save = {}
    a10_diff = a11_diff = a17_diff = a18_diff = None
    a17_rel = a14_diff = a15_diff = a21_diff = a22_diff = None
    a10_ok = a11_ok = a17_ok = a18_ok = False
    a14_ok = a15_ok = a21_ok = a22_ok = False

    if not anchor_prelim:
        verdict = 'anchor_fail_all_void'
    else:
        # ---------- patch curve (2958 verbatim) ----------
        for tag, li_dose in (('L17', LI_SWITCH),
                             ('L16', LI_GRAD)):
            xo_np = runs['I0_%d' % li_dose]['cap'][2][li_dose]
            xo_t = torch.tensor(xo_np, device='cuda',
                                dtype=torch.bfloat16)
            coef = {li_dose: 1.0}
            s = S_SWITCH if tag == 'L17' else S_GRAD
            ck = {}
            for k in KS:
                fins = []
                cap0 = None
                for rep in range(2):
                    fin, x, am, heads = forward_batch(
                        coef=coef, scale=s,
                        abl_li=li_dose,
                        abl_set=TOP5_G[li_dose],
                        abl_k=k, abl_xo=xo_t,
                        capture=(rep == 0))
                    fins.append(float(
                        (fin @ u35)[lab_lang == 0].mean()
                        - (fin @ u35)[lab_lang == 1]
                        .mean()))
                    if rep == 0:
                        cap0 = (x, am, heads, fin)
                    else:
                        a8_diff = max(a8_diff, float(
                            np.abs(fin - cap0[3]).max()))
                ck[k] = {'sep_med': float(np.median(fins)),
                         'fins': fins, 'cap': cap0}
                log('%s k=%.2f sep %.2f'
                    % (tag, k, ck[k]['sep_med']), lines)
            curves[tag] = ck
        a8_ok = bool(a8_diff < 1e-6)
        log('a8 (incl curve reps) determinism %.2e ok=%s'
            % (a8_diff, a8_ok), lines)
        if not a8_ok:
            verdict = 'anchor_fail_all_void'
        else:
            # a14: patch closure
            a14_diff = 0.0
            for tag, li_dose in (('L17', LI_SWITCH),
                                 ('L16', LI_GRAD)):
                a14_diff = max(a14_diff, float(np.abs(
                    curves[tag][1.0]['cap'][3]
                    - runs['I0_%d' % li_dose]['cap'][3])
                    .max()))
                a14_diff = max(a14_diff, float(np.abs(
                    curves[tag][0.0]['cap'][3]
                    - runs['I1_%d' % li_dose]['cap'][3])
                    .max()))
            a14_ok = bool(a14_diff < 1e-6)
            log('a14 patch closure %.2e ok=%s'
                % (a14_diff, a14_ok), lines)

            # a15: k=0 profiles vs 2957 npz (bit) + D_abl
            a15_diff = 0.0
            for tag, li_dose in (('L17', LI_SWITCH),
                                 ('L16', LI_GRAD)):
                a0v, (aa0, mm0), _, _ = \
                    runs['I0_%d' % li_dose]['cap']
                ak, (aak, mmk), _, _ = \
                    curves[tag][0.0]['cap']
                Sa = np.array([csep(aak[l] - aa0[l])
                               for l in range(NL)])
                Sm = np.array([csep(mmk[l] - mm0[l])
                               for l in range(NL)])
                a15_diff = max(a15_diff, float(np.abs(
                    Sa - z57['S_att_%s' % tag]).max()))
                a15_diff = max(a15_diff, float(np.abs(
                    Sm - z57['S_mlp_%s' % tag]).max()))
                a15_diff = max(a15_diff, abs(
                    round(D_abl[tag], 3)
                    - r57['D1_accounting'][tag]['D_abl']))
            a15_ok = bool(a15_diff < 1e-3)
            log('a15 k=0 profiles+D_abl vs 2957 %.2e ok=%s'
                % (a15_diff, a15_ok), lines)

            # a10/a11 vs 2950
            a10_diff = abs(runs['I1_17']['sep_med']
                           - runs['I0_17']['sep_med']
                           - dsep50['L17'])
            a11_diff = abs(runs['I1_16']['sep_med']
                           - runs['I0_16']['sep_med']
                           - dsep50['L16'])
            a10_ok = bool(a10_diff < 1.0)
            a11_ok = bool(a11_diff < 1.0)
            log('a10 dSep_fin L17 vs 2950 %.3f ok=%s | a11 '
                'L16 %.3f ok=%s'
                % (a10_diff, a10_ok, a11_diff, a11_ok),
                lines)

            # a17/a18 on k=0 (= I1)
            a17_diff = 0.0
            a17_rel = 0.0
            a18_diff = 0.0
            for tag, li_dose in (('L17', LI_SWITCH),
                                 ('L16', LI_GRAD)):
                x0, (a0, m0), _, fin0 = \
                    runs['I0_%d' % li_dose]['cap']
                x1, (a1c, m1c), _, fin1 = \
                    curves[tag][0.0]['cap']
                dsep_pre = csep(fin1 - fin0)
                S_att = np.array([csep(a1c[l] - a0[l])
                                  for l in range(NL)])
                S_mlp = np.array([csep(m1c[l] - m0[l])
                                  for l in range(NL)])
                a17_d = abs(float(S_att.sum() + S_mlp.sum())
                            - dsep_pre)
                a17_r = a17_d / max(abs(dsep_pre), 1e-30)
                a17_diff = max(a17_diff, a17_d)
                a17_rel = max(a17_rel, a17_r)
                C = S_att + S_mlp
                a18_d = float(np.abs(
                    C[:li_dose]).max())
                a18_diff = max(a18_diff, a18_d)
            a17_ok = bool(a17_rel < 0.01)
            a18_ok = bool(a18_diff < 1e-12)
            log('a17 rel %.2e ok=%s | a18 %.2e ok=%s'
                % (a17_rel, a17_ok, a18_diff, a18_ok),
                lines)

            if not (a14_ok and a15_ok and a10_ok and a11_ok
                    and a17_ok and a18_ok):
                verdict = 'anchor_fail_all_void'
            else:
                # ---------- profile family + a22 ----------
                # a21 passive-linear chain (2958 verbatim)
                a21_diff = 0.0
                for tag, li_dose in (('L17', LI_SWITCH),
                                     ('L16', LI_GRAD)):
                    x0, (a0, m0), _, _ = \
                        runs['I0_%d' % li_dose]['cap']
                    for k in KS:
                        ak, (aak, mmk), _, _ = \
                            curves[tag][k]['cap']
                        Sa = np.array(
                            [csep(aak[l] - a0[l])
                             for l in range(NL)])
                        a21_diff = max(
                            a21_diff, abs(
                                float(Sa[li_dose])
                                + (1.0 - k)
                                * D_abl[tag]))
                a21_ok = bool(a21_diff < 1.0)
                log('a21 passive-linear chain max %.3f ok=%s'
                    % (a21_diff, a21_ok), lines)

                # a22 vs 2958 npz (storage-precision gates)
                a22_diff = 0.0
                a22_sep = 0.0
                profs = {}
                for tag, li_dose in (('L17', LI_SWITCH),
                                     ('L16', LI_GRAD)):
                    x0, (a0, m0), _, _ = \
                        runs['I0_%d' % li_dose]['cap']
                    Rk = []
                    Sm05 = None
                    Sk_all = []
                    for k in KS:
                        ak, (aak, mmk), _, _ = \
                            curves[tag][k]['cap']
                        Sm = np.array(
                            [csep(mmk[l] - m0[l])
                             for l in range(NL)])
                        Rk.append(float(Sm[li_dose + 1:]
                                        .sum()))
                        Sk_all.append(Sm)
                        if k == 0.5:
                            Sm05 = Sm
                    Rk = np.array(Rk)
                    a22_diff = max(a22_diff, float(np.abs(
                        Rk - z58['R_mlp_k_%s' % tag]).max()))
                    a22_diff = max(a22_diff, float(np.abs(
                        Sm05
                        - z58['S_mlp_k05_%s' % tag]).max()))
                    seps = np.array(
                        [curves[tag][k]['sep_med']
                         for k in KS])
                    a22_sep = max(a22_sep, float(np.abs(
                        seps
                        - z58['sep_k_%s' % tag]).max()))
                    profs[tag] = {'Rk': Rk,
                                  'S': np.array(Sk_all),
                                  'seps': seps}
                a22_ok = bool(a22_diff < 1e-3
                              and a22_sep < 5.01e-3)
                log('a22 R/S05 vs 2958 %.2e (gate 1e-3) | '
                    'sep %.2e (gate 5.01e-3) ok=%s'
                    % (a22_diff, a22_sep, a22_ok), lines)

                if not (a21_ok and a22_ok):
                    verdict = 'anchor_fail_all_void'
                else:
                    # ---------- tests ----------
                    tres = {}
                    geo = {}
                    for tag, li_dose in (('L17',
                                          LI_SWITCH),
                                         ('L16',
                                          LI_GRAD)):
                        lmin = li_dose + 1
                        Sfam = profs[tag]['S'][:, lmin:]
                        n_l = Sfam.shape[1]
                        mean_v = Sfam.mean(axis=0)
                        dev = Sfam - mean_v
                        # T1: rank-1 energy of deviations
                        U, sv, Vt = np.linalg.svd(
                            dev, full_matrices=False)
                        e_tot = float((sv ** 2).sum())
                        e1 = float(sv[0] ** 2) \
                            / max(e_tot, 1e-30)
                        t1 = 'rank1_rotation' \
                            if e1 >= E_RANK1 \
                            else 'multi_rank_rotation'
                        # T2: per-layer linear fit in k
                        # Sfam (5, n_l); solve A(5,2) @ coef
                        # (2, n_l) ≈ Sfam
                        A = np.stack([np.ones(len(KS)),
                                      np.array(KS)],
                                     axis=1)
                        coef, *_ = np.linalg.lstsq(
                            A, Sfam, rcond=None)
                        fit = A @ coef
                        resid = Sfam - fit
                        cen = Sfam - mean_v[None, :]
                        rel = float(
                            np.linalg.norm(resid)
                            / max(np.linalg.norm(cen),
                                  1e-30))
                        t2 = 'trajectory_linear' \
                            if rel <= RES_LIN \
                            else 'trajectory_curved'
                        # T3: fixed component share
                        nrm = np.linalg.norm(Sfam,
                                             axis=1)
                        fshare = float(
                            np.linalg.norm(mean_v)
                            / max(nrm.mean(), 1e-30))
                        t3 = 'fixed_dominant' \
                            if fshare >= FIX_DOM \
                            else 'rotation_dominant'
                        tres[tag] = {'T1': t1, 'T2': t2,
                                     'T3': t3}
                        geo[tag] = {
                            'e1': round(e1, 4),
                            'rel_lin': round(rel, 4),
                            'fixed_share': round(fshare,
                                                 4),
                            'sv': [round(float(v), 3)
                                   for v in sv[:4]],
                            'v_rot_top_layers':
                                [int(i) for i in
                                 np.argsort(
                                     -np.abs(Vt[0]))[:5]],
                            'mean_profile':
                                [round(float(v), 3)
                                 for v in mean_v],
                            'cos_k0_k1': round(float(
                                Sfam[0] @ Sfam[-1]
                                / max(float(np.linalg.norm(
                                    Sfam[0])
                                    * np.linalg.norm(
                                    Sfam[-1])),
                                    1e-30)), 4),
                            'lmin': lmin,
                            'Rk': [round(float(v), 3)
                                   for v in
                                   profs[tag]['Rk']]}
                        log('%s e1 %.3f -> %s | rel_lin '
                            '%.3f -> %s | fixed %.3f -> '
                            '%s'
                            % (tag, e1, t1, rel, t2,
                               fshare, t3), lines)

                    def agree(key):
                        v0 = tres['L17'][key]
                        v1 = tres['L16'][key]
                        return v0 if v0 == v1 \
                            else 'mixed_' + {
                                'T1': 'rank',
                                'T2': 'trajectory',
                                'T3': 'fixed'}[key]
                    t1r = agree('T1')
                    t2r = agree('T2')
                    t3r = agree('T3')
                    verdict = '%s_%s_%s' % (t1r, t2r, t3r)
                    d1 = {tag: geo[tag]
                          for tag in ('L17', 'L16')}
                    d2 = {tag: {
                        'S_profiles_k':
                            [[round(float(v), 3)
                              for v in row]
                             for row in
                             profs[tag]['S']]}
                        for tag in ('L17', 'L16')}
                    d3 = {'note': 'deviation SVD gives the '
                                  'rotation axis in layer '
                                  'space; v_rot_top_layers '
                                  'lists the 5 layers with '
                                  'largest |Vt[0]| weight'}
                    save = {
                        'words': np.array(
                            ['%s:%s:%s' % w
                             for w in words],
                            dtype=object),
                        'labels_lang': lab_lang,
                        'R_mlp_k_L17':
                            profs['L17']['Rk'],
                        'R_mlp_k_L16':
                            profs['L16']['Rk'],
                        'S_profiles_L17':
                            profs['L17']['S'],
                        'S_profiles_L16':
                            profs['L16']['S'],
                        'sep_k_L17':
                            profs['L17']['seps'],
                        'sep_k_L16':
                            profs['L16']['seps'],
                        'geo_L17': np.array(
                            [geo['L17']['e1'],
                             geo['L17']['rel_lin'],
                             geo['L17']['fixed_share']]),
                        'geo_L16': np.array(
                            [geo['L16']['e1'],
                             geo['L16']['rel_lin'],
                             geo['L16']['fixed_share']]),
                    }

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2960, 'model': 'qwen3-4b',
           'prereg': PREREG,
           'anchors': {
               'a1_diff': float('%.3e' % a1_diff),
               'a1_ok': a1_ok,
               'a2_rel': float('%.3e' % a2_rel),
               'a2_ok': a2_ok,
               'a3_diff': float('%.3e' % a3_diff),
               'a3_ok': a3_ok,
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
               'a12_diff': float('%.3e' % a12_diff),
               'a12_ok': a12_ok,
               'a13_diff': float('%.3e' % a13_diff),
               'a13_ok': a13_ok,
               'a14_diff': None if a14_diff is None
               else float('%.3e' % a14_diff),
               'a14_ok': a14_ok,
               'a15_diff': None if a15_diff is None
               else float('%.3e' % a15_diff),
               'a15_ok': a15_ok,
               'a16_ratio': float('%.3f' % id_ratio),
               'a16_ok': a16_ok,
               'a17_diff': None if a17_diff is None
               else float('%.3e' % a17_diff),
               'a17_rel': None if a17_rel is None
               else float('%.3e' % a17_rel),
               'a17_ok': a17_ok,
               'a18_diff': None if a18_diff is None
               else float('%.3e' % a18_diff),
               'a18_ok': a18_ok,
               'a19_diff': float('%.3e' % a19_diff),
               'a19_ok': a19_ok,
               'a21_diff': None if a21_diff is None
               else float('%.3f' % a21_diff),
               'a21_ok': a21_ok,
               'a22_diff': None if a22_diff is None
               else float('%.3e' % a22_diff),
               'a22_sep': None if a22_sep is None
               else float('%.3e' % a22_sep),
               'a22_ok': a22_ok,
               'ok': bool(anchor_prelim and a8_ok
                          and a10_ok and a11_ok
                          and a14_ok and a15_ok
                          and a17_ok and a18_ok
                          and a21_ok and a22_ok)},
           'T1_rank': t1r, 'T2_trajectory': t2r,
           'T3_fixed': t3r,
           'geometry': d1, 'S_profiles': d2,
           'D3_axis_note': d3,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(os.path.join(
            OUT, 'profile_rotation_geometry.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2960 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
