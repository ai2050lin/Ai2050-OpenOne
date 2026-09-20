# -*- coding: utf-8 -*-
"""Phase 2957: R_mlp constancy anatomy - 2956 found the MLP
rebalancing totals nearly identical across the two dose
families (R_mlp L17 = -16.26, L16 = -16.76). Is the per-layer
MLP response profile pointwise shared across families, at
unity scale, and is it an echo of the injection response
rather than an ablation-specific interaction?

Design (2956 verbatim protocol + B0 capture):
  Conditions (func batch, 57 prompts):
    B0    (no inj, no abl, CAPTURED) - injection-free ref
    I0_17 (inj L17 s=1.0)
    I1_17 (inj L17 s=1.0 + abl top5_g L17)
    I0_16 (inj L16 s=2.0)
    I1_16 (inj L16 s=2.0 + abl top5_g L16)
  K=3 repeats for seps (a8); captures on rep 0.
  Per-layer profiles (csep vs u35):
    S_mlp_f[l] = csep(m^I1_f[l] - m^I0_f[l])   (2956 verbatim)
    S_att_f[l] = csep(a^I1_f[l] - a^I0_f[l])
    M_inj_f[l] = csep(m^I0_f[l] - m^B0[l])     (new)
    A_inj_f[l] = csep(a^I0_f[l] - a^B0[l])     (descriptive)
  Common downstream region: l >= 18 (both dose layers <= 17).

Tests (frozen):
  T1 family identity: cos(s17, s16) >= 0.95 AND spearman
     >= 0.9 => profile_shared; cos >= 0.8 => profile_partial;
     else profile_divergent (s_f = S_mlp_f[18:]).
  T2 scale: origin OLS slope b = <s17,s16>/<s16,s16>, R2 =
     cos^2; b in [0.8, 1.25] AND R2 >= 0.85 => scale_unity;
     b in [0.5, 2.0) AND R2 >= 0.85 => scale_proportional;
     else scale_divergent.
  T3 mechanism: cos(S_mlp_f[18:], M_inj_f[18:]) >= 0.8 for
     BOTH families => injection_echo; else
     ablation_specific.
Verdict (frozen): anchor fail => anchor_fail_all_void; else
  f'{t1}_{t2}_{t3}'.

Anchors (frozen):
  a1 dirs rebuild < 1e-5; a3 Vt8 < 1e-6; a7 xdir < 1e-9;
  a9 slice self-check + o_proj gate; a12 group mask;
  a2 base determinism < 1e-4; a8 K=3 rep determinism < 1e-6;
  a13 ablated-head slices zero (I1);
  a16 residual identity bf16-bound ratio <= 2.0 (5 conds);
  a17 module accounting relative < 1% (both families);
  a18 causal null C_l < 1e-12 for l < dose layer;
  a10 dSep_fin L17 vs 2950 < 1.0; a11 L16 < 1.0;
  a14 S_att/S_mlp full profiles vs 2956 npz < 1e-6 (bit);
  a15 D1 scalars (dsep_fin, D_abl, R_att, R_mlp, R_tot) vs
      2956 result.json (rounded 3dp) < 1e-3.

Output: phase2957/rebalance_mlp_constancy/.
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
SRC_2956 = os.path.join(BASE, 'phase2956',
                        'rebalance_module_localization',
                        'rebalance_module_localization.npz')
SRC_2956R = os.path.join(BASE, 'phase2956',
                         'rebalance_module_localization',
                         'result.json')
OUT = os.path.join(BASE, 'phase2957',
                   'rebalance_mlp_constancy')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2957_run_report.txt')
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
LMIN = 18
COS_SHARED = 0.95
COS_PARTIAL = 0.8
RHO_SHARED = 0.9
B_UNITY = (0.8, 1.25)
B_PROP = (0.5, 2.0)
R2_MIN = 0.85
COS_ECHO = 0.8

PREREG = {
    'mode': '2956 verbatim forward family (K=3 repeats) plus '
            'captured B0: B0 + {I0,I1} x {L17 s=1.0, L16 '
            's=2.0}; captures: input_layernorm INPUT (true '
            'residual), self_attn OUTPUT, mlp OUTPUT, pos1; '
            'o_proj input at dose layers (2950/2956 verbatim)',
    'question': 'is the near-identical R_mlp across dose '
                'families (2956: -16.26 vs -16.76) a pointwise '
                'shared per-layer profile at unity scale, and '
                'is it an echo of the injection MLP response '
                '(B0 decomposition) rather than an '
                'ablation-specific interaction?',
    'top5_g_frozen': {str(k): v for k, v in TOP5_G.items()},
    'region': 'common downstream l >= LMIN=18 (both dose '
              'layers <= 17)',
    'anchors': {
        'a1': 'dirs_word rebuild vs 2927 < 1e-5',
        'a2': 'base determinism < 1e-4',
        'a3': 'Vt8 vs 2939 < 1e-6',
        'a7': 'xdir self-check < 1e-9',
        'a8': 'same-session K=3 rep determinism < 1e-6',
        'a9': 'ablation-slice self-check < 1e-12 + o_proj '
              'in_features == 4096',
        'a10': 'dSep_fin L17 vs 2950 < 1.0',
        'a11': 'dSep_fin L16 vs 2950 < 1.0',
        'a12': 'group-ablation mask self-check < 1e-12',
        'a13': 'ablated-head capture-zero (I1) == 0',
        'a14': 'S_att/S_mlp full 36-layer profiles vs 2956 '
               'npz max abs diff < 1e-6 (bit-level expected; '
               'same batch composition, same session '
               'determinism)',
        'a15': 'D1 scalars vs 2956 result.json (3dp rounding '
               'tolerance 1e-3): dsep_fin, D_abl, R_att, '
               'R_mlp, R_tot per family',
        'a16': 'residual identity x_{l+1}=x_l+a_l+m_l, '
               'bf16-bound-normalized ratio <= 2.0, 35 links '
               'x 5 conds (rep0, incl B0)',
        'a17': 'module accounting |sum - dsep_pre| / '
               '|dsep_pre| < 0.01 (2956 v2)',
        'a18': 'causal null: C_l < 1e-12 for l < dose layer',
    },
    'T1': 'family identity on S_mlp[LMIN:]: cos >= 0.95 and '
          'spearman >= 0.9 => profile_shared; cos >= 0.8 => '
          'profile_partial; else profile_divergent',
    'T2': 'origin slope b and R2 = cos^2: b in [0.8,1.25] '
          'and R2 >= 0.85 => scale_unity; b in [0.5,2.0) and '
          'R2 >= 0.85 => scale_proportional; else '
          'scale_divergent',
    'T3': 'cos(S_mlp_f[LMIN:], M_inj_f[LMIN:]) >= 0.8 for '
          'BOTH families => injection_echo; else '
          'ablation_specific',
    'verdict': 'anchor fail => anchor_fail_all_void; else '
               '{t1}_{t2}_{t3}',
    'correction_note': 'none (first run)',
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
        json.dump({'phase': 2957,
                   'name': 'rebalance_mlp_constancy',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2887': sha8(SRC_2887),
                               's2927': sha8(SRC_2927),
                               's2939': sha8(SRC_2939),
                               's2950': sha8(SRC_2950),
                               's2956': sha8(SRC_2956),
                               's2956r': sha8(SRC_2956R)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'seed': SEED, 'k_repeat': K_REPEAT,
                   'li_switch': LI_SWITCH, 'li_grad': LI_GRAD,
                   's_switch': S_SWITCH, 's_grad': S_GRAD,
                   'top5_g': {str(k): list(v)
                              for k, v in TOP5_G.items()},
                   'lmin': LMIN,
                   'cos_shared': COS_SHARED,
                   'cos_partial': COS_PARTIAL,
                   'rho_shared': RHO_SHARED,
                   'b_unity': list(B_UNITY),
                   'b_prop': list(B_PROP),
                   'r2_min': R2_MIN,
                   'cos_echo': COS_ECHO,
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
    z56 = np.load(SRC_2956, allow_pickle=True)
    r56 = json.load(open(SRC_2956R, encoding='utf-8'))

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

    # ---------- hooks (2956 verbatim) ----------
    cap_ln = {'on': False, 'store': {}}
    cap_nm = {'on': False, 'store': {}}
    cap_a = {'on': False, 'store': {}}
    cap_m = {'on': False, 'store': {}}
    ocap = {'li': {}, 'on': False}
    fin_cap = {}
    state_fin = {'on': False}
    inj = {'li': None, 'scale': 0.0, 'vec': None}
    abl = {'li': None, 'hset': None}
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
                for hi in hs:
                    x[:, 1, hi * HD:(hi + 1) * HD] = 0.0
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
                      abl_set=None, capture=False):
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
        with torch.no_grad():
            model(torch.tensor(batch, device='cuda'))
        inj['li'] = None
        inj['vec'] = None
        state_fin['on'] = False
        abl['li'] = None
        abl['hset'] = None
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

    # B0 captured (determinism covered by a2; capture rep 0)
    fin_b0, x_b0, am_b0, _ = forward_batch(capture=True)
    runs['B0'] = {'sep_med': float(
        (fin_b0 @ u35)[lab_lang == 0].mean()
        - (fin_b0 @ u35)[lab_lang == 1].mean()),
        'fins': [fin_b0], 'cap': (x_b0, am_b0, None, fin_b0)}
    log('B0 sep %.2f' % runs['B0']['sep_med'], lines)

    # a13: ablated slices zero in I1 o_proj input
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

    # a16: residual identity chain (2956 v2, incl B0)
    id_rel = 0.0
    id_ratio = 0.0
    id_worst = None
    for cname in ('B0', 'I0_17', 'I1_17', 'I0_16', 'I1_16'):
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
            rel = float(diff.max()) \
                / max(float(np.abs(x[li + 1]).max()), 1e-30)
            if rel > id_rel:
                id_rel = rel
                id_worst = {
                    'cond': cname, 'link': li,
                    'max_abs_diff': float(diff.max()),
                    'max_x1': float(
                        np.abs(x[li + 1]).max()),
                    'max_a': float(np.abs(a[li]).max()),
                    'max_m': float(np.abs(m[li]).max()),
                    'ratio': r}
    a16_ok = bool(id_ratio <= 2.0)
    log('a16 identity bf16-bound ratio max %.3f ok=%s | '
        'raw rel max %.2e (worst %s)'
        % (id_ratio, a16_ok, id_rel,
           json.dumps(id_worst)), lines)

    # ---------- per-head snapshot sc maps (2956 verbatim) --
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
                         and a13_ok and a16_ok)

    verdict = None
    t1r = t2r = t3r = None
    d1 = d2 = d3 = None
    fams = {}
    a10_diff = a11_diff = a17_diff = a18_diff = None
    a17_rel = a14_diff = a15_diff = None
    a10_ok = a11_ok = a17_ok = a18_ok = False
    a14_ok = a15_ok = False

    if not anchor_prelim:
        verdict = 'anchor_fail_all_void'
    else:
        m_b0 = am_b0[1]
        a_b0 = am_b0[0]
        for tag, li_dose in (('L17', LI_SWITCH),
                             ('L16', LI_GRAD)):
            x0, (a0, m0), _, fin0 = runs['I0_%d' % li_dose][
                'cap']
            x1, (a1c, m1c), _, fin1 = runs['I1_%d' % li_dose][
                'cap']
            dsep_pre = csep(fin1 - fin0)
            dsep_fin = runs['I1_%d' % li_dose]['sep_med'] \
                - runs['I0_%d' % li_dose]['sep_med']
            S_att = np.array([csep(a1c[l] - a0[l])
                              for l in range(NL)])
            S_mlp = np.array([csep(m1c[l] - m0[l])
                              for l in range(NL)])
            M_inj = np.array([csep(m0[l] - m_b0[l])
                              for l in range(NL)])
            A_inj = np.array([csep(a0[l] - a_b0[l])
                              for l in range(NL)])
            R_att = float(S_att.sum() + D_abl[tag])
            R_mlp = float(S_mlp.sum())
            R_tot = R_att + R_mlp
            C = S_att + S_mlp
            fams[tag] = {
                'dsep_pre': dsep_pre, 'dsep_fin': dsep_fin,
                'S_att': S_att, 'S_mlp': S_mlp,
                'M_inj': M_inj, 'A_inj': A_inj, 'C': C,
                'R_att': R_att, 'R_mlp': R_mlp,
                'R_tot': R_tot, 'D_abl': D_abl[tag]}
            a17_d = abs(float(S_att.sum() + S_mlp.sum())
                        - dsep_pre)
            a17_r = a17_d / max(abs(dsep_pre), 1e-30)
            a17_diff = max(a17_diff or 0.0, a17_d)
            a17_rel = max(a17_rel or 0.0, a17_r)
            a18_d = float(np.abs(
                C[:li_dose]).max()) if li_dose > 0 else 0.0
            a18_diff = max(a18_diff or 0.0, a18_d)
            log('%s dSep_pre %.2f dSep_fin %.2f D_abl %.2f | '
                'R_att %.2f R_mlp %.2f R_tot %.2f | a17 %.2e '
                '(rel %.2e) a18 %.2e'
                % (tag, dsep_pre, dsep_fin, D_abl[tag],
                   R_att, R_mlp, R_tot, a17_d, a17_r,
                   a18_d), lines)
        a17_ok = bool(a17_rel is not None
                      and a17_rel < 0.01)
        a18_ok = bool(a18_diff < 1e-12)

        a10_diff = abs(fams['L17']['dsep_fin']
                       - dsep50['L17'])
        a11_diff = abs(fams['L16']['dsep_fin']
                       - dsep50['L16'])
        a10_ok = bool(a10_diff < 1.0)
        a11_ok = bool(a11_diff < 1.0)
        log('a10 dSep_fin L17 vs 2950 %.3f ok=%s | a11 L16 '
            '%.3f ok=%s' % (a10_diff, a10_ok, a11_diff,
                            a11_ok), lines)

        # a14: S_att/S_mlp vs 2956 npz (bit-level expected)
        a14_diff = max(
            float(np.abs(fams['L17']['S_att']
                         - z56['S_att_L17']).max()),
            float(np.abs(fams['L17']['S_mlp']
                         - z56['S_mlp_L17']).max()),
            float(np.abs(fams['L16']['S_att']
                         - z56['S_att_L16']).max()),
            float(np.abs(fams['L16']['S_mlp']
                         - z56['S_mlp_L16']).max()))
        a14_ok = bool(a14_diff < 1e-6)
        log('a14 S_att/S_mlp vs 2956 npz %.2e ok=%s'
            % (a14_diff, a14_ok), lines)

        # a15: D1 scalars vs 2956 result.json (3dp rounding)
        d156 = r56['D1_accounting']
        a15_diff = 0.0
        for tag in ('L17', 'L16'):
            for key in ('dsep_fin', 'D_abl', 'R_att',
                        'R_mlp', 'R_tot'):
                a15_diff = max(a15_diff, abs(
                    round(fams[tag][key], 3)
                    - d156[tag][key]))
        a15_ok = bool(a15_diff < 1e-3)
        log('a15 D1 scalars vs 2956 %.2e ok=%s'
            % (a15_diff, a15_ok), lines)

        if not (a17_ok and a18_ok and a10_ok and a11_ok
                and a14_ok and a15_ok):
            verdict = 'anchor_fail_all_void'
        else:
            # ---------- T1: family identity ----------
            s17 = fams['L17']['S_mlp'][LMIN:]
            s16 = fams['L16']['S_mlp'][LMIN:]
            from scipy.stats import spearmanr
            cos_f = float(s17 @ s16
                          / max(float(np.linalg.norm(s17)
                                      * np.linalg.norm(s16)),
                                1e-30))
            rho_f, _ = spearmanr(s17, s16)
            rho_f = float(rho_f)
            if cos_f >= COS_SHARED and rho_f >= RHO_SHARED:
                t1r = 'profile_shared'
            elif cos_f >= COS_PARTIAL:
                t1r = 'profile_partial'
            else:
                t1r = 'profile_divergent'
            # ---------- T2: scale ----------
            b = float(s17 @ s16
                      / max(float(s16 @ s16), 1e-30))
            r2 = cos_f ** 2
            if B_UNITY[0] <= b <= B_UNITY[1] \
                    and r2 >= R2_MIN:
                t2r = 'scale_unity'
            elif B_PROP[0] <= b < B_PROP[1] \
                    and r2 >= R2_MIN:
                t2r = 'scale_proportional'
            else:
                t2r = 'scale_divergent'
            # ---------- T3: mechanism (B0 decomposition) --
            cos_e = {}
            for tag in ('L17', 'L16'):
                sm = fams[tag]['S_mlp'][LMIN:]
                mi = fams[tag]['M_inj'][LMIN:]
                cos_e[tag] = float(
                    sm @ mi
                    / max(float(np.linalg.norm(sm)
                                * np.linalg.norm(mi)),
                          1e-30))
            t3r = 'injection_echo' \
                if all(cos_e[t] >= COS_ECHO
                       for t in ('L17', 'L16')) \
                else 'ablation_specific'
            verdict = '%s_%s_%s' % (t1r, t2r, t3r)
            log('T1 cos %.4f rho %.4f -> %s | T2 b %.4f R2 '
                '%.4f -> %s | T3 cos L17 %.4f L16 %.4f -> %s'
                % (cos_f, rho_f, t1r, b, r2, t2r,
                   cos_e['L17'], cos_e['L16'], t3r), lines)

            # interaction share (descriptive): how much of
            # S_mlp is NOT explained by adding M_inj
            d3 = {tag: {
                'cos_sm_minj': round(cos_e[tag], 4),
                'sum_S_mlp': round(
                    float(fams[tag]['S_mlp'][LMIN:].sum()),
                    3),
                'sum_M_inj': round(
                    float(fams[tag]['M_inj'][LMIN:].sum()),
                    3),
                'sum_A_inj': round(
                    float(fams[tag]['A_inj'][LMIN:].sum()),
                    3)}
                for tag in ('L17', 'L16')}
            d1 = {tag: {
                'R_att': round(fams[tag]['R_att'], 3),
                'R_mlp': round(fams[tag]['R_mlp'], 3),
                'R_tot': round(fams[tag]['R_tot'], 3),
                'D_abl': round(fams[tag]['D_abl'], 3),
                'dsep_pre': round(fams[tag]['dsep_pre'], 3),
                'dsep_fin': round(fams[tag]['dsep_fin'], 3)}
                for tag in ('L17', 'L16')}
            d2 = {tag: {
                'S_mlp': [round(float(v), 3) for v in
                          fams[tag]['S_mlp']],
                'M_inj': [round(float(v), 3) for v in
                          fams[tag]['M_inj']]}
                for tag in ('L17', 'L16')}
            save = {
                'words': np.array(['%s:%s:%s' % w
                                   for w in words],
                                  dtype=object),
                'labels_lang': lab_lang,
                'S_att_L17': fams['L17']['S_att'],
                'S_mlp_L17': fams['L17']['S_mlp'],
                'S_att_L16': fams['L16']['S_att'],
                'S_mlp_L16': fams['L16']['S_mlp'],
                'M_inj_L17': fams['L17']['M_inj'],
                'M_inj_L16': fams['L16']['M_inj'],
                'A_inj_L17': fams['L17']['A_inj'],
                'A_inj_L16': fams['L16']['A_inj'],
                'sc_I0_L17': sc_I0['L17'],
                'sc_I0_L16': sc_I0['L16'],
                'dirs_word': dirs_word,
            }

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2957, 'model': 'qwen3-4b',
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
               'a16_rel_raw': float('%.3e' % id_rel),
               'a16_worst': id_worst,
               'a16_ok': a16_ok,
               'a17_diff': None if a17_diff is None
               else float('%.3e' % a17_diff),
               'a17_rel': None if a17_rel is None
               else float('%.3e' % a17_rel),
               'a17_ok': a17_ok,
               'a18_diff': None if a18_diff is None
               else float('%.3e' % a18_diff),
               'a18_ok': a18_ok,
               'ok': bool(anchor_prelim and a10_ok
                          and a11_ok and a14_ok
                          and a15_ok and a17_ok
                          and a18_ok)},
           'T1_family_identity': t1r,
           'T2_scale': t2r,
           'T3_mechanism': t3r,
           'T1_cos_rho': None if t1r is None else
           {'cos': round(cos_f, 4), 'rho': round(rho_f, 4)},
           'T2_b_r2': None if t2r is None else
           {'b': round(b, 4), 'r2': round(r2, 4)},
           'T3_cos': None if t3r is None else
           {k: round(v, 4) for k, v in cos_e.items()},
           'D1_accounting': d1, 'D2_profiles': d2,
           'D3_b0_decomposition': d3,
           'b0_sep': runs.get('B0', {}).get('sep_med'),
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(os.path.join(
            OUT, 'rebalance_mlp_constancy.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2957 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
