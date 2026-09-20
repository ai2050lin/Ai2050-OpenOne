# -*- coding: utf-8 -*-
"""Phase 2958: imprint dose-response - is the ablation-specific
MLP rebalancing (2957: ablation_specific) driven by the
magnitude of the passive-loss imprint? Restore a fraction k of
the ablated heads' o_proj input (k*x_orig on the ablated
slices; o_proj linear => output imprint k*delta), and measure
the dose-response of the rebalancing.

Design (2956/2957 verbatim base):
  Session: B0(det) + {I0,I1} x {L17, L16} K=3 (a8, captures,
    x_orig = I0 o_proj input at the dose layer) + patch curve
    per family: k in {0, 0.25, 0.5, 0.75, 1.0}, K=2, capture
    rep0. Patch: ablated head slices := k * x_orig (k=None =>
    zero, verbatim I1).
  Downstream region: L17 l in [18,35]; L16 l in [17,35].
  R_mlp(k) = sum_l S_mlp_l(k), S_mlp_l(k) = csep(m^k - m^I0).

Tests (frozen):
  T1 dose-response: dev = max_k |R(k) - (1-k)R(0)| / |R(0)|
     over the 5 k points; dev <= 0.15 => imprint_linear_dose;
     else spearman(k, R(k)) >= 0.99 => imprint_nonlinear_dose;
     else imprint_insensitive.
  T2 profile stability: cos(S_mlp(k=0.5)[down], S_mlp(k=0)
     [down]) >= 0.9 => profile_fixed else profile_rotating.
  T3 readout dose: max_k |sep(k) - (sep1 + k*(sep0 - sep1))|
     / |sep0 - sep1| <= 0.1 => readout_linear else
     readout_nonlinear.
  Family agreement: both families agree => that label; else
  mixed_<axis>. Verdict f'{t1_all}_{t2_all}_{t3_all}'.

Anchors (frozen):
  a1/a3/a7/a9/a12/a2/a8/a13/a16/a17/a18/a10/a11 (2957
    verbatim thresholds);
  a14 patch closure: fin(k=1) == fin(I0) bit-level AND
    fin(k=0) == fin(I1) bit-level (< 1e-6; slice restore is
    exact: k=1 writes x_orig back, k=0 writes exact zeros);
  a15 S_att/S_mlp(k=0) profiles + D_abl vs 2957 bit/1e-3;
  a19 non-ablated o_proj-input slices bit-identical (I0 vs
    I1) at both dose layers;
  a21 passive-linear chain: |S_att[dose](k) + (1-k)*D_abl|
    < 1.0 for all k, both families (o_proj linearity under
    bf16 slice rounding).

Output: phase2958/imprint_dose_response/.
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
OUT = os.path.join(BASE, 'phase2958',
                   'imprint_dose_response')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2958_run_report.txt')
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
DEV_LIN = 0.15
RHO_MONO = 0.99
COS_FIX = 0.9
DEV_SEP = 0.1

PREREG = {
    'mode': '2956/2957 verbatim base + patch curve: ablated '
            'head o_proj-input slices := k * x_orig (x_orig = '
            'I0 capture at the dose layer), k in '
            '{0,0.25,0.5,0.75,1.0}, K=2, capture rep0; '
            'o_proj linearity makes the output imprint scale '
            'as k*delta exactly',
    'question': 'is the ablation-specific MLP rebalancing '
                '(2957) driven linearly by the passive-loss '
                'imprint magnitude, with a stable profile '
                'direction, and is the readout dose-response '
                'linear?',
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
               '< 1e-6 (bit-level expected)',
        'a15': 'S_att/S_mlp(k=0) profiles vs 2957 npz < 1e-6 '
               '(bit); D_abl vs 2957 D1 (3dp) < 1e-3',
        'a16': 'residual identity bf16-bound ratio <= 2.0 '
               '(I0/I1 both families + B0, rep0)',
        'a17': 'module accounting |sum - dsep_pre| / '
               '|dsep_pre| < 0.01 (k=0, both families)',
        'a18': 'causal null: C_l < 1e-12 for l < dose layer '
               '(k=0)',
        'a19': 'non-ablated o_proj-input slices bit-identical '
               'I0 vs I1 (both dose layers)',
        'a21': 'passive-linear chain: |S_att[dose](k) + '
               '(1-k)*D_abl| < 1.0 all k both families',
    },
    'T1': 'dev = max_k |R(k)-(1-k)R(0)|/|R(0)| over 5 k '
          'points; dev<=0.15 => imprint_linear_dose; else '
          'spearman(k,R)>=0.99 => imprint_nonlinear_dose; '
          'else imprint_insensitive',
    'T2': 'cos(S_mlp(k=.5)[down], S_mlp(k=0)[down]) >= 0.9 '
          '=> profile_fixed else profile_rotating',
    'T3': 'max_k |sep(k)-(sep1+k*(sep0-sep1))| / |sep0-sep1| '
          '<= 0.1 => readout_linear else readout_nonlinear',
    'verdict': 'anchor fail => anchor_fail_all_void; else '
               '{t1_all}_{t2_all}_{t3_all} (family agreement, '
               'else mixed_<axis>)',
    'correction_note': 'run1: (a) a15 compared the k=0 '
                       'captures against the I1 captures '
                       '(difference identically zero) instead '
                       'of the I0 captures - the 2957 profiles '
                       'are csep(I1 - I0), so the correct '
                       'reference is I0; the observed 9.37 = '
                       'D_abl signature exposed exactly this '
                       'reference error. (b) save dict '
                       'uninitialized on the anchor_fail path '
                       '(UnboundLocalError after verdict '
                       'printing; no results were affected - '
                       'result.json was already written). run2 '
                       'is authoritative.',
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
        json.dump({'phase': 2958,
                   'name': 'imprint_dose_response',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2887': sha8(SRC_2887),
                               's2927': sha8(SRC_2927),
                               's2939': sha8(SRC_2939),
                               's2950': sha8(SRC_2950),
                               's2957': sha8(SRC_2957),
                               's2957r': sha8(SRC_2957R)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'seed': SEED, 'k_repeat': K_REPEAT,
                   'li_switch': LI_SWITCH, 'li_grad': LI_GRAD,
                   's_switch': S_SWITCH, 's_grad': S_GRAD,
                   'top5_g': {str(k): list(v)
                              for k, v in TOP5_G.items()},
                   'ks': KS, 'dev_lin': DEV_LIN,
                   'rho_mono': RHO_MONO, 'cos_fix': COS_FIX,
                   'dev_sep': DEV_SEP,
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

    # ---------- hooks (2956/2957 verbatim + patch) ----------
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
    id_rel = 0.0
    id_ratio = 0.0
    id_worst = None
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
    a17_rel = a14_diff = a15_diff = a21_diff = None
    a10_ok = a11_ok = a17_ok = a18_ok = False
    a14_ok = a15_ok = a21_ok = False

    if not anchor_prelim:
        verdict = 'anchor_fail_all_void'
    else:
        # ---------- patch curve ----------
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
            # k=0 IS I1, so S_att(k=0) = csep(a_k0 - a_I0)
            # must equal 2957's S_att (= csep(a_I1 - a_I0))
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
                # ---------- tests per family ----------
                from scipy.stats import spearmanr
                tres = {}
                d3 = {}
                for tag, li_dose in (('L17', LI_SWITCH),
                                     ('L16', LI_GRAD)):
                    lmin = li_dose + 1
                    R = []
                    S05 = None
                    S00 = None
                    sa_dose = []
                    for k in KS:
                        x0, (a0, m0), _, _ = \
                            runs['I0_%d' % li_dose]['cap']
                        ak, (aak, mmk), _, _ = \
                            curves[tag][k]['cap']
                        Sa = np.array([csep(aak[l] - a0[l])
                                       for l in range(NL)])
                        Sm = np.array([csep(mmk[l] - m0[l])
                                       for l in range(NL)])
                        R.append(float(Sm[lmin:].sum()))
                        sa_dose.append(float(Sa[li_dose]))
                        if k == 0.0:
                            S00 = Sm
                        if k == 0.5:
                            S05 = Sm
                    R = np.array(R)
                    R0 = R[0]
                    pred = (1.0 - np.array(KS)) * R0
                    dev = float(np.abs(R - pred).max()
                                / max(abs(R0), 1e-30))
                    rho, _ = spearmanr(KS, R)
                    rho = float(rho)
                    if dev <= DEV_LIN:
                        t1 = 'imprint_linear_dose'
                    elif rho >= RHO_MONO:
                        t1 = 'imprint_nonlinear_dose'
                    else:
                        t1 = 'imprint_insensitive'
                    cs = float(S05[lmin:] @ S00[lmin:]
                               / max(float(
                                   np.linalg.norm(S05[lmin:])
                                   * np.linalg.norm(
                                       S00[lmin:])), 1e-30))
                    t2 = 'profile_fixed' \
                        if cs >= COS_FIX \
                        else 'profile_rotating'
                    sep1 = curves[tag][0.0]['sep_med']
                    sep0 = curves[tag][1.0]['sep_med']
                    seps = np.array(
                        [curves[tag][k]['sep_med']
                         for k in KS])
                    sepp = sep1 + np.array(KS) \
                        * (sep0 - sep1)
                    dev_s = float(np.abs(seps - sepp).max()
                                  / max(abs(sep0 - sep1),
                                        1e-30))
                    t3 = 'readout_linear' \
                        if dev_s <= DEV_SEP \
                        else 'readout_nonlinear'
                    tres[tag] = {'T1': t1, 'T2': t2,
                                 'T3': t3, 'dev_R': dev,
                                 'rho_R': rho, 'cos05': cs,
                                 'dev_sep': dev_s}
                    # a21 passive-linear chain
                    for k, sad in zip(KS, sa_dose):
                        a21_d = abs(sad + (1.0 - k)
                                    * D_abl[tag])
                        a21_diff = a21_d \
                            if a21_diff is None \
                            else max(a21_diff, a21_d)
                    log('%s R(k) %s | dev %.3f rho %.3f -> '
                        '%s | cos05 %.4f -> %s | sep dev '
                        '%.3f -> %s'
                        % (tag, [round(v, 2) for v in R],
                           dev, rho, t1, cs, t2, dev_s,
                           t3), lines)
                a21_ok = bool(a21_diff is not None
                              and a21_diff < 1.0)
                log('a21 passive-linear chain max %.3f ok=%s'
                    % (a21_diff or 0.0, a21_ok), lines)
                if not a21_ok:
                    verdict = 'anchor_fail_all_void'
                else:
                    def agree(key):
                        v0 = tres['L17'][key]
                        v1 = tres['L16'][key]
                        return v0 if v0 == v1 \
                            else 'mixed_' + {
                                'T1': 'dose_response',
                                'T2': 'profile',
                                'T3': 'readout'}[key]
                    t1r = agree('T1')
                    t2r = agree('T2')
                    t3r = agree('T3')
                    verdict = '%s_%s_%s' % (t1r, t2r, t3r)
                    d1 = {}
                    for tag, li_dose in (('L17', LI_SWITCH),
                                         ('L16', LI_GRAD)):
                        lmin = li_dose + 1
                        Rk = []
                        for k in KS:
                            x0, (a0, m0), _, _ = \
                                runs['I0_%d'
                                     % li_dose]['cap']
                            ak, (aak, mmk), _, _ = \
                                curves[tag][k]['cap']
                            Sm = np.array(
                                [csep(mmk[l] - m0[l])
                                 for l in range(NL)])
                            Rk.append(round(
                                float(Sm[lmin:].sum()), 3))
                        d1[tag] = {
                            'R_mlp_k': Rk,
                            'sep_k': [round(
                                curves[tag][k]['sep_med'],
                                2) for k in KS],
                            'R_mlp_k_pred': [round(
                                float((1.0 - k)
                                      * (Rk[0])), 3)
                                for k in KS],
                            'D_abl': round(D_abl[tag], 3),
                            **{kk: (round(vv, 4)
                                    if isinstance(vv, float)
                                    else vv)
                               for kk, vv
                               in tres[tag].items()}}
                    d2 = {}
                    for tag, li_dose in (('L17', LI_SWITCH),
                                         ('L16', LI_GRAD)):
                        x0, (a0, m0), _, _ = \
                            runs['I0_%d' % li_dose]['cap']
                        ak, (aak, mmk), _, _ = \
                            curves[tag][0.5]['cap']
                        Sm = np.array([csep(mmk[l] - m0[l])
                                       for l in range(NL)])
                        d2[tag] = {
                            'S_mlp_k05': [round(float(v), 3)
                                          for v in Sm]}
                    save = {
                        'words': np.array(
                            ['%s:%s:%s' % w
                             for w in words],
                            dtype=object),
                        'labels_lang': lab_lang,
                        'sc_I0_L17': sc_I0['L17'],
                        'sc_I0_L16': sc_I0['L16'],
                        'dirs_word': dirs_word,
                        'R_mlp_k_L17': np.array(d1['L17']
                                                ['R_mlp_k']),
                        'R_mlp_k_L16': np.array(d1['L16']
                                                ['R_mlp_k']),
                        'sep_k_L17': np.array(
                            d1['L17']['sep_k']),
                        'sep_k_L16': np.array(
                            d1['L16']['sep_k']),
                        'S_mlp_k05_L17': np.array(
                            d2['L17']['S_mlp_k05']),
                        'S_mlp_k05_L16': np.array(
                            d2['L16']['S_mlp_k05']),
                    }

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2958, 'model': 'qwen3-4b',
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
               'a8_diff': None if a8_diff is None
               else float('%.3e' % a8_diff),
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
               'ok': bool(anchor_prelim and a8_ok
                          and a10_ok and a11_ok
                          and a14_ok and a15_ok
                          and a17_ok and a18_ok
                          and a21_ok)},
           'T1_dose_response': t1r,
           'T2_profile': t2r,
           'T3_readout': t3r,
           'family_tests': None if t1r is None else
           {tag: {kk: (round(vv, 4)
                       if isinstance(vv, float) else vv)
                  for kk, vv in d1[tag].items()
                  if kk in ('dev_R', 'rho_R', 'cos05',
                            'dev_sep')}
            for tag in ('L17', 'L16')}
           if d1 else None,
           'D1_curves': d1, 'D2_profiles': d2,
           'D3_passive_chain': {'a21_max':
                                None if a21_diff is None
                                else round(a21_diff, 4)},
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(os.path.join(
            OUT, 'imprint_dose_response.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2958 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
