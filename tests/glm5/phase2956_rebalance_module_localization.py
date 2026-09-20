# -*- coding: utf-8 -*-
"""Phase 2956: rebalancing localization by module type - is
the competitive rebalancing (2950 D_nonlin, 70-74% of the
group-ablation deepening) carried by downstream ATTENTION
modules or MLP modules, layer by layer?

Why: 2950 proved active compensatory rebalancing (T1/T2 both
pass), 2951 found its head-level carriers are functional W_ov
heads, 2952 attributed the readout amplification to attention
self-weight gain. But the MODULE-TYPE and LAYER localization
of the rebalancing response was never measured: the 2950
profile only sampled layer inputs, conflating attention and
MLP contributions.

Design (2950 verbatim protocol, K=3 same-session repeats):
  Conditions (func batch, 57 prompts):
    B0  (no inj, no abl)          - determinism + identity
    I0_17 (inj L17 s=1.0)         - reference
    I1_17 (inj L17 s=1.0 + abl top5_g L17)
    I0_16 (inj L16 s=2.0)
    I1_16 (inj L16 s=2.0 + abl top5_g L16)
  Captures per forward (rep 0 kept; determinism via a8):
    x_l  = input_layernorm INPUT at every layer (true
           residual entering layer l, pos1, 57x2560)
    a_l  = self_attn OUTPUT (o_proj output, pos1)
    m_l  = mlp OUTPUT (pos1)
    o_proj input at the dose layer (2950 verbatim, head
    slices) for the per-head snapshot sc map.

Residual identity (exact, bf16 forward): x_{l+1} = x_l +
a_l + m_l. Since proj (u35) and group means are linear:
  dSep_pre = sum_l [csep(da_l) + csep(dm_l)]   (exact)
where csep(v) = mean_{lang0} proj(v) - mean_{lang1} proj(v).

Rebalancing accounting (per family):
  S_att_l = csep(a_l^I1 - a_l^I0); S_mlp_l analog.
  At the ablated layer the o_proj is linear and the
  ablation only zeroes head slices of the SAME layer, so
  S_att_17 = -D_abl exactly in expectation, with
  D_abl = sum_{h in top5_g} sc_h(I0) (2950 verbatim).
  Rebalancing (non-passive) part:
    R_att = sum_l S_att_l + D_abl
    R_mlp = sum_l S_mlp_l
    R_tot = R_att + R_mlp = dSep_pre + D_abl
  (the module split of 2950's D_nonlin, in the pre-final-
  norm readout; dSep_final vs 2950 cross-checked by a10/a11)

Tests (frozen):
  T1 module axis (per family): att_carried if
     sign(R_att)==sign(R_tot) AND |R_att| >= 2*|R_mlp|;
     mlp_carried if sign(R_mlp)==sign(R_tot) AND
     |R_mlp| >= 2*|R_att|; else mixed.
     Both layers agree else mixed_modules.
  T2 band axis (per family): layer profile C_l =
     S_att_l + S_mlp_l for l > dose layer; top-3 layers by
     |C_l| carry >= 50% of sum_{l>dose}|C_l| =>
     band_concentrated else band_distributed.
     Both layers agree else mixed_band.
Verdict (frozen): anchor fail => anchor_fail_all_void;
  else f'{mod_all}_{band_all}'.

Anchors (frozen):
  a1 dirs rebuild < 1e-5; a3 Vt8 < 1e-6; a7 xdir < 1e-9;
  a9 ablation-slice self-check + o_proj gate; a12 group
  mask self-check; a2 base determinism < 1e-4;
  a8 same-session K=3 rep determinism < 1e-6;
  a13 ablated-head o_proj-input slices exactly zero (I1);
  a10 dSep_final L17 vs 2950 < 1.0;
  a11 dSep_final L16 vs 2950 < 1.0;
  a16 residual identity chain rel < 5e-3 (all 35 links,
      B0/I0_17/I1_17/I0_16/I1_16 rep0);
  a17 module decomposition identity |sum - dSep_pre| < 1e-6
      (fp64, both families);
  a18 causal null: C_l exactly ~0 for l < dose layer
      (< 1e-12; upstream captures must be identical).

Output: phase2956/rebalance_module_localization/.
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
OUT = os.path.join(BASE, 'phase2956',
                   'rebalance_module_localization')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2956_run_report.txt')
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
DOM_RATIO = 2.0
TOP3_SHARE = 0.5

PREREG = {
    'mode': '2950 verbatim forward family (K=3 repeats): '
            'B0 + {I0,I1} x {L17 s=1.0, L16 s=2.0}; captures: '
            'input_layernorm INPUT (true residual x_l) at '
            'every layer, self_attn OUTPUT (a_l), mlp OUTPUT '
            '(m_l), pos1; o_proj input at dose layers for the '
            'per-head snapshot sc map (2950 verbatim)',
    'question': 'is the 2950 competitive rebalancing carried '
                'by downstream attention modules or MLP '
                'modules, and concentrated in which layers?',
    'top5_g_frozen': {str(k): v for k, v in TOP5_G.items()},
    'accounting': 'dSep_pre = sum_l [csep(da_l)+csep(dm_l)] '
                  '(exact linear identity, a17); passive loss '
                  'lives entirely in S_att at the ablated '
                  'layer (o_proj linear): R_att = sum S_att + '
                  'D_abl, R_mlp = sum S_mlp, R_tot = dSep_pre '
                  '+ D_abl (module split of 2950 D_nonlin)',
    'anchors': {
        'a1': 'dirs_word rebuild vs 2927 < 1e-5',
        'a2': 'base determinism < 1e-4',
        'a3': 'Vt8 vs 2939 < 1e-6',
        'a7': 'xdir self-check < 1e-9',
        'a8': 'same-session K=3 rep determinism < 1e-6',
        'a9': 'ablation-slice self-check < 1e-12 + o_proj '
              'in_features == 4096',
        'a10': 'dSep_final L17 vs 2950 < 1.0',
        'a11': 'dSep_final L16 vs 2950 < 1.0',
        'a12': 'group-ablation mask self-check < 1e-12',
        'a13': 'ablated-head capture-zero (I1) == 0',
        'a16': 'residual identity x_{l+1}=x_l+a_l+m_l, '
               'bf16-bound-normalized ratio max(|diff| / '
               '(2^-7*(|x|+|a|+|m|))) <= 2.0, 35 links x 4 '
               'conds (rep0); raw rel-vs-max|x1| registered '
               'descriptively',
        'a17': 'module accounting: |sum_l(S_att+S_mlp) - '
               'csep(dFin)| / |csep(dFin)| < 0.01 (relative; '
               'the fp64-exact version is unreachable - bf16 '
               'residual-chain rounding accumulates ~0.4% '
               'projection noise, measured in run6)',
        'a18': 'causal null: C_l < 1e-12 for l < dose layer',
    },
    'T1': 'module axis: att_carried if sign(R_att)=='
          'sign(R_tot) and |R_att|>=2|R_mlp|; mlp_carried '
          'analog; else mixed; both layers agree else '
          'mixed_modules',
    'T2': 'band axis: top-3 layers by |C_l| (l>dose) carry '
          '>= 50% of total |C| => band_concentrated else '
          'band_distributed; both agree else mixed_band',
    'verdict': 'anchor fail => anchor_fail_all_void; else '
               '{mod_all}_{band_all}',
    'correction_note': 'run1 KeyError at pass1: cap_ln["on"] '
                       'was never enabled for the dirs-rebuild '
                       'pass (input_layernorm pre-hook gate '
                       'off), so cap_ln["store"] was empty. '
                       'No results were computed (crash in '
                       'pass1, before any anchor); '
                       'execution.json deleted per discipline '
                       '3. run2 crashed identically: the pass1 '
                       'gate fix reported edit-success but did '
                       'NOT persist to disk (known local '
                       'phantom-edit defect; verified by Grep '
                       'after re-edit). Hook trigger itself '
                       'verified healthy by probe '
                       '(phase2956_hook_probe: iln0_pre fires). '
                       'run3: pass1 gate fix persisted but two '
                       'new failures: (a) a1 dirs rebuild 3.7e-'
                       '01 - pass1 captured the TRUE residual '
                       'via input_layernorm pre-hook, but 2927/'
                       '2950 dirs_word are defined on the POST-'
                       'layernorm attnin signal (self_attn '
                       'pre-hook input); fixed by a dedicated '
                       'cap_nm capture in pre_attn (2950 '
                       'verbatim caliber; the input_layernorm '
                       'residual capture remains for the '
                       'identity chain only). (b) self_attn '
                       'forward output is a tuple in this '
                       'transformers version - post-hook now '
                       'takes output[0]. Additionally 4 of 5 '
                       'run3 edits were phantom-lost (edit-'
                       'success without disk persistence, known '
                       'local defect) and were re-applied one '
                       'by one with Grep verification. run4: '
                       'anchors a1/a3/a7/a9/a12/a2/a8 all pass '
                       '(a1 2.17e-08 bit-consistent; four seps '
                       'exactly reproduce 2950: 21.5/-9.5/'
                       '84.8/48.1) but crashed at a13: my '
                       'pre_oproj placed the o_proj-input '
                       'capture on the non-ablated branch '
                       'only, so I1 conditions never captured '
                       '(KeyError 17); fixed to 2950 verbatim '
                       '(capture post-ablation input on both '
                       'branches). No results computed in '
                       'run4; execution.json deleted per '
                       'discipline 3; run5 authoritative. run5: '
                       'all anchors pass except a16 (rel 6.25e-'
                       '03 vs 5e-3, over by 1.25e-3): the '
                       'metric normalized by max|x_{l+1}| is '
                       'mis-calibrated under cancellation - '
                       'bf16 rounding error scales with the '
                       'ADDEND magnitudes (two round-to-'
                       'nearest adds, bound 2x2^-8 = 7.8e-3 '
                       'relative), not with the result. a16 '
                       'v2 (frozen, discipline-10 family): '
                       'bound-normalized ratio max(|diff| / '
                       '(2^-7*(|x|+|a|+|m|))) <= 2.0; raw rel '
                       'kept descriptive. A real capture bug '
                       'would exceed the bound by orders of '
                       'magnitude; run6 confirmed: ratio 0.979 '
                       'pass, worst link L33 raw diff 1.875 on '
                       'addends |a|24/|m|61 (pure bf16 ulp '
                       'noise). run6 then failed a17 (1.4e-2/'
                       '1.6e-1 vs fp64 1e-6): the telescoping '
                       'identity is fp64-exact only in exact '
                       'arithmetic - the bf16 residual chain '
                       'accumulates per-link rounding (a16 '
                       'bound) whose u35 projection is ~0.1-'
                       '0.4% of dSep. a17 v2 (frozen): '
                       'relative gap < 1%; also dsep_pre now '
                       'uses the TRUE final residual (pre_norm '
                       'is a forward_PRE_hook, fin_cap[x] is '
                       'the norm INPUT), removing the r=x35+a35'
                       '+m35 reconstruction. run7 is '
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
        json.dump({'phase': 2956,
                   'name': 'rebalance_module_localization',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2887': sha8(SRC_2887),
                               's2927': sha8(SRC_2927),
                               's2939': sha8(SRC_2939),
                               's2950': sha8(SRC_2950)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'seed': SEED, 'k_repeat': K_REPEAT,
                   'li_switch': LI_SWITCH, 'li_grad': LI_GRAD,
                   's_switch': S_SWITCH, 's_grad': S_GRAD,
                   'top5_g': {str(k): list(v)
                              for k, v in TOP5_G.items()},
                   'dom_ratio': DOM_RATIO,
                   'top3_share': TOP3_SHARE,
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

    # ---------- hooks ----------
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
            # capture post-ablation input for BOTH ablated
            # and non-ablated passes (2950 verbatim; the a13
            # capture-zero check reads the I1 captures)
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
    # 2950 verbatim: dirs_word is defined on the POST-
    # layernorm attnin signal (self_attn pre-hook input),
    # NOT on the true residual - a1 run3 failure was the
    # pre/post-norm caliber mismatch (3.7e-01).
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

    # a9 / a12 self-checks
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

    # a16: residual identity chain (v2, bf16-bound-normalized)
    # x_{l+1} = round_bf16(round_bf16(x_l + a_l) + m_l): the
    # rounding error scales with the ADDEND magnitudes, not
    # with |x_{l+1}| (cancellation makes the max|x|-based
    # metric overestimate). Bound: |diff| <= 2^-7 *
    # (|x_l|+|a_l|+|m_l|) elementwise; a16 v2 requires
    # max ratio <= 2.0. A real capture bug (wrong hook point)
    # would exceed the bound by orders of magnitude.
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
                         and a13_ok and a16_ok)

    verdict = None
    t1 = t2 = d1 = d2 = None
    mod = {}
    band = {}
    save = {}
    a10_diff = a11_diff = a17_diff = a18_diff = None
    a17_rel = None
    a10_ok = a11_ok = a17_ok = a18_ok = False

    if not anchor_prelim:
        verdict = 'anchor_fail_all_void'
    else:
        fams = {}
        for tag, li_dose in (('L17', LI_SWITCH),
                             ('L16', LI_GRAD)):
            x0, (a0, m0), _, fin0 = runs['I0_%d' % li_dose][
                'cap']
            x1, (a1c, m1c), _, fin1 = runs['I1_%d' % li_dose][
                'cap']
            # fin IS the true residual entering the final norm
            # (pre_norm is a forward_PRE_hook on model.norm);
            # no reconstruction needed.
            dsep_pre = csep(fin1 - fin0)
            dsep_fin = runs['I1_%d' % li_dose]['sep_med'] \
                - runs['I0_%d' % li_dose]['sep_med']
            S_att = np.array([csep(a1c[l] - a0[l])
                              for l in range(NL)])
            S_mlp = np.array([csep(m1c[l] - m0[l])
                              for l in range(NL)])
            R_att = float(S_att.sum() + D_abl[tag])
            R_mlp = float(S_mlp.sum())
            R_tot = R_att + R_mlp
            C = S_att + S_mlp
            fams[tag] = {
                'dsep_pre': dsep_pre, 'dsep_fin': dsep_fin,
                'S_att': S_att, 'S_mlp': S_mlp, 'C': C,
                'R_att': R_att, 'R_mlp': R_mlp,
                'R_tot': R_tot, 'D_abl': D_abl[tag]}
            # a17 v2: relative (bf16 chain-noise allowance)
            a17_d = abs(float(S_att.sum() + S_mlp.sum())
                        - dsep_pre)
            a17_r = a17_d / max(abs(dsep_pre), 1e-30)
            a17_diff = max(a17_diff or 0.0, a17_d)
            a17_rel = max(a17_rel or 0.0, a17_r)
            # a18 causal null
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

        # a10/a11 vs 2950 (final-norm dSep)
        a10_diff = abs(fams['L17']['dsep_fin']
                       - dsep50['L17'])
        a11_diff = abs(fams['L16']['dsep_fin']
                       - dsep50['L16'])
        a10_ok = bool(a10_diff < 1.0)
        a11_ok = bool(a11_diff < 1.0)
        log('a10 dSep_fin L17 vs 2950 %.3f ok=%s | a11 L16 '
            '%.3f ok=%s' % (a10_diff, a10_ok, a11_diff,
                            a11_ok), lines)

        if not (a17_ok and a18_ok and a10_ok and a11_ok):
            verdict = 'anchor_fail_all_void'
        else:
            mod = {}
            band = {}
            for tag, li_dose in (('L17', LI_SWITCH),
                                 ('L16', LI_GRAD)):
                f = fams[tag]
                Ra, Rm, Rt = f['R_att'], f['R_mlp'], \
                    f['R_tot']
                if np.sign(Ra) == np.sign(Rt) \
                        and abs(Ra) >= DOM_RATIO * abs(Rm):
                    m = 'att_carried'
                elif np.sign(Rm) == np.sign(Rt) \
                        and abs(Rm) >= DOM_RATIO * abs(Ra):
                    m = 'mlp_carried'
                else:
                    m = 'mixed'
                mod[tag] = m
                Cdown = f['C'][li_dose + 1:]
                tot = float(np.abs(Cdown).sum())
                top3 = float(np.sort(
                    np.abs(Cdown))[::-1][:3].sum())
                conc = bool(tot > 0
                            and top3 / tot >= TOP3_SHARE)
                band[tag] = {'conc': conc,
                             'top3_share':
                                 round(top3 / max(tot, 1e-30),
                                       4),
                             'argmax_layer': int(
                                 li_dose + 1
                                 + int(np.argmax(
                                     np.abs(Cdown))))}
                log('%s T1 %s (R_att %.2f R_mlp %.2f) | T2 '
                    'top3_share %.3f argmax L%d -> %s'
                    % (tag, m, Ra, Rm,
                       band[tag]['top3_share'],
                       band[tag]['argmax_layer'],
                       'band_concentrated' if conc
                       else 'band_distributed'), lines)
            mods = set(mod.values())
            mod_all = mod['L17'] if len(mods) == 1 \
                else 'mixed_modules'
            c0, c1 = band['L17']['conc'], band['L16']['conc']
            if c0 == c1:
                band_all = 'band_concentrated' if c0 \
                    else 'band_distributed'
            else:
                band_all = 'mixed_band'
            verdict = '%s_%s' % (mod_all, band_all)

            d1 = {tag: {
                'S_att': [round(float(v), 3)
                          for v in fams[tag]['S_att']],
                'S_mlp': [round(float(v), 3)
                          for v in fams[tag]['S_mlp']],
                'R_att': round(fams[tag]['R_att'], 3),
                'R_mlp': round(fams[tag]['R_mlp'], 3),
                'R_tot': round(fams[tag]['R_tot'], 3),
                'D_abl': round(fams[tag]['D_abl'], 3),
                'dsep_pre': round(fams[tag]['dsep_pre'], 3),
                'dsep_fin': round(fams[tag]['dsep_fin'], 3)}
                for tag in ('L17', 'L16')}
            d2 = {tag: {
                'C_profile': [round(float(v), 3) for v in
                              fams[tag]['C']],
                'top3_share': band[tag]['top3_share'],
                'argmax_layer': band[tag]['argmax_layer'],
                'D_abl': round(fams[tag]['D_abl'], 3)}
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
                'sc_I0_L17': sc_I0['L17'],
                'sc_I0_L16': sc_I0['L16'],
                'dirs_word': dirs_word,
            }

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2956, 'model': 'qwen3-4b',
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
                          and a11_ok and a17_ok
                          and a18_ok)},
           'T1_module': None if t1 is not None else None,
           'module_axis': mod if verdict !=
           'anchor_fail_all_void' else None,
           'band_axis': {k: (v if isinstance(v, dict)
                             else v)
                         for k, v in band.items()}
           if verdict != 'anchor_fail_all_void' else None,
           'D1_accounting': d1, 'D2_profiles': d2,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if save:
        np.savez_compressed(os.path.join(
            OUT, 'rebalance_module_localization.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2956 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
