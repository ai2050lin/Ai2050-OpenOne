# -*- coding: utf-8 -*-
"""Phase 3004: Omega-P4 layer transfer operator structure
(qwen side; plan v5 P4).

Why: 3002 localized the qwen mid-band as an xdir-specific,
dose-linear, sign-asymmetric amplification band (3003).
Open question (plan v5 P4): what is the STRUCTURE of the
layer-to-layer transfer operator Dl that carries this?
Shared low-rank redirect (a meaningful "conversion
operator", the 2985-corrected non-orthogonal non-fixed
version) vs per-cell independent scatter (another piece of
evidence that single-point operationalization stays
closed).  NOTE (plan v5 P4 premise correction, registered
before any run): the 3001/3002 npz stored only the MEAN
trk profiles (1-D), not per-cell per-layer Delta - so P4
requires a capture-extended rerun, not a zero-forward
reanalysis.  Cost is small (qwen ~20 s).

Design (3002 machine verbatim: 57 words 2887, dirs_word
rebuild a1 vs 2927, Vt8 a3 vs 2939, u35 readout, dcks from
2939 coords, xdir = dcks_S @ Vt8_S, attn_in pos-1
single-layer coef injection, bf16):
  T2-lite scan at L15/L17 s=2 K=2 (a9 raw ratio anchor vs
     2945, chain continuity with 3002/3003).
  T4 EXTENDED capture: inject xdir s=2 at L4 and L17;
     capture per-cell pos-1 attn_in at EVERY layer; Dl_i(l)
     = inj_i(l) - base_i(l) stored in full (no medians).
  A OPERATOR STRUCTURE per layer l > L_inj:
     M_l = per-cell Delta matrix (57 x 2560); SVD ->
     r1(l) = S1^2/sum(S^2) (energy concentration),
     erank(l) = entropy effective rank, medcos(l) = median
     per-cell |cos(Dl_i, v1)| (shared-direction alignment).
  E ENERGY ACCOUNT per layer (descriptive): per cell,
     source = Delta_i(L_inj) (injection layer, contains
     the injected delta); aligned energy fraction = 
     (Dl_i(l) . uh_i)^2 / ||Dl_i(l)||^2; amplitude ratio =
     ||Dl_i(l)|| / ||source_i||.  Median profiles.

Verdict (frozen; PRIMARY = L17 amplification band, layers
18..25):
  anchor fail                  => anchor_fail_all_void
  med r1 >= 0.5 AND med medcos >= 0.6
                               => shared_lowrank_operator_qwen
  med r1 < 0.2 OR med medcos < 0.3
                               => cell_scatter_operator_qwen
  otherwise                    => mixed_operator_qwen
  (null calibration descriptive: r1 of a row-normalized
  gaussian 57x2560 matrix; expected ~1e-2 scale.  Energy
  account fully descriptive, no gate.)

Anchors (frozen):
  a0 words == 2887 re-export (57)
  a1 dirs_word vs 2927 < 1e-5
  a2 baseline determinism < 1e-4
  a3 Vt8 vs 2939 < 1e-6
  a4 proj_func vs 2935 s_base[func] < 1e-4 (bit-level)
  a5 proj_null0 vs 2935 s_base[null0] < 1e-4 (bit-level)
  a6 sep_f > 0
  a7 xdir identity < 1e-9
  a8 null0 collision-free
  a9 2945 repro (raw ratio): |ratio_L15 - 1.2149| < 0.1
     AND |ratio_L17 - 1.5004| < 0.1
  a10 3002 source integrity: result hash == seal AND
      verdict == context_entangled_qwen
  a11 same-session determinism < 1e-6 (all K-repeat arms)
  a12 trk profile reproduction vs 3002 stored trk_L4 /
      trk_L17: max |Delta| < 1e-3 (3002 stored values
      rounded to 4 dp; cross-session gate)

Tags: Omega-P4 / lang axis / len-2 / snapshot machine (no
ablation) / dimensionless gates / operator structure /
energy account / plan v5 P4.
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2887 = os.path.join(BASE, 'phase2887',
                        'language_axis_mlp',
                        'language_axis_mlp.npz')
SRC_2927 = os.path.join(BASE, 'phase2927',
                        'probe_relativity',
                        'probe_relativity.npz')
SRC_2935 = os.path.join(BASE, 'phase2935',
                        'null_amp_anatomy',
                        'null_amp_anatomy.npz')
SRC_2939 = os.path.join(BASE, 'phase2939',
                        'rotation_target',
                        'rotation_target.npz')
SRC_2945 = os.path.join(BASE, 'phase2945',
                        'threshold_curves',
                        'result.json')
D_3002 = os.path.join(BASE, 'phase3002',
                      'omega_g2_robustness_source_qwen')
OUT = os.path.join(BASE, 'phase3004',
                   'omega_p4_operator_structure_qwen')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NL, HID, VOCAB = 36, 2560, 151936
SEL_LAYERS = (15, 17)
REF_LAYER = 17
TRK_LAYERS = (4, 17)
AMP_BAND = (18, 19, 20, 21, 22, 23, 24, 25)
S_IDX = (0, 1, 4)         # 3002 verbatim (xdir subspace)
S_SCAN = 2.0
K_SCAN = 2
SEED_NULL = 2896          # 3000/3002 verbatim
SEED_RND = 3004           # 3002 verbatim (a9 chain)
R1_SHARED = 0.5
COS_SHARED = 0.6
R1_SCATTER = 0.2
COS_SCATTER = 0.3
A12_TOL = 1e-3
REF_2945 = {'15': 1.2149, '17': 1.5004}
A9_TOL = 0.1
N_SUB_RND = 2
N_FULL_RND = 1

PREREG = {
    'mode': 'qwen3-4b only; 3002 machine verbatim (57 '
            'words 2887, dirs_word a1 vs 2927, Vt8 a3 vs '
            '2939, u35 readout, dcks from 2939 coords, '
            'xdir = dcks_S@Vt8_S, attn_in pos-1 '
            'single-layer coef injection, bf16); plan v5 '
            'P4 operator structure',
    'question': 'what is the structure of the layer-to-'
                'layer transfer operator Dl carrying the '
                'qwen mid-band amplification: shared low-'
                'rank redirect vs per-cell independent '
                'scatter?  Plus a descriptive energy '
                'account (aligned / orthogonalized / '
                'amplitude).',
    'premise_correction':
        'plan v5 P4 said zero-forward reanalysis of '
        'stored per-cell per-layer Delta; the 3001/3002 '
        'npz stored only MEAN trk profiles (1-D), so P4 '
        'requires a capture-extended rerun (registered '
        'BEFORE any run of this phase)',
    'a12_redefinition_note':
        'QUASI-POST-HOC, registered after run3: the '
        'frozen a12 (per-cell trk reproduction vs 3002 '
        'stored profiles < 1e-3) is impossible because '
        'the 3002 T3 stored values are themselves a '
        'mis-declared quantity (axis bug, run3 probe '
        'proof); run4 replaces a12 with a12a = '
        '3002-formula reproduction < 1e-3 (bug-identity '
        'anchor) and registers the corrected per-cell '
        'profile as T3_corrected (descriptive)',
    'T2lite': 'scan at L15/L17 s=2 K=2 (xdir + 2 sub-'
              'random + 1 full-random), a9 raw ratio '
              'anchor vs 2945 only - no selectivity '
              'verdict here (3002/3003 own it)',
    'T4': 'extended capture: xdir s=2 at L4 and L17; '
          'per-cell pos-1 attn_in at EVERY layer; '
          'Dl_i(l) = inj_i(l) - base_i(l) stored in full',
    'A': 'per layer l > L_inj: M_l = (57 x 2560) per-cell '
         'Delta matrix; r1(l) = S1^2/sum(S^2); erank(l) = '
         'entropy effective rank; medcos(l) = median '
         'per-cell |cos(Dl_i, v1)|',
    'E': 'per layer (descriptive): source = Delta_i(L_inj)'
         '; aligned energy fraction = (Dl_i(l).uh_i)^2 / '
         '||Dl_i(l)||^2; amplitude ratio = ||Dl_i(l)|| / '
         '||source_i||; median profiles',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'PRIMARY = L17 band layers 18..25: med r1 '
               '>= 0.5 AND med medcos >= 0.6 => '
               'shared_lowrank_operator_qwen; med r1 < '
               '0.2 OR med medcos < 0.3 => '
               'cell_scatter_operator_qwen; else '
               'mixed_operator_qwen; null calibration '
               'descriptive (row-normalized gaussian '
               'r1); energy account no gate',
    'tags': 'Omega-P4 / lang axis / len-2 / snapshot '
            'machine (no ablation) / dimensionless gates '
            '/ operator structure / energy account / '
            'plan v5 P4',
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
    lines.append('[%s] %s' % (time.strftime('%H:%M:%S'),
                              msg))
    with open(os.path.join(OUT, 'run_log.txt'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def op_structure(D):
    """D: (n_cells, hid) per-cell delta matrix."""
    U, S, Vt = np.linalg.svd(D, full_matrices=False)
    e = S ** 2
    tot = float(e.sum())
    r1 = float(e[0]) / max(tot, 1e-30)
    p = e / max(tot, 1e-30)
    p = p[p > 0]
    erank = float(np.exp(-float((p * np.log(p)).sum())))
    nrm = np.linalg.norm(D, axis=1)
    cos = np.abs(D @ Vt[0]) / np.maximum(nrm, 1e-30)
    medcos = float(np.median(cos))
    return r1, erank, medcos, Vt[0]


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 3004,
                   'name':
                       'omega_p4_operator_structure_qwen',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {
                       's2887': sha8(SRC_2887),
                       's2927': sha8(SRC_2927),
                       's2935': sha8(SRC_2935),
                       's2939': sha8(SRC_2939),
                       's2945': sha8(SRC_2945),
                       's3002result': sha8(
                           D_3002 + r'\result.json'),
                       's3002seal': sha8(
                           D_3002 + r'\seal.json')},
                   'model': 'qwen3-4b',
                   'n_layers': NL, 'hidden': HID,
                   'vocab': VOCAB,
                   'sel_layers': list(SEL_LAYERS),
                   'ref_layer': REF_LAYER,
                   'trk_layers': list(TRK_LAYERS),
                   'amp_band': list(AMP_BAND),
                   's_scan': S_SCAN, 'k_scan': K_SCAN,
                   'seed_null': SEED_NULL,
                   'seed_rnd': SEED_RND,
                   'r1_shared': R1_SHARED,
                   'cos_shared': COS_SHARED,
                   'r1_scatter': R1_SCATTER,
                   'cos_scatter': COS_SCATTER,
                   'a12_tol': A12_TOL,
                   'ref_2945': REF_2945,
                   'a9_tol': A9_TOL,
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- sources ----------
    z87 = np.load(SRC_2887, allow_pickle=True)
    words = [tuple(str(w).split(':'))
             for w in z87['words']]
    lab_lang = np.asarray(z87['labels_lang']).astype(int)
    n_words = len(words)
    a0_ok = bool(n_words == 57)
    log('a0 words == 2887 re-export: %s (n=%d)'
        % (a0_ok, n_words), lines)

    z27 = np.load(SRC_2927, allow_pickle=True)
    dirs27 = z27['dirs_word'].astype(np.float64)
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
    d3 = r45['D3_ratio']

    # a10 3002 source integrity
    seal02 = json.load(open(D_3002 + r'\seal.json',
                            encoding='utf-8'))
    a10_ok = bool(seal02['result_sha256_8']
                  == sha8(D_3002 + r'\result.json'))
    r02 = json.load(open(D_3002 + r'\result.json',
                         encoding='utf-8'))
    a10_ok = a10_ok and bool(
        r02['final_verdict'] == 'context_entangled_qwen'
        and r02['anchor_all_ok'] is True)
    trk02_L4 = r02['T3']['trk']['L4']['profile']
    trk02_L17 = r02['T3']['trk']['L17']['profile']
    log('a10 3002 integrity %s' % a10_ok, lines)

    # ---------- model ----------
    import torch
    import sys
    sys.path.insert(
        0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
    from phase2662_symmetric_mapping_contract \
        import load_native
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
            assert tid_map[w] == int(ck), \
                'key mismatch %s' % w
    func_tid = tid('the')
    word_tids = set(tid_map.values())

    rng0 = np.random.default_rng(SEED_NULL)
    null0_tids = []
    while len(null0_tids) < n_words:
        r = int(rng0.integers(0, VOCAB))
        if r not in word_tids and r > 0:
            null0_tids.append(r)
    a8_ok = bool(len(null0_tids) == n_words
                 and not (set(null0_tids) & word_tids))
    log('a8 null0 collision-free: %s' % a8_ok, lines)

    batch = {'func': [[func_tid, tid_map[words[i][2]]]
                      for i in range(n_words)],
             'null0': [[null0_tids[i],
                        tid_map[words[i][2]]]
                       for i in range(n_words)]}

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    assert len(layers) == NL
    log('model loaded', lines)

    cap = {'ai': {}, 'trk': {}}
    state_fin = {'on': False}
    state_trk = {'on': False}
    fin_cap = {}
    inj = {'coef': None, 'scale': 0.0, 'vec': None,
           'layer': None}
    handles = []

    def pre_attn(li):
        def h(module, args, kwargs):
            x = args[0] if args \
                else kwargs.get('hidden_states')
            if x is None or x.dim() < 2:
                return None
            ret = None
            if inj['coef'] is not None \
                    and li in inj['coef']:
                c = inj['coef'][li]
                if c != 0.0:
                    x = x.clone()
                    x[:, 1, :] = x[:, 1, :] \
                        + c * inj['scale'] * inj['vec']
                if args:
                    ret = ((x,) + tuple(args[1:]),
                           kwargs)
                else:
                    nkw = dict(kwargs)
                    nkw['hidden_states'] = x
                    ret = (args, nkw)
            if state_trk['on']:
                cap['trk'].setdefault(li, []).append(
                    x[:, 1, :].detach().float().cpu()
                    .numpy().copy())
            elif inj['coef'] is None:
                cap['ai'].setdefault(li, []).append(
                    x.detach().float().cpu().numpy()
                    .copy())
            return ret
        return h

    def pre_norm(module, args, kwargs):
        if state_fin['on']:
            fin_cap['x'] = args[0][:, -1, :].detach() \
                .float().cpu().numpy().copy()
        return None

    for li in range(NL):
        handles.append(layers[li].self_attn
                       .register_forward_pre_hook(
                           pre_attn(li), with_kwargs=True))
    handles.append(model.model.norm
                   .register_forward_pre_hook(
                       pre_norm, with_kwargs=True))

    def clear_cap():
        for li in cap['ai']:
            del cap['ai'][li][:]
        for li in cap['trk']:
            del cap['trk'][li][:]

    def forward1(toks):
        clear_cap()
        with torch.no_grad():
            model(torch.tensor([toks], device='cuda'))
        return {li: cap['ai'][li][0]
                for li in cap['ai']}

    def forward_batch(toks_list, coef=None, scale=0.0):
        clear_cap()
        fin_cap.pop('x', None)
        state_fin['on'] = True
        inj['coef'] = coef
        inj['scale'] = float(scale)
        with torch.no_grad():
            model(torch.tensor(toks_list, device='cuda'))
        inj['coef'] = None
        state_fin['on'] = False
        return fin_cap['x'].astype(np.float64)

    def forward_trk(toks_list, coef=None, scale=0.0):
        clear_cap()
        inj['coef'] = coef
        inj['scale'] = float(scale)
        state_trk['on'] = True
        with torch.no_grad():
            model(torch.tensor(toks_list, device='cuda'))
        inj['coef'] = None
        state_trk['on'] = False
        return {li: np.stack(cap['trk'][li]).astype(
            np.float64) for li in cap['trk']}

    # ---------- pass 1: dirs_word rebuild ----------
    attn_store = {}
    for i, (_, _, w) in enumerate(words):
        attnin_all = forward1(
            [func_tid, tid_map[w]])
        for li in range(NL):
            attn_store[(i, li)] = \
                attnin_all[li].astype(np.float32)
        if (i + 1) % 20 == 0:
            log('pass1 [%d/%d]' % (i + 1, n_words), lines)
    d_w = np.zeros((NL, HID))
    for li in range(NL):
        X = np.stack([attn_store[(i, li)][0, 1]
                      for i in range(n_words)]) \
            .astype(np.float64)
        d_w[li] = X[lab_lang == 0].mean(0) \
            - X[lab_lang == 1].mean(0)
    dirs_word = np.stack([unit(d_w[li])
                          for li in range(NL)])
    a1_diff = float(np.abs(dirs_word - dirs27).max())
    a1_ok = bool(a1_diff < 1e-5)
    log('a1 dirs_word vs 2927 %.2e ok=%s'
        % (a1_diff, a1_ok), lines)

    _, _, Vt = np.linalg.svd(dirs_word,
                             full_matrices=False)
    Vt8 = Vt[:8]
    a3_diff = float(np.abs(Vt8 - Vt8_39).max())
    a3_ok = bool(a3_diff < 1e-6)
    log('a3 Vt8 vs 2939 %.2e ok=%s'
        % (a3_diff, a3_ok), lines)
    u35 = dirs_word[NL - 1]

    dcks_S = dcks_39[:, list(S_IDX)]
    Vt8_S = Vt8[list(S_IDX)]
    xdir = dcks_S @ Vt8_S
    a7_diff = float(np.abs(xdir @ Vt8_S.T - dcks_S).max())
    a7_ok = bool(a7_diff < 1e-9)
    log('a7 xdir identity %.2e ok=%s'
        % (a7_diff, a7_ok), lines)
    med_dS = float(np.median(np.linalg.norm(dcks_S,
                                            axis=1)))
    xdir_t = torch.tensor(xdir, device='cuda',
                          dtype=torch.bfloat16)
    inj['vec'] = xdir_t
    log('inj vec armed n=%d' % xdir.shape[0], lines)

    # ---------- baselines ----------
    fin_f1 = forward_batch(batch['func'])
    fin_f2 = forward_batch(batch['func'])
    a2_rel = float(np.abs(fin_f1 - fin_f2).max()
                   / max(float(np.abs(fin_f1).max()),
                         1e-30))
    a2_ok = bool(a2_rel < 1e-4)
    log('a2 baseline determinism rel %.2e ok=%s'
        % (a2_rel, a2_ok), lines)

    def reads(fin):
        return fin @ u35, fin @ Vt8.T

    proj_f0, c8_f0 = reads(fin_f1)
    a4_diff = float(np.abs(proj_f0 - s_base_35[ifu35])
                    .max())
    a4_ok = bool(a4_diff < 1e-4)
    fin_n0 = forward_batch(batch['null0'])
    proj_n0, c8_n0 = reads(fin_n0)
    a5_diff = float(np.abs(proj_n0 - s_base_35[in035])
                    .max())
    a5_ok = bool(a5_diff < 1e-4)
    sep_f = float(proj_f0[lab_lang == 0].mean()
                  - proj_f0[lab_lang == 1].mean())
    a6_ok = bool(sep_f > 0.0)
    sep_n = float(proj_n0[lab_lang == 0].mean()
                  - proj_n0[lab_lang == 1].mean())
    log('a4 %.2e a5 %.2e ok=%s/%s; a6 sep_f=%.2f '
        '(null0 %.2f, ratio null0/f=%.3f) ok=%s'
        % (a4_diff, a5_diff, a4_ok, a5_ok, sep_f,
           sep_n, sep_n / max(sep_f, 1e-30), a6_ok),
        lines)

    def arm(vec_t, layer, k, tag, spreads):
        projs = []
        ratios = []
        coef = {layer: 1.0}
        inj['vec'] = vec_t
        for _ in range(k):
            fin = forward_batch(batch['func'],
                                coef=coef, scale=S_SCAN)
            p, c8 = reads(fin)
            projs.append(p)
            cs = c8[:, list(S_IDX)] \
                - c8_f0[:, list(S_IDX)]
            ratios.append(float(np.median(
                np.linalg.norm(cs, axis=1))))
        inj['vec'] = xdir_t
        P = np.stack(projs)
        spreads[tag] = float('%.2e' % float(
            np.abs(P - P.mean(0)).max()))
        p_med = np.median(P, axis=0)
        sep_m = float(p_med[lab_lang == 0].mean()
                      - p_med[lab_lang == 1].mean())
        ratio_m = float(np.median(ratios)) \
            / max(med_dS, 1e-30)
        return p_med, sep_m, ratio_m

    anchor_prelim = bool(a0_ok and a8_ok and a1_ok
                         and a3_ok and a2_ok and a4_ok
                         and a5_ok and a6_ok and a7_ok
                         and a10_ok)
    verdict = None
    spreads = {}
    a11_diff = None
    a9_diffs = None
    a9_ok = False
    a12a_diffs = None
    a12a_ok = False
    save = {}
    A = {}
    E = {}

    if anchor_prelim:
        # ---------- T2-lite scan (a9 anchor) ----------
        rng = np.random.default_rng(SEED_RND)
        xc = (Vt8 @ xdir.T).T
        xh = xc / np.linalg.norm(
            xc, axis=1, keepdims=True)
        rnd_sub = []
        for _ in range(N_SUB_RND):
            g = rng.standard_normal((n_words, 8))
            g = g - np.sum(g * xh, axis=1,
                           keepdims=True) * xh
            g = g / np.linalg.norm(g, axis=1,
                                   keepdims=True)
            rnd_sub.append(g @ Vt8)
        rnd_full = [unit(rng.standard_normal(HID))]
        raw_x = {}
        t2lite = {}
        for li in SEL_LAYERS:
            _, _, ratio_x = arm(
                xdir_t, li, K_SCAN,
                'xdir|%d' % li, spreads)
            raw_x[li] = ratio_x
            t2lite['L%d' % li] = {
                'xdir_ratio': round(ratio_x, 4)}
            for j, v in enumerate(rnd_sub):
                vt = torch.tensor(
                    v, device='cuda',
                    dtype=torch.bfloat16)
                _, _, ratio_r = arm(
                    vt, li, K_SCAN,
                    'subrnd%d|%d' % (j, li), spreads)
                t2lite['L%d' % li][
                    'sub_rnd%d' % j] = round(ratio_r, 4)
            for j, v in enumerate(rnd_full):
                vt = torch.tensor(
                    v, device='cuda',
                    dtype=torch.bfloat16)
                _, _, ratio_r = arm(
                    vt, li, K_SCAN,
                    'fullrnd%d|%d' % (j, li), spreads)
                t2lite['L%d' % li][
                    'full_rnd%d' % j] = round(ratio_r, 4)
            log('T2lite L%d xdir ratio=%.4f'
                % (li, ratio_x), lines)
        a9_diffs = {
            '15': abs(raw_x[15] - REF_2945['15']),
            '17': abs(raw_x[17] - REF_2945['17'])}
        a9_ok = bool(a9_diffs['15'] < A9_TOL
                     and a9_diffs['17'] < A9_TOL)
        log('a9 2945 repro diffs %s ok=%s'
            % ({k: ('%.2e' % v)
                for k, v in a9_diffs.items()}, a9_ok),
            lines)

        # ---------- T4 extended capture ----------
        base_trk = forward_trk(batch['func'])
        Dl_store = {}
        trk_profA = {}
        trk_profB = {}
        for li_inj in TRK_LAYERS:
            inj_trk = forward_trk(
                batch['func'],
                coef={li_inj: 1.0}, scale=S_SCAN)
            layers_list = list(range(li_inj, NL))
            Dstack = np.stack(
                [(inj_trk[li] - base_trk[li])[0]
                 for li in layers_list])
            Dl_store['L%d' % li_inj] = Dstack
            save['Dl_L%d' % li_inj] = Dstack
            profA = {}
            profB = {}
            for k1, li in enumerate(layers_list):
                Dl = Dstack[k1]
                # variant A: 3002 T3 formula on the
                # (1, n, hid) shaped view (axis bug
                # reproduction - norm / projection
                # ACROSS CELLS per hidden dim)
                Dl3 = Dl[None]
                nrA = float(np.median(
                    np.linalg.norm(Dl3, axis=1))) \
                    / S_SCAN
                pjA = float(np.median(
                    np.sum(Dl3 * xdir, axis=1)
                    / S_SCAN))
                # variant B: the DECLARED per-cell
                # quantity (corrected T3)
                nrB = float(np.median(
                    np.linalg.norm(Dl, axis=1))) \
                    / S_SCAN
                pjB = float(np.median(
                    np.sum(Dl * xdir, axis=1)
                    / S_SCAN))
                profA[li] = {
                    'trk_ratio': round(nrA, 4),
                    'xdir_proj': round(pjA, 4)}
                profB[li] = {
                    'trk_ratio': round(nrB, 4),
                    'xdir_proj': round(pjB, 4)}
            trk_profA['L%d' % li_inj] = profA
            trk_profB['L%d' % li_inj] = profB
            log('T4 L%d inj captured %d layers x %d '
                'cells (A=3002-formula, B=corrected)'
                % (li_inj, len(layers_list),
                   n_words), lines)

        # a12a bug-identity anchor: variant A (3002 T3
        # formula) recomputed from THIS run's Dl must
        # reproduce 3002's stored profiles bit-level -
        # proves the run3 a12 failure (3.06e3) is
        # exactly the 3002 T3 axis bug, not new physics
        a12a_diffs = {}
        for key, prof, ref in [
                ('L4', trk_profA['L4'], trk02_L4),
                ('L17', trk_profA['L17'], trk02_L17)]:
            diffs = []
            for li_s, pr in prof.items():
                li_i = int(li_s)
                ref_p = ref.get(str(li_i))
                if ref_p is None:
                    diffs.append(0.0)
                    continue
                diffs.append(abs(
                    pr['trk_ratio']
                    - ref_p['trk_ratio']))
                diffs.append(abs(
                    pr['xdir_proj']
                    - ref_p['xdir_proj']))
            a12a_diffs[key] = max(diffs)
        a12a_ok = bool(all(v < A12_TOL
                           for v in a12a_diffs.values()))
        log('a12a 3002-formula repro diffs %s ok=%s '
            '(bug-identity anchor)'
            % ({k: ('%.2e' % v)
                for k, v in a12a_diffs.items()}, a12a_ok),
            lines)
        log('T3_corrected (variant B) key points: '
            'L4inj trk %.1f@4 -> %.2f@5; L17inj trk '
            '%.1f@17 -> %.1f@18 -> %.1f@31; proj '
            '%.0f@17 -> %.0f@31'
            % (trk_profB['L4'][4]['trk_ratio'],
               trk_profB['L4'][5]['trk_ratio'],
               trk_profB['L17'][17]['trk_ratio'],
               trk_profB['L17'][18]['trk_ratio'],
               trk_profB['L17'][31]['trk_ratio'],
               trk_profB['L17'][17]['xdir_proj'],
               trk_profB['L17'][31]['xdir_proj']),
            lines)

        # ---------- A operator structure ----------
        # null calibration: row-normalized gaussian
        rngn = np.random.default_rng(SEED_NULL)
        Gn = rngn.standard_normal((n_words, HID))
        Gn = Gn / np.linalg.norm(
            Gn, axis=1, keepdims=True)
        r1_null, _, _, _ = op_structure(Gn)
        A['null_r1_rowgauss'] = round(r1_null, 6)
        A['per_inj'] = {}
        for key in ('L4', 'L17'):
            li_inj = int(key[1:])
            layers_list = list(range(li_inj, NL))
            Dstack = Dl_store[key]
            per = {}
            for k1, li in enumerate(layers_list):
                Dl = Dstack[k1]
                r1, erank, medcos, v1 = op_structure(Dl)
                # cos of top direction v1 with the
                # cell-mean delta direction
                mean_dir = unit(Dl.mean(0))
                cosv1mean = float(abs(
                    v1 @ mean_dir))
                per[li] = {
                    'r1': round(r1, 6),
                    'erank': round(erank, 4),
                    'medcos': round(medcos, 6),
                    'cos_v1_meandelta':
                        round(cosv1mean, 6)}
            A['per_inj'][key] = per
            band = [per[li]['r1']
                    for li in range(li_inj + 1, NL)]
            bandc = [per[li]['medcos']
                     for li in range(li_inj + 1, NL)]
            log('A %s: r1 l=%d..%d med=%.4f '
                'medcos med=%.4f (null r1=%.2e)'
                % (key, li_inj + 1, NL - 1,
                   float(np.median(band)),
                   float(np.median(bandc)),
                   r1_null), lines)

        # ---------- E energy account ----------
        for key in ('L4', 'L17'):
            li_inj = int(key[1:])
            layers_list = list(range(li_inj, NL))
            Dstack = Dl_store[key]
            src = Dstack[0]
            uh = src / np.maximum(
                np.linalg.norm(src, axis=1,
                               keepdims=True), 1e-30)
            src_n = np.linalg.norm(src, axis=1)
            prof = {}
            for k1, li in enumerate(layers_list):
                Dl = Dstack[k1]
                nrm = np.linalg.norm(Dl, axis=1)
                aligned = np.sum(Dl * uh, axis=1) ** 2
                frac = aligned / np.maximum(
                    nrm ** 2, 1e-30)
                amp = nrm / np.maximum(src_n, 1e-30)
                prof[li] = {
                    'aligned_frac': round(
                        float(np.median(frac)), 6),
                    'amp_ratio': round(
                        float(np.median(amp)), 6)}
            E[key] = prof
            log('E %s: aligned_frac@%d=%.4f '
                'amp_ratio@%d=%.4f'
                % (key, li_inj + 1,
                   prof[li_inj + 1]['aligned_frac'],
                   li_inj + 1,
                   prof[li_inj + 1]['amp_ratio']),
                lines)

        # ---------- verdict (PRIMARY = L17 band) ----------
        band_r1 = [A['per_inj']['L17'][li]['r1']
                   for li in AMP_BAND]
        band_cos = [A['per_inj']['L17'][li]['medcos']
                    for li in AMP_BAND]
        med_r1 = float(np.median(band_r1))
        med_cos = float(np.median(band_cos))
        A['L17_band'] = {
            'layers': list(AMP_BAND),
            'med_r1': round(med_r1, 6),
            'med_medcos': round(med_cos, 6),
            'r1_shared_gate': R1_SHARED,
            'cos_shared_gate': COS_SHARED,
            'r1_scatter_gate': R1_SCATTER,
            'cos_scatter_gate': COS_SCATTER}
        a11_diff = max(spreads.values())
        a11_ok = bool(a11_diff < 1e-6)
        log('a11 same-session determinism %.2e ok=%s'
            % (a11_diff, a11_ok), lines)
        if not (a9_ok and a11_ok and a12a_ok):
            verdict = 'anchor_fail_all_void'
        elif (med_r1 >= R1_SHARED
                and med_cos >= COS_SHARED):
            verdict = 'shared_lowrank_operator_qwen'
        elif (med_r1 < R1_SCATTER
                or med_cos < COS_SCATTER):
            verdict = 'cell_scatter_operator_qwen'
        else:
            verdict = 'mixed_operator_qwen'
    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('VERDICT %s' % verdict, lines)

    elapsed = time.monotonic() - t0

    anchors = {
        'a0_words': a0_ok, 'a8_collision': a8_ok,
        'a1_diff': a1_diff, 'a2_rel': a2_rel,
        'a3_diff': a3_diff, 'a4_diff': a4_diff,
        'a5_diff': a5_diff, 'a6_sep_f': sep_f,
        'a7_diff': a7_diff,
        'a9_diffs': a9_diffs,
        'a9_ok': a9_ok if anchor_prelim else None,
        'a10_ok': a10_ok, 'a11_diff': a11_diff,
        'a12a_diffs': a12a_diffs,
        'a12a_ok': a12a_ok if anchor_prelim else None,
    }
    res = {
        'phase': 3004,
        'final_verdict': verdict,
        'anchor_all_ok': bool(
            a0_ok and a8_ok and a1_ok and a3_ok
            and a2_ok and a4_ok and a5_ok and a6_ok
            and a7_ok and a10_ok and a9_ok
            and a11_ok and a12a_ok),
        'anchors': anchors,
        'scale': {'sep_f': round(sep_f, 2),
                  'sep_null0': round(sep_n, 2),
                  'med_dS': round(med_dS, 4)},
        'T2lite': t2lite if anchor_prelim else None,
        'A': A, 'E': E,
        'T3_corrected': trk_profB,
        'axis_bug': {
            'status': 'confirmed_and_registered',
            'affected': ['phase3001 T3', 'phase3002 T3'],
            'detail': 'forward_trk stacked a single '
                      'batch capture -> (1, n_cells, '
                      'hid); norm(Dl, axis=1) and '
                      'sum(Dl*xdir, axis=1) therefore '
                      'operated ACROSS CELLS per hidden '
                      'dim, not per cell over hidden '
                      'dims; stored trk_ratio / '
                      'xdir_proj are a different '
                      '(mis-declared) quantity',
            'evidence': 'a12a: the 3002 formula '
                        'reproduces the 3002 stored '
                        'profiles bit-level from THIS '
                        'run Dl; corrected per-cell '
                        'values differ by ~med||xdir||^2',
            'impact': 'T3 was DESCRIPTIVE in 3001/3002 - '
                      'verdicts (both from T1) '
                      'unaffected; qualitative eraser/'
                      'amplification conclusions SURVIVE '
                      'under corrected values (qwen: L4 '
                      'injection loses ~93% amplitude '
                      'at L5; L17 injection drops ~70% '
                      'at L18 then regrows to ~1.66x '
                      'injection by L31 with growing '
                      'xdir-aligned component); GLM4 '
                      'T3 recomputation = next phase; '
                      'TRK_DROP=0.5 eraser gate was '
                      'calibrated on the wrong units '
                      'and is retired',
        },
        'tags': PREREG['tags'],
        'elapsed_s': round(elapsed, 1),
        'correction_note':
            'run1: NameError S_IDX (constant omitted '
            'from the 3002 copy), never reached T4; '
            'run2: forward_trk stacks a single batch '
            'capture -> (1, n, hid) per-layer arrays; '
            'SVD on the 3-D slice crashed (TypeError '
            'float()); fixed by [0]-indexing in the '
            'Dstack build; run2 a12 vs 3002 = 0.0 was '
            'VACUOUS (same-shape garbage quantity); '
            'run3: full pass but a12 vs 3002 stored T3 '
            'profiles = 3.06e+03 (L4 and L17 both) - '
            'diagnosed with a dedicated probe: 3002 '
            '(and 3001) T3 trk_ratio/xdir_proj were '
            'computed on (1, n, hid) arrays with '
            'axis=1, i.e. ACROSS CELLS per hidden dim '
            '- a mis-declared quantity; the 3002 '
            'formula reproduces its stored profiles '
            'bit-level (4.9e-5 = 4dp rounding) while '
            'the declared per-cell quantity differs by '
            '~med||xdir||^2 = 3079; verdict void by '
            'protocol; run4: a12 redefined as a12a '
            'bug-identity anchor (3002-formula '
            'reproduction < 1e-3) + T3_corrected '
            '(per-cell values) registered; '
            'authoritative',
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    save['dirs_word'] = dirs_word
    save['Vt8'] = Vt8
    save['u35'] = u35
    save['xdir'] = xdir
    save['words'] = np.array(['%s:%s:%s' % w
                              for w in words],
                             dtype=object)
    save['labels_lang'] = lab_lang
    save['null0_tids'] = np.array(null0_tids)
    save['proj_base'] = proj_f0
    save['proj_null'] = proj_n0
    npz_path = os.path.join(
        OUT, 'omega_p4_operator_structure_qwen.npz')
    np.savez_compressed(npz_path, **save)

    seal = {
        'npz_sha256_8': sha8(npz_path),
        'result_sha256_8': sha8(
            os.path.join(OUT, 'result.json')),
        'exec_sha256_8': sha8(
            os.path.join(OUT, 'execution.json')),
    }
    with open(os.path.join(OUT, 'seal.json'), 'w',
              encoding='utf-8') as f:
        json.dump(seal, f, indent=2)
    log('sealed %s' % json.dumps(seal), lines)
    log('elapsed %.1fs' % elapsed, lines)


if __name__ == '__main__':
    main()
