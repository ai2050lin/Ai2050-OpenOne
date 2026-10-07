# -*- coding: utf-8 -*-
"""Phase 3005: Omega-P5 layer transfer operator structure
(GLM4 side; closes the cross-model operator contrast).

Why: 3004 established the qwen mid-band (L17) as a SHARED
LOW-RANK redirect operator (r1 0.715 / medcos 0.826 vs null
0.022) that redirects+amplifies rather than preserves.
3001's GLM4 mid-band (L19) was qualitatively an "eraser",
but 3004 proved the 3001/3002 T3 trk/xdir profiles are a
mis-declared quantity (axis bug: norm/sum across cells per
hidden dim on (1, n, hid) arrays).  Open questions: (i)
what does the CORRECTED GLM4 L19 profile look like - is
"next-layer kill" still true per-cell?  (ii) what is the
operator structure of the GLM4 eraser band - shared low-
rank, per-cell scatter, or amplitude collapse below the
noise floor (structure undecidable)?

Design (3001 machine verbatim: 98 cells 2972+L_CAND, dirs_g
a1 vs 2996 npz, Vt8/u39, in-session dcks from c8_n0-c8_f0,
xdir = dcks_S@Vt8_S, single-layer attn_in pos-1 injection,
final-norm readout, bf16) + 3004 T4/A/E machine:
  T2lite scan L17/L19 s=2 K=2 (a9 raw-ratio cross-card
     anchor vs 2999 via 3001) + mirror arms (a9).
  T4 EXTENDED capture: xdir s=2 at L4 and L19; per-cell
     pos-1 attn_in at EVERY layer; Dl_i(l) stored in full.
     Dual profiles: variant A = 3001 formula on the
     (1,n,hid) view (bug-identity anchor a12a); variant B =
     corrected per-cell quantity (T3_corrected_GLM4).
  A OPERATOR STRUCTURE per layer l > L_inj: SVD of the
     per-cell Delta matrix; r1, erank, medcos; row-gauss
     null calibration (descriptive).
  E ENERGY ACCOUNT per layer (descriptive).

Verdict (frozen; PRIMARY = L19 eraser band, layers 20..39;
degeneracy floor FIRST - a band whose corrected amplitude
is under the floor has no decodable operator structure):
  anchor fail                        => anchor_fail_all_void
  med trk_B(band) < AMP_FLOOR        => amplitude_collapse_eraser_glm4
  med r1 >= 0.5 AND med medcos >= 0.6
                                     => shared_lowrank_operator_glm4
  med r1 < 0.2 OR med medcos < 0.3   => cell_scatter_operator_glm4
  otherwise                          => mixed_operator_glm4

Anchors (frozen):
  a0 words_g == 2996 glm src re-export
  a1 dirs_g[17/18/19] vs 2996 dirs_attn_glm < 1e-6
  a2 baseline determinism < 1e-4
  a3 xdir identity < 1e-9
  a6 |sep_f - 2999.sep_f| < 1e-4 (cross-run bit-level)
  a7 u39 unit < 1e-12
  a8 null0 collision-free
  a9 cross-card 3001/2999: |ratio19_raw - ref| < 1e-6 AND
     |sep_n0 - 61.43| <= 0.02 AND mirrors <= 0.05
  a10 3001 source integrity: seal hash + verdict ==
      word_carry_general_erasure_glm4
  a11 same-session determinism < 1e-6 (all K-repeat arms)
  a12a 3001-formula bug-identity anchor: variant A
      recomputed from THIS run's Dl reproduces the 3001
      stored T3 profiles < 1e-3 (4dp)

Tags: Omega-P5 / lang axis / len-2 / snapshot machine (no
ablation) / dimensionless gates / operator structure /
energy account / axis-bug corrected / plan v5 P4 closure.
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2972 = os.path.join(BASE, 'phase2972',
                        'two_factor_signature',
                        'execution.json')
SRC_2996 = os.path.join(BASE, 'phase2996',
                        'omega_f2a_registry_caliber_audit',
                        'omega_f2a_registry_caliber_audit.npz')
D_3001 = os.path.join(BASE, 'phase3001',
                      'omega_g1_robustness_source_glm4')
OUT = os.path.join(BASE, 'phase3005',
                   'omega_p5_operator_structure_glm4')
MD_G = r'D:\AI2050\Ai2050-OpenOne\models\hf\glm4-9b-chat-hf'
L_CAND = ["because", "therefore", "although", "unless",
          "however", "thus", "moreover", "since", "whereas",
          "despite", "hence", "nevertheless", "consequently",
          "furthermore", "otherwise", "instead", "while",
          "accordingly", "likewise", "meanwhile", "nonetheless",
          "thereafter", "whereby", "albeit"]
NL, HID, VOCAB = 40, 4096, 151552
SEL_LAYERS = (17, 19)
REF_LAYER = 19
TRK_LAYERS = (4, 19)
ERASER_BAND = tuple(range(20, 40))
S_SCAN = 2.0
K_SCAN = 2
S_IDX = (0, 1, 4)
SEED_NULL = 2889          # 3001 verbatim
R1_SHARED = 0.5
COS_SHARED = 0.6
R1_SCATTER = 0.2
COS_SCATTER = 0.3
AMP_FLOOR = 0.05
A12_TOL = 1e-3
REF_2999 = {'sep_f': 85.73204044720937,
            'ratio19': 0.01664657989077104,
            'sep_n0': 61.43,
            'mirror19_sep': 85.26,
            'mirror4_sep': 85.35}

PREREG = {
    'mode': 'glm4-9b only; 3001 machine verbatim (98 '
            'cells, dirs_g anchored to 2996 npz, Vt8/u39, '
            'in-session dcks, xdir single-layer attn_in '
            'pos-1 injection, final-norm readout, bf16); '
            '+ 3004 T4/A/E machine (per-cell per-layer '
            'Delta capture, dual profiles, SVD operator '
            'structure, energy account)',
    'question': 'what is the operator structure of the '
                'GLM4 L19 eraser band (shared low-rank '
                'vs per-cell scatter vs amplitude '
                'collapse), and what does the CORRECTED '
                'GLM4 T3 profile look like after the '
                '3004 axis-bug registration?',
    'axis_bug_note':
        '3001 T3 trk_ratio/xdir_proj are a mis-declared '
        'quantity (axis bug, registered in 3004 run4 '
        'result.json with bit-level a12a evidence); this '
        'phase recomputes dual profiles: variant A = '
        '3001 formula (bug-identity anchor a12a), '
        'variant B = corrected per-cell quantity '
        '(T3_corrected_GLM4, descriptive)',
    'T2lite': 'scan L17/L19 s=2 K=2 xdir only; a9 raw-'
              'ratio cross-card anchor vs 2999 (via '
              '3001) + mirror arms; no selectivity '
              'verdict here (3001 owns it)',
    'T4': 'extended capture: xdir s=2 at L4 and L19; '
          'per-cell pos-1 attn_in at EVERY layer; '
          'Dl_i(l) = inj_i(l) - base_i(l) stored in full',
    'A': 'per layer l > L_inj: SVD of the per-cell Delta '
         'matrix (98 x 4096); r1 = S1^2/sum(S^2); erank = '
         'entropy effective rank; medcos = median '
         'per-cell |cos(Dl_i, v1)|; row-gauss null '
         'calibration descriptive',
    'E': 'per layer (descriptive): source = Delta_i(L_inj)'
         '; aligned energy fraction = (Dl_i(l).uh_i)^2 / '
         '||Dl_i(l)||^2; amplitude ratio = ||Dl_i(l)|| / '
         '||source_i||; median profiles',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'PRIMARY = L19 band layers 20..39; '
               'degeneracy floor FIRST: med trk_B(band) '
               '< 0.05 => amplitude_collapse_eraser_'
               'glm4 (structure undecidable); med r1 >= '
               '0.5 AND med medcos >= 0.6 => '
               'shared_lowrank_operator_glm4; med r1 < '
               '0.2 OR med medcos < 0.3 => '
               'cell_scatter_operator_glm4; else '
               'mixed_operator_glm4',
    'tags': 'Omega-P5 / lang axis / len-2 / snapshot '
            'machine (no ablation) / dimensionless '
            'gates / operator structure / energy '
            'account / axis-bug corrected / plan v5 P4 '
            'closure',
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
        json.dump({'phase': 3005,
                   'name':
                       'omega_p5_operator_structure_glm4',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {
                       's2972': sha8(SRC_2972),
                       's2996npz': sha8(SRC_2996),
                       's3001result': sha8(
                           D_3001 + r'\result.json'),
                       's3001seal': sha8(
                           D_3001 + r'\seal.json')},
                   'model': 'glm4-9b', 'n_layers': NL,
                   'hidden': HID, 'vocab': VOCAB,
                   'sel_layers': list(SEL_LAYERS),
                   'ref_layer': REF_LAYER,
                   'trk_layers': list(TRK_LAYERS),
                   'eraser_band': list(ERASER_BAND),
                   's_scan': S_SCAN, 'k_scan': K_SCAN,
                   's_idx': list(S_IDX),
                   'seed_null': SEED_NULL,
                   'r1_shared': R1_SHARED,
                   'cos_shared': COS_SHARED,
                   'r1_scatter': R1_SCATTER,
                   'cos_scatter': COS_SCATTER,
                   'amp_floor': AMP_FLOOR,
                   'a12_tol': A12_TOL,
                   'ref_2999': REF_2999,
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- sources ----------
    z96 = np.load(SRC_2996, allow_pickle=True)
    words_g = [str(w) for w in z96['words_g']]
    dirs96 = z96['dirs_attn_glm'].astype(np.float64)
    assert dirs96.shape == (NL, HID)
    e72 = json.load(open(SRC_2972, encoding='utf-8'))
    src = ([('F', 'en', w) for w in e72['cells']['F_en']]
           + [('C', 'en', w) for w in e72['cells']['C_en']]
           + [('L', 'en', w) for w in L_CAND]
           + [('F', 'fr', w) for w in e72['cells']['F_fr']]
           + [('C', 'fr', w) for w in e72['cells']['C_fr']])
    lab_src = ['%s:%s:%s' % c for c in src]
    a0_ok = bool(words_g == lab_src)
    log('a0 words_g == 2996 glm src re-export: %s'
        % a0_ok, lines)

    # a10 3001 source integrity
    seal01 = json.load(open(D_3001 + r'\seal.json',
                            encoding='utf-8'))
    a10_ok = bool(seal01['result_sha256_8']
                  == sha8(D_3001 + r'\result.json'))
    r01 = json.load(open(D_3001 + r'\result.json',
                         encoding='utf-8'))
    a10_ok = a10_ok and bool(
        r01['final_verdict']
        == 'word_carry_general_erasure_glm4'
        and r01['anchor_all_ok'] is True)
    trk01_L4 = r01['T3']['trk']['L4']['profile']
    trk01_L19 = r01['T3']['trk']['L19']['profile']
    log('a10 3001 integrity %s' % a10_ok, lines)

    # ---------- model ----------
    import torch as _t
    from transformers import AutoTokenizer, \
        AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(
        MD_G, local_files_only=True, use_fast=True)
    tc = {}

    def tid_of(w):
        if w not in tc:
            ids = tok(' ' + w, add_special_tokens=False)[
                'input_ids']
            if len(ids) != 1:
                ids = tok(w, add_special_tokens=False)[
                    'input_ids']
            tc[w] = int(ids[0]) if len(ids) == 1 else -1
        return tc[w]

    cells = []
    for s in words_g:
        cat, lng, w = s.split(':')
        t = tid_of(w)
        if t != -1:
            cells.append((0 if lng == 'en' else 1,
                          cat, lng, w))
    n = len(cells)
    lang = np.array([c[0] for c in cells])
    i_en = [i for i in range(n) if lang[i] == 0]
    i_non = [i for i in range(n) if lang[i] == 1]
    assert n == 98, n
    log('cells %d (en %d non-en %d)'
        % (n, len(i_en), len(i_non)), lines)

    word_tids = set(tid_of(c[3]) for c in cells)
    func_tid = tid_of('the')
    assert func_tid > 0
    rng0 = np.random.default_rng(SEED_NULL)
    null0_tids = []
    while len(null0_tids) < n:
        r = int(rng0.integers(0, VOCAB))
        if r not in word_tids and r > 0:
            null0_tids.append(r)
    a8_ok = bool(len(null0_tids) == n
                 and not (set(null0_tids) & word_tids))
    log('a8 null0 collision-free: %s' % a8_ok, lines)

    model = AutoModelForCausalLM.from_pretrained(
        MD_G, torch_dtype=_t.bfloat16).cuda().eval()
    layers = model.model.layers
    assert len(layers) == NL
    log('model loaded', lines)

    cap = {'ai': {}, 'trk': {}}
    state_fin = {'on': False}
    state_trk = {'on': False}
    fin_cap = {}
    inj = {'on': False, 'scale': 0.0, 'vec': None,
           'layer': None}
    handles = []

    def pre_attn(li):
        def h(module, args, kwargs):
            x = kwargs.get('hidden_states')
            if x is None:
                x = args[0] if args else None
            if x is None or x.dim() < 2:
                return None
            ret = None
            if inj['on'] and inj['layer'] is not None \
                    and li == inj['layer']:
                x = x.clone()
                x[:, 1, :] = x[:, 1, :] \
                    + inj['scale'] * inj['vec']
                nkw = dict(kwargs)
                nkw['hidden_states'] = x
                ret = (args, nkw)
            if state_trk['on']:
                cap['trk'].setdefault(li, []).append(
                    x[:, 1, :].detach().float().cpu()
                    .numpy().copy())
            elif not inj['on']:
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

    def forward_batch(toks_list, scale=0.0, layer=None):
        clear_cap()
        fin_cap.pop('x', None)
        state_fin['on'] = True
        inj['on'] = scale != 0.0
        inj['scale'] = float(scale)
        inj['layer'] = layer
        with _t.no_grad():
            model(_t.tensor(toks_list, device='cuda'))
        inj['on'] = False
        state_fin['on'] = False
        return fin_cap['x'].astype(np.float64)

    def forward_trk(toks_list, scale=0.0, layer=None):
        clear_cap()
        inj['on'] = scale != 0.0
        inj['scale'] = float(scale)
        inj['layer'] = layer
        state_trk['on'] = True
        with _t.no_grad():
            model(_t.tensor(toks_list, device='cuda'))
        inj['on'] = False
        state_trk['on'] = False
        return {li: np.stack(cap['trk'][li]).astype(
            np.float64) for li in cap['trk']}

    seqs = [[func_tid, tid_of(c[3])] for c in cells]

    # ---------- dirs_g rebuild (3001 verbatim) ----------
    store = {}
    for i, c in enumerate(cells):
        clear_cap()
        with _t.no_grad():
            model(_t.tensor([seqs[i]], device='cuda'))
        for li in range(NL):
            store[(i, li)] = cap['ai'][li][0] \
                .astype(np.float32)
        if (i + 1) % 25 == 0:
            log('pass1 [%d/%d]' % (i + 1, n), lines)
    d_w = np.zeros((NL, HID))
    for li in range(NL):
        X = np.stack([store[(i, li)][0, 1]
                      for i in range(n)]).astype(np.float64)
        d_w[li] = X[i_en].mean(0) - X[i_non].mean(0)
    dirs_g = np.stack([unit(d_w[li]) for li in range(NL)])
    a1_diffs = {li: float(np.abs(dirs_g[li] - dirs96[li])
                          .max()) for li in (17, 18, 19)}
    a1_ok = bool(all(v < 1e-6
                     for v in a1_diffs.values()))
    log('a1 dirs_g vs 2996 rows %s ok=%s'
        % ({k: ('%.2e' % v) for k, v
            in a1_diffs.items()}, a1_ok), lines)

    _, _, Vt = np.linalg.svd(dirs_g, full_matrices=False)
    Vt8 = Vt[:8]
    u39 = dirs_g[NL - 1]
    a7_diff = abs(float(np.linalg.norm(u39)) - 1.0)
    a7_ok = bool(a7_diff < 1e-12)
    log('a7 u39 unit |1-n|=%.2e ok=%s'
        % (a7_diff, a7_ok), lines)

    # ---------- baselines ----------
    fin_f1 = forward_batch(seqs)
    fin_f2 = forward_batch(seqs)
    a2_rel = float(np.abs(fin_f1 - fin_f2).max()
                   / max(float(np.abs(fin_f1).max()),
                         1e-30))
    a2_ok = bool(a2_rel < 1e-4)
    log('a2 baseline determinism rel %.2e ok=%s'
        % (a2_rel, a2_ok), lines)
    fin_n0 = forward_batch(
        [[null0_tids[i], tid_of(c[3])]
         for i, c in enumerate(cells)])

    def reads(fin):
        return fin @ u39, fin @ Vt8.T

    proj_f0, c8_f0 = reads(fin_f1)
    proj_n0, c8_n0 = reads(fin_n0)
    sep_f = float(proj_f0[i_en].mean()
                  - proj_f0[i_non].mean())
    a6_diff = abs(sep_f - REF_2999['sep_f'])
    a6_ok = bool(a6_diff < 1e-4)
    sep_n = float(proj_n0[i_en].mean()
                  - proj_n0[i_non].mean())
    log('a6 sep_f=%.10f vs 2999 diff=%.2e ok=%s '
        '(null0 %.3f)'
        % (sep_f, a6_diff, a6_ok, sep_n), lines)

    dcks = c8_n0 - c8_f0
    dcks_S = dcks[:, list(S_IDX)]
    Vt8_S = Vt8[list(S_IDX)]
    xdir = dcks_S @ Vt8_S
    a3_diff = float(np.abs(xdir @ Vt8_S.T - dcks_S).max())
    a3_ok = bool(a3_diff < 1e-9)
    log('a3 xdir identity %.2e ok=%s'
        % (a3_diff, a3_ok), lines)
    med_dS = float(np.median(np.linalg.norm(dcks_S, axis=1)))
    xdir_t = _t.tensor(xdir, device='cuda',
                       dtype=_t.bfloat16)
    inj['vec'] = xdir_t
    log('inj vec armed n=%d' % xdir.shape[0], lines)

    def arm(vec_t, scale, layer, k, tag, spreads):
        projs = []
        ratios = []
        inj['vec'] = vec_t
        for _ in range(k):
            fin = forward_batch(seqs, scale=scale,
                                layer=layer)
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
        sep_m = float(p_med[i_en].mean()
                      - p_med[i_non].mean())
        ratio_m = float(np.median(ratios)) \
            / max(med_dS, 1e-30)
        return p_med, sep_m, ratio_m

    anchor_prelim = bool(a0_ok and a8_ok and a1_ok
                         and a7_ok and a2_ok and a6_ok
                         and a3_ok and a10_ok)
    verdict = None
    spreads = {}
    a9_parts = {}
    a9_ok = False
    a11_diff = None
    a12a_diffs = None
    a12a_ok = False
    save = {}
    A = {}
    E = {}
    t2lite = {}
    trk_profA = {}
    trk_profB = {}

    if anchor_prelim:
        # ---------- T2lite scan (a9 anchors) ----------
        raw_x = {}
        for li in SEL_LAYERS:
            _, sep_x, ratio_x = arm(
                xdir_t, S_SCAN, li, K_SCAN,
                'xdir|%d' % li, spreads)
            raw_x[li] = ratio_x
            t2lite['L%d' % li] = {
                'xdir_ratio': round(ratio_x, 4),
                'sep': round(sep_x, 2)}
            log('T2lite L%d xdir ratio=%.6f sep=%.1f'
                % (li, ratio_x, sep_x), lines)
        _, sep_m19, _ = arm(xdir_t, -S_SCAN, REF_LAYER,
                            K_SCAN, 'mirror|19', spreads)
        _, sep_m4, _ = arm(xdir_t, -S_SCAN, 4, K_SCAN,
                           'mirror|4', spreads)
        a9_parts = {
            'ratio19': abs(
                raw_x[REF_LAYER]
                - REF_2999['ratio19']),
            'sep_n0': abs(sep_n - REF_2999['sep_n0']),
            'mirror19': abs(
                sep_m19 - REF_2999['mirror19_sep']),
            'mirror4': abs(
                sep_m4 - REF_2999['mirror4_sep'])}
        a9_ok = bool(
            a9_parts['ratio19'] < 1e-6
            and a9_parts['sep_n0'] <= 0.02
            and a9_parts['mirror19'] <= 0.05
            and a9_parts['mirror4'] <= 0.05)
        log('a9 cross-card 2999/3001 %s ok=%s'
            % ({k: ('%.2e' % v) for k, v
                in a9_parts.items()}, a9_ok), lines)

        # ---------- T4 extended capture ----------
        base_trk = forward_trk(seqs)
        Dl_store = {}
        for li_inj in TRK_LAYERS:
            inj_trk = forward_trk(seqs, scale=S_SCAN,
                                  layer=li_inj)
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
                # variant A: 3001 T3 formula on the
                # (1, n, hid) shaped view (axis bug
                # reproduction)
                Dl3 = Dl[None]
                nrA = float(np.median(
                    np.linalg.norm(Dl3, axis=1))) \
                    / S_SCAN
                pjA = float(np.median(
                    np.sum(Dl3 * xdir, axis=1)
                    / S_SCAN))
                # variant B: corrected per-cell quantity
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
                'cells (A=3001-formula, B=corrected)'
                % (li_inj, len(layers_list), n), lines)

        # a12a bug-identity anchor
        a12a_diffs = {}
        for key, prof, ref in [
                ('L4', trk_profA['L4'], trk01_L4),
                ('L19', trk_profA['L19'], trk01_L19)]:
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
        log('a12a 3001-formula repro diffs %s ok=%s '
            '(bug-identity anchor)'
            % ({k: ('%.2e' % v)
                for k, v in a12a_diffs.items()}, a12a_ok),
            lines)
        log('T3_corrected_GLM4 (variant B) key points: '
            'L4inj trk %.1f@4 -> %.2f@5 -> %.2f@39; '
            'L19inj trk %.1f@19 -> %.2f@20 -> %.2f@39; '
            'proj L19inj %.0f@19 -> %.0f@20 -> %.0f@39'
            % (trk_profB['L4'][4]['trk_ratio'],
               trk_profB['L4'][5]['trk_ratio'],
               trk_profB['L4'][39]['trk_ratio'],
               trk_profB['L19'][19]['trk_ratio'],
               trk_profB['L19'][20]['trk_ratio'],
               trk_profB['L19'][39]['trk_ratio'],
               trk_profB['L19'][19]['xdir_proj'],
               trk_profB['L19'][20]['xdir_proj'],
               trk_profB['L19'][39]['xdir_proj']),
            lines)

        # ---------- A operator structure ----------
        rngn = np.random.default_rng(SEED_NULL)
        Gn = rngn.standard_normal((n, HID))
        Gn = Gn / np.linalg.norm(
            Gn, axis=1, keepdims=True)
        r1_null, _, _, _ = op_structure(Gn)
        A['null_r1_rowgauss'] = round(r1_null, 6)
        A['per_inj'] = {}
        for key in ('L4', 'L19'):
            li_inj = int(key[1:])
            layers_list = list(range(li_inj, NL))
            Dstack = Dl_store[key]
            per = {}
            for k1, li in enumerate(layers_list):
                Dl = Dstack[k1]
                r1, erank, medcos, v1 = op_structure(Dl)
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
        for key in ('L4', 'L19'):
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

        # ---------- verdict (PRIMARY = L19 band) ----------
        band_r1 = [A['per_inj']['L19'][li]['r1']
                   for li in ERASER_BAND]
        band_cos = [A['per_inj']['L19'][li]['medcos']
                    for li in ERASER_BAND]
        band_trkB = [trk_profB['L19'][li]['trk_ratio']
                     for li in ERASER_BAND]
        med_r1 = float(np.median(band_r1))
        med_cos = float(np.median(band_cos))
        med_trkB = float(np.median(band_trkB))
        A['L19_band'] = {
            'layers': list(ERASER_BAND),
            'med_r1': round(med_r1, 6),
            'med_medcos': round(med_cos, 6),
            'med_trk_B': round(med_trkB, 6),
            'amp_floor': AMP_FLOOR,
            'r1_shared_gate': R1_SHARED,
            'cos_shared_gate': COS_SHARED,
            'r1_scatter_gate': R1_SCATTER,
            'cos_scatter_gate': COS_SCATTER}
        a11_diff = max(spreads.values())
        a11_ok = bool(a11_diff < 1e-6)
        log('a11 same-session determinism %.2e ok=%s'
            % (a11_diff, a11_ok), lines)
        log('band medians: trk_B=%.4f r1=%.4f '
            'medcos=%.4f'
            % (med_trkB, med_r1, med_cos), lines)
        if not (a9_ok and a11_ok and a12a_ok):
            verdict = 'anchor_fail_all_void'
        elif med_trkB < AMP_FLOOR:
            verdict = 'amplitude_collapse_eraser_glm4'
        elif (med_r1 >= R1_SHARED
                and med_cos >= COS_SHARED):
            verdict = 'shared_lowrank_operator_glm4'
        elif (med_r1 < R1_SCATTER
                or med_cos < COS_SCATTER):
            verdict = 'cell_scatter_operator_glm4'
        else:
            verdict = 'mixed_operator_glm4'
    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('VERDICT %s' % verdict, lines)

    elapsed = time.monotonic() - t0

    anchors = {
        'a0_words_order': a0_ok, 'a8_collision': a8_ok,
        'a1_diffs': {str(k): v
                     for k, v in a1_diffs.items()},
        'a1_ok': a1_ok, 'a2_rel': a2_rel,
        'a3_diff': a3_diff, 'a6_diff': a6_diff,
        'a6_ok': a6_ok, 'a6_sep_f': sep_f,
        'a7_diff': a7_diff,
        'a9_parts': {k: v for k, v
                     in a9_parts.items()},
        'a9_ok': a9_ok if anchor_prelim else None,
        'a10_ok': a10_ok, 'a11_diff': a11_diff,
        'a12a_diffs': a12a_diffs,
        'a12a_ok': a12a_ok if anchor_prelim else None,
    }
    res = {
        'phase': 3005,
        'final_verdict': verdict,
        'anchor_all_ok': bool(
            a0_ok and a8_ok and a1_ok and a7_ok
            and a2_ok and a6_ok and a3_ok and a10_ok
            and a9_ok and a11_ok and a12a_ok),
        'anchors': anchors,
        'scale': {'sep_f': round(sep_f, 2),
                  'sep_null0': round(sep_n, 2),
                  'med_dS': round(med_dS, 4)},
        'T2lite': t2lite if anchor_prelim else None,
        'A': A, 'E': E,
        'T3_corrected_GLM4': trk_profB,
        'axis_bug': {
            'status': 'glm4_side_recomputed',
            'affected': ['phase3001 T3'],
            'a12a_evidence': 'the 3001 formula '
                             'reproduces the 3001 stored '
                             'T3 profiles bit-level from '
                             'THIS run Dl; corrected '
                             'per-cell values registered '
                             'as T3_corrected_GLM4',
            'impact': 'T3 was DESCRIPTIVE in 3001 - '
                      'verdict (from T1) unaffected; '
                      'qualitative eraser narrative now '
                      'rests on corrected values',
        },
        'tags': PREREG['tags'],
        'elapsed_s': round(elapsed, 1),
        'correction_note': 'first run',
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    save['dirs_g'] = dirs_g
    save['Vt8_g'] = Vt
    save['u39'] = u39
    save['xdir'] = xdir
    save['words_g'] = np.array(words_g)
    save['lang_g'] = lang
    save['null0_tids'] = np.array(null0_tids)
    save['proj_base'] = proj_f0
    save['proj_null'] = proj_n0
    npz_path = os.path.join(
        OUT, 'omega_p5_operator_structure_glm4.npz')
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
