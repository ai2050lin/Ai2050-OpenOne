# -*- coding: utf-8 -*-
"""Phase 2975: functional meaning of the early-layer
handshake subspace - aligned-channel ablation (preregistered).

Why: 2974 found the attention-write/MLP-read subspace
alignment (handshake) significant ONLY at early layers
L1-3 (S 0.76-0.83 beyond both nulls), absent at deep
carrier layers. Open question: is the aligned early
subspace functionally load-bearing, or a geometric
epiphenomenon?

Design (frozen before any observation):
  Words: F_en(15) + C_en(22) from 2972 execution.json
  (identity gate), English only (2972 language binding
  lesson). Protocol 2964 verbatim: single forward
  [the, w], sep = x_36 . u35, B = band profile readout.
  Ablation: o_proj FORWARD-hook output modification,
  c := c - QA(QA^T c) where QA = top-64 left singular
  vectors of Wo[li] (the aligned write subspace, 2974
  machinery f64), applied at layers {1,2,3} jointly, all
  positions. Conditions per word:
    base / abl_L123_align / abl_L123_rand / abl_L20_align
  (rand = matched orthonormal random basis per layer,
  rngs 2975-2977; L20 = layer control at a
  no-handshake layer).

Tests (frozen):
  T1 readout: per-word delta = |dsep_align| - |dsep_rand|
     (dsep = sep_cond - sep_base); sign-flip permutation
     rng 2978 x10000, one-sided p for median > 0,
     gate p <= 0.01.
  T2 band signature: same on |dB|, rng 2979.
  T3 descriptive: L20 control effect vs L123 align;
     fraction of words delta > 0; effect ratios.

Anchors (frozen):
  a1 Vt8 rebuild from 2927 vs 2939 npz < 1e-6
  a2 determinism < 1e-4
  a3 handshake identity: recomputed S(li,li) L1-3 vs
     2974 result.json stored (4dp) |diff| < 1e-4
  a4 single-token 37/37 + anchors 3/3
  a5 B of 3 anchor words vs 2963 npz < 1e-4 rel
  a6 ablation sanity: QA/rand orthonormality < 1e-8;
     after ablation |QA^T c| < 1e-5 rel; ablated fin
     differs from base (max rel > 1e-8)
  a7 word-set identity vs 2972 execution.json

Verdict (frozen):
  anchor fail => anchor_fail_all_void
  T1 & T2 gate met => handshake_subspace_functionally_
     load_bearing
  T1 only => handshake_subspace_readout_load_bearing
  T2 only => handshake_subspace_band_load_bearing
  else => handshake_subspace_not_load_bearing
"""
import hashlib
import io
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                        'probe_relativity.npz')
SRC_2939 = os.path.join(BASE, 'phase2939', 'rotation_target',
                        'rotation_target.npz')
SRC_2963 = os.path.join(BASE, 'phase2963',
                        'frequency_controlled_band',
                        'freq_band.npz')
EXEC_2972 = os.path.join(BASE, 'phase2972',
                         'two_factor_signature',
                         'execution.json')
RES_2974 = os.path.join(BASE, 'phase2974',
                        'cross_module_alignment',
                        'result.json')
OUT = os.path.join(BASE, 'phase2975',
                   'handshake_functional_ablation')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2975_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
K = 64
N_PERM = 10000
ABL_LAYERS = [1, 2, 3]
CTRL_LAYER = 20
RNG_RAND = [2975, 2976, 2977]
RNG_T1 = 2978
RNG_T2 = 2979
ANCHOR_WORDS = ['people', 'for', 'garden']

PREREG = {
    'mode': 'single forwards [the, w] 2964 protocol '
            'verbatim; o_proj forward-hook channel '
            'ablation c := c - QA(QA^T c) at layers '
            '{1,2,3} (align basis = top-64 left sing vec '
            'of Wo, 2974 machinery; matched random basis '
            'control; L20 layer control); readouts sep '
            '= x_36.u35 and B band profile',
    'question': 'is the early-layer handshake subspace '
                '(2974 L1-3 alignment) functionally '
                'load-bearing for the u35 readout and the '
                'band signature, beyond a matched random '
                'channel ablation?',
    'words_source': '2972 execution.json F_en + C_en '
                    '(identity gate, English only)',
    'anchors': {
        'a1': 'Vt8 rebuild vs 2939 npz < 1e-6',
        'a2': 'determinism < 1e-4',
        'a3': 'S(li,li) L1-3 vs 2974 stored 4dp < 1e-4',
        'a4': 'single-token 37/37 + anchors 3/3',
        'a5': 'B of 3 anchors vs 2963 < 1e-4 rel',
        'a6': 'orthonormality < 1e-8; |QA^T c_mod| < '
              '1e-5 rel; ablated fin differs from base',
        'a7': 'word-set identity vs 2972',
    },
    'T1': 'paired |dsep_align| vs |dsep_rand|, sign-flip '
          'perm rng 2978 x10000, one-sided median>0, '
          'gate p<=0.01',
    'T2': 'paired |dB_align| vs |dB_rand|, rng 2979',
    'T3': 'descriptive: L20 control vs L123 align; '
          'fraction delta>0; ratios',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'T1&T2 => '
               'handshake_subspace_functionally_load_'
               'bearing; T1 only => '
               'handshake_subspace_readout_load_bearing; '
               'T2 only => '
               'handshake_subspace_band_load_bearing; '
               'else => handshake_subspace_not_load_'
               'bearing',
    'correction_note': '',
}


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def log(msg, lines):
    lines.append(msg)
    print(msg, flush=True)


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)

    # ---------- sources ----------
    z27 = np.load(SRC_2927, allow_pickle=True)
    dirs27 = z27['dirs_word'].astype(np.float64)
    z39 = np.load(SRC_2939, allow_pickle=True)
    _, _, Vt_loc = np.linalg.svd(dirs27,
                                 full_matrices=False)
    a1_diff = float(np.abs(Vt_loc[:8]
                           - z39['Vt8']).max())
    a1_ok = bool(a1_diff < 1e-6)
    log('a1 Vt8 rebuild diff %.2e ok=%s'
        % (a1_diff, a1_ok), lines)
    u35 = dirs27[NL - 1]
    z63 = np.load(SRC_2963, allow_pickle=True)
    w63 = [str(w).split(':') for w in z63['words']]
    B63 = z63['B'].astype(np.float64)
    r74 = json.load(io.open(RES_2974, encoding='utf-8'))
    diag74 = dict((int(li), v)
                  for li, v in r74['T1']['top5_diag'])

    # ---------- words (a7 identity) ----------
    e72 = json.load(io.open(EXEC_2972, encoding='utf-8'))
    F_EN = e72['cells']['F_en']
    C_EN = e72['cells']['C_en']
    words = list(F_EN) + list(C_EN)
    a7_ok = bool(len(F_EN) == 15 and len(C_EN) == 22
                 and words == list(F_EN) + list(C_EN))
    log('a7 word-set identity ok=%s (37 words)'
        % a7_ok, lines)

    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2975,
                   'name': 'handshake_functional_ablation',
                   'created': time.strftime(
                       '%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2927': sha8(SRC_2927),
                               's2939': sha8(SRC_2939),
                               's2963': sha8(SRC_2963),
                               's2972exec':
                                   sha8(EXEC_2972),
                               's2974': sha8(RES_2974)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'k': K, 'n_perm': N_PERM,
                   'rng': {'rand': RNG_RAND,
                           'T1': RNG_T1, 'T2': RNG_T2},
                   'abl_layers': ABL_LAYERS,
                   'ctrl_layer': CTRL_LAYER,
                   'words': words,
                   'anchors_words': ANCHOR_WORDS,
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen (37 words)', lines)

    # ---------- model ----------
    import sys
    sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
    from phase2662_symmetric_mapping_contract import \
        load_native
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        MD, local_files_only=True, trust_remote_code=True,
        use_fast=True)
    tid_map = {}
    n_single = 0
    for w in words + ANCHOR_WORDS:
        ids = tok(' ' + w, add_special_tokens=False)[
            'input_ids']
        if len(ids) != 1:
            ids = tok(w, add_special_tokens=False)[
                'input_ids']
        if len(ids) == 1:
            n_single += 1
        tid_map[w] = int(ids[0]) if len(ids) == 1 else -1
    a4_ok = bool(n_single == len(words) + 3)
    log('a4 single-token %d/%d ok=%s'
        % (n_single, len(words) + 3, a4_ok), lines)
    ids_the = tok(' the', add_special_tokens=False)[
        'input_ids']
    assert len(ids_the) == 1
    func_tid = int(ids_the[0])

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    log('model loaded', lines)

    # ---------- weights ----------
    Wo32 = {}
    Wo64 = {}
    for li in range(NL):
        W = layers[li].self_attn.o_proj.weight.detach() \
            .float().cpu().numpy()
        Wo32[li] = W
        Wo64[li] = W.astype(np.float64)
    M = np.zeros((NL, NH * HD))
    for li in range(NL):
        M[li] = u35 @ Wo32[li]

    # ---------- bases ----------
    def basis_align(li):
        U, _, _ = np.linalg.svd(Wo64[li],
                                full_matrices=False)
        return U[:, :K]

    def basis_rand(seed):
        rng = np.random.default_rng(seed)
        G = rng.standard_normal((2560, K))
        Q, _ = np.linalg.qr(G)
        return Q

    QA = {li: basis_align(li) for li in ABL_LAYERS}
    QA[CTRL_LAYER] = basis_align(CTRL_LAYER)
    QR = {li: basis_rand(RNG_RAND[i])
          for i, li in enumerate(ABL_LAYERS)}
    orth = [0.0]

    def chk(Q):
        orth[0] = max(orth[0], float(np.abs(
            Q.T @ Q - np.eye(K)).max()))

    for li in ABL_LAYERS:
        chk(QA[li])
        chk(QR[li])
    chk(QA[CTRL_LAYER])
    a6_orth_ok = bool(orth[0] < 1e-8)

    # ---------- a3 handshake identity ----------
    a3_diff = 0.0
    for li in ABL_LAYERS:
        _, s_up, Vt = np.linalg.svd(
            layers[li].mlp.up_proj.weight.detach()
            .float().cpu().numpy().astype(np.float64),
            full_matrices=False)
        Qm = Vt[:K].T
        s = np.linalg.svd(QA[li].T @ Qm,
                          compute_uv=False)
        S_rec = float(s[:16].mean())
        a3_diff = max(a3_diff, abs(S_rec - diag74[li]))
    a3_ok = bool(a3_diff < 1e-4)
    log('a3 handshake identity max diff %.2e ok=%s'
        % (a3_diff, a3_ok), lines)

    # ---------- hooks ----------
    cap_op = {li: [] for li in range(NL)}
    fin_cap = {}
    state_fin = {'on': False}
    abl_state = {'active': False,
                 'layers': {},
                 'proj_resid_max': 0.0}
    handles = []

    def hook_op_pre(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('input')
            if x is None or x.dim() < 2:
                return None
            cap_op[li].append(
                x[:, 1, :].detach().float().cpu().numpy())
            return None
        return h

    def make_post(li):
        def hook_op_post(module, args, kwargs,
                         output):
            if not abl_state['active']:
                return None
            Q = abl_state['layers'].get(li)
            if Q is None:
                return None
            orig_shape = output.shape
            c = output.reshape(-1,
                               output.shape[-1]) \
                .float()
            Qt = torch.tensor(Q.T,
                              device=c.device,
                              dtype=c.dtype)
            proj = c @ Qt.T
            c_mod = c - proj @ Qt
            with torch.no_grad():
                resid = float(
                    (Qt @ c_mod.T).abs().max())
                ref = float(c.abs().max())
            abl_state['proj_resid_max'] = max(
                abl_state['proj_resid_max'],
                resid / max(ref, 1e-30))
            return c_mod.reshape(orig_shape) \
                .to(output.dtype)
        return hook_op_post

    def pre_norm(module, args, kwargs):
        if state_fin['on']:
            fin_cap['x'] = args[0][:, -1, :].detach() \
                .float().cpu().numpy()

    for li in range(NL):
        handles.append(
            layers[li].self_attn.o_proj
            .register_forward_pre_hook(
                hook_op_pre(li), with_kwargs=True))
        if li in ABL_LAYERS or li == CTRL_LAYER:
            handles.append(
                layers[li].self_attn.o_proj
                .register_forward_hook(
                    make_post(li),
                    with_kwargs=True))
    handles.append(model.model.norm
                   .register_forward_pre_hook(
                       pre_norm, with_kwargs=True))

    def clear_cap():
        for li in cap_op:
            del cap_op[li][:]

    def forward1(toks, ablate=None):
        """ablate: None | 'align' | 'rand' | 'L20align'"""
        clear_cap()
        fin_cap.pop('x', None)
        abl_state['active'] = False
        abl_state['layers'] = {}
        if ablate is not None:
            if ablate == 'align':
                ls = {li: QA[li] for li in ABL_LAYERS}
            elif ablate == 'rand':
                ls = {li: QR[li] for li in ABL_LAYERS}
            elif ablate == 'L20align':
                ls = {CTRL_LAYER: QA[CTRL_LAYER]}
            else:
                raise ValueError(ablate)
            abl_state['layers'] = dict(ls)
            abl_state['active'] = True
        state_fin['on'] = True
        with torch.no_grad():
            model(torch.tensor([toks], device='cuda'))
        state_fin['on'] = False
        abl_state['active'] = False
        return (fin_cap['x'].astype(np.float64),
                {li: cap_op[li][0].astype(np.float64)
                 for li in range(NL)})

    def contributions(op):
        prof = np.zeros(NL)
        for li in range(NL):
            x = op[li].reshape(-1)
            prof[li] = float(np.dot(x, M[li]))
        return prof

    def band_of(prof):
        return (float(prof[6:13].mean())
                - float(prof[28:36].mean()))

    # ---------- a2 determinism ----------
    fin_a, _ = forward1([func_tid, tid_map['people']])
    fin_b, _ = forward1([func_tid, tid_map['people']])
    a2_rel = float(np.abs(fin_a - fin_b).max()
                   / max(float(np.abs(fin_a).max()),
                         1e-30))
    a2_ok = bool(a2_rel < 1e-4)
    log('a2 determinism rel %.2e ok=%s'
        % (a2_rel, a2_ok), lines)

    # ---------- a5 anchors ----------
    a5_rel = 0.0
    for w in ANCHOR_WORDS:
        fin, op = forward1([func_tid, tid_map[w]])
        prof = contributions(op)
        i63 = [i for i, (g, ww) in enumerate(w63)
               if ww == w][0]
        a5_rel = max(a5_rel,
                     abs(band_of(prof) - B63[i63])
                     / max(abs(B63[i63]), 1e-30))
    a5_ok = bool(a5_rel < 1e-4)
    log('a5 B vs 2963 rel %.2e ok=%s'
        % (a5_rel, a5_ok), lines)

    # ---------- a6 ablation sanity ----------
    fin_b0, _ = forward1([func_tid, tid_map['people']])
    fin_b1, _ = forward1([func_tid, tid_map['people']],
                         ablate='align')
    abl_rel = float(np.abs(fin_b1 - fin_b0).max()
                    / max(float(np.abs(fin_b0).max()),
                          1e-30))
    a6_ok = bool(a6_orth_ok
                 and abl_state['proj_resid_max'] < 1e-5
                 and abl_rel > 1e-8)
    log('a6 orth %.2e | proj resid %.2e | abl-vs-base '
        'rel %.2e ok=%s'
        % (orth[0], abl_state['proj_resid_max'],
           abl_rel, a6_ok), lines)

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok and a6_ok and a7_ok)
    verdict = None
    t1 = t2 = t3 = None
    save = {'words': np.array(words)}

    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        n_w = len(words)
        sep = {c: np.zeros(n_w) for c in
               ('base', 'align', 'rand', 'L20align')}
        Bee = {c: np.zeros(n_w) for c in
               ('base', 'align', 'rand', 'L20align')}
        conds = [('base', None), ('align', 'align'),
                 ('rand', 'rand'),
                 ('L20align', 'L20align')]
        for i, w in enumerate(words):
            toks = [func_tid, tid_map[w]]
            for cname, abl in conds:
                fin, op = forward1(toks, ablate=abl)
                sep[cname][i] = float(
                    fin.reshape(-1) @ u35)
                Bee[cname][i] = band_of(
                    contributions(op))
            if (i + 1) % 10 == 0:
                log('sweep [%d/%d]' % (i + 1, n_w), lines)

        def paired_test(dabs_a, dabs_r, seed):
            delta = dabs_a - dabs_r
            med = float(np.median(delta))
            frac = float((delta > 0).mean())
            rng = np.random.default_rng(seed)
            cnt = 0
            for _ in range(N_PERM):
                s = rng.choice([-1.0, 1.0], len(delta))
                if float(np.median(delta * s)) \
                        >= med - 1e-12:
                    cnt += 1
            p = (cnt + 1) / (N_PERM + 1)
            return delta, med, frac, p

        dsep_a = np.abs(sep['align'] - sep['base'])
        dsep_r = np.abs(sep['rand'] - sep['base'])
        d1, med1, frac1, p1 = paired_test(
            dsep_a, dsep_r, RNG_T1)
        dB_a = np.abs(Bee['align'] - Bee['base'])
        dB_r = np.abs(Bee['rand'] - Bee['base'])
        d2, med2, frac2, p2 = paired_test(
            dB_a, dB_r, RNG_T2)
        t1 = {'median_delta': round(med1, 5),
              'frac_gt0': round(frac1, 4),
              'p': float('%.3e' % p1),
              'med_dsep_align': round(float(
                  np.median(dsep_a)), 5),
              'med_dsep_rand': round(float(
                  np.median(dsep_r)), 5)}
        t2 = {'median_delta': round(med2, 5),
              'frac_gt0': round(frac2, 4),
              'p': float('%.3e' % p2),
              'med_dB_align': round(float(
                  np.median(dB_a)), 5),
              'med_dB_rand': round(float(
                  np.median(dB_r)), 5)}
        log('T1 align %.5f vs rand %.5f | delta %.5f '
            'frac %.2f p %.3e'
            % (t1['med_dsep_align'],
               t1['med_dsep_rand'], med1, frac1, p1),
            lines)
        log('T2 align %.5f vs rand %.5f | delta %.5f '
            'frac %.2f p %.3e'
            % (t2['med_dB_align'],
               t2['med_dB_rand'], med2, frac2, p2),
            lines)

        dL20 = np.abs(sep['L20align'] - sep['base'])
        t3 = {'med_dsep_L20': round(float(
                  np.median(dL20)), 5),
              'ratio_L123_vs_L20': round(float(
                  np.median(dsep_a)
                  / max(float(np.median(dL20)),
                        1e-30)), 4),
              'med_dsep_L20_to_rand': round(float(
                  np.median(dL20)
                  / max(float(np.median(dsep_r)),
                        1e-30)), 4)}
        log('T3 L20 control med dsep %.5f | L123/L20 '
            'ratio %.4f' % (t3['med_dsep_L20'],
                            t3['ratio_L123_vs_L20']),
            lines)

        save.update({'sep_base': sep['base'],
                     'sep_align': sep['align'],
                     'sep_rand': sep['rand'],
                     'sep_L20align': sep['L20align'],
                     'B_base': Bee['base'],
                     'B_align': Bee['align'],
                     'B_rand': Bee['rand'],
                     'B_L20align': Bee['L20align']})

        # verdict (frozen map)
        g1 = p1 <= 0.01
        g2 = p2 <= 0.01
        if g1 and g2:
            verdict = ('handshake_subspace_functionally'
                       '_load_bearing')
        elif g1:
            verdict = ('handshake_subspace_readout'
                       '_load_bearing')
        elif g2:
            verdict = ('handshake_subspace_band'
                       '_load_bearing')
        else:
            verdict = ('handshake_subspace_not_load'
                       '_bearing')
    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2975, 'model': 'qwen3-4b',
           'prereg': PREREG,
           'anchors': {'a1_diff': float('%.3e' % a1_diff),
                       'a1_ok': a1_ok,
                       'a2_rel': float('%.3e' % a2_rel),
                       'a2_ok': a2_ok,
                       'a3_diff': float('%.3e' % a3_diff),
                       'a3_ok': a3_ok,
                       'a4_ok': a4_ok,
                       'a5_rel': float('%.3e' % a5_rel),
                       'a5_ok': a5_ok,
                       'a6_ok': a6_ok,
                       'a7_ok': a7_ok,
                       'ok': anchor_ok},
           'T1': t1, 'T2': t2, 'T3': t3,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if verdict != 'anchor_fail_all_void':
        np.savez_compressed(os.path.join(
            OUT, 'handshake_functional_ablation.npz'),
            **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2975 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    import torch  # noqa: E402
    main()
