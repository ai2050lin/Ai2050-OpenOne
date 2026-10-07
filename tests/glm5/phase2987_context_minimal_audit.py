# -*- coding: utf-8 -*-
"""Phase 2987: context-minimal audit (interleaved priority).

2986 found the L34 F/C word-class signature is len-2 specific
(p=.033 -> p~.7 under any context). Minimal ladder + filler
content controls to locate the collapse boundary, and survival
audit of three key chain cards under context:
  card A (2962): L34 readout F/C contrast
  card B (2963): L34 readout en/fr contrast
  card C (2964): L34/h15 per-head F/C carrier contrast

Conditions (frozen):
  L2   : [the, w]                     (2979 protocol verbatim)
  L3   : 1 NEUTRAL filler + tail
  L16N : 14 NEUTRAL filler (2986 sentence)
  L16R : 14 RAND filler (seeded perm of common-token pool)
  L16T : 14 x ' the'

Primary verdict = T1 ladder on card A:
  p(L2)>=.05                    -> signature_not_present
  p(L3)>=.01                    -> collapse_at_single_token
  p(L3)<.01 & p(L16N)>=.01      -> dose_gradual
  else                          -> persists_all
Secondary: T2 content control, T3 lang survival, T4 carrier
survival (+ head migration), T5 single-token restructure step.
"""
import json
import os
import time
import hashlib
import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2939 = os.path.join(BASE, 'phase2939', 'rotation_target',
                        'rotation_target.npz')
SRC_2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                        'probe_relativity.npz')
SRC_2973 = os.path.join(BASE, 'phase2973', 'fr_scale_audit',
                        'fr_scale_audit.npz')
EXEC_2977 = os.path.join(BASE, 'phase2977',
                         'two_axis_fusion_injection',
                         'execution.json')
SRC_2979 = os.path.join(BASE, 'phase2979', 'reversal_anatomy',
                        'reversal_anatomy.npz')
SRC_2986 = os.path.join(BASE, 'phase2986',
                        'length_context_drift',
                        'length_context_drift.npz')
OUT = os.path.join(BASE, 'phase2987', 'context_minimal_audit')

MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
L34 = 34
N_PERM = 10000
RNG_MAIN = 2987
P_PRE = 0.05     # L2 precondition
P_GATE = 0.01    # survival gate
T5_RATIO = 0.5
FILLER_NEUTRAL = (' The sun rises in the east and sets in '
                  'the west .')
POOL_TEXT = ('the of and to in a is that it for as with was '
             'on are by this be at from or an which one had '
             'not but what all were when we there can been')


def sha8(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]


def log(msg, lines):
    lines.append('[%s] %s' % (time.strftime('%H:%M:%S'), msg))
    with open(os.path.join(OUT, 'run_log.txt'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def median(a):
    return float(np.median(a))


def perm_p(v, labels, rng, n_perm=N_PERM):
    obs = median(v[labels == 1]) - median(v[labels == 0])
    cnt = 0
    for _ in range(n_perm):
        pl = rng.permutation(labels)
        pv = median(v[pl == 1]) - median(v[pl == 0])
        if abs(pv) >= abs(obs):
            cnt += 1
    return float(obs), float(cnt) / float(n_perm)


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)

    # ---------- cells ----------
    e77 = json.load(open(EXEC_2977, encoding='utf-8'))
    F_EN = e77['cells']['F_en']
    F_FR = e77['cells']['F_fr']
    C_EN = e77['cells']['C_en']
    C_FR = e77['cells']['C_fr']
    assert (len(F_EN), len(F_FR), len(C_EN), len(C_FR)) \
        == (15, 15, 22, 22), 'cell size drift'
    cells = [('F', 'en', w) for w in F_EN] \
        + [('F', 'fr', w) for w in F_FR] \
        + [('C', 'en', w) for w in C_EN] \
        + [('C', 'fr', w) for w in C_FR]
    words77 = ['%s:%s:%s' % c for c in cells]
    n_test = len(cells)
    lang = np.array([0 if c[1] == 'en' else 1 for c in cells])
    cls = np.array([0 if c[0] == 'F' else 1 for c in cells])

    rng_pool = np.random.default_rng(RNG_MAIN)
    prereg = {
        'design': 'conditions {L2:[the,w]; L3: 1 neutral '
                  'filler; L16N: 14 neutral (2986 sentence); '
                  'L16R: 14 seeded-permuted common-token pool '
                  '(rng 2987); L16T: 14x " the"}; target tail '
                  '[func_tid, word_tid]; per-condition '
                  'independent single-sample forwards',
        'cards': 'A=2962 L34 F/C contrast; B=2963 L34 en/fr '
                 'contrast; C=2964 L34/h15 per-head F/C '
                 'contrast (+ head migration census)',
        'T1_primary': 'ladder on card A with NEUTRAL filler: '
                      'precondition p(L2)<%.2f; branches: '
                      'p(L3)>=%.2f -> collapse_at_single_token; '
                      'p(L3)<%.2f and p(L16N)>=%.2f -> '
                      'dose_gradual; else persists_all'
                      % (P_PRE, P_GATE, P_GATE, P_GATE),
        'T2_content': 'card A at L16 {N,R,T}: all p>=%.2f -> '
                      'collapse_content_independent; else '
                      'collapse_content_dependent (list)'
                      % P_GATE,
        'T3_lang': 'card B survival: all conds p<%.2f -> '
                   'lang_effect_robust; only L2 -> '
                   'lang_effect_len2_only; else '
                   'lang_effect_mixed' % P_GATE,
        'T4_carrier': 'card C head-15 survival (same branch '
                      'style) + migration census: top heads '
                      'by |F/C contrast| with p<%.2f at L16N'
                      % P_GATE,
        'T5_step': 'D34(L3) >= %.1f * D34(L16N) -> '
                   'restructure_at_single_token; else '
                   'restructure_gradual' % T5_RATIO,
        'anchors': {
            'a1': 'Vt8 rebuild < 1e-6',
            'a2': 'determinism L2 and L16N rel < 1e-4',
            'a3': 'L2 norms/coss vs 2973: rel<1e-4, '
                  'cos<1e-6',
            'a4': 'single-token 74/74',
            'a5': 'L16N prof34+BL vs 2986 npz bit-level '
                  '< 1e-12',
            'a6': 'L2 d_lang_u/d_cls_u vs 2979 < 1e-9',
            'a7': 'L2 n17 vs 2979 rel < 1e-9',
            'a8': 'L2 B_rec vs 2973 rel < 1e-4',
            'a9': 'sum_h headC == prof34 identity < 1e-9'},
        'stats': 'two-sided label permutation n=%d, '
                 'family=5 conds x 3 cards raw p; T1 primary, '
                 'T2-T5 secondary' % N_PERM,
    }

    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2987,
                   'name': 'context_minimal_audit',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2939': sha8(SRC_2939),
                               's2927': sha8(SRC_2927),
                               's2973': sha8(SRC_2973),
                               's2977exec': sha8(EXEC_2977),
                               's2979npz': sha8(SRC_2979),
                               's2986npz': sha8(SRC_2986)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'l34': L34, 'n_perm': N_PERM,
                   'rng': RNG_MAIN, 'p_pre': P_PRE,
                   'p_gate': P_GATE, 't5_ratio': T5_RATIO,
                   'filler_neutral': FILLER_NEUTRAL,
                   'pool_text': POOL_TEXT,
                   'cells': {'F_en': F_EN, 'F_fr': F_FR,
                             'C_en': C_EN, 'C_fr': C_FR},
                   'prereg': prereg},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- sources ----------
    z39 = np.load(SRC_2939, allow_pickle=True)
    z27 = np.load(SRC_2927, allow_pickle=True)
    dirs27 = z27['dirs_word'].astype(np.float64)
    _, _, Vt_loc = np.linalg.svd(dirs27, full_matrices=False)
    a1_diff = float(np.abs(Vt_loc[:8] - z39['Vt8']).max())
    a1_ok = bool(a1_diff < 1e-6)
    log('a1 Vt8 rebuild diff %.2e ok=%s' % (a1_diff, a1_ok),
        lines)
    u35 = dirs27[NL - 1]
    z73 = np.load(SRC_2973, allow_pickle=True)
    norms73 = z73['norms'].astype(np.float64)
    coss73 = z73['coss'].astype(np.float64)
    B73 = z73['B_rec'].astype(np.float64)
    words73 = [str(w) for w in z73['words']]
    z79 = np.load(SRC_2979, allow_pickle=True)
    d79_l = z79['d_lang_u'].astype(np.float64)
    d79_c = z79['d_cls_u'].astype(np.float64)
    n17_79 = z79['n17'].astype(np.float64)
    z86 = np.load(SRC_2986, allow_pickle=True)
    prof34_86 = z86['prof34'].astype(np.float64)
    BL_86 = z86['BL'].astype(np.float64)
    lens86 = [int(v) for v in z86['lengths']]
    i16 = lens86.index(16)

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
    assert words77 == words73, 'word order drift'
    tid_map = {}
    n_single = 0
    for _, _, w in cells:
        ids = tok(' ' + w, add_special_tokens=False)[
            'input_ids']
        if len(ids) != 1:
            ids = tok(w, add_special_tokens=False)[
                'input_ids']
        if len(ids) == 1:
            n_single += 1
        tid_map[w] = int(ids[0]) if len(ids) == 1 else -1
    a4_ok = bool(n_single == n_test)
    log('a4 single-token %d/%d ok=%s'
        % (n_single, n_test, a4_ok), lines)
    ids_the = tok(' the', add_special_tokens=False)[
        'input_ids']
    assert len(ids_the) == 1
    func_tid = int(ids_the[0])

    pool_ids = []
    for wd in POOL_TEXT.split():
        ii = tok(' ' + wd, add_special_tokens=False)[
            'input_ids']
        if len(ii) == 1:
            pool_ids.append(int(ii[0]))
    assert len(pool_ids) >= 20, 'token pool too small'
    pool_perm = [int(pool_ids[j]) for j in rng_pool.permutation(
        len(pool_ids))]
    rand14 = [pool_perm[k % len(pool_perm)]
              for k in range(14)]
    the14 = [func_tid] * 14
    neutral_pool = tok(FILLER_NEUTRAL,
                       add_special_tokens=False)['input_ids']
    assert len(neutral_pool) >= 1
    n14 = [int(neutral_pool[k % len(neutral_pool)])
           for k in range(14)]
    one_n = [int(neutral_pool[0])]

    CONDS = [('L2', []), ('L3', one_n), ('L16N', n14),
             ('L16R', rand14), ('L16T', the14)]

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    log('model loaded (pool %d tok, rand14 head %s)'
        % (len(pool_ids), rand14[:4]), lines)

    cap_op = {li: [] for li in range(NL)}
    cap_x17 = []
    handles = []

    def hs_of(args, kwargs):
        if args and args[0] is not None:
            return args[0]
        return kwargs.get('hidden_states')

    def hook_x17(module, args, kwargs):
        x = hs_of(args, kwargs)
        if x is None or x.dim() < 2:
            return None
        cap_x17.append(
            x[:, -1, :].detach().float().cpu().numpy())
        return None

    def hook_op(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('input')
            if x is None or x.dim() < 2:
                return None
            cap_op[li].append(
                x[:, -1, :].detach().float().cpu().numpy())
            return None
        return h

    handles.append(
        layers[17].self_attn
        .register_forward_pre_hook(
            hook_x17, with_kwargs=True))
    for li in range(NL):
        handles.append(
            layers[li].self_attn.o_proj
            .register_forward_pre_hook(
                hook_op(li), with_kwargs=True))

    def clear_cap():
        for li in cap_op:
            del cap_op[li][:]
        del cap_x17[:]

    def forward1(toks):
        clear_cap()
        with torch.no_grad():
            model(torch.tensor([toks], device='cuda'))
        return {li: cap_op[li][0].astype(np.float64)
                for li in range(NL)}

    M = np.zeros((NL, NH * HD))
    Mnorm = np.zeros(NL)
    for li in range(NL):
        Wo = layers[li].self_attn.o_proj.weight.detach() \
            .float().cpu().numpy()
        M[li] = u35 @ Wo
        Mnorm[li] = float(np.linalg.norm(M[li]))
    Wo34 = layers[L34].self_attn.o_proj.weight.detach() \
        .float().cpu().numpy().astype(np.float64)
    C34 = u35 @ Wo34  # (4096,) per-head readout row

    n_cond = len(CONDS)
    prof_all = np.zeros((n_cond, n_test, NL))
    normsL = np.zeros((n_cond, n_test, NL))
    cossL = np.zeros((n_cond, n_test, NL))
    BL = np.zeros((n_cond, n_test))
    n17L = np.zeros((n_cond, n_test))
    headC = np.zeros((n_cond, n_test, NH))
    for ci, (cname, fill) in enumerate(CONDS):
        for i, (_, _, w) in enumerate(cells):
            seq = list(fill) + [func_tid, tid_map[w]]
            assert len(seq) == len(fill) + 2
            op = forward1(seq)
            for li in range(NL):
                x = op[li].reshape(-1)
                normsL[ci, i, li] = \
                    float(np.linalg.norm(x))
                pr = float(np.dot(x, M[li]))
                prof_all[ci, i, li] = pr
                cossL[ci, i, li] = pr / max(
                    normsL[ci, i, li] * Mnorm[li], 1e-30)
            BL[ci, i] = (float(prof_all[ci, i, 6:13].mean())
                         - float(prof_all[ci, i,
                                          28:36].mean()))
            n17L[ci, i] = float(np.linalg.norm(
                cap_x17[0].astype(np.float64).reshape(-1)))
            x34 = op[L34].reshape(-1)
            for h in range(NH):
                headC[ci, i, h] = float(np.dot(
                    C34[h * HD:(h + 1) * HD],
                    x34[h * HD:(h + 1) * HD]))
        log('cond %s done' % cname, lines)

    # ---------- anchors ----------
    seq2 = [func_tid, tid_map['man']]
    op_a = forward1(seq2)
    op_b = forward1(seq2)
    a2a_rel = float(np.abs(op_a[30] - op_b[30]).max()
                    / max(float(np.abs(op_a[30]).max()),
                          1e-30))
    seq16 = list(n14) + [func_tid, tid_map['man']]
    op_c = forward1(seq16)
    op_d = forward1(seq16)
    a2b_rel = float(np.abs(op_c[30] - op_d[30]).max()
                    / max(float(np.abs(op_c[30]).max()),
                          1e-30))
    a2_ok = bool(a2a_rel < 1e-4 and a2b_rel < 1e-4)
    log('a2 determinism L2 %.2e L16N %.2e ok=%s'
        % (a2a_rel, a2b_rel, a2_ok), lines)

    nrm_rel = float(np.max(np.abs(normsL[0] - norms73)
                           / np.maximum(norms73, 1e-30)))
    cos_abs = float(np.max(np.abs(cossL[0] - coss73)))
    a3_ok = bool(nrm_rel < 1e-4 and cos_abs < 1e-6)
    log('a3 identity vs 2973 (L2): %.2e / %.2e ok=%s'
        % (nrm_rel, cos_abs, a3_ok), lines)

    a5a = float(np.abs(prof_all[2, :, L34]
                       - prof34_86[i16]).max())
    a5b = float(np.abs(BL[2] - BL_86[i16]).max())
    a5_ok = bool(a5a < 1e-12 and a5b < 1e-12)
    log('a5 L16N vs 2986 npz: prof34 %.2e BL %.2e ok=%s'
        % (a5a, a5b, a5_ok), lines)

    X17_2 = np.zeros((n_test, 2560))
    for i, (_, _, w) in enumerate(cells):
        forward1([func_tid, tid_map[w]])
        X17_2[i] = cap_x17[0].astype(
            np.float64).reshape(-1)
    d_lang2 = X17_2[lang == 1].mean(axis=0) \
        - X17_2[lang == 0].mean(axis=0)
    d_cls2 = X17_2[cls == 1].mean(axis=0) \
        - X17_2[cls == 0].mean(axis=0)
    d_lang_u2 = d_lang2 / float(np.linalg.norm(d_lang2))
    d_cls_u2 = d_cls2 / float(np.linalg.norm(d_cls2))
    a6_diff = float(max(np.abs(d_lang_u2 - d79_l).max(),
                        np.abs(d_cls_u2 - d79_c).max()))
    a6_ok = bool(a6_diff < 1e-9)
    n17_rel = float(np.max(np.abs(n17L[0] - n17_79)
                           / np.maximum(n17_79, 1e-30)))
    a7_ok = bool(n17_rel < 1e-9)
    log('a6 axis dirs %.2e ok=%s; a7 n17 %.2e ok=%s'
        % (a6_diff, a6_ok, n17_rel, a7_ok), lines)

    a8_rel = float(np.max(np.abs(BL[0] - B73)
                          / np.maximum(np.abs(B73), 1e-30)))
    a8_ok = bool(a8_rel < 1e-4)
    log('a8 B_rec vs 2973: %.2e ok=%s'
        % (a8_rel, a8_ok), lines)

    head_sum = headC.sum(axis=2)
    a9_diff = float(np.abs(head_sum
                           - prof_all[:, :, L34]).max())
    a9_ok = bool(a9_diff < 1e-9)
    log('a9 head-sum identity: %.2e ok=%s'
        % (a9_diff, a9_ok), lines)

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok and a6_ok and a7_ok and a8_ok
                     and a9_ok)

    # ---------- statistics ----------
    rng = np.random.default_rng(RNG_MAIN + 1)
    prof34 = prof_all[:, :, L34]
    sigA = np.zeros(n_cond)
    pA = np.zeros(n_cond)
    sigB = np.zeros(n_cond)
    pB = np.zeros(n_cond)
    h15c = np.zeros(n_cond)
    p_h15 = np.zeros(n_cond)
    headC_sig = {}
    for ci in range(n_cond):
        v = prof34[ci]
        sigA[ci], pA[ci] = perm_p(v, cls, rng)
        sigB[ci], pB[ci] = perm_p(v, lang, rng)
        h15c[ci], p_h15[ci] = perm_p(
            headC[ci, :, 15], cls, rng)
        if ci == 2:
            ps = np.zeros(NH)
            cs = np.zeros(NH)
            for h in range(NH):
                cs[h], ps[h] = perm_p(
                    headC[ci, :, h], cls, rng)
            headC_sig['L16N'] = {
                str(h): {'contrast': round(float(cs[h]), 4),
                         'p': round(float(ps[h]), 4)}
                for h in np.argsort(-np.abs(cs))[:5]}
    log('sigA %s p %s' % ([round(float(v), 4) for v in sigA],
                          [round(float(v), 4) for v in pA]),
        lines)
    log('sigB %s p %s' % ([round(float(v), 4) for v in sigB],
                          [round(float(v), 4) for v in pB]),
        lines)
    log('h15c %s p %s'
        % ([round(float(v), 4) for v in h15c],
           [round(float(v), 4) for v in p_h15]), lines)

    d34 = np.array([float(np.median(np.abs(
        prof34[ci] - prof34[0]))) for ci in range(n_cond)])
    log('D34 per cond %s'
        % ([round(float(v), 4) for v in d34],), lines)

    # ---------- verdicts ----------
    verdict = None
    T1 = T2 = T3 = T4 = T5 = None
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
        log('ANCHOR FAIL -> all void', lines)
    else:
        if pA[0] >= P_PRE:
            verdict = 'signature_not_present'
            T1 = {'p': [round(float(v), 4) for v in pA],
                  'note': 'precondition failed'}
        elif pA[1] >= P_GATE:
            verdict = 'collapse_at_single_token'
        elif pA[2] >= P_GATE:
            verdict = 'dose_gradual'
        else:
            verdict = 'persists_all'
        T1 = {'p': [round(float(v), 4) for v in pA],
              'sig': [round(float(v), 4) for v in sigA],
              'verdict': verdict}
        log('T1 primary: %s' % verdict, lines)

        p16 = [pA[2], pA[3], pA[4]]
        if all(p >= P_GATE for p in p16):
            t2v = 'collapse_content_independent'
        else:
            t2v = 'collapse_content_dependent'
        T2 = {'verdict': t2v,
              'p_L16NRT': [round(float(p), 4)
                           for p in p16],
              'sig_L16NRT': [round(float(v), 4)
                             for v in sigA[2:5]],
              'sig_diff_NvsR': round(
                  float(abs(sigA[2] - sigA[3])), 4),
              'sig_diff_NvsT': round(
                  float(abs(sigA[2] - sigA[4])), 4)}
        log('T2 content: %s' % json.dumps(T2), lines)

        if all(p < P_GATE for p in pB):
            t3v = 'lang_effect_robust'
        elif pB[0] < P_GATE and all(
                p >= P_GATE for p in pB[1:]):
            t3v = 'lang_effect_len2_only'
        else:
            t3v = 'lang_effect_mixed'
        T3 = {'verdict': t3v,
              'p': [round(float(v), 4) for v in pB],
              'sig': [round(float(v), 4) for v in sigB]}
        log('T3 lang: %s' % json.dumps(T3), lines)

        if all(p < P_GATE for p in p_h15):
            t4v = 'carrier_h15_robust'
        elif p_h15[0] < P_GATE and all(
                p >= P_GATE for p in p_h15[1:]):
            t4v = 'carrier_h15_len2_only'
        else:
            t4v = 'carrier_h15_mixed'
        T4 = {'verdict': t4v,
              'p': [round(float(v), 4) for v in p_h15],
              'contrast': [round(float(v), 4)
                           for v in h15c],
              'L16N_top5': headC_sig.get('L16N')}
        log('T4 carrier: %s' % json.dumps(T4), lines)

        t5_ok = bool(d34[1] >= T5_RATIO * d34[2])
        T5 = {'D34': [round(float(v), 4) for v in d34],
              'ratio_L3_over_L16N': round(
                  float(d34[1] / max(d34[2], 1e-30)), 4),
              'verdict': ('restructure_at_single_token'
                          if t5_ok
                          else 'restructure_gradual')}
        log('T5 step: %s' % json.dumps(T5), lines)

    elapsed = round(time.monotonic() - t0, 1)
    result = {
        'phase': 2987, 'name': 'context_minimal_audit',
        'final_verdict': verdict,
        'anchor_all_ok': anchor_ok,
        'anchors': {
            'a1_Vt8_diff': a1_diff, 'a1_ok': a1_ok,
            'a2a_rel': a2a_rel, 'a2b_rel': a2b_rel,
            'a2_ok': a2_ok,
            'a3_norm_rel': nrm_rel, 'a3_cos_abs': cos_abs,
            'a3_ok': a3_ok, 'a4_ok': a4_ok,
            'a5_prof34_diff': a5a, 'a5_BL_diff': a5b,
            'a5_ok': a5_ok, 'a6_diff': a6_diff,
            'a6_ok': a6_ok, 'a7_n17_rel': n17_rel,
            'a7_ok': a7_ok, 'a8_B_rel': a8_rel,
            'a8_ok': a8_ok, 'a9_headsum_diff': a9_diff,
            'a9_ok': a9_ok},
        'T1': T1, 'T2': T2, 'T3': T3, 'T4': T4, 'T5': T5,
        'conds': [c[0] for c in CONDS],
        'B_median': [round(float(np.median(BL[ci])), 4)
                     for ci in range(n_cond)],
        'n17_median': [round(float(np.median(n17L[ci])), 4)
                       for ci in range(n_cond)],
        'elapsed_s': elapsed}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    np.savez(os.path.join(
        OUT, 'context_minimal_audit.npz'),
        prof34=prof34, prof_all=prof_all, norms=normsL,
        coss=cossL, BL=BL, n17L=n17L, headC=headC,
        sigA=sigA, pA=pA, sigB=sigB, pB=pB,
        h15c=h15c, p_h15=p_h15, d34=d34,
        words=np.array(words77))
    log('saved npz+result.json elapsed %.1fs' % elapsed,
        lines)
    print('PHASE2987 DONE verdict=%s elapsed=%s'
          % (verdict, elapsed))


if __name__ == '__main__':
    main()
