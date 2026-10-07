# -*- coding: utf-8 -*-
"""Phase 2972: language x word-class two-factor signature matrix
(preregistered). Plan-v2 stage-2 closing test.

Why: 2963/2964 established the function-vs-content load-band
class effect (Freedman-Lane p 2e-4; carrier L34/h15) on
ENGLISH-only lists, and 2969/2970 established a paired
language effect on transient peak timing. Open question: on
a MERGED bilingual list, are the language axis and the word-
class axis separable main effects on the static band
signature B, or do they interact (language changes the class
effect)?

Design (frozen before any observation):
  2x2 cells, 74 test words + 3 anchors:
    F-en 15 (2964 FUNC list verbatim: closed-class)
    F-fr 15 (frozen French closed-class list)
    C-en 22 (2887 en words, 22 concepts, sorted by en tid)
    C-fr 22 (2887 L(=fr) words, same concept order)
  Protocol 2937 pass1 verbatim: single forward [the, w],
  capture o_proj input pos1 all 36 layers ->
  C[li,w,h] per-head contribution to u35 readout (2964 spec).
  Covariate: rank(tid) WITHIN language (cross-language tid
  is not a common frequency scale), standardized.

Tests (frozen):
  T1 two-way Freedman-Lane on B: reduced B ~ 1 + cov;
     residuals permuted (rng 2971, 10000, joint draws);
     full B ~ 1 + cov + lang + class + lang:class;
     two-sided p for lang / class / interaction, gate
     p <= 0.01 each.
  T2 layer-level 2x2: four contrasts per layer
     (class gap within en, class gap within fr, language
     gap within F, language gap within C) + interaction
     (classgap_en - classgap_fr); per-contrast maxT family
     36 (rngs 2972-2975, 10000); report p <= 0.01 layers.
  T3 head-level: class gap (en) heads at the top
     class-gap layer, and language gap (F) heads at the top
     F-language-gap layer; maxT family 32 (rngs 2976/2977).
  T4 descriptive (no gate): spearman(class-gap profile en,
     2964 gap_layers); pooled vs within-language class gap
     (Simpson check); within-concept paired language
     difference on B for the 22 concept pairs.

Anchors (frozen):
  a1 Vt8 rebuild from 2927 vs 2939 npz < 1e-6
  a2 determinism < 1e-4
  a3 chunk-vs-direct < 1e-9 at L16/L17 (anchor words)
  a4 single-token 74/74 test + 3/3 anchors
  a5 B of 3 anchor words vs 2963 npz < 1e-4 rel
  a6 non-degeneracy: per-layer std > 0 (36/36) and
     per-head std at L17 > 0 (32/32)
  a7 2887 yields >= 20 concepts with both en and L words

Verdict (frozen):
  anchor fail => anchor_fail_all_void
  T1 interaction p <= 0.01 => significant_lang_class_interaction
  T1 class p <= 0.01 and lang p <= 0.01 => both_main_effects_additive
  T1 class p <= 0.01 => class_effect_replicates_bilingual_lang_ns
  T1 lang p <= 0.01 => lang_effect_only_class_ns
  else => two_factor_all_void
"""
import hashlib
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
SRC_2887 = os.path.join(BASE, 'phase2887', 'language_axis_mlp',
                        'language_axis_mlp.npz')
OUT = os.path.join(BASE, 'phase2972', 'two_factor_signature')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2972_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
N_PERM = 10000
P_TH = 0.01
RNG_T1 = 2971
RNG_T2 = {'class_en': 2972, 'class_fr': 2973,
          'lang_F': 2974, 'lang_C': 2975}
RNG_T3 = {'class_en': 2976, 'lang_F': 2977}

F_EN = ['he', 'we', 'us', 'his', 'they', 'their', 'our',
        'her', 'them', 'its', 'she', 'him', 'without',
        'among', 'unless']
F_FR = ['le', 'la', 'et', 'dans', 'pour', 'avec', 'sans',
        'elle', 'ils', 'nous', 'sur', 'mais', 'donc',
        'leur', 'comme']
ANCHOR_WORDS = ['people', 'for', 'garden']

PREREG = {
    'mode': '77 single forwards (74 two-factor test words '
            '+ 3 anchors), 2937 pass1 protocol verbatim, NO '
            'ablation; captures o_proj input pos1 all 36 '
            'layers -> C[li,w,h] per-head contribution',
    'question': 'on a merged bilingual 2x2 design, are the '
                'language axis (en/fr) and the word-class '
                'axis (F/C) separable main effects on the '
                'static band signature B, or do they '
                'interact?',
    'cells': {'F_en': F_EN, 'F_fr': F_FR,
              'C_en': '2887 en words, 22 concepts, sorted '
                      'by en tid ascending',
              'C_fr': '2887 L(=fr) words, same concept '
                      'order as C_en'},
    'anchors': {
        'a1': 'Vt8 rebuild vs 2939 npz < 1e-6',
        'a2': 'determinism < 1e-4',
        'a3': 'chunk-vs-direct < 1e-9 L16/L17',
        'a4': 'single-token 74/74 + anchors 3/3',
        'a5': 'B of 3 anchor words vs 2963 npz < 1e-4 rel',
        'a6': 'non-degeneracy 36/36 layers, 32/32 heads',
        'a7': '2887 concepts with en+L >= 20',
    },
    'T1': 'two-way Freedman-Lane on B: reduced B ~ 1+cov '
          '(cov = within-language rank(tid) standardized); '
          'residual perm rng 2971 x10000 joint; full B ~ '
          '1+cov+lang+class+lang:class; two-sided p<=0.01 '
          'per factor',
    'T2': 'layer-level contrasts (class gap en / class gap '
          'fr / language gap F / language gap C); '
          'per-contrast maxT family 36, rngs 2972-2975; '
          'interaction contrast reported descriptively '
          '(gated by T1 instead)',
    'T3': 'head-level: class gap (en) at top class-gap '
          'layer; language gap (F) at top lang-gap layer; '
          'maxT family 32, rngs 2976/2977',
    'T4': 'descriptive: vs 2964 gap_layers spearman; '
          'Simpson check pooled vs within-language; '
          'within-concept paired language B difference',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'interaction p<=0.01 => '
               'significant_lang_class_interaction; class '
               '& lang p<=0.01 => both_main_effects_additive; '
               'class only => '
               'class_effect_replicates_bilingual_lang_ns; '
               'lang only => lang_effect_only_class_ns; '
               'else => two_factor_all_void',
    'correction_note': 'run1: log line KeyError '
                       '(t2 interaction_descriptive has '
                       'no sig_layers key) after all '
                       'anchors passed and forward sweep '
                       'completed; stats code unchanged, '
                       'artifacts deleted and rerun per '
                       'discipline 3.',
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


def rankdata(x):
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman(a, b):
    ra = rankdata(np.asarray(a, dtype=np.float64))
    rb = rankdata(np.asarray(b, dtype=np.float64))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    den = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    if den < 1e-30:
        return 0.0
    return float((ra * rb).sum() / den)


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)

    # ---------- 2887 concept table (before freeze: it is
    # a sealed source; selection rule is preregistered) ----
    z87 = np.load(SRC_2887, allow_pickle=True)
    concepts = {}
    for w in z87['words']:
        parts = str(w).split(':')
        lang, tid, word = parts[0], int(parts[1]), \
            ':'.join(parts[2:])
        concepts.setdefault(tid, {})[lang] = word
    both = [(tid, d['en'], d['L'])
            for tid, d in sorted(concepts.items())
            if 'en' in d and 'L' in d]
    a7_ok = bool(len(both) >= 20)
    C_en = [w for _, w, _ in both]
    C_fr = [w for _, _, w in both]

    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2972,
                   'name': 'two_factor_signature',
                   'created': time.strftime(
                       '%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2927': sha8(SRC_2927),
                               's2939': sha8(SRC_2939),
                               's2963': sha8(SRC_2963),
                               's2887': sha8(SRC_2887)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'n_perm': N_PERM,
                   'rng': {'T1': RNG_T1, 'T2': RNG_T2,
                           'T3': RNG_T3},
                   'p_threshold': P_TH,
                   'cells': {'F_en': F_EN, 'F_fr': F_FR,
                             'C_en': C_en, 'C_fr': C_fr},
                   'anchors_words': ANCHOR_WORDS,
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen (cells: %d/%d/%d/%d)'
        % (len(F_EN), len(F_FR), len(C_en), len(C_fr)),
        lines)

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

    # ---------- model ----------
    import sys
    sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
    from phase2662_symmetric_mapping_contract import \
        load_native
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        MD, local_files_only=True, trust_remote_code=True,
        use_fast=True)
    cells = [('F', 'en', w) for w in F_EN] \
        + [('F', 'fr', w) for w in F_FR] \
        + [('C', 'en', w) for w in C_en] \
        + [('C', 'fr', w) for w in C_fr]
    tid_map = {}
    n_single = 0
    for _, _, w in cells + [('A', 'en', w)
                            for w in ANCHOR_WORDS]:
        ids = tok(' ' + w, add_special_tokens=False)[
            'input_ids']
        if len(ids) != 1:
            ids = tok(w, add_special_tokens=False)[
                'input_ids']
        if len(ids) == 1:
            n_single += 1
        tid_map[w] = int(ids[0]) if len(ids) == 1 else -1
    n_test = len(cells)
    a4_ok = bool(n_single == n_test + 3)
    log('a4 single-token %d/%d ok=%s'
        % (n_single, n_test + 3, a4_ok), lines)
    ids_the = tok(' the', add_special_tokens=False)[
        'input_ids']
    assert len(ids_the) == 1
    func_tid = int(ids_the[0])

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    log('model loaded', lines)

    cap_op = {li: [] for li in range(NL)}
    fin_cap = {}
    state_fin = {'on': False}
    handles = []

    def hook_op(li):
        def h(module, args, kwargs):
            x = args[0] if args else kwargs.get('input')
            if x is None or x.dim() < 2:
                return None
            cap_op[li].append(
                x[:, 1, :].detach().float().cpu().numpy())
            return None
        return h

    def pre_norm(module, args, kwargs):
        if state_fin['on']:
            fin_cap['x'] = args[0][:, -1, :].detach() \
                .float().cpu().numpy()

    for li in range(NL):
        handles.append(
            layers[li].self_attn.o_proj
            .register_forward_pre_hook(
                hook_op(li), with_kwargs=True))
    handles.append(model.model.norm
                   .register_forward_pre_hook(
                       pre_norm, with_kwargs=True))

    def clear_cap():
        for li in cap_op:
            del cap_op[li][:]

    def forward1(toks):
        clear_cap()
        fin_cap.pop('x', None)
        state_fin['on'] = True
        with torch.no_grad():
            model(torch.tensor([toks], device='cuda'))
        state_fin['on'] = False
        return (fin_cap['x'].astype(np.float64),
                {li: cap_op[li][0].astype(np.float64)
                 for li in range(NL)})

    M = np.zeros((NL, NH * HD))
    for li in range(NL):
        Wo = layers[li].self_attn.o_proj.weight.detach() \
            .float().cpu().numpy()
        M[li] = u35 @ Wo

    def contributions(op):
        C = np.zeros((NL, NH))
        prof = np.zeros(NL)
        for li in range(NL):
            x = op[li].reshape(-1)
            xm = (x * M[li]).reshape(NH, HD)
            C[li] = xm.sum(axis=1)
            prof[li] = float(C[li].sum())
        return C, prof

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

    # ---------- anchor forwards (a3/a5) ----------
    a5_rel = 0.0
    a3_rel = 0.0
    for w in ANCHOR_WORDS:
        fin, op = forward1([func_tid, tid_map[w]])
        _, prof = contributions(op)
        i63 = [i for i, (g, ww) in enumerate(w63)
               if ww == w][0]
        a5_rel = max(a5_rel,
                     abs(band_of(prof) - B63[i63])
                     / max(abs(B63[i63]), 1e-30))
        for li in (16, 17):
            x = op[li].reshape(-1)
            direct = float(np.dot(x, M[li]))
            xm = (x * M[li]).reshape(NH, HD)
            a3_rel = max(a3_rel,
                         abs(direct - float(xm.sum()))
                         / max(abs(direct), 1e-30))
    a5_ok = bool(a5_rel < 1e-4)
    a3_ok = bool(a3_rel < 1e-9)
    log('a3 chunk-vs-direct rel %.2e ok=%s | a5 B vs '
        '2963 rel %.2e ok=%s'
        % (a3_rel, a3_ok, a5_rel, a5_ok), lines)

    # ---------- main sweep: 74 test words ----------
    C_all = np.zeros((NL, n_test, NH))
    B_all = np.zeros(n_test)
    for i, (_, _, w) in enumerate(cells):
        fin, op = forward1([func_tid, tid_map[w]])
        C, prof = contributions(op)
        C_all[:, i, :] = C
        B_all[i] = band_of(prof)
        if (i + 1) % 20 == 0:
            log('sweep [%d/%d]' % (i + 1, n_test), lines)

    # ---------- a6 non-degeneracy ----------
    a6_ok = bool(all(C_all[li].std(axis=0).min() > 0
                     for li in range(NL))
                 and C_all[17].std(axis=0).min() > 0)
    log('a6 non-degeneracy ok=%s (C17 min std %.3e)'
        % (a6_ok, C_all[17].std(axis=0).min()), lines)

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok and a6_ok and a7_ok)
    verdict = None
    t1 = t2 = t3 = t4 = None
    save = {'C': C_all.astype(np.float32), 'B': B_all,
            'words': np.array(['%s:%s:%s' % c
                               for c in cells]),
            'tids': np.array([tid_map[w]
                              for _, _, w in cells]),
            'anchors_words': np.array(ANCHOR_WORDS)}

    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        lang = np.array([0 if c[1] == 'en' else 1
                         for c in cells])
        cls = np.array([0 if c[0] == 'F' else 1
                        for c in cells])
        tids = np.array([tid_map[w]
                         for _, _, w in cells],
                        dtype=np.float64)
        # covariate: within-language rank(tid), standardized
        cov = np.zeros(n_test)
        for lv in (0, 1):
            m = lang == lv
            cov[m] = rankdata(tids[m])
        cov = (cov - cov.mean()) / cov.std()

        # ---------- T1 two-way Freedman-Lane ----------
        Xr = np.stack([np.ones(n_test), cov], axis=1)
        beta_r, *_ = np.linalg.lstsq(Xr, B_all,
                                     rcond=None)
        resid = B_all - Xr @ beta_r
        Xf = np.stack([np.ones(n_test), cov,
                       lang.astype(float),
                       cls.astype(float),
                       (lang * cls).astype(float)],
                      axis=1)

        def full_coefs(bb):
            beta, *_ = np.linalg.lstsq(Xf, bb,
                                       rcond=None)
            return beta[2], beta[3], beta[4]

        obs1 = full_coefs(B_all)
        rng1 = np.random.default_rng(RNG_T1)
        cnt = np.zeros(3)
        for _ in range(N_PERM):
            ep = rng1.permutation(resid)
            bc = full_coefs(Xr @ beta_r + ep)
            for k in range(3):
                if abs(bc[k]) >= abs(obs1[k]) - 1e-12:
                    cnt[k] += 1
        p1 = [(c + 1) / (N_PERM + 1) for c in cnt]
        cell_med = {}
        for lab_v in range(4):
            m = ((lang == lab_v % 2)
                 & (cls == lab_v // 2))
            cell_med['%s_%s' % (
                'F' if lab_v // 2 == 0 else 'C',
                'en' if lab_v % 2 == 0 else 'fr')] \
                = round(float(np.median(B_all[m])), 4)
        t1 = {'coef_lang': round(float(obs1[0]), 4),
              'coef_class': round(float(obs1[1]), 4),
              'coef_interaction':
                  round(float(obs1[2]), 4),
              'p_lang': float('%.3e' % p1[0]),
              'p_class': float('%.3e' % p1[1]),
              'p_interaction':
                  float('%.3e' % p1[2]),
              'cell_medians_B': cell_med}
        log('T1 lang p %.3e class p %.3e inter p %.3e | '
            'cells %s' % (p1[0], p1[1], p1[2], cell_med),
            lines)

        # ---------- T2 layer-level contrasts ----------
        mFe, mCe = (lang == 0) & (cls == 0), \
            (lang == 0) & (cls == 1)
        mFf, mCf = (lang == 1) & (cls == 0), \
            (lang == 1) & (cls == 1)
        masks4 = {'class_en': (mFe, mCe),
                  'class_fr': (mFf, mCf),
                  'lang_F': (mFe, mFf),
                  'lang_C': (mCe, mCf)}

        def lay_gaps(mat, mA, mB):
            return np.array([float(np.median(
                mat[li][mA].sum(axis=1))
                - np.median(mat[li][mB].sum(axis=1)))
                for li in range(NL)])

        lay_obs = {}
        for name, (mA, mB) in masks4.items():
            lay_obs[name] = lay_gaps(C_all, mA, mB)
        inter_l = lay_obs['class_en'] \
            - lay_obs['class_fr']

        rngs = {k: np.random.default_rng(v)
                for k, v in RNG_T2.items()}
        fam = {k: np.zeros(N_PERM) for k in masks4}
        idx = np.arange(n_test)
        for k in range(N_PERM):
            for name, (mA, mB) in masks4.items():
                r = rngs[name].permutation(idx)
                Cr = C_all[:, r, :]
                g = lay_gaps(Cr, mA, mB)
                fam[name][k] = float(np.abs(g).max())
        inter_l = lay_obs['class_en'] \
            - lay_obs['class_fr']
        t2 = {}
        for name in masks4:
            thr = float(np.quantile(fam[name],
                                    1 - P_TH))
            g = lay_obs[name]
            sig = [li for li in range(NL)
                   if abs(g[li]) >= thr]
            order = np.argsort(-np.abs(g))
            t2[name] = {
                'sig_layers': sig,
                'top3': [(int(li),
                          round(float(g[li]), 4))
                         for li in order[:3]],
                'maxT_thr': round(thr, 4)}
        inter_top3 = [(int(li),
                       round(float(inter_l[li]), 4))
                      for li in np.argsort(
                          -np.abs(inter_l))[:3]]
        t2['interaction_descriptive'] = {
            'top3_layers': inter_top3}
        log('T2 sig layers: %s'
            % {k: v.get('sig_layers', 'descriptive')
               for k, v in t2.items()}, lines)

        # ---------- T3 head-level ----------
        def head_gaps(mat1, mA, mB):
            return np.array([float(np.median(
                mat1[mA][:, h]) - np.median(
                mat1[mB][:, h]))
                for h in range(NH)])

        def maxT_heads(mat1, mA, mB, seed):
            g = head_gaps(mat1, mA, mB)
            rng = np.random.default_rng(seed)
            f = np.zeros(N_PERM)
            for k in range(N_PERM):
                r = rng.permutation(idx)
                f[k] = float(np.abs(head_gaps(
                    mat1[r, :], mA, mB)).max())
            thr = float(np.quantile(f, 1 - P_TH))
            sig = [h for h in range(NH)
                   if abs(g[h]) >= thr]
            order = np.argsort(-np.abs(g))
            return g, {'sig_heads': sig,
                       'top5': [(int(h), round(
                           float(g[h]), 4))
                           for h in order[:5]],
                       'maxT_thr': round(thr, 4)}

        li_class = int(np.argsort(-np.abs(
            lay_obs['class_en']))[0])
        li_lang = int(np.argsort(-np.abs(
            lay_obs['lang_F']))[0])
        _, t3_class = maxT_heads(C_all[li_class],
                                 mFe, mCe,
                                 RNG_T3['class_en'])
        _, t3_lang = maxT_heads(C_all[li_lang],
                                mFe, mFf,
                                RNG_T3['lang_F'])
        t3 = {'layer_class_en': li_class,
              'class_en': t3_class,
              'layer_lang_F': li_lang,
              'lang_F': t3_lang}
        log('T3 L%d class sig heads %s | L%d lang sig '
            'heads %s' % (li_class,
                          t3_class['sig_heads'],
                          li_lang,
                          t3_lang['sig_heads']), lines)

        # ---------- T4 descriptive ----------
        z64 = np.load(os.path.join(BASE, 'phase2964',
                                   'carrier_anatomy',
                                   'carrier_anatomy.npz'),
                      allow_pickle=True)
        gap64 = z64['gap_layers'].astype(np.float64)
        mCe_only = np.array([c[0] == 'C'
                             and c[1] == 'en'
                             for c in cells])
        rho64 = spearman(lay_obs['class_en'], gap64)
        # Simpson check: pooled class gap vs within-lang
        pooled = float(np.median(B_all[cls == 0])
                       - np.median(B_all[cls == 1]))
        within = [float(np.median(B_all[(cls == 0)
                                        & (lang == lv)])
                        - np.median(B_all[(cls == 1)
                                          & (lang == lv)]))
                  for lv in (0, 1)]
        # within-concept paired language B difference (C)
        pair_d = []
        for j in range(len(C_en)):
            i_en = [i for i, c in enumerate(cells)
                    if c[0] == 'C' and c[1] == 'en'
                    and c[2] == C_en[j]][0]
            i_fr = [i for i, c in enumerate(cells)
                    if c[0] == 'C' and c[1] == 'fr'
                    and c[2] == C_fr[j]][0]
            pair_d.append(B_all[i_fr] - B_all[i_en])
        pair_d = np.array(pair_d)
        t4 = {'spearman_classgap_en_vs_2964':
                  round(float(rho64), 4),
              'pooled_class_gap_B': round(pooled, 4),
              'within_lang_class_gap_B':
                  {'en': round(within[0], 4),
                   'fr': round(within[1], 4)},
              'paired_lang_B': {
                  'n_pairs': int(len(pair_d)),
                  'median_d': round(float(
                      np.median(pair_d)), 4),
                  'n_fr_larger': int((pair_d > 0)
                                     .sum())}}
        log('T4 rho64 %.4f | pooled %.4f within en %.4f '
            'fr %.4f | paired n=%d med %.4f fr> %d'
            % (rho64, pooled, within[0], within[1],
               len(pair_d),
               float(np.median(pair_d)),
               int((pair_d > 0).sum())), lines)

        # ---------- verdict (frozen map) ----------
        if p1[2] <= P_TH:
            verdict = \
                'significant_lang_class_interaction'
        elif p1[1] <= P_TH and p1[0] <= P_TH:
            verdict = 'both_main_effects_additive'
        elif p1[1] <= P_TH:
            verdict = \
                'class_effect_replicates_bilingual_lang_ns'
        elif p1[0] <= P_TH:
            verdict = 'lang_effect_only_class_ns'
        else:
            verdict = 'two_factor_all_void'
        for _n, _g in lay_obs.items():
            save['gap_layers_' + _n] = _g
        save['interaction_layers'] = inter_l

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2972, 'model': 'qwen3-4b',
           'prereg': PREREG,
           'anchors': {'a1_diff': float('%.3e' % a1_diff),
                       'a1_ok': a1_ok,
                       'a2_rel': float('%.3e' % a2_rel),
                       'a2_ok': a2_ok,
                       'a3_rel': float('%.3e' % a3_rel),
                       'a3_ok': a3_ok,
                       'a4_ok': a4_ok,
                       'a5_rel': float('%.3e' % a5_rel),
                       'a5_ok': a5_ok,
                       'a6_ok': a6_ok,
                       'a7_ok': a7_ok,
                       'ok': anchor_ok},
           'T1': t1, 'T2': t2, 'T3': t3, 'T4': t4,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if verdict != 'anchor_fail_all_void':
        np.savez_compressed(os.path.join(
            OUT, 'two_factor_signature.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2972 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    import torch  # noqa: E402
    main()
