"""Phase 2993: Omega-D logic-signature registration
(plan v4 P3 / plan v3 Omega-D, applicability-tagged).

Question.  The F/C machine (2962 word-class signature, 2973
scale audit, 2986/2987 context audit) separates function vs
content words.  Omega-D asks whether LOGIC/CONNECTIVE words
(because, therefore, although, ...) carry a signature of
their own at the L34 locus, and whether that signature
EMERGES WITH CONTEXT (plan v3: "does logic aggregation
appear with context?").  Every claim must carry an
applicability-domain tag (plan v4 P0 rule).

Design (preregistered; frozen BEFORE any observation).
  cells    74 base words (2977 exec order verbatim) + L_EN
           logic connectives: preregistered 24-candidate
           list, single-token filter, alphabetical order,
           deterministic split L_A (first half) / L_B (second
           half) -- axis construction uses L_A ONLY, tests
           use held-out L_B (split-half, anti-circular).
  conds    length bins {2,16,64,256,1024} x 2986 make_seq
           verbatim (filler pool cycling, exact seq length);
           per-(length,word) independent batch (no concat).
  readout  prof_all over 36 layers along M[li]=u35@Wo[li]
           (2986/2987 verbatim, o_proj-input head space);
           headC at L34 (C34 per-head slicing, 2991/2992
           verbatim); res34 = L34 decoder-layer input
           residual (2560, d_lang_u caliber, 2992 verbatim).
  T1       PRIMARY.  Axes w2 (L2) and w1024 (L1024), each =
           unit(mean res34(L_A) - mean res34(C_en)) at that
           length.  Generalization contrast per length:
           D = mean(res34(L_B)@w) - mean(res34(C_en)@w);
           permutation N_PERM=10000 two-sided on the pooled
           L_B+C_en projection values per (axis, length);
           Bonferroni x10 family (2 axes x 5 lengths).
           Verdict from union of family-significant bins:
             no bin            -> logic_signature_absent
             short bins only   -> logic_signature_len2_only
             long bins only    -> logic_signature_
                                  context_emergent
             short + long      -> logic_signature_
                                  length_robust
  T2       SECONDARY (exploratory, maxT).  Per-head contrast
           mean_L(all logic) - mean_C(C_en) per length;
           maxT across 32 heads within length; topography
           cos(hc(bi), hc(0)) vs label-permutation null.
  T3       DESCRIPTIVE.  Position on the u35 F-C axis at
           L2/L1024: contrasts L-C and L-F (Bonferroni x4);
           answers "is logic absorbed into the F pole or
           distinct from both".

Anchors:
  a1   Vt8 rebuild vs 2939 < 1e-6
  a1w  words identity 2986/2987/here (exact, composite fmt)
  a3   single-token all cells (base74 + L_EN)
  a4   prof_all L2 bin vs 2986 npz bit < 1e-12 (74 words)
  a5   prof_all L16 bin vs 2986 npz bit < 1e-12
  a6   headC L2 & L16 vs 2987 npz bit < 1e-12
  a7   determinism double forward < 1e-4 (L2, L16)
  a8   prof34 (=prof_all[...,34]) vs 2986 'prof34' bit
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2939 = os.path.join(BASE, 'phase2939', 'rotation_target',
                        'rotation_target.npz')
SRC_2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                        'probe_relativity.npz')
SRC_2986 = os.path.join(BASE, 'phase2986',
                        'length_context_drift',
                        'length_context_drift.npz')
SRC_2987 = os.path.join(BASE, 'phase2987',
                        'context_minimal_audit',
                        'context_minimal_audit.npz')
EXEC_2977 = os.path.join(BASE, 'phase2977',
                         'two_axis_fusion_injection',
                         'execution.json')
OUT = os.path.join(BASE, 'phase2993',
                   'logic_signature_registration')

MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
L34 = 34
LENGTHS = [2, 16, 64, 256, 1024]
SHORT_BINS = (0, 1)          # L2, L16
LONG_BINS = (2, 3, 4)        # L64, L256, L1024
FILLER_TEXT = (' The sun rises in the east and sets in the '
               'west .')
L_CANDIDATES = ['because', 'therefore', 'although', 'unless',
                'however', 'thus', 'moreover', 'since',
                'whereas', 'despite', 'hence',
                'nevertheless', 'consequently',
                'furthermore', 'otherwise', 'instead',
                'while', 'accordingly', 'likewise',
                'meanwhile', 'nonetheless', 'thereafter',
                'whereby', 'albeit']
N_PERM = 10000
N_PERM_TOPO = 2000
N_FAM_T1 = 10                # 2 axes x 5 lengths
N_FAM_T3 = 4                 # 2 lengths x 2 contrasts
P_GATE = 0.05
RNG_MAIN = 2993


def sha8(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:8]


def log(msg, lines):
    lines.append('[%s] %s' % (time.strftime('%H:%M:%S'), msg))
    with open(os.path.join(OUT, 'run_log.txt'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)

    # ---------- execution freeze (BEFORE any compute) ----------
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2993,
                   'name': 'logic_signature_registration',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'prereg': {
                       'design': 'plan v4 P3 / v3 Omega-D: '
                                 '74 base words + L_EN '
                                 'connectives (%d candidates, '
                                 'single-token filter, '
                                 'alphabetical, split-half '
                                 'L_A/L_B) x length bins '
                                 '%s, 2986 make_seq verbatim; '
                                 'T1 PRIMARY split-half '
                                 'generalization: axes from '
                                 'L_A at L2 and L1024, '
                                 'contrasts on held-out L_B '
                                 'vs C_en per length, perm '
                                 'N=%d two-sided, '
                                 'Bonferroni x%d; T2 '
                                 'secondary per-head '
                                 'maxT (32 heads) + '
                                 'topography cos vs perm '
                                 'null; T3 descriptive u35 '
                                 'F-C axis position '
                                 '(L-C, L-F) x (L2, L1024) '
                                 'Bonferroni x%d'
                                 % (len(L_CANDIDATES),
                                    LENGTHS, N_PERM,
                                    N_FAM_T1, N_FAM_T3),
                       'T1_primary_verdict': 'union of '
                                            'family-sig bins: '
                                            'none -> '
                                            'logic_signature_'
                                            'absent; short '
                                            'only -> '
                                            'logic_signature_'
                                            'len2_only; long '
                                            'only -> '
                                            'logic_signature_'
                                            'context_emergent;'
                                            ' short+long -> '
                                            'logic_signature_'
                                            'length_robust; '
                                            'every claim gets '
                                            'applicability tag '
                                            '(plan v4 P0)',
                       'anchors': {
                           'a1': 'Vt8 rebuild < 1e-6',
                           'a1w': 'words identity '
                                  '2986/2987/here exact',
                           'a3': 'single-token all cells',
                           'a4': 'prof_all L2 vs 2986 '
                                 'bit < 1e-12',
                           'a5': 'prof_all L16 vs 2986 '
                                 'bit < 1e-12',
                           'a6': 'headC L2&L16 vs 2987 '
                                 'bit < 1e-12',
                           'a7': 'determinism < 1e-4',
                           'a8': 'prof34 vs 2986 key bit'},
                       'rng': RNG_MAIN,
                       'n_perm': N_PERM,
                       'n_perm_topo': N_PERM_TOPO,
                       'p_gate': P_GATE,
                       'l_candidates': L_CANDIDATES,
                       'lengths': LENGTHS}},
                  f, indent=1)
    log('execution.json frozen', lines)

    # ---------- cells (2977 exec verbatim) ----------
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
    n_base = len(cells)
    iF_en = list(range(0, 15))
    iC_en = list(range(30, 52))
    words_here_base = ['%s:%s:%s' % c for c in cells]

    # ---------- upstream artifacts ----------
    z39 = np.load(SRC_2939, allow_pickle=True)
    z27 = np.load(SRC_2927, allow_pickle=True)
    dirs27 = z27['dirs_word'].astype(np.float64)
    _, _, Vt_loc = np.linalg.svd(dirs27,
                                 full_matrices=False)
    a1_diff = float(np.abs(Vt_loc[:8]
                           - z39['Vt8']).max())
    a1_ok = bool(a1_diff < 1e-6)
    log('a1 Vt8 rebuild diff %.2e ok=%s'
        % (a1_diff, a1_ok), lines)
    u35 = dirs27[NL - 1]
    z86 = np.load(SRC_2986, allow_pickle=True)
    z87 = np.load(SRC_2987, allow_pickle=True)
    words86 = [str(w) for w in z86['words']]
    words87 = [str(w) for w in z87['words']]
    a1w_ok = bool(words86 == words87 == words_here_base)
    log('a1w words identity 2986/2987/here: %s'
        % a1w_ok, lines)
    prof86 = z86['prof_all'].astype(np.float64)
    prof34_86 = z86['prof34'].astype(np.float64)
    headC87 = z87['headC'].astype(np.float64)

    # ---------- L_EN registration ----------
    l_words_all = sorted(set(L_CANDIDATES)
                         - set(w for _, _, w in cells))
    l_words = []            # single-token only (tokenizer
    l_tid = {}              # filled after tokenizer load)
    verdict = None
    T1 = T2 = T3 = None
    tags = {}

    # ---------- model ----------
    import sys
    sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
    from phase2662_symmetric_mapping_contract import \
        load_native
    from transformers import AutoTokenizer
    import torch as _t

    tok = AutoTokenizer.from_pretrained(
        MD, local_files_only=True, trust_remote_code=True,
        use_fast=True)

    def tid_of(w):
        ids = tok(' ' + w, add_special_tokens=False)[
            'input_ids']
        if len(ids) != 1:
            ids = tok(w, add_special_tokens=False)[
                'input_ids']
        return int(ids[0]) if len(ids) == 1 else -1

    tid_map = {}
    n_single = 0
    for _, _, w in cells:
        t = tid_of(w)
        tid_map[w] = t
        if t != -1:
            n_single += 1
    a3_base_ok = bool(n_single == n_base)
    for w in l_words_all:
        t = tid_of(w)
        if t != -1:
            l_words.append(w)
            l_tid[w] = t
    l_words = sorted(l_words)
    n_L = len(l_words)
    a3_L_ok = bool(n_L >= 16)
    a3_ok = bool(a3_base_ok and a3_L_ok)
    log('a3 single-token: base %d/%d, L_EN %d/%d ok=%s'
        % (n_single, n_base, n_L, len(l_words_all), a3_ok),
        lines)
    # deterministic alphabetical split-half
    half = n_L // 2
    l_A = l_words[:half]
    l_B = l_words[half:]
    log('L_EN registered: %s | A=%s | B=%s'
        % (l_words, l_A, l_B), lines)

    ids_the = tok(' the', add_special_tokens=False)[
        'input_ids']
    assert len(ids_the) == 1
    func_tid = int(ids_the[0])
    fill_pool = tok(FILLER_TEXT,
                    add_special_tokens=False)['input_ids']
    assert len(fill_pool) >= 1

    def make_seq(w, L):
        fill = [int(fill_pool[k % len(fill_pool)])
                for k in range(L - 2)]
        seq = fill + [func_tid, tid_map[w] if w in tid_map
                      else l_tid[w]]
        assert len(seq) == L, 'seq length drift'
        return seq

    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    log('model loaded (fill_pool %d tok)'
        % len(fill_pool), lines)

    cap_op = {li: [] for li in range(NL)}
    cap_res34 = []
    handles = []

    def hs_of(args, kwargs):
        if args and args[0] is not None:
            return args[0]
        return kwargs.get('hidden_states')

    def hook_res34(module, args, kwargs):
        x = hs_of(args, kwargs)
        if x is None or x.dim() < 2:
            return None
        cap_res34.append(
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
        layers[L34].register_forward_pre_hook(
            hook_res34, with_kwargs=True))
    for li in range(NL):
        handles.append(
            layers[li].self_attn.o_proj
            .register_forward_pre_hook(
                hook_op(li), with_kwargs=True))

    def clear_cap():
        for li in cap_op:
            del cap_op[li][:]
        del cap_res34[:]

    def forward1(toks):
        clear_cap()
        with _t.no_grad():
            model(_t.tensor([toks], device='cuda'))
        return ({li: cap_op[li][0].astype(np.float64)
                 for li in range(NL)},
                cap_res34[0].astype(np.float64)
                .reshape(-1))

    # weight-space readout rows (2986/2987 verbatim)
    M = np.zeros((NL, NH * HD))
    Mnorm = np.zeros(NL)
    for li in range(NL):
        Wo = layers[li].self_attn.o_proj.weight.detach() \
            .float().cpu().numpy()
        M[li] = u35 @ Wo
        Mnorm[li] = float(np.linalg.norm(M[li]))
    Wo34 = layers[L34].self_attn.o_proj.weight.detach() \
        .float().cpu().numpy().astype(np.float64)
    C34 = u35 @ Wo34
    M34v = M[L34]

    # ---------- sweep: all words x all lengths ----------
    all_words = [c[2] for c in cells] + l_words
    n_all = len(all_words)
    iL_all = list(range(n_base, n_all))
    iL_A = [n_base + l_words.index(w) for w in l_A]
    iL_B = [n_base + l_words.index(w) for w in l_B]
    prof_all = np.zeros((len(LENGTHS), n_all, NL))
    normsL = np.zeros((len(LENGTHS), n_all, NL))
    headC = np.zeros((len(LENGTHS), n_all, NH))
    res34_all = np.zeros((len(LENGTHS), n_all, 2560))
    for bi, L in enumerate(LENGTHS):
        for i, w in enumerate(all_words):
            seq = make_seq(w, L)
            ops, res = forward1(seq)
            res34_all[bi, i] = res
            x34 = ops[L34].reshape(-1)
            for li in range(NL):
                x = ops[li].reshape(-1)
                normsL[bi, i, li] = \
                    float(np.linalg.norm(x))
                prof_all[bi, i, li] = \
                    float(np.dot(x, M[li]))
            for h in range(NH):
                headC[bi, i, h] = float(np.dot(
                    C34[h * HD:(h + 1) * HD],
                    x34[h * HD:(h + 1) * HD]))
        log('sweep L=%d done (%d words)'
            % (L, n_all), lines)

    # ---------- anchors ----------
    op_a = forward1(make_seq('man', 2))[0]
    op_b = forward1(make_seq('man', 2))[0]
    a7a = float(np.abs(op_a[30] - op_b[30]).max()
                / max(float(np.abs(op_a[30]).max()),
                      1e-30))
    op_c = forward1(make_seq('man', 16))[0]
    op_d = forward1(make_seq('man', 16))[0]
    a7b = float(np.abs(op_c[30] - op_d[30]).max()
                / max(float(np.abs(op_c[30]).max()),
                      1e-30))
    a7_ok = bool(a7a < 1e-4 and a7b < 1e-4)
    log('a7 determinism %.2e / %.2e ok=%s'
        % (a7a, a7b, a7_ok), lines)

    # a4/a5: prof_all L2/L16 vs 2986 (bit level)
    a4_diff = float(np.abs(prof_all[0, :n_base]
                           - prof86[0]).max())
    a5_diff = float(np.abs(prof_all[1, :n_base]
                           - prof86[1]).max())
    a4_ok = bool(a4_diff < 1e-12)
    a5_ok = bool(a5_diff < 1e-12)
    log('a4 prof L2 vs 2986: %.2e ok=%s; a5 L16: %.2e ok=%s'
        % (a4_diff, a4_ok, a5_diff, a5_ok), lines)

    # a6: headC vs 2987 cond0 (L2) / cond2 (L16N)
    a6a = float(np.abs(headC[0, :n_base]
                       - headC87[0]).max())
    a6b = float(np.abs(headC[1, :n_base]
                       - headC87[2]).max())
    a6_ok = bool(a6a < 1e-12 and a6b < 1e-12)
    log('a6 headC vs 2987: L2 %.2e L16 %.2e ok=%s'
        % (a6a, a6b, a6_ok), lines)

    # a8: prof34 key identity vs 2986
    a8_diff = float(np.abs(prof_all[:, :n_base, L34]
                           - prof34_86).max())
    a8_ok = bool(a8_diff < 1e-12)
    log('a8 prof34 vs 2986 key: %.2e ok=%s'
        % (a8_diff, a8_ok), lines)

    anchor_ok = bool(a1_ok and a1w_ok and a3_ok and a4_ok
                     and a5_ok and a6_ok and a7_ok and a8_ok)

    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
        log('ANCHOR FAIL -> all void', lines)
    else:
        # ---------- T1 split-half generalization ----------
        rng_t1 = np.random.default_rng(RNG_MAIN + 10)
        # projections per axis: proj[bi, i]
        T1 = {'axes': {}, 'family': {}, 'sig_bins': {}}
        sig_union = set()
        for ax_name, src_bin in (('w2', 0),
                                 ('w1024',
                                  len(LENGTHS) - 1)):
            w = (res34_all[src_bin, iL_A].mean(axis=0)
                 - res34_all[src_bin, iC_en].mean(axis=0))
            wn = float(np.linalg.norm(w))
            w = w / max(wn, 1e-30)
            proj = res34_all @ w      # (n_len, n_all)
            T1['axes'][ax_name] = {
                'src_bin': src_bin,
                'mean_norm_L_A_L%s' % LENGTHS[src_bin]:
                    round(float(np.linalg.norm(
                        res34_all[src_bin, iL_A],
                        axis=1).mean()), 4),
                'mean_norm_C':
                    round(float(np.linalg.norm(
                        res34_all[src_bin, iC_en],
                        axis=1).mean()), 4)}
            sig_bins = []
            fam = {}
            for bi, L in enumerate(LENGTHS):
                v1 = proj[bi, iL_B]
                v2 = proj[bi, iC_en]
                pool = np.concatenate([v1, v2])
                n1 = v1.size
                D_obs = float(v1.mean() - v2.mean())
                cnt = 0
                for _ in range(N_PERM):
                    pm = rng_t1.permutation(pool.size)
                    d = pool[pm[:n1]].mean() \
                        - pool[pm[n1:]].mean()
                    if abs(d) >= abs(D_obs):
                        cnt += 1
                p_raw = (cnt + 1) / (N_PERM + 1)
                p_fam = min(1.0, p_raw * N_FAM_T1)
                sd = float(pool.std())
                d_eff = D_obs / max(sd, 1e-30)
                fam['L%d' % L] = {
                    'D': round(D_obs, 6),
                    'd_eff': round(d_eff, 4),
                    'p_raw': round(p_raw, 5),
                    'p_fam': round(p_fam, 5)}
                if p_fam < P_GATE:
                    sig_bins.append(bi)
                    sig_union.add(bi)
            T1['family'][ax_name] = fam
            T1['sig_bins'][ax_name] = sig_bins
            log('T1 axis %s sig bins %s (p_fam %s)'
                % (ax_name, sig_bins,
                   json.dumps({k: v['p_fam']
                               for k, v in fam.items()})),
                lines)
        short_hit = bool(sig_union & set(SHORT_BINS))
        long_hit = bool(sig_union & set(LONG_BINS))
        T1['sig_union_bins'] = sorted(sig_union)
        if not sig_union:
            verdict = 'logic_signature_absent'
            tags['T1_logic_generalization'] = '未检出'
        elif short_hit and not long_hit:
            verdict = 'logic_signature_len2_only'
            tags['T1_logic_generalization'] = \
                '短上下文专属(L<=16)'
        elif long_hit and not short_hit:
            verdict = 'logic_signature_context_emergent'
            tags['T1_logic_generalization'] = \
                '仅长上下文(L>=%d)' % min(
                    LENGTHS[b] for b in sig_union
                    if b in LONG_BINS)
        else:
            verdict = 'logic_signature_length_robust'
            tags['T1_logic_generalization'] = '长度稳健'
        log('T1 union sig bins %s -> verdict %s'
            % (sorted(sig_union), verdict), lines)

        # ---------- T2 per-head maxT (secondary) ----------
        rng_t2 = np.random.default_rng(RNG_MAIN + 20)
        T2 = {'per_length': {}}
        hc0 = (headC[0, iL_all].mean(axis=0)
               - headC[0, iC_en].mean(axis=0))
        for bi, L in enumerate(LENGTHS):
            Ml = headC[bi, iL_all]     # (n_L, 32)
            Mc = headC[bi, iC_en]
            hc = Ml.mean(axis=0) - Mc.mean(axis=0)
            pooled = np.vstack([Ml, Mc])
            n1 = Ml.shape[0]
            null_max = np.zeros(N_PERM_TOPO)
            null_cos = np.zeros(N_PERM_TOPO)
            for pi in range(N_PERM_TOPO):
                pm = rng_t2.permutation(pooled.shape[0])
                m1 = pooled[pm[:n1]].mean(axis=0)
                m2 = pooled[pm[n1:]].mean(axis=0)
                d = m1 - m2
                null_max[pi] = float(np.abs(d).max())
                if bi > 0:
                    null_cos[pi] = float(
                        d @ hc0
                        / max(float(np.linalg.norm(d)
                                    * np.linalg.norm(hc0)),
                              1e-30))
            p_maxT = np.ones(NH)
            for h in range(NH):
                p_maxT[h] = (float(
                    np.sum(null_max
                           >= abs(hc[h]))) + 1) \
                    / (N_PERM_TOPO + 1)
            top = np.argsort(-np.abs(hc))[:3]
            ent = {'top_heads': {
                int(h): {
                    'contrast': round(float(hc[h]), 4),
                    'p_maxT': round(float(p_maxT[h]),
                                    4)}
                for h in top}}
            if bi > 0:
                cos_obs = float(
                    hc @ hc0
                    / max(float(np.linalg.norm(hc)
                                * np.linalg.norm(hc0)),
                          1e-30))
                p_cos = (float(np.sum(
                    np.abs(null_cos)
                    >= abs(cos_obs))) + 1) \
                    / (N_PERM_TOPO + 1)
                ent['topo_cos_vs_L2'] = round(cos_obs, 4)
                ent['topo_p'] = round(p_cos, 4)
            T2['per_length']['L%d' % L] = ent
            log('T2 L%d top heads %s'
                % (L, json.dumps(ent['top_heads'])),
                lines)
        tags['T2_head_topography'] = '探索性(maxT参考,不作确证)'

        # ---------- T3 u35 F-C axis position ----------
        rng_t3 = np.random.default_rng(RNG_MAIN + 30)
        T3 = {'family': {}}
        p_fam_min3 = 1.0
        for bi in (0, len(LENGTHS) - 1):
            L = LENGTHS[bi]
            proj = res34_all[bi] @ u35
            for c1, i1, c2, i2 in (
                    ('L-C', iL_all, 'C', iC_en),
                    ('L-F', iL_all, 'F', iF_en)):
                v1 = proj[i1]
                v2 = proj[i2]
                pool = np.concatenate([v1, v2])
                n1 = v1.size
                D_obs = float(np.median(v1)
                              - np.median(v2))
                cnt = 0
                for _ in range(N_PERM):
                    pm = rng_t3.permutation(pool.size)
                    d = np.median(pool[pm[:n1]]) \
                        - np.median(pool[pm[n1:]])
                    if abs(d) >= abs(D_obs):
                        cnt += 1
                p_raw = (cnt + 1) / (N_PERM + 1)
                p_fam = min(1.0, p_raw * N_FAM_T3)
                T3['family']['L%d_%s' % (L, c1)] = {
                    'median_L': round(
                        float(np.median(v1)), 4),
                    'median_ref': round(
                        float(np.median(v2)), 4),
                    'D': round(D_obs, 4),
                    'p_raw': round(p_raw, 5),
                    'p_fam': round(p_fam, 5)}
                p_fam_min3 = min(p_fam_min3, p_fam)
        T3['p_fam_min'] = round(p_fam_min3, 5)
        log('T3 F-C axis family: %s'
            % json.dumps(T3['family']), lines)
        tags['T3_FC_axis_position'] = (
            '确证性(Bonferroni x%d)' % N_FAM_T3
            if p_fam_min3 < P_GATE else '未检出')

        log('VERDICT: %s' % verdict, lines)

    elapsed = round(time.monotonic() - t0, 1)
    result = {
        'phase': 2993,
        'name': 'logic_signature_registration',
        'final_verdict': verdict,
        'anchor_all_ok': anchor_ok,
        'anchors': {
            'a1_Vt8': a1_ok,
            'a1w_words': a1w_ok,
            'a3_single_token': a3_ok,
            'a4_prof_L2_diff': a4_diff,
            'a5_prof_L16_diff': a5_diff,
            'a6_headC_diff': [a6a, a6b],
            'a7_determinism': a7_ok,
            'a8_prof34_diff': a8_diff},
        'L_EN': {'words': l_words, 'A': l_A, 'B': l_B},
        'T1': T1,
        'T2': T2,
        'T3': T3,
        'applicability_tags': tags,
        'elapsed_s': elapsed,
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    npz_path = os.path.join(
        OUT, 'logic_signature_registration.npz')
    save = {'words': np.array(['L?en?%s' % w.replace(
                                    ':', '_')
                               for w in l_words]),
            'words_base': np.array(words_here_base),
            'l_words': np.array(l_words),
            'l_A': np.array(l_A), 'l_B': np.array(l_B),
            'lengths': np.array(LENGTHS),
            'prof_all': prof_all.astype(np.float32),
            'norms': normsL.astype(np.float32),
            'headC': headC,
            'res34': res34_all.astype(np.float32)}
    if anchor_ok and T1 is not None:
        save['w2'] = (res34_all[0, iL_A].mean(axis=0)
                      - res34_all[0, iC_en].mean(axis=0))
        save['w1024'] = (res34_all[-1, iL_A].mean(axis=0)
                         - res34_all[-1, iC_en]
                         .mean(axis=0))
    np.savez_compressed(npz_path, **save)
    with open(os.path.join(OUT, 'seal.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'sealed_at':
                   time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'verdict': verdict,
                   'npz_sha256_8':
                       sha8(npz_path),
                   'result_sha256_8':
                       sha8(os.path.join(OUT,
                                         'result.json'))},
                  f, indent=1)
    log('PHASE2993 DONE elapsed=%ss verdict=%s'
        % (elapsed, verdict), lines)
    print('PHASE2993 DONE', verdict, elapsed)


if __name__ == '__main__':
    main()
