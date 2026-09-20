# -*- coding: utf-8 -*-
"""Phase 2962: word-class mechanism signature matrix
(preregistered). Plan-v2 stage-2 opening move.

Why: 2961 compressed the 23-link chain into primitive
cards; the corrected idea-1 says class-specific mechanism
should live at the ROUTING/GAIN layer (2952/2953 head
gains), possibly in the load-band profile (2932/2935),
while the readout-coordinate layer is word-attribute
BLIND (2940). This phase tests that prediction with a
balanced 3-class word list.

Design (frozen before any observation):
  45 English single-token words, 3 groups x 15
  (concrete / abstract / function; tokenizer pre-check
  passed 15/15/15 before freezing, probe file
  p2962_tok_probe.txt). Protocol per word: ONE single
  forward [the, w] (func_tid first, word at pos 1;
  2937 pass1 protocol verbatim). NO ablation.

Captures per forward:
  - o_proj input pos1 (4096) at all 36 layers
    (o_proj forward_pre_hook, 2947/2953 spec)
  - final-norm input residual pos1 (model.norm pre-hook)

Signatures per word w:
  S1 routing head distribution: C[17, w, :] (32 per-head
     linear snapshot contributions to u35 readout,
     2947 style), per-word L2-normalized -> 3-group
     between/within variance ratio stat; label
     permutation null (rng 2962, 10000).
  S2 readout SVD coordinates: coords[w, 1..8] =
     fin_res @ Vt8.T (2939 basis); per-k stat
     med(concrete) - med(abstract), family k=1..8,
     maxT correction (rng 2963, 10000).
  S3 load-band profile: prof(w, li) = sum_h C[li, w, h];
     B(w) = mean(prof L6..L12) - mean(prof L28..L35)
     (2932/2935 bands); stat med(concrete) - med(abstract),
     permutation (rng 2964, 10000).

Anchors (frozen):
  a1 Vt8 rebuild from 2927 dirs_word vs 2939 npz < 1e-6
  a2 determinism (repeat concrete[0] forward) rel < 1e-4
  a3 Vt8 orthonormality ||Vt8 Vt8^T - I|| < 1e-8
  a4 single-token check 45/45 (in-run recheck)
  a5 head-chunk sum vs direct dot rel < 1e-9 at
     L16/L17 (implementation gate, 2948 rel-threshold
     convention)
  a6 non-degeneracy gates (discipline 12): per-word
     std of C[17] > 0 for all words; per-dim std of
     coords > 0; std of band profile > 0

Family accounting (discipline 7): 3 primary stats
(S1, S2-maxT, S3); S2 internal family size 8 handled by
maxT; primary threshold 0.01; permutation granularity
1/10001. Verdict mapping (frozen):
  anchor fail => anchor_fail_all_void
  S1<=0.01 and S3<=0.01 and S2>0.05
      => signature_routing_band_not_readout
  S1<=0.01 and S2<=0.01 and S3<=0.01
      => word_class_signature_full
  S1<=0.01 and S3>0.05 => signature_routing_only
  S1>0.05 and S2>0.05 and S3>0.05
      => signature_absent_at_measured_layers
  else => signature_mixed_pattern_registered

Confound notes registered descriptively (2940 lesson):
all 45 words English (no language confound);
rho(token_id, S3) and rho(token_id, c3) reported.
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
OUT = os.path.join(BASE, 'phase2962',
                   'word_class_signature_matrix')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2962_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
N_PERM = 10000
RNG_S1, RNG_S2, RNG_S3 = 2962, 2963, 2964
P_TH, P_NULL_TH = 0.01, 0.05
BAND_LOAD = (6, 12)    # inclusive layer band, 2932/2935
BAND_DEEP = (28, 35)

CONCRETE = ['apple', 'river', 'mountain', 'chair', 'dog',
            'car', 'tree', 'house', 'book', 'rain', 'stone',
            'bread', 'horse', 'hand', 'clock']
ABSTRACT = ['freedom', 'justice', 'memory', 'truth', 'hope',
            'love', 'fear', 'thought', 'reason', 'fate',
            'peace', 'doubt', 'glory', 'idea', 'time']
FUNCTION = ['the', 'of', 'and', 'but', 'in', 'on', 'with',
            'because', 'if', 'however', 'although', 'since',
            'while', 'then', 'also']

PREREG = {
    'mode': '45 single forwards [the, w] (2937 pass1 '
            'protocol verbatim), NO ablation; captures: '
            'o_proj input pos1 all 36 layers + final-norm '
            'input pos1; per-head contribution C[li,w,h] '
            '= x_h . (u35 @ Wo_h)',
    'question': 'do concrete / abstract / function words '
                'carry distinct mechanism signatures at '
                'the routing-gain layer (S1), the '
                'readout SVD coordinate layer (S2), and '
                'the load-band profile layer (S3)? '
                'Corrected idea-1 prediction: routing '
                'and band yes, readout no (2940).',
    'word_list': {'concrete': CONCRETE,
                  'abstract': ABSTRACT,
                  'function': FUNCTION},
    'anchors': {
        'a1': 'Vt8 rebuild from 2927 dirs_word vs 2939 '
              'npz < 1e-6',
        'a2': 'determinism repeat concrete[0] forward '
              'rel < 1e-4',
        'a3': 'Vt8 orthonormality < 1e-8',
        'a4': 'single-token 45/45 in-run recheck',
        'a5': 'head-chunk sum vs direct dot rel < 1e-9 '
              'at L16/L17',
        'a6': 'non-degeneracy: per-word std C[17] > 0 '
              '(45/45), per-dim std coords > 0 (8/8), '
              'std band profile > 0 (45/45)',
    },
    'S1': 'C[17] per-word L2-normalized (45x32), 3-group '
          'between/within variance ratio, label perm '
          'rng 2962 x10000; L16 secondary descriptive',
    'S2': 'coords = fin @ Vt8.T (45x8); per-k '
          'med(concrete)-med(abstract), maxT family 8, '
          'rng 2963 x10000',
    'S3': 'prof = sum_h C[li]; B = mean(prof L6-12) - '
          'mean(prof L28-35); med(concrete)-med(abstract), '
          'perm rng 2964 x10000',
    'family': '3 primary stats; S2 internal family 8 '
              'maxT; primary threshold 0.01; granularity '
              '1/10001',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'S1<=0.01 and S3<=0.01 and S2>0.05 => '
               'signature_routing_band_not_readout; '
               'S1<=0.01 and S2<=0.01 and S3<=0.01 => '
               'word_class_signature_full; S1<=0.01 and '
               'S3>0.05 => signature_routing_only; '
               'all >0.05 => '
               'signature_absent_at_measured_layers; '
               'else => signature_mixed_pattern_registered',
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
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2962,
                   'name': 'word_class_signature_matrix',
                   'created': time.strftime(
                       '%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2927': sha8(SRC_2927),
                               's2939': sha8(SRC_2939)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'n_perm': N_PERM,
                   'rng': {'S1': RNG_S1, 'S2': RNG_S2,
                           'S3': RNG_S3},
                   'p_thresholds': {'primary': P_TH,
                                    'null': P_NULL_TH},
                   'bands': {'load': list(BAND_LOAD),
                             'deep': list(BAND_DEEP)},
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- sources ----------
    z27 = np.load(SRC_2927, allow_pickle=True)
    dirs27 = z27['dirs_word'].astype(np.float64)
    z39 = np.load(SRC_2939, allow_pickle=True)
    Vt8_39 = z39['Vt8'].astype(np.float64)
    U, s_loc, Vt_loc = np.linalg.svd(dirs27,
                                     full_matrices=False)
    a1_diff = float(np.abs(Vt_loc[:8] - Vt8_39).max())
    a1_ok = bool(a1_diff < 1e-6)
    log('a1 Vt8 rebuild diff %.2e ok=%s'
        % (a1_diff, a1_ok), lines)
    a3_diff = float(np.abs(
        Vt8_39 @ Vt8_39.T
        - np.eye(8)).max())
    a3_ok = bool(a3_diff < 1e-8)
    log('a3 orthonormality %.2e ok=%s'
        % (a3_diff, a3_ok), lines)
    u35 = dirs27[NL - 1]

    # ---------- model ----------
    import torch
    import sys
    sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
    from phase2662_symmetric_mapping_contract import \
        load_native
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        MD, local_files_only=True, trust_remote_code=True,
        use_fast=True)
    groups = ([('concrete', w) for w in CONCRETE]
              + [('abstract', w) for w in ABSTRACT]
              + [('function', w) for w in FUNCTION])
    tid_map = {}
    n_single = 0
    for _, w in groups:
        ids = tok(' ' + w, add_special_tokens=False)[
            'input_ids']
        if len(ids) != 1:
            ids = tok(w, add_special_tokens=False)[
                'input_ids']
        assert len(ids) == 1, '%s -> %s' % (w, ids)
        tid_map[w] = int(ids[0])
        n_single += 1
    a4_ok = bool(n_single == 45)
    log('a4 single-token %d/45 ok=%s'
        % (n_single, a4_ok), lines)
    func_tid = tid_map['the']

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
            x = args[0] if args else kwargs.get(
                'input')
            if x is None or x.dim() < 2:
                return None
            cap_op[li].append(
                x[:, 1, :].detach().float().cpu()
                .numpy())
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

    # ---------- readout projectors ----------
    M = np.zeros((NL, NH * HD))
    for li in range(NL):
        Wo = layers[li].self_attn.o_proj.weight.detach() \
            .float().cpu().numpy()
        M[li] = u35 @ Wo          # (4096,)

    # ---------- a2 determinism ----------
    w0 = CONCRETE[0]
    fin_a, op_a = forward1([func_tid, tid_map[w0]])
    fin_b, _ = forward1([func_tid, tid_map[w0]])
    a2_rel = float(np.abs(fin_a - fin_b).max()
                   / max(float(np.abs(fin_a).max()),
                         1e-30))
    a2_ok = bool(a2_rel < 1e-4)
    log('a2 determinism rel %.2e ok=%s'
        % (a2_rel, a2_ok), lines)

    # ---------- main sweep: 45 single forwards ------
    fin_all = np.zeros((45, 2560))
    C = np.zeros((NL, 45, NH))
    for i, (_, w) in enumerate(groups):
        fin, op = forward1([func_tid, tid_map[w]])
        fin_all[i] = fin
        for li in range(NL):
            x = op[li]                       # (4096,)
            xm = (x * M[li]).reshape(NH, HD)
            C[li, i] = xm.sum(axis=1)
        if (i + 1) % 15 == 0:
            log('sweep [%d/45]' % (i + 1), lines)

    # ---------- a5 implementation gate ----------
    # nontrivial check: chunk-sum of per-head contributions
    # vs direct dot (Wo x).u35 recomputed from a fresh
    # capture of the same input (5-word re-forward).
    recheck = [0, 15, 30, 5, 20]
    a5_rel = 0.0
    for i in recheck:
        _, w = groups[i]
        fin_r, op = forward1([func_tid, tid_map[w]])
        for li in (16, 17):
            x = op[li]
            direct = float(np.dot(x.reshape(-1), M[li]))
            chunked = float(C[li, i].sum())
            a5_rel = max(a5_rel, abs(direct - chunked)
                         / max(abs(direct), 1e-30))
    a5_ok = bool(a5_rel < 1e-9)
    log('a5 chunk-vs-direct rel %.2e ok=%s'
        % (a5_rel, a5_ok), lines)

    # ---------- a6 non-degeneracy ----------
    sd17 = C[17].std(axis=1)
    coords = fin_all @ Vt8_39.T          # (45, 8)
    prof = C.sum(axis=2)                 # (36, 45)
    li_lo, li_hi = BAND_LOAD
    ld_lo, ld_hi = BAND_DEEP
    B = prof[li_lo:li_hi + 1].mean(axis=0) \
        - prof[ld_lo:ld_hi + 1].mean(axis=0)
    a6_ok = bool(sd17.min() > 0
                 and coords.std(axis=0).min() > 0
                 and B.std() > 0)
    log('a6 non-degeneracy: sd17 min %.3e, coord std min '
        '%.3e, B std %.3e ok=%s'
        % (sd17.min(), coords.std(axis=0).min(),
           B.std(), a6_ok), lines)

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok and a6_ok)
    verdict = None
    s1 = s2 = s3 = None

    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        lab = np.array([0] * 15 + [1] * 15 + [2] * 15)
        gi = {g: np.nonzero(lab == k)[0]
              for k, g in enumerate(
                  ['concrete', 'abstract', 'function'])}

        # ---------- S1 routing head distribution ----
        H17 = C[17] / np.maximum(
            np.linalg.norm(C[17], axis=1, keepdims=True),
            1e-30)
        means = np.stack([H17[gi['concrete']].mean(0),
                          H17[gi['abstract']].mean(0),
                          H17[gi['function']].mean(0)])
        sizes = [15, 15, 15]
        grand = H17.mean(0)

        def icc_stat(mat, mns):
            ssb = sum(sizes[k] * ((mns[k] - grand) ** 2)
                      .sum() for k in range(3))
            ssw = 0.0
            for k in range(3):
                idx = np.nonzero(lab == k)[0]
                ssw += ((mat[idx] - mns[k]) ** 2).sum()
            return ssb / max(ssw, 1e-30)

        obs1 = icc_stat(H17, means)
        rng1 = np.random.default_rng(RNG_S1)
        cnt1 = 0
        for _ in range(N_PERM):
            r = rng1.permutation(45)
            mns = np.stack([H17[r[:15]].mean(0),
                            H17[r[15:30]].mean(0),
                            H17[r[30:]].mean(0)])
            ssb = sum(15 * ((mns[k] - grand) ** 2).sum()
                      for k in range(3))
            ssw = 0.0
            for k in range(3):
                seg = H17[r[k * 15:(k + 1) * 15]]
                ssw += ((seg - mns[k]) ** 2).sum()
            v = ssb / max(ssw, 1e-30)
            if v >= obs1 - 1e-12:
                cnt1 += 1
        p1 = (cnt1 + 1) / (N_PERM + 1)
        H16 = C[16] / np.maximum(
            np.linalg.norm(C[16], axis=1, keepdims=True),
            1e-30)
        m16 = np.stack([H16[gi['concrete']].mean(0),
                        H16[gi['abstract']].mean(0),
                        H16[gi['function']].mean(0)])
        g16 = H16.mean(0)
        ssb16 = sum(15 * ((m16[k] - g16) ** 2).sum()
                    for k in range(3))
        ssw16 = sum(((H16[gi[g]] - m16[k]) ** 2).sum()
                    for k, g in enumerate(
                        ['concrete', 'abstract',
                         'function']))
        icc16 = ssb16 / max(ssw16, 1e-30)
        gap17 = means[0] - means[1]
        top5_heads = [(int(h),
                       round(float(gap17[h]), 4))
                      for h in np.argsort(
                          -np.abs(gap17))[:5]]
        s1 = {'icc17': round(float(obs1), 4),
              'p': float('%.3e' % p1),
              'icc16_descriptive': round(float(icc16), 4),
              'top5_gap_heads_ca': top5_heads}
        log('S1 ICC17 %.4f p %.3e | ICC16 %.4f (desc)'
            % (obs1, p1, icc16), lines)

        # ---------- S2 readout coordinates ----------
        st2 = np.array([float(np.median(
            coords[gi['concrete'], k])
            - np.median(coords[gi['abstract'], k]))
            for k in range(8)])

        def s2_stats(perm_lab):
            return np.array([
                float(np.median(coords[perm_lab == 0, k])
                      - np.median(coords[perm_lab == 1, k]))
                for k in range(8)])

        rng2 = np.random.default_rng(RNG_S2)
        obs2max = float(np.abs(st2).max())
        kmax = int(np.argmax(np.abs(st2)))
        cnt2 = 0
        for _ in range(N_PERM):
            r = rng2.permutation(45)
            v = float(np.abs(s2_stats(r)).max())
            if v >= obs2max - 1e-12:
                cnt2 += 1
        p2 = (cnt2 + 1) / (N_PERM + 1)
        # per-k uncorrected descriptives
        p2_per_k = []
        rng2b = np.random.default_rng(RNG_S2 + 500)
        for k in range(8):
            c = 0
            for _ in range(N_PERM):
                r = rng2b.permutation(45)
                v = abs(s2_stats(r)[k])
                if v >= abs(st2[k]) - 1e-12:
                    c += 1
            p2_per_k.append((c + 1) / (N_PERM + 1))
        s2 = {'kmax': kmax,
              'obs_max': round(obs2max, 4),
              'p_maxT': float('%.3e' % p2),
              'stats': [round(float(v), 4) for v in st2],
              'p_per_k_uncorrected': [
                  float('%.3e' % v) for v in p2_per_k],
              'func_vs_concrete_med': round(float(
                  np.median(coords[gi['function'], kmax])
                  - np.median(coords[gi['concrete'],
                                      kmax])), 4)}
        log('S2 k=%d obs %.4f maxT p %.3e | per-k %s'
            % (kmax, obs2max, p2,
               ['%.1e' % v for v in p2_per_k]), lines)

        # ---------- S3 load-band profile ----------
        st3 = float(np.median(B[gi['concrete']])
                    - np.median(B[gi['abstract']]))
        rng3 = np.random.default_rng(RNG_S3)
        cnt3 = 0
        for _ in range(N_PERM):
            r = rng3.permutation(45)
            v = float(np.median(B[r[:15]])
                      - np.median(B[r[15:30]]))
            if abs(v) >= abs(st3) - 1e-12:
                cnt3 += 1
        p3 = (cnt3 + 1) / (N_PERM + 1)
        rho_tid_B = spearman(
            [tid_map[w] for _, w in groups], B)
        rho_tid_c = spearman(
            [tid_map[w] for _, w in groups],
            coords[:, kmax])
        s3 = {'B_med_c': round(float(np.median(
                  B[gi['concrete']])), 4),
              'B_med_a': round(float(np.median(
                  B[gi['abstract']])), 4),
              'B_med_f': round(float(np.median(
                  B[gi['function']])), 4),
              'obs': round(st3, 4),
              'p': float('%.3e' % p3),
              'rho_tokid_B': round(rho_tid_B, 4),
              'rho_tokid_c': round(rho_tid_c, 4)}
        log('S3 B med c/a/f %.4f/%.4f/%.4f obs %.4f '
            'p %.3e | rho(tokid,B) %.4f'
            % (s3['B_med_c'], s3['B_med_a'],
               s3['B_med_f'], st3, p3, rho_tid_B),
            lines)

        # per-layer class-sep profile (descriptive)
        sep_prof = []
        for li in range(NL):
            pl = prof[li]
            sep_prof.append(round(float(
                np.median(pl[gi['concrete']])
                - np.median(pl[gi['abstract']])), 4))
        s3['sep_profile_concrete_minus_abstract'] = \
            sep_prof

        # ---------- verdict ----------
        if (p1 <= P_TH and p3 <= P_TH
                and p2 > P_NULL_TH):
            verdict = 'signature_routing_band_not_readout'
        elif (p1 <= P_TH and p2 <= P_TH and p3 <= P_TH):
            verdict = 'word_class_signature_full'
        elif p1 <= P_TH and p3 > P_NULL_TH:
            verdict = 'signature_routing_only'
        elif (p1 > P_NULL_TH and p2 > P_NULL_TH
                and p3 > P_NULL_TH):
            verdict = 'signature_absent_at_measured_layers'
        else:
            verdict = 'signature_mixed_pattern_registered'

        save = {'C': C.astype(np.float32),
                'coords': coords,
                'prof': prof,
                'B': B,
                'fin': fin_all,
                'tids': np.array([tid_map[w]
                                  for _, w in groups]),
                'labels': lab,
                'words': np.array(
                    ['%s:%s' % (g, w)
                     for g, w in groups])}

    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)
    res = {'phase': 2962, 'model': 'qwen3-4b',
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
                       'ok': anchor_ok},
           'S1': s1, 'S2': s2, 'S3': s3,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    if verdict != 'anchor_fail_all_void':
        np.savez_compressed(os.path.join(
            OUT, 'signature_matrix.npz'), **save)
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2962 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
