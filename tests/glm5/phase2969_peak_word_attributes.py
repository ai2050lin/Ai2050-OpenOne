# -*- coding: utf-8 -*-
"""Phase 2969: peak-position (biphasic C15 peak loc) word-attribute
explanation. 2968 showed individual peak positions span 0.29-1.61
(median 0.6746 locked to s_c 0.6567).  Question: what explains the
WIDTH of that distribution?

QUASI-POST-HOC PHASE (discipline 9): peak positions derive from the
SEALED 2968 npz and were partially displayed there (n=40, median,
A11 rho).  Every test below is registered quasi_post_hoc; verdicts
carry a _descriptive suffix and carry explanatory (not confirmatory)
weight.  No new forward passes; all quantities recomputed offline
from sealed npz with verbatim 2968 implementations.

Design variables (result-free):
  words 57 'en:<cid>:<name>' / 'L:<cid>:<name>' pairs (en/fr);
  labels_lang (0=en, 1=L); tid from tokenizer (' '+name single token;
  1 multi-token word exists, tid=-1, excluded from tid tests);
  concept id = middle field (pair structure).

Anchors (recomputation anchors vs sealed 2968 result.json):
  a1: n_peak recomputed == 40 (int exact)
  a2: median_peak_loc recomputed == 0.6746 (|diff| < 5.01e-4)
  a3: A11pk_vs_C15pk_rho recomputed == 0.9542 (|diff| < 5.01e-4)
  a4: cross-phase npz anchor: C34_A (11,57,32) vs 2967 C_all[:,34,:,:]
      max|diff| == 0 (bit level; 2968 a12 already proved identity)
  a5: peak_buckets recomputed histogram == sealed buckets (int exact)

Tests:
  T1 (quasi-post-hoc, tid/frequency): within-lang spearman(pk, tid)
      for en group and L group separately (peak words with single-
      token tid), permutation p within group (rng 2979, 10000),
      family=2 combined via maxT (min-p across the two groups);
      gate: maxT q < 0.05 -> tid explanation.  Simpson audit
      (discipline 2963): full-sample spearman reported alongside;
      sign disagreement full-vs-within is registered explicitly.
  T2 (quasi-post-hoc, language, PAIRED): same-concept en/L pairs
      where BOTH members are peak words; d_k = pk_L - pk_en;
      sign-flip permutation (rng 2980, 10000), statistic mean(d);
      validity gate n_pairs >= 10; gate p < 0.05 -> language
      explanation.
  T3 (descriptive): variance decomposition of pk: total var, share
      explained by lang (paired structure), residual after tid
      within-lang regression, residual-pk vs A11-peak rho.

Verdict map (frozen):
  t1 & t2        -> peak_source_tid_and_lang_descriptive
  t1 & ~t2       -> peak_source_tid_within_lang_descriptive
  ~t1 & t2       -> peak_source_lang_within_pair_descriptive
  ~t1 & ~t2      -> peak_source_not_word_attributes_descriptive
  (invalid branches registered as-is, no rerun)

Discipline applied: 9 (quasi-post-hoc labelling), 10 (gates checked
for reachability: group sizes 22/35, tid std 8145/23557; n_pairs
validity gate with invalid branch), 2963 Simpson norm, 3 (rerun
rules), 24 (explicit constants), 2948 (rounding-aware anchor gates).
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result' \
       r'\rdc_query_construction_20260913'
SRC68 = os.path.join(BASE, 'phase2968', 'h15_peak_anatomy')
SRC67 = os.path.join(BASE, 'phase2967', 'collapse_carrier_anatomy')
OUT = os.path.join(BASE, 'phase2969', 'peak_word_attributes')
SCRIPT = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\phase2969_peak_word_attributes.py'

N_PERM = 10000
RNG_T1, RNG_T2 = 2979, 2980
H_TGT = 15
GATE_ANCHOR = 5.01e-4
GATE_PAIRS_MIN = 10
GATE_MAXT_Q = 0.05
GATE_T2_P = 0.05

PEAK_LOC_SRC = '''(verbatim 2968 peak_loc: inner argmax -> 3-point
parabola vertex, None for endpoint peaks, offset clipped to
[-0.5, 0.5] of grid span)'''


def peak_loc(s_grid, y):
    am = int(np.argmax(y))
    if am == 0 or am == len(y) - 1:
        return None
    den = y[am - 1] - 2.0 * y[am] + y[am + 1]
    if abs(den) < 1e-30:
        return float(s_grid[am])
    off = 0.5 * (y[am - 1] - y[am + 1]) / den
    return float(s_grid[am] + np.clip(off, -0.5, 0.5)
                 * (s_grid[am + 1] - s_grid[am - 1]))


def log(msg, lines):
    lines.append(msg)
    print(msg)


def rankdata(x):
    """Average ranks (ties)."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind='stable')
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


def spearman(x, y):
    rx, ry = rankdata(x), rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    if den < 1e-30:
        return 0.0
    return float((rx * ry).sum() / den)


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2969,
                   'name': 'peak_word_attributes',
                   'created':
                       time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       hashlib.sha256(open(SCRIPT,
                                           'rb').read())
                       .hexdigest()[:8],
                   'prereg': {
                       'source': '2968 npz verbatim (no new '
                                 'forwards)',
                       'quasi_post_hoc': True,
                       'anchors': {
                           'a1': 'n_peak recompute == 40 exact',
                           'a2': 'median peak recompute == '
                                 '0.6746 (< 5.01e-4)',
                           'a3': 'A11pk_vs_C15pk_rho recompute '
                                 '== 0.9542 (< 5.01e-4)',
                           'a4': 'C34_A vs 2967 C_all bit 0',
                           'a5': 'peak_buckets recompute exact'},
                       'T1': 'within-lang spearman(pk, tid), '
                             'permutation 10000 rng 2979, '
                             'family=2 maxT gate q<0.05; '
                             'Simpson audit full-vs-within',
                       'T2': 'paired en/L same-concept peak '
                             'diff, sign-flip 10000 rng 2980, '
                             'validity n_pairs>=10, gate '
                             'p<0.05',
                       'T3': 'descriptive variance '
                             'decomposition',
                       'verdict_map': {
                           't1_t2': 'peak_source_tid_and_lang_'
                                    'descriptive',
                           't1_only': 'peak_source_tid_within_'
                                      'lang_descriptive',
                           't2_only': 'peak_source_lang_within_'
                                      'pair_descriptive',
                           'neither': 'peak_source_not_word_'
                                      'attributes_'
                                      'descriptive'}}},
                  f, ensure_ascii=False, indent=1)

    # ---------- load sealed sources ----------
    z68 = np.load(os.path.join(SRC68, 'h15_peak.npz'),
                  allow_pickle=True)
    z67 = np.load(os.path.join(SRC67, 'collapse_carrier.npz'),
                  allow_pickle=True)
    r68 = json.load(open(os.path.join(SRC68, 'result.json'),
                         encoding='utf-8'))
    words_raw = [str(w) for w in z68['words']]
    labels = z68['labels_lang'].astype(int)
    s_grid = z68['s_grid'].astype(np.float64)
    C34_A = z68['C34_A'].astype(np.float64)   # (11, 57, 32)
    A11_A = z68['A11_L34_A'].astype(np.float64)  # (11, 57, 32)
    n_words = len(words_raw)

    # parse word fields: lang = field0, concept id = field1
    lang = np.array([0 if w.startswith('en:') else 1
                     for w in words_raw])
    concept = np.array([w.split(':')[1] for w in words_raw])
    name = [w.split(':')[2] for w in words_raw]

    # design-variable tokenizer probe (result-free)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b',
        local_files_only=True, trust_remote_code=True,
        use_fast=True)
    tid = np.full(n_words, -1, dtype=np.int64)
    n_multi = 0
    for i, nm in enumerate(name):
        ids = tok(' ' + nm, add_special_tokens=False)['input_ids']
        if len(ids) == 1:
            tid[i] = int(ids[0])
        else:
            n_multi += 1

    # ---------- peaks (verbatim 2968) ----------
    C15 = C34_A[:, :, H_TGT]          # (11, 57)
    pk_all = [peak_loc(s_grid, C15[:, i]) for i in range(n_words)]
    peak_words = [i for i in range(n_words)
                  if pk_all[i] is not None]
    pk = np.array([pk_all[i] for i in peak_words])
    n_peak = len(peak_words)
    med_peak = float(np.median(pk))

    # A11 peaks (h15) for the same peak words
    A15 = A11_A[:, :, H_TGT]
    apk_all = [peak_loc(s_grid, A15[:, i]) for i in peak_words]
    ap_ok = [k for k, v in enumerate(apk_all) if v is not None]
    pk_sub = pk[ap_ok]
    ap_sub = np.array([apk_all[k] for k in ap_ok])

    # ---------- anchors ----------
    t1_68 = r68['T1_peaks']
    t3_68 = r68['T3_descriptive']
    a1_diff = int(abs(n_peak - int(t1_68['n_peak_words'])))
    a1_ok = bool(a1_diff == 0)
    a2_diff = abs(med_peak - float(t1_68['median_peak_loc']))
    a2_ok = bool(a2_diff < GATE_ANCHOR)
    a3_recomp = spearman(pk_sub, ap_sub)
    a3_diff = abs(a3_recomp - float(t3_68['A11pk_vs_C15pk_rho']))
    a3_ok = bool(a3_diff < GATE_ANCHOR)
    a4_diff = float(np.abs(
        C34_A - z67['C_all'][:, 34, :, :].astype(np.float64)).max())
    a4_ok = bool(a4_diff == 0.0)

    # a5: bucket histogram recomputation
    buckets = {}
    for v in pk:
        k = str(round(v, 2))
        buckets[k] = buckets.get(k, 0) + 1
    sealed_buckets = {k: int(v) for k, v
                      in t3_68['peak_buckets'].items()}
    a5_ok = bool(buckets == sealed_buckets)
    a5_diff = int(sum(abs(buckets.get(k, 0)
                          - sealed_buckets.get(k, 0))
                      for k in set(buckets) | set(sealed_buckets)))

    # ---------- T1: within-lang tid spearman (quasi-post-hoc) ----------
    pw_set = set(peak_words)
    group_res = {}
    perm_rng = np.random.default_rng(RNG_T1)
    for gval, gname in ((0, 'en'), (1, 'L')):
        idx = [i for i in peak_words
               if lang[i] == gval and tid[i] > 0]
        res = {'n': len(idx)}
        if len(idx) >= 8 and np.std(tid[idx]) > 0:
            pv_g = np.array([pk_all[i] for i in idx])
            tv_g = tid[idx].astype(np.float64)
            obs = spearman(tv_g, pv_g)
            cnt = 0
            for _ in range(N_PERM):
                tv_p = perm_rng.permutation(tv_g)
                if abs(spearman(tv_p, pv_g)) >= abs(obs):
                    cnt += 1
            p_two = (cnt + 1) / (N_PERM + 1)
            res.update({'rho': round(obs, 4),
                        'p_two_sided': float('%.4g' % p_two),
                        'valid': True})
        else:
            res.update({'valid': False,
                        'reason': 'n<8 or tid std==0'})
        group_res[gname] = res

    # maxT across the (up to 2) valid within-lang tests
    valid_gs = [g for g in ('en', 'L') if group_res[g]['valid']]
    t1_sig = False
    if len(valid_gs) == 2:
        # per-group null arrays + joint maxT stream
        gdata = {}
        for g in valid_gs:
            gval = 0 if g == 'en' else 1
            idx = [i for i in peak_words
                   if lang[i] == gval and tid[i] > 0]
            gdata[g] = (tid[idx].astype(np.float64),
                        np.array([pk_all[i] for i in idx]))
        nulls = {}
        for g in valid_gs:
            tv, pv = gdata[g]
            rr = np.random.default_rng(RNG_T1)
            arr = np.empty(N_PERM)
            for b in range(N_PERM):
                arr[b] = abs(spearman(rr.permutation(tv), pv))
            nulls[g] = arr
        # maxT: threshold = max over groups of |obs| under joint
        # null of the same label permutation stream
        jr = np.random.default_rng(RNG_T1 + 555)
        joint = np.empty(N_PERM)
        for b in range(N_PERM):
            joint[b] = max(abs(spearman(
                jr.permutation(gdata[g][0]), gdata[g][1]))
                for g in valid_gs)
        obs_max = max(abs(group_res[g]['rho']) for g in valid_gs)
        q_maxt = float((np.sum(joint >= obs_max) + 1)
                       / (N_PERM + 1))
        t1_sig = bool(q_maxt < GATE_MAXT_Q)
        t1_extra = {'q_maxT': float('%.4g' % q_maxt),
                    'obs_max_abs_rho': round(float(obs_max), 4)}
    else:
        # single valid group: use its own p, family=1
        t1_extra = {'family': 1, 'groups_valid': valid_gs}
        if len(valid_gs) == 1:
            t1_sig = bool(group_res[valid_gs[0]]['p_two_sided']
                          < GATE_MAXT_Q)
    # Simpson audit: full-sample vs within-group signs
    full_idx = [i for i in peak_words if tid[i] > 0]
    rho_full = spearman(tid[full_idx].astype(np.float64),
                        np.array([pk_all[i] for i in full_idx]))
    signs = [np.sign(group_res[g]['rho']) for g in valid_gs]
    simpson_flag = bool(len(valid_gs) >= 1
                        and any(s != 0 and s != np.sign(rho_full)
                                for s in signs))

    # ---------- T2: paired language test (quasi-post-hoc) ----------
    pairs = []
    cidx = {}
    for i in peak_words:
        cidx.setdefault(concept[i], {})[lang[i]] = i
    for cid, d in cidx.items():
        if 0 in d and 1 in d:
            pairs.append((d[0], d[1]))
    n_pairs = len(pairs)
    t2_valid = bool(n_pairs >= GATE_PAIRS_MIN)
    t2_sig = False
    t2 = {'n_pairs': n_pairs, 'valid': t2_valid}
    if t2_valid:
        d = np.array([pk_all[fr] - pk_all[en]
                      for en, fr in pairs])
        obs_d = float(d.mean())
        rr2 = np.random.default_rng(RNG_T2)
        cnt2 = 0
        for _ in range(N_PERM):
            sgn = rr2.choice([-1.0, 1.0], size=len(d))
            if abs(float((sgn * d).mean())) >= abs(obs_d):
                cnt2 += 1
        p2 = (cnt2 + 1) / (N_PERM + 1)
        t2_sig = bool(p2 < GATE_T2_P)
        t2.update({'mean_d_L_minus_en': round(obs_d, 4),
                   'p_signflip': float('%.4g' % p2),
                   'median_d': round(float(np.median(d)), 4),
                   'frac_d_negative': round(
                       float((d < 0).mean()), 4)})

    # ---------- T3: variance decomposition (descriptive) ----------
    pv = pk.astype(np.float64)
    var_total = float(pv.var())
    # lang share via paired structure: between-concept vs
    # within-concept (lang) decomposition over full peak set
    # using available pairs + singletons: simpler ANOVA-style
    # between-lang on unpaired data (descriptive)
    en_pk = np.array([pk_all[i] for i in peak_words
                      if lang[i] == 0])
    l_pk = np.array([pk_all[i] for i in peak_words
                     if lang[i] == 1])
    grand = float(pv.mean())
    ss_lang = (len(en_pk) * (en_pk.mean() - grand) ** 2
               + len(l_pk) * (l_pk.mean() - grand) ** 2)
    ss_tot = float(((pv - grand) ** 2).sum())
    eta2_lang = float(ss_lang / ss_tot) if ss_tot > 0 else None
    # residual after within-lang linear tid detrend
    resid = pv.copy()
    for g in valid_gs:
        gval = 0 if g == 'en' else 1
        idx = [k for k, i in enumerate(peak_words)
               if lang[i] == gval and tid[i] > 0]
        tv = tid[[peak_words[k] for k in idx]].astype(np.float64)
        b = np.polyfit(tv, pv[idx], 1)
        for k, j in enumerate(idx):
            resid[j] = pv[j] - (b[0] * tv[k] + b[1])
    var_resid = float(resid.var())
    # residual pk vs A11 peak (common subset)
    resid_sub = np.array([resid[k] for k in ap_ok])
    rho_resid_ap = spearman(resid_sub, ap_sub)
    t3 = {'var_total': round(var_total, 4),
          'eta2_lang_unpaired': (None if eta2_lang is None
                                 else round(eta2_lang, 4)),
          'var_after_tid_detrend': round(var_resid, 4),
          'var_share_after_detrend': round(
              var_resid / var_total, 4) if var_total > 0 else None,
          'resid_pk_vs_A11pk_rho': round(rho_resid_ap, 4)}

    # ---------- verdict ----------
    anchors_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                      and a5_ok)
    if t1_sig and t2_sig:
        verdict = 'peak_source_tid_and_lang_descriptive'
    elif t1_sig and not t2_sig:
        verdict = 'peak_source_tid_within_lang_descriptive'
    elif (not t1_sig) and t2_sig:
        verdict = 'peak_source_lang_within_pair_descriptive'
    else:
        verdict = 'peak_source_not_word_attributes_descriptive'

    log('anchors: a1 %s a2 %s a3 %s a4 %s a5 %s'
        % (a1_ok, a2_ok, a3_ok, a4_ok, a5_ok), lines)
    log('T1 groups: %s' % group_res, lines)
    log('T1 extra: %s sig=%s simpson_flag=%s rho_full=%.4f'
        % (t1_extra, t1_sig, simpson_flag, rho_full), lines)
    log('T2: %s sig=%s' % (t2, t2_sig), lines)
    log('T3: %s' % t3, lines)
    log('verdict: %s' % verdict, lines)

    result = {
        'phase': 2969,
        'name': 'peak_word_attributes',
        'model': 'qwen3-4b',
        'quasi_post_hoc': True,
        'quasi_rationale': (
            'peak positions derive from sealed 2968 npz and were '
            'partially displayed in its result (n=40, median, '
            'A11 rho); all tests explanatory weight only'),
        'anchors': {'ok': anchors_ok,
                    'a1_diff': a1_diff, 'a2_diff': round(a2_diff, 6),
                    'a3_diff': round(a3_diff, 6),
                    'a3_recomp': round(a3_recomp, 6),
                    'a4_max_abs_diff': a4_diff,
                    'a5_bucket_total_absdiff': a5_diff,
                    'n_multi_token_words': n_multi},
        'T1_within_lang_tid': dict(group_res, **{
            'pass': t1_sig, 'extra': t1_extra,
            'rho_full_sample': round(rho_full, 4),
            'simpson_flag': simpson_flag}),
        'T2_paired_lang': dict(t2, pass_=t2_sig),
        'T3_descriptive': t3,
        'final_verdict': verdict,
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    np.savez_compressed(
        os.path.join(OUT, 'peak_attr.npz'),
        words=np.array(words_raw), labels_lang=labels,
        tid=tid, s_grid=s_grid,
        peak_word_idx=np.array(peak_words),
        pk=pk, C15=C15, A11_h15=A15)

    with open(os.path.join(OUT, 'run_log.txt'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('DONE %s' % verdict)


if __name__ == '__main__':
    main()
