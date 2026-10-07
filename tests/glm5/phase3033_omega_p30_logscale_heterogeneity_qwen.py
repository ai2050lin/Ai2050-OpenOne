# -*- coding: utf-8 -*-
"""Phase 3033: Omega-P30 - log-scale reparameterization
of the dose-response heterogeneity (reanalysis phase,
CPU only, no model run).

Question (3031 follow-up A / 3032 menu B): 3031 showed
rel_dev heterogeneity is tag-idiosyncratic (0/9 family
survivors) with an exploratory scale dependence
(erase_mag rho -0.527; the largest-baseline tags P0/P7
are the only negative rel_dev tags).  Is the convexity
exponent a SCALE ARTIFACT (small-baseline tags inflate
ratios) or a GENUINE SHAPE difference?  Reparameterize
in log space: outcome = log js_ratio2 = log js2 - log je
(scale-free), predictor = log je (scale).

Design: pure reanalysis of sealed artifacts
(3028 dose npz = js_dose/js_erase/rel_dev;
 3029 recruitment npz = rho/g_exp/band shares;
 3022 relay npz = s_relay -> coalition features;
 3020 readout npz = lens traj -> lstar;
 3031 npz = sealed heterogeneity table for anchors).

T2a PRIMARY: Spearman(gamma_12, log je) across 11 tags
  where gamma_12 = log2(js2/js_erase) is the local
  log-log dose exponent (alpha 1->2); exact MC
  permutation null (200k, seed 3033).  Negative+sig =>
  exponent shrinks with scale (artifact direction).
T2b: OLS log js2 ~ a + b*log je; pairs bootstrap CI
  (200k, seed 30331) for b; R2.  b=1 <=> pure
  amplitude law (js2 = c*je, constant ratio); b<1 =>
  superlinearity concentrated in small-baseline tags.
T2c: family re-test in scale-free space: outcome =
  log js_ratio2; candidates = the 10 frozen 3031
  candidates + log_je (11 total), Spearman + exact MC
  permutation maxT family correction (200k, seed
  30332), alpha 0.05.

Verdict map (frozen, evaluated in this order):
  anchor fail => anchor_fail_all_void
  (p_t2a < .05 and rho_t2a < 0) or ci_hi < 1
                              => hetero_scale_artifact_qwen
  (p_t2a < .05 and rho_t2a > 0) or ci_lo > 1
    or T2c eligible non-empty => hetero_shape_genuine_qwen
  ci contains 1 and p_t2a >= .05 and no eligible
    and R2_loglog >= 0.5      => hetero_amplitude_law_qwen
  else                        => hetero_idiosyncratic_
                                    persists_qwen

Anchors (all CPU, cross-artifact):
  a0  source npz integrity vs their own seal.json
      (3028/3029/3022/3020/3031)
  a1  3028 internal identity: rel_dev == (js2 - pred2)
      / max(pred2, 1e-30) (bit-level)
  a2  |3028 js_dose[:,-1] - 3029 js_alpha2| <= 1e-12
  a3  |3028 js_erase - 3029 js_erase| <= 1e-12
  a4  |3028 js_dose[:,0] - 3029 js_alpha0| <= 1e-12
  a5  3031 npz js_ratio2 == recompute js[:,-1]/je
      (bit-level)
  a6  ratio identity: |js_ratio2 - 2*(1+rel_dev)|
      (bit-level; holds iff pred2 == 2*je exactly)
  a7  3022 conc recompute med vs 0.8236 <= 5e-5
  a8  tags identity across 3028/3029/3022/3020/3031
  a9  |med(rel_dev) - 0.886534| <= 5e-5

NOTE (registered): 3028 pred2 == 2*js_erase exactly
(js0 term absent, 3031 correction); gamma_12 and
js_ratio2 are exact monotone transforms of the stored
rel_dev, so T2a/T2b test the SHAPE of the stored
(authoritative) quantity, not a re-derived one.
"""
import hashlib
import io
import json
import os
import time

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
BASE = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
D28 = BASE + (r'\phase3028\omega_p2v_dose_symmetry'
              r'_qwen')
D29 = BASE + (r'\phase3029\omega_p2w_recruitment'
              r'_decomp_qwen')
D22 = BASE + (r'\phase3022\omega_p2p_l3_relay_neurons'
              r'_qwen')
D20 = BASE + (r'\phase3020\omega_p2n_readout_'
              r'specificity_qwen')
D31 = BASE + (r'\phase3031\omega_p2y_tag_'
              r'heterogeneity_qwen')
PHASE = 3033
NAME = 'omega_p30_logscale_heterogeneity_qwen'
OUT = BASE + r'\phase3033' + r'\omega_p30_logscale_' \
    r'heterogeneity_qwen'
NPERM = 200000
SEED_A = 3033
SEED_B = 30331
SEED_C = 30332
ALPHA_FAM = 0.05
ALPHA_PRIM = 0.05

V_SCALE = 'hetero_scale_artifact_qwen'
V_SHAPE = 'hetero_shape_genuine_qwen'
V_AMP = 'hetero_amplitude_law_qwen'
V_PERSIST = 'hetero_idiosyncratic_persists_qwen'

CAND_ORDER = ['erase_mag', 'restore_asym',
              'coal_share', 'coal_jac', 'mag2',
              'recruit_rho', 'growth_g', 'pos_frac',
              'band_mid', 'readout_lstar',
              'log_je']

PREREG = {
    'mode': 'reanalysis of sealed artifacts (3028/'
            '3029/3022/3020/3031 npz), CPU only, no '
            'model run; source integrity re-verified '
            'by a0 vs each source seal.json',
    'question': 'is the 3028 dose-response convexity '
                'exponent heterogeneity (rel_dev '
                '-0.697..+3.874 across tags) a scale '
                'artifact of ratio parameterization '
                '(small-baseline tags inflate ratios) '
                'or a genuine shape difference?  '
                'log-space reparameterization: '
                'outcome log js_ratio2 (scale-free), '
                'predictor log je (scale)',
    'T2a': 'PRIMARY: gamma_12 = log2(js2/js_erase) '
           '(local log-log dose exponent alpha 1->2, '
           'exact monotone transform of stored '
           'rel_dev); Spearman(gamma_12, log je) '
           'over 11 tags; exact MC permutation null '
           'NPERM=200000 seed 3033; two-sided; '
           'negative+sig => artifact direction',
    'T2b': 'OLS log js2 ~ a + b*log je; pairs '
           'bootstrap (200k resamples, seed 30331) '
           'percentile CI for b; R2; b==1 <=> pure '
           'amplitude law',
    'T2c': 'family re-test in scale-free space: '
           'outcome = log js_ratio2; candidates '
           '(frozen order) = erase_mag, '
           'restore_asym, coal_share, coal_jac, '
           'mag2, recruit_rho, growth_g, pos_frac, '
           'band_mid, readout_lstar, log_je; '
           'Spearman + exact MC permutation maxT '
           'family correction (NPERM=200000, seed '
           '30332), eligible iff p_maxT < 0.05; '
           'zero-variance/nonfinite candidates '
           'dropped with note (family guard)',
    'T3': 'margin caveat: n=11, all p-values '
          'exploratory; gamma_12 depends on stored '
          'rel_dev which depends on the 3028 pred2 '
          'implementation (2*js_erase, correction '
          'registered in 3031); no multivariate '
          'regression (n too small)',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               '(pA<.05 & rhoA<0) or ci_hi<1 => '
               'hetero_scale_artifact_qwen; (pA<.05 '
               '& rhoA>0) or ci_lo>1 or eligible => '
               'hetero_shape_genuine_qwen; ci '
               'contains 1 and pA>=.05 and no '
               'eligible and R2>=0.5 => '
               'hetero_amplitude_law_qwen; else => '
               'hetero_idiosyncratic_persists_qwen',
    'tags': 'Omega-P30 / log-scale reparameterization '
            '/ reanalysis with cross-artifact '
            'anchors / maxT family correction / no '
            'hallucination naming',
}


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20),
                          b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def log(msg, lines):
    lines.append('[%s] %s'
                 % (time.strftime('%H:%M:%S'), msg))
    with open(os.path.join(OUT, 'run_log.txt'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def rank_avg(v):
    v = np.asarray(v, dtype=float)
    order = np.argsort(v, kind='stable')
    ranks = np.empty(v.size, dtype=float)
    sv = v[order]
    i = 0
    while i < v.size:
        j = i
        while j + 1 < v.size and sv[j + 1] == sv[i]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def spearman(x, y):
    rx = rank_avg(x)
    ry = rank_avg(y)
    rc = rx - rx.mean()
    yc = ry - ry.mean()
    denom = np.sqrt((rc * rc).sum()
                    * (yc * yc).sum())
    if denom <= 0:
        return float('nan')
    return float((rc * yc).sum() / denom)


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)
    for fn in ('execution.json', 'result.json',
               'seal.json'):
        p = os.path.join(OUT, fn)
        if os.path.exists(p):
            os.remove(p)
    npz_path = os.path.join(OUT, NAME + '.npz')
    if os.path.exists(npz_path):
        os.remove(npz_path)

    script = os.path.join(
        ROOT, r'tests\glm5',
        'phase%d_%s.py' % (PHASE, NAME))
    with open(os.path.join(OUT, 'execution.json'),
              'w', encoding='utf-8') as f:
        json.dump({'phase': PHASE, 'name': NAME,
                   'created':
                       time.strftime(
                           '%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8': sha8(script),
                   'PREREG': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json written (PREREG frozen)',
        lines)

    # ---------- sources ----------
    z28 = np.load(D28 + r'\omega_p2v_dose_symmetry_'
                  r'qwen.npz', allow_pickle=True)
    z29 = np.load(D29 + r'\omega_p2w_recruitment_'
                  r'decomp_qwen.npz',
                  allow_pickle=True)
    z22 = np.load(D22 + r'\omega_p2p_l3_relay_'
                  r'neurons_qwen.npz',
                  allow_pickle=True)
    z20 = np.load(D20 + r'\omega_p2n_readout_'
                  r'specificity_qwen.npz',
                  allow_pickle=True)
    z31 = np.load(D31 + r'\omega_p2y_tag_'
                  r'heterogeneity_qwen.npz',
                  allow_pickle=True)
    tags = [str(t) for t in z28['tags']]
    n = len(tags)

    # ---------- a0 source integrity ----------
    a0_ok = True
    for d, stem in ((D28, 'omega_p2v_dose_symmetry_'
                     'qwen'),
                    (D29, 'omega_p2w_recruitment_'
                     'decomp_qwen'),
                    (D22, 'omega_p2p_l3_relay_'
                     'neurons_qwen'),
                    (D20, 'omega_p2n_readout_'
                     'specificity_qwen'),
                    (D31, 'omega_p2y_tag_'
                     'heterogeneity_qwen')):
        seal = json.load(io.open(
            d + r'\seal.json', encoding='utf-8'))
        cur = sha8(d + r'\%s.npz' % stem)
        if cur != seal['npz_sha256_8']:
            a0_ok = False
            log('a0 FAIL %s %s vs %s'
                % (stem, cur,
                   seal['npz_sha256_8']), lines)
    log('a0 source integrity ok=%s' % a0_ok, lines)

    # ---------- a1-a4 3028/3029 identities --
    js = z28['js_dose']
    je = z28['js_erase']
    pred2 = z28['pred2']
    rel_dev = z28['rel_dev']
    a1_diff = float(np.max(np.abs(
        rel_dev
        - (js[:, -1] - pred2)
        / np.maximum(pred2, 1e-30))))
    a2_diff = float(np.max(np.abs(
        js[:, -1] - z29['js_alpha2'])))
    a3_diff = float(np.max(np.abs(
        je - z29['js_erase'])))
    a4_diff = float(np.max(np.abs(
        js[:, 0] - z29['js_alpha0'])))
    log('a1 %.3e a2 %.3e a3 %.3e a4 %.3e'
        % (a1_diff, a2_diff, a3_diff, a4_diff),
        lines)

    # ---------- a5/a6 ratio identities -----
    ratio2 = js[:, -1] / np.maximum(je, 1e-30)
    a5_diff = float(np.max(np.abs(
        ratio2 - z31['js_ratio2'])))
    a6_diff = float(np.max(np.abs(
        ratio2 - 2.0 * (1.0 + rel_dev))))
    log('a5 %.3e a6 %.3e' % (a5_diff, a6_diff),
        lines)

    # ---------- a7 3022 conc recompute ------
    s_relay = z22['s_relay'].astype(np.float64)
    TOPK = 32
    conc = []
    own_sets = []
    for i in range(n):
        s = s_relay[i]
        nm = float(-np.sum(np.minimum(s, 0.0)))
        pm = float(np.sum(np.maximum(s, 0.0)))
        if nm >= pm:
            idx = np.argpartition(s, TOPK - 1)[:TOPK]
        else:
            idx = np.argpartition(-s,
                                  TOPK - 1)[:TOPK]
        own_sets.append(set(int(u) for u in idx))
        conc.append(float(np.abs(s[idx]).sum())
                    / max(nm + pm, 1e-30))
    conc = np.array(conc)
    conc_med = float(np.median(conc))
    a7_diff = abs(conc_med - 0.8236)
    log('a7 conc med %.6f diff %.3e'
        % (conc_med, a7_diff), lines)

    # ---------- a8 tags identity -----------
    tags29 = [str(t) for t in z29['tags']]
    tags22 = [str(t) for t in z22['tags']]
    tags20 = [str(t) for t in z20['tags']]
    tags31 = [str(t) for t in z31['tags']]
    keys20 = [str(t)
              for t in z20['traj_keys_logic']]
    a8_ok = bool(tags == tags29 == tags22
                 == tags20 == tags31
                 and set(keys20) == set(tags))
    log('a8 tags identity ok=%s' % a8_ok, lines)

    # ---------- a9 med rel_dev -------------
    a9_diff = abs(float(np.median(rel_dev))
                  - 0.886534)
    log('a9 %.3e' % a9_diff, lines)

    # ---------- 3031 machinery features ----
    sp = s_relay.sum(axis=0)
    top32_pooled = np.argpartition(sp,
                                   -TOPK)[-TOPK:]
    pooled_set = set(int(u)
                     for u in top32_pooled)
    coal_share = np.array([
        float(s_relay[i][top32_pooled].sum())
        / max(float(np.abs(s_relay[i]).sum()),
              1e-30) for i in range(n)])
    coal_jac = np.array([
        float(len(own_sets[i] & pooled_set))
        / float(len(own_sets[i] | pooled_set))
        for i in range(n)])

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        ROOT + r'\models\hf\qwen3-4b')
    prompts = [str(p) for p in z28['prompts']]
    pos_frac = []
    for t in tags:
        pidx = int(t.split(':')[0][1:])
        pos = int(t.split(':')[1])
        L = len(tok.encode(prompts[pidx]))
        pos_frac.append(pos / max(L - 1, 1))
    pos_frac = np.array(pos_frac)

    layers20 = z20['layers']
    idx20 = [keys20.index(t) for t in tags20]
    traj = z20['traj_logic'][idx20, :]
    lstar = np.array(
        [int(layers20[int(np.argmax(traj[i]))])
         for i in range(n)], dtype=float)

    lje = np.log(np.maximum(je, 1e-30))
    cands = {
        'erase_mag': je.copy(),
        'restore_asym': js[:, 0].copy(),
        'coal_share': coal_share,
        'coal_jac': coal_jac,
        'mag2': z28['mag2'].copy(),
        'recruit_rho': z29['rho'].copy(),
        'growth_g': z29['g_exp'].copy(),
        'pos_frac': pos_frac,
        'band_mid': z29['band_share_mid'].copy(),
        'readout_lstar': lstar,
        'log_je': lje,
    }

    dropped = []
    for k in CAND_ORDER:
        v = cands[k]
        if not np.all(np.isfinite(v)):
            dropped.append(k + ':nonfinite')
        elif float(np.std(v)) <= 0:
            dropped.append(k + ':zero_var')
    family = [k for k in CAND_ORDER
              if k not in [d.split(':')[0]
                           for d in dropped]]
    log('family n=%d dropped=%s'
        % (len(family), str(dropped)), lines)

    # ---------- T2a PRIMARY ----------
    lg_ratio = np.log(ratio2)
    gamma12 = lg_ratio / np.log(2.0)
    rho_a = spearman(gamma12, lje)
    rx_a = rank_avg(gamma12)
    rx_a = rx_a - rx_a.mean()
    rx_a = rx_a / max(float(np.sqrt(
        (rx_a * rx_a).sum())), 1e-30)
    ry_a = rank_avg(lje)
    ry_a = ry_a - ry_a.mean()
    ry_a = ry_a / max(float(np.sqrt(
        (ry_a * ry_a).sum())), 1e-30)
    rng_a = np.random.default_rng(SEED_A)
    perms_a = np.argsort(
        rng_a.random((NPERM, n)), axis=1)
    null_a = np.abs(
        (rx_a[None, :]
         * ry_a[perms_a]).sum(axis=1))
    p_a = float((np.sum(null_a >= abs(rho_a)) + 1)
                / (NPERM + 1))
    log('T2a rho(gamma12, log_je)=%+.4f p=%.6f'
        % (rho_a, p_a), lines)

    # ---------- T2b log-log OLS + bootstrap --
    x = lje
    y = np.log(js[:, -1])
    xm = x - x.mean()
    ym = y - y.mean()
    sxx = float((xm * xm).sum())
    beta = float((xm * ym).sum() / max(sxx, 1e-30))
    alpha0 = float(y.mean() - beta * x.mean())
    yhat = alpha0 + beta * x
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float((ym * ym).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    rng_b = np.random.default_rng(SEED_B)
    bs = np.empty(NPERM)
    XY = np.stack([x, y])
    for b in range(NPERM):
        idx = rng_b.integers(0, n, n)
        xb = XY[0, idx]
        yb = XY[1, idx]
        xbm = xb - xb.mean()
        ybm = yb - yb.mean()
        sxxb = float((xbm * xbm).sum())
        if sxxb <= 0:
            bs[b] = np.nan
            continue
        bs[b] = float((xbm * ybm).sum() / sxxb)
    bs = bs[np.isfinite(bs)]
    ci_lo = float(np.percentile(bs, 2.5))
    ci_hi = float(np.percentile(bs, 97.5))
    log('T2b beta=%.4f CI=[%.4f, %.4f] R2=%.4f'
        % (beta, ci_lo, ci_hi, r2), lines)

    # gamma_05 descriptive (alpha 0.5 -> 1)
    js05 = js[:, 1]
    gamma05 = np.log(np.maximum(
        je, 1e-30) / np.maximum(js05, 1e-30)) \
        / np.log(2.0)
    log('gamma05 med %.4f gamma12 med %.4f'
        % (float(np.median(gamma05)),
           float(np.median(gamma12))), lines)

    # ---------- T2c family in log space ----
    y_log = lg_ratio.copy()
    rx = {}
    for k in family:
        r = rank_avg(cands[k])
        r = r - r.mean()
        rx[k] = r / max(
            float(np.sqrt((r * r).sum())), 1e-30)
    ry = rank_avg(y_log)
    yc = ry - ry.mean()
    yc_n = yc / max(
        float(np.sqrt((yc * yc).sum())), 1e-30)
    rho_c = {k: float(rx[k] @ yc_n)
             for k in family}
    rng_c = np.random.default_rng(SEED_C)
    perms_c = np.argsort(
        rng_c.random((NPERM, n)), axis=1)
    ry_perm = yc_n[perms_c]
    R = np.stack([rx[k] for k in family])
    null_c = np.abs(R @ ry_perm.T)
    max_null = null_c.max(axis=0)
    p_maxT = {}
    for ci, k in enumerate(family):
        obs = abs(rho_c[k])
        p_maxT[k] = float(
            (np.sum(max_null >= obs) + 1)
            / (NPERM + 1))
    eligible = [k for k in family
                if p_maxT[k] < ALPHA_FAM]
    for k in family:
        log('cand %-13s rho=%+.4f p_maxT=%.4f'
            % (k, rho_c[k], p_maxT[k]), lines)
    log('eligible=%s' % str(eligible), lines)

    # ---------- verdict tree ----------
    verdict = None
    winner = None
    if (p_a < ALPHA_PRIM and rho_a < 0) \
            or ci_hi < 1.0:
        verdict = V_SCALE
    elif (p_a < ALPHA_PRIM and rho_a > 0) \
            or ci_lo > 1.0 or eligible:
        verdict = V_SHAPE
        if eligible:
            winner = max(eligible,
                         key=lambda k:
                             abs(rho_c[k]))
    elif (ci_lo <= 1.0 <= ci_hi
          and p_a >= ALPHA_PRIM
          and not eligible and r2 >= 0.5):
        verdict = V_AMP
    else:
        verdict = V_PERSIST

    # ---------- per-tag table ----------
    for i in range(n):
        log('tag %s je=%.5f js2=%.5f ratio=%.3f '
            'gamma12=%+.3f gamma05=%+.3f '
            'rel_dev=%+.4f'
            % (tags[i], je[i], js[:, -1][i],
               ratio2[i], gamma12[i], gamma05[i],
               rel_dev[i]), lines)

    anchor_prelim = bool(
        a0_ok and a1_diff <= 1e-12
        and a2_diff <= 1e-12 and a3_diff <= 1e-12
        and a4_diff <= 1e-12 and a5_diff <= 1e-12
        and a6_diff <= 1e-12 and a7_diff <= 5e-5
        and a8_ok and a9_diff <= 5e-5)
    log('ANCHORS %s' % anchor_prelim, lines)
    if not anchor_prelim:
        verdict = 'anchor_fail_all_void'
        winner = None
    log('VERDICT %s' % verdict, lines)

    elapsed = time.monotonic() - t0
    anchors = {
        'a0_source_integrity': a0_ok,
        'a1_reldev_identity': a1_diff,
        'a2_dose_js2_cross': a2_diff,
        'a3_erase_cross': a3_diff,
        'a4_alpha0_cross': a4_diff,
        'a5_ratio2_vs_3031': a5_diff,
        'a6_ratio_reldev_identity': a6_diff,
        'a7_conc_med_diff': a7_diff,
        'a8_tags_identity': a8_ok,
        'a9_med_reldev_diff': a9_diff,
    }
    res = {
        'phase': PHASE,
        'final_verdict': verdict,
        'anchor_all_ok': bool(anchor_prelim),
        'anchors': anchors,
        'T2a': {
            'rho_gamma12_logje':
                round(rho_a, 6),
            'p_perm': round(p_a, 6),
            'nperm': NPERM, 'seed': SEED_A,
            'med_gamma12':
                round(float(np.median(gamma12)), 6),
            'med_gamma05':
                round(float(np.median(gamma05)), 6),
            'alpha_prim': ALPHA_PRIM,
        },
        'T2b': {
            'beta': round(beta, 6),
            'ci_lo': round(ci_lo, 6),
            'ci_hi': round(ci_hi, 6),
            'r2_loglog': round(r2, 6),
            'nboot': NPERM, 'seed': SEED_B,
            'mad_log_ratio': round(float(
                np.median(np.abs(
                    lg_ratio
                    - np.median(lg_ratio)))), 6),
            'mad_rel_dev': round(float(
                np.median(np.abs(
                    rel_dev
                    - np.median(rel_dev)))), 6),
            'med_log_ratio': round(float(
                np.median(lg_ratio)), 6),
        },
        'T2c': {
            'family': family,
            'dropped': dropped,
            'rho_obs': {k: round(v, 6)
                        for k, v in
                        rho_c.items()},
            'p_maxT': {k: round(v, 6)
                       for k, v in
                       p_maxT.items()},
            'eligible': eligible,
            'winner': winner,
            'nperm': NPERM, 'seed': SEED_C,
            'alpha_fam': ALPHA_FAM,
        },
        'T3': PREREG['T3'],
        'tags': PREREG['tags'],
        'elapsed_s': round(elapsed, 1),
        'correction_note': '',
    }
    with open(os.path.join(OUT, 'result.json'),
              'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2,
                  ensure_ascii=False)

    save = {
        'tags': np.array(tags, dtype=object),
        'rel_dev': rel_dev,
        'js_dose': js, 'js_erase': je,
        'js_ratio2': ratio2,
        'log_ratio2': lg_ratio,
        'log_je': lje,
        'gamma12': gamma12, 'gamma05': gamma05,
        'beta_ols': np.float64(beta),
        'ci_lo': np.float64(ci_lo),
        'ci_hi': np.float64(ci_hi),
        'r2_loglog': np.float64(r2),
        'rho_prim': np.float64(rho_a),
        'p_prim': np.float64(p_a),
        'mag2': z28['mag2'],
        'rho': z29['rho'], 'g_exp': z29['g_exp'],
        'band_share_mid': z29['band_share_mid'],
        'coal_share': coal_share,
        'coal_jac': coal_jac,
        'pos_frac': pos_frac,
        'lstar': lstar,
        'conc_per_tag': conc,
        'top32_pooled': top32_pooled,
        'rho_cand': np.array(
            [rho_c.get(k, np.nan)
             for k in CAND_ORDER]),
        'p_cand': np.array(
            [p_maxT.get(k, np.nan)
             for k in CAND_ORDER]),
        'cand_order': np.array(CAND_ORDER,
                               dtype=object),
    }
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
