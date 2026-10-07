# -*- coding: utf-8 -*-
"""Phase 3031: Omega-P2y - per-tag heterogeneity source
(reanalysis phase, CPU only, no model run).

Question (3028 follow-up A): the dose-response convexity
metric rel_dev spans -0.697..+3.874 across the 11 logic
tags.  WHAT determines the tag-to-tag difference:
position context, coalition composition, erasure
magnitude, midband amplification, recruitment share,
growth exponent, or readout peak?

Design: pure reanalysis of sealed artifacts
(3028 dose npz = outcome rel_dev + magnitude/mag2;
 3029 recruitment npz = rho/g_exp/band shares;
 3022 relay npz = s_relay -> coalition features;
 3020 readout npz = lens trajectory -> lstar;
 tag names P{i}:{pos} + tokenizer -> position context).

PRIMARY: Spearman rho per candidate vs rel_dev, exact
Monte-Carlo permutation null (200k draws, seed 3031),
maxT family correction over the 10 preregistered
candidates (n=11, exploratory margin registered).

Anchors (all CPU, cross-artifact):
  a0  source npz integrity vs their own seal.json
  a1  3028 internal identity: rel_dev == (js2 - pred2) /
      max(pred2, 1e-30) from stored arrays (<= 1e-12)
  a2  |3028 js_dose[:,3] - 3029 js_alpha2| <= 1e-12
  a3  |3028 js_erase - 3029 js_erase| <= 1e-12
  a4  |3028 js_dose[:,0] - 3029 js_alpha0| <= 1e-12
  a5  3020 traj (key-aligned) last col vs
      js_final_logic <= 1e-6
  a6  3022 conc recompute med vs 0.8236 <= 5e-5
  a7  3029 g_exp recompute (U1 cap U2 median log2) and
      rho recompute vs stored, both <= 1e-12
  a8  tags identity across 3028/3029/3022/3020 (+ set
      equality with 3020 traj_keys_logic)
  a9  |med(rel_dev) - 0.8865| <= 5e-5

CORRECTION registered (3028): stored pred2 == 2*js_erase
exactly (js0 term absent; design text said 2*js1 - js0).
Recomputing with the alpha=0 arm gives med rel_dev 1.315
> 0.15 -> 3028 verdict superlinear unchanged.  3031
outcome uses the stored rel_dev (authoritative, matches
the sealed verdict).

Verdict map (frozen):
  anchor fail                    => anchor_fail_all_void
  all candidates degenerate      => hetero_family_
                                    degenerate_void
  no candidate p_maxT < 0.05     => hetero_
                                    idiosyncratic_qwen
  winner = max |rho| among eligible, mapped:
  erase_mag    => hetero_by_erase_magnitude_qwen
  restore_asym => hetero_by_restore_asymmetry_qwen
  coal_share   => hetero_by_coalition_share_qwen
  coal_jac     => hetero_by_coalition_identity_qwen
  mag2         => hetero_by_midband_amplification_qwen
  recruit_rho  => hetero_by_recruitment_share_qwen
  growth_g     => hetero_by_growth_exponent_qwen
  pos_frac     => hetero_by_position_context_qwen
  band_mid     => hetero_by_midband_energy_qwen
  readout_lstar=> hetero_by_readout_peak_qwen
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
PHASE = 3031
NAME = 'omega_p2y_tag_heterogeneity_qwen'
OUT = BASE + r'\phase3031' + r'\omega_p2y_tag_' \
    r'heterogeneity_qwen'
NPERM = 200000
SEED = 3031
ALPHA_FAM = 0.05

PREREG = {
    'mode': 'reanalysis of sealed artifacts (3028/3029/'
            '3022/3020 npz), CPU only, no model run; '
            'source integrity re-verified by a0 vs '
            'each source seal.json',
    'question': '3028 rel_dev spans -0.697..+3.874 '
                'across tags - which preregistered '
                'feature (erase magnitude / restore '
                'asymmetry / coalition share / coalition '
                'identity / midband amplification / '
                'recruitment share / growth exponent / '
                'position context / midband energy / '
                'readout peak) drives the tag-to-tag '
                'heterogeneity of the convexity '
                'response?',
    'T1': 'sources: 3028 npz (js_erase, js_dose, '
          'pred2, rel_dev, mag2), 3029 npz (rho, '
          'g_exp, band_share_mid, d1/d2/theta), '
          '3022 npz (s_relay), 3020 npz (traj_logic '
          '+ traj_keys_logic + layers + '
          'js_final_logic), tag names P{i}:{pos} + '
          'tokenizer (models/hf/qwen3-4b) for '
          'position context; anchors a0-a9 as in '
          'docstring',
    'T2a': 'PRIMARY: outcome = stored 3028 rel_dev '
           '(authoritative, matches sealed verdict); '
           'candidates (frozen order) = erase_mag, '
           'restore_asym, coal_share, coal_jac, mag2, '
           'recruit_rho, growth_g, pos_frac, band_mid, '
           'readout_lstar; Spearman rho (average-rank '
           'ties); exact MC permutation null on the '
           'outcome vector, NPERM=200000, seed 3031; '
           'maxT family correction; eligible iff '
           'p_maxT < 0.05; winner = max |rho_obs| '
           'among eligible; prereg primary = '
           'erase_mag (3028 noted tag-level magnitude '
           'heterogeneity); zero-variance candidates '
           'dropped with note (family guard)',
    'T2b': 'DESCRIPTIVE: 10x10 Spearman matrix among '
           'candidates; winner rho vs secondary '
           'outcome js_ratio2 = js_dose[:,3]/'
           'js_erase; per-tag table in run_log',
    'T3': 'margin caveat: n=11, all p-values '
          'exploratory; no multivariate regression '
          '(n too small); P8 absent from tags (11 of '
          '12 prompts)',
    'corrections': '3028 pred2 implementation '
                   'deviation: stored pred2 == '
                   '2*js_erase exactly (js0 term '
                   'absent vs design text 2*js1-js0); '
                   'recompute with the alpha=0 arm '
                   'gives med rel_dev 1.315 > 0.15 -> '
                   '3028 verdict superlinear '
                   'unchanged; 3031 outcome = stored '
                   'rel_dev (authoritative)',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'all candidates degenerate => '
               'hetero_family_degenerate_void; no '
               'eligible candidate => '
               'hetero_idiosyncratic_qwen; else '
               'winner-mapped hetero_by_*_qwen (see '
               'docstring map)',
    'tags': 'Omega-P2y / per-tag heterogeneity source '
            '/ reanalysis with cross-artifact anchors '
            '/ maxT family correction / no '
            'hallucination naming',
}

VERDICT_MAP = {
    'erase_mag': 'hetero_by_erase_magnitude_qwen',
    'restore_asym':
        'hetero_by_restore_asymmetry_qwen',
    'coal_share': 'hetero_by_coalition_share_qwen',
    'coal_jac': 'hetero_by_coalition_identity_qwen',
    'mag2': 'hetero_by_midband_amplification_qwen',
    'recruit_rho':
        'hetero_by_recruitment_share_qwen',
    'growth_g': 'hetero_by_growth_exponent_qwen',
    'pos_frac': 'hetero_by_position_context_qwen',
    'band_mid': 'hetero_by_midband_energy_qwen',
    'readout_lstar': 'hetero_by_readout_peak_qwen',
}
CAND_ORDER = ['erase_mag', 'restore_asym',
              'coal_share', 'coal_jac', 'mag2',
              'recruit_rho', 'growth_g', 'pos_frac',
              'band_mid', 'readout_lstar']


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
                     'specificity_qwen')):
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

    # ---------- a5 3020 traj alignment ------
    keys20 = [str(t)
              for t in z20['traj_keys_logic']]
    tags20 = [str(t) for t in z20['tags']]
    idx20 = [keys20.index(t) for t in tags20]
    traj = z20['traj_logic'][idx20, :]
    a5_diff = float(np.max(np.abs(
        traj[:, -1] - z20['js_final_logic'])))
    layers20 = z20['layers']
    lstar = np.array(
        [int(layers20[int(np.argmax(traj[i]))])
         for i in range(n)], dtype=float)
    log('a5 %.3e lstar=%s'
        % (a5_diff, str(lstar.astype(int))), lines)

    # ---------- a6 3022 conc recompute ------
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
    a6_diff = abs(conc_med - 0.8236)
    log('a6 conc med %.6f diff %.3e'
        % (conc_med, a6_diff), lines)

    # ---------- a7 3029 g/rho recompute ----
    d1 = z29['d1']
    d2 = z29['d2']
    theta = float(z29['theta'])
    g_rec = []
    rho_rec = []
    for i in range(n):
        m1 = d1[i] > theta
        m2 = d2[i] > theta
        mi = m1 & m2
        g_rec.append(float(np.median(
            np.log2(d2[i][mi] / d1[i][mi]))))
        new = m2 & (~m1)
        denom = float((d2[i][m2] ** 2).sum())
        rho_rec.append(float(
            (d2[i][new] ** 2).sum()
            / max(denom, 1e-30)))
    g_rec = np.array(g_rec)
    rho_rec = np.array(rho_rec)
    a7_g = float(np.max(np.abs(
        g_rec - z29['g_exp'])))
    a7_r = float(np.max(np.abs(
        rho_rec - z29['rho'])))
    log('a7 g %.3e rho %.3e' % (a7_g, a7_r), lines)

    # ---------- a8 tags identity -----------
    tags29 = [str(t) for t in z29['tags']]
    tags22 = [str(t) for t in z22['tags']]
    a8_ok = bool(tags == tags29 == tags22
                 == tags20
                 and set(keys20) == set(tags))
    log('a8 tags identity ok=%s' % a8_ok, lines)

    # ---------- a9 med rel_dev vs recorded -
    a9_diff = abs(float(np.median(rel_dev))
                  - 0.8865)
    log('a9 %.3e' % a9_diff, lines)

    anchor_prelim = bool(
        a0_ok and a1_diff <= 1e-12
        and a2_diff <= 1e-12 and a3_diff <= 1e-12
        and a4_diff <= 1e-12 and a5_diff <= 1e-6
        and a6_diff <= 5e-5 and a7_g <= 1e-12
        and a7_r <= 1e-12 and a8_ok
        and a9_diff <= 5e-5)
    log('ANCHORS %s' % anchor_prelim, lines)

    # ---------- candidates ----------
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

    # ---------- T2a stats ----------
    T2b = {'note': 'family empty'}
    if not family:
        verdict = 'hetero_family_degenerate_void'
        rho_obs = {}
        p_maxT = {}
        T2a = {'family': [], 'dropped': dropped,
               'note': 'all candidates degenerate'}
    else:
        y = rel_dev.copy()
        rx = {}
        for k in family:
            r = rank_avg(cands[k])
            r = r - r.mean()
            rx[k] = r / max(
                float(np.sqrt((r * r).sum())),
                1e-30)
        ry = rank_avg(y)
        yc = ry - ry.mean()
        yc_n = yc / max(
            float(np.sqrt((yc * yc).sum())),
            1e-30)
        rho_obs = {k: float(rx[k] @ yc_n)
                   for k in family}
        rng = np.random.default_rng(SEED)
        perms = np.argsort(
            rng.random((NPERM, n)), axis=1)
        ry_perm = yc_n[perms]
        R = np.stack([rx[k] for k in family])
        null = np.abs(R @ ry_perm.T)
        max_null = null.max(axis=0)
        p_maxT = {}
        for ci, k in enumerate(family):
            obs = abs(rho_obs[k])
            p_maxT[k] = float(
                (np.sum(max_null >= obs) + 1)
                / (NPERM + 1))
        for k in family:
            log('cand %-13s rho=%+.4f p_maxT=%.4f'
                % (k, rho_obs[k], p_maxT[k]),
                lines)
        eligible = [k for k in family
                    if p_maxT[k] < ALPHA_FAM]
        if not eligible:
            verdict = 'hetero_idiosyncratic_qwen'
        else:
            winner = max(eligible,
                         key=lambda k:
                             abs(rho_obs[k]))
            verdict = VERDICT_MAP[winner]
        T2a = {
            'family': family,
            'dropped': dropped,
            'rho_obs': {k: round(v, 6)
                        for k, v in
                        rho_obs.items()},
            'p_maxT': {k: round(v, 6)
                       for k, v in
                       p_maxT.items()},
            'eligible': eligible,
            'winner': (verdict
                       if verdict.startswith(
                           'hetero_by_')
                       else None),
            'prereg_primary':
                'erase_mag' if 'erase_mag'
                in rho_obs else 'dropped',
            'primary_confirmed':
                bool('erase_mag' in eligible),
            'nperm': NPERM, 'seed': SEED,
            'alpha_fam': ALPHA_FAM,
            'med_rel_dev':
                round(float(np.median(rel_dev)),
                      6),
        }

        # ---------- T2b descriptive ----------
        names = family + ['outcome',
                          'js_ratio2']
        allv = {k: cands[k] for k in family}
        allv['outcome'] = rel_dev
        allv['js_ratio2'] = (js[:, -1]
                             / np.maximum(je,
                                          1e-30))
        M = np.zeros((len(names), len(names)))
        for a in range(len(names)):
            for b0 in range(len(names)):
                M[a, b0] = spearman(
                    allv[names[a]],
                    allv[names[b0]])
        T2b = {
            'names': names,
            'spearman_matrix':
                [[round(float(v), 4)
                  for v in row] for row in M],
            'winner_vs_js_ratio2':
                round(float(spearman(
                    allv[T2a['winner']]
                    if T2a['winner']
                    else allv[family[0]],
                    allv['js_ratio2'])), 6)
            if T2a['winner'] else None,
            'med_js_ratio2': round(float(
                np.median(
                    allv['js_ratio2'])), 4),
        }
        for i in range(n):
            log('tag %s rel_dev=%+.4f '
                'erase=%.5f coalshare=%.3f '
                'jac=%.3f mag2=%.3f rho=%.3f '
                'g=%.3f pos=%.3f bandmid=%.3f '
                'lstar=%d'
                % (tags[i], rel_dev[i], je[i],
                   coal_share[i], coal_jac[i],
                   cands['mag2'][i],
                   cands['recruit_rho'][i],
                   cands['growth_g'][i],
                   pos_frac[i],
                   cands['band_mid'][i],
                   int(lstar[i])), lines)

    if not anchor_prelim:
        verdict = 'anchor_fail_all_void'
    log('VERDICT %s' % verdict, lines)

    elapsed = time.monotonic() - t0
    anchors = {
        'a0_source_integrity': a0_ok,
        'a1_reldev_identity': a1_diff,
        'a2_dose_js2_cross': a2_diff,
        'a3_erase_cross': a3_diff,
        'a4_alpha0_cross': a4_diff,
        'a5_traj_terminal': a5_diff,
        'a6_conc_med_diff': a6_diff,
        'a7_g_rho_recompute': {'g': a7_g,
                               'rho': a7_r},
        'a8_tags_identity': a8_ok,
        'a9_med_reldev_diff': a9_diff,
    }
    res = {
        'phase': PHASE,
        'final_verdict': verdict,
        'anchor_all_ok': bool(anchor_prelim),
        'anchors': anchors,
        'T2a': T2a, 'T2b': T2b,
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
        'pred2': pred2,
        'js_ratio2': js[:, -1]
        / np.maximum(je, 1e-30),
        'mag2': z28['mag2'],
        'rho': z29['rho'], 'g_exp': z29['g_exp'],
        'band_share_mid': z29['band_share_mid'],
        'coal_share': coal_share,
        'coal_jac': coal_jac,
        'pos_frac': pos_frac,
        'lstar': lstar,
        'conc_per_tag': conc,
        'top32_pooled': top32_pooled,
        'rho_obs': np.array(
            [rho_obs.get(k, np.nan)
             for k in CAND_ORDER]),
        'p_maxT': np.array(
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
