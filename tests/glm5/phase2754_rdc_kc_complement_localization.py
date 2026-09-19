"""Phase2754: P5 step 2 closure — localize the knowledge_chain complement effect.

CPU-only follow-up to Phase2753 (no model load, no GPU).  Phase2753 found:
the true_token runs' complement (all trained coordinates outside the Phase2751
union subset) is net beneficial overall (+0.149/+0.132 NLL reduction) but
knowledge_chain-specific harmful (-1.10/-2.04), offset by word_sense and
negation_scope gains.  This phase decomposes that per-question structure from
the frozen Phase2753 per-question arrays, and links it to the Phase2750
paired-world stepwise divergence.

Questions (frozen before any Phase2754 statistic):
  B1  Universality: is the kc complement damage distributed across the 64 kc
      questions or driven by a few extremes?  (top-5 share of negative mass)
  B2  Discriminative shift: does the complement move probability mass from the
      true target toward the permuted same-class answer (dual-world criterion,
      Delta margin), or is it generic NLL inflation (Delta margin ~ 0)?
      base margin recovered from base__permuted_own - base__true_own (same
      original weights; Phase2753 evaluated base per condition).
  B3  Family profile of per-question complement effect, both true runs and both
      permuted runs (descriptive).
  B4  Cross-generation reproducibility of per-question kc damage (Spearman
      between true_2747 and true_2748 per-question red_comp on kc).
  B5  Link to Phase2750: Spearman between per-question B0 stepwise divergence
      (m0, true-vs-permuted pairs, matched generation) and per-question
      complement effect / margin shift on the 64 kc questions (descriptive).
  B6  Identification of the most damaged kc questions with their material
      annotation (annotation_scope / component_ids) for MEMO reporting.

Preregistered readouts:
  B1: damaged fraction, top-5 negative-mass share per true run; verdict
      'few_extremes' if a single run's top-5 share > 0.5, else 'distributed'
      (both runs must agree on the label for a firm verdict).
  B2: support if mean Delta margin > 0 in both true runs AND the kc mean Delta
      margin exceeds the panel-wide mean in both runs.
  B4/B5: Spearman rho with permutation p (10000 draws, seeds 2754012/2754014);
      descriptive, no pass/fail.

Storage: direct on D: under BASE (post-migration layout); no junction; no GPU.
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from rdc_construction_common import (BASE, stamp, sha, read, save, snapshot,
                                     guard, ledger, npz)

OUT54 = BASE / 'phase2754'
SRC53 = BASE / 'phase2753/qwen4_block16'
SRC50 = BASE / 'phase2750/qwen4'
RUNS_TRUE = ['true_token_2747', 'true_token_2748']
RUNS_PERM = ['within_surface_class_permuted_token_2747',
             'within_surface_class_permuted_token_2748']
RUNS_MASS = ['surface_class_mass_2747', 'surface_class_mass_2748']
PAIRS_P0 = {'true_token_2747': 0, 'true_token_2748': 1}   # 2750 pair_code of true-vs-perm
BOOT_DRAWS = 2000
BOOT_SEED = 2754010
PERM_DRAWS = 10000
PERM_SEEDS = {'B4': 2754012, 'B5': 2754014}
START = time.monotonic()


def elapsed():
    return round(time.monotonic() - START, 1)


def ranks(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind='mergesort')
    r = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 0.5
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = ranks(x), ranks(y)
    rx = (rx - rx.mean()) / (rx.std() + 1e-30)
    ry = (ry - ry.mean()) / (ry.std() + 1e-30)
    return float((rx * ry).mean())


def perm_p(x, y, seed, draws=PERM_DRAWS):
    obs = spearman(x, y)
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    cnt = 0
    for _ in range(draws):
        cnt += abs(spearman(x, y[rng.permutation(len(y))])) >= abs(obs) - 1e-12
    return obs, (cnt + 1) / (draws + 1)


def main():
    import shutil
    folder = OUT54 / 'qwen4_cpu'
    finish = folder / 'result.json'
    if finish.exists():
        return
    guard(0)
    assert shutil.disk_usage('D:/').free > 1 * 1024 ** 3
    folder.mkdir(parents=True, exist_ok=True)

    # ---- frozen design (before any Phase2754 statistic) ----
    z53_path = SRC53 / 'isolation_scores.npz'
    execution = {'source': snapshot(__file__),
                 'phase2753_isolation_scores_sha256': sha(z53_path),
                 'phase2753_result_sha256': sha(SRC53 / 'result.json'),
                 'phase2750_question_scores_sha256': sha(SRC50 / 'question_scores.npz'),
                 'phase2750_question_scores_meta_sha256': sha(SRC50 / 'question_scores_meta.json'),
                 'storage': 'direct on D: under BASE (post-migration layout); CPU-only',
                 'inputs': 'Phase2753 per-question arrays (896 panel rows, frozen order) + '
                           'Phase2750 per-question paired-world divergence (512 own-history '
                           'questions, joined by sample_id)',
                 'boot': {'draws': BOOT_DRAWS, 'seed': BOOT_SEED},
                 'permutation': {'draws': PERM_DRAWS, 'seeds': PERM_SEEDS},
                 'criteria': {
                     'B1_universality':
                         'kc complement damage label per true run: few_extremes if top-5 '
                         'negative-mass share > 0.5 else distributed; firm verdict needs '
                         'both runs to agree',
                     'B2_discriminative_shift':
                         'support if mean Delta margin (nll_permuted - nll_target relative to '
                         'base) > 0 in both true runs on kc AND kc mean exceeds panel-wide '
                         'mean in both runs',
                     'B4_cross_generation': 'descriptive Spearman of per-question kc red_comp '
                                            'between the two true runs',
                     'B5_phase2750_link': 'descriptive Spearman of kc per-question m0 vs '
                                          'red_comp and vs Delta margin, per generation'},
                 'question': 'Is the knowledge_chain complement damage in the true_token runs '
                             'a distributed discriminative shift toward the permuted answer, '
                             'and does it track the Phase2750 B0 divergence?',
                 'status': 'frozen_before_any_Phase2754_statistic'}
    exec_path = folder / 'execution.json'
    if exec_path.exists():
        old = read(exec_path)
        assert old['execution'] == execution, 'Phase2754 execution drift'
    else:
        save(exec_path, {'timestamp': stamp(), 'execution': execution})

    z = np.load(z53_path, allow_pickle=False)
    meta = read(SRC53 / 'isolation_scores_meta.json')
    sids = list(meta['panel_sample_ids'])
    fams = list(meta['families'])
    assert len(sids) == len(fams) == 896
    fam = np.array(fams)
    kc = fam == 'knowledge_chain'
    assert kc.sum() == 64
    sid_arr = np.array(sids)

    # base margins: base__permuted_own is base nll_permuted, base__true_own is base nll_target
    base_tgt = z['base__true_token__own'].astype(np.float64)
    base_prm = z['base__within_surface_class_permuted_token__own'].astype(np.float64)
    base_margin = base_prm - base_tgt          # >0 = target preferred

    def arr(run, variant, field):
        return z[f'{run}__{variant}__{field}'].astype(np.float64)

    def red_per_q(run, variant):
        cond = run.rsplit('_', 1)[0]
        base_own = z['base__' + cond + '__own'].astype(np.float64)
        return base_own - arr(run, variant, 'own')

    # ---- B1 universality on kc ----
    b1 = {}
    neg_mass_share = {}
    for run in RUNS_TRUE:
        rc = red_per_q(run, 'only_complement')[kc]
        dmg = rc < 0
        neg = np.minimum(rc, 0.0)
        total_neg = float(neg.sum())
        order = np.argsort(neg)          # most negative first
        top5_share = float(neg[order[:5]].sum() / total_neg) if total_neg < 0 else float('nan')
        rng = np.random.default_rng(BOOT_SEED)
        boots = np.array([rc[rng.integers(0, len(rc), len(rc))].mean() for _ in range(BOOT_DRAWS)])
        b1[run] = {'n_questions': int(len(rc)),
                   'mean': float(rc.mean()), 'median': float(np.median(rc)),
                   'iqr': [float(np.percentile(rc, 25)), float(np.percentile(rc, 75))],
                   'damaged_n': int(dmg.sum()), 'damaged_fraction': float(dmg.mean()),
                   'total_negative_mass': total_neg,
                   'top5_negative_mass_share': top5_share,
                   'mean_bootstrap_ci95': [float(np.percentile(boots, 2.5)),
                                           float(np.percentile(boots, 97.5))]}
        neg_mass_share[run] = top5_share
    labels = {'few_extremes' if neg_mass_share[r] > 0.5 else 'distributed' for r in RUNS_TRUE}
    b1_verdict = labels.pop() if len(labels) == 1 else 'mixed'
    b1['verdict'] = b1_verdict

    # ---- B2 discriminative shift ----
    b2 = {}
    for run in RUNS_TRUE:
        m_comp = arr(run, 'only_complement', 'nll_permuted') - arr(run, 'only_complement', 'nll_target')
        m_full = arr(run, 'full_noR', 'nll_permuted') - arr(run, 'full_noR', 'nll_target')
        d_comp = m_comp - base_margin            # >0 = shifted toward permuted
        d_full = m_full - base_margin
        b2[run] = {'kc_mean_dmargin_comp': float(d_comp[kc].mean()),
                   'panel_mean_dmargin_comp': float(d_comp.mean()),
                   'kc_mean_dmargin_full_noR': float(d_full[kc].mean()),
                   'kc_median_dmargin_comp': float(np.median(d_comp[kc])),
                   'kc_positive_fraction_comp': float((d_comp[kc] > 0).mean()),
                   'kc_dmargin_comp_by_family': {f: float(d_comp[fam == f].mean())
                                                 for f in sorted(set(fams))}}
    b2_ok = all(b2[r]['kc_mean_dmargin_comp'] > 0 and
                b2[r]['kc_mean_dmargin_comp'] > b2[r]['panel_mean_dmargin_comp'] for r in RUNS_TRUE)
    b2['verdict'] = 'supported' if b2_ok else 'not_supported'

    # ---- B3 family profile of per-question complement effect ----
    b3 = {}
    for run in RUNS_TRUE + RUNS_PERM:
        rc = red_per_q(run, 'only_complement')
        b3[run] = {f: {'mean': float(rc[fam == f].mean()),
                       'damaged_fraction': float((rc[fam == f] < 0).mean())}
                   for f in sorted(set(fams))}

    # ---- B4 cross-generation reproducibility of kc damage ----
    rc47 = red_per_q('true_token_2747', 'only_complement')[kc]
    rc48 = red_per_q('true_token_2748', 'only_complement')[kc]
    rho_b4, p_b4 = perm_p(rc47, rc48, PERM_SEEDS['B4'])
    b4 = {'spearman_red_comp_47_vs_48': rho_b4, 'permutation_p': p_b4,
          'mean_2747': float(rc47.mean()), 'mean_2748': float(rc48.mean())}

    # ---- B5 link to Phase2750 ----
    z50 = np.load(SRC50 / 'question_scores.npz', allow_pickle=False)
    m50 = read(SRC50 / 'question_scores_meta.json')
    ids50 = np.array(m50['sample_ids'])
    pair_code = z50['pair_code']
    fam_code = z50['family_code']
    sid_idx = z50['sample_id_idx']
    m0 = z50['m0']
    fam_name = {v: k for k, v in m50['family_code'].items()}
    kc_code = [int(k) for k, v in m50['family_code'].items() if v == 'knowledge_chain'][0]
    pos_of_sid = {s: i for i, s in enumerate(sids)}
    b5 = {}
    for run in RUNS_TRUE:
        pc = PAIRS_P0[run]
        sel = (pair_code == pc) & (fam_code == kc_code)
        assert sel.sum() == 64, (run, int(sel.sum()))
        sids50 = ids50[sid_idx[sel]]
        m0_kc = m0[sel]
        ok = np.isfinite(m0_kc)
        # join by sample_id (panel order)
        idx_panel = np.array([pos_of_sid[s] for s in sids50])
        d_comp_panel = (arr(run, 'only_complement', 'nll_permuted') -
                        arr(run, 'only_complement', 'nll_target') - base_margin)[idx_panel]
        rc_panel = red_per_q(run, 'only_complement')[idx_panel]
        rho_rc, p_rc = perm_p(m0_kc[ok], rc_panel[ok], PERM_SEEDS['B5'])
        rho_dm, p_dm = perm_p(m0_kc[ok], d_comp_panel[ok], PERM_SEEDS['B5'] + 1)
        b5[run] = {'n_finite_m0': int(ok.sum()),
                   'spearman_m0_vs_red_comp': rho_rc, 'p_m0_vs_red_comp': p_rc,
                   'spearman_m0_vs_dmargin': rho_dm, 'p_m0_vs_dmargin': p_dm,
                   'm0_median': float(np.median(m0_kc[ok]))}

    # ---- B6 most damaged kc questions + material annotation ----
    import phase2747_rdc_material as mat2747
    _, data = mat2747.freeze()
    row_by_sid = {}
    for part in ('validation', 'diagnostic', 'fresh'):
        for r in data[part]:
            row_by_sid[r['sample_id']] = r
    b6 = []
    for run in RUNS_TRUE:
        idx = np.where(kc)[0]
        rc_all = red_per_q(run, 'only_complement')
        d_all = (arr(run, 'only_complement', 'nll_permuted') -
                 arr(run, 'only_complement', 'nll_target') - base_margin)
        m0_by_panel = {}
        pc = PAIRS_P0[run]
        sel = (pair_code == pc) & (fam_code == kc_code)
        for s, v in zip(ids50[sid_idx[sel]], m0[sel]):
            m0_by_panel[s] = float(v)
        worst = idx[np.argsort(rc_all[idx])[:10]]
        for i in worst:
            r = row_by_sid.get(sids[i], {})
            mv = m0_by_panel.get(sids[i])
            b6.append({'run': run, 'sample_id': sids[i],
                       'red_comp': float(rc_all[i]), 'dmargin_comp': float(d_all[i]),
                       'm0_2750': None if mv is None or not np.isfinite(mv) else float(mv),
                       'source_group': r.get('source_group'),
                       'annotation_scope': r.get('annotation_scope')})

    value = {'timestamp': stamp(), 'source': snapshot(__file__),
             'status': 'P5 step 2 closure complete (CPU-only)',
             'model': 'qwen4 (no model load; Phase2753 frozen arrays)',
             'panel_n': 896, 'kc_n': 64,
             'B1_universality': b1, 'B2_discriminative_shift': b2,
             'B3_family_profile': b3, 'B4_cross_generation': b4,
             'B5_phase2750_link': b5, 'B6_top_damaged': b6,
             'base_margin_consistency': {
                 'base_own_true_vs_nll_target_identity_note':
                     'base__true_token__own equals base nll_target; '
                     'base__within_surface_class_permuted_token__own equals base nll_permuted '
                     '(same original weights; Phase2753 evaluated base once per condition)'},
             'seconds': elapsed(),
             'limits': ['Phase2753 arrays carry own/nll_target/nll_permuted only; no per-question '
                        'argmax, so the dual-world criterion is the log-probability margin, not '
                        'argmax flips.',
                        'kc n=64; bootstrap/permutation CIs are correspondingly wide.',
                        'The complement is defined by the Phase2751 union (1,415,316 coordinates); '
                        'its internal structure is not resolved here.',
                        'The 2750 join covers only questions whose generation produced a '
                        'comparable same-prefix prefix (m0 finite).']}
    arrays = {'kc_red_comp_' + r: red_per_q(r, 'only_complement')[kc]
              for r in RUNS_TRUE + RUNS_PERM}
    for run in RUNS_TRUE + RUNS_PERM:
        d = (arr(run, 'only_complement', 'nll_permuted') -
             arr(run, 'only_complement', 'nll_target') - base_margin)
        arrays['kc_dmargin_comp_' + run] = d[kc]
        arrays['panel_red_comp_' + run] = red_per_q(run, 'only_complement')
        if run in PAIRS_P0:
            pc = PAIRS_P0[run]
            sel50 = (pair_code == pc) & (fam_code == kc_code)
            m0_map = {s: float(v) for s, v in zip(ids50[sid_idx[sel50]], m0[sel50])}
            arrays['kc_m0_' + run] = np.array([m0_map.get(s, np.nan) for s in sid_arr[kc]])
    arrays['kc_sample_ids'] = sid_arr[kc].astype(np.str_)
    npz(folder / 'kc_localization.npz', **arrays)
    save(folder / 'kc_localization_meta.json',
         {'timestamp': stamp(), 'kc_sample_ids': [sids[i] for i in np.where(kc)[0]],
          'runs_true': RUNS_TRUE, 'runs_perm': RUNS_PERM})
    save(finish, value)
    ledger('phase2754_kc_complement_localization', value['seconds'])
    print('PHASE2754_KC_LOCALIZATION', json.dumps(
        {'B1_verdict': b1['verdict'], 'B2': b2['verdict'],
         'B4_rho': b4['spearman_red_comp_47_vs_48'],
         'B5': {r: {k: v for k, v in d.items() if k.startswith('spearman')}
                for r, d in b5.items()}}, ensure_ascii=False), elapsed(), flush=True)


if __name__ == '__main__':
    main()
