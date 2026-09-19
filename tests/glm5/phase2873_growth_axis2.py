# -*- coding: utf-8 -*-
"""Phase 2873: growth curve point 2 - attribute axis into density-gating protocol.

Zero-forward. Reads only immutable artifacts:
  - phase2872/attr_census.npz : drops (28,36,32) attr-axis causal spectrum,
    target_list (word:axis prefix), axis_names (10)
  - phase2868/growth_v2.npz  : sig_mask (1152,) class-axis 43 significant heads
  - phase2867/word_coords_v1.npz : B1/B2/B3 (80 words) recorded numbers only

Prereg (frozen before any readout; execution.json written before stats):
  Protocol (mirrors 2867/2868 word-coordinate retrieval):
    block  = per-head z-score across words, then L2-normalize each word row
    acc    = leave-one-out nearest-neighbor by cosine, correct iff NN shares
             the axis label (attr words: label = prefix before ':')
    select = per-head one-way eta2 over axis labels, F-test p < 0.01
             (scipy.stats.f.sf, correct tail; 2871 erratum lesson applied)
  Judgements:
    C1 sparse_sufficiency_replicates on axis 2:
       acc(sig2_attr) > acc(full_attr)  AND  acc(sig2_attr) > null_p95
       where null = full-pipeline label permutation (N=200, SEED=2873):
       permute labels -> reselect heads on permuted labels -> re-evaluate
       acc on permuted labels (selection bias mirrored into null)
    C2 cross-axis head overlap:
       |sig2_attr AND class43| vs hypergeometric (population 1152,
       draws |sig2_attr|, successes 43); shared_core iff p < 0.05,
       else axis_specific.  Growth datum: n_new = |sig2_attr \\ class43|.
    C3 transfer of class-axis subset to axis 2:
       acc(class43 on attr words) vs acc(full on attr words);
       reuse_signal iff acc43 > acc_full, else novel_dominant.
    growth point recorded: axis2 components = |sig2_attr|,
       new heads = n_new, plus acc ladder (full / sig2 / class43).
  Degenerate guard: if |sig2_attr| == 0 -> C1 = no_signal (recorded, not fatal).
  Negative-denominator guards: all shares/ratios use abs-threshold > 1e-30.

Determinism: no RNG except np.random.default_rng(2873) for permutations.
"""
import hashlib
import json
import os
import time

import numpy as np
from scipy import stats

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_ATTR = os.path.join(BASE, 'phase2872', 'attr_census', 'attr_census.npz')
SRC_68 = os.path.join(BASE, 'phase2868', 'growth_v2', 'growth_v2.npz')
OUT = os.path.join(BASE, 'phase2873', 'growth_axis2')

SEED = 2873
N_NULL = 200
P_SIG = 0.01

PREREG = {
    'C1': 'sparse_sufficiency_replicates iff acc(sig2)>acc(full) and '
          'acc(sig2)>null_p95(full-pipeline-mirrored, N=200)',
    'C2': 'shared_core iff hypergeom p(|intersect|) < 0.05, else axis_specific',
    'C3': 'reuse_signal iff acc(class43 on attr) > acc(full on attr)',
    'null_seed': SEED,
    'n_null': N_NULL,
    'p_sig': P_SIG,
}


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def zmat(M):
    """2867/2868 protocol: column z-score across words, unit rows."""
    mu = M.mean(axis=0, keepdims=True)
    sd = M.std(axis=0, keepdims=True)
    Z = (M - mu) / np.maximum(sd, 1e-30)
    n = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / np.maximum(n, 1e-30)


def loo_acc(C, lab):
    n = C.shape[0]
    S = C @ C.T
    np.fill_diagonal(S, -2.0)
    nn = S.argmax(axis=1)
    return float(np.mean(lab[nn] == lab))


def eta2_p(x, lab):
    """One-way eta2 with F p-value (upper tail). Returns (eta2, p)."""
    labs = np.unique(lab)
    grand = x.mean()
    ss_b = 0.0
    ss_w = 0.0
    for g in labs:
        m = lab == g
        if m.sum() == 0:
            continue
        d = x[m]
        ss_b += m.sum() * (d.mean() - grand) ** 2
        ss_w += ((d - d.mean()) ** 2).sum()
    ss_t = ((x - grand) ** 2).sum()
    et = ss_b / max(ss_t, 1e-30)
    k = len(labs)
    n = len(x)
    df_b = k - 1
    df_w = n - k
    if df_w <= 0 or ss_w <= 1e-30:
        p = 0.0 if ss_b > 1e-30 else 1.0
    else:
        f = (ss_b / df_b) / (ss_w / df_w)
        p = float(stats.f.sf(f, df_b, df_w))
    return float(et), p


def select_sig(X, lab, p_thresh):
    keep = []
    ps = np.full(X.shape[1], 1.0)
    for j in range(X.shape[1]):
        _, p = eta2_p(X[:, j], lab)
        ps[j] = p
        if p < p_thresh:
            keep.append(j)
    return keep, ps


def log(msg):
    print(msg, flush=True)


def main():
    t0 = time.monotonic()
    os.makedirs(OUT, exist_ok=True)

    # ---- execution.json frozen BEFORE any statistic is computed ----
    script = os.path.abspath(__file__)
    exec_doc = {
        'phase': 2873,
        'name': 'growth_axis2',
        'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'sources': {'attr_census': sha256_of(SRC_ATTR),
                    'growth_v2': sha256_of(SRC_68)},
        'prereg': PREREG,
        'script_sha256_8': sha256_of(script),
    }
    with open(os.path.join(OUT, 'execution.json'), 'w', encoding='utf-8') as f:
        json.dump(exec_doc, f, indent=2, ensure_ascii=False)
    log('execution.json frozen')

    za = np.load(SRC_ATTR, allow_pickle=True)
    zg = np.load(SRC_68, allow_pickle=True)

    drops = za['drops'].astype(np.float64)          # (28,36,32)
    words = list(za['target_list'])
    X_full = drops.reshape(len(words), -1)           # (28,1152)
    labels = np.array([w.split(':')[0] for w in words])
    axes = sorted(set(labels.tolist()))
    lab_idx = np.array([axes.index(w.split(':')[0]) for w in words])
    n_words, n_heads_full = X_full.shape
    log('attr words=%d axes=%d heads=%d' % (n_words, len(axes), n_heads_full))

    class_sig = np.where(zg['sig_mask'].astype(bool))[0]
    log('class43 heads: %d' % len(class_sig))

    rng = np.random.default_rng(SEED)

    # ---------- observed statistics ----------
    C_full = zmat(X_full)
    acc_full = loo_acc(C_full, lab_idx)

    sig2, ps2 = select_sig(X_full, lab_idx, P_SIG)
    log('sig2 heads (p<%.2f): %d' % (P_SIG, len(sig2)))

    if sig2:
        C_sig2 = zmat(X_full[:, sig2])
        acc_sig2 = loo_acc(C_sig2, lab_idx)
    else:
        acc_sig2 = float('nan')

    C_43 = zmat(X_full[:, class_sig])
    acc_43 = loo_acc(C_43, lab_idx)

    # top-43 by eta2 (fixed-k, symmetric overlap partner)
    etas = np.array([eta2_p(X_full[:, j], lab_idx)[0]
                     for j in range(n_heads_full)])
    top43 = list(np.argsort(-etas)[:43])

    # ---------- C1: full-pipeline-mirrored permutation null ----------
    null_full, null_sig = [], []
    for it in range(N_NULL):
        pl = rng.permutation(lab_idx)
        Cp = zmat(X_full)
        null_full.append(loo_acc(Cp, pl))
        ks, _ = select_sig(X_full, pl, P_SIG)
        if ks:
            acc_p = loo_acc(zmat(X_full[:, ks]), pl)
        else:
            acc_p = 0.0
        null_sig.append(acc_p)
        if (it + 1) % 50 == 0:
            log('  null %d/%d' % (it + 1, N_NULL))

    nf = np.array(null_full)
    ns = np.array(null_sig)
    p95_sig = float(np.percentile(ns, 95))
    p_full = float(np.mean(nf >= acc_full))
    p_sig2 = float(np.mean(ns >= acc_sig2)) if sig2 else float('nan')

    # ---------- C2: cross-axis overlap ----------
    s2 = set(sig2)
    c43 = set(class_sig.tolist())
    inter = s2 & c43
    n2 = len(s2)
    if n2 > 0:
        p_over = float(stats.hypergeom.sf(len(inter) - 1, 1152, 43, n2))
    else:
        p_over = float('nan')
    exp_inter = 43.0 * n2 / 1152.0
    n_new = n2 - len(inter)
    inter43 = len(set(top43) & c43)
    p_over43 = float(stats.hypergeom.sf(inter43 - 1, 1152, 43, 43))

    # ---------- verdicts ----------
    if sig2:
        c1 = bool(acc_sig2 > acc_full and acc_sig2 > p95_sig)
    else:
        c1 = False  # no_signal branch
    c2 = bool(p_over < 0.05) if n2 > 0 else False
    c3 = bool(acc_43 > acc_full)

    res = {
        'phase': 2873,
        'protocol': 'attr-axis causal spectrum (28 words, 10 axes) into '
                    '2867/2868 word-coordinate retrieval + density gating',
        'acc_ladder': {
            'full_1152': round(acc_full, 4),
            'sig2_p001': round(acc_sig2, 4) if sig2 else None,
            'class43_on_attr': round(acc_43, 4),
        },
        'n_sig2': n2,
        'n_class43': len(class_sig),
        'C1': {
            'verdict': 'sparse_sufficiency_replicates' if c1 else
                       ('no_signal' if not sig2 else 'not_replicated'),
            'acc_sig2': round(acc_sig2, 4) if sig2 else None,
            'acc_full': round(acc_full, 4),
            'null_sig_p95': round(p95_sig, 4),
            'null_sig_mean': round(float(ns.mean()), 4),
            'p_sig2_vs_null': round(p_sig2, 4) if sig2 else None,
            'p_full_vs_null': round(p_full, 4),
        },
        'C2': {
            'verdict': 'shared_core' if c2 else 'axis_specific',
            'intersect': len(inter),
            'expected_intersect': round(exp_inter, 2),
            'hypergeom_p': round(p_over, 5) if n2 > 0 else None,
            'n_new_heads': n_new,
            'top43_intersect': inter43,
            'top43_hypergeom_p': round(p_over43, 5),
        },
        'C3': {
            'verdict': 'reuse_signal' if c3 else 'novel_dominant',
            'acc_class43_on_attr': round(acc_43, 4),
            'acc_full_on_attr': round(acc_full, 4),
        },
        'growth_point_axis2': {
            'components_axis2': n2,
            'components_axis1_class43': int(len(class_sig)),
            'shared': len(inter),
            'new': n_new,
            'reuse_ratio': round(len(inter) / max(min(n2, 43), 1), 4)
                           if n2 > 0 else None,
            'acc_best_axis2': round(max([a for a in [acc_sig2, acc_43,
                                                      acc_full]
                                         if a == a]), 4),
        },
        'nulls': {'N': N_NULL, 'seed': SEED,
                  'null_full_mean': round(float(nf.mean()), 4),
                  'null_sig_mean': round(float(ns.mean()), 4)},
        'sig2_headlist': sorted(s2),
        'runtime_s': round(time.monotonic() - t0, 1),
    }

    with open(os.path.join(OUT, 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    np.savez_compressed(os.path.join(OUT, 'growth_axis2.npz'),
                        X_attr=X_full.astype(np.float32),
                        labels=lab_idx,
                        sig2=np.array(sorted(s2), dtype=np.int64),
                        class43=class_sig.astype(np.int64),
                        eta2_attr=etas.astype(np.float32),
                        p_attr=ps2.astype(np.float32),
                        null_sig=ns.astype(np.float32),
                        null_full=nf.astype(np.float32))

    log('==== VERDICTS ====')
    log('acc ladder: full=%.4f sig2=%s class43=%.4f'
        % (acc_full, ('%.4f' % acc_sig2) if sig2 else 'NA', acc_43))
    log('null_sig mean/p95 = %.4f / %.4f' % (ns.mean(), p95_sig))
    log('C1=%s C2=%s(inter=%d exp=%.2f p=%.5f new=%d) C3=%s'
        % (res['C1']['verdict'], res['C2']['verdict'], len(inter),
           exp_inter, p_over, n_new, res['C3']['verdict']))
    log('runtime %.1fs' % (time.monotonic() - t0))


if __name__ == '__main__':
    main()
