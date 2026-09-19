# -*- coding: utf-8 -*-
"""Phase 2895: qwen3-4b language-direction rotation structure (zero
forward) - cross-model control for 2894.

2894 (glm4): rotation statistically concentrated but no global plane
(top2 0.1648 vs null p95 0.1084), no low-dim manifold (er 26.7/40),
segment tangent spans near-orthogonal (83-89 deg) - the rotation
plane itself rotates with depth.  Control question: does qwen3-4b,
whose language mlp carrier is POSITIVE (0.7719, 2887), show a
STABLER direction geometry?  If yes => cross-model anticorrelation
between direction rotation and carrier strength.

Source: 2886 S_last (80, 37, 2560), qwen3-4b, 36 layers (li 0..36),
40 en/fr PAIRS, labels i%2 (frozen; assert against npz labels).
AMENDMENT carried from 2894: check dnorm(0); exclude dir(0) if
zero/degenerate (document in execution.json).

Definitions (frozen, same as 2894):
  dir(li)=unit(mean_en-mean_fr S_last[:,li]); theta(li)=angle between
  consecutive dirs (deg); tangent u(li)=dir(li+1)-dir(li).
  Qwen-adapted segments (frozen, descriptive): window [26,36)
  (2884 precedent) -> EARLY theta(li=1..12), MID(13..25),
  DEEP(26..35).
R2: top2 centered-PCA share of tangents vs null p95 (1000 draws of
  n_tan+1 random unit vectors, same tangent construction, SEED=2895).
R3: effective rank (entropy, uncentered dir matrix) vs null median;
  lowdim_manifold if er_obs < 0.5 * null_median.
R4 (descriptive): cos(top2 PCs, ref dir) where ref = 2887-style
  lang_dir at li=18 (frozen: 2887 used li=18); cos(PC, d36);
  principal angles between segment top-2 spans.
Verdict labels as 2894.  SEED=2895.
Output: phase2895/rotation_structure_qwen/.
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2886 = os.path.join(BASE, 'phase2886', 'hourglass_cka',
                        'hourglass_cka.npz')
OUT = os.path.join(BASE, 'phase2895', 'rotation_structure_qwen')
SEED = 2895
N_NULL = 1000

PREREG = {
    'source': '2886 hourglass_cka.npz S_last (80,37,2560), qwen3-4b, '
              'labels i%2 asserted',
    'amendment': 'dnorm(0) checked; dir(0) excluded if zero/'
                 'degenerate (2894 precedent), documented',
    'defs': 'dir(li)=unit(mean_en-mean_fr S_last[:,li]); theta=angle '
            'consecutive dirs; tangent u=dir(li+1)-dir(li)',
    'segments': 'qwen-adapted (window [26,36) 2884 precedent): '
                'EARLY theta(li 1..12), MID(13..25), DEEP(26..35)',
    'R2': 'top2_share centered PCA of tangents > null p95 (1000 '
          'draws, SEED=2895) => plane_consistent',
    'R3': 'er_obs < 0.5 * null_median (1000 draws) => '
          'lowdim_manifold',
    'R4': 'cos(top2 PCs, dir18/dir36) + segment principal angles, '
          'descriptive; cross-model comparison vs 2894 glm4 values '
          'recorded as observation',
    'verdict': 'combination of R2/R3 labels; no post-hoc thresholds',
}


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-30)


def eff_rank(s):
    s = np.maximum(s, 1e-30)
    p = s / s.sum()
    ent = float(-(p * np.log(p)).sum())
    return float(np.exp(ent))


def log(msg):
    print(msg, flush=True)


def main():
    t0 = time.monotonic()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2895, 'name': 'rotation_structure_qwen',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'sources': {'s_last_2886': sha8(SRC_2886)},
                   'model': 'qwen3-4b',
                   'prereg': PREREG, 'seed': SEED, 'n_null': N_NULL},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen')

    z86 = np.load(SRC_2886, allow_pickle=True)
    S_last = z86['S_last'].astype(np.float64)   # (80, 37, 2560)
    labels = z86['labels']
    lab_sent = np.asarray(labels).astype(int)
    assert sorted(set(lab_sent.tolist())) == [0, 1]
    assert all(int(lab_sent[i]) == i % 2 for i in range(80)), \
        'labels must follow PAIRS i%2 order'
    mean_en = S_last[lab_sent == 0].mean(0)      # (37, d)
    mean_fr = S_last[lab_sent == 1].mean(0)
    diffs = mean_en - mean_fr
    dnorm = np.linalg.norm(diffs, axis=1)
    log('dnorm(0..5): %s' % [round(float(x), 6)
                             for x in dnorm[:6]])

    # amendment check: exclude degenerate dir(0)
    li_lo = 1 if float(dnorm[0]) < 1e-8 else 0
    if li_lo == 1:
        log('amendment applied: dnorm(0)=%.3e -> dir(0) excluded'
            % float(dnorm[0]))
    dirs = np.stack([unit(diffs[li])
                     for li in range(li_lo, 37)])  # (n_dir, d)
    n_dir = dirs.shape[0]
    n_tan = n_dir - 1

    theta = np.array([np.degrees(np.arccos(np.clip(
        float(dirs[k] @ dirs[k + 1]), -1.0, 1.0)))
        for k in range(n_tan)])
    tangents = np.stack([dirs[k + 1] - dirs[k]
                         for k in range(n_tan)])
    tang_norm = np.linalg.norm(tangents, axis=1)
    audit_max = float(np.max(np.abs(
        tang_norm - 2.0 * np.sin(np.radians(theta) / 2.0))))
    log('theta curve: %s' % [round(float(x), 3) for x in theta])
    log('audit |u|=2sin(theta/2): max dev %.3e' % audit_max)

    # R1 segment shares: theta index k <-> li = k + li_lo
    # EARLY li 1..12 -> k 0..11 ; MID li 13..25 -> k 12..24 ;
    # DEEP li 26..35 -> k 25..34
    tot = float(theta.sum())
    shares = {'early': float(theta[0:12].sum()) / tot,
              'mid': float(theta[12:25].sum()) / tot,
              'deep': float(theta[25:35].sum()) / tot}
    log('R1 shares: %s' % {k: round(v, 4) for k, v in shares.items()})

    # R2 top-2 plane of tangents + null
    Xc = tangents - tangents.mean(0, keepdims=True)
    U, sv, vt = np.linalg.svd(Xc, full_matrices=False)
    lam = sv ** 2
    top2_share = float((lam[0] + lam[1]) / lam.sum())
    recon = (U * sv) @ vt
    pca_recon_err = float(np.abs(Xc - recon).max())
    rng = np.random.default_rng(SEED)
    d_model = dirs.shape[1]
    null_shares = []
    for _ in range(N_NULL):
        rd = rng.standard_normal((n_tan + 1, d_model))
        rd = rd / np.linalg.norm(rd, axis=1, keepdims=True)
        rt = rd[1:] - rd[:-1]
        rtc = rt - rt.mean(0, keepdims=True)
        s2 = np.linalg.svd(rtc, compute_uv=False) ** 2
        null_shares.append(float((s2[0] + s2[1]) / s2.sum()))
    null_p95 = float(np.percentile(null_shares, 95))
    null_mean = float(np.mean(null_shares))
    r2 = bool(top2_share > null_p95)
    log('R2: top2_share=%.4f null_mean=%.4f null_p95=%.4f -> %s'
        % (top2_share, null_mean, null_p95,
           'plane_consistent' if r2 else 'plane_distributed'))

    # R3 effective rank + null
    svD = np.linalg.svd(dirs, compute_uv=False)
    er_obs = eff_rank(svD)
    null_er = []
    for _ in range(N_NULL):
        rd = rng.standard_normal((n_dir, d_model))
        rd = rd / np.linalg.norm(rd, axis=1, keepdims=True)
        null_er.append(eff_rank(np.linalg.svd(rd, compute_uv=False)))
    er_null_med = float(np.median(null_er))
    r3 = bool(er_obs < 0.5 * er_null_med)
    log('R3: er_obs=%.3f null_median=%.3f -> %s'
        % (er_obs, er_null_med,
           'lowdim_manifold' if r3 else 'full_rank_like'))

    # R4 plane alignment (descriptive); ref = dir at li=18 (2887
    # lang_dir definition); dirs index k <-> li = k + li_lo
    k18, k36 = 18 - li_lo, n_dir - 1
    pc1, pc2 = vt[0], vt[1]
    cos_pc_ref = [float(pc1 @ dirs[k18]), float(pc2 @ dirs[k18])]
    cos_pc_last = [float(pc1 @ dirs[k36]), float(pc2 @ dirs[k36])]
    seg_slices = {'early': slice(0, 12), 'mid': slice(12, 25),
                  'deep': slice(25, 35)}
    plane_angles = {}
    for sa, sb in [('early', 'deep'), ('mid', 'deep'),
                   ('early', 'mid')]:
        A = Xc[seg_slices[sa]]
        B = Xc[seg_slices[sb]]
        _, _, vtA = np.linalg.svd(A, full_matrices=False)
        _, _, vtB = np.linalg.svd(B, full_matrices=False)
        M = vtA[:2] @ vtB[:2].T
        sang = np.linalg.svd(M, compute_uv=False)
        plane_angles['%s_vs_%s' % (sa, sb)] = [
            round(float(np.degrees(np.arccos(np.clip(x, -1, 1)))), 2)
            for x in sang[:2]]
    log('R4: cos(PC1/PC2, dir18)=%s cos(PC,d_last)=%s'
        % ([round(x, 4) for x in cos_pc_ref],
           [round(x, 4) for x in cos_pc_last]))
    log('R4 principal angles: %s' % plane_angles)

    # window cos decay (2894 analog, li=26..35 vs ref dir18)
    win_cos = {int(k + li_lo): round(float(dirs[k] @ dirs[k18]), 4)
               for k in range(25, n_dir)}
    log('window cos(dir(li), dir18): %s' % win_cos)

    top_sv = [round(float(x), 4) for x in svD[:8]]

    if r2 and r3:
        verdict = 'rotation_plane_in_lowdim_manifold'
    elif r2:
        verdict = 'rotation_plane_consistent_manifold_fullrank'
    elif r3:
        verdict = 'lowdim_manifold_plane_distributed'
    else:
        verdict = 'rotation_structure_distributed'

    res = {
        'phase': 2895, 'model': 'qwen3-4b', 'prereg': PREREG,
        'li_lo': li_lo, 'dnorm_head': [round(float(x), 6)
                                       for x in dnorm[:6]],
        'theta_curve_deg': [round(float(x), 4) for x in theta],
        'dnorm_curve': [round(float(x), 4) for x in dnorm],
        'audit_2sin_crosscheck_max_dev': audit_max,
        'pca_recon_max_err': pca_recon_err,
        'R1_shares': {k: round(v, 4) for k, v in shares.items()},
        'R2': {'top2_share': round(top2_share, 4),
               'null_mean': round(null_mean, 4),
               'null_p95': round(null_p95, 4),
               'verdict': 'plane_consistent' if r2
               else 'plane_distributed'},
        'R3': {'er_obs': round(er_obs, 4),
               'null_median': round(er_null_med, 4),
               'verdict': 'lowdim_manifold' if r3
               else 'full_rank_like'},
        'R4': {'cos_pc_dir18': [round(x, 4) for x in cos_pc_ref],
               'cos_pc_dir_last': [round(x, 4) for x in cos_pc_last],
               'principal_angles_deg': plane_angles},
        'window_cos_to_dir18': win_cos,
        'dirs_singular_values_top8': top_sv,
        'cross_model_2894_glm4': {'top2_share': 0.1648,
                                  'null_p95': 0.1084,
                                  'er_obs': 26.657,
                                  'er_null_median': 39.952,
                                  'principal_angles_83_89': True},
        'final_verdict': verdict,
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        os.path.join(OUT, 'rotation_structure_qwen.npz'),
        dirs=dirs.astype(np.float32),
        theta=theta.astype(np.float32),
        dnorm=dnorm.astype(np.float32),
        tangent_pc=np.stack([pc1, pc2]).astype(np.float32))
    log('==== VERDICT: %s ====' % verdict)
    log('runtime %.1fs' % (time.monotonic() - t0))


if __name__ == '__main__':
    main()
