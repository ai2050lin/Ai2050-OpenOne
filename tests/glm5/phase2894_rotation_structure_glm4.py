# -*- coding: utf-8 -*-
"""Phase 2894: GLM4 language-direction rotation structure (zero forward).

Chain: 2893 confirmed that GLM4 deep layers write language separation
with LAYER-MATCHED eigen-directions, and cos(dir_q(li), lang_dir)
decays 0.42 -> 0.08 across W=[28,40) - the write direction rotates
with depth.  This phase quantifies the rotation geometry (zero
forward, S_last reused from 2890 npz sha 74933303), testing the
Unified Theory rotation-plane hypothesis.

Definitions (frozen before observation):
  dir(li)   = unit(mean_en S_last[:,li] - mean_fr S_last[:,li]),
              li = 0..40   (41 unit directions, 4096-d)
  dnorm(li) = ||mean_en S_last[:,li] - mean_fr S_last[:,li]||
  theta(li) = angle(dir(li), dir(li+1)) in degrees, li = 0..39
  tangent u(li) = dir(li+1) - dir(li), li = 0..39  (40 vectors)

R1 rotation profile: segment shares of total rotation
  share_s = sum theta over segment / sum theta(0..39);
  segments EARLY(0..12) / MID(13..27) / DEEP(28..39) as 2892.
  Descriptive: which segment rotates most.

R2 rotation-plane consistency (RANDOM NULL + ALGEBRAIC AUDIT,
2809 discipline): PCA (centered) of the 40 tangent vectors;
  top2_share = (s1^2+s2^2)/sum(s_i^2).
  plane_consistent if top2_share > null p95, where null = 1000
  draws of 40 random unit directions in 4096-d, same construction
  (differences of consecutive unit vectors, centered PCA).
  Audit: centered PCA verified by reconstructing; increments are
  differences of unit vectors so |u| = 2 sin(theta/2) recorded as
  cross-check of theta curve.

R3 low-dim manifold of the direction set: effective rank (entropy
  definition, singular values of uncentered matrix D=[dir(0..40)])
  er_obs vs null median over 1000 draws of 41 random unit vectors
  in 4096-d.  lowdim_manifold if er_obs < 0.5 * null_median.

R4 plane alignment: cos of top-2 tangent PCs with lang_dir (li=20)
  and with deep eigen-directions dir(39); angle stability of the
  top plane across segments (principal angles between top-2 span
  of EARLY tangents vs DEEP tangents, descriptive).

Verdict combinations recorded; all rules frozen in PREREG before
Stage-1 observation.  SEED=2894.
Output: phase2894/rotation_structure_glm4/.
"""
import hashlib
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2890 = os.path.join(BASE, 'phase2890', 'language_axis_glm4',
                        'language_axis_glm4.npz')
OUT = os.path.join(BASE, 'phase2894', 'rotation_structure_glm4')
SEED = 2894
N_NULL = 1000

PREREG = {
    'defs': 'dir(li)=unit(mean_en-mean_fr S_last[:,li]) li=0..40; '
            'theta(li)=angle(dir(li),dir(li+1)) deg; tangent u(li)='
            'dir(li+1)-dir(li)',
    'R1': 'segment shares of total rotation: EARLY theta(0..12), '
          'MID(13..27), DEEP(28..39); descriptive',
    'R2': 'top2_share of centered PCA of 40 tangents; '
          'plane_consistent if > null p95 (1000 draws of 40 random '
          'unit dirs, same tangent construction, SEED=2894); audit: '
          '|u(li)| = 2 sin(theta/2) cross-check, centered-PCA '
          'reconstruction err < 1e-10',
    'R3': 'effective rank (entropy, uncentered D=[dir(0..40)]); '
          'lowdim_manifold if er_obs < 0.5 * null_median (1000 draws '
          'of 41 random unit dirs)',
    'R4': 'cos(top2 tangent PCs, lang_dir/dir39) and principal '
          'angles EARLY-span vs DEEP-span, descriptive only',
    'verdict': 'combination of R2/R3 labels + descriptive R1/R4; no '
               'post-hoc thresholds',
    'amendment': 'run 1 crashed at R4 (PC convention error) before '
                 'completion; probe found dnorm(0)=0.0 exactly -> '
                 'dir(0) undefined zero vector, spurious theta[0]=90 '
                 'and audit dev 0.414; direction sequence amended to '
                 'li=1..40 (39 tangents) before any valid statistic '
                 'was frozen; old execution.json/result.json deleted',
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
        json.dump({'phase': 2894, 'name': 'rotation_structure_glm4',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'sources': {'s_last_2890': sha8(SRC_2890)},
                   'model': 'glm4-9b-chat-hf',
                   'prereg': PREREG, 'seed': SEED, 'n_null': N_NULL},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen')

    z90 = np.load(SRC_2890, allow_pickle=True)
    S_last = z90['S_last'].astype(np.float64)   # (80, 41, 4096)
    lang_dir = z90['lang_dir'].astype(np.float64)
    lab_sent = np.array([i % 2 for i in range(S_last.shape[0])])
    mean_en = S_last[lab_sent == 0].mean(0)      # (41, d)
    mean_fr = S_last[lab_sent == 1].mean(0)
    diffs = mean_en - mean_fr                    # (41, d)
    dnorm = np.linalg.norm(diffs, axis=1)
    # AMENDMENT (documented, pre-statistics): dnorm(0) = 0.0 EXACTLY
    # (2890 S_last: mean en sentence state == mean fr sentence state at
    # li=0, language separation starts from zero), so dir(0) is a zero
    # vector and undefined.  Direction sequence uses li = 1..40 only.
    # First run crashed at R4 (PC convention) before completion; old
    # execution.json/result.json deleted per discipline.
    assert float(dnorm[0]) == 0.0
    DIRS_LO = 1
    dirs = np.stack([unit(diffs[li])
                     for li in range(DIRS_LO, 41)])  # (40, d)

    # theta curve + tangent vectors (li = 1..39 -> 39 tangents)
    theta = np.array([np.degrees(np.arccos(np.clip(
        float(dirs[k] @ dirs[k + 1]), -1.0, 1.0)))
        for k in range(dirs.shape[0] - 1)])
    tangents = np.stack([dirs[k + 1] - dirs[k]
                         for k in range(dirs.shape[0] - 1)])
    tang_norm = np.linalg.norm(tangents, axis=1)
    audit_max = float(np.max(np.abs(
        tang_norm - 2.0 * np.sin(np.radians(theta) / 2.0))))
    log('theta curve (li=1..): %s' % [round(float(x), 3)
                                      for x in theta])
    log('audit |u|=2sin(theta/2): max dev %.3e' % audit_max)

    # R1 segment shares (theta indexed by li-1: EARLY li 1..12,
    # MID 13..27, DEEP 28..39, consistent with 2892 boundaries)
    tot = float(theta.sum())
    shares = {'early': float(theta[0:12].sum()) / tot,
              'mid': float(theta[12:27].sum()) / tot,
              'deep': float(theta[27:40].sum()) / tot}
    log('R1 shares: %s' % {k: round(v, 4) for k, v in shares.items()})

    # R2 top-2 plane of tangents (centered PCA) + null
    Xc = tangents - tangents.mean(0, keepdims=True)
    U, sv, _ = np.linalg.svd(Xc, full_matrices=False)
    lam = sv ** 2
    top2_share = float((lam[0] + lam[1]) / lam.sum())
    # algebraic audit: full centered PCA reconstruction
    _, sv2, vt2 = np.linalg.svd(Xc, full_matrices=False)
    recon = (U * sv2) @ vt2
    pca_recon_err = float(np.abs(Xc - recon).max())
    rng = np.random.default_rng(SEED)
    d_model = dirs.shape[1]
    n_tan = tangents.shape[0]          # 39
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

    # R3 effective rank of direction set + null
    svD = np.linalg.svd(dirs, compute_uv=False)
    er_obs = eff_rank(svD)
    null_er = []
    for _ in range(N_NULL):
        rd = rng.standard_normal((dirs.shape[0], d_model))
        rd = rd / np.linalg.norm(rd, axis=1, keepdims=True)
        null_er.append(eff_rank(np.linalg.svd(rd, compute_uv=False)))
    er_null_med = float(np.median(null_er))
    r3 = bool(er_obs < 0.5 * er_null_med)
    log('R3: er_obs=%.3f null_median=%.3f -> %s'
        % (er_obs, er_null_med,
           'lowdim_manifold' if r3 else 'full_rank_like'))

    # R4 plane alignment (descriptive); PCs = rows of vt (d-dim dirs);
    # dirs index k corresponds to layer li = k+1, so li=20 -> dirs[19],
    # li=40 (last) -> dirs[39]
    pc1, pc2 = vt2[0], vt2[1]
    cos_pc_lang = [float(pc1 @ lang_dir), float(pc2 @ lang_dir)]
    cos_pc_d40 = [float(pc1 @ dirs[39]), float(pc2 @ dirs[39])]
    cos_pc_d20 = [float(pc1 @ dirs[19]), float(pc2 @ dirs[19])]
    seg_slices = {'early': slice(0, 12), 'mid': slice(12, 27),
                  'deep': slice(27, 39)}
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
    log('R4: cos(PC1/PC2, lang_dir)=%s cos(PC,d40)=%s cos(PC,d20)=%s'
        % ([round(x, 4) for x in cos_pc_lang],
           [round(x, 4) for x in cos_pc_d40],
           [round(x, 4) for x in cos_pc_d20]))
    log('R4 principal angles: %s' % plane_angles)

    # singular spectrum of dirs (top-8) for the record
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
        'phase': 2894, 'model': 'glm4-9b-chat-hf', 'prereg': PREREG,
        'theta_curve_deg': [round(float(x), 4) for x in theta],
        'dnorm_curve': [round(float(x), 4) for x in dnorm],
        'tangent_norm': [round(float(x), 4) for x in tang_norm],
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
        'R4': {'cos_pc_langdir': [round(x, 4) for x in cos_pc_lang],
               'cos_pc_dir40': [round(x, 4) for x in cos_pc_d40],
               'cos_pc_dir20': [round(x, 4) for x in cos_pc_d20],
               'principal_angles_deg': plane_angles},
        'dirs_singular_values_top8': top_sv,
        'final_verdict': verdict,
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        os.path.join(OUT, 'rotation_structure_glm4.npz'),
        dirs=dirs.astype(np.float32),
        theta=theta.astype(np.float32),
        dnorm=dnorm.astype(np.float32),
        tangent_pc=np.stack([pc1, pc2]).astype(np.float32),
        lang_dir=lang_dir.astype(np.float32))
    log('==== VERDICT: %s ====' % verdict)
    log('runtime %.1fs' % (time.monotonic() - t0))


if __name__ == '__main__':
    main()
