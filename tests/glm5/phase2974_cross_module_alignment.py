# -*- coding: utf-8 -*-
"""Phase 2974: plan-v3 Omega-A cross-module subspace alignment
(preregistered, zero-forward).

Why (attachment gap 3): does the Attention write space and
the MLP read space at the same/next layer show geometric
alignment beyond chance ("handshake protocol")? Prior
lessons (2809/2928/2931): subspace-overlap claims in 2560-d
MUST be null-calibrated; targeted carrier tests are
quasi-post-hoc (2923).

Design (frozen before any observation):
  A-space (attention write space) at layer li: left top-64
  singular vectors of Wo[li] head-slice stack
  (2560 x 128|H|). T1 uses ALL 32 heads (2560x4096);
  T2 uses the 2964 L34 carrier top5 heads [15,8,21,28,11]
  (QUASI-POST-HOC: identified in upstream 2964).
  M-space (MLP read space) at layer lj: top-64 right
  singular vectors of mlp.up_proj.weight (I x 2560).
  Metric: principal cosines s = svd(Q_A^T Q_M);
  alignment score S = mean(s[:16]).

Tests (frozen):
  T1 same-layer family: S(li,li) for li 0..35; nulls =
  N1 random-orthogonal (200 draws, maxT over 36 layers)
  and N2 off-diagonal pool max (conservative); sig layer:
  S(li,li) >= max(N1 maxT p95, N2 max p95).
  T2 carrier profile: S(A34carrier, M(lj)) lj 0..35;
  nulls = N1 (200) + N2m 200 draws of 5 random L34 heads
  (matched size, maxT over 36 lj); gate: sig at lj in
  {34,35} AND profile argmax in {33,34,35}.
  T3 descriptive: full 36x36 S matrix; gate robustness
  with gate_proj (descriptive only).

Anchors (frozen):
  a1 structural gate: o_proj.in_features == 4096 ==
     32*128 (36/36); up_proj.in_features == 2560 (36/36)
  a2 orthonormality: max|Q^T Q - I| < 1e-8 (all Q)
  a3 self-alignment: S(Q,Q) top16 mean == 1 < 1e-9
  a4 determinism: S(A34carrier, M34) rerun |diff| < 1e-12
  a5 carrier-set identity: heads == 2964 T3 top5_heads
  a6 non-degeneracy: all S finite in [0,1]; up sigma64/
     sigma1 > 1e-6 per layer (36/36)

Verdict (frozen):
  anchor fail => anchor_fail_all_void
  T2 gate met and T1 sig >= 1 layer =>
     cross_module_alignment_localized
  T1 sig >= 6 layers (T2 gate not met) =>
     cross_module_alignment_widespread
  T1 sig >= 1 => cross_module_alignment_partial
  else => omega_a_alignment_all_void
"""
import hashlib
import io
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
SRC_2964 = os.path.join(BASE, 'phase2964', 'carrier_anatomy',
                        'result.json')
OUT = os.path.join(BASE, 'phase2974', 'cross_module_alignment')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2974_run_report.txt')
MD = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
NH, HD = 32, 128
NL = 36
K = 64
K_TOP = 16
N_NULL = 200
P_TH = 0.01
RNG = 2974
CARRIER_LAYER = 34
CARRIER_GATE_LJ = (34, 35)
CARRIER_PEAK_WIN = (33, 34, 35)
T1_WIDESPREAD = 6

PREREG = {
    'mode': 'zero-forward; weight-space SVD subspaces; '
            'A = o_proj input head-slice stack left top-64 '
            'svectors; M = mlp.up_proj right top-64 '
            'svectors; metric = mean top-16 principal '
            'cosines; nulls N1 random-orthogonal x200 '
            '(maxT) + N2 matched-random-heads x200 (maxT) '
            '+ off-diagonal pool',
    'question': 'does the attention write space align with '
                'the MLP read space at the same/adjacent '
                'layer beyond matched random nulls '
                '(cross-module handshake)?',
    'carrier_set_source': 'phase2964 T3 top5_heads L34 '
                          '[15,8,21,28,11] (quasi-post-hoc '
                          'targeted test, discipline 9)',
    'anchors': {
        'a1': 'structural gate 4096==32*128 and '
              'up in==2560 (36/36)',
        'a2': 'orthonormality < 1e-8',
        'a3': 'self-alignment == 1 < 1e-9',
        'a4': 'determinism < 1e-12',
        'a5': 'carrier set == 2964 T3 top5_heads',
        'a6': 'S finite in [0,1]; up sigma64/sigma1 > 1e-6',
    },
    'T1': 'same-layer S(li,li) family 36; sig if >= '
          'max(N1-maxT p95, offdiag-max p95)',
    'T2': 'carrier profile S(A34c, M(lj)) with N1+N2m '
          'maxT; gate sig at lj in {34,35} and argmax in '
          '{33,34,35}',
    'T3': 'descriptive: 36x36 matrix; gate_proj '
          'robustness (no gate)',
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'T2 met & T1 sig>=1 => '
               'cross_module_alignment_localized; '
               'T1 sig>=6 => '
               'cross_module_alignment_widespread; '
               'T1 sig>=1 => '
               'cross_module_alignment_partial; '
               'else => omega_a_alignment_all_void',
    'correction_note': 'run1: missing import io (pre-freeze, no artifacts). run2: anchors a1/a2/a3 failed - a1 shape check inverted (o_proj weight is (2560, 4096) = (out, in)) and float32 SVD orthonormality error 2.4e-07 exceeded 1e-8 gate; fix: weights cast to float64 + shape gate corrected; artifacts deleted and rerun per discipline 3.',
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


def orthonorm_basis(Amat, k):
    U, s, _ = np.linalg.svd(Amat, full_matrices=False)
    Q = U[:, :k]
    return Q, s


def align_score(Qa, Qm):
    s = np.linalg.svd(Qa.T @ Qm, compute_uv=False)
    return float(s[:K_TOP].mean())


def main():
    t0 = time.monotonic()
    lines = []
    os.makedirs(OUT, exist_ok=True)

    # ---------- a5 carrier set from 2964 ----------
    r64 = json.load(io.open(SRC_2964, encoding='utf-8'))
    carrier = [int(x) for x, _ in
               r64['T3']['top5_heads']]
    a5_ok = bool(carrier == [15, 8, 21, 28, 11]
                 and r64['T3']['layer'] == CARRIER_LAYER
                 and r64['T3']['sig_heads'] == [15])
    log('a5 carrier set %s ok=%s' % (carrier, a5_ok),
        lines)

    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2974,
                   'name': 'cross_module_alignment',
                   'created': time.strftime(
                       '%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8':
                       sha8(os.path.abspath(__file__)),
                   'sources': {'s2964': sha8(SRC_2964)},
                   'model': 'qwen3-4b', 'heads': NH,
                   'head_dim': HD, 'n_layers': NL,
                   'k': K, 'k_top': K_TOP,
                   'n_null': N_NULL, 'p_threshold': P_TH,
                   'rng': RNG,
                   'carrier_layer': CARRIER_LAYER,
                   'carrier_heads': carrier,
                   'prereg': PREREG},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen', lines)

    # ---------- model weights ----------
    import sys
    sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
    from phase2662_symmetric_mapping_contract import \
        load_native
    model, _ = load_native('qwen4')
    model.eval()
    layers = model.model.layers
    log('model loaded', lines)

    # ---------- a1 structural gate ----------
    a1_ok = True
    Wos, Ups, Gates = {}, {}, {}
    for li in range(NL):
        Wo = layers[li].self_attn.o_proj.weight.detach() \
            .float().cpu().numpy().astype(np.float64)
        Wup = layers[li].mlp.up_proj.weight.detach() \
            .float().cpu().numpy().astype(np.float64)
        Wg = layers[li].mlp.gate_proj.weight.detach() \
            .float().cpu().numpy().astype(np.float64)
        if not (Wo.shape == (2560, NH * HD)
                and Wup.shape[1] == 2560
                and Wg.shape[1] == 2560):
            a1_ok = False
        Wos[li] = Wo
        Ups[li] = Wup
        Gates[li] = Wg
    a1_ok = bool(a1_ok)
    log('a1 structural gate ok=%s' % a1_ok, lines)

    # ---------- M spaces ----------
    QM = {}
    sig_ratio = np.ones(NL)
    for lj in range(NL):
        _, s_up, Vt = np.linalg.svd(Ups[lj],
                                    full_matrices=False)
        QM[lj] = Vt[:K].T
        sig_ratio[lj] = s_up[K - 1] / s_up[0]
    log('M spaces done (36 up SVD)', lines)

    def Qheads(li, heads):
        cols = np.concatenate(
            [np.arange(h * HD, (h + 1) * HD)
             for h in heads])
        Q, _ = orthonorm_basis(Wos[li][:, cols], K)
        return Q

    orth_err = [0.0]

    def chk(Q):
        orth_err[0] = max(orth_err[0], float(
            np.abs(Q.T @ Q - np.eye(K)).max()))

    # ---------- a2/a3 ----------
    chk(QM[0])
    Qa_self, _ = orthonorm_basis(
        Wos[CARRIER_LAYER][:, :K], K)
    chk(Qa_self)
    a2_ok = bool(orth_err[0] < 1e-8)
    a3_val = align_score(Qa_self, Qa_self)
    a3_ok = bool(abs(a3_val - 1.0) < 1e-9)
    log('a2 orth err %.2e ok=%s | a3 self %.12f ok=%s'
        % (orth_err[0], a2_ok, a3_val, a3_ok), lines)

    # ---------- a4 determinism ----------
    Qc = Qheads(CARRIER_LAYER, carrier)
    chk(Qc)
    s_a = align_score(Qc, QM[CARRIER_LAYER])
    Qc2 = Qheads(CARRIER_LAYER, carrier)
    s_b = align_score(Qc2, QM[CARRIER_LAYER])
    a4_diff = abs(s_a - s_b)
    a4_ok = bool(a4_diff < 1e-12)
    log('a4 determinism diff %.2e ok=%s'
        % (a4_diff, a4_ok), lines)

    # ---------- T1: same-layer family ----------
    S_diag = np.zeros(NL)
    for li in range(NL):
        Qa, _ = orthonorm_basis(Wos[li], K)
        chk(Qa)
        S_diag[li] = align_score(Qa, QM[li])
    rng = np.random.default_rng(RNG)
    n1_max = np.zeros(N_NULL)
    for d in range(N_NULL):
        G = rng.standard_normal((2560, K))
        Qr, _ = np.linalg.qr(G)
        smax = 0.0
        for li in range(NL):
            smax = max(smax, align_score(Qr, QM[li]))
        n1_max[d] = smax
    # off-diagonal pool
    S_mat = np.zeros((NL, NL))
    for li in range(NL):
        Qa, _ = orthonorm_basis(Wos[li], K)
        for lj in range(NL):
            S_mat[li, lj] = align_score(Qa, QM[lj])
    off = S_mat[~np.eye(NL, dtype=bool)]
    off_max = float(off.max())
    thr1 = float(max(np.quantile(n1_max, 1 - P_TH),
                     off_max))
    sig_layers = [li for li in range(NL)
                  if S_diag[li] >= thr1]
    log('T1 S_diag top5 %s | thr %.4f (N1maxT %.4f / '
        'offmax %.4f) sig %s'
        % ([(int(li), round(float(S_diag[li]), 4))
            for li in np.argsort(-S_diag)[:5]],
           thr1, float(np.quantile(n1_max, 1 - P_TH)),
           off_max, sig_layers), lines)

    # ---------- T2: carrier profile ----------
    S_car = np.array([align_score(Qc, QM[lj])
                      for lj in range(NL)])
    n2_max = np.zeros(N_NULL)
    for d in range(N_NULL):
        hs = list(rng.choice(NH, 5, replace=False))
        Qr = Qheads(CARRIER_LAYER, hs)
        chk(Qr)
        smax = 0.0
        for lj in range(NL):
            smax = max(smax, align_score(Qr, QM[lj]))
        n2_max[d] = smax
    thr2 = float(max(np.quantile(n1_max, 1 - P_TH),
                     np.quantile(n2_max, 1 - P_TH)))
    sig_car = [lj for lj in range(NL)
               if S_car[lj] >= thr2]
    argmax_car = int(np.argmax(S_car))
    t2_met = bool(any(lj in sig_car
                      for lj in CARRIER_GATE_LJ)
                  and argmax_car in CARRIER_PEAK_WIN)
    log('T2 S_car top5 %s | argmax L%d | thr %.4f | '
        'sig %s | gate met %s'
        % ([(int(lj), round(float(S_car[lj]), 4))
            for lj in np.argsort(-S_car)[:5]],
           argmax_car, thr2, sig_car, t2_met), lines)

    # ---------- a6 ----------
    a6_ok = bool(np.all(np.isfinite(S_mat))
                 and S_mat.max() <= 1.0 + 1e-9
                 and S_mat.min() >= -1e-9
                 and (sig_ratio > 1e-6).all())
    log('a6 non-degeneracy ok=%s (min sig ratio %.2e)'
        % (a6_ok, sig_ratio.min()), lines)

    # ---------- T3 descriptive: gate robustness ----------
    QG = {}
    for lj in range(NL):
        _, _, Vt = np.linalg.svd(Gates[lj],
                                 full_matrices=False)
        QG[lj] = Vt[:K].T
    S_car_gate = np.array([align_score(Qc, QG[lj])
                           for lj in range(NL)])
    rho_gate = float(np.corrcoef(S_car, S_car_gate)[0, 1])
    band_mean = float(np.mean([
        S_mat[li, lj] for li in range(NL)
        for lj in range(NL)
        if abs(li - lj) <= 1 and li >= 14 and lj <= 35
        and li >= lj - 1 and li <= lj + 1]))
    t3 = {'S_car_gate_profile_top5': [
              (int(lj), round(float(S_car_gate[lj]), 4))
              for lj in np.argsort(-S_car_gate)[:5]],
          'rho_up_vs_gate': round(rho_gate, 4),
          'near_diag_band_mean_L14plus':
              round(band_mean, 4)}
    log('T3 gate rho %.4f | near-diag band mean %.4f'
        % (rho_gate, band_mean), lines)

    anchor_ok = bool(a1_ok and a2_ok and a3_ok and a4_ok
                     and a5_ok and a6_ok)
    verdict = None
    if not anchor_ok:
        verdict = 'anchor_fail_all_void'
    else:
        if t2_met and len(sig_layers) >= 1:
            verdict = 'cross_module_alignment_localized'
        elif len(sig_layers) >= T1_WIDESPREAD:
            verdict = 'cross_module_alignment_widespread'
        elif len(sig_layers) >= 1:
            verdict = 'cross_module_alignment_partial'
        else:
            verdict = 'omega_a_alignment_all_void'
    if verdict is None:
        verdict = 'anchor_fail_all_void'
    log('==== VERDICT: %s ====' % verdict, lines)

    res = {'phase': 2974, 'model': 'qwen3-4b',
           'prereg': PREREG,
           'anchors': {'a1_ok': a1_ok,
                       'a2_err': float('%.3e'
                                       % orth_err[0]),
                       'a2_ok': a2_ok,
                       'a3_val': a3_val,
                       'a3_ok': a3_ok,
                       'a4_diff': float('%.3e'
                                        % a4_diff),
                       'a4_ok': a4_ok,
                       'a5_ok': a5_ok,
                       'a6_ok': a6_ok,
                       'ok': anchor_ok},
           'T1': {'sig_layers': sig_layers,
                  'threshold': round(thr1, 4),
                  'thr_N1_maxT': round(float(
                      np.quantile(n1_max, 1 - P_TH)), 4),
                  'thr_offdiag_max': round(off_max, 4),
                  'top5_diag': [(int(li), round(
                      float(S_diag[li]), 4))
                      for li in np.argsort(-S_diag)[:5]]},
           'T2': {'sig_layers': sig_car,
                  'argmax_layer': argmax_car,
                  'threshold': round(thr2, 4),
                  'gate_met': t2_met,
                  'top5_profile': [(int(lj), round(
                      float(S_car[lj]), 4))
                      for lj in np.argsort(-S_car)[:5]]},
           'T3': t3,
           'final_verdict': verdict,
           'runtime_s': round(time.monotonic() - t0, 1)}
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    np.savez_compressed(os.path.join(
        OUT, 'cross_module_alignment.npz'),
        S_matrix=S_mat, S_diag=S_diag,
        S_carrier=S_car, S_carrier_gate=S_car_gate,
        n1_max=n1_max, n2_max=n2_max,
        sig_ratio=sig_ratio,
        carrier_heads=np.array(carrier))
    with open(REPORT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('OK phase2974 verdict=%s' % verdict, flush=True)


if __name__ == '__main__':
    main()
