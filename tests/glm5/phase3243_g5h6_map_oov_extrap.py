# -*- coding: utf-8 -*-
"""Phase 3243 G5-H6: transfer-map OOV-extrapolation improvement arm. Zero GPU.

Prereg (frozen before any observation): MEMO Phase 3242 tail arm (a).
Target: 3242 located the gap precisely — seen rows are linearly alignable
across models (map R2 0.81-0.84) but the dual-ridge map does NOT extrapolate
to OOV rows (r2_map_oov -0.295 / +0.113), so only ~0.12-0.24 cos of true OOV
class structure is recovered and T3_dual stops just short of the +0.5 gate.
This phase improves the DONOR->4b MAP (not the panel, not the metric): three
preregistered map variants, all trained on 4b SEEN train rows only.

Leakage declaration (prereg): unchanged from 3242 — allowed = donor-model
(qwen3-14b / glm4-9b) activations on the SAME sealed 3169 panel rows (donor
OOV class means are the DECLARED side information) and seen-row-trained
maps. forbidden = any qwen3-4b OOV activation as prediction INPUT; 4b OOV
activations are used only as the evaluation target and for headline-only
transfer-fidelity diagnostics (cos vs true OOV class direction).

Map variants (preregistered, all trained on seen train rows per seed):
  base    verbatim 3242 dual-ridge with bias (replication anchor)
  M1      CENTERED dual-ridge: fit on row-centered (Hd - mu_d) -> (Y - mu_y),
          no bias term; removes mean absorption that dominates OOV bias error
  M2      CLASS-CONDITIONAL AFFINE: M1 map + per-seen-class mapping residual
          correction r_c = cls_mean_c - M1(m_c_donor); OOV correction weights
          w = softmax_1(cos_donor(m_oov - mu_d, m_c_seen - mu_d)), beta=1 fixed
  M3      MULTI-ANCHOR MIN-NORM (Procrustes-style interpolation): affine map
          W3 = pinv(A6) @ d_cls with A6 = donor seen class directions
          (m_c - mu_d), rcond=1e-6 — interpolates seen anchors exactly,
          min-norm extension outside (no shrinkage parameter)

Recipes (preregistered, no post-hoc selection):
  G0        verbatim 3240-F2 subspace completion (replication anchor)
  T3_rep    verbatim 3242 T3_dual (base map, joint span 6+4+4) — bit-level
            anchor vs sealed 3242 result.json per_seed T3_dual
  M1_dual   joint span 6 + 4 + 4 with M1 (centered) maps, both donors
  M2_dual   joint span with M2 (class-conditional affine) maps
  M3_dual   joint span with M3 (multi-anchor min-norm) maps
  M1r_dual  matched control of M1_dual: donor OOV row->class assignment
            permuted (seed 42, row-level, verbatim 3242 construction)
  M2r_dual  matched control of M2_dual (same permutation)
  M3r_dual  matched control of M3_dual (same permutation)

Gates:
  G1 replication: d83/d_tc/drift69/d88 <1e-9, d89 <1e-6, census spectra
     cross-check vs sealed 3238, G0 per-seed vs sealed 3240 F2 <1e-9 AND
     T3_rep per-seed vs sealed 3242 T3_dual <1e-9 (formal mode only; smoke
     row subset recorded-only, 3240 lesson (2))
  G2 efficacy: best(M1_dual, M2_dual, M3_dual) improves BOTH SCLS and ENT
     fidelity by > +0.5 vs G0 (3-seed mean), min(d_DVU, d_DCC) > -0.05,
     R2 drop < 0.01, AND beats its matched permuted control on
     min(SCLS, ENT)
  G3 determinism: identical-GEMM-shape lens recompute bitwise 0.0
  G4 registry v1.33 untouched (byte sha8)
Headline (not gates): SCLS/ENT sign flip; per-donor per-variant map R2
(seen-test / OOV) and transfer cos vs true OOV direction; KL ordering.
SMOKE: P3243_SMOKE=1 (seed 7 only, KL first 200 OOV rows, smoke_* files).
Registry untouched (v1.33 remains head).
"""
import os, sys, json, time, hashlib
import numpy as np

T0 = time.time()
SRC_DIR = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913'
NAME = 'g5h6_map_oov_extrap'
SMOKE = os.environ.get('P3243_SMOKE') == '1'
MODEL = 'qwen3-4b'
OUTDIR = os.path.join(SRC_DIR, 'phase3243', NAME)
if SMOKE:
    OUTDIR = os.path.join(OUTDIR, 'smoke')
LOG = []

SEEDS_ALL = [7, 8, 9]
SEEDS_RUN = [7] if SMOKE else SEEDS_ALL
KL_ROWS_SMOKE = 200
BATCH_DEFAULT = 128
LAM = 1e-3
FRAC_S1 = 0.2
REPL_TOL = 1e-9
F32_TOL = 1e-6
FAMS = ['DVU', 'DCC', 'ENT', 'USTAR', 'SCLS', 'MASSIVE']
RECIPES = ['G0', 'T3_rep', 'M1_dual', 'M2_dual', 'M3_dual',
           'M1r_dual', 'M2r_dual', 'M3r_dual']
EFF_DELTA = 0.5
REG_DROP = 0.05
R2_DROP = 0.01
RAND_SEED = 42
DONORS = {'14b': 'qwen3-14b', 'glm': 'glm4-9b'}
CTRL_OF = {'M1_dual': ['M1r_dual'], 'M2_dual': ['M2r_dual'], 'M3_dual': ['M3r_dual']}

ANCH_3157 = {'qwen3-4b': dict(nl=36, kout=35, tc_mean_cos=0.5629908442497253)}
CLASSES_SEEN = ['水果', '动物', '交通工具', '家具', '金属', '颜色']
ENT_SEEN = {
    '水果': ['苹果', '香蕉', '梨', '西瓜', '葡萄', '草莓', '芒果', '柠檬'],
    '动物': ['狗', '猫', '老虎', '大象', '兔子', '猴子', '马', '牛'],
    '交通工具': ['汽车', '火车', '飞机', '摩托车', '卡车', '地铁'],
    '家具': ['桌子', '椅子', '床', '沙发', '地毯', '窗帘'],
    '金属': ['铁', '铜', '铝', '金', '银', '锌', '铅'],
    '颜色': ['红', '蓝', '绿', '黄', '黑', '白'],
}
TPL = {0: '{e}是一种{c}。', 1: '{e}属于{c}这一类。', 2: '{e}，一种常见的{c}。'}
CLASSES_OOV = ['乐器', '天气', '运动', '电器']
ENT_OOV_PANEL = {
    '乐器': ['钢琴', '小提琴', '吉他', '鼓', '笛子', '二胡', '琵琶', '口琴'],
    '天气': ['雨', '雪', '雷', '雾', '冰雹', '台风', '露水', '霜'],
    '运动': ['足球', '篮球', '乒乓球', '游泳', '跑步', '体操', '拳击', '围棋'],
    '电器': ['电视', '冰箱', '洗衣机', '空调', '微波炉', '电饭煲', '吸尘器', '风扇'],
}
NT = 3
CLASSES = CLASSES_SEEN + CLASSES_OOV
ENT = dict(ENT_SEEN)
ENT.update(ENT_OOV_PANEL)
ENTS = [e for cl in CLASSES for e in ENT[cl]]
NE = len(ENTS)
NC = len(CLASSES)
PAIRS = [(i, c) for i in range(NE) for c in range(NC)]
NP_ = len(PAIRS)
assert (NE, NC, NP_) == (73, 10, 730)
PI_OF_PAIR = {p: j for j, p in enumerate(PAIRS)}
ALL_OOV_ROWS = [t * NP_ + pi for t in range(NT) for pi in range(NP_)
                if PAIRS[pi][1] >= len(CLASSES_SEEN)]
assert len(ALL_OOV_ROWS) == 876
NCS = len(CLASSES_SEEN)
NTR_ENT = 41
OOV_CLASSES = list(range(NCS, NC))

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC = {
    'p3169_result': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\result.json'),
    'p3169_4b': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\collect_qwen3-4b.npz'),
    'p3169_14b': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\collect_qwen3-14b.npz'),
    'p3169_glm': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\collect_glm4-9b.npz'),
    'p3157_4b': os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\qwen3-4b\collect.npz'),
    'p3183_result': os.path.join(SRC_DIR, r'phase3183\g5b8_dv_lowrank\result.json'),
    'p3188_result': os.path.join(SRC_DIR, r'phase3188\g5c1_gap4_dv_attribution\result.json'),
    'p3189_result': os.path.join(SRC_DIR, r'phase3189\g5c2_residual_structure\result.json'),
    'census_npz': os.path.join(SRC_DIR, r'phase3238\g5h1_census_rank\census_spectra.npz'),
    'registry_v133': os.path.join(SRC_DIR, r'phase3238\g5h1_census_rank\atlas_registry_v1_33.json'),
    'p3240_result': os.path.join(SRC_DIR, r'phase3240\g5h3_cls_extrapolation\result.json'),
    'p3242_result': os.path.join(SRC_DIR, r'phase3242\g5h5_xmodel_transfer\result.json'),
    'hf_shard_emb': os.path.join(ROOT, r'models\hf\qwen3-4b\model-00001-of-00003.safetensors'),
    'hf_shard_norm': os.path.join(ROOT, r'models\hf\qwen3-4b\model-00003-of-00003.safetensors'),
    'hf_config': os.path.join(ROOT, r'models\hf\qwen3-4b\config.json'),
}

DESIGN = {
    'phase': '3243', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3242 tail arm (a), frozen before any observation; '
              'zero GPU (transfer-map OOV-extrapolation improvement on the '
              '3239/3240/3242 fidelity loop)',
    'leakage_declaration': 'allowed = donor-model (qwen3-14b / glm4-9b) activations '
                           'on the SAME sealed 3169 panel rows — donor OOV class means '
                           'remain the DECLARED side information (unchanged from 3242); '
                           'all three map variants are trained EXCLUSIVELY on 4b SEEN '
                           'train rows (split_s1_246 per seed, verbatim B-protocol). '
                           'forbidden = any qwen3-4b OOV activation as prediction '
                           'INPUT; 4b OOV activations appear only as the evaluation '
                           'target and in the headline-only transfer-fidelity '
                           'DIAGNOSTIC (cos vs true OOV class direction, never gates).',
    'map_variants': {
        'base': 'verbatim 3242 dual-ridge with bias (pred(v) = v@Hd^T@alpha + '
                'alpha.sum(0)); replication anchor only',
        'M1_centered': 'centered dual-ridge: Ac = Hd_tr - mu_d, Bc = Y_tr - mu_y; '
                       'K = Ac@Ac^T + lam*I, alpha = solve(K, Bc); '
                       'map(v) = (v - mu_d)@Ac^T@alpha + mu_y; no bias term',
        'M2_clscond': 'class-conditional affine on top of M1: r_c = cls_mean_c - '
                      'M1(m_c_donor_seen) per seen class; OOV correction = '
                      'sum_c softmax_1(cos_donor(m_oov-mu_d, m_c_seen-mu_d))_c * r_c; '
                      'beta=1 fixed, no temperature tuning',
        'M3_anchor_min_norm': 'multi-anchor min-norm interpolation: W3 = pinv(A6, '
                              'rcond=1e-6) @ d_cls with A6 = donor seen class '
                              'directions (m_c_donor_seen - mu_d); interpolates seen '
                              'anchors exactly, min-norm extension, no shrinkage param',
    },
    'recipes': {
        'G0': 'verbatim 3240-F2 subspace completion (replication anchor, per-seed '
              'bit-level vs sealed 3240 result.json F2)',
        'T3_rep': 'verbatim 3242 T3_dual (base dual-ridge map, joint span 6+4+4) — '
                  'per-seed bit-level anchor vs sealed 3242 result.json T3_dual',
        'M1_dual': 'joint span 6 train + 4 (14b) + 4 (glm4) transferred directions '
                   'with M1 centered maps; prediction = Yhat_ridge + P_span(B0 - '
                   'Yhat_ridge) — F2 mechanism, span extended to 14',
        'M2_dual': 'same span construction with M2 class-conditional affine maps',
        'M3_dual': 'same span construction with M3 multi-anchor min-norm maps',
        'M1r_dual': 'matched control of M1_dual: donor OOV row->class assignment '
                    'permuted (seed 42, row-level, verbatim 3242 construction) '
                    'before computing donor OOV class means',
        'M2r_dual': 'matched control of M2_dual (same row-level permutation)',
        'M3r_dual': 'matched control of M3_dual (same row-level permutation)',
    },
    'gates': {
        'G1': 'replication anchors + census spectra cross-check + G0 per-seed '
              'bit-level vs sealed 3240 F2 + T3_rep per-seed bit-level vs sealed '
              '3242 T3_dual (formal mode; smoke recorded-only per 3240 correction '
              'protocol)',
        'G2': 'efficacy: best(M1_dual, M2_dual, M3_dual) improves BOTH SCLS and '
              'ENT fidelity > +%.1f vs G0 (3-seed mean; min(d_DVU,d_DCC) > -%.2f; '
              'R2 drop < %.2f), AND beats its matched permuted control on '
              'min(SCLS, ENT)'
              % (EFF_DELTA, REG_DROP, R2_DROP),
        'G3': 'determinism: lens first-batch identical-GEMM-shape recompute diff 0.0',
        'G4': 'registry v1.33 untouched (byte sha8)',
    },
    'verdicts': {'pass': 'mapfix_registered', 'partial': 'mapfix_partial'},
    'note': 'headline (SCLS/ENT sign flip; per-variant map R2 seen/oov; transfer '
            'cos; KL ordering) is a registered finding, NOT a gate. Registry '
            'append not triggered: prediction-recipe repair, no feature-level '
            'statement change.',
}


def log(s):
    ln = '[%7.1f] %s' % (time.time() - T0, s)
    LOG.append(ln)
    try:
        print(ln, flush=True)
    except Exception:
        pass


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


def sha8_file(p):
    return sha8(open(p, 'rb').read())


def canon(o):
    return json.dumps(o, ensure_ascii=False, sort_keys=True, indent=1)


def freeze():
    eblob = canon(DESIGN).encode('utf-8')
    dsha = hashlib.sha256(eblob).hexdigest()
    os.makedirs(OUTDIR, exist_ok=True)
    exe_p = os.path.join(OUTDIR, 'execution.json' if not SMOKE else 'smoke_execution.json')
    if os.path.exists(exe_p) and not SMOKE:
        prev = json.load(open(exe_p, encoding='utf-8'))
        assert prev['design_sha8'] == dsha[:8], 'DESIGN DRIFT'
        log('freeze: existing execution.json OK (%s)' % dsha[:8])
    else:
        srcs = {k: sha8_file(v) for k, v in SRC.items()}
        json.dump({'phase': 3243, 'name': NAME, 'design_sha8': dsha[:8],
                   'design': DESIGN, 'sources': srcs,
                   'frozen_before': 'any observation', 'smoke': SMOKE,
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8', newline=''), ensure_ascii=False, indent=1)
        log('freeze: execution.json written (%s)' % dsha[:8])
    return dsha[:8]


def unit(v):
    return v / (float(np.linalg.norm(v)) + 1e-18)


def unit_mean(rows):
    v = rows.mean(0)
    return v / (float(np.linalg.norm(v)) + 1e-18)


# ---------------- 3157-derived families (verbatim from 3241) ----------------
def build_3157(m):
    A = ANCH_3157[m]
    z = np.load(SRC['p3157_4b'])
    H = z['H'].astype(np.float32)
    NROWS, NH, D = H.shape
    NL = NH - 1
    KOUT = NL - 1
    assert NROWS == 128 and (NL, KOUT) == (A['nl'], A['kout']), ('layer', m)
    EI = z['ei'].astype(int)
    RL = z['rel'].astype(int)
    PL = z['pol'].astype(int)
    CX = z['ctx'].astype(int)
    IDX = {(int(EI[i]), int(RL[i]), int(PL[i]), int(CX[i])): i for i in range(NROWS)}
    assert len(IDX) == 128
    dC = []
    for ei in range(16):
        for rl in range(2):
            for pl in range(2):
                dC.append(H[IDX[(ei, rl, pl, 1)], KOUT] - H[IDX[(ei, rl, pl, 0)], KOUT])
    dC = np.array(dC)
    assert np.isfinite(dC).all()
    dCn = dC / (np.linalg.norm(dC, axis=1, keepdims=True) + 1e-18)
    cosM = dCn @ dCn.T
    tri = cosM[np.triu_indices(len(dC), 1)]
    tc_full = float(tri.mean())
    d_tc = abs(tc_full - A['tc_mean_cos'])

    def dC_rows(rl_f):
        return [ei * 4 + rl * 2 + pl for ei in range(16) for rl in range(2)
                for pl in range(2) if rl == rl_f]

    v_isa = unit_mean(dCn[dC_rows(0)])
    v_hasa = unit_mean(dCn[dC_rows(1)])
    cos_rl = float(v_isa @ v_hasa)
    dv = v_isa - v_hasa
    norm_dv = float(np.linalg.norm(dv))
    v_full = unit_mean(dCn)
    frac_full = float((dv @ v_full) ** 2 / max(norm_dv ** 2, 1e-300))
    U = []
    for e in range(16):
        a_e = dCn[[e * 4 + 0 * 2 + p for p in (0, 1)]].mean(0)
        b_e = dCn[[e * 4 + 1 * 2 + p for p in (0, 1)]].mean(0)
        U.append(unit(a_e - b_e))
    U = np.stack(U)
    s0 = np.linalg.svd(U, compute_uv=False)
    pc_share = float(s0[0] ** 2 / (s0 ** 2).sum())
    Un = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-18)
    C = Un @ Un.T
    med_cos = float(np.median(C[np.triu_indices(16, 1)]))
    ENTm = np.stack([unit_mean(H[[IDX[(e, rl, pl, cx)] for rl in range(2)
                                 for pl in range(2) for cx in range(2)], KOUT])
                     for e in range(16)])
    del z, H
    return dict(DCC=dCn, DVU=U, ENT=ENTm, D=D, kout=KOUT,
                anchors={'tc_full': tc_full, 'cos_rl': cos_rl, 'norm_dv': norm_dv,
                         'frac_full': frac_full, 'pc_share': pc_share, 'med_cos': med_cos},
                d_tc=d_tc)


def split_s1_246(seed):
    SEEN_PAIR_SET = set((i, c) for i in range(NTR_ENT) for c in range(NCS))
    pairs246 = sorted(SEEN_PAIR_SET)
    assert len(pairs246) == 246
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(pairs246))
    n_test = int(round(FRAC_S1 * len(pairs246)))
    test = set(pairs246[j] for j in idx[:n_test])
    return SEEN_PAIR_SET - test, test


def ridge_primal(Xtr, Ytr, lam=LAM):
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1], dtype=np.float32)
    W = np.linalg.solve(A, Xtr.T @ Ytr)
    return W


def phi_row(i, c, t):
    v = np.zeros(NE + NC + NT + 1, np.float32)
    v[i] = 1.0
    v[NE + c] = 1.0
    v[NE + NC + t] = 1.0
    v[-1] = 1.0
    return v


def build_3169(m, sealed_3169, sealed_3188, sealed_3189):
    rd = int(sealed_3169['per_model'][m]['readout'])
    z = np.load(SRC['p3169_4b'])
    H16 = z['H']
    NTz, NPz, NHz, Dz = H16.shape
    assert NTz == NT and NPz == NP_, ('shape', H16.shape)
    kout = NHz - 2
    assert kout == rd, ('readout slot', m, kout, rd)
    Y = H16[:, :, kout, :].reshape(NT * NP_, Dz).astype(np.float32)
    assert np.isfinite(Y).all()

    tr_base = {}
    te_seen_all = {}
    for s in SEEDS_ALL:
        tr_pairs, te_pairs = split_s1_246(s)
        tr_base[s] = [t * NP_ + PI_OF_PAIR[p] for t in range(NT)
                      for p in PAIRS if p in tr_pairs]
        te_seen_all[s] = [t * NP_ + PI_OF_PAIR[p] for t in range(NT)
                          for p in PAIRS if p in te_pairs]
    oov_rows = ALL_OOV_ROWS

    E69 = sealed_3169['per_model'][m]['B']
    E88 = sealed_3188['per_model'][m]
    drift69 = 0.0
    Q_acc = np.zeros((len(oov_rows), Dz), dtype=np.float32)
    proj_oov_acc = np.zeros(len(oov_rows))
    proj_seen_acc = {}
    share_oov_acc = []
    share_seen_acc = []
    for si, s in enumerate(SEEDS_ALL):
        tr_rows = tr_base[s]
        te_rows = te_seen_all[s]
        Xtr = np.stack([phi_row(PAIRS[r % NP_][0], PAIRS[r % NP_][1], r // NP_)
                        for r in tr_rows])
        W = ridge_primal(Xtr, Y[tr_rows])
        ref = Y[tr_rows].mean(0)
        Dk = float(((Y[tr_rows] - ref) ** 2).sum(1).mean()) + 1e-9
        Xs = np.stack([phi_row(PAIRS[r % NP_][0], PAIRS[r % NP_][1], r // NP_)
                       for r in te_rows])
        es = ((Xs @ W - Y[te_rows]) ** 2).sum(1) / Dk
        Xo = np.stack([phi_row(PAIRS[r % NP_][0], PAIRS[r % NP_][1], r // NP_)
                       for r in oov_rows])
        eo = ((Xo @ W - Y[oov_rows]) ** 2).sum(1) / Dk
        drift69 = max(drift69, abs(float(es.mean()) - E69['E_seen_per_seed'][si]),
                      abs(float(eo.mean()) - E69['E_oov_per_seed'][si]))
        R_oov = Y[oov_rows] - Xo @ W
        for j in range(len(oov_rows)):
            rvv = R_oov[j]
            nv = float(np.linalg.norm(rvv))
            Q_acc[j] += rvv / (nv + 1e-18)
            pr = float(rvv @ sealed_3188_dvu.get(m, np.zeros(Dz, np.float32)))
            proj_oov_acc[j] += pr / (nv + 1e-18)
            share_oov_acc.append(pr * pr / max(nv * nv, 1e-300))
        R_seen = Y[te_rows] - Xs @ W
        for j, r in enumerate(te_rows):
            rvv = R_seen[j]
            nv = float(np.linalg.norm(rvv))
            pr = float(rvv @ sealed_3188_dvu.get(m, np.zeros(Dz, np.float32)))
            proj_seen_acc.setdefault(r, []).append(pr / (nv + 1e-18))
            share_seen_acc.append(pr * pr / max(nv * nv, 1e-300))
        del W, Xtr
    Qn = Q_acc / len(SEEDS_ALL)
    Qn = Qn / (np.linalg.norm(Qn, axis=1, keepdims=True) + 1e-18)
    proj_oov = proj_oov_acc / len(SEEDS_ALL)
    proj_seen = np.array([float(np.mean(v)) for v in proj_seen_acc.values()])

    def ks_stat(a, b):
        a = np.sort(np.asarray(a, dtype=np.float64))
        b = np.sort(np.asarray(b, dtype=np.float64))
        allv = np.concatenate([a, b])
        cdf_a = np.searchsorted(a, allv, side='right') / float(len(a))
        cdf_b = np.searchsorted(b, allv, side='right') / float(len(b))
        return float(np.max(np.abs(cdf_a - cdf_b)))

    ks = ks_stat(proj_oov, proj_seen)
    med_oov = float(np.median(proj_oov))
    med_seen = float(np.median(proj_seen))
    share_oov = float(np.mean(share_oov_acc))
    share_seen = float(np.mean(share_seen_acc))
    d88 = max(abs(share_oov - E88['share_oov']), abs(share_seen - E88['share_seen']),
              abs(ks - E88['ks']), abs(med_oov - E88['med_oov']),
              abs(med_seen - E88['med_seen']))
    sv = np.linalg.svd(Qn, compute_uv=False)
    lam_ = sv ** 2
    top1 = float(lam_[0] / lam_.sum())
    top2 = float(lam_[:2].sum() / lam_.sum())
    CLS_ROW = {}
    for j, r in enumerate(oov_rows):
        CLS_ROW.setdefault(PAIRS[r % NP_][1], []).append(j)
    cls_dirs = [unit_mean(Qn[CLS_ROW[c]]) for c in sorted(CLS_ROW)]
    cross = []
    for a in range(len(cls_dirs)):
        for b in range(a + 1, len(cls_dirs)):
            cross.append(float(cls_dirs[a] @ cls_dirs[b]))
    cross_med = float(np.median(cross))
    E89 = sealed_3189['per_model'][m]
    d89 = max(abs(top1 - E89['top1']), abs(top2 - E89['top2']),
              abs(cross_med - E89['cross_med']))
    SCLS = []
    for c in range(NC):
        rows_c = [t * NP_ + PI_OF_PAIR[(i, c)] for t in range(NT) for i in range(NE)]
        SCLS.append(unit_mean(Y[rows_c]))
    SCLS = np.stack(SCLS)
    del z, H16
    return dict(Y=Y, tr_base=tr_base, te_seen_all=te_seen_all, oov_rows=oov_rows,
                USTAR=Qn, SCLS=SCLS, D=Dz, kout=kout,
                anchors={'drift69': drift69, 'd88': d88, 'd89': d89,
                         'top1': top1, 'cross_med': cross_med})


sealed_3188_dvu = {}


def spectrum(M):
    sv = np.linalg.svd(np.asarray(M, np.float64), compute_uv=False)
    lam = sv ** 2
    tot = lam.sum()
    pr = float(tot * tot / (lam ** 2).sum() + 1e-300)
    cum = np.cumsum(lam) / tot
    k90 = int(np.searchsorted(cum, 0.9) + 1)
    top1 = float(lam[0] / tot)
    return dict(PR=pr, k90=k90, top1=top1, n=int(len(sv)))


def subspace_basis(M, k):
    _, _, Vt = np.linalg.svd(np.asarray(M, np.float64), full_matrices=False)
    return Vt[:k]


# ---------------- CPU lens (4b only) ----------------
LENS = {}


def load_lens():
    from safetensors import safe_open
    cfg = json.load(open(SRC['hf_config'], encoding='utf-8'))
    eps = float(cfg['rms_norm_eps'])
    with safe_open(SRC['hf_shard_emb'], framework='pt', device='cpu') as f:
        E = f.get_tensor('model.embed_tokens.weight').float().numpy()
    with safe_open(SRC['hf_shard_norm'], framework='pt', device='cpu') as f:
        gamma = f.get_tensor('model.norm.weight').float().numpy()
    assert E.shape == (151936, 2560) and gamma.shape == (2560,), (E.shape, gamma.shape)
    LENS['E'] = np.ascontiguousarray(E)
    LENS['gamma'] = gamma
    LENS['eps'] = eps
    LENS['V'] = int(E.shape[0])
    del E
    log('lens loaded: E %s gamma %s eps=%g (emb shard sha8 %s)'
        % (LENS['E'].shape, LENS['gamma'].shape, eps, sha8_file(SRC['hf_shard_emb'])))


def lens_logp(Hrows, batch=128):
    E, gamma, eps = LENS['E'], LENS['gamma'], LENS['eps']
    n = Hrows.shape[0]
    out = np.empty((n, LENS['V']), np.float32)
    for i0 in range(0, n, batch):
        h = Hrows[i0:i0 + batch].astype(np.float32)
        rms = np.sqrt((h * h).mean(1) + eps).astype(np.float32)
        nh = h / rms[:, None] * gamma[None, :]
        lg = nh @ E.T
        m = lg.max(1, keepdims=True)
        lse = m + np.log(np.exp(lg - m).sum(1, keepdims=True))
        out[i0:i0 + batch] = lg - lse
    return out


def kl_rows(logp_a, logp_p):
    p_a = np.exp(logp_a.astype(np.float64))
    kl = (p_a * (logp_a.astype(np.float64) - logp_p.astype(np.float64))).sum(1)
    return kl


# ---------------- donor loading + dual-ridge transfer map ----------------
def donor_npz_key(dname):
    return 'p3169_14b' if '14' in dname else 'p3169_glm'


def load_donor_kout(key):
    z = np.load(SRC[donor_npz_key(key)])
    H = z['H']
    NTz, NPz, NHz, Dz = H.shape
    assert NTz == NT and NPz == NP_, ('donor shape', key, H.shape)
    kout = NHz - 2
    Yd = H[:, :, kout, :].reshape(NT * NP_, Dz).astype(np.float32)
    del H, z
    assert np.isfinite(Yd).all()
    return Yd, kout, Dz


def ridge_dual_fit(Hd_tr, Ytr, lam=LAM):
    """dual ridge with bias: pred(v) = v@Hd^T@alpha + alpha.sum(0).
    Equivalent to primal ridge on [Hd, 1] -> Y with the same lam."""
    n = len(Hd_tr)
    K = (Hd_tr @ Hd_tr.T).astype(np.float64)
    K[np.diag_indices(n)] += lam
    K += 1.0
    alpha = np.linalg.solve(K, Ytr.astype(np.float64)).astype(np.float32)
    return Hd_tr, alpha


def ridge_dual_apply(Hd_tr, alpha, V):
    return (V @ Hd_tr.T @ alpha + alpha.sum(0, keepdims=True)).astype(np.float32)


def softmax_weights(cosv, beta=1.0):
    """rowwise softmax with fixed beta (prereg: no temperature tuning)"""
    e = np.exp(beta * (cosv - cosv.max(axis=1, keepdims=True)))
    return e / e.sum(axis=1, keepdims=True)


def main():
    dsha = freeze()
    log('design_sha8=%s smoke=%s seeds=%s' % (dsha[:8], SMOKE, SEEDS_RUN))
    sealed_3169 = json.load(open(SRC['p3169_result'], encoding='utf-8'))
    sealed_3183 = json.load(open(SRC['p3183_result'], encoding='utf-8'))
    sealed_3188 = json.load(open(SRC['p3188_result'], encoding='utf-8'))
    sealed_3189 = json.load(open(SRC['p3189_result'], encoding='utf-8'))
    sealed_3240 = json.load(open(SRC['p3240_result'], encoding='utf-8'))
    sealed_3242 = json.load(open(SRC['p3242_result'], encoding='utf-8'))
    reg33_sha_sealed = sha8_file(SRC['registry_v133'])

    m = MODEL
    b57 = build_3157(m)
    dCn = b57['DCC']

    def dC_rows(rl_f):
        return [ei * 4 + rl * 2 + pl for ei in range(16) for rl in range(2)
                for pl in range(2) if rl == rl_f]

    dv_vec = unit_mean(dCn[dC_rows(0)]) - unit_mean(dCn[dC_rows(1)])
    dv_vec = dv_vec / (float(np.linalg.norm(dv_vec)) + 1e-18)
    sealed_3188_dvu[m] = dv_vec
    E83 = sealed_3183['per_model'][m]
    a = b57['anchors']
    d83 = max(abs(a['cos_rl'] - E83['cos_rl']), abs(a['norm_dv'] - E83['norm_dv']),
              abs(a['frac_full'] - E83['frac_full']),
              abs(a['pc_share'] - E83['pc_share']),
              abs(a['med_cos'] - E83['med_cos']),
              abs(a['tc_full'] - E83['tc_full']))
    log('3157 anchors: tc=%.9f d83=%.3e' % (a['tc_full'], d83))
    b69 = build_3169(m, sealed_3169, sealed_3188, sealed_3189)
    log('3169 anchors: drift69=%.3e d88=%.3e d89=%.3e'
        % (b69['anchors']['drift69'], b69['anchors']['d88'], b69['anchors']['d89']))

    v32_path = os.path.join(SRC_DIR, r'phase3237\g5g31_registry_landing\atlas_registry_v1_32.json')
    ftr12 = [f for f in json.load(open(v32_path, encoding='utf-8'))['features']
             if f['id'] == 'FTR-12'][0]
    d1 = int(ftr12['values']['d1_4b'])
    massive = np.zeros((1, b57['D']), np.float32)
    massive[0, d1] = 1.0
    fams = {'DVU': b57['DVU'], 'DCC': b57['DCC'], 'ENT': b57['ENT'],
            'USTAR': b69['USTAR'], 'SCLS': b69['SCLS'], 'MASSIVE': massive}

    zc = np.load(SRC['census_npz'])
    spectra = {}
    bases = {}
    for fam in FAMS:
        met = spectrum(fams[fam])
        meta_sealed = zc['%s_%s_meta' % (m, fam)]
        k90_ok = (met['k90'] == int(round(meta_sealed[2])))
        pr_ok = abs(met['PR'] - float(meta_sealed[0])) < F32_TOL
        top_ok = abs(met['top1'] - float(meta_sealed[3])) < F32_TOL
        spectra[fam] = {**met, 'k90_match_sealed': bool(k90_ok),
                        'pr_match_sealed': bool(pr_ok), 'top1_match_sealed': bool(top_ok)}
        bases[fam] = subspace_basis(fams[fam], met['k90'])
        log('spec %s: PR=%.4f k90=%d top1=%.6f sealed_match=%s'
            % (fam, met['PR'], met['k90'], met['top1'], k90_ok and pr_ok and top_ok))

    load_lens()
    Y = b69['Y']
    oov_rows = b69['oov_rows']
    kl_n = len(oov_rows) if not SMOKE else min(KL_ROWS_SMOKE, len(oov_rows))
    kl_rows_idx = oov_rows[:kl_n]
    log('KL rows: %d (of %d OOV)' % (kl_n, len(oov_rows)))

    # donors: load kout activations once
    donors = {}
    for dname in set(DONORS.values()):
        Yd, kd, Dd = load_donor_kout(dname)
        donors[dname] = dict(Y=Yd, kout=kd, D=Dd)
        log('donor %s: kout=%d D=%d (sha8 npz %s)'
            % (dname, kd, Dd, sha8_file(SRC[donor_npz_key(dname)])))

    # donor OOV row grouping per class (real labels)
    don_oov_rows_c = {c: [r for r in oov_rows if PAIRS[r % NP_][1] == c]
                      for c in OOV_CLASSES}

    # preregistered control: ROW-LEVEL permutation of donor OOV class labels
    # (seed 42). SMOKE-stage correction: a class-level permutation of the 4
    # OOV class means only reorders the transferred set (identical span ->
    # identical projection, T1r==T1 verbatim in first SMOKE); the control must
    # destroy the row->class assignment instead.
    rng_perm = np.random.RandomState(RAND_SEED)
    oov_sizes = {c: len(don_oov_rows_c[c]) for c in OOV_CLASSES}
    all_oov_shuffled = rng_perm.permutation(oov_rows)   # row-level shuffle
    perm_rows_c = {}
    pos = 0
    for c in OOV_CLASSES:
        perm_rows_c[c] = sorted(all_oov_shuffled[pos:pos + oov_sizes[c]])
        pos += oov_sizes[c]
    assert pos == len(oov_rows)

    per_seed = {}
    det_diff = None
    for s in SEEDS_RUN:
        tr_rows = b69['tr_base'][s]
        te_rows = b69['te_seen_all'][s]
        Xtr = np.stack([phi_row(PAIRS[r % NP_][0], PAIRS[r % NP_][1], r // NP_)
                        for r in tr_rows])
        W = ridge_primal(Xtr, Y[tr_rows])
        B0 = Y[tr_rows].mean(0)
        cls_mean = {}
        for r in tr_rows:
            cls_mean.setdefault(PAIRS[r % NP_][1], []).append(Y[r])
        cls_mean = {c: np.stack(v).mean(0) for c, v in cls_mean.items()}
        d_cls = np.stack([cls_mean[c] - B0 for c in range(NCS)])          # (6, d)
        Qs, _ = np.linalg.qr(d_cls.T.astype(np.float32))
        Qs = Qs[:, :NCS].astype(np.float32)

        Xo = np.stack([phi_row(PAIRS[r % NP_][0], PAIRS[r % NP_][1], r // NP_)
                       for r in kl_rows_idx])
        Yhat0 = Xo @ W
        Yact = Y[kl_rows_idx]
        Yb0 = np.stack([B0] * kl_n)
        mse_b0 = float(((Yact - Yb0) ** 2).sum(1).mean())

        # true 4b OOV class directions (EVALUATION-ONLY diagnostic, never input)
        d_true = {}
        for c in OOV_CLASSES:
            rows_c = [r for r in oov_rows if PAIRS[r % NP_][1] == c]
            d_true[c] = unit(Y[rows_c].mean(0) - B0)

        # per-donor maps: base (3242 verbatim) + M1/M2/M3 variants.
        # All maps trained on seen train rows only (per leakage declaration).
        d_hat = {}          # variant -> {'14b': (4,D), 'glm': (4,D), key+'_perm': ...}
        d_hat['base'] = {}
        d_hat['M1'] = {}
        d_hat['M2'] = {}
        d_hat['M3'] = {}
        for k in ('14b', 'glm'):
            d_hat['base'][k + '_perm'] = None
            d_hat['M1'][k + '_perm'] = None
            d_hat['M2'][k + '_perm'] = None
            d_hat['M3'][k + '_perm'] = None
        xfer_diag = {}
        for dk, dname in DONORS.items():
            Yd = donors[dname]['Y']
            Hd_tr = Yd[tr_rows]
            mu_d = Hd_tr.mean(0)
            # ---- base: verbatim 3242 dual-ridge with bias ----
            Hd_b, alpha_b = ridge_dual_fit(Hd_tr, Y[tr_rows])
            m_seen = np.stack([Yd[[r for r in tr_rows
                                   if PAIRS[r % NP_][1] == c]].mean(0)
                               for c in range(NCS)])                  # (6, Dd)
            m_oov = np.stack([Yd[don_oov_rows_c[c]].mean(0) for c in OOV_CLASSES])
            m_oov_perm = np.stack([Yd[perm_rows_c[c]].mean(0) for c in OOV_CLASSES])
            d_hat['base'][dk] = (ridge_dual_apply(Hd_b, alpha_b, m_oov)
                                 - B0[None, :])
            d_hat['base'][dk + '_perm'] = (ridge_dual_apply(Hd_b, alpha_b, m_oov_perm)
                                           - B0[None, :])
            # map-quality diagnostics (base): seen-test + OOV extrapolation
            pred_te = ridge_dual_apply(Hd_b, alpha_b, Yd[te_rows])
            dk_ = float(((Y[te_rows] - Y[te_rows].mean(0)) ** 2).sum(1).mean()) + 1e-9
            r2_seen = float(1.0 - ((pred_te - Y[te_rows]) ** 2).sum(1).mean() / dk_)
            pred_oov = ridge_dual_apply(Hd_b, alpha_b, Yd[oov_rows])
            r2_oov = float(1.0 - ((pred_oov - Y[oov_rows]) ** 2).sum(1).mean() / dk_)
            cos_seen = [float(unit(ridge_dual_apply(Hd_b, alpha_b,
                                                    m_seen[c:c + 1])[0] - B0)
                               @ unit(d_cls[c])) for c in range(NCS)]
            diag = {'base': {'r2_map_seen': r2_seen, 'r2_map_oov': r2_oov,
                             'cos_seen_train_dirs': cos_seen}}
            # ---- M1: centered dual-ridge (no bias) ----
            Ac = Hd_tr - mu_d[None, :]
            Kc = (Ac @ Ac.T).astype(np.float64)
            Kc[np.diag_indices(len(Ac))] += LAM
            alpha_c = np.linalg.solve(Kc, (Y[tr_rows] - B0[None, :]).astype(np.float64)
                                      ).astype(np.float32)

            def map_m1(V, Ac=Ac, alpha_c=alpha_c, mu_d=mu_d):
                return ((V - mu_d[None, :]) @ Ac.T @ alpha_c
                        + B0[None, :]).astype(np.float32)

            d_hat['M1'][dk] = map_m1(m_oov) - B0[None, :]
            d_hat['M1'][dk + '_perm'] = map_m1(m_oov_perm) - B0[None, :]
            pred_te1 = map_m1(Yd[te_rows])
            r2_seen1 = float(1.0 - ((pred_te1 - Y[te_rows]) ** 2).sum(1).mean() / dk_)
            pred_oov1 = map_m1(Yd[oov_rows])
            r2_oov1 = float(1.0 - ((pred_oov1 - Y[oov_rows]) ** 2).sum(1).mean() / dk_)
            diag['M1'] = {'r2_map_seen': r2_seen1, 'r2_map_oov': r2_oov1}
            # ---- M2: class-conditional affine on top of M1 ----
            m_seen_m1 = map_m1(m_seen)                                # (6, D)
            r_res = np.stack([cls_mean[c] - m_seen_m1[c] for c in range(NCS)])
            d_seen_donor = m_seen - mu_d[None, :]                     # (6, Dd)
            d_oov_donor = m_oov - mu_d[None, :]
            d_oov_donor_perm = m_oov_perm - mu_d[None, :]
            nrm = np.linalg.norm(d_seen_donor, axis=1) + 1e-18
            cos_sm = (d_oov_donor @ d_seen_donor.T) / (nrm[None, :]
                       * (np.linalg.norm(d_oov_donor, axis=1)[:, None] + 1e-18))
            cos_sm_p = (d_oov_donor_perm @ d_seen_donor.T) / (nrm[None, :]
                         * (np.linalg.norm(d_oov_donor_perm, axis=1)[:, None] + 1e-18))
            w_sm = softmax_weights(cos_sm)                            # (4, 6) rowwise
            w_sm_p = softmax_weights(cos_sm_p)
            d_hat['M2'][dk] = d_hat['M1'][dk] + w_sm @ r_res
            d_hat['M2'][dk + '_perm'] = d_hat['M1'][dk + '_perm'] + w_sm_p @ r_res
            diag['M2'] = {'res_norm_mean': float(np.linalg.norm(r_res, axis=1).mean()),
                          'w_entropy_mean': float(
                              -(w_sm * np.log(w_sm + 1e-300)).sum(1).mean())}
            # ---- M3: multi-anchor min-norm interpolation ----
            # anchors: rows of X6 = donor seen class dirs (6, Dd) -> 4b dirs (6, D)
            X6 = d_seen_donor.astype(np.float64)
            W3 = (np.linalg.pinv(X6, rcond=1e-6)
                  @ d_cls.astype(np.float64)).astype(np.float32)      # (Dd, D)
            d_hat['M3'][dk] = (d_oov_donor @ W3).astype(np.float32)
            d_hat['M3'][dk + '_perm'] = (d_oov_donor_perm @ W3).astype(np.float32)
            # anchor reproduction sanity (headline diagnostic)
            anchor_rep = float(np.max(np.abs(X6 @ W3.astype(np.float64)
                                             - d_cls.astype(np.float64))))
            diag['M3'] = {'anchor_repro_max': anchor_rep}
            # ---- transfer-fidelity diagnostics (eval-only, headline) ----
            d_true_mat = np.stack([d_true[c] for c in OOV_CLASSES])   # (4, D)
            for vn in ('base', 'M1', 'M2', 'M3'):
                cr = [float(unit(d_hat[vn][dk][j]) @ d_true_mat[j])
                      for j in range(len(OOV_CLASSES))]
                cp = [float(unit(d_hat[vn][dk + '_perm'][j]) @ d_true_mat[j])
                      for j in range(len(OOV_CLASSES))]
                diag[vn]['cos_transfer_real'] = cr
                diag[vn]['cos_transfer_perm'] = cp
            xfer_diag[dname] = diag
            log('seed %d donor %s: base R2 s/oov=%.4f/%.4f cos=%.3f/%.3f | '
                'M1 R2=%.4f/%.4f cos=%.3f/%.3f | M2 cos=%.3f/%.3f | '
                'M3 cos=%.3f/%.3f anchor_rep=%.2e' %
                (s, dname, r2_seen, r2_oov, float(np.mean(diag['base']['cos_transfer_real'])),
                 float(np.mean(diag['base']['cos_transfer_perm'])),
                 r2_seen1, r2_oov1, float(np.mean(diag['M1']['cos_transfer_real'])),
                 float(np.mean(diag['M1']['cos_transfer_perm'])),
                 float(np.mean(diag['M2']['cos_transfer_real'])),
                 float(np.mean(diag['M2']['cos_transfer_perm'])),
                 float(np.mean(diag['M3']['cos_transfer_real'])),
                 float(np.mean(diag['M3']['cos_transfer_perm'])), anchor_rep))

        # spans
        def qr_span(mat_rows):
            Q, _ = np.linalg.qr(mat_rows.T.astype(np.float32))
            return Q[:, :mat_rows.shape[0]].astype(np.float32)

        Q_G0 = Qs
        Q_T3rep = qr_span(np.concatenate([d_cls, d_hat['base']['14b'],
                                          d_hat['base']['glm']], 0))
        Q_M1 = qr_span(np.concatenate([d_cls, d_hat['M1']['14b'],
                                       d_hat['M1']['glm']], 0))
        Q_M2 = qr_span(np.concatenate([d_cls, d_hat['M2']['14b'],
                                       d_hat['M2']['glm']], 0))
        Q_M3 = qr_span(np.concatenate([d_cls, d_hat['M3']['14b'],
                                       d_hat['M3']['glm']], 0))
        Q_M1r = qr_span(np.concatenate([d_cls, d_hat['M1']['14b_perm'],
                                        d_hat['M1']['glm_perm']], 0))
        Q_M2r = qr_span(np.concatenate([d_cls, d_hat['M2']['14b_perm'],
                                        d_hat['M2']['glm_perm']], 0))
        Q_M3r = qr_span(np.concatenate([d_cls, d_hat['M3']['14b_perm'],
                                        d_hat['M3']['glm_perm']], 0))
        QMAP = {'G0': Q_G0, 'T3_rep': Q_T3rep, 'M1_dual': Q_M1, 'M2_dual': Q_M2,
                'M3_dual': Q_M3, 'M1r_dual': Q_M1r, 'M2r_dual': Q_M2r,
                'M3r_dual': Q_M3r}

        Ypreds = {}
        for rn in RECIPES:
            Qp = QMAP[rn]
            Ypreds[rn] = (Yhat0 + (B0[None, :] - Yhat0) @ Qp @ Qp.T).astype(np.float32)

        seed_rec = {}
        for rn in RECIPES:
            P = Ypreds[rn]
            mse_r = float(((Yact - P) ** 2).sum(1).mean())
            r2 = 1.0 - mse_r / mse_b0
            Rr = Yact - P
            fam_fid = {}
            for fam in FAMS:
                Bf = bases[fam].astype(np.float32)
                em_ = float((np.linalg.norm(Rr @ Bf.T, axis=1) ** 2).mean())
                eb_ = float((np.linalg.norm((Yact - Yb0) @ Bf.T, axis=1) ** 2).mean())
                fam_fid[fam] = 1.0 - em_ / max(eb_, 1e-300)
            seed_rec[rn] = {'r2_oov': r2, 'mse': mse_r, 'fam_fid': fam_fid}
        seed_rec['G0']['r2_seen'] = None   # seen-test unchanged by construction
        # KL for all recipes (headline)
        la = lens_logp(Yact)
        if det_diff is None:
            nb = min(BATCH_DEFAULT, kl_n)
            la2 = lens_logp(Yact[:nb])
            det_diff = float(np.max(np.abs(la[:nb] - la2)))
        for rn in RECIPES:
            lp = lens_logp(Ypreds[rn])
            klr = kl_rows(la, lp)
            seed_rec[rn]['kl_mean'] = float(klr.mean())
            seed_rec[rn]['kl_med'] = float(np.median(klr))
        # G0 replication vs sealed 3240 F2 (formal rows only)
        ps40 = sealed_3240['per_seed'][str(s)]['F2']
        d40 = max(abs(seed_rec['G0']['r2_oov'] - ps40['r2_oov']),
                  abs(seed_rec['G0']['kl_mean'] - ps40['kl_mean']),
                  max(abs(seed_rec['G0']['fam_fid'][f] - ps40['fam_fid'][f])
                      for f in FAMS))
        seed_rec['G0']['d_vs_sealed_3240'] = d40
        # T3_rep anchor vs sealed 3242 T3_dual (formal rows only)
        ps42 = sealed_3242['per_seed'][str(s)]['T3_dual']
        d42 = max(abs(seed_rec['T3_rep']['r2_oov'] - ps42['r2_oov']),
                  abs(seed_rec['T3_rep']['kl_mean'] - ps42['kl_mean']),
                  max(abs(seed_rec['T3_rep']['fam_fid'][f] - ps42['fam_fid'][f])
                      for f in FAMS))
        seed_rec['T3_rep']['d_vs_sealed_3242'] = d42
        seed_rec['_xfer_diag'] = xfer_diag
        per_seed[s] = seed_rec
        log('seed %d: G0 d40=%.3e | T3rep d42=%.3e | '
            'M1 r2=%.4f SCLS=%.3f ENT=%.3f | '
            'M2 r2=%.4f SCLS=%.3f ENT=%.3f | M3 r2=%.4f SCLS=%.3f ENT=%.3f | '
            'M1r SCLS=%.3f M2r SCLS=%.3f M3r SCLS=%.3f' %
            (s, d40, d42,
             seed_rec['M1_dual']['r2_oov'], seed_rec['M1_dual']['fam_fid']['SCLS'],
             seed_rec['M1_dual']['fam_fid']['ENT'],
             seed_rec['M2_dual']['r2_oov'], seed_rec['M2_dual']['fam_fid']['SCLS'],
             seed_rec['M2_dual']['fam_fid']['ENT'],
             seed_rec['M3_dual']['r2_oov'], seed_rec['M3_dual']['fam_fid']['SCLS'],
             seed_rec['M3_dual']['fam_fid']['ENT'],
             seed_rec['M1r_dual']['fam_fid']['SCLS'],
             seed_rec['M2r_dual']['fam_fid']['SCLS'],
             seed_rec['M3r_dual']['fam_fid']['SCLS']))
        log('seed %d: KL G0=%.4f T3rep=%.4f M1=%.4f M2=%.4f M3=%.4f '
            'M1r=%.4f M2r=%.4f M3r=%.4f' %
            (s, seed_rec['G0']['kl_mean'], seed_rec['T3_rep']['kl_mean'],
             seed_rec['M1_dual']['kl_mean'], seed_rec['M2_dual']['kl_mean'],
             seed_rec['M3_dual']['kl_mean'], seed_rec['M1r_dual']['kl_mean'],
             seed_rec['M2r_dual']['kl_mean'], seed_rec['M3r_dual']['kl_mean']))
        del W, Xtr, la

    agg = {}
    for rn in RECIPES:
        agg[rn] = {
            'r2_oov_mean': float(np.mean([per_seed[s][rn]['r2_oov'] for s in SEEDS_RUN])),
            'kl_mean': float(np.mean([per_seed[s][rn]['kl_mean'] for s in SEEDS_RUN])),
            'kl_med': float(np.mean([per_seed[s][rn]['kl_med'] for s in SEEDS_RUN])),
            'fam_fid_mean': {f: float(np.mean([per_seed[s][rn]['fam_fid'][f]
                                               for s in SEEDS_RUN])) for f in FAMS},
        }
    agg['G0']['d_vs_sealed_3240_max'] = float(max(per_seed[s]['G0']['d_vs_sealed_3240']
                                                  for s in SEEDS_RUN))
    agg['T3_rep']['d_vs_sealed_3242_max'] = float(max(
        per_seed[s]['T3_rep']['d_vs_sealed_3242'] for s in SEEDS_RUN))
    agg['transfer_diag'] = {}
    for dname in set(DONORS.values()):
        variants = list(per_seed[SEEDS_RUN[0]]['_xfer_diag'][dname].keys())
        dv = {}
        for vn in variants:
            keys = list(per_seed[SEEDS_RUN[0]]['_xfer_diag'][dname][vn].keys())
            dd = {}
            for k in keys:
                vals = [per_seed[s]['_xfer_diag'][dname][vn][k] for s in SEEDS_RUN]
                dd[k] = (float(np.mean(vals)) if isinstance(vals[0], (int, float))
                         else [float(np.mean([v[i] for v in vals]))
                               for i in range(len(vals[0]))])
            dv[vn] = dd
        agg['transfer_diag'][dname] = dv
    delta = {}
    for rn in RECIPES[1:]:
        delta[rn] = {
            'd_fam_fid': {f: agg[rn]['fam_fid_mean'][f] - agg['G0']['fam_fid_mean'][f]
                          for f in FAMS},
            'd_r2_oov': agg[rn]['r2_oov_mean'] - agg['G0']['r2_oov_mean'],
            'd_kl_mean': agg[rn]['kl_mean'] - agg['G0']['kl_mean'],
        }
    log('AGG G0: r2=%.4f SCLS=%.3f ENT=%.3f' %
        (agg['G0']['r2_oov_mean'], agg['G0']['fam_fid_mean']['SCLS'],
         agg['G0']['fam_fid_mean']['ENT']))
    for rn in RECIPES[1:]:
        log('AGG %s: r2=%.4f (%+.4f) SCLS=%.3f (%+.3f) ENT=%.3f (%+.3f) '
            'DVU=%+.3f DCC=%+.3f KL=%.4f (%+.4f)' %
            (rn, agg[rn]['r2_oov_mean'], delta[rn]['d_r2_oov'],
             agg[rn]['fam_fid_mean']['SCLS'], delta[rn]['d_fam_fid']['SCLS'],
             agg[rn]['fam_fid_mean']['ENT'], delta[rn]['d_fam_fid']['ENT'],
             delta[rn]['d_fam_fid']['DVU'], delta[rn]['d_fam_fid']['DCC'],
             agg[rn]['kl_mean'], delta[rn]['d_kl_mean']))

    # gates
    g1 = (all(v < REPL_TOL for v in (d83, b57['d_tc'], b69['anchors']['drift69'],
                                     b69['anchors']['d88']))
          and b69['anchors']['d89'] < 1e-6
          and all(spectra[f]['k90_match_sealed'] and spectra[f]['pr_match_sealed']
                  and spectra[f]['top1_match_sealed'] for f in FAMS)
          and (agg['G0']['d_vs_sealed_3240_max'] < REPL_TOL if not SMOKE else True)
          and (agg['T3_rep']['d_vs_sealed_3242_max'] < REPL_TOL if not SMOKE else True))
    cand = {rn: min(delta[rn]['d_fam_fid']['SCLS'], delta[rn]['d_fam_fid']['ENT'])
            for rn in ('M1_dual', 'M2_dual', 'M3_dual')}
    best = max(cand, key=cand.get)
    ctrls = CTRL_OF[best]
    ctrl_val = min(min(agg[ct]['fam_fid_mean']['SCLS'],
                       agg[ct]['fam_fid_mean']['ENT']) for ct in ctrls)
    g2 = bool(cand[best] > EFF_DELTA
              and min(delta[best]['d_fam_fid']['DVU'],
                      delta[best]['d_fam_fid']['DCC']) > -REG_DROP
              and -delta[best]['d_r2_oov'] < R2_DROP
              and min(agg[best]['fam_fid_mean']['SCLS'],
                      agg[best]['fam_fid_mean']['ENT']) > ctrl_val)
    g3 = (det_diff == 0.0)
    g4 = (reg33_sha_sealed == sha8(open(SRC['registry_v133'], 'rb').read()))
    d40_ok = (agg['G0']['d_vs_sealed_3240_max'] < REPL_TOL) if not SMOKE else None
    d42_ok = (agg['T3_rep']['d_vs_sealed_3242_max'] < REPL_TOL) if not SMOKE else None
    gates = {'G1_replication': bool(g1), 'G2_efficacy': g2,
             'G3_determinism': bool(g3), 'G4_registry_untouched': bool(g4),
             'determinism_diff': det_diff, 'best_recipe': best,
             'd40_vs_sealed_3240': d40_ok, 'd42_vs_sealed_3242': d42_ok}
    allg = [v for k, v in gates.items() if k.startswith('G')]
    verdict = DESIGN['verdicts']['pass'] if all(allg) else DESIGN['verdicts']['partial']
    log('VERDICT: %s gates=%s' % (verdict, gates))

    headline = {
        'scls_sign_flipped': {rn: bool(agg[rn]['fam_fid_mean']['SCLS'] > 0)
                              for rn in ('M1_dual', 'M2_dual', 'M3_dual')},
        'ent_sign_flipped': {rn: bool(agg[rn]['fam_fid_mean']['ENT'] > 0)
                             for rn in ('M1_dual', 'M2_dual', 'M3_dual')},
        'kl_order_best_vs_G0': bool(agg[best]['kl_mean'] < agg['G0']['kl_mean']),
        'beats_random_control': {rn: bool(min(agg[rn]['fam_fid_mean']['SCLS'],
                                              agg[rn]['fam_fid_mean']['ENT'])
                                         > min(min(agg[ct]['fam_fid_mean']['SCLS'],
                                                   agg[ct]['fam_fid_mean']['ENT'])
                                               for ct in CTRL_OF[rn]))
                                 for rn in ('M1_dual', 'M2_dual', 'M3_dual')},
        'map_r2_seen_oov': {dn: {vn: [agg['transfer_diag'][dn][vn].get('r2_map_seen'),
                                      agg['transfer_diag'][dn][vn].get('r2_map_oov')]
                                 for vn in agg['transfer_diag'][dn]}
                            for dn in agg['transfer_diag']},
        'transfer_cos_real_mean': {dn: {vn: agg['transfer_diag'][dn][vn]['cos_transfer_real']
                                        for vn in agg['transfer_diag'][dn]}
                                   for dn in agg['transfer_diag']},
    }
    result = {
        'phase': 3243, 'name': NAME, 'mode': 'smoke' if SMOKE else 'full',
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'runtime_s': round(time.time() - T0, 1),
        'design_sha8': dsha[:8], 'smoke': SMOKE,
        'model': MODEL, 'donors': sorted(set(DONORS.values())),
        'seeds_run': SEEDS_RUN,
        'kl_rows': kl_n, 'oov_rows_total': len(oov_rows),
        'anchors': {'d83': d83, 'd_tc': b57['d_tc'], **b69['anchors'],
                    'kout': int(b69['kout']), 'D': int(b69['D']), 'd1': d1},
        'spectra': spectra,
        'per_seed': per_seed,
        'aggregate': agg,
        'delta_vs_G0': delta,
        'headline': headline,
        'gates': gates,
        'verdict': verdict,
        'sources': {k: sha8_file(v) for k, v in SRC.items()},
    }
    # res_sha8 over DETERMINISTIC fields only (mirror freeze discipline, 3240 lesson)
    res_sha = sha8(canon({k: v for k, v in result.items()
                          if k not in ('res_sha8', 'created', 'runtime_s')}).encode('utf-8'))
    result['res_sha8'] = res_sha
    rp = os.path.join(OUTDIR, 'smoke_result.json' if SMOKE else 'result.json')
    json.dump(result, open(rp, 'w', encoding='utf-8', newline=''), ensure_ascii=False, indent=1)
    with open(os.path.join(OUTDIR, 'smoke_run_log.txt' if SMOKE else 'run_log.txt'),
              'w', encoding='utf-8') as fh:
        fh.write(chr(10).join(LOG) + chr(10))
    npz_data = {'oov_row_ids': np.array(kl_rows_idx, np.int64),
                'recipes': np.array(RECIPES), 'fams': np.array(FAMS)}
    for s in SEEDS_RUN:
        for rn in RECIPES:
            npz_data['seed%d_fid_%s' % (s, rn)] = np.array(
                [per_seed[s][rn]['fam_fid'][f] for f in FAMS], np.float64)
    npz_data['agg_fid_matrix'] = np.array(
        [[agg[rn]['fam_fid_mean'][f] for f in FAMS] for rn in RECIPES], np.float64)
    npz_data['agg_r2'] = np.array([agg[rn]['r2_oov_mean'] for rn in RECIPES], np.float64)
    npz_data['agg_kl'] = np.array([agg[rn]['kl_mean'] for rn in RECIPES], np.float64)
    for dn in agg['transfer_diag']:
        for vn in agg['transfer_diag'][dn]:
            npz_data['xfer_cos_real_%s_%s' % (vn, dn.replace('-', '_'))] = np.array(
                agg['transfer_diag'][dn][vn]['cos_transfer_real'], np.float64)
    npz_path = os.path.join(OUTDIR, 'smoke_mapfix.npz' if SMOKE else 'mapfix.npz')
    np.savez(npz_path, **npz_data)
    log('npz saved: %s (%d KB)' % (os.path.basename(npz_path),
                                   os.path.getsize(npz_path) // 1024))
    log('RUN_COMPLETE %s' % verdict)
    return verdict


if __name__ == '__main__':
    v = main()
    sys.exit(0 if v == DESIGN['verdicts']['pass'] else 2)
