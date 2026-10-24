# -*- coding: utf-8 -*-
"""Phase 3248 G5-H11: transfer-map family deepening (poly lineage). Zero GPU.

Prereg (frozen before any observation): MEMO Phase 3247 tail arms.
3247 found XC_poly_b4 (poly deg-2 kernel seen-only transfer map inside the
X_combo beta=4 mechanism) is the new best declared recipe: SCLS -0.651 /
ENT -0.435, zero DVU/DCC regression, closure +0.142 of the remaining gap
(the ONLY positive menu item). This phase asks whether the MAP-FAMILY gain
is saturated, along four axes (all seen-only, declaration untouched):

  Axis 1 - kernel degree ladder: poly deg 1 (near-linear control) /
           deg 2 (3247 anchor) / deg 3
  Axis 2 - RBF bandwidth sensitivity: sigma = median heuristic x {0.5, 1,
           2} (x1 is the 3247 rbf family)
  Axis 3 - input normalization: poly2 / rbf on row-L2-normalized donor
           activations (fit AND apply in normalized space)
  Axis 4 - beta re-coupling on the poly2 base: beta in {2, 4, 6, 8}
           (beta=4 was optimal on the LIN base in 3246; the optimum may
           move on the poly base)
Recipes (16, preregistered):
  anchors: G0 (d40) / T3_rep (d42) / XC_b4 (d46) / XC_p2_b4 (d47
           bit-level vs sealed 3247 XC_poly_b4)
  degree:  XC_p1_b4 / XC_p3_b4 (p2 == anchor)
  sigma:   XC_rbf_s05 / XC_rbf_s2 (s1 covered by 3247, not repeated)
  norm:    XC_p2n_b4 / XC_rbfn_b4
  beta:    XC_p2_b2 / XC_p2_b6 / XC_p2_b8 (b4 == anchor)
  pull:    K_p3_T3 (deg-3 pure pull, no combo mechanism)
  reference: ORC_ls_cls (mechanism-free perfect-class oracle,
             EVALUATION-ONLY, closure denominator)
Gates:
  G1 anchors + census spectra + d40/d42/d46/d47 bit-level 0.0 (formal;
     smoke recorded-only)
  G2 characterization: (a) transfer realism of every NEW map spec
     (p3, rbf s05/s2, normalized variants): margin > 0 per donor;
     (b) saturation table recorded (no direction clause)
  G3 determinism 0.0;  G4 registry v1.33 untouched (byte sha8)
Verdict: pass = poly_deepening_registered / partial = poly_deepening_partial.
Headline (not gates): saturation verdict - does any axis move past the
3247 best? closure measured against [XC_p2_b4 baseline -> ORC_ls_cls].
SMOKE: P3248_SMOKE=1 (seed 7 only, KL first 200 OOV rows, smoke_* files).
Registry untouched (v1.33 remains head).
"""
import os, sys, json, time, hashlib
import numpy as np

T0 = time.time()
SRC_DIR = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913'
NAME = 'g5h11_poly_deepening'
SMOKE = os.environ.get('P3248_SMOKE') == '1'
MODEL = 'qwen3-4b'
OUTDIR = os.path.join(SRC_DIR, 'phase3248', NAME)
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
RECIPES = ['G0', 'T3_rep', 'XC_b4', 'XC_p2_b4',
           'XC_p1_b4', 'XC_p3_b4',
           'XC_rbf_s05', 'XC_rbf_s2',
           'XC_p2n_b4', 'XC_rbfn_b4',
           'XC_p2_b2', 'XC_p2_b6', 'XC_p2_b8',
           'K_p3_T3', 'ORC_ls_cls']
ANCHOR_RECIPES = ['G0', 'T3_rep', 'XC_b4', 'XC_p2_b4']
XC_NEW = ['XC_p1_b4', 'XC_p3_b4', 'XC_rbf_s05', 'XC_rbf_s2',
          'XC_p2n_b4', 'XC_rbfn_b4', 'XC_p2_b2', 'XC_p2_b6', 'XC_p2_b8']
PULL_NEW = ['K_p3_T3']
ORACLE_RECIPES = ['ORC_ls_cls']
ANCHOR_SEM = 0.05
RAND_SEED = 42
DONORS = {'T1_14b': 'qwen3-14b', 'T2_glm': 'glm4-9b'}

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
    'p3246_result': os.path.join(SRC_DIR, r'phase3246\g5h9_regime_boundary\result.json'),
    'p3247_result': os.path.join(SRC_DIR, r'phase3247\g5h10_oov_info_menu\result.json'),
    'hf_shard_emb': os.path.join(ROOT, r'models\hf\qwen3-4b\model-00001-of-00003.safetensors'),
    'hf_shard_norm': os.path.join(ROOT, r'models\hf\qwen3-4b\model-00003-of-00003.safetensors'),
    'hf_config': os.path.join(ROOT, r'models\hf\qwen3-4b\config.json'),
}

DESIGN = {
    'phase': '3248', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3247 tail arms, frozen before any '
              'observation; zero GPU (transfer-map family deepening: is '
              'the 3247 poly-kernel gain saturated?)',
    'leakage_declaration': 'identical to 3242+3244+3246+3247: prediction '
                           'recipes use token-identity embed rows, donor OOV '
                           'class means as DECLARED side information, '
                           'donor->4b maps trained on SEEN train rows only '
                           '(all map hyperparameters -- degree, sigma scale, '
                           'normalization -- derived from seen rows or fixed '
                           'constants, never from OOV data); forbidden = any '
                           '4b OOV activation as prediction INPUT. ORC_ls_cls '
                           'DELIBERATELY crosses the declaration and is '
                           'EVALUATION-ONLY: closure denominator, never a '
                           'recipe, never gated.',
    'recipes': {
        'G0': 'verbatim 3240-F2 subspace completion (anchor, d40 bit-level)',
        'T3_rep': 'verbatim 3242 T3_dual lin pull k=14 (anchor d42 bit-level)',
        'XC_b4': 'verbatim 3246 XC_b4 X_combo beta=4 lin base (anchor d46)',
        'XC_p2_b4': 'verbatim 3247 XC_poly_b4 poly2 base beta=4 (anchor d47)',
        'XC_p1_b4': 'X_combo beta=4 with poly deg-1 kernel map (near-linear '
                    'control)',
        'XC_p3_b4': 'X_combo beta=4 with poly deg-3 kernel map',
        'XC_rbf_s05': 'X_combo beta=4 with rbf map, sigma = 0.5x median '
                      'heuristic',
        'XC_rbf_s2': 'X_combo beta=4 with rbf map, sigma = 2x median',
        'XC_p2n_b4': 'X_combo beta=4 with poly2 map on row-L2-normalized '
                     'donor activations',
        'XC_rbfn_b4': 'X_combo beta=4 with rbf map on row-L2-normalized '
                      'donor activations',
        'XC_p2_b2': 'X_combo with poly2 map, beta=2',
        'XC_p2_b6': 'X_combo with poly2 map, beta=6',
        'XC_p2_b8': 'X_combo with poly2 map, beta=8',
        'K_p3_T3': 'T3-style pull k=14 with poly deg-3 maps (pure pull, '
                   'no combo mechanism)',
        'ORC_ls_cls': 'EVALUATION-ONLY mechanism-free oracle: B0 + true '
                      'class offset per row (closure denominator)',
    },
    'gates': {
        'G1': 'replication anchors + census spectra + d40/d42/d46/d47 '
              'bit-level (formal mode; smoke recorded-only)',
        'G2': 'characterization (ALL clauses): (a) transfer realism of '
              'every NEW map spec (p3, rbf s05/s2, p2n, rbfn): margin > 0 '
              'per donor (eval-only diagnostic); (b) saturation table '
              'recorded (no direction clause)',
        'G3': 'determinism 0.0',
        'G4': 'registry v1.33 untouched (byte sha8)',
    },
    'verdicts': {'pass': 'poly_deepening_registered',
                 'partial': 'poly_deepening_partial'},
    'note': 'headline: saturation verdict across 4 axes (degree / sigma / '
            'normalization / beta); closure vs [XC_p2_b4 -> ORC_ls_cls]. '
            'Registry append not triggered: characterization.',
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
        json.dump({'phase': 3248, 'name': NAME, 'design_sha8': dsha[:8],
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


# ---------------- 3157-derived families (verbatim) ----------------
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


# ---------------- CPU lens + token embeddings (verbatim) ----------------
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


def token_row(word):
    from transformers import AutoTokenizer
    if 'tok' not in LENS:
        LENS['tok'] = AutoTokenizer.from_pretrained(os.path.join(ROOT, r'models\hf\qwen3-4b'))
    ids = LENS['tok'].encode(word, add_special_tokens=False)
    assert 1 <= len(ids) <= 8, ('token count out of band', word, ids)
    return LENS['E'][np.array(ids, int)].astype(np.float32).mean(0)


def softmax_weights(cosv, beta=1.0):
    e = np.exp(beta * (cosv - cosv.max()))
    return e / e.sum()


# ---------------- donor loading + transfer maps (verbatim 3242/3247 + NEW) --
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
    n = len(Hd_tr)
    K = (Hd_tr @ Hd_tr.T).astype(np.float64)
    K[np.diag_indices(n)] += lam
    K += 1.0
    alpha = np.linalg.solve(K, Ytr.astype(np.float64)).astype(np.float32)
    return Hd_tr, alpha


def ridge_dual_apply(Hd_tr, alpha, V):
    return (V @ Hd_tr.T @ alpha + alpha.sum(0, keepdims=True)).astype(np.float32)


def sqdist(A, B):
    d2 = ((A.astype(np.float64) ** 2).sum(1)[:, None]
          + (B.astype(np.float64) ** 2).sum(1)[None, :]
          - 2.0 * (A.astype(np.float64) @ B.astype(np.float64).T))
    return np.maximum(d2, 0.0)


def row_l2(A):
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


def make_kfn(fam, Hd_tr, deg=2, sig_scale=1.0):
    if fam == 'rbf':
        D2 = sqdist(Hd_tr, Hd_tr)
        iu = np.triu_indices(len(Hd_tr), 1)
        sig2 = float(np.median(D2[iu])) * sig_scale ** 2
        sig2 = max(sig2, 1e-12)
        return (lambda A, B: np.exp(-sqdist(A, B) / (2.0 * sig2))), sig2
    if fam == 'poly':
        Dd = float(Hd_tr.shape[1])
        return (lambda A, B: ((A.astype(np.float64) @ B.astype(np.float64).T) / Dd + 1.0) ** deg), None
    raise ValueError(fam)


def kernel_dual_fit(Hd_tr, Ytr, kfn, lam=LAM):
    n = len(Hd_tr)
    K = kfn(Hd_tr, Hd_tr)
    K[np.diag_indices(n)] += lam
    K += 1.0
    alpha = np.linalg.solve(K, Ytr.astype(np.float64)).astype(np.float32)
    return alpha, kfn


def kernel_dual_apply(alpha, kfn, Hd_tr, V):
    return (kfn(V, Hd_tr) @ alpha + alpha.sum(0, keepdims=True)).astype(np.float32)


# map spec: (fam, deg, sig_scale, norm). Returns (d_hat, d_hat_perm,
# cos_real, cos_perm, meta) for BOTH donors.
def build_map(spec, donors, don_oov_rows_c, perm_rows_c, tr_rows, Y, B0,
              d_true, OOV_CLASSES):
    fam, deg, sig_scale, norm = spec
    d_hat, d_hat_p, cr, cp, meta = {}, {}, {}, {}, {}
    for dname in set(DONORS.values()):
        Yd = donors[dname]['Y']
        Hd = Yd[tr_rows]
        if norm:
            Hd = row_l2(Hd)
            m_oov = row_l2(np.stack([Yd[don_oov_rows_c[c]].mean(0)
                                     for c in OOV_CLASSES]))
            m_oov_p = row_l2(np.stack([Yd[perm_rows_c[c]].mean(0)
                                       for c in OOV_CLASSES]))
        else:
            m_oov = np.stack([Yd[don_oov_rows_c[c]].mean(0) for c in OOV_CLASSES])
            m_oov_p = np.stack([Yd[perm_rows_c[c]].mean(0) for c in OOV_CLASSES])
        if fam == 'lin':
            _, alpha = ridge_dual_fit(Hd, Y[tr_rows])
            d_hat[dname] = ridge_dual_apply(Hd, alpha, m_oov) - B0[None, :]
            d_hat_p[dname] = ridge_dual_apply(Hd, alpha, m_oov_p) - B0[None, :]
        else:
            kfn, sig2 = make_kfn(fam, Hd, deg=deg, sig_scale=sig_scale)
            alpha, _ = kernel_dual_fit(Hd, Y[tr_rows], kfn)
            d_hat[dname] = kernel_dual_apply(alpha, kfn, Hd, m_oov) - B0[None, :]
            d_hat_p[dname] = kernel_dual_apply(alpha, kfn, Hd, m_oov_p) - B0[None, :]
            meta[dname] = {'sigma2': sig2}
        cr[dname] = [float(unit(d_hat[dname][k]) @ d_true[c])
                     for k, c in enumerate(OOV_CLASSES)]
        cp[dname] = [float(unit(d_hat_p[dname][k]) @ d_true[c])
                     for k, c in enumerate(OOV_CLASSES)]
    return d_hat, d_hat_p, cr, cp, meta


def main():
    dsha = freeze()
    log('design_sha8=%s smoke=%s seeds=%s' % (dsha[:8], SMOKE, SEEDS_RUN))
    sealed_3169 = json.load(open(SRC['p3169_result'], encoding='utf-8'))
    sealed_3183 = json.load(open(SRC['p3183_result'], encoding='utf-8'))
    sealed_3188 = json.load(open(SRC['p3188_result'], encoding='utf-8'))
    sealed_3189 = json.load(open(SRC['p3189_result'], encoding='utf-8'))
    sealed_3240 = json.load(open(SRC['p3240_result'], encoding='utf-8'))
    sealed_3242 = json.load(open(SRC['p3242_result'], encoding='utf-8'))
    sealed_3246 = json.load(open(SRC['p3246_result'], encoding='utf-8'))
    sealed_3247 = json.load(open(SRC['p3247_result'], encoding='utf-8'))
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

    cls_words = CLASSES_SEEN + CLASSES_OOV
    W_cls = np.stack([token_row(w) for w in cls_words])
    W_ent = np.stack([token_row(w) for w in ENTS])
    cos_cm = np.array([[float(unit(W_cls[a]) @ unit(W_cls[b])) for b in range(NCS)]
                       for a in range(NC)])
    cos_em = np.array([[float(unit(W_ent[a]) @ unit(W_ent[b])) for b in range(NTR_ENT)]
                       for a in range(NE)])
    log('bridge cos: cls min=%.3f max=%.3f | ent min=%.3f max=%.3f'
        % (cos_cm.min(), cos_cm.max(), cos_em.min(), cos_em.max()))

    rng_lex = np.random.RandomState(RAND_SEED)
    perm_cls = rng_lex.permutation(NC)
    perm_ent = rng_lex.permutation(NE)

    donors = {}
    for dname in set(DONORS.values()):
        Yd, kd, Dd = load_donor_kout(dname)
        donors[dname] = dict(Y=Yd, kout=kd, D=Dd)
        log('donor %s: kout=%d D=%d (sha8 npz %s)'
            % (dname, kd, Dd, sha8_file(SRC[donor_npz_key(dname)])))
    don_oov_rows_c = {c: [r for r in oov_rows if PAIRS[r % NP_][1] == c]
                      for c in OOV_CLASSES}
    rng_perm = np.random.RandomState(RAND_SEED)
    oov_sizes = {c: len(don_oov_rows_c[c]) for c in OOV_CLASSES}
    all_oov_shuffled = rng_perm.permutation(oov_rows)
    perm_rows_c = {}
    pos = 0
    for c in OOV_CLASSES:
        perm_rows_c[c] = sorted(all_oov_shuffled[pos:pos + oov_sizes[c]])
        pos += oov_sizes[c]
    assert pos == len(oov_rows)

    # map specs: anchor families + new deepening axes
    SPEC_LIN = ('lin', 1, 1.0, False)
    SPEC_P2 = ('poly', 2, 1.0, False)
    SPECS_NEW = {
        'p3': ('poly', 3, 1.0, False),
        'p1': ('poly', 1, 1.0, False),
        'rbf_s05': ('rbf', 2, 0.5, False),
        'rbf_s2': ('rbf', 2, 2.0, False),
        'p2n': ('poly', 2, 1.0, True),
        'rbfn': ('rbf', 2, 1.0, True),
    }
    BETA_XC = 4.0
    BETA_LADDER = [2.0, 6.0, 8.0]

    per_seed = {}
    det_diff = None
    realism_all = {}
    for s in SEEDS_RUN:
        tr_rows = b69['tr_base'][s]
        Xtr = np.stack([phi_row(PAIRS[r % NP_][0], PAIRS[r % NP_][1], r // NP_)
                        for r in tr_rows])
        W = ridge_primal(Xtr, Y[tr_rows])
        B0 = Y[tr_rows].mean(0)
        cls_mean = {}
        for r in tr_rows:
            cls_mean.setdefault(PAIRS[r % NP_][1], []).append(Y[r])
        cls_mean = {c: np.stack(v).mean(0) for c, v in cls_mean.items()}
        ent_mean = {}
        for r in tr_rows:
            ent_mean.setdefault(PAIRS[r % NP_][0], []).append(Y[r])
        ent_mean = {i: np.stack(v).mean(0) for i, v in ent_mean.items()}
        d_cls = np.stack([cls_mean[c] - B0 for c in range(NCS)])
        e_ent = np.stack([ent_mean[i] - B0 for i in range(NTR_ENT)])
        Qs, _ = np.linalg.qr(d_cls.T.astype(np.float32))
        Qs = Qs[:, :NCS].astype(np.float32)

        Xo = np.stack([phi_row(PAIRS[r % NP_][0], PAIRS[r % NP_][1], r // NP_)
                       for r in kl_rows_idx])
        Yhat0 = Xo @ W
        corr_G0 = (B0[None, :] - Yhat0) @ Qs @ Qs.T

        d_true = {c: unit(Y[don_oov_rows_c[c]].mean(0) - B0) for c in OOV_CLASSES}

        # map building: lin (anchor) + poly2 (anchor) + 6 new specs
        maps = {}
        realism_all[s] = {}
        for key, spec in [('lin', SPEC_LIN), ('p2', SPEC_P2)] + list(SPECS_NEW.items()):
            d_hat, d_hat_p, cr, cp, mt = build_map(
                spec, donors, don_oov_rows_c, perm_rows_c, tr_rows, Y, B0,
                d_true, OOV_CLASSES)
            maps[key] = (d_hat, d_hat_p)
            realism_all[s][key] = {
                'cos_real_mean': float(np.mean([v for dn in cr for v in cr[dn]])),
                'cos_perm_mean': float(np.mean([v for dn in cp for v in cp[dn]])),
                'meta': mt}
            log('seed %d map %s: real=%.4f perm=%.4f'
                % (s, key, realism_all[s][key]['cos_real_mean'],
                   realism_all[s][key]['cos_perm_mean']))

        def qr_span(mat_rows):
            Q, _ = np.linalg.qr(mat_rows.T.astype(np.float32))
            return Q[:, :mat_rows.shape[0]].astype(np.float32)

        def q_of(key):
            dh = maps[key][0]
            return qr_span(np.concatenate([d_cls, dh['qwen3-14b'], dh['glm4-9b']], 0))

        Q_T3 = q_of('lin')
        QK = {key: q_of(key) for key in maps if key != 'lin'}

        d_orc = np.stack([Y[don_oov_rows_c[c]].mean(0) - B0
                          for c in OOV_CLASSES])

        def pull_of(Q):
            return (B0[None, :] - Yhat0) @ Q @ Q.T

        pull_T3 = pull_of(Q_T3)

        # lexical correction cf(beta) (verbatim 3246/3247 mechanism)
        d_mean = d_cls.mean(0)
        e_mean = e_ent.mean(0)
        SJE_dirs = np.concatenate([d_cls, e_ent], axis=0)
        Qje, _ = np.linalg.qr(SJE_dirs.T.astype(np.float32))
        Qje = Qje[:, :SJE_dirs.shape[0]].astype(np.float32)
        cf_betas = {}
        for beta in [BETA_XC] + BETA_LADDER:
            cf = np.zeros_like(Yhat0)
            for j, r in enumerate(kl_rows_idx):
                i, c = PAIRS[r % NP_]
                wf = lambda cv, bb=beta: softmax_weights(cv, bb)
                dA_c = (wf(cos_cm[c])[:, None] * (d_cls - d_mean[None, :])).sum(0)
                dA_e = (wf(cos_em[i])[:, None] * (e_ent - e_mean[None, :])).sum(0)
                cf[j] = (B0 + dA_c + dA_e - Yhat0[j]) @ Qje @ Qje.T
            cf_betas[beta] = cf
        Bd_rel = np.concatenate([bases['DVU'], bases['DCC']], axis=0).astype(np.float32)
        Qcl, _ = np.linalg.qr(Bd_rel.T.astype(np.float32))
        Bd = Qcl[:, :Bd_rel.shape[0]].T.astype(np.float32)

        def xc_arm(base_pull, cf):
            extra = cf - base_pull
            extra_clean = extra - (extra @ Bd.T) @ Bd
            return (Yhat0 + base_pull + extra_clean).astype(np.float32)

        Ypreds = {}
        Ypreds['G0'] = (Yhat0 + corr_G0).astype(np.float32)
        Ypreds['T3_rep'] = (Yhat0 + pull_T3).astype(np.float32)
        Ypreds['XC_b4'] = xc_arm(pull_T3, cf_betas[BETA_XC])
        Ypreds['XC_p2_b4'] = xc_arm(pull_of(QK['p2']), cf_betas[BETA_XC])
        Ypreds['XC_p1_b4'] = xc_arm(pull_of(QK['p1']), cf_betas[BETA_XC])
        Ypreds['XC_p3_b4'] = xc_arm(pull_of(QK['p3']), cf_betas[BETA_XC])
        Ypreds['XC_rbf_s05'] = xc_arm(pull_of(QK['rbf_s05']), cf_betas[BETA_XC])
        Ypreds['XC_rbf_s2'] = xc_arm(pull_of(QK['rbf_s2']), cf_betas[BETA_XC])
        Ypreds['XC_p2n_b4'] = xc_arm(pull_of(QK['p2n']), cf_betas[BETA_XC])
        Ypreds['XC_rbfn_b4'] = xc_arm(pull_of(QK['rbfn']), cf_betas[BETA_XC])
        Ypreds['XC_p2_b2'] = xc_arm(pull_of(QK['p2']), cf_betas[2.0])
        Ypreds['XC_p2_b6'] = xc_arm(pull_of(QK['p2']), cf_betas[6.0])
        Ypreds['XC_p2_b8'] = xc_arm(pull_of(QK['p2']), cf_betas[8.0])
        Ypreds['K_p3_T3'] = (Yhat0 + pull_of(QK['p3'])).astype(np.float32)
        orc_cls_row = np.stack([d_orc[c - NCS]
                                for r in kl_rows_idx
                                for c in [PAIRS[r % NP_][1]]])
        Ypreds['ORC_ls_cls'] = (B0[None, :] + orc_cls_row).astype(np.float32)

        Yact = Y[kl_rows_idx]
        Yb0 = np.stack([B0] * kl_n)
        mse_b0 = float(((Yact - Yb0) ** 2).sum(1).mean())

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

        # anchors
        ps40 = sealed_3240['per_seed'][str(s)]['F2']
        d40 = max(abs(seed_rec['G0']['r2_oov'] - ps40['r2_oov']),
                  abs(seed_rec['G0']['kl_mean'] - ps40['kl_mean']),
                  max(abs(seed_rec['G0']['fam_fid'][f] - ps40['fam_fid'][f])
                      for f in FAMS))
        seed_rec['G0']['d_vs_sealed_3240'] = d40
        ps42 = sealed_3242['per_seed'][str(s)]['T3_dual']
        d42 = max(abs(seed_rec['T3_rep']['r2_oov'] - ps42['r2_oov']),
                  abs(seed_rec['T3_rep']['kl_mean'] - ps42['kl_mean']),
                  max(abs(seed_rec['T3_rep']['fam_fid'][f] - ps42['fam_fid'][f])
                      for f in FAMS))
        seed_rec['T3_rep']['d_vs_sealed_3242'] = d42
        ps46 = sealed_3246['per_seed'][str(s)]['XC_b4']
        d46 = max(abs(seed_rec['XC_b4']['r2_oov'] - ps46['r2_oov']),
                  abs(seed_rec['XC_b4']['kl_mean'] - ps46['kl_mean']),
                  max(abs(seed_rec['XC_b4']['fam_fid'][f] - ps46['fam_fid'][f])
                      for f in FAMS))
        seed_rec['XC_b4']['d_vs_sealed_3246'] = d46
        ps47 = sealed_3247['per_seed'][str(s)]['XC_poly_b4']
        d47 = max(abs(seed_rec['XC_p2_b4']['r2_oov'] - ps47['r2_oov']),
                  abs(seed_rec['XC_p2_b4']['kl_mean'] - ps47['kl_mean']),
                  max(abs(seed_rec['XC_p2_b4']['fam_fid'][f] - ps47['fam_fid'][f])
                      for f in FAMS))
        seed_rec['XC_p2_b4']['d_vs_sealed_3247'] = d47
        per_seed[s] = seed_rec
        log('seed %d: d40=%.3e d42=%.3e d46=%.3e d47=%.3e'
            % (s, d40, d42, d46, d47))
        for rn in RECIPES:
            ff = seed_rec[rn]['fam_fid']
            log('seed %d %s: SCLS=%.3f ENT=%.3f DVU=%.3f DCC=%.3f r2=%.4f KL=%.4f'
                % (s, rn, ff['SCLS'], ff['ENT'], ff['DVU'], ff['DCC'],
                   seed_rec[rn]['r2_oov'], seed_rec[rn]['kl_mean']))
        del W, Xtr, la

    agg = {}
    for rn in RECIPES:
        agg[rn] = {
            'r2_oov_mean': float(np.mean([per_seed[s][rn]['r2_oov'] for s in SEEDS_RUN])),
            'kl_mean': float(np.mean([per_seed[s][rn]['kl_mean'] for s in SEEDS_RUN])),
            'fam_fid_mean': {f: float(np.mean([per_seed[s][rn]['fam_fid'][f]
                                               for s in SEEDS_RUN])) for f in FAMS},
        }
    agg['G0']['d_vs_sealed_3240_max'] = float(max(per_seed[s]['G0']['d_vs_sealed_3240']
                                                  for s in SEEDS_RUN))
    agg['T3_rep']['d_vs_sealed_3242_max'] = float(max(per_seed[s]['T3_rep']['d_vs_sealed_3242']
                                                      for s in SEEDS_RUN))
    agg['XC_b4']['d_vs_sealed_3246_max'] = float(max(per_seed[s]['XC_b4']['d_vs_sealed_3246']
                                                     for s in SEEDS_RUN))
    agg['XC_p2_b4']['d_vs_sealed_3247_max'] = float(max(per_seed[s]['XC_p2_b4']['d_vs_sealed_3247']
                                                        for s in SEEDS_RUN))
    delta = {}
    for rn in RECIPES:
        delta[rn] = {
            'd_fam_fid': {f: agg[rn]['fam_fid_mean'][f] - agg['G0']['fam_fid_mean'][f]
                          for f in FAMS},
            'd_r2_oov': agg[rn]['r2_oov_mean'] - agg['G0']['r2_oov_mean'],
            'd_kl_mean': agg[rn]['kl_mean'] - agg['G0']['kl_mean'],
        }
    log('AGG G0: SCLS=%.3f ENT=%.3f r2=%.4f' %
        (agg['G0']['fam_fid_mean']['SCLS'], agg['G0']['fam_fid_mean']['ENT'],
         agg['G0']['r2_oov_mean']))
    for rn in RECIPES[1:]:
        d = delta[rn]['d_fam_fid']
        log('AGG %s: SCLS=%.3f (%+.3f) ENT=%.3f (%+.3f) DVU=%+.3f DCC=%+.3f r2=%.4f KL=%.4f'
            % (rn, agg[rn]['fam_fid_mean']['SCLS'], d['SCLS'],
               agg[rn]['fam_fid_mean']['ENT'], d['ENT'],
               d['DVU'], d['DCC'], agg[rn]['r2_oov_mean'], agg[rn]['kl_mean']))

    # realism aggregate over seeds (clause a)
    realism = {}
    for key in ('p1', 'p3', 'rbf_s05', 'rbf_s2', 'p2n', 'rbfn'):
        cr = [realism_all[s][key]['cos_real_mean'] for s in SEEDS_RUN]
        cp = [realism_all[s][key]['cos_perm_mean'] for s in SEEDS_RUN]
        realism[key] = {'cos_real_mean': float(np.mean(cr)),
                        'cos_perm_mean': float(np.mean(cp)),
                        'margin': float(np.mean(cr) - np.mean(cp))}
        log('realism %s: real=%.4f perm=%.4f margin=%+.4f'
            % (key, realism[key]['cos_real_mean'], realism[key]['cos_perm_mean'],
               realism[key]['margin']))

    # gates
    g1 = (all(v < REPL_TOL for v in (d83, b57['d_tc'], b69['anchors']['drift69'],
                                     b69['anchors']['d88']))
          and b69['anchors']['d89'] < 1e-6
          and all(spectra[f]['k90_match_sealed'] and spectra[f]['pr_match_sealed']
                  and spectra[f]['top1_match_sealed'] for f in FAMS)
          and (agg['G0']['d_vs_sealed_3240_max'] < REPL_TOL if not SMOKE else True)
          and (agg['T3_rep']['d_vs_sealed_3242_max'] < REPL_TOL if not SMOKE else True)
          and (agg['XC_b4']['d_vs_sealed_3246_max'] < REPL_TOL if not SMOKE else True)
          and (agg['XC_p2_b4']['d_vs_sealed_3247_max'] < REPL_TOL if not SMOKE else True))
    clause_a = all(realism[k]['margin'] > 0 for k in realism)
    clause_b = True
    g2 = bool(clause_a and clause_b)
    g3 = (det_diff == 0.0)
    g4 = (reg33_sha_sealed == sha8(open(SRC['registry_v133'], 'rb').read()))
    gates = {'G1_replication': bool(g1), 'G2_characterization': g2,
             'clause_a_transfer_realism': bool(clause_a),
             'clause_b_saturation_recorded': bool(clause_b),
             'G3_determinism': bool(g3), 'G4_registry_untouched': bool(g4),
             'determinism_diff': det_diff}
    allg = [gates[k] for k in ('G1_replication', 'G2_characterization',
                               'G3_determinism', 'G4_registry_untouched')]
    verdict = DESIGN['verdicts']['pass'] if all(allg) else DESIGN['verdicts']['partial']
    log('VERDICT: %s gates=%s' % (verdict, gates))

    # saturation verdict (headline): does ANY axis beat the 3247 best?
    def minf(rn):
        return min(agg[rn]['fam_fid_mean']['SCLS'], agg[rn]['fam_fid_mean']['ENT'])

    base_min = minf('XC_p2_b4')
    ceil_min = minf('ORC_ls_cls')
    denom = ceil_min - base_min
    closure = {}
    for rn in RECIPES:
        if rn in ('XC_p2_b4', 'ORC_ls_cls'):
            continue
        closure[rn] = float((minf(rn) - base_min) / denom) if abs(denom) > 1e-12 else 0.0
    best_new = max(closure, key=closure.get) if closure else 'XC_p2_b4'
    saturated = bool(all(v <= 0.0 for v in closure.values()))
    log('CLOSURE: %s' % json.dumps({k: round(v, 3) for k, v in closure.items()}))
    log('BEST_NEW: %s (%.3f) SATURATED=%s' % (best_new, closure.get(best_new, 0.0), saturated))

    headline = {
        'closure_table': closure,
        'best_new_item': best_new,
        'best_new_closure': closure.get(best_new, 0.0),
        'saturated': saturated,
        'gap_baseline_minSCLSENT': base_min,
        'gap_ceiling_minSCLSENT': ceil_min,
        'transfer_realism': realism,
        'saturation_axes': {
            'degree_ladder_SCLS': {rn: agg[rn]['fam_fid_mean']['SCLS']
                                   for rn in ('XC_p1_b4', 'XC_p2_b4', 'XC_p3_b4')},
            'sigma_ladder_SCLS': {rn: agg[rn]['fam_fid_mean']['SCLS']
                                  for rn in ('XC_rbf_s05', 'XC_rbf_s2')},
            'beta_ladder_minSCLSENT': {rn: minf(rn) for rn in
                                       ('XC_p2_b2', 'XC_p2_b4', 'XC_p2_b6', 'XC_p2_b8')},
            'norm_variants_SCLS': {rn: agg[rn]['fam_fid_mean']['SCLS']
                                   for rn in ('XC_p2n_b4', 'XC_rbfn_b4')},
        },
    }
    result = {
        'phase': 3248, 'name': NAME, 'mode': 'smoke' if SMOKE else 'full',
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'runtime_s': round(time.time() - T0, 1),
        'design_sha8': dsha[:8], 'smoke': SMOKE,
        'model': MODEL, 'seeds_run': SEEDS_RUN,
        'kl_rows': kl_n, 'oov_rows_total': len(oov_rows),
        'anchors': {'d83': d83, 'd_tc': b57['d_tc'], **b69['anchors'],
                    'kout': int(b69['kout']), 'D': int(b69['D']), 'd1': d1},
        'spectra': spectra,
        'cleaning_basis_dims': int(Bd.shape[0]),
        'per_seed': per_seed,
        'aggregate': agg,
        'delta_vs_G0': delta,
        'headline': headline,
        'gates': gates,
        'verdict': verdict,
        'sources': {k: sha8_file(v) for k, v in SRC.items()},
    }
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
    npz_data['closure'] = np.array([closure.get(rn, 0.0) for rn in RECIPES], np.float64)
    npz_path = os.path.join(OUTDIR, 'smoke_poly.npz' if SMOKE else 'poly.npz')
    np.savez(npz_path, **npz_data)
    log('npz saved: %s (%d KB)' % (os.path.basename(npz_path),
                                   os.path.getsize(npz_path) // 1024))
    log('RUN_COMPLETE %s' % verdict)
    return verdict


if __name__ == '__main__':
    v = main()
    sys.exit(0 if v == DESIGN['verdicts']['pass'] else 2)
