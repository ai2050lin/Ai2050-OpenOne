# -*- coding: utf-8 -*-
"""Phase 3242 G5-H5: cross-model class-direction transfer arm. Zero GPU.

Prereg (frozen before any observation): MEMO Phase 3241 tail arm (a).
Target: 3241 showed the lexical channel cannot carry OOV class directions
(class bridge ~zero effect; dual bridge positive but regressions). The
remaining activation-side option is CROSS-MODEL transfer: donor models
(qwen3-14b / glm4-9b) encode the same panel; their OOV class structure may
transfer into qwen3-4b space through a linear map trained on SEEN rows only.
Prior: census spectral shapes are cross-model stable (DCC shape corr 0.996).

Leakage declaration (prereg): allowed = donor-model activations on the SAME
sealed panel rows (the transfer arm's declared side information: donor OOV
class means enter the prediction recipe), and the seen-row linear map; the
map is trained EXCLUSIVELY on qwen3-4b SEEN train rows (split_s1_246, per
seed). forbidden = any qwen3-4b OOV activation as prediction INPUT; 4b OOV
activations are used only as the evaluation target (metrics) and for the
post-hoc transfer-fidelity diagnostic (cos vs true OOV class direction),
which is headline-only and never gates.

Recipes (preregistered, no post-hoc selection):
  G0       verbatim 3240-F2 subspace completion (replication anchor, per-seed
           bit-level vs sealed 3240 result.json F2, <1e-9)
  T1_14b   extended span: [6 train class dirs (4b)] + 4 OOV class directions
           transferred from qwen3-14b via the seen-only dual-ridge map;
           prediction = Yhat_ridge + P_span(B0 - Yhat_ridge) (F2 mechanism)
  T2_glm   same with glm4-9b donor
  T3_dual  span = 6 + 4 + 4 (both donors' transferred directions)
  T1r_14b  randomization control: donor OOV class labels permuted (seed 42)
           before computing donor class means; matched control of T1_14b
  T2r_glm  same for glm4-9b; matched control of T2_glm

Gates:
  G1 replication: d83/d_tc/drift69/d88 <1e-9, d89 <1e-6, census spectra
     cross-check vs sealed 3238, G0 per-seed vs sealed 3240 F2 <1e-9
     (formal mode only; smoke row subset recorded-only, 3240 lesson (2))
  G2 efficacy: best(T1_14b, T2_glm, T3_dual) improves BOTH SCLS and ENT
     fidelity by > +0.5 vs G0 (3-seed mean), min(d_DVU, d_DCC) > -0.05,
     R2 drop < 0.01, AND beats its matched permuted control on
     min(SCLS, ENT) (T3_dual must beat BOTH controls)
  G3 determinism: identical-GEMM-shape lens recompute bitwise 0.0
  G4 registry v1.33 untouched (byte sha8)
Headline (not gates): SCLS/ENT sign flip; per-donor transfer fidelity
cos(d_hat_c, d_c^true) and mapping R2 (seen-test / OOV); KL ordering.
SMOKE: P3242_SMOKE=1 (seed 7 only, KL first 200 OOV rows, smoke_* files).
Registry untouched (v1.33 remains head).
"""
import os, sys, json, time, hashlib
import numpy as np

T0 = time.time()
SRC_DIR = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913'
NAME = 'g5h5_xmodel_transfer'
SMOKE = os.environ.get('P3242_SMOKE') == '1'
MODEL = 'qwen3-4b'
OUTDIR = os.path.join(SRC_DIR, 'phase3242', NAME)
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
RECIPES = ['G0', 'T1_14b', 'T2_glm', 'T3_dual', 'T1r_14b', 'T2r_glm']
EFF_DELTA = 0.5
REG_DROP = 0.05
R2_DROP = 0.01
RAND_SEED = 42
DONORS = {'T1_14b': 'qwen3-14b', 'T2_glm': 'glm4-9b'}
CTRL_OF = {'T1_14b': ['T1r_14b'], 'T2_glm': ['T2r_glm'],
           'T3_dual': ['T1r_14b', 'T2r_glm']}

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
    'hf_shard_emb': os.path.join(ROOT, r'models\hf\qwen3-4b\model-00001-of-00003.safetensors'),
    'hf_shard_norm': os.path.join(ROOT, r'models\hf\qwen3-4b\model-00003-of-00003.safetensors'),
    'hf_config': os.path.join(ROOT, r'models\hf\qwen3-4b\config.json'),
}

DESIGN = {
    'phase': '3242', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3241 tail arm (a), frozen before any observation; '
              'zero GPU (cross-model class-direction transfer on the 3239/3240 '
              'fidelity loop)',
    'leakage_declaration': 'allowed = donor-model (qwen3-14b / glm4-9b) activations '
                           'on the SAME sealed 3169 panel rows — the transfer arm has '
                           'no non-trivial content without donor OOV structure, so '
                           'donor OOV class means are the DECLARED side information; '
                           'the donor->4b linear map is trained EXCLUSIVELY on 4b '
                           'SEEN train rows (split_s1_246 per seed, verbatim B-'
                           'protocol). forbidden = any qwen3-4b OOV activation as '
                           'prediction INPUT; 4b OOV activations appear only as the '
                           'evaluation target and in the post-hoc transfer-fidelity '
                           'DIAGNOSTIC (headline-only, never gates).',
    'recipes': {
        'G0': 'verbatim 3240-F2 subspace completion (replication anchor, per-seed '
              'bit-level vs sealed 3240 result.json F2)',
        'T1_14b': 'extended span: 6 train class dirs (4b) + 4 OOV class dirs '
                  'transferred from qwen3-14b (dual-ridge map trained on seen rows '
                  'only, bias included; d_hat_c = M(m_c_donor) - B0_4b); prediction '
                  '= Yhat_ridge + P_span(B0 - Yhat_ridge) — F2 mechanism, span '
                  'extended to 10',
        'T2_glm': 'same as T1_14b with glm4-9b donor',
        'T3_dual': 'span = 6 train + 4 (14b) + 4 (glm4) = 14 transferred+trained '
                   'directions, single joint QR',
        'T1r_14b': 'matched control of T1_14b: donor OOV row->class assignment '
                   'permuted (seed 42, row-level) before computing donor OOV class '
                   'means — destroys class structure, preserves the row pool',
        'T2r_glm': 'matched control of T2_glm (same row-level permutation)',
    },
    'gates': {
        'G1': 'replication anchors + census spectra cross-check + G0 per-seed '
              'bit-level vs sealed 3240 F2 (formal mode; smoke recorded-only per '
              '3240 correction protocol)',
        'G2': 'efficacy: best(T1_14b, T2_glm, T3_dual) improves BOTH SCLS and ENT '
              'fidelity > +%.1f vs G0 (3-seed mean; min(d_DVU,d_DCC) > -%.2f; '
              'R2 drop < %.2f), AND beats matched permuted control on '
              'min(SCLS, ENT) (T3_dual beats both controls)'
              % (EFF_DELTA, REG_DROP, R2_DROP),
        'G3': 'determinism: lens first-batch identical-GEMM-shape recompute diff 0.0',
        'G4': 'registry v1.33 untouched (byte sha8)',
    },
    'verdicts': {'pass': 'xmodel_transfer_registered', 'partial': 'xmodel_transfer_partial'},
    'note': 'headline (SCLS/ENT sign flip; transfer fidelity; KL ordering) is a '
            'registered finding, NOT a gate. Registry append not triggered: '
            'prediction-recipe repair, no feature-level statement change.',
    'prereg_correction': 'one correction at SMOKE stage (P3242_SMOKE=1), before '
                         'any formal-run observation: the preregistered control '
                         '(class-level label permutation over the 4 OOV classes) '
                         'is vacuous — it only reorders the 4 transferred class '
                         'means, leaving the projected span identical (first SMOKE '
                         'showed T1r == T1 to 4 decimals, verbatim). Redesigned to '
                         'ROW-LEVEL permutation of the donor OOV row->class '
                         'assignment (seed 42), which destroys class structure '
                         'while preserving the row pool. Recipe span construction, '
                         'gates, and all other prereg elements unchanged. No '
                         'formal-run metric numbers entered the design.',
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
        json.dump({'phase': 3242, 'name': NAME, 'design_sha8': dsha[:8],
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


def main():
    dsha = freeze()
    log('design_sha8=%s smoke=%s seeds=%s' % (dsha[:8], SMOKE, SEEDS_RUN))
    sealed_3169 = json.load(open(SRC['p3169_result'], encoding='utf-8'))
    sealed_3183 = json.load(open(SRC['p3183_result'], encoding='utf-8'))
    sealed_3188 = json.load(open(SRC['p3188_result'], encoding='utf-8'))
    sealed_3189 = json.load(open(SRC['p3189_result'], encoding='utf-8'))
    sealed_3240 = json.load(open(SRC['p3240_result'], encoding='utf-8'))
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

        # per-donor transfer
        d_hat = {}          # recipe-key -> (4, 2560) transferred OOV class dirs
        xfer_diag = {}
        for rn, dname in DONORS.items():
            Yd = donors[dname]['Y']
            Hd_tr = Yd[tr_rows]
            Hd_tr, alpha_d = ridge_dual_fit(Hd_tr, Y[tr_rows])
            # mapping quality (diagnostic): seen-test (in-distribution) + OOV rows
            pred_te = ridge_dual_apply(Hd_tr, alpha_d, Yd[te_rows])
            dk = float(((Y[te_rows] - Y[te_rows].mean(0)) ** 2).sum(1).mean()) + 1e-9
            r2_map_seen = float(1.0 - ((pred_te - Y[te_rows]) ** 2).sum(1).mean() / dk)
            pred_oov = ridge_dual_apply(Hd_tr, alpha_d, Yd[oov_rows])
            r2_map_oov = float(1.0 - ((pred_oov - Y[oov_rows]) ** 2).sum(1).mean() / dk)
            # seen-class transfer sanity: mapped donor seen-class mean vs 4b train class dir
            m_seen = np.stack([Yd[[r for r in tr_rows
                                   if PAIRS[r % NP_][1] == c]].mean(0) for c in range(NCS)])
            m_seen_mapped = ridge_dual_apply(Hd_tr, alpha_d, m_seen)
            cos_seen = [float(unit(m_seen_mapped[c] - B0) @ unit(d_cls[c]))
                        for c in range(NCS)]
            # OOV class means: real and row-permuted (donor space)
            m_oov = np.stack([Yd[don_oov_rows_c[c]].mean(0) for c in OOV_CLASSES])
            m_oov_perm = np.stack([Yd[perm_rows_c[c]].mean(0) for c in OOV_CLASSES])
            d_hat[rn] = ridge_dual_apply(Hd_tr, alpha_d, m_oov) - B0[None, :]
            d_hat[rn + '_perm'] = (ridge_dual_apply(Hd_tr, alpha_d, m_oov_perm)
                                   - B0[None, :])
            # transfer fidelity diagnostic (eval-only): cos vs true OOV dir
            cos_real = [float(unit(d_hat[rn][k]) @ d_true[c])
                        for k, c in enumerate(OOV_CLASSES)]
            cos_perm = [float(unit(d_hat[rn + '_perm'][k]) @ d_true[c])
                        for k, c in enumerate(OOV_CLASSES)]
            nrm_ratio = [float(np.linalg.norm(d_hat[rn][k])
                               / (np.linalg.norm(Y[don_oov_rows_c[c]].mean(0) - B0) + 1e-18))
                         for k, c in enumerate(OOV_CLASSES)]
            xfer_diag[dname] = {'r2_map_seen': r2_map_seen, 'r2_map_oov': r2_map_oov,
                                'cos_seen_train_dirs': cos_seen,
                                'cos_transfer_real': cos_real,
                                'cos_transfer_perm': cos_perm,
                                'norm_ratio_transfer': nrm_ratio}
            log('seed %d donor %s: map R2 seen=%.4f oov=%.4f | cos_seen=%.3f | '
                'cos_real=%.3f cos_perm=%.3f' %
                (s, dname, r2_map_seen, r2_map_oov, float(np.mean(cos_seen)),
                 float(np.mean(cos_real)), float(np.mean(cos_perm))))

        # spans
        def qr_span(mat_rows):
            Q, _ = np.linalg.qr(mat_rows.T.astype(np.float32))
            return Q[:, :mat_rows.shape[0]].astype(np.float32)

        Q_G0 = Qs
        Q_T1 = qr_span(np.concatenate([d_cls, d_hat['T1_14b']], 0))
        Q_T2 = qr_span(np.concatenate([d_cls, d_hat['T2_glm']], 0))
        Q_T3 = qr_span(np.concatenate([d_cls, d_hat['T1_14b'], d_hat['T2_glm']], 0))
        Q_T1r = qr_span(np.concatenate([d_cls, d_hat['T1_14b_perm']], 0))
        Q_T2r = qr_span(np.concatenate([d_cls, d_hat['T2_glm_perm']], 0))
        QMAP = {'G0': Q_G0, 'T1_14b': Q_T1, 'T2_glm': Q_T2, 'T3_dual': Q_T3,
                'T1r_14b': Q_T1r, 'T2r_glm': Q_T2r}

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
        seed_rec['_xfer_diag'] = xfer_diag
        per_seed[s] = seed_rec
        log('seed %d: G0 d40=%.3e | T1 r2=%.4f SCLS=%.3f ENT=%.3f | '
            'T2 r2=%.4f SCLS=%.3f ENT=%.3f | T3 r2=%.4f SCLS=%.3f ENT=%.3f | '
            'T1r SCLS=%.3f T2r SCLS=%.3f' %
            (s, d40,
             seed_rec['T1_14b']['r2_oov'], seed_rec['T1_14b']['fam_fid']['SCLS'],
             seed_rec['T1_14b']['fam_fid']['ENT'],
             seed_rec['T2_glm']['r2_oov'], seed_rec['T2_glm']['fam_fid']['SCLS'],
             seed_rec['T2_glm']['fam_fid']['ENT'],
             seed_rec['T3_dual']['r2_oov'], seed_rec['T3_dual']['fam_fid']['SCLS'],
             seed_rec['T3_dual']['fam_fid']['ENT'],
             seed_rec['T1r_14b']['fam_fid']['SCLS'],
             seed_rec['T2r_glm']['fam_fid']['SCLS']))
        log('seed %d: KL G0=%.4f T1=%.4f T2=%.4f T3=%.4f T1r=%.4f T2r=%.4f' %
            (s, seed_rec['G0']['kl_mean'], seed_rec['T1_14b']['kl_mean'],
             seed_rec['T2_glm']['kl_mean'], seed_rec['T3_dual']['kl_mean'],
             seed_rec['T1r_14b']['kl_mean'], seed_rec['T2r_glm']['kl_mean']))
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
    agg['transfer_diag'] = {}
    for dname in set(DONORS.values()):
        keys = list(per_seed[SEEDS_RUN[0]]['_xfer_diag'][dname].keys())
        dd = {}
        for k in keys:
            vals = [per_seed[s]['_xfer_diag'][dname][k] for s in SEEDS_RUN]
            dd[k] = (float(np.mean(vals)) if isinstance(vals[0], (int, float))
                     else [float(np.mean([v[i] for v in vals]))
                           for i in range(len(vals[0]))])
        agg['transfer_diag'][dname] = dd
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
          and (agg['G0']['d_vs_sealed_3240_max'] < REPL_TOL if not SMOKE else True))
    cand = {rn: min(delta[rn]['d_fam_fid']['SCLS'], delta[rn]['d_fam_fid']['ENT'])
            for rn in ('T1_14b', 'T2_glm', 'T3_dual')}
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
    gates = {'G1_replication': bool(g1), 'G2_efficacy': g2,
             'G3_determinism': bool(g3), 'G4_registry_untouched': bool(g4),
             'determinism_diff': det_diff, 'best_recipe': best,
             'd40_vs_sealed_3240': d40_ok}
    allg = [v for k, v in gates.items() if k.startswith('G')]
    verdict = DESIGN['verdicts']['pass'] if all(allg) else DESIGN['verdicts']['partial']
    log('VERDICT: %s gates=%s' % (verdict, gates))

    headline = {
        'scls_sign_flipped': {rn: bool(agg[rn]['fam_fid_mean']['SCLS'] > 0)
                              for rn in ('T1_14b', 'T2_glm', 'T3_dual')},
        'ent_sign_flipped': {rn: bool(agg[rn]['fam_fid_mean']['ENT'] > 0)
                             for rn in ('T1_14b', 'T2_glm', 'T3_dual')},
        'kl_order_best_vs_G0': bool(agg[best]['kl_mean'] < agg['G0']['kl_mean']),
        'beats_random_control': {rn: bool(min(agg[rn]['fam_fid_mean']['SCLS'],
                                              agg[rn]['fam_fid_mean']['ENT'])
                                         > min(min(agg[ct]['fam_fid_mean']['SCLS'],
                                                   agg[ct]['fam_fid_mean']['ENT'])
                                               for ct in CTRL_OF[rn]))
                                 for rn in ('T1_14b', 'T2_glm', 'T3_dual')},
        'transfer_cos_real_mean': {dn: agg['transfer_diag'][dn]['cos_transfer_real']
                                   for dn in agg['transfer_diag']},
    }
    result = {
        'phase': 3242, 'name': NAME, 'mode': 'smoke' if SMOKE else 'full',
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
        npz_data['xfer_cos_real_%s' % dn.replace('-', '_')] = np.array(
            agg['transfer_diag'][dn]['cos_transfer_real'], np.float64)
    npz_path = os.path.join(OUTDIR, 'smoke_xmodel.npz' if SMOKE else 'xmodel.npz')
    np.savez(npz_path, **npz_data)
    log('npz saved: %s (%d KB)' % (os.path.basename(npz_path),
                                   os.path.getsize(npz_path) // 1024))
    log('RUN_COMPLETE %s' % verdict)
    return verdict


if __name__ == '__main__':
    v = main()
    sys.exit(0 if v == DESIGN['verdicts']['pass'] else 2)
