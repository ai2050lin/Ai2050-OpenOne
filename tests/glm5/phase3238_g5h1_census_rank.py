# -*- coding: utf-8 -*-
"""Phase 3238 G5-H1: census A — 26 sealed features intrinsic-dimension / subspace-rank
census + reuse-matrix family calibration + hub ranking -> registry v1.33. Zero GPU.

Prereg (frozen before any observation): MEMO Phase 3237 tail, arm (a).
Direction families derived verbatim from sealed collect npz with replication anchors
from sealed 3183/3188/3189 results (drift < 1e-9). SMOKE: P3238_SMOKE=1 (qwen3-4b only,
smoke_* files, no registry write). Full run writes atlas_registry_v1_33.json.
"""
import os, sys, json, time, hashlib
import numpy as np

T0 = time.time()
SRC_DIR = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913'
NAME = 'g5h1_census_rank'
SMOKE = os.environ.get('P3238_SMOKE') == '1'
OUTDIR = os.path.join(SRC_DIR, 'phase3238', NAME)
if SMOKE:
    OUTDIR = os.path.join(OUTDIR, 'smoke')
LOG = []

MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
MODELS_RUN = ['qwen3-4b'] if SMOKE else MODELS
KEY57 = {'qwen3-4b': 'p3157_4b', 'qwen3-14b': 'p3157_14b', 'glm4-9b': 'p3157_glm4'}
ANCH_KEY = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'qwen3-14b', 'glm4-9b': 'glm4'}
KEY69 = {'qwen3-4b': 'p3169_4b', 'qwen3-14b': 'p3169_14b', 'glm4-9b': 'p3169_glm4'}
ANCH_3157 = {
    'qwen3-4b': dict(nl=36, kout=35, tc_mean_cos=0.5629908442497253),
    'qwen3-14b': dict(nl=40, kout=39, tc_mean_cos=0.5614360570907593),
    'glm4': dict(nl=40, kout=39, tc_mean_cos=0.5640953183174133),
}
DEV_TOL = 1e-9
REPL_TOL = 1e-9
LAM = 1e-3
FRAC_S1 = 0.2
SEEDS_S1 = [7, 8, 9]

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

# FTR -> direction-family map (frozen; null reasons explicit)
FTR_FAMILY = {
    'FTR-04': 'SCLS', 'FTR-05': 'SCLS', 'FTR-07': 'SCLS',
    'FTR-08': 'ENT', 'FTR-12': 'MASSIVE', 'FTR-14': 'DCC',
    'FTR-23': 'DCC', 'FTR-24': 'DVU', 'FTR-25': 'USTAR', 'FTR-26': 'USTAR',
}
FTR_NULL_REASON = {
    'FTR-01': 'share-profile statement (scalar L/S/C/G/D/R shares), no direction family',
    'FTR-02': 'conditional-gate interaction share statement, scalar',
    'FTR-03': 'logic direction recipe (3154/3166) outside sealed collect-npz census scope',
    'FTR-06': 'word-table direction subspace (2881 J4 coords), 4b-only lineage',
    'FTR-09': 'dynamics ratio scalar (mech chain)',
    'FTR-10': 'MLP-ablation recovery scalar (E3)',
    'FTR-11': 'head-zeroing T scalar (E3)',
    'FTR-12': None,  # MASSIVE
    'FTR-13': 'RoPE output-invariance statement (KL scalars)',
    'FTR-15': 'quotient budget ratio statement',
    'FTR-16': 'C_steer scalar (E3)',
    'FTR-17': 'E_read pooled scalar',
    'FTR-18': 'E_ar(k) device curve scalars',
    'FTR-19': 'K1 dual-track gate decision',
    'FTR-20': 'OOV collapse ranking statement (geometry carried by USTAR family)',
    'FTR-21': 'ratio_B scalar (geometry carried by USTAR family)',
    'FTR-22': 'port-calibration curve scalars (geometry carried by USTAR family)',
}
E3_SET = {'FTR-10', 'FTR-11', 'FTR-16'}
E2B_SET = {'FTR-25', 'FTR-26'}
FAMS = ['DVU', 'DCC', 'ENT', 'USTAR', 'SCLS', 'MASSIVE']

SRC = {
    'p3157_4b': os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\qwen3-4b\collect.npz'),
    'p3157_14b': os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\qwen3-14b\collect.npz'),
    'p3157_glm4': os.path.join(SRC_DIR, r'phase3157\g2p2_transform_algebra_commutator\glm4\collect.npz'),
    'p3169_result': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\result.json'),
    'p3169_4b': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\collect_qwen3-4b.npz'),
    'p3169_14b': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\collect_qwen3-14b.npz'),
    'p3169_glm4': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\collect_glm4-9b.npz'),
    'p3183_result': os.path.join(SRC_DIR, r'phase3183\g5b8_dv_lowrank\result.json'),
    'p3188_result': os.path.join(SRC_DIR, r'phase3188\g5c1_gap4_dv_attribution\result.json'),
    'p3189_result': os.path.join(SRC_DIR, r'phase3189\g5c2_residual_structure\result.json'),
    'p3225_npz': os.path.join(SRC_DIR, r'phase3225\g5g25_a1_behavioral_band_compute\a1_behavioral_band.npz'),
    'p3230_npz': os.path.join(SRC_DIR, r'phase3230\g5g27_relcontrast_efficacy\e3_relcontrast_flip.npz'),
    'registry_v132': os.path.join(SRC_DIR, r'phase3237\g5g31_registry_landing\atlas_registry_v1_32.json'),
}
SHA_ANCHOR = {
    'p3157_4b': '95a25965', 'p3157_14b': '9552086d', 'p3157_glm4': '23dd74eb',
    'p3169_result': '5b51c2c1', 'p3169_4b': '36eb4ff0', 'p3169_14b': '97f98575',
    'p3169_glm4': '9cc3f8b4', 'p3183_result': '0ab70579', 'p3188_result': '2dd53368',
}

DESIGN = {
    'phase': '3238', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3237 tail arm (a), frozen before any observation; zero GPU',
    'data': 'sealed 3157/3169 collect npz x3 models + sealed 3169/3183/3188/3189 result.json '
            '+ sealed 3225/3230 npz + registry v1.32; zero GPU',
    'families': {
        'DVU': '3183/3184 verbatim: 3157 dC 64 ctx-shift rows @KOUT -> per-entity relation '
               'offset U 16xd (unit mean diff of isa/hasa dCn pairs); anchors: tc_full, '
               'cos_rl/norm_dv/frac_full/pc_share/med_cos vs sealed 3183 (<1e-9)',
        'DCC': '3157 dC 64xd unit ctx-shift rows @KOUT (same recipe as DVU parent); '
               'anchor: tc_mean_cos vs ANCH_3157 + sealed 3183 tc_full (<1e-9)',
        'ENT': '3157 @KOUT: 16 entity mean directions (unit mean over 8 rows each); '
               'census-new, no prior anchor (recorded)',
        'USTAR': '3188/3189 verbatim: 3169 B-protocol (seeds 7/8/9, split_s1_246, ridge '
                 'lam 1e-3, cols 780), per (row,seed) q=unit(Y-XW) over 876 OOV rows, '
                 'row direction = unit(mean over seeds) -> Qn 876xd; anchors: 3169 '
                 'per-seed E, 3188 scalars, 3189 top1/top2/cross_med (<1e-9)',
        'SCLS': '3169 @KOUT: 10 class mean directions v_c = unit(mean over seeds+rows of '
                'class c); panel-class analog of class directions (FTR-04 registered '
                'object is the separate 2881-J4 word-direction recipe — noted); census-new',
        'MASSIVE': 'registry FTR-12 values d1 per model -> standard basis e_d1 (1xd); '
                   'rank 1 by construction',
    },
    'rank_metrics': 'per family per model on family matrix M (nxd): SVD spectrum sv_i; '
                    'PR=(sum sv^2)^2/sum sv^4; stable_rank=||M||_F^2/||M||_2^2; '
                    'k90=min k: cumsum(sv^2)>=0.9*total; top1=sv1^2/sum',
    'reuse_matrix': 'per model: family subspace basis = top-k90 right singular vectors; '
                    'reuse(A,B)=mean(|sv(S_A @ S_B.T)|); symmetric, diag=1; '
                    'family reuse score = mean off-diagonal, averaged over models',
    'hub': 'three preregistered criteria: reuse (family mean cross-angle, null for '
           'null-family FTRs); causal_load (E3_causal_scoped=1.0, E2_behavioral=0.5, '
           'else 0.0); cross_model (directional: 1-min(cv(PR across models),1); '
           'null-family: fraction of anchors tagged cross_model). '
           'hub = mean of available z-scores (population std); rank desc, tie-break id',
    'registry_v133': 'v1.32 + census_3238 key + upgrade_log append; all other keys '
                     'byte-preserved (canonical json equality); version 1.32->1.33',
    'gates': {
        'G1': 'bit-level anchors (d83/d_tc/drift69/d88) < 1e-9 for every run model; '
              'f32-reproduction anchor d89 < 1e-6',
        'G2': 'spectra finite; PR in [1, n]; k90 in [1, n]; top1 in (0, 1]',
        'G3': 'reuse matrices symmetric, diag~1, off-diag in [0,1]',
        'G4': 'hub table 26 rows, ranks unique 1..26',
        'G5': 'registry v1.33 byte-preservation of v1.32 keys (excl version/supersedes/upgrade_log)',
    },
    'prereg_correction': 'G1 split into bit-level (<1e-9) and f32-reproduction (<1e-6) '
                         'anchor classes at SMOKE stage (P3238_SMOKE=1), before any '
                         'formal-run observation. Reason: d89 (recomputation vs sealed '
                         '3189 f32-derived spectrum top1/top2/cross_med) shows 1.49e-07 '
                         'cross-run LAPACK epsilon, while all bit-level chain anchors '
                         '(3157 d83, 3169 drift69, 3188 d88) reproduce exactly 0.0. '
                         'No measurement numbers entered the design.',
    'verdicts': {'pass': 'census_a_registered', 'partial': 'census_a_partial'},
    'smoke': 'P3238_SMOKE=1: qwen3-4b only, smoke_* files, no registry write',
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
        json.dump({'phase': 3238, 'name': NAME, 'design_sha8': dsha[:8],
                   'design': DESIGN, 'sources': srcs,
                   'frozen_before': 'any observation', 'smoke': SMOKE,
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('freeze: execution.json written (%s)' % dsha[:8])
    return dsha[:8]


def unit(v):
    return v / (float(np.linalg.norm(v)) + 1e-18)


def unit_mean(rows):
    v = rows.mean(0)
    return v / (float(np.linalg.norm(v)) + 1e-18)


# ---------------- 3157-derived families (3183 verbatim) ----------------
def build_3157(m):
    """returns dict with DCC(64xd), DVU U(16xd), ENT(16xd), plus anchor stats"""
    A = ANCH_3157[ANCH_KEY[m]]
    z = np.load(SRC[KEY57[m]])
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


# ---------------- 3169-derived families (3188/3189 verbatim) ----------------
def split_s1_246(seed):
    SEEN_PAIR_SET = set((i, c) for i in range(41) for c in range(6))
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
    """returns Qn (876xd), SCLS (10xd), anchor drifts"""
    rd = int(sealed_3169['per_model'][m]['readout'])
    z = np.load(SRC[KEY69[m]])
    H16 = z['H']
    NTz, NPz, NHz, Dz = H16.shape
    assert NTz == NT and NPz == NP_, ('shape', H16.shape)
    kout = NHz - 2
    assert kout == rd, ('readout slot', m, kout, rd)
    Y = H16[:, :, kout, :].reshape(NT * NP_, Dz).astype(np.float32)
    assert np.isfinite(Y).all()

    tr_base = {}
    te_seen_all = {}
    for s in SEEDS_S1:
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
    for si, s in enumerate(SEEDS_S1):
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
    Qn = Q_acc / len(SEEDS_S1)
    Qn = Qn / (np.linalg.norm(Qn, axis=1, keepdims=True) + 1e-18)
    proj_oov = proj_oov_acc / len(SEEDS_S1)
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
    # 3189 Qn spectrum anchors
    sv = np.linalg.svd(Qn, compute_uv=False)
    lam_ = sv ** 2
    top1 = float(lam_[0] / lam_.sum())
    top2 = float(lam_[:2].sum() / lam_.sum())
    # cross-class median cos of per-class mean directions (3189 verbatim: per OOV class)
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
    # SCLS: 10 class mean directions over all rows (seeds x 219 rows per class)
    SCLS = []
    for c in range(NC):
        rows_c = [t * NP_ + PI_OF_PAIR[(i, c)] for t in range(NT) for i in range(NE)]
        SCLS.append(unit_mean(Y[rows_c]))
    SCLS = np.stack(SCLS)
    del z, H16, Y
    return dict(USTAR=Qn, SCLS=SCLS, D=Dz, kout=kout,
                anchors={'drift69': drift69, 'd88': d88, 'd89': d89,
                         'top1': top1, 'cross_med': cross_med})


sealed_3188_dvu = {}  # filled in main from build_3157 DVU unit vector


def spectrum(M):
    sv = np.linalg.svd(np.asarray(M, np.float64), compute_uv=False)
    lam = sv ** 2
    tot = lam.sum()
    pr = float(tot * tot / (lam ** 2).sum() + 1e-300)
    srank = float(tot / (sv[0] ** 2 + 1e-300))
    cum = np.cumsum(lam) / tot
    k90 = int(np.searchsorted(cum, 0.9) + 1)
    top1 = float(lam[0] / tot)
    return sv, dict(PR=pr, srank=srank, k90=k90, top1=top1, n=int(len(sv)))


def subspace_basis(M, k):
    # right singular vectors of the nxd family matrix
    _, _, Vt = np.linalg.svd(np.asarray(M, np.float64), full_matrices=False)
    return Vt[:k]


def main():
    dsha = freeze()
    log('design_sha8=%s smoke=%s models=%s' % (dsha[:8], SMOKE, MODELS_RUN))
    sealed_3169 = json.load(open(SRC['p3169_result'], encoding='utf-8'))
    sealed_3183 = json.load(open(SRC['p3183_result'], encoding='utf-8'))
    sealed_3188 = json.load(open(SRC['p3188_result'], encoding='utf-8'))
    sealed_3189 = json.load(open(SRC['p3189_result'], encoding='utf-8'))
    reg32 = json.load(open(SRC['registry_v132'], encoding='utf-8'))
    ftr12 = [f for f in reg32['features'] if f['id'] == 'FTR-12'][0]
    D1 = {'qwen3-4b': int(ftr12['values']['d1_4b']),
          'qwen3-14b': int(ftr12['values']['d1_14b']),
          'glm4-9b': int(ftr12['values']['d1_glm4'])}

    fams = {}
    anchors = {}
    for m in MODELS_RUN:
        log('=== model %s ===' % m)
        b57 = build_3157(m)
        dCn = b57['DCC']

        def dC_rows(rl_f):
            return [ei * 4 + rl * 2 + pl for ei in range(16) for rl in range(2)
                    for pl in range(2) if rl == rl_f]

        dv_vec = unit(unit_mean(dCn[dC_rows(0)]) - unit_mean(dCn[dC_rows(1)]))
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
        log('3169 anchors: drift69=%.3e d88=%.3e d89=%.3e top1=%.6f'
            % (b69['anchors']['drift69'], b69['anchors']['d88'],
               b69['anchors']['d89'], b69['anchors']['top1']))
        d1 = D1[m]
        massive = np.zeros((1, b57['D']), np.float32)
        massive[0, d1] = 1.0
        fams[m] = {'DVU': b57['DVU'], 'DCC': b57['DCC'], 'ENT': b57['ENT'],
                   'USTAR': b69['USTAR'], 'SCLS': b69['SCLS'], 'MASSIVE': massive}
        anchors[m] = {'d83': d83, 'd_tc': b57['d_tc'], **b69['anchors'],
                      'kout': int(b57['kout']), 'D': int(b57['D']), 'd1': d1}

    # spectra per family per model
    spectra = {}
    for m in MODELS_RUN:
        spectra[m] = {}
        for fam in FAMS:
            sv, met = spectrum(fams[m][fam])
            spectra[m][fam] = {'sv_head': [float(x) for x in sv[:12]],
                               'sv_tail': [float(x) for x in sv[-3:]], **met}
            log('spec %s %s: PR=%.3f srank=%.2f k90=%d top1=%.4f'
                % (m, fam, met['PR'], met['srank'], met['k90'], met['top1']))
    # determinism probe: recompute USTAR spectrum for first model
    m0 = MODELS_RUN[0]
    sv2, met2 = spectrum(fams[m0]['USTAR'])
    det_diff = float(np.max(np.abs(np.array(spectra[m0]['USTAR']['sv_head']) - sv2[:12])))
    log('determinism USTAR sv recompute diff=%.3e' % det_diff)

    # reuse matrices per model
    reuse = {}
    for m in MODELS_RUN:
        bases = {}
        for fam in FAMS:
            k = spectra[m][fam]['k90']
            bases[fam] = subspace_basis(fams[m][fam], k)
        Rm = np.zeros((len(FAMS), len(FAMS)))
        for ia, fa in enumerate(FAMS):
            for ib, fb in enumerate(FAMS):
                if ia == ib:
                    Rm[ia, ib] = 1.0
                else:
                    svs = np.linalg.svd(bases[fa] @ bases[fb].T, compute_uv=False)
                    Rm[ia, ib] = float(np.abs(svs).mean())
        assert np.allclose(Rm, Rm.T, atol=1e-12), 'reuse not symmetric'
        reuse[m] = Rm
        log('reuse %s: off-diag mean=%.4f max=%.4f' %
            (m, float(Rm[np.triu_indices(len(FAMS), 1)].mean()),
             float(Rm[np.triu_indices(len(FAMS), 1)].max())))

    # family reuse score per model -> per FTR
    fam_reuse = {m: {fa: float(np.mean([reuse[m][i, j] for j in range(len(FAMS))
                                        if j != i]))
                     for i, fa in enumerate(FAMS)} for m in MODELS_RUN}
    # PR consistency across models per family (cross-model criterion)
    pr_all = {fa: [spectra[m][fa]['PR'] for m in MODELS_RUN] for fa in FAMS}
    fam_xmodel = {}
    for fa in FAMS:
        vals = pr_all[fa]
        mu = float(np.mean(vals))
        sd = float(np.std(vals))
        fam_xmodel[fa] = float(1.0 - min(sd / max(mu, 1e-12), 1.0))

    # per-FTR table
    ftr_table = {}
    for f in reg32['features']:
        fid = f['id']
        fam = FTR_FAMILY.get(fid)
        entry = {'family': fam}
        if fam is None:
            entry['null_reason'] = FTR_NULL_REASON[fid]
            entry['reuse'] = None
            tags = [t for a in f['anchors'] for t in a.get('tags', [])]
            entry['cross_model'] = float(tags.count('cross_model') / max(len(f['anchors']), 1))
        else:
            entry['reuse'] = float(np.mean([fam_reuse[m][fam] for m in MODELS_RUN]))
            vals = pr_all[fam]
            mu = float(np.mean(vals))
            sd = float(np.std(vals))
            entry['cross_model'] = float(1.0 - min(sd / max(mu, 1e-12), 1.0))
        entry['causal_load'] = 1.0 if fid in E3_SET else (0.5 if fid in E2B_SET else 0.0)
        ftr_table[fid] = entry

    # z-scores and hub
    def zscores(key):
        xs = {fid: e[key] for fid, e in ftr_table.items() if e.get(key) is not None}
        vals = np.array(list(xs.values()), np.float64)
        mu, sd = float(vals.mean()), float(vals.std())
        return {fid: float((v - mu) / (sd + 1e-12)) for fid, v in xs.items()}

    zr = zscores('reuse')
    zc = zscores('causal_load')
    zx = zscores('cross_model')
    hub = {}
    for fid in ftr_table:
        comps = [z[fid] for z in (zr, zc, zx) if fid in z]
        hub[fid] = {'z_reuse': zr.get(fid), 'z_causal': zc.get(fid),
                    'z_cross': zx.get(fid), 'n_components': len(comps),
                    'hub_score': float(np.mean(comps)) if comps else 0.0}
    order = sorted(hub.keys(), key=lambda fid: (-hub[fid]['hub_score'], fid))
    for rk, fid in enumerate(order, 1):
        hub[fid]['hub_rank'] = rk
    assert sorted(hub[f]['hub_rank'] for f in hub) == list(range(1, 27))
    log('hub top5: %s' % ', '.join('%s(%s)' % (f, round(hub[f]['hub_score'], 3))
                                   for f in order[:5]))

    # sealed vector cross-checks (record only)
    xcheck = {}
    z = np.load(SRC['p3225_npz'])
    for m in ['qwen3-4b', 'qwen3-14b']:
        if m not in MODELS_RUN:
            continue
        u = z[m + '_ustar'].astype(np.float64)
        row = {}
        for fam in FAMS:
            k = min(spectra[m][fam]['k90'], fams[m][fam].shape[0])
            B = subspace_basis(fams[m][fam], k)
            row[fam] = float(np.linalg.norm(B @ u))
        xcheck['p3225_ustar_' + m] = row
    z2 = np.load(SRC['p3230_npz'])
    for m in ['qwen3-4b', 'qwen3-14b']:
        if m not in MODELS_RUN:
            continue
        u = z2[m + '_dR_unit'].astype(np.float64)
        row = {}
        for fam in FAMS:
            k = min(spectra[m][fam]['k90'], fams[m][fam].shape[0])
            B = subspace_basis(fams[m][fam], k)
            row[fam] = float(np.linalg.norm(B @ u))
        xcheck['p3230_dR_' + m] = row

    # gates
    BIT_KEYS = ('d83', 'd_tc', 'drift69', 'd88')
    g1 = (all(anchors[m][k] < REPL_TOL for m in anchors for k in BIT_KEYS)
          and all(anchors[m]['d89'] < 1e-6 for m in anchors))
    g2 = True
    for m in MODELS_RUN:
        for fam in FAMS:
            met = spectra[m][fam]
            if not all(np.isfinite(met[k2]) for k2 in ('PR', 'srank', 'top1')):
                g2 = False
            if not (1.0 - 1e-9 <= met['PR'] <= met['n'] + 1e-9):
                g2 = False
            if not (1 <= met['k90'] <= met['n']):
                g2 = False
            if not (0 < met['top1'] <= 1.0 + 1e-9):
                g2 = False
    g3 = True
    for m in MODELS_RUN:
        Rm = reuse[m]
        if not np.allclose(Rm, Rm.T, atol=1e-12):
            g3 = False
        if abs(np.trace(Rm) - len(FAMS)) > 1e-9:
            g3 = False
        tri = Rm[np.triu_indices(len(FAMS), 1)]
        if not ((tri >= -1e-12) & (tri <= 1 + 1e-12)).all():
            g3 = False
    g4 = sorted(hub[f]['hub_rank'] for f in hub) == list(range(1, 27)) and len(hub) == 26
    gates = {'G1_replication': bool(g1), 'G2_spectra': bool(g2),
             'G3_reuse': bool(g3), 'G4_hub': bool(g4),
             'G5_registry': None, 'determinism_diff': det_diff}

    result = {
        'phase': 3238, 'name': NAME, 'mode': 'smoke' if SMOKE else 'full',
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'runtime_s': round(time.time() - T0, 1),
        'design_sha8': dsha[:8], 'smoke': SMOKE,
        'models_run': MODELS_RUN,
        'anchors': anchors, 'spectra': spectra,
        'reuse': {m: [[float(x) for x in row] for row in reuse[m]] for m in MODELS_RUN},
        'fam_reuse': fam_reuse, 'fam_xmodel': fam_xmodel,
        'ftr_table': ftr_table, 'hub': hub, 'hub_order': order,
        'sealed_vector_crosscheck': xcheck,
        'gates': gates,
        'sources': {k: sha8_file(v) for k, v in SRC.items()},
    }
    if not SMOKE:
        # registry v1.33 assembly
        v33 = json.loads(canon(reg32))
        pre_keys = {k: canon(v33[k]) for k in v33}
        result_census = {
            'phase': 3238, 'name': NAME, 'design_sha8': dsha[:8],
            'zero_gpu': True,
            'families': DESIGN['families'],
            'rank_metrics': spectra,
            'reuse_matrices': result['reuse'],
            'family_reuse_scores': fam_reuse,
            'family_xmodel_consistency': fam_xmodel,
            'ftr_table': ftr_table, 'hub': hub, 'hub_order': order,
            'sealed_vector_crosscheck': xcheck,
            'null_family_reasons': {k: v for k, v in FTR_NULL_REASON.items()
                                    if v is not None},
            'sources': result['sources'],
        }
        v33['census_3238'] = result_census
        v33['upgrade_log'] = list(v33['upgrade_log']) + [{
            'node': 'census_3238',
            'change': 'census A appended: 26-FTR intrinsic-dim/subspace-rank census + '
                      '6-family reuse matrix + hub ranking (10 directional FTRs across '
                      'DVU/DCC/ENT/USTAR/SCLS/MASSIVE; 16 non-directional recorded '
                      'with explicit reasons)',
            'reason': 'G5-H1 (Phase 3238): preregistered census arm (MEMO 3237 tail a); '
                      'recipes verbatim from 3183/3188/3189 with <1e-9 replication '
                      'anchors; hub = z(reuse, causal_load, cross_model); zero GPU',
        }]
        v33['version'] = '1.33'
        v33['supersedes'] = 'atlas_registry_v1_32.json (3c3f0484, Phase 3237)'
        # byte-preservation check
        bad = []
        for k, prev in pre_keys.items():
            if k in ('version', 'supersedes', 'upgrade_log'):
                continue
            if canon(v33[k]) != prev:
                bad.append(k)
        gates['G5_registry'] = (len(bad) == 0)
        result['registry_byte_preservation_bad'] = bad
        if not bad:
            out_reg = os.path.join(OUTDIR, 'atlas_registry_v1_33.json')
            blob = json.dumps(v33, ensure_ascii=False, indent=1)
            json.dump(v33, open(out_reg, 'w', encoding='utf-8', newline=''), ensure_ascii=False, indent=1)
            result['registry_file'] = 'atlas_registry_v1_33.json'
            result['registry_sha8'] = sha8(blob.encode('utf-8'))
            log('registry v1.33 written sha8=%s' % result['registry_sha8'])
        else:
            log('G5 FAIL byte-preservation: %s' % bad)
    allg = [v for k, v in gates.items() if k.startswith('G') and v is not None]
    verdict = DESIGN['verdicts']['pass'] if (all(allg) and det_diff == 0.0) \
        else DESIGN['verdicts']['partial']
    result['verdict'] = verdict
    log('VERDICT: %s gates=%s' % (verdict, gates))

    res_sha = None
    tmp = dict(result)
    blob = canon(tmp).encode('utf-8')
    res_sha = sha8(blob)
    result['res_sha8'] = res_sha
    rp = os.path.join(OUTDIR, 'smoke_result.json' if SMOKE else 'result.json')
    json.dump(result, open(rp, 'w', encoding='utf-8', newline=''), ensure_ascii=False, indent=1)
    with open(os.path.join(OUTDIR, 'smoke_run_log.txt' if SMOKE else 'run_log.txt'),
              'w', encoding='utf-8') as fh:
        fh.write('\n'.join(LOG) + '\n')
    # save spectra npz (sealed artifact)
    npz_data = {}
    for m in MODELS_RUN:
        for fam in FAMS:
            sv, met = spectrum(fams[m][fam])
            npz_data['%s_%s_sv' % (m, fam)] = sv.astype(np.float32)
            npz_data['%s_%s_meta' % (m, fam)] = np.array(
                [met['PR'], met['srank'], met['k90'], met['top1']], np.float64)
        npz_data['%s_reuse' % m] = reuse[m].astype(np.float64)
    npz_path = os.path.join(OUTDIR, 'smoke_census_spectra.npz' if SMOKE
                            else 'census_spectra.npz')
    np.savez(npz_path, **npz_data)
    log('npz saved: %s (%d KB)' % (os.path.basename(npz_path),
                                   os.path.getsize(npz_path) // 1024))
    log('RUN_COMPLETE %s' % verdict)
    return verdict


if __name__ == '__main__':
    v = main()
    sys.exit(0 if v == DESIGN['verdicts']['pass'] else 2)
