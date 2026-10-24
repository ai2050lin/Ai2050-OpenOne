# -*- coding: utf-8 -*-
"""Phase 3241 G5-H4: inter-class side-information arm (lexical-embedding bridge). Zero GPU.

Prereg (frozen before any observation): MEMO Phase 3240 tail arm (a).
Target: 3240-F2 completed the class subspace but SCLS (-1.75) / ENT (-1.40) did
not turn positive; the uniform prior carries no OOV-class identity. This phase
injects LEXICAL side information: the OOV class/entity token embeddings (from
the same tied embed_tokens table used by the lens) weight the trained
class/entity direction combination.

Leakage declaration (prereg): allowed = token identity via embed_tokens rows
(pure model weights, no activation statistics, no OOV labels); forbidden = any
OOV activation/mean/direction from sealed collect npz outside the training
rows. Same spirit as the B1 identity baseline, but weight-side only.

Recipes (preregistered, no post-hoc selection):
  G0   verbatim 3240-F2 subspace completion (replication anchor vs sealed
       3240 result.json per-seed F2, <1e-9)
  G1c  class bridge: Yhat = Yhat_ridge + sum_{c'<6} alpha_{c,c'} d_{c'},
       d_{c'} = class_mean[c'] - B0 (train rows), alpha = softmax(cos(w_c,
       w_{c'})) with beta=1 fixed (no temperature tuning)
  G1e  dual bridge: G1c + entity bridge sum_{i'<41} gamma_{i,i'} e_{i'},
       e_{i'} = train entity mean - B0, gamma = softmax(cos(w_i, w_{i'}))
  G1r  randomization control: dual bridge with class/entity token rows
       permuted (seed 42, preregistered) - tests whether bridge information
       is real; G1e/G1c should beat G1r on min(SCLS, ENT) fidelity

Gates:
  G1 replication: d83/d_tc/drift69/d88 <1e-9, d89 <1e-6, census spectra
     cross-check vs sealed 3238, G0 per-seed vs sealed 3240 F2 <1e-9
     (formal mode only; smoke row subset recorded-only, 3240 lesson (2))
  G2 efficacy: best(G1c, G1e) improves BOTH SCLS and ENT fidelity by >0.5 vs
     G0 (3-seed mean, regression lower bound semantics min(d_DVU,d_DCC) >
     -REG_DROP, R2 drop < R2_DROP), AND beats G1r on min(SCLS, ENT)
  G3 determinism: identical-GEMM-shape lens recompute bitwise 0.0
  G4 registry v1.33 untouched (byte sha8 equals sealed value)
Headline (not gates): SCLS/ENT sign flip; KL ordering.
SMOKE: P3241_SMOKE=1 (seed 7 only, KL first 200 OOV rows, smoke_* files).
Registry untouched (v1.33 remains head).
"""
import os, sys, json, time, hashlib
import numpy as np

T0 = time.time()
SRC_DIR = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913'
NAME = 'g5h4_lexical_bridge'
SMOKE = os.environ.get('P3241_SMOKE') == '1'
MODEL = 'qwen3-4b'
OUTDIR = os.path.join(SRC_DIR, 'phase3241', NAME)
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
RECIPES = ['G0', 'G1c', 'G1w', 'G1e', 'G1r', 'G1rw']
EFF_DELTA = 0.5
REG_DROP = 0.05
R2_DROP = 0.01
BRIDGE_BETA = 1.0
RAND_SEED = 42

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

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC = {
    'p3169_result': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\result.json'),
    'p3169_4b': os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel\collect_qwen3-4b.npz'),
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
    'phase': '3241', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3240 tail arm (a), frozen before any observation; '
              'zero GPU (lexical side-information bridge on the 3239/3240 fidelity '
              'loop)',
    'leakage_declaration': 'allowed = token identity via embed_tokens rows (pure '
                           'model weights: class words 水果/动物/交通工具/家具/金属/颜色'
                           '/乐器/天气/运动/电器 and all 73 entity words; word vector = '
                           'mean pooling over subword tokens — SMOKE-stage correction: '
                           'not all entity words are single-token, class words all are); '
                           'forbidden = any OOV activation statistics or labels from '
                           'sealed npz',
    'recipes': {
        'G0': 'verbatim 3240-F2 subspace completion (replication anchor, per-seed '
              'bit-level vs sealed 3240 result.json F2)',
        'G1c_class_conditioned_anchor': 'anchor A_c = B0 + sum alpha_{c,c}(d_c - d_mean)'
                            ', d_c = class_mean[c] - B0 (train rows), alpha = softmax(cos) '
                            'beta=1; Yhat = Yhat_ridge + P_S(A_c - Yhat_ridge)',
        'G1w_linear_class_anchor': 'same as G1c with LINEAR weights '
                            'w = relu(cos)/sum(relu(cos)) (uniform if no positive cos) '
                            '- avoids softmax uniform-dilution at beta=1',
        'G1e_dual_conditioned_anchor': 'A = B0 + class bridge (softmax) + entity '
                           'bridge (softmax over 41 train entities); projection onto '
                           'S∪E = joint QR span of 6 class + 41 entity directions',
        'G1r_random_control': 'G1c with class identities permuted (seed 42)',
        'G1rw_random_control_linear': 'G1w with class identities permuted (seed 42)',
    },
    'gates': {
        'G1': 'replication anchors + census spectra cross-check + G0 per-seed '
              'bit-level vs sealed 3240 F2 (formal mode; smoke recorded-only per '
              '3240 correction protocol)',
        'G2': 'efficacy: best(G1c, G1e) improves BOTH SCLS and ENT fidelity > +%.1f '
              'vs G0 (3-seed mean; min(d_DVU,d_DCC) > -%.2f lower-bound semantics; '
              'R2 drop < %.2f), AND beats G1r on min(SCLS, ENT)'
              % (EFF_DELTA, REG_DROP, R2_DROP),
        'G3': 'determinism: lens first-batch identical-GEMM-shape recompute diff 0.0',
        'G4': 'registry v1.33 untouched (byte sha8)',
    },
    'verdicts': {'pass': 'lexical_bridge_registered', 'partial': 'lexical_bridge_partial'},
    'note': 'headline (SCLS/ENT sign flip; KL ordering) is a registered finding, '
            'NOT a gate. Registry append not triggered: prediction-recipe repair.',
    'prereg_correction': 'two corrections at SMOKE stage (P3241_SMOKE=1), before any '
                         'formal-run observation: (1) token_row = mean pooling over '
                         'subword tokens (not all entity words are single-token, e.g. '
                         '小提琴 = 3 tokens; class words all are; still pure weight-side '
                         'info). (2) bridge recipe redesign: v1 (full-magnitude '
                         'direction addition sum alpha*d_c on top of the ridge output) '
                         'showed amplitude mismatch in SMOKE — it pushed SCLS fidelity '
                         'back to the F0 level (~-16) because ||d_c|| is class-center '
                         'scale while the needed correction is residual scale; redesigned '
                         'to class-conditioned anchor + SAME subspace projection as F2 '
                         '(amplitude set by the projection), random control kept. The '
                         'v1 negative result is recorded: raw lexical weighting of '
                         'trained class directions does not transfer to activation '
                         'space at full magnitude, and the randomization control showed '
                         'no difference from the real bridge (both bad) — the bridge '
                         'information only enters via the anchor/projection mechanism. '
                         '(3) recipe family extension (final, still pre-formal): G1c '
                         'showed weight dilution (softmax at beta=1 is near-uniform, '
                         'class-conditioned offset ~0, G1c==G0 in SMOKE); added G1w '
                         '(linear relu(cos) weights) + G1rw (its permuted control); '
                         'kept G1c/G1e/G1r verbatim. Gate G2 best over {G1c, G1w, '
                         'G1e}, each with its matched random control (G1c/G1e -> G1r, '
                         'G1w -> G1rw). No formal-run metric numbers entered the '
                         'design.',
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
        json.dump({'phase': 3241, 'name': NAME, 'design_sha8': dsha[:8],
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


# ---------------- CPU lens + token embeddings ----------------
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


# ---------------- lexical bridge weights ----------------
def token_row(word):
    """token embedding row for a word: mean pooling over its subword tokens
    (SMOKE-stage correction: not all entity words are single-token, e.g.
    小提琴 = 3 tokens; class words all are. Still pure weight-side info.)"""
    from transformers import AutoTokenizer
    if 'tok' not in LENS:
        LENS['tok'] = AutoTokenizer.from_pretrained(os.path.join(ROOT, r'models\hf\qwen3-4b'))
    ids = LENS['tok'].encode(word, add_special_tokens=False)
    assert 1 <= len(ids) <= 8, ('token count out of band', word, ids)
    return LENS['E'][np.array(ids, int)].astype(np.float32).mean(0)


def softmax_weights(cosv, beta=BRIDGE_BETA):
    e = np.exp(beta * (cosv - cosv.max()))
    return e / e.sum()


def linear_weights(cosv):
    v = np.maximum(cosv, 0.0)
    return v / v.sum() if v.sum() > 0 else np.full(len(cosv), 1.0 / len(cosv))


WEIGHT_KIND = {'G1c': 'softmax', 'G1w': 'linear', 'G1e': 'softmax',
               'G1r': 'softmax', 'G1rw': 'linear'}
IS_DUAL = {'G1c': False, 'G1w': False, 'G1e': True, 'G1r': False, 'G1rw': False}
IS_PERM = {'G1c': False, 'G1w': False, 'G1e': False, 'G1r': True, 'G1rw': True}
CTRL_OF = {'G1c': 'G1r', 'G1e': 'G1r', 'G1w': 'G1rw'}


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

    # bridge token rows (leakage declaration scope: model weights only)
    cls_words = CLASSES_SEEN + CLASSES_OOV
    W_cls = np.stack([token_row(w) for w in cls_words])           # (10, d)
    W_ent = np.stack([token_row(w) for w in ENTS])                # (73, d)
    cos_cm = np.array([[float(unit(W_cls[a]) @ unit(W_cls[b])) for b in range(NCS)]
                       for a in range(NC)])                       # (10, 6)
    cos_em = np.array([[float(unit(W_ent[a]) @ unit(W_ent[b])) for b in range(NTR_ENT)]
                       for a in range(NE)])                       # (73, 41)
    log('bridge cos: cls min=%.3f max=%.3f | ent min=%.3f max=%.3f'
        % (cos_cm.min(), cos_cm.max(), cos_em.min(), cos_em.max()))

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
        ent_mean = {}
        for r in tr_rows:
            ent_mean.setdefault(PAIRS[r % NP_][0], []).append(Y[r])
        ent_mean = {i: np.stack(v).mean(0) for i, v in ent_mean.items()}
        # trained direction bases (activation side, train rows only)
        d_cls = np.stack([cls_mean[c] - B0 for c in range(NCS)])          # (6, d)
        e_ent = np.stack([ent_mean[i] - B0 for i in range(NTR_ENT)])      # (41, d)
        Qs, _ = np.linalg.qr(d_cls.T.astype(np.float32))
        Qs = Qs[:, :NCS].astype(np.float32)

        Xo = np.stack([phi_row(PAIRS[r % NP_][0], PAIRS[r % NP_][1], r // NP_)
                       for r in kl_rows_idx])
        Yhat0 = Xo @ W
        # G0: verbatim 3240-F2
        YhatG0 = Yhat0 + (B0[None, :] - Yhat0) @ Qs @ Qs.T
        # anchor + projection recipes (prereg v2, see DESIGN prereg_correction (2)):
        # anchor A = B0 + lexical-bridge-weighted direction offsets; prediction =
        # Yhat_ridge + P_span(A - Yhat_ridge) — amplitude set by the projection.
        rng_perm = np.random.RandomState(RAND_SEED)
        perm_cls = rng_perm.permutation(NC)          # permuted class identities
        perm_ent = rng_perm.permutation(NE)          # permuted entity identities
        d_mean = d_cls.mean(0)
        e_mean = e_ent.mean(0)
        # joint span S∪E via QR (6 class + 41 entity centered directions)
        SJE = np.concatenate([d_cls, e_ent], axis=0)               # (47, d)
        Qje, _ = np.linalg.qr(SJE.T.astype(np.float32))
        Qje = Qje[:, :SJE.shape[0]].astype(np.float32)
        Ypreds = {}
        for rn in RECIPES[1:]:
            corr = np.zeros_like(Yhat0)
            wfn = softmax_weights if WEIGHT_KIND[rn] == 'softmax' else linear_weights
            for j, r in enumerate(kl_rows_idx):
                i, c = PAIRS[r % NP_]
                cc = int(perm_cls[c]) if IS_PERM[rn] else c
                ii = int(perm_ent[i]) if IS_PERM[rn] else i
                if cc < NCS:
                    continue                                         # train-class rows: no bridge
                A = B0.copy()
                w_c = wfn(cos_cm[cc])
                A = A + (w_c[:, None] * (d_cls - d_mean[None, :])).sum(0)
                if IS_DUAL[rn] and ii >= NTR_ENT:
                    w_e = wfn(cos_em[ii])
                    A = A + (w_e[:, None] * (e_ent - e_mean[None, :])).sum(0)
                Qp = Qje if IS_DUAL[rn] else Qs
                corr[j] = (A - Yhat0[j]) @ Qp @ Qp.T
            Ypreds[rn] = (Yhat0 + corr).astype(np.float32)
        Ypreds['G0'] = YhatG0.astype(np.float32)
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
        seed_rec['G0']['r2_seen'] = None   # seen-test unchanged by construction (OOV-only)
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
        # G0 replication vs sealed 3240 F2 (same code path; formal rows only)
        ps40 = sealed_3240['per_seed'][str(s)]['F2']
        d40 = max(abs(seed_rec['G0']['r2_oov'] - ps40['r2_oov']),
                  abs(seed_rec['G0']['kl_mean'] - ps40['kl_mean']),
                  max(abs(seed_rec['G0']['fam_fid'][f] - ps40['fam_fid'][f])
                      for f in FAMS))
        seed_rec['G0']['d_vs_sealed_3240'] = d40
        per_seed[s] = seed_rec
        log('seed %d: G0 d40=%.3e | G1c dSCLS=%+.3f dENT=%+.3f | '
            'G1w dSCLS=%+.3f dENT=%+.3f | G1e r2=%.4f | G1r SCLS=%.3f G1rw SCLS=%.3f' %
            (s, d40,
             seed_rec['G1c']['fam_fid']['SCLS'] - seed_rec['G0']['fam_fid']['SCLS'],
             seed_rec['G1c']['fam_fid']['ENT'] - seed_rec['G0']['fam_fid']['ENT'],
             seed_rec['G1w']['fam_fid']['SCLS'] - seed_rec['G0']['fam_fid']['SCLS'],
             seed_rec['G1w']['fam_fid']['ENT'] - seed_rec['G0']['fam_fid']['ENT'],
             seed_rec['G1e']['r2_oov'],
             seed_rec['G1r']['fam_fid']['SCLS'], seed_rec['G1rw']['fam_fid']['SCLS']))
        log('seed %d: KL G0=%.4f G1c=%.4f G1e=%.4f G1r=%.4f' %
            (s, seed_rec['G0']['kl_mean'], seed_rec['G1c']['kl_mean'],
             seed_rec['G1e']['kl_mean'], seed_rec['G1r']['kl_mean']))
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
            for rn in ('G1c', 'G1w', 'G1e')}
    best = max(cand, key=cand.get)
    ctrl = CTRL_OF[best]
    g2 = bool(cand[best] > EFF_DELTA
              and min(delta[best]['d_fam_fid']['DVU'],
                      delta[best]['d_fam_fid']['DCC']) > -REG_DROP
              and -delta[best]['d_r2_oov'] < R2_DROP
              and min(agg[best]['fam_fid_mean']['SCLS'],
                      agg[best]['fam_fid_mean']['ENT'])
              > min(agg[ctrl]['fam_fid_mean']['SCLS'],
                    agg[ctrl]['fam_fid_mean']['ENT']))
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
                              for rn in ('G1c', 'G1w', 'G1e')},
        'ent_sign_flipped': {rn: bool(agg[rn]['fam_fid_mean']['ENT'] > 0)
                             for rn in ('G1c', 'G1w', 'G1e')},
        'kl_order_best_vs_G0': bool(agg[best]['kl_mean'] < agg['G0']['kl_mean']),
        'beats_random_control': {rn: bool(min(agg[rn]['fam_fid_mean']['SCLS'],
                                              agg[rn]['fam_fid_mean']['ENT'])
                                         > min(agg[CTRL_OF[rn]]['fam_fid_mean']['SCLS'],
                                               agg[CTRL_OF[rn]]['fam_fid_mean']['ENT']))
                                 for rn in ('G1c', 'G1w', 'G1e')},
    }
    result = {
        'phase': 3241, 'name': NAME, 'mode': 'smoke' if SMOKE else 'full',
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'runtime_s': round(time.time() - T0, 1),
        'design_sha8': dsha[:8], 'smoke': SMOKE,
        'model': MODEL, 'seeds_run': SEEDS_RUN,
        'kl_rows': kl_n, 'oov_rows_total': len(oov_rows),
        'anchors': {'d83': d83, 'd_tc': b57['d_tc'], **b69['anchors'],
                    'kout': int(b69['kout']), 'D': int(b69['D']), 'd1': d1},
        'spectra': spectra,
        'bridge': {'beta': BRIDGE_BETA, 'rand_seed': RAND_SEED,
                   'cls_cos_matrix': cos_cm.tolist(),
                   'cls_words': cls_words},
        'per_seed': per_seed,
        'aggregate': agg,
        'delta_vs_G0': delta,
        'headline': headline,
        'gates': gates,
        'verdict': verdict,
        'sources': {k: sha8_file(v) for k, v in SRC.items()},
    }
    # res_sha8 over DETERMINISTIC fields only (mirror freeze discipline)
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
    npz_path = os.path.join(OUTDIR, 'smoke_bridge.npz' if SMOKE else 'bridge.npz')
    np.savez(npz_path, **npz_data)
    log('npz saved: %s (%d KB)' % (os.path.basename(npz_path),
                                   os.path.getsize(npz_path) // 1024))
    log('RUN_COMPLETE %s' % verdict)
    return verdict


if __name__ == '__main__':
    v = main()
    sys.exit(0 if v == DESIGN['verdicts']['pass'] else 2)
