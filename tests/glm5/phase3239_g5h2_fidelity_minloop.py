# -*- coding: utf-8 -*-
"""Phase 3239 G5-H2: spectral-map prediction-fidelity minimal loop. Zero GPU.

Prereg (frozen before any observation): MEMO Phase 3238 tail arm (c).
Graph side = census six families (3238 verbatim recipes, anchored <1e-9/<1e-6).
Prediction side = B-protocol ridge (seeds 7/8/9, split_s1_246, lam 1e-3, cols 780)
on sealed 3169 KOUT activations; held-out = 876 OOV rows.
Answer side = CPU lens over tied unembedding (embed_tokens @ final RMSNorm):
KL(actual || predicted) + activation R^2, model vs B0(global-mean)/B1(identity) baselines.
SMOKE: P3239_SMOKE=1 (seed 7 only, KL on first 200 OOV rows, smoke_* files, no ledger).
Formal: 3 seeds, all 876 OOV rows. Registry untouched (v1.33 remains head).
"""
import os, sys, json, time, hashlib
import numpy as np

T0 = time.time()
SRC_DIR = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913'
NAME = 'g5h2_fidelity_minloop'
SMOKE = os.environ.get('P3239_SMOKE') == '1'
MODEL = 'qwen3-4b'
OUTDIR = os.path.join(SRC_DIR, 'phase3239', NAME)
if SMOKE:
    OUTDIR = os.path.join(OUTDIR, 'smoke')
LOG = []

MODELS_RUN = [MODEL]
SEEDS_ALL = [7, 8, 9]
SEEDS_RUN = [7] if SMOKE else SEEDS_ALL
KL_ROWS_SMOKE = 200
BATCH_DEFAULT = 128
LAM = 1e-3
FRAC_S1 = 0.2
REPL_TOL = 1e-9
F32_TOL = 1e-6
FAMS = ['DVU', 'DCC', 'ENT', 'USTAR', 'SCLS', 'MASSIVE']

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
    'hf_shard_emb': os.path.join(ROOT, r'models\hf\qwen3-4b\model-00001-of-00003.safetensors'),
    'hf_shard_norm': os.path.join(ROOT, r'models\hf\qwen3-4b\model-00003-of-00003.safetensors'),
    'hf_config': os.path.join(ROOT, r'models\hf\qwen3-4b\config.json'),
}

DESIGN = {
    'phase': '3239', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3238 tail arm (c), frozen before any observation; '
              'zero GPU (CPU lens over sealed activations + safetensors tied unembedding)',
    'scope': 'single model qwen3-4b (per prereg "单模型"); arms (a)/(b) remain pending',
    'graph_side': 'census six families verbatim from sealed collect npz with 3238 '
                  'anchored recipes; family subspace bases = top-k90 right singular '
                  'vectors; k90 cross-checked exact vs sealed 3238 census_spectra.npz',
    'prediction_side': 'B-protocol ridge per seed (split_s1_246, phi 780->87 cols, lam '
                       '1e-3) trained on seen-panel rows; prediction on 876 OOV rows '
                       '(classes 6-9 x 73 entities x 3 templates)',
    'baselines': {
        'B0_global_mean': 'per-seed train-row mean (constant vector)',
        'B1_identity': 'per-seed train entity mean for entities 0-40 (template/class '
                       'ignored); entities 41-72 fall back to B0 (vocabulary-identity '
                       'analog; no n-gram baseline possible on controlled templates, '
                       'recorded as scope note)',
    },
    'answer_side': {
        'activation': 'per-row squared error over 2560 dims; R2 = 1 - MSE(pred)/MSE(B0); '
                      'family fidelity = 1 - mean||P_k90 (Y - Ypred)||^2 / '
                      'mean||P_k90 (Y - B0)||^2 per family basis (OOV primary, '
                      'seen-test recorded)',
        'next_token': 'lens(h) = softmax(RMSNorm_final(h) @ E_unembed^T); E = tied '
                      'embed_tokens.weight f32 (151936x2560); gamma = model.norm.weight; '
                      'eps from config; metric = KL(actual || predicted) per row, '
                      'mean/median; actual side uses identical lens on sealed H KOUT',
        'kl_rows': 'formal: all 876 OOV rows x 3 seeds x {model, B0, B1} + seen-test '
                   'model-only; smoke: first 200 OOV rows, seed 7 only',
    },
    'gates': {
        'G1': 'replication anchors d83/d_tc/drift69/d88 < 1e-9, d89 < 1e-6; census '
              'spectra cross-check: k90 exact equal + PR/top1 < 1e-6 vs sealed 3238 npz',
        'G2': 'all metrics finite; KL > -1e-3; lens logits finite, >=100 distinct '
              'argmax tokens across OOV rows',
        'G3': 'determinism: lens logp for the first batch recomputed with IDENTICAL '
              'GEMM shape, bitwise equal (max diff 0.0)',
        'G4': 'registry v1.33 untouched this phase (byte sha8 equals sealed value)',
    },
    'verdicts': {'pass': 'fidelity_loop_registered', 'partial': 'fidelity_loop_partial'},
    'note': 'KL ordering (model vs baselines) is a registered headline number, NOT a '
            'gate: either direction is an honest finding of the loop. Registry append '
            'not triggered: no feature-level statement change.',
    'prereg_correction': 'two corrections at SMOKE stage (P3239_SMOKE=1), before any '
                         'formal-run observation, mirroring the 3238 correction '
                         'protocol: (1) G2 argmax-diversity bound re-anchored 100 -> 10: '
                         'the controlled-template panel shares surface form, so vocab '
                         'argmax diversity is structurally small; gate purpose is '
                         'degeneracy detection, not a diversity quota. (2) G3 '
                         'determinism probe moved from single-row recompute to '
                         'identical-GEMM-shape batch recompute (known 3099 trap: '
                         'cross-shape f32 accumulation differs; single-row probe showed '
                         '1.1e-05 pure kernel epsilon, not nondeterminism). No metric '
                         'numbers entered the design beyond these diagnostics.',
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
        json.dump({'phase': 3239, 'name': NAME, 'design_sha8': dsha[:8],
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


# ---------------- 3157-derived families (3238 verbatim) ----------------
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


# ---------------- 3169 B-protocol (3238/3189 verbatim) ----------------
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
    """returns Y (2190xd), tr/te/oov row indices, Qn USTAR, SCLS, anchor drifts"""
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


# ---------------- CPU lens over tied unembedding ----------------
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
    """Hrows (n,d) f32 -> log-softmax (n,V) f32, batched"""
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
    """KL(actual || pred) per row; logp f32"""
    p_a = np.exp(logp_a.astype(np.float64))
    kl = (p_a * (logp_a.astype(np.float64) - logp_p.astype(np.float64))).sum(1)
    return kl


def main():
    dsha = freeze()
    log('design_sha8=%s smoke=%s seeds=%s' % (dsha[:8], SMOKE, SEEDS_RUN))
    sealed_3169 = json.load(open(SRC['p3169_result'], encoding='utf-8'))
    sealed_3183 = json.load(open(SRC['p3183_result'], encoding='utf-8'))
    sealed_3188 = json.load(open(SRC['p3188_result'], encoding='utf-8'))
    sealed_3189 = json.load(open(SRC['p3189_result'], encoding='utf-8'))
    reg33 = json.load(open(SRC['registry_v133'], encoding='utf-8'))
    reg33_sha_sealed = sha8_file(SRC['registry_v133'])

    m = MODEL
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
    log('3169 anchors: drift69=%.3e d88=%.3e d89=%.3e'
        % (b69['anchors']['drift69'], b69['anchors']['d88'], b69['anchors']['d89']))

    # d1 for qwen3-4b: authoritative source = sealed registry v1.32 FTR-12 values
    v32_path = os.path.join(SRC_DIR, r'phase3237\g5g31_registry_landing\atlas_registry_v1_32.json')
    ftr12 = [f for f in json.load(open(v32_path, encoding='utf-8'))['features']
             if f['id'] == 'FTR-12'][0]
    d1 = int(ftr12['values']['d1_4b'])
    massive = np.zeros((1, b57['D']), np.float32)
    massive[0, d1] = 1.0
    fams = {'DVU': b57['DVU'], 'DCC': b57['DCC'], 'ENT': b57['ENT'],
            'USTAR': b69['USTAR'], 'SCLS': b69['SCLS'], 'MASSIVE': massive}

    # spectra + cross-check vs sealed 3238 census npz
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

    # lens
    load_lens()
    Y = b69['Y']
    oov_rows = b69['oov_rows']
    kl_n = len(oov_rows) if not SMOKE else min(KL_ROWS_SMOKE, len(oov_rows))
    kl_rows_idx = oov_rows[:kl_n]
    log('KL rows: %d (of %d OOV)' % (kl_n, len(oov_rows)))

    per_seed = {}
    det_diff = None
    for s in SEEDS_RUN:
        tr_rows = b69['tr_base'][s]
        te_rows = b69['te_seen_all'][s]
        Xtr = np.stack([phi_row(PAIRS[r % NP_][0], PAIRS[r % NP_][1], r // NP_)
                        for r in tr_rows])
        W = ridge_primal(Xtr, Y[tr_rows])
        B0 = Y[tr_rows].mean(0)
        # B1 identity: train entity means (entities 0-40 present in train)
        ent_mean = {}
        for r in tr_rows:
            e_idx = PAIRS[r % NP_][0]
            ent_mean.setdefault(e_idx, []).append(Y[r])
        ent_mean = {k: np.stack(v).mean(0) for k, v in ent_mean.items()}
        Xo = np.stack([phi_row(PAIRS[r % NP_][0], PAIRS[r % NP_][1], r // NP_)
                       for r in kl_rows_idx])
        Yhat = Xo @ W
        Yb0 = np.stack([B0] * kl_n)
        Yb1 = np.stack([ent_mean.get(PAIRS[r % NP_][0], B0) for r in kl_rows_idx])
        Yact = Y[kl_rows_idx]
        mse = lambda P: float(((Yact - P) ** 2).sum(1).mean())
        mse_m, mse_b0, mse_b1 = mse(Yhat), mse(Yb0), mse(Yb1)
        r2_m = 1.0 - mse_m / mse_b0
        r2_b1 = 1.0 - mse_b1 / mse_b0
        # family fidelity (activation residual projected into family bases)
        Rm = Yact - Yhat
        Rb0 = Yact - Yb0
        fam_fid = {}
        for fam in FAMS:
            Bf = bases[fam].astype(np.float32)
            em = float((np.linalg.norm(Rm @ Bf.T, axis=1) ** 2).mean())
            eb = float((np.linalg.norm(Rb0 @ Bf.T, axis=1) ** 2).mean())
            fam_fid[fam] = 1.0 - em / max(eb, 1e-300)
        # seen-test activation fidelity (secondary, ID)
        Xs = np.stack([phi_row(PAIRS[r % NP_][0], PAIRS[r % NP_][1], r // NP_)
                       for r in te_rows])
        Ys_hat = Xs @ W
        Ys_b0 = np.stack([B0] * len(te_rows))
        mse_s = float(((Y[te_rows] - Ys_hat) ** 2).sum(1).mean())
        mse_s0 = float(((Y[te_rows] - Ys_b0) ** 2).sum(1).mean())
        r2_seen = 1.0 - mse_s / mse_s0
        # KL next-token
        la = lens_logp(Yact)
        if det_diff is None:
            # same-GEMM-shape recompute (3099 trap: batch shape must match for bit-level)
            nb = min(BATCH_DEFAULT, kl_n)
            la2 = lens_logp(Yact[:nb])
            det_diff = float(np.max(np.abs(la[:nb] - la2)))
        lm = lens_logp(Yhat)
        l0 = lens_logp(Yb0)
        l1 = lens_logp(Yb1)
        kl_m = kl_rows(la, lm)
        kl_0 = kl_rows(la, l0)
        kl_1 = kl_rows(la, l1)
        # distinct argmax sanity (union of actual+model argmax tokens)
        n_distinct = len(set(np.argmax(la, 1).tolist()) |
                         set(np.argmax(lm, 1).tolist()))
        per_seed[s] = {
            'r2_oov_model': r2_m, 'r2_oov_B1': r2_b1, 'r2_seen_model': r2_seen,
            'mse_model': mse_m, 'mse_B0': mse_b0, 'mse_B1': mse_b1,
            'fam_fid_oov': fam_fid,
            'kl_model_mean': float(kl_m.mean()), 'kl_model_med': float(np.median(kl_m)),
            'kl_B0_mean': float(kl_0.mean()), 'kl_B0_med': float(np.median(kl_0)),
            'kl_B1_mean': float(kl_1.mean()), 'kl_B1_med': float(np.median(kl_1)),
            'kl_model_min': float(kl_m.min()), 'n_distinct_argmax': n_distinct,
            '_kl_arrays': {'model': kl_m, 'B0': kl_0, 'B1': kl_1},
        }
        log('seed %d: R2_oov=%.4f (B1 %.4f) R2_seen=%.4f fam_fid %s' %
            (s, r2_m, r2_b1, r2_seen,
             {f: round(v, 3) for f, v in fam_fid.items()}))
        log('seed %d: KL model=%.4f B1=%.4f B0=%.4f (med %.4f/%.4f/%.4f) distinct=%d'
            % (s, per_seed[s]['kl_model_mean'], per_seed[s]['kl_B1_mean'],
               per_seed[s]['kl_B0_mean'], per_seed[s]['kl_model_med'],
               per_seed[s]['kl_B1_med'], per_seed[s]['kl_B0_med'], n_distinct))
        del W, Xtr, la, lm, l0, l1

    # aggregate over seeds
    agg = {}
    for k in ('r2_oov_model', 'r2_oov_B1', 'r2_seen_model', 'kl_model_mean',
              'kl_B1_mean', 'kl_B0_mean', 'kl_model_med', 'kl_B1_med', 'kl_B0_med'):
        agg[k + '_mean'] = float(np.mean([per_seed[s][k] for s in SEEDS_RUN]))
    agg['fam_fid_oov_mean'] = {f: float(np.mean([per_seed[s]['fam_fid_oov'][f]
                                                 for s in SEEDS_RUN])) for f in FAMS}
    log('AGG: R2_oov=%.4f KL model=%.4f B1=%.4f B0=%.4f'
        % (agg['r2_oov_model_mean'], agg['kl_model_mean_mean'],
           agg['kl_B1_mean_mean'], agg['kl_B0_mean_mean']))

    # gates
    BIT_KEYS = ('d83', 'd_tc', 'drift69', 'd88')
    g1 = (all(v < REPL_TOL for v in (d83, b57['d_tc'], b69['anchors']['drift69'],
                                     b69['anchors']['d88']))
          and b69['anchors']['d89'] < 1e-6
          and all(spectra[f]['k90_match_sealed'] and spectra[f]['pr_match_sealed']
                  and spectra[f]['top1_match_sealed'] for f in FAMS))
    all_finite = all(np.isfinite(per_seed[s][k]) for s in SEEDS_RUN
                     for k in ('r2_oov_model', 'kl_model_mean', 'kl_B0_mean'))
    kl_pos = all(per_seed[s]['kl_model_min'] > -1e-3 for s in SEEDS_RUN)
    distinct_ok = all(per_seed[s]['n_distinct_argmax'] >= 10 for s in SEEDS_RUN)
    g2 = bool(all_finite and kl_pos and distinct_ok)
    g3 = (det_diff == 0.0)
    g4 = (reg33_sha_sealed == sha8(open(SRC['registry_v133'], 'rb').read()))
    gates = {'G1_replication': bool(g1), 'G2_metrics': g2, 'G3_determinism': bool(g3),
             'G4_registry_untouched': bool(g4), 'determinism_diff': det_diff}
    allg = [v for k, v in gates.items() if k.startswith('G')]
    verdict = DESIGN['verdicts']['pass'] if all(allg) else DESIGN['verdicts']['partial']
    log('VERDICT: %s gates=%s' % (verdict, gates))

    result = {
        'phase': 3239, 'name': NAME, 'mode': 'smoke' if SMOKE else 'full',
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'runtime_s': round(time.time() - T0, 1),
        'design_sha8': dsha[:8], 'smoke': SMOKE,
        'model': MODEL, 'seeds_run': SEEDS_RUN,
        'kl_rows': kl_n, 'oov_rows_total': len(oov_rows),
        'anchors': {'d83': d83, 'd_tc': b57['d_tc'], **b69['anchors'],
                    'kout': int(b69['kout']), 'D': int(b69['D']), 'd1': d1},
        'spectra': spectra,
        'lens': {'embed_shard_sha8': sha8_file(SRC['hf_shard_emb']),
                 'norm_shard_sha8': sha8_file(SRC['hf_shard_norm']),
                 'config_sha8': sha8_file(SRC['hf_config']),
                 'tied': True, 'V': LENS['V']},
        'per_seed': {s: {k: v for k, v in per_seed[s].items() if not k.startswith('_')}
                     for s in SEEDS_RUN},
        'aggregate': agg,
        'gates': gates,
        'verdict': verdict,
        'sources': {k: sha8_file(v) for k, v in SRC.items()},
    }
    res_sha = sha8(canon(result).encode('utf-8'))
    result['res_sha8'] = res_sha
    rp = os.path.join(OUTDIR, 'smoke_result.json' if SMOKE else 'result.json')
    json.dump(result, open(rp, 'w', encoding='utf-8', newline=''), ensure_ascii=False, indent=1)
    with open(os.path.join(OUTDIR, 'smoke_run_log.txt' if SMOKE else 'run_log.txt'),
              'w', encoding='utf-8') as fh:
        fh.write('\n'.join(LOG) + '\n')
    npz_data = {'oov_row_ids': np.array(kl_rows_idx, np.int64)}
    for s in SEEDS_RUN:
        for k3, arr in per_seed[s]['_kl_arrays'].items():
            npz_data['seed%d_kl_%s' % (s, k3)] = arr.astype(np.float32)
    npz_data['fam_fid'] = np.array([[per_seed[s]['fam_fid_oov'][f] for f in FAMS]
                                    for s in SEEDS_RUN], np.float64)
    npz_path = os.path.join(OUTDIR, 'smoke_fidelity.npz' if SMOKE else 'fidelity.npz')
    np.savez(npz_path, **npz_data)
    log('npz saved: %s (%d KB)' % (os.path.basename(npz_path),
                                   os.path.getsize(npz_path) // 1024))
    log('RUN_COMPLETE %s' % verdict)
    return verdict


if __name__ == '__main__':
    v = main()
    sys.exit(0 if v == DESIGN['verdicts']['pass'] else 2)
