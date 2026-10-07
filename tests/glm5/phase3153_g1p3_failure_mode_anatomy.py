# -*- coding: utf-8 -*-
# Phase 3153 (G1-P3): 失败模态解剖 —— worst-20 跨模型分类 + 读出层残差谱解剖 + M1 因子对齐
# 预注册：AGI_GPT5_MEMO L14878（观测前冻结）
# 全零 GPU：4b/14b 读 3152 collect.npz；glm4 读 3151 collect.npz；SMOKE 读 3152 smoke npz
# 运行模式（P3153_MODEL）: qwen3-4b | qwen3-14b | glm4 | summary
# 四任务:
#   (1) worst-20 跨模型并/交 Jaccard + 失败模态分类表
#       分类树(互斥, 冻结): M1 错配对(c != CLS_OF) > M2 难类真对(c == CLS_OF 且 c in top2 难类)
#       > M3 高秩散布真对(其余且 E_g < 0.5) > M4 交互特异真对(其余且 E_g >= 0.5)
#       E_g = (i,c) 网格均值能量(外积空间投影, per-row)
#       coverage := (M1+M2+M4)/20 结构可归类比例, 门 >= 0.8
#   (2) 读出层残差谱解剖: S1_s7 test B4@KOUT 残差 R
#       spec_cell = 网格均值矩阵 G 的归一化奇异值谱(top10); spec_row = R 的谱(top10)
#       ANOVA 份额指纹 = [类间, 加性实体增量, 交互增量, 模板内] (嵌套增量, 和=1)
#       对照: k* 层同分解(预期低秩尖锐)
#   (3) M1 rank-5 因子(k*, seed7, ALS 100it) 列空间 vs Pc(k*) / Pc(KOUT) 主角谱
#   (4) 验收: 谱指纹两两 Pearson >= 0.8(3 对) + coverage 门 + worst 交叉验证
#       死线: 谱指纹任一对 < 0.8 -> g1_readout_class_subspace_narrative_dropped
# 锚断言: worst20 rel_err vs 3152 a3_worst20 |d|<1e-6; V5 e_ks/e_ko |d|<1e-4;
#         glm4 B4@k39 vs 0.389835258324941 |d|<1e-4; M1@k* s7 margin vs 3152 |d|<1e-6;
#         M1 grid k*_r5_l0.01 vs 3152 |d|<1e-4; S2@KOUT vs 3152 v3 per_fold(4b/14b) |d|<1e-6
import os, sys, json, time, hashlib
import numpy as np

T0 = time.time()
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
MODEL = os.environ.get('P3153_MODEL') or (sys.argv[1] if len(sys.argv) > 1 else 'qwen3-4b')
SMOKE = os.environ.get('P3153_SMOKE') == '1'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
NAME = 'g1p3_failure_mode_anatomy'
BASE = os.path.join(RDIR, 'phase3153', NAME, MODEL)
if SMOKE:
    BASE = os.path.join(BASE, 'smoke')
os.makedirs(BASE, exist_ok=True)
LOGP = os.path.join(BASE, 'run_log.txt')

def log(s):
    ln = '[%7.1f] %s' % (time.time() - T0, s)
    with open(LOGP, 'a', encoding='utf-8') as f:
        f.write(ln + '\n')
    try:
        print(ln, flush=True)
    except Exception:
        pass

def freeze_design(phase_name, design):
    eblob = json.dumps(design, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
    sha = hashlib.sha256(eblob).hexdigest()
    exe_p = os.path.join(BASE, 'execution.json')
    if os.path.exists(exe_p):
        prev = json.load(open(exe_p, encoding='utf-8'))
        assert prev['design_sha'] == sha, 'DESIGN DRIFT'
        log('execution.json match (sha %s)' % sha[:8])
    else:
        json.dump({'phase': 3153, 'name': phase_name, 'design_sha': sha,
                   'design': design, 'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('execution.json FROZEN (sha %s)' % sha[:8])
    return sha

def seal_result(result, out_name):
    blob = json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
    res_sha8 = hashlib.sha256(blob).hexdigest()[:8]
    result['res_sha8'] = res_sha8
    result['verdict'] = result['verdict'] + '|sha8_' + res_sha8
    rp = os.path.join(BASE, out_name)
    json.dump(result, open(rp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    seal = hashlib.sha256(open(rp, 'rb').read()).hexdigest()[:8]
    result['seal_sha8'] = seal
    json.dump(result, open(rp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    log('RESULT %s res_sha8=%s seal=%s verdict=%s' % (out_name, res_sha8, seal, result['verdict']))
    return res_sha8, seal

# ---------------- 冻结面板材料（与 3152 逐字一致） ----------------
CLASSES = ['水果', '动物', '交通工具', '家具', '金属', '颜色']
ENT = {
    '水果': ['苹果', '香蕉', '梨', '西瓜', '葡萄', '草莓', '芒果', '柠檬'],
    '动物': ['狗', '猫', '老虎', '大象', '兔子', '猴子', '马', '牛'],
    '交通工具': ['汽车', '火车', '飞机', '摩托车', '卡车', '地铁'],
    '家具': ['桌子', '椅子', '床', '沙发', '地毯', '窗帘'],
    '金属': ['铁', '铜', '铝', '金', '银', '锌', '铅'],
    '颜色': ['红', '蓝', '绿', '黄', '黑', '白'],
}
TPL = {0: '{e}是一种{c}。', 1: '{e}属于{c}这一类。', 2: '{e}，一种常见的{c}。'}
SEEDS_S1 = [7, 8, 9]
FRAC_S1 = 0.2
RANK_M1 = 5
ALS_SEED = 7
ALS_RIDGE = 1e-2
ALS_ITERS = 5 if SMOKE else 100
ADD_ERR_GATE = 0.05
KSTAR_FRAC = 0.075
GRID_FRACS = (0.375, 0.575, 0.775, 0.975)
M1_GRID_ITERS = 3 if SMOKE else 40
WORST_N = 20
PANEL_MIN = 672
EG_GATE = 0.5          # M3/M4 网格能量阈值
COV_GATE = 0.8         # 覆盖率门
FP_GATE = 0.8          # 谱指纹相关门

ENTS = [e for cl in CLASSES for e in ENT[cl]]
CLS_OF = [CLASSES.index(cl) for cl in CLASSES for e in ENT[cl]]
NE = len(ENTS)
NC = len(CLASSES)
PAIRS = [(i, c) for i in range(NE) for c in range(NC)]
NP_ = len(PAIRS)
NT = len(TPL)
PANEL_FULL = NT * NP_

if SMOKE:
    keep_e = []
    cnt = {}
    for i, cl in enumerate(CLS_OF):
        if cnt.get(cl, 0) < 2:
            keep_e.append(i)
            cnt[cl] = cnt.get(cl, 0) + 1
    PAIRS = [(i, c) for (i, c) in PAIRS if i in set(keep_e)]
    NP_ = len(PAIRS)
    NE_KEEP = len(keep_e)
else:
    keep_e = list(range(NE))
    NE_KEEP = NE
PANEL = NT * NP_
ALLP = set(PAIRS)

def split_s1(seed):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(NP_)
    n_test = int(round(FRAC_S1 * NP_))
    test = set([PAIRS[j] for j in idx[:n_test]])
    return ALLP - test, test

def rows_of(pair_set):
    return [t * NP_ + pi for t in range(NT) for pi, p in enumerate(PAIRS) if p in pair_set]

def ridge_primal(Xtr, Ytr, lam=1e-3):
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1], dtype=np.float32)
    W = np.linalg.solve(A, Xtr.T @ Ytr)
    return W

def als_complete(R, mask, rank, iters, ridge, seed):
    E, C, Dm = R.shape
    rng = np.random.RandomState(seed)
    mu = (R * mask[..., None]).sum((0, 1)) / max(1, mask.sum())
    R0 = (R - mu) * mask[..., None]
    V = (rng.randn(C, rank, Dm) * 0.1).astype(np.float32)
    U = np.zeros((E, rank, Dm), np.float32)
    rid = np.arange(rank)
    for it in range(iters):
        A = np.einsum('ec,crd,csd->ersd', mask, V, V)
        b = np.einsum('ec,ecd,crd->erd', mask, R0, V)
        A = A.transpose(3, 0, 1, 2)
        A[..., rid, rid] += ridge
        U = np.linalg.solve(A, b.transpose(2, 0, 1)[..., None])[..., 0].transpose(1, 2, 0)
        if not np.isfinite(U).all():
            return None
        A = np.einsum('ec,erd,esd->crsd', mask, U, U)
        b = np.einsum('ec,ecd,erd->crd', mask, R0, U)
        A = A.transpose(3, 0, 1, 2)
        A[..., rid, rid] += ridge
        V = np.linalg.solve(A, b.transpose(2, 0, 1)[..., None])[..., 0].transpose(1, 2, 0)
        if not np.isfinite(V).all():
            return None
    Rhat = np.einsum('erd,crd->ecd', U, V) + mu
    return Rhat.astype(np.float32)

def phi_main(train_set):
    cols = NE_KEEP + NC + NT + 1
    tr_rows = rows_of(train_set)
    def rowvec(t, pi):
        i, c = PAIRS[pi]
        v = np.zeros(cols, np.float32)
        v[keep_e.index(i)] = 1.0
        v[NE_KEEP + c] = 1.0
        v[NE_KEEP + NC + t] = 1.0
        v[-1] = 1.0
        return v
    Xtr = np.stack([rowvec(r // NP_, r % NP_) for r in tr_rows])
    return Xtr, tr_rows, rowvec, cols

def b4_fit(Y, seed, k):
    train_set, test_set = split_s1(seed)
    tr_rows = rows_of(train_set)
    te_rows = rows_of(test_set)
    Xtr, _, rowvec, cols = phi_main(train_set)
    Ytr = Y[tr_rows]
    ref = Ytr.mean(0)
    Dk = float(((Ytr - ref) ** 2).sum(1).mean()) + 1e-9
    W = ridge_primal(Xtr, Ytr, lam=1e-3)
    Xte = np.stack([rowvec(r // NP_, r % NP_) for r in te_rows])
    B4te = Xte @ W
    B4tr = Xtr @ W
    return dict(train_set=train_set, test_set=test_set, tr_rows=tr_rows,
                te_rows=te_rows, B4te=B4te, B4tr=B4tr, Dk=Dk)

# =================================================================
# 模式 A/B/C：qwen3-4b / qwen3-14b / glm4（全零 GPU，读缓存 npz）
# =================================================================
if MODEL in ('qwen3-4b', 'qwen3-14b', 'glm4'):
    if MODEL in ('qwen3-4b', 'qwen3-14b'):
        SRC = os.path.join(RDIR, 'phase3152', 'g1p2_tri_model_k1', MODEL, 'collect.npz')
    else:
        SRC = os.path.join(RDIR, 'phase3151', 'g1p1_combo_additive_vs_interaction', 'collect.npz')
    if SMOKE:
        SRC = os.path.join(RDIR, 'phase3152', 'g1p2_tri_model_k1', 'qwen3-4b', 'smoke', 'collect_smoke.npz')
    src_sha = hashlib.sha256(open(SRC, 'rb').read()).hexdigest()[:8]
    z = np.load(SRC)
    H16 = z['H']
    MARG = z['marg']
    NL = H16.shape[2] - 1
    D = int(H16.shape[3])
    KSTAR = int(round(KSTAR_FRAC * NL))
    KOUT = NL - 1
    assert H16.shape[:3] == (NT, NP_, NL + 1), H16.shape
    log('src=%s sha8=%s H=%s NL=%d D=%d k*=%d readout=%d' %
        (os.path.basename(SRC), src_sha, H16.shape, NL, D, KSTAR, KOUT))
    assert PANEL >= PANEL_MIN or SMOKE, ('panel too small', PANEL)

    G3152 = os.path.join(RDIR, 'phase3152', 'g1p2_tri_model_k1')
    if MODEL == 'qwen3-4b':
        R3152 = os.path.join(G3152, 'qwen3-4b', 'smoke' if SMOKE else '', 'result.json')
    elif MODEL == 'qwen3-14b':
        R3152 = os.path.join(G3152, 'qwen3-14b', 'result.json')
    else:
        R3152 = os.path.join(G3152, 'glm4k1', 'result.json')
    if SMOKE:
        R3152 = os.path.join(G3152, 'qwen3-4b', 'smoke', 'result.json')
    R3152 = os.path.normpath(R3152)
    r3152 = json.load(open(R3152, encoding='utf-8'))
    log('3152 ref result loaded: %s (sha8=%s)' % (os.path.basename(R3152), r3152.get('res_sha8')))

    def Y_at(k):
        return H16[:, :, k, :].reshape(NT * NP_, D).astype(np.float32)

    design = dict(model=MODEL, src=SRC, src_sha8=src_sha, nl=NL, hidden=D,
                  kstar=KSTAR, readout=KOUT, worst_n=WORST_N, eg_gate=EG_GATE,
                  cov_gate=COV_GATE, fp_gate=FP_GATE, als_iters=ALS_ITERS,
                  als_ridge=ALS_RIDGE, m1_grid_iters=M1_GRID_ITERS,
                  anova='nested increments [class, ent|cls, inter, within]',
                  spec='normalized sigma/sigma1 top10 of cell-mean grid G and row residual R',
                  classify_tree='M1 mismatch > M2 hard-class > M3 scatter(E_g<0.5) > M4 grid-specific(E_g>=0.5)',
                  ref_3152=R3152, smoke=SMOKE,
                  pre_reg='MEMO L14878; all-zero-GPU recompute from 3151/3152 collect.npz')
    exe_sha = freeze_design('g1p3_%s' % MODEL, design)

    # ---- (0) 复算 B4@k*/KOUT（S1 seed7）并断言锚 ----
    Yk = Y_at(KSTAR)
    Yo = Y_at(KOUT)
    f7k = b4_fit(Yk, SEEDS_S1[0], KSTAR)
    f7o = b4_fit(Yo, SEEDS_S1[0], KOUT)
    e7o = ((f7o['B4te'] - Yo[f7o['te_rows']]) ** 2).sum(1) / f7o['Dk']
    # worst20 交叉验证 vs 3152
    w20 = r3152['a3_worst20']
    assert len(w20) == WORST_N
    order = np.argsort(-e7o)[:WORST_N]
    drift_w = max(abs(float(e7o[int(a)]) - w20[j]['rel_err']) for j, a in enumerate(order))
    assert drift_w < 1e-6, ('worst20 rel_err drift vs 3152', drift_w)
    prompts = []
    for t in range(NT):
        for (i, c) in PAIRS:
            prompts.append(TPL[t].format(e=ENTS[i], c=CLASSES[c]))
    for j, a in enumerate(order):
        assert prompts[f7o['te_rows'][int(a)]] == w20[j]['prompt'], ('worst prompt mismatch', j)
    log('worst20 cross-check OK drift=%.2e' % drift_w)

    # ---- S2 留一类 B4@KOUT（自算 + 4b/14b 断言 3152 v3） ----
    s2_b4 = {}
    for cstar in range(NC):
        te = set([p for p in PAIRS if p[1] == cstar])
        train_set = ALLP - te
        tr_rows = rows_of(train_set)
        te_rows = rows_of(te)
        Xtr, _, rowvec, cols = phi_main(train_set)
        Y = Yo
        Ytr = Y[tr_rows]
        ref = Ytr.mean(0)
        Dk = float(((Ytr - ref) ** 2).sum(1).mean()) + 1e-9
        W = ridge_primal(Xtr, Ytr, lam=1e-3)
        Xte = np.stack([rowvec(r // NP_, r % NP_) for r in te_rows])
        P = Xte @ W
        Yte = Y[te_rows]
        s2_b4[cstar] = float((((P - Yte) ** 2).sum(1) / Dk).mean())
    v3152 = r3152.get('v3', {}).get('per_fold', {})
    if v3152:
        for cstar in range(NC):
            key = 'c%d_k%d' % (cstar, KOUT)
            if key in v3152:
                dv = abs(s2_b4[cstar] - v3152[key]['b4'])
                assert dv < 1e-6, ('S2 b4 drift vs 3152 v3', cstar, dv)
        log('S2@KOUT cross-check vs 3152 v3 OK')
    rank_cls = sorted(range(NC), key=lambda c: -s2_b4[c])
    hard2 = set(rank_cls[:2])
    log('hard classes(top2)=%s s2_b4=%s' %
        ([CLASSES[c] for c in rank_cls[:2]], ['%.3f' % s2_b4[c] for c in rank_cls]))

    # ---- V5 重算（锚断言 e_ks/e_ko + 存 Pc） ----
    def v5_energy(k):
        tr_rows = rows_of(split_s1(SEEDS_S1[0])[0])
        Y = Y_at(k)
        Xtr, _, rowvec, cols = phi_main(split_s1(SEEDS_S1[0])[0])
        W = ridge_primal(Xtr, Y[tr_rows], lam=1e-3)
        Rg = np.zeros((NE_KEEP, NC, D), np.float32)
        for t in range(NT):
            grid = H16[t, :, k, :].astype(np.float32)
            for pi in range(NP_):
                pv = rowvec(t, pi) @ W
                i, c = PAIRS[pi]
                Rg[keep_e.index(i), c] = grid[pi] - pv
        Rf = Rg.reshape(-1, D)
        cm = np.stack([Rg[[keep_e.index(i) for i in keep_e if CLS_OF[i] == c], c].mean(0)
                       for c in range(NC)])
        cmc = cm - cm.mean(0)
        U, S, Vt = np.linalg.svd(cmc, full_matrices=False)
        Pc = Vt[:5].T
        proj = Rf @ Pc
        e_class = float((proj ** 2).sum() / ((Rf ** 2).sum() + 1e-9))
        return e_class, float(S[:5].sum() / (S.sum() + 1e-9)), Pc
    e_ks, sv_ks, Pc_ks = v5_energy(KSTAR)
    e_ko, sv_ko, Pc_ko = v5_energy(KOUT)
    v5_3152 = r3152.get('v5', {})
    if v5_3152:
        d1 = abs(e_ks - v5_3152['class_subspace_energy_kstar'])
        d2 = abs(e_ko - v5_3152['class_subspace_energy_readout'])
        assert d1 < 1e-4 and d2 < 1e-4, ('V5 drift', d1, d2)
        log('V5 cross-check OK (e_ks=%.4f e_ko=%.4f)' % (e_ks, e_ko))
    else:
        log('V5 recomputed (no 3152 v5 ref): e_ks=%.4f e_ko=%.4f' % (e_ks, e_ko))

    # ---- (2) 读出层残差谱解剖 + k* 对照 ----
    def anatomy(k):
        Y = Y_at(k)
        f = b4_fit(Y, SEEDS_S1[0], k)
        e = ((f['B4te'] - Y[f['te_rows']]) ** 2).sum(1) / f['Dk']
        R = (Y[f['te_rows']] - f['B4te']).astype(np.float64)
        R = R - R.mean(0)
        SS = float((R ** 2).sum()) + 1e-18
        # 行谱
        Ur, Sr, Vtr = np.linalg.svd(R, full_matrices=False)
        spec_row = (Sr[:10] / (Sr[0] + 1e-18)).tolist()
        # 网格均值矩阵 G（(i,c) cell 均值）
        cell_rows = {}
        for j, r in enumerate(f['te_rows']):
            key = PAIRS[r % NP_]
            cell_rows.setdefault(key, []).append(j)
        cells = sorted(cell_rows.keys())
        G = np.stack([R[cell_rows[cl]].mean(0) for cl in cells])
        Uc, Sc, Vtc = np.linalg.svd(G, full_matrices=False)
        spec_cell = (Sc[:10] / (Sc[0] + 1e-18)).tolist()
        # ANOVA 嵌套增量份额
        mu = R.mean(0)
        m_c = {}
        for c in range(NC):
            idx = [j for j, r in enumerate(f['te_rows']) if PAIRS[r % NP_][1] == c]
            m_c[c] = R[idx].mean(0) if idx else mu
        R_cls = np.stack([m_c[PAIRS[r % NP_][1]] for r in f['te_rows']])
        E_class = float((((R_cls - mu) ** 2).sum()) / SS)
        m_i = {}
        for i in range(NE):
            idx = [j for j, r in enumerate(f['te_rows']) if PAIRS[r % NP_][0] == i]
            m_i[i] = R[idx].mean(0) if idx else mu
        R_add = np.stack([m_i[PAIRS[r % NP_][0]] + m_c[PAIRS[r % NP_][1]] - mu
                          for r in f['te_rows']])
        E_add = float((((R_add - mu) ** 2).sum()) / SS)
        R_cell = np.stack([R[cell_rows[PAIRS[r % NP_]]].mean(0) for r in f['te_rows']])
        E_cell = float((((R_cell - mu) ** 2).sum()) / SS)
        fp = [E_class, max(E_add - E_class, 0.0), max(E_cell - E_add, 0.0), max(1.0 - E_cell, 0.0)]
        return dict(e=e, te_rows=f['te_rows'], R=R, cells=cells,
                    cell_rows=cell_rows, spec_row=spec_row, spec_cell=spec_cell,
                    fp=fp, SS=SS, B4te=f['B4te'], Dk=f['Dk'])
    an_out = anatomy(KOUT)
    an_ks = anatomy(KSTAR)
    # k* 层谱应显著低秩尖锐（对照）；读出层预期平坦
    log('anatomy@KOUT spec_cell[:5]=%s fp=%s' %
        (['%.3f' % x for x in an_out['spec_cell'][:5]], ['%.3f' % x for x in an_out['fp']]))
    log('anatomy@KSTAR spec_cell[:5]=%s fp=%s' %
        (['%.3f' % x for x in an_ks['spec_cell'][:5]], ['%.3f' % x for x in an_ks['fp']]))

    # ---- (1) worst20 失败模态分类表 ----
    R = an_out['R']
    te_rows = an_out['te_rows']
    cls_tab = []
    e7o_ord = np.argsort(-an_out['e'])[:WORST_N]
    for li in e7o_ord:
        j = int(li)
        row = te_rows[j]
        i, c = PAIRS[row % NP_]
        r = R[j]
        rn2 = float((r ** 2).sum()) + 1e-18
        # E_g: 网格均值能量（cell 内含自身 3 模板行）
        g = R[an_out['cell_rows'][PAIRS[row % NP_]]].mean(0)
        E_g = float((g ** 2).sum() / rn2)
        # 类子空间投影（Pc_ko, Pc_ks）
        p_ko = float(((r @ Pc_ko) ** 2).sum() / rn2)
        p_ks = float(((r @ Pc_ks) ** 2).sum() / rn2)
        if c != CLS_OF[i]:
            mod = 'M1_mismatch'
        elif c in hard2:
            mod = 'M2_hardclass'
        elif E_g < EG_GATE:
            mod = 'M3_scatter'
        else:
            mod = 'M4_gridspecific'
        cls_tab.append(dict(local_i=j, row=int(row), tpl=int(row // NP_),
                            ent=ENTS[i], cls=CLASSES[c], truth=CLASSES[CLS_OF[i]],
                            rel_err=float(an_out['e'][j]), Eg=E_g,
                            proj_pcko=p_ko, proj_pcks=p_ks, mode=mod))
    n_mod = {}
    for m in ['M1_mismatch', 'M2_hardclass', 'M3_scatter', 'M4_gridspecific']:
        n_mod[m] = sum(1 for x in cls_tab if x['mode'] == m)
    coverage = (n_mod['M1_mismatch'] + n_mod['M2_hardclass'] +
                n_mod['M4_gridspecific']) / WORST_N
    log('modes=%s coverage=%.2f' % (n_mod, coverage))

    # ---- (3) M1 rank-5 因子（k*, seed7, gate 路径 ALS100）与 Pc 主角谱 ----
    f7k_r = f7k
    Y = Yk
    tr_set = set(f7k_r['train_set'])
    Rm1 = np.zeros((NE_KEEP, NC, D), np.float32)
    Mte = np.zeros((len(f7k_r['te_rows']), D), np.float32)
    for t in range(NT):
        Rg = np.zeros((NE_KEEP, NC, D), np.float32)
        mask = np.zeros((NE_KEEP, NC), bool)
        for pi, (i, c) in enumerate(PAIRS):
            if (i, c) in tr_set:
                Rg[keep_e.index(i), c] = \
                    H16[t, pi, KSTAR, :].astype(np.float32) - \
                    f7k_r['B4tr'][f7k_r['tr_rows'].index(t * NP_ + pi)]
                mask[keep_e.index(i), c] = True
        Rhat = als_complete(Rg, mask, RANK_M1, ALS_ITERS, ALS_RIDGE,
                            ALS_SEED + 1000 * SEEDS_S1[0] + KSTAR + t)
        if Rhat is None:
            Rhat = np.zeros_like(Rg)
        Rm1 += Rhat / NT
        for j, r in enumerate(f7k_r['te_rows']):
            if r // NP_ == t:
                i2, c2 = PAIRS[r % NP_]
                Mte[j] = f7k_r['B4te'][j] + Rhat[keep_e.index(i2), c2]
        log('M1@k* tpl%d done' % t)
    Yte = Y[f7k_r['te_rows']]
    e_m1_mean = float((((Mte - Yte) ** 2).sum(1) / f7k_r['Dk']).mean())
    e_b4_k = float((((f7k_r['B4te'] - Y[f7k_r['te_rows']]) ** 2).sum(1)
                    / f7k_r['Dk']).mean())
    m1_margin_s7 = e_m1_mean - e_b4_k
    # 断言 vs 3152 gates.M1_kstar.margins[0]
    mk = r3152.get('gates', {}).get('M1_kstar', {})
    if (not SMOKE) and mk.get('margins') and mk['margins'][0] is not None:
        dv = abs(m1_margin_s7 - mk['margins'][0])
        assert dv < 1e-6, ('M1@k* s7 margin drift vs 3152', m1_margin_s7, mk['margins'][0])
        log('M1@k* s7 margin cross-check OK (=%.4f)' % m1_margin_s7)
    # M1 网格锚（k*_r5_l0.01, iters=M1_GRID_ITERS=40）
    Rm1g = np.zeros((NE_KEEP, NC, D), np.float32)
    Mte = np.zeros((len(f7k_r['te_rows']), D), np.float32)
    for t in range(NT):
        Rg = np.zeros((NE_KEEP, NC, D), np.float32)
        mask = np.zeros((NE_KEEP, NC), bool)
        for pi, (i, c) in enumerate(PAIRS):
            if (i, c) in tr_set:
                Rg[keep_e.index(i), c] = \
                    H16[t, pi, KSTAR, :].astype(np.float32) - \
                    f7k_r['B4tr'][f7k_r['tr_rows'].index(t * NP_ + pi)]
                mask[keep_e.index(i), c] = True
        Rhat = als_complete(Rg, mask, RANK_M1, M1_GRID_ITERS, ALS_RIDGE,
                            ALS_SEED + 1000 * SEEDS_S1[0] + KSTAR + t + RANK_M1)
        if Rhat is None:
            Rhat = np.zeros_like(Rg)
        for j, r in enumerate(f7k_r['te_rows']):
            if r // NP_ == t:
                i2, c2 = PAIRS[r % NP_]
                Mte[j] = f7k_r['B4te'][j] + Rhat[keep_e.index(i2), c2]
    e_m1_grid = float((((Mte - Y[f7k_r['te_rows']]) ** 2).sum(1) / f7k_r['Dk']).mean())
    gkey = 'k%d_r%d_l%g' % (KSTAR, RANK_M1, ALS_RIDGE)
    g3152 = r3152.get('m1_grid', {}).get(gkey)
    if (not SMOKE) and g3152 is not None:
        dv = abs(e_m1_grid - g3152)
        assert dv < 1e-4, ('M1 grid anchor drift', gkey, e_m1_grid, g3152)
        log('M1 grid %s cross-check OK (=%.4f)' % (gkey, e_m1_grid))
    # 主角谱
    Rm1f = Rm1.reshape(-1, D)
    _, _, Vm = np.linalg.svd(Rm1f, full_matrices=False)
    Vm5 = Vm[:5].T
    def principal_angles(P):
        M = P.T @ Vm5
        sv = np.linalg.svd(M, compute_uv=False)
        return [float(x) for x in sv]
    align_ks = principal_angles(Pc_ks)
    align_ko = principal_angles(Pc_ko)
    log('principal spectrum M1vsPc(k*)=%s M1vsPc(KOUT)=%s' %
        (['%.3f' % x for x in align_ks], ['%.3f' % x for x in align_ko]))

    verdict = ('g1p3_%s|spec_cell_ko_s1_%.3f_s2_%.3f|fp_ko_%s|coverage_%.2f|'
               'modes_%d_%d_%d_%d|align_ks_top1_%.3f|align_ko_top1_%.3f' %
               (MODEL, an_out['spec_cell'][1], an_out['spec_cell'][2],
                '_'.join('%.2f' % x for x in an_out['fp']),
                coverage, n_mod['M1_mismatch'], n_mod['M2_hardclass'],
                n_mod['M3_scatter'], n_mod['M4_gridspecific'],
                align_ks[0], align_ko[0]))
    result = dict(
        phase=3153, name=NAME, mode=MODEL,
        created=time.strftime('%Y-%m-%d %H:%M:%S'),
        runtime_s=round(time.time() - T0, 1),
        design_sha=exe_sha, smoke=SMOKE, src=SRC, src_sha8=src_sha,
        ref_3152=R3152, ref_3152_sha8=r3152.get('res_sha8'),
        panel_rows=PANEL, n_pairs=NP_, NL=NL, hidden=D,
        kstar=KSTAR, readout=KOUT,
        anchor_checks=dict(worst20_relerr_max_drift=float(drift_w),
                           s2_vs_3152_v3=bool(bool(v3152)),
                           v5_drift=None if not v5_3152 else
                           dict(e_ks=abs(e_ks - v5_3152['class_subspace_energy_kstar']),
                                e_ko=abs(e_ko - v5_3152['class_subspace_energy_readout'])),
                           m1_margin_s7=float(m1_margin_s7),
                           m1_grid_kstar_r5=float(e_m1_grid)),
        s2_b4_readout={CLASSES[c]: s2_b4[c] for c in range(NC)},
        hard_classes=[CLASSES[c] for c in rank_cls[:2]],
        v5_recheck=dict(e_ks=e_ks, sv_ks=sv_ks, e_ko=e_ko, sv_ko=sv_ko),
        anatomy_kout=dict(spec_row=an_out['spec_row'], spec_cell=an_out['spec_cell'],
                          fp=an_out['fp'], n_cells=len(an_out['cells'])),
        anatomy_kstar=dict(spec_row=an_ks['spec_row'], spec_cell=an_ks['spec_cell'],
                           fp=an_ks['fp'], n_cells=len(an_ks['cells'])),
        worst20_modes=cls_tab, mode_counts=n_mod,
        coverage=coverage, coverage_gate_pass=bool(coverage >= COV_GATE),
        m1_align=dict(margin_s7=m1_margin_s7, e_m1_mean=e_m1_mean, e_b4_k=e_b4_k,
                      principal_vs_pckstar=align_ks, principal_vs_pckout=align_ko),
        grade='statistical', verdict=verdict)
    seal_result(result, 'result.json')
    log('DONE mode=%s runtime %.1fs' % (MODEL, time.time() - T0))
    sys.exit(0)

# =================================================================
# 模式 D：summary —— 跨模型 Jaccard + 指纹 Pearson + 验收/死线判决
# =================================================================
if MODEL == 'summary':
    B3153 = os.path.join(RDIR, 'phase3153', NAME)
    P4B = os.path.join(B3153, 'qwen3-4b', 'result.json')
    P14B = os.path.join(B3153, 'qwen3-14b', 'result.json')
    PG4 = os.path.join(B3153, 'glm4', 'result.json')
    r4b = json.load(open(P4B, encoding='utf-8'))
    r14b = json.load(open(P14B, encoding='utf-8'))
    rg4 = json.load(open(PG4, encoding='utf-8'))
    design = dict(inputs={'qwen3-4b': dict(path=P4B, sha8=r4b.get('res_sha8')),
                          'qwen3-14b': dict(path=P14B, sha8=r14b.get('res_sha8')),
                          'glm4': dict(path=PG4, sha8=rg4.get('res_sha8'))},
                  fp_gate=FP_GATE, cov_gate=COV_GATE,
                  dead_line='if any fingerprint pair corr < 0.8 -> drop readout class-subspace narrative',
                  pre_reg='MEMO L14878 item (4)')
    exe_sha = freeze_design('g1p3_summary', design)
    mods = {'qwen3-4b': r4b, 'qwen3-14b': r14b, 'glm4-9b': rg4}

    def jac(a, b, keyf):
        A = set(keyf(x) for x in a)
        B = set(keyf(x) for x in b)
        return len(A & B) / max(1, len(A | B))
    names = list(mods.keys())
    j_sample = {}
    j_pair = {}
    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            a = mods[names[x]]['worst20_modes']
            b = mods[names[y]]['worst20_modes']
            j_sample['%s__%s' % (names[x], names[y])] = jac(
                a, b, lambda w: (w['ent'], w['cls'], w['tpl']))
            j_pair['%s__%s' % (names[x], names[y])] = jac(
                a, b, lambda w: (w['ent'], w['cls']))
    # 指纹 Pearson：spec_cell(top10) 与 fp(4 维)
    def pear(u, v):
        u = np.asarray(u, np.float64)
        v = np.asarray(v, np.float64)
        if u.std() < 1e-12 or v.std() < 1e-12:
            return 0.0
        return float(((u - u.mean()) * (v - v.mean())).mean() /
                     (u.std() * v.std() + 1e-18))
    fp_corr = {}
    spec_corr = {}
    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            k = '%s__%s' % (names[x], names[y])
            fp_corr[k] = pear(mods[names[x]]['anatomy_kout']['fp'],
                              mods[names[y]]['anatomy_kout']['fp'])
            spec_corr[k] = pear(mods[names[x]]['anatomy_kout']['spec_cell'],
                                mods[names[y]]['anatomy_kout']['spec_cell'])
    fp_ok = all(v >= FP_GATE for v in fp_corr.values())
    spec_ok = all(v >= FP_GATE for v in spec_corr.values())
    fingerprint_consistent = fp_ok and spec_ok
    cov_all = all(m['coverage_gate_pass'] for m in mods.values())
    cov_mean = float(np.mean([m['coverage'] for m in mods.values()]))
    if fingerprint_consistent and cov_all:
        verdict_line = ('g1p3_anatomy_fingerprint_consistent_'
                        'readout_highrank_scatter_confirmed')
    elif not fingerprint_consistent:
        verdict_line = ('g1p3_fingerprint_inconsistent_'
                        'g1_readout_class_subspace_narrative_dropped')
    else:
        verdict_line = 'g1p3_fingerprint_consistent_coverage_partial'
    align_note = {}
    for mn, m in mods.items():
        align_note[mn] = dict(
            top1_pckstar=m['m1_align']['principal_vs_pckstar'][0],
            top1_pckout=m['m1_align']['principal_vs_pckout'][0],
            interpretation=('m1_factor_in_class_subspace'
                            if m['m1_align']['principal_vs_pckstar'][0] >= 0.8
                            else 'm1_factor_not_class_subspace'))
    modes_tab = {mn: m['mode_counts'] for mn, m in mods.items()}
    verdict = ('g1p3_summary|%s|fp_corr_%s|spec_corr_%s|cov_mean_%.2f|'
               'jpair_max_%.2f' %
               (verdict_line,
                '_'.join('%.2f' % v for v in fp_corr.values()),
                '_'.join('%.2f' % v for v in spec_corr.values()),
                cov_mean, max(j_pair.values())))
    result = dict(
        phase=3153, name=NAME, mode='summary',
        created=time.strftime('%Y-%m-%d %H:%M:%S'),
        runtime_s=round(time.time() - T0, 1),
        design_sha=exe_sha, inputs_used=design['inputs'],
        worst20_jaccard_sample=j_sample, worst20_jaccard_pair=j_pair,
        fp_corr=fp_corr, spec_corr=spec_corr,
        fingerprint_consistent=bool(fingerprint_consistent),
        coverage_all_pass=bool(cov_all), coverage_mean=cov_mean,
        modes_table=modes_tab, align_note=align_note,
        dead_line_triggered=bool(not fingerprint_consistent),
        grade='statistical', verdict=verdict)
    seal_result(result, 'result_summary.json')
    log('DONE mode=summary runtime %.1fs' % (time.time() - T0))
    sys.exit(0)

log('unknown mode %s' % MODEL)
sys.exit(2)
