# -*- coding: utf-8 -*-
# Phase 3152 (G1-P2): K1 三模型定夺 —— 全管线复现 + 层位冻结 + M1 过拟合解剖
# 预注册：AGI_GPT5_MEMO L14847（观测前冻结）
# 运行模式（P3152_MODEL 环境变量 或 argv[1]）:
#   qwen3-4b  : 全管线 GPU（NL=36, D=2560; k*=3, readout=35）
#   qwen3-14b : 全管线 GPU（NL=40, D=5120; k*=3, readout=39）
#   glm4k1    : 零 GPU —— 读 3151 collect.npz 重算 B4@k3/k39(S1x3seed) + M1@k3 门
#               + M1 网格解剖 + A3 worst20；B4@k39 对 3151 锚 0.389835 断言
#   summary   : 零 GPU —— 读三份 result，K1 三模型判决（判决层位=机制层 k*，深度~7.5%）
# 内置 3151 教训: paired 负=优(m<=-2*MDE); ALS solve 尾维[...,None]; one-hot 共线 lam>=1e-3;
#                SMOKE 目录分离; Pc=Vt[:5].T; stdout reconfigure
# 层位深度分数对齐: k*=round(0.075*NL); KOUT=NL-1; KGRID=round({0.375,0.575,0.775,0.975}*NL)
#   NL40 -> k*=3,KOUT=39,KGRID=[15,23,31,39]（与 3151 完全同）; NL36 -> k*=3,KOUT=35,KGRID=[14,21,28,35]
import os, sys, json, time, hashlib
import numpy as np

T0 = time.time()
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
MODEL = os.environ.get('P3152_MODEL') or (sys.argv[1] if len(sys.argv) > 1 else 'qwen3-4b')
SMOKE = os.environ.get('P3152_SMOKE') == '1'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result',
                    'rdc_query_construction_20260913')
NAME = 'g1p2_tri_model_k1'
BASE = os.path.join(RDIR, 'phase3152', NAME, MODEL)
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
    eblob = json.dumps(design, ensure_ascii=False, sort_keys=True,
                       indent=1).encode('utf-8')
    sha = hashlib.sha256(eblob).hexdigest()
    exe_p = os.path.join(BASE, 'execution.json')
    if os.path.exists(exe_p):
        prev = json.load(open(exe_p, encoding='utf-8'))
        assert prev['design_sha'] == sha, 'DESIGN DRIFT'
        log('execution.json match (sha %s)' % sha[:8])
    else:
        json.dump({'phase': 3152, 'name': phase_name, 'design_sha': sha,
                   'design': design,
                   'frozen_before': 'any model observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        log('execution.json FROZEN (sha %s)' % sha[:8])
    return sha

def seal_result(result, out_name):
    blob = json.dumps(result, ensure_ascii=False, indent=1,
                      sort_keys=True).encode('utf-8')
    res_sha8 = hashlib.sha256(blob).hexdigest()[:8]
    result['res_sha8'] = res_sha8
    result['verdict'] = result['verdict'] + '|sha8_' + res_sha8
    rp = os.path.join(BASE, out_name)
    json.dump(result, open(rp, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    seal = hashlib.sha256(open(rp, 'rb').read()).hexdigest()[:8]
    result['seal_sha8'] = seal
    json.dump(result, open(rp, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    log('RESULT %s res_sha8=%s seal=%s verdict=%s' %
        (out_name, res_sha8, seal, result['verdict']))
    return res_sha8, seal

# ---------------- 冻结面板材料（与 3151 逐字一致） ----------------
CLASSES = ['水果', '动物', '交通工具', '家具', '金属', '颜色']
ENT = {
    '水果': ['苹果', '香蕉', '梨', '西瓜', '葡萄', '草莓', '芒果', '柠檬'],
    '动物': ['狗', '猫', '老虎', '大象', '兔子', '猴子', '马', '牛'],
    '交通工具': ['汽车', '火车', '飞机', '摩托车', '卡车', '地铁'],
    '家具': ['桌子', '椅子', '床', '沙发', '地毯', '窗帘'],
    '金属': ['铁', '铜', '铝', '金', '银', '锌', '铅'],
    '颜色': ['红', '蓝', '绿', '黄', '黑', '白'],
}
TPL = {0: '{e}是一种{c}。',
       1: '{e}属于{c}这一类。',
       2: '{e}，一种常见的{c}。'}
SEEDS_S1 = [7, 8, 9]
FRAC_S1 = 0.2
RANK_M1 = 5
ALS_SEED = 7
ALS_RIDGE = 1e-2
ALS_ITERS = 20 if SMOKE else 100
M2_M = 32 if SMOKE else 128
B5_UNITS = 64
LAMBDA_GRID = [1e-3, 1e-2, 1e-1, 1.0]
V0_MARGIN_MIN = 1.0
ADD_ERR_GATE = 0.05
KSTAR_FRAC = 0.075
GRID_FRACS = (0.375, 0.575, 0.775, 0.975)
M1_GRID_RANKS = [2, 5] if SMOKE else [1, 2, 5, 10]
M1_GRID_RIDGES = [1e-2] if SMOKE else [1e-3, 1e-2, 1e-1]
M1_GRID_ITERS = 5 if SMOKE else 40
CURVE_FRACS = [0.075 + 0.1 * i for i in range(10)]
WORST_N = 20
PANEL_MIN = 672

ENTS = [e for cl in CLASSES for e in ENT[cl]]
CLS_OF = [CLASSES.index(cl) for cl in CLASSES for e in ENT[cl]]
NE = len(ENTS)
NC = len(CLASSES)
PAIRS = [(i, c) for i in range(NE) for c in range(NC)]
NP_ = len(PAIRS)                       # 246
NT = len(TPL)                          # 3
PANEL_FULL = NT * NP_                  # 738

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
    return [t * NP_ + pi for t in range(NT)
            for pi, p in enumerate(PAIRS) if p in pair_set]

S2 = []
for cstar in range(NC):
    te = set([p for p in PAIRS if p[1] == cstar])
    S2.append((cstar, ALLP - te, te))
S3 = []
for e in keep_e:
    te = set([p for p in PAIRS if p[0] == e])
    S3.append((e, ALLP - te, te))

# ---------------- 共享拟合器（与 3151 位级一致） ----------------
def ridge_primal(Xtr, Ytr, lam=1e-3):
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1], dtype=np.float32)
    W = np.linalg.solve(A, Xtr.T @ Ytr)
    return W

def ridge_dual(Ftr, Ytr, lam):
    K = Ftr @ Ftr.T
    n = Ftr.shape[0]
    al = np.linalg.solve(K + lam * n * np.eye(n, dtype=np.float32), Ytr)
    return al, Ftr

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
        U = np.linalg.solve(
            A, b.transpose(2, 0, 1)[..., None])[..., 0].transpose(1, 2, 0)
        if not np.isfinite(U).all():
            return None
        A = np.einsum('ec,erd,esd->crsd', mask, U, U)
        b = np.einsum('ec,ecd,erd->crd', mask, R0, U)
        A = A.transpose(3, 0, 1, 2)
        A[..., rid, rid] += ridge
        V = np.linalg.solve(
            A, b.transpose(2, 0, 1)[..., None])[..., 0].transpose(1, 2, 0)
        if not np.isfinite(V).all():
            return None
    Rhat = np.einsum('erd,crd->ecd', U, V) + mu
    return Rhat.astype(np.float32)

def elm_fit(Xtr, Ytr, seed=ALS_SEED):
    rng = np.random.RandomState(seed)
    W1 = rng.randn(Xtr.shape[1], B5_UNITS).astype(np.float32) / np.sqrt(Xtr.shape[1])
    b1 = rng.randn(B5_UNITS).astype(np.float32) * 0.1
    Ftr = np.maximum(Xtr @ W1 + b1, 0)
    Ftr = np.concatenate([Ftr, np.ones((len(Ftr), 1), np.float32)], 1)
    A = Ftr.T @ Ftr + 1e-3 * np.eye(Ftr.shape[1], dtype=np.float32)
    W2 = np.linalg.solve(A, Ftr.T @ Ytr)
    return W1, b1, W2

def elm_pred(W1, b1, W2, X):
    F = np.maximum(X @ W1 + b1, 0)
    F = np.concatenate([F, np.ones((len(F), 1), np.float32)], 1)
    return F @ W2

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

RES = {}
PAIR_ERR = {}

def evaluate(name, split_id, k, pred_rows, err_rows, Ytrue, ref, Dk,
             keep_pair_err=False):
    pr = np.stack(pred_rows)
    tr = np.stack([Ytrue[r] for r in err_rows])
    e = ((pr - tr) ** 2).sum(1) / Dk
    cs = float(np.mean([float((a * b).sum() /
                              (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
                        for a, b in zip(pr, tr)]))
    RES[(name, split_id, k)] = dict(mean_rel=float(e.mean()),
                                    mean_cos=cs, n=len(err_rows))
    if keep_pair_err:
        PAIR_ERR[(name, split_id, k)] = e
    return e

def paired(mA, mB, split_id, k):
    """margin = err_cand - err_base; 负 = 候选优; pass 当 m <= -2*MDE"""
    a = PAIR_ERR.get((mA, split_id, k))
    b = PAIR_ERR.get((mB, split_id, k))
    if a is None or b is None:
        return None
    d = a - b
    n = len(d)
    m = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    mde = 1.96 * sd / np.sqrt(n)
    return dict(margin=m, mde=float(mde), n=n,
                pass_gate=bool(m <= -2 * mde))

def gate_pack(cand, k, seeds=SEEDS_S1):
    gs = [paired(cand, 'B4', 'S1_s%d' % s, k) for s in seeds]
    ok = all(g is not None and g['pass_gate'] for g in gs)
    return dict(pass_all3=bool(ok),
                margins=[None if g is None else g['margin'] for g in gs],
                mdes=[None if g is None else g['mde'] for g in gs])

# =================================================================
# 模式 A：qwen3-4b / qwen3-14b 全管线（GPU）
# =================================================================
if MODEL in ('qwen3-4b', 'qwen3-14b'):
    MDIR_MAP = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'Qwen3-14B'}
    MDIR = os.path.join(ROOT, 'models', 'hf', MDIR_MAP[MODEL])
    cfg = json.load(open(os.path.join(MDIR, 'config.json'), encoding='utf-8'))
    NL = cfg['num_hidden_layers']
    HID = cfg['hidden_size']
    KSTAR = int(round(KSTAR_FRAC * NL))
    KOUT = NL - 1
    GATE_LAYERS = [KSTAR, KOUT]
    KGRID = sorted(set(int(round(f * NL)) for f in GRID_FRACS))
    M1_CURVE = sorted(set(int(round(f * NL)) for f in CURVE_FRACS))
    CURVE_LAYERS = list(range(NL + 1))
    design = dict(model=MODEL, mdir=MDIR, classes=CLASSES, ents=ENTS, tpl=TPL,
                  nl=NL, hidden=HID, kstar=KSTAR, readout=KOUT, kgrid=KGRID,
                  m1_curve_layers=M1_CURVE, curve_layers='0..%d' % NL,
                  seeds_s1=SEEDS_S1, frac_s1=FRAC_S1, rank_m1=RANK_M1,
                  als_iters=ALS_ITERS, als_ridge=ALS_RIDGE, m2_m=M2_M,
                  b5_units=B5_UNITS, lambda_grid=LAMBDA_GRID,
                  v0_margin_min=V0_MARGIN_MIN, add_err_gate=ADD_ERR_GATE,
                  kstar_frac=KSTAR_FRAC, grid_fracs=list(GRID_FRACS),
                  m1_grid_ranks=M1_GRID_RANKS, m1_grid_ridges=M1_GRID_RIDGES,
                  m1_grid_iters=M1_GRID_ITERS, worst_n=WORST_N,
                  panel_rows_full=PANEL_FULL, panel_rows=PANEL, smoke=SMOKE,
                  pre_reg='MEMO L14847; 3151 lessons built-in')
    exe_sha = freeze_design('g1p2_full_%s' % MODEL, design)
    log('panel rows = %d (min %d full), smoke=%s model=%s NL=%d D=%d '
        'k*=%d readout=%d kgrid=%s' %
        (PANEL, PANEL_MIN, SMOKE, MODEL, NL, HID, KSTAR, KOUT, KGRID))
    assert PANEL >= PANEL_MIN or SMOKE, ('panel too small', PANEL)

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
    log('model load begin')
    model = AutoModelForCausalLM.from_pretrained(
        MDIR, dtype=torch.bfloat16, trust_remote_code=True).to('cuda').eval()
    log('model loaded: %s' % type(model).__name__)
    assert model.config.num_hidden_layers == NL
    D = HID

    def ids_of(text):
        return tok(text, add_special_tokens=False)['input_ids']

    CLS_TOK = [ids_of(c)[0] for c in CLASSES]
    E_TOK = [ids_of(ENTS[i])[0] for i in range(NE)]
    assert len(set(CLS_TOK)) == NC, 'class first tokens collide'

    prompts = []
    for t in range(NT):
        for (i, c) in PAIRS:
            prompts.append(TPL[t].format(e=ENTS[i], c=CLASSES[c]))
    assert len(prompts) == PANEL
    TOKIDS = [ids_of(p) for p in prompts]

    NH = NL + 1
    cache = os.path.join(BASE, 'collect_smoke.npz' if SMOKE else 'collect.npz')
    if os.path.exists(cache):
        z = np.load(cache)
        H16 = z['H']
        MARG = z['marg']
        log('collect cache hit %s H=%s' % (os.path.basename(cache), H16.shape))
    else:
        H16 = np.zeros((NT, NP_, NH, D), np.float16)
        MARG = np.zeros((NT, NP_, NC), np.float32)
        with torch.no_grad():
            for t in range(NT):
                for pi in range(NP_):
                    ii = torch.tensor([TOKIDS[t * NP_ + pi]], device='cuda')
                    o = model(input_ids=ii, output_hidden_states=True)
                    hs = o.hidden_states
                    hv = np.stack([h[0, -1].float().detach().cpu().numpy()
                                   for h in hs], 0)
                    H16[t, pi] = hv.astype(np.float16)
                    lg = o.logits[0, -1].float().detach().cpu().numpy()
                    cl = lg[[CLS_TOK[k] for k in range(NC)]]
                    MARG[t, pi] = cl - cl.mean()
                    del o, hs, hv, lg, cl
                log('collect tpl%d done' % t)
        np.savez_compressed(cache, H=H16, marg=MARG)
        log('collect saved %s' % os.path.basename(cache))
    # ---- embedding rows prefetch (model still on GPU) ----
    need = sorted(set(E_TOK + CLS_TOK +
                      sorted({t for tk in TOKIDS for t in tk})))
    with torch.no_grad():
        _idx = torch.tensor(need, device='cuda')
        _emb_rows = model.get_input_embeddings().weight[_idx].float().cpu().numpy()
    EMB = {'need': need, 'rows': _emb_rows}
    log('embedding rows prefetched: %s' % (_emb_rows.shape,))
    del model
    torch.cuda.empty_cache()
    log('model released (H16 in RAM)')

    def Y_at(k):
        return H16[:, :, k, :].reshape(NT * NP_, D).astype(np.float32)


    POS = {t: j for j, t in enumerate(EMB['need'])}
    def bag_feat(row):
        tk = TOKIDS[row]
        v = np.zeros(EMB['rows'].shape[1], np.float32)
        for t in tk:
            v += EMB['rows'][POS[t]]
        return v / np.sqrt(len(tk))

    def embpair_feat(t, pi):
        i, c = PAIRS[pi]
        er = EMB['rows'][POS[E_TOK[i]]]
        gr = EMB['rows'][POS[CLS_TOK[c]]]
        return er, gr

    rng_f = np.random.RandomState(ALS_SEED)
    R1 = rng_f.randn(M2_M, EMB['rows'].shape[1]).astype(np.float32) / np.sqrt(EMB['rows'].shape[1])
    R2 = rng_f.randn(M2_M, EMB['rows'].shape[1]).astype(np.float32) / np.sqrt(EMB['rows'].shape[1])

    all_rows = list(range(PANEL))
    log('prefetch B2/M2 features begin')
    FB2_all = np.stack([bag_feat(r) for r in all_rows])
    FM2_all = np.zeros((PANEL, M2_M), np.float32)
    for j, r in enumerate(all_rows):
        t, pi = r // NP_, r % NP_
        er, gr = embpair_feat(t, pi)
        FM2_all[j] = (R1 @ er) * (R2 @ gr)
    log('prefetch done FB2=%s FM2=%s' % (FB2_all.shape, FM2_all.shape))

    # ---- λ 选择（冻结：S1 seed7 KOUT 训练内 5-fold CV） ----
    LAMBDA_LIVE = {'M2': LAMBDA_GRID[0], 'B2': LAMBDA_GRID[1]}
    def select_lambdas():
        train_set, _ = split_s1(SEEDS_S1[0])
        tr_rows = rows_of(train_set)
        Y = Y_at(KOUT)
        Xtr, _, _, _ = phi_main(train_set)
        rng = np.random.RandomState(0)
        order = rng.permutation(len(tr_rows))
        folds = np.array_split(order, 5)
        for name, Ftr in [('M2', np.concatenate([Xtr, FM2_all[tr_rows]], 1)),
                          ('B2', FB2_all[tr_rows])]:
            best, best_v = None, 1e18
            for lam in LAMBDA_GRID:
                vs = []
                for f in range(5):
                    va = set(folds[f].tolist())
                    trn = [r for j, r in enumerate(tr_rows) if j not in va]
                    Fsub = Ftr[[j for j in range(len(tr_rows)) if j not in va]]
                    al, _ = ridge_dual(Fsub, Y[trn], lam)
                    Pv = Ftr[list(va)] @ Fsub.T @ al
                    v = float(((Pv - Y[[tr_rows[j] for j in folds[f]]]) ** 2).mean())
                    vs.append(v)
                if np.mean(vs) < best_v:
                    best_v, best = float(np.mean(vs)), lam
            LAMBDA_LIVE[name] = best
            log('lambda %s = %g (cv %.5f)' % (name, best, best_v))

    select_lambdas()

    # ---- 通用拟合评估（层循环；特征已预计算） ----
    def fit_and_eval(split_id, train_set, test_set, ks, models,
                     lam2=None, lamB2=None, keep_pair_err=False):
        out_errs = {}
        tr_rows = rows_of(train_set)
        te_rows = rows_of(test_set)
        tr_set = set(train_set)
        Xtr, _, rowvec, cols = phi_main(train_set)
        Xte = np.stack([rowvec(r // NP_, r % NP_) for r in te_rows])
        Ftr2 = np.concatenate([Xtr, FM2_all[tr_rows]], 1)
        Fte2 = np.concatenate([Xte, FM2_all[te_rows]], 1)
        FtrB = FB2_all[tr_rows]
        FteB = FB2_all[te_rows]
        for k in ks:
            Y = Y_at(k)
            Ytr = Y[tr_rows]
            ref = Ytr.mean(0)
            Dk = float(((Ytr - ref) ** 2).sum(1).mean()) + 1e-9
            preds = {}
            preds['P0'] = [ref] * len(te_rows)
            tpl_mean = {}
            for t in range(NT):
                rr = [r for r in tr_rows if r // NP_ == t]
                tpl_mean[t] = Y[rr].mean(0) if rr else ref
            preds['B3'] = [tpl_mean[r // NP_] for r in te_rows]
            W = ridge_primal(Xtr, Y[tr_rows], lam=1e-3)
            preds['B4'] = list(Xte @ W)
            c1 = 0
            mu_c = {}
            for c in range(NC):
                rr = [r for r in tr_rows if PAIRS[r % NP_][1] == c]
                mu_c[c] = Y[rr].mean(0) if rr else ref
            p1 = []
            for r in te_rows:
                t, pi = r // NP_, r % NP_
                i, c = PAIRS[pi]
                if (i, c1) in tr_set:
                    base = Y[t * NP_ + [pi2 for pi2, p2 in enumerate(PAIRS)
                                        if p2 == (i, c1)][0]]
                    p1.append(base + mu_c[c] - mu_c[c1])
                else:
                    p1.append(preds['B4'][len(p1)])
            preds['B1'] = p1
            lamB2_ = LAMBDA_LIVE['B2'] if lamB2 is None else lamB2
            al, _ = ridge_dual(FtrB, Ytr, lamB2_)
            preds['B2'] = list(FteB @ FtrB.T @ al)
            lam2_ = LAMBDA_LIVE['M2'] if lam2 is None else lam2
            al2, _ = ridge_dual(Ftr2, Ytr, lam2_)
            preds['M2'] = list(Fte2 @ Ftr2.T @ al2)
            if 'B5' in models:
                W1, b1, W2 = elm_fit(Xtr, Y[tr_rows])
                preds['B5'] = list(elm_pred(W1, b1, W2, Xte))
            for m in ['P0', 'B3', 'B1', 'B4', 'B2', 'M2'] + (
                    ['B5'] if 'B5' in models else []):
                keep = keep_pair_err and (k in GATE_LAYERS)
                evaluate(m, split_id, k, preds[m], te_rows, Y, ref, Dk,
                         keep_pair_err=keep)
                if keep:
                    out_errs[m] = PAIR_ERR[(m, split_id, k)]
            log('k=%d %s done (B4=%.4f M2=%.4f)' %
                (k, split_id, RES[('B4', split_id, k)]['mean_rel'],
                 RES[('M2', split_id, k)]['mean_rel']))
        return out_errs

    def fit_s1_with_m1(seed, ks):
        train_set, test_set = split_s1(seed)
        tr_rows = rows_of(train_set)
        te_rows = rows_of(test_set)
        tr_set = set(train_set)
        for k in ks:
            Y = Y_at(k)
            Ytr = Y[tr_rows]
            ref = Ytr.mean(0)
            Dk = float(((Ytr - ref) ** 2).sum(1).mean()) + 1e-9
            Xtr, _, rowvec, cols = phi_main(train_set)
            W = ridge_primal(Xtr, Ytr, lam=1e-3)
            Xte = np.stack([rowvec(r // NP_, r % NP_) for r in te_rows])
            B4te = Xte @ W
            B4tr = Xtr @ W
            M1te = np.zeros((len(te_rows), D), np.float32)
            for t in range(NT):
                Rg = np.zeros((NE_KEEP, NC, D), np.float32)
                mask = np.zeros((NE_KEEP, NC), bool)
                for pi, (i, c) in enumerate(PAIRS):
                    if (i, c) in tr_set:
                        Rg[keep_e.index(i), c] = \
                            H16[t, pi, k, :].astype(np.float32) - \
                            B4tr[tr_rows.index(t * NP_ + pi)]
                        mask[keep_e.index(i), c] = True
                Rhat = als_complete(Rg, mask, RANK_M1, ALS_ITERS, ALS_RIDGE,
                                    ALS_SEED + 1000 * seed + k + t)
                if Rhat is None:
                    Rhat = np.zeros_like(Rg)
                for j, r in enumerate(te_rows):
                    if r // NP_ != t:
                        continue
                    i, c = PAIRS[r % NP_]
                    M1te[j] = B4te[j] + Rhat[keep_e.index(i), c]
            Yte = Y[te_rows]
            for m, P in [('B4', B4te), ('M1', M1te)]:
                e = ((P - Yte) ** 2).sum(1) / Dk
                cs = float(np.mean([float((a * b).sum() /
                                          (np.linalg.norm(a) *
                                           np.linalg.norm(b) + 1e-9))
                                    for a, b in zip(P, Yte)]))
                RES[(m, 'S1_s%d' % seed, k)] = dict(mean_rel=float(e.mean()),
                                                    mean_cos=cs,
                                                    n=len(te_rows))
                if k in GATE_LAYERS:
                    PAIR_ERR[(m, 'S1_s%d' % seed, k)] = e
            log('S1 s%d k=%d B4=%.4f M1=%.4f' %
                (seed, k, RES[('B4', 'S1_s%d' % seed, k)]['mean_rel'],
                 RES[('M1', 'S1_s%d' % seed, k)]['mean_rel']))

    def m1_grid_eval(seed, k, rank, ridge, iters, b4_cache):
        if (seed, k) in b4_cache:
            B4te, B4tr, te_rows, tr_rows, tr_set = b4_cache[(seed, k)]
        else:
            train_set, test_set = split_s1(seed)
            tr_rows = rows_of(train_set)
            te_rows = rows_of(test_set)
            tr_set = set(train_set)
            Xtr, _, rowvec, cols = phi_main(train_set)
            Y = Y_at(k)
            W = ridge_primal(Xtr, Y[tr_rows], lam=1e-3)
            Xte = np.stack([rowvec(r // NP_, r % NP_) for r in te_rows])
            B4te = Xte @ W
            B4tr = Xtr @ W
            b4_cache[(seed, k)] = (B4te, B4tr, te_rows, tr_rows, tr_set)
        Y = Y_at(k)
        ref = Y[tr_rows].mean(0)
        Dk = float(((Y[tr_rows] - ref) ** 2).sum(1).mean()) + 1e-9
        M1te = np.zeros((len(te_rows), D), np.float32)
        for t in range(NT):
            Rg = np.zeros((NE_KEEP, NC, D), np.float32)
            mask = np.zeros((NE_KEEP, NC), bool)
            for pi, (i, c) in enumerate(PAIRS):
                if (i, c) in tr_set:
                    Rg[keep_e.index(i), c] = \
                        H16[t, pi, k, :].astype(np.float32) - \
                        B4tr[tr_rows.index(t * NP_ + pi)]
                    mask[keep_e.index(i), c] = True
            Rhat = als_complete(Rg, mask, rank, iters, ridge,
                                ALS_SEED + 1000 * seed + k + t + rank)
            if Rhat is None:
                Rhat = np.zeros_like(Rg)
            for j, r in enumerate(te_rows):
                if r // NP_ != t:
                    continue
                i, c = PAIRS[r % NP_]
                M1te[j] = B4te[j] + Rhat[keep_e.index(i), c]
        Yte = Y[te_rows]
        e_m1 = float((((M1te - Yte) ** 2).sum(1) / Dk).mean())
        e_b4 = float((((B4te - Yte) ** 2).sum(1) / Dk).mean())
        return e_m1, e_b4

    # ---- 主流程 ----
    if SMOKE:
        KS = GATE_LAYERS + [(KSTAR + KOUT) // 2]
        M1KS = GATE_LAYERS
    else:
        KS = CURVE_LAYERS
        M1KS = M1_CURVE
    log('curve start: KS=%s M1KS=%s' % (KS, M1KS))
    fit_and_eval('S1_s7', *split_s1(SEEDS_S1[0]), ks=KS, models=['B5'],
                 keep_pair_err=True)
    for k in M1KS:
        fit_s1_with_m1(SEEDS_S1[0], [k])
    for sd in SEEDS_S1[1:]:
        fit_and_eval('S1_s%d' % sd, *split_s1(sd), ks=GATE_LAYERS,
                     models=['B5'], keep_pair_err=True)
        fit_s1_with_m1(sd, GATE_LAYERS)
    # S2 留一类
    s2_tab = {}
    for cstar, tr, te in S2:
        fit_and_eval('S2_c%d' % cstar, tr, te, ks=GATE_LAYERS, models=[],
                     keep_pair_err=True)
        for k in GATE_LAYERS:
            s2_tab[(cstar, k)] = (RES[('B4', 'S2_c%d' % cstar, k)]['mean_rel'],
                                  RES[('M2', 'S2_c%d' % cstar, k)]['mean_rel'])
    # S3 留一实例（读出层）
    s3_tab = {}
    for e, tr, te in S3:
        fit_and_eval('S3_e%d' % e, tr, te, ks=[KOUT], models=[],
                     keep_pair_err=True)
        s3_tab[e] = RES[('B4', 'S3_e%d' % e, KOUT)]['mean_rel']
    # M1 网格解剖
    log('M1 grid begin (%d fits)' %
        (len(KGRID) * len(M1_GRID_RANKS) * len(M1_GRID_RIDGES)))
    b4_cache = {}
    grid_res = {}
    for k in KGRID:
        for rank in M1_GRID_RANKS:
            for ridge in M1_GRID_RIDGES:
                e_m1, e_b4 = m1_grid_eval(SEEDS_S1[0], k, rank, ridge,
                                          M1_GRID_ITERS, b4_cache)
                grid_res['k%d_r%d_l%g' % (k, rank, ridge)] = e_m1
                grid_res['k%d_B4' % k] = e_b4
                log('grid k=%d rank=%d ridge=%g -> M1=%.4f (B4=%.4f)' %
                    (k, rank, ridge, e_m1, e_b4))
    opt_rank = {}
    for k in KGRID:
        key = lambda r: grid_res['k%d_r%d_l%g' % (k, r, M1_GRID_RIDGES[0])]
        if len(M1_GRID_RIDGES) > 1:
            key = lambda r: min(grid_res['k%d_r%d_l%g' % (k, r, lg)]
                                for lg in M1_GRID_RIDGES)
        opt_rank[k] = int(min(M1_GRID_RANKS, key=key))
    log('optimal rank per k: %s' % opt_rank)

    # V0 材料有效性
    true_pairs = [(t, pi) for t in range(NT) for pi, (i, c) in
                  enumerate(PAIRS) if c == CLS_OF[i]]
    med_true = float(np.median([MARG[t, pi, c] for (t, pi) in true_pairs]))
    # V2 容量对照（KOUT）
    v2 = None
    m1_ok_k = gate_pack('M1', KOUT)
    if m1_ok_k['pass_all3']:
        g2 = [paired('M1', 'B5', 'S1_s%d' % s, KOUT) for s in SEEDS_S1]
        ok2 = all(g and g['pass_gate'] for g in g2)
        v2 = 'gain_above_capacity' if ok2 else 'gain_is_capacity'
    # V3 留一类
    fruit_ok = all(s2_tab[(0, k)][1] - s2_tab[(0, k)][0] >= -0.005
                   for k in GATE_LAYERS)
    worst_c = max(range(NC), key=lambda c: s2_tab[(c, KOUT)][0])
    s2_res = {}
    for (cstar, k), (b4, m2) in sorted(s2_tab.items()):
        s2_res['c%d_k%d' % (cstar, k)] = dict(b4=b4, m2=m2,
                                              margin_m2_minus_b4=m2 - b4)
    # V5 类子空间能量
    def v5_energy(k):
        train_set, _ = split_s1(SEEDS_S1[0])
        tr_rows = rows_of(train_set)
        Y = Y_at(k)
        Xtr, _, rowvec, cols = phi_main(train_set)
        W = ridge_primal(Xtr, Y[tr_rows], lam=1e-3)
        Rg = np.zeros((NE_KEEP, NC, D), np.float32)
        for t in range(NT):
            grid = H16[t, :, k, :].astype(np.float32)
            for pi in range(NP_):
                pv = rowvec(t, pi) @ W
                i, c = PAIRS[pi]
                Rg[keep_e.index(i), c] = grid[pi] - pv
        Rf = Rg.reshape(-1, D)
        cm = np.stack([Rg[[keep_e.index(i) for i in keep_e
                           if CLS_OF[i] == c], c].mean(0)
                       for c in range(NC)])
        cmc = cm - cm.mean(0)
        U, S, Vt = np.linalg.svd(cmc, full_matrices=False)
        Pc = Vt[:5].T
        proj = Rf @ Pc
        e_class = float((proj ** 2).sum() / ((Rf ** 2).sum() + 1e-9))
        return e_class, float(S[:5].sum() / (S.sum() + 1e-9))
    e_ks, sv_ks = v5_energy(KSTAR)
    e_ko, sv_ko = v5_energy(KOUT)
    # A3 worst-20（S1 seed7 读出层 B4）
    b4e7 = PAIR_ERR[('B4', 'S1_s7', KOUT)]
    te7 = rows_of(split_s1(SEEDS_S1[0])[1])
    order = np.argsort(-b4e7)[:WORST_N]
    worst = []
    for li in order:
        row = te7[int(li)]
        i, c = PAIRS[row % NP_]
        worst.append(dict(local_i=int(li), row=int(row), tpl=int(row // NP_),
                          ent=ENTS[i], cls=CLASSES[c],
                          prompt=prompts[row], rel_err=float(b4e7[int(li)])))
    # K1 模型报告
    b4_k = [float(np.mean(PAIR_ERR[('B4', 'S1_s%d' % s, KSTAR)]))
            for s in SEEDS_S1]
    b4_o = [float(np.mean(PAIR_ERR[('B4', 'S1_s%d' % s, KOUT)]))
            for s in SEEDS_S1]
    k1_model_report = dict(
        model=MODEL, NL=NL, kstar=KSTAR, readout=KOUT,
        b4_rel_kstar_mean3seed=float(np.mean(b4_k)),
        b4_rel_kstar_per_seed=b4_k,
        b4_rel_readout_mean3seed=float(np.mean(b4_o)),
        b4_rel_readout_per_seed=b4_o,
        m1_kstar=gate_pack('M1', KSTAR), m2_kstar=gate_pack('M2', KSTAR),
        m1_readout=gate_pack('M1', KOUT), m2_readout=gate_pack('M2', KOUT),
        above_add_gate=bool(np.mean(b4_k) > ADD_ERR_GATE))
    curve = {}
    for m in ['P0', 'B3', 'B1', 'B4', 'B2', 'M2', 'B5']:
        curve[m] = {str(k): RES[(m, 'S1_s7', k)]['mean_rel']
                    for k in KS if (m, 'S1_s7', k) in RES}
    m1_curve = {str(k): RES[('M1', 'S1_s7', k)]['mean_rel'] for k in M1KS}
    b4_curve_full = [curve['B4'][str(k)] for k in KS if str(k) in curve['B4']]
    argmin_k = int(min([int(kk) for kk in curve['B4'].keys()],
                       key=lambda kk: curve['B4'][str(kk)]))
    # v1 判决（层位冻结：k* 为主，KOUT 并报）
    if k1_model_report['m1_kstar']['pass_all3'] or \
            k1_model_report['m2_kstar']['pass_all3']:
        v1_ks = 'interaction_pair_generalizes_at_kstar'
    elif np.mean(b4_k) <= ADD_ERR_GATE:
        v1_ks = 'combo_approx_additive_at_kstar'
    else:
        v1_ks = 'combo_underdetermined_at_kstar'
    verdict = ('g1p2_%s|kstar_k%d|' % (MODEL, KSTAR) + v1_ks +
               '|readout_b4_%.4f' % np.mean(b4_o) +
               '|kstar_b4_%.4f' % np.mean(b4_k))
    GRADE = 'statistical'
    result = dict(
        phase=3152, name=NAME, mode=MODEL,
        created=time.strftime('%Y-%m-%d %H:%M:%S'),
        runtime_s=round(time.time() - T0, 1),
        design_sha=exe_sha, smoke=SMOKE,
        panel_rows=PANEL, n_pairs=NP_, NL=NL, hidden=HID,
        kstar=KSTAR, readout=KOUT, kgrid=KGRID, m1_curve_layers=M1_CURVE,
        v0_material=dict(median_true_margin=med_true,
                         ok=bool(med_true >= V0_MARGIN_MIN)),
        curve=curve, m1_curve=m1_curve, curve_argmin_b4=argmin_k,
        gates=dict(
            M1_kstar=k1_model_report['m1_kstar'],
            M2_kstar=k1_model_report['m2_kstar'],
            M1_readout=k1_model_report['m1_readout'],
            M2_readout=k1_model_report['m2_readout']),
        v1_verdict_kstar=v1_ks, v2=v2,
        v3=dict(n2h1_prediction_confirmed=bool(fruit_ok),
                worst_class_b4=CLASSES[worst_c], per_fold=s2_res),
        v4=dict(mean_b4_rel=float(np.mean(list(s3_tab.values()))),
                max_b4_rel=float(np.max(list(s3_tab.values()))),
                min_b4_rel=float(np.min(list(s3_tab.values())))),
        v5=dict(class_subspace_energy_kstar=e_ks, sv_ratio_kstar=sv_ks,
                class_subspace_energy_readout=e_ko, sv_ratio_readout=sv_ko),
        m1_grid=grid_res, m1_grid_opt_rank=opt_rank,
        m1_overfit_note=('rank5_optimal_at_readout'
                         if opt_rank.get(KOUT) == 5 else
                         'rank%d_optimal_at_readout_k%d' %
                         (opt_rank.get(KOUT), KOUT)),
        a3_worst20=worst,
        k1_model_report=k1_model_report,
        lambdas=LAMBDA_LIVE,
        grade=GRADE,
        verdict=verdict)
    seal_result(result, 'result.json')
    log('DONE mode=%s runtime %.1fs' % (MODEL, time.time() - T0))
    sys.exit(0)

# =================================================================
# 模式 B：glm4k1 —— 零 GPU，读 3151 collect.npz 重算
# =================================================================
if MODEL == 'glm4k1':
    G3151 = os.path.join(RDIR, 'phase3151', 'g1p1_combo_additive_vs_interaction')
    NPZ = os.path.join(G3151, 'collect.npz')
    npz_sha = hashlib.sha256(open(NPZ, 'rb').read()).hexdigest()[:8]
    NL = 40
    _z0 = np.load(NPZ)
    D = int(_z0['H'].shape[3])
    del _z0
    KSTAR = int(round(KSTAR_FRAC * NL))
    KOUT = NL - 1
    GATE_LAYERS = [KSTAR, KOUT]
    KGRID = sorted(set(int(round(f * NL)) for f in GRID_FRACS))
    design = dict(model='glm4-9b (from 3151 collect.npz)', npz=NPZ,
                  npz_sha8=npz_sha, nl=NL, hidden=D, kstar=KSTAR,
                  readout=KOUT, kgrid=KGRID, seeds_s1=SEEDS_S1,
                  frac_s1=FRAC_S1, rank_m1=RANK_M1, als_iters=ALS_ITERS,
                  als_ridge=ALS_RIDGE, add_err_gate=ADD_ERR_GATE,
                  m1_grid_ranks=M1_GRID_RANKS, m1_grid_ridges=M1_GRID_RIDGES,
                  m1_grid_iters=M1_GRID_ITERS, worst_n=WORST_N,
                  anchor_b4_k39_s1_mean3seed=0.389835258324941,
                  panel_rows=PANEL, smoke=SMOKE,
                  pre_reg='MEMO L14847; recompute-only, no new observation')
    exe_sha = freeze_design('g1p2_glm4k1', design)
    log('glm4k1 begin npz_sha8=%s panel=%d' % (npz_sha, PANEL))
    assert PANEL >= PANEL_MIN or SMOKE, ('panel too small', PANEL)
    z = np.load(NPZ)
    H16 = z['H']
    MARG = z['marg']
    assert H16.shape == (NT, 246, NL + 1, D), H16.shape
    log('npz loaded H=%s' % (H16.shape,))

    def Y_at(k):
        return H16[:, :, k, :].reshape(NT * NP_, D).astype(np.float32)

    # ---- B4 重算（S1 三 seed @ k* 与 KOUT；复核 0.3898 锚） ----
    def b4_fit(seed, k):
        train_set, test_set = split_s1(seed)
        tr_rows = rows_of(train_set)
        te_rows = rows_of(test_set)
        Xtr, _, rowvec, cols = phi_main(train_set)
        Y = Y_at(k)
        W = ridge_primal(Xtr, Y[tr_rows], lam=1e-3)
        Xte = np.stack([rowvec(r // NP_, r % NP_) for r in te_rows])
        B4te = Xte @ W
        B4tr = Xtr @ W
        ref = Y[tr_rows].mean(0)
        Dk = float(((Y[tr_rows] - ref) ** 2).sum(1).mean()) + 1e-9
        Yte = Y[te_rows]
        e = ((B4te - Yte) ** 2).sum(1) / Dk
        return e, B4te, B4tr, te_rows, tr_rows, train_set

    b4_k = []
    b4_o = []
    b4_store = {}
    for s in SEEDS_S1:
        e3, B4te3, B4tr3, te3, tr3, ts3 = b4_fit(s, KSTAR)
        e9, B4te9, B4tr9, te9, tr9, ts9 = b4_fit(s, KOUT)
        b4_k.append(float(e3.mean()))
        b4_o.append(float(e9.mean()))
        b4_store[s] = dict(kstar=(e3, B4te3, B4tr3, te3, tr3, ts3),
                           kout=(e9, B4te9, B4tr9, te9, tr9, ts9))
        PAIR_ERR[('B4', 'S1_s%d' % s, KSTAR)] = e3
        PAIR_ERR[('B4', 'S1_s%d' % s, KOUT)] = e9
        log('B4 recompute s%d k*=%.4f kout=%.4f' % (s, b4_k[-1], b4_o[-1]))
    anchor = 0.389835258324941
    drift = abs(np.mean(b4_o) - anchor)
    log('B4@k39 3-seed mean=%.6f anchor=%.6f drift=%.2e' %
        (np.mean(b4_o), anchor, drift))
    assert drift < 1e-4, ('B4 k39 anchor drift', drift)

    # ---- M1@k* 门（S1 三 seed，ALS 正式 iters） ----
    def m1_gate(seed, k, b4_entry):
        e_b4, B4te, B4tr, te_rows, tr_rows, train_set = b4_entry
        tr_set = set(train_set)
        Y = Y_at(k)
        ref = Y[tr_rows].mean(0)
        Dk = float(((Y[tr_rows] - ref) ** 2).sum(1).mean()) + 1e-9
        Yte = Y[te_rows]
        M1te = np.zeros((len(te_rows), D), np.float32)
        for t in range(NT):
            Rg = np.zeros((NE_KEEP, NC, D), np.float32)
            mask = np.zeros((NE_KEEP, NC), bool)
            for pi, (i, c) in enumerate(PAIRS):
                if (i, c) in tr_set:
                    Rg[keep_e.index(i), c] = \
                        H16[t, pi, k, :].astype(np.float32) - \
                        B4tr[tr_rows.index(t * NP_ + pi)]
                    mask[keep_e.index(i), c] = True
            Rhat = als_complete(Rg, mask, RANK_M1, ALS_ITERS, ALS_RIDGE,
                                ALS_SEED + 1000 * seed + k + t)
            if Rhat is None:
                Rhat = np.zeros_like(Rg)
            for j, r in enumerate(te_rows):
                if r // NP_ != t:
                    continue
                i, c = PAIRS[r % NP_]
                M1te[j] = B4te[j] + Rhat[keep_e.index(i), c]
        e_m1 = ((M1te - Yte) ** 2).sum(1) / Dk
        return e_m1

    m1_kstar_store = {}
    for s in SEEDS_S1:
        e_m1 = m1_gate(s, KSTAR, b4_store[s]['kstar'])
        PAIR_ERR[('M1', 'S1_s%d' % s, KSTAR)] = e_m1
        m1_kstar_store[s] = float(e_m1.mean())
        log('M1@k* s%d mean=%.4f (B4=%.4f)' %
            (s, e_m1.mean(), b4_store[s]['kstar'][0].mean()))

    # ---- M1 网格解剖（seed7 @KGRID） ----
    log('M1 grid begin')
    grid_res = {}
    for k in KGRID:
        if k == KSTAR:
            be = b4_store[SEEDS_S1[0]]['kstar']
        elif k == KOUT:
            be = b4_store[SEEDS_S1[0]]['kout']
        else:
            _, B4te, B4tr, te_rows, tr_rows, train_set = b4_fit(
                SEEDS_S1[0], k)
            be = (None, B4te, B4tr, te_rows, tr_rows, train_set)
        for rank in M1_GRID_RANKS:
            for ridge in M1_GRID_RIDGES:
                e_b4, B4te, B4tr, te_rows, tr_rows, train_set = be
                tr_set = set(train_set)
                Y = Y_at(k)
                ref = Y[tr_rows].mean(0)
                Dk = float(((Y[tr_rows] - ref) ** 2).sum(1).mean()) + 1e-9
                Yte = Y[te_rows]
                M1te = np.zeros((len(te_rows), D), np.float32)
                for t in range(NT):
                    Rg = np.zeros((NE_KEEP, NC, D), np.float32)
                    mask = np.zeros((NE_KEEP, NC), bool)
                    for pi, (i, c) in enumerate(PAIRS):
                        if (i, c) in tr_set:
                            Rg[keep_e.index(i), c] = \
                                H16[t, pi, k, :].astype(np.float32) - \
                                B4tr[tr_rows.index(t * NP_ + pi)]
                            mask[keep_e.index(i), c] = True
                    Rhat = als_complete(Rg, mask, rank, M1_GRID_ITERS,
                                        ridge,
                                        ALS_SEED + 1000 * SEEDS_S1[0] + k + t + rank)
                    if Rhat is None:
                        Rhat = np.zeros_like(Rg)
                    for j, r in enumerate(te_rows):
                        if r // NP_ != t:
                            continue
                        i, c = PAIRS[r % NP_]
                        M1te[j] = B4te[j] + Rhat[keep_e.index(i), c]
                e_m1 = float((((M1te - Yte) ** 2).sum(1) / Dk).mean())
                grid_res['k%d_r%d_l%g' % (k, rank, ridge)] = e_m1
                log('grid k=%d rank=%d ridge=%g -> %.4f' % (k, rank, ridge, e_m1))
        if be[0] is not None:
            grid_res['k%d_B4' % k] = float(be[0].mean())
    opt_rank = {}
    for k in KGRID:
        cands = {}
        for r in M1_GRID_RANKS:
            vs = [grid_res['k%d_r%d_l%g' % (k, r, lg)]
                  for lg in M1_GRID_RIDGES]
            cands[r] = min(vs)
        opt_rank[k] = int(min(cands, key=cands.get))
    log('optimal rank per k: %s' % opt_rank)

    # ---- A3 worst20（seed7 KOUT B4） ----
    b4e7 = PAIR_ERR[('B4', 'S1_s%d' % SEEDS_S1[0], KOUT)]
    te7 = b4_store[SEEDS_S1[0]]['kout'][3]
    order = np.argsort(-b4e7)[:WORST_N]
    prompts_g = []
    for t in range(NT):
        for (i, c) in PAIRS:
            prompts_g.append(TPL[t].format(e=ENTS[i], c=CLASSES[c]))
    worst = []
    for li in order:
        row = te7[int(li)]
        i, c = PAIRS[row % NP_]
        worst.append(dict(local_i=int(li), row=int(row), tpl=int(row // NP_),
                          ent=ENTS[i], cls=CLASSES[c],
                          prompt=prompts_g[row], rel_err=float(b4e7[int(li)])))
    # V0 材料（3151 MARG）
    true_pairs = [(t, pi) for t in range(NT) for pi, (i, c) in
                  enumerate(PAIRS) if c == CLS_OF[i]]
    med_true = float(np.median([MARG[t, pi, c] for (t, pi) in true_pairs]))
    b4_k_mean = float(np.mean(b4_k))
    b4_o_mean = float(np.mean(b4_o))
    k1_model_report = dict(
        model='glm4-9b', NL=NL, kstar=KSTAR, readout=KOUT,
        b4_rel_kstar_mean3seed=b4_k_mean,
        b4_rel_kstar_per_seed=[float(x) for x in b4_k],
        b4_rel_readout_mean3seed=b4_o_mean,
        b4_rel_readout_per_seed=[float(x) for x in b4_o],
        m1_kstar=gate_pack('M1', KSTAR), m2_kstar=dict(pass_all3=False,
                                                       margins=None, mdes=None,
                                                       note='M2 not recomputed; 3151 gates show M2 fails all layers (margins positive)'),
        m1_readout=dict(pass_all3=False,
                        note='M1@k39 from 3151 fails (margins +0.77/+1.05/+0.90 positive = worse)'),
        m2_readout=dict(pass_all3=False),
        above_add_gate=bool(b4_k_mean > ADD_ERR_GATE),
        anchor_b4_k39_drift=float(drift),
        m1_kstar_per_seed_mean=[m1_kstar_store[s] for s in SEEDS_S1])
    verdict = ('g1p2_glm4k1|kstar_k%d|m1_%s|b4kstar_%.4f|b4readout_%.4f|'
               'anchor_drift_%.1e' %
               (KSTAR,
                'pass' if k1_model_report['m1_kstar']['pass_all3'] else 'fail',
                b4_k_mean, b4_o_mean, drift))
    result = dict(
        phase=3152, name=NAME, mode='glm4k1',
        created=time.strftime('%Y-%m-%d %H:%M:%S'),
        runtime_s=round(time.time() - T0, 1),
        design_sha=exe_sha, smoke=SMOKE, source_npz=NPZ,
        source_npz_sha8=npz_sha, panel_rows=PANEL, n_pairs=NP_,
        NL=NL, hidden=D, kstar=KSTAR, readout=KOUT, kgrid=KGRID,
        v0_material=dict(median_true_margin=med_true, ok=bool(med_true >= 1.0)),
        m1_grid=grid_res, m1_grid_opt_rank=opt_rank,
        m1_overfit_note=('rank5_optimal_at_readout'
                         if opt_rank.get(KOUT) == 5 else
                         'rank%d_optimal_at_readout_k%d' %
                         (opt_rank.get(KOUT), KOUT)),
        a3_worst20=worst,
        k1_model_report=k1_model_report,
        grade='statistical',
        verdict=verdict)
    seal_result(result, 'result.json')
    log('DONE mode=glm4k1 runtime %.1fs' % (time.time() - T0))
    sys.exit(0)

# =================================================================
# 模式 C：summary —— 零 GPU，K1 三模型判决
# =================================================================
if MODEL == 'summary':
    P4B = os.path.join(RDIR, 'phase3152', NAME, 'qwen3-4b', 'result.json')
    P14B = os.path.join(RDIR, 'phase3152', NAME, 'qwen3-14b', 'result.json')
    PG4 = os.path.join(RDIR, 'phase3152', NAME, 'glm4k1', 'result.json')
    r4b = json.load(open(P4B, encoding='utf-8'))
    r14b = json.load(open(P14B, encoding='utf-8'))
    rg4 = json.load(open(PG4, encoding='utf-8'))
    design = dict(inputs={'qwen3-4b': dict(path=P4B, sha8=r4b.get('res_sha8'),
                                           seal=r4b.get('seal_sha8')),
                          'qwen3-14b': dict(path=P14B, sha8=r14b.get('res_sha8'),
                                            seal=r14b.get('seal_sha8')),
                          'glm4-9b': dict(path=PG4, sha8=rg4.get('res_sha8'),
                                          seal=rg4.get('seal_sha8'))},
                  add_err_gate=ADD_ERR_GATE, kstar_frac=KSTAR_FRAC,
                  decision_layer='kstar (mechanism, depth~7.5%)',
                  pre_reg='MEMO L14847 item (3): K1 gate applies at k*')
    exe_sha = freeze_design('g1p2_summary', design)
    reports = [r4b['k1_model_report'], r14b['k1_model_report'],
               rg4['k1_model_report']]
    per_model = {}
    for rp in reports:
        per_model[rp['model']] = dict(
            NL=rp['NL'], kstar=rp['kstar'], readout=rp['readout'],
            b4_rel_kstar=rp['b4_rel_kstar_mean3seed'],
            b4_rel_readout=rp['b4_rel_readout_mean3seed'],
            m1_kstar_pass_all3=rp['m1_kstar']['pass_all3'],
            m1_kstar_margins=rp['m1_kstar']['margins'],
            m2_kstar_pass_all3=rp['m2_kstar']['pass_all3'],
            above_add_gate=rp['above_add_gate'])
    all_above = all(pm['above_add_gate'] for pm in per_model.values())
    m1_all = all(pm['m1_kstar_pass_all3'] for pm in per_model.values())
    m1_any = any(pm['m1_kstar_pass_all3'] for pm in per_model.values())
    m2_any = any(pm['m2_kstar_pass_all3'] for pm in per_model.values())
    if not all_above:
        k1_verdict = ('k1_not_triggered_b4_additive_at_kstar_'
                      'operator_line_kept')
        trigger = False
    elif m1_all:
        k1_verdict = ('k1_not_triggered_interaction_generalizes_3model_'
                      'operator_line_kept')
        trigger = False
    elif m1_any or m2_any:
        k1_verdict = ('k1_partial_interaction_repairs_subset_'
                      'operator_line_conditional')
        trigger = False
    else:
        k1_verdict = 'k1_TRIGGERED_operator_algebra_dropped'
        trigger = True
    depth_note = dict()
    for mn, pm in per_model.items():
        depth_note[mn] = dict(kstar_depth_frac=round(pm['kstar'] / pm['NL'], 4),
                              readout_depth_frac=round(
                                  pm['readout'] / pm['NL'], 4))
    worst20_join = dict(
        glm4_9b=[w['prompt'] for w in rg4['a3_worst20'][:10]],
        qwen3_4b=[w['prompt'] for w in r4b['a3_worst20'][:10]],
        qwen3_14b=[w['prompt'] for w in r14b['a3_worst20'][:10]])
    verdict = ('g1p2_summary|' + k1_verdict +
               '|models_above_gate=%d/3|m1_pass=%d/3' %
               (sum(1 for pm in per_model.values() if pm['above_add_gate']),
                sum(1 for pm in per_model.values()
                    if pm['m1_kstar_pass_all3'])))
    result = dict(
        phase=3152, name=NAME, mode='summary',
        created=time.strftime('%Y-%m-%d %H:%M:%S'),
        runtime_s=round(time.time() - T0, 1),
        design_sha=exe_sha,
        k1_3model=per_model, k1_depth_note=depth_note,
        k1_verdict=k1_verdict, k1_triggered=bool(trigger),
        b4_curve_argmin=dict(qwen3_4b=r4b.get('curve_argmin_b4'),
                             qwen3_14b=r14b.get('curve_argmin_b4')),
        m1_grid_opt_rank=dict(qwen3_4b=r4b.get('m1_grid_opt_rank'),
                              qwen3_14b=r14b.get('m1_grid_opt_rank'),
                              glm4_9b=rg4.get('m1_grid_opt_rank')),
        a3_worst20_sample=worst20_join,
        inputs_used=design['inputs'],
        grade='statistical',
        verdict=verdict)
    seal_result(result, 'result_summary.json')
    log('DONE mode=summary runtime %.1fs' % (time.time() - T0))
    sys.exit(0)

log('unknown mode %s' % MODEL)
sys.exit(2)
