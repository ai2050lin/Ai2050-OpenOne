# Phase 3151 (G1-P1): 组合可加性 vs 交互项 —— one-shot 判决（模型 1/3: glm4-9b）
# 预注册：TESTPLAN v1 §4.1 原案 + UNIFIED_REVIEW_ADJUDICATION_v1 修正 A1-A3（观测前冻结）
# 面板：41 实体 x 6 类 x 3 模板 = 738 行（>= 672 硬约束）
# 切分：S1 未见组合(主, 3 seeds) / S2 留一类(6 folds) / S3 留一实例(41 folds)
# 模型：P0 均值 / B3 模板主效应 / B1 端点可加 / B4 全加性 / B2 词袋(嵌入和) /
#        M1 网格ALS rank5(仅S1) / M2 嵌入中介双线性(S1S2S3) / B5 ELM(64) 容量对照(仅S1)
# 门层：k=3 (n2h1b GLM4 承诺层) 与 k=39 (末前层)；门=配对胜 B4 >= 2xMDE 且 3 seed 同号
import os, sys, json, time, hashlib
import numpy as np

T0 = time.time()
SMOKE = os.environ.get('P3151_SMOKE') == '1'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
MDIR = os.path.join(ROOT, 'models', 'hf', 'glm4-9b-chat-hf')
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result',
                    'rdc_query_construction_20260913')
NAME = 'g1p1_combo_additive_vs_interaction'
OUT = os.path.join(RDIR, 'phase3151', NAME)
if SMOKE:
    OUT = os.path.join(OUT, 'smoke')
os.makedirs(OUT, exist_ok=True)
LOGP = os.path.join(OUT, 'run_log.txt')

def log(s):
    ln = '[%7.1f] %s' % (time.time() - T0, s)
    with open(LOGP, 'a', encoding='utf-8') as f:
        f.write(ln + '\n')
    print(ln, flush=True)

# ---------------- 冻结设计（观测前） ----------------
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
GATE_LAYERS = [3, 39]
CURVE_LAYERS = list(range(41))
M1_CURVE_LAYERS = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39]
SEEDS_S1 = [7, 8, 9]
FRAC_S1 = 0.2
RANK_M1 = 5
ALS_ITERS = 20 if SMOKE else 100
ALS_RIDGE = 1e-2
M2_M = 32 if SMOKE else 128
B5_UNITS = 64
LAMBDA_GRID = [1e-3, 1e-2, 1e-1, 1.0]
V0_MARGIN_MIN = 1.0
ADD_ERR_GATE = 0.05
ALS_SEED = 7

ENTS = [e for cl in CLASSES for e in ENT[cl]]
CLS_OF = [CLASSES.index(cl) for cl in CLASSES for e in ENT[cl]]
NE = len(ENTS)
NC = len(CLASSES)
PAIRS = [(i, c) for i in range(NE) for c in range(NC)]
NP_ = len(PAIRS)                       # 246
NT = len(TPL)                          # 3
PANEL = NT * NP_                       # 738

DESIGN = dict(clASSES=CLASSES, ents=ENTS, tpl=TPL, gate_layers=GATE_LAYERS,
              seeds_s1=SEEDS_S1, frac_s1=FRAC_S1, rank_m1=RANK_M1,
              als_iters=ALS_ITERS, als_ridge=ALS_RIDGE, m2_m=M2_M,
              b5_units=B5_UNITS, lambda_grid=LAMBDA_GRID,
              v0_margin_min=V0_MARGIN_MIN, add_err_gate=ADD_ERR_GATE,
              amendments=['A1 三切分', 'A2 B5 容量对照', 'A3 worst-20 失败样例'],
              panel_rows=PANEL, smoke=SMOKE)

# ---- execution.json 幂等冻结 ----
exe_p = os.path.join(OUT, 'execution.json')
eblob = json.dumps(DESIGN, ensure_ascii=False, sort_keys=True,
                   indent=1).encode('utf-8')
exe_sha = hashlib.sha256(eblob).hexdigest()
if os.path.exists(exe_p):
    prev = json.load(open(exe_p, encoding='utf-8'))
    assert prev['design_sha'] == exe_sha, 'DESIGN DRIFT'
    log('execution.json match (sha %s)' % exe_sha[:8])
else:
    json.dump({'phase': 3151, 'name': NAME, 'design_sha': exe_sha,
               'design': DESIGN,
               'frozen_before': 'any model observation',
               'created': time.strftime('%Y-%m-%d %H:%M:%S')},
              open(exe_p, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    log('execution.json FROZEN (sha %s)' % exe_sha[:8])

PANEL_MIN = 672
if SMOKE:
    keep_e = []
    cnt = {}
    for i, cl in enumerate(CLS_OF):
        if cnt.get(cl, 0) < 2:
            keep_e.append(i)
            cnt[cl] = cnt.get(cl, 0) + 1
    PAIRS = [(i, c) for (i, c) in PAIRS if i in set(keep_e)]
    NP_ = len(PAIRS)
    PANEL = NT * NP_
    NE_KEEP = len(keep_e)
else:
    keep_e = list(range(NE))
    NE_KEEP = NE
assert PANEL >= PANEL_MIN or SMOKE, ('panel too small', PANEL)

# ---------------- 模型与采集 ----------------
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
log('model load begin')
model = AutoModelForCausalLM.from_pretrained(
    MDIR, dtype=torch.bfloat16, trust_remote_code=True).to('cuda').eval()
log('model loaded: %s' % type(model).__name__)

HID = model.config.hidden_size
NL = model.config.num_hidden_layers          # 40
NH = NL + 1                                  # 41 hidden states
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

cache = os.path.join(OUT, 'collect_smoke.npz' if SMOKE else 'collect.npz')
if os.path.exists(cache):
    z = np.load(cache)
    H16 = z['H']
    MARG = z['marg']
    log('collect cache hit %s H=%s' % (os.path.basename(cache), H16.shape))
else:
    emb_w = model.get_input_embeddings().weight
    H16 = np.zeros((NT, NP_, NH, D), np.float16)
    MARG = np.zeros((NT, NP_, NC), np.float32)
    with torch.no_grad():
        for t in range(NT):
            for pi, (i, c) in enumerate(PAIRS):
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

# 行布局: row = t*NP_ + pi
def Y_at(k):
    return H16[:, :, k, :].reshape(NT * NP_, D).astype(np.float32)

def rows_of(pair_set):
    return [t * NP_ + pi for t in range(NT)
            for pi, p in enumerate(PAIRS) if p in pair_set]

ALLP = set(PAIRS)

# ---------------- 切分 ----------------
def split_s1(seed):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(NP_)
    n_test = int(round(FRAC_S1 * NP_))
    test = set([PAIRS[j] for j in idx[:n_test]])
    train = ALLP - test
    return train, test

S2 = []
for cstar in range(NC):
    test = set([p for p in PAIRS if p[1] == cstar])
    S2.append((cstar, ALLP - test, test))
S3 = []
for e in keep_e:
    test = set([p for p in PAIRS if p[0] == e])
    S3.append((e, ALLP - test, test))

# ---------------- 特征 ----------------
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

EMB = None
def get_emb():
    global EMB
    if EMB is None:
        w = model.get_input_embeddings().weight
        need = sorted(set(E_TOK + CLS_TOK +
                          sorted({t for tk in TOKIDS for t in tk})))
        idx = torch.tensor(need, device='cuda')
        EMB = {'need': need,
               'rows': w[idx].float().detach().cpu().numpy()}
    return EMB

def bag_feat(row):
    tk = TOKIDS[row]
    em = get_emb()
    pos = {t: j for j, t in enumerate(em['need'])}
    v = np.zeros(D, np.float32)
    for t in tk:
        v += em['rows'][pos[t]]
    return v / np.sqrt(len(tk))

def embpair_feat(t, pi):
    i, c = PAIRS[pi]
    em = get_emb()
    pos = {t_: j for j, t_ in enumerate(em['need'])}
    er = em['rows'][pos[E_TOK[i]]]
    gr = em['rows'][pos[CLS_TOK[c]]]
    return er, gr

rng_f = np.random.RandomState(ALS_SEED)
R1 = rng_f.randn(M2_M, D).astype(np.float32) / np.sqrt(D)
R2 = rng_f.randn(M2_M, D).astype(np.float32) / np.sqrt(D)

def m2_extra(rows):
    F = np.zeros((len(rows), M2_M), np.float32)
    for j, r in enumerate(rows):
        t, pi = r // NP_, r % NP_
        er, gr = embpair_feat(t, pi)
        F[j] = (R1 @ er) * (R2 @ gr)
    return F

# ---------------- 拟合器 ----------------
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

# ---------------- 评估框架 ----------------
RES = {}          # (model, split, k) -> dict(mean_rel, mean_cos)
PAIR_ERR = {}     # (model, split, k) -> per-pair rel errs (S1 gate only)

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

def fit_and_eval(split_id, train_set, test_set, ks, models,
                 lam2=None, lamB2=None, keep_pair_err=False):
    """returns per-k dict of per-pair errs keyed by model (gate layers)"""
    out_errs = {}
    tr_rows = rows_of(train_set)
    te_rows = rows_of(test_set)
    tr_set = set(train_set)
    for k in ks:
        Y = Y_at(k)
        Ytr = Y[tr_rows]
        ref = Ytr.mean(0)
        Dk = float(((Ytr - ref) ** 2).sum(1).mean()) + 1e-9
        preds = {}
        # P0 / B3
        preds['P0'] = [ref] * len(te_rows)
        tpl_mean = {}
        for t in range(NT):
            rr = [r for r in tr_rows if r // NP_ == t]
            tpl_mean[t] = Y[rr].mean(0) if rr else ref
        preds['B3'] = [tpl_mean[r // NP_] for r in te_rows]
        # B4 全加性
        Xtr, tr_rows2, rowvec, cols = phi_main(train_set)
        W = ridge_primal(Xtr, Y[tr_rows], lam=1e-3)
        Xte = np.stack([rowvec(r // NP_, r % NP_) for r in te_rows])
        preds['B4'] = list(Xte @ W)
        # B1 端点可加: h(e,c) = h(e,c1) + mu_c - mu_c1  (c1=水果, 同模板)
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
        # B2 词袋（嵌入和，dual；lamB2 冻结复用）
        if lamB2 is None:
            lamB2_ = LAMBDA_GRID[1]
        else:
            lamB2_ = lamB2
        Ftr = np.stack([bag_feat(r) for r in tr_rows])
        Fte = np.stack([bag_feat(r) for r in te_rows])
        al, _ = ridge_dual(Ftr, Ytr, lamB2_)
        preds['B2'] = list(Fte @ Ftr.T @ al)
        # M2 嵌入中介双线性
        Ftr2 = np.concatenate([Xtr, m2_extra(tr_rows)], 1)
        Fte2 = np.concatenate([Xte, m2_extra(te_rows)], 1)
        lam2_ = lam2 if lam2 is not None else LAMBDA_GRID[0]
        al2, _ = ridge_dual(Ftr2, Ytr, lam2_)
        preds['M2'] = list(Fte2 @ Ftr2.T @ al2)
        # B5 ELM
        if 'B5' in models:
            W1, b1, W2 = elm_fit(Xtr, Y[tr_rows])
            preds['B5'] = list(elm_pred(W1, b1, W2, Xte))
        for m in ['P0', 'B3', 'B1', 'B4', 'B2', 'M2'] + (
                ['B5'] if 'B5' in models else []):
            if preds[m] is None:
                continue
            keep = keep_pair_err and (k in GATE_LAYERS)
            e = evaluate(m, split_id, k, preds[m], te_rows, Y, ref, Dk,
                         keep_pair_err=keep)
            if keep:
                out_errs[m] = PAIR_ERR[(m, split_id, k)]
        log('k=%d %s done (B4=%.4f M2=%.4f)' %
            (k, split_id, RES[('B4', split_id, k)]['mean_rel'],
             RES[('M2', split_id, k)]['mean_rel']))
    return out_errs

# ---------------- S1 主判决（含 M1） ----------------
def fit_s1_with_m1(seed, ks):
    """S1: B-family + M2 + M1(ALS) at ks"""
    train_set, test_set = split_s1(seed)
    tr_rows = rows_of(train_set)
    te_rows = rows_of(test_set)
    tr_set = set(train_set)
    out = {}
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
        # M1: per-template ALS on B4 residual
        M1te = np.zeros((len(te_rows), D), np.float32)
        for t in range(NT):
            Rg = np.zeros((NE_KEEP, NC, D), np.float32)
            mask = np.zeros((NE_KEEP, NC), bool)
            for pi, (i, c) in enumerate(PAIRS):
                if (i, c) in tr_set:
                    Rg[keep_e.index(i), c] = \
                        H16[t, pi, k, :].astype(np.float32) - B4tr[tr_rows.index(t * NP_ + pi)]
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
        # M2
        Ftr2 = np.concatenate([Xtr, m2_extra(tr_rows)], 1)
        Fte2 = np.concatenate([Xte, m2_extra(te_rows)], 1)
        al2, _ = ridge_dual(Ftr2, Ytr, LAMBDA_LIVE['M2'])
        M2te = Fte2 @ Ftr2.T @ al2
        # store
        Yte = Y[te_rows]
        for m, P in [('B4', B4te), ('M1', M1te), ('M2', M2te)]:
            e = ((P - Yte) ** 2).sum(1) / Dk
            cs = float(np.mean([float((a * b).sum() /
                                      (np.linalg.norm(a) *
                                       np.linalg.norm(b) + 1e-9))
                                for a, b in zip(P, Yte)]))
            RES[(m, 'S1_s%d' % seed, k)] = dict(mean_rel=float(e.mean()),
                                                mean_cos=cs, n=len(te_rows))
            if k in GATE_LAYERS:
                PAIR_ERR[(m, 'S1_s%d' % seed, k)] = e
        log('S1 s%d k=%d B4=%.4f M1=%.4f M2=%.4f' %
            (seed, k, RES[('B4', 'S1_s%d' % seed, k)]['mean_rel'],
             RES[('M1', 'S1_s%d' % seed, k)]['mean_rel'],
             RES[('M2', 'S1_s%d' % seed, k)]['mean_rel']))
    return out

# λ 选择（冻结：S1 seed7 k=39 训练内 5-fold CV）
LAMBDA_LIVE = {'M2': LAMBDA_GRID[0], 'B2': LAMBDA_GRID[1]}
def select_lambdas():
    train_set, test_set = split_s1(SEEDS_S1[0])
    tr_rows = rows_of(train_set)
    k = 39
    Y = Y_at(k)
    Ytr = Y[tr_rows]
    Xtr, _, _, _ = phi_main(train_set)
    rng = np.random.RandomState(0)
    order = rng.permutation(len(tr_rows))
    folds = np.array_split(order, 5)
    for name, Ftr in [('M2', np.concatenate([Xtr, m2_extra(tr_rows)], 1)),
                      ('B2', np.stack([bag_feat(r) for r in tr_rows]))]:
        best, best_v = None, 1e18
        for lam in LAMBDA_GRID:
            vs = []
            for f in range(5):
                va = set(folds[f].tolist())
                trn = [r for j, r in enumerate(tr_rows) if j not in va]
                van = [tr_rows[j] for j in folds[f]]
                sub = [j for j in range(len(tr_rows))
                        if j not in va]
                Fsub = Ftr[sub]
                al, _ = ridge_dual(Fsub, Y[trn], lam)
                Pv = Ftr[list(va)] @ Fsub.T @ al
                v = float(((Pv - Y[van]) ** 2).mean())
                vs.append(v)
            if np.mean(vs) < best_v:
                best_v, best = float(np.mean(vs)), lam
        LAMBDA_LIVE[name] = best
        log('lambda %s = %g (cv %.5f)' % (name, best, best_v))

# ---------------- 主流程 ----------------
log('panel rows = %d (min %d), smoke=%s' % (PANEL, PANEL_MIN, SMOKE))
select_lambdas()

if SMOKE:
    KS = GATE_LAYERS + [20]
    M1KS = KS
else:
    KS = CURVE_LAYERS
    M1KS = M1_CURVE_LAYERS

# S1 seed7 曲线（B 族 + M2 全层；M1/B5 曲线层）
fit_and_eval('S1_s7', *split_s1(SEEDS_S1[0]), ks=[k for k in KS],
             models=['B5'], lam2=LAMBDA_LIVE['M2'],
             lamB2=LAMBDA_LIVE['B2'], keep_pair_err=True)
# M1 曲线
for k in M1KS:
    fit_s1_with_m1(SEEDS_S1[0], [k])
# S1 seeds 8/9 门层
for sd in SEEDS_S1[1:]:
    fit_and_eval('S1_s%d' % sd, *split_s1(sd), ks=GATE_LAYERS,
                 models=['B5'], lam2=LAMBDA_LIVE['M2'],
                 lamB2=LAMBDA_LIVE['B2'], keep_pair_err=True)
    fit_s1_with_m1(sd, GATE_LAYERS)
# S2 留一类（6 folds；M2 可用）
s2_tab = {}
for cstar, tr, te in S2:
    fit_and_eval('S2_c%d' % cstar, tr, te, ks=GATE_LAYERS,
                 models=[], lam2=LAMBDA_LIVE['M2'],
                 lamB2=LAMBDA_LIVE['B2'], keep_pair_err=True)
    for k in GATE_LAYERS:
        s2_tab[(cstar, k)] = (RES[('B4', 'S2_c%d' % cstar, k)]['mean_rel'],
                              RES[('M2', 'S2_c%d' % cstar, k)]['mean_rel'])
# S3 留一实例（B 族 + M2）
s3_tab = {}
for e, tr, te in S3:
    fit_and_eval('S3_e%d' % e, tr, te, ks=[39],
                 models=[], lam2=LAMBDA_LIVE['M2'],
                 lamB2=LAMBDA_LIVE['B2'], keep_pair_err=True)
    s3_tab[e] = RES[('B4', 'S3_e%d' % e, 39)]['mean_rel']

# ---------------- 判决 ----------------
def paired(mA, mB, split_id, k):
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

gates = {}
for k in GATE_LAYERS:
    for cand in ['M1', 'M2']:
        gs = [paired(cand, 'B4', 'S1_s%d' % s, k) for s in SEEDS_S1]
        gates['%s_k%d' % (cand, k)] = gs
v1a_m1 = any(all(g and g['pass_gate'] for g in gates['M1_k%d' % k])
             for k in GATE_LAYERS)
v1a_m2 = any(all(g and g['pass_gate'] for g in gates['M2_k%d' % k])
             for k in GATE_LAYERS)
v1a = v1a_m1 or v1a_m2
b4_s1_k39 = float(np.mean([RES[('B4', 'S1_s%d' % s, 39)]['mean_rel']
                           for s in SEEDS_S1]))
if v1a:
    v1 = 'interaction_pair_generalizes'
elif b4_s1_k39 < ADD_ERR_GATE:
    v1 = 'combo_approx_additive'
else:
    v1 = 'combo_underdetermined'
# V2 容量对照
v2 = None
if v1a:
    cand = 'M1' if v1a_m1 else 'M2'
    g2 = [paired(cand, 'B5', 'S1_s%d' % s, 39) for s in SEEDS_S1]
    ok2 = all(g and g['pass_gate'] for g in g2)
    v2 = 'gain_above_capacity' if ok2 else 'gain_is_capacity'
# V3 留一类（预注册预言：水果折 M2 不优于 B4）
s2_res = {}
for (cstar, k), (b4, m2) in sorted(s2_tab.items()):
    s2_res['c%d_k%d' % (cstar, k)] = dict(b4=b4, m2=m2,
                                          margin_m2_minus_b4=m2 - b4)
fruit_ok = all(s2_tab[(0, k)][1] - s2_tab[(0, k)][0] >= -0.005
               for k in GATE_LAYERS)
worst_c = max(range(NC), key=lambda c: np.mean(
    [s2_tab[(c, 39)][0]]))
v3 = dict(n2h1_prediction_confirmed=bool(fruit_ok),
          worst_class_b4=CLASSES[worst_c], per_fold=s2_res)
# V4 留一实例
v4 = dict(mean_b4_rel=float(np.mean(list(s3_tab.values()))),
          max_b4_rel=float(np.max(list(s3_tab.values()))),
          min_b4_rel=float(np.min(list(s3_tab.values()))))
# V0 行为材料有效性
true_pairs = [(t, pi) for t in range(NT) for pi, (i, c) in
              enumerate(PAIRS) if c == CLS_OF[i]]
med_true = float(np.median([MARG[t, pi, c] for (t, pi) in true_pairs]))
v0_ok = med_true >= V0_MARGIN_MIN
# V5 描述性：k=3 残差类子空间能量份额（seed7）
def v5_energy(k):
    train_set, test_set = split_s1(SEEDS_S1[0])
    tr_rows = rows_of(train_set)
    Y = Y_at(k)
    Xtr, _, rowvec, cols = phi_main(train_set)
    W = ridge_primal(Xtr, Y[tr_rows], lam=1e-3)
    Rg = np.zeros((NE_KEEP, NC, D), np.float32)
    for t in range(NT):
        grid = H16[t, :, k, :].astype(np.float32)
        for pi, (i, c) in enumerate(PAIRS):
            pv = rowvec(t, pi) @ W
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
e3, s3v = v5_energy(3)
e39, s39v = v5_energy(39)
v5 = dict(class_subspace_energy_k3=e3, sv_ratio_k3=s3v,
          class_subspace_energy_k39=e39, sv_ratio_k39=s39v)

# ---------------- result ----------------
verdict = 'a_3150_ok|' + v1
if v2:
    verdict += '|' + v2
verdict += '|v3_%s' % ('confirmed' if fruit_ok else 'missed')
GRADE = 'statistical'
result = dict(
    phase=3151, name=NAME, created=time.strftime('%Y-%m-%d %H:%M:%S'),
    runtime_s=round(time.time() - T0, 1),
    design_sha=exe_sha, smoke=SMOKE,
    panel_rows=PANEL, panel_min=PANEL_MIN, n_pairs=NP_,
    v0_material=dict(median_true_margin=med_true, ok=bool(v0_ok)),
    v1=dict(verdict=v1, b4_s1_k39_rel=b4_s1_k39,
            gates={kk: gg for kk, gg in gates.items()}),
    v2=v2, v3=v3, v4=v4, v5=v5,
    lambdas=LAMBDA_LIVE,
    k1_status='model_1_of_3_no_trigger_alone',
    grade=GRADE,
    verdict=verdict)
blob = json.dumps(result, ensure_ascii=False, indent=1,
                  sort_keys=True).encode('utf-8')
res_sha8 = hashlib.sha256(blob).hexdigest()[:8]
result['res_sha8'] = res_sha8
result['verdict'] = verdict + '|sha8_' + res_sha8
rp = os.path.join(OUT, 'result.json')
json.dump(result, open(rp, 'w', encoding='utf-8'), ensure_ascii=False,
          indent=1)
seal = hashlib.sha256(open(rp, 'rb').read()).hexdigest()[:8]
result['seal_sha8'] = seal
json.dump(result, open(rp, 'w', encoding='utf-8'), ensure_ascii=False,
          indent=1)
log('RESULT res_sha8=%s seal=%s verdict=%s' % (res_sha8, seal, verdict))
log('DONE runtime %.1fs' % (time.time() - T0))
