# p3151_v2fix.py — 独立复算 k=3 门层判决 + V2 容量对照（正确层位）
# 背景：主跑 V2 固定在 k=39（M1 于中后层过拟合，层位选错）。
# 本脚本独立重算 S1 三 seed 在 k=3 的：B4/M1/M2/B5 逐对误差 →
#   ①门复核 M1-B4、M2-B4（对照主跑 gates，内部复现检验）
#   ②容量对照 M1-B5、M2-B5（配对 margin >= 2xMDE）
# 结果追加到 result.json 的 'v2_k3' 键（append 纪律，rev-3151a）。
import os, json, time, hashlib
import numpy as np

T0 = time.time()
ROOT = r'D:\AI2050\Ai2050-OpenOne'
MDIR = os.path.join(ROOT, 'models', 'hf', 'glm4-9b-chat-hf')
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result',
                    'rdc_query_construction_20260913')
NAME = 'g1p1_combo_additive_vs_interaction'
OUT = os.path.join(RDIR, 'phase3151', NAME)
RP = os.path.join(OUT, 'result.json')
LOGP = os.path.join(OUT, 'v2fix_log.txt')

def log(s):
    with open(LOGP, 'a', encoding='utf-8') as f:
        f.write('[%7.1f] %s\n' % (time.time() - T0, s))
    print('[%7.1f] %s' % (time.time() - T0, s), flush=True)

# ---- 设计常量（与主跑冻结一致） ----
CLASSES = ['水果', '动物', '交通工具', '家具', '金属', '颜色']
ENT = {
    '水果': ['苹果', '香蕉', '梨', '西瓜', '葡萄', '草莓', '芒果', '柠檬'],
    '动物': ['狗', '猫', '老虎', '大象', '兔子', '猴子', '马', '牛'],
    '交通工具': ['汽车', '火车', '飞机', '摩托车', '卡车', '地铁'],
    '家具': ['桌子', '椅子', '床', '沙发', '地毯', '窗帘'],
    '金属': ['铁', '铜', '铝', '金', '银', '锌', '铅'],
    '颜色': ['红', '蓝', '绿', '黄', '黑', '白'],
}
TPL = {0: '{e}是一种{c}。', 1: '{e}属于{c}这一类。',
       2: '{e}，一种常见的{c}。'}
SEEDS_S1 = [7, 8, 9]
FRAC_S1 = 0.2
RANK_M1 = 5
ALS_ITERS = 100
ALS_RIDGE = 1e-2
ALS_SEED = 7
B5_UNITS = 64
K = 3

ENTS = [e for cl in CLASSES for e in ENT[cl]]
CLS_OF = [CLASSES.index(cl) for cl in CLASSES for e in ENT[cl]]
NE = len(ENTS)
NC = len(CLASSES)
PAIRS = [(i, c) for i in range(NE) for c in range(NC)]
NP_ = len(PAIRS)
NT = len(TPL)
ALLP = set(PAIRS)

z = np.load(os.path.join(OUT, 'collect.npz'))
H16 = z['H']
assert H16.shape[0] == NT and H16.shape[1] == NP_

def split_s1(seed):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(NP_)
    n_test = int(round(FRAC_S1 * NP_))
    test = set([PAIRS[j] for j in idx[:n_test]])
    return ALLP - test, test

def rows_of(pair_set):
    return [t * NP_ + pi for t in range(NT)
            for pi, p in enumerate(PAIRS) if p in pair_set]

def Y_at(k):
    return H16[:, :, k, :].reshape(NT * NP_, D).astype(np.float32)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MDIR, dtype=torch.bfloat16, trust_remote_code=True).to('cuda').eval()
D = model.config.hidden_size

def ids_of(t):
    return tok(t, add_special_tokens=False)['input_ids']
E_TOK = [ids_of(e)[0] for e in ENTS]
CLS_TOK = [ids_of(c)[0] for c in CLASSES]
w = model.get_input_embeddings().weight
need = sorted(set(E_TOK + CLS_TOK))
pos = {t: j for j, t in enumerate(need)}
ROWS = w[torch.tensor(need, device='cuda')].float().detach().cpu().numpy()
rng_f = np.random.RandomState(ALS_SEED)
R1 = rng_f.randn(128, D).astype(np.float32) / np.sqrt(D)
R2 = rng_f.randn(128, D).astype(np.float32) / np.sqrt(D)

def als_complete(R, mask, rank, iters, ridge, seed):
    E, C, Dm = R.shape
    rng = np.random.RandomState(seed)
    mu = (R * mask[..., None]).sum((0, 1)) / max(1, mask.sum())
    R0 = (R - mu) * mask[..., None]
    V = (rng.randn(C, rank, Dm) * 0.1).astype(np.float32)
    rid = np.arange(rank)
    for it in range(iters):
        A = np.einsum('ec,crd,csd->ersd', mask, V, V).transpose(3, 0, 1, 2)
        A[..., rid, rid] += ridge
        b = np.einsum('ec,ecd,crd->erd', mask, R0, V)
        U = np.linalg.solve(
            A, b.transpose(2, 0, 1)[..., None])[..., 0].transpose(1, 2, 0)
        if not np.isfinite(U).all():
            return None
        A = np.einsum('ec,erd,esd->crsd', mask, U, U).transpose(3, 0, 1, 2)
        A[..., rid, rid] += ridge
        b = np.einsum('ec,ecd,erd->crd', mask, R0, U)
        V = np.linalg.solve(
            A, b.transpose(2, 0, 1)[..., None])[..., 0].transpose(1, 2, 0)
        if not np.isfinite(V).all():
            return None
    return (np.einsum('erd,crd->ecd', U, V) + mu).astype(np.float32)

def elm_fit(Xtr, Ytr, seed=ALS_SEED):
    rng = np.random.RandomState(seed)
    W1 = rng.randn(Xtr.shape[1], B5_UNITS).astype(np.float32) / np.sqrt(Xtr.shape[1])
    b1 = rng.randn(B5_UNITS).astype(np.float32) * 0.1
    F = np.maximum(Xtr @ W1 + b1, 0)
    F = np.concatenate([F, np.ones((len(F), 1), np.float32)], 1)
    A = F.T @ F + 1e-3 * np.eye(F.shape[1], dtype=np.float32)
    return W1, b1, np.linalg.solve(A, F.T @ Ytr)

def elm_pred(W1, b1, W2, X):
    F = np.maximum(X @ W1 + b1, 0)
    F = np.concatenate([F, np.ones((len(F), 1), np.float32)], 1)
    return F @ W2

COLS = NE + NC + NT + 1
def rowvec(t, pi):
    i, c = PAIRS[pi]
    v = np.zeros(COLS, np.float32)
    v[i] = 1.0
    v[NE + c] = 1.0
    v[NE + NC + t] = 1.0
    v[-1] = 1.0
    return v

def m2_extra(rows):
    F = np.zeros((len(rows), 128), np.float32)
    for j, r in enumerate(rows):
        t, pi = r // NP_, r % NP_
        i, c = PAIRS[pi]
        er = ROWS[pos[E_TOK[i]]]
        gr = ROWS[pos[CLS_TOK[c]]]
        F[j] = (R1 @ er) * (R2 @ gr)
    return F

def dual_pred(Ftr, Ytr, lam, Fte):
    Kmat = Ftr @ Ftr.T
    n = Ftr.shape[0]
    al = np.linalg.solve(Kmat + lam * n * np.eye(n, dtype=np.float32), Ytr)
    return Fte @ Ftr.T @ al

Y = Y_at(K)
ERR = {}
for seed in SEEDS_S1:
    train_set, test_set = split_s1(seed)
    tr_rows = rows_of(train_set)
    te_rows = rows_of(test_set)
    tr_set = set(train_set)
    Ytr = Y[tr_rows]
    ref = Ytr.mean(0)
    Dk = float(((Ytr - ref) ** 2).sum(1).mean()) + 1e-9
    Xtr = np.stack([rowvec(r // NP_, r % NP_) for r in tr_rows])
    Xte = np.stack([rowvec(r // NP_, r % NP_) for r in te_rows])
    A = Xtr.T @ Xtr + 1e-3 * np.eye(COLS, dtype=np.float32)
    W = np.linalg.solve(A, Xtr.T @ Ytr)
    B4te = Xte @ W
    B4tr = Xtr @ W
    # M1
    M1te = np.zeros((len(te_rows), D), np.float32)
    for t in range(NT):
        Rg = np.zeros((NE, NC, D), np.float32)
        mask = np.zeros((NE, NC), bool)
        for pi, (i, c) in enumerate(PAIRS):
            if (i, c) in tr_set:
                Rg[i, c] = H16[t, pi, K, :].astype(np.float32) - \
                    B4tr[tr_rows.index(t * NP_ + pi)]
                mask[i, c] = True
        Rhat = als_complete(Rg, mask, RANK_M1, ALS_ITERS, ALS_RIDGE,
                            ALS_SEED + 1000 * seed + K + t)
        if Rhat is None:
            Rhat = np.zeros_like(Rg)
        for j, r in enumerate(te_rows):
            if r // NP_ != t:
                continue
            i, c = PAIRS[r % NP_]
            M1te[j] = B4te[j] + Rhat[i, c]
    # M2 (lam=0.1 冻结自主跑 CV)
    Ftr2 = np.concatenate([Xtr, m2_extra(tr_rows)], 1)
    Fte2 = np.concatenate([Xte, m2_extra(te_rows)], 1)
    M2te = dual_pred(Ftr2, Ytr, 0.1, Fte2)
    # B5
    W1, b1, W2 = elm_fit(Xtr, Ytr)
    B5te = elm_pred(W1, b1, W2, Xte)
    Yte = Y[te_rows]
    for m, P in [('B4', B4te), ('M1', M1te), ('M2', M2te), ('B5', B5te)]:
        ERR[(m, seed)] = ((P - Yte) ** 2).sum(1) / Dk
    log('seed %d done' % seed)

def paired(a, b):
    d = ERR[(a, 7)] - ERR[(b, 7)]
    res = {}
    for s in SEEDS_S1:
        dd = ERR[(a, s)] - ERR[(b, s)]
        n = len(dd)
        m = float(dd.mean())
        sd = float(dd.std(ddof=1))
        res['s%d' % s] = dict(margin=m, mde=float(1.96 * sd / np.sqrt(n)))
    all_pass = all(v['margin'] <= -2 * v['mde']
                   for v in res.values())
    return dict(per_seed=res, pass_gate=bool(all_pass))

v2k3 = dict(k=K, note='rev-3151a: V2 at correct layer; gate recheck',
            M1_vs_B4=paired('M1', 'B4'),
            M2_vs_B4=paired('M2', 'B4'),
            M1_vs_B5=paired('M1', 'B5'),
            M2_vs_B5=paired('M2', 'B5'))
out2 = dict(phase=3151, rev='3151a', kind='v2fix_k3_gate_recheck',
            created=time.strftime('%Y-%m-%d %H:%M:%S'),
            runtime_s=round(time.time() - T0, 1), v2_k3=v2k3)
blob = json.dumps(out2, ensure_ascii=False, indent=1,
                  sort_keys=True).encode('utf-8')
out2['res_sha8'] = hashlib.sha256(blob).hexdigest()[:8]
rp2 = os.path.join(OUT, 'result_v2fix.json')
open(rp2, 'w', encoding='utf-8').write(
    json.dumps(out2, ensure_ascii=False, indent=1, sort_keys=True))
seal2 = hashlib.sha256(open(rp2, 'rb').read()).hexdigest()[:8]
out2['seal_sha8'] = seal2
open(rp2, 'w', encoding='utf-8').write(
    json.dumps(out2, ensure_ascii=False, indent=1, sort_keys=True))
log('v2fix saved: M1vB4 pass=%s M1vB5 pass=%s M2vB4 pass=%s M2vB5 pass=%s'
    % (v2k3['M1_vs_B4']['pass_gate'], v2k3['M1_vs_B5']['pass_gate'],
       v2k3['M2_vs_B4']['pass_gate'], v2k3['M2_vs_B5']['pass_gate']))
