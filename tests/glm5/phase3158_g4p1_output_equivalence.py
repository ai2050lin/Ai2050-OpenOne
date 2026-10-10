# -*- coding: utf-8 -*-
# Phase 3158 (G4-P1): 输出等价类 P1 —— 读出映射的商结构（软零谱 + 扰动预算 + 直径-曲率）
# 预注册: AGI_GPT5_MEMO Phase 3158（3157 closeout 冻结，观测前）:
#   假设: 读出层存在等价类 h~h' <=> P(.|h)~=P(.|h'); 3156 B 臂为首个案例(KL_B<=0.0023, top1 9/9)
#   设计:
#     (1) 读出矩阵 U=lm_head: 敏感度谱 s(u)=||Uu||/||u|| 奇异值谱 -> 敏感子空间维数(e90/e99)
#         与软零维数(D-e99), 参与率 PR=(sum s^2)^2/sum s^4; 128 点归一谱曲线
#     (2) 零空间 vs 随机方向扰动预算: 在 3157 KOUT 状态(操作化为槽 NL=末块输出=lm_head 真输入,
#         postnorm 坐标) 上注 eps*u, 扫 eps 至 KL=0.1, 比较 eps_null/eps_random (预期 >=10x)
#         KL 定义 = Jeffreys 0.5*(KL(b||p)+KL(p||b)), float64; 预算=log-log 插值首越点; 删失=网格上限
#     (3) 等价类直径: 3156 npz 复算 ||h_B-h_A0||(postnorm) vs KL 全 k 曲线(位置臂) + A 臂对照
#     (4) 跨模型: U 谱形状 + 预算曲线指纹 (Pearson >= 0.8)
#   门: (a) 每模型 ratio = median(null 预算)/median(rand 预算) >= 10 -> quotient_supported;
#       3-10 partial; <3 absent;  (b) summary 谱指纹 + 曲线指纹全 3 对 >= 0.8
#   协议: logits = W_U @ rmsnorm(h, gamma); gamma=model.norm.weight; 4b tied -> W_U=embed_tokens
#   正确性锚(4b): 协议 logits vs 3156 存储真 logits LG: top1 4/4 + 相对差 < 0.05
#   确定性: G 矩阵两次计算 bitwise; 基线 logits 两次 bitwise; 全程无模型前向(GPU 仅线性代数)
# 教训内置: SMOKE 目录分离; design 全 str 键 JSON; fail-fast 断言; 无 % 字面陷阱; 反斜杠仅在文件内
import os, sys, json, time, hashlib
import numpy as np

T0 = time.time()
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
MODEL = os.environ.get('P3158_MODEL') or (sys.argv[1] if len(sys.argv) > 1 else 'qwen3-4b')
SMOKE = os.environ.get('P3158_SMOKE') == '1'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
NAME = 'g4p1_output_equivalence_class'
BASE = os.path.join(RDIR, 'phase3158', NAME, MODEL)
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

def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]

def freeze_design(phase_name, design):
    eblob = json.dumps(design, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
    sha = hashlib.sha256(eblob).hexdigest()
    exe_p = os.path.join(BASE, 'execution.json')
    if os.path.exists(exe_p):
        prev = json.load(open(exe_p, encoding='utf-8'))
        assert prev['design_sha'] == sha, 'DESIGN DRIFT: delete execution.json+result.json after script change'
        log('execution.json match (sha %s)' % sha[:8])
    else:
        json.dump({'phase': 3158, 'name': phase_name, 'design_sha': sha,
                   'design': design, 'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('execution.json FROZEN (sha %s)' % sha[:8])
    return sha

def seal_result(result, out_name):
    blob = json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
    res_sha8 = sha8(blob)
    result['res_sha8'] = res_sha8
    result['verdict'] = result['verdict'] + '|sha8_' + res_sha8
    rp = os.path.join(BASE, out_name)
    json.dump(result, open(rp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    seal = sha8(open(rp, 'rb').read())
    result['seal_sha8'] = seal
    json.dump(result, open(rp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    log('RESULT %s res_sha8=%s seal=%s verdict=%s' % (out_name, res_sha8, seal, result['verdict']))
    return res_sha8, seal

# ---------------- 冻结设计 ----------------
N_ANCH = 4 if SMOKE else 16
N_DIR = 4 if SMOKE else 8
NVEC = 64
SPEC_PTS = 128
SEED = 31580308
if SMOKE:
    ALPHAS = np.geomspace(1e-3, 100.0, 9)
else:
    ALPHAS = np.geomspace(1e-4, 100.0, 25)
KL_GATE = 0.1
GATE_SUP, GATE_PAR = 10.0, 3.0
FP_GATE = 0.8
ARMS = ['null', 'rand', 'top']

design = {
    'phase': '3158', 'name': NAME, 'model': MODEL, 'smoke': bool(SMOKE),
    'prereg': 'AGI_GPT5_MEMO Phase 3158 (frozen at 3157 closeout, before any observation)',
    'protocol': 'logits = W_U @ h_slotNL — HF output_hidden_states last slot is ALREADY '
                'post-final-norm (verified: W_U@h matches stored LG cos=1.0000 maxdiff=fp16); '
                'anchors = 3157 collect.npz H slot NL (exact lm_head input); '
                'prereg KOUT-state operationalized to slot NL readout input',
    'perturb': "p' = p + alpha*||p||*u; u from W_U right singular spectrum (D space)",
    'kl_def': 'jeffreys 0.5*(KL(base||pert)+KL(pert||base)) in float64 via logsumexp identity',
    'n_anchors': N_ANCH, 'anchor_pick': 'linspace(0,nrows-1,N_ANCH) rounded on 3157 row order',
    'n_dirs_per_arm': N_DIR, 'arms': ARMS,
    'dir_pick': 'evenly spaced indices within top64/bottom64 eigvector blocks; rand=unit gauss QR',
    'alphas': ['%.6g' % a for a in ALPHAS], 'alpha_rel': 'alpha * ||p|| (postnorm norm)',
    'kl_gate': KL_GATE, 'gate_ratio_supported': GATE_SUP, 'gate_ratio_partial': GATE_PAR,
    'ratio_def': 'median(null budgets)/median(rand budgets) over rows x dirs; '
                 'censored budget reported as grid max (lower bound, flagged)',
    'spec_pts': SPEC_PTS, 'nvec': NVEC, 'seed': SEED,
    'part3': 'qwen3-4b only: 3156 npz recompute; verify anchor protocol-vs-LG top1 4/4 rel<0.05; '
             'd_rel(postnorm) vs forward-KL(A0||k) curves, position arm B vs context arm A',
    'fp_gate': FP_GATE,
    'summary_gates': 'spec128 pearson 3 pairs >= 0.8 AND budget-curve pearson 3 pairs >= 0.8',
}
DESIGN_SHA = freeze_design('g4p1_output_equivalence_class', design)

MDIR_MAP = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'Qwen3-14B', 'glm4': 'glm4-9b-chat-hf'}
R7 = os.path.join(RDIR, 'phase3157', 'g2p2_transform_algebra_commutator')
R6 = os.path.join(RDIR, 'phase3156', 'g3p1_position_shift_family')

def logsoftmax64(x):
    m = x.max()
    return x - (m + np.log(np.exp(x - m).sum()))

def logsumexp64(M, axis):
    m = M.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(M - m).sum(axis=axis, keepdims=True))).squeeze(axis)

# ================= summary 模式 =================
if MODEL == 'summary':
    log('summary mode: fingerprint verdict over 3 models')
    rs, spec_curves, kl_curves = {}, {}, {}
    for m in ('qwen3-4b', 'qwen3-14b', 'glm4'):
        rp = os.path.join(RDIR, 'phase3158', NAME, m, 'result.json')
        r = json.load(open(rp, encoding='utf-8'))
        rs[m] = r
        z = np.load(os.path.join(RDIR, 'phase3158', NAME, m, 'collect.npz'))
        spec_curves[m] = z['spec128'].astype(np.float64)
        kl_curves[m] = z['arm_curves'].astype(np.float64)   # (3, NA)
        log('%s: verdict=%s ratio=%.3f' % (m, r['verdict'], r['ratio']))
    NA = kl_curves['qwen3-4b'].shape[1]
    assert all(kl_curves[m].shape[1] == NA for m in kl_curves), 'alpha grid mismatch'
    spec_pairs, curve_pairs = {}, {}
    models = list(spec_curves)
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = models[i], models[j]
            spec_pairs['%s_vs_%s' % (a, b)] = float(np.corrcoef(spec_curves[a], spec_curves[b])[0, 1])
            # 曲线指纹: null 与 rand 两臂的 log10 KL 曲线拼接
            ca = np.concatenate([np.log10(kl_curves[a][0] + 1e-300), np.log10(kl_curves[a][1] + 1e-300)])
            cb = np.concatenate([np.log10(kl_curves[b][0] + 1e-300), np.log10(kl_curves[b][1] + 1e-300)])
            curve_pairs['%s_vs_%s' % (a, b)] = float(np.corrcoef(ca, cb)[0, 1])
    fpmin_spec = float(min(spec_pairs.values()))
    fpmin_curve = float(min(curve_pairs.values()))
    qcls = [rs[m]['quotient_class'] for m in models]
    q_ok = sum(1 for c in qcls if c == 'supported')
    fp_ok = (fpmin_spec >= FP_GATE) and (fpmin_curve >= FP_GATE)
    if fp_ok and q_ok == 3:
        verdict = 'g4p1_fingerprint_consistent|quotient_3/3'
    elif fp_ok:
        verdict = 'g4p1_fingerprint_consistent|quotient_%d/3_mixed' % q_ok
    else:
        verdict = 'g4p1_fingerprint_divergent_material_downgrade'
    result = {
        'phase': 3158, 'name': NAME, 'mode': 'summary', 'smoke': bool(SMOKE),
        'prereg': design['prereg'], 'models': models,
        'quotient_classes': {m: rs[m]['quotient_class'] for m in models},
        'ratios': {m: rs[m]['ratio'] for m in models},
        'spec_pairs': spec_pairs, 'curve_pairs': curve_pairs,
        'fpmin_spec': fpmin_spec, 'fpmin_curve': fpmin_curve,
        'fp_gate': FP_GATE, 'gates': {'fingerprint': bool(fp_ok), 'quotient_3of3': q_ok == 3},
        'verdict': verdict, 'design_sha': DESIGN_SHA,
        'runtime_s': round(time.time() - T0, 1),
    }
    seal_result(result, 'result_summary.json')
    log('SUMMARY DONE')
    sys.exit(0)

# ================= 模型模式 =================
import torch
from safetensors import safe_open

MDIR = os.path.join(ROOT, 'models', 'hf', MDIR_MAP[MODEL])
cfgm = json.load(open(os.path.join(MDIR, 'config.json'), encoding='utf-8'))
V, D = int(cfgm['vocab_size']), int(cfgm['hidden_size'])
NL = int(cfgm['num_hidden_layers'])
EPS = float(cfgm.get('rms_norm_eps', 1e-6))
log('model=%s V=%d D=%d NL=%d eps=%g smoke=%s' % (MODEL, V, D, NL, EPS, SMOKE))

# --- 载入 W_U（直接 safetensors 读取，不实例化模型；槽 NL 已含 final norm，无需 gamma）---
tied = bool(cfgm.get('tie_word_embeddings', False))
want = 'model.embed_tokens.weight' if tied else 'lm_head.weight'
W_t = None
for sh in sorted(os.listdir(MDIR)):
    if not sh.endswith('.safetensors'):
        continue
    with safe_open(os.path.join(MDIR, sh), framework='pt') as f:
        if want in set(f.keys()):
            W_t = f.get_tensor(want)
            break
assert W_t is not None, 'unembed not found'
assert tuple(W_t.shape) == (V, D), (W_t.shape, V, D)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
W = W_t.to(dev, torch.float32)
del W_t
log('W_U loaded tied=%s from %s dev=%s' % (tied, want, dev))

# --- Part 1: 谱（G = W^T W, eigh float64 CPU; 两次 bitwise）---
def gram():
    G = torch.zeros((D, D), dtype=torch.float32, device=dev)
    CH = 8192
    for s in range(0, V, CH):
        blk = W[s:s + CH]
        G += blk.T @ blk
    return G
G1 = gram()
G2 = gram()
assert torch.equal(G1, G2), 'gram bitwise mismatch'
log('gram bitwise OK; eig start')
Gc = G1.cpu().numpy().astype(np.float64)
del G1, G2
if dev == 'cuda':
    torch.cuda.empty_cache()
w0, U0 = np.linalg.eigh(Gc)            # 升序特征值
w0 = np.clip(w0, 0.0, None)
sig = np.sqrt(w0)                       # (D,) 升序
tot = float(w0.sum()) + 1e-300
cum = np.cumsum(w0[::-1]) / tot
e99 = int(np.searchsorted(cum, 0.99) + 1)
e90 = int(np.searchsorted(cum, 0.90) + 1)
PR = float(tot ** 2 / (w0 ** 2).sum() + 1e-300)
log('spec: sig_max=%.4g sig_min=%.4g e90=%d e99=%d PR=%.2f' % (sig[-1], sig[0], e90, e99, PR))
ranks_f = np.geomspace(1.0, float(D), SPEC_PTS)
sig2_desc = w0[::-1] / tot
spec128 = np.exp(np.interp(ranks_f - 1.0, np.arange(D), np.log(sig2_desc + 1e-300))).astype(np.float32)
top64 = U0[:, ::-1][:, :NVEC].copy()    # (D,64) 降序
bot64 = U0[:, :NVEC].copy()             # (D,64) 升序(最小)
sig_top8 = sig[::-1][:8].astype(np.float32)
sig_bot8 = sig[:8].astype(np.float32)
log('eig done; vectors sliced')

# --- 方向组 ---
pick = np.linspace(0, NVEC - 1, N_DIR).round().astype(int)
assert len(set(pick)) == N_DIR
dirs = {}
dirs['null'] = bot64[:, pick]
dirs['top'] = top64[:, pick]
rng = np.random.default_rng(SEED)
Rr = rng.standard_normal((D, N_DIR))
Rr, _ = np.linalg.qr(Rr)
dirs['rand'] = Rr
UU = np.zeros((3, N_DIR, V), np.float64)
for ai, arm in enumerate(ARMS):
    Ut = torch.from_numpy(dirs[arm].astype(np.float32)).to(dev)
    UUt = (W @ Ut).T.contiguous().cpu().numpy().astype(np.float64)   # (N_DIR, V)
    UU[ai] = UUt
    log('arm %s uu done norm_mean=%.4g' % (arm, float(np.linalg.norm(UUt, axis=1).mean())))
del W, Ut
if dev == 'cuda':
    torch.cuda.empty_cache()

# --- Part 2: 3157 KOUT(槽 NL) 锚点扰动预算 ---
z7 = np.load(os.path.join(R7, MODEL, 'collect.npz'))
r7 = json.load(open(os.path.join(R7, MODEL, 'result.json'), encoding='utf-8'))
H7 = z7['H']
nrows7 = int(r7.get('n_rows', H7.shape[0]))
assert H7.shape[0] == nrows7 and H7.shape[1] == NL + 1 and H7.shape[2] == D, (H7.shape, nrows7, NL + 1, D)
slot = NL
idx = np.unique(np.linspace(0, nrows7 - 1, N_ANCH).round().astype(int))
assert len(idx) == N_ANCH
P = np.zeros((N_ANCH, D), np.float64)
for i, r0 in enumerate(idx):
    P[i] = H7[r0, slot, :].astype(np.float64)   # slot NL already post-final-norm
p_norms = np.linalg.norm(P, axis=1)
log('anchors %s rows=%s slot=%d p_norm mean=%.3f cv=%.3f' % (
    idx.tolist(), nrows7, slot, float(p_norms.mean()), float(p_norms.std() / (p_norms.mean() + 1e-18))))

# 基线 logits（协议）两次 bitwise
WT = torch.from_numpy(P.astype(np.float32)).to(dev)
def reload_W():
    Wt = None
    for sh in sorted(os.listdir(MDIR)):
        if not sh.endswith('.safetensors'):
            continue
        with safe_open(os.path.join(MDIR, sh), framework='pt') as f:
            if want in set(f.keys()):
                Wt = f.get_tensor(want)
                break
    return Wt.to(dev, torch.float32)
W = reload_W()
LB = torch.zeros((N_ANCH, V), dtype=torch.float32, device=dev)
CH = 4096
for s in range(0, V, CH):
    LB[:, s:s + CH] = WT @ W[s:s + CH].T
LB2 = torch.zeros((N_ANCH, V), dtype=torch.float32, device=dev)
for s in range(0, V, CH):
    LB2[:, s:s + CH] = WT @ W[s:s + CH].T
assert torch.equal(LB, LB2), 'base logits bitwise mismatch'
del WT, LB2
LBc = LB.cpu().numpy().astype(np.float64)
del LB
log('base logits bitwise OK (N_ANCH x V)')
pexp = np.zeros((N_ANCH, V), np.float64)
logZ_b = np.zeros(N_ANCH, np.float64)
for i in range(N_ANCH):
    logZ_b[i] = logsumexp64(LBc[i], 0)
    pexp[i] = np.exp(LBc[i] - logZ_b[i])
pdu = np.einsum('av,irv->air', pexp, UU)   # (N_ANCH, 3, N_DIR) <p_b, u>
log('pdu done')

# KL 扫描: 每 anchor 行, alpha 分块
NAC = 5
KL = np.zeros((3, N_ANCH, N_DIR, len(ALPHAS)), np.float64)
for i in range(N_ANCH):
    lb = LBc[i]
    pn = float(p_norms[i])   # 状态步长 eps = alpha*||p||
    for cs in range(0, len(ALPHAS), NAC):
        al = ALPHAS[cs:cs + NAC]
        eps_a = al * pn
        # KL_inv = <P, delta> - (logZ_p - logZ_b); delta = eps*u -> <P,delta> = eps * <p_b_norm?, u>
        for ai in range(3):
            for di in range(N_DIR):
                uud = UU[ai, di]
                Mk = lb[None, :] + eps_a[:, None] * uud[None, :]
                lzk = logsumexp64(Mk, 1)
                Pk = np.exp(Mk - lzk[:, None])
                kfwd = -eps_a * pdu[i, ai, di] + lzk - logZ_b[i]
                kinv = eps_a * (Pk @ uud) - (lzk - logZ_b[i])
                KL[ai, i, di, cs:cs + len(al)] = 0.5 * (kfwd + kinv)
    log('anchor row %d/%d KL scan done' % (i + 1, N_ANCH))
del LBc, pexp

# 预算提取
BUD = np.zeros((3, N_ANCH, N_DIR), np.float64)
CEN = np.zeros((3, N_ANCH, N_DIR), np.uint8)
SMALL = np.zeros((3, N_ANCH, N_DIR), np.uint8)
la = np.log(ALPHAS)
for ai in range(3):
    for i in range(N_ANCH):
        for di in range(N_DIR):
            k = KL[ai, i, di]
            bad = np.where(k >= KL_GATE)[0]
            if len(bad) == 0:
                BUD[ai, i, di] = ALPHAS[-1]
                CEN[ai, i, di] = 1
            elif bad[0] == 0:
                BUD[ai, i, di] = ALPHAS[0]
                SMALL[ai, i, di] = 1
            else:
                j = bad[0]
                t = (np.log(KL_GATE) - np.log(k[j - 1])) / (np.log(k[j]) - np.log(k[j - 1]) + 1e-300)
                BUD[ai, i, di] = float(np.exp(la[j - 1] + t * (la[j] - la[j - 1])))
med = {arm: float(np.median(BUD[ai])) for ai, arm in enumerate(ARMS)}
cen_frac = {arm: float(CEN[ai].mean()) for ai, arm in enumerate(ARMS)}
ratio = med['null'] / (med['rand'] + 1e-300)
if CEN[0].mean() > 0.5 and med['rand'] < ALPHAS[-1]:
    ratio = max(ratio, ALPHAS[-1] / med['rand'])   # 删失下界
if ratio >= GATE_SUP:
    qcls = 'supported'
elif ratio >= GATE_PAR:
    qcls = 'partial'
else:
    qcls = 'absent'
log('budgets: null=%.4g rand=%.4g top=%.4g ratio=%.3f cens=%s -> %s' % (
    med['null'], med['rand'], med['top'], ratio, {a: round(c, 2) for a, c in cen_frac.items()}, qcls))
arm_curves = np.zeros((3, len(ALPHAS)), np.float64)
for ai in range(3):
    arm_curves[ai] = np.median(KL[ai].reshape(-1, len(ALPHAS)), axis=0)

# --- Part 3 (仅 qwen3-4b): 3156 复算 + 协议正确性锚 ---
part3 = None
if MODEL == 'qwen3-4b':
    z6 = np.load(os.path.join(R6, 'qwen3-4b', 'collect.npz'))
    H6 = z6['H'].astype(np.float64)     # (36, 11, 37, D)
    LG6 = z6['LG'].astype(np.float64)   # (36, V)
    langs6 = [str(x) for x in z6['lang']]
    arms6 = [str(x) for x in z6['arm']]
    ks6 = [int(x) for x in z6['k']]
    ntg6 = [int(x) for x in z6['n_tgt']]
    IDX6 = {(langs6[s], arms6[s], ks6[s]): s for s in range(len(langs6))}
    assert H6.shape[2] == NL + 1 and LG6.shape[1] == V
    # 正确性锚: 协议 logits（W_U @ h_slotNL 原始）vs LG（4 序列）
    vs = [('zh', 'A', 0), ('zh', 'B', 128), ('en', 'A', 0), ('en', 'B', 128)]
    vrows = [IDX6[v] for v in vs]
    Pv = np.stack([H6[s, ntg6[s] - 1, slot].astype(np.float64) for s in vrows])
    PT = torch.from_numpy(Pv.astype(np.float32)).to(dev)
    LP = torch.zeros((len(vrows), V), dtype=torch.float32, device=dev)
    for s0 in range(0, V, CH):
        LP[:, s0:s0 + CH] = PT @ W[s0:s0 + CH].T
    LPc = LP.cpu().numpy().astype(np.float64)
    del PT, LP
    t1 = [int(np.argmax(LPc[i]) == np.argmax(LG6[s])) for i, s in enumerate(vrows)]
    relerr = [float(np.abs(LPc[i] - LG6[s]).max() / (np.abs(LG6[s]).max() + 1e-9)) for i, s in enumerate(vrows)]
    log('verify anchor: top1=%s relerr=%s' % (t1, ['%.4f' % x for x in relerr]))
    assert all(t1) and max(relerr) < 0.05, ('protocol anchor FAIL', t1, relerr)
    # 直径-曲率: 全 36 序列（槽 NL 原始状态）
    Pn = np.zeros((len(langs6), D), np.float64)
    for s in range(len(langs6)):
        Pn[s] = H6[s, ntg6[s] - 1, slot].astype(np.float64)
    d_rel = {}
    kl_f = {}
    # 逐语言逐臂
    PnT = torch.from_numpy(Pn.astype(np.float32)).to(dev)
    LN = torch.zeros((len(langs6), V), dtype=torch.float32, device=dev)
    for s0 in range(0, V, CH):
        LN[:, s0:s0 + CH] = PnT @ W[s0:s0 + CH].T
    LNc = LN.cpu().numpy().astype(np.float64)
    del PnT, LN, W
    if dev == 'cuda':
        torch.cuda.empty_cache()
    for lg in ('zh', 'en'):
        ia0 = IDX6[(lg, 'A', 0)]
        lp0 = logsoftmax64(LNc[ia0])
        p0 = np.exp(lp0)
        for arm in ('A', 'B'):
            for s in range(len(langs6)):
                if langs6[s] != lg or arms6[s] != arm:
                    continue
                k = ks6[s]
                lpk = logsoftmax64(LNc[s])
                klv = float(np.sum(p0 * (lp0 - lpk)))
                drv = float(np.linalg.norm(Pn[s] - Pn[ia0]) / (np.linalg.norm(Pn[ia0]) + 1e-18))
                d_rel['%s_%s_k%d' % (lg, arm, k)] = drv
                kl_f['%s_%s_k%d' % (lg, arm, k)] = klv
    diam = {}
    for lg in ('zh', 'en'):
        pts_b = [(d_rel['%s_B_k%d' % (lg, k)], kl_f['%s_B_k%d' % (lg, k)]) for k in sorted(set(ks6))]
        pts_a = [(d_rel['%s_A_k%d' % (lg, k)], kl_f['%s_A_k%d' % (lg, k)]) for k in sorted(set(ks6))]
        diam['%s_B' % lg] = {'dmax_kl001': max([d for d, k in pts_b if k < 0.01] or [0.0]),
                             'dmax_kl01': max([d for d, k in pts_b if k < 0.1] or [0.0]),
                             'kl_max': max(k for d, k in pts_b)}
        diam['%s_A' % lg] = {'d_kl_min': min(d for d, k in pts_a), 'kl_min': min(k for d, k in pts_a),
                             'kl_max': max(k for d, k in pts_a)}
    part3 = {'verify_top1': t1, 'verify_relerr': relerr, 'verify_rows': ['%s_%s_k%d' % v for v in vs],
             'diam': diam}
    log('part3 diam: %s' % json.dumps(diam)[:300])

# --- npz + result ---
npz_p = os.path.join(BASE, 'collect.npz')
np.savez_compressed(
    npz_p,
    spec128=spec128, sig_top8=sig_top8, sig_bot8=sig_bot8,
    top64=top64.astype(np.float32), bot64=bot64.astype(np.float32),
    alphas=ALPHAS, anchor_idx=idx, p_norms=p_norms.astype(np.float32),
    UU=UU.astype(np.float32), KL=KL.astype(np.float32),
    BUD=BUD, CEN=CEN, SMALL=SMALL, arm_curves=arm_curves.astype(np.float32),
)
npz_sha = sha8(open(npz_p, 'rb').read())
log('npz saved %s sha8=%s' % (os.path.basename(npz_p), npz_sha))

result = {
    'phase': 3158, 'name': NAME, 'mode': MODEL, 'smoke': bool(SMOKE),
    'prereg': design['prereg'], 'protocol': design['protocol'],
    'model': {'V': V, 'D': D, 'NL': NL, 'tied': tied, 'rms_eps': EPS},
    'part1': {'sigma_max': float(sig[-1]), 'sigma_min': float(sig[0]),
              'e90_dim': e90, 'e99_dim': e99, 'soft_null_dim': int(D - e99),
              'participation_ratio': PR,
              'spec_head': [float(x) for x in sig_top8[:4]],
              'spec_tail': [float(x) for x in sig_bot8[:4]]},
    'part2': {'n_anchors': N_ANCH, 'n_dirs': N_DIR,
              'budget_median': med, 'censored_frac': cen_frac,
              'small_frac': {arm: float(SMALL[ai].mean()) for ai, arm in enumerate(ARMS)},
              'ratio': float(ratio), 'kl_gate': KL_GATE,
              'p_norm_mean': float(p_norms.mean()), 'p_norm_cv': float(p_norms.std() / (p_norms.mean() + 1e-18))},
    'quotient_class': qcls, 'ratio': float(ratio),
    'part3_3156': part3,
    'det': {'gram_bitwise': True, 'base_logits_bitwise': True},
    'gates': {'quotient_supported': qcls == 'supported',
              'ratio_ge10': bool(ratio >= GATE_SUP)},
    'verdict': 'g4p1_quotient_%s|ratio_%.3f|e99dim_%d|nullfrac_%.3f' % (
        qcls, ratio, e99, (D - e99) / D),
    'npz_sha8': npz_sha, 'design_sha': DESIGN_SHA,
    'runtime_s': round(time.time() - T0, 1),
}
seal_result(result, 'result.json')
log('DONE model=%s smoke=%s' % (MODEL, SMOKE))
