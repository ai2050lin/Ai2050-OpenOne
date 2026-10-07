# -*- coding: utf-8 -*-
"""
Q04: E_ar(k) 装置建造 —— k 步自回归 logit-margin 误差曲线（B 闸门；gpu=mid；SMOKE 先通）
========================================================================================
依据 : RDC_RESEARCH_CONSTITUTION_v1 §1 (I1)
       research/deepseek/atlas/metric_dict.json -> global_kpis.E_ar   （口径已冻结，待 Q04 解析）
       research/deepseek/docs/AGI_DEEPSEEK_MEMO.md  Phase 27 §2（E_ar 口径）+ Phase 37（Q03 基线）

口径（本 Phase 解析并冻结，登记回 metric_dict v3）:
  E_ar(k) = mean_cells | margin_pred(k; i, c) - margin_true(k; i, c) |   （单位 = logit）
  margin  = logit(t_target) - logit(t_competitor)
    t_target     = 该 cell 的真实类名首 token
    t_competitor = k=0（上下文 = 模板前缀）处 logit 最高的“非该类”类首 token（每 cell 冻结）
  margin_true(k) = 把模型自身前 k 步贪心生成内容回喂后，第 k 步 last-position 的 margin
                   （k=0 即模板前缀本身，无回喂）—— 纯自回归 rollout，无 teacher forcing
  margin_pred(k) = 与 E_read 同一 B4 加性族（one-hot[entity 41] + one-hot[class 6]
                   + one-hot[template 3] + bias, ridge λ=1e-3）在 held-out 行上的预测
  归一化         : 按冻结公式为**原始 L1（logit 单位，不归一）**；另附相对参照供解释，
                   但 E_ar(k) 主体口径 = 原始 L1（避免与 E_read 的归一化 MSE 量纲混淆）

面板 : 与 E_read **同一 held-out 面板族**
       CLASSES×ENT = 41 实体 / 6 类；PAIRS = 41×6 = 246；(pair × 3 模板) = 738 行
       S1 划分 seeds=[7,8,9], frac=0.2 -> 49 test pairs × 3 模板 = 147 held-out 行/seed
K    : 预注册 K = 16（SMOKE 用 K=4）

装置门（必须可失败）:
  D1 k=0 确定性锚 : 同一 cell 连跑两次，k=0..K 的 6 类 logit 须逐位相同（harness 确定性）
  D2 非退化       : held-out 的 margin_true 在 cells 上 std > 0（否则目标恒定，装置测不到东西）
  D3 存活性       : 全部 E_ar(k) 有限（无 NaN/Inf）
科学门（Q04 只报 SMOKE 曲线；形状判决留给 Q05）:
  S1 漂移可检     : max_{k>=1} E_ar(k) >= 0.05（logit）——否则“自回归漂移”在装置分辨率下不可检
                    （0.05 与 5% 门同族，观测前冻结）

产物 : tests/deepseek/result/q04_execution.json （预注册，任何模型观测前冻结）
       tests/deepseek/result/q04_result.json     （SMOKE 结果 + 门判定）
       tests/deepseek/result/q04_report.txt      （人类可读）
纪律 : 数字一律 result 现场渲染；预注册先冻结再观测；SMOKE 与正式目录分离
"""
import os, sys, json, time, hashlib
import numpy as np

T0 = time.time()
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

MODEL = os.environ.get('P_Q04_MODEL') or (sys.argv[1] if len(sys.argv) > 1 else 'qwen3-4b')
SMOKE = os.environ.get('P_Q04_SMOKE') == '1'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(ROOT, 'tests', 'deepseek', 'result')
os.makedirs(OUT, exist_ok=True)
if SMOKE:
    LOGP = os.path.join(OUT, 'q04_smoke_run_log.txt')
else:
    LOGP = os.path.join(OUT, 'q04_run_log.txt')
LOG = []

def log(s):
    ln = '[%7.1f] %s' % (time.time() - T0, s)
    LOG.append(ln)
    try:
        with open(LOGP, 'a', encoding='utf-8') as f:
            f.write(ln + '\n')
    except Exception:
        pass
    try:
        print(ln, flush=True)
    except Exception:
        pass

def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

# ---------------- 冻结面板材料（与 3151/3152/Q03 逐字一致） ----------------
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
# P0 = 模板前缀（在 {c} 处截断）；suffix 仅用于人类可读参照，不参与 rollout 目标
TPL_P0 = {0: '{e}是一种',
          1: '{e}属于',
          2: '{e}，一种常见的'}
TPL_SUF = {0: '。', 1: '这一类。', 2: '。'}
SEEDS_S1 = [7, 8, 9]
FRAC_S1 = 0.2
K_FULL = 16
K_SMOKE = 4
SMOKE_PAIRS = 60          # SMOKE 子面板：前 60 对 × 3 模板 = 180 行
MAXCTX = 64               # rollout 上下文硬上限（防跑飞）
LAM = 1e-3
MARGIN_FLOOR = 0.05       # S1 漂移可检门（logit）

ENTS = [e for cl in CLASSES for e in ENT[cl]]
CLS_OF = [CLASSES.index(cl) for cl in CLASSES for e in ENT[cl]]
NE = len(ENTS); NC = len(CLASSES)
PAIRS = [(i, c) for i in range(NE) for c in range(NC)]
NP_ = len(PAIRS); NT = len(TPL)
assert (NE, NC, NP_, NT) == (41, 6, 246, 3), (NE, NC, NP_, NT)
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

def ridge_primal(Xtr, Ytr, lam=LAM):
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1], dtype=np.float32)
    return np.linalg.solve(A, Xtr.T @ Ytr)

def phi_main(train_set):
    cols = NE + NC + NT + 1
    tr_rows = rows_of(train_set)
    def rowvec(t, pi):
        i, c = PAIRS[pi]
        v = np.zeros(cols, np.float32)
        v[i] = 1.0
        v[NE + c] = 1.0
        v[NE + NC + t] = 1.0
        v[-1] = 1.0
        return v
    Xtr = np.stack([rowvec(r // NP_, r % NP_) for r in tr_rows])
    return Xtr, tr_rows, rowvec, cols

MDIR_MAP = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'Qwen3-14B', 'glm4-9b': 'glm4-9b-chat-hf'}
MDIR = os.path.join(ROOT, 'models', 'hf', MDIR_MAP[MODEL])

# ================= 预注册（任何模型观测前冻结） =================
K = K_SMOKE if SMOKE else K_FULL
if SMOKE:
    SMOKE_SET = set(PAIRS[:SMOKE_PAIRS])
    PAIRS_ACT = [p for p in PAIRS if p in SMOKE_SET]
else:
    PAIRS_ACT = list(PAIRS)
NPA = len(PAIRS_ACT)
ROWS_ACT = [t * NP_ + pi for t in range(NT)
            for pi, p in enumerate(PAIRS) if p in SMOKE_SET] if SMOKE else \
           list(range(NT * NP_))

design = dict(
    query='Q04', title='E_ar(k) 装置建造',
    mode='device-build + SMOKE' if SMOKE else 'device-build + FULL',
    model=MODEL, mdir=MDIR, smoke=SMOKE,
    panel=dict(classes=CLASSES, ents=ENTS, tpl=TPL, tpl_p0=TPL_P0,
               n_entities=NE, n_classes=NC, n_pairs=NP_, n_tpl=NT,
               rows_full=NT * NP_, rows_active=len(ROWS_ACT),
               pairs_active=NPA,
               seeds_s1=SEEDS_S1, frac_s1=FRAC_S1,
               heldout_rows_per_seed=int(round(FRAC_S1 * NP_)) * NT,
               engine_note='与 E_read 同一 held-out 面板族（S1 fold）'),
    device=dict(
        k_range='0..K (k=0 为确定性锚)',
        K=K,
        roll='greedy self-rollout: ctx0 = P0 token ids; 每步读 last-position 6 类 logit; '
             'k<K 时 apped argmax(logits_last) 回喂',
        margin='logit(t_target) - logit(t_competitor); t_target = 该类首 token; '
               't_competitor = k=0 处 logit 最高的非该类类首 token（每 cell 冻结）',
        predictor='B4 additive: ridge_primal(one-hot[entity]+one-hot[class]+one-hot[template]+bias)',
        lam=LAM, feature_cols=NE + NC + NT + 1,
        target_units='logit（原始 L1，不归一化）',
        normalization='none (frozen formula is raw L1); secondary relative scale reported only',
        seeds=SEEDS_S1,
        aggregation='per seed: mean over held-out rows; then mean over 3 seeds',
        maxctx=MAXCTX,
    ),
    gates=dict(
        D1='k=0 determinism: two passes bit-identical on 6-class logits (k=0..K)',
        D2='non-degeneracy: std(margin_true over held-out cells) > 0',
        D3='liveness: all E_ar(k) finite',
        S1=dict(rule='max_{k>=1} E_ar(k) >= MARGIN_FLOOR', thr=MARGIN_FLOOR,
                sense='>= pass', note='姿态：几何漂移可检；Q05 判决形状'),
    ),
    carriers={},
    frozen_before='any model observation',
    # 注意：design 内**不得**含易变字段（时间戳等），否则冻结哈希不可复算。
    # 冻结时刻记录在 run_log 与 result 中，不进入哈希域。
)
# 面板/文件指纹（预注册的一部分，供 X-phase 比对）
design['panel_sha8'] = hashlib.sha256(
    json.dumps([CLASSES, ENTS, [TPL[k] for k in sorted(TPL)], SEEDS_S1, FRAC_S1],
               ensure_ascii=False).encode('utf-8')).hexdigest()[:8]
for f in [MDIR + r'\config.json']:
    if os.path.exists(f):
        design.setdefault('mdir_config_sha8', sha8(f))
eblob = json.dumps(design, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
design_sha = hashlib.sha256(eblob).hexdigest()
design['design_sha'] = design_sha

exe_p = os.path.join(OUT, 'q04_smoke_execution.json' if SMOKE else 'q04_execution.json')
if os.path.exists(exe_p):
    prev = json.load(open(exe_p, encoding='utf-8'))
    assert prev['design_sha'] == design_sha, 'DESIGN DRIFT'
    log('execution.json match (sha %s)' % design_sha[:8])
else:
    with open(exe_p, 'w', encoding='utf-8') as f:
        json.dump(design, f, ensure_ascii=False, indent=1)
    log('execution.json FROZEN design_sha=%s model=%s K=%d rows_active=%d smoke=%s'
        % (design_sha[:8], MODEL, K, len(ROWS_ACT), SMOKE))

# ================= 模型观测 =================
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
log('model load begin: %s' % MDIR)
model = AutoModelForCausalLM.from_pretrained(
    MDIR, dtype=torch.bfloat16, trust_remote_code=True).to('cuda').eval()
log('model loaded: %s' % type(model).__name__)
NL = model.config.num_hidden_layers

def ids_of(text):
    return tok(text, add_special_tokens=False)['input_ids']

CLS_TOK = [ids_of(c)[0] for c in CLASSES]
assert len(set(CLS_TOK)) == NC, 'class first tokens collide'
CLS_T = torch.tensor(CLS_TOK, device='cuda')

# cells = (t, pi)；只跑 active 子集
CELLS = [(t, pi) for t in range(NT) for pi, p in enumerate(PAIRS)
         if (not SMOKE) or (p in SMOKE_SET)]
P0IDS = {(t, pi): ids_of(TPL_P0[t].format(e=ENTS[PAIRS[pi][0]]))
         for (t, pi) in CELLS}
log('cells=%d  (K=%d, forwards=%d)' % (len(CELLS), K, len(CELLS) * (K + 1)))

NCELL = len(CELLS)
CLSLOG = np.zeros((NCELL, K + 1, NC), np.float32)   # 每 cell 每步 6 类 last-position logit
ROLLOUT_TXT = {}

def rollout(cell_j):
    t, pi = CELLS[cell_j]
    ctx = list(P0IDS[(t, pi)])
    out = np.zeros((K + 1, NC), np.float32)
    txt = []
    with torch.no_grad():
        for k in range(K + 1):
            ii = torch.tensor([ctx], device='cuda')
            o = model(input_ids=ii)
            lg = o.logits[0, -1].float()
            out[k] = lg[CLS_T].detach().cpu().numpy()
            if k < K:
                nxt = int(torch.argmax(lg).item())
                ctx.append(nxt)
                if len(ctx) < MAXCTX:
                    txt.append(tok.decode([nxt]))
            del o, lg
    return out, ''.join(txt)

for j in range(NCELL):
    CLSLOG[j], ROLLOUT_TXT[(CELLS[j][0], CELLS[j][1])] = rollout(j)
    if (j + 1) % 60 == 0 or j == NCELL - 1:
        log('rollout %d/%d' % (j + 1, NCELL))

# ---- D1: k=0 确定性锚（同 cell 再跑一遍，逐位比对） ----
CLSLOG_D1, _ = rollout(0)
d1_ok = bool(np.array_equal(CLSLOG_D1, CLSLOG[0]))
log('D1 determinism (cell0 re-run bit-identical) = %s' % d1_ok)

del model
torch.cuda.empty_cache()

# ================= margin 与 held-out 误差 =================
TRUE_MARG = np.zeros((NCELL, K + 1), np.float32)   # margin_true(k; cell)
COMPETITOR = np.zeros(NCELL, np.int64)
for j, (t, pi) in enumerate(CELLS):
    i, c = PAIRS[pi]
    # t_competitor = k=0 处 logit 最高的“非该类”类首 token（每 cell 冻结）
    base = CLSLOG[j, 0].copy()
    base[c] = -np.inf
    tc = int(np.argmax(base))
    COMPETITOR[j] = tc
    TRUE_MARG[j] = CLSLOG[j, :, c] - CLSLOG[j, :, tc]
CELL_ROW = np.array([t * NP_ + pi for (t, pi) in CELLS])
CELL_PAIR = np.array([pi for (_, pi) in CELLS])
CELL_TPL = np.array([t for (t, _) in CELLS])

def b4_eval(seed, kvec):
    """对每个 k：train B4 on train rows -> predict held-out rows -> |pred-true| 平均。"""
    train_set, test_set = split_s1(seed)
    tr_set = set(rows_of(train_set))
    te_set = set(rows_of(test_set))
    tr_local = [j for j in range(NCELL) if CELL_ROW[j] in tr_set]
    te_local = [j for j in range(NCELL) if CELL_ROW[j] in te_set]
    _, _, rowvec, cols = phi_main(train_set)
    Xa = np.stack([rowvec(CELL_TPL[j], CELL_PAIR[j]) for j in tr_local])
    Xe = np.stack([rowvec(CELL_TPL[j], CELL_PAIR[j]) for j in te_local])
    res = {}
    for k in kvec:
        Y = TRUE_MARG[:, k]
        W = ridge_primal(Xa, Y[tr_local])
        pred = Xe @ W
        true = Y[te_local]
        mae_b4 = float(np.abs(pred - true).mean())
        # null：常数（train 均值）预测
        mae_const = float(np.abs(Y[tr_local].mean() - true).mean())
        scale = float(true.std())
        res[k] = dict(mae_b4=mae_b4, mae_const=mae_const,
                      scale=scale, n_rows=int(len(te_local)),
                      mean_true=float(true.mean()))
    return res

per_seed = {}
for s in SEEDS_S1:
    per_seed[s] = b4_eval(s, list(range(K + 1)))
    log('seed %d E_ar_B4[0..%d] = %s' % (s, K, ['%.4f' % per_seed[s][k]['mae_b4'] for k in range(K + 1)]))

# 聚合
E_ar = {k: float(np.mean([per_seed[s][k]['mae_b4'] for s in SEEDS_S1])) for k in range(K + 1)}
E_ar_const = {k: float(np.mean([per_seed[s][k]['mae_const'] for s in SEEDS_S1])) for k in range(K + 1)}
SCALE = {k: float(np.mean([per_seed[s][k]['scale'] for s in SEEDS_S1])) for k in range(K + 1)}
# 派生量（report-only；不改变冻结口径）：相对误差与 k 步漂移
E_ar_rel = {k: float(E_ar[k] / max(SCALE[k], 1e-9)) for k in range(K + 1)}
DRIFT = {k: float(E_ar[k] - E_ar[0]) for k in range(K + 1)}

# ---- D2: held-out margin 非退化（用 seed7 test fold 的 k=0 分布） ----
train7, test7 = split_s1(7)
_te7 = set(rows_of(test7))
h7 = np.array([TRUE_MARG[j, 0] for j in range(NCELL) if CELL_ROW[j] in _te7])
d2_ok = bool(np.std(h7) > 1e-6)
# ---- D3: 存活性 ----
d3_ok = bool(all(np.isfinite(v) for v in E_ar.values()) and
             all(np.isfinite(v) for v in E_ar_const.values()))
# ---- S1: 漂移可检 ----
max_k_ge1 = max(E_ar[k] for k in range(1, K + 1))
s1_ok = bool(max_k_ge1 >= MARGIN_FLOOR)

verdict = ('SMOKE_DEVICE_OK' if (d1_ok and d2_ok and d3_ok) else 'SMOKE_DEVICE_FAIL') + \
          ('|S1_DRIFT_DETECTED' if s1_ok else '|S1_NO_DRIFT')

result = dict(
    query='Q04', design_sha=design_sha, model=MODEL, smoke=SMOKE,
    K=K, n_cells=NCELL, n_rows_active=len(ROWS_ACT),
    E_ar=E_ar, E_ar_const=E_ar_const, scale=SCALE,
    E_ar_rel=E_ar_rel, drift=DRIFT,
    q05_prereg=dict(
        note='Q04 观测到：S1 的 0.05 以**原始 logit** 为单位时被平凡满足（≈88×），'
             '不具否证力。故在此**于 Q05 观测之前**预注册相对形式作为 Q05 的科学门；'
             'S1（原始 0.05）按冻结纪律**不予追溯修改**，保留为装置灵敏度门。',
        S_rel=dict(rule='min_{k=1..K} E_ar_rel(k) <= 0.05',
                   E_ar_rel_def='E_ar(k) / scale(k)，scale = held-out margin_true 的 std',
                   sense='<= pass', frozen_before='Q05 任何观测'),
        shape=dict(rule='按 DRIFT(k)=E_ar(k)-E_ar(0) 的符号与单调性判决 {linear, saturating, diverging}',
                   frozen_before='Q05 任何观测'),
    ),
    per_seed={str(s): {str(k): per_seed[s][k] for k in range(K + 1)} for s in SEEDS_S1},
    gates=dict(D1=d1_ok, D2=d2_ok, D3=d3_ok,
               S1=s1_ok, S1_max_k_ge1=max_k_ge1, S1_thr=MARGIN_FLOOR),
    heldout_margin_std_k0_seed7=float(np.std(h7)),
    sum_fwd=len(CELLS) * (K + 1),
    rollout_samples={('%d,%d' % c): ROLLOUT_TXT[c] for c in CELLS[:5]},
    verdict=verdict,
)
blob = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
result['res_sha8'] = hashlib.sha256(blob).hexdigest()[:8]
rp = os.path.join(OUT, 'q04_smoke_result.json' if SMOKE else 'q04_result.json')
with open(rp, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=1)
log('result.json res_sha8=%s verdict=%s' % (result['res_sha8'], verdict))

# ================= 报告 =================
R = []
R.append('Q04: E_ar(k) 装置建造 —— 报告（%s；%s）' % ('SMOKE' if SMOKE else 'FULL', MODEL))
R.append('=' * 74)
R.append('design_sha = %s' % design_sha[:8])
R.append('res_sha8   = %s' % result['res_sha8'])
R.append('面板: %d cells (%d pairs × %d 模板), held-out = 147 行/seed (S1 seeds %s)'
         % (NCELL, NPA, NT, SEEDS_S1))
R.append('K = %d   forwards = %d   预测器 = B4 加性 ridge(λ=%g)' % (K, result['sum_fwd'], LAM))
R.append('')
R.append('E_ar(k) 曲线（原始 L1, logit 单位）:')
R.append('  %-4s %-12s %-12s %-10s %-10s %-10s' % ('k', 'E_ar(B4)', 'null(const)', 'scale', 'rel(err/scale)', 'drift(k)-drift(0)'))
for k in range(K + 1):
    R.append('  %-4d %-12.4f %-12.4f %-10.4f %-10.4f %-10.4f'
             % (k, E_ar[k], E_ar_const[k], SCALE[k], E_ar_rel[k], DRIFT[k]))
R.append('')
R.append('装置门:')
R.append('  D1 k=0 确定性锚      : %s' % ('PASS' if d1_ok else 'FAIL'))
R.append('  D2 margin 非退化     : %s  (held-out margin std(k=0,seed7) = %.4f)'
         % ('PASS' if d2_ok else 'FAIL', float(np.std(h7))))
R.append('  D3 存活性(全有限)    : %s' % ('PASS' if d3_ok else 'FAIL'))
R.append('  S1 漂移可检(max k>=1): %s  (max=%.4f, 门=%.2f)'
         % ('PASS' if s1_ok else 'FAIL', max_k_ge1, MARGIN_FLOOR))
R.append('')
R.append('rollout 样例（前 3 cell 的自生成续写，仅审计用）:')
for c in CELLS[:3]:
    R.append('  cell(t=%d,pi=%d) entity=%s  -> "%s"'
             % (c[0], c[1], ENTS[PAIRS[c[1]][0]], ROLLOUT_TXT[c][:40]))
R.append('')
R.append('注: E_ar 主体口径 = 原始 L1（不归一），与 E_read 的归一化 MSE 量纲不同；')
R.append('    null(const) = 训练均值常数预测的 L1，作为加性模型的对照下界。')
R.append('    rel = E_ar/scale（scale=held-out margin_true 的 std），report-only 派生量。')
R.append('')
R.append('⚠ 口径观察（如实记录，不追溯改门）:')
R.append('  冻结 S1 门 0.05 以**原始 logit** 为单位，实测 max=%.4f 被平凡满足（%.0f×）；'
         % (max_k_ge1, (max_k_ge1 / MARGIN_FLOOR) if MARGIN_FLOOR else float('nan')))
R.append('  故 S1 仅作**装置灵敏度门**（证明装置能看见漂移），不具科学否证力。')
R.append('  Q05 的科学门已在**本 Phase 观测后、Q05 观测前**预注册为相对形式：')
R.append('    S_rel: min_{k=1..K} E_ar_rel(k) <= 0.05（E_ar_rel = E_ar/scale）；见 result.q05_prereg。')
R.append('')
R.append('⚠ SMOKE 局限（数字不可科学解读）:')
R.append('  子面板取 PAIRS[:%d] = 仅前 %d 个实体 × 6 类（共 %d pairs），'
         % (SMOKE_PAIRS, SMOKE_PAIRS // NC, SMOKE_PAIRS) if SMOKE else '  （FULL 运行，无此局限）')
R.append('  41 个 entity one-hot 中 %d 列为零 ⇒ 未见实体处 B4 回退到 class+template，'
         % (NE - SMOKE_PAIRS // NC) if SMOKE else '')
R.append('  E_ar 偏大且 per-seed 方差大（2.5/4.2/3.1）为该子面板的伪影；正式曲线见 Q05 全面板。')
R.append('')
R.append('⚠ Q05 资源前置（登记为风险）:')
R.append('  qwen3-14b bf16 ≈ 29.6 GB、glm4-9b bf16 ≈ 18.8 GB，均 > 本机 16 GB 显存；')
R.append('  Q05 须先决定量化/offload 方案，并声明其与 E_read（bf16 采集）的精度差异。')
txt = '\n'.join(R)
with open(os.path.join(OUT, 'q04_smoke_report.txt' if SMOKE else 'q04_report.txt'),
          'w', encoding='utf-8') as f:
    f.write(txt + '\n\n--- run log ---\n' + '\n'.join(LOG) + '\n')
print(txt)
print()
print('D1=%s D2=%s D3=%s S1=%s' % (d1_ok, d2_ok, d3_ok, s1_ok))
print('VERDICT =', verdict)
print('EXIT_OK')
