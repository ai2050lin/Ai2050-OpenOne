# -*- coding: utf-8 -*-
"""
Q05: E_ar(k) 正式测量 —— 三模型全面板曲线（B 闸门；gpu=mid）
========================================================================================
依据 : RDC_RESEARCH_CONSTITUTION_v1 §1 (I1)
       research/deepseek/atlas/phase_queue_v1.json -> Q05
       tests/deepseek/result/q04_smoke_result.json   （Q04 装置 + 预注册 S_rel/shape）
       research/deepseek/atlas/metric_dict.json v3   （E_ar 口径 device_built）

本脚本 = Q04 装置的**逐字复制**（面板/划分/margin/B4/rollout 一字不改），仅新增：
  (1) 精度策略 PREC ∈ {bf16, nf4}：14B/9B 因 16 GB 显存无法 bf16 常驻，改用 4-bit NF4；
  (2) 装置等价性门 D0：SMOKE + bf16 + qwen3-4b 时必须逐位复现 Q04 SMOKE（证明采集器未漂移）；
  (3) 精度桥 D4（在聚合阶段计算）：同模型 bf16 vs nf4 的 E_ar_rel 差。
纪律 : 面板指纹必须等于 Q04 的 panel_sha8（读 q04_execution.json 断言）；预注册先冻结再观测；
       数字一律现场渲染；产物按 arm 分离（q05_<model>_<prec>_*.json）。

用法 : python q05_e_ar_measure.py <model> <prec>     # model ∈ qwen3-4b|qwen3-14b|glm4-9b ; prec ∈ bf16|nf4
       P_Q05_SMOKE=1 环境变量 -> 子面板 SMOKE
"""
import os, sys, json, time, hashlib
import numpy as np

T0 = time.time()
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

MODEL = os.environ.get('P_Q05_MODEL') or (sys.argv[1] if len(sys.argv) > 1 else 'qwen3-4b')
PREC = os.environ.get('P_Q05_PREC') or (sys.argv[2] if len(sys.argv) > 2 else 'bf16')
SMOKE = os.environ.get('P_Q05_SMOKE') == '1'
assert PREC in ('bf16', 'nf4'), PREC
ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(ROOT, 'tests', 'deepseek', 'result')
os.makedirs(OUT, exist_ok=True)
ARM = '%s__%s' % (MODEL, PREC)
SUF = '_smoke' if SMOKE else ''
LOGP = os.path.join(OUT, 'q05_run_%s%s.txt' % (ARM, SUF))
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

# ---------------- 冻结面板材料（与 3151/3152/Q03/Q04 逐字一致） ----------------
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
TPL_P0 = {0: '{e}是一种', 1: '{e}属于', 2: '{e}，一种常见的'}
TPL_SUF = {0: '。', 1: '这一类。', 2: '。'}
SEEDS_S1 = [7, 8, 9]
FRAC_S1 = 0.2
K_FULL = 16
K_SMOKE = 4
SMOKE_PAIRS = 60
MAXCTX = 64
LAM = 1e-3
MARGIN_FLOOR = 0.05

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
PANEL_SHA8 = hashlib.sha256(
    json.dumps([CLASSES, ENTS, [TPL[k] for k in sorted(TPL)], SEEDS_S1, FRAC_S1],
               ensure_ascii=False).encode('utf-8')).hexdigest()[:8]

# ---- 面板指纹必须与 Q04 冻结值一致 ----
_q04exe = os.path.join(OUT, 'q04_smoke_execution.json')
Q04_PANEL_SHA8 = None
if os.path.exists(_q04exe):
    Q04_PANEL_SHA8 = json.load(open(_q04exe, encoding='utf-8')).get('panel_sha8')
    assert Q04_PANEL_SHA8 == PANEL_SHA8, 'PANEL DRIFT vs Q04: %s != %s' % (PANEL_SHA8, Q04_PANEL_SHA8)

# ================= 预注册（任何模型观测前冻结；不含时间戳） =================
K = K_SMOKE if SMOKE else K_FULL
if SMOKE:
    SMOKE_SET = set(PAIRS[:SMOKE_PAIRS])
    PAIRS_ACT = [p for p in PAIRS if p in SMOKE_SET]
else:
    SMOKE_SET = set()
    PAIRS_ACT = list(PAIRS)
NPA = len(PAIRS_ACT)
ROWS_ACT = [t * NP_ + pi for t in range(NT)
            for pi, p in enumerate(PAIRS) if p in SMOKE_SET] if SMOKE else \
           list(range(NT * NP_))

PREC_POLICY = {
    'bf16': 'GPU-resident bf16（与 E_read 采集同精度；仅 qwen3-4b 可行）',
    'nf4': '4-bit NF4（bnb_4bit_quant_type=nf4, double_quant=True, compute_dtype=bf16）；'
           '14B/9B 因 16 GB 显存无法 bf16 常驻而采用——与 E_read 的 bf16 存在量化差',
}

design = dict(
    query='Q05', title='E_ar(k) 正式测量',
    mode='full-measure + SMOKE' if SMOKE else 'full-measure',
    model=MODEL, prec=PREC, mdir=MDIR, smoke=SMOKE, arm=ARM,
    panel=dict(classes=CLASSES, ents=ENTS, tpl=TPL, tpl_p0=TPL_P0,
               n_entities=NE, n_classes=NC, n_pairs=NP_, n_tpl=NT,
               rows_full=NT * NP_, rows_active=len(ROWS_ACT), pairs_active=NPA,
               seeds_s1=SEEDS_S1, frac_s1=FRAC_S1,
               heldout_rows_per_seed=int(round(FRAC_S1 * NP_)) * NT,
               panel_sha8=PANEL_SHA8, q04_panel_sha8=Q04_PANEL_SHA8,
               engine_note='Q04 采集器逐字复制；面板指纹断言对齐 Q04'),
    device=dict(
        k_range='0..K (k=0 为确定性锚)', K=K,
        roll='greedy self-rollout: ctx0 = P0 token ids; 每步读 last-position 6 类 logit; '
             'k<K 时 apped argmax(logits_last) 回喂',
        margin='logit(t_target) - logit(t_competitor); t_target = 该类首 token; '
               't_competitor = k=0 处 logit 最高的非该类类首 token（每 cell 冻结）',
        predictor='B4 additive: ridge_primal(one-hot[entity]+one-hot[class]+one-hot[template]+bias)',
        lam=LAM, feature_cols=NE + NC + NT + 1,
        target_units='logit（原始 L1，不归一化）',
        normalization='none (frozen formula is raw L1); secondary relative scale reported only',
        seeds=SEEDS_S1, aggregation='per seed: mean over held-out rows; then mean over 3 seeds',
        maxctx=MAXCTX,
    ),
    precision=dict(policy=PREC_POLICY[PREC], prec=PREC,
                   declare='nf4 引入与 E_read bf16 的量化差；由 D4 精度桥约束其可用性'),
    gates=dict(
        D0='collector equivalence: SMOKE+bf16+qwen3-4b 时逐位复现 Q04 SMOKE 的 E_ar(k)',
        D1='k=0 determinism: two passes bit-identical on 6-class logits',
        D2='non-degeneracy: std(margin_true over held-out cells) > 0',
        D3='liveness: all E_ar(k) finite',
        S1=dict(rule='max_{k>=1} E_ar(k) >= MARGIN_FLOOR', thr=MARGIN_FLOOR, sense='>= pass',
                note='装置灵敏度门（Q04 观测后判定其不具科学否证力，保留不动）'),
    ),
    frozen_before='any Q05 model observation',
)
for f in [MDIR + r'\config.json']:
    if os.path.exists(f):
        design['mdir_config_sha8'] = sha8(f)
eblob = json.dumps(design, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
design_sha = hashlib.sha256(eblob).hexdigest()
design['design_sha'] = design_sha

exe_p = os.path.join(OUT, 'q05_%s%s_execution.json' % (ARM, SUF))
if os.path.exists(exe_p):
    prev = json.load(open(exe_p, encoding='utf-8'))
    assert prev['design_sha'] == design_sha, 'DESIGN DRIFT'
    log('execution.json match (sha %s)' % design_sha[:8])
else:
    with open(exe_p, 'w', encoding='utf-8') as f:
        json.dump(design, f, ensure_ascii=False, indent=1)
    log('execution.json FROZEN design_sha=%s arm=%s K=%d rows_active=%d smoke=%s'
        % (design_sha[:8], ARM, K, len(ROWS_ACT), SMOKE))

# ================= 模型观测 =================
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
log('model load begin: %s prec=%s' % (MDIR, PREC))
if PREC == 'bf16':
    model = AutoModelForCausalLM.from_pretrained(
        MDIR, dtype=torch.bfloat16, trust_remote_code=True).to('cuda').eval()
else:
    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        MDIR, quantization_config=bnb, device_map={'': 0}, trust_remote_code=True).eval()
log('model loaded: %s  vram_alloc=%.2f GB' % (type(model).__name__, torch.cuda.memory_allocated() / 1e9))
NL = model.config.num_hidden_layers

def ids_of(text):
    return tok(text, add_special_tokens=False)['input_ids']

CLS_TOK = [ids_of(c)[0] for c in CLASSES]
assert len(set(CLS_TOK)) == NC, 'class first tokens collide'
CLS_T = torch.tensor(CLS_TOK, device='cuda')

CELLS = [(t, pi) for t in range(NT) for pi, p in enumerate(PAIRS)
         if (not SMOKE) or (p in SMOKE_SET)]
P0IDS = {(t, pi): ids_of(TPL_P0[t].format(e=ENTS[PAIRS[pi][0]]))
         for (t, pi) in CELLS}
log('cells=%d  (K=%d, forwards=%d)' % (len(CELLS), K, len(CELLS) * (K + 1)))

NCELL = len(CELLS)
CLSLOG = np.zeros((NCELL, K + 1, NC), np.float32)
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

CLSLOG_D1, _ = rollout(0)
d1_ok = bool(np.array_equal(CLSLOG_D1, CLSLOG[0]))
log('D1 determinism (cell0 re-run bit-identical) = %s' % d1_ok)

del model
torch.cuda.empty_cache()

# ================= margin 与 held-out 误差 =================
TRUE_MARG = np.zeros((NCELL, K + 1), np.float32)
COMPETITOR = np.zeros(NCELL, np.int64)
for j, (t, pi) in enumerate(CELLS):
    i, c = PAIRS[pi]
    base = CLSLOG[j, 0].copy()
    base[c] = -np.inf
    tc = int(np.argmax(base))
    COMPETITOR[j] = tc
    TRUE_MARG[j] = CLSLOG[j, :, c] - CLSLOG[j, :, tc]
CELL_ROW = np.array([t * NP_ + pi for (t, pi) in CELLS])
CELL_PAIR = np.array([pi for (_, pi) in CELLS])
CELL_TPL = np.array([t for (t, _) in CELLS])

def b4_eval(seed, kvec):
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
        res[k] = dict(mae_b4=float(np.abs(pred - true).mean()),
                      mae_const=float(np.abs(Y[tr_local].mean() - true).mean()),
                      scale=float(true.std()), n_rows=int(len(te_local)),
                      mean_true=float(true.mean()))
    return res

per_seed = {}
for s in SEEDS_S1:
    per_seed[s] = b4_eval(s, list(range(K + 1)))
    log('seed %d E_ar_B4[0..%d] = %s' % (s, K, ['%.4f' % per_seed[s][k]['mae_b4'] for k in range(K + 1)]))

E_ar = {k: float(np.mean([per_seed[s][k]['mae_b4'] for s in SEEDS_S1])) for k in range(K + 1)}
E_ar_const = {k: float(np.mean([per_seed[s][k]['mae_const'] for s in SEEDS_S1])) for k in range(K + 1)}
SCALE = {k: float(np.mean([per_seed[s][k]['scale'] for s in SEEDS_S1])) for k in range(K + 1)}
E_ar_rel = {k: float(E_ar[k] / max(SCALE[k], 1e-9)) for k in range(K + 1)}
DRIFT = {k: float(E_ar[k] - E_ar[0]) for k in range(K + 1)}

train7, test7 = split_s1(7)
_te7 = set(rows_of(test7))
h7 = np.array([TRUE_MARG[j, 0] for j in range(NCELL) if CELL_ROW[j] in _te7])
d2_ok = bool(np.std(h7) > 1e-6)
d3_ok = bool(all(np.isfinite(v) for v in E_ar.values()) and
             all(np.isfinite(v) for v in E_ar_const.values()))
max_k_ge1 = max(E_ar[k] for k in range(1, K + 1))
s1_ok = bool(max_k_ge1 >= MARGIN_FLOOR)

# ---- D0: 采集器等价性（仅 SMOKE+bf16+qwen3-4b 时校验） ----
d0 = None
if SMOKE and PREC == 'bf16' and MODEL == 'qwen3-4b':
    q4 = os.path.join(OUT, 'q04_smoke_result.json')
    if os.path.exists(q4):
        q4r = json.load(open(q4, encoding='utf-8'))
        devs = {k: abs(E_ar[k] - q4r['E_ar'][str(k)]) for k in range(K + 1)}
        d0_ok = bool(max(devs.values()) == 0.0)
        d0 = dict(ok=d0_ok, max_abs_dev=float(max(devs.values())),
                  q04_res_sha8=q4r.get('res_sha8'))
        log('D0 collector equivalence vs Q04 SMOKE: %s (max_abs_dev=%.2e)'
            % (d0_ok, d0['max_abs_dev']))

verdict = ('Q05_ARM_OK' if (d1_ok and d2_ok and d3_ok and ((d0 is None) or d0['ok']))
           else 'Q05_ARM_FAIL') + ('|S1_DRIFT_DETECTED' if s1_ok else '|S1_NO_DRIFT')

result = dict(
    query='Q05', design_sha=design_sha, model=MODEL, prec=PREC, arm=ARM, smoke=SMOKE,
    K=K, n_cells=NCELL, n_rows_active=len(ROWS_ACT), panel_sha8=PANEL_SHA8,
    E_ar=E_ar, E_ar_const=E_ar_const, scale=SCALE, E_ar_rel=E_ar_rel, drift=DRIFT,
    per_seed={str(s): {str(k): per_seed[s][k] for k in range(K + 1)} for s in SEEDS_S1},
    gates=dict(D0=d0, D1=d1_ok, D2=d2_ok, D3=d3_ok, S1=s1_ok,
               S1_max_k_ge1=max_k_ge1, S1_thr=MARGIN_FLOOR),
    heldout_margin_std_k0_seed7=float(np.std(h7)),
    sum_fwd=len(CELLS) * (K + 1),
    precision=PREC,
    rollout_samples={('%d,%d' % c): ROLLOUT_TXT[c] for c in CELLS[:5]},
    verdict=verdict,
)
blob = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
result['res_sha8'] = hashlib.sha256(blob).hexdigest()[:8]
rp = os.path.join(OUT, 'q05_%s%s_result.json' % (ARM, SUF))
with open(rp, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=1)
log('result.json res_sha8=%s verdict=%s' % (result['res_sha8'], verdict))

# ================= 报告 =================
R = []
R.append('Q05: E_ar(k) 正式测量 —— arm=%s（%s；%s）' % (ARM, 'SMOKE' if SMOKE else 'FULL', PREC))
R.append('=' * 74)
R.append('design_sha = %s   res_sha8 = %s' % (design_sha[:8], result['res_sha8']))
R.append('面板: %d cells (%d pairs × %d 模板), held-out = 147 行/seed (S1 seeds %s)  panel_sha8=%s'
         % (NCELL, NPA, NT, SEEDS_S1, PANEL_SHA8))
R.append('K = %d   forwards = %d   预测器 = B4 加性 ridge(λ=%g)   精度 = %s'
         % (K, result['sum_fwd'], LAM, PREC))
R.append('')
R.append('E_ar(k) 曲线（原始 L1, logit 单位）:')
R.append('  %-4s %-12s %-12s %-10s %-12s %-12s' % ('k', 'E_ar(B4)', 'null(const)', 'scale', 'rel', 'drift(k)-drift(0)'))
for k in range(K + 1):
    R.append('  %-4d %-12.4f %-12.4f %-10.4f %-12.4f %-12.4f'
             % (k, E_ar[k], E_ar_const[k], SCALE[k], E_ar_rel[k], DRIFT[k]))
R.append('')
R.append('装置门:')
R.append('  D0 采集器等价(Q04)   : %s' % ('PASS' if (d0 and d0['ok']) else ('n/a' if d0 is None else 'FAIL')))
R.append('  D1 k=0 确定性锚      : %s' % ('PASS' if d1_ok else 'FAIL'))
R.append('  D2 margin 非退化     : %s  (std(k=0,seed7)=%.4f)' % ('PASS' if d2_ok else 'FAIL', float(np.std(h7))))
R.append('  D3 存活性(全有限)    : %s' % ('PASS' if d3_ok else 'FAIL'))
R.append('  S1 漂移可检(max k>=1): %s  (max=%.4f, 门=%.2f)' % ('PASS' if s1_ok else 'FAIL', max_k_ge1, MARGIN_FLOOR))
R.append('')
R.append('rollout 样例（前 3 cell）:')
for c in CELLS[:3]:
    R.append('  cell(t=%d,pi=%d) entity=%s -> "%s"' % (c[0], c[1], ENTS[PAIRS[c[1]][0]], ROLLOUT_TXT[c][:40]))
txt = '\n'.join(R)
with open(os.path.join(OUT, 'q05_%s%s_report.txt' % (ARM, SUF)), 'w', encoding='utf-8') as f:
    f.write(txt + '\n\n--- run log ---\n' + '\n'.join(LOG) + '\n')
print(txt)
print()
print('D0=%s D1=%s D2=%s D3=%s S1=%s' % ((d0 or {}).get('ok'), d1_ok, d2_ok, d3_ok, s1_ok))
print('VERDICT =', verdict)
print('EXIT_OK')
