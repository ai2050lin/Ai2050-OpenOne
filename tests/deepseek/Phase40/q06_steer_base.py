# -*- coding: utf-8 -*-
"""Q06 C_steer 基座测量（v1 承重轴 + 端口替换）—— Phase 40 主脚本 v2。

预注册：tests/deepseek/result/q06_prereg_design_v1.json（design_sha8=ebf960cf，2026-10-03 22:36 冻结）。
annex v2 修订（SMOKE v1 后、正式运行前冻结；坑 26/28 合规路径）：
  R1 t 规则：t = mu+sgn*alpha*sigma -> t = s + sgn*alpha*sigma（相对当前读出口分量的 push/pull，
     替换算子形式 h+(t-s)v 保持；对齐 M14 剂量幅度语义。理由：SMOKE v1 显示绝对值规则对
     z=(s-mu)/sigma≈0 的 cell 替换动量退化为近零，argmax 0/72 移动）。
  R2 target 规则：c' = base 第二高类 -> c' = true_class（cell 真值类，per-cell 特异）；
     eligible = base argmax != true_class。理由：SMOKE v1 显示 6 类先验恒水果最高
     （base argmax==true 仅 8.3%），第二高类目标退化为常量。
  R3 G2 门语义：liveness(>=1 成功) -> computability（全部配置成功率有限且 in [0,1]）；
     新增 G2b 灵敏度描述（argmax 移动计数 + maxd 分布；maxd 全程 <0.05 才 fail）。
  未改动：collateral 定义 v1（差分）、预注册 KPI 公式、alpha 五点、方向减法禁令、面板。
算子：readout-substitution（端口替换/读出替换）—— h <- h + (t - h.v)v = h + dt*v，禁止方向减法。
对照：identity 恢复（硬断言逐位恒等）+ 随机方向同规则（范数匹配 + 同替换规则）。
面板：E_read 同 held-out 面板族（S1 seeds [7,8,9] frac=0.2 -> 147 held-out 行/seed）。
精度：qwen3-4b bf16；单模型单进程（坑 54）。
"""
import os, sys, json, time, hashlib
import numpy as np

T0 = time.time()
SMOKE = os.environ.get('SMOKE', '0') == '1'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(ROOT, 'tests', 'deepseek', 'result')
TEMP = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase40')
TAG = 'smoke' if SMOKE else 'formal'
LOGP = os.path.join(TEMP, TAG, '_q06_stdout.log') if SMOKE else os.path.join(TEMP, '_q06_stdout.log')
if SMOKE:
    os.makedirs(os.path.dirname(LOGP), exist_ok=True)
LOG = []

def log(s):
    ln = '[%7.1f] %s' % (time.time() - T0, s)
    LOG.append(ln)
    with open(LOGP, 'a', encoding='utf-8') as f:
        f.write(ln + '\n')
    try:
        print(ln, flush=True)
    except Exception:
        pass

def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

# ---------------- 冻结面板材料（与 3151/3152/Q03/Q04/Q05 逐字一致） ----------------
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
TPL_P0 = {0: '{e}是一种',
          1: '{e}属于',
          2: '{e}，一种常见的'}
SEEDS_S1 = [7, 8, 9]
FRAC_S1 = 0.2
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

# ---------------- Q06 冻结参数（annex v2；观测前不得改） ----------------
MODEL = 'qwen3-4b'
MDIR = os.path.join(ROOT, 'models', 'hf', MODEL)
LAY_STEER = 29
ALPHAS = [0.05, 0.10, 0.15, 0.25, 0.50]
SGNS = [+1, -1]
AXIS_SEED = 7
RAND_SEED = 20261007
N_PROBE = 13

# ---------------- annex design v2（冻结域，无时间戳） ----------------
prereg_p = os.path.join(OUT, 'q06_prereg_design_v1.json')
PREREG_SHA = sha8(prereg_p)
design = dict(
    query='Q06', title='C_steer 基座测量（v1 承重轴 + 端口替换）',
    version=2,
    revise_log=[
        'v1->v2 (SMOKE v1 后、正式前冻结)：R1 t 规则 mu+sgn*alpha*sigma -> s+sgn*alpha*sigma '
        '(SMOKE v1 argmax 0/72 移动，绝对值规则对 z≈0 cell 动量退化)；'
        'R2 target cprime=base第二高类(退化为常量, base argmax==true 仅 8.3%) -> true_class；'
        'R3 G2 门 liveness -> computability + G2b 灵敏度描述。'
        '未改动：collateral 定义 v1、KPI 公式、alpha 五点、方向减法禁令、面板。'],
    model=MODEL, prec='bf16', smoke=SMOKE,
    prereg_path='tests/deepseek/result/q06_prereg_design_v1.json',
    prereg_sha8=PREREG_SHA,
    panel=dict(classes=CLASSES, ents=ENTS, tpl_p0=TPL_P0,
               n_entities=NE, n_classes=NC, n_pairs=NP_, n_tpl=NT,
               seeds_s1=SEEDS_S1, frac_s1=FRAC_S1,
               heldout_rows_per_seed=int(round(FRAC_S1 * NP_)) * NT,
               engine_note='与 E_read 同一 held-out 面板族（S1 fold）'),
    v1_axis=dict(
        site='L29 层输出（残差流 post-block）；qwen3-4b NL=36',
        extract=('seed7 train fold（197 pairs x 3 tpl = 591 行）TPL_P0 前缀 last-position '
                 'L29 残差流 H；加性分解 X=[onehot(41)+onehot(6)+onehot(3)+bias]，'
                 'ridge(lambda=1e-3) 拟合；交互残差 R=H-Xb；'
                 'v1 = SVD(R) 第一右奇异向量（2560 维单位向量），符号约定 sum(R@v1)>=0；'
                 '与 gpt5 线 M14（GLM4 T4 面板）为同构移植定义（跨模型坐标不对应，AGENTS.md §3）'),
        mu_sigma='mu29/sigma29 = seed7 train fold 上 s=H@v1 的 mean/std（float64）',
        rand_dir='default_rng(20261007).normal(2560) 归一化；mu_r/sigma_r 同一批 H 上 s=H@vrand 的 mean/std',
    ),
    operator=dict(
        name='readout-substitution（端口替换）',
        formula='h <- h + (t - h.v)*v = h + dt*v, dt = sgn*alpha*sigma；禁方向减法'
                '（M15 cancel 0.820 反向加重；metric_dict.intervention_rules）',
        t_rule='t = s + sgn*alpha*sigma（v2；python float64 标量、fp32 乘 v 后 cast bf16 写回单点）',
        site_pos='行为 prompt = last-position；collateral 拼接 prompt = 主体句末位（探针段经注意力被波及）',
        alphas=ALPHAS, sgns=SGNS,
        arms=['base', 'identity', 'steer(sgn x alpha, 10)', 'rand(sgn x alpha, 10)'],
    ),
    behaviour=dict(
        readout='单次前向 last-position 6 类类名首 token logits（Q04 k=0 同构；prompt=TPL_P0 逐字）',
        target="c' = true_class（cell 真值类，per-cell 特异；v2）；eligible = base argmax != true_class",
        success='argmax_after == true_class 且 collateral == 0（仅 eligible 计入分母）',
    ),
    collateral=dict(
        n_probes=N_PROBE,
        probe_rule=('default_rng(20261007) 从 seed7 train fold pairs 抽 13 个（replace=False）；'
                    '句式 TPL_P0[0]+{c}。；探针读数位 = 各探针 P0 段末位'),
        concat='主体 P0 + [探针 k P0 + {c_k} + 。] x13（一次前向，多读数位）',
        err='探针错误 = 该读数位 6 类 argmax != 探针真值类',
        diff='collateral(cell,arm) = 干预后 13 探针错误数 - 同 cell 无干预拼接基线错误数（>0 记附带损伤）',
        note='v1 操作化定义；err_base 为 6 类 argmax 口径基线（SMOKE 实测 ~11.75/13，'
             '历史 3.5/13 口径的源文件 LOOP_DIAGNOSIS_AND_EXIT_v1.md 已失传，不可逐字复刻）',
    ),
    stats=dict(
        aggregation='per seed: mean over eligible held-out rows; then mean over 3 seeds（与 E_read 同族）',
        main_readout='C_steer_main = max over 10 steer configs 的池化成功率（机制族乐观上界读数）；'
                     'rand 对照同口径同报；差分 = 机制特异性',
        ci='per-seed Wilson 95% + MDE80 = 2.80*sqrt(p(1-p)/n)',
        gate='本 Phase 不设通过门（metric_dict v4：低分即诚实总成绩，不得事后改判）',
    ),
    reporting=dict(
        must_report=['E_read', 'E_ar(k)', 'C_steer'],
        rule='I1：未降低任一全局 KPI => Ledger 登记 catalog，不得登记 advance',
        refs=dict(E_read='0.3316153089205424 (Q03 锁定)', E_ar='Q05 measured (aggregate_res_sha8=775d7dce)'),
    ),
)
PANEL_SHA = hashlib.sha256(
    json.dumps([CLASSES, ENTS, [TPL[k] for k in sorted(TPL)], SEEDS_S1, FRAC_S1],
               ensure_ascii=False).encode('utf-8')).hexdigest()[:8]
design['panel_sha8'] = PANEL_SHA
assert PANEL_SHA == 'be17ef8a', 'PANEL DRIFT: %s' % PANEL_SHA
assert PREREG_SHA == 'ebf960cf', 'PREREG DRIFT: %s' % PREREG_SHA
eblob = json.dumps(design, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
DESIGN_SHA = hashlib.sha256(eblob).hexdigest()
design['design_sha'] = DESIGN_SHA

exe_p = os.path.join(OUT, 'q06_smoke_execution.json' if SMOKE else 'q06_execution.json')
if os.path.exists(exe_p):
    prev = json.load(open(exe_p, encoding='utf-8'))
    assert prev['design_sha'] == DESIGN_SHA, 'DESIGN DRIFT vs frozen execution.json'
    log('execution.json match (sha %s)' % DESIGN_SHA[:8])
else:
    with open(exe_p, 'w', encoding='utf-8') as f:
        json.dump(design, f, ensure_ascii=False, indent=1)
    log('execution.json FROZEN design_sha=%s smoke=%s' % (DESIGN_SHA[:8], SMOKE))

# ================= 模型观测 =================
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
log('model load begin: %s (bf16)' % MDIR)
model = AutoModelForCausalLM.from_pretrained(
    MDIR, dtype=torch.bfloat16, trust_remote_code=True).to('cuda').eval()
NL = model.config.num_hidden_layers
assert NL == 36 and LAY_STEER < NL - 1, (NL, LAY_STEER)
log('model loaded NL=%d  device_mem=%.1f GB' % (NL, torch.cuda.memory_allocated() / 2**30))

def ids_of(text):
    return tok(text, add_special_tokens=False)['input_ids']

CLS_TOK = [ids_of(c)[0] for c in CLASSES]
assert len(set(CLS_TOK)) == NC, 'F1b class first tokens collide'
log('CLS_TOK=%s' % CLS_TOK)
CLS_T = torch.tensor(CLS_TOK, device='cuda')

# ---------------- L29 hook（读出替换；v2 语义 dt = sgn*alpha*sigma） ----------------
STATE = {'mode': 'off', 'pos': 0, 'dt': 0.0, 'v': None, 'hits': 0, 'cap': None}

def l29_hook(module, inp, out):
    h = out[0] if isinstance(out, tuple) else out
    if STATE['mode'] == 'off':
        return out
    pos = STATE['pos']
    hf = h[0, pos].float()
    v = STATE['v']                      # fp32 (2560,) on cuda
    if STATE['mode'] == 'capture':
        if STATE['cap'] is not None:
            STATE['cap'].append(hf.detach().cpu().numpy())
        return out
    if STATE['mode'] == 'identity':
        hf2 = hf                        # t = s：精确零增量（bf16 往返保真）
    else:
        s = float(hf @ v)               # t = s + dt（端口替换：替换为 s+dt）
        hf2 = hf + (s + STATE['dt'] - s) * v
    h2 = h.clone()
    h2[0, pos] = hf2.to(h.dtype)
    STATE['hits'] += 1
    return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2

model.model.layers[LAY_STEER].register_forward_hook(l29_hook)

def six_cls_logits(ids, steer=None):
    """单次前向返回读数位 6 类 logit（steer=(mode,pos,dt,v) 时施加替换）。"""
    STATE['mode'] = 'off'; STATE['cap'] = None
    if steer is not None:
        mode, pos, dt, v = steer
        STATE['mode'] = mode; STATE['pos'] = pos; STATE['dt'] = dt; STATE['v'] = v; STATE['hits'] = 0
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids], device='cuda'))
    lg = out.logits[0, -1 if steer is None else steer[1], :].float()
    cl = lg[CLS_T].detach().cpu().numpy()
    STATE['mode'] = 'off'
    return cl

# ---------------- v1 轴抽取（seed7 train fold；零 held-out 接触） ----------------
train7, test7 = split_s1(AXIS_SEED)
tr_rows = rows_of(train7)
log('v1 axis extract: seed%d train rows=%d' % (AXIS_SEED, len(tr_rows)))

CAP = []
STATE['mode'] = 'capture'; STATE['cap'] = CAP
with torch.no_grad():
    for r in tr_rows:
        t_, pi_ = r // NP_, r % NP_
        ids = ids_of(TPL_P0[t_].format(e=ENTS[PAIRS[pi_][0]]))
        STATE['pos'] = len(ids) - 1
        model(input_ids=torch.tensor([ids], device='cuda'))
STATE['mode'] = 'off'; STATE['cap'] = None
H = np.stack(CAP).astype(np.float64)
assert H.shape == (len(tr_rows), model.config.hidden_size), H.shape
log('H captured: %s  std_per_coord med=%.4f' % (H.shape, float(np.median(H.std(0)))))

ncol = NE + NC + NT + 1
def rowvec(t_, pi_):
    i, c = PAIRS[pi_]
    v = np.zeros(ncol, np.float64)
    v[i] = 1.0; v[NE + c] = 1.0; v[NE + NC + t_] = 1.0; v[-1] = 1.0
    return v
X = np.stack([rowvec(r // NP_, r % NP_) for r in tr_rows])
LAM = 1e-3
Beta = np.linalg.solve(X.T @ X + LAM * np.eye(ncol), X.T @ H)
Rres = H - X @ Beta
U_, S_, Vt = np.linalg.svd(Rres, full_matrices=False)
v1 = Vt[0].copy()
if float(np.sum(Rres @ v1)) < 0:
    v1 = -v1
sv_share = float(S_[0] ** 2 / np.sum(S_ ** 2))
s_all = H @ v1
MU29, SIG29 = float(s_all.mean()), float(s_all.std())
log('v1 axis: sv_share=%.4f  mu29=%.4f  sigma29=%.4f  |v1|=%.6f'
    % (sv_share, MU29, SIG29, float(np.linalg.norm(v1))))
assert abs(float(np.linalg.norm(v1)) - 1.0) < 1e-9, 'v1 not unit'
assert SIG29 > 0, 'F7 sigma degenerate'

rng = np.random.default_rng(RAND_SEED)
vr = rng.normal(size=(model.config.hidden_size,))
vr = vr / np.linalg.norm(vr)
s_r = H @ vr
MU_R, SIG_R = float(s_r.mean()), float(s_r.std())
cos_vr = float(abs(np.dot(v1, vr)))
log('rand dir: mu_r=%.4f sigma_r=%.4f  |cos(v1,vr)|=%.4f' % (MU_R, SIG_R, cos_vr))
assert cos_vr < 0.2, 'F6 rand dir too aligned: %.3f' % cos_vr

V1_T = torch.tensor(v1, dtype=torch.float32, device='cuda')
VR_T = torch.tensor(vr, dtype=torch.float32, device='cuda')

# ---------------- 探针 13 个（seed7 train fold；排除全部 held-out） ----------------
probe_pool = [p for p in PAIRS if p in train7]
pidx = rng.choice(len(probe_pool), N_PROBE, replace=False)
PROBES = [probe_pool[int(j)] for j in pidx]
PROBE_SEG = []
for (pi_k, c_k) in PROBES:
    p0k = ids_of(TPL_P0[0].format(e=ENTS[pi_k]))
    PROBE_SEG.append((p0k + ids_of(CLASSES[c_k]) + ids_of('。'), len(p0k) - 1, c_k))
log('probes: %s' % (PROBES,))

def concat_prompt(t_, pi_):
    main = ids_of(TPL_P0[t_].format(e=ENTS[PAIRS[pi_][0]]))
    ids = list(main)
    probe_pos, probe_c = [], []
    for seg, plen, ck in PROBE_SEG:
        probe_pos.append(len(ids) + plen)
        probe_c.append(ck)
        ids = ids + seg
    return ids, len(main) - 1, probe_pos, probe_c

# ---------------- cells ----------------
if SMOKE:
    rows7 = rows_of(test7)[:12]
    CELLS = [(AXIS_SEED, r // NP_, r % NP_) for r in rows7]
else:
    CELLS = []
    for s_ in SEEDS_S1:
        _, te = split_s1(s_)
        for r in rows_of(te):
            CELLS.append((s_, r // NP_, r % NP_))
log('cells=%d  arms=22  prompts=2/cell  forwards=%d' % (len(CELLS), len(CELLS) * 44))

STEER_CFG = [('steer', sg, a) for sg in SGNS for a in ALPHAS] + \
            [('rand', sg, a) for sg in SGNS for a in ALPHAS]

RES = {}
t_last = time.time()
for n_, (s_, t_, pi_) in enumerate(CELLS):
    main_ids = ids_of(TPL_P0[t_].format(e=ENTS[PAIRS[pi_][0]]))
    pos_last = len(main_ids) - 1
    true_cls = int(PAIRS[pi_][1])
    c_ids, cm_pos, probe_pos, probe_c = concat_prompt(t_, pi_)

    beh_base = six_cls_logits(main_ids)
    am_base = int(np.argmax(beh_base))
    eligible = (am_base != true_cls)

    col_base = six_cls_logits(c_ids, steer=None)
    STATE['mode'] = 'off'; STATE['cap'] = None
    with torch.no_grad():
        out_c = model(input_ids=torch.tensor([c_ids], device='cuda'))
    lgc = out_c.logits[0].float()
    probe_base = [lgc[p, CLS_T].detach().cpu().numpy() for p in probe_pos]
    err_base = int(sum(int(np.argmax(pb)) != ck for pb, ck in zip(probe_base, probe_c)))

    row = dict(true_class=true_cls, err_base=err_base,
               beh_base=[float(x) for x in beh_base],
               beh_argmax_base=am_base, eligible=bool(eligible))
    for (kind, sg, al) in STEER_CFG:
        v = V1_T if kind == 'steer' else VR_T
        sig = SIG29 if kind == 'steer' else SIG_R
        dtv = float(sg) * al * sig
        beh = six_cls_logits(main_ids, steer=('replace', pos_last, dtv, v))
        STATE['mode'] = 'replace'; STATE['pos'] = cm_pos; STATE['dt'] = dtv; STATE['v'] = v; STATE['hits'] = 0
        with torch.no_grad():
            out_ci = model(input_ids=torch.tensor([c_ids], device='cuda'))
        lgc2 = out_ci.logits[0].float()
        probe_after = [lgc2[p, CLS_T].detach().cpu().numpy() for p in probe_pos]
        STATE['mode'] = 'off'
        err_after = int(sum(int(np.argmax(pb)) != ck for pb, ck in zip(probe_after, probe_c)))
        key = '%s|%+d|%.2f' % (kind, sg, al)
        row[key] = dict(beh=[float(x) for x in beh],
                        argmax=int(np.argmax(beh)),
                        maxd=float(np.max(np.abs(beh - beh_base))),
                        collat=err_after - err_base,
                        hook_hits=STATE['hits'])
    beh_id = six_cls_logits(main_ids, steer=('identity', pos_last, 0.0, V1_T))
    row['identity_maxd'] = float(np.max(np.abs(beh_id - beh_base)))
    STATE['mode'] = 'identity'; STATE['pos'] = cm_pos; STATE['v'] = V1_T; STATE['hits'] = 0
    with torch.no_grad():
        out_id = model(input_ids=torch.tensor([c_ids], device='cuda'))
    STATE['mode'] = 'off'
    lgi = out_id.logits[0].float()
    probe_id = [lgi[p, CLS_T].detach().cpu().numpy() for p in probe_pos]
    row['identity_maxd_concat'] = float(max(
        float(np.max(np.abs(p2 - pb))) for p2, pb in zip(probe_id, probe_base)))
    RES['%d|%d|%d' % (s_, t_, pi_)] = row
    if (n_ + 1) % 50 == 0:
        done = n_ + 1
        log('cells %d/%d  (%.2f s/cell)  id_maxd=%.2e' %
            (done, len(CELLS), (time.time() - t_last) / done, row['identity_maxd']))

log('forward loop done: %.1f min' % ((time.time() - t_last) / 60.0))

# ---------------- 聚合与判据 ----------------
F1 = max(max(r['identity_maxd'], r['identity_maxd_concat']) for r in RES.values())
F1_ok = (F1 == 0.0)
hook_ok = all(r[k]['hook_hits'] == 1 for r in RES.values()
              for k in r if isinstance(r[k], dict) and 'hook_hits' in r[k])

def pool_success(kind, sg, al):
    key = '%s|%+d|%.2f' % (kind, sg, al)
    per_seed = {}
    for s_ in SEEDS_S1:
        cells_s = [k for k in RES if k.startswith('%d|' % s_)]
        elig = [k for k in cells_s if RES[k]['eligible']]
        if not elig:
            continue
        hits = sum(1 for k in elig
                   if RES[k][key]['argmax'] == RES[k]['true_class'] and RES[k][key]['collat'] == 0)
        per_seed[str(s_)] = dict(num=hits, den=len(elig), rate=hits / len(elig))
    tot_n = sum(v['den'] for v in per_seed.values())
    tot_h = sum(v['num'] for v in per_seed.values())
    return dict(per_seed=per_seed, pool_rate=(tot_h / tot_n if tot_n else None),
                pool_num=tot_h, pool_den=tot_n)

steer_curves = {}
for sg in SGNS:
    for al in ALPHAS:
        steer_curves['steer|%+d|%.2f' % (sg, al)] = pool_success('steer', sg, al)
rand_curves = {}
for sg in SGNS:
    for al in ALPHAS:
        rand_curves['rand|%+d|%.2f' % (sg, al)] = pool_success('rand', sg, al)

def wilson(h, n, z=1.96):
    if not n:
        return None
    p = h / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - hw), float(c + hw)]

best_steer = max(steer_curves.items(), key=lambda kv: (kv[1]['pool_rate'] or 0))
best_rand = max(rand_curves.items(), key=lambda kv: (kv[1]['pool_rate'] or 0))
C_MAIN = best_steer[1]['pool_rate']
C_RAND = best_rand[1]['pool_rate']

all_collat = [r[k]['collat'] for r in RES.values() for k in r
              if isinstance(r[k], dict) and 'collat' in r[k]]
collat_frac0 = float(np.mean([c == 0 for c in all_collat])) if all_collat else None
err_base_all = [r['err_base'] for r in RES.values()]
n_elig = sum(1 for r in RES.values() if r['eligible'])
maxd_all = [r[k]['maxd'] for r in RES.values() for k in r
            if isinstance(r[k], dict) and 'maxd' in r[k] and k.startswith('steer')]
argmax_moved = sum(1 for r in RES.values() for k in r
                   if isinstance(r[k], dict) and 'argmax' in r[k] and k.startswith('steer')
                   and r[k]['argmax'] != r['beh_argmax_base'])
argmax_tot = len(maxd_all)

n_per_seed = {str(s_): sum(1 for k in RES if k.startswith('%d|' % s_)) for s_ in SEEDS_S1}
n_eff = max((v['den'] for v in best_steer[1]['per_seed'].values()), default=0)
p0 = C_MAIN if C_MAIN is not None else 0.0
MDE80 = float(2.80 * np.sqrt(max(p0 * (1 - p0), 1e-9) / max(n_eff, 1)))

G1 = F1_ok
G2 = all(v['pool_rate'] is not None or v['pool_den'] == 0 for v in steer_curves.values())
G2b = (min(maxd_all) > 0.0) if maxd_all else False
G2b_strong = (max(maxd_all) >= 0.05) if maxd_all else False
G3 = all(isinstance(c, int) and -N_PROBE <= c <= N_PROBE for c in all_collat) and len(all_collat) > 0
G4 = (SIG29 > 0) and (cos_vr < 0.2)

result = dict(
    query='Q06', mode='SMOKE' if SMOKE else 'FULL', annex_version=2,
    model=MODEL, prec='bf16',
    prereg_sha8=PREREG_SHA, panel_sha8=PANEL_SHA, design_sha=DESIGN_SHA,
    v1_axis=dict(site=LAY_STEER, sv_share=sv_share, mu29=MU29, sigma29=SIG29,
                 mu_r=MU_R, sigma_r=SIG_R, abs_cos_v1_rand=cos_vr,
                 train_rows=len(tr_rows), vhat_sha8=hashlib.sha256(
                     v1.astype(np.float64).tobytes()).hexdigest()[:8]),
    cells=dict(per_seed=n_per_seed, total=len(CELLS), eligible=n_elig,
               base_argmax_eq_true=float(np.mean([not r['eligible'] for r in RES.values()]))),
    floors=dict(F1_identity_maxd=F1, F1_ok=F1_ok, F2_panel=PANEL_SHA == 'be17ef8a',
                F3_prereg=PREREG_SHA == 'ebf960cf', F5_unit=True,
                F6_cos_lt02=cos_vr < 0.2, F7_sigma_pos=SIG29 > 0,
                hook_hits_all_1=hook_ok),
    steer_curves=steer_curves, rand_curves=rand_curves,
    sensitivity=dict(argmax_moved=argmax_moved, argmax_total=argmax_tot,
                     maxd_min=float(min(maxd_all)) if maxd_all else None,
                     maxd_max=float(max(maxd_all)) if maxd_all else None),
    C_steer_main=dict(rule='max over 10 steer configs pool_rate (eligible only)',
                      config=best_steer[0], value=C_MAIN,
                      wilson=wilson(best_steer[1]['pool_num'], best_steer[1]['pool_den']),
                      rand_config=best_rand[0], rand_value=C_RAND,
                      spec_diff=(None if (C_MAIN is None or C_RAND is None) else C_MAIN - C_RAND),
                      MDE80=MDE80),
    collateral=dict(mean=float(np.mean(all_collat)) if all_collat else None,
                    max=int(np.max(all_collat)) if all_collat else None,
                    frac_zero=collat_frac0,
                    err_base_mean=float(np.mean(err_base_all)) if err_base_all else None,
                    floor_proxy_ref='3.5/13 (metric_dict.proxy_evidence; 口径失传，本轮 6 类 argmax 操作化)'),
    gates_smoke=dict(G1_identity=G1, G2_computability=G2, G2b_sensitivity=[G2b, G2b_strong],
                     G3_collat_int=G3, G4_nondegen=G4),
    kpi_report=dict(E_read=0.3316153089205424, E_ar='Q05 measured (775d7dce)',
                    C_steer=C_MAIN,
                    note='E_read/E_ar 引用锁定值未变；C_steer 首测为新增读数；I1 => catalog'),
)

res_blob = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
result['res_sha8'] = hashlib.sha256(res_blob).hexdigest()[:8]
res_p = os.path.join(OUT, 'q06_smoke_result.json' if SMOKE else 'q06_result.json')
with open(res_p, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=1)
det_p = os.path.join(TEMP, TAG, 'q06_cells_detail.json') if SMOKE else os.path.join(TEMP, 'q06_cells_detail.json')
with open(det_p, 'w', encoding='utf-8') as f:
    json.dump(RES, f, ensure_ascii=False, indent=1)
np.savez_compressed(os.path.join(TEMP, TAG, 'q06_v1_axis.npz') if SMOKE
                    else os.path.join(TEMP, 'q06_v1_axis.npz'),
                    v1=v1, vr=vr, mu29=MU29, sigma29=SIG29, mu_r=MU_R, sigma_r=SIG_R)

Rp = []
Rp.append('Q06 C_steer 基座测量 %s (annex v2)  res_sha8=%s' % (TAG.upper(), result['res_sha8']))
Rp.append('prereg=%s  panel=%s  design=%s' % (PREREG_SHA, PANEL_SHA, DESIGN_SHA[:8]))
Rp.append('v1 axis: sv_share=%.4f mu29=%.4f sigma29=%.4f |cos(v1,vr)|=%.4f train_rows=%d'
          % (sv_share, MU29, SIG29, cos_vr, len(tr_rows)))
Rp.append('floors: F1_identity_maxd=%.3e (ok=%s)  hook_hits_all_1=%s  F6=%s'
          % (F1, F1_ok, hook_ok, cos_vr < 0.2))
Rp.append('cells: %s  eligible(需真实改动)=%d/%d  base argmax==true=%.3f'
          % (n_per_seed, n_elig, len(CELLS), result['cells']['base_argmax_eq_true']))
Rp.append('')
Rp.append('steer 曲线（池化成功率 = argmax==true_class 且 collateral==0，eligible only）:')
for k in sorted(steer_curves):
    v = steer_curves[k]
    Rp.append('  %-16s %s/%s = %s   (per_seed %s)'
              % (k, v['pool_num'], v['pool_den'],
                 ('%.4f' % v['pool_rate']) if v['pool_rate'] is not None else 'N/A',
                 {kk: '%d/%d' % (vv['num'], vv['den']) for kk, vv in v['per_seed'].items()}))
Rp.append('rand 对照曲线:')
for k in sorted(rand_curves):
    v = rand_curves[k]
    Rp.append('  %-16s %s/%s = %s' % (k, v['pool_num'], v['pool_den'],
                                      ('%.4f' % v['pool_rate']) if v['pool_rate'] is not None else 'N/A'))
Rp.append('')
Rp.append('灵敏度: argmax moved %d/%d  maxd [%.4f, %.4f]' %
          (argmax_moved, argmax_tot,
           min(maxd_all) if maxd_all else -1, max(maxd_all) if maxd_all else -1))
Rp.append('C_steer_main = %s @ %s   rand_same_rule = %s @ %s   spec_diff = %s'
          % (C_MAIN, best_steer[0], C_RAND, best_rand[0], result['C_steer_main']['spec_diff']))
Rp.append('Wilson95=%s  MDE80=%.4f (n=%d/seed)' % (result['C_steer_main']['wilson'], MDE80, n_eff))
Rp.append('collateral: mean=%.3f max=%s frac_zero=%.4f  err_base(拼接基线) mean=%.3f'
          % (result['collateral']['mean'], result['collateral']['max'],
             collat_frac0, result['collateral']['err_base_mean']))
Rp.append('gates(smoke): %s' % result['gates_smoke'])
rep_p = os.path.join(TEMP, TAG, 'q06_report.txt') if SMOKE else os.path.join(TEMP, 'q06_report.txt')
with open(rep_p, 'w', encoding='utf-8') as f:
    f.write('\n'.join(Rp) + '\n')
log('REPORT -> %s' % rep_p)
log('RESULT  -> %s' % res_p)
log('gates: %s' % result['gates_smoke'])
log('sensitivity: moved %d/%d maxd[%.4f,%.4f]' % (argmax_moved, argmax_tot,
                                                  min(maxd_all) if maxd_all else -1,
                                                  max(maxd_all) if maxd_all else -1))
log('C_steer_main=%s @%s rand=%s MDE80=%.4f' % (C_MAIN, best_steer[0], C_RAND, MDE80))
