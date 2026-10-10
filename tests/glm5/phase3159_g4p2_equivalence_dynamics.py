# -*- coding: utf-8 -*-
# Phase 3159 (G4-P2): 等价类动力学 —— mid 层注入读出谱方向, 剩余层是保持还是破坏软零性
# 预注册: AGI_GPT5_MEMO Phase 3159（3158 closeout 冻结, 观测前）:
#   假设: 3158 证读出谱平坦(无 inherited 零空间, 预算比 1.03-1.33 全 absent); P2 问
#         动力学是否主动消除软零性 —— 在中间层注入读出谱意义下的 bottom-sigma 方向,
#         剩余层把它旋转回敏感子空间(quotient destroyed, 更强: 各向同性是计算出来的)
#         还是保持(quotient stable)。
#   设计: 锚 = 3157 H 槽 L_mid = round(0.5*NL) 16 行(= 3158 anchor_idx, 同锚复用);
#         GPU 注入前向: hook 替换 last-token 态(块 L_mid-1 输出 = 槽 L_mid, 即进入块
#         L_mid 的残差流), delta = alpha * ||h_mid|| * u, 剩余 NL - L_mid 层传播;
#         u 三臂同 3158: bottom64/top64/rand 各 8(复用 3158 collect.npz 方向, rand 由
#         同 SEED 重放 QR 并断言单位范数); alpha 相对 ||h_mid|| 扫 6 值;
#         KL = Jeffreys float64(数值前向, 非解析); 预算 = log-log 插值首越 KL=0.1,
#         删失 = 网格上限(同 3158 口径)。
#   测: (1) ratio_mid = median(null 预算)/median(rand 预算); 门 >=3 quotient_stable /
#           <3 dynamics_destroyed(10 参照同记);
#       (2) re-emergence: 注入后逐层位移 dh 在 top-64 子空间能量份额曲线
#           share_top(arm, dir, alpha, layer); 汇总 per-arm 均值曲线; 装置锚:
#           槽 < L_mid dh 恒 0; bottom 臂 share_top(L_mid) < 0.05; top 臂 > 0.95;
#           re_gain_bottom = share_top(NL) - share_top(L_mid);
#       (3) 预算传递 pass-through = med_mid / med_kout(3158 result, 描述性无门)。
#   确定性: 基线前向两次 bitwise(logits + 全层 last-token 态); 锚态 vs 3157
#           H[r0, L_mid] fp16 bitwise; 行重建 vs 3157 npz (ei/rel/pol/ctx) 逐元素一致;
#           方向单位范数断言; bottom/top 正交装置锚。
#   summary: 指纹 = mid KL 曲线(log10, null+rand 拼接) 3 对 + re-emergence bottom 臂
#            share_top 曲线(截 min NH) 3 对, 各 Pearson >= 0.8。
# 教训内置: SMOKE 目录分离; design 全 str 键 JSON; fail-fast 断言; 无 % 字面陷阱;
#           反斜杠仅在文件内; 数字一律 result 现场渲染。
import os, sys, json, time, hashlib
import numpy as np

T0 = time.time()
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
MODEL = os.environ.get('P3159_MODEL') or (sys.argv[1] if len(sys.argv) > 1 else 'qwen3-4b')
SMOKE = os.environ.get('P3159_SMOKE') == '1'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
NAME = 'g4p2_equivalence_dynamics'
BASE = os.path.join(RDIR, 'phase3159', NAME, MODEL)
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
        json.dump({'phase': 3159, 'name': phase_name, 'design_sha': sha,
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
SEED = 31580308                      # 同 3158 (rand 臂重放)
ALPHAS = np.array([0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
KL_GATE = 0.1
GATE_STABLE = 3.0                    # 预注册二分门
FP_GATE = 0.8
ARMS = ['null', 'rand', 'top']
L_MID_FRAC = 0.5

design = {
    'phase': '3159', 'name': NAME, 'model': MODEL, 'smoke': bool(SMOKE),
    'prereg': 'AGI_GPT5_MEMO Phase 3159 (frozen at 3158 closeout, before any observation)',
    'anchors': '3157 collect.npz rows=3158 anchor_idx; slot L_mid=round(0.5*NL); '
               'anchor state bitwise vs 3157 fp16 (same model/input/determinism)',
    'hook_semantics': 'forward hook on block L_mid-1 output -> slot L_mid value replaced '
                      'for last token; hidden_states slots < L_mid unchanged; perturbation '
                      'propagates through blocks L_mid..NL-1 (NL-L_mid layers)',
    'dirs': 'reuse 3158 collect.npz top64/bot64 (full 64 cols); pick=linspace(0,63,N_DIR); '
            'rand=QR(default_rng(31580308).standard_normal((D,N_DIR))) replay; unit-norm assert',
    'perturb': "h' = h + alpha*||h_mid||*u at slot L_mid last token (bf16 add)",
    'kl_def': 'jeffreys 0.5*(KL(base||pert)+KL(pert||base)) float64 from forward logits',
    'alphas': ['%.6g' % a for a in ALPHAS], 'alpha_rel': 'alpha * ||h_mid|| (slot L_mid fp32 norm)',
    'kl_gate': KL_GATE, 'gate_stable': GATE_STABLE,
    'ratio_def': 'median(null budgets)/median(rand budgets); censored=grid max (lower bound)',
    're_emergence': 'share_top(layer) = ||P_top64 dh_layer||^2/||dh_layer||^2; dh = pert-base '
                    'last-token states over slots 0..NL; per-arm mean over dirs x alphas; '
                    're_gain_bottom = share_top(NL) - share_top(L_mid) (null arm)',
    'passthrough': 'med_mid(arm)/med_kout_3158(arm) descriptive, no gate',
    'n_anchors': N_ANCH, 'n_dirs_per_arm': N_DIR, 'nvec': NVEC, 'seed': SEED,
    'fp_gate': FP_GATE,
    'summary_gates': 'mid KL curve (log10, null+rand concat) 3 pairs >= 0.8 AND '
                     're-emergence bottom-arm share_top curve (min-NH truncated) 3 pairs >= 0.8',
}
DESIGN_SHA = freeze_design('g4p2_equivalence_dynamics', design)

R7 = os.path.join(RDIR, 'phase3157', 'g2p2_transform_algebra_commutator')
R8 = os.path.join(RDIR, 'phase3158', 'g4p1_output_equivalence_class')

def logsumexp64(M, axis):
    m = M.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(M - m).sum(axis=axis, keepdims=True))).squeeze(axis)

# ================= summary 模式 =================
if MODEL == 'summary':
    log('summary mode: fingerprint verdict over 3 models')
    rs, kl_curves, re_curves, ratios = {}, {}, {}, {}
    stables = []
    for m in ('qwen3-4b', 'qwen3-14b', 'glm4'):
        rp = os.path.join(RDIR, 'phase3159', NAME, m, 'result.json')
        r = json.load(open(rp, encoding='utf-8'))
        rs[m] = r
        stables.append(r['dyn_class'])
        ratios[m] = r['ratio_mid']
        z = np.load(os.path.join(RDIR, 'phase3159', NAME, m, 'collect.npz'))
        kl_curves[m] = z['arm_curves_mid'].astype(np.float64)   # (3, NA)
        re_curves[m] = z['re_curve'].astype(np.float64)         # (3, NH)
        log('%s: verdict=%s ratio_mid=%.3f' % (m, r['verdict'], r['ratio_mid']))
    NA = kl_curves['qwen3-4b'].shape[1]
    NH = min(re_curves[m].shape[1] for m in re_curves)
    assert all(kl_curves[m].shape[1] == NA for m in kl_curves), 'alpha grid mismatch'
    kl_pairs, re_pairs = {}, {}
    models = list(kl_curves)
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = models[i], models[j]
            ca = np.concatenate([np.log10(kl_curves[a][0] + 1e-300), np.log10(kl_curves[a][1] + 1e-300)])
            cb = np.concatenate([np.log10(kl_curves[b][0] + 1e-300), np.log10(kl_curves[b][1] + 1e-300)])
            kl_pairs['%s_vs_%s' % (a, b)] = float(np.corrcoef(ca, cb)[0, 1])
            ra, rb = re_curves[a][0][:NH], re_curves[b][0][:NH]
            re_pairs['%s_vs_%s' % (a, b)] = float(np.corrcoef(ra, rb)[0, 1])
    fpmin_kl = float(min(kl_pairs.values()))
    fpmin_re = float(min(re_pairs.values()))
    st_n = sum(1 for c in stables if c == 'quotient_stable')
    fp_ok = (fpmin_kl >= FP_GATE) and (fpmin_re >= FP_GATE)
    if fp_ok and st_n == 3:
        verdict = 'g4p2_fingerprint_consistent|stable_3/3'
    elif fp_ok:
        verdict = 'g4p2_fingerprint_consistent|stable_%d/3' % st_n
    else:
        verdict = 'g4p2_fingerprint_divergent_material_downgrade'
    result = {
        'phase': 3159, 'name': NAME, 'mode': 'summary', 'smoke': bool(SMOKE),
        'prereg': design['prereg'], 'models': models,
        'dyn_classes': {m: rs[m]['dyn_class'] for m in models},
        'ratios_mid': ratios,
        'kl_pairs': kl_pairs, 're_pairs': re_pairs,
        'fpmin_kl': fpmin_kl, 'fpmin_re': fpmin_re,
        'fp_gate': FP_GATE,
        'gates': {'fingerprint': bool(fp_ok), 'stable_3of3': st_n == 3},
        'verdict': verdict, 'design_sha': DESIGN_SHA,
        'runtime_s': round(time.time() - T0, 1),
    }
    seal_result(result, 'result_summary.json')
    log('SUMMARY DONE')
    sys.exit(0)

# ================= 模型模式 =================
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MDIR_MAP = {'qwen3-4b': 'qwen3-4b', 'qwen3-14b': 'Qwen3-14B', 'glm4': 'glm4-9b-chat-hf'}
MDIR = os.path.join(ROOT, 'models', 'hf', MDIR_MAP[MODEL])
cfgm = json.load(open(os.path.join(MDIR, 'config.json'), encoding='utf-8'))
V, D = int(cfgm['vocab_size']), int(cfgm['hidden_size'])
NL = int(cfgm['num_hidden_layers'])
L_MID = int(round(L_MID_FRAC * NL))
log('model=%s V=%d D=%d NL=%d L_mid=%d smoke=%s' % (MODEL, V, D, NL, L_MID, SMOKE))
assert 0 < L_MID < NL - 2, ('L_mid out of range', L_MID, NL)

# --- 方向: 复用 3158 npz (bottom64/top64 全 64 列) + rand 重放 ---
z8 = np.load(os.path.join(R8, MODEL, 'collect.npz'))
r8 = json.load(open(os.path.join(R8, MODEL, 'result.json'), encoding='utf-8'))
top64 = z8['top64'].astype(np.float64)     # (D, 64) 降序
bot64 = z8['bot64'].astype(np.float64)     # (D, 64) 升序
assert top64.shape == (D, NVEC) and bot64.shape == (D, NVEC)
anchor_idx = z8['anchor_idx'].astype(int)
med_kout = dict(r8['part2']['budget_median'])   # 3158 KOUT 预算(描述对照)
pick = np.linspace(0, NVEC - 1, N_DIR).round().astype(int)
assert len(set(pick)) == N_DIR
dirs = {}
dirs['null'] = bot64[:, pick]
dirs['top'] = top64[:, pick]
rng = np.random.default_rng(SEED)
Rr = rng.standard_normal((D, N_DIR))
Rr, _ = np.linalg.qr(Rr)
dirs['rand'] = Rr
for arm in ARMS:
    nn = np.linalg.norm(dirs[arm], axis=0)
    assert np.abs(nn - 1.0).max() < 1e-5, ('unit norm', arm, nn)
# 正交性装置锚
xov = float(np.abs(dirs['null'].T @ dirs['top']).max())
assert xov < 1e-6, ('bottom/top not orthogonal', xov)
log('dirs reused from 3158 npz; orth_check=%.2e; kout_med=%s' % (
    xov, {k: round(v, 4) for k, v in med_kout.items()}))

# --- 3157 行重建 (零抄写: execution.json design) + npz 对表 ---
exe7 = json.load(open(os.path.join(R7, MODEL, 'execution.json'), encoding='utf-8'))
d7 = exe7['design']
z7 = np.load(os.path.join(R7, MODEL, 'collect.npz'))
H7 = z7['H']
assert H7.shape[0] == int(d7['n_rows']) and H7.shape[1] == NL + 1 and H7.shape[2] == D, (H7.shape,)
TPL7 = {tuple(k.split('|')): v for k, v in d7['tpl'].items()}
ENTS7 = [tuple(e) for e in d7['ents']]
RELS7 = list(d7['rels']); POLS7 = list(d7['pols']); CTXS7 = list(d7['ctx'])
rows7 = []
for ei in range(len(ENTS7)):
    for rel in RELS7:
        for pol in POLS7:
            for cx in CTXS7:
                e, c, p = ENTS7[ei]
                rows7.append(dict(ei=ei, rel=rel, pol=pol, ctx=cx,
                                  prompt=TPL7[(rel, pol)].replace('{E}', e).replace('{C}', c).replace('{P}', p)))
assert len(rows7) == int(d7['n_rows'])
assert np.array_equal(np.array([r['ei'] for r in rows7], np.int16), z7['ei']), 'ei mismatch'
assert np.array_equal(np.array([RELS7.index(r['rel']) for r in rows7], np.int8), z7['rel']), 'rel mismatch'
assert np.array_equal(np.array([POLS7.index(r['pol']) for r in rows7], np.int8), z7['pol']), 'pol mismatch'
assert np.array_equal(np.array([r['ctx'] for r in rows7], np.int8), z7['ctx']), 'ctx mismatch'
idx = np.unique(np.linspace(0, len(rows7) - 1, N_ANCH).round().astype(int))
assert len(idx) == N_ANCH
if not SMOKE:
    assert np.array_equal(idx, anchor_idx), ('anchor_idx mismatch vs 3158', idx.tolist(), anchor_idx.tolist())
log('rows rebuilt + verified vs 3157 npz; anchors=%s' % idx.tolist())

# --- 载模型 ---
torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained(MDIR, trust_remote_code=True)
log('model load begin')
model = AutoModelForCausalLM.from_pretrained(
    MDIR, dtype=torch.bfloat16, trust_remote_code=True).to('cuda').eval()
log('model loaded: %s' % type(model).__name__)
assert model.config.num_hidden_layers == NL
dev = 'cuda'
NH = NL + 1
pre_ids = tok(d7['prefix'], add_special_tokens=False)['input_ids']
ctx_ids = (pre_ids * (int(d7['k_ctx']) // len(pre_ids) + 1))[:int(d7['k_ctx'])]
assert len(ctx_ids) == int(d7['k_ctx'])

# --- 注入 hook (块 L_mid-1 输出 = 槽 L_mid) ---
INJ = {'on': False, 'delta': None}

def _inj_hook(module, args, output):
    if INJ['on'] and INJ['delta'] is not None:
        # 该版本 decoder layer 可能返回裸 Tensor 或 tuple; probe 已证 qwen3-4b 为裸 Tensor
        out0 = output[0] if isinstance(output, tuple) else output
        new0 = out0.clone()
        new0[:, -1, :] = new0[:, -1, :] + INJ['delta'].to(new0.dtype)
        if isinstance(output, tuple):
            return (new0,) + tuple(output[1:])
        return new0
    return None

model.model.layers[L_MID - 1].register_forward_hook(_inj_hook)

def last_token_states(o):
    # batch 输入: 取全部样本 last token -> (B, NH, D)
    return np.stack([h[:, -1].float().detach().cpu().numpy() for h in o.hidden_states], 1)

BATCH_REP = 6   # 基线与注入统一 batch(同 kernel 路径, 消除 batch-size 数值效应)

def forward_base(ids):
    ii = torch.tensor([ids] * BATCH_REP, dtype=torch.int64, device=dev)
    with torch.no_grad():
        o = model(input_ids=ii, output_hidden_states=True)
    lg_all = o.logits[:, -1, :].float().detach().cpu().numpy().astype(np.float64)
    hs_all = last_token_states(o)          # (B, NH, D)
    del o
    batch_rel = float(np.abs(lg_all - lg_all[0][None]).max() /
                      (np.abs(lg_all[0]).max() + 1e-18))
    return lg_all[0], hs_all[0], batch_rel

def forward_base1(ids):
    # batch=1 (与 3157 采集同 kernel 路径) —— 仅用于锚态 bitwise 验证
    ii = torch.tensor([ids], dtype=torch.int64, device=dev)
    with torch.no_grad():
        o = model(input_ids=ii, output_hidden_states=True)
    lg = o.logits[0, -1].float().detach().cpu().numpy().astype(np.float64)
    hs = np.stack([h[0, -1].float().detach().cpu().numpy() for h in o.hidden_states], 0)
    del o
    return lg, hs

# --- 逐锚: 基线 + 注入扫描 ---
KL = np.zeros((3, N_ANCH, N_DIR, len(ALPHAS)), np.float64)
SHARE = np.zeros((3, N_ANCH, N_DIR, len(ALPHAS), NH), np.float64)
anchor_meta = []
pre_rel_max = 0.0
batch_rel_max = 0.0
anchor_rel_max = 0.0
for ai0, r0 in enumerate(idx):
    row = rows7[int(r0)]
    ids = (ctx_ids if row['ctx'] == 1 else []) + tok(row['prompt'], add_special_tokens=False)['input_ids']
    la1, ha1 = forward_base1(ids)
    la2, ha2 = forward_base1(ids)
    assert np.array_equal(la1, la2) and np.array_equal(ha1, ha2), ('base1 bitwise', int(r0))
    assert np.array_equal(ha1[L_MID].astype(np.float16), H7[int(r0), L_MID]), \
        ('anchor bitwise vs 3157 (batch=1 kernel path)', int(r0))
    lg1, hs1, br1 = forward_base(ids)
    lg2, hs2, br2 = forward_base(ids)
    assert np.array_equal(lg1, lg2) and np.array_equal(hs1, hs2), ('base bitwise', int(r0))
    b16 = float(np.abs(hs1[L_MID].astype(np.float64) - ha1[L_MID].astype(np.float64)).max() /
                (np.abs(ha1[L_MID]).max() + 1e-18))
    batch_rel_max = max(batch_rel_max, br1, b16)
    pnorm = float(np.linalg.norm(hs1[L_MID].astype(np.float64)))
    lzb = logsumexp64(lg1[None, :], 1)[0]
    logb = lg1 - lzb
    pb = np.exp(logb)
    eps_a = ALPHAS * pnorm
    for arm_i, arm in enumerate(ARMS):
        for di in range(N_DIR):
            u = dirs[arm][:, di]
            INJ['on'] = True
            INJ['delta'] = torch.from_numpy(
                (eps_a[:, None] * u[None, :]).astype(np.float32)).to(dev)
            ii = torch.tensor([ids] * len(ALPHAS), dtype=torch.int64, device=dev)
            with torch.no_grad():
                o = model(input_ids=ii, output_hidden_states=True)
            lps = o.logits[:, -1, :].float().detach().cpu().numpy().astype(np.float64)
            hss = last_token_states(o)     # (NA, NH, D)
            INJ['on'] = False
            INJ['delta'] = None
            del o
            lzp = logsumexp64(lps, 1)
            logp = lps - lzp[:, None]
            pp = np.exp(logp)
            kl_bp = (pb[None, :] * (logb[None, :] - logp)).sum(1)
            kl_pb = (pp * (logp - logb[None, :])).sum(1)
            KL[arm_i, ai0, di] = 0.5 * (kl_bp + kl_pb)
            dh = hss - hs1[None, :, :]                     # (NA, NH, D)
            pre_rel = float(np.abs(dh[:, :L_MID, :]).max() / (pnorm + 1e-18))
            pre_rel_max = max(pre_rel_max, pre_rel)
            assert pre_rel < 1e-5, ('pre-slot dh too large', arm, di, pre_rel)
            num = ((dh[:, L_MID:, :] @ top64.astype(np.float64)[None]) ** 2).sum(2)
            den = (dh[:, L_MID:, :] ** 2).sum(2) + 1e-18
            SHARE[arm_i, ai0, di, :, L_MID:] = num / den
            del lps, hss, dh, num, den
    # 装置锚: 槽 L_mid 位移方向精确 = u (cos=1), bottom/top 份额
    for arm_i, arm in enumerate(ARMS):
        sh_mid = SHARE[arm_i, ai0, :, :, L_MID].mean()
        if arm == 'null':
            assert sh_mid < 0.05, ('bottom share_top at L_mid too high', sh_mid)
        if arm == 'top':
            assert sh_mid > 0.95, ('top share_top at L_mid too low', sh_mid)
    anchor_meta.append(dict(row=int(r0), ei=int(row['ei']), rel=row['rel'], pol=row['pol'],
                            ctx=int(row['ctx']), pnorm=pnorm,
                            batch_rel=br1, batch6_vs_batch1_rel=b16))
    log('anchor %d/%d row=%d pnorm=%.1f b16rel=%.2e' % (ai0 + 1, N_ANCH, int(r0), pnorm, b16))

# --- 预算提取 (同 3158 log-log 插值) ---
BUD = np.zeros((3, N_ANCH, N_DIR), np.float64)
CEN = np.zeros((3, N_ANCH, N_DIR), np.uint8)
SMALL = np.zeros((3, N_ANCH, N_DIR), np.uint8)
la = np.log(ALPHAS)
for arm_i in range(3):
    for i in range(N_ANCH):
        for di in range(N_DIR):
            k = KL[arm_i, i, di]
            bad = np.where(k >= KL_GATE)[0]
            if len(bad) == 0:
                BUD[arm_i, i, di] = ALPHAS[-1]
                CEN[arm_i, i, di] = 1
            elif bad[0] == 0:
                BUD[arm_i, i, di] = ALPHAS[0]
                SMALL[arm_i, i, di] = 1
            else:
                j = bad[0]
                t = (np.log(KL_GATE) - np.log(k[j - 1])) / (np.log(k[j]) - np.log(k[j - 1]) + 1e-300)
                BUD[arm_i, i, di] = float(np.exp(la[j - 1] + t * (la[j] - la[j - 1])))
med = {arm: float(np.median(BUD[ai])) for ai, arm in enumerate(ARMS)}
cen_frac = {arm: float(CEN[ai].mean()) for ai, arm in enumerate(ARMS)}
ratio_mid = med['null'] / (med['rand'] + 1e-300)
if CEN[0].mean() > 0.5 and med['rand'] < ALPHAS[-1]:
    ratio_mid = max(ratio_mid, ALPHAS[-1] / med['rand'])
dyn_class = 'quotient_stable' if ratio_mid >= GATE_STABLE else 'dynamics_destroyed'
passthru = {arm: float(med[arm] / (med_kout[arm] + 1e-300)) for arm in ARMS}
log('budgets_mid: null=%.4g rand=%.4g top=%.4g ratio=%.3f cens=%s -> %s' % (
    med['null'], med['rand'], med['top'], ratio_mid,
    {a: round(c, 2) for a, c in cen_frac.items()}, dyn_class))
log('passthrough mid/kout: %s' % {k: round(v, 3) for k, v in passthru.items()})

# --- re-emergence 汇总 ---
re_curve = SHARE.mean(axis=(1, 2, 3))            # (3, NH) per-arm mean over anchors/dirs/alphas
re_gain_bottom = float(re_curve[0, NL] - re_curve[0, L_MID])
re_gain_rand = float(re_curve[1, NL] - re_curve[1, L_MID])
re_gain_top = float(re_curve[2, NL] - re_curve[2, L_MID])
log('re_curve bottom L_mid=%.4f NL=%.4f; gains bot=%.4f rand=%.4f top=%.4f' % (
    re_curve[0, L_MID], re_curve[0, NL], re_gain_bottom, re_gain_rand, re_gain_top))

arm_curves_mid = np.zeros((3, len(ALPHAS)), np.float64)
for arm_i in range(3):
    arm_curves_mid[arm_i] = np.median(KL[arm_i].reshape(-1, len(ALPHAS)), axis=0)

# --- npz + result ---
npz_p = os.path.join(BASE, 'collect.npz')
np.savez_compressed(
    npz_p,
    alphas=ALPHAS, anchor_idx=idx.astype(np.int64),
    anchor_meta=json.dumps(anchor_meta, ensure_ascii=False),
    KL=KL.astype(np.float32), SHARE=SHARE.astype(np.float32),
    BUD=BUD, CEN=CEN, SMALL=SMALL,
    arm_curves_mid=arm_curves_mid.astype(np.float32),
    re_curve=re_curve.astype(np.float32),
    l_mid=np.int64(L_MID),
)
npz_sha = sha8(open(npz_p, 'rb').read())
log('npz saved %s sha8=%s' % (os.path.basename(npz_p), npz_sha))

result = {
    'phase': 3159, 'name': NAME, 'mode': MODEL, 'smoke': bool(SMOKE),
    'prereg': design['prereg'], 'hook_semantics': design['hook_semantics'],
    'model': {'V': V, 'D': D, 'NL': NL, 'L_mid': L_MID},
    'anchor_meta': anchor_meta,
    'budget_median': med, 'censored_frac': cen_frac,
    'small_frac': {arm: float(SMALL[ai].mean()) for ai, arm in enumerate(ARMS)},
    'ratio_mid': float(ratio_mid), 'kl_gate': KL_GATE, 'gate_stable': GATE_STABLE,
    'dyn_class': dyn_class,
    'passthrough_mid_over_kout': passthru,
    're_emergence': {'re_gain_bottom': re_gain_bottom, 're_gain_rand': re_gain_rand,
                     're_gain_top': re_gain_top,
                     'share_mid_bottom': float(re_curve[0, L_MID]),
                     'share_mid_top': float(re_curve[2, L_MID]),
                     'share_NL_bottom': float(re_curve[0, NL])},
    'det': {'base1_bitwise': True, 'anchor_bitwise_3157': True,
            'batch6_vs_batch1_rel_max': batch_rel_max, 'pre_slot_rel_max': pre_rel_max,
            'dir_orth': xov,
            'note': 'batch=1 anchor bitwise vs 3157 (same kernel path); injection/base '
                    'unified batch=6 for fair KL; batch effect quantized and recorded'},
    'verdict': 'g4p2_dynamics_%s|ratio_%.3f|regain_bot_%.4f|pass_null_%.3f' % (
        dyn_class.replace('quotient_', '').replace('dynamics_', ''),
        ratio_mid, re_gain_bottom, passthru['null']),
    'npz_sha8': npz_sha, 'design_sha': DESIGN_SHA,
    'runtime_s': round(time.time() - T0, 1),
}
seal_result(result, 'result.json')
log('DONE model=%s smoke=%s' % (MODEL, SMOKE))
