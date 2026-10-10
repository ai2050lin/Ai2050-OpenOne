# -*- coding: utf-8 -*-
# Phase 3156 (G3-P1): 位置平移族基座（位置主线 P1, 用户 2026-10-08 附件二则）
# 预注册: AGI_GPT5_MEMO 2026-10-08（观测前冻结）:
#   (1) 双臂平移: 目标句冻结, 前置中性前缀 k in {0,1,2,4,8,16,32,64,128}
#       A 臂 = 真实前缀(上下文+位置混杂); B 臂 = 同 token 序列但 attention_mask 屏蔽前缀
#       + position_ids 重置(0..n_tgt-1) -> B 与 k=0 的差 = RoPE 纯相对性检验
#   (2) 四图: P x L(逐层位移曲线) / P x D(位移子空间能量谱+公共方向) / P x token(逐 token 敏感度)
#       / P x output(KL + top-1 一致率)
#   (3) IC 补偿指数: IC(k,l) = disp_A(k,l) / (KL(k)+eps); 峰层 = 内部位移大而输出不变的层
#   (4) T 变换探索 v0: 逐层 平移可解释份额 shift_share + 形状保持 resid_shape + 子空间线性 T gain
#   (5) 门: g1_rope(B max rel disp < 1e-3) / g2_ic(峰层 frac in [0.35,0.97] 且 peak_ratio>=5)
#       / g3_out(KL128 < 1.0 且 top1 >= 0.3) / g4_curve(zh-en 位移曲线 Pearson >= 0.8)
# 材料: 目标句 zh='我喜欢吃苹果，因为它又甜又多汁。' en='I like apples because they are sweet and juicy.'
#   (用户附件原句); 前缀=中性句循环截断至恰 k token; 36 序列 = 2 lang x 2 arm x 9 k
# 采集: 目标句段全隐层 H fp16 (NSEQ, NTGT_MAX, NL+1, D) + final logits fp16 (NSEQ, V)
# 本 phase 仅 qwen3-4b; 14b/glm4 跨模型指纹 = 3157
# 教训内置: SMOKE 目录分离; design 全 str 键 JSON; fail-fast 断言; emb 层预期零位移(RoPE 不进 h0)
import os, sys, json, time, hashlib
import numpy as np

T0 = time.time()
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
MODEL = os.environ.get('P3156_MODEL') or (sys.argv[1] if len(sys.argv) > 1 else 'qwen3-4b')
SMOKE = os.environ.get('P3156_SMOKE') == '1'
assert MODEL in ('qwen3-4b',), MODEL
ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
NAME = 'g3p1_position_shift_family'
BASE = os.path.join(RDIR, 'phase3156', NAME, MODEL)
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
        json.dump({'phase': 3156, 'name': phase_name, 'design_sha': sha,
                   'design': design, 'frozen_before': 'any model observation',
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

# ---------------- 冻结材料 ----------------
TARGETS = {'zh': '我喜欢吃苹果，因为它又甜又多汁。',
           'en': 'I like apples because they are sweet and juicy.'}
PREFIX = {'zh': '今天天气很好，我们在讨论语言模型如何表示语言。',
          'en': 'The weather is fine today and we keep discussing how models represent language. '}
K_FULL = [0, 1, 2, 4, 8, 16, 32, 64, 128]
K_SMOKE = [0, 1, 2]
LANGS = ['zh', 'en']
ARMS = ['A', 'B']
ROPE_TOL = 1e-3
IC_FRAC_LO, IC_FRAC_HI, IC_PEAK_RATIO = 0.35, 0.97, 5.0
OUT_KL_GATE, OUT_TOP_GATE = 1.0, 0.3
CURVE_GATE = 0.8
T_RANK = 32

KG = K_SMOKE if SMOKE else K_FULL
LGS = ['zh'] if SMOKE else LANGS
NSEQ = len(LGS) * len(ARMS) * len(KG)
MAT_NOTES = []
for lg in LGS:
    MAT_NOTES.append('%s target chars=%d' % (lg, len(TARGETS[lg])))
if not SMOKE:
    MAT_NOTES.append('k=0 rows duplicated across arms for d0 bitwise anchor')
    MAT_NOTES.append('B arm = same token ids, prefix masked, position_ids reset to 0..n_tgt-1')

# ---------------- 模式: GPU 采集 + 分析 ----------------
MDIR = os.path.join(ROOT, 'models', 'hf', 'qwen3-4b')
cfg = json.load(open(os.path.join(MDIR, 'config.json'), encoding='utf-8'))
NL = cfg['num_hidden_layers']
HID = cfg['hidden_size']
KOUT = NL - 1
MID = NL // 2
design = dict(model=MODEL, mdir=MDIR, phase=NAME, nl=NL, hidden=HID, readout=KOUT,
              k_grid=KG, langs=LGS, arms=ARMS,
              targets={k: v for k, v in TARGETS.items() if k in LGS},
              prefix={k: v for k, v in PREFIX.items() if k in LGS},
              rope_tol=ROPE_TOL, ic_frac=[IC_FRAC_LO, IC_FRAC_HI], ic_peak_ratio=IC_PEAK_RATIO,
              out_kl_gate=OUT_KL_GATE, out_top_gate=OUT_TOP_GATE, curve_gate=CURVE_GATE,
              t_rank=T_RANK, n_seq=NSEQ, material_asserts=MAT_NOTES, smoke=SMOKE,
              pre_reg='MEMO 2026-10-08 (G3-P1 position shift family base, from user attachment); frozen before observation')
exe_sha = freeze_design('g3p1_%s' % MODEL, design)
log('model=%s NL=%d D=%d readout=%d seqs=%d' % (MODEL, NL, HID, KOUT, NSEQ))

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
NH = NL + 1

# tokenize + 构造序列表
TGT_IDS = {lg: tok(TARGETS[lg], add_special_tokens=False)['input_ids'] for lg in LGS}
PRE_IDS = {lg: tok(PREFIX[lg], add_special_tokens=False)['input_ids'] for lg in LGS}
NTGT = {lg: len(TGT_IDS[lg]) for lg in LGS}
NTGT_MAX = max(NTGT.values())
for lg in LGS:
    assert NTGT[lg] >= 8, ('target too short', lg, NTGT[lg])
    assert len(PRE_IDS[lg]) * 16 >= max(KG), ('prefix too short even cycled', lg, len(PRE_IDS[lg]))
log('tgt tokens %s; prefix lens %s' % (NTGT, {lg: len(PRE_IDS[lg]) for lg in LGS}))

SEQS = []   # (lang, arm, k)
for lg in LGS:
    for arm in ARMS:
        for k in KG:
            SEQS.append((lg, arm, k))
assert len(SEQS) == NSEQ

def make_inputs(lg, arm, k):
    tids = TGT_IDS[lg]
    n = NTGT[lg]
    if k == 0:
        ids = list(tids)
        mask = [1] * n
        pos = list(range(n))
    else:
        pre = (PRE_IDS[lg] * (k // len(PRE_IDS[lg]) + 1))[:k]
        ids = pre + list(tids)
        ntot = len(ids)
        assert ntot == k + n
        if arm == 'A':
            mask = [1] * ntot
            pos = list(range(ntot))
        else:
            mask = [0] * k + [1] * n
            pos = [0] * k + list(range(n))
    return ids, mask, pos

# k=0 时 A/B 输入恒等断言
i0a, m0a, p0a = make_inputs(LGS[0], 'A', 0)
i0b, m0b, p0b = make_inputs(LGS[0], 'B', 0)
assert i0a == i0b and m0a == m0b and p0a == p0b, 'k=0 arm inputs must be identical'
log('k=0 arm identity assert OK')

cache = os.path.join(BASE, 'collect_smoke.npz' if SMOKE else 'collect.npz')
det_note = 'cache_hit (determinism not rechecked this run)'
if os.path.exists(cache):
    z = np.load(cache)
    H16 = z['H']
    LG16 = z['LG']
    log('collect cache hit %s H=%s' % (os.path.basename(cache), H16.shape))
else:
    H16 = np.zeros((NSEQ, NTGT_MAX, NH, D), np.float16)
    LG16 = np.zeros((NSEQ, cfg['vocab_size']), np.float16)
    eff = np.zeros((NSEQ, NTGT_MAX), np.int8)
    with torch.no_grad():
        for i, (lg, arm, k) in enumerate(SEQS):
            ids, mask, pos = make_inputs(lg, arm, k)
            n = NTGT[lg]
            ii = torch.tensor([ids], dtype=torch.int64, device='cuda')
            mm = torch.tensor([mask], dtype=torch.int64, device='cuda')
            pp = torch.tensor([pos], dtype=torch.int64, device='cuda')
            o = model(input_ids=ii, attention_mask=mm, position_ids=pp,
                      output_hidden_states=True)
            hs = o.hidden_states
            seg = torch.stack([h[0, k:k + n, :] for h in hs], 0)   # (NH, n, D)
            H16[i, :n] = seg.permute(1, 0, 2).float().detach().cpu().numpy().astype(np.float16)
            eff[i, :n] = 1
            LG16[i] = o.logits[0, k + n - 1, :].float().detach().cpu().numpy().astype(np.float16)
            del o, hs, seg
            if (i + 1) % 8 == 0:
                log('collect %d/%d' % (i + 1, NSEQ))
    # 确定性锚: 重采 3 行(含 k=0 的 A/B 对)
    det_rows = [0, NSEQ // 2, NSEQ - 1]
    det_ok = True
    det_max = 0.0
    with torch.no_grad():
        for i in det_rows:
            lg, arm, k = SEQS[i]
            ids, mask, pos = make_inputs(lg, arm, k)
            n = NTGT[lg]
            ii = torch.tensor([ids], dtype=torch.int64, device='cuda')
            mm = torch.tensor([mask], dtype=torch.int64, device='cuda')
            pp = torch.tensor([pos], dtype=torch.int64, device='cuda')
            o = model(input_ids=ii, attention_mask=mm, position_ids=pp,
                      output_hidden_states=True)
            seg = torch.stack([h[0, k:k + n, :] for h in o.hidden_states], 0)
            hv = seg.permute(1, 0, 2).float().detach().cpu().numpy().astype(np.float16)
            dmax = float(np.abs(hv.astype(np.float32) - H16[i, :n].astype(np.float32)).max())
            det_max = max(det_max, dmax)
            if dmax != 0.0:
                det_ok = False
            del o
    det_note = 'bitwise' if det_ok else 'fp16 max abs diff %.3e (tolerance pass)' % det_max
    assert det_max < 1e-3, ('determinism check fail', det_max)
    log('determinism recheck rows=%s -> %s' % (det_rows, det_note))
    np.savez_compressed(cache, H=H16, LG=LG16, eff=eff,
                        lang=np.array([s[0] for s in SEQS]),
                        arm=np.array([s[1] for s in SEQS]),
                        k=np.array([s[2] for s in SEQS], np.int32),
                        n_tgt=np.array([NTGT[s[0]] for s in SEQS], np.int32))
    log('collect saved %s' % os.path.basename(cache))
npz_sha = hashlib.sha256(open(cache, 'rb').read()).hexdigest()[:8]
del model
torch.cuda.empty_cache()
log('model released; npz sha8=%s' % npz_sha)

# ---- 分析（CPU） ----
z = np.load(cache)
H16 = z['H'].astype(np.float32)
LG16 = z['LG'].astype(np.float32)
SEQ_LANG = [s.decode() if isinstance(s, bytes) else str(s) for s in z['lang']]
SEQ_ARM = [s.decode() if isinstance(s, bytes) else str(s) for s in z['arm']]
SEQ_K = [int(v) for v in z['k']]
NTGTV = [int(v) for v in z['n_tgt']]
IDX = {(SEQ_LANG[i], SEQ_ARM[i], SEQ_K[i]): i for i in range(NSEQ)}

def rel_disp(Ha, Hb, n):
    num = float(np.linalg.norm((Ha[:n] - Hb[:n]).ravel()))
    den = float(np.linalg.norm(Ha[:n].ravel())) + 1e-18
    return num / den

# d0 锚: A0 vs B0 位级
d0_max = 0.0
for lg in LGS:
    ia, ib = IDX[(lg, 'A', 0)], IDX[(lg, 'B', 0)]
    n = NTGTV[ia]
    d0_max = max(d0_max, float(np.abs(H16[ia, :n] - H16[ib, :n]).max()))
log('d0 anchor (A0 vs B0) max abs diff = %.3e' % d0_max)

# g1 RoPE 纯相对性: B(k>0) vs A0
rope = {}
rope_max = 0.0
for lg in LGS:
    ia0 = IDX[(lg, 'A', 0)]
    n = NTGTV[ia0]
    for k in KG:
        if k == 0:
            continue
        ib = IDX[(lg, 'B', k)]
        r = rel_disp(H16[ia0], H16[ib], n)
        rope['%s_k%d' % (lg, k)] = r
        rope_max = max(rope_max, r)
log('rope check max rel disp (B vs A0) = %.3e' % rope_max)

# A 臂位移曲线 disp_A(k, l) per lang (P x L 图)
curve_A = {}
for lg in LGS:
    ia0 = IDX[(lg, 'A', 0)]
    n = NTGTV[ia0]
    for k in KG:
        ia = IDX[(lg, 'A', k)]
        per_l = [float(np.linalg.norm(H16[ia, t, l] - H16[ia0, t, l]) /
                       (np.linalg.norm(H16[ia0, t, l]) + 1e-18))
                 for t in range(n) for l in range(NH)]
        # 逐层聚合(Frobenius 比值)
        cl = [float(np.linalg.norm((H16[ia, :n, l] - H16[ia0, :n, l]).ravel()) /
                    (np.linalg.norm(H16[ia0, :n, l].ravel()) + 1e-18)) for l in range(NH)]
        curve_A['%s_k%d' % (lg, k)] = cl
# emb 层物理锚: A 臂 emb 层跨 k 恒定(RoPE 不进 h0)
emb_zero = max(max(c[0] for kk, c in curve_A.items() if kk.startswith(lg + '_')) for lg in LGS)
log('emb-layer max disp across k = %.3e (expected ~0)' % emb_zero)

# 两臂差(同 k 上下文贡献)
ctx_eff = {}
for lg in LGS:
    n = NTGTV[IDX[(lg, 'A', 0)]]
    for k in KG:
        if k == 0:
            continue
        ia, ib = IDX[(lg, 'A', k)], IDX[(lg, 'B', k)]
        cl = [float(np.linalg.norm((H16[ia, :n, l] - H16[ib, :n, l]).ravel()) /
                    (np.linalg.norm(H16[ia, :n, l].ravel()) + 1e-18)) for l in range(NH)]
        ctx_eff['%s_k%d' % (lg, k)] = cl

# KL 与 top-1 一致率 (P x output 图)
def logsoftmax(v):
    v = v - v.max()
    return v - np.log(np.exp(v).sum())

KL = {}
TOP1 = {}
for lg in LGS:
    ia0 = IDX[(lg, 'A', 0)]
    p0 = np.exp(logsoftmax(LG16[ia0]))
    for k in KG:
        ia = IDX[(lg, 'A', k)]
        pk = np.exp(logsoftmax(LG16[ia]))
        kl = float((p0 * (np.log(p0 + 1e-30) - np.log(pk + 1e-30))).sum())
        KL['%s_k%d' % (lg, k)] = kl
        TOP1['%s_k%d' % (lg, k)] = float(int(np.argmax(LG16[ia])) == int(np.argmax(LG16[ia0])))
kl128 = float(np.mean([KL['%s_k%d' % (lg, 128)] for lg in LGS if (lg, 'A', 128) in IDX]))
top128 = float(np.mean([TOP1['%s_k%d' % (lg, 128)] for lg in LGS if (lg, 'A', 128) in IDX]))
log('KL(128)=%.4f top1(128)=%.2f' % (kl128, top128))

# IC 补偿指数: IC(k,l) = disp_A(k,l)/(KL(k)+eps), 主曲线 k=最大
KMAX = max(KG)
ic_curve = {}
for lg in LGS:
    klk = KL['%s_k%d' % (lg, KMAX)]
    ic_curve[lg] = [curve_A['%s_k%d' % (lg, KMAX)][l] / (klk + 1e-6) for l in range(NH)]
ic_mean = np.mean([ic_curve[lg] for lg in LGS], 0)
ic_l1 = int(np.argmax(ic_mean[1:]) + 1)
ic_frac = ic_l1 / float(NL)
ic_peak_ratio = float(ic_mean[ic_l1] / (np.median(ic_mean[1:]) + 1e-18))
log('IC peak layer=%d frac=%.2f peak_ratio=%.1f' % (ic_l1, ic_frac, ic_peak_ratio))

# P x D: 位移子空间(k=KMAX, 三代表层 emb/mid/readout)
def subspace_stats(lg, layer):
    ia0, ia = IDX[(lg, 'A', 0)], IDX[(lg, 'A', KMAX)]
    n = NTGTV[ia0]
    dlt = H16[ia, :n, layer] - H16[ia0, :n, layer]     # (n, D)
    nb = float(np.linalg.norm(dlt)) + 1e-18
    if nb < 1e-12:
        return dict(top1=0.0, top2=0.0, top4=0.0, top8=0.0, cos_mean=0.0)
    U, s, _ = np.linalg.svd(dlt, full_matrices=False)
    e = s ** 2
    tot = e.sum() + 1e-18
    dbar = dlt.mean(0)
    dbar_n = np.linalg.norm(dbar) + 1e-18
    cos_mean = float(np.mean([float(dlt[t] @ dbar) / (np.linalg.norm(dlt[t]) * dbar_n + 1e-18)
                              for t in range(n) if np.linalg.norm(dlt[t]) > 1e-12]))
    return dict(top1=float(e[0] / tot), top2=float(e[:2].sum() / tot),
                top4=float(e[:4].sum() / tot), top8=float(e[:8].sum() / tot) if len(s) >= 8 else 1.0,
                cos_mean=cos_mean)
subspace = {}
for lg in LGS:
    for tag, layer in (('emb', 0), ('mid', MID), ('readout', KOUT)):
        subspace['%s_%s' % (lg, tag)] = subspace_stats(lg, layer)

# P x token: readout 层逐 token 敏感度(k=KMAX)
per_token = {}
for lg in LGS:
    ia0, ia = IDX[(lg, 'A', 0)], IDX[(lg, 'A', KMAX)]
    n = NTGTV[ia0]
    per_token[lg] = [float(np.linalg.norm(H16[ia, t, KOUT] - H16[ia0, t, KOUT]) /
                           (np.linalg.norm(H16[ia0, t, KOUT]) + 1e-18)) for t in range(n)]

# T 变换探索 v0: k=0 -> KMAX, pooled langs, 全层
# shift_share = 平移可解释份额; resid_shape = 中心化后恒等残差; gain = 线性 T 相对恒等的改进
def t_stats(layer):
    X0, X1 = [], []
    for lg in LGS:
        ia0, ia = IDX[(lg, 'A', 0)], IDX[(lg, 'A', KMAX)]
        n = NTGTV[ia0]
        X0.append(H16[ia0, :n, layer])
        X1.append(H16[ia, :n, layer])
    X0 = np.concatenate(X0, 0).astype(np.float64)
    X1 = np.concatenate(X1, 0).astype(np.float64)
    dlt = X1 - X0
    den = float((dlt ** 2).sum()) + 1e-18
    if den < 1e-24:
        return 0.0, 0.0, 0.0
    shift_share = 1.0 - float(((dlt - dlt.mean(0, keepdims=True)) ** 2).sum()) / den
    mu0 = X0.mean(0)
    X0c, X1c = X0 - mu0, X1 - mu0
    U, s, Vt = np.linalg.svd(X0c, full_matrices=False)
    r = min(T_RANK, Vt.shape[0])
    V = Vt[:r].T
    Zi, Zj = X0c @ V, X1c @ V
    d_z = float(np.linalg.norm(Zj - Zi)) + 1e-18
    resid_shape = d_z / (float(np.linalg.norm(Zi)) + 1e-18)
    G = Zi.T @ Zi + 1e-3 * float(np.trace(Zi.T @ Zi) + 1e-18) / r * np.eye(r)
    Tm = np.linalg.solve(G, Zi.T @ Zj)
    resid_T = float(np.linalg.norm(Zj - Zi @ Tm)) / (float(np.linalg.norm(Zi)) + 1e-18)
    gain = (resid_shape - resid_T) / (resid_shape + 1e-18)
    return shift_share, resid_shape, float(gain)

t_curve = {}
for l in range(NH):
    ss, rs, gn = t_stats(l)
    t_curve[str(l)] = dict(shift_share=ss, resid_shape=rs, lin_gain=gn)
ss_read = t_curve[str(KOUT)]['shift_share']
ss_max = max(t_curve[str(l)]['shift_share'] for l in range(1, NH))
log('T v0: shift_share readout=%.3f max=%.3f' % (ss_read, ss_max))

# g4 跨句曲线一致(层 1..NL)
def _p(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a - a.mean(), b - b.mean()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-18))
curve_consist = _p(curve_A['zh_k%d' % KMAX][1:], curve_A['en_k%d' % KMAX][1:]) \
    if ('en_k%d' % KMAX) in curve_A else float('nan')

# 门
g1 = bool(rope_max < ROPE_TOL)
g2 = bool(IC_FRAC_LO <= ic_frac <= IC_FRAC_HI and ic_peak_ratio >= IC_PEAK_RATIO)
g3 = bool(kl128 < OUT_KL_GATE and top128 >= OUT_TOP_GATE) if KMAX == 128 else bool(kl128 < OUT_KL_GATE)
g4 = bool(curve_consist >= CURVE_GATE) if KMAX == 128 else True
log('gates g1=%s g2=%s g3=%s g4=%s curve=%.3f' % (g1, g2, g3, g4, curve_consist))

verdict = 'g3p1_rope_%s|ic_peak_L%d_frac_%.2f_ratio_%.1f|out_kl%.3f_top%.2f|curve_%.3f|shift_%.2f' % (
    'supported' if g1 else 'violated', ic_l1, ic_frac, ic_peak_ratio, kl128, top128,
    curve_consist, ss_max)
if SMOKE:
    verdict = 'SMOKE_' + verdict

result = dict(phase=3156, name=NAME, model=MODEL, smoke=SMOKE,
              design_sha=exe_sha, nl=NL, hidden=D, readout=KOUT, kmax=KMAX,
              n_seq=NSEQ, n_tgt=NTGT, runtime_s=round(time.time() - T0, 1),
              determinism_note=det_note, npz_sha8=npz_sha,
              anchors=dict(d0_max_abs=d0_max, emb_layer_max_disp=emb_zero,
                           rope_max_rel=rope_max),
              rope_check={kk: float(v) for kk, v in rope.items()},
              curve_A={kk: [float(x) for x in v] for kk, v in curve_A.items()},
              ctx_effect={kk: [float(x) for x in v] for kk, v in ctx_eff.items()},
              kl={kk: float(v) for kk, v in KL.items()},
              top1={kk: float(v) for kk, v in TOP1.items()},
              ic_curve={lg: [float(x) for x in ic_curve[lg]] for lg in LGS},
              ic_peak=dict(layer=ic_l1, frac=ic_frac, peak_ratio=ic_peak_ratio),
              subspace=subspace, per_token=per_token,
              t_transform=t_curve,
              curve_consist=curve_consist,
              gates=dict(g1_rope=g1, g2_ic=g2, g3_out=g3, g4_curve=g4),
              verdict=verdict)
res_sha8, seal = seal_result(result, 'result.json')
log('DONE runtime=%.1fs' % (time.time() - T0))
